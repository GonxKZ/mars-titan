"""Postentrenamiento offline con identidad completa y cursor confirmado tras cada paso."""

import fcntl
import json
import math
import os
import platform
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.models.klpo import MODES as KLPO
from mars_titan.models.klpo import behavior_log_probabilities, token_loss
from mars_titan.models.predictive_adaptation import (
    LinearResidualPolicy,
    gaussian_log_probabilities,
    objective,
)
from mars_titan.training.checkpoints import (
    StopRequest,
    capture_rng,
    load_training_state,
    restore_rng,
    save_training_state,
)
from mars_titan.training.run_receipts import initialize_receipt

from .evaluation import centers, evaluate
from .inputs import CONDITIONS
from .parents import require_device
from .selection import select_epoch, selection_policy

MODES = ("reinforce", "expected", "mae", *KLPO, "neural_mae", "neural_mse")


def code_identity():
    root = Path(__file__).parents[1]
    names = (
        "posttraining/run.py",
        "posttraining/inputs.py",
        "posttraining/parents.py",
        "posttraining/evaluation.py",
        "posttraining/selection.py",
        "training/selection.py",
        "training/checkpoints.py",
        "training/run_receipts.py",
        "episodes/parents.py",
        "episodes/augmentation.py",
        "episodes/windows.py",
        "episodes/worlds.py",
        "episodes/encoding.py",
        "episodes/storage.py",
        "models/predictive_adaptation.py",
        "models/klpo.py",
        "environments/actions.py",
        "environments/cohorts.py",
        "environments/corpus_source.py",
        "evaluation/session_metrics.py",
        "models/baselines/multimodal.py",
        "models/baselines/dlinear.py",
        "models/baselines/ridge.py",
        "models/baselines/external_boosting.py",
        "models/baselines/inputs.py",
        "training/predictive_parents.py",
        "training/corpus_inputs.py",
        "training/experiment_resources.py",
        "data/embeddings.py",
        "data/cohort_files.py",
        "data/storage.py",
        "profiling.py",
    )
    return {name: sha256(root / name) for name in names}


def validate_case(case):
    required = {
        "mode",
        "condition",
        "seed",
        "epochs",
        "learning_rate",
        "weight_decay",
        "clip_norm",
        "beta",
        "behavior_epsilon",
        "auxiliary_samples",
    }
    if (
        not isinstance(case, dict)
        or not required <= set(case) <= required | {"selection"}
        or case["mode"] not in MODES
        or case["condition"] not in CONDITIONS
        or type(case["seed"]) is not int
        or not 0 <= case["seed"] < 2**32
        or type(case["epochs"]) is not int
        or not 1 <= case["epochs"] <= 30
        or type(case["auxiliary_samples"]) is not int
        or not 1 <= case["auxiliary_samples"] <= 4096
    ):
        raise ValueError("El caso no pertenece al diseño emparejado")
    for key in ("learning_rate", "weight_decay", "clip_norm", "beta", "behavior_epsilon"):
        v = case[key]
        if (
            type(v) not in (int, float)
            or not math.isfinite(v)
            or v < 0
            or (key != "weight_decay" and v == 0)
        ):
            raise ValueError("El caso requiere parámetros finitos y positivos")
    if case["behavior_epsilon"] >= 1:
        raise ValueError("La mezcla exploratoria debe ser inferior a uno")
    selection_policy(case)


def _statistics():
    return dict(samples=0, real_rows=0, extra_rows=0, absolute_error=0.0, squared_error=0.0)


def _loss(prediction, batch, grid, case, generators, device):
    values = torch.tensor(grid.values, dtype=torch.float64, device=device)
    target = torch.as_tensor(batch["target"], dtype=torch.float64, device=device)
    mode = case["mode"]
    if mode.startswith("neural_"):
        error = prediction - target
        loss = error.square() if mode == "neural_mse" else error.abs()
    elif mode in KLPO:
        logp = gaussian_log_probabilities(prediction, values, grid.scale)
        parent = torch.as_tensor(batch["parent"], dtype=torch.float64, device=device)
        logq = behavior_log_probabilities(parent, values, grid.scale, case["behavior_epsilon"])
        rewards = -(values[None, :] - target[:, None]).abs() / grid.scale
        loss, _ = token_loss(
            logp,
            logq,
            rewards,
            case["beta"],
            mode,
            generator=generators["actions"],
            auxiliary_generator=generators["auxiliaries"],
            auxiliary_samples=case["auxiliary_samples"],
        )
    else:
        loss, _ = objective(
            prediction, target, values, grid.scale, mode, generator=generators["actions"]
        )
    result = loss.mean()
    if not torch.isfinite(result):
        raise ValueError("La pérdida del postentrenamiento no es finita")
    return result


def _validate_run(
    dataset,
    case,
    grid,
    normalization,
    *,
    parent,
    batch_size,
    device,
    diagnostic,
    lease,
    checkpoint_seconds,
    max_updates,
):
    require_device(device, diagnostic, lease)
    validate_case(case)
    if (
        type(checkpoint_seconds) not in (int, float)
        or not math.isfinite(checkpoint_seconds)
        or not 0 < checkpoint_seconds <= 900
        or (
            max_updates is not None
            and (not diagnostic or type(max_updates) is not int or max_updates < 1)
        )
        or (diagnostic and max(dataset.counts.values()) > 5000)
    ):
        raise ValueError("El presupuesto de diagnóstico o checkpoint no es válido")
    if not diagnostic and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError("Configura CUBLAS_WORKSPACE_CONFIG antes de iniciar PyTorch")
    if (
        grid.source_sha256 != dataset.train.source_sha256
        or grid.training_samples != dataset.counts["train"]
    ):
        raise ValueError("La rejilla no se ha ajustado con el entrenamiento real")
    expected_norm = dict(
        train_sha256=dataset.train.manifest_sha256,
        parent_sha256=dataset.parent.parent_sha256,
        encoding=dataset.parent.encoding,
        samples=dataset.counts["train"],
        fit_partition="train",
    )
    if any(normalization.get(k) != v for k, v in expected_norm.items()):
        raise ValueError("La normalización no corresponde a las filas reales de entrenamiento")
    if (
        len(normalization.get("mean", [])) != dataset.features
        or len(normalization.get("scale", [])) != dataset.features
    ):
        raise ValueError("Las dimensiones de normalización no coinciden con el adaptador")
    neural = case["mode"].startswith("neural_")
    if (neural or not diagnostic) and (
        parent is None or parent.identity["checkpoint_sha256"] != dataset.parent.parent_sha256
    ):
        raise ValueError("La continuación necesita los mismos pesos verificados del padre")
    if not diagnostic and (
        parent.identity.get("source_sha256") != dataset.train.source_sha256
        or parent.identity.get("counts") != dataset.counts
    ):
        raise ValueError("El padre no acredita la misma población completa")
    budget = dataset.budget(case["condition"], batch_size)
    if diagnostic and budget["rows"] > 5000:
        raise ValueError("El diagnóstico supera el presupuesto de filas aumentadas")
    return neural, budget


def _identity(dataset, parent, case, grid, normalization, budget, batch_size, device, diagnostic):
    identity = dict(
        dataset=dataset.identity,
        case=case,
        parent=parent.identity if parent is not None else None,
        grid=grid.to_dict(),
        normalization=normalization,
        batch_size=batch_size,
        budget=budget,
        device=device,
        diagnostic=diagnostic,
        code=code_identity(),
        numpy=np.__version__,
        torch=str(torch.__version__),
        python=platform.python_version(),
        optimizer="new_adamw",
        selection_policy=selection_policy(case),
        fit_timing="offline_after_training_cutoff",
        cuda=torch.version.cuda if device == "cuda:0" else None,
        gpu=torch.cuda.get_device_name(0) if device == "cuda:0" else None,
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        numerics=dict(
            matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            torch_threads=torch.get_num_threads(),
            deterministic_algorithms=True,
            cudnn_version=torch.backends.cudnn.version() if device == "cuda:0" else None,
        ),
    )
    # Los recibos y checkpoints usan el mismo árbol de tipos JSON.
    return json.loads(json.dumps(identity, allow_nan=False))


def _best_state(output, identity, selection, expected_sha256=None):
    selected = load_training_state(
        output / "checkpoints",
        expected_identity=identity,
        selection="best",
        expected_sha256=expected_sha256,
    )
    if (
        selection is None
        or selected["epoch"] != selection["best_epoch"]
        or selected["best_score"] != selection["best_score"]
        or selected["cursor"] is not None
        or selected["statistics"]["samples"] != 0
    ):
        raise ValueError("El checkpoint no corresponde a la época seleccionada")
    return selected


def run_case(
    dataset,
    output,
    case,
    grid,
    normalization,
    *,
    parent=None,
    batch_size=256,
    device="cuda:0",
    diagnostic=False,
    lease=None,
    resume=False,
    stop=None,
    checkpoint_seconds=60,
    max_updates=None,
):
    """Comparar un objetivo y condición. CPU solo admite diagnósticos de hasta 5000 filas."""
    started = time.perf_counter()
    neural, budget = _validate_run(
        dataset,
        case,
        grid,
        normalization,
        parent=parent,
        batch_size=batch_size,
        device=device,
        diagnostic=diagnostic,
        lease=lease,
        checkpoint_seconds=checkpoint_seconds,
        max_updates=max_updates,
    )
    identity = _identity(
        dataset, parent, case, grid, normalization, budget, batch_size, device, diagnostic
    )
    policy = identity["selection_policy"]
    output = Path(output)
    safe_destination(output)
    if (output.exists() and not resume) or (resume and not output.is_dir()):
        raise ValueError("Usa una salida nueva o solicita una recuperación existente")
    output.mkdir(parents=True, exist_ok=resume)
    descriptor = os.open(output / ".run.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        confirmed = initialize_receipt(output, identity, record="run.json", lock=".run.lock")
        report = (
            read_manifest(output / "run.json", 8 * 1024**2)[0]
            if confirmed
            else dict(
                schema_version=1,
                kind="paired_posttraining",
                identity=identity,
                status="running",
                activity="supervised_continuation" if neural else "predictive_adaptation",
                model=parent.kind if parent is not None else "adaptation",
                domain="technical" if diagnostic else "real",
                condition=case["condition"],
                mode=case["mode"],
                final_test_opened=False,
                global_step=0,
                total_steps=budget["updates"] * case["epochs"],
                epochs=[],
                selection_policy=policy,
                checkpoint_retention=dict(recent=2, best=1, pinned=0),
                samples=dataset.counts,
                budget=budget,
                attempts=[],
            )
        )
        if report["identity"] != identity:
            raise ValueError("El recibo pertenece a otra identidad")
        if report.get("recovery_checkpoint") and not (output / "checkpoints/latest.json").exists():
            raise ValueError("Falta el índice del checkpoint previamente confirmado")
        if report["status"] == "completed":
            _best_state(
                output,
                identity,
                report["selection"],
                expected_sha256=report["checkpoint"]["sha256"],
            )
            from mars_titan.training.predictive_parents import _verified_file

            _verified_file(output, report["predictions"]["validation"])
            return report
        report["status"] = "running"
        atomic_json(output / "run.json", report)
        random.seed(case["seed"])
        np.random.seed(case["seed"])
        torch.manual_seed(case["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        model = (
            parent.continuation()
            if neural
            else LinearResidualPolicy(
                normalization["mean"], normalization["scale"], target_scale=grid.scale
            )
        )
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=case["learning_rate"], weight_decay=case["weight_decay"]
        )
        generators = {
            name: torch.Generator(device=device).manual_seed(case["seed"] + i + 1)
            for i, name in enumerate(("actions", "auxiliaries"))
        }
        state = dict(
            global_step=0,
            epoch=0,
            cursor=None,
            statistics=_statistics(),
            history=[],
            best_score=None,
            best_epoch=None,
            selection=None,
            baseline=None,
        )
        if (output / "checkpoints/latest.json").exists():
            state = load_training_state(output / "checkpoints", expected_identity=identity)
            if state["selection"] is not None:
                _best_state(output, identity, state["selection"])
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"], device)
            for key, generator in generators.items():
                generator.set_state(state["generators"][key])
        stop = stop or StopRequest()
        last_saved, updates = time.perf_counter(), 0
        evaluating_selected = False

        def stopped_early():
            return bool(
                policy["patience"] is not None
                and state["selection"]
                and state["selection"]["should_stop"]
            )

        def save(best=False):
            nonlocal last_saved
            if code_identity() != identity["code"]:
                raise ValueError("El código ha cambiado durante el postentrenamiento")
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("El optimizador ha producido pesos no finitos")
            state.update(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=capture_rng(device),
                generators={key: g.get_state() for key, g in generators.items()},
            )
            path = save_training_state(output / "checkpoints", state, identity=identity, best=best)
            report.update(
                global_step=state["global_step"],
                epochs=state["history"],
                selection=state["selection"],
                baseline=state["baseline"],
                best_epoch=state["best_epoch"],
                best_score=state["best_score"],
                recovery_checkpoint=dict(path=str(path.relative_to(output)), sha256=sha256(path)),
            )
            last_saved = time.perf_counter()

        if device == "cuda:0":
            torch.cuda.reset_peak_memory_stats(0)
        try:
            save()
            if policy["version"] == 2 and state["baseline"] is None:
                baseline = evaluate(
                    model,
                    dataset,
                    grid,
                    batch_size=batch_size,
                    neural=neural,
                    device=device,
                    stop=stop,
                )
                state["selection"] = select_epoch(
                    None, baseline["session_mae"], 0, policy, case["epochs"]
                )
                state["baseline"] = baseline
                state["best_score"], state["best_epoch"] = baseline["session_mae"], 0
                save(best=True)
                atomic_json(output / "run.json", report)
            while state["epoch"] < case["epochs"] and not stopped_early():
                if stop.requested:
                    raise InterruptedError
                model.train()
                for batch in dataset.batches(
                    partition="train",
                    condition=case["condition"],
                    batch_size=batch_size,
                    epoch=state["epoch"],
                    seed=case["seed"],
                    cursor=state["cursor"],
                ):
                    if lease is not None:
                        lease.check()
                    optimizer.zero_grad(set_to_none=True)
                    prediction = centers(model, batch, neural=neural, device=device)
                    loss = _loss(prediction, batch, grid, case, generators, device)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), case["clip_norm"], error_if_nonfinite=True
                    )
                    optimizer.step()
                    error = prediction.detach().cpu().numpy() - batch["target"]
                    stats = state["statistics"]
                    stats["samples"] += len(error)
                    stats["real_rows" if batch["origin"] == "real" else "extra_rows"] += len(error)
                    stats["absolute_error"] += float(np.abs(error).sum())
                    stats["squared_error"] += float(np.square(error).sum())
                    state["cursor"], state["global_step"] = (
                        batch["confirmed_cursor"],
                        state["global_step"] + 1,
                    )
                    updates += 1
                    if stop.requested or (max_updates is not None and updates >= max_updates):
                        raise InterruptedError
                    if time.perf_counter() - last_saved >= checkpoint_seconds:
                        save()
                        atomic_json(output / "run.json", report)
                stats = state["statistics"]
                if (stats["samples"], stats["real_rows"], stats["extra_rows"]) != (
                    budget["rows"],
                    budget["real_rows"],
                    budget["extra_rows"],
                ):
                    raise ValueError("La época no conserva el presupuesto de filas emparejado")
                validation = evaluate(
                    model,
                    dataset,
                    grid,
                    batch_size=batch_size,
                    neural=neural,
                    device=device,
                    stop=stop,
                )
                state["history"].append(
                    dict(epoch=state["epoch"] + 1, train=dict(stats), validation=validation)
                )
                state["epoch"] += 1
                state["cursor"], state["statistics"] = None, _statistics()
                score = validation["session_mae"]
                state["selection"] = select_epoch(
                    state["selection"], score, state["epoch"], policy, case["epochs"]
                )
                state["best_score"] = state["selection"]["best_score"]
                state["best_epoch"] = state["selection"]["best_epoch"]
                save(best=state["selection"]["last_improved"])
            if state["global_step"] != budget["updates"] * state["epoch"] or (
                not stopped_early() and state["global_step"] != report["total_steps"]
            ):
                raise ValueError("El número de actualizaciones no coincide con el diseño")
            report["stopped_early"] = state["epoch"] < case["epochs"]
            selected = _best_state(output, identity, state["selection"])
            model.load_state_dict(selected["model"])
            evaluating_selected = True
            record = read_manifest(output / "checkpoints/latest.json")[0]["best"]
            path = output / "validation-predictions.parquet"
            metrics = evaluate(
                model,
                dataset,
                grid,
                batch_size=batch_size,
                neural=neural,
                device=device,
                stop=stop,
                destination=path,
            )
            if code_identity() != identity["code"]:
                raise ValueError("El código ha cambiado durante la evaluación final")
            report.update(
                status="completed",
                checkpoint=dict(path=f"checkpoints/{record['name']}", sha256=record["sha256"]),
                best_epoch=state["best_epoch"],
                predictions=dict(
                    validation=dict(path=path.name, sha256=sha256(path), metrics=metrics)
                ),
            )
        except InterruptedError:
            # La evaluación seleccionada no debe mezclarse con el optimizador de la última época.
            if not evaluating_selected:
                save()
            report["status"] = "paused"
        except BaseException as error:
            report.update(
                status="failed", last_failure=dict(type=type(error).__name__, message=str(error))
            )
            raise
        finally:
            if device == "cuda:0":
                torch.cuda.synchronize(0)
            report["attempts"].append(
                dict(
                    total_seconds=time.perf_counter() - started,
                    updates=updates,
                    process_lifetime_peak_rss_bytes=resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    * 1024,
                    peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(0)
                    if device == "cuda:0"
                    else None,
                )
            )
            atomic_json(output / "run.json", report)
        return report
    finally:
        os.close(descriptor)
