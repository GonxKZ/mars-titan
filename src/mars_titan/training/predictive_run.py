"""Ajuste offline del adaptador predictivo, con presupuesto y recuperación comparables."""

import copy
import fcntl
import math
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from mars_titan.budget_training import seed_run
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.embeddings import require_cuda
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.models.predictive_adaptation import LinearResidualPolicy, objective

from .checkpoints import (
    StopRequest,
    capture_rng,
    load_training_state,
    restore_rng,
    save_training_state,
)
from .predictive_evaluation import evaluate_predictive
from .predictive_inputs import PredictiveDataset, fit_standardizer


def _code():
    root = Path(__file__).parents[1]
    return {
        name: sha256(root / name)
        for name in (
            "training/predictive_run.py",
            "training/predictive_evaluation.py",
            "training/predictive_inputs.py",
            "training/predictive_parents.py",
            "training/checkpoints.py",
            "models/predictive_adaptation.py",
            "training/corpus_inputs.py",
            "environments/corpus_source.py",
            "environments/cohorts.py",
            "environments/actions.py",
            "data/cohort_files.py",
            "data/storage.py",
            "evaluation/session_metrics.py",
            "budget_training.py",
            "data/embeddings.py",
        )
    }


def _options(case, batch_size, checkpoint_steps, checkpoint_seconds):
    if (
        not isinstance(case, dict)
        or set(case) != {"mode", "epochs", "seed", "learning_rate", "weight_decay", "clip_norm"}
        or case["mode"] not in {"reinforce", "expected", "mae"}
        or type(case["epochs"]) is not int
        or not 1 <= case["epochs"] <= 30
        or type(case["seed"]) is not int
        or not 0 <= case["seed"] < 2**32
        or type(batch_size) is not int
        or not 1 <= batch_size <= 4096
        or type(checkpoint_steps) is not int
        or checkpoint_steps < 0
        or not math.isfinite(checkpoint_seconds)
        or not 0 < checkpoint_seconds <= 900
    ):
        raise ValueError("La configuración de adaptación no es válida")
    for name, allow_zero in (
        ("learning_rate", False),
        ("weight_decay", True),
        ("clip_norm", False),
    ):
        value = case[name]
        if (
            type(value) not in (float, int)
            or not math.isfinite(value)
            or value < 0
            or (not allow_zero and value == 0)
        ):
            raise ValueError("Los parámetros del optimizador no son válidos")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError("Configura CUBLAS_WORKSPACE_CONFIG antes de iniciar PyTorch")


def _statistics():
    return dict(samples=0, absolute_error=0.0, squared_error=0.0, objective_sum=0.0)


def run_predictive_case(
    ordered,
    parent_cache,
    output,
    case,
    *,
    batch_size=256,
    checkpoint_steps=0,
    checkpoint_seconds=60,
    resume=False,
    stop=None,
    normalization=None,
):
    """Usar todos los ejemplos de train, con selección al terminar épocas completas."""
    _options(case, batch_size, checkpoint_steps, checkpoint_seconds)
    output, ordered, parent_cache = Path(output), Path(ordered), Path(parent_cache)
    safe_destination(output)
    for protected in (ordered.parent, parent_cache.parent):
        outside_source(protected, output)
        outside_source(output, protected)
    if output.exists() and not resume or resume and not (output / "initialization.json").is_file():
        raise ValueError("Usa una salida nueva o una ejecución recuperable")
    if type(resume) is not bool:
        raise ValueError("La recuperación debe solicitarse de forma explícita")
    output.mkdir(parents=True, exist_ok=resume)
    descriptor = os.open(output / ".run.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialization = dict(
            ordered_sha256=read_manifest(ordered)[1],
            parent_cache_sha256=read_manifest(parent_cache)[1],
            case=case,
            batch_size=batch_size,
            code=_code(),
        )
        if resume and read_manifest(output / "initialization.json")[0] != initialization:
            raise ValueError("La preparación pertenece a otra identidad de datos o código")
        if not resume:
            atomic_json(output / "initialization.json", initialization)
        return _setup_run(
            ordered,
            parent_cache,
            output,
            case,
            batch_size,
            checkpoint_steps,
            checkpoint_seconds,
            resume and (output / "run.json").is_file(),
            stop,
            normalization,
        )
    finally:
        os.close(descriptor)


def _setup_run(
    ordered,
    parent_cache,
    output,
    case,
    batch_size,
    checkpoint_steps,
    checkpoint_seconds,
    resume,
    stop,
    normalization,
):
    device = require_cuda()
    seed_run(case["seed"])
    stop = stop or StopRequest()
    with PredictiveDataset(ordered, parent_cache) as dataset:
        if resume:
            statistics, statistics_hash = read_manifest(output / "normalization.json", 4 * 1024**2)
        else:
            statistics = (
                read_manifest(Path(normalization), 4 * 1024**2)[0]
                if normalization is not None
                else fit_standardizer(dataset, batch_size=batch_size)
            )
            if (
                statistics.get("ordered_manifest_sha256") != dataset.ordered_hash
                or statistics.get("parent_cache_sha256") != dataset.cache_hash
                or statistics.get("fit_partition") != "train"
                or statistics.get("samples") != dataset.counts["train"]
                or statistics.get("code_sha256")
                != sha256(Path(__file__).with_name("predictive_inputs.py"))
            ):
                raise ValueError("La normalización no corresponde al entrenamiento de esta edición")
            atomic_json(output / "normalization.json", statistics)
            statistics_hash = sha256(output / "normalization.json")
        identity = dict(
            ordered_manifest_sha256=dataset.ordered_hash,
            parent_cache_sha256=dataset.cache_hash,
            manifest_sha256=dataset.metadata["source_sha256"],
            case=copy.deepcopy(case),
            batch_size=batch_size,
            normalization_sha256=statistics_hash,
            grid=dataset.metadata["grid"],
            features=dataset.features,
            weighting=dataset.cache_metadata["identity"]["weighting"],
            market_weights=dataset.weights,
            parent_checkpoint_sha256=dataset.cache_metadata["checkpoint_sha256"],
            fit_timing="offline_after_training_cutoff",
            fit_cutoff_utc="2023-01-01T00:00:00Z",
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            numpy=np.__version__,
            python=platform.python_version(),
            gpu=torch.cuda.get_device_name(0),
            numerics=dict(
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            ),
            code=_code(),
            optimizer="AdamW",
            precision="parameters_float32_objective_float64_parent_float64",
        )
        report = (
            read_manifest(output / "run.json")[0]
            if resume
            else dict(
                schema_version=1,
                status="running",
                identity=identity,
                model="linear_residual_adapter",
                parent_frozen=True,
                device="cuda:0",
                samples=dataset.counts,
                scope=dataset.metadata["scope"],
                cohort_complete=dataset.metadata["cohort_complete"],
                final_test_opened=False,
                epochs=[],
                global_step=0,
            )
        )
        if report["identity"] != identity:
            raise ValueError("La ejecución pertenece a otros datos, padre, normalización o código")
        if report["status"] == "completed":
            load_training_state(
                output / "checkpoints",
                expected_identity=identity,
                selection="best",
                expected_sha256=report["checkpoint"]["sha256"],
            )
            for item in report["predictions"].values():
                if sha256(output / item["path"]) != item["sha256"]:
                    raise ValueError("Una predicción confirmada ha cambiado")
            return report
        model = LinearResidualPolicy(
            statistics["mean"], statistics["scale"], target_scale=identity["grid"]["scale"]
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=case["learning_rate"], weight_decay=case["weight_decay"]
        )
        generator = torch.Generator(device=device).manual_seed(case["seed"])
        state = dict(
            global_step=0,
            epoch=0,
            cursor=None,
            statistics=_statistics(),
            epochs=[],
            best_score=None,
            best_epoch=None,
        )
        confirmed = output / "checkpoints/latest.json"
        if resume and not (confirmed.exists() or confirmed.is_symlink()):
            if report["global_step"] != 0 or report["epochs"] or "recovery_checkpoint" in report:
                raise ValueError("Falta el índice de un punto de control previamente confirmado")
            # El primer guardado falló antes de confirmar el estado inicial.
            # Ninguna actualización puede preceder a ese guardado.
        elif resume:
            state = load_training_state(output / "checkpoints", expected_identity=identity)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"], "cuda:0")
            generator.set_state(state["sampling_rng"])
        report["trainable_parameters"] = sum(p.numel() for p in model.parameters())
        return _train(
            dataset,
            output,
            case,
            batch_size,
            checkpoint_steps,
            checkpoint_seconds,
            stop,
            model,
            optimizer,
            generator,
            state,
            report,
        )


def _train(
    dataset,
    output,
    case,
    batch_size,
    checkpoint_steps,
    checkpoint_seconds,
    stop,
    model,
    optimizer,
    generator,
    state,
    report,
):
    started, last_saved = time.perf_counter(), time.monotonic()
    identity = report["identity"]
    values = torch.tensor(identity["grid"]["values"], dtype=torch.float64, device="cuda:0")
    scale = identity["grid"]["scale"]
    torch.cuda.reset_peak_memory_stats(0)

    def save(best=False):
        nonlocal last_saved
        if _code() != identity["code"]:
            raise ValueError("El código de adaptación ha cambiado durante el entrenamiento")
        state.update(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=capture_rng("cuda:0"),
            sampling_rng=generator.get_state(),
        )
        path = save_training_state(output / "checkpoints", state, identity=identity, best=best)
        report.update(
            global_step=state["global_step"],
            epochs=copy.deepcopy(state["epochs"]),
            recovery_checkpoint=dict(path=str(path.relative_to(output)), sha256=sha256(path)),
            best_epoch=state["best_epoch"],
            best_score=state["best_score"],
        )
        atomic_json(output / "run.json", report)
        last_saved = time.monotonic()

    report["status"] = "running"
    try:
        save()
        while state["epoch"] < case["epochs"]:
            model.train()
            for batch in dataset.batches(
                partition="train",
                batch_size=batch_size,
                epoch=state["epoch"],
                seed=case["seed"],
                cursor=state["cursor"],
            ):
                if stop.requested:
                    save()
                    raise InterruptedError("Entrenamiento pausado antes del siguiente lote")
                features = torch.as_tensor(batch["features"], device="cuda:0")
                parent = torch.as_tensor(batch["parent"], dtype=torch.float64, device="cuda:0")
                targets = torch.as_tensor(batch["target"], dtype=torch.float64, device="cuda:0")
                weights = torch.as_tensor(batch["weight"], dtype=torch.float64, device="cuda:0")
                optimizer.zero_grad(set_to_none=True)
                centers = model(features, parent)
                losses, _ = objective(
                    centers, targets, values, scale, case["mode"], generator=generator
                )
                (losses * weights).mean().backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), case["clip_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                error = centers.detach().cpu().numpy() - batch["target"]
                stats = state["statistics"]
                stats["samples"] += len(error)
                stats["absolute_error"] += float(np.abs(error).sum())
                stats["squared_error"] += float(np.square(error).sum())
                stats["objective_sum"] += float((losses.detach() * weights).sum().cpu())
                state["global_step"] += 1
                state["cursor"] = batch["confirmed_cursor"]
                if (
                    checkpoint_steps
                    and state["global_step"] % checkpoint_steps == 0
                    or time.monotonic() - last_saved >= checkpoint_seconds
                    or stop.requested
                ):
                    save()
            stats = state["statistics"]
            if stats["samples"] != dataset.counts["train"]:
                raise ValueError("La época no conserva todas las filas de entrenamiento")
            save()
            validation = evaluate_predictive(
                model, dataset, batch_size, partition="validation", stop=stop
            )
            state["epochs"].append(
                dict(
                    epoch=state["epoch"] + 1,
                    train=dict(
                        stats,
                        mae=stats["absolute_error"] / stats["samples"],
                        mse=stats["squared_error"] / stats["samples"],
                    ),
                    validation=validation,
                )
            )
            state["epoch"] += 1
            state["cursor"], state["statistics"] = None, _statistics()
            score = validation["median"]["session_mae"]
            better = state["best_score"] is None or score < state["best_score"]
            if better:
                state["best_score"], state["best_epoch"] = score, state["epoch"]
            save(best=better)
        # La evaluación seleccionada no se guarda con el optimizador de la última época.
        selected = load_training_state(
            output / "checkpoints", expected_identity=identity, selection="best"
        )
        model.load_state_dict(selected["model"])
        index = read_manifest(output / "checkpoints/latest.json")[0]["best"]
        report["checkpoint"] = dict(path=f"checkpoints/{index['name']}", sha256=index["sha256"])
        predictions = {}
        for partition in ("train", "validation"):
            path = output / f"{partition}-predictions.parquet"
            metrics = evaluate_predictive(
                model, dataset, batch_size, partition=partition, destination=path, stop=stop
            )
            predictions[partition] = dict(path=path.name, sha256=sha256(path), metrics=metrics)
        if _code() != identity["code"]:
            raise ValueError("El código ha cambiado durante la evaluación")
        report.update(status="completed", predictions=predictions)
    except InterruptedError:
        report["status"] = "paused"
    except BaseException as error:
        report.update(
            status="failed", last_failure=dict(type=type(error).__name__, message=str(error))
        )
        raise
    finally:
        report.update(
            attempt_seconds=time.perf_counter() - started,
            process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(0),
        )
        atomic_json(output / "run.json", report)
    return report
