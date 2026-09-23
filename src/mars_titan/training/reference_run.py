"""Épocas completas de referencias multimodales con cursor y estado recuperables."""

import json
import math
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from mars_titan.budget_training import seed_run, validate_loss
from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.embeddings import require_cuda
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.models.baselines.multimodal import MultimodalReference, validate_architecture
from mars_titan.profiling import CostProbe

from .checkpoints import (
    StopRequest,
    capture_rng,
    load_training_state,
    restore_rng,
    save_training_state,
)
from .corpus_inputs import CorpusDataset


class _Pause(Exception):
    """Solicitud de parada entre operaciones, sin declarar un fallo científico."""


def read_json(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024**2:
        raise ValueError("Falta un recibo regular dentro del presupuesto")
    return json.loads(path.read_text())


def scientific_identity():
    """Compartir versiones, política numérica y transformaciones entre todos los casos."""
    root = Path(__file__).parents[1]
    sources = (
        "training/reference_run.py",
        "training/reference_campaign.py",
        "training/checkpoints.py",
        "training/corpus_inputs.py",
        "profiling.py",
        "models/baselines/dlinear.py",
        "models/baselines/multimodal.py",
        "training/cohort_contract.py",
        "models/baselines/campaign.py",
        "budget_training.py",
        "data/streaming.py",
        "data/batches.py",
        "data/storage.py",
        "data/embeddings.py",
    )
    return dict(
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        numpy=np.__version__,
        pyarrow=pa.__version__,
        python=platform.python_version(),
        gpu=torch.cuda.get_device_name(0),
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        numerics={
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        code={name: sha256(root / name) for name in sources},
    )


def _options(case, batch_size, checkpoint_seconds, checkpoint_steps):
    required = {"kind", "loss", "learning_rate", "seed", "epochs", "huber_delta"}
    if (
        set(case) not in (required, required | {"architecture"})
        or case["kind"] not in {"rnn", "lstm", "gru", "dlinear"}
        or type(case["epochs"]) is not int
        or not 1 <= case["epochs"] <= 1000
        or type(case["seed"]) is not int
        or not 0 <= case["seed"] < 2**32
        or type(batch_size) is not int
        or not 1 <= batch_size <= 4096
        or type(checkpoint_steps) is not int
        or checkpoint_steps < 0
        or not math.isfinite(checkpoint_seconds)
        or checkpoint_seconds <= 0
        or not math.isfinite(case["learning_rate"])
        or case["learning_rate"] <= 0
    ):
        raise ValueError("La configuración del entrenamiento no es válida")
    if "architecture" in case:
        architecture = case["architecture"]
        if not isinstance(architecture, dict) or set(architecture) != {
            "hidden_size",
            "layers",
            "dropout",
        }:
            raise ValueError("La arquitectura necesita anchura, profundidad y regularización")
        validate_architecture(**architecture)
    validate_loss(case["loss"], case["huber_delta"])
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError("Configura CUBLAS_WORKSPACE_CONFIG antes de iniciar PyTorch")


def _inputs(batch, device):
    return {k: torch.from_numpy(v).to(device) for k, v in batch["inputs"].items()}


def _statistics():
    return dict(samples=0, squared_error=0.0, absolute_error=0.0, elapsed_seconds=0.0)


def _metrics(statistics):
    count = statistics["samples"]
    if not count:
        raise ValueError("No se puede evaluar una partición vacía")
    return {
        **statistics,
        "mse": statistics["squared_error"] / count,
        "mae": statistics["absolute_error"] / count,
        "samples_per_second": count / max(statistics["elapsed_seconds"], 1e-9),
    }


def _update(statistics, prediction, target):
    error = prediction.detach().to(torch.float64) - target.to(torch.float64)
    sums = torch.stack([error.square().sum(), error.abs().sum()]).tolist()
    if not all(math.isfinite(v) for v in sums):
        raise ValueError("El error contiene valores no finitos")
    statistics["samples"] += len(target)
    statistics["squared_error"] += sums[0]
    statistics["absolute_error"] += sums[1]


def _evaluate(model, dataset, batch_size, *, partition="validation", destination=None, stop=None):
    model.eval()
    statistics = _statistics()
    start = time.perf_counter()

    def tables():
        with torch.inference_mode():
            for batch in dataset.batches(
                partition=partition, batch_size=batch_size, epoch=0, seed=0
            ):
                if stop is not None and stop.requested:
                    raise _Pause
                target = torch.from_numpy(batch["target"]).to("cuda:0")
                predicted = model(_inputs(batch, "cuda:0"))
                if predicted.shape != target.shape:
                    raise ValueError("Predicción y etiqueta no tienen la misma forma")
                _update(statistics, predicted, target)
                yield pa.table(
                    {
                        "sample_id": batch["sample_ids"],
                        "asset_id": ["/".join(key.split("/")[:2]) for key in batch["sample_ids"]],
                        "market": batch["market"],
                        "prediction_at": pa.array(
                            batch["prediction_at"], type=pa.timestamp("us", tz="UTC")
                        ),
                        "target": batch["target"],
                        "prediction": predicted.cpu().numpy(),
                        "zero": np.zeros(len(target), dtype=np.float64),
                    }
                )

    if destination is None:
        for _ in tables():
            pass
    else:
        atomic_parquet_batches(destination, tables())
    torch.cuda.synchronize(0)
    statistics["elapsed_seconds"] = time.perf_counter() - start
    if statistics["samples"] != dataset.manifest["counts"][partition]:
        raise ValueError("La evaluación no reconcilia toda la población")
    return _metrics(statistics)


def training_weights(assets, weighting):
    """Ponderar por mercado con recuentos de ajuste, sin retirar filas de ninguno."""
    if weighting not in {"natural", "balanced_markets"}:
        raise ValueError("La ponderación debe ser natural o equilibrada entre mercados")
    counts = {}
    for asset in assets:
        market = asset["market"]
        counts[market] = counts.get(market, 0) + asset["counts"]["train"]
    if not counts or min(counts.values()) < 1:
        raise ValueError("Cada mercado necesita una población de ajuste no vacía")
    total = sum(counts.values())
    return {
        m: total / (len(counts) * n) if weighting == "balanced_markets" else 1.0
        for m, n in counts.items()
    }


def _parent(parent, dataset, case, model, batch_size, weighting):
    if parent is None:
        return None
    report = read_json(parent / "run.json")
    identity = report["identity"]
    if (
        report.get("status") != "completed"
        or identity["manifest_sha256"] != dataset.identity
        or identity["case"]["kind"] != case["kind"]
        or identity["case"]["seed"] != case["seed"]
        or identity["case"].get("architecture") != case.get("architecture")
        or identity["batch_size"] != batch_size
        or identity["weighting"] != weighting
    ):
        raise ValueError("El origen no corresponde a la población, arquitectura y semilla")
    checkpoint = report["checkpoint"]
    if sha256(parent / checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("El punto de control final de origen ha cambiado")
    if any(identity.get(k) != v for k, v in scientific_identity().items()):
        raise ValueError("El entorno o el código no coincide con el origen de la continuación")
    index = read_json(parent / "checkpoints/latest.json")
    if index["latest"][0]["sha256"] != checkpoint["sha256"]:
        raise ValueError("El punto de control de origen ha cambiado")
    state = load_training_state(parent / "checkpoints", expected_identity=identity)
    model.load_state_dict(state["model"])
    return dict(parent_checkpoint_sha256=checkpoint["sha256"], optimizer_policy="new_adamw")


def run_reference_case(
    manifest: Path,
    output: Path,
    case: dict,
    *,
    batch_size: int = 16,
    resume: bool = False,
    checkpoint_seconds: float = 900,
    checkpoint_steps: int = 0,
    stop: StopRequest | None = None,
    initialize_from: Path | None = None,
    weighting: str = "natural",
) -> dict:
    """Ajustar una referencia sobre toda la edición, sin abrir el test final."""
    _options(case, batch_size, checkpoint_seconds, checkpoint_steps)
    start = time.perf_counter()
    device = require_cuda()
    dataset = CorpusDataset(manifest)
    if min(dataset.manifest["counts"].values()) < 1:
        raise ValueError("Se necesitan datos de ajuste y validación en este brazo")
    for protected in (*dataset.roots.values(), Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)
    if output.is_symlink() or (output.exists() and not resume) or (resume and not output.is_dir()):
        raise ValueError("Usa una ejecución nueva o solicita continuar una existente")
    seed_run(case["seed"])
    first = next(
        dataset.batches(partition="train", batch_size=batch_size, epoch=0, seed=case["seed"])
    )
    dimensions = {name: value.shape[-1] for name, value in first["inputs"].items()}
    model = (
        MultimodalReference(
            case["kind"], dimensions, context=dataset.context, **case["architecture"]
        )
        if "architecture" in case
        else CostProbe(case["kind"], dimensions, context=dataset.context)
    ).to(device)
    weights = training_weights(dataset.assets, weighting)
    initialization = _parent(initialize_from, dataset, case, model, batch_size, weighting)
    identity = dict(
        **scientific_identity(),
        manifest_sha256=dataset.identity,
        case=case,
        model_family="scientific_multimodal_reference"
        if "architecture" in case
        else "legacy_cost_probe",
        batch_size=batch_size,
        dimensions=dimensions,
        context=dataset.context,
        initialization=initialization,
        weighting=weighting,
        market_weights=weights,
        optimizer="AdamW",
        precision="float32",
        device="cuda:0",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=case["learning_rate"])
    report_path = output / "run.json"
    if resume:
        report = read_json(report_path)
        if report["identity"] != identity:
            raise ValueError("La identidad o configuración de la ejecución ha cambiado")
        if report["status"] == "completed":
            if sha256(output / report["checkpoint"]["path"]) != report["checkpoint"]["sha256"]:
                raise ValueError("El punto de control final ha cambiado")
            for item in report["predictions"].values():
                if sha256(output / item["path"]) != item["sha256"]:
                    raise ValueError("Han cambiado las predicciones confirmadas")
            load_training_state(output / "checkpoints", expected_identity=identity)
            return report
    else:
        output.mkdir(parents=True, exist_ok=False)
        report = dict(
            schema_version=1,
            status="running",
            identity=identity,
            scope=dataset.manifest["scope"],
            cohort_complete=dataset.manifest["cohort_complete"],
            samples=dataset.manifest["counts"],
            epochs=[],
            global_step=0,
            parameters=sum(p.numel() for p in model.parameters()),
            device="cuda:0",
            final_test_opened=False,
            initialization=initialization,
            started_at_utc=datetime.now(UTC).isoformat(),
            attempts=[],
        )
        atomic_json(report_path, report)
    epoch, cursor, step, statistics, history = 0, None, 0, _statistics(), []
    if resume and (output / "checkpoints/latest.json").exists():
        state = load_training_state(output / "checkpoints", expected_identity=identity)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], "cuda:0")
        epoch, cursor, step = state["epoch"], state["confirmed_cursor"], state["global_step"]
        statistics, history = state["statistics"], state["history"]
    stop = stop or StopRequest()
    last_saved = time.perf_counter()

    def save(*, pin=False):
        nonlocal last_saved
        if any(
            sha256(Path(__file__).parents[1] / name) != expected
            for name, expected in identity["code"].items()
        ):
            raise ValueError("El código ha cambiado durante la ejecución")
        state = dict(
            global_step=step,
            epoch=epoch,
            confirmed_cursor=cursor,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=capture_rng("cuda:0"),
            statistics=statistics,
            history=history,
        )
        path = save_training_state(output / "checkpoints", state, identity=identity, pin=pin)
        report["checkpoint"] = dict(path=str(path.relative_to(output)), sha256=sha256(path))
        last_saved = time.perf_counter()

    torch.cuda.reset_peak_memory_stats(0)
    save()
    try:
        while epoch < case["epochs"]:
            if stop.requested:
                raise _Pause
            model.train()
            iterator = dataset.batches(
                partition="train",
                batch_size=batch_size,
                epoch=epoch,
                seed=case["seed"],
                cursor=cursor,
            )
            segment = time.perf_counter()
            for batch in iterator:
                optimizer.zero_grad(set_to_none=True)
                target = torch.from_numpy(batch["target"]).to(device, dtype=torch.float32)
                prediction = model(_inputs(batch, device))
                if prediction.shape != target.shape:
                    raise ValueError("Predicción y etiqueta no tienen la misma forma")
                if case["loss"] == "huber":
                    loss = torch.nn.functional.huber_loss(
                        prediction, target, delta=case["huber_delta"], reduction="none"
                    )
                elif case["loss"] == "mae":
                    loss = torch.nn.functional.l1_loss(prediction, target, reduction="none")
                else:
                    loss = torch.nn.functional.mse_loss(prediction, target, reduction="none")
                if weighting != "natural":
                    loss = loss * torch.tensor([weights[m] for m in batch["market"]], device=device)
                loss = loss.mean()
                if not torch.isfinite(loss).item():
                    raise ValueError("La pérdida no es finita")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), math.inf, error_if_nonfinite=True
                )
                optimizer.step()
                _update(statistics, prediction, torch.from_numpy(batch["target"]).to(device))
                cursor, step = batch["confirmed_cursor"], step + 1
                if (
                    stop.requested
                    or time.perf_counter() - last_saved >= checkpoint_seconds
                    or (checkpoint_steps and step % checkpoint_steps == 0)
                ):
                    torch.cuda.synchronize(0)
                    statistics["elapsed_seconds"] += time.perf_counter() - segment
                    save()
                    segment = time.perf_counter()
                if stop.requested:
                    report.update(status="paused", global_step=step, epochs=history)
                    atomic_json(report_path, report)
                    return report
            torch.cuda.synchronize(0)
            statistics["elapsed_seconds"] += time.perf_counter() - segment
            if statistics["samples"] != dataset.manifest["counts"]["train"]:
                raise ValueError("La época no recorrió exactamente toda la población")
            validation = _evaluate(model, dataset, batch_size, stop=stop)
            history.append(dict(epoch=epoch + 1, train=_metrics(statistics), validation=validation))
            epoch, cursor, statistics = epoch + 1, None, _statistics()
            save(pin=epoch == case["epochs"])
            report.update(global_step=step, epochs=history, status="running")
            atomic_json(report_path, report)
        predictions = {}
        for partition in ("train", "validation"):
            path = output / f"{partition}-predictions.parquet"
            metrics = _evaluate(
                model, dataset, batch_size, partition=partition, destination=path, stop=stop
            )
            predictions[partition] = dict(path=path.name, sha256=sha256(path), metrics=metrics)
        report.update(
            status="completed",
            predictions=predictions,
            global_step=step,
            finished_at_utc=datetime.now(UTC).isoformat(),
        )
        return report
    except _Pause:
        save()
        report.update(status="paused", global_step=step, epochs=history)
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["attempts"].append(
            dict(
                seconds=time.perf_counter() - start,
                peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(0),
                peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(0),
            )
        )
        atomic_json(report_path, report)
