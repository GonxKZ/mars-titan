"""Referencia XGBoost CUDA recuperable, con lecturas acotadas del corpus supervisado."""

import argparse
import fcntl
import os
import re
import resource
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.models.baselines.external_boosting import (
    ExternalBoostingModel,
    _libraries,
    fit_external_boosting,
)
from mars_titan.models.baselines.inputs import MODALITIES

from .checkpoints import StopRequest
from .corpus_inputs import CorpusDataset
from .tabular_corpus import _matrix, _predict


class _Paused(Exception):
    """Parada confirmada tras guardar una ronda completa."""


def _identity(dataset, options, cp, xgb):
    root = Path(__file__).parents[1]
    names = (
        "training/external_corpus.py",
        "training/tabular_corpus.py",
        "training/corpus_inputs.py",
        "training/cohort_contract.py",
        "models/baselines/external_boosting.py",
        "models/baselines/inputs.py",
        "evaluation/session_metrics.py",
        "data/streaming.py",
        "data/batches.py",
        "data/storage.py",
        "data/cohort_files.py",
        "data/embeddings.py",
    )
    return dict(
        manifest_sha256=dataset.identity,
        cohort=dataset.cohort,
        options=options,
        xgboost=xgb.__version__,
        cupy=cp.__version__,
        numpy=np.__version__,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        cuda_driver=cp.cuda.runtime.driverGetVersion(),
        code={name: sha256(root / name) for name in names},
    )


def _load(output, checkpoint, rows):
    if not isinstance(checkpoint, dict) or not re.fullmatch(
        r"checkpoints/attempt-\d{4}-round-\d{4}\.ubj", checkpoint.get("path", "")
    ):
        raise ValueError("El recibo no contiene una ruta de modelo confirmada")
    path = output / checkpoint["path"]
    safe_destination(path)
    return ExternalBoostingModel.load(path, checkpoint["sha256"], training_rows=rows)


def run_external_reference(
    manifest,
    output,
    *,
    rounds=100,
    max_depth=4,
    max_bin=128,
    learning_rate=0.05,
    seed=42,
    batch_size=256,
    max_batch_bytes=64 * 1024**2,
    max_host_cache_bytes=16 * 1024**3,
    on_host=True,
    checkpoint_interval=10,
    resume=False,
    stop=None,
):
    """Recorrer todas las filas admitidas sin abrir el test ni reducir la población."""
    if type(batch_size) is not int or not 1 <= batch_size <= 4096 or type(resume) is not bool:
        raise ValueError("El lote o el modo de recuperación no son válidos")
    dataset, output = CorpusDataset(Path(manifest)), Path(output)
    if min(dataset.manifest["counts"].values()) < 1:
        raise ValueError("Se necesitan entrenamiento y validación no vacíos")
    for protected in (*dataset.roots.values(), Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)
    safe_destination(output)
    if output.exists() and not resume:
        raise ValueError("La referencia necesita una salida nueva o recuperación explícita")
    if resume and not (output / "run.json").is_file():
        raise ValueError("No existe un informe recuperable")
    cp, xgb = _libraries()
    options = dict(
        rounds=rounds,
        max_depth=max_depth,
        max_bin=max_bin,
        learning_rate=learning_rate,
        seed=seed,
        batch_size=batch_size,
        max_batch_bytes=max_batch_bytes,
        max_host_cache_bytes=max_host_cache_bytes,
        on_host=on_host,
        checkpoint_interval=checkpoint_interval,
    )
    identity = _identity(dataset, options, cp, xgb)
    output.mkdir(parents=True, exist_ok=resume)
    lock = os.open(output / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = (
            read_manifest(output / "run.json", 8 * 1024**2)[0]
            if resume
            else dict(
                schema_version=1,
                model="xgboost_external_cuda",
                device="cuda:0",
                identity=identity,
                samples=dataset.manifest["counts"],
                scope=dataset.manifest["scope"],
                cohort_complete=dataset.manifest["cohort_complete"],
                final_test_opened=False,
                feature_order=list(MODALITIES),
                fitted_rows=0,
                completed_rounds=0,
                status="pending",
                checkpoint=None,
                attempts=[],
            )
        )
        if report.get("identity") != identity:
            raise ValueError("La identidad de datos, código o configuración ha cambiado")
        parent = (
            _load(output, report["checkpoint"], report["samples"]["train"])
            if report["checkpoint"]
            else None
        )
        if report["status"] == "completed":
            if parent is None or parent.audit["rounds"] != rounds:
                raise ValueError("La referencia terminada no conserva todas sus rondas")
            for partition in ("train", "validation"):
                item = report["predictions"][partition]
                path = output / f"{partition}-predictions.parquet"
                safe_destination(path)
                if item["path"] != path.name or sha256(path) != item["sha256"]:
                    raise ValueError("Una predicción terminada no conserva su integridad")
            return report
        if len(report["attempts"]) >= 9999:
            raise ValueError("Se ha alcanzado el límite de intentos de recuperación")
        return _execute(dataset, output, report, parent, cp, xgb, stop or StopRequest())
    finally:
        os.close(lock)


def _execute(dataset, output, report, parent, cp, xgb, stop):
    options = report["identity"]["options"]
    batch_size = options["batch_size"]
    started = time.perf_counter()
    attempt = dict(started_at_utc=datetime.now(UTC).isoformat(), observed_device_used_bytes_max=0)
    report["attempts"].append(attempt)
    report["status"] = "running"

    def observed_memory():
        free, total = cp.cuda.runtime.memGetInfo()
        attempt["observed_device_used_bytes_max"] = max(
            attempt["observed_device_used_bytes_max"], total - free
        )

    def confirm(model):
        if _identity(dataset, options, cp, xgb) != report["identity"]:
            raise ValueError("El código científico ha cambiado durante el ajuste")
        count = model.booster.num_boosted_rounds()
        path = output / "checkpoints" / f"attempt-{len(report['attempts']):04}-round-{count:04}.ubj"
        digest = model.save(path)
        report.update(
            checkpoint=dict(path=str(path.relative_to(output)), sha256=digest),
            fitted_rows=model.training_rows,
            completed_rounds=count,
            audit=model.audit,
        )
        observed_memory()
        atomic_json(output / "run.json", report)
        if stop.requested:
            raise _Paused

    def factory():
        for batch in dataset.batches(partition="train", batch_size=batch_size, epoch=0, seed=0):
            if stop.requested:
                raise _Paused
            yield _matrix(batch, np.float32), batch["target"]

    try:
        observed_memory()
        atomic_json(output / "run.json", report)
        if stop.requested:
            raise _Paused
        fit_start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="external-", dir=output) as temporary:
            model = fit_external_boosting(
                factory,
                Path(temporary) / "pages",
                expected_rows=report["samples"]["train"],
                **{key: value for key, value in options.items() if key != "batch_size"},
                resume=parent,
                checkpoint=confirm,
            )
        attempt["fit_seconds"] = time.perf_counter() - fit_start
        if report["completed_rounds"] != options["rounds"]:
            confirm(model)
        restored = _load(output, report["checkpoint"], report["samples"]["train"])
        predictions = {}
        for partition in ("train", "validation"):
            if stop.requested:
                raise _Paused
            path = output / f"{partition}-predictions.parquet"
            metrics = _predict(
                model, restored, dataset, partition, batch_size, path, dtype=np.float32
            )
            predictions[partition] = dict(path=path.name, sha256=sha256(path), metrics=metrics)
        if _identity(dataset, options, cp, xgb) != report["identity"]:
            raise ValueError("La identidad cambió durante la evaluación")
        report.update(status="completed", predictions=predictions, restored_predictions_equal=True)
    except _Paused:
        report["status"] = "paused"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        observed_memory()
        attempt.update(
            status=report["status"],
            total_seconds=time.perf_counter() - started,
            process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            finished_at_utc=datetime.now(UTC).isoformat(),
            memory_scope=(
                "Muestras de toda la GPU, incluidas otras aplicaciones. No es el pico del proceso."
            ),
        )
        atomic_json(output / "run.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-bin", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batch-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--max-host-cache-bytes", type=int, default=16 * 1024**3)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--disk-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    options = vars(parser.parse_args())
    options["on_host"] = not options.pop("disk_cache")
    with StopRequest() as stop:
        result = run_external_reference(**options, stop=stop)
    print(f"Estado: {result['status']}. Rondas confirmadas: {result['completed_rounds']}")


if __name__ == "__main__":
    main()
