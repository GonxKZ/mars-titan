"""Ridge y boosting sobre la misma edición supervisada que las referencias neurales."""

import argparse
import importlib.metadata
import json
import math
import platform
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.embeddings import require_cuda
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.models.baselines.boosting import BoostingModel, fit_boosting_batches
from mars_titan.models.baselines.inputs import MODALITIES
from mars_titan.models.baselines.ridge import RidgeModel, fit_ridge_blocks

from .corpus_inputs import CorpusDataset


def _matrix(batch):
    count = len(batch["target"])
    return np.concatenate(
        [batch["inputs"][name].reshape(count, -1) for name in MODALITIES], axis=1
    ).astype(np.float64)


def _predict(model, restored, dataset, partition, batch_size, destination):
    count, square, absolute, zero_square, zero_absolute = 0, 0.0, 0.0, 0.0, 0.0

    def tables():
        nonlocal count, square, absolute, zero_square, zero_absolute
        for batch in dataset.batches(partition=partition, batch_size=batch_size, epoch=0, seed=0):
            matrix, target = _matrix(batch), batch["target"]
            prediction = model.predict(matrix)
            if (
                not np.array_equal(prediction, restored.predict(matrix))
                or not np.isfinite(prediction).all()
            ):
                raise ValueError(
                    "Las predicciones restauradas difieren o contienen valores no finitos"
                )
            error = prediction - target
            square += float(np.square(error).sum())
            absolute += float(np.abs(error).sum())
            zero_square += float(np.square(target).sum())
            zero_absolute += float(np.abs(target).sum())
            count += len(target)
            yield pa.table(
                dict(
                    sample_id=batch["sample_ids"],
                    asset_id=["/".join(s.split("/")[:2]) for s in batch["sample_ids"]],
                    market=batch["market"],
                    prediction_at=pa.array(
                        batch["prediction_at"], type=pa.timestamp("us", tz="UTC")
                    ),
                    target=target,
                    prediction=prediction,
                    zero=np.zeros(len(target), dtype=np.float64),
                )
            )

    atomic_parquet_batches(destination, tables())
    if count != dataset.manifest["counts"][partition] or not count:
        raise ValueError("Las predicciones no recorren exactamente la población declarada")
    return dict(
        samples=count,
        mse=square / count,
        mae=absolute / count,
        zero_mse=zero_square / count,
        zero_mae=zero_absolute / count,
    )


def run_tabular_reference(
    manifest: Path,
    output: Path,
    *,
    kind: str,
    alpha: float = 1.0,
    batch_size: int = 256,
    max_matrix_bytes: int = 256 * 1024**2,
) -> dict:
    """Ajustar todas las filas o declarar falta de presupuesto, nunca reducir la población."""
    if (
        kind not in {"ridge", "boosting"}
        or type(batch_size) is not int
        or not 1 <= batch_size <= 4096
        or type(max_matrix_bytes) is not int
        or not 1 <= max_matrix_bytes <= 4 * 1024**3
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("La configuración tabular no es válida")
    started = time.perf_counter()
    device = str(require_cuda()) if kind == "ridge" else "cpu"
    dataset = CorpusDataset(manifest)
    if min(dataset.manifest["counts"].values()) < 1:
        raise ValueError("Se necesitan particiones de ajuste y validación no vacías")
    for protected in (*dataset.roots.values(), Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)
    if output.exists() or output.is_symlink():
        raise ValueError("Usa un directorio nuevo para conservar las ejecuciones existentes")
    first = next(dataset.batches(partition="train", batch_size=batch_size, epoch=0, seed=0))
    features = _matrix(first).shape[1]
    estimated = dataset.manifest["counts"]["train"] * (features + 1) * 8
    root = Path(__file__).parents[1]
    sources = (
        "training/tabular_corpus.py",
        "training/corpus_inputs.py",
        "data/streaming.py",
        "data/batches.py",
        "models/baselines/inputs.py",
        "models/baselines/ridge.py",
        "models/baselines/boosting.py",
    )
    code = {name: sha256(root / name) for name in sources}
    report = dict(
        schema_version=1,
        status="running",
        model=kind,
        alpha=alpha if kind == "ridge" else None,
        scope=dataset.manifest["scope"],
        cohort_complete=dataset.manifest["cohort_complete"],
        manifest_sha256=dataset.identity,
        samples=dataset.manifest["counts"],
        fitted_rows=0,
        feature_order=list(MODALITIES),
        features=features,
        batch_size=batch_size,
        matrix_bytes_estimate=estimated,
        max_matrix_bytes=max_matrix_bytes,
        device=device,
        inference_device=device,
        final_test_opened=False,
        started_at_utc=datetime.now(UTC).isoformat(),
        code=code,
        numpy=np.__version__,
        pyarrow=pa.__version__,
        torch=str(torch.__version__),
        sklearn=importlib.metadata.version("scikit-learn"),
        runtime={
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if kind == "ridge" else None,
            "machine": platform.machine(),
            "boosting_threads": 4 if kind == "boosting" else None,
        },
    )
    output.mkdir(parents=True, exist_ok=False)
    if kind == "boosting" and estimated > max_matrix_bytes:
        report.update(
            status="not_executed_budget",
            reason="La matriz completa supera el presupuesto declarado",
        )
        atomic_json(output / "run.json", report)
        return report
    atomic_json(output / "run.json", report)
    counts = []

    def factory():
        visited = 0
        for batch in dataset.batches(partition="train", batch_size=batch_size, epoch=0, seed=0):
            visited += len(batch["target"])
            yield _matrix(batch), batch["target"]
        if visited != dataset.manifest["counts"]["train"]:
            raise ValueError("El ajuste no recorre exactamente toda la población")
        counts.append(visited)

    if kind == "ridge":
        torch.cuda.reset_peak_memory_stats(0)
    try:
        fitting = time.perf_counter()
        model = (
            fit_ridge_blocks(factory, alpha=alpha)
            if kind == "ridge"
            else fit_boosting_batches(factory, max_bytes=max_matrix_bytes)
        )
        if not counts:
            raise ValueError("El estimador no ha consumido la población de ajuste")
        report.update(
            fit_seconds=time.perf_counter() - fitting, fitted_rows=counts[0], fit_passes=len(counts)
        )
        checkpoint = output / ("model.npz" if kind == "ridge" else "model.joblib")
        model.save(checkpoint)
        restored = (
            RidgeModel.load(checkpoint)
            if kind == "ridge"
            else BoostingModel.load_local(checkpoint, sha256(checkpoint))
        )
        if kind == "boosting":
            report["parameters"] = model.estimator.get_params()
        report["checkpoint"] = dict(path=checkpoint.name, sha256=sha256(checkpoint))
        predictions = {}
        for partition in ("train", "validation"):
            path = output / f"{partition}-predictions.parquet"
            metrics = _predict(model, restored, dataset, partition, batch_size, path)
            predictions[partition] = dict(path=path.name, sha256=sha256(path), metrics=metrics)
        if any(sha256(root / name) != digest for name, digest in code.items()):
            raise ValueError("El código ha cambiado durante el ajuste")
        report.update(
            status="completed",
            predictions=predictions,
            restored_predictions_equal=True,
            finished_at_utc=datetime.now(UTC).isoformat(),
        )
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(
            total_seconds=time.perf_counter() - started,
            process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(0)
            if kind == "ridge"
            else None,
        )
        atomic_json(output / "run.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--kind", choices=("ridge", "boosting"), required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-matrix-bytes", type=int, default=256 * 1024**2)
    args = parser.parse_args()
    result = run_tabular_reference(
        args.manifest,
        args.output,
        kind=args.kind,
        alpha=args.alpha,
        batch_size=args.batch_size,
        max_matrix_bytes=args.max_matrix_bytes,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
