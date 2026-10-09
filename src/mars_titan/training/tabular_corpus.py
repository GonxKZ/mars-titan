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
from mars_titan.data.input_policy import (
    INPUT_POLICIES,
    STRICT_INPUTS,
    masked_inputs,
    policy_identity,
)
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.session_metrics import SessionErrors
from mars_titan.models.baselines.boosting import BoostingModel, fit_boosting_batches
from mars_titan.models.baselines.inputs import MODALITIES
from mars_titan.models.baselines.ridge import RidgeModel, fit_ridge_blocks

from .corpus_inputs import CorpusDataset
from .learning_hold import require_learning_allowed
from .reference_run import FULL_TRAIN_VALIDATION, PREDICTION_RETENTIONS


def retained_partitions(retention):
    """Particiones con tabla por fila. La retención reservada solo resume el ajuste."""
    if retention not in PREDICTION_RETENTIONS:
        raise ValueError("La retención de predicciones no pertenece al contrato")
    if retention == FULL_TRAIN_VALIDATION:
        return ("train", "validation")
    return ("validation", "calibration", "evaluation")


def feature_order(input_policy):
    """Bloques en orden canónico. Con máscaras se añaden al final los cinco bits."""
    return [*MODALITIES, "presence"] if masked_inputs(input_policy) else list(MODALITIES)


def _matrix(batch, dtype=np.float64, *, presence=False):
    """Concatenar las modalidades y, si se declara, los bits de presencia del lector."""
    count = len(batch["target"])
    if ("presence" in batch) != presence:
        raise ValueError("Los bits de presencia no coinciden con la política declarada")
    blocks = [batch["inputs"][name].reshape(count, -1) for name in MODALITIES]
    if presence:
        bits = batch["presence"]
        if bits.dtype != np.bool_ or bits.shape != (count, len(MODALITIES)):
            raise ValueError("La presencia necesita cinco booleanos por fila")
        blocks.append(bits)
    return np.concatenate(blocks, axis=1).astype(dtype, copy=False)


def _predict(
    model,
    restored,
    dataset,
    partition,
    batch_size,
    destination,
    *,
    dtype=np.float64,
    presence=False,
):
    """Predecir una partición completa. Sin destino solo se resumen los errores.

    Sin `restored` no se repite la comparación con el modelo recargado, como en las
    predicciones de un modelo ya confirmado en otra ventana.
    """
    count, square, absolute, zero_square, zero_absolute = 0, 0.0, 0.0, 0.0, 0.0
    sessions = SessionErrors()

    def tables():
        nonlocal count, square, absolute, zero_square, zero_absolute
        for batch in dataset.batches(partition=partition, batch_size=batch_size, epoch=0, seed=0):
            matrix, target = _matrix(batch, dtype, presence=presence), batch["target"]
            prediction = model.predict(matrix)
            if (
                restored is not None and not np.array_equal(prediction, restored.predict(matrix))
            ) or not np.isfinite(prediction).all():
                raise ValueError(
                    "Las predicciones restauradas difieren o contienen valores no finitos"
                )
            error = prediction - target
            sessions.update(batch["market"], batch["prediction_at"], error)
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

    if destination is None:
        for _ in tables():
            pass
    else:
        atomic_parquet_batches(destination, tables())
    if count != dataset.manifest["counts"][partition] or not count:
        raise ValueError("Las predicciones no recorren exactamente la población declarada")
    return dict(
        samples=count,
        mse=square / count,
        mae=absolute / count,
        zero_mse=zero_square / count,
        zero_mae=zero_absolute / count,
        **{key: value for key, value in sessions.summary().items() if key != "samples"},
    )


def run_tabular_reference(
    manifest: Path,
    output: Path,
    *,
    kind: str,
    alpha: float = 1.0,
    batch_size: int = 256,
    max_matrix_bytes: int = 256 * 1024**2,
    input_policy: str = STRICT_INPUTS,
    prediction_retention: str = FULL_TRAIN_VALIDATION,
) -> dict:
    """Ajustar todas las filas o declarar falta de presupuesto, nunca reducir la población.

    La retención reservada escribe validación, calibración y evaluación por fila y
    resume el ajuste sin tabla. La retención anterior conserva ajuste y validación.
    """
    require_learning_allowed("el ajuste de la referencia tabular")
    partitions = retained_partitions(prediction_retention)
    if (
        kind not in {"ridge", "boosting"}
        or input_policy not in INPUT_POLICIES
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
    masked = masked_inputs(input_policy)
    dataset = CorpusDataset(manifest, input_policy=input_policy)
    if min(dataset.manifest["counts"].values()) < 1:
        raise ValueError("Se necesitan particiones de ajuste y validación no vacías")
    for protected in (*dataset.roots.values(), Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)
    if output.exists() or output.is_symlink():
        raise ValueError("Usa un directorio nuevo para conservar las ejecuciones existentes")
    first = next(dataset.batches(partition="train", batch_size=batch_size, epoch=0, seed=0))
    features = _matrix(first, presence=masked).shape[1]
    estimated = dataset.manifest["counts"]["train"] * (features + 1) * 8
    root = Path(__file__).parents[1]
    sources = (
        "training/tabular_corpus.py",
        "training/corpus_inputs.py",
        "training/temporal_corpus.py",
        "evaluation/splits.py",
        "evaluation/split_readiness.py",
        "data/streaming.py",
        "data/batches.py",
        "models/baselines/inputs.py",
        "models/baselines/ridge.py",
        "models/baselines/boosting.py",
    ) + (("data/input_policy.py",) if masked else ())
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
        feature_order=feature_order(input_policy),
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
        **policy_identity(input_policy),
    )
    # El campo solo aparece fuera de la retención anterior, que conserva su recibo.
    if prediction_retention != FULL_TRAIN_VALIDATION:
        report["prediction_retention"] = prediction_retention
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
            yield _matrix(batch, presence=masked), batch["target"]
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
        for partition in partitions:
            path = output / f"{partition}-predictions.parquet"
            metrics = _predict(
                model, restored, dataset, partition, batch_size, path, presence=masked
            )
            predictions[partition] = dict(path=path.name, sha256=sha256(path), metrics=metrics)
        if "train" not in partitions:
            # Las particiones reservadas ya han comparado el modelo recargado.
            report["train_metrics"] = _predict(
                model, None, dataset, "train", batch_size, None, presence=masked
            )
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
    parser.add_argument("--input-policy", choices=INPUT_POLICIES, default=STRICT_INPUTS)
    parser.add_argument(
        "--prediction-retention", choices=PREDICTION_RETENTIONS, default=FULL_TRAIN_VALIDATION
    )
    args = parser.parse_args()
    result = run_tabular_reference(
        args.manifest,
        args.output,
        kind=args.kind,
        alpha=args.alpha,
        batch_size=args.batch_size,
        max_matrix_bytes=args.max_matrix_bytes,
        input_policy=args.input_policy,
        prediction_retention=args.prediction_retention,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
