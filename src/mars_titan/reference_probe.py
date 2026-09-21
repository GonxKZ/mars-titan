"""Sondas de referencias sobre una muestra estricta, no una comparación confirmatoria."""

import argparse
import json
import resource
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.budget_training import gpu_sensors, prepare_targets, process_memory
from mars_titan.data.embeddings import require_cuda
from mars_titan.data.preparation import atomic_parquet
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.data.streaming import iter_windows
from mars_titan.models.baselines.inputs import MODALITIES, feature_vector, tabular_batches
from mars_titan.models.baselines.ridge import RidgeModel, fit_ridge_blocks


def run_reference_probe(
    prepared: Path, samples: Path, output: Path, report_path: Path, *, alpha: float = 1.0
):
    outside_source(output, report_path)
    for target in (output, report_path):
        for protected in (Path("dataset"), prepared, samples):
            outside_source(protected, target)
    if output.exists() or report_path.exists():
        raise ValueError("Usa un directorio y un informe nuevos para conservar las ejecuciones")
    listing = subprocess.run(
        ["rg", "--files", "--no-ignore", str(samples), "-g", "samples.parquet"],
        capture_output=True,
        text=True,
        check=True,
    )
    paths = []
    for name in sorted(listing.stdout.splitlines()):
        path = Path(name)
        if not path.resolve().is_relative_to(samples.resolve()) or path.is_symlink():
            raise ValueError("Una muestra sale del directorio declarado")
        with pq.ParquetFile(path) as file:
            if not file.metadata.num_rows:
                continue
        manifest = json.loads((path.parent / "manifest.json").read_text())
        original = json.loads((prepared / "US" / path.parent.name / "manifest.json").read_text())
        if any(
            value.get("news_content_policy") != "verified_full_articles"
            for value in (manifest, original)
        ):
            raise ValueError("La sonda requiere noticias completas verificadas")
        paths.append(path)
    if not 1 <= len(paths) <= 64:
        raise ValueError("La sonda necesita entre uno y 64 activos con muestras estrictas")
    started = time.perf_counter()
    targets, audit, hashes = prepare_targets(paths, prepared, output / "targets")
    counts = {
        partition: sum(asset[partition] for asset in audit["assets"].values())
        for partition in ("train", "validation")
    }
    if counts["train"] < 2 or counts["validation"] < 1:
        raise ValueError("No hay cobertura suficiente en entrenamiento y validación")
    if sum(counts.values()) > 100_000:
        raise ValueError("La muestra supera el presupuesto de esta sonda")
    for path in (
        Path(__file__),
        Path(__file__).parent / "models/baselines/ridge.py",
        Path(__file__).parent / "models/baselines/inputs.py",
    ):
        hashes["src/mars_titan/" + path.relative_to(Path(__file__).parent).as_posix()] = sha256(
            path
        )
    import torch

    device = require_cuda()
    sensors_before = gpu_sensors()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fit_started = time.perf_counter()

    def training():
        return tabular_batches(
            iter_windows(paths, prepared, decision_cutoff="2022-12-31"), targets, "train"
        )

    model = fit_ridge_blocks(training, alpha=alpha, device=str(device))
    torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - fit_started
    checkpoint = output / "ridge.npz"
    model.save(checkpoint)
    restored = RidgeModel.load(checkpoint)
    predictions = []
    predict_started = time.perf_counter()
    for row in iter_windows(paths, prepared, decision_cutoff="2023-12-31"):
        label = targets.get(row["cursor"][0], {}).get(row["prediction_at"].isoformat())
        if label is None or label[1] != "validation":
            continue
        x = feature_vector(row["inputs"])[None, :]
        predicted = model.predict(x)
        value = float(predicted[0])
        if not np.array_equal(predicted, restored.predict(x)):
            raise ValueError("El modelo restaurado produce otra predicción")
        predictions.append(
            {
                "asset_id": row["cursor"][0],
                "prediction_at": row["prediction_at"],
                "target": label[0],
                "ridge": value,
                "zero": 0.0,
            }
        )
    torch.cuda.synchronize()
    predict_seconds = time.perf_counter() - predict_started
    atomic_parquet(output / "predictions.parquet", pa.Table.from_pylist(predictions))
    target = np.array([row["target"] for row in predictions])
    metrics = {
        name: {
            "mae": float(np.abs(np.array([row[name] for row in predictions]) - target).mean()),
            "mse": float(np.square(np.array([row[name] for row in predictions]) - target).mean()),
        }
        for name in ("ridge", "zero")
    }
    report = {
        "purpose": "strict_supervised_execution_probe_not_confirmatory_comparison",
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "model": "ridge",
        "alpha": alpha,
        "samples": counts,
        "features": len(model.mean),
        "feature_order": list(MODALITIES),
        "scaling": "train_only_population_variance",
        "intercept": "unpenalized_with_residual_centering_correction",
        "loss": "sum_squared_error_plus_alpha_l2",
        "training_cutoff": "2022-12-31",
        "validation_year": 2023,
        "final_test_opened": False,
        "fit_seconds": fit_seconds,
        "validation_and_restore_check_seconds": predict_seconds,
        "total_seconds": time.perf_counter() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "sensors_before": sensors_before,
        "sensors_after": gpu_sensors(),
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "process_memory_at_end": process_memory(),
        "diagnostic_row_metrics": metrics,
        "restored_predictions_equal": True,
        "target_audit": audit,
        "input_hashes": hashes,
        "checkpoint_sha256": sha256(checkpoint),
        "predictions_sha256": sha256(output / "predictions.parquet"),
        "limitations": [
            "small_directed_editorial_sample",
            "one_fixed_configuration",
            "no_search_or_predictive_superiority_claim",
            "operating_system_cache_uncontrolled",
        ],
    }
    atomic_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_reference_probe(args.prepared, args.samples, args.output, args.report), indent=2
        )
    )


if __name__ == "__main__":
    main()
