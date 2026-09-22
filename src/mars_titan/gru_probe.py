"""Medir referencias temporales multimodales con recuperación exacta."""

import argparse
import json
import os
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader

from mars_titan.budget_training import (
    SupervisedWorkload,
    gpu_sensors,
    load_checkpoint,
    process_memory,
    run_epoch,
    save_checkpoint,
    seed_run,
)
from mars_titan.data.embeddings import require_cuda
from mars_titan.data.preparation import atomic_parquet
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.streaming import iter_windows
from mars_titan.profiling import MODALITIES, CostProbe
from mars_titan.reference_probe import prepare_probe


def run_temporal_probe(prepared, samples, output, report_path, *, epochs=3, kind="gru"):
    """Ejecutar una configuración de coste, sin selección sobre el test final."""
    if not isinstance(epochs, int) or not 2 <= epochs <= 10:
        raise ValueError("La medición requiere entre 2 y 10 épocas")
    if kind not in {"gru", "dlinear"}:
        raise ValueError("La referencia temporal solicitada no está implementada")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError("Configura CUBLAS_WORKSPACE_CONFIG antes de iniciar PyTorch")
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    paths, targets, audit, hashes, counts = prepare_probe(prepared, samples, output, report_path)
    for relative in ("gru_probe.py", "reference_probe.py"):
        hashes[f"src/mars_titan/{relative}"] = sha256(Path(__file__).with_name(relative))
    if kind == "dlinear":
        source = Path(__file__).parent / "models/baselines/dlinear.py"
        hashes["src/mars_titan/models/baselines/dlinear.py"] = sha256(source)
    device = require_cuda()
    seed_run(42)
    loaders = {
        partition: DataLoader(
            SupervisedWorkload(paths, prepared, targets, partition),
            batch_size=16,
            num_workers=0,
            pin_memory=True,
        )
        for partition in ("train", "validation")
    }
    first, _ = next(iter(loaders["train"]))
    dimensions = {name: value.shape[-1] for name, value in first.items()}
    model = CostProbe(kind, dimensions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    config = {
        "kind": kind,
        "dimensions": dimensions,
        "context": 64,
        "hidden_size": 32,
        "batch_size": 16,
        "workers": 0,
        "seed": 42,
        "epochs": epochs,
        "precision": "float32",
        "lr": 1e-4,
        "optimizer": "AdamW",
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "state_policy": "reset_each_window",
        "moving_average_kernel": 25 if kind == "dlinear" else None,
        "price_output_horizon": 1 if kind == "dlinear" else None,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    report = {
        "purpose": "strict_supervised_execution_probe_not_confirmatory_comparison",
        "started_at_utc": started_at,
        "model": kind,
        "config": config,
        "samples": counts,
        "modalities": list(MODALITIES),
        "input_hashes": hashes,
        "target_audit": audit,
        "training_cutoff": "2022-12-31",
        "validation_year": 2023,
        "final_test_opened": False,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "parameters": sum(value.numel() for value in model.parameters()),
        "sensors_before": gpu_sensors(),
        "preparation_seconds": time.perf_counter() - started,
        "epochs": [],
        "status": "running",
        "timing_scope": "runner_with_preparation_and_replay_without_imports_or_process_start",
        "limitations": [
            "small_directed_editorial_sample",
            "fixed_epochs_no_model_selection",
            "synchronized_step_timing",
            "operating_system_cache_uncontrolled",
        ],
    }
    atomic_json(report_path, report)
    for epoch in range(epochs):
        training = run_epoch(model, optimizer, loaders["train"], device)
        validation = run_epoch(model, None, loaders["validation"], device)
        if training["samples"] != counts["train"] or validation["samples"] != counts["validation"]:
            raise ValueError("La época no reconcilia con las etiquetas de cada partición")
        checkpoint_started = time.perf_counter()
        checkpoint = output / f"epoch-{epoch + 1}.pt"
        save_checkpoint(
            checkpoint, model, optimizer, next_epoch=epoch + 1, config=config, hashes=hashes
        )
        report["epochs"].append(
            {
                "epoch": epoch + 1,
                "train": training,
                "validation": validation,
                "checkpoint_seconds": time.perf_counter() - checkpoint_started,
                "checkpoint_sha256": sha256(checkpoint),
            }
        )
        atomic_json(report_path, report)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    resume_started = time.perf_counter()
    next_epoch = load_checkpoint(
        output / "epoch-1.pt", model, optimizer, config=config, hashes=hashes
    )
    for _ in range(next_epoch, epochs):
        run_epoch(model, optimizer, loaders["train"], device)
        run_epoch(model, None, loaders["validation"], device)
    exact = all(torch.equal(expected[name], value) for name, value in model.state_dict().items())
    if not exact:
        raise ValueError("La reanudación no reproduce los pesos de la ejecución continua")
    report["resume_check"] = {
        "from_epoch": 1,
        "exact_weights": exact,
        "elapsed_seconds": time.perf_counter() - resume_started,
    }
    restored = CostProbe(kind, dimensions).to(device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    load_checkpoint(checkpoint, restored, restored_optimizer, config=config, hashes=hashes)
    model.eval()
    restored.eval()
    predictions = []
    with torch.inference_mode():
        for row in iter_windows(paths, prepared, decision_cutoff="2023-12-31"):
            label = targets.get(row["cursor"][0], {}).get(row["prediction_at"].isoformat())
            if label is None or label[1] != "validation":
                continue
            inputs = {
                name: torch.from_numpy(value).unsqueeze(0).to(device)
                for name, value in row["inputs"].items()
            }
            predicted = model(inputs)
            if not torch.isfinite(predicted).all() or not torch.equal(predicted, restored(inputs)):
                raise ValueError("La predicción no es finita o cambia tras restaurar")
            predictions.append(
                {
                    "asset_id": row["cursor"][0],
                    "prediction_at": row["prediction_at"],
                    "target": label[0],
                    kind: float(predicted.item()),
                    "zero": 0.0,
                }
            )
    if len(predictions) != counts["validation"]:
        raise ValueError("Las predicciones no reconcilian con las etiquetas de validación")
    atomic_parquet(output / "predictions.parquet", pa.Table.from_pylist(predictions))
    target = np.array([row["target"] for row in predictions])
    report["diagnostic_row_metrics"] = {
        name: {
            "mae": float(np.abs(np.array([row[name] for row in predictions]) - target).mean()),
            "mse": float(np.square(np.array([row[name] for row in predictions]) - target).mean()),
        }
        for name in (kind, "zero")
    }
    report.update(
        status="completed",
        restored_predictions_equal=True,
        predictions_sha256=sha256(output / "predictions.parquet"),
        sensors_after=gpu_sensors(),
        total_seconds=time.perf_counter() - started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        process_memory_at_end=process_memory(),
        finished_at_utc=datetime.now(UTC).isoformat(),
    )
    atomic_json(report_path, report)
    return report


def run_gru_probe(prepared, samples, output, report_path, *, epochs=3):
    """Conservar la entrada anterior de la sonda GRU."""
    return run_temporal_probe(prepared, samples, output, report_path, epochs=epochs, kind="gru")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("prepared", "samples", "output", "report"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--kind", choices=["gru", "dlinear"], default="gru")
    args = parser.parse_args()
    print(
        json.dumps(
            run_temporal_probe(
                args.prepared,
                args.samples,
                args.output,
                args.report,
                epochs=args.epochs,
                kind=args.kind,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
