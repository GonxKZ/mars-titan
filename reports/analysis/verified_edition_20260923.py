"""Analizar la edición local de 405 muestras con los diagnósticos ya comprobados."""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.models.baselines.analysis import (
    _artifact,
    _breakdown,
    _cross_sectional,
    _posttraining_pairs,
    _predictions,
    _seed_groups,
)
from mars_titan.models.baselines.diagnostics import paired_date_bootstrap, point_diagnostics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "destination",
    type=Path,
    help="Directorio para los artefactos derivados, sin sobrescribir los anteriores",
)
destination = parser.parse_args().destination
for protected in (
    Path("dataset"),
    Path("data/processed"),
    Path("data/interim/verified-streaming-campaign-20260923"),
    Path("data/interim/tabular-edition-20260923"),
):
    outside_source(protected, destination)
if any(
    (destination / name).exists()
    for name in ("verified-20260923-metrics.csv", "streaming-reference-diagnostics-20260923.json")
):
    raise ValueError("La salida ya contiene resultados de esta edición")
destination.mkdir(parents=True, exist_ok=True)

root = Path("data/interim/verified-streaming-campaign-20260923")
summary_path = root / "summary.json"
summary = json.loads(summary_path.read_text())
assert (
    summary["status"] == "completed" and summary["completed_runs"] == summary["planned_runs"] == 96
)
assert all(r["arm"] == "US" and r["weighting"] == "natural" for r in summary["runs"])
assert summary["scope"] == "development_snapshot" and not summary["cohort_complete"]
population, predictions, reports, runs, items, metrics_rows = {}, {}, {}, [], [], []
bootstrap = dict(block_length=5, repetitions=2000, seed=42)
sources = []


def diagnostics(folder, report, identifier):
    outcome = {}
    predictions[identifier] = {}
    for partition in ("train", "validation"):
        artifact = report["predictions"][partition]
        path = _artifact(
            folder.resolve(),
            {"predictions_path": artifact["path"], "predictions_sha256": artifact["sha256"]},
            "predictions",
        )
        frame = _predictions(path, "prediction", partition, report["samples"][partition])
        identity = frame[["asset_id", "prediction_at", "date", "target"]]
        assert population.setdefault(partition, identity).equals(identity)
        target = frame.target.to_numpy(dtype=np.float64)
        estimate = frame.prediction.to_numpy(dtype=np.float64)
        checked_mae = math.fsum(
            abs(float(p) - float(y)) for p, y in zip(estimate, target, strict=True)
        ) / len(target)
        checked_mse = math.fsum(
            (float(p) - float(y)) ** 2 for p, y in zip(estimate, target, strict=True)
        ) / len(target)
        recorded = artifact["metrics"]
        assert abs(checked_mae - recorded["mae"]) < 1e-12
        assert abs(checked_mse - recorded["mse"]) < 1e-12
        metrics = point_diagnostics(target, estimate)
        assert (
            abs(metrics["mae"] - checked_mae) < 1e-12 and abs(metrics["mse"] - checked_mse) < 1e-12
        )
        predictions[identifier][partition] = estimate
        outcome[partition] = dict(
            metrics=metrics,
            by_asset=_breakdown(frame, "prediction", "asset_id"),
            by_month=_breakdown(frame, "prediction", frame.prediction_at.dt.strftime("%Y-%m")),
            cross_sectional=_cross_sectional(frame, "prediction"),
        )
        if partition == "validation":
            outcome[partition]["bootstrap_vs_zero"] = paired_date_bootstrap(
                frame.date.to_numpy(), target, estimate, **bootstrap
            )
        sources.append(dict(path=str(path.relative_to(Path.cwd())), sha256=artifact["sha256"]))
    return outcome


for item in summary["runs"]:
    folder = root / item["path"]
    report_path = folder / "run.json"
    assert sha256(report_path) == item["report_sha256"]
    report = json.loads(report_path.read_text())
    assert report["status"] == "completed" and report["final_test_opened"] is False
    assert report["identity"]["case"] == item["case"]
    assert all(report["identity"][k] == v for k, v in summary["identity"]["scientific"].items())
    assert len(report["epochs"]) == item["case"]["epochs"]
    assert all(
        e[p]["samples"] == report["samples"][p]
        for e in report["epochs"]
        for p in ("train", "validation")
    )
    checkpoint = _artifact(
        folder.resolve(),
        {
            "checkpoint_path": report["checkpoint"]["path"],
            "checkpoint_sha256": report["checkpoint"]["sha256"],
        },
        "checkpoint",
    )
    assert (
        report["global_step"]
        == math.ceil(report["samples"]["train"] / report["identity"]["batch_size"])
        * item["case"]["epochs"]
    )
    if "parent" in item:
        assert (
            report["initialization"]["parent_checkpoint_sha256"]
            == reports[item["parent"]]["checkpoint"]["sha256"]
        )
        assert report["initialization"]["optimizer_policy"] == "new_adamw"
    reports[item["id"]] = report
    run = dict(
        id=item["id"],
        **{k: v for k, v in item["case"].items() if k != "huber_delta"},
        stage="posttraining" if "parent" in item else "base",
        report_sha256=item["report_sha256"],
        checkpoint_sha256=report["checkpoint"]["sha256"],
        artifacts={
            "report": {"path": str(report_path.relative_to(root)), "sha256": item["report_sha256"]}
        },
        attempts=report["attempts"],
        **diagnostics(folder, report, item["id"]),
    )
    runs.append(run)
    items.append({**run, **({"parent": item["parent"]} if "parent" in item else {})})

tabular_root = Path("data/interim/tabular-edition-20260923")
tabular_summary = json.loads((tabular_root / "summary.json").read_text())
assert tabular_summary["status"] == "completed" and tabular_summary["completed_runs"] == 4
tabular_runs = []
for item in tabular_summary["runs"]:
    folder = tabular_root / item["path"]
    assert sha256(folder / "run.json") == item["report_sha256"]
    report = json.loads((folder / "run.json").read_text())
    assert report["status"] == "completed" and report["final_test_opened"] is False
    _artifact(
        folder.resolve(),
        {
            "checkpoint_path": report["checkpoint"]["path"],
            "checkpoint_sha256": report["checkpoint"]["sha256"],
        },
        "checkpoint",
    )
    tabular_runs.append(
        dict(
            id=item["id"],
            kind=report["model"],
            alpha=report["alpha"],
            stage="tabular",
            device=report["device"],
            report_sha256=item["report_sha256"],
            fit_seconds=report["fit_seconds"],
            total_seconds=report["total_seconds"],
            **diagnostics(folder, report, item["id"]),
        )
    )
common = set(population["train"].asset_id) & set(population["validation"].asset_id)
for run in runs + tabular_runs:
    row = {
        k: run.get(k)
        for k in ("id", "kind", "stage", "loss", "learning_rate", "seed", "epochs", "alpha")
    }
    for partition in ("train", "validation"):
        row.update(
            {
                partition + "_" + k: run[partition]["metrics"][k]
                for k in ("n", "mae", "mse", "rmse", "bias")
            }
        )
    metrics_rows.append(row)
metrics_path = destination / "verified-20260923-metrics.csv"
with metrics_path.open("x", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(metrics_rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(metrics_rows)
old = json.loads(Path("reports/resources/verified-reference-diagnostics.json").read_text())
result = dict(
    schema_version=2,
    kind="streaming_reference_analysis",
    scope="development_snapshot",
    campaign_sha256=sha256(summary_path),
    tabular_summary_sha256=sha256(tabular_root / "summary.json"),
    script_sha256=sha256(Path(__file__)),
    metrics_sha256=sha256(metrics_path),
    sources=sources,
    analysis_helpers_sha256=sha256(Path("src/mars_titan/models/baselines/analysis.py")),
    diagnostics_sha256=sha256(Path("src/mars_titan/models/baselines/diagnostics.py")),
    final_test_opened=False,
    exact_population_and_targets_match=True,
    independently_recomputed_cases=100,
    population={
        p: dict(
            n=len(f),
            n_dates=int(f.date.nunique()),
            assets=sorted(f.asset_id.unique()),
            first_date=f.date.min(),
            last_date=f.date.max(),
        )
        for p, f in population.items()
    },
    common_assets=sorted(common),
    runs=runs,
    tabular_runs=tabular_runs,
    seed_groups=_seed_groups(runs),
    posttraining_pairs=_posttraining_pairs(items, predictions, population, bootstrap),
    elapsed_campaign_wall_seconds=(
        datetime.fromisoformat(summary["finished_at_utc"])
        - datetime.fromisoformat(summary["started_at_utc"])
    ).total_seconds(),
    not_computed=old["not_computed"],
    limitations=old["limitations"]
    + [
        "Edición distinta de la anterior. Sus diferencias no se atribuyen solo al ejecutor.",
        "Análisis local limitado por el lector de predicciones a 100.000 filas por partición.",
        "No se ha repetido cada entrenamiento para comprobar recuperación. "
        "Existen pruebas de continuidad del motor.",
    ],
)
atomic_json(destination / "streaming-reference-diagnostics-20260923.json", result)
zero = runs[0]["validation"]["metrics"]["reference"]
print(
    json.dumps(
        dict(
            population=result["population"],
            common=result["common_assets"],
            zero=zero,
            better={
                m: sum(r["validation"]["metrics"][m] < zero[m] for r in runs + tabular_runs)
                for m in ("mae", "mse")
            },
            seconds=result["elapsed_campaign_wall_seconds"],
        )
    )
)
