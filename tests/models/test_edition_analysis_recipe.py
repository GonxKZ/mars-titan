"""Reproducir el análisis con artefactos sintéticos de resultados conocidos."""

import csv
import itertools
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import atomic_json, sha256


def fixture_campaign(root):
    neural, tabular = root / "neural", root / "tabular"
    cases = []
    for kind, loss, lr, seed in itertools.product(
        ("rnn", "lstm", "gru", "dlinear"), ("mse", "mae", "huber"), (0.0001, 0.001), (42, 43, 44)
    ):
        cases.append(
            dict(
                id=f"{kind}-{loss}-{lr:g}-s{seed}",
                case=dict(kind=kind, loss=loss, learning_rate=lr, seed=seed, epochs=30),
            )
        )
    for kind, loss, seed in itertools.product(
        ("rnn", "lstm", "gru", "dlinear"), ("mse", "mae"), (42, 43, 44)
    ):
        cases.append(
            dict(
                id=f"{kind}-post-{loss}-s{seed}",
                parent=f"{kind}-mse-0.001-s{seed}",
                case=dict(kind=kind, loss=loss, learning_rate=0.0001, seed=seed, epochs=5),
            )
        )
    reports, neural_runs, tabular_runs = {}, [], []
    tabular_cases = [dict(id=f"ridge-{alpha}", alpha=alpha) for alpha in (0.1, 1, 10)]
    tabular_cases.append(dict(id="boosting", alpha=1))
    for item in cases + tabular_cases:
        is_neural = "case" in item
        folder = (neural if is_neural else tabular) / item["id"]
        folder.mkdir(parents=True)
        (folder / "weights.pt").write_bytes(b"Pesos simulados para comprobar el analisis")
        predictions = {}
        for partition, year in (("train", 2022), ("validation", 2023)):
            rows = [
                dict(
                    asset_id=f"US/{symbol}",
                    prediction_at=datetime(year, 1, day, tzinfo=UTC),
                    target=value,
                    prediction=value / 2,
                    zero=0.0,
                )
                for day in range(3, 9)
                for symbol, value in (("A", -1.0), ("B", 0.0), ("C", 1.0))
            ]
            path = folder / f"{partition}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            predictions[partition] = dict(
                path=path.name, sha256=sha256(path), metrics=dict(mae=1 / 3, mse=1 / 6)
            )
        report = dict(
            status="completed",
            final_test_opened=False,
            samples=dict(train=18, validation=18),
            predictions=predictions,
            checkpoint=dict(path="weights.pt", sha256=sha256(folder / "weights.pt")),
        )
        if is_neural:
            case = item["case"]
            report.update(
                identity=dict(case=case, batch_size=16),
                global_step=2 * case["epochs"],
                attempts=[],
                epochs=[dict(train=dict(samples=18), validation=dict(samples=18))] * case["epochs"],
            )
            if "parent" in item:
                report["initialization"] = dict(
                    optimizer_policy="new_adamw",
                    parent_checkpoint_sha256=reports[item["parent"]]["checkpoint"]["sha256"],
                )
        else:
            report.update(
                model="ridge" if item["id"].startswith("ridge") else "boosting",
                alpha=item["alpha"] if item["id"].startswith("ridge") else None,
                device="cpu",
                fit_seconds=1.0,
                total_seconds=2.0,
            )
        atomic_json(folder / "run.json", report)
        reports[item["id"]] = report
        record = dict(id=item["id"], path=item["id"], report_sha256=sha256(folder / "run.json"))
        if is_neural:
            record.update(item, arm="US", weighting="natural")
            neural_runs.append(record)
        else:
            tabular_runs.append(record)
    atomic_json(
        neural / "summary.json",
        dict(
            status="completed",
            completed_runs=96,
            planned_runs=96,
            scope="development_snapshot",
            cohort_complete=False,
            runs=neural_runs,
            identity=dict(
                scientific={},
                recipe_sha256=sha256(Path("configs/baselines/expanded-reference-variants.json")),
            ),
            started_at_utc="2026-01-01T00:00:00+00:00",
            finished_at_utc="2026-01-01T00:01:00+00:00",
        ),
    )
    atomic_json(
        tabular / "summary.json", dict(status="completed", completed_runs=4, runs=tabular_runs)
    )
    return neural, tabular


def test_selected_edition_produces_metrics_and_preserves_existing_output():
    Path("tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="edition-analysis-test-", dir="tmp") as directory:
        root = Path(directory)
        neural, tabular = fixture_campaign(root)
        output = root / "results"
        command = [
            sys.executable,
            "reports/analysis/verified_edition_20260923.py",
            str(output),
            "--campaign",
            str(neural),
            "--tabular",
            str(tabular),
            "--label",
            "synthetic",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        diagnostic = output / "streaming-reference-diagnostics-synthetic.json"
        data = json.loads(diagnostic.read_text())
        assert data["population"]["train"]["n"] == data["population"]["validation"]["n"] == 18
        assert len(data["runs"]) == 96 and len(data["tabular_runs"]) == 4
        with (output / "verified-synthetic-metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 100
        assert all(abs(float(row["validation_mae"]) - 1 / 3) < 1e-14 for row in rows)
        assert all(abs(float(row["validation_mse"]) - 1 / 6) < 1e-14 for row in rows)
        before = diagnostic.read_bytes()
        repeated = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert repeated.returncode != 0
        assert diagnostic.read_bytes() == before


def test_unsafe_edition_label_cannot_write_outside_the_destination(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "reports/analysis/verified_edition_20260923.py",
            str(tmp_path / "output"),
            "--label",
            "../../escape",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode != 0
    assert "identificador" in result.stderr
    assert not (tmp_path / "output").exists()


def test_external_campaigns_use_relative_artifact_paths(tmp_path):
    neural, tabular = fixture_campaign(tmp_path)
    output = tmp_path / "results"
    result = subprocess.run(
        [
            sys.executable,
            "reports/analysis/verified_edition_20260923.py",
            str(output),
            "--campaign",
            str(neural),
            "--tabular",
            str(tabular),
            "--label",
            "external",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads((output / "streaming-reference-diagnostics-external.json").read_text())
    assert {source["root"] for source in data["sources"]} == {"neural", "tabular"}
    assert all(not Path(source["path"]).is_absolute() for source in data["sources"])
    for source in data["sources"]:
        root = neural if source["root"] == "neural" else tabular
        assert sha256(root / source["path"]) == source["sha256"]


def test_analysis_cannot_disable_its_assertions(tmp_path):
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "reports/analysis/verified_edition_20260923.py",
            str(output),
            "--campaign",
            str(tmp_path / "absent"),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode != 0
    assert "comprobaciones" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("group", ["neural", "tabular"])
@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_declared_counters_cannot_hide_missing_or_duplicate_runs(tmp_path, group, change):
    neural, tabular = fixture_campaign(tmp_path)
    path = (neural if group == "neural" else tabular) / "summary.json"
    metadata = json.loads(path.read_text())
    if change == "missing":
        metadata["runs"].pop(0)
    else:
        metadata["runs"][0] = metadata["runs"][1]
    atomic_json(path, metadata)
    output = tmp_path / "results"
    result = subprocess.run(
        [
            sys.executable,
            "reports/analysis/verified_edition_20260923.py",
            str(output),
            "--campaign",
            str(neural),
            "--tabular",
            str(tabular),
            "--label",
            "invalid",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert not (output / "verified-invalid-metrics.csv").exists()
