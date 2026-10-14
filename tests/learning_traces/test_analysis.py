"""Análisis reproducible de trazas y su lectura en el observatorio (#448).

Las trazas se escriben en la prueba con valores fijos. No hay modelos ni optimizadores.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mars_titan.learning_traces.analysis import Run, align, analyze, diagnose, discover, markdown
from mars_titan.learning_traces.recorder import TraceConfig, TraceRecorder
from mars_titan.observatory.traces import bundle_name, sync_learning_traces

ROOT = Path(__file__).resolve().parents[2]


def write(folder, family, *, scores, losses, every=1, extra=(), step_unit="optimizer_step"):
    traces = TraceRecorder(
        folder,
        TraceConfig(every, 1 << 20),
        dict(family=family, run_identity_sha256="a" * 64),
        step_unit=step_unit,
    )
    for step, loss in enumerate(losses, start=1):
        with traces.at(step, "train"):
            traces.scalar("loss", loss)
            traces.scalar("grad_l2", 2.0 * step, group="predictor.memory")
            traces.scalar("update_ratio", 0.0, group="predictor.memory")
            traces.scalar("grad_l2_total", 1.5 if step % 2 else 0.5)
            traces.scalar("clip_threshold", 1.0)
            traces.scalar("memory.alpha.mean", 0.1 * step, group="predictor.blocks.0.memory")
            traces.scalar("memory.alpha.high_fraction", 0.0, group="predictor.blocks.0.memory")
    for step, score in scores:
        with traces.at(step, "validation"):
            traces.scalar("validation.score", score)
            traces.scalar("weight_l2", 4.0, group="predictor.memory")
    for step, metric, group, value in extra:
        with traces.at(step, "validation"):
            traces.scalar(metric, value, group=group)
    traces.flush()
    return folder


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "campaign"
    write(
        root / "jobs/a/traces",
        "titans_mac",
        scores=[(0, 0.5), (2, 0.25), (4, 0.375)],
        losses=[1.0, 0.75, 0.5, 0.25],
        extra=[(4, "adapter.delta_l2", "predictor.head", 0.125)],
    )
    write(root / "jobs/b/traces", "titans_mac", scores=[(0, 0.5), (3, 0.125)], losses=[2.0, 1.0])
    write(
        root / "jobs/c/traces",
        "b6",
        scores=[(5, math.inf)],
        losses=[],
        extra=[(5, "associative.writes", "all", 12.0)],
        step_unit="write_event",
    )
    return root


def test_each_run_is_summarised_with_the_declared_rules(campaign):
    run = diagnose(Run(campaign / "jobs/a/traces"))
    assert run["family"] == "titans_mac" and run["steps"] == [0, 4]
    # La validación inicial del paso 0 y los cuatro pasos del optimizador.
    assert run["recorded_steps"] == 5
    assert run["validation"]["best_step"] == 2 and run["validation"]["best_score"] == 0.25
    assert run["validation"]["last_score"] == 0.375
    optimization = run["optimization"]
    assert optimization["loss_first"] == 1.0 and optimization["loss_last"] == 0.25
    assert optimization["clip_fraction"] == 0.5 and optimization["traced_steps"] == 4
    group = optimization["groups"]["predictor.memory"]
    assert group["grad_l2_median"] == 5.0 and group["moved"] is False
    assert group["weight_l2_change"] == 0.0
    memory = run["titans"]["predictor.blocks.0.memory"]
    assert memory["alpha_last_mean"] == pytest.approx(0.4)
    assert memory["alpha_high_fraction_mean"] == 0.0
    assert run["adapters"] == {"predictor.head": {"delta_l2": 0.125}}
    assert run["integrity"] == dict(nonfinite_values=0, partially_finite_statistics=0)
    assert 0 < run["budget_fraction"] < 1 and run["exhausted_at_step"] is None


def test_non_finite_values_are_counted_and_never_chosen(campaign):
    run = diagnose(Run(campaign / "jobs/c/traces"))
    assert run["validation"]["best_step"] is None and run["validation"]["last_score"] is None
    assert run["integrity"]["nonfinite_values"] == 1
    assert run["b6"] == {"validation/associative.writes/all": 12.0}
    assert run["step_unit"] == "write_event"
    assert run["optimization"]["loss_first"] is None
    assert run["optimization"]["traced_steps"] == 0
    # Sin pasos del optimizador, los pasos con filas son los de sus escrituras.
    assert run["recorded_steps"] == 1


def test_curves_are_aligned_on_the_union_of_steps_without_interpolation(campaign):
    runs = {name: Run(campaign / f"jobs/{name}/traces") for name in ("a", "b")}
    curves = align(runs, "validation.score")
    assert curves["steps"] == [0, 2, 3, 4]
    assert curves["runs"] == {"a": [0.5, 0.25, None, 0.375], "b": [0.5, None, 0.125, None]}


def test_the_report_groups_families_and_renders_markdown(campaign):
    folders = discover(campaign)
    assert [f.relative_to(campaign).as_posix() for f in folders] == [
        "jobs/a/traces",
        "jobs/b/traces",
        "jobs/c/traces",
    ]
    report = analyze(folders, root=campaign)
    assert set(report["validation_by_family"]) == {"titans_mac", "b6"}
    assert report["validation_by_family"]["titans_mac"]["steps"] == [0, 2, 3, 4]
    text = markdown(report, note="Trazas de prueba.")
    assert "## `jobs/a/traces`" in text and "Trazas de prueba." in text
    assert "mejor 0.25 en el paso 2" in text
    assert "cada 1 escrituras de la memoria asociativa" in text
    assert "| `jobs/c/traces` | b6 | escrituras de la memoria asociativa | 1 |" in text
    rows = [line for line in text.splitlines() if line.startswith("| `jobs/")]
    assert len(rows) == 3 and all(line.endswith("| no |") for line in rows)
    with pytest.raises(ValueError, match="límite"):
        discover(campaign, limit=2)


def test_the_script_writes_the_same_report(campaign, tmp_path):
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_learning_traces.py"),
            "--root",
            str(campaign),
            "--output",
            str(output),
            "--markdown",
            str(tmp_path / "report.md"),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    expected = analyze(discover(campaign), root=campaign)
    for run in expected["runs"].values():
        run["folder"] = str(Path(run["folder"]).relative_to(campaign))
    assert report == json.loads(json.dumps(expected))
    assert (tmp_path / "report.md").read_text().startswith("# Análisis de trazas")


def test_the_observatory_sync_converts_only_changed_folders(campaign, tmp_path):
    destination, seen = tmp_path / "bundles", {}
    first = sync_learning_traces(campaign, destination, seen=seen)
    assert first["found"] == 3 and len(first["written"]) == 3 and first["skipped"] == []
    index = json.loads((destination / "index.json").read_text())
    names = {entry["name"] for entry in index["bundles"]}
    assert names == {bundle_name(Path(f"jobs/{n}/traces")) for n in "abc"}
    b6 = json.loads((destination / f"{bundle_name(Path('jobs/c/traces'))}.json").read_text())
    assert b6["model_id"] == "b6" and b6["cadence"] == 1 and b6["x_unit"] == "write_event"
    assert sync_learning_traces(campaign, destination, seen=seen)["written"] == []
    write(campaign / "jobs/d/traces", "gru", scores=[(1, 0.5)], losses=[1.0])
    assert sync_learning_traces(campaign, destination, seen=seen)["written"] == ["jobs/d/traces"]
    (campaign / "jobs/a/traces/parts/part-00000001.parquet").write_bytes(b"roto")
    broken = sync_learning_traces(campaign, destination, seen=seen)
    assert [item["folder"] for item in broken["skipped"]] == ["jobs/a/traces"]


def test_bundle_names_are_stable_and_valid():
    long = Path("jobs/US+CN/fold-012/mars_titan_m1_k2/seed-43/" + "x" * 90 + "/traces")
    name = bundle_name(long)
    assert name == bundle_name(long) and len(name) <= 81
    assert bundle_name(Path("a/traces")) != bundle_name(Path("a.traces"))
