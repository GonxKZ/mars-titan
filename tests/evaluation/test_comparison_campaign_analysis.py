"""Análisis de dos ZIP sin conservar los pesos ni los datos originales."""

import json
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.evaluation import comparison_campaign_analysis as analysis
from mars_titan.evaluation.comparison_bundle import export_results
from tests.evaluation.comparison_results_fixtures import campaign_state


def packages(tmp_path, *, failed=False, changed_target=False, changed_commit=False):
    declaration, digest, layout = campaign_state(tmp_path / "source")
    first = tmp_path / "references.zip"
    export_results(declaration, digest, layout, "references", first)
    key = "titans_mac/2019/seed-43"
    if failed:
        path = layout.record(key)
        record = json.loads(path.read_text())
        record["status"] = "failed"
        record["attempts"].append(dict(status="failed", error="fallo de prueba", seconds=2))
        path.write_text(json.dumps(record))
    if changed_target:
        directory = layout.output(key)
        path = directory / "evaluation-predictions.parquet"
        table = pq.read_table(path)
        target = table["target"].to_numpy().copy()
        target[0] += 0.01
        table = table.set_column(table.schema.get_field_index("target"), "target", pa.array(target))
        pq.write_table(table, path)
        report = json.loads((directory / "run.json").read_text())
        report["predictions"]["evaluation"].update(sha256=sha256(path), bytes=path.stat().st_size)
        (directory / "run.json").write_text(json.dumps(report))
    if changed_commit:
        permission = json.loads(layout.authorization.read_text())
        permission["code_commit"] = "b" * 40
        layout.authorization.write_text(json.dumps(permission))
    second = tmp_path / "mars-titan.zip"
    export_results(declaration, digest, layout, "mars_titan", second)
    return first, second


def test_all_metrics_and_figures_are_rebuilt_after_the_sources_are_removed(tmp_path):
    first, second = packages(tmp_path)
    shutil.rmtree(tmp_path / "source")

    report = analysis.analyze([first, second], tmp_path / "analysis")

    assert report["complete"] is True and report["missing_jobs"] == {}
    assert report["scope"]["expected_jobs"] == 4
    assert report["models"]["transformer"]["pooled"]["mae_raw"] > 0
    assert report["models"]["transformer"]["pooled"]["mae_calibrated"] == pytest.approx(
        report["models"]["transformer"]["pooled"]["zero_mae"]
    )
    assert report["models"]["transformer"]["by_origin"]["2019"]["seed_count"] == 2
    assert report["per_seed"][0]["calibration_factors"]["US"] is not None
    assert report["comparisons"] and report["protocol"]["bootstrap"]["stratification"] == "origin"
    for file in (
        "report.json",
        "metrics.csv",
        "resources.csv",
        "learning-curves.csv",
        "README.md",
        "plots/mae.png",
        "plots/information-coefficient.svg",
    ):
        assert (tmp_path / "analysis" / file).is_file(), file
    saved = json.loads((tmp_path / "analysis/report.json").read_text())
    assert saved == report


def test_archive_order_does_not_change_the_measured_results(tmp_path):
    first, second = packages(tmp_path)
    one = analysis.analyze([first, second], tmp_path / "one", plots=False)
    two = analysis.analyze([second, first], tmp_path / "two", plots=False)
    assert one["models"] == two["models"]
    assert one["comparisons"] == two["comparisons"]


def test_repeated_jobs_are_never_counted_twice(tmp_path):
    first, _ = packages(tmp_path)
    with pytest.raises(ValueError, match="repetid"):
        analysis.analyze([first, first], tmp_path / "analysis", plots=False)


def test_different_training_commits_are_not_mixed(tmp_path):
    first, second = packages(tmp_path, changed_commit=True)
    with pytest.raises(ValueError, match="código|commit"):
        analysis.analyze([first, second], tmp_path / "analysis", plots=False)


def test_missing_seeds_prevent_a_complete_comparison(tmp_path):
    first, second = packages(tmp_path, failed=True)
    with pytest.raises(ValueError, match="trabajos|incomplet"):
        analysis.analyze([first, second], tmp_path / "strict", plots=False)
    report = analysis.analyze(
        [first, second], tmp_path / "partial", allow_partial=True, plots=False
    )
    assert report["complete"] is False
    assert "titans_mac/2019/seed-43" in report["missing_jobs"]
    assert report["comparisons"] == {}
    assert report["inference_reason"]


def test_different_targets_are_rejected_even_with_valid_file_hashes(tmp_path):
    first, second = packages(tmp_path, changed_target=True)
    with pytest.raises(ValueError, match="filas|objetivos|población"):
        analysis.analyze([first, second], tmp_path / "analysis", plots=False)


def test_analysis_never_overwrites_previous_outputs(tmp_path):
    first, second = packages(tmp_path)
    destination = tmp_path / "analysis"
    destination.mkdir()
    (destination / "previous.txt").write_text("resultado anterior")
    with pytest.raises(ValueError, match="existe"):
        analysis.analyze([first, second], destination, plots=False)
    assert (destination / "previous.txt").read_text() == "resultado anterior"


def test_offline_analysis_has_a_reproducible_cli(tmp_path):
    first, second = packages(tmp_path)
    assert (
        analysis.main(
            [
                "--bundle",
                str(first),
                "--bundle",
                str(second),
                "--output",
                str(tmp_path / "cli"),
                "--no-plots",
            ]
        )
        == 0
    )
    assert (tmp_path / "cli/report.json").is_file()


def test_neural_seed_blocks_are_joined_without_ensembling_predictions(tmp_path):
    declaration, digest, layout = campaign_state(tmp_path / "source", blocks=True, reader=True)
    files = []
    for machine in ("references", "mars_titan"):
        path = tmp_path / (machine + ".zip")
        export_results(declaration, digest, layout, machine, path)
        files.append(path)
    report = analysis.analyze(files, tmp_path / "analysis", plots=False)
    assert report["complete"] and report["scope"]["expected_jobs"] == 6
    assert report["models"]["mars_titan_m3"]["by_origin"]["2019"]["seed_count"] == 2
    rows = [r for r in report["per_seed"] if r["model"] == "transformer"]
    observed = [r["partitions"]["evaluation"]["raw"]["point"]["mae"] for r in rows]
    assert report["models"]["transformer"]["pooled"]["mae_raw"] == pytest.approx(sum(observed) / 2)
    assert report["analysis_implementation"]["files"]["forecast_scores.py"]
    reader = next(r for r in report["resources"] if r["model"] == "mars_titan_m3")
    assert reader["wall_seconds"] == 5
    assert reader["pipeline_wall_seconds"] == 10
    assert reader["pipeline_cost_jobs"] == [reader["parent_job"], reader["job"]]


def test_single_machine_archive_stays_descriptive_until_both_are_joined(tmp_path):
    first, _ = packages(tmp_path)
    report = analysis.analyze([first], tmp_path / "one-machine", allow_partial=True, plots=False)
    assert report["complete"] is False and len(report["missing_jobs"]) == 2
    assert report["models"]["titans_mac"]["pooled"] == {}
    assert report["comparisons"] == {}


def test_failed_analysis_leaves_a_status_without_a_success_report(tmp_path, monkeypatch):
    first, second = packages(tmp_path)

    def failure(*_, **__):
        raise ValueError("tabla ilegible de prueba")

    monkeypatch.setattr(analysis.ResultsBundle, "read_table", failure)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="ilegible"):
        analysis.analyze([first, second], output, plots=False)
    status = json.loads((output / "analysis-status.json").read_text())
    assert status["status"] == "failed" and "ilegible" in status["error"]
    assert not (output / "report.json").exists()
