"""Revisión independiente de predicciones confirmadas, sin abrir el test final."""

import importlib
import json
import subprocess
import sys
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import atomic_json, sha256


def reviewer():
    try:
        return importlib.import_module("mars_titan.evaluation.prediction_review")
    except ModuleNotFoundError:
        pytest.fail("Falta la revisión incremental de predicciones")


def fixture_run(root, *, nested=False, year=2023, predictions=None):
    folder = root / "runs" / "case"
    folder.mkdir(parents=True)
    times = [datetime(year, 1, day, tzinfo=UTC) for day in (2, 2, 3)]
    table = pa.table(
        dict(
            market=["US"] * 3,
            prediction_at=pa.array(times, type=pa.timestamp("us", tz="UTC")),
            target=[1.0, 3.0, 6.0],
            prediction=[0.0, 0.0, 0.0] if predictions is None else predictions,
            parent=[1.0, 3.0, 5.0],
            zero=[0.0] * 3,
        )
    )
    prediction_path = folder / "validation-predictions.parquet"
    pq.write_table(table, prediction_path, row_group_size=2)
    metrics = dict(samples=3, session_count=2, session_mae=4.0, session_mse=20.5)
    if nested:
        metrics = dict(
            median=metrics,
            parent=dict(samples=3, session_count=2, session_mae=0.5, session_mse=0.5),
            zero=metrics,
        )
    report = dict(
        status="completed",
        final_test_opened=False,
        cohort_complete=True,
        predictions=dict(
            validation=dict(
                path=prediction_path.name, sha256=sha256(prediction_path), metrics=metrics
            )
        ),
    )
    atomic_json(folder / "run.json", report)
    summary = dict(
        final_test_opened=False,
        runs=[
            dict(
                id="case",
                path="runs/case",
                status="completed",
                report_sha256=sha256(folder / "run.json"),
            )
        ],
    )
    atomic_json(root / "summary.json", summary)
    return root / "summary.json", folder / "run.json", prediction_path


def review(tmp_path, source, **kwargs):
    return reviewer().review_campaigns(
        [{"id": "study", "summary": str(source)}],
        tmp_path / "review.json",
        partitions=("validation",),
        **kwargs,
    )


def test_recalculates_equal_session_weights_and_controls(tmp_path):
    source, _, _ = fixture_run(tmp_path / "source", nested=True)
    result = review(tmp_path, source)
    assert result["counts"] == dict(confirmed=1, verified=1, failed=0, pending=0)
    entry = result["entries"]["study/case/validation"]
    assert entry["metrics"]["median"]["session_mae"] == 4.0
    assert entry["metrics"]["median"]["mae"] == pytest.approx(10 / 3)
    assert entry["metrics"]["median"]["session_mse"] == 20.5
    assert entry["delta_session_mae"]["parent"] == 3.5
    assert entry["delta_session_mae"]["zero"] == 0.0
    assert result["final_test_opened"] is False


def test_cache_skips_unchanged_predictions_and_detects_replacement(tmp_path):
    source, _, prediction = fixture_run(tmp_path / "source")
    first = review(tmp_path, source)
    second = review(tmp_path, source)
    assert first["processed_jobs"] == 1
    assert second["processed_jobs"] == second["hashed_bytes"] == 0
    prediction.write_bytes(b"corrupted")
    third = review(tmp_path, source)
    assert third["counts"]["failed"] == 1
    assert "huella" in third["entries"]["study/case/validation"]["error"]


@pytest.mark.parametrize("year", [2022, 2024])
def test_rejects_other_partition_dates(tmp_path, year):
    source, _, _ = fixture_run(tmp_path / "source", year=year)
    result = review(tmp_path, source)
    assert result["counts"]["failed"] == 1
    assert "temporal" in result["entries"]["study/case/validation"]["error"]


@pytest.mark.parametrize("values", [[float("nan"), 0.0, 0.0], [1.0, 1.0, 1.0]])
def test_rejects_nonfinite_or_inconsistent_published_metrics(tmp_path, values):
    source, _, _ = fixture_run(tmp_path / "source", predictions=values)
    result = review(tmp_path, source)
    assert result["counts"]["verified"] == 0
    assert result["counts"]["failed"] == 1


def test_report_hash_must_match_confirmed_summary(tmp_path):
    source, report, _ = fixture_run(tmp_path / "source")
    report.write_text(report.read_text() + " ")
    result = review(tmp_path, source)
    assert result["counts"]["verified"] == 0
    assert result["source_errors"]


def test_budget_leaves_pending_jobs_and_resumes_without_duplicate_work(tmp_path):
    source, _, _ = fixture_run(tmp_path / "source")
    other, _, _ = fixture_run(tmp_path / "other")
    sources = [{"id": "a", "summary": str(source)}, {"id": "b", "summary": str(other)}]
    state = tmp_path / "review.json"
    first = reviewer().review_campaigns(sources, state, partitions=("validation",), max_jobs=1)
    assert first["counts"] == dict(confirmed=2, verified=1, failed=0, pending=1)
    second = reviewer().review_campaigns(sources, state, partitions=("validation",), max_jobs=1)
    assert second["counts"] == dict(confirmed=2, verified=2, failed=0, pending=0)
    assert second["processed_jobs"] == 1


def test_does_not_open_unconfirmed_run_or_test_partition(tmp_path):
    source, _, prediction = fixture_run(tmp_path / "source")
    data = json.loads(source.read_text())
    data["runs"][0]["status"] = "running"
    atomic_json(source, data)
    prediction.unlink()
    result = review(tmp_path, source)
    assert result["counts"]["confirmed"] == result["processed_jobs"] == 0
    with pytest.raises(ValueError, match="particiones"):
        reviewer().review_campaigns([], tmp_path / "state.json", partitions=("test",))


@pytest.mark.parametrize("path", ["../outside", "/tmp/outside"])
def test_rejects_run_paths_outside_campaign(tmp_path, path):
    source, _, _ = fixture_run(tmp_path / "source")
    data = json.loads(source.read_text())
    data["runs"][0]["path"] = path
    atomic_json(source, data)
    result = review(tmp_path, source)
    assert result["source_errors"]
    assert result["processed_jobs"] == 0


def test_rejects_file_above_budget_before_hashing(tmp_path):
    source, _, _ = fixture_run(tmp_path / "source")
    result = review(tmp_path, source, max_file_bytes=1)
    assert result["counts"]["failed"] == 1
    assert result["hashed_bytes"] == 0


def test_accepts_confirmed_tabular_attempt(tmp_path):
    source, _, _ = fixture_run(tmp_path / "source")
    data = json.loads(source.read_text())
    run = data["runs"][0]
    run["attempts"] = [dict(path=run.pop("path"), status="completed")]
    atomic_json(source, data)
    assert review(tmp_path, source)["counts"]["verified"] == 1


def test_missing_future_parent_summary_is_explicit_and_does_not_erase_results(tmp_path):
    source, _, _ = fixture_run(tmp_path / "source")
    sources = [
        {"id": "ready", "summary": str(source)},
        {"id": "future", "summary": str(tmp_path / "future" / "missing.json")},
    ]
    result = reviewer().review_campaigns(
        sources, tmp_path / "state.json", partitions=("validation",)
    )
    assert result["counts"]["verified"] == 1
    assert result["unavailable_sources"] == ["future"]


def test_existing_state_survives_invalid_source_configuration(tmp_path):
    state = tmp_path / "state.json"
    atomic_json(state, {"sentinel": 42})
    with pytest.raises(ValueError):
        reviewer().review_campaigns([{"id": "../invalid", "summary": "none"}], state)
    assert json.loads(state.read_text()) == {"sentinel": 42}


@pytest.mark.parametrize("target", ["summary", "report", "prediction"])
def test_review_output_cannot_overwrite_source_artifacts(tmp_path, target):
    source, report, prediction = fixture_run(tmp_path / "source")
    paths = dict(summary=source, report=report, prediction=prediction)
    before = {name: path.read_bytes() for name, path in paths.items()}
    with pytest.raises(ValueError, match="salida|origen|fuente"):
        reviewer().review_campaigns(
            [{"id": "study", "summary": str(source)}],
            paths[target],
            partitions=("validation",),
        )
    assert {name: path.read_bytes() for name, path in paths.items()} == before


def test_misnamed_test_artifact_is_rejected_before_reading(tmp_path):
    source, report_path, _ = fixture_run(tmp_path / "source")
    report = json.loads(report_path.read_text())
    report["predictions"]["validation"]["path"] = "test-predictions.parquet"
    atomic_json(report_path, report)
    summary = json.loads(source.read_text())
    summary["runs"][0]["report_sha256"] = sha256(report_path)
    atomic_json(source, summary)
    result = review(tmp_path, source)
    assert result["processed_jobs"] == 0
    assert result["source_errors"]


def test_prediction_symlink_is_rejected_before_hashing(tmp_path):
    source, _, prediction = fixture_run(tmp_path / "source")
    renamed = prediction.with_name("test-predictions.parquet")
    prediction.rename(renamed)
    prediction.symlink_to(renamed.name)
    result = review(tmp_path, source)
    assert result["processed_jobs"] == 0
    assert result["source_errors"]


def test_cli_reports_real_metrics_and_failure_exit_code(tmp_path):
    source, _, prediction = fixture_run(tmp_path / "source")
    config, state = tmp_path / "config.json", tmp_path / "state.json"
    atomic_json(config, dict(sources=[dict(id="study", summary=str(source))]))
    command = [
        sys.executable,
        "-m",
        "mars_titan.evaluation.prediction_review",
        "--config",
        str(config),
        "--state",
        str(state),
        "--partition",
        "validation",
    ]
    complete = subprocess.run(command, text=True, capture_output=True, check=False)
    assert complete.returncode == 0, complete.stderr
    assert json.loads(complete.stdout)["counts"]["verified"] == 1
    prediction.write_bytes(b"broken")
    failed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["counts"]["failed"] == 1


def test_corrupt_prior_state_is_not_silently_discarded(tmp_path):
    source, _, _ = fixture_run(tmp_path / "source")
    state = tmp_path / "review.json"
    state.write_text("corrupt state")
    with pytest.raises(ValueError):
        review(tmp_path, source)
    assert state.read_text() == "corrupt state"


def test_increased_budget_retries_previously_rejected_file(tmp_path):
    source, _, _ = fixture_run(tmp_path / "source")
    assert review(tmp_path, source, max_file_bytes=1)["counts"]["failed"] == 1
    assert review(tmp_path, source)["counts"]["verified"] == 1


@pytest.mark.parametrize("where", ["report", "summary"])
def test_rejects_declared_population_different_from_prediction_count(tmp_path, where):
    source, report_path, _ = fixture_run(tmp_path / "source")
    report, summary = json.loads(report_path.read_text()), json.loads(source.read_text())
    if where == "report":
        report["samples"] = dict(validation=999)
        atomic_json(report_path, report)
        summary["runs"][0]["report_sha256"] = sha256(report_path)
    else:
        summary["counts"] = dict(validation=999)
    atomic_json(source, summary)
    result = review(tmp_path, source)
    assert result["counts"]["verified"] == 0
    assert result["source_errors"]


def test_train_and_validation_are_reviewed_as_separate_jobs(tmp_path):
    source, report_path, prediction = fixture_run(tmp_path / "source")
    table = pq.read_table(prediction)
    train_path = prediction.with_name("train-predictions.parquet")
    times = pa.array(
        [datetime(2022, 12, day, tzinfo=UTC) for day in (28, 28, 29)],
        type=pa.timestamp("us", tz="UTC"),
    )
    table = table.set_column(table.schema.get_field_index("prediction_at"), "prediction_at", times)
    pq.write_table(table, train_path)
    report = json.loads(report_path.read_text())
    report["predictions"]["train"] = dict(
        report["predictions"]["validation"], path=train_path.name, sha256=sha256(train_path)
    )
    atomic_json(report_path, report)
    summary = json.loads(source.read_text())
    summary["runs"][0]["report_sha256"] = sha256(report_path)
    atomic_json(source, summary)
    result = reviewer().review_campaigns(
        [dict(id="study", summary=str(source))], tmp_path / "state.json"
    )
    assert result["counts"]["verified"] == 2
    assert result["entries"]["study/case/train"]["last_prediction_at"].startswith("2022-12-29")
    assert result["entries"]["study/case/validation"]["last_prediction_at"].startswith("2023-01-03")


@pytest.mark.parametrize("defect", ["null", "empty", "missing", "timezone", "nonzero_control"])
def test_invalid_prediction_table_is_not_counted_as_verified(tmp_path, defect):
    source, report_path, prediction = fixture_run(tmp_path / "source")
    table = pq.read_table(prediction)
    if defect == "null":
        table = table.set_column(
            table.schema.get_field_index("target"), "target", pa.array([1.0, None, 6.0])
        )
    elif defect == "empty":
        table = table.slice(0, 0)
    elif defect == "missing":
        table = table.drop(["target"])
    elif defect == "timezone":
        table = table.set_column(
            table.schema.get_field_index("prediction_at"),
            "prediction_at",
            table["prediction_at"].cast(pa.timestamp("us")),
        )
    else:
        table = table.set_column(table.schema.get_field_index("zero"), "zero", pa.array([1.0] * 3))
    pq.write_table(table, prediction)
    report = json.loads(report_path.read_text())
    report["predictions"]["validation"]["sha256"] = sha256(prediction)
    atomic_json(report_path, report)
    summary = json.loads(source.read_text())
    summary["runs"][0]["report_sha256"] = sha256(report_path)
    atomic_json(source, summary)
    result = review(tmp_path, source)
    assert result["counts"]["failed"] == 1


def test_cli_main_propagates_real_status_and_does_not_print_entries(tmp_path, capsys):
    source, _, prediction = fixture_run(tmp_path / "source")
    config, state = tmp_path / "config.json", tmp_path / "state.json"
    atomic_json(config, dict(sources=[dict(id="study", summary=str(source))]))
    args = ["--config", str(config), "--state", str(state), "--partition", "validation"]
    assert reviewer().main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["counts"]["verified"] == 1
    assert "entries" not in output
    prediction.write_bytes(b"bad")
    assert reviewer().main(args) == 1
