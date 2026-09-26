"""Contrato público de campañas, sin leer pesos ni reconstruir fechas."""

import json
from pathlib import Path

import pytest

from mars_titan.observatory.collector import (
    Collector,
    planned_runs,
    validation_metrics,
    write_pages,
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source(tmp_path):
    folder = tmp_path / "private" / "campaign"
    report = {
        "status": "completed",
        "identity": {"case": {"kind": "gru", "seed": 42, "epochs": 2}, "manifest_sha256": "a" * 64},
        "samples": {"train": 30, "validation": 10},
        "epochs": [{"epoch": 1, "validation": {"mae": 0.2}}],
        "global_step": 5,
        "predictions": {
            "validation": {"metrics": {"mae": 0.1, "mse": 0.02}},
            "test": {"metrics": {"mae": 999}},
        },
        "secret": "/home/private/key",
    }
    dump(folder / "runs" / "one" / "run.json", report)
    dump(
        folder / "summary.json",
        {"runs": [{"id": "one", "path": "runs/one", "status": "completed"}]},
    )
    return {"id": "campaign", "path": "campaign", "kind": "archive", "domain": "real"}


def test_counts_follow_existing_configurations():
    root = Path(__file__).parents[2]

    def load(name):
        return json.loads((root / "configs/baselines" / name).read_text())

    assert planned_runs("neural", load("scientific-search-us.json")) == 80
    assert planned_runs("tabular", load("tabular-search-us.json")) == 17
    assert planned_runs("adaptation", load("klpo-adaptation.json"), parents=6) == 216


def test_incremental_collection_redacts_and_preserves_undated_epochs(tmp_path):
    def numeric_values(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from numeric_values(child)
        elif isinstance(value, list):
            for child in value:
                yield from numeric_values(child)
        elif type(value) in (int, float):
            yield value

    entry = source(tmp_path)
    report_path = tmp_path / "private/campaign/runs/one/run.json"
    report = json.loads(report_path.read_text())
    report["updated_at_utc"] = "2026-01-01T00:00:00Z"
    dump(report_path, report)
    with Collector(tmp_path / "private", tmp_path / "cache.sqlite") as collector:
        snapshot = collector.collect([entry], now="2026-01-02T00:00:00.999999Z")
        assert collector.bytes_read > 0
        run = snapshot["runs"][0]
        assert run["metrics"]["mae"] == 0.1
        assert run["history"][0]["recorded_at"] is None
        assert run["heartbeat_at"] is None
        assert run["metadata"]["train_rows"] == 30
        assert "private" not in json.dumps(snapshot)
        assert 999 not in numeric_values(snapshot)
        assert "predictions" not in run
        collector.collect([entry])
        assert collector.bytes_read == 0
    with Collector(tmp_path / "private", tmp_path / "cache.sqlite") as recovered:
        assert recovered.collect([entry])["runs"] == snapshot["runs"]
        assert recovered.bytes_read == 0


def test_corruption_keeps_last_valid_publication(tmp_path):
    entry = source(tmp_path)
    output = tmp_path / "public"
    with Collector(tmp_path / "private", tmp_path / "cache.sqlite") as collector:
        write_pages(collector.collect([entry]), output)
        previous = (output / "observatory.json").read_bytes()
        (tmp_path / "private/campaign/summary.json").write_text("{")
        with pytest.raises(ValueError):
            write_pages(collector.collect([entry]), output)
        assert (output / "observatory.json").read_bytes() == previous


def test_pagination_keeps_every_attempt_without_initial_full_download(tmp_path):
    entry = source(tmp_path)
    with Collector(tmp_path / "private", tmp_path / "cache.sqlite") as collector:
        snapshot = collector.collect([entry])
    prototype = snapshot["runs"][0]
    snapshot["runs"] = [{**prototype, "run_id": f"r{i}"} for i in range(313)]
    index = write_pages(snapshot, tmp_path / "public", page_size=32)
    assert len(index["runs"]) == 32
    assert index["pagination"]["total_runs"] == 313
    runs = list(index["runs"])
    for page in index["pagination"]["pages"]:
        runs.extend(json.loads((tmp_path / "public" / page).read_text())["runs"])
    assert len({run["run_id"] for run in runs}) == 313


def test_symlink_source_is_rejected(tmp_path):
    entry = source(tmp_path)
    (tmp_path / "private/link").symlink_to(tmp_path / "private/campaign", target_is_directory=True)
    with Collector(tmp_path / "private", tmp_path / "cache.sqlite") as collector:
        with pytest.raises(ValueError):
            collector.collect([{**entry, "path": "link"}])


def test_only_a_held_campaign_lock_confirms_a_running_process(tmp_path):
    import fcntl

    entry = source(tmp_path)
    report = tmp_path / "private/campaign/runs/one/run.json"
    raw = json.loads(report.read_text())
    raw["status"] = "running"
    dump(report, raw)
    with Collector(tmp_path / "private", tmp_path / "cache.sqlite") as collector:
        assert collector.collect([entry])["runs"][0]["heartbeat_at"] is None
        with (report.parents[2] / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            observed = collector.collect([entry])["runs"][0]
            assert observed["heartbeat_at"] is not None
            assert observed["updated_at"] != observed["heartbeat_at"]


def test_tabular_failed_attempts_remain_in_history(tmp_path):
    entry = source(tmp_path)
    dump(
        tmp_path / "private/campaign/summary.json",
        {
            "runs": [
                {
                    "id": "ridge",
                    "kind": "ridge",
                    "status": "completed",
                    "attempts": [
                        {
                            "path": "runs/ridge/attempt-0001",
                            "status": "failed",
                            "error_type": "ValueError",
                        },
                        {"path": "runs/ridge/attempt-0002", "status": "completed"},
                    ],
                }
            ]
        },
    )
    with Collector(tmp_path / "private", tmp_path / "cache.sqlite") as collector:
        runs = collector.collect([entry])["runs"]
        assert {r["attempt_id"] for r in runs} == {"legacy", "attempt-0001", "attempt-0002"}
        assert next(r for r in runs if r["attempt_id"] == "attempt-0001")["status"] == "failed"


def test_adaptation_median_and_tabular_receipts_keep_their_metrics(tmp_path):
    entry = source(tmp_path)
    folder = tmp_path / "private/campaign/runs"
    dump(
        folder / "adapt/run.json",
        {
            "status": "completed",
            "identity": {"case": {"mode": "klpo_full"}},
            "predictions": {"validation": {"metrics": {"median": {"mae": 0.12}}}},
            "epochs": [{"epoch": 1, "validation": {"median": {"mae": 0.14}}}],
        },
    )
    dump(
        folder / "tabular/run.json",
        {
            "status": "completed",
            "model": "ridge",
            "manifest_sha256": "a" * 64,
            "predictions": {"validation": {"metrics": {"mae": 0.1}}},
        },
    )
    with Collector(tmp_path / "private", tmp_path / "cache.sqlite") as collector:
        runs = collector.collect([entry])["runs"]
    adapted = next(run for run in runs if run["model_id"] == "adaptation")
    assert adapted["metrics"]["mae"] == 0.12
    assert adapted["history"][0]["mae"] == 0.14
    tabular = next(run for run in runs if run["model_id"] == "ridge")
    assert tabular["comparison_group"] is not None


def test_missing_receipt_preserves_history(tmp_path):
    entry = source(tmp_path)
    with Collector(tmp_path / "private", tmp_path / "cache.sqlite") as collector:
        previous = collector.collect([entry])["runs"]
        (tmp_path / "private/campaign/runs/one/run.json").rename(tmp_path / "retired.json")
        assert collector.collect([entry])["runs"] == previous


@pytest.mark.parametrize(
    "changes",
    [
        {"global_step": 2.5},
        {"predictions": []},
        {"epochs": [{"epoch": 1.5}]},
        {"samples": {"train": 30.5}},
        {"epochs": [{"epoch": 6}]},
    ],
)
def test_invalid_receipt_does_not_replace_the_cache(tmp_path, changes):
    entry = source(tmp_path)
    path = tmp_path / "private/campaign/runs/one/run.json"
    with Collector(tmp_path / "private", tmp_path / "cache.sqlite") as collector:
        collector.collect([entry])
        dump(path, {**json.loads(path.read_text()), **changes})
        with pytest.raises(ValueError):
            collector.collect([entry])


def test_legacy_diagnostic_curve_keeps_observed_loss(tmp_path):
    entry = source(tmp_path)
    dump(
        tmp_path / "private/campaign/runs/old/run.json",
        {
            "status": "completed",
            "model": "rnn",
            "epochs": [
                {
                    "epoch": 1,
                    "validation": {
                        "diagnostic_mae": 0.1,
                        "diagnostic_mse": 0.03,
                        "objective_loss": 0.03,
                    },
                }
            ],
        },
    )
    with Collector(tmp_path / "private", tmp_path / "cache.sqlite") as collector:
        runs = collector.collect([entry])["runs"]
    run = next(r for r in runs if r["model_id"] == "rnn")
    assert run["history"][0]["mae"] == 0.1
    assert run["history"][0]["loss"] == 0.03


def test_empty_existing_receipt_is_corruption(tmp_path):
    entry = source(tmp_path)
    with Collector(tmp_path / "private", tmp_path / "cache.sqlite") as collector:
        collector.collect([entry])
        dump(tmp_path / "private/campaign/runs/one/run.json", {})
        with pytest.raises(ValueError):
            collector.collect([entry])


def test_metric_ranges_match_browser_and_preserve_signed_values():
    metrics = validation_metrics({"rank_ic": -0.5, "objective_loss": -2, "coverage_95": 0.9}, None)
    assert metrics["rank_ic"] == -0.5
    assert metrics["loss"] == -2
    for values in ({"rank_ic": 2}, {"coverage_95": 2}, {"mae": -1}, {"mae": "x"}):
        with pytest.raises(ValueError):
            validation_metrics(values, None)


@pytest.mark.parametrize("planned", [2.5, "2", True, -1])
def test_invalid_archive_plan_is_rejected(tmp_path, planned):
    entry = source(tmp_path)
    path = tmp_path / "private/campaign/summary.json"
    dump(path, {**json.loads(path.read_text()), "planned_runs": planned})
    with Collector(tmp_path / "private", tmp_path / "cache.sqlite") as collector:
        with pytest.raises(ValueError):
            collector.collect([entry])


@pytest.mark.parametrize("status", ["running", "failed", "paused", "completed"])
def test_campaign_without_runs_preserves_explicit_state_despite_dependencies(tmp_path, status):
    dump(
        tmp_path / "private/adaptation/summary.json",
        {"status": status, "parents": {}, "planned_runs": 216, "completed_runs": 0},
    )
    source = {
        "id": "adaptation",
        "path": "adaptation",
        "kind": "adaptation",
        "domain": "real",
        "dependencies": ["neural", "tabular"],
    }
    with Collector(tmp_path / "private", tmp_path / "cache.sqlite") as collector:
        campaign = collector.collect([source])["campaigns"][0]
    assert campaign["status"] == status
    assert campaign["registered_runs"] == 0
    assert campaign["counts"]["not_started"] == 216


def test_unstarted_campaign_with_dependencies_remains_blocked(tmp_path):
    dump(tmp_path / "private/adaptation/summary.json", {"parents": {}, "planned_runs": 216})
    source = {
        "id": "adaptation",
        "path": "adaptation",
        "kind": "adaptation",
        "domain": "real",
        "dependencies": ["neural", "tabular"],
    }
    with Collector(tmp_path / "private", tmp_path / "cache.sqlite") as collector:
        campaign = collector.collect([source])["campaigns"][0]
    assert campaign["status"] == "blocked"
