"""Separar actividad, procedencia y objetivos sin publicar datos privados."""

import json

import pytest

from mars_titan.observatory.collector import Collector


def collect(tmp_path, report, *, domain="synthetic", summary=None):
    folder = tmp_path / "campaign"
    folder.mkdir(exist_ok=True)
    (folder / "run.json").write_text(json.dumps(report))
    if summary is not None:
        (folder / "summary.json").write_text(json.dumps(summary))
    source = {"id": "experiment", "path": "campaign", "kind": "archive", "domain": domain}
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        return collector.collect([source])


def receipt(model="ppo", activity="rl"):
    return {
        "schema_version": 1,
        "activity": activity,
        "model": model,
        "domain": "synthetic",
        "status": "completed",
        "identity": {"case": {"kind": model, "seed": 43}, "manifest_sha256": "a" * 64},
        "samples": {"train": 1024, "validation": 64},
        "global_step": 4,
        "total_steps": 10,
        "parent_frozen": True,
        "final_test_opened": False,
        "total_seconds": 1.5,
        "predictions": {"validation": {"metrics": {"mae": 71, "loss": 72}}},
        "epochs": [{"epoch": 1, "validation": {"mae": 73, "loss": 74}}],
        "financial_validation": {
            "net_return": -0.03,
            "max_drawdown": 0.12,
            "costs": 0.002,
            "turnover": 2.5,
            "steps": 4,
            "completed": True,
            "invalid_reason": None,
            "private_path": "/home/private/operations.csv",
        },
        "test": {"net_return": 987654321},
    }


@pytest.mark.parametrize(
    "model,activity",
    [
        ("factor_world", "synthetic_generation"),
        ("ppo", "rl"),
        ("double_dqn", "rl"),
        ("simulator", "simulation"),
        ("simulator", "evaluation"),
        ("cash", "evaluation"),
        ("hold_initial", "evaluation"),
        ("rebalance_50", "evaluation"),
        ("financial_comparison", "rl"),
    ],
)
def test_producers_keep_activity_without_predictive_errors(tmp_path, model, activity):
    snapshot = collect(tmp_path, receipt(model, activity))
    run = snapshot["runs"][0]
    assert run["model_id"] == model
    assert run["activity"] == activity
    assert run["metadata"]["domain"] == "synthetic"
    assert run["metrics"]["mae"] is None
    assert run["metrics"]["loss"] is None
    assert run["history"] == []
    assert run["metrics"]["elapsed_seconds"] == 1.5
    assert run["total_steps"] == 10
    assert run["heartbeat_at"] is None
    assert run["metadata"]["parent_frozen"] is True
    assert (
        run["financial_validation"] is None
        if activity == "synthetic_generation"
        else (run["financial_validation"]["net_return"] == -0.03)
    )
    assert "private_path" not in json.dumps(snapshot)
    assert "987654321" not in json.dumps(snapshot)


def test_generation_imports_the_existing_producer_shape(tmp_path):
    report = receipt("factor_world", "synthetic_generation")
    for key in ("epochs", "global_step", "total_steps", "financial_validation", "predictions"):
        report.pop(key)
    report["samples"] = {"train": 1024}
    run = collect(tmp_path, report)["runs"][0]
    assert run["metadata"]["method"] == "synthetic_generation"
    assert run["completed_steps"] is None
    assert run["epoch"] is None
    assert run["seed"] == 43


def test_new_model_cannot_be_inferred_as_predictive_training(tmp_path):
    report = receipt()
    report.pop("activity")
    with pytest.raises(ValueError):
        collect(tmp_path, report)


def test_incomplete_and_ruined_financial_episodes_keep_their_meaning(tmp_path):
    report = receipt()
    report["financial_validation"].update(
        net_return=None, max_drawdown=None, completed=False, invalid_reason="missing_close"
    )
    financial = collect(tmp_path, report)["runs"][0]["financial_validation"]
    assert financial["net_return"] is None
    assert financial["completed"] is False
    report["financial_validation"].update(
        net_return=-1, max_drawdown=1, completed=True, invalid_reason="ruined"
    )
    financial = collect(tmp_path, report)["runs"][0]["financial_validation"]
    assert financial["net_return"] == -1
    assert financial["completed"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"final_test_opened": True},
        {"phase": "test"},
        {"phase": "evaluation"},
        {"partition": "test"},
    ],
)
def test_financial_results_remain_closed_for_test(tmp_path, changes):
    report = {**receipt(), **changes}
    run = collect(tmp_path, report)["runs"][0]
    assert run["financial_validation"] is None
    assert run["test_released"] is False


def test_comparison_separates_activities_and_objectives(tmp_path):
    groups = []
    for index, (activity, objective) in enumerate(
        [
            ("initial_training", "row_mae"),
            ("supervised_continuation", "row_mae"),
            ("predictive_adaptation", "row_mae"),
            ("rl", "net_return"),
            ("rl", "risk_adjusted"),
            ("simulation", "net_return"),
        ]
    ):
        folder = tmp_path / str(index)
        folder.mkdir()
        report = receipt("gru" if index < 3 else "ppo", activity)
        report["identity"]["objective"] = objective
        groups.append(collect(folder, report)["runs"][0]["comparison_group"])
    assert None not in groups
    assert len(set(groups)) == 6


@pytest.mark.parametrize(
    "field,value",
    [
        ("domain", "real"),
        ("activity", "unknown"),
        ("total_steps", 3),
    ],
)
def test_new_producer_rejects_inconsistent_contract(tmp_path, field, value):
    with pytest.raises(ValueError):
        collect(tmp_path, {**receipt(), field: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"net_return": True},
        {"max_drawdown": 1.5},
        {"costs": -1},
        {"steps": 2.5},
        {"completed": "yes"},
        {"invalid_reason": "/home/private/error"},
    ],
)
def test_financial_summary_rejects_invalid_values(tmp_path, changes):
    report = receipt()
    report["financial_validation"].update(changes)
    with pytest.raises(ValueError):
        collect(tmp_path, report)


def test_checkpoint_zero_and_explicit_attempt_survive(tmp_path):
    report = receipt()
    report.update(global_step=0, attempt_id="resume-02", status="blocked", epochs=[])
    report["checkpoint"] = {
        "step": 0,
        "saved_at": "2026-01-01T00:00:00Z",
        "resumable": True,
        "path": "/home/private/state.pt",
    }
    run = collect(tmp_path, report)["runs"][0]
    assert run["completed_steps"] == 0
    assert run["checkpoint"] == {"step": 0, "saved_at": "2026-01-01T00:00:00Z", "resumable": True}
    assert run["attempt_id"] == "resume-02"
    assert run["status"] == "blocked"


def test_financial_comparison_uses_tape_and_execution_conditions(tmp_path):
    groups = []
    for index, (currency, cost) in enumerate([("USD", 0), ("USD", 10), ("CNY", 10)]):
        folder = tmp_path / str(index)
        folder.mkdir()
        report = receipt("cash", "evaluation")
        report["identity"] = {"tape_sha256": "a" * 64}
        report.update(currency=currency, cost_bps=cost, partition="validation")
        run = collect(folder, report)["runs"][0]
        groups.append(run["comparison_group"])
        assert run["metadata"]["currency"] == currency
    assert None not in groups
    assert len(set(groups)) == 3


def test_rl_producer_keeps_nested_configuration_and_original_dates(tmp_path):
    report = receipt()
    report["identity"] = {"config": {"batch_size": 4}, "environment": {"tape_sha256": "a" * 64}}
    report.update(started_at="2026-01-01T01:00:00+00:00", updated_at="2026-01-01T01:01:00+00:00")
    run = collect(tmp_path, report)["runs"][0]
    assert run["started_at"] == "2026-01-01T01:00:00Z"
    assert run["updated_at"] == "2026-01-01T01:01:00Z"
    assert run["metadata"]["configuration_sha256"] is not None
    assert run["comparison_group"] is not None
