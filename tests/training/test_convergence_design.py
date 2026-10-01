"""Edición de referencias con mínimos, controles fijos y cuatro ventanas."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mars_titan.posttraining.queue import read_design
from mars_titan.training.reference_design import design_cases
from mars_titan.training.reference_search import _configuration, _execute_design
from mars_titan.training.temporal_search import _inputs
from tests.training.test_temporal_search import metadata_views

REFERENCE = Path("configs/baselines/convergence-temporal-search-us.json")
POSTTRAINING = Path("configs/baselines/real-continuations-v3.json")


def test_new_reference_design_expands_only_training_budget():
    old, old_cases, _ = _configuration(Path("configs/baselines/strict-temporal-search-us.json"))
    new, cases, _ = _configuration(REFERENCE)
    assert new["schema_version"] == 3
    assert new["max_epochs"] == 100
    assert new["minimum_epochs"] == new["patience"] == 10
    assert new["min_delta"] == 1e-5
    for key in ("models", "case_indices", "finalist_seeds", "continuation_selection"):
        assert new[key] == old[key]
    for old_case, case in zip(old_cases, cases, strict=True):
        assert case["id"] == old_case["id"]
        assert case["case"]["selection"]["minimum_epochs"] == 10
        assert {k: v for k, v in case["case"].items() if k not in {"epochs", "selection"}} == {
            k: v for k, v in old_case["case"].items() if k not in {"epochs", "selection"}
        }


def test_temporal_convergence_keeps_the_four_closed_test_windows(tmp_path):
    records, identity, count = _inputs(REFERENCE, metadata_views(tmp_path))
    assert len(records) == 4 and count == 40
    assert len(identity["manifests"]) == 4


def test_control_tasks_keep_five_epochs_and_parent_selection():
    plan, cases, _ = _configuration(REFERENCE)
    tasks = []
    study = SimpleNamespace(plan=plan, summary={}, by_id={}, visited=set(), save=lambda: None)

    def execute(task):
        tasks.append(task)
        study.by_id[task["id"]] = task
        study.visited.add(task["id"])
        return dict(task, session_mae=0.1)

    study.execute = execute
    _execute_design(study, cases)
    controls = [task for task in tasks if task["stage"] == "posttraining"]
    assert len(controls) == 24
    assert {task["case"]["loss"] for task in controls} == {"mae", "mse"}
    for task in controls:
        assert task["case"]["epochs"] == 5
        assert task["case"]["selection"] == plan["continuation_selection"]
        assert "minimum_epochs" not in task["case"]["selection"]
        assert task["parent"] in study.by_id


def test_real_continuations_include_six_variants_and_two_neural_controls():
    plan, cases, _ = read_design(POSTTRAINING)
    assert plan["conditions"] == ["real"]
    assert plan["epochs"] == 50
    assert plan["selection"] == dict(
        version=3, metric="session_mae", minimum_epochs=5, patience=8, min_delta=1e-5
    )
    assert len(cases("gru")) == 24 and len(cases("ridge")) == 18


@pytest.mark.parametrize(
    "change",
    [
        {"minimum_epochs": 100},
        {"minimum_epochs": True},
        {"max_epochs": 101},
        {"schema_version": 2},
        {"posttraining_epochs": 6},
    ],
)
def test_new_reference_contract_rejects_invalid_budget(tmp_path, change):
    plan = json.loads(REFERENCE.read_text()) | change
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        _configuration(path)


def test_legacy_design_does_not_silently_expand_its_budget():
    with pytest.raises(ValueError):
        design_cases(["gru"], epochs=31)
