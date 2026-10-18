"""Límites y selección del microensayo, sin abrir datos reales ni ajustar parámetros."""

import copy
import json
from pathlib import Path

import pytest

from benchmarks.memory_microstudy import (
    Budget,
    _execute,
    configuration,
    jobs,
    prepare,
    project_view,
    run_job,
)
from mars_titan.training.learning_hold import LearningHoldError


def manifest():
    return dict(
        final_test_opened=False,
        input_policy="historical_masked_2000_v1",
        temporal_view=dict(
            protocol=dict(
                market="US",
                train_start="2000-01-01",
                final_test_start="2024-01-01",
                selection=dict(max_epochs=30),
            ),
            fold=dict(id="fold-000"),
        ),
        assets=[
            dict(
                market="US",
                symbol=name,
                samples=10,
                counts=dict(train=count, validation=2, calibration=2, evaluation=2),
            )
            for name, count in (("A", 4), ("B", 4), ("C", 0), ("D", 4), ("E", 4), ("F", 4))
        ],
        coverage=[],
    )


def test_selection_uses_only_training_availability_and_is_independent_of_asset_order():
    original = manifest()
    first = project_view(original)
    changed = copy.deepcopy(original)
    changed["assets"].reverse()
    for asset in changed["assets"]:
        asset["counts"]["evaluation"] = 999
    second = project_view(changed)
    assert [a["symbol"] for a in first["assets"]] == [a["symbol"] for a in second["assets"]]
    assert len(first["assets"]) == 4 and all(a["symbol"] != "C" for a in first["assets"])
    assert first["counts"]["train"] == 16
    assert first["scope"] == "development_snapshot" and first["cohort_complete"] is False
    assert original == manifest()


@pytest.mark.parametrize("change", ["test", "start", "window", "insufficient"])
def test_projection_rejects_a_different_scope(change):
    value = manifest()
    if change == "test":
        value["final_test_opened"] = True
    elif change == "start":
        value["temporal_view"]["protocol"]["train_start"] = "2010-01-01"
    elif change == "window":
        value["temporal_view"]["fold"]["id"] = "fold-018"
    else:
        value["assets"] = value["assets"][:2]
    with pytest.raises(ValueError):
        project_view(value)


def test_schedule_keeps_three_paired_pt1_seeds_and_separate_factorial_parents():
    schedule = jobs()
    assert len({job["id"] for job in schedule}) == len(schedule)
    for seed in (42, 43, 44):
        assert {f"{name}-{seed}" for name in ("mac_online", "pt1", "pt1_init")} <= {
            job["id"] for job in schedule
        }
    by_id = {job["id"]: job for job in schedule}
    assert by_id["cm_v1_bm-42"]["parent"] == "cm_v1_core_b-42"
    assert by_id["cm_v1_bcm-42"]["parent"] == "cm_v1_core_c-42"
    assert by_id["post-head+readout+fusion-42"]["parent"] == "mac_online-42"


def test_global_hold_is_checked_before_opening_a_work_directory(monkeypatch, tmp_path):
    hold = tmp_path / "hold.json"
    hold.write_text(json.dumps(dict(training_allowed=False)))
    monkeypatch.setenv("MARS_TITAN_TRAINING_HOLD", str(hold))
    with pytest.raises(LearningHoldError):
        run_job(tmp_path / "nonexistent", "mac_online-42")


def test_step_limit_rejects_the_next_update_before_touching_cuda():
    budget = Budget(configuration())
    budget.steps = configuration()["max_optimizer_steps"]
    with pytest.raises(RuntimeError, match="presupuesto"):
        budget.before(None, (), {})


def test_all_declared_jobs_reach_existing_entrypoints_with_valid_recipes(tmp_path, monkeypatch):
    from mars_titan.evaluation.splits import build_folds
    from mars_titan.posttraining import chronological_windows
    from mars_titan.training import (
        candidate_walk_forward,
        cm_v1_factorial,
        mars_titan_walk_forward,
        titans_walk_forward,
    )

    source = tmp_path / "source" / "manifest.json"
    source.parent.mkdir()
    value = manifest()
    protocol = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "configs/evaluation/historical-masked-us-walk-forward-v2.json"
        ).read_text()
    )
    value["temporal_view"].update(protocol=protocol, fold=build_folds(protocol)[0])
    source.write_text(json.dumps(value))
    work = tmp_path / "study"
    prepared = prepare(source, work)
    called = []

    def record(*args, **kwargs):
        called.append(kwargs)
        return dict(status="prepared")

    for module, name in (
        (titans_walk_forward, "run_titans_window"),
        (candidate_walk_forward, "fit_window"),
        (cm_v1_factorial, "run_cm_v1_core_window"),
        (cm_v1_factorial, "run_cm_v1_window"),
        (mars_titan_walk_forward, "run_mars_titan_window"),
        (chronological_windows, "run_titans_posttraining"),
    ):
        monkeypatch.setattr(module, name, record)
    for job in prepared["jobs"]:
        assert _execute(work, job, Budget(configuration()))["status"] == "prepared"
    assert len(called) == 28 and all(call["device"] == "cuda:0" for call in called)
