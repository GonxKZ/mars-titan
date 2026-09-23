"""Cohortes simultáneas, recompensas maduras y recuperación sin etiquetas anticipadas."""

import copy
import importlib
import json

import numpy as np
import pytest

from mars_titan.environments.actions import ActionGrid

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("gymnasium") is None,
    reason="Requiere el extra reinforcement",
)

SHAPES = dict(prices=(2, 5), news=(3,), charts=(2,), fundamentals=(4,), macro=(2,))
SOURCE = "a" * 64


def module():
    return importlib.import_module("mars_titan.environments.prediction")


def cohort(at, ids=("US/B", "US/A"), *, maturity=30, target=0.02):
    count = len(ids)
    return dict(
        prediction_at=at,
        asset_ids=list(ids),
        available_at=np.full(count, at),
        target_available_at=np.full(count, maturity),
        target=np.full(count, target),
        inputs={name: np.ones((count, *shape), dtype=np.float32) for name, shape in SHAPES.items()},
    )


def environment(rows=None, **options):
    rows = (
        rows if rows is not None else [cohort(10), cohort(20, maturity=40), cohort(30, maturity=50)]
    )
    grid = ActionGrid.fit(np.linspace(-0.1, 0.1, 101), source_sha256=SOURCE, partition="train")
    return module().CausalPredictionEnv(
        lambda position: rows[position] if position < len(rows) else None,
        source_sha256=SOURCE,
        grid=grid,
        shapes=SHAPES,
        max_assets=3,
        **options,
    )


def test_gymnasium_contract_and_observations_do_not_expose_targets():
    from gymnasium.utils.env_checker import check_env

    env = environment()
    check_env(env, skip_render_check=True)
    observed, info = env.reset(seed=42)
    assert set(observed) == set(SHAPES) | {"active"}
    assert set(info) == {"prediction_at", "asset_ids", "pending"}
    assert info["asset_ids"] == ["US/A", "US/B"]
    assert observed["active"].tolist() == [1, 1, 0]
    for name in SHAPES:
        assert observed[name].dtype == np.float32
        assert np.isfinite(observed[name]).all()


def test_rewards_are_delayed_until_maturity_and_all_rows_are_credited_once():
    env = environment()
    env.reset(seed=42)
    actions = np.array([10, 10, 10])
    _, reward, done, truncated, info = env.step(actions)
    assert reward == 0 and not done and not truncated and info["matured"] == []
    assert info["prediction_at"] == 20 and info["pending"] == 2
    _, reward, done, _, second = env.step(actions)
    assert reward < 0 and not done and len(second["matured"]) == 2
    assert second["prediction_at"] == 30
    observed, _, done, truncated, third = env.step(actions)
    assert done and not truncated and observed["active"].sum() == 0
    events = second["matured"] + third["matured"]
    assert len({e["event_id"] for e in events}) == len(events) == 6
    assert all(e["target_available_at"] <= third["prediction_at"] for e in events)
    assert events == sorted(events, key=lambda e: (e["target_available_at"], e["event_id"]))
    with pytest.raises(RuntimeError):
        env.step(actions)


def test_future_changes_and_physical_row_permutations_do_not_change_observations():
    first = cohort(10)
    first["inputs"]["news"][0] *= 2
    changed = copy.deepcopy(first)
    for name in ("asset_ids", "available_at", "target_available_at", "target"):
        changed[name] = changed[name][::-1]
    changed["inputs"] = {name: value[::-1] for name, value in changed["inputs"].items()}
    changed["target"] = np.full(2, 100.0)
    one, _ = environment([first]).reset(seed=42)
    two, _ = environment([changed]).reset(seed=42)
    for name in one:
        np.testing.assert_array_equal(one[name], two[name])


def test_snapshot_restores_queue_cursor_and_action_rng_exactly():
    env = environment()
    env.reset(seed=42)
    env.action_space.seed(43)
    env.step(np.array([7, 13, 0]))
    snapshot = json.loads(json.dumps(env.snapshot()))
    actions = env.action_space.sample()
    expected = env.step(actions)
    restored = environment()
    restored.restore(snapshot)
    np.testing.assert_array_equal(restored.action_space.sample(), actions)
    result = restored.step(actions)
    for name in expected[0]:
        np.testing.assert_array_equal(result[0][name], expected[0][name])
    assert result[1:] == expected[1:]
    assert restored.snapshot() == env.snapshot()


@pytest.mark.parametrize(
    "change", ["future_input", "invalid_shape", "duplicate", "missing_modality", "target_in_past"]
)
def test_invalid_cohort_is_rejected_before_any_decision(change):
    row = cohort(10)
    if change == "future_input":
        row["available_at"][0] = 11
    if change == "invalid_shape":
        row["inputs"]["news"] = np.ones((2, 2), dtype=np.float32)
    if change == "duplicate":
        row["asset_ids"][0] = row["asset_ids"][1]
    if change == "missing_modality":
        del row["inputs"]["charts"]
    if change == "target_in_past":
        row["target_available_at"][0] = 10
    with pytest.raises(ValueError):
        environment([row]).reset(seed=42)


def test_queue_overflow_and_invalid_action_do_not_change_confirmed_state():
    env = environment(max_pending=2)
    env.reset(seed=42)
    env.step(np.array([10, 10, 0]))
    before = env.snapshot()
    with pytest.raises(ValueError, match="presupuesto"):
        env.step(np.array([10, 10, 0]))
    assert env.snapshot() == before
    with pytest.raises(ValueError):
        env.step(np.array([-1, 10, 0]))
    assert env.snapshot() == before


def test_restore_rejects_modified_current_source_and_invalid_pending_records():
    env = environment()
    env.reset(seed=42)
    env.step(np.array([10, 10, 0]))
    state = env.snapshot()
    rows = [cohort(10), cohort(20, maturity=40, target=0.03), cohort(30, maturity=50)]
    with pytest.raises(ValueError, match="fuente|huella"):
        environment(rows).restore(state)
    state["pending"][0]["target_available_at"] = 0
    with pytest.raises(ValueError):
        environment().restore(state)


def test_training_environment_does_not_accept_validation_year_or_crossing_maturity():
    boundary = 1_672_531_200_000_000
    for row in (
        cohort(boundary, maturity=boundary + 100),
        cohort(boundary - 100, maturity=boundary),
    ):
        with pytest.raises(ValueError, match="partición"):
            environment([row]).reset(seed=42)
    valid = environment([cohort(boundary, maturity=boundary + 100)], partition="validation")
    assert valid.reset(seed=42)[0]["active"].sum() == 2


def test_targets_that_overflow_float64_are_rejected_at_reset():
    row = cohort(10)
    row["target"] = np.full(2, np.finfo(np.longdouble).max, dtype=np.longdouble)
    with pytest.raises(ValueError, match="etiquetas"):
        environment([row]).reset(seed=42)


def test_observation_mutation_does_not_rewrite_the_source_or_resume_state():
    rows = [cohort(10), cohort(20, maturity=40)]
    env = environment(rows)
    observed, _ = env.reset(seed=42)
    before = env.snapshot()
    observed["news"][:] = 100
    rows[0]["inputs"]["news"][:] = 200
    assert env.snapshot() == before


@pytest.mark.parametrize(
    "part,field", [("train", "maturity"), ("validation", "prediction"), ("train", "finished_clock")]
)
def test_restore_cannot_move_pending_events_or_finished_clock_out_of_partition(part, field):
    boundary = 1_672_531_200_000_000
    offset = boundary if part == "validation" else 0
    rows = [cohort(offset + 10, maturity=offset + 30), cohort(offset + 20, maturity=offset + 40)]
    env = environment(rows, partition=part)
    env.reset(seed=42)
    env.step(np.array([10, 10, 0]))
    if field == "finished_clock":
        env.step(np.array([10, 10, 0]))
    state = env.snapshot()
    if field == "maturity":
        state["pending"][0]["target_available_at"] = boundary + 1
    if field == "prediction":
        state["pending"][0]["prediction_at"] = 10
        state["pending"][0]["event_id"] = f"{10:020d}/{state['pending'][0]['asset_id']}"
    if field == "finished_clock":
        state["clock"] = boundary + 1
    before = env.snapshot()
    with pytest.raises(ValueError):
        env.restore(state)
    assert env.snapshot() == before
