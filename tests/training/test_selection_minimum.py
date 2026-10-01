"""Paciencia posterior al mínimo y selección recuperable sin usar CUDA."""

import json

import pytest

from mars_titan.training.selection import advance_selection, initial_selection, validate_selection

POLICY = dict(metric="session_mae", patience=2, min_delta=0.01, minimum_epochs=3)


def test_minimum_keeps_early_improvement_then_counts_only_later_epochs():
    state = initial_selection(0.5, POLICY)
    for epoch, score in enumerate((0.4, 0.6, 0.7, 0.6, 0.6), 1):
        state = advance_selection(state, score, epoch, POLICY)
        assert state["stale_epochs"] == max(0, epoch - 3)
        assert state["should_stop"] is (epoch == 5)
    assert state["best_epoch"] == 1


def test_recovery_after_initial_decline_and_later_improvement_reset_patience():
    state = initial_selection(0.5, POLICY)
    for epoch, score in enumerate((0.8, 0.7, 0.6, 0.4, 0.5, 0.3, 0.4, 0.4), 1):
        state = advance_selection(json.loads(json.dumps(state)), score, epoch, POLICY)
        assert state["should_stop"] is (epoch == 8)
    assert state["best_epoch"] == 6


def test_minimum_can_keep_epoch_zero():
    state = initial_selection(0.1, POLICY)
    for epoch in range(1, 6):
        state = advance_selection(state, 0.2, epoch, POLICY)
    assert state["best_epoch"] == 0
    assert state["best_score"] == 0.1


@pytest.mark.parametrize("minimum", [True, -1, 1.0, float("nan"), float("inf"), 1000])
def test_minimum_rejects_ambiguous_or_unbounded_values(minimum):
    with pytest.raises(ValueError):
        validate_selection(POLICY | {"minimum_epochs": minimum})


def test_minimum_must_leave_at_least_one_later_evaluation():
    with pytest.raises(ValueError):
        validate_selection(POLICY, epochs=3)
    validate_selection(POLICY, epochs=4)


def test_old_policy_and_serialized_state_are_unchanged():
    policy = {k: v for k, v in POLICY.items() if k != "minimum_epochs"}
    first = advance_selection(None, 0.5, 1, policy)
    assert first == dict(
        last_epoch=1,
        best_epoch=1,
        best_score=0.5,
        stale_epochs=0,
        should_stop=False,
        last_improved=True,
    )
    second = advance_selection(first, 0.6, 2, policy)
    assert second["stale_epochs"] == 1
    assert advance_selection(second, 0.6, 3, policy)["should_stop"]
