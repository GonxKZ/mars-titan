"""Presupuesto fijo con selección del mejor estado, sin entrenar ni abrir datos."""

import pytest

from mars_titan.training.selection import (
    FIXED_BUDGET,
    VALIDATION_PLATEAU,
    advance_selection,
    initial_selection,
    validate_selection,
)


def policy(**changes):
    return dict(dict(metric="session_mae", patience=2, min_delta=0.0), **changes)


def trace(scores, options):
    state, states = None, []
    for epoch, score in enumerate(scores, 1):
        if state is not None and state["should_stop"]:
            break
        state = advance_selection(state, score, epoch, options)
        states.append(state)
    return states


def test_fixed_budget_evaluates_every_epoch_and_keeps_the_same_best_state():
    scores = (0.5, 0.3, 0.31, 0.32, 0.29, 0.33)
    plateau = trace(scores, policy(stopping=VALIDATION_PLATEAU))
    fixed = trace(scores, policy(stopping=FIXED_BUDGET))
    assert len(plateau) == 4 and plateau[-1]["should_stop"]
    assert len(fixed) == len(scores)
    assert not any(state["should_stop"] for state in fixed)
    assert [s["plateau_epoch"] for s in fixed] == [None, None, None, 4, 4, 4]
    assert fixed[3]["best_epoch"] == plateau[-1]["best_epoch"] == 2
    assert fixed[-1]["best_epoch"] == 5 and fixed[-1]["best_score"] == 0.29
    assert fixed[-1]["stale_epochs"] == 1


def test_default_policy_keeps_its_previous_state_fields():
    state = advance_selection(None, 0.2, 1, policy())
    explicit = advance_selection(None, 0.2, 1, policy(stopping=VALIDATION_PLATEAU))
    assert state == explicit
    assert set(state) == {
        "last_epoch",
        "best_epoch",
        "best_score",
        "stale_epochs",
        "should_stop",
        "last_improved",
    }


def test_minimum_improvement_and_minimum_epochs_still_govern_the_selection():
    options = policy(stopping=FIXED_BUDGET, min_delta=0.01, minimum_epochs=2)
    states = trace((0.2, 0.205, 0.195, 0.194, 0.18), options)
    assert [s["best_epoch"] for s in states] == [1, 1, 1, 1, 5]
    assert [s["stale_epochs"] for s in states] == [0, 0, 1, 2, 0]
    assert states[-1]["plateau_epoch"] == 4 and not states[-1]["should_stop"]


def test_parent_can_remain_selected_without_consuming_the_budget():
    options = policy(stopping=FIXED_BUDGET, patience=1)
    state = initial_selection(0.2, options)
    assert state["plateau_epoch"] is None and state["best_epoch"] == 0
    for epoch in range(1, 4):
        state = advance_selection(state, 0.25, epoch, options)
    assert state["best_epoch"] == 0 and state["plateau_epoch"] == 1
    assert not state["should_stop"] and state["last_epoch"] == 3


@pytest.mark.parametrize("value", ["fixed", "", None, 1, ["fixed_budget"]])
def test_unknown_stopping_mode_is_rejected(value):
    with pytest.raises(ValueError, match="parada"):
        validate_selection(policy(stopping=value))


def test_a_stopped_plateau_cannot_continue():
    states = trace((0.2, 0.3, 0.3), policy(stopping=VALIDATION_PLATEAU))
    assert states[-1]["should_stop"]
    with pytest.raises(ValueError):
        advance_selection(states[-1], 0.1, 4, policy(stopping=VALIDATION_PLATEAU))
