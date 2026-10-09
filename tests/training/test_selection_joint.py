"""Pruebas de la parada conjunta y de la decisión por época, sin entrenar ni abrir datos."""

from dataclasses import dataclass, field

import pytest

from mars_titan.training.selection import (
    AWAIT,
    CONTINUE,
    FINISH,
    FIXED_BUDGET,
    JOINT_PLATEAU,
    VALIDATION_PLATEAU,
    advance_selection,
    awaiting,
    bind_joint_epoch,
    campaign_rule,
    epoch_decision,
    individual_stop,
    initial_selection,
    with_rule,
)


def policy(**changes):
    return dict(dict(metric="session_mae", patience=2, min_delta=0.0), **changes)


def states(scores, options):
    state, result = None, []
    for epoch, score in enumerate(scores, 1):
        state = advance_selection(state, score, epoch, options)
        result.append(state)
    return result


SCORES = (0.5, 0.3, 0.31, 0.32, 0.29, 0.33, 0.34)


def test_joint_plateau_never_stops_alone_and_records_the_first_plateau():
    joint = states(SCORES, policy(stopping=JOINT_PLATEAU))
    fixed = states(SCORES, policy(stopping=FIXED_BUDGET))
    # La meseta conjunta registra lo mismo que el presupuesto fijo y nunca corta sola.
    assert joint == fixed
    assert not any(state["should_stop"] for state in joint)
    assert [state["plateau_epoch"] for state in joint] == [None, None, None, 4, 4, 4, 4]
    assert [individual_stop(state, 7) for state in joint] == [None, None, None, 4, 4, 4, 4]


def test_individual_stop_is_the_maximum_when_no_plateau_arrives():
    improving = states((0.5, 0.4, 0.3), policy(stopping=JOINT_PLATEAU))
    assert [individual_stop(state, 3) for state in improving] == [None, None, 3]
    assert individual_stop(None, 3) is None
    assert individual_stop(improving[1], 2) == 2


def test_epoch_decision_awaits_the_group_at_the_first_plateau_or_the_maximum():
    options = policy(stopping=JOINT_PLATEAU)
    trace = states(SCORES, options)
    assert epoch_decision(None, options, 7) == CONTINUE
    assert [epoch_decision(state, options, 7) for state in trace] == [
        CONTINUE,
        CONTINUE,
        CONTINUE,
        AWAIT,
        AWAIT,
        AWAIT,
        AWAIT,
    ]
    improving = states((0.5, 0.4, 0.3), options)
    assert [epoch_decision(state, options, 3) for state in improving] == [
        CONTINUE,
        CONTINUE,
        AWAIT,
    ]


@pytest.mark.parametrize("joint_epoch", [4, 5, 7])
def test_epoch_decision_runs_exactly_to_the_joint_epoch(joint_epoch):
    options = policy(stopping=JOINT_PLATEAU)
    trace = states(SCORES, options)
    decisions = [epoch_decision(state, options, 7, joint_epoch) for state in trace[3:joint_epoch]]
    assert decisions == [CONTINUE] * (joint_epoch - 4) + [FINISH]


def test_epoch_decision_rejects_an_invalid_joint_epoch():
    options = policy(stopping=JOINT_PLATEAU)
    trace = states(SCORES, options)
    for state, joint_epoch in (
        (trace[3], 3),  # Es anterior a la meseta del propio ajuste.
        (trace[3], 8),  # Es posterior al máximo de épocas.
        (trace[5], 5),  # Es anterior a la última época ya evaluada.
        (trace[1], 4),  # El ajuste aún no ha llegado a su meseta.
        (None, 4),
        (trace[3], 4.0),
        (trace[3], True),
    ):
        with pytest.raises(ValueError, match="época conjunta"):
            epoch_decision(state, options, 7, joint_epoch)
    for mode in (FIXED_BUDGET, VALIDATION_PLATEAU):
        other = policy(stopping=mode)
        with pytest.raises(ValueError, match="época conjunta"):
            epoch_decision(states(SCORES[:4], other)[-1], other, 7, 4)


def test_epoch_decision_keeps_the_other_modes():
    plateau = policy(stopping=VALIDATION_PLATEAU)
    stopped = states(SCORES[:4], plateau)
    assert [epoch_decision(state, plateau, 7) for state in stopped] == [CONTINUE] * 3 + [FINISH]
    fixed = policy(stopping=FIXED_BUDGET)
    budget = states(SCORES, fixed)
    assert [epoch_decision(state, fixed, 7) for state in budget] == [CONTINUE] * 6 + [FINISH]
    legacy = states(SCORES[:4], policy())
    assert epoch_decision(legacy[-1], policy(), 7) == FINISH
    assert epoch_decision(None, fixed, 7) == CONTINUE


def test_initial_state_of_the_parent_counts_in_the_joint_selection():
    options = policy(stopping=JOINT_PLATEAU)
    state = initial_selection(0.2, options)
    assert epoch_decision(state, options, 5) == CONTINUE
    for epoch, score in enumerate((0.3, 0.25), 1):
        state = advance_selection(state, score, epoch, options)
    # El padre sigue siendo el mejor estado y la meseta llega sin haber mejorado.
    assert (state["best_epoch"], state["plateau_epoch"]) == (0, 2)
    assert epoch_decision(state, options, 5) == AWAIT
    report = awaiting(state, 5)
    assert report == dict(
        status=AWAIT,
        individual_stop_epoch=2,
        plateau_epoch=2,
        best_epoch=0,
        best_score=0.2,
        last_epoch=2,
    )


def test_bind_joint_epoch_validates_and_keeps_the_first_epoch_received():
    options = policy(stopping=JOINT_PLATEAU)
    report = dict(selection=states(SCORES[:4], options)[-1])
    bind_joint_epoch(report, None, options, 7)
    assert "joint_stop_epoch" not in report
    # Una época no válida no llega a fijarse.
    for broken in (3, 8):
        with pytest.raises(ValueError, match="época conjunta"):
            bind_joint_epoch(report, broken, options, 7)
        assert "joint_stop_epoch" not in report
    with pytest.raises(ValueError, match="época conjunta"):
        bind_joint_epoch(dict(report), 6, policy(stopping=FIXED_BUDGET), 7)
    with pytest.raises(ValueError, match="época conjunta"):
        bind_joint_epoch({}, 6, options, 7)
    bind_joint_epoch(report, 6, options, 7)
    bind_joint_epoch(report, 6, options, 7)
    assert report["joint_stop_epoch"] == 6
    for other in (5, None):
        with pytest.raises(ValueError, match="otra época conjunta"):
            bind_joint_epoch(report, other, options, 7)


RULE = dict(metric="session_mae", patience=5, min_delta=1e-05, stopping=FIXED_BUDGET, max_epochs=30)
JOINT = dict(RULE, stopping=JOINT_PLATEAU, minimum_epochs=5)


def test_campaign_rule_keeps_the_metric_and_validates_the_override():
    assert campaign_rule(RULE, None) == RULE and campaign_rule(RULE, None) is not RULE
    assert campaign_rule(RULE, JOINT) == JOINT
    for broken in (
        dict(JOINT, metric="mae"),
        {k: v for k, v in JOINT.items() if k != "max_epochs"},
        dict(JOINT, max_epochs=0),
        dict(JOINT, max_epochs=1001),
        dict(JOINT, max_epochs=30.0),
        dict(JOINT, minimum_epochs=30),
        dict(JOINT, stopping="unbounded"),
        dict(JOINT, patience=0),
        dict(JOINT, extra=1),
        "joint",
    ):
        with pytest.raises(ValueError):
            campaign_rule(RULE, broken)


def test_with_rule_changes_only_epochs_and_selection():
    @dataclass(frozen=True)
    class Recipe:
        epochs: int
        selection: dict = field(default_factory=dict)
        learning_rate: float = 1e-3

    selection = {k: v for k, v in RULE.items() if k != "max_epochs"}
    recipe = Recipe(epochs=30, selection=selection)
    assert with_rule(recipe, RULE) is recipe
    changed = with_rule(recipe, dict(JOINT, max_epochs=20))
    assert changed.epochs == 20 and changed.learning_rate == recipe.learning_rate
    assert changed.selection == {k: v for k, v in JOINT.items() if k != "max_epochs"}
    assert recipe.selection == selection and changed != recipe
