"""Época común de la criba (#746): regla pura, sin entrenar ni abrir datos."""

import math

import numpy as np
import pytest

from mars_titan.training.selection import (
    AWAIT,
    CHRONOLOGICAL_MODES,
    COMMON_EPOCH,
    CONTINUE,
    FINISH,
    JOINT_PLATEAU,
    STOPPING_MODES,
    advance_selection,
    awaiting,
    bind_joint_epoch,
    campaign_rule,
    common_epoch,
    epoch_decision,
    initial_selection,
    seed_epoch_scores,
    validate_selection,
)

COMMON = dict(metric="session_mae", patience=2, min_delta=0.0, stopping=COMMON_EPOCH)
# El MAE empeora dos épocas seguidas tras la 2: una meseta que la época común no puede usar.
MAE = (0.5, 0.3, 0.31, 0.32, 0.29, 0.33)


def trace(scores, options=COMMON):
    state, result = None, []
    for epoch, score in enumerate(scores, 1):
        state = advance_selection(state, score, epoch, options)
        result.append(state)
    return result


def test_only_the_chronological_trainers_accept_the_common_epoch():
    assert COMMON_EPOCH not in STOPPING_MODES and COMMON_EPOCH in CHRONOLOGICAL_MODES
    with pytest.raises(ValueError, match="solo la admiten Titans-MAC"):
        validate_selection(COMMON)
    validate_selection(COMMON, modes=CHRONOLOGICAL_MODES)
    rule = dict(COMMON, stopping="fixed_budget", max_epochs=6)
    with pytest.raises(ValueError, match="época común"):
        campaign_rule(rule, dict(COMMON, max_epochs=6))
    assert campaign_rule(rule, dict(COMMON, max_epochs=6), modes=CHRONOLOGICAL_MODES) == dict(
        COMMON, max_epochs=6
    )


def test_the_common_epoch_runs_the_whole_budget_and_keeps_the_mae_best_as_sensitivity():
    states = trace(MAE)
    assert not any(state["should_stop"] for state in states)
    # No hay meseta: la paciencia se cuenta, pero nunca fija una parada individual.
    assert [state["plateau_epoch"] for state in states] == [None] * len(MAE)
    assert [state["stale_epochs"] for state in states] == [0, 0, 1, 2, 0, 1]
    assert (states[-1]["best_epoch"], states[-1]["best_score"]) == (5, 0.29)
    joint = trace(MAE, dict(COMMON, stopping=JOINT_PLATEAU))
    assert joint[-1]["plateau_epoch"] == 4


def test_the_common_epoch_continues_awaits_the_group_and_finishes_with_any_evaluated_epoch():
    states = trace(MAE)
    epochs = len(MAE)
    assert epoch_decision(None, COMMON, epochs) == CONTINUE
    assert [epoch_decision(state, COMMON, epochs) for state in states] == [CONTINUE] * (
        epochs - 1
    ) + [AWAIT]
    for epoch in range(1, epochs + 1):
        assert epoch_decision(states[-1], COMMON, epochs, epoch) == FINISH
    report = awaiting(states[-1], epochs)
    assert report["status"] == AWAIT and report["individual_stop_epoch"] == epochs
    assert report["plateau_epoch"] is None


@pytest.mark.parametrize("broken", [0, 7, 3.0, True, "3"])
def test_the_common_epoch_must_be_an_evaluated_epoch(broken):
    states = trace(MAE)
    with pytest.raises(ValueError, match="época común"):
        epoch_decision(states[-1], COMMON, len(MAE), broken)


def test_the_common_epoch_arrives_only_after_the_whole_budget():
    states = trace(MAE)
    for state in (None, *states[:-1]):
        with pytest.raises(ValueError, match="época común"):
            epoch_decision(state, COMMON, len(MAE), 2)
    # Un ajuste nuevo no puede recibirla antes de empezar.
    with pytest.raises(ValueError, match="época común"):
        bind_joint_epoch({}, 2, COMMON, len(MAE))
    report = dict(selection=states[-1])
    bind_joint_epoch(report, 2, COMMON, len(MAE))
    assert report["joint_stop_epoch"] == 2
    with pytest.raises(ValueError, match="otra época conjunta"):
        bind_joint_epoch(report, 3, COMMON, len(MAE))


def test_an_initial_parent_state_does_not_count_as_an_evaluated_epoch():
    state = initial_selection(0.2, COMMON)
    assert epoch_decision(state, COMMON, 3) == CONTINUE
    with pytest.raises(ValueError, match="época común"):
        epoch_decision(state, COMMON, 3, 1)


def series(*epochs):
    """Pares (valores, definidas) de una semilla, uno por época."""
    return [(np.asarray(values, dtype=float), np.asarray(defined)) for values, defined in epochs]


def test_seed_scores_average_seeds_on_the_sessions_that_all_define():
    first = series(([0.1, 0.3, 0.5], [True, True, False]), ([0.2, -0.2, 0.0], [True] * 3))
    second = series(([0.3, 0.1, 0.9], [True, True, True]), ([0.0, 0.2, 0.4], [True] * 3))
    scores = seed_epoch_scores([first, second])
    # Época 1: la tercera sesión no la define la primera semilla.
    assert scores[0] == pytest.approx(np.mean([0.2, 0.2]))
    assert scores[1] == pytest.approx(np.mean([0.1, 0.0, 0.2]))


@pytest.mark.parametrize(
    "broken, message",
    [
        ([], "mismas épocas"),
        ([[]], "mismas épocas"),
        ([series(([0.1], [True])), series(([0.1], [True]), ([0.2], [True]))], "mismas épocas"),
        ([series(([0.1, 0.2], [True, True])), series(([0.1], [True]))], "sesiones"),
        ([series(([0.1, 0.2], [True, False])), series(([0.1, 0.2], [False, True]))], "ninguna"),
        ([series(([math.nan, 0.2], [True, True]))], "no finito"),
    ],
)
def test_seed_scores_reject_incomplete_groups(broken, message):
    with pytest.raises(ValueError, match=message):
        seed_epoch_scores(broken)


def test_the_smoothed_maximum_prefers_a_plateau_over_a_lone_peak():
    scores = [0.0, 0.06, 0.0, 0.03, 0.035, 0.03, 0.0]
    chosen = common_epoch(scores)
    assert chosen["epoch"] == 5 and max(scores) == scores[1]
    assert chosen["smoothed"][0] == pytest.approx(0.03)
    assert chosen["smoothed"][4] == pytest.approx(0.095 / 3)
    assert chosen["smoothed"][-1] == pytest.approx(0.015)
    # Sin suavizado vuelve a ser el máximo de una sola época.
    assert common_epoch(scores, radius=0)["epoch"] == 2


def test_ties_keep_the_earliest_epoch():
    # Valores exactos en binario, para que el empate no dependa del redondeo.
    assert common_epoch([0.25, 0.25, 0.25])["epoch"] == 1
    assert common_epoch([0.125, 0.375, 0.125, 0.375, 0.125], radius=0)["epoch"] == 2
    assert common_epoch([0.5])["epoch"] == 1


@pytest.mark.parametrize(
    "scores, radius",
    [([], 1), ([0.1, math.inf], 1), ([0.1, None], 1), ([0.1], -1), ([0.1], 1.0), ([True], 1)],
)
def test_the_common_epoch_rejects_invalid_scores(scores, radius):
    with pytest.raises(ValueError, match="puntuaciones finitas"):
        common_epoch(scores, radius=radius)
