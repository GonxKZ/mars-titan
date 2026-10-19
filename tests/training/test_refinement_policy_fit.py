"""Objetivos de utilidad firmados y ajuste posterior a la selección del padre."""

import importlib

import pytest
import torch

from mars_titan.training.learning_hold import LearningHoldError


def api():
    return importlib.import_module("mars_titan.training.refinement_policy_fit")


def data():
    features = torch.zeros((6, 9), dtype=torch.float64)
    features[:, 0] = torch.tensor([-1.0, -1, -1, 1, 1, 1])
    # K2C ayuda con el rasgo positivo y perjudica con el negativo.
    predictions = torch.ones((6, 6), dtype=torch.float64)
    predictions[:, 1] = 1 - 0.5 * features[:, 0]
    return dict(
        features=features,
        predictions=predictions,
        targets=torch.zeros(6, dtype=torch.float64),
        decision_at=torch.tensor([20, 20, 20, 30, 30, 30]),
        maturity_at=torch.tensor([21, 21, 21, 31, 31, 31]),
        session_weights=torch.ones(6, dtype=torch.float64) / 3,
        parent_selected_before=10,
        fit_start=20,
        fit_end=40,
        ridge=0.01,
        costs=torch.zeros(5, dtype=torch.float64),
        min_gain=0.0,
    )


def test_gain_targets_keep_negative_values_and_do_not_encode_retrospective_winners():
    m = api()
    sample = data()
    gains = m.signed_gains(sample["predictions"], sample["targets"])
    assert gains[:, 0].tolist() == [-0.5] * 3 + [0.5] * 3
    assert torch.equal(gains[:, 1:], torch.zeros((6, 4), dtype=torch.float64))


def test_finite_inputs_that_overflow_the_gain_do_not_return_nan():
    with pytest.raises(ValueError, match="finit"):
        api().signed_gains(
            torch.full((1, 6), 1e308, dtype=torch.float64),
            torch.tensor([-1e308], dtype=torch.float64),
        )


def test_fit_requires_authorization_before_inspecting_any_arrays(learning_hold):
    learning_hold(False)
    with pytest.raises(LearningHoldError):
        api().fit_policy(**(data() | {"features": None}))


def test_weighted_ridge_selects_useful_actions_outside_the_parent_fit():
    sample = data()
    policy, receipt = api().fit_policy(**sample)
    assert policy(sample["features"]).tolist() == [0, 0, 0, 1, 1, 1]
    assert receipt["target"] == "absolute_error_k1_minus_absolute_error_route"
    assert receipt["rows"] == 6
    assert receipt["decision_sessions"] == 2
    assert receipt["maximum_maturity_at"] == 31


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"parent_selected_before": 25}, "padre"),
        ({"fit_end": 31}, "madur"),
        ({"fit_start": 21}, "ajuste"),
        ({"maturity_at": torch.tensor([20, 21, 21, 31, 31, 31])}, "madur"),
        ({"session_weights": torch.zeros(6, dtype=torch.float64)}, "peso"),
        ({"ridge": 0.0}, "regularización"),
    ],
)
def test_ineligible_fit_inputs_are_rejected(changes, match):
    with pytest.raises(ValueError, match=match):
        api().fit_policy(**(data() | changes))


def test_duplication_with_compensated_session_weights_does_not_change_the_policy():
    sample = data()
    left, _ = api().fit_policy(**sample)
    repeats = torch.tensor([2, 1, 1, 3, 1, 1])
    sample["session_weights"] /= repeats
    for key in (
        "features",
        "predictions",
        "targets",
        "decision_at",
        "maturity_at",
        "session_weights",
    ):
        sample[key] = sample[key].repeat_interleave(repeats, dim=0)
    right, _ = api().fit_policy(**sample)
    torch.testing.assert_close(left.coefficients, right.coefficients, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(left.mean, right.mean, rtol=1e-12, atol=1e-12)


def test_large_finite_weights_are_normalized_without_sum_overflow():
    sample = data()
    reference, _ = api().fit_policy(**sample)
    sample["session_weights"] *= 1e308
    actual, _ = api().fit_policy(**sample)
    torch.testing.assert_close(actual.coefficients, reference.coefficients, rtol=1e-12, atol=1e-12)
