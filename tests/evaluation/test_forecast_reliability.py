"""Diagnósticos de retornos residuales y radios fijados con calibración."""

import copy

import numpy as np
import pytest

from mars_titan.evaluation.forecast_reliability import (
    calibrate_absolute_error,
    directional_diagnostics,
    interval_diagnostics,
)


def calibration(confidence=0.9):
    return calibrate_absolute_error(np.ones(20), np.zeros(20), confidence=confidence)


def test_direction_counts_exclude_zero_targets_and_keep_abstentions_in_total_accuracy():
    result = directional_diagnostics([2, 2, -2, -2, 3, -3, 0, 0], [1, -1, -1, 1, 0, 0, 1, 0])
    assert result["target_kind"] == "residual_return"
    assert result["samples"] == 8
    assert result["zero_targets"] == 2
    assert result["eligible_targets"] == 6
    assert result["calls"] == 4
    assert result["abstentions"] == 2
    assert result["correct_calls"] == 2
    assert result["call_coverage"] == pytest.approx(2 / 3)
    assert result["conditional_accuracy"] == 0.5
    assert result["direction_accuracy"] == pytest.approx(1 / 3)
    assert result["positive_precision"] == result["negative_precision"] == 0.5
    assert result["positive_recall"] == result["negative_recall"] == pytest.approx(1 / 3)
    assert result["positive_targets"] == result["negative_targets"] == 3
    assert result["positive_calls"] == result["negative_calls"] == 2
    assert result["correct_positive"] == result["correct_negative"] == 1
    assert result["nonzero_predictions"] == 5
    assert result["nonzero_prediction_fraction"] == 5 / 8


def test_absent_classes_and_abstention_are_not_perfect_accuracy():
    no_calls = directional_diagnostics([1, -1], [0, 0])
    assert no_calls["conditional_accuracy"] is None
    assert no_calls["direction_accuracy"] == 0
    assert no_calls["positive_precision"] is None
    assert no_calls["negative_precision"] is None
    assert no_calls["positive_recall"] == no_calls["negative_recall"] == 0
    positives = directional_diagnostics([1, 2], [1, 1])
    assert positives["positive_precision"] == positives["positive_recall"] == 1
    assert positives["negative_precision"] is None
    assert positives["negative_recall"] is None


@pytest.mark.parametrize("target,prediction", [([], []), ([0, 0], [1, -1])])
def test_no_eligible_targets_leave_direction_metrics_undefined(target, prediction):
    result = directional_diagnostics(target, prediction)
    assert result["eligible_targets"] == result["calls"] == result["abstentions"] == 0
    for field in (
        "call_coverage",
        "conditional_accuracy",
        "direction_accuracy",
        "positive_precision",
        "negative_precision",
        "positive_recall",
        "negative_recall",
    ):
        assert result[field] is None


@pytest.mark.parametrize("confidence,rank,radius", [(0.9, 19, 19.0), (0.95, 20, 20.0)])
def test_calibration_selects_the_finite_sample_order_statistic_without_interpolation(
    confidence, rank, radius
):
    result = calibrate_absolute_error(
        np.arange(1, 21, dtype=np.float64), np.zeros(20), confidence=confidence
    )
    assert result["calibration_samples"] == 20
    assert result["order_statistic"] == rank
    assert result["radius"] == radius
    assert result["reason"] is None
    assert result["partition"] == "calibration"
    assert result["coverage_guaranteed"] is False
    assert result["target_kind"] == "residual_return"


@pytest.mark.parametrize("confidence,n", [(0.9, 0), (0.9, 8), (0.95, 18)])
def test_insufficient_calibration_does_not_invent_an_interval(confidence, n):
    fitted = calibrate_absolute_error(np.zeros(n), np.zeros(n), confidence=confidence)
    assert fitted["radius"] is None
    assert fitted["reason"]
    result = interval_diagnostics([1], [1], fitted)
    assert result["intervals"] == 0
    assert result["coverage"] is None
    assert result["mean_width"] is None
    assert result["direction"] is None
    assert result["reason"] == fitted["reason"]


def test_intervals_include_both_endpoints_but_zero_at_an_endpoint_abstains():
    result = interval_diagnostics([1, 3, -1, -3, 0], [1, 2, -1, -2, 0], calibration())
    assert result["intervals"] == result["covered"] == 5
    assert result["coverage"] == 1
    assert result["mean_width"] == 2
    assert result["direction"]["calls"] == 2
    assert result["direction"]["abstentions"] == 2
    assert result["direction"]["conditional_accuracy"] == 1
    assert result["direction"]["direction_accuracy"] == 0.5


def test_evaluation_counts_misses_and_never_recalibrates_on_evaluation_errors():
    fitted = calibration()
    before = copy.deepcopy(fitted)
    result = interval_diagnostics([0, 100, -100], [0, 2, -2], fitted)
    assert result["coverage"] == pytest.approx(1 / 3)
    assert result["covered"] == 1
    assert result["mean_width"] == 2
    assert fitted == before


def test_exact_calibration_can_produce_zero_width_and_empty_evaluation_stays_undefined():
    fitted = calibrate_absolute_error(np.ones(20), np.ones(20), confidence=0.95)
    assert fitted["radius"] == 0
    result = interval_diagnostics([-1, 0, 1], [-1, 0, 1], fitted)
    assert result["coverage"] == 1
    assert result["mean_width"] == 0
    assert result["direction"]["calls"] == 2
    empty = interval_diagnostics([], [], fitted)
    assert empty["coverage"] is None
    assert empty["mean_width"] is None
    assert empty["direction"]["conditional_accuracy"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"confidence": 0.8},
        {"partition": "evaluation"},
        {"schema_version": 2},
        {"calibration_samples": True},
        {"calibration_samples": -1},
        {"order_statistic": 1},
        {"radius": -1},
        {"radius": np.nan},
        {"radius": np.inf},
        {"radius": 10**1000},
        {"radius": None},
        {"radius": True},
        {"coverage_guaranteed": True},
        {"target_kind": "raw_return"},
        {"reason": "La calibración no terminó"},
    ],
)
def test_corrupted_calibration_is_rejected(changes):
    with pytest.raises(ValueError):
        interval_diagnostics([0], [0], calibration() | changes)


def test_missing_fields_and_fabricated_radius_for_insufficient_calibration_are_rejected():
    fitted = calibration()
    fitted.pop("radius")
    with pytest.raises(ValueError):
        interval_diagnostics([0], [0], fitted)
    insufficient = calibrate_absolute_error([0], [0], confidence=0.95)
    with pytest.raises(ValueError):
        interval_diagnostics([0], [0], insufficient | {"radius": 0.0})


def test_calibration_cannot_be_fitted_on_evaluation_or_an_undeclared_level():
    for options in ({"partition": "evaluation"}, {"confidence": 0.99}, {"confidence": True}):
        with pytest.raises(ValueError):
            calibrate_absolute_error(np.zeros(20), np.zeros(20), **({"confidence": 0.9} | options))


@pytest.mark.parametrize(
    "target,prediction",
    [
        ([1, 2], [1]),
        ([[1]], [[1]]),
        ([np.nan], [0]),
        ([1], [np.inf]),
        ([1 + 2j], [0]),
        ([True], [False]),
        ([2**53 + 1], [0]),
        ([1], ["1"]),
    ],
)
def test_all_entrypoints_reject_invalid_or_implicitly_broadcast_inputs(target, prediction):
    for operation in (
        lambda: directional_diagnostics(target, prediction),
        lambda: calibrate_absolute_error(target, prediction, confidence=0.9),
        lambda: interval_diagnostics(target, prediction, calibration()),
    ):
        with pytest.raises(ValueError):
            operation()


def test_inputs_are_not_mutated_and_the_row_budget_is_enforced():
    target = np.arange(20, dtype=np.float64)
    prediction = np.zeros(20)
    before = target.copy(), prediction.copy()
    fitted = calibrate_absolute_error(target, prediction, confidence=0.9)
    directional_diagnostics(target, prediction)
    interval_diagnostics(target, prediction, fitted)
    np.testing.assert_array_equal(target, before[0])
    np.testing.assert_array_equal(prediction, before[1])
    values = np.zeros(1_000_001)
    with pytest.raises(ValueError):
        directional_diagnostics(values, values)


def test_large_finite_values_keep_direction_but_overflow_is_never_silenced():
    assert directional_diagnostics([1e308, -1e308], [1e308, -1e308])["direction_accuracy"] == 1
    with pytest.raises(ValueError):
        calibrate_absolute_error(np.full(20, 1e308), np.full(20, -1e308), confidence=0.9)
    with pytest.raises(ValueError):
        calibrate_absolute_error(np.full(20, 1e308), np.zeros(20), confidence=0.9)
    fitted = calibrate_absolute_error(np.full(20, 1e307), np.zeros(20), confidence=0.9)
    with pytest.raises(ValueError):
        interval_diagnostics([1.79e308], [1.79e308], fitted)
