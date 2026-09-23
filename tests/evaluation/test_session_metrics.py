"""Errores conocidos, sin dar más peso a las sesiones con más activos."""

import importlib

import numpy as np
import pytest


def accumulator(**kwargs):
    try:
        cls = importlib.import_module("mars_titan.evaluation.session_metrics").SessionErrors
    except ModuleNotFoundError:
        pytest.fail("Falta la agregación de errores por sesión")
    return cls(**kwargs)


def test_unbalanced_sessions_have_equal_weight_and_keep_row_metrics():
    metrics = accumulator()
    metrics.update(["US"] * 3, [10, 10, 20], [-1.0, 3.0, 6.0])
    result = metrics.summary()
    assert result["samples"] == 3
    assert result["session_count"] == 2
    assert result["session_mae"] == 4.0
    assert result["session_mse"] == 20.5
    assert result["absolute_error"] == 10.0
    assert result["squared_error"] == 46.0
    assert result["session_metrics_reason"] is None


def test_physical_batches_and_market_order_do_not_change_session_means():
    whole = accumulator()
    whole.update(["US", "CN", "US"], [10, 10, 10], [1.0, 6.0, 3.0])
    parts = accumulator()
    parts.update(["US"], [10], [3.0])
    parts.update(["US", "CN"], [10, 10], [1.0, 6.0])
    assert parts.summary() == whole.summary()
    assert parts.summary()["session_count"] == 2
    assert parts.summary()["session_mae"] == 4.0
    assert parts.summary()["by_market_session"]["CN"]["mae"] == 6.0


def test_empty_accumulator_does_not_claim_zero_error():
    result = accumulator().summary()
    assert result["samples"] == result["session_count"] == 0
    assert result["session_mae"] is result["session_mse"] is None
    assert result["session_metrics_reason"]


@pytest.mark.parametrize(
    "markets,times,errors",
    [
        (["US"], [1], [float("nan")]),
        (["US"], [1], [1e200]),
        (["US"], [1, 2], [1.0]),
        (["other"], [1], [1.0]),
        (["US"], [1.5], [1.0]),
    ],
)
def test_invalid_batch_does_not_mutate_previously_confirmed_statistics(markets, times, errors):
    metrics = accumulator()
    metrics.update(["US"], [1], [2.0])
    before = metrics.summary()
    with pytest.raises(ValueError):
        metrics.update(markets, times, errors)
    assert metrics.summary() == before


def test_session_budget_fails_before_accepting_a_partial_batch():
    metrics = accumulator(max_sessions=1)
    metrics.update(["US"], [1], [2.0])
    before = metrics.summary()
    with pytest.raises(ValueError, match="presupuesto|sesiones"):
        metrics.update(["US", "US"], [1, 2], [3.0, 4.0])
    assert metrics.summary() == before


def test_numpy_datetime_batches_preserve_microsecond_session_keys():
    metrics = accumulator()
    metrics.update(["US", "US"], np.array([10, 10], dtype="datetime64[us]"), [1.0, 3.0])
    metrics.update(["US"], [20], [6.0])
    assert metrics.summary()["session_count"] == 2
    assert metrics.summary()["session_mae"] == 4.0


@pytest.mark.parametrize(
    "times", [np.array(["NaT"], dtype="datetime64[us]"), np.array([1], dtype="datetime64[ns]")]
)
def test_missing_or_unrepresentable_date_does_not_merge_sessions(times):
    with pytest.raises(ValueError, match="fechas|precisión"):
        accumulator().update(["US"], times, [1.0])
