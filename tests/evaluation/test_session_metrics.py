"""Errores conocidos, sin dar más peso a las sesiones con más activos."""

import importlib
from copy import deepcopy

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


def row_reference(state, markets, times, errors):
    """Oráculo secuencial con el orden de suma de cada lote explícito."""
    pending = {}
    for market, moment, value in zip(markets, times, errors, strict=True):
        entry = pending.setdefault((str(market), int(moment)), [0, 0.0, 0.0])
        value = float(value)
        entry[0] += 1
        entry[1] += abs(value)
        entry[2] += value * value
    for key, values in pending.items():
        previous = state.get(key, [0, 0.0, 0.0])
        state[key] = [previous[i] + values[i] for i in range(3)]


@pytest.mark.parametrize("rows", [31, 32, 33, 64, 65, 128, 129, 512, 4096])
@pytest.mark.parametrize("layout", ["cohorts", "unique"])
@pytest.mark.parametrize("dtype", ["int64", "uint64", "datetime64[us]"])
def test_grouping_matches_sequential_batch_sums_and_preserves_timestamp_range(rows, layout, dtype):
    rng = np.random.default_rng(984)
    markets = rng.choice(["CN", "US"], rows)
    times = np.arange(rows, dtype=np.int64)
    if layout == "cohorts":
        times //= 16
    if dtype == "uint64":
        times = times.astype(np.uint64) + np.uint64(2**64 - rows)
    elif dtype == "int64":
        times += np.iinfo(np.int64).min
    raw_times = times
    times = times.astype(dtype)
    metrics, expected = accumulator(), {}
    for _ in range(3):
        errors = rng.normal(size=rows)
        errors[::7] = 2**50
        errors[::11] = 2**-50
        metrics.update(markets, times, errors)
        row_reference(expected, markets, raw_times, errors)
        assert metrics.sessions == expected
        assert list(metrics.sessions) == list(expected)


@pytest.mark.parametrize("previous", [False, True])
def test_overflow_in_a_group_or_its_previous_state_rejects_the_entire_batch(previous):
    metrics = accumulator()
    metrics.update(["US"], [1], [1e154 if previous else 2.0])
    before = deepcopy(metrics.sessions)
    markets = ["CN", "US"] if previous else ["CN", "US", "US"]
    times = [2, 1] if previous else [2, 1, 1]
    errors = [1.0, 1e154] if previous else [1.0, 1e154, 1e154]
    with pytest.raises(ValueError, match="acumulación|rango"):
        metrics.update(markets, times, errors)
    assert metrics.sessions == before


@pytest.mark.parametrize("rows", [0, 4097])
def test_row_limit_rejects_the_batch_without_changing_the_confirmed_state(rows):
    metrics = accumulator()
    metrics.update(["US"], [1], [2.0])
    before = deepcopy(metrics.sessions)
    with pytest.raises(ValueError, match="4096"):
        metrics.update(["US"] * rows, np.arange(rows), np.ones(rows))
    assert metrics.sessions == before


@pytest.mark.parametrize("rows", [31, 32, 33, 64, 65, 128, 129, 512, 4096])
def test_each_batch_is_reduced_before_adding_the_previous_large_state(rows):
    metrics = accumulator()
    metrics.update(["US"], [7], [2**53])
    metrics.update(["US"] * rows, [7] * rows, np.ones(rows))
    assert metrics.sessions == {("US", 7): [1 + rows, float(2**53 + rows), float(2**106)]}
    assert metrics.summary()["absolute_error"] == float(2**53 + rows)


def test_group_reduction_preserves_row_order_when_rounding_is_observable():
    first, last = accumulator(), accumulator()
    values = np.array([2**53, *([1.0] * 128)])
    first.update(["US"] * 129, [7] * 129, values)
    last.update(["US"] * 129, [7] * 129, values[::-1])
    assert first.sessions[("US", 7)] == [129, float(2**53), float(2**106)]
    assert last.sessions[("US", 7)] == [129, float(2**53 + 128), float(2**106)]
