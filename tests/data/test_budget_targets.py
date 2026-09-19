import importlib

import numpy as np
import pandas as pd
import pytest

from mars_titan.data.temporal import MarketClock


def target_module():
    try:
        return importlib.import_module("mars_titan.data.budget_targets")
    except ModuleNotFoundError:
        pytest.fail("Missing causal budget targets")


def fixture_prices():
    clock = MarketClock("US", "2021-01-01", "2023-12-31")
    market_returns = np.resize(np.array([-0.02, 0.01, 0.03, -0.01]), len(clock.days))
    base = {
        "session": [d.isoformat() for d in clock.days],
        "open": 100.0,
        "available_at": clock.decisions,
    }
    market = pd.DataFrame({**base, "close": 100 * (1 + market_returns)})
    asset = pd.DataFrame({**base, "close": 100 * (1 + 0.003 + 1.5 * market_returns)})
    return clock, asset, market


def test_known_intercept_slope_and_future_shock_do_not_leak():
    clock, asset, market = fixture_prices()
    asset.loc[300, "close"] += 2.0
    result = target_module().residual_targets(asset, market, clock, cutoff="2023-12-31")
    row = result.iloc[299]
    assert row.alpha == pytest.approx(0.003)
    assert row.beta == pytest.approx(1.5)
    assert row.target == pytest.approx(0.02)
    assert row.history_pairs == 252
    asset.loc[301:, "close"] *= 1.8
    changed = target_module().residual_targets(asset, market, clock, cutoff="2023-12-31")
    pd.testing.assert_series_equal(row, changed.iloc[299])


def test_missing_next_session_is_not_replaced_by_next_row():
    clock, asset, market = fixture_prices()
    asset = asset.drop(index=300)
    result = target_module().residual_targets(asset, market, clock, cutoff="2023-12-31")
    assert result.iloc[299].reason == "missing_next_session"
    assert pd.isna(result.iloc[299].target)
    assert result.iloc[400].history_pairs == 251


def test_minimum_history_zero_variance_and_maturity_cutoff():
    clock, asset, market = fixture_prices()
    result = target_module().residual_targets(asset, market, clock, cutoff="2022-12-31")
    assert result.iloc[124].reason == "insufficient_history"
    assert result.iloc[125].reason == "accepted"
    assert result.iloc[-1].reason == "target_after_cutoff"
    assert result.prediction_at.max().year == 2022
    market["close"] = 100.0
    flat = target_module().residual_targets(asset, market, clock, cutoff="2022-12-31")
    assert flat.iloc[200].reason == "zero_market_variance"


def test_unavailable_historical_close_is_excluded_from_regression():
    clock, asset, market = fixture_prices()
    asset.loc[0, "available_at"] = clock.decisions[200]
    result = target_module().residual_targets(asset, market, clock, cutoff="2023-12-31")
    assert result.iloc[125].reason == "insufficient_history"
    assert result.iloc[126].reason == "accepted"


def test_unknown_or_late_target_availability_is_not_an_observed_label():
    clock, asset, market = fixture_prices()
    market.loc[300, "available_at"] = pd.NaT
    asset.loc[302, "available_at"] = pd.Timestamp("2024-01-02", tz="UTC")
    result = target_module().residual_targets(asset, market, clock, cutoff="2023-12-31")
    assert result.iloc[299].reason == "target_after_cutoff"
    assert result.iloc[301].reason == "target_after_cutoff"
