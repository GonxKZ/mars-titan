import importlib

import numpy as np
import pandas as pd
import pytest

from mars_titan.data.budget_targets import residual_targets
from tests.data.test_budget_targets import fixture_prices


def calculate(*args, **kwargs):
    try:
        module = importlib.import_module("mars_titan.data.residual_arrays")
    except ModuleNotFoundError:
        pytest.fail("Falta el cálculo de residuales sobre arrays")
    return module.residual_targets_array(*args, **kwargs)


@pytest.mark.parametrize("change", ["none", "gaps", "late", "missing", "constant", "near_constant"])
def test_array_calculation_preserves_every_reference_value_and_reason(change):
    clock, asset, market = fixture_prices()
    asset.loc[300, "close"] += 2.0
    if change == "gaps":
        asset = asset.drop(index=[4, 126, 300])
        market.loc[145, "close"] = np.nan
    elif change == "late":
        asset.loc[0, "available_at"] = clock.decisions[200]
        market.loc[128, "available_at"] = pd.Timestamp("2024-01-02", tz="UTC")
    elif change == "missing":
        market.loc[[3, 130, 302], "available_at"] = pd.NaT
    elif change == "constant":
        market["close"] = 100.0
    elif change == "near_constant":
        market["close"] = 100 * (1 + np.resize([0.01 - 1e-8, 0.01 + 1e-8], len(market)))
    expected = residual_targets(asset, market, clock, cutoff="2023-12-31")
    actual = calculate(asset, market, clock, cutoff="2023-12-31")
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_known_residual_does_not_change_when_later_prices_change():
    clock, asset, market = fixture_prices()
    asset.loc[300, "close"] += 2.0
    row = calculate(asset, market, clock, cutoff="2023-12-31").iloc[299]
    assert row.alpha == pytest.approx(0.003)
    assert row.beta == pytest.approx(1.5)
    assert row.target == pytest.approx(0.02)
    assert row.history_pairs == 252
    asset.loc[301:, "close"] *= 1.8
    changed = calculate(asset, market, clock, cutoff="2023-12-31").iloc[299]
    pd.testing.assert_series_equal(row, changed, check_exact=True)


@pytest.mark.parametrize("history,minimum", [(2, 2), (31, 10), (252, 126), (1000, 600)])
def test_history_windows_and_cutoff_are_unchanged(history, minimum):
    clock, asset, market = fixture_prices()
    options = dict(cutoff="2022-12-31", history=history, minimum=minimum)
    pd.testing.assert_frame_equal(
        calculate(asset, market, clock, **options),
        residual_targets(asset, market, clock, **options),
        check_exact=True,
    )


@pytest.mark.parametrize("history,minimum", [(1, 1), (10, 11), (0, 2)])
def test_invalid_window_fails_before_calculating(history, minimum):
    clock, asset, market = fixture_prices()
    with pytest.raises(ValueError, match="historial"):
        calculate(asset, market, clock, cutoff="2023-12-31", history=history, minimum=minimum)


def test_duplicate_prices_are_not_silently_used():
    clock, asset, market = fixture_prices()
    with pytest.raises(ValueError, match="duplicada"):
        calculate(pd.concat([asset, asset.iloc[:1]]), market, clock, cutoff="2023-12-31")


def test_no_sessions_before_cutoff_preserves_empty_reference():
    clock, asset, market = fixture_prices()
    pd.testing.assert_frame_equal(
        calculate(asset, market, clock, cutoff="2020-12-31"),
        residual_targets(asset, market, clock, cutoff="2020-12-31"),
        check_exact=True,
    )
