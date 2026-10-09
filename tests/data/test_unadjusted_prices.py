"""Inversión del ajuste del proveedor sobre series con precios negociados conocidos."""

import numpy as np
import pytest

from mars_titan.data.unadjusted_prices import (
    CN_A_SHARE_TICK,
    US_CENT_TICK,
    US_FRACTION_TICK,
    US_SUBDOLLAR_TICK,
    ProviderHistory,
    dividend_offsets,
    fit_terminal_constant,
    grid_concordance,
    price_grid,
    price_increments,
    read_provider_history,
    reconstruct_unadjusted,
    shifted_events,
    split_multipliers,
)


def provider_series(raw, dividends, splits, terminal_factor=1.0, float32=False):
    """Aplicar hacia delante la convención observada del proveedor.

    El dividendo se publica dividido por los splits estrictamente posteriores y su
    factor ``1 - D / C`` usa el cierre previo dividido por todos los splits
    posteriores a esa sesión, incluido el del mismo día.
    """
    rows = len(raw)
    later = np.ones(rows)
    for t in range(rows - 2, -1, -1):
        later[t] = later[t + 1] * (splits[t + 1] if splits[t + 1] else 1.0)
    split_adjusted = raw / later
    published = dividends / later
    factor = np.full(rows, terminal_factor)
    for t in range(rows - 1, 0, -1):
        step = 1 - published[t] / split_adjusted[t - 1] if published[t] else 1.0
        factor[t - 1] = factor[t] * step
    adjusted = split_adjusted * factor
    if float32:
        adjusted = adjusted.astype(np.float32).astype(np.float64)
    return adjusted, published


def history(sessions, close, dividends, splits, scale=1.0):
    values = np.asarray(close, dtype=np.float64)
    return ProviderHistory(
        np.asarray(sessions, dtype="datetime64[D]"),
        values * 0.99,
        values * 1.01,
        values * 0.98,
        values,
        np.full(len(values), 1000.0 * scale),
        np.asarray(dividends, dtype=np.float64),
        np.asarray(splits, dtype=np.float64),
    )


@pytest.fixture
def market_path():
    rng = np.random.default_rng(20261009)
    rows = 420
    sessions = np.datetime64("2021-01-04") + np.arange(rows) * np.timedelta64(1, "D")
    cents = 4000 + np.cumsum(rng.integers(-40, 41, rows))
    raw = cents / 100.0
    dividends = np.zeros(rows)
    splits = np.zeros(rows)
    dividends[[30, 95, 160, 230]] = [0.22, 0.24, 0.24, 0.26]
    splits[120] = 2.0
    dividends[300], splits[300] = 0.35, 1.5  # Split y dividendo en la misma fecha ex.
    dividends[395] = 0.18  # Posterior al corte: no debe leerse.
    splits[405] = 3.0  # Posterior al corte: solo aporta escala.
    return sessions, raw, dividends, splits


def test_reconstruction_recovers_traded_closes_without_post_cutoff_prices(market_path):
    sessions, raw, dividends, splits = market_path
    adjusted, published = provider_series(raw, dividends, splits, terminal_factor=0.993)
    source = history(sessions, adjusted, published, splits)
    cutoff = sessions[389]
    result = reconstruct_unadjusted(source, "CN", cutoff)

    assert result["fit"]["gamma"] is not None
    assert result["fit"]["hits"] == result["fit"]["window_rows"] == 60
    np.testing.assert_allclose(result["close"], raw[:390], rtol=1e-12)
    np.testing.assert_allclose(result["open"], raw[:390] * 0.99, rtol=1e-12)
    assert result["split_ratio_after_cutoff"] == 3.0
    # El volumen del proveedor está multiplicado por los splits posteriores.
    np.testing.assert_allclose(result["volume"][:120], 1000 / 9.0)
    np.testing.assert_allclose(result["volume"][300:], 1000 / 3.0)
    # Dividendo por acción negociada en su fecha, sin dividir por el split del mismo día.
    np.testing.assert_allclose(result["raw_dividend"][[30, 300]], [0.22, 0.35])

    changed = adjusted.copy()
    changed[390:] *= 7.0
    altered = history(sessions, changed, published, splits)
    again = reconstruct_unadjusted(altered, "CN", cutoff)
    np.testing.assert_array_equal(again["close"], result["close"])


def test_float32_provider_values_stay_on_grid_within_declared_tolerance(market_path):
    sessions, raw, dividends, splits = market_path
    adjusted, published = provider_series(raw, dividends, splits, float32=True)
    result = reconstruct_unadjusted(
        history(sessions, adjusted, published, splits), "CN", sessions[389]
    )
    on_grid, chance = price_grid("CN", sessions[:390], result["close"])
    assert on_grid.all()
    assert np.nanmax(np.abs(result["close"] / raw[:390] - 1)) < 2e-7
    assert grid_concordance(on_grid, chance)["excess"] == pytest.approx(1.0)


@pytest.mark.parametrize("shift", [-1, 1])
def test_shifted_ex_dates_are_detected_as_negative_controls(market_path, shift):
    sessions, raw, dividends, splits = market_path
    adjusted, published = provider_series(raw, dividends, splits)
    wrong = history(sessions, adjusted, shifted_events(published, shift), splits)
    result = reconstruct_unadjusted(wrong, "CN", sessions[389])
    early = sessions[:230] < np.datetime64(result["fit"]["first_fit_session"])
    on_grid, chance = price_grid("CN", sessions[:230][early], result["close"][:230][early])
    summary = grid_concordance(on_grid, chance)
    assert summary["rate"] < 0.2
    assert summary["excess"] < 0.2


def test_ignoring_dividends_breaks_concordance_before_the_last_ex_date(market_path):
    sessions, raw, dividends, splits = market_path
    adjusted, published = provider_series(raw, dividends, splits)
    result = reconstruct_unadjusted(
        history(sessions, adjusted, np.zeros_like(published), splits), "CN", sessions[389]
    )
    on_grid, _ = price_grid("CN", sessions[:300], result["close"][:300])
    assert on_grid.mean() < 0.2


def test_split_multipliers_use_only_strictly_later_ratios():
    np.testing.assert_allclose(split_multipliers([0, 2, 0, 0.5, 4]), [4.0, 2.0, 2.0, 4.0, 1.0])
    with pytest.raises(ValueError):
        split_multipliers([1, -2])


def test_dividend_offsets_skip_first_row_and_propagate_invalid_previous_closes():
    close = np.array([10.0, 10.0, np.nan, 20.0, 20.0])
    offsets = dividend_offsets(close, [5.0, 1.0, 0.0, 2.0, 1.0])
    np.testing.assert_allclose(offsets[3:], [0.05, 0.0])
    assert np.isnan(offsets[:3]).all()
    clean = dividend_offsets([10.0, 10.0, 20.0], [5.0, 1.0, 2.0])
    np.testing.assert_allclose(clean, [1 / 10 + 2 / 10, 2 / 10, 0.0])


def test_us_increments_follow_the_decimalization_schedule_and_subdollar_rule():
    sessions = np.array(
        ["2000-08-25", "2000-08-28", "2001-04-06", "2001-04-09", "2015-01-02"],
        dtype="datetime64[D]",
    )
    ticks = price_increments("US", sessions, [50.0, 50.0, 50.0, 50.0, 0.5])
    np.testing.assert_allclose(
        ticks[:, 0], [US_FRACTION_TICK] * 3 + [US_CENT_TICK, US_SUBDOLLAR_TICK]
    )
    assert np.isnan(ticks[[0, 3, 4], 1]).all()
    np.testing.assert_allclose(ticks[1:3, 1], US_CENT_TICK)
    cn = price_increments("CN", sessions[:1], [3.0])
    assert cn[0, 0] == CN_A_SHARE_TICK
    with pytest.raises(ValueError):
        price_increments("HK", sessions[:1], [3.0])


def test_price_grid_accepts_fractions_only_where_they_were_quoted():
    sessions = np.array(["2000-03-01", "2000-12-01", "2001-05-01", "2001-05-01"], "datetime64[D]")
    on_grid, chance = price_grid("US", sessions, [111.9375, 21.78, 111.9375, 0.1234])
    assert on_grid.tolist() == [True, True, False, True]
    np.testing.assert_allclose(chance[0], 2 * 2e-6 * 111.9375 / US_FRACTION_TICK)
    assert grid_concordance([True, False], [1.0, 1.0])["excess"] is None
    assert grid_concordance([], [])["rows"] == 0
    with pytest.raises(ValueError):
        price_grid("US", sessions, [1, 2, 3, 4], relative_tolerance=0)


def test_terminal_fit_rejects_short_windows_and_series_without_grid():
    sessions = np.datetime64("2023-01-02") + np.arange(80) * np.timedelta64(1, "D")
    rng = np.random.default_rng(3)
    noise = 30 + rng.random(80)
    fit = fit_terminal_constant("CN", sessions, noise, np.zeros(80))
    assert fit["gamma"] is None and fit["reason"] == "no_grid_solution"
    short = fit_terminal_constant("CN", sessions[:10], np.full(10, 30.0), np.zeros(10))
    assert short["gamma"] is None and short["reason"] == "insufficient_rows"


def test_terminal_fit_accepts_gamma_one_despite_storage_rounding():
    sessions = np.datetime64("2023-01-02") + np.arange(70) * np.timedelta64(1, "D")
    cents = np.arange(3000, 3070) / 100.0
    stored = (cents * (1 + 1e-7)).astype(np.float64)
    fit = fit_terminal_constant("CN", sessions, stored, np.zeros(70))
    assert fit["gamma"] == pytest.approx(1.0, abs=1e-6)


def test_provider_history_rejects_unordered_sessions_and_missing_events(tmp_path):
    path = tmp_path / "a.csv"
    header = "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
    path.write_text(
        header + "2020-01-03 00:00:00-05:00,1,2,1,2,10,0.0,0.0\n"
        "2020-01-02 00:00:00-05:00,1,2,1,2,10,0.0,0.0\n"
    )
    with pytest.raises(ValueError, match="crecientes"):
        read_provider_history(path)
    path.write_text(header + "2020-01-02 00:00:00-05:00,1,2,1,2,10,,0.0\n")
    with pytest.raises(ValueError, match="ausente"):
        read_provider_history(path)
    path.write_text("Date,Open,High,Low,Close,Volume\n2020-01-02,1,2,1,2,10\n")
    with pytest.raises(Exception):  # noqa: B017 - pyarrow informa la columna ausente.
        read_provider_history(path)
    path.write_text(header + "2020-01-02 00:00:00-05:00,1,2,1,,10,0.5,0.0\n")
    parsed = read_provider_history(path)
    assert np.isnan(parsed.close[0]) and parsed.dividends[0] == 0.5


def test_terminal_fit_identifies_gamma_for_high_prices_stored_in_float32():
    """Con precios altos, la tolerancia de validación no separa candidatos contiguos."""
    rng = np.random.default_rng(7)
    rows = 300
    sessions = np.datetime64("2022-01-03") + np.arange(rows) * np.timedelta64(1, "D")
    raw = (170_000 + np.cumsum(rng.integers(-900, 901, rows))) / 100.0
    gamma = 1.0371
    stored = (raw / gamma).astype(np.float32).astype(np.float64)
    fit = fit_terminal_constant("CN", sessions, stored, np.zeros(rows))
    assert fit["gamma"] == pytest.approx(gamma, rel=1e-6)
    recovered = stored * fit["gamma"]
    np.testing.assert_allclose(recovered, raw, rtol=3e-7)
