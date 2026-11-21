"""Rasgos técnicos de horizonte largo: causalidad, valores conocidos y el alfa de la etiqueta."""

import numpy as np
import pandas as pd
import pytest

from mars_titan.data import technical_features as technical
from mars_titan.data.budget_targets import residual_targets
from mars_titan.data.temporal import MarketClock


def market_prices(seed=0, start="2019-01-01", end="2023-12-31"):
    """Precios sintéticos del factor y de un activo sobre el calendario de EE. UU."""
    clock = MarketClock("US", start, end)
    random = np.random.default_rng(seed)
    count = len(clock.days)
    sessions = np.array([day.isoformat() for day in clock.days], dtype="datetime64[D]")
    market_intraday = random.normal(0.0002, 0.01, count)
    asset_intraday = 0.001 + 1.3 * market_intraday + random.normal(0, 0.012, count)
    close = 50 * np.exp(np.cumsum(random.normal(0.0003, 0.02, count)))
    opened = close / (1 + asset_intraday)
    table = np.column_stack(
        [
            opened,
            np.maximum(opened, close) * 1.01,
            np.minimum(opened, close) * 0.99,
            close,
            random.uniform(1e5, 5e6, count),
        ]
    )
    return clock, sessions, table, market_intraday


def test_the_market_model_is_the_one_of_the_label(tmp_path):
    clock, sessions, table, market = market_prices()
    # El activo no cotiza en unas sesiones: el OLS usa los pares que quedan, como la etiqueta.
    kept = np.ones(len(sessions), dtype=bool)
    kept[[40, 41, 300, 555]] = False
    features = technical.technical_features(table[kept], sessions[kept], sessions, market)
    days = [day.isoformat() for day in clock.days]
    base = dict(session=days, available_at=clock.decisions)
    market_frame = pd.DataFrame({**base, "open": 100.0, "close": 100 * (1 + market)})
    asset_frame = pd.DataFrame(
        dict(
            session=np.array(days)[kept],
            available_at=np.array(clock.decisions, dtype=object)[kept],
            open=table[kept, 0],
            close=table[kept, 3],
        )
    )
    labels = residual_targets(asset_frame, market_frame, clock, cutoff=days[-1])
    labels = labels[labels.reason == "accepted"].reset_index(drop=True)
    assert len(labels) > 800
    position = {day: row for row, day in enumerate(np.array(days)[kept])}
    names = technical.NAMES
    for _, label in labels.iloc[::37].iterrows():
        row = position[label.prediction_at.date().isoformat()]
        assert features[row, names.index("has_market_model")] == 1
        assert features[row, names.index("alpha_percent")] == pytest.approx(
            label.alpha * 100, rel=2e-5, abs=1e-6
        )
        assert features[row, names.index("beta")] == pytest.approx(label.beta, rel=2e-5)
    # Antes de reunir 126 pares no hay modelo de mercado: ceros con su indicador apagado.
    early = features[:100, [names.index(n) for n in ("alpha_percent", "beta", "has_market_model")]]
    assert not early.any()


def test_no_row_depends_on_later_prices_or_on_a_later_market():
    _, sessions, table, market = market_prices(seed=1)
    reference = technical.technical_features(table, sessions, sessions, market)
    cut = 700
    changed, later = table.copy(), market.copy()
    changed[cut:] *= np.linspace(1.5, 3.0, len(table) - cut)[:, None]
    later[cut:] = -later[cut:] + 0.03
    again = technical.technical_features(changed, sessions, sessions, later)
    assert np.array_equal(reference[:cut], again[:cut])
    assert not np.array_equal(reference[cut:], again[cut:])
    # Con menos historia delante, las filas comunes tampoco cambian.
    short = technical.technical_features(table[:cut], sessions[:cut], sessions[:cut], market[:cut])
    assert np.array_equal(short, reference[:cut])


def test_known_values_flags_and_bounds():
    _, sessions, table, market = market_prices(seed=2)
    features = technical.technical_features(table, sessions, sessions, market)
    names = technical.NAMES
    assert features.shape == (len(table), technical.WIDTH) and features.dtype == np.float32
    assert np.isfinite(features).all() and np.abs(features).max() <= technical.LIMIT
    close, row = table[:, 3], 600
    daily = np.diff(np.log(close))
    year = daily[row - 252 : row].std()
    expected = np.log(close[row] / close[row - 21]) / (year * np.sqrt(21))
    assert features[row, names.index("return_21")] == pytest.approx(expected, rel=1e-5)
    low = np.log(close[row] / close[row - 251 : row + 1].min())
    assert features[row, names.index("distance_to_low_252")] == pytest.approx(low, abs=1e-6)
    flags = features[:, [names.index("has_63_sessions"), names.index("has_252_sessions")]]
    assert flags[62].tolist() == [0, 0] and flags[63].tolist() == [1, 0]
    assert flags[251].tolist() == [1, 0] and flags[252].tolist() == [1, 1]
    # Sin historia suficiente, los rasgos de un año valen cero.
    assert features[100, names.index("return_252")] == 0
    assert features[100, names.index("momentum_12_1")] == 0


def test_without_a_market_factor_only_the_market_model_is_missing():
    _, sessions, table, market = market_prices(seed=3)
    with_factor = technical.technical_features(table, sessions, sessions, market)
    without = technical.technical_features(table, sessions)
    names = technical.NAMES
    model = [names.index(n) for n in names[:5]]
    rest = [index for index in range(technical.WIDTH) if index not in model]
    assert not without[:, model].any() and with_factor[:, model].any()
    assert np.array_equal(without[:, rest], with_factor[:, rest])


@pytest.mark.parametrize("problem", ["shape", "order", "length"])
def test_invalid_tables_are_rejected(problem):
    _, sessions, table, _ = market_prices(seed=4)
    if problem == "shape":
        table = table[:, :4]
    elif problem == "order":
        sessions = sessions[::-1]
    else:
        sessions = sessions[:-1]
    with pytest.raises(ValueError, match="rasgos técnicos"):
        technical.technical_features(table, sessions)
