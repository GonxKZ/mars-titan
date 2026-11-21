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


def test_the_second_version_keeps_the_first_bit_for_bit_and_adds_its_columns():
    _, sessions, table, market = market_prices(seed=5)
    first = technical.technical_features(table, sessions, sessions, market)
    second = technical.technical_features(
        table, sessions, sessions, market, version=technical.NAME_V2
    )
    names = technical.VERSIONS[technical.NAME_V2]
    assert names[: technical.WIDTH] == technical.NAMES and len(names) == technical.WIDTH + 16
    assert second.shape == (len(table), len(names)) and second.dtype == np.float32
    assert np.array_equal(second[:, : technical.WIDTH], first)
    assert np.isfinite(second).all() and np.abs(second).max() <= technical.LIMIT
    # Sin factor de mercado, los rasgos que lo necesitan quedan a cero y los demás no cambian.
    alone = technical.technical_features(table, sessions, version=technical.NAME_V2)
    market_based = [
        "realized_label_1",
        "realized_label_5",
        "realized_label_21",
        "alpha_21_percent",
        "alpha_63_percent",
    ]
    for name in names[technical.WIDTH :]:
        column = names.index(name)
        if name in market_based:
            assert not alone[:, column].any() and second[300:, column].any()
        else:
            assert np.array_equal(alone[:, column], second[:, column])
    with pytest.raises(ValueError, match="versión"):
        technical.technical_features(table, sessions, version="long_horizon_v9")


def test_the_second_version_is_causal_row_by_row():
    _, sessions, table, market = market_prices(seed=6)
    cut = 700
    complete = technical.technical_features(
        table, sessions, sessions, market, version=technical.NAME_V2
    )
    changed, other = table.copy(), market.copy()
    changed[cut:] *= np.linspace(1.5, 0.4, len(table) - cut)[:, None]
    other[cut:] = -other[cut:] + 0.01
    after = technical.technical_features(
        changed, sessions, sessions, other, version=technical.NAME_V2
    )
    # Cambiar precios y factor desde una sesión no cambia ninguna fila anterior.
    assert np.array_equal(after[:cut], complete[:cut])
    assert not np.array_equal(after[cut:], complete[cut:])
    truncated = technical.technical_features(
        table[:cut], sessions[:cut], sessions[:cut], market[:cut], version=technical.NAME_V2
    )
    assert np.array_equal(truncated, complete[:cut])


def test_the_realized_label_is_the_label_of_the_previous_session():
    clock, sessions, table, market = market_prices(seed=7)
    features = technical.technical_features(
        table, sessions, sessions, market, version=technical.NAME_V2
    )
    names = technical.VERSIONS[technical.NAME_V2]
    days = [day.isoformat() for day in clock.days]
    base = dict(session=days, available_at=clock.decisions)
    market_frame = pd.DataFrame({**base, "open": 100.0, "close": 100 * (1 + market)})
    asset_frame = pd.DataFrame({**base, "open": table[:, 0], "close": table[:, 3]})
    labels = residual_targets(asset_frame, market_frame, clock, cutoff=days[-1])
    labels = labels[labels.reason == "accepted"].reset_index(drop=True)
    position = {day: row for row, day in enumerate(days)}
    alpha, beta, residual, _ = technical.market_model(table[:, 3] / table[:, 0] - 1, market)
    checked = 0
    for _, label in labels.iloc[::29].iterrows():
        row = position[label.prediction_at.date().isoformat()]
        # La etiqueta de la predicción en `row` se realiza al cierre de la sesión siguiente.
        expected = label.target / residual[row]
        assert abs(expected) < technical.LIMIT
        assert features[row + 1, names.index("realized_label_1")] == pytest.approx(
            expected, rel=2e-5, abs=1e-6
        )
        checked += 1
    assert checked > 25
    # Las sumas de 5 y 21 sesiones son las de la última etiqueta realizada.
    one = features[:, names.index("realized_label_1")].astype(np.float64)
    row = 600
    assert features[row, names.index("realized_label_5")] == pytest.approx(
        one[row - 4 : row + 1].sum() / np.sqrt(5), rel=1e-4, abs=1e-5
    )
    assert features[row, names.index("realized_label_21")] == pytest.approx(
        one[row - 20 : row + 1].sum() / np.sqrt(21), rel=1e-4, abs=1e-5
    )
    free = table[:, 3] / table[:, 0] - 1 - beta * market
    assert features[row, names.index("alpha_63_percent")] == pytest.approx(
        np.nansum(free[row - 62 : row + 1]) / 63 * 100, rel=1e-4, abs=1e-6
    )


def test_the_added_price_features_have_their_known_values():
    sessions = np.arange("2020-01-01", "2021-06-01", dtype="datetime64[D]")
    rows = len(sessions)
    close = 20 * np.exp(0.001 * np.arange(rows))
    opened = close * np.exp(-0.002)
    high, low = close * 1.01, opened * 0.99
    volume = np.full(rows, 1e6)
    volume[-10:] = 0.0
    table = np.column_stack([opened, high, low, close, volume])
    features = technical.technical_features(table, sessions, version=technical.NAME_V2)
    names = technical.VERSIONS[technical.NAME_V2]
    last = features[-1]
    # Cada sesión sube un 0,2 % de apertura a cierre y baja un 0,1 % de cierre a apertura.
    assert last[names.index("intraday_mean_252_percent")] == pytest.approx(0.2, rel=1e-4)
    assert last[names.index("overnight_mean_252_percent")] == pytest.approx(-0.1, rel=1e-4)
    assert last[names.index("untraded_63")] == pytest.approx(10 / 63, rel=1e-5)
    expected = (2 * close[-1] - high[-1] - low[-1]) / (high[-1] - low[-1])
    assert last[names.index("close_location_1")] == pytest.approx(expected, rel=1e-5)
    assert last[names.index("close_location_5")] == pytest.approx(expected, rel=1e-4)
    # Sin historia suficiente las medias valen cero.
    assert features[100, names.index("intraday_mean_252_percent")] == 0
    assert features[3, names.index("close_location_5")] == 0
    assert features[61, names.index("untraded_63")] == 0
