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


def test_an_independent_implementation_gives_the_same_features():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).with_name("technical_features_reference.py")
    spec = importlib.util.spec_from_file_location("technical_features_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    names = technical.VERSIONS[technical.NAME_V2]
    assert tuple(reference.NOMBRES) == names
    worst = np.zeros(len(names))
    for seed in range(4):
        _, sessions, table, market = market_prices(seed=20 + seed)
        random = np.random.default_rng(seed)
        # Sesiones sin cotizar, volumen nulo y sesiones sin rango: los casos de los datos reales.
        kept = np.ones(len(sessions), dtype=bool)
        kept[random.choice(len(sessions), size=25, replace=False)] = False
        prices = table[kept].copy()
        still = random.choice(len(prices), size=15, replace=False)
        prices[still, 4] = 0.0
        prices[still[:5], :4] = prices[still[:5], 3:4]
        factor = market.copy()
        factor[random.choice(len(factor), size=6, replace=False)] = np.nan
        for calendar, values in ((sessions, factor), (None, None)):
            mine = technical.technical_features(
                prices, sessions[kept], calendar, values, version=technical.NAME_V2
            )
            other = reference.rasgos(prices, sessions[kept], calendar, values)
            assert other.shape == mine.shape and other.dtype == np.float32
            worst = np.maximum(worst, np.abs(mine.astype(float) - other.astype(float)).max(axis=0))
    assert worst.max() < 2e-4, dict(zip(names, worst.round(6).tolist(), strict=True))


# --- long_horizon_v3: los 46 rasgos de la v2 más dieciséis de precio y volumen ---------------

ADDED_V3 = (
    "days_since_high_21",
    "days_since_high_63",
    "days_since_high_252",
    "days_since_low_63",
    "days_since_low_252",
    "up_move_share_cc_14",
    "up_move_share_oc_14",
    "up_day_fraction_cc_21",
    "up_day_fraction_cc_63",
    "up_day_fraction_oc_63",
    "price_volume_corr_63",
    "volume_cv_21",
    "move_weighted_volume_cv_cc_21",
    "move_weighted_volume_cv_oc_21",
    "candle_upper_shadow_5",
    "candle_upper_shadow_21",
)


def rough_prices(rows=420, seed=7):
    """Precios con sesiones sin volumen, velas sin rango y un tramo de máximos repetidos."""
    generator = np.random.default_rng(seed)
    close = 20.0 * np.exp(np.cumsum(generator.normal(0.0, 0.02, rows)))
    opened = close * np.exp(generator.normal(0.0, 0.01, rows))
    high = np.maximum(opened, close) * (1.0 + np.abs(generator.normal(0.0, 0.006, rows)))
    low = np.minimum(opened, close) * (1.0 - np.abs(generator.normal(0.0, 0.006, rows)))
    volume = np.exp(generator.normal(12.0, 0.6, rows))
    volume[generator.choice(rows, 12, replace=False)] = 0.0
    flat = generator.choice(np.arange(5, rows), 6, replace=False)
    opened[flat] = high[flat] = low[flat] = close[flat]
    if rows > 320:
        # Un máximo repetido: el empate lo gana la sesión más reciente.
        high[300] = high[305] = high[280:320].max() * 1.01
    sessions = pd.bdate_range("2012-01-02", periods=rows).values.astype("datetime64[D]")
    return np.column_stack([opened, high, low, close, volume]), sessions


def _reference_v3(prices):
    """Los dieciséis rasgos añadidos, fila a fila y sin vectorizar."""
    import math

    rows = len(prices)
    opened, high, low, close, volume = (prices[:, k].tolist() for k in range(5))
    nan = float("nan")
    cc = [nan] + [math.log(close[t] / close[t - 1]) for t in range(1, rows)]
    oc = [math.log(close[t] / opened[t]) for t in range(rows)]
    upper = []
    for t in range(rows):
        span = high[t] - low[t]
        top = (high[t] - max(opened[t], close[t])) / span if span > 0 else nan
        upper.append(min(max(top, 0.0), 1.0) if span > 0 else nan)

    def needed(window):
        return math.ceil(0.75 * window)

    def window_of(series, t, window, first=0):
        if t - window + 1 < first:
            return None
        return series[t - window + 1 : t + 1]

    def since(series, t, window, largest):
        values = window_of(series, t, window)
        if values is None:
            return nan
        best, age = None, None
        for back, value in enumerate(reversed(values)):
            if best is None or (value > best if largest else value < best):
                best, age = value, back
        return 2.0 * age / (window - 1) - 1.0

    def share(series, t, window, first):
        values = window_of(series, t, window, first)
        if values is None:
            return nan
        total = sum(abs(v) for v in values)
        return 2.0 * sum(max(v, 0.0) for v in values) / total - 1.0 if total > 0 else nan

    def fraction(series, t, window, first):
        values = window_of(series, t, window, first)
        if values is None:
            return nan
        return 2.0 * sum(v > 0 for v in values) / len(values) - 1.0

    def correlation(left, right, t, window):
        a, b = window_of(left, t, window), window_of(right, t, window)
        if a is None:
            return nan
        ma, mb = sum(a) / window, sum(b) / window
        saa = sum((x - ma) ** 2 for x in a)
        sbb = sum((x - mb) ** 2 for x in b)
        sab = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
        return sab / math.sqrt(saa * sbb) if saa > 0 and sbb > 0 else nan

    def variation(series, t, window, first, neutral):
        values = window_of(series, t, window, first)
        if values is None:
            return nan
        mean = sum(values) / window
        if mean <= 0:
            return nan
        deviation = math.sqrt(sum((v - mean) ** 2 for v in values) / (window - 1))
        return math.log(deviation / mean / neutral) if deviation > 0 else nan

    def shadow(t, window):
        values = window_of(upper, t, window)
        if values is None:
            return nan
        known = [v for v in values if not math.isnan(v)]
        if len(known) < needed(window):
            return nan
        return 4.0 * (sum(known) / len(known) - 0.25)

    log_close = [math.log(c) for c in close]
    log_volume = [math.log1p(v) for v in volume]
    moved_cc = [nan] + [abs(cc[t]) * volume[t] for t in range(1, rows)]
    moved_oc = [abs(oc[t]) * volume[t] for t in range(rows)]
    table = np.full((rows, len(ADDED_V3)), np.nan)
    for t in range(rows):
        table[t] = [
            since(high, t, 21, True),
            since(high, t, 63, True),
            since(high, t, 252, True),
            since(low, t, 63, False),
            since(low, t, 252, False),
            share(cc, t, 14, 1),
            share(oc, t, 14, 0),
            fraction(cc, t, 21, 1),
            fraction(cc, t, 63, 1),
            fraction(oc, t, 63, 0),
            correlation(log_close, log_volume, t, 63),
            variation(volume, t, 21, 0, 0.5),
            variation(moved_cc, t, 21, 1, 1.0),
            variation(moved_oc, t, 21, 0, 1.0),
            shadow(t, 5),
            shadow(t, 21),
        ]
    return np.clip(np.where(np.isfinite(table), table, 0.0), -8.0, 8.0).astype(np.float32)


def test_the_third_version_keeps_the_second_bit_for_bit_and_adds_sixteen_columns():
    assert technical.NAMES_V3[:46] == technical.NAMES_V2 and technical.NAMES_V3[46:] == ADDED_V3
    assert technical.VERSIONS[technical.NAME_V3] is technical.NAMES_V3
    _, sessions, table, market = market_prices(seed=5)
    rough, days = rough_prices()
    # Con el factor de mercado y sin él, y con precios limpios y con huecos.
    for prices, when, factor in ((table, sessions, (sessions, market)), (rough, days, ())):
        second = technical.technical_features(prices, when, *factor, version=technical.NAME_V2)
        third = technical.technical_features(prices, when, *factor, version=technical.NAME_V3)
        assert third.shape == (len(prices), 62) and third.dtype == np.float32
        assert np.array_equal(third[:, :46], second)
        assert np.isfinite(third).all() and np.abs(third).max() <= technical.LIMIT
        # Las dos primeras versiones no cambian al existir la tercera.
        first = technical.technical_features(prices, when, *factor)
        assert np.array_equal(second[:, :30], first)


def test_the_third_version_matches_a_row_by_row_reference():
    prices, sessions = rough_prices()
    third = technical.technical_features(prices, sessions, version=technical.NAME_V3)
    expected = _reference_v3(prices)
    for column, name in enumerate(ADDED_V3):
        np.testing.assert_allclose(
            third[:, 46 + column], expected[:, column], rtol=2e-5, atol=2e-6, err_msg=name
        )
    # Ninguna columna está muerta: con historia suficiente todas varían.
    assert all(third[260:, 46 + column].std() > 1e-3 for column in range(len(ADDED_V3)))


def test_the_third_version_is_causal_row_by_row():
    prices, sessions = rough_prices()
    whole = technical.technical_features(prices, sessions, version=technical.NAME_V3)
    for cut in (30, 63, 252, 300, 419):
        changed = prices.copy()
        changed[cut:] *= np.random.default_rng(cut).uniform(0.5, 2.0, (len(prices) - cut, 5))
        changed[cut:, 1] = np.maximum(changed[cut:, 1], changed[cut:, [0, 3]].max(axis=1))
        changed[cut:, 2] = np.minimum(changed[cut:, 2], changed[cut:, [0, 3]].min(axis=1))
        again = technical.technical_features(changed, sessions, version=technical.NAME_V3)
        assert np.array_equal(again[:cut], whole[:cut]), cut
        prefix = technical.technical_features(
            prices[:cut], sessions[:cut], version=technical.NAME_V3
        )
        assert np.array_equal(prefix, whole[:cut]), cut


def test_the_third_version_is_zero_without_history_range_or_volume():
    prices, sessions = rough_prices(rows=300)
    third = technical.technical_features(prices, sessions, version=technical.NAME_V3)
    index = {name: 46 + column for column, name in enumerate(ADDED_V3)}
    # Sin la ventana completa no hay valor.
    assert (third[:20, index["days_since_high_21"]] == 0).all()
    assert (third[:251, index["days_since_high_252"]] == 0).all()
    assert (third[:14, index["up_move_share_cc_14"]] == 0).all()
    assert (third[:13, index["up_move_share_oc_14"]] == 0).all()
    assert (
        third[20, index["days_since_high_21"]] != 0 or third[21, index["days_since_high_21"]] != 0
    )
    # Precio constante y sin volumen: ni velas, ni movimientos, ni variabilidad.
    still = np.tile(np.array([10.0, 10.0, 10.0, 10.0, 0.0]), (300, 1))
    quiet = technical.technical_features(still, sessions, version=technical.NAME_V3)
    for name in (
        "up_move_share_cc_14",
        "price_volume_corr_63",
        "volume_cv_21",
        "candle_upper_shadow_21",
    ):
        assert (quiet[:, index[name]] == 0).all(), name
    # El máximo de hoy es la distancia cero: −1 tras centrar.
    rising = np.column_stack([np.linspace(10, 20, 300)] * 4 + [np.full(300, 1e6)])
    rising[:, 1] += 0.5
    rising[:, 2] -= 0.5
    up = technical.technical_features(rising, sessions, version=technical.NAME_V3)
    assert (up[251:, index["days_since_high_252"]] == -1.0).all()
    assert (up[251:, index["days_since_low_252"]] == 1.0).all()


def test_the_third_version_needs_three_quarters_of_each_window():
    _, sessions, prices, _ = market_prices(seed=11)
    prices, sessions = prices[:300].copy(), sessions[:300]
    # Velas con sombra superior distinta de un cuarto del rango, todas válidas.
    prices[:, 1] = np.maximum(prices[:, 0], prices[:, 3]) * 1.02
    index = {name: 46 + column for column, name in enumerate(ADDED_V3)}
    clean = technical.technical_features(prices, sessions, version=technical.NAME_V3)
    assert (clean[4:, index["candle_upper_shadow_5"]] != 0).all()
    # Dos velas sin rango en cinco sesiones: quedan tres sombras válidas y hacen falta cuatro.
    holed = prices.copy()
    for row in (200, 202):
        holed[row, :4] = holed[row, 3]
    third = technical.technical_features(holed, sessions, version=technical.NAME_V3)
    assert (third[202:205, index["candle_upper_shadow_5"]] == 0).all()
    assert third[206, index["candle_upper_shadow_5"]] != 0
    # Una sola vela sin rango sí deja valor: cuatro de cinco.
    single = prices.copy()
    single[200, :4] = single[200, 3]
    one = technical.technical_features(single, sessions, version=technical.NAME_V3)
    assert (one[200:205, index["candle_upper_shadow_5"]] != 0).all()
    # Con seis precios no válidos de catorce el reparto de subidas no se calcula (hacen falta once).
    broken = prices.copy()
    broken[150:156, 3] = np.nan
    gaps = technical.technical_features(broken, sessions, version=technical.NAME_V3)
    assert gaps[160, index["up_move_share_oc_14"]] == 0
    assert gaps[175, index["up_move_share_oc_14"]] != 0


def _third(prices, sessions):
    return technical.technical_features(prices, sessions, version=technical.NAME_V3)


def test_the_third_version_counts_valid_rows_at_the_exact_boundary():
    _, sessions, prices, _ = market_prices(seed=11)
    prices, sessions = prices[:300].copy(), sessions[:300]
    index = {name: 46 + column for column, name in enumerate(ADDED_V3)}
    # Once de catorce bastan y diez no: tres cuartos de la ventana, redondeados hacia arriba.
    for missing, known in ((3, True), (4, False)):
        holed = prices.copy()
        holed[150 : 150 + missing, 3] = np.nan
        value = _third(holed, sessions)[163, index["up_move_share_oc_14"]]
        assert (value != 0) == known, missing
    # Con el volumen, dieciséis de veintiuno: un volumen negativo no es un dato y cuenta como
    # ausente, igual que un precio que no es positivo.
    for missing, known in ((5, True), (6, False)):
        holed = prices.copy()
        holed[190 : 190 + missing, 4] = -1.0
        value = _third(holed, sessions)[205, index["volume_cv_21"]]
        assert (value != 0) == known, missing


def test_the_third_version_needs_the_price_of_the_session_and_breaks_ties_to_the_latest():
    _, sessions, prices, _ = market_prices(seed=11)
    prices, sessions = prices[:300].copy(), sessions[:300]
    index = {name: 46 + column for column, name in enumerate(ADDED_V3)}
    high, low = index["days_since_high_21"], index["days_since_low_63"]
    spiked = prices.copy()
    spiked[195, 1] *= 1.5
    whole = _third(spiked, sessions)
    assert whole[200, high] == pytest.approx(-0.5) and whole[201, high] == pytest.approx(-0.4)
    # Sin el precio de hoy no hay «días desde el máximo», aunque la ventana tenga datos de sobra.
    for bad in (np.nan, 0.0, -5.0):
        holed = spiked.copy()
        holed[200, 1] = bad
        third = _third(holed, sessions)
        assert third[200, high] == 0, bad
        assert third[201, high] == pytest.approx(-0.4), bad
    # Dos mínimos iguales: cuenta el más reciente, a siete sesiones.
    tied = prices.copy()
    tied[195, 2] = tied[198, 2] = tied[150:230, 2].min() * 0.5
    assert _third(tied, sessions)[205, low] == pytest.approx(2 * 7 / 62 - 1)


def test_the_third_version_on_a_window_without_movement_or_with_rounding_noise():
    sessions = pd.bdate_range("2012-01-02", periods=300).values.astype("datetime64[D]")
    index = {name: 46 + column for column, name in enumerate(ADDED_V3)}
    # Sin movimiento no falta historia: el extremo «es hoy» y ninguna sesión subió. Es la
    # firma de un activo suspendido. El resto de columnas no tiene nada que medir.
    still = np.tile(np.array([10.0, 10.0, 10.0, 10.0, 0.0]), (300, 1))
    quiet = _third(still, sessions)
    for name in ADDED_V3:
        column = quiet[252:, index[name]]
        expected = -1.0 if name.startswith(("days_since", "up_day_fraction")) else 0.0
        assert (column == expected).all(), name
    # Un volumen constante no tiene variabilidad, se sume como se sume.
    for level in (0.3, 0.1, 0.7, 1 / 3, 1e6 / 7):
        constant = still.copy()
        constant[:, 4] = level
        assert (_third(constant, sessions)[:, index["volume_cv_21"]] == 0).all(), level
    # Un rango del tamaño del redondeo no es una vela, ni un movimiento así una subida.
    noisy = np.tile(np.array([10.0, 10.0, 10.0, 10.0, 1e6]), (300, 1))
    noisy[:, 1] = np.nextafter(10.0, 11.0)
    assert (_third(noisy, sessions)[:, index["candle_upper_shadow_5"]] == 0).all()
    creeping = np.full(300, 10.0)
    for row in range(1, 300):
        creeping[row] = np.nextafter(creeping[row - 1], np.inf)
    crept = np.column_stack([creeping] * 4 + [np.full(300, 1e6)])
    assert (_third(crept, sessions)[:, index["up_move_share_cc_14"]] == 0).all()


@pytest.mark.parametrize("rows", [1, 2, 4])
def test_the_third_version_on_a_series_shorter_than_every_window(rows):
    _, sessions, prices, _ = market_prices(seed=11)
    third = _third(prices[:rows], sessions[:rows])
    assert third.shape == (rows, 62) and (third[:, 46:] == 0).all()


def test_the_third_version_reads_rounding_noise_in_an_untraded_stretch_as_no_movement():
    _, sessions, prices, _ = market_prices(seed=11)
    prices, sessions = prices[:300].copy(), sessions[:300]
    index = {name: 46 + column for column, name in enumerate(ADDED_V3)}
    # Cien sesiones sin negociar cuyos precios ajustados bailan en el último bit, como en los
    # datos reales: la apertura, el máximo, el mínimo y el cierre valen el nivel o el número
    # siguiente.
    level = prices[199, 3]
    noise = np.random.default_rng(3).integers(0, 2, size=(100, 4)).astype(bool)
    prices[200:, :4] = np.where(noise, np.nextafter(level, np.inf), level)
    prices[200:, 4] = 0.0
    third = _third(prices, sessions)
    # Con la ventana entera dentro del tramo: ninguna subida y el extremo es hoy.
    still = ("up_day_fraction_oc_63", "up_day_fraction_cc_63", "up_day_fraction_cc_21")
    still += ("days_since_high_21", "days_since_high_63", "days_since_low_63")
    for name in still:
        window = int(name.rsplit("_", 1)[1])
        assert (third[200 + window :, index[name]] == -1.0).all(), name
    # Y nada que medir en los movimientos, las velas, la variabilidad ni la correlación.
    empty = ("up_move_share_oc_14", "up_move_share_cc_14", "candle_upper_shadow_5")
    empty += ("candle_upper_shadow_21", "volume_cv_21", "move_weighted_volume_cv_oc_21")
    empty += ("move_weighted_volume_cv_cc_21", "price_volume_corr_63")
    for name in empty:
        window = int(name.rsplit("_", 1)[1])
        assert (third[200 + window :, index[name]] == 0).all(), name
    # Con negociación pero el precio clavado: el ruido del último bit no se correlaciona con
    # el volumen, que sí varía.
    pinned = prices.copy()
    pinned[200:, 4] = np.random.default_rng(4).uniform(1e5, 5e6, 100)
    traded = _third(pinned, sessions)
    assert (traded[263:, index["price_volume_corr_63"]] == 0).all()
    assert (traded[221:, index["volume_cv_21"]] != 0).all()
    assert (traded[263:, index["up_day_fraction_oc_63"]] == -1.0).all()


# --- long_horizon_v4: mismas 62 columnas que v3, sin anular historia 63..251 ni días sin volumen


def _independent_v4_volatility_features(prices):
    """Cálculo independiente, fila a fila, de los rasgos normalizados por volatilidad en v4."""
    import math

    rows = len(prices)
    opened, high, low, close = (prices[:, k].astype(np.float64) for k in range(4))
    log_close = np.log(close)
    daily = np.full(rows, np.nan)
    daily[1:] = log_close[1:] - log_close[:-1]
    filled = np.nan_to_num(daily)
    intraday = np.log(close / opened)
    overnight = np.full(rows, np.nan)
    overnight[1:] = np.log(opened[1:] / close[:-1])
    clean_intraday = np.where(np.isfinite(intraday), intraday, 0.0)
    clean_overnight = np.where(np.isfinite(overnight), overnight, 0.0)

    def clip32(value):
        if not math.isfinite(value):
            return np.float32(0.0)
        return np.float32(min(max(value, -technical.LIMIT), technical.LIMIT))

    expected = {}
    for row in range(63, min(rows, 252)):
        window_returns = filled[1 : row + 1]
        vol = float(np.std(window_returns, ddof=0))
        vol_21 = float(np.std(filled[row - 20 : row + 1], ddof=0))
        vol_63 = float(np.std(filled[row - 62 : row + 1], ddof=0))
        entry = {
            "return_1": clip32(daily[row] / vol),
            "return_5": clip32((log_close[row] - log_close[row - 5]) / (vol * math.sqrt(5))),
            "return_21": clip32((log_close[row] - log_close[row - 21]) / (vol * math.sqrt(21))),
            "return_63": clip32((log_close[row] - log_close[row - 63]) / (vol * math.sqrt(63))),
            "return_126": (
                clip32((log_close[row] - log_close[row - 126]) / (vol * math.sqrt(126)))
                if row >= 126
                else np.float32(0.0)
            ),
            "return_252": np.float32(0.0),
            "momentum_12_1": np.float32(0.0),
            "log_volatility_252": clip32(math.log(vol * 100.0)),
            "volatility_21_over_252": clip32(math.log(vol_21 / vol)),
            "volatility_63_over_252": clip32(math.log(vol_63 / vol)),
            "intraday_1": clip32(intraday[row] / vol),
            "overnight_1": clip32(overnight[row] / vol),
            "range_1": clip32(math.log(high[row] / low[row]) / vol),
            "intraday_5": clip32(clean_intraday[row - 4 : row + 1].mean() / vol * math.sqrt(5)),
            "overnight_5": clip32(clean_overnight[row - 4 : row + 1].mean() / vol * math.sqrt(5)),
            "intraday_21": clip32(clean_intraday[row - 20 : row + 1].mean() / vol * math.sqrt(21)),
            "overnight_21": clip32(
                clean_overnight[row - 20 : row + 1].mean() / vol * math.sqrt(21)
            ),
            "intraday_63": clip32(clean_intraday[row - 62 : row + 1].mean() / vol * math.sqrt(63)),
            "overnight_63": clip32(
                clean_overnight[row - 62 : row + 1].mean() / vol * math.sqrt(63)
            ),
            "intraday_252": (
                clip32(clean_intraday[row - 251 : row + 1].mean() / vol * math.sqrt(252))
                if row >= 251
                else np.float32(0.0)
            ),
            "overnight_252": np.float32(0.0),
            "maximum_return_21": clip32(float(filled[row - 20 : row + 1].max()) / vol),
        }
        expected[row] = entry
    return expected


def test_the_fourth_version_keeps_the_third_schema_and_leaves_earlier_versions_bit_identical():
    import hashlib

    assert technical.NAME_V4 == "long_horizon_v4"
    assert technical.NAMES_V4 == technical.NAMES_V3 and len(technical.NAMES_V4) == 62
    assert technical.VERSIONS[technical.NAME_V4] is technical.NAMES_V4
    _, sessions, table, market = market_prices(seed=5)
    rough, days = rough_prices()
    # Huellas SHA-256 tomadas antes de introducir la v4: v1, v2 y v3 no cambian ni un bit.
    baseline_digests = {
        ("market", technical.NAME): (
            "4c65af7fd14aa86dff7a7337387bc0b1449fea0cf2ecdbf934594a1ca4e653d4"
        ),
        ("market", technical.NAME_V2): (
            "66a20474e2f2d2a8f6d0d04b8a2774810fe7e4c0ea615f83cdcb39ad20e2fe51"
        ),
        ("market", technical.NAME_V3): (
            "8be412e452142868536a2cc28fdb98556d9c43d764923ebf13ca442cb274aa0c"
        ),
        ("rough", technical.NAME): (
            "f11339713a1bf5152ae1c12893cdedc03576121d2083db2c964a61509fd9e57a"
        ),
        ("rough", technical.NAME_V2): (
            "ae3a64003dbb0eb3f522ea3624462188d842dc48d316ebb7b5edb838456f96ca"
        ),
        ("rough", technical.NAME_V3): (
            "ee57be158bd5ee40aa098b366f2448b36319c3e2c486decfaac628b8aeca2c87"
        ),
    }
    for label, prices, when, factor in (
        ("market", table, sessions, (sessions, market)),
        ("rough", rough, days, ()),
    ):
        first = technical.technical_features(prices, when, *factor)
        second = technical.technical_features(prices, when, *factor, version=technical.NAME_V2)
        third = technical.technical_features(prices, when, *factor, version=technical.NAME_V3)
        fourth = technical.technical_features(prices, when, *factor, version=technical.NAME_V4)
        for ver, arr in (
            (technical.NAME, first),
            (technical.NAME_V2, second),
            (technical.NAME_V3, third),
        ):
            assert hashlib.sha256(arr.tobytes()).hexdigest() == baseline_digests[(label, ver)], (
                label,
                ver,
            )
        assert np.array_equal(second[:, :30], first)
        assert np.array_equal(third[:, :46], second)
        assert fourth.shape == (len(prices), 62) and fourth.dtype == np.float32
        assert np.isfinite(fourth).all() and np.abs(fourth).max() <= technical.LIMIT
    # Sin factor de mercado en v4 solo se apagan las diez columnas del modelo de mercado.
    with_factor = technical.technical_features(
        table, sessions, sessions, market, version=technical.NAME_V4
    )
    without = technical.technical_features(table, sessions, version=technical.NAME_V4)
    names = technical.NAMES_V4
    market_cols = {
        "alpha_percent",
        "alpha_t",
        "beta",
        "log_residual_volatility",
        "has_market_model",
        "realized_label_1",
        "realized_label_5",
        "realized_label_21",
        "alpha_21_percent",
        "alpha_63_percent",
    }
    for column, name in enumerate(names):
        if name in market_cols:
            assert not without[:, column].any(), name
            assert with_factor[300:, column].any(), name
        else:
            assert np.array_equal(without[:, column], with_factor[:, column]), name


def test_the_fourth_version_normalizes_63_to_251_with_available_vol_and_matches_v3_from_252():
    _, sessions, table, market = market_prices(seed=17)
    prices, sessions, market = table[:300].copy(), sessions[:300], market[:300]
    assert (prices[:, 4] > 0).all()
    third = technical.technical_features(
        prices, sessions, sessions, market, version=technical.NAME_V3
    )
    fourth = technical.technical_features(
        prices, sessions, sessions, market, version=technical.NAME_V4
    )
    names = technical.NAMES_V4
    idx = {name: i for i, name in enumerate(names)}
    # Los indicadores de historia no cambian de significado.
    flags = fourth[:, [idx["has_63_sessions"], idx["has_252_sessions"]]]
    assert flags[62].tolist() == [0.0, 0.0] and flags[63].tolist() == [1.0, 0.0]
    assert flags[251].tolist() == [1.0, 0.0] and flags[252].tolist() == [1.0, 1.0]
    assert np.array_equal(fourth[:, idx["has_63_sessions"]], third[:, idx["has_63_sessions"]])
    assert np.array_equal(fourth[:, idx["has_252_sessions"]], third[:, idx["has_252_sessions"]])
    # Antes de la sesión 63, todos los rasgos divididos por volatilidad siguen ausentes (en cero).
    vol_cols = (
        "return_1",
        "return_5",
        "return_21",
        "return_63",
        "return_126",
        "return_252",
        "momentum_12_1",
        "log_volatility_252",
        "volatility_21_over_252",
        "volatility_63_over_252",
        "intraday_1",
        "overnight_1",
        "range_1",
        "intraday_5",
        "overnight_5",
        "intraday_21",
        "overnight_21",
        "intraday_63",
        "overnight_63",
        "intraday_252",
        "overnight_252",
        "maximum_return_21",
    )
    for name in vol_cols:
        assert (fourth[:63, idx[name]] == 0.0).all(), name
    # Entre la 63 y la 251 coinciden con la cuenta independiente sobre toda la historia disponible.
    reference = _independent_v4_volatility_features(prices)
    for row in (63, 64, 100, 125, 126, 200, 250, 251):
        for name, expected in reference[row].items():
            assert fourth[row, idx[name]] == pytest.approx(float(expected), rel=2e-5, abs=2e-6), (
                row,
                name,
            )
    # En la 63 ya hay rendimientos normalizados (en v3 eran cero) y vol_63/vol_disponible == 1.
    for name in ("return_5", "return_21", "return_63", "log_volatility_252", "intraday_21"):
        assert fourth[63:252, idx[name]].any() and not third[63:252, idx[name]].any(), name
    assert fourth[63, idx["volatility_63_over_252"]] == 0.0
    assert fourth[100, idx["volatility_63_over_252"]] != 0.0
    assert fourth[125, idx["return_126"]] == 0.0 and fourth[126, idx["return_126"]] != 0.0
    assert fourth[250, idx["intraday_252"]] == 0.0 and fourth[251, idx["intraday_252"]] != 0.0
    assert (fourth[:252, idx["return_252"]] == 0.0).all()
    assert (fourth[:252, idx["momentum_12_1"]] == 0.0).all()
    # Desde la sesión 252, sin días sin volumen, las 62 columnas coinciden bit a bit con v3.
    assert np.array_equal(fourth[252:], third[252:])
    # Series cortas alrededor del umbral 63 y menores que cualquier ventana.
    at_63 = technical.technical_features(prices[:63], sessions[:63], version=technical.NAME_V4)
    assert (at_63[:, idx["return_21"]] == 0.0).all()
    at_64 = technical.technical_features(prices[:64], sessions[:64], version=technical.NAME_V4)
    assert at_64[63, idx["return_21"]] == fourth[63, idx["return_21"]] != 0.0
    for short_rows in (0, 1, 2, 4, 20):
        short = technical.technical_features(
            prices[:short_rows], sessions[:short_rows], version=technical.NAME_V4
        )
        assert short.shape == (short_rows, 62) and np.isfinite(short).all()
    # Precio constante (volatilidad cero): todo finito y los rasgos divididos por volatilidad a 0.
    flat = np.tile(np.array([10.0, 10.0, 10.0, 10.0, 1e5]), (300, 1))
    quiet = technical.technical_features(flat, sessions, version=technical.NAME_V4)
    assert np.isfinite(quiet).all()
    for name in vol_cols:
        assert (quiet[:, idx[name]] == 0.0).all(), name


def test_the_fourth_version_handles_zero_volume_in_abnormal_volume_and_amihud():
    import math

    _, sessions, table, _ = market_prices(seed=23)
    prices, sessions = table[:300].copy(), sessions[:300]
    # Insertamos: días sin volumen en el borde de 21 sesiones (19 y 20), días sueltos, ventana
    # con un único día negociado, ventana entera sin negociación y volúmenes inválidos.
    prices[[10, 19, 20], 4] = 0.0
    prices[[80, 82, 85], 4] = 0.0
    prices[140:161, 4] = 0.0
    prices[150, 4] = 2e5
    prices[200:225, 4] = 0.0
    prices[260, 4] = 0.0
    prices[280, 4] = -10.0
    prices[285, 4] = np.nan

    third = technical.technical_features(prices, sessions, version=technical.NAME_V3)
    fourth = technical.technical_features(prices, sessions, version=technical.NAME_V4)
    names = technical.NAMES_V4
    abn, amihud = names.index("abnormal_volume"), names.index("log_illiquidity_21")

    # Antes de 21 sesiones (hasta la fila 19), un día sin volumen sigue ausente; desde 20, -8.
    assert fourth[10, abn] == 0.0 and fourth[19, abn] == 0.0
    assert fourth[20, abn] == -technical.LIMIT and third[20, abn] == 0.0
    # En un día sin volumen con historia, abnormal_volume queda en el límite inferior del recorte.
    for row in (80, 82, 85, 160, 260):
        assert third[row, abn] == 0.0
        assert fourth[row, abn] == -technical.LIMIT
    # Incluso en una suspensión de 21+ sesiones (volume_21 == 0), sigue en el límite inferior.
    for row in range(220, 225):
        assert fourth[row, abn] == -technical.LIMIT
    # Un día con volumen positivo conserva el mismo log(volume / volume_21) que v3.
    for row in (81, 83, 86, 120, 261):
        assert fourth[row, abn] == third[row, abn]
    # Un volumen negativo o NaN no es un día sin negociación: queda ausente (en cero).
    assert fourth[280, abn] == 0.0 and fourth[285, abn] == 0.0
    # Con un volumen NaN o negativo dominante en la ventana, un día con volumen 0 tampoco da -LIMIT.
    nan_vol = table[:300].copy()
    nan_vol[100, 4] = np.nan
    nan_vol[105, 4] = 0.0
    neg_vol = table[:300].copy()
    neg_vol[150, 4] = -1e12
    neg_vol[155, 4] = 0.0
    inf_vol = table[:300].copy()
    inf_vol[180, 4] = np.inf
    nan_v4 = technical.technical_features(nan_vol, sessions, version=technical.NAME_V4)
    neg_v4 = technical.technical_features(neg_vol, sessions, version=technical.NAME_V4)
    inf_v4 = technical.technical_features(inf_vol, sessions, version=technical.NAME_V4)
    assert nan_v4[105, abn] == 0.0 and neg_v4[155, abn] == 0.0
    assert np.isfinite(nan_v4).all() and np.isfinite(neg_v4).all() and np.isfinite(inf_v4).all()
    # Un volumen infinito no es un día negociado válido y no diluye la media de Amihud.
    inf_zero = inf_vol.copy()
    inf_zero[180, 4] = 0.0
    inf_zero_v4 = technical.technical_features(inf_zero, sessions, version=technical.NAME_V4)
    assert inf_v4[185, amihud] == inf_zero_v4[185, amihud]
    # Antes de la fila 20 Amihud está ausente; en la fila 20 ya usa solo los días negociados.
    assert fourth[19, amihud] == 0.0

    # Cuenta independiente del impacto de Amihud solo sobre los días con negociación (dollars > 0).
    close, volume = prices[:, 3], prices[:, 4]
    daily = np.zeros(len(prices), dtype=np.float64)
    daily[1:] = np.log(close[1:] / close[:-1])

    def expected_amihud(row):
        window_close = close[row - 20 : row + 1]
        window_vol = volume[row - 20 : row + 1]
        window_ret = daily[row - 20 : row + 1]
        impacts = [
            abs(r) / (c * v)
            for c, v, r in zip(window_close, window_vol, window_ret, strict=True)
            if math.isfinite(c) and math.isfinite(v) and c > 0 and v > 0
        ]
        if not impacts:
            return 0.0
        mean_impact = sum(impacts) / len(impacts)
        value = (math.log(mean_impact) - math.log(1e-9)) / 3.0
        return float(np.float32(min(max(value, -technical.LIMIT), technical.LIMIT)))

    # Con días sin negociación en la ventana, v4 promedia solo los negociados (más ilíquido que v3).
    for row in (20, 80, 85, 95, 160, 225, 260, 280, 285):
        assert fourth[row, amihud] == pytest.approx(expected_amihud(row), rel=2e-5, abs=2e-6), row
    assert fourth[85, amihud] > third[85, amihud]
    assert fourth[160, amihud] > third[160, amihud]
    # Cuando los 21 días son negociados, coincide bit a bit con v3.
    assert fourth[120, amihud] == third[120, amihud]
    # Ventana sin ningún día negociado: el rasgo de Amihud queda ausente (en cero).
    for row in range(220, 225):
        assert fourth[row, amihud] == 0.0
    # Desde la 252, las otras 60 columnas (todas salvo abnormal_volume y log_illiquidity_21) son
    # idénticas bit a bit a v3 incluso cuando hay días sin negociación.
    other = [i for i in range(len(names)) if i not in (abn, amihud)]
    assert np.array_equal(fourth[252:, other], third[252:, other])


def test_the_fourth_version_is_causal_across_history_regimes_and_zero_volume():
    _, sessions, table, market = market_prices(seed=31)
    prices, sessions, market = table[:320].copy(), sessions[:320], market[:320]
    prices[[50, 80, 120, 260], 4] = 0.0
    prices[180:205, 4] = 0.0
    whole = technical.technical_features(
        prices, sessions, sessions, market, version=technical.NAME_V4
    )
    for cut in (15, 20, 62, 63, 64, 150, 202, 251, 252, 270, 319):
        changed, later = prices.copy(), market.copy()
        rng = np.random.default_rng(cut)
        changed[cut:] *= rng.uniform(0.5, 2.0, (len(prices) - cut, 5))
        changed[cut:, 1] = np.maximum(changed[cut:, 1], changed[cut:, [0, 3]].max(axis=1))
        changed[cut:, 2] = np.minimum(changed[cut:, 2], changed[cut:, [0, 3]].min(axis=1))
        # Alternamos volúmenes futuros entre cero y positivo para comprobar su causalidad.
        changed[cut::3, 4] = 0.0
        changed[cut + 1 :: 3, 4] = 5e5
        later[cut:] = -later[cut:] + 0.02
        again = technical.technical_features(
            changed, sessions, sessions, later, version=technical.NAME_V4
        )
        assert np.array_equal(again[:cut], whole[:cut]), cut
        assert not np.array_equal(again[cut:], whole[cut:]), cut
        prefix = technical.technical_features(
            prices[:cut], sessions[:cut], sessions[:cut], market[:cut], version=technical.NAME_V4
        )
        assert np.array_equal(prefix, whole[:cut]), cut


def test_the_fourth_version_is_deterministic_and_keeps_dense_branch_gradients_finite():
    import torch

    from mars_titan.models.titans.technical_branch import TechnicalDense, TechnicalDenseConfig

    _, sessions, table, market = market_prices(seed=41)
    prices, sessions, market = table[:300].copy(), sessions[:300], market[:300]
    prices[[80, 150, 260], 4] = 0.0
    prices[200:225, 4] = 0.0
    one = technical.technical_features(
        prices, sessions, sessions, market, version=technical.NAME_V4
    )
    two = technical.technical_features(
        prices, sessions, sessions, market, version=technical.NAME_V4
    )
    assert np.array_equal(one, two)
    # Comprobamos FP32 y FP64 con la capa de salida iniciada a cero (predicción constante).
    rows = one[[10, 63, 80, 150, 220, 251, 252, 260]]
    markets = ["US"] * len(rows)
    config = TechnicalDenseConfig(columns=62, hidden=32, depth=2, session_ranks=True)
    for dtype in (torch.float32, torch.float64):
        branch = TechnicalDense(16, config, clip=technical.LIMIT, seed=7, dtype=dtype)
        charts = torch.tensor(rows, dtype=dtype, requires_grad=True)
        out = branch(charts, markets)
        assert out.dtype == dtype and torch.isfinite(out).all()
        assert torch.count_nonzero(out) == 0
        out.sum().backward()
        assert torch.isfinite(charts.grad).all()
        for name, param in branch.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), (dtype, name)
        assert branch.outer.weight.grad.abs().sum() > 0
        assert branch.outer.bias.grad.abs().sum() > 0
