"""Rasgos técnicos causales de horizonte largo de un activo (`long_horizon_v1` y `long_horizon_v2`).

La etiqueta de la edición histórica es el residuo de la rentabilidad intradía de la sesión
siguiente, `r(t+1) - alfa(t) - beta(t) * m(t+1)`, con alfa y beta de un OLS con intercepto sobre
las 252 sesiones del calendario que terminan en `t` (`data.budget_targets.residual_targets`).
Alfa y beta se conocen al predecir, pero la ventana del predictor solo ve 64 sesiones. Este
módulo calcula, con los precios disponibles hasta cada sesión, esos dos valores tal como los usa
la etiqueta y el resto de rasgos de precio y liquidez de horizonte largo.

Cada fila usa solo sus sesiones y las anteriores: cambiar un precio posterior no cambia ninguna
fila anterior. Los valores van en float32, acotados a ±8, con cero donde falta historia y un
indicador de historia suficiente.

`long_horizon_v2` conserva las 30 columnas de la v1, bit a bit, y añade 16: las medias intradía
y nocturna de 5, 63 y 252 sesiones (la de 252 también en porcentaje, que aproxima el alfa en un
mercado cuyo factor no esté disponible), la etiqueta ya realizada de la última sesión y de las
5 y 21 últimas, el alfa de 21 y 63 sesiones, la fracción de sesiones sin negociación y la
posición del cierre en el rango del día. Medido en el origen 2013 de EE. UU. con un modelo de
árboles ajustado solo con estos rasgos: IC de rangos de 0,111 en la evaluación con la v1 y de
0,114 con las columnas añadidas.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

NAME = "long_horizon_v1"
NAME_V2 = "long_horizon_v2"
# Ventana y mínimo de pares del OLS de la etiqueta.
HISTORY, MINIMUM = 252, 126
LIMIT = 8.0
NAMES = (
    "alpha_percent",
    "alpha_t",
    "beta",
    "log_residual_volatility",
    "has_market_model",
    "return_5",
    "return_21",
    "return_63",
    "return_126",
    "return_252",
    "momentum_12_1",
    "log_volatility_252",
    "volatility_21_over_252",
    "volatility_63_over_252",
    "distance_to_high_252",
    "distance_to_low_252",
    "has_252_sessions",
    "log_dollar_volume_21",
    "dollar_volume_21_over_252",
    "dollar_volume_5_over_63",
    "abnormal_volume",
    "log_illiquidity_21",
    "return_1",
    "intraday_1",
    "overnight_1",
    "range_1",
    "intraday_21",
    "overnight_21",
    "maximum_return_21",
    "has_63_sessions",
)
WIDTH = len(NAMES)
NAMES_V2 = (
    *NAMES,
    "intraday_mean_252_percent",
    "overnight_mean_252_percent",
    "intraday_5",
    "intraday_63",
    "intraday_252",
    "overnight_5",
    "overnight_63",
    "overnight_252",
    "realized_label_1",
    "realized_label_5",
    "realized_label_21",
    "alpha_21_percent",
    "alpha_63_percent",
    "untraded_63",
    "close_location_1",
    "close_location_5",
)
# Columnas de cada versión declarada. Las de una versión anterior son siempre un prefijo.
VERSIONS = {NAME: NAMES, NAME_V2: NAMES_V2}


def _window_sum(values, window):
    """Suma de las `window` últimas filas hasta cada una, con las que haya al principio."""
    total = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    index = np.arange(1, len(values) + 1)
    return total[index] - total[np.maximum(index - window, 0)]


def _window_extreme(values, window, function):
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        result[window - 1 :] = function(sliding_window_view(values, window), axis=1)
    return result


def market_model(intraday, market):
    """Alfa, beta, desviación residual y pares del OLS de la etiqueta en cada sesión.

    `intraday` y `market` son las rentabilidades de apertura a cierre del activo y del factor
    de mercado sobre el mismo calendario, con NaN donde falta la sesión. Usa los pares válidos
    de las 252 sesiones que terminan en cada una y exige 126, como la etiqueta.
    """
    valid = np.isfinite(intraday) & np.isfinite(market)
    x, y = np.where(valid, market, 0.0), np.where(valid, intraday, 0.0)
    pairs = _window_sum(valid.astype(np.float64), HISTORY)
    sum_x, sum_y = _window_sum(x, HISTORY), _window_sum(y, HISTORY)
    with np.errstate(divide="ignore", invalid="ignore"):
        xx = _window_sum(x * x, HISTORY) - sum_x * sum_x / pairs
        xy = _window_sum(x * y, HISTORY) - sum_x * sum_y / pairs
        yy = _window_sum(y * y, HISTORY) - sum_y * sum_y / pairs
        beta = xy / xx
        alpha = sum_y / pairs - beta * sum_x / pairs
        residual = np.sqrt(np.maximum(yy - beta * xy, 0.0) / np.maximum(pairs - 2, 1))
    known = (pairs >= MINIMUM) & (xx > np.finfo(float).eps)
    for values in (alpha, beta, residual):
        values[~known] = np.nan
    return alpha, beta, residual, pairs


def _aligned_market_model(opened, close, sessions, calendar, market):
    """Alinea las rentabilidades intradía al calendario del factor y calcula el OLS de mercado."""
    calendar = np.asarray(calendar, dtype="datetime64[D]")
    market = np.asarray(market, dtype=np.float64)
    position = np.searchsorted(calendar, sessions)
    inside = position < len(calendar)
    inside[inside] = calendar[position[inside]] == sessions[inside]
    ratio = np.where((opened > 0) & (close > 0), close / opened - 1, np.nan)
    aligned = np.full(len(calendar), np.nan)
    aligned[position[inside]] = ratio[inside]
    alpha, beta, residual, pairs = market_model(aligned, market)
    return calendar, market, position, inside, aligned, alpha, beta, residual, pairs


def technical_features(
    prices, sessions, calendar=None, market=None, version=NAME, *, _precomputed_market=None
):
    """Rasgos de cada fila de `prices` (apertura, máximo, mínimo, cierre y volumen).

    `sessions` es el día de cada fila. `calendar` y `market` son los días del factor de mercado
    y su rentabilidad intradía; sin ellos los rasgos del modelo de mercado quedan a cero con su
    indicador apagado. Devuelve un array float32 de forma [filas, columnas de `version`].
    """
    if version not in VERSIONS:
        raise ValueError("La versión de los rasgos técnicos no está declarada")
    names = VERSIONS[version]
    prices = np.asarray(prices, dtype=np.float64)
    sessions = np.asarray(sessions, dtype="datetime64[D]")
    if prices.ndim != 2 or prices.shape[1] != 5 or len(sessions) != len(prices):
        raise ValueError("Los rasgos técnicos necesitan una tabla OHLCV y el día de cada fila")
    if len(sessions) > 1 and np.any(np.diff(sessions.astype(np.int64)) <= 0):
        raise ValueError("Los rasgos técnicos necesitan sesiones en orden creciente")
    opened, high, low, close, volume = prices.T
    rows = len(prices)
    features = {name: np.full(rows, np.nan) for name in names}
    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.log(close)
        daily = np.full(rows, np.nan)
        daily[1:] = log_close[1:] - log_close[:-1]
        filled = np.nan_to_num(daily)

        def change(sessions_back):
            result = np.full(rows, np.nan)
            result[sessions_back:] = log_close[sessions_back:] - log_close[:-sessions_back]
            return result

        def volatility(window):
            mean = _window_sum(filled, window) / window
            squared = _window_sum(filled * filled, window) / window
            result = np.sqrt(np.maximum(squared - mean * mean, 0.0))
            result[:window] = np.nan
            return result

        year = volatility(252)
        for window in (5, 21, 63, 126, 252):
            features[f"return_{window}"] = change(window) / (year * np.sqrt(window))
        momentum = np.full(rows, np.nan)
        if rows > 252:
            momentum[252:] = log_close[231 : rows - 21] - log_close[: rows - 252]
        features["momentum_12_1"] = momentum / (year * np.sqrt(231))
        features["log_volatility_252"] = np.log(year * 100)
        features["volatility_21_over_252"] = np.log(volatility(21) / year)
        features["volatility_63_over_252"] = np.log(volatility(63) / year)
        features["distance_to_high_252"] = log_close - np.log(_window_extreme(close, 252, np.max))
        features["distance_to_low_252"] = log_close - np.log(_window_extreme(close, 252, np.min))
        features["has_252_sessions"] = (np.arange(rows) >= 252).astype(np.float64)
        features["has_63_sessions"] = (np.arange(rows) >= 63).astype(np.float64)
        dollars = close * np.maximum(volume, 0.0)
        mean_21, mean_252 = _window_sum(dollars, 21) / 21, _window_sum(dollars, 252) / 252
        mean_21[:20], mean_252[:251] = np.nan, np.nan
        mean_5, mean_63 = _window_sum(dollars, 5) / 5, _window_sum(dollars, 63) / 63
        mean_63[:62] = np.nan
        # Volumen en dólares centrado en diez millones diarios y en unidades de e³.
        features["log_dollar_volume_21"] = (np.log(mean_21) - np.log(1e7)) / 3
        features["dollar_volume_21_over_252"] = np.log(mean_21 / mean_252)
        features["dollar_volume_5_over_63"] = np.log(mean_5 / mean_63)
        volume_21 = _window_sum(volume, 21) / 21
        volume_21[:20] = np.nan
        features["abnormal_volume"] = np.log(volume / volume_21)
        impact = np.where(dollars > 0, np.abs(filled) / dollars, 0.0)
        illiquidity = _window_sum(impact, 21) / 21
        illiquidity[:20] = np.nan
        # Ilíquidez de Amihud centrada en 1e-9 por dólar y en unidades de e³.
        features["log_illiquidity_21"] = (np.log(illiquidity) - np.log(1e-9)) / 3
        overnight = np.full(rows, np.nan)
        overnight[1:] = np.log(opened[1:] / close[:-1])
        intraday_log = np.log(close / opened)
        features["return_1"] = daily / year
        features["intraday_1"] = intraday_log / year
        features["overnight_1"] = overnight / year
        features["range_1"] = np.log(high / low) / year
        sum_intraday = _window_sum(np.nan_to_num(intraday_log), 21) / 21
        sum_overnight = _window_sum(np.nan_to_num(overnight), 21) / 21
        sum_intraday[:20], sum_overnight[:21] = np.nan, np.nan
        features["intraday_21"] = sum_intraday / year * np.sqrt(21)
        features["overnight_21"] = sum_overnight / year * np.sqrt(21)
        features["maximum_return_21"] = _window_extreme(filled, 21, np.max) / year
        if version == NAME_V2:
            clean_intraday = np.where(np.isfinite(intraday_log), intraday_log, 0.0)
            clean_overnight = np.where(np.isfinite(overnight), overnight, 0.0)
            for window in (5, 63, 252):
                mean_intraday = _window_sum(clean_intraday, window) / window
                mean_overnight = _window_sum(clean_overnight, window) / window
                # La primera fila no tiene rendimiento nocturno.
                mean_intraday[: window - 1], mean_overnight[:window] = np.nan, np.nan
                features[f"intraday_{window}"] = mean_intraday / year * np.sqrt(window)
                features[f"overnight_{window}"] = mean_overnight / year * np.sqrt(window)
            features["intraday_mean_252_percent"] = mean_intraday * 100
            features["overnight_mean_252_percent"] = mean_overnight * 100
            location = np.where(high > low, (2 * close - high - low) / (high - low), 0.0)
            location = np.where(np.isfinite(location), location, 0.0)
            features["close_location_1"] = location
            mean_location = _window_sum(location, 5) / 5
            mean_location[:4] = np.nan
            features["close_location_5"] = mean_location
            untraded = ((volume <= 0) | ((opened == close) & (high == low))).astype(np.float64)
            fraction = _window_sum(untraded, 63) / 63
            fraction[:62] = np.nan
            features["untraded_63"] = fraction
        features["has_market_model"] = np.zeros(rows)
        if _precomputed_market is not None or (calendar is not None and market is not None):
            if _precomputed_market is not None:
                calendar, market, position, inside, aligned, alpha, beta, residual, pairs = (
                    _precomputed_market
                )
            else:
                calendar, market, position, inside, aligned, alpha, beta, residual, pairs = (
                    _aligned_market_model(opened, close, sessions, calendar, market)
                )
            for name, values in (
                ("alpha_percent", alpha * 100),
                ("alpha_t", alpha / (residual / np.sqrt(pairs))),
                ("beta", beta),
                ("log_residual_volatility", np.log(residual * 100)),
            ):
                features[name][inside] = values[position[inside]]
            features["has_market_model"] = np.isfinite(features["alpha_percent"]).astype(np.float64)
            if version == NAME_V2:
                # Etiqueta de la sesión anterior, ya realizada al cierre de esta:
                # r(t) - alfa(t-1) - beta(t-1) * m(t), en desviaciones residuales de t-1.
                realized = np.full(len(calendar), np.nan)
                realized[1:] = (aligned[1:] - alpha[:-1] - beta[:-1] * market[1:]) / residual[:-1]
                realized = np.where(np.isfinite(realized), realized, 0.0)
                free = aligned - beta * market
                free = np.where(np.isfinite(free), free, 0.0)
                for name, values in (
                    ("realized_label_1", realized),
                    ("realized_label_5", _window_sum(realized, 5) / np.sqrt(5)),
                    ("realized_label_21", _window_sum(realized, 21) / np.sqrt(21)),
                    ("alpha_21_percent", _window_sum(free, 21) / 21 * 100),
                    ("alpha_63_percent", _window_sum(free, 63) / 63 * 100),
                ):
                    features[name][inside] = values[position[inside]]
    table = np.column_stack([features[name] for name in names])
    table = np.clip(np.where(np.isfinite(table), table, 0.0), -LIMIT, LIMIT)
    return table.astype(np.float32)
