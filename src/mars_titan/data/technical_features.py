"""Rasgos técnicos causales de horizonte largo de un activo (`long_horizon_v1` a `_v5`).

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

`long_horizon_v3` conserva las 46 columnas de la v2, bit a bit, y añade 16 de precio y volumen:
las sesiones desde el máximo y el mínimo de la ventana, la parte de los movimientos y de las
sesiones que fueron al alza, la correlación entre el logaritmo del cierre y el del volumen más
uno, la variabilidad del volumen y del volumen ponderado por el movimiento, y la sombra superior
media de la vela. Cada una va centrada en su punto neutro, de modo que el cero de la historia
insuficiente no señala nada. Una ventana sin movimiento no es historia insuficiente: en ella los
días desde el extremo y la fracción de subidas valen −1, la firma de un activo suspendido. Son
las que pasaron la validación rasgo a rasgo en la tabla de China del origen 2013, solo con el
tramo de ajuste (señal estable por trimestres y correlación menor de 0,7 con las 46). Medido
allí: con árboles, de 0,117 a 0,123 de IC de rangos en la evaluación; con una red de dos capas,
de 0,126 a 0,134. En EE. UU. no cambian el resultado de los árboles.

`long_horizon_v4` conserva las 62 columnas de la v3 con los mismos nombres y posiciones, y
corrige dos pérdidas de información: entre 63 y 251 sesiones el divisor de volatilidad usa la
volatilidad de toda la historia disponible en lugar de anular los rasgos (sin cambiar el
significado de `has_63_sessions` ni `has_252_sessions`), y un día sin volumen deja
`abnormal_volume` en el límite inferior del recorte (−8) y no entra en la media del impacto de
Amihud (que se calcula solo sobre los días con negociación de la ventana, o queda ausente si no
hay ninguno).

`long_horizon_v5` son las 46 columnas de la v2 con esas dos mismas correcciones: celda a celda,
las 46 primeras columnas de la v4. Es la versión para el mercado que no usa las 16 columnas
añadidas por la v3.
"""

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

NAME = "long_horizon_v1"
NAME_V2 = "long_horizon_v2"
NAME_V3 = "long_horizon_v3"
NAME_V4 = "long_horizon_v4"
NAME_V5 = "long_horizon_v5"
# Parte mínima de filas válidas en una ventana de la v3.
COVERAGE = 0.75
# Por debajo de esta parte de su escala, una diferencia de la v3 es redondeo y no un dato.
NEGLIGIBLE = 1e-12
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
NAMES_V3 = (
    *NAMES_V2,
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
NAMES_V4 = NAMES_V3
NAMES_V5 = NAMES_V2
# Versiones que llevan las dos correcciones de la cuarta.
CORRECTED = (NAME_V4, NAME_V5)
# Columnas de cada versión declarada. De la primera a la cuarta, las de una versión son un prefijo
# de las de la siguiente; la quinta son las 46 de la segunda.
VERSIONS = {
    NAME: NAMES,
    NAME_V2: NAMES_V2,
    NAME_V3: NAMES_V3,
    NAME_V4: NAMES_V4,
    NAME_V5: NAMES_V5,
}


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


def _windows(values, window, first=0):
    """Ventanas de `window` filas hasta cada fila desde `first + window - 1`, y esa fila."""
    start = first + window - 1
    if len(values) <= start:
        return None, start
    return sliding_window_view(values[first:], window), start


def _since_extreme(values, window, largest):
    """Sesiones desde el extremo más reciente de la ventana: −1 si es hoy, 1 el más antiguo.

    Necesita el valor de la sesión, además de la cobertura mínima. En una ventana sin
    movimiento el empate lo gana hoy y vale −1, para el máximo y para el mínimo.
    """
    result = np.full(len(values), np.nan)
    windows, start = _windows(values, window)
    if windows is None:
        return result
    valid = np.isfinite(windows)
    # Del revés, para que un empate lo gane la sesión más reciente. Empatan los valores que
    # distan del extremo menos que el redondeo: en un tramo sin negociar los precios ajustados
    # bailan en el último bit.
    recent = np.where(valid, windows, -np.inf if largest else np.inf)[:, ::-1]
    extreme = recent.max(axis=1) if largest else recent.min(axis=1)
    margin = NEGLIGIBLE * np.abs(np.where(np.isfinite(extreme), extreme, 0.0))
    with np.errstate(invalid="ignore"):
        tied = (
            recent >= (extreme - margin)[:, None]
            if largest
            else recent <= (extreme + margin)[:, None]
        )
    age = np.argmax(tied, axis=1)
    known = (valid.sum(axis=1) >= math.ceil(COVERAGE * window)) & valid[:, -1]
    result[start:] = np.where(known, 2.0 * age / (window - 1) - 1.0, np.nan)
    return result


def _up_share(returns, window, first):
    """Parte del movimiento de la ventana que fue al alza, de −1 (todo bajadas) a 1."""
    result = np.full(len(returns), np.nan)
    windows, start = _windows(returns, window, first)
    if windows is None:
        return result
    valid = np.isfinite(windows)
    clean = np.where(valid, windows, 0.0)
    total = np.abs(clean).sum(axis=1)
    # Un movimiento total del tamaño del redondeo no es movimiento.
    known = (valid.sum(axis=1) >= math.ceil(COVERAGE * window)) & (total > NEGLIGIBLE)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.maximum(clean, 0.0).sum(axis=1) / total
    result[start:] = np.where(known, 2.0 * share - 1.0, np.nan)
    return result


def _up_fraction(returns, window, first):
    """Parte de las sesiones válidas de la ventana que cerraron al alza, de −1 a 1.

    Una sesión sin cambio no es una subida: una ventana sin movimiento vale −1.
    """
    result = np.full(len(returns), np.nan)
    windows, start = _windows(returns, window, first)
    if windows is None:
        return result
    valid = np.isfinite(windows)
    count = valid.sum(axis=1)
    with np.errstate(invalid="ignore"):
        # Una subida del tamaño del redondeo no es una subida.
        rising = (valid & (windows > NEGLIGIBLE)).sum(axis=1)
    known = count >= math.ceil(COVERAGE * window)
    result[start:] = np.where(known, 2.0 * rising / np.maximum(count, 1) - 1.0, np.nan)
    return result


def _centered(windows, valid):
    """Cada ventana menos su media sobre las filas válidas, y si sus valores se separan.

    Una ventana constante no tiene dispersión. Se decide con sus extremos, respecto a su
    mayor valor absoluto, y no con una resta de sumas, que deja residuos de redondeo.
    """
    count = np.maximum(valid.sum(axis=1), 1)
    mean = np.where(valid, windows, 0.0).sum(axis=1) / count
    spread = np.where(valid, windows, -np.inf).max(axis=1) - np.where(valid, windows, np.inf).min(
        axis=1
    )
    largest = np.where(valid, np.abs(windows), 0.0).max(axis=1)
    varies = spread > NEGLIGIBLE * np.maximum(largest, np.finfo(float).tiny)
    return np.where(valid, windows - mean[:, None], 0.0), mean, varies


def _window_correlation(left, right, window):
    """Correlación de Pearson de las filas válidas de la ventana."""
    result = np.full(len(left), np.nan)
    one, start = _windows(left, window)
    if one is None:
        return result
    other, _ = _windows(right, window)
    valid = np.isfinite(one) & np.isfinite(other)
    with np.errstate(divide="ignore", invalid="ignore"):
        x, _, varies_x = _centered(one, valid)
        y, _, varies_y = _centered(other, valid)
        scale = np.sqrt(np.einsum("ij,ij->i", x, x) * np.einsum("ij,ij->i", y, y))
        value = np.einsum("ij,ij->i", x, y) / scale
    count = valid.sum(axis=1)
    known = (count >= max(math.ceil(COVERAGE * window), 2)) & varies_x & varies_y & (scale > 0)
    result[start:] = np.where(known, np.clip(value, -1.0, 1.0), np.nan)
    return result


def _log_variation(values, window, first, neutral):
    """Logaritmo del coeficiente de variación de la ventana entre su valor neutro."""
    result = np.full(len(values), np.nan)
    windows, start = _windows(values, window, first)
    if windows is None:
        return result
    valid = np.isfinite(windows)
    count = valid.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        centered, mean, varies = _centered(windows, valid)
        deviation = np.sqrt(np.einsum("ij,ij->i", centered, centered) / np.maximum(count - 1, 1))
        value = np.log(deviation / mean / neutral)
    known = (count >= max(math.ceil(COVERAGE * window), 2)) & (mean > 0) & varies
    result[start:] = np.where(known, value, np.nan)
    return result


def _window_mean(values, window):
    """Media de las filas válidas de la ventana, con su cobertura mínima."""
    result = np.full(len(values), np.nan)
    windows, start = _windows(values, window)
    if windows is None:
        return result
    valid = np.isfinite(windows)
    count = valid.sum(axis=1)
    total = np.where(valid, windows, 0.0).sum(axis=1)
    known = count >= math.ceil(COVERAGE * window)
    result[start:] = np.where(known, total / np.maximum(count, 1), np.nan)
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


def technical_features(prices, sessions, calendar=None, market=None, version=NAME):
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

        def available_volatility(window, minimum):
            # Entre `minimum` y `window - 1` sesiones usa los `t` rendimientos disponibles
            # (`filled[0]` es cero); desde `window` coincide bit a bit con `volatility(window)`.
            count = np.minimum(np.arange(rows, dtype=np.float64), float(window))
            mean = _window_sum(filled, window) / count
            squared = _window_sum(filled * filled, window) / count
            result = np.sqrt(np.maximum(squared - mean * mean, 0.0))
            result[:minimum] = np.nan
            return result

        year = available_volatility(252, 63) if version in CORRECTED else volatility(252)
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
        abnormal = np.log(volume / volume_21)
        if version in CORRECTED:
            abnormal = np.where(
                np.isfinite(volume_21) & (volume_21 >= 0.0) & (volume == 0.0),
                -LIMIT,
                abnormal,
            )
        features["abnormal_volume"] = abnormal
        if version in CORRECTED:
            traded = np.isfinite(dollars) & (dollars > 0)
            impact = np.where(traded, np.abs(filled) / dollars, 0.0)
            traded_21 = _window_sum(traded.astype(np.float64), 21)
            illiquidity = np.where(traded_21 > 0, _window_sum(impact, 21) / traded_21, np.nan)
        else:
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
        if version in (NAME_V2, NAME_V3, NAME_V4, NAME_V5):
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
        if version in (NAME_V3, NAME_V4):
            open_v, high_v, low_v, close_v = (
                np.where(np.isfinite(v) & (v > 0), v, np.nan) for v in (opened, high, low, close)
            )
            volume_v = np.where(np.isfinite(volume) & (volume >= 0), volume, np.nan)
            close_to_close = np.full(rows, np.nan)
            close_to_close[1:] = np.log(close_v[1:] / close_v[:-1])
            open_to_close = np.log(close_v / open_v)
            for window in (21, 63, 252):
                features[f"days_since_high_{window}"] = _since_extreme(high_v, window, True)
            for window in (63, 252):
                features[f"days_since_low_{window}"] = _since_extreme(low_v, window, False)
            # El primer rendimiento de cierre a cierre es el de la segunda fila.
            features["up_move_share_cc_14"] = _up_share(close_to_close, 14, 1)
            features["up_move_share_oc_14"] = _up_share(open_to_close, 14, 0)
            features["up_day_fraction_cc_21"] = _up_fraction(close_to_close, 21, 1)
            features["up_day_fraction_cc_63"] = _up_fraction(close_to_close, 63, 1)
            features["up_day_fraction_oc_63"] = _up_fraction(open_to_close, 63, 0)
            features["price_volume_corr_63"] = _window_correlation(
                np.log(close_v), np.log1p(volume_v), 63
            )
            # Neutros: un coeficiente de variación de 0,5 en el volumen y de 1 con el movimiento.
            features["volume_cv_21"] = _log_variation(volume_v, 21, 0, 0.5)
            features["move_weighted_volume_cv_cc_21"] = _log_variation(
                np.abs(close_to_close) * volume_v, 21, 1, 1.0
            )
            features["move_weighted_volume_cv_oc_21"] = _log_variation(
                np.abs(open_to_close) * volume_v, 21, 0, 1.0
            )
            span = high_v - low_v
            shadow = np.clip((high_v - np.maximum(open_v, close_v)) / span, 0.0, 1.0)
            # Un rango del tamaño del redondeo no es una vela.
            shadow = np.where(span > NEGLIGIBLE * high_v, shadow, np.nan)
            for window in (5, 21):
                # Una sombra superior de un cuarto del rango es lo habitual: su punto neutro.
                features[f"candle_upper_shadow_{window}"] = 4.0 * (
                    _window_mean(shadow, window) - 0.25
                )
        features["has_market_model"] = np.zeros(rows)
        if calendar is not None and market is not None:
            calendar = np.asarray(calendar, dtype="datetime64[D]")
            market = np.asarray(market, dtype=np.float64)
            position = np.searchsorted(calendar, sessions)
            inside = position < len(calendar)
            inside[inside] = calendar[position[inside]] == sessions[inside]
            ratio = np.where((opened > 0) & (close > 0), close / opened - 1, np.nan)
            aligned = np.full(len(calendar), np.nan)
            aligned[position[inside]] = ratio[inside]
            alpha, beta, residual, pairs = market_model(aligned, market)
            for name, values in (
                ("alpha_percent", alpha * 100),
                ("alpha_t", alpha / (residual / np.sqrt(pairs))),
                ("beta", beta),
                ("log_residual_volatility", np.log(residual * 100)),
            ):
                features[name][inside] = values[position[inside]]
            features["has_market_model"] = np.isfinite(features["alpha_percent"]).astype(np.float64)
            if version in (NAME_V2, NAME_V3, NAME_V4, NAME_V5):
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
