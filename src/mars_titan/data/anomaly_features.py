"""Anomalías conocidas de cada sesión con la información disponible en su decisión (#747).

Las características repiten anomalías publicadas de la sección cruzada de acciones. Para la
sesión t de un activo, con C el cierre, V el volumen y r el retorno cierre a cierre:

- `reversal`: `reversal_1` y `reversal_5` son C_t / C_{t−k} − 1 con k = 1 y 5, la reversión
  de corto plazo (Jegadeesh, 1990, y Lehmann, 1990).
- `momentum`: `momentum_12_1` es C_{t−21} / C_{t−252} − 1, el momentum de doce meses que salta
  el último (Jegadeesh y Titman, 1993). Cada extremo toma el último cierre de su sesión o de
  las cinco anteriores, porque una ausencia aislada del activo no debe anular un año.
- `idiosyncratic_volatility`: desviación del residuo de la regresión por mínimos cuadrados con
  intercepto de r del activo sobre r del factor de mercado en las 20 y 60 últimas sesiones
  (Ang et al., 2006), con n − 2 grados de libertad.
- `volume`: `volume_shock_20` es log(1 + V_t) menos la media de log(1 + V) en las 20 sesiones
  anteriores (Gervais et al., 2001). `relative_turnover_20_252` es la media de log(1 + V) de
  las 20 últimas sesiones menos la de las 252 últimas, una rotación relativa a la del propio
  activo (Datar et al., 1998).
- `year_high`: `distance_to_52_week_high` es C_t entre el máximo cierre de las 252 últimas
  sesiones, menos uno (George y Hwang, 2004).
- `filings`: `sessions_since_filing` cuenta las sesiones desde la última presentación con
  estados financieros disponible en la decisión de t, y `recent_filing_5` vale uno si esa
  presentación tiene menos de cinco sesiones. La deriva posterior a los resultados (Bernard y
  Thomas, 1989) se mide con el anuncio, que la edición no tiene. La presentación llega días o
  semanas después, así que esta familia solo aproxima esa anomalía.

Una ventana exige el 75 % de sus observaciones, y una característica sin ellas queda ausente.

La rotación no se divide entre las acciones en circulación. La edición distribuye precios y
volúmenes con los ajustes retrospectivos de su fuente, y las acciones de cada presentación no
los tienen. Mezclarlos introduciría desdoblamientos posteriores a la decisión. Las razones de
precios o de volúmenes de una misma serie no cambian con un factor de ajuste común, así que
todas las características son cocientes o diferencias de logaritmos de la propia serie.

Las sesiones son las del calendario declarado del mercado sin las ausentes en todo él, igual
que las ventanas de precios de la edición. En cada sesión, `session_ranks` lleva cada
característica a rangos en [−1, 1] entre los activos con precio en esa sesión, como Gu et al.
(2020), y una ausencia vale cero con su indicador. La sección cruzada solo depende de los
precios de la sesión, no de qué filas tienen después una etiqueta madura.
"""

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import rankdata

IDENTITY = "known_anomalies_v1"
FAMILIES = {
    "reversal": ("reversal_1", "reversal_5"),
    "momentum": ("momentum_12_1",),
    "idiosyncratic_volatility": ("idiosyncratic_volatility_20", "idiosyncratic_volatility_60"),
    "volume": ("volume_shock_20", "relative_turnover_20_252"),
    "year_high": ("distance_to_52_week_high",),
    "filings": ("sessions_since_filing", "recent_filing_5"),
}
FEATURES = tuple(name for names in FAMILIES.values() for name in names)
YEAR, MONTH = 252, 21
# Sesiones hacia atrás en las que un extremo del momentum busca su último cierre.
MOMENTUM_TOLERANCE = 5
# Fracción de observaciones válidas que exige cada ventana.
MINIMUM_SHARE = 0.75
RECENT_FILING = 5


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _minimum(length):
    return math.ceil(MINIMUM_SHARE * length)


def market_sessions(positions, absent):
    """Índice entre las sesiones con mercado de cada posición del calendario declarado.

    `absent` son las posiciones ausentes en todo el mercado, ordenadas. Una posición ausente
    toma el índice de la siguiente sesión con mercado, la primera en que se conoce lo que llegó
    en ella.
    """
    positions, absent = np.asarray(positions, dtype=np.int64), np.asarray(absent, dtype=np.int64)
    _require(
        positions.ndim == 1 and absent.ndim == 1 and np.all(np.diff(absent) > 0),
        "Las posiciones y las ausencias deben ser vectores, las ausencias ordenadas",
    )
    return positions - np.searchsorted(absent, positions, side="left")


def dense_series(values, sessions, length):
    """Serie de `length` sesiones con mercado y NaN donde el activo no tiene valor."""
    values, sessions = np.asarray(values, dtype=np.float64), np.asarray(sessions, dtype=np.int64)
    _require(
        len(values) == len(sessions)
        and np.all(np.diff(sessions) > 0)
        and (not len(sessions) or 0 <= sessions[0] and sessions[-1] < length),
        "Las sesiones de la serie deben ser únicas, crecientes y del calendario",
    )
    series = np.full((length, *values.shape[1:]), np.nan)
    series[sessions] = values
    return series


def filing_sessions(accessions, available, decisions, absent):
    """Sesión con mercado en la que se conoce cada presentación, una vez por presentación.

    Una presentación es un número de acceso y se conoce con su primer hecho disponible. Se
    conoce en la decisión de la sesión t si llega en ese instante o antes, así que su sesión es
    la de la primera decisión igual o posterior. Las que llegan después del calendario se omiten.
    """
    accessions = np.asarray(accessions, dtype=object)
    available = np.asarray(available, dtype=np.int64)
    decisions = np.asarray(decisions, dtype=np.int64)
    _require(
        accessions.shape == available.shape and np.all(np.diff(decisions) > 0),
        "Las presentaciones necesitan un acceso y una disponibilidad por hecho",
    )
    if not len(accessions):
        return np.empty(0, dtype=np.int64)
    first = {}
    for accession, moment in zip(accessions.tolist(), available.tolist(), strict=True):
        _require(isinstance(accession, str) and accession, "Una presentación no tiene acceso")
        first[accession] = min(moment, first.get(accession, moment))
    positions = np.searchsorted(decisions, np.fromiter(first.values(), np.int64), side="left")
    return np.unique(market_sessions(positions[positions < len(decisions)], absent))


def _lagged(values, lag):
    result = np.full(len(values), np.nan)
    result[lag:] = values[: len(values) - lag]
    return result


def _trailing(values, length):
    """Ventanas de `length` sesiones que terminan en cada sesión, con NaN antes del inicio."""
    return sliding_window_view(np.concatenate([np.full(length - 1, np.nan), values]), length)


def _window_mean(values, length, minimum):
    window = _trailing(values, length)
    valid = np.isfinite(window)
    count = valid.sum(axis=1)
    total = np.where(valid, window, 0.0).sum(axis=1)
    return np.where(count >= minimum, total / np.maximum(count, 1), np.nan)


def _last_valid(values, tolerance):
    """Último valor válido de cada sesión o de las `tolerance` anteriores."""
    index = np.arange(len(values))
    last = np.maximum.accumulate(np.where(np.isfinite(values), index, -1))
    usable = (last >= 0) & (index - last <= tolerance)
    return np.where(usable, values[np.maximum(last, 0)], np.nan)


def _residual_volatility(asset, market, length):
    """Desviación del residuo de MCO con intercepto en cada ventana con pares suficientes."""
    y, x = _trailing(asset, length), _trailing(market, length)
    valid = np.isfinite(y) & np.isfinite(x)
    count = valid.sum(axis=1)
    safe = np.maximum(count, 1)
    dx = np.where(valid, x - (np.where(valid, x, 0.0).sum(axis=1) / safe)[:, None], 0.0)
    dy = np.where(valid, y - (np.where(valid, y, 0.0).sum(axis=1) / safe)[:, None], 0.0)
    spread = np.einsum("ij,ij->i", dx, dx)
    slope = np.einsum("ij,ij->i", dx, dy) / np.where(spread > 0, spread, 1.0)
    residual = np.where(valid, dy - slope[:, None] * dx, 0.0)
    squares = np.einsum("ij,ij->i", residual, residual)
    defined = (count >= _minimum(length)) & (spread > 0)
    return np.where(defined, np.sqrt(squares / np.maximum(count - 2, 1)), np.nan)


def asset_features(close, volume, market_close, filings):
    """Características de un activo en cada sesión con mercado, con NaN donde no se definen.

    `close`, `volume` y `market_close` son series densas de las mismas sesiones, con NaN donde
    falta el valor. `filings` son las sesiones de `filing_sessions`. La fila t solo depende de
    las posiciones hasta t de las tres series y de las presentaciones conocidas en t.
    """
    close, volume = np.asarray(close, dtype=np.float64), np.asarray(volume, dtype=np.float64)
    market_close = np.asarray(market_close, dtype=np.float64)
    length = len(close)
    _require(
        close.ndim == 1 and close.shape == volume.shape == market_close.shape,
        "El cierre, el volumen y el mercado deben ser series de las mismas sesiones",
    )
    _require(
        np.all(close[np.isfinite(close)] > 0)
        and np.all(market_close[np.isfinite(market_close)] > 0)
        and np.all(volume[np.isfinite(volume)] >= 0),
        "Los cierres deben ser positivos y los volúmenes no negativos",
    )
    filings = np.asarray(filings, dtype=np.int64)
    _require(
        filings.ndim == 1 and np.all(np.diff(filings) > 0) and np.all(filings >= 0),
        "Las sesiones de las presentaciones deben ser únicas y crecientes",
    )
    returns = close / _lagged(close, 1) - 1
    market = market_close / _lagged(market_close, 1) - 1
    anchored = _last_valid(close, MOMENTUM_TOLERANCE)
    logs = np.log1p(volume)
    trailing_high = _trailing(close, YEAR)
    observed = np.isfinite(trailing_high).sum(axis=1)
    high = np.where(np.isfinite(trailing_high), trailing_high, -np.inf).max(axis=1)
    sessions = np.arange(length)
    latest = np.searchsorted(filings, sessions, side="right") - 1
    # Un activo sin presentaciones, como un fondo cotizado, no tiene ninguna sesión conocida.
    known = latest >= 0
    age = np.full(length, -1, dtype=np.int64)
    age[known] = sessions[known] - filings[latest[known]]
    columns = dict(
        reversal_1=returns,
        reversal_5=close / _lagged(close, 5) - 1,
        momentum_12_1=_lagged(anchored, MONTH) / _lagged(anchored, YEAR) - 1,
        idiosyncratic_volatility_20=_residual_volatility(returns, market, 20),
        idiosyncratic_volatility_60=_residual_volatility(returns, market, 60),
        volume_shock_20=logs - _lagged(_window_mean(logs, 20, _minimum(20)), 1),
        relative_turnover_20_252=_window_mean(logs, 20, _minimum(20))
        - _window_mean(logs, YEAR, _minimum(YEAR)),
        distance_to_52_week_high=np.where(observed >= _minimum(YEAR), close / high - 1, np.nan),
        sessions_since_filing=np.where(age >= 0, age.astype(np.float64), np.nan),
        recent_filing_5=np.where(age >= 0, (age < RECENT_FILING).astype(np.float64), np.nan),
    )
    return np.column_stack([columns[name] for name in FEATURES])


def session_ranks(values, present):
    """Rangos en [−1, 1] de cada característica y sesión entre los activos con precio.

    `values` tiene forma (activos, sesiones, características) y `present` marca los activos
    con precio en cada sesión. Con n valores válidos, el rango medio r de 1 a n pasa a
    2 (r − 1) / (n − 1) − 1, y un único valor vale cero. Devuelve los rangos, con cero donde
    falta el valor o el precio, y la máscara de esas ausencias.
    """
    values, present = np.asarray(values, dtype=np.float64), np.asarray(present)
    _require(
        values.ndim == 3 and present.shape == values.shape[:2] and present.dtype == np.bool_,
        "Los rangos necesitan activos, sesiones y características con su presencia",
    )
    valid = present[..., None] & np.isfinite(values)
    ranks = rankdata(np.where(valid, values, np.nan), axis=0, nan_policy="omit")
    count = valid.sum(axis=0, keepdims=True)
    scaled = np.where(count > 1, 2 * (ranks - 1) / np.maximum(count - 1, 1) - 1, 0.0)
    return np.where(valid, scaled, 0.0), ~valid
