"""Reconstrucción de precios sin ajustar invirtiendo el ajuste del proveedor original.

Los CSV de FinMultiTime distribuyen OHLC ajustados por splits y dividendos, volumen
ajustado por splits y las columnas ``Dividends`` y ``Stock Splits`` en la fila de la
fecha ex. El proveedor multiplica los precios anteriores a cada fecha ex ``e`` por
``1 - D_e / C_{e-1}``, con ``C_{e-1}`` el cierre negociado de la sesión previa en
acciones posteriores a todos los splits. Con ``A`` el cierre ajustado, la inversa del
factor de dividendo es aditiva:

    1 / F_{e-1} = 1 / F_e + D_e / A_{e-1}

Por tanto, el precio negociado de cada sesión ``t`` es

    P_t = A_t * K_t * (gamma + S_t),    S_t = sum_{e > t} D_e / A_{e-1}

donde ``K_t`` acumula las razones de split posteriores a ``t`` y ``gamma`` es una
constante por activo. ``gamma`` recoge los dividendos posteriores al corte temporal,
incluidos los que el proveedor aplicó después de la última fila del archivo. Se
estima con las últimas sesiones anteriores al corte, exigiendo que esos cierres
caigan en la unidad mínima de cotización. No se leen precios ni dividendos
posteriores al corte. Las razones de split posteriores se leen solo como escala.

El dividendo publicado está en unidades posteriores a los splits estrictamente
posteriores a su fecha ex. Si coinciden split y dividendo, el proveedor no divide
el dividendo por ese split. El módulo invierte ese procedimiento sin corregirlo.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pv

METHOD = "provider_inverse_factor_terminal_constant_v1"
PRICE_COLUMNS = ("Open", "High", "Low", "Close")
SOURCE_COLUMNS = ("Date", *PRICE_COLUMNS, "Volume", "Dividends", "Stock Splits")

# Calendario de decimalización de EE. UU. (SEC, Release 34-42360, y GAO-05-535).
# La NYSE empezó el piloto el 28-08-2000 y Nasdaq completó la conversión el 09-04-2001.
US_DECIMAL_PILOT_START = np.datetime64("2000-08-28")
US_DECIMAL_COMPLETE = np.datetime64("2001-04-09")
US_FRACTION_TICK = 1 / 256
US_CENT_TICK = 0.01
US_SUBDOLLAR_TICK = 0.0001
CN_A_SHARE_TICK = 0.01
DEFAULT_RELATIVE_TOLERANCE = 2e-6
DEFAULT_FIT_WINDOWS = (60, 125, 250, 500)
DEFAULT_GAMMA_MAX = 1.25
MIN_FIT_SHARE = 0.9
MIN_FIT_MARGIN = 0.1
# Cuantil 0,9 aproximado de los residuos relativos de aciertos medidos en activos US y CN.
FIT_RELATIVE_TOLERANCE = 3e-7
FIT_MIN_SHARE = 0.6
CANDIDATE_CHUNK = 65_536


@dataclass(frozen=True)
class ProviderHistory:
    """Serie del proveedor en el orden del archivo original."""

    sessions: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    dividends: np.ndarray
    splits: np.ndarray

    def __post_init__(self):
        sessions = self.sessions
        if sessions.ndim != 1 or sessions.dtype != np.dtype("datetime64[D]"):
            raise ValueError("Las sesiones deben ser fechas diarias en una dimensión")
        if len(sessions) and (np.diff(sessions) <= np.timedelta64(0, "D")).any():
            raise ValueError("Las sesiones del proveedor deben ser estrictamente crecientes")
        arrays = [getattr(self, field.name) for field in fields(self)[1:]]
        if any(a.shape != sessions.shape or a.dtype != np.float64 for a in arrays):
            raise ValueError("Las columnas del proveedor no tienen la forma de las sesiones")
        for values in (self.dividends, self.splits):
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Los dividendos y splits deben ser finitos y no negativos")

    def until(self, cutoff) -> "ProviderHistory":
        keep = self.sessions <= np.datetime64(cutoff, "D")
        return ProviderHistory(*(getattr(self, field.name)[keep] for field in fields(self)))

    def split_ratio_after(self, cutoff) -> float:
        """Escala de los splits posteriores al corte, sin leer otros campos."""
        after = self.splits[self.sessions > np.datetime64(cutoff, "D")]
        return float(np.prod(after[after > 0]))


def read_provider_history(path: Path) -> ProviderHistory:
    """Leer un CSV original sin reparar filas. Los precios no válidos quedan como NaN."""
    table = pv.read_csv(
        path,
        read_options=pv.ReadOptions(use_threads=False),
        convert_options=pv.ConvertOptions(
            column_types={
                name: pa.string() if name == "Date" else pa.float64() for name in SOURCE_COLUMNS
            },
            include_columns=list(SOURCE_COLUMNS),
            strings_can_be_null=False,
        ),
    )
    if table.column_names != list(SOURCE_COLUMNS):
        raise ValueError(f"Faltan columnas del proveedor en {Path(path).name}")
    dates = table.column("Date").to_pylist()
    if any(len(value) < 10 for value in dates):
        raise ValueError(f"Fecha ilegible en {Path(path).name}")
    sessions = np.array([value[:10] for value in dates], dtype="datetime64[D]")
    values = [
        table.column(name).to_numpy(zero_copy_only=False).astype(np.float64)
        for name in SOURCE_COLUMNS[1:]
    ]
    for events in values[-2:]:
        if np.isnan(events).any():
            raise ValueError(f"Acción corporativa ausente en {Path(path).name}")
    return ProviderHistory(sessions, *values)


def split_multipliers(splits) -> np.ndarray:
    """``K_t``: producto de las razones de split estrictamente posteriores a ``t``."""
    splits = np.asarray(splits, dtype=np.float64)
    if splits.ndim != 1 or not np.isfinite(splits).all() or (splits < 0).any():
        raise ValueError("Las razones de split deben ser finitas y no negativas")
    ratios = np.where(splits > 0, splits, 1.0)
    result = np.ones(len(splits))
    if len(splits) > 1:
        result[:-1] = np.cumprod(ratios[:0:-1])[::-1]
    return result


def dividend_offsets(close, dividends) -> np.ndarray:
    """``S_t``: suma de ``D_e / A_{e-1}`` para las fechas ex posteriores a ``t``.

    Es NaN en las filas anteriores a un dividendo cuyo cierre previo no es válido. Un
    dividendo en la primera fila no ajusta ninguna sesión conservada.
    """
    close = np.asarray(close, dtype=np.float64)
    dividends = np.asarray(dividends, dtype=np.float64)
    if close.ndim != 1 or close.shape != dividends.shape:
        raise ValueError("Cierres y dividendos necesitan la misma longitud")
    if not np.isfinite(dividends).all() or (dividends < 0).any():
        raise ValueError("Los dividendos deben ser finitos y no negativos")
    events = np.flatnonzero(dividends > 0)
    events = events[events > 0]
    previous = close[events - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(
            np.isfinite(previous) & (previous > 0), dividends[events] / previous, np.nan
        )
    after = np.zeros(len(events) + 1)
    after[:-1] = np.cumsum(terms[::-1])[::-1]  # NaN se propaga hacia sesiones anteriores.
    return after[np.searchsorted(events, np.arange(len(close)), side="right")]


def price_increments(market: str, sessions, prices) -> np.ndarray:
    """Unidades mínimas de cotización admisibles por fila, con NaN si solo hay una."""
    sessions = np.asarray(sessions, dtype="datetime64[D]")
    prices = np.asarray(prices, dtype=np.float64)
    if sessions.shape != prices.shape or sessions.ndim != 1:
        raise ValueError("Sesiones y precios necesitan la misma forma")
    ticks = np.full((len(prices), 2), np.nan)
    if market == "CN":
        ticks[:, 0] = CN_A_SHARE_TICK
        return ticks
    if market != "US":
        raise ValueError(f"Mercado sin reglas de cotización: {market}")
    fractional = sessions < US_DECIMAL_PILOT_START
    transition = ~fractional & (sessions < US_DECIMAL_COMPLETE)
    decimal = sessions >= US_DECIMAL_COMPLETE
    ticks[fractional | transition, 0] = US_FRACTION_TICK
    ticks[transition, 1] = US_CENT_TICK
    ticks[decimal, 0] = np.where(prices[decimal] < 1, US_SUBDOLLAR_TICK, US_CENT_TICK)
    return ticks


def _hits(prices, ticks, relative_tolerance):
    """Precios en rejilla. ``prices`` admite una dimensión extra inicial de candidatos."""
    tolerance = relative_tolerance * prices
    with np.errstate(invalid="ignore"):
        hits = np.zeros(prices.shape, dtype=bool)
        for column in range(ticks.shape[1]):
            tick = ticks[:, column]
            residual = np.abs(prices - np.round(prices / tick) * tick)
            hits |= residual <= tolerance
    return hits & np.isfinite(prices) & (prices > 0)


def price_grid(
    market: str, sessions, prices, relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE
):
    """Comprobar si cada precio cae en su rejilla y la probabilidad de acertar por azar.

    La probabilidad de azar supone un residuo uniforme dentro de cada unidad mínima,
    ``min(1, sum 2 * tol * P / tick)``. Las filas no finitas devuelven ``False`` y NaN.
    """
    if not 0 < relative_tolerance < 1e-3:
        raise ValueError("La tolerancia relativa debe ser positiva y pequeña")
    prices = np.asarray(prices, dtype=np.float64)
    ticks = price_increments(market, sessions, prices)
    hits = _hits(prices, ticks, relative_tolerance)
    defined = np.isfinite(prices) & (prices > 0)
    with np.errstate(invalid="ignore"):
        chance = np.minimum(1.0, np.nansum(2 * relative_tolerance * prices[:, None] / ticks, 1))
    return hits, np.where(defined, chance, np.nan)


def grid_concordance(on_grid, chance) -> dict:
    """Resumir la concordancia observada frente a la esperada por azar."""
    on_grid, chance = np.asarray(on_grid, dtype=bool), np.asarray(chance, dtype=np.float64)
    defined = np.isfinite(chance)
    rows = int(defined.sum())
    if not rows:
        return {"rows": 0, "on_grid": 0, "rate": None, "chance": None, "excess": None}
    rate = float(on_grid[defined].mean())
    expected = float(chance[defined].mean())
    excess = None if expected >= 1 else (rate - expected) / (1 - expected)
    return {
        "rows": rows,
        "on_grid": int(on_grid[defined].sum()),
        "rate": rate,
        "chance": expected,
        "excess": excess,
    }


def fit_terminal_constant(
    market: str,
    sessions,
    scaled_close,
    offsets,
    *,
    windows: tuple[int, ...] = DEFAULT_FIT_WINDOWS,
    gamma_max: float = DEFAULT_GAMMA_MAX,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> dict:
    """Elegir ``gamma`` en ``[1, gamma_max]`` que sitúa más cierres finales en rejilla.

    ``scaled_close`` es ``A_t * K_t``. Los candidatos son exactamente los valores que
    llevan el último cierre válido a un múltiplo de su unidad mínima. La
    identificación cuenta aciertos con ``FIT_RELATIVE_TOLERANCE``, más estricta que la
    de validación, porque con precios altos dos candidatos contiguos difieren menos
    que la tolerancia de validación en sesiones de precio parecido. Si el mejor no
    supera al segundo por ``MIN_FIT_MARGIN`` de la ventana, se amplía la ventana con
    los candidatos que seguían por encima de ``FIT_MIN_SHARE``. El candidato elegido
    debe situar además ``MIN_FIT_SHARE`` de la ventana en rejilla con
    ``relative_tolerance``.
    """
    sessions = np.asarray(sessions, dtype="datetime64[D]")
    scaled_close = np.asarray(scaled_close, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    if not (windows and windows[0] >= 2 and list(windows) == sorted(set(windows))):
        raise ValueError("Las ventanas deben ser crecientes y de al menos dos filas")
    if not 1 < gamma_max <= 2:
        raise ValueError("El intervalo de gamma no es válido")
    valid = np.flatnonzero(np.isfinite(scaled_close) & (scaled_close > 0) & np.isfinite(offsets))
    result = {
        "gamma": None,
        "hits": 0,
        "runner_up": 0,
        "window_rows": int(min(len(valid), windows[0])),
    }
    if len(valid) < windows[0]:
        result["reason"] = "insufficient_rows"
        return result
    last = valid[-1]
    low = scaled_close[last] * (1 + offsets[last])
    tick = price_increments(market, sessions[last : last + 1], np.array([low]))[0, 0]
    high = scaled_close[last] * (gamma_max + offsets[last])
    # El cierre almacenado tiene error de redondeo. Un gamma ligeramente inferior a 1
    # dentro de la tolerancia representa la ausencia de dividendos posteriores.
    slack = 10 * relative_tolerance
    numbers = np.arange(np.floor(low * (1 - slack) / tick), np.floor(high / tick) + 1)
    gammas = numbers * tick / scaled_close[last] - offsets[last]
    gammas = gammas[(gammas >= 1 - slack) & (gammas <= gamma_max)]
    result["candidates"] = int(len(gammas))
    if not len(gammas):
        result["reason"] = "no_candidates"
        return result
    for window in windows:
        rows = valid[-window:]
        base = scaled_close[rows] * (1 + offsets[rows])
        ticks = price_increments(market, sessions[rows], base)
        counts = np.zeros(len(gammas), dtype=np.int64)
        for start in range(0, len(gammas), CANDIDATE_CHUNK):
            chunk = gammas[start : start + CANDIDATE_CHUNK, None]
            prices = scaled_close[rows][None, :] * (chunk + offsets[rows][None, :])
            hits = _hits(prices, ticks, FIT_RELATIVE_TOLERANCE)
            counts[start : start + CANDIDATE_CHUNK] = hits.sum(1)
        order = np.argsort(-counts, kind="stable")
        best = int(counts[order[0]])
        runner_up = int(counts[order[1]]) if len(order) > 1 else 0
        result.update(
            hits=best,
            runner_up=runner_up,
            window_rows=int(len(rows)),
            first_fit_session=str(sessions[rows[0]]),
        )
        if best < FIT_MIN_SHARE * len(rows):
            result["reason"] = "no_grid_solution"
            return result
        if best - runner_up >= MIN_FIT_MARGIN * len(rows):
            gamma = gammas[order[0]]
            prices = scaled_close[rows] * (gamma + offsets[rows])
            accepted = int(_hits(prices, ticks, relative_tolerance).sum())
            result["validation_tolerance_hits"] = accepted
            if accepted < MIN_FIT_SHARE * len(rows):
                result["reason"] = "no_grid_solution"
                return result
            result["gamma"] = float(gamma)
            result.pop("reason", None)
            return result
        result["reason"] = "ambiguous"
        if len(rows) == len(valid):
            break
        gammas = gammas[counts >= FIT_MIN_SHARE * len(rows)]
    return result


def reconstruct_unadjusted(
    history: ProviderHistory,
    market: str,
    cutoff,
    *,
    windows: tuple[int, ...] = DEFAULT_FIT_WINDOWS,
    gamma_max: float = DEFAULT_GAMMA_MAX,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> dict:
    """Precios negociados hasta ``cutoff`` y diagnóstico del ajuste terminal."""
    split_after = history.split_ratio_after(cutoff)
    kept = history.until(cutoff)
    multipliers = split_multipliers(kept.splits) * split_after
    offsets = dividend_offsets(kept.close, kept.dividends)
    fit = fit_terminal_constant(
        market,
        kept.sessions,
        kept.close * multipliers,
        offsets,
        windows=windows,
        gamma_max=gamma_max,
        relative_tolerance=relative_tolerance,
    )
    gamma = np.nan if fit["gamma"] is None else fit["gamma"]
    scale = multipliers * (gamma + offsets)
    return {
        "history": kept,
        "fit": fit,
        "split_ratio_after_cutoff": split_after,
        "split_multiplier": multipliers,
        "dividend_offset": offsets,
        "scale": scale,
        "open": kept.open * scale,
        "high": kept.high * scale,
        "low": kept.low * scale,
        "close": kept.close * scale,
        "volume": kept.volume / multipliers,
        "raw_dividend": kept.dividends * multipliers,
    }


def shifted_events(values, sessions: int) -> np.ndarray:
    """Desplazar acciones corporativas para los controles negativos del método."""
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(values)
    if sessions > 0:
        result[sessions:] = values[:-sessions]
    elif sessions < 0:
        result[:sessions] = values[-sessions:]
    else:
        result[:] = values
    return result
