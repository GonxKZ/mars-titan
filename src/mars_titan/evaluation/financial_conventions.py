"""Convenciones financieras comunes de la cartera larga y corta y del informe de políticas.

`long_short` y `financial_metrics` calculan aquí sus estadísticos, así que una misma serie
de retornos da las mismas cifras en los dos informes. Las convenciones se declararon antes
de ver resultados (`docs/research/long-short-portfolio.md`):

- Un año tiene 252 sesiones en US y 243 en CN.
- El tipo sin riesgo es cero, porque el efectivo no remunera. Sharpe y Sortino usan los
  retornos simples por sesión sin restar ningún tipo.
- La riqueza se compone desde 1 con log(1 + r). Una sesión con r ≤ −1 arruina la serie, que
  queda con rentabilidad acumulada y anualizada −1 y drawdown 1.
- La rentabilidad anualizada es geométrica, exp(G · A / n) − 1, con G la suma de los
  log(1 + r) de las n sesiones y A las sesiones por año del mercado.
- La volatilidad es la desviación típica muestral (ddof=1) por la raíz de A. El Sharpe no
  está definido si esa desviación no supera 64 épsilon de la media en valor absoluto,
  porque entonces solo mide redondeo.
- La desviación a la baja de Sortino es la raíz de la media de min(r, 0)², con objetivo
  cero y denominador n. Sin sesiones con pérdidas el Sortino no está definido.
- El drawdown máximo es la mayor caída relativa de la riqueza desde su máximo previo,
  contando el capital inicial, y queda entre 0 y 1.

El remuestreo es el bootstrap circular por bloques de `paired_comparisons`, con un PCG64
sembrado que sortea las réplicas en lotes fijos de `CHUNK`. Con la misma semilla todos los
consumidores obtienen los mismos índices. En cada réplica, los estadísticos que no dependen
del orden se calculan con las veces que aparece cada sesión, y el drawdown con el
recorrido remuestreado en su orden.
"""

import math

import numpy as np

from .paired_comparisons import circular_block_indices

SESSIONS_PER_YEAR = {"US": 252, "CN": 243}
RISK_FREE_RATE = 0.0
# Réplicas por lote del generador. Forma parte de la convención: el generador de enteros
# acotados de NumPy descarta la mitad sobrante de su último sorteo de 64 bits en cada
# llamada, así que otro tamaño de lote podría cambiar los índices de una misma semilla.
CHUNK = 32
STATISTICS = (
    "cumulative_return",
    "annualized_return",
    "log_growth",
    "mean_return",
    "volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
)
CONVENTIONS = dict(
    sessions_per_year=SESSIONS_PER_YEAR,
    risk_free_rate=RISK_FREE_RATE,
    returns="simple_session_returns",
    wealth="compounded_from_one_with_ruin_at_minus_one",
    annualized_return="geometric",
    volatility="sample_std_ddof_1",
    sharpe="undefined_when_std_within_64_eps_of_mean",
    downside_deviation="root_mean_square_of_negative_returns_target_zero",
    max_drawdown="running_peak_including_initial_capital",
    resampling=dict(
        method="circular_block_bootstrap",
        unit="session",
        bit_generator="PCG64",
        chunk=CHUNK,
        order_free_statistics="session_counts",
        max_drawdown="ordered_resampled_path",
    ),
)
_FLAT = 64 * np.finfo(np.float64).eps


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def sessions_per_year(market):
    """Sesiones por año del mercado declarado."""
    _require(market in SESSIONS_PER_YEAR, f"No hay sesiones por año declaradas para {market}")
    return SESSIONS_PER_YEAR[market]


def growth_terms(returns):
    """log(1 + r) de cada sesión, con −inf en las que arruinan la serie."""
    with np.errstate(divide="ignore"):
        return np.log1p(np.maximum(np.asarray(returns, dtype=np.float64), -1.0))


def _ratios(mean, spread, downside, annual):
    scale = math.sqrt(annual)
    with np.errstate(divide="ignore", invalid="ignore"):
        active = spread > _FLAT * np.maximum(np.abs(mean), 1e-300)
        sharpe = np.where(active, mean / np.where(active, spread, 1.0) * scale, np.nan)
        losses = downside > 0
        sortino = np.where(losses, mean / np.where(losses, downside, 1.0) * scale, np.nan)
    return spread * scale, sharpe, sortino


def _drawdown(terms):
    """Drawdown máximo a lo largo del penúltimo eje a partir de los log(1 + r)."""
    with np.errstate(invalid="ignore"):
        log_wealth = np.cumsum(terms, axis=-2)
        peak = np.maximum.accumulate(np.maximum(log_wealth, 0.0), axis=-2)
        drawdown = (1 - np.exp(log_wealth - peak)).max(axis=-2)
    return np.nan_to_num(drawdown, nan=1.0)


def max_drawdown(returns):
    """Drawdown máximo de series de retornos en el penúltimo eje."""
    return _drawdown(growth_terms(returns))


def statistics(returns, annual):
    """Estadísticos de series de retornos en el penúltimo eje (``[..., sesiones, series]``).

    ``annual`` son las sesiones por año del mercado. Devuelve un vector por estadístico con
    la forma de entrada sin el eje de sesiones. Lo no definido vale NaN.
    """
    returns = np.asarray(returns, dtype=np.float64)
    sessions = returns.shape[-2]
    _require(sessions >= 1, "Se necesita al menos una sesión")
    terms = growth_terms(returns)
    growth = terms.sum(axis=-2)
    mean = returns.mean(axis=-2)
    spread = returns.std(axis=-2, ddof=1) if sessions > 1 else np.full(mean.shape, np.nan)
    downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2, axis=-2))
    volatility, sharpe, sortino = _ratios(mean, spread, downside, annual)
    return dict(
        cumulative_return=np.expm1(growth),
        annualized_return=np.expm1(growth * annual / sessions),
        log_growth=growth,
        mean_return=mean,
        volatility=volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=_drawdown(terms),
    )


def resamples(sessions, *, block_length, replicates, seed):
    """Lotes de índices del bootstrap circular por bloques: (primera réplica, [lote, sesiones])."""
    for name, value, low in (
        ("sesiones", sessions, 2),
        ("bloque", block_length, 1),
        ("réplicas", replicates, 1),
        ("semilla", seed, 0),
    ):
        _require(type(value) is int and low <= value, f"Parámetro {name} inválido")
    _require(block_length < sessions, "Se necesitan más sesiones que la longitud del bloque")
    rng = np.random.default_rng(seed)
    for offset in range(0, replicates, CHUNK):
        size = min(CHUNK, replicates - offset)
        yield offset, circular_block_indices(rng, size, sessions, block_length)


def session_counts(index, sessions):
    """Veces que aparece cada sesión en cada réplica, ``[réplicas, sesiones]`` en FP64."""
    replicates = index.shape[0]
    flat = (index + sessions * np.arange(replicates)[:, None]).ravel()
    counts = np.bincount(flat, minlength=replicates * sessions)
    return counts.reshape(replicates, sessions).astype(np.float64)


def weighted_sums(counts, values):
    """Suma de cada serie de ``values`` (``[..., sesiones, series]``) en cada réplica.

    Cada serie se reduce por separado con el mismo producto matriz-vector, de modo que dos
    series idénticas dan sumas idénticas en todas las réplicas.
    """
    values = np.asarray(values, dtype=np.float64)
    lead, columns = values.shape[:-2], values.shape[-1]
    result = np.empty((*lead, counts.shape[0], columns))
    for position in np.ndindex(*lead, columns):
        *outer, column = position
        series = np.ascontiguousarray(values[(*outer, slice(None), column)])
        result[(*outer, slice(None), column)] = counts @ series
    return result


def resampled(returns, annual, index, *, path=True):
    """Estadísticos de cada réplica de ``index`` (``[réplicas, sesiones]``).

    ``returns`` tiene forma ``[..., sesiones, series]`` y el resultado ``[..., réplicas,
    series]`` por estadístico. Con ``path`` falso no se calcula el drawdown, que necesita
    el recorrido ordenado y es lo más costoso, y queda en NaN.
    """
    returns = np.asarray(returns, dtype=np.float64)
    sessions = returns.shape[-2]
    counts = session_counts(index, sessions)
    terms = growth_terms(returns)
    ruin = np.isneginf(terms)
    mean = weighted_sums(counts, returns) / sessions
    # Centrar con la media de la serie evita la cancelación de E[r²] − E[r]² y da varianza
    # exactamente nula con una serie constante.
    centered = returns - returns.mean(axis=-2, keepdims=True)
    first = weighted_sums(counts, centered)
    second = weighted_sums(counts, centered**2)
    variance = np.maximum(second - first**2 / sessions, 0.0) / (sessions - 1)
    downside = np.sqrt(weighted_sums(counts, np.minimum(returns, 0.0) ** 2) / sessions)
    growth = weighted_sums(counts, np.where(ruin, 0.0, terms))
    if ruin.any():
        # Una ruina solo cuenta en las réplicas que la contienen.
        growth[weighted_sums(counts, ruin) > 0] = -math.inf
    volatility, sharpe, sortino = _ratios(mean, np.sqrt(variance), downside, annual)
    drawdown = np.full(mean.shape, np.nan)
    if path:
        drawdown = _drawdown(np.take(terms, index, axis=-2))
    return dict(
        cumulative_return=np.expm1(growth),
        annualized_return=np.expm1(growth * annual / sessions),
        log_growth=growth,
        mean_return=mean,
        volatility=volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=drawdown,
    )
