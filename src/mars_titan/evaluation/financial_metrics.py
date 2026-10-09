"""Métricas financieras comunes sobre series de patrimonio por sesión.

Las usan la etapa de políticas financieras y la cartera larga y corta de la comparación
predictiva. Parten del patrimonio valorado en cada cierre, nunca de retornos residuales.

Convenciones declaradas antes de ver resultados:

- Un año tiene 252 sesiones. La rentabilidad anualizada es geométrica.
- El tipo sin riesgo es cero, porque el efectivo de la simulación no remunera. Sharpe y
  Sortino usan retornos simples por sesión, sin restar ningún tipo.
- La desviación a la baja de Sortino es la raíz de la media de min(r, 0)², con objetivo cero
  y denominador igual al número de sesiones.
- La volatilidad usa la desviación típica muestral (ddof=1) de los retornos por sesión.

El remuestreo usa el bootstrap circular por bloques de `paired_comparisons`: cada réplica
cuenta cuántas veces aparece cada sesión, y todas las series de una familia comparten las
mismas réplicas. Las métricas remuestreadas no dependen del orden de las sesiones. El
drawdown máximo sí depende de él y solo se publica como estimación puntual.
"""

import math

import numpy as np

from .paired_comparisons import _intervals, circular_block_counts

PERIODS_PER_YEAR = 252
CONVENTIONS = dict(
    periods_per_year=PERIODS_PER_YEAR,
    risk_free_rate=0.0,
    returns="simple_session_returns_from_close_valuation",
    annualized_return="geometric",
    volatility="sample_std_ddof_1",
    downside_deviation="root_mean_square_of_negative_returns_target_zero",
)
RESAMPLED = ("annualized_return", "volatility", "sharpe", "sortino", "mean_session_return")
MAX_REPLICATES = 100_000
_CHUNK = 256


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def check_nav(nav):
    """Validar una serie de patrimonio: positiva al empezar, finita y sin deuda.

    Solo la última sesión puede valer cero, porque la ruina termina el episodio.
    """
    values = np.asarray(nav, dtype=np.float64)
    _require(
        values.ndim == 1
        and len(values) >= 2
        and np.isfinite(values).all()
        and (values[:-1] > 0).all()
        and values[-1] >= 0,
        "El patrimonio necesita al menos dos cierres finitos, positivos salvo una ruina final",
    )
    return values


def session_returns(nav):
    """Retornos simples entre cierres consecutivos."""
    values = check_nav(nav)
    return values[1:] / values[:-1] - 1


def max_drawdown(nav):
    """Mayor caída relativa desde un máximo anterior, entre 0 y 1."""
    values = check_nav(nav)
    return float(np.max(1 - values / np.maximum.accumulate(values)))


def _ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def _moments(returns):
    """Estadísticos que solo dependen de los retornos, no de su orden."""
    count = len(returns)
    mean = math.fsum(returns) / count
    variance = math.fsum((returns - mean) ** 2) / (count - 1) if count > 1 else 0.0
    downside = math.sqrt(math.fsum(np.minimum(returns, 0) ** 2) / count)
    growth = math.fsum(np.log1p(returns)) if (returns > -1).all() else -math.inf
    return mean, math.sqrt(variance), downside, growth


def _annualized(growth, sessions):
    if growth == -math.inf:
        return -1.0
    return math.expm1(growth * PERIODS_PER_YEAR / sessions)


def equity_metrics(nav, *, turnover=None, costs=None):
    """Métricas de una serie de patrimonio. `turnover` y `costs` vienen de la contabilidad.

    Las rentabilidades se dan en tanto por uno y en porcentaje. Sharpe y Sortino son None
    si su denominador es cero, por ejemplo con la cartera siempre en efectivo.
    """
    values = check_nav(nav)
    returns = values[1:] / values[:-1] - 1
    mean, deviation, downside, growth = _moments(returns)
    scale = math.sqrt(PERIODS_PER_YEAR)
    cumulative = values[-1] / values[0] - 1
    annualized = _annualized(growth, len(returns))
    for name, value in (("turnover", turnover), ("costs", costs)):
        _require(
            value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0),
            f"{name} debe ser un número finito no negativo",
        )
    return dict(
        sessions=len(returns),
        initial_nav=float(values[0]),
        final_nav=float(values[-1]),
        cumulative_return=float(cumulative),
        cumulative_return_percent=float(100 * cumulative),
        annualized_return=annualized,
        annualized_return_percent=100 * annualized,
        log_growth=growth if math.isfinite(growth) else None,
        mean_session_return=mean,
        volatility=deviation * scale,
        sharpe=None if deviation == 0 else mean / deviation * scale,
        sortino=None if downside == 0 else mean / downside * scale,
        max_drawdown=max_drawdown(values),
        turnover=turnover,
        costs=costs,
        ruined=bool(values[-1] == 0),
    )


def _resampled(returns, counts):
    """Métricas de cada réplica a partir de cuántas veces aparece cada sesión."""
    weights = counts.astype(np.float64)
    sessions = weights.sum(axis=1)
    mean = weights @ returns / sessions
    second = weights @ returns**2 / sessions
    variance = np.maximum(second - mean**2, 0) * sessions / (sessions - 1)
    deviation = np.sqrt(variance)
    downside = np.sqrt(weights @ np.minimum(returns, 0) ** 2 / sessions)
    # Una ruina (retorno −1) solo cuenta en las réplicas que la contienen.
    ruin = returns <= -1
    growth = weights @ np.log1p(np.where(ruin, 0.0, returns))
    growth[weights[:, ruin].sum(axis=1) > 0] = -math.inf
    scale = math.sqrt(PERIODS_PER_YEAR)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = dict(
            annualized_return=np.expm1(growth * PERIODS_PER_YEAR / sessions),
            volatility=deviation * scale,
            sharpe=np.where(deviation > 0, mean / deviation * scale, np.nan),
            sortino=np.where(downside > 0, mean / downside * scale, np.nan),
            mean_session_return=mean,
        )
    return result


def _point(returns):
    mean, deviation, downside, growth = _moments(returns)
    scale = math.sqrt(PERIODS_PER_YEAR)
    return dict(
        annualized_return=_annualized(growth, len(returns)),
        volatility=deviation * scale,
        sharpe=math.nan if deviation == 0 else mean / deviation * scale,
        sortino=math.nan if downside == 0 else mean / downside * scale,
        mean_session_return=mean,
    )


def _pair(lower, upper):
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return None
    return [float(lower), float(upper)]


def block_bootstrap(
    returns, *, base, block_length, replicates, seed, confidence=0.95, sensitivity=()
):
    """Intervalos por bloques de cada serie y diferencias emparejadas frente a `base`.

    `returns` asigna a cada brazo sus retornos simples sobre las mismas sesiones, en el
    mismo orden. Todas las series se remuestrean con las mismas réplicas. Para cada
    métrica, las diferencias `brazo − base` forman una familia con intervalos marginales
    percentiles e intervalos simultáneos por máximo estudentizado. Un intervalo simultáneo
    que excluye el cero es la regla para afirmar una diferencia dentro de la familia.
    """
    _require(isinstance(returns, dict) and base in returns, "Falta la serie base")
    arms = list(returns)
    matrix = np.column_stack([np.asarray(returns[arm], dtype=np.float64) for arm in arms])
    sessions = matrix.shape[0]
    _require(
        len(arms) >= 2 and sessions >= 2 and np.isfinite(matrix).all() and (matrix >= -1).all(),
        "Las series necesitan al menos dos sesiones comunes con retornos finitos",
    )
    for name, value, low, high in (
        ("bloque", block_length, 1, 100_000),
        ("réplicas", replicates, 10, MAX_REPLICATES),
        ("semilla", seed, 0, 2**63 - 1),
    ):
        _require(type(value) is int and low <= value <= high, f"Parámetro {name} inválido")
    _require(
        isinstance(confidence, float) and 0.5 <= confidence < 1,
        "El nivel de confianza no es válido",
    )
    points = {arm: _point(matrix[:, j]) for j, arm in enumerate(arms)}

    def family(length):
        if length >= sessions:
            return None, "Se necesitan más sesiones que la longitud del bloque"
        rng = np.random.default_rng(seed)
        draws = {name: np.empty((replicates, len(arms))) for name in RESAMPLED}
        for offset in range(0, replicates, _CHUNK):
            size = min(_CHUNK, replicates - offset)
            counts = circular_block_counts(rng, size, sessions, length)
            for j in range(len(arms)):
                values = _resampled(matrix[:, j], counts)
                for name in RESAMPLED:
                    draws[name][offset : offset + size, j] = values[name]
        return draws, None

    def summarize(draws, reason):
        tail = (1 - confidence) / 2
        result = dict(arms={}, differences={}, reason=reason)
        for j, arm in enumerate(arms):
            entry = {}
            for name in RESAMPLED:
                value = points[arm][name]
                interval = None
                if draws is not None and math.isfinite(value):
                    column = draws[name][:, j]
                    if np.isfinite(column).all():
                        lower, upper = np.quantile(column, [tail, 1 - tail], method="linear")
                        interval = _pair(lower, upper)
                entry[name] = dict(
                    estimate=value if math.isfinite(value) else None, interval=interval
                )
            result["arms"][arm] = entry
        others = [arm for arm in arms if arm != base]
        b = arms.index(base)
        for name in RESAMPLED:
            estimate = np.array([points[arm][name] - points[base][name] for arm in others])
            rows = {
                arm: dict(estimate=None, interval=None, simultaneous_interval=None)
                for arm in others
            }
            known = np.isfinite(estimate)
            if draws is not None and known.any():
                columns = [arms.index(arm) for arm in others]
                difference = draws[name][:, columns] - draws[name][:, [b]]
                usable = known & np.isfinite(difference).all(axis=0)
                if usable.any():
                    bounds = _intervals(estimate[usable], difference[:, usable], confidence)
                    for k, index in enumerate(np.flatnonzero(usable)):
                        rows[others[index]].update(
                            interval=_pair(bounds["lower"][k], bounds["upper"][k]),
                            simultaneous_interval=_pair(
                                bounds["joint_lower"][k], bounds["joint_upper"][k]
                            ),
                        )
            for k, arm in enumerate(others):
                if known[k]:
                    rows[arm]["estimate"] = float(estimate[k])
                joint = rows[arm]["simultaneous_interval"]
                rows[arm]["simultaneous_excludes_zero"] = (
                    None if joint is None else bool(joint[0] > 0 or joint[1] < 0)
                )
            result["differences"][name] = rows
        return result

    draws, reason = family(block_length)
    main = summarize(draws, reason)
    return dict(
        schema_version=1,
        kind="financial_block_bootstrap",
        conventions=CONVENTIONS,
        base=base,
        resampling=dict(
            method="circular_block_bootstrap",
            unit="session",
            sessions=sessions,
            block_length=block_length,
            replicates=replicates,
            seed=seed,
            bit_generator="PCG64",
            numpy_version=np.__version__,
        ),
        confidence=confidence,
        multiplicity=dict(
            method="max_absolute_studentized_bootstrap",
            family="differences_with_base_per_metric",
            family_size=len(arms) - 1,
        ),
        **main,
        sensitivity=[
            dict(block_length=length, **summarize(*family(length))) for length in sensitivity
        ],
    )
