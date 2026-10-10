"""Métricas financieras de series de patrimonio por sesión para el informe de políticas.

Parten del patrimonio valorado en cada cierre, nunca de retornos residuales. Las
convenciones son las de `financial_conventions`, comunes con la cartera larga y corta de
la comparación predictiva: sesiones por año del mercado (252 en US y 243 en CN), tipo sin
riesgo cero, rentabilidad anualizada geométrica, volatilidad muestral, Sortino con objetivo
cero y drawdown desde el máximo previo incluido el capital inicial. Una misma serie da las
mismas cifras en los dos informes.

El remuestreo usa los índices del bootstrap circular por bloques de `financial_conventions`,
y todas las series de una familia comparten las mismas réplicas. Las métricas remuestreadas
no dependen del orden de las sesiones y se calculan con las veces que aparece cada una. El
drawdown máximo sí depende del orden y aquí solo se publica como estimación puntual, porque
recorrer cada réplica en orden multiplica el coste del informe.
"""

import math

import numpy as np

from . import financial_conventions
from .financial_conventions import (
    CONVENTIONS,
    resampled,
    resamples,
    sessions_per_year,
    statistics,
)
from .paired_comparisons import family_intervals

RESAMPLED = ("annualized_return", "volatility", "sharpe", "sortino", "mean_session_return")
# Nombre de cada métrica del informe en las convenciones comunes.
_COMMON = dict(mean_session_return="mean_return")
MAX_REPLICATES = 100_000


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
    return float(financial_conventions.max_drawdown(session_returns(nav)[:, None])[0])


def _defined(value):
    value = float(value)
    return value if math.isfinite(value) else None


def equity_metrics(nav, *, market, turnover=None, costs=None):
    """Métricas de una serie de patrimonio del mercado, con giro y costes de la contabilidad.

    Las rentabilidades se dan en tanto por uno y en porcentaje. Sharpe y Sortino son None
    si no están definidos, por ejemplo con la cartera siempre en efectivo.
    """
    values = check_nav(nav)
    returns = values[1:] / values[:-1] - 1
    point = {
        name: float(value[0])
        for name, value in statistics(returns[:, None], sessions_per_year(market)).items()
    }
    for name, value in (("turnover", turnover), ("costs", costs)):
        _require(
            value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0),
            f"{name} debe ser un número finito no negativo",
        )
    return dict(
        sessions=len(returns),
        initial_nav=float(values[0]),
        final_nav=float(values[-1]),
        cumulative_return=point["cumulative_return"],
        cumulative_return_percent=100 * point["cumulative_return"],
        annualized_return=point["annualized_return"],
        annualized_return_percent=100 * point["annualized_return"],
        log_growth=_defined(point["log_growth"]),
        mean_session_return=point["mean_return"],
        volatility=point["volatility"],
        sharpe=_defined(point["sharpe"]),
        sortino=_defined(point["sortino"]),
        max_drawdown=point["max_drawdown"],
        turnover=turnover,
        costs=costs,
        ruined=bool(values[-1] == 0),
    )


def _pair(lower, upper):
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return None
    return [float(lower), float(upper)]


def block_bootstrap(
    returns, *, market, base, block_length, replicates, seed, confidence=0.95, sensitivity=()
):
    """Intervalos por bloques de cada serie y diferencias emparejadas frente a `base`.

    `returns` asigna a cada brazo sus retornos simples sobre las mismas sesiones del
    mercado, en el mismo orden. Todas las series se remuestrean con las mismas réplicas. Para cada
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
    annual = sessions_per_year(market)
    estimate = statistics(matrix, annual)
    points = {
        arm: {name: float(estimate[_COMMON.get(name, name)][j]) for name in RESAMPLED}
        for j, arm in enumerate(arms)
    }

    def family(length):
        if length >= sessions:
            return None, "Se necesitan más sesiones que la longitud del bloque"
        draws = {name: np.empty((replicates, len(arms))) for name in RESAMPLED}
        for offset, index in resamples(
            sessions, block_length=length, replicates=replicates, seed=seed
        ):
            values = resampled(matrix, annual, index, path=False)
            for name in RESAMPLED:
                draws[name][offset : offset + len(index)] = values[_COMMON.get(name, name)]
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
                    bounds = family_intervals(estimate[usable], difference[:, usable], confidence)
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
        market=market,
        sessions_per_year=annual,
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
