"""Cartera larga y corta por cuartiles del protocolo: selección, libro por sesión y estadísticos.

La regla se declara antes de evaluar en la sección ``long_short`` de la comparación:

- En cada sesión (mercado e instante de decisión) con al menos ``min_assets`` filas
  evaluadas se fija k = floor(fraction · n). Una fila es larga si como mucho k filas de la
  sesión tienen una predicción mayor o igual que la suya, y corta si como mucho k la tienen
  menor o igual. Un grupo empatado que cruza la frontera queda fuera entero, así que una
  predicción constante, como la del control cero, no abre posiciones.
- Cada larga pesa +long/k y cada corta −short/k del capital de la sesión. Lo que no se
  ejecuta queda en efectivo y no se renormaliza.
- Entrada en la apertura de la sesión siguiente y salida en su cierre, partiendo de
  efectivo en cada sesión. El coste por lado se cobra sobre el nocional de entrada, |w|, y
  sobre el de salida, |w| · cierre / apertura. El timbre chino se suma aparte.

Solo se usa la predicción emitida en la decisión. Los precios y permisos de ejecución
proceden de ``simulation.session_prices``. Los estadísticos de una serie de retornos
netos diarios son el rendimiento acumulado y anualizado, la media, la volatilidad, el
Sharpe sin tipo libre de riesgo, el drawdown máximo y la rotación media.
"""

import math

import numpy as np

from mars_titan.evaluation.paired_comparisons import circular_block_indices, family_intervals

KIND = "quartile_long_short_v1"
STATUS = "secondary_financial"
USE = "description_and_paired_contrasts_not_for_model_selection"
TIES = "exclude_tied_group_crossing_the_boundary"
HOLDING = "next_session_open_to_close_from_cash"
UNFILLED = "cash_without_renormalization"
PRICE_BASIS = "unadjusted_reconstructed"
SEEDS = "mean_of_seed_session_returns"
MARKET_RULES = {"US": "none", "CN": "cn_a_share_v1"}
STATISTICS = (
    "cumulative_return",
    "annualized_return",
    "mean_net_return",
    "volatility",
    "sharpe",
    "max_drawdown",
    "turnover",
)
# Sentido de mejora de cada estadístico, solo para describir los contrastes.
HIGHER_IS_BETTER = dict(
    cumulative_return=True,
    annualized_return=True,
    mean_net_return=True,
    volatility=None,
    sharpe=True,
    max_drawdown=False,
    turnover=None,
)
FIELDS = {
    "kind",
    "status",
    "declared_at",
    "use",
    "signal",
    "fraction",
    "min_assets",
    "ties",
    "exposure",
    "unfilled",
    "holding",
    "price_basis",
    "market_rules",
    "cost_bps_per_side",
    "annualization_sessions",
    "seeds",
    "statistics",
    "views",
}
# Simplificaciones declaradas con la regla. El informe las copia tal cual.
ASSUMPTIONS = dict(
    china_t_plus_one=(
        "T+1 impide vender en la misma sesión lo comprado en la apertura. La regla del "
        "protocolo liquida al cierre, así que el resultado chino es una cota sin esa fricción"
    ),
    short_sale=(
        "Se supone préstamo disponible sin coste para las cortas. En acciones A el préstamo "
        "de valores solo existe para una lista de valores y no hay datos de disponibilidad"
    ),
    exits=(
        "La salida al cierre se supone ejecutada. En China se cuentan las salidas con el "
        "cierre en la banda que la bloquearía"
    ),
    lots_and_participation="Pesos continuos, sin lotes, participación ni impacto",
    costs="Coste ilustrativo por lado, igual en ambos mercados y fechas, sin préstamo",
    survivorship="Población condicionada a seguir cotizando en marzo de 2025, sin bajas",
    cash="El efectivo no remunera y no se resta un tipo libre de riesgo",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _real(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def declaration(section):
    """Validar la sección declarada sin leer datos y devolverla."""
    _require(
        isinstance(section, dict) and set(section) == FIELDS,
        "La sección long_short no declara exactamente sus campos",
    )
    _require(
        section["kind"] == KIND
        and section["status"] == STATUS
        and section["use"] == USE
        and isinstance(section["declared_at"], str)
        and len(section["declared_at"]) == 10
        and section["signal"] == "point_prediction"
        and section["ties"] == TIES
        and section["holding"] == HOLDING
        and section["unfilled"] == UNFILLED
        and section["price_basis"] == PRICE_BASIS
        and section["market_rules"] == MARKET_RULES
        and section["seeds"] == SEEDS
        and section["views"] == "per_market",
        "La regla de la cartera no es la declarada en el protocolo",
    )
    exposure = section["exposure"]
    _require(
        _real(section["fraction"], 0.01, 0.5)
        and type(section["min_assets"]) is int
        and 4 <= section["min_assets"] <= 100_000
        and isinstance(exposure, dict)
        and set(exposure) == {"long", "short"}
        and all(_real(value, 0.0, 1.0) for value in exposure.values())
        and exposure["long"] + exposure["short"] > 0,
        "La fracción, el mínimo de activos o la exposición no son válidos",
    )
    costs = section["cost_bps_per_side"]
    _require(
        isinstance(costs, list)
        and costs
        and all(type(cost) is int and 0 <= cost <= 1000 for cost in costs)
        and costs == sorted(set(costs)),
        "Los costes deben ser enteros crecientes en puntos básicos",
    )
    annual = section["annualization_sessions"]
    _require(
        isinstance(annual, dict)
        and set(annual) == set(MARKET_RULES)
        and all(type(value) is int and 200 <= value <= 260 for value in annual.values()),
        "Las sesiones por año deben declararse por mercado",
    )
    statistics = section["statistics"]
    _require(
        isinstance(statistics, list) and statistics == list(STATISTICS),
        "Los estadísticos deben ser los declarados y en su orden",
    )
    return section


def quartile_weights(panel, *, fraction, min_assets, exposure):
    """Peso de cada fila del panel, en su orden canónico, según la regla de los extremos."""
    order = np.lexsort((panel.prediction, panel.session))
    ordered = panel.prediction[order]
    session = panel.session[order]
    position = np.arange(panel.rows) - panel.session_starts[session]
    start = np.ones(panel.rows, dtype=bool)
    start[1:] = (session[1:] != session[:-1]) | (ordered[1:] != ordered[:-1])
    group = np.cumsum(start) - 1
    first = position[start]
    last = np.r_[np.flatnonzero(start)[1:], panel.rows] - 1
    last = position[last]
    size = panel.session_samples[session]
    above = size - first[group]  # filas con predicción mayor o igual
    below = last[group] + 1  # filas con predicción menor o igual
    k = np.where(size >= min_assets, np.floor(fraction * size + 1e-12), 0).astype(np.int64)
    weights = np.zeros(panel.rows)
    long_ = (k > 0) & (above <= k)
    short = (k > 0) & (below <= k)
    weights[order[long_]] = exposure["long"] / k[long_]
    weights[order[short]] = -exposure["short"] / k[short]
    return weights


def _sessions(panel, values):
    return np.bincount(panel.session, weights=values, minlength=panel.sessions)


def session_book(panel, weights, execution):
    """Libro de una sesión por fila del panel: rendimiento, nocional, impuestos y recuentos.

    ``execution`` tiene vectores alineados con el panel (``open``, ``close``,
    ``long_entry``, ``short_entry``, ``buy_tax``, ``sell_tax``, ``exit_limit_long`` y
    ``exit_limit_short``). Devuelve sumas por sesión.
    """
    long_ = weights > 0
    short = weights < 0
    filled_long = long_ & execution["long_entry"]
    filled_short = short & execution["short_entry"]
    filled = filled_long | filled_short
    size = np.abs(weights)
    ratio = np.where(filled, execution["close"] / np.where(filled, execution["open"], 1.0), 1.0)
    gross = np.where(filled_long, size * (ratio - 1), 0.0) + np.where(
        filled_short, size * (1 - ratio), 0.0
    )
    traded = np.where(filled, size * (1 + ratio), 0.0)
    taxes = np.where(
        filled_long, size * (execution["buy_tax"] + ratio * execution["sell_tax"]), 0.0
    ) + np.where(filled_short, size * (execution["sell_tax"] + ratio * execution["buy_tax"]), 0.0)
    exits = (filled_long & execution["exit_limit_long"]) | (
        filled_short & execution["exit_limit_short"]
    )
    return dict(
        gross_return=_sessions(panel, gross),
        traded=_sessions(panel, traded),
        taxes=_sessions(panel, taxes),
        long_exposure=_sessions(panel, np.where(filled_long, size, 0.0)),
        short_exposure=_sessions(panel, np.where(filled_short, size, 0.0)),
        selected_long=_sessions(panel, long_.astype(float)).astype(np.int64),
        selected_short=_sessions(panel, short.astype(float)).astype(np.int64),
        filled_long=_sessions(panel, filled_long.astype(float)).astype(np.int64),
        filled_short=_sessions(panel, filled_short.astype(float)).astype(np.int64),
        exits_at_limit=_sessions(panel, exits.astype(float)).astype(np.int64),
    )


def net_returns(book, costs):
    """Rendimiento neto por sesión y coste: bruto − coste por lado · nocional − timbre."""
    return np.stack(
        [book["gross_return"] - cost / 10_000 * book["traded"] - book["taxes"] for cost in costs]
    )


def statistics(returns, traded, sessions_per_year):
    """Estadísticos de series diarias en el penúltimo eje (``[..., sesiones, brazos]``).

    El rendimiento acumulado y el drawdown se calculan con la riqueza compuesta desde 1.
    Una sesión con rendimiento de −100 % o peor arruina la serie: acumulado −1 y drawdown 1.
    El Sharpe no está definido (NaN) con volatilidad nula.
    """
    returns = np.asarray(returns, dtype=np.float64)
    sessions = returns.shape[-2]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_wealth = np.cumsum(np.log(np.maximum(1 + returns, 0.0)), axis=-2)
        total = log_wealth[..., -1, :]
        peak = np.maximum.accumulate(np.maximum(log_wealth, 0.0), axis=-2)
        drawdown = 1 - np.exp(log_wealth - peak)
        mean = returns.mean(axis=-2)
        spread = returns.std(axis=-2, ddof=1) if sessions > 1 else np.full(mean.shape, np.nan)
        scale = math.sqrt(sessions_per_year)
        active = spread > 64 * np.finfo(np.float64).eps * np.maximum(np.abs(mean), 1e-300)
        sharpe = np.where(active, mean / np.where(active, spread, 1.0) * scale, np.nan)
    return dict(
        cumulative_return=np.expm1(total),
        annualized_return=np.expm1(total * sessions_per_year / sessions),
        mean_net_return=mean,
        volatility=spread * scale,
        sharpe=sharpe,
        max_drawdown=np.nan_to_num(drawdown.max(axis=-2), nan=1.0),
        turnover=np.broadcast_to(np.asarray(traded, dtype=np.float64).mean(axis=-2), mean.shape),
    )


def _pair(lower, upper):
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return None
    return [float(lower), float(upper)]


def bootstrap(returns, traded, *, sessions_per_year, block_length, replicates, seed, chunk=32):
    """Estadísticos y réplicas por bloques circulares de sesiones del mismo mercado.

    ``returns`` tiene forma [costes, sesiones, brazos] y ``traded`` [sesiones, brazos].
    Todas las series se remuestrean con los mismos índices, en el orden de cada réplica,
    porque el drawdown depende del recorrido. Devuelve los estadísticos, las réplicas
    ([réplicas, costes, brazos] por estadístico) o el motivo de no tenerlas.
    """
    sessions = returns.shape[1]
    estimate = statistics(returns, traded[None], sessions_per_year)
    if block_length >= sessions:
        return estimate, None, "Se necesitan más sesiones que la longitud del bloque"
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in STATISTICS}
    for offset in range(0, replicates, chunk):
        size = min(chunk, replicates - offset)
        index = circular_block_indices(rng, size, sessions, block_length)
        sampled = statistics(returns[:, index, :], traded[index][None], sessions_per_year)
        for name in STATISTICS:
            draws[name].append(np.moveaxis(sampled[name], 0, 1))
    return estimate, {name: np.concatenate(values) for name, values in draws.items()}, None


def contrasts(estimate, draws, arms, family, confidence):
    """Contrastes lineales de una familia sobre un estadístico, con intervalos simultáneos.

    ``estimate`` es un vector por brazo y ``draws`` [réplicas, brazos]. Un contraste con
    algún brazo sin valor en la estimación o en alguna réplica queda sin estimar con su
    motivo y fuera de la familia. La corrección es la de ``compare_series``.
    """
    names = list(family)
    weights = np.zeros((len(names), len(arms)))
    for row, name in enumerate(names):
        for arm, coefficient in family[name].items():
            weights[row, arms.index(arm)] = coefficient
    used = weights != 0
    point = np.array([float(weights[r][used[r]] @ estimate[used[r]]) for r in range(len(names))])
    sampled = None
    if draws is not None:
        sampled = np.stack(
            [draws[:, used[r]] @ weights[r][used[r]] for r in range(len(names))], axis=1
        )
    defined = np.isfinite(point)
    if sampled is not None:
        defined &= np.isfinite(sampled).all(axis=0)
    bounds, critical = None, None
    if sampled is not None and defined.any():
        bounds = family_intervals(point[defined], sampled[:, defined], confidence)
        critical = bounds.pop("critical")
    rows, position = [], 0
    for row, name in enumerate(names):
        entry = dict(
            name=name,
            coefficients={arm: float(c) for arm, c in family[name].items()},
            estimate=float(point[row]) if math.isfinite(point[row]) else None,
            interval=None,
            simultaneous_interval=None,
            bootstrap_standard_error=None,
            simultaneous_excludes_zero=None,
            reason=None,
        )
        if not defined[row]:
            entry["reason"] = (
                "Algún brazo no define el estadístico en la estimación o en una réplica"
            )
        elif bounds is None:
            entry["reason"] = "No hay remuestreo disponible"
        else:
            entry["interval"] = _pair(bounds["lower"][position], bounds["upper"][position])
            joint = _pair(bounds["joint_lower"][position], bounds["joint_upper"][position])
            entry["simultaneous_interval"] = joint
            entry["bootstrap_standard_error"] = float(bounds["spread"][position])
            entry["simultaneous_excludes_zero"] = (
                None if joint is None else bool(joint[0] > 0 or joint[1] < 0)
            )
            position += 1
        rows.append(entry)
    return dict(
        family_size=int(defined.sum()),
        method="max_absolute_studentized_bootstrap",
        critical_value=critical,
        contrasts=rows,
    )


def level_intervals(estimate, draws, confidence):
    """Intervalo percentil marginal de cada brazo, sin corrección de familia."""
    tail = (1 - confidence) / 2
    result = []
    for column, value in enumerate(estimate):
        interval = None
        if draws is not None and math.isfinite(value) and np.isfinite(draws[:, column]).all():
            lower, upper = np.quantile(draws[:, column], [tail, 1 - tail])
            interval = _pair(lower, upper)
        result.append(
            dict(estimate=float(value) if math.isfinite(value) else None, interval=interval)
        )
    return result
