"""Contrastes emparejados por sesión con bootstrap circular por bloques de días UTC.

Cada réplica remuestrea bloques de días consecutivos y conserva juntas todas las
sesiones de un día, con todos sus activos, para todos los modelos a la vez. Los
contrastes son combinaciones lineales de medias por sesión calculadas sobre las
sesiones definidas para todos los modelos que intervienen. La familia de una
llamada recibe además intervalos simultáneos por máximo estudentizado.
"""

import math
import numbers

import numpy as np

from mars_titan.evaluation.forecast_panel import WEIGHTINGS, SessionSeries, weighted_session_mean

MAX_REPLICATES = 100_000
MAX_CONTRASTS = 64
_CHUNK = 256
_INSUFFICIENT = "Se necesitan más días que la longitud del bloque"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def delta(base, variant):
    """Delta = métrica_variante − métrica_base. Con pérdidas, negativo favorece a la variante."""
    return {variant: 1.0, base: -1.0}


def interaction(base, first, second, joint):
    """I = joint − first − second + base, por ejemplo MAE_CM − MAE_C − MAE_M + MAE_B."""
    return {joint: 1.0, first: -1.0, second: -1.0, base: 1.0}


def level(model):
    """Media de un único modelo, por ejemplo el Rank IC medio con su intervalo."""
    return {model: 1.0}


def circular_block_indices(rng, replicates, periods, block_length):
    """Días de cada réplica del bootstrap circular por bloques, en el orden remuestreado.

    Cada réplica concatena ceil(P / L) bloques de L días consecutivos con inicio
    uniforme y recorrido circular, y trunca a P días. Todos los días tienen la
    misma probabilidad de aparecer, también los de los extremos. El orden sirve
    a los estadísticos que dependen del recorrido, como el drawdown máximo.
    """
    blocks = -(-periods // block_length)
    starts = rng.integers(0, periods, size=(replicates, blocks))
    index = (starts[:, :, None] + np.arange(block_length)) % periods
    return index.reshape(replicates, -1)[:, :periods]


def circular_block_counts(rng, replicates, periods, block_length):
    """Veces que aparece cada día en cada réplica, con los mismos sorteos que los índices."""
    index = circular_block_indices(rng, replicates, periods, block_length)
    index = index + periods * np.arange(replicates)[:, None]
    counts = np.bincount(index.ravel(), minlength=replicates * periods)
    return counts.reshape(replicates, periods)


def _pairwise_base(coefficients):
    if len(coefficients) != 2:
        return None
    (first, a), (second, b) = coefficients.items()
    if (a, b) == (1.0, -1.0):
        return second
    return first if (a, b) == (-1.0, 1.0) else None


def _validate(series, contrasts):
    _require(isinstance(series, dict) and series, "Se necesita al menos una serie por modelo")
    _require(
        all(isinstance(item, SessionSeries) for item in series.values()),
        "Las series deben ser SessionSeries",
    )
    first = next(iter(series.values()))
    _require(
        all(
            item.metric == first.metric
            and item.loss == first.loss
            and item.markets == first.markets
            and item.cohort_sha256 == first.cohort_sha256
            for item in series.values()
        ),
        "Las series no comparten métrica, mercados y población",
    )
    _require(
        isinstance(contrasts, dict) and 1 <= len(contrasts) <= MAX_CONTRASTS,
        "La familia de contrastes debe declararse y estar acotada",
    )
    for name, coefficients in contrasts.items():
        _require(isinstance(name, str) and name, "Cada contraste necesita un nombre")
        _require(
            isinstance(coefficients, dict)
            and coefficients
            and all(model in series for model in coefficients)
            and all(
                isinstance(value, numbers.Real)
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value != 0
                for value in coefficients.values()
            ),
            f"El contraste {name} no usa modelos y coeficientes válidos",
        )
    return first


def _columns(series, contrasts, loss):
    values, defined, bases = [], [], {}
    for coefficients in contrasts.values():
        mask = np.logical_and.reduce([series[model].defined for model in coefficients])
        combined = sum(float(c) * series[model].values for model, c in coefficients.items())
        values.append(np.where(mask, combined, 0.0))
        defined.append(mask)
    for position, (name, coefficients) in enumerate(contrasts.items()):
        base = _pairwise_base({m: float(c) for m, c in coefficients.items()})
        if loss and base is not None:
            bases[name] = len(values)
            values.append(np.where(defined[position], series[base].values, 0.0))
            defined.append(defined[position])
    return np.column_stack(values), np.column_stack(defined), bases


def _cells(reference, values, defined):
    """Sumas y recuentos por día, mercado y columna: forma [días, mercados, columnas]."""
    markets = len(reference.markets)
    periods = int(reference.session_period.max()) + 1
    cell = reference.session_period * markets + reference.session_market
    columns = values.shape[1]
    sums = np.empty((columns, periods * markets))
    sizes = np.empty_like(sums)
    for j in range(columns):
        sums[j] = np.bincount(cell, weights=values[:, j], minlength=periods * markets)
        sizes[j] = np.bincount(cell, weights=defined[:, j], minlength=periods * markets)
    shape = (columns, periods, markets)
    return sums.reshape(shape).transpose(1, 2, 0), sizes.reshape(shape).transpose(1, 2, 0)


def _replicates(counts, sums, sizes, weighting):
    """Medias de cada réplica, con NaN interno si la réplica no contiene un mercado necesario."""
    periods, markets, columns = sums.shape
    draws = counts.astype(np.float64)
    numerator = (draws @ sums.reshape(periods, -1)).reshape(-1, markets, columns)
    denominator = (draws @ sizes.reshape(periods, -1)).reshape(-1, markets, columns)
    if weighting == "session":
        total = denominator.sum(axis=1)
        return np.where(total > 0, numerator.sum(axis=1) / np.maximum(total, 1), np.nan)
    present = sizes.sum(axis=0) > 0
    means = np.where(denominator > 0, numerator / np.maximum(denominator, 1), np.nan)
    means = np.where(present[None], means, 0.0)
    return means.sum(axis=1) / np.maximum(present.sum(axis=0), 1)


def _bootstrap(reference, values, defined, weighting, block_length, replicates, seed):
    sums, sizes = _cells(reference, values, defined)
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, values.shape[1]))
    for offset in range(0, replicates, _CHUNK):
        size = min(_CHUNK, replicates - offset)
        counts = circular_block_counts(rng, size, sums.shape[0], block_length)
        draws[offset : offset + size] = _replicates(counts, sums, sizes, weighting)
    return draws


def family_intervals(estimate, draws, confidence):
    """Percentiles marginales e intervalos simultáneos por máximo estudentizado.

    ``estimate`` tiene un valor por contraste y ``draws`` una fila por réplica. Es la
    corrección de la familia que usan todas las comparaciones emparejadas.
    """
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(draws, [tail, 1 - tail], axis=0, method="linear")
    spread = np.std(draws, axis=0, ddof=1)
    scale = np.maximum(np.abs(estimate), np.max(np.abs(draws), axis=0))
    active = spread > 64 * np.finfo(np.float64).eps * scale
    critical = None
    joint_lower, joint_upper = lower.copy(), upper.copy()
    if active.any():
        studentized = np.abs(draws[:, active] - estimate[active]) / spread[active]
        critical = float(np.quantile(studentized.max(axis=1), confidence, method="higher"))
        joint_lower[active] = estimate[active] - critical * spread[active]
        joint_upper[active] = estimate[active] + critical * spread[active]
    return dict(
        lower=lower,
        upper=upper,
        joint_lower=joint_lower,
        joint_upper=joint_upper,
        spread=spread,
        critical=critical,
    )


def _family(reference, values, defined, estimate, options, block_length, size):
    """Intervalos de los ``size`` contrastes para una longitud de bloque, o el motivo de no darlos.

    Las columnas posteriores son bases auxiliares de la mejora relativa. Se
    remuestrean con los mismos índices, pero no forman parte de la familia.
    """
    periods = int(reference.session_period.max()) + 1
    if block_length >= periods:
        return None, _INSUFFICIENT
    known = ~np.isnan(estimate)
    if not known[:size].any():
        return None, "Ningún contraste tiene sesiones definidas"
    draws = np.full((options["replicates"], len(estimate)), np.nan)
    draws[:, known] = _bootstrap(
        reference,
        values[:, known],
        defined[:, known],
        options["market_weighting"],
        block_length,
        options["replicates"],
        options["seed"],
    )
    if np.isnan(draws[:, known]).any():
        return None, "Alguna réplica no contiene sesiones definidas de un mercado necesario"
    claims = known[:size]
    bounds = family_intervals(
        estimate[:size][claims], draws[:, :size][:, claims], options["confidence"]
    )
    family = dict(critical=bounds.pop("critical"), draws=draws)
    for key, value in bounds.items():
        family[key] = np.full(size, np.nan)
        family[key][claims] = value
    return family, None


def _pair(lower, upper, index):
    if np.isnan(lower[index]):
        return None
    return [float(lower[index]), float(upper[index])]


def compare_series(
    series,
    contrasts,
    *,
    block_length,
    replicates=2000,
    seed,
    confidence=0.95,
    market_weighting="session",
    sensitivity_block_lengths=(),
):
    """Estimar una familia declarada de contrastes emparejados y sus intervalos.

    ``series`` asigna a cada modelo su SessionSeries sobre la misma población.
    ``contrasts`` asigna a cada nombre sus coeficientes por modelo. La longitud
    de bloque, las réplicas, la semilla y la familia deben fijarse antes de ver
    los resultados. Un intervalo simultáneo que excluye el cero es la regla para
    afirmar un efecto dentro de la familia. Un intervalo que incluye el cero no
    demuestra equivalencia.
    """
    reference = _validate(series, contrasts)
    _require(market_weighting in WEIGHTINGS, "Ponderación entre mercados no declarada")
    for name, value, lower, upper in (
        ("bloque", block_length, 1, 100_000),
        ("réplicas", replicates, 10, MAX_REPLICATES),
        ("semilla", seed, 0, 2**63 - 1),
    ):
        _require(type(value) is int and lower <= value <= upper, f"Parámetro {name} inválido")
    _require(
        isinstance(confidence, float) and 0.5 <= confidence < 1,
        "El nivel de confianza no es válido",
    )
    sensitivity = tuple(sensitivity_block_lengths)
    _require(
        len(sensitivity) <= 16
        and all(type(value) is int and 1 <= value <= 100_000 for value in sensitivity),
        "Las longitudes de sensibilidad deben ser enteros positivos",
    )
    values, defined, bases = _columns(series, contrasts, reference.loss)
    markets = len(reference.markets)

    def estimate(j):
        return weighted_session_mean(
            values[:, j], defined[:, j], reference.session_market, markets, market_weighting
        )

    estimates = np.full(values.shape[1], np.nan)
    for j in range(values.shape[1]):
        value = estimate(j)
        if value is not None:
            estimates[j] = value
    options = dict(
        market_weighting=market_weighting,
        replicates=replicates,
        seed=seed,
        confidence=confidence,
    )
    size = len(contrasts)
    family, reason = _family(reference, values, defined, estimates, options, block_length, size)
    rows = []
    for j, (name, coefficients) in enumerate(contrasts.items()):
        row = dict(
            name=name,
            coefficients={model: float(value) for model, value in coefficients.items()},
            estimate=None if np.isnan(estimates[j]) else float(estimates[j]),
            sessions=int(np.sum(defined[:, j])),
            excluded_sessions=int(np.sum(~defined[:, j])),
            interval=None if family is None else _pair(family["lower"], family["upper"], j),
            simultaneous_interval=(
                None if family is None else _pair(family["joint_lower"], family["joint_upper"], j)
            ),
            bootstrap_standard_error=(
                None
                if family is None or np.isnan(family["spread"][j])
                else float(family["spread"][j])
            ),
        )
        joint = row["simultaneous_interval"]
        row["simultaneous_excludes_zero"] = (
            None if joint is None else bool(joint[0] > 0 or joint[1] < 0)
        )
        if name in bases and row["estimate"] is not None:
            row.update(
                _relative(estimates[j], estimates[bases[name]], family, j, bases[name], confidence)
            )
        rows.append(row)
    result = dict(
        schema_version=1,
        kind="paired_session_contrasts",
        metric=reference.metric,
        loss=reference.loss,
        cohort_sha256=reference.cohort_sha256,
        market_weighting=market_weighting,
        confidence=confidence,
        resampling=dict(
            method="circular_block_bootstrap",
            unit="utc_calendar_day_with_all_sessions_and_assets",
            periods=int(reference.session_period.max()) + 1,
            sessions=len(reference.session_period),
            block_length=block_length,
            replicates=replicates,
            seed=seed,
            bit_generator="PCG64",
            numpy_version=np.__version__,
            reason=reason,
        ),
        multiplicity=dict(
            method="max_absolute_studentized_bootstrap",
            family_size=len(contrasts),
            critical_value=None if family is None else family["critical"],
        ),
        contrasts=rows,
        sensitivity=[],
    )
    for length in sensitivity:
        local, local_reason = _family(reference, values, defined, estimates, options, length, size)
        result["sensitivity"].append(
            dict(
                block_length=length,
                reason=local_reason,
                contrasts=[
                    dict(
                        name=name,
                        interval=None
                        if local is None
                        else _pair(local["lower"], local["upper"], j),
                        simultaneous_interval=(
                            None
                            if local is None
                            else _pair(local["joint_lower"], local["joint_upper"], j)
                        ),
                    )
                    for j, name in enumerate(contrasts)
                ],
            )
        )
    return result


def _relative(difference, base, family, column, base_column, confidence):
    """Mejora relativa = 100 · (base − variante) / base, no definida si la base es cero."""
    result = dict(
        base_estimate=float(base),
        variant_estimate=float(base + difference),
        relative_improvement_percent=None,
        relative_improvement_interval=None,
        relative_improvement_reason=None,
    )
    if base == 0:
        result["relative_improvement_reason"] = "La métrica de la base es cero"
        return result
    result["relative_improvement_percent"] = float(-100 * difference / base)
    if family is None:
        result["relative_improvement_reason"] = "No hay remuestreo disponible"
        return result
    bases = family["draws"][:, base_column]
    if np.any(bases == 0):
        result["relative_improvement_reason"] = "Alguna réplica tiene una base igual a cero"
        return result
    ratios = -100 * family["draws"][:, column] / bases
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(ratios, [tail, 1 - tail], method="linear")
    result["relative_improvement_interval"] = [float(lower), float(upper)]
    return result
