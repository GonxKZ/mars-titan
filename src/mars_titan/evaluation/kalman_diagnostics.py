"""Cobertura de la varianza predictiva y D3 de la regla kalman de B6 (PT3, #455).

Además del MAE, la regla de PT3 usa dos criterios. La cobertura del intervalo de ±1,645
desviaciones predictivas de la corrección debe quedar entre el 85 % y el 95 %. D3 exige
conservar al menos la mitad de la ganancia sobre el núcleo cuando cada etiqueta llega una
sesión más tarde. La ventana B6 kalman emite en cada fila la varianza predictiva kᵀPk + σ² y
la predicción de una memoria igual que recibe cada etiqueta una sesión después
(`training.mars_titan_correction`).

En cada ventana, sobre las mismas filas que puntúa la comparación, se guardan por sesión las
filas cubiertas (|y − ŷ| ≤ z·√v) y las puntuaciones de la predicción retrasada. El informe,
con las sesiones de los años declarados, da por modelo y vista:

- la cobertura como fracción de filas, por semilla y con las semillas juntas, por año y con
  un intervalo por bloques circulares de días;
- D3 con el MAE por sesión promediado entre semillas: la ganancia sin retraso
  g₀ = media(MAE control − MAE modelo), la ganancia con retraso
  g₁ = media(MAE control − MAE retrasado) y la fracción retenida g₁/g₀, con un intervalo por
  bloques circulares de días que remuestrea los mismos días para las dos ganancias.

La regla se aplica a la estimación y el intervalo se informa al lado. No elige modelos ni
cambia los contrastes principales.
"""

import numpy as np

from mars_titan.evaluation.forecast_panel import SessionSeries
from mars_titan.evaluation.forecast_scores import score_sessions
from mars_titan.evaluation.paired_comparisons import circular_block_counts

FIELD = "kalman_diagnostics"
STATUS = "declared_not_executed"
VARIANCE, DELAYED = "predictive_variance", "prediction_label_delay1"
COLUMNS = (VARIANCE, DELAYED)
DAY_MICROSECONDS = 86_400_000_000
_FIELDS = {"status", "declared_at", "arms", "z", "assessment_years", "decision"}
_DECISION_FIELDS = {"coverage", "min_retained_gain"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def declaration(section, arms):
    """Validar la sección antes de evaluar: modelos kalman con su control y la regla."""
    _require(
        isinstance(section, dict) and set(section) == _FIELDS and section["status"] == STATUS,
        "Los diagnósticos de kalman no cumplen su contrato",
    )
    declared = section["arms"]
    _require(
        isinstance(declared, dict)
        and declared
        and all(
            name in arms and control in arms and name != control
            for name, control in declared.items()
        ),
        "Cada modelo kalman necesita un control sin corrección declarado en la comparación",
    )
    _require(type(section["z"]) is float and section["z"] > 0, "z debe ser un decimal positivo")
    years = section["assessment_years"]
    _require(
        isinstance(years, list)
        and len(years) == 2
        and all(type(year) is int for year in years)
        and 2000 <= years[0] <= years[1] <= 2023,
        "Los años evaluados deben ser un intervalo anterior a 2024",
    )
    decision = section["decision"]
    _require(
        isinstance(decision, dict)
        and set(decision) == _DECISION_FIELDS
        and isinstance(decision["coverage"], list)
        and len(decision["coverage"]) == 2
        and all(type(value) is float for value in decision["coverage"])
        and 0 < decision["coverage"][0] < decision["coverage"][1] < 1
        and type(decision["min_retained_gain"]) is float
        and 0 < decision["min_retained_gain"] <= 1,
        "La regla necesita el intervalo de cobertura y la ganancia retenida mínima",
    )
    return section


def score_window(panel, variance, delayed, *, z, rank_ic_min_assets):
    """Filas cubiertas por sesión y puntuaciones de la predicción retrasada de una ventana.

    `variance` y `delayed` son paneles de las mismas filas cuyo punto es la varianza
    predictiva y la predicción retrasada, construidos igual que `panel`.
    """
    for other in (variance, delayed):
        _require(
            other.row_id.equals(panel.row_id) and np.array_equal(other.session, panel.session),
            "Los diagnósticos no tienen las filas del modelo",
        )
    _require(bool(np.all(variance.prediction > 0)), "La varianza predictiva debe ser positiva")
    inside = np.abs(panel.target - panel.prediction) <= z * np.sqrt(variance.prediction)
    inside = inside.astype(np.float64)
    sessions = len(panel.session_market)
    return dict(
        session_market=panel.session_market.copy(),
        session_time=panel.session_time.copy(),
        rows=panel.session_samples.astype(np.float64),
        covered=np.bincount(panel.session, weights=inside, minlength=sessions).astype(np.float64),
        delayed=score_sessions(delayed, rank_ic_min_assets=rank_ic_min_assets),
    )


def _joined(items):
    """Unir las ventanas de una semilla en el orden canónico (mercado, instante)."""
    market = np.concatenate([item["session_market"] for item in items])
    time = np.concatenate([item["session_time"] for item in items])
    order = np.lexsort((time, market))
    joined = {
        key: np.concatenate([item[key] for item in items])[order] for key in ("rows", "covered")
    }
    delayed = type(items[0]["delayed"]).concatenate([item["delayed"] for item in items])
    _require(
        np.array_equal(delayed.session_market, market[order])
        and np.array_equal(delayed.session_time, time[order]),
        "Las sesiones de los diagnósticos no coinciden entre sí",
    )
    return dict(joined, market=market[order], time=time[order], delayed=delayed)


def _years(times):
    return times.astype("datetime64[us]").astype("datetime64[Y]").astype(int) + 1970


def _days(times):
    return np.unique(times // DAY_MICROSECONDS, return_inverse=True)[1].reshape(-1)


def _interval(draws, confidence):
    tail = (1 - confidence) / 2
    return [float(value) for value in np.quantile(draws, [tail, 1 - tail])]


def _counts(comparison, periods):
    rng = np.random.default_rng(comparison["seed"])
    return circular_block_counts(
        rng, comparison["replicates"], periods, comparison["block_length"]
    ).astype(np.float64)


def _coverage(seeds, mask, comparison):
    """Cobertura por semilla, con las semillas juntas, por año y con intervalo por días."""
    rows = np.sum([seed["rows"][mask] for seed in seeds], axis=0)
    covered = np.sum([seed["covered"][mask] for seed in seeds], axis=0)
    days = _days(seeds[0]["time"][mask])
    day_rows = np.bincount(days, weights=rows)
    day_covered = np.bincount(days, weights=covered)
    counts = _counts(comparison, len(day_rows))
    years = _years(seeds[0]["time"][mask])
    return dict(
        rows=int(rows.sum()),
        by_seed=[float(seed["covered"][mask].sum() / seed["rows"][mask].sum()) for seed in seeds],
        estimate=float(covered.sum() / rows.sum()),
        interval=_interval((counts @ day_covered) / (counts @ day_rows), comparison["confidence"]),
        by_year={
            str(year): float(covered[years == year].sum() / rows[years == year].sum())
            for year in np.unique(years)
        },
    )


def _retained(control, model, delayed, times, comparison):
    """Ganancias sin y con retraso y fracción retenida, con los mismos días remuestreados."""
    first, second = control - model, control - delayed
    days = _days(times)
    sessions = np.bincount(days).astype(np.float64)
    sums = [np.bincount(days, weights=values) for values in (first, second)]
    counts = _counts(comparison, len(sessions))
    weight = counts @ sessions
    draws = [(counts @ values) / weight for values in sums]
    gain, delayed_gain = float(first.mean()), float(second.mean())
    result = dict(
        sessions=len(times),
        gain=gain,
        gain_interval=_interval(draws[0], comparison["confidence"]),
        delayed_gain=delayed_gain,
        delayed_gain_interval=_interval(draws[1], comparison["confidence"]),
        retained=None,
        retained_interval=None,
    )
    if gain > 0:
        result["retained"] = delayed_gain / gain
        if np.all(draws[0] > 0):
            result["retained_interval"] = _interval(draws[1] / draws[0], comparison["confidence"])
        else:
            # Con alguna réplica sin ganancia el cociente no está definido en todas.
            result["retained_reason"] = "Alguna réplica no tiene ganancia sin retraso"
    return result


def _decision(section, coverage, gains):
    """Aplicar los dos criterios declarados a las estimaciones de una vista."""
    low, high = section["decision"]["coverage"]
    if gains["gain"] <= 0:
        delay = "no_gain_to_retain"
    elif gains["delayed_gain"] <= 0:
        delay = "gain_disappears"
    elif gains["retained"] >= section["decision"]["min_retained_gain"]:
        delay = "keeps_the_declared_share"
    else:
        delay = "keeps_less_than_the_declared_share"
    return dict(coverage_in_range=low <= coverage["estimate"] <= high, label_delay=delay)


def report(section, comparison, per_window, views, series_of):
    """Sección del informe para cada modelo kalman declarado y cada vista.

    `series_of(arm, view)` da el MAE por sesión de un modelo promediado entre semillas, con
    el instante y el mercado de cada sesión, como en los contrastes.
    """
    first, last = section["assessment_years"]
    arms = {}
    for name, control in section["arms"].items():
        keys = sorted(key for key in next(iter(per_window.values())) if key[0] == name)
        if not keys:
            arms[name] = dict(control=control, reason="El modelo no está en el ámbito")
            continue
        seeds = [_joined([results[key][FIELD] for results in per_window.values()]) for key in keys]
        result = {}
        for view in views:
            model, defined, times, market_codes, markets = series_of(name, view)
            reference, reference_defined, _, _, _ = series_of(control, view)
            in_view = np.isin(seeds[0]["market"], [markets.index(m) for m in view.split("+")])
            delayed = SessionSeries.average(
                [
                    seed["delayed"].select_sessions(in_view, label=view).series("mae")
                    for seed in seeds
                ]
            )
            _require(
                np.array_equal(seeds[0]["time"][in_view], times)
                and np.array_equal(seeds[0]["market"][in_view], market_codes),
                "Los diagnósticos no tienen las sesiones de la comparación",
            )
            years = _years(times)
            mask = (
                (years >= first) & (years <= last) & defined & reference_defined & delayed.defined
            )
            if not mask.any():
                result[view] = dict(reason="No hay sesiones en los años evaluados")
                continue
            coverage = _coverage(
                [
                    {key: seed[key][in_view] for key in ("rows", "covered", "time")}
                    for seed in seeds
                ],
                mask,
                comparison,
            )
            gains = _retained(
                reference[mask],
                model[mask],
                delayed.values[mask],
                times[mask],
                comparison,
            )
            result[view] = dict(
                coverage=coverage,
                label_delay=gains,
                decision=_decision(section, coverage, gains),
            )
        arms[name] = dict(control=control, seeds=[key[1] for key in keys], views=result)
    return dict(
        declaration=section,
        bootstrap=dict(
            kind="circular_blocks_of_utc_days",
            block_length=comparison["block_length"],
            replicates=comparison["replicates"],
            confidence=comparison["confidence"],
            seed=comparison["seed"],
        ),
        arms=arms,
    )
