"""Calibración en línea de PT2 (#454) dentro de la comparación walk-forward.

La comparación ajusta en cada ventana la CQR estática con la partición de calibración y la
aplica a toda la evaluación (``cqr_static``, el control). Con la sección
``online_calibration`` declarada, la misma ventana emite además los intervalos de dos
calibraciones que solo usan etiquetas maduras, sobre las mismas predicciones y filas:

- ``cqr_online@κ``: el seguimiento del cuantil de la puntuación CQR por cohortes de mercado
  de ``calibration.online_conformal``, con cada fracción κ declarada. Cada cohorte madura
  antes de emitir en el instante en que su etiqueta está disponible.
- ``cqr_periodic_refit``: la alternativa trivial, que vuelve a ajustar la CQR con las filas
  maduras de los últimos días cada cierto número de sesiones de cada mercado.

La madurez de cada fila es su ``target_available_at``, que el panel de predicciones no lleva.
Se lee de las etiquetas de la vista de la ventana, que ``CorpusDataset`` comprueba con sus
huellas, y se alinea con las filas del panel exigiendo las mismas filas y objetivos. Ninguna
ventana traslada su estado a la siguiente: cada una parte de su CQR estática.

El informe sigue la regla declarada antes de evaluar. κ se elige por modelo con la métrica
primaria en los años de selección, en todos los mercados del ámbito. La métrica primaria es
la desviación absoluta media de la cobertura respecto al nominal por año y mercado,
promediada sobre los intervalos declarados. En los años de evaluación se compara con la CQR
estática con un bootstrap circular por bloques de días y el interval score por sesión con la
comparación emparejada común. Las semillas se promedian sesión a sesión. Es un análisis de
calibración: la mediana no cambia, así que el MAE y los demás contrastes tampoco.

Si la comparación declara los eventos de D2 (``declared_events``), cada modelo da también la
cobertura de todas las variantes en las sesiones de cada evento, con las filas como peso y el
tramo de años al que pertenecen. Los eventos de los años de selección sirven para describir,
no para contrastar, porque κ se eligió con ellos.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.calibration import conformal_quantiles as cqr
from mars_titan.calibration import online_conformal, periodic_conformal
from mars_titan.evaluation import declared_events
from mars_titan.evaluation.forecast_panel import SessionSeries
from mars_titan.evaluation.forecast_scores import score_sessions
from mars_titan.evaluation.paired_comparisons import circular_block_counts, compare_series, delta
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.training.corpus_inputs import CorpusDataset

FIELD = "online_calibration"
STATIC = "cqr_static"
PERIODIC = "cqr_periodic_refit"
PARTITIONS = ("calibration", "evaluation")
MATURITY_SOURCE = "view_labels_target_available_at_v1"
DAY_MICROSECONDS = 86_400_000_000
MAX_SAMPLE_ROWS = 1_000_000
_FIELDS = {
    "status",
    "method",
    "maturity_source",
    "rate_fractions",
    "max_pending",
    "periodic_refit",
    "selection_years",
    "assessment_years",
    "decision",
}
_REFIT_FIELDS = {"method", "every_sessions", "lookback_days"}
_DECISION_FIELDS = {"min_relative_reduction", "max_relative_interval_score_worsening"}
_LAST_YEAR = 2023


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def online_name(fraction):
    """Nombre de la variante en línea con su fracción κ."""
    return f"cqr_online@{fraction:g}"


def variants(section):
    """Variantes que emite la ventana además del control, en el orden del informe."""
    return [*(online_name(value) for value in section["rate_fractions"]), PERIODIC]


def _years(value, label):
    _require(
        isinstance(value, list)
        and len(value) == 2
        and all(type(year) is int for year in value)
        and 2000 <= value[0] <= value[1] <= _LAST_YEAR,
        f"Los {label} deben ser un intervalo de años anteriores a 2024",
    )
    return value


def declaration(section):
    """Validar la sección declarada antes de evaluar. No lee predicciones."""
    _require(
        isinstance(section, dict)
        and set(section) == _FIELDS
        and section["status"] == "declared_not_executed"
        and section["method"] == online_conformal.METHOD
        and section["maturity_source"] == MATURITY_SOURCE,
        "La calibración en línea no cumple su contrato",
    )
    fractions = section["rate_fractions"]
    _require(
        isinstance(fractions, list)
        and fractions
        and all(type(value) is float and 0 < value <= 1 for value in fractions)
        and fractions == sorted(set(fractions)),
        "Las fracciones κ deben ser decimales distintos en (0, 1], en orden creciente",
    )
    pending = section["max_pending"]
    _require(
        type(pending) is int and 1 <= pending <= online_conformal.MAX_PENDING,
        f"El máximo de cohortes pendientes debe estar entre 1 y {online_conformal.MAX_PENDING}",
    )
    refit = section["periodic_refit"]
    _require(
        isinstance(refit, dict)
        and set(refit) == _REFIT_FIELDS
        and refit["method"] == periodic_conformal.METHOD
        and type(refit["every_sessions"]) is int
        and refit["every_sessions"] >= 1
        and type(refit["lookback_days"]) is int
        and refit["lookback_days"] >= 1,
        "La recalibración periódica necesita su método, sus sesiones y sus días",
    )
    selection = _years(section["selection_years"], "años de selección")
    assessment = _years(section["assessment_years"], "años de evaluación")
    _require(selection[1] < assessment[0], "κ se elige en años anteriores a los evaluados")
    decision = section["decision"]
    _require(
        isinstance(decision, dict)
        and set(decision) == _DECISION_FIELDS
        and type(decision["min_relative_reduction"]) is float
        and 0 < decision["min_relative_reduction"] < 1
        and type(decision["max_relative_interval_score_worsening"]) is float
        and 0 <= decision["max_relative_interval_score_worsening"] < 1,
        "La regla de decisión necesita la reducción mínima y el empeoramiento máximo",
    )
    return section


def view_maturity(path, input_policy, expected_sha256, partitions=PARTITIONS):
    """Filas de los tramos de una vista con su objetivo y su madurez.

    Devuelve, por tramo, una tabla con ``asset_id``, ``market``, ``prediction_at`` y
    ``target`` en el formato de las predicciones y ``matures_at`` en microsegundos UTC. Por
    defecto lee la calibración y la evaluación. La criba de #746 pide también la validación.
    """
    dataset = CorpusDataset(Path(path), cache_bytes=0, input_policy=input_policy)
    _require(
        dataset.masked and dataset.temporals,
        "La madurez necesita una vista walk-forward con máscaras",
    )
    _require(dataset.identity == expected_sha256, "La vista cambió antes de leer la madurez")
    columns = {
        name: dict(asset_id=[], market=[], prediction_at=[], target=[], matures_at=[])
        for name in partitions
    }
    for asset in sorted(dataset.assets, key=lambda row: (row["market"], row["symbol"])):
        if not any(asset["counts"][name] for name in partitions):
            continue
        with pq.ParquetFile(dataset._file(asset, "samples")) as file:
            rows = file.metadata.num_rows
        _require(1 <= rows <= MAX_SAMPLE_ROWS, "Las muestras superan el presupuesto de filas")
        for name in partitions:
            if not asset["counts"][name]:
                continue
            _, moments, targets, maturity = dataset._labels(asset, name, rows)
            part = columns[name]
            part["asset_id"] += [f"{asset['market']}/{asset['symbol']}"] * len(moments)
            part["market"] += [asset["market"]] * len(moments)
            part["prediction_at"].append(moments)
            part["target"].append(targets)
            part["matures_at"].append(maturity)
    tables = {}
    for name, part in columns.items():
        _require(part["prediction_at"], f"La vista no tiene filas de {name}")
        moments = pa.array(np.concatenate(part["prediction_at"]), type=pa.int64())
        tables[name] = pa.table(
            dict(
                asset_id=pa.array(part["asset_id"], type=pa.string()),
                market=pa.array(part["market"], type=pa.string()),
                prediction_at=moments.cast(pa.timestamp("us", tz="UTC")),
                target=np.concatenate(part["target"]).astype(np.float64),
                matures_at=np.concatenate(part["matures_at"]).astype(np.int64),
            )
        )
    return tables


def _columns(panel, matures_at):
    return dict(
        target=panel.target,
        quantiles=panel.quantiles,
        groups=np.asarray(panel.markets)[panel.market],
        prediction_at=panel.prediction_at,
        matures_at=matures_at,
    )


def calibrate_window(section, record, fitted, panel, maturity, *, rank_ic_min_assets):
    """Puntuar por sesión las variantes de una ventana con su CQR estática ya fijada.

    `fitted` y `panel` son los paneles de calibración y evaluación del modelo y `maturity`
    la madurez de cada fila en su orden canónico. Devuelve las puntuaciones de cada
    variante y un registro de sus emisiones, actualizaciones y reajustes.
    """
    calibration = _columns(fitted, maturity["calibration"])
    evaluation = _columns(panel, maturity["evaluation"])
    scores, records = {}, {}

    def scored(quantiles):
        quantiles.setflags(write=False)
        return score_sessions(
            dataclasses.replace(panel, quantiles=quantiles), rank_ic_min_assets=rank_ic_min_assets
        )

    static_sha256 = cqr.calibrator_sha256(record)
    for fraction in section["rate_fractions"]:
        calibrator = online_conformal.OnlineConformal.start(
            calibration["target"],
            calibration["quantiles"],
            calibration["groups"],
            levels=record["levels"],
            nominals=record["nominals"],
            min_rows=record["min_rows"],
            rate_fraction=fraction,
            max_pending=section["max_pending"],
        )
        state = calibrator.export()
        # La corrección inicial es la del control, con su misma huella.
        _require(
            state["header"]["static_calibrator_sha256"] == static_sha256,
            "La calibración en línea no parte de la CQR estática de la ventana",
        )
        emitted, emissions, updates = online_conformal.replay_online_conformal(
            calibrator,
            evaluation["quantiles"],
            evaluation["groups"],
            evaluation["prediction_at"],
            evaluation["target"],
            evaluation["matures_at"],
        )
        name = online_name(fraction)
        scores[name] = scored(emitted)
        records[name] = dict(
            rate_fraction=fraction,
            emissions=len(emissions),
            updates=len(updates),
            state_sha256=calibrator.state_sha256(),
        )
    emitted, refits = periodic_conformal.replay_periodic_refit(
        record,
        calibration,
        evaluation,
        every_sessions=section["periodic_refit"]["every_sessions"],
        lookback_days=section["periodic_refit"]["lookback_days"],
    )
    scores[PERIODIC] = scored(emitted)
    records[PERIODIC] = dict(
        refits=len(refits), used=sum(item["used"] for item in refits), log=refits
    )
    return dict(scores=scores, records=records)


def _session_years(scores):
    return scores.session_time.astype("datetime64[us]").astype("datetime64[Y]").astype(int) + 1970


def _coverages(scores, nominals):
    """Cobertura por sesión de cada intervalo declarado, en las columnas de `nominals`."""
    return np.column_stack([scores.coverage[:, scores._interval_index(n)] for n in nominals])


def _seed_coverage(items, nominals):
    """Cobertura media de las semillas sesión a sesión. Las filas son las mismas."""
    first = items[0]
    _require(
        all(
            np.array_equal(item.session_time, first.session_time)
            and np.array_equal(item.session_market, first.session_market)
            and np.array_equal(item.samples, first.samples)
            for item in items
        ),
        "Las semillas no puntúan las mismas sesiones",
    )
    return np.mean([_coverages(item, nominals) for item in items], axis=0)


def _deviation(covered, rows, nominals):
    """Desviación media |cobertura − nominal| sobre celdas con filas y sobre intervalos.

    `covered` tiene forma [..., celdas, intervalos, variantes] y `rows` [..., celdas].
    """
    present = rows > 0
    coverage = covered / np.where(present, rows, 1)[..., None, None]
    error = np.abs(coverage - np.asarray(nominals)[:, None])
    error = np.where(present[..., None, None], error, 0.0)
    cells = present.sum(axis=-1)
    return error.sum(axis=(-3, -2)) / (np.maximum(cells, 1)[..., None] * len(nominals))


def _cells(scores, coverage, sessions):
    """Filas y filas cubiertas por día y mercado de las sesiones elegidas.

    Devuelve los días en orden, su año y matrices [días, mercados] y [días, mercados,
    intervalos, variantes].
    """
    days = scores.session_time[sessions] // DAY_MICROSECONDS
    unique, period = np.unique(days, return_inverse=True)
    markets = len(scores.markets)
    rows = np.zeros((len(unique), markets))
    covered = np.zeros((len(unique), markets, *coverage.shape[1:]))
    weights = scores.samples[sessions].astype(np.float64)
    market = scores.session_market[sessions]
    np.add.at(rows, (period, market), weights)
    np.add.at(covered, (period, market), coverage[sessions] * weights[:, None, None])
    years = (unique * DAY_MICROSECONDS).astype("datetime64[us]").astype("datetime64[Y]")
    return years.astype(int) + 1970, rows, covered


def _by_year(years, rows, covered):
    """Agregar días en celdas año y mercado: [años · mercados] y [años · mercados, ...]."""
    names = np.unique(years)
    cell_rows = np.stack([rows[years == year].sum(axis=0) for year in names])
    cell_covered = np.stack([covered[years == year].sum(axis=0) for year in names])
    return cell_rows.reshape(-1), cell_covered.reshape(-1, *covered.shape[2:])


def coverage_deviation(scores, coverage, sessions, nominals):
    """Métrica primaria de cada variante en las sesiones elegidas, sin remuestrear."""
    rows, covered = _by_year(*_cells(scores, coverage, sessions))
    return _deviation(covered, rows, nominals)


def _relative_draws(scores, coverage, sessions, nominals, comparison):
    """Métrica primaria de cada variante en réplicas del bootstrap circular por días."""
    years, rows, covered = _cells(scores, coverage, sessions)
    rng = np.random.default_rng(comparison["seed"])
    counts = circular_block_counts(
        rng, comparison["replicates"], len(years), comparison["block_length"]
    ).astype(np.float64)
    draws = []
    for year in np.unique(years):
        selected = years == year
        draws.append(
            (
                counts[:, selected] @ rows[selected],
                (counts[:, selected] @ covered[selected].reshape(selected.sum(), -1)).reshape(
                    len(counts), *covered.shape[1:]
                ),
            )
        )
    cell_rows = np.concatenate([item[0] for item in draws], axis=1)
    cell_covered = np.concatenate([item[1] for item in draws], axis=1)
    return _deviation(cell_covered, cell_rows, nominals)


def _ratio(estimate, draws, base, variant, confidence):
    """Cambio relativo de la desviación de `variant` frente a `base`, con intervalo."""
    result = dict(estimate=None, interval=None, reason=None)
    if estimate[base] == 0 or np.any(draws[:, base] == 0):
        result["reason"] = "La desviación de la base es cero en la estimación o en una réplica"
        return result
    result["estimate"] = float(estimate[variant] / estimate[base] - 1)
    tail = (1 - confidence) / 2
    ratios = draws[:, variant] / draws[:, base] - 1
    result["interval"] = [float(v) for v in np.quantile(ratios, [tail, 1 - tail])]
    return result


def _interval_scores(series_by_variant, chosen, comparison, weighting):
    """Interval score por sesión de la variante elegida y la alternativa frente al control."""
    contrasts = {
        f"{chosen}-{STATIC}": delta(STATIC, chosen),
        f"{PERIODIC}-{STATIC}": delta(STATIC, PERIODIC),
        f"{chosen}-{PERIODIC}": delta(PERIODIC, chosen),
    }
    return compare_series(
        {name: series_by_variant[name] for name in (STATIC, chosen, PERIODIC)},
        contrasts,
        block_length=comparison["block_length"],
        replicates=comparison["replicates"],
        seed=comparison["seed"],
        confidence=comparison["confidence"],
        market_weighting=weighting,
    )


def _decision(section, primary, interval_scores, chosen):
    """Aplicar la regla declarada a la variante elegida frente a la CQR estática."""
    rule = section["decision"]
    relative = primary[chosen]
    worsening = []
    for compared in interval_scores.values():
        row = next(r for r in compared["contrasts"] if r["name"] == f"{chosen}-{STATIC}")
        worsening.append(row.get("relative_improvement_percent"))
    if relative["estimate"] is None or None in worsening:
        return "not_estimable"
    limit = -100 * rule["max_relative_interval_score_worsening"]
    if min(worsening) < limit or relative["estimate"] >= 0:
        return "abandon"
    reduction, interval = -relative["estimate"], relative["interval"]
    if reduction >= rule["min_relative_reduction"] and interval[1] < 0:
        return "keep"
    return "inconclusive"


def _matches(primary, interval_scores, chosen):
    """La alternativa trivial iguala a la elegida si ningún intervalo excluye el cero."""
    interval = primary["online_vs_periodic"]["interval"]
    if interval is None:
        return None
    rows = [
        next(r for r in compared["contrasts"] if r["name"] == f"{chosen}-{PERIODIC}")
        for compared in interval_scores.values()
    ]
    if any(row["interval"] is None for row in rows):
        return None
    return bool(
        interval[0] <= 0 <= interval[1]
        and all(row["interval"][0] <= 0 <= row["interval"][1] for row in rows)
    )


def _period(years, section):
    """Tramo de años de un evento: selección, evaluación o fuera de los dos."""
    for name in ("selection", "assessment"):
        first, last = section[f"{name}_years"]
        if all(first <= year <= last for year in years):
            return name
    return "outside_or_mixed"


def _event_coverage(events, section, reference, coverage, names, nominals):
    """Cobertura de cada variante en las sesiones de cada evento, con las filas como peso."""
    years = _session_years(reference)
    result = {}
    for name, event in events["events"].items():
        if event["market"] not in reference.markets:
            continue
        found = declared_events.event_sessions(
            event, reference.session_time, reference.session_market, reference.markets
        )
        if found is None:
            result[name] = dict(market=event["market"], reason="Sin sesiones evaluadas en el tramo")
            continue
        index, counts = found
        weights = reference.samples[index].astype(np.float64)
        covered = np.tensordot(weights, coverage[index], axes=1) / weights.sum()
        spanned = sorted({int(year) for year in years[index]})
        result[name] = dict(
            market=event["market"],
            **counts,
            rows=int(weights.sum()),
            years=spanned,
            period=_period(spanned, section),
            coverage={
                variant: {
                    f"{nominal:g}": float(covered[position, column])
                    for position, nominal in enumerate(nominals)
                }
                for column, variant in enumerate(names)
            },
        )
    return result


def _arm_report(section, comparison, weighting, nominals, per_seed, events=None):
    """Selección de κ, métrica primaria, interval score y regla de un modelo."""
    names = [STATIC, *variants(section)]
    joined = [
        {name: type(items[0][name]).concatenate([item[name] for item in items]) for name in names}
        for items in per_seed
    ]
    reference = joined[0][STATIC]
    coverage = np.stack(
        [_seed_coverage([seed[name] for seed in joined], nominals) for name in names], axis=-1
    )
    years = _session_years(reference)
    first, last = section["selection_years"]
    selection = (years >= first) & (years <= last)
    online = [online_name(value) for value in section["rate_fractions"]]
    result = dict(selection=None, chosen=None, views={})
    if not selection.any():
        result["reason"] = "No hay sesiones en los años de selección"
        return result
    values = coverage_deviation(reference, coverage, selection, nominals)
    candidates = {name: float(values[names.index(name)]) for name in online}
    # Con un empate se queda la tasa más pequeña, la primera de la lista.
    chosen = min(online, key=lambda name: (candidates[name], online.index(name)))
    result.update(selection=candidates, chosen=chosen)
    if events is not None:
        result["events"] = _event_coverage(events, section, reference, coverage, names, nominals)
    first, last = section["assessment_years"]
    assessed = (years >= first) & (years <= last)
    views = {"+".join(reference.markets): np.ones(len(years), dtype=bool)}
    if len(reference.markets) > 1:
        views.update(
            {
                market: reference.session_market == code
                for code, market in enumerate(reference.markets)
            }
        )
    order = [STATIC, chosen, PERIODIC]
    columns = [names.index(name) for name in order]
    for view, mask in views.items():
        sessions = assessed & mask
        if not sessions.any():
            result["views"][view] = dict(reason="No hay sesiones en los años evaluados")
            continue
        estimate = coverage_deviation(reference, coverage[..., columns], sessions, nominals)
        draws = _relative_draws(reference, coverage[..., columns], sessions, nominals, comparison)
        estimate = dict(zip(order, (float(v) for v in estimate), strict=True))
        index = {name: position for position, name in enumerate(order)}
        values = np.array([estimate[name] for name in order])
        primary = {
            chosen: _ratio(values, draws, index[STATIC], index[chosen], comparison["confidence"]),
            PERIODIC: _ratio(
                values, draws, index[STATIC], index[PERIODIC], comparison["confidence"]
            ),
            "online_vs_periodic": _ratio(
                values, draws, index[PERIODIC], index[chosen], comparison["confidence"]
            ),
        }
        interval_scores = {}
        for nominal in nominals:
            metric = f"interval_score@{nominal:g}"
            series = {
                name: SessionSeries.average(
                    [
                        seed[name]
                        .select_sessions(sessions, label=f"{view} {first}-{last}")
                        .series(metric)
                        for seed in joined
                    ]
                )
                for name in order
            }
            interval_scores[f"{nominal:g}"] = _interval_scores(
                series, chosen, comparison, weighting
            )
        result["views"][view] = dict(
            sessions=int(sessions.sum()),
            coverage_deviation=estimate,
            relative_to_static={name: primary[name] for name in (chosen, PERIODIC)},
            online_vs_periodic=primary["online_vs_periodic"],
            interval_score=interval_scores,
            decision=_decision(section, primary, interval_scores, chosen),
            periodic_matches_online=_matches(primary, interval_scores, chosen),
        )
    return result


def report(config, per_window):
    """Informe de PT2 de cada modelo con cuantiles a partir de las puntuaciones por ventana."""
    section, comparison = config[FIELD], config["comparison"]
    nominals = config["calibration"]["nominals"]
    weighting = config["metrics"]["market_weighting"]
    arms = {}
    for name, arm in config["arms"].items():
        if arm["output"] != QUANTILE_HEAD:
            continue
        entries = {
            seed: [results[name, seed] for results in per_window.values()] for seed in arm["seeds"]
        }
        if any(entry.get(FIELD) is None for items in entries.values() for entry in items):
            arms[name] = dict(reason="Alguna ventana no tiene calibrador")
            continue
        per_seed = [
            [dict(entry[FIELD]["scores"], **{STATIC: entry["calibrated"]}) for entry in items]
            for items in entries.values()
        ]
        result = _arm_report(
            section,
            comparison,
            weighting,
            nominals,
            per_seed,
            config.get(declared_events.FIELD),
        )
        totals = {}
        for items in entries.values():
            for entry in items:
                for variant, record in entry[FIELD]["records"].items():
                    total = totals.setdefault(variant, {})
                    for key, value in record.items():
                        if key in ("emissions", "updates", "refits", "used"):
                            total[key] = total.get(key, 0) + value
        arms[name] = dict(result, totals=totals)
    return dict(declaration=section, arms=arms)
