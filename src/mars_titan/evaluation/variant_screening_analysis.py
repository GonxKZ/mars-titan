"""Análisis de la criba de variantes (#746) sobre las predicciones de sus trabajos.

Cada modelo del catálogo da, en cada origen y semilla, una predicción de la evaluación de ese
origen, que es su bloque de S:

- un ajuste, un lector o una ventana E2 emiten la de su trabajo;
- una corrección elige su caso con el MAE por sesión de su validación, sin mirar S;
- la referencia textual de una corrección emite las predicciones de su padre;
- una etapa de salida (A1, A2, A3 y C4) se ajusta con la calibración de su fuente y se aplica
  sin cambios a S;
- el control cero predice un residuo nulo.

El IC por sesión es la correlación de Pearson entre predicción y residuo con al menos tres
activos. El MAE calibrado multiplica la predicción de un modelo por el λ de cada mercado
elegido en su calibración. Una etapa de salida ya emite su predicción final y se mide sin
otro λ. Las semillas se promedian sesión a sesión y los cuatro orígenes forman una única
serie de 2019 a 2022.

Un contraste resta a la serie del modelo la de su control. Cada familia se remuestrea con
bloques circulares de días UTC, con los mismos índices para todos sus miembros. Recibe el paso
descendente de Romano y Wolf sobre el máximo estudentizado bilateral, Holm sobre los valores p
marginales del mismo bootstrap e intervalos simultáneos de un paso. Una variante pasa si el
paso descendente de la familia declarada rechaza su contraste principal en la dirección
favorable con bloques de 16 días y cada contraste exigido la favorece por sí solo al mismo
nivel. Las otras longitudes de bloque, Holm y las familias de sensibilidad se publican sin
cambiar la decisión.

El módulo no lee predicciones ni importa PyTorch. El motor de la criba le entrega los paneles
ya verificados de cada trabajo y la madurez de las etiquetas.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from statsmodels.stats.multitest import multipletests

from mars_titan.calibration import regime_weighted_conformal
from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.evaluation import neutral_abstention, pit_region_gate
from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation.forecast_panel import DAY_MICROSECONDS
from mars_titan.evaluation.paired_comparisons import circular_block_counts, family_intervals
from mars_titan.training import variant_screening

KIND = "variant_screening_analysis_v1"
PARTITIONS = ("validation", "calibration", "evaluation")
SOURCE = ("references", "quantile_reference")
ZERO = ("common", "zero")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class Origin:
    """Trabajos de un origen con sus paneles verificados y la madurez de sus etiquetas.

    `load(job_id)` devuelve los paneles de validación, calibración y evaluación de un trabajo
    terminado. `cohorts` da la huella de las filas y objetivos de cada tramo en la vista, que
    todos los trabajos deben reproducir, y `maturity` la madurez de cada etiqueta en el orden
    canónico de esos paneles. `sectors` asigna a cada activo el sector de las fichas actuales,
    o None si no lo tiene, y solo lo lee el control sectorial de C4.
    """

    year: int
    load: Callable
    cohorts: dict
    maturity: dict
    sectors: dict | None = None
    stages: dict = field(default_factory=dict)

    def panels(self, job_id):
        panels = self.load(job_id)
        for name in PARTITIONS:
            _require(
                panels[name].cohort_sha256 == self.cohorts[name],
                f"El trabajo {job_id} no predice las filas de {name} de su vista",
            )
        _require(
            int(panels["evaluation"].prediction_at.max()) < FINAL_TEST_START_US,
            "La criba no puede leer el test final",
        )
        return panels

    def history(self, panels):
        """Etiquetas de los tres tramos con su activo, mercado y madurez."""
        return dict(
            asset_id=np.concatenate(
                [pit_region_gate.panel_assets(panels[name]) for name in PARTITIONS]
            ),
            market=np.concatenate(
                [np.asarray(panels[name].markets)[panels[name].market] for name in PARTITIONS]
            ),
            target=np.concatenate([panels[name].target for name in PARTITIONS]),
            matures_at=np.concatenate([self.maturity[name] for name in PARTITIONS]),
        )


@dataclass(frozen=True)
class Emission:
    """Predicción de S de un modelo y la calibración que fija su λ, si la necesita."""

    panel: object
    prediction: np.ndarray
    calibration: object | None


def job_index(jobs):
    """Trabajos de cada modelo por origen y semilla, uno por caso de búsqueda."""
    index = {}
    for job in jobs:
        for label in job["used_by"]:
            key = (tuple(label.split("/", 1)), job["origin"], job["seed"])
            index.setdefault(key, []).append(job)
    for candidates in index.values():
        candidates.sort(key=lambda job: str(job["definition"].get("search_case")))
    return index


def choose_case(origin, candidates):
    """Trabajo con menor MAE medio por sesión en su validación. Empata el primer caso."""
    scores = {}
    for job in candidates:
        validation = origin.panels(job["id"])["validation"]
        scores[job["id"]] = float(np.mean(sd.session_mae(validation, validation.prediction)))
    chosen = min(candidates, key=lambda job: scores[job["id"]])
    return chosen, scores


def _model(origin, index, label, seed, cases):
    candidates = index.get((label, origin.year, seed))
    _require(bool(candidates), f"No hay trabajos de {'/'.join(label)} en {origin.year}")
    chosen, scores = choose_case(origin, candidates)
    if len(candidates) > 1:
        cases.append(
            dict(
                model="/".join(label),
                origin=origin.year,
                seed=seed,
                job=chosen["id"],
                case=chosen["definition"].get("search_case"),
                validation_session_mae=scores,
            )
        )
    return origin.panels(chosen["id"])


def _abstention(protocol, origin, index, seed, cases):
    """Salidas de A2 y de sus controles con la misma cobertura, una vez por semilla."""
    config = protocol["catalog"]["A2"]["arms"]["abstention"]["stage"][neutral_abstention.IDENTITY]
    key = (neutral_abstention.IDENTITY, seed)
    if key not in origin.stages:
        panels = _model(origin, index, SOURCE, seed, cases)
        evaluation = panels["evaluation"]
        record = neutral_abstention.fit(
            panels["calibration"], origin.maturity["calibration"], config
        )
        volatility = neutral_abstention.past_volatility(
            pit_region_gate.panel_assets(evaluation),
            evaluation.prediction_at,
            origin.history(panels),
            window=config["volatility_window"],
            min_labels=config["volatility_min_labels"],
        )
        seeds = [
            _model(origin, index, SOURCE, other, cases)["evaluation"] for other in protocol["seeds"]
        ]
        outputs, summary = neutral_abstention.emit(
            record,
            evaluation,
            stream=seed,
            volatility=volatility,
            disagreement=neutral_abstention.seed_disagreement(seeds),
        )
        origin.stages[key] = outputs, summary
    return origin.stages[key]


def _stage(protocol, origin, index, seed, entry, value, cases):
    """Predicción de S de una etapa de salida ajustada con la calibración de su fuente."""
    panels = _model(origin, index, SOURCE, seed, cases)
    calibration, evaluation = panels["calibration"], panels["evaluation"]
    sd.ensure_before(calibration, evaluation)
    stage = value["stage"]
    if stage == "none":
        return evaluation.prediction
    if stage == "market_lambda_of_the_calibration":
        factors = sd.market_shrinkage(calibration, calibration.prediction)
        return sd.apply_shrinkage(evaluation, evaluation.prediction, factors)
    if stage == "matched_control":
        return _abstention(protocol, origin, index, seed, cases)[0][value["control"]]
    ((identity, config),) = stage.items()
    if identity == neutral_abstention.IDENTITY:
        return _abstention(protocol, origin, index, seed, cases)[0][identity]
    history = origin.history(panels)
    if identity == regime_weighted_conformal.IDENTITY:
        _require(value.get("then") == "A2.abstention", "A3 se mide a través de la abstención")
        record = regime_weighted_conformal.fit(
            calibration, origin.maturity["calibration"], history, config
        )
        calibrated, _ = regime_weighted_conformal.replay(
            record,
            (calibration, origin.maturity["calibration"]),
            (evaluation, origin.maturity["evaluation"]),
            history,
            weighting=value["weighting"],
        )
        nominal = protocol["catalog"]["A2"]["arms"]["abstention"]["stage"][
            neutral_abstention.IDENTITY
        ]["nominal"]
        return neutral_abstention.emit_calibrated(evaluation, calibrated, nominal)[0]
    _require(identity == pit_region_gate.IDENTITY, f"{entry} declara una etapa desconocida")
    sectors = origin.sectors if config["regions"] == "current_sector" else None
    _require(
        config["regions"] != "current_sector" or sectors is not None,
        "El control sectorial de C4 necesita los sectores de las fichas actuales",
    )
    record = pit_region_gate.fit(
        calibration,
        origin.maturity["calibration"],
        pit_region_gate.panel_assets(calibration),
        history,
        config,
        sectors=sectors,
    )
    return pit_region_gate.apply(record, evaluation, pit_region_gate.panel_assets(evaluation))


def emission(protocol, arms, index, origin, label, seed, cases):
    """Predicción de S de un modelo en un origen y una semilla."""
    if label == ZERO:
        evaluation = _model(origin, index, SOURCE, seed, cases)["evaluation"]
        return Emission(evaluation, np.zeros(evaluation.rows), None)
    _require(label[0] != "common", f"El control común {label[1]} no está implementado")
    arm = arms[label]
    runner = arm.runner
    if runner == "output":
        _require("combine" not in arm.definition, "La media de semillas de F2 está pendiente")
        _require(arm.definition["source"] == SOURCE[1], "Las etapas parten de la referencia B")
        prediction = _stage(protocol, origin, index, seed, label[0], arm.definition, cases)
        evaluation = _model(origin, index, SOURCE, seed, cases)["evaluation"]
        return Emission(evaluation, np.asarray(prediction, dtype=np.float64), None)
    if runner == "reference":
        label = ("references", arm.definition["reference"])
    panels = _model(origin, index, label, seed, cases)
    return Emission(panels["evaluation"], panels["evaluation"].prediction, panels["calibration"])


def session_metric(emission, metric, *, min_assets):
    """Valor por sesión y máscara de sesiones definidas de una métrica del protocolo."""
    panel = emission.panel
    if metric == "session_ic":
        return sd.session_correlations(panel, emission.prediction, min_assets=min_assets)
    if metric == "session_rank_ic":
        return sd.session_correlations(panel, emission.prediction, rank=True, min_assets=min_assets)
    prediction = emission.prediction
    if metric == "session_mae_calibrated" and emission.calibration is not None:
        sd.ensure_before(emission.calibration, panel)
        factors = sd.market_shrinkage(emission.calibration, emission.calibration.prediction)
        prediction = sd.apply_shrinkage(panel, prediction, factors)
    else:
        _require(metric in ("session_mae_calibrated", "session_mae_raw"), f"Métrica {metric}")
    return sd.session_mae(panel, prediction), np.ones(panel.sessions, dtype=bool)


def _average(items):
    """Media de las semillas en cada sesión. Una sesión cuenta si todas la definen."""
    defined = np.logical_and.reduce([defined for _, defined in items])
    values = np.mean([values for values, _ in items], axis=0)
    return np.where(defined, values, 0.0), defined


def _bootstrap(values, day, *, block_length, replicates, seed):
    """Réplicas de la media de cada columna con bloques circulares de días UTC."""
    _, period = np.unique(day, return_inverse=True)
    periods = int(period.max()) + 1
    _require(1 <= block_length < periods, "Se necesitan más días que la longitud del bloque")
    counts = circular_block_counts(np.random.default_rng(seed), replicates, periods, block_length)
    weights = counts[:, period].astype(np.float64)
    return (weights @ values) / weights.sum(axis=1, keepdims=True)


def stepdown(estimate, draws):
    """Valores p marginales y del paso descendente de Romano y Wolf, bilaterales.

    t_k = |θ̂_k| / σ_k y t*_rk = |θ*_rk − θ̂_k| / σ_k, con σ_k la desviación de las réplicas.
    Con las hipótesis ordenadas de mayor a menor t, el valor ajustado de la j-ésima es la
    proporción de réplicas cuyo máximo de t* sobre las hipótesis j, j + 1, … alcanza t_(j),
    con la monotonía impuesta después (Romano y Wolf, 2016). Los recuentos usan (1 + n)/(1 + R),
    que nunca da cero. Un miembro sin variación entre réplicas vale 1.
    """
    replicates = draws.shape[0]
    spread = np.std(draws, axis=0, ddof=1)
    scale = np.maximum(np.abs(estimate), np.max(np.abs(draws), axis=0))
    active = spread > 64 * np.finfo(np.float64).eps * scale
    statistic = np.zeros(len(estimate))
    centred = np.zeros(draws.shape)
    statistic[active] = np.abs(estimate[active]) / spread[active]
    centred[:, active] = np.abs(draws[:, active] - estimate[active]) / spread[active]
    marginal = (1 + np.count_nonzero(centred >= statistic, axis=0)) / (1 + replicates)
    order = np.argsort(-statistic, kind="stable")
    tail = np.maximum.accumulate(centred[:, order[::-1]], axis=1)[:, ::-1]
    ordered = (1 + np.count_nonzero(tail >= statistic[order], axis=0)) / (1 + replicates)
    adjusted = np.empty(len(estimate))
    adjusted[order] = np.maximum.accumulate(ordered)
    marginal[~active] = adjusted[~active] = 1.0
    return dict(statistic=statistic, spread=spread, marginal=marginal, stepdown=adjusted)


def _favourable(estimate, direction):
    return bool(estimate > 0) if direction == "greater" else bool(estimate < 0)


def evaluate_family(members, day, *, block_length, inference):
    """Estadísticos de una familia de contrastes sobre las sesiones que todos definen."""
    defined = np.logical_and.reduce([member["defined"] for member in members])
    _require(defined.any(), "La familia no tiene sesiones comunes definidas")
    values = np.column_stack([member["values"][defined] for member in members])
    draws = _bootstrap(
        values,
        day[defined],
        block_length=block_length,
        replicates=inference["replicates"],
        seed=inference["seed"],
    )
    estimate = values.mean(axis=0)
    tests = stepdown(estimate, draws)
    bounds = family_intervals(estimate, draws, inference["confidence"])
    holm = multipletests(tests["marginal"], method="holm")[1]
    alpha = 1 - inference["confidence"]
    rows = []
    for k, member in enumerate(members):
        favourable = _favourable(estimate[k], member["direction"])
        rows.append(
            dict(
                name=member["name"],
                estimate=float(estimate[k]),
                interval=[float(bounds["lower"][k]), float(bounds["upper"][k])],
                simultaneous=[float(bounds["joint_lower"][k]), float(bounds["joint_upper"][k])],
                spread=float(tests["spread"][k]),
                marginal_pvalue=float(tests["marginal"][k]),
                stepdown_pvalue=float(tests["stepdown"][k]),
                holm_pvalue=float(holm[k]),
                favourable=favourable,
                passes=favourable and bool(tests["stepdown"][k] <= alpha),
                passes_holm=favourable and bool(holm[k] <= alpha),
                passes_marginal=favourable and bool(tests["marginal"][k] <= alpha),
            )
        )
    return dict(
        block_days=block_length,
        sessions=int(np.count_nonzero(defined)),
        days=len(np.unique(day[defined])),
        critical=bounds["critical"],
        members=rows,
    )


def _name(row):
    return f"{row['entry']}/{row['role']}/{'.'.join(row['variant'])}-{'.'.join(row['control'])}"


def _directions(protocol):
    contrast = protocol["contrast"]
    return {
        contrast["primary"]["metric"]: contrast["primary"]["direction"],
        contrast["amplitude"]["metric"]: contrast["amplitude"]["direction"],
    }


def analyze(protocol, jobs, origins, root=variant_screening.ROOT):
    """Series por sesión, contrastes, familias y decisión de la criba sellada.

    `jobs` es el catálogo sellado por el motor y `origins` asigna a cada año su `Origin`.
    Todos los trabajos deben haber terminado.
    """
    _require(
        sorted(origins) == [origin["year"] for origin in protocol["origins"]],
        "El análisis necesita los cuatro orígenes de la criba",
    )
    arms = variant_screening.all_arms(protocol, root)
    rows = variant_screening.contrasts(protocol, root)
    index = job_index(jobs)
    directions = _directions(protocol)
    min_assets = protocol["contrast"]["primary"]["minimum_assets"]
    # Cada modelo de un contraste se mide con la métrica principal, las secundarias y la de
    # su contraste. El MAE menos el del cero se deriva después de las series.
    secondary = {protocol["contrast"]["primary"]["metric"], *protocol["contrast"]["secondary"]}
    secondary.discard("session_mae_minus_zero")
    needed = {}
    for row in rows:
        for side in ("variant", "control"):
            needed.setdefault(row[side], set(secondary)).add(row["metric"])
    cases, series, times = [], {}, []
    for year in sorted(origins):
        origin = origins[year]
        per_origin = {}
        for label, metrics in needed.items():
            for seed in protocol["seeds"]:
                output = emission(protocol, arms, index, origin, label, seed, cases)
                for metric in metrics:
                    per_origin.setdefault((label, metric), []).append(
                        session_metric(output, metric, min_assets=min_assets)
                    )
        times.append(np.asarray(output.panel.session_time))
        for key, items in per_origin.items():
            series.setdefault(key, []).append(_average(items))
        origin.stages.clear()
    session_time = np.concatenate(times)
    _require(np.all(np.diff(session_time) > 0), "Las sesiones de S no siguen el orden temporal")
    day = session_time // DAY_MICROSECONDS
    merged = {
        key: (
            np.concatenate([values for values, _ in parts]),
            np.concatenate([defined for _, defined in parts]),
        )
        for key, parts in series.items()
    }
    contrasts = []
    for row in rows:
        variant = merged[(row["variant"], row["metric"])]
        control = merged[(row["control"], row["metric"])]
        defined = variant[1] & control[1]
        contrasts.append(
            dict(
                row,
                name=_name(row),
                direction=directions[row["metric"]],
                values=np.where(defined, variant[0] - control[0], 0.0),
                defined=defined,
            )
        )
    report = families(protocol, contrasts, day)
    report.update(
        kind=KIND,
        sessions=len(session_time),
        first_session=int(session_time[0]),
        last_session=int(session_time[-1]),
        cases=cases,
        models={"/".join(label): _summary(merged, label) for label in needed},
    )
    return report, dict(session_time=session_time, merged=merged, contrasts=contrasts)


def _mean(values, defined):
    return float(values[defined].mean()) if defined.any() else None


def _summary(merged, label):
    """Media de cada métrica en S, también la del MAE calibrado menos el del control cero."""
    result = {
        metric: dict(mean=_mean(values, defined), defined_sessions=int(defined.sum()))
        for (other, metric), (values, defined) in merged.items()
        if other == label
    }
    mae, zero = merged[(label, "session_mae_calibrated")], merged[(ZERO, "session_mae_calibrated")]
    result["session_mae_minus_zero"] = dict(
        mean=_mean(mae[0] - zero[0], mae[1]), defined_sessions=int(mae[1].sum())
    )
    return result


def families(protocol, contrasts, day):
    """Familia declarada, sensibilidades y decisión de cada entrada del catálogo."""
    inference = protocol["inference"]
    blocks = [inference["block_days"], *inference["sensitivity_block_days"]]
    primary = [row for row in contrasts if row["role"] == "primary"]
    declared = {
        length: evaluate_family(primary, day, block_length=length, inference=inference)
        for length in blocks
    }
    main = inference["block_days"]
    groups = {}
    for row in primary:
        groups.setdefault(row["entry"][0], []).append(row)
    sensitivity = dict(
        groups={
            group: evaluate_family(members, day, block_length=main, inference=inference)
            for group, members in groups.items()
        },
        all_contrasts=evaluate_family(contrasts, day, block_length=main, inference=inference),
    )
    required = {
        row["name"]: evaluate_family([row], day, block_length=main, inference=inference)["members"][
            0
        ]
        for row in contrasts
        if row["role"] == "required"
    }
    effects = {}
    for row in primary:
        effects[row["entry"]] = sd.minimum_detectable_effect(
            row["values"][row["defined"]],
            candidates=(len(primary),),
            alpha=inference["minimum_detectable_effect"]["alpha"],
            power=inference["minimum_detectable_effect"]["power"],
        )
    everything = {m["name"]: m for m in sensitivity["all_contrasts"]["members"]}
    decisions = {}
    for k, row in enumerate(primary):
        member = declared[main]["members"][k]
        group = {m["name"]: m for m in sensitivity["groups"][row["entry"][0]]["members"]}
        own = [
            r["name"] for r in contrasts if r["role"] == "required" and r["entry"] == row["entry"]
        ]
        decisions[row["entry"]] = dict(
            contrast=row["name"],
            metric=row["metric"],
            passes=member["passes"] and all(required[name]["passes_marginal"] for name in own),
            primary_passes=member["passes"],
            required={name: required[name]["passes_marginal"] for name in own},
            robustness=dict(
                {f"blocks_{length}": declared[length]["members"][k]["passes"] for length in blocks},
                holm=member["passes_holm"],
                group_family=group[row["name"]]["passes"],
                all_contrasts_family=everything[row["name"]]["passes"],
            ),
        )
    return dict(
        declared_family={str(length): family for length, family in declared.items()},
        sensitivity_families=sensitivity,
        required_contrasts=required,
        minimum_detectable_effect=effects,
        decisions=decisions,
        decision_rule=protocol["decision"],
        contrasts=[
            {key: row[key] for key in ("name", "entry", "role", "metric", "direction")}
            | dict(
                variant="/".join(row["variant"]),
                control="/".join(row["control"]),
                defined_sessions=int(np.count_nonzero(row["defined"])),
                estimate=_mean(row["values"], row["defined"]),
            )
            for row in contrasts
        ],
    )
