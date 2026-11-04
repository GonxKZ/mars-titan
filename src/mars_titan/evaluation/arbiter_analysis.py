"""Etapa del análisis de los árbitros en línea sobre las predicciones de la criba (#789).

Las dos etapas de `online_arbiters` leen predicciones que la criba de #746 ya emite y no
ajustan ningún parámetro.

- `version_arbiter_v1` mezcla, en cada modelo de `selection.versions`, la época común con las
  versiones de las épocas declaradas del mismo ajuste. Una época común que también es versión
  cuenta una sola vez, porque el motor no la emite como versión. Sus controles son la emisión
  de la época común, la media uniforme de las opciones y el mismo árbitro con los resultados
  desordenados en el tiempo.
- `model_combination_v1` mezcla, en cada origen y semilla, los modelos comparados que emiten
  sus propias predicciones de calibración y de S. Quedan fuera el control cero, las etapas de
  salida, que se ajustan con la calibración, la media de semillas de F2, que ya combina
  semillas, y la referencia textual, que repite la emisión de su padre. Dos modelos que
  comparten trabajo cuentan una vez. Su control es la media simple de los modelos tipificados.

Cada opción se lee en la calibración y en S de su origen y semilla. Las dos particiones
forman un único panel, de modo que el estado de AdaNormalHedge.TV recorre la calibración y
después S, y cada sesión solo usa las etiquetas maduras al decidirla. El λ de cada mercado se
elige con la salida de la etapa en calibración y se aplica sin cambios en S. Una sesión
abstenida que puntúa cuenta como IC 0, así que el árbitro no se mide solo en las sesiones que
elige.

Las semillas se promedian sesión a sesión y los cuatro orígenes forman una única serie, como en
el análisis de la criba. El árbitro de versiones tiene su propia familia, con el contraste de
cada modelo frente a su época común, y dos contrastes exigidos por modelo. Se decide con la
regla de la criba, pero esa familia no se mezcla con la declarada de la primera ola. F2 y F3
ya promedian pesos en el tiempo, así que el árbitro sobre las versiones de la referencia se
compara con sus variantes en otra familia que se informa aparte. La combinación de modelos
también se informa aparte y no se atribuye a MARS-TITAN.
"""

from dataclasses import replace

import numpy as np
import pyarrow as pa

from mars_titan.evaluation import online_arbiters as arbiters
from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation import variant_screening_analysis as analysis
from mars_titan.evaluation.forecast_panel import DAY_MICROSECONDS, ForecastPanel
from mars_titan.training import variant_screening

KIND = "online_arbiter_analysis_v1"
VERSIONS = "version_arbiter"
COMBINATION = "model_combination"
TIME_AVERAGING = "time_averaging"
# Salidas mezcladas del árbitro de versiones y el modo de `mixed` que las calcula.
VERSION_MODES = (
    ("arbiter", "online"),
    ("uniform_mixture", "uniform"),
    ("shuffled_outcomes", "shuffled"),
)
# Controles del árbitro de versiones con el papel de su contraste.
VERSION_CONTROLS = (
    ("common_epoch", "primary"),
    ("uniform_mixture", "required"),
    ("shuffled_outcomes", "required"),
)
# Ejecutores cuya calibración y cuya S son las de su propio trabajo.
OWN_RUNNERS = ("fit", "reader", "correction", "spaced", "correction_profile", "rule", "tabular")
IC_METRICS = ("session_ic", "session_rank_ic")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def joined(calibration, evaluation):
    """Panel de calibración y S en orden canónico y la fila de origen de cada una de sus filas.

    La fila j del panel conjunto procede de la fila `positions[j]` de la calibración seguida de
    S. La posición viaja como predicción, así que el orden es exactamente el que fija
    `ForecastPanel.from_columns`.
    """
    sd.ensure_before(calibration, evaluation)
    _require(
        calibration.markets == evaluation.markets, "Calibración y S declaran los mismos mercados"
    )
    parts = (calibration, evaluation)
    panel = ForecastPanel.from_columns(
        pa.concat_arrays([part.row_id for part in parts]),
        np.concatenate([np.asarray(part.markets)[part.market] for part in parts]),
        np.concatenate([part.prediction_at for part in parts]),
        np.concatenate([part.target for part in parts]),
        np.arange(calibration.rows + evaluation.rows, dtype=np.float64),
        markets=calibration.markets,
    )
    return panel, panel.prediction.astype(np.int64)


def scorable(panel, min_assets):
    """Sesiones con al menos `min_assets` activos y objetivo no constante."""
    starts = panel.session_starts
    spread = np.maximum.reduceat(panel.target, starts) - np.minimum.reduceat(panel.target, starts)
    return (panel.session_samples >= min_assets) & (spread > 0)


def mixed(options, maturity, stage, mode):
    """Emisión de S de una mezcla de opciones y las sesiones de S en que se abstiene.

    `options` son emisiones con la calibración y la S de las mismas filas, `maturity` la
    madurez de cada partición en su orden canónico y `stage` las opciones declaradas de la
    etapa. `mode` es `online`, `shuffled` o `uniform`. La calibración de la emisión devuelta
    lleva la salida de la etapa, que es la que fija su λ.
    """
    _require(mode in ("online", "shuffled", "uniform"), f"Modo de mezcla desconocido: {mode}")
    calibration, evaluation = options[0].calibration, options[0].panel
    _require(
        all(
            option.calibration is not None
            and option.calibration.cohort_sha256 == calibration.cohort_sha256
            and option.panel.cohort_sha256 == evaluation.cohort_sha256
            for option in options
        ),
        "Las opciones de una mezcla predicen las mismas filas de calibración y de S",
    )
    panel, positions = joined(calibration, evaluation)
    predictions = np.column_stack(
        [
            np.concatenate([option.calibration.prediction, option.prediction])[positions]
            for option in options
        ]
    )
    abstained = np.zeros(panel.sessions, dtype=bool)
    if mode == "uniform":
        output = arbiters.uniform_mixture(panel, predictions)
    else:
        matures_at = np.concatenate([maturity["calibration"], maturity["evaluation"]])
        result = arbiters.online_mixture(
            panel,
            predictions,
            matures_at[positions],
            window=stage["window"],
            min_assets=stage["min_assets"],
            margin=stage["margin"],
            seed=stage["seed"] if mode == "shuffled" else None,
        )
        output, abstained = result["prediction"], result["abstained"]
    source = np.empty(panel.rows)
    source[positions] = output
    sessions = zip(panel.session_market.tolist(), panel.session_time.tolist(), strict=True)
    joint = {key: position for position, key in enumerate(sessions)}
    own = zip(evaluation.session_market.tolist(), evaluation.session_time.tolist(), strict=True)
    return (
        analysis.Emission(
            evaluation,
            source[calibration.rows :],
            replace(calibration, prediction=source[: calibration.rows]),
        ),
        abstained[[joint[key] for key in own]],
    )


def stage_metric(emission, abstained, metric, *, min_assets):
    """Métrica por sesión de S en la que cada sesión abstenida que puntúa vale IC 0."""
    values, defined = analysis.session_metric(emission, metric, min_assets=min_assets)
    if metric in IC_METRICS:
        zero = abstained & scorable(emission.panel, min_assets)
        values, defined = np.where(zero, 0.0, values), defined | zero
    return values, defined


def compared_models(protocol, arms, root=variant_screening.ROOT):
    """Modelos comparados con su propia calibración y su propia S, en orden de etiqueta."""
    used = {
        row[side]
        for row in (
            *variant_screening.contrasts(protocol, root),
            *variant_screening.common_contrasts(protocol, root),
        )
        for side in ("variant", "control")
    }
    return sorted(label for label in used if label in arms and arms[label].runner in OWN_RUNNERS)


def version_options(protocol, arms, index, origin, label, seed, cases):
    """Emisión de la época común de un modelo y las de sus versiones, en orden de época."""
    common = analysis.emission(protocol, arms, index, origin, label, seed, cases)
    job = analysis.chosen_job(protocol, origin, index, label, seed, cases)
    versions = origin.epoch_versions(job["id"])
    declared = protocol["selection"]["versions"]["epochs"]
    _require(
        set(versions) <= set(declared) and len(versions) >= len(declared) - 1,
        f"{job['id']} no emite las versiones declaradas, salvo su época común",
    )
    return [common] + [
        analysis.Emission(
            panels["evaluation"], panels["evaluation"].prediction, panels["calibration"]
        )
        for panels in versions.values()
    ]


def _version_emissions(protocol, arms, index, origin, seed, cases, stage):
    """Emisiones de S del árbitro de versiones y de sus controles en un origen y una semilla."""
    for model in protocol["selection"]["versions"]["models"]:
        label = tuple(model.split("/", 1))
        if not analysis.has_emission(protocol, arms, origin, label):
            continue
        options = version_options(protocol, arms, index, origin, label, seed, cases)
        unchanged = np.zeros(options[0].panel.sessions, dtype=bool)
        yield (VERSIONS, model, "common_epoch"), options[0], unchanged
        for mode, kind in VERSION_MODES:
            yield (VERSIONS, model, mode), *mixed(options, origin.maturity, stage, kind)


def _averaging_emissions(protocol, arms, index, origin, seed, cases, names):
    """Emisiones de S de las variantes de F2 y F3 con las que se compara el árbitro."""
    for name in names:
        if not analysis.has_emission(protocol, arms, origin, tuple(name.split("/", 1))):
            continue
        emission = analysis.emission(
            protocol, arms, index, origin, tuple(name.split("/", 1)), seed, cases
        )
        yield (TIME_AVERAGING, name, "emission"), emission, np.zeros(emission.panel.sessions, bool)


def _combination_emissions(protocol, arms, index, origin, seed, cases, stage, compared, members):
    """Emisiones de S de la combinación de modelos y de su media simple."""
    # Dos modelos que comparten trabajo son la misma opción y cuentan una vez.
    chosen = {}
    for model in compared:
        if not analysis.has_emission(protocol, arms, origin, model):
            continue
        chosen.setdefault(
            analysis.chosen_job(protocol, origin, index, model, seed, cases)["id"], model
        )
    members[f"{origin.year}/{seed}"] = ["/".join(model) for model in chosen.values()]
    options = [
        analysis.emission(protocol, arms, index, origin, model, seed, cases)
        for model in chosen.values()
    ]
    for mode, kind in (("combination", "online"), ("simple_mean", "uniform")):
        yield (COMBINATION, "all", mode), *mixed(options, origin.maturity, stage, kind)


def _contrast(merged, variant, control, metric, role, direction, against=None):
    """Diferencia por sesión de `variant` menos `control` en las sesiones que ambos definen.

    El nombre termina en el modo del control, o en `against` si el control es otro modelo.
    """
    first, second = merged[(variant, metric)], merged[(control, metric)]
    defined = first[1] & second[1]
    return dict(
        name=f"{'/'.join(variant)}-{against or control[-1]}",
        role=role,
        model="/".join(variant[1:-1]),
        variant=variant,
        control=control,
        metric=metric,
        direction=direction,
        values=np.where(defined, first[0] - second[0], 0.0),
        defined=defined,
    )


def _public(row):
    return dict(
        name=row["name"],
        role=row["role"],
        metric=row["metric"],
        variant="/".join(row["variant"]),
        control="/".join(row["control"]),
        defined_sessions=int(np.count_nonzero(row["defined"])),
    )


def families(protocol, rows, averaging, combination, day):
    """Familia del árbitro de versiones, sus contrastes exigidos, decisiones y lo informado.

    `averaging` son las comparaciones con F2 y F3, que forman una familia propia, y
    `combination` el contraste de la combinación con su media simple. Ninguno decide nada.
    """
    inference = protocol["inference"]
    main = inference["block_days"]
    blocks = [main, *inference["sensitivity_block_days"]]
    primary = [row for row in rows if row["role"] == "primary"]
    declared = {
        length: analysis.evaluate_family(primary, day, block_length=length, inference=inference)
        for length in blocks
    }
    required = {
        row["name"]: analysis.evaluate_family([row], day, block_length=main, inference=inference)[
            "members"
        ][0]
        for row in rows
        if row["role"] == "required"
    }
    decisions = {}
    for k, row in enumerate(primary):
        member = declared[main]["members"][k]
        own = [r["name"] for r in rows if r["role"] == "required" and r["model"] == row["model"]]
        decisions[row["model"]] = dict(
            contrast=row["name"],
            passes=member["passes"] and all(required[name]["passes_marginal"] for name in own),
            primary_passes=member["passes"],
            required={name: required[name]["passes_marginal"] for name in own},
            robustness=dict(
                {f"blocks_{length}": declared[length]["members"][k]["passes"] for length in blocks},
                holm=member["passes_holm"],
            ),
        )
    reported = analysis.evaluate_family([combination], day, block_length=main, inference=inference)
    return dict(
        declared_family={str(length): family for length, family in declared.items()},
        required_contrasts=required,
        decisions=decisions,
        decision_rule=protocol["decision"]["passes"],
        time_averaging=analysis.evaluate_family(
            averaging, day, block_length=main, inference=inference
        ),
        model_combination=dict(reported["members"][0], sessions=reported["sessions"]),
        contrasts=[_public(row) for row in (*rows, *averaging, combination)],
    )


def _checked_stages(protocol, declaration):
    """Opciones de las dos etapas, que puntúan las sesiones como la criba."""
    stages = {
        name: arbiters.stage_options(declaration, identity)
        for name, identity in (
            (VERSIONS, arbiters.VERSION_ARBITER),
            (COMBINATION, arbiters.MODEL_COMBINATION),
        )
    }
    min_assets = protocol["contrast"]["primary"]["minimum_assets"]
    _require(
        all(stage["min_assets"] == min_assets for stage in stages.values()),
        "Los árbitros puntúan las sesiones con los activos mínimos de la criba",
    )
    metric = protocol["contrast"]["primary"]["metric"]
    _require(
        all(stage["metric"] == metric for stage in stages.values()),
        "Los árbitros se miden con la métrica principal de la criba",
    )
    return stages


def _series(protocol, arms, index, origins, stages, compared):
    """Serie por sesión de cada salida y métrica, con su media de semillas y los orígenes unidos.

    Devuelve también las sesiones de S y lo que el informe guarda de la lectura: la cobertura
    de cada salida, los miembros de la combinación y los casos de selección.
    """
    min_assets = protocol["contrast"]["primary"]["minimum_assets"]
    # El MAE menos el del control cero no se calcula aquí, porque el cero no es una opción.
    secondary = [m for m in protocol["contrast"]["secondary"] if m != "session_mae_minus_zero"]
    metrics = [protocol["contrast"]["primary"]["metric"], *secondary]
    averaging = stages[VERSIONS]["time_averaging"]["compared_with"]
    cases, series, times, coverage, members = [], {}, [], {}, {}
    for year in sorted(origins):
        origin, per_origin = origins[year], {}
        missing = [
            (TIME_AVERAGING, name, "emission")
            for name in averaging
            if not analysis.has_emission(protocol, arms, origin, tuple(name.split("/", 1)))
        ]
        for model in protocol["selection"]["versions"]["models"]:
            if not analysis.has_emission(protocol, arms, origin, tuple(model.split("/", 1))):
                missing.extend(
                    (VERSIONS, model, mode)
                    for mode in ("common_epoch", *(mode for mode, _ in VERSION_MODES))
                )
        for seed in protocol["seeds"]:
            emissions = (
                *_version_emissions(protocol, arms, index, origin, seed, cases, stages[VERSIONS]),
                *_averaging_emissions(protocol, arms, index, origin, seed, cases, averaging),
                *_combination_emissions(
                    protocol,
                    arms,
                    index,
                    origin,
                    seed,
                    cases,
                    stages[COMBINATION],
                    compared,
                    members,
                ),
            )
            for label, emission, abstained in emissions:
                for name in metrics:
                    value = stage_metric(emission, abstained, name, min_assets=min_assets)
                    per_origin.setdefault((label, name), []).append(value)
                coverage.setdefault("/".join(label), []).append(float(1.0 - abstained.mean()))
            for label in missing:
                sessions = emissions[0][1].panel.sessions
                for metric in metrics:
                    per_origin.setdefault((label, metric), []).append(
                        (np.zeros(sessions), np.zeros(sessions, dtype=bool))
                    )
                coverage.setdefault("/".join(label), []).append(0.0)
        # Todas las emisiones de un origen comparten las sesiones de S.
        times.append(np.asarray(emissions[0][1].panel.session_time))
        for key, items in per_origin.items():
            series.setdefault(key, []).append(analysis.seed_average(items))
        origin.stages.clear()
    session_time = np.concatenate(times)
    _require(np.all(np.diff(session_time) > 0), "Las sesiones de S no siguen el orden temporal")
    merged = {
        key: (
            np.concatenate([values for values, _ in parts]),
            np.concatenate([defined for _, defined in parts]),
        )
        for key, parts in series.items()
    }
    read = dict(
        coverage={label: float(np.mean(values)) for label, values in sorted(coverage.items())},
        combination_members=members,
        cases=cases,
    )
    return session_time, merged, read


def _contrasts(protocol, merged, averaging):
    """Contrastes del árbitro de versiones, de las medias de pesos y de la combinación."""
    metric = protocol["contrast"]["primary"]["metric"]
    direction = protocol["contrast"]["primary"]["direction"]
    rows = [
        _contrast(
            merged,
            (VERSIONS, model, "arbiter"),
            (VERSIONS, model, control),
            metric,
            role,
            direction,
        )
        for model in protocol["selection"]["versions"]["models"]
        for control, role in VERSION_CONTROLS
    ]
    averaged = [
        _contrast(
            merged,
            (VERSIONS, averaging["arbiter_of"], "arbiter"),
            (TIME_AVERAGING, name, "emission"),
            metric,
            "reported",
            direction,
            against=name,
        )
        for name in averaging["compared_with"]
    ]
    combination = _contrast(
        merged,
        (COMBINATION, "all", "combination"),
        (COMBINATION, "all", "simple_mean"),
        metric,
        "reported",
        direction,
    )
    return rows, averaged, combination


def _means(merged):
    """Media y sesiones definidas de cada salida en cada métrica, por orden de salida."""
    means = {"/".join(label): {} for label in sorted({label for label, _ in merged})}
    for (label, name), (values, defined) in merged.items():
        means["/".join(label)][name] = dict(
            mean=float(values[defined].mean()) if defined.any() else None,
            defined_sessions=int(defined.sum()),
        )
    return means


def analyze(protocol, declaration, jobs, origins, root=variant_screening.ROOT):
    """Series por sesión, contrastes, familia y decisiones de los árbitros sobre la criba.

    `declaration` es la de `online_arbiters`, `jobs` el catálogo sellado por el motor y
    `origins` asigna a cada año su `Origin`, que debe recibir las versiones de los ajustes.
    """
    _require(
        sorted(origins) == [origin["year"] for origin in protocol["origins"]],
        "Los árbitros necesitan los cuatro orígenes de la criba",
    )
    stages = _checked_stages(protocol, declaration)
    arms = variant_screening.all_arms(protocol, root)
    index = analysis.job_index(jobs)
    versioned = protocol["selection"]["versions"]["models"]
    compared = compared_models(protocol, arms, root)
    averaging = stages[VERSIONS]["time_averaging"]
    _require(
        averaging["arbiter_of"] in versioned
        and all(tuple(name.split("/", 1)) in compared for name in averaging["compared_with"]),
        "La comparación con las medias de pesos usa un modelo con versiones y modelos de la "
        "criba con predicciones propias",
    )
    session_time, merged, read = _series(protocol, arms, index, origins, stages, compared)
    rows, averaged, combination = _contrasts(protocol, merged, averaging)
    report = families(protocol, rows, averaged, combination, session_time // DAY_MICROSECONDS)
    report.update(
        kind=KIND,
        sessions=len(session_time),
        first_session=int(session_time[0]),
        last_session=int(session_time[-1]),
        stages=stages,
        **read,
        models=_means(merged),
    )
    return report, dict(
        session_time=session_time, merged=merged, contrasts=[*rows, *averaged, combination]
    )
