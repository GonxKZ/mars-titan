"""Agente cognitivo con herramientas sobre las emisiones de la campaña DGX.

Lleva a la evaluación la cadena de Karpathy (núcleo cognitivo pequeño, herramientas, bucle de
realimentación y agente que funciona) sin entrenar nada nuevo ni abrir el test final:

- Núcleo: la emisión del Titans-MAC padre (`core`), cuya memoria ya aprende en tiempo de prueba
  y cuyos parámetros no necesitan memorizar el mercado.
- Herramientas: emisiones de otros modelos de la misma campaña agrupadas por la capacidad que
  aportan (recuperación episódica, bucle de refinamiento, puerta metacognitiva, pares entre
  activos, áreas holográficas...). Cada herramienta declara uno o varios modelos.
- Bucle de realimentación: AdaNormalHedge.TV por mercado (`online_arbiters.online_mixture`)
  reparte la confianza con la pérdida (1 − IC)/2 de las sesiones cuyas etiquetas ya maduraron,
  con la madurez real de cada etiqueta de la vista. El estado recorre la calibración y después
  S del mismo origen, así que cada sesión de S decide solo con resultados ya conocidos. Con
  `margin`, la sesión se abstiene (predicción cero) si ninguna opción tiene fiabilidad madura
  por encima del margen.
- Agente: la mezcla emitida en S.

Variantes por origen y semilla, todas sobre las mismas filas:

- `core`: el núcleo solo.
- `agent`: núcleo y todas las herramientas con realimentación y abstención.
- `agent_without_<h>`: el agente sin la herramienta h (contribución marginal con las demás).
- `core_with_<h>`: el núcleo con la herramienta h sola (contribución aislada).
- `agent_uniform`: media uniforme de las mismas opciones tipificadas, el control obligado por el
  enigma de la combinación de previsiones.
- `agent_shuffled_feedback`: el agente con los resultados maduros desordenados en el tiempo; mide
  si el bucle aprende del orden real de los resultados.
- `agent_with_placebo`: el agente con una herramienta sin información, la predicción del núcleo
  permutada entre activos dentro de cada sesión. No debería mejorar al agente.
- `agent_without_abstention`: el mismo agente sin margen, si se declara margen.
- `agent_contextual`: el agente con enrutado por régimen, si se declara `context`. Cada opción
  tiene un generalista siempre despierto y un especialista por régimen que solo despierta en su
  régimen (los expertos especialistas de Freund, Schapire, Singer y Warmuth, 1997, con la
  extensión de AdaNormalHedge para opciones que duermen). El régimen de una sesión se calcula
  solo con sesiones ya maduras: dispersión transversal reciente alta o baja frente a la mediana
  de las maduras, y mercado reciente al alza o a la baja. La mezcla compite así con la mejor
  política que elige herramienta según el régimen, con la misma garantía de arrepentimiento.
- `shapley/<subconjunto>`: el núcleo con cada subconjunto de herramientas, sin abstención, si se
  declara `shapley`. El valor de Shapley de cada herramienta (Shapley, 1953) reparte
  exactamente la ganancia del agente sin abstención sobre el núcleo entre las herramientas,
  con sus redundancias y complementos. Es lineal en las métricas por sesión, así que su serie
  por sesión admite el mismo bootstrap por bloques, y la suma de las de todas las herramientas
  es, sesión a sesión, la ganancia del agente completo.

La métrica por sesión es una correlación del protocolo (la mezcla emite predicciones
tipificadas, que solo ordenan activos). Una sesión abstenida que puntúa vale IC 0, de modo que
el agente no se mide solo donde elige actuar. Las semillas se promedian sesión a sesión y los
orígenes forman una única serie de días UTC. `evaluate_family` da, por familia, intervalos por
bootstrap circular por bloques de días, intervalos simultáneos y valores p de Romano y Wolf y
de Holm. El informe guarda además, para investigar después, la serie por sesión de cada
variante, la cobertura de cada mezcla y el peso medio de cada herramienta por mes.
"""

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.evaluation import online_arbiters as arbiters
from mars_titan.evaluation import variant_screening_analysis as analysis
from mars_titan.evaluation.arbiter_analysis import IC_METRICS, joined, stage_metric
from mars_titan.evaluation.forecast_panel import DAY_MICROSECONDS, ForecastPanel

KIND = "cognitive_agent_v1"
PARTITIONS = ("calibration", "evaluation")
CORE = "core"
AGENT = "agent"
UNIFORM = "agent_uniform"
SHUFFLED = "agent_shuffled_feedback"
PLACEBO = "agent_with_placebo"
NO_ABSTENTION = "agent_without_abstention"
WITHOUT = "agent_without_"
WITH = "core_with_"
PLACEBO_OPTION = "placebo"
CONTEXTUAL = "agent_contextual"
SUBSET = "shapley/"
EMPTY_SUBSET = "shapley/"
# Regímenes: dispersión transversal reciente alta o baja y mercado reciente al alza o a la baja.
REGIMES = ("low_dispersion_down", "low_dispersion_up", "high_dispersion_down", "high_dispersion_up")
UNKNOWN_REGIME = -1
# Shapley exacto enumera 2^T subconjuntos por origen y semilla: con 8 herramientas son 256
# mezclas, el límite de un cálculo exacto razonable en CPU.
MAX_SHAPLEY_TOOLS = 8
# Herramientas por agente: con el núcleo y el placebo, la mezcla queda por debajo del máximo
# de opciones de AdaNormalHedge (128) aunque cada herramienta declare varios modelos.
MAX_TOOLS = 16
MAX_TOOL_MODELS = 6
# Mínimo de réplicas del bootstrap para que los valores p de Romano y Wolf, que usan
# (1 + n)/(1 + R), puedan bajar de 0,01.
MIN_REPLICATES = 100
FIRST_YEAR = 2005


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum, maximum=None):
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def checked_declaration(document):
    """Validar la declaración del agente antes de leer ninguna predicción."""
    _require(isinstance(document, dict), "La declaración del agente es un objeto JSON")
    _require(
        document.get("kind") == KIND and document.get("schema_version") == 1,
        f"La declaración debe ser {KIND} con schema_version 1",
    )
    _require(document.get("final_test_access") is False, "El agente no abre el test final")
    core, tools = document.get("core"), document.get("tools")
    _require(isinstance(core, str) and core, "El núcleo es un modelo de la campaña")
    _require(
        isinstance(tools, dict) and 1 <= len(tools) <= MAX_TOOLS,
        f"Se declaran entre 1 y {MAX_TOOLS} herramientas",
    )
    seen = {core}
    for name, models in tools.items():
        _require(
            isinstance(name, str) and name.isidentifier() and name != PLACEBO_OPTION,
            f"Nombre de herramienta no válido: {name!r}",
        )
        _require(
            isinstance(models, list)
            and 1 <= len(models) <= MAX_TOOL_MODELS
            and all(isinstance(model, str) and model for model in models),
            f"La herramienta {name} declara entre 1 y {MAX_TOOL_MODELS} modelos",
        )
        _require(
            not seen & set(models) and len(set(models)) == len(models),
            f"Un modelo solo pertenece al núcleo o a una herramienta ({name})",
        )
        seen |= set(models)
    feedback = document.get("feedback")
    _require(
        isinstance(feedback, dict)
        and set(feedback) == {"window", "min_assets", "margin", "shuffled_seed"}
        and _integer(feedback["window"], 1, arbiters.MAX_WINDOW)
        and _integer(feedback["min_assets"], 2)
        and (
            feedback["margin"] is None
            or (
                isinstance(feedback["margin"], (int, float))
                and not isinstance(feedback["margin"], bool)
                and math.isfinite(feedback["margin"])
            )
        )
        and _integer(feedback["shuffled_seed"], 0),
        "La realimentación declara ventana, mínimo de activos, margen y semilla del desorden",
    )
    placebo = document.get("placebo")
    _require(
        placebo is None
        or (
            isinstance(placebo, dict) and set(placebo) == {"seed"} and _integer(placebo["seed"], 0)
        ),
        "El placebo declara solo su semilla, o no se declara",
    )
    context = document.get("context")
    _require(
        context is None
        or (
            isinstance(context, dict)
            and set(context) == {"window"}
            and _integer(context["window"], 2, arbiters.MAX_WINDOW)
        ),
        "El contexto declara solo la ventana de sesiones maduras del régimen, o no se declara",
    )
    _require(
        type(document.get("shapley")) is bool
        and (not document["shapley"] or len(tools) <= MAX_SHAPLEY_TOOLS),
        f"shapley es booleano y exacto con hasta {MAX_SHAPLEY_TOOLS} herramientas",
    )
    _require(document.get("metric") in IC_METRICS, f"La métrica es una de {IC_METRICS}")
    inference = document.get("inference")
    _require(
        isinstance(inference, dict)
        and set(inference) == {"block_days", "replicates", "seed", "confidence"}
        and _integer(inference["block_days"], 1)
        and _integer(inference["replicates"], MIN_REPLICATES)
        and _integer(inference["seed"], 0)
        and type(inference["confidence"]) is float
        and 0.5 < inference["confidence"] < 1,
        "La inferencia declara bloque en días, réplicas, semilla y confianza",
    )
    origins, seeds = document.get("origins"), document.get("seeds")
    _require(
        isinstance(origins, list)
        and origins
        and all(_integer(year, FIRST_YEAR, 2023) for year in origins)
        and origins == sorted(set(origins)),
        "Los orígenes son años crecientes sin repetir, anteriores al test final",
    )
    _require(
        isinstance(seeds, list)
        and seeds
        and all(_integer(seed, 0) for seed in seeds)
        and len(set(seeds)) == len(seeds),
        "Las semillas son enteros sin repetir",
    )
    if feedback["margin"] is not None:
        document = dict(document, feedback=dict(feedback, margin=float(feedback["margin"])))
    return document


def load_emission(runs, model, year, seed, markets):
    """Calibración y S de un trabajo terminado de la campaña, con su predicción puntual."""
    folder = Path(runs) / model / str(year) / f"seed-{seed}"
    panels = {}
    for name in PARTITIONS:
        path = folder / f"{name}-predictions.parquet"
        _require(path.is_file(), f"Falta la emisión {path}")
        panel = ForecastPanel.from_arrow(pq.read_table(path), markets=markets)
        _require(
            int(panel.prediction_at.max()) < FINAL_TEST_START_US,
            f"{path} llega al test final",
        )
        panels[name] = panel
    evaluation = panels["evaluation"]
    return analysis.Emission(evaluation, evaluation.prediction, panels["calibration"])


def _sample_keys(table):
    moments = pc.cast(pc.cast(table["prediction_at"], pa.int64()), pa.string())
    return pc.binary_join_element_wise(table["asset_id"], moments, "/").to_pylist()


def aligned_maturity(tables, emission):
    """Madurez de cada fila de la calibración y de S en el orden canónico de sus paneles.

    `tables` son las de `online_calibration.view_maturity`. El objetivo de cada fila debe
    coincidir con el de la vista: si no, la emisión no predice las etiquetas de esa vista.
    """
    result = {}
    for name, panel in (("calibration", emission.calibration), ("evaluation", emission.panel)):
        table = tables[name]
        keys = _sample_keys(table)
        position = dict(zip(keys, range(len(keys)), strict=True))
        rows = panel.row_id.to_pylist()
        _require(
            len(rows) == len(keys) and all(row in position for row in rows),
            f"Las filas de {name} no son las de la vista",
        )
        order = np.fromiter((position[row] for row in rows), dtype=np.int64, count=len(rows))
        _require(
            np.array_equal(table["target"].to_numpy()[order], panel.target),
            f"Los objetivos de {name} no son los de la vista",
        )
        result[name] = table["matures_at"].to_numpy()[order].astype(np.int64)
    return result


def permuted(emission, seed):
    """Placebo: la predicción permutada entre activos dentro de cada sesión, en las dos
    particiones. Conserva la distribución de cada sesión y destruye su información."""
    rng = np.random.default_rng(seed)

    def shuffle(panel, prediction):
        values = np.array(prediction, dtype=np.float64)
        for start, count in zip(panel.session_starts, panel.session_samples, strict=True):
            values[start : start + count] = values[start : start + count][rng.permutation(count)]
        return values

    calibration = emission.calibration
    calibration = replace(calibration, prediction=shuffle(calibration, calibration.prediction))
    evaluation = shuffle(emission.panel, emission.prediction)
    return analysis.Emission(
        replace(emission.panel, prediction=evaluation), evaluation, calibration
    )


def _checked_options(options):
    calibration, evaluation = options[0].calibration, options[0].panel
    _require(
        all(
            option.calibration.cohort_sha256 == calibration.cohort_sha256
            and option.panel.cohort_sha256 == evaluation.cohort_sha256
            for option in options
        ),
        "Las opciones de una mezcla predicen las mismas filas de calibración y de S",
    )
    return calibration, evaluation


def _stacked(options, positions):
    return np.column_stack(
        [
            np.concatenate([option.calibration.prediction, option.prediction])[positions]
            for option in options
        ]
    )


def _emitted(panel, positions, output, calibration, evaluation):
    """Emisión de S de una salida del panel conjunto y la sesión conjunta de cada sesión de S."""
    source = np.empty(panel.rows)
    source[positions] = output
    joint = {
        key: index
        for index, key in enumerate(
            zip(panel.session_market.tolist(), panel.session_time.tolist(), strict=True)
        )
    }
    own = np.array(
        [
            joint[key]
            for key in zip(
                evaluation.session_market.tolist(), evaluation.session_time.tolist(), strict=True
            )
        ],
        dtype=np.int64,
    )
    emission = analysis.Emission(
        evaluation,
        source[calibration.rows :],
        replace(calibration, prediction=source[: calibration.rows]),
    )
    return emission, own


def mixture(options, maturity, feedback, mode):
    """Emisión de S de una mezcla, sus sesiones abstenidas y los pesos de cada sesión de S.

    `mode` es `online` (el bucle con los resultados maduros en orden), `shuffled` (los mismos
    resultados desordenados en el tiempo) o `uniform` (media de las opciones tipificadas). Con
    una sola opción la emisión es esa opción, sin abstención: tipificar no cambia su IC.
    """
    _require(mode in ("online", "shuffled", "uniform"), f"Modo de mezcla desconocido: {mode}")
    calibration, evaluation = _checked_options(options)
    if len(options) == 1:
        return (
            analysis.Emission(evaluation, options[0].prediction, calibration),
            np.zeros(evaluation.sessions, dtype=bool),
            np.ones((evaluation.sessions, 1)),
        )
    panel, positions = joined(calibration, evaluation)
    predictions = _stacked(options, positions)
    if mode == "uniform":
        output = arbiters.uniform_mixture(panel, predictions)
        abstained = np.zeros(panel.sessions, dtype=bool)
        weights = np.full((panel.sessions, len(options)), 1.0 / len(options))
    else:
        matures_at = np.concatenate([maturity["calibration"], maturity["evaluation"]])
        result = arbiters.online_mixture(
            panel,
            predictions,
            matures_at[positions],
            window=feedback["window"],
            min_assets=feedback["min_assets"],
            margin=feedback["margin"],
            seed=feedback["shuffled_seed"] if mode == "shuffled" else None,
        )
        output, abstained, weights = result["prediction"], result["abstained"], result["mixtures"]
    emission, own = _emitted(panel, positions, output, calibration, evaluation)
    return emission, abstained[own], weights[own]


def regimes(panel, session_matures, window):
    """Régimen causal de cada sesión del panel, con solo sesiones ya maduras al decidirla.

    Con las últimas `window` sesiones maduras del mismo mercado (por instante de decisión), la
    sesión es de dispersión alta si su dispersión transversal media supera la mediana de las
    dispersiones de todas las sesiones maduras, y al alza si su rendimiento medio es positivo.
    Mientras no haya `window` sesiones maduras el régimen es desconocido y solo despiertan los
    generalistas. Devuelve el índice de `REGIMES` o `UNKNOWN_REGIME` de cada sesión.
    """
    starts, counts = panel.session_starts, panel.session_samples
    dispersion = np.array(
        [
            panel.target[start : start + count].std()
            for start, count in zip(starts, counts, strict=True)
        ]
    )
    level = np.add.reduceat(panel.target, starts) / counts
    result = np.full(panel.sessions, UNKNOWN_REGIME, dtype=np.int64)
    for market in np.unique(panel.session_market):
        sessions = np.flatnonzero(panel.session_market == market)
        by_maturity = sessions[np.argsort(session_matures[sessions], kind="stable")]
        matured = session_matures[by_maturity]
        for session in sessions:
            known = by_maturity[
                : np.searchsorted(matured, panel.session_time[session], side="right")
            ]
            if known.size < window:
                continue
            recent = known[np.argsort(panel.session_time[known], kind="stable")][-window:]
            high = dispersion[recent].mean() > np.median(dispersion[known])
            up = level[recent].mean() > 0
            result[session] = 2 * int(high) + int(up)
    return result


def contextual_mixture(options, maturity, feedback, window):
    """Mezcla con generalistas y especialistas por régimen: emisión de S, abstenciones, pesos de
    las opciones base y régimen de cada sesión de S.

    Cada opción b aporta 1 + len(REGIMES) copias a AdaNormalHedge.TV: un generalista siempre
    despierto y un especialista por régimen, despierto solo en el suyo. Todas las copias de b
    sufren la pérdida de b, pero solo las despiertas acumulan arrepentimiento. El peso de b en
    una sesión es la suma de los de sus copias despiertas. La fiabilidad y la abstención son
    las de `online_mixture` sobre las opciones base.
    """
    calibration, evaluation = _checked_options(options)
    _require(len(options) >= 2, "El enrutado por régimen necesita al menos dos opciones")
    panel, positions = joined(calibration, evaluation)
    predictions = _stacked(options, positions)
    matures_at = np.concatenate([maturity["calibration"], maturity["evaluation"]])[positions]
    _require(
        matures_at.dtype.kind == "i" and (matures_at > panel.prediction_at).all(),
        "Cada etiqueta madura después de su decisión",
    )
    base, groups = len(options), 1 + len(REGIMES)
    tipified = np.column_stack(
        [arbiters.standardized(panel, predictions[:, option]) for option in range(base)]
    )
    losses, scored = arbiters.session_losses(panel, predictions, min_assets=feedback["min_assets"])
    ic = 1.0 - 2.0 * losses
    session_matures = np.maximum.reduceat(matures_at, panel.session_starts)
    regime = regimes(panel, session_matures, window)
    weights = np.full((panel.sessions, base), np.nan)
    reliability = np.full((panel.sessions, base), np.nan)
    for market in np.unique(panel.session_market):
        sessions = np.flatnonzero(panel.session_market == market)
        awake = np.zeros((sessions.size, base * groups), dtype=bool)
        awake[:, :base] = True
        for index in range(len(REGIMES)):
            awake[regime[sessions] == index, base * (1 + index) : base * (2 + index)] = True
        mixtures, _ = arbiters.mature_mixture(
            panel.session_time[sessions],
            session_matures[sessions],
            np.tile(losses[sessions], (1, groups)),
            awake,
            np.ones(base * groups),
            scored=awake & scored[sessions, None],
            hedge=arbiters.AdaNormalHedgeTV,
        )
        weights[sessions] = mixtures.reshape(sessions.size, groups, base).sum(
            axis=1, where=awake.reshape(sessions.size, groups, base), initial=0.0
        )
        reliability[sessions] = arbiters._reliability(
            ic[sessions],
            scored[sessions],
            panel.session_time[sessions],
            session_matures[sessions],
            feedback["window"],
        )
    margin = feedback["margin"]
    abstained = (
        np.zeros(panel.sessions, dtype=bool)
        if margin is None
        else ~np.any(reliability > margin, axis=1)
    )
    output = np.zeros(panel.rows)
    for session in np.flatnonzero(~abstained):
        start, count = panel.session_starts[session], panel.session_samples[session]
        output[start : start + count] = tipified[start : start + count] @ weights[session]
    emission, own = _emitted(panel, positions, output, calibration, evaluation)
    return emission, abstained[own], weights[own], regime[own]


def subset_name(tools):
    """Nombre de la variante del núcleo con un subconjunto de herramientas, en orden alfabético."""
    return SUBSET + "+".join(sorted(tools))


def variants(declaration, emissions, maturity):
    """Todas las variantes del agente en un origen y una semilla.

    Devuelve, por variante, la emisión de S, sus sesiones abstenidas, los pesos por sesión, el
    nombre de cada opción en el orden de esos pesos y, con enrutado, el régimen de cada sesión.
    Dos variantes con las mismas opciones, modo y margen comparten un único cálculo.
    """
    core_name, tools, feedback = declaration["core"], declaration["tools"], declaration["feedback"]
    core = emissions[core_name]
    members = {
        tool: [(model, emissions[model]) for model in models] for tool, models in tools.items()
    }
    cache = {}

    def options(selected, extra=()):
        named = [(core_name, core)]
        named += [pair for tool in tools if tool in selected for pair in members[tool]]
        return named + list(extra)

    def run(named, mode="online", margin=feedback["margin"]):
        key = (tuple(name for name, _ in named), mode, margin)
        if key not in cache:
            chosen, settings, regime = [e for _, e in named], dict(feedback, margin=margin), None
            if mode == "contextual":
                emission, abstained, weights, regime = contextual_mixture(
                    chosen, maturity, settings, declaration["context"]["window"]
                )
            else:
                emission, abstained, weights = mixture(chosen, maturity, settings, mode)
            cache[key] = dict(
                emission=emission,
                abstained=abstained,
                weights=weights,
                options=list(key[0]),
                regime=regime,
            )
        return cache[key]

    every = set(tools)
    result = {
        CORE: run(options(set())),
        AGENT: run(options(every)),
        UNIFORM: run(options(every), "uniform"),
        SHUFFLED: run(options(every), "shuffled"),
    }
    for tool in tools:
        result[WITHOUT + tool] = run(options(every - {tool}))
        result[WITH + tool] = run(options({tool}))
    if declaration["placebo"] is not None:
        placebo = permuted(core, declaration["placebo"]["seed"])
        result[PLACEBO] = run(options(every, [(PLACEBO_OPTION, placebo)]))
    if feedback["margin"] is not None:
        result[NO_ABSTENTION] = run(options(every), margin=None)
    if declaration["context"] is not None:
        result[CONTEXTUAL] = run(options(every), "contextual")
    if declaration["shapley"]:
        for size in range(len(tools) + 1):
            for subset in combinations(tools, size):
                result[subset_name(subset)] = run(options(set(subset)), margin=None)
    return result


def shapley_values(tools, series):
    """Serie por sesión del valor de Shapley de cada herramienta.

    El juego es v(S) = métrica por sesión del núcleo con las herramientas S, sin abstención.
    φ_i = Σ_{S ⊆ T∖{i}} |S|! (|T| − |S| − 1)! / |T|! · (v(S ∪ {i}) − v(S)). Una sesión cuenta si
    todos los subconjuntos la definen. En cada sesión, Σ_i φ_i = v(T) − v(∅).
    """
    tools, count = list(tools), len(tools)
    every = [subset for size in range(count + 1) for subset in combinations(tools, size)]
    defined = np.logical_and.reduce([series[subset_name(subset)][1] for subset in every])
    result = {}
    for tool in tools:
        others = [other for other in tools if other != tool]
        total = np.zeros(defined.shape)
        for size in range(count):
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            for subset in combinations(others, size):
                total += weight * (
                    series[subset_name((*subset, tool))][0] - series[subset_name(subset)][0]
                )
        result[tool] = (np.where(defined, total, 0.0), defined)
    return result


def tool_weights(declaration, outcome):
    """Peso medio de cada herramienta (suma de sus modelos) y del núcleo en las sesiones de S que
    emiten, en total, por mes UTC y, con enrutado, por régimen. Telemetría para investigar cómo
    reparte el bucle su confianza."""
    owner = {declaration["core"]: CORE, PLACEBO_OPTION: PLACEBO_OPTION}
    for tool, models in declaration["tools"].items():
        owner.update({model: tool for model in models})
    panel, weights = outcome["emission"].panel, outcome["weights"]
    acting = ~outcome["abstained"] & np.isfinite(weights).all(axis=1)
    months = np.datetime_as_string(panel.session_time.astype("datetime64[us]"), unit="M")
    groups = sorted({owner[name] for name in outcome["options"]})
    shares = {group: np.zeros(panel.sessions) for group in groups}
    for column, name in enumerate(outcome["options"]):
        shares[owner[name]] += np.nan_to_num(weights[:, column])

    def mean_shares(rows):
        return (
            {group: float(shares[group][rows].mean()) for group in groups} if rows.any() else None
        )

    result = dict(
        overall=mean_shares(acting),
        monthly={
            str(month): mean_shares(acting & (months == month))
            for month in np.unique(months[acting])
        },
    )
    if outcome["regime"] is not None:
        result["by_regime"] = {
            name: dict(
                sessions=int((acting & (outcome["regime"] == index)).sum()),
                shares=mean_shares(acting & (outcome["regime"] == index)),
            )
            for index, name in enumerate(REGIMES)
        }
    return result


def origin_series(declaration, campaign, state, views, year, input_policy=HISTORICAL_MASKED):
    """Serie por sesión de S de cada variante en un origen, promediada sobre las semillas."""
    from mars_titan.evaluation.online_calibration import view_maturity
    from mars_titan.training.corpus_inputs import CorpusDataset

    fold = next(o["fold"] for o in campaign["origins"] if o["year"] == year)
    view = Path(views) / fold / "manifest.json"
    identity = CorpusDataset(view, cache_bytes=0, input_policy=input_policy).identity
    tables = view_maturity(view, input_policy, identity, partitions=PARTITIONS)
    markets = (campaign["market"],)
    models = [declaration["core"]] + [m for ms in declaration["tools"].values() for m in ms]
    metric, min_assets = declaration["metric"], declaration["feedback"]["min_assets"]
    per_seed, telemetry, inputs, reference = {}, {}, {}, None
    for seed in declaration["seeds"]:
        emissions = {
            model: load_emission(Path(state) / "runs", model, year, seed, markets)
            for model in models
        }
        for model in models:
            folder = Path(state) / "runs" / model / str(year) / f"seed-{seed}"
            for name in PARTITIONS:
                path = folder / f"{name}-predictions.parquet"
                inputs[str(path.relative_to(state))] = hashlib.sha256(path.read_bytes()).hexdigest()
        maturity = aligned_maturity(tables, emissions[declaration["core"]])
        outcomes = variants(declaration, emissions, maturity)
        reference = outcomes[CORE]["emission"].panel
        per_seed[seed] = {
            name: stage_metric(o["emission"], o["abstained"], metric, min_assets=min_assets)
            for name, o in outcomes.items()
        }
        telemetry[str(seed)] = {
            name: dict(
                coverage=float(1.0 - o["abstained"].mean()),
                weights=tool_weights(declaration, o) if len(o["options"]) > 1 else None,
            )
            for name, o in outcomes.items()
            if not name.startswith(SUBSET)
        }
    names = list(per_seed[declaration["seeds"][0]])
    averaged = {
        name: analysis.seed_average([per_seed[seed][name] for seed in declaration["seeds"]])
        for name in names
    }
    return dict(
        year=year,
        session_time=reference.session_time.copy(),
        day=np.floor_divide(reference.session_time, DAY_MICROSECONDS),
        series=averaged,
        telemetry=telemetry,
        inputs=inputs,
    )


def families(declaration, series, day):
    """Familias de contrastes declaradas por diseño, cada una con su propia corrección."""
    inference = declaration["inference"]

    def member(name, values, defined, direction="greater"):
        return dict(
            name=name, values=np.where(defined, values, 0.0), defined=defined, direction=direction
        )

    def contrast(variant, control):
        values_v, defined_v = series[variant]
        values_c, defined_c = series[control]
        return member(f"{variant} − {control}", values_v - values_c, defined_v & defined_c)

    tools = list(declaration["tools"])
    declared = {
        "marginal_tools": [contrast(AGENT, WITHOUT + tool) for tool in tools],
        "isolated_tools": [contrast(WITH + tool, CORE) for tool in tools],
        "system": [contrast(AGENT, CORE), contrast(AGENT, UNIFORM), contrast(AGENT, SHUFFLED)],
    }
    if CONTEXTUAL in series:
        declared["system"] += [contrast(CONTEXTUAL, CORE), contrast(CONTEXTUAL, AGENT)]
    if PLACEBO in series:
        declared["placebo"] = [contrast(PLACEBO, AGENT)]
    if NO_ABSTENTION in series:
        declared["abstention"] = [contrast(AGENT, NO_ABSTENTION)]
    if declaration["shapley"]:
        values = shapley_values(tools, series)
        declared["shapley"] = [member(f"shapley({tool})", *values[tool]) for tool in tools]
    return {
        name: analysis.evaluate_family(
            members, day, block_length=inference["block_days"], inference=inference
        )
        for name, members in declared.items()
    }


def analyze(declaration, campaign, state, views, *, processes=1):
    """Informe completo del agente con las emisiones de la campaña y su serie por sesión."""
    declaration = checked_declaration(declaration)
    _require(campaign.get("final_test_access") is False, "La campaña no abre el test final")
    arguments = [
        (declaration, campaign, str(state), str(views), year) for year in declaration["origins"]
    ]
    if processes == 1:
        origins = [origin_series(*item) for item in arguments]
    else:
        with ProcessPoolExecutor(processes) as pool:
            origins = list(pool.map(origin_series, *zip(*arguments, strict=True)))
    origins.sort(key=lambda item: item["year"])
    session_time = np.concatenate([item["session_time"] for item in origins])
    _require(
        np.all(np.diff(session_time) > 0),
        "Las evaluaciones de los orígenes no se solapan y siguen el calendario",
    )
    days = np.concatenate([item["day"] for item in origins])
    series = {
        name: (
            np.concatenate([item["series"][name][0] for item in origins]),
            np.concatenate([item["series"][name][1] for item in origins]),
        )
        for name in origins[0]["series"]
    }
    if declaration["shapley"]:
        tools = list(declaration["tools"])
        for tool, value in shapley_values(tools, series).items():
            series[f"shapley({tool})"] = value
    means = {
        name: dict(
            mean=float(values[defined].mean()) if defined.any() else None,
            sessions=int(defined.sum()),
        )
        for name, (values, defined) in series.items()
    }
    document = json.dumps(declaration, sort_keys=True, ensure_ascii=False)
    report = dict(
        kind=f"{KIND}_report",
        declaration_sha256=hashlib.sha256(document.encode()).hexdigest(),
        final_test_opened=False,
        metric=declaration["metric"],
        means=means,
        families=families(declaration, series, days),
        origins={
            str(item["year"]): dict(telemetry=item["telemetry"], inputs=item["inputs"])
            for item in origins
        },
    )
    return report, dict(session_time=session_time, series=series)


def write_series(path, sessions):
    """Serie por sesión de cada variante, promediada sobre semillas, para analizar después."""
    columns = {
        "session_time": pa.array(sessions["session_time"], type=pa.int64()).cast(
            pa.timestamp("us", tz="UTC")
        )
    }
    for name, (values, defined) in sessions["series"].items():
        columns[name] = pa.array(np.where(defined, values, np.nan), type=pa.float64())
    pq.write_table(pa.table(columns), path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True, help="Estado de la campaña")
    parser.add_argument("--views", type=Path, required=True, help="Vistas por ventana")
    parser.add_argument("--output", type=Path, required=True, help="Carpeta nueva del informe")
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args(argv)
    declaration = json.loads(args.declaration.read_text(encoding="utf-8"))
    campaign_path = Path(declaration["campaign"])
    if not campaign_path.is_absolute():
        campaign_path = Path(__file__).resolve().parents[3] / campaign_path
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    _require(_integer(args.processes, 1, os.cpu_count() or 1), "Procesos entre 1 y las CPU")
    args.output.mkdir(parents=True)
    report, sessions = analyze(
        declaration, campaign, args.state, args.views, processes=args.processes
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_series(args.output / "sessions.parquet", sessions)
    print(json.dumps(report["means"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
