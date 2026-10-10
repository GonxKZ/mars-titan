"""Declarar y planificar la etapa de políticas financieras de la campaña con máscaras.

La etapa se declara en dos archivos. Las políticas comunes fijan antes de evaluar la
semilla del predictor, los dos niveles de la comparación, las ventanas de ajuste, el
universo, el entorno, los costes, las semillas, el presupuesto de transiciones, el criterio
de selección de cartera, los brazos aprendidos con su motor, las referencias sin
aprendizaje, el contraste con KLPO como brazo principal, el informe financiero y la
sensibilidad secundaria a la supervivencia. Cada variante nombra su campaña base, sus
ámbitos y sus límites. El plan enumera cada ajuste, traslado y referencia sin leer datos.
Este módulo no lee cintas ni ejecuta ningún ajuste.

La versión 2 añade dos referencias que no usan predicciones: la cartera 1/N reequilibrada
cada 21 sesiones y el índice de mercado comprado y mantenido. El índice se ejecuta en el
motor con el instrumento que declara cada mercado (SPY en EE. UU.). Un mercado sin
instrumento en la edición no planifica ese trabajo y el informe lo sustituye por el
benchmark de niveles declarado (el CSI 300 en China, `simulation.index_benchmark`).

El nivel `all_predictors` aplica KLPO y las referencias a todos los brazos con
productor en la campaña base, resueltos desde su configuración. Una familia que la campaña
registre más adelante entra así sin cambiar la etapa. El nivel `algorithms` compara las
demás políticas aprendidas solo sobre los predictores que declara, porque cada brazo
aprendido multiplica los ajustes y el contraste principal es KLPO.

Un ámbito con varios mercados, como el conjunto US+CN, monta una cinta por mercado con las
predicciones del modelo conjunto en ese mercado. Cada mercado solo recorre las ventanas en
las que la comparación lo declara elegible, así que China empieza con su propia historia
mínima aunque el modelo conjunto se ajuste con todo su pasado.
"""

import math
import re
from collections import Counter
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.posttraining.staged_chain import chain_job_id
from mars_titan.training.campaign_chain import policy_rule
from mars_titan.training.campaign_plan import DECLARED, _arm_specs, load_campaign, scope_arms

from . import window_tapes
from .environment import ACTIONS
from .evaluation import REFERENCE_ALLOCATIONS
from .index_benchmark import BENCHMARKS
from .market import CURRENCIES
from .reconstructed_tape import EDITION_KIND

STAGE_KIND = "historical_masked_rl_stage"
POLICIES_KIND = "historical_masked_rl_policies"
FIT, CARRY, REFERENCE = "fit", "carry", "reference"
ALL_PREDICTORS, ALGORITHMS = "all_predictors", "algorithms"
# El nivel completo se resuelve con todos los brazos con productor de la campaña base.
CAMPAIGN_PRODUCERS = "campaign_producers"
# Contrastes de componente: cada variante aprendida frente a la política aprendida que permite
# descartarla, como Dr. GRPO y GSPO frente a GRPO.
COMPONENTS = "components"
SEEDS = [42, 43, 44]
# Referencias sin aprendizaje de `simulation.evaluation.fixed_policy`.
REFERENCES = tuple(REFERENCE_ALLOCATIONS)
MARKET_INDEX = "market_index"
POLICIES_SCHEMA = 2
# Límite de costes de evaluación que acepta el motor nativo (`frozen_costs`).
MAX_EVALUATION_COSTS = 16
SURVIVAL_RULE = "universe_assets_whose_series_ends_in_evaluation"
# La ventana de ajuste alternativa es una sensibilidad secundaria. Queda declarada con su
# identidad y su coste, pero solo se lanza si sobra presupuesto y alguien la activa.
WINDOW_SENSITIVITY_LAUNCH = "only_if_budget_remains"
# Criterios de cartera del ejecutor nativo. Ninguno usa el error del predictor.
SELECTION_METRICS = (
    "ruin_count_then_mean_liquidated_log_growth",
    "ruin_count_then_mean_log_growth",
)
PPO_OBJECTIVES = {
    "ppo_clip_full_kl_v1": {"schema_version", "id"},
    "ppo_clip_kl_epoch_stop_v1": {"schema_version", "id", "target_kl"},
    "ppo_kl_penalty_adaptive_v1": {
        "schema_version",
        "id",
        "target_kl",
        "beta_initial",
        "beta_min",
        "beta_max",
    },
}
KLPO = dict(objective="klpo_terminal_token_full_v1", controller="klpo_full_fresh_waves_v1")
# Las políticas solo aprenden, se seleccionan y se evalúan con cintas reales de la edición.
DATA_POLICY = "real_edition_only"
# Predicciones que llevan las cintas: las del predictor elegido en la campaña base o las del
# predictor de la cadena, el estado que el posentrenamiento elige con la validación.
BASE_SELECTED = "base_campaign_selected_v1"
CHAIN = "posttraining_chain_v1"
PREDICTOR_SOURCES = (BASE_SELECTED, CHAIN)
# Variantes de valor de mars-titan-ppo. Las dos cuantílicas fijan en el binario 32 cuantiles y
# actúan con la media (qr_dqn) o con el CVaR inferior al 25 % (qr_dqn_cvar).
VALUE_VARIANTS = ("double_dqn", "qr_dqn", "qr_dqn_cvar")
# Objetivos relativos al grupo de mars-titan-klpo. Recogen las mismas oleadas que KLPO y cada
# identidad fija en el binario las constantes de su artículo.
GROUP_CONTROLLER = "group_relative_fresh_waves_v1"
# Motores que consumen oleadas completas de episodios dentro del mismo presupuesto.
WAVE_ENGINES = ("native_klpo", "native_group_relative")
GROUP_OBJECTIVES = (
    "grpo_outcome_v1",
    "dr_grpo_outcome_v1",
    "dapo_outcome_static_v1",
    "gspo_outcome_v1",
)


_STAGE = {
    "schema_version",
    "kind",
    "status",
    "name",
    "campaign",
    "policies",
    "scopes",
    "limits",
    "final_test_opened",
}
_POLICIES = {
    "schema_version",
    "kind",
    "status",
    "predictor",
    "levels",
    "train_windows",
    "universe",
    "data",
    "environment",
    "evaluation_costs_bps",
    "seeds",
    "budget",
    "selection",
    "hyperparameters",
    "policies",
    "references",
    "market_index",
    "report",
    "survival_sensitivity",
    "window_sensitivity",
    "contrasts",
    "final_test_opened",
}
_REPORT = {
    "primary_cost_bps",
    "block_length",
    "block_length_sensitivity",
    "replicates",
    "seed",
    "confidence",
    "benchmarks",
}
_ENVIRONMENT = {
    "capital",
    "cost_bps",
    "participation",
    "score_scale",
    "ruin_penalty",
    "dividend_payment_lag_sessions",
}
_BUDGET = {"transitions", "environments", "rollout_transitions", "evaluation_transitions"}
_SELECTION = {"metric", "partition", "min_delta", "patience", "early_stopping"}
_HYPERPARAMETERS = {
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip",
    "entropy",
    "value_weight",
    "gradient_norm",
    "epochs",
    "minibatch_size",
}
_LIMITS = {"max_training_jobs", "max_evaluation_jobs"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, low, high=math.inf):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _policy(name, entry):
    """Validar un brazo aprendido y devolver su motor."""
    _require(comparison._name(name) and isinstance(entry, dict), f"El brazo {name} no es válido")
    engine = entry.get("engine")
    if engine == "native_klpo":
        _require(
            set(entry) == {"engine", "objective", "controller", "confirmed_updates_per_reference"}
            and {key: entry[key] for key in KLPO} == KLPO
            and _integer(entry["confirmed_updates_per_reference"], 1, 64),
            f"{name} no declara el objetivo y el controlador KLPO terminal",
        )
        return engine
    if engine == "native_group_relative":
        _require(
            set(entry) == {"engine", "objective", "controller", "confirmed_updates_per_reference"}
            and entry["objective"] in GROUP_OBJECTIVES
            and entry["controller"] == GROUP_CONTROLLER
            and _integer(entry["confirmed_updates_per_reference"], 1, 64),
            f"{name} no declara un objetivo de grupo y su controlador de oleadas",
        )
        return engine
    objective = entry.get("policy_objective")
    _require(
        engine == "native_ppo"
        and set(entry) == {"engine", "variant", "policy_objective"}
        and entry["variant"] in ("ppo", *VALUE_VARIANTS),
        f"{name} necesita un motor y una variante declarados",
    )
    if entry["variant"] in VALUE_VARIANTS:
        _require(objective is None, f"{entry['variant']} conserva su identidad sin objetivo PPO")
        return engine
    _require(
        isinstance(objective, dict)
        and objective.get("schema_version") == 1
        and set(objective) == PPO_OBJECTIVES.get(objective.get("id"))
        and all(
            _number(value, 0) and value > 0
            for key, value in objective.items()
            if key not in ("schema_version", "id")
        ),
        f"{name} necesita un objetivo PPO identificado con parámetros positivos",
    )
    if "beta_initial" in objective:
        _require(
            objective["beta_min"] <= objective["beta_initial"] <= objective["beta_max"],
            f"{name} declara una beta inicial fuera de sus límites",
        )
    return engine


def _read_policies(path):
    """Validar brazos, referencias, presupuesto, entorno, selección y contraste."""
    config, digest = read_manifest(path, 1024**2)
    _require(
        isinstance(config, dict)
        and set(config) == _POLICIES
        and config["schema_version"] == POLICIES_SCHEMA
        and config["kind"] == POLICIES_KIND
        and config["status"] == DECLARED
        and config["final_test_opened"] is False
        and config["seeds"] == SEEDS
        and len(ACTIONS) == 6,
        "Las políticas no cumplen su contrato, sus semillas o sus seis acciones",
    )
    environment, budget = config["environment"], config["budget"]
    costs = config["evaluation_costs_bps"]
    _require(
        isinstance(environment, dict)
        and set(environment) == _ENVIRONMENT
        and _number(environment["capital"], 1)
        and _number(environment["cost_bps"], 0, 1000)
        and _number(environment["participation"], 0, 1)
        and environment["participation"] > 0
        and _number(environment["score_scale"], 0)
        and environment["score_scale"] > 0
        and _number(environment["ruin_penalty"], -1e6, 0)
        and environment["ruin_penalty"] < 0
        and _integer(environment["dividend_payment_lag_sessions"], 0, 252)
        and isinstance(costs, list)
        and 1 <= len(costs) <= MAX_EVALUATION_COSTS
        and all(_number(cost, 0, 1000) for cost in costs)
        and all(a < b for a, b in zip(costs, costs[1:], strict=False))
        and environment["cost_bps"] in costs,
        "El entorno y los costes de evaluación, crecientes, deben declararse antes de evaluar",
    )
    _require(
        isinstance(budget, dict)
        and set(budget) == _BUDGET
        and _integer(budget["transitions"], 1, 2**31)
        and _integer(budget["environments"], 1, 4096)
        and _integer(budget["rollout_transitions"], 1, budget["transitions"])
        and _integer(budget["evaluation_transitions"], 1, budget["transitions"]),
        "El presupuesto de transiciones debe ser entero y estar fijado",
    )
    selection = config["selection"]
    _require(
        isinstance(selection, dict)
        and set(selection) == _SELECTION
        and selection["metric"] in SELECTION_METRICS
        and selection["partition"] == "validation"
        and _number(selection["min_delta"], 0)
        and _integer(selection["patience"], 1, 1000)
        and selection["early_stopping"] is False,
        "La selección usa un criterio de cartera en validación, con presupuesto fijo",
    )
    hyper = config["hyperparameters"]
    _require(
        isinstance(hyper, dict)
        and set(hyper) == _HYPERPARAMETERS
        and all(
            _number(hyper[key], 0, 1) for key in _HYPERPARAMETERS - {"epochs", "minibatch_size"}
        )
        and _integer(hyper["epochs"], 1, 32)
        and _integer(hyper["minibatch_size"], 1, budget["rollout_transitions"]),
        "Los hiperparámetros comunes no están acotados",
    )
    policies, references = config["policies"], config["references"]
    _require(
        isinstance(policies, dict)
        and policies
        and isinstance(references, list)
        and sorted(references) == sorted(REFERENCES)
        and len(set(references)) == len(references)
        and not set(policies) & set(REFERENCES),
        "Los brazos aprendidos y las cinco referencias deben ser distintos",
    )
    engines = {name: _policy(name, entry) for name, entry in policies.items()}
    contrasts = config["contrasts"]
    primary = [name for name, engine in engines.items() if engine == "native_klpo"]
    _require(
        isinstance(contrasts, dict)
        and set(contrasts) - {COMPONENTS} == {"primary", "controls"}
        and primary == [contrasts["primary"]] == list(policies)[:1]
        and contrasts["controls"] == [*list(policies)[1:], *references],
        "KLPO es el brazo principal, va primero y se contrasta con todos los demás",
    )
    components, learned = contrasts.get(COMPONENTS, {}), list(policies)[1:]
    _require(
        isinstance(components, dict)
        and (COMPONENTS not in contrasts or components)
        and set(components) <= set(learned)
        and set(components.values()) <= set(learned)
        and not set(components.values()) & set(components),
        "Cada contraste de componente enfrenta una política aprendida distinta de KLPO con "
        "su control, otra política aprendida que no es a su vez una variante",
    )
    levels = config["levels"]
    _require(
        isinstance(levels, dict)
        and set(levels) == {ALL_PREDICTORS, ALGORITHMS}
        and all(isinstance(v, dict) and set(v) == {"predictors", "arms"} for v in levels.values())
        and levels[ALL_PREDICTORS]["predictors"] == CAMPAIGN_PRODUCERS
        and levels[ALL_PREDICTORS]["arms"] == [contrasts["primary"], *references]
        and levels[ALGORITHMS]["arms"] == list(policies)[1:]
        and isinstance(levels[ALGORITHMS]["predictors"], list)
        and levels[ALGORITHMS]["predictors"]
        and len(set(levels[ALGORITHMS]["predictors"])) == len(levels[ALGORITHMS]["predictors"]),
        "El nivel completo aplica KLPO y las referencias a todos los predictores y el de "
        "algoritmos, las demás políticas a predictores declarados",
    )
    predictor, universe = config["predictor"], config["universe"]
    _require(
        isinstance(predictor, dict)
        and set(predictor) == {"seed", "source"}
        and predictor["source"] in PREDICTOR_SOURCES
        and isinstance(universe, dict)
        and universe.get("rule") == window_tapes.UNIVERSE_RULE
        and set(universe) == {"rule", "max_assets"}
        and _integer(universe["max_assets"], 1, 4096),
        "El predictor, el universo y las ventanas de ajuste deben estar declarados",
    )
    # KLPO asigna a cada entorno una cinta de ajuste fija en todas sus oleadas, de modo que
    # ninguna política puede ajustarse con más ventanas que entornos.
    _require(
        window_tapes.train_rule(config["train_windows"])[2] <= budget["environments"],
        "Cada entorno recorre una sola cinta de ajuste: el máximo de ventanas de ajuste no "
        "supera los entornos",
    )
    rule, window = window_tapes.train_rule(config["train_windows"]), config["window_sensitivity"]
    _require(
        isinstance(window, dict)
        and set(window) == {"id", "role", "enabled", "launch", "train_windows"}
        and isinstance(window["id"], str)
        and re.fullmatch(r"[a-z0-9_]+_v[0-9]+", window["id"]) is not None
        and window["role"] == "secondary"
        and type(window["enabled"]) is bool
        and window["launch"] == WINDOW_SENSITIVITY_LAUNCH
        and window_tapes.train_rule(window["train_windows"]) != rule
        and window["train_windows"]["maximum"] <= budget["environments"],
        "La sensibilidad de ventanas es secundaria, tiene identidad propia, otra regla de "
        "ajuste con tantas cintas como entornos como máximo y se lanza solo si se activa",
    )
    # Los objetivos de grupo comparan los episodios que parten de la misma cinta de ajuste, así
    # que cada cinta necesita al menos dos carriles. La sensibilidad comparte los brazos de la
    # etapa principal y por eso también debe cumplirlo, aunque esté desactivada.
    if "native_group_relative" in engines.values():
        _require(
            2 * max(rule[2], window["train_windows"]["maximum"]) <= budget["environments"],
            "Los objetivos de grupo necesitan al menos dos entornos por cinta de ajuste, "
            "también en la sensibilidad de ventanas",
        )
    data = config["data"]
    _require(
        isinstance(data, dict)
        and set(data) == {"policy", "edition", "edition_id"}
        and data["policy"] == DATA_POLICY
        and data["edition"] == EDITION_KIND
        and isinstance(data["edition_id"], str)
        and re.fullmatch(r"[a-f0-9]{64}", data["edition_id"]) is not None,
        "Las políticas aprenden solo con la edición real de precios reconstruidos declarada",
    )
    _read_report(config)
    return dict(config, sha256=digest, path=str(Path(path).resolve()), engines=engines)


def _read_report(config):
    """Validar el índice de mercado, el informe financiero y la sensibilidad de supervivencia."""
    index, report = config["market_index"], config["report"]
    _require(
        isinstance(index, dict)
        and set(index) <= set(CURRENCIES)
        and all(
            isinstance(symbol, str) and symbol.isascii() and symbol.isalnum() and len(symbol) <= 16
            for symbol in index.values()
        ),
        "El índice de mercado declara un instrumento de la edición por mercado",
    )
    sensitivity = report.get("block_length_sensitivity") if isinstance(report, dict) else None
    benchmarks = report.get("benchmarks") if isinstance(report, dict) else None
    _require(
        isinstance(report, dict)
        and set(report) == _REPORT
        and report["primary_cost_bps"] in config["evaluation_costs_bps"]
        and _integer(report["block_length"], 1, 252)
        and isinstance(sensitivity, list)
        and all(_integer(length, 1, 252) for length in sensitivity)
        and report["block_length"] not in sensitivity
        and all(a < b for a, b in zip(sensitivity, sensitivity[1:], strict=False))
        and _integer(report["replicates"], 100, 100_000)
        and _integer(report["seed"], 0, 2**63 - 1)
        and type(report["confidence"]) is float
        and 0.5 <= report["confidence"] < 1
        and isinstance(benchmarks, dict)
        and all(
            BENCHMARKS.get(name, {}).get("market") == market for market, name in benchmarks.items()
        )
        and not set(benchmarks) & set(index),
        "El informe declara coste principal, bootstrap por bloques y benchmarks antes de "
        "ver resultados, con un único índice por mercado",
    )
    survival = config["survival_sensitivity"]
    returns = survival.get("exit_returns") if isinstance(survival, dict) else None
    _require(
        isinstance(survival, dict)
        and set(survival) == {"role", "applies_to", "exit_returns"}
        and survival["role"] == "secondary"
        and survival["applies_to"] == SURVIVAL_RULE
        and isinstance(returns, list)
        and returns
        and all(type(value) is float and -1 <= value <= 0 for value in returns)
        and len(set(returns)) == len(returns),
        "La sensibilidad de supervivencia es secundaria y declara retornos de salida entre -1 y 0",
    )


def load_stage(path):
    """Validar la etapa, su campaña base y sus políticas sin leer datos."""
    path = Path(path)
    config, digest = read_manifest(path, 1024**2)
    _require(
        isinstance(config, dict)
        and set(config) == _STAGE
        and config["schema_version"] == 1
        and config["kind"] == STAGE_KIND
        and config["status"] == DECLARED
        and config["final_test_opened"] is False
        and isinstance(config["name"], str)
        and comparison._name(config["name"].replace("-", "_")),
        "La etapa de refuerzo no cumple su contrato",
    )
    base = path.parent
    campaign = load_campaign((base / config["campaign"]).resolve())
    policies = _read_policies((base / config["policies"]).resolve())
    design = campaign.get("walk_forward_stages")
    # El walk-forward por etapas fija la regla de ventanas de la RL y que lea la cadena. Sin
    # esta comprobación, unas políticas con otra regla se planificarían igual.
    _require(
        design is None
        or (
            policies["train_windows"] == policy_rule(design)
            and policies["predictor"]["source"] == CHAIN
        ),
        "La campaña declara el walk-forward por etapas: las políticas ajustan con la regla "
        "de su RL y leen el predictor de la cadena",
    )
    scopes = config["scopes"]
    _require(
        isinstance(scopes, list)
        and scopes
        and len(set(scopes)) == len(scopes)
        and scopes == [scope for scope in campaign["scopes"] if scope in scopes],
        "Los ámbitos pertenecen a la campaña y siguen su orden",
    )
    levels = resolve_levels(campaign, policies, scopes)
    limits = config["limits"]
    _require(
        isinstance(limits, dict)
        and set(limits) == _LIMITS
        and all(_integer(value, 0, 100_000) for value in limits.values()),
        "Los límites de trabajos deben ser enteros declarados",
    )
    return dict(
        config,
        sha256=digest,
        path=str(path.resolve()),
        campaign=campaign,
        policies=policies,
        levels=levels,
        predictors=levels[ALL_PREDICTORS]["predictors"],
        # Todos los brazos de la campaña predicen las mismas filas de cada ventana. El universo
        # de un ancla es común a todos los predictores y lo fija el primero de algoritmos.
        universe_predictor=levels[ALGORITHMS]["predictors"][0],
    )


def resolve_levels(campaign, policies, scopes=None):
    """Predictores y brazos de cada nivel, con los productores de la campaña en su orden.

    Con `scopes`, solo cuentan los brazos que la campaña ajusta en todos esos ámbitos.
    """
    seed = policies["predictor"]["seed"]
    # Los auxiliares, como los núcleos de CM-v1, no publican recibo de ventana ni predicen.
    planned = [set(scope_arms(campaign, scope)) for scope in scopes or ()]
    specs = [
        spec
        for spec in _arm_specs(campaign)
        if not spec["helper"] and all(spec["arm"] in arms for arms in planned)
    ]
    produced = [spec["arm"] for spec in specs]
    _require(
        produced and all(seed in spec["seeds"] for spec in specs),
        "Cada brazo predictor de la campaña debe tener productor y la semilla declarada",
    )
    declared = policies["levels"]
    algorithms = declared[ALGORITHMS]["predictors"]
    _require(
        set(algorithms) <= set(produced),
        "Los predictores de la comparación de algoritmos deben tener productor en la campaña",
    )
    return {
        ALL_PREDICTORS: dict(predictors=produced, arms=declared[ALL_PREDICTORS]["arms"]),
        ALGORITHMS: dict(
            predictors=[arm for arm in produced if arm in algorithms],
            arms=declared[ALGORITHMS]["arms"],
        ),
    }


def window_sensitivity(stage):
    """Etapa de la sensibilidad de ventanas, con la regla de ajuste alternativa declarada.

    Comparte políticas, presupuesto, semillas y predictores con la etapa principal y solo
    cambia las ventanas de ajuste. Lleva su propia identidad para que sus salidas nunca se
    mezclen con las principales. Mientras la configuración la declare desactivada no se
    puede lanzar, aunque sí contarla.
    """
    entry = stage["policies"]["window_sensitivity"]
    policies = dict(stage["policies"], train_windows=entry["train_windows"])
    return dict(stage, policies=policies, sensitivity=dict(entry))


def count_tapes(stage):
    """Cintas que monta cada predictor por ámbito, sin contar las del índice de mercado.

    Cada ancla monta sus cintas de ajuste y su validación, y cada ventana de política su
    evaluación. Todas se reutilizan entre brazos, semillas y referencias del predictor.
    """
    result = {}
    for scope in stage["scopes"]:
        markets = stage["campaign"]["comparison_config"]["resolved_scopes"][scope]["markets"]
        train = validation = evaluation = 0
        # Cada mercado solo monta cintas en las ventanas en las que es elegible.
        for market in markets:
            rows = scope_windows(stage, scope, market)
            anchors = [row for row in rows if row["trained"]]
            train += sum(len(row["train"]) for row in anchors)
            validation, evaluation = validation + len(anchors), evaluation + len(rows)
        result[scope] = dict(
            train=train,
            validation=validation,
            evaluation=evaluation,
            total=train + validation + evaluation,
        )
    return result


def scope_windows(stage, scope, market=None):
    """Ventanas de política de un ámbito con su ancla, con el periodo de la campaña base.

    Con `market`, solo las ventanas del ámbito en las que ese mercado es elegible.
    """
    campaign = stage["campaign"]
    resolved = campaign["comparison_config"]["resolved_scopes"][scope]
    folds = resolved["windows"]
    if market is not None:
        folds = {
            window: fold for window, fold in folds.items() if window in resolved["eligible"][market]
        }
    rows = window_tapes.policy_windows(list(folds.values()), stage["policies"]["train_windows"])
    return window_tapes.policy_schedule(rows, campaign["period"], folds)


def _arms(stage, predictor):
    """Brazos aprendidos de un predictor con su nivel: KLPO primero y después algoritmos."""
    levels = stage["levels"]
    arms = [(ALL_PREDICTORS, levels[ALL_PREDICTORS]["arms"][0])]
    if predictor in levels[ALGORITHMS]["predictors"]:
        arms += [(ALGORITHMS, arm) for arm in levels[ALGORITHMS]["arms"]]
    return arms


def predictor_reads(stage, job):
    """Ámbito, ventana y predictor de cada evaluación que leen las cintas de un trabajo.

    Un trabajo lee las evaluaciones de ajuste y validación de su ancla y la evaluación de su
    ventana con su predictor, y el universo del ancla con el predictor del universo. Con la
    cadena, cada una exige su selección confirmada en el posentrenamiento. La retención usa
    la misma lista para saber qué tablas de la base sigue necesitando una política.
    """
    read = [*job["train"], job["validation"]]
    reads = {job["predictor"]: [*read, job["window"]]}
    reads.setdefault(stage["universe_predictor"], read)
    return [(job["scope"], window, arm) for arm, windows in reads.items() for window in windows]


def plan_stage(stage):
    """Enumerar ajustes, traslados y referencias por ámbito, mercado, ventana y predictor.

    `depends` empieza por el ajuste del ancla en los traslados y sigue con las selecciones de
    la cadena que confirman las predicciones de sus cintas, si las políticas las declaran.
    """
    policies = stage["policies"]
    jobs = []
    for scope in stage["scopes"]:
        markets = stage["campaign"]["comparison_config"]["resolved_scopes"][scope]["markets"]
        for market in markets:
            rows = scope_windows(stage, scope, market)
            anchors = {row["window"]: row for row in rows}
            for row in rows:
                anchor = anchors[row["anchor"]]
                for predictor in stage["predictors"]:
                    prefix = f"{scope}/{market}/{row['window']}/{predictor}"
                    common = dict(
                        scope=scope,
                        market=market,
                        window=row["window"],
                        anchor=row["anchor"],
                        train=anchor["train"],
                        validation=anchor["validation"],
                        predictor=predictor,
                        predictor_seed=policies["predictor"]["seed"],
                    )
                    chain = [
                        chain_job_id(*read, policies["predictor"]["seed"])
                        for read in predictor_reads(stage, common)
                        if policies["predictor"]["source"] == CHAIN
                    ]
                    for level, arm in _arms(stage, predictor):
                        engine = policies["engines"][arm]
                        for seed in policies["seeds"]:
                            kind = FIT if row["trained"] else CARRY
                            fitted = (
                                f"{scope}/{market}/{row['anchor']}/{predictor}/{arm}/fit-s{seed}"
                            )
                            jobs.append(
                                dict(
                                    common,
                                    id=f"{prefix}/{arm}/{kind}-s{seed}",
                                    level=level,
                                    arm=arm,
                                    engine=engine,
                                    seed=seed,
                                    kind=kind,
                                    depends=([] if row["trained"] else [fitted]) + chain,
                                )
                            )
                    for reference in policies["references"]:
                        if reference == MARKET_INDEX and market not in policies[MARKET_INDEX]:
                            # Sin instrumento en la edición, el informe usa el benchmark.
                            continue
                        jobs.append(
                            dict(
                                common,
                                id=f"{prefix}/{reference}/reference",
                                level=ALL_PREDICTORS,
                                arm=reference,
                                engine="reference",
                                seed=None,
                                kind=REFERENCE,
                                depends=list(chain),
                            )
                        )
    _require(len({job["id"] for job in jobs}) == len(jobs), "El plan contiene trabajos repetidos")
    return jobs


def count_stage(stage, jobs=None):
    """Contar trabajos por ámbito, brazo y semilla y aplicar los límites declarados."""
    jobs = plan_stage(stage) if jobs is None else jobs
    scopes = {}
    for scope in stage["scopes"]:
        markets = stage["campaign"]["comparison_config"]["resolved_scopes"][scope]["markets"]
        by_market = {market: scope_windows(stage, scope, market) for market in markets}
        # Las ventanas de un ámbito son las de todos sus mercados, sin repetir y en orden.
        rows = list({row["window"]: row for value in by_market.values() for row in value}.values())
        rows.sort(key=lambda row: row["window"])
        selected = [job for job in jobs if job["scope"] == scope]
        arms = {}
        for job in selected:
            seed = "none" if job["seed"] is None else str(job["seed"])
            entry = arms.setdefault(job["arm"], {}).setdefault(seed, Counter())
            entry[job["kind"]] += 1
        kinds = Counter(job["kind"] for job in selected)
        scopes[scope] = dict(
            windows=[row["window"] for row in rows],
            anchors=[row["window"] for row in rows if row["trained"]],
            carried_windows=sum(not row["trained"] for row in rows),
            **(
                {}
                if len(markets) == 1
                else {"markets": {m: [row["window"] for row in v] for m, v in by_market.items()}}
            ),
            training_jobs=kinds[FIT],
            carried_jobs=kinds[CARRY],
            reference_jobs=kinds[REFERENCE],
            arms={
                arm: {seed: dict(value) for seed, value in seeds.items()}
                for arm, seeds in arms.items()
            },
        )
    costs = len(stage["policies"]["evaluation_costs_bps"])

    def totals_of(selected):
        kinds = Counter(job["kind"] for job in selected)
        return dict(
            training_jobs=kinds[FIT],
            carried_jobs=kinds[CARRY],
            reference_jobs=kinds[REFERENCE],
            evaluation_jobs=kinds[CARRY] + kinds[REFERENCE],
            evaluation_episodes=len(selected) * costs,
        )

    totals = totals_of(jobs)
    levels = {
        level: dict(
            totals_of([job for job in jobs if job["level"] == level]),
            predictors=stage["levels"][level]["predictors"],
            arms=stage["levels"][level]["arms"],
        )
        for level in (ALL_PREDICTORS, ALGORITHMS)
    }
    limits = stage["limits"]
    for kind, limit in (
        ("training_jobs", "max_training_jobs"),
        ("evaluation_jobs", "max_evaluation_jobs"),
    ):
        _require(
            totals[kind] <= limits[limit],
            f"La etapa prevé {totals[kind]} trabajos ({kind}) y supera el límite "
            f"declarado {limit}={limits[limit]}",
        )
    return dict(scopes=scopes, levels=levels, **totals)
