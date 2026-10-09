"""Declarar y planificar la etapa de políticas financieras de la campaña con máscaras.

La etapa se declara en dos archivos. Las políticas comunes fijan antes de evaluar la
semilla del predictor, los dos niveles de la comparación, las ventanas de ajuste, el
universo, el entorno, los costes, las semillas, el presupuesto de transiciones, el criterio
de selección de cartera, los brazos aprendidos con su motor, las referencias sin
aprendizaje y el contraste con KLPO como brazo principal. Cada variante nombra su campaña
base, sus ámbitos y sus límites. El plan enumera cada ajuste, traslado y referencia sin
leer datos. Este módulo no lee cintas ni ejecuta ningún ajuste.

El nivel `all_predictors` aplica KLPO y las tres referencias a todos los brazos con
productor en la campaña base, resueltos desde su configuración. Una familia que la campaña
registre más adelante entra así sin cambiar la etapa. El nivel `algorithms` compara las
demás políticas aprendidas solo sobre los predictores que declara, porque cada brazo
aprendido multiplica los ajustes y el contraste principal es KLPO.
"""

import math
from collections import Counter
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.training.campaign_plan import DECLARED, _arm_specs, load_campaign

from . import window_tapes
from .environment import ACTIONS

STAGE_KIND = "historical_masked_rl_stage"
POLICIES_KIND = "historical_masked_rl_policies"
FIT, CARRY, REFERENCE = "fit", "carry", "reference"
ALL_PREDICTORS, ALGORITHMS = "all_predictors", "algorithms"
# El nivel completo se resuelve con todos los brazos con productor de la campaña base.
CAMPAIGN_PRODUCERS = "campaign_producers"
SEEDS = [42, 43, 44]
# Referencias sin aprendizaje de `simulation.evaluation.fixed_policy`.
REFERENCES = ("cash", "hold_initial", "rebalance_50")
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
    "environment",
    "evaluation_costs_bps",
    "seeds",
    "budget",
    "selection",
    "hyperparameters",
    "policies",
    "references",
    "contrasts",
    "final_test_opened",
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
    objective = entry.get("policy_objective")
    _require(
        engine == "native_ppo"
        and set(entry) == {"engine", "variant", "policy_objective"}
        and entry["variant"] in ("ppo", "double_dqn"),
        f"{name} necesita un motor y una variante declarados",
    )
    if entry["variant"] == "double_dqn":
        _require(objective is None, "Double DQN conserva su identidad sin objetivo PPO")
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
        and config["schema_version"] == 1
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
        and costs
        and len(set(costs)) == len(costs)
        and all(_number(cost, 0, 1000) for cost in costs)
        and environment["cost_bps"] in costs,
        "El entorno y los costes de evaluación deben declararse antes de evaluar",
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
        "Los brazos aprendidos y las tres referencias deben ser distintos",
    )
    engines = {name: _policy(name, entry) for name, entry in policies.items()}
    contrasts = config["contrasts"]
    primary = [name for name, engine in engines.items() if engine == "native_klpo"]
    _require(
        isinstance(contrasts, dict)
        and set(contrasts) == {"primary", "controls"}
        and primary == [contrasts["primary"]] == list(policies)[:1]
        and contrasts["controls"] == [*list(policies)[1:], *references],
        "KLPO es el brazo principal, va primero y se contrasta con todos los demás",
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
        and set(predictor) == {"seed"}
        and isinstance(universe, dict)
        and universe.get("rule") == window_tapes.UNIVERSE_RULE
        and set(universe) == {"rule", "max_assets"}
        and _integer(universe["max_assets"], 1, 4096)
        and _integer(config["train_windows"], 1, 12),
        "El predictor, el universo y las ventanas de ajuste deben estar declarados",
    )
    return dict(config, sha256=digest, path=str(Path(path).resolve()), engines=engines)


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
    scopes = config["scopes"]
    _require(
        isinstance(scopes, list)
        and scopes
        and len(set(scopes)) == len(scopes)
        and scopes == [scope for scope in campaign["scopes"] if scope in scopes],
        "Los ámbitos pertenecen a la campaña y siguen su orden",
    )
    levels = resolve_levels(campaign, policies)
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


def resolve_levels(campaign, policies):
    """Predictores y brazos de cada nivel, con los productores de la campaña en su orden."""
    seed = policies["predictor"]["seed"]
    specs = _arm_specs(campaign)
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


def scope_windows(stage, scope):
    """Ventanas de política de un ámbito con su ancla, con el periodo de la campaña base."""
    campaign = stage["campaign"]
    folds = campaign["comparison_config"]["resolved_scopes"][scope]["windows"]
    rows = window_tapes.policy_windows(list(folds.values()), stage["policies"]["train_windows"])
    return window_tapes.policy_schedule(rows, campaign["period"], folds)


def _arms(stage, predictor):
    """Brazos aprendidos de un predictor con su nivel: KLPO primero y después algoritmos."""
    levels = stage["levels"]
    arms = [(ALL_PREDICTORS, levels[ALL_PREDICTORS]["arms"][0])]
    if predictor in levels[ALGORITHMS]["predictors"]:
        arms += [(ALGORITHMS, arm) for arm in levels[ALGORITHMS]["arms"]]
    return arms


def plan_stage(stage):
    """Enumerar ajustes, traslados y referencias por ámbito, mercado, ventana y predictor."""
    policies = stage["policies"]
    jobs = []
    for scope in stage["scopes"]:
        rows = scope_windows(stage, scope)
        anchors = {row["window"]: row for row in rows}
        markets = stage["campaign"]["comparison_config"]["resolved_scopes"][scope]["markets"]
        for market in markets:
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
                    )
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
                                    depends=[] if row["trained"] else [fitted],
                                )
                            )
                    for reference in policies["references"]:
                        jobs.append(
                            dict(
                                common,
                                id=f"{prefix}/{reference}/reference",
                                level=ALL_PREDICTORS,
                                arm=reference,
                                engine="reference",
                                seed=None,
                                kind=REFERENCE,
                                depends=[],
                            )
                        )
    _require(len({job["id"] for job in jobs}) == len(jobs), "El plan contiene trabajos repetidos")
    return jobs


def count_stage(stage, jobs=None):
    """Contar trabajos por ámbito, brazo y semilla y aplicar los límites declarados."""
    jobs = plan_stage(stage) if jobs is None else jobs
    scopes = {}
    for scope in stage["scopes"]:
        rows = scope_windows(stage, scope)
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
