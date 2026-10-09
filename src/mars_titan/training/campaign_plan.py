"""Declarar y planificar la campaña con máscaras de la edición desde 2000.

La configuración de campaña declara la variante de presupuesto y los entrenadores
conectados. Brazos, semillas, ámbitos, protocolos y ventanas salen de la comparación
walk-forward declarada, de modo que productores y evaluación comparten la misma
definición. El plan enumera cada ajuste y cada predicción trasladada antes de leer
datos y respeta los límites declarados. Este módulo no lee vistas, no reserva la GPU
y no ejecuta ningún ajuste.

Variante A: cada ventana anual se reentrena desde cero. Variante B: se reentrena desde
cero en la primera ventana y cada ``retrain_every_months`` meses. Las ventanas
intermedias se predicen con el estado seleccionado en la última ventana reentrenada,
que dejó de aprender al final de su validación. Cada ventana conserva sus filas, su
purga por intervalo de etiqueta y su calibración común, ajustada con las predicciones
del modelo trasladado en el tramo de calibración de esa ventana.
"""

import json
import math
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.input_policy import masked_inputs
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.evaluation.splits import build_folds, stopping_rule

from .reference_design import PINBALL, QUANTILE_HEAD, candidate_indices, design_cases

CAMPAIGN_KIND = "historical_masked_campaign"
DECLARED = "declared_not_executed"
VARIANTS = ("A", "B")
# Repite reference_run.HELDOUT_FULL_TRAIN_SESSIONS sin importar PyTorch. Una prueba lo fija.
HELDOUT_RETENTION = "heldout_full_train_sessions_v1"
NEURAL_KINDS = ("rnn", "lstm", "gru", "dlinear", "transformer")
TABULAR_KINDS = ("ridge", "xgboost")
NEURAL, TABULAR = "neural_reference", "tabular_reference"
FIT, CARRY = "fit", "carry"

# Familias de la comparación sin entrenador conectado a estas vistas. Cada una se
# conectará con un planificador y un ejecutor propios en este mismo registro.
EXTENSION_POINTS = {
    "episodic_gru": dict(
        issue=383, pending="Entrenador cronológico de la GRU candidata con banco episódico"
    ),
    "titans_mac": dict(
        issue=23,
        pending="Entrenador cronológico de los controles Titans-MAC sin conectar a las vistas v2",
    ),
    "mars_titan": dict(
        issue=366, pending="Ampliaciones de MARS-TITAN sobre el núcleo con sus puntos de inserción"
    ),
    "cm_v1": dict(issue=293, pending="Brazos B, B+C, B+M y B+C+M sobre la B fijada por protocolo"),
}
# Etapas que parten de los padres seleccionados en cada ventana. Solo se declaran.
LATER_STAGES = {
    "posttraining_adapter_matrix": dict(
        config="configs/posttraining/adapter-matrix-v1.json",
        issue=364,
        pending=[
            "ejecución desde la cola",
            "conexión con las ventanas walk-forward",
            "objetivo pinball para padres con cuantiles",
        ],
    )
}
_FIELDS = {
    "schema_version",
    "kind",
    "status",
    "name",
    "variant",
    "retrain_every_months",
    "comparison",
    "scopes",
    "neural",
    "tabular",
    "limits",
    "final_test_opened",
}
_NEURAL = {
    "arms",
    "case_indices",
    "search_seed",
    "head",
    "batch_size",
    "context_sessions",
    "checkpoint_seconds",
    "prediction_retention",
}
_TABULAR = {"config", "arms", "cpu_workers"}
_LIMITS = {"max_training_jobs", "max_prediction_jobs"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _seeds(value, label):
    _require(
        isinstance(value, list)
        and value
        and len(set(value)) == len(value)
        and all(type(seed) is int and 0 <= seed < 2**32 for seed in value),
        f"{label} necesita semillas distintas",
    )
    return value


def _neural(section, arms, rule, policy):
    _require(isinstance(section, dict) and set(section) == _NEURAL, "La sección neuronal no cumple")
    declared = {name for name, arm in arms.items() if arm["family"] == NEURAL}
    mapping = section["arms"]
    _require(
        isinstance(mapping, dict)
        and set(mapping) == declared
        and all(kind in NEURAL_KINDS for kind in mapping.values())
        and len(set(mapping.values())) == len(mapping),
        "Cada brazo neuronal de la comparación necesita una familia de referencia distinta",
    )
    _require(
        section["head"] == QUANTILE_HEAD
        and all(arms[name]["output"] == QUANTILE_HEAD for name in mapping),
        "Los brazos neuronales emiten quantile_head_v1 y se ajustan con su pinball",
    )
    seed = section["search_seed"]
    _require(
        all(seed in _seeds(arms[name]["seeds"], name) for name in mapping),
        "La semilla de búsqueda neuronal debe estar en todos los brazos",
    )
    _require(
        type(section["batch_size"]) is int
        and 1 <= section["batch_size"] <= 256
        and section["context_sessions"] == 64
        and type(section["checkpoint_seconds"]) in (int, float)
        and math.isfinite(section["checkpoint_seconds"])
        and 0 < section["checkpoint_seconds"] <= 900
        and section["prediction_retention"] == HELDOUT_RETENTION
        and masked_inputs(policy),
        "Lote, contexto, checkpoints o retención neuronales no válidos",
    )
    indices = candidate_indices(
        dict(schema_version=4, case_indices=section["case_indices"], models=list(mapping.values()))
    )
    protocol_selection = {key: value for key, value in rule.items() if key != "max_epochs"}
    candidates = {}
    for name, kind in mapping.items():
        design = design_cases(
            [kind],
            seed=seed,
            epochs=rule["max_epochs"],
            patience=rule["patience"],
            min_delta=rule["min_delta"],
            minimum_epochs=rule.get("minimum_epochs"),
            stopping=rule["stopping"],
        )
        candidates[name] = []
        for index in indices:
            # La familia de pérdidas del diseño se sustituye por la pinball de la cabeza común.
            case = design[index]["case"] | dict(head=QUANTILE_HEAD, loss=PINBALL)
            _require(
                case["selection"] == protocol_selection and case["epochs"] == rule["max_epochs"],
                "El caso neuronal no aplica la regla de parada del protocolo",
            )
            candidates[name].append((f"{kind}-{index:02d}", case))
    return dict(section, arms=mapping, seed=seed, candidates=candidates)


def _tabular(section, arms, policy, base):
    from .tabular_search import _configuration

    _require(isinstance(section, dict) and set(section) == _TABULAR, "La sección tabular no cumple")
    path = (base / section["config"]).resolve()
    config, cases, digest = _configuration(path)
    _require(
        config.get("input_policy") == policy,
        "La configuración tabular declara otra política de entradas que la comparación",
    )
    declared = {name for name, arm in arms.items() if arm["family"] == TABULAR}
    mapping = section["arms"]
    _require(
        isinstance(mapping, dict)
        and set(mapping) == declared
        and set(mapping.values()) <= set(TABULAR_KINDS)
        and len(set(mapping.values())) == len(mapping)
        and all(arms[name]["output"] == "point" for name in mapping),
        "Cada brazo tabular de la comparación necesita Ridge o XGBoost con salida puntual",
    )
    seed = config["search_seed"]
    for name, kind in mapping.items():
        seeds = _seeds(arms[name]["seeds"], name)
        _require(
            seeds == [seed]
            if kind == "ridge"
            else sorted(seeds) == sorted(config["finalist_seeds"]),
            f"Las semillas de {name} no coinciden con la configuración tabular",
        )
    _require(
        type(section["cpu_workers"]) is int and 1 <= section["cpu_workers"] <= 8,
        "La concurrencia CPU de los tabulares debe estar entre 1 y 8",
    )
    candidates = {
        name: [(case["id"], case["parameters"]) for case in cases if case["kind"] == kind]
        for name, kind in mapping.items()
    }
    return dict(
        section,
        path=str(path),
        sha256=digest,
        arms=mapping,
        seed=seed,
        candidates=candidates,
        batch_size=config["batch_size"],
    )


def load_campaign(path):
    """Validar la campaña y resolver comparación, protocolos, regla y candidatos."""
    path = Path(path)
    config, digest = read_manifest(path, 1024**2)
    _require(
        isinstance(config, dict)
        and set(config) == _FIELDS
        and config["schema_version"] == 1
        and config["kind"] == CAMPAIGN_KIND
        and config["status"] == DECLARED
        and config["final_test_opened"] is False
        and isinstance(config["name"], str)
        and comparison._name(config["name"].replace("-", "_"))
        and config["variant"] in VARIANTS,
        "La campaña no cumple su contrato",
    )
    base = path.parent
    comparison_path = (base / config["comparison"]).resolve()
    declared = comparison.load_config(comparison_path)
    policy = declared["input_policy"]
    scopes = config["scopes"]
    _require(
        isinstance(scopes, list)
        and scopes
        and len(set(scopes)) == len(scopes)
        and set(scopes) <= set(declared["resolved_scopes"]),
        "Los ámbitos de la campaña deben estar declarados en la comparación",
    )
    rules, steps = set(), set()
    for scope in scopes:
        resolved = declared["resolved_scopes"][scope]
        for protocol in resolved["protocols"].values():
            rules.add(json.dumps(stopping_rule(protocol), sort_keys=True))
            steps.add(protocol["step_months"])
            _require(
                list(resolved["windows"]) == [fold["id"] for fold in build_folds(protocol)],
                f"La campaña recorre todas las ventanas del protocolo de {scope}",
            )
    _require(
        len(rules) == 1 and len(steps) == 1,
        "Todos los protocolos deben declarar la misma regla de parada y el mismo paso",
    )
    rule, step = json.loads(rules.pop()), steps.pop()
    every = config["retrain_every_months"]
    _require(
        type(every) is int
        and step <= every <= 120
        and every % step == 0
        and (every == step) == (config["variant"] == "A"),
        "La variante A reentrena cada ventana y la B cada múltiplo mayor del paso",
    )
    limits = config["limits"]
    _require(
        isinstance(limits, dict)
        and set(limits) == _LIMITS
        and all(type(v) is int and 0 <= v <= 100_000 for v in limits.values()),
        "Los límites de trabajos deben ser enteros declarados",
    )
    arms = declared["arms"]
    families = {arm["family"] for arm in arms.values() if arm["output"] != "zero_control"}
    _require(
        families <= {NEURAL, TABULAR, *EXTENSION_POINTS},
        "La comparación declara familias sin entrenador ni punto de extensión",
    )
    return dict(
        config,
        sha256=digest,
        path=str(path.resolve()),
        comparison_path=str(comparison_path),
        comparison_config=declared,
        input_policy=policy,
        rule=rule,
        step_months=step,
        period=every // step,
        neural=_neural(config["neural"], arms, rule, policy),
        tabular=_tabular(config["tabular"], arms, policy, base),
    )


def schedule(folds, period):
    """Asignar a cada ventana la última ventana reentrenada que no ve su futuro."""
    rows = []
    for index, fold in enumerate(folds):
        anchor = folds[index - index % period]
        # El ancla dejó de aprender al final de su validación, antes de esta calibración.
        _require(
            anchor["validation"][1] <= fold["calibration"][0],
            "Una ventana trasladada vería información posterior a su calibración",
        )
        rows.append(dict(window=fold["id"], anchor=anchor["id"], trained=anchor is fold))
    return rows


def _job(scope, window, arm, family, model, stage, seed, **fields):
    name = {"search": f"search-{fields.get('candidate')}"}.get(stage, f"{stage}-s{seed}")
    return dict(
        id=f"{scope}/{window}/{arm}/{name}",
        scope=scope,
        window=window,
        arm=arm,
        family=family,
        model=model,
        stage=stage,
        kind=CARRY if stage == "carry" else FIT,
        seed=seed,
        candidate=fields.get("candidate"),
        case=fields.get("case"),
        anchor=fields.get("anchor", window),
        depends=fields.get("depends", []),
    )


def _arm_specs(campaign):
    """Brazos con entrenador: familia, modelo, semilla de búsqueda y candidatos."""
    specs = []
    arms = campaign["comparison_config"]["arms"]
    for family, section in ((NEURAL, campaign["neural"]), (TABULAR, campaign["tabular"])):
        for name, kind in section["arms"].items():
            model = "neural" if family == NEURAL else kind
            specs.append(
                dict(
                    arm=name,
                    family=family,
                    model=model,
                    seed=section["seed"],
                    seeds=arms[name]["seeds"],
                    candidates=section["candidates"][name],
                )
            )
    return specs


def plan_campaign(campaign):
    """Enumerar todos los trabajos con sus dependencias sin leer vistas ni datos."""
    jobs = []
    specs = _arm_specs(campaign)
    for scope in campaign["scopes"]:
        folds = list(campaign["comparison_config"]["resolved_scopes"][scope]["windows"].values())
        for row in schedule(folds, campaign["period"]):
            window = row["window"]
            for spec in specs:
                common = (scope, window, spec["arm"], spec["family"], spec["model"])
                prefix = f"{scope}/{row['anchor']}/{spec['arm']}"
                searches = [f"{prefix}/search-{name}" for name, _ in spec["candidates"]]
                if row["trained"]:
                    for name, case in spec["candidates"]:
                        jobs.append(
                            _job(*common, "search", spec["seed"], candidate=name, case=case)
                        )
                    for seed in spec["seeds"]:
                        if seed != spec["seed"]:
                            jobs.append(_job(*common, "finalist", seed, depends=searches))
                    continue
                for seed in spec["seeds"]:
                    depends = searches if seed == spec["seed"] else [f"{prefix}/finalist-s{seed}"]
                    jobs.append(_job(*common, "carry", seed, anchor=row["anchor"], depends=depends))
    _require(len({job["id"] for job in jobs}) == len(jobs), "El plan contiene trabajos repetidos")
    return jobs


def count_jobs(campaign, jobs=None):
    """Contar ajustes y traslados por ámbito, brazo y semilla, y comprobar los límites."""
    jobs = plan_campaign(campaign) if jobs is None else jobs
    scopes = {}
    for scope in campaign["scopes"]:
        folds = list(campaign["comparison_config"]["resolved_scopes"][scope]["windows"].values())
        rows = schedule(folds, campaign["period"])
        selected = [job for job in jobs if job["scope"] == scope]
        arms = {}
        for job in selected:
            seeds = arms.setdefault(job["arm"], {})
            entry = seeds.setdefault(str(job["seed"]), dict(fit=0, carry=0))
            entry[job["kind"]] += 1
        scopes[scope] = dict(
            windows=len(rows),
            retrained_windows=[row["window"] for row in rows if row["trained"]],
            carried_windows=sum(not row["trained"] for row in rows),
            training_jobs=sum(job["kind"] == FIT for job in selected),
            prediction_jobs=sum(job["kind"] == CARRY for job in selected),
            arms=arms,
        )
    totals = dict(
        training_jobs=sum(job["kind"] == FIT for job in jobs),
        prediction_jobs=sum(job["kind"] == CARRY for job in jobs),
    )
    limits = campaign["limits"]
    for kind, limit in (
        ("training_jobs", "max_training_jobs"),
        ("prediction_jobs", "max_prediction_jobs"),
    ):
        if totals[kind] > limits[limit]:
            raise ValueError(
                f"La campaña prevé {totals[kind]} trabajos ({kind}) y supera el límite "
                f"declarado {limit}={limits[limit]}"
            )
    return dict(scopes=scopes, **totals)


def pending_families(campaign):
    """Brazos de la comparación que esperan un entrenador conectado."""
    result = {}
    for name, arm in campaign["comparison_config"]["arms"].items():
        if arm["family"] in EXTENSION_POINTS:
            entry = result.setdefault(arm["family"], dict(EXTENSION_POINTS[arm["family"]], arms=[]))
            entry["arms"].append(name)
    return result


def check_campaign(path):
    """Validar, planificar y presupuestar sin leer datos, reservar la GPU ni entrenar."""
    campaign = load_campaign(path)
    return dict(
        status="checked",
        name=campaign["name"],
        variant=campaign["variant"],
        retrain_every_months=campaign["retrain_every_months"],
        campaign_sha256=campaign["sha256"],
        comparison_sha256=campaign["comparison_config"]["sha256"],
        input_policy=campaign["input_policy"],
        stopping_rule=campaign["rule"],
        neural_loss=dict(head=QUANTILE_HEAD, loss=PINBALL),
        counts=count_jobs(campaign),
        pending_families=pending_families(campaign),
        later_stages=LATER_STAGES,
        scientific_training_started=False,
        final_test_opened=False,
    )
