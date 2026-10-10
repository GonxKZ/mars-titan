"""Declarar y planificar la campaña con máscaras de la edición desde 2000.

La configuración de campaña declara la variante de presupuesto y los entrenadores
conectados. Brazos, semillas, ámbitos, protocolos y ventanas salen de la comparación
walk-forward declarada, de modo que productores y evaluación comparten la misma
definición. El plan enumera cada ajuste y cada predicción trasladada antes de leer
datos y respeta los límites declarados. Este módulo no lee vistas, no reserva la GPU
y no ejecuta ningún ajuste.

Las secciones opcionales ``episodic_gru``, ``titans_mac`` y ``mars_titan`` conectan la GRU
candidata, los controles de Titans-MAC y el lector episódico de MARS-TITAN con sus recetas
cronológicas. Sin ellas, sus brazos siguen declarados como punto de extensión pendiente.
Cada ajuste de MARS-TITAN parte del Titans-MAC ``mac_online`` elegido en la misma ventana y
semilla, así que depende de esos trabajos. La sección ``cm_v1`` conecta el factorial CM-v1:
sus dos núcleos son trabajos auxiliares sin traslado y cada brazo parte de uno de ellos.
`extend_campaign` añade a una campaña cargada las secciones de una declaración preparada con
las mismas reglas, para medir y contar sin cambiar su archivo.

La versión 2 de la configuración declara además la política de semillas, el modo de parada,
las opciones de memoria pendientes, el orden de ejecución, la precisión numérica, que solo
admite FP32 estricto (`campaign_numerics`), y la política de datos, que solo admite la edición
real verificada (`campaign_data_policy`). Con `by_window` el plan recorre
cada ventana de campaña completa antes de la siguiente (`campaign_schedule`). Los brazos de
cada ámbito salen de la comparación: con su diseño conjunto, el ámbito conjunto ajusta todos
los brazos y cada ámbito de un mercado solo los controles separados, con los auxiliares que
necesiten.

Semillas: cada caso de búsqueda se ajusta con la semilla de búsqueda. El caso con menor MAE
por sesión de validación (desempate por identificador) se repite con las demás semillas del
brazo. Esos finalistas dependen de todas las búsquedas de su brazo y ventana y, si el brazo
parte de otro, del finalista del padre con su semilla. Un brazo determinista tiene una sola
semilla y su caso elegido no se repite.

La sección opcional ``early_stop`` declara la parada temprana de la campaña, con la misma
métrica que el protocolo. Sin ella se conserva la regla del protocolo (presupuesto fijo) y
la identidad de cada trabajo. Con ``validation_plateau`` cada ajuste para en su primera
meseta. Con ``joint_plateau`` los brazos de cada grupo emparejado declarado paran en la
misma época, que es la mayor de sus primeras mesetas en el mismo ámbito, ventana, semilla y caso
de búsqueda (o caso elegido, en las semillas finalistas). Cada ajuste agrupado tiene dos
trabajos: el de meseta, que se detiene en su primera meseta en un estado recuperable, y el
final, que depende de las mesetas de todo su grupo y continúa hasta la época común. Los
brazos sin grupo paran en su propia meseta.

En la variante A, cada ventana anual se reentrena desde cero. En la variante B, se reentrena
desde cero en la primera ventana y cada ``retrain_every_months`` meses. Las ventanas
intermedias se predicen con el estado seleccionado en la última ventana reentrenada,
que dejó de aprender al final de su validación. Cada ventana conserva sus filas, su
purga por intervalo de etiqueta y su calibración común, ajustada con las predicciones
del modelo trasladado en el tramo de calibración de esa ventana.
"""

import heapq
import json
import math
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.input_policy import HISTORICAL_MASKED, masked_inputs
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.evaluation.splits import build_folds, stopping_rule

from . import campaign_data_policy, campaign_numerics, campaign_schedule
from .reference_design import PINBALL, QUANTILE_HEAD, candidate_indices, design_cases
from .search_cases import SEARCHED
from .selection import JOINT_PLATEAU, VALIDATION_PLATEAU, campaign_rule

CAMPAIGN_KIND = "historical_masked_campaign"
DECLARED = "declared_not_executed"
VARIANTS = ("A", "B")
# Repite reference_run.HELDOUT_FULL_TRAIN_SESSIONS sin importar PyTorch. Una prueba lo fija.
HELDOUT_RETENTION = "heldout_full_train_sessions_v1"
NEURAL_KINDS = ("rnn", "lstm", "gru", "dlinear", "transformer")
TABULAR_KINDS = ("ridge", "xgboost")
NEURAL, TABULAR = "neural_reference", "tabular_reference"
# Controles de Titans-MAC: brazo de la comparación y variante del entrenador cronológico.
TITANS = "titans_mac"
TITANS_VARIANTS = ("transformer_direct", "mac_disabled", "mac_frozen", "mac_online")
TITANS_RECIPE = "titans_financial_chronological_v1"
# Un caso de búsqueda de las recetas cronológicas solo puede variar estos hiperparámetros.
TITANS_SEARCHED = SEARCHED
FIT, CARRY = "fit", "carry"
# El control en línea parte del estado elegido de otro brazo en su misma ventana. Su brazo
# en la comparación pertenece a la familia ONLINE_CONTROL.
ONLINE = "online"
ONLINE_CONTROL = "online_control"
# Un ajuste con parada conjunta pasa por la meseta individual y por la continuación común.
PLATEAU, JOINT = "plateau", "joint"
EARLY_STOP = "early_stop"
# La época común del grupo es la mayor de las primeras mesetas de sus ajustes.
GROUP_EPOCH = "maximum_of_first_plateaus"
# GRU candidata con banco episódico. Repite candidate_run.RECIPE sin importar PyTorch.
EPISODIC = "episodic_gru"
CANDIDATE_RECIPE = "candidate_gru_chronological_v1"
# Lector episódico de MARS-TITAN sobre Titans-MAC. Repite mars_titan_run.RECIPE y SEARCHED.
MARS = "mars_titan"
MARS_RECIPE = "mars_titan_episodic_readout_chronological_v1"
MARS_SEARCHED = TITANS_SEARCHED
# Escrituras con lector que ajustar. Las demás combinaciones se rechazan al ejecutar.
MARS_BANKS = ("m0_no_bank", "m1", "m2", "m3")
# Factorial CM-v1. Repite los nombres de training.cm_v1_factorial sin importar PyTorch.
CM = "cm_v1"
CM_NAME = "mars_titan_cm_v1_factorial"
CM_CORE_MODEL = "cm_v1_core"
CM_CORES = ("cm_v1_core_b", "cm_v1_core_c")
# Brazo y núcleo del que parte: C cambia el núcleo y M solo la retención del lector.
CM_ARMS = {
    "cm_v1_b": "cm_v1_core_b",
    "cm_v1_bc": "cm_v1_core_c",
    "cm_v1_bm": "cm_v1_core_b",
    "cm_v1_bcm": "cm_v1_core_c",
}
# Familias con sección opcional en la campaña. Sin ella siguen como punto de extensión.
OPTIONAL = (EPISODIC, TITANS, MARS, CM)

# Familias de la comparación sin entrenador conectado a estas vistas. Cada una se
# conectará con un planificador y un ejecutor propios en este mismo registro.
EXTENSION_POINTS = {
    EPISODIC: dict(
        issue=383,
        pending=(
            "La entrada por ventana y la predicción trasladada existen y se conectan con la "
            "sección episodic_gru. Falta declararla en las campañas A y B después de medir "
            "memoria y caudal en cuda:0 y elegir accumulation_rows o recompute. La "
            "declaración preparada está en historical-masked-campaign-extensions.json"
        ),
    ),
    TITANS: dict(
        issue=23,
        pending=(
            "El ajuste por ventana y la predicción trasladada se conectan con la sección "
            "titans_mac y una receta con tantos casos de búsqueda como índices neuronales. "
            "Esta campaña no la declara"
        ),
    ),
    MARS: dict(
        issue=366,
        pending=(
            "El lector por ventana y la predicción trasladada se conectan con la sección "
            "mars_titan sobre el padre titans_mac_online. Falta declararla en las campañas A "
            "y B después de medir memoria y caudal en cuda:0. La declaración preparada está "
            "en historical-masked-campaign-extensions.json"
        ),
    ),
    CM: dict(
        issue=293,
        pending=(
            "Los núcleos y los brazos B, B+C, B+M y B+C+M se conectan con la sección cm_v1 "
            "y su declaración. Falta declararla en las campañas A y B después de medir la "
            "penalización C y el lector en cuda:0. La declaración preparada está en "
            "historical-masked-campaign-extensions.json"
        ),
    ),
    ONLINE_CONTROL: dict(
        issue=443,
        pending=(
            "El ejecutor del control en línea existe y el motor lo registra como trabajo "
            "online, que parte de transformer_compact y usa las etiquetas y el tope del banco "
            "de mars_titan_m1 en la misma ventana y semilla. Sus trabajos y su regla se "
            "declaran en la sección online_controls de la campaña A por etapas. Esta campaña "
            "no la declara"
        ),
    ),
}
# Etapas que parten de los padres seleccionados en cada ventana de una campaña base
# confirmada. Se ejecutan con su propia orden (`run_masked_campaign.py posttraining`).
# `stages` son las de las campañas A y B de tres ámbitos y `joint_stage`, la de la campaña
# A v2 con el modelo conjunto y los controles separados.
LATER_STAGES = {
    # A declara la matriz v3, con los casos de Titans-MAC y la variedad de adaptadores. B
    # conserva la v2 porque no se ejecuta, y sus casos de las redes están en la v3 salvo por
    # la huella.
    "posttraining_adapter_matrix": dict(
        config="configs/posttraining/adapter-matrix-v3.json",
        stages=dict(
            A="configs/posttraining/historical-masked-adapter-stage-a.json",
            B="configs/posttraining/historical-masked-adapter-stage-b.json",
        ),
        joint_stage="configs/posttraining/historical-masked-adapter-stage-a-v2.json",
        entry="mars_titan.posttraining.campaign_stage:run_stage",
        issue=364,
        pending=[],
    ),
    # Políticas financieras por ventana (`run_masked_campaign.py rl`). Las pendientes son
    # capacidades de `simulation.campaign_stage.CAPABILITIES` que el motor aún no tiene.
    "rl_policy_comparison": dict(
        config="configs/simulation/historical-masked-rl-policies.json",
        stages=dict(
            A="configs/simulation/historical-masked-rl-stage-a.json",
            B="configs/simulation/historical-masked-rl-stage-b.json",
        ),
        joint_stage="configs/simulation/historical-masked-rl-stage-a-v2.json",
        entry="mars_titan.simulation.campaign_stage:run_stage",
        issue=137,
        pending=[],
    ),
    # Ablación de modalidades en inferencia (`run_masked_campaign.py ablation`). Vuelve a
    # predecir la evaluación con el estado elegido de cada brazo, sin ajustar nada.
    "modality_ablation": dict(
        config="configs/evaluation/historical-masked-2000-comparison.json",
        stages=dict(
            A="configs/evaluation/historical-masked-ablation-stage-a.json",
            B="configs/evaluation/historical-masked-ablation-stage-b.json",
        ),
        joint_stage="configs/evaluation/historical-masked-ablation-stage-a-v2.json",
        entry="mars_titan.training.modality_ablation_stage:run_stage",
        issue=414,
        pending=[],
    ),
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
_EARLY_STOP = {"stopping", "patience", "min_delta", "max_epochs"}
_EARLY_OPTIONAL = {"minimum_epochs", "groups", "group_epoch"}
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
# Campos que añade la versión 2 de la configuración.
_FIELDS_V2 = {
    "seed_policy",
    "stopping",
    "memory_options",
    "execution",
    "numerics",
    "data_policy",
}
_SEED_POLICY = {"search_seed", "selected_case_seeds", "deterministic_arms"}
# Modos de parada de la versión 2: la regla del protocolo o la sección early_stop, que
# declara la meseta individual o la parada conjunta de los grupos emparejados.
STOPPING_MODES = ("protocol", "early_stop")
PENDING = "pending"
# Familias con opciones de memoria en su receta, que fijan las medidas en cuda:0.
MEMORY_OPTIONS = {
    TITANS: ("accumulation_rows",),
    EPISODIC: ("accumulation_rows", "recompute"),
    CM: ("accumulation_rows",),
}
_TABULAR = {"config", "arms", "cpu_workers"}
_EPISODIC = {"recipe", "arms", "search_seed"}
_TITANS = {"recipe", "arms", "search_seed"}
_MARS = {"recipe", "arms", "pending_arms", "parent_arm", "search_seed"}
_CM = {"declaration", "search_seed"}
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


def _neural(section, arms, rules, policy):
    """Devuelve los casos de las referencias neuronales con la regla de parada de cada brazo."""
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
    candidates = {}
    for name, kind in mapping.items():
        rule = rules(name)
        protocol_selection = {key: value for key, value in rule.items() if key != "max_epochs"}
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


def _stopping(case, rules, arm):
    """Añadir al caso la parada temprana del brazo, si la campaña la declara."""
    rule = None if rules is None else rules(arm)
    return case if rule is None else dict(case, stopping_rule=rule)


def _episodic(section, arms, rule, policy, base, count, rules=None):
    """Declara los brazos de la GRU candidata con su receta, variante y casos sin importar PyTorch.

    Para que la búsqueda sea equitativa, la receta declara tantos casos como índices del
    diseño ajusta cada referencia neuronal (`count`), como Titans-MAC y el lector.
    """
    if section is None:
        return None
    _require(
        isinstance(section, dict) and set(section) == _EPISODIC,
        "La sección de la GRU candidata no cumple",
    )
    path = (base / section["recipe"]).resolve()
    document, digest = read_manifest(path, 64 * 1024)
    _require(
        isinstance(document, dict)
        and all(isinstance(document.get(key), dict) for key in ("model", "recipe", "variants")),
        "La receta de la GRU candidata no conserva su esquema",
    )
    model, mapping, seed = document["model"], section["arms"], section["search_seed"]
    _require(
        document.get("recipe_name") == CANDIDATE_RECIPE
        and model.get("input_policy") == policy
        and model.get("output_head") == QUANTILE_HEAD
        and isinstance(mapping, dict)
        and set(mapping) == {name for name, arm in arms.items() if arm["family"] == EPISODIC}
        and all(
            name == document.get("arm")
            and variant in document["variants"]
            and arms[name]["output"] == QUANTILE_HEAD
            and set(_seeds(arms[name]["seeds"], name)) <= set(model.get("seeds", []))
            and seed in arms[name]["seeds"]
            for name, variant in mapping.items()
        ),
        "Cada brazo de la GRU candidata necesita su receta, una variante y las semillas, "
        "política y cabeza de la comparación",
    )
    selection = {key: value for key, value in rule.items() if key != "max_epochs"}
    cases = (document.get("walk_forward") or {}).get("search_cases")
    _require(
        isinstance(cases, dict)
        and len(cases) == count
        and all(
            isinstance(case, dict) and case and set(case) <= set(SEARCHED)
            for case in cases.values()
        ),
        f"La receta de la GRU candidata necesita {count} casos de búsqueda del optimizador, "
        "tantos como índices ajusta cada referencia neuronal",
    )
    candidates = {}
    for name, variant in mapping.items():
        options = document["recipe"] | document["variants"][variant]
        _require(
            options.get("epochs") == rule["max_epochs"] and options.get("selection") == selection,
            "La receta de la GRU candidata no aplica la regla de parada del protocolo",
        )
        candidates[name] = [
            (
                case,
                _stopping(
                    dict(
                        recipe=str(path),
                        recipe_sha256=digest,
                        variant=variant,
                        seed=seed,
                        search_case=case,
                    ),
                    rules,
                    name,
                ),
            )
            for case in cases
        ]
    return dict(section, path=str(path), sha256=digest, seed=seed, candidates=candidates)


def _titans(section, arms, rule, policy, base, count, rules=None):
    """Brazos de Titans-MAC con su receta común y su control, sin importar PyTorch.

    Para que la búsqueda sea equitativa, la receta declara tantos casos como índices del
    diseño ajusta cada referencia neuronal (`count`), con el mismo presupuesto y la misma
    selección por validación. `training.titans_walk_forward` vuelve a validar la receta
    completa en cada ajuste.
    """
    if section is None:
        return None
    _require(
        isinstance(section, dict) and set(section) == _TITANS and policy == HISTORICAL_MASKED,
        "La sección Titans no cumple o la campaña no usa la política con máscaras",
    )
    path = (base / section["recipe"]).resolve()
    recipe, digest = read_manifest(path, 64 * 1024)
    declared = {name for name, arm in arms.items() if arm["family"] == TITANS}
    mapping = section["arms"]
    _require(
        isinstance(mapping, dict)
        and set(mapping) == declared
        and set(mapping.values()) <= set(TITANS_VARIANTS)
        and len(set(mapping.values())) == len(mapping)
        and all(arms[name]["output"] == QUANTILE_HEAD for name in mapping),
        "Cada brazo Titans de la comparación necesita una variante distinta con cuantiles",
    )
    seed = section["search_seed"]
    _require(
        all(seed in _seeds(arms[name]["seeds"], name) for name in mapping),
        "La semilla de búsqueda de Titans debe estar en todos sus brazos",
    )
    cases = _titans_cases(recipe, rule, count)
    candidates = {
        name: [
            (
                case,
                _stopping(
                    dict(
                        recipe=str(path),
                        recipe_sha256=digest,
                        variant=variant,
                        seed=seed,
                        search_case=case,
                    ),
                    rules,
                    name,
                ),
            )
            for case in cases
        ]
        for name, variant in mapping.items()
    }
    return dict(section, path=str(path), sha256=digest, seed=seed, candidates=candidates)


def _titans_cases(recipe, rule, count):
    """Casos de búsqueda de una receta de Titans-MAC con la cabeza y la regla comunes."""
    selection = {key: value for key, value in rule.items() if key != "max_epochs"}
    _require(
        isinstance(recipe, dict)
        and recipe.get("recipe_name") == TITANS_RECIPE
        and recipe["predictor"].get("head") == QUANTILE_HEAD
        and recipe["recipe"].get("loss") == PINBALL
        and recipe["recipe"].get("epochs") == rule["max_epochs"]
        and recipe["recipe"].get("selection") == selection
        and isinstance(recipe.get("walk_forward"), dict),
        "La receta de Titans no aplica la cabeza común ni la regla de parada del protocolo",
    )
    cases = recipe["walk_forward"].get("search_cases")
    _require(
        isinstance(cases, dict)
        and len(cases) == count
        and all(
            isinstance(case, dict) and case and set(case) <= set(TITANS_SEARCHED)
            for case in cases.values()
        ),
        f"La receta de Titans necesita {count} casos de búsqueda del optimizador, tantos "
        "como índices ajusta cada referencia neuronal",
    )
    return cases


def _mars_titan(section, arms, rule, policy, base, count, titans, rules=None):
    """Brazos de MARS-TITAN: combinación de componentes, receta del lector y padre.

    El padre es un brazo `mac_online` de la sección de Titans-MAC con las mismas semillas.
    Un brazo declarado sin productor queda en `pending_arms` con su motivo. M3 ya tiene
    productor: estima sus escalas con el tramo de entrenamiento de cada ventana.
    `training.mars_titan_walk_forward` valida la combinación completa en cada ajuste.
    """
    if section is None:
        return None
    _require(
        isinstance(section, dict) and set(section) == _MARS and policy == HISTORICAL_MASKED,
        "La sección de MARS-TITAN no cumple o la campaña no usa la política con máscaras",
    )
    parent = section["parent_arm"]
    _require(
        titans is not None and titans["arms"].get(parent) == "mac_online",
        "MARS-TITAN necesita como padre el brazo mac_online de la sección de Titans-MAC",
    )
    path = (base / section["recipe"]).resolve()
    recipe, digest = read_manifest(path, 64 * 1024)
    mapping, pending, seed = section["arms"], section["pending_arms"], section["search_seed"]
    declared = {name for name, arm in arms.items() if arm["family"] == MARS}
    _require(
        isinstance(mapping, dict)
        and isinstance(pending, dict)
        and not set(mapping) & set(pending)
        and set(mapping) | set(pending) == declared
        and all(isinstance(motive, str) and motive for motive in pending.values())
        and all(
            isinstance(components, dict)
            and components.get("episodic_bank") in MARS_BANKS
            and arms[name]["output"] == QUANTILE_HEAD
            for name, components in mapping.items()
        )
        and len({json.dumps(c, sort_keys=True) for c in mapping.values()}) == len(mapping),
        "Cada brazo de MARS-TITAN necesita una combinación distinta con banco episódico y "
        "cuantiles, o un motivo pendiente",
    )
    _require(
        seed == titans["seed"]
        and all(
            seed in _seeds(arms[name]["seeds"], name)
            and set(arms[name]["seeds"]) <= set(arms[parent]["seeds"])
            for name in mapping
        ),
        "Cada semilla de MARS-TITAN necesita su padre Titans-MAC y la misma semilla de búsqueda",
    )
    cases = _readout_cases(recipe, rule, count)
    candidates = {
        name: [
            (
                case,
                _stopping(
                    dict(
                        recipe=str(path),
                        recipe_sha256=digest,
                        components=components,
                        seed=seed,
                        search_case=case,
                        parent_arm=parent,
                    ),
                    rules,
                    name,
                ),
            )
            for case in cases
        ]
        for name, components in mapping.items()
    }
    return dict(section, path=str(path), sha256=digest, seed=seed, candidates=candidates)


def _readout_cases(recipe, rule, count):
    """Casos de búsqueda del lector episódico con la pinball y la regla comunes."""
    selection = {key: value for key, value in rule.items() if key != "max_epochs"}
    cases = (
        (recipe.get("walk_forward") or {}).get("search_cases") if isinstance(recipe, dict) else None
    )
    _require(
        isinstance(recipe, dict)
        and recipe.get("recipe_name") == MARS_RECIPE
        and recipe["recipe"].get("loss") == PINBALL
        and recipe["recipe"].get("epochs") == rule["max_epochs"]
        and recipe["recipe"].get("selection") == selection,
        "La receta del lector no aplica la pinball común ni la regla de parada del protocolo",
    )
    _require(
        isinstance(cases, dict)
        and len(cases) == count
        and all(
            isinstance(case, dict) and case and set(case) <= set(MARS_SEARCHED)
            for case in cases.values()
        ),
        f"La receta del lector necesita {count} casos de búsqueda del optimizador, tantos "
        "como índices ajusta cada referencia neuronal",
    )
    return cases


def _cm_v1(section, arms, rule, policy, base, count, rules=None):
    """Núcleos y brazos del factorial CM-v1 desde su declaración, sin importar PyTorch.

    Los núcleos `cm_v1_core_b` y `cm_v1_core_c` son trabajos auxiliares sin traslado ni
    recibo de ventana: cada brazo parte del núcleo elegido en su ventana y semilla.
    `training.cm_v1_factorial` vuelve a validar declaración, control y retención.
    """
    if section is None:
        return None
    _require(
        isinstance(section, dict) and set(section) == _CM and policy == HISTORICAL_MASKED,
        "La sección de CM-v1 no cumple o la campaña no usa la política con máscaras",
    )
    path = (base / section["declaration"]).resolve()
    document, digest = read_manifest(path, 64 * 1024)
    declared = {name for name, arm in arms.items() if arm["family"] == CM}
    _require(
        isinstance(document, dict)
        and document.get("name") == CM_NAME
        and isinstance(document.get("base"), dict)
        and isinstance(document.get("arms"), dict)
        and set(document["arms"]) == set(CM_ARMS) == declared
        and all(
            document["arms"][arm].get("control") == (CM_ARMS[arm] == CM_CORES[1]) for arm in CM_ARMS
        )
        and document.get("cores") == {core: core == CM_CORES[1] for core in CM_CORES}
        and all(arms[name]["output"] == QUANTILE_HEAD for name in CM_ARMS),
        "La declaración de CM-v1 no corresponde a sus cuatro brazos con cuantiles y sus núcleos",
    )
    seed = section["search_seed"]
    seeds = sorted({s for name in CM_ARMS for s in _seeds(arms[name]["seeds"], name)})
    _require(
        all(seed in arms[name]["seeds"] for name in CM_ARMS),
        "La semilla de búsqueda de CM-v1 debe estar en todos sus brazos",
    )
    recipes = {}
    for name in ("core_recipe", "readout_recipe"):
        value = document["base"].get(name)
        _require(isinstance(value, str), "La declaración de CM-v1 no nombra sus recetas")
        recipes[name] = (path.parent / value).resolve()
    core, core_sha = read_manifest(recipes["core_recipe"], 64 * 1024)
    readout, readout_sha = read_manifest(recipes["readout_recipe"], 64 * 1024)
    # Los dos núcleos comparten receta, también `accumulation_rows`, que la penalización C
    # admite porque su término se descompone por flujos.
    core_cases = _titans_cases(core, rule, count)
    common = dict(declaration=str(path), declaration_sha256=digest, seed=seed)
    candidates = {
        name: [
            (
                case,
                _stopping(
                    dict(
                        common,
                        recipe=str(recipes["core_recipe"]),
                        recipe_sha256=core_sha,
                        core=name,
                        search_case=case,
                    ),
                    rules,
                    name,
                ),
            )
            for case in core_cases
        ]
        for name in CM_CORES
    }
    for name, parent in CM_ARMS.items():
        candidates[name] = [
            (
                case,
                _stopping(
                    dict(
                        common,
                        recipe=str(recipes["readout_recipe"]),
                        recipe_sha256=readout_sha,
                        arm=name,
                        search_case=case,
                        parent_arm=parent,
                    ),
                    rules,
                    name,
                ),
            )
            for case in _readout_cases(readout, rule, count)
        ]
    # Los núcleos van primero: el plan respeta así las dependencias de cada brazo.
    models = {**{name: CM_CORE_MODEL for name in CM_CORES}, **dict.fromkeys(CM_ARMS, CM)}
    return dict(
        section,
        path=str(path),
        sha256=digest,
        seed=seed,
        recipes={name: str(value) for name, value in recipes.items()},
        arms=models,
        candidates=candidates,
        parents=dict(CM_ARMS),
        seeds={name: seeds for name in CM_CORES},
        helpers=CM_CORES,
        outputs=dict.fromkeys(CM_CORES, QUANTILE_HEAD),
    )


def load_campaign(path):
    """Validar la campaña y resolver comparación, protocolos, regla y candidatos."""
    path = Path(path)
    config, digest = read_manifest(path, 1024**2)
    version = config.get("schema_version") if isinstance(config, dict) else None
    fields = _FIELDS | (_FIELDS_V2 if version == 2 else set())
    _require(
        isinstance(config, dict)
        and fields <= set(config) <= fields | set(OPTIONAL) | {EARLY_STOP}
        and version in (1, 2)
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
    early = _early_stop(config.get(EARLY_STOP), rule, declared["arms"])
    every = config["retrain_every_months"]
    _require(
        type(every) is int
        and step <= every <= 120
        and every % step == 0
        and (every == step) == (config["variant"] == "A"),
        "La variante A reentrena cada ventana y la B cada múltiplo mayor del paso",
    )
    _checked_limits(config["limits"])
    arms = declared["arms"]
    families = {arm["family"] for arm in arms.values() if arm["output"] != "zero_control"}
    _require(
        families <= {NEURAL, TABULAR, *EXTENSION_POINTS},
        "La comparación declara familias sin entrenador ni punto de extensión",
    )
    count = len(config["neural"]["case_indices"])
    if version == 2:
        _require(
            config["stopping"] in [{"mode": mode} for mode in STOPPING_MODES]
            and (config["stopping"]["mode"] == EARLY_STOP) == (EARLY_STOP in config),
            "La campaña declara su modo de parada: protocol sin sección early_stop o "
            "early_stop con ella",
        )
        _require(
            isinstance(config["execution"], dict)
            and set(config["execution"]) == {"order"}
            and config["execution"]["order"] in campaign_schedule.ORDERS,
            "La campaña declara su orden de ejecución: by_scope o by_window",
        )
        campaign_numerics.declared(config["numerics"])
        campaign_data_policy.declared(config["data_policy"])
    rules = _arm_rules(rule, early)
    campaign = dict(
        config,
        sha256=digest,
        path=str(path.resolve()),
        comparison_path=str(comparison_path),
        comparison_config=declared,
        input_policy=policy,
        # Esta es la regla común de la campaña. Con parada temprana es la individual o la conjunta.
        rule=rule if early is None else early[early["stopping"]],
        protocol_rule=rule,
        early_stop=early,
        step_months=step,
        period=every // step,
        neural=_neural(config["neural"], arms, lambda arm: rules(arm) or rule, policy),
        tabular=_tabular(config["tabular"], arms, policy, base),
        **_optional_sections(config, arms, rule, policy, base, count, rules),
    )
    for scope in scopes:
        _arm_specs(campaign, scope)
    if version == 2:
        _seed_policy(config["seed_policy"], campaign)
        campaign["memory_options"] = _memory_options(config["memory_options"], campaign)
    return campaign


def _seed_policy(policy, campaign):
    """Exigir que cada brazo siga la política de semillas declarada."""
    _require(
        isinstance(policy, dict)
        and set(policy) == _SEED_POLICY
        and type(policy["search_seed"]) is int
        and isinstance(policy["deterministic_arms"], list)
        and len(set(policy["deterministic_arms"])) == len(policy["deterministic_arms"]),
        "La política de semillas no cumple su contrato",
    )
    search = policy["search_seed"]
    repeats = _seeds(policy["selected_case_seeds"], "La política de semillas")
    _require(search not in repeats, "Las semillas del caso elegido no repiten la de búsqueda")
    specs = [spec for spec in _arm_specs(campaign) if not spec["helper"]]
    deterministic = set(policy["deterministic_arms"])
    _require(
        deterministic <= {spec["arm"] for spec in specs},
        "Los brazos deterministas deben tener productor en la campaña",
    )
    for spec in specs:
        expected = [search] if spec["arm"] in deterministic else [search, *repeats]
        _require(
            spec["seed"] == search and sorted(spec["seeds"]) == sorted(expected),
            f"{spec['arm']} no sigue la política de semillas: búsqueda con {search} y "
            f"semillas {expected}",
        )


def _memory_options(declared, campaign):
    """Opciones de memoria de cada receta: pendientes o iguales al valor de la receta."""
    families = {family: options for family, options in MEMORY_OPTIONS.items() if campaign[family]}
    _require(
        isinstance(declared, dict)
        and set(declared) == set(families)
        and all(
            isinstance(declared[family], dict) and set(declared[family]) == set(options)
            for family, options in families.items()
        ),
        "La campaña declara las opciones de memoria de cada familia con receta",
    )
    resolved = {}
    for family in families:
        section = campaign[family]
        path = section["recipes"]["core_recipe"] if family == CM else section["path"]
        recipe = read_manifest(Path(path), 64 * 1024)[0]["recipe"]
        for option, value in declared[family].items():
            _require(
                value == PENDING or (option in recipe and recipe[option] == value),
                f"{family}.{option} debe estar pendiente o coincidir con su receta",
            )
        resolved[family] = dict(declared[family])
    return resolved


def launch_blockers(campaign):
    """Motivos que impiden lanzar la campaña aunque su plan sea válido."""
    return [
        f"{family}.{option} sigue pendiente de la medida de memoria en cuda:0"
        for family, options in (campaign.get("memory_options") or {}).items()
        for option, value in options.items()
        if value == PENDING
    ]


def _early_stop(section, rule, arms):
    """Lee la parada temprana que declara la campaña, con la métrica del protocolo.

    Devuelve None sin sección. `validation_plateau` detiene cada ajuste en su primera
    meseta. `joint_plateau` necesita grupos de brazos con entrenador, disjuntos y de al
    menos dos brazos, y la época común `maximum_of_first_plateaus`. Los brazos sin grupo
    usan la meseta individual con los mismos valores.
    """
    if section is None:
        return None
    _require(
        isinstance(section, dict)
        and _EARLY_STOP <= set(section) <= _EARLY_STOP | _EARLY_OPTIONAL
        and section["stopping"] in (VALIDATION_PLATEAU, JOINT_PLATEAU),
        "La parada temprana declara modo individual o conjunto, paciencia, mejora mínima y "
        "máximo de épocas",
    )
    values = {
        key: section[key] for key in ("patience", "min_delta", "minimum_epochs") if key in section
    }
    individual = campaign_rule(
        rule,
        dict(
            metric=rule["metric"],
            stopping=VALIDATION_PLATEAU,
            max_epochs=section["max_epochs"],
            **values,
        ),
    )
    resolved = dict(section, **{VALIDATION_PLATEAU: individual, JOINT_PLATEAU: None}, membership={})
    if section["stopping"] == VALIDATION_PLATEAU:
        _require(
            not {"groups", "group_epoch"} & set(section),
            "La parada individual no declara grupos",
        )
        return resolved
    groups = section.get("groups")
    _require(
        section.get("group_epoch") == GROUP_EPOCH
        and isinstance(groups, dict)
        and groups
        and all(isinstance(name, str) and name for name in groups)
        and all(
            isinstance(members, list)
            and len(members) >= 2
            and len(set(members)) == len(members)
            and all(isinstance(arm, str) for arm in members)
            for members in groups.values()
        ),
        f"La parada conjunta declara grupos de al menos dos brazos y la época común {GROUP_EPOCH}",
    )
    members = [arm for names in groups.values() for arm in names]
    trainable = {
        name
        for name, arm in arms.items()
        if arm["family"] != TABULAR and arm["output"] != "zero_control"
    } | set(CM_CORES)
    _require(
        len(set(members)) == len(members) and set(members) <= trainable,
        "Cada brazo agrupado debe tener entrenador neuronal y pertenecer a un solo grupo",
    )
    resolved[JOINT_PLATEAU] = dict(individual, stopping=JOINT_PLATEAU)
    resolved["membership"] = {arm: group for group, names in groups.items() for arm in names}
    return resolved


def _arm_rules(rule, early):
    """Devuelve la regla de parada de cada brazo, o None si la campaña conserva la del protocolo."""
    if early is None:
        return lambda arm: None
    return lambda arm: early[JOINT_PLATEAU if arm in early["membership"] else VALIDATION_PLATEAU]


def _checked_limits(limits):
    _require(
        isinstance(limits, dict)
        and set(limits) == _LIMITS
        and all(type(v) is int and 0 <= v <= 100_000 for v in limits.values()),
        "Los límites de trabajos deben ser enteros declarados",
    )
    return limits


def _optional_sections(config, arms, rule, policy, base, count, rules, titans=None):
    """Resolver las secciones opcionales. `titans` es la ya resuelta si `config` no la trae.

    `rule` es la regla del protocolo, que declaran las recetas, y `rules(arm)` la parada
    temprana de cada brazo si la campaña la declara.
    """
    titans = _titans(config.get(TITANS), arms, rule, policy, base, count, rules) or titans
    return {
        EPISODIC: _episodic(config.get(EPISODIC), arms, rule, policy, base, count, rules),
        TITANS: titans,
        MARS: _mars_titan(config.get(MARS), arms, rule, policy, base, count, titans, rules),
        CM: _cm_v1(config.get(CM), arms, rule, policy, base, count, rules),
    }


def extend_campaign(campaign, sections, *, limits=None):
    """Añadir a una campaña cargada secciones opcionales que su archivo no declara.

    Cada sección se valida con las mismas reglas que en el archivo y sus rutas relativas
    parten de la carpeta de la campaña, así que copiarla al archivo no cambia su
    significado. Sirve para medir y contar una declaración preparada sin activarla. Cada
    sección añadida queda marcada con `declared_in_campaign=False` y la configuración y su
    huella no cambian. `limits` sustituye los límites declarados.
    """
    _require(
        isinstance(sections, dict)
        and sections
        and set(sections) <= set(OPTIONAL)
        and all(isinstance(section, dict) for section in sections.values()),
        "Solo se añaden secciones opcionales de la campaña",
    )
    repeated = sorted(family for family in sections if campaign.get(family))
    _require(not repeated, f"La campaña ya declara {', '.join(repeated)}")
    resolved = _optional_sections(
        sections,
        campaign["comparison_config"]["arms"],
        campaign["protocol_rule"],
        campaign["input_policy"],
        Path(campaign["path"]).parent,
        len(campaign["neural"]["case_indices"]),
        _arm_rules(campaign["protocol_rule"], campaign["early_stop"]),
        titans=campaign.get(TITANS),
    )
    extended = dict(
        campaign,
        **{family: dict(resolved[family], declared_in_campaign=False) for family in sections},
    )
    if limits is not None:
        extended["limits"] = _checked_limits(limits)
    return extended


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
    # Solo los brazos que parten de otro predictor elegido declaran su padre.
    parent = {"parent": fields["parent"]} if fields.get("parent") else {}
    return dict(
        **parent,
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


def scope_arms(campaign, scope):
    """Brazos que la comparación evalúa en un ámbito con predicciones propias del ámbito."""
    resolved = campaign["comparison_config"]["resolved_scopes"][scope]
    return [
        name
        for name, arm in resolved["arms"].items()
        if arm["output"] != "zero_control" and name not in resolved["borrowed"]
    ]


def _arm_specs(campaign, scope=None):
    """Brazos con entrenador: familia, modelo, semilla de búsqueda y candidatos.

    Con `scope`, solo los brazos que se ajustan en ese ámbito, con los auxiliares de los que
    parten. Un padre que es un brazo comparado debe ajustarse también en el ámbito, porque
    solo los auxiliares se incorporan sin declararlos.
    """
    specs = _all_arm_specs(campaign)
    if scope is None:
        return specs
    wanted = set(scope_arms(campaign, scope))
    helpers = {spec["arm"] for spec in specs if spec["helper"]}
    parents = {spec["parent"] for spec in specs if spec["arm"] in wanted} & helpers
    selected = [spec for spec in specs if spec["arm"] in wanted | parents]
    names = {spec["arm"] for spec in selected}
    missing = sorted(
        f"{spec['arm']} sin {spec['parent']}"
        for spec in selected
        if spec["parent"] and spec["parent"] not in names
    )
    _require(not missing, f"En {scope} faltan padres: {', '.join(missing)}")
    return selected


def _all_arm_specs(campaign):
    specs = []
    arms = campaign["comparison_config"]["arms"]
    sections = [(NEURAL, campaign["neural"]), (TABULAR, campaign["tabular"])]
    sections += [(family, campaign[family]) for family in OPTIONAL if campaign.get(family)]
    for family, section in sections:
        for name, kind in section["arms"].items():
            # En las secciones opcionales, la familia también nombra el modelo del ejecutor.
            model = {NEURAL: "neural", EPISODIC: EPISODIC, TITANS: TITANS, MARS: MARS}.get(
                family, kind
            )
            # Los trabajos auxiliares, como los núcleos de CM-v1, no son brazos comparados.
            seeds = section.get("seeds", {}).get(name) or arms[name]["seeds"]
            specs.append(
                dict(
                    arm=name,
                    family=family,
                    model=model,
                    seed=section["seed"],
                    seeds=seeds,
                    candidates=section["candidates"][name],
                    parent=section.get("parents", {}).get(name, section.get("parent_arm")),
                    helper=name in section.get("helpers", ()),
                )
            )
    return specs


def arm_output(campaign, arm):
    """Salida de un brazo comparado o de un trabajo auxiliar, que emite la de sus brazos."""
    declared = campaign["comparison_config"]["arms"]
    if arm in declared:
        return declared[arm]["output"]
    for family in OPTIONAL:
        outputs = (campaign.get(family) or {}).get("outputs", {})
        if arm in outputs:
            return outputs[arm]
    raise ValueError(f"{arm} no es un brazo ni un trabajo auxiliar de la campaña")


def plan_campaign(campaign):
    """Enumerar todos los trabajos con sus dependencias sin leer vistas ni datos.

    Con `data_policy` declarada, comprueba antes la política de datos de la campaña y de
    sus etapas registradas (`campaign_data_policy`).
    """
    if campaign.get("data_policy") is not None:
        campaign_data_policy.check(campaign, LATER_STAGES)
    jobs = []
    for scope in campaign["scopes"]:
        specs = _arm_specs(campaign, scope)
        names = {spec["arm"]: spec for spec in specs}
        folds = list(campaign["comparison_config"]["resolved_scopes"][scope]["windows"].values())
        for row in schedule(folds, campaign["period"]):
            window = row["window"]
            for spec in specs:
                common = (scope, window, spec["arm"], spec["family"], spec["model"])
                prefix = f"{scope}/{row['anchor']}/{spec['arm']}"
                searches = [f"{prefix}/search-{name}" for name, _ in spec["candidates"]]
                if row["trained"]:
                    # El padre de la búsqueda es el ganador de sus búsquedas en la ventana y
                    # el de cada finalista, su finalista con la misma semilla.
                    parent, above = spec["parent"], []
                    if parent:
                        origin = f"{scope}/{window}/{parent}"
                        above = [f"{origin}/search-{n}" for n, _ in names[parent]["candidates"]]
                    for name, case in spec["candidates"]:
                        jobs.append(
                            _job(
                                *common,
                                "search",
                                spec["seed"],
                                candidate=name,
                                case=case,
                                depends=above,
                                parent=parent,
                            )
                        )
                    for seed in spec["seeds"]:
                        if seed != spec["seed"]:
                            depends = searches + ([f"{origin}/finalist-s{seed}"] if parent else [])
                            jobs.append(
                                _job(*common, "finalist", seed, depends=depends, parent=parent)
                            )
                    continue
                if spec["helper"]:
                    # Un auxiliar solo existe para que su brazo parta de él al ajustar.
                    continue
                for seed in spec["seeds"]:
                    depends = searches if seed == spec["seed"] else [f"{prefix}/finalist-s{seed}"]
                    jobs.append(_job(*common, "carry", seed, anchor=row["anchor"], depends=depends))
    _require(len({job["id"] for job in jobs}) == len(jobs), "El plan contiene trabajos repetidos")
    jobs = _joint_phases(campaign, jobs, {spec["arm"]: spec for spec in _arm_specs(campaign)})
    if execution_order(campaign) == "by_window":
        return campaign_schedule.order_by_window(campaign, jobs)
    return jobs


def _ancestors(arm, specs):
    """Devuelve los brazos de los que parte un brazo siguiendo la cadena de padres."""
    found = []
    while (arm := specs[arm]["parent"]) is not None:
        found.append(arm)
    return found


def _checked_groups(campaign, specs):
    """Comprueba que cada grupo conjunto tiene brazos conectados que se emparejan caso a caso.

    Los brazos de un grupo comparten semilla de búsqueda y número de casos, que se emparejan
    por posición, y ninguno parte de otro del mismo grupo, porque su parada dependería de sí
    misma. Cada contraste emparejado de la comparación (delta o factorial) entre brazos
    conectados sin relación de padre debe quedar dentro de un mismo grupo.
    """
    early = campaign["early_stop"]
    membership = early["membership"]
    for group, members in early["groups"].items():
        connected = [arm for arm in members if arm in specs]
        _require(
            len({specs[arm]["seed"] for arm in connected}) <= 1
            and len({len(specs[arm]["candidates"]) for arm in connected}) <= 1
            and not any(set(_ancestors(arm, specs)) & set(members) for arm in connected),
            f"El grupo {group} necesita la misma semilla y el mismo número de casos, sin "
            "padres dentro",
        )
    for family in campaign["comparison_config"]["comparison"]["families"].values():
        kind = family.get("kind")
        if kind == "delta":
            pairs = [(family["base"], variant) for variant in family["variants"]]
        elif kind == "factorial":
            pairs = [(family["base"], family[key]) for key in ("first", "second", "joint")]
        else:
            continue
        for base, variant in pairs:
            if base not in specs or variant not in specs:
                continue
            if base in _ancestors(variant, specs) or variant in _ancestors(base, specs):
                continue
            _require(
                membership.get(base) is not None
                and membership.get(base) == membership.get(variant),
                f"El contraste {base} frente a {variant} necesita un mismo grupo de parada "
                "conjunta",
            )


def _joint_phases(campaign, jobs, specs):
    """Dividir cada ajuste agrupado en su meseta y su continuación hasta la época común.

    El grupo de un ajuste es el de su brazo en el mismo ámbito, ventana y semilla, con el
    caso de búsqueda en la misma posición o, en las semillas finalistas, con el caso elegido
    de cada brazo. El ajuste final depende de las mesetas de todo su grupo y continúa en la
    carpeta de su meseta. Los trabajos quedan en un orden compatible con sus dependencias.
    """
    early = campaign.get("early_stop")
    if not early or early["stopping"] != JOINT_PLATEAU:
        return jobs
    _checked_groups(campaign, specs)
    membership, result, members = early["membership"], [], {}
    positions = {
        (arm, name): index
        for arm, spec in specs.items()
        for index, (name, _) in enumerate(spec["candidates"])
    }
    for job in jobs:
        group = membership.get(job["arm"])
        if job["kind"] != FIT or group is None:
            result.append(job)
            continue
        head, _, name = job["id"].rpartition("/")
        plateau = dict(job, id=f"{head}/plateau-{name}", phase=PLATEAU)
        final = dict(job, phase=JOINT, plateau=plateau["id"])
        slot = positions[job["arm"], job["candidate"]] if job["stage"] == "search" else "selected"
        key = (job["scope"], job["window"], job["seed"], group, slot)
        members.setdefault(key, []).append(final)
        result += [plateau, final]
    for finals in members.values():
        group = [final["plateau"] for final in finals]
        for final in finals:
            final["joint_group"] = group
            final["depends"] = [*final["depends"], *group]
    return _ordered(result)


def _ordered(jobs):
    """Ordena los trabajos de forma estable para que cada uno vaya después de sus dependencias."""
    position = {job["id"]: index for index, job in enumerate(jobs)}
    waiting = {job["id"]: set(job["depends"]) for job in jobs}
    _require(
        all(depends <= set(position) for depends in waiting.values()),
        "El plan depende de trabajos que no contiene",
    )
    users = {}
    for job in jobs:
        for dependency in job["depends"]:
            users.setdefault(dependency, []).append(job["id"])
    ready = [position[key] for key, depends in waiting.items() if not depends]
    heapq.heapify(ready)
    ordered = []
    while ready:
        job = jobs[heapq.heappop(ready)]
        ordered.append(job)
        for user in users.get(job["id"], ()):
            waiting[user].discard(job["id"])
            if not waiting[user]:
                heapq.heappush(ready, position[user])
    _require(len(ordered) == len(jobs), "El plan contiene dependencias circulares")
    return ordered


def execution_order(campaign):
    """Orden declarado de la campaña. La versión 1 recorre los ámbitos uno tras otro."""
    return (campaign.get("execution") or {"order": "by_scope"})["order"]


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
            # La meseta de un ajuste conjunto es la primera parte del mismo ajuste.
            kind = PLATEAU if job.get("phase") == PLATEAU else job["kind"]
            entry[kind] = entry.get(kind, 0) + 1
        scopes[scope] = dict(
            windows=len(rows),
            retrained_windows=[row["window"] for row in rows if row["trained"]],
            carried_windows=sum(not row["trained"] for row in rows),
            training_jobs=sum(_fit(job) for job in selected),
            prediction_jobs=sum(job["kind"] == CARRY for job in selected),
            arms=arms,
        )
        plateaus = sum(job.get("phase") == PLATEAU for job in selected)
        if plateaus:
            scopes[scope]["plateau_jobs"] = plateaus
    totals = dict(
        training_jobs=sum(_fit(job) for job in jobs),
        prediction_jobs=sum(job["kind"] == CARRY for job in jobs),
    )
    plateaus = sum(job.get("phase") == PLATEAU for job in jobs)
    if plateaus:
        totals["plateau_jobs"] = plateaus
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


def _fit(job):
    """Indica si es un ajuste completo, es decir, si no es la meseta de un ajuste conjunto."""
    return job["kind"] == FIT and job.get("phase") != PLATEAU


def pending_families(campaign):
    """Brazos de la comparación que esperan un entrenador conectado, con su motivo."""
    result = {}
    connected = {spec["arm"] for spec in _arm_specs(campaign)}
    for name, arm in campaign["comparison_config"]["arms"].items():
        family = arm["family"]
        if family in EXTENSION_POINTS and name not in connected:
            entry = result.setdefault(family, dict(EXTENSION_POINTS[family], arms=[]))
            entry["arms"].append(name)
            motive = (campaign.get(family) or {}).get("pending_arms", {}).get(name)
            if motive:
                entry.setdefault("motives", {})[name] = motive
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
        protocol_stopping_rule=campaign["protocol_rule"],
        early_stop=_early_record(campaign["early_stop"]),
        neural_loss=dict(head=QUANTILE_HEAD, loss=PINBALL),
        counts=count_jobs(campaign),
        scope_arms={scope: scope_arms(campaign, scope) for scope in campaign["scopes"]},
        seed_policy=campaign.get("seed_policy"),
        stopping=campaign.get("stopping", {"mode": STOPPING_MODES[0]}),
        execution_order=execution_order(campaign),
        numerics=campaign.get("numerics"),
        data_policy=campaign.get("data_policy"),
        memory_options=campaign.get("memory_options"),
        launch_blockers=launch_blockers(campaign),
        pending_families=pending_families(campaign),
        later_stages=LATER_STAGES,
        scientific_training_started=False,
        final_test_opened=False,
    )


def _early_record(early):
    """Resume la parada temprana declarada. Devuelve None si rige la regla del protocolo."""
    if early is None:
        return None
    return dict(
        stopping=early["stopping"],
        individual_rule=early[VALIDATION_PLATEAU],
        joint_rule=early[JOINT_PLATEAU],
        groups=early.get("groups", {}),
        group_epoch=early.get("group_epoch"),
    )
