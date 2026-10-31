"""Protocolo de la criba de variantes de #742 y enumeración de sus trabajos (#746).

El protocolo se registra en `configs/experiments/variant-screening-v1.json` antes de abrir
S. Este módulo lo valida, resuelve los modelos de cada entrada del catálogo a partir de su
declaración y enumera los trabajos por origen y semilla. Dos modelos con la misma
definición son un solo trabajo, así que una referencia común se ajusta una vez por origen
y semilla aunque varias entradas la usen como control. Los controles comunes con trabajo, la
regla de momentum y reversión y LightGBM, tienen uno por origen que vale para todas las
semillas. No importa PyTorch ni lee datos.
"""

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from mars_titan.models.baselines import momentum_reversal, same_inputs_lightgbm

from .screening_cohorts import ORIGIN_FOLDS

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "configs/experiments/variant-screening-v1.json"
KIND = "variant_screening_protocol_v1"
CATALOG = tuple(
    f"{group}{number}"
    for group, size in (("A", 4), ("B", 4), ("C", 6), ("D", 2), ("E", 4), ("F", 5), ("G", 3))
    for number in range(1, size + 1)
)
# Ejecutores con un trabajo por origen y semilla. Los de salida se calculan en el análisis
# sobre las predicciones guardadas de su fuente y los demás quedan fuera de la familia.
JOB_RUNNERS = ("fit", "correction", "correction_profile", "reader", "spaced")
RUNNERS = (*JOB_RUNNERS, "output", "diagnostic", "policy", "audit")
METRICS = ("session_ic", "session_mae_calibrated")
# Igual que walk_forward_phases.MAX_FIT_MONTHS, que no se importa porque arrastra PyTorch.
MAX_FIT_MONTHS = 12
# Controles comunes. El cero y la mediana pasada se calculan en el análisis sobre las etiquetas
# de la referencia B. La regla y LightGBM son trabajos de un origen.
COMMON = ("zero", "past_median", "momentum_reversal", "lightgbm")
COMMON_RUNNERS = ("analysis", "analysis", "rule", "tabular")
PAST_MEDIAN = "past_median_v1"
# Un trabajo de origen vale para las cinco semillas: la regla no tiene semilla y LightGBM
# ajusta las cinco en el mismo trabajo para elegir sus rondas en común, con una carpeta por
# semilla. Ninguno usa la GPU.
ORIGIN_RUNNERS = ("rule", "tabular")
SEED_FOLDERS = ("tabular",)
CPU_RUNNERS = ORIGIN_RUNNERS
# Dependencias: padre congelado, variante emparejada y padre del origen anterior.
REQUIRES = ("parent", "after", "continues")
# Modelos de una declaración que solo existen como etapa que continúa desde su padre (F3).
CONTINUATION = "walk_forward_stage_started_from_its_parent"
# Papeles de los contrastes de continuación. Solo tienen sesiones desde el segundo origen y
# forman su propia familia, fuera de la declarada y de sus sensibilidades.
CONTINUATION_ROLES = ("continuation", "continuation_reported")
# Enmienda de la selección registrada el 28 de octubre antes de abrir S. Los ajustes y los
# lectores emiten la época común de sus semillas, elegida con el IC medio de validación.
SELECTION_RULE = "common_epoch_session_ic_v1"
EPOCH_RUNNERS = ("fit", "reader")
# Réplicas de una media de semillas: todos los subconjuntos del tamaño declarado.
EVERY_SUBSET = "every_subset_of_the_screening_seeds"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Arm:
    """Modelo resuelto de una entrada, sin origen ni semilla."""

    entry: str
    name: str
    definition: dict

    @property
    def runner(self):
        return self.definition["runner"]


def checked_protocol(document):
    """Validar la forma del protocolo y sus restricciones antes de enumerar nada."""
    _require(
        isinstance(document, dict)
        and document.get("schema_version") == 1
        and document.get("kind") == KIND
        and document.get("status") == "registered_before_opening_s"
        and document.get("default_enabled") is False
        and document.get("final_test_access") is False,
        "El protocolo no es la criba registrada, desactivada y sin acceso al test final",
    )
    origins = document.get("origins")
    _require(
        isinstance(origins, list)
        and {item.get("year"): item.get("fold") for item in origins} == ORIGIN_FOLDS
        and len(origins) == len(ORIGIN_FOLDS),
        "Los orígenes deben ser los de las cohortes de la criba",
    )
    seeds = document.get("seeds")
    _require(
        isinstance(seeds, list)
        and len(seeds) == 5
        and all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds)
        and len(set(seeds)) == 5,
        "La criba usa cinco semillas distintas",
    )
    fit = document.get("fit", {})
    _require(
        type(fit.get("fit_months")) is int
        and 1 <= fit["fit_months"] <= MAX_FIT_MONTHS
        and type(fit.get("epochs")) is int
        and 1 <= fit["epochs"] <= 30
        and fit.get("stopping") == "common_epoch"
        and fit.get("selection") == SELECTION_RULE
        and isinstance(fit.get("search_case"), str)
        and fit.get("precision") == "fp32_strict",
        "El ajuste necesita meses, épocas con la época común, caso de búsqueda y FP32",
    )
    checked_selection(document.get("selection"), fit["epochs"])
    checked_common(document.get("common_controls"))
    partitions = document.get("partitions", {})
    _require(
        partitions.get("confirmation", {}).get("opened_by_the_screening") is False
        and partitions.get("final_test", {}).get("opened") is False,
        "La criba no abre la confirmación ni el test final",
    )
    inference = document.get("inference", {})
    _require(
        type(inference.get("block_days")) is int
        and inference["block_days"] >= 1
        and all(
            type(days) is int and days >= 1 for days in inference.get("sensitivity_block_days", [])
        )
        and type(inference.get("replicates")) is int
        and 100 <= inference["replicates"] <= 100_000
        and type(inference.get("confidence")) is float
        and 0 < inference["confidence"] < 1,
        "La inferencia necesita bloques, réplicas y confianza válidos",
    )
    catalog = document.get("catalog")
    _require(
        isinstance(catalog, dict) and tuple(catalog) == CATALOG,
        "El catálogo debe contener las entradas de #742 en su orden",
    )
    references = document.get("references")
    _require(
        isinstance(references, dict)
        and references
        and all(
            isinstance(value, dict) and value.get("runner") in JOB_RUNNERS
            for value in references.values()
        ),
        "Las referencias deben declarar un ejecutor con trabajo",
    )
    for name, entry in catalog.items():
        _require(
            isinstance(entry, dict) and entry.get("runner") in RUNNERS,
            f"{name} necesita un ejecutor conocido",
        )
        if "status" not in entry:
            _require(
                entry.get("metric") in METRICS
                and isinstance(entry.get("primary"), list)
                and len(entry["primary"]) == 2,
                f"{name} necesita su métrica y su contraste principal",
            )
    budget = document.get("budget", {})
    _require(
        budget.get("concurrent_gpu_jobs") == 1
        and isinstance(budget.get("max_total_gpu_hours"), (int, float))
        and budget["max_total_gpu_hours"] > 0
        and type(budget.get("max_job_seconds")) is int
        and budget["max_job_seconds"] > 0
        and type(budget.get("max_job_updates")) is int
        and budget["max_job_updates"] > 0,
        "El presupuesto necesita horas totales, segundos por trabajo y una carga GPU",
    )
    return document


def checked_selection(selection, epochs):
    """Validar la regla de la época común, sus versiones y el registro de la enmienda."""
    selection = selection if isinstance(selection, dict) else {}
    versions = selection.get("versions", {})
    amendment = selection.get("amendment", {})
    check = selection.get("rule_check", {})
    _require(
        selection.get("rule") == SELECTION_RULE
        and selection.get("applies_to") == list(EPOCH_RUNNERS)
        and selection.get("metric") == "session_ic"
        and selection.get("partition") == "validation"
        and type(selection.get("smoothing_radius")) is int
        and 0 <= selection["smoothing_radius"] < epochs,
        "La selección necesita la época común con el IC de validación y su radio",
    )
    listed = versions.get("epochs")
    models = versions.get("models")
    _require(
        isinstance(listed, list)
        and all(type(epoch) is int and 1 <= epoch <= epochs for epoch in listed)
        and listed == sorted(set(listed))
        and isinstance(models, list)
        and all(isinstance(model, str) and "/" in model for model in models)
        and len(set(models)) == len(models)
        and versions.get("partitions") == ["calibration", "evaluation"],
        "Las versiones declaran épocas evaluadas, modelos sin repetir y sus dos tramos",
    )
    _require(
        type(check.get("choose_months")) is int
        and type(check.get("measure_months")) is int
        and check["choose_months"] >= 1
        and check["measure_months"] >= 1
        and check.get("partition") == "validation"
        and check.get("device") == "cpu_only",
        "La comprobación de la regla elige y mide con meses de validación, solo en CPU",
    )
    _require(
        amendment.get("registered_before_opening_s") is True
        and amendment.get("previous_rule") == "best_validation_session_mae_within_the_fixed_budget"
        and isinstance(amendment.get("date"), str)
        and isinstance(amendment.get("reason"), str)
        and amendment["reason"].strip(),
        "La enmienda de la selección registra su fecha, su motivo y la regla anterior",
    )


def checked_common(common):
    """Validar los controles comunes y su familia, que se publica fuera de las decisiones."""
    common = common if isinstance(common, dict) else {}
    models = common.get("models")
    _require(
        common.get("registered_before_opening_s") is True
        and isinstance(models, dict)
        and tuple(models) == COMMON
        and all(isinstance(value, dict) for value in models.values())
        and tuple(value.get("runner") for value in models.values()) == COMMON_RUNNERS,
        "Los controles comunes declaran el cero, la mediana pasada, la regla y LightGBM",
    )
    past = models["past_median"]
    _require(
        past.get("identity") == PAST_MEDIAN
        and type(past.get("lag_sessions")) is int
        and past["lag_sessions"] >= 1
        and type(past.get("window")) is int
        and type(past.get("min_labels")) is int
        and 1 <= past["min_labels"] <= past["window"],
        "La mediana pasada declara su retraso, su ventana y su mínimo de etiquetas",
    )
    momentum_reversal.checked_config(models["momentum_reversal"].get("config"))
    same_inputs_lightgbm.checked_config(models["lightgbm"].get("config"))
    family = common.get("family", {})
    pairs = family.get("contrasts")
    _require(
        family.get("role") == "reported_outside_the_decisions"
        and family.get("metric") in METRICS
        and isinstance(pairs, list)
        and pairs
        and all(isinstance(pair, list) and len(pair) == 2 and pair[1] in COMMON for pair in pairs)
        and len({tuple(pair) for pair in pairs}) == len(pairs),
        "La familia de los controles comunes declara su métrica y contrastes frente a ellos",
    )


def load_protocol(root=ROOT):
    root = Path(root)
    return checked_protocol(json.loads((root / PROTOCOL).read_text()))


def _declaration(root, path):
    document = json.loads((Path(root) / path).read_text())
    _require(
        document.get("default_enabled") is False and document.get("final_test_access") is False,
        f"{path} debe estar desactivada por defecto y sin acceso al test final",
    )
    return document


def _fit(entry, document, name, value):
    base = entry.get("base_recipe") or document.get("base_recipe")
    _require(isinstance(base, str), f"{name} necesita una receta base")
    definition = dict(
        runner="fit",
        base_recipe=base,
        head=document.get("base_head"),
        loss=document.get("base_loss"),
        predictor_changes=value.get("predictor_changes") or {},
        recipe_changes=value.get("recipe_changes") or {},
        updates_multiplier=value.get("updates_multiplier", 1),
    )
    _require(
        type(definition["updates_multiplier"]) is int
        and 1 <= definition["updates_multiplier"] <= 4,
        f"{name} multiplica las actualizaciones por un entero entre 1 y 4",
    )
    return definition


def _combination(entry, declared, name, value, seeds):
    """Media de semillas de una entrada, con su fuente y sus réplicas sobre las semillas.

    La media no se ajusta: el análisis combina las emisiones de su fuente. Cada réplica
    promedia `size` semillas de la criba y se toman todos los subconjuntos, así que cada
    semilla aparece el mismo número de veces y el resultado no depende de su orden. La media
    de las réplicas estima el IC esperado de un conjunto de `size` semillas. Su control
    multiplica las actualizaciones por `size`, como declara F2.
    """
    combination = entry.get("combinations", {}).get(name)
    _require(
        isinstance(combination, dict)
        and set(combination) == {"source", "size", "replicates"}
        and combination["replicates"] == EVERY_SUBSET,
        f"{name} combina semillas y declara su fuente, su tamaño y sus réplicas",
    )
    source, size = combination["source"], combination["size"]
    _require(
        source in declared and source != name and "combine" not in declared[source],
        f"La fuente de {name} es otro modelo ajustado de su entrada",
    )
    control = declared.get(value.get("control"), {})
    _require(
        type(size) is int
        and 2 <= size <= len(seeds)
        and size == len(value.get("seeds", ()))
        and control.get("updates_multiplier") == size,
        f"{name} promedia las semillas que declara y su control multiplica por ellas las "
        "actualizaciones",
    )
    replicates = [list(group) for group in itertools.combinations(sorted(seeds), size)]
    return dict(runner="output", source=source, combine=value["combine"], replicates=replicates)


def _continued(name, entry, declared, excluded):
    """Padre de las continuaciones de una entrada, o None si no declara ninguna ejecutable.

    El padre es un modelo ajustado de la misma entrada que no continúa a su vez. Cada
    continuación del origen k empieza en el estado que ese padre eligió en el origen k − 1
    con la misma semilla.
    """
    arms = {arm for arm, value in declared.items() if value.get("applies_to") == CONTINUATION}
    continuation = entry.get("continuation")
    if not arms - excluded:
        _require(continuation is None, f"{name} declara una continuación sin modelos que continúen")
        return None
    _require(
        isinstance(continuation, dict)
        and set(continuation) == {"parent", "primary", "reported", "date", "note"}
        and continuation["parent"] in declared
        and continuation["parent"] not in arms | excluded
        and "combine" not in declared[continuation["parent"]],
        f"{name} continúa desde un padre ajustado de la entrada que no continúa",
    )
    return continuation["parent"]


def _parented(entry, references, runner):
    parent = entry.get("parent")
    _require(parent in references, f"El ejecutor {runner} necesita una referencia como padre")
    return parent


def entry_arms(protocol, name, root=ROOT):
    """Modelos de una entrada del catálogo, desde su declaración o desde el protocolo."""
    entry = protocol["catalog"][name]
    references = protocol["references"]
    runner = entry["runner"]
    if "status" in entry or runner in ("diagnostic", "policy", "audit"):
        return {}
    document = _declaration(root, entry["declaration"]) if "declaration" in entry else {}
    arms = {}
    if runner == "output":
        for arm, value in entry["arms"].items():
            arms[arm] = dict(runner="output", source=entry["source"], **value)
    elif runner == "fit":
        declared = entry.get("arms") or document.get("arms")
        excluded = set(entry.get("pending_arms", ()))
        parent = _continued(name, entry, declared, excluded)
        for arm, value in declared.items():
            if arm in excluded:
                continue
            arms[arm] = (
                _combination(entry, declared, arm, value, protocol["seeds"])
                if "combine" in value
                else _fit(entry, document, arm, value)
            )
            if value.get("applies_to") == CONTINUATION:
                arms[arm]["continues"] = parent
    elif runner == "correction":
        parent = _parented(entry, references, runner)
        sources = dict(document["arms"])
        for alias, value in document.get("references", {}).items():
            # Una referencia textual son las predicciones del padre, que η = 0 reproduce.
            sources[alias] = value if isinstance(value, dict) else None
        for arm, value in sources.items():
            arms[arm] = (
                dict(runner="reference", reference=parent)
                if value is None
                else dict(
                    runner="correction",
                    recipe=value["recipe"],
                    components=value["components"],
                    parent=parent,
                )
            )
    elif runner == "correction_profile":
        parent = _parented(entry, references, runner)
        base = document["arguments"]
        for arm, value in document["arms"].items():
            # Sin perfil, la ventana es la corrección de la misma receta y el mismo padre.
            arms[arm] = dict(
                runner="correction",
                recipe=base["recipe"],
                components=base["components"],
                parent=parent,
            )
            if isinstance(value, dict) and "control_of" in value:
                # El control emparejado lee la ventana de su variante en la misma semilla.
                _require(value["control_of"] in document["arms"], f"{arm} no tiene variante")
                value = dict(matched_to=value["control_of"], builder=value["builder"])
            if value is not None:
                arms[arm] |= dict(
                    runner="correction_profile", argument=entry["argument"], value=value
                )
    elif runner == "reader":
        parent = _parented(entry, references, runner)
        for arm, value in document["arms"].items():
            arms[arm] = dict(
                runner="reader",
                recipe=entry["recipe"],
                components=value["components"],
                recipe_changes=value.get("recipe_changes") or {},
                parent=parent,
            )
    else:
        parent = _parented(entry, references, runner)
        for arm in document["arms"]:
            arms[arm] = dict(runner="spaced", study=entry["declaration"], arm=arm, parent=parent)
    return {arm: Arm(name, arm, definition) for arm, definition in arms.items()}


def reference_arms(protocol):
    arms = {}
    for name, value in protocol["references"].items():
        definition = {key: item for key, item in value.items() if key != "role"}
        if definition["runner"] == "fit":
            definition = dict(
                runner="fit",
                base_recipe=definition["base_recipe"],
                head=definition.get("head"),
                loss=definition.get("loss"),
                predictor_changes=definition.get("predictor_changes") or {},
                recipe_changes=definition.get("recipe_changes") or {},
                updates_multiplier=1,
            )
        arms[name] = Arm("references", name, definition)
    return arms


def common_arms(protocol):
    """Controles comunes con trabajo, la regla y LightGBM, con su declaración."""
    models = protocol["common_controls"]["models"]
    return {
        name: Arm("common", name, dict(runner=value["runner"], config=value["config"]))
        for name, value in models.items()
        if value["runner"] in ORIGIN_RUNNERS
    }


def resolve(protocol, name, endpoint, root=ROOT):
    """Modelo al que se refiere un extremo de un contraste de la entrada `name`."""
    _require(isinstance(endpoint, str) and endpoint, "Cada extremo es un nombre")
    if endpoint in COMMON:
        return ("common", endpoint)
    if endpoint in protocol["references"]:
        return ("references", endpoint)
    entry, _, arm = endpoint.rpartition(".")
    entry = entry or name
    _require(
        arm in entry_arms(protocol, entry, root),
        f"{name} usa {endpoint}, que no es un modelo de {entry}",
    )
    return (entry, arm)


def contrasts(protocol, root=ROOT):
    """Contrastes declarados con su papel: principal, exigido o solo publicado.

    Los de una continuación llevan los papeles de `CONTINUATION_ROLES`. Sus contrastes
    principales comparan dos continuaciones, así que los dos extremos empiezan en el mismo
    origen, y cada contraste publicado incluye al menos una.
    """
    rows = []
    for name, entry in protocol["catalog"].items():
        if "status" in entry:
            continue
        pairs = [("primary", entry["primary"])]
        pairs += [("required", pair) for pair in entry.get("required", ())]
        pairs += [("reported", pair) for pair in entry.get("reported", ())]
        continuation = entry.get("continuation")
        if continuation is not None:
            _require(
                continuation["primary"] and all(len(pair) == 2 for pair in continuation["primary"]),
                f"{name} contrasta sus continuaciones con al menos un par",
            )
            pairs += [(CONTINUATION_ROLES[0], pair) for pair in continuation["primary"]]
            pairs += [(CONTINUATION_ROLES[1], pair) for pair in continuation["reported"]]
        arms = entry_arms(protocol, name, root)
        for role, (variant, control) in pairs:
            row = dict(
                entry=name,
                role=role,
                metric=entry["metric"],
                variant=resolve(protocol, name, variant, root),
                control=resolve(protocol, name, control, root),
            )
            if role in CONTINUATION_ROLES:
                continued = [
                    side[0] == name and "continues" in arms[side[1]].definition
                    for side in (row["variant"], row["control"])
                ]
                _require(
                    all(continued) if role == CONTINUATION_ROLES[0] else any(continued),
                    f"{name} compara en su familia de continuación modelos que no continúan",
                )
            rows.append(row)
    return rows


def common_contrasts(protocol, root=ROOT):
    """Contrastes de la familia de los controles comunes, que no decide ninguna entrada."""
    family = protocol["common_controls"]["family"]
    return [
        dict(
            entry="common",
            role="reported",
            metric=family["metric"],
            variant=resolve(protocol, "common", variant, root),
            control=resolve(protocol, "common", control, root),
        )
        for variant, control in family["contrasts"]
    ]


def all_arms(protocol, root=ROOT):
    arms = {("references", name): arm for name, arm in reference_arms(protocol).items()}
    arms |= {("common", name): arm for name, arm in common_arms(protocol).items()}
    for name in protocol["catalog"]:
        for arm_name, arm in entry_arms(protocol, name, root).items():
            arms[(name, arm_name)] = arm
    return arms


def _cases(protocol, definition, root):
    """Casos de búsqueda de un modelo: el declarado para ajustes y lectores, todos los de la
    receta para una corrección, que elige el suyo con la validación en el análisis."""
    runner = definition["runner"]
    if runner in ("fit", "reader"):
        return (protocol["fit"]["search_case"],)
    if runner in ("spaced", *ORIGIN_RUNNERS):
        return (None,)
    recipe = json.loads((Path(root) / definition["recipe"]).read_text())
    cases = tuple(recipe.get("walk_forward", {}).get("search_cases", ()))
    _require(cases, f"La receta {definition['recipe']} necesita casos de búsqueda")
    return cases


def _definitions(protocol, arms, arm, root):
    """Definiciones de trabajo de un modelo, una por caso, con la huella de su padre.

    Un control emparejado guarda en su valor la huella de la ventana de su variante en el
    mismo caso, que tiene que terminar antes. Una continuación guarda la huella del modelo
    desde cuyo estado elegido empieza.
    """
    parent = None
    if "parent" in arm.definition:
        (parent,) = _definitions(
            protocol, arms, arms[("references", arm.definition["parent"])], root
        )
    continues = None
    if "continues" in arm.definition:
        (continues,) = _definitions(
            protocol, arms, arms[(arm.entry, arm.definition["continues"])], root
        )
    value = arm.definition.get("value")
    matched = None
    if isinstance(value, dict) and "matched_to" in value:
        sibling = arms[(arm.entry, value["matched_to"])]
        matched = {
            definition.get("search_case"): key
            for key, definition in _definitions(protocol, arms, sibling, root)
        }
    result = []
    for case in _cases(protocol, arm.definition, root):
        definition = dict(arm.definition)
        if case is not None:
            definition["search_case"] = case
        if parent is not None:
            definition["parent"] = parent[0]
        if continues is not None:
            definition["continues"] = continues[0]
        if matched is not None:
            definition["value"] = dict(value, matched_to=matched[case])
        result.append((_digest(definition), definition))
    return result


def combination_sources(arms):
    """Modelos cuyas semillas combina una etapa de salida, con las etapas que los combinan.

    La época de una combinación se elige con el IC de la propia combinación en validación,
    así que los trabajos de su fuente guardan la predicción puntual de cada época.
    """
    sources = {}
    for (entry, _), arm in arms.items():
        if arm.runner == "output" and "combine" in arm.definition:
            source = (entry, arm.definition["source"])
            _require(source in arms, f"La combinación de {entry} no tiene fuente")
            sources.setdefault(source, []).append(arm)
    return sources


def screening_jobs(protocol, root=ROOT):
    """Trabajos únicos por definición, caso, origen y semilla, con su padre y sus usos.

    Los ajustes y los lectores llevan además su grupo de semillas (`common_group`), las
    épocas que emiten como versiones (`versions`) y si guardan la predicción de cada época
    de validación porque una combinación los usa (`epoch_predictions`). Un trabajo de origen
    no tiene semilla (`seed` es None) y declara las que cubre (`seeds`). Una continuación
    lleva en `continues` el trabajo del origen anterior con la misma semilla.
    """
    arms = all_arms(protocol, root)
    combined = {"/".join(label) for label in combination_sources(arms)}
    versioned = set(protocol["selection"]["versions"]["models"])
    epochs = protocol["selection"]["versions"]["epochs"]
    jobs = {}
    for label, arm in arms.items():
        if arm.runner not in (*JOB_RUNNERS, *ORIGIN_RUNNERS):
            continue
        seeds = (None,) if arm.runner in ORIGIN_RUNNERS else protocol["seeds"]
        for key, definition in _definitions(protocol, arms, arm, root):
            for position, origin in enumerate(protocol["origins"]):
                # El primer origen no tiene un estado anterior desde el que continuar.
                if "continues" in definition and position == 0:
                    continue
                for seed in seeds:
                    prefix = str(origin["year"]) if seed is None else f"{origin['year']}-{seed}"
                    job_id = f"{prefix}-{key[:16]}"
                    job = jobs.setdefault(
                        job_id,
                        dict(
                            id=job_id,
                            origin=origin["year"],
                            fold=origin["fold"],
                            seed=seed,
                            definition_sha256=key,
                            definition=definition,
                            parent=None,
                            after=None,
                            continues=None,
                            used_by=[],
                        ),
                    )
                    _require(job["definition_sha256"] == key, "Dos definiciones comparten prefijo")
                    if "parent" in definition:
                        job["parent"] = f"{prefix}-{definition['parent'][:16]}"
                    if (
                        isinstance(definition.get("value"), dict)
                        and "matched_to" in definition["value"]
                    ):
                        job["after"] = f"{prefix}-{definition['value']['matched_to'][:16]}"
                    if "continues" in definition:
                        previous = protocol["origins"][position - 1]["year"]
                        job["continues"] = f"{previous}-{seed}-{definition['continues'][:16]}"
                    job["used_by"].append("/".join(label))
    for job in jobs.values():
        job["used_by"].sort()
        if job["seed"] is None:
            job["seeds"] = list(protocol["seeds"])
        if job["definition"]["runner"] in EPOCH_RUNNERS:
            uses = set(job["used_by"])
            job.update(
                common_group=f"{job['origin']}-{job['definition_sha256'][:16]}",
                versions=list(epochs) if uses & versioned else [],
                epoch_predictions=bool(uses & combined),
            )
    used = {label for job in jobs.values() if "common_group" in job for label in job["used_by"]}
    _require(
        versioned <= used, "Las versiones solo se declaran sobre ajustes o lectores de la criba"
    )
    _require(
        all(
            job[field] is None or job[field] in jobs for job in jobs.values() for field in REQUIRES
        ),
        "Un trabajo depende de otro que no está en la criba",
    )
    depth = {}

    def level(job_id):
        # Profundidad en el grafo de requisitos, que no tiene ciclos por construcción.
        if job_id not in depth:
            job = jobs[job_id]
            depth[job_id] = 1 + max(
                (level(job[field]) for field in REQUIRES if job[field] is not None), default=-1
            )
        return depth[job_id]

    # Las cinco semillas de cada definición van seguidas. Así cada grupo de la época común
    # decide en cuanto termina su última semilla y libera sus estados antes de que empiece
    # el siguiente.
    return sorted(
        jobs.values(),
        key=lambda job: (
            level(job["id"]),
            job["origin"],
            job["definition_sha256"],
            -1 if job["seed"] is None else job["seed"],
        ),
    )


def pending(protocol):
    """Entradas y modelos que la criba declara pero todavía no puede ejecutar."""
    rows = []
    for name, entry in protocol["catalog"].items():
        if "status" in entry:
            rows.append(dict(entry=name, status=entry["status"]))
        for arm in entry.get("pending_arms", ()):
            rows.append(dict(entry=name, arm=arm, status="pending_arm"))
    return rows


def cost_estimate(protocol, jobs, rows):
    """Horas de GPU estimadas con las medidas de los pilotos que declara el protocolo.

    `rows` da por origen las filas de ajuste, validación, calibración y evaluación. Un
    ajuste repite cada época el ajuste y la validación y predice una vez la calibración y
    la evaluación. Una corrección predice una vez los tres tramos medidos con su caso, y un
    lector cuesta lo que declara la medida de referencia. Cada versión de la época común
    predice otra vez la calibración y la evaluación, y la época de una combinación puede
    añadir una más a cada semilla de su fuente. La regla y LightGBM cuestan por origen la
    cota que declara la referencia, que no es una medida, y se cobran en el mismo presupuesto
    aunque no usen la GPU.
    """
    reference = protocol["cost_reference"]
    epochs = protocol["fit"]["epochs"]
    seconds = 0.0
    by_runner = {}
    for job in jobs:
        definition = job["definition"]
        counts = rows[job["origin"]]
        if definition["runner"] == "fit":
            passes = epochs * definition["updates_multiplier"]
            cost = (
                passes
                * (
                    counts["fit"] * reference["fit_seconds_per_fit_row_epoch"]
                    + counts["validation"] / reference["validation_rows_per_second"]
                )
                + (counts["calibration"] + counts["evaluation"])
                / reference["inference_rows_per_second"]
            )
        elif definition["runner"] in ("correction", "correction_profile"):
            measured = counts["validation"] + counts["calibration"] + counts["evaluation"]
            cost = measured / reference["inference_rows_per_second"]
        elif definition["runner"] in ORIGIN_RUNNERS:
            cost = reference[f"{definition['runner']}_window_seconds"]
        else:
            cost = reference["reader_window_seconds"]
        if definition["runner"] in EPOCH_RUNNERS:
            extra = len(job["versions"]) + int(job["epoch_predictions"])
            cost += (
                extra
                * (counts["calibration"] + counts["evaluation"])
                / reference["inference_rows_per_second"]
            )
        seconds += cost
        by_runner[definition["runner"]] = by_runner.get(definition["runner"], 0.0) + cost
    hours = seconds / 3600
    return dict(
        jobs=len(jobs),
        hours=hours,
        hours_by_runner={name: value / 3600 for name, value in sorted(by_runner.items())},
        budget_hours=protocol["budget"]["max_total_gpu_hours"],
        within_budget=hours <= protocol["budget"]["max_total_gpu_hours"],
        estimate="pilot_measurements_not_a_measured_screening_cost",
    )
