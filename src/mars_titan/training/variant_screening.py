"""Protocolo de la criba de variantes de #742 y enumeración de sus trabajos (#746).

El protocolo se registra en `configs/experiments/variant-screening-v1.json` antes de abrir
S. Este módulo lo valida, resuelve los modelos de cada entrada del catálogo a partir de su
declaración y enumera los trabajos por origen y semilla. Dos modelos con la misma
definición son un solo trabajo, así que una referencia común se ajusta una vez por origen
y semilla aunque varias entradas la usen como control. No importa PyTorch ni lee datos.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

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
# Controles comunes que no necesitan ningún ajuste.
COMMON = ("zero", "past_median")
# Trabajos que deben haber terminado antes: el padre congelado y la variante emparejada.
REQUIRES = ("parent", "after")


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

    @property
    def label(self):
        return f"{self.entry}/{self.name}"


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
        and fit.get("stopping") == "fixed_budget"
        and isinstance(fit.get("search_case"), str)
        and fit.get("precision") == "fp32_strict",
        "El ajuste necesita meses, épocas con presupuesto fijo, caso de búsqueda y FP32",
    )
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
    if "combine" in value:
        # La media de semillas no se ajusta: combina las emisiones de su fuente en el análisis.
        return dict(runner="output", source=value["control"], combine=value["combine"])
    return definition


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
        for arm, value in declared.items():
            if arm not in excluded:
                arms[arm] = _fit(entry, document, arm, value)
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
    """Contrastes declarados con su papel: principal, exigido o solo publicado."""
    rows = []
    for name, entry in protocol["catalog"].items():
        if "status" in entry:
            continue
        pairs = [("primary", entry["primary"])]
        pairs += [("required", pair) for pair in entry.get("required", ())]
        pairs += [("reported", pair) for pair in entry.get("reported", ())]
        for role, (variant, control) in pairs:
            rows.append(
                dict(
                    entry=name,
                    role=role,
                    metric=entry["metric"],
                    variant=resolve(protocol, name, variant, root),
                    control=resolve(protocol, name, control, root),
                )
            )
    return rows


def all_arms(protocol, root=ROOT):
    arms = {("references", name): arm for name, arm in reference_arms(protocol).items()}
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
    if runner == "spaced":
        return (None,)
    recipe = json.loads((Path(root) / definition["recipe"]).read_text())
    cases = tuple(recipe.get("walk_forward", {}).get("search_cases", ()))
    _require(cases, f"La receta {definition['recipe']} necesita casos de búsqueda")
    return cases


def _definitions(protocol, arms, arm, root):
    """Definiciones de trabajo de un modelo, una por caso, con la huella de su padre.

    Un control emparejado guarda en su valor la huella de la ventana de su variante en el
    mismo caso, que tiene que terminar antes.
    """
    parent = None
    if "parent" in arm.definition:
        (parent,) = _definitions(
            protocol, arms, arms[("references", arm.definition["parent"])], root
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
        if matched is not None:
            definition["value"] = dict(value, matched_to=matched[case])
        result.append((_digest(definition), definition))
    return result


def screening_jobs(protocol, root=ROOT):
    """Trabajos únicos por definición, caso, origen y semilla, con su padre y sus usos."""
    arms = all_arms(protocol, root)
    jobs = {}
    for label, arm in arms.items():
        if arm.runner not in JOB_RUNNERS:
            continue
        for key, definition in _definitions(protocol, arms, arm, root):
            for origin in protocol["origins"]:
                for seed in protocol["seeds"]:
                    job_id = f"{origin['year']}-{seed}-{key[:16]}"
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
                            used_by=[],
                        ),
                    )
                    _require(job["definition_sha256"] == key, "Dos definiciones comparten prefijo")
                    prefix = f"{origin['year']}-{seed}"
                    if "parent" in definition:
                        job["parent"] = f"{prefix}-{definition['parent'][:16]}"
                    if (
                        isinstance(definition.get("value"), dict)
                        and "matched_to" in definition["value"]
                    ):
                        job["after"] = f"{prefix}-{definition['value']['matched_to'][:16]}"
                    job["used_by"].append("/".join(label))
    for job in jobs.values():
        job["used_by"].sort()
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

    return sorted(jobs.values(), key=lambda job: (level(job["id"]), job["id"]))


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
    lector cuesta lo que declara la medida de referencia.
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
        else:
            cost = reference["reader_window_seconds"]
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
