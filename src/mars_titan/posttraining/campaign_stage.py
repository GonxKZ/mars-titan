"""Etapa de postentrenamiento de la campaña con máscaras: walk-forward por etapas.

La etapa parte de una campaña base confirmada (`training.masked_campaign`). En la variante
A (decisión del 9 de octubre de 2026, «Opción 1 + control»), en cada ventana k ≥ 1 de un
ámbito, el padre de un brazo base y una semilla es el estado elegido de ese brazo en la
ventana k-1: el ganador de la búsqueda o el finalista de la semilla. Ese padre ajustó sus
pesos con `train_{k-1}` y usó `val_{k-1}` y `cal_{k-1}` para elegir y calibrar. Por eso
los casos de la matriz solo ajustan con las filas de `train_k` cuya decisión cae desde el
final de `cal_{k-1}` hasta el final de `train_k`. `staged_rows.fit_rows_proof` demuestra
por identidad de fila que no cortan ninguna fila que el padre usó y la etapa la guarda por
ventana. Cada caso se selecciona con `val_k`, con el padre elegible en la época cero, y
predice validación, calibración y evaluación con las filas de la campaña base en k.

Por cada brazo base y semilla, un trabajo de predicción aplica sin ajustar el padre
congelado a la ventana k, y una selección elige el predictor de la cadena con
`chain_validation_score_v1` (`staged_chain`) entre el padre congelado, los casos de
adaptadores y la continuación completa. La selección escribe los recibos walk-forward de
la cadena y `selection.json` al final. En la ventana 0 no hay postentrenamiento y el
predictor de la cadena es el estado elegido de la base. El reentreno completo de la ventana
k (la campaña base) es el contraste, no un candidato.

La variante B (reentreno cada 36 meses con traslados) conserva su plan para los recuentos,
pero no se ejecuta por decisión del 9 de octubre de 2026.

Las cohortes de ajuste y validación de los brazos neuronales se leen por bloques desde la
vista (`view_blocks`), con un presupuesto de memoria, y cada ventana solo guarda su índice.
Los brazos con entrenador cronológico (Titans-MAC, MARS-TITAN, CM-v1 y la GRU candidata)
cuya sección declara la campaña usan la matriz de versión 3 y los ejecutores de
`chronological_windows` y `candidate_adapters`. Todos los casos ajustados de un mismo padre
aplican el mismo número de actualizaciones.

Cada trabajo confirma un recibo con su identidad, huellas, filas, objetivos, puntuación de
validación y última etiqueta usada, y escribe el recibo walk-forward de cada mercado. Los
trabajos confirmados no se repiten y los pendientes se reanudan desde su punto de control.

Solo se aprende con datos reales de la edición: las vistas de la campaña base, con su
edición verificada por mercado, y la división walk-forward de cada ventana. La etapa
rechaza cualquier declaración de condiciones remuestreadas o sintéticas, aumento o mundos
de episodios, y no importa los módulos del postentrenamiento emparejado anterior.
"""

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mars_titan.data import prediction_files
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.environments.view_cohorts import prepare_cohort_index
from mars_titan.environments.walk_forward_receipt import (
    RECEIPT_KIND as WINDOW_RECEIPT_KIND,
)
from mars_titan.environments.walk_forward_receipt import read_window_receipt
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.models.quantile_head import MEDIAN_INDEX, QUANTILE_COLUMNS, QUANTILE_HEAD
from mars_titan.training import campaign_numerics, campaign_schedule, masked_campaign
from mars_titan.training.campaign_plan import (
    CARRY,
    DECLARED,
    FIT,
    load_campaign,
    plan_campaign,
    schedule,
    scope_arms,
)
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.learning_hold import LearningHoldError, require_learning_allowed

from . import adapter_matrix, staged_chain, staged_rows
from . import chronological_matrix as cm
from .heldout import evaluate_partition
from .matrix_runs import MatrixParent, MatrixWindow, index_manifest, predict_heldout
from .parents import load_parent

STAGE_KIND = "historical_masked_posttraining_stage"
RUN_KIND = "historical_masked_posttraining_stage_run"
RECEIPT_KIND = "masked_posttraining_job"
FROZEN_KIND = "masked_posttraining_frozen_parent"
KEEP, RELEASE = "keep", "release_after_window_fits"
ORDERED, BLOCKS = "ordered_corpus", "view_blocks"
# Memoria de las filas de un bloque, sin el proceso, el modelo ni las cachés del lector.
BLOCK_BYTES = (256 * 1024**2, 16 * 1024**3)
_FIELDS = {
    "schema_version",
    "kind",
    "status",
    "name",
    "campaign",
    "matrix",
    "scopes",
    "arms",
    "cohort_reading",
    "limits",
    "final_test_opened",
}
# Campos que solo declara la variante A, la única con walk-forward por etapas.
_STAGED_FIELDS = {"chain_rule", "data_policy"}
_LIMITS = {"max_training_jobs", "max_prediction_jobs"}
# Diseños de la etapa: por etapas en A y el plan anclado de B, que no se ejecuta.
STAGED, ANCHORED = "staged_chain_v1", "anchored_not_executed"
DATA_POLICY = "real_edition_only"
# Trabajos de la etapa además de los ajustes: padre congelado y selección de la cadena.
FROZEN, SELECT = "frozen", "select"
# Ridge y XGBoost no tienen puntos de adaptación. Su cadena es trivial: en cada ventana k ≥ 1
# solo compite el padre congelado de k-1, con el mismo recibo que los demás brazos, para que
# las políticas lean de todas las familias el mismo tipo de predicción fuera de muestra.
TABULAR = "frozen_parent_only"
PREDICTED = ("validation", *masked_campaign.COMPARED)
B_NOT_EXECUTED = (
    "La variante B no se ejecuta por decisión del 9 de octubre de 2026: el walk-forward por "
    "etapas solo existe en A. Su plan se conserva para los recuentos"
)
# Solo datos reales. Claves y condiciones del postentrenamiento emparejado de #128 (episodios
# remuestreados o sintéticos, aumento y mundos generados) que la etapa rechaza.
REAL, TRAINING_DATA = "real", "real_walk_forward_only"
NOT_REAL_KEYS = frozenset(
    {
        "condition",
        "conditions",
        "augmentation",
        "augmentations",
        "resampling",
        "synthetic",
        "episodes",
        "world",
        "worlds",
    }
)
NOT_REAL_VALUES = frozenset({"real_resampled", "real_synthetic"})
REAL_ONLY = (
    "La etapa de adaptadores solo aprende con datos reales de la edición: no admite "
    "condiciones remuestreadas o sintéticas, aumento ni mundos de episodios"
)


class Paused(Exception):
    """Parada solicitada o ajuste pendiente en una barrera confirmada."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def declares_other_data(value):
    """Si una declaración contiene condiciones, aumento o mundos ajenos a los datos reales."""
    if isinstance(value, dict):
        return any(key in NOT_REAL_KEYS or declares_other_data(v) for key, v in value.items())
    if isinstance(value, list):
        return any(declares_other_data(item) for item in value)
    return isinstance(value, str) and value in NOT_REAL_VALUES


def _names(values, allowed, label):
    _require(
        isinstance(values, list)
        and values
        and len(set(values)) == len(values)
        and all(value in allowed for value in values),
        f"{label} deben ser distintos y pertenecer a la campaña",
    )
    return values


def load_stage(path):
    """Validar la etapa, su campaña y su matriz sin leer datos."""
    path = Path(path)
    config, digest = read_manifest(path, 1024**2)
    _require(not declares_other_data(config), REAL_ONLY)
    _require(
        isinstance(config, dict)
        and set(config) in (_FIELDS, _FIELDS | _STAGED_FIELDS)
        and config["schema_version"] == 1
        and config["kind"] == STAGE_KIND
        and config["status"] == DECLARED
        and config["final_test_opened"] is False
        and isinstance(config["name"], str)
        and comparison._name(config["name"].replace("-", "_")),
        "La etapa de postentrenamiento no cumple su contrato",
    )
    _reading(config["cohort_reading"])
    base = path.parent
    campaign = load_campaign((base / config["campaign"]).resolve())
    _require(campaign["input_policy"] == HISTORICAL_MASKED, REAL_ONLY)
    staged = campaign["variant"] == "A"
    if staged:
        folds = campaign["comparison_config"]["resolved_scopes"]
        _require(
            config.keys() >= _STAGED_FIELDS
            and config["chain_rule"] == staged_chain.RULE
            and config["data_policy"] == DATA_POLICY
            and config["cohort_reading"]["source"] == BLOCKS,
            "La variante A declara la regla de la cadena, la política real_edition_only y "
            "la lectura por bloques que necesita el ajuste con las filas nuevas",
        )
        _require(
            all(
                row["trained"]
                for scope in config["scopes"]
                if scope in folds
                for row in schedule(list(folds[scope]["windows"].values()), campaign["period"])
            ),
            "El walk-forward por etapas necesita una campaña base reentrenada en cada ventana",
        )
    else:
        _require(
            not config.keys() & _STAGED_FIELDS,
            "Solo la variante A declara la cadena del walk-forward por etapas",
        )
    matrix_path = (base / config["matrix"]).resolve()
    _require(not declares_other_data(read_manifest(matrix_path, 1024**2)[0]), REAL_ONLY)
    matrix, matrix_sha256 = adapter_matrix.read_matrix(matrix_path)
    # Todos los brazos neuronales de la campaña emiten cuantiles y se ajustan con pinball.
    adapter_matrix.objectives(matrix, QUANTILE_HEAD)
    _require(
        matrix["input_policy"] == campaign["input_policy"]
        and matrix["budget"]["batch_size"] == campaign["neural"]["batch_size"],
        "La matriz debe compartir la política y el lote de la campaña",
    )
    scopes = _names(config["scopes"], campaign["scopes"], "Los ámbitos")
    _require(
        scopes == [scope for scope in campaign["scopes"] if scope in scopes],
        "Los ámbitos siguen el orden de la campaña",
    )
    neural, tabular = campaign["neural"]["arms"], campaign["tabular"]["arms"]
    declared = campaign["comparison_config"]["arms"]
    # Brazos cronológicos de la comparación, con o sin sección en la campaña.
    known = {name for name, arm in declared.items() if arm["family"] in cm.CAMPAIGN_DESIGNS}
    arms = _names(config["arms"], {*neural, *tabular, *known}, "Los brazos")
    _require(
        all(set(arms) <= set(scope_arms(campaign, scope)) for scope in scopes),
        "Cada brazo de la etapa debe ajustarse en todos sus ámbitos de la campaña",
    )
    _require(
        matrix["schema_version"] >= 3 or not set(arms) & known,
        "Los brazos cronológicos necesitan la matriz de versión 3",
    )
    _require(
        staged or not set(arms) & set(tabular),
        "La cadena trivial de Ridge y XGBoost solo existe en el walk-forward por etapas",
    )
    # Los tabulares no tienen casos de la matriz: su cadena sigue las semillas de su brazo.
    _require(
        all(
            sorted(declared[arm]["seeds"]) == sorted(matrix["budget"]["seeds"])
            for arm in arms
            if arm not in tabular
        ),
        "Cada semilla de la matriz parte del padre elegido con esa semilla",
    )
    limits = config["limits"]
    _require(
        isinstance(limits, dict)
        and set(limits) == _LIMITS
        and all(type(v) is int and 0 <= v <= 100_000 for v in limits.values()),
        "Los límites de trabajos deben ser enteros declarados",
    )
    return dict(
        config,
        sha256=digest,
        path=str(path.resolve()),
        campaign=campaign,
        matrix=matrix,
        matrix_path=str(matrix_path),
        matrix_sha256=matrix_sha256,
        design=STAGED if staged else ANCHORED,
        families={arm: neural[arm] for arm in arms if arm in neural},
    )


def _reading(value):
    """Lectura de las cohortes: corpus ordenado con su retención o bloques de la vista."""
    ordered = (
        isinstance(value, dict)
        and value.keys() == {"source", "retention"}
        and value["source"] == ORDERED
        and value["retention"] in (KEEP, RELEASE)
    )
    blocks = (
        isinstance(value, dict)
        and value.keys() == {"source", "max_block_bytes"}
        and value["source"] == BLOCKS
        and type(value["max_block_bytes"]) is int
        and BLOCK_BYTES[0] <= value["max_block_bytes"] <= BLOCK_BYTES[1]
    )
    _require(ordered or blocks, "La lectura de cohortes de la etapa no es válida")
    return value


def chronological_arms(campaign):
    """Brazos cronológicos con sección en la campaña: familia, variante y banco."""
    from mars_titan.training import campaign_plan as plan

    result = {}
    for arm, variant in (campaign.get(plan.TITANS) or {}).get("arms", {}).items():
        result[arm] = dict(family=plan.TITANS, variant=variant, bank=True)
    for arm, components in (campaign.get(plan.MARS) or {}).get("arms", {}).items():
        bank = components["episodic_bank"] != plan.MARS_BANKS[0]
        result[arm] = dict(family=plan.MARS, variant=None, bank=bank)
    if campaign.get(plan.CM):
        # Los cuatro brazos de CM-v1 leen con el banco M1 de su declaración.
        result.update({arm: dict(family=plan.CM, variant=None, bank=True) for arm in plan.CM_ARMS})
    for arm in (campaign.get(plan.EPISODIC) or {}).get("arms", {}):
        result[arm] = dict(family=plan.EPISODIC, variant=None, bank=True)
    return result


def stage_arms(stage):
    """Brazos activos de la etapa con su familia, y los que esperan su sección."""
    campaign = stage["campaign"]
    neural, chronological = campaign["neural"]["arms"], chronological_arms(campaign)
    tabular = campaign["tabular"]["arms"]
    active, awaiting = {}, {}
    for arm in stage["arms"]:
        if arm in neural:
            active[arm] = dict(family=neural[arm], design=None)
        elif arm in tabular:
            active[arm] = dict(family=tabular[arm], design=TABULAR)
        elif arm in chronological:
            active[arm] = dict(
                chronological[arm], design=cm.CAMPAIGN_DESIGNS[chronological[arm]["family"]]
            )
        else:
            awaiting[arm] = campaign["comparison_config"]["arms"][arm]["family"]
    return active, awaiting


def arm_name(base_arm, point):
    """Nombre del brazo postentrenado, válido para la comparación walk-forward."""
    return f"{base_arm}__{point.replace('+', '_')}"


def _cases(stage, spec):
    """Casos de la matriz de un brazo base: los neuronales o los cronológicos."""
    matrix, digest = stage["matrix"], stage["matrix_sha256"]
    if spec["design"] == TABULAR:
        return []
    if spec["design"] is None:
        items = adapter_matrix.cases(matrix, digest, spec["family"], head=QUANTILE_HEAD)
    else:
        items = cm.cases(matrix, digest, spec["family"], variant=spec["variant"], bank=spec["bank"])
    for item in items:
        _require(item["case"].get("condition", REAL) == REAL, REAL_ONLY)
    return items


def plan_stage(stage):
    """Trabajos de postentrenamiento de la etapa: los de A por etapas o el plan de B."""
    if stage["design"] == ANCHORED:
        return _anchored_plan(stage)
    campaign = stage["campaign"]
    active, _ = stage_arms(stage)
    base_jobs = plan_campaign(campaign)
    cases = {base_arm: _cases(stage, spec) for base_arm, spec in active.items()}
    jobs = []
    for scope in stage["scopes"]:
        windows = [name for name, _ in staged_chain.scope_windows(campaign, scope)]
        for parent_window, window in zip(windows, windows[1:], strict=False):
            for base_arm, spec in active.items():
                if spec["design"] == TABULAR:
                    seeds = sorted(campaign["comparison_config"]["arms"][base_arm]["seeds"])
                else:
                    seeds = list(dict.fromkeys(item["case"]["seed"] for item in cases[base_arm]))
                for seed in seeds:
                    common = dict(
                        scope=scope,
                        window=window,
                        anchor=window,
                        parent_window=parent_window,
                        base_arm=base_arm,
                        family=spec["family"],
                        seed=seed,
                        depends=staged_chain.parent_jobs(
                            base_jobs, scope, parent_window, base_arm, seed
                        ),
                    )
                    arm = staged_chain.frozen_arm(base_arm)
                    jobs.append(
                        dict(
                            common,
                            id=f"{scope}/{window}/{arm}/frozen-s{seed}",
                            arm=arm,
                            point="frozen_parent",
                            control="frozen_parent",
                            kind=FROZEN,
                            case=None,
                        )
                    )
                    for item in cases[base_arm]:
                        if item["case"]["seed"] != seed:
                            continue
                        point = item["id"].split("/", 1)[1]
                        arm = arm_name(base_arm, point)
                        jobs.append(
                            dict(
                                common,
                                id=f"{scope}/{window}/{arm}/{FIT}-s{seed}",
                                arm=arm,
                                point=point,
                                control=item["control"],
                                kind=FIT,
                                case=item["case"],
                            )
                        )
    _require(len({job["id"] for job in jobs}) == len(jobs), "El plan contiene trabajos repetidos")
    return jobs


def _anchored_plan(stage):
    """Plan de B: ajustes en las ventanas reentrenadas y traslados desde su ancla.

    Solo sirve para los recuentos. B no se ejecuta.
    """
    campaign = stage["campaign"]
    active, _ = stage_arms(stage)
    jobs = []
    for scope in stage["scopes"]:
        folds = list(campaign["comparison_config"]["resolved_scopes"][scope]["windows"].values())
        for row in schedule(folds, campaign["period"]):
            for base_arm, spec in active.items():
                for item in _cases(stage, spec):
                    seed, point = item["case"]["seed"], item["id"].split("/", 1)[1]
                    arm = arm_name(base_arm, point)
                    kind = FIT if row["trained"] else CARRY
                    jobs.append(
                        dict(
                            id=f"{scope}/{row['window']}/{arm}/{kind}-s{seed}",
                            scope=scope,
                            window=row["window"],
                            anchor=row["anchor"],
                            arm=arm,
                            base_arm=base_arm,
                            family=spec["family"],
                            point=point,
                            control=item["control"],
                            seed=seed,
                            kind=kind,
                            case=item["case"],
                            depends=[]
                            if row["trained"]
                            else [f"{scope}/{row['anchor']}/{arm}/{FIT}-s{seed}"],
                        )
                    )
    _require(len({job["id"] for job in jobs}) == len(jobs), "El plan contiene trabajos repetidos")
    return jobs


def plan_chain(stage, jobs):
    """Selecciones de la cadena de cada ámbito, ventana, brazo base y semilla.

    En la ventana 0 dependen de los trabajos base que eligen el estado. En las demás, del
    padre congelado y de todos los casos de su ventana, brazo base y semilla.
    """
    _require(stage["design"] == STAGED, "Solo el walk-forward por etapas tiene cadena")
    campaign = stage["campaign"]
    base_jobs = plan_campaign(campaign)
    grouped = {}
    for job in jobs:
        key = (job["scope"], job["window"], job["base_arm"], job["seed"])
        grouped.setdefault(key, []).append(job["id"])
    families = {(job["scope"], job["base_arm"]): job["family"] for job in jobs}
    chains = []
    for scope in stage["scopes"]:
        windows = [name for name, _ in staged_chain.scope_windows(campaign, scope)]
        pairs = list(
            dict.fromkeys((job["base_arm"], job["seed"]) for job in jobs if job["scope"] == scope)
        )
        for index, window in enumerate(windows):
            for base_arm, seed in pairs:
                if index:
                    depends = grouped.get((scope, window, base_arm, seed), [])
                    _require(depends, f"{scope}/{window}/{base_arm} no tiene posentrenamiento")
                else:
                    depends = staged_chain.parent_jobs(base_jobs, scope, window, base_arm, seed)
                chains.append(
                    dict(
                        id=staged_chain.chain_job_id(scope, window, base_arm, seed),
                        scope=scope,
                        window=window,
                        anchor=window,
                        parent_window=windows[index - 1] if index else None,
                        arm=staged_chain.chain_arm(base_arm),
                        base_arm=base_arm,
                        family=families[scope, base_arm],
                        seed=seed,
                        kind=SELECT,
                        stage="chain",
                        depends=depends,
                    )
                )
    return chains


def window_plan(campaign, jobs, chains, window):
    """Trabajos y selecciones de una ventana de campaña y los pares de la base que leen.

    Cada trabajo parte del estado elegido en la ventana anterior de su ámbito, que es un
    trabajo de la campaña base y no de la etapa. Por eso los pares incluyen esa ventana
    además de la propia, y la reanudación confirma sus recibos antes de seguir.
    """
    pairs = campaign_schedule.window_pairs(campaign, window)
    jobs = [job for job in jobs if (job["scope"], job["window"]) in pairs]
    chains = [job for job in chains if (job["scope"], job["window"]) in pairs]
    # Cada brazo base y semilla tiene una selección por ventana, así que las selecciones
    # nombran todos los pares de los ámbitos de la etapa.
    own = {(job["scope"], job["window"]) for job in chains}
    parents = {(job["scope"], job["parent_window"]) for job in chains if job["parent_window"]}
    return jobs, chains, own | parents


def ordered_jobs(jobs, chains):
    """Orden de ejecución: por ventana, los trabajos de cada brazo base y semilla y su
    selección de la cadena justo después."""
    grouped = {}
    for job in jobs:
        grouped.setdefault((job["scope"], job["window"], job["base_arm"], job["seed"]), []).append(
            job
        )
    order = []
    for chain in chains:
        order.extend(
            grouped.get((chain["scope"], chain["window"], chain["base_arm"], chain["seed"]), [])
        )
        order.append(chain)
    _require(len(order) == len(jobs) + len(chains), "Hay trabajos sin selección de la cadena")
    return order


def count_stage(stage, jobs=None):
    """Contar ajustes, predicciones y selecciones por ámbito, brazo y semilla."""
    jobs = plan_stage(stage) if jobs is None else jobs
    chains = plan_chain(stage, jobs) if stage["design"] == STAGED else []
    prediction = FROZEN if stage["design"] == STAGED else CARRY
    scopes = {}
    for scope in stage["scopes"]:
        selected = [job for job in jobs if job["scope"] == scope]
        arms = {}
        for job in selected:
            entry = arms.setdefault(job["arm"], {}).setdefault(str(job["seed"]), Counter())
            entry[job["kind"]] += 1
        windows = list(staged_chain.scope_windows(stage["campaign"], scope))
        fitted = list(dict.fromkeys(job["window"] for job in selected if job["kind"] == FIT))
        scopes[scope] = dict(
            windows=len(windows),
            fitted_windows=fitted,
            training_jobs=sum(job["kind"] == FIT for job in selected),
            prediction_jobs=sum(job["kind"] == prediction for job in selected),
            selection_jobs=sum(chain["scope"] == scope for chain in chains),
            arms={arm: {seed: dict(c) for seed, c in seeds.items()} for arm, seeds in arms.items()},
        )
    totals = dict(
        training_jobs=sum(job["kind"] == FIT for job in jobs),
        prediction_jobs=sum(job["kind"] == prediction for job in jobs),
        selection_jobs=len(chains),
    )
    limits = stage["limits"]
    for kind, limit in (
        ("training_jobs", "max_training_jobs"),
        ("prediction_jobs", "max_prediction_jobs"),
    ):
        _require(
            totals[kind] <= limits[limit],
            f"La etapa prevé {totals[kind]} trabajos ({kind}) y supera el límite "
            f"declarado {limit}={limits[limit]}",
        )
    return dict(scopes=scopes, **totals)


def check_stage(path):
    """Validar, planificar y contar sin leer datos, reservar la GPU ni ajustar."""
    stage = load_stage(path)
    campaign = stage["campaign"]
    _, awaiting = stage_arms(stage)
    return dict(
        status="checked",
        name=stage["name"],
        design=stage["design"],
        executable=stage["design"] == STAGED,
        not_executed_reason=None if stage["design"] == STAGED else B_NOT_EXECUTED,
        chain_rule=stage.get("chain_rule"),
        data_policy=stage.get("data_policy"),
        awaiting_sections=awaiting,
        variant=campaign["variant"],
        stage_sha256=stage["sha256"],
        campaign_sha256=campaign["sha256"],
        matrix_sha256=stage["matrix_sha256"],
        input_policy=campaign["input_policy"],
        objectives=adapter_matrix.objectives(stage["matrix"], QUANTILE_HEAD),
        excluded_controls=adapter_matrix.excluded_controls(stage["matrix"], QUANTILE_HEAD),
        cohort_reading=stage["cohort_reading"],
        counts=count_stage(stage),
        scientific_training_started=False,
        final_test_opened=False,
    )


def _code():
    root = Path(__file__).parents[1]
    names = (
        "posttraining/campaign_stage.py",
        "posttraining/staged_rows.py",
        "posttraining/staged_chain.py",
        "training/campaign_chain.py",
        "posttraining/matrix_runs.py",
        "environments/view_cohorts.py",
        "posttraining/adapter_matrix.py",
        "posttraining/run.py",
        "posttraining/heldout.py",
        "posttraining/evaluation.py",
        "posttraining/inputs.py",
        "environments/cohort_order.py",
        "posttraining/parents.py",
        "posttraining/chronological_matrix.py",
        "posttraining/chronological_windows.py",
        "posttraining/readout_adapters.py",
        "posttraining/candidate_adapters.py",
        "models/predictive_adaptation.py",
        "models/quantile_head.py",
        "training/masked_campaign.py",
        "training/campaign_plan.py",
        "training/carried_predictions.py",
        "environments/walk_forward_receipt.py",
        "evaluation/walk_forward_comparison.py",
        "evaluation/session_metrics.py",
    )
    return {name: sha256(root / name) for name in names}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _base_receipts(base, campaign, stage, pairs=None):
    """Confirmar los trabajos base de los ámbitos y brazos de la etapa, en orden del plan.

    Un brazo que parte de otro predictor elegido (MARS-TITAN, CM-v1) necesita también los
    recibos de sus padres, aunque la etapa no los adapte. `pairs` limita la confirmación a
    esos pares (ámbito, ventana) al ejecutar una sola ventana.
    """
    jobs = [job for job in plan_campaign(campaign) if job["scope"] in stage["scopes"]]
    needed, size = set(stage_arms(stage)[0]), 0
    while len(needed) != size:
        size = len(needed)
        needed |= {job["parent"] for job in jobs if job["arm"] in needed and job.get("parent")}
    for job in jobs:
        if job["arm"] not in needed:
            continue
        if pairs is not None and (job["scope"], job["window"]) not in pairs:
            continue
        case, _, sources = base.resolve(job)
        receipt = base.confirmed(job, base.job_identity(job, case, sources))
        _require(receipt is not None, f"Falta confirmar {job['id']} en la campaña base")
        base.receipts[job["id"]] = receipt


def _ordered_quantiles(table, label):
    """Exigir cinco niveles no decrecientes y la mediana como predicción, con los mismos bits."""
    levels = np.stack([table[name].to_numpy() for name in QUANTILE_COLUMNS], axis=1)
    _require(
        np.isfinite(levels).all() and (np.diff(levels, axis=1) >= 0).all(),
        f"{label}: los cuantiles no son finitos o no conservan su orden",
    )
    _require(
        np.array_equal(table["prediction"].to_numpy(), levels[:, MEDIAN_INDEX]),
        f"{label}: la predicción no es la mediana de la cabeza",
    )


def _attempt(folder, name, limit=16):
    """Carpeta de un padre congelado cronológico: el último intento completado o uno nuevo.

    Sus ejecutores escriben en un directorio nuevo. Un intento interrumpido se conserva y
    el siguiente usa otra carpeta, como los intentos de la campaña base.
    """
    attempts = sorted(folder.glob("attempt-*"))
    if attempts and (attempts[-1] / name).is_file():
        receipt, _ = read_manifest(attempts[-1] / name, 16 * 1024**2)
        if receipt.get("status") == "completed":
            return attempts[-1], receipt
    _require(len(attempts) < limit, f"{folder} alcanzó el límite de intentos")
    return folder / f"attempt-{len(attempts) + 1:04d}", None


def release_indices(folder):
    """Borrar los índices de observaciones de un trabajo cronológico ya confirmado.

    Solo los reconstruye una nueva ejecución del mismo trabajo, que ya no ocurre tras su
    recibo. Devuelve los bytes liberados.
    """
    released = 0
    for path in sorted(folder.rglob("indices"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            released += sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            shutil.rmtree(path)
    return released


def candidate_kind(job):
    """Clase del candidato de la cadena que aporta un trabajo de la ventana."""
    if job["kind"] == FROZEN:
        return "frozen_parent"
    return "continuation" if job["control"] == "full_continuation" else "adapter"


class _Stage:
    """Estado confirmado de la etapa: ventanas, padres, recibos y cadena."""

    def __init__(self, stage, base, campaign_output, output, identity, *, device, lease, stop):
        self.stage, self.base, self.output = stage, base, output
        self.campaign = stage["campaign"]
        self.campaign_output, self.identity = campaign_output, identity
        self.identity_sha256 = _digest(identity)
        self.device, self.lease, self.stop = device, lease, stop
        self.policy = self.campaign["input_policy"]
        self.batch_size = stage["matrix"]["budget"]["batch_size"]
        self.receipts, self.budgets, self.released_indices = {}, {}, {}
        self.proofs, self.populations, self.selections = {}, {}, {}
        self.window_key = self.parent_key = self.dataset_key = None
        self.window = self.parent = self.dataset = None

    # Contextos: una ventana, un padre y una vista abiertos como máximo.
    def close_parent(self):
        if self.parent is not None:
            self.parent.close()
        self.parent_key = self.parent = None
        gc.collect()

    def close_window(self):
        self.close_parent()
        if self.window is not None:
            self.window.close()
        self.window_key = self.window = None

    def close(self):
        self.close_window()
        self.dataset_key = self.dataset = None

    def view(self, scope, window):
        return self.base.views[scope]["windows"][window]

    def fold(self, scope, window):
        return self.campaign["comparison_config"]["resolved_scopes"][scope]["windows"][window]

    def window_folder(self, scope, window):
        return self.output / "windows-data" / scope / window

    def open_dataset(self, scope, window):
        if self.dataset_key != (scope, window):
            self.dataset = CorpusDataset(
                Path(self.view(scope, window)["path"]), input_policy=self.policy
            )
            self.dataset_key = (scope, window)
        return self.dataset

    def population(self, scope, window):
        """Índice de cohortes de una vista: identifica la población con la que ajustó el padre."""
        key = (scope, window)
        if key not in self.populations:
            folder = self.window_folder(scope, window) / "cohorts"
            prepared = prepare_cohort_index(
                CorpusDataset(Path(self.view(scope, window)["path"]), input_policy=self.policy),
                folder,
                input_policy=self.policy,
                stop=self.stop,
            )
            _require(
                prepared["source_sha256"] == self.view(scope, window)["sha256"],
                f"El índice de {scope}/{window} no corresponde a su vista",
            )
            self.populations[key] = index_manifest(self.window_folder(scope, window))
        return self.populations[key]

    def proof(self, scope, window, parent_window):
        """Prueba de disjunción de las filas nuevas de una ventana, calculada una vez."""
        key = (scope, window)
        if key not in self.proofs:
            path = self.window_folder(scope, window) / "fit-rows.json"
            expected = (
                self.view(scope, parent_window)["sha256"],
                self.view(scope, window)["sha256"],
            )
            if not path.is_file():
                proof = staged_rows.fit_rows_proof(
                    CorpusDataset(
                        Path(self.view(scope, parent_window)["path"]), input_policy=self.policy
                    ),
                    CorpusDataset(Path(self.view(scope, window)["path"]), input_policy=self.policy),
                    parent_fold=self.fold(scope, parent_window),
                    fold=self.fold(scope, window),
                )
                atomic_json(path, proof)
            proof, digest = read_manifest(path, 1024**2)
            _require(
                proof.get("kind") == staged_rows.PROOF_KIND
                and (proof["parent_view_sha256"], proof["view_sha256"]) == expected
                and (proof["parent_window"], proof["window"]) == (parent_window, window)
                and not any(proof["intersection"].values()),
                f"La prueba de filas nuevas de {scope}/{window} no corresponde a sus vistas",
            )
            self.proofs[key] = dict(proof, file_sha256=digest)
        return self.proofs[key]

    def open_window(self, scope, window, since):
        if self.window_key != (scope, window, since):
            self.close_window()
            view = self.view(scope, window)
            self.window = MatrixWindow(
                view["path"],
                self.window_folder(scope, window),
                encoding=view["sha256"],
                input_policy=self.policy,
                batch_size=self.batch_size,
                stop=self.stop,
                max_block_bytes=self.stage["cohort_reading"]["max_block_bytes"],
                since=since,
            )
            self.window_key = (scope, window, since)
            _require(
                self.window.source_sha256 == view["sha256"],
                "El índice de cohortes no corresponde a la vista de la ventana",
            )
        return self.window

    def base_parent(self, scope, window, base_arm, seed):
        """Trabajo base elegido para la semilla en la ventana y su informe verificado."""
        key, receipt = self.base.selected(scope, window, base_arm, seed)
        report = self.campaign_output / receipt["report"]["path"]
        _require(
            sha256(report) == receipt["report"]["sha256"],
            f"El informe del padre {key} ha cambiado",
        )
        return key, receipt, report

    def open_parent(self, job):
        key = (job["scope"], job["window"], job["base_arm"], job["seed"])
        if self.parent_key != key:
            self.close_parent()
            proof = self.proof(job["scope"], job["window"], job["parent_window"])
            window = self.open_window(job["scope"], job["window"], proof["start_us"])
            _, _, report = self.base_parent(
                job["scope"], job["parent_window"], job["base_arm"], job["seed"]
            )
            folder = self.window_folder(job["scope"], job["window"])
            self.parent = MatrixParent(
                window,
                report,
                folder / "parents" / job["base_arm"] / f"seed-{job['seed']}",
                matrix=self.stage["matrix"],
                digest=self.stage["matrix_sha256"],
                seed=job["seed"],
                device=self.device,
                lease=self.lease,
                stop=self.stop,
                population=self.population(job["scope"], job["parent_window"]),
            )
            self.parent_key = key
            self.budgets["/".join(map(str, key))] = self.parent.budget
            _require(
                self.parent.data.counts["train"] == proof["rows"],
                f"{job['id']}: el ajuste no lee exactamente las filas nuevas de la prueba",
            )
        return self.parent

    def job_identity(self, job):
        key, receipt, _ = self.base_parent(
            job["scope"], job["parent_window"], job["base_arm"], job["seed"]
        )
        proof = self.proof(job["scope"], job["window"], job["parent_window"])
        return dict(
            stage_identity_sha256=self.identity_sha256,
            **{
                name: job[name]
                for name in (
                    "id",
                    "scope",
                    "window",
                    "parent_window",
                    "arm",
                    "base_arm",
                    "family",
                    "point",
                    "control",
                    "seed",
                    "kind",
                    "case",
                )
            },
            view_sha256=self.view(job["scope"], job["window"])["sha256"],
            parent_view_sha256=self.view(job["scope"], job["parent_window"])["sha256"],
            parent=dict(
                job=key,
                receipt_sha256=receipt["sha256"],
                checkpoint_sha256=receipt["parent"]["sha256"],
            ),
            fit_rows=dict(sha256=proof["sha256"], proof_sha256=proof["file_sha256"])
            if job["kind"] == FIT
            else None,
        )

    def folder(self, job):
        return self.output / "jobs" / job["id"]

    def confirmed(self, job, identity):
        path = self.folder(job) / "receipt.json"
        if not path.is_file():
            return None
        receipt, digest = read_manifest(path, 8 * 1024**2)
        _require(receipt.get("identity") == identity, f"{job['id']} cambió de identidad")
        _require(
            sha256(self.output / receipt["run"]["path"]) == receipt["run"]["sha256"],
            f"Un artefacto confirmado de {job['id']} ha cambiado",
        )
        for record in receipt["predictions"].values():
            prediction_files.verify(
                self.output / record["path"],
                record["sha256"],
                label=f"Un artefacto confirmado de {job['id']} ha cambiado",
            )
        return dict(receipt, sha256=digest)

    def fit(self, job, folder):
        """Ajustar o recuperar el caso desde el padre de k-1 con las filas nuevas de k."""
        if job["family"] in cm.CAMPAIGN_DESIGNS:
            return self.fit_chronological(job, folder)
        parent = self.open_parent(job)
        rows = [row for row in parent.rows if row["id"] == f"seed-{job['seed']}/{job['point']}"]
        _require(
            len(rows) == 1 and rows[0]["case"] == job["case"],
            f"{job['id']} no corresponde al plan de su padre",
        )
        row = rows[0]
        report = parent.run(
            row,
            folder / "run",
            checkpoint_seconds=self.campaign["neural"]["checkpoint_seconds"],
            stop=self.stop,
        )
        if report["status"] != "completed":
            raise Paused
        run_path = folder / "run" / "run.json"
        predictions = predict_heldout(
            parent.parent,
            run_path,
            report,
            self.open_dataset(job["scope"], job["window"]),
            folder,
            device=self.device,
            batch_size=self.batch_size,
            stop=self.stop,
            partitions=PREDICTED,
        )
        return dict(
            run=run_path,
            predictions=predictions,
            parent=dict(id=job["id"], sha256=report["checkpoint"]["sha256"]),
            state=folder / "run",
            score=report["predictions"]["validation"]["metrics"]["session_mae"],
            updates=report["global_step"],
            selection=report["selection"],
        )

    def fit_chronological(self, job, folder):
        """Postentrenar el caso cronológico desde el padre de k-1 con las filas nuevas."""
        from mars_titan.training.titans_walk_forward import unfused_attention

        from .candidate_adapters import run_candidate_posttraining
        from .chronological_windows import run_readout_posttraining, run_titans_posttraining

        self.close_window()
        scope = job["scope"]
        _, _, report = self.base_parent(scope, job["parent_window"], job["base_arm"], job["seed"])
        runner = {
            cm.TITANS: run_titans_posttraining,
            cm.READOUT: run_readout_posttraining,
            cm.CANDIDATE: run_candidate_posttraining,
        }[cm.CAMPAIGN_DESIGNS[job["family"]]]
        output = folder / "run"
        with unfused_attention():
            result = runner(
                report.parent,
                Path(self.view(scope, job["window"])["path"]),
                output,
                case=job["case"],
                matrix=self.stage["matrix"],
                digest=self.stage["matrix_sha256"],
                device=self.device,
                stop=self.stop,
                parent_view=Path(self.view(scope, job["parent_window"])["path"]),
            )
        if result["status"] != "completed":
            raise Paused
        fit = result.get("fit", result)
        return dict(
            run=output / ("run.json" if "fit" in result else "window.json"),
            predictions=self._chronological_predictions(output, folder, result["predictions"]),
            parent=dict(id=job["id"], sha256=result["checkpoint"]["sha256"]),
            state=output,
            score=result["predictions"]["validation"]["metrics"]["session_mae"],
            updates=fit["global_step"],
            selection=fit["selection"],
        )

    @staticmethod
    def _chronological_predictions(output, folder, records):
        """Rutas de las predicciones del ejecutor, relativas a la carpeta del trabajo."""
        return {
            name: dict(
                path=str((output / record["path"]).relative_to(folder)), sha256=record["sha256"]
            )
            for name, record in records.items()
        }

    def frozen(self, job, folder):
        """Padre congelado: el estado elegido de la base en k-1 aplicado a la ventana k."""
        key, receipt, report = self.base_parent(
            job["scope"], job["parent_window"], job["base_arm"], job["seed"]
        )
        result = dict(
            parent=dict(id=key, sha256=receipt["parent"]["sha256"]),
            state=report.parent,
            score=None,
            updates=0,
            selection=None,
        )
        if job["family"] in cm.CAMPAIGN_DESIGNS:
            return dict(result, **self.frozen_chronological(job, folder, report))
        if self.tabular(job):
            return dict(result, **self.frozen_tabular(job, folder, report))
        diagnostic = self.device == "cpu"
        parent = load_parent(
            self.population(job["scope"], job["parent_window"]),
            report,
            device=self.device,
            diagnostic=diagnostic,
            lease=self.lease,
        )
        dataset = self.open_dataset(job["scope"], job["window"])
        predictions = {}
        for partition in PREDICTED:
            path = folder / f"{partition}-predictions.parquet"
            # El propio modelo del padre emite sus cuantiles por el recorrido de los casos.
            metrics = evaluate_partition(
                dataset,
                parent,
                partition,
                path,
                stop=self.stop,
                device=self.device,
                model=parent.model,
                neural=True,
                batch_size=self.batch_size,
            )
            predictions[partition] = dict(path=path.name, sha256=sha256(path), metrics=metrics)
        run = folder / "frozen.json"
        atomic_json(
            run,
            dict(
                schema_version=1,
                kind=FROZEN_KIND,
                status="completed",
                parent=dict(
                    job=key,
                    receipt_sha256=receipt["sha256"],
                    checkpoint_sha256=receipt["parent"]["sha256"],
                    report=str(report),
                ),
                view_sha256=self.view(job["scope"], job["window"])["sha256"],
                predictions=predictions,
                final_test_opened=False,
            ),
        )
        return dict(result, run=run, predictions=predictions)

    def tabular(self, job):
        """Si el brazo base es Ridge o XGBoost, cuya cadena solo tiene el padre congelado."""
        return job["base_arm"] in self.campaign["tabular"]["arms"]

    def frozen_tabular(self, job, folder, report):
        """Padre congelado de Ridge o XGBoost: el traslado de la campaña base desde k-1.

        Usa el mismo ejecutor que los traslados de la campaña, con el estado elegido en k-1
        como ancla, y además predice la validación de k, con la que la cadena puntúa. Las
        tablas no tienen cuantiles porque estos modelos solo emiten la predicción puntual.
        """
        from mars_titan.training.carried_predictions import carry_tabular

        output, result = _attempt(folder, "carry.json")
        if result is None:
            result = carry_tabular(
                report.parent,
                Path(self.view(job["scope"], job["parent_window"])["path"]),
                Path(self.view(job["scope"], job["window"])["path"]),
                output,
                kind=job["family"],
                batch_size=self.campaign["tabular"]["batch_size"],
                input_policy=self.policy,
                frozen_parent=True,
            )
        _require(
            result.get("status") == "completed" and result.get("frozen_parent") is True,
            f"{job['id']}: el traslado del padre tabular no está confirmado",
        )
        return dict(
            run=output / "carry.json",
            predictions=self._chronological_predictions(output, folder, result["predictions"]),
        )

    def frozen_chronological(self, job, folder, report):
        """Padre congelado cronológico: sus ejecutores predicen la ventana sin ajustar."""
        from mars_titan.training.titans_walk_forward import unfused_attention

        from .candidate_adapters import frozen_candidate
        from .chronological_windows import frozen_readout, frozen_titans

        self.close_window()
        runner = {
            cm.TITANS: frozen_titans,
            cm.READOUT: frozen_readout,
            cm.CANDIDATE: frozen_candidate,
        }[cm.CAMPAIGN_DESIGNS[job["family"]]]
        output, result = _attempt(folder, "frozen.json")
        if result is None:
            with unfused_attention():
                result = runner(
                    report.parent,
                    Path(self.view(job["scope"], job["parent_window"])["path"]),
                    Path(self.view(job["scope"], job["window"])["path"]),
                    output,
                    device=self.device,
                    stop=self.stop,
                )
        if result["status"] != "completed":
            raise Paused
        return dict(
            run=output / "frozen.json",
            predictions=self._chronological_predictions(output, folder, result["predictions"]),
        )

    def confirm(self, job, identity, folder, result):
        """Comprobar tramos, filas, objetivos, cuantiles y validación, y escribir el recibo."""
        resolved = self.campaign["comparison_config"]["resolved_scopes"][job["scope"]]
        window = resolved["windows"][job["window"]]
        view = self.view(job["scope"], job["window"])
        proof = self.proof(job["scope"], job["window"], job["parent_window"])
        # Ridge y XGBoost emiten solo la predicción puntual, sin la cabeza de cuantiles.
        quantiles = not self.tabular(job)
        columns = comparison.COLUMNS + (QUANTILE_COLUMNS if quantiles else ())
        if self.campaign.get("numerics"):
            campaign_numerics.require_job(self.campaign["numerics"], job["id"], result)
        predictions, score = {}, None
        for partition in PREDICTED:
            record = result["predictions"][partition]
            path = folder / record["path"]
            safe_destination(path)
            label = f"{job['id']} ({partition})"
            table = comparison._read_predictions(dict(path=path, sha256=record["sha256"]), columns)
            comparison._check_segment(table, window, partition, resolved["markets"], label)
            _require(
                table.num_rows == view["counts"][partition],
                f"{label}: {table.num_rows} filas frente a {view['counts'][partition]} de la vista",
            )
            if quantiles:
                _ordered_quantiles(table, label)
            rows = masked_campaign._rows_digest(table)
            if partition == "validation":
                score = staged_chain.validation_score(table)
            else:
                source, expected = self.base.rows[(job["scope"], job["window"], partition)]
                _require(
                    rows == expected,
                    f"{label}: no evalúa las mismas filas ni objetivos que {source} de la "
                    "campaña base",
                )
            predictions[partition] = dict(
                path=str(path.relative_to(self.output)),
                sha256=record["sha256"],
                rows=table.num_rows,
                rows_sha256=rows,
                markets=masked_campaign._fingerprints(table, resolved["markets"]),
            )
        if result["score"] is not None:
            _require(
                math.isclose(score, result["score"], rel_tol=1e-6, abs_tol=1e-12),
                f"{job['id']}: la validación guardada no reproduce la puntuación elegida",
            )
        fit_rows = None
        if job["kind"] == FIT:
            fit_rows = {
                name: proof[name]
                for name in (
                    "start",
                    "end",
                    "rows",
                    "sha256",
                    "first_decision",
                    "last_decision",
                    "intersection",
                    "parent_rows",
                )
            }
        run_path = result["run"]
        receipt = dict(
            schema_version=1,
            kind=RECEIPT_KIND,
            status="completed",
            identity=identity,
            run=dict(path=str(run_path.relative_to(self.output)), sha256=sha256(run_path)),
            parent=result["parent"],
            state=str(Path(result["state"]).resolve()),
            score=score,
            reported_score=result["score"],
            updates=result["updates"],
            selection=result["selection"],
            fit_rows=fit_rows,
            labels_used_until=proof["labels_used_until"],
            predictions=predictions,
            final_test_opened=False,
            confirmed_at_utc=datetime.now(UTC).isoformat(),
        )
        path = self.folder(job) / "receipt.json"
        atomic_json(path, receipt)
        return dict(receipt, sha256=sha256(path))

    def _market_receipts(self, folder, scope, window, parent, labels, predictions):
        """Recibo walk-forward de cada mercado con evaluación, con el contrato de #390."""
        resolved = self.campaign["comparison_config"]["resolved_scopes"][scope]
        digests = {}
        for market in resolved["markets"]:
            if market not in predictions["evaluation"]["markets"]:
                continue
            record = dict(
                kind=WINDOW_RECEIPT_KIND,
                schema_version=1,
                protocol=resolved["protocols"][market],
                fold=resolved["windows"][window],
                parent=parent,
                labels_used_until=labels,
                predictions={
                    partition: value["markets"][market]
                    for partition, value in predictions.items()
                    if market in value["markets"]
                },
            )
            read_window_receipt(record)
            path = folder / f"{market}.json"
            if path.is_file():
                _require(
                    read_manifest(path, 1024**2)[0] == record,
                    f"El recibo de {path.relative_to(self.output)} no corresponde a sus trabajos",
                )
            else:
                atomic_json(path, record)
            digests[market] = sha256(path)
        return digests

    def publish(self, job, receipt):
        """Recibos walk-forward del brazo del trabajo, con su última etiqueta usada."""
        self._market_receipts(
            self.output
            / "windows"
            / job["scope"]
            / job["window"]
            / job["arm"]
            / f"seed-{job['seed']}",
            job["scope"],
            job["window"],
            dict(id=job["id"], sha256=receipt["sha256"]),
            receipt["labels_used_until"],
            receipt["predictions"],
        )

    def equal_updates(self, job, receipt):
        """Todos los casos ajustados de un padre aplican el mismo número de actualizaciones."""
        if job["kind"] != FIT:
            return
        key = (job["scope"], job["window"], job["base_arm"], job["seed"])
        for name, other in self.receipts.items():
            identity = other["identity"]
            same = (identity["scope"], identity["window"], identity["base_arm"], identity["seed"])
            if identity["kind"] == FIT and same == key:
                _require(
                    other["updates"] == receipt["updates"],
                    f"{job['id']} aplica {receipt['updates']} actualizaciones y {name} "
                    f"{other['updates']} con el mismo padre",
                )

    def _base_selected(self, job):
        """Ventana 0: trabajo base elegido, su recibo y su informe, sin leer filas."""
        key, receipt, report_path = self.base_parent(
            job["scope"], job["window"], job["base_arm"], job["seed"]
        )
        selected = dict(kind="base", arm=job["base_arm"], job=key, receipt_sha256=receipt["sha256"])
        return key, receipt, report_path, selected

    def _base_choice(self, job):
        """Ventana 0: el estado elegido de la base, con su validación, como predictor."""
        scope, window = job["scope"], job["window"]
        key, receipt, report_path, selected = self._base_selected(job)
        resolved = self.campaign["comparison_config"]["resolved_scopes"][scope]
        report, _ = read_manifest(report_path, 16 * 1024**2)
        record = report["predictions"]["validation"]
        path = report_path.parent / record["path"]
        safe_destination(path)
        table = comparison._read_predictions(
            dict(path=path, sha256=record["sha256"]), comparison.COLUMNS
        )
        comparison._check_segment(
            table, resolved["windows"][window], "validation", resolved["markets"], key
        )
        _require(
            table.num_rows == self.view(scope, window)["counts"]["validation"],
            f"{key}: la validación no tiene las filas de la vista",
        )
        predictions = dict(
            validation=dict(markets=masked_campaign._fingerprints(table, resolved["markets"])),
            **{name: receipt["predictions"][name] for name in masked_campaign.COMPARED},
        )
        state = dict(path=str(report_path.parent.resolve()), sha256=receipt["parent"]["sha256"])
        labels = staged_rows.labels_used_until(self.open_dataset(scope, window))
        return None, [], selected, state, None, labels, predictions

    def _chain_choice(self, job):
        """Ventana k ≥ 1: el candidato de `chain_validation_score_v1` entre los trabajos."""
        candidates = []
        for name in job["depends"]:
            receipt = self.receipts[name]
            identity = receipt["identity"]
            candidates.append(
                dict(
                    kind=candidate_kind(identity),
                    arm=identity["arm"],
                    job=name,
                    receipt_sha256=receipt["sha256"],
                    score=receipt["score"],
                )
            )
        _require(
            len(
                {
                    self.receipts[name]["predictions"]["validation"]["rows_sha256"]
                    for name in job["depends"]
                }
            )
            == 1,
            f"{job['id']}: los candidatos no se validan con las mismas filas",
        )
        chosen = staged_chain.choose(candidates)
        receipt = self.receipts[chosen["job"]]
        frozen = next(c for c in candidates if c["kind"] == "frozen_parent")
        parent = dict(self.receipts[frozen["job"]]["identity"]["parent"])
        selected = {key: chosen[key] for key in ("kind", "arm", "job", "receipt_sha256")}
        state = dict(path=receipt["state"], sha256=receipt["parent"]["sha256"])
        fit_rows = None
        if chosen["kind"] != "frozen_parent":
            fit_rows = {
                name: receipt["fit_rows"][name]
                for name in ("first_decision", "last_decision", "rows", "sha256")
            }
        return (
            parent,
            candidates,
            selected,
            state,
            fit_rows,
            receipt["labels_used_until"],
            receipt["predictions"],
        )

    def select(self, job):
        """Elegir y publicar el predictor de la cadena. `selection.json` se escribe la última."""
        scope, window, base_arm, seed = job["scope"], job["window"], job["base_arm"], job["seed"]
        first = job["parent_window"] is None
        existing = staged_chain.read_selection(self.output, scope, window, base_arm, seed)
        if existing is not None:
            # Una selección confirmada se comprueba con los recibos, sin releer filas. La
            # retención v2 puede haber liberado la validación de la base al cerrar la ventana.
            if first:
                parent, candidates, selected = None, [], self._base_selected(job)[-1]
            else:
                parent, candidates, selected = self._chain_choice(job)[:3]
            _require(
                existing["selected"] == selected
                and existing["candidates"] == candidates
                and existing["parent"] == parent
                and existing["campaign_sha256"] == self.campaign["sha256"]
                and existing["stage_sha256"] == self.stage["sha256"],
                f"La selección confirmada de {job['id']} no corresponde a sus trabajos",
            )
            self.selections[job["id"]] = existing
            return
        parent, candidates, selected, state, fit_rows, labels, predictions = (
            self._base_choice(job) if first else self._chain_choice(job)
        )
        folder = staged_chain.chain_folder(self.output, scope, window, base_arm, seed)
        markets = self._market_receipts(
            folder,
            scope,
            window,
            dict(id=selected["job"], sha256=selected["receipt_sha256"]),
            labels,
            predictions,
        )
        atomic_json(
            folder / staged_chain.SELECTION,
            dict(
                kind=staged_chain.SELECTION_KIND,
                schema_version=1,
                campaign_sha256=self.campaign["sha256"],
                stage_sha256=self.stage["sha256"],
                scope=scope,
                window=window,
                base_arm=base_arm,
                seed=seed,
                rule=staged_chain.RULE,
                parent_window=job["parent_window"],
                parent=parent,
                candidates=candidates,
                selected=selected,
                state=state,
                fit_rows=fit_rows,
                markets=markets,
                labels_used_until=labels,
                confirmed_at_utc=datetime.now(UTC).isoformat(),
            ),
        )
        confirmed = staged_chain.read_selection(self.output, scope, window, base_arm, seed)
        _require(confirmed is not None, f"La selección de {job['id']} no se confirmó")
        self.selections[job["id"]] = confirmed

    def execute(self, order):
        """Recorrer el plan en orden y confirmar cada trabajo y cada selección."""
        for job in order:
            if self.stop.requested:
                raise Paused
            if job["kind"] == SELECT:
                self.select(job)
                continue
            identity = self.job_identity(job)
            receipt = self.confirmed(job, identity)
            if receipt is None:
                require_learning_allowed(f"el trabajo {job['id']}")
                folder = self.folder(job)
                safe_destination(folder)
                folder.mkdir(parents=True, exist_ok=True)
                try:
                    result = (self.fit if job["kind"] == FIT else self.frozen)(job, folder)
                except InterruptedError as error:
                    raise Paused from error
                receipt = self.confirm(job, identity, folder, result)
            self.equal_updates(job, receipt)
            self.receipts[job["id"]] = receipt
            if job["family"] in cm.CAMPAIGN_DESIGNS:
                freed = release_indices(self.folder(job))
                if freed:
                    self.released_indices[job["id"]] = freed
            self.publish(job, receipt)
        return "completed"


def real_views(stage, views):
    """Ediciones por mercado de las vistas reales que la campaña base ya verificó."""
    resolved = stage["campaign"]["comparison_config"]["resolved_scopes"]
    editions = {}
    for scope in stage["scopes"]:
        edition = views[scope].get("edition")
        _require(
            isinstance(edition, dict)
            and set(edition) == set(resolved[scope]["markets"])
            and all(isinstance(value, str) and len(value) == 64 for value in edition.values()),
            f"Las vistas de {scope} no conservan la edición verificada de cada mercado",
        )
        editions[scope] = dict(edition)
    return editions


def _identity(stage, views):
    campaign = stage["campaign"]
    return dict(
        schema_version=1,
        kind=RUN_KIND,
        stage_sha256=stage["sha256"],
        campaign_sha256=campaign["sha256"],
        matrix_sha256=stage["matrix_sha256"],
        input_policy=campaign["input_policy"],
        training_data=TRAINING_DATA,
        data_policy=stage["data_policy"],
        design=stage["design"],
        chain_rule=stage["chain_rule"],
        variant=campaign["variant"],
        editions=real_views(stage, views),
        views={
            scope: {window: value["sha256"] for window, value in record["windows"].items()}
            for scope, record in views.items()
        },
        code=_code(),
        final_test_opened=False,
    )


def _summary(output, identity, jobs, chains, state, status, **extra):
    planned = Counter(job["kind"] for job in jobs)
    done = Counter(job["kind"] for job in jobs if job["id"] in state.receipts)
    summary = dict(
        schema_version=1,
        kind=RUN_KIND,
        status=status,
        identity_sha256=_digest(identity),
        planned=dict(
            training_jobs=planned[FIT], prediction_jobs=planned[FROZEN], selection_jobs=len(chains)
        ),
        completed=dict(
            training_jobs=done[FIT],
            prediction_jobs=done[FROZEN],
            selection_jobs=len(state.selections),
        ),
        updates={job_id: receipt["updates"] for job_id, receipt in state.receipts.items()},
        chain={
            job_id: dict(kind=value["selected"]["kind"], job=value["selected"]["job"])
            for job_id, value in state.selections.items()
        },
        budgets=state.budgets,
        released_index_bytes=state.released_indices,
        jobs={
            job["id"]: job["id"] in state.receipts or job["id"] in state.selections
            for job in [*jobs, *chains]
        },
        final_test_opened=False,
        updated_at_utc=datetime.now(UTC).isoformat(),
        **extra,
    )
    atomic_json(output / "summary.json", summary)
    return summary


def _gpu_lease():
    from mars_titan.training.experiment_resources import GpuLease

    return GpuLease()


def run_stage(
    path, views, campaign_output, output, *, lease=None, stop=None, device="cuda:0", window=None
):
    """Ejecutar o reanudar el walk-forward por etapas sobre una campaña base confirmada.

    `window` limita la etapa a una ventana de campaña, que solo necesita la base confirmada
    de esa ventana.

    `lease` sustituye la reserva de la GPU y `device="cpu"` limita la ejecución a los
    diagnósticos de hasta 5000 filas de `run_case`. La protección del aprendizaje se
    comprueba antes de abrir fuentes y antes de cada trabajo pendiente. La variante B se
    rechaza antes de leer nada.
    """
    from mars_titan.training.checkpoints import StopRequest

    require_learning_allowed("la etapa de postentrenamiento de la campaña")
    stage = load_stage(path)
    _require(stage["design"] == STAGED, B_NOT_EXECUTED)
    jobs = plan_stage(stage)
    count_stage(stage, jobs)
    chains = plan_chain(stage, jobs)
    campaign = stage["campaign"]
    pairs = None
    if window is not None:
        jobs, chains, pairs = window_plan(campaign, jobs, chains, window)
    order = ordered_jobs(jobs, chains)
    _require(device in ("cpu", "cuda:0"), "El dispositivo debe ser cpu o cuda:0")
    _require(
        isinstance(views, dict) and set(views) == set(campaign["scopes"]),
        "Se necesitan las vistas de todos los ámbitos de la campaña base",
    )
    views = {scope: Path(value) for scope, value in views.items()}
    campaign_output, output = Path(campaign_output), Path(output)
    safe_destination(output)
    for protected in (*views.values(), campaign_output, Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)
    _, base = masked_campaign._confirmed_state(campaign["path"], views, campaign_output)
    _base_receipts(base, campaign, stage, pairs)
    identity = _identity(stage, base.views)
    output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = output / "stage.json"
        if marker.exists():
            _require(
                read_manifest(marker, 8 * 1024**2)[0] == identity,
                "La salida pertenece a otra etapa, campaña, matriz, vista o código",
            )
        else:
            _require(
                all(p.name in {".lock", "summary.json"} for p in output.iterdir()),
                "La salida sin identidad contiene artefactos ajenos",
            )
            atomic_json(marker, identity)
        signals = StopRequest() if stop is None else nullcontext(stop)
        reservation = (lease or _gpu_lease)()
        if campaign.get("numerics"):
            # La precisión de la campaña base, antes de crear cualquier modelo.
            campaign_numerics.apply(campaign["numerics"])
        state = _Stage(
            stage, base, campaign_output, output, identity, device=device, lease=None, stop=None
        )
        _summary(output, identity, jobs, chains, state, "running")
        try:
            with signals as state.stop, reservation as state.lease:
                try:
                    status = state.execute(order)
                finally:
                    state.close()
        except Paused:
            status = "paused"
        except LearningHoldError as error:
            _summary(output, identity, jobs, chains, state, "blocked", error=str(error))
            raise
        except BaseException as error:
            _summary(output, identity, jobs, chains, state, "failed", error=str(error))
            raise
        return _summary(output, identity, jobs, chains, state, status)
    finally:
        os.close(descriptor)


def _views_argument(values):
    views = {}
    for value in values or []:
        scope, _, directory = value.partition("=")
        _require(scope in comparison.SCOPES and directory, "Usa --views ÁMBITO=DIRECTORIO")
        _require(scope not in views, f"El ámbito {scope} aparece dos veces")
        views[scope] = Path(directory)
    return views


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Validar y contar trabajos sin leer datos")
    execute = commands.add_parser("run", help="Ejecutar o reanudar la etapa")
    for command in (check, execute):
        command.add_argument("--stage", type=Path, required=True)
    execute.add_argument("--views", action="append", required=True)
    execute.add_argument("--campaign-output", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--window", help="Ventana de campaña que se ejecuta")
    args = parser.parse_args(argv)
    if args.command == "check":
        result = check_stage(args.stage)
    else:
        result = run_stage(
            args.stage,
            _views_argument(args.views),
            args.campaign_output,
            args.output,
            window=args.window,
        )
        result.pop("jobs")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"completed", "checked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
