"""Motor de la criba de variantes de #742 con orígenes rodantes (#746).

- `plan` enumera los trabajos del protocolo y estima su coste sin leer datos.
- `prepare` sella en un directorio nuevo las vistas de cada origen, las recetas y el
  catálogo de trabajos a partir de las cohortes verificadas de `screening_cohorts`.
- `run` ejecuta un trabajo sellado. Comprueba la protección del aprendizaje antes de leer
  nada, exige que sus requisitos hayan terminado, cobra el intento contra el presupuesto y
  usa una GPU exclusiva. Un trabajo interrumpido se reanuda desde su carpeta.
- `queue` ejecuta con `run`, en el orden sellado, los trabajos pendientes. Un ajuste o un
  lector recorre su presupuesto y espera a las demás semillas de su grupo. Cuando todas
  esperan, la cola fija la época común del grupo (`decide_group`) y las reanuda para que
  emitan sus predicciones. Se detiene al llegar a su límite, ante el primer trabajo que no
  termina ni espera o cuando ningún trabajo pendiente puede avanzar.
- `status` resume los trabajos terminados, los que esperan, los pendientes y el tiempo cobrado.
- `rule-check` repite, solo en CPU y dentro de validación, la elección de la época con los
  primeros meses y la mide en los últimos, frente a la regla anterior por MAE y la última
  época. No abre S.
- `analyze` comprueba la protección, exige todos los trabajos terminados, verifica la huella
  de cada archivo de predicciones y escribe los contrastes, las familias y la decisión.

Las etapas de salida (A1, A2, A3, C4 y la media de semillas) no son trabajos. Se calculan
en el análisis sobre las predicciones guardadas de su fuente.
"""

import argparse
import copy
import fcntl
import functools
import json
import math
import resource
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from benchmarks.memory_microstudy import Budget
from benchmarks.refinement_study import environment_identity
from benchmarks.refinement_study_traces import attempt_cost, settle_attempt
from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation import seed_ensemble, variant_screening_analysis
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.splits import stopping_rule
from mars_titan.training import screening_cohorts, selection, variant_screening
from mars_titan.training.learning_hold import require_learning_allowed

ROOT = Path(__file__).resolve().parents[1]
KIND = "variant_screening_work_v1"
GROUP_KIND = "variant_screening_common_epoch_v1"
RULE_CHECK_KIND = "variant_screening_rule_check_v1"
# Filas aproximadas de un origen sin cohortes: 256 activos y las sesiones de cada tramo.
DEFAULT_ROWS = dict(fit=256 * 252, validation=256 * 126, calibration=256 * 63, evaluation=256 * 252)
GIB = 1024**3
QUANTILE_PREFIX = "quantile_"
# Trabajos con sus tres tramos en memoria a la vez, entre todos los orígenes. Las cinco
# semillas de la referencia de A2 caben con margen, unos 12 MB por trabajo con 256 activos.
CACHED_RUNS = 32


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def code_identity():
    """Huellas del motor y de todo el paquete, que fija el comportamiento de los trabajos."""
    paths = [Path(__file__), *sorted((ROOT / "src/mars_titan").rglob("*.py"))]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def origin_rows(cohorts=None):
    """Filas por origen para estimar el coste, de las cohortes o aproximadas.

    El ajuste de doce meses se aproxima con las filas de la evaluación, que cubre también
    un año de la misma cohorte. Los recuentos del manifiesto cuentan todo el tramo de
    entrenamiento desde 2000 y no sirven para ese ajuste acotado.
    """
    if cohorts is None:
        return {year: dict(DEFAULT_ROWS) for year in screening_cohorts.ORIGIN_FOLDS}
    rows = {}
    for year in screening_cohorts.ORIGIN_FOLDS:
        counts = read_manifest(Path(cohorts) / str(year) / "manifest.json")[0]["counts"]
        rows[year] = dict(
            fit=counts["evaluation"],
            validation=counts["validation"],
            calibration=counts["calibration"],
            evaluation=counts["evaluation"],
        )
    return rows


def plan(cohorts=None):
    protocol = variant_screening.load_protocol(ROOT)
    jobs = variant_screening.screening_jobs(protocol, ROOT)
    return dict(
        protocol_sha256=sha256(ROOT / variant_screening.PROTOCOL),
        jobs=len(jobs),
        cost=variant_screening.cost_estimate(protocol, jobs, origin_rows(cohorts)),
        pending=variant_screening.pending(protocol),
        contrasts=len(variant_screening.contrasts(protocol, ROOT)),
    )


def _multiplier(definition):
    return definition.get("updates_multiplier", 1) if definition["runner"] == "fit" else 1


def project_view(original, protocol, year, multiplier):
    """Conservar la cohorte de su origen y fijar las épocas y semillas de la criba."""
    temporal = original.get("temporal_view", {})
    _require(
        original.get("input_policy") == protocol["population"]["input_policy"]
        and original.get("final_test_opened") is False
        and len(original.get("assets", ())) == protocol["population"]["assets_per_origin"]
        and temporal.get("fold", {}).get("id") == screening_cohorts.ORIGIN_FOLDS[year]
        and temporal.get("protocol", {}).get("final_test_start") == "2024-01-01",
        "La vista no es la cohorte de la criba de su origen",
    )
    view = copy.deepcopy(original)
    walk = view["temporal_view"]["protocol"]
    walk["selection"]["max_epochs"] = protocol["fit"]["epochs"] * multiplier
    walk["seeds"] = list(protocol["seeds"])
    return view


def fit_recipe(protocol, definition):
    """Receta base con la cabeza, la pérdida y los cambios declarados del modelo."""
    document = read_manifest(ROOT / definition["base_recipe"])[0]
    if definition["head"] is not None:
        document["predictor"]["head"] = definition["head"]
    if definition["loss"] is not None:
        document["recipe"]["loss"] = definition["loss"]
    document["recipe"].update(definition["recipe_changes"])
    document["recipe"]["epochs"] = protocol["fit"]["epochs"] * definition["updates_multiplier"]
    document["walk_forward"]["fit_months"] = protocol["fit"]["fit_months"]
    document["pending"] = [
        "Receta propia de la criba de variantes (#746). No sustituye la campaña histórica."
    ]
    return document


def reader_recipe(protocol, definition):
    document = read_manifest(ROOT / definition["recipe"])[0]
    document["recipe"].update(definition["recipe_changes"])
    document["recipe"]["epochs"] = protocol["fit"]["epochs"]
    document["pending"] = [
        "Receta del lector para la criba de variantes (#746). No sustituye la campaña histórica."
    ]
    return document


def _documents(protocol, jobs, cohorts):
    """Vistas, protocolos y recetas que necesitan los trabajos, con su nombre en la carpeta."""
    documents, paths = {}, {}
    multipliers = sorted({_multiplier(job["definition"]) for job in jobs})
    for year in screening_cohorts.ORIGIN_FOLDS:
        original = read_manifest(Path(cohorts) / str(year) / "manifest.json", 8 * 1024**2)[0]
        for multiplier in multipliers:
            view = project_view(original, protocol, year, multiplier)
            documents[f"origins/{year}/view-x{multiplier}.json"] = view
            documents[f"origins/{year}/protocol-x{multiplier}.json"] = view["temporal_view"][
                "protocol"
            ]
    for job in jobs:
        definition = job["definition"]
        runner = definition["runner"]
        multiplier = _multiplier(definition)
        files = dict(
            view=f"origins/{job['origin']}/view-x{multiplier}.json",
            protocol=f"origins/{job['origin']}/protocol-x{multiplier}.json",
        )
        if runner in ("fit", "reader"):
            document = (
                fit_recipe(protocol, definition)
                if runner == "fit"
                else reader_recipe(protocol, definition)
            )
            name = f"recipes/{variant_screening._digest(document)[:16]}.json"
            documents[name] = document
            files["recipe"] = name
        paths[job["id"]] = files
    return documents, paths


def prepare(cohorts, work):
    cohorts, work = Path(cohorts).resolve(), Path(work).resolve()
    outside_source(cohorts, work)
    outside_source(work, cohorts)
    _require(not work.exists(), "La criba necesita un directorio nuevo")
    cohort_receipt = screening_cohorts.verify(cohorts)
    protocol = variant_screening.load_protocol(ROOT)
    _require(
        cohort_receipt["size"] == protocol["population"]["assets_per_origin"],
        "Las cohortes no tienen el tamaño del protocolo",
    )
    jobs = variant_screening.screening_jobs(protocol, ROOT)
    documents, paths = _documents(protocol, jobs, cohorts)
    for job in jobs:
        job["files"] = paths[job["id"]]
    work.mkdir(parents=True)
    for name, value in documents.items():
        atomic_json(work / name, value)
    declarations = sorted(
        {
            entry["declaration"]
            for entry in protocol["catalog"].values()
            if "declaration" in entry and "status" not in entry
        }
        | {
            value
            for job in jobs
            for key, value in job["definition"].items()
            if key in ("recipe", "base_recipe", "study")
        }
    )
    receipt = dict(
        schema_version=1,
        kind=KIND,
        prepared_at_utc=datetime.now(UTC).isoformat(),
        protocol=protocol,
        protocol_sha256=sha256(ROOT / variant_screening.PROTOCOL),
        cohorts=str(cohorts),
        cohorts_receipt_sha256=sha256(cohorts / "prepared.json"),
        declarations={path: sha256(ROOT / path) for path in declarations},
        jobs=jobs,
        artifacts={name: sha256(work / name) for name in documents},
        code=code_identity(),
        environment=environment_identity(),
        final_test_access=False,
        optimizer_steps=0,
    )
    atomic_json(work / "prepared.json", receipt)
    return receipt


def verify_prepared(work):
    """Rechazar cualquier cambio del protocolo, las cohortes, el código o los artefactos."""
    work = Path(work).resolve()
    receipt = read_manifest(work / "prepared.json", 64 * 1024**2)[0]
    protocol = variant_screening.load_protocol(ROOT)
    _require(
        receipt.get("schema_version") == 1
        and receipt.get("kind") == KIND
        and receipt.get("final_test_access") is False
        and receipt["protocol"] == protocol
        and receipt["protocol_sha256"] == sha256(ROOT / variant_screening.PROTOCOL)
        and receipt["code"] == code_identity()
        and receipt["environment"] == environment_identity()
        and all(sha256(ROOT / path) == digest for path, digest in receipt["declarations"].items()),
        "El protocolo, sus declaraciones, el código o el entorno de la criba han cambiado",
    )
    cohorts = Path(receipt["cohorts"])
    _require(
        sha256(cohorts / "prepared.json") == receipt["cohorts_receipt_sha256"],
        "Las cohortes de la criba han cambiado",
    )
    screening_cohorts.verify(cohorts)
    for name, digest in receipt["artifacts"].items():
        _require(sha256(work / name) == digest, f"Ha cambiado {name}")
    expected = variant_screening.screening_jobs(protocol, ROOT)
    _require(
        [{k: v for k, v in job.items() if k != "files"} for job in receipt["jobs"]] == expected,
        "El catálogo de trabajos ha cambiado",
    )
    return receipt


def _report(work, job_id, status):
    path = Path(work) / "runs" / job_id / "run.json"
    if not path.exists():
        return None
    report = read_manifest(path, 16 * 1024**2)[0]
    return report if report.get("status") == status else None


def _completed(work, job_id):
    return _report(work, job_id, "completed")


def _awaiting(work, job_id):
    return _report(work, job_id, selection.AWAIT)


def _decision(work, group):
    path = Path(work) / "groups" / f"{group}.json"
    return read_manifest(path, 4 * 1024**2)[0] if path.exists() else None


def _members(receipt, group):
    members = [job for job in receipt["jobs"] if job.get("common_group") == group]
    return sorted(members, key=lambda job: job["seed"])


def _budget(protocol, job):
    """Épocas que recorre un ajuste o un lector de la criba."""
    return protocol["fit"]["epochs"] * _multiplier(job["definition"])


def _fit_report(work, protocol, job):
    """Informe del entrenador de un trabajo que espera la época común, con su huella."""
    path = Path(work) / "runs" / job["id"] / "fit" / "run.json"
    report = read_manifest(path, 16 * 1024**2)[0]
    epochs = _budget(protocol, job)
    history = [entry for entry in report["history"] if entry["epoch"] >= 1]
    _require(
        report.get("status") == selection.AWAIT
        and [entry["epoch"] for entry in history] == list(range(1, epochs + 1))
        and sorted(report.get("epoch_states", {}), key=int)
        == [str(epoch) for epoch in range(1, epochs + 1)],
        f"El trabajo {job['id']} no espera la época común con todas sus épocas evaluadas",
    )
    return report, sha256(path), history


def _same_sessions(histories):
    """Todas las épocas de todas las semillas puntúan las mismas filas de validación."""
    cohorts = {entry["validation"]["cohort_sha256"] for history in histories for entry in history}
    _require(len(cohorts) == 1, "Las semillas del grupo no validan las mismas filas")


def _epoch_panels(work, job, epochs):
    """Predicción puntual de validación de cada época de un trabajo, con su huella."""
    report = read_manifest(Path(work) / "runs" / job["id"] / "fit" / "run.json", 16 * 1024**2)[0]
    history = [entry for entry in report["history"] if entry["epoch"] >= 1]
    _require(len(history) == epochs, f"El trabajo {job['id']} no ha evaluado todas sus épocas")
    panels = []
    for entry in history:
        record = entry["validation"]["epoch_predictions"]
        path = Path(work) / "runs" / job["id"] / record["path"]
        _require(sha256(path) == record["sha256"], f"Ha cambiado {path.name} de {job['id']}")
        panels.append(ForecastPanel.from_arrow(pq.read_table(path)))
    return panels


def combination_epoch(work, members, arm, protocol):
    """Época de una media de semillas con el IC por sesión de sus réplicas en validación.

    Cada réplica promedia por fila la predicción de validación de sus semillas con la suma
    ordenada de `seed_ensemble`. En cada época se promedian las réplicas en las sesiones que
    todas definen, como las semillas de un grupo, y la época sale de `common_epoch` con el
    mismo radio. Es una sola época para todas las réplicas del origen.
    """
    replicates = arm["definition"]["replicates"]
    by_seed = {job["seed"]: job for job in members}
    needed = sorted({seed for replicate in replicates for seed in replicate})
    _require(set(needed) <= set(by_seed), f"{arm['label']} no tiene todas sus semillas")
    epochs = _budget(protocol, members[0])
    panels = {seed: _epoch_panels(work, by_seed[seed], epochs) for seed in needed}
    minimum = protocol["contrast"]["primary"]["minimum_assets"]
    series = []
    for replicate in replicates:
        scored = []
        for epoch in range(epochs):
            items = [panels[seed][epoch] for seed in replicate]
            _require(
                len({panel.cohort_sha256 for panel in items}) == 1,
                "Las semillas de la combinación no predicen las mismas filas",
            )
            mean = seed_ensemble.ordered_mean([panel.prediction for panel in items])
            scored.append(sd.session_correlations(items[0], mean, min_assets=minimum))
        series.append(scored)
    radius = protocol["selection"]["smoothing_radius"]
    result = selection.common_epoch(selection.seed_epoch_scores(series), radius=radius)
    return dict(result, model=arm["label"], replicates=replicates)


def decide_group(work, receipt, group):
    """Fijar la época común de un grupo cuyas semillas esperan tras recorrer su presupuesto.

    La época común maximiza el IC medio de las semillas en validación, suavizado con sus
    vecinas. Si una etapa combina las semillas del modelo, su época se elige con el IC de la
    combinación y se añade a las versiones del grupo. La decisión queda escrita antes de
    reanudar ninguna semilla y no se reescribe.
    """
    work = Path(work)
    existing = _decision(work, group)
    if existing is not None:
        return existing
    protocol = receipt["protocol"]
    members = _members(receipt, group)
    _require(
        len(members) == len(protocol["seeds"]), f"El grupo {group} no tiene todas sus semillas"
    )
    _require(
        _group_waits(work, receipt, group),
        f"Alguna semilla del grupo {group} no espera la época común",
    )
    fits = {job["id"]: _fit_report(work, protocol, job) for job in members}
    histories = [history for _, _, history in fits.values()]
    _same_sessions(histories)
    series = [
        [
            (entry["validation"]["session_ic_values"], entry["validation"]["session_ic_defined"])
            for entry in history
        ]
        for history in histories
    ]
    radius = protocol["selection"]["smoothing_radius"]
    chosen = selection.common_epoch(selection.seed_epoch_scores(series), radius=radius)
    combinations = []
    if members[0]["epoch_predictions"]:
        arms = variant_screening.all_arms(protocol, ROOT)
        sources = variant_screening.combination_sources(arms)
        for label in sorted(set(members[0]["used_by"])):
            for arm in sources.get(tuple(label.split("/", 1)), ()):
                combinations.append(
                    combination_epoch(
                        work,
                        members,
                        dict(label=f"{arm.entry}/{arm.name}", definition=arm.definition),
                        protocol,
                    )
                )
    extra = set(members[0]["versions"]) | {item["epoch"] for item in combinations}
    decision = dict(
        schema_version=1,
        kind=GROUP_KIND,
        group=group,
        rule=variant_screening.SELECTION_RULE,
        members={job_id: digest for job_id, (_, digest, _) in fits.items()},
        mae_best_epochs={
            job_id: report["selection"]["best_epoch"] for job_id, (report, _, _) in fits.items()
        },
        **chosen,
        combinations=combinations,
        versions=sorted(extra - {chosen["epoch"]}),
        decided_at_utc=datetime.now(UTC).isoformat(),
        final_test_access=False,
    )
    atomic_json(work / "groups" / f"{group}.json", decision)
    return decision


def _charged(work):
    costs = [
        attempt_cost(read_manifest(path)[0]) for path in sorted((work / "attempts").glob("*.json"))
    ]
    _require(
        all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in costs),
        "El coste de un intento debe ser finito y no negativo",
    )
    return sum(costs)


def _reserve(work, receipt, job):
    budget = receipt["protocol"]["budget"]
    remaining = budget["max_total_gpu_hours"] * 3600 - _charged(work)
    if remaining <= 0:
        raise RuntimeError("Se agotó el presupuesto o hay una reserva pendiente de cierre")
    record = dict(
        accounting_version=2,
        attempt_id=uuid4().hex,
        job=job["id"],
        status="running",
        prepared_sha256=sha256(work / "prepared.json"),
        started_at_utc=datetime.now(UTC).isoformat(),
        reserved_seconds=remaining,
        attempt_limit_seconds=min(remaining, budget["max_job_seconds"]),
        actual_seconds=None,
        charged_seconds=remaining,
        closure_confirmed=False,
        final_test_opened=False,
    )
    path = work / "attempts" / f"{record['attempt_id']}.json"
    atomic_json(path, record)
    return path, record


def _value(definition, work, receipt, job):
    """Configuración del factor E3 o E4, con el control emparejado leído de su variante."""
    value = definition["value"]
    if definition["argument"] == "salience":
        from mars_titan.memory.salience_budget import SalienceBudgetConfig

        return SalienceBudgetConfig(**value)
    from mars_titan.memory.regime_evidence_gate import RegimeGateConfig

    if "matched_to" not in value:
        return RegimeGateConfig(**value)
    from mars_titan.training import regime_gate_inference

    _require(
        value["builder"] == "mars_titan.training.regime_gate_inference.matched_control",
        "El control emparejado no usa el constructor declarado",
    )
    variant = next(item for item in receipt["jobs"] if item["id"] == job["after"])
    return regime_gate_inference.matched_control(
        RegimeGateConfig(**variant["definition"]["value"]), _completed(work, job["after"])
    )


def common_options(work, receipt, job, files):
    """Época común de un ajuste o un lector: regla, puntuación, época del grupo y versiones.

    La regla es la del protocolo de su origen con la parada `common_epoch`, que conserva la
    métrica, la paciencia y el presupuesto. Hasta que el grupo decide, el trabajo recorre su
    presupuesto y espera.
    """
    protocol = receipt["protocol"]
    rule = stopping_rule(read_manifest(files["protocol"], 8 * 1024**2)[0])
    decision = _decision(work, job["common_group"])
    return dict(
        stopping=dict(rule, stopping=selection.COMMON_EPOCH),
        epoch_scoring=dict(
            min_assets=protocol["contrast"]["primary"]["minimum_assets"],
            predictions=job["epoch_predictions"],
        ),
        joint_epoch=None if decision is None else decision["epoch"],
        versions=None if decision is None else decision["versions"],
    )


def _dispatch(work, receipt, job, budget):
    """Llamar al ejecutor del trabajo con sus artefactos sellados."""
    definition = job["definition"]
    runner = definition["runner"]
    files = {name: work / path for name, path in job["files"].items()}
    output = work / "runs" / job["id"]
    indices = work / "indices" / str(job["origin"])
    parent = work / "runs" / job["parent"] if job["parent"] else None
    common = (
        common_options(work, receipt, job, files)
        if runner in variant_screening.EPOCH_RUNNERS
        else {}
    )
    if runner == "fit":
        from mars_titan.training.titans_walk_forward import run_titans_window

        return run_titans_window(
            files["view"],
            files["protocol"],
            job["fold"],
            files["recipe"],
            variant="mac_online",
            seed=job["seed"],
            output=output,
            device="cuda:0",
            indices=indices,
            stop=budget,
            search_case=definition["search_case"],
            predictor_changes=definition["predictor_changes"] or None,
            **common,
        )
    if runner in ("correction", "correction_profile"):
        from mars_titan.training.mars_titan_correction import run_correction_window

        factor = {}
        if runner == "correction_profile":
            factor[definition["argument"]] = _value(definition, work, receipt, job)
        return run_correction_window(
            files["view"],
            parent,
            ROOT / definition["recipe"],
            components=definition["components"],
            seed=job["seed"],
            output=output,
            search_case=definition["search_case"],
            device="cuda:0",
            indices=indices,
            stop=budget,
            **factor,
        )
    if runner == "reader":
        from mars_titan.training.mars_titan_walk_forward import run_mars_titan_window

        return run_mars_titan_window(
            files["view"],
            parent,
            files["recipe"],
            components=definition["components"],
            seed=job["seed"],
            output=output,
            search_case=definition["search_case"],
            device="cuda:0",
            indices=indices,
            stop=budget,
            **common,
        )
    from mars_titan.training.spaced_window import run_spaced_window

    return run_spaced_window(
        files["view"],
        parent,
        Path(receipt["cohorts"]) / str(job["origin"]) / "manifest.json",
        ROOT / definition["study"],
        arm=definition["arm"],
        output=output,
        device="cuda:0",
        indices=indices,
        stop=budget,
    )


def run_job(work, job_id):
    require_learning_allowed("la criba de variantes")
    work = Path(work).resolve()
    receipt = verify_prepared(work)
    job = next((item for item in receipt["jobs"] if item["id"] == job_id), None)
    _require(job is not None, "El trabajo no pertenece al catálogo sellado")
    done = _completed(work, job_id)
    if done is not None:
        return done
    waiting = _awaiting(work, job_id)
    if waiting is not None and _decision(work, job["common_group"]) is None:
        # Sin la época común del grupo no hay nada que hacer: no se cobra ningún intento.
        return waiting
    for field in variant_screening.REQUIRES:
        _require(
            job[field] is None or _completed(work, job[field]) is not None,
            f"El trabajo necesita antes {job[field]}",
        )
    (work / "attempts").mkdir(exist_ok=True)
    with (work / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run(work, receipt, job)
    # Cada ajuste espera una sola vez. Si volviera a esperar con la época de su grupo, la cola
    # lo repetiría y cobraría un intento tras otro.
    _require(
        waiting is None or result.get("status") != selection.AWAIT,
        "El trabajo volvió a esperar después de recibir la época común de su grupo",
    )
    return result


def _run(work, receipt, job):
    started = time.monotonic()
    path, record = _reserve(work, receipt, job)
    record["status"] = "failed"
    hooks, budget, lease, torch = [], None, None, None
    try:
        import torch
        from torch.optim.optimizer import (
            register_optimizer_step_post_hook,
            register_optimizer_step_pre_hook,
        )

        from mars_titan.training.experiment_resources import GpuLease
        from mars_titan.training.titans_walk_forward import unfused_attention

        # La ventana E2 toma su propia GPU exclusiva y no admite un segundo cerrojo.
        if job["definition"]["runner"] != "spaced":
            lease = GpuLease(max_ram=6 * GIB, max_vram=6 * GIB)
            lease.__enter__()
            record["admission"] = dict(lease.record)
        torch.cuda.set_device("cuda:0")
        budget = Budget(
            dict(
                max_job_seconds=record["attempt_limit_seconds"],
                max_optimizer_steps=receipt["protocol"]["budget"]["max_job_updates"],
            )
        )
        budget.started = started
        hooks = [
            register_optimizer_step_pre_hook(budget.before),
            register_optimizer_step_post_hook(budget.after),
        ]
        with unfused_attention():
            result = _dispatch(work, receipt, job, budget)
        if lease is not None:
            lease.check()
        record.update(status=result["status"], run_id=result.get("run_id"))
        return result
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for hook in hooks:
            hook.remove()
        record["optimizer_steps"] = budget.steps if budget is not None else 0
        record["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        try:
            if torch is not None and torch.cuda.is_initialized():
                record["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                record["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
            settle_attempt(path, record, actual_seconds=time.monotonic() - started)
        finally:
            if lease is not None:
                lease.__exit__()


def _ready(work, job):
    """Un trabajo pendiente puede avanzar si sus requisitos han terminado y, si espera la
    época común, su grupo ya la ha fijado."""
    if any(
        job[field] is not None and _completed(work, job[field]) is None
        for field in variant_screening.REQUIRES
    ):
        return False
    return _awaiting(work, job["id"]) is None or _decision(work, job["common_group"]) is not None


def _group_waits(work, receipt, group):
    return all(_awaiting(work, job["id"]) is not None for job in _members(receipt, group))


def run_queue(work, limit=None):
    """Ejecutar los trabajos pendientes en el orden sellado, que pone cada padre antes.

    Cada paso ejecuta el primer trabajo listo. Un ajuste o un lector que espera la época
    común no detiene la cola. Cuando todas las semillas de su grupo esperan, la cola fija la
    época del grupo, y como sus semillas van seguidas y antes que el resto, el paso siguiente
    ya las reanuda y libera sus estados. Un trabajo que falla detiene la cola con su error y
    queda cobrado. Uno pausado, por ejemplo por el límite de tiempo de su intento, la detiene
    sin error. El límite cuenta ejecuciones, porque un ajuste que recorre su presupuesto y
    espera ya ha consumido su entrenamiento.
    """
    require_learning_allowed("la cola de la criba de variantes")
    _require(
        limit is None or (type(limit) is int and limit >= 1), "El límite es un entero positivo"
    )
    work = Path(work).resolve()
    receipt = verify_prepared(work)
    completed, executed = [], 0
    while True:
        job = next(
            (
                item
                for item in receipt["jobs"]
                if _completed(work, item["id"]) is None and _ready(work, item)
            ),
            None,
        )
        if job is None:
            break
        if limit is not None and executed == limit:
            return dict(
                completed=completed, stopped_at=None, status="limit_reached", executed=executed
            )
        result = run_job(work, job["id"])
        executed += 1
        status = result.get("status")
        if status == selection.AWAIT:
            if _group_waits(work, receipt, job["common_group"]):
                decide_group(work, receipt, job["common_group"])
        elif status != "completed":
            return dict(completed=completed, stopped_at=job["id"], status=status, executed=executed)
        else:
            completed.append(job["id"])
    pending = [job["id"] for job in receipt["jobs"] if _completed(work, job["id"]) is None]
    return dict(
        completed=completed,
        stopped_at=None,
        status="blocked" if pending else "queued_jobs_completed",
        executed=executed,
        pending=len(pending),
    )


def status(work):
    work = Path(work).resolve()
    jobs = read_manifest(work / "prepared.json", 64 * 1024**2)[0]["jobs"]
    completed = [job["id"] for job in jobs if _completed(work, job["id"]) is not None]
    waiting = [job["id"] for job in jobs if _awaiting(work, job["id"]) is not None]
    ready = [
        job["id"]
        for job in jobs
        if job["id"] not in completed and job["id"] not in waiting and _ready(work, job)
    ]
    groups = {job["common_group"] for job in jobs if "common_group" in job}
    return dict(
        jobs=len(jobs),
        completed=len(completed),
        awaiting_common_epoch=len(waiting),
        groups=len(groups),
        decided_groups=sum(_decision(work, group) is not None for group in groups),
        ready=ready[:20],
        ready_count=len(ready),
        charged_seconds=_charged(work) if (work / "attempts").exists() else 0.0,
    )


def _panel(table):
    """Panel de una tabla de predicciones. `quantile_0025` es el nivel 0,025."""
    columns = sorted(c for c in table.column_names if c.startswith(QUANTILE_PREFIX))
    return ForecastPanel.from_arrow(
        table,
        quantile_columns=columns or None,
        levels=tuple(int(c.removeprefix(QUANTILE_PREFIX)) / 1000 for c in columns) or None,
    )


def _tables(folder, records, names, reads):
    """Tablas confirmadas de unos tramos, con la huella de cada archivo."""
    tables = {}
    for name in names:
        path = folder / records[name]["path"]
        digest = sha256(path)
        _require(digest == records[name]["sha256"], f"Ha cambiado {folder.name}/{path.name}")
        tables[name] = pq.read_table(path)
        reads[f"{folder.name}/{path.name}"] = digest
    return tables


def run_panels(folder, reads):
    """Paneles de los tres tramos de un trabajo terminado, con la huella de cada archivo."""
    report = read_manifest(folder / "run.json", 16 * 1024**2)[0]
    _require(report.get("status") == "completed", f"El trabajo {folder.name} no ha terminado")
    tables = _tables(folder, report["predictions"], variant_screening_analysis.PARTITIONS, reads)
    return {name: _panel(table) for name, table in tables.items()}


def epoch_tables(folder, epoch, reads):
    """Calibración y evaluación de un trabajo terminado con el estado de `epoch`.

    La época común está en sus predicciones principales y cualquier otra, en sus versiones.
    """
    report = read_manifest(folder / "run.json", 16 * 1024**2)[0]
    _require(report.get("status") == "completed", f"El trabajo {folder.name} no ha terminado")
    if epoch == report["fit"]["joint_stop_epoch"]:
        records = report["predictions"]
    else:
        records = report.get("versions", {}).get("predictions", {}).get(str(epoch))
        _require(records is not None, f"El trabajo {folder.name} no emite la época {epoch}")
    return _tables(folder, records, ("calibration", "evaluation"), reads)


def combination_panels(work, receipt, year, reads):
    """Paneles de cada réplica de una media de semillas en un origen, para el análisis.

    `combine(modelo, réplica)` lee la época que fijó la decisión del grupo de la fuente,
    promedia con `seed_mean` las tablas de las semillas de la réplica y devuelve sus paneles
    de calibración y evaluación. Las tablas de cada semilla se leen una vez por origen.
    """
    arms = variant_screening.all_arms(receipt["protocol"], ROOT)
    index = variant_screening_analysis.job_index(receipt["jobs"])
    # Una entrada por semilla: todas las réplicas de un origen leen la misma época.
    load = functools.lru_cache(maxsize=len(receipt["protocol"]["seeds"]))(
        lambda job_id, epoch: epoch_tables(work / "runs" / job_id, epoch, reads)
    )

    def combine(label, replicate):
        arm = arms[label]
        source = (label[0], arm.definition["source"])
        jobs = {seed: index[(source, year, seed)] for seed in replicate}
        _require(
            all(len(items) == 1 for items in jobs.values()),
            "La fuente de una media de semillas tiene un solo caso",
        )
        decision = _decision(work, jobs[replicate[0]][0]["common_group"])
        _require(decision is not None, f"El grupo de {'/'.join(source)} no ha decidido")
        found = [item for item in decision["combinations"] if item["model"] == "/".join(label)]
        _require(
            len(found) == 1 and found[0]["replicates"] == arm.definition["replicates"],
            f"La decisión del grupo no combina las réplicas declaradas de {'/'.join(label)}",
        )
        epoch = found[0]["epoch"]
        tables = {seed: load(items[0]["id"], epoch) for seed, items in jobs.items()}
        return {
            name: _panel(seed_ensemble.seed_mean({s: tables[s][name] for s in replicate})[0])
            for name in ("calibration", "evaluation")
        }

    return combine


def current_sectors(descriptions, assets):
    """Sector de las fichas actuales de cada activo, o None si la ficha no lo tiene."""
    from mars_titan.data.sector_map import UNKNOWN, SectorMap

    sectors = SectorMap.from_finmultitime(descriptions)
    result = {}
    for flow in assets:
        group = sectors.groups[sectors.group(flow)]
        result[flow] = None if group.endswith(f"/{UNKNOWN}") else group
    return result


def origin_maturity(view, input_policy):
    """Huella de las filas de cada tramo de la vista y madurez en el orden canónico del panel."""
    from mars_titan.evaluation.online_calibration import view_maturity

    tables = view_maturity(view, input_policy, sha256(view), variant_screening_analysis.PARTITIONS)
    cohorts, maturity = {}, {}
    for name, table in tables.items():
        times = table["prediction_at"].cast(pa.int64())
        ids = pc.binary_join_element_wise(table["asset_id"], times.cast(pa.string()), "/")
        panel = ForecastPanel.from_columns(
            ids,
            table["market"].to_numpy(zero_copy_only=False),
            times.to_numpy(),
            table["target"].to_numpy(),
            np.zeros(table.num_rows),
        )
        order = pc.index_in(panel.row_id, value_set=ids.cast(pa.large_string())).to_numpy()
        cohorts[name] = panel.cohort_sha256
        maturity[name] = table["matures_at"].to_numpy()[order]
    return cohorts, maturity


def _series_table(arrays):
    times = pa.array(arrays["session_time"], type=pa.timestamp("us", tz="UTC"))
    tables = []
    for (label, metric), (values, defined) in sorted(arrays["merged"].items()):
        tables.append(
            pa.table(
                dict(
                    model=pa.array(["/".join(label)] * len(values)),
                    metric=pa.array([metric] * len(values)),
                    session_time=times,
                    value=pa.array(values, type=pa.float64()),
                    defined=pa.array(defined),
                )
            )
        )
    return tables


def _emits_its_group(work, job):
    """El trabajo terminado emite la época y las versiones que fijó su grupo."""
    decision = _decision(work, job["common_group"])
    report = _completed(work, job["id"])
    return (
        decision is not None
        and report["fit"]["joint_stop_epoch"] == decision["epoch"]
        and report["versions"]["epochs"] == decision["versions"]
    )


def analyze(work, output, descriptions):
    """Analizar una criba terminada y escribir el informe y las series por sesión."""
    require_learning_allowed("el análisis de la criba de variantes")
    work, output = Path(work).resolve(), Path(output).resolve()
    receipt = verify_prepared(work)
    _require(not output.exists(), "El análisis necesita un directorio nuevo")
    missing = [job["id"] for job in receipt["jobs"] if _completed(work, job["id"]) is None]
    _require(not missing, f"Faltan {len(missing)} trabajos por terminar, como {missing[:1]}")
    for job in receipt["jobs"]:
        if "common_group" in job:
            _require(
                _emits_its_group(work, job),
                f"El trabajo {job['id']} no emite la época común y las versiones de su grupo",
            )
    protocol, reads, origins = receipt["protocol"], {}, {}
    # Una sola caché para los cuatro orígenes. Los identificadores de trabajo llevan su año.
    load = functools.lru_cache(maxsize=CACHED_RUNS)(
        lambda job_id: run_panels(work / "runs" / job_id, reads)
    )
    for year in screening_cohorts.ORIGIN_FOLDS:
        view = work / "origins" / str(year) / "view-x1.json"
        assets = [
            f"{a['market']}/{a['symbol']}" for a in read_manifest(view, 8 * 1024**2)[0]["assets"]
        ]
        cohorts, maturity = origin_maturity(view, protocol["population"]["input_policy"])
        origins[year] = variant_screening_analysis.Origin(
            year=year,
            load=load,
            cohorts=cohorts,
            maturity=maturity,
            sectors=current_sectors(descriptions, assets),
            combine=combination_panels(work, receipt, year, reads),
        )
    report, arrays = variant_screening_analysis.analyze(protocol, receipt["jobs"], origins, ROOT)
    report.update(
        prepared_sha256=sha256(work / "prepared.json"),
        protocol_sha256=receipt["protocol_sha256"],
        code=code_identity(),
        environment=environment_identity(),
        descriptions_sha256={
            name: sha256(Path(descriptions) / name)
            for name in ("sp500stock_data_description.csv", "hs300stock_data_description.csv")
        },
        predictions_sha256=dict(sorted(reads.items())),
        final_test_access=False,
        optimizer_steps=0,
    )
    output.mkdir(parents=True)
    atomic_parquet_batches(output / "series.parquet", _series_table(arrays))
    atomic_json(output / "analysis.json", report)
    return dict(
        decisions={entry: row["passes"] for entry, row in report["decisions"].items()},
        sessions=report["sessions"],
        analysis_sha256=sha256(output / "analysis.json"),
    )


def rule_check(work, output):
    """Comprobar la regla de la época común dentro de validación, sin abrir S.

    Lee el historial de validación de cada semilla de cada grupo, con su IC y su MAE por
    sesión, y escribe la comparación con la regla anterior y con la última época.
    """
    require_learning_allowed("la comprobación de la regla de la criba de variantes")
    work, output = Path(work).resolve(), Path(output).resolve()
    receipt = verify_prepared(work)
    _require(not output.exists(), "La comprobación necesita un directorio nuevo")
    protocol, reads, groups = receipt["protocol"], {}, []
    names = sorted({job["common_group"] for job in receipt["jobs"] if "common_group" in job})
    for group in names:
        members = []
        for job in _members(receipt, group):
            path = work / "runs" / job["id"] / "fit" / "run.json"
            _require(path.exists(), f"El trabajo {job['id']} no ha recorrido su presupuesto")
            report = read_manifest(path, 16 * 1024**2)[0]
            history = [entry["validation"] for entry in report["history"] if entry["epoch"] >= 1]
            _require(
                len(history) == _budget(protocol, job),
                f"El trabajo {job['id']} no ha evaluado todas sus épocas",
            )
            reads[f"{job['id']}/fit/run.json"] = sha256(path)
            members.append(history)
        groups.append((group, members))
    result = variant_screening_analysis.rule_check(
        groups,
        protocol["selection"]["rule_check"],
        radius=protocol["selection"]["smoothing_radius"],
        replicates=protocol["inference"]["replicates"],
        seed=protocol["inference"]["seed"],
        confidence=protocol["inference"]["confidence"],
    )
    report = dict(
        schema_version=1,
        kind=RULE_CHECK_KIND,
        prepared_sha256=sha256(work / "prepared.json"),
        protocol_sha256=receipt["protocol_sha256"],
        code=code_identity(),
        reads=reads,
        final_test_access=False,
        optimizer_steps=0,
        **result,
    )
    output.mkdir(parents=True)
    atomic_json(output / "rule-check.json", report)
    return dict(groups=len(groups), summary=result["summary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="Enumerar trabajos y estimar su coste")
    plan_parser.add_argument("--cohorts", type=Path)
    prepare_parser = commands.add_parser("prepare", help="Sellar vistas, recetas y trabajos")
    prepare_parser.add_argument("--cohorts", type=Path, required=True)
    prepare_parser.add_argument("--work", type=Path, required=True)
    run_parser = commands.add_parser("run", help="Ejecutar un trabajo sellado")
    run_parser.add_argument("--work", type=Path, required=True)
    run_parser.add_argument("--job", required=True)
    queue_parser = commands.add_parser("queue", help="Ejecutar en orden los trabajos pendientes")
    queue_parser.add_argument("--work", type=Path, required=True)
    queue_parser.add_argument("--limit", type=int)
    status_parser = commands.add_parser("status", help="Resumir el avance")
    status_parser.add_argument("--work", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze", help="Contrastes y decisión de la criba")
    analyze_parser.add_argument("--work", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--descriptions", type=Path, required=True)
    check_parser = commands.add_parser("rule-check", help="Comprobar la regla en validación")
    check_parser.add_argument("--work", type=Path, required=True)
    check_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(args.cohorts)
    elif args.command == "prepare":
        receipt = prepare(args.cohorts, args.work)
        result = dict(jobs=len(receipt["jobs"]), protocol_sha256=receipt["protocol_sha256"])
    elif args.command == "run":
        result = run_job(args.work, args.job)
    elif args.command == "queue":
        result = run_queue(args.work, args.limit)
    elif args.command == "analyze":
        result = analyze(args.work, args.output, args.descriptions)
    elif args.command == "rule-check":
        result = rule_check(args.work, args.output)
    else:
        result = status(args.work)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
