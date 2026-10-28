"""Motor de la criba de variantes de #742 con orígenes rodantes (#746).

- `plan` enumera los trabajos del protocolo y estima su coste sin leer datos.
- `prepare` sella en un directorio nuevo las vistas de cada origen, las recetas y el
  catálogo de trabajos a partir de las cohortes verificadas de `screening_cohorts`.
- `run` ejecuta un trabajo sellado. Comprueba la protección del aprendizaje antes de leer
  nada, exige que sus requisitos hayan terminado, cobra el intento contra el presupuesto y
  usa una GPU exclusiva. Un trabajo interrumpido se reanuda desde su carpeta.
- `status` resume los trabajos terminados, los pendientes y el tiempo cobrado.

Las etapas de salida (A1, A2, A3, C4 y la media de semillas) no son trabajos. Se calculan
en el análisis sobre las predicciones guardadas de su fuente.
"""

import argparse
import copy
import fcntl
import json
import math
import resource
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from benchmarks.memory_microstudy import Budget
from benchmarks.refinement_study import environment_identity
from benchmarks.refinement_study_traces import attempt_cost, settle_attempt
from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.training import screening_cohorts, variant_screening
from mars_titan.training.learning_hold import require_learning_allowed

ROOT = Path(__file__).resolve().parents[1]
KIND = "variant_screening_work_v1"
# Filas aproximadas de un origen sin cohortes: 256 activos y las sesiones de cada tramo.
DEFAULT_ROWS = dict(fit=256 * 252, validation=256 * 126, calibration=256 * 63, evaluation=256 * 252)
GIB = 1024**3


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


def _completed(work, job_id):
    path = Path(work) / "runs" / job_id / "run.json"
    if not path.exists():
        return None
    report = read_manifest(path, 16 * 1024**2)[0]
    return report if report.get("status") == "completed" else None


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


def _dispatch(work, receipt, job, budget):
    """Llamar al ejecutor del trabajo con sus artefactos sellados."""
    definition = job["definition"]
    runner = definition["runner"]
    files = {name: work / path for name, path in job["files"].items()}
    output = work / "runs" / job["id"]
    indices = work / "indices" / str(job["origin"])
    parent = work / "runs" / job["parent"] if job["parent"] else None
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
    for field in variant_screening.REQUIRES:
        _require(
            job[field] is None or _completed(work, job[field]) is not None,
            f"El trabajo necesita antes {job[field]}",
        )
    (work / "attempts").mkdir(exist_ok=True)
    with (work / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(work, receipt, job)


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


def status(work):
    work = Path(work).resolve()
    jobs = read_manifest(work / "prepared.json", 64 * 1024**2)[0]["jobs"]
    completed = [job["id"] for job in jobs if _completed(work, job["id"]) is not None]
    ready = [
        job["id"]
        for job in jobs
        if job["id"] not in completed
        and all(job[f] is None or job[f] in completed for f in variant_screening.REQUIRES)
    ]
    return dict(
        jobs=len(jobs),
        completed=len(completed),
        ready=ready[:20],
        ready_count=len(ready),
        charged_seconds=_charged(work) if (work / "attempts").exists() else 0.0,
    )


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
    status_parser = commands.add_parser("status", help="Resumir el avance")
    status_parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(args.cohorts)
    elif args.command == "prepare":
        receipt = prepare(args.cohorts, args.work)
        result = dict(jobs=len(receipt["jobs"]), protocol_sha256=receipt["protocol_sha256"])
    elif args.command == "run":
        result = run_job(args.work, args.job)
    else:
        result = status(args.work)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
