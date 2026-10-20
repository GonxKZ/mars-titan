"""Pilotos neuronales con recetas selladas, costes recuperables y bloqueo por defecto."""

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
from benchmarks.refinement_study import code_identity, environment_identity
from benchmarks.refinement_study_traces import attempt_cost, settle_attempt
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.training.learning_hold import require_learning_allowed

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/neural-plasticity-study-v1.json"
VARIANTS = ROOT / "configs/experiments/neural-plasticity-v1.json"
FILES = dict(view="input/view.json", protocol="recipes/protocol.json", recipe="recipes/recipe.json")


def study_files(version):
    if version == "v1":
        return CONFIG, VARIANTS
    if version == "v2":
        return (
            ROOT / "configs/experiments/neural-plasticity-study-v2.json",
            ROOT / "configs/experiments/neural-plasticity-v2.json",
        )
    if version == "key_competition_v1":
        return (
            ROOT / "configs/experiments/key-competition-study-v1.json",
            ROOT / "configs/experiments/key-competition-v1.json",
        )
    if version == "quantile_routing_v1":
        return (
            ROOT / "configs/experiments/quantile-routing-study-v1.json",
            ROOT / "configs/experiments/quantile-routing-v1.json",
        )
    raise ValueError(
        "La versión del piloto debe ser v1, v2, key_competition_v1 o quantile_routing_v1"
    )


def configuration(version="v1"):
    config, _ = study_files(version)
    return json.loads(config.read_text())


def sources(version="v1"):
    config, variants = study_files(version)
    return code_identity() | {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            Path(__file__),
            config,
            variants,
            *sorted((ROOT / "src/mars_titan").rglob("*.py")),
        )
    }


def project_view(original, config):
    """Conservar todas las filas de la cohorte y modificar únicamente las épocas declaradas."""
    temporal = original.get("temporal_view", {})
    fold = temporal.get("fold", {})
    protocol = temporal.get("protocol", {})
    assets = original.get("assets")
    if not isinstance(assets, list) or any(not isinstance(asset, dict) for asset in assets):
        raise ValueError("La vista necesita una lista de activos identificados")
    identities = [(asset.get("market"), asset.get("symbol")) for asset in assets]
    if any(
        market != config["market"] or not isinstance(symbol, str) or not symbol
        for market, symbol in identities
    ):
        raise ValueError("La cohorte no conserva el mercado y los símbolos declarados")
    if (
        original.get("input_policy") != "historical_masked_2000_v1"
        or original.get("final_test_opened") is not False
        or len(identities) != config["cohort_size"]
        or len(set(identities)) != len(identities)
        or fold != dict(id=config["fold"], **config["window"])
        or protocol.get("final_test_start") != "2024-01-01"
        or protocol.get("primary_metric") != "session_mae"
        or protocol.get("market") != config["market"]
    ):
        raise ValueError("La vista no corresponde a la cohorte de desarrollo declarada")
    view = copy.deepcopy(original)
    view["temporal_view"]["protocol"]["selection"]["max_epochs"] = config["epochs"]
    return view


def prepare(source, work, *, version="v1"):
    source, work = Path(source).resolve(), Path(work).resolve()
    outside_source(source.parent, work)
    outside_source(work, source.parent)
    if work.exists():
        raise FileExistsError("El piloto necesita un directorio nuevo")
    config = configuration(version)
    _, catalog = study_files(version)
    variants = json.loads(catalog.read_text())
    view = project_view(json.loads(source.read_text()), config)
    recipe = json.loads((ROOT / variants["base_recipe"]).read_text())
    recipe["recipe"].update(
        epochs=config["epochs"], block_rows=config["cohort_size"], accumulation_rows=None
    )
    recipe["predictor"].update(
        max_batch=config["cohort_size"],
        memory_projections=config["memory_projections"],
        attention_layout=config["attention_layout"],
    )
    recipe["pending"] = ["Receta propia del piloto neuronal. No sustituye la campaña histórica."]
    work.mkdir(parents=True)
    for name, value in (
        (FILES["view"], view),
        (FILES["protocol"], view["temporal_view"]["protocol"]),
        (FILES["recipe"], recipe),
    ):
        atomic_json(work / name, value)
    jobs = [
        dict(id=f"{arm}-{seed}", arm=arm, seed=seed, changes=value["predictor_changes"])
        for seed in config["seeds"]
        for arm, value in variants["arms"].items()
    ]
    receipt = dict(
        schema_version=2,
        study_version=version,
        configuration=config,
        jobs=jobs,
        prepared_at_utc=datetime.now(UTC).isoformat(),
        original_view_sha256=sha256(source),
        artifacts={name: sha256(work / name) for name in FILES.values()},
        code=sources(version),
        environment=environment_identity(),
        counts=view["counts"],
        assets=[a["symbol"] for a in view["assets"]],
        final_test_access=False,
    )
    atomic_json(work / "prepared.json", receipt)
    return receipt


def verify_prepared(work):
    receipt = json.loads((work / "prepared.json").read_text())
    version = receipt.get("study_version", "v1")
    _, catalog = study_files(version)
    if (
        receipt.get("schema_version") != 2
        or receipt["configuration"] != configuration(version)
        or receipt["code"] != sources(version)
        or receipt["environment"] != environment_identity()
        or receipt.get("final_test_access") is not False
        or set(receipt["artifacts"]) != set(FILES.values())
    ):
        raise ValueError("La configuración, el código o el entorno del piloto han cambiado")
    for name, expected in receipt["artifacts"].items():
        if name not in FILES.values() or sha256(work / name) != expected:
            raise ValueError("Un artefacto preparado ha cambiado")
    config = configuration(version)
    variants = json.loads(catalog.read_text())
    expected_jobs = [
        dict(id=f"{arm}-{seed}", arm=arm, seed=seed, changes=value["predictor_changes"])
        for seed in config["seeds"]
        for arm, value in variants["arms"].items()
    ]
    if receipt["jobs"] != expected_jobs:
        raise ValueError("El catálogo de trabajos ha cambiado")
    return receipt


def _reserve(work, receipt, job):
    costs = [
        attempt_cost(json.loads(p.read_text())) for p in (work / "measurements").glob("*.json")
    ]
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in costs
    ):
        raise ValueError("El coste de un intento debe ser finito y no negativo")
    remaining = receipt["configuration"]["max_total_seconds"] - sum(costs)
    if remaining <= 0:
        raise RuntimeError("Se agotó el presupuesto o hay una reserva pendiente de cierre")
    record = dict(
        accounting_version=2,
        attempt_id=uuid4().hex,
        job=job,
        status="running",
        prepared_sha256=sha256(work / "prepared.json"),
        started_at_utc=datetime.now(UTC).isoformat(),
        reserved_seconds=remaining,
        attempt_limit_seconds=min(remaining, receipt["configuration"]["max_job_seconds"]),
        actual_seconds=None,
        charged_seconds=remaining,
        closure_confirmed=False,
        final_test_opened=False,
    )
    path = work / "measurements" / f"{record['attempt_id']}.json"
    atomic_json(path, record)
    return path, record


def run_job(work, job_id):
    require_learning_allowed("el piloto de plasticidad neuronal")
    work = Path(work).resolve()
    receipt = verify_prepared(work)
    job = next((j for j in receipt["jobs"] if j["id"] == job_id), None)
    if job is None:
        raise ValueError("El trabajo no pertenece al catálogo sellado")
    with (work / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(work, receipt, job)


def _run(work, receipt, job):
    started = time.monotonic()
    path, record = _reserve(work, receipt, job)
    record["status"] = "failed"
    before = after = budget = lease = torch = None
    try:
        import torch
        from torch.optim.optimizer import (
            register_optimizer_step_post_hook,
            register_optimizer_step_pre_hook,
        )

        from mars_titan.training.experiment_resources import GpuLease
        from mars_titan.training.titans_walk_forward import run_titans_window, unfused_attention

        lease = GpuLease(max_ram=6 * 1024**3, max_vram=6 * 1024**3)
        lease.__enter__()
        record["admission"] = dict(lease.record)
        torch.cuda.set_device("cuda:0")
        torch.set_num_threads(2)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        budget = Budget(
            dict(receipt["configuration"], max_job_seconds=record["attempt_limit_seconds"])
        )
        budget.started = started
        before = register_optimizer_step_pre_hook(budget.before)
        after = register_optimizer_step_post_hook(budget.after)
        with unfused_attention():
            result = run_titans_window(
                work / FILES["view"],
                work / FILES["protocol"],
                receipt["configuration"]["fold"],
                work / FILES["recipe"],
                variant="mac_online",
                seed=job["seed"],
                output=work / "runs" / job["id"],
                device="cuda:0",
                indices=work / "indices",
                stop=budget,
                search_case=receipt["configuration"]["search_case"],
                predictor_changes=job["changes"] or None,
            )
        lease.check()
        record.update(status=result["status"], run_id=result.get("run_id"))
        return result
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for hook in (before, after):
            if hook is not None:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--view", type=Path)
    parser.add_argument("--job")
    parser.add_argument(
        "--version", choices=("v1", "v2", "key_competition_v1", "quantile_routing_v1")
    )
    args = parser.parse_args()
    if args.command == "prepare":
        if args.view is None or args.job is not None:
            parser.error("prepare necesita --view y no admite --job")
        result = prepare(args.view, args.work, version=args.version or "v1")
    else:
        if args.job is None or args.view is not None or args.version is not None:
            parser.error("run necesita --job y toma la versión de la preparación sellada")
        result = run_job(args.work, args.job)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
