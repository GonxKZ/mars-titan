"""Preparar y ejecutar un catálogo acotado de combinaciones sobre ventanas de desarrollo."""

import argparse
import copy
import json
import math
import os
import platform
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.learning_hold import require_learning_allowed

from .combined_memory_cohort import choose_assets, project_view
from .memory_microstudy import Budget

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/combined-memory-v1.json"
PT1 = dict(
    gate_bias=dict(alpha_half_life=256.0, eta=0.15, theta=0.05),
    memory_stability=dict(alpha_floor=0.002, eta_ceiling=0.3, gradient_clip=4.0, gate_box=True),
)


def configuration():
    return json.loads(CONFIG.read_text())


def extensions(bits):
    if type(bits) is not int or not 0 <= bits < 8:
        raise ValueError("La celda factorial debe estar entre 0 y 7")
    result = copy.deepcopy(PT1) if bits & 1 else {}
    if bits & 2:
        result["memory_multiscale"] = dict(form="cascade", coupling=0.0625)
    if bits & 4:
        result["presence_modulation"] = "film_v1"
    return result


def jobs():
    """Orden fijado antes de observar resultados, con dependencias explícitas."""
    result = []

    def add(name, kind="core", seed=42, fold="fold-018", **options):
        record = dict(
            id=f"{fold}/{name}-{seed}", name=name, kind=kind, seed=seed, fold=fold, **options
        )
        result.append(record)
        return record["id"]

    for offset, seed in enumerate(configuration()["seeds"]):
        order = list(range(8))
        for bits in order[offset:] + order[:offset]:
            add(f"cell{bits}", seed=seed, bits=bits, changes=extensions(bits), factorial=True)
    for variant in ("transformer_direct", "mac_disabled", "mac_frozen", "mac_paper"):
        add(variant, variant=variant)
    add("gru_episodic", "candidate")
    for bits in (0, 7):
        add(f"cell{bits}_frozen", "reference", parent=f"fold-018/cell{bits}-42")
    for name in ("b", "c"):
        add(f"cm_v1_core_{name}", "cm_core")
    for name, core in (("b", "b"), ("bc", "c"), ("bm", "b"), ("bcm", "c")):
        add(f"cm_v1_{name}", "cm_readout", parent=f"fold-018/cm_v1_core_{core}-42")
    for bits in (0, 7):
        parent = f"fold-018/cell{bits}-42"
        for admission in ("m0_no_bank", "m1", "m2", "m3"):
            for initialization in ("pytorch_default", "zero_refinement"):
                add(
                    f"cell{bits}_{admission}_{initialization}",
                    "readout",
                    parent=parent,
                    bits=bits,
                    admission=admission,
                    initialization=initialization,
                    refinements=1,
                )
        for refinements in (2, 4):
            add(
                f"cell{bits}_m1_zero_k{refinements}",
                "readout",
                parent=parent,
                bits=bits,
                admission="m1",
                initialization="zero_refinement",
                refinements=refinements,
            )
    for bits in (0, 7):
        parent = add(f"cell{bits}_context", bits=bits, changes=extensions(bits), context=True)
        add(f"cell{bits}_context_frozen", "reference", parent=parent)
        add(
            f"cell{bits}_context_m1_zero",
            "readout",
            parent=parent,
            bits=bits,
            admission="m1",
            initialization="zero_refinement",
            refinements=1,
        )
    for seed in configuration()["seeds"]:
        add("pt1_init", seed=seed, changes=dict(gate_bias=copy.deepcopy(PT1["gate_bias"])))
    for name, bits, form, modulation in (
        ("ema", 2, "ema_bank", None),
        ("wide", 4, None, "wide_fusion_v1"),
        ("pt1_ema_film", 7, "ema_bank", "film_v1"),
        ("pt1_cascade_wide", 7, "cascade", "wide_fusion_v1"),
        ("pt1_ema_wide", 7, "ema_bank", "wide_fusion_v1"),
    ):
        changes = extensions(bits)
        if form:
            changes["memory_multiscale"]["form"] = form
        if modulation:
            changes["presence_modulation"] = modulation
        add(name, changes=changes)
    for bits in range(8):
        add(f"cell{bits}", fold="fold-015", bits=bits, changes=extensions(bits), factorial=True)
    for name in ("full_continuation", "head", "head+fusion", "head+readout+fusion"):
        add(name, "post", parent="fold-018/cell0-42")
    return result


def prepare(sources, audit, work):
    """Proyectar las cohortes por entradas pasadas y sellar recetas y lista de trabajos."""
    work, sources, audit = Path(work), Path(sources), Path(audit)
    if work.exists():
        raise FileExistsError("La preparación necesita un destino nuevo")
    config = configuration()
    rows = json.loads(audit.read_text())["rows"]
    work.mkdir(parents=True)
    cohorts = {}
    for fold, count in config["folds"].items():
        source = sources / fold / "manifest.json"
        original = json.loads(source.read_text())
        chosen = choose_assets(original, rows, fold, count)
        view = project_view(original, chosen, fold, config["epochs"])
        folder = work / fold
        folder.mkdir()
        atomic_json(folder / "view.json", view)
        atomic_json(folder / "protocol.json", view["temporal_view"]["protocol"])
        cohorts[fold] = dict(
            source_sha256=sha256(source),
            counts=view["counts"],
            assets=[a["symbol"] for a in chosen],
        )
        for name, original_recipe in (
            ("core", "titans/chronological-training-historical-masked.json"),
            ("readout", "titans/episodic-readout-historical-masked.json"),
            ("candidate", "candidate/chronological-training.json"),
        ):
            recipe = json.loads((ROOT / "configs" / original_recipe).read_text())
            recipe["recipe"].update(epochs=config["epochs"], checkpoint_updates=256)
            if "bank_capacity" in recipe["recipe"]:
                recipe["recipe"]["bank_capacity"] = config["bank_capacity"]
            if name == "core":
                recipe["recipe"]["accumulation_rows"] = None
            if name == "candidate":
                recipe["protocol"] = "protocol.json"
            atomic_json(folder / f"{name}.json", recipe)
            if name == "core":
                paper = copy.deepcopy(recipe)
                paper["predictor"]["memory_projections"] = "titans_mac_paper_projections_v2"
                atomic_json(folder / "paper.json", paper)
                recipe["predictor"]["session_context"] = "market_memory_v1"
                atomic_json(folder / "context.json", recipe)
            elif name == "readout":
                recipe["recipe"]["refinement_initialization"] = "zero_refinement"
                atomic_json(folder / "zero-readout.json", recipe)
        factorial = json.loads((ROOT / "configs/titans/cm-v1-factorial.json").read_text())
        factorial["base"].update(core_recipe="core.json", readout_recipe="readout.json")
        atomic_json(folder / "factorial.json", factorial)
    matrix = json.loads((ROOT / "configs/posttraining/adapter-matrix-v3.json").read_text())
    matrix["budget"].update(epochs=config["epochs"], seeds=[42])
    atomic_json(work / "adapters.json", matrix)
    artifacts = {str(p.relative_to(work)): sha256(p) for p in work.rglob("*.json")}
    receipt = dict(
        configuration_sha256=sha256(CONFIG),
        configuration=config,
        jobs=jobs(),
        artifacts=artifacts,
        cohorts=cohorts,
        audit_sha256=sha256(audit),
        prepared_at_utc=datetime.now(UTC).isoformat(),
        final_test_opened=False,
    )
    atomic_json(work / "prepared.json", receipt)
    return receipt


def verify_prepared(work):
    work = Path(work).resolve()
    receipt = json.loads((work / "prepared.json").read_text())
    if receipt["configuration_sha256"] != sha256(CONFIG) or receipt["jobs"] != jobs():
        raise ValueError("La configuración o los trabajos cambiaron después de preparar")
    for name, digest in receipt["artifacts"].items():
        path = (work / name).resolve()
        if not path.is_relative_to(work) or sha256(path) != digest:
            raise ValueError("Un artefacto preparado cambió o sale del estudio")
    return receipt


def remaining_seconds(work):
    config = configuration()
    costs = [
        json.loads(p.read_text())["process_seconds"]
        for p in (Path(work) / "measurements").glob("*.json")
    ]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in costs):
        raise ValueError("El coste de un intento no es finito y no negativo")
    spent = sum(costs)
    return max(0, config["max_total_seconds"] - config["reserved_technical_seconds"] - spent)


def execute(work, job, budget):
    from mars_titan.training import cm_v1_factorial as cm
    from mars_titan.training import mars_titan_walk_forward as mars
    from mars_titan.training import titans_walk_forward as titans

    folder, output = work / job["fold"], work / "runs" / job["id"]
    view = folder / "view.json"
    common = dict(
        seed=job["seed"],
        output=output,
        device="cuda:0",
        stop=budget,
        search_case=configuration()["search_case"],
        indices=work / "indices" / job["fold"],
    )
    with titans.unfused_attention():
        if job["kind"] == "core":
            variant = job.get("variant", "mac_online")
            recipe = (
                "context" if job.get("context") else "paper" if variant == "mac_paper" else "core"
            )
            return titans.run_titans_window(
                view,
                folder / "protocol.json",
                job["fold"],
                folder / f"{recipe}.json",
                variant="mac_online" if variant == "mac_paper" else variant,
                predictor_changes=job.get("changes") or None,
                **common,
            )
        if job["kind"] == "cm_core":
            return cm.run_cm_v1_core_window(
                view, job["fold"], core=job["name"], declaration=folder / "factorial.json", **common
            )
        if job["kind"] == "candidate":
            from mars_titan.training.candidate_run import load_recipe
            from mars_titan.training.candidate_walk_forward import fit_window

            recipe, document = load_recipe(
                folder / "candidate.json",
                variant="m1_k1",
                search_case=configuration()["search_case"],
            )
            return fit_window(
                view,
                output,
                recipe,
                seed=job["seed"],
                model=document["model"],
                parent_id=f"combined/{job['id']}",
                device="cuda:0",
                warmup_months=12,
                stop=budget,
            )
        parent = work / "runs" / job["parent"]
        if job["kind"] == "reference":
            from .combined_memory_reference import frozen_reference

            return frozen_reference(
                view, parent, output, seed=job["seed"], device="cuda:0", indices=common["indices"]
            )
        if job["kind"] == "cm_readout":
            return cm.run_cm_v1_window(
                view, parent, arm=job["name"], declaration=folder / "factorial.json", **common
            )
        if job["kind"] == "readout":
            recipe = "zero-readout" if job["initialization"] == "zero_refinement" else "readout"
            return mars.run_mars_titan_window(
                view,
                parent,
                folder / f"{recipe}.json",
                components=dict(episodic_bank=job["admission"], refinements=job["refinements"]),
                **common,
            )
        from mars_titan.posttraining import (
            adapter_matrix,
            chronological_matrix,
            chronological_windows,
        )

        matrix, digest = adapter_matrix.read_matrix(work / "adapters.json")
        case = next(
            item["case"]
            for item in chronological_matrix.cases(
                matrix, digest, "titans_mac", variant="mac_online"
            )
            if item["id"] == f"seed-42/{job['name']}"
        )
        return chronological_windows.run_titans_posttraining(
            parent,
            view,
            output,
            case=case,
            matrix=matrix,
            digest=digest,
            device="cuda:0",
            stop=budget,
        )


def run_job(work, job_id):
    """Una ejecución CUDA con coste acumulado de todos los intentos, incluidos los fallidos."""
    require_learning_allowed("el estudio combinado de desarrollo")
    import torch
    from torch.optim.optimizer import (
        register_optimizer_step_post_hook,
        register_optimizer_step_pre_hook,
    )

    work = Path(work).resolve()
    receipt = verify_prepared(work)
    job = next((j for j in receipt["jobs"] if j["id"] == job_id), None)
    if job is None:
        raise ValueError("El trabajo no pertenece al catálogo cerrado")
    remaining = remaining_seconds(work)
    if remaining <= 0:
        raise RuntimeError("Se agotó el presupuesto total del estudio")
    if not torch.cuda.is_available():
        raise RuntimeError("El estudio requiere cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.75, 0)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(0)
    config = configuration()
    config["max_job_seconds"] = min(config["max_job_seconds"], remaining)
    budget = Budget(config)
    before, after = (
        register_optimizer_step_pre_hook(budget.before),
        register_optimizer_step_post_hook(budget.after),
    )
    report = dict(
        job=job,
        status="failed",
        prepared_sha256=sha256(work / "prepared.json"),
        started_at_utc=datetime.now(UTC).isoformat(),
        final_test_opened=False,
        script_sha256=sha256(Path(__file__)),
        runtime=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
            device="cuda:0",
            precision="fp32_strict",
            cpu_threads=2,
            native_artifacts={
                name: sha256(Path(os.environ[name]))
                for name in ("MARS_TITAN_EPISODIC_NATIVE", "MARS_TITAN_NATIVE_LIBRARY")
                if name in os.environ
            },
        ),
    )
    try:
        result = execute(work, job, budget)
        report.update(status=result["status"], run_id=result.get("run_id"))
        return result
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        before.remove()
        after.remove()
        torch.cuda.synchronize(0)
        report.update(
            process_seconds=time.monotonic() - budget.started,
            optimizer_steps=budget.steps,
            optimizer_step_seconds=budget.step_seconds,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0),
        )
        folder = work / "measurements"
        folder.mkdir(exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        atomic_json(folder / f"{job_id.replace('/', '_')}-{stamp}.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.sources is None or args.audit is None or args.job is not None:
            parser.error("prepare necesita --sources y --audit")
        result = prepare(args.sources, args.audit, args.work)
    else:
        if args.job is None or args.sources is not None or args.audit is not None:
            parser.error("run necesita --job")
        result = run_job(args.work, args.job)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
