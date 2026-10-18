"""Microensayo de una ventana histórica con recetas y trabajos fijados de antemano."""

import argparse
import copy
import hashlib
import json
import os
import platform
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.training.learning_hold import require_learning_allowed

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/memory-microstudy-v1.json"
STABILITY_SHA256 = "8df0cc06cdc1f207a141e9d574a85986de01eea531ce7c65cd9716c4f81a6b01"
FOLD = dict(
    id="fold-000",
    train=["2000-01-01", "2004-04-01"],
    validation=["2004-04-01", "2004-10-01"],
    calibration=["2004-10-01", "2005-01-01"],
    evaluation=["2005-01-01", "2006-01-01"],
)


def configuration():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def jobs():
    """Lista cerrada. Los tres contrastes PT1 rotan de posición entre semillas."""
    result = []
    paired = ["mac_online", "pt1_init", "pt1"]
    for index, seed in enumerate(configuration()["seeds"]):
        for name in paired[index:] + paired[:index]:
            result.append(dict(id=f"{name}-{seed}", kind="titans", name=name, seed=seed))
    for name in ("transformer_direct", "mac_disabled", "mac_frozen", "mac_paper"):
        result.append(dict(id=f"{name}-42", kind="titans", name=name, seed=42))
    result.append(dict(id="gru_episodic-42", kind="candidate", seed=42))
    for name in ("cm_v1_core_b", "cm_v1_core_c"):
        result.append(dict(id=f"{name}-42", kind="cm_core", name=name, seed=42))
    for name in ("cm_v1_b", "cm_v1_bc", "cm_v1_bm", "cm_v1_bcm"):
        core = "c" if name in ("cm_v1_bc", "cm_v1_bcm") else "b"
        result.append(
            dict(
                id=f"{name}-42",
                kind="cm_readout",
                name=name,
                seed=42,
                parent=f"cm_v1_core_{core}-42",
            )
        )
    for name in ("m0_no_bank", "m1", "m2", "m3"):
        result.append(
            dict(id=f"mars_{name}-42", kind="mars", name=name, seed=42, parent="mac_online-42")
        )
    for name in ("full_continuation", "head", "head+fusion", "head+readout+fusion"):
        result.append(
            dict(id=f"post-{name}-42", kind="post", name=name, seed=42, parent="mac_online-42")
        )
    return result


def project_view(source):
    """Seleccionar activos por identidad y disponibilidad de ajuste, sin mirar resultados."""
    config = configuration()
    temporal = source.get("temporal_view", {})
    protocol = temporal.get("protocol", {})
    if (
        source.get("final_test_opened") is not False
        or source.get("input_policy") != "historical_masked_2000_v1"
        or protocol.get("market") != config["market"]
        or protocol.get("train_start") != "2000-01-01"
        or protocol.get("final_test_start") != "2024-01-01"
        or temporal.get("fold") != FOLD
    ):
        raise ValueError("La fuente no es la ventana histórica de desarrollo declarada")
    eligible = [a for a in source["assets"] if a["market"] == "US" and a["counts"]["train"] > 0]
    eligible.sort(
        key=lambda a: hashlib.sha256((config["asset_order_salt"] + a["symbol"]).encode()).digest()
    )
    if len(eligible) < config["assets"]:
        raise ValueError("Faltan activos con filas de ajuste para el microensayo")
    result = copy.deepcopy(source)
    result["assets"] = copy.deepcopy(eligible[: config["assets"]])
    selected = {(a["market"], a["symbol"]) for a in result["assets"]}
    result["coverage"] = [
        row for row in result["coverage"] if (row["market"], row["symbol"]) in selected
    ]
    result.update(
        scope="development_snapshot",
        cohort_complete=False,
        candidate_count=len(selected),
        samples=sum(a["samples"] for a in result["assets"]),
        training_ready=False,
    )
    result["counts"] = {
        name: sum(a["counts"][name] for a in result["assets"])
        for name in ("train", "validation", "calibration", "evaluation")
    }
    result["temporal_view"]["protocol"]["selection"]["max_epochs"] = config["epochs"]
    if "purged_by_boundary" in result:
        result["purged_by_boundary"] = {
            name: sum(a["purged_by_boundary"][name] for a in result["assets"])
            for name in result["counts"]
        }
    return result


def prepare(source, work):
    """Preparar copias de manifiestos y recetas. No abrir muestras ni ejecutar aprendizaje."""
    source, work = Path(source).resolve(), Path(work).resolve()
    outside_source(source.parent, work)
    outside_source(work, source.parent)
    if work.exists():
        raise FileExistsError("La preparación necesita una carpeta nueva")
    config = configuration()
    document = project_view(json.loads(source.read_text(encoding="utf-8")))
    document["microstudy"] = dict(
        source_manifest_sha256=sha256(source), configuration_sha256=sha256(CONFIG)
    )
    work.mkdir(parents=True)
    (work / "input").mkdir()
    (work / "recipes").mkdir()
    atomic_json(work / "input/view.json", document)
    atomic_json(work / "recipes/protocol.json", document["temporal_view"]["protocol"])
    for name, original in (
        ("core", "configs/titans/chronological-training-historical-masked.json"),
        ("readout", "configs/titans/episodic-readout-historical-masked.json"),
        ("candidate", "configs/candidate/chronological-training.json"),
    ):
        recipe = json.loads((ROOT / original).read_text())
        recipe["recipe"]["epochs"] = config["epochs"]
        recipe["recipe"]["checkpoint_updates"] = 64
        if "bank_capacity" in recipe["recipe"]:
            recipe["recipe"]["bank_capacity"] = config["bank_capacity"]
        if name == "candidate":
            recipe["protocol"] = "protocol.json"
        atomic_json(work / f"recipes/{name}.json", recipe)
        if name == "core":
            recipe["predictor"]["memory_projections"] = "titans_mac_paper_projections_v2"
            atomic_json(work / "recipes/paper.json", recipe)
    factorial = json.loads((ROOT / "configs/titans/cm-v1-factorial.json").read_text())
    factorial["base"].update(core_recipe="core.json", readout_recipe="readout.json")
    atomic_json(work / "recipes/factorial.json", factorial)
    matrix = json.loads((ROOT / "configs/posttraining/adapter-matrix-v3.json").read_text())
    matrix["budget"].update(epochs=1, seeds=[42])
    atomic_json(work / "recipes/adapters.json", matrix)
    artifacts = {
        str(path.relative_to(work)): sha256(path) for path in sorted((work / "recipes").iterdir())
    }
    artifacts["input/view.json"] = sha256(work / "input/view.json")
    receipt = dict(
        configuration=config,
        configuration_sha256=sha256(CONFIG),
        source_manifest_sha256=sha256(source),
        prepared_at_utc=datetime.now(UTC).isoformat(),
        selected_assets=[a["symbol"] for a in document["assets"]],
        counts=document["counts"],
        jobs=jobs(),
        artifacts=artifacts,
        final_test_opened=False,
    )
    atomic_json(work / "prepared.json", receipt)
    return receipt


class Budget:
    """Parar en una barrera del entrenador por tiempo o por actualizaciones observadas."""

    def __init__(self, config):
        self.config, self.started, self.steps = config, time.monotonic(), 0
        self.step_seconds = []
        self.step_started = None

    @property
    def requested(self):
        return (
            time.monotonic() - self.started >= self.config["max_job_seconds"]
            or self.steps >= self.config["max_optimizer_steps"]
        )

    def before(self, optimizer, args, kwargs):
        if self.steps >= self.config["max_optimizer_steps"]:
            raise RuntimeError("El trabajo ha agotado su presupuesto de actualizaciones")
        import torch

        torch.cuda.synchronize(0)
        self.step_started = time.perf_counter()

    def after(self, optimizer, args, kwargs):
        import torch

        torch.cuda.synchronize(0)
        self.steps += 1
        self.step_seconds.append(time.perf_counter() - self.step_started)


def _execute(work, job, budget):
    from mars_titan.training.titans_walk_forward import unfused_attention

    with unfused_attention():
        return _dispatch(work, job, budget)


def _dispatch(work, job, budget):
    from mars_titan.training import cm_v1_factorial, mars_titan_walk_forward, titans_walk_forward

    config = configuration()
    view, recipes = work / "input/view.json", work / "recipes"
    output = work / "runs" / job["id"]
    common = dict(
        seed=job["seed"],
        output=output,
        device="cuda:0",
        stop=budget,
        search_case=config["search_case"],
    )
    if job["kind"] == "titans":
        name = job["name"]
        changes = None
        if name in ("pt1", "pt1_init"):
            path = ROOT / "configs/evaluation/titans-memory-stability-comparison.json"
            if (
                path.is_symlink()
                or path.stat().st_size > 64 * 1024
                or sha256(path) != STABILITY_SHA256
            ):
                raise ValueError("La declaración PT1 no es la fijada para este microensayo")
            declaration = json.loads(path.read_text())
            changes = declaration["arms"][f"titans_mac_online_{name}"]["predictor_changes"]
        recipe = "paper" if name == "mac_paper" else "core"
        variant = (
            name if name in ("transformer_direct", "mac_disabled", "mac_frozen") else "mac_online"
        )
        return titans_walk_forward.run_titans_window(
            view,
            recipes / "protocol.json",
            config["window"],
            recipes / f"{recipe}.json",
            variant=variant,
            predictor_changes=changes,
            **common,
        )
    if job["kind"] == "cm_core":
        return cm_v1_factorial.run_cm_v1_core_window(
            view,
            config["window"],
            core=job["name"],
            declaration=recipes / "factorial.json",
            **common,
        )
    if job["kind"] == "candidate":
        from mars_titan.training.candidate_run import load_recipe
        from mars_titan.training.candidate_walk_forward import fit_window

        recipe, document = load_recipe(
            recipes / "candidate.json", variant="m1_k1", search_case=config["search_case"]
        )
        return fit_window(
            view,
            output,
            recipe,
            seed=job["seed"],
            model=document["model"],
            parent_id=f"microstudy/{job['id']}",
            device="cuda:0",
            warmup_months=12,
            stop=budget,
        )
    parent = work / "runs" / job["parent"]
    if job["kind"] == "cm_readout":
        return cm_v1_factorial.run_cm_v1_window(
            view, parent, arm=job["name"], declaration=recipes / "factorial.json", **common
        )
    if job["kind"] == "mars":
        return mars_titan_walk_forward.run_mars_titan_window(
            view,
            parent,
            recipes / "readout.json",
            components=dict(episodic_bank=job["name"]),
            **common,
        )
    from mars_titan.posttraining import adapter_matrix, chronological_matrix, chronological_windows

    matrix, digest = adapter_matrix.read_matrix(recipes / "adapters.json")
    case = next(
        item["case"]
        for item in chronological_matrix.cases(matrix, digest, "titans_mac", variant="mac_online")
        if item["id"] == f"seed-42/{job['name']}"
    )
    return chronological_windows.run_titans_posttraining(
        parent, view, output, case=case, matrix=matrix, digest=digest, device="cuda:0", stop=budget
    )


def run_job(work, job_id):
    """Ejecutar un único trabajo declarado con la protección habitual y sin fallback CPU."""
    require_learning_allowed("el microensayo de memoria")
    import torch
    from torch.optim.optimizer import (
        register_optimizer_step_post_hook,
        register_optimizer_step_pre_hook,
    )

    work = Path(work).resolve()
    prepared = json.loads((work / "prepared.json").read_text())
    if prepared["configuration_sha256"] != sha256(CONFIG) or prepared["jobs"] != jobs():
        raise ValueError("La configuración o la lista de trabajos ha cambiado")
    for name, digest in prepared["artifacts"].items():
        if sha256(work / name) != digest:
            raise ValueError(f"Ha cambiado el artefacto preparado {name}")
    job = next((job for job in jobs() if job["id"] == job_id), None)
    if job is None:
        raise ValueError("El trabajo no pertenece al microensayo")
    if not torch.cuda.is_available():
        raise RuntimeError("El microensayo necesita cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.75, 0)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.reset_peak_memory_stats(0)
    budget = Budget(configuration())
    before = register_optimizer_step_pre_hook(budget.before)
    after = register_optimizer_step_post_hook(budget.after)
    report = dict(
        job=job,
        started_at_utc=datetime.now(UTC).isoformat(),
        status="failed",
        prepared_sha256=sha256(work / "prepared.json"),
        script_sha256=sha256(Path(__file__)),
        final_test_opened=False,
        runtime=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            torch_commit=torch.version.git_version,
            cuda=torch.version.cuda,
            device="cuda:0",
            gpu=torch.cuda.get_device_name(0),
            cpu_threads=2,
            native_artifacts={
                name: sha256(Path(os.environ[name]))
                for name in ("MARS_TITAN_EPISODIC_NATIVE", "MARS_TITAN_NATIVE_LIBRARY")
                if name in os.environ
            },
        ),
    )
    try:
        result = _execute(work, job, budget)
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
        atomic_json(folder / f"{job_id}-{stamp}.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.source is None or args.job is not None:
            parser.error("prepare necesita --source y no acepta --job")
        result = prepare(args.source, args.work)
    else:
        if args.job is None or args.source is not None:
            parser.error("run necesita --job y no acepta --source")
        result = run_job(args.work, args.job)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
