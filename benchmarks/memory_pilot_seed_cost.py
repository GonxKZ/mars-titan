"""Medir el caudal del control de PT7 con 64 activos de EE. UU. y estimar las horas por semilla.

El piloto de memoria (PT1, PT4, PT5, PT6 y el control de semillas de PT7) usa hasta 64
activos de EE. UU., así que cada instante del ajuste cronológico tiene como mucho 64 flujos.
El coste fijo de cada actualización se reparte entre esas filas, y el caudal de la vista
completa no sirve para estimarlo.

La medida tiene dos fases para no ocupar la GPU mientras se leen datos. `prepare` se ejecuta
sin GPU: copia el manifiesto de una vista US con solo los 64 activos con más filas de ajuste
cuyos archivos siguen presentes y construye sus índices cronológicos. Las filas, los archivos
y sus huellas son los de la vista. `measure` reutiliza esos índices y recorre en `cuda:0` el
ajuste cronológico de `titans_mac_online` con la receta de 2000 y `_NoStepOptimizer`, con el
gancho que prohíbe pasos, y la inferencia congelada. Ningún peso cambia y no se registran
pérdidas ni errores, así que la medida no es una evaluación. Con los caudales medidos estima
las horas de una semilla del control en el piloto declarado.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ASSETS = 64
KINDS = ("prices", "samples", "labels")
SETTINGS = dict(segments=160, segment_warmup=10, events=512, event_warmup=32)
# La opción vacía es la receta: bloque y acumulación de 1.024, que con 64 flujos por
# instante equivalen a un único bloque por evento. La segunda los fija en 64.
OPTIONS = [{}, {"block_rows": 64, "accumulation_rows": 64}]
CAMPAIGN = Path("configs/baselines/historical-masked-campaign-a.json")
DECLARATION = Path("configs/evaluation/memory-pilot-seed-control.json")


def _commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _subset(view, output):
    """Manifiesto de la vista con los 64 activos de EE. UU. con más filas de ajuste."""
    meta = json.loads(Path(view).read_text())
    roots = meta["roots"]

    def present(asset):
        return all(
            (
                Path(roots["prepared" if kind == "prices" else kind])
                / asset["market"]
                / asset["symbol"]
                / f"{kind}.parquet"
            ).is_file()
            for kind in KINDS
        )

    us = [asset for asset in meta["assets"] if asset["market"] == "US" and present(asset)]
    if len(us) < ASSETS:
        raise ValueError("La vista no tiene 64 activos de EE. UU. con sus archivos")
    chosen = sorted(us, key=lambda asset: (-asset["counts"]["train"], asset["symbol"]))[:ASSETS]
    keys = {(asset["market"], asset["symbol"]) for asset in chosen}
    meta["assets"] = [a for a in meta["assets"] if (a["market"], a["symbol"]) in keys]
    meta["coverage"] = [a for a in meta["coverage"] if (a["market"], a["symbol"]) in keys]
    meta["counts"] = {
        name: sum(a["counts"][name] for a in meta["assets"]) for name in meta["counts"]
    }
    meta["candidate_count"] = len(meta["coverage"])
    meta["samples"] = sum(row["samples"] for row in meta["coverage"])
    meta["failed_assets"] = 0
    meta["markets"] = ["US"]
    for key in ("market_factors", "temporal_views"):
        if isinstance(meta.get(key), dict) and "US" in meta[key]:
            meta[key] = {"US": meta[key]["US"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(meta))


def _open(work):
    """Receta, caso de búsqueda medido y fuentes de la vista reducida."""
    from mars_titan.training.campaign_plan import load_campaign
    from mars_titan.training.campaign_throughput import _chronological_sources
    from mars_titan.training.financial_run import load_recipe

    campaign = load_campaign(CAMPAIGN)
    _, document = load_recipe(campaign["titans_mac"]["path"])
    subset = Path(work) / "view-64" / "manifest.json"
    fold, sources, inputs = _chronological_sources(subset, document, work)
    if max(inputs) > ASSETS:
        raise ValueError("Un evento de la vista reducida tiene más de 64 flujos")
    cases = campaign["titans_mac"]["candidates"]["titans_mac_online"]
    return campaign, document, cases, fold, sources, inputs


def prepare(view, work):
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("La fase de lectura se ejecuta sin GPU, con CUDA_VISIBLE_DEVICES=-1")
    started = time.perf_counter()
    _subset(view, Path(work) / "view-64" / "manifest.json")
    _, _, _, fold, _, inputs = _open(work)
    print(
        json.dumps(
            dict(
                window=fold["id"],
                events=len(inputs),
                rows=sum(inputs),
                seconds=time.perf_counter() - started,
            )
        )
    )


def seed_hours(train_rate, inference_rate, *, epochs, cases, warmup_months, windows):
    """Horas de GPU de una semilla del control en el piloto, con los caudales medidos.

    Cota superior de filas: 64 activos en todas las sesiones de la NYSE desde 2000 hasta el
    inicio de cada validación, `epochs` épocas por caso de búsqueda y `epochs + 2` pasadas
    de validación por caso (antes de la primera época, tras cada una y con el estado
    elegido), cada una con sus meses de calentamiento. No incluye lecturas de disco,
    índices, normalizadores ni reanudaciones.
    """
    import exchange_calendars

    if warmup_months != 12:
        raise ValueError("La estimación supone 12 meses de calentamiento")
    nyse = exchange_calendars.get_calendar("XNYS", start="2000-01-03", end="2020-12-31")

    def sessions(start, end):
        return len(nyse.sessions_in_range(start, end))

    train = sum(sessions("2000-01-03", f"{year - 1}-12-31") for year in windows) * ASSETS
    validation = sum(sessions(f"{year - 1}-01-01", f"{year}-12-31") for year in windows) * ASSETS
    fit_rows = train * epochs * cases
    inference_rows = validation * (epochs + 2) * cases
    seconds = fit_rows / train_rate + inference_rows / inference_rate
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Los caudales no permiten estimar las horas")
    return dict(
        assumptions=dict(
            assets=ASSETS,
            validation_windows=list(windows),
            training_from="2000-01-03",
            epochs=epochs,
            search_cases=cases,
            validation_passes_per_case=epochs + 2,
            warmup_months=warmup_months,
            excludes="lecturas de disco, índices, normalizadores y reanudaciones",
        ),
        train_rows_per_epoch=train,
        validation_rows_per_pass=validation,
        fit_rows_per_seed=fit_rows,
        inference_rows_per_seed=inference_rows,
        hours_per_seed=seconds / 3600,
        hours_extra_control_seeds=2 * seconds / 3600,
    )


def _gpu():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,driver_version,clocks.sm,temperature.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    )
    return out.strip()


def measure(work, output):
    import torch

    from mars_titan.training import campaign_throughput as ct
    from mars_titan.training import titans_walk_forward as titans
    from mars_titan.training.financial_run import Paused, _Pass

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible. No se cambia a CPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Falta CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(2)
    started = time.perf_counter()
    campaign, document, cases, fold, sources, inputs = _open(work)
    opened = time.perf_counter() - started
    name, search = cases[0]
    recipe = titans.case_recipe(document, search["search_case"])
    build = ct._core_builder(
        document,
        recipe,
        sources,
        work,
        variant=campaign["titans_mac"]["arms"]["titans_mac_online"],
        seed=campaign["titans_mac"]["search_seed"],
        device="cuda:0",
    )
    before = _gpu()
    guard = ct._forbid_steps()
    try:
        with titans.unfused_attention():
            began = time.perf_counter()
            record = ct._chronological(build, lambda _: _Pass(), Paused, OPTIONS, inputs, SETTINGS)
            seconds = time.perf_counter() - began
    finally:
        guard.remove()
    declared = record["options"][record["declared_option"]]
    pilot = json.loads(DECLARATION.read_text())["pilot"]
    hours = seed_hours(
        declared["train"],
        record["inference"],
        epochs=recipe.epochs,
        cases=len(cases),
        warmup_months=titans.walk_forward_options(document)["warmup_months"],
        windows=pilot["validation_windows"],
    )
    subset = Path(work) / "view-64" / "manifest.json"
    meta = json.loads(subset.read_text())
    receipt = dict(
        schema_version=1,
        kind="memory_pilot_seed_control_cost",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        commit=_commit(),
        scope=(
            "Ajuste cronológico con _NoStepOptimizer y el gancho que prohíbe pasos, e inferencia "
            "congelada. Sin pérdidas ni errores registrados."
        ),
        device="cuda:0",
        gpu_before=before,
        gpu_after=_gpu(),
        versions=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
        ),
        precision=dict(
            dtype="float32",
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            tf32_cudnn=torch.backends.cudnn.allow_tf32,
            matmul_precision=torch.get_float32_matmul_precision(),
            cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        ),
        threads=torch.get_num_threads(),
        campaign=str(CAMPAIGN),
        campaign_sha256=_sha256(CAMPAIGN),
        declaration=str(DECLARATION),
        model="titans_mac_online",
        search_case=search["search_case"],
        search_cases=[case for case, _ in cases],
        measured_case=name,
        recipe_block_rows=recipe.block_rows,
        recipe_accumulation_rows=recipe.accumulation_rows,
        inputs=dict(
            window=fold["id"],
            assets=[asset["symbol"] for asset in meta["assets"]],
            selection="64 activos de EE. UU. con más filas de ajuste y sus archivos presentes",
            events=len(inputs),
            rows=sum(inputs),
            max_event_rows=max(inputs),
            labels_saved=False,
        ),
        settings=SETTINGS,
        sources_open_seconds=opened,
        measurement_seconds=seconds,
        record=record,
        estimate=hours,
        optimizer_steps=0,
        training_runs=0,
        evaluation=False,
        final_test_opened=False,
    )
    from mars_titan.data.storage import atomic_json

    atomic_json(Path(output), receipt)
    print(json.dumps(dict(record=record, estimate=hours), default=str))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser("prepare")
    read.add_argument("--view", type=Path, required=True)
    read.add_argument("--work", type=Path, required=True)
    run = commands.add_parser("measure")
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.view, args.work)
    else:
        measure(args.work, args.output)


if __name__ == "__main__":
    main()
