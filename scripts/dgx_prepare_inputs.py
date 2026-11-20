"""Preparar las entradas de la campaña en la DGX antes de lanzar los trabajos.

Por cada origen construye, con las mismas funciones que `run_titans_window`, los índices
compartidos de sus cuatro tramos, las escalas de aprendizaje del ajuste, la especificación de
entradas de cada tramo y, si se pide, la copia Arrow verificada de cada tramo. Los trabajos
encuentran después estos resultados y los verifican por identidad antes de usarlos; si algo no
coincidiera, lo recalcularían, de modo que esta preparación nunca cambia un resultado.

Uso:

    python scripts/dgx_prepare_inputs.py origin --view COHORTE/manifest.json --fold fold-018 \
        --recipe configs/titans/dgx-us-core-v1.json [--archive]
    python scripts/dgx_prepare_inputs.py campaign --state ESTADO --recipe RECETA \
        --years 2005,...,2023 --archive-years 2005,...,2012 --processes 5

`origin` necesita MARS_TITAN_SHARED_INDICES y, con --archive, MARS_TITAN_TRAIN_ARCHIVE.
`campaign` las fija por origen bajo el estado de la campaña, como hace su cola, y prepara el
primer origen solo para llenar la caché de huellas de archivos que comparten los demás.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# El origen 2005 es fold-000 en las vistas de la campaña A.
FIRST_YEAR = 2005
# Margen de disco que debe quedar libre para empezar la copia de un origen. La copia de la
# ventana mayor (2023) ocupa del orden de 50 GB y la reserva interna de la caché es de 4 GiB.
DISK_MARGIN = 200 * 1024**3
PIPELINE = {
    "MARS_TITAN_DECODE_WORKERS": "4",
    "MARS_TITAN_PREFETCH_BATCHES": "4",
    "MARS_TITAN_GROUP_CACHE_MIB": "2048",
}


def prepare_origin(view, fold, recipe, archive):
    """Índices, escalas, especificaciones y copias de un origen. Devuelve su informe."""
    from mars_titan.memory import training_event_cache as cache
    from mars_titan.training import titans_walk_forward as twf
    from mars_titan.training.learning_statistics import learning_statistics
    from mars_titan.training.window_indices import SHARED_INDICES_ENV

    if not os.environ.get(SHARED_INDICES_ENV):
        raise SystemExit(f"Falta {SHARED_INDICES_ENV}: los índices deben ir a la raíz común")
    if archive and not os.environ.get(cache.ENV):
        raise SystemExit(f"Falta {cache.ENV} para construir las copias empaquetadas")
    started = time.perf_counter()
    protocol = twf.view_protocol(view)
    protocol_document, _, rule, window = twf._protocol(protocol, fold, 42)
    chronological, document, options = twf._recipe(recipe, rule)
    dataset = twf.CorpusDataset(view, input_policy=twf.HISTORICAL_MASKED)
    twf._check_view(dataset, protocol_document, window)
    opened = time.perf_counter()
    phases = twf.window_phases(window, options["warmup_months"], options.get("fit_months"))
    root = twf.index_root(dataset, Path(os.environ[SHARED_INDICES_ENV]))
    sources = twf._sources(dataset, phases, root)
    indexed = time.perf_counter()
    scale = twf.parse_learning_scale(twf.predictor_options(document, {}).get("learning_scale"))
    statistics = None if scale is None else learning_statistics(scale, sources["train"])
    scaled = time.perf_counter()
    for source in sources.values():
        source.specification()
    specified = time.perf_counter()
    report = dict(
        fold=fold,
        view_identity=dataset.identity,
        index_root=str(root),
        sources={name: type(source).__name__ for name, source in sources.items()},
        train_events=len(sources["train"].metadata["groups"]),
        statistics_stride=None if statistics is None else statistics["stride"],
        seconds=dict(
            open=opened - started,
            indices=indexed - opened,
            statistics=scaled - indexed,
            specifications=specified - scaled,
        ),
    )
    if archive:
        # Cada tramo construye y verifica su copia en su carpeta; el primer evento basta.
        report["archive"] = {}
        for name, source in sources.items():
            begun = time.perf_counter()
            cache.reset_usage()
            events = source.batched_events(start_cursor=0, block_rows=chronological.block_rows)
            try:
                next(events)
            finally:
                events.close()
            report["archive"][name] = dict(cache.receipt(), seconds=time.perf_counter() - begun)
    report["seconds"]["total"] = time.perf_counter() - started
    return report


def _origin_command(state, recipe, year, archive):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "origin",
        "--view",
        str(state / "cohorts" / str(year) / "manifest.json"),
        "--fold",
        f"fold-{year - FIRST_YEAR:03d}",
        "--recipe",
        str(recipe),
        *(["--archive"] if archive else []),
    ]


def prepare_campaign(state, recipe, years, archive_years, processes):
    """Preparar los orígenes en orden cronológico, el primero solo y el resto en paralelo."""
    logs = state / "preparacion"
    logs.mkdir(parents=True, exist_ok=True)

    def run(year):
        archive = year in archive_years and shutil.disk_usage(state).free > DISK_MARGIN
        environment = dict(
            os.environ,
            **PIPELINE,
            MARS_TITAN_SHARED_INDICES=str(state / "indices"),
            MARS_TITAN_TRAIN_ARCHIVE=str(state / "packed-inputs" / str(year)),
            MARS_TITAN_DIGEST_CACHE=str(state / "file-digests.json"),
            PYTHONUNBUFFERED="1",
        )
        started = time.time()
        with (logs / f"{year}.log").open("w") as log:
            code = subprocess.run(
                _origin_command(state, recipe, year, archive),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            ).returncode
        line = dict(year=year, archive=archive, code=code, seconds=round(time.time() - started, 1))
        with (logs / "resumen.jsonl").open("a") as summary:
            summary.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        return code

    first, *rest = years
    codes = [run(first)]
    with ThreadPoolExecutor(processes) as pool:
        codes.extend(pool.map(run, rest))
    return 0 if not any(codes) else 1


def _years(text):
    return [int(year) for year in text.split(",") if year]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    origin = commands.add_parser("origin", help="Preparar un origen en este proceso")
    origin.add_argument("--view", type=Path, required=True)
    origin.add_argument("--fold", required=True)
    origin.add_argument("--recipe", type=Path, required=True)
    origin.add_argument("--archive", action="store_true")
    campaign = commands.add_parser("campaign", help="Preparar los orígenes de una campaña")
    campaign.add_argument("--state", type=Path, required=True)
    campaign.add_argument("--recipe", type=Path, required=True)
    campaign.add_argument("--years", type=_years, required=True)
    campaign.add_argument("--archive-years", type=_years, default=[])
    campaign.add_argument("--processes", type=int, default=4)
    args = parser.parse_args()
    if args.command == "origin":
        report = prepare_origin(args.view, args.fold, args.recipe, args.archive)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0
    if not 1 <= args.processes <= 16:
        raise SystemExit("--processes debe estar entre 1 y 16")
    return prepare_campaign(
        args.state, args.recipe, args.years, set(args.archive_years), args.processes
    )


if __name__ == "__main__":
    raise SystemExit(main())
