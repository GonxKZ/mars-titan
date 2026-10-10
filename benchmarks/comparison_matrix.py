"""Medir tiempo y memoria de la matriz de comparaciones con tablas por sesión sintéticas.

La tabla tiene las sesiones reales del ámbito (ventanas del protocolo v2 y calendario de
cada mercado) y los brazos y semillas de la comparación declarada, con las variantes en
bruto y calibrada. Cada sesión tiene siete activos, porque el coste de la matriz depende
de las sesiones, los brazos y las familias, no de las filas. Los valores son sintéticos y
no representan resultados de mercado. ``prepare`` escribe el informe y su tabla y
``measure`` evalúa la matriz declarada en un proceso aparte con ``/usr/bin/time``.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import zlib
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.evaluation import comparison_matrix
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.forecast_scores import score_sessions
from mars_titan.models.quantile_head import LEVELS, QUANTILE_HEAD

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/evaluation/comparison-matrix-a.json"
ASSETS = 7
OFFSETS = np.array([-1.96, -1.28, 0.0, 1.28, 1.96])


def _sessions(markets, segment):
    end = str(np.datetime64(segment[1]) - np.timedelta64(1, "D"))
    return [
        (market, np.datetime64(moment.replace(tzinfo=None), "us"))
        for market in markets
        for moment in MarketClock(market, segment[0], end).decisions
    ]


def _scores(sessions, window, arm, seed, output):
    """Puntúa las sesiones sintéticas de un brazo y semilla.

    Todos los brazos comparten los objetivos de la ventana, como en la comparación real, para
    que los contrastes emparejados tengan la misma estructura que con datos de mercado.
    """
    market = np.repeat([m for m, _ in sessions], ASSETS)
    moment = np.repeat(np.array([t for _, t in sessions]), ASSETS)
    rows = len(market)
    target = np.random.default_rng(zlib.crc32(window.encode())).normal(0, 0.02, rows)
    rng = np.random.default_rng(zlib.crc32(f"{window}/{arm}/{seed}".encode()))
    prediction = np.zeros(rows) if output == walk.ZERO_CONTROL else 0.4 * target
    prediction = prediction + (0 if output == walk.ZERO_CONTROL else rng.normal(0, 0.02, rows))
    options = {}
    if output == QUANTILE_HEAD:
        width = rng.uniform(0.005, 0.03, rows)
        options = dict(quantiles=prediction[:, None] + width[:, None] * OFFSETS, levels=LEVELS)
    panel = ForecastPanel.from_columns(
        np.array([f"{window}/{i}" for i in range(rows)]),
        market,
        moment,
        target,
        prediction,
        **options,
    )
    return score_sessions(panel).to_table()


def prepare(root, scope):
    """Escribe un informe walk-forward mínimo con su tabla por sesión y las fuentes.

    El informe solo contiene los campos que comprueba la matriz, porque la medida busca el
    coste de la evaluación y no el de la comparación que lo publicaría.
    """
    matrix = comparison_matrix.load_matrix(MATRIX)
    config = matrix["comparison_config"]
    resolved = config["resolved_scopes"][scope]
    tables, windows = [], {}
    for window, fold in resolved["windows"].items():
        sessions = _sessions(resolved["markets"], fold["evaluation"])
        windows[window] = dict(
            evaluation=fold["evaluation"], view_sha256=f"{zlib.crc32(window.encode()):064x}"
        )
        for arm, spec in config["arms"].items():
            for seed in spec["seeds"] or [None]:
                table = _scores(sessions, window, arm, seed, spec["output"])
                variants = ("raw", "calibrated") if spec["output"] == QUANTILE_HEAD else ("raw",)
                for variant in variants:
                    tagged = table
                    for name, value in (
                        ("arm", arm),
                        ("seed", seed),
                        ("window", window),
                        ("quantiles", variant),
                    ):
                        tagged = tagged.append_column(name, pa.array([value] * len(table)))
                    tables.append(tagged)
    folder = root / "walk"
    folder.mkdir(parents=True)
    joined = pa.concat_tables(tables, promote_options="default")
    pq.write_table(joined, folder / "sessions.parquet", compression="zstd")
    report = dict(
        kind=walk.REPORT_KIND,
        status="completed",
        final_test_opened=False,
        scope=scope,
        markets=resolved["markets"],
        edition=dict(name="synthetic"),
        windows=windows,
        metrics=config["metrics"],
        artifacts={"sessions.parquet": sha256(folder / "sessions.parquet")},
    )
    atomic_json(folder / "comparison.json", report)
    sources = dict(
        schema_version=1,
        kind="comparison_matrix_sources",
        scope=scope,
        reports=[
            dict(
                kind=walk.REPORT_KIND,
                path=str(folder / "comparison.json"),
                sha256=sha256(folder / "comparison.json"),
            )
        ],
        hours=None,
    )
    atomic_json(root / "sources.json", sources)
    shape = dict(
        scope=scope,
        windows=len(windows),
        session_rows=joined.num_rows,
        series=sum(len(spec["seeds"]) or 1 for spec in config["arms"].values()),
        arms=len(config["arms"]),
        assets_per_session=ASSETS,
        table_bytes=(folder / "sessions.parquet").stat().st_size,
    )
    atomic_json(root / "shape.json", shape)
    return shape


def measure(root, scope, output):
    """Evalúa la matriz en un proceso aparte y registra tiempo, CPU y memoria máxima.

    El proceso aparte evita que la memoria de la preparación cuente en el pico medido.
    """
    timing = output.with_suffix(".time")
    command = [
        sys.executable,
        "-m",
        "mars_titan.evaluation.comparison_matrix",
        "evaluate",
        "--matrix",
        str(MATRIX),
        "--sources",
        str(root / "sources.json"),
        "--scope",
        scope,
        "--output",
        str(output),
    ]
    began = time.perf_counter()
    result = subprocess.run(
        ["/usr/bin/time", "-f", "%e %U %S %M", "-o", str(timing), *command],
        capture_output=True,
        text=True,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "-1"},
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr[-4000:])
    wall, user, system, rss = timing.read_text().split()[-4:]
    report = json.loads((output / "matrix.json").read_text())
    estimable = sum(len(f["estimable"]) for f in report["families"].values())
    return dict(
        wall_seconds=float(wall),
        user_seconds=float(user),
        system_seconds=float(system),
        peak_rss_bytes=int(rss) * 1024,
        harness_seconds=time.perf_counter() - began,
        report_bytes=(output / "matrix.json").stat().st_size,
        estimable_contrasts=estimable,
        families_with_contrasts=sum(bool(f["estimable"]) for f in report["families"].values()),
    )


def main():
    """Ofrece ``prepare`` y ``measure`` con raíz y salida nuevas para no mezclar medidas."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "measure"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scope", choices=tuple(walk.SCOPES), default="US")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.root.exists():
            parser.error("La raíz debe ser nueva")
        began = time.perf_counter()
        shape = prepare(args.root, args.scope)
        print(json.dumps(dict(shape, prepare_seconds=time.perf_counter() - began), indent=1))
        return
    if args.output is None or args.output.exists():
        parser.error("La medida necesita una salida nueva")
    result = measure(args.root, args.scope, args.output)
    result.update(
        shape=json.loads((args.root / "shape.json").read_text()),
        versions={name: version(name) for name in ("numpy", "pyarrow")},
        host=dict(
            platform=platform.platform(),
            cpus=os.cpu_count(),
            load_average=os.getloadavg(),
            threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        ),
        measured_at_utc=datetime.now(UTC).isoformat(),
    )
    atomic_json(args.output.with_suffix(".json"), result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
