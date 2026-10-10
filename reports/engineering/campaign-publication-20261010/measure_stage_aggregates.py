"""Bytes de los agregados por ventana de la comparación postentrenada de A v2.

Para cada padre de `configs/posttraining/historical-masked-adapter-comparison-a-v2.json` y
una ventana de US+CN, escribe con `window_aggregates.write` los agregados que guarda la fase
de agregados de la retención v2 (`stage_comparison.write_window_aggregates`) y el manifiesto
de fuentes de esa ventana. Las predicciones son sintéticas con la forma real: las filas de
calibración y evaluación de cada mercado en la ventana según
`reports/data/campaign-a-v2-window-counts-20261009.json`, las decisiones del calendario de
cada mercado, los brazos y semillas de cada padre y la cabeza de cuantiles. Ninguna sale de
un modelo, no se lee la edición ni ninguna vista real y no se ajusta nada.

Los estratos de presencia necesitan las muestras de la vista. Con `--strata` cada sesión
reparte sus filas entre los cuatro estratos, el caso con más series por estrato, así que la
medida es una cota superior. Sin `--strata` los agregados no llevan estratos.

    PYTHONPATH=src uv run --no-sync python \
        reports/engineering/campaign-publication-20261010/measure_stage_aggregates.py \
        --root <carpeta nueva> --window fold-018 --strata --output <medida.json>
"""

import argparse
import json
import os
import platform
import resource
import shutil
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES, policy_identity
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation import window_aggregates
from mars_titan.models.quantile_head import QUANTILE_COLUMNS, QUANTILE_HEAD
from mars_titan.posttraining import stage_comparison

ROOT = Path(__file__).resolve().parents[3]
DECLARATION = ROOT / "configs/posttraining/historical-masked-adapter-comparison-a-v2.json"
COUNTS = ROOT / "reports/data/campaign-a-v2-window-counts-20261009.json"
SCOPE = "US+CN"
OFFSETS = np.array([-1.96, -1.28, 0.0, 1.28, 1.96])
# Tablas distintas que comparten en ciclo los brazos y semillas, como en
# `scripts/benchmark_evaluation_scale.py`: la puntuación se repite para cada serie.
VARIANTS = 3


def market_counts(counts, window):
    """Filas de calibración y evaluación de cada mercado en una ventana de US+CN.

    China tiene sus propias ventanas, emparejadas con las últimas de US+CN.
    """
    joint, us, cn = (counts["counts"][scope] for scope in (SCOPE, "US", "CN"))
    offset = len(joint) - len(cn)
    index = int(window.split("-")[1])
    per_market = {"US": us[window]}
    if index >= offset:
        per_market["CN"] = cn[f"fold-{index - offset:03d}"]
    for partition in ("calibration", "evaluation"):
        total = sum(value[partition] for value in per_market.values())
        if total != joint[window][partition]:
            raise SystemExit(f"Los recuentos de {window} no suman los de US+CN en {partition}")
    return per_market


def decisions(market, segment):
    """Cierres del tramo sin el último, que la purga retira porque t+1 queda fuera."""
    end = str(np.datetime64(segment[1]) - np.timedelta64(1, "D"))
    return list(MarketClock(market, segment[0], end).decisions)[:-1]


def table(rows_by_market, segment, variant, seed):
    """Predicciones de un tramo con el mismo objetivo en todas las variantes."""
    parts, width = [], {}
    for market, rows in rows_by_market.items():
        moments = decisions(market, segment)
        per = np.full(len(moments), rows // len(moments))
        per[: rows % len(moments)] += 1
        width[market] = int(per.max())
        times = np.repeat([int(m.timestamp()) * 1_000_000 for m in moments], per)
        index = np.concatenate([np.arange(k) for k in per])
        assets = np.array([f"{market}/S{k:05d}" for k in range(width[market])])
        target = np.random.default_rng([seed, zlib.crc32(market.encode())]).normal(0, 0.02, rows)
        noise = np.random.default_rng([seed, variant, zlib.crc32(market.encode())])
        prediction = (0.1 + 0.05 * variant) * target + noise.normal(0.0, 0.02, rows)
        columns = dict(
            asset_id=pa.array(assets[index], pa.string()),
            market=pa.array([market] * rows, pa.string()),
            prediction_at=pa.array(times, pa.timestamp("us", tz="UTC")),
            target=target,
            prediction=prediction,
        )
        quantiles = (prediction[:, None] + 0.02 * OFFSETS).T
        columns.update(zip(QUANTILE_COLUMNS, quantiles, strict=True))
        parts.append(pa.table(columns))
    return pa.concat_tables(parts), width


def joint_view(scope, window):
    """Vista conjunta declarada de la ventana, sin muestras."""
    views = {
        market: dict(
            schema_version=2,
            **policy_identity(HISTORICAL_MASKED),
            protocol=scope["protocols"][market],
            fold=scope["windows"][window],
            selection_partition="validation",
            recover_annual_boundaries=True,
            parent_manifest="/unavailable/parent/manifest.json",
            parent_sha256="e" * 64,
        )
        for market in scope["markets"]
    }
    return dict(
        kind="corpus_supervision",
        cohort_complete=True,
        final_test_opened=False,
        **policy_identity(HISTORICAL_MASKED),
        temporal_views=views,
        markets=list(scope["markets"]),
        assets=[dict(market=market) for market in scope["markets"]],
    )


def every_stratum(sources, config, window_id, reference):
    """Presencia sintética con los cuatro estratos en cada sesión (cota superior)."""
    rows = len(reference[1].target)
    bits = np.ones((rows, len(MODALITIES)), dtype=bool)
    pattern = np.arange(rows) % 4
    bits[:, MODALITIES.index("news")] = pattern < 2
    bits[:, MODALITIES.index("fundamentals")] = pattern % 2 == 0
    bits.setflags(write=False)
    return bits, dict(rows=rows, incomplete=0)


def prepare(root, window, scope, counts):
    """Vista y tablas compartidas de la ventana, con su huella."""
    view = root / "views" / f"{window}.json"
    atomic_json(view, joint_view(scope, window))
    tables, widths = {}, {}
    segments = scope["windows"][window]
    for partition in ("calibration", "evaluation"):
        rows = {market: value[partition] for market, value in counts.items()}
        seed = zlib.crc32(f"{window}/{partition}".encode())
        for variant in range(VARIANTS):
            data, width = table(rows, segments[partition], variant, seed)
            widths[partition] = width
            path = root / "campaign" / "jobs" / SCOPE / window / f"variant-{variant}"
            path.mkdir(parents=True, exist_ok=True)
            path = path / f"{partition}-predictions.parquet"
            pq.write_table(data, path)
            tables[partition, variant] = dict(path=path, sha256=sha256(path))
    return dict(path=view, sha256=sha256(view)), tables, widths


def manifest(folder, view, tables, config, window):
    """Fuentes de un padre en la ventana, con rutas relativas como las de la etapa."""
    arms, k = {}, 0
    for name, arm in config["arms"].items():
        if arm["output"] == walk.ZERO_CONTROL:
            continue
        arms[name] = {}
        for seed in arm["seeds"]:
            entry = dict(input_policy=HISTORICAL_MASKED, view_sha256=view["sha256"])
            quantile = arm["output"] == QUANTILE_HEAD
            for part in ("calibration", "evaluation") if quantile else ("evaluation",):
                record = tables[part, k % VARIANTS]
                entry[part] = dict(
                    path=os.path.relpath(record["path"], folder), sha256=record["sha256"]
                )
            arms[name][str(seed)] = {window: entry}
            k += 1
    return dict(
        schema_version=1,
        kind=walk.SOURCES_KIND,
        scope=SCOPE,
        input_policy=HISTORICAL_MASKED,
        windows={
            window: dict(
                view=dict(path=os.path.relpath(view["path"], folder), sha256=view["sha256"])
            )
        },
        arms=arms,
    )


def measure(root, window, strata, limit=None):
    loaded = stage_comparison.load_declaration(DECLARATION)
    counts = market_counts(json.loads(COUNTS.read_text()), window)
    first = next(iter(loaded["configs"].values()))
    scope = walk.restrict_windows(first, SCOPE, [window])["resolved_scopes"][SCOPE]
    view, tables, widths = prepare(root, window, scope, counts)
    if strata:
        walk._presence_bits = every_stratum
    parents = {}
    for base_arm in list(loaded["configs"])[:limit]:
        if window not in stage_comparison.compared_windows(loaded, SCOPE, base_arm):
            continue
        config = walk.restrict_windows(loaded["configs"][base_arm], SCOPE, [window])
        scoped = walk.scope_config(config, SCOPE)
        if not strata:
            scoped = {key: value for key, value in scoped.items() if key != walk.STRATA_FIELD}
        folder = root / "adapters" / "sources" / "windows" / window / SCOPE
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{base_arm}.json"
        atomic_json(path, manifest(folder, view, tables, config, window))
        sources = walk.load_sources(path, config, SCOPE)
        began = time.perf_counter()
        record = window_aggregates.write(
            stage_comparison.aggregates_folder(root / "aggregates", base_arm),
            scoped,
            sources,
            window,
        )
        elapsed = time.perf_counter() - began
        series = sum(len(arm["seeds"]) for arm in sources["arms"].values())
        parents[base_arm] = dict(
            series=series,
            aggregate_bytes=record["bytes"],
            sources_bytes=path.stat().st_size,
            seconds=round(elapsed, 2),
        )
        print(base_arm, json.dumps(parents[base_arm]), flush=True)
    return dict(counts=counts, widths=widths, parents=parents)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--strata", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep", action="store_true", help="Conservar tablas y agregados")
    parser.add_argument("--parents", type=int, help="Medir solo los primeros padres (ensayo)")
    args = parser.parse_args()
    if args.root.exists():
        parser.error("La raíz debe ser nueva")
    began = time.perf_counter()
    result = measure(args.root, args.window, args.strata, args.parents)
    parents = result["parents"]
    per_series = {
        name: (item["aggregate_bytes"] + item["sources_bytes"]) / item["series"]
        for name, item in parents.items()
    }
    document = dict(
        schema_version=1,
        kind="stage_comparison_window_aggregates_measure",
        declaration=str(DECLARATION.relative_to(ROOT)),
        declaration_sha256=sha256(DECLARATION),
        counts=str(COUNTS.relative_to(ROOT)),
        scope=SCOPE,
        window=args.window,
        strata="every_stratum_in_every_session" if args.strata else "without_strata",
        rows=result["counts"],
        assets_per_session_max=result["widths"],
        parents=parents,
        series=sum(item["series"] for item in parents.values()),
        aggregate_bytes=sum(item["aggregate_bytes"] for item in parents.values()),
        sources_bytes=sum(item["sources_bytes"] for item in parents.values()),
        bytes_per_series_max=max(per_series.values()),
        bytes_per_series_min=min(per_series.values()),
        seconds=round(time.perf_counter() - began, 1),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        host=dict(
            platform=platform.platform(),
            python=platform.python_version(),
            numpy=np.__version__,
            pyarrow=pa.__version__,
            threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            load_average=os.getloadavg(),
        ),
        measured_at_utc=datetime.now(UTC).isoformat(),
        training_executed=False,
        final_test_opened=False,
    )
    atomic_json(args.output, document)
    if not args.keep:
        shutil.rmtree(args.root)
    print(json.dumps({k: v for k, v in document.items() if k != "parents"}, indent=1))


if __name__ == "__main__":
    main()
