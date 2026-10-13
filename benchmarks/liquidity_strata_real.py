"""Comprobar con datos reales y medir los estratos de liquidez de #532.

Usa solo precios y volúmenes de la edición sin ajustar. No lee objetivos, predicciones ni el
año sellado, no carga modelos, no usa la GPU y no ejecuta pasos de optimizador:

1. Asigna cada fila de todos los activos como sesión publicada de una decisión y cuenta los
   estratos y los motivos de las filas sin clasificar por mercado y año.
2. Trunca una muestra de activos en una sesión al azar y exige que la asignación de todas las
   filas hasta el corte no cambie. Es la comprobación de que ninguna fila posterior interviene.
3. Resume el límite de participación de la etapa de políticas en cada estrato: el 1 % del
   volumen de la sesión publicada por su último cierre negociado, en moneda local.
4. Mide ``EditionLiquidity.assign`` con las decisiones de cuatro meses de todos los activos,
   con una instancia por mes, como si cada ventana leyera de nuevo la edición, y con una sola,
   como hacen la comparación y la retención con todas sus ventanas. Mide también
   ``window_extremes`` con un panel de 100.000 filas y la puntuación de los cinco estratos de
   un modelo con un panel de 625.000, ambos con errores sintéticos.

El recibo solo contiene recuentos, cuantiles, tiempos y huellas.
"""

import argparse
import json
import os
import platform
import resource
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.data.unadjusted_edition import PRICE_SCHEMA
from mars_titan.evaluation import liquidity_strata as ls
from mars_titan.evaluation import modality_strata
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.forecast_scores import score_sessions
from mars_titan.simulation.reconstructed_tape import _snap
from mars_titan.simulation.session_prices import _table

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/historical-masked-2000-comparison.json"


def _rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _arrays(liquidity, key):
    prices = _table(liquidity.root, liquidity.manifest, key, "prices.parquet", PRICE_SCHEMA)
    return dict(
        sessions=np.array(prices["session"].to_pylist(), dtype="datetime64[D]"),
        close=prices["close"].to_numpy(),
        volume=prices["volume"].to_numpy(),
        verified=prices["verified"].to_numpy(zero_copy_only=False).astype(bool),
    )


def _codes(market, arrays, section, rows=None):
    part = {k: v if rows is None else v[:rows] for k, v in arrays.items()}
    return ls.asset_codes(
        market, part["sessions"], part["close"], part["volume"], part["verified"], section
    )


def _quantiles(values):
    if not len(values):
        return None
    points = np.quantile(values, [0.01, 0.5, 0.99])
    return dict(rows=len(values), p01=points[0], median=points[1], p99=points[2])


def population(liquidity, section, participation, sample, rng):
    """Recuentos por mercado, año y estrato, límite de participación y truncamientos."""
    counts, reasons = Counter(), Counter()
    capacity = {market: {name: [] for name in ls.NAMES[:-1]} for market in ("US", "CN")}
    keys = sorted(liquidity.manifest["receipts"])
    checked = set(rng.choice(len(keys), size=min(sample, len(keys)), replace=False).tolist())
    truncations = dict(assets=0, rows=0, mismatches=0)
    kgji = None
    for index, key in enumerate(keys):
        market = key.split("/")[0]
        arrays = _arrays(liquidity, key)
        code, reason = _codes(market, arrays, section)
        years = arrays["sessions"].astype("datetime64[Y]").astype(int) + 1970
        combined, number = np.unique(years * 8 + code, return_counts=True)
        for value, total in zip(combined.tolist(), number.tolist(), strict=True):
            counts[market, value // 8, ls.NAMES[value % 8]] += total
        values, number = np.unique(reason, return_counts=True)
        for value, total in zip(values.tolist(), number.tolist(), strict=True):
            reasons[market, ls.REASONS[value]] += total
        closed, _ = _snap(market, arrays["sessions"], arrays["close"])
        traded = arrays["verified"] & (arrays["volume"] > 0)
        last = np.maximum.accumulate(np.where(traded, np.arange(len(code)), -1))
        notional = participation * arrays["volume"] * np.where(last >= 0, closed[last], np.nan)
        for value, name in enumerate(ls.NAMES[:-1]):
            selected = notional[(code == value) & arrays["verified"]]
            capacity[market][name].append(selected[np.isfinite(selected)])
        if index in checked and len(code) > section["volume_sessions"] + 1:
            cut = int(rng.integers(section["volume_sessions"], len(code)))
            prefix, _ = _codes(market, arrays, section, rows=cut)
            truncations["assets"] += 1
            truncations["rows"] += cut
            truncations["mismatches"] += int(np.count_nonzero(prefix != code[:cut]))
        if key == "US/KGJI":
            year = years == 2022
            kgji = dict(
                rows_2022=int(year.sum()),
                strata_2022=dict(Counter(ls.NAMES[v] for v in code[year].tolist())),
                close_2022=dict(min=float(closed[year].min()), max=float(closed[year].max())),
                volume_2022=dict(
                    min=float(arrays["volume"][year].min()),
                    median=float(np.median(arrays["volume"][year])),
                ),
            )
    tables = {}
    for (market, year, name), value in sorted(counts.items()):
        tables.setdefault(market, {}).setdefault(str(year), {})[name] = value
    return dict(
        strata_by_year=tables,
        reasons={f"{market}/{name}": value for (market, name), value in sorted(reasons.items())},
        participation=participation,
        capacity_local_currency={
            market: {
                name: _quantiles(np.concatenate(parts) if parts else np.array([]))
                for name, parts in by_name.items()
            }
            for market, by_name in capacity.items()
        },
        truncations=truncations,
        kgji=kgji,
    )


def _month(market, month, keys):
    """Filas (mercado, activo, instante) con las decisiones de un mes de todos los activos."""
    clock = MarketClock(market, f"{month}-01", f"{month}-28")
    moments = np.array([int(m.timestamp()) * 1_000_000 for m in clock.decisions], np.int64)
    assets = np.repeat(np.array(keys, dtype=object), len(moments))
    return np.full(len(assets), market, dtype=object), assets, np.tile(moments, len(keys))


def assignment_cost(edition, section, months):
    """Asignación de las decisiones de varios meses de todos los activos de cada mercado.

    ``fresh`` crea una instancia por mes, como si cada ventana leyera de nuevo la edición, y
    ``shared`` reutiliza una, como hacen la comparación y la retención. Las dos deben dar los
    mismos códigos.
    """
    result = {}
    for market in ("US", "CN"):
        keys = [
            key
            for key in sorted(ls.EditionLiquidity(edition, section).manifest["receipts"])
            if key.startswith(f"{market}/")
        ]
        batches = [_month(market, month, keys) for month in months]
        shared = ls.EditionLiquidity(edition, section)
        entry, outputs = {}, {}
        for mode in ("fresh", "shared"):
            times = []
            for batch in batches:
                liquidity = ls.EditionLiquidity(edition, section) if mode == "fresh" else shared
                started = time.perf_counter()
                outputs.setdefault(mode, []).append(liquidity.assign(*batch))
                times.append(time.perf_counter() - started)
            entry[mode] = dict(seconds=times, total_seconds=sum(times))
        cache = sum(array.nbytes for arrays in shared._assets.values() for array in arrays)
        codes = outputs["shared"][-1][0]
        result[market] = dict(
            months=months,
            rows_per_month=[len(batch[1]) for batch in batches],
            assets=len(keys),
            **entry,
            fresh_microseconds_per_row=entry["fresh"]["total_seconds"]
            / sum(len(batch[1]) for batch in batches)
            * 1e6,
            identical=all(
                np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                for a, b in zip(outputs["fresh"], outputs["shared"], strict=True)
            ),
            cache_bytes=int(cache),
            strata_last_month=dict(Counter(ls.NAMES[v] for v in codes.tolist())),
        )
    return result


def strata_cost(assets, sessions, repeats, rng):
    """Puntuar los cinco estratos de un modelo frente a puntuar la ventana entera.

    Usa un panel sintético con la forma del de los estratos de modalidades (2.500 activos y
    250 sesiones) y estratos al azar con las proporciones de US.
    """
    rows = assets * sessions
    moments = np.repeat(np.arange(sessions, dtype=np.int64) * 86_400_000_000, assets)
    ids = np.char.add("US/US/S", np.tile(np.arange(assets), sessions).astype(str))
    ids = np.char.add(np.char.add(ids, "/"), moments.astype(str))
    target = rng.standard_t(3, rows) * 0.02
    panel = ForecastPanel.from_columns(
        ids, np.full(rows, "US"), moments, target, 0.5 * target + rng.normal(0, 0.01, rows)
    )
    codes = rng.choice(len(ls.NAMES), rows, p=[0.82, 0.02, 0.012, 0.001, 0.147]).astype(np.int8)
    whole, strata = [], []
    for _ in range(repeats):
        started = time.perf_counter()
        score_sessions(panel, rank_ic_min_assets=30)
        whole.append(time.perf_counter() - started)
        started = time.perf_counter()
        modality_strata.score_strata(panel, codes, rank_ic_min_assets=30, names=ls.NAMES)
        strata.append(time.perf_counter() - started)
    return dict(rows=rows, whole_best_seconds=min(whole), strata_best_seconds=min(strata))


def extremes_cost(rows, sessions, count, repeats, rng):
    """``window_extremes`` sobre un panel sintético con las filas de una ventana."""
    markets = np.where(np.arange(rows) % 5 == 0, "CN", "US")
    moments = (np.arange(rows) % sessions).astype(np.int64) * 86_400_000_000
    ids = [f"{m}/{m}/S{i}/{t}" for i, (m, t) in enumerate(zip(markets, moments, strict=True))]
    panel = ForecastPanel.from_columns(
        ids, markets, moments, np.zeros(rows), rng.standard_t(3, rows) * 0.02
    )
    codes = rng.integers(0, len(ls.NAMES), rows).astype(np.int8)
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        ls.window_extremes(panel, codes, count)
        times.append(time.perf_counter() - started)
    return dict(rows=rows, sessions=sessions, count=count, best_seconds=min(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=400)
    parser.add_argument("--months", default="2022-03,2022-04,2022-05,2022-06")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    started = time.perf_counter()
    config = walk.load_config(CONFIG)
    section = config[walk.LIQUIDITY_FIELD]
    liquidity = ls.EditionLiquidity(args.edition, section)
    rng = np.random.default_rng(args.seed)
    load = os.getloadavg()
    whole = time.perf_counter()
    counted = population(liquidity, section, 0.01, args.sample, rng)
    whole = time.perf_counter() - whole
    receipt = dict(
        schema_version=1,
        kind="liquidity_strata_real_data_check",
        issue=532,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        code_sha256={
            name: sha256(ROOT / "src/mars_titan" / name)
            for name in ("evaluation/liquidity_strata.py", "evaluation/walk_forward_comparison.py")
        },
        configuration=dict(path=str(CONFIG.relative_to(ROOT)), sha256=config["sha256"]),
        declaration=section,
        edition=dict(path=str(args.edition), **liquidity.identity()),
        reads=dict(targets=False, predictions=False, final_test=False, gpu=False),
        population=counted,
        cost=dict(
            population_seconds=whole,
            assignment=assignment_cost(args.edition, section, args.months.split(",")),
            extremes=extremes_cost(100_000, 42, section["extremes"][-1], args.repeats, rng),
            strata=strata_cost(2_500, 250, args.repeats, rng),
        ),
        machine=dict(
            platform=platform.platform(),
            processor=platform.processor(),
            cpus=os.cpu_count(),
            load_average_at_start=load,
            threads=os.environ.get("OMP_NUM_THREADS"),
        ),
        peak_rss_mib=_rss_mib(),
        elapsed_seconds=time.perf_counter() - started,
    )
    atomic_json(args.output, json.loads(json.dumps(receipt, default=float)))
    print(json.dumps(dict(cost=receipt["cost"], truncations=counted["truncations"]), default=float))


if __name__ == "__main__":
    main()
