"""Materializar un panel macro diario desde adquisiciones históricas verificadas."""

import csv
import resource
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa

from .macro import calculate_macro
from .macro_acquisition import execution_catalog, iter_vintages
from .preparation import atomic_parquet
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock


def prepare_macro(
    source: Path,
    catalog_path: Path,
    output: Path,
    report_path: Path,
    *,
    market: str,
    start: str = "2000-01-01",
    end: str = "2025-03-31",
) -> dict:
    for target in (output, report_path):
        outside_source(Path("dataset"), target)
        outside_source(source, target)
    started = time.perf_counter()
    with catalog_path.open(encoding="utf-8") as stream:
        design = list(csv.DictReader(stream))
    catalog = execution_catalog(design, source)
    clock = MarketClock(market, start, end)
    rows = calculate_macro(iter_vintages(source), catalog, clock)
    atomic_parquet(output, pa.Table.from_pylist(rows))
    report = {
        "schema_version": 2,
        "market": market,
        "start": start,
        "end": end,
        "decisions": len(clock.decisions),
        "rows": len(rows),
        "computed_values": sum(row["value"] is not None for row in rows),
        "nonempty_indicators": sorted(
            {row["indicator_id"] for row in rows if row["value"] is not None}
        ),
        "missing_reasons": dict(
            Counter(row["missing_reason"] for row in rows if row["value"] is None)
        ),
        "unit_policy": "historical_native_units_no_synthetic_rebasing",
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "sha256": sha256(output),
        "catalog_sha256": sha256(catalog_path),
    }
    atomic_json(report_path, report)
    return report
