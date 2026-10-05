"""Recalcular una edición macro acotada sin sustituir fuentes ni resultados anteriores."""

import argparse
import resource
import tempfile
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa

from .cohort_files import safe_destination
from .macro import calculate_macro, macro_calculation_contract
from .macro_acquisition import execution_catalog, iter_vintages
from .macro_coverage import _publish_directory, _read_catalog
from .preparation import atomic_parquet
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

_SCHEMA = pa.schema(
    [
        ("prediction_at", pa.timestamp("us", tz="UTC")),
        ("indicator_id", pa.string()),
        ("value", pa.float64()),
        ("available_at", pa.timestamp("us", tz="UTC")),
        ("period_start", pa.string()),
        ("missing_reason", pa.string()),
        ("source_hashes", pa.list_(pa.string())),
        ("unit", pa.string()),
        ("seasonal_adjustment", pa.string()),
    ]
)


def recalculate_macro(
    source,
    catalog_path,
    output,
    *,
    market,
    start,
    end,
    daily_lag_policy="source_records",
    indicators=None,
):
    """Seleccionar las dependencias reales y publicar panel y recibo en un destino nuevo."""
    began = time.perf_counter()
    source, catalog_path, output = map(Path, (source, catalog_path, output))
    for path in (source, catalog_path, output):
        safe_destination(path)
    for path in (source, catalog_path, Path("dataset")):
        outside_source(path, output)
        outside_source(output, path)
    if output.exists():
        raise FileExistsError("La edición macro debe tener un destino nuevo")
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or last >= date(2024, 1, 1) or (last - first).days > 366 * 50:
        raise ValueError("El periodo no está acotado o invade la reserva final")
    contract = macro_calculation_contract(daily_lag_policy=daily_lag_policy)
    entries = _read_catalog(catalog_path)
    requested = list(entries) if indicators is None else indicators
    if (
        not isinstance(requested, list)
        or not requested
        or any(not isinstance(v, str) or v not in entries for v in requested)
        or len(set(requested)) != len(requested)
    ):
        raise ValueError("La selección no pertenece al catálogo")
    needed = set(requested)
    while True:
        extended = needed | {
            dep for name in needed for dep in entries[name]["input_ids"].split("|") if dep
        }
        if extended == needed:
            break
        needed = extended
    design = [entry for name, entry in entries.items() if name in needed]
    raw = [entry["id"] for entry in design if entry["kind"] == "raw"]
    database = source / "macro.sqlite3"
    sources = {database: sha256(database), catalog_path: sha256(catalog_path)}
    names = (
        "macro_recalculation.py",
        "macro.py",
        "macro_formulas.py",
        "macro_model_vintages.py",
        "macro_acquisition.py",
        "macro_release_contracts.py",
        "temporal.py",
    )
    code = {name: sha256(Path(__file__).with_name(name)) for name in names}
    effective = execution_catalog(design, source)
    clock = MarketClock(market, start, end)
    if not clock.decisions or len(clock.decisions) * len(needed) > 1_000_000:
        raise ValueError("La edición no tiene sesiones o supera un millón de celdas de cálculo")
    observations = iter_vintages(
        source, before=(last + timedelta(days=1)).isoformat(), indicator_ids=raw
    )
    rows = [
        row
        for row in calculate_macro(
            observations, effective, clock, daily_lag_policy=daily_lag_policy
        )
        if row["indicator_id"] in requested
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "edition"
        stage.mkdir()
        atomic_parquet(stage / "macro.parquet", pa.Table.from_pylist(rows, schema=_SCHEMA))
        report = dict(
            schema_version=1,
            status="completed",
            final_test_opened=False,
            market=market,
            start=start,
            end=end,
            rows=len(rows),
            decisions=len(clock.decisions),
            indicator_ids=sorted(requested),
            raw_dependencies=sorted(raw),
            calculation=contract,
            source_database_sha256=sources[database],
            catalog_sha256=sources[catalog_path],
            computed_values=sum(row["value"] is not None for row in rows),
            missing_reasons=dict(
                Counter(row["missing_reason"] for row in rows if row["value"] is None)
            ),
            code_sha256=code,
            sha256=sha256(stage / "macro.parquet"),
            elapsed_seconds=time.perf_counter() - began,
            process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            admission_required=True,
        )
        atomic_json(stage / "report.json", report)
        if any(sha256(path) != signature for path, signature in sources.items()) or any(
            sha256(Path(__file__).with_name(name)) != signature for name, signature in code.items()
        ):
            raise ValueError("Una fuente o el código cambió durante el cálculo")
        _publish_directory(stage, output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "catalog", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--market", choices=("US", "CN"), required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--indicators", nargs="+")
    parser.add_argument(
        "--daily-lag-policy",
        default="source_records",
        choices=("source_records", "valid_observations"),
    )
    args = parser.parse_args(argv)
    result = recalculate_macro(
        args.source,
        args.catalog,
        args.output,
        market=args.market,
        start=args.start,
        end=args.end,
        indicators=args.indicators,
        daily_lag_policy=args.daily_lag_policy,
    )
    print(f"Calculadas {result['rows']} filas en una edición nueva. Admisión todavía necesaria.")


if __name__ == "__main__":
    main()
