"""Auditorías de cobertura del universo original y permisos de admisión."""

import hashlib
import json
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa

from .fundamentals import read_fundamentals
from .inventory import entries
from .preparation import atomic_parquet
from .prices import read_prices
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock


def _write_price_details(frame, details: dict, folder: Path) -> dict:
    schemas = {
        "exclusions": pa.schema(
            [
                ("source_row", pa.int64()),
                ("source_date", pa.string()),
                ("reasons", pa.list_(pa.string())),
            ]
        ),
        "corporate_actions": pa.schema(
            [
                ("source_row", pa.int64()),
                ("source_date", pa.string()),
                ("dividends", pa.string()),
                ("stock_splits", pa.string()),
                ("values_valid", pa.bool_()),
            ]
        ),
        "coverage": pa.schema(
            [
                ("year", pa.int64()),
                ("first_observed_session", pa.string()),
                ("last_observed_session", pa.string()),
                *[
                    (name, pa.int64())
                    for name in (
                        "observed_rows",
                        "accepted_rows",
                        "excluded_rows",
                        "expected_sessions_within_observed_span",
                        "absent_sessions_within_observed_span",
                    )
                ],
            ]
        ),
    }
    tables = {
        name: pa.Table.from_pylist(details[name], schema=schema) for name, schema in schemas.items()
    }
    tables["prices"] = pa.Table.from_pandas(frame, preserve_index=False)
    artifacts = {}
    for name, table in tables.items():
        path = folder / f"{name}.parquet"
        atomic_parquet(path, table)
        artifacts[path.name] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "rows": table.num_rows,
        }
    return artifacts


def audit_prices(
    source: Path, database: Path, state_path: Path, *, details_root: Path | None = None
) -> dict:
    outside_source(source, state_path)
    if state_path.resolve() == database.resolve():
        raise ValueError("El estado no puede sobrescribir el inventario")
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    detail_location = str(details_root.resolve()) if details_root is not None else None
    if details_root is not None:
        outside_source(source, details_root)
        outside_source(details_root, database)
        outside_source(details_root, state_path)
        if details_root.exists() and not previous:
            raise ValueError("Usa un directorio nuevo o el estado de su auditoría")
    if previous and previous.get("details_root") != detail_location:
        raise ValueError("El estado pertenece a otro destino de auditoría")
    started = time.perf_counter()
    price_entries = [row for row in entries(database) if row["modality"] == "prices"]
    snapshot = hashlib.sha256(
        json.dumps(
            [(row["path"], row["sha256"]) for row in price_entries], separators=(",", ":")
        ).encode()
    ).hexdigest()
    if details_root is not None and previous and previous.get("source_snapshot") != snapshot:
        raise ValueError("La instantánea ha cambiado, usa otro estado y directorio")
    clocks = {
        m: MarketClock(m, "1990-01-01" if m == "US" else "2000-01-01", "2026-01-01")
        for m in ("US", "CN")
    }
    policy = sha256(Path(__file__).with_name("prices.py")) + sha256(
        Path(__file__).with_name("temporal.py")
    )
    policy += sha256(Path(__file__))
    policy += json.dumps({m: [d.isoformat() for d in c.decisions] for m, c in clocks.items()})
    policy = hashlib.sha256(policy.encode()).hexdigest()
    if details_root is not None and previous and previous.get("policy") != policy:
        raise ValueError("La política ha cambiado, usa otro estado y directorio")
    done = previous.get("files", {}) if previous.get("policy") == policy else {}
    checkpoint = {
        "policy": policy,
        "details_root": detail_location,
        "source_snapshot": snapshot,
        "files": done,
    }
    if details_root is not None:
        atomic_json(state_path, checkpoint)
    current, markets = {}, defaultdict(Counter)
    reused = 0
    identities = set()
    for entry in price_entries:
        path = source / entry["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("La fuente de precios sale del directorio de origen")
        key = (entry["market"], entry["symbol"])
        if key in identities:
            raise ValueError("Hay varias series para la misma identidad, deben revisarse")
        identities.add(key)
        folder = (
            details_root / entry["market"] / entry["symbol"] if details_root is not None else None
        )
        if folder is not None:
            if not folder.resolve().is_relative_to(details_root.resolve()):
                raise ValueError("La partición sale del directorio de derivados")
            outside_source(source, folder)
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Ha cambiado una fuente inventariada: {entry['path']}")
        if entry["path"] in done and done[entry["path"]]["source_sha256"] == entry["sha256"]:
            audit = done[entry["path"]]
            if folder is not None:
                for name, artifact in audit["artifacts"].items():
                    if not (folder / name).is_file() or sha256(folder / name) != artifact["sha256"]:
                        raise ValueError(f"Ha cambiado un derivado de precios: {folder / name}")
            reused += 1
        else:
            frame, audit = read_prices(
                path, clocks[entry["market"]], include_details=folder is not None
            )
            if sha256(path) != entry["sha256"]:
                raise ValueError("La fuente ha cambiado durante la auditoría")
            if folder is not None:
                audit["artifacts"] = _write_price_details(frame, audit.pop("details"), folder)
            audit = {
                **audit,
                "source_sha256": entry["sha256"],
                "market": entry["market"],
                "symbol": entry["symbol"],
                "asset_id": f"finmultitime:{entry['market']}:{entry['symbol']}",
            }
        current[entry["path"]] = audit
        totals = markets[entry["market"]]
        totals["files"] += 1
        for key in (
            "rows",
            "accepted",
            "excluded",
            "invalid_ohlc",
            "duplicate_session_rows",
            "non_session_rows",
            "invalid_date_rows",
        ):
            totals[key] += audit[key]
        if len(current) % 64 == 0:
            atomic_json(state_path, {**checkpoint, "files": {**done, **current}})
            print(f"Auditoría de precios: {len(current)} archivos", file=sys.stderr, flush=True)
    atomic_json(state_path, {**checkpoint, "files": current})
    artifacts = [
        artifact for audit in current.values() for artifact in audit.get("artifacts", {}).values()
    ]
    return {
        "markets": dict(markets),
        "files": len(current),
        "reused_files": reused,
        "artifacts": len(artifacts),
        "derived_bytes": sum(item["bytes"] for item in artifacts),
        "corporate_action_rows": sum(
            audit.get("artifacts", {}).get("corporate_actions.parquet", {}).get("rows", 0)
            for audit in current.values()
        ),
        "coverage_rows": sum(
            audit.get("artifacts", {}).get("coverage.parquet", {}).get("rows", 0)
            for audit in current.values()
        ),
        "source_snapshot": snapshot,
        "state_sha256": sha256(state_path),
        "policy": policy,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "identity_scope": "instrument_in_original_archive_not_verified_historical_constituents",
        "adjustment_limit": "original_retrospectively_adjusted_series_retained_without_repair",
    }


def audit_china_publication(source: Path, database: Path) -> dict:
    started = time.perf_counter()
    clock = MarketClock("CN", "2000-01-01", "2026-01-01")
    totals = Counter()
    for entry in entries(database):
        if entry["modality"] != "fundamentals" or entry["market"] != "CN":
            continue
        path = source / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Ha cambiado una fuente inventariada: {entry['path']}")
        _, audit = read_fundamentals([path], "CN", clock)
        totals.update(audit)
        totals["files"] += 1
    return {
        **totals,
        "elapsed_seconds": time.perf_counter() - started,
        "training_admissible": bool(totals["accepted"]),
        "rule": "period_end_does_not_establish_publication",
    }
