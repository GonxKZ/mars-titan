"""Auditorías de cobertura del universo original y permisos de admisión."""

import hashlib
import json
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from .fundamentals import read_fundamentals
from .inventory import entries
from .prices import read_prices
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock


def audit_prices(source: Path, database: Path, state_path: Path) -> dict:
    outside_source(source, state_path)
    started = time.perf_counter()
    clocks = {
        m: MarketClock(m, "1990-01-01" if m == "US" else "2000-01-01", "2026-01-01")
        for m in ("US", "CN")
    }
    policy = sha256(Path(__file__).with_name("prices.py")) + sha256(
        Path(__file__).with_name("temporal.py")
    )
    policy += json.dumps({m: [d.isoformat() for d in c.decisions] for m, c in clocks.items()})
    policy = hashlib.sha256(policy.encode()).hexdigest()
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    done = previous.get("files", {}) if previous.get("policy") == policy else {}
    current, markets = {}, defaultdict(Counter)
    for entry in entries(database):
        if entry["modality"] != "prices":
            continue
        path = source / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Ha cambiado una fuente inventariada: {entry['path']}")
        if entry["path"] in done and done[entry["path"]]["source_sha256"] == entry["sha256"]:
            audit = done[entry["path"]]
        else:
            _, audit = read_prices(path, clocks[entry["market"]])
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
        ):
            totals[key] += audit[key]
        if len(current) % 64 == 0:
            atomic_json(state_path, {"policy": policy, "files": {**done, **current}})
            print(f"Auditoría de precios: {len(current)} archivos", file=sys.stderr, flush=True)
    atomic_json(state_path, {"policy": policy, "files": current})
    return {
        "markets": dict(markets),
        "files": len(current),
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
