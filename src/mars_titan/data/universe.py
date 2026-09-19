"""Piloto de verificación elegido con cobertura anterior a un corte fijo."""

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from .inventory import entries
from .preparation import prepare_asset
from .samples import eligible_samples
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock, aware


def selection_order(candidates: list[dict], *, seed: int = 42) -> list[dict]:
    return sorted(
        candidates, key=lambda row: hashlib.sha256(f"{seed}:{row['symbol']}".encode()).hexdigest()
    )


def training_coverage(samples, macro_decisions: set[datetime], cutoff: datetime) -> int:
    cutoff = aware(cutoff)
    return sum(
        1
        for sample in samples
        if sample["prediction_at"] <= cutoff and sample["prediction_at"] in macro_decisions
    )


def select_verification_pilot(
    source: Path,
    database: Path,
    macro_path: Path,
    prepared: Path,
    output: Path,
    *,
    size: int = 4,
    selection_date: str = "2018-12-31",
    minimum_samples: int = 252,
) -> dict:
    outside_source(source, output)
    if size < 1 or minimum_samples < 1:
        raise ValueError("El tamaño del piloto y su historial mínimo deben ser positivos")
    cutoff = datetime.fromisoformat(selection_date).replace(tzinfo=UTC, hour=23, minute=59)
    grouped = {}
    for row in entries(database):
        if row["market"] != "US" or row["modality"] not in {"prices", "news", "fundamentals"}:
            continue
        if row["state"] == "error":
            continue
        asset = grouped.setdefault(
            row["symbol"], {"symbol": row["symbol"], "paths": defaultdict(list)}
        )
        asset["paths"][row["modality"]].append(row["path"])
    pool = [
        x
        for x in grouped.values()
        if all(x["paths"].get(m) for m in ("prices", "news", "fundamentals"))
    ]
    macro = pq.read_table(macro_path, columns=["prediction_at", "value"])
    known = macro.filter(pc.is_valid(macro["value"]))["prediction_at"].unique().to_pylist()
    macro_decisions = {value for value in known if value <= cutoff}
    clock = MarketClock("US", "1990-01-01", "2026-01-01")
    selected, inspected = [], []
    for asset in selection_order(pool):
        receipt = prepare_asset(source, prepared, asset, clock)
        folder = prepared / "US" / asset["symbol"]
        rows = eligible_samples(
            pq.read_table(folder / "prices.parquet").to_pandas(),
            pq.read_table(folder / "news.parquet").to_pylist(),
            pq.read_table(folder / "fundamentals.parquet").to_pylist(),
            clock,
        )
        count = training_coverage(rows, macro_decisions, cutoff)
        inspected.append(
            {
                "symbol": asset["symbol"],
                "eligible_before_cutoff": count,
                "admitted": count >= minimum_samples,
                "reason": None if count >= minimum_samples else "insufficient_past_coverage",
            }
        )
        if count >= minimum_samples:
            selected.append(
                {
                    **asset,
                    "eligible_before_cutoff": count,
                    "prepared_fingerprint": receipt["fingerprint"],
                }
            )
        if len(selected) == size:
            break
    if len(selected) != size:
        raise ValueError(
            "La cobertura original de las cuatro modalidades no basta para el piloto solicitado"
        )
    result = {
        "market": "US",
        "purpose": "verification_pilot_not_confirmatory_universe",
        "selection_cutoff": cutoff.isoformat(),
        "minimum_samples": minimum_samples,
        "seed": 42,
        "size": size,
        "order": "sha256(seed:symbol)",
        "sector_used": False,
        "future_survival_required": False,
        "universe_scope": "retrospective_original_archive_not_historical_index_membership",
        "candidate_assets": len(pool),
        "assets": selected,
        "inspected": inspected,
        "macro_sha256": sha256(macro_path),
        "implementation_sha256": sha256(Path(__file__)),
    }
    atomic_json(output, result)
    return result


if __name__ == "__main__":
    print(
        json.dumps(
            select_verification_pilot(
                Path("dataset"),
                Path("data/interim/source-inventory.sqlite"),
                Path("data/processed/phase1/macro-US.parquet"),
                Path("data/processed/pilot"),
                Path("data/manifests/pilot.json"),
            ),
            indent=2,
        )
    )
