"""Hechos contables por publicación. Los envoltorios no fechan las cifras internas."""

import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import ijson
from ijson.common import ObjectBuilder

from .temporal import MarketClock, aware


def _us_facts(path: Path):
    with path.open("rb") as stream:
        builder, level, identity = None, 0, None
        for prefix, event, value in ijson.parse(stream, use_float=True):
            parts = prefix.split(".")
            if (
                builder is None
                and event == "start_map"
                and len(parts) == 8
                and parts[:3] == ["filings", "item", "facts"]
                and parts[5] == "units"
                and parts[7] == "item"
            ):
                builder, level, identity = ObjectBuilder(), 0, (parts[3], parts[4], parts[6])
            if builder is not None:
                builder.event(event, value)
                if event in {"start_map", "start_array"}:
                    level += 1
                elif event in {"end_map", "end_array"}:
                    level -= 1
                if level == 0:
                    yield identity, builder.value
                    builder = None


def read_fundamentals(
    paths: list[Path], market: str, clock: MarketClock
) -> tuple[list[dict], dict]:
    unique, ambiguous = {}, set()
    counts = Counter(rows=0, missing_publication=0, duplicates=0, invalid=0, ambiguous_facts=0)
    for path in paths:
        if market == "CN":
            with path.open("rb") as stream:
                for raw in ijson.items(stream, "item", use_float=True):
                    counts["rows"] += 1
                    # Original tables lack a verified release date. Never use end_date.
                    if not raw.get("ann_date") and not raw.get("f_ann_date"):
                        counts["missing_publication"] += 1
                    else:
                        counts["unverified_publication_field"] += 1
            continue
        for (namespace, concept, unit), fact in _us_facts(path):
            counts["rows"] += 1
            filed = fact.get("filed")
            if not filed:
                counts["missing_publication"] += 1
                continue
            try:
                value = float(fact["val"])
                if not math.isfinite(value) or not fact.get("accn"):
                    raise ValueError("Nonfinite value or missing accession")
                end = datetime.fromisoformat(fact["end"]).date()
                if end > datetime.fromisoformat(filed).date():
                    raise ValueError("Period ends after filing date")
                available = clock.date_available(filed)
            except (ValueError, KeyError, TypeError):
                counts["invalid"] += 1
                continue
            key = (namespace, concept, unit, fact.get("start"), fact["end"], filed, fact["accn"])
            row = {
                "concept": f"{namespace}:{concept}:{unit}",
                "unit": unit,
                "period_start": fact.get("start"),
                "period_end": fact["end"],
                "filed": filed,
                "accession": fact["accn"],
                "value": value,
                "available_at": available,
                "source_file": path.name,
                "availability_rule": "inner_fact_filed_next_session_close",
            }
            if key in unique:
                if unique[key]["value"] != value:
                    ambiguous.add(key)
                else:
                    counts["duplicates"] += 1
            else:
                unique[key] = row
    counts["ambiguous_facts"] = len(ambiguous)
    rows = [row for key, row in unique.items() if key not in ambiguous]
    rows.sort(key=lambda r: (r["available_at"], r["period_end"], r["accession"], r["concept"]))
    counts["accepted"] = len(rows)
    return rows, dict(counts)


def snapshot(rows: list[dict], cutoff: datetime) -> dict[str, dict]:
    """Conservar el periodo más reciente conocido y su última revisión no ambigua."""
    cutoff = aware(cutoff)
    result = {}
    for row in rows:
        if row["available_at"] > cutoff:
            continue
        current = result.get(row["concept"])
        rank = (row["period_end"], row["available_at"])
        if current is None or rank > (current["period_end"], current["available_at"]):
            result[row["concept"]] = row
        elif rank == (current["period_end"], current["available_at"]):
            # Do not compare YTD with quarterly flows as if they had the same period.
            if row["period_start"] != current["period_start"]:
                starts = (row["period_start"] or "", current["period_start"] or "")
                if starts[0] > starts[1]:
                    result[row["concept"]] = row
            elif row["value"] != current["value"]:
                raise ValueError(f"Ambiguous contemporaneous fact: {row['concept']}")
    return result
