"""Hechos contables por publicación. El contenedor no fecha las cifras internas."""

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import ijson
from ijson.common import ObjectBuilder

from .input_policy import STRICT_INPUTS, masked_inputs
from .storage import sha256
from .temporal import MarketClock, aware

_MAX_EXCLUSION_BYTES = 16 * 1024**2


def _record_exclusion(exclusions, key, entry, total_bytes):
    """Agrupar registros excluidos sin perder su fuente ni crecer sin límite."""
    previous = exclusions.get(key)
    old_bytes = len(json.dumps(previous).encode()) if previous else 0
    if previous:
        entry["first_ordinal"] = min(previous["first_ordinal"], entry["first_ordinal"])
        entry["last_ordinal"] = max(previous["last_ordinal"], entry["last_ordinal"])
        entry["source_records"] += previous["source_records"]
    total_bytes += len(json.dumps(entry).encode()) + (2 if previous is None else 0) - old_bytes
    if total_bytes > _MAX_EXCLUSION_BYTES:
        raise ValueError("El diagnóstico de exclusiones supera el presupuesto")
    exclusions[key] = entry
    return total_bytes


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
    paths: list[Path],
    market: str,
    clock: MarketClock,
    *,
    max_unique_facts: int = 100_000,
    input_policy: str = STRICT_INPUTS,
) -> tuple[list[dict], dict]:
    masked = masked_inputs(input_policy)
    if type(max_unique_facts) is not int or max_unique_facts < 1:
        raise ValueError("El presupuesto de hechos contables debe ser positivo")
    unique, ambiguous = {}, set()
    exclusions, source_hashes, exclusion_bytes = {}, {}, 2
    counts = Counter(rows=0, missing_publication=0, duplicates=0, invalid=0, ambiguous_facts=0)
    for path in paths:
        if market == "CN":
            with path.open("rb") as stream:
                for raw in ijson.items(stream, "item", use_float=True):
                    counts["rows"] += 1
                    # Sin publicación verificada, end_date no acredita disponibilidad.
                    if not raw.get("ann_date") and not raw.get("f_ann_date"):
                        counts["missing_publication"] += 1
                    else:
                        counts["unverified_publication_field"] += 1
            continue
        for ordinal, ((namespace, concept, unit), fact) in enumerate(_us_facts(path), 1):
            counts["rows"] += 1
            filed = fact.get("filed")
            if not filed:
                counts["missing_publication"] += 1
                if not masked:
                    continue
            try:
                if not isinstance(fact.get("accn"), str) or (
                    fact.get("start") is not None and not isinstance(fact["start"], str)
                ):
                    raise ValueError(
                        "El identificador o el inicio del periodo no son escalares válidos"
                    )
                if masked and (
                    isinstance(fact.get("val"), bool)
                    or (filed is not None and not isinstance(filed, str))
                ):
                    raise ValueError("El valor o la publicación tienen un tipo no válido")
                value = float(fact["val"])
                if not math.isfinite(value) or not fact.get("accn"):
                    raise ValueError(
                        "El valor no es finito o falta el identificador de presentación"
                    )
                if masked and any(
                    not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day)
                    for day in (fact["end"], fact.get("start"), filed)
                    if day is not None and day != ""
                ):
                    raise ValueError("Las fechas contables deben ser días ISO completos")
                end = datetime.fromisoformat(fact["end"]).date()
                if fact.get("start") and datetime.fromisoformat(fact["start"]).date() > end:
                    raise ValueError("El inicio del periodo es posterior a su cierre")
                filed_day = datetime.fromisoformat(filed).date() if filed else None
                if filed_day and end > filed_day and not masked:
                    raise ValueError("El periodo termina después de la fecha de presentación")
                available = clock.date_available(filed) if filed else None
            except (ValueError, KeyError, TypeError) as error:
                if masked:
                    raise ValueError(
                        "El hecho contable contiene un valor o una fecha no válidos"
                    ) from error
                counts["invalid"] += 1
                continue
            key = (namespace, concept, unit, fact.get("start"), fact["end"], filed, fact["accn"])
            if filed_day and end > filed_day:
                if path not in source_hashes:
                    source_hashes[path] = sha256(path)
                excluded_key = (path.name, source_hashes[path], *key)
                if (
                    excluded_key not in exclusions
                    and len(unique) + len(exclusions) >= max_unique_facts
                ):
                    raise ValueError("Los hechos y exclusiones únicos superan el presupuesto")
                exclusion_bytes = _record_exclusion(
                    exclusions,
                    excluded_key,
                    dict(
                        reason="period_after_filing",
                        concept=f"{namespace}:{concept}:{unit}",
                        unit=unit,
                        period_start=fact.get("start"),
                        period_end=fact["end"],
                        filed=filed,
                        accession=fact["accn"],
                        source_file=path.name,
                        source_sha256=source_hashes[path],
                        diagnostic_available_at=available.isoformat(),
                        first_ordinal=ordinal,
                        last_ordinal=ordinal,
                        source_records=1,
                    ),
                    exclusion_bytes,
                )
                counts["temporal_excluded"] += 1
                continue
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
                if len(unique) + len(exclusions) >= max_unique_facts:
                    raise ValueError("Los hechos contables únicos superan el presupuesto")
                unique[key] = row
    counts["ambiguous_facts"] = len(ambiguous)
    rows = [row for key, row in unique.items() if key not in ambiguous]
    rows.sort(
        key=lambda r: (
            r["available_at"] is None,
            r["available_at"],
            r["period_end"],
            r["accession"],
            r["concept"],
        )
    )
    counts["accepted"] = len(rows)
    audit = dict(counts)
    if masked:
        audit.update(
            temporal_excluded=counts["temporal_excluded"],
            temporal_exclusions=list(exclusions.values()),
        )
    return rows, audit


def snapshot(rows: list[dict], cutoff: datetime) -> dict[str, dict]:
    """Conservar el periodo más reciente conocido y su última revisión no ambigua."""
    cutoff = aware(cutoff)
    result, ranks, ambiguous = {}, {}, set()
    for row in rows:
        if row["available_at"] > cutoff:
            continue
        concept = row["concept"]
        # A igual cierre y publicación, se prefiere el flujo del periodo más corto.
        # El vector preparado usa saldos de balance, no flujos de distinta duración.
        rank = (row["period_end"], row["available_at"], row["period_start"] or "")
        if concept not in ranks or rank > ranks[concept]:
            result[concept], ranks[concept] = row, rank
            ambiguous.discard(concept)
        elif rank == ranks[concept] and row["value"] != result[concept]["value"]:
            ambiguous.add(concept)
    return {concept: row for concept, row in result.items() if concept not in ambiguous}
