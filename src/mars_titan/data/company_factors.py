"""Ratios de balance por presentación, sin mezclar periodos, unidades o revisiones."""

import math
from collections import Counter, defaultdict
from itertools import groupby

from .temporal import aware

FACTOR_DEFINITIONS = (
    ("current_ratio", (("AssetsCurrent", 1),), "LiabilitiesCurrent"),
    ("working_capital_to_assets", (("AssetsCurrent", 1), ("LiabilitiesCurrent", -1)), "Assets"),
    ("liabilities_to_assets", (("Liabilities", 1),), "Assets"),
    ("equity_to_assets", (("StockholdersEquity", 1),), "Assets"),
    ("receivables_to_assets", (("AccountsReceivableNetCurrent", 1),), "Assets"),
    ("payables_to_assets", (("AccountsPayableCurrent", 1),), "Assets"),
    ("liabilities_to_positive_equity", (("Liabilities", 1),), "StockholdersEquity"),
)
FACTOR_CONCEPTS = tuple(f"company:{name}:ratio" for name, _, _ in FACTOR_DEFINITIONS)
_TAGS = {tag for _, terms, denominator in FACTOR_DEFINITIONS for tag, _ in terms} | {
    denominator for _, _, denominator in FACTOR_DEFINITIONS
}


def _component(rows):
    if not rows:
        return None, "missing_component"
    compatible = [r for r in rows if r["unit"] == "USD" and r["period_start"] is None]
    if not compatible:
        return None, "incompatible_component"
    if len({r["value"] for r in compatible}) != 1:
        return None, "ambiguous_component"
    return min(compatible, key=lambda r: (r["available_at"], r.get("source_file", ""))), None


def _factor(group, definition, identity, observed_at):
    name, terms, denominator_tag = definition
    tags = sorted({tag for tag, _ in terms} | {denominator_tag})
    selected, problems = {}, []
    for tag in tags:
        row, problem = _component(group.get(tag, []))
        if problem:
            problems.append((tag, problem))
        else:
            selected[tag] = row
    numerator = denominator = value = None
    status = problems[0][1] if problems else "accepted"
    if denominator_tag in selected:
        denominator = float(selected[denominator_tag]["value"])
    if all(tag in selected for tag, _ in terms):
        try:
            numerator = math.fsum(selected[tag]["value"] * sign for tag, sign in terms)
        except OverflowError:
            status = "nonfinite_result"
    if not problems and status == "accepted":
        if denominator <= 0:
            status = "nonpositive_denominator"
        elif numerator is not None:
            candidate = numerator / denominator
            if math.isfinite(candidate):
                value = candidate
            else:
                status = "nonfinite_result"
    period_end, filed, accession = identity
    return {
        "concept": f"company:{name}:ratio",
        "unit": "ratio",
        "period_start": None,
        "period_end": period_end,
        "filed": filed,
        "accession": accession,
        "value": value,
        "available_at": max(r["available_at"] for r in selected.values())
        if status == "accepted"
        else observed_at,
        "availability_rule": "same_filing_stock_usd_latest_component",
        "status": status,
        "numerator": numerator,
        "denominator": denominator,
        "unavailable_components": [f"us-gaap:{tag}:USD" for tag, _ in problems],
        "component_problems": [
            {"concept": f"us-gaap:{tag}:USD", "reason": problem} for tag, problem in problems
        ],
        "components": [
            {field: row.get(field) for field in ("concept", "value", "available_at", "source_file")}
            for row in selected.values()
        ],
    }


def derive_company_factors(rows, *, max_facts: int = 100_000):
    """Emite siete resultados por grupo de una empresa, incluidas ausencias explícitas.

    La entrada contiene hechos normalizados de un único activo. Se acota antes
    de agrupar y no se acumula la salida. Los cocientes se calculan en float64
    antes de aplicar la transformación fija de los vectores de entrenamiento.
    """
    if type(max_facts) is not int or max_facts < 1:
        raise ValueError("El presupuesto de hechos debe ser positivo")
    groups = defaultdict(list)
    for count, row in enumerate(rows, 1):
        if count > max_facts:
            raise ValueError("Los hechos contables superan el presupuesto")
        namespace, tag, unit = row["concept"].split(":", 2)
        if namespace != "us-gaap" or (
            tag not in _TAGS and (row["period_start"] is not None or unit != "USD")
        ):
            continue
        if not math.isfinite(row["value"]):
            raise ValueError("El componente contable debe ser finito")
        if unit != row["unit"] or not row["accession"] or not row["filed"]:
            raise ValueError("El componente contable no tiene una identidad coherente")
        aware(row["available_at"])
        key = (row["period_end"], row["filed"], row["accession"])
        groups[key].append(row)
    for identity, facts in sorted(groups.items()):
        known, previous = defaultdict(list), {}
        facts.sort(key=lambda r: r["available_at"])
        for observed_at, arrivals in groupby(facts, key=lambda r: r["available_at"]):
            for row in arrivals:
                known[row["concept"].split(":", 2)[1]].append(row)
            for definition in FACTOR_DEFINITIONS:
                factor = _factor(known, definition, identity, observed_at)
                if previous.get(factor["concept"]) != factor:
                    previous[factor["concept"]] = factor
                    yield factor


def write_company_factors(path, rows, *, batch_rows: int = 256, max_facts: int = 100_000):
    """Escribe la trazabilidad por bloques y conserva solo el estado de la instantánea."""
    import pyarrow as pa

    from .batches import atomic_parquet_batches

    if type(batch_rows) is not int or not 1 <= batch_rows <= 1024:
        raise ValueError("El bloque de factores debe contener entre 1 y 1024 filas")
    timestamp = pa.timestamp("us", tz="UTC")
    schema = pa.schema(
        [
            (name, pa.string())
            for name in (
                "concept",
                "unit",
                "period_start",
                "period_end",
                "filed",
                "accession",
                "availability_rule",
                "status",
            )
        ]
        + [("available_at", timestamp)]
        + [(name, pa.float64()) for name in ("value", "numerator", "denominator")]
        + [
            ("unavailable_components", pa.list_(pa.string())),
            (
                "component_problems",
                pa.list_(
                    pa.struct(
                        [
                            ("concept", pa.string()),
                            ("reason", pa.string()),
                        ]
                    )
                ),
            ),
            (
                "components",
                pa.list_(
                    pa.struct(
                        [
                            ("concept", pa.string()),
                            ("value", pa.float64()),
                            ("available_at", timestamp),
                            ("source_file", pa.string()),
                        ]
                    )
                ),
            ),
        ]
    )
    compact, counts = [], Counter()

    def batches():
        pending = []
        for row in derive_company_factors(rows, max_facts=max_facts):
            if len(compact) >= max_facts:
                raise ValueError("Los factores derivados superan el presupuesto")
            compact.append(
                {
                    name: row[name]
                    for name in (
                        "concept",
                        "value",
                        "period_start",
                        "period_end",
                        "filed",
                        "accession",
                        "available_at",
                    )
                }
            )
            counts[row["status"]] += 1
            pending.append(row)
            if len(pending) == batch_rows:
                yield pa.Table.from_pylist(pending, schema=schema)
                pending = []
        if pending or not compact:
            yield pa.Table.from_pylist(pending, schema=schema)

    atomic_parquet_batches(path, batches())
    return compact, dict(counts)
