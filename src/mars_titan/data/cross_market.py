"""Contexto de la otra bolsa con disponibilidad UTC y antigüedad explícitas."""

import math
from bisect import bisect_right

import pyarrow as pa

from .temporal import aware


def asof_cross_market_context(observations: pa.Table, decisions: pa.Table) -> pa.Table:
    """Cruzar un bloque de decisiones sin adelantar observaciones ni rellenar ausencias."""
    if observations.num_rows > 200_000 or decisions.num_rows > 65_536:
        raise ValueError("El bloque de contexto cruzado supera su presupuesto")
    series = {market: [] for market in ("US", "CN")}
    units = {}
    for row in observations.to_pylist():
        market, value, unit = row["market"], row["value"], row["unit"]
        if market not in series or not isinstance(unit, str) or not 1 <= len(unit) <= 64:
            raise ValueError("La observación necesita mercado y unidad válidos")
        if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
            raise ValueError("El contexto contiene un valor no finito")
        if market in units and units[market] != unit:
            raise ValueError("La unidad cambia dentro de una serie de mercado")
        units[market] = unit
        series[market].append((aware(row["available_at"]), value))
    for rows in series.values():
        rows.sort(key=lambda r: r[0])
        if len({r[0] for r in rows}) != len(rows):
            raise ValueError("Hay observaciones ambiguas en el mismo instante y mercado")
    times = {market: [r[0] for r in rows] for market, rows in series.items()}
    result = []
    for row in decisions.to_pylist():
        if row["market"] not in series:
            raise ValueError("El mercado de decisión no está admitido")
        decision = aware(row["prediction_at"])
        other = "US" if row["market"] == "CN" else "CN"
        position = bisect_right(times[other], decision) - 1
        available, value = series[other][position] if position >= 0 else (None, None)
        result.append(
            dict(
                asset_id=row["asset_id"],
                prediction_at=decision,
                source_market=other,
                value=value,
                unit=units.get(other),
                available_at=available,
                age_seconds=(decision - available).total_seconds() if available else None,
                observed=value is not None,
            )
        )
    schema = pa.schema(
        [
            ("asset_id", pa.string()),
            ("prediction_at", pa.timestamp("us", tz="UTC")),
            ("source_market", pa.string()),
            ("value", pa.float64()),
            ("unit", pa.string()),
            ("available_at", pa.timestamp("us", tz="UTC")),
            ("age_seconds", pa.float64()),
            ("observed", pa.bool_()),
        ]
    )
    return pa.Table.from_pylist(result, schema=schema)
