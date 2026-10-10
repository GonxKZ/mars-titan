"""Utilidades comunes de las pruebas de paridad de trazas de cada familia (#448).

Las pruebas de cada familia viven junto a sus fixtures y comparan un recorrido sin trazas
con otro con trazas en cada paso. Estas funciones leen las trazas con la validación del
contrato y exigen que no haya filas repetidas.
"""

import math

from mars_titan.learning_traces.recorder import read_traces

# Cada paso registrado y un presupuesto holgado para las fixtures.
EVERY_STEP = dict(every=1, max_bytes=16 * 1024**2)


def trace_rows(folder):
    """Manifiesto y filas de unas trazas, sin claves repetidas."""
    manifest, table = read_traces(folder)
    rows = table.to_pylist()
    keys = [(r["step"], r["phase"], r["metric"], r["group"], r["stat"]) for r in rows]
    assert len(keys) == len(set(keys)), "Hay eventos de traza duplicados"
    assert rows, "Las trazas están vacías"
    return manifest, rows


def same_rows(left, right):
    """Las mismas filas con los mismos bits, con NaN igual a NaN."""

    def key(row):
        return (row["step"], row["phase"], row["metric"], row["group"], row["stat"])

    left, right = sorted(left, key=key), sorted(right, key=key)
    if [key(r) for r in left] != [key(r) for r in right]:
        return False
    return all(
        a["value"] == b["value"] or (math.isnan(a["value"]) and math.isnan(b["value"]))
        for a, b in zip(left, right, strict=True)
    )


def metrics(rows, phase=None):
    return {row["metric"] for row in rows if phase is None or row["phase"] == phase}
