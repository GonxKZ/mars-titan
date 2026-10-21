"""Recibos regionales por fila para el archivo JSON del controlador."""

import numpy as np

from .episodic_snapshot import _digest_id
from .regional_routes import REFRESH_AT, RegionalRouteResult

KIND = "regional_mechanism_trace_v1"


def estimated_bytes(reader, rows, episodes, steps):
    r, h = len(reader.config.evidence_bridge.regions), reader.config.hidden_size
    k = min(episodes, reader.config.neighbors)
    # Estimación conservadora de números y metadatos JSON, no del asignador de Python.
    return 32768 + rows * (1024 + steps * (2048 + 40 * (k * (r + 1) + r * h + h)))


def export(result, snapshot):
    if type(result) is not RegionalRouteResult:
        raise ValueError("El controlador necesita el resultado de una ruta regional")
    rows = [
        dict(
            kind=KIND,
            snapshot_sha256=snapshot._digest if snapshot is not None else None,
            episodes=snapshot.count if snapshot is not None else 0,
            value_projections=result.value_projections,
            reads=[],
        )
        for _ in range(len(result.state))
    ]
    for read in result.reads:
        bridge = read.bridge
        if (bridge is None) != (snapshot is None) or (
            bridge is not None and bridge.snapshot_sha256 != snapshot._digest
        ):
            raise ValueError("La exportación regional mezcla snapshots")
        if bridge is None:
            parts = [
                dict(evidence_ids=[], regional_weights=[], regional_values=[], read_values=[])
                for _ in rows
            ]
        else:
            arrays = {
                name: value.detach().cpu().tolist()
                for name, value in dict(
                    evidence_ids=bridge.evidence_ids,
                    regional_weights=bridge.evidence_weights,
                    regional_values=bridge.region_values,
                    read_values=read.values,
                ).items()
            }
            parts = [{name: value[i] for name, value in arrays.items()} for i in range(len(rows))]
        for row, part in zip(rows, parts, strict=True):
            row["reads"].append(part)
    return rows


def verify(trace, reader, *, snapshot_sha256, steps, searches, route):
    if (
        type(route) is not str
        or route not in REFRESH_AT
        or type(steps) is not int
        or steps != int(route[1])
        or type(searches) is not int
    ):
        raise ValueError("La traza regional necesita una ruta y trabajo válidos")
    if (
        type(trace) is not dict
        or set(trace) != {"kind", "snapshot_sha256", "episodes", "value_projections", "reads"}
        or trace["kind"] != KIND
        or trace["snapshot_sha256"] != snapshot_sha256
        or type(trace["episodes"]) is not int
        or not 0 <= trace["episodes"] <= 1024
        or type(trace["value_projections"]) is not int
        or trace["value_projections"] != (steps if trace["episodes"] else 0)
        or type(trace["reads"]) is not list
        or len(trace["reads"]) != steps
    ):
        raise ValueError("El recibo regional no corresponde al trabajo declarado")
    if reader.config.mode == "no_bank":
        if snapshot_sha256 is not None or trace["episodes"] != 0:
            raise ValueError("El control regional sin banco no puede atribuirse episodios")
    else:
        _digest_id(snapshot_sha256, "El snapshot regional del controlador")
    r, h = len(reader.config.evidence_bridge.regions), reader.config.hidden_size
    k = min(trace["episodes"], reader.config.neighbors)
    expected_searches = (1 + int("r" in route)) if trace["episodes"] else 0
    if searches != expected_searches:
        raise ValueError("El recibo regional no conserva las búsquedas de su ruta")
    previous_ids = None
    for step, read in enumerate(trace["reads"]):
        if type(read) is not dict or set(read) != {
            "evidence_ids",
            "regional_weights",
            "regional_values",
            "read_values",
        }:
            raise ValueError("La lectura regional no conserva su esquema")
        ids = read["evidence_ids"]
        if (
            type(ids) is not list
            or len(ids) != k
            or any(type(value) is not int or value <= 0 for value in ids)
            or len(set(ids)) != k
        ):
            raise ValueError("Los IDs comunes del recibo regional no son válidos")
        if previous_ids is not None and step not in REFRESH_AT[route] and ids != previous_ids:
            raise ValueError("El bloque regional fijo cambió de episodios")
        previous_ids = ids
        for name, shape in (
            ("regional_weights", (r, k)),
            ("regional_values", (r, h)),
            ("read_values", (h,)),
        ):
            if snapshot_sha256 is None:
                if read[name] != []:
                    raise ValueError("El recibo sin banco no debe inventar vistas")
                continue
            values = np.asarray(read[name])
            if values.shape != shape or values.dtype.kind != "f" or not np.isfinite(values).all():
                raise ValueError("Las vistas regionales tienen forma o valores incompatibles")
        if snapshot_sha256 is not None:
            weights = np.asarray(read["regional_weights"])
            if k and (
                (weights < 0).any() or not np.allclose(weights.sum(-1), 1, atol=1e-6, rtol=1e-6)
            ):
                raise ValueError("El recibo regional no conserva pesos normalizados")
            if not k and any(
                np.count_nonzero(read[name]) for name in ("regional_values", "read_values")
            ):
                raise ValueError("Una lectura regional vacía debe conservar ceros")
