"""Recibos regionales por fila para el archivo JSON del controlador."""

import numpy as np

from .episodic_snapshot import _digest_id
from .evidence_bridge import HierarchyBridgeRead
from .regional_episode_readout import hierarchy_weights
from .regional_routes import REFRESH_AT, RegionalRouteResult

KIND = "regional_mechanism_trace_v1"
HIERARCHY_KIND = "hierarchical_mechanism_trace_v1"


def verify_work(work, reader, *, value_projections):
    """Conciliar lecturas, vistas R0 y pasos J con el lector sellado de la sesión."""
    expected = dict(bridge_value_projections=value_projections)
    expected.update(reader.recurrence_work(value_projections))
    if any(
        type(work.get(name)) is not int or work[name] != value for name, value in expected.items()
    ):
        raise ValueError(
            "Las proyecciones y actualizaciones regionales no concilian con su trabajo"
        )


def estimated_bytes(reader, rows, episodes, steps):
    r, h = len(reader.config.evidence_bridge.regions), reader.config.hidden_size
    k = min(episodes, reader.config.neighbors)
    # Estimación conservadora de números y metadatos JSON, no del asignador de Python.
    total = 32768 + rows * (1024 + steps * (2048 + 40 * (k * (r + 1) + r * h + h)))
    if reader.config.evidence_bridge.hierarchy is not None:
        total += rows * (2048 + 40 * 65 + steps * (512 + 40 * 3 * h))
    return total


def export(result, snapshot, *, reader=None):
    if type(result) is not RegionalRouteResult:
        raise ValueError("El controlador necesita el resultado de una ruta regional")
    full = hierarchy_weights(reader) if reader is not None else None
    hierarchical = full is not None
    rows = [
        dict(
            kind=HIERARCHY_KIND if hierarchical else KIND,
            snapshot_sha256=snapshot._digest if snapshot is not None else None,
            episodes=snapshot.count if snapshot is not None else 0,
            value_projections=result.value_projections,
            reads=[],
        )
        for _ in range(len(result.state))
    ]
    if hierarchical:
        for row in rows:
            row.update(hierarchy_weights=full.tolist(), bridge_weights=full[:4, :4].tolist())
    for read in result.reads:
        bridge = read.bridge
        if (bridge is None) != (snapshot is None) or (
            bridge is not None and bridge.snapshot_sha256 != snapshot._digest
        ):
            raise ValueError("La exportación regional mezcla snapshots")
        if bridge is not None and (type(bridge) is HierarchyBridgeRead) != hierarchical:
            raise ValueError("La exportación jerárquica necesita su lector identificado")
        if bridge is None:
            parts = [
                dict(evidence_ids=[], regional_weights=[], regional_values=[], read_values=[])
                for _ in rows
            ]
            if hierarchical:
                for part in parts:
                    part.update(group_values=[], root_values=[])
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
            if hierarchical:
                if not np.array_equal(
                    bridge.hierarchy_weights.detach().cpu().numpy(),
                    full.expand(len(rows), -1, -1).numpy(),
                ) or not np.array_equal(
                    bridge.bridge_weights.detach().cpu().numpy(),
                    full[:4, :4].expand(len(rows), -1, -1).numpy(),
                ):
                    raise ValueError(
                        "La lectura no conserva el grafo jerárquico efectivo de la sesión"
                    )
                arrays.update(
                    group_values=bridge.group_values.detach().cpu().tolist(),
                    root_values=bridge.root_values.detach().cpu().tolist(),
                )
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
    full = hierarchy_weights(reader)
    fields = {"kind", "snapshot_sha256", "episodes", "value_projections", "reads"}
    if full is not None:
        fields.update(("hierarchy_weights", "bridge_weights"))
    if (
        type(trace) is not dict
        or set(trace) != fields
        or trace["kind"] != (HIERARCHY_KIND if full is not None else KIND)
        or trace["snapshot_sha256"] != snapshot_sha256
        or type(trace["episodes"]) is not int
        or not 0 <= trace["episodes"] <= 1024
        or type(trace["value_projections"]) is not int
        or trace["value_projections"] != (steps if trace["episodes"] else 0)
        or type(trace["reads"]) is not list
        or len(trace["reads"]) != steps
    ):
        raise ValueError("El recibo regional no corresponde al trabajo declarado")
    if full is not None:
        for name, expected in (("hierarchy_weights", full), ("bridge_weights", full[:4, :4])):
            value = np.asarray(trace[name])
            if value.dtype.kind != "f" or not np.array_equal(value, expected.numpy()):
                raise ValueError("El recibo cambió el grafo jerárquico efectivo")
    if reader.config.mode == "no_bank":
        if snapshot_sha256 is not None or trace["episodes"] != 0:
            raise ValueError("El control regional sin banco no puede atribuirse episodios")
    else:
        _digest_id(snapshot_sha256, "El snapshot regional del controlador")
    r, h = len(reader.config.evidence_bridge.regions), reader.config.hidden_size
    k = min(trace["episodes"], reader.config.neighbors)
    active = np.asarray(reader.config.evidence_bridge.active_views())
    expected_searches = (1 + int("r" in route)) if trace["episodes"] else 0
    if searches != expected_searches:
        raise ValueError("El recibo regional no conserva las búsquedas de su ruta")
    previous_ids = None
    for step, read in enumerate(trace["reads"]):
        read_fields = {
            "evidence_ids",
            "regional_weights",
            "regional_values",
            "read_values",
        }
        if full is not None:
            read_fields.update(("group_values", "root_values"))
        if type(read) is not dict or set(read) != read_fields:
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
        shapes = (
            ("regional_weights", (r, k)),
            ("regional_values", (r, h)),
            ("read_values", (h,)),
        )
        if full is not None:
            shapes += (("group_values", (2, h)), ("root_values", (1, h)))
        for name, shape in shapes:
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
                (weights < 0).any()
                or not np.allclose(weights.sum(-1), active, atol=1e-6, rtol=1e-6)
            ):
                raise ValueError("El recibo regional no conserva pesos normalizados")
            if (
                np.count_nonzero(weights[~active])
                or np.count_nonzero(np.asarray(read["regional_values"])[~active])
                or (not active.any() and np.count_nonzero(read["read_values"]))
            ):
                raise ValueError("Una región retirada debe conservar valores cero")
            if not k and any(
                np.count_nonzero(read[name])
                for name in ("regional_values", "read_values")
                + (("group_values", "root_values") if full is not None else ())
            ):
                raise ValueError("Una lectura regional vacía debe conservar ceros")
