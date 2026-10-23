"""Trazas acotadas de vistas episódicas, sin duplicar valores fuente o etiquetas."""

import hashlib

import torch

from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.episodic_snapshot import _digest_id
from mars_titan.models.titans.regional_episode_readout import BridgedReadoutResult

KIND = "regional_episode_trace_v1"
_TENSORS = ("evidence_ids", "regional_weights", "regional_values", "read_values", "presence")


def enabled(reader):
    return reader is not None and reader.config.evidence_bridge is not None


def estimated_bytes(reader, flows, episodes, block_rows):
    r, h, k, steps = (
        len(reader.config.evidence_bridge.regions),
        reader.config.hidden_size,
        min(episodes, reader.config.neighbors),
        reader.config.refinements,
    )
    size = reader.step_logit.element_size()
    tensors = len(flows) * steps * (k * 8 + size * (r * k + r * h + h) + 1)
    blocks = (len(flows) + block_rows - 1) // block_rows
    metadata = blocks * (512 + steps * 5 * 224)
    return 65536 + tensors + metadata + sum(len(flow.encode()) + 512 for flow in flows)


def begin(reader, flows, snapshot, *, context_id, cutoff, block_rows):
    count = 0 if snapshot is None else snapshot.count
    if estimated_bytes(reader, flows, count, block_rows) > reader.config.max_working_bytes:
        raise ValueError("La traza regional del evento supera su presupuesto antes de emitir")
    edges = reader.evidence_bridge.edges.detach().cpu().to(reader.step_logit.dtype)
    graph = edges / edges.sum(-1, keepdim=True).clamp_min(1)
    return dict(
        schema_version=1,
        kind=KIND,
        configuration_sha256=hashlib.sha256(
            canonical(reader.config.identity()).encode()
        ).hexdigest(),
        context_id=context_id,
        cutoff=cutoff,
        snapshot_sha256=snapshot._digest if snapshot is not None else None,
        regions=reader.config.evidence_bridge.regions,
        hidden_size=reader.config.hidden_size,
        episodes=count,
        block_rows=block_rows,
        rows=len(flows),
        refinements=reader.config.refinements,
        bridge_weights=graph,
        blocks=[],
    )


def append(trace, flows, result):
    if type(result) is not BridgedReadoutResult or len(result.reads) != trace["refinements"]:
        raise ValueError("Falta el resultado regional tipado de cada refinamiento")
    reads = []
    for read in result.reads:
        b, h = read.values.shape
        r = len(trace["regions"])
        bridge = read.bridge
        if bridge is not None and bridge.snapshot_sha256 != trace["snapshot_sha256"]:
            raise ValueError("La traza mezcla snapshots de evidencia")
        raw = dict(
            evidence_ids=bridge.evidence_ids
            if bridge is not None
            else torch.empty((b, 0), dtype=torch.int64),
            regional_weights=bridge.evidence_weights
            if bridge is not None
            else read.values.new_empty((b, r, 0)),
            regional_values=bridge.region_values
            if bridge is not None
            else read.values.new_zeros((b, r, h)),
            read_values=read.values,
            presence=read.presence,
        )
        reads.append(
            {
                name: value.detach().to(device="cpu", copy=True).contiguous()
                for name, value in raw.items()
            }
        )
    trace["blocks"].append(dict(flow_ids=tuple(flows), reads=reads))


def _verify_header(trace, reader, *, flows, context_id, cutoff, refinements, block_rows):
    if refinements == 0:
        if trace is not None:
            raise ValueError("Un evento sin lectura no debe inventar una traza regional")
        return False
    fields = {
        "schema_version",
        "kind",
        "configuration_sha256",
        "context_id",
        "cutoff",
        "snapshot_sha256",
        "regions",
        "hidden_size",
        "episodes",
        "block_rows",
        "rows",
        "refinements",
        "bridge_weights",
        "blocks",
    }
    if (
        type(trace) is not dict
        or set(trace) != fields
        or trace["schema_version"] != 1
        or any(
            type(trace[name]) is not int
            for name in ("schema_version", "hidden_size", "rows", "refinements", "block_rows")
        )
        or trace["kind"] != KIND
        or trace["configuration_sha256"]
        != hashlib.sha256(canonical(reader.config.identity()).encode()).hexdigest()
        or trace["context_id"] != context_id
        or trace["cutoff"] != cutoff
        or type(trace["cutoff"]) is not int
        or type(trace["regions"]) not in (list, tuple)
        or tuple(trace["regions"]) != reader.config.evidence_bridge.regions
        or trace["hidden_size"] != reader.config.hidden_size
        or trace["rows"] != len(flows)
        or trace["block_rows"] != block_rows
        or trace["refinements"] != reader.config.refinements
        or refinements != reader.config.refinements
        or type(trace["episodes"]) is not int
        or not 0 <= trace["episodes"] <= 1024
    ):
        raise ValueError("La traza regional no corresponde al evento y lector")
    if reader.config.mode == "no_bank":
        if trace["snapshot_sha256"] is not None or trace["episodes"] != 0:
            raise ValueError("El control sin banco no debe atribuirse un snapshot")
    else:
        _digest_id(trace["snapshot_sha256"], "El snapshot de la traza regional")
    if (
        type(trace["blocks"]) is not list
        or not 1 <= block_rows <= reader.config.max_batch
        or len(trace["blocks"]) != (len(flows) + block_rows - 1) // block_rows
    ):
        raise ValueError("Los bloques de la traza regional no están acotados")
    dtype = reader.step_logit.dtype
    graph = reader.evidence_bridge.edges.detach().cpu().to(dtype)
    graph /= graph.sum(-1, keepdim=True).clamp_min(1)
    if (
        not isinstance(trace["bridge_weights"], torch.Tensor)
        or trace["bridge_weights"].device.type != "cpu"
        or trace["bridge_weights"].layout != torch.strided
        or trace["bridge_weights"].dtype != dtype
        or not torch.equal(trace["bridge_weights"], graph)
    ):
        raise ValueError("La traza regional cambió los puentes")
    return True


def _verify_read(read, rows, r, h, expected_count, dtype, active):
    if type(read) is not dict or set(read) != set(_TENSORS):
        raise ValueError("La lectura regional no conserva sus campos")
    ids = read["evidence_ids"]
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != rows:
        raise ValueError("Los IDs regionales no corresponden al bloque")
    k = ids.shape[1]
    if k != expected_count:
        raise ValueError("La traza regional mezcla tamaños del snapshot")
    shapes = (
        (rows, k),
        (rows, r, k),
        (rows, r, h),
        (rows, h),
        (rows, 1),
    )
    for name, shape, required_dtype in zip(
        _TENSORS, shapes, (torch.int64, dtype, dtype, dtype, torch.bool), strict=True
    ):
        value = read[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.layout != torch.strided
            or value.shape != shape
            or value.dtype != required_dtype
            or value.requires_grad
            or value.grad_fn is not None
            or not torch.isfinite(value).all()
        ):
            raise ValueError("La traza regional contiene un tensor incompatible")
    if k:
        ordered = ids.sort(-1).values
        weights = read["regional_weights"]
        if (
            (ids <= 0).any()
            or (ordered[:, 1:] == ordered[:, :-1]).any()
            or not torch.equal(read["presence"], torch.full((rows, 1), any(active)))
            or (weights < 0).any()
            or not torch.allclose(
                weights.sum(-1),
                torch.tensor(active, dtype=dtype).expand(rows, -1),
                atol=1e-6,
                rtol=1e-6,
            )
        ):
            raise ValueError("La traza no conserva IDs únicos y pesos regionales normalizados")
        inactive = ~torch.tensor(active)
        if (
            weights[:, inactive].count_nonzero()
            or read["regional_values"][:, inactive].count_nonzero()
            or (not any(active) and read["read_values"].count_nonzero())
        ):
            raise ValueError("Una región retirada debe conservar valores cero")
    elif (
        read["presence"].any()
        or read["regional_values"].count_nonzero()
        or read["read_values"].count_nonzero()
    ):
        raise ValueError("Una lectura ausente debe conservar valores cero")
    return ids


def verify(trace, reader, *, flows, context_id, cutoff, refinements, block_rows):
    if not _verify_header(
        trace,
        reader,
        flows=flows,
        context_id=context_id,
        cutoff=cutoff,
        refinements=refinements,
        block_rows=block_rows,
    ):
        return
    r, h, dtype = len(trace["regions"]), trace["hidden_size"], reader.step_logit.dtype
    actual = []
    for position, block in enumerate(trace["blocks"]):
        if (
            type(block) is not dict
            or set(block) != {"flow_ids", "reads"}
            or type(block["flow_ids"]) not in (list, tuple)
            or len(block["flow_ids"]) != min(block_rows, len(flows) - position * block_rows)
            or type(block["reads"]) is not list
            or len(block["reads"]) != refinements
        ):
            raise ValueError("El bloque no conserva sus flujos y lecturas regionales")
        actual.extend(block["flow_ids"])
        first_ids = None
        for read in block["reads"]:
            ids = _verify_read(
                read,
                len(block["flow_ids"]),
                r,
                h,
                min(trace["episodes"], reader.config.neighbors),
                dtype,
                reader.config.evidence_bridge.active_views(),
            )
            if first_ids is None:
                first_ids = ids
            elif reader.config.episode_selection == "first_read" and not torch.equal(
                ids, first_ids
            ):
                raise ValueError("La selección regional fija cambió de episodios")
    if (
        actual != list(flows)
        or estimated_bytes(reader, flows, trace["episodes"], block_rows)
        > reader.config.max_working_bytes
    ):
        raise ValueError("Los flujos o bytes de la traza regional no concilian")
