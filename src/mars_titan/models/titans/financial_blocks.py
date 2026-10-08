"""Transporte de estados por bloques y reemplazo de referencias, sin publicar artefactos."""

import re
import sys
from dataclasses import replace

import torch

from .config import bounded_integer, canonical, require_identity
from .financial import FinancialState
from .state import MACState, NeuralMemoryState, require_payload

MAX_REFERENCE_FLOWS = 8192
MAX_REFERENCE_BYTES = 16 * 1024**2
REFERENCE_FIELDS = {
    "block_id",
    "row",
    "config_id",
    "parameter_id",
    "observed_steps",
    "last_prediction_at",
    "last_sample_id",
}


def _tensors(state):
    return (state.observed_steps,) + (
        (*state.mac.memory.weights, *state.mac.memory.momentum, state.mac.memory.steps)
        if state.mac
        else ()
    )


def _decode(predictor, payload, *, device):
    """Validar las fuentes sin copiar sus tensores ni transferirlos de dispositivo."""
    value = require_payload(
        payload,
        {
            "schema_version",
            "configuration",
            "parameter_id",
            "flow_ids",
            "last_sample_ids",
            "last_prediction_at",
            "observed_steps",
            "mac",
        },
    )
    require_identity(value["configuration"], predictor.get_extra_state())
    state = FinancialState(
        predictor._config_id(),
        value["parameter_id"],
        value["flow_ids"],
        value["last_sample_ids"],
        value["last_prediction_at"],
        value["observed_steps"],
        None,
    )
    predictor._validate_cursor(state)
    if predictor.mac:
        mac = require_payload(value["mac"], {"schema_version", "configuration", "memory"})
        require_identity(mac["configuration"], predictor.mac.config.identity())
        memory = require_payload(
            mac["memory"],
            {
                "schema_version",
                "configuration",
                "weights",
                "momentum",
                "steps",
            },
        )
        require_identity(memory["configuration"], predictor.mac.memory.config.identity())
        for name in ("weights", "momentum"):
            if type(memory[name]) is not tuple or len(memory[name]) != 2:
                raise ValueError("El estado no conserva las dos capas de memoria")
        state = replace(
            state,
            mac=MACState(
                NeuralMemoryState(
                    memory["weights"],
                    memory["momentum"],
                    memory["steps"],
                    predictor.mac.memory.config.fingerprint(),
                ),
                predictor.mac.config.fingerprint(),
            ),
        )
    elif value["mac"] is not None:
        raise ValueError("El control directo no admite estado MAC")
    predictor._check_bytes(predictor._usage(state)["total_bytes"])
    predictor._validate_state(state, device=device)
    return state


def _copy_to(state, device):
    def copy(value, target):
        return value.detach().to(device=target, copy=True).contiguous()

    mac = state.mac
    if mac:
        memory = mac.memory
        mac = replace(
            mac,
            memory=replace(
                memory,
                weights=tuple(copy(w, device) for w in memory.weights),
                momentum=tuple(copy(m, device) for m in memory.momentum),
                steps=copy(memory.steps, device),
            ),
        )
    return replace(
        state,
        flow_ids=tuple(state.flow_ids),
        last_sample_ids=tuple(state.last_sample_ids),
        last_prediction_at=tuple(state.last_prediction_at),
        observed_steps=copy(state.observed_steps, "cpu"),
        mac=mac,
    )


def export_state_cpu(predictor, state):
    predictor.verify_parameter_identity()
    predictor._validate_state(state)
    copied = _copy_to(state, torch.device("cpu"))
    payload = predictor._metadata(
        copied.flow_ids, copied.last_sample_ids, copied.last_prediction_at
    )
    payload["observed_steps"] = copied.observed_steps
    if copied.mac:
        payload["mac"]["memory"].update(
            weights=copied.mac.memory.weights,
            momentum=copied.mac.memory.momentum,
            steps=copied.mac.memory.steps,
        )
    return payload


def restore_state(predictor, payload, *, device=None):
    predictor.verify_parameter_identity()
    target = predictor.head.weight.device
    if device is not None and torch.device(device) != target:
        raise ValueError("El dispositivo solicitado no coincide con el predictor")
    # Sin device se conserva la exigencia anterior de dispositivo del payload.
    source = target if device is None else torch.device("cpu")
    state = _decode(predictor, payload, device=source)
    return _copy_to(state, target)


def gather_state(predictor, blocks, flow_ids, *, max_source_bytes=512 * 1024**2):
    """blocks asigna cada ID pedido a (payload CPU, fila), sin inicialización implícita."""
    predictor.verify_parameter_identity()
    if not isinstance(flow_ids, (tuple, list)):
        raise ValueError("Los flujos deben ser una lista o tupla acotada")
    predictor._check_flows(flow_ids)
    flow_ids = tuple(flow_ids)
    bounded_integer(max_source_bytes, "presupuesto de bloques fuente", 1, 2 * 1024**3)
    if not isinstance(blocks, dict) or len(blocks) != len(flow_ids) or set(blocks) != set(flow_ids):
        raise ValueError("Cada flujo pedido necesita exactamente una fuente explícita")
    decoded, selected, source_bytes = {}, [], 0
    for flow in flow_ids:
        entry = blocks[flow]
        if type(entry) is not tuple or len(entry) != 2 or type(entry[1]) is not int:
            raise ValueError("Cada fuente debe indicar su payload y una fila entera")
        payload, row = entry
        key = id(payload)
        if key not in decoded:
            state = _decode(predictor, payload, device=torch.device("cpu"))
            if any(value.requires_grad or value.grad_fn is not None for value in _tensors(state)):
                raise ValueError("Los bloques confirmados no pueden conservar un grafo")
            source_bytes += predictor._usage(state)["total_bytes"]
            if source_bytes > max_source_bytes:
                raise ValueError("Los bloques fuente superan el presupuesto agregado de bytes")
            decoded[key] = state
        state = decoded[key]
        if not 0 <= row < len(state.flow_ids) or state.flow_ids[row] != flow:
            raise ValueError("La fila de la fuente no corresponde al flujo pedido")
        selected.append((state, row))
    ids = tuple(state.last_sample_ids[row] for state, row in selected)
    moments = tuple(state.last_prediction_at[row] for state, row in selected)
    predictor._state_size(flow_ids, ids, moments)

    def collect(field):
        return torch.cat([field(state)[row : row + 1] for state, row in selected], dim=0)

    mac = selected[0][0].mac
    if mac:
        mac = replace(
            mac,
            memory=replace(
                mac.memory,
                weights=tuple(
                    collect(lambda state, layer=layer: state.mac.memory.weights[layer])
                    for layer in range(2)
                ),
                momentum=tuple(
                    collect(lambda state, layer=layer: state.mac.memory.momentum[layer])
                    for layer in range(2)
                ),
                steps=collect(lambda state: state.mac.memory.steps),
            ),
        )
    result = FinancialState(
        predictor._config_id(),
        predictor._parameter_id,
        flow_ids,
        ids,
        moments,
        collect(lambda state: state.observed_steps),
        mac,
    )
    if predictor.head.weight.device.type != "cpu":
        result = _copy_to(result, predictor.head.weight.device)
    predictor._validate_state(result)
    return result


def _block_id(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("El bloque necesita un ID de contenido SHA256")


def _validate_references(predictor, references):
    if not isinstance(references, dict) or len(references) > MAX_REFERENCE_FLOWS:
        raise ValueError("El índice supera el límite de flujos o no es un diccionario")
    rows, retained = set(), sys.getsizeof(references)
    config_id, parameter_id = predictor._config_id(), predictor._parameter_id
    for flow, record in references.items():
        predictor._check_flows((flow,))
        if not isinstance(record, dict) or set(record) != REFERENCE_FIELDS:
            raise ValueError("La referencia de estado tiene campos incompatibles")
        _block_id(record["block_id"])
        bounded_integer(record["row"], "fila de bloque", 0, predictor.config.max_batch - 1)
        bounded_integer(record["observed_steps"], "observaciones", 0, torch.iinfo(torch.int64).max)
        if record["config_id"] != config_id or record["parameter_id"] != parameter_id:
            raise ValueError("La referencia pertenece a otro contrato o parámetros")
        slot = (record["block_id"], record["row"])
        if slot in rows:
            raise ValueError("Dos flujos no pueden apuntar a la misma fila de bloque")
        rows.add(slot)
        retained += sys.getsizeof(flow) + sys.getsizeof(record)
        retained += sum(sys.getsizeof(key) for key in record)
        retained += sum(sys.getsizeof(value) for value in record.values())
        if retained > MAX_REFERENCE_BYTES:
            raise ValueError("El índice supera el presupuesto de metadatos")
    items = list(references.items())
    for start in range(0, len(items), predictor.config.max_batch):
        chunk = items[start : start + predictor.config.max_batch]
        state = FinancialState(
            config_id,
            parameter_id,
            tuple(flow for flow, _ in chunk),
            tuple(record["last_sample_id"] for _, record in chunk),
            tuple(record["last_prediction_at"] for _, record in chunk),
            torch.tensor(
                [record["observed_steps"] for _, record in chunk], dtype=torch.int64, device="cpu"
            ),
            None,
        )
        predictor._validate_cursor(state)
    if retained + len(canonical(references).encode()) > MAX_REFERENCE_BYTES:
        raise ValueError("El índice supera el presupuesto de metadatos")


def replace_state_references(predictor, references, previous, following, block_id):
    """Reemplazar exactamente los flujos avanzados y conservar las demás referencias."""
    predictor.verify_parameter_identity()
    _validate_references(predictor, references)
    _block_id(block_id)
    predictor._validate_state(previous)
    predictor._validate_state(following)
    if previous.flow_ids != following.flow_ids:
        raise ValueError("El reemplazo debe conservar los mismos flujos y su orden")
    counts, next_counts = previous.observed_steps.tolist(), following.observed_steps.tolist()
    updates = {}
    for row, flow in enumerate(previous.flow_ids):
        expected = dict(
            config_id=previous.config_id,
            parameter_id=previous.parameter_id,
            observed_steps=counts[row],
            last_prediction_at=previous.last_prediction_at[row],
            last_sample_id=previous.last_sample_ids[row],
        )
        record = references.get(flow)
        if record is None:
            if counts[row] != 0:
                raise ValueError("Un flujo nuevo requiere un estado inicial explícito")
        elif any(record[key] != value for key, value in expected.items()):
            raise ValueError("El estado anterior no corresponde a la referencia confirmada")
        if (
            next_counts[row] != counts[row] + 1
            or following.last_prediction_at[row] <= previous.last_prediction_at[row]
        ):
            raise ValueError("El reemplazo requiere una sola observación posterior")
        updates[flow] = dict(
            block_id=block_id,
            row=row,
            config_id=following.config_id,
            parameter_id=following.parameter_id,
            observed_steps=next_counts[row],
            last_prediction_at=following.last_prediction_at[row],
            last_sample_id=following.last_sample_ids[row],
        )
    result = {flow: dict(record) for flow, record in references.items()} | updates
    _validate_references(predictor, result)
    return result
