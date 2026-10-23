"""Ocho estados de trabajo anteriores a MAC sobre un operador regional compartido."""

import hashlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass, fields

import torch
from torch import nn
from torch.nn import functional as F

from ..baselines.multimodal import MODALITIES
from .config import MAX_BLOCK_ROWS, bounded_integer, canonical, require_identity
from .regional_fusion import (
    RegionalFusion,
    RegionalFusionConfig,
    validate_assignment,
    validate_assignment_request,
)
from .state import check_finite, deferred_checks, finite_indicator, is_finite

FAMILY = "pre_mac_regional_hierarchy_v1"
PROGRAM = "hierarchical_group_root_v1"
WORKSPACE_LIMIT = 128 * 1024**2
PARTITIONS = (((0, 2), (1, 3, 4)), ((0, 1), (2, 3, 4)))
WORK_FIELDS = frozenset(("local_node_evaluations", "summary_evaluations"))
EXTRA_FIELDS = frozenset(("configuration", "hidden_size", "max_batch"))


@dataclass(frozen=True)
class RegionalHierarchyConfig:
    topology: str = "tree"
    groups: tuple[tuple[int, ...], ...] = PARTITIONS[0]
    feedback: bool = True
    readout: str = "local_only"

    def __post_init__(self):
        if type(self.topology) is not str or self.topology not in ("tree", "star", "flat_control"):
            raise ValueError("La topología jerárquica no pertenece al registro")
        if type(self.readout) is not str or self.readout not in ("local_only", "common_skip"):
            raise ValueError("La salida jerárquica no pertenece al registro")
        if type(self.feedback) is not bool:
            raise ValueError("La realimentación jerárquica necesita un booleano explícito")
        if (
            type(self.groups) not in (tuple, list)
            or len(self.groups) != 2
            or any(type(group) not in (tuple, list) for group in self.groups)
            or any(type(index) is not int for group in self.groups for index in group)
        ):
            raise ValueError("Los grupos necesitan dos listas completas de posiciones enteras")
        groups = tuple(tuple(group) for group in self.groups)
        if groups not in PARTITIONS:
            raise ValueError("La partición jerárquica no pertenece al registro 2+3")
        object.__setattr__(self, "groups", groups)

    def identity(self):
        return dict(
            schema_version=1,
            kind=FAMILY,
            topology=self.topology,
            groups=[list(group) for group in self.groups],
            feedback=self.feedback,
            readout=self.readout,
            program=PROGRAM,
            nodes=[*MODALITIES, "group_0", "group_1", "root"],
            node_types=[*("local",) * 5, "group", "group", "root"],
            rounds=2,
            local_steps=2,
            phases_per_round=["group", "root", "group", "local", "local"],
            messages="masked_uniform_registered_senders_including_declared_self_edges",
            auxiliary_update="shared_summary_norm_residual_summary_update",
            local_attention="self_uniform_full_qkv_with_zero_query_and_key",
            context="explicit_confirmed_seed_clamped_across_all_local_phases",
            feedback_cut="local_receiver_rows_become_identity_and_are_renormalized",
            common_skip="context_plus_summary_to_local_of_final_root",
            state="eight_working_states_reset_per_prediction_not_persistent_memory",
            workspace=dict(
                max_bytes=WORKSPACE_LIMIT,
                formula="rows*(512*hidden_size*element_size+1024)",
                scope="conservative_logical_tensor_estimate",
                excludes=[
                    "allocator_peak",
                    "MAC",
                    "runtime",
                    "opaque_library_workspace",
                    "Python_metadata",
                    "serialized_JSON",
                ],
            ),
        )

    @classmethod
    def restore(cls, identity):
        if type(identity) is not dict:
            raise ValueError("La identidad jerárquica necesita un mapa completo")
        try:
            config = cls(
                **{name: identity[name] for name in ("topology", "groups", "feedback", "readout")}
            )
        except (KeyError, TypeError) as error:
            raise ValueError("La identidad jerárquica está incompleta") from error
        require_identity(identity, config.identity())
        return config


def validate_hierarchy_operator_config(config):
    if type(config) is not RegionalFusionConfig:
        raise ValueError("La jerarquía necesita la configuración del operador regional")
    config.__post_init__()
    if (config.topology, config.routing, config.cycles, config.local_steps, config.summary) != (
        "self",
        "uniform",
        2,
        2,
        True,
    ):
        raise ValueError(
            "La jerarquía exige operador self/uniform, dos ciclos de dos pasos y resumen"
        )


def logical_workspace_bytes(rows, hidden_size, element_size):
    """Estimar tensores lógicos, sin certificar picos ni incluir metadatos Python o JSON."""
    bounded_integer(rows, "filas de la estimación jerárquica", 0, MAX_BLOCK_ROWS)
    bounded_integer(hidden_size, "dimensión de la estimación jerárquica", 1, 512)
    if type(element_size) is not int or element_size not in (4, 8):
        raise ValueError("La estimación jerárquica exige elementos FP32 o FP64")
    return rows * (512 * hidden_size * element_size + 1024)


def _edges(config, *, device):
    senders = [{index} for index in range(8)]
    if config.topology == "tree":
        for group, members in enumerate(config.groups):
            parent = 5 + group
            senders[parent].update((*members, 7))
            for index in members:
                senders[index].add(parent)
        senders[7].update((5, 6))
    elif config.topology == "star":
        for index in range(7):
            senders[index].add(7)
        senders[7].update(range(7))
    else:
        for index in range(5):
            senders[index].add(7)
        for group, members in enumerate(config.groups):
            senders[5 + group].update(members)
        senders[7] = set(range(5))
    if not config.feedback:
        senders[:5] = [{index} for index in range(5)]
    return torch.tensor(
        [[sender in row for sender in range(8)] for row in senders], device=device, dtype=torch.bool
    )


def _availability(presence, groups):
    group_presence = torch.stack([presence[:, list(group)].any(1) for group in groups], dim=1)
    return torch.cat((presence, group_presence, presence.any(1, keepdim=True)), dim=1)


def _expected_phases():
    phases = []
    for round_index in range(2):
        for phase, receivers in (
            ("group", (5, 6)),
            ("root", (7,)),
            ("group", (5, 6)),
            ("local", (0, 1, 2, 3, 4)),
            ("local", (0, 1, 2, 3, 4)),
        ):
            version = len(phases)
            phases.append(
                dict(
                    round=round_index,
                    phase=phase,
                    receivers=receivers,
                    read_version=version,
                    write_version=version + 1,
                )
            )
    return phases


def _phase_work(phases):
    expected = _expected_phases()
    if type(phases) not in (tuple, list) or len(phases) != len(expected):
        raise ValueError("La traza no conserva las diez fases jerárquicas")
    work = dict.fromkeys(WORK_FIELDS, 0)
    for actual, required in zip(phases, expected, strict=True):
        if (
            type(actual) is not dict
            or set(actual) != set(required)
            or any(
                type(actual[name]) is not int or actual[name] != required[name]
                for name in ("round", "read_version", "write_version")
            )
            or type(actual["phase"]) is not str
            or actual["phase"] != required["phase"]
            or type(actual["receivers"]) not in (tuple, list)
            or any(type(index) is not int for index in actual["receivers"])
            or tuple(actual["receivers"]) != required["receivers"]
        ):
            raise ValueError("La fase jerárquica cambia orden, receptores o versiones")
        name = "local_node_evaluations" if actual["phase"] == "local" else "summary_evaluations"
        work[name] += len(actual["receivers"])
    return work


def _descriptor_contract(descriptor):
    if (
        type(descriptor) is not dict
        or set(descriptor) != {"schema_version", "execution", "operator", "dtype"}
        or type(descriptor["schema_version"]) is not int
        or descriptor["schema_version"] != 2
        or descriptor["dtype"] not in ("torch.float32", "torch.float64")
    ):
        raise ValueError("El descriptor jerárquico no conserva su esquema 2")
    execution, operator = descriptor["execution"], descriptor["operator"]
    for item in (execution, operator):
        if type(item) is not dict or set(item) != EXTRA_FIELDS:
            raise ValueError("El descriptor jerárquico no conserva sus dimensiones")
        bounded_integer(item["hidden_size"], "dimensión del descriptor jerárquico", 1, 512)
        bounded_integer(item["max_batch"], "lote del descriptor jerárquico", 1, MAX_BLOCK_ROWS)
    if any(execution[name] != operator[name] for name in ("hidden_size", "max_batch")):
        raise ValueError("La jerarquía y su operador no conservan dimensiones compatibles")
    config = RegionalHierarchyConfig.restore(execution["configuration"])
    try:
        local = RegionalFusionConfig(
            **{
                field.name: operator["configuration"][field.name]
                for field in fields(RegionalFusionConfig)
            }
        )
    except (KeyError, TypeError) as error:
        raise ValueError("El descriptor perdió la configuración del operador compartido") from error
    validate_hierarchy_operator_config(local)
    require_identity(operator["configuration"], local.identity())
    return (
        config,
        hashlib.sha256(canonical(execution).encode()).hexdigest(),
        hashlib.sha256(canonical(operator).encode()).hexdigest(),
    )


def verify_hierarchy_traces(rows, descriptor):
    """Validar estructura archivada sin reconstruir estados neuronales ni ejecutar módulos."""
    config, identity, operator_id = _descriptor_contract(descriptor)
    if type(rows) not in (tuple, list) or len(rows) > 8192:
        raise ValueError("El archivo jerárquico supera las 8192 filas lógicas")
    totals = dict.fromkeys(WORK_FIELDS, 0)
    row_fields = {
        "configuration_id",
        "operator_id",
        "program",
        "context_norm",
        "phases",
        "availability",
        "work",
    }
    for row in rows:
        if (
            type(row) is not dict
            or set(row) != row_fields
            or type(row["configuration_id"]) is not str
            or row["configuration_id"] != identity
            or type(row["operator_id"]) is not str
            or row["operator_id"] != operator_id
            or type(row["program"]) is not str
            or row["program"] != PROGRAM
            or type(row["context_norm"]) not in (int, float)
            or not 0 <= row["context_norm"] <= sys.float_info.max
        ):
            raise ValueError("La fila jerárquica cambia identidad, programa, esquema o norma")
        available = row["availability"]
        if (
            type(available) not in (tuple, list)
            or len(available) != 8
            or any(type(value) is not bool for value in available)
            or any(
                available[5 + group] != any(available[index] for index in members)
                for group, members in enumerate(config.groups)
            )
            or available[7] != any(available[:5])
        ):
            raise ValueError("La disponibilidad no corresponde a las hojas, grupos y raíz")
        observed = _phase_work(row["phases"])
        work = row["work"]
        if type(work) is not dict or set(work) != WORK_FIELDS:
            raise ValueError("La fila jerárquica no conserva sus dos unidades de trabajo")
        for name, count in observed.items():
            if type(work[name]) is not int or work[name] != count:
                raise ValueError("El trabajo jerárquico no corresponde a sus fases")
            totals[name] += count
    return totals


@dataclass(frozen=True)
class RegionalHierarchyResult:
    blocks: torch.Tensor
    trace: dict
    working_state: torch.Tensor


class RegionalHierarchy(nn.Module):
    """Ejecutar un calendario fijo sin poseer los pesos del operador recibido."""

    def __init__(self, hidden_size, config, *, max_batch=256):
        super().__init__()
        bounded_integer(hidden_size, "dimensión jerárquica", 1, 512)
        bounded_integer(max_batch, "lote jerárquico", 1, MAX_BLOCK_ROWS)
        if type(config) is not RegionalHierarchyConfig:
            raise ValueError("La jerarquía necesita su configuración identificada")
        config.__post_init__()
        self.hidden_size, self.max_batch, self.config = hidden_size, max_batch, config
        self.register_buffer("edges", _edges(config, device="cpu"))

    def get_extra_state(self):
        return dict(
            configuration=self.config.identity(),
            hidden_size=self.hidden_size,
            max_batch=self.max_batch,
        )

    def set_extra_state(self, state):
        require_identity(state, self.get_extra_state())

    def validate_structure(self):
        """Comprobar estructura física y permitir reparar valores con una carga válida."""
        if type(self.config) is not RegionalHierarchyConfig:
            raise ValueError("La jerarquía perdió su configuración identificada")
        self.config.__post_init__()
        bounded_integer(self.hidden_size, "dimensión jerárquica", 1, 512)
        bounded_integer(self.max_batch, "lote jerárquico", 1, MAX_BLOCK_ROWS)
        edges = self._buffers.get("edges")
        if (
            not isinstance(edges, torch.Tensor)
            or edges.shape != (8, 8)
            or edges.dtype != torch.bool
            or edges.layout != torch.strided
            or "edges" in self._non_persistent_buffers_set
            or set(dict(self.named_buffers())) != {"edges"}
            or self._modules
            or list(self.parameters())
        ):
            raise ValueError("La jerarquía no conserva únicamente su buffer geométrico")

    def validate_geometry(self):
        self.validate_structure()
        if not torch.equal(self.edges, _edges(self.config, device=self.edges.device)):
            raise ValueError("Las aristas jerárquicas no corresponden a la configuración")

    def validate_payload(self, state_dict, prefix=""):
        self.validate_structure()
        if not isinstance(state_dict, Mapping) or any(type(key) is not str for key in state_dict):
            raise ValueError("El checkpoint jerárquico necesita claves de texto")
        supplied = {
            key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)
        }
        if set(supplied) != {"edges", "_extra_state"}:
            raise ValueError("El checkpoint no conserva todo el estado jerárquico")
        require_identity(supplied["_extra_state"], self.get_extra_state())
        edges = supplied["edges"]
        if (
            not isinstance(edges, torch.Tensor)
            or edges.shape != (8, 8)
            or edges.dtype != torch.bool
            or edges.layout != torch.strided
        ):
            raise ValueError("El checkpoint no conserva la forma y tipo de la geometría")
        if not torch.equal(edges, _edges(self.config, device=edges.device)):
            raise ValueError("El checkpoint cambia las aristas de la familia jerárquica")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        validate_assignment_request(state_dict, assign)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args, **kwargs):
        validate_assignment(state_dict, prefix, local_metadata)
        self.validate_payload(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, *args, **kwargs)

    def _validate_operator(self, operator):
        self.validate_geometry()
        if (
            type(operator) is not RegionalFusion
            or operator.hidden_size != self.hidden_size
            or operator.max_batch != self.max_batch
        ):
            raise ValueError("La jerarquía necesita su operador y dimensiones compatibles")
        validate_hierarchy_operator_config(operator.config)
        operator.validate_geometry()
        if self.edges.device != operator.qkv.weight.device or operator.qkv.weight.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError(
                "La jerarquía y operador necesitan el mismo dispositivo y precisión explícita"
            )

    def trace_descriptor(self, operator):
        self._validate_operator(operator)
        descriptor = dict(
            schema_version=2,
            execution=self.get_extra_state(),
            operator=operator.get_extra_state(),
            dtype=str(operator.qkv.weight.dtype),
        )
        _descriptor_contract(descriptor)
        return descriptor

    def verify_exported_traces(self, rows, operator):
        return verify_hierarchy_traces(rows, self.trace_descriptor(operator))

    def export_trace(self, trace, operator):
        descriptor = self.trace_descriptor(operator)
        _, identity, operator_id = _descriptor_contract(descriptor)
        if (
            type(trace) is not dict
            or set(trace)
            != {"configuration_id", "programs", "context_norms", "phases", "availability", "work"}
            or trace["configuration_id"] != identity
            or type(trace["programs"]) is not tuple
            or not 1 <= len(trace["programs"]) <= self.max_batch
            or any(program != PROGRAM for program in trace["programs"])
        ):
            raise ValueError("La traza jerárquica no conserva su configuración y filas")
        count = len(trace["programs"])
        norms, availability, phases, work = (
            trace[name] for name in ("context_norms", "availability", "phases", "work")
        )
        if (
            not isinstance(norms, torch.Tensor)
            or norms.shape != (count,)
            or norms.dtype != operator.qkv.weight.dtype
            or norms.layout != torch.strided
            or norms.requires_grad
            or norms.grad_fn is not None
            or not is_finite(finite_indicator(norms))
            or not isinstance(availability, torch.Tensor)
            or availability.shape != (count, 8)
            or availability.dtype != torch.bool
            or availability.layout != torch.strided
            or type(phases) is not dict
            or set(phases) != {PROGRAM}
            or type(work) is not dict
            or set(work) != WORK_FIELDS
        ):
            raise ValueError("La traza jerárquica perdió normas, disponibilidad, fases o trabajo")
        per_row = _phase_work(phases[PROGRAM])
        if any(
            type(work[name]) is not int or work[name] != count * value
            for name, value in per_row.items()
        ):
            raise ValueError("El agregado jerárquico no coincide con las fases observadas")
        rows = tuple(
            dict(
                configuration_id=identity,
                operator_id=operator_id,
                program=PROGRAM,
                context_norm=norm,
                phases=[
                    dict(record, receivers=list(record["receivers"])) for record in phases[PROGRAM]
                ],
                availability=present,
                work=dict(per_row),
            )
            for norm, present in zip(norms.cpu().tolist(), availability.cpu().tolist(), strict=True)
        )
        verify_hierarchy_traces(rows, descriptor)
        return rows

    @staticmethod
    def _record(phases, work, rows, round_index, phase, receivers):
        version = len(phases)
        phases.append(
            dict(
                round=round_index,
                phase=phase,
                receivers=receivers,
                read_version=version,
                write_version=version + 1,
            )
        )
        resource = "local_node_evaluations" if phase == "local" else "summary_evaluations"
        work[resource] += rows * len(receivers)

    @staticmethod
    def _auxiliary(operator, state, weights, availability, receivers):
        start, stop = receivers[0], receivers[-1] + 1
        current = state[:, start:stop]
        message = torch.bmm(weights[:, start:stop], state)
        update = operator.summary_update(operator.summary_norm(current) + message)
        check_finite(update, "La actualización auxiliar jerárquica")
        updated = current + operator.config.summary_gain * update
        check_finite(updated, "El estado auxiliar jerárquico anterior a la máscara")
        masked = torch.where(availability[:, start:stop, None], updated, torch.zeros_like(updated))
        return torch.cat((state[:, :start], masked, state[:, stop:]), dim=1)

    @staticmethod
    def _local(operator, state, weights, presence, context):
        message = torch.bmm(weights[:, :5], state)
        current = state[:, :5]
        argument = current + context[:, None] + operator.summary_to_local(message)
        normalized = operator.local_norm(argument)
        check_finite(normalized, "La normalización local jerárquica")
        normalized = torch.where(presence[..., None], normalized, torch.zeros_like(normalized))
        projections = operator.qkv(normalized)
        check_finite(projections, "La proyección QKV jerárquica")
        query, key, value = projections.chunk(3, dim=-1)
        query, key = torch.zeros_like(query), torch.zeros_like(key)
        messages = F.scaled_dot_product_attention(
            query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1), attn_mask=operator.edges
        ).squeeze(1)
        updated = current + operator.config.local_gain * operator.local_out(messages)
        check_finite(updated, "La actualización regional local")
        local = torch.where(presence[..., None], updated, torch.zeros_like(updated))
        check_finite(local, "El estado local jerárquico")
        return torch.cat((local, state[:, 5:]), dim=1)

    def forward(self, operator, blocks, presence, context):
        self._validate_operator(operator)
        presence = operator._checked_presence(blocks, presence)
        if (
            len(blocks) > self.max_batch
            or not isinstance(context, torch.Tensor)
            or context.layout != torch.strided
            or context.shape != (len(blocks), self.hidden_size)
            or context.dtype != blocks.dtype
            or context.device != blocks.device
        ):
            raise ValueError("El contexto jerárquico no corresponde a sus bloques")
        if torch.is_autocast_enabled(blocks.device.type):
            raise ValueError("La jerarquía exige FP32 o FP64 explícitos, sin autocast")
        if (
            logical_workspace_bytes(len(blocks), self.hidden_size, blocks.element_size())
            > WORKSPACE_LIMIT
        ):
            raise ValueError("La estimación de tensores lógicos supera los 128 MiB jerárquicos")
        norms = torch.linalg.vector_norm(context.detach(), dim=-1)
        indicators = torch.stack([finite_indicator(value) for value in (blocks, context, norms)])
        if not is_finite(indicators.amax()):
            raise ValueError("Los bloques, contexto y norma jerárquicos deben ser finitos")
        availability = _availability(presence, self.config.groups)
        mask = self.edges[None] & availability[:, :, None] & availability[:, None, :]
        weights = mask.to(blocks.dtype) / mask.sum(-1, keepdim=True).clamp_min(1).to(blocks.dtype)
        initial = torch.where(presence[..., None], blocks, torch.zeros_like(blocks))
        state = torch.cat((initial, blocks.new_zeros(len(blocks), 3, self.hidden_size)), dim=1)
        phases, work = [], dict.fromkeys(WORK_FIELDS, 0)
        with deferred_checks():
            for round_index in range(2):
                for phase, receivers in (("group", (5, 6)), ("root", (7,)), ("group", (5, 6))):
                    state = self._auxiliary(operator, state, weights, availability, receivers)
                    self._record(phases, work, len(blocks), round_index, phase, receivers)
                for _ in range(2):
                    state = self._local(operator, state, weights, presence, context)
                    self._record(phases, work, len(blocks), round_index, "local", (0, 1, 2, 3, 4))
            difference = state[:, :5] - initial
            if self.config.readout == "common_skip":
                difference = (
                    difference + (context + operator.summary_to_local(state[:, 7]))[:, None]
                )
            projected = operator.readout(difference)
            check_finite(projected, "La proyección jerárquica")
            residual = blocks + projected
            check_finite(residual, "La suma residual jerárquica")
            result = torch.where(presence[..., None], residual, blocks)
            check_finite(result, "La salida jerárquica")
        trace = dict(
            configuration_id=hashlib.sha256(canonical(self.get_extra_state()).encode()).hexdigest(),
            programs=(PROGRAM,) * len(blocks),
            context_norms=norms.detach().clone(),
            phases={PROGRAM: phases},
            availability=availability.clone(),
            work=work,
        )
        return RegionalHierarchyResult(result, trace, state)
