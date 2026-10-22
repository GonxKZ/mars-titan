"""Calendarios regionales finitos sobre un operador compartido, anteriores a MAC."""

import hashlib
import sys
from dataclasses import asdict, dataclass, fields
from types import MappingProxyType

import torch
from torch import nn

from .config import MAX_BLOCK_ROWS, bounded_integer, canonical, require_identity
from .regional_fusion import RegionalFusion, RegionalFusionConfig
from .state import deferred_checks, finite_indicator, is_finite

PROGRAMS = MappingProxyType(
    {
        "summary_at_end_v1": (1, 4),
        "summary_interleaved_v1": (2, 2),
    }
)


@dataclass(frozen=True)
class RegionalExecutionConfig:
    selection: str = "fixed"
    program: str = "summary_at_end_v1"
    context_threshold: float | None = None

    def __post_init__(self):
        if type(self.selection) is not str or self.selection not in ("fixed", "context_norm_v1"):
            raise ValueError("La selección regional no pertenece al registro")
        if type(self.program) is not str or self.program not in PROGRAMS:
            raise ValueError("El programa regional no pertenece al registro")
        if self.selection == "fixed":
            if self.context_threshold is not None:
                raise ValueError("El programa fijo no utiliza un umbral de contexto")
        elif (
            self.program != "summary_at_end_v1"
            or type(self.context_threshold) not in (float, int)
            or not 0 <= self.context_threshold <= sys.float_info.max
        ):
            raise ValueError("La selección contextual necesita fallback y umbral explícitos")

    def identity(self):
        return dict(
            schema_version=1,
            kind="pre_mac_regional_program_v1",
            **asdict(self),
            programs={
                name: dict(cycles=cycles, local_steps=steps)
                for name, (cycles, steps) in PROGRAMS.items()
            },
            context="explicit_confirmed_seed_reused_after_each_summary",
            selection_gradient="discrete_not_differentiated",
            state="working_only_reset_per_prediction",
            operator="shared_regional_fusion_weights_topology_routing_and_gains",
            schedule="registered_program_replaces_operator_default_cycles_and_local_steps",
        )

    @classmethod
    def restore(cls, identity):
        try:
            config = cls(
                **{name: identity[name] for name in ("selection", "program", "context_threshold")}
            )
            require_identity(identity, config.identity())
        except (KeyError, TypeError) as error:
            raise ValueError("La identidad del programa está incompleta") from error
        return config


@dataclass(frozen=True)
class RegionalExecutionResult:
    blocks: torch.Tensor
    trace: dict


def _descriptor_contract(descriptor):
    if (
        type(descriptor) is not dict
        or set(descriptor) != {"schema_version", "execution", "operator", "dtype"}
        or type(descriptor["schema_version"]) is not int
        or descriptor["schema_version"] != 1
        or descriptor["dtype"] not in ("torch.float32", "torch.float64")
    ):
        raise ValueError("El descriptor regional no conserva su esquema")
    execution, operator = descriptor["execution"], descriptor["operator"]
    for item in (execution, operator):
        if type(item) is not dict or set(item) != {"configuration", "hidden_size", "max_batch"}:
            raise ValueError("El descriptor regional no conserva dimensiones y configuración")
        bounded_integer(item["hidden_size"], "dimensión del descriptor", 1, 512)
        bounded_integer(item["max_batch"], "lote del descriptor", 1, MAX_BLOCK_ROWS)
    if any(execution[name] != operator[name] for name in ("hidden_size", "max_batch")):
        raise ValueError("El programa y operador archivados no son compatibles")
    config = RegionalExecutionConfig.restore(execution["configuration"])
    try:
        fusion = RegionalFusionConfig(
            **{
                field.name: operator["configuration"][field.name]
                for field in fields(RegionalFusionConfig)
            }
        )
        require_identity(operator["configuration"], fusion.identity())
    except (KeyError, TypeError) as error:
        raise ValueError("El descriptor no conserva el operador regional") from error
    dtype = torch.float32 if descriptor["dtype"] == "torch.float32" else torch.float64
    threshold = config.context_threshold
    if threshold is not None:
        # La comparación original redondea el umbral a la precisión de la norma.
        threshold = (
            float("inf")
            if threshold > torch.finfo(dtype).max
            else torch.scalar_tensor(float(threshold), dtype=dtype, device="cpu").item()
        )
    return (
        config,
        fusion.summary,
        hashlib.sha256(canonical(execution).encode()).hexdigest(),
        hashlib.sha256(canonical(operator).encode()).hexdigest(),
        dtype,
        threshold,
    )


def verify_exported_traces(rows, descriptor):
    """Comprobar el contrato archivado, sin autenticar contextos o kernels no guardados."""
    config, summary, identity, operator_id, _, threshold = _descriptor_contract(descriptor)
    if type(rows) not in (tuple, list) or len(rows) > 8192:
        raise ValueError("La traza excede el tamaño de cohorte admitido")
    totals = dict(local_node_evaluations=0, summary_evaluations=0)
    fields = {"configuration_id", "operator_id", "program", "context_norm", "phases", "work"}
    expected_phases = {
        program: tuple(
            phase
            for _ in range(cycles)
            for phase in (
                *("local",) * steps,
                *(("summary",) if summary else ()),
            )
        )
        for program, (cycles, steps) in PROGRAMS.items()
    }
    for row in rows:
        if (
            type(row) is not dict
            or set(row) != fields
            or row["configuration_id"] != identity
            or row["operator_id"] != operator_id
            or type(row["context_norm"]) not in (int, float)
            or not 0 <= row["context_norm"] <= sys.float_info.max
        ):
            raise ValueError("La fila regional cambia su identidad, esquema o norma")
        program = config.program
        if config.selection == "context_norm_v1":
            program = (
                "summary_interleaved_v1" if row["context_norm"] > threshold else "summary_at_end_v1"
            )
        phases, work = row["phases"], row["work"]
        if (
            row["program"] != program
            or type(phases) not in (tuple, list)
            or tuple(phases) != expected_phases[program]
            or type(work) is not dict
            or set(work) != set(totals)
        ):
            raise ValueError("Las fases regionales no corresponden a la selección registrada")
        observed = dict(
            local_node_evaluations=5 * phases.count("local"),
            summary_evaluations=phases.count("summary"),
        )
        for name, count in observed.items():
            if type(work[name]) is not int or work[name] != count:
                raise ValueError("El trabajo regional no corresponde a las fases archivadas")
            totals[name] += count
    return totals


class RegionalExecution(nn.Module):
    """Elegir un programa por fila sin parámetros adicionales ni código arbitrario."""

    def __init__(self, hidden_size, config, *, max_batch=256):
        super().__init__()
        bounded_integer(hidden_size, "dimensión del programa regional", 1, 512)
        bounded_integer(max_batch, "lote del programa regional", 1, MAX_BLOCK_ROWS)
        if type(config) is not RegionalExecutionConfig:
            raise ValueError("El ejecutor regional necesita una configuración identificada")
        config.__post_init__()
        self.hidden_size, self.max_batch, self.config = hidden_size, max_batch, config

    def get_extra_state(self):
        return dict(
            configuration=self.config.identity(),
            hidden_size=self.hidden_size,
            max_batch=self.max_batch,
        )

    def set_extra_state(self, state):
        require_identity(state, self.get_extra_state())

    def _trace_contract(self, operator):
        _, _, identity, operator_id, dtype, threshold = _descriptor_contract(
            self.trace_descriptor(operator)
        )
        return identity, operator_id, dtype, threshold

    def trace_descriptor(self, operator):
        """Describir precisión, dimensiones y contratos sin copiar parámetros."""
        if type(operator) is not RegionalFusion or operator.hidden_size != self.hidden_size:
            raise ValueError("El descriptor necesita el mismo operador regional compatible")
        descriptor = dict(
            schema_version=1,
            execution=self.get_extra_state(),
            operator=operator.get_extra_state(),
            dtype=str(operator.qkv.weight.dtype),
        )
        _descriptor_contract(descriptor)
        return descriptor

    def verify_exported_traces(self, rows, operator):
        return verify_exported_traces(rows, self.trace_descriptor(operator))

    def export_trace(self, trace, operator):
        """Validar y exportar fases observadas, sin atribuirles FLOPs o estados internos."""
        identity, operator_id, dtype, _ = self._trace_contract(operator)
        if (
            type(trace) is not dict
            or set(trace) != {"configuration_id", "programs", "context_norms", "phases", "work"}
            or trace["configuration_id"] != identity
            or type(operator) is not RegionalFusion
            or type(trace["programs"]) is not tuple
            or not 1 <= len(trace["programs"]) <= self.max_batch
            or any(program not in PROGRAMS for program in trace["programs"])
        ):
            raise ValueError("La traza del programa no conserva su contrato")
        programs, norms, phases, work = (
            trace["programs"],
            trace["context_norms"],
            trace["phases"],
            trace["work"],
        )
        if (
            not isinstance(norms, torch.Tensor)
            or norms.shape != (len(programs),)
            or norms.dtype != dtype
            or norms.layout != torch.strided
            or norms.requires_grad
            or norms.grad_fn is not None
            or not is_finite(finite_indicator(norms))
            or type(phases) is not dict
            or set(phases) != set(programs)
            or type(work) is not dict
            or set(work) != {"local_node_evaluations", "summary_evaluations"}
        ):
            raise ValueError("La traza regional no conserva normas y contadores")
        totals = dict.fromkeys(work, 0)
        norms_cpu = norms.cpu().tolist()
        rows = []
        for program, norm in zip(programs, norms_cpu, strict=True):
            cycles, steps = PROGRAMS[program]
            expected = tuple(
                phase
                for _ in range(cycles)
                for phase in (
                    *("local",) * steps,
                    *(("summary",) if operator.config.summary else ()),
                )
            )
            if type(phases[program]) not in (tuple, list) or tuple(phases[program]) != expected:
                raise ValueError("La secuencia de fases no corresponde al programa ejecutado")
            observed = dict(
                local_node_evaluations=5 * phases[program].count("local"),
                summary_evaluations=phases[program].count("summary"),
            )
            for name, count in observed.items():
                totals[name] += count
            rows.append(
                dict(
                    configuration_id=identity,
                    operator_id=operator_id,
                    program=program,
                    context_norm=norm,
                    phases=tuple(phases[program]),
                    work=observed,
                )
            )
        if any(type(work[name]) is not int or work[name] != totals[name] for name in totals):
            raise ValueError("La traza cambia el trabajo de las fases observadas")
        self.verify_exported_traces(rows, operator)
        return tuple(rows)

    def forward(self, operator, blocks, presence, context):
        if type(self.config) is not RegionalExecutionConfig:
            raise ValueError("El ejecutor perdió su configuración identificada")
        self.config.__post_init__()
        if type(operator) is not RegionalFusion or operator.hidden_size != self.hidden_size:
            raise ValueError("El programa necesita su operador regional compatible")
        presence = operator._checked_presence(blocks, presence)
        if (
            len(blocks) > self.max_batch
            or not isinstance(context, torch.Tensor)
            or context.layout != torch.strided
            or context.shape != (len(blocks), self.hidden_size)
            or context.dtype != blocks.dtype
            or context.device != blocks.device
        ):
            raise ValueError("El contexto del programa no corresponde a sus bloques")
        norms = torch.linalg.vector_norm(context, dim=-1)
        indicators = torch.stack([finite_indicator(value) for value in (blocks, context, norms)])
        if not is_finite(indicators.amax()):
            raise ValueError("Los bloques, contexto y norma regional deben ser finitos")
        with deferred_checks():
            programs = (self.config.program,) * len(blocks)
            if self.config.selection == "context_norm_v1":
                # La decisión discreta pasa al anfitrión una vez, sin cortar el contexto usado.
                selected = (norms.detach() > float(self.config.context_threshold)).tolist()
                programs = tuple(
                    "summary_interleaved_v1" if chosen else "summary_at_end_v1"
                    for chosen in selected
                )
            result = torch.zeros_like(blocks)
            phases, work = {}, dict(local_node_evaluations=0, summary_evaluations=0)
            for program in sorted(set(programs)):
                rows = [index for index, chosen in enumerate(programs) if chosen == program]
                indices = torch.tensor(rows, dtype=torch.int64, device=blocks.device)
                phases[program] = []
                count = len(rows)

                def observe(phase, *, selected_program=program, row_count=count):
                    phases[selected_program].append(phase)
                    if phase == "local":
                        work["local_node_evaluations"] += row_count * blocks.shape[1]
                    else:
                        work["summary_evaluations"] += row_count

                cycles, local_steps = PROGRAMS[program]
                transformed = operator._execute(
                    blocks.index_select(0, indices),
                    presence.index_select(0, indices),
                    cycles=cycles,
                    local_steps=local_steps,
                    context_seed=context.index_select(0, indices),
                    observe=observe,
                )
                result = result.index_copy(0, indices, transformed)
        return RegionalExecutionResult(
            result,
            dict(
                configuration_id=hashlib.sha256(
                    canonical(self.get_extra_state()).encode()
                ).hexdigest(),
                programs=programs,
                context_norms=norms.detach().clone(),
                phases=phases,
                work=work,
            ),
        )
