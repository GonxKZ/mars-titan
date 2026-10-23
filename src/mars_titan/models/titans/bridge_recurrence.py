"""Comunicación regional no lineal, sin parámetros o memoria persistente propios."""

import math
from dataclasses import dataclass

import torch

from .config import MAX_BLOCK_ROWS, bounded_integer
from .state import check_finite, deferred_checks


def _tolerance(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name} debe ser un número finito no negativo")
    try:
        valid = math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} debe ser un número finito no negativo")
    return float(value)


@dataclass(frozen=True)
class BridgeStoppingConfig:
    atol: float
    rtol: float
    backend: str = "adaptive_reference_full"

    def __post_init__(self):
        _tolerance(self.atol, "La tolerancia absoluta")
        _tolerance(self.rtol, "La tolerancia relativa")
        if type(self.backend) is not str or self.backend not in (
            "adaptive_reference_full",
            "adaptive_compact",
        ):
            raise ValueError("El backend de parada no está admitido")

    def options(self):
        self.__post_init__()
        return dict(atol=float(self.atol), rtol=float(self.rtol), backend=self.backend)

    def identity(self):
        return dict(
            schema_version=1,
            kind="present_coordinate_linf_symmetric_scale_v1",
            predicate_dtype="torch.float64",
            derivative="selected_prefix_branch_without_a_global_halting_derivative",
            **self.options(),
        )


@dataclass(frozen=True)
class BridgeRecurrenceConfig:
    steps: int
    max_node_updates_per_row: int
    stopping: BridgeStoppingConfig | None = None

    def __post_init__(self):
        if type(self.steps) is not int or self.steps not in (1, 2, 4):
            raise ValueError("J debe ser 1, 2 o 4")
        # Dos contextos de 17 lecturas y una emisión K4, con ocho regiones y J4.
        bounded_integer(self.max_node_updates_per_row, "actualizaciones regionales", 1, 1216)
        if self.stopping is not None:
            if type(self.stopping) is not BridgeStoppingConfig:
                raise ValueError("La parada necesita BridgeStoppingConfig explícito")
            self.stopping.__post_init__()

    def identity(self):
        self.__post_init__()
        result = dict(
            schema_version=1,
            kind="regional_mix_tanh_v1",
            steps=self.steps,
            max_node_updates_per_row=self.max_node_updates_per_row,
            weights="receiver_region_value_reused_at_every_step",
            update="synchronous_fixed_graph_mixture_tanh",
        )
        if self.stopping is not None:
            result.update(
                schema_version=2,
                kind="regional_mix_tanh_adaptive_v1",
                stopping=self.stopping.identity(),
            )
        return result

    def options(self):
        self.__post_init__()
        result = dict(steps=self.steps, max_node_updates_per_row=self.max_node_updates_per_row)
        if self.stopping is not None:
            result["stopping"] = self.stopping.options()
        return result

    def node_updates(self, regions, reads):
        """Contar las lecturas del evento completo, incluida su emisión posterior."""
        bounded_integer(regions, "regiones", 1, 8)
        bounded_integer(reads, "lecturas por fila del evento", 0, 38)
        return regions * self.steps * reads


def _validate_mix(local, graph, weights, available, *, steps, gain):
    if type(steps) is not int or steps not in (1, 2, 4):
        raise ValueError("J debe ser 1, 2 o 4")
    if type(gain) not in (int, float) or not math.isfinite(gain) or not 0 <= gain <= 1:
        raise ValueError("La ganancia regional debe pertenecer a [0, 1]")
    if (
        not isinstance(local, torch.Tensor)
        or local.layout != torch.strided
        or local.ndim != 3
        or local.dtype not in (torch.float32, torch.float64)
        or local.device.type not in ("cpu", "cuda")
    ):
        raise ValueError("Los estados regionales necesitan un tensor [B,R,H] FP32 o FP64")
    batch, regions, width = local.shape
    bounded_integer(batch, "filas regionales", 1, MAX_BLOCK_ROWS)
    bounded_integer(regions, "regiones", 1, 8)
    bounded_integer(width, "dimensión regional", 1, 128)
    if torch.is_autocast_enabled(local.device.type):
        raise ValueError("La recurrencia requiere precisión explícita sin autocast")
    for value, shape in ((graph, (batch, regions, regions)), (weights, (regions, width, width))):
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.shape != shape
            or value.dtype != local.dtype
            or value.device != local.device
        ):
            raise ValueError("El grafo y los pesos deben conservar formas, precisión y dispositivo")
    if (
        not isinstance(available, torch.Tensor)
        or available.layout != torch.strided
        or available.shape != (batch, regions)
        or available.dtype != torch.bool
        or available.device != local.device
    ):
        raise ValueError("La disponibilidad regional debe ser una máscara booleana [B,R]")


def _initial_mix(local, graph, weights, available):
    for value, name in ((local, "Los estados"), (graph, "El grafo"), (weights, "Los pesos")):
        check_finite(value, name)
    present = available[..., None]
    current = torch.where(present, local, torch.zeros_like(local))
    allowed = available[:, :, None] & available[:, None, :]
    masked_graph = torch.where(allowed, graph, torch.zeros_like(graph))
    return current, masked_graph, present


def _mix_step(current, graph, weights, present, gain):
    message = torch.bmm(graph, current)
    transformed = torch.einsum("brh,rhd->brd", message, weights)
    check_finite(transformed, "La transformación recurrente antes de tanh")
    updated = (1 - gain) * current + gain * torch.tanh(transformed)
    check_finite(updated, "La mezcla recurrente antes de la máscara")
    return torch.where(present, updated, torch.zeros_like(updated))


def nonlinear_mix(local, graph, weights, available, *, steps, gain):
    """Reutilizar los pesos de la región receptora con actualizaciones síncronas."""
    _validate_mix(local, graph, weights, available, steps=steps, gain=gain)
    with deferred_checks():
        current, masked_graph, present = _initial_mix(local, graph, weights, available)
        for _ in range(steps):
            current = _mix_step(current, masked_graph, weights, present, gain)
    return current


@dataclass(frozen=True)
class AdaptiveMixResult:
    values: torch.Tensor
    selected_steps: torch.Tensor
    evaluated_steps: torch.Tensor
    criterion_met: torch.Tensor


def residual_stopping(previous, current, available, *, atol, rtol):
    """Comparar estados representables, sin derivar el predicado ni formar un umbral infinito."""
    atol = _tolerance(atol, "La tolerancia absoluta")
    rtol = _tolerance(rtol, "La tolerancia relativa")
    if (
        not isinstance(previous, torch.Tensor)
        or previous.layout != torch.strided
        or previous.ndim != 3
        or previous.dtype not in (torch.float32, torch.float64)
        or previous.device.type not in ("cpu", "cuda")
    ):
        raise ValueError("El residual necesita estados [B,R,H] FP32 o FP64")
    for value, size, name in zip(
        previous.shape, (MAX_BLOCK_ROWS, 8, 128), ("filas", "regiones", "ancho"), strict=True
    ):
        bounded_integer(value, name, 1, size)
    if (
        not isinstance(current, torch.Tensor)
        or current.layout != torch.strided
        or current.shape != previous.shape
        or current.dtype != previous.dtype
        or current.device != previous.device
        or not isinstance(available, torch.Tensor)
        or available.layout != torch.strided
        or available.shape != previous.shape[:2]
        or available.dtype != torch.bool
        or available.device != previous.device
        or torch.is_autocast_enabled(previous.device.type)
    ):
        raise ValueError(
            "El residual necesita formas, máscara, precisión y dispositivo compatibles"
        )
    with deferred_checks(), torch.no_grad():
        check_finite(previous, "El estado anterior del residual")
        check_finite(current, "El estado actual del residual")
        present = available[..., None]
        before = torch.where(present, previous.detach().to(torch.float64), 0.0)
        after = torch.where(present, current.detach().to(torch.float64), 0.0)
        difference = after - before
        check_finite(difference, "La diferencia FP64 del residual")
        delta = difference.abs().amax(dim=(1, 2))
        absolute = delta <= atol
        if rtol == 0.0:
            return absolute
        scale = torch.maximum(before.abs().amax(dim=(1, 2)), after.abs().amax(dim=(1, 2)))
        denominator = torch.where(scale > 0, scale, torch.ones_like(scale))
        relative = delta / denominator
        # En la rama pendiente atol < delta. Esta cota evita una división desbordada
        # también en las filas que ya cumplen la condición absoluta.
        absolute_part = torch.minimum(delta, torch.full_like(delta, atol)) / denominator
        threshold = absolute_part + rtol
        for value in (relative, absolute_part, threshold):
            check_finite(value, "La comparación relativa del residual")
        return absolute | (relative <= threshold)


def _adaptive_full(current, graph, weights, present, *, steps, gain, stopping):
    batch = len(current)
    selected = torch.full((batch,), steps, device=current.device, dtype=torch.int64)
    chosen = torch.zeros_like(current)
    criterion = torch.zeros(batch, device=current.device, dtype=torch.bool)
    pending = torch.ones_like(criterion)
    for depth in range(1, steps + 1):
        previous, current = current, _mix_step(current, graph, weights, present, gain)
        if depth not in (1, 2, 4):
            continue
        met = residual_stopping(
            previous, current, present[..., 0], atol=stopping.atol, rtol=stopping.rtol
        )
        take = pending & (met | (depth == steps))
        chosen = torch.where(take[:, None, None], current, chosen)
        selected = torch.where(take, depth, selected)
        criterion = torch.where(take, met, criterion)
        pending = pending & ~take
    evaluated = torch.full_like(selected, steps)
    return AdaptiveMixResult(chosen, selected, evaluated, criterion)


def _adaptive_compact(current, graph, weights, present, *, steps, gain, stopping):
    batch = len(current)
    indices = torch.arange(batch, device=current.device)
    output = current
    evaluated = torch.zeros(batch, device=current.device, dtype=torch.int64)
    criterion = torch.zeros(batch, device=current.device, dtype=torch.bool)
    for depth in range(1, steps + 1):
        previous, current = current, _mix_step(current, graph, weights, present, gain)
        output = output.index_copy(0, indices, current)
        evaluated = evaluated.index_fill(0, indices, depth)
        if depth not in (1, 2, 4):
            continue
        met = residual_stopping(
            previous, current, present[..., 0], atol=stopping.atol, rtol=stopping.rtol
        )
        criterion = criterion.index_copy(0, indices, met)
        if depth == steps:
            break
        remaining = (~met).nonzero(as_tuple=True)[0]
        if remaining.numel() == 0:
            break
        indices = indices.index_select(0, remaining)
        current = current.index_select(0, remaining)
        graph = graph.index_select(0, remaining)
        present = present.index_select(0, remaining)
    return AdaptiveMixResult(output, evaluated.clone(), evaluated, criterion)


def adaptive_mix(local, graph, weights, available, *, steps, gain, stopping):
    """Elegir prefijos por fila y declarar por separado las vueltas realmente evaluadas."""
    if type(stopping) is not BridgeStoppingConfig:
        raise ValueError("La mezcla adaptativa necesita BridgeStoppingConfig")
    stopping.__post_init__()
    _validate_mix(local, graph, weights, available, steps=steps, gain=gain)
    with deferred_checks():
        current, graph, present = _initial_mix(local, graph, weights, available)
        execute = (
            _adaptive_full if stopping.backend == "adaptive_reference_full" else _adaptive_compact
        )
        result = execute(
            current, graph, weights, present, steps=steps, gain=gain, stopping=stopping
        )
    return result
