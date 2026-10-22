"""Intercambio fijo entre cuatro vistas, dos grupos y una raíz efímera."""

import math
from dataclasses import dataclass

import torch

from .config import MAX_BLOCK_ROWS, bounded_integer
from .state import check_finite, deferred_checks


@dataclass(frozen=True)
class BridgeHierarchyConfig:
    topology: str = "tree"
    local_steps: int = 2
    feedback: bool = True
    max_node_updates_per_row: int = 988

    def __post_init__(self):
        if type(self.topology) is not str or self.topology not in ("tree", "star"):
            raise ValueError("La jerarquía requiere tree o star")
        if type(self.local_steps) is not int or self.local_steps not in (1, 2, 4):
            raise ValueError("Los pasos locales deben ser 1, 2 o 4")
        if type(self.feedback) is not bool:
            raise ValueError("La realimentación necesita un booleano explícito")
        bounded_integer(self.max_node_updates_per_row, "actualizaciones jerárquicas", 1, 1596)

    def identity(self):
        return dict(
            schema_version=1,
            kind="fixed_two_level_hierarchy_v1",
            topology=self.topology,
            local_steps=self.local_steps,
            feedback=self.feedback,
            max_node_updates_per_row=self.max_node_updates_per_row,
            rounds=2,
            groups=((0, 1), (2, 3)),
            schedule="two_rounds_group_root_group_local",
            auxiliaries="zero_each_read_identity_message_no_learned_weights",
            local_update="receiver_region_value_tanh_with_residual",
            quota="local_plus_group_plus_root_initial_views_separate",
        )

    def edges(self):
        edges = [[row == column for column in range(7)] for row in range(7)]
        pairs = (
            ((0, 4), (1, 4), (2, 5), (3, 5), (4, 6), (5, 6))
            if self.topology == "tree"
            else tuple((node, 6) for node in range(6))
        )
        for first, second in pairs:
            edges[first][second] = edges[second][first] = True
        return tuple(tuple(row) for row in edges)

    def node_updates(self, regions, reads):
        if type(regions) is not int or regions != 4:
            raise ValueError("La jerarquía requiere exactamente cuatro vistas")
        bounded_integer(reads, "lecturas por fila del evento", 0, 38)
        return (8 * self.local_steps + 10) * reads


def hierarchy_options(value):
    """Interpretar únicamente los cuatro campos configurables de la jerarquía."""
    if type(value) is not dict or set(value) != {
        "topology",
        "local_steps",
        "feedback",
        "max_node_updates_per_row",
    }:
        raise ValueError("La jerarquía declara topología, pasos, realimentación y cuota")
    return BridgeHierarchyConfig(**value)


def hierarchy_graph(available, edges, *, feedback):
    """Derivar disponibilidad y el operador efectivo completo por fila."""
    if (
        not isinstance(available, torch.Tensor)
        or available.layout != torch.strided
        or available.ndim != 2
        or available.shape[1] != 4
        or available.dtype != torch.bool
        or available.device.type not in ("cpu", "cuda")
        or type(feedback) is not bool
    ):
        raise ValueError("Las cuatro vistas requieren disponibilidad booleana [B,4]")
    bounded_integer(len(available), "filas de jerarquía", 1, MAX_BLOCK_ROWS)
    if (
        not isinstance(edges, torch.Tensor)
        or edges.layout != torch.strided
        or edges.shape != (7, 7)
        or edges.dtype != torch.bool
        or edges.device != available.device
    ):
        raise ValueError("La jerarquía necesita aristas booleanas [7,7] en su dispositivo")
    groups = available.reshape(-1, 2, 2).any(-1)
    present = torch.cat((available, groups, available.any(-1, keepdim=True)), -1)
    allowed = edges[None] & present[:, :, None] & present[:, None, :]
    if not feedback:
        identity = torch.eye(7, device=available.device, dtype=torch.bool)
        allowed = torch.cat((identity[None, :4] & available[:, :, None], allowed[:, 4:]), 1)
    graph = allowed.to(torch.float64) / allowed.sum(-1, keepdim=True).clamp_min(1)
    return graph, present


def _phase(current, graph, weights, present, gain, start, stop):
    message = torch.bmm(graph[:, start:stop], current)
    check_finite(message, "El mensaje jerárquico")
    if start == 0:
        transformed = torch.einsum("brh,rhd->brd", message, weights)
        check_finite(transformed, "La transformación jerárquica antes de tanh")
        message = transformed.tanh()
    updated = (1 - gain) * current[:, start:stop] + gain * message
    check_finite(updated, "La actualización jerárquica antes de la máscara")
    updated = torch.where(present[:, start:stop, None], updated, torch.zeros_like(updated))
    return torch.cat((current[:, :start], updated, current[:, stop:]), 1)


def hierarchy_mix(local, graph, weights, available, *, local_steps, gain):
    """Ejecutar dos intercambios síncronos, conservando auxiliares entre rondas."""
    if type(local_steps) is not int or local_steps not in (1, 2, 4):
        raise ValueError("Los pasos locales deben ser 1, 2 o 4")
    if type(gain) not in (int, float) or not math.isfinite(gain) or not 0 <= gain <= 1:
        raise ValueError("La ganancia jerárquica debe pertenecer a [0,1]")
    if (
        not isinstance(local, torch.Tensor)
        or local.layout != torch.strided
        or local.ndim != 3
        or local.shape[1] != 4
        or local.dtype not in (torch.float32, torch.float64)
        or local.device.type not in ("cpu", "cuda")
        or torch.is_autocast_enabled(local.device.type)
    ):
        raise ValueError("La jerarquía necesita cuatro vistas [B,4,H] FP32 o FP64")
    batch, _, width = local.shape
    bounded_integer(batch, "filas de jerarquía", 1, MAX_BLOCK_ROWS)
    bounded_integer(width, "dimensión jerárquica", 1, 128)
    for value, shape in ((graph, (batch, 7, 7)), (weights, (4, width, width))):
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.shape != shape
            or value.dtype != local.dtype
            or value.device != local.device
        ):
            raise ValueError(
                "Grafo y pesos jerárquicos necesitan formas, precisión y dispositivo comunes"
            )
    if (
        not isinstance(available, torch.Tensor)
        or available.layout != torch.strided
        or available.shape != (batch, 4)
        or available.dtype != torch.bool
        or available.device != local.device
    ):
        raise ValueError("La disponibilidad necesita una máscara booleana [B,4]")
    with deferred_checks():
        for value in (local, graph, weights):
            check_finite(value, "Las entradas de la jerarquía")
        groups = available.reshape(batch, 2, 2).any(-1)
        present = torch.cat((available, groups, available.any(-1, keepdim=True)), -1)
        masked_local = torch.where(available[..., None], local, torch.zeros_like(local))
        current = torch.cat((masked_local, local.new_zeros((batch, 3, width))), 1)
        allowed = present[:, :, None] & present[:, None, :]
        graph = torch.where(allowed, graph, torch.zeros_like(graph))
        for _ in range(2):
            current = _phase(current, graph, weights, present, gain, 4, 6)
            current = _phase(current, graph, weights, present, gain, 6, 7)
            current = _phase(current, graph, weights, present, gain, 4, 6)
            for _ in range(local_steps):
                current = _phase(current, graph, weights, present, gain, 0, 4)
    return current
