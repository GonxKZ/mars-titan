"""Comunicación regional no lineal, sin parámetros o memoria persistente propios."""

import math
from dataclasses import dataclass

import torch

from .config import MAX_BLOCK_ROWS, bounded_integer
from .state import check_finite, deferred_checks


@dataclass(frozen=True)
class BridgeRecurrenceConfig:
    steps: int
    max_node_updates_per_row: int

    def __post_init__(self):
        if type(self.steps) is not int or self.steps not in (1, 2, 4):
            raise ValueError("J debe ser 1, 2 o 4")
        # Dos contextos de 17 lecturas y una emisión K4, con ocho regiones y J4.
        bounded_integer(self.max_node_updates_per_row, "actualizaciones regionales", 1, 1216)

    def identity(self):
        return dict(
            schema_version=1,
            kind="regional_mix_tanh_v1",
            steps=self.steps,
            max_node_updates_per_row=self.max_node_updates_per_row,
            weights="receiver_region_value_reused_at_every_step",
            update="synchronous_fixed_graph_mixture_tanh",
        )

    def node_updates(self, regions, reads):
        """Contar las lecturas del evento completo, incluida su emisión posterior."""
        bounded_integer(regions, "regiones", 1, 8)
        bounded_integer(reads, "lecturas por fila del evento", 0, 38)
        return regions * self.steps * reads


def nonlinear_mix(local, graph, weights, available, *, steps, gain):
    """Reutilizar los pesos de la región receptora con actualizaciones síncronas."""
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
    with deferred_checks():
        for value, name in ((local, "Los estados"), (graph, "El grafo"), (weights, "Los pesos")):
            check_finite(value, name)
        present = available[..., None]
        current = torch.where(present, local, torch.zeros_like(local))
        allowed = available[:, :, None] & available[:, None, :]
        masked_graph = torch.where(allowed, graph, torch.zeros_like(graph))
        for _ in range(steps):
            message = torch.bmm(masked_graph, current)
            transformed = torch.einsum("brh,rhd->brd", message, weights)
            check_finite(transformed, "La transformación recurrente antes de tanh")
            updated = (1 - gain) * current + gain * torch.tanh(transformed)
            check_finite(updated, "La mezcla recurrente antes de la máscara")
            current = torch.where(present, updated, torch.zeros_like(updated))
    return current
