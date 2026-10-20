"""Representaciones centradas y competitivas con referencias numéricas explícitas."""

import torch
from torch.nn import functional as F

from .key_competition_config import KeyCompetition
from .state import check_finite


def competitive_normalize(values, config):
    """Transformar [..., D] sin alterar signos ni consumir RNG.

    La selección por magnitud usa posiciones estables en los empates. Su derivada
    mantiene fija la máscara elegida y no demuestra diferenciabilidad en cambios
    de selección. La salida sigue siendo densa y no reduce el banco de pesos.
    """
    if not isinstance(config, KeyCompetition):
        raise ValueError("Se necesita KeyCompetition")
    if (
        not isinstance(values, torch.Tensor)
        or values.ndim < 1
        or values.numel() == 0
        or values.dtype not in (torch.float32, torch.float64)
        or values.layout != torch.strided
    ):
        raise ValueError("La representación necesita un tensor real no vacío FP32 o FP64")
    width = values.shape[-1]
    retained = width if config.retained is None else config.retained
    if retained > width:
        raise ValueError("retained supera la dimensión de la representación")
    check_finite(values, "La representación de claves")
    if not config.center and retained == width:
        # Mantener el grafo y el orden de reducción originales en el control completo.
        return F.normalize(values, dim=-1, eps=1e-12)
    if config.center:
        values = values - values.mean(dim=-1, keepdim=True)
        check_finite(values, "El centrado de claves")
    if retained < width:
        if config.selection == "magnitude":
            indices = torch.argsort(values.abs(), dim=-1, descending=True, stable=True)[
                ..., :retained
            ]
        else:
            indices = torch.arange(retained, device=values.device) * width // retained
            indices = indices.expand(*values.shape[:-1], retained)
        mask = torch.zeros_like(values, dtype=torch.bool).scatter_(-1, indices, True)
        values = values * mask
    norm = values.norm(p=2, dim=-1, keepdim=True)
    check_finite(norm, "La norma de claves")
    result = values / norm.clamp_min(1e-12)
    check_finite(result, "La normalización competitiva")
    return result
