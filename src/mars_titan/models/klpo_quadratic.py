"""Términos del objetivo KLPO exacto para un centro afín, sin resolver un ajuste."""

import math
from typing import NamedTuple

import torch

from .klpo import _inputs

CONTRACT = "klpo_exact_center_quadratic_v1"
REFERENCE_COMMIT = "30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696"
MAX_ELEMENTS = 262144


class QuadraticTerms(NamedTuple):
    """Coeficientes FP64 por fila de F(c) = constant + linear*c + curvature*c²/2."""

    constant: torch.Tensor
    linear: torch.Tensor
    curvature: torch.Tensor


def quadratic_terms(logq, rewards, values, scale, beta):
    """Descomponer la varianza exacta con q, recompensa y rejilla fijas.

    El centro c conserva sus unidades originales. La gaussiana discreta tiene
    anchura scale y beta debe ser positiva. Se admiten FP32/FP64 en CPU o CUDA,
    hasta 4096 filas y acciones, y 262144 elementos en su producto. La cota
    limita los buffers del cálculo, no el RSS ni los datos del llamante.
    No se invierte la curvatura ni se modifican parámetros o generadores.
    """
    if (
        not isinstance(logq, torch.Tensor)
        or logq.layout != torch.strided
        or logq.ndim != 2
        or logq.device.type not in {"cpu", "cuda"}
    ):
        raise ValueError("La política histórica necesita una matriz CPU o CUDA")
    if logq.numel() > MAX_ELEMENTS:
        raise ValueError("Los términos cuadráticos superan el presupuesto de elementos")
    if (
        type(scale) not in (int, float)
        or not math.isfinite(scale)
        or scale <= 0
        or not isinstance(values, torch.Tensor)
        or values.layout != torch.strided
        or values.shape != (logq.shape[1],)
        or values.device != logq.device
    ):
        raise ValueError("La rejilla necesita dimensiones, dispositivo y escala compatibles")
    for value in (logq, rewards, values):
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError("Los términos cuadráticos admiten solamente FP32 y FP64")
    if not torch.isfinite(values).all() or not (values[1:] > values[:-1]).all():
        raise ValueError("La rejilla necesita valores finitos estrictamente crecientes")
    _, logq, rewards = _inputs(logq.detach(), logq, rewards, beta)
    q = logq.exp()
    z = values.detach().double() / scale
    centered_z = z[None, :] - (q * z).sum(1, keepdim=True)
    intercept = rewards + beta * logq + (beta / 2) * z.square()
    centered = intercept - (q * intercept).sum(1, keepdim=True)
    terms = QuadraticTerms(
        (q * centered.square()).sum(1) / (2 * beta),
        -(q * centered_z * centered).sum(1) / scale,
        beta * (q * centered_z.square()).sum(1) / scale / scale,
    )
    if any(not torch.isfinite(value).all() for value in terms):
        raise ValueError("Los términos cuadráticos han desbordado la representación FP64")
    return terms
