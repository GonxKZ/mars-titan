"""Estimaciones del radio numérico y penalizaciones sobre operadores proporcionados.

La corrección angular procede de una desigualdad en aritmética exacta. Los
resultados de coma flotante no son cotas certificadas. Ninguna función elige
el operador de un modelo ni demuestra estabilidad de una dinámica variable.
"""

import math
from dataclasses import dataclass
from numbers import Real

import torch


@dataclass(frozen=True)
class NumericalRadiusEstimates:
    """Cantidades por operador y estimaciones de bytes de tensores propios."""

    grid_lower_estimate: torch.Tensor
    angular_corrected_estimate: torch.Tensor
    frobenius_norm: torch.Tensor
    spectral_norm: torch.Tensor
    estimated_forward_bytes: int
    estimated_saved_bytes: int


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} debe ser un entero positivo")


def numerical_radius_estimates(
    matrix: torch.Tensor,
    *,
    grid_size: int = 64,
    angle_block_size: int = 8,
    max_estimated_bytes: int = 128 * 1024**2,
) -> NumericalRadiusEstimates:
    """Calcula máximos hermíticos por bloques conservando dispositivo y gradientes.

    Se acepta float32, float64, complex64 o complex128, con hasta 16 matrices
    de orden 256 y 2–1024 ángulos. La referencia utiliza complex128. La memoria
    estimada incluye tensores guardados para autograd, pero no certifica el
    espacio interno del solver ni la memoria total del proceso.
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("Se necesita un tensor PyTorch")
    if matrix.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        raise TypeError("El operador debe tener tipo real o complejo de 32 o 64 bits")
    if matrix.ndim < 2 or matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError("El operador debe ser cuadrado")
    order = matrix.shape[-1]
    batch = math.prod(matrix.shape[:-2])
    if not 1 <= order <= 256 or not 1 <= batch <= 16:
        raise ValueError("Se admiten órdenes 1–256 y lotes de 1–16 matrices")
    for value, name in (
        (grid_size, "grid_size"),
        (angle_block_size, "angle_block_size"),
        (max_estimated_bytes, "max_estimated_bytes"),
    ):
        _positive_integer(value, name)
    if not 2 <= grid_size <= 1024:
        raise ValueError("grid_size debe estar entre 2 y 1024")
    block = min(grid_size, angle_block_size)
    matrix_bytes = batch * order**2 * 16
    forward_bytes = matrix_bytes * (8 + 4 * block) + 8 * block * batch * order + 32 * grid_size
    # Las fases y reducciones también se retienen, incluso para matrices de orden 1.
    saved_bytes = (
        matrix_bytes * (4 * grid_size + 8) + 32 * grid_size * (batch + 1)
        if matrix.requires_grad and torch.is_grad_enabled()
        else 0
    )
    if forward_bytes + saved_bytes > max_estimated_bytes:
        raise ValueError("La estimación de memoria del operador supera el presupuesto")
    if not torch.isfinite(matrix).all():
        raise ValueError("El operador contiene NaN o infinito")

    reference = matrix.to(dtype=torch.complex128)
    frobenius = torch.linalg.matrix_norm(reference, ord="fro")
    spectral = torch.linalg.matrix_norm(reference, ord=2)
    lower = None
    for start in range(0, grid_size, block):
        angles = torch.arange(
            start, min(start + block, grid_size), dtype=torch.float64, device=matrix.device
        ) * (2 * math.pi / grid_size)
        phases = torch.exp(-1j * angles).reshape((-1,) + (1,) * matrix.ndim)
        hermitian = (0.5 * phases) * reference + (0.5 * phases.conj()) * reference.mH
        maximum = torch.linalg.eigvalsh(hermitian)[..., -1].amax(dim=0)
        lower = maximum if lower is None else torch.maximum(lower, maximum)
    lower = lower.clamp_min(0)
    corrected = lower + 2 * frobenius * math.sin(math.pi / (2 * grid_size))
    if not all(torch.isfinite(value).all() for value in (lower, corrected, frobenius, spectral)):
        raise ValueError("El cálculo del operador produjo un resultado no finito")
    return NumericalRadiusEstimates(
        lower, corrected, frobenius, spectral, forward_bytes, saved_bytes
    )


def radius_penalty(
    estimates: NumericalRadiusEstimates,
    threshold: float,
    *,
    measure: str = "angular_corrected_estimate",
) -> torch.Tensor:
    """Aplica una bisagra cuadrática por operador, sin modificar sus pesos.

    Autograd conserva una elección de subgradiente en ciertos empates. En
    autovalores máximos múltiples o empates angulares no se promete una
    derivada única ni diferenciabilidad en todo punto.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise ValueError("El umbral debe ser un número real finito no negativo")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("El umbral debe ser un número real finito no negativo")
    if measure not in ("grid_lower_estimate", "angular_corrected_estimate", "spectral_norm"):
        raise ValueError("La medida de penalización no está admitida")
    penalty = (getattr(estimates, measure) - threshold).clamp_min(0).square()
    if not torch.isfinite(penalty).all():
        raise ValueError("La penalización produjo un resultado no finito")
    return penalty
