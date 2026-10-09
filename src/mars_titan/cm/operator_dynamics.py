"""Tres lecturas separadas de los operadores de transición: punto, operador fijo y secuencia.

`numerical_radius.py` da el diagnóstico heurístico de cada operador por separado. Este
módulo añade las otras dos lecturas sin mezclarlas con la primera.

- Operador fijo. En aritmética exacta, `w(A^j) ≤ w(A)^j` y `||T||₂ ≤ 2 w(T)` implican
  `||A^j||₂ ≤ 2 w(A)^j`. Se compara esa expresión, con la estimación corregida de w(A),
  con las normas medidas de las potencias. Solo describe la repetición de una misma A.
- Dinámica variable. Para una secuencia A_1, …, A_T se miden las normas y radios
  espectrales de los productos acumulados `A_t ⋯ A_1`. Ninguna condición puntual sobre
  cada A_t acota esos productos, como muestra el par alternante A1/A2.

Las cantidades son medidas en coma flotante sobre los operadores recibidos. No certifican
cotas ni estabilidad de la red completa.
"""

from dataclasses import dataclass

import torch

from .numerical_radius import NumericalRadiusEstimates, numerical_radius_estimates

MAX_ORDER = 256
MAX_POWER = 64
MAX_STEPS = 256
_FIELDS = ("grid_lower_estimate", "angular_corrected_estimate", "frobenius_norm", "spectral_norm")


@dataclass(frozen=True)
class FixedOperatorPowers:
    """Potencias de un operador fijo frente a `2 ŵ^j`, con ŵ la estimación corregida."""

    powers: tuple[int, ...]
    estimates: NumericalRadiusEstimates
    norms: torch.Tensor
    bound: torch.Tensor
    # ŵ < 1: la expresión decrece con j. No es una certificación numérica.
    contracting_estimate: bool


@dataclass(frozen=True)
class VariableProducts:
    """Lectura puntual de cada operador y medida de los productos acumulados."""

    pointwise: NumericalRadiusEstimates
    product_norms: torch.Tensor
    product_spectral_radii: torch.Tensor
    # Todas las estimaciones corregidas por debajo del umbral y algún producto por encima.
    pointwise_below_but_product_above: bool


def _operators(value, *, batched):
    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.float64):
        raise ValueError("Los operadores deben ser tensores reales de 32 o 64 bits")
    shape = value.shape
    if (
        value.ndim != (3 if batched else 2)
        or shape[-1] != shape[-2]
        or not 1 <= shape[-1] <= MAX_ORDER
        or (batched and not 1 <= shape[0] <= MAX_STEPS)
    ):
        raise ValueError("Se admiten operadores cuadrados de orden 1–256 y hasta 256 pasos")
    if not torch.isfinite(value).all():
        raise ValueError("El operador contiene NaN o infinito")
    return value.detach().to(dtype=torch.float64)


def _estimates(matrices, grid_size):
    """Estimaciones por bloques de 16, el máximo de `numerical_radius_estimates`."""
    parts = [
        numerical_radius_estimates(matrices[start : start + 16], grid_size=grid_size)
        for start in range(0, len(matrices), 16)
    ]
    return NumericalRadiusEstimates(
        *(torch.cat([getattr(p, name) for p in parts]) for name in _FIELDS),
        sum(p.estimated_forward_bytes for p in parts),
        sum(p.estimated_saved_bytes for p in parts),
    )


def fixed_operator_powers(matrix, powers, *, grid_size=64):
    """Normas de `A^j` y la expresión `2 ŵ^j` para las potencias pedidas de una A fija."""
    operator = _operators(matrix, batched=False)
    if (
        not isinstance(powers, tuple)
        or not powers
        or any(type(j) is not int or not 1 <= j <= MAX_POWER for j in powers)
        or list(powers) != sorted(set(powers))
    ):
        raise ValueError("Las potencias son enteros crecientes de 1 a 64")
    estimates = numerical_radius_estimates(operator, grid_size=grid_size)
    estimate = estimates.angular_corrected_estimate
    norms = torch.stack(
        [torch.linalg.matrix_norm(torch.linalg.matrix_power(operator, j), ord=2) for j in powers]
    )
    exponents = torch.tensor(powers, dtype=torch.float64, device=operator.device)
    return FixedOperatorPowers(
        powers, estimates, norms, 2 * estimate.pow(exponents), bool(estimate < 1)
    )


def variable_products(matrices, *, threshold=1.0, grid_size=64):
    """Productos acumulados `A_t ⋯ A_1` de una secuencia, junto a la lectura puntual."""
    operators = _operators(matrices, batched=True)
    if type(threshold) not in (int, float) or not 0 < threshold < float("inf"):
        raise ValueError("El umbral debe ser positivo y finito")
    pointwise = _estimates(operators, grid_size)
    product = torch.eye(operators.shape[-1], dtype=torch.float64, device=operators.device)
    norms, radii = [], []
    for operator in operators:
        product = operator @ product
        norms.append(torch.linalg.matrix_norm(product, ord=2))
        radii.append(torch.linalg.eigvals(product).abs().max())
    norms, radii = torch.stack(norms), torch.stack(radii)
    below = bool((pointwise.angular_corrected_estimate < threshold).all())
    return VariableProducts(pointwise, norms, radii, below and bool((norms > threshold).any()))
