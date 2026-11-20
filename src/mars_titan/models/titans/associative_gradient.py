"""Gradiente asociativo de la memoria de Titans en forma cerrada, sin autograd interno.

La memoria escribe cada token con el gradiente de la pérdida asociativa
ℓ = Σ_filas ||M(k; W) − v||² respecto a sus pesos rápidos. La ruta de referencia lo obtiene con
`torch.autograd.grad(create_graph=True)` sobre una copia de los pesos. Esta función escribe
las mismas derivadas como operaciones tensoriales explícitas. Con ellas, el gradiente exterior
del ajuste (el metagradiente a través del bucle interno) es la derivada de primer orden de un
grafo ordinario, que autograd, Inductor o un kernel propio pueden fusionar.

Para una fila, con la convención de `_apply_memory` (x @ Wᵀ):

    h = W₁k,  a = GELU(h),  z = W₂a                (profundidad 2; con profundidad 1, z = W₁k)
    n = LN(z) = (z − μ)·r,  r = (var(z) + ε)^(−1/2)  (sin afinidad, varianza poblacional)
    ŷ = k + n con la memoria residual, o ŷ = z sin ella
    e = ŷ − v,  ℓ = Σ e²

    g = ∂ℓ/∂ŷ = 2e
    ∂ℓ/∂z = r·(g − media(g) − n·media(g ⊙ n))      (retropropagación de LN; g sin LN)
    ∂ℓ/∂W₂ = (∂ℓ/∂z) ⊗ a
    ∂ℓ/∂h = (W₂ᵀ ∂ℓ/∂z) ⊙ GELU′(h),  GELU′(x) = Φ(x) + x·φ(x)
    ∂ℓ/∂W₁ = (∂ℓ/∂h) ⊗ k

El término directo k de la memoria residual no depende de W y no aparece en las derivadas. La
pasada directa reutiliza las mismas operaciones que `_apply_memory` (bmm, GELU exacta y
`F.layer_norm`), de modo que la predicción coincide con la de la referencia. El corte vertical
de la DGX comprobó en FP64, durante seis pasos encadenados, estados, pérdidas y metagradientes
de todos los parámetros con un error relativo máximo de 9·10⁻¹⁵.
"""

import math

import torch
from torch import Tensor
from torch.nn import functional as F

_SQRT_HALF = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def gelu_derivative(x: Tensor) -> Tensor:
    """Derivada exacta de GELU(x) = x·Φ(x): Φ(x) + x·φ(x)."""
    return 0.5 * (1.0 + torch.erf(x * _SQRT_HALF)) + x * torch.exp(-0.5 * x * x) * _INV_SQRT_2PI


def associative_gradients(
    weights: tuple[Tensor, ...],
    keys: Tensor,
    values: Tensor,
    *,
    residual_layer_norm: bool,
    eps: float,
) -> tuple[tuple[Tensor, ...], Tensor]:
    """Gradientes ∂ℓ/∂Wᵢ con forma [B, D, D] y los residuos al cuadrado e² con forma [B, D].

    `weights` son las matrices de la memoria por flujo, de profundidad 1 o 2, `keys` y
    `values` tienen forma [B, D]. Las derivadas son diferenciables: con grafo, su gradiente
    exterior es el metagradiente de la ruta de referencia.
    """
    if len(weights) not in (1, 2):
        raise ValueError("El gradiente analítico cubre memorias de profundidad 1 o 2")
    k = keys.unsqueeze(1)
    if len(weights) == 2:
        first, second = weights
        hidden = torch.bmm(k, first.transpose(1, 2))
        activated = F.gelu(hidden)
        output = torch.bmm(activated, second.transpose(1, 2))
    else:
        (first,) = weights
        output = torch.bmm(k, first.transpose(1, 2))
    if residual_layer_norm:
        normalized = F.layer_norm(output, (output.shape[-1],), eps=eps)
        inverse_std = torch.rsqrt(output.var(-1, unbiased=False, keepdim=True) + eps)
        prediction = k + normalized
    else:
        prediction = output
    residual = prediction - values.unsqueeze(1)
    upstream = 2.0 * residual
    if residual_layer_norm:
        upstream = inverse_std * (
            upstream
            - upstream.mean(-1, keepdim=True)
            - normalized * (upstream * normalized).mean(-1, keepdim=True)
        )
    if len(weights) == 2:
        second_gradient = upstream.transpose(1, 2) * activated
        hidden_gradient = torch.bmm(upstream, second) * gelu_derivative(hidden)
        gradients = (hidden_gradient.transpose(1, 2) * k, second_gradient)
    else:
        gradients = (upstream.transpose(1, 2) * k,)
    return gradients, residual.squeeze(1).square()
