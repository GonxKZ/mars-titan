"""Escritura asociativa de la memoria de Titans en un único kernel por flujo.

Un paso de escritura de la memoria de profundidad 2, con la convención x @ Wᵀ:

    h = W₁k,  a = GELU(h),  z = W₂a
    n = LN(z) = (z − μ)·r,  r = (var(z) + ε)^(−1/2),  ŷ = k + n   (sin la capa: ŷ = z)
    e = ŷ − v,  g = 2e,  δz = r·(g − media(g) − n·media(g ⊙ n))   (sin la capa: δz = g)
    G₂ = δz aᵀ,  δa = W₂ᵀδz,  δh = δa ⊙ GELU′(h),  G₁ = δh kᵀ
    S′ᵢ = η Sᵢ − θ Gᵢ,  W′ᵢ = (1 − α) ⊙ Wᵢ + S′ᵢ    (α por fila de salida, η y θ por flujo)

`associative_gradient.py` escribe las mismas derivadas con operaciones de PyTorch, y cada
operación lee y escribe sus matrices en memoria. Aquí un programa de Triton por flujo lee W₁,
W₂, S₁ y S₂ una sola vez y escribe W′ y S′: con D = 64 son 128 KiB por flujo y paso, el
mínimo que exige la escritura en una GB10 limitada por el ancho de banda de su memoria.

El gradiente exterior es el VJP cerrado del paso. El kernel inverso recalcula la pasada
directa de su flujo (no se guarda ningún intermedio) y propaga, con Pᵢ = dS′ᵢ + dW′ᵢ:

    dSᵢ = η Pᵢ,  dη = Σᵢ ⟨Sᵢ, Pᵢ⟩,  dθ = −Σᵢ ⟨Gᵢ, Pᵢ⟩,  dα_r = −Σᵢ Σ_c Wᵢ[r, c]·dW′ᵢ[r, c]
    dWᵢ = (1 − α) ⊙ dW′ᵢ más la propagación de dGᵢ = −θ Pᵢ por Gᵢ, δa, δh, la retropropagación
    de LN que define δz y la pasada directa hasta W₁, W₂, k y v

con GELU″(x) = φ(x)·(2 − x²). Cada programa escribe solo los gradientes de su flujo, sin
operaciones atómicas, así que el resultado es determinista. La segunda derivada no existe:
el ajuste solo necesita la primera, y pedirla lanza un error en lugar de aproximarla.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable

DIMENSION = 64
_DTYPES = (torch.float32, torch.float64)
# Un programa por flujo con las matrices de 64 × 64 en registros: cuatro warps bastan para la
# pasada directa, que tiene dos matrices vivas, y ocho para la inversa, que mantiene cuatro.
_FORWARD_WARPS = 4
_BACKWARD_WARPS = 8


@triton.jit
def _gelu_terms(h):
    """GELU(h), GELU′(h) y GELU″(h) exactas, con Φ por la función de error."""
    cdf = 0.5 * (1.0 + tl.erf(h * 0.7071067811865476))
    pdf = tl.exp(-0.5 * h * h) * 0.3989422804014327
    return h * cdf, cdf + h * pdf, pdf * (2.0 - h * h)


@triton.jit
def _forward_kernel(
    w1_ptr,
    w2_ptr,
    s1_ptr,
    s2_ptr,
    k_ptr,
    v_ptr,
    alpha_ptr,
    eta_ptr,
    theta_ptr,
    w1_out,
    w2_out,
    s1_out,
    s2_out,
    elements_out,
    EPS: tl.constexpr,
    D: tl.constexpr,
    LAYER_NORM: tl.constexpr,
):
    flow = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, D)
    cols = tl.arange(0, D)
    square = flow * D * D + rows[:, None] * D + cols[None, :]
    vector = flow * D
    k_in = tl.load(k_ptr + vector + cols)
    keep = 1.0 - tl.load(alpha_ptr + vector + rows)
    eta = tl.load(eta_ptr + flow)
    theta = tl.load(theta_ptr + flow)

    w1 = tl.load(w1_ptr + square)
    hidden = tl.sum(w1 * k_in[None, :], axis=1)
    activated, slope, _ = _gelu_terms(hidden)
    w2 = tl.load(w2_ptr + square)
    output = tl.sum(w2 * activated[None, :], axis=1)
    value = tl.load(v_ptr + vector + rows)
    if LAYER_NORM:
        centered = output - tl.sum(output, axis=0) / D
        inverse = 1.0 / tl.sqrt(tl.sum(centered * centered, axis=0) / D + EPS)
        normalized = centered * inverse
        residual = tl.load(k_ptr + vector + rows) + normalized - value
        gradient = 2.0 * residual
        upstream = inverse * (
            gradient
            - tl.sum(gradient, axis=0) / D
            - normalized * (tl.sum(gradient * normalized, axis=0) / D)
        )
    else:
        residual = output - value
        upstream = 2.0 * residual
    tl.store(elements_out + vector + rows, residual * residual)

    momentum = eta * tl.load(s2_ptr + square) - theta * (upstream[:, None] * activated[None, :])
    tl.store(s2_out + square, momentum)
    tl.store(w2_out + square, keep[:, None] * w2 + momentum)
    delta = tl.sum(w2 * upstream[:, None], axis=0) * slope
    momentum = eta * tl.load(s1_ptr + square) - theta * (delta[:, None] * k_in[None, :])
    tl.store(s1_out + square, momentum)
    tl.store(w1_out + square, keep[:, None] * w1 + momentum)


@triton.jit
def _backward_kernel(
    w1_ptr,
    w2_ptr,
    s1_ptr,
    s2_ptr,
    k_ptr,
    v_ptr,
    alpha_ptr,
    eta_ptr,
    theta_ptr,
    dw1_next_ptr,
    dw2_next_ptr,
    ds1_next_ptr,
    ds2_next_ptr,
    delements_ptr,
    dw1_ptr,
    dw2_ptr,
    ds1_ptr,
    ds2_ptr,
    dk_ptr,
    dv_ptr,
    dalpha_ptr,
    deta_ptr,
    dtheta_ptr,
    EPS: tl.constexpr,
    D: tl.constexpr,
    LAYER_NORM: tl.constexpr,
):
    flow = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, D)
    cols = tl.arange(0, D)
    square = flow * D * D + rows[:, None] * D + cols[None, :]
    vector = flow * D
    k_in = tl.load(k_ptr + vector + cols)
    keep = 1.0 - tl.load(alpha_ptr + vector + rows)
    eta = tl.load(eta_ptr + flow)
    theta = tl.load(theta_ptr + flow)

    # Pasada directa del flujo, recalculada.
    w1 = tl.load(w1_ptr + square)
    hidden = tl.sum(w1 * k_in[None, :], axis=1)
    activated, slope, curvature = _gelu_terms(hidden)
    w2 = tl.load(w2_ptr + square)
    output = tl.sum(w2 * activated[None, :], axis=1)
    value = tl.load(v_ptr + vector + rows)
    if LAYER_NORM:
        centered = output - tl.sum(output, axis=0) / D
        inverse = 1.0 / tl.sqrt(tl.sum(centered * centered, axis=0) / D + EPS)
        normalized = centered * inverse
        residual = tl.load(k_ptr + vector + rows) + normalized - value
        gradient = 2.0 * residual
        mean_gradient = tl.sum(gradient * normalized, axis=0) / D
        upstream = inverse * (gradient - tl.sum(gradient, axis=0) / D - normalized * mean_gradient)
    else:
        residual = output - value
        gradient = 2.0 * residual
        upstream = gradient
    hidden_upstream = tl.sum(w2 * upstream[:, None], axis=0)
    delta = hidden_upstream * slope

    # Escritura de W₂ y S₂: P₂ = dS′₂ + dW′₂ y dG₂ = −θ P₂.
    dw2_next = tl.load(dw2_next_ptr + square)
    total = tl.load(ds2_next_ptr + square) + dw2_next
    dalpha = -tl.sum(w2 * dw2_next, axis=1)
    tl.store(ds2_ptr + square, eta * total)
    deta = tl.sum(tl.sum(tl.load(s2_ptr + square) * total, axis=1), axis=0)
    total_activated = tl.sum(total * activated[None, :], axis=1)
    dtheta = -tl.sum(upstream * total_activated, axis=0)
    d_upstream = -theta * total_activated
    d_activated = -theta * tl.sum(total * upstream[:, None], axis=0)
    dw2 = keep[:, None] * dw2_next

    # Escritura de W₁ y S₁: P₁ = dS′₁ + dW′₁ y dG₁ = −θ P₁.
    dw1_next = tl.load(dw1_next_ptr + square)
    total = tl.load(ds1_next_ptr + square) + dw1_next
    dalpha -= tl.sum(w1 * dw1_next, axis=1)
    tl.store(ds1_ptr + square, eta * total)
    deta += tl.sum(tl.sum(tl.load(s1_ptr + square) * total, axis=1), axis=0)
    total_key = tl.sum(total * k_in[None, :], axis=1)
    dtheta -= tl.sum(delta * total_key, axis=0)
    d_delta = -theta * total_key
    d_key = -theta * tl.sum(total * delta[:, None], axis=0)
    dw1 = keep[:, None] * dw1_next

    # δh = δa ⊙ GELU′(h) y δa = W₂ᵀ δz.
    d_hidden_upstream = d_delta * slope
    d_hidden = d_delta * hidden_upstream * curvature
    dw2 += upstream[:, None] * d_hidden_upstream[None, :]
    d_upstream += tl.sum(w2 * d_hidden_upstream[None, :], axis=1)

    # δz = r·(g − media(g) − n·media(g ⊙ n)), g = 2e y la pérdida e².
    if LAYER_NORM:
        mean_d = tl.sum(d_upstream, axis=0) / D
        mean_dn = tl.sum(d_upstream * normalized, axis=0) / D
        d_gradient = inverse * (d_upstream - mean_d - normalized * mean_dn)
        d_normalized = -inverse * (d_upstream * mean_gradient + gradient * mean_dn)
        d_inverse = tl.sum(d_upstream * upstream, axis=0) / inverse
    else:
        d_gradient = d_upstream
    d_residual = 2.0 * d_gradient + 2.0 * residual * tl.load(delements_ptr + vector + rows)
    tl.store(dv_ptr + vector + rows, -d_residual)
    if LAYER_NORM:
        # ŷ = k + n con n = c·r: la retropropagación de LN con el término extra de r.
        d_normalized += d_residual
        d_centered = d_normalized * inverse
        d_inverse += tl.sum(d_normalized * centered, axis=0)
        d_centered += (-d_inverse * inverse * inverse * inverse / D) * centered
        d_output = d_centered - tl.sum(d_centered, axis=0) / D
    else:
        d_output = d_residual

    # z = W₂a, a = GELU(h) y h = W₁k.
    dw2 += d_output[:, None] * activated[None, :]
    d_activated += tl.sum(w2 * d_output[:, None], axis=0)
    d_hidden += d_activated * slope
    dw1 += d_hidden[:, None] * k_in[None, :]
    d_key += tl.sum(w1 * d_hidden[:, None], axis=0)
    if LAYER_NORM:
        d_key += d_residual
    tl.store(dw1_ptr + square, dw1)
    tl.store(dw2_ptr + square, dw2)
    tl.store(dk_ptr + vector + cols, d_key)
    tl.store(dalpha_ptr + vector + rows, dalpha)
    tl.store(deta_ptr + flow, deta)
    tl.store(dtheta_ptr + flow, dtheta)


def _launch_forward(tensors, layer_norm, eps):
    w1, w2, s1, s2, keys, values, alpha, eta, theta = tensors
    outputs = (*(torch.empty_like(w1) for _ in range(4)), torch.empty_like(keys))
    _forward_kernel[(keys.shape[0],)](
        w1,
        w2,
        s1,
        s2,
        keys,
        values,
        alpha,
        eta,
        theta,
        *outputs,
        EPS=eps,
        D=DIMENSION,
        LAYER_NORM=layer_norm,
        num_warps=_FORWARD_WARPS,
    )
    return outputs


class _FusedWrite(torch.autograd.Function):
    """Paso de escritura con su VJP cerrado. Guarda solo las entradas del paso."""

    @staticmethod
    def forward(ctx, w1, w2, s1, s2, keys, values, alpha, eta, theta, layer_norm, eps):
        tensors = (w1, w2, s1, s2, keys, values, alpha, eta, theta)
        ctx.save_for_backward(*tensors)
        ctx.layer_norm, ctx.eps = layer_norm, eps
        return _launch_forward(tensors, layer_norm, eps)

    @staticmethod
    @once_differentiable
    def backward(ctx, dw1_next, dw2_next, ds1_next, ds2_next, delements):
        tensors = ctx.saved_tensors
        upstream = tuple(
            value.contiguous() for value in (dw1_next, dw2_next, ds1_next, ds2_next, delements)
        )
        gradients = tuple(torch.empty_like(value) for value in tensors)
        _backward_kernel[(tensors[4].shape[0],)](
            *tensors,
            *upstream,
            *gradients,
            EPS=ctx.eps,
            D=DIMENSION,
            LAYER_NORM=ctx.layer_norm,
            num_warps=_BACKWARD_WARPS,
        )
        return (*gradients, None, None)


def _checked(weights, momentum, keys, values, alpha, eta, theta):
    """Entradas contiguas del kernel con las formas, el tipo y el dispositivo exigidos."""
    if len(weights) != 2 or len(momentum) != 2:
        raise ValueError("La escritura fusionada cubre memorias de profundidad 2")
    rows = keys.shape[0] if keys.ndim == 2 else -1
    square = (rows, DIMENSION, DIMENSION)
    expected = (
        *((value, square) for value in (*weights, *momentum)),
        (keys, (rows, DIMENSION)),
        (values, (rows, DIMENSION)),
        (alpha, (rows, DIMENSION, 1)),
        (eta, (rows, 1, 1)),
        (theta, (rows, 1, 1)),
    )
    if rows < 1 or any(tuple(value.shape) != shape for value, shape in expected):
        raise ValueError(f"La escritura fusionada necesita flujos de dimensión {DIMENSION}")
    if (
        keys.dtype not in _DTYPES
        or keys.device.type != "cuda"
        or any(value.dtype != keys.dtype or value.device != keys.device for value, _ in expected)
    ):
        raise ValueError("La escritura fusionada necesita tensores CUDA float32 o float64")
    return (
        *(value.contiguous() for value in (*weights, *momentum, keys, values)),
        alpha.reshape(rows, DIMENSION).contiguous(),
        eta.reshape(rows).contiguous(),
        theta.reshape(rows).contiguous(),
    )


def fused_associative_write(
    weights: tuple[Tensor, Tensor],
    momentum: tuple[Tensor, Tensor],
    keys: Tensor,
    values: Tensor,
    gates: tuple[Tensor, Tensor, Tensor],
    *,
    residual_layer_norm: bool,
    eps: float,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor]:
    """Pesos y momentum tras el paso, y los residuos al cuadrado e² con forma [B, D].

    `gates` son (α, η, θ) con las formas de `NeuralMemory._gates`: [B, D, 1], [B, 1, 1] y
    [B, 1, 1]. Con grafo, el gradiente exterior llega a todas las entradas por el VJP cerrado.
    """
    alpha, eta, theta = gates
    tensors = _checked(weights, momentum, keys, values, alpha, eta, theta)
    if torch.is_grad_enabled() and any(value.requires_grad for value in tensors):
        w1, w2, s1, s2, elements = _FusedWrite.apply(*tensors, residual_layer_norm, eps)
    else:
        w1, w2, s1, s2, elements = _launch_forward(tensors, residual_layer_norm, eps)
    return (w1, w2), (s1, s2), elements
