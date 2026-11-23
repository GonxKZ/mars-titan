"""Kernels Triton fusionados de memoria asociativa y atención de último token de Titans.

Un paso de escritura de la memoria de profundidad 2, con la convención x @ Wᵀ:

    h = W₁k,  a = GELU(h),  z = W₂a
    n = LN(z) = (z − μ)·r,  r = (var(z) + ε)^(−1/2),  ŷ = k + n   (sin la capa: ŷ = z)
    e = ŷ − v,  g = 2e,  δz = r·(g − media(g) − n·media(g ⊙ n))   (sin la capa: δz = g)
    G₂ = δz aᵀ,  δa = W₂ᵀδz,  δh = δa ⊙ GELU′(h),  G₁ = δh kᵀ
    S′ᵢ = η Sᵢ − θ Gᵢ,  W′ᵢ = (1 − α) ⊙ Wᵢ + S′ᵢ    (α por fila de salida, η y θ por flujo)

Además de la escritura aislada (`fused_associative_write`), este módulo ofrece:

- `fused_memory_read`: lectura M(q; W₁, W₂) en un solo kernel por flujo (directa + VJP cerrado),
  sustituyendo dos `bmm` M=1 + GELU + LayerNorm + suma por una lectura directa de W₁ y W₂.
- `fused_associative_write_and_read`: fusiona la escritura con la lectura posterior M(o; W′₁, W′₂)
  y el máximo absoluto seguro ante NaN de (W′₁, W′₂, S′₁, S′₂) mientras las matrices ya están en
  registros; la inversa incorpora el VJP de la lectura posterior en registros (con ruta rápida de
  rango 1 cuando no llegan cotangentes del estado siguiente y omitiendo escrituras de gradientes
  innecesarios en el primer instante del tramo).
- `fused_mac_last_token_attention`: atención exacta IEEE de último token con prefijo persistente
  factorizado [P, 2D] en lugar de expandirlo por todo el lote.

Todos los kernels son deterministas (cero atómicas) y admiten FP32 IEEE estricto y FP64.
"""

import math

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable

from .gpu_profile import active_gpu_profile

DIMENSION = 64
_DTYPES = (torch.float32, torch.float64)


@triton.jit
def _gelu_terms(h):
    """GELU(h), GELU′(h) y GELU″(h) exactas, con Φ por la función de error."""
    cdf = 0.5 * (1.0 + tl.erf(h * 0.7071067811865476))
    pdf = tl.exp(-0.5 * h * h) * 0.3989422804014327
    return h * cdf, cdf + h * pdf, pdf * (2.0 - h * h)


@triton.jit
def _absmax_nan_safe(x):
    """Máximo absoluto que propaga NaN como +inf para indicadores de finitud exactos."""
    return tl.max(tl.where(x == x, tl.abs(x), float("inf")))


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
    post_q_ptr,
    w1_out,
    w2_out,
    s1_out,
    s2_out,
    elements_out,
    post_out,
    max_out,
    EPS: tl.constexpr,
    D: tl.constexpr,
    LAYER_NORM: tl.constexpr,
    FUSED_POST_READ: tl.constexpr,
    FUSED_ABSMAX: tl.constexpr,
    FUSED_ABSMAX_IN: tl.constexpr,
    EVICT_STREAM: tl.constexpr,
):
    flow = tl.program_id(0).to(tl.int64)
    n_flows = tl.num_programs(0).to(tl.int64)
    rows = tl.arange(0, D)
    cols = tl.arange(0, D)
    square = flow * (D * D) + rows[:, None] * D + cols[None, :]
    vector = flow * D
    k_in = tl.load(k_ptr + vector + cols)
    keep = 1.0 - tl.load(alpha_ptr + vector + rows)
    eta = tl.load(eta_ptr + flow)
    theta = tl.load(theta_ptr + flow)

    w1 = tl.load(w1_ptr + square)
    if FUSED_ABSMAX_IN:
        tl.store(max_out + flow + 4 * n_flows, _absmax_nan_safe(w1))
    hidden = tl.sum(w1 * k_in[None, :], axis=1)
    activated, slope, _ = _gelu_terms(hidden)
    w2 = tl.load(w2_ptr + square)
    if FUSED_ABSMAX_IN:
        tl.store(max_out + flow + 5 * n_flows, _absmax_nan_safe(w2))
    output = tl.sum(w2 * activated[None, :], axis=1)
    value = tl.load(v_ptr + vector + rows)
    if LAYER_NORM:
        centered = output - tl.sum(output, axis=0) / D
        denom = tl.sqrt(tl.sum(centered * centered, axis=0) / D + EPS)
        inverse = tl.div_rn(1.0, denom) if denom.dtype == tl.float32 else (1.0 / denom)
        normalized = centered * inverse
        residual = k_in + normalized - value
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

    delta = tl.sum(w2 * upstream[:, None], axis=0) * slope

    # Actualizar S₁ y W₁ primero para liberar W₁ antes de cargar S₂.
    if EVICT_STREAM:
        s1_old = tl.load(s1_ptr + square, eviction_policy="evict_first")
    else:
        s1_old = tl.load(s1_ptr + square)
    if FUSED_ABSMAX_IN:
        tl.store(max_out + flow + 6 * n_flows, _absmax_nan_safe(s1_old))
    s1_new = eta * s1_old - theta * (delta[:, None] * k_in[None, :])
    if EVICT_STREAM:
        tl.store(s1_out + square, s1_new, eviction_policy="evict_first")
    else:
        tl.store(s1_out + square, s1_new)
    w1_new = keep[:, None] * w1 + s1_new
    tl.store(w1_out + square, w1_new)
    if FUSED_ABSMAX:
        tl.store(max_out + flow, _absmax_nan_safe(w1_new))
        tl.store(max_out + flow + 2 * n_flows, _absmax_nan_safe(s1_new))
    if FUSED_POST_READ:
        post_q = tl.load(post_q_ptr + vector + cols)
        post_h = tl.sum(w1_new * post_q[None, :], axis=1)
        post_a, _, _ = _gelu_terms(post_h)

    if EVICT_STREAM:
        s2_old = tl.load(s2_ptr + square, eviction_policy="evict_first")
    else:
        s2_old = tl.load(s2_ptr + square)
    if FUSED_ABSMAX_IN:
        tl.store(max_out + flow + 7 * n_flows, _absmax_nan_safe(s2_old))
    s2_new = eta * s2_old - theta * (upstream[:, None] * activated[None, :])
    if EVICT_STREAM:
        tl.store(s2_out + square, s2_new, eviction_policy="evict_first")
    else:
        tl.store(s2_out + square, s2_new)
    w2_new = keep[:, None] * w2 + s2_new
    tl.store(w2_out + square, w2_new)
    if FUSED_ABSMAX:
        tl.store(max_out + flow + n_flows, _absmax_nan_safe(w2_new))
        tl.store(max_out + flow + 3 * n_flows, _absmax_nan_safe(s2_new))
    if FUSED_POST_READ:
        post_z = tl.sum(w2_new * post_a[None, :], axis=1)
        if LAYER_NORM:
            pc = post_z - tl.sum(post_z, axis=0) / D
            pden = tl.sqrt(tl.sum(pc * pc, axis=0) / D + EPS)
            pinv = tl.div_rn(1.0, pden) if pden.dtype == tl.float32 else (1.0 / pden)
            post_y = post_q + pc * pinv
        else:
            post_y = post_z
        tl.store(post_out + vector + rows, post_y)


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
    post_q_ptr,
    dw1_next_ptr,
    dw2_next_ptr,
    ds1_next_ptr,
    ds2_next_ptr,
    delements_ptr,
    dpost_ptr,
    dw1_ptr,
    dw2_ptr,
    ds1_ptr,
    ds2_ptr,
    dk_ptr,
    dv_ptr,
    dalpha_ptr,
    deta_ptr,
    dtheta_ptr,
    dpost_q_ptr,
    EPS: tl.constexpr,
    D: tl.constexpr,
    LAYER_NORM: tl.constexpr,
    FUSED_POST_READ: tl.constexpr,
    HAS_DW1_NEXT: tl.constexpr,
    HAS_DW2_NEXT: tl.constexpr,
    HAS_DS1_NEXT: tl.constexpr,
    HAS_DS2_NEXT: tl.constexpr,
    HAS_DELEMENTS: tl.constexpr,
    NEED_DW1: tl.constexpr,
    NEED_DW2: tl.constexpr,
    NEED_DS1: tl.constexpr,
    NEED_DS2: tl.constexpr,
):
    flow = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, D)
    cols = tl.arange(0, D)
    square = flow * (D * D) + rows[:, None] * D + cols[None, :]
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
        denom = tl.sqrt(tl.sum(centered * centered, axis=0) / D + EPS)
        inverse = tl.div_rn(1.0, denom) if denom.dtype == tl.float32 else (1.0 / denom)
        normalized = centered * inverse
        residual = k_in + normalized - value
        gradient = 2.0 * residual
        mean_gradient = tl.sum(gradient * normalized, axis=0) / D
        upstream = inverse * (gradient - tl.sum(gradient, axis=0) / D - normalized * mean_gradient)
    else:
        residual = output - value
        gradient = 2.0 * residual
        upstream = gradient
    hidden_upstream = tl.sum(w2 * upstream[:, None], axis=0)
    delta = hidden_upstream * slope

    # Ruta rápida de rango 1: cuando solo la lectura posterior aporta cotangentes al estado
    # siguiente (último paso del tramo BPTT), ninguna matriz dw_next/ds_next se lee de memoria.
    if (
        FUSED_POST_READ
        and (not HAS_DW1_NEXT)
        and (not HAS_DW2_NEXT)
        and (not HAS_DS1_NEXT)
        and (not HAS_DS2_NEXT)
    ):
        post_q = tl.load(post_q_ptr + vector + cols)
        dpost = tl.load(dpost_ptr + vector + rows)
        s1 = tl.load(s1_ptr + square, eviction_policy="evict_first")
        s1_pq = tl.sum(s1 * post_q[None, :], axis=1)
        w1_pq = tl.sum(w1 * post_q[None, :], axis=1)
        pq_dot_k = tl.sum(post_q * k_in, axis=0)
        post_h = keep * w1_pq + eta * s1_pq - (theta * pq_dot_k) * delta
        post_a, post_slope, _ = _gelu_terms(post_h)

        s2 = tl.load(s2_ptr + square, eviction_policy="evict_first")
        s2_pa = tl.sum(s2 * post_a[None, :], axis=1)
        w2_pa = tl.sum(w2 * post_a[None, :], axis=1)
        pa_dot_a = tl.sum(post_a * activated, axis=0)
        post_z = keep * w2_pa + eta * s2_pa - (theta * pa_dot_a) * upstream
        if LAYER_NORM:
            pc = post_z - tl.sum(post_z, axis=0) / D
            pden = tl.sqrt(tl.sum(pc * pc, axis=0) / D + EPS)
            pinv = tl.div_rn(1.0, pden) if pden.dtype == tl.float32 else (1.0 / pden)
            pn = pc * pinv
            post_dz = pinv * (
                dpost - tl.sum(dpost, axis=0) / D - pn * (tl.sum(dpost * pn, axis=0) / D)
            )
        else:
            post_dz = dpost

        kp_dz = keep * post_dz
        dz_dot_up = tl.sum(post_dz * upstream, axis=0)
        post_da = (
            tl.sum(w2 * kp_dz[:, None], axis=0)
            + eta * tl.sum(s2 * post_dz[:, None], axis=0)
            - (theta * dz_dot_up) * activated
        )
        post_dh = post_da * post_slope

        kp_dh = keep * post_dh
        dh_dot_del = tl.sum(post_dh * delta, axis=0)
        d_post_q = (
            tl.sum(w1 * kp_dh[:, None], axis=0)
            + eta * tl.sum(s1 * post_dh[:, None], axis=0)
            - (theta * dh_dot_del) * k_in
        )
        if LAYER_NORM:
            d_post_q = d_post_q + dpost
        tl.store(dpost_q_ptr + vector + cols, d_post_q)

        if NEED_DS2:
            tl.store(
                ds2_ptr + square,
                (eta * post_dz)[:, None] * post_a[None, :],
                eviction_policy="evict_first",
            )
        if NEED_DS1:
            tl.store(
                ds1_ptr + square,
                (eta * post_dh)[:, None] * post_q[None, :],
                eviction_policy="evict_first",
            )
        dalpha = -(post_dz * w2_pa) - (post_dh * w1_pq)
        deta = tl.sum(post_dz * s2_pa, axis=0) + tl.sum(post_dh * s1_pq, axis=0)
        dtheta = -(dz_dot_up * pa_dot_a) - (dh_dot_del * pq_dot_k)
        d_upstream = (-theta * pa_dot_a) * post_dz
        d_activated = (-theta * dz_dot_up) * post_a
        d_delta = (-theta * pq_dot_k) * post_dh
        d_key = (-theta * dh_dot_del) * post_q

        d_hidden_upstream = d_delta * slope
        d_hidden = d_delta * hidden_upstream * curvature
        d_upstream = d_upstream + tl.sum(w2 * d_hidden_upstream[None, :], axis=1)
        if LAYER_NORM:
            mean_d = tl.sum(d_upstream, axis=0) / D
            mean_dn = tl.sum(d_upstream * normalized, axis=0) / D
            d_gradient = inverse * (d_upstream - mean_d - normalized * mean_dn)
            d_normalized = -inverse * (d_upstream * mean_gradient + gradient * mean_dn)
            du_up = tl.sum(d_upstream * upstream, axis=0)
            d_inverse = (
                tl.div_rn(du_up, inverse) if inverse.dtype == tl.float32 else (du_up / inverse)
            )
        else:
            d_gradient = d_upstream
        if HAS_DELEMENTS:
            d_residual = 2.0 * d_gradient + 2.0 * residual * tl.load(delements_ptr + vector + rows)
        else:
            d_residual = 2.0 * d_gradient
        tl.store(dv_ptr + vector + rows, -d_residual)
        if LAYER_NORM:
            d_normalized = d_normalized + d_residual
            d_centered = d_normalized * inverse
            d_inverse = d_inverse + tl.sum(d_normalized * centered, axis=0)
            d_centered = d_centered + (-d_inverse * inverse * inverse * inverse / D) * centered
            d_output = d_centered - tl.sum(d_centered, axis=0) / D
        else:
            d_output = d_residual

        if NEED_DW2:
            dw2 = (
                kp_dz[:, None] * post_a[None, :]
                + upstream[:, None] * d_hidden_upstream[None, :]
                + d_output[:, None] * activated[None, :]
            )
            tl.store(dw2_ptr + square, dw2)
        d_activated = d_activated + tl.sum(w2 * d_output[:, None], axis=0)
        d_hidden = d_hidden + d_activated * slope
        if NEED_DW1:
            dw1 = kp_dh[:, None] * post_q[None, :] + d_hidden[:, None] * k_in[None, :]
            tl.store(dw1_ptr + square, dw1)
        d_key = d_key + tl.sum(w1 * d_hidden[:, None], axis=0)
        if LAYER_NORM:
            d_key = d_key + d_residual
        tl.store(dk_ptr + vector + cols, d_key)
        tl.store(dalpha_ptr + vector + rows, dalpha)
        tl.store(deta_ptr + flow, deta)
        tl.store(dtheta_ptr + flow, dtheta)
        return

    # Ruta general (con cotangentes de estado siguiente, con o sin lectura posterior fusionada).
    if FUSED_POST_READ:
        post_q = tl.load(post_q_ptr + vector + cols)
        dpost = tl.load(dpost_ptr + vector + rows)
        s1 = tl.load(s1_ptr + square, eviction_policy="evict_first")
        s1_pq = tl.sum(s1 * post_q[None, :], axis=1)
        w1_pq = tl.sum(w1 * post_q[None, :], axis=1)
        pq_dot_k = tl.sum(post_q * k_in, axis=0)
        post_h = keep * w1_pq + eta * s1_pq - (theta * pq_dot_k) * delta
        post_a, post_slope, _ = _gelu_terms(post_h)

        s2 = tl.load(s2_ptr + square, eviction_policy="evict_first")
        s2_pa = tl.sum(s2 * post_a[None, :], axis=1)
        w2_pa = tl.sum(w2 * post_a[None, :], axis=1)
        pa_dot_a = tl.sum(post_a * activated, axis=0)
        post_z = keep * w2_pa + eta * s2_pa - (theta * pa_dot_a) * upstream
        if LAYER_NORM:
            pc = post_z - tl.sum(post_z, axis=0) / D
            pden = tl.sqrt(tl.sum(pc * pc, axis=0) / D + EPS)
            pinv = tl.div_rn(1.0, pden) if pden.dtype == tl.float32 else (1.0 / pden)
            pn = pc * pinv
            post_dz = pinv * (
                dpost - tl.sum(dpost, axis=0) / D - pn * (tl.sum(dpost * pn, axis=0) / D)
            )
        else:
            post_dz = dpost
        kp_dz = keep * post_dz
        dz_dot_up = tl.sum(post_dz * upstream, axis=0)
        post_da = (
            tl.sum(w2 * kp_dz[:, None], axis=0)
            + eta * tl.sum(s2 * post_dz[:, None], axis=0)
            - (theta * dz_dot_up) * activated
        )
        post_dh = post_da * post_slope
        kp_dh = keep * post_dh
        dh_dot_del = tl.sum(post_dh * delta, axis=0)
        d_post_q = (
            tl.sum(w1 * kp_dh[:, None], axis=0)
            + eta * tl.sum(s1 * post_dh[:, None], axis=0)
            - (theta * dh_dot_del) * k_in
        )
        if LAYER_NORM:
            d_post_q = d_post_q + dpost
        tl.store(dpost_q_ptr + vector + cols, d_post_q)

    # Escritura de W₂ y S₂: P₂ = dS′₂ + dW′₂ y dG₂ = −θ P₂.
    if HAS_DW2_NEXT:
        dw2_eff = tl.load(dw2_next_ptr + square, eviction_policy="evict_first")
        if FUSED_POST_READ:
            dw2_eff = dw2_eff + post_dz[:, None] * post_a[None, :]
    else:
        if FUSED_POST_READ:
            dw2_eff = post_dz[:, None] * post_a[None, :]
        else:
            dw2_eff = tl.zeros((D, D), dtype=w2.dtype)
    if HAS_DS2_NEXT:
        total2 = tl.load(ds2_next_ptr + square, eviction_policy="evict_first") + dw2_eff
    else:
        total2 = dw2_eff
    dalpha = -tl.sum(w2 * dw2_eff, axis=1)
    if NEED_DS2:
        tl.store(ds2_ptr + square, eta * total2, eviction_policy="evict_first")
    s2_use = s2 if FUSED_POST_READ else tl.load(s2_ptr + square, eviction_policy="evict_first")
    deta = tl.sum(tl.sum(s2_use * total2, axis=1), axis=0)
    total_activated = tl.sum(total2 * activated[None, :], axis=1)
    dtheta = -tl.sum(upstream * total_activated, axis=0)
    d_upstream = -theta * total_activated
    d_activated = -theta * tl.sum(total2 * upstream[:, None], axis=0)
    if NEED_DW2:
        dw2 = keep[:, None] * dw2_eff

    # Escritura de W₁ y S₁: P₁ = dS′₁ + dW′₁ y dG₁ = −θ P₁.
    if HAS_DW1_NEXT:
        dw1_eff = tl.load(dw1_next_ptr + square, eviction_policy="evict_first")
        if FUSED_POST_READ:
            dw1_eff = dw1_eff + post_dh[:, None] * post_q[None, :]
    else:
        if FUSED_POST_READ:
            dw1_eff = post_dh[:, None] * post_q[None, :]
        else:
            dw1_eff = tl.zeros((D, D), dtype=w1.dtype)
    if HAS_DS1_NEXT:
        total1 = tl.load(ds1_next_ptr + square, eviction_policy="evict_first") + dw1_eff
    else:
        total1 = dw1_eff
    dalpha = dalpha - tl.sum(w1 * dw1_eff, axis=1)
    if NEED_DS1:
        tl.store(ds1_ptr + square, eta * total1, eviction_policy="evict_first")
    s1_use = s1 if FUSED_POST_READ else tl.load(s1_ptr + square, eviction_policy="evict_first")
    deta = deta + tl.sum(tl.sum(s1_use * total1, axis=1), axis=0)
    total_key = tl.sum(total1 * k_in[None, :], axis=1)
    dtheta = dtheta - tl.sum(delta * total_key, axis=0)
    d_delta = -theta * total_key
    d_key = -theta * tl.sum(total1 * delta[:, None], axis=0)
    if NEED_DW1:
        dw1 = keep[:, None] * dw1_eff

    # δh = δa ⊙ GELU′(h) y δa = W₂ᵀ δz.
    d_hidden_upstream = d_delta * slope
    d_hidden = d_delta * hidden_upstream * curvature
    if NEED_DW2:
        dw2 = dw2 + upstream[:, None] * d_hidden_upstream[None, :]
    d_upstream = d_upstream + tl.sum(w2 * d_hidden_upstream[None, :], axis=1)

    # δz = r·(g − media(g) − n·media(g ⊙ n)), g = 2e y la pérdida e².
    if LAYER_NORM:
        mean_d = tl.sum(d_upstream, axis=0) / D
        mean_dn = tl.sum(d_upstream * normalized, axis=0) / D
        d_gradient = inverse * (d_upstream - mean_d - normalized * mean_dn)
        d_normalized = -inverse * (d_upstream * mean_gradient + gradient * mean_dn)
        du_up = tl.sum(d_upstream * upstream, axis=0)
        d_inverse = (
            tl.div_rn(du_up, inverse) if inverse.dtype == tl.float32 else (du_up / inverse)
        )
    else:
        d_gradient = d_upstream
    if HAS_DELEMENTS:
        d_residual = 2.0 * d_gradient + 2.0 * residual * tl.load(delements_ptr + vector + rows)
    else:
        d_residual = 2.0 * d_gradient
    tl.store(dv_ptr + vector + rows, -d_residual)
    if LAYER_NORM:
        d_normalized = d_normalized + d_residual
        d_centered = d_normalized * inverse
        d_inverse = d_inverse + tl.sum(d_normalized * centered, axis=0)
        d_centered = d_centered + (-d_inverse * inverse * inverse * inverse / D) * centered
        d_output = d_centered - tl.sum(d_centered, axis=0) / D
    else:
        d_output = d_residual

    # z = W₂a, a = GELU(h) y h = W₁k.
    if NEED_DW2:
        dw2 = dw2 + d_output[:, None] * activated[None, :]
        tl.store(dw2_ptr + square, dw2)
    d_activated = d_activated + tl.sum(w2 * d_output[:, None], axis=0)
    d_hidden = d_hidden + d_activated * slope
    if NEED_DW1:
        dw1 = dw1 + d_hidden[:, None] * k_in[None, :]
        tl.store(dw1_ptr + square, dw1)
    d_key = d_key + tl.sum(w1 * d_hidden[:, None], axis=0)
    if LAYER_NORM:
        d_key = d_key + d_residual
    tl.store(dk_ptr + vector + cols, d_key)
    tl.store(dalpha_ptr + vector + rows, dalpha)
    tl.store(deta_ptr + flow, deta)
    tl.store(dtheta_ptr + flow, dtheta)


@triton.jit
def _read_fwd_kernel(
    w1_ptr,
    w2_ptr,
    q_ptr,
    out_ptr,
    EPS: tl.constexpr,
    D: tl.constexpr,
    LAYER_NORM: tl.constexpr,
):
    flow = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, D)
    cols = tl.arange(0, D)
    sq = flow * (D * D) + rows[:, None] * D + cols[None, :]
    vec = flow * D
    q = tl.load(q_ptr + vec + cols)
    w1 = tl.load(w1_ptr + sq)
    h = tl.sum(w1 * q[None, :], axis=1)
    a, _, _ = _gelu_terms(h)
    w2 = tl.load(w2_ptr + sq)
    z = tl.sum(w2 * a[None, :], axis=1)
    if LAYER_NORM:
        c = z - tl.sum(z, axis=0) / D
        den = tl.sqrt(tl.sum(c * c, axis=0) / D + EPS)
        inv = tl.div_rn(1.0, den) if den.dtype == tl.float32 else (1.0 / den)
        y = q + c * inv
    else:
        y = z
    tl.store(out_ptr + vec + rows, y)


@triton.jit
def _read_bwd_kernel(
    w1_ptr,
    w2_ptr,
    q_ptr,
    dy_ptr,
    dw1_ptr,
    dw2_ptr,
    dq_ptr,
    EPS: tl.constexpr,
    D: tl.constexpr,
    LAYER_NORM: tl.constexpr,
    NEED_DW1: tl.constexpr,
    NEED_DW2: tl.constexpr,
    NEED_DQ: tl.constexpr,
):
    flow = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, D)
    cols = tl.arange(0, D)
    sq = flow * (D * D) + rows[:, None] * D + cols[None, :]
    vec = flow * D
    q = tl.load(q_ptr + vec + cols)
    dy = tl.load(dy_ptr + vec + rows)
    w1 = tl.load(w1_ptr + sq)
    h = tl.sum(w1 * q[None, :], axis=1)
    a, slope, _ = _gelu_terms(h)
    w2 = tl.load(w2_ptr + sq)
    if LAYER_NORM:
        z = tl.sum(w2 * a[None, :], axis=1)
        c = z - tl.sum(z, axis=0) / D
        den = tl.sqrt(tl.sum(c * c, axis=0) / D + EPS)
        inv = tl.div_rn(1.0, den) if den.dtype == tl.float32 else (1.0 / den)
        n = c * inv
        dz = inv * (dy - tl.sum(dy, axis=0) / D - n * (tl.sum(dy * n, axis=0) / D))
    else:
        dz = dy
    if NEED_DW2:
        tl.store(dw2_ptr + sq, dz[:, None] * a[None, :])
    da = tl.sum(w2 * dz[:, None], axis=0)
    dh = da * slope
    if NEED_DW1:
        tl.store(dw1_ptr + sq, dh[:, None] * q[None, :])
    if NEED_DQ:
        dq = tl.sum(w1 * dh[:, None], axis=0)
        if LAYER_NORM:
            dq = dq + dy
        tl.store(dq_ptr + vec + cols, dq)


@triton.jit
def _mac_last_attn_fwd_kernel(
    q_seg_ptr,
    kv_pref_ptr,
    kv_ret_ptr,
    kv_seg_ptr,
    out_ptr,
    p_ptr,
    B,
    P: tl.constexpr,
    P_PAD: tl.constexpr,
    H: tl.constexpr,
    HD: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    d_off = tl.arange(0, HD)
    q = tl.load(q_seg_ptr + b * D + h * HD + d_off)

    p_off = tl.arange(0, P_PAD)
    p_mask = p_off < P
    kp = tl.load(
        kv_pref_ptr + p_off[:, None] * (2 * D) + h * HD + d_off[None, :],
        mask=p_mask[:, None],
        other=0.0,
    )
    vp = tl.load(
        kv_pref_ptr + p_off[:, None] * (2 * D) + D + h * HD + d_off[None, :],
        mask=p_mask[:, None],
        other=0.0,
    )
    sp = tl.where(p_mask, tl.sum(kp * q[None, :], axis=1) * SCALE, float("-inf"))

    kr = tl.load(kv_ret_ptr + b * (2 * D) + h * HD + d_off)
    vr = tl.load(kv_ret_ptr + b * (2 * D) + D + h * HD + d_off)
    ks = tl.load(kv_seg_ptr + b * (2 * D) + h * HD + d_off)
    vs = tl.load(kv_seg_ptr + b * (2 * D) + D + h * HD + d_off)

    sr = tl.sum(kr * q, axis=0) * SCALE
    ss = tl.sum(ks * q, axis=0) * SCALE

    m = tl.maximum(tl.max(sp, axis=0), tl.maximum(sr, ss))
    ep = tl.where(p_mask, tl.exp(sp - m), 0.0)
    er = tl.exp(sr - m)
    es = tl.exp(ss - m)
    denom = tl.sum(ep, axis=0) + er + es
    wp = tl.div_rn(ep, denom) if q.dtype == tl.float32 else ep / denom
    wr = tl.div_rn(er, denom) if q.dtype == tl.float32 else er / denom
    ws = tl.div_rn(es, denom) if q.dtype == tl.float32 else es / denom

    out = tl.sum(vp * wp[:, None], axis=0) + vr * wr + vs * ws
    tl.store(out_ptr + b * D + h * HD + d_off, out)

    p_base = (b * H + h) * (P + 2)
    tl.store(p_ptr + p_base + p_off, wp, mask=p_mask)
    tl.store(p_ptr + p_base + P, wr)
    tl.store(p_ptr + p_base + P + 1, ws)


@triton.jit
def _mac_last_attn_bwd_kernel(
    q_seg_ptr,
    kv_pref_ptr,
    kv_ret_ptr,
    kv_seg_ptr,
    p_ptr,
    dout_ptr,
    dq_seg_ptr,
    dkv_pref_part_ptr,
    dkv_ret_ptr,
    dkv_seg_ptr,
    B,
    P: tl.constexpr,
    P_PAD: tl.constexpr,
    H: tl.constexpr,
    HD: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    d_off = tl.arange(0, HD)
    p_off = tl.arange(0, P_PAD)
    p_mask = p_off < P

    kp = tl.load(
        kv_pref_ptr + p_off[:, None] * (2 * D) + h * HD + d_off[None, :],
        mask=p_mask[:, None],
        other=0.0,
    )
    vp = tl.load(
        kv_pref_ptr + p_off[:, None] * (2 * D) + D + h * HD + d_off[None, :],
        mask=p_mask[:, None],
        other=0.0,
    )

    acc_dkp = tl.zeros((P_PAD, HD), dtype=kp.dtype)
    acc_dvp = tl.zeros((P_PAD, HD), dtype=vp.dtype)

    b_start = pid * ROWS_PER_CTA
    b_end = tl.minimum(b_start + ROWS_PER_CTA, B)

    for b in range(b_start, b_end):
        q = tl.load(q_seg_ptr + b * D + h * HD + d_off)
        dout = tl.load(dout_ptr + b * D + h * HD + d_off)
        kr = tl.load(kv_ret_ptr + b * (2 * D) + h * HD + d_off)
        vr = tl.load(kv_ret_ptr + b * (2 * D) + D + h * HD + d_off)
        ks = tl.load(kv_seg_ptr + b * (2 * D) + h * HD + d_off)
        vs = tl.load(kv_seg_ptr + b * (2 * D) + D + h * HD + d_off)

        p_base = (b * H + h) * (P + 2)
        wp = tl.load(p_ptr + p_base + p_off, mask=p_mask, other=0.0)
        wr = tl.load(p_ptr + p_base + P)
        ws = tl.load(p_ptr + p_base + P + 1)

        acc_dvp = acc_dvp + wp[:, None] * dout[None, :]
        tl.store(dkv_ret_ptr + b * (2 * D) + D + h * HD + d_off, wr * dout)
        tl.store(dkv_seg_ptr + b * (2 * D) + D + h * HD + d_off, ws * dout)

        dwp = tl.sum(vp * dout[None, :], axis=1)
        dwr = tl.sum(vr * dout, axis=0)
        dws = tl.sum(vs * dout, axis=0)
        dot_w = tl.sum(wp * dwp, axis=0) + wr * dwr + ws * dws

        dsp = wp * (dwp - dot_w) * SCALE
        dsr = wr * (dwr - dot_w) * SCALE
        dss = ws * (dws - dot_w) * SCALE

        acc_dkp = acc_dkp + dsp[:, None] * q[None, :]
        tl.store(dkv_ret_ptr + b * (2 * D) + h * HD + d_off, dsr * q)
        tl.store(dkv_seg_ptr + b * (2 * D) + h * HD + d_off, dss * q)

        dq = tl.sum(kp * dsp[:, None], axis=0) + dsr * kr + dss * ks
        tl.store(dq_seg_ptr + b * D + h * HD + d_off, dq)

    part_base = pid * (P * 2 * D) + p_off[:, None] * (2 * D) + h * HD + d_off[None, :]
    tl.store(dkv_pref_part_ptr + part_base, acc_dkp, mask=p_mask[:, None])
    tl.store(dkv_pref_part_ptr + part_base + D, acc_dvp, mask=p_mask[:, None])


def _launch_forward(
    tensors,
    layer_norm,
    eps,
    *,
    post_query=None,
    return_absmax=False,
    return_input_absmax=False,
):
    w1, w2, s1, s2, keys, values, alpha, eta, theta = tensors
    rows = keys.shape[0]
    profile = active_gpu_profile(keys.device)
    w1_o, w2_o, s1_o, s2_o = (torch.empty_like(w1) for _ in range(4))
    elements_o = torch.empty_like(keys)
    fused_post = post_query is not None
    post_o = torch.empty_like(post_query) if fused_post else keys
    max_rows = 8 if return_input_absmax else (4 if return_absmax else 0)
    max_o = torch.empty((max_rows, rows), dtype=keys.dtype, device=keys.device) if max_rows else keys
    _forward_kernel[(rows,)](
        w1,
        w2,
        s1,
        s2,
        keys,
        values,
        alpha,
        eta,
        theta,
        post_query if fused_post else keys,
        w1_o,
        w2_o,
        s1_o,
        s2_o,
        elements_o,
        post_o,
        max_o,
        EPS=eps,
        D=DIMENSION,
        LAYER_NORM=layer_norm,
        FUSED_POST_READ=fused_post,
        FUSED_ABSMAX=return_absmax,
        FUSED_ABSMAX_IN=return_input_absmax,
        EVICT_STREAM=profile.mem_evict_stream,
        num_warps=profile.mem_fwd_warps,
        num_stages=1,
    )
    return (
        w1_o,
        w2_o,
        s1_o,
        s2_o,
        elements_o,
        post_o if fused_post else None,
        max_o if max_rows else None,
    )


class _FusedWrite(torch.autograd.Function):
    """Paso de escritura con su VJP cerrado. Guarda solo las entradas del paso."""

    @staticmethod
    def forward(ctx, w1, w2, s1, s2, keys, values, alpha, eta, theta, layer_norm, eps):
        ctx.set_materialize_grads(False)
        tensors = (w1, w2, s1, s2, keys, values, alpha, eta, theta)
        ctx.save_for_backward(*tensors)
        ctx.layer_norm, ctx.eps = layer_norm, eps
        w1_o, w2_o, s1_o, s2_o, elements_o, _, _ = _launch_forward(tensors, layer_norm, eps)
        return w1_o, w2_o, s1_o, s2_o, elements_o

    @staticmethod
    @once_differentiable
    def backward(ctx, dw1_next, dw2_next, ds1_next, ds2_next, delements):
        tensors = ctx.saved_tensors
        w1, w2, s1, s2, keys = tensors[:5]
        has_dw1, has_dw2 = dw1_next is not None, dw2_next is not None
        has_ds1, has_ds2 = ds1_next is not None, ds2_next is not None
        has_del = delements is not None
        if not (has_dw1 or has_dw2 or has_ds1 or has_ds2 or has_del):
            return (None,) * 11
        profile = active_gpu_profile(keys.device)
        need_dw1, need_dw2, need_ds1, need_ds2 = ctx.needs_input_grad[:4]
        dw1_in = dw1_next.contiguous() if has_dw1 else w1
        dw2_in = dw2_next.contiguous() if has_dw2 else w2
        ds1_in = ds1_next.contiguous() if has_ds1 else s1
        ds2_in = ds2_next.contiguous() if has_ds2 else s2
        del_in = delements.contiguous() if has_del else keys
        dw1 = torch.empty_like(w1) if need_dw1 else w1
        dw2 = torch.empty_like(w2) if need_dw2 else w2
        ds1 = torch.empty_like(s1) if need_ds1 else s1
        ds2 = torch.empty_like(s2) if need_ds2 else s2
        dk, dv, da = (torch.empty_like(v) for v in tensors[4:7])
        de, dt = torch.empty_like(tensors[7]), torch.empty_like(tensors[8])
        _backward_kernel[(keys.shape[0],)](
            *tensors,
            keys,
            dw1_in,
            dw2_in,
            ds1_in,
            ds2_in,
            del_in,
            keys,
            dw1,
            dw2,
            ds1,
            ds2,
            dk,
            dv,
            da,
            de,
            dt,
            keys,
            EPS=ctx.eps,
            D=DIMENSION,
            LAYER_NORM=ctx.layer_norm,
            FUSED_POST_READ=False,
            HAS_DW1_NEXT=has_dw1,
            HAS_DW2_NEXT=has_dw2,
            HAS_DS1_NEXT=has_ds1,
            HAS_DS2_NEXT=has_ds2,
            HAS_DELEMENTS=has_del,
            NEED_DW1=need_dw1,
            NEED_DW2=need_dw2,
            NEED_DS1=need_ds1,
            NEED_DS2=need_ds2,
            num_warps=profile.mem_bwd_warps,
            num_stages=1,
        )
        return (
            dw1 if need_dw1 else None,
            dw2 if need_dw2 else None,
            ds1 if need_ds1 else None,
            ds2 if need_ds2 else None,
            dk,
            dv,
            da,
            de,
            dt,
            None,
            None,
        )


class _FusedWriteAndRead(torch.autograd.Function):
    """Escritura fusionada con la lectura posterior M(post_query; W′) e indicadores absmax."""

    @staticmethod
    def forward(
        ctx,
        w1,
        w2,
        s1,
        s2,
        keys,
        values,
        alpha,
        eta,
        theta,
        post_query,
        layer_norm,
        eps,
        return_input_absmax,
    ):
        ctx.set_materialize_grads(False)
        tensors = (w1, w2, s1, s2, keys, values, alpha, eta, theta)
        ctx.save_for_backward(*tensors, post_query)
        ctx.layer_norm, ctx.eps = layer_norm, eps
        w1_o, w2_o, s1_o, s2_o, elements_o, post_o, max_o = _launch_forward(
            tensors,
            layer_norm,
            eps,
            post_query=post_query,
            return_absmax=True,
            return_input_absmax=return_input_absmax,
        )
        ctx.mark_non_differentiable(max_o)
        return w1_o, w2_o, s1_o, s2_o, elements_o, post_o, max_o

    @staticmethod
    @once_differentiable
    def backward(ctx, dw1_next, dw2_next, ds1_next, ds2_next, delements, dpost, _dmax):
        *tensors, post_query = ctx.saved_tensors
        w1, w2, s1, s2, keys = tensors[:5]
        has_dw1, has_dw2 = dw1_next is not None, dw2_next is not None
        has_ds1, has_ds2 = ds1_next is not None, ds2_next is not None
        has_del = delements is not None
        has_post = dpost is not None
        if not (has_dw1 or has_dw2 or has_ds1 or has_ds2 or has_del or has_post):
            return (None,) * 13
        profile = active_gpu_profile(keys.device)
        need_dw1, need_dw2, need_ds1, need_ds2 = ctx.needs_input_grad[:4]
        dw1_in = dw1_next.contiguous() if has_dw1 else w1
        dw2_in = dw2_next.contiguous() if has_dw2 else w2
        ds1_in = ds1_next.contiguous() if has_ds1 else s1
        ds2_in = ds2_next.contiguous() if has_ds2 else s2
        del_in = delements.contiguous() if has_del else keys
        dpost_in = dpost.contiguous() if has_post else post_query
        dw1 = torch.empty_like(w1) if need_dw1 else w1
        dw2 = torch.empty_like(w2) if need_dw2 else w2
        ds1 = torch.empty_like(s1) if need_ds1 else s1
        ds2 = torch.empty_like(s2) if need_ds2 else s2
        dk, dv, da = (torch.empty_like(v) for v in tensors[4:7])
        de, dt = torch.empty_like(tensors[7]), torch.empty_like(tensors[8])
        dpost_q = torch.empty_like(post_query) if has_post else post_query
        _backward_kernel[(keys.shape[0],)](
            *tensors,
            post_query,
            dw1_in,
            dw2_in,
            ds1_in,
            ds2_in,
            del_in,
            dpost_in,
            dw1,
            dw2,
            ds1,
            ds2,
            dk,
            dv,
            da,
            de,
            dt,
            dpost_q,
            EPS=ctx.eps,
            D=DIMENSION,
            LAYER_NORM=ctx.layer_norm,
            FUSED_POST_READ=has_post,
            HAS_DW1_NEXT=has_dw1,
            HAS_DW2_NEXT=has_dw2,
            HAS_DS1_NEXT=has_ds1,
            HAS_DS2_NEXT=has_ds2,
            HAS_DELEMENTS=has_del,
            NEED_DW1=need_dw1,
            NEED_DW2=need_dw2,
            NEED_DS1=need_ds1,
            NEED_DS2=need_ds2,
            num_warps=profile.mem_bwd_warps,
            num_stages=1,
        )
        return (
            dw1 if need_dw1 else None,
            dw2 if need_dw2 else None,
            ds1 if need_ds1 else None,
            ds2 if need_ds2 else None,
            dk,
            dv,
            da,
            de,
            dt,
            dpost_q if has_post else None,
            None,
            None,
            None,
        )


class _FusedRead(torch.autograd.Function):
    """Lectura M(q; W₁, W₂) de un token por flujo en un solo kernel Triton."""

    @staticmethod
    def forward(ctx, w1, w2, query, layer_norm, eps):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(w1, w2, query)
        ctx.layer_norm, ctx.eps = layer_norm, eps
        profile = active_gpu_profile(query.device)
        out = torch.empty_like(query)
        _read_fwd_kernel[(query.shape[0],)](
            w1,
            w2,
            query,
            out,
            EPS=eps,
            D=DIMENSION,
            LAYER_NORM=layer_norm,
            num_warps=profile.mem_read_warps,
            num_stages=1,
        )
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        if dy is None:
            return None, None, None, None, None
        w1, w2, query = ctx.saved_tensors
        dy = dy.contiguous()
        profile = active_gpu_profile(query.device)
        need_dw1, need_dw2, need_dq = ctx.needs_input_grad[:3]
        dw1 = torch.empty_like(w1) if need_dw1 else w1
        dw2 = torch.empty_like(w2) if need_dw2 else w2
        dq = torch.empty_like(query) if need_dq else query
        _read_bwd_kernel[(query.shape[0],)](
            w1,
            w2,
            query,
            dy,
            dw1,
            dw2,
            dq,
            EPS=ctx.eps,
            D=DIMENSION,
            LAYER_NORM=ctx.layer_norm,
            NEED_DW1=need_dw1,
            NEED_DW2=need_dw2,
            NEED_DQ=need_dq,
            num_warps=profile.mem_read_warps,
            num_stages=1,
        )
        return (
            dw1 if need_dw1 else None,
            dw2 if need_dw2 else None,
            dq if need_dq else None,
            None,
            None,
        )


class _FusedMACLastAttn(torch.autograd.Function):
    """Atención causal exacta del último token de MAC con prefijo persistente factorizado."""

    @staticmethod
    def forward(ctx, q_seg, kv_pref, kv_ret, kv_seg, heads):
        ctx.set_materialize_grads(False)
        rows, dim = q_seg.shape
        prefix = kv_pref.shape[0]
        head_dim = dim // heads
        p_pad = triton.next_power_of_2(max(prefix, 2))
        out = torch.empty_like(q_seg)
        probs = torch.empty((rows, heads, prefix + 2), dtype=q_seg.dtype, device=q_seg.device)
        scale = 1.0 / math.sqrt(head_dim)
        _mac_last_attn_fwd_kernel[(rows, heads)](
            q_seg,
            kv_pref,
            kv_ret,
            kv_seg,
            out,
            probs,
            rows,
            P=prefix,
            P_PAD=p_pad,
            H=heads,
            HD=head_dim,
            D=dim,
            SCALE=scale,
            num_warps=1,
            num_stages=1,
        )
        ctx.save_for_backward(q_seg, kv_pref, kv_ret, kv_seg, probs)
        ctx.heads = heads
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        if dout is None:
            return None, None, None, None, None
        q_seg, kv_pref, kv_ret, kv_seg, probs = ctx.saved_tensors
        dout = dout.contiguous()
        rows, dim = q_seg.shape
        prefix = kv_pref.shape[0]
        heads = ctx.heads
        head_dim = dim // heads
        p_pad = triton.next_power_of_2(max(prefix, 2))
        scale = 1.0 / math.sqrt(head_dim)
        profile = active_gpu_profile(q_seg.device)
        rows_per_cta, num_ctas = profile.wave_aligned_ctas(rows, profile.mac_bwd_ctas)
        dq_seg = torch.empty_like(q_seg)
        dkv_ret = torch.empty_like(kv_ret)
        dkv_seg = torch.empty_like(kv_seg)
        dkv_pref_part = torch.empty(
            (num_ctas, prefix, 2 * dim), dtype=q_seg.dtype, device=q_seg.device
        )
        _mac_last_attn_bwd_kernel[(num_ctas, heads)](
            q_seg,
            kv_pref,
            kv_ret,
            kv_seg,
            probs,
            dout,
            dq_seg,
            dkv_pref_part,
            dkv_ret,
            dkv_seg,
            rows,
            P=prefix,
            P_PAD=p_pad,
            H=heads,
            HD=head_dim,
            D=dim,
            SCALE=scale,
            ROWS_PER_CTA=rows_per_cta,
            num_warps=1,
            num_stages=1,
        )
        return dq_seg, dkv_pref_part.sum(0), dkv_ret, dkv_seg, None


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
    """Pesos y momentum tras el paso, y los residuos al cuadrado e² con forma [B, D]."""
    alpha, eta, theta = gates
    tensors = _checked(weights, momentum, keys, values, alpha, eta, theta)
    if torch.is_grad_enabled() and any(value.requires_grad for value in tensors):
        w1, w2, s1, s2, elements = _FusedWrite.apply(*tensors, residual_layer_norm, eps)
    else:
        w1, w2, s1, s2, elements, _, _ = _launch_forward(tensors, residual_layer_norm, eps)
    return (w1, w2), (s1, s2), elements


def fused_associative_write_and_read(
    weights: tuple[Tensor, Tensor],
    momentum: tuple[Tensor, Tensor],
    keys: Tensor,
    values: Tensor,
    gates: tuple[Tensor, Tensor, Tensor],
    post_query: Tensor,
    *,
    residual_layer_norm: bool,
    eps: float,
    return_input_absmax: bool = False,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor, Tensor, Tensor]:
    """Escritura + lectura posterior M(post_query; W′) + absmax [4|8, B] en un único lanzamiento."""
    alpha, eta, theta = gates
    tensors = _checked(weights, momentum, keys, values, alpha, eta, theta)
    rows = keys.shape[0]
    if (
        post_query.shape != (rows, DIMENSION)
        or post_query.dtype != keys.dtype
        or post_query.device != keys.device
    ):
        raise ValueError("La consulta posterior necesita [B, 64] en la misma precisión y dispositivo")
    pq = post_query.contiguous()
    if torch.is_grad_enabled() and (
        pq.requires_grad or any(value.requires_grad for value in tensors)
    ):
        w1, w2, s1, s2, elements, read_post, absmax = _FusedWriteAndRead.apply(
            *tensors, pq, residual_layer_norm, eps, return_input_absmax
        )
    else:
        w1, w2, s1, s2, elements, read_post, absmax = _launch_forward(
            tensors,
            residual_layer_norm,
            eps,
            post_query=pq,
            return_absmax=True,
            return_input_absmax=return_input_absmax,
        )
    return (w1, w2), (s1, s2), elements, read_post, absmax


def fused_memory_read(
    weights: tuple[Tensor, Tensor],
    query: Tensor,
    *,
    residual_layer_norm: bool,
    eps: float,
) -> Tensor:
    """Lectura fusionada M(query; W₁, W₂) para query [B, 1, 64] y profundidad 2 en CUDA."""
    w1, w2 = weights[0].contiguous(), weights[1].contiguous()
    q = query.squeeze(1).contiguous()
    if torch.is_grad_enabled() and (w1.requires_grad or w2.requires_grad or q.requires_grad):
        out = _FusedRead.apply(w1, w2, q, residual_layer_norm, eps)
    else:
        profile = active_gpu_profile(q.device)
        out = torch.empty_like(q)
        _read_fwd_kernel[(q.shape[0],)](
            w1,
            w2,
            q,
            out,
            EPS=eps,
            D=DIMENSION,
            LAYER_NORM=residual_layer_norm,
            num_warps=profile.mem_read_warps,
            num_stages=1,
        )
    return out.unsqueeze(1)


def fused_mac_last_token_attention(
    q_seg: Tensor,
    kv_prefix: Tensor,
    kv_ret: Tensor,
    kv_seg: Tensor,
    heads: int,
) -> Tensor:
    """Atención exacta del último token [B, D] con prefijo persistente [P, 2D] factorizado."""
    q_seg = q_seg.contiguous()
    kv_prefix = kv_prefix.contiguous()
    kv_ret = kv_ret.contiguous()
    kv_seg = kv_seg.contiguous()
    if torch.is_grad_enabled() and any(
        value.requires_grad for value in (q_seg, kv_prefix, kv_ret, kv_seg)
    ):
        return _FusedMACLastAttn.apply(q_seg, kv_prefix, kv_ret, kv_seg, heads)
    rows, dim = q_seg.shape
    prefix = kv_prefix.shape[0]
    head_dim = dim // heads
    p_pad = triton.next_power_of_2(max(prefix, 2))
    out = torch.empty_like(q_seg)
    probs = torch.empty((rows, heads, prefix + 2), dtype=q_seg.dtype, device=q_seg.device)
    _mac_last_attn_fwd_kernel[(rows, heads)](
        q_seg,
        kv_prefix,
        kv_ret,
        kv_seg,
        out,
        probs,
        rows,
        P=prefix,
        P_PAD=p_pad,
        H=heads,
        HD=head_dim,
        D=dim,
        SCALE=1.0 / math.sqrt(head_dim),
        num_warps=1,
        num_stages=1,
    )
    return out
