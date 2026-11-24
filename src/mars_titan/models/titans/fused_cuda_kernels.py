"""Kernels Triton FP32 IEEE / FP64 deterministas para la ruta caliente de MARS-TITAN en CUDA.

Contiene seis piezas independientes, todas sin operaciones atómicas (determinismo estricto), sin
precisión reducida ni TF32 (`input_precision="ieee"` en todo producto matricial), y con paridad
verificada frente a las referencias de PyTorch en FP64 y FP32:

1. `fused_price_proj_pos_ln`: proyección lineal de precios + posición sinusoidal + LayerNorm
   (y compuerta opcional de presencia) en una pasada; inversa determinista con reducción parcial.
2. `fused_last_token_sdpa`: atención exacta de último token leyendo directamente `kv` [B, T, 2D]
   sin transponer, copiar ni escalar K en memoria global.
3. `fused_flash_cross_asset`: FlashAttention-2 exacta IEEE para atención entre activos (C5),
   con memoria O(N) e inversa determinista en dos kernels (dQ y dK/dV) sin atómicas.
4. `fused_price_returns_and_scale`: rendimientos por sesión + escalado robusto + recorte +
   máscara de presencia de `LearningScaler` en un solo lanzamiento.
5. `fused_tie_rank_probabilities`: probabilidades de rangos con empate promediado tras `sort`
   usando división IEEE exacta (`tl.div_rn` en FP32), idéntico bit a bit a `searchsorted`.
6. `fused_stable_topk_desc`: selección estable exacta top-k descendente para `EpisodicReadout`,
   idéntica bit a bit a `argsort(..., descending=True, stable=True)[:, :k]`.
"""

import math

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable

from .gpu_profile import active_gpu_profile


# ===========================================================================
# 1. FUSED PRICE PROJECTION + POSITIONS + LAYERNORM
# ===========================================================================
@triton.jit
def _price_proj_pos_ln_fwd_kernel(
    prices_ptr,
    w_ptr,
    b_ptr,
    pos_ptr,
    ln_w_ptr,
    ln_b_ptr,
    norm_out_ptr,
    res_last_ptr,
    mean_out_ptr,
    rstd_out_ptr,
    stride_pb,
    stride_pt,
    stride_pc,
    T: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    GATE_LAST: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    tb = tl.program_id(1).to(tl.int64)
    t_off = tb * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.arange(0, D)

    b_val = tl.load(b_ptr + d_off)
    ln_w = tl.load(ln_w_ptr + d_off)
    ln_b = tl.load(ln_b_ptr + d_off)

    pos = tl.load(pos_ptr + t_off[:, None] * D + d_off[None, :])
    acc = pos + b_val[None, :]

    if GATE_LAST:
        pres = tl.load(prices_ptr + b * stride_pb + t_off * stride_pt + (C - 1) * stride_pc)
        mask = pres > 0.0
        for c in tl.static_range(C - 1):
            p = tl.load(prices_ptr + b * stride_pb + t_off * stride_pt + c * stride_pc)
            p = tl.where(mask, p, 0.0)
            wc = tl.load(w_ptr + d_off * C + c)
            acc = acc + p[:, None] * wc[None, :]
        wc_last = tl.load(w_ptr + d_off * C + (C - 1))
        acc = acc + pres[:, None] * wc_last[None, :]
    else:
        for c in tl.static_range(C):
            p = tl.load(prices_ptr + b * stride_pb + t_off * stride_pt + c * stride_pc)
            wc = tl.load(w_ptr + d_off * C + c)
            acc = acc + p[:, None] * wc[None, :]

    if (tb + 1) * BLOCK_T == T:
        is_last = (t_off == (T - 1))[:, None]
        res_ptrs = res_last_ptr + b * D + tl.zeros((BLOCK_T, 1), dtype=tl.int64) + d_off[None, :]
        tl.store(res_ptrs, acc, mask=is_last)

    mean = tl.sum(acc, axis=1) / D
    centered = acc - mean[:, None]
    var = tl.sum(centered * centered, axis=1) / D
    rstd = 1.0 / tl.sqrt(var + EPS)
    normed = centered * rstd[:, None] * ln_w[None, :] + ln_b[None, :]

    out_off = (b * T + t_off[:, None]) * D + d_off[None, :]
    tl.store(norm_out_ptr + out_off, normed)
    tl.store(mean_out_ptr + b * T + t_off, mean)
    tl.store(rstd_out_ptr + b * T + t_off, rstd)


@triton.jit
def _price_proj_pos_ln_bwd_kernel(
    prices_ptr,
    w_ptr,
    b_ptr,
    pos_ptr,
    ln_w_ptr,
    mean_ptr,
    rstd_ptr,
    dnorm_ptr,
    dres_last_ptr,
    dp_part_w_ptr,
    dp_part_b_ptr,
    dp_part_lnw_ptr,
    dp_part_lnb_ptr,
    dprices_ptr,
    stride_pb,
    stride_pt,
    stride_pc,
    B,
    T: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    GATE_LAST: tl.constexpr,
    NEED_DPRICES: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    d_off = tl.arange(0, D)
    b_val = tl.load(b_ptr + d_off)
    ln_w = tl.load(ln_w_ptr + d_off)

    acc_db = tl.zeros((D,), dtype=b_val.dtype)
    acc_dlnw = tl.zeros((D,), dtype=b_val.dtype)
    acc_dlnb = tl.zeros((D,), dtype=b_val.dtype)
    c_off = tl.arange(0, 8)
    c_mask = c_off < C
    acc_dw = tl.zeros((D, 8), dtype=b_val.dtype)

    b_start = pid * ROWS_PER_CTA
    b_end = tl.minimum(b_start + ROWS_PER_CTA, B)

    for b in range(b_start, b_end):
        dres_last = tl.load(dres_last_ptr + b * D + d_off)
        for tb in range(0, T, BLOCK_T):
            t_off = tb + tl.arange(0, BLOCK_T)
            pos = tl.load(pos_ptr + t_off[:, None] * D + d_off[None, :])
            x = pos + b_val[None, :]
            if GATE_LAST:
                pres = tl.load(
                    prices_ptr + b * stride_pb + t_off * stride_pt + (C - 1) * stride_pc
                )
                pmask = pres > 0.0
            for c in tl.static_range(C):
                pc = tl.load(prices_ptr + b * stride_pb + t_off * stride_pt + c * stride_pc)
                if GATE_LAST and c < C - 1:
                    pc = tl.where(pmask, pc, 0.0)
                wc = tl.load(w_ptr + d_off * C + c)
                x = x + pc[:, None] * wc[None, :]

            mean = tl.load(mean_ptr + b * T + t_off)
            rstd = tl.load(rstd_ptr + b * T + t_off)
            n = (x - mean[:, None]) * rstd[:, None]

            dn = tl.load(dnorm_ptr + (b * T + t_off[:, None]) * D + d_off[None, :])
            acc_dlnb = acc_dlnb + tl.sum(dn, axis=0)
            acc_dlnw = acc_dlnw + tl.sum(dn * n, axis=0)

            dy = dn * ln_w[None, :]
            m1 = tl.sum(dy, axis=1) / D
            m2 = tl.sum(dy * n, axis=1) / D
            dx = rstd[:, None] * (dy - m1[:, None] - n * m2[:, None])

            if tb + BLOCK_T == T:
                is_last = (t_off == (T - 1))[:, None]
                dx = tl.where(is_last, dx + dres_last[None, :], dx)

            acc_db = acc_db + tl.sum(dx, axis=0)
            for c in tl.static_range(C):
                pc = tl.load(prices_ptr + b * stride_pb + t_off * stride_pt + c * stride_pc)
                if GATE_LAST and c < C - 1:
                    pc = tl.where(pmask, pc, 0.0)
                dwc = tl.sum(dx * pc[:, None], axis=0)
                acc_dw = tl.where((c_off == c)[None, :], acc_dw + dwc[:, None], acc_dw)

            if NEED_DPRICES:
                for c in tl.static_range(C):
                    wc = tl.load(w_ptr + d_off * C + c)
                    dpc = tl.sum(dx * wc[None, :], axis=1)
                    if GATE_LAST and c < C - 1:
                        dpc = tl.where(pmask, dpc, 0.0)
                    tl.store(
                        dprices_ptr + b * stride_pb + t_off * stride_pt + c * stride_pc, dpc
                    )

    tl.store(dp_part_b_ptr + pid * D + d_off, acc_db)
    tl.store(dp_part_lnw_ptr + pid * D + d_off, acc_dlnw)
    tl.store(dp_part_lnb_ptr + pid * D + d_off, acc_dlnb)
    tl.store(
        dp_part_w_ptr + pid * (D * C) + d_off[:, None] * C + c_off[None, :],
        acc_dw,
        mask=c_mask[None, :],
    )


class _FusedPriceProjPosLN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prices, weight, bias, positions, ln_w, ln_b, gate_last, eps):
        rows, steps, channels = prices.shape
        dim = weight.shape[0]
        block_t = 16 if steps % 16 == 0 else steps
        norm_out = torch.empty((rows, steps, dim), dtype=prices.dtype, device=prices.device)
        res_last = torch.empty((rows, 1, dim), dtype=prices.dtype, device=prices.device)
        mean_out = torch.empty((rows, steps), dtype=prices.dtype, device=prices.device)
        rstd_out = torch.empty((rows, steps), dtype=prices.dtype, device=prices.device)
        _price_proj_pos_ln_fwd_kernel[(rows, steps // block_t)](
            prices,
            weight,
            bias,
            positions,
            ln_w,
            ln_b,
            norm_out,
            res_last,
            mean_out,
            rstd_out,
            prices.stride(0),
            prices.stride(1),
            prices.stride(2),
            T=steps,
            C=channels,
            D=dim,
            GATE_LAST=gate_last,
            EPS=eps,
            BLOCK_T=block_t,
            num_warps=4,
            num_stages=1,
        )
        ctx.save_for_backward(prices, weight, bias, positions, ln_w, mean_out, rstd_out)
        ctx.gate_last, ctx.block_t = gate_last, block_t
        ctx.set_materialize_grads(False)
        return norm_out, res_last

    @staticmethod
    @once_differentiable
    def backward(ctx, dnorm, dres_last):
        if dnorm is None and dres_last is None:
            return (None,) * 8
        prices, weight, bias, positions, ln_w, mean_out, rstd_out = ctx.saved_tensors
        rows, steps, channels = prices.shape
        dim = weight.shape[0]
        dnorm = (
            dnorm.contiguous()
            if dnorm is not None
            else torch.zeros((rows, steps, dim), dtype=prices.dtype, device=prices.device)
        )
        if dres_last is None:
            dres_last = torch.zeros((rows, 1, dim), dtype=prices.dtype, device=prices.device)
        else:
            dres_last = dres_last.contiguous()
        need_dp = ctx.needs_input_grad[0]
        dprices = torch.empty_like(prices) if need_dp else prices
        profile = active_gpu_profile(prices.device)
        rows_per_cta, num_ctas = profile.wave_aligned_ctas(rows, profile.fac_bwd_ctas)
        pw = torch.empty((num_ctas, dim, channels), dtype=prices.dtype, device=prices.device)
        pb = torch.empty((num_ctas, dim), dtype=prices.dtype, device=prices.device)
        plnw = torch.empty((num_ctas, dim), dtype=prices.dtype, device=prices.device)
        plnb = torch.empty((num_ctas, dim), dtype=prices.dtype, device=prices.device)
        _price_proj_pos_ln_bwd_kernel[(num_ctas,)](
            prices,
            weight,
            bias,
            positions,
            ln_w,
            mean_out,
            rstd_out,
            dnorm,
            dres_last,
            pw,
            pb,
            plnw,
            plnb,
            dprices,
            prices.stride(0),
            prices.stride(1),
            prices.stride(2),
            rows,
            T=steps,
            C=channels,
            D=dim,
            GATE_LAST=ctx.gate_last,
            NEED_DPRICES=need_dp,
            ROWS_PER_CTA=rows_per_cta,
            BLOCK_T=ctx.block_t,
            num_warps=4,
            num_stages=1,
        )
        return (
            dprices if need_dp else None,
            pw.sum(0) if ctx.needs_input_grad[1] else None,
            pb.sum(0) if ctx.needs_input_grad[2] else None,
            None,
            plnw.sum(0) if ctx.needs_input_grad[4] else None,
            plnb.sum(0) if ctx.needs_input_grad[5] else None,
            None,
            None,
        )


def fused_price_proj_pos_ln(
    prices: Tensor,
    weight: Tensor,
    bias: Tensor,
    positions: Tensor,
    ln_weight: Tensor,
    ln_bias: Tensor,
    *,
    gate_last: bool,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Proyección de precios + posición + LayerNorm (y residuo del último token) en un kernel."""
    weight = weight.contiguous()
    bias = bias.contiguous()
    positions = positions.contiguous()
    ln_weight = ln_weight.contiguous()
    ln_bias = ln_bias.contiguous()
    if torch.is_grad_enabled() and any(
        value.requires_grad for value in (prices, weight, bias, ln_weight, ln_bias)
    ):
        return _FusedPriceProjPosLN.apply(
            prices, weight, bias, positions, ln_weight, ln_bias, gate_last, eps
        )
    rows, steps, channels = prices.shape
    dim = weight.shape[0]
    block_t = 16 if steps % 16 == 0 else steps
    norm_out = torch.empty((rows, steps, dim), dtype=prices.dtype, device=prices.device)
    res_last = torch.empty((rows, 1, dim), dtype=prices.dtype, device=prices.device)
    mean_out = torch.empty((rows, steps), dtype=prices.dtype, device=prices.device)
    rstd_out = torch.empty((rows, steps), dtype=prices.dtype, device=prices.device)
    _price_proj_pos_ln_fwd_kernel[(rows, steps // block_t)](
        prices,
        weight,
        bias,
        positions,
        ln_weight,
        ln_bias,
        norm_out,
        res_last,
        mean_out,
        rstd_out,
        prices.stride(0),
        prices.stride(1),
        prices.stride(2),
        T=steps,
        C=channels,
        D=dim,
        GATE_LAST=gate_last,
        EPS=eps,
        BLOCK_T=block_t,
        num_warps=4,
        num_stages=1,
    )
    return norm_out, res_last


# ===========================================================================
# 2. FUSED LAST-TOKEN SDPA READING DIRECTLY FROM KV [B, T, 2*D]
# ===========================================================================
@triton.jit
def _last_token_sdpa_fwd_kernel(
    q_ptr,
    kv_ptr,
    out_ptr,
    p_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    HD: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    t_off = tl.arange(0, T)
    d_off = tl.arange(0, HD)

    q = tl.load(q_ptr + b * D + h * HD + d_off)
    kv_base = b * (T * 2 * D) + t_off[:, None] * (2 * D) + h * HD + d_off[None, :]
    k = tl.load(kv_ptr + kv_base)
    v = tl.load(kv_ptr + kv_base + D)

    scores = tl.sum(k * q[None, :], axis=1) * SCALE
    m = tl.max(scores, axis=0)
    e = tl.exp(scores - m)
    denom = tl.sum(e, axis=0)
    p = tl.div_rn(e, denom) if q.dtype == tl.float32 else e / denom
    out = tl.sum(v * p[:, None], axis=0)

    tl.store(out_ptr + b * D + h * HD + d_off, out)
    tl.store(p_ptr + (b * H + h) * T + t_off, p)


@triton.jit
def _last_token_sdpa_bwd_kernel(
    q_ptr,
    kv_ptr,
    p_ptr,
    dout_ptr,
    dq_ptr,
    dkv_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    HD: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    t_off = tl.arange(0, T)
    d_off = tl.arange(0, HD)

    q = tl.load(q_ptr + b * D + h * HD + d_off)
    dout = tl.load(dout_ptr + b * D + h * HD + d_off)
    p = tl.load(p_ptr + (b * H + h) * T + t_off)
    kv_base = b * (T * 2 * D) + t_off[:, None] * (2 * D) + h * HD + d_off[None, :]
    k = tl.load(kv_ptr + kv_base)
    v = tl.load(kv_ptr + kv_base + D)

    tl.store(dkv_ptr + kv_base + D, p[:, None] * dout[None, :])
    dp = tl.sum(v * dout[None, :], axis=1)
    dscores = p * (dp - tl.sum(p * dp, axis=0)) * SCALE
    tl.store(dkv_ptr + kv_base, dscores[:, None] * q[None, :])
    dq = tl.sum(k * dscores[:, None], axis=0)
    tl.store(dq_ptr + b * D + h * HD + d_off, dq)


class _FusedLastTokenSDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, heads):
        rows, steps, two_d = kv.shape
        dim = two_d // 2
        head_dim = dim // heads
        out = torch.empty((rows, 1, dim), dtype=q.dtype, device=q.device)
        probs = torch.empty((rows, heads, steps), dtype=q.dtype, device=q.device)
        _last_token_sdpa_fwd_kernel[(rows, heads)](
            q,
            kv,
            out,
            probs,
            T=steps,
            H=heads,
            HD=head_dim,
            D=dim,
            SCALE=1.0 / math.sqrt(head_dim),
            num_warps=2,
            num_stages=1,
        )
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(q, kv, probs)
        ctx.heads = heads
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        if dout is None:
            return None, None, None
        q, kv, probs = ctx.saved_tensors
        rows, steps, two_d = kv.shape
        dim = two_d // 2
        heads = ctx.heads
        head_dim = dim // heads
        dout = dout.contiguous()
        dq = torch.empty_like(q)
        dkv = torch.empty_like(kv)
        _last_token_sdpa_bwd_kernel[(rows, heads)](
            q,
            kv,
            probs,
            dout,
            dq,
            dkv,
            T=steps,
            H=heads,
            HD=head_dim,
            D=dim,
            SCALE=1.0 / math.sqrt(head_dim),
            num_warps=2,
            num_stages=1,
        )
        return dq, dkv, None


def fused_last_token_sdpa(q: Tensor, kv: Tensor, heads: int) -> Tensor:
    """Atención exacta del último token leyendo `kv` [B, T, 2D] directamente sin copiar."""
    q, kv = q.contiguous(), kv.contiguous()
    if torch.is_grad_enabled() and (q.requires_grad or kv.requires_grad):
        return _FusedLastTokenSDPA.apply(q, kv, heads)
    rows, steps, two_d = kv.shape
    dim = two_d // 2
    head_dim = dim // heads
    out = torch.empty((rows, 1, dim), dtype=q.dtype, device=q.device)
    probs = torch.empty((rows, heads, steps), dtype=q.dtype, device=q.device)
    _last_token_sdpa_fwd_kernel[(rows, heads)](
        q,
        kv,
        out,
        probs,
        T=steps,
        H=heads,
        HD=head_dim,
        D=dim,
        SCALE=1.0 / math.sqrt(head_dim),
        num_warps=2,
        num_stages=1,
    )
    return out


# ===========================================================================
# 3. EXACT IEEE FP32/FP64 FLASH ATTENTION-2 FOR CROSS-ASSET ATTENTION (C5)
# ===========================================================================
@triton.jit
def _flash_c5_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
    N,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oh,
    stride_on,
    stride_od,
    stride_lh,
    stride_ln,
    SCALE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N

    q = tl.load(
        q_ptr + pid_h * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None],
        other=0.0,
    )
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=q.dtype)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=q.dtype)
    l_i = tl.zeros((BLOCK_M,), dtype=q.dtype)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        k = tl.load(
            k_ptr + pid_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptr + pid_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
        qk = tl.where(mask_n[None, :], qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        m_i = m_ij

    out = tl.div_rn(acc, l_i[:, None]) if q.dtype == tl.float32 else acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    tl.store(
        out_ptr + pid_h * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od,
        out,
        mask=mask_m[:, None],
        eviction_policy="evict_first",
    )
    tl.store(
        lse_ptr + pid_h * stride_lh + offs_m * stride_ln,
        lse,
        mask=mask_m,
        eviction_policy="evict_first",
    )


@triton.jit
def _flash_c5_bwd_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    dout_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    N,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oh,
    stride_on,
    stride_od,
    stride_lh,
    stride_ln,
    SCALE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N

    q = tl.load(
        q_ptr + pid_h * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None],
        other=0.0,
    )
    o = tl.load(
        out_ptr + pid_h * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od,
        mask=mask_m[:, None],
        other=0.0,
    )
    do = tl.load(
        dout_ptr + pid_h * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od,
        mask=mask_m[:, None],
        other=0.0,
    )
    lse = tl.load(lse_ptr + pid_h * stride_lh + offs_m * stride_ln, mask=mask_m, other=0.0)
    di = tl.sum(o * do, axis=1)
    tl.store(delta_ptr + pid_h * stride_lh + offs_m * stride_ln, di, mask=mask_m)

    dq = tl.zeros((BLOCK_M, HEAD_DIM), dtype=q.dtype)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        k = tl.load(
            k_ptr + pid_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptr + pid_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
        p = tl.where(mask_n[None, :], tl.exp(qk - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = p * (dp - di[:, None]) * SCALE
        dq = dq + tl.dot(ds.to(k.dtype), k, input_precision="ieee")

    tl.store(
        dq_ptr + pid_h * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
        dq,
        mask=mask_m[:, None],
        eviction_policy="evict_first",
    )


@triton.jit
def _flash_c5_bwd_dkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    dout_ptr,
    lse_ptr,
    delta_ptr,
    dk_ptr,
    dv_ptr,
    N,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oh,
    stride_on,
    stride_od,
    stride_lh,
    stride_ln,
    SCALE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_n = offs_n < N

    k = tl.load(
        k_ptr + pid_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None],
        other=0.0,
    )
    v = tl.load(
        v_ptr + pid_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        mask=mask_n[:, None],
        other=0.0,
    )
    dk = tl.zeros((BLOCK_N, HEAD_DIM), dtype=k.dtype)
    dv = tl.zeros((BLOCK_N, HEAD_DIM), dtype=v.dtype)

    for start_m in range(0, N, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < N
        q = tl.load(
            q_ptr + pid_h * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None],
            other=0.0,
        )
        do = tl.load(
            dout_ptr
            + pid_h * stride_oh
            + offs_m[:, None] * stride_on
            + offs_d[None, :] * stride_od,
            mask=mask_m[:, None],
            other=0.0,
        )
        lse = tl.load(lse_ptr + pid_h * stride_lh + offs_m * stride_ln, mask=mask_m, other=0.0)
        di = tl.load(delta_ptr + pid_h * stride_lh + offs_m * stride_ln, mask=mask_m, other=0.0)

        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
        p = tl.where(mask_m[:, None] & mask_n[None, :], tl.exp(qk - lse[:, None]), 0.0)
        dv = dv + tl.dot(tl.trans(p.to(do.dtype)), do, input_precision="ieee")
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = p * (dp - di[:, None]) * SCALE
        dk = dk + tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision="ieee")

    tl.store(
        dk_ptr + pid_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        dk,
        mask=mask_n[:, None],
        eviction_policy="evict_first",
    )
    tl.store(
        dv_ptr + pid_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        dv,
        mask=mask_n[:, None],
        eviction_policy="evict_first",
    )


def _c5_tiles(device: torch.device, dtype: torch.dtype, head_dim: int) -> tuple[int, int, int, int]:
    """Mosaicos sin spills ajustados por perfil GPU (GB10 sm_121 / RTX PRO 6000 sm_120)."""
    return active_gpu_profile(device).c5_tiles(dtype, head_dim)


class _FusedFlashC5(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        ctx.set_materialize_grads(False)
        heads, rows, head_dim = q.shape
        out = torch.empty_like(q)
        lse = torch.empty((heads, rows), dtype=q.dtype, device=q.device)
        bm, bn, nw, ns = _c5_tiles(q.device, q.dtype, head_dim)
        _flash_c5_fwd_kernel[(triton.cdiv(rows, bm), heads)](
            q,
            k,
            v,
            out,
            lse,
            rows,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            SCALE=1.0 / math.sqrt(head_dim),
            HEAD_DIM=head_dim,
            BLOCK_M=bm,
            BLOCK_N=bn,
            num_warps=nw,
            num_stages=ns,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        if dout is None:
            return None, None, None
        q, k, v, out, lse = ctx.saved_tensors
        dout = dout.contiguous()
        heads, rows, head_dim = q.shape
        bm, bn, nw, ns = _c5_tiles(q.device, q.dtype, head_dim)
        scale = 1.0 / math.sqrt(head_dim)
        delta = torch.empty_like(lse)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _flash_c5_bwd_dq_kernel[(triton.cdiv(rows, bm), heads)](
            q,
            k,
            v,
            out,
            dout,
            lse,
            delta,
            dq,
            rows,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            SCALE=scale,
            HEAD_DIM=head_dim,
            BLOCK_M=bm,
            BLOCK_N=bn,
            num_warps=nw,
            num_stages=ns,
        )
        _flash_c5_bwd_dkv_kernel[(triton.cdiv(rows, bn), heads)](
            q,
            k,
            v,
            dout,
            lse,
            delta,
            dk,
            dv,
            rows,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            SCALE=scale,
            HEAD_DIM=head_dim,
            BLOCK_M=bm,
            BLOCK_N=bn,
            num_warps=nw,
            num_stages=ns,
        )
        return dq, dk, dv


def fused_flash_cross_asset(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """FlashAttention-2 IEEE exacta sobre [H, N, HD] en CUDA sin materializar la matriz N×N."""
    if q.shape[1] == 1:
        # Con un único activo softmax([s]) = 1 y sus derivadas respecto a q y k son nulas.
        return v + q * 0.0 + k * 0.0
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return _FusedFlashC5.apply(q, k, v)
    heads, rows, head_dim = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((heads, rows), dtype=q.dtype, device=q.device)
    bm, bn, nw, ns = _c5_tiles(q.device, q.dtype, head_dim)
    _flash_c5_fwd_kernel[(triton.cdiv(rows, bm), heads)](
        q,
        k,
        v,
        out,
        lse,
        rows,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        SCALE=1.0 / math.sqrt(head_dim),
        HEAD_DIM=head_dim,
        BLOCK_M=bm,
        BLOCK_N=bn,
        num_warps=nw,
        num_stages=ns,
    )
    return out


# ===========================================================================
# 4. FUSED PRICE_RETURNS + LEARNING_SCALE + PRESENCE MASK
# ===========================================================================
@triton.jit
def _price_returns_scale_kernel(
    in_ptr,
    center_ptr,
    inv_ptr,
    out_ptr,
    stride_ib,
    stride_it,
    stride_ic,
    stride_ob,
    stride_ot,
    stride_oc,
    B,
    T: tl.constexpr,
    HAS_PRESENCE: tl.constexpr,
    DO_RETURNS: tl.constexpr,
    DO_SCALE: tl.constexpr,
    CLIP: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b = pid * 64 + tl.arange(0, 64)
    b_mask = b < B

    if DO_SCALE:
        c0 = tl.load(center_ptr + 0)
        i0 = tl.load(inv_ptr + 0)
        c1 = tl.load(center_ptr + 1)
        i1 = tl.load(inv_ptr + 1)
        c2 = tl.load(center_ptr + 2)
        i2 = tl.load(inv_ptr + 2)
        c3 = tl.load(center_ptr + 3)
        i3 = tl.load(inv_ptr + 3)
        c4 = tl.load(center_ptr + 4)
        i4 = tl.load(inv_ptr + 4)

    has_prev = tl.zeros((64,), dtype=tl.int1)
    prev_close = tl.zeros((64,), dtype=tl.float32)

    base_in = b * stride_ib
    base_out = b * stride_ob

    for t in tl.static_range(T):
        off_in = base_in + t * stride_it
        off_out = base_out + t * stride_ot
        o = tl.load(in_ptr + off_in + 0 * stride_ic, mask=b_mask, other=0.0)
        h = tl.load(in_ptr + off_in + 1 * stride_ic, mask=b_mask, other=0.0)
        l = tl.load(in_ptr + off_in + 2 * stride_ic, mask=b_mask, other=0.0)
        c = tl.load(in_ptr + off_in + 3 * stride_ic, mask=b_mask, other=0.0)
        v = tl.load(in_ptr + off_in + 4 * stride_ic, mask=b_mask, other=0.0)
        if t == 0:
            prev_close = tl.zeros_like(c)
        if HAS_PRESENCE:
            pr = tl.load(in_ptr + off_in + 5 * stride_ic, mask=b_mask, other=0.0)
            present = pr > 0.0
        else:
            present = b_mask

        if DO_RETURNS:
            r0 = tl.where(present & has_prev, o - prev_close, 0.0)
            r1 = tl.where(present, c - o, 0.0)
            r2 = tl.where(present, h - tl.maximum(o, c), 0.0)
            r3 = tl.where(present, tl.minimum(o, c) - l, 0.0)
            r4 = v
            prev_close = tl.where(present, c, prev_close)
            has_prev = has_prev | present
        else:
            r0, r1, r2, r3, r4 = o, h, l, c, v

        if DO_SCALE:
            r0 = tl.minimum(tl.maximum((r0 - c0) * i0, -CLIP), CLIP)
            r1 = tl.minimum(tl.maximum((r1 - c1) * i1, -CLIP), CLIP)
            r2 = tl.minimum(tl.maximum((r2 - c2) * i2, -CLIP), CLIP)
            r3 = tl.minimum(tl.maximum((r3 - c3) * i3, -CLIP), CLIP)
            r4 = tl.minimum(tl.maximum((r4 - c4) * i4, -CLIP), CLIP)
            if HAS_PRESENCE:
                r0 = tl.where(present, r0, 0.0)
                r1 = tl.where(present, r1, 0.0)
                r2 = tl.where(present, r2, 0.0)
                r3 = tl.where(present, r3, 0.0)
                r4 = tl.where(present, r4, 0.0)

        tl.store(out_ptr + off_out + 0 * stride_oc, r0, mask=b_mask)
        tl.store(out_ptr + off_out + 1 * stride_oc, r1, mask=b_mask)
        tl.store(out_ptr + off_out + 2 * stride_oc, r2, mask=b_mask)
        tl.store(out_ptr + off_out + 3 * stride_oc, r3, mask=b_mask)
        tl.store(out_ptr + off_out + 4 * stride_oc, r4, mask=b_mask)
        if HAS_PRESENCE:
            tl.store(out_ptr + off_out + 5 * stride_oc, pr, mask=b_mask)


@triton.jit
def _max_i32(a, b):
    return tl.maximum(a, b)


@triton.jit
def _price_returns_scale_vec_kernel(
    in_ptr,
    center_ptr,
    inv_ptr,
    out_ptr,
    B,
    T: tl.constexpr,
    C: tl.constexpr,
    HAS_PRESENCE: tl.constexpr,
    DO_RETURNS: tl.constexpr,
    DO_SCALE: tl.constexpr,
    CLIP: tl.constexpr,
):
    """Rendimientos + escalado + máscara totalmente coalescidos en T con scan paralelo."""
    b = tl.program_id(0).to(tl.int64)
    t = tl.arange(0, T)
    base = b * (T * C)
    off = base + t * C

    o = tl.load(in_ptr + off + 0)
    h = tl.load(in_ptr + off + 1)
    l = tl.load(in_ptr + off + 2)
    c = tl.load(in_ptr + off + 3)
    v = tl.load(in_ptr + off + 4)

    if HAS_PRESENCE:
        pr = tl.load(in_ptr + off + 5)
        present = pr > 0.0
    else:
        present = tl.full((T,), True, dtype=tl.int1)

    if DO_RETURNS:
        if HAS_PRESENCE:
            t_prev = tl.maximum(t - 1, 0)
            prev_pr = tl.load(in_ptr + base + t_prev * C + 5) > 0.0
            shifted_idx = tl.where((t > 0) & prev_pr, t - 1, -1)
            prev_idx = tl.associative_scan(shifted_idx, axis=0, combine_fn=_max_i32)
            has_prev = prev_idx >= 0
            prev_close = tl.load(in_ptr + base + tl.maximum(prev_idx, 0) * C + 3)
        else:
            has_prev = t > 0
            prev_close = tl.load(in_ptr + base + tl.maximum(t - 1, 0) * C + 3)

        z = tl.zeros_like(o)
        r0 = tl.where(present & has_prev, o - prev_close, z)
        r1 = tl.where(present, c - o, z)
        r2 = tl.where(present, h - tl.maximum(o, c), z)
        r3 = tl.where(present, tl.minimum(o, c) - l, z)
        r4 = v
    else:
        r0, r1, r2, r3, r4 = o, h, l, c, v

    if DO_SCALE:
        c0 = tl.load(center_ptr + 0)
        i0 = tl.load(inv_ptr + 0)
        c1 = tl.load(center_ptr + 1)
        i1 = tl.load(inv_ptr + 1)
        c2 = tl.load(center_ptr + 2)
        i2 = tl.load(inv_ptr + 2)
        c3 = tl.load(center_ptr + 3)
        i3 = tl.load(inv_ptr + 3)
        c4 = tl.load(center_ptr + 4)
        i4 = tl.load(inv_ptr + 4)
        clip_hi = tl.full([], CLIP, dtype=r0.dtype)
        clip_lo = -clip_hi
        r0 = tl.minimum(tl.maximum((r0 - c0) * i0, clip_lo), clip_hi)
        r1 = tl.minimum(tl.maximum((r1 - c1) * i1, clip_lo), clip_hi)
        r2 = tl.minimum(tl.maximum((r2 - c2) * i2, clip_lo), clip_hi)
        r3 = tl.minimum(tl.maximum((r3 - c3) * i3, clip_lo), clip_hi)
        r4 = tl.minimum(tl.maximum((r4 - c4) * i4, clip_lo), clip_hi)
        if HAS_PRESENCE:
            z = tl.zeros_like(r0)
            r0 = tl.where(present, r0, z)
            r1 = tl.where(present, r1, z)
            r2 = tl.where(present, r2, z)
            r3 = tl.where(present, r3, z)
            r4 = tl.where(present, r4, z)

    tl.store(out_ptr + off + 0, r0)
    tl.store(out_ptr + off + 1, r1)
    tl.store(out_ptr + off + 2, r2)
    tl.store(out_ptr + off + 3, r3)
    tl.store(out_ptr + off + 4, r4)
    if HAS_PRESENCE:
        tl.store(out_ptr + off + 5, pr)


def fused_price_returns_and_scale(
    window: Tensor,
    center: Tensor | None,
    inverse: Tensor | None,
    *,
    do_returns: bool,
    do_scale: bool,
    clip: float = 5.0,
) -> Tensor:
    """Rendimientos de precios + escalado + recorte + presencia en un único kernel Triton."""
    window = window.contiguous()
    rows, steps, channels = window.shape
    out = torch.empty_like(window)
    has_presence = channels == 6
    c_ptr = center if do_scale else window
    i_ptr = inverse if do_scale else window
    profile = active_gpu_profile(window.device)
    if steps >= 2 and (steps & (steps - 1)) == 0 and steps <= 256:
        _price_returns_scale_vec_kernel[(rows,)](
            window,
            c_ptr,
            i_ptr,
            out,
            rows,
            T=steps,
            C=channels,
            HAS_PRESENCE=has_presence,
            DO_RETURNS=do_returns,
            DO_SCALE=do_scale,
            CLIP=float(clip),
            num_warps=profile.price_returns_warps,
            num_stages=1,
        )
        return out
    _price_returns_scale_kernel[(triton.cdiv(rows, 64),)](
        window,
        c_ptr,
        i_ptr,
        out,
        window.stride(0),
        window.stride(1),
        window.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        rows,
        T=steps,
        HAS_PRESENCE=has_presence,
        DO_RETURNS=do_returns,
        DO_SCALE=do_scale,
        CLIP=float(clip),
        num_warps=2,
        num_stages=1,
    )
    return out


# ===========================================================================
# 4B. FUSED LEARNING_SCALE FOR DENSE (PRESENCE) & NUMERIC-TRIPLE MODALITIES
# ===========================================================================
@triton.jit
def _scale_presence_kernel(
    in_ptr,
    center_ptr,
    inv_ptr,
    pres_ptr,
    out_ptr,
    stride_ib,
    stride_ob,
    stride_pb,
    B,
    F_DIM: tl.constexpr,
    PRES_COL: tl.constexpr,
    CLIP: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_f = tl.program_id(1).to(tl.int64)
    b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
    b_mask = b < B
    f_mask = f < F_DIM
    mask = b_mask[:, None] & f_mask[None, :]

    pres = tl.load(pres_ptr + b * stride_pb + PRES_COL, mask=b_mask, other=False)
    cen = tl.load(center_ptr + f, mask=f_mask, other=0.0)
    inv = tl.load(inv_ptr + f, mask=f_mask, other=0.0)
    x = tl.load(in_ptr + b[:, None] * stride_ib + f[None, :], mask=mask, other=0.0)
    clip_hi = tl.full([], CLIP, dtype=x.dtype)
    y = tl.minimum(tl.maximum((x - cen[None, :]) * inv[None, :], -clip_hi), clip_hi)
    y = tl.where(pres[:, None], y, tl.zeros_like(y))
    tl.store(out_ptr + b[:, None] * stride_ob + f[None, :], y, mask=mask)


@triton.jit
def _scale_numeric_triple_kernel(
    in_ptr,
    center_ptr,
    inv_ptr,
    out_ptr,
    stride_ib,
    stride_ob,
    B,
    THIRD: tl.constexpr,
    CLIP: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_t = tl.program_id(1).to(tl.int64)
    b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    b_mask = b < B
    t_mask = t < THIRD
    mask = b_mask[:, None] & t_mask[None, :]

    in_base = b[:, None] * stride_ib + t[None, :]
    out_base = b[:, None] * stride_ob + t[None, :]

    v = tl.load(in_ptr + in_base, mask=mask, other=0.0)
    obs = tl.load(in_ptr + in_base + THIRD, mask=mask, other=0.0)
    age = tl.load(in_ptr + in_base + 2 * THIRD, mask=mask, other=0.0)

    cv = tl.load(center_ptr + t, mask=t_mask, other=0.0)
    iv = tl.load(inv_ptr + t, mask=t_mask, other=0.0)
    ca = tl.load(center_ptr + t + 2 * THIRD, mask=t_mask, other=0.0)
    ia = tl.load(inv_ptr + t + 2 * THIRD, mask=t_mask, other=0.0)

    clip_hi = tl.full([], CLIP, dtype=v.dtype)
    sv = tl.minimum(tl.maximum((v - cv[None, :]) * iv[None, :], -clip_hi), clip_hi) * obs
    sa = tl.minimum(tl.maximum((age - ca[None, :]) * ia[None, :], -clip_hi), clip_hi) * obs

    tl.store(out_ptr + out_base, sv, mask=mask)
    tl.store(out_ptr + out_base + THIRD, obs, mask=mask)
    tl.store(out_ptr + out_base + 2 * THIRD, sa, mask=mask)


def fused_scale_presence(
    values: Tensor,
    center: Tensor,
    inverse: Tensor,
    presence: Tensor,
    presence_col: int,
    *,
    clip: float,
) -> Tensor:
    """Escalado + recorte + máscara de presencia en un único lanzamiento (bit-exacto)."""
    values = values.contiguous()
    presence = presence.contiguous()
    rows, width = values.shape
    out = torch.empty_like(values)
    profile = active_gpu_profile(values.device)
    bb = profile.scale_block_b
    bf = profile.scale_block_f if width >= 64 else max(16, triton.next_power_of_2(width))
    _scale_presence_kernel[(triton.cdiv(rows, bb), triton.cdiv(width, bf))](
        values,
        center.contiguous(),
        inverse.contiguous(),
        presence,
        out,
        values.stride(0),
        out.stride(0),
        presence.stride(0),
        rows,
        F_DIM=width,
        PRES_COL=presence_col,
        CLIP=float(clip),
        BLOCK_B=bb,
        BLOCK_F=bf,
        num_warps=profile.scale_warps,
        num_stages=1,
    )
    return out


def fused_scale_numeric_triple(
    values: Tensor,
    center: Tensor,
    inverse: Tensor,
    *,
    third: int,
    clip: float,
) -> Tensor:
    """Escalado + recorte + enmascarado [val*obs, obs, age*obs] en un único lanzamiento."""
    values = values.contiguous()
    rows = values.shape[0]
    out = torch.empty_like(values)
    profile = active_gpu_profile(values.device)
    bb = profile.scale_block_b
    bt = 32 if third <= 32 else 64
    _scale_numeric_triple_kernel[(triton.cdiv(rows, bb), triton.cdiv(third, bt))](
        values,
        center.contiguous(),
        inverse.contiguous(),
        out,
        values.stride(0),
        out.stride(0),
        rows,
        THIRD=third,
        CLIP=float(clip),
        BLOCK_B=bb,
        BLOCK_T=bt,
        num_warps=profile.scale_warps,
        num_stages=1,
    )
    return out


# ===========================================================================
# 4C. ALGEBRAIC-FACTORIZED FULLY-FUSED COMPACT PRICE TRANSFORMER (EXACT IEEE)
# ===========================================================================
@triton.jit
def _fac_price_attn_fwd_kernel(
    prices_ptr,
    w_proj_ptr,
    b_proj_ptr,
    pos_ptr,
    gamma_ptr,
    beta_ptr,
    q_tilde_ptr,
    q_dot_bk_ptr,
    c_out_ptr,
    lse_out_ptr,
    B,
    T: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    GATE_LAST: tl.constexpr,
    SAVE_LSE: tl.constexpr,
):
    """Atención factorizada exacta del último token fusionada con proj+pos+LN1 (cero KV en DRAM)."""
    b = tl.program_id(0).to(tl.int64)
    hi = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, D)

    bp = tl.load(b_proj_ptr + d)
    gam = tl.load(gamma_ptr + d)
    bet = tl.load(beta_ptr + d)
    w0 = tl.load(w_proj_ptr + d * C + 0)
    if C >= 2:
        w1 = tl.load(w_proj_ptr + d * C + 1)
    if C >= 3:
        w2 = tl.load(w_proj_ptr + d * C + 2)
    if C >= 4:
        w3 = tl.load(w_proj_ptr + d * C + 3)
    if C >= 5:
        w4 = tl.load(w_proj_ptr + d * C + 4)
    if C >= 6:
        w5 = tl.load(w_proj_ptr + d * C + 5)
    if C >= 7:
        w6 = tl.load(w_proj_ptr + d * C + 6)
    if C >= 8:
        w7 = tl.load(w_proj_ptr + d * C + 7)

    qt = tl.load(q_tilde_ptr + b * (H * D) + hi * D + d)
    qbk = tl.load(q_dot_bk_ptr + b * H + hi)

    m_i = tl.full([], float("-inf"), dtype=bp.dtype)
    l_i = tl.full([], 0.0, dtype=bp.dtype)
    acc = tl.zeros((D,), dtype=bp.dtype)

    inv_d = tl.full([], 1.0 / D, dtype=bp.dtype)
    eps_t = tl.full([], EPS, dtype=bp.dtype)
    scale_t = tl.full([], SCALE, dtype=bp.dtype)
    p_base = b * (T * C)

    for t0 in tl.static_range(0, T, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        t_off = p_base + t * C
        p0 = tl.load(prices_ptr + t_off + 0)
        if C >= 2:
            p1 = tl.load(prices_ptr + t_off + 1)
        if C >= 3:
            p2 = tl.load(prices_ptr + t_off + 2)
        if C >= 4:
            p3 = tl.load(prices_ptr + t_off + 3)
        if C >= 5:
            p4 = tl.load(prices_ptr + t_off + 4)
        if C >= 6:
            p5 = tl.load(prices_ptr + t_off + 5)
        if C >= 7:
            p6 = tl.load(prices_ptr + t_off + 6)
        if C >= 8:
            p7 = tl.load(prices_ptr + t_off + 7)
        if GATE_LAST and C == 6:
            pres = p5 > 0.0
            z0 = tl.zeros_like(p0)
            p0 = tl.where(pres, p0, z0)
            p1 = tl.where(pres, p1, z0)
            p2 = tl.where(pres, p2, z0)
            p3 = tl.where(pres, p3, z0)
            p4 = tl.where(pres, p4, z0)

        y = p0[:, None] * w0[None, :]
        if C >= 2:
            y = y + p1[:, None] * w1[None, :]
        if C >= 3:
            y = y + p2[:, None] * w2[None, :]
        if C >= 4:
            y = y + p3[:, None] * w3[None, :]
        if C >= 5:
            y = y + p4[:, None] * w4[None, :]
        if C >= 6:
            y = y + p5[:, None] * w5[None, :]
        if C >= 7:
            y = y + p6[:, None] * w6[None, :]
        if C >= 8:
            y = y + p7[:, None] * w7[None, :]
        pos = tl.load(pos_ptr + t[:, None] * D + d[None, :])
        y = y + bp[None, :] + pos

        mean = tl.sum(y, axis=1) * inv_d
        diff = y - mean[:, None]
        var = tl.sum(diff * diff, axis=1) * inv_d
        denom = tl.sqrt(var + eps_t)
        rstd = tl.div_rn(1.0, denom) if denom.dtype == tl.float32 else (1.0 / denom)
        x = diff * rstd[:, None] * gam[None, :] + bet[None, :]

        scores = (tl.sum(x * qt[None, :], axis=1) + qbk) * scale_t
        m_blk = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_blk)
        alpha = tl.exp(m_i - m_new)
        p_exp = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p_exp, axis=0)
        acc = acc * alpha + tl.sum(p_exp[:, None] * x, axis=0)
        m_i = m_new

    inv_l = tl.div_rn(1.0, l_i) if l_i.dtype == tl.float32 else (1.0 / l_i)
    tl.store(c_out_ptr + b * (H * D) + hi * D + d, acc * inv_l)
    if SAVE_LSE:
        tl.store(lse_out_ptr + b * H + hi, m_i + tl.log(l_i))


@triton.jit
def _fac_price_attn_bwd_kernel(
    prices_ptr,
    w_proj_ptr,
    b_proj_ptr,
    pos_ptr,
    gamma_ptr,
    beta_ptr,
    q_tilde_ptr,
    q_dot_bk_ptr,
    lse_ptr,
    dc_ptr,
    c_ptr,
    dq_tilde_ptr,
    dq_dot_bk_ptr,
    dw_part_ptr,
    db_part_ptr,
    dg_part_ptr,
    dbet_part_ptr,
    dprices_ptr,
    B,
    T: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    GATE_LAST: tl.constexpr,
    NEED_DPRICES: tl.constexpr,
):
    """Inversa determinista de 1 pasada sin tensores 3D ni atómicas (usa c · dc = sum p·dp)."""
    pid = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, D)

    bp = tl.load(b_proj_ptr + d)
    gam = tl.load(gamma_ptr + d)
    bet = tl.load(beta_ptr + d)
    w0 = tl.load(w_proj_ptr + d * C + 0)
    if C >= 2:
        w1 = tl.load(w_proj_ptr + d * C + 1)
    if C >= 3:
        w2 = tl.load(w_proj_ptr + d * C + 2)
    if C >= 4:
        w3 = tl.load(w_proj_ptr + d * C + 3)
    if C >= 5:
        w4 = tl.load(w_proj_ptr + d * C + 4)
    if C >= 6:
        w5 = tl.load(w_proj_ptr + d * C + 5)
    if C >= 7:
        w6 = tl.load(w_proj_ptr + d * C + 6)
    if C >= 8:
        w7 = tl.load(w_proj_ptr + d * C + 7)

    acc_dw0 = tl.zeros((D,), dtype=bp.dtype)
    if C >= 2:
        acc_dw1 = tl.zeros((D,), dtype=bp.dtype)
    if C >= 3:
        acc_dw2 = tl.zeros((D,), dtype=bp.dtype)
    if C >= 4:
        acc_dw3 = tl.zeros((D,), dtype=bp.dtype)
    if C >= 5:
        acc_dw4 = tl.zeros((D,), dtype=bp.dtype)
    if C >= 6:
        acc_dw5 = tl.zeros((D,), dtype=bp.dtype)
    if C >= 7:
        acc_dw6 = tl.zeros((D,), dtype=bp.dtype)
    if C >= 8:
        acc_dw7 = tl.zeros((D,), dtype=bp.dtype)
    acc_db = tl.zeros((D,), dtype=bp.dtype)
    acc_dg = tl.zeros((D,), dtype=bp.dtype)
    acc_dbet = tl.zeros((D,), dtype=bp.dtype)

    inv_d = tl.full([], 1.0 / D, dtype=bp.dtype)
    eps_t = tl.full([], EPS, dtype=bp.dtype)
    scale_t = tl.full([], SCALE, dtype=bp.dtype)

    b_start = pid * ROWS_PER_CTA
    b_end = tl.minimum(b_start + ROWS_PER_CTA, B)

    for b in range(b_start, b_end):
        for hi in tl.static_range(H):
            tl.store(dq_tilde_ptr + b * (H * D) + hi * D + d, tl.zeros((D,), dtype=bp.dtype))
            tl.store(dq_dot_bk_ptr + b * H + hi, tl.full([], 0.0, dtype=bp.dtype))

        p_base = b * (T * C)
        for t0 in tl.static_range(0, T, BLOCK_T):
            t = t0 + tl.arange(0, BLOCK_T)
            t_off = p_base + t * C
            p0 = tl.load(prices_ptr + t_off + 0)
            if C >= 2:
                p1 = tl.load(prices_ptr + t_off + 1)
            if C >= 3:
                p2 = tl.load(prices_ptr + t_off + 2)
            if C >= 4:
                p3 = tl.load(prices_ptr + t_off + 3)
            if C >= 5:
                p4 = tl.load(prices_ptr + t_off + 4)
            if C >= 6:
                p5 = tl.load(prices_ptr + t_off + 5)
            if C >= 7:
                p6 = tl.load(prices_ptr + t_off + 6)
            if C >= 8:
                p7 = tl.load(prices_ptr + t_off + 7)
            if GATE_LAST and C == 6:
                pres = p5 > 0.0
                z0 = tl.zeros_like(p0)
                p0 = tl.where(pres, p0, z0)
                p1 = tl.where(pres, p1, z0)
                p2 = tl.where(pres, p2, z0)
                p3 = tl.where(pres, p3, z0)
                p4 = tl.where(pres, p4, z0)

            y = p0[:, None] * w0[None, :]
            if C >= 2:
                y = y + p1[:, None] * w1[None, :]
            if C >= 3:
                y = y + p2[:, None] * w2[None, :]
            if C >= 4:
                y = y + p3[:, None] * w3[None, :]
            if C >= 5:
                y = y + p4[:, None] * w4[None, :]
            if C >= 6:
                y = y + p5[:, None] * w5[None, :]
            if C >= 7:
                y = y + p6[:, None] * w6[None, :]
            if C >= 8:
                y = y + p7[:, None] * w7[None, :]
            pos = tl.load(pos_ptr + t[:, None] * D + d[None, :])
            y = y + bp[None, :] + pos

            mean = tl.sum(y, axis=1) * inv_d
            diff = y - mean[:, None]
            var = tl.sum(diff * diff, axis=1) * inv_d
            denom = tl.sqrt(var + eps_t)
            rstd = tl.div_rn(1.0, denom) if denom.dtype == tl.float32 else (1.0 / denom)
            xhat = diff * rstd[:, None]
            x = xhat * gam[None, :] + bet[None, :]

            dx = tl.zeros((BLOCK_T, D), dtype=bp.dtype)
            for hi in tl.static_range(H):
                off_hd = b * (H * D) + hi * D + d
                off_h = b * H + hi
                qt_h = tl.load(q_tilde_ptr + off_hd)
                dc_h = tl.load(dc_ptr + off_hd)
                c_h = tl.load(c_ptr + off_hd)
                qbk_h = tl.load(q_dot_bk_ptr + off_h)
                lse_h = tl.load(lse_ptr + off_h)
                dot_h = tl.sum(c_h * dc_h, axis=0)

                s_h = (tl.sum(x * qt_h[None, :], axis=1) + qbk_h) * scale_t
                p_h = tl.exp(s_h - lse_h)
                dp_h = tl.sum(x * dc_h[None, :], axis=1)
                ds_h = p_h * (dp_h - dot_h) * scale_t

                dx = dx + ds_h[:, None] * qt_h[None, :] + p_h[:, None] * dc_h[None, :]

                dqt_prev = tl.load(dq_tilde_ptr + off_hd)
                tl.store(dq_tilde_ptr + off_hd, dqt_prev + tl.sum(ds_h[:, None] * x, axis=0))
                dqbk_prev = tl.load(dq_dot_bk_ptr + off_h)
                tl.store(dq_dot_bk_ptr + off_h, dqbk_prev + tl.sum(ds_h, axis=0))

            acc_dbet = acc_dbet + tl.sum(dx, axis=0)
            acc_dg = acc_dg + tl.sum(dx * xhat, axis=0)
            dxhat = dx * gam[None, :]
            c1 = tl.sum(dxhat * xhat, axis=1) * inv_d
            c2 = tl.sum(dxhat, axis=1) * inv_d
            dy = (dxhat - c2[:, None] - xhat * c1[:, None]) * rstd[:, None]

            acc_db = acc_db + tl.sum(dy, axis=0)
            acc_dw0 = acc_dw0 + tl.sum(dy * p0[:, None], axis=0)
            if C >= 2:
                acc_dw1 = acc_dw1 + tl.sum(dy * p1[:, None], axis=0)
            if C >= 3:
                acc_dw2 = acc_dw2 + tl.sum(dy * p2[:, None], axis=0)
            if C >= 4:
                acc_dw3 = acc_dw3 + tl.sum(dy * p3[:, None], axis=0)
            if C >= 5:
                acc_dw4 = acc_dw4 + tl.sum(dy * p4[:, None], axis=0)
            if C >= 6:
                acc_dw5 = acc_dw5 + tl.sum(dy * p5[:, None], axis=0)
            if C >= 7:
                acc_dw6 = acc_dw6 + tl.sum(dy * p6[:, None], axis=0)
            if C >= 8:
                acc_dw7 = acc_dw7 + tl.sum(dy * p7[:, None], axis=0)

            if NEED_DPRICES:
                dp0 = tl.sum(dy * w0[None, :], axis=1)
                if C >= 2:
                    dp1 = tl.sum(dy * w1[None, :], axis=1)
                if C >= 3:
                    dp2 = tl.sum(dy * w2[None, :], axis=1)
                if C >= 4:
                    dp3 = tl.sum(dy * w3[None, :], axis=1)
                if C >= 5:
                    dp4 = tl.sum(dy * w4[None, :], axis=1)
                if GATE_LAST and C == 6:
                    z0 = tl.zeros_like(dp0)
                    dp0 = tl.where(pres, dp0, z0)
                    dp1 = tl.where(pres, dp1, z0)
                    dp2 = tl.where(pres, dp2, z0)
                    dp3 = tl.where(pres, dp3, z0)
                    dp4 = tl.where(pres, dp4, z0)
                tl.store(dprices_ptr + t_off + 0, dp0)
                if C >= 2:
                    tl.store(dprices_ptr + t_off + 1, dp1)
                if C >= 3:
                    tl.store(dprices_ptr + t_off + 2, dp2)
                if C >= 4:
                    tl.store(dprices_ptr + t_off + 3, dp3)
                if C >= 5:
                    tl.store(dprices_ptr + t_off + 4, dp4)
                if C >= 6:
                    tl.store(dprices_ptr + t_off + 5, tl.sum(dy * w5[None, :], axis=1))
                if C >= 7:
                    tl.store(dprices_ptr + t_off + 6, tl.sum(dy * w6[None, :], axis=1))
                if C >= 8:
                    tl.store(dprices_ptr + t_off + 7, tl.sum(dy * w7[None, :], axis=1))

    dw_base = pid * (D * C) + d * C
    tl.store(dw_part_ptr + dw_base + 0, acc_dw0)
    if C >= 2:
        tl.store(dw_part_ptr + dw_base + 1, acc_dw1)
    if C >= 3:
        tl.store(dw_part_ptr + dw_base + 2, acc_dw2)
    if C >= 4:
        tl.store(dw_part_ptr + dw_base + 3, acc_dw3)
    if C >= 5:
        tl.store(dw_part_ptr + dw_base + 4, acc_dw4)
    if C >= 6:
        tl.store(dw_part_ptr + dw_base + 5, acc_dw5)
    if C >= 7:
        tl.store(dw_part_ptr + dw_base + 6, acc_dw6)
    if C >= 8:
        tl.store(dw_part_ptr + dw_base + 7, acc_dw7)
    tl.store(db_part_ptr + pid * D + d, acc_db)
    tl.store(dg_part_ptr + pid * D + d, acc_dg)
    tl.store(dbet_part_ptr + pid * D + d, acc_dbet)


def _fac_block_params(device: torch.device, dtype: torch.dtype, steps: int, dim: int):
    profile = active_gpu_profile(device)
    max_bt_f = profile.fac_fwd_block_t
    max_bt_b = profile.fac_bwd_block_t
    fw = profile.fac_fwd_warps
    bw = profile.fac_bwd_warps
    if dtype == torch.float64:
        max_bt_f = min(max_bt_f, 32)
        max_bt_b = min(max_bt_b, 8)
        fw = min(fw, 4)
        bw = 8
    elif dim > 64:
        max_bt_f = min(max_bt_f, 32)
        max_bt_b = min(max_bt_b, 16)
        fw = min(fw, 4)
        bw = 8
    bt_f = min(steps, max_bt_f)
    bt_b = min(steps, max_bt_b)
    while steps % bt_f != 0:
        bt_f //= 2
    while steps % bt_b != 0:
        bt_b //= 2
    return bt_f, bt_b, fw, bw, profile


class _FusedFactorizedPriceAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prices, w_proj, b_proj, pos, gamma, beta, q_tilde, q_dot_bk, eps, gate_last):
        ctx.set_materialize_grads(False)
        rows, steps, channels = prices.shape
        heads, dim = q_tilde.shape[1], q_tilde.shape[2]
        head_dim = dim // heads
        bt_f, bt_b, fw, bw, profile = _fac_block_params(prices.device, prices.dtype, steps, dim)
        c_out = torch.empty((rows, heads, dim), dtype=prices.dtype, device=prices.device)
        lse = torch.empty((rows, heads), dtype=prices.dtype, device=prices.device)
        _fac_price_attn_fwd_kernel[(rows, heads)](
            prices,
            w_proj,
            b_proj,
            pos,
            gamma,
            beta,
            q_tilde,
            q_dot_bk,
            c_out,
            lse,
            rows,
            T=steps,
            C=channels,
            D=dim,
            H=heads,
            BLOCK_T=bt_f,
            SCALE=1.0 / math.sqrt(head_dim),
            EPS=eps,
            GATE_LAST=gate_last,
            SAVE_LSE=True,
            num_warps=fw,
            num_stages=1,
        )
        ctx.save_for_backward(prices, w_proj, b_proj, pos, gamma, beta, q_tilde, q_dot_bk, lse, c_out)
        ctx.meta = (eps, gate_last, bt_b, bw, profile.fac_bwd_ctas)
        return c_out

    @staticmethod
    @once_differentiable
    def backward(ctx, dc):
        if dc is None:
            return (None,) * 10
        prices, w_proj, b_proj, pos, gamma, beta, q_tilde, q_dot_bk, lse, c_out = ctx.saved_tensors
        eps, gate_last, bt_b, bw, base_ctas = ctx.meta
        dc = dc.contiguous()
        rows, steps, channels = prices.shape
        heads, dim = q_tilde.shape[1], q_tilde.shape[2]
        head_dim = dim // heads
        profile = active_gpu_profile(prices.device)
        rows_per_cta, num_ctas = profile.wave_aligned_ctas(rows, base_ctas)
        need_dp = ctx.needs_input_grad[0]
        dqt = torch.empty_like(q_tilde)
        dqbk = torch.empty_like(q_dot_bk)
        dw_p = torch.empty((num_ctas, dim, channels), dtype=prices.dtype, device=prices.device)
        db_p = torch.empty((num_ctas, dim), dtype=prices.dtype, device=prices.device)
        dg_p = torch.empty((num_ctas, dim), dtype=prices.dtype, device=prices.device)
        dbet_p = torch.empty((num_ctas, dim), dtype=prices.dtype, device=prices.device)
        dprices = torch.empty_like(prices) if need_dp else prices
        _fac_price_attn_bwd_kernel[(num_ctas,)](
            prices,
            w_proj,
            b_proj,
            pos,
            gamma,
            beta,
            q_tilde,
            q_dot_bk,
            lse,
            dc,
            c_out,
            dqt,
            dqbk,
            dw_p,
            db_p,
            dg_p,
            dbet_p,
            dprices,
            rows,
            T=steps,
            C=channels,
            D=dim,
            H=heads,
            BLOCK_T=bt_b,
            ROWS_PER_CTA=rows_per_cta,
            SCALE=1.0 / math.sqrt(head_dim),
            EPS=eps,
            GATE_LAST=gate_last,
            NEED_DPRICES=need_dp,
            num_warps=bw,
            num_stages=1,
        )
        return (
            dprices if need_dp else None,
            dw_p.sum(0) if ctx.needs_input_grad[1] else None,
            db_p.sum(0) if ctx.needs_input_grad[2] else None,
            None,
            dg_p.sum(0) if ctx.needs_input_grad[4] else None,
            dbet_p.sum(0) if ctx.needs_input_grad[5] else None,
            dqt,
            dqbk,
            None,
            None,
        )


def fused_price_factorized_last_attn(
    prices: Tensor,
    w_proj: Tensor,
    b_proj: Tensor,
    positions: Tensor,
    ln_weight: Tensor,
    ln_bias: Tensor,
    q_tilde: Tensor,
    q_dot_bk: Tensor,
    *,
    eps: float = 1e-5,
    gate_last: bool = False,
) -> Tensor:
    """Contexto de atención [B, H, D] fusionado con proj+pos+LN1 sin materializar normalized ni KV."""
    prices = prices.contiguous()
    w_proj = w_proj.contiguous()
    b_proj = b_proj.contiguous()
    positions = positions.contiguous()
    ln_weight = ln_weight.contiguous()
    ln_bias = ln_bias.contiguous()
    q_tilde = q_tilde.contiguous()
    q_dot_bk = q_dot_bk.contiguous()
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (prices, w_proj, b_proj, ln_weight, ln_bias, q_tilde, q_dot_bk)
    ):
        return _FusedFactorizedPriceAttn.apply(
            prices, w_proj, b_proj, positions, ln_weight, ln_bias, q_tilde, q_dot_bk, eps, gate_last
        )
    rows, steps, channels = prices.shape
    heads, dim = q_tilde.shape[1], q_tilde.shape[2]
    head_dim = dim // heads
    bt_f, _, fw, _, _ = _fac_block_params(prices.device, prices.dtype, steps, dim)
    c_out = torch.empty((rows, heads, dim), dtype=prices.dtype, device=prices.device)
    _fac_price_attn_fwd_kernel[(rows, heads)](
        prices,
        w_proj,
        b_proj,
        positions,
        ln_weight,
        ln_bias,
        q_tilde,
        q_dot_bk,
        c_out,
        q_dot_bk,
        rows,
        T=steps,
        C=channels,
        D=dim,
        H=heads,
        BLOCK_T=bt_f,
        SCALE=1.0 / math.sqrt(head_dim),
        EPS=eps,
        GATE_LAST=gate_last,
        SAVE_LSE=False,
        num_warps=fw,
        num_stages=1,
    )
    return c_out


# ===========================================================================
# 5. FUSED TIE-AVERAGED RANK PROBABILITIES AFTER SORT (BIT-EXACT)
# ===========================================================================
@triton.jit
def _tie_rank_prob_kernel(
    sorted_ptr,
    idx_ptr,
    count_ptr,
    prob_out_ptr,
    N,
    stride_sc,
    stride_sn,
    stride_pn,
    stride_pc,
    BLOCK_N: tl.constexpr,
    MAX_LOG2: tl.constexpr,
):
    c = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    pos = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    cnt = tl.load(count_ptr + c)
    in_bounds = pos < N
    is_present = pos < cnt

    col_base = c * stride_sc
    val = tl.load(sorted_ptr + col_base + pos * stride_sn, mask=in_bounds, other=0.0)
    orig_row = tl.load(idx_ptr + col_base + pos * stride_sn, mask=in_bounds, other=0)

    prev_val = tl.load(
        sorted_ptr + col_base + tl.maximum(pos - 1, 0) * stride_sn, mask=in_bounds, other=0.0
    )
    next_val = tl.load(
        sorted_ptr + col_base + tl.minimum(pos + 1, tl.maximum(cnt - 1, 0)) * stride_sn,
        mask=in_bounds,
        other=0.0,
    )

    need_left = is_present & (pos > 0) & (prev_val == val)
    need_right = is_present & (pos + 1 < cnt) & (next_val == val)

    lo = pos
    if tl.max(need_left.to(tl.int32), axis=0) > 0:
        left = tl.where(need_left, 0, pos)
        right = pos
        for _ in tl.static_range(MAX_LOG2):
            active = left < right
            mid = (left + right) >> 1
            mv = tl.load(sorted_ptr + col_base + mid * stride_sn, mask=active, other=0.0)
            go_right = mv < val
            left = tl.where(active & go_right, mid + 1, left)
            right = tl.where(active & (~go_right), mid, right)
        lo = left

    hi = pos + 1
    if tl.max(need_right.to(tl.int32), axis=0) > 0:
        left = pos + 1
        right = tl.where(need_right, cnt, pos + 1)
        for _ in tl.static_range(MAX_LOG2):
            active = left < right
            mid = (left + right) >> 1
            mv = tl.load(sorted_ptr + col_base + mid * stride_sn, mask=active, other=0.0)
            go_right = mv <= val
            left = tl.where(active & go_right, mid + 1, left)
            right = tl.where(active & (~go_right), mid, right)
        hi = left

    rank = (lo + hi + 1).to(val.dtype) * 0.5
    cnt_f = tl.maximum(cnt, 1).to(val.dtype)
    raw_prob = tl.div_rn(rank - 0.5, cnt_f) if val.dtype == tl.float32 else (rank - 0.5) / cnt_f
    prob = tl.where(is_present, raw_prob, 0.5)
    tl.store(prob_out_ptr + orig_row * stride_pn + c * stride_pc, prob, mask=in_bounds)


def fused_tie_rank_probabilities(ordered: Tensor, indices: Tensor, counts: Tensor) -> Tensor:
    """Probabilidades (r − 0,5) / n con empates promediados en [N, C], idéntico bit a bit."""
    cols, rows = ordered.shape
    prob = torch.empty((rows, cols), dtype=ordered.dtype, device=ordered.device)
    block_n = 256
    max_log2 = max(1, (rows - 1).bit_length() + 1)
    _tie_rank_prob_kernel[(cols, triton.cdiv(rows, block_n))](
        ordered,
        indices,
        counts,
        prob,
        rows,
        ordered.stride(0),
        ordered.stride(1),
        prob.stride(0),
        prob.stride(1),
        BLOCK_N=block_n,
        MAX_LOG2=max_log2,
        num_warps=4,
        num_stages=1,
    )
    return prob


# ===========================================================================
# 6. FUSED EXACT STABLE TOP-K DESCENDING FOR EPISODIC READOUT (BIT-EXACT)
# ===========================================================================
@triton.jit
def _stable_topk_desc_kernel(
    scores_ptr,
    out_ptr,
    N,
    stride_sb,
    stride_sn,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_N)
    valid = offs < N
    s = tl.load(scores_ptr + b * stride_sb + offs * stride_sn, mask=valid, other=float("-inf"))
    for k in tl.static_range(K):
        best = tl.argmax(s, axis=0, tie_break_left=True)
        tl.store(out_ptr + b * K + k, best.to(tl.int64))
        s = tl.where(offs == best, float("-inf"), s)


def fused_stable_topk_desc(scores: Tensor, count: int) -> Tensor:
    """Índices top-k descendentes estables (empates al menor índice), idéntico bit a bit."""
    rows, total = scores.shape
    out = torch.empty((rows, count), dtype=torch.int64, device=scores.device)
    block_n = triton.next_power_of_2(max(total, 2))
    _stable_topk_desc_kernel[(rows,)](
        scores,
        out,
        total,
        scores.stride(0),
        scores.stride(1),
        K=count,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=1,
    )
    return out
