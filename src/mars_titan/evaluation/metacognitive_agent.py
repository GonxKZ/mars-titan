"""Cartera Convexa Diferenciable `P+` y Núcleo Agéntico Metacognitivo (`A5–A8`).

Resuelve directamente la tensión empírica entre neutralización factorial (+27 % ICIR) y
turnover (+53 % en A4) incorporando la penalización factorial y los costes `L1` + impacto `3/2`
dentro de un optimizador convexo rápido (FISTA con proyección exacta), disponible tanto en
NumPy (inferencia/post-entrenamiento) como en PyTorch diferenciable (entrenamiento end-to-end).

Incluye el Gobernador Metacognitivo de Inferencia Activa (`A5–A8`):
  - A5: Abstención epistémica por Energía Libre (desacuerdo, Venn-Abers, Hopfield OOD, router).
  - A6: Fusión bayesiana de creencias multi-especialista con ponderación causal.
  - A8: Monitor CUSUM / Page-Hinkley de decaimiento de alfa por especialista.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _HAS_TRITON = False

AGENT_VARIATIONS = (
    "agent_none",
    "agent_p_plus_only",
    "agent_moe_routing_p_plus",
    "agent_full_active_inference",
)


def _gram_max_eig_power_iter(gram: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    """Estima/acota superiormente `lambda_max(B^T B)` en 5 iters puras sin cuSOLVER ni syncs."""
    k = gram.shape[0]
    g = gram.detach()
    idx = torch.arange(1, k + 1, device=g.device, dtype=g.dtype)
    v = g.abs().sum(dim=-1) + 1e-3 * idx
    v = v / v.norm().clamp_min(1e-12)
    for _ in range(n_iters):
        gv = g @ v
        nrm = gv.norm()
        v = torch.where(nrm > 1e-12, gv / nrm.clamp_min(1e-12), v)
    rayleigh = (v * (g @ v)).sum().clamp_min(0.0)
    # Margen conservador 1.02x para garantizar paso FISTA estrictamente <= 1/L
    return rayleigh * 1.02


if _HAS_TRITON:  # pragma: no cover

    @triton.jit
    def _project_pplus_1cta_kernel(
        v_ptr,
        cap_ptr,
        out_ptr,
        n_rows,
        gross_limit,
        BLOCK_N: tl.constexpr,
    ):
        """Proyección euclídea exacta `{1^T w = 0, ||w||_1 <= G, |w_i| <= cap_i}` en 1 CTA."""
        offs = tl.arange(0, BLOCK_N)
        mask = offs < n_rows
        v = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        cap = tl.load(cap_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        zero = tl.full([], 0.0, tl.float32)
        one = tl.full([], 1.0, tl.float32)
        n_f = tl.sum(mask.to(tl.float32), axis=0)
        mu_cur = tl.sum(tl.where(mask, v, zero), axis=0) / tl.maximum(n_f, one)
        v_min = tl.min(tl.where(mask, v - cap, 1e30), axis=0)
        v_max = tl.max(tl.where(mask, v + cap, -1e30), axis=0)

        # Paso 1: resolver mu para lam = 0 (Newton salvaguardado por bisección)
        lo_m = v_min - 1e-3
        hi_m = v_max + 1e-3
        for _ in tl.static_range(0, 12):
            sh = v - mu_cur
            w = tl.minimum(tl.maximum(sh, -cap), cap)
            f = tl.sum(tl.where(mask, w, zero), axis=0)
            done_m = tl.abs(f) <= 1e-10
            lo_m = tl.where(f > zero, mu_cur, lo_m)
            hi_m = tl.where(f < zero, mu_cur, hi_m)
            act = tl.sum((mask & (tl.abs(sh) < cap)).to(tl.float32), axis=0)
            cand = mu_cur + f / tl.maximum(act, one)
            use_n = (act > 0.5) & (cand >= lo_m) & (cand <= hi_m)
            mu_next = tl.where(use_n, cand, 0.5 * (lo_m + hi_m))
            mu_cur = tl.where(done_m, mu_cur, mu_next)

        sh0 = v - mu_cur
        w0 = tl.minimum(tl.maximum(sh0, -cap), cap)
        l1_0 = tl.sum(tl.where(mask, tl.abs(w0), zero), axis=0)
        need = l1_0 > gross_limit

        # Paso 2: si ||w0||_1 > gross_limit, resolver par dual (mu, lam) exacto
        lo_l = zero
        hi_l = tl.max(tl.where(mask, tl.abs(v), zero), axis=0) + 1e-3
        n_p0 = tl.sum((mask & (sh0 > zero) & (tl.abs(sh0) < cap)).to(tl.float32), axis=0)
        n_n0 = tl.sum((mask & (sh0 < zero) & (tl.abs(sh0) < cap)).to(tl.float32), axis=0)
        sl0 = (4.0 * n_p0 * n_n0) / tl.maximum(n_p0 + n_n0, one)
        lam_init = (l1_0 - gross_limit) / tl.maximum(sl0, one)
        lam = tl.where(need, tl.minimum(tl.maximum(lam_init, zero), hi_l), zero)

        for _ in tl.static_range(0, 10):
            lo_m = v_min - lam - 1e-3
            hi_m = v_max + lam + 1e-3
            for _ in tl.static_range(0, 8):
                sh = v - mu_cur
                ab = tl.abs(sh)
                shr = tl.where(sh >= zero, one, -one) * tl.maximum(ab - lam, zero)
                w = tl.minimum(tl.maximum(shr, -cap), cap)
                f = tl.sum(tl.where(mask, w, zero), axis=0)
                done_m = tl.abs(f) <= 1e-10
                lo_m = tl.where(f > zero, mu_cur, lo_m)
                hi_m = tl.where(f < zero, mu_cur, hi_m)
                act = tl.sum((mask & (ab > lam) & (tl.abs(shr) < cap)).to(tl.float32), axis=0)
                cand = mu_cur + f / tl.maximum(act, one)
                use_n = (act > 0.5) & (cand >= lo_m) & (cand <= hi_m)
                mu_next = tl.where(use_n, cand, 0.5 * (lo_m + hi_m))
                mu_cur = tl.where(done_m, mu_cur, mu_next)
            sh = v - mu_cur
            ab = tl.abs(sh)
            shr = tl.where(sh >= zero, one, -one) * tl.maximum(ab - lam, zero)
            w = tl.minimum(tl.maximum(shr, -cap), cap)
            g = tl.sum(tl.where(mask, tl.abs(w), zero), axis=0) - gross_limit
            done_l = tl.abs(g) <= 1e-9
            lo_l = tl.where(g > zero, lam, lo_l)
            hi_l = tl.where(g < zero, lam, hi_l)
            n_p = tl.sum((mask & (sh > lam) & (tl.abs(shr) < cap)).to(tl.float32), axis=0)
            n_n = tl.sum((mask & (sh < -lam) & (tl.abs(shr) < cap)).to(tl.float32), axis=0)
            sl = (4.0 * n_p * n_n) / tl.maximum(n_p + n_n, one)
            cand = lam + g / tl.maximum(sl, 1e-6)
            use_n = (sl > 0.5) & (cand >= lo_l) & (cand <= hi_l)
            lam_next = tl.where(use_n, cand, 0.5 * (lo_l + hi_l))
            lam = tl.where(need, tl.where(done_l, lam, lam_next), zero)

        # Ajuste final de mu para el lam convergente (garantiza 1^T w == 0 exacto)
        lo_m = v_min - lam - 1e-3
        hi_m = v_max + lam + 1e-3
        for _ in tl.static_range(0, 8):
            sh = v - mu_cur
            ab = tl.abs(sh)
            shr = tl.where(sh >= zero, one, -one) * tl.maximum(ab - lam, zero)
            w = tl.minimum(tl.maximum(shr, -cap), cap)
            f = tl.sum(tl.where(mask, w, zero), axis=0)
            done_m = tl.abs(f) <= 1e-10
            lo_m = tl.where(f > zero, mu_cur, lo_m)
            hi_m = tl.where(f < zero, mu_cur, hi_m)
            act = tl.sum((mask & (ab > lam) & (tl.abs(shr) < cap)).to(tl.float32), axis=0)
            cand = mu_cur + f / tl.maximum(act, one)
            use_n = (act > 0.5) & (cand >= lo_m) & (cand <= hi_m)
            mu_next = tl.where(use_n, cand, 0.5 * (lo_m + hi_m))
            mu_cur = tl.where(done_m, mu_cur, mu_next)

        sh = v - mu_cur
        ab = tl.abs(sh)
        shr = tl.where(sh >= zero, one, -one) * tl.maximum(ab - lam, zero)
        w_f = tl.minimum(tl.maximum(shr, -cap), cap)
        w_out = tl.where(need, w_f, w0)
        tl.store(out_ptr + offs, w_out, mask=mask)

    @triton.jit
    def _project_box_neutral_1cta_kernel(
        v_ptr,
        out_ptr,
        act_ptr,
        n_rows,
        pos_cap: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Proyección euclídea exacta `{1^T w = 0, |w_i| <= pos_cap}` + máscara activa en 1 CTA."""
        offs = tl.arange(0, BLOCK_N)
        mask = offs < n_rows
        v = tl.load(v_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        zero = tl.full([], 0.0, ACC_DTYPE)
        one = tl.full([], 1.0, ACC_DTYPE)
        cap_t = tl.full([], pos_cap, ACC_DTYPE)
        n_f = tl.sum(mask.to(ACC_DTYPE), axis=0)
        mu_cur = tl.sum(tl.where(mask, v, zero), axis=0) / tl.maximum(n_f, one)
        lo_m = tl.min(tl.where(mask, v - cap_t, 1e30), axis=0) - 1e-3
        hi_m = tl.max(tl.where(mask, v + cap_t, -1e30), axis=0) + 1e-3
        tol = tl.full([], 1e-12 if ACC_DTYPE == tl.float64 else 1e-7, ACC_DTYPE)
        for _ in tl.static_range(0, 32):
            sh = v - mu_cur
            w = tl.minimum(tl.maximum(sh, -cap_t), cap_t)
            f = tl.sum(tl.where(mask, w, zero), axis=0)
            done_m = tl.abs(f) <= tol
            lo_m = tl.where(f > zero, mu_cur, lo_m)
            hi_m = tl.where(f < zero, mu_cur, hi_m)
            act = tl.sum((mask & (tl.abs(sh) < cap_t)).to(ACC_DTYPE), axis=0)
            cand = mu_cur + f / tl.maximum(act, one)
            use_n = (act > 0.5) & (cand >= lo_m) & (cand <= hi_m)
            mu_next = tl.where(use_n, cand, 0.5 * (lo_m + hi_m))
            mu_cur = tl.where(done_m, mu_cur, mu_next)
        # Pulido sin cancelación FP32: resta `dmu` sobre `sh0 = v - mu_cur` (|v| >> cap)
        sh0 = v - mu_cur
        w0_p = tl.minimum(tl.maximum(sh0, -cap_t), cap_t)
        f0 = tl.sum(tl.where(mask, w0_p, zero), axis=0)
        act0 = tl.sum((mask & (tl.abs(sh0) < cap_t)).to(ACC_DTYPE), axis=0)
        dmu = tl.where(act0 > 0.5, f0 / tl.maximum(act0, one), zero)
        sh_f = sh0 - dmu
        w_f = tl.minimum(tl.maximum(sh_f, -cap_t), cap_t)
        act_f = (mask & (tl.abs(sh_f) < cap_t)).to(ACC_DTYPE)
        tl.store(out_ptr + offs, w_f, mask=mask)
        tl.store(act_ptr + offs, act_f, mask=mask)

    @triton.jit
    def _diff_portfolio_step_fwd_1cta_kernel(
        mu_ptr,
        var_ptr,
        w0_ptr,
        y_ptr,
        w_old_ptr,
        fac_ptr,
        step_ptr,
        w_next_ptr,
        y_next_ptr,
        w_proj_ptr,
        act_ptr,
        st_ptr,
        l1_ptr,
        n_rows,
        beta_fista,
        fac_scale,
        risk_aversion: tl.constexpr,
        eff_l1: tl.constexpr,
        pos_cap: tl.constexpr,
        gross_limit: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Paso FISTA completo (grad + soft-thresh + proyección box-neutral + L1) en 1 CTA."""
        offs = tl.arange(0, BLOCK_N)
        mask = offs < n_rows
        zero = tl.full([], 0.0, ACC_DTYPE)
        one = tl.full([], 1.0, ACC_DTYPE)
        step_v = tl.load(step_ptr).to(ACC_DTYPE)
        mu = tl.load(mu_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        var_d = tl.load(var_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        w0 = tl.load(w0_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        w_old = tl.load(w_old_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        fac = tl.load(fac_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        grad = (
            -mu
            + tl.full([], risk_aversion, ACC_DTYPE) * (var_d * y)
            + fac_scale.to(ACC_DTYPE) * fac
        )
        v = y - step_v * grad
        d = v - w0
        thr = step_v * tl.full([], eff_l1, ACC_DTYPE)
        abs_d = tl.abs(d)
        st_mask = mask & (abs_d > thr)
        sgn_d = tl.where(d > zero, one, tl.where(d < zero, -one, zero))
        w_raw = w0 + sgn_d * tl.maximum(abs_d - thr, zero)

        cap_t = tl.full([], pos_cap, ACC_DTYPE)
        n_f = tl.sum(mask.to(ACC_DTYPE), axis=0)
        mu_cur = tl.sum(tl.where(mask, w_raw, zero), axis=0) / tl.maximum(n_f, one)
        lo_m = tl.min(tl.where(mask, w_raw - cap_t, 1e30), axis=0) - 1e-3
        hi_m = tl.max(tl.where(mask, w_raw + cap_t, -1e30), axis=0) + 1e-3
        tol = tl.full([], 1e-12 if ACC_DTYPE == tl.float64 else 1e-7, ACC_DTYPE)
        for _ in tl.static_range(0, 32):
            sh = w_raw - mu_cur
            w_c = tl.minimum(tl.maximum(sh, -cap_t), cap_t)
            f = tl.sum(tl.where(mask, w_c, zero), axis=0)
            done_m = tl.abs(f) <= tol
            lo_m = tl.where(f > zero, mu_cur, lo_m)
            hi_m = tl.where(f < zero, mu_cur, hi_m)
            act = tl.sum((mask & (tl.abs(sh) < cap_t)).to(ACC_DTYPE), axis=0)
            cand = mu_cur + f / tl.maximum(act, one)
            use_n = (act > 0.5) & (cand >= lo_m) & (cand <= hi_m)
            mu_next = tl.where(use_n, cand, 0.5 * (lo_m + hi_m))
            mu_cur = tl.where(done_m, mu_cur, mu_next)
        sh0 = w_raw - mu_cur
        w0_p = tl.minimum(tl.maximum(sh0, -cap_t), cap_t)
        f0 = tl.sum(tl.where(mask, w0_p, zero), axis=0)
        act0 = tl.sum((mask & (tl.abs(sh0) < cap_t)).to(ACC_DTYPE), axis=0)
        dmu = tl.where(act0 > 0.5, f0 / tl.maximum(act0, one), zero)
        sh_f = sh0 - dmu
        w_proj = tl.minimum(tl.maximum(sh_f, -cap_t), cap_t)
        act_f = (mask & (tl.abs(sh_f) < cap_t)).to(ACC_DTYPE)

        l1 = tl.sum(tl.where(mask, tl.abs(w_proj), zero), axis=0)
        g_lim = tl.full([], gross_limit, ACC_DTYPE)
        scale = tl.minimum(g_lim / tl.maximum(l1, tl.full([], 1e-8, ACC_DTYPE)), one)
        w_next = w_proj * scale
        y_next = w_next + beta_fista.to(ACC_DTYPE) * (w_next - w_old)
        tl.store(w_next_ptr + offs, w_next, mask=mask)
        tl.store(y_next_ptr + offs, y_next, mask=mask)
        tl.store(w_proj_ptr + offs, w_proj, mask=mask)
        tl.store(act_ptr + offs, act_f, mask=mask)
        tl.store(st_ptr + offs, st_mask.to(ACC_DTYPE), mask=mask)
        tl.store(l1_ptr + offs, l1, mask=(offs == 0))

    @triton.jit
    def _diff_portfolio_step_bwd_1cta_kernel(
        gw_ptr,
        gy_base_ptr,
        fac_gv_ptr,
        gmu_ptr,
        var_ptr,
        w_proj_ptr,
        act_ptr,
        st_ptr,
        l1_ptr,
        step_ptr,
        gw_out_ptr,
        gy_base_out_ptr,
        gv_out_ptr,
        gmu_out_ptr,
        n_rows,
        y_mult,
        beta_k,
        fac_scale,
        risk_aversion: tl.constexpr,
        gross_limit: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Paso inverso exacto (BPTT FISTA + derivada implícita proyección + L1) en 1 CTA."""
        offs = tl.arange(0, BLOCK_N)
        mask = offs < n_rows
        zero = tl.full([], 0.0, ACC_DTYPE)
        one = tl.full([], 1.0, ACC_DTYPE)
        step_v = tl.load(step_ptr).to(ACC_DTYPE)
        gw = tl.load(gw_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        gy_b = tl.load(gy_base_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        fac_gv = tl.load(fac_gv_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        gy = gy_b - (step_v * fac_scale.to(ACC_DTYPE)) * fac_gv

        g_wnext = gw + y_mult.to(ACC_DTYPE) * gy
        gw_prev = -beta_k.to(ACC_DTYPE) * gy

        w_proj = tl.load(w_proj_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        l1 = tl.load(l1_ptr).to(ACC_DTYPE)
        g_lim = tl.full([], gross_limit, ACC_DTYPE)
        l1_safe = tl.maximum(l1, tl.full([], 1e-8, ACC_DTYPE))
        scale = tl.minimum(g_lim / l1_safe, one)
        dot_gw = tl.sum(tl.where(mask, g_wnext * w_proj, zero), axis=0)
        coeff = tl.where(l1 > g_lim, (scale * dot_gw) / l1_safe, zero)
        sgn_w = tl.where(w_proj > zero, one, tl.where(w_proj < zero, -one, zero))
        g_proj = scale * g_wnext - coeff * sgn_w

        act_f = tl.load(act_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        n_act = tl.sum(tl.where(mask, act_f, zero), axis=0)
        sum_ga = tl.sum(tl.where(mask, g_proj * act_f, zero), axis=0)
        mean_ga = tl.where(n_act > 0.5, sum_ga / tl.maximum(n_act, one), zero)
        g_raw = act_f * (g_proj - mean_ga)

        st_f = tl.load(st_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        g_v = st_f * g_raw

        gmu = tl.load(gmu_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        var_d = tl.load(var_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        gmu_new = gmu + step_v * g_v
        gy_base_new = (one - step_v * tl.full([], risk_aversion, ACC_DTYPE) * var_d) * g_v

        tl.store(gw_out_ptr + offs, gw_prev, mask=mask)
        tl.store(gy_base_out_ptr + offs, gy_base_new, mask=mask)
        tl.store(gv_out_ptr + offs, g_v, mask=mask)
        tl.store(gmu_out_ptr + offs, gmu_new, mask=mask)


@torch.library.custom_op("mars_titan_pplus::project_box_neutral_fwd", mutates_args=())
def _project_box_neutral_fwd_op(
    v: torch.Tensor,
    pos_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Proyección `{1^T w = 0, |w_i| <= pos_cap}` en 1-CTA Triton + respaldo CPU (0 syncs)."""
    n = int(v.shape[0])
    cap = float(pos_cap)
    if n <= 1 or cap <= 0.0:
        return (
            torch.zeros_like(v, memory_format=torch.contiguous_format),
            torch.zeros_like(v, memory_format=torch.contiguous_format),
        )
    if _HAS_TRITON and v.is_cuda and n <= 8192 and v.dtype in (torch.float32, torch.float64):
        v_c = v.contiguous()
        out = torch.empty_like(v_c)
        act_out = torch.empty_like(v_c)
        block_n = 1 << (max(32, n) - 1).bit_length()
        num_warps = 8 if block_n >= 2048 else 4
        tl_acc = tl.float64 if v_c.dtype == torch.float64 else tl.float32
        _project_box_neutral_1cta_kernel[(1,)](
            v_c,
            out,
            act_out,
            n,
            pos_cap=cap,
            BLOCK_N=block_n,
            ACC_DTYPE=tl_acc,
            num_warps=num_warps,
        )
        return out, act_out

    calc_dtype = torch.float64 if v.dtype == torch.float64 else torch.float32
    v_f = v.to(dtype=calc_dtype).contiguous()
    lo_m = (v_f - cap).min() - 1e-3
    hi_m = (v_f + cap).max() + 1e-3
    mu_cur = v_f.mean()
    tol = 1e-12 if calc_dtype == torch.float64 else 1e-7
    for _ in range(32):
        sh = v_f - mu_cur
        w = torch.clamp(sh, min=-cap, max=cap)
        f_sum = w.sum()
        done_m = f_sum.abs() <= tol
        lo_m = torch.where(f_sum > 0.0, mu_cur, lo_m)
        hi_m = torch.where(f_sum < 0.0, mu_cur, hi_m)
        act = (sh.abs() < cap).to(calc_dtype).sum()
        cand = mu_cur + f_sum / act.clamp_min(1.0)
        use_n = (act > 0.5) & (cand >= lo_m) & (cand <= hi_m)
        mu_next = torch.where(use_n, cand, 0.5 * (lo_m + hi_m))
        mu_cur = torch.where(done_m, mu_cur, mu_next)
    # Pulido sin cancelación FP32 cuando |v| >> cap
    sh0 = v_f - mu_cur
    w0_p = torch.clamp(sh0, min=-cap, max=cap)
    f0 = w0_p.sum()
    act0 = (sh0.abs() < cap).to(calc_dtype).sum()
    dmu = torch.where(act0 > 0.5, f0 / act0.clamp_min(1.0), torch.zeros_like(f0))
    sh_f = sh0 - dmu
    w_f = torch.clamp(sh_f, min=-cap, max=cap)
    act_f = (sh_f.abs() < cap).to(dtype=calc_dtype)
    return w_f.to(dtype=v.dtype).contiguous(), act_f.to(dtype=v.dtype).contiguous()


@_project_box_neutral_fwd_op.register_fake
def _project_box_neutral_fwd_fake(
    v: torch.Tensor,
    pos_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ = pos_cap
    return (
        torch.empty_like(v, memory_format=torch.contiguous_format),
        torch.empty_like(v, memory_format=torch.contiguous_format),
    )


def _project_box_neutral_setup_ctx(
    ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]
) -> None:
    _ = inputs
    _w_proj, active_f = output
    ctx.save_for_backward(active_f)


def _project_box_neutral_backward(
    ctx: Any,
    grad_w: torch.Tensor | None,
    _grad_act: torch.Tensor | None,
) -> tuple[torch.Tensor | None, None]:
    if grad_w is None or not ctx.needs_input_grad[0]:
        return None, None
    (active_f,) = ctx.saved_tensors
    # Derivada implícita exacta de P_{1^T w=0, |w|<=c}(v): J^T g = m * (g - mean_{i in A}(g_i))
    g_act = grad_w * active_f
    n_act = active_f.sum()
    mean_g = torch.where(n_act > 0.5, g_act.sum() / n_act.clamp_min(1.0), torch.zeros_like(n_act))
    return active_f * (grad_w - mean_g), None


_project_box_neutral_fwd_op.register_autograd(
    _project_box_neutral_backward, setup_context=_project_box_neutral_setup_ctx
)


def project_dollar_neutral_box_torch(
    v: torch.Tensor,
    *,
    pos_cap: float,
    gross_limit: float,
) -> torch.Tensor:
    """Proyección euclídea diferenciable rápida `{1^T w=0, |w_i|<=pos_cap}` + L1 (0 syncs)."""
    n = v.shape[0]
    if n <= 1 or pos_cap <= 0.0 or gross_limit <= 0.0:
        return torch.zeros_like(v)
    orig_dtype = v.dtype
    calc_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    with torch.amp.autocast(device_type=v.device.type, enabled=False):
        v_c = v.to(dtype=calc_dtype)
        w_proj, _ = _project_box_neutral_fwd_op(v_c, float(pos_cap))
        l1 = w_proj.abs().sum()
        scale = torch.clamp(float(gross_limit) / l1.clamp_min(1e-8), max=1.0)
        return (w_proj * scale).to(dtype=orig_dtype)


def _fista_betas(steps: int) -> list[float]:
    t_fista = 1.0
    betas: list[float] = []
    for _ in range(steps):
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_fista * t_fista))
        betas.append((t_fista - 1.0) / t_next)
        t_fista = t_next
    return betas


@torch.library.custom_op("mars_titan_pplus::diff_portfolio_fista_fwd", mutates_args=())
def _diff_portfolio_fista_fwd_op(
    mu_c: torch.Tensor,
    var_diag: torch.Tensor,
    w0: torch.Tensor,
    b_mat: torch.Tensor,
    step_t: torch.Tensor,
    steps: int,
    risk_aversion: float,
    factor_penalty: float,
    eff_l1: float,
    pos_cap: float,
    gross_limit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FISTA diferenciable fusionado en 1 custom_op (1 CTA/paso en CUDA, 0 syncs)."""
    n = int(mu_c.shape[0])
    s_n = max(int(steps), 0)
    dev, dt = mu_c.device, mu_c.dtype
    w_proj_hist = torch.empty((s_n, n), device=dev, dtype=dt)
    act_hist = torch.empty((s_n, n), device=dev, dtype=dt)
    st_hist = torch.empty((s_n, n), device=dev, dtype=dt)
    l1_hist = torch.empty((s_n,), device=dev, dtype=dt)
    if s_n == 0 or n <= 1:
        return (
            w0.clone(memory_format=torch.contiguous_format),
            w_proj_hist,
            act_hist,
            st_hist,
            l1_hist,
        )

    mu_k = mu_c.contiguous()
    var_k = var_diag.contiguous()
    w0_k = w0.contiguous()
    step_k = step_t.reshape(()).contiguous()
    has_fac = b_mat.ndim == 2 and b_mat.shape[1] > 0 and factor_penalty > 0.0
    b_k = b_mat.contiguous() if has_fac else b_mat
    b_t = b_k.T.contiguous() if has_fac else b_mat
    betas = _fista_betas(s_n)

    if _HAS_TRITON and mu_k.is_cuda and n <= 8192 and dt in (torch.float32, torch.float64):
        w_cur = w0_k.clone()
        y_cur = w0_k.clone()
        w_next = torch.empty_like(w_cur)
        y_next = torch.empty_like(y_cur)
        zero_fac = torch.zeros_like(w_cur)
        block_n = 1 << (max(32, n) - 1).bit_length()
        num_warps = 8 if block_n >= 2048 else 4
        tl_acc = tl.float64 if dt == torch.float64 else tl.float32
        fac_sc = float(factor_penalty) if has_fac else 0.0
        for k_i in range(s_n):
            fac_vec = (b_k @ (b_t @ y_cur)) if has_fac else zero_fac
            _diff_portfolio_step_fwd_1cta_kernel[(1,)](
                mu_k,
                var_k,
                w0_k,
                y_cur,
                w_cur,
                fac_vec,
                step_k,
                w_next,
                y_next,
                w_proj_hist[k_i],
                act_hist[k_i],
                st_hist[k_i],
                l1_hist[k_i : k_i + 1],
                n,
                float(betas[k_i]),
                fac_sc,
                risk_aversion=float(risk_aversion),
                eff_l1=float(eff_l1),
                pos_cap=float(pos_cap),
                gross_limit=float(gross_limit),
                BLOCK_N=block_n,
                ACC_DTYPE=tl_acc,
                num_warps=num_warps,
            )
            w_cur, w_next = w_next, w_cur
            y_cur, y_next = y_next, y_cur
        return w_cur.contiguous(), w_proj_hist, act_hist, st_hist, l1_hist

    w_cur = w0_k.clone()
    y_cur = w0_k.clone()
    thr = step_k * float(eff_l1)
    for k_i in range(s_n):
        grad = -mu_k + float(risk_aversion) * (var_k * y_cur)
        if has_fac:
            grad = grad + float(factor_penalty) * (b_k @ (b_t @ y_cur))
        v = y_cur - step_k * grad
        d = v - w0_k
        abs_d = d.abs()
        st_m = (abs_d > thr).to(dtype=dt)
        w_raw = w0_k + torch.sign(d) * torch.relu(abs_d - thr)
        w_p, act_f = _project_box_neutral_fwd_op(w_raw, float(pos_cap))
        l1_v = w_p.abs().sum()
        scale = torch.clamp(float(gross_limit) / l1_v.clamp_min(1e-8), max=1.0)
        w_n = w_p * scale
        y_cur = w_n + float(betas[k_i]) * (w_n - w_cur)
        w_cur = w_n
        w_proj_hist[k_i] = w_p
        act_hist[k_i] = act_f
        st_hist[k_i] = st_m
        l1_hist[k_i] = l1_v
    return w_cur.contiguous(), w_proj_hist, act_hist, st_hist, l1_hist


@_diff_portfolio_fista_fwd_op.register_fake
def _diff_portfolio_fista_fwd_fake(
    mu_c: torch.Tensor,
    var_diag: torch.Tensor,
    w0: torch.Tensor,
    b_mat: torch.Tensor,
    step_t: torch.Tensor,
    steps: int,
    risk_aversion: float,
    factor_penalty: float,
    eff_l1: float,
    pos_cap: float,
    gross_limit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _ = (var_diag, w0, b_mat, step_t, risk_aversion, factor_penalty, eff_l1, pos_cap, gross_limit)
    n = mu_c.shape[0]
    s_n = max(int(steps), 0)
    dev, dt = mu_c.device, mu_c.dtype
    return (
        torch.empty_like(mu_c, memory_format=torch.contiguous_format),
        torch.empty((s_n, n), device=dev, dtype=dt),
        torch.empty((s_n, n), device=dev, dtype=dt),
        torch.empty((s_n, n), device=dev, dtype=dt),
        torch.empty((s_n,), device=dev, dtype=dt),
    )


@torch.library.custom_op("mars_titan_pplus::diff_portfolio_fista_bwd", mutates_args=())
def _diff_portfolio_fista_bwd_op(
    grad_out: torch.Tensor,
    var_diag: torch.Tensor,
    b_mat: torch.Tensor,
    step_t: torch.Tensor,
    w_proj_hist: torch.Tensor,
    act_hist: torch.Tensor,
    st_hist: torch.Tensor,
    l1_hist: torch.Tensor,
    steps: int,
    risk_aversion: float,
    factor_penalty: float,
    gross_limit: float,
) -> torch.Tensor:
    """BPTT inverso exacto de FISTA en 1 custom_op (1 CTA/paso en CUDA, 0 grafo autograd)."""
    n = int(grad_out.shape[0])
    s_n = max(int(steps), 0)
    dt = grad_out.dtype
    if s_n == 0 or n <= 1:
        return torch.zeros_like(grad_out, memory_format=torch.contiguous_format)

    gw = grad_out.contiguous()
    var_k = var_diag.contiguous()
    step_k = step_t.reshape(()).contiguous()
    has_fac = b_mat.ndim == 2 and b_mat.shape[1] > 0 and factor_penalty > 0.0
    b_k = b_mat.contiguous() if has_fac else b_mat
    b_t = b_k.T.contiguous() if has_fac else b_mat
    betas = _fista_betas(s_n)

    if _HAS_TRITON and gw.is_cuda and n <= 8192 and dt in (torch.float32, torch.float64):
        gy_base = torch.zeros_like(gw)
        gmu = torch.zeros_like(gw)
        gw_out = torch.empty_like(gw)
        gy_out = torch.empty_like(gw)
        gv_out = torch.empty_like(gw)
        gmu_out = torch.empty_like(gw)
        zero_fac = torch.zeros_like(gw)
        block_n = 1 << (max(32, n) - 1).bit_length()
        num_warps = 8 if block_n >= 2048 else 4
        tl_acc = tl.float64 if dt == torch.float64 else tl.float32
        fac_gv = zero_fac
        for k_i in range(s_n - 1, -1, -1):
            is_last = k_i == s_n - 1
            y_mult = 0.0 if is_last else (1.0 + betas[k_i])
            beta_k = 0.0 if is_last else betas[k_i]
            fac_sc = 0.0 if (is_last or not has_fac) else float(factor_penalty)
            _diff_portfolio_step_bwd_1cta_kernel[(1,)](
                gw,
                gy_base,
                fac_gv,
                gmu,
                var_k,
                w_proj_hist[k_i],
                act_hist[k_i],
                st_hist[k_i],
                l1_hist[k_i : k_i + 1],
                step_k,
                gw_out,
                gy_out,
                gv_out,
                gmu_out,
                n,
                float(y_mult),
                float(beta_k),
                fac_sc,
                risk_aversion=float(risk_aversion),
                gross_limit=float(gross_limit),
                BLOCK_N=block_n,
                ACC_DTYPE=tl_acc,
                num_warps=num_warps,
            )
            gw, gw_out = gw_out, gw
            gy_base, gy_out = gy_out, gy_base
            gmu, gmu_out = gmu_out, gmu
            if has_fac and k_i > 0:
                fac_gv = b_k @ (b_t @ gv_out)
        return gmu.contiguous()

    gy = torch.zeros_like(gw)
    gmu = torch.zeros_like(gw)
    g_lim = float(gross_limit)
    for k_i in range(s_n - 1, -1, -1):
        is_last = k_i == s_n - 1
        y_mult = 0.0 if is_last else (1.0 + betas[k_i])
        beta_k = 0.0 if is_last else betas[k_i]
        g_wnext = gw + y_mult * gy
        gw_prev = -beta_k * gy

        w_p = w_proj_hist[k_i]
        l1_v = l1_hist[k_i]
        l1_safe = l1_v.clamp_min(1e-8)
        scale = torch.clamp(g_lim / l1_safe, max=1.0)
        dot_gw = (g_wnext * w_p).sum()
        coeff = torch.where(l1_v > g_lim, (scale * dot_gw) / l1_safe, torch.zeros_like(dot_gw))
        g_proj = scale * g_wnext - coeff * torch.sign(w_p)

        act_f = act_hist[k_i]
        n_act = act_f.sum()
        mean_ga = torch.where(
            n_act > 0.5, (g_proj * act_f).sum() / n_act.clamp_min(1.0), torch.zeros_like(n_act)
        )
        g_raw = act_f * (g_proj - mean_ga)
        g_v = st_hist[k_i] * g_raw

        gmu = gmu + step_k * g_v
        gy_new = (1.0 - step_k * float(risk_aversion) * var_k) * g_v
        if has_fac and k_i > 0:
            gy_new = gy_new - (step_k * float(factor_penalty)) * (b_k @ (b_t @ g_v))
        gw = gw_prev
        gy = gy_new
    return gmu.contiguous()


@_diff_portfolio_fista_bwd_op.register_fake
def _diff_portfolio_fista_bwd_fake(
    grad_out: torch.Tensor,
    var_diag: torch.Tensor,
    b_mat: torch.Tensor,
    step_t: torch.Tensor,
    w_proj_hist: torch.Tensor,
    act_hist: torch.Tensor,
    st_hist: torch.Tensor,
    l1_hist: torch.Tensor,
    steps: int,
    risk_aversion: float,
    factor_penalty: float,
    gross_limit: float,
) -> torch.Tensor:
    _ = (
        var_diag,
        b_mat,
        step_t,
        w_proj_hist,
        act_hist,
        st_hist,
        l1_hist,
        steps,
        risk_aversion,
        factor_penalty,
        gross_limit,
    )
    return torch.empty_like(grad_out, memory_format=torch.contiguous_format)


def _diff_portfolio_fista_setup_ctx(
    ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]
) -> None:
    (
        _mu_c,
        var_diag,
        _w0,
        b_mat,
        step_t,
        steps,
        risk_aversion,
        factor_penalty,
        _eff_l1,
        _pos_cap,
        gross_limit,
    ) = inputs
    _w_out, w_proj_hist, act_hist, st_hist, l1_hist = output
    ctx.save_for_backward(var_diag, b_mat, step_t, w_proj_hist, act_hist, st_hist, l1_hist)
    ctx.steps = int(steps)
    ctx.risk_aversion = float(risk_aversion)
    ctx.factor_penalty = float(factor_penalty)
    ctx.gross_limit = float(gross_limit)


def _diff_portfolio_fista_backward(
    ctx: Any,
    grad_w: torch.Tensor | None,
    _g_wp: torch.Tensor | None,
    _g_act: torch.Tensor | None,
    _g_st: torch.Tensor | None,
    _g_l1: torch.Tensor | None,
) -> tuple[torch.Tensor | None, None, None, None, None, None, None, None, None, None, None]:
    if grad_w is None or not ctx.needs_input_grad[0]:
        return (None,) * 11
    var_diag, b_mat, step_t, w_proj_hist, act_hist, st_hist, l1_hist = ctx.saved_tensors
    g_mu = _diff_portfolio_fista_bwd_op(
        grad_w,
        var_diag,
        b_mat,
        step_t,
        w_proj_hist,
        act_hist,
        st_hist,
        l1_hist,
        ctx.steps,
        ctx.risk_aversion,
        ctx.factor_penalty,
        ctx.gross_limit,
    )
    return (g_mu, None, None, None, None, None, None, None, None, None, None)


_diff_portfolio_fista_fwd_op.register_autograd(
    _diff_portfolio_fista_backward, setup_context=_diff_portfolio_fista_setup_ctx
)


def _project_dollar_neutral_box_l1_torch(
    v: torch.Tensor,
    gross_limit: float,
    cap: torch.Tensor,
) -> torch.Tensor:
    """Proyección euclídea exacta en PyTorch/Triton sin host syncs (paridad CPU/GPU > 0.99999)."""
    n = v.shape[0]
    if _HAS_TRITON and v.is_cuda and n <= 8192 and not torch.compiler.is_compiling():
        v_c = v.contiguous()
        cap_c = cap.contiguous()
        out = torch.empty_like(v_c)
        block_n = 1 << (max(32, n) - 1).bit_length()
        num_warps = 8 if block_n >= 2048 else 4
        _project_pplus_1cta_kernel[(1,)](
            v_c,
            cap_c,
            out,
            n,
            float(gross_limit),
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
        return out

    zero = torch.zeros((), device=v.device, dtype=v.dtype)

    def _solve_mu(
        lam: torch.Tensor, mu_init: torch.Tensor, n_iters: int = 12
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lo_m = (v - cap).min() - lam - 1e-3
        hi_m = (v + cap).max() + lam + 1e-3
        mu_cur = mu_init
        for _ in range(n_iters):
            sh = v - mu_cur
            ab = sh.abs()
            shr = torch.sign(sh) * torch.relu(ab - lam)
            w = torch.clamp(shr, min=-cap, max=cap)
            f = w.sum()
            done_m = f.abs() <= 1e-10
            lo_m = torch.where(f > 0.0, mu_cur, lo_m)
            hi_m = torch.where(f < 0.0, mu_cur, hi_m)
            act = ((ab > lam) & (shr.abs() < cap)).to(v.dtype).sum()
            cand = mu_cur + f / act.clamp_min(1.0)
            use_n = (act > 0.5) & (cand >= lo_m) & (cand <= hi_m)
            mu_next = torch.where(use_n, cand, 0.5 * (lo_m + hi_m))
            mu_cur = torch.where(done_m, mu_cur, mu_next)
        sh = v - mu_cur
        shr = torch.sign(sh) * torch.relu(sh.abs() - lam)
        w = torch.clamp(shr, min=-cap, max=cap)
        return w, mu_cur, sh, shr

    w0, mu_cur, sh0, shr0 = _solve_mu(zero, v.mean(), n_iters=12)
    l1_0 = w0.abs().sum()
    need = l1_0 > gross_limit
    lo_l = zero
    hi_l = v.abs().max() + 1e-3
    n_p0 = ((sh0 > 0.0) & (shr0.abs() < cap)).to(v.dtype).sum()
    n_n0 = ((sh0 < 0.0) & (shr0.abs() < cap)).to(v.dtype).sum()
    sl0 = (4.0 * n_p0 * n_n0) / (n_p0 + n_n0).clamp_min(1.0)
    lam = torch.where(
        need,
        torch.clamp((l1_0 - gross_limit) / sl0.clamp_min(1.0), min=0.0, max=hi_l),
        zero,
    )
    for _ in range(10):
        w_c, mu_cur, sh, shr = _solve_mu(lam, mu_cur, n_iters=8)
        g = w_c.abs().sum() - gross_limit
        done_l = g.abs() <= 1e-9
        lo_l = torch.where(g > 0.0, lam, lo_l)
        hi_l = torch.where(g < 0.0, lam, hi_l)
        n_p = ((sh > lam) & (shr.abs() < cap)).to(v.dtype).sum()
        n_n = ((sh < -lam) & (shr.abs() < cap)).to(v.dtype).sum()
        sl = (4.0 * n_p * n_n) / (n_p + n_n).clamp_min(1.0)
        cand = lam + g / sl.clamp_min(1e-6)
        use_n = (sl > 0.5) & (cand >= lo_l) & (cand <= hi_l)
        lam_next = torch.where(use_n, cand, 0.5 * (lo_l + hi_l))
        lam = torch.where(need, torch.where(done_l, lam, lam_next), zero)
    w_f, _, _, _ = _solve_mu(lam, mu_cur, n_iters=8)
    return torch.where(need, w_f, w0)


def align_weights_by_asset_ids(
    prev_weights_by_id: dict[Any, float],
    current_asset_ids: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Alinea `w_prev` por `asset_id` canónico (cero para IPOs) y calcula liquidación de bajas."""
    curr_ids = np.asarray(current_asset_ids)
    curr_keys = [
        x.item() if isinstance(x, np.generic) else x for x in curr_ids.tolist()
    ]
    w_aligned = np.array(
        [float(prev_weights_by_id.get(aid, 0.0)) for aid in curr_keys],
        dtype=np.float64,
    )
    curr_set = set(curr_keys)
    exited_turnover = float(
        sum(abs(float(w_val)) for aid, w_val in prev_weights_by_id.items() if aid not in curr_set)
    )
    return w_aligned, exited_turnover


def _prox_l1_impact32(
    v: np.ndarray,
    w_prev: np.ndarray,
    step_l1: np.ndarray,
    step_imp: np.ndarray,
) -> np.ndarray:
    """Operador proximal exacto en forma cerrada para `c|w - w0| + eta|w - w0|^(3/2)`.

    Con `d = v - w0`, si `|d| <= c` entonces `w = w0` (banda exacta sin operación!).
    Si `r = |d| - c > 0`, resuelve `x + 1.5*eta*sqrt(x) = r` en forma cerrada cuadrática.
    """
    d = v - w_prev
    abs_d = np.abs(d)
    r = np.maximum(abs_d - step_l1, 0.0)
    # u = sqrt(x) satisface u^2 + (1.5 * step_imp) * u - r = 0
    b = 1.5 * step_imp
    u = 0.5 * (-b + np.sqrt(b * b + 4.0 * r))
    delta = np.sign(d) * (u * u)
    return w_prev + delta


def _project_dollar_neutral_box_l1(
    v: np.ndarray,
    gross_limit: float,
    pos_cap: np.ndarray,
) -> np.ndarray:
    """Proyección euclídea exacta sobre `{1^T w = 0, ||w||_1 <= gross_limit, |w_i| <= cap_i}`.

    Usa dualidad exacta: `w(mu, lam) = clip(soft_thresh(v - mu, lam), -cap, cap)` resuelta por
    bisección en `mu` (neutralidad dólar) y `lam >= 0` (límite bruto L1).
    """
    cap = np.asarray(pos_cap, dtype=np.float64)

    def _solve_mu(lam: float) -> np.ndarray:
        lo_m = float(np.min(v - cap) - lam - 1.0)
        hi_m = float(np.max(v + cap) + lam + 1.0)
        for _ in range(36):
            mid = 0.5 * (lo_m + hi_m)
            shifted = v - mid
            shrunk = np.sign(shifted) * np.maximum(np.abs(shifted) - lam, 0.0)
            w_cand = np.clip(shrunk, -cap, cap)
            if float(w_cand.sum()) > 0.0:
                lo_m = mid
            else:
                hi_m = mid
        shifted = v - 0.5 * (lo_m + hi_m)
        return np.clip(np.sign(shifted) * np.maximum(np.abs(shifted) - lam, 0.0), -cap, cap)

    w0 = _solve_mu(0.0)
    if float(np.abs(w0).sum()) <= gross_limit + 1e-10:
        return w0

    lo_l, hi_l = 0.0, float(np.max(np.abs(v)) + 1.0)
    for _ in range(32):
        mid_l = 0.5 * (lo_l + hi_l)
        wm = _solve_mu(mid_l)
        if float(np.abs(wm).sum()) > gross_limit:
            lo_l = mid_l
        else:
            hi_l = mid_l
    return _solve_mu(hi_l)


@dataclass
class ConvexPortfolioOptimizerPPlus:
    """Optimizador convexo rápido FISTA de la capa de cartera `P+`."""

    risk_aversion: float = 2.0
    factor_penalty: float = 25.0
    l1_cost: float = 0.001
    impact_cost: float = 0.01
    gross_limit: float = 1.0
    max_iters: int = 80
    tol: float = 1e-7
    holding_days: float = 5.0

    def solve(
        self,
        mu: np.ndarray,
        *,
        vol: np.ndarray | None = None,
        exposures: np.ndarray | None = None,
        w_prev: np.ndarray | None = None,
        pos_cap: np.ndarray | float = 0.05,
        l1_costs_vec: np.ndarray | None = None,
        impact_costs_vec: np.ndarray | None = None,
    ) -> np.ndarray:
        """Resuelve la cartera óptima `w_t` para una sesión con costes y neutralidad factorial."""
        m = np.nan_to_num(np.asarray(mu, dtype=np.float64), nan=0.0)
        n = len(m)
        if n < 2:
            return np.zeros(n, dtype=np.float64)
        var_diag = (
            np.full(n, 0.02**2, dtype=np.float64)
            if vol is None
            else np.maximum(np.asarray(vol, dtype=np.float64), 1e-4) ** 2
        )
        w0 = (
            np.zeros(n, dtype=np.float64)
            if w_prev is None
            else np.asarray(w_prev, dtype=np.float64)
        )
        cap = (
            np.full(n, float(pos_cap), dtype=np.float64)
            if np.isscalar(pos_cap)
            else np.asarray(pos_cap, dtype=np.float64)
        )
        # Amortizar el coste lineal `l1_cost` sobre el horizonte de tenencia `holding_days`
        # para evitar el colapso de zona muerta `w==0` ante alfas diarios realistas (3-8 bps)
        h_days = max(float(self.holding_days), 1.0)
        c_vec = (
            np.full(n, self.l1_cost / h_days, dtype=np.float64)
            if l1_costs_vec is None
            else np.asarray(l1_costs_vec, dtype=np.float64) / h_days
        )
        e_vec = (
            np.full(n, self.impact_cost, dtype=np.float64)
            if impact_costs_vec is None
            else np.asarray(impact_costs_vec, dtype=np.float64)
        )

        b_mat = None
        if exposures is not None and self.factor_penalty > 0.0:
            b_raw = np.nan_to_num(np.asarray(exposures, dtype=np.float64), nan=0.0)
            if b_raw.ndim == 1:
                b_raw = b_raw[:, None]
            # Columnas centradas con norma L2 unitaria (equivalente a 1/N * B_z @ B_z^T -> lip O(1))
            b_c = b_raw - np.mean(b_raw, axis=0, keepdims=True)
            col_norms = np.sqrt(np.sum(b_c * b_c, axis=0))
            keep = col_norms > 1e-12
            if np.any(keep):
                b_mat = b_c[:, keep] / col_norms[keep]

        # Constante de Lipschitz O(1) independiente de N: gamma*max(var) + lam_fac*||B_tilde||_2^2
        lip = self.risk_aversion * float(np.max(var_diag))
        if b_mat is not None:
            gram = b_mat.T @ b_mat
            s_max2 = max(float(np.linalg.eigvalsh(gram)[-1]), 0.0)
            lip += self.factor_penalty * s_max2
        step = 1.0 / max(lip, 1e-6)

        w = _project_dollar_neutral_box_l1(w0, self.gross_limit, cap)
        y_acc = w.copy()
        t_fista = 1.0
        step_l1 = step * c_vec
        step_imp = step * e_vec

        for _ in range(self.max_iters):
            grad = -m + self.risk_aversion * (var_diag * y_acc)
            if b_mat is not None:
                grad = grad + self.factor_penalty * (b_mat @ (b_mat.T @ y_acc))
            v = y_acc - step * grad
            w_prox = _prox_l1_impact32(v, w0, step_l1, step_imp)
            w_next = _project_dollar_neutral_box_l1(w_prox, self.gross_limit, cap)
            diff = float(np.linalg.norm(w_next - w))
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_fista * t_fista))
            y_acc = w_next + ((t_fista - 1.0) / t_next) * (w_next - w)
            w = w_next
            t_fista = t_next
            if diff < self.tol:
                break
        return w

    @torch.no_grad()
    def solve_torch(
        self,
        mu: torch.Tensor,
        *,
        vol: torch.Tensor | None = None,
        exposures: torch.Tensor | None = None,
        w_prev: torch.Tensor | None = None,
        pos_cap: torch.Tensor | float = 0.05,
        l1_costs_vec: torch.Tensor | None = None,
        impact_costs_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Versión 100 % nativa PyTorch/Triton (GPU/CPU) de `P+` sin syncs ni cuSOLVER."""
        device = mu.device
        orig_dtype = mu.dtype
        n = mu.shape[0]
        if n < 2:
            return torch.zeros(n, device=device, dtype=orig_dtype)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            m = torch.nan_to_num(mu.float(), nan=0.0, posinf=0.0, neginf=0.0)
            var_diag = (
                torch.full((n,), 0.02**2, device=device, dtype=torch.float32)
                if vol is None
                else torch.clamp(vol.to(device=device, dtype=torch.float32), min=1e-4).pow(2)
            )
            w0 = (
                torch.zeros(n, device=device, dtype=torch.float32)
                if w_prev is None
                else w_prev.to(device=device, dtype=torch.float32)
            )
            cap = (
                torch.full((n,), float(pos_cap), device=device, dtype=torch.float32)
                if isinstance(pos_cap, int | float)
                else pos_cap.to(device=device, dtype=torch.float32)
            )
            h_days = max(float(self.holding_days), 1.0)
            c_vec = (
                torch.full((n,), self.l1_cost / h_days, device=device, dtype=torch.float32)
                if l1_costs_vec is None
                else l1_costs_vec.to(device=device, dtype=torch.float32) / h_days
            )
            e_vec = (
                torch.full((n,), self.impact_cost, device=device, dtype=torch.float32)
                if impact_costs_vec is None
                else impact_costs_vec.to(device=device, dtype=torch.float32)
            )

            b_mat = None
            lip = self.risk_aversion * var_diag.max().clamp_min(1e-8)
            if exposures is not None and self.factor_penalty > 0.0 and exposures.shape[-1] > 0:
                b_raw = torch.nan_to_num(
                    exposures.to(device=device, dtype=torch.float32), nan=0.0
                )
                if b_raw.ndim == 1:
                    b_raw = b_raw.unsqueeze(-1)
                # Columnas centradas con norma L2 unitaria (0 host syncs, lip O(1) en N=2600)
                b_c = b_raw - b_raw.mean(dim=0, keepdim=True)
                col_norm = torch.sqrt((b_c * b_c).sum(dim=0, keepdim=True) + 1e-12)
                b_mat = torch.where(col_norm > 1e-6, b_c / col_norm, torch.zeros_like(b_c))
                gram = b_mat.T @ b_mat
                s_max2 = _gram_max_eig_power_iter(gram, n_iters=5)
                lip = lip + self.factor_penalty * s_max2
            step = 1.0 / lip.clamp_min(1e-6)

            w = _project_dollar_neutral_box_l1_torch(w0, self.gross_limit, cap)
            y_acc = w.clone()
            t_fista = 1.0
            step_l1 = step * c_vec
            step_imp = step * e_vec

            for _ in range(self.max_iters):
                grad = -m + self.risk_aversion * (var_diag * y_acc)
                if b_mat is not None:
                    grad = grad + self.factor_penalty * (b_mat @ (b_mat.T @ y_acc))
                v = y_acc - step * grad
                # Operador proximal exacto L1 + impacto^(3/2) 100 % en GPU
                d = v - w0
                r = torch.relu(d.abs() - step_l1)
                b = 1.5 * step_imp
                u = 0.5 * (-b + torch.sqrt(b * b + 4.0 * r))
                w_prox = w0 + torch.sign(d) * (u * u)
                w_next = _project_dollar_neutral_box_l1_torch(w_prox, self.gross_limit, cap)
                t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_fista * t_fista))
                y_acc = w_next + ((t_fista - 1.0) / t_next) * (w_next - w)
                w = w_next
                t_fista = t_next
            return w.to(dtype=orig_dtype)


class DifferentiablePortfolioLayer(nn.Module):
    """Capa convexa diferenciable `P+` desenrollada en PyTorch (BF16 AMP y CUDA Graph safe)."""

    def __init__(
        self,
        *,
        steps: int = 10,
        risk_aversion: float = 2.0,
        factor_penalty: float = 10.0,
        l1_cost: float = 0.001,
        gross_limit: float = 1.0,
        pos_cap: float = 0.1,
        holding_days: float = 5.0,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.risk_aversion = risk_aversion
        self.factor_penalty = factor_penalty
        self.l1_cost = l1_cost
        self.gross_limit = gross_limit
        self.pos_cap = pos_cap
        self.holding_days = holding_days

    def forward(
        self,
        mu: torch.Tensor,
        *,
        vol: torch.Tensor | None = None,
        exposures: torch.Tensor | None = None,
        w_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mapea `mu [N]` a pesos neutrales `w [N]` con `|w_i| <= pos_cap` exacto y 0 syncs."""
        n = mu.shape[0]
        device, orig_dtype = mu.device, mu.dtype
        if n < 2:
            return torch.zeros_like(mu)

        calc_dtype = (
            torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
        )
        with torch.amp.autocast(device_type=device.type, enabled=False):
            mu_c = mu.to(dtype=calc_dtype)
            var_diag = (
                torch.full((n,), 0.02**2, device=device, dtype=calc_dtype)
                if vol is None
                else torch.clamp(vol.to(device=device, dtype=calc_dtype), min=1e-4).pow(2)
            )
            w = (
                torch.zeros(n, device=device, dtype=calc_dtype)
                if w_prev is None
                else w_prev.to(device=device, dtype=calc_dtype).clone()
            )
            w0 = w.detach()

            b_mat = None
            lip = self.risk_aversion * var_diag.max().detach().clamp_min(1e-8)
            if exposures is not None and self.factor_penalty > 0.0 and exposures.shape[-1] > 0:
                exp_c = torch.nan_to_num(
                    exposures.to(device=device, dtype=calc_dtype),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                if exp_c.ndim == 1:
                    exp_c = exp_c.unsqueeze(-1)
                # Columnas centradas de norma L2 unitaria (equiv. 1/N * B_z @ B_z^T -> lip O(1))
                b_c = exp_c - exp_c.mean(dim=0, keepdim=True)
                col_norm = torch.sqrt((b_c * b_c).sum(dim=0, keepdim=True) + 1e-12)
                b_mat = torch.where(col_norm > 1e-6, b_c / col_norm, torch.zeros_like(b_c))
                gram = b_mat.detach().T @ b_mat.detach()
                s_max2 = _gram_max_eig_power_iter(gram, n_iters=5)
                lip = lip + self.factor_penalty * s_max2
            step = 1.0 / lip.clamp_min(1e-4)
            eff_l1 = self.l1_cost / max(float(self.holding_days), 1.0)
            aux_req_grad = (
                var_diag.requires_grad
                or (b_mat is not None and b_mat.requires_grad)
                or w.requires_grad
            )
            if not aux_req_grad:
                b_arg = (
                    b_mat
                    if b_mat is not None
                    else torch.empty((n, 0), device=device, dtype=calc_dtype)
                )
                w_out, _, _, _, _ = _diff_portfolio_fista_fwd_op(
                    mu_c,
                    var_diag,
                    w0,
                    b_arg,
                    step,
                    int(self.steps),
                    float(self.risk_aversion),
                    float(self.factor_penalty),
                    float(eff_l1),
                    float(self.pos_cap),
                    float(self.gross_limit),
                )
                return w_out.to(dtype=orig_dtype)

            thr = step * eff_l1
            y_acc = w
            t_fista = 1.0
            for _ in range(self.steps):
                grad = -mu_c + self.risk_aversion * (var_diag * y_acc)
                if b_mat is not None:
                    grad = grad + self.factor_penalty * (b_mat @ (b_mat.T @ y_acc))
                v = y_acc - step * grad
                # Banda suave diferenciable alrededor de w0 (soft-thresholding amortizado)
                d = v - w0
                w_raw = w0 + torch.sign(d) * torch.relu(torch.abs(d) - thr)
                # Proyección diferenciable exacta 1-CTA Triton + derivada implícita (0 syncs)
                w_next = project_dollar_neutral_box_torch(
                    w_raw, pos_cap=self.pos_cap, gross_limit=self.gross_limit
                )
                t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_fista * t_fista))
                y_acc = w_next + ((t_fista - 1.0) / t_next) * (w_next - w)
                w = w_next
                t_fista = t_next
            return w.to(dtype=orig_dtype)


class MetacognitiveGovernor:
    """Gobernador Metacognitivo de Inferencia Activa (`A5–A8`).

    Combina incertidumbre epistémica (desacuerdo multi-especialista, anchura Venn-Abers, energía
    Hopfield OOD, entropía del router MoE y energía no resuelta del bucle), abstención racional
    escalada a la magnitud de la señal `fused` y detección de decaimiento de alfa CUSUM (A8).
    """

    def __init__(
        self,
        n_specialists: int,
        *,
        epistemic_kappa: float = 1.0,
        ood_threshold: float = 2.5,
        entropy_weight: float = 0.002,
        loop_energy_weight: float = 0.002,
        cusum_drift: float = 0.01,
        cusum_threshold: float = 0.25,
        ewma_halflife: float = 21.0,
        scale_by_signal: bool = True,
        signal_scale_ref: float = 0.02,
    ) -> None:
        self.n_specialists = n_specialists
        self.epistemic_kappa = epistemic_kappa
        self.ood_threshold = ood_threshold
        self.entropy_weight = entropy_weight
        self.loop_energy_weight = loop_energy_weight
        self.cusum_drift = cusum_drift
        self.cusum_threshold = cusum_threshold
        self.decay = math.pow(0.5, 1.0 / max(ewma_halflife, 1.0))
        self.scale_by_signal = scale_by_signal
        self.signal_scale_ref = max(float(signal_scale_ref), 1e-6)

        self._weights_tensor_cache: dict[torch.device, torch.Tensor] = {}
        self._specialist_weights = np.full(
            n_specialists, 1.0 / n_specialists, dtype=np.float64
        )
        self.ewma_ic = np.zeros(n_specialists, dtype=np.float64)
        self.peak_ic = np.zeros(n_specialists, dtype=np.float64)
        self.cusum_stat = np.zeros(n_specialists, dtype=np.float64)
        self.decay_alerts = np.zeros(n_specialists, dtype=bool)
        self.history_updates = 0

    @property
    def specialist_weights(self) -> np.ndarray:
        return self._specialist_weights

    @specialist_weights.setter
    def specialist_weights(self, value: np.ndarray) -> None:
        self._specialist_weights = np.asarray(value, dtype=np.float64)
        self._weights_tensor_cache.clear()

    def decide(
        self,
        *,
        specialist_preds: np.ndarray,
        va_widths: np.ndarray | None = None,
        hopfield_energy: np.ndarray | None = None,
        router_entropy: np.ndarray | None = None,
        loop_energy: np.ndarray | None = None,
        estimated_costs: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Emite señal fusionada, incertidumbre epistémica y máscara de abstención racional (A5)."""
        preds = np.asarray(specialist_preds, dtype=np.float64)
        n = preds.shape[0]
        w = self._specialist_weights
        fused = preds @ w

        # Incertidumbre epistémica = desacuerdo + señales adimensionales escaladas a `fused`
        disagreement = np.sqrt(np.sum(w[None, :] * (preds - fused[:, None]) ** 2, axis=1))
        z0 = np.zeros(n, dtype=np.float64)
        va_w = z0 if va_widths is None else np.asarray(va_widths, dtype=np.float64)
        hop_e = z0 if hopfield_energy is None else np.asarray(hopfield_energy, dtype=np.float64)
        r_ent = z0 if router_entropy is None else np.asarray(router_entropy, dtype=np.float64)
        l_eng = z0 if loop_energy is None else np.asarray(loop_energy, dtype=np.float64)
        costs = z0 if estimated_costs is None else np.asarray(estimated_costs, dtype=np.float64)

        if self.scale_by_signal:
            sig_rms = math.sqrt(max(float(np.mean(fused * fused)), 1e-12)) if n > 0 else 0.0
            rel_scale = min(1.0, sig_rms / self.signal_scale_ref)
        else:
            rel_scale = 1.0

        epistemic_unc = (
            disagreement
            + (0.02 * rel_scale) * va_w
            + (self.entropy_weight * rel_scale) * r_ent
            + (self.loop_energy_weight * rel_scale) * l_eng
        )
        # Regla de Energía Libre: |mu| - kappa*sigma_epist > coste y OOD <= umbral
        net_edge = np.abs(fused) - self.epistemic_kappa * epistemic_unc - costs
        active = (net_edge > 0.0) & (hop_e <= self.ood_threshold)
        gated = np.where(active, fused, 0.0)
        return dict(
            fused_signal=fused,
            epistemic_uncertainty=epistemic_unc,
            net_edge=net_edge,
            active_mask=active,
            gated_signal=gated,
        )

    @torch.no_grad()
    def decide_torch(
        self,
        *,
        specialist_preds: torch.Tensor,
        va_widths: torch.Tensor | None = None,
        hopfield_energy: torch.Tensor | None = None,
        router_entropy: torch.Tensor | None = None,
        loop_energy: torch.Tensor | None = None,
        estimated_costs: torch.Tensor | None = None,
        specialist_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Versión 100 % nativa GPU/PyTorch de A5 con caché de pesos en GPU (cero H2D syncs)."""
        preds = specialist_preds.float()
        n = preds.shape[0]
        device = preds.device
        f32 = torch.float32
        if specialist_weights is not None:
            w = specialist_weights.to(device=device, dtype=f32)
        else:
            w_cached = self._weights_tensor_cache.get(device)
            if w_cached is None:
                w_cached = torch.as_tensor(self._specialist_weights, device=device, dtype=f32)
                self._weights_tensor_cache[device] = w_cached
            w = w_cached

        fused = preds @ w
        disagreement = torch.sqrt(
            (w.unsqueeze(0) * (preds - fused.unsqueeze(-1)).pow(2)).sum(dim=-1).clamp_min(0.0)
        )
        z0 = torch.zeros(n, device=device, dtype=f32)
        va_w = z0 if va_widths is None else va_widths.to(device=device, dtype=f32)
        hop_e = z0 if hopfield_energy is None else hopfield_energy.to(device=device, dtype=f32)
        r_ent = z0 if router_entropy is None else router_entropy.to(device=device, dtype=f32)
        l_eng = z0 if loop_energy is None else loop_energy.to(device=device, dtype=f32)
        costs = z0 if estimated_costs is None else estimated_costs.to(device=device, dtype=f32)

        if self.scale_by_signal:
            sig_rms = torch.sqrt((fused * fused).mean().clamp_min(1e-12))
            rel_scale: torch.Tensor | float = torch.clamp(
                sig_rms / self.signal_scale_ref, max=1.0
            )
        else:
            rel_scale = 1.0

        epistemic_unc = (
            disagreement
            + (0.02 * rel_scale) * va_w
            + (self.entropy_weight * rel_scale) * r_ent
            + (self.loop_energy_weight * rel_scale) * l_eng
        )
        net_edge = fused.abs() - self.epistemic_kappa * epistemic_unc - costs
        active = (net_edge > 0.0) & (hop_e <= self.ood_threshold)
        gated = torch.where(active, fused, torch.zeros_like(fused))
        return dict(
            fused_signal=fused,
            epistemic_uncertainty=epistemic_unc,
            net_edge=net_edge,
            active_mask=active,
            gated_signal=gated,
        )

    def update_matured(self, specialist_ics: np.ndarray) -> None:
        """Actualiza pesos causales y detector CUSUM (A8) al madurar los IC de cada especialista."""
        ics = np.asarray(specialist_ics, dtype=np.float64)
        self.history_updates += 1
        if self.history_updates == 1:
            self.ewma_ic = ics.copy()
            self.peak_ic = np.maximum(ics, 0.0)
        else:
            self.ewma_ic = self.decay * self.ewma_ic + (1.0 - self.decay) * ics
            self.peak_ic = np.maximum(self.peak_ic * 0.995, self.ewma_ic)

        # Page-Hinkley / CUSUM unilateral para detectar caída persistente de IC
        drop = (self.peak_ic - ics) - self.cusum_drift
        self.cusum_stat = np.maximum(0.0, self.cusum_stat + drop)
        self.decay_alerts = self.cusum_stat >= self.cusum_threshold

        # Pesos positivos suaves penalizando especialistas en alerta CUSUM
        scores = np.maximum(self.ewma_ic, 0.0) + 1e-4
        scores = np.where(self.decay_alerts, scores * 0.15, scores)
        self.specialist_weights = scores / float(np.sum(scores))


def run_agent_variation(
    *,
    variant: str,
    specialist_preds: np.ndarray,
    target: np.ndarray,
    bounds: list[tuple[int, int]],
    volatility: np.ndarray | None = None,
    exposures: np.ndarray | None = None,
    regime_signal: np.ndarray | None = None,
    va_widths: np.ndarray | None = None,
    hopfield_energy: np.ndarray | None = None,
    router_entropy: np.ndarray | None = None,
    loop_energy: np.ndarray | None = None,
    asset_ids: np.ndarray | None = None,
    l1_cost: float = 0.001,
    factor_penalty: float = 20.0,
    holding_days: float = 5.0,
    horizon_sessions: int = 1,
) -> dict[str, object]:
    """Ejecuta y compara limpiamente una variación del Núcleo Agéntico con cola causal madura."""
    if variant not in AGENT_VARIATIONS:
        raise ValueError(
            f"Variación agéntica desconocida: {variant!r}; válidas: {AGENT_VARIATIONS}"
        )
    h_sess = int(horizon_sessions)
    if h_sess < 1:
        raise ValueError(f"horizon_sessions debe ser >= 1; recibido {horizon_sessions}")

    preds = np.asarray(specialist_preds, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    n, k_spec = preds.shape
    aids_all = None if asset_ids is None else np.asarray(asset_ids)
    vol = (
        np.full(n, 0.02, dtype=np.float64)
        if volatility is None
        else np.asarray(volatility, dtype=np.float64)
    )
    reg = (
        np.zeros(len(bounds), dtype=np.float64)
        if regime_signal is None
        else np.asarray(regime_signal, dtype=np.float64)
    )

    weights = np.zeros(n, dtype=np.float64)
    active_rates: list[float] = []
    gov = MetacognitiveGovernor(k_spec, epistemic_kappa=0.5, ood_threshold=3.0)
    h_days = max(float(holding_days), 1.0)
    opt = ConvexPortfolioOptimizerPPlus(
        risk_aversion=2.0,
        factor_penalty=factor_penalty,
        l1_cost=l1_cost,
        impact_cost=0.01,
        gross_limit=1.0,
        holding_days=h_days,
    )

    w_prev = None
    prev_w_by_id: dict[Any, float] = {}
    pending_ics: list[tuple[int, np.ndarray]] = []
    gross_list, net1_list, to_list, exited_to_list = [], [], [], []
    for s_idx, (lo, hi) in enumerate(bounds):
        # Cola causal estricta: solo madurar sesiones con matures_at <= s_idx (seguro para h > 1)
        if pending_ics:
            ready = [item for item in pending_ics if item[0] <= s_idx]
            if ready:
                pending_ics = [item for item in pending_ics if item[0] > s_idx]
                ready.sort(key=lambda item: item[0])
                for _, ics_m in ready:
                    gov.update_matured(ics_m)

        p_seg = preds[lo:hi]
        y_seg = y[lo:hi]
        v_seg = vol[lo:hi]
        b_seg = None if exposures is None else exposures[lo:hi]
        if aids_all is not None:
            ids_seg = aids_all[lo:hi]
            w_prev_aligned, exited_to = align_weights_by_asset_ids(prev_w_by_id, ids_seg)
            w_prev_opt: np.ndarray | None = w_prev_aligned if prev_w_by_id else None
        else:
            ids_seg = None
            same_prev = w_prev is not None and len(w_prev) == (hi - lo)
            w_prev_aligned = w_prev if same_prev else np.zeros(hi - lo, dtype=np.float64)
            w_prev_opt = w_prev if same_prev else None
            exited_to = 0.0

        if variant == "agent_none":
            sig = np.mean(p_seg, axis=1)
            z = sig - float(np.mean(sig))
            s = float(np.sum(np.abs(z)))
            wt = z / s if s > 1e-12 else np.zeros_like(z)
            active_rates.append(1.0)
        elif variant == "agent_p_plus_only":
            sig = np.mean(p_seg, axis=1)
            wt = opt.solve(
                sig,
                vol=v_seg,
                exposures=b_seg,
                w_prev=w_prev_opt,
            )
            active_rates.append(1.0)
        elif variant == "agent_moe_routing_p_plus":
            # Enrutamiento dinámico de especialistas guiado por rendimiento causal maduro + régimen
            scale_reg = 1.0 + 0.5 * float(np.clip(reg[s_idx], -2.0, 2.0))
            opt.risk_aversion = max(0.5, 2.0 * scale_reg)
            sig = p_seg @ gov.specialist_weights
            wt = opt.solve(
                sig,
                vol=v_seg,
                exposures=b_seg,
                w_prev=w_prev_opt,
            )
            active_rates.append(1.0)
        else:  # agent_full_active_inference
            scale_reg = 1.0 + 0.5 * float(np.clip(reg[s_idx], -2.0, 2.0))
            opt.risk_aversion = max(0.5, 2.0 * scale_reg)
            dec = gov.decide(
                specialist_preds=p_seg,
                va_widths=None if va_widths is None else va_widths[lo:hi],
                hopfield_energy=None if hopfield_energy is None else hopfield_energy[lo:hi],
                router_entropy=None if router_entropy is None else router_entropy[lo:hi],
                loop_energy=None if loop_energy is None else loop_energy[lo:hi],
                estimated_costs=np.full(hi - lo, l1_cost / h_days),
            )
            active_rates.append(float(dec["active_mask"].mean()))
            wt = opt.solve(
                dec["gated_signal"],
                vol=v_seg,
                exposures=b_seg,
                w_prev=w_prev_opt,
            )

        weights[lo:hi] = wt
        g = float(np.dot(wt, y_seg))
        to = float(np.sum(np.abs(wt - w_prev_aligned))) + exited_to
        gross_list.append(g)
        to_list.append(to)
        exited_to_list.append(exited_to)
        net1_list.append(g - l1_cost * to)
        w_prev = wt
        if ids_seg is not None:
            curr_keys = [
                x.item() if isinstance(x, np.generic) else x for x in ids_seg.tolist()
            ]
            prev_w_by_id = {aid: float(w_val) for aid, w_val in zip(curr_keys, wt, strict=True)}

        # Encolar ICs para madurar estrictamente en `s_idx + horizon_sessions`
        if hi - lo >= 5:
            ics = np.array(
                [
                    float(np.corrcoef(p_seg[:, j], y_seg)[0, 1])
                    if float(np.std(p_seg[:, j])) > 1e-12 and float(np.std(y_seg)) > 1e-12
                    else 0.0
                    for j in range(k_spec)
                ],
                dtype=np.float64,
            )
            pending_ics.append((s_idx + h_sess, ics))

    g_arr = np.asarray(gross_list, dtype=np.float64)
    n1_arr = np.asarray(net1_list, dtype=np.float64)
    to_arr = np.asarray(to_list, dtype=np.float64)
    ex_arr = np.asarray(exited_to_list, dtype=np.float64)
    sd_n1 = float(np.std(n1_arr, ddof=1)) if len(n1_arr) > 1 else 0.0
    sd_g = float(np.std(g_arr, ddof=1)) if len(g_arr) > 1 else 0.0
    eq = np.cumsum(n1_arr)
    dd = np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq
    g_sr = float(np.mean(g_arr) / sd_g * math.sqrt(252.0)) if sd_g > 1e-12 else 0.0
    n1_sr = float(np.mean(n1_arr) / sd_n1 * math.sqrt(252.0)) if sd_n1 > 1e-12 else 0.0
    return dict(
        variant=variant,
        weights=weights,
        summary=dict(
            gross_bps=float(np.mean(g_arr)) * 1e4,
            gross_sharpe=g_sr,
            net_sharpe_1x=n1_sr,
            turnover=float(np.mean(to_arr)),
            exited_turnover_mean=float(np.mean(ex_arr)),
            max_drawdown=float(dd.max()) if len(dd) else 0.0,
            active_rate=float(np.mean(active_rates)) if active_rates else 1.0,
            cusum_alerts=int(gov.decay_alerts.sum()),
            matured_updates=int(gov.history_updates),
        ),
    )
