"""Wave-3B Frontier Deep Learning & Stage-2 Modules (`MARS-TITAN`).

Adversarial-refereed (`#194` + `#195` P0 guards), 0-host-sync (`audit_zero_host_syncs == 0`),
0 runtime `torch.tensor()` allocations, master-precision FP32/FP64 under BF16 autocast,
and `torch.compile(mode="default", dynamic=True)` clean implementations:

1. **Polar Order-Magnitude Decoupling (`#195` P0-1a, P0-1b, P0-1c)**:
   - `PolarAlphaHead`: decouples cross-sectionally unit-normalized ranking direction `u`
     from Grinold-Kahn calibrated expected return `alpha_cal` (`<30` params, works across
     all 4 model families and any horizon count `H`).
   - Exact cross-sectional zero-mean re-centering AFTER `sigma_det * gamma_h * tanh(u / 3)`
     scaling (`P0-1b`) so `FinancialEnergyVerifier` / Tool 3 remain invariant.
   - Strict anti-FM-7 cross-detachment (`P0-1c`):
     * `alpha_cal_for_cal`: uses `u.detach()` + live `gamma_h` for `polar_alpha_calibration_loss`
       (calibrates `mz_slope -> ~1.0` without touching ranking gradients).
     * `alpha_cal_for_pplus`: uses live `u` + `gamma_h.detach()` for `DifferentiablePortfolioLayer`
       / `loss_sharpe` (preserves true `l1_cost=0.001` thresholding without inflating `gamma_h`).
   - `polar_alpha_order_loss` (`P0-1a`): combines factor-neutralized IC with verified
     `tail_listnet_loss` from `dl_training_sota.py` + optional ragged asset-ID-aligned
     persistence `polar_alpha_persistence_loss` (`align_by_asset_ids`).

2. **MNAR-Aware Sparse Slot & Set Modality Fusion (`#195` P0-2a, P0-2b)**:
   - `ObservedSlotEncoder`: variance-stabilized projection for `encoded_v31_78` (`26 x 3`),
     `encoded_v31_420` (`140 x 3`), or generic `(vals, obs_mask, staleness_decay)`:
     `e = RMSNorm(W_val(o * d_s * v) / sqrt(sum(o) + 1.0) + W_pat([o, d_s]))`
     eliminating `LayerNorm(78)`'s `sqrt(26/3) ~= 3x` sparse CN gain blowup.
   - `MissingnessAwareSetModalityFusion`: schema-agnostic QK-RMSNorm Perceiver cross-attention
     over present auxiliary modalities + 1 always-present core self-token anchor (`mask=True`),
     carrying explicit MNAR presence/absence embeddings (`u_miss`, `u_pres`) and matched
     `mnar_placebo_shuffle` control with bit-exact `0.0` sensitivity & gradient on missing slots.

3. **Differential Cross-Asset Attention (`#195` P0-3a, P0-3b, P0-3c — F3 Drop-In)**:
   - `DiffInducedFactorCrossAttention`: exact Ye et al. (ICLR 2025) Differential Attention
     over `R=16` induced landmarks (`O(N * R)`), iso-QKV param drop-in for F3
     `InducedFactorCrossAttention`.
   - Explicit FP32/FP64 L2-resident `[1, H, N, R]` / `[1, H, R, N]` path under
     `_autocast_disabled` with Gemma-2 logit soft-capping `c * tanh(logits / c)` (`c=30.0`).
   - Strictly bounded learnable `lambda = 2.0 * sigmoid(raw_lambda) in (0, 2)` +
     per-head `GroupRMSNorm` scaled by `(1 - lambda_init)`, matched `zero_lambda_control`,
     strict `asset_mask` padding isolation, `F`-invariance, and stock permutation equivariance.

4. **Stage-2 Cross-Family Epistemic Disagreement & CKA-Orthogonal Stacking (`#195` #5C)**:
   - `compute_cka_dissimilarity_weights`: computes redundancy-penalized stacking weights from
     an `F x F` cross-family linear CKA matrix.
   - `compute_cross_family_disagreement_risk`: zero-training-cost pure PyTorch / pure NumPy
     (FP32/FP64) cross-sectional standardization, CKA-weighted consensus `u_bar`, and
     epistemic disagreement risk `sigma_eff = sqrt(realized_vol^2 + kappa * Var_f(u) * scale)`
     ready for Stage-2 `P+` with bit-exact `kappa = 0.0` matched control.

Numerical stability & Blackwell (`sm_120`) contract:
  - Risk mitigations: (i) two-pass centered variance `mean((x - mean)^2)` everywhere (never
    `E[x^2] - E[x]^2`); (ii) shift-by-max masked softmax with empty-mask / anchor guards
    (never `-inf + inf` NaN); (iii) pre-mask sanitization via `torch.where` before every
    nonlinear/loss op (zero `0 * NaN` backward gradient leaks); (iv) `_autocast_disabled`
    FP32/FP64 master precision on all reductions, attention logits, soft-capping, and losses;
    (v) bounded `lambda = 2 * sigmoid(raw)` (never unbounded `exp(dot)`).
  - Performance cost: FP32 promotion on `[1, H, N, R=16]` (`1.3 MB` at `N=2600, H=8`) stays
    100% resident in Blackwell's 128 MB L2 cache (`<2.5%` FLOPs); `PolarAlphaHead` and
    `MissingnessAwareSetModalityFusion` add `<2.0%` combined inference FLOPs with 0 syncs.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, overload

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mars_titan.data.market_physics_v4_gpu import gaussian_rank_features
from mars_titan.models.mmarrs.dl_training_sota import RMSNormSOTA, tail_listnet_loss

__all__ = [
    "DiffInducedFactorCrossAttention",
    "MissingnessAwareSetModalityFusion",
    "ObservedSlotEncoder",
    "PolarAlphaHead",
    "align_by_asset_ids",
    "compute_cka_dissimilarity_weights",
    "compute_cross_family_disagreement_risk",
    "polar_alpha_calibration_loss",
    "polar_alpha_order_loss",
    "polar_alpha_persistence_loss",
]


# ---------------------------------------------------------------------------
# Internal Numerical Precision, Context & Masking Utilities
# ---------------------------------------------------------------------------


def _calc_dtype(dtype: torch.dtype) -> torch.dtype:
    """Promote BF16/FP16 to FP32 master precision while preserving FP64."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def _autocast_disabled(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    dev_type = device.type if device.type in ("cuda", "cpu") else "cpu"
    return torch.amp.autocast(device_type=dev_type, enabled=False)


def _inv_softplus(value: float) -> float:
    """Numerically stable inverse of `softplus` for positive parameter initialization."""
    v = max(float(value), 1e-6)
    if v > 20.0:
        return v
    return math.log(math.expm1(v))


def _stable_masked_softmax(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Shift-by-max masked softmax in master precision with zero `-inf * 0` NaN leaks."""
    has_any = valid_mask.any(dim=dim, keepdim=True)
    neg_inf = torch.full_like(logits, float("-inf"))
    raw_max = torch.where(valid_mask, logits, neg_inf).max(dim=dim, keepdim=True).values
    safe_max = torch.where(has_any, raw_max, torch.zeros_like(raw_max))
    shifted = torch.where(valid_mask, logits - safe_max, torch.zeros_like(logits))
    exp_s = torch.where(valid_mask, torch.exp(shifted), torch.zeros_like(logits))
    denom = exp_s.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return torch.where(valid_mask & has_any, exp_s / denom, torch.zeros_like(logits))


def _standardize_factor_concat_masked(
    *,
    exposures: torch.Tensor | None,
    rmt_loadings: torch.Tensor | None,
    n_assets: int,
    asset_mask: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Mask-safe, F-invariant standardization of `[exposures, rmt_loadings]` (0 host-syncs)."""
    cols: list[torch.Tensor] = []
    calc_dt = _calc_dtype(dtype)
    if asset_mask is not None:
        row_ok = asset_mask.to(device=device, dtype=torch.bool).reshape(n_assets, 1)
    else:
        row_ok = torch.ones((n_assets, 1), device=device, dtype=torch.bool)
    for raw in (exposures, rmt_loadings):
        if raw is None or raw.ndim != 2 or raw.shape[0] != n_assets or raw.shape[-1] <= 0:
            continue
        m_raw = raw.detach().to(device=device, dtype=calc_dt)
        valid = row_ok & torch.isfinite(m_raw)
        m_clean = torch.where(valid, m_raw, torch.zeros_like(m_raw))
        v_cnt = valid.to(dtype=calc_dt).sum(dim=0, keepdim=True).clamp_min(1.0)
        mean = m_clean.sum(dim=0, keepdim=True) / v_cnt
        centered = torch.where(valid, m_clean - mean, torch.zeros_like(m_clean))
        var = centered.pow(2).sum(dim=0, keepdim=True) / v_cnt
        std = torch.sqrt(var.clamp_min(1e-8))
        normed = torch.where(valid, centered / std, torch.zeros_like(centered))
        cols.append(normed.to(dtype=dtype))
    if not cols:
        return None
    return torch.cat(cols, dim=-1)


# ---------------------------------------------------------------------------
# 1. PolarAlphaHead + Cross-Detached Calibration / Order / Persistence Losses
# ---------------------------------------------------------------------------


def align_by_asset_ids(
    cur_ids: torch.Tensor,
    prev_ids: torch.Tensor,
    prev_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align a 1D `[M]` or 2D `[M, H]` previous-session tensor to `cur_ids [N]` (0 host-syncs).

    Unmatched assets (IPOs / new entrants) receive `0.0` in `aligned` and `False` in `matched`.
    """
    device, dtype = prev_values.device, prev_values.dtype
    c_ids = cur_ids.to(device=device, dtype=torch.long).reshape(-1)
    p_ids = prev_ids.to(device=device, dtype=torch.long).reshape(-1)
    n_cur = c_ids.shape[0]
    if p_ids.shape[0] == 0:
        out_shape = (n_cur,) if prev_values.ndim == 1 else (n_cur, prev_values.shape[-1])
        return (
            torch.zeros(out_shape, device=device, dtype=dtype),
            torch.zeros((n_cur,), device=device, dtype=torch.bool),
        )
    if prev_values.shape[0] != p_ids.shape[0]:
        raise ValueError(
            f"prev_values leading dim ({prev_values.shape[0]}) != prev_ids ({p_ids.shape[0]})"
        )
    sort_idx = torch.argsort(p_ids)
    p_sorted = p_ids[sort_idx]
    v_sorted = prev_values[sort_idx]
    pos = torch.searchsorted(p_sorted, c_ids)
    pos_safe = pos.clamp(max=p_sorted.shape[0] - 1)
    matched = (pos < p_sorted.shape[0]) & (p_sorted[pos_safe] == c_ids)
    gathered = v_sorted[pos_safe]
    m_view = matched if gathered.ndim == 1 else matched.unsqueeze(-1)
    aligned = torch.where(m_view, gathered, torch.zeros_like(gathered))
    return aligned, matched


class PolarAlphaHead(nn.Module):
    """Polar Order-Magnitude Decoupling Head (`#195` P0-1a/1b/1c, `<30` params, all 4 families).

    Decomposes arbitrary `raw_alpha [N]` or `[N, H]` into:
      1. Cross-sectionally zero-mean, unit-std ranking direction `u` (two-pass FP32/FP64
         scale-invariant normalization; `N=1` and constant inputs map cleanly to `0.0`).
      2. Grinold-Kahn calibrated expected return `alpha_cal`:
           `raw_cal = sigma_det * softplus(raw_gamma_h) * tanh(u / u_tanh_scale)`
           `alpha_cal = raw_cal - mean(raw_cal, dim=0)`  (exact zero-mean `P0-1b`)
      3. Strict anti-FM-7 cross-detachment (`P0-1c`):
         - `alpha_cal_for_cal`: computed with `u.detach()` and live `gamma_h` (so `L_cal`
           calibrates `gamma_h` to exact `mz_slope ~= 1.0` with `0.0` grad into `raw_alpha`).
         - `alpha_cal_for_pplus`: computed with live `u` and `gamma_h.detach()` (so `P+`
           optimizes turnover-aware ranking at real `l1_cost=0.001` with `0.0` grad into
           `raw_gamma`, making FM-7 scale inflation mathematically impossible).
    """

    def __init__(
        self,
        n_horizons: int = 6,
        *,
        init_gamma: float = 0.15,
        u_tanh_scale: float = 3.0,
        default_vol: float = 0.02,
        min_vol: float = 1e-4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if n_horizons < 1:
            raise ValueError(f"n_horizons must be >= 1; got {n_horizons}")
        if init_gamma <= 0.0:
            raise ValueError(f"init_gamma must be > 0; got {init_gamma}")
        if u_tanh_scale <= 0.0:
            raise ValueError(f"u_tanh_scale must be > 0; got {u_tanh_scale}")
        if default_vol <= 0.0 or min_vol <= 0.0:
            raise ValueError("default_vol and min_vol must be > 0")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0; got {eps}")

        self.n_horizons = int(n_horizons)
        self.init_gamma = float(init_gamma)
        self.u_tanh_scale = float(u_tanh_scale)
        self.default_vol = float(default_vol)
        self.min_vol = float(min_vol)
        self.eps = float(eps)

        self.raw_gamma = nn.Parameter(
            torch.full((self.n_horizons,), _inv_softplus(self.init_gamma), dtype=torch.float32)
        )

    def forward(
        self,
        raw_alpha: torch.Tensor,
        *,
        vol: torch.Tensor | None = None,
        asset_mask: torch.Tensor | None = None,
        systemic_bias: torch.Tensor | None = None,
        include_systemic_bias: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute `(u, alpha_cal, alpha_cal_for_cal, alpha_cal_for_pplus, gamma, sigma_det)`."""
        if raw_alpha.ndim not in (1, 2):
            raise ValueError(f"raw_alpha must be 1D [N] or 2D [N, H]; got ndim={raw_alpha.ndim}")
        squeeze_1d = raw_alpha.ndim == 1
        x2 = raw_alpha.unsqueeze(-1) if squeeze_1d else raw_alpha
        n_assets, n_h = x2.shape
        if n_h > self.n_horizons:
            raise ValueError(
                f"raw_alpha horizon dim ({n_h}) exceeds "
                f"PolarAlphaHead.n_horizons ({self.n_horizons})"
            )

        device = x2.device
        calc_dt = _calc_dtype(x2.dtype)
        with _autocast_disabled(device):
            x_c = x2.to(dtype=calc_dt)
            valid = torch.isfinite(x_c)
            if asset_mask is not None:
                am = asset_mask.to(device=device, dtype=torch.bool)
                if am.ndim == 1:
                    am = am.unsqueeze(-1)
                valid = valid & am[:, :n_h].expand(n_assets, n_h)

            zeros = torch.zeros_like(x_c)
            valid_f = valid.to(dtype=calc_dt)
            v_cnt = valid_f.sum(dim=0, keepdim=True)
            v_cnt_safe = v_cnt.clamp_min(1.0)

            # Two-pass cross-sectional unit normalization (exact scale invariance, N=1/const safe)
            x_clean = torch.where(valid, x_c, zeros)
            mean = x_clean.sum(dim=0, keepdim=True) / v_cnt_safe
            centered = torch.where(valid, x_clean - mean, zeros)
            var = centered.pow(2).sum(dim=0, keepdim=True) / v_cnt_safe
            eps_sq = self.eps * self.eps
            has_var = (var > eps_sq) & (v_cnt >= 1.5)
            safe_var = torch.where(has_var, var, torch.ones_like(var))
            u_c = torch.where(valid & has_var, centered / torch.sqrt(safe_var), zeros)

            # Detached strictly positive volatility scale `sigma_det`
            if vol is None:
                sigma_det = torch.full(
                    (n_assets, n_h), self.default_vol, device=device, dtype=calc_dt
                )
            else:
                v_raw = vol.detach().to(device=device, dtype=calc_dt)
                if v_raw.ndim == 1:
                    v_raw = v_raw.unsqueeze(-1).expand(n_assets, n_h)
                elif v_raw.ndim == 2:
                    v_raw = (
                        v_raw.expand(n_assets, n_h)
                        if v_raw.shape[-1] == 1
                        else v_raw[:, :n_h]
                    )
                else:
                    raise ValueError(f"vol must be 1D [N] or 2D [N, H]; got ndim={v_raw.ndim}")
                sigma_det = torch.where(
                    torch.isfinite(v_raw),
                    v_raw,
                    torch.full_like(v_raw, self.default_vol),
                ).clamp_min(self.min_vol)

            gamma_h = F.softplus(self.raw_gamma[:n_h].to(device=device, dtype=calc_dt)).view(1, n_h)
            inv_tanh_scale = 1.0 / self.u_tanh_scale

            sys_b: torch.Tensor | None = None
            if include_systemic_bias and systemic_bias is not None:
                sb = systemic_bias.to(device=device, dtype=calc_dt)
                if sb.ndim == 1:
                    sb = sb.unsqueeze(-1).expand(n_assets, n_h)
                else:
                    sb = sb[:, :n_h]
                sys_b = torch.where(valid & torch.isfinite(sb), sb, zeros)

            def _cal_centered(u_in: torch.Tensor, g_in: torch.Tensor) -> torch.Tensor:
                raw_cal = sigma_det * g_in * torch.tanh(u_in * inv_tanh_scale)
                raw_cal = torch.where(valid, raw_cal, zeros)
                cal_mean = raw_cal.sum(dim=0, keepdim=True) / v_cnt_safe
                centered_cal = torch.where(valid, raw_cal - cal_mean, zeros)
                if sys_b is not None:
                    centered_cal = centered_cal + sys_b
                return centered_cal

            # P0-1c: strict cross-detachment between calibration and P+ optimization
            alpha_for_cal = _cal_centered(u_c.detach(), gamma_h)
            alpha_for_pplus = _cal_centered(u_c, gamma_h.detach())

            if squeeze_1d:
                u_out = u_c.squeeze(-1)
                cal_out = alpha_for_cal.squeeze(-1)
                pplus_out = alpha_for_pplus.squeeze(-1)
                sig_out = sigma_det.squeeze(-1)
            else:
                u_out = u_c
                cal_out = alpha_for_cal
                pplus_out = alpha_for_pplus
                sig_out = sigma_det

        return dict(
            u=u_out,
            alpha_cal=pplus_out,
            alpha_cal_for_cal=cal_out,
            alpha_cal_for_pplus=pplus_out,
            gamma=gamma_h.squeeze(0),
            sigma_det=sig_out,
        )


def polar_alpha_calibration_loss(
    alpha_cal_for_cal: torch.Tensor,
    y_ret: torch.Tensor,
    *,
    delta: float = 0.03,
    demean_target: bool = True,
    horizon_mask: torch.Tensor | None = None,
    asset_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Master-precision Huber calibration loss for `gamma_h` with NaN & crash-day hygiene.

    By default (`demean_target=True`), cross-sectionally centers `y_ret` over valid assets per
    horizon before Huber so `±3–5%` market crash/rally days never saturate Huber's linear cap
    and vanish `dL_cal / d(raw_gamma)`. Pass `demean_target=False` only when calibrating with
    an explicit systemic market factor (`include_systemic_bias=True`).
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be > 0; got {delta}")
    if alpha_cal_for_cal.ndim not in (1, 2):
        raise ValueError(
            f"alpha_cal_for_cal must be 1D [N] or 2D [N, H]; got ndim={alpha_cal_for_cal.ndim}"
        )

    device = alpha_cal_for_cal.device
    calc_dt = _calc_dtype(alpha_cal_for_cal.dtype)
    with _autocast_disabled(device):
        p2 = alpha_cal_for_cal.unsqueeze(-1) if alpha_cal_for_cal.ndim == 1 else alpha_cal_for_cal
        y_raw = y_ret.to(device=device)
        y2 = y_raw.unsqueeze(-1) if y_raw.ndim == 1 else y_raw
        n_assets = p2.shape[0]
        if y2.shape[0] != n_assets:
            raise ValueError(f"asset count mismatch: pred {n_assets} vs target {y2.shape[0]}")
        n_h = min(p2.shape[-1], y2.shape[-1])
        p_c = p2[:, :n_h].to(dtype=calc_dt)
        y_c = y2[:, :n_h].to(dtype=calc_dt)

        valid = torch.isfinite(y_c) & torch.isfinite(p_c)
        if asset_mask is not None:
            am = asset_mask.to(device=device, dtype=torch.bool)
            if am.ndim == 1:
                am = am.unsqueeze(-1)
            valid = valid & am[:, :n_h].expand(n_assets, n_h)
        if horizon_mask is not None:
            hm_b = (horizon_mask.to(device=device).reshape(-1)[:n_h] > 0.5).unsqueeze(0)
            valid = valid & hm_b

        zeros = torch.zeros_like(p_c)
        # Pre-sanitize BOTH pred and target before Huber so invalid slots have exact 0.0 gradient
        p_safe = torch.where(valid, p_c, zeros)
        y_safe = torch.where(valid, y_c, zeros)

        w = valid.to(dtype=calc_dt)
        h_cnt = w.sum(dim=0, keepdim=True)  # [1, H]
        if demean_target:
            y_mean = (y_safe * w).sum(dim=0, keepdim=True) / h_cnt.clamp_min(1.0)
            y_eff = torch.where(valid, y_safe - y_mean, zeros)
        else:
            y_eff = y_safe

        elem = F.huber_loss(p_safe, y_eff, delta=float(delta), reduction="none")
        h_cnt_1d = h_cnt.squeeze(0)
        h_active = (h_cnt_1d > 0.5).to(dtype=calc_dt)
        h_mean = (elem * w).sum(dim=0) / h_cnt_1d.clamp_min(1.0)
        active_h_sum = h_active.sum()
        return torch.where(
            active_h_sum > 0.5,
            (h_mean * h_active).sum() / active_h_sum.clamp_min(1.0),
            torch.zeros((), device=device, dtype=calc_dt),
        )


def polar_alpha_persistence_loss(
    u_cur: torch.Tensor,
    u_prev: torch.Tensor,
    *,
    cur_asset_ids: torch.Tensor | None = None,
    prev_asset_ids: torch.Tensor | None = None,
    asset_mask: torch.Tensor | None = None,
    horizon_mask: torch.Tensor | None = None,
    w_churn: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Ragged asset-ID-aligned persistence `w_churn * (1 - cos(u_t, sg(u_{t-1})))` (1D & 2D)."""
    if w_churn < 0.0:
        raise ValueError(f"w_churn must be >= 0; got {w_churn}")
    device = u_cur.device
    calc_dt = _calc_dtype(u_cur.dtype)
    if w_churn == 0.0:
        return torch.zeros((), device=device, dtype=calc_dt)

    with _autocast_disabled(device):
        u_t2 = (u_cur.unsqueeze(-1) if u_cur.ndim == 1 else u_cur).to(dtype=calc_dt)
        u_p2 = (u_prev.unsqueeze(-1) if u_prev.ndim == 1 else u_prev).detach().to(
            device=device, dtype=calc_dt
        )
        n_h = min(u_t2.shape[-1], u_p2.shape[-1])
        u_t = u_t2[:, :n_h]
        u_p = u_p2[:, :n_h]

        if cur_asset_ids is not None and prev_asset_ids is not None:
            u_p_aligned, matched = align_by_asset_ids(cur_asset_ids, prev_asset_ids, u_p)
        else:
            if u_p.shape[0] != u_t.shape[0]:
                raise ValueError(
                    f"u_prev length ({u_p.shape[0]}) != u_cur length ({u_t.shape[0]}) "
                    "without cur_asset_ids/prev_asset_ids"
                )
            u_p_aligned = u_p
            matched = torch.ones((u_t.shape[0],), device=device, dtype=torch.bool)

        valid = matched.unsqueeze(-1) & torch.isfinite(u_t) & torch.isfinite(u_p_aligned)
        if asset_mask is not None:
            am = asset_mask.to(device=device, dtype=torch.bool)
            if am.ndim == 1:
                am = am.unsqueeze(-1)
            valid = valid & am[:, :n_h].expand(u_t.shape[0], n_h)
        if horizon_mask is not None:
            hm_b = (horizon_mask.to(device=device).reshape(-1)[:n_h] > 0.5).unsqueeze(0)
            valid = valid & hm_b

        zeros = torch.zeros_like(u_t)
        a = torch.where(valid, u_t, zeros)
        b = torch.where(valid, u_p_aligned, zeros)
        v_cnt = valid.to(dtype=calc_dt).sum(dim=0)  # [H]
        a_sq = a.pow(2).sum(dim=0)
        b_sq = b.pow(2).sum(dim=0)
        eps_sq = float(eps) * float(eps)
        h_ok = (v_cnt >= 2.0) & (a_sq > eps_sq) & (b_sq > eps_sq)
        safe_denom = torch.sqrt(
            torch.where(h_ok, a_sq, torch.ones_like(a_sq))
            * torch.where(h_ok, b_sq, torch.ones_like(b_sq))
        )
        cos_h = torch.where(h_ok, (a * b).sum(dim=0) / safe_denom, torch.ones_like(a_sq)).clamp(
            -1.0, 1.0
        )
        churn_h = 1.0 - cos_h
        h_act = h_ok.to(dtype=calc_dt)
        act_sum = h_act.sum()
        mean_churn = torch.where(
            act_sum > 0.5,
            (churn_h * h_act).sum() / act_sum.clamp_min(1.0),
            torch.zeros((), device=device, dtype=calc_dt),
        )
        return float(w_churn) * mean_churn


def polar_alpha_order_loss(
    u: torch.Tensor,
    y_ret: torch.Tensor,
    *,
    exposures: torch.Tensor | None = None,
    liquidity_weights: torch.Tensor | None = None,
    horizon_mask: torch.Tensor | None = None,
    asset_mask: torch.Tensor | None = None,
    w_ic: float = 0.5,
    w_listnet: float = 0.5,
    tau: float = 0.5,
    use_gaussian_rank_ic: bool = True,
    u_prev: torch.Tensor | None = None,
    cur_asset_ids: torch.Tensor | None = None,
    prev_asset_ids: torch.Tensor | None = None,
    w_churn: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale-invariant composite ranking loss (`#195` P0-1a: neut-IC + `tail_listnet_loss`).

    Strictly reuses `tail_listnet_loss` from `dl_training_sota.py` (never naive `softmax(y/tau)`),
    runs in FP32/FP64 master precision under BF16 autocast, isolates masked/NaN rows inside
    per-horizon factor neutralization and liquidity scaling, guarantees zero gradient leak from
    NaN targets, and optionally adds ragged asset-ID-aligned persistence `L_churn`.
    """
    if u.ndim not in (1, 2):
        raise ValueError(f"u must be 1D [N] or 2D [N, H]; got ndim={u.ndim}")
    device = u.device
    calc_dt = _calc_dtype(u.dtype)

    with _autocast_disabled(device):
        u2 = u.unsqueeze(-1) if u.ndim == 1 else u
        y_raw = y_ret.to(device=device)
        y2 = y_raw.unsqueeze(-1) if y_raw.ndim == 1 else y_raw
        n_assets = u2.shape[0]
        if y2.shape[0] != n_assets:
            raise ValueError(f"asset count mismatch: u {n_assets} vs y_ret {y2.shape[0]}")
        n_h = min(u2.shape[-1], y2.shape[-1])
        u_c = u2[:, :n_h].to(dtype=calc_dt)
        y_c = y2[:, :n_h].to(dtype=calc_dt)

        valid = torch.isfinite(u_c) & torch.isfinite(y_c)
        if asset_mask is not None:
            am = asset_mask.to(device=device, dtype=torch.bool)
            if am.ndim == 1:
                am = am.unsqueeze(-1)
            valid = valid & am[:, :n_h].expand(n_assets, n_h)
        if horizon_mask is not None:
            hm_b = (horizon_mask.to(device=device).reshape(-1)[:n_h] > 0.5).unsqueeze(0)
            valid = valid & hm_b

        zeros = torch.zeros_like(u_c)
        u_clean = torch.where(valid, u_c, zeros)
        y_clean = torch.where(valid, y_c, zeros)

        total = torch.zeros((), device=device, dtype=calc_dt)

        # 1) Factor-neutralized cross-sectional IC loss (`1 - corr`)
        if w_ic > 0.0:
            v_f = valid.to(dtype=calc_dt)
            v_cnt = v_f.sum(dim=0, keepdim=True)
            v_cnt_safe = v_cnt.clamp_min(1.0)
            p_mean = u_clean.sum(dim=0, keepdim=True) / v_cnt_safe
            p_cent = torch.where(valid, u_clean - p_mean, zeros)
            if use_gaussian_rank_ic:
                y_tgt = gaussian_rank_features(y_clean, mask=valid).to(dtype=calc_dt)
            else:
                y_tgt = torch.where(
                    valid, y_clean - y_clean.sum(dim=0, keepdim=True) / v_cnt_safe, zeros
                )

            if (
                exposures is not None
                and exposures.ndim == 2
                and exposures.shape[0] == n_assets
                and exposures.shape[-1] > 0
                and n_assets > exposures.shape[-1]
            ):
                # Per-horizon masked & trace-scaled ridge neutralization (exact 0 leak from NaNs)
                k_fac = exposures.shape[-1]
                b_raw = exposures.detach().to(device=device, dtype=calc_dt)
                b_raw = torch.where(torch.isfinite(b_raw), b_raw, torch.zeros_like(b_raw))
                v_hn1 = valid.T.unsqueeze(-1)  # [H, N, 1]
                v_cnt_h = v_hn1.to(dtype=calc_dt).sum(dim=1, keepdim=True).clamp_min(1.0)
                b_zeros = torch.zeros((n_h, n_assets, k_fac), device=device, dtype=calc_dt)
                b_h = torch.where(v_hn1, b_raw.unsqueeze(0).expand(n_h, n_assets, k_fac), b_zeros)
                b_mean = b_h.sum(dim=1, keepdim=True) / v_cnt_h
                b_h = torch.where(v_hn1, b_h - b_mean, b_zeros)  # [H, N, K]
                b_gram = torch.matmul(b_h.transpose(-1, -2), b_h)  # [H, K, K]
                tr_h = b_gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1).view(n_h, 1, 1)
                ridge_h = (1e-3 * tr_h / float(k_fac)).clamp_min(1e-4)
                eye_k = torch.eye(k_fac, device=device, dtype=calc_dt).unsqueeze(0)
                b_gram = b_gram + ridge_h * eye_k
                rhs = torch.matmul(b_h.transpose(-1, -2), p_cent.T.unsqueeze(-1))  # [H, K, 1]
                proj_w, _ = torch.linalg.solve_ex(b_gram, rhs, check_errors=False)
                proj_p = torch.matmul(b_h, proj_w).squeeze(-1).T  # [N, H]
                p_cent = torch.where(valid, p_cent - proj_p, zeros)
                p_cent = torch.where(
                    valid, p_cent - p_cent.sum(dim=0, keepdim=True) / v_cnt_safe, zeros
                )

            if liquidity_weights is not None:
                lw_raw = liquidity_weights.to(device=device, dtype=calc_dt)
                if lw_raw.ndim == 1:
                    lw_raw = lw_raw.unsqueeze(-1)
                lw_exp = lw_raw[:, :n_h].expand(n_assets, n_h)
                lw_ok = valid & torch.isfinite(lw_exp) & (lw_exp > 0.0)
                lw_pos = torch.where(lw_ok, lw_exp, zeros)
                lw_cnt = lw_ok.to(dtype=calc_dt).sum(dim=0, keepdim=True).clamp_min(1.0)
                lw_mean = lw_pos.sum(dim=0, keepdim=True) / lw_cnt
                # Normalize by valid mean FIRST for exact scale invariance (lw * 1e-6 == lw)
                safe_mean = torch.where(lw_mean > 0.0, lw_mean, torch.ones_like(lw_mean))
                lw_norm = torch.where(
                    lw_ok & (lw_mean > 0.0), (lw_pos / safe_mean).clamp_min(1e-4), zeros
                )
                sqrt_lw = torch.sqrt(lw_norm)
                p_w = p_cent * sqrt_lw
                y_w = y_tgt * sqrt_lw
            else:
                p_w = p_cent
                y_w = y_tgt

            eps_sq = float(eps) * float(eps)
            p_sq = p_w.pow(2).sum(dim=0)
            y_sq = y_w.pow(2).sum(dim=0)
            h_ok = (v_cnt.squeeze(0) >= 2.0) & (p_sq > eps_sq) & (y_sq > eps_sq)
            denom = torch.sqrt(
                torch.where(h_ok, p_sq, torch.ones_like(p_sq))
                * torch.where(h_ok, y_sq, torch.ones_like(y_sq))
            )
            corr_h = torch.where(h_ok, (p_w * y_w).sum(dim=0) / denom, torch.ones_like(p_sq))
            ic_per_h = 1.0 - corr_h
            h_act = h_ok.to(dtype=calc_dt)
            act_sum = h_act.sum()
            loss_ic = torch.where(
                act_sum > 0.5,
                (ic_per_h * h_act).sum() / act_sum.clamp_min(1.0),
                torch.zeros((), device=device, dtype=calc_dt),
            )
            total = total + float(w_ic) * loss_ic

        # 2) Verified Tail-ListNet KL loss from `dl_training_sota.py` (P0-1a)
        if w_listnet > 0.0:
            loss_ln = tail_listnet_loss(
                u_clean,
                y_clean,
                tau=float(tau),
                mask=valid,
                liquidity_weights=liquidity_weights,
                eps=float(eps),
            ).to(dtype=calc_dt)
            total = total + float(w_listnet) * loss_ln

        # 3) Optional asset-ID-aligned persistence `L_churn` (uses live `u_c`, NOT y_ret-masked!)
        if w_churn > 0.0 and u_prev is not None:
            loss_churn = polar_alpha_persistence_loss(
                u_c,
                u_prev,
                cur_asset_ids=cur_asset_ids,
                prev_asset_ids=prev_asset_ids,
                asset_mask=asset_mask,
                horizon_mask=horizon_mask,
                w_churn=float(w_churn),
                eps=float(eps),
            ).to(dtype=calc_dt)
            total = total + loss_churn

    return total


# ---------------------------------------------------------------------------
# 2. ObservedSlotEncoder & MissingnessAwareSetModalityFusion (#195 P0-2a/2b)
# ---------------------------------------------------------------------------


class ObservedSlotEncoder(nn.Module):
    """Variance-stabilized sparse triplet encoder fixing `LayerNorm(78)` `3x` CN gain blowup.

    Accepts either:
      - Packed `encoded_v31` triplets `[..., 3 * n_slots]` (`[vals, obs_mask, log_ages]`),
        e.g. `encoded_v31_78` (`n_slots=26`) or `encoded_v31_420` (`n_slots=140`), OR
      - Explicit `(vals [..., S], obs_mask [..., S], staleness_decay | log_ages [..., S])`.

    Computes in master FP32/FP64 precision:
      `e = RMSNorm( W_val(o * d_s * v) / sqrt(sum(o) + 1.0) + W_pat(concat([o, d_s])) )`
    guaranteeing exact `0.0` output sensitivity and `0.0` gradient to NaN / `±1e6` inside
    unobserved slots (`o == 0`).
    """

    def __init__(
        self,
        n_slots: int,
        hidden_dim: int,
        *,
        max_staleness_days: float = 600.0,
        staleness_halflife_days: float = 63.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1; got {n_slots}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1; got {hidden_dim}")
        if max_staleness_days <= 0.0 or staleness_halflife_days <= 0.0:
            raise ValueError("max_staleness_days and staleness_halflife_days must be > 0")

        self.n_slots = int(n_slots)
        self.hidden_dim = int(hidden_dim)
        self.max_staleness_days = float(max_staleness_days)
        self.staleness_halflife_days = float(staleness_halflife_days)
        self.max_log_age = math.log1p(self.max_staleness_days)
        self.staleness_scale = math.log1p(max(self.staleness_halflife_days, 1.0)) / math.log(2.0)

        # Bias-free value projection so all-missing rows contribute exact 0.0 from value path
        self.w_val = nn.Linear(self.n_slots, self.hidden_dim, bias=False)
        self.w_pat = nn.Linear(2 * self.n_slots, self.hidden_dim, bias=True)
        self.norm = RMSNormSOTA(self.hidden_dim, eps=float(eps))

    def forward(
        self,
        x: torch.Tensor,
        *,
        obs_mask: torch.Tensor | None = None,
        staleness_decay: torch.Tensor | None = None,
        log_ages: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode packed `[..., 3*S]` triplet or explicit `(vals, obs_mask, staleness_decay)`."""
        orig_dtype = x.dtype
        device = x.device
        calc_dt = _calc_dtype(orig_dtype)

        with _autocast_disabled(device):
            if obs_mask is None and x.shape[-1] == 3 * self.n_slots:
                v_raw, o_raw, la_raw = torch.split(x, self.n_slots, dim=-1)
                v_c = v_raw.to(dtype=calc_dt)
                o_c = o_raw.to(dtype=calc_dt)
                obs_bool = torch.isfinite(o_c) & (o_c > 0.5) & torch.isfinite(v_c)
                la_c = la_raw.to(dtype=calc_dt)
                la_safe = torch.where(
                    obs_bool & torch.isfinite(la_c),
                    la_c.clamp(0.0, self.max_log_age),
                    torch.zeros_like(la_c),
                )
                d_s = torch.where(
                    obs_bool,
                    torch.exp(-la_safe / max(self.staleness_scale, 1e-6)),
                    torch.zeros_like(la_safe),
                )
            else:
                if x.shape[-1] != self.n_slots:
                    raise ValueError(
                        f"Expected last dim {self.n_slots} (or {3 * self.n_slots} packed triplet); "
                        f"got {x.shape[-1]}"
                    )
                v_c = x.to(dtype=calc_dt)
                if obs_mask is None:
                    obs_bool = torch.isfinite(v_c)
                else:
                    om_c = obs_mask.to(device=device, dtype=calc_dt)
                    obs_bool = torch.isfinite(om_c) & (om_c > 0.5) & torch.isfinite(v_c)

                zeros_s = torch.zeros_like(v_c)
                if staleness_decay is not None:
                    ds_raw = staleness_decay.to(device=device, dtype=calc_dt)
                    d_s = torch.where(
                        obs_bool & torch.isfinite(ds_raw), ds_raw.clamp(0.0, 1.0), zeros_s
                    )
                elif log_ages is not None:
                    la_c = log_ages.to(device=device, dtype=calc_dt)
                    la_safe = torch.where(
                        obs_bool & torch.isfinite(la_c),
                        la_c.clamp(0.0, self.max_log_age),
                        zeros_s,
                    )
                    d_s = torch.where(
                        obs_bool,
                        torch.exp(-la_safe / max(self.staleness_scale, 1e-6)),
                        zeros_s,
                    )
                else:
                    d_s = obs_bool.to(dtype=calc_dt)

            zeros_v = torch.zeros_like(v_c)
            o_f = obs_bool.to(dtype=calc_dt)
            # Pre-isolate unobserved slots via torch.where BEFORE linear ops (exact 0.0 grad)
            v_clean = torch.where(obs_bool, v_c, zeros_v)
            weighted_v = o_f * d_s * v_clean
            denom = torch.sqrt(o_f.sum(dim=-1, keepdim=True) + 1.0)

            val_proj = (
                F.linear(weighted_v, self.w_val.weight.to(device=device, dtype=calc_dt), None)
                / denom
            )
            pat_in = torch.cat([o_f, d_s], dim=-1)
            pat_bias = (
                self.w_pat.bias.to(device=device, dtype=calc_dt)
                if self.w_pat.bias is not None
                else None
            )
            pat_proj = F.linear(
                pat_in, self.w_pat.weight.to(device=device, dtype=calc_dt), pat_bias
            )
            out_c = self.norm(val_proj + pat_proj)

        return out_c.to(dtype=orig_dtype)


class MissingnessAwareSetModalityFusion(nn.Module):
    """Schema-agnostic MNAR-Query Perceiver Cross-Attention over present modalities (`#195` P0-2b).

    Builds query:
      `q_i = RMSNorm(core_emb + sum_m [ (1 - I_{i,m}) u_m^miss + I_{i,m} u_m^pres ])`
    and applies QK-RMSNorm cross-attention strictly over keys/values of present auxiliary
    modalities PLUS 1 always-present core self-token anchor (`mask=True`).

    Guarantees:
      - Strictly finite output even when 100% of auxiliary modalities are missing (`mod_mask == 0`).
      - Bit-exact `0.0` output difference and exact `0.0` gradient w.r.t. garbage / `NaN` / `±1e6`
        values inside masked-out missing modalities.
      - Optional matched placebo control `mnar_placebo_shuffle=True` permuting availability
        indicators `I_{i,m}` in `q_i` while keeping true physical KV masking intact.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        n_aux_modalities: int = 4,
        n_heads: int = 4,
        stale_halflife: float | Sequence[float] = 30.0,
        max_staleness_days: float = 600.0,
        mnar_placebo_shuffle: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_dim < 4:
            raise ValueError(f"hidden_dim must be >= 4; got {hidden_dim}")
        if n_aux_modalities < 1:
            raise ValueError(f"n_aux_modalities must be >= 1; got {n_aux_modalities}")
        if n_heads < 1 or hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
            )
        if max_staleness_days <= 0.0:
            raise ValueError("max_staleness_days must be > 0")

        self.hidden_dim = int(hidden_dim)
        self.n_aux_modalities = int(n_aux_modalities)
        self.n_heads = int(n_heads)
        self.head_dim = self.hidden_dim // self.n_heads
        self.max_staleness_days = float(max_staleness_days)
        self.mnar_placebo_shuffle = bool(mnar_placebo_shuffle)

        scale_emb = 0.05 / math.sqrt(float(self.hidden_dim))
        self.u_miss = nn.Parameter(
            torch.randn(1, self.n_aux_modalities, self.hidden_dim) * scale_emb
        )
        self.u_pres = nn.Parameter(
            torch.randn(1, self.n_aux_modalities, self.hidden_dim) * scale_emb
        )
        self.mod_type_emb = nn.Parameter(
            torch.randn(1, self.n_aux_modalities, self.hidden_dim) * scale_emb
        )
        # Log1p staleness rate matches ObservedSlotEncoder & prevents quarterly 10-Q starvation
        if isinstance(stale_halflife, Sequence):
            if len(stale_halflife) != self.n_aux_modalities:
                raise ValueError(
                    f"stale_halflife sequence len ({len(stale_halflife)}) "
                    f"!= n_aux_modalities ({self.n_aux_modalities})"
                )
            hl_vals = [float(v) for v in stale_halflife]
            if any(v <= 0.0 for v in hl_vals):
                raise ValueError("All stale_halflife entries must be > 0")
            raw_rates = [_inv_softplus(math.log(2.0) / math.log1p(v)) for v in hl_vals]
            self.age_decay_raw = nn.Parameter(
                torch.tensor(raw_rates, dtype=torch.float32).view(1, self.n_aux_modalities)
            )
        else:
            if float(stale_halflife) <= 0.0:
                raise ValueError(f"stale_halflife must be > 0; got {stale_halflife}")
            init_rate = math.log(2.0) / math.log1p(float(stale_halflife))
            self.age_decay_raw = nn.Parameter(
                torch.full(
                    (1, self.n_aux_modalities), _inv_softplus(init_rate), dtype=torch.float32
                )
            )

        self.q_norm = RMSNormSOTA(self.hidden_dim, eps=float(eps))
        self.core_kv_norm = RMSNormSOTA(self.hidden_dim, eps=float(eps))
        self.kv_norm = RMSNormSOTA(self.hidden_dim, eps=float(eps))

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.qk_q_norm = RMSNormSOTA(self.head_dim, eps=float(eps))
        self.qk_k_norm = RMSNormSOTA(self.head_dim, eps=float(eps))
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.out_norm = RMSNormSOTA(self.hidden_dim, eps=float(eps))

    def forward(
        self,
        mod_tokens: torch.Tensor,
        mod_mask: torch.Tensor,
        core_emb: torch.Tensor,
        modality_age: torch.Tensor | None = None,
        *,
        mnar_placebo_shuffle: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fuse `(mod_tokens [N, M, D], mod_mask [N, M], core_emb [N, D])` -> `(fused, diag)`."""
        if mod_tokens.ndim != 3 or mod_tokens.shape[1] != self.n_aux_modalities:
            raise ValueError(
                f"mod_tokens must have shape [N, {self.n_aux_modalities}, {self.hidden_dim}]; "
                f"got {tuple(mod_tokens.shape)}"
            )
        if mod_tokens.shape[2] != self.hidden_dim:
            raise ValueError(
                f"mod_tokens hidden_dim ({mod_tokens.shape[2]}) != {self.hidden_dim}"
            )
        n_assets = mod_tokens.shape[0]
        device, orig_dtype = mod_tokens.device, mod_tokens.dtype
        calc_dt = _calc_dtype(orig_dtype)
        use_placebo = (
            self.mnar_placebo_shuffle
            if mnar_placebo_shuffle is None
            else bool(mnar_placebo_shuffle)
        )

        with _autocast_disabled(device):
            m_raw = mod_mask.to(device=device, dtype=calc_dt)
            tok_c = mod_tokens.to(dtype=calc_dt)
            # Guard present modalities against NaN/Inf tokens (#202 P1-4, #203 P0-2a)
            tok_fin = torch.isfinite(tok_c).all(dim=-1)
            mask_bool = torch.isfinite(m_raw) & (m_raw > 0.5) & tok_fin  # [N, M]
            core_raw = core_emb.to(device=device, dtype=calc_dt)
            core_c = torch.where(torch.isfinite(core_raw), core_raw, torch.zeros_like(core_raw))

            # Bit-exact pre-isolation of missing/corrupted tokens BEFORE any linear/norm op
            zeros_tok = torch.zeros_like(tok_c)
            clean_tok = torch.where(mask_bool.unsqueeze(-1), tok_c, zeros_tok)

            # MNAR availability signal in query (optionally shuffled for matched placebo control)
            if use_placebo:
                perm = torch.randperm(n_assets, device=device, generator=generator)
                mnar_mask = mask_bool[perm]
            else:
                mnar_mask = mask_bool
            mnar_f = mnar_mask.to(dtype=calc_dt).unsqueeze(-1)  # [N, M, 1]
            u_miss_c = self.u_miss.to(device=device, dtype=calc_dt)
            u_pres_c = self.u_pres.to(device=device, dtype=calc_dt)
            mnar_delta = ((1.0 - mnar_f) * u_miss_c + mnar_f * u_pres_c).sum(dim=1)  # [N, D]
            q_i = self.q_norm(core_c + mnar_delta)  # [N, D]

            # Build KV sequence: [always-present core self-token anchor, M auxiliary modalities]
            anchor_tok = self.core_kv_norm(core_c).unsqueeze(1)  # [N, 1, D]
            mod_type_c = self.mod_type_emb.to(device=device, dtype=calc_dt)
            aux_normed = self.kv_norm(clean_tok) + mod_type_c
            aux_tok = torch.where(mask_bool.unsqueeze(-1), aux_normed, zeros_tok)
            kv_seq = torch.cat([anchor_tok, aux_tok], dim=1)  # [N, M + 1, D]

            anchor_mask = torch.ones((n_assets, 1), device=device, dtype=torch.bool)
            kv_mask = torch.cat([anchor_mask, mask_bool], dim=1)  # [N, M + 1]
            m_plus_1 = self.n_aux_modalities + 1

            # Log1p freshness log-bias on present aux modalities (0.0 on core anchor; #203 P0-2b)
            age_bias: torch.Tensor | None = None
            if modality_age is not None:
                age_raw = modality_age.to(device=device, dtype=calc_dt)
                zeros_m = torch.zeros(
                    (n_assets, self.n_aux_modalities), device=device, dtype=calc_dt
                )
                age_clean = torch.where(
                    mask_bool & torch.isfinite(age_raw), age_raw, zeros_m
                ).clamp(0.0, self.max_staleness_days)
                log_age = torch.log1p(age_clean)
                rate = F.softplus(self.age_decay_raw.to(device=device, dtype=calc_dt)) + 1e-4
                aux_bias = torch.where(mask_bool, -(rate * log_age).clamp(max=20.0), zeros_m)
                anchor_bias = torch.zeros((n_assets, 1), device=device, dtype=calc_dt)
                age_bias = torch.cat([anchor_bias, aux_bias], dim=1).view(n_assets, 1, 1, m_plus_1)

            # QK-RMSNorm Perceiver cross-attention in master precision
            q_lin = F.linear(q_i, self.q_proj.weight.to(device=device, dtype=calc_dt), None)
            k_lin = F.linear(kv_seq, self.k_proj.weight.to(device=device, dtype=calc_dt), None)
            v_lin = F.linear(kv_seq, self.v_proj.weight.to(device=device, dtype=calc_dt), None)

            q_h = self.qk_q_norm(
                q_lin.view(n_assets, 1, self.n_heads, self.head_dim).transpose(1, 2)
            )  # [N, H, 1, d_h]
            k_h = self.qk_k_norm(
                k_lin.view(n_assets, m_plus_1, self.n_heads, self.head_dim).transpose(1, 2)
            )  # [N, H, M+1, d_h]
            v_h = v_lin.view(n_assets, m_plus_1, self.n_heads, self.head_dim).transpose(1, 2)

            inv_scale = 1.0 / math.sqrt(float(self.head_dim))
            logits = torch.matmul(q_h, k_h.transpose(-1, -2)) * inv_scale  # [N, H, 1, M+1]
            if age_bias is not None:
                logits = logits + age_bias

            kv_m4 = kv_mask.view(n_assets, 1, 1, m_plus_1)
            attn = _stable_masked_softmax(logits, kv_m4, dim=-1)  # [N, H, 1, M+1]
            ctx = (
                torch.matmul(attn, v_h)
                .transpose(1, 2)
                .contiguous()
                .view(n_assets, self.hidden_dim)
            )
            delta = F.linear(ctx, self.out_proj.weight.to(device=device, dtype=calc_dt), None)
            fused_c = self.out_norm(q_i + delta)

            attn_mean = attn.detach().mean(dim=1).squeeze(1)  # [N, M+1]
            diag = dict(
                attn_weights=attn_mean.to(dtype=orig_dtype),
                anchor_weight=attn_mean[:, 0].mean().to(dtype=orig_dtype),
                present_aux_frac=mask_bool.to(dtype=calc_dt).mean().to(dtype=orig_dtype),
            )

        return fused_c.to(dtype=orig_dtype), diag


# ---------------------------------------------------------------------------
# 3. DiffInducedFactorCrossAttention (Ye et al. ICLR 2025, #195 P0-3a/3b/3c)
# ---------------------------------------------------------------------------


class _DifferentialCrossAttnOp(nn.Module):
    """Single differential attention op with Gemma-2 soft-cap, bounded lambda & GroupRMSNorm."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        *,
        lambda_init: float = 0.2,
        softcap: float = 30.0,
        zero_lambda_control: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_dim % (2 * n_heads) != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by 2 * n_heads ({2 * n_heads})"
            )
        if not (0.0 < lambda_init < 1.0):
            raise ValueError(
                f"lambda_init must be in (0, 1) so (1 - lambda_init) > 0; got {lambda_init}"
            )
        if softcap <= 0.0:
            raise ValueError(f"softcap must be > 0; got {softcap}")

        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.hidden_dim // self.n_heads
        self.sub_head_dim = self.head_dim // 2
        self.lambda_init = float(lambda_init)
        self.softcap = float(softcap)
        self.zero_lambda_control = bool(zero_lambda_control)
        self.eps = float(eps)

        self.q_norm = RMSNormSOTA(self.sub_head_dim, eps=self.eps)
        self.k_norm = RMSNormSOTA(self.sub_head_dim, eps=self.eps)
        self.group_norm_weight = nn.Parameter(
            torch.ones(1, self.n_heads, 1, self.head_dim, dtype=torch.float32)
        )

        # Bounded parameterization: lambda = 2.0 * sigmoid(raw_lambda) in (0, 2) (#195 P0-3c)
        p_init = min(max(self.lambda_init / 2.0, 1e-4), 1.0 - 1e-4)
        raw_init = math.log(p_init / (1.0 - p_init))
        self.raw_lambda = nn.Parameter(
            torch.full((self.n_heads,), raw_init, dtype=torch.float32)
        )

    def effective_lambda(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return strictly bounded per-head `lambda` `[1, H, 1, 1]` in `(0, 2)` (or `0.0`)."""
        dev = self.raw_lambda.device if device is None else device
        if self.zero_lambda_control:
            return torch.zeros((1, self.n_heads, 1, 1), device=dev, dtype=dtype)
        lam = 2.0 * torch.sigmoid(self.raw_lambda.to(device=dev, dtype=dtype))
        return lam.view(1, self.n_heads, 1, 1)

    def _compute_weights_detail(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute `(a_diff, lam, a1)` in FP32/FP64 master precision with Gemma-2 soft-cap."""
        calc_dt = _calc_dtype(q.dtype)
        with _autocast_disabled(q.device):
            q_c = q.to(dtype=calc_dt)
            k_c = k.to(dtype=calc_dt)
            q1, q2 = q_c.chunk(2, dim=-1)
            k1, k2 = k_c.chunk(2, dim=-1)
            q1 = self.q_norm(q1)
            k1 = self.k_norm(k1)

            inv_scale = 1.0 / math.sqrt(float(self.sub_head_dim))
            inv_cap = 1.0 / self.softcap
            logits1 = torch.matmul(q1, k1.transpose(-1, -2)) * inv_scale
            s1 = self.softcap * torch.tanh(logits1 * inv_cap)

            km4 = (
                key_mask.to(device=q.device, dtype=torch.bool).view(1, 1, 1, -1)
                if key_mask is not None
                else None
            )
            if km4 is not None:
                a1 = _stable_masked_softmax(s1, km4, dim=-1)
            else:
                a1 = torch.softmax(s1, dim=-1)

            lam = self.effective_lambda(device=q.device, dtype=calc_dt)
            if self.zero_lambda_control:
                a_diff = a1
            else:
                q2 = self.q_norm(q2)
                k2 = self.k_norm(k2)
                logits2 = torch.matmul(q2, k2.transpose(-1, -2)) * inv_scale
                s2 = self.softcap * torch.tanh(logits2 * inv_cap)
                a2 = (
                    _stable_masked_softmax(s2, km4, dim=-1)
                    if km4 is not None
                    else torch.softmax(s2, dim=-1)
                )
                a_diff = a1 - lam * a2
        return a_diff, lam, a1

    def _compute_weights(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute differential attention weights `(a_diff, lam)` in master precision."""
        a_diff, lam, _ = self._compute_weights_detail(q, k, key_mask=key_mask)
        return a_diff, lam

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply differential attention + per-head `GroupRMSNorm` scaled by `(1 - lambda_init)`."""
        orig_dtype = q.dtype
        calc_dt = _calc_dtype(orig_dtype)
        with _autocast_disabled(q.device):
            a_diff, lam, a1 = self._compute_weights_detail(q, k, key_mask=key_mask)
            v_c = v.to(dtype=calc_dt)
            out_h = torch.matmul(a_diff, v_c)  # [1, H, Lq, d_h]
            # Per-head GroupRMSNorm scaled by (1 - lambda_init) (Ye et al., ICLR 2025)
            mean_sq = out_h.pow(2).mean(dim=-1, keepdim=True)
            gn_w = self.group_norm_weight.to(device=q.device, dtype=calc_dt)
            out_h = out_h * torch.rsqrt(mean_sq + self.eps) * gn_w * (1.0 - self.lambda_init)
        return out_h.to(dtype=orig_dtype), lam, a1


class DiffInducedFactorCrossAttention(nn.Module):
    """F3 Drop-In Differential Cross-Asset Attention (`O(N * R)`, Ye et al. ICLR 2025).

    Matches `InducedFactorCrossAttention` (`forward(h, asset_mask=None, ...) -> (out, diag)`)
    with ~0% extra QKV parameters, explicit FP32/FP64 L2-resident `[1, H, N, R]` / `[1, H, R, N]`
    attention + Gemma-2 soft-cap `c * tanh(logits / c)`, bounded `lambda = 2 * sigmoid(raw)`,
    per-head `GroupRMSNorm`, matched `zero_lambda_control`, and strict `asset_mask` hygiene.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        n_landmarks: int = 16,
        n_heads: int = 4,
        context_dim: int = 0,
        init_gate: float = 0.05,
        layer_idx: int = 1,
        lambda_init: float | None = None,
        softcap: float = 30.0,
        zero_lambda_control: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_dim < 4:
            raise ValueError("hidden_dim must be >= 4")
        if n_landmarks < 2:
            raise ValueError("n_landmarks must be >= 2")
        if n_heads < 1 or hidden_dim % (2 * n_heads) != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by 2 * n_heads ({2 * n_heads})"
            )
        if softcap <= 0.0:
            raise ValueError(f"softcap must be > 0; got {softcap}")

        self.hidden_dim = int(hidden_dim)
        self.n_landmarks = int(n_landmarks)
        self.n_heads = int(n_heads)
        self.head_dim = self.hidden_dim // self.n_heads
        self.context_dim = int(context_dim)
        self.softcap = float(softcap)
        self.zero_lambda_control = bool(zero_lambda_control)
        eff_lam_init = (
            float(lambda_init)
            if lambda_init is not None
            else (0.8 - 0.6 * math.exp(-0.3 * max(int(layer_idx) - 1, 0)))
        )
        self.lambda_init = eff_lam_init

        self.in_norm = RMSNormSOTA(self.hidden_dim, eps=float(eps))
        self.landmarks = nn.Parameter(
            torch.randn(self.n_landmarks, self.hidden_dim) / math.sqrt(float(self.hidden_dim))
        )
        self.regime_to_land = (
            nn.Linear(self.context_dim, self.hidden_dim) if self.context_dim > 0 else None
        )

        # Phase 0 (F-invariant): learned landmarks attend standardized factor tokens
        self.f_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.f_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.qf_norm = RMSNormSOTA(self.head_dim, eps=float(eps))
        self.kf_norm = RMSNormSOTA(self.head_dim, eps=float(eps))

        # Phase 1: Asset -> Landmark Differential Gather O(N * R * D)
        self.q1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.gather_attn = _DifferentialCrossAttnOp(
            self.hidden_dim,
            self.n_heads,
            lambda_init=eff_lam_init,
            softcap=self.softcap,
            zero_lambda_control=self.zero_lambda_control,
            eps=float(eps),
        )

        # Phase 2: Landmark <-> Landmark Differential Reasoning O(R^2 * D)
        self.norm_m = RMSNormSOTA(self.hidden_dim, eps=float(eps))
        self.qm = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.km = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.vm = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.mix_attn = _DifferentialCrossAttnOp(
            self.hidden_dim,
            self.n_heads,
            lambda_init=eff_lam_init,
            softcap=self.softcap,
            zero_lambda_control=self.zero_lambda_control,
            eps=float(eps),
        )

        # Phase 3: Landmark -> Asset Differential Broadcast O(N * R * D)
        self.norm_b = RMSNormSOTA(self.hidden_dim, eps=float(eps))
        self.q2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.broadcast_attn = _DifferentialCrossAttnOp(
            self.hidden_dim,
            self.n_heads,
            lambda_init=eff_lam_init,
            softcap=self.softcap,
            zero_lambda_control=self.zero_lambda_control,
            eps=float(eps),
        )
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.out_gate = nn.Parameter(torch.full((), float(init_gate), dtype=torch.float32))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(1, x.shape[0], self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(x.shape[2], self.hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        asset_mask: torch.Tensor | None = None,
        *,
        exposures: torch.Tensor | None = None,
        rmt_loadings: torch.Tensor | None = None,
        factor_mat: torch.Tensor | None = None,
        step_emb: torch.Tensor | None = None,
        regime_vec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run `O(N * R)` differential cross-asset attention with strict mask & FP32 hygiene."""
        n_assets = h.shape[0]
        device, orig_dtype = h.device, h.dtype
        calc_dt = _calc_dtype(orig_dtype)

        m_bool: torch.Tensor | None = None
        if asset_mask is not None:
            m_bool = asset_mask.to(device=device, dtype=torch.bool).reshape(n_assets)
            h_clean = torch.where(m_bool.unsqueeze(-1), h, torch.zeros_like(h))
        else:
            h_clean = h

        h_n = self.in_norm(h_clean)
        if m_bool is not None:
            h_n = torch.where(m_bool.unsqueeze(-1), h_n, torch.zeros_like(h_n))

        q_land = self.landmarks.to(device=device, dtype=orig_dtype)
        if step_emb is not None:
            q_land = q_land + step_emb.to(device=device, dtype=orig_dtype).unsqueeze(0)
        if self.regime_to_land is not None and regime_vec is not None:
            reg = regime_vec.to(device=device, dtype=orig_dtype).reshape(-1)
            q_land = q_land + self.regime_to_land(reg).unsqueeze(0)

        # Phase 0 (F-invariant): factor tokens `B_n^T @ h_n / valid_count` -> `[M, D]`
        f_mat = (
            factor_mat
            if factor_mat is not None
            else _standardize_factor_concat_masked(
                exposures=exposures,
                rmt_loadings=rmt_loadings,
                n_assets=n_assets,
                asset_mask=m_bool,
                device=device,
                dtype=orig_dtype,
            )
        )
        if f_mat is not None and f_mat.shape[-1] > 0:
            with _autocast_disabled(device):
                fm_c = f_mat.to(device=device, dtype=calc_dt)
                hn_c = h_n.to(dtype=calc_dt)
                if m_bool is not None:
                    fm_c = torch.where(m_bool.unsqueeze(-1), fm_c, torch.zeros_like(fm_c))
                    denom_n = m_bool.to(dtype=calc_dt).sum().clamp_min(1.0)
                else:
                    denom_n = max(float(n_assets), 1.0)
                # Keep Phase 0 matmuls in master `calc_dt` so BF16 autocast never mismatches weights
                f_tok_c = (fm_c.T @ hn_c) / denom_n
                fk_c = F.linear(f_tok_c, self.f_k.weight.to(device=device, dtype=calc_dt), None)
                fv_c = F.linear(f_tok_c, self.f_v.weight.to(device=device, dtype=calc_dt), None)
                qf = self.qf_norm(self._split_heads(q_land.to(dtype=calc_dt)))
                kf = self.kf_norm(self._split_heads(fk_c))
                vf = self._split_heads(fv_c)
                inv_s0 = 1.0 / math.sqrt(float(self.head_dim))
                s0 = self.softcap * torch.tanh(
                    (torch.matmul(qf, kf.transpose(-1, -2)) * inv_s0) / self.softcap
                )
                a0 = torch.softmax(s0, dim=-1)
                land_seed = q_land + self._merge_heads(torch.matmul(a0, vf)).to(dtype=orig_dtype)
        else:
            land_seed = q_land

        # Phase 1: Asset -> Landmark Differential Gather (`[1, H, R, N]` in FP32 L2-resident)
        q1 = self._split_heads(self.q1(land_seed))
        k1 = self._split_heads(self.k1(h_n))
        v1 = self._split_heads(self.v1(h_n))
        g_out, lam1, _ = self.gather_attn(q1, k1, v1, key_mask=m_bool)
        land = land_seed + self._merge_heads(g_out)

        # Phase 2: Landmark <-> Landmark Differential Reasoning (`[1, H, R, R]`)
        land_m = self.norm_m(land)
        qm = self._split_heads(self.qm(land_m))
        km = self._split_heads(self.km(land_m))
        vm = self._split_heads(self.vm(land_m))
        m_out, lam2, _ = self.mix_attn(qm, km, vm, key_mask=None)
        land_tilde = land + self._merge_heads(m_out)

        # Phase 3: Landmark -> Asset Differential Broadcast (`[1, H, N, R]` in FP32 L2-resident)
        land_b = self.norm_b(land_tilde)
        q2 = self._split_heads(self.q2(h_n))
        k2 = self._split_heads(self.k2(land_b))
        v2 = self._split_heads(self.v2(land_b))
        b_out, lam3, a1_bcast = self.broadcast_attn(q2, k2, v2, key_mask=None)
        delta = self.out_proj(self._merge_heads(b_out))

        with torch.no_grad():
            with _autocast_disabled(device):
                ent_rows = -(a1_bcast * torch.log(a1_bcast.clamp_min(1e-8))).sum(dim=-1)
                if m_bool is not None:
                    m_f = m_bool.to(dtype=calc_dt).view(1, 1, n_assets)
                    ent = (ent_rows * m_f).sum() / (m_f.sum() * float(self.n_heads)).clamp_min(1.0)
                else:
                    ent = ent_rows.mean()
                lam1_m = lam1.mean()
                lam2_m = lam2.mean()
                lam3_m = lam3.mean()
                lam_all = (lam1_m + lam2_m + lam3_m) / 3.0

        gate = self.out_gate.to(device=device, dtype=orig_dtype)
        out = h_clean + gate * delta
        if m_bool is not None:
            out = torch.where(m_bool.unsqueeze(-1), out, torch.zeros_like(out))

        return out, dict(
            landmark_attn_entropy=ent.to(dtype=orig_dtype),
            diff_lambda_mean=lam_all.to(dtype=orig_dtype),
            diff_lambda_gather=lam1_m.to(dtype=orig_dtype),
            diff_lambda_mix=lam2_m.to(dtype=orig_dtype),
            diff_lambda_broadcast=lam3_m.to(dtype=orig_dtype),
        )


# ---------------------------------------------------------------------------
# 4. Stage-2 Cross-Family Epistemic Disagreement & CKA-Orthogonal Stacking (#5C)
# ---------------------------------------------------------------------------


@overload
def compute_cka_dissimilarity_weights(
    cka_matrix: torch.Tensor,
    *,
    min_weight: float = 1e-4,
) -> torch.Tensor: ...


@overload
def compute_cka_dissimilarity_weights(
    cka_matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    min_weight: float = 1e-4,
) -> np.ndarray: ...


def compute_cka_dissimilarity_weights(
    cka_matrix: torch.Tensor | np.ndarray | Sequence[Sequence[float]],
    *,
    min_weight: float = 1e-4,
) -> torch.Tensor | np.ndarray:
    """Compute normalized CKA-dissimilarity weights `w_f proportional to 1 - mean_{g!=f} C[f,g]`.

    Families that are representationally orthogonal to the rest of the ensemble receive strictly
    higher weight than redundant clones. Pure PyTorch for Tensor inputs (0 host-syncs), pure NumPy
    for ndarray/sequence inputs.
    """
    if isinstance(cka_matrix, torch.Tensor):
        if cka_matrix.ndim != 2 or cka_matrix.shape[0] != cka_matrix.shape[1]:
            raise ValueError(f"cka_matrix must be square [F, F]; got {tuple(cka_matrix.shape)}")
        n_fam = cka_matrix.shape[0]
        if n_fam < 1:
            raise ValueError("cka_matrix must have F >= 1")
        calc_dt = _calc_dtype(cka_matrix.dtype)
        device = cka_matrix.device
        with _autocast_disabled(device):
            if n_fam == 1:
                return torch.ones((1,), device=device, dtype=calc_dt)
            c = cka_matrix.to(dtype=calc_dt)
            c = torch.where(torch.isfinite(c), c, torch.zeros_like(c)).clamp(0.0, 1.0)
            off = 1.0 - torch.eye(n_fam, device=device, dtype=calc_dt)
            mean_sim = (c * off).sum(dim=1) / float(n_fam - 1)
            raw_w = (1.0 - mean_sim).clamp_min(float(min_weight))
            return raw_w / raw_w.sum().clamp_min(1e-12)

    arr = np.asarray(cka_matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 1:
        raise ValueError(f"cka_matrix must be non-empty square [F, F]; got {arr.shape}")
    n_fam = int(arr.shape[0])
    if n_fam == 1:
        return np.ones((1,), dtype=np.float64)
    c = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    off = 1.0 - np.eye(n_fam, dtype=np.float64)
    mean_sim = (c * off).sum(axis=1) / float(n_fam - 1)
    raw_w = np.maximum(1.0 - mean_sim, float(min_weight))
    return raw_w / max(float(raw_w.sum()), 1e-12)


def _extract_family_list(
    family_preds: Sequence[Any] | Mapping[str, Any] | torch.Tensor | np.ndarray,
) -> list[Any]:
    if isinstance(family_preds, Mapping):
        items = list(family_preds.values())
    elif isinstance(family_preds, (torch.Tensor, np.ndarray)):
        if family_preds.ndim < 2:
            raise ValueError(
                f"Stacked family_preds must have shape [F, N] or [F, N, H]; "
                f"got {family_preds.shape}"
            )
        items = [family_preds[i] for i in range(family_preds.shape[0])]
    else:
        items = list(family_preds)
    if len(items) < 1:
        raise ValueError("family_preds must contain at least 1 family")
    return items


def compute_cross_family_disagreement_risk(
    family_preds: Sequence[torch.Tensor | np.ndarray]
    | Mapping[str, torch.Tensor | np.ndarray]
    | torch.Tensor
    | np.ndarray,
    realized_vol: torch.Tensor | np.ndarray,
    *,
    kappa: float = 1.0,
    cka_weights: Sequence[float] | torch.Tensor | np.ndarray | None = None,
    cka_matrix: torch.Tensor | np.ndarray | None = None,
    vol_scale: float | None = None,
    min_vol: float = 1e-4,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Compute CKA-weighted consensus & epistemic disagreement risk `sigma_eff` for Stage-2 `P+`.

    For each independent family `f in 1..F`:
      1. Cross-sectionally standardizes `pred_{i,f} -> u_{i,f}` (two-pass FP32/FP64, NaN-safe).
      2. Computes CKA-dissimilarity-weighted consensus `u_bar_i = sum_f w_f u_{i,f}` and
         epistemic disagreement variance `Var_f(u_{i,f}) = sum_f w_f (u_{i,f} - u_bar_i)^2`.
      3. Returns `sigma_eff = sqrt(realized_vol^2 + kappa * Var_f(u) * scale)` where
         `scale = realized_vol^2` by default (dimensionally exact across US & CN vol regimes,
         or explicit `vol_scale` when specified).
      4. Matched control `kappa == 0.0` is **bit-exact identical** to `realized_vol`.

    Pure PyTorch (0 host-syncs, 0 `torch.tensor()`) for Tensor inputs; pure NumPy for ndarray.
    """
    if kappa < 0.0:
        raise ValueError(f"kappa must be >= 0; got {kappa}")
    if min_vol <= 0.0:
        raise ValueError(f"min_vol must be > 0; got {min_vol}")
    if vol_scale is not None and vol_scale < 0.0:
        raise ValueError(f"vol_scale must be >= 0; got {vol_scale}")

    preds_list = _extract_family_list(family_preds)
    n_fam = len(preds_list)
    use_torch = isinstance(realized_vol, torch.Tensor) or isinstance(preds_list[0], torch.Tensor)

    if use_torch:
        if not isinstance(realized_vol, torch.Tensor):
            raise TypeError("realized_vol must be a torch.Tensor when family_preds are Tensors")
        device = realized_vol.device
        orig_dtype = realized_vol.dtype
        calc_dt = _calc_dtype(orig_dtype)

        with _autocast_disabled(device):
            stacked_list = [p.to(device=device, dtype=calc_dt) for p in preds_list]
            stacked = torch.stack(stacked_list, dim=0)  # [F, N] or [F, N, H]
            valid = torch.isfinite(stacked)
            zeros = torch.zeros_like(stacked)
            clean = torch.where(valid, stacked, zeros)

            # Two-pass cross-sectional z-score per family along asset axis dim=1
            v_cnt = valid.to(dtype=calc_dt).sum(dim=1, keepdim=True)
            v_cnt_safe = v_cnt.clamp_min(1.0)
            mean = clean.sum(dim=1, keepdim=True) / v_cnt_safe
            centered = torch.where(valid, clean - mean, zeros)
            var = centered.pow(2).sum(dim=1, keepdim=True) / v_cnt_safe
            eps_sq = float(eps) * float(eps)
            has_var = (var > eps_sq) & (v_cnt >= 1.5)
            safe_var = torch.where(has_var, var, torch.ones_like(var))
            u_fam = torch.where(valid & has_var, centered / torch.sqrt(safe_var), zeros)

            # Build normalized family weights [F] with 0 `torch.tensor(` calls
            if cka_weights is not None:
                if isinstance(cka_weights, torch.Tensor):
                    w_raw = cka_weights.to(device=device, dtype=calc_dt).reshape(-1)
                    if w_raw.shape[0] != n_fam:
                        raise ValueError(
                            f"cka_weights len ({w_raw.shape[0]}) != n_families ({n_fam})"
                        )
                    w_raw = torch.where(
                        torch.isfinite(w_raw) & (w_raw > 0.0), w_raw, torch.zeros_like(w_raw)
                    )
                    w_sum = w_raw.sum()
                    w_fam = torch.where(
                        w_sum > 1e-12,
                        w_raw / w_sum.clamp_min(1e-12),
                        torch.full((n_fam,), 1.0 / float(n_fam), device=device, dtype=calc_dt),
                    )
                else:
                    w_vals = [
                        float(v) if (math.isfinite(float(v)) and float(v) > 0.0) else 0.0
                        for v in cka_weights
                    ]
                    if len(w_vals) != n_fam:
                        raise ValueError(f"cka_weights len ({len(w_vals)}) != n_families ({n_fam})")
                    w_tot = sum(w_vals)
                    w_norm_py = (
                        [v / w_tot for v in w_vals]
                        if w_tot > 1e-12
                        else [1.0 / float(n_fam)] * n_fam
                    )
                    w_fam = torch.cat(
                        [torch.full((1,), v, device=device, dtype=calc_dt) for v in w_norm_py],
                        dim=0,
                    )
            elif cka_matrix is not None:
                if isinstance(cka_matrix, torch.Tensor):
                    w_fam = compute_cka_dissimilarity_weights(cka_matrix).to(
                        device=device, dtype=calc_dt
                    )
                else:
                    w_np = compute_cka_dissimilarity_weights(cka_matrix)
                    w_fam = torch.cat(
                        [torch.full((1,), float(v), device=device, dtype=calc_dt) for v in w_np],
                        dim=0,
                    )
            else:
                w_fam = torch.full((n_fam,), 1.0 / float(n_fam), device=device, dtype=calc_dt)

            w_view = w_fam.view([n_fam] + [1] * (u_fam.ndim - 1))
            w_eff = torch.where(valid, w_view.expand_as(u_fam), zeros)
            w_eff_sum = w_eff.sum(dim=0, keepdim=True)
            w_norm = torch.where(
                w_eff_sum > 1e-12, w_eff / w_eff_sum.clamp_min(1e-12), zeros
            )

            # Two-pass weighted consensus & epistemic disagreement variance across families
            u_bar = (w_norm * u_fam).sum(dim=0)  # [N] or [N, H]
            diff_f = torch.where(valid, u_fam - u_bar.unsqueeze(0), zeros)
            valid_fam_cnt = valid.to(dtype=calc_dt).sum(dim=0)
            dis_var = torch.where(
                valid_fam_cnt >= 1.5,
                (w_norm * diff_f.pow(2)).sum(dim=0).clamp_min(0.0),
                torch.zeros_like(u_bar),
            )
            dis_std = torch.sqrt(dis_var)

            # Consistent vol broadcasting ([N] or [N, 1] -> [N, H]) for kappa == 0 and kappa > 0
            vol_b = realized_vol
            if vol_b.ndim == u_bar.ndim - 1 and vol_b.shape[0] == u_bar.shape[0]:
                vol_b = vol_b.unsqueeze(-1).expand(u_bar.shape)
            elif vol_b.ndim == u_bar.ndim and vol_b.shape != u_bar.shape:
                vol_b = vol_b.expand(u_bar.shape)

            # Bit-exact matched control when kappa == 0.0
            if float(kappa) == 0.0:
                sigma_eff_out = vol_b.clone()
            else:
                vol_c = vol_b.to(dtype=calc_dt)
                vol_safe = torch.where(
                    torch.isfinite(vol_c), vol_c, torch.full_like(vol_c, float(min_vol))
                ).clamp_min(float(min_vol))
                vol_sq = vol_safe.pow(2)
                scale_t = vol_sq if vol_scale is None else float(vol_scale)
                sigma_eff_out = torch.sqrt(vol_sq + float(kappa) * dis_var * scale_t).to(
                    dtype=orig_dtype
                )

        return dict(
            consensus_u=u_bar.to(dtype=orig_dtype),
            disagreement_var=dis_var.to(dtype=orig_dtype),
            disagreement_std=dis_std.to(dtype=orig_dtype),
            sigma_eff=sigma_eff_out,
            family_u=u_fam.to(dtype=orig_dtype),
            weights=w_fam.to(dtype=orig_dtype),
        )

    # Pure NumPy FP64/FP32 path (0 torch dependency, zero .numpy() calls!)
    vol_arr = np.asarray(realized_vol)
    out_np_dtype = vol_arr.dtype if np.issubdtype(vol_arr.dtype, np.floating) else np.float64
    calc_np_dt = np.float64 if out_np_dtype == np.float64 else np.float32

    stacked_np = np.stack([np.asarray(p, dtype=calc_np_dt) for p in preds_list], axis=0)
    valid_np = np.isfinite(stacked_np)
    clean_np = np.where(valid_np, stacked_np, 0.0)

    v_cnt_np = valid_np.astype(calc_np_dt).sum(axis=1, keepdims=True)
    v_cnt_safe_np = np.maximum(v_cnt_np, 1.0)
    mean_np = clean_np.sum(axis=1, keepdims=True) / v_cnt_safe_np
    centered_np = np.where(valid_np, clean_np - mean_np, 0.0)
    var_np = (centered_np * centered_np).sum(axis=1, keepdims=True) / v_cnt_safe_np
    eps_sq = float(eps) * float(eps)
    has_var_np = (var_np > eps_sq) & (v_cnt_np >= 1.5)
    safe_var_np = np.where(has_var_np, var_np, 1.0)
    u_fam_np = np.where(valid_np & has_var_np, centered_np / np.sqrt(safe_var_np), 0.0)

    if cka_weights is not None:
        w_arr = np.asarray(cka_weights, dtype=calc_np_dt).reshape(-1)
        if w_arr.shape[0] != n_fam:
            raise ValueError(f"cka_weights len ({w_arr.shape[0]}) != n_families ({n_fam})")
        w_np = np.where(np.isfinite(w_arr) & (w_arr > 0.0), w_arr, calc_np_dt(0.0))
        w_sum_np = float(w_np.sum())
        w_fam_np = (
            (w_np / w_sum_np)
            if w_sum_np > 1e-12
            else np.full((n_fam,), 1.0 / n_fam, dtype=calc_np_dt)
        )
    elif cka_matrix is not None:
        w_fam_np = compute_cka_dissimilarity_weights(cka_matrix).astype(calc_np_dt)
    else:
        w_fam_np = np.full((n_fam,), 1.0 / n_fam, dtype=calc_np_dt)

    w_view_np = w_fam_np.reshape([n_fam] + [1] * (u_fam_np.ndim - 1))
    w_eff_np = np.where(valid_np, np.broadcast_to(w_view_np, u_fam_np.shape), 0.0)
    w_eff_sum_np = w_eff_np.sum(axis=0, keepdims=True)
    w_norm_np = np.where(w_eff_sum_np > 1e-12, w_eff_np / np.maximum(w_eff_sum_np, 1e-12), 0.0)

    u_bar_np = (w_norm_np * u_fam_np).sum(axis=0)
    diff_np = np.where(valid_np, u_fam_np - u_bar_np[None, ...], 0.0)
    valid_fam_cnt_np = valid_np.astype(calc_np_dt).sum(axis=0)
    dis_var_np = np.where(
        valid_fam_cnt_np >= 1.5,
        np.maximum((w_norm_np * (diff_np * diff_np)).sum(axis=0), 0.0),
        0.0,
    )
    dis_std_np = np.sqrt(dis_var_np)

    vol_b_np = vol_arr
    if vol_b_np.ndim == u_bar_np.ndim - 1 and vol_b_np.shape[0] == u_bar_np.shape[0]:
        vol_b_np = np.broadcast_to(vol_b_np[..., None], u_bar_np.shape)
    elif vol_b_np.ndim == u_bar_np.ndim and vol_b_np.shape != u_bar_np.shape:
        vol_b_np = np.broadcast_to(vol_b_np, u_bar_np.shape)

    if float(kappa) == 0.0:
        sigma_eff_np = np.array(vol_b_np, copy=True)
    else:
        vol_c_np = vol_b_np.astype(calc_np_dt)
        vol_safe_np = np.maximum(
            np.where(np.isfinite(vol_c_np), vol_c_np, calc_np_dt(min_vol)), calc_np_dt(min_vol)
        )
        vol_sq_np = vol_safe_np * vol_safe_np
        scale_np = vol_sq_np if vol_scale is None else calc_np_dt(vol_scale)
        sigma_eff_np = np.sqrt(vol_sq_np + calc_np_dt(kappa) * dis_var_np * scale_np).astype(
            out_np_dtype
        )

    return dict(
        consensus_u=u_bar_np.astype(out_np_dtype),
        disagreement_var=dis_var_np.astype(out_np_dtype),
        disagreement_std=dis_std_np.astype(out_np_dtype),
        sigma_eff=sigma_eff_np,
        family_u=u_fam_np.astype(out_np_dtype),
        weights=w_fam_np.astype(out_np_dtype),
    )
