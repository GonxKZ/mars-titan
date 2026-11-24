"""Tests for Wave-3B Frontier Deep Learning & Stage-2 Modules (`frontier_modules_sota.py`).

Covers all 4 adversarial-refereed deliverables (`#194` + `#195` P0 guards):
  1. `PolarAlphaHead`, `polar_alpha_calibration_loss`, `polar_alpha_order_loss`,
     `polar_alpha_persistence_loss`, `align_by_asset_ids` (P0-1a/1b/1c anti-FM-7 cross-detach).
  2. `ObservedSlotEncoder` & `MissingnessAwareSetModalityFusion` (P0-2a/2b sparse CN gain
     stabilization, bit-exact 0.0 sensitivity/grad on NaN/±1e6 absent slots, MNAR placebo).
  3. `DiffInducedFactorCrossAttention` (P0-3a/3b/3c Ye et al. ICLR 2025 differential cross-asset
     attention over R=16 landmarks, Gemma-2 soft-cap in FP32, bounded lambda in (0, 2),
     zero-lambda matched control, permutation equivariance, common-mode rejection, N=2600).
  4. `compute_cross_family_disagreement_risk` & `compute_cka_dissimilarity_weights` (#5C Stage-2
     epistemic disagreement risk feeding P+ with bit-exact kappa=0.0 control).
  5. Global Blackwell/Dynamo gates: `audit_zero_host_syncs == 0`, 0 `torch.tensor(` outside
     `__init__`, BF16 autocast & FP64 parity, `torch.compile(mode="default", dynamic=True)`.
"""

from __future__ import annotations

import ast
import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mars_titan.evaluation.metacognitive_agent import (
    ConvexPortfolioOptimizerPPlus,
    DifferentiablePortfolioLayer,
)
from mars_titan.models.mmarrs.blackwell_kernels import audit_zero_host_syncs
from mars_titan.models.mmarrs.frontier_modules_sota import (
    DiffInducedFactorCrossAttention,
    MissingnessAwareSetModalityFusion,
    ObservedSlotEncoder,
    PolarAlphaHead,
    align_by_asset_ids,
    compute_cka_dissimilarity_weights,
    compute_cross_family_disagreement_risk,
    polar_alpha_calibration_loss,
    polar_alpha_order_loss,
    polar_alpha_persistence_loss,
)
from mars_titan.models.mmarrs.mor_loom_pc import InducedFactorCrossAttention

_SRC_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "mars_titan"
    / "models"
    / "mmarrs"
    / "frontier_modules_sota.py"
)


# ---------------------------------------------------------------------------
# 1. PolarAlphaHead + Cross-Detached Calibration / Order / Persistence Losses
# ---------------------------------------------------------------------------


def test_polar_alpha_head_init_scale_param_budget_and_exact_zero_mean() -> None:
    torch.manual_seed(7)
    head = PolarAlphaHead(n_horizons=6, init_gamma=0.15)
    n_params = sum(p.numel() for p in head.parameters())
    assert n_params < 30, f"Expected <30 params, got {n_params}"

    # softplus(raw_gamma) initialized to ~0.15
    gamma = torch.nn.functional.softplus(head.raw_gamma)
    assert torch.allclose(gamma, torch.full_like(gamma, 0.15), atol=1e-5)

    # Skewed raw_alpha + heterogeneous sigma_det ~= 0.015 -> exact zero-mean alpha_cal ~ O(0.001)
    n_assets = 256
    raw_alpha = torch.randn(n_assets, 6) * 2.5 + 1.7
    sigma_det = torch.full((n_assets, 6), 0.015) * (0.7 + 0.6 * torch.rand(n_assets, 6))
    out = head(raw_alpha, vol=sigma_det)

    alpha_cal = out["alpha_cal"]
    assert alpha_cal.shape == (n_assets, 6)
    assert torch.equal(out["alpha_cal_for_cal"].detach(), alpha_cal.detach())
    assert torch.equal(out["alpha_cal_for_pplus"].detach(), alpha_cal.detach())
    # P0-1b: exact zero-mean cross-sectionally (< 1e-6)
    assert float(alpha_cal.detach().mean(dim=0).abs().max()) < 1e-6
    # Realistic Grinold-Kahn calibration magnitude around l1_cost = 0.001
    cal_std = float(alpha_cal.detach().std(dim=0).mean())
    assert 3e-4 <= cal_std <= 3e-3, f"Expected std(alpha_cal) ~ O(0.001), got {cal_std:.6f}"

    # Optional systemic factor bias only when explicitly requested
    sys_bias = torch.full((n_assets, 6), 0.002)
    out_sys = head(raw_alpha, vol=sigma_det, systemic_bias=sys_bias, include_systemic_bias=True)
    assert torch.allclose(out_sys["alpha_cal"], alpha_cal + sys_bias, atol=1e-6)


def test_polar_alpha_head_cross_detachment_anti_fm7_guards() -> None:
    """Verify P0-1c: L_cal only trains raw_gamma; P+ loss_sharpe only trains raw_alpha."""
    torch.manual_seed(11)
    n_assets = 96
    head = PolarAlphaHead(n_horizons=3, init_gamma=0.15)
    raw_alpha = (torch.randn(n_assets, 3) * 0.5).requires_grad_(True)
    vol = (0.01 + 0.02 * torch.rand(n_assets)).requires_grad_(True)
    y_ret = 0.004 * raw_alpha.detach() + 0.015 * torch.randn(n_assets, 3)

    # (a) L_cal must have exact 0.0 grad w.r.t. raw_alpha (and vol) & non-zero grad w.r.t. gamma
    out = head(raw_alpha, vol=vol)
    loss_cal = polar_alpha_calibration_loss(out["alpha_cal_for_cal"], y_ret)
    g_raw, g_gamma, g_vol = torch.autograd.grad(
        loss_cal, (raw_alpha, head.raw_gamma, vol), allow_unused=True
    )
    assert g_raw is None or torch.equal(g_raw, torch.zeros_like(raw_alpha))
    assert g_vol is None or torch.equal(g_vol, torch.zeros_like(vol))
    assert g_gamma is not None and float(g_gamma.abs().sum()) > 1e-6

    # (b) P+ Sharpe loss on alpha_cal_for_pplus must have exact 0.0 grad w.r.t. raw_gamma
    pplus = DifferentiablePortfolioLayer(
        steps=6, risk_aversion=2.0, factor_penalty=4.0, l1_cost=0.001, gross_limit=1.0
    )
    exposures = torch.randn(n_assets, 4)
    w_t = pplus(out["alpha_cal_for_pplus"][:, 0], vol=vol.detach(), exposures=exposures)
    gross = (w_t * y_ret[:, 0]).sum()
    net = gross - 0.001 * w_t.abs().sum()
    port_vol = torch.sqrt((w_t * vol.detach()).pow(2).sum() + 1e-8)
    loss_sharpe = -net / (port_vol + 1e-4)

    g_raw_p, g_gamma_p = torch.autograd.grad(
        loss_sharpe, (raw_alpha, head.raw_gamma), allow_unused=True
    )
    assert g_gamma_p is None or torch.equal(g_gamma_p, torch.zeros_like(head.raw_gamma))
    assert g_raw_p is not None and float(g_raw_p[:, 0].abs().sum()) > 1e-6


def test_polar_alpha_scale_invariance_edge_cases_and_mz_calibration() -> None:
    """Verify scale invariance under raw_alpha * 100, N=1/constant safety, and MZ calibration."""
    torch.manual_seed(17)
    head = PolarAlphaHead(n_horizons=2, init_gamma=0.15)
    raw = torch.randn(128, 2) * 0.25 + 0.1
    vol = torch.full((128, 2), 0.02)

    out_1 = head(raw, vol=vol)
    out_100 = head(raw * 100.0, vol=vol)
    out_shift = head(raw + 42.0, vol=vol)
    assert torch.allclose(out_1["u"], out_100["u"], atol=1e-5, rtol=1e-5)
    # Adding 42.0 to FP32 values of std=0.25 loses ~7 mantissa bits on the input itself (~1e-4)
    assert torch.allclose(out_1["u"], out_shift["u"], atol=2e-4, rtol=2e-4)
    assert torch.allclose(out_1["alpha_cal"], out_100["alpha_cal"], atol=1e-6)

    # Double-precision exact scale + shift invariance (< 1e-12)
    head_64 = PolarAlphaHead(n_horizons=2, init_gamma=0.15).double()
    raw_64 = raw.double()
    vol_64 = vol.double()
    o64_1 = head_64(raw_64, vol=vol_64)
    o64_100 = head_64(raw_64 * 100.0, vol=vol_64)
    o64_shift = head_64(raw_64 + 42.0, vol=vol_64)
    assert torch.allclose(o64_1["u"], o64_100["u"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(o64_1["u"], o64_shift["u"], atol=1e-12, rtol=1e-12)

    # Edge cases: N=1 and constant raw_alpha -> finite 0.0 output & exact 0.0 grad
    raw_1 = torch.tensor([[3.5, -1.2]], requires_grad=True)
    out_n1 = head(raw_1, vol=torch.tensor([[0.02, 0.02]]))
    assert torch.isfinite(out_n1["u"]).all() and torch.all(out_n1["u"] == 0.0)
    assert torch.isfinite(out_n1["alpha_cal"]).all() and torch.all(out_n1["alpha_cal"] == 0.0)
    (out_n1["u"].sum() + out_n1["alpha_cal_for_pplus"].sum()).backward()
    assert raw_1.grad is not None and torch.isfinite(raw_1.grad).all()

    raw_const = torch.full((32, 2), 4.2, requires_grad=True)
    out_c = head(raw_const, vol=torch.full((32, 2), 0.02))
    assert torch.isfinite(out_c["u"]).all() and torch.all(out_c["u"] == 0.0)
    out_c["alpha_cal_for_pplus"].sum().backward()
    assert raw_const.grad is not None and torch.isfinite(raw_const.grad).all()

    # Calibration loop drives Mincer-Zarnowitz slope toward ~1.0 without moving u
    head_cal = PolarAlphaHead(n_horizons=1, init_gamma=0.05)
    opt = torch.optim.Adam(head_cal.parameters(), lr=0.1)
    true_u = torch.randn(512, 1)
    vol_c = torch.full((512, 1), 0.02)
    u_true_norm = (true_u - true_u.mean(dim=0, keepdim=True)) / true_u.std(
        dim=0, unbiased=False, keepdim=True
    )
    y_true = 0.02 * 0.40 * torch.tanh(u_true_norm / 3.0)
    y_true = y_true - y_true.mean(dim=0, keepdim=True)
    for _ in range(120):
        opt.zero_grad()
        o = head_cal(true_u, vol=vol_c)
        loss = polar_alpha_calibration_loss(o["alpha_cal_for_cal"], y_true, delta=0.05)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = head_cal(true_u, vol=vol_c)["alpha_cal"][:, 0]
        tgt = y_true[:, 0]
        p_c = pred - pred.mean()
        t_c = tgt - tgt.mean()
        mz_slope = float((p_c * t_c).sum() / p_c.pow(2).sum())
    assert 0.85 <= mz_slope <= 1.15, f"Expected MZ slope ~1.0, got {mz_slope:.4f}"


def test_polar_alpha_order_loss_ragged_persistence_and_bf16_nan_safety() -> None:
    """Verify P0-1a tail_listnet reuse, ragged asset_ids alignment, BF16 parity, NaN 0-grad."""
    torch.manual_seed(23)
    # 1) Ragged asset_ids alignment (overlapping, shuffled, IPOs, delistings)
    prev_ids = torch.tensor([105, 101, 109, 102], dtype=torch.long)
    prev_u = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float32)
    cur_ids = torch.tensor([101, 999, 105, 888, 102], dtype=torch.long)
    aligned, matched = align_by_asset_ids(cur_ids, prev_ids, prev_u)
    assert torch.equal(matched, torch.tensor([True, False, True, False, True]))
    assert torch.allclose(aligned, torch.tensor([-2.0, 0.0, 1.0, 0.0, 3.0]))

    # Persistence loss: identical aligned signal -> ~0.0; opposite aligned -> ~2.0 * w_churn
    u_same = torch.tensor([-2.0, 5.0, 1.0, -7.0, 3.0], requires_grad=True)
    l_same = polar_alpha_persistence_loss(
        u_same, prev_u, cur_asset_ids=cur_ids, prev_asset_ids=prev_ids, w_churn=0.25
    )
    assert float(l_same.detach()) < 1e-6

    u_opp = torch.tensor([2.0, 5.0, -1.0, -7.0, -3.0], requires_grad=True)
    l_opp = polar_alpha_persistence_loss(
        u_opp, prev_u, cur_asset_ids=cur_ids, prev_asset_ids=prev_ids, w_churn=0.25
    )
    assert math.isclose(float(l_opp.detach()), 0.50, rel_tol=1e-5)
    l_opp.backward()
    assert u_opp.grad is not None
    # Unmatched IPO indices (1, 3) receive exact 0.0 gradient!
    assert float(u_opp.grad[1]) == 0.0 and float(u_opp.grad[3]) == 0.0

    # Zero overlap returns exact 0.0 with finite grad
    l_no_ov = polar_alpha_persistence_loss(
        u_same.detach().requires_grad_(True),
        prev_u,
        cur_asset_ids=torch.tensor([500, 501, 502, 503, 504]),
        prev_asset_ids=prev_ids,
        w_churn=0.25,
    )
    assert float(l_no_ov.detach()) == 0.0

    # 2) Order loss ordering & NaN target zero-grad-leak (both direct AND via masked head)
    n_assets = 80
    y = torch.randn(n_assets, 2) * 0.02
    u_good = (y / 0.02 + 0.1 * torch.randn_like(y)).requires_grad_(True)
    u_bad = (-y / 0.02).requires_grad_(True)
    exp_m = torch.randn(n_assets, 3)
    l_good = polar_alpha_order_loss(u_good, y, exposures=exp_m)
    l_bad = polar_alpha_order_loss(u_bad, y, exposures=exp_m)
    assert float(l_good.detach()) < float(l_bad.detach())

    # Inject NaNs into targets for rows 0..9 -> exact 0.0 grad on rows 0..9 in BOTH losses!
    y_nan = y.clone()
    y_nan[:10, :] = float("nan")

    # (i) Direct loss gradient zero-leak check on u & alpha_cal
    u_direct = torch.randn(n_assets, 2, requires_grad=True)
    cal_direct = (torch.randn(n_assets, 2) * 0.002).requires_grad_(True)
    l_ord_d = polar_alpha_order_loss(u_direct, y_nan)
    l_cal_d = polar_alpha_calibration_loss(cal_direct, y_nan)
    (l_ord_d + l_cal_d).backward()
    assert u_direct.grad is not None and torch.all(u_direct.grad[:10] == 0.0)
    assert cal_direct.grad is not None and torch.all(cal_direct.grad[:10] == 0.0)
    assert float(u_direct.grad[10:].abs().sum()) > 1e-6
    assert float(cal_direct.grad[10:].abs().sum()) > 1e-6

    # (ii) End-to-end PolarAlphaHead with asset_mask=isfinite(y_nan) -> exact 0.0 grad on rows 0..9
    raw_in = torch.randn(n_assets, 2, requires_grad=True)
    head = PolarAlphaHead(n_horizons=2, init_gamma=0.15)
    out = head(raw_in, vol=torch.full((n_assets, 2), 0.02), asset_mask=torch.isfinite(y_nan))
    loss_ord_nan = polar_alpha_order_loss(out["u"], y_nan)
    loss_cal_nan = polar_alpha_calibration_loss(out["alpha_cal_for_cal"], y_nan)
    assert torch.isfinite(loss_ord_nan) and torch.isfinite(loss_cal_nan)
    (loss_ord_nan + loss_cal_nan).backward()
    assert raw_in.grad is not None and torch.isfinite(raw_in.grad).all()
    assert head.raw_gamma.grad is not None and torch.isfinite(head.raw_gamma.grad).all()
    assert torch.all(raw_in.grad[:10] == 0.0)

    # 3) BF16 autocast exactness & FP64 reference parity
    raw_64 = torch.randn(64, 2, dtype=torch.float64) * 1.5
    vol_64 = torch.full((64, 2), 0.018, dtype=torch.float64)
    y_64 = torch.randn(64, 2, dtype=torch.float64) * 0.02
    head_64 = PolarAlphaHead(n_horizons=2, init_gamma=0.15).double()
    head_32 = PolarAlphaHead(n_horizons=2, init_gamma=0.15)
    with torch.no_grad():
        head_32.raw_gamma.copy_(head_64.raw_gamma.float())

    out_64 = head_64(raw_64, vol=vol_64)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_bf16 = head_32(raw_64.bfloat16(), vol=vol_64.bfloat16())
        l_cal_bf16 = polar_alpha_calibration_loss(out_bf16["alpha_cal_for_cal"], y_64.bfloat16())
        l_ord_bf16 = polar_alpha_order_loss(out_bf16["u"], y_64.bfloat16())

    assert torch.isfinite(out_bf16["alpha_cal"]).all()
    assert float(out_bf16["alpha_cal"].detach().mean(dim=0).abs().max()) < 1e-6
    assert torch.allclose(out_bf16["alpha_cal"].double(), out_64["alpha_cal"], atol=1e-4, rtol=5e-2)
    assert torch.isfinite(l_cal_bf16) and torch.isfinite(l_ord_bf16)


def test_polar_alpha_adversarial_regressions_crash_day_exposures_liq_and_churn() -> None:
    """Regressions for #202 P1-1/2/3 & #203 P1-1/2: crash-day Huber, masked B, lw scale, churn."""
    torch.manual_seed(101)
    n_assets, n_h = 64, 2
    head = PolarAlphaHead(n_horizons=n_h, init_gamma=0.15)
    raw = torch.randn(n_assets, n_h)
    vol = torch.full((n_assets, n_h), 0.02)
    y_idio = 0.004 * torch.tanh(raw / 3.0) + 0.01 * torch.randn(n_assets, n_h)
    y_idio = y_idio - y_idio.mean(dim=0, keepdim=True)

    # 1) #203 P1-2: -5% market crash day must NOT zero out dL_cal/d(raw_gamma)!
    out_calm = head(raw, vol=vol)
    l_calm = polar_alpha_calibration_loss(out_calm["alpha_cal_for_cal"], y_idio)
    (g_calm,) = torch.autograd.grad(l_calm, (head.raw_gamma,))

    out_crash = head(raw, vol=vol)
    l_crash = polar_alpha_calibration_loss(out_crash["alpha_cal_for_cal"], y_idio - 0.05)
    (g_crash,) = torch.autograd.grad(l_crash, (head.raw_gamma,))
    assert float(g_crash.abs().sum()) > 1e-7
    assert torch.allclose(g_crash, g_calm, atol=1e-6, rtol=1e-5)

    # 2) #202 P1-1 & #203 P1-1a: masked/NaN rows in exposures must have exact 0.0 effect on loss_ic
    u = torch.randn(n_assets, n_h, requires_grad=True)
    y_nan = y_idio.clone()
    y_nan[:8, :] = float("nan")
    a_mask = torch.ones(n_assets, dtype=torch.bool)
    a_mask[:5] = False
    exp_clean = torch.randn(n_assets, 4)
    exp_poison = exp_clean.clone()
    exp_poison[:8, :] = 1e4

    l_exp_clean = polar_alpha_order_loss(
        u, y_nan, exposures=exp_clean, asset_mask=a_mask, w_listnet=0.0
    )
    l_exp_poison = polar_alpha_order_loss(
        u, y_nan, exposures=exp_poison, asset_mask=a_mask, w_listnet=0.0
    )
    assert float((l_exp_poison - l_exp_clean).detach().abs()) < 1e-6

    # 3) #202 P1-2: liquidity_weights scale invariance under lw * 1e-6
    lw = torch.exp(torch.randn(n_assets))
    l_lw_1 = polar_alpha_order_loss(u.detach(), y_idio, liquidity_weights=lw)
    l_lw_tiny = polar_alpha_order_loss(u.detach(), y_idio, liquidity_weights=lw * 1e-6)
    assert float((l_lw_1 - l_lw_tiny).abs()) < 1e-5

    # 4) #202 P1-3 & #203 P1-1b: NaN y_ret targets must NOT distort L_churn when u == u_prev,
    #    and multi-horizon L_churn tracks all active horizons!
    u_same = torch.randn(n_assets, n_h, requires_grad=True)
    l_churn_nan = polar_alpha_order_loss(
        u_same,
        y_nan,
        w_ic=0.0,
        w_listnet=0.0,
        u_prev=u_same.detach(),
        w_churn=0.5,
    )
    assert float(l_churn_nan.detach()) < 1e-6

    # Multi-horizon sensitivity: flipping horizon 1 (while keeping horizon 0 identical) > 0
    u_prev_mh = u_same.detach().clone()
    u_prev_mh[:, 1] = -u_prev_mh[:, 1]
    l_mh = polar_alpha_persistence_loss(u_same, u_prev_mh, w_churn=0.5)
    assert float(l_mh.detach()) > 0.45


# ---------------------------------------------------------------------------
# 2. ObservedSlotEncoder & MissingnessAwareSetModalityFusion
# ---------------------------------------------------------------------------


def test_observed_slot_encoder_eliminates_cn_sparse_gain_blowup_and_isolates_nans() -> None:
    """Verify P0-2a: 99.2% CN missingness bounded norm (no 3x blowup) & bit-exact NaN 0-grad."""
    torch.manual_seed(31)
    n_slots = 26  # encoded_v31_78 (26 vals + 26 obs + 26 log_ages)
    hidden_dim = 64
    enc = ObservedSlotEncoder(n_slots=n_slots, hidden_dim=hidden_dim, max_staleness_days=600.0)

    # Compare dense (26/26), sparse CN (3/26), & 99.2%-missing CN (250/252 missing + 2 sparse)
    n_assets = 252
    vals = torch.randn(n_assets, n_slots)
    obs = torch.zeros(n_assets, n_slots)
    obs[0, :] = 1.0          # dense 26/26
    obs[1, :3] = 1.0         # sparse 3/26
    obs[2, :3] = 1.0         # sparse 3/26 (2/252 ~= 0.8% observed, 99.2% missing!)
    log_ages = torch.where(obs > 0.5, torch.full_like(obs, math.log1p(30.0)), torch.zeros_like(obs))
    triplet_clean = torch.cat([vals * obs, obs, log_ages], dim=-1)

    e_clean = enc(triplet_clean)
    assert e_clean.shape == (n_assets, hidden_dim)
    assert torch.isfinite(e_clean).all()

    # Bounded output norm across dense (26/26), sparse (3/26), and fully missing (0/26)
    rms_dense = float(e_clean[0].detach().pow(2).mean().sqrt())
    rms_sparse = float(e_clean[1].detach().pow(2).mean().sqrt())
    rms_missing = float(e_clean[3:].detach().pow(2).mean(dim=-1).sqrt().max())
    assert 0.5 <= rms_dense <= 1.5
    assert 0.5 <= rms_sparse <= 1.5
    assert rms_missing <= 1.5
    # Verify NO 3x sparse gain blowup (sparse/dense ratio tightly near 1.0)
    assert abs(rms_sparse / max(rms_dense, 1e-6) - 1.0) < 0.25

    # Contrast: raw LayerNorm(78) blows up the 3 observed slots by ~2-3x relative to dense!
    ln78 = nn.LayerNorm(78, elementwise_affine=False)
    ln_out = ln78(triplet_clean)
    assert float(ln_out[1, :3].abs().mean()) > 1.5 * float(ln_out[0, :3].abs().mean())

    # Bit-exact 0.0 sensitivity AND exact 0.0 grad to NaN / ±1e6 in unobserved slots!
    vals_corrupt = vals.clone()
    vals_corrupt[1, 3:15] = float("nan")
    vals_corrupt[1, 15:] = 1e6
    vals_corrupt[3:, :] = -1e6
    vals_corrupt[4, :] = float("nan")
    vals_corrupt.requires_grad_(True)
    log_ages_corrupt = log_ages.clone()
    log_ages_corrupt[1, 3:] = float("nan")

    e_corrupt = enc(vals_corrupt, obs_mask=obs, log_ages=log_ages_corrupt)
    assert torch.equal(e_corrupt.detach(), e_clean.detach())
    e_corrupt.sum().backward()
    assert vals_corrupt.grad is not None
    assert torch.isfinite(vals_corrupt.grad).all()
    assert torch.all(vals_corrupt.grad[obs < 0.5] == 0.0)
    assert float(vals_corrupt.grad[obs > 0.5].abs().sum()) > 1e-6

    # Also test encoded_v31_420 (140 * 3) + BF16 autocast
    enc_420 = ObservedSlotEncoder(n_slots=140, hidden_dim=32)
    tri_420 = torch.cat(
        [torch.randn(16, 140), (torch.rand(16, 140) > 0.8).float(), torch.rand(16, 140) * 3.0],
        dim=-1,
    )
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_420 = enc_420(tri_420.bfloat16())
    assert out_420.shape == (16, 32) and torch.isfinite(out_420).all()


def test_missingness_aware_set_modality_fusion_bit_exact_isolation_and_mnar_placebo() -> None:
    """Verify P0-2b: exact 0.0 NaN/±1e6 sensitivity & grad, 100%-missing stability, MNAR placebo."""
    torch.manual_seed(41)
    n_assets, n_aux, d = 64, 4, 32
    fusion = MissingnessAwareSetModalityFusion(hidden_dim=d, n_aux_modalities=n_aux, n_heads=4)

    core_emb = torch.randn(n_assets, d)
    mod_tokens_clean = torch.randn(n_assets, n_aux, d)
    mod_mask = (torch.rand(n_assets, n_aux) > 0.5).float()
    # Force row 0..7 to have 100% missing aux modalities, row 8..15 all present
    mod_mask[:8, :] = 0.0
    mod_mask[8:16, :] = 1.0
    mod_age = torch.rand(n_assets, n_aux) * 60.0

    # Zero out missing tokens cleanly as reference
    tokens_ref = mod_tokens_clean * mod_mask.unsqueeze(-1)
    fused_ref, diag_ref = fusion(tokens_ref, mod_mask, core_emb, modality_age=mod_age)
    assert fused_ref.shape == (n_assets, d)
    assert torch.isfinite(fused_ref).all()

    # (c) 100% missing aux modalities -> exact 1.0 attention weight on core self-token anchor
    attn_w = diag_ref["attn_weights"]  # [N, M + 1]
    assert torch.allclose(attn_w[:8, 0], torch.ones(8), atol=1e-6)
    assert torch.allclose(attn_w[:8, 1:], torch.zeros(8, n_aux), atol=1e-6)

    # (b) Masked-out values & ages set to NaN / ±1e6 -> bit-exact 0.0 diff & exact 0.0 grad!
    tokens_poison = mod_tokens_clean.clone()
    age_poison = mod_age.clone()
    miss_bool = mod_mask < 0.5
    tokens_poison[miss_bool] = 1e6
    half_miss = miss_bool.clone()
    half_miss[n_assets // 2 :] = False
    tokens_poison[half_miss] = float("nan")
    age_poison[miss_bool] = float("nan")
    tokens_poison.requires_grad_(True)

    fused_poison, _ = fusion(tokens_poison, mod_mask, core_emb, modality_age=age_poison)
    max_abs_diff = float((fused_poison.detach() - fused_ref.detach()).abs().max())
    assert max_abs_diff == 0.0, f"Expected bit-exact 0.0 diff, got {max_abs_diff}"

    fused_poison.sum().backward()
    assert tokens_poison.grad is not None
    assert torch.isfinite(tokens_poison.grad).all()
    assert torch.all(tokens_poison.grad[miss_bool] == 0.0)
    assert float(tokens_poison.grad[~miss_bool].abs().sum()) > 1e-6

    # (d) MNAR placebo shuffle changes output while staying bit-exact safe on missing KV NaNs
    fusion_placebo = MissingnessAwareSetModalityFusion(
        hidden_dim=d, n_aux_modalities=n_aux, n_heads=4, mnar_placebo_shuffle=True
    )
    fusion_placebo.load_state_dict(fusion.state_dict())
    gen = torch.Generator().manual_seed(99)
    fused_pl, _ = fusion_placebo(
        tokens_poison.detach(), mod_mask, core_emb, modality_age=age_poison, generator=gen
    )
    assert torch.isfinite(fused_pl).all()
    assert not torch.allclose(fused_pl, fused_ref, atol=1e-5)

    # Schema-agnostic M=5 support + BF16 autocast parity vs FP64 reference
    fusion_m5_64 = MissingnessAwareSetModalityFusion(
        hidden_dim=d, n_aux_modalities=5, n_heads=4
    ).double()
    fusion_m5_32 = MissingnessAwareSetModalityFusion(hidden_dim=d, n_aux_modalities=5, n_heads=4)
    fusion_m5_32.load_state_dict({k: v.float() for k, v in fusion_m5_64.state_dict().items()})
    t5 = torch.randn(32, 5, d, dtype=torch.float64)
    m5 = (torch.rand(32, 5) > 0.3).to(dtype=torch.float64)
    c5 = torch.randn(32, d, dtype=torch.float64)
    out5_64, _ = fusion_m5_64(t5, m5, c5)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out5_bf16, _ = fusion_m5_32(t5.bfloat16(), m5.bfloat16(), c5.bfloat16())
    assert torch.isfinite(out5_bf16).all()
    assert torch.allclose(out5_bf16.double(), out5_64, atol=5e-2, rtol=5e-2)

    # #202 P1-4 & #203 P0-2a: NaN inside a PRESENT modality token must not poison row output/grad
    tok_pres_nan = mod_tokens_clean.clone()
    tok_pres_nan[8, 0, 0] = float("nan")  # row 8 has mod_mask == 1.0 on all aux modalities
    tok_pres_nan.requires_grad_(True)
    fused_pnan, diag_pnan = fusion(tok_pres_nan, mod_mask, core_emb, modality_age=mod_age)
    assert torch.isfinite(fused_pnan).all()
    assert float(diag_pnan["attn_weights"][8, 1]) == 0.0  # corrupted token excluded cleanly
    fused_pnan.sum().backward()
    assert tok_pres_nan.grad is not None and torch.isfinite(tok_pres_nan.grad).all()
    assert torch.all(tok_pres_nan.grad[8, 0] == 0.0)

    # #203 P0-2b: 90d US 10-Q & 180d CN filings must NOT starve at init (retain > 0.05 weight!)
    fusion_age = MissingnessAwareSetModalityFusion(hidden_dim=d, n_aux_modalities=4, n_heads=4)
    n_ag = 32
    tok_ag = torch.randn(n_ag, 4, d)
    mask_ag = torch.ones(n_ag, 4)
    core_ag = torch.randn(n_ag, d)
    # Columns: [fund_90d, fund_180d, news_1d, chart_1d]
    ages_ag = torch.tensor([[90.0, 180.0, 1.0, 1.0]]).expand(n_ag, 4)
    _, diag_ag = fusion_age(tok_ag, mask_ag, core_ag, modality_age=ages_ag)
    mean_w = diag_ag["attn_weights"].mean(dim=0)  # [anchor, fund_90d, fund_180d, news_1d, chart_1d]
    assert float(mean_w[1]) > 0.05, f"90d US fundamental starved at init: weight={float(mean_w[1])}"
    assert float(mean_w[2]) > 0.05, f"180d CN filing starved at init: weight={float(mean_w[2])}"


# ---------------------------------------------------------------------------
# 3. DiffInducedFactorCrossAttention (Ye et al. ICLR 2025 + Gemma-2 SoftCap)
# ---------------------------------------------------------------------------


def test_diff_induced_cross_attention_dropin_perm_equivariance_and_mask_hygiene() -> None:
    """Verify F3 drop-in API, ~0% QKV overhead, permutation equivariance, F-invariance & mask."""
    torch.manual_seed(53)
    d, r_land, n_heads = 64, 16, 4
    base_attn = InducedFactorCrossAttention(d, n_landmarks=r_land, n_heads=n_heads, context_dim=8)
    diff_attn = DiffInducedFactorCrossAttention(
        d, n_landmarks=r_land, n_heads=n_heads, context_dim=8
    )

    # ~0% extra QKV params vs InducedFactorCrossAttention (only small GroupRMSNorm + lambdas)
    base_params = sum(p.numel() for p in base_attn.parameters())
    diff_params = sum(p.numel() for p in diff_attn.parameters())
    overhead = (diff_params - base_params) / float(base_params)
    assert overhead < 0.02, f"Expected <2% param overhead, got {overhead:.4%}"

    # (a) Exact permutation equivariance across stocks + F-invariance
    n_assets = 48
    h = torch.randn(n_assets, d)
    exposures = torch.randn(n_assets, 5)
    rmt_loadings = torch.randn(n_assets, 3)
    step_emb = torch.randn(d)
    regime_vec = torch.randn(8)
    asset_mask = torch.ones(n_assets, dtype=torch.bool)
    asset_mask[-6:] = False  # 6 padded assets

    out_ref, diag_ref = diff_attn(
        h,
        asset_mask=asset_mask,
        exposures=exposures,
        rmt_loadings=rmt_loadings,
        step_emb=step_emb,
        regime_vec=regime_vec,
    )
    assert out_ref.shape == (n_assets, d)
    assert "landmark_attn_entropy" in diag_ref and torch.isfinite(diag_ref["landmark_attn_entropy"])

    perm = torch.randperm(n_assets)
    out_perm, _ = diff_attn(
        h[perm],
        asset_mask=asset_mask[perm],
        exposures=exposures[perm],
        rmt_loadings=rmt_loadings[perm],
        step_emb=step_emb,
        regime_vec=regime_vec,
    )
    max_perm_err = float((out_perm.detach() - out_ref.detach()[perm]).abs().max())
    assert max_perm_err < 1e-5, f"Permutation equivariance error {max_perm_err:.2e} >= 1e-5"

    # Strict asset_mask padding hygiene: NaN/±1e6 in padded rows has exact 0.0 impact on valid!
    h_poison = h.clone()
    h_poison[-6:] = float("nan")
    h_poison[-3:] = 1e6
    exp_poison = exposures.clone()
    exp_poison[-6:] = float("nan")
    out_poison, _ = diff_attn(
        h_poison,
        asset_mask=asset_mask,
        exposures=exp_poison,
        rmt_loadings=rmt_loadings,
        step_emb=step_emb,
        regime_vec=regime_vec,
    )
    assert torch.isfinite(out_poison).all()
    assert torch.equal(out_poison.detach()[:-6], out_ref.detach()[:-6])
    assert torch.all(out_poison.detach()[-6:] == 0.0)

    # F-invariance: works seamlessly with arbitrary factor column count F=12
    out_f12, _ = diff_attn(h, exposures=torch.randn(n_assets, 12))
    assert out_f12.shape == (n_assets, d) and torch.isfinite(out_f12).all()

    # All-False asset_mask stays strictly finite and exact 0.0; lambda_init >= 1.0 rejected
    out_empty, diag_empty = diff_attn(h, asset_mask=torch.zeros(n_assets, dtype=torch.bool))
    assert torch.isfinite(out_empty).all() and torch.all(out_empty == 0.0)
    assert torch.isfinite(diag_empty["landmark_attn_entropy"])
    with pytest.raises(ValueError, match="lambda_init"):
        DiffInducedFactorCrossAttention(d, n_landmarks=r_land, n_heads=n_heads, lambda_init=1.0)


def test_diff_induced_cross_attention_common_mode_rejection_and_zero_lambda_control() -> None:
    """Verify (b) common-mode suppression vs softmax and (c) zero_lambda_control behavior."""
    torch.manual_seed(61)
    d, r_land, n_heads = 64, 16, 4
    diff_active = DiffInducedFactorCrossAttention(
        d, n_landmarks=r_land, n_heads=n_heads, lambda_init=0.85, init_gate=1.0
    )
    diff_ctrl = DiffInducedFactorCrossAttention(
        d,
        n_landmarks=r_land,
        n_heads=n_heads,
        lambda_init=0.85,
        init_gate=1.0,
        zero_lambda_control=True,
    )
    diff_ctrl.load_state_dict(diff_active.state_dict())

    # (c) Verify zero_lambda_control exact behavior
    n_assets = 96
    x = torch.randn(n_assets, d)
    out_ctrl, diag_ctrl = diff_ctrl(x)
    assert float(diag_ctrl["diff_lambda_mean"]) == 0.0
    out_ctrl.sum().backward()
    for name, p in diff_ctrl.named_parameters():
        if "raw_lambda" in name:
            assert p.grad is None or torch.all(p.grad == 0.0)
    diff_ctrl.zero_grad()

    # (b) Common-mode shift rejection: x_shifted = x + c (same shift added to all N assets)
    c_shift = torch.randn(1, d) * 3.0
    with torch.no_grad():
        # CMRR drift ratio = ||mean(delta(x+c) - delta(x))|| / ||std(delta(x))||
        d_act_0 = diff_active(x)[0] - x
        d_act_1 = diff_active(x + c_shift)[0] - (x + c_shift)
        d_ctl_0 = diff_ctrl(x)[0] - x
        d_ctl_1 = diff_ctrl(x + c_shift)[0] - (x + c_shift)

        act_cm_drift = float((d_act_1 - d_act_0).mean(dim=0).norm()) / max(
            float(d_act_0.std(dim=0).norm()), 1e-6
        )
        ctl_cm_drift = float((d_ctl_1 - d_ctl_0).mean(dim=0).norm()) / max(
            float(d_ctl_0.std(dim=0).norm()), 1e-6
        )
    assert act_cm_drift < ctl_cm_drift, (
        f"Expected differential attention to suppress common-mode shift better than lambda=0: "
        f"diff={act_cm_drift:.4f} vs ctrl={ctl_cm_drift:.4f}"
    )

    # Also test raw differential weights on constant common-mode V: (1 - 0.85)*c vs 1.0*c
    with torch.no_grad():
        q_s = torch.randn(1, n_heads, r_land, d // n_heads)
        k_s = torch.randn(1, n_heads, n_assets, d // n_heads)
        v_cm = torch.ones(1, n_heads, n_assets, d // n_heads) * 5.0
        a_act, lam_act = diff_active.gather_attn._compute_weights(q_s, k_s, key_mask=None)
        a_ctl, lam_ctl = diff_ctrl.gather_attn._compute_weights(q_s, k_s, key_mask=None)
        assert float(lam_ctl.abs().max()) == 0.0
        assert float(lam_act.min()) > 0.0 and float(lam_act.max()) < 2.0
        cm_leak_act = float(torch.matmul(a_act, v_cm).abs().mean())
        cm_leak_ctl = float(torch.matmul(a_ctl, v_cm).abs().mean())
        assert cm_leak_act < 0.25 * cm_leak_ctl


def test_diff_induced_cross_attention_fat_tails_bf16_bounded_lambda_and_n2600() -> None:
    """Verify (d) ±1e3 fat-tail BF16 stability + bounded lambda in (0, 2) + (e) N=2600 check."""
    torch.manual_seed(71)
    d, r_land, n_heads = 64, 16, 4
    mod = DiffInducedFactorCrossAttention(d, n_landmarks=r_land, n_heads=n_heads)

    # Push raw_lambda to extreme ±50 -> lambda stays strictly inside [0, 2] (never overflows!)
    with torch.no_grad():
        mod.gather_attn.raw_lambda.fill_(50.0)
        mod.mix_attn.raw_lambda.fill_(-50.0)
        mod.broadcast_attn.raw_lambda.fill_(0.0)
    lam_hi = mod.gather_attn.effective_lambda().detach()
    lam_lo = mod.mix_attn.effective_lambda().detach()
    lam_mid = mod.broadcast_attn.effective_lambda().detach()
    assert torch.isfinite(lam_hi).all()
    assert 1.99 < float(lam_hi.min()) <= float(lam_hi.max()) <= 2.0
    assert torch.isfinite(lam_lo).all()
    assert 0.0 <= float(lam_lo.min()) <= float(lam_lo.max()) < 1e-6
    assert torch.allclose(lam_mid, torch.ones_like(lam_mid), atol=1e-6)

    # Reset to default & test adversarial ±1e3 fat-tail inputs under BF16 autocast + FP64 parity
    mod = DiffInducedFactorCrossAttention(d, n_landmarks=r_land, n_heads=n_heads)
    mod_64 = DiffInducedFactorCrossAttention(d, n_landmarks=r_land, n_heads=n_heads).double()
    mod_64.load_state_dict({k: v.double() for k, v in mod.state_dict().items()})

    x_fat = torch.randn(64, d, dtype=torch.float64) * 500.0
    x_fat[0] = 1e3
    x_fat[1] = -1e3
    out_64, _ = mod_64(x_fat)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        x_bf16 = x_fat.bfloat16().requires_grad_(True)
        out_bf16, diag_bf16 = mod(x_bf16)
        out_bf16.sum().backward()
    assert torch.isfinite(out_bf16).all()
    assert x_bf16.grad is not None and torch.isfinite(x_bf16.grad).all()
    assert 0.0 < float(diag_bf16["diff_lambda_mean"]) < 2.0
    assert torch.allclose(out_bf16.detach().double(), out_64.detach(), atol=1e-1, rtol=5e-2)

    # #203 P0-1 regression: BF16 autocast with exposures + rmt_loadings + step_emb + regime_vec
    mod_ctx = DiffInducedFactorCrossAttention(d, n_landmarks=r_land, n_heads=n_heads, context_dim=8)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        h_ctx = torch.randn(48, d, dtype=torch.bfloat16, requires_grad=True)
        exp_ctx = torch.randn(48, 5, dtype=torch.bfloat16)
        rmt_ctx = torch.randn(48, 3, dtype=torch.bfloat16)
        step_ctx = torch.randn(d, dtype=torch.bfloat16)
        reg_ctx = torch.randn(8, dtype=torch.bfloat16)
        am_ctx = torch.ones(48, dtype=torch.bool)
        am_ctx[-4:] = False
        out_ctx, diag_ctx = mod_ctx(
            h_ctx,
            asset_mask=am_ctx,
            exposures=exp_ctx,
            rmt_loadings=rmt_ctx,
            step_emb=step_ctx,
            regime_vec=reg_ctx,
        )
        out_ctx.sum().backward()
    assert torch.isfinite(out_ctx).all() and torch.isfinite(diag_ctx["landmark_attn_entropy"])
    assert h_ctx.grad is not None and torch.isfinite(h_ctx.grad).all()

    # (e) N=2600, R=16 full US cross-section smoke & latency check on CPU
    x_2600 = torch.randn(2600, d)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_2600, diag_2600 = mod(x_2600)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out_2600.shape == (2600, d)
    assert torch.isfinite(out_2600).all()
    assert torch.isfinite(diag_2600["landmark_attn_entropy"])
    assert elapsed_ms < 5000.0, f"N=2600 CPU forward unexpectedly slow: {elapsed_ms:.1f}ms"


# ---------------------------------------------------------------------------
# 4. Stage-2 Cross-Family Epistemic Disagreement & CKA-Orthogonal Stacking
# ---------------------------------------------------------------------------


def test_cross_family_disagreement_risk_kappa0_and_pplus_shrinkage() -> None:
    """Verify bit-exact kappa=0.0, CKA weighting, and strict P+ shrinkage on disagreement."""
    torch.manual_seed(83)
    n_assets = 128
    realized_vol_t = 0.01 + 0.03 * torch.rand(n_assets, dtype=torch.float32)

    # Build 4 independent zero-mean family predictions:
    # Assets 0..63: all 4 families AGREE identically.
    # Assets 64..127: balanced long/short DISAGREEMENT (half +2, half -2 flipped across families).
    base_sig = torch.randn(n_assets, dtype=torch.float32)
    base_sig[:64] = base_sig[:64] - base_sig[:64].mean()
    disagree_pat = torch.cat([torch.full((32,), 2.0), torch.full((32,), -2.0)], dim=0)
    f_preds_t = []
    for f_idx in range(4):
        p = base_sig.clone()
        sign = 1.0 if (f_idx % 2 == 0) else -1.0
        p[64:] = sign * disagree_pat
        # Apply arbitrary positive family scale to test cross-sectional standardization
        f_preds_t.append(p * (0.005 * (f_idx + 1)))

    # 1) CKA dissimilarity weights: orthogonal family 3 beats redundant pair (0, 1)
    cka_mat = np.array(
        [
            [1.00, 0.95, 0.50, 0.20],
            [0.95, 1.00, 0.50, 0.20],
            [0.50, 0.50, 1.00, 0.25],
            [0.20, 0.20, 0.25, 1.00],
        ],
        dtype=np.float64,
    )
    w_cka = compute_cka_dissimilarity_weights(cka_mat)
    assert math.isclose(float(w_cka.sum()), 1.0, rel_tol=1e-9)
    assert float(w_cka[3]) > float(w_cka[0]) and float(w_cka[3]) > float(w_cka[1])

    # 2) Torch path: kappa == 0.0 MUST be bit-exact identical to realized_vol!
    res_k0 = compute_cross_family_disagreement_risk(
        f_preds_t, realized_vol_t, kappa=0.0, cka_weights=w_cka.tolist()
    )
    assert torch.equal(res_k0["sigma_eff"], realized_vol_t)

    # Dict-input & cka_matrix auto-weighting smoke check
    f_dict = {f"f{i + 1}": t for i, t in enumerate(f_preds_t)}
    res_dict = compute_cross_family_disagreement_risk(
        f_dict, realized_vol_t, kappa=2.0, cka_matrix=cka_mat
    )

    # 3) kappa > 0: high disagreement strictly increases sigma_eff and shrinks P+ position size!
    res_k2 = compute_cross_family_disagreement_risk(
        f_preds_t, realized_vol_t, kappa=2.0, cka_weights=w_cka.tolist()
    )
    assert torch.allclose(res_k2["sigma_eff"], res_dict["sigma_eff"], atol=1e-6)
    assert float(res_k2["disagreement_var"][:64].max()) < 1e-6
    assert float(res_k2["disagreement_var"][64:].mean()) > 1.0
    assert torch.all(res_k2["sigma_eff"] >= realized_vol_t - 1e-7)
    assert float((res_k2["sigma_eff"][64:] - realized_vol_t[64:]).mean()) > 0.01

    # Feed identical alpha into DifferentiablePortfolioLayer & ConvexPortfolioOptimizerPPlus:
    # asset 0 agrees (var=0), asset 1 disagrees strongly (high var), equal alpha & base vol
    mu_test = torch.zeros(32, dtype=torch.float32)
    mu_test[0] = 0.004
    mu_test[1] = 0.004
    mu_test[2] = -0.004
    mu_test[3] = -0.004
    vol_base = torch.full((32,), 0.02, dtype=torch.float32)

    p_f = [torch.zeros(32, dtype=torch.float32) for _ in range(4)]
    for fi in range(4):
        p_f[fi][0] = 2.0
        p_f[fi][2] = -2.0
        p_f[fi][1] = 2.0 if fi % 2 == 0 else -2.0
        p_f[fi][3] = -2.0 if fi % 2 == 0 else 2.0
        p_f[fi][4:] = torch.linspace(-0.5, 0.5, 28)

    r0 = compute_cross_family_disagreement_risk(p_f, vol_base, kappa=0.0)
    r3 = compute_cross_family_disagreement_risk(p_f, vol_base, kappa=3.0)
    assert torch.equal(r0["sigma_eff"], vol_base)
    assert float(r3["sigma_eff"][1]) > float(r3["sigma_eff"][0])

    # DifferentiablePortfolioLayer shrinkage check
    dpl = DifferentiablePortfolioLayer(
        steps=10,
        risk_aversion=20.0,
        factor_penalty=0.0,
        l1_cost=0.0001,
        gross_limit=1.0,
        pos_cap=0.2,
    )
    w_k0 = dpl(mu_test, vol=r0["sigma_eff"])
    w_k3 = dpl(mu_test, vol=r3["sigma_eff"])
    assert float(w_k3[1].abs()) < float(w_k0[1].abs())
    assert float(w_k3[1].abs()) < float(w_k3[0].abs())

    # ConvexPortfolioOptimizerPPlus (NumPy & Torch paths + pure NumPy bit-exact kappa=0)
    p_f_np = [t.numpy() for t in p_f]
    vol_np = vol_base.numpy().astype(np.float64)
    r0_np = compute_cross_family_disagreement_risk(p_f_np, vol_np, kappa=0.0)
    r3_np = compute_cross_family_disagreement_risk(p_f_np, vol_np, kappa=3.0)
    assert isinstance(r0_np["sigma_eff"], np.ndarray)
    assert np.array_equal(r0_np["sigma_eff"], vol_np)
    assert float(r3_np["sigma_eff"][1]) > float(r3_np["sigma_eff"][0])

    opt_p = ConvexPortfolioOptimizerPPlus(
        risk_aversion=20.0, factor_penalty=0.0, l1_cost=0.0001, impact_cost=0.0, gross_limit=1.0
    )
    w_np_0 = opt_p.solve(mu_test.numpy(), vol=r0_np["sigma_eff"], pos_cap=0.2)
    w_np_3 = opt_p.solve(mu_test.numpy(), vol=r3_np["sigma_eff"], pos_cap=0.2)
    assert abs(float(w_np_3[1])) < abs(float(w_np_0[1]))
    assert abs(float(w_np_3[1])) < abs(float(w_np_3[0]))

    w_pt_0 = opt_p.solve_torch(mu_test, vol=r0["sigma_eff"], pos_cap=0.2)
    w_pt_3 = opt_p.solve_torch(mu_test, vol=r3["sigma_eff"], pos_cap=0.2)
    assert float(w_pt_3[1].abs()) < float(w_pt_0[1].abs())
    assert float(w_pt_3[1].abs()) < float(w_pt_3[0].abs())

    # #202 P1-5a: non-finite cka_weights (inf/nan) sanitized identically on Tensor, list & NumPy
    bad_w_inf = [0.5, float("inf"), 0.5, 0.0]
    bad_w_nan = [0.5, float("nan"), 0.5, 0.0]
    expected_w = torch.tensor([0.5, 0.0, 0.5, 0.0], dtype=torch.float32)
    for bad_w in (bad_w_inf, bad_w_nan):
        r_list = compute_cross_family_disagreement_risk(p_f, vol_base, kappa=1.0, cka_weights=bad_w)
        r_ten = compute_cross_family_disagreement_risk(
            p_f, vol_base, kappa=1.0, cka_weights=torch.tensor(bad_w, dtype=torch.float32)
        )
        r_np = compute_cross_family_disagreement_risk(
            p_f_np, vol_np, kappa=1.0, cka_weights=np.array(bad_w, dtype=np.float64)
        )
        assert torch.allclose(r_list["weights"], expected_w, atol=1e-6)
        assert torch.allclose(r_ten["weights"], expected_w, atol=1e-6)
        assert np.allclose(r_np["weights"], expected_w.numpy(), atol=1e-6)
        assert torch.isfinite(r_list["sigma_eff"]).all() and np.isfinite(r_np["sigma_eff"]).all()

    # #202 P1-5b & #203 P1-3: multi-horizon [F, N, H] preds with 1D [N] or [N, 1] realized_vol
    mh_t = torch.randn(4, 32, 6, dtype=torch.float32)
    mh_np = mh_t.numpy().astype(np.float64)
    for v_shape in ((32,), (32, 1)):
        v_t = vol_base.reshape(v_shape)
        v_n = vol_np.reshape(v_shape)
        r_mh_0 = compute_cross_family_disagreement_risk(mh_t, v_t, kappa=0.0)
        r_mh_1 = compute_cross_family_disagreement_risk(mh_t, v_t, kappa=1.5)
        assert r_mh_0["sigma_eff"].shape == (32, 6)
        assert r_mh_1["sigma_eff"].shape == (32, 6)
        assert torch.equal(r_mh_0["sigma_eff"][:, 0], vol_base)
        assert torch.all(r_mh_1["sigma_eff"] >= r_mh_0["sigma_eff"] - 1e-7)

        rn_mh_0 = compute_cross_family_disagreement_risk(mh_np, v_n, kappa=0.0)
        rn_mh_1 = compute_cross_family_disagreement_risk(mh_np, v_n, kappa=1.5)
        assert rn_mh_0["sigma_eff"].shape == (32, 6)
        assert rn_mh_1["sigma_eff"].shape == (32, 6)
        assert np.array_equal(rn_mh_0["sigma_eff"][:, 0], vol_np)
        assert np.all(rn_mh_1["sigma_eff"] >= rn_mh_0["sigma_eff"] - 1e-12)


# ---------------------------------------------------------------------------
# 5. Global Engineering Gates: 0 Host-Syncs, 0 torch.tensor() Outside __init__, torch.compile
# ---------------------------------------------------------------------------


def test_frontier_modules_static_zero_host_syncs_and_no_runtime_torch_tensor() -> None:
    assert _SRC_PATH.is_file()
    audit = audit_zero_host_syncs(_SRC_PATH)
    assert audit["files_scanned"] == 1
    assert audit["forward_findings"] == 0, f"Forward host-syncs: {audit['findings']}"
    assert audit["total_findings"] == 0, f"Host-syncs found: {audit['findings']}"
    assert audit["clean"] is True
    assert audit["is_zero_host_sync_forward"] is True

    # AST audit: 0 `torch.tensor(` calls outside `__init__`
    tree = ast.parse(_SRC_PATH.read_text(encoding="utf-8"), filename=str(_SRC_PATH))
    bad_calls: list[tuple[str, int]] = []

    class _TensorCallVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.fn_stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.fn_stack.append(node.name)
            self.generic_visit(node)
            self.fn_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "tensor"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
            ):
                cur = self.fn_stack[-1] if self.fn_stack else "<module>"
                if cur != "__init__":
                    bad_calls.append((cur, int(node.lineno)))
            self.generic_visit(node)

    _TensorCallVisitor().visit(tree)
    assert bad_calls == [], f"Found torch.tensor() outside __init__: {bad_calls}"


@pytest.mark.parametrize("dynamic", [True])
def test_frontier_modules_torch_compile_dynamic_clean(dynamic: bool) -> None:
    """All 4 frontier modules compile cleanly with `torch.compile(mode='default', dynamic=True)`."""
    torch.manual_seed(97)
    d = 32

    polar = torch.compile(PolarAlphaHead(n_horizons=2), mode="default", dynamic=dynamic)
    slot_enc = torch.compile(
        ObservedSlotEncoder(n_slots=12, hidden_dim=d), mode="default", dynamic=dynamic
    )
    set_fus = torch.compile(
        MissingnessAwareSetModalityFusion(hidden_dim=d, n_aux_modalities=4, n_heads=4),
        mode="default",
        dynamic=dynamic,
    )
    diff_ca = torch.compile(
        DiffInducedFactorCrossAttention(hidden_dim=d, n_landmarks=8, n_heads=4),
        mode="default",
        dynamic=dynamic,
    )

    for n_assets in (24, 36):
        raw_a = torch.randn(n_assets, 2, requires_grad=True)
        vol = torch.full((n_assets, 2), 0.02)
        p_out = polar(raw_a, vol=vol)
        assert torch.isfinite(p_out["alpha_cal"]).all()
        (p_out["alpha_cal_for_pplus"].sum() + p_out["alpha_cal_for_cal"].sum()).backward()

        tri = torch.cat(
            [
                torch.randn(n_assets, 12),
                (torch.rand(n_assets, 12) > 0.5).float(),
                torch.rand(n_assets, 12),
            ],
            dim=-1,
        )
        s_out = slot_enc(tri)
        assert s_out.shape == (n_assets, d) and torch.isfinite(s_out).all()
        s_out.sum().backward()

        mod_tok = torch.randn(n_assets, 4, d, requires_grad=True)
        mod_m = (torch.rand(n_assets, 4) > 0.4).float()
        core = torch.randn(n_assets, d)
        f_out, _ = set_fus(mod_tok, mod_m, core)
        assert f_out.shape == (n_assets, d) and torch.isfinite(f_out).all()
        f_out.sum().backward()

        h = torch.randn(n_assets, d, requires_grad=True)
        d_out, _ = diff_ca(h)
        assert d_out.shape == (n_assets, d) and torch.isfinite(d_out).all()
        d_out.sum().backward()
