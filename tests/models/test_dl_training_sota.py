"""TDD test suite for `mars_titan.models.mmarrs.dl_training_sota`.

Validates every axis of the Deep Learning SOTA Training & Architecture Taxonomy Kit:
  1. Modern SOTA layers: `DynamicTanh`, `RMSNormSOTA`, `build_norm_layer`, `SwiGLUFFN`,
     `LayerScale`, and `DeepNorm` (`deepnorm_gains`, `deepnorm_residual`, `DeepNormResidual`)
     in FP32, FP64 reference parity, and BF16.
  2. Maximal Update Parameterization (`apply_mup_init_and_param_groups`) across all 4
     independent model families + empirical `run_mup_coordinate_check` width-invariance.
  3. Scale-invariant SOTA losses & anti-collapse regularizers: `tail_listnet_loss`
     (`#164` P0-2 scale invariance, FM-7 radial neutrality, >70% D10/D1 mass, NaN/CN ties),
     `vicreg_covariance_loss` (0 on orthogonal, high on collapsed, raises eRank), and
     `cross_sectional_rank_contrastive_loss` (tail separation & CN tie safety).
  4. Zero-sync gradient dynamics: `adaptive_gradient_clip_` (Brock et al. AGC via `_foreach_*`)
     and `cautious_weight_decay_mask` / `apply_cautious_weight_decay_`.
  5. Strictly causal training paradigms: `CausalCurriculumScheduler`, `CausalSSLPretrainer`
     across all 4 independent families + CN 99.3%/100% missing NaN fundamentals, and
     `StructuredModalityNoiseAugmenter` (50/50 session+asset dropout, prices-safe, eval identity).
  6. Static AST verification via `audit_zero_host_syncs` == 0 findings across the entire module.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from mars_titan.data.market_physics_v4_gpu import gaussian_rank_features
from mars_titan.models.mmarrs.blackwell_kernels import audit_zero_host_syncs
from mars_titan.models.mmarrs.dl_training_sota import (
    CausalCurriculumScheduler,
    CausalSSLPretrainer,
    DeepNormResidual,
    DynamicTanh,
    LayerScale,
    RMSNormSOTA,
    StructuredModalityNoiseAugmenter,
    SwiGLUFFN,
    adaptive_gradient_clip_,
    apply_cautious_weight_decay_,
    apply_mup_init_and_param_groups,
    build_norm_layer,
    cautious_weight_decay_mask,
    compute_swiglu_hidden_dim,
    cross_sectional_rank_contrastive_loss,
    deepnorm_gains,
    deepnorm_residual,
    run_mup_coordinate_check,
    tail_listnet_loss,
    vicreg_covariance_loss,
)
from mars_titan.models.mmarrs.model import build_family_model

_TINY_DIMS: dict[str, int] = dict(
    hidden_dim=32,
    prices_dim=8,
    fundamentals_dim=4,
    macro_dim=4,
    news_dim=4,
    charts_dim=4,
    physics_dim=6,
    context_dim=8,
    exposures_dim=4,
)

_FOUR_FAMILIES: tuple[tuple[str, str], ...] = (
    ("titans_original", "t15b_full_v2"),
    ("holographic_brain", "nb_v2_full"),
    ("mmarrs_titan", "v14c_tied_xasset_cf"),
    ("titans_cognitive_core", "cog_v2_10tools_soft_k3"),
)


def _make_synthetic_session(
    n_assets: int = 24,
    *,
    seed: int = 0,
    fund_missing_frac: float = 0.0,
    nan_missing_fund: bool = False,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    prices = torch.randn((n_assets, _TINY_DIMS["prices_dim"]), generator=g)
    fundamentals = torch.randn((n_assets, _TINY_DIMS["fundamentals_dim"]), generator=g)
    macro = torch.randn((1, _TINY_DIMS["macro_dim"]), generator=g).expand(n_assets, -1).clone()
    news = torch.randn((n_assets, _TINY_DIMS["news_dim"]), generator=g)
    charts = torch.randn((n_assets, _TINY_DIMS["charts_dim"]), generator=g)
    physics = torch.randn((n_assets, _TINY_DIMS["physics_dim"]), generator=g)
    ctx = torch.randn((n_assets, _TINY_DIMS["context_dim"]), generator=g)
    exposures = torch.randn((n_assets, _TINY_DIMS["exposures_dim"]), generator=g)
    mask = torch.ones((n_assets, 5), dtype=torch.float32)

    if fund_missing_frac > 0.0:
        n_miss = min(n_assets, int(round(n_assets * fund_missing_frac)))
        mask[:n_miss, 1] = 0.0
        if nan_missing_fund:
            fundamentals[:n_miss] = float("nan")
        else:
            fundamentals[:n_miss] = 0.0

    # Synthetic causal next-day returns with modest signal tied to prices[:, 0]
    next_ret = 0.015 * prices[:, 0:1] + 0.01 * torch.randn((n_assets, 1), generator=g)
    target_returns = torch.cat(
        [next_ret, 0.02 * torch.randn((n_assets, 5), generator=g)], dim=-1
    )
    return dict(
        prices=prices,
        fundamentals=fundamentals,
        macro=macro,
        news=news,
        charts=charts,
        physics_v4=physics,
        high_snr_context=ctx,
        exposures=exposures,
        modality_mask=mask,
        target_returns=target_returns,
        asset_ids=torch.arange(n_assets, dtype=torch.long),
    )


class _SimpleWidthMLP(nn.Module):
    """3-layer width-parameterized MLP with pre-head RMSNorm for muP coordinate check tests."""

    def __init__(self, width: int, *, in_dim: int = 16, out_dim: int = 1) -> None:
        super().__init__()
        self.enc_in = nn.Linear(in_dim, width)
        self.hidden1 = nn.Linear(width, width)
        self.hidden2 = nn.Linear(width, width)
        self.norm = RMSNormSOTA(width)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.silu(self.enc_in(x))
        h = torch.nn.functional.silu(self.hidden1(h))
        h = torch.nn.functional.silu(self.hidden2(h))
        h = self.norm(h)
        return self.head(h)


class _NoNormWidthMLP(nn.Module):
    """Un-normalized width MLP where SP genuinely explodes with width (`#183` P0-2 test)."""

    def __init__(self, width: int, *, in_dim: int = 16, out_dim: int = 1) -> None:
        super().__init__()
        self.enc_in = nn.Linear(in_dim, width)
        self.hidden1 = nn.Linear(width, width)
        self.hidden2 = nn.Linear(width, width)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.silu(self.enc_in(x))
        h = torch.nn.functional.silu(self.hidden1(h))
        h = torch.nn.functional.silu(self.hidden2(h))
        return self.head(h)


# ---------------------------------------------------------------------------
# 1. Modern SOTA Layers: DyT, RMSNormSOTA, SwiGLUFFN, LayerScale, DeepNorm
# ---------------------------------------------------------------------------


def test_dynamic_tanh_fp32_bf16_reference_parity_and_adversarial_stability() -> None:
    torch.manual_seed(7)
    dim = 64
    dyt = DynamicTanh(dim, alpha_init=0.5)
    assert math.isclose(float(dyt.alpha.detach()), 0.5, rel_tol=1e-6)

    with torch.no_grad():
        dyt.gamma.copy_(torch.linspace(0.5, 1.5, dim))
        dyt.beta.copy_(torch.linspace(-0.2, 0.2, dim))

    # Adversarial inputs: normal, huge, tiny, mixed signs, identical
    x_fp64 = torch.cat(
        [
            torch.randn(8, dim, dtype=torch.float64),
            torch.full((4, dim), 1e4, dtype=torch.float64),
            torch.full((4, dim), -1e4, dtype=torch.float64),
            torch.full((4, dim), 1e-5, dtype=torch.float64),
            torch.zeros(4, dim, dtype=torch.float64),
        ],
        dim=0,
    )
    ref = (
        dyt.gamma.double() * torch.tanh(dyt.alpha.double() * x_fp64) + dyt.beta.double()
    )

    x_f32 = x_fp64.float().requires_grad_(True)
    y_f32 = dyt(x_f32)
    assert y_f32.dtype == torch.float32
    assert torch.isfinite(y_f32).all()
    assert torch.allclose(y_f32.double(), ref, rtol=1e-5, atol=1e-6)
    y_f32.pow(2).mean().backward()
    assert x_f32.grad is not None and torch.isfinite(x_f32.grad).all()
    assert dyt.alpha.grad is not None and torch.isfinite(dyt.alpha.grad).all()
    assert dyt.gamma.grad is not None and torch.isfinite(dyt.gamma.grad).all()
    assert dyt.beta.grad is not None and torch.isfinite(dyt.beta.grad).all()

    dyt.zero_grad(set_to_none=True)
    x_bf16 = x_fp64.to(dtype=torch.bfloat16).requires_grad_(True)
    y_bf16 = dyt(x_bf16)
    assert y_bf16.dtype == torch.bfloat16
    assert torch.isfinite(y_bf16).all()
    assert torch.allclose(y_bf16.double(), ref, rtol=5e-2, atol=1e-2)
    y_bf16.float().pow(2).mean().backward()
    assert x_bf16.grad is not None and torch.isfinite(x_bf16.grad).all()


def test_rmsnorm_sota_fp64_reference_bf16_and_factory() -> None:
    torch.manual_seed(11)
    dim = 48
    norm = RMSNormSOTA(dim, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.7, 1.3, dim))

    x_fp64 = torch.cat(
        [
            torch.randn(10, dim, dtype=torch.float64),
            torch.randn(4, dim, dtype=torch.float64) * 500.0,
            torch.randn(4, dim, dtype=torch.float64) * 1e-4,
            torch.full((4, dim), 42.0, dtype=torch.float64),
        ],
        dim=0,
    )
    ref = (
        x_fp64
        * torch.rsqrt(x_fp64.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        * norm.weight.double()
    )

    x_f32 = x_fp64.float().requires_grad_(True)
    y_f32 = norm(x_f32)
    assert y_f32.dtype == torch.float32
    assert torch.isfinite(y_f32).all()
    assert torch.allclose(y_f32.double(), ref, rtol=1e-5, atol=1e-6)
    y_f32.sum().backward()
    assert x_f32.grad is not None and torch.isfinite(x_f32.grad).all()

    x_bf16 = x_fp64.to(dtype=torch.bfloat16).requires_grad_(True)
    y_bf16 = norm(x_bf16)
    assert y_bf16.dtype == torch.bfloat16
    assert torch.isfinite(y_bf16).all()
    assert torch.allclose(y_bf16.double(), ref, rtol=5e-2, atol=2e-2)
    y_bf16.float().sum().backward()
    assert x_bf16.grad is not None and torch.isfinite(x_bf16.grad).all()

    assert isinstance(build_norm_layer("layernorm", dim), nn.LayerNorm)
    assert isinstance(build_norm_layer("rmsnorm", dim), RMSNormSOTA)
    assert isinstance(build_norm_layer("dyt", dim), DynamicTanh)
    with pytest.raises(ValueError):
        build_norm_layer("unknown_norm", dim)  # type: ignore[arg-type]


def test_swiglu_ffn_iso_param_scaling_layerscale_and_deepnorm_bf16() -> None:
    torch.manual_seed(13)
    dim = 128
    h_dim = compute_swiglu_hidden_dim(dim, expansion=4.0, multiple_of=8)
    assert h_dim % 8 == 0
    ffn = SwiGLUFFN(dim, expansion=4.0, multiple_of=8, bias=False)
    assert ffn.hidden_dim == h_dim

    # Iso-param check vs standard 2-matrix 4x FFN (within 3%)
    std_params = 2 * dim * (4 * dim)
    swiglu_params = sum(p.numel() for p in ffn.parameters())
    assert abs(swiglu_params - std_params) / std_params < 0.03

    # Exact formula check
    x = torch.randn(16, dim)
    expected = ffn.w_down(torch.nn.functional.silu(ffn.w_gate(x)) * ffn.w_up(x))
    assert torch.allclose(ffn(x), expected, atol=1e-6)

    # LayerScale check
    ls = LayerScale(dim, gamma_init=1e-2)
    assert torch.allclose(ls(x), x * 1e-2, atol=1e-6)

    # DeepNorm exact Wang et al. gains & residual helper
    alpha_e, beta_e = deepnorm_gains(6, mode="encoder")
    assert math.isclose(alpha_e, (2.0 * 6) ** 0.25, rel_tol=1e-9)
    assert math.isclose(beta_e, (8.0 * 6) ** -0.25, rel_tol=1e-9)
    alpha_d, beta_d = deepnorm_gains(6, mode="decoder")
    assert math.isclose(alpha_d, (3.0 * 6) ** 0.25, rel_tol=1e-9)
    assert math.isclose(beta_d, (12.0 * 6) ** -0.25, rel_tol=1e-9)

    dn = DeepNormResidual(dim, num_layers=6, norm_kind="rmsnorm")
    sub = ls(ffn(x))
    out_dn = dn(x, sub)
    out_fn = deepnorm_residual(x, sub, alpha=alpha_e, norm=dn.norm)
    assert torch.allclose(out_dn, out_fn, atol=1e-6)

    # BF16 end-to-end check (autocast + explicit bf16 tensors)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        x_b = x.to(dtype=torch.bfloat16).requires_grad_(True)
        y_b = dn(x_b, ls(ffn(x_b)))
        assert torch.isfinite(y_b).all()
        y_b.float().pow(2).mean().backward()
        assert x_b.grad is not None and torch.isfinite(x_b.grad).all()


# ---------------------------------------------------------------------------
# 2. muP Init, Param Groups & Empirical Coordinate Check
# ---------------------------------------------------------------------------


def test_mup_param_groups_role_isolation_across_all_four_families() -> None:
    base_width = 16
    width = 32
    base_lr = 4e-4
    expected_scaled_lr = base_lr * (base_width / width)

    for fam, var in _FOUR_FAMILIES:
        model = build_family_model(var, market="US", family=fam, **_TINY_DIMS)
        # Snapshot encoder, router, norm, readout, hidden, and tabm params before muP init
        enc_before = model.enc_prices.net[1].weight.detach().clone()
        norm_before = model.head.norm.weight.detach().clone()
        alpha_before = model.head.alpha_head.weight.detach().clone()
        fusion_before = model.fusion_proj.weight.detach().clone()
        tabm_r = getattr(model.head, "tabm_r", None)
        tabm_before = tabm_r.detach().clone() if isinstance(tabm_r, torch.Tensor) else None

        groups = apply_mup_init_and_param_groups(
            model,
            base_width=base_width,
            width=width,
            base_lr=base_lr,
            weight_decay=1e-2,
        )
        # Map every param id to its group
        p_to_group = {}
        for g in groups:
            for p in g["params"]:
                assert id(p) not in p_to_group, f"Duplicate param in muP groups ({fam})"
                p_to_group[id(p)] = g

        # Every trainable parameter must be covered exactly once
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(p_to_group) == len(trainable), f"Missing trainable param in {fam}"

        # Input encoders, norms, biases, routers, factor_load stay at 1.0x LR and untouched init
        assert torch.equal(model.enc_prices.net[1].weight, enc_before)
        assert torch.equal(model.head.norm.weight, norm_before)
        # Default reinit_weights=False preserves calibrated hidden & rescaled readout inits
        assert torch.equal(model.fusion_proj.weight, fusion_before)
        assert torch.allclose(
            model.head.alpha_head.weight, alpha_before * (base_width / width), atol=1e-7
        )
        assert math.isclose(
            p_to_group[id(model.enc_prices.net[1].weight)]["lr"], base_lr, rel_tol=1e-9
        )
        assert math.isclose(p_to_group[id(model.head.norm.weight)]["lr"], base_lr, rel_tol=1e-9)
        assert p_to_group[id(model.head.norm.weight)]["weight_decay"] == 0.0
        if tabm_before is not None and isinstance(tabm_r, torch.Tensor):
            assert torch.equal(tabm_r, tabm_before)
            assert math.isclose(p_to_group[id(tabm_r)]["lr"], base_lr, rel_tol=1e-9)
            assert p_to_group[id(tabm_r)]["weight_decay"] == 0.0

        # Readout head linear weight gets scaled LR (base_width / width) & rescaled init
        g_alpha = p_to_group[id(model.head.alpha_head.weight)]
        assert g_alpha["mup_role"] == "readout"
        assert math.isclose(g_alpha["lr"], expected_scaled_lr, rel_tol=1e-9)

        # 2D hidden matrix (`fusion_proj.weight`) gets scaled LR
        g_hid = p_to_group[id(model.fusion_proj.weight)]
        assert g_hid["mup_role"] == "hidden_2d"
        assert math.isclose(g_hid["lr"], expected_scaled_lr, rel_tol=1e-9)

        # `#183` P0-1 guards: 3D MoE/dendritic weights scaled, TabM r_in/s_mid wd==0,
        # fixed small-fan-in F4 scalar tools at 1.0x LR, routers at 1.0x LR
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            low = name.lower()
            leaf = low.rsplit(".", 1)[-1]
            if any(tok in low for tok in ("router", "halt_gate", "balance_bias")):
                assert math.isclose(
                    p_to_group[id(p)]["lr"], base_lr, rel_tol=1e-9
                ), f"Router param {name} in {fam} should remain at 1.0x LR"
            if leaf in ("r_in", "s_mid", "tabm_r", "tabm_s"):
                assert p_to_group[id(p)]["weight_decay"] == 0.0, f"TabM {name} must have wd=0"
                assert math.isclose(p_to_group[id(p)]["lr"], base_lr, rel_tol=1e-9)
            if leaf in ("shared_w1", "shared_w2", "routed_w1", "routed_w2", "basal_w", "soma_w"):
                assert p_to_group[id(p)]["mup_role"] == "hidden_2d", f"3D {name} must be hidden_2d"
                assert math.isclose(p_to_group[id(p)]["lr"], expected_scaled_lr, rel_tol=1e-9)
            if "tool3_proj.0.weight" in low or "tool4_proj.0.weight" in low:
                assert math.isclose(
                    p_to_group[id(p)]["lr"], base_lr, rel_tol=1e-9
                ), f"Fixed small-fan-in scalar tool {name} must remain at 1.0x LR"


def test_mup_coordinate_check_invariance_vs_sp_divergence() -> None:
    widths = (64, 128, 256)
    # 1) No-final-norm MLP (`#183` P0-2): both hidden AND readout updates stay width-invariant
    res_nonorm = run_mup_coordinate_check(
        lambda w: _NoNormWidthMLP(w, in_dim=16, out_dim=1),
        widths=widths,
        base_width=128,
        input_dim=16,
        batch_size=48,
        n_steps=5,
        base_lr=1e-2,
        seed=123,
    )
    assert res_nonorm["all_finite"] is True
    assert res_nonorm["passed"] is True
    assert res_nonorm["max_to_min_ratio"] < 1.8
    assert res_nonorm["out_max_to_min_ratio"] < 1.8
    assert abs(res_nonorm["width_scaling_exponent"]) < 0.35
    assert abs(res_nonorm["out_width_scaling_exponent"]) < 0.35
    for w in widths:
        assert res_nonorm["delta_h_rms"][w] > 1e-4
        assert res_nonorm["delta_out_rms"][w] > 1e-4

    # 2) Pre-head RMSNorm MLP also passes BOTH hidden AND un-normalized readout checks
    res_norm = run_mup_coordinate_check(
        lambda w: _SimpleWidthMLP(w, in_dim=16, out_dim=1),
        widths=widths,
        base_width=128,
        input_dim=16,
        batch_size=48,
        n_steps=5,
        base_lr=1e-2,
        seed=123,
    )
    assert res_norm["passed"] is True
    assert res_norm["out_max_to_min_ratio"] < 1.8

    # 3) Standard Parameterization (SP) on _NoNormWidthMLP genuinely diverges in BOTH h and out
    sp_h_deltas: dict[int, float] = {}
    sp_step_out_deltas: dict[int, float] = {}
    g = torch.Generator().manual_seed(123)
    x = torch.randn(48, 16, generator=g)
    y = torch.randn(48, 1, generator=g)
    for w in widths:
        torch.manual_seed(123)
        m = _NoNormWidthMLP(w, in_dim=16, out_dim=1)
        captured: list[torch.Tensor] = []
        handle = m.head.register_forward_pre_hook(
            lambda _mod, inp, sink=captured: sink.append(inp[0])
        )
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.0)
        with torch.no_grad():
            captured.clear()
            out_prev = m(x).detach().clone()
            h0 = captured[-1].detach().clone()
        step_outs: list[float] = []
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            out = m(x)
            torch.nn.functional.mse_loss(out, y).backward()
            opt.step()
            with torch.no_grad():
                captured.clear()
                out_cur = m(x).detach().clone()
                step_outs.append(float((out_cur - out_prev).pow(2).mean().sqrt()))
                out_prev = out_cur
        with torch.no_grad():
            captured.clear()
            _ = m(x)
            h_end = captured[-1].detach().clone()
        handle.remove()
        sp_h_deltas[w] = float((h_end - h0).pow(2).mean().sqrt())
        sp_step_out_deltas[w] = sum(step_outs) / float(len(step_outs))

    sp_h_ratio = max(sp_h_deltas.values()) / max(min(sp_h_deltas.values()), 1e-12)
    sp_step_out_ratio = max(sp_step_out_deltas.values()) / max(
        min(sp_step_out_deltas.values()), 1e-12
    )
    assert sp_h_ratio > 2.0
    assert sp_step_out_ratio > 2.0
    assert res_nonorm["max_to_min_ratio"] < sp_h_ratio
    assert res_nonorm["step_out_max_to_min_ratio"] < sp_step_out_ratio


# ---------------------------------------------------------------------------
# 3. Scale-Invariant SOTA Losses & Anti-Collapse Regularizers
# ---------------------------------------------------------------------------


def test_tail_listnet_loss_scale_invariance_fm7_tail_mass_and_nan_tie_safety() -> None:
    torch.manual_seed(42)
    n_assets = 400
    target = torch.randn(n_assets, dtype=torch.float64) * 0.02
    pred = torch.randn(n_assets, dtype=torch.float64) * 0.01

    # 1) Exact scale invariance: pred * 10 == pred == pred * 0.01 & translation pred + 5 == pred
    loss_1 = tail_listnet_loss(pred, target, tau=0.5)
    loss_10 = tail_listnet_loss(pred * 10.0, target, tau=0.5)
    loss_tiny = tail_listnet_loss(pred * 0.01, target, tau=0.5)
    loss_shift = tail_listnet_loss(pred + 5.0, target, tau=0.5)
    assert torch.allclose(loss_1, loss_10, rtol=1e-12, atol=1e-12)
    assert torch.allclose(loss_1, loss_tiny, rtol=1e-12, atol=1e-12)
    assert torch.allclose(loss_1, loss_shift, rtol=1e-12, atol=1e-12)

    # Same scale invariance in FP32 and BF16
    p32 = pred.float()
    y32 = target.float()
    assert torch.allclose(
        tail_listnet_loss(p32, y32, tau=0.5),
        tail_listnet_loss(p32 * 10.0, y32, tau=0.5),
        rtol=1e-5,
        atol=1e-6,
    )
    loss_bf16 = tail_listnet_loss(p32.bfloat16(), y32.bfloat16(), tau=0.5)
    assert torch.isfinite(loss_bf16)

    # 2) FM-7 radial gradient neutrality: zero incentive to inflate |pred|
    p_grad = pred.float().clone().requires_grad_(True)
    l_fm7 = tail_listnet_loss(p_grad, y32, tau=0.5)
    l_fm7.backward()
    assert p_grad.grad is not None and torch.isfinite(p_grad.grad).all()
    radial = float((p_grad.grad * (p_grad.detach() - p_grad.detach().mean())).sum().abs())
    assert radial < 1e-4

    # 3) Ordering sensitivity & >70% mass on D10 / D1 tails at tau=0.5
    g_y = gaussian_rank_features(target.unsqueeze(-1)).squeeze(-1)
    q_pos = torch.softmax(g_y / 0.5, dim=0)
    q_neg = torch.softmax(-g_y / 0.5, dim=0)
    d10_idx = torch.topk(target, k=n_assets // 10, largest=True).indices
    d1_idx = torch.topk(target, k=n_assets // 10, largest=False).indices
    assert float(q_pos[d10_idx].sum()) > 0.70
    assert float(q_neg[d1_idx].sum()) > 0.70

    loss_perfect = tail_listnet_loss(g_y, target, tau=0.5)
    loss_aligned = tail_listnet_loss(target + 0.005 * torch.randn_like(target), target, tau=0.5)
    loss_inverted = tail_listnet_loss(-target, target, tau=0.5)
    assert float(loss_perfect) < 1e-9
    assert float(loss_aligned) < float(loss_1) < float(loss_inverted)

    # 4) Multi-horizon [N, H], liquidity_weights, NaNs, CN all-tied limit-lock, empty mask
    p_mh = torch.randn(64, 3, requires_grad=True)
    y_mh = torch.randn(64, 3) * 0.02
    y_mh[:10, 0] = float("nan")
    y_mh[:, 1] = 0.0  # CN all-tied limit-lock horizon
    mask_mh = torch.ones(64, 3, dtype=torch.bool)
    mask_mh[-5:, :] = False
    liq_w = torch.rand(64) + 0.1
    l_mh = tail_listnet_loss(p_mh, y_mh, tau=0.5, mask=mask_mh, liquidity_weights=liq_w)
    assert torch.isfinite(l_mh) and float(l_mh.detach()) >= 0.0
    l_mh.backward()
    assert p_mh.grad is not None and torch.isfinite(p_mh.grad).all()
    assert torch.all(p_mh.grad[-5:, :] == 0.0)

    # Empty mask -> exact finite 0.0
    p_emp = torch.randn(16, requires_grad=True)
    l_emp = tail_listnet_loss(p_emp, torch.randn(16), mask=torch.zeros(16, dtype=torch.bool))
    assert torch.isfinite(l_emp) and float(l_emp.detach()) == 0.0

    # Constant all-zero pred at step 0 -> finite O(1) gradient (no 1/eps explosion!)
    p_const = torch.zeros(64, requires_grad=True)
    l_const = tail_listnet_loss(p_const, torch.randn(64) * 0.02, tau=0.5)
    assert torch.isfinite(l_const)
    l_const.backward()
    assert p_const.grad is not None and torch.isfinite(p_const.grad).all()
    assert float(p_const.grad.detach().abs().max()) < 10.0

    # `#183` P1-1: all-tied CN limit-lock target horizon -> exact 0.0 loss and 0.0 grad!
    p_tied = torch.randn(40, requires_grad=True)
    l_tied_cn = tail_listnet_loss(p_tied, torch.zeros(40), tau=0.5)
    assert float(l_tied_cn.detach()) == 0.0
    l_tied_cn.backward()
    assert p_tied.grad is not None and torch.all(p_tied.grad == 0.0)


def test_vicreg_covariance_loss_orthogonal_vs_collapsed_and_erank() -> None:
    torch.manual_seed(17)
    n_samples, dim = 128, 32

    # Orthogonal zero-mean unit-variance columns: H^T H / (N - 1) == I_D
    raw = torch.randn(n_samples, dim, dtype=torch.float64)
    raw = raw - raw.mean(dim=0, keepdim=True)
    q_mat, _ = torch.linalg.qr(raw, mode="reduced")
    h_orth = q_mat * math.sqrt(float(n_samples - 1))
    l_orth = vicreg_covariance_loss(h_orth)
    assert torch.isfinite(l_orth)
    assert float(l_orth) < 1e-6

    # Rank-1 correlated collapsed representation (width-invariant O(1) across D=32 and D=128)
    u = torch.randn(n_samples, 1, dtype=torch.float64)
    u = (u - u.mean()) / u.std(unbiased=True)
    h_rank1 = u.expand(n_samples, dim).clone()
    l_rank1 = vicreg_covariance_loss(h_rank1)
    assert math.isclose(float(l_rank1), 1.0, rel_tol=1e-2)
    h_rank1_wide = u.expand(n_samples, 128).clone()
    assert math.isclose(float(vicreg_covariance_loss(h_rank1_wide)), 1.0, rel_tol=1e-2)

    # Zero-variance constant collapsed representation
    h_const = torch.full((n_samples, dim), 3.14, dtype=torch.float64)
    l_const = vicreg_covariance_loss(h_const)
    assert float(l_const) > 0.9

    # BF16, mask, NaN row, and N=1 edge case
    h_bf16 = torch.randn(32, 16, dtype=torch.bfloat16, requires_grad=True)
    l_bf16 = vicreg_covariance_loss(h_bf16)
    assert torch.isfinite(l_bf16)
    l_bf16.backward()
    assert h_bf16.grad is not None and torch.isfinite(h_bf16.grad).all()
    assert float(vicreg_covariance_loss(torch.randn(1, 16))) == 0.0

    # Optimizing collapsed representation raises effective rank (eRank) substantially
    h_opt = (h_rank1.float() + 0.05 * torch.randn(n_samples, dim)).detach().requires_grad_(True)

    def _erank(mat: torch.Tensor) -> float:
        m = mat.detach().double()
        m = m - m.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(m)
        p = s.pow(2)
        p = p / p.sum().clamp_min(1e-12)
        return float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))

    erank_before = _erank(h_opt)
    opt = torch.optim.Adam([h_opt], lr=0.1)
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        vicreg_covariance_loss(h_opt).backward()
        opt.step()
    assert _erank(h_opt) > erank_before * 5.0


def test_cross_sectional_rank_contrastive_loss_separates_tails_and_is_cn_safe() -> None:
    torch.manual_seed(29)
    n_assets, dim = 100, 16
    target = torch.linspace(-0.05, 0.05, n_assets)

    # Coherently separated tails vs mixed tails
    v = torch.nn.functional.normalize(torch.randn(1, dim), dim=-1)
    h_sep = torch.randn(n_assets, dim) * 0.05
    h_sep[-20:] = h_sep[-20:] + 2.0 * v
    h_sep[:20] = h_sep[:20] - 2.0 * v

    h_mix = torch.randn(n_assets, dim)
    l_sep = cross_sectional_rank_contrastive_loss(h_sep, target, top_frac=0.2, temp=0.1)
    l_mix = cross_sectional_rank_contrastive_loss(h_mix, target, top_frac=0.2, temp=0.1)
    assert torch.isfinite(l_sep) and torch.isfinite(l_mix)
    assert float(l_sep) < float(l_mix) * 0.25

    # Gradient descent pulls top/bottom tails into coherent opposite clusters
    h_train = (torch.randn(n_assets, dim) * 0.5).requires_grad_(True)
    z0 = torch.nn.functional.normalize(h_train.detach(), dim=-1)
    c_top0 = torch.nn.functional.normalize(z0[-20:].mean(dim=0, keepdim=True), dim=-1)
    cohesion_before = float((z0[-20:] * c_top0).sum(dim=-1).mean())

    opt = torch.optim.Adam([h_train], lr=0.1)
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        loss = cross_sectional_rank_contrastive_loss(h_train, target, top_frac=0.2, temp=0.1)
        loss.backward()
        opt.step()

    z = torch.nn.functional.normalize(h_train.detach(), dim=-1)
    c_top = torch.nn.functional.normalize(z[-20:].mean(dim=0, keepdim=True), dim=-1)
    c_bot = torch.nn.functional.normalize(z[:20].mean(dim=0, keepdim=True), dim=-1)
    assert float((z[-20:] * c_top).sum(dim=-1).mean()) > max(0.75, cohesion_before + 0.4)
    assert float((z[:20] * c_bot).sum(dim=-1).mean()) > 0.75
    assert float((c_top * c_bot).sum()) < -0.7

    # CN all-tied limit-lock + NaN targets + BF16 safety
    h_cn = torch.randn(32, 16, dtype=torch.bfloat16, requires_grad=True)
    y_tied = torch.zeros(32, dtype=torch.bfloat16)
    l_tied = cross_sectional_rank_contrastive_loss(h_cn, y_tied)
    assert torch.isfinite(l_tied) and float(l_tied.detach()) == 0.0
    l_tied.backward()
    assert h_cn.grad is not None and torch.isfinite(h_cn.grad).all()

    y_nan = torch.randn(32)
    y_nan[:12] = float("nan")
    h_nan = torch.randn(32, 16, requires_grad=True)
    l_nan = cross_sectional_rank_contrastive_loss(h_nan, y_nan)
    assert torch.isfinite(l_nan)
    l_nan.backward()
    assert h_nan.grad is not None and torch.isfinite(h_nan.grad).all()


# ---------------------------------------------------------------------------
# 4. Gradient Dynamics: Adaptive Gradient Clipping (AGC) & Cautious WD
# ---------------------------------------------------------------------------


def test_adaptive_gradient_clip_per_unit_exactness_and_zero_sync() -> None:
    torch.manual_seed(31)
    # 2D weight with 4 output units: unit 0 huge grad, unit 1 small grad, unit 2 zero weight
    w2 = nn.Parameter(torch.randn(4, 8))
    with torch.no_grad():
        w2[2].zero_()
    w2.grad = torch.randn(4, 8)
    with torch.no_grad():
        w2.grad[0].mul_(50.0)  # must be clipped
        w2.grad[1].mul_(1e-5)  # must stay untouched
        w2.grad[2].mul_(5.0)  # zero weight -> eps floor
        w2.grad[3].zero_()  # zero grad -> stays zero

    # 1D bias & 3D MoE tensor + BF16 parameter
    b1 = nn.Parameter(torch.zeros(8))
    b1.grad = torch.full((8,), 10.0)
    w3 = nn.Parameter(torch.randn(3, 4, 8))
    w3.grad = torch.randn(3, 4, 8) * 20.0
    w_bf16 = nn.Parameter(torch.randn(4, 8, dtype=torch.bfloat16))
    w_bf16.grad = torch.randn(4, 8, dtype=torch.bfloat16) * 100.0

    unit1_before = w2.grad[1].clone()
    clip_factor = 0.01
    eps = 1e-3

    clipped_frac = adaptive_gradient_clip_(
        [w2, b1, w3, w_bf16], clip_factor=clip_factor, eps=eps
    )
    assert isinstance(clipped_frac, torch.Tensor)
    assert torch.isfinite(clipped_frac) and float(clipped_frac) > 0.0

    # Unit 0 clipped tightly to clip_factor * ||w_0||
    w0_norm = float(w2[0].detach().norm())
    g0_norm = float(w2.grad[0].detach().norm())
    assert math.isclose(g0_norm, clip_factor * max(w0_norm, eps), rel_tol=1e-5)

    # Unit 1 untouched bit-identically
    assert torch.equal(w2.grad[1], unit1_before)

    # Unit 2 (zero weight) clipped to clip_factor * eps
    assert math.isclose(float(w2.grad[2].detach().norm()), clip_factor * eps, rel_tol=1e-5)

    # Unit 3 zero grad stays exact zero
    assert torch.all(w2.grad[3] == 0.0)

    # 1D bias, 3D MoE, and BF16 all satisfy per-unit bound with finite grads
    assert float(b1.grad.detach().norm()) <= clip_factor * eps + 1e-6
    for e_idx in range(3):
        bound = clip_factor * max(float(w3[e_idx].detach().norm()), eps)
        assert float(w3.grad[e_idx].detach().norm()) <= bound + 1e-5
    assert torch.isfinite(w_bf16.grad).all()
    for row in range(4):
        bound = clip_factor * max(float(w_bf16[row].detach().float().norm()), eps)
        assert float(w_bf16.grad[row].detach().float().norm()) <= bound * 1.05 + 1e-4


def test_cautious_weight_decay_mask_and_application() -> None:
    g = torch.tensor([1.0, -1.0, 2.0, -3.0, 0.0, 0.5])
    p = torch.tensor([2.0, 2.0, -2.0, -2.0, 5.0, 0.0])
    m = cautious_weight_decay_mask(g, p)
    # Aligned (g*p >= 0): indices 0 (+,+), 3 (-,-), 4 (0,+), 5 (+,0)
    expected = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert torch.equal(m, expected)

    m_rescaled = cautious_weight_decay_mask(g, p, rescale=True)
    assert math.isclose(float(m_rescaled.mean()), 1.0, rel_tol=1e-6)

    # `#183` P1-2: BF16 tiny opposite-sign product must NOT underflow to 0.0 -> aligned!
    g_tiny = torch.tensor([-1e-22, 1e-22, 1e-22, -1e-22], dtype=torch.bfloat16)
    p_tiny = torch.tensor([1e-22, -1e-22, 1e-22, -1e-22], dtype=torch.bfloat16)
    m_tiny = cautious_weight_decay_mask(g_tiny, p_tiny)
    assert m_tiny.dtype == torch.bfloat16
    assert torch.equal(m_tiny, torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.bfloat16))

    # In-place cautious weight decay updates ONLY aligned entries
    param = nn.Parameter(p.clone())
    param.grad = g.clone()
    p_orig = param.detach().clone()
    apply_cautious_weight_decay_([param], lr=0.1, weight_decay=0.5)
    p_after = param.detach()
    # Unaligned indices 1, 2 unchanged; aligned non-zero indices 0, 3, 4 decayed
    assert math.isclose(float(p_after[1]), float(p_orig[1]), rel_tol=1e-7)
    assert math.isclose(float(p_after[2]), float(p_orig[2]), rel_tol=1e-7)
    assert abs(float(p_after[0])) < abs(float(p_orig[0]))
    assert abs(float(p_after[3])) < abs(float(p_orig[3]))
    assert abs(float(p_after[4])) < abs(float(p_orig[4]))


# ---------------------------------------------------------------------------
# 5. Strictly Causal Training Paradigms
# ---------------------------------------------------------------------------


def test_causal_curriculum_scheduler_monotonic_annealing() -> None:
    sched = CausalCurriculumScheduler(
        10,
        liq_gamma_start=2.0,
        liq_gamma_end=1.0,
        noise_start=0.05,
        noise_end=0.0,
        horizon_warmup_epochs=3,
    )
    gammas = [sched.liq_gamma(ep) for ep in range(10)]
    noises = [sched.noise_std(ep) for ep in range(10)]

    assert math.isclose(gammas[0], 2.0, rel_tol=1e-9)
    assert math.isclose(gammas[-1], 1.0, rel_tol=1e-9)
    assert math.isclose(noises[0], 0.05, rel_tol=1e-9)
    assert math.isclose(noises[-1], 0.0, abs_tol=1e-12)
    assert all(gammas[i] >= gammas[i + 1] for i in range(9))
    assert all(noises[i] >= noises[i + 1] for i in range(9))

    # Multi-horizon weights: emphasize short horizons early, uniform after warmup
    hw0 = sched.horizon_weights(0, n_horizons=6)
    hw1 = sched.horizon_weights(1, n_horizons=6)
    hw3 = sched.horizon_weights(3, n_horizons=6)
    assert hw0[0] > hw0[-1]
    assert hw1[-1] > hw0[-1]
    assert torch.allclose(hw3, torch.ones(6), atol=1e-6)

    # Liquidity weighting sharpens on liquid assets at epoch 0, flattens by final epoch
    liq = torch.tensor([0.1, 0.5, 0.9, float("nan")])
    w_start = sched.weight_liquidity(liq, epoch=0)
    w_end = sched.weight_liquidity(liq, epoch=9)
    assert torch.isfinite(w_start).all() and torch.isfinite(w_end).all()
    assert float(w_start[2] / w_start[0]) > float(w_end[2] / w_end[0])


def test_causal_ssl_pretrainer_all_four_families_cn_missing_and_loss_reduction() -> None:
    # 1) CN 99.3% & 100% missing fundamentals filled with NaN across ALL 4 families
    batch_cn_99 = _make_synthetic_session(
        n_assets=24, seed=77, fund_missing_frac=0.993, nan_missing_fund=True
    )
    batch_cn_100 = _make_synthetic_session(
        n_assets=24, seed=78, fund_missing_frac=1.0, nan_missing_fund=True
    )

    for fam, var in _FOUR_FAMILIES:
        torch.manual_seed(42)
        model = build_family_model(var, market="CN", family=fam, **_TINY_DIMS)
        ssl = CausalSSLPretrainer(hidden_dim=_TINY_DIMS["hidden_dim"], mask_prob=0.5)

        for cn_batch in (batch_cn_99, batch_cn_100):
            model.zero_grad(set_to_none=True)
            ssl.zero_grad(set_to_none=True)
            out = ssl.compute_ssl_loss(
                model,
                cn_batch,
                next_returns=cn_batch["target_returns"][:, 0],
                generator=torch.Generator().manual_seed(5),
            )
            assert torch.isfinite(out["total"]), f"Non-finite SSL total in {fam}"
            assert torch.isfinite(out["masked_mod_loss"]), f"Non-finite mod loss in {fam}"
            assert torch.isfinite(out["vol_loss"]), f"Non-finite vol loss in {fam}"
            assert torch.isfinite(out["dispersion_loss"]), f"Non-finite disp loss in {fam}"
            out["total"].backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"NaN grad in {fam}.{name}"

    # 2) Pretraining steps strictly reduce SSL loss on a real family model
    torch.manual_seed(99)
    model_f1 = build_family_model(
        "t15b_full_v2", market="US", family="titans_original", **_TINY_DIMS
    )
    ssl_f1 = CausalSSLPretrainer(hidden_dim=_TINY_DIMS["hidden_dim"], mask_prob=0.4)
    opt = torch.optim.AdamW(
        list(model_f1.parameters()) + list(ssl_f1.parameters()), lr=5e-3, weight_decay=0.0
    )
    session = _make_synthetic_session(n_assets=32, seed=101, fund_missing_frac=0.5)
    losses: list[float] = []
    for _ in range(12):
        step_out = ssl_f1.pretrain_step(
            model_f1,
            opt,
            session,
            next_returns=session["target_returns"][:, 0],
            generator=torch.Generator().manual_seed(123),
        )
        losses.append(float(step_out["total"].detach()))
    assert losses[-1] < losses[0] * 0.7

    # 3) `#183` P0-3: norm-collapse shortcut guard & cross-epoch episodic buffer reset
    model_f2 = build_family_model(
        "nb_v2_full", market="CN", family="holographic_brain", **_TINY_DIMS
    )
    ssl_f2 = CausalSSLPretrainer(hidden_dim=_TINY_DIMS["hidden_dim"], mask_prob=0.5)
    b_norm = _make_synthetic_session(n_assets=20, seed=303)
    # Scaling teacher target token norms by 0.01x must leave masked_mod_loss invariant
    orig_enc = model_f2._encode_modalities
    res_normal = ssl_f2.compute_ssl_loss(
        model_f2, b_norm, generator=torch.Generator().manual_seed(99)
    )
    call_count = [0]

    def _shrunk_teacher_enc(b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        toks, msk = orig_enc(b)
        call_count[0] += 1
        # Call 1 is teacher target encoding; call 2 is student forward
        return (toks * 0.01 if call_count[0] == 1 else toks), msk

    model_f2._encode_modalities = _shrunk_teacher_enc  # type: ignore[method-assign]
    res_shrunk = ssl_f2.compute_ssl_loss(
        model_f2, b_norm, generator=torch.Generator().manual_seed(99)
    )
    model_f2._encode_modalities = orig_enc  # type: ignore[method-assign]
    assert math.isclose(
        float(res_shrunk["masked_mod_loss"].detach()),
        float(res_normal["masked_mod_loss"].detach()),
        rel_tol=1e-2,
        abs_tol=1e-2,
    )

    # Multi-session chronological `pretrain_epochs` resets episodic buffers cleanly
    for b_name, buf in model_f2.named_buffers():
        if b_name.endswith("write_ptr"):
            buf.fill_(7)
    opt_f2 = torch.optim.AdamW(
        list(model_f2.parameters()) + list(ssl_f2.parameters()), lr=1e-3
    )
    sessions = [
        _make_synthetic_session(
            n_assets=20, seed=200 + t, fund_missing_frac=0.95, nan_missing_fund=True
        )
        for t in range(3)
    ]
    ep_losses = ssl_f2.pretrain_epochs(model_f2, sessions, opt_f2, epochs=2, seed=7)
    assert len(ep_losses) == 2
    assert all(torch.isfinite(x) for x in ep_losses)
    for b_name, buf in model_f2.named_buffers():
        if b_name.endswith("write_ptr"):
            assert int(buf.detach().sum()) == 0


def test_structured_modality_noise_augmenter_train_only_prices_safe_and_50_50_split() -> None:
    aug = StructuredModalityNoiseAugmenter(
        modality_dropout=0.50,
        session_level_prob=0.50,
        noise_std=0.10,
    )
    batch = _make_synthetic_session(
        n_assets=40, seed=55, fund_missing_frac=0.5, nan_missing_fund=False
    )
    # Put a NaN in prices[0, 0] to verify SessionHealthGate NaN preservation
    batch["prices"][0, 0] = float("nan")
    orig_prices = batch["prices"].clone()
    orig_fund = batch["fundamentals"].clone()
    orig_mask = batch["modality_mask"].clone()

    # 1) Eval mode: bit-exact identity
    aug.eval()
    out_eval = aug(batch, generator=torch.Generator().manual_seed(1))
    assert torch.equal(out_eval["modality_mask"], orig_mask)
    assert torch.equal(
        torch.nan_to_num(out_eval["prices"], nan=-999.0),
        torch.nan_to_num(orig_prices, nan=-999.0),
    )
    assert torch.equal(out_eval["fundamentals"], orig_fund)

    # 2) Train mode: prices never dropped, NaN preserved, missing CN fund stays exact 0,
    #    both session-wide and asset-level drops occur across draws
    aug.train()
    gen = torch.Generator().manual_seed(2025)
    n_session_uniform = 0
    n_asset_mixed = 0
    for _ in range(60):
        out_tr = aug(batch, generator=gen)
        m = out_tr["modality_mask"]
        # Prices channel 0 strictly preserved
        assert torch.equal(m[:, 0], orig_mask[:, 0])
        # Originally missing fundamentals stay 0 in both mask and values (no noise injected!)
        assert torch.all(m[:20, 1] == 0.0)
        assert torch.all(out_tr["fundamentals"][:20] == 0.0)
        # NaN in prices[0, 0] preserved for SessionHealthGate
        assert torch.isnan(out_tr["prices"][0, 0])
        # Present finite prices receive non-zero Gaussian noise
        assert not torch.allclose(out_tr["prices"][1:], orig_prices[1:])
        # Caller batch not mutated in place
        assert torch.equal(batch["modality_mask"], orig_mask)

        # Inspect auxiliary channels 2..4 (originally all-1) for session-wide vs asset-level
        aux = m[:, 2:]
        is_uniform_per_col = bool(((aux == aux[0:1, :]).all()).detach())
        if is_uniform_per_col:
            n_session_uniform += 1
        else:
            n_asset_mixed += 1

    # With session_level_prob=0.50 and N=40, both regimes must clearly appear
    assert n_session_uniform >= 12
    assert n_asset_mixed >= 12


# ---------------------------------------------------------------------------
# 6. Static AST Zero-Host-Sync Audit
# ---------------------------------------------------------------------------


def test_dl_training_sota_static_zero_host_sync_audit() -> None:
    mod_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mars_titan"
        / "models"
        / "mmarrs"
        / "dl_training_sota.py"
    )
    assert mod_path.is_file()
    audit = audit_zero_host_syncs(mod_path)
    assert audit["files_scanned"] == 1
    assert audit["forward_findings"] == 0, f"Forward host-syncs found: {audit['findings']}"
    assert audit["total_findings"] == 0, f"Host-syncs found: {audit['findings']}"
    assert audit["clean"] is True
    assert audit["is_zero_host_sync_forward"] is True
