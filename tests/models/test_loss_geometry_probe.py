"""TDD test suite for `mars_titan.models.mmarrs.loss_geometry_probe`.

Verifies mathematical exactness, numerical stability (FP64/FP32/BF16), zero weight
mutation, graceful fallback on non-double-differentiable ops, and end-to-end
compatibility across ALL 4 independent model families (`titans_original`,
`holographic_brain`, `mmarrs_titan`, `titans_cognitive_core`):

  1. `HessianSpectrumAnalyzer`: exact analytic quadratic eigenvalues (< 1e-4),
     Rademacher Hutchinson trace (< 1e-4 diagonal & unbiased dense), negative
     eigenvalue ratio, effective condition number, SAM neighborhood sharpness
     with bit-exact parameter & grad preservation, and custom non-double-diff
     op fallback (`finite_difference` and `skip`).
  2. `RepresentationGeometryAnalyzer`: Roy & Vetterli eRank, RankMe, Stable Rank,
     Dead-Unit Fraction, Spectral Decay Exponent (full-rank vs collapsed +
     adversarial scales), Continuous-Target Decile Neural Collapse (`NC1`, `NC2`,
     `ordinal_monotonicity_rho`, `NC3`), and Linear CKA invariances + loop
     stagnation detection (`CKA > 0.995`).
  3. `InformationBottleneckProbe`: Gaussian MI compression monotonicity (DPI),
     strict OOS ridge linear probe + RBF nonlinear probe sufficiency &
     nonlinear-vs-linear gap, and epoch trajectory phase classification.
  4. `GradientDynamicsTracker`, `DeepLearningHealthReport`, `diagnose_training_health`,
     `DeepLearningGeometryProbe` (all 4 families), and compact lossless RFC-1952
     `.jsonl.gz` roundtrip (`write_geometry_trace_gz` / `read_geometry_trace_gz`).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mars_titan.models.mmarrs.loss_geometry_probe import (
    CANONICAL_DIAGNOSES,
    DeepLearningGeometryProbe,
    DeepLearningHealthReport,
    EpochDynamicsSnapshot,
    GradientDynamicsTracker,
    HessianConfig,
    HessianSpectrumAnalyzer,
    HessianSpectrumResult,
    InformationBottleneckProbe,
    InformationBottleneckResult,
    LoopCKAResult,
    NeuralCollapseResult,
    RepresentationGeometryAnalyzer,
    RepresentationHealthResult,
    compute_linear_cka,
    diagnose_training_health,
    read_geometry_trace_gz,
    write_geometry_trace_gz,
)
from mars_titan.models.mmarrs.model import build_family_model


def _make_synthetic_session_batch(
    seed: int = 17,
    n_assets: int = 30,
    *,
    prices_dim: int = 12,
    fundamentals_dim: int = 6,
    macro_dim: int = 4,
    news_dim: int = 6,
    charts_dim: int = 6,
    physics_dim: int = 8,
    context_dim: int = 8,
    exposures_dim: int = 4,
    rmt_k: int = 4,
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)

    def _t(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(rng.standard_normal(shape), dtype=torch.float32)

    prices = _t((n_assets, prices_dim))
    physics = _t((n_assets, physics_dim))
    # Weak realistic cross-sectional alpha signal + noise
    w_sig = torch.linspace(0.5, -0.5, prices_dim, dtype=torch.float32)
    y_1d = 0.015 * (prices @ w_sig) / math.sqrt(prices_dim) + 0.01 * _t((n_assets,))
    mask = torch.ones((n_assets, 5), dtype=torch.float32)
    mask[0::5, 3] = 0.0
    return dict(
        prices=prices,
        fundamentals=_t((n_assets, fundamentals_dim)),
        macro=_t((n_assets, macro_dim)),
        news=_t((n_assets, news_dim)),
        charts=_t((n_assets, charts_dim)),
        modality_mask=mask,
        physics_v4=physics,
        high_snr_context=_t((n_assets, context_dim)),
        exposures=_t((n_assets, exposures_dim)),
        rmt_loadings=_t((n_assets, rmt_k)),
        regime_vec=_t((context_dim,)),
        target_returns=y_1d,
        asset_ids=torch.arange(n_assets, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# 1. Hessian Spectrum & Loss Landscape Curvature
# ---------------------------------------------------------------------------


class _QuadraticModel(nn.Module):
    """Analytic quadratic L(w) = 0.5 * w^T H w + b^T w with exact Hessian H."""

    def __init__(self, hessian: torch.Tensor, init_w: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("H", hessian.clone())
        self.w = nn.Parameter(init_w.clone())

    def loss(self) -> torch.Tensor:
        return 0.5 * torch.dot(self.w, self.H @ self.w)


def test_hessian_exact_analytic_quadratic_eigenvalues_and_trace():
    """Exact Pearlmutter HVP, Lanczos & Power Iteration eigenvalues, and Rademacher trace < 1e-4."""
    true_eigs = torch.tensor([12.0, 5.0, 2.0, 0.5, -1.0, -4.0], dtype=torch.float64)
    H_diag = torch.diag(true_eigs)
    init_w = torch.tensor([1.0, -0.5, 0.25, 2.0, -1.5, 0.75], dtype=torch.float64)
    model = _QuadraticModel(H_diag, init_w)

    analyzer = HessianSpectrumAnalyzer(
        HessianConfig(
            top_k=4,
            lanczos_steps=12,
            hutchinson_probes=16,
            sam_rho=0.05,
            seed=123,
        )
    )

    # Direct Pearlmutter HVP check
    v_probe = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 0.25], dtype=torch.float64)
    hv = analyzer.compute_hvp(model.loss, [model.w], [v_probe])[0]
    assert torch.max(torch.abs(hv - H_diag @ v_probe)).item() < 1e-10

    # Full Lanczos analysis on diagonal indefinite quadratic
    res = analyzer.analyze(model, model.loss)
    assert isinstance(res, HessianSpectrumResult)
    assert res.hvp_method == "exact_autograd"
    assert len(res.top_eigenvalues) == 4
    expected_top = [12.0, 5.0, 2.0, 0.5]
    max_eig_err = max(
        abs(a - b) for a, b in zip(res.top_eigenvalues, expected_top, strict=True)
    )
    assert max_eig_err < 1e-4
    assert abs(res.lambda_max - 12.0) < 1e-4
    assert abs(res.min_eigenvalue - (-4.0)) < 1e-4

    # Rademacher Hutchinson trace on diagonal H is mathematically exact (zero variance!)
    expected_trace = float(true_eigs.sum().item())
    assert abs(res.hessian_trace - expected_trace) < 1e-4
    assert res.hessian_trace_se < 1e-6

    # Negative eigenvalue ratio (2 of 6 negative) and outlier-to-bulk effective condition number
    assert abs(res.negative_eigenvalue_ratio - (2.0 / 6.0)) < 1e-4
    # abs(true_eigs) sorted = [0.5, 1.0, 2.0, 4.0, 5.0, 12.0] -> median = 3.0 > |trace|/6 = 2.4167
    expected_cond = 12.0 / 3.0
    assert abs(res.effective_condition_number - expected_cond) < 1e-4

    # P0-3 regression: effective_condition_number is UNBOUNDED by lanczos_steps (exceeds 50!)
    spike_eigs = torch.ones(120, dtype=torch.float64)
    spike_eigs[0] = 1000.0
    spike_model = _QuadraticModel(torch.diag(spike_eigs), torch.ones(120, dtype=torch.float64))
    spike_res = HessianSpectrumAnalyzer(
        HessianConfig(top_k=3, lanczos_steps=12, hutchinson_probes=8, seed=7)
    ).analyze(spike_model, spike_model.loss)
    assert spike_res.effective_condition_number > 50.0

    # Deflated Power Iteration on SPD diagonal quadratic also recovers top-k < 1e-4
    spd_eigs = torch.tensor([15.0, 7.5, 3.0, 1.0, 0.4, 0.1], dtype=torch.float64)
    spd_model = _QuadraticModel(torch.diag(spd_eigs), init_w)
    pi_analyzer = HessianSpectrumAnalyzer(
        HessianConfig(
            top_k=3,
            method="power_iteration",
            power_iters=80,
            hutchinson_probes=8,
            seed=42,
        )
    )
    res_pi = pi_analyzer.analyze(spd_model, spd_model.loss)
    pi_err = max(
        abs(a - b)
        for a, b in zip(res_pi.top_eigenvalues, [15.0, 7.5, 3.0], strict=True)
    )
    assert pi_err < 1e-4
    assert res_pi.negative_eigenvalue_ratio == pytest.approx(0.0, abs=1e-6)


def test_hessian_dense_rotated_quadratic_recovery_fp32():
    """Dense non-diagonal rotated Hessian Q Lambda Q^T recovers top-k eigenvalues < 1e-4 in FP32."""
    gen = torch.Generator().manual_seed(99)
    d = 10
    mat = torch.randn((d, d), generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(mat)
    eigs = torch.tensor(
        [20.0, 11.0, 6.0, 3.5, 2.0, 1.0, 0.5, 0.2, -0.5, -2.5], dtype=torch.float64
    )
    H_dense = (q @ torch.diag(eigs) @ q.T).to(dtype=torch.float32)
    init_w = torch.randn((d,), generator=gen, dtype=torch.float32)
    model = _QuadraticModel(H_dense, init_w)

    analyzer = HessianSpectrumAnalyzer(
        HessianConfig(
            top_k=4,
            lanczos_steps=10,
            hutchinson_probes=64,
            sam_rho=0.05,
            seed=7,
        )
    )
    res = analyzer.analyze(model, model.loss)
    max_err = max(
        abs(a - b)
        for a, b in zip(res.top_eigenvalues, [20.0, 11.0, 6.0, 3.5], strict=True)
    )
    assert max_err < 1e-4
    assert abs(res.min_eigenvalue - (-2.5)) < 1e-4
    exact_trace = float(eigs.sum().item())
    # Exact Lanczos full-space spectral trace matches exact trace < 1e-4
    assert abs(res.spectral_trace - exact_trace) < 1e-4
    # Hutchinson Rademacher trace is within 3.5 standard errors of exact trace
    assert abs(res.hessian_trace - exact_trace) <= max(3.5 * res.hessian_trace_se, 1e-3)


def test_hessian_non_double_differentiable_custom_op_fallback():
    """Non-double-differentiable custom autograd op triggers safe FD fallback or skip cleanly."""

    class _NonDoubleDiffSquare(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            ctx.save_for_backward(x.detach())
            return 0.5 * (x * x)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
            (x,) = ctx.saved_tensors
            # Detached x -> breaks second-order autograd graph!
            return (grad_output.detach() * x).detach()

    class _CustomOpQuadratic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.tensor([1.0, -2.0, 0.5, 1.5], dtype=torch.float32))
            self.register_buffer("scales", torch.tensor([8.0, 4.0, 2.0, 1.0], dtype=torch.float32))

        def loss(self) -> torch.Tensor:
            sq = _NonDoubleDiffSquare.apply(self.w)
            return (self.scales * sq).sum()

    model = _CustomOpQuadratic()
    w_before = model.w.detach().clone()

    # 1) Finite-difference fallback still recovers eigenvalues & trace accurately!
    fd_analyzer = HessianSpectrumAnalyzer(
        HessianConfig(
            top_k=3,
            lanczos_steps=4,
            hutchinson_probes=8,
            fallback_mode="finite_difference",
            seed=11,
        )
    )
    res_fd = fd_analyzer.analyze(model, model.loss)
    assert res_fd.hvp_method == "central_fd_fallback"
    assert max(
        abs(a - b) for a, b in zip(res_fd.top_eigenvalues, [8.0, 4.0, 2.0], strict=True)
    ) < 1e-2
    assert abs(res_fd.hessian_trace - 15.0) < 1e-2
    assert torch.equal(model.w.detach(), w_before)

    # 2) Skip mode returns clean skipped report without raising
    skip_analyzer = HessianSpectrumAnalyzer(
        HessianConfig(top_k=3, fallback_mode="skip", sam_rho=0.05, seed=11)
    )
    res_skip = skip_analyzer.analyze(model, model.loss)
    assert res_skip.hvp_method == "skipped_non_double_diff"
    assert res_skip.fallback_reason is not None
    # SAM sharpness is first-order so it still computes cleanly even when HVP is skipped!
    assert math.isfinite(res_skip.sam_sharpness) and res_skip.sam_sharpness > 0.0
    assert torch.equal(model.w.detach(), w_before)


def test_sam_sharpness_exact_weight_preservation_fp32_bf16_and_error_safety():
    """SAM sharpness discriminates sharp vs flat minima and NEVER mutates weights or grads."""
    w0 = torch.tensor([0.3, -0.4, 0.2, 0.5], dtype=torch.float32)
    sharp_model = _QuadraticModel(torch.diag(torch.tensor([40.0, 30.0, 20.0, 10.0])), w0)
    flat_model = _QuadraticModel(torch.diag(torch.tensor([0.4, 0.3, 0.2, 0.1])), w0)

    # Attach pre-existing .grad to confirm SAM does NOT clobber it
    sentinel_grad = torch.tensor([9.0, -8.0, 7.0, -6.0], dtype=torch.float32)
    sharp_model.w.grad = sentinel_grad.clone()
    w_sharp_orig = sharp_model.w.detach().clone()

    analyzer = HessianSpectrumAnalyzer(HessianConfig(sam_rho=0.05))
    s_sharp, rel_sharp = analyzer.compute_sam_sharpness(sharp_model, sharp_model.loss)
    s_flat, _ = analyzer.compute_sam_sharpness(flat_model, flat_model.loss)

    assert s_sharp > 10.0 * s_flat > 0.0
    assert math.isfinite(rel_sharp)
    # Exact bit-for-bit preservation (no floating-point add/sub drift!)
    assert torch.equal(sharp_model.w.detach(), w_sharp_orig)
    assert sharp_model.w.grad is not None and torch.equal(sharp_model.w.grad, sentinel_grad)

    # BF16 bit-exact preservation check
    bf16_model = _QuadraticModel(
        torch.diag(torch.tensor([5.0, 3.0, 2.0, 1.0], dtype=torch.bfloat16)),
        w0.to(dtype=torch.bfloat16),
    )
    w_bf16_orig = bf16_model.w.detach().clone()
    s_bf16, _ = analyzer.compute_sam_sharpness(bf16_model, bf16_model.loss)
    assert math.isfinite(s_bf16) and s_bf16 > 0.0
    assert torch.equal(bf16_model.w.detach(), w_bf16_orig)

    # Exception safety: restore weights even if second loss call fails
    call_count = 0

    def _flaky_loss() -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("injected failure during perturbed evaluation")
        return sharp_model.loss()

    with pytest.raises(RuntimeError, match="injected failure"):
        analyzer.compute_sam_sharpness(sharp_model, _flaky_loss)
    assert torch.equal(sharp_model.w.detach(), w_sharp_orig)

    # P1-2 regression: stochastic dropout layer inside loss_closure preserves CRN + caller RNG
    class _StochasticQuad(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.tensor([1.0, -1.0, 0.5, -0.5, 1.2, -0.8, 0.3, -0.4]))
            self.drop = nn.Dropout(p=0.5)

        def loss(self) -> torch.Tensor:
            h = self.drop(self.w)
            return 0.5 * (h * h).sum()

    stoch_model = _StochasticQuad().train()
    torch.manual_seed(999)
    expected_next = torch.rand(4)
    torch.manual_seed(999)
    s_stoch, _ = analyzer.compute_sam_sharpness(stoch_model, stoch_model.loss)
    actual_next = torch.rand(4)
    # With Common Random Numbers (fork_rng), sharpness is strictly positive and caller RNG untouched
    assert math.isfinite(s_stoch) and s_stoch > 0.0
    assert torch.equal(actual_next, expected_next)

    # #171 P1 regression: inside bf16 autocast at tiny rho=1e-7, loss_0 and loss_pert MUST share
    # identical FP32 master precision (no BF16-vs-FP32 quantization mismatch!)
    lin_mod = nn.Linear(16, 4, bias=True)
    x_ac = torch.randn(32, 16, dtype=torch.float32)
    y_ac = torch.randn(32, 4, dtype=torch.float32)

    def _lin_mse() -> torch.Tensor:
        return 0.5 * (lin_mod(x_ac) - y_ac).pow(2).mean()

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        s_tiny, _ = analyzer.compute_sam_sharpness(lin_mod, _lin_mse, rho=1e-7)
    assert math.isfinite(s_tiny)
    assert 0.0 <= s_tiny < 1e-4


# ---------------------------------------------------------------------------
# 2. Latent Representation Health, Neural Collapse & Linear CKA
# ---------------------------------------------------------------------------


def test_representation_health_full_rank_vs_collapsed_and_adversarial_scales():
    """eRank, RankMe, Stable Rank, Dead Units & Spectral Decay distinguish full-rank vs collapse."""
    gen = torch.Generator().manual_seed(2026)
    n, d = 256, 32
    # 1) Full-rank orthonormal columns (exact rank d)
    q_full, _ = torch.linalg.qr(torch.randn((n, d), generator=gen, dtype=torch.float64))
    res_full = RepresentationGeometryAnalyzer.analyze_representation_health(q_full)
    assert isinstance(res_full, RepresentationHealthResult)
    assert res_full.effective_rank > 28.0
    assert res_full.rankme > 28.0
    assert res_full.stable_rank > 28.0
    assert res_full.dead_unit_fraction == pytest.approx(0.0, abs=1e-9)

    # 2) Rank-1 collapsed representation
    u = torch.randn((n, 1), generator=gen, dtype=torch.float64)
    v = torch.randn((1, d), generator=gen, dtype=torch.float64)
    z_rank1 = u @ v
    res_col = RepresentationGeometryAnalyzer.analyze_representation_health(z_rank1)
    assert res_col.effective_rank == pytest.approx(1.0, abs=1e-4)
    assert res_col.rankme == pytest.approx(1.0, abs=1e-4)
    assert res_col.stable_rank == pytest.approx(1.0, abs=1e-4)

    # 3) Half dead units
    z_half_dead = q_full.clone()
    z_half_dead[:, d // 2 :] = 0.0
    res_dead = RepresentationGeometryAnalyzer.analyze_representation_health(z_half_dead)
    assert res_dead.dead_unit_fraction == pytest.approx(0.5, abs=1e-6)
    assert res_dead.effective_rank < 17.5

    # 4) Exact power-law spectrum: variance lambda_k ~ k^{-1.5} => singular values ~ k^{-0.75}
    ks = torch.arange(1, d + 1, dtype=torch.float64)
    sigmas = ks.pow(-0.75)
    z_pow = q_full * sigmas.unsqueeze(0)
    res_pow = RepresentationGeometryAnalyzer.analyze_representation_health(z_pow)
    assert abs(res_pow.spectral_decay_exponent - 1.5) < 0.08

    # 5) Adversarial scale & dtype tests (1e4, 1e-4, BF16, constant offset, all-zeros)
    z_big = (q_full * 1e4 + 5e3).to(dtype=torch.float32)
    z_small = (q_full * 1e-4).to(dtype=torch.float32)
    z_bf16 = q_full.to(dtype=torch.bfloat16)
    r_big = RepresentationGeometryAnalyzer.analyze_representation_health(z_big)
    r_small = RepresentationGeometryAnalyzer.analyze_representation_health(z_small)
    r_bf16 = RepresentationGeometryAnalyzer.analyze_representation_health(z_bf16)
    assert abs(r_big.effective_rank - res_full.effective_rank) < 0.2
    assert abs(r_small.effective_rank - res_full.effective_rank) < 0.2
    assert abs(r_bf16.effective_rank - res_full.effective_rank) < 1.0
    # Uncentered RankMe detects huge constant-offset dominance while centered eRank stays full!
    assert r_big.rankme < r_big.effective_rank

    r_zero = RepresentationGeometryAnalyzer.analyze_representation_health(torch.zeros((32, 8)))
    assert r_zero.effective_rank == 0.0
    assert r_zero.rankme == 0.0
    assert r_zero.stable_rank == 0.0
    assert r_zero.dead_unit_fraction == 1.0


def test_continuous_decile_neural_collapse_separated_etf_and_collapsed():
    """NC1/NC2/NC3 cleanly separate ordered/ETF representations from collapsed noise."""
    gen = torch.Generator().manual_seed(314)
    q_bins = 10
    per_bin = 25
    n = q_bins * per_bin
    d = 16

    # Target smoothly increasing across assets
    y = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)

    # 1) Well-separated monotonic ordinal representation + aligned head
    axis = torch.zeros(d, dtype=torch.float64)
    axis[0] = 1.0
    axis[1] = 0.5
    axis = axis / axis.norm()
    z_sep = y.unsqueeze(-1) * axis.unsqueeze(0) * 5.0 + 0.05 * torch.randn(
        (n, d), generator=gen, dtype=torch.float64
    )
    w_head = axis.clone()
    nc_sep = RepresentationGeometryAnalyzer.analyze_neural_collapse(
        z_sep, y, w_head=w_head, n_quantiles=q_bins
    )
    assert isinstance(nc_sep, NeuralCollapseResult)
    assert nc_sep.nc1_active_rank == 1
    assert nc_sep.nc1_within_between_ratio < 0.05
    assert nc_sep.ordinal_monotonicity_rho > 0.98
    assert nc_sep.nc3_head_duality_cosine is not None
    assert nc_sep.nc3_head_duality_cosine > 0.99

    # 2) Pure Simplex ETF decile means -> nc2_etf_simplex_deviation < 1e-4!
    q_etf = 5
    eye_q = torch.eye(q_etf, dtype=torch.float64)
    ones_q = torch.ones((q_etf, q_etf), dtype=torch.float64)
    g_etf = (q_etf / (q_etf - 1.0)) * eye_q - (1.0 / (q_etf - 1.0)) * ones_q
    eigvals, eigvecs = torch.linalg.eigh(g_etf)
    etf_means = eigvecs @ torch.diag(eigvals.clamp_min(0.0).sqrt())  # [Q, Q]
    # Pad into D=16 dimensions and repeat per bin with tiny within-class noise
    etf_pad = torch.zeros((q_etf, d), dtype=torch.float64)
    etf_pad[:, :q_etf] = etf_means
    y_etf = torch.repeat_interleave(torch.arange(q_etf, dtype=torch.float64), 20)
    y_etf = y_etf + 1e-4 * torch.arange(len(y_etf), dtype=torch.float64) / len(y_etf)
    z_etf = torch.repeat_interleave(etf_pad, 20, dim=0) + 1e-5 * torch.randn(
        (len(y_etf), d), generator=gen, dtype=torch.float64
    )
    nc_etf = RepresentationGeometryAnalyzer.analyze_neural_collapse(
        z_etf, y_etf, n_quantiles=q_etf
    )
    assert nc_etf.nc1_active_rank == q_etf - 1
    assert nc_etf.nc2_etf_simplex_deviation < 1e-3
    assert nc_etf.nc1_within_between_ratio < 1e-6

    # 3) Collapsed / pure noise independent of Y
    z_noise = torch.randn((n, d), generator=gen, dtype=torch.float64)
    nc_col = RepresentationGeometryAnalyzer.analyze_neural_collapse(
        z_noise, y, w_head=w_head, n_quantiles=q_bins
    )
    assert nc_col.nc1_within_between_ratio > 10.0 * nc_sep.nc1_within_between_ratio
    assert abs(nc_col.ordinal_monotonicity_rho) < 0.7
    assert nc_col.nc3_head_duality_cosine is not None
    assert abs(nc_col.nc3_head_duality_cosine) < 0.7


def test_linear_cka_invariances_and_loop_stagnation_detection():
    """Linear CKA is invariant to orthogonal transform, scaling, shift, and catches flat loops."""
    gen = torch.Generator().manual_seed(77)
    n, d1, d2 = 128, 16, 24
    x = torch.randn((n, d1), generator=gen, dtype=torch.float64)
    # Orthogonal matrix in R^{d1 x d1} and isometric embedding into R^{d2}
    q_orth, _ = torch.linalg.qr(torch.randn((d1, d1), generator=gen, dtype=torch.float64))
    q_embed, _ = torch.linalg.qr(torch.randn((d2, d1), generator=gen, dtype=torch.float64))
    shift = torch.randn((1, d1), generator=gen, dtype=torch.float64) * 50.0

    assert compute_linear_cka(x, x) == pytest.approx(1.0, abs=1e-6)
    assert compute_linear_cka(x, x @ q_orth) == pytest.approx(1.0, abs=1e-6)
    assert compute_linear_cka(x, 7.5 * x + shift) == pytest.approx(1.0, abs=1e-6)
    assert compute_linear_cka(x, x @ q_embed.T) == pytest.approx(1.0, abs=1e-6)

    # Independent noise has low CKA
    z_indep = torch.randn((n, d2), generator=gen, dtype=torch.float64)
    assert compute_linear_cka(x, z_indep) < 0.25

    # Loop CKA: stagnant loops vs evolving loops
    stagnant = [
        x,
        x + 1e-3 * torch.randn((n, d1), generator=gen, dtype=torch.float64),
        x + 2e-3 * torch.randn((n, d1), generator=gen, dtype=torch.float64),
    ]
    res_stag = RepresentationGeometryAnalyzer.analyze_loop_cka(
        stagnant, stagnation_threshold=0.995
    )
    assert isinstance(res_stag, LoopCKAResult)
    assert res_stag.stagnant_loop_detected is True
    assert res_stag.mean_adjacent_cka > 0.995

    evolving = [
        x,
        torch.tanh(x @ q_orth + 0.8 * torch.randn((n, d1), generator=gen, dtype=torch.float64)),
        torch.randn((n, d1), generator=gen, dtype=torch.float64),
    ]
    res_evol = RepresentationGeometryAnalyzer.analyze_loop_cka(
        evolving, stagnation_threshold=0.995
    )
    assert res_evol.stagnant_loop_detected is False
    assert res_evol.mean_adjacent_cka < 0.95


# ---------------------------------------------------------------------------
# 3. Information Bottleneck Trajectory
# ---------------------------------------------------------------------------


def test_information_bottleneck_compression_sufficiency_and_phase():
    """I_Gauss(X;Z) obeys DPI monotonicity; OOS linear/nonlinear probes detect sufficiency & gap."""
    gen = torch.Generator().manual_seed(404)
    n, d = 320, 12
    x = torch.randn((n, d), generator=gen, dtype=torch.float64)
    noise1 = torch.randn((n, d), generator=gen, dtype=torch.float64)
    noise2 = torch.randn((n, d), generator=gen, dtype=torch.float64)

    probe = InformationBottleneckProbe(ridge_mi=1e-2, train_frac=0.7, seed=404)

    # 1) Compression monotonicity under additive noise (Data Processing Inequality)
    mi_identity = probe.compute_gaussian_mi(x, x)
    mi_noisy_lo = probe.compute_gaussian_mi(x, x + 0.5 * noise1)
    mi_noisy_hi = probe.compute_gaussian_mi(x, x + 2.5 * noise1)
    mi_indep = probe.compute_gaussian_mi(x, noise2)
    assert mi_identity > mi_noisy_lo > mi_noisy_hi > mi_indep >= 0.0

    # 2) Compression monotonicity under rank bottleneck projection
    z_rank8 = x[:, :8]
    z_rank3 = x[:, :3]
    mi_r8 = probe.compute_gaussian_mi(x, z_rank8)
    mi_r3 = probe.compute_gaussian_mi(x, z_rank3)
    assert mi_identity > mi_r8 > mi_r3 > 0.0

    # 3) Linear sufficiency on linear target Y = X w + eps
    w_true = torch.randn((d,), generator=gen, dtype=torch.float64)
    w_true = w_true / w_true.norm()
    y_lin = x @ w_true + 0.25 * torch.randn((n,), generator=gen, dtype=torch.float64)
    ib_lin = probe.probe(x, x + 0.1 * noise1, y_lin)
    assert isinstance(ib_lin, InformationBottleneckResult)
    assert ib_lin.linear_probe_r2 > 0.7
    assert ib_lin.linear_probe_rank_ic > 0.75
    assert ib_lin.nonlinear_linear_gap_r2 < 0.15

    # 4) Purely nonlinear target Y = Z_0^2 - Z_1^2 => nonlinear_probe_r2 >> linear_probe_r2!
    z_nl = torch.randn((n, 6), generator=gen, dtype=torch.float64)
    y_nl = (z_nl[:, 0] ** 2 - z_nl[:, 1] ** 2) + 0.1 * torch.randn(
        (n,), generator=gen, dtype=torch.float64
    )
    ib_nl = probe.probe(z_nl, z_nl, y_nl)
    assert ib_nl.linear_probe_r2 < 0.15
    assert ib_nl.nonlinear_probe_r2 > 0.55
    assert ib_nl.nonlinear_linear_gap_r2 > 0.35

    # 5) Pure noise Z independent of Y => OOS linear probe R^2 <= 0.1 (no in-sample leakage!)
    ib_noise = probe.probe(x, noise2, y_lin)
    assert ib_noise.linear_probe_r2 < 0.1
    assert abs(ib_noise.linear_probe_rank_ic) < 0.25

    # 6) Trajectory phase detection: fitting -> compression -> memorization
    phase_fit = probe.classify_trajectory(
        [
            {"i_xz_gauss": 1.0, "i_zy_linear": 0.05, "val_loss": 1.0},
            {"i_xz_gauss": 2.5, "i_zy_linear": 0.25, "val_loss": 0.7},
        ]
    )
    phase_comp = probe.classify_trajectory(
        [
            {"i_xz_gauss": 3.0, "i_zy_linear": 0.25, "val_loss": 0.70},
            {"i_xz_gauss": 1.8, "i_zy_linear": 0.35, "val_loss": 0.62},
        ]
    )
    phase_mem = probe.classify_trajectory(
        [
            {"i_xz_gauss": 2.0, "i_zy_linear": 0.35, "val_loss": 0.60},
            {"i_xz_gauss": 3.4, "i_zy_linear": 0.15, "val_loss": 0.85},
        ]
    )
    assert phase_fit == "fitting"
    assert phase_comp == "compression"
    assert phase_mem == "memorization"


# ---------------------------------------------------------------------------
# 4. Gradient Dynamics, GSNR, Health Diagnosis & .jsonl.gz Roundtrip
# ---------------------------------------------------------------------------


def test_gradient_dynamics_gsnr_welford_and_update_ratio():
    """GSNR separates coherent vs noisy grads, avoids cancellation, and tracks Karpathy ratio."""
    gen = torch.Generator().manual_seed(55)
    p_dim = 64

    # 1) Coherent gradients vs noisy gradients
    coherent_tracker = GradientDynamicsTracker()
    noisy_tracker = GradientDynamicsTracker()
    w_param = nn.Parameter(torch.ones(p_dim, dtype=torch.float32))
    g_base = torch.randn(p_dim, generator=gen, dtype=torch.float32)
    g_base = g_base / g_base.norm()

    coh_grads = []
    for _ in range(24):
        g_c = g_base + 0.01 * torch.randn(p_dim, generator=gen, dtype=torch.float32)
        g_n = 0.02 * g_base + 2.0 * torch.randn(p_dim, generator=gen, dtype=torch.float32)
        coh_grads.append(g_c.to(dtype=torch.float64))
        coherent_tracker.record_step([w_param], grads=[g_c])
        noisy_tracker.record_step([w_param], grads=[g_n])

    snap_coh = coherent_tracker.finalize_epoch(
        epoch=0, train_loss=1.0, val_loss=1.05, model_or_params=[w_param]
    )
    snap_noisy = noisy_tracker.finalize_epoch(
        epoch=0, train_loss=1.0, val_loss=1.05, model_or_params=[w_param]
    )
    assert isinstance(snap_coh, EpochDynamicsSnapshot)
    assert snap_coh.gsnr > 100.0
    assert snap_coh.grad_cosine_stability > 0.95
    assert snap_noisy.gsnr < 0.5
    assert snap_coh.gsnr > 500.0 * snap_noisy.gsnr

    # Exact parity against FP64 two-pass reference!
    stacked = torch.stack(coh_grads, dim=0)
    ref_mean = stacked.mean(dim=0)
    ref_var = ((stacked - ref_mean) ** 2).sum(dim=-1).mean()
    ref_gsnr = float((ref_mean.pow(2).sum() / ref_var).item())
    assert abs(snap_coh.gsnr - ref_gsnr) / ref_gsnr < 1e-4

    # 2) Adversarial catastrophic-cancellation stress test: g_t = 1000 + 1e-3 * eps_t
    cancel_tracker = GradientDynamicsTracker()
    adv_grads = [
        torch.full((p_dim,), 1000.0, dtype=torch.float32)
        + 1e-3 * torch.randn(p_dim, generator=gen, dtype=torch.float32)
        for _ in range(16)
    ]
    for g in adv_grads:
        cancel_tracker.record_step([w_param], grads=[g])
    snap_adv = cancel_tracker.finalize_epoch(epoch=0, train_loss=0.5, val_loss=0.5)
    assert snap_adv.grad_variance >= 0.0 and math.isfinite(snap_adv.grad_variance)
    assert math.isfinite(snap_adv.gsnr) and snap_adv.gsnr > 1e6

    # 3) Update-to-weight ratio & exponential convergence_rate (log-loss slope)
    dyn_tracker = GradientDynamicsTracker()
    p_dyn = nn.Parameter(torch.full((16,), 2.0, dtype=torch.float32))  # ||W|| = 8.0
    for ep in range(4):
        dyn_tracker.record_step([p_dyn], grads=[torch.ones(16)])
        with torch.no_grad():
            p_dyn.add_(0.02)  # ||dW|| = 0.08 -> ratio = 0.08 / 8.0 = 0.01 (first ep)
        dyn_tracker.record_post_step([p_dyn])
        loss_e = math.exp(-0.5 * ep)
        snap = dyn_tracker.finalize_epoch(
            epoch=ep, train_loss=loss_e, val_loss=loss_e + 0.01, model_or_params=[p_dyn]
        )
    assert snap.update_to_weight_ratio == pytest.approx(0.08 / float(p_dyn.norm().item()), rel=2e-2)
    assert snap.convergence_rate == pytest.approx(-0.5, abs=1e-4)

    # P0-2 regression: negative MMARRSMultiTaskLoss values produce accurate negative slope (not 0!)
    neg_tracker = GradientDynamicsTracker()
    for ep, neg_loss in enumerate([0.2, 0.0, -0.2, -0.4]):
        neg_tracker.record_step([p_dyn], grads=[torch.ones(16)])
        neg_snap = neg_tracker.finalize_epoch(epoch=ep, train_loss=neg_loss, val_loss=neg_loss)
    assert neg_snap.convergence_rate == pytest.approx(-0.2, abs=1e-4)


def test_gradient_dynamics_tracker_zero_host_syncs_in_hot_loop(monkeypatch: pytest.MonkeyPatch):
    """`record_step` and `record_post_step` perform ZERO `.item()` / `.cpu()` host syncs."""
    tracker = GradientDynamicsTracker(vec_dtype=torch.float32)
    params = [
        nn.Parameter(torch.randn(32, dtype=torch.float32)),
        nn.Parameter(torch.randn(16, dtype=torch.float32)),
    ]
    grads_seq = [
        [torch.randn(32, dtype=torch.float32), torch.randn(16, dtype=torch.float32)]
        for _ in range(5)
    ]

    def _forbid_sync(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Forbidden host-sync (.item/.cpu) inside hot training step!")

    monkeypatch.setattr(torch.Tensor, "item", _forbid_sync)
    monkeypatch.setattr(torch.Tensor, "cpu", _forbid_sync)
    for g_step in grads_seq:
        tracker.record_step(params, grads=g_step)
        with torch.no_grad():
            for p, g in zip(params, g_step, strict=True):
                p.add_(g, alpha=-1e-3)
        tracker.record_post_step(params)

    monkeypatch.undo()
    snap = tracker.finalize_epoch(epoch=0, train_loss=0.42, val_loss=0.45, model_or_params=params)
    assert snap.n_steps == 5
    assert math.isfinite(snap.gsnr) and snap.gsnr > 0.0
    assert math.isfinite(snap.update_to_weight_ratio) and snap.update_to_weight_ratio > 0.0


def test_health_diagnosis_all_six_canonical_regimes():
    """Actionable diagnosis cleanly distinguishes all 6 canonical deep learning regimes."""
    assert set(CANONICAL_DIAGNOSES) == {
        "healthy_generalizing",
        "underfitting",
        "memorizing_overfit",
        "representation_collapse",
        "sharp_minimum",
        "gradient_instability",
    }

    healthy_rep = RepresentationHealthResult(
        n_samples=100,
        dim=32,
        effective_rank=22.0,
        effective_rank_ratio=22.0 / 32.0,
        rankme=21.5,
        stable_rank=18.0,
        dead_unit_fraction=0.0,
        spectral_decay_exponent=1.0,
        singular_values_top_k=[1.0, 0.9, 0.8],
    )
    healthy_hess = HessianSpectrumResult(
        top_eigenvalues=[5.0, 3.0, 2.0],
        lambda_max=5.0,
        min_eigenvalue=-0.1,
        hessian_trace=25.0,
        hessian_trace_se=0.5,
        spectral_trace=25.0,
        negative_eigenvalue_ratio=0.05,
        saddle_curvature_ratio=0.02,
        effective_condition_number=2.5,
        sam_sharpness=0.01,
        sam_relative_sharpness=0.02,
        sam_rho=0.05,
        hvp_method="exact_autograd",
    )
    healthy_ib = InformationBottleneckResult(
        i_xz_gauss=1.5,
        i_xz_per_dim=0.1,
        input_latent_cka=0.5,
        linear_probe_r2=0.25,
        linear_probe_rank_ic=0.18,
        nonlinear_probe_r2=0.28,
        nonlinear_probe_rank_ic=0.20,
        nonlinear_linear_gap_r2=0.03,
        nonlinear_linear_gap_ic=0.02,
        i_zy_linear=0.14,
        i_zy_nonlinear=0.16,
        compression_to_sufficiency_ratio=10.7,
        ib_phase="compression",
    )
    healthy_dyn = EpochDynamicsSnapshot(
        epoch=5,
        train_loss=0.40,
        val_loss=0.42,
        generalization_gap=0.02,
        relative_generalization_gap=0.05,
        grad_norm=1.2,
        grad_variance=0.4,
        gsnr=3.5,
        grad_cosine_stability=0.6,
        weight_norm=10.0,
        update_to_weight_ratio=0.002,
        convergence_rate=-0.08,
        n_steps=20,
    )

    # 1) healthy_generalizing
    d1, flags1, _, _ = diagnose_training_health(
        hessian=healthy_hess,
        representation=healthy_rep,
        information_bottleneck=healthy_ib,
        dynamics=healthy_dyn,
    )
    assert d1 == "healthy_generalizing"
    assert flags1 == []

    # 2) gradient_instability
    unstable_dyn = EpochDynamicsSnapshot(
        **{**healthy_dyn.to_dict(), "grad_norm": 5e4, "update_to_weight_ratio": 0.4}
    )
    d2, _, _, _ = diagnose_training_health(dynamics=unstable_dyn)
    assert d2 == "gradient_instability"

    # 3) representation_collapse
    collapsed_rep = RepresentationHealthResult(
        **{
            **healthy_rep.to_dict(),
            "effective_rank": 1.1,
            "effective_rank_ratio": 1.1 / 32.0,
            "dead_unit_fraction": 0.75,
        }
    )
    d3, _, _, _ = diagnose_training_health(representation=collapsed_rep, dynamics=healthy_dyn)
    assert d3 == "representation_collapse"

    # 4) memorizing_overfit
    overfit_dyn = EpochDynamicsSnapshot(
        **{
            **healthy_dyn.to_dict(),
            "train_loss": 0.10,
            "val_loss": 0.55,
            "generalization_gap": 0.45,
            "relative_generalization_gap": 4.5,
            "convergence_rate": -0.15,
        }
    )
    d4, _, _, _ = diagnose_training_health(
        representation=healthy_rep,
        hessian=healthy_hess,
        information_bottleneck=InformationBottleneckResult(
            **{**healthy_ib.to_dict(), "ib_phase": "memorization"}
        ),
        dynamics=overfit_dyn,
    )
    assert d4 == "memorizing_overfit"

    # 5) sharp_minimum
    sharp_hess = HessianSpectrumResult(
        **{
            **healthy_hess.to_dict(),
            "lambda_max": 2500.0,
            "sam_sharpness": 1.5,
            "sam_relative_sharpness": 3.0,
            "effective_condition_number": 80.0,
        }
    )
    d5, _, _, _ = diagnose_training_health(
        hessian=sharp_hess, representation=healthy_rep, dynamics=healthy_dyn
    )
    assert d5 == "sharp_minimum"

    # 6) underfitting
    underfit_dyn = EpochDynamicsSnapshot(
        **{
            **healthy_dyn.to_dict(),
            "epoch": 10,
            "train_loss": 1.2,
            "val_loss": 1.2,
            "generalization_gap": 0.0,
            "relative_generalization_gap": 0.0,
            "gsnr": 0.02,
            "convergence_rate": -1e-5,
        }
    )
    underfit_ib = InformationBottleneckResult(
        **{**healthy_ib.to_dict(), "linear_probe_r2": -0.02, "linear_probe_rank_ic": 0.005}
    )
    d6, _, _, _ = diagnose_training_health(
        hessian=healthy_hess,
        representation=healthy_rep,
        information_bottleneck=underfit_ib,
        dynamics=underfit_dyn,
    )
    assert d6 == "underfitting"

    # P0-1 regression: world-class daily financial champion (IC=+0.07, R^2=0.0049, GSNR=0.03)
    # MUST diagnose as `healthy_generalizing`, NEVER false `underfitting`!
    fin_champion_ib = InformationBottleneckResult(
        **{
            **healthy_ib.to_dict(),
            "linear_probe_r2": 0.0049,
            "linear_probe_rank_ic": 0.070,
            "nonlinear_probe_r2": 0.0060,
            "nonlinear_probe_rank_ic": 0.075,
        }
    )
    d_champ, flags_champ, _, _ = diagnose_training_health(
        hessian=healthy_hess,
        representation=healthy_rep,
        information_bottleneck=fin_champion_ib,
        dynamics=underfit_dyn,
    )
    assert d_champ == "healthy_generalizing"
    assert flags_champ == []

    # #171 minor regression: N=1 degenerate representation sample must NOT fire collapse
    n1_rep = RepresentationGeometryAnalyzer.analyze_representation_health(torch.randn(1, 16))
    d_n1, flags_n1, _, _ = diagnose_training_health(representation=n1_rep, dynamics=healthy_dyn)
    assert d_n1 == "healthy_generalizing"
    assert "representation_collapse" not in flags_n1


def test_jsonl_gz_lossless_compact_roundtrip_and_append(tmp_path: Path):
    """RFC-1952 `.jsonl.gz` writer/reader is deterministic, compact, lossless & append-safe."""
    trace_path = tmp_path / "geometry_trace.jsonl.gz"
    trace_path_2 = tmp_path / "geometry_trace_copy.jsonl.gz"

    rep1 = DeepLearningHealthReport(
        family="mmarrs_titan",
        variant="v06_double_loop_pc_trm_only",
        epoch=1,
        step=100,
        diagnosis="healthy_generalizing",
        secondary_flags=[],
        evidence={"lambda_max": 4.25, "nan_sentinel": float("nan"), "inf_sentinel": float("inf")},
        recommendations=["Continue current schedule."],
    )
    rep2 = DeepLearningHealthReport(
        family="titans_original",
        variant="t15a_dual_axis_mem",
        epoch=2,
        step=200,
        diagnosis="sharp_minimum",
        secondary_flags=["sharp_minimum"],
        evidence={"sam_sharpness": 0.72, "eRank": 19.3},
        recommendations=["Enable SAM or ScheduleFreeAdamW."],
    )

    write_geometry_trace_gz(trace_path, [rep1, rep2])
    write_geometry_trace_gz(trace_path_2, [rep1, rep2])

    # RFC-1952 gzip magic bytes + byte-for-byte determinism!
    raw_bytes_1 = trace_path.read_bytes()
    raw_bytes_2 = trace_path_2.read_bytes()
    assert raw_bytes_1[:2] == b"\x1f\x8b"
    assert raw_bytes_1 == raw_bytes_2

    # Lossless roundtrip (with RFC 8259 NaN/Inf -> None sanitization)
    loaded = read_geometry_trace_gz(trace_path)
    assert len(loaded) == 2
    assert loaded[0]["family"] == "mmarrs_titan"
    assert loaded[0]["evidence"]["lambda_max"] == pytest.approx(4.25)
    assert loaded[0]["evidence"]["nan_sentinel"] is None
    assert loaded[0]["evidence"]["inf_sentinel"] is None
    assert loaded[1]["diagnosis"] == "sharp_minimum"

    # Append mode creates valid multi-member RFC-1952 stream
    rep3 = DeepLearningHealthReport(
        family="holographic_brain",
        variant="bare_holographic_3l",
        epoch=3,
        step=300,
        diagnosis="healthy_generalizing",
    )
    write_geometry_trace_gz(trace_path, [rep3], append=True)
    loaded_after = read_geometry_trace_gz(trace_path)
    assert len(loaded_after) == 3
    assert loaded_after[2]["family"] == "holographic_brain"

    # Compactness check on repeated trace records
    many_records = [rep1.to_dict() for _ in range(40)]
    big_gz = tmp_path / "many.jsonl.gz"
    big_plain = tmp_path / "many.jsonl"
    write_geometry_trace_gz(big_gz, many_records)
    big_plain.write_text("\n".join(str(r) for r in many_records), encoding="utf-8")
    assert big_gz.stat().st_size < big_plain.stat().st_size * 0.25


# ---------------------------------------------------------------------------
# 5. End-to-End Integration Across ALL 4 Independent Model Families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "variant"),
    [
        ("titans_original", "t15a_dual_axis_mem"),
        ("titans_original", "t15b_full_v2"),
        ("holographic_brain", "bare_holographic_3l"),
        ("mmarrs_titan", "v06_double_loop_pc_trm_only"),
        ("titans_cognitive_core", "cog_soft_tools_k3"),
    ],
)
def test_end_to_end_probe_across_all_four_families(family: str, variant: str, tmp_path: Path):
    """End-to-end probe works identically on real models from ALL 4 independent families."""
    torch.manual_seed(1234)
    batch_tr = _make_synthetic_session_batch(seed=101, n_assets=30)
    batch_val = _make_synthetic_session_batch(seed=202, n_assets=30)

    common_dims = dict(
        prices_dim=12,
        fundamentals_dim=6,
        macro_dim=4,
        news_dim=6,
        charts_dim=6,
        physics_dim=8,
        context_dim=8,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=16,
    )
    if family == "holographic_brain":
        model = build_family_model(variant, market="US", family=family, **common_dims)
    else:
        model = build_family_model(
            variant,
            market="US",
            family=family,
            use_fused_blackwell_kernels=False,
            **common_dims,
        )
    model.eval()

    # Snapshot every weight before probing to prove ZERO weight mutation!
    orig_weights = {name: p.detach().clone() for name, p in model.named_parameters()}

    probe = DeepLearningGeometryProbe(
        hessian_cfg=HessianConfig(
            top_k=2,
            lanczos_steps=4,
            hutchinson_probes=4,
            sam_rho=0.02,
            seed=42,
        ),
        n_quantiles=5,
        ib_probe=InformationBottleneckProbe(ridge_mi=1e-2, train_frac=0.7, seed=42),
    )

    report = probe.probe_model(
        model,
        batch_tr,
        val_batch=batch_val,
        family=family,
        variant=variant,
        epoch=1,
        step=10,
    )

    assert isinstance(report, DeepLearningHealthReport)
    assert report.family == family
    assert report.variant == variant
    assert report.diagnosis in CANONICAL_DIAGNOSES

    # 1) Hessian metrics finite & weights bit-exact untouched
    assert report.hessian is not None
    assert len(report.hessian.top_eigenvalues) == 2
    assert math.isfinite(report.hessian.lambda_max)
    assert math.isfinite(report.hessian.hessian_trace)
    assert math.isfinite(report.hessian.sam_sharpness)
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), orig_weights[name]), f"Weight mutated: {name}"

    # 2) Representation health & Neural Collapse finite
    assert report.representation is not None
    assert report.representation.dim == 16
    assert report.representation.effective_rank > 0.0
    assert report.representation.rankme > 0.0
    assert report.representation.stable_rank > 0.0
    assert 0.0 <= report.representation.dead_unit_fraction <= 1.0

    assert report.neural_collapse is not None
    assert math.isfinite(report.neural_collapse.nc1_within_between_ratio)
    assert math.isfinite(report.neural_collapse.nc2_etf_simplex_deviation)
    assert report.neural_collapse.nc3_head_duality_cosine is not None
    assert -1.0001 <= report.neural_collapse.nc3_head_duality_cosine <= 1.0001

    # 3) Multi-step loop models expose Loop CKA diagnostics
    if variant in ("v06_double_loop_pc_trm_only", "cog_soft_tools_k3", "bare_holographic_3l"):
        assert report.loop_cka is not None
        assert len(report.loop_cka.adjacent_cka) >= 1

    # 4) Information Bottleneck finite
    assert report.information_bottleneck is not None
    assert report.information_bottleneck.i_xz_gauss >= 0.0
    assert math.isfinite(report.information_bottleneck.linear_probe_r2)
    assert math.isfinite(report.information_bottleneck.nonlinear_probe_r2)

    # 5) Clean RFC-1952 .jsonl.gz serialization roundtrip
    out_gz = tmp_path / f"{family}_{variant}.jsonl.gz"
    write_geometry_trace_gz(out_gz, [report])
    back = read_geometry_trace_gz(out_gz)
    assert len(back) == 1
    assert back[0]["family"] == family
    assert back[0]["diagnosis"] == report.diagnosis


def test_extract_latents_prefers_post_gate_final_hidden():
    """P1-1 regression: prefers post-monotonic `out['final_hidden']` over pre-gate hook `h_cand`."""

    class _MockMonotonicModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Module()
            self.head.norm = nn.LayerNorm(8)
            self.head.alpha_head = nn.Linear(8, 1)

        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            x = batch["prices"]
            h_cand = x * 3.0  # Pre-gate candidate seen by head hook
            _ = self.head.alpha_head(self.head.norm(h_cand))
            # Simulated monotonic gate rejects h_cand and keeps distinct final_hidden
            h_final = -x
            return {
                "pred_return": self.head.alpha_head(self.head.norm(h_final)),
                "final_hidden": h_final,
            }

    model = _MockMonotonicModel().eval()
    x = torch.randn(12, 8)
    probe = DeepLearningGeometryProbe()
    z_final, _, w_head, _ = probe.extract_latents_and_head_direction(model, {"prices": x})
    expected = model.head.norm(-x).detach()
    assert torch.allclose(z_final, expected, atol=1e-6)
    assert w_head is not None and w_head.shape == (8,)
