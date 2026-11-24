"""Pruebas TDD del simulador contrafactual compartido (`CounterfactualStressSimulator`).

Verifica de forma estricta el contrato bloqueado de `#133` Sección C:
  1. Constantes nombradas de `high_snr_context`, 5 escenarios ortogonales, 0 parámetros entrenables
     por defecto y aislamiento estricto (0 imports desde MoE / Holo3L / CogCore).
  2. Seguridad de apuesta nula: `alpha = 0` implica `p_flip == 0` exacto (jamás incentiva apuestas
     falsas bajo estrés extremo).
  3. Ortogonalidad factorial y continuidad en `S3` sin singularidad `1/||beta||`: `alpha ⊥ B_n`
     produce `alpha_sys == 0`; `alpha ⊥ R_n` produce `r_sys == 0`, continuidad suave en `beta_R=0`
     y gradientes finitos (sin `0/0 = NaN` en `||beta_R||_2`).
  4. Invarianza de escala anti-FM-7 y resistencia a inflación de escala de `|alpha|`.
  5. Disciplina de gradientes: ÚNICAMENTE `alpha` recibe gradientes finitos no nulos; `vol` y el
     resto de tensores de entrada reciben estrictamente `grad is None`.
  6. Placebo determinista sin puntos fijos (`placebo_mode=True`) para `N in (2, 7, 64, 2600)`:
     preserva la distribución marginal exacta de shocks destruyendo la alineación por activo.
  7. Paridad exacta de `factor_cache` izado, monotonía económica de `S0..S4`, higiene ante `NaN`,
     `BF16 AMP` con `N = 2600`, `torch.compile(mode="default", dynamic=True)` y auditoría AST
     `audit_zero_host_syncs == 0`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from mars_titan.models.mmarrs.blackwell_kernels import audit_zero_host_syncs
from mars_titan.models.mmarrs.counterfactual_sim import (
    CF_TRACE_DIM,
    CTX_CROSS_DISP,
    CTX_LIQ_PCT,
    CTX_MACRO_STRESS,
    CTX_RMT_ABS,
    CTX_RMT_ENT,
    CTX_RMT_NMODES_FRAC,
    CTX_VOL_PCT,
    CTX_WASS,
    N_SCENARIOS,
    SCENARIO_NAMES,
    CounterfactualStressSimulator,
    prepare_cf_factor_cache,
)


def _make_context(
    n_assets: int,
    *,
    vol_pct: float | torch.Tensor = 0.5,
    liq_pct: float | torch.Tensor = 0.5,
    wass: float | torch.Tensor = 0.5,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    dev = device or torch.device("cpu")
    ctx = torch.zeros((n_assets, 8), device=dev, dtype=dtype)
    ctx[:, CTX_RMT_ABS] = 0.3
    ctx[:, CTX_RMT_ENT] = 0.7
    ctx[:, CTX_RMT_NMODES_FRAC] = 0.05
    ctx[:, CTX_WASS] = wass if isinstance(wass, torch.Tensor) else float(wass)
    ctx[:, CTX_CROSS_DISP] = 0.4
    ctx[:, CTX_MACRO_STRESS] = 0.2
    ctx[:, CTX_VOL_PCT] = vol_pct if isinstance(vol_pct, torch.Tensor) else float(vol_pct)
    ctx[:, CTX_LIQ_PCT] = liq_pct if isinstance(liq_pct, torch.Tensor) else float(liq_pct)
    return ctx


def _make_quantiles(
    n_assets: int,
    *,
    center: float = 0.0,
    left_width: float = 0.02,
    right_width: float = 0.02,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    dev = device or torch.device("cpu")
    fracs_lo = torch.tensor([-1.5, -1.0, -0.8, -0.4], device=dev, dtype=dtype) * left_width
    fracs_hi = torch.tensor([0.4, 0.8, 1.0, 1.5], device=dev, dtype=dtype) * right_width
    q = torch.cat([fracs_lo, torch.zeros(1, device=dev, dtype=dtype), fracs_hi], dim=0) + center
    return q.unsqueeze(0).expand(n_assets, -1).clone()


def test_constants_standalone_purity_and_zero_params():
    """Constantes verificadas, 0 parámetros entrenables y 0 imports acoplados entre familias."""
    assert CTX_RMT_ABS == 0
    assert CTX_RMT_ENT == 1
    assert CTX_RMT_NMODES_FRAC == 2
    assert CTX_WASS == 3
    assert CTX_CROSS_DISP == 4
    assert CTX_MACRO_STRESS == 5
    assert CTX_VOL_PCT == 6
    assert CTX_LIQ_PCT == 7
    assert N_SCENARIOS == 5
    assert CF_TRACE_DIM == 10
    assert SCENARIO_NAMES == (
        "vol_regime_spike",
        "liquidity_freeze_impact",
        "factor_crowding_unwind",
        "rmt_eigenmode_rotation",
        "tail_conflict_shock",
    )

    sim = CounterfactualStressSimulator()
    assert sum(p.numel() for p in sim.parameters()) == 0

    src_path = (
        Path(__file__).resolve().parents[2]
        / "src/mars_titan/models/mmarrs/counterfactual_sim.py"
    )
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    forbidden = {
        "mor_loom_pc",
        "hypergraph_hrr",
        "cognitive_core",
        "neuro_brain_sota",
        "neuromod_memory",
        "model",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for bad in forbidden:
                assert bad not in node.module, f"Import prohibido en counterfactual_sim.py: {bad}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                for bad in forbidden:
                    assert bad not in alias.name

    audit = audit_zero_host_syncs(src_path)
    assert audit["clean"] is True
    assert audit["total_findings"] == 0
    assert audit["forward_findings"] == 0


def test_zero_alpha_exact_zero_p_flip_under_extreme_stress():
    """Cuando `alpha == 0`, `p_flip` debe ser exactamente 0 incluso bajo estrés máximo."""
    torch.manual_seed(7)
    n_assets = 32
    sim = CounterfactualStressSimulator()
    alpha = torch.zeros(n_assets, dtype=torch.float32, requires_grad=True)
    vol = torch.full((n_assets,), 0.025)
    ctx = _make_context(n_assets, vol_pct=1.0, liq_pct=0.0, wass=5.0)
    exposures = torch.randn(n_assets, 6)
    rmt = torch.randn(n_assets, 4)
    w_opt = torch.full((n_assets,), 0.05)
    quantiles = _make_quantiles(n_assets, left_width=0.10, right_width=0.01)
    conflict = torch.ones(n_assets)

    out = sim(
        alpha,
        vol,
        high_snr_context=ctx,
        exposures=exposures,
        rmt_loadings=rmt,
        w_opt=w_opt,
        quantiles=quantiles,
        conflict=conflict,
    )
    torch.testing.assert_close(out["p_flip"], torch.zeros(n_assets), atol=0.0, rtol=0.0)
    assert out["cf_trace"].shape == (n_assets, 10)
    assert out["scenario_drawdowns"].shape == (n_assets, 5)
    assert torch.all(out["e_cf"] >= 0.0) and torch.all(out["e_cf"] <= 3.0)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"Valor no finito en {k} con alpha=0"

    (out["e_cf"].sum() + out["p_flip"].sum() + out["cf_trace"].pow(2).sum()).backward()
    assert alpha.grad is not None and torch.isfinite(alpha.grad).all()


def test_factor_orthogonal_alpha_zero_sys_and_continuous_s3_at_beta_zero():
    """`alpha ⊥ B_n` da `alpha_sys == 0`; `alpha ⊥ R_n` da `S3` continuo en `beta_R == 0`."""
    torch.manual_seed(13)
    n_assets, f_dim, k_dim = 64, 5, 4
    sim = CounterfactualStressSimulator()

    exposures = torch.randn(n_assets, f_dim)
    rmt = torch.randn(n_assets, k_dim)

    # Construir vector `alpha_orth` estrictamente ortogonal a [1, B_n, R_n]
    b_c = exposures - exposures.mean(dim=0, keepdim=True)
    b_n = b_c / b_c.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    r_c = rmt - rmt.mean(dim=0, keepdim=True)
    r_n = r_c / r_c.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    basis = torch.cat([torch.ones(n_assets, 1), b_n, r_n], dim=-1)
    q_mat, _ = torch.linalg.qr(basis, mode="reduced")

    raw = torch.randn(n_assets) * 0.02
    alpha_orth = (raw - q_mat @ (q_mat.T @ raw)).detach().requires_grad_(True)
    vol = torch.full((n_assets,), 0.02)
    ctx = _make_context(n_assets, vol_pct=0.2, liq_pct=1.0, wass=0.0)

    out0 = sim(
        alpha_orth,
        vol,
        high_snr_context=ctx,
        exposures=exposures,
        rmt_loadings=rmt,
    )
    torch.testing.assert_close(out0["alpha_sys"], torch.zeros(n_assets), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(out0["r_sys"], torch.zeros(n_assets), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(
        out0["scenario_drawdowns"][:, 2], torch.zeros(n_assets), atol=1e-5, rtol=0.0
    )
    torch.testing.assert_close(
        out0["scenario_drawdowns"][:, 3], torch.zeros(n_assets), atol=1e-5, rtol=0.0
    )

    # Backward en `beta_R == 0` debe ser 100% finito (sin 0/0 = NaN en ||beta_R||_2)
    (out0["e_cf"].sum() + out0["cf_trace"].pow(2).sum()).backward()
    assert alpha_orth.grad is not None and torch.isfinite(alpha_orth.grad).all()

    # Continuidad en S3 cerca de beta_R = 0: perturbación 1e-4 produce cambio O(1e-4), NO O(1)!
    eps = 1e-4
    alpha_eps = (alpha_orth.detach() + eps * r_n[:, 0]).detach()
    out_eps = sim(
        alpha_eps,
        vol,
        high_snr_context=ctx,
        exposures=exposures,
        rmt_loadings=rmt,
    )
    s3_diff = (
        (out_eps["scenario_drawdowns"][:, 3] - out0["scenario_drawdowns"][:, 3].detach())
        .abs()
        .max()
    )
    assert float(s3_diff) < 0.02, f"Discontinuidad en S3 cerca de beta_R=0: {float(s3_diff)}"


def test_scale_invariance_anti_fm7_and_systematic_non_gameability():
    """Escalar `(alpha, vol, quantiles)` conserva métricas; inflar `|alpha|` no engaña S2."""
    torch.manual_seed(21)
    n_assets = 48
    sim = CounterfactualStressSimulator()

    alpha = torch.randn(n_assets) * 0.03
    vol = torch.full((n_assets,), 0.02)
    # liq_pct=1.0 -> término fijo S1 es 0 para evaluar homogeneidad pura de S0/S2/S3/S4
    ctx = _make_context(n_assets, vol_pct=0.8, liq_pct=1.0, wass=2.0)
    exposures = torch.randn(n_assets, 5)
    rmt = torch.randn(n_assets, 3)
    quantiles = _make_quantiles(n_assets, left_width=0.04, right_width=0.015)
    conflict = torch.rand(n_assets)

    base = sim(
        alpha,
        vol,
        high_snr_context=ctx,
        exposures=exposures,
        rmt_loadings=rmt,
        quantiles=quantiles,
        conflict=conflict,
    )
    for scale in (2.0, 5.0):
        scaled = sim(
            alpha * scale,
            vol * scale,
            high_snr_context=ctx,
            exposures=exposures,
            rmt_loadings=rmt,
            quantiles=quantiles * scale,
            conflict=conflict,
        )
        torch.testing.assert_close(scaled["p_flip"], base["p_flip"], atol=1.5e-2, rtol=2e-2)
        torch.testing.assert_close(scaled["e_cf"], base["e_cf"], atol=1.5e-2, rtol=2e-2)
        torch.testing.assert_close(
            scaled["scenario_drawdowns"], base["scenario_drawdowns"], atol=1.5e-2, rtol=2e-2
        )

    # Anti-FM-7: una apuesta 100% factorial (`alpha = B_n[:, 0]`) NO puede engañar S2 inflando 5x
    b_c = exposures - exposures.mean(dim=0, keepdim=True)
    b_n = b_c / b_c.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    alpha_sys_only = b_n[:, 0] * 0.02
    out_1x = sim(alpha_sys_only, vol, high_snr_context=ctx, exposures=exposures)
    out_5x = sim(alpha_sys_only * 5.0, vol, high_snr_context=ctx, exposures=exposures)
    assert float(out_5x["p_flip"].mean()) >= float(out_1x["p_flip"].mean()) - 1e-3
    assert float(out_5x["e_cf"].mean()) >= float(out_1x["e_cf"].mean()) - 1e-3


def test_gradient_isolation_only_alpha_gets_grad():
    """Solo `alpha` recibe gradiente finito no nulo; `vol` y demás entradas tienen `grad=None`."""
    torch.manual_seed(33)
    n_assets = 40
    sim = CounterfactualStressSimulator()

    alpha = (torch.randn(n_assets) * 0.02).requires_grad_(True)
    vol = (torch.full((n_assets,), 0.02)).requires_grad_(True)
    ctx = _make_context(n_assets, vol_pct=0.85, liq_pct=0.15, wass=2.5).requires_grad_(True)
    exposures = torch.randn(n_assets, 5, requires_grad=True)
    rmt = torch.randn(n_assets, 4, requires_grad=True)
    w_opt = (torch.randn(n_assets) * 0.02).requires_grad_(True)
    quantiles = _make_quantiles(n_assets, left_width=0.05, right_width=0.01).requires_grad_(True)
    conflict = torch.rand(n_assets, requires_grad=True)

    out = sim(
        alpha,
        vol,
        high_snr_context=ctx,
        exposures=exposures,
        rmt_loadings=rmt,
        w_opt=w_opt,
        quantiles=quantiles,
        conflict=conflict,
    )
    loss = (
        out["e_cf"].sum()
        + out["p_flip"].sum()
        + out["worst_pnl"].sum()
        + out["scenario_drawdowns"].sum()
        + out["cf_trace"].pow(2).sum()
        + out["alpha_sys"].pow(2).sum()
        + out["r_sys"].pow(2).sum()
    )
    loss.backward()

    assert alpha.grad is not None
    assert torch.isfinite(alpha.grad).all()
    assert float(alpha.grad.abs().sum()) > 1e-4

    for name, tensor in (
        ("vol", vol),
        ("high_snr_context", ctx),
        ("exposures", exposures),
        ("rmt_loadings", rmt),
        ("w_opt", w_opt),
        ("quantiles", quantiles),
        ("conflict", conflict),
    ):
        assert tensor.grad is None, f"Fuga de gradiente prohibida hacia `{name}`!"


@pytest.mark.parametrize("n_assets", [2, 7, 64, 2600])
def test_placebo_mode_zero_fixed_point_preserves_marginals_destroys_alignment(n_assets: int):
    """`placebo_mode=True` aplica derangement sin puntos fijos para N=2, 7, 64, 2600."""
    torch.manual_seed(100 + n_assets)
    sim = CounterfactualStressSimulator()

    # Mitad superior ilíquida y conflictiva, mitad inferior tranquila para probar alineación
    half = n_assets // 2
    alpha = torch.full((n_assets,), 0.015, dtype=torch.float32)
    alpha_grad = alpha.clone().requires_grad_(True)
    vol = torch.full((n_assets,), 0.02, dtype=torch.float32)
    liq = torch.cat([torch.zeros(half), torch.ones(n_assets - half)])
    vol_pct = torch.cat([torch.full((half,), 0.95), torch.full((n_assets - half,), 0.10)])
    ctx = _make_context(n_assets, vol_pct=vol_pct, liq_pct=liq, wass=2.0)
    conflict = torch.cat([torch.ones(half), torch.zeros(n_assets - half)])
    exposures = torch.randn(n_assets, 5) if n_assets >= 7 else torch.randn(n_assets, 1)
    rmt = torch.randn(n_assets, 4) if n_assets >= 7 else torch.randn(n_assets, 1)

    out_real = sim(
        alpha,
        vol,
        high_snr_context=ctx,
        exposures=exposures,
        rmt_loadings=rmt,
        conflict=conflict,
        placebo_mode=False,
    )
    out_pl = sim(
        alpha_grad,
        vol,
        high_snr_context=ctx,
        exposures=exposures,
        rmt_loadings=rmt,
        conflict=conflict,
        placebo_mode=True,
    )

    for out in (out_real, out_pl):
        assert out["e_cf"].shape == (n_assets,)
        assert out["p_flip"].shape == (n_assets,)
        assert out["worst_pnl"].shape == (n_assets,)
        assert out["cf_trace"].shape == (n_assets, 10)
        assert out["scenario_drawdowns"].shape == (n_assets, 5)
        assert torch.isfinite(out["cf_trace"]).all()
        assert torch.all((out["p_flip"] >= 0.0) & (out["p_flip"] <= 1.0))
        assert torch.all((out["e_cf"] >= 0.0) & (out["e_cf"] <= 3.0))

    # Con alpha constante = 0.015 y vol constante = 0.02, los drawdowns por escenario del placebo
    # son exactamente `torch.roll(..., shifts=max(1, N // 2), dims=0)` de los reales!
    shift = max(1, n_assets // 2)
    torch.testing.assert_close(
        out_pl["scenario_drawdowns"],
        torch.roll(out_real["scenario_drawdowns"], shifts=shift, dims=0),
        atol=1e-5,
        rtol=1e-5,
    )
    # Distribución marginal idéntica (ordenada bit a bit)
    torch.testing.assert_close(
        torch.sort(out_pl["e_cf"]).values,
        torch.sort(out_real["e_cf"]).values,
        atol=1e-5,
        rtol=1e-5,
    )
    # Alineación destruida (en real la mitad 0 tiene mayor estrés; en placebo se invierte)
    assert float(out_real["e_cf"][:half].detach().mean()) > float(
        out_real["e_cf"][half:].detach().mean()
    )
    assert float(out_pl["e_cf"][:half].detach().mean()) < float(
        out_real["e_cf"][:half].detach().mean()
    )

    # Gradientes en alpha siguen activos y finitos incluso en modo placebo
    out_pl["e_cf"].sum().backward()
    assert alpha_grad.grad is not None and torch.isfinite(alpha_grad.grad).all()


def test_hoisted_factor_cache_parity_and_scenario_monotonicity():
    """`factor_cache` izado iguala al cálculo directo y `S0..S4` reaccionan monotónicamente."""
    torch.manual_seed(42)
    n_assets = 48
    sim = CounterfactualStressSimulator()
    exposures = torch.randn(n_assets, 6)
    rmt = torch.randn(n_assets, 4)
    vol = torch.full((n_assets,), 0.02)
    ctx = _make_context(n_assets, vol_pct=0.8, liq_pct=0.2, wass=2.0)

    cache = prepare_cf_factor_cache(exposures=exposures, rmt_loadings=rmt)
    for step in range(3):
        alpha_k = torch.randn(n_assets) * (0.015 + 0.005 * step)
        direct = sim(alpha_k, vol, high_snr_context=ctx, exposures=exposures, rmt_loadings=rmt)
        cached = sim(
            alpha_k,
            vol,
            high_snr_context=ctx,
            exposures=exposures,
            rmt_loadings=rmt,
            factor_cache=cache,
        )
        for key in direct:
            torch.testing.assert_close(cached[key], direct[key], atol=1e-5, rtol=1e-5)

    # Monotonía económica S0, S1, S4
    a_pos = torch.full((n_assets,), 0.015)
    low_vol = sim(a_pos, vol, high_snr_context=_make_context(n_assets, vol_pct=0.1, wass=0.0))
    hi_vol = sim(a_pos, vol, high_snr_context=_make_context(n_assets, vol_pct=0.95, wass=3.0))
    assert float(hi_vol["scenario_drawdowns"][:, 0].mean()) > float(
        low_vol["scenario_drawdowns"][:, 0].mean()
    )

    liq_ok = sim(
        a_pos,
        vol,
        high_snr_context=_make_context(n_assets, liq_pct=0.95),
        w_opt=torch.zeros(n_assets),
    )
    liq_bad = sim(
        a_pos,
        vol,
        high_snr_context=_make_context(n_assets, liq_pct=0.05),
        w_opt=torch.full((n_assets,), 0.05),
    )
    assert float(liq_bad["scenario_drawdowns"][:, 1].mean()) > float(
        liq_ok["scenario_drawdowns"][:, 1].mean()
    )

    q_sym = _make_quantiles(n_assets, left_width=0.02, right_width=0.02)
    q_left_fat = _make_quantiles(n_assets, left_width=0.08, right_width=0.01)
    calm_s4 = sim(a_pos, vol, quantiles=q_sym, conflict=torch.zeros(n_assets))
    stress_s4 = sim(a_pos, vol, quantiles=q_left_fat, conflict=torch.ones(n_assets))
    assert float(stress_s4["scenario_drawdowns"][:, 4].mean()) > float(
        calm_s4["scenario_drawdowns"][:, 4].mean()
    )

    # Casos límite: N=1, todo None, NaNs en entradas opcionales y contexto corto C=4
    out_n1 = sim(torch.tensor([0.02]), torch.tensor([0.02]))
    assert out_n1["cf_trace"].shape == (1, 10) and torch.isfinite(out_n1["cf_trace"]).all()
    ctx_nan = ctx.clone()
    ctx_nan[0, :] = float("nan")
    exp_nan = exposures.clone()
    exp_nan[1, 0] = float("inf")
    out_nan = sim(a_pos, vol, high_snr_context=ctx_nan, exposures=exp_nan, rmt_loadings=rmt)
    assert torch.isfinite(out_nan["cf_trace"]).all()
    out_short_ctx = sim(a_pos, vol, high_snr_context=torch.randn(n_assets, 4))
    assert torch.isfinite(out_short_ctx["cf_trace"]).all()


def test_bf16_amp_n2600_and_torch_compile_dynamic():
    """BF16 AMP a escala N=2600 y `torch.compile(mode='default', dynamic=True)` sin errores."""
    torch.manual_seed(99)
    n_assets = 2600
    sim = CounterfactualStressSimulator()

    alpha = (torch.randn(n_assets) * 0.02).requires_grad_(True)
    vol = torch.full((n_assets,), 0.02)
    ctx = _make_context(n_assets, vol_pct=0.75, liq_pct=0.25, wass=1.8)
    exposures = torch.randn(n_assets, 9)
    rmt = torch.randn(n_assets, 5)
    quantiles = _make_quantiles(n_assets, left_width=0.04, right_width=0.02)
    conflict = torch.rand(n_assets)

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = sim(
            alpha.to(torch.bfloat16),
            vol.to(torch.bfloat16),
            high_snr_context=ctx.to(torch.bfloat16),
            exposures=exposures.to(torch.bfloat16),
            rmt_loadings=rmt.to(torch.bfloat16),
            quantiles=quantiles.to(torch.bfloat16),
            conflict=conflict.to(torch.bfloat16),
        )
    assert out["e_cf"].dtype == torch.bfloat16
    assert out["cf_trace"].shape == (n_assets, 10)
    for v in out.values():
        assert torch.isfinite(v).all()

    compiled = torch.compile(sim, mode="default", dynamic=True)
    for n_dyn in (32, 67, 128):
        a_d = torch.randn(n_dyn) * 0.02
        v_d = torch.full((n_dyn,), 0.02)
        c_d = _make_context(n_dyn, vol_pct=0.6, liq_pct=0.4, wass=1.2)
        e_d = torch.randn(n_dyn, 5)
        r_d = torch.randn(n_dyn, 4)
        out_c = compiled(a_d, v_d, high_snr_context=c_d, exposures=e_d, rmt_loadings=r_d)
        assert out_c["cf_trace"].shape == (n_dyn, 10)
        assert torch.isfinite(out_c["cf_trace"]).all()
        out_pl = compiled(
            a_d, v_d, high_snr_context=c_d, exposures=e_d, rmt_loadings=r_d, placebo_mode=True
        )
        assert out_pl["cf_trace"].shape == (n_dyn, 10)
        assert torch.isfinite(out_pl["cf_trace"]).all()
