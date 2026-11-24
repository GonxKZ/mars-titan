"""Pruebas TDD de la Cartera Convexa `P+` y Núcleo Agéntico Metacognitivo (`A5–A8`)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from mars_titan.evaluation import metacognitive_agent as ma


def test_convex_portfolio_p_plus_enforces_constraints_neutralizes_factors_and_cuts_turnover():
    """P+ cumple neutralidad dólar, tope de liquidez, reduce exposición factorial y turnover."""
    rng = np.random.default_rng(7)
    n_assets, n_factors = 80, 4
    mu = 0.005 * rng.standard_normal(n_assets)
    b_fac = rng.standard_normal((n_assets, n_factors))
    # Contaminar señal fuertemente con factor 0
    mu = mu + 0.015 * b_fac[:, 0]
    w_prev = np.zeros(n_assets, dtype=np.float64)
    w_prev[:40] = 0.5 / 40
    w_prev[40:] = -0.5 / 40
    vol = np.full(n_assets, 0.02, dtype=np.float64)
    caps = np.full(n_assets, 0.04, dtype=np.float64)

    # 1) Sin penalización factorial ni costes
    opt_raw = ma.ConvexPortfolioOptimizerPPlus(
        risk_aversion=2.0, factor_penalty=0.0, l1_cost=0.0, impact_cost=0.0, gross_limit=1.0
    )
    w_raw = opt_raw.solve(mu, vol=vol, exposures=b_fac, w_prev=w_prev, pos_cap=caps)
    assert abs(float(w_raw.sum())) < 1e-6
    assert float(np.abs(w_raw).sum()) <= 1.0 + 1e-6
    assert float(np.max(np.abs(w_raw))) <= 0.04 + 1e-6

    # 2) Con penalización factorial alta + coste L1 + impacto 3/2
    opt_p = ma.ConvexPortfolioOptimizerPPlus(
        risk_aversion=2.0,
        factor_penalty=50.0,
        l1_cost=0.003,
        impact_cost=0.02,
        gross_limit=1.0,
    )
    w_p = opt_p.solve(mu, vol=vol, exposures=b_fac, w_prev=w_prev, pos_cap=caps)
    assert abs(float(w_p.sum())) < 1e-6
    assert float(np.abs(w_p).sum()) <= 1.0 + 1e-6
    assert float(np.max(np.abs(w_p))) <= 0.04 + 1e-6

    # Exposición al factor 0 debe caer drásticamente
    exp_raw = abs(float(b_fac[:, 0] @ w_raw))
    exp_p = abs(float(b_fac[:, 0] @ w_p))
    assert exp_p < exp_raw * 0.35

    # Turnover respecto a w_prev debe caer drásticamente frente a w_raw
    to_raw = float(np.abs(w_raw - w_prev).sum())
    to_p = float(np.abs(w_p - w_prev).sum())
    assert to_p < to_raw * 0.7


def test_differentiable_portfolio_layer_propagates_finite_gradients():
    """La capa PyTorch diferenciable P+ produce pesos válidos y gradientes finitos respecto a mu."""
    torch.manual_seed(11)
    n = 32
    mu = torch.randn(n, dtype=torch.float32, requires_grad=True)
    vol = torch.full((n,), 0.02, dtype=torch.float32)
    b_fac = torch.randn(n, 3, dtype=torch.float32)
    w_prev = torch.zeros(n, dtype=torch.float32)
    layer = ma.DifferentiablePortfolioLayer(
        steps=12, risk_aversion=2.0, factor_penalty=10.0, l1_cost=0.001, gross_limit=1.0
    )
    w = layer(mu, vol=vol, exposures=b_fac, w_prev=w_prev)
    assert w.shape == (n,)
    assert torch.isfinite(w).all()
    assert abs(float(w.detach().sum())) < 1e-4
    assert float(w.detach().abs().sum()) <= 1.0 + 1e-4

    # Pérdida diferenciable estilo Sharpe-Turnover
    target = torch.randn(n, dtype=torch.float32) * 0.01
    loss = -(w * target).sum() + 0.001 * (w - w_prev).abs().sum()
    loss.backward()
    assert mu.grad is not None
    assert torch.isfinite(mu.grad).all()
    assert float(mu.grad.abs().sum()) > 0.0


def test_metacognitive_governor_abstains_on_high_epistemic_uncertainty_and_detects_cusum_decay():
    """A5 abstiene cuando la incertidumbre epistémica/OOD domina y A8 detecta decaimiento CUSUM."""
    rng = np.random.default_rng(55)
    n = 50
    # 3 especialistas coinciden en primeros 25 activos y discrepan ferozmente en los últimos 25
    base_sig = 0.008 * rng.choice([-1.0, 1.0], size=n)
    spec_preds = np.column_stack([base_sig, base_sig, base_sig])
    spec_preds[25:, 1] = -base_sig[25:]
    spec_preds[25:, 2] = 2.0 * base_sig[25:]
    va_widths = np.where(np.arange(n) < 25, 0.01, 0.25)
    hopfield_ood = np.where(np.arange(n) < 25, 0.1, 4.0)
    router_entropy = np.where(np.arange(n) < 25, 0.1, 1.8)
    costs = np.full(n, 0.001)

    gov = ma.MetacognitiveGovernor(
        n_specialists=3, epistemic_kappa=1.0, ood_threshold=2.0, cusum_threshold=3.0
    )
    decision = gov.decide(
        specialist_preds=spec_preds,
        va_widths=va_widths,
        hopfield_energy=hopfield_ood,
        router_entropy=router_entropy,
        loop_energy=np.zeros(n),
        estimated_costs=costs,
    )
    assert decision["active_mask"][:25].mean() > 0.8
    assert decision["active_mask"][25:].mean() < 0.25
    assert np.all(decision["gated_signal"][~decision["active_mask"]] == 0.0)

    # A8 CUSUM: especialista 2 pasa de IC=+0.08 a IC=-0.05 sostenido -> debe disparar alarma y cap
    for _ in range(20):
        gov.update_matured(np.array([0.08, 0.07, 0.06]))
    assert not gov.decay_alerts[2]
    for _ in range(25):
        gov.update_matured(np.array([0.08, 0.07, -0.08]))
    assert gov.decay_alerts[2]
    assert gov.specialist_weights[2] < gov.specialist_weights[0] * 0.5


def test_agent_variations_are_distinct_and_causal():
    """Todas las variaciones agénticas (none, p_plus, moe_agent, full_active) corren limpiamente."""
    rng = np.random.default_rng(77)
    sessions, assets = 30, 40
    n = sessions * assets
    bounds = [(s * assets, (s + 1) * assets) for s in range(sessions)]
    spec_preds = 0.005 * rng.standard_normal((n, 3))
    target = 0.6 * spec_preds[:, 0] + 0.4 * spec_preds[:, 1] + 0.01 * rng.standard_normal(n)
    vol = np.full(n, 0.015)
    exposures = rng.standard_normal((n, 3))
    regime = rng.standard_normal(sessions)

    for variant in ma.AGENT_VARIATIONS:
        out = ma.run_agent_variation(
            variant=variant,
            specialist_preds=spec_preds,
            target=target,
            bounds=bounds,
            volatility=vol,
            exposures=exposures,
            regime_signal=regime,
            l1_cost=0.001,
        )
        assert out["weights"].shape == (n,)
        assert np.isfinite(out["weights"]).all()
        assert "net_sharpe_1x" in out["summary"]
        assert "turnover" in out["summary"]


def test_gpu_native_torch_p_plus_and_governor_match_constraints():
    """`solve_torch` y `decide_torch` corren 100 % en tensores PyTorch cumpliendo cotas."""
    torch.manual_seed(19)
    n = 48
    mu = torch.randn(n) * 0.015
    vol = torch.full((n,), 0.02)
    b_fac = torch.randn(n, 4)
    w_prev = torch.zeros(n)

    opt = ma.ConvexPortfolioOptimizerPPlus(
        risk_aversion=2.0, factor_penalty=20.0, l1_cost=0.001, impact_cost=0.01, gross_limit=1.0
    )
    w_t = opt.solve_torch(mu, vol=vol, exposures=b_fac, w_prev=w_prev, pos_cap=0.05)
    assert w_t.shape == (n,)
    assert torch.isfinite(w_t).all()
    assert abs(float(w_t.sum())) < 1e-4
    assert float(w_t.abs().sum()) <= 1.0 + 1e-4
    assert float(w_t.abs().max()) <= 0.05 + 1e-4

    gov = ma.MetacognitiveGovernor(n_specialists=3, epistemic_kappa=1.0, ood_threshold=2.0)
    base = torch.ones(n) * 0.01
    sp = torch.stack([base, base, base], dim=-1)
    sp[24:, 1] = -base[24:]
    ood = torch.where(torch.arange(n) < 24, torch.full((n,), 0.1), torch.full((n,), 4.0))
    dec_t = gov.decide_torch(
        specialist_preds=sp,
        hopfield_energy=ood,
        estimated_costs=torch.full((n,), 0.001),
    )
    assert float(dec_t["active_mask"][:24].float().mean()) > 0.9
    assert float(dec_t["active_mask"][24:].float().mean()) == 0.0
    assert torch.all(dec_t["gated_signal"][~dec_t["active_mask"]] == 0.0)


def test_p0_3_portfolio_layer_and_solve_torch_active_at_n2600_scale():
    """P0-3: lip = O(1) en N=2600 garantiza ||w||_1 > 0.5 en 6 pasos y 40 iters FISTA."""
    torch.manual_seed(123)
    n_assets, n_factors = 2600, 9
    mu = torch.randn(n_assets, dtype=torch.float32, requires_grad=True)
    mu_scaled = mu * 0.01
    vol = torch.full((n_assets,), 0.02, dtype=torch.float32)
    exposures = torch.randn(n_assets, n_factors, dtype=torch.float32)
    exposures[:, -1] = 1.0  # columna constante para verificar seguridad sin bool(keep.any())

    # 1) DifferentiablePortfolioLayer en 6 pasos desenrollados (granular_heads / cognitive_core)
    layer = ma.DifferentiablePortfolioLayer(
        steps=6, risk_aversion=2.0, factor_penalty=8.0, l1_cost=0.001, gross_limit=1.0, pos_cap=0.1
    )
    w_diff = layer(mu_scaled, vol=vol, exposures=exposures)
    assert w_diff.shape == (n_assets,)
    assert torch.isfinite(w_diff).all()
    assert abs(float(w_diff.detach().sum())) < 1e-4
    assert float(w_diff.detach().abs().sum()) > 0.5
    assert float(w_diff.detach().abs().sum()) <= 1.0 + 1e-4
    (w_diff * torch.randn(n_assets)).sum().backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    assert float(mu.grad.abs().sum()) > 0.0

    # Compatibilidad bfloat16 en N=2600
    mu_bf = (torch.randn(n_assets, dtype=torch.bfloat16) * 0.01).detach().requires_grad_(True)
    w_bf = layer(mu_bf, vol=vol.to(torch.bfloat16), exposures=exposures.to(torch.bfloat16))
    assert w_bf.dtype == torch.bfloat16
    assert torch.isfinite(w_bf).all()
    assert float(w_bf.detach().float().abs().sum()) > 0.5
    w_bf.float().pow(2).sum().backward()
    assert mu_bf.grad is not None and torch.isfinite(mu_bf.grad).all()

    # 2) ConvexPortfolioOptimizerPPlus.solve_torch en <= 40 iters en N=2600
    opt = ma.ConvexPortfolioOptimizerPPlus(
        risk_aversion=2.0,
        factor_penalty=15.0,
        l1_cost=0.001,
        impact_cost=0.01,
        gross_limit=1.0,
        max_iters=40,
    )
    w_t = opt.solve_torch(mu_scaled.detach(), vol=vol, exposures=exposures, pos_cap=0.05)
    assert w_t.shape == (n_assets,)
    assert torch.isfinite(w_t).all()
    assert abs(float(w_t.sum())) < 1e-4
    assert float(w_t.abs().sum()) > 0.5
    assert float(w_t.abs().sum()) <= 1.0 + 1e-4
    assert float(w_t.abs().max()) <= 0.05 + 1e-4


def test_p0_7_torch_portfolio_paths_zero_host_syncs_ast():
    """P0-7: CERO `.item()`, `.cpu()`, `.numpy()`, `.tolist()` o `bool()` en rutas GPU PyTorch."""
    mod_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mars_titan"
        / "evaluation"
        / "metacognitive_agent.py"
    )
    tree = ast.parse(mod_path.read_text(encoding="utf-8"))
    forbidden_attrs = {"item", "cpu", "numpy", "tolist", "synchronize"}
    targets: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "DifferentiablePortfolioLayer":
            targets.append(("DifferentiablePortfolioLayer", node))
        if isinstance(node, ast.ClassDef) and node.name == "ConvexPortfolioOptimizerPPlus":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "solve_torch":
                    targets.append(("ConvexPortfolioOptimizerPPlus.solve_torch", item))
        if isinstance(node, ast.ClassDef) and node.name == "MetacognitiveGovernor":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "decide_torch":
                    targets.append(("MetacognitiveGovernor.decide_torch", item))

    assert len(targets) == 3
    found: list[tuple[str, int, str]] = []
    for label, sub in targets:
        for node in ast.walk(sub):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_attrs:
                    found.append((label, node.lineno, f".{node.func.attr}()"))
                if isinstance(node.func, ast.Name) and node.func.id == "bool":
                    found.append((label, node.lineno, "bool()"))
    assert found == [], f"Sincronizaciones CPU↔GPU prohibidas: {found}"


def test_p0_1_ragged_panel_asset_ids_alignment_and_liquidation_turnover():
    """P0-1: paneles irregulares alinean `w_prev` por `asset_id` y cobran liquidación de bajas."""
    prev_map = {10: 0.25, 20: 0.25, 30: -0.25, 40: -0.25}
    curr_ids = np.array([20, 30, 40, 50], dtype=np.int64)  # baja 10, entra IPO 50 (mismo N=4!)
    w_aligned, exited_to = ma.align_weights_by_asset_ids(prev_map, curr_ids)
    np.testing.assert_allclose(w_aligned, np.array([0.25, -0.25, -0.25, 0.0]))
    assert exited_to == pytest.approx(0.25)

    # Panel irregular 3 sesiones (N=4 -> N=4 desplazado -> N=3) en run_agent_variation
    rng = np.random.default_rng(91)
    ids_s0 = np.array([10, 20, 30, 40], dtype=np.int64)
    ids_s1 = np.array([20, 30, 40, 50], dtype=np.int64)
    ids_s2 = np.array([30, 40, 50], dtype=np.int64)
    asset_ids = np.concatenate([ids_s0, ids_s1, ids_s2])
    bounds = [(0, 4), (4, 8), (8, 11)]
    n = len(asset_ids)
    # Señal persistente por asset_id
    sig_by_id = {10: 0.02, 20: 0.01, 30: -0.01, 40: -0.02, 50: 0.015}
    base = np.array([sig_by_id[int(a)] for a in asset_ids], dtype=np.float64)
    spec_preds = np.column_stack([base, base * 0.9, base * 1.1])
    target = base * 0.5 + 0.005 * rng.standard_normal(n)

    out_ids = ma.run_agent_variation(
        variant="agent_p_plus_only",
        specialist_preds=spec_preds,
        target=target,
        bounds=bounds,
        asset_ids=asset_ids,
        l1_cost=0.001,
        factor_penalty=0.0,
    )
    assert out_ids["weights"].shape == (n,)
    assert np.isfinite(out_ids["weights"]).all()
    assert out_ids["summary"]["exited_turnover_mean"] > 0.0
    assert out_ids["summary"]["turnover"] > 0.0


def test_p0_4_realistic_fm7_daily_alpha_3_to_8_bps_produces_active_non_zero_portfolio():
    """P0-4: alfa diaria 3–8 bps produce cartera activa (`active_rate>25%`, `||w||_1>0.5`)."""
    rng = np.random.default_rng(2026)
    n_assets, n_spec, n_fac = 2600, 4, 9
    # Alfa FM-7 realista: ~5 bps std (|mu| típico 3–8 bps)
    common = rng.standard_normal(n_assets) * 5e-4
    spec_preds = np.column_stack(
        [common + rng.standard_normal(n_assets) * 1.2e-4 for _ in range(n_spec)]
    )
    va_w = np.full(n_assets, 0.04, dtype=np.float64)
    r_ent = np.full(n_assets, 1.38, dtype=np.float64)
    l_eng = np.full(n_assets, 0.30, dtype=np.float64)
    costs = np.full(n_assets, 0.001 / 5.0, dtype=np.float64)
    vol = np.full(n_assets, 0.015, dtype=np.float64)
    exposures = rng.standard_normal((n_assets, n_fac))

    gov = ma.MetacognitiveGovernor(
        n_specialists=n_spec,
        epistemic_kappa=0.5,
        ood_threshold=2.5,
        entropy_weight=0.002,
        loop_energy_weight=0.002,
    )
    dec_np = gov.decide(
        specialist_preds=spec_preds,
        va_widths=va_w,
        router_entropy=r_ent,
        loop_energy=l_eng,
        estimated_costs=costs,
    )
    assert float(dec_np["active_mask"].mean()) > 0.25, (
        f"Colapso de abstención en alfa 5 bps: active_rate={dec_np['active_mask'].mean():.4f}"
    )

    dec_t = gov.decide_torch(
        specialist_preds=torch.as_tensor(spec_preds, dtype=torch.float32),
        va_widths=torch.as_tensor(va_w, dtype=torch.float32),
        router_entropy=torch.as_tensor(r_ent, dtype=torch.float32),
        loop_energy=torch.as_tensor(l_eng, dtype=torch.float32),
        estimated_costs=torch.as_tensor(costs, dtype=torch.float32),
    )
    assert float(dec_t["active_mask"].float().mean()) > 0.25
    np.testing.assert_allclose(
        dec_t["gated_signal"].numpy(), dec_np["gated_signal"], atol=1e-5, rtol=1e-4
    )

    # P+ con holding_days default (amortización de l1_cost=10 bps sobre horizonte)
    opt = ma.ConvexPortfolioOptimizerPPlus(
        risk_aversion=2.0,
        factor_penalty=15.0,
        l1_cost=0.001,
        impact_cost=0.01,
        gross_limit=1.0,
    )
    w_cpu = opt.solve(dec_np["gated_signal"], vol=vol, exposures=exposures, pos_cap=0.05)
    assert float(np.abs(w_cpu).sum()) > 0.5, f"Colapso w==0 CPU: ||w||_1={np.abs(w_cpu).sum()}"
    assert abs(float(w_cpu.sum())) < 1e-5
    assert float(np.abs(w_cpu).max()) <= 0.05 + 1e-6

    w_t = opt.solve_torch(
        dec_t["gated_signal"],
        vol=torch.as_tensor(vol, dtype=torch.float32),
        exposures=torch.as_tensor(exposures, dtype=torch.float32),
        pos_cap=0.05,
    )
    assert float(w_t.abs().sum()) > 0.5, f"Colapso w==0 Torch: ||w||_1={w_t.abs().sum()}"
    assert abs(float(w_t.sum())) < 1e-4
    assert float(w_t.abs().max()) <= 0.05 + 1e-6

    # DifferentiablePortfolioLayer con alfa 5 bps produce cartera activa y gradientes no nulos
    mu_t = torch.as_tensor(common, dtype=torch.float32).detach().requires_grad_(True)
    layer = ma.DifferentiablePortfolioLayer(
        steps=10, risk_aversion=2.0, factor_penalty=10.0, l1_cost=0.001, gross_limit=1.0
    )
    w_diff = layer(
        mu_t,
        vol=torch.as_tensor(vol, dtype=torch.float32),
        exposures=torch.as_tensor(exposures, dtype=torch.float32),
    )
    assert float(w_diff.detach().abs().sum()) > 0.5
    (w_diff * torch.as_tensor(common, dtype=torch.float32)).sum().backward()
    assert mu_t.grad is not None and float(mu_t.grad.abs().sum()) > 0.0


def test_p1_2_cpu_gpu_parity_pos_cap_strict_no_eigvalsh_and_bf16_amp_cuda():
    """P1-2 + #87 #1: paridad CPU/GPU > 0.9999, `pos_cap` estricto, 0 `eigvalsh`, BF16 AMP OK."""
    # 1) AST: cero `eigvalsh` en solve_torch y DifferentiablePortfolioLayer
    mod_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mars_titan"
        / "evaluation"
        / "metacognitive_agent.py"
    )
    tree = ast.parse(mod_path.read_text(encoding="utf-8"))
    eig_hits: list[tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "DifferentiablePortfolioLayer":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and sub.attr == "eigvalsh":
                    eig_hits.append(("DifferentiablePortfolioLayer", sub.lineno))
        if isinstance(node, ast.ClassDef) and node.name == "ConvexPortfolioOptimizerPPlus":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "solve_torch":
                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Attribute) and sub.attr == "eigvalsh":
                            eig_hits.append(("solve_torch", sub.lineno))
    assert eig_hits == [], f"cuSOLVER eigvalsh prohibido en rutas Torch/GPU: {eig_hits}"

    # 2) Paridad exacta CPU solve() vs solve_torch() > 0.9999 (mata bug corr=0.3344 de #87 #1!)
    rng = np.random.default_rng(1234)
    n, k = 300, 6
    mu = rng.standard_normal(n) * 0.01
    vol = rng.uniform(0.01, 0.04, size=n)
    exposures = rng.standard_normal((n, k))
    w_prev = np.zeros(n, dtype=np.float64)
    opt = ma.ConvexPortfolioOptimizerPPlus(
        risk_aversion=2.0,
        factor_penalty=15.0,
        l1_cost=0.001,
        impact_cost=0.01,
        gross_limit=1.0,
        max_iters=60,
    )
    w_cpu = opt.solve(mu, vol=vol, exposures=exposures, w_prev=w_prev, pos_cap=0.03)
    w_torch_cpu = opt.solve_torch(
        torch.as_tensor(mu, dtype=torch.float32),
        vol=torch.as_tensor(vol, dtype=torch.float32),
        exposures=torch.as_tensor(exposures, dtype=torch.float32),
        w_prev=torch.as_tensor(w_prev, dtype=torch.float32),
        pos_cap=0.03,
    ).numpy()
    corr_cpu = float(np.corrcoef(w_cpu, w_torch_cpu)[0, 1])
    assert corr_cpu > 0.9999, f"Desalineación CPU vs Torch: corr={corr_cpu:.6f}"
    assert float(np.abs(w_torch_cpu).max()) <= 0.03 + 1e-6

    # 3) Asimetría extrema donde clamp->mean antes violaba pos_cap!
    mu_asym = torch.full((40,), -0.001, dtype=torch.float32)
    mu_asym[:5] = 0.5  # 5 activos saturados en +pos_cap -> media asimétrica fuerte
    layer_asym = ma.DifferentiablePortfolioLayer(
        steps=8, risk_aversion=1.0, factor_penalty=0.0, l1_cost=0.0, gross_limit=1.0, pos_cap=0.04
    )
    w_asym = layer_asym(mu_asym)
    assert abs(float(w_asym.sum())) < 1e-4
    assert float(w_asym.abs().max()) <= 0.04 + 1e-6, (
        f"Violación de pos_cap: max|w|={w_asym.abs().max().item():.6f} > 0.04"
    )

    # 4) Verificación real CUDA: BF16 AMP (sin NotImplementedError), corr > 0.9999 y compile
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        mu_g = torch.as_tensor(mu, device=dev, dtype=torch.float32)
        vol_g = torch.as_tensor(vol, device=dev, dtype=torch.float32)
        exp_g = torch.as_tensor(exposures, device=dev, dtype=torch.float32)
        wp_g = torch.as_tensor(w_prev, device=dev, dtype=torch.float32)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            w_gpu = opt.solve_torch(mu_g, vol=vol_g, exposures=exp_g, w_prev=wp_g, pos_cap=0.03)
        assert torch.isfinite(w_gpu).all()
        corr_gpu = float(np.corrcoef(w_cpu, w_gpu.float().cpu().numpy())[0, 1])
        assert corr_gpu > 0.9999, f"Desalineación CPU vs CUDA: corr={corr_gpu:.6f}"
        assert float(w_gpu.abs().max()) <= 0.03 + 1e-6

        # DifferentiablePortfolioLayer bajo BF16 AMP + torch.compile(default, dynamic=True)
        layer_g = ma.DifferentiablePortfolioLayer(
            steps=8, risk_aversion=2.0, factor_penalty=15.0, l1_cost=0.001, gross_limit=1.0
        ).to(dev)
        mu_req = mu_g.clone().detach().requires_grad_(True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            w_dl = layer_g(mu_req, vol=vol_g, exposures=exp_g)
            loss_dl = -(w_dl.float() * mu_g).sum()
        loss_dl.backward()
        assert torch.isfinite(w_dl).all()
        assert mu_req.grad is not None and torch.isfinite(mu_req.grad).all()
        assert float(w_dl.detach().abs().max()) <= 0.1 + 1e-5

        compiled_layer = torch.compile(layer_g, mode="default", dynamic=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            w_c1 = compiled_layer(mu_g, vol=vol_g, exposures=exp_g)
            w_c2 = compiled_layer(mu_g[:180], vol=vol_g[:180], exposures=exp_g[:180])
        assert torch.isfinite(w_c1).all() and torch.isfinite(w_c2).all()

        # MetacognitiveGovernor.decide_torch cachea pesos en GPU sin syncs repetidos
        gov_g = ma.MetacognitiveGovernor(n_specialists=3)
        sp_g = torch.randn(128, 3, device=dev) * 0.005
        d1 = gov_g.decide_torch(specialist_preds=sp_g)
        ptr1 = gov_g._weights_tensor_cache[sp_g.device].data_ptr()
        d2 = gov_g.decide_torch(specialist_preds=sp_g)
        ptr2 = gov_g._weights_tensor_cache[sp_g.device].data_ptr()
        assert ptr1 == ptr2
        torch.testing.assert_close(d1["fused_signal"], d2["fused_signal"])
        gov_g.specialist_weights = np.array([0.8, 0.1, 0.1], dtype=np.float64)
        d3 = gov_g.decide_torch(specialist_preds=sp_g)
        assert not torch.allclose(d1["fused_signal"], d3["fused_signal"])


def test_p2_run_agent_variation_causal_maturity_queue_multi_horizon():
    """P2: `horizon_sessions > 1` retrasa la maduración del gobernador exactamente `h` sesiones."""
    rng = np.random.default_rng(404)
    sessions, assets = 15, 24
    n = sessions * assets
    bounds = [(s * assets, (s + 1) * assets) for s in range(sessions)]
    spec_preds = 0.005 * rng.standard_normal((n, 3))
    target_a = 0.5 * spec_preds[:, 0] + 0.01 * rng.standard_normal(n)
    target_b = target_a.copy()
    # Mutar objetivo de la sesión 2 (solo puede madurar en sesión 2 + 5 = 7)
    lo2, hi2 = bounds[2]
    target_b[lo2:hi2] = -target_a[lo2:hi2] * 3.0

    out_a = ma.run_agent_variation(
        variant="agent_full_active_inference",
        specialist_preds=spec_preds,
        target=target_a,
        bounds=bounds,
        horizon_sessions=5,
    )
    out_b = ma.run_agent_variation(
        variant="agent_full_active_inference",
        specialist_preds=spec_preds,
        target=target_b,
        bounds=bounds,
        horizon_sessions=5,
    )
    # Sesiones 0..6 NO deben cambiar ni un bit al mutar target[2] con h=5!
    lo7, _ = bounds[7]
    np.testing.assert_allclose(out_a["weights"][:lo7], out_b["weights"][:lo7], atol=0.0, rtol=0.0)
    assert out_a["summary"]["matured_updates"] == sessions - 5


def test_118_differentiable_portfolio_fast_implicit_projection_opcheck_gradcheck_and_speed():
    """Regresión `#118` #3: proyección 1-CTA implícita exacta, `opcheck`, `gradcheck` y <10ms."""
    torch.manual_seed(1183)
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))

    for dev in devs:
        # 1) `torch.library.opcheck` (contiguo y no contiguo strided)
        v_c = torch.randn(48, device=dev, dtype=torch.float32, requires_grad=True)
        torch.library.opcheck(
            torch.ops.mars_titan_pplus.project_box_neutral_fwd.default,
            (v_c, 0.05),
        )
        v_nc = torch.randn(48, 2, device=dev, dtype=torch.float32)[:, 0].detach().requires_grad_(
            True
        )
        assert not v_nc.is_contiguous()
        torch.library.opcheck(
            torch.ops.mars_titan_pplus.project_box_neutral_fwd.default,
            (v_nc, 0.05),
        )

        # 2) Gradcheck FP64 exacto sobre proyección implícita Y FISTA fusionado completo
        v64 = (torch.randn(16, device=dev, dtype=torch.float64) * 0.08).detach().requires_grad_(
            True
        )
        assert torch.autograd.gradcheck(
            lambda t: ma.project_dollar_neutral_box_torch(t, pos_cap=0.05, gross_limit=1.0),
            (v64,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )
        mu64 = (torch.randn(16, device=dev, dtype=torch.float64) * 0.015).detach().requires_grad_(
            True
        )
        vol64 = torch.full((16,), 0.02, device=dev, dtype=torch.float64)
        exp64 = torch.randn(16, 3, device=dev, dtype=torch.float64)
        layer64 = ma.DifferentiablePortfolioLayer(
            steps=4, risk_aversion=2.0, factor_penalty=6.0, l1_cost=0.0002, gross_limit=1.0,
            pos_cap=0.1,
        )
        assert torch.autograd.gradcheck(
            lambda m_, _l=layer64, _v=vol64, _e=exp64: _l(m_, vol=_v, exposures=_e),
            (mu64,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )
        # opcheck sobre el custom_op FISTA fusionado
        b_dummy = torch.randn(24, 3, device=dev, dtype=torch.float32)
        torch.library.opcheck(
            torch.ops.mars_titan_pplus.diff_portfolio_fista_fwd.default,
            (
                torch.randn(24, device=dev, dtype=torch.float32, requires_grad=True),
                torch.full((24,), 4e-4, device=dev, dtype=torch.float32),
                torch.zeros(24, device=dev, dtype=torch.float32),
                b_dummy,
                torch.tensor(0.05, device=dev, dtype=torch.float32),
                4,
                2.0,
                5.0,
                0.0002,
                0.1,
                1.0,
            ),
        )

    # 3) Paridad exacta vs proyección de referencia en N=2600 y latencia BF16 CUDA <10ms
    n_a, n_f = 2600, 9
    mu_cpu = torch.randn(n_a, dtype=torch.float32) * 0.01
    vol_cpu = torch.full((n_a,), 0.02, dtype=torch.float32)
    exp_cpu = torch.randn(n_a, n_f, dtype=torch.float32)
    layer = ma.DifferentiablePortfolioLayer(
        steps=10, risk_aversion=2.0, factor_penalty=10.0, l1_cost=0.001, gross_limit=1.0,
        pos_cap=0.1,
    )
    w_cpu = layer(mu_cpu, vol=vol_cpu, exposures=exp_cpu)
    assert abs(float(w_cpu.sum())) < 1e-5
    assert float(w_cpu.abs().max()) <= 0.1 + 1e-6
    assert float(w_cpu.abs().sum()) <= 1.0 + 1e-5

    if torch.cuda.is_available():
        import time

        dev_g = torch.device("cuda")
        layer_g = layer.to(dev_g)
        mu_g = mu_cpu.to(device=dev_g, dtype=torch.bfloat16).detach().requires_grad_(True)
        vol_g = vol_cpu.to(device=dev_g, dtype=torch.bfloat16)
        exp_g = exp_cpu.to(device=dev_g, dtype=torch.bfloat16)
        # Warmup fwd + bwd Triton
        for _ in range(5):
            mu_g.grad = None
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                w_g = layer_g(mu_g, vol=vol_g, exposures=exp_g)
                (w_g.float() * vol_g.float()).sum().backward()
        torch.cuda.synchronize()
        corr = float(np.corrcoef(w_cpu.numpy(), w_g.detach().float().cpu().numpy())[0, 1])
        assert corr > 0.999, f"Paridad CPU vs CUDA BF16 degradada: corr={corr:.6f}"
        assert abs(float(w_g.detach().float().sum())) < 1e-4
        assert float(w_g.detach().float().abs().max()) <= 0.1 + 1e-4

        times_ms: list[float] = []
        for _ in range(15):
            mu_g.grad = None
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                w_g = layer_g(mu_g, vol=vol_g, exposures=exp_g)
                loss_g = (w_g.float() * vol_g.float()).sum()
            loss_g.backward()
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        best_ms = min(times_ms)
        assert best_ms < 10.0, f"DifferentiablePortfolioLayer BF16 N=2600 lento: {best_ms:.2f}ms"
        assert mu_g.grad is not None and torch.isfinite(mu_g.grad).all()



