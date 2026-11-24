"""Pruebas TDD de `ContextEngineerGPU`, `DifferentiableQuantToolbox` y `TitansCognitiveCore`.

Verifica de forma estricta:
  1. `ContextEngineerGPU` (Anthropic Effective Context Engineering): poda exacta de modalidades
     ausentes (`salience == 0`) y obsoletas (`modality_age >> 0`), aislamiento causal de notas
     externas comprimidas `[M_notes, D]` y gradientes finitos en todos los parámetros.
  2. `DifferentiableQuantToolbox`: las 6 herramientas deterministas nativas en GPU/PyTorch
     (`rmt_factor_decompose`, `hrr_episodic_lookup`, `hypergraph_spillover`,
     `structural_verifier`, `portfolio_feasibility`, `epistemic_risk_check`), paridad exacta
     con `HRRModernHopfieldMemory`, `RMTSpilloverHypergraph` y `DifferentiablePortfolioLayer`,
     y gradientes finitos en todos los parámetros entrenables.
  3. `TitansCognitiveCore` (Karpathy Compact Weight-Tied Cognitive Core + Tool-Use Loop):
     default `thinking_steps=3`, límite duro `1 <= thinking_steps <= 4`, compuerta aprendida de
     parada adaptativa (`halt_gate`) + consistencia monótona, compartición estricta de pesos
     (`thinking_steps=4` añade ~0 parámetros vs `thinking_steps=1`), estabilidad contractiva
     normalizada espectralmente (`Lipschitz < 1`), los 4 modos (`soft`, `sparsemax`, `uniform`,
     `no_tools`), determinismo FP32 bit a bit, gradientes finitos y equivarianza por permutación.
  4. Cero sincronizaciones CPU↔GPU (`.item()`, `.cpu()`, `.numpy()`, `.tolist()`) verificadas
     tanto por auditoría AST estática como en ejecución.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch

from mars_titan.models.mmarrs.cognitive_core import (
    DEFAULT_PONDER_WEIGHT,
    DEFAULT_THINKING_STEPS,
    MAX_THINKING_STEPS,
    TOOL_NAMES,
    TOOL_ROUTER_MODES,
    AdaptiveHaltGate,
    ContextEngineerGPU,
    DifferentiableQuantToolbox,
    TitansCognitiveCore,
    ToolRouter,
)
from mars_titan.models.mmarrs.granular_heads import MultiHorizonGranularHead
from mars_titan.models.mmarrs.hypergraph_hrr import (
    HRRModernHopfieldMemory,
    RMTSpilloverHypergraph,
)


def test_context_engineer_masks_prunes_stale_preserves_causality_and_gradients():
    """ContextEngineerGPU poda ausentes/obsoletas, preserva causalidad y emite gradientes."""
    torch.manual_seed(7)
    n_assets, n_mod, d_model, c_dim, n_notes = 18, 5, 24, 10, 4
    ce = ContextEngineerGPU(
        hidden_dim=d_model,
        context_dim=c_dim,
        n_modalities=n_mod,
        n_notes=n_notes,
        stale_halflife=4.0,
    )

    tokens = torch.randn(n_assets, n_mod, d_model, requires_grad=True)
    mask = torch.ones(n_assets, n_mod)
    mask[:, 3] = 0.0  # noticias ausentes en toda la sesión
    mask[0::2, 1] = 0.0  # fundamentales ausentes en la mitad de los activos
    phys = torch.randn(n_assets, d_model, requires_grad=True)
    ctx = torch.randn(n_assets, c_dim, requires_grad=True)
    age_fresh = torch.zeros(n_assets, n_mod)

    compact_x, salience, pruned_ratio, notes = ce(
        tokens, mask, phys, ctx, modality_age=age_fresh
    )
    assert compact_x.shape == (n_assets, d_model)
    assert salience.shape == (n_assets, n_mod)
    assert pruned_ratio.ndim == 0
    assert notes.shape == (n_notes, d_model)
    assert torch.isfinite(compact_x).all()
    assert torch.isfinite(salience).all()
    assert torch.isfinite(notes).all()
    assert 0.0 <= float(pruned_ratio) <= 1.0

    # 1) Modalidades enmascaradas reciben saliencia exactamente 0 y aislamiento bit a bit
    torch.testing.assert_close(
        salience[:, 3], torch.zeros(n_assets), atol=1e-7, rtol=0.0
    )
    torch.testing.assert_close(
        salience[0::2, 1], torch.zeros(n_assets // 2), atol=1e-7, rtol=0.0
    )
    with torch.no_grad():
        tokens_poisoned = tokens.detach().clone()
        tokens_poisoned[:, 3, :] = float("nan")
        tokens_poisoned[0::2, 1, :] = float("inf")
        age_poisoned = age_fresh.clone()
        age_poisoned[:, 3] = float("nan")
        cx_clean, sal_clean, pr_clean, notes_clean = ce(
            tokens.detach(), mask, phys.detach(), ctx.detach(), modality_age=age_fresh
        )
        cx_pois, sal_pois, pr_pois, notes_pois = ce(
            tokens_poisoned, mask, phys.detach(), ctx.detach(), modality_age=age_poisoned
        )
        torch.testing.assert_close(cx_clean, cx_pois, atol=0.0, rtol=0.0)
        torch.testing.assert_close(sal_clean, sal_pois, atol=0.0, rtol=0.0)
        torch.testing.assert_close(pr_clean, pr_pois, atol=0.0, rtol=0.0)
        torch.testing.assert_close(notes_clean, notes_pois, atol=0.0, rtol=0.0)

    # 2) Modalidad obsoleta (modality_age >> halflife) es atenuada/podada drásticamente
    with torch.no_grad():
        age_stale = torch.zeros(n_assets, n_mod)
        age_stale[:, 4] = 60.0  # gráficos con 60 sesiones de retraso
        _, sal_stale, pr_stale, _ = ce(
            tokens.detach(), mask, phys.detach(), ctx.detach(), modality_age=age_stale
        )
        assert float(sal_stale[:, 4].mean()) < 1e-2
        assert float(sal_stale[:, 4].mean()) < float(sal_clean[:, 4].mean()) * 0.1 + 1e-4
        assert float(pr_stale) >= float(pr_clean)

    # 3) Causalidad temporal estricta con notas externas comprimidas (t=0 -> t=1 -> t=2)
    with torch.no_grad():
        t0_tok = torch.randn(n_assets, n_mod, d_model)
        t1_tok = torch.randn(n_assets, n_mod, d_model)
        t2_tok_a = torch.randn(n_assets, n_mod, d_model)
        t2_tok_b = torch.randn(n_assets, n_mod, d_model) * 5.0

        cx0_a, _, _, n0_a = ce(t0_tok, mask, phys.detach(), ctx.detach())
        cx1_a, _, _, n1_a = ce(t1_tok, mask, phys.detach(), ctx.detach(), prev_notes=n0_a)
        _ = ce(t2_tok_a, mask, phys.detach(), ctx.detach(), prev_notes=n1_a)

        cx0_b, _, _, n0_b = ce(t0_tok, mask, phys.detach(), ctx.detach())
        cx1_b, _, _, n1_b = ce(t1_tok, mask, phys.detach(), ctx.detach(), prev_notes=n0_b)
        _ = ce(t2_tok_b, mask, phys.detach(), ctx.detach(), prev_notes=n1_b)

        # Modificar sesión futura t=2 jamás altera t=0 ni t=1
        torch.testing.assert_close(cx0_a, cx0_b, atol=0.0, rtol=0.0)
        torch.testing.assert_close(cx1_a, cx1_b, atol=0.0, rtol=0.0)
        # Las notas previas n0 sí enriquecen t=1 respecto a estado sin notas previas
        cx1_no_prev, _, _, _ = ce(t1_tok, mask, phys.detach(), ctx.detach(), prev_notes=None)
        assert not torch.allclose(cx1_a, cx1_no_prev, atol=1e-6)

    # 4) Caso límite: activo con TODAS las modalidades ausentes permanece finito y sin NaN
    with torch.no_grad():
        mask_all_zero = mask.clone()
        mask_all_zero[0, :] = 0.0
        cx_z, sal_z, pr_z, n_z = ce(tokens.detach(), mask_all_zero, phys.detach(), ctx.detach())
        assert torch.isfinite(cx_z).all()
        assert torch.isfinite(sal_z).all()
        assert torch.isfinite(pr_z).all()
        assert torch.isfinite(n_z).all()
        torch.testing.assert_close(sal_z[0], torch.zeros(n_mod), atol=1e-7, rtol=0.0)

    # 5) Gradientes finitos en todos los parámetros entrenables y en entradas
    age_grad = torch.rand(n_assets, n_mod) * 10.0
    cx_g, sal_g, _, notes_g = ce(tokens, mask, phys, ctx, modality_age=age_grad)
    loss = cx_g.pow(2).mean() + sal_g.pow(2).mean() + notes_g.pow(2).mean()
    loss.backward()
    for name, p in ce.named_parameters():
        assert p.grad is not None, f"Parámetro sin gradiente en ContextEngineerGPU: {name}"
        assert torch.isfinite(p.grad).all(), f"Gradiente no finito en {name}"
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert phys.grad is not None and torch.isfinite(phys.grad).all()
    assert ctx.grad is not None and torch.isfinite(ctx.grad).all()


def test_differentiable_quant_toolbox_six_tools_parity_and_finite_gradients():
    """Las 6 herramientas deterministas ejecutan en dispositivo con paridad y gradientes."""
    torch.manual_seed(19)
    n_assets, d_model, exp_dim, rmt_dim = 22, 24, 4, 4
    horizons = (1, 2, 5, 10, 21, 63)

    assert len(TOOL_NAMES) == 6
    assert TOOL_NAMES == (
        "rmt_factor_decompose",
        "hrr_episodic_lookup",
        "hypergraph_spillover",
        "structural_verifier",
        "portfolio_feasibility",
        "epistemic_risk_check",
    )

    toolbox = DifferentiableQuantToolbox(
        hidden_dim=d_model,
        rmt_dim=rmt_dim,
        n_hyperedges=6,
        horizons=horizons,
    )
    hrr = HRRModernHopfieldMemory(hidden_dim=d_model, max_slots=16, beta=6.0)
    head = MultiHorizonGranularHead(hidden_dim=d_model, horizons=horizons, exposures_dim=exp_dim)

    h = torch.randn(n_assets, d_model, requires_grad=True)
    exposures = torch.randn(n_assets, exp_dim)
    rmt_loadings = torch.randn(n_assets, rmt_dim)
    w_prev = torch.randn(n_assets) * 0.01
    w_prev = w_prev - w_prev.mean()
    regime_key = torch.randn(d_model)

    # 1) Tool 0: ortogonalidad exacta del residuo idiosincrático frente a factores B
    h_sys, h_idio = toolbox.decompose_factors(h.detach(), exposures=exposures)
    b_c = exposures - exposures.mean(dim=0, keepdim=True)
    b_n = b_c / b_c.std(dim=0, keepdim=True).clamp_min(1e-4)
    residual_proj = (b_n.T @ h_idio).abs().max()
    sys_proj = (b_n.T @ h_sys).abs().max()
    assert float(residual_proj) < float(sys_proj) * 0.02 + 1e-4

    # 2) Tool 1: paridad exacta con HRRModernHopfieldMemory (vacía y poblada)
    ret_empty, ood_empty = toolbox.lookup_hrr_memory(h.detach(), hrr, regime_key=regime_key)
    torch.testing.assert_close(ret_empty, torch.zeros_like(h), atol=1e-7, rtol=0.0)
    torch.testing.assert_close(ood_empty, torch.zeros(n_assets), atol=1e-7, rtol=0.0)

    q_store = torch.randn(4, d_model)
    v_store = torch.randn(4, d_model)
    hrr.store(q_store, v_store, regime_key=regime_key)
    ret_fast, ood_fast = toolbox.lookup_hrr_memory(q_store, hrr, regime_key=regime_key)
    ref_hrr = hrr.retrieve(q_store, regime_key=regime_key)
    torch.testing.assert_close(ret_fast, ref_hrr["retrieved"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ood_fast, ref_hrr["ood_energy"], atol=1e-5, rtol=1e-5)
    _, ood_far = toolbox.lookup_hrr_memory(
        q_store * 2.0 + torch.randn_like(q_store), hrr, regime_key=-regime_key
    )
    assert float(ood_far.mean()) > float(ood_fast.mean()) + 0.1

    # 3) Tool 2: paridad exacta con RMTSpilloverHypergraph (incluyendo instancia externa)
    hg_ref = RMTSpilloverHypergraph(hidden_dim=d_model, n_hyperedges=6, rmt_dim=rmt_dim, n_rounds=1)
    with torch.no_grad():
        hg_ref.out_scale.fill_(0.25)
    hg_out_zero_sync, _ = toolbox.run_hypergraph_spillover(
        h.detach(), hg_ref, rmt_loadings=rmt_loadings
    )
    hg_out_ref, _ = hg_ref(h.detach(), rmt_loadings=rmt_loadings)
    torch.testing.assert_close(hg_out_zero_sync, hg_out_ref, atol=1e-5, rtol=1e-5)

    # 4) Tool 4: invariantes exactos P+ (neutralidad dólar, caja, L1, reducción factorial)
    mu_test = torch.randn(n_assets) * 0.02
    vol_test = torch.full((n_assets,), 0.02)
    w_zero_sync = toolbox.run_portfolio_feasibility(
        mu_test, vol=vol_test, exposures=exposures, w_prev=w_prev
    )
    assert abs(float(w_zero_sync.sum())) < 1e-4
    assert float(w_zero_sync.abs().sum()) <= toolbox.portfolio_layer.gross_limit + 1e-4
    assert float(w_zero_sync.abs().max()) <= toolbox.portfolio_layer.pos_cap + 1e-5
    assert float(w_zero_sync.abs().sum()) > 1e-2

    # 4b) P0-3 Escala real N=2,600: O(1) Lipschitz impide congelamiento (||w||_1 > 0.5 en 6 pasos!)
    rng_big = torch.Generator().manual_seed(2600)
    n_big, k_exp = 2600, 9
    mu_big = torch.randn((n_big,), generator=rng_big, requires_grad=True)
    mu_big_scaled = mu_big * 0.01
    vol_big = torch.full((n_big,), 0.02)
    exp_big = torch.randn((n_big, k_exp), generator=rng_big)
    w_big = toolbox.run_portfolio_feasibility(mu_big_scaled, vol=vol_big, exposures=exp_big)
    assert torch.isfinite(w_big).all()
    assert float(w_big.abs().sum().detach()) > 0.5
    assert abs(float(w_big.sum().detach())) < 1e-4
    assert float(w_big.abs().max().detach()) <= toolbox.portfolio_layer.pos_cap + 1e-5
    # Verificar reducción real de exposición factorial frente a señal no penalizada
    b_big_c = exp_big - exp_big.mean(dim=0, keepdim=True)
    b_big_u = b_big_c / b_big_c.norm(dim=0, keepdim=True).clamp_min(1e-4)
    mu_uncon = (mu_big_scaled.detach() - mu_big_scaled.detach().mean())
    mu_uncon = mu_uncon / mu_uncon.abs().sum().clamp_min(1e-6)
    assert float((b_big_u.T @ w_big.detach()).norm()) < float((b_big_u.T @ mu_uncon).norm())
    w_big.pow(2).sum().backward()
    assert mu_big.grad is not None and torch.isfinite(mu_big.grad).all()
    assert float(mu_big.grad.abs().sum()) > 0.0

    # 4c) P1-4(c) Sin bypass crudo de h en Tools 0/1/2 (proyectan estrictamente deltas D+1)
    assert toolbox.tool0_proj.in_features == d_model + 1
    assert toolbox.tool1_proj.in_features == d_model + 1
    assert toolbox.tool2_proj.in_features == d_model + 1
    with torch.no_grad():
        hrr_empty_chk = HRRModernHopfieldMemory(hidden_dim=d_model, max_slots=8)
        t_a, _ = toolbox(torch.randn(n_assets, d_model), hrr_memory=hrr_empty_chk)
        t_b, _ = toolbox(torch.randn(n_assets, d_model) * 5.0, hrr_memory=hrr_empty_chk)
        # Con HRR vacía, Tool 1 no tiene bypass crudo de h -> salida idéntica para h_a vs h_b
        torch.testing.assert_close(t_a[:, 1, :], t_b[:, 1, :], atol=1e-6, rtol=0.0)

    # 5) Forward completo (sin pred_k externo): todos los parámetros internos reciben gradientes
    tool_out, aux = toolbox(
        h,
        exposures=exposures,
        rmt_loadings=rmt_loadings,
        hrr_memory=hrr,
        w_prev=w_prev,
        regime_key=regime_key,
    )
    assert tool_out.shape == (n_assets, 6, d_model)
    assert torch.isfinite(tool_out).all()
    for key in (
        "ood_energy",
        "structural_energy",
        "portfolio_weights",
        "turnover_pressure",
        "free_energy_margin",
        "epistemic_uncertainty",
    ):
        assert key in aux and isinstance(aux[key], torch.Tensor)
        assert torch.isfinite(aux[key]).all()

    loss = tool_out.pow(2).mean()
    loss.backward()
    for name, p in toolbox.named_parameters():
        assert p.grad is not None, f"Parámetro sin gradiente en DifferentiableQuantToolbox: {name}"
        assert torch.isfinite(p.grad).all(), f"Gradiente no finito en {name}"
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert float(h.grad.abs().sum()) > 0.0

    # 6) Forward con hipótesis externa `pred_k` de MultiHorizonGranularHead: gradiente a `head`
    toolbox.zero_grad()
    head.zero_grad()
    h2 = torch.randn(n_assets, d_model, requires_grad=True)
    pred_k = head(h2, exposures=exposures)
    tool_out2, _ = toolbox(
        h2,
        exposures=exposures,
        rmt_loadings=rmt_loadings,
        hrr_memory=hrr,
        pred_k=pred_k,
        w_prev=w_prev,
        regime_key=regime_key,
    )
    assert tool_out2.shape == (n_assets, 6, d_model)
    tool_out2.pow(2).mean().backward()
    assert h2.grad is not None and torch.isfinite(h2.grad).all()
    assert head.alpha_head.weight.grad is not None
    assert torch.isfinite(head.alpha_head.weight.grad).all()

    # 7) Tool 5: mayor energía OOD reduce el margen de Energía Libre |mu| - kappa * sigma_epist
    with torch.no_grad():
        _, aux_low_ood = toolbox(
            h2.detach(), exposures=exposures, pred_k=pred_k, ood_energy=torch.zeros(n_assets)
        )
        _, aux_hi_ood = toolbox(
            h2.detach(), exposures=exposures, pred_k=pred_k, ood_energy=torch.full((n_assets,), 5.0)
        )
        assert float(aux_hi_ood["free_energy_margin"].mean()) < float(
            aux_low_ood["free_energy_margin"].mean()
        )


def test_titans_cognitive_core_compact_weight_tied_params_hard_cap_and_contraction():
    """Default K=3, tope duro 1<=K<=4, weight-tied (~0 params extra K=4 vs K=1) y Lipschitz<1."""
    torch.manual_seed(29)
    assert DEFAULT_THINKING_STEPS == 3
    assert MAX_THINKING_STEPS == 4

    core_default = TitansCognitiveCore(hidden_dim=64, context_dim=12, core_dim=32)
    assert core_default.thinking_steps == 3
    assert isinstance(core_default.halt_gate, AdaptiveHaltGate)

    core_k1 = TitansCognitiveCore(
        hidden_dim=64,
        context_dim=12,
        core_dim=32,
        thinking_steps=1,
        tool_mode="soft",
        lipschitz_bound=0.88,
    )
    core_k4 = TitansCognitiveCore(
        hidden_dim=64,
        context_dim=12,
        core_dim=32,
        thinking_steps=4,
        tool_mode="soft",
        lipschitz_bound=0.88,
    )
    core_full_dim = TitansCognitiveCore(
        hidden_dim=64,
        context_dim=12,
        core_dim=64,
        thinking_steps=4,
        tool_mode="soft",
        lipschitz_bound=0.88,
    )

    p1 = sum(p.numel() for p in core_k1.parameters())
    p4 = sum(p.numel() for p in core_k4.parameters())
    # Compartición estricta de pesos: K=4 solo añade el pequeño step_embedding (~24 params, <0.5%)
    assert p4 - p1 <= 64
    assert (p4 - p1) / float(p1) < 0.005

    # Cuello de botella core_dim=32 < hidden_dim=64 reduce los parámetros del núcleo
    s_compact = core_k4.parameter_summary()
    s_wide = core_full_dim.parameter_summary()
    assert s_compact["core_params"] < s_wide["core_params"]
    assert s_compact["core_dim"] == 32

    # Validación estricta del tope duro 1 <= thinking_steps <= 4 y demás contratos
    with pytest.raises(ValueError):
        TitansCognitiveCore(hidden_dim=32, context_dim=8, core_dim=64)
    with pytest.raises(ValueError):
        TitansCognitiveCore(hidden_dim=32, context_dim=8, thinking_steps=0)
    with pytest.raises(ValueError):
        TitansCognitiveCore(hidden_dim=32, context_dim=8, thinking_steps=5)
    with pytest.raises(ValueError):
        TitansCognitiveCore(hidden_dim=32, context_dim=8, lipschitz_bound=1.0)
    with pytest.raises(ValueError):
        ToolRouter(context_dim=8, max_steps=4, mode="invalid_mode")

    # Contracción empírica de Banach en la celda: ||f(h_a) - f(h_b)|| < L * ||h_a - h_b||
    core_k4.eval()
    with torch.no_grad():
        x0 = torch.randn(16, 64)
        tool_msg = torch.randn(16, 64)
        h_a = torch.randn(16, 64)
        h_b = torch.randn(16, 64)
        out_a = core_k4.core_cell(x0, h_a, tool_msg)
        out_b = core_k4.core_cell(x0, h_b, tool_msg)
        ratio = float((out_a - out_b).norm() / (h_a - h_b).norm().clamp_min(1e-8))
        assert ratio < 0.88
        assert float(core_k4.core_cell.lipschitz_upper_bound()) < 0.88


def test_adaptive_halt_gate_monotonic_consistency_and_bounded_steps():
    """halt_gate aprende cuándo detenerse en pasos 1..3 (máx 4) y reacciona a consistencia."""
    torch.manual_seed(53)
    n_assets, d_model, c_dim = 16, 24, 10
    core = TitansCognitiveCore(
        hidden_dim=d_model,
        context_dim=c_dim,
        core_dim=16,
        thinking_steps=3,
        tool_mode="soft",
        rmt_dim=4,
        n_hyperedges=6,
    )
    x0 = torch.randn(n_assets, d_model)
    ctx = torch.randn(n_assets, c_dim)
    exposures = torch.randn(n_assets, 4)
    rmt = torch.randn(n_assets, 4)

    # 1) Distribución de parada válida en [1, 3] y pérdida de ponderación diferenciable
    core.train()
    ref_h, steps_h, diag = core(x0, ctx, exposures=exposures, rmt_loadings=rmt)
    halt_dist = diag["halting_distribution"]
    exp_steps = diag["expected_thinking_steps"]
    ponder_kl = diag["ponder_kl_loss"]
    mono_rate = diag["monotonic_accept_rate"]

    assert len(steps_h) == 3
    assert halt_dist.shape == (n_assets, 3)
    assert torch.all(halt_dist >= 0.0)
    torch.testing.assert_close(halt_dist.sum(dim=-1), torch.ones(n_assets), atol=1e-5, rtol=1e-5)
    assert 1.0 - 1e-5 <= float(exp_steps.detach()) <= 3.0 + 1e-5
    assert ponder_kl.ndim == 0 and torch.isfinite(ponder_kl) and float(ponder_kl.detach()) >= 0.0
    assert 0.0 < float(mono_rate.detach()) <= 1.0 + 1e-6

    # Verificar que ponder_kl_loss propaga gradientes finitos directamente a halt_gate
    core.zero_grad()
    (ref_h.pow(2).mean() + 0.1 * ponder_kl).backward()
    for name, p in core.halt_gate.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Gradiente inválido: {name}"

    # 2) Consistencia monótona: si la energía empeora (worsened > 0), halt_gate sube prob de parada
    core.eval()
    with torch.no_grad():
        h_s = torch.randn(n_assets, d_model)
        delta_n = torch.full((n_assets,), 0.2)
        e_base = torch.full((n_assets,), 0.5)
        step_e = core.tool_router.get_step_embedding(0, dtype=h_s.dtype)
        lam_good = core.halt_gate(
            h_s, delta_n, e_base, torch.zeros(n_assets), ctx, step_e
        )
        lam_worse = core.halt_gate(
            h_s, delta_n, e_base + 2.0, torch.full((n_assets,), 2.0), ctx, step_e
        )
        assert torch.all(lam_worse > lam_good)

    # 3) Sesgar halt_gate a parada temprana -> expected_steps -> 1 y refined_h -> step_hiddens[0]
    with torch.no_grad():
        core.halt_gate.out_head[-1].bias.fill_(10.0)
        ref_early, steps_early, diag_early = core(x0, ctx, exposures=exposures, rmt_loadings=rmt)
        assert float(diag_early["expected_thinking_steps"]) < 1.02
        torch.testing.assert_close(ref_early, steps_early[0], atol=1e-3, rtol=1e-3)

        # Sesgar halt_gate a ponderar al máximo -> expected_steps -> 3 y refined_h -> último paso
        core.halt_gate.out_head[-1].bias.fill_(-12.0)
        ref_late, steps_late, diag_late = core(x0, ctx, exposures=exposures, rmt_loadings=rmt)
        assert float(diag_late["expected_thinking_steps"]) > 2.95
        torch.testing.assert_close(ref_late, steps_late[-1], atol=1e-2, rtol=1e-2)

    # 4) Caso límite de sesión de 1 solo activo (n_assets == 1) sin NaN/Inf
    with torch.no_grad():
        ref_1, steps_1, diag_1 = core(
            x0[:1], ctx[:1], exposures=exposures[:1], rmt_loadings=rmt[:1]
        )
        assert ref_1.shape == (1, d_model) and torch.isfinite(ref_1).all()
        assert len(steps_1) == 3 and torch.isfinite(diag_1["expected_thinking_steps"])

    # 5) Cero desfase monótono en paso 0: un candidato h_cand que empeora E_struct es amortiguado
    #    inmediatamente en paso 0 (accept_0 < 0.5) y eleva halting_distribution[:, 0]
    core_mono = TitansCognitiveCore(
        hidden_dim=d_model,
        context_dim=c_dim,
        core_dim=16,
        thinking_steps=3,
        tool_mode="soft",
        rmt_dim=4,
        n_hyperedges=6,
        monotonic_penalty=5.0,
    )
    core_mono.eval()
    with torch.no_grad():
        core_mono.halt_gate.out_head[-1].bias.fill_(-2.0)
        shared_h = MultiHorizonGranularHead(hidden_dim=d_model, exposures_dim=4)
        orig_head_fwd = shared_h.forward
        orig_cell_fwd = core_mono.core_cell.forward
        x_calm = torch.randn(n_assets, d_model)
        _, _, diag_good = core_mono(
            x_calm, ctx, exposures=exposures, rmt_loadings=rmt, head=shared_h
        )

        # Inyectar candidato patológico en paso 0 con incoherencia multi-horizonte extrema
        call_counter = {"n": 0}
        bad_cand = torch.randn(n_assets, d_model) * 5.0
        alt_shock = torch.tensor([5.0, -5.0, 5.0, -5.0, 5.0, -5.0]).unsqueeze(0)

        def _bad_step0_cell(a_x, h_in, t_msg, *, cached_weights=None):
            if call_counter["n"] == 0:
                call_counter["n"] += 1
                return bad_cand
            call_counter["n"] += 1
            return orig_cell_fwd(a_x, h_in, t_msg, cached_weights=cached_weights)

        def _shock_head(h_in, *, exposures=None):
            res = orig_head_fwd(h_in, exposures=exposures)
            if torch.allclose(h_in, bad_cand, atol=1e-5):
                res = dict(res)
                res["pred_return"] = res["pred_return"] + alt_shock
            return res

        core_mono.core_cell.forward = _bad_step0_cell  # type: ignore[method-assign]
        shared_h.forward = _shock_head  # type: ignore[method-assign]
        _, steps_bad, diag_bad = core_mono(
            x_calm, ctx, exposures=exposures, rmt_loadings=rmt, head=shared_h
        )
        accept_step0 = diag_bad["monotonic_accepts_per_step"][0]
        assert float(accept_step0.mean()) < 0.05
        assert float((steps_bad[0] - x_calm).norm()) < float((bad_cand - x_calm).norm()) * 0.2
        assert (
            float(diag_bad["halting_distribution"][:, 0].mean())
            > float(diag_good["halting_distribution"][:, 0].mean()) + 0.2
        )


def test_cognitive_core_bfloat16_and_float16_blackwell_amp_safety():
    """Soporte nativo `bfloat16` (RTX PRO 6000 `bf16_amp`) y `float16` sin crash ni NaN."""
    torch.manual_seed(61)
    n_assets, d_model, c_dim, exp_dim, rmt_dim = 16, 24, 10, 4, 4

    # 1) TitansCognitiveCore completo en bfloat16 (con ContextEngineerGPU + HRR vacía y poblada)
    core_bf16 = TitansCognitiveCore(
        hidden_dim=d_model,
        context_dim=c_dim,
        core_dim=16,
        thinking_steps=3,
        tool_mode="soft",
        rmt_dim=rmt_dim,
        n_hyperedges=6,
        use_context_engineer=True,
    ).to(dtype=torch.bfloat16)
    hrr_empty = HRRModernHopfieldMemory(hidden_dim=d_model, max_slots=12)
    hrr_full = HRRModernHopfieldMemory(hidden_dim=d_model, max_slots=12)
    hrr_full.store(torch.randn(6, d_model), torch.randn(6, d_model))

    mod_tokens = torch.randn(n_assets, 5, d_model, dtype=torch.bfloat16, requires_grad=True)
    mod_mask = torch.ones(n_assets, 5, dtype=torch.bfloat16)
    mod_mask[:, 3] = 0.0
    phys = torch.randn(n_assets, d_model, dtype=torch.bfloat16)
    ctx = torch.randn(n_assets, c_dim, dtype=torch.bfloat16)
    exposures = torch.randn(n_assets, exp_dim, dtype=torch.bfloat16)
    rmt_loadings = torch.randn(n_assets, rmt_dim, dtype=torch.bfloat16)
    w_prev = torch.zeros(n_assets, dtype=torch.bfloat16)

    for hrr_mem in (hrr_empty, hrr_full):
        ref_h, steps_h, diag = core_bf16(
            x0=None,
            high_snr_context=ctx,
            mod_tokens=mod_tokens,
            modality_mask=mod_mask,
            physics_emb=phys,
            exposures=exposures,
            rmt_loadings=rmt_loadings,
            hrr_memory=hrr_mem,
            w_prev=w_prev,
        )
        assert ref_h.dtype == torch.bfloat16
        assert torch.isfinite(ref_h).all()
        assert len(steps_h) == 3 and all(torch.isfinite(s).all() for s in steps_h)
        assert torch.isfinite(diag["halting_distribution"]).all()
        assert torch.isfinite(diag["ponder_kl_loss"]).all()
        assert torch.isfinite(diag["tool_usage_distribution"]).all()

    (ref_h.float().pow(2).mean() + 0.1 * diag["ponder_kl_loss"].float()).backward()
    assert mod_tokens.grad is not None and torch.isfinite(mod_tokens.grad).all()

    # 2) FP16 crítico: decompose_factors y lookup_hrr_memory con buffer vacío sin overflow/NaN
    h_fp16 = torch.randn(n_assets, d_model, dtype=torch.float16)
    exp_fp16 = torch.randn(n_assets, exp_dim, dtype=torch.float16)
    h_sys, h_idio = DifferentiableQuantToolbox.decompose_factors(h_fp16, exposures=exp_fp16)
    assert h_sys.dtype == torch.float16 and torch.isfinite(h_sys).all()
    assert h_idio.dtype == torch.float16 and torch.isfinite(h_idio).all()

    ret_fp16, ood_fp16 = DifferentiableQuantToolbox.lookup_hrr_memory(h_fp16, hrr_empty)
    assert ret_fp16.dtype == torch.float16 and torch.isfinite(ret_fp16).all()
    assert ood_fp16.dtype == torch.float16 and torch.isfinite(ood_fp16).all()
    torch.testing.assert_close(ood_fp16, torch.zeros_like(ood_fp16), atol=0.0, rtol=0.0)


def test_titans_cognitive_core_modes_determinism_gradients_and_equivariance():
    """Verifica modos soft/sparsemax/uniform/no_tools, determinismo FP32, gradientes y equivar."""
    n_assets, d_model, c_dim, exp_dim, rmt_dim = 20, 32, 10, 4, 4
    assert TOOL_ROUTER_MODES == ("soft", "sparsemax", "uniform", "no_tools")

    torch.manual_seed(41)
    x0 = torch.randn(n_assets, d_model)
    ctx = torch.randn(n_assets, c_dim)
    exposures = torch.randn(n_assets, exp_dim)
    rmt_loadings = torch.randn(n_assets, rmt_dim)
    w_prev = torch.randn(n_assets) * 0.01
    w_prev = w_prev - w_prev.mean()

    # 1) Los 4 modos con state_dict compartido para ablación limpia (default thinking_steps=3)
    torch.manual_seed(101)
    base_core = TitansCognitiveCore(
        hidden_dim=d_model,
        context_dim=c_dim,
        core_dim=16,
        thinking_steps=3,
        tool_mode="soft",
        rmt_dim=rmt_dim,
        n_hyperedges=6,
    )
    outputs_by_mode = {}
    for mode in TOOL_ROUTER_MODES:
        core_m = TitansCognitiveCore(
            hidden_dim=d_model,
            context_dim=c_dim,
            core_dim=16,
            thinking_steps=3,
            tool_mode=mode,
            rmt_dim=rmt_dim,
            n_hyperedges=6,
        )
        core_m.load_state_dict(base_core.state_dict())
        core_m.eval()
        with torch.no_grad():
            ref_h, steps_h, diag = core_m(
                x0,
                ctx,
                exposures=exposures,
                rmt_loadings=rmt_loadings,
                w_prev=w_prev,
            )
        assert ref_h.shape == (n_assets, d_model)
        assert len(steps_h) == 3
        assert all(s.shape == (n_assets, d_model) for s in steps_h)
        assert torch.isfinite(ref_h).all()
        assert diag["tool_usage_distribution"].shape == (6,)
        assert diag["tool_entropy"].ndim == 0
        assert diag["pruned_context_ratio"].ndim == 0
        assert float(diag["lipschitz_upper_bound"]) < 1.0
        outputs_by_mode[mode] = (ref_h, steps_h, diag)

    # Verificar propiedades exactas de cada modo de enrutamiento
    _, _, d_soft = outputs_by_mode["soft"]
    _, _, d_sparse = outputs_by_mode["sparsemax"]
    _, _, d_uni = outputs_by_mode["uniform"]
    _, _, d_notools = outputs_by_mode["no_tools"]

    torch.testing.assert_close(
        d_soft["tool_usage_distribution"].sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5
    )
    assert float(d_soft["tool_entropy"]) > 0.0

    torch.testing.assert_close(
        d_sparse["tool_usage_distribution"].sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5
    )
    # Con logits agudos, sparsemax produce ceros exactos
    router_sp = ToolRouter(context_dim=c_dim, max_steps=3, mode="sparsemax", temperature=0.2)
    with torch.no_grad():
        router_sp.router_head[-1].bias.zero_()
        router_sp.router_head[-1].bias[0] = 4.0
        w_sp, _ = router_sp(ctx, 0)
    assert int((w_sp == 0.0).sum()) > 0
    torch.testing.assert_close(w_sp.sum(dim=-1), torch.ones(n_assets), atol=1e-5, rtol=1e-5)

    torch.testing.assert_close(
        d_uni["tool_usage_distribution"], torch.full((6,), 1.0 / 6.0), atol=1e-6, rtol=1e-6
    )
    assert float(d_uni["tool_entropy"]) == pytest.approx(math.log(6.0), rel=1e-4)

    torch.testing.assert_close(
        d_notools["tool_usage_distribution"], torch.zeros(6), atol=1e-7, rtol=0.0
    )
    assert float(d_notools["tool_entropy"]) == pytest.approx(0.0, abs=1e-7)
    assert not torch.allclose(outputs_by_mode["soft"][0], outputs_by_mode["no_tools"][0], atol=1e-5)

    # 2) Determinismo FP32 bit a bit
    torch.manual_seed(77)
    m1 = TitansCognitiveCore(
        hidden_dim=d_model, context_dim=c_dim, core_dim=16, thinking_steps=4, rmt_dim=rmt_dim
    )
    torch.manual_seed(77)
    m2 = TitansCognitiveCore(
        hidden_dim=d_model, context_dim=c_dim, core_dim=16, thinking_steps=4, rmt_dim=rmt_dim
    )
    m1.eval()
    m2.eval()
    with torch.no_grad():
        h1, s1, diag1 = m1(x0, ctx, exposures=exposures, rmt_loadings=rmt_loadings, w_prev=w_prev)
        h2, s2, diag2 = m2(x0, ctx, exposures=exposures, rmt_loadings=rmt_loadings, w_prev=w_prev)
    torch.testing.assert_close(h1, h2, atol=0.0, rtol=0.0)
    for a, b in zip(s1, s2, strict=True):
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        diag1["tool_usage_distribution"], diag2["tool_usage_distribution"], atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        diag1["halting_distribution"], diag2["halting_distribution"], atol=0.0, rtol=0.0
    )

    # 3) Equivarianza por permutación entre activos para refined_h, step_hiddens, halt y ToolRouter
    perm = torch.randperm(n_assets)
    with torch.no_grad():
        hp, sp, diagp = m1(
            x0[perm],
            ctx[perm],
            exposures=exposures[perm],
            rmt_loadings=rmt_loadings[perm],
            w_prev=w_prev[perm],
        )
    torch.testing.assert_close(h1[perm], hp, atol=1e-5, rtol=1e-5)
    for a, b in zip(s1, sp, strict=True):
        torch.testing.assert_close(a[perm], b, atol=1e-5, rtol=1e-5)
    for w_orig, w_perm in zip(
        diag1["tool_weights_per_step"], diagp["tool_weights_per_step"], strict=True
    ):
        torch.testing.assert_close(w_orig[perm], w_perm, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        diag1["halting_distribution"][perm], diagp["halting_distribution"], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        diag1["tool_usage_distribution"], diagp["tool_usage_distribution"], atol=1e-5, rtol=1e-5
    )

    # 5) Gradientes finitos en todos los parámetros internos y pérdida supervisada probe_aux_loss
    m_tr = TitansCognitiveCore(
        hidden_dim=d_model,
        context_dim=c_dim,
        core_dim=16,
        thinking_steps=4,
        tool_mode="soft",
        rmt_dim=rmt_dim,
        n_hyperedges=6,
        use_context_engineer=True,
    )
    m_tr.train()
    assert DEFAULT_PONDER_WEIGHT == pytest.approx(0.01)
    assert m_tr.ponder_weight == pytest.approx(0.01)
    mod_tokens = torch.randn(n_assets, 5, d_model, requires_grad=True)
    mod_mask = torch.ones(n_assets, 5)
    mod_mask[:, 2] = 0.0
    phys = torch.randn(n_assets, d_model, requires_grad=True)
    ctx_tr = torch.randn(n_assets, c_dim, requires_grad=True)
    age = torch.rand(n_assets, 5) * 5.0
    y_probe = torch.randn(n_assets, 6) * 0.02

    ref_tr, steps_tr, diag_tr = m_tr(
        x0=None,
        high_snr_context=ctx_tr,
        mod_tokens=mod_tokens,
        modality_mask=mod_mask,
        physics_emb=phys,
        modality_age=age,
        exposures=exposures,
        rmt_loadings=rmt_loadings,
        w_prev=w_prev,
        probe_targets=y_probe,
    )
    assert len(steps_tr) == 4
    assert float(diag_tr["pruned_context_ratio"]) >= 0.2 - 1e-6
    assert diag_tr["salience_weights"] is not None
    assert diag_tr["context_notes"] is not None
    assert "probe_aux_loss" in diag_tr and diag_tr["probe_aux_loss"].ndim == 0
    assert float(diag_tr["probe_aux_loss"].detach()) > 0.0
    torch.testing.assert_close(
        diag_tr["weighted_ponder_loss"],
        diag_tr["ponder_kl_loss"] * 0.01,
        atol=1e-7,
        rtol=1e-6,
    )
    # Persistencia causal de notas vía state["cognitive_notes"] (detached, cero fuga!)
    st_t0 = diag_tr["state"]
    assert "cognitive_notes" in st_t0
    assert not st_t0["cognitive_notes"].requires_grad
    assert st_t0["cognitive_notes"].shape == diag_tr["context_notes"].shape

    loss_tr = (
        sum(s.pow(2).mean() for s in steps_tr)
        + ref_tr.pow(2).mean()
        + diag_tr["weighted_ponder_loss"]
        + diag_tr["probe_aux_loss"]
    )
    loss_tr.backward()
    for name, p in m_tr.named_parameters():
        assert p.grad is not None, f"Parámetro sin gradiente en TitansCognitiveCore: {name}"
        assert torch.isfinite(p.grad).all(), f"Gradiente no finito en {name}"
    assert mod_tokens.grad is not None and torch.isfinite(mod_tokens.grad).all()
    assert phys.grad is not None and torch.isfinite(phys.grad).all()
    assert ctx_tr.grad is not None and torch.isfinite(ctx_tr.grad).all()

    # Persistencia causal entre sesiones: pasar state=st_t0 influye en t=1 vs state=None
    m_tr.eval()
    with torch.no_grad():
        ref_t1_virgin, _, _ = m_tr(
            x0=None,
            high_snr_context=ctx_tr.detach(),
            mod_tokens=mod_tokens.detach(),
            modality_mask=mod_mask,
            physics_emb=phys.detach(),
            state=None,
        )
        ref_t1_persisted, _, _ = m_tr(
            x0=None,
            high_snr_context=ctx_tr.detach(),
            mod_tokens=mod_tokens.detach(),
            modality_mask=mod_mask,
            physics_emb=phys.detach(),
            state=st_t0,
        )
    assert not torch.allclose(ref_t1_virgin, ref_t1_persisted, atol=1e-6)

    # P1-4(b) Cabeza compartida `head`: Tools 3/4/5 usan `head` y propagan gradiente a `head`
    shared_head = MultiHorizonGranularHead(hidden_dim=d_model, exposures_dim=exp_dim)
    m_sh = TitansCognitiveCore(
        hidden_dim=d_model,
        context_dim=c_dim,
        core_dim=16,
        thinking_steps=3,
        tool_mode="soft",
        rmt_dim=rmt_dim,
        n_hyperedges=6,
    )
    m_sh.train()
    ref_sh, _, diag_sh = m_sh(
        x0, ctx, exposures=exposures, rmt_loadings=rmt_loadings, head=shared_head
    )
    assert len(diag_sh["step_predictions"]) == 3
    (ref_sh.pow(2).mean() + diag_sh["weighted_ponder_loss"]).backward()
    assert shared_head.alpha_head.weight.grad is not None
    assert torch.isfinite(shared_head.alpha_head.weight.grad).all()


def test_cognitive_core_zero_host_syncs_static_ast_audit():
    """Auditoría AST estricta: CERO llamadas `.item()`, `.cpu()`, `.numpy()` o `.tolist()`."""
    mod_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mars_titan"
        / "models"
        / "mmarrs"
        / "cognitive_core.py"
    )
    source = mod_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"item", "cpu", "numpy", "tolist", "synchronize"}
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in forbidden:
                found.append((node.lineno, node.func.attr))
    assert found == [], f"Sincronizaciones CPU↔GPU prohibidas en cognitive_core.py: {found}"

    # Cero `torch.tensor(...)` HtoD fuera de `__init__` (CUDA-Graph / 0-HtoD safe)
    htod_found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name != "__init__":
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "tensor"
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == "torch"
                ):
                    htod_found.append((sub.lineno, node.name))
    assert htod_found == [], f"Asignaciones HtoD torch.tensor() fuera de __init__: {htod_found}"


def test_p0_4_real_scale_5bps_daily_alpha_n2600_no_zero_collapse_and_noise_suppression():
    """P0-4: alfa 5 bps a N=2600 mantiene ||w||_1 > 0.5 y suprime ruido sub-coste."""
    torch.manual_seed(404)
    n_us, k_fac = 2600, 9
    toolbox = DifferentiableQuantToolbox(hidden_dim=24, rmt_dim=4, n_hyperedges=6)

    # 1) Alfa diario FM-7 realista: std = 5 bps (0.0005), vol = 200 bps (0.02)
    mu_raw = torch.randn(n_us, requires_grad=True)
    mu_5bps = mu_raw * 0.0005
    vol = torch.full((n_us,), 0.02)
    exposures = torch.randn(n_us, k_fac)

    w = toolbox.run_portfolio_feasibility(mu_5bps, vol=vol, exposures=exposures)
    assert torch.isfinite(w).all()
    assert float(w.abs().sum().detach()) > 0.5
    assert abs(float(w.sum().detach())) < 1e-5
    assert float(w.abs().max().detach()) <= toolbox.portfolio_layer.pos_cap + 1e-6

    # Reducción real de exposición factorial frente a señal sin penalizar
    b_c = exposures - exposures.mean(dim=0, keepdim=True)
    b_u = b_c / b_c.norm(dim=0, keepdim=True).clamp_min(1e-4)
    mu_c = mu_5bps.detach() - mu_5bps.detach().mean()
    mu_u = mu_c / mu_c.abs().sum().clamp_min(1e-6)
    assert float((b_u.T @ w.detach()).norm()) < float((b_u.T @ mu_u).norm())

    # Gradientes vivos hacia mu
    w.pow(2).sum().backward()
    assert mu_raw.grad is not None and torch.isfinite(mu_raw.grad).all()
    assert float(mu_raw.grad.abs().sum()) > 0.0

    # 2) Ruido puro por debajo del coste amortizado (< 0.5 bps) sigue colapsando a w == 0
    with torch.no_grad():
        mu_noise = torch.clamp(torch.randn(n_us) * 1e-5, min=-4e-5, max=4e-5)
        w_noise = toolbox.run_portfolio_feasibility(mu_noise, vol=vol, exposures=exposures)
        torch.testing.assert_close(w_noise, torch.zeros_like(w_noise), atol=1e-7, rtol=0.0)


def test_p1_2_pos_cap_hard_guarantee_under_extreme_skew():
    """P1-2: pos_cap y neutralidad dólar se cumplen SIEMPRE (incluso bajo sesgo extremo)."""
    toolbox = DifferentiableQuantToolbox(
        hidden_dim=16, rmt_dim=4, n_hyperedges=4, portfolio_pos_cap=0.05, portfolio_gross_limit=1.0
    )
    for n_assets in (3, 10, 64):
        mu = torch.full((n_assets,), -0.005)
        mu[0] = 5.0  # outlier extremo en 1 solo activo
        mu = mu.clone().detach().requires_grad_(True)
        w = toolbox.run_portfolio_feasibility(mu, vol=torch.full((n_assets,), 0.02))
        assert torch.isfinite(w).all()
        assert float(w.abs().max().detach()) <= 0.05 + 1e-6, f"pos_cap violado en N={n_assets}"
        assert abs(float(w.sum().detach())) < 1e-5, f"neutralidad violada en N={n_assets}"
        assert float(w.abs().sum().detach()) <= 1.0 + 1e-5
        w.pow(2).sum().backward()
        assert mu.grad is not None and torch.isfinite(mu.grad).all()


def test_p1_2_hoisted_exposures_no_eigvalsh_head_reuse_and_spectral_post_step_hook(monkeypatch):
    """P1-2: cero eigvalsh, solve hoisted 1x, head() <= K+1, y _SpectralLinear sin mutar forward."""
    torch.manual_seed(412)
    n_assets, d_model, c_dim, exp_dim, rmt_dim = 32, 24, 10, 6, 4

    # 1) AST estricto: CERO llamadas a `eigvalsh` en cognitive_core.py
    mod_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mars_titan"
        / "models"
        / "mmarrs"
        / "cognitive_core.py"
    )
    tree = ast.parse(mod_path.read_text(encoding="utf-8"))
    eig_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "eigvalsh"
    ]
    assert eig_calls == [], f"cuSOLVER eigvalsh prohibido en cognitive_core.py: {eig_calls}"

    core = TitansCognitiveCore(
        hidden_dim=d_model,
        context_dim=c_dim,
        core_dim=16,
        thinking_steps=3,
        tool_mode="soft",
        rmt_dim=rmt_dim,
        n_hyperedges=6,
    )
    shared_head = MultiHorizonGranularHead(hidden_dim=d_model, exposures_dim=exp_dim)

    x0 = torch.randn(n_assets, d_model)
    ctx = torch.randn(n_assets, c_dim)
    exposures = torch.randn(n_assets, exp_dim)
    rmt = torch.randn(n_assets, rmt_dim)

    # 2) Contar llamadas a torch.linalg.solve_ex (hoisted 1x, 0 syncs!) y a shared_head (<= K+1)
    solve_ex_calls = {"n": 0}
    solve_sync_calls = {"n": 0}
    orig_solve_ex = torch.linalg.solve_ex
    orig_solve = torch.linalg.solve

    def _counted_solve_ex(*args, **kwargs):
        solve_ex_calls["n"] += 1
        return orig_solve_ex(*args, **kwargs)

    def _counted_solve(*args, **kwargs):
        solve_sync_calls["n"] += 1
        return orig_solve(*args, **kwargs)

    head_calls = {"n": 0}
    orig_head = shared_head.forward

    def _counted_head(h_in: torch.Tensor, *, exposures: torch.Tensor | None = None):
        head_calls["n"] += 1
        return orig_head(h_in, exposures=exposures)

    monkeypatch.setattr(torch.linalg, "solve_ex", _counted_solve_ex)
    monkeypatch.setattr(torch.linalg, "solve", _counted_solve)
    monkeypatch.setattr(shared_head, "forward", _counted_head)

    # 3) En train(): _SpectralLinear NO muta buffers u/v durante forward(), solo en post-step hook
    core.train()
    uv_before = {
        k: v.clone() for k, v in core.core_cell.named_buffers() if k.endswith((".u", ".v"))
    }
    assert len(uv_before) == 10
    opt = torch.optim.SGD(core.parameters(), lr=0.1)
    hook = core.register_spectral_post_step_hook(opt)

    ref_h, _, diag = core(
        x0, ctx, exposures=exposures, rmt_loadings=rmt, head=shared_head
    )
    assert solve_sync_calls["n"] == 0, f"linalg.solve sincrono prohibido: {solve_sync_calls['n']}"
    assert solve_ex_calls["n"] == 1, f"Esperado 1 solve_ex hoisted, recibido {solve_ex_calls['n']}"
    assert head_calls["n"] <= core.thinking_steps + 1, (
        f"Esperado <= {core.thinking_steps + 1} llamadas a head, recibido {head_calls['n']}"
    )
    for k, prev in uv_before.items():
        cur = dict(core.core_cell.named_buffers())[k]
        torch.testing.assert_close(cur, prev, atol=0.0, rtol=0.0)

    (ref_h.pow(2).mean() + diag["weighted_ponder_loss"]).backward()
    opt.step()
    changed = any(
        not torch.allclose(dict(core.core_cell.named_buffers())[k], prev, atol=0.0, rtol=0.0)
        for k, prev in uv_before.items()
    )
    assert changed
    hook.remove()


def test_v1_bugfixes_bias_guard_clone_horizon_mask_triton_fista_and_t0_t1_t2(monkeypatch):
    """5 correcciones v1 (v1+v2): bias guard+.clone, horizon_mask, FISTA fast, T0-T2."""
    import mars_titan.evaluation.metacognitive_agent as ma

    torch.manual_seed(501)
    n_assets, d_model, c_dim, exp_dim, rmt_dim = 24, 24, 10, 4, 4

    # 1) ToolRouter.update_balance_bias: no-op en eval(), train() seguro bajo BPTT in-place
    router = ToolRouter(context_dim=c_dim, max_steps=3, n_tools=6, mode="soft")
    ctx_r = torch.randn(n_assets, c_dim, requires_grad=True)
    skewed_loads = torch.tensor([0.90, 0.02, 0.02, 0.02, 0.02, 0.02])

    router.eval()
    before_eval = router.balance_bias.clone()
    router.update_balance_bias(skewed_loads, lr=0.1)
    torch.testing.assert_close(router.balance_bias, before_eval, atol=0.0, rtol=0.0)

    router.train()
    w_r, _ = router(ctx_r, 0)
    # Simular update_balance_bias in-place antes de backward (BPTT): .clone() impide version error
    router.update_balance_bias(skewed_loads, lr=0.1)
    assert not torch.allclose(router.balance_bias, before_eval, atol=1e-6)
    assert float(router.balance_bias.abs().max()) <= 3.0 + 1e-6
    w_r.pow(2).sum().backward()
    assert ctx_r.grad is not None and torch.isfinite(ctx_r.grad).all()

    # 2) Tool 3 (structural_verifier): P0-4 horizon_mask anula horizontes inactivos h=2..63
    toolbox = DifferentiableQuantToolbox(
        hidden_dim=d_model, rmt_dim=rmt_dim, n_hyperedges=6, horizons=(1, 2, 5, 10, 21, 63)
    )
    ret_clean = torch.randn(n_assets, 6) * 0.01
    alpha_clean = ret_clean - ret_clean.mean(dim=0, keepdim=True)
    vol_clean = torch.full((n_assets, 6), 0.02)
    epist_clean = torch.full((n_assets, 6), 0.005)
    q_base = torch.linspace(-0.04, 0.04, steps=9).view(1, 1, 9).expand(n_assets, 6, 9).clone()
    pred_clean = dict(
        pred_return=ret_clean.clone(),
        alpha=alpha_clean.clone(),
        vol_pred=vol_clean.clone(),
        epistemic_error=epist_clean.clone(),
        quantiles=q_base.clone(),
    )
    pred_poison_inactive = dict(
        pred_return=ret_clean.clone(),
        alpha=alpha_clean.clone(),
        vol_pred=vol_clean.clone(),
        epistemic_error=epist_clean.clone(),
        quantiles=q_base.clone(),
    )
    # Corromper masivamente horizontes inactivos h=1..5 (retornos y cuantiles cruzados)
    pred_poison_inactive["pred_return"][:, 1:] = 50.0
    pred_poison_inactive["quantiles"][:, 1:, :] = q_base[:, 1:, :].flip(-1) * 20.0
    hmask_1d = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    _, e_clean_1d = toolbox.verify_structure(pred_clean, horizon_mask=hmask_1d)
    _, e_pois_1d = toolbox.verify_structure(pred_poison_inactive, horizon_mask=hmask_1d)
    _, e_pois_all = toolbox.verify_structure(pred_poison_inactive, horizon_mask=None)
    torch.testing.assert_close(e_clean_1d, e_pois_1d, atol=1e-6, rtol=1e-6)
    assert float(e_pois_all.mean()) > float(e_pois_1d.mean()) + 10.0

    # 3) P0-5 Tool 4: SIEMPRE ejecuta `_diff_portfolio_fista_fwd_op` (aux_req_grad==False) + liq_pct
    fista_calls = {"n": 0}
    orig_fista = ma._diff_portfolio_fista_fwd_op

    def _spy_fista(*args, **kwargs):
        fista_calls["n"] += 1
        return orig_fista(*args, **kwargs)

    monkeypatch.setattr(ma, "_diff_portfolio_fista_fwd_op", _spy_fista)
    mu_live = (torch.randn(n_assets) * 0.01).requires_grad_(True)
    vol_live = torch.full((n_assets,), 0.02, requires_grad=True)
    w_prev_live = torch.zeros(n_assets, requires_grad=True)
    exp_live = torch.randn(n_assets, exp_dim, requires_grad=True)
    liq_live = torch.linspace(0.05, 0.95, steps=n_assets, requires_grad=True)

    w_opt = toolbox.run_portfolio_feasibility(
        mu_live, vol=vol_live, exposures=exp_live, w_prev=w_prev_live, liq_pct=liq_live
    )
    assert fista_calls["n"] == 1, "Tool 4 cayó al fallback Python (no _diff_portfolio_fista_fwd_op)"
    w_opt.pow(2).sum().backward()
    assert mu_live.grad is not None and float(mu_live.grad.abs().sum()) > 0.0
    # Disciplina P0-5: vol, w_prev, exposures, liq_pct desconectados en Tool 4
    assert vol_live.grad is None
    assert w_prev_live.grad is None
    assert exp_live.grad is None
    assert liq_live.grad is None

    # Activo ilíquido (liq_pct=0.05) recibe menor |w| que líquido (liq_pct=0.95) a escala FM-7
    with torch.no_grad():
        mu_sym = torch.tensor([0.0008, -0.0008, 0.0008, -0.0008] * 4)
        vol_sym = torch.full((16,), 0.04)
        liq_sym = torch.tensor([0.95, 0.95, 0.05, 0.05] * 4)
        w_sym = toolbox.run_portfolio_feasibility(mu_sym, vol=vol_sym, liq_pct=liq_sym)
        assert float(w_sym[0::4].abs().mean()) > float(w_sym[2::4].abs().mean()) + 1e-4

    # 4) T0: clip +/- 6*MAD protege el hiperplano OLS frente a outlier extremo (+1000)
    h_base = torch.randn(32, d_model)
    exp_t0 = torch.randn(32, exp_dim)
    h_out = h_base.clone()
    h_out[0] = 1000.0
    sys_clean, _ = toolbox.decompose_factors(h_base, exposures=exp_t0)
    sys_out, _ = toolbox.decompose_factors(h_out, exposures=exp_t0)
    # Los otros 31 activos no deben explotar por culpa del outlier en el activo 0
    assert float((sys_out[1:] - sys_clean[1:]).abs().max()) < 2.0

    # 5) T1: enlaza regime_key y sector_key (mismatch en sector_key eleva OOD)
    hrr = HRRModernHopfieldMemory(hidden_dim=d_model, max_slots=16, beta=6.0)
    rk = torch.randn(d_model)
    sk_a = torch.randn(d_model)
    sk_b = -sk_a
    q_s = torch.randn(6, d_model)
    hrr.store(q_s, torch.randn(6, d_model), regime_key=rk, sector_key=sk_a)
    _, ood_match = toolbox.lookup_hrr_memory(q_s, hrr, regime_key=rk, sector_key=sk_a)
    _, ood_wrong_sec = toolbox.lookup_hrr_memory(q_s, hrr, regime_key=rk, sector_key=sk_b)
    assert float(ood_wrong_sec.mean()) > float(ood_match.mean()) + 0.05

    # 6) T2: peer_disp refleja dispersión transversal intra-hiperarista
    h_calm = torch.ones(n_assets, d_model) * 0.1 + torch.randn(n_assets, d_model) * 0.01
    h_disp = torch.randn(n_assets, d_model) * 3.0
    _, _, disp_calm = toolbox.run_hypergraph_spillover(h_calm, return_peer_disp=True)
    _, _, disp_high = toolbox.run_hypergraph_spillover(h_disp, return_peer_disp=True)
    assert disp_calm.shape == (n_assets,) and torch.all(disp_calm >= 0.0)
    assert float(disp_high.detach().mean()) > float(disp_calm.detach().mean()) * 5.0


def test_v2_10tools_t6_rank_calibrator_t7_conflict_arbiter_nan_cn_and_macro_exclusion():
    """T6 (gaussian_rank_features + MAD/Huber/liq terciles) y T7 (conflicto sin macro + CN NaN)."""
    from mars_titan.models.mmarrs.cognitive_core import TOOL_NAMES_V2

    torch.manual_seed(602)
    assert len(TOOL_NAMES_V2) == 10
    assert TOOL_NAMES_V2[:6] == TOOL_NAMES
    assert TOOL_NAMES_V2[6:] == (
        "cross_sectional_rank_calibrator",
        "cross_modal_conflict_arbiter",
        "regime_stress_counterfactual",
        "working_memory_scratchpad_ops",
    )

    n_assets, d_model, rmt_dim = 30, 24, 4
    tb = DifferentiableQuantToolbox(
        hidden_dim=d_model,
        rmt_dim=rmt_dim,
        n_hyperedges=6,
        toolbox_version="v2_10tools",
    )

    # 1) T6: empates CN (+/-10%), NaN/Inf, constante (iqr==0), outlier Huber y terciles de liquidez
    alpha_ties = torch.cat([torch.zeros(15), torch.linspace(-0.02, 0.02, 15)]).requires_grad_(True)
    vol = torch.full((n_assets,), 0.02)
    liq = torch.linspace(0.05, 0.95, n_assets)
    f6, out_mag = tb.calibrate_cross_sectional_ranks(alpha_ties, vol, liq_pct=liq)
    assert f6.shape == (n_assets, 10)
    assert torch.isfinite(f6).all() and torch.isfinite(out_mag).all()
    # Empates idénticos reciben el mismo g_exact (sin sesgo por orden de ticker)
    torch.testing.assert_close(f6[:15, 0], f6[0:1, 0].expand(15), atol=1e-5, rtol=1e-5)
    f6.pow(2).sum().backward()
    assert alpha_ties.grad is not None and torch.isfinite(alpha_ties.grad).all()

    # Caso límite: NaN + Inf + señal constante (iqr==0 -> clamp_min(1e-4) seguro)
    alpha_bad = torch.full((n_assets,), 0.005)
    alpha_bad[0] = float("nan")
    alpha_bad[1] = float("inf")
    f6_bad, out_bad = tb.calibrate_cross_sectional_ranks(alpha_bad, vol, liq_pct=liq)
    assert torch.isfinite(f6_bad).all() and torch.isfinite(out_bad).all()

    # Outlier extremo activa Huber outlier_mag > 0
    alpha_ext = torch.randn(n_assets) * 0.01
    alpha_ext[0] = 0.50
    _, out_ext = tb.calibrate_cross_sectional_ranks(alpha_ext, vol, liq_pct=liq)
    assert float(out_ext[0]) > 2.0
    assert float(out_ext[1:].mean()) < float(out_ext[0]) * 0.25

    # 2) T7: aislamiento CN NaN en fundamentales ausentes + exclusión de macro + release day
    mod_tokens = torch.randn(n_assets, 5, d_model)
    # Alinear precios (0), fundamentales (1), noticias (3), gráficos (4)
    base_dir = torch.randn(n_assets, d_model)
    mod_aligned = torch.stack(
        [base_dir, base_dir, torch.randn(1, d_model).expand(n_assets, -1), base_dir, base_dir],
        dim=1,
    )
    mod_conflict = torch.stack(
        [
            base_dir,
            -base_dir,
            torch.randn(1, d_model).expand(n_assets, -1),
            -base_dir,
            base_dir,
        ],
        dim=1,
    )
    mask_cn = torch.ones(n_assets, 5)
    mask_cn[:, 1] = 0.0  # 99.3% fundamentales ausentes en CN
    age_no_rel = torch.full((n_assets, 5), 5.0)
    age_rel = age_no_rel.clone()
    age_rel[:, 2] = 0.5  # día de publicación macro (<= 1.0)

    # Veneno NaN/Inf en fundamentales ausentes -> aislamiento bit-a-bit exacto
    mod_pois = mod_tokens.clone()
    mod_pois[:, 1, :] = float("nan")
    mod_pois[0::2, 1, :] = float("inf")
    alpha0 = torch.randn(n_assets) * 0.01
    c_clean = tb.prepare_modality_cache(mod_tokens, mask_cn, modality_age=age_no_rel)
    c_pois = tb.prepare_modality_cache(mod_pois, mask_cn, modality_age=age_no_rel)
    d_cl, s_cl, cs_cl, _ = tb.arbitrate_modal_conflict(alpha0, vol, mod_cache=c_clean)
    d_po, s_po, cs_po, _ = tb.arbitrate_modal_conflict(alpha0, vol, mod_cache=c_pois)
    torch.testing.assert_close(d_cl, d_po, atol=0.0, rtol=0.0)
    torch.testing.assert_close(s_cl, s_po, atol=0.0, rtol=0.0)
    torch.testing.assert_close(cs_cl, cs_po, atol=0.0, rtol=0.0)

    # Exclusión de macro constante de sesión + flag macro_event_flag SOLO en días de publicación
    mask_all = torch.ones(n_assets, 5)
    mod_macro_a = mod_aligned.clone()
    mod_macro_b = mod_aligned.clone()
    mod_macro_b[:, 2, :] = -mod_macro_a[:, 2, :] * 10.0  # flip masivo solo en macro
    c_ma = tb.prepare_modality_cache(mod_macro_a, mask_all, modality_age=age_no_rel)
    c_mb = tb.prepare_modality_cache(mod_macro_b, mask_all, modality_age=age_no_rel)
    _, s_ma, cs_ma, _ = tb.arbitrate_modal_conflict(alpha0, vol, mod_cache=c_ma)
    _, s_mb, cs_mb, _ = tb.arbitrate_modal_conflict(alpha0, vol, mod_cache=c_mb)
    # El conflicto transversal NO depende de macro
    torch.testing.assert_close(cs_ma, cs_mb, atol=1e-6, rtol=1e-6)
    # macro_event_flag (col 6 de scalars): 0 fuera de release day, 1 en release day
    c_rel = tb.prepare_modality_cache(mod_macro_a, mask_all, modality_age=age_rel)
    _, s_rel, _, _ = tb.arbitrate_modal_conflict(alpha0, vol, mod_cache=c_rel)
    torch.testing.assert_close(s_ma[:, 6], torch.zeros(n_assets), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(s_rel[:, 6], torch.ones(n_assets), atol=1e-6, rtol=0.0)

    # Modalidades opuestas producen conflict_score estrictamente mayor que modalidades alineadas
    c_al = tb.prepare_modality_cache(mod_aligned, mask_all, modality_age=age_no_rel)
    c_cf = tb.prepare_modality_cache(mod_conflict, mask_all, modality_age=age_no_rel)
    _, _, cs_al, _ = tb.arbitrate_modal_conflict(alpha0, vol, mod_cache=c_al)
    _, _, cs_cf, _ = tb.arbitrate_modal_conflict(alpha0, vol, mod_cache=c_cf)
    assert float(cs_al.detach().min()) >= 0.0 and float(cs_cf.detach().max()) <= 1.0
    assert float(cs_cf.detach().mean()) > float(cs_al.detach().mean()) + 0.25


def test_v2_10tools_t8_counterfactual_placebo_and_t9_merged_scratchpad():
    """T8 delega en CounterfactualStressSimulator (+placebo) y T9 usa cognitive_notes [8,D]."""
    from mars_titan.models.mmarrs.counterfactual_sim import CounterfactualStressSimulator

    torch.manual_seed(703)
    n_assets, d_model, c_dim, exp_dim, rmt_dim = 28, 24, 10, 4, 4

    core = TitansCognitiveCore(
        hidden_dim=d_model,
        context_dim=c_dim,
        core_dim=16,
        thinking_steps=3,
        tool_mode="soft",
        toolbox_version="v2_10tools",
        rmt_dim=rmt_dim,
        n_hyperedges=6,
        use_context_engineer=True,
    )
    assert isinstance(core.toolbox.cf_sim, CounterfactualStressSimulator)
    assert sum(p.numel() for p in core.toolbox.cf_sim.parameters()) == 0
    assert core.context_engineer is not None
    assert core.context_engineer.n_notes == 8
    assert core.context_engineer.n_persistent_notes == 4

    # 1) T8: alfa crowded en factores sufre más e_cf que alfa ortogonal; placebo rompe alineación
    exposures = torch.randn(n_assets, exp_dim)
    rmt = torch.randn(n_assets, rmt_dim)
    ctx = torch.rand(n_assets, c_dim)
    ctx[:, 6] = 0.85  # vol_pct alto
    ctx[:, 7] = 0.15  # liq_pct bajo
    vol_t = torch.full((n_assets,), 0.02, requires_grad=True)

    # Alfa 100% alineado al factor 0 vs alfa ortogonal a exposiciones
    f0 = exposures[:, 0] - exposures[:, 0].mean()
    alpha_crowded = (0.02 * f0 / f0.std().clamp_min(1e-4)).requires_grad_(True)
    alpha_orth_raw = torch.randn(n_assets)
    b_c = exposures - exposures.mean(dim=0, keepdim=True)
    proj = b_c @ torch.linalg.lstsq(b_c, alpha_orth_raw.unsqueeze(-1)).solution.squeeze(-1)
    alpha_idio = (0.02 * (alpha_orth_raw - proj) / (alpha_orth_raw - proj).std().clamp_min(1e-4))

    cf_crowd = core.toolbox.cf_sim(
        alpha_crowded, vol_t, high_snr_context=ctx, exposures=exposures, rmt_loadings=rmt
    )
    cf_idio = core.toolbox.cf_sim(
        alpha_idio, vol_t.detach(), high_snr_context=ctx, exposures=exposures, rmt_loadings=rmt
    )
    cf_placebo = core.toolbox.cf_sim(
        alpha_crowded.detach(),
        vol_t.detach(),
        high_snr_context=ctx,
        exposures=exposures,
        rmt_loadings=rmt,
        placebo_mode=True,
    )
    assert float(cf_crowd["e_cf"].detach().mean()) > float(cf_idio["e_cf"].mean())
    assert not torch.allclose(cf_crowd["cf_trace"], cf_placebo["cf_trace"], atol=1e-4)
    # Disciplina de gradiente: vol_t no recibe gradiente de T8
    cf_crowd["e_cf"].mean().backward()
    assert alpha_crowded.grad is not None and torch.isfinite(alpha_crowded.grad).all()
    assert vol_t.grad is None

    # 2) T9 fusionado en cognitive_notes [8, D]: slots 0..3 persisten t-1->t, slots 4..7 se resetean
    core.eval()
    mod_tokens = torch.randn(n_assets, 5, d_model)
    mod_mask = torch.ones(n_assets, 5)
    phys = torch.randn(n_assets, d_model)
    with torch.no_grad():
        _, _, diag0 = core(
            x0=None,
            high_snr_context=ctx,
            mod_tokens=mod_tokens,
            modality_mask=mod_mask,
            physics_emb=phys,
            exposures=exposures,
            rmt_loadings=rmt,
        )
        notes0 = diag0["state"]["cognitive_notes"]
        assert notes0.shape == (8, d_model)
        assert not notes0.requires_grad
        assert float(diag0["scratchpad_novelty_mean"]) > 0.0

        # Alterar SOLO slots efímeros 4..7 de prev_notes -> salida de sesión t+1 BIT-A-BIT idéntica!
        notes_mut_ephem = notes0.clone()
        notes_mut_ephem[4:] = torch.randn(4, d_model) * 50.0
        h_a, _, _ = core(
            x0=None,
            high_snr_context=ctx,
            mod_tokens=mod_tokens,
            modality_mask=mod_mask,
            physics_emb=phys,
            exposures=exposures,
            rmt_loadings=rmt,
            state=dict(cognitive_notes=notes0),
        )
        h_b, _, _ = core(
            x0=None,
            high_snr_context=ctx,
            mod_tokens=mod_tokens,
            modality_mask=mod_mask,
            physics_emb=phys,
            exposures=exposures,
            rmt_loadings=rmt,
            state=dict(cognitive_notes=notes_mut_ephem),
        )
        torch.testing.assert_close(h_a, h_b, atol=0.0, rtol=0.0)

        # Alterar slots persistentes 0..3 -> SÍ influye causalmente en sesión t+1
        notes_mut_pers = notes0.clone()
        notes_mut_pers[:4] = torch.randn(4, d_model) * 5.0
        h_c, _, _ = core(
            x0=None,
            high_snr_context=ctx,
            mod_tokens=mod_tokens,
            modality_mask=mod_mask,
            physics_emb=phys,
            exposures=exposures,
            rmt_loadings=rmt,
            state=dict(cognitive_notes=notes_mut_pers),
        )
        assert not torch.allclose(h_a, h_c, atol=1e-5)


def test_v2_10tools_closed_loop_router_anti_collapse_adaptive_halt_and_telemetry():
    """Verifica v2: modos+placebo, anti-colapso limpio, easy<hard halt, telemetría y grads."""
    from mars_titan.models.mmarrs.cognitive_core import TOOL_NAMES_V2

    torch.manual_seed(804)
    n_assets, d_model, c_dim, exp_dim, rmt_dim = 32, 24, 10, 4, 4

    # 0) Confirmar que tool_marginal_utility_credit fue CORTADO per #133
    mod_src = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mars_titan"
        / "models"
        / "mmarrs"
        / "cognitive_core.py"
    ).read_text(encoding="utf-8")
    assert "tool_marginal_utility_credit" not in mod_src

    mod_tokens = torch.randn(n_assets, 5, d_model, requires_grad=True)
    mod_mask = torch.ones(n_assets, 5)
    mod_mask[:, 1] = 0.0
    phys = torch.randn(n_assets, d_model, requires_grad=True)
    ctx = torch.rand(n_assets, c_dim)
    exposures = torch.randn(n_assets, exp_dim)
    rmt = torch.randn(n_assets, rmt_dim)
    w_prev = torch.randn(n_assets) * 0.01
    w_prev = w_prev - w_prev.mean()
    hmask_1d = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 1) Todos los modos v2 (soft/sparsemax/uniform/no_tools/placebo) comparten state_dict exacto
    torch.manual_seed(805)
    base_v2 = TitansCognitiveCore(
        hidden_dim=d_model,
        context_dim=c_dim,
        core_dim=16,
        thinking_steps=3,
        tool_mode="soft",
        toolbox_version="v2_10tools",
        rmt_dim=rmt_dim,
        n_hyperedges=6,
        use_context_engineer=True,
    )
    required_diag_keys = (
        "tool_weights_mean",
        "tool_weights_by_step",
        "tool_load_max_min_ratio",
        "tool_active_count",
        "tool_corr_max",
        "expected_steps_mean",
        "expected_steps_easy_q25",
        "expected_steps_hard_q75",
        "cf_sign_flip_k0",
        "cf_sign_flip_final",
        "conflict_mean",
        "scratchpad_novelty_mean",
        "tool_aux_loss",
    )

    mode_out = {}
    for label, mode, placebo in (
        ("soft", "soft", False),
        ("sparsemax", "sparsemax", False),
        ("uniform", "uniform", False),
        ("no_tools", "no_tools", False),
        ("placebo", "soft", True),
    ):
        m = TitansCognitiveCore(
            hidden_dim=d_model,
            context_dim=c_dim,
            core_dim=16,
            thinking_steps=3,
            tool_mode=mode,
            toolbox_version="v2_10tools",
            cf_placebo_mode=placebo,
            rmt_dim=rmt_dim,
            n_hyperedges=6,
            use_context_engineer=True,
        )
        m.load_state_dict(base_v2.state_dict())
        m.eval()
        with torch.no_grad():
            ref_h, steps_h, diag = m(
                x0=None,
                high_snr_context=ctx,
                mod_tokens=mod_tokens.detach(),
                modality_mask=mod_mask,
                physics_emb=phys.detach(),
                exposures=exposures,
                rmt_loadings=rmt,
                w_prev=w_prev,
                horizon_mask=hmask_1d,
            )
        assert ref_h.shape == (n_assets, d_model) and torch.isfinite(ref_h).all()
        assert len(steps_h) == 3
        assert diag["tool_names"] == TOOL_NAMES_V2
        assert diag["tool_weights_mean"].shape == (10,)
        assert diag["tool_weights_by_step"].shape == (3, 10)
        for k in required_diag_keys:
            assert k in diag, f"Falta diagnóstico v2 requerido: {k}"
            assert isinstance(diag[k], torch.Tensor)
            assert torch.isfinite(diag[k]).all(), f"Diagnóstico no finito: {k}"
        mode_out[label] = (ref_h, diag)

    torch.testing.assert_close(
        mode_out["uniform"][1]["tool_weights_mean"],
        torch.full((10,), 0.1),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        mode_out["no_tools"][1]["tool_weights_mean"],
        torch.zeros(10),
        atol=1e-7,
        rtol=0.0,
    )
    assert not torch.allclose(mode_out["soft"][0], mode_out["placebo"][0], atol=1e-6)
    assert not torch.allclose(mode_out["soft"][0], mode_out["no_tools"][0], atol=1e-5)

    # 2) Router v2: lazo cerrado (sensible a prev_diagnostics) + annealing coseno de temperatura
    base_v2.train()
    t_init = float(base_v2.tool_router.current_temperature())
    base_v2.step_router_schedule(steps=400)
    t_mid = float(base_v2.tool_router.current_temperature())
    assert t_mid < t_init
    base_v2.eval()
    base_v2.step_router_schedule(steps=400)
    assert float(base_v2.tool_router.current_temperature()) == pytest.approx(t_mid, abs=1e-7)

    with torch.no_grad():
        h_r = torch.randn(n_assets, d_model)
        d_low = torch.zeros(n_assets, 8)
        d_hi = torch.full((n_assets, 8), 3.0)
        w_low, _ = base_v2.tool_router(ctx, 1, h_state=h_r, prev_diagnostics=d_low)
        w_hi, _ = base_v2.tool_router(ctx, 1, h_state=h_r, prev_diagnostics=d_hi)
        assert not torch.allclose(w_low, w_hi, atol=1e-5)

    # 3) Parada adaptativa easy vs hard: expected_steps_hard_q75 > expected_steps_easy_q25
    n_eh = 40
    half = n_eh // 2
    base_dir = torch.randn(n_eh, d_model)
    # Primera mitad: limpios/fáciles (modalidades alineadas, alta liq, baja vol, 0 OOD)
    # Segunda mitad: difíciles/conflicto (modalidades opuestas, baja liq, alta vol, alto OOD)
    mod_eh = torch.stack(
        [
            base_dir,
            torch.cat([base_dir[:half], -base_dir[half:]], dim=0),
            torch.randn(1, d_model).expand(n_eh, -1),
            torch.cat([base_dir[:half], -base_dir[half:]], dim=0),
            base_dir,
        ],
        dim=1,
    )
    mask_eh = torch.ones(n_eh, 5)
    ctx_eh = torch.zeros(n_eh, c_dim)
    ctx_eh[:half, 6] = 0.1  # vol_pct baja
    ctx_eh[:half, 7] = 0.95  # liq_pct alta
    ctx_eh[half:, 3] = 4.0  # wasserstein stress
    ctx_eh[half:, 6] = 0.95  # vol_pct alta
    ctx_eh[half:, 7] = 0.05  # liq_pct baja
    ood_eh = torch.cat([torch.zeros(half), torch.full((half,), 8.0)], dim=0)
    phys_eh = torch.randn(n_eh, d_model) * 0.2
    exp_eh = torch.randn(n_eh, exp_dim)
    rmt_eh = torch.randn(n_eh, rmt_dim)

    base_v2.eval()
    with torch.no_grad():
        _, _, diag_eh = base_v2(
            x0=None,
            high_snr_context=ctx_eh,
            mod_tokens=mod_eh,
            modality_mask=mask_eh,
            physics_emb=phys_eh,
            exposures=exp_eh,
            rmt_loadings=rmt_eh,
            ood_energy=ood_eh,
        )
    steps_hard = float(diag_eh["expected_steps_hard_q75"])
    steps_easy = float(diag_eh["expected_steps_easy_q25"])
    assert steps_hard > steps_easy + 0.05

    # 3b) Congelación zero-sync en eval() cuando cum_halt > 0.95 (torch.where)
    orig_bias = base_v2.halt_gate.out_head[-1].bias.detach().clone()
    with torch.no_grad():
        base_v2.halt_gate.out_head[-1].bias.fill_(12.0)
        _, steps_frz, diag_frz = base_v2(
            x0=None,
            high_snr_context=ctx_eh,
            mod_tokens=mod_eh,
            modality_mask=mask_eh,
            physics_emb=phys_eh,
            exposures=exp_eh,
            rmt_loadings=rmt_eh,
            enable_early_exit=True,
        )
        assert float(diag_frz["halting_distribution"][:, 0].min()) > 0.99
        torch.testing.assert_close(steps_frz[1], steps_frz[0], atol=0.0, rtol=0.0)
        torch.testing.assert_close(steps_frz[2], steps_frz[0], atol=0.0, rtol=0.0)
        base_v2.halt_gate.out_head[-1].bias.copy_(orig_bias)

    # 4) Equivarianza por permutación completa en v2_10tools
    perm = torch.randperm(n_assets)
    with torch.no_grad():
        ref_orig, _, d_orig = base_v2(
            x0=None,
            high_snr_context=ctx,
            mod_tokens=mod_tokens.detach(),
            modality_mask=mod_mask,
            physics_emb=phys.detach(),
            exposures=exposures,
            rmt_loadings=rmt,
            w_prev=w_prev,
        )
        ref_perm, _, d_perm = base_v2(
            x0=None,
            high_snr_context=ctx[perm],
            mod_tokens=mod_tokens.detach()[perm],
            modality_mask=mod_mask[perm],
            physics_emb=phys.detach()[perm],
            exposures=exposures[perm],
            rmt_loadings=rmt[perm],
            w_prev=w_prev[perm],
        )
    torch.testing.assert_close(ref_orig[perm], ref_perm, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        d_orig["tool_weights_mean"], d_perm["tool_weights_mean"], atol=1e-4, rtol=1e-4
    )

    # 5) Gradientes finitos en TODOS los parámetros entrenables de v2_10tools + tool_aux_loss
    base_v2.train()
    base_v2.zero_grad()
    mod_g = torch.randn(n_assets, 5, d_model, requires_grad=True)
    phys_g = torch.randn(n_assets, d_model, requires_grad=True)
    age_g = torch.rand(n_assets, 5) * 6.0
    y_g = torch.randn(n_assets, 6) * 0.02
    ref_g, steps_g, diag_g = base_v2(
        x0=None,
        high_snr_context=ctx,
        mod_tokens=mod_g,
        modality_mask=torch.ones(n_assets, 5),
        physics_emb=phys_g,
        modality_age=age_g,
        exposures=exposures,
        rmt_loadings=rmt,
        w_prev=w_prev,
        probe_targets=y_g,
    )
    loss_g = (
        ref_g.pow(2).mean()
        + sum(s.pow(2).mean() for s in steps_g)
        + diag_g["weighted_ponder_loss"]
        + diag_g["probe_aux_loss"]
        + diag_g["tool_aux_loss"]
    )
    loss_g.backward()
    for name, p in base_v2.named_parameters():
        assert p.grad is not None, f"Parámetro sin gradiente en v2_10tools: {name}"
        assert torch.isfinite(p.grad).all(), f"Gradiente no finito en v2_10tools: {name}"
    assert mod_g.grad is not None and torch.isfinite(mod_g.grad).all()
    assert phys_g.grad is not None and torch.isfinite(phys_g.grad).all()

    # 5b) Regresión BPTT=4 multi-sesión: update_router_balance_bias + schedule sin error versión
    base_v2.train()
    base_v2.zero_grad()
    st_bptt: dict[str, object] | None = None
    bptt_loss = torch.zeros(())
    for _ in range(4):
        ref_b, _, d_b = base_v2(
            x0=None,
            high_snr_context=ctx,
            mod_tokens=torch.randn(n_assets, 5, d_model),
            modality_mask=torch.ones(n_assets, 5),
            physics_emb=torch.randn(n_assets, d_model),
            exposures=exposures,
            rmt_loadings=rmt,
            state=st_bptt,
        )
        base_v2.update_router_balance_bias(d_b["tool_weights_mean"])
        base_v2.step_router_schedule(steps=1)
        st_bptt = d_b["state"]
        bptt_loss = bptt_loss + ref_b.pow(2).mean() + d_b["tool_aux_loss"]
    bptt_loss.backward()

    # 6) Edge cases v2: N=1/2/7, modalidades ausentes NaN+grad, use_context_engineer=False,
    #    compartición estricta de pesos K=4 vs K=1 (<=64 params) y penalización anti-colapso
    v2_k1 = TitansCognitiveCore(
        hidden_dim=d_model,
        context_dim=c_dim,
        core_dim=16,
        thinking_steps=1,
        toolbox_version="v2_10tools",
        use_context_engineer=False,
        rmt_dim=rmt_dim,
        n_hyperedges=6,
    )
    v2_k4 = TitansCognitiveCore(
        hidden_dim=d_model,
        context_dim=c_dim,
        core_dim=16,
        thinking_steps=4,
        toolbox_version="v2_10tools",
        use_context_engineer=False,
        rmt_dim=rmt_dim,
        n_hyperedges=6,
    )
    p_v2_k1 = sum(p.numel() for p in v2_k1.parameters())
    p_v2_k4 = sum(p.numel() for p in v2_k4.parameters())
    assert p_v2_k4 - p_v2_k1 <= 64

    v2_k4.train()
    for n_edge in (1, 2, 7):
        x_e = torch.randn(n_edge, d_model, requires_grad=True)
        c_e = torch.rand(n_edge, c_dim)
        ref_e, _, d_e = v2_k4(
            x_e,
            c_e,
            exposures=torch.randn(n_edge, exp_dim),
            rmt_loadings=torch.randn(n_edge, rmt_dim),
            cf_placebo_mode=(n_edge == 7),
        )
        assert ref_e.shape == (n_edge, d_model) and torch.isfinite(ref_e).all()
        assert d_e["state"]["cognitive_notes"].shape == (8, d_model)
        (ref_e.pow(2).mean() + d_e["tool_aux_loss"] + d_e["weighted_ponder_loss"]).backward()
        assert x_e.grad is not None and torch.isfinite(x_e.grad).all()
        v2_k4.zero_grad()

    # Todas modalidades ausentes con NaN + requires_grad=True -> salida y gradientes finitos
    base_v2.train()
    base_v2.zero_grad()
    mod_nan = torch.full((8, 5, d_model), float("nan"), requires_grad=True)
    mask_zero = torch.zeros(8, 5)
    phys_z = torch.randn(8, d_model, requires_grad=True)
    ref_z, _, d_z = base_v2(
        None,
        torch.rand(8, c_dim),
        mod_tokens=mod_nan,
        modality_mask=mask_zero,
        physics_emb=phys_z,
    )
    assert torch.isfinite(ref_z).all()
    (ref_z.pow(2).mean() + d_z["tool_aux_loss"]).backward()
    assert mod_nan.grad is not None and torch.isfinite(mod_nan.grad).all()
    assert phys_z.grad is not None and torch.isfinite(phys_z.grad).all()

    # Sesgo colapsado dispara tool_balance_loss >> balanceado
    with torch.no_grad():
        saved_bias = base_v2.tool_router.balance_bias.clone()
        base_v2.tool_router.balance_bias.zero_()
        _, _, d_bal = base_v2(
            None,
            ctx,
            mod_tokens=mod_tokens.detach(),
            modality_mask=mod_mask,
            physics_emb=phys.detach(),
        )
        base_v2.tool_router.balance_bias.fill_(-8.0)
        base_v2.tool_router.balance_bias[0] = 8.0
        _, _, d_col = base_v2(
            None,
            ctx,
            mod_tokens=mod_tokens.detach(),
            modality_mask=mod_mask,
            physics_emb=phys.detach(),
        )
        base_v2.tool_router.balance_bias.copy_(saved_bias)
        assert float(d_col["tool_balance_loss"]) > float(d_bal["tool_balance_loss"]) + 1.0


def test_v2_10tools_zero_host_syncs_audit_n2600_bf16_and_torch_compile_dynamic(monkeypatch):
    """audit_zero_host_syncs==0, N=2600 BF16 con Tool 4 FISTA fast path y torch.compile dynamic."""
    import mars_titan.evaluation.metacognitive_agent as ma
    from mars_titan.models.mmarrs.blackwell_kernels import audit_zero_host_syncs

    # 1) Auditoría oficial `audit_zero_host_syncs == 0` en cognitive_core.py y counterfactual_sim.py
    mmarrs_dir = Path(__file__).resolve().parents[2] / "src" / "mars_titan" / "models" / "mmarrs"
    audit_res = audit_zero_host_syncs(
        [mmarrs_dir / "cognitive_core.py", mmarrs_dir / "counterfactual_sim.py"]
    )
    assert audit_res["total_findings"] == 0, f"Host syncs detectados: {audit_res['findings']}"
    assert audit_res["clean"] is True

    # 2) N=2600 BF16 AMP verificando ruta rápida _diff_portfolio_fista_fwd_op en Tool 4
    torch.manual_seed(905)
    n_big, d_model, c_dim, exp_dim, rmt_dim = 2600, 32, 10, 9, 5
    core_big = TitansCognitiveCore(
        hidden_dim=d_model,
        context_dim=c_dim,
        core_dim=16,
        thinking_steps=3,
        tool_mode="soft",
        toolbox_version="v2_10tools",
        rmt_dim=rmt_dim,
        n_hyperedges=8,
        use_context_engineer=True,
    ).to(dtype=torch.bfloat16)

    fista_hits = {"n": 0}
    orig_fista = ma._diff_portfolio_fista_fwd_op

    def _count_fista(*args, **kwargs):
        fista_hits["n"] += 1
        return orig_fista(*args, **kwargs)

    monkeypatch.setattr(ma, "_diff_portfolio_fista_fwd_op", _count_fista)

    mod_big = torch.randn(n_big, 5, d_model, dtype=torch.bfloat16)
    mask_big = torch.ones(n_big, 5, dtype=torch.bfloat16)
    mask_big[:, 1] = 0.0
    phys_big = torch.randn(n_big, d_model, dtype=torch.bfloat16)
    ctx_big = torch.rand(n_big, c_dim, dtype=torch.bfloat16)
    exp_big = torch.randn(n_big, exp_dim, dtype=torch.bfloat16)
    rmt_big = torch.randn(n_big, rmt_dim, dtype=torch.bfloat16)
    w_prev_big = torch.zeros(n_big, dtype=torch.bfloat16)

    core_big.train()
    ref_big, _, diag_big = core_big(
        x0=None,
        high_snr_context=ctx_big,
        mod_tokens=mod_big,
        modality_mask=mask_big,
        physics_emb=phys_big,
        exposures=exp_big,
        rmt_loadings=rmt_big,
        w_prev=w_prev_big,
    )
    assert fista_hits["n"] == 3, f"Esperadas 3 llamadas FISTA, recibidas {fista_hits['n']}"
    assert ref_big.dtype == torch.bfloat16 and torch.isfinite(ref_big).all()
    (
        ref_big.float().pow(2).mean()
        + diag_big["weighted_ponder_loss"].float()
        + diag_big["tool_aux_loss"].float()
    ).backward()

    # 3) torch.compile(mode="default", dynamic=True) limpio en dos tamaños dinámicos N=32 y N=48
    monkeypatch.setattr(ma, "_diff_portfolio_fista_fwd_op", orig_fista)
    core_c = TitansCognitiveCore(
        hidden_dim=24,
        context_dim=10,
        core_dim=16,
        thinking_steps=3,
        tool_mode="soft",
        toolbox_version="v2_10tools",
        rmt_dim=4,
        n_hyperedges=6,
        use_context_engineer=True,
    ).eval()
    compiled = torch.compile(core_c, mode="default", dynamic=True)
    with torch.no_grad():
        for n_dyn in (32, 48):
            m_d = torch.randn(n_dyn, 5, 24)
            mk_d = torch.ones(n_dyn, 5)
            p_d = torch.randn(n_dyn, 24)
            c_d = torch.rand(n_dyn, 10)
            e_d = torch.randn(n_dyn, 4)
            r_d = torch.randn(n_dyn, 4)
            out_eager, _, _ = core_c(
                None,
                c_d,
                mod_tokens=m_d,
                modality_mask=mk_d,
                physics_emb=p_d,
                exposures=e_d,
                rmt_loadings=r_d,
            )
            out_comp, _, diag_comp = compiled(
                None,
                c_d,
                mod_tokens=m_d,
                modality_mask=mk_d,
                physics_emb=p_d,
                exposures=e_d,
                rmt_loadings=r_d,
            )
            assert out_comp.shape == (n_dyn, 24) and torch.isfinite(out_comp).all()
            assert torch.isfinite(diag_comp["expected_steps_mean"]).all()
            torch.testing.assert_close(out_eager, out_comp, atol=1e-4, rtol=1e-4)
