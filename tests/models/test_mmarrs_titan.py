"""Pruebas TDD integrales de la arquitectura de frontera `MMARRS-TITAN` y sus 14 variaciones.

Verifica de forma estricta:
  1. Contratos inmutables `MMARRSConfig`, huella SHA-256 y las 14 variaciones + 4 controles.
  2. `NeuromodulatedTitansMemory`: caja contractiva PT1 (Lyapunov), ACh vs NE, Huber y cascada 3x.
  3. `RMTSpilloverHypergraph` (equivarianza por permutación O(N*K)), `HRRModernHopfieldMemory`
     (convolución circular FFT + energía OOD mayor en patrones inéditos) y `HolographicBrain3L`
     (3 niveles: cortical + hipergrafo + tálamo FiLM, identidad inicial y seguridad de máscaras).
  4. `HighSNRMoELayer`: Shared + Routed Soft/Sparsemax, balanceo Aux-Loss-Free sin colapso,
     control uniforme `1/E` (v04) y placebo de contexto barajado (v05).
  5. `FinancialEnergyVerifier` + `PCTRMRecursiveLoop`: energía estructural 4 términos, contracción
     Lipschitz < 1, parada temprana monótona `Early Exit`, routers distintos por bucle `R^(k)` y
     re-inyección de entrada `x_0`.
  6. `MultiHorizonGranularHead` + `MMARRSMultiTaskLoss`: 9 cuantiles estrictamente crecientes en
     todos los horizontes, Triple-Barrier válido, Deep Supervision, CRPS, IC neutralizado,
     Sharpe-Turnover diferenciable (`P+`) y decorrelación bi-hemisférica con gradientes finitos.
  7. `MMARRSTitan` extremo a extremo: determinismo FP32, causalidad temporal estricta, macro-bucle
     causal `step_matured` y ejecución del arnés `ablation_runner` sobre las 14 variaciones.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mars_titan.models.mmarrs import (
    BLACKWELL_SPECS,
    MMARRS_VARIATIONS,
    VARIATION_REGISTRY,
    AsyncSessionPrefetcher,
    BlackwellRuntimeConfig,
    FinancialEnergyVerifier,
    FullHistorySplitConfig,
    HighSNRMoELayer,
    HolographicBrain3L,
    HRRModernHopfieldMemory,
    MMARRSConfig,
    MMARRSMultiTaskLoss,
    MultiHorizonGranularHead,
    NeuromodulatedTitansMemory,
    PCTRMRecursiveLoop,
    RMTSpilloverHypergraph,
    blackwell_mps_launch_plan,
    build_optimizer,
    build_variation,
    configure_blackwell_runtime,
    fused_pt1_huber_gate_update,
    run_variation_ablation,
    split_chronological_sessions,
    train_full_history_variation,
    variation_config,
)


def _synthetic_session_batch(
    rng: np.random.Generator,
    n_assets: int = 24,
    *,
    prices_dim: int = 16,
    fundamentals_dim: int = 8,
    macro_dim: int = 6,
    news_dim: int = 8,
    charts_dim: int = 8,
    physics_dim: int = 12,
    context_dim: int = 10,
    exposures_dim: int = 4,
    rmt_k: int = 4,
) -> dict[str, torch.Tensor]:
    def _t(shape):
        return torch.tensor(rng.standard_normal(shape), dtype=torch.float32)

    mask = torch.ones((n_assets, 5), dtype=torch.float32)
    mask[0::5, 3] = 0.0  # noticias parcialmente ausentes
    mask[1::7, 1] = 0.0  # fundamentales parcialmente ausentes
    return dict(
        prices=_t((n_assets, prices_dim)),
        fundamentals=_t((n_assets, fundamentals_dim)),
        macro=_t((n_assets, macro_dim)),
        news=_t((n_assets, news_dim)),
        charts=_t((n_assets, charts_dim)),
        modality_mask=mask,
        physics_v4=_t((n_assets, physics_dim)),
        high_snr_context=_t((n_assets, context_dim)),
        exposures=_t((n_assets, exposures_dim)),
        rmt_loadings=_t((n_assets, rmt_k)),
        regime_vec=_t((context_dim,)),
        ach_signal=torch.abs(_t((n_assets, 1))) * 0.5,
        ne_signal=torch.tensor([0.2], dtype=torch.float32),
    )


def test_mmarrs_config_registry_14_variations_and_sha256_identity():
    """Las 16 variaciones v1 + todas las v2 existen con SHA-256 único e iso-parámetros."""
    assert len(MMARRS_VARIATIONS) == 16
    assert set(MMARRS_VARIATIONS).issubset(set(VARIATION_REGISTRY.keys()))

    fps = set()
    audits = {}
    for name in MMARRS_VARIATIONS:
        cfg = variation_config(
            name,
            prices_dim=16,
            fundamentals_dim=8,
            macro_dim=6,
            news_dim=8,
            charts_dim=8,
            physics_dim=12,
            context_dim=10,
            hidden_dim=32,
        )
        assert isinstance(cfg, MMARRSConfig)
        assert cfg.max_loops <= 3
        fp = cfg.fingerprint()
        assert len(fp) == 64
        fps.add(fp)
        model = build_variation(
            name,
            prices_dim=16,
            fundamentals_dim=8,
            macro_dim=6,
            news_dim=8,
            charts_dim=8,
            physics_dim=12,
            context_dim=10,
            hidden_dim=32,
        )
        audits[name] = model.parameter_audit()

    # Todas las variaciones v1 tienen identidad SHA-256 distinta (y todas las del registro también)
    assert len(fps) == 16
    all_fps = {
        variation_config(
            name,
            prices_dim=16,
            fundamentals_dim=8,
            macro_dim=6,
            news_dim=8,
            charts_dim=8,
            physics_dim=12,
            context_dim=10,
            hidden_dim=32,
        ).fingerprint()
        for name in VARIATION_REGISTRY
    }
    assert len(all_fps) == len(VARIATION_REGISTRY)

    # Verificar controles emparejados
    c_moe = variation_config("v03_moe_only_k1")
    c_uni = variation_config("v04_moe_uniform_control")
    c_shf = variation_config("v05_moe_shuffled_ctx_placebo")
    assert c_moe.use_moe and c_moe.moe_mode == "soft" and c_moe.max_loops == 1
    assert c_uni.use_moe and c_uni.moe_mode == "uniform" and c_uni.max_loops == 1
    assert c_shf.use_moe and c_shf.moe_mode == "shuffled_context" and c_shf.max_loops == 1

    # Control denso ancho (v01) debe tener tamaño comparable al MoE (v03) y mayor que v00
    assert audits["v01_dense_wide_iso_param"]["total_params"] > audits["v00_dense_baseline"][
        "total_params"
    ]
    ratio_iso = (
        audits["v01_dense_wide_iso_param"]["total_params"]
        / audits["v03_moe_only_k1"]["total_params"]
    )
    assert 0.70 <= ratio_iso <= 1.45

    with pytest.raises(ValueError):
        variation_config("variacion_inexistente")


def test_neuromodulated_titans_memory_pt1_bounds_huber_and_ach_ne_asymmetry():
    """PT1 garantiza caja contractiva, Huber resiste outliers 100x y ACh/NE modulan distinto."""
    torch.manual_seed(7)
    mem = NeuromodulatedTitansMemory(
        hidden_dim=24,
        pt1_alpha_lo=0.05,
        pt1_eta_hi=0.95,
        pt1_theta_max=0.25,
        huber_delta=1.0,
    )
    n_assets = 12
    state = mem.init_state(n_assets, device=torch.device("cpu"), dtype=torch.float32)
    x = torch.randn(n_assets, 24)

    # 1) Comparar ACh bajo (señal limpia) vs ACh alto (ruido esperado)
    s_low_ach, d_low = mem.update(
        x, state, ach_signal=torch.zeros(n_assets, 1), ne_signal=torch.zeros(1)
    )
    s_hi_ach, d_hi = mem.update(
        x, state, ach_signal=torch.full((n_assets, 1), 6.0), ne_signal=torch.zeros(1)
    )
    assert d_low["alpha_min"] >= 0.05 - 1e-6
    assert d_low["eta_max"] <= 0.95 + 1e-6
    assert d_low["theta_max"] <= 0.25 + 1e-6
    # ACh alto amortigua la escritura theta -> menor cambio en memoria rápida
    assert d_hi["theta_mean"] < d_low["theta_mean"] * 0.75
    assert s_hi_ach["fast"].norm() < s_low_ach["fast"].norm()

    # 2) NE alto (shock inesperado de régimen) acelera el olvido alpha respecto a NE=0
    s_base = s_low_ach
    _, d_calm = mem.update(
        x, s_base, ach_signal=torch.zeros(n_assets, 1), ne_signal=torch.tensor([-3.0])
    )
    _, d_shock = mem.update(
        x, s_base, ach_signal=torch.zeros(n_assets, 1), ne_signal=torch.tensor([5.0])
    )
    assert d_shock["alpha_mean"] > d_calm["alpha_mean"] + 0.05

    # 3) Estabilidad ante outliers extremos (100x) gracias a Huber + PT1 + cascada 3 escalas
    s_cur = state
    for step in range(15):
        spike = torch.randn(n_assets, 24) * (100.0 if step % 3 == 0 else 1.0)
        s_cur, _ = mem.update(
            spike, s_cur, ach_signal=torch.ones(n_assets, 1), ne_signal=torch.tensor([1.0])
        )
    for bank in ("fast", "medium", "slow"):
        assert torch.isfinite(s_cur[bank]).all()
        assert float(s_cur[bank].detach().abs().max()) < 20.0


def test_hypergraph_equivariance_hrr_hopfield_ood_and_holographic_3l():
    """Hipergrafo equivariante O(N*K), HRR-Hopfield detecta OOD y Cerebro Holográfico 3L."""
    torch.manual_seed(11)
    n_assets, d_model, k_edges = 20, 24, 6

    # 1) Equivarianza por permutación en RMTSpilloverHypergraph
    hg = RMTSpilloverHypergraph(hidden_dim=d_model, n_hyperedges=k_edges, rmt_dim=4)
    with torch.no_grad():
        hg.out_scale.fill_(0.5)
    x = torch.randn(n_assets, d_model)
    rmt = torch.randn(n_assets, 4)
    y1, diag1 = hg(x, rmt_loadings=rmt)
    perm = torch.randperm(n_assets)
    y2, _ = hg(x[perm], rmt_loadings=rmt[perm])
    torch.testing.assert_close(y1[perm], y2, atol=1e-5, rtol=1e-5)
    assert diag1["incidence_entropy"] > 0.0

    # 2) HRR-Modern-Hopfield: FFT circular binding + energía OOD mayor en consultas inéditas
    hrr = HRRModernHopfieldMemory(hidden_dim=d_model, max_slots=16, beta=6.0)
    a = torch.randn(4, d_model)
    b = torch.randn(4, d_model)
    bound = hrr.circular_bind(a, b)
    assert bound.shape == (4, d_model)
    assert torch.isfinite(bound).all()

    # Guardar 6 patrones conocidos en memoria episódica
    reg_known = torch.randn(d_model)
    sec_known = torch.randn(6, d_model)
    q_known = torch.randn(6, d_model)
    v_known = torch.randn(6, d_model)
    hrr.store(q_known, v_known, regime_key=reg_known, sector_key=sec_known)

    res_in = hrr.retrieve(q_known, regime_key=reg_known, sector_key=sec_known)
    # Consulta totalmente ortogonal / fuera de distribución
    q_ood = -q_known * 1.5 + torch.randn_like(q_known) * 2.0
    res_ood = hrr.retrieve(q_ood, regime_key=-reg_known, sector_key=-sec_known)
    assert float(res_ood["ood_energy"].mean()) > float(res_in["ood_energy"].mean()) + 0.15

    # 3) Cerebro Holográfico de 3 Niveles (identidad inicial + máscaras seguras + FiLM)
    hb3 = HolographicBrain3L(
        hidden_dim=d_model,
        n_modalities=5,
        n_cortical_areas=12,
        n_hidden_slots=4,
        n_hyperedges=k_edges,
        regime_dim=10,
        rmt_dim=4,
    )
    mod_tokens = torch.randn(n_assets, 5, d_model)
    mod_mask = torch.ones(n_assets, 5)
    mod_mask[:, 4] = 0.0  # modalidad 4 completamente ausente
    regime = torch.randn(10)
    out0, diag0 = hb3(mod_tokens, mod_mask, regime_vec=regime, rmt_loadings=rmt)
    # single_zero_v1: readback pequeño (std=0.01) -> casi identidad al nacer y gradiente paso-0 vivo
    torch.testing.assert_close(out0[:, :4, :], mod_tokens[:, :4, :], atol=0.15, rtol=0.15)
    torch.testing.assert_close(out0[:, 4, :], torch.zeros(n_assets, d_model), atol=1e-6, rtol=1e-6)
    assert diag0["levels"] == 3


def test_high_snr_moe_aux_free_balancing_uniform_control_and_shuffled_placebo():
    """MoE equilibra carga sin aux-loss, uniforme promedia 1/E y placebo baraja el contexto."""
    torch.manual_seed(23)
    n_assets, d_model, c_dim = 64, 24, 10
    x = torch.randn(n_assets, d_model)
    ctx = torch.randn(n_assets, c_dim)

    # 1) Soft MoE + balanceo Aux-Loss-Free
    moe = HighSNRMoELayer(
        hidden_dim=d_model,
        context_dim=c_dim,
        n_shared_experts=2,
        n_routed_experts=6,
        mode="soft",
    )
    # Sesgar deliberadamente al experto 0 para verificar que el balanceo aux-free lo corrige
    with torch.no_grad():
        moe.router.bias.zero_()
        moe.router.bias[0] = 3.0
    _, diag_before = moe(x, ctx)
    assert diag_before["load_max_min_ratio"] > 2.5
    for _ in range(40):
        _, d_step = moe(x, ctx)
        moe.update_balance_bias(d_step["expert_loads"], lr=0.25)
    _, diag_after = moe(x, ctx)
    assert diag_after["load_max_min_ratio"] < 2.0
    assert diag_after["router_entropy"] > 0.5

    # 2) Control uniforme (v04): todos los pesos exactamente 1/E
    moe_u = HighSNRMoELayer(
        hidden_dim=d_model,
        context_dim=c_dim,
        n_shared_experts=2,
        n_routed_experts=6,
        mode="uniform",
    )
    _, diag_u = moe_u(x, ctx)
    torch.testing.assert_close(
        diag_u["router_probs"],
        torch.full((n_assets, 6), 1.0 / 6.0),
        atol=1e-6,
        rtol=1e-6,
    )

    # 3) Control placebo barajado (v05): permuta el contexto respecto al modo soft con mismos pesos
    moe_s = HighSNRMoELayer(
        hidden_dim=d_model,
        context_dim=c_dim,
        n_shared_experts=2,
        n_routed_experts=6,
        mode="shuffled_context",
    )
    moe_s.load_state_dict(moe.state_dict())
    _, d_real = moe(x, ctx)
    _, d_shuf = moe_s(x, ctx)
    assert not torch.allclose(d_real["router_probs"], d_shuf["router_probs"], atol=1e-3)


def test_pc_trm_recursive_loop_energy_verifier_contraction_and_early_exit():
    """El doble bucle verifica 4 energías, es contractivo (Lipschitz<1) y aplica Early Exit."""
    torch.manual_seed(31)
    n_assets, d_model, c_dim = 28, 24, 10
    horizons = (1, 2, 5, 10, 21, 63)
    head = MultiHorizonGranularHead(hidden_dim=d_model, horizons=horizons, exposures_dim=4)
    verifier = FinancialEnergyVerifier(horizons=horizons)
    loop = PCTRMRecursiveLoop(
        hidden_dim=d_model,
        context_dim=c_dim,
        max_loops=4,
        use_moe=True,
        n_shared_experts=2,
        n_routed_experts=4,
        moe_mode="soft",
        lipschitz_bound=0.88,
        early_exit_tol=1e-4,
    )

    x0 = torch.randn(n_assets, d_model)
    ctx = torch.randn(n_assets, c_dim)
    exposures = torch.randn(n_assets, 4)
    ood_energy = torch.abs(torch.randn(n_assets)) * 0.5

    # Modo train: devuelve los 4 pasos para Deep Supervision + gradientes finitos
    loop.train()
    out_tr = loop(
        x0,
        ctx,
        head=head,
        verifier=verifier,
        exposures=exposures,
        ood_energy=ood_energy,
        enable_early_exit=False,
    )
    assert len(out_tr["step_outputs"]) == 4
    assert out_tr["diagnostics"]["max_spectral_norm"] <= 0.90 + 1e-3
    # Verificar que los routers por iteración son módulos independientes (LOOM)
    assert len(loop.step_blocks) == 4
    assert loop.step_blocks[0] is not loop.step_blocks[1]

    # Modo eval con Early Exit: energía final <= energía inicial (aceptación monótona)
    loop.eval()
    with torch.no_grad():
        out_ev = loop(
            x0,
            ctx,
            head=head,
            verifier=verifier,
            exposures=exposures,
            ood_energy=ood_energy,
            enable_early_exit=True,
        )
    e_traj = out_ev["diagnostics"]["energy_trajectory"]
    assert e_traj[-1] <= e_traj[0] + 1e-6
    assert 1 <= out_ev["diagnostics"]["k_used"] <= 4
    assert out_ev["energy"]["e_horizon"].item() >= 0.0
    assert out_ev["energy"]["e_cdf_vol"].item() >= 0.0
    assert out_ev["energy"]["e_cross_section"].item() >= 0.0
    assert out_ev["energy"]["e_ood"].item() >= 0.0


def test_granular_heads_non_crossing_quantiles_and_multitask_loss_gradients():
    """9 cuantiles estrictamente crecientes, Triple-Barrier válido y pérdida multi-tarea finita."""
    torch.manual_seed(42)
    n_assets, d_model = 32, 24
    horizons = (1, 2, 5, 10, 21, 63)
    head = MultiHorizonGranularHead(hidden_dim=d_model, horizons=horizons, exposures_dim=4)
    loss_fn = MMARRSMultiTaskLoss(horizons=horizons)

    h = torch.randn(n_assets, d_model, requires_grad=True)
    b_fac = torch.randn(n_assets, 4)
    pred = head(h, exposures=b_fac)

    # 9 cuantiles estrictamente no cruzados en TODOS los activos y horizontes
    q = pred["quantiles"]
    assert q.shape == (n_assets, len(horizons), 9)
    assert torch.all(q[..., 1:] > q[..., :-1])
    # Triple-Barrier probs suman 1 y turning_prob en (0, 1)
    torch.testing.assert_close(
        pred["tb_probs"].sum(dim=-1), torch.ones(n_assets, len(horizons)), atol=1e-5, rtol=1e-5
    )
    assert torch.all((pred["turning_prob"] > 0.0) & (pred["turning_prob"] < 1.0))
    assert torch.all(pred["vol_pred"] > 0.0)
    assert torch.all(pred["epistemic_error"] >= 0.0)

    # Targets multi-horizonte + companion frozen (decorrelación bi-hemisférica)
    y_ret = torch.randn(n_assets, len(horizons)) * 0.015
    y_tb = torch.randint(0, 3, (n_assets, len(horizons)))
    y_turn = (torch.rand(n_assets, len(horizons)) > 0.7).float()
    y_vol = torch.full((n_assets, len(horizons)), 0.02)
    companion_pred = torch.randn(n_assets) * 0.01

    loss_dict = loss_fn(
        step_outputs=[pred, pred],
        targets=dict(returns=y_ret, tb_class=y_tb, turning=y_turn, vol=y_vol),
        exposures=b_fac,
        energy_total=torch.tensor(0.05),
        companion_pred=companion_pred,
    )
    assert torch.isfinite(loss_dict["total"])
    loss_dict["total"].backward()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    assert float(h.grad.abs().sum()) > 0.0


def test_mmarrs_titan_full_determinism_causality_macro_loop_and_ablation_runner():
    """MMARRS-TITAN completo: determinismo FP32, causalidad, macro-bucle y arnés de 14 variantes."""
    rng = np.random.default_rng(77)
    batch_t0 = _synthetic_session_batch(rng, n_assets=20)
    batch_t1 = _synthetic_session_batch(rng, n_assets=20)

    # 1) Determinismo FP32 exacto en v13_mmarrs_titan_full
    torch.manual_seed(101)
    m1 = build_variation(
        "v13_mmarrs_titan_full",
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        hidden_dim=24,
    )
    torch.manual_seed(101)
    m2 = build_variation(
        "v13_mmarrs_titan_full",
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        hidden_dim=24,
    )
    m1.eval()
    m2.eval()
    with torch.no_grad():
        o1 = m1.forward_session(batch_t0)
        o2 = m2.forward_session(batch_t0)
    torch.testing.assert_close(o1["pred_return"], o2["pred_return"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(o1["quantiles"], o2["quantiles"], atol=0.0, rtol=0.0)

    # 2) Macro-bucle externo causal `step_matured`: actualiza memoria Titans, HRR, Pi_l y sesgo MoE
    st0 = o1["state"]
    y_mat = torch.tensor(rng.standard_normal(20) * 0.015, dtype=torch.float32)
    st1 = m1.step_matured(
        st0,
        matured_batch=batch_t0,
        matured_targets=y_mat,
        matured_preds=o1["pred_return"][:, 0].detach(),
    )
    assert st1["matured_steps"] == 1
    assert st1["hrr_slots_used"] > 0
    # Tras madurar t0, la predicción en t1 cambia respecto a estado virgen (memoria causal activa!)
    with torch.no_grad():
        o_t1_virgin = m1.forward_session(batch_t1, state=None)
        o_t1_mem = m1.forward_session(batch_t1, state=st1)
    assert not torch.allclose(o_t1_virgin["pred_return"], o_t1_mem["pred_return"], atol=1e-6)

    # 3) Arnés de ablación `run_variation_ablation` sobre un subconjunto representativo + full
    # (incluyendo los 4 controles: v00, v01, v03, v04, v05, v06, v07, v08, v11, v12, v13)
    sessions = []
    for _ in range(18):
        b = _synthetic_session_batch(rng, n_assets=16)
        # Señal real débil embebida en precios[:, 0] + physics_v4[:, 0]
        y1 = (
            0.008 * b["prices"][:, 0]
            + 0.006 * b["physics_v4"][:, 0]
            + 0.012 * torch.randn(16)
        )
        b["target_returns"] = torch.stack(
            [y1 * np.sqrt(h) + 0.005 * torch.randn(16) for h in (1, 2, 5, 10, 21, 63)], dim=-1
        )
        sessions.append(b)

    subset = (
        "v00_dense_baseline",
        "v03_moe_only_k1",
        "v04_moe_uniform_control",
        "v05_moe_shuffled_ctx_placebo",
        "v07_moe_double_loop",
        "v11_moe_plus_agent_core",
        "v13_mmarrs_titan_full",
    )
    report = run_variation_ablation(
        sessions,
        variations=subset,
        train_end=10,
        epochs=2,
        lr=3e-3,
        hidden_dim=16,
        seed=42,
        purge_sessions=1,
        embargo_sessions=1,
        allow_short_synthetic=True,
    )
    assert set(report["results"].keys()) == set(subset)
    for name in subset:
        entry = report["results"][name]
        assert "granular_report" in entry
        assert "parameter_audit" in entry
        assert "diagnostics" in entry
        assert np.isfinite(entry["granular_report"]["dim2_ranking"]["rank_ic_mean"])
        assert np.isfinite(entry["granular_report"]["dim6_economics"]["net_sharpe_1x"])
    assert "control_comparisons" in report


def test_blackwell_sm120_runtime_optimizer_fused_kernel_and_mps_plan():
    """Perfil RTX PRO 6000 Blackwell (sm_120): precisión FP32, router LR 3x menor, kernel y MPS."""
    assert BLACKWELL_SPECS["arch"] == "sm_120"
    assert BLACKWELL_SPECS["sms"] == 188
    assert BLACKWELL_SPECS["vram_gb"] == 96.0
    assert BLACKWELL_SPECS["l2_cache_mb"] == 128

    rt_cfg = BlackwellRuntimeConfig(precision_mode="fp32_strict", router_lr_scale=1.0 / 3.0)
    info = configure_blackwell_runtime(rt_cfg, device="cpu")
    assert info["device"] == "cpu"
    assert info["amp_dtype"] == "float32"

    # Verificar que el optimizador asigna LR 3x menor a los routers MoE
    model = build_variation("v07_moe_double_loop", hidden_dim=16)
    opt = build_optimizer(model, lr=3e-3, weight_decay=1e-4, cfg=rt_cfg)
    lrs = sorted({round(g["lr"], 6) for g in opt.param_groups})
    assert len(lrs) == 2
    assert lrs[0] == pytest.approx(1e-3, rel=1e-4)
    assert lrs[1] == pytest.approx(3e-3, rel=1e-4)

    # Paridad exacta del kernel fusionado PT1 + Huber
    torch.manual_seed(5)
    s = torch.randn(16, 24)
    m = torch.randn(16, 24)
    e = torch.randn(16, 24) * 5.0
    a = torch.full((16, 24), 0.15)
    et = torch.full((16, 24), 0.80)
    th = torch.full((16, 24), 0.10)
    s_new, m_new = fused_pt1_huber_gate_update(
        s, m, e, a, et, th, huber_delta=1.0, use_triton=False
    )
    m_ref = et * m + th * torch.clamp(e, -1.0, 1.0)
    s_ref = (1.0 - a) * s + m_ref
    torch.testing.assert_close(s_new, s_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(m_new, m_ref, atol=0.0, rtol=0.0)

    # AsyncSessionPrefetcher en CPU conserva exactitud de tensores
    batches = [dict(x=torch.ones(4, 3) * i) for i in range(3)]
    fetched = list(AsyncSessionPrefetcher(batches, torch.device("cpu")))
    assert len(fetched) == 3
    torch.testing.assert_close(fetched[2]["x"], torch.full((4, 3), 2.0))

    # Plan CUDA MPS para las 16 variaciones en 1x RTX PRO 6000 (96 GB)
    plan = blackwell_mps_launch_plan(MMARRS_VARIATIONS, vram_gb=96.0, workers=6)
    assert plan["n_variations"] == 16
    assert plan["n_waves"] == 3
    assert plan["vram_gb_per_worker"] <= 16.0
    assert plan["env"]["TORCH_CUDA_ARCH_LIST"] == "12.0"


def test_full_history_2000_2023_split_seals_2024_and_trains_with_ema_checkpoint(tmp_path):
    """El entrenador 2000-2023 aplica purga+embargo, sella 2024+ y guarda checkpoint EMA."""
    rng = np.random.default_rng(123)
    # 20 sesiones repartidas en 2020..2024 (4 por año)
    years = [y for y in (2020, 2021, 2022, 2023, 2024) for _ in range(4)]
    sessions = []
    for _ in years:
        b = _synthetic_session_batch(rng, n_assets=14)
        y1 = 0.007 * b["prices"][:, 0] + 0.012 * torch.randn(14)
        b["target_returns"] = torch.stack([y1 for _ in range(6)], dim=-1)
        sessions.append(b)

    cfg = FullHistorySplitConfig(
        mode="full_history_final",
        train_start_year=2020,
        train_end_year=2022,
        val_year=2023,
        sealed_test_start_year=2024,
        purge_sessions=2,
        embargo_sessions=1,
        allow_sealed_test=False,
        allow_short_synthetic=True,
    )
    parts = split_chronological_sessions(sessions, years, cfg)
    assert len(parts["train"]) == 10  # 12 de 2020-2022 menos 2 de purga
    assert len(parts["val"]) == 3  # 4 de 2023 menos 1 de embargo
    # 2024+ retenido estrictamente sellado
    assert len(parts["sealed_test"]) == 0
    assert parts["n_sealed_withheld"] == 4

    res = train_full_history_variation(
        sessions,
        years,
        variation="v13_mmarrs_titan_full",
        split_cfg=cfg,
        hidden_dim=16,
        epochs=2,
        lr=2e-3,
        device="cpu",
        checkpoint_dir=tmp_path,
    )
    assert res["split_summary"]["n_sealed_withheld"] == 4
    assert not res["split_summary"]["sealed_test_accessible"]
    assert res["checkpoint_path"] is not None
    ckpt = torch.load(res["checkpoint_path"], weights_only=False)
    assert ckpt["variation"] == "v13_mmarrs_titan_full"
    assert "ema_state_dict" in ckpt
    assert np.isfinite(res["val_report"]["dim2_ranking"]["rank_ic_mean"])


def test_p0_1_ragged_panel_asset_ids_persistence_and_ipo_zero_init():
    """P0-1: paneles irregulares N_t != N_{t-1} preservan memoria por asset_id e IPOs en 0."""
    torch.manual_seed(101)
    rng = np.random.default_rng(101)
    mem = NeuromodulatedTitansMemory(hidden_dim=16)

    ids0 = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.long)
    x0 = torch.randn(6, 16)
    st0, _ = mem.update(
        x0,
        None,
        ach_signal=torch.ones(6, 1) * 0.2,
        ne_signal=torch.tensor([0.1]),
        target_hint=torch.randn(6, 16) * 0.5,
        asset_ids=ids0,
    )
    assert "asset_ids" in st0
    assert torch.equal(st0["asset_ids"], ids0)
    # Guardar el banco del activo 30 y del activo 50 antes de cambiar N_t
    idx30 = int((st0["asset_ids"] == 30).nonzero(as_tuple=False).item())
    idx50 = int((st0["asset_ids"] == 50).nonzero(as_tuple=False).item())
    fast30_before = st0["fast"][idx30].detach().clone()
    slow50_before = st0["slow"][idx50].detach().clone()
    assert float(fast30_before.norm()) > 1e-4

    # Sesión 1 irregular (N=8 != 6): permutada, salen 20 y 50, entran IPOs 70, 80, 90, 100
    ids1 = torch.tensor([90, 30, 70, 10, 100, 60, 80, 40], dtype=torch.long)
    x1 = torch.zeros(8, 16)
    out1, _ = mem.read(x1, st0, asset_ids=ids1)
    # IPOs (posiciones 0, 2, 4, 6) tienen memoria exactamente 0 -> read(0) == out_proj(read_norm(0))
    z_ref = mem.out_proj(mem.read_norm(torch.zeros(1, 16))).squeeze(0)
    for ipo_pos in (0, 2, 4, 6):
        torch.testing.assert_close(out1[ipo_pos], z_ref, atol=1e-6, rtol=1e-6)
    # Activo superviviente 30 (posición 1) SÍ recibe su memoria persistente (no borrada!)
    assert not torch.allclose(out1[1], z_ref, atol=1e-5)

    # Update en sesión 1: el activo pausado 50 NO se pierde del banco lento Benna-Fusi
    st1, _ = mem.update(
        torch.randn(8, 16),
        st0,
        ach_signal=torch.ones(8, 1) * 0.2,
        ne_signal=torch.tensor([0.1]),
        asset_ids=ids1,
    )
    idx50_after = int((st1["asset_ids"] == 50).nonzero(as_tuple=False).item())
    torch.testing.assert_close(st1["slow"][idx50_after], slow50_before, atol=1e-6, rtol=1e-6)

    # End-to-end en MMARRSTitan con paneles irregulares N=12 -> N=15
    model = build_variation(
        "v02_titans_neuromod_only",
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=16,
    )
    model.eval()
    b0 = _synthetic_session_batch(rng, n_assets=12)
    b0["asset_ids"] = torch.arange(100, 112, dtype=torch.long)
    b1 = _synthetic_session_batch(rng, n_assets=15)
    b1["asset_ids"] = torch.tensor(
        [105, 102, 200, 100, 201, 111, 103, 202, 108, 101, 203, 109, 104, 107, 110],
        dtype=torch.long,
    )
    with torch.no_grad():
        o0 = model.forward_session(b0, state=None)
        st = model.step_matured(
            o0["state"],
            matured_batch=b0,
            matured_targets=torch.randn(12) * 0.01,
            matured_preds=o0["pred_return"][:, 0],
        )
        o1_mem = model.forward_session(b1, state=st)
        o1_virgin = model.forward_session(b1, state=None)
    # Supervivientes cambian por memoria persistida pese a N=15 != 12
    assert not torch.allclose(o1_mem["pred_return"], o1_virgin["pred_return"], atol=1e-6)


def test_p0_2_bptt_trainable_titans_and_hrr_prototypes_plus_trace_readout():
    """P0-2: BPTT entrena Titans (`fused_proj`/`ach_gain`/`ne_gain`/`target_encoder`) + HRR 512."""
    torch.manual_seed(202)
    rng = np.random.default_rng(202)
    model = build_variation(
        "v02_titans_neuromod_only",
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=16,
        bptt_steps=4,
    )
    model.train()
    assert hasattr(model, "target_encoder")
    loss_fn = MMARRSMultiTaskLoss()

    state = None
    chunk_loss = torch.zeros(())
    for t in range(4):
        b = _synthetic_session_batch(rng, n_assets=14 + (t % 3))
        b["asset_ids"] = torch.arange(t, t + 14 + (t % 3), dtype=torch.long)
        y_ret = torch.randn(b["prices"].shape[0], 6) * 0.015
        out = model.forward_session(b, state=state, enable_early_exit=False)
        ld = loss_fn(
            step_outputs=out["loop_outputs"],
            targets=dict(returns=y_ret),
            exposures=b["exposures"],
            energy_total=out["energy"]["total"],
        )
        chunk_loss = chunk_loss + ld["total"]
        state = model.step_matured(
            out["state"],
            matured_batch=b,
            matured_targets=y_ret,
            matured_preds=out["pred_return"][:, 0].detach(),
            keep_grad=True,
        )

    (chunk_loss / 4.0).backward()
    for p_name, param in (
        ("fused_proj.weight", model.neuromod_memory.fused_proj.weight),
        ("ach_gain", model.neuromod_memory.ach_gain),
        ("ne_gain", model.neuromod_memory.ne_gain),
        ("target_encoder.weight", model.target_encoder.weight),
    ):
        assert param.grad is not None, f"{p_name} recibió grad=None (BPTT roto!)"
        assert torch.isfinite(param.grad).all(), f"{p_name} tiene grad no finito"
        assert float(param.grad.abs().sum()) > 0.0, f"{p_name} tiene grad cero"

    # HRRModernHopfieldMemory: default >= 512, k_proto=4 retiene múltiples sesiones + traces activos
    hrr = HRRModernHopfieldMemory(hidden_dim=24)
    assert hrr.max_slots >= 512
    assert hrr.k_proto == 4
    first_session_q = None
    first_reg = None
    for s_idx in range(12):
        q_s = torch.randn(40, 24)
        v_s = torch.randn(40, 24)
        reg_s = torch.randn(24)
        if s_idx == 0:
            first_session_q = q_s
            first_reg = reg_s
        filled = hrr.store(q_s, v_s, regime_key=reg_s, return_signal=torch.randn(40) * 0.02)
        assert int(filled) == (s_idx + 1) * 4

    assert int(hrr.slot_filled.sum()) == 48
    assert first_session_q is not None and first_reg is not None
    # Verificar que self.traces (5d/21d/63d/252d) participa activamente en retrieve()
    res_base = hrr.retrieve(first_session_q[:4], regime_key=first_reg)
    assert "trace_readout" in res_base
    assert float(res_base["trace_readout"].detach().abs().sum()) > 0.0
    saved_traces = hrr.traces.clone()
    with torch.no_grad():
        hrr.traces.zero_()
    res_no_trace = hrr.retrieve(first_session_q[:4], regime_key=first_reg)
    with torch.no_grad():
        hrr.traces.copy_(saved_traces)
    assert not torch.allclose(res_base["retrieved"], res_no_trace["retrieved"], atol=1e-6)


def test_p0_3_scale_n2600_portfolio_layers_and_vol_normalized_sharpe_loss():
    """P0-3: escala US N=2600 (||w||_1 > 0.5 en DiffLayer, P+ y Tool 4) + Sharpe normalizado."""
    from mars_titan.evaluation.metacognitive_agent import (
        ConvexPortfolioOptimizerPPlus,
        DifferentiablePortfolioLayer,
    )
    from mars_titan.models.mmarrs.cognitive_core import TitansCognitiveCore

    torch.manual_seed(303)
    n_us, k_fac = 2600, 9
    alpha = (torch.randn(n_us) * 0.02).requires_grad_(True)
    vol = torch.full((n_us,), 0.02)
    exposures = torch.randn(n_us, k_fac)

    # 1) DifferentiablePortfolioLayer a N=2600
    dpl = DifferentiablePortfolioLayer(steps=6, factor_penalty=8.0, gross_limit=1.0)
    w_dpl = dpl(alpha, vol=vol, exposures=exposures)
    assert float(w_dpl.detach().abs().sum()) > 0.5
    assert abs(float(w_dpl.detach().sum())) < 1e-3
    w_dpl.pow(2).sum().backward()
    assert alpha.grad is not None and torch.isfinite(alpha.grad).all()

    # 2) ConvexPortfolioOptimizerPPlus.solve_torch a N=2600
    opt_p = ConvexPortfolioOptimizerPPlus(factor_penalty=15.0, gross_limit=1.0)
    w_p = opt_p.solve_torch(alpha.detach(), vol=vol, exposures=exposures)
    assert float(w_p.detach().abs().sum()) > 0.5
    assert abs(float(w_p.detach().sum())) < 1e-3

    # 3) Cognitive Core Tool 4 a N=2600
    core = TitansCognitiveCore(hidden_dim=16, context_dim=8, thinking_steps=2, tool_mode="soft")
    x_us = torch.randn(n_us, 16, requires_grad=True)
    ctx_us = torch.randn(n_us, 8)
    h_out, _, cog_diag = core(x_us, ctx_us, exposures=exposures)
    w_tool4 = cog_diag["last_tool_aux"]["portfolio_weights"]
    assert float(w_tool4.detach().abs().sum()) > 0.5
    h_out.pow(2).mean().backward()
    assert x_us.grad is not None and torch.isfinite(x_us.grad).all()

    # 4) loss_sharpe normalizada por volatilidad de cartera (no -net_ret * 50.0)
    head = MultiHorizonGranularHead(hidden_dim=16, exposures_dim=4)
    loss_fn = MMARRSMultiTaskLoss(l1_cost=0.0)
    h = torch.randn(32, 16)
    pred = head(h, exposures=torch.randn(32, 4))
    y_ret = torch.randn(32, 6) * 0.02
    vol_low = torch.full((32, 6), 0.01)
    vol_hi = torch.full((32, 6), 0.04)
    ld_low = loss_fn(step_outputs=[pred], targets=dict(returns=y_ret, vol=vol_low))
    ld_hi = loss_fn(step_outputs=[pred], targets=dict(returns=y_ret, vol=vol_hi))
    # Con mayor volatilidad de cartera, el Sharpe normalizado se atenúa (no es constante -net*50)
    assert not torch.allclose(ld_low["sharpe_turnover"], ld_hi["sharpe_turnover"], atol=1e-4)
    assert torch.isfinite(ld_low["sharpe_turnover"]) and torch.isfinite(ld_hi["sharpe_turnover"])


def test_p0_4_multihorizon_purge_embargo_and_1d_horizon_mask():
    """P0-4: exige purge>=max_h y embargo>=ceil(0.2*max_h) y máscara horizon_mask en targets 1D."""
    # Config rechaza purge < 63 con 6 horizontes salvo allow_short_synthetic=True
    with pytest.raises(ValueError, match="purge_sessions"):
        MMARRSConfig(purge_sessions=5, embargo_sessions=13)
    with pytest.raises(ValueError, match="embargo_sessions"):
        MMARRSConfig(purge_sessions=63, embargo_sessions=5)
    ok_syn = MMARRSConfig(purge_sessions=5, embargo_sessions=5, allow_short_synthetic=True)
    assert ok_syn.purge_sessions == 5

    # FullHistorySplitConfig con active_horizons explícito rechaza purge=5
    with pytest.raises(ValueError, match="purge_sessions"):
        FullHistorySplitConfig(
            active_horizons=(1, 2, 5, 10, 21, 63), purge_sessions=5, embargo_sessions=13
        )

    # Targets 1D: solo entrena h=0 (cero gradiente hacia horizontes h=1..5 y e_horizon==0)
    torch.manual_seed(404)
    head = MultiHorizonGranularHead(hidden_dim=16, exposures_dim=4)
    verifier = FinancialEnergyVerifier()
    loss_fn = MMARRSMultiTaskLoss()

    pred = head(torch.randn(20, 16), exposures=torch.randn(20, 4))
    pred_ret = pred["pred_return"].detach().requires_grad_(True)
    pred_copy = dict(pred)
    pred_copy["pred_return"] = pred_ret

    h_mask = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    en = verifier(pred_copy, horizon_mask=h_mask)
    assert float(en["e_horizon"].detach()) == pytest.approx(0.0, abs=1e-7)

    y_1d = torch.randn(20) * 0.015
    ld = loss_fn(step_outputs=[pred_copy], targets=dict(returns=y_1d), energy_total=en["total"])
    ld["ic"].backward()
    assert pred_ret.grad is not None
    assert float(pred_ret.grad[:, 0].abs().sum()) > 0.0
    assert float(pred_ret.grad[:, 1:].abs().sum()) == pytest.approx(0.0, abs=1e-7)


def test_p0_5_no_self_gamed_vol_pred_real_adv_and_cumulative_n_trials(monkeypatch):
    """P0-5: nunca pasa `vol_pred` al evaluador; pasa `realized_vol`, `adv_usd` y `n_trials`."""
    from mars_titan.models.mmarrs import full_history_train as fht

    captured: dict[str, object] = {}
    orig_eval = fht.evaluate_granular_six_dimensions

    def _spy_eval(**kwargs):
        captured.update(kwargs)
        return orig_eval(**kwargs)

    monkeypatch.setattr(fht, "evaluate_granular_six_dimensions", _spy_eval)

    rng = np.random.default_rng(505)
    years = [y for y in (2020, 2021, 2022, 2023) for _ in range(3)]
    sessions = []
    for _ in years:
        b = _synthetic_session_batch(rng, n_assets=14)
        b["target_returns"] = torch.randn(14, 6) * 0.015
        b["realized_vol"] = torch.full((14,), 0.0314)
        b["adv_usd"] = torch.full((14,), 2.5e7)
        b["asset_ids"] = torch.arange(14, dtype=torch.long)
        sessions.append(b)

    cfg = FullHistorySplitConfig(
        train_start_year=2020,
        train_end_year=2022,
        val_year=2023,
        purge_sessions=1,
        embargo_sessions=1,
        allow_short_synthetic=True,
    )
    fht.train_full_history_variation(
        sessions,
        years,
        variation="v00_dense_baseline",
        split_cfg=cfg,
        hidden_dim=16,
        epochs=1,
        device="cpu",
        n_trials=17,
        market="US",
    )
    assert captured["n_trials"] == 17
    assert captured["market"] == "US"
    assert captured["volatility"] is not None
    np.testing.assert_allclose(captured["volatility"], 0.0314, atol=1e-5)
    assert captured["liquidity"] is not None
    np.testing.assert_allclose(captured["liquidity"], 2.5e7, rtol=1e-5)
    assert captured["asset_ids"] is not None


def test_p0_7_zero_host_syncs_audit_sdpa_math_store_and_canonical_cli(tmp_path):
    """P0-7: `audit_zero_host_syncs == 0` en los 5 módulos core + SDPBackend.MATH + CLI 8 flags."""
    from pathlib import Path

    from mars_titan.data.market_physics_v4_gpu import ZeroCopySessionStore
    from mars_titan.models.mmarrs import (
        CANONICAL_TRAIN_CLI_FLAGS,
        audit_zero_host_syncs,
        build_cli_parser,
        granular_heads,
        hypergraph_hrr,
        mor_loom_pc,
        neuromod_memory,
        sdpa_math_context,
    )
    from mars_titan.models.mmarrs import model as model_mod

    core_files = [
        Path(neuromod_memory.__file__),
        Path(hypergraph_hrr.__file__),
        Path(mor_loom_pc.__file__),
        Path(granular_heads.__file__),
        Path(model_mod.__file__),
    ]
    audit = audit_zero_host_syncs(core_files)
    assert audit["files_scanned"] == 5
    assert audit["forward_findings"] == 0, f"Forward host-syncs encontrados: {audit['findings']}"
    assert audit["total_findings"] == 0, f"Host-syncs encontrados: {audit['findings']}"
    assert audit["clean"] is True

    # SDPBackend.MATH y determinismo en fp32_strict
    rt_info = configure_blackwell_runtime(
        BlackwellRuntimeConfig(precision_mode="fp32_strict"), device="cpu"
    )
    assert rt_info["sdpa_backend"] == "MATH"
    assert rt_info["deterministic_algorithms"] is True
    with sdpa_math_context("fp32_strict"):
        q = torch.randn(2, 2, 4, 8)
        out_sdpa = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        assert torch.isfinite(out_sdpa).all()

    # CLI canónico 8 flags de blackwell_kernels.CANONICAL_TRAIN_CLI_FLAGS + FM-8 US|CN
    parser = build_cli_parser()
    option_strings = {opt for act in parser._actions for opt in act.option_strings}
    for flag in CANONICAL_TRAIN_CLI_FLAGS:
        assert flag in option_strings, f"Falta flag CLI canónico: {flag}"

    parsed = parser.parse_args(
        [
            "--variation",
            "v13_mmarrs_titan_full",
            "--market",
            "CN",
            "--family",
            "mmarrs_titan",
            "--out-dir",
            str(tmp_path / "out"),
            "--precision",
            "fp32_strict",
            "--vram-fraction",
            "0.14",
            "--epochs",
            "3",
            "--seed",
            "7",
        ]
    )
    assert parsed.market == "CN"
    assert parsed.family == "mmarrs_titan"
    assert parsed.vram_fraction == pytest.approx(0.14)

    # FM-8: rechazar US+CN en CLI
    with pytest.raises(SystemExit):
        parser.parse_args(["--market", "US+CN"])

    # #90: guardia PyTorch 2.14 contra `reduce-overhead` + `dynamic=True`
    from mars_titan.models.mmarrs import maybe_compile_model

    tiny = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="reduce-overhead"):
        maybe_compile_model(
            tiny,
            BlackwellRuntimeConfig(compile_mode="reduce-overhead"),
            dynamic=True,
        )

    # #84 P1-6: TODOS los IDs Familia 3 de `tfm_catalog.py` + `config_overrides` instancian limpio
    from mars_titan.models.mmarrs.tfm_catalog import list_variants

    for spec_f3 in list_variants(family="mmarrs_titan"):
        m_cat = build_variation(spec_f3.variant_id, hidden_dim=16, **spec_f3.config_overrides)
        assert m_cat.config.max_loops <= 3

    # Streaming ZeroCopySessionStore.iter_pinned_h2d dentro de train_full_history_variation
    rng = np.random.default_rng(707)
    years = [y for y in (2020, 2021, 2022, 2023) for _ in range(3)]
    sessions = []
    for _ in years:
        b = _synthetic_session_batch(rng, n_assets=12)
        b.pop("regime_vec")
        b.pop("ne_signal")
        b["target_returns"] = torch.randn(12, 6) * 0.015
        b["asset_ids"] = torch.arange(12, dtype=torch.long)
        sessions.append(b)
    store = ZeroCopySessionStore.from_sessions(tmp_path / "zc_store", sessions, years)
    split_cfg = FullHistorySplitConfig(
        train_start_year=2020,
        train_end_year=2022,
        val_year=2023,
        purge_sessions=1,
        embargo_sessions=1,
        allow_short_synthetic=True,
    )
    res_zc = train_full_history_variation(
        store,
        years,
        variation="v00_dense_baseline",
        split_cfg=split_cfg,
        hidden_dim=16,
        epochs=1,
        device="cpu",
    )
    assert np.isfinite(res_zc["val_report"]["dim2_ranking"]["rank_ic_mean"])


def test_p1_2_holographic_3l_step0_nonzero_gradients_and_diverse_domain_priors():
    """P1-2: HolographicBrain3L rompe triple-zero (gradiente != 0 en paso 0) + priors diversos."""
    torch.manual_seed(1212)
    hb3 = HolographicBrain3L(
        hidden_dim=16,
        n_modalities=5,
        n_cortical_areas=19,
        n_hidden_slots=4,
        n_hyperedges=6,
        regime_dim=10,
        rmt_dim=4,
    )
    # Priors diversos por dominio en membership_logits (no uniforme 1/19!)
    p_ab = torch.softmax(hb3.membership_logits.detach(), dim=-1)
    assert float(p_ab.std(dim=-1).min()) > 0.01
    assert not torch.allclose(p_ab[0], p_ab[2], atol=1e-2)

    mod_tokens = torch.randn(16, 5, 16)
    mod_mask = torch.ones(16, 5)
    mod_mask[0, 3] = 0.0
    mod_tokens[0, 3] = float("nan")  # NaN en modalidad ausente no debe fugarse!
    out, _ = hb3(mod_tokens, mod_mask, regime_vec=torch.randn(10), rmt_loadings=torch.randn(16, 4))
    assert torch.isfinite(out).all()
    assert float(out[0, 3].detach().abs().max()) == 0.0

    loss = out.pow(2).mean()
    loss.backward()
    for name, p in (
        ("area_u", hb3.area_u),
        ("membership_logits", hb3.membership_logits),
        ("thalamus_film[0].weight", hb3.thalamus_film[0].weight),
        ("thalamus_film[2].weight", hb3.thalamus_film[2].weight),
    ):
        assert p.grad is not None, f"{name}.grad es None en paso 0 (FM-5!)"
        assert torch.isfinite(p.grad).all()
        assert float(p.grad.abs().sum()) > 0.0, f"{name}.grad es 0 en paso 0 (FM-5!)"


def test_p1_3_v05_placebo_contraction_and_eval_frozen_balance_bias():
    """P1-3: placebo v05 biyectivo/cross-session con 7|N, layer_precision<=1 y balance congelado."""
    torch.manual_seed(1313)
    rng = np.random.default_rng(1313)
    # N=21 (divisible por 7!) y contexto constante de mercado entre activos
    n_a, d_m, c_d = 21, 16, 8
    moe_soft = HighSNRMoELayer(hidden_dim=d_m, context_dim=c_d, mode="soft")
    moe_shuf = HighSNRMoELayer(hidden_dim=d_m, context_dim=c_d, mode="shuffled_context")
    moe_shuf.load_state_dict(moe_soft.state_dict())

    x = torch.randn(n_a, d_m)
    const_ctx = torch.randn(1, c_d).expand(n_a, c_d).clone()
    other_session_ctx = torch.randn(1, c_d).expand(n_a, c_d).clone()
    _, d_soft = moe_soft(x, const_ctx)
    _, d_shuf = moe_shuf(x, const_ctx, placebo_context=other_session_ctx)
    assert not torch.allclose(d_soft["router_probs"], d_shuf["router_probs"], atol=1e-3)

    # En eval(), update_balance_bias y step_matured NO mutan balance_bias
    moe_soft.eval()
    bias_before = moe_soft.balance_bias.clone()
    moe_soft.update_balance_bias(torch.tensor([0.9, 0.02, 0.02, 0.02, 0.02, 0.02]), lr=0.5)
    assert torch.equal(moe_soft.balance_bias, bias_before)

    # layer_precision acotada <= 1.0 tras step_matured con error diminuto
    m = build_variation(
        "v07_moe_double_loop",
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=16,
    )
    m.eval()
    b = _synthetic_session_batch(rng, n_assets=14)
    bias_blk0 = m.recursive_core.step_blocks[0].balance_bias.clone()
    with torch.no_grad():
        o = m.forward_session(b)
        m.step_matured(
            o["state"],
            matured_batch=b,
            matured_targets=o["pred_return"][:, 0],  # error exacto = 0 -> antes subía Pi a 2.0!
            matured_preds=o["pred_return"][:, 0],
        )
    assert float(m.recursive_core.layer_precision.max()) <= 1.0 + 1e-6
    assert torch.equal(m.recursive_core.step_blocks[0].balance_bias, bias_blk0)


def test_four_independent_families_factory_sota_multimodal_telemetry_and_3seed(tmp_path):
    """Las 4 familias independientes instancian y corren fwd/bwd/step_matured + 3-seed ensemble."""
    from mars_titan.models.mmarrs import (
        CrossModalConflictFusion,
        SelfSupervisedMultimodalAlignmentLoss,
        SkewStudentTDistributionHead,
        StandaloneHolographicBrainModel,
        TitansCognitiveCore,
        build_family_model,
        read_telemetry_jsonl,
        train_multiseed_rank_ensemble,
    )

    # FM-8 y tope max_loops <= 4
    with pytest.raises(ValueError, match="FM-8|market|US"):
        MMARRSConfig(market="US+CN")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_loops"):
        MMARRSConfig(max_loops=5)
    assert MMARRSConfig().max_loops == 3
    assert MMARRSConfig().weight_decay == pytest.approx(1e-2)

    rng = np.random.default_rng(1414)
    dims = dict(
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=16,
    )
    b0 = _synthetic_session_batch(rng, n_assets=16)
    b0["asset_ids"] = torch.arange(16, dtype=torch.long)
    b1 = _synthetic_session_batch(rng, n_assets=16)
    b1["asset_ids"] = torch.arange(16, dtype=torch.long)
    y_mat = torch.randn(16, 6) * 0.015

    # Familia 1: titans_original
    m_f1 = build_family_model("titans_neuromod_sota", market="US", **dims)
    assert not any(isinstance(m, HighSNRMoELayer) for m in m_f1.modules())
    assert not any(isinstance(m, TitansCognitiveCore) for m in m_f1.modules())
    assert not any(isinstance(m, HolographicBrain3L) for m in m_f1.modules())

    # Familia 2: holographic_brain
    m_f2 = build_family_model("nb_full", **dims)
    assert isinstance(m_f2, StandaloneHolographicBrainModel)
    assert not any(isinstance(m, HighSNRMoELayer) for m in m_f2.modules())
    assert not any(isinstance(m, TitansCognitiveCore) for m in m_f2.modules())

    # Familia 3: mmarrs_titan (incluye v13b multimodal SOTA + v13c noic pinball twin)
    m_f3b = build_family_model("v13b_mmarrs_multimodal_sota", market="US", **dims)
    assert any(isinstance(m, CrossModalConflictFusion) for m in m_f3b.modules())
    assert any(isinstance(m, SelfSupervisedMultimodalAlignmentLoss) for m in m_f3b.modules())
    assert any(isinstance(m, SkewStudentTDistributionHead) for m in m_f3b.modules())
    m_f3c = build_family_model("v13c_mmarrs_noic_pinball_twin", market="US", **dims)
    assert m_f3c.config.noic_pinball_twin is True

    # Familia 4: titans_cognitive_core
    m_f4 = build_family_model("cog_soft_tools_k3", market="US", **dims)
    assert any(isinstance(m, TitansCognitiveCore) for m in m_f4.modules())
    assert not any(isinstance(m, HighSNRMoELayer) for m in m_f4.modules())

    # Verificar fwd + bwd + step_matured en LAS 4 FAMILIAS
    loss_fn = MMARRSMultiTaskLoss()
    for fam_name, fam_model in (
        ("titans_original", m_f1),
        ("holographic_brain", m_f2),
        ("mmarrs_titan_v13b", m_f3b),
        ("titans_cognitive_core", m_f4),
    ):
        fam_model.train()
        out0 = fam_model.forward_session(b0, state=None)
        aux = {}
        if "ssl_loss" in out0:
            aux["ssl"] = out0["ssl_loss"]
        if "weighted_ponder_loss" in out0:
            aux["ponder"] = out0["weighted_ponder_loss"]
        step_outs = out0.get("loop_outputs") or out0.get("cycle_outputs")
        en_total = out0["energy"]["total"] if "energy" in out0 else None
        ld = loss_fn(
            step_outputs=step_outs,
            targets=dict(returns=y_mat),
            exposures=b0["exposures"],
            energy_total=en_total,
            aux_losses=aux,
        )
        assert torch.isfinite(ld["total"]), f"{fam_name}: pérdida no finita"
        ld["total"].backward()
        grads = [p.grad for p in fam_model.parameters() if p.grad is not None]
        assert len(grads) > 0, f"{fam_name}: sin gradientes"
        assert all(torch.isfinite(g).all() for g in grads), f"{fam_name}: grad no finito"

        if isinstance(fam_model, StandaloneHolographicBrainModel):
            st1 = fam_model.step_matured(
                out0["state"],
                matured_targets=y_mat,
                matured_preds=out0["pred_return"][:, 0].detach(),
            )
        else:
            st1 = fam_model.step_matured(
                out0["state"],
                matured_batch=b0,
                matured_targets=y_mat,
                matured_preds=out0["pred_return"][:, 0].detach(),
            )
        if fam_name == "titans_cognitive_core":
            assert "cognitive_notes" in st1
            assert "ponder_kl_loss" in out0
        with torch.no_grad():
            out1 = fam_model.forward_session(b1, state=st1)
            assert torch.isfinite(out1["pred_return"]).all()

    # Verificar ResearchTelemetryMonitor (.jsonl.gz) + OptimalFitController + 3-seed rank ensemble
    years = [y for y in (2020, 2021, 2022, 2023) for _ in range(3)]
    sessions = []
    for _ in years:
        b = _synthetic_session_batch(rng, n_assets=14)
        y1 = 0.01 * b["prices"][:, 0] + 0.01 * torch.randn(14)
        b["target_returns"] = torch.stack([y1 for _ in range(6)], dim=-1)
        b["realized_vol"] = torch.full((14,), 0.02)
        b["adv_usd"] = torch.full((14,), 1.5e7)
        b["asset_ids"] = torch.arange(14, dtype=torch.long)
        sessions.append(b)
    split_cfg = FullHistorySplitConfig(
        train_start_year=2020,
        train_end_year=2022,
        val_year=2023,
        purge_sessions=1,
        embargo_sessions=1,
        allow_short_synthetic=True,
    )
    ens_res = train_multiseed_rank_ensemble(
        sessions,
        years,
        variation="v00_dense_baseline",
        seeds=(42, 43, 44),
        split_cfg=split_cfg,
        hidden_dim=16,
        epochs=2,
        optimizer_type="muon_hybrid",
        device="cpu",
        telemetry_dir=tmp_path / "tel",
    )
    assert ens_res["ensemble_summary"]["n_seeds"] == 3
    assert np.isfinite(ens_res["ensemble_val_report"]["dim2_ranking"]["rank_ic_mean"])
    gz_files = list((tmp_path / "tel").rglob("*.jsonl.gz"))
    assert len(gz_files) >= 1
    recs = read_telemetry_jsonl(gz_files[0])
    assert len(recs) >= 1


def test_all_four_families_full_history_train_fp32_bf16_and_compile_default_dynamic(tmp_path):
    """#91: 4 familias independientes entrenan en `full_history_train` (fp32_strict y bf16_amp)."""
    from mars_titan.models.mmarrs import build_family_model, maybe_compile_model

    rng = np.random.default_rng(9191)
    years = [y for y in (2020, 2021, 2022, 2023) for _ in range(3)]
    # Paneles irregulares N_t variable entre sesiones para ejercitar asset_ids + dynamic=True
    n_seq = [12, 14, 16, 13, 15, 14, 12, 16, 15, 14, 13, 15]
    sessions = []
    for n_a in n_seq:
        b = _synthetic_session_batch(rng, n_assets=n_a)
        y1 = 0.01 * b["prices"][:, 0] + 0.01 * torch.randn(n_a)
        b["target_returns"] = torch.stack([y1 for _ in range(6)], dim=-1)
        b["realized_vol"] = torch.full((n_a,), 0.02)
        b["adv_usd"] = torch.full((n_a,), 2.0e7)
        b["asset_ids"] = torch.arange(n_a, dtype=torch.long)
        sessions.append(b)

    split_cfg = FullHistorySplitConfig(
        train_start_year=2020,
        train_end_year=2022,
        val_year=2023,
        purge_sessions=1,
        embargo_sessions=1,
        allow_short_synthetic=True,
    )

    # Anti-sopa estricto: mezclar family y variation incompatibles debe fallar
    with pytest.raises(ValueError, match="Incompatibilidad de familia"):
        train_full_history_variation(
            sessions,
            years,
            family="titans_original",
            variation="v13_mmarrs_titan_full",
            split_cfg=split_cfg,
            hidden_dim=16,
            epochs=1,
            device="cpu",
        )

    family_cases = (
        ("titans_original", "titans_neuromod_sota"),
        ("holographic_brain", "nb_full"),
        ("mmarrs_titan", "v13b_mmarrs_multimodal_sota"),
        ("titans_cognitive_core", "cog_soft_tools_k3"),
    )
    for fam, var in family_cases:
        for prec in ("fp32_strict", "bf16_amp"):
            rt_cfg = BlackwellRuntimeConfig(precision_mode=prec, compile_mode=None)
            res = train_full_history_variation(
                sessions,
                years,
                family=fam,
                variation=var,
                market="US",
                split_cfg=split_cfg,
                runtime_cfg=rt_cfg,
                hidden_dim=16,
                epochs=1,
                lr=2e-3,
                bptt_steps=2,
                device="cpu",
                checkpoint_dir=tmp_path / f"{fam}_{prec}",
            )
            assert res["family"] == fam
            assert res["variation"] == var
            assert res["hardware"]["amp_dtype"] == (
                "bfloat16" if prec == "bf16_amp" else "float32"
            )
            assert all(np.isfinite(x) for x in res["train_losses"])
            assert np.isfinite(res["val_report"]["dim2_ranking"]["rank_ic_mean"])
            ckpt = torch.load(res["checkpoint_path"], weights_only=False)
            assert ckpt["family"] == fam
            # Pesos maestros siempre FP32 incluso en bf16_amp (cero degradación)
            for p_val in ckpt["base_state_dict"].values():
                if isinstance(p_val, torch.Tensor) and p_val.is_floating_point():
                    assert p_val.dtype == torch.float32

    # Verificar compatibilidad `maybe_compile_model(mode='default', dynamic=True)` en paneles N_t
    dims = dict(
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=16,
    )
    compile_cfg = BlackwellRuntimeConfig(precision_mode="bf16_amp", compile_mode="default")
    m_v00 = build_family_model("v00_dense_baseline", market="US", **dims)
    c_v00 = maybe_compile_model(m_v00, compile_cfg, dynamic=True)
    with torch.no_grad():
        o_a = c_v00(sessions[0], state=None, enable_early_exit=False)
        o_b = c_v00(sessions[2], state=None, enable_early_exit=False)
    assert o_a["pred_return"].shape[0] == 12
    assert o_b["pred_return"].shape[0] == 16
    assert torch.isfinite(o_a["pred_return"]).all() and torch.isfinite(o_b["pred_return"]).all()


def test_verifier_101_replay_ema_derangement_horizon_mask_and_buffer_fixes():
    """Regresión `#101`: EMA/reset replay F2, derangement exacto, horizon_mask 2D, buffer GPU."""
    import ast
    import copy
    from pathlib import Path

    from mars_titan.models.mmarrs import build_family_model
    from mars_titan.models.mmarrs.ablation_runner import _deranged_perm
    from mars_titan.models.mmarrs.full_history_train import (
        _build_deranged_context_indices,
        _update_ema,
        reset_episodic_buffers,
    )

    rng = np.random.default_rng(1010)
    dims = dict(
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=16,
    )

    # 1) Familia 2 `replay_memory`: `_update_ema` NO copia buffers episódicos y `reset` limpia todo
    m_nb = build_family_model("nb_full", **dims)
    ema_nb = copy.deepcopy(m_nb).eval()
    reset_episodic_buffers(ema_nb)
    b0 = _synthetic_session_batch(rng, n_assets=16)
    b0["asset_ids"] = torch.arange(16, dtype=torch.long)
    y_mat = torch.randn(16, 6) * 0.05
    with torch.no_grad():
        o0 = m_nb.forward_session(b0)
        m_nb.step_matured(
            o0["state"],
            matured_targets=y_mat,
            matured_preds=torch.zeros(16),
            asset_ids=b0["asset_ids"],
        )
    assert m_nb.replay_memory is not None and bool(m_nb.replay_memory.proto_filled.any())
    assert float(m_nb.replay_memory.proto_latent.abs().sum()) > 0.0

    _update_ema(ema_nb, m_nb, decay=0.5)
    assert ema_nb.replay_memory is not None
    assert not bool(ema_nb.replay_memory.proto_filled.any())
    assert float(ema_nb.replay_memory.proto_latent.abs().sum()) == 0.0

    reset_episodic_buffers(m_nb)
    assert not bool(m_nb.replay_memory.proto_filled.any())
    assert float(m_nb.replay_memory.proto_latent.abs().sum()) == 0.0
    assert float(m_nb.replay_memory.proto_target.abs().sum()) == 0.0
    assert m_nb.hrr_hopfield is not None and not bool(m_nb.hrr_hopfield.slot_filled.any())

    # 2) Derangement estrictamente sin puntos fijos para todo n >= 2 y múltiples semillas
    for fn in (_build_deranged_context_indices, _deranged_perm):
        for n in (2, 3, 4, 5, 10, 21):
            for s in range(25):
                perm = fn(n, s)
                assert perm.shape == (n,)
                assert sorted(perm.tolist()) == list(range(n))
                assert bool(np.all(perm != np.arange(n))), f"punto fijo en n={n}, seed={s}: {perm}"

    # 3) `horizon_mask` parcial en 2D == paridad exacta con 1D y gradiente 0 en horizontes 1..5
    torch.manual_seed(1013)
    n_a, d_m = 20, 16
    head = MultiHorizonGranularHead(hidden_dim=d_m, exposures_dim=4)
    loss_fn = MMARRSMultiTaskLoss(use_fused_blackwell_kernels=False)
    h = torch.randn(n_a, d_m)
    exp = torch.randn(n_a, 4)
    y_2d = torch.randn(n_a, 6) * 0.02
    tb_2d = torch.randint(0, 3, (n_a, 6))
    turn_2d = torch.randint(0, 2, (n_a, 6)).float()
    vol_2d = torch.full((n_a, 6), 0.02)
    hm_only0 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    with torch.no_grad():
        base_pred = head(h, exposures=exp)
    pred_2d = {k: v.detach().clone().requires_grad_(True) for k, v in base_pred.items()}
    pred_1d = {k: v.detach().clone().requires_grad_(True) for k, v in base_pred.items()}

    l_2d = loss_fn._step_loss(
        pred_2d,
        dict(returns=y_2d, tb_class=tb_2d, turning=turn_2d, vol=vol_2d),
        exp,
        horizon_mask=hm_only0,
    )
    l_1d = loss_fn._step_loss(
        pred_1d,
        dict(returns=y_2d[:, 0], tb_class=tb_2d[:, 0], turning=turn_2d[:, 0], vol=vol_2d[:, 0]),
        exp,
        horizon_mask=None,
    )
    for k in ("ic", "crps", "tb", "turn", "vol", "epist", "total"):
        torch.testing.assert_close(l_2d[k], l_1d[k], atol=1e-5, rtol=1e-5)

    l_2d["total"].backward()
    for key in (
        "pred_return",
        "quantiles",
        "tb_logits",
        "turn_logits",
        "log_vol",
        "epistemic_error",
    ):
        g = pred_2d[key].grad
        assert g is not None
        assert float(g[:, 1:].abs().max()) == 0.0, f"Gradiente filtrado a horizontes 1..5 en {key}"

    # 4) `FinancialEnergyVerifier` registra `horizon_scales` en buffer (cero `torch.tensor` en fwd)
    ver = FinancialEnergyVerifier()
    assert hasattr(ver, "horizon_scales")
    assert ver.horizon_scales.shape == (6,)
    mor_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mars_titan"
        / "models"
        / "mmarrs"
        / "mor_loom_pc.py"
    )
    tree = ast.parse(mor_path.read_text(encoding="utf-8"))
    tensor_calls_in_ver_fwd = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FinancialEnergyVerifier":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    for sub in ast.walk(item):
                        if (
                            isinstance(sub, ast.Call)
                            and isinstance(sub.func, ast.Attribute)
                            and sub.func.attr == "tensor"
                        ):
                            tensor_calls_in_ver_fwd.append(sub.lineno)
    assert tensor_calls_in_ver_fwd == []

    # 5) `ablation_runner` entrena `v13b` (`skew_t_head`) y `nb_full` con paneles ragged `asset_ids`
    ragged_s = []
    for n_a in (12, 14, 16, 13, 15, 14, 12, 16, 15, 14, 13, 15):
        bs = _synthetic_session_batch(rng, n_assets=n_a)
        y1 = 0.01 * bs["prices"][:, 0] + 0.01 * torch.randn(n_a)
        bs["target_returns"] = torch.stack([y1 for _ in range(6)], dim=-1)
        bs["realized_vol"] = torch.full((n_a,), 0.02)
        bs["adv_usd"] = torch.full((n_a,), 2.0e7)
        bs["asset_ids"] = torch.arange(n_a, dtype=torch.long)
        ragged_s.append(bs)
    ab_rep = run_variation_ablation(
        ragged_s,
        variations=("v13b_mmarrs_multimodal_sota", "nb_full"),
        train_end=8,
        epochs=1,
        hidden_dim=16,
        seed=77,
        purge_sessions=1,
        embargo_sessions=1,
        allow_short_synthetic=True,
    )
    assert np.isfinite(
        ab_rep["results"]["v13b_mmarrs_multimodal_sota"]["granular_report"]["dim2_ranking"][
            "rank_ic_mean"
        ]
    )
    assert np.isfinite(
        ab_rep["results"]["nb_full"]["granular_report"]["dim2_ranking"]["rank_ic_mean"]
    )


def test_118_119_zero_host_syncs_all_four_families_and_bf16_autocast_loss():
    """Regresión `#118`/`#119`: 0 host-syncs E2E en las 4 familias + loss segura en autocast."""
    from mars_titan.models.mmarrs import build_family_model
    from mars_titan.models.mmarrs.full_history_train import (
        _collect_aux_losses,
        _promote_fp32_outputs,
    )
    from mars_titan.models.mmarrs.neuro_brain_sota import StandaloneHolographicBrainModel

    torch.manual_seed(118119)
    # 1) Multi-paso ragged en NeuromodulatedTitansMemory sin `torch.unique`
    mem = NeuromodulatedTitansMemory(hidden_dim=16)
    st = None
    ids_seq = [
        torch.tensor([10, 20, 30, 40], dtype=torch.long),
        torch.tensor([30, 50, 10, 60, 20], dtype=torch.long),
        torch.tensor([60, 10, 70], dtype=torch.long),
    ]
    for ids in ids_seq:
        st, _ = mem.update(
            torch.randn(ids.shape[0], 16),
            st,
            ach_signal=torch.ones(ids.shape[0], 1) * 0.2,
            ne_signal=torch.tensor([0.1]),
            asset_ids=ids,
        )
    # Todos los activos vistos (incl. pausados 20, 40, 50) siguen accesibles vía read()
    probe_ids = torch.tensor([10, 20, 30, 40, 50, 60, 70], dtype=torch.long)
    out_p, _ = mem.read(torch.zeros(7, 16), st, asset_ids=probe_ids)
    z_ref = mem.out_proj(mem.read_norm(torch.zeros(1, 16))).squeeze(0)
    for idx_i in range(7):
        assert not torch.allclose(out_p[idx_i], z_ref, atol=1e-6)
    # Sesión 0 con IDs duplicados: se deduplican con sentinel al final sin host-syncs (#132)
    dup_ids0 = torch.tensor([20, 10, 20], dtype=torch.long)
    st_dup, _ = mem.update(
        torch.randn(3, 16),
        None,
        ach_signal=torch.ones(3, 1) * 0.2,
        ne_signal=torch.tensor([0.1]),
        asset_ids=dup_ids0,
    )
    real_dup = st_dup["asset_ids"][st_dup["asset_ids"] < (1 << 61)]
    assert real_dup.tolist() == [10, 20]

    # 2) `_neutralized_ic_loss` segura bajo autocast BF16 (CPU o CUDA)
    dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_bf = torch.randn(32, 6, device=dev_t, dtype=torch.bfloat16, requires_grad=True)
    y_bf = torch.randn(32, 6, device=dev_t, dtype=torch.bfloat16)
    b_bf = torch.randn(32, 4, device=dev_t, dtype=torch.bfloat16)
    with torch.amp.autocast(dev_t.type, dtype=torch.bfloat16):
        ic_l = MMARRSMultiTaskLoss._neutralized_ic_loss(p_bf, y_bf, exposures=b_bf)
    assert torch.isfinite(ic_l).all()
    ic_l.backward()
    assert p_bf.grad is not None and torch.isfinite(p_bf.grad).all()

    if not torch.cuda.is_available():
        return

    # 3) Profiler real CUDA (tiny N=64, D=64): 0 syncs en las 4 familias!
    from torch.profiler import ProfilerActivity, profile

    dev = torch.device("cuda")
    dims = dict(
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=64,
    )
    n = 64
    batch = {
        "prices": torch.randn(n, 16, device=dev),
        "fundamentals": torch.randn(n, 8, device=dev),
        "macro": torch.randn(n, 6, device=dev),
        "news": torch.randn(n, 8, device=dev),
        "charts": torch.randn(n, 8, device=dev),
        "physics": torch.randn(n, 12, device=dev),
        "context": torch.randn(n, 10, device=dev),
        "exposures": torch.randn(n, 4, device=dev),
        "rmt": torch.randn(n, 4, device=dev),
        "realized_vol": torch.full((n,), 0.02, device=dev),
        "adv_usd": torch.full((n,), 1e7, device=dev),
        "asset_ids": torch.arange(n, device=dev, dtype=torch.long),
    }
    y_mat = torch.randn(n, 6, device=dev) * 0.015
    loss_fn = MMARRSMultiTaskLoss().to(dev)

    def _e2e_step(model, state):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.forward_session(batch, state=state)
        with torch.amp.autocast("cuda", enabled=False):
            aux = _collect_aux_losses(out, model, y_mat)
            raw_steps = out.get("loop_outputs") or out.get("cycle_outputs") or [out]
            step_outs = _promote_fp32_outputs(raw_steps)
            en_dict = out.get("energy")
            en_raw = en_dict.get("total") if isinstance(en_dict, dict) else None
            en_total = en_raw.float() if isinstance(en_raw, torch.Tensor) else None
            ld = loss_fn(
                step_outputs=step_outs,
                targets=dict(returns=y_mat),
                exposures=batch["exposures"],
                energy_total=en_total,
                aux_losses=aux if aux else None,
            )
        ld["total"].backward()
        if isinstance(model, StandaloneHolographicBrainModel):
            return model.step_matured(
                out["state"],
                matured_targets=y_mat,
                matured_preds=out["pred_return"][:, 0].detach(),
            )
        return model.step_matured(
            out["state"],
            matured_batch=batch,
            matured_targets=y_mat,
            matured_preds=out["pred_return"][:, 0].detach(),
        )

    for var in ("titans_neuromod_sota", "nb_full", "v13_mmarrs_titan_full", "cog_soft_tools_k3"):
        m = build_family_model(var, market="US", **dims).to(dev)
        m.train()
        st_w = _e2e_step(m, None)
        m.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            _ = _e2e_step(m, st_w)
        torch.cuda.synchronize()
        bad = [
            e.name
            for e in prof.events()
            if any(
                k in e.name
                for k in (
                    "_local_scalar_dense",
                    "Memcpy DtoH",
                    "Memcpy HtoD",
                    "cudaStreamSynchronize",
                )
            )
        ]
        assert bad == [], f"{var} todavía tiene host-syncs en CUDA: {bad}"


def test_wave2_all_four_v2_families_e2e_training_bptt4_controls_and_zero_syncs(tmp_path):
    """Wave-2 E2E: las 4 familias v2 + controles, BPTT=4, bf16/fp32, liqw3, spectral y 0 syncs."""
    from pathlib import Path

    from mars_titan.models.mmarrs import (
        TOOL_NAMES_V2,
        CounterfactualStressSimulator,
        DualAxisTitansMemory,
        InducedFactorCrossAttention,
        LoopedCrossAssetTransformer,
        StandaloneHolographicBrainModel,
        TitansCognitiveCore,
        audit_zero_host_syncs,
        build_family_model,
        cognitive_core,
        counterfactual_sim,
        granular_heads,
        hypergraph_hrr,
        mor_loom_pc,
        neuro_brain_sota,
        neuromod_memory,
        read_telemetry_jsonl,
    )
    from mars_titan.models.mmarrs import model as model_mod

    # 1) Auditoría AST 0 host-syncs en TODOS los módulos core + v2
    core_v2_files = [
        Path(neuromod_memory.__file__),
        Path(hypergraph_hrr.__file__),
        Path(mor_loom_pc.__file__),
        Path(granular_heads.__file__),
        Path(model_mod.__file__),
        Path(counterfactual_sim.__file__),
        Path(neuro_brain_sota.__file__),
        Path(cognitive_core.__file__),
    ]
    sync_audit = audit_zero_host_syncs(core_v2_files)
    assert sync_audit["forward_findings"] == 0, f"Forward host-syncs: {sync_audit['findings']}"
    assert sync_audit["total_findings"] == 0, f"Host-syncs: {sync_audit['findings']}"
    assert sync_audit["clean"] is True
    assert len(TOOL_NAMES_V2) == 10
    assert CounterfactualStressSimulator is not None
    assert InducedFactorCrossAttention is not None

    dims = dict(
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=16,
        allow_short_synthetic=True,
    )
    rng = np.random.default_rng(202604)

    # 2) Familia 1 v2: DualAxisTitansMemory + last_ne_signal persistido + dual [z, rank] + TabM K=8
    m_f1 = build_family_model("t15b_full_v2", market="US", family="titans_original", **dims)
    assert isinstance(m_f1.neuromod_memory, DualAxisTitansMemory)
    assert m_f1.head.tabm_k == 8
    assert m_f1.enc_prices.net[1].in_features == 2 * 16
    assert m_f1.enc_physics.net[1].in_features == 2 * 12

    b_f1_0 = _synthetic_session_batch(rng, n_assets=16)
    b_f1_0.pop("ne_signal", None)
    b_f1_0["asset_ids"] = torch.arange(16, dtype=torch.long)
    b_f1_1 = _synthetic_session_batch(rng, n_assets=18)
    b_f1_1.pop("ne_signal", None)
    b_f1_1["asset_ids"] = torch.arange(2, 20, dtype=torch.long)

    m_f1.eval()
    with torch.no_grad():
        o_f1_0 = m_f1.forward_session(b_f1_0, state=None)
        assert o_f1_0["loop_outputs"][-1]["alpha_members"].shape == (16, 8, 6)
        # NaN en modalidad ausente aislado de forma exacta en canal dual [clipped_z, g_rank]
        b_nan = dict(b_f1_0)
        b_nan["fundamentals"] = b_f1_0["fundamentals"].clone()
        b_nan["modality_mask"] = b_f1_0["modality_mask"].clone()
        b_nan["modality_mask"][0, 1] = 0.0
        b_nan["fundamentals"][0] = float("nan")
        o_nan = m_f1.forward_session(b_nan, state=None)
        assert torch.isfinite(o_nan["pred_return"]).all()

        st_f1_1 = m_f1.step_matured(
            o_f1_0["state"],
            matured_batch=b_f1_0,
            matured_targets=torch.randn(16, 6) * 0.02,
            matured_preds=o_f1_0["pred_return"][:, 0],
        )
        assert isinstance(st_f1_1.get("last_ne_signal"), torch.Tensor)
        o_f1_1 = m_f1.forward_session(b_f1_1, state=st_f1_1)
        assert o_f1_1["diagnostics"]["neuromod"]["ne_slow_shift"] is not None
        # Shock grande en sesión 1 -> cambia last_ne_signal y descongela ne_slow_shift en sesión 2
        st_f1_2 = m_f1.step_matured(
            o_f1_1["state"],
            matured_batch=b_f1_1,
            matured_targets=torch.full((18, 6), 0.5),
            matured_preds=o_f1_1["pred_return"][:, 0],
        )
        o_f1_2 = m_f1.forward_session(b_f1_1, state=st_f1_2)
        ne_shift_1 = float(o_f1_1["diagnostics"]["neuromod"]["ne_slow_shift"])
        ne_shift_2 = float(o_f1_2["diagnostics"]["neuromod"]["ne_slow_shift"])
        assert abs(ne_shift_2 - ne_shift_1) > 1e-4

    # 3) Familia 2 v2: StandaloneHolographicBrainModel presets sin doble prefijo
    m_f2 = build_family_model("nb_v2_full", market="US", family="holographic_brain", **dims)
    assert isinstance(m_f2, StandaloneHolographicBrainModel)
    assert m_f2.config.use_unitary_fhrr is True
    assert m_f2.config.use_lgn_dual_input is True
    assert m_f2.config.use_active_inference_pi is True
    assert m_f2.config.use_homeostatic_bcm is True
    assert m_f2.config.use_stratified_swr is True

    # 4) Familia 3 v2: LoopedCrossAssetTransformer + CF + balance_bias train-only + aux losses
    m_f3 = build_family_model("v14c_tied_xasset_cf", market="US", family="mmarrs_titan", **dims)
    assert isinstance(m_f3.recursive_core, LoopedCrossAssetTransformer)
    assert m_f3.recursive_core.use_counterfactual is True
    assert m_f3.recursive_core.placebo_mode is False
    m_f3_pl = build_family_model("v14c_placebo", market="US", family="mmarrs_titan", **dims)
    assert isinstance(m_f3_pl.recursive_core, LoopedCrossAssetTransformer)
    assert m_f3_pl.recursive_core.placebo_mode is True

    m_f3.eval()
    bb_f3_eval = m_f3.recursive_core.step_routers[0].balance_bias.clone()
    with torch.no_grad():
        o_f3_ev = m_f3.forward_session(b_f1_0)
        assert "ponder_kl_loss" in o_f3_ev and "cf_fragility_loss" in o_f3_ev
        m_f3.step_matured(
            o_f3_ev["state"],
            matured_batch=b_f1_0,
            matured_targets=torch.randn(16, 6) * 0.02,
            matured_preds=o_f3_ev["pred_return"][:, 0],
        )
    assert torch.equal(m_f3.recursive_core.step_routers[0].balance_bias, bb_f3_eval)

    # 5) Familia 4 v2: TitansCognitiveCore v2_10tools + w_prev + tool_router.update_balance_bias
    m_f4 = build_family_model(
        "cog_v2_10tools_soft_k3", market="US", family="titans_cognitive_core", **dims
    )
    assert isinstance(m_f4.cognitive_core, TitansCognitiveCore)
    assert m_f4.cognitive_core.toolbox_version == "v2_10tools"
    assert m_f4.cognitive_core.n_tools == 10
    m_f4.eval()
    bb_f4_eval = m_f4.cognitive_core.tool_router.balance_bias.clone()
    with torch.no_grad():
        o_f4_ev = m_f4.forward_session(b_f1_0)
        assert "tool_aux_loss" in o_f4_ev and "ponder_kl_loss" in o_f4_ev
        m_f4.step_matured(
            o_f4_ev["state"],
            matured_batch=b_f1_0,
            matured_targets=torch.randn(16, 6) * 0.02,
            matured_preds=o_f4_ev["pred_return"][:, 0],
        )
    assert torch.equal(m_f4.cognitive_core.tool_router.balance_bias, bb_f4_eval)

    # En train(), balance_bias de F3 v2 y F4 v2 sí se actualizan en step_matured
    m_f4.train()
    o_f4_tr = m_f4.forward_session(b_f1_0)
    m_f4.step_matured(
        o_f4_tr["state"],
        matured_batch=b_f1_0,
        matured_targets=torch.randn(16, 6) * 0.02,
        matured_preds=o_f4_tr["pred_return"][:, 0].detach(),
    )
    assert not torch.equal(m_f4.cognitive_core.tool_router.balance_bias, bb_f4_eval)

    # 6) Entrenamiento E2E en `full_history_train` (`bf16_amp` + `fp32_strict`, `bptt_steps=4`)
    years = [y for y in (2020, 2021, 2022, 2023) for _ in range(3)]
    n_seq = [12, 14, 16, 13, 15, 14, 12, 16, 15, 14, 13, 15]
    sessions = []
    for n_a in n_seq:
        bs = _synthetic_session_batch(rng, n_assets=n_a)
        bs.pop("ne_signal", None)
        bs["high_snr_context"][:, 7] = torch.linspace(0.1, 0.95, n_a)
        bs["modality_age"] = torch.rand(n_a, 5) * 5.0
        y1 = 0.01 * bs["prices"][:, 0] + 0.01 * torch.randn(n_a)
        bs["target_returns"] = torch.stack([y1 for _ in range(6)], dim=-1)
        bs["realized_vol"] = torch.full((n_a,), 0.02)
        bs["adv_usd"] = torch.full((n_a,), 2.0e7)
        bs["asset_ids"] = torch.arange(n_a, dtype=torch.long)
        sessions.append(bs)

    split_cfg = FullHistorySplitConfig(
        train_start_year=2020,
        train_end_year=2022,
        val_year=2023,
        purge_sessions=1,
        embargo_sessions=1,
        allow_short_synthetic=True,
    )
    v2_flagships = (
        ("titans_original", "t15b_full_v2"),
        ("holographic_brain", "nb_v2_full"),
        ("mmarrs_titan", "v14c_tied_xasset_cf"),
        ("titans_cognitive_core", "cog_v2_10tools_soft_k3"),
    )
    for fam, var in v2_flagships:
        for prec in ("fp32_strict", "bf16_amp"):
            rt_cfg = BlackwellRuntimeConfig(precision_mode=prec, compile_mode=None)
            res = train_full_history_variation(
                sessions,
                years,
                family=fam,
                variation=var,
                market="US",
                split_cfg=split_cfg,
                runtime_cfg=rt_cfg,
                hidden_dim=16,
                epochs=1,
                lr=2e-3,
                bptt_steps=4,
                device="cpu",
                checkpoint_dir=tmp_path / f"v2_{fam}_{prec}",
            )
            assert res["family"] == fam
            assert res["variation"] == var
            assert all(np.isfinite(x) for x in res["train_losses"])
            assert np.isfinite(res["val_report"]["dim2_ranking"]["rank_ic_mean"])
            assert res["telemetry_path"] is not None
            tel_recs = read_telemetry_jsonl(res["telemetry_path"])
            assert len(tel_recs) >= 1
            ckpt = torch.load(res["checkpoint_path"], weights_only=False)
            for p_val in ckpt["base_state_dict"].values():
                if isinstance(p_val, torch.Tensor) and p_val.is_floating_point():
                    assert p_val.dtype == torch.float32

    # 7) Verificar hook post-step `update_spectral_buffers()`, `liqw3`/F1 loss y todos los controles
    spec_hits = {"f2": 0, "f3": 0, "f4": 0}
    orig_f2_sp = StandaloneHolographicBrainModel.update_spectral_buffers
    orig_f3_sp = LoopedCrossAssetTransformer.update_spectral_buffers
    orig_f4_sp = TitansCognitiveCore.update_spectral_buffers

    def _spy_f2(self, *a, **kw):
        spec_hits["f2"] += 1
        return orig_f2_sp(self, *a, **kw)

    def _spy_f3(self, *a, **kw):
        spec_hits["f3"] += 1
        return orig_f3_sp(self, *a, **kw)

    def _spy_f4(self, *a, **kw):
        spec_hits["f4"] += 1
        return orig_f4_sp(self, *a, **kw)

    StandaloneHolographicBrainModel.update_spectral_buffers = _spy_f2  # type: ignore[method-assign]
    LoopedCrossAssetTransformer.update_spectral_buffers = _spy_f3  # type: ignore[method-assign]
    TitansCognitiveCore.update_spectral_buffers = _spy_f4  # type: ignore[method-assign]
    try:
        ab_vars = (
            "t15b_full_v2",
            "nb_v2_full",
            "v14c_tied_xasset_cf",
            "cog_v2_10tools_soft_k3",
        )
        ab_res = run_variation_ablation(
            sessions,
            variations=ab_vars,
            train_end=8,
            epochs=1,
            hidden_dim=16,
            bptt_steps=4,
            seed=42,
            purge_sessions=1,
            embargo_sessions=1,
            allow_short_synthetic=True,
        )
    finally:
        StandaloneHolographicBrainModel.update_spectral_buffers = orig_f2_sp  # type: ignore[method-assign]
        LoopedCrossAssetTransformer.update_spectral_buffers = orig_f3_sp  # type: ignore[method-assign]
        TitansCognitiveCore.update_spectral_buffers = orig_f4_sp  # type: ignore[method-assign]

    assert spec_hits["f2"] >= 1 and spec_hits["f3"] >= 1 and spec_hits["f4"] >= 1
    assert set(ab_res["results"].keys()) == set(ab_vars)
    for v_name in ab_vars:
        assert np.isfinite(
            ab_res["results"][v_name]["granular_report"]["dim2_ranking"]["rank_ic_mean"]
        )

    # Smoke forward + step_matured sobre TODOS los controles v2 emparejados + alias v1
    matched_v2_controls = (
        "t15a_dual_axis_mem",
        "t15c_noic_twin",
        "nb_v2_fhrr_lgn_pi",
        "nb_v2_noic_twin",
        "nb_v2_moe_soft_k3",
        "v14a_tied_loop",
        "v14b_tied_xasset",
        "v14c_placebo",
        "v14c_noic_twin",
        "cog_v2_10tools_uniform_k3",
        "cog_v2_10tools_shuffled_cf_placebo_k3",
        "cog_v2_10tools_sparsemax_k3",
        "cog_v2_10tools_noic_twin",
        "cog_v1_6tools_k3",
    )
    for ctrl_id in matched_v2_controls:
        cm = build_family_model(ctrl_id, market="US", **dims)
        cm.eval()
        with torch.no_grad():
            co = cm.forward_session(b_f1_0)
            assert torch.isfinite(co["pred_return"]).all(), f"{ctrl_id} pred no finito"
            if isinstance(cm, StandaloneHolographicBrainModel):
                cst = cm.step_matured(
                    co["state"],
                    matured_targets=torch.randn(16, 6) * 0.02,
                    matured_preds=co["pred_return"][:, 0],
                    asset_ids=b_f1_0["asset_ids"],
                )
            else:
                cst = cm.step_matured(
                    co["state"],
                    matured_batch=b_f1_0,
                    matured_targets=torch.randn(16, 6) * 0.02,
                    matured_preds=co["pred_return"][:, 0],
                )
            co2 = cm.forward_session(b_f1_1, state=cst)
            assert torch.isfinite(co2["pred_return"]).all(), f"{ctrl_id} paso 2 no finito"


def test_verifier_169_skew_t_h3_1d_horizon_mask_and_ablation_bf16_recursive_state(tmp_path):
    """Regresión `#169`: `skew_t` con H=3 y 1D `[N]` sin fuga P0-4 + FP32 recursivo en ablation."""
    from mars_titan.models.mmarrs import build_family_model
    from mars_titan.models.mmarrs.full_history_train import _collect_aux_losses
    from mars_titan.models.mmarrs.granular_heads import infer_horizon_mask

    rng = np.random.default_rng(169169)
    dims = dict(
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=16,
        allow_short_synthetic=True,
    )

    # 1) Gradiente exacto 0 en horizontes inactivos (1..5) cuando `y_ret` es 1D `[N]` o `[N, 3]`
    m_v14c = build_family_model("v14c_tied_xasset_cf", market="US", **dims)
    m_v14c.train()
    b0 = _synthetic_session_batch(rng, n_assets=16)
    b0["asset_ids"] = torch.arange(16, dtype=torch.long)
    out = m_v14c.forward_session(b0, state=None, enable_early_exit=False)

    # 1a) 1D `[N]` -> solo horizonte 0 activo (grad == 0 en 1..5)
    st_1d = {
        k: v.detach().clone().requires_grad_(True) if isinstance(v, torch.Tensor) else v
        for k, v in out["skew_t"].items()
    }
    y_1d = torch.randn(16) * 0.02
    hm_1d = infer_horizon_mask(y_1d, 6)
    aux_1d = _collect_aux_losses({"skew_t": st_1d}, m_v14c, y_1d, horizon_mask=hm_1d)
    aux_1d["skew_t_nll"].backward()
    assert st_1d["loc"].grad is not None
    assert float(st_1d["loc"].grad[:, 0].abs().sum()) > 0.0
    assert float(st_1d["loc"].grad[:, 1:].abs().sum()) == 0.0

    # 1b) H=3 `[N, 3]` -> solo horizontes 0..2 activos (sin RuntimeError, grad == 0 en 3..5)
    st_h3 = {
        k: v.detach().clone().requires_grad_(True) if isinstance(v, torch.Tensor) else v
        for k, v in out["skew_t"].items()
    }
    y_h3 = torch.randn(16, 3) * 0.02
    hm_h3 = infer_horizon_mask(y_h3, 6)
    aux_h3 = _collect_aux_losses({"skew_t": st_h3}, m_v14c, y_h3, horizon_mask=hm_h3)
    assert torch.isfinite(aux_h3["skew_t_nll"])
    aux_h3["skew_t_nll"].backward()
    assert st_h3["loc"].grad is not None
    assert float(st_h3["loc"].grad[:, :3].abs().sum()) > 0.0
    assert float(st_h3["loc"].grad[:, 3:].abs().sum()) == 0.0

    # 2) E2E `run_variation_ablation` y `train_full_history_variation` con `[N, 3]` y 1D `[N]`
    years = [y for y in (2020, 2021, 2022, 2023) for _ in range(3)]
    split_cfg = FullHistorySplitConfig(
        train_start_year=2020,
        train_end_year=2022,
        val_year=2023,
        purge_sessions=1,
        embargo_sessions=1,
        allow_short_synthetic=True,
    )
    for tag, target_cols in (("h3", 3), ("1d", 0)):
        sess_list = []
        for _ in years:
            bs = _synthetic_session_batch(rng, n_assets=14)
            y1 = 0.01 * bs["prices"][:, 0] + 0.01 * torch.randn(14)
            bs["target_returns"] = (
                torch.stack([y1 * np.sqrt(h) for h in (1, 5, 21)], dim=-1)
                if target_cols == 3
                else y1
            )
            bs["realized_vol"] = torch.full((14,), 0.02)
            bs["adv_usd"] = torch.full((14,), 2.0e7)
            bs["asset_ids"] = torch.arange(14, dtype=torch.long)
            sess_list.append(bs)

        ab_r = run_variation_ablation(
            sess_list,
            variations=("v14c_tied_xasset_cf",),
            train_end=8,
            epochs=1,
            hidden_dim=16,
            bptt_steps=2,
            seed=42,
            purge_sessions=1,
            embargo_sessions=1,
            allow_short_synthetic=True,
        )
        assert np.isfinite(
            ab_r["results"]["v14c_tied_xasset_cf"]["granular_report"]["dim2_ranking"][
                "rank_ic_mean"
            ]
        )

        fht_r = train_full_history_variation(
            sess_list,
            years,
            variation="v14c_tied_xasset_cf",
            market="US",
            split_cfg=split_cfg,
            hidden_dim=16,
            epochs=1,
            bptt_steps=2,
            device="cpu",
            checkpoint_dir=tmp_path / f"v14c_{tag}",
        )
        assert all(np.isfinite(x) for x in fht_r["train_losses"])
        assert np.isfinite(fht_r["val_report"]["dim2_ranking"]["rank_ic_mean"])

    # 3) P1 Minor: `ablation_runner` con `nb_v2_full` bajo `bf16_amp` promueve estado anidado a FP32
    sess_bf = []
    for _ in range(10):
        bs = _synthetic_session_batch(rng, n_assets=14)
        bs["target_returns"] = torch.randn(14, 6) * 0.02
        bs["realized_vol"] = torch.full((14,), 0.02)
        bs["adv_usd"] = torch.full((14,), 2.0e7)
        bs["asset_ids"] = torch.arange(14, dtype=torch.long)
        sess_bf.append(bs)

    seen_dtypes: list[torch.dtype] = []
    from mars_titan.models.mmarrs.neuro_brain_sota import StandaloneHolographicBrainModel

    orig_step = StandaloneHolographicBrainModel.step_matured

    def _spy_step(self, st_arg, *a, **kw):
        if isinstance(st_arg, dict) and isinstance(st_arg.get("plasticity_state"), dict):
            for v in st_arg["plasticity_state"].values():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    seen_dtypes.append(v.dtype)
        return orig_step(self, st_arg, *a, **kw)

    StandaloneHolographicBrainModel.step_matured = _spy_step  # type: ignore[method-assign]
    try:
        ab_bf = run_variation_ablation(
            sess_bf,
            variations=("nb_v2_full",),
            train_end=7,
            epochs=1,
            hidden_dim=16,
            bptt_steps=4,
            seed=42,
            purge_sessions=1,
            embargo_sessions=1,
            allow_short_synthetic=True,
            runtime_cfg=BlackwellRuntimeConfig(precision_mode="bf16_amp", compile_mode=None),
        )
    finally:
        StandaloneHolographicBrainModel.step_matured = orig_step  # type: ignore[method-assign]

    assert np.isfinite(
        ab_bf["results"]["nb_v2_full"]["granular_report"]["dim2_ranking"]["rank_ic_mean"]
    )
    assert len(seen_dtypes) > 0
    assert all(dt == torch.float32 for dt in seen_dtypes)


def test_training_dynamics_162_164_schedule_free_ema_bias_and_epoch_train_ic(
    tmp_path, monkeypatch
):
    """Regresión `#162`/`#164`: SF sin EMA+warmup 5%, EMA bias-correct y `train_ic` por época."""
    import mars_titan.models.mmarrs.full_history_train as fht
    from mars_titan.models.mmarrs.autotuner_sota import (
        OptimalFitConfig,
        ScheduleFreeAdamW,
    )
    from mars_titan.models.mmarrs.granular_heads import MMARRSMultiTaskLoss

    # 1) Fórmula pura `_effective_ema_decay` (0 host-syncs)
    assert abs(fht._effective_ema_decay(0.995, 0, bias_correct=True) - 0.1) < 1e-12
    assert abs(fht._effective_ema_decay(0.995, 1, bias_correct=True) - (2.0 / 11.0)) < 1e-12
    assert abs(fht._effective_ema_decay(0.995, 9, bias_correct=True) - (10.0 / 19.0)) < 1e-12
    assert abs(fht._effective_ema_decay(0.995, 5000, bias_correct=True) - 0.995) < 1e-12
    assert abs(fht._effective_ema_decay(0.995, 0, bias_correct=False) - 0.995) < 1e-12

    rng = np.random.default_rng(162164)
    years = [y for y in (2020, 2021, 2022, 2023) for _ in range(3)]
    sessions = []
    for idx, _ in enumerate(years):
        bs = _synthetic_session_batch(rng, n_assets=14)
        sign = -1.0 if idx == 7 else 1.0
        y1 = sign * 0.02 * bs["prices"][:, 0] + 0.005 * torch.randn(14)
        bs["target_returns"] = torch.stack([y1 * np.sqrt(h) for h in (1, 5, 21, 63, 126, 252)], -1)
        bs["realized_vol"] = torch.full((14,), 0.02)
        bs["adv_usd"] = torch.full((14,), 2.0e7)
        bs["asset_ids"] = torch.arange(14, dtype=torch.long)
        sessions.append(bs)

    split_cfg = FullHistorySplitConfig(
        train_start_year=2020,
        train_end_year=2022,
        val_year=2023,
        purge_sessions=1,
        embargo_sessions=1,
        allow_short_synthetic=True,
    )
    fit_cfg = OptimalFitConfig(patience=5, max_lr_decays=2, top_k_soup=3)

    # 2) ScheduleFreeAdamW por defecto: ZERO EMA stacking, eval directo de `base_model` en `x_t`,
    #    `opt.train()` restaurado entre épocas y `sf_warmup_steps = max(1, total_steps // 20)`
    ema_calls_sf: list[float] = []
    orig_update_ema = fht._update_ema
    monkeypatch.setattr(
        fht,
        "_update_ema",
        lambda em, m, d: (ema_calls_sf.append(float(d)), orig_update_ema(em, m, d))[1],
    )

    sf_opts: list[ScheduleFreeAdamW] = []
    orig_sf_init = ScheduleFreeAdamW.__init__

    def _spy_sf_init(self, *a, **kw):
        orig_sf_init(self, *a, **kw)
        sf_opts.append(self)

    monkeypatch.setattr(ScheduleFreeAdamW, "__init__", _spy_sf_init)

    eval_checks: list[tuple[bool, bool]] = []
    orig_eval_causal = fht._evaluate_model_causal

    def _spy_eval(eval_m, *a, **kw):
        opt_cur = sf_opts[-1]
        in_eval_mode = all(not g.get("train_mode", True) for g in opt_cur.param_groups)
        # Verificar que `eval_m` comparte parámetros exactos con `opt_cur` (i.e. es `base_model`)
        opt_param_ids = {id(p) for g in opt_cur.param_groups for p in g["params"]}
        is_base_model = all(id(p) in opt_param_ids for p in eval_m.parameters())
        eval_checks.append((in_eval_mode, is_base_model))
        return orig_eval_causal(eval_m, *a, **kw)

    monkeypatch.setattr(fht, "_evaluate_model_causal", _spy_eval)

    # n_train = 8 sesiones, bptt_steps = 2 -> 4 steps/epoch, 6 epochs -> total_steps = 24
    # => default sf_warmup_steps = max(1, 24 // 20) = 1
    res_sf = train_full_history_variation(
        sessions,
        years,
        variation="titans_neuromod_sota",
        market="US",
        split_cfg=split_cfg,
        fit_cfg=fit_cfg,
        hidden_dim=16,
        epochs=6,
        lr=2e-3,
        optimizer_type="schedule_free_adamw",
        bptt_steps=2,
        device="cpu",
        checkpoint_dir=tmp_path / "sf_default",
    )
    assert len(ema_calls_sf) == 0, f"ScheduleFree por defecto no debe apilar EMA: {ema_calls_sf}"
    assert res_sf["training_dynamics"]["ema_enabled"] is False
    assert res_sf["training_dynamics"]["ema_updates"] == 0
    assert res_sf["training_dynamics"]["sf_disable_ema"] is True
    assert res_sf["training_dynamics"]["sf_warmup_steps"] == max(1, (6 * 4) // 20)
    assert sf_opts[-1].param_groups[0]["warmup_steps"] == max(1, (6 * 4) // 20)
    # 6 evaluaciones por época + 1 evaluación final: todas con `opt.eval()` y `base_model` (`x_t`)
    assert len(eval_checks) == 7
    assert all(in_ev and is_base for in_ev, is_base in eval_checks)
    ckpt_sf = torch.load(res_sf["checkpoint_path"], weights_only=False)
    assert ckpt_sf["ema_enabled"] is False

    # 2b) Control Round 0A `SF(buggy-EMA)`:
    #     `sf_disable_ema=False, sf_warmup_steps=0, ema_bias_correct=False`
    monkeypatch.setattr(fht, "_evaluate_model_causal", orig_eval_causal)
    ema_calls_ctrl: list[float] = []
    monkeypatch.setattr(
        fht,
        "_update_ema",
        lambda em, m, d: (ema_calls_ctrl.append(float(d)), orig_update_ema(em, m, d))[1],
    )
    res_sf_ctrl = train_full_history_variation(
        sessions,
        years,
        variation="titans_neuromod_sota",
        market="US",
        split_cfg=split_cfg,
        fit_cfg=fit_cfg,
        hidden_dim=16,
        epochs=2,
        lr=2e-3,
        optimizer_type="schedule_free_adamw",
        sf_disable_ema=False,
        sf_warmup_steps=0,
        ema_bias_correct=False,
        ema_decay=0.995,
        bptt_steps=2,
        device="cpu",
    )
    assert res_sf_ctrl["training_dynamics"]["ema_enabled"] is True
    assert res_sf_ctrl["training_dynamics"]["sf_warmup_steps"] == 0
    assert sf_opts[-1].param_groups[0]["warmup_steps"] == 0
    assert len(ema_calls_ctrl) == 8  # 2 epochs * 4 steps
    assert all(abs(d - 0.995) < 1e-12 for d in ema_calls_ctrl)

    # 3) EMA bias-correct para `adamw_fused` y `muon_hybrid` + control `ema_bias_correct=False`
    for opt_name in ("adamw_fused", "muon_hybrid"):
        decays_bc: list[float] = []
        monkeypatch.setattr(
            fht,
            "_update_ema",
            lambda em, m, d, _buf=decays_bc: (_buf.append(float(d)), orig_update_ema(em, m, d))[1],
        )
        res_opt = train_full_history_variation(
            sessions,
            years,
            variation="titans_neuromod_sota",
            market="US",
            split_cfg=split_cfg,
            fit_cfg=fit_cfg,
            hidden_dim=16,
            epochs=2,
            lr=1e-3,
            optimizer_type=opt_name,
            ema_decay=0.995,
            bptt_steps=2,
            device="cpu",
        )
        expected_decays = [min(0.995, (1.0 + s) / (10.0 + s)) for s in range(len(decays_bc))]
        assert len(decays_bc) == 8
        assert res_opt["training_dynamics"]["ema_bias_correct"] is True
        assert np.allclose(decays_bc, expected_decays, atol=1e-12)

    decays_raw: list[float] = []
    monkeypatch.setattr(
        fht,
        "_update_ema",
        lambda em, m, d: (decays_raw.append(float(d)), orig_update_ema(em, m, d))[1],
    )
    res_raw = train_full_history_variation(
        sessions,
        years,
        variation="titans_neuromod_sota",
        market="US",
        split_cfg=split_cfg,
        fit_cfg=fit_cfg,
        hidden_dim=16,
        epochs=2,
        lr=1e-3,
        optimizer_type="adamw_fused",
        ema_decay=0.995,
        ema_bias_correct=False,
        bptt_steps=2,
        device="cpu",
    )
    assert res_raw["training_dynamics"]["ema_bias_correct"] is False
    assert all(abs(d - 0.995) < 1e-12 for d in decays_raw)

    # 4) `OptimalFitController` `train_ic` acumula `ld["rho_ic"]` en TODAS las sesiones de la época
    monkeypatch.setattr(fht, "_update_ema", orig_update_ema)
    recorded_rhos: list[float] = []
    orig_loss_fwd = MMARRSMultiTaskLoss.forward

    def _spy_loss_fwd(self, *a, **kw):
        out_ld = orig_loss_fwd(self, *a, **kw)
        # Inyectar valores distintos en la última sesión de cada época para demostrar media global
        step_in_ep = len(recorded_rhos) % 8
        fake_rho = torch.tensor(
            -0.80 if step_in_ep == 7 else (0.10 + 0.05 * step_in_ep),
            device=out_ld["rho_ic"].device,
            dtype=torch.float32,
        )
        out_ld = dict(out_ld)
        out_ld["rho_ic"] = fake_rho
        recorded_rhos.append(float(fake_rho.item()))
        return out_ld

    monkeypatch.setattr(MMARRSMultiTaskLoss, "forward", _spy_loss_fwd)
    res_ic = train_full_history_variation(
        sessions,
        years,
        variation="titans_neuromod_sota",
        market="US",
        split_cfg=split_cfg,
        fit_cfg=fit_cfg,
        hidden_dim=16,
        epochs=2,
        lr=1e-3,
        optimizer_type="adamw_fused",
        bptt_steps=2,
        device="cpu",
    )
    assert len(recorded_rhos) == 16  # 2 epochs * 8 train sessions
    for ep in range(2):
        ep_rhos = recorded_rhos[ep * 8 : (ep + 1) * 8]
        expected_mean_ic = float(np.mean(ep_rhos))
        last_only_ic = ep_rhos[-1]
        assert abs(expected_mean_ic - last_only_ic) > 0.5
        assert abs(res_ic["train_ics"][ep] - expected_mean_ic) < 1e-5
        assert abs(res_ic["fit_decisions"][ep]["train_ic"] - expected_mean_ic) < 1e-5


def test_wave3c_frontier_switches_across_all_4_families_and_set_fusion_78_420():
    """Wave-3C: PolarAlphaHead + norm_kind across all 4 families + SetModalityFusion (78/420)."""
    from mars_titan.models.mmarrs.dl_training_sota import DynamicTanh, RMSNormSOTA
    from mars_titan.models.mmarrs.frontier_modules_sota import (
        MissingnessAwareSetModalityFusion,
        ObservedSlotEncoder,
    )
    from mars_titan.models.mmarrs.model import build_family_model

    torch.manual_seed(301)
    rng = np.random.default_rng(301)

    base_dims = dict(
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=24,
        allow_short_synthetic=True,
    )
    b = _synthetic_session_batch(rng, n_assets=20)
    b["realized_vol"] = torch.full((20,), 0.022)

    # 1) PolarAlphaHead + final_hidden + norm_kind across ALL 4 independent families
    four_fam_variants = (
        ("t15b_full_v2", "titans_original"),
        ("nb_v2_full", "holographic_brain"),
        ("v14c_tied_xasset_cf", "mmarrs_titan"),
        ("cog_v2_10tools_soft_k3", "titans_cognitive_core"),
    )
    for var_id, fam in four_fam_variants:
        for nk, n_cls in (("rmsnorm", RMSNormSOTA), ("dyt", DynamicTanh)):
            m = build_family_model(
                var_id,
                family=fam,
                use_polar_alpha_head=True,
                norm_kind=nk,
                **base_dims,
            )
            assert isinstance(m.head.norm, n_cls), f"{var_id} ({fam}) head.norm not {n_cls}"
            m.train()
            out = m.forward_session(b)
            assert "final_hidden" in out and out["final_hidden"].shape == (20, 24)
            for key in (
                "alpha_raw",
                "alpha_u",
                "alpha_cal",
                "alpha_cal_for_cal",
                "alpha_cal_for_pplus",
            ):
                assert key in out, f"{var_id} ({fam}) missing `{key}`"
                assert torch.isfinite(out[key]).all()
            torch.testing.assert_close(out["alpha"], out["alpha_cal"], atol=0.0, rtol=0.0)
            torch.testing.assert_close(
                out["alpha"].mean(dim=0), torch.zeros(6), atol=1e-5, rtol=1e-5
            )
            (
                out["alpha_u"].pow(2).mean()
                + out["alpha_cal_for_cal"].pow(2).mean()
                + out["alpha_cal_for_pplus"].pow(2).mean()
            ).backward()
            assert m.polar_alpha_head.raw_gamma.grad is not None
            assert torch.isfinite(m.polar_alpha_head.raw_gamma.grad).all()

    # 2) MissingnessAwareSetModalityFusion + ObservedSlotEncoder on 78/420 triplet inputs
    m_set = build_variation(
        "v02_titans_neuromod_only",
        prices_dim=16,
        fundamentals_dim=78,
        macro_dim=420,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=24,
        use_set_modality_fusion=True,
        allow_short_synthetic=True,
    )
    assert isinstance(m_set.enc_fundamentals, ObservedSlotEncoder)
    assert isinstance(m_set.enc_macro, ObservedSlotEncoder)
    assert isinstance(m_set.set_modality_fusion, MissingnessAwareSetModalityFusion)
    m_set.eval()

    b78 = _synthetic_session_batch(rng, n_assets=20)
    f_vals = torch.randn(20, 26)
    f_obs = (torch.rand(20, 26) > 0.5).float()
    f_age = torch.rand(20, 26) * 2.0
    m_vals = torch.randn(20, 140)
    m_obs = (torch.rand(20, 140) > 0.3).float()
    m_age = torch.rand(20, 140) * 2.0
    b78["fundamentals"] = torch.cat([f_vals, f_obs, f_age], dim=-1)
    b78["macro"] = torch.cat([m_vals, m_obs, m_age], dim=-1)
    b78["modality_mask"][:10, 3] = 0.0  # news absent for first 10 assets

    # Poison unobserved slots (obs==0) and absent modality rows (mask==0) -> bit-exact invariance!
    b78_pois = {k: v.clone() for k, v in b78.items()}
    f_vals_p = torch.where(f_obs > 0.5, f_vals, torch.full_like(f_vals, 1e6))
    m_vals_p = torch.where(m_obs > 0.5, m_vals, torch.full_like(m_vals, -1e6))
    b78_pois["fundamentals"] = torch.cat([f_vals_p, f_obs, f_age], dim=-1)
    b78_pois["macro"] = torch.cat([m_vals_p, m_obs, m_age], dim=-1)
    b78_pois["news"][:10] = float("nan")

    with torch.no_grad():
        o_clean = m_set.forward_session(b78)
        o_pois = m_set.forward_session(b78_pois)
    assert torch.isfinite(o_pois["pred_return"]).all()
    torch.testing.assert_close(o_clean["pred_return"], o_pois["pred_return"], atol=1e-6, rtol=1e-6)

    # 3) Strict 4-family preservation: v13b keeps CrossModalConflictFusion (never overwritten!)
    m_v13b = build_variation(
        "v13b_mmarrs_multimodal_sota",
        use_set_modality_fusion=True,
        **base_dims,
    )
    assert m_v13b.conflict_fusion is not None
    assert m_v13b.set_modality_fusion is None


def test_wave3c_full_history_train_dl_taxonomy_featurizer_ssl_probe_and_exports(tmp_path):
    """Wave-3C: full_history_train DL taxonomy, featurizer, curriculum, SSL, probe & exports."""
    import mars_titan.models.mmarrs as mmarrs_pkg

    # 1) Verify key SOTA pillar exports in mmarrs.__init__
    for sym in (
        "DynamicTanh",
        "RMSNormSOTA",
        "SwiGLUFFN",
        "LayerScale",
        "DeepNormResidual",
        "build_norm_layer",
        "apply_mup_init_and_param_groups",
        "tail_listnet_loss",
        "vicreg_covariance_loss",
        "cross_sectional_rank_contrastive_loss",
        "adaptive_gradient_clip_",
        "apply_cautious_weight_decay_",
        "CausalCurriculumScheduler",
        "CausalSSLPretrainer",
        "StructuredModalityNoiseAugmenter",
        "PolarAlphaHead",
        "ObservedSlotEncoder",
        "MissingnessAwareSetModalityFusion",
        "DiffInducedFactorCrossAttention",
        "compute_cross_family_disagreement_risk",
        "DeepLearningGeometryProbe",
        "DeepLearningHealthReport",
        "diagnose_training_health",
    ):
        assert hasattr(mmarrs_pkg, sym), f"Missing export `{sym}` in mars_titan.models.mmarrs"

    # 2) Build synthetic sessions WITH NaNs in train & val to prove MissingnessResilientFeaturizer
    rng = np.random.default_rng(302)
    years = [y for y in (2020, 2021, 2022, 2023) for _ in range(3)]
    sessions = []
    for s_i, _ in enumerate(years):
        b = _synthetic_session_batch(rng, n_assets=16)
        b["prices"][0, 1] = float("nan")
        b["fundamentals"][1, 2] = float("nan")
        b["high_snr_context"][2, 0] = float("nan")
        b["realized_vol"] = torch.full((16,), 0.02)
        b["adv_usd"] = torch.full((16,), 1.5e7)
        b["asset_ids"] = torch.arange(s_i, s_i + 16, dtype=torch.long)
        b["target_returns"] = torch.randn(16, 6) * 0.015
        sessions.append(b)

    split_cfg = FullHistorySplitConfig(
        train_start_year=2020,
        train_end_year=2022,
        val_year=2023,
        purge_sessions=1,
        embargo_sessions=1,
        allow_short_synthetic=True,
    )

    # Full SOTA DL taxonomy flags active simultaneously on Family 3
    res = train_full_history_variation(
        sessions,
        years,
        variation="v14c_tied_xasset_cf",
        market="US",
        split_cfg=split_cfg,
        hidden_dim=16,
        epochs=2,
        lr=1e-3,
        optimizer_type="muon_hybrid",
        bptt_steps=2,
        device="cpu",
        w_crps=10.0,
        w_listnet_tail=0.3,
        w_latent_cov=0.05,
        w_supcon=0.05,
        w_churn=0.1,
        use_mup=True,
        mup_base_width=16,
        use_agc=True,
        agc_clip_factor=0.02,
        use_cautious_wd=True,
        use_missingness_featurizer=True,
        mod_drop_prob=0.15,
        input_noise_std=0.02,
        curriculum_enabled=True,
        ssl_pretrain_epochs=1,
        collect_geometry_probe=True,
        use_polar_alpha_head=True,
        use_diff_xasset_attn=True,
        norm_kind="rmsnorm",
    )
    assert np.isfinite(res["val_report"]["dim2_ranking"]["rank_ic_mean"])
    assert "geometry_report" in res and isinstance(res["geometry_report"], dict)
    assert "diagnosis" in res["geometry_report"]
    assert "representation" in res["geometry_report"]
    dyn = res["training_dynamics"]
    assert dyn["use_mup"] is True
    assert dyn["use_agc"] is True
    assert dyn["use_cautious_wd"] is True
    assert dyn["use_missingness_featurizer"] is True
    assert dyn["curriculum_enabled"] is True
    assert dyn["ssl_pretrain_epochs"] == 1
    assert len(dyn["ssl_pretrain_losses"]) == 1
    assert np.isfinite(dyn["ssl_pretrain_losses"][0])

    # Verify MissingnessResilientFeaturizer + Polar + Geometry Probe across remaining 3 families!
    for var_id, fam in (
        ("t15b_full_v2", "titans_original"),
        ("nb_v2_full", "holographic_brain"),
        ("cog_v2_10tools_soft_k3", "titans_cognitive_core"),
    ):
        res_f = train_full_history_variation(
            sessions,
            years,
            variation=var_id,
            family=fam,
            market="US",
            split_cfg=split_cfg,
            hidden_dim=16,
            epochs=1,
            lr=1e-3,
            optimizer_type="schedule_free_adamw",
            bptt_steps=2,
            device="cpu",
            use_mup=True,
            mup_base_width=16,
            use_missingness_featurizer=True,
            use_polar_alpha_head=True,
            w_churn=0.05,
            ssl_pretrain_epochs=1,
            collect_geometry_probe=True,
        )
        assert res_f["family"] == fam
        assert np.isfinite(res_f["val_report"]["dim2_ranking"]["rank_ic_mean"])
        assert isinstance(res_f["geometry_report"], dict)
        assert "diagnosis" in res_f["geometry_report"]


def test_wave3c_233_p0_p1_regressions_polar_cwd_triplet_mup_f4_f2_guard(monkeypatch):
    """#233 regressions: P0-1 Polar pred_return, P0-2 SF CWD, P0-3 triplet noise, P1-1/P1-2."""
    from mars_titan.models.mmarrs import full_history_train as fht
    from mars_titan.models.mmarrs.dl_training_sota import StructuredModalityNoiseAugmenter
    from mars_titan.models.mmarrs.model import build_family_model

    torch.manual_seed(233)
    rng = np.random.default_rng(233)

    # 1) P0-1: With large non-zero exposures, Polar pred_return == alpha_cal (factor_return == 0)
    base_dims = dict(
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=24,
        allow_short_synthetic=True,
    )
    b_exp = _synthetic_session_batch(rng, n_assets=24)
    b_exp["exposures"] = torch.randn(24, 4) * 5.0
    b_exp["realized_vol"] = torch.full((24,), 0.02)
    for var_id, fam in (
        ("t15b_full_v2", "titans_original"),
        ("nb_v2_full", "holographic_brain"),
        ("v14c_tied_xasset_cf", "mmarrs_titan"),
        ("cog_v2_10tools_soft_k3", "titans_cognitive_core"),
    ):
        m_p = build_family_model(var_id, family=fam, use_polar_alpha_head=True, **base_dims)
        m_p.eval()
        with torch.no_grad():
            o_p = m_p.forward_session(b_exp)
        torch.testing.assert_close(
            o_p["factor_return"], torch.zeros_like(o_p["alpha_cal"]), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(o_p["pred_return"], o_p["alpha_cal"], atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            o_p["pred_return"].mean(dim=0), torch.zeros(6), atol=1e-5, rtol=1e-5
        )

    # 2) P0-3: StructuredModalityNoiseAugmenter preserves obs_flag & log_age on 78/420 triplets!
    aug = StructuredModalityNoiseAugmenter(modality_dropout=0.0, noise_std=0.25)
    aug.train()
    f_v = torch.randn(20, 26)
    f_o = (torch.rand(20, 26) > 0.5).float()
    f_a = torch.rand(20, 26) * 3.0
    m_v = torch.randn(20, 140)
    m_o = (torch.rand(20, 140) > 0.4).float()
    m_a = torch.rand(20, 140) * 3.0
    b_tri = dict(
        prices=torch.randn(20, 16),
        fundamentals=torch.cat([f_v, f_o, f_a], dim=-1),
        macro=torch.cat([m_v, m_o, m_a], dim=-1),
        modality_mask=torch.ones(20, 5),
    )
    gen = torch.Generator().manual_seed(77)
    b_aug = aug(b_tri, generator=gen)
    # obs_flag and log_age MUST be bit-exact; unobserved vals (obs==0) bit-exact; observed noisy!
    torch.testing.assert_close(b_aug["fundamentals"][:, 26:52], f_o, atol=0.0, rtol=0.0)
    torch.testing.assert_close(b_aug["fundamentals"][:, 52:78], f_a, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        b_aug["fundamentals"][:, :26][f_o < 0.5], f_v[f_o < 0.5], atol=0.0, rtol=0.0
    )
    assert not torch.allclose(b_aug["fundamentals"][:, :26][f_o > 0.5], f_v[f_o > 0.5], atol=1e-5)
    # Even if noise_keys includes macro:
    aug_m = StructuredModalityNoiseAugmenter(
        modality_dropout=0.0, noise_std=0.25, noise_keys=("fundamentals", "macro")
    )
    aug_m.train()
    b_aug_m = aug_m(b_tri, generator=torch.Generator().manual_seed(78))
    torch.testing.assert_close(b_aug_m["macro"][:, 140:280], m_o, atol=0.0, rtol=0.0)
    torch.testing.assert_close(b_aug_m["macro"][:, 280:420], m_a, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        b_aug_m["macro"][:, :140][m_o < 0.5], m_v[m_o < 0.5], atol=0.0, rtol=0.0
    )

    # 3) P1-2: (a) F4 CogCore+ContextEngineer never builds set_modality_fusion; (b) F2 rejects diff!
    m_f4 = build_family_model(
        "cog_v2_10tools_soft_k3",
        family="titans_cognitive_core",
        use_set_modality_fusion=True,
        **base_dims,
    )
    assert m_f4.cognitive_core is not None and m_f4.cognitive_core.context_engineer is not None
    assert m_f4.set_modality_fusion is None

    with pytest.raises(ValueError, match="Family 3|mmarrs_titan"):
        build_family_model("nb_v2_full", use_diff_xasset_attn=True, **base_dims)

    # 4) P0-2 + P1-1: ScheduleFree CWD active steps >= 2, 2D no_decay safe, muP router LR scale
    years = [y for y in (2020, 2021, 2022, 2023) for _ in range(3)]
    sessions = []
    for s_i, _ in enumerate(years):
        bs = _synthetic_session_batch(rng, n_assets=14)
        bs["asset_ids"] = torch.arange(s_i, s_i + 14, dtype=torch.long)
        bs["target_returns"] = torch.randn(14, 6) * 0.02
        sessions.append(bs)
    split_cfg = FullHistorySplitConfig(
        train_start_year=2020,
        train_end_year=2022,
        val_year=2023,
        purge_sessions=1,
        embargo_sessions=1,
        allow_short_synthetic=True,
    )

    # P0-2: ScheduleFree + use_cautious_wd=True with wd=1.0 MUST diverge from wd=0.0 across 6 steps!
    res_sf_nowd = train_full_history_variation(
        sessions,
        years,
        variation="t15b_full_v2",
        family="titans_original",
        split_cfg=split_cfg,
        hidden_dim=16,
        epochs=2,
        lr=1e-2,
        weight_decay=0.0,
        optimizer_type="schedule_free_adamw",
        sf_warmup_steps=0,
        bptt_steps=2,
        use_cautious_wd=True,
        use_optimal_fit=False,
        modality_dropout=0.0,
        seed=233,
        device="cpu",
    )
    res_sf_cwd = train_full_history_variation(
        sessions,
        years,
        variation="t15b_full_v2",
        family="titans_original",
        split_cfg=split_cfg,
        hidden_dim=16,
        epochs=2,
        lr=1e-2,
        weight_decay=1.0,
        optimizer_type="schedule_free_adamw",
        sf_warmup_steps=0,
        bptt_steps=2,
        use_cautious_wd=True,
        use_optimal_fit=False,
        modality_dropout=0.0,
        seed=233,
        device="cpu",
    )
    assert not np.allclose(
        res_sf_nowd["val_predictions"], res_sf_cwd["val_predictions"], atol=1e-6
    ), "ScheduleFree + use_cautious_wd=True had 0 effect!"

    # P0-2 secondary: 2D no_decay params (`tabm_r`, `r_in`, `step_embed`, `shared_b1`) excluded!
    m_check = build_variation("v14c_tied_xasset_cf", tabm_k=4, **base_dims)
    no_dec_ids = fht._collect_no_decay_param_ids(m_check)
    for name, p in m_check.named_parameters():
        if any(
            tok in name
            for tok in ("tabm_r", "tabm_s", "r_in", "s_mid", "step_embed", "shared_b1", "routed_b1")
        ):
            assert id(p) in no_dec_ids, f"2D no_decay param `{name}` missing from no_decay set!"

    # P1-1: use_mup=True scales router LR by rt_cfg.router_lr_scale (1/3) across all 3 optimizers!
    captured_lrs: dict[str, list[float]] = {}
    orig_muon_init = fht._build_train_optimizer

    def _spy_opt(*a, **kw):
        opt_obj = orig_muon_init(*a, **kw)
        opt_k = kw.get("opt_kind", "unknown")
        router_lrs = [
            float(g["lr"])
            for g in opt_obj.param_groups
            if g.get("mup_role") in ("router_decay", "router_no_decay")
        ]
        captured_lrs[str(opt_k)] = router_lrs
        return opt_obj

    monkeypatch.setattr(fht, "_build_train_optimizer", _spy_opt)
    for opt_k in ("adamw_fused", "muon_hybrid", "schedule_free_adamw"):
        train_full_history_variation(
            sessions,
            years,
            variation="v14c_tied_xasset_cf",
            split_cfg=split_cfg,
            hidden_dim=16,
            epochs=1,
            lr=3e-3,
            optimizer_type=opt_k,
            use_mup=True,
            mup_base_width=16,
            use_optimal_fit=False,
            device="cpu",
        )
        assert captured_lrs[opt_k], f"No router groups captured for {opt_k}+muP"
        for r_lr in captured_lrs[opt_k]:
            assert r_lr == pytest.approx(1e-3, rel=1e-4), (
                f"{opt_k}+muP dropped router_lr_scale: got {r_lr}, expected 1e-3"
            )









