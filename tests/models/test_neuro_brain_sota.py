"""Pruebas TDD de la Familia 2 Independiente Neuro-IA (`StandaloneHolographicBrainModel`).

Verifica de forma estricta:
  1. `LateralInhibitionCompetition`: normalización divisiva (Carandini & Heeger), supresión
     de ruido correlacionado / decorrelación real de representaciones, estabilidad ante outliers
     100x, equivarianza por permutación y gradientes finitos (2D y 3D).
  2. `DendriticCompartmentLayer`: detección de coincidencia basal × apical (Larkum) en `einsum`
     3D sin bucles Python, conmutación de régimen sin olvido (ramas disjuntas protegidas),
     equivarianza por permutación y gradientes finitos.
  3. `ThreeFactorDopaminePlasticity`: trazas de elegibilidad causales en `t`, consolidación
     estrictamente diferida en `t+1` por RPE/IC maduro, estabilidad certificada PT1 Lyapunov
     ante tormentas de outliers 100x e invarianza ante cambios de `N` entre sesiones.
  4. `HippocampalSharpWaveReplay`: retención prioritaria de episodios de crisis (2008/2020)
     frente a decenas de sesiones tranquilas y reducción empírica de olvido catastrófico al
     entrenar con `replay_regularization_loss`.
  5. `StandaloneHolographicBrainModel`: Familia 2 independiente por defecto (`use_moe_bridge=False`,
     cero `HighSNRMoELayer` / `TitansCognitiveCore` / `ToolRouter`), soporte opcional
     `use_moe_bridge=True` (`soft` vs control `uniform` `1/E`), `max_cycles=3` por defecto y tope
     duro `1 <= max_cycles <= 4` con verificación monótona sin desfase, determinismo FP32 bit a
     bit, causalidad estricta, seguridad `bfloat16`/`float16` y auditoría AST de cero sync CPU.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from mars_titan.models.mmarrs.granular_heads import MultiHorizonGranularHead
from mars_titan.models.mmarrs.neuro_brain_sota import (
    DEFAULT_BRAIN_CYCLES,
    MAX_BRAIN_CYCLES,
    NEURO_BRAIN_ABLATIONS,
    DendriticCompartmentLayer,
    HippocampalSharpWaveReplay,
    LateralInhibitionCompetition,
    NeuroBrainConfig,
    ThreeFactorDopaminePlasticity,
    build_neuro_brain_ablation,
)


def _synthetic_brain_batch(
    rng: np.random.Generator,
    n_assets: int = 20,
    *,
    prices_dim: int = 16,
    fundamentals_dim: int = 8,
    macro_dim: int = 6,
    news_dim: int = 8,
    charts_dim: int = 8,
    physics_dim: int = 12,
    context_dim: int = 10,
    exposures_dim: int = 4,
    rmt_dim: int = 4,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    def _t(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(rng.standard_normal(shape), dtype=dtype)

    mask = torch.ones((n_assets, 5), dtype=dtype)
    mask[:, 3] = 0.0  # noticias ausentes
    mask[0::3, 1] = 0.0  # fundamentales parcialmente ausentes
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
        rmt_loadings=_t((n_assets, rmt_dim)),
        regime_vec=_t((context_dim,)),
    )


def _mean_abs_offdiag_corr(x: torch.Tensor) -> float:
    xc = x - x.mean(dim=0, keepdim=True)
    xn = xc / xc.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    corr = (xn.T @ xn) / float(max(x.shape[0], 1))
    d = corr.shape[0]
    off = corr[~torch.eye(d, dtype=torch.bool, device=x.device)]
    return float(off.abs().mean().detach())


def test_lateral_inhibition_divisive_norm_decorrelates_and_is_equivariant():
    """LateralInhibitionCompetition decorrelaciona ruido común, acota outliers y es equivariante."""
    torch.manual_seed(11)
    n_assets, d_model = 48, 24
    layer = LateralInhibitionCompetition(hidden_dim=d_model)

    # 1) Entrada con factor común fuerte (correlación entre características > 0.70)
    common = torch.randn(n_assets, 1) * 3.0
    x_corr = common.expand(n_assets, d_model) + torch.randn(n_assets, d_model) * 0.5
    x_corr.requires_grad_(True)
    out, diag = layer(x_corr, return_diagnostics=True)
    assert out.shape == (n_assets, d_model)
    assert torch.isfinite(out).all()

    corr_in = _mean_abs_offdiag_corr(x_corr.detach())
    corr_out = _mean_abs_offdiag_corr(out.detach())
    assert corr_in > 0.70
    assert corr_out < corr_in * 0.75
    assert float(diag["mean_offdiag_corr_out"].detach()) < float(
        diag["mean_offdiag_corr_in"].detach()
    )

    # 2) Compresión divisiva ante outliers 100x (ganancia acotada, sin explosión)
    with torch.no_grad():
        x_base = torch.randn(n_assets, d_model)
        x_spike = x_base * 100.0
        o_base, _ = layer(x_base)
        o_spike, _ = layer(x_spike)
        assert torch.isfinite(o_spike).all()
        assert float(o_spike.norm() / o_base.norm().clamp_min(1e-6)) < 25.0

    # 3) Equivarianza por permutación entre activos
    perm = torch.randperm(n_assets)
    with torch.no_grad():
        o1, _ = layer(x_base)
        o2, _ = layer(x_base[perm])
        torch.testing.assert_close(o1[perm], o2, atol=1e-5, rtol=1e-5)

    # 4) Soporte 3D [N, A, D] respetando máscara de áreas ausentes incluso con NaN/Inf
    x3 = torch.randn(n_assets, 5, d_model)
    x3[:, 4, :] = float("nan")
    x3[0::2, 4, :] = float("inf")
    mask3 = torch.ones(n_assets, 5)
    mask3[:, 4] = 0.0
    o3, _ = layer(x3, mask=mask3)
    assert o3.shape == (n_assets, 5, d_model)
    assert torch.isfinite(o3).all()
    torch.testing.assert_close(o3[:, 4, :], torch.zeros(n_assets, d_model), atol=1e-6, rtol=0.0)

    # 5) Gradientes finitos en todos los parámetros entrenables
    out.pow(2).mean().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"Parámetro sin gradiente en LateralInhibition: {name}"
        assert torch.isfinite(p.grad).all(), f"Gradiente no finito en {name}"
    assert x_corr.grad is not None and torch.isfinite(x_corr.grad).all()


def test_dendritic_compartment_apical_coincidence_regime_switching_and_gradients():
    """DendriticCompartmentLayer conmuta ramas por contexto apical y protege ramas inactivas."""
    torch.manual_seed(17)
    n_assets, d_model, c_dim, n_branches = 24, 24, 10, 8
    layer = DendriticCompartmentLayer(
        hidden_dim=d_model,
        apical_dim=c_dim,
        n_branches=n_branches,
        branch_dim=16,
    )

    x = torch.randn(n_assets, d_model, requires_grad=True)
    ctx_calm = torch.full((n_assets, c_dim), -2.0)
    ctx_crisis = torch.full((n_assets, c_dim), 2.0)

    out_calm, d_calm = layer(x, ctx_calm)
    out_crisis, d_crisis = layer(x, ctx_crisis)
    assert out_calm.shape == (n_assets, d_model)
    assert d_calm["apical_gates"].shape == (n_assets, n_branches)
    assert torch.isfinite(out_calm).all() and torch.isfinite(out_crisis).all()
    assert not torch.allclose(d_calm["apical_gates"], d_crisis["apical_gates"], atol=1e-3)
    assert not torch.allclose(out_calm, out_crisis, atol=1e-4)

    # Protección contra olvido: ramas apagadas por la compuerta apical reciben gradiente ~0
    layer_iso = DendriticCompartmentLayer(
        hidden_dim=d_model, apical_dim=c_dim, n_branches=n_branches, branch_dim=16
    )
    with torch.no_grad():
        layer_iso.apical_gate_net[-1].weight.zero_()
        layer_iso.apical_gate_net[-1].bias.copy_(
            torch.tensor([-12.0, -12.0, -12.0, -12.0, 12.0, 12.0, 12.0, 12.0])
        )
    o_iso, _ = layer_iso(x.detach(), ctx_crisis)
    o_iso.pow(2).mean().backward()
    grad_off = layer_iso.basal_w.grad[:4].abs().mean()
    grad_on = layer_iso.basal_w.grad[4:].abs().mean()
    assert float(grad_off) < float(grad_on) * 1e-4

    # Equivarianza por permutación (también acepta vector apical 1D [C] compartido)
    perm = torch.randperm(n_assets)
    ctx_var = torch.randn(n_assets, c_dim)
    with torch.no_grad():
        y1, _ = layer(x.detach(), ctx_var)
        y2, _ = layer(x.detach()[perm], ctx_var[perm])
        torch.testing.assert_close(y1[perm], y2, atol=1e-5, rtol=1e-5)
        y_1d, _ = layer(x.detach(), ctx_var[0])
        assert y_1d.shape == (n_assets, d_model) and torch.isfinite(y_1d).all()

    # Gradientes finitos en todos los parámetros entrenables
    layer.zero_grad()
    y_g, _ = layer(x, ctx_var)
    y_g.pow(2).mean().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"Parámetro sin gradiente en DendriticCompartmentLayer: {name}"
        assert torch.isfinite(p.grad).all(), f"Gradiente no finito en {name}"


def test_three_factor_dopamine_plasticity_causality_rpe_and_pt1_bounds():
    """Plasticidad de 3 factores: causalidad estricta, RPE maduro, cambio de N y cota PT1."""
    torch.manual_seed(23)
    n_assets, d_model, rank = 20, 24, 16
    plast = ThreeFactorDopaminePlasticity(
        hidden_dim=d_model,
        plasticity_rank=rank,
        pt1_alpha_lo=0.05,
        pt1_theta_max=0.25,
        pt1_w_max=3.0,
        huber_delta=1.0,
    )

    x_t0 = torch.randn(n_assets, d_model, requires_grad=True)
    h_t0 = torch.randn(n_assets, d_model)
    st0 = plast.init_state(device=x_t0.device, dtype=x_t0.dtype)

    # 1) En tiempo t: paso 0 recibe gradientes vivos (sin stall P0-5b) y record_eligibility no fuga
    y_virgin, d_read0 = plast.read(x_t0, st0)
    assert float((y_virgin - x_t0).detach().norm() / x_t0.detach().norm().clamp_min(1e-6)) < 0.15
    y_virgin.pow(2).mean().backward()
    for p_name in ("pre_proj.weight", "post_proj.weight", "out_proj.weight"):
        param = dict(plast.named_parameters())[p_name]
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert float(param.grad.abs().sum()) > 0.0, f"Step-0 zero-grad stall en {p_name}"
    plast.zero_grad()
    st_elig = plast.record_eligibility(x_t0.detach(), h_t0, st0)
    torch.testing.assert_close(st_elig["fast_weights"], torch.zeros(rank, rank), atol=0.0, rtol=0.0)
    with torch.no_grad():
        y_after_elig, _ = plast.read(x_t0.detach(), st_elig)
        torch.testing.assert_close(y_virgin.detach(), y_after_elig, atol=0.0, rtol=0.0)
    assert float(st_elig["eligibility"].abs().sum()) > 0.0

    # 2) En t+1: consolidación con RPE positivo vs negativo produce actualizaciones opuestas
    rpe_pos = torch.full((n_assets,), 1.5)
    rpe_neg = torch.full((n_assets,), -1.5)
    st_pos, d_pos = plast.consolidate(st_elig, matured_rpe=rpe_pos, matured_ic=torch.tensor(0.08))
    st_neg, d_neg = plast.consolidate(st_elig, matured_rpe=rpe_neg, matured_ic=torch.tensor(-0.08))
    assert float(st_pos["fast_weights"].norm()) > 1e-4
    cos_dir = torch.nn.functional.cosine_similarity(
        st_pos["fast_weights"].reshape(1, -1), st_neg["fast_weights"].reshape(1, -1)
    )
    assert float(cos_dir) < -0.5

    # 3) Lectura en t+1 con distinto número de activos (N=13 != 20) y gradientes finitos
    x_t1 = torch.randn(13, d_model, requires_grad=True)
    y_t1, _ = plast.read(x_t1, st_pos)
    assert y_t1.shape == (13, d_model)
    assert not torch.allclose(y_t1, x_t1, atol=1e-6)
    y_t1.pow(2).mean().backward()
    for name, p in plast.named_parameters():
        assert p.grad is not None, f"Parámetro sin gradiente en ThreeFactorDopamine: {name}"
        assert torch.isfinite(p.grad).all(), f"Gradiente no finito en {name}"
    assert x_t1.grad is not None and torch.isfinite(x_t1.grad).all()

    # 4) Tormenta de outliers 100x durante 40 pasos: PT1 Lyapunov mantiene norma y tasas acotadas
    st_cur = st_pos
    for step in range(40):
        spike_scale = 100.0 if step % 2 == 0 else 1.0
        xs = torch.randn(n_assets, d_model) * spike_scale
        hs = torch.randn(n_assets, d_model) * spike_scale
        st_cur = plast.record_eligibility(xs, hs, st_cur)
        rpe_s = torch.randn(n_assets) * spike_scale
        st_cur, d_step = plast.consolidate(
            st_cur, matured_rpe=rpe_s, matured_ic=torch.tensor(0.5 if step % 3 == 0 else -0.5)
        )
        assert float(d_step["alpha"]) >= 0.05 - 1e-6
        assert float(d_step["theta"]) <= 0.25 + 1e-6
        assert float(d_step["fast_weight_norm"]) <= 3.0 + 1e-4
        assert torch.isfinite(st_cur["fast_weights"]).all()
        assert torch.isfinite(st_cur["eligibility"]).all()
    assert float(d_read0["fast_weight_norm"]) == 0.0
    assert torch.isfinite(d_pos["dopamine_rpe"]) and torch.isfinite(d_neg["dopamine_rpe"])


def test_hippocampal_sharp_wave_replay_priority_crisis_retention_and_zero_forgetting():
    """SWR retiene prototipos de crisis tras decenas de sesiones calmadas y frena el olvido."""
    torch.manual_seed(31)
    d_model, c_dim, n_h = 24, 10, 6
    replay = HippocampalSharpWaveReplay(
        hidden_dim=d_model, context_dim=c_dim, n_horizons=n_h, capacity=8
    )
    head = MultiHorizonGranularHead(hidden_dim=d_model, exposures_dim=4)

    # 1) Buffer vacío: replay_loss == 0 exacto y forward sin NaN
    loss_empty = replay.replay_regularization_loss(head, batch_size=4)
    assert float(loss_empty.detach()) == 0.0
    h_in = torch.randn(10, d_model)
    h_out0, d0 = replay(h_in)
    torch.testing.assert_close(h_out0, h_in, atol=1e-6, rtol=1e-6)
    assert float(d0["filled_count"]) == 0.0

    # 2) Almacenar 2 episodios de crisis (alta sorpresa + alto |RPE|) y luego 40 sesiones calmadas
    crisis_lat_1 = torch.randn(d_model) + 3.0
    crisis_lat_2 = torch.randn(d_model) - 3.0
    crisis_ctx = torch.randn(c_dim)
    crisis_tgt_1 = torch.full((n_h,), -0.05)
    crisis_tgt_2 = torch.full((n_h,), 0.05)

    replay.store_episode(
        crisis_lat_1,
        crisis_ctx,
        crisis_tgt_1,
        surprise=torch.tensor(4.0),
        matured_rpe=torch.tensor(5.0),
    )
    replay.store_episode(
        crisis_lat_2,
        crisis_ctx,
        crisis_tgt_2,
        surprise=torch.tensor(3.5),
        matured_rpe=torch.tensor(-4.5),
    )

    for _ in range(40):
        calm_lat = torch.randn(d_model) * 0.2
        calm_tgt = torch.zeros(n_h)
        replay.store_episode(
            calm_lat,
            torch.zeros(c_dim),
            calm_tgt,
            surprise=torch.tensor(0.05),
            matured_rpe=torch.tensor(0.02),
        )

    batch = replay.sample_replay_batch(batch_size=4)
    assert bool(batch["has_replay"])
    top_prios = batch["priorities"]
    assert float(top_prios[0]) >= 8.0
    assert float(top_prios[1]) >= 7.5
    top_lats = batch["latents"][:2]
    d_c1 = (top_lats - crisis_lat_1.unsqueeze(0)).norm(dim=-1).min()
    d_c2 = (top_lats - crisis_lat_2.unsqueeze(0)).norm(dim=-1).min()
    assert float(d_c1) < 1e-5 and float(d_c2) < 1e-5

    # 3) Prueba empírica anti-olvido: entrenar cabeza en régimen calmado con vs sin replay SWR
    torch.manual_seed(99)
    head_no_replay = MultiHorizonGranularHead(hidden_dim=d_model, exposures_dim=0)
    torch.manual_seed(99)
    head_with_replay = MultiHorizonGranularHead(hidden_dim=d_model, exposures_dim=0)
    opt_no = torch.optim.Adam(head_no_replay.parameters(), lr=2e-2)
    opt_yes = torch.optim.Adam(head_with_replay.parameters(), lr=2e-2)

    crisis_batch_x = torch.stack([crisis_lat_1, crisis_lat_2], dim=0)
    crisis_batch_y = torch.stack([crisis_tgt_1, crisis_tgt_2], dim=0)
    for _ in range(35):
        for h_m, opt_m in ((head_no_replay, opt_no), (head_with_replay, opt_yes)):
            opt_m.zero_grad()
            pred_c = h_m(crisis_batch_x)["pred_return"]
            (pred_c - crisis_batch_y).pow(2).mean().backward()
            opt_m.step()

    calm_x = torch.randn(16, d_model) * 0.5 + crisis_lat_1.unsqueeze(0) * 0.5
    calm_y = torch.full((16, n_h), 0.04)
    for _ in range(40):
        opt_no.zero_grad()
        (head_no_replay(calm_x)["pred_return"] - calm_y).pow(2).mean().backward()
        opt_no.step()

        opt_yes.zero_grad()
        l_calm = (head_with_replay(calm_x)["pred_return"] - calm_y).pow(2).mean()
        l_rep = replay.replay_regularization_loss(head_with_replay, batch_size=4)
        (l_calm + 10.0 * l_rep).backward()
        opt_yes.step()

    with torch.no_grad():
        err_no = (head_no_replay(crisis_batch_x)["pred_return"] - crisis_batch_y).pow(2).mean()
        err_yes = (head_with_replay(crisis_batch_x)["pred_return"] - crisis_batch_y).pow(2).mean()
    assert float(err_yes) < float(err_no) * 0.5


def test_standalone_holographic_brain_model_ablations_moe_bridge_determinism_and_causality():
    """Familia 2: standalone puro, MoE bridge opcional (soft vs uniform), determinismo y causa."""
    assert DEFAULT_BRAIN_CYCLES == 3
    assert MAX_BRAIN_CYCLES == 4
    for expected in (
        "standalone_k1",
        "full",
        "no_dendritic",
        "no_lateral_inhibition",
        "no_three_factor",
        "no_replay",
        "bare_holographic_3l",
        "holographic_moe_soft_k3",
        "holographic_moe_uniform_k3",
    ):
        assert expected in NEURO_BRAIN_ABLATIONS

    # Validación de contrato y tope duro de ciclos 1..4
    cfg_def = NeuroBrainConfig()
    assert cfg_def.max_cycles == 3
    assert cfg_def.use_moe_bridge is False
    assert cfg_def.moe_mode == "soft"
    assert len(cfg_def.fingerprint()) == 64
    with pytest.raises(ValueError):
        NeuroBrainConfig(max_cycles=0)
    with pytest.raises(ValueError):
        NeuroBrainConfig(max_cycles=5)
    with pytest.raises(ValueError):
        NeuroBrainConfig(lipschitz_bound=1.0)
    with pytest.raises(ValueError):
        NeuroBrainConfig(moe_mode="invalid_mode")

    rng = np.random.default_rng(77)
    b_t0 = _synthetic_brain_batch(rng, n_assets=18)
    b_t1 = _synthetic_brain_batch(rng, n_assets=14)  # N distinto entre sesiones

    overrides = dict(
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
        n_cortical_areas=12,
        n_hidden_slots=4,
        n_hyperedges=6,
        n_dendritic_branches=6,
        plasticity_rank=12,
        replay_capacity=12,
        hrr_max_slots=16,
    )

    # 1) Todas las ablaciones emparejadas ejecutan y respetan aislamiento de Familia 2
    fps = set()
    for ab_name in NEURO_BRAIN_ABLATIONS:
        model = build_neuro_brain_ablation(ab_name, **overrides)
        fps.add(model.config.fingerprint())
        mod_classes = {type(m).__name__ for m in model.modules()}
        assert "TitansCognitiveCore" not in mod_classes
        assert "ToolRouter" not in mod_classes
        if not model.config.use_moe_bridge:
            assert "HighSNRMoELayer" not in mod_classes
        else:
            assert "HighSNRMoELayer" in mod_classes

        model.eval()
        with torch.no_grad():
            out = model.forward_session(b_t0)
        q = out["quantiles"]
        assert q.shape == (18, 6, 9)
        assert torch.all(q[..., 1:] > q[..., :-1])
        assert torch.isfinite(out["pred_return"]).all()
        assert len(out["cycle_outputs"]) == model.config.max_cycles
        assert float(out["diagnostics"]["lipschitz_upper_bound"]) < 1.0
        assert 0.0 < float(out["diagnostics"]["monotonic_accept_rate"]) <= 1.0 + 1e-6
        if model.config.use_moe_bridge and model.config.moe_mode == "uniform":
            loads = out["diagnostics"]["moe_expert_loads"][-1]
            torch.testing.assert_close(
                loads, torch.full_like(loads, 1.0 / float(loads.numel())), atol=1e-5, rtol=1e-5
            )
        audit = model.parameter_audit()
        assert audit["family"] == "holographic_brain"
        assert audit["total_params"] > 0
    assert len(fps) == len(NEURO_BRAIN_ABLATIONS)

    # 2) Determinismo FP32 bit a bit (tanto standalone como con MoE bridge)
    for preset in ("full", "holographic_moe_soft_k3"):
        torch.manual_seed(123)
        m1 = build_neuro_brain_ablation(preset, **overrides)
        torch.manual_seed(123)
        m2 = build_neuro_brain_ablation(preset, **overrides)
        m1.eval()
        m2.eval()
        with torch.no_grad():
            o1 = m1.forward_session(b_t0)
            o2 = m2.forward_session(b_t0)
        torch.testing.assert_close(o1["pred_return"], o2["pred_return"], atol=0.0, rtol=0.0)
        torch.testing.assert_close(o1["quantiles"], o2["quantiles"], atol=0.0, rtol=0.0)

    # 3) Aislamiento bit a bit de modalidades ausentes incluso con 1e6, NaN e Inf
    torch.manual_seed(123)
    m_pure = build_neuro_brain_ablation("full", **overrides)
    m_pure.eval()
    with torch.no_grad():
        o_clean = m_pure.forward_session(b_t0)
        b_pois = {k: v.clone() for k, v in b_t0.items()}
        b_pois["news"] = torch.full_like(b_pois["news"], float("nan"))
        b_pois["fundamentals"][0::3, :] = float("inf")
        o_pois = m_pure.forward_session(b_pois)
    assert torch.isfinite(o_pois["pred_return"]).all()
    torch.testing.assert_close(o_clean["pred_return"], o_pois["pred_return"], atol=0.0, rtol=0.0)

    # 3b) Parcial-NaN en matured_targets NO envenena memorias ni sesiones futuras (#57.2)
    y_nan_partial = torch.tensor(rng.standard_normal((18, 6)) * 0.02, dtype=torch.float32)
    y_nan_partial[0, :] = float("nan")
    y_nan_partial[5, 0] = float("nan")
    m_nan = build_neuro_brain_ablation("full", **overrides)
    m_nan.eval()
    with torch.no_grad():
        o_nan0 = m_nan.forward_session(b_t0)
        st_nan = m_nan.step_matured(
            o_nan0["state"],
            matured_targets=y_nan_partial,
            matured_preds=o_nan0["pred_return"].detach(),
        )
        o_nan1 = m_nan.forward_session(b_t1, state=st_nan)
        rep_nan = m_nan.compute_replay_loss(batch_size=4)
    assert torch.isfinite(st_nan["plasticity_state"]["fast_weights"]).all()
    assert torch.isfinite(m_nan.layer_precision).all()
    assert torch.isfinite(m_nan.hrr_hopfield.slot_vals).all()
    assert torch.isfinite(m_nan.replay_memory.proto_target).all()
    assert torch.isfinite(o_nan1["pred_return"]).all()
    assert torch.isfinite(rep_nan)

    # 3c) En eval(), step_matured JAMÁS muta `balance_bias` del puente MoE (#57.3)
    m_moe_ev = build_neuro_brain_ablation("holographic_moe_soft_k3", **overrides)
    m_moe_ev.eval()
    with torch.no_grad():
        bias_before = [blk.balance_bias.clone() for blk in m_moe_ev.moe_bridges]
        o_ev = m_moe_ev.forward_session(b_t0)
        y_ok = torch.tensor(rng.standard_normal((18, 6)) * 0.02, dtype=torch.float32)
        m_moe_ev.step_matured(
            o_ev["state"], matured_targets=y_ok, matured_preds=o_ev["pred_return"].detach()
        )
        for b0, blk in zip(bias_before, m_moe_ev.moe_bridges, strict=True):
            torch.testing.assert_close(b0, blk.balance_bias, atol=0.0, rtol=0.0)

    # 4) Macro-bucle causal step_matured: actualiza plasticidad, HRR y replay -> cambia t1
    y_mat = torch.tensor(rng.standard_normal((18, 6)) * 0.02, dtype=torch.float32)
    st1 = m_pure.step_matured(
        o_clean["state"],
        matured_targets=y_mat,
        matured_preds=o_clean["pred_return"].detach(),
    )
    assert float(st1["hrr_slots_used"]) == 2.0  # 2 prototipos/sesión (no borra todo el buffer!)
    assert float(st1["replay_slots_used"]) > 0.0
    with torch.no_grad():
        o_t1_virgin = m_pure.forward_session(b_t1, state=None)
        o_t1_matured = m_pure.forward_session(b_t1, state=st1)
    assert not torch.allclose(o_t1_virgin["pred_return"], o_t1_matured["pred_return"], atol=1e-6)

    # 4b) Verificación monótona sin desfase: candidato con incoherencia se amortigua en k=0
    m_mono = build_neuro_brain_ablation("full", **overrides)
    m_mono.eval()
    with torch.no_grad():
        orig_pc_fwd = m_mono.pc_cycle.forward
        orig_head_fwd = m_mono.head.forward
        calls = {"pc": 0}
        bad_pc_h = torch.randn(18, 24) * 5.0
        alt_shock = torch.tensor([5.0, -5.0, 5.0, -5.0, 5.0, -5.0]).unsqueeze(0)

        def _bad_cycle0(x0_in, h_in, pi_in, *, cached_weights=None):
            if calls["pc"] == 0:
                calls["pc"] += 1
                return bad_pc_h, torch.zeros_like(h_in)
            calls["pc"] += 1
            return orig_pc_fwd(x0_in, h_in, pi_in, cached_weights=cached_weights)

        def _shock_head(h_in, *, exposures=None):
            res = orig_head_fwd(h_in, exposures=exposures)
            if torch.allclose(h_in, bad_pc_h, atol=1e-5):
                res = dict(res)
                res["pred_return"] = res["pred_return"] + alt_shock
            return res

        m_mono.pc_cycle.forward = _bad_cycle0  # type: ignore[method-assign]
        m_mono.head.forward = _shock_head  # type: ignore[method-assign]
        out_bad = m_mono.forward_session(b_t0)
        acc0 = out_bad["diagnostics"]["monotonic_accepts_per_step"][0]
        assert float(acc0.mean()) < 0.05

    # 5) Gradientes finitos end-to-end en standalone Y en holographic_moe_soft_k3
    for preset in ("full", "holographic_moe_soft_k3"):
        m_tr = build_neuro_brain_ablation(preset, **overrides)
        m_tr.train()
        out_pre = m_tr.forward_session(b_t0)
        st_tr = m_tr.step_matured(
            out_pre["state"],
            matured_targets=y_mat,
            matured_preds=out_pre["pred_return"].detach(),
        )
        out_tr = m_tr.forward_session(b_t1, state=st_tr)
        rep_loss = m_tr.compute_replay_loss(batch_size=4)
        assert torch.isfinite(rep_loss) and float(rep_loss.detach()) >= 0.0
        total_loss = (
            sum(c["pred_return"].pow(2).mean() for c in out_tr["cycle_outputs"])
            + out_tr["quantiles"].pow(2).mean()
            + out_tr["tb_logits"].pow(2).mean()
            + out_tr["turn_logits"].pow(2).mean()
            + out_tr["log_vol"].pow(2).mean()
            + out_tr["epistemic_error"].pow(2).mean()
            + rep_loss
        )
        total_loss.backward()
        for name, p in m_tr.named_parameters():
            assert p.grad is not None, f"Parámetro sin gradiente ({preset}): {name}"
            assert torch.isfinite(p.grad).all(), f"Gradiente no finito ({preset}): {name}"


def test_neuro_brain_sota_bfloat16_float16_edge_cases_and_ast_zero_host_syncs():
    """Seguridad BF16/FP16 en Blackwell, caso límite N=1 y auditoría AST de cero sync CPU."""
    torch.manual_seed(83)
    rng = np.random.default_rng(83)
    overrides = dict(
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
        n_cortical_areas=12,
        n_hidden_slots=4,
        n_hyperedges=6,
        n_dendritic_branches=6,
        plasticity_rank=12,
        replay_capacity=12,
        hrr_max_slots=16,
    )

    # 1) Forward + step_matured + replay + backward completo en bfloat16 (con y sin MoE bridge)
    for preset in ("full", "holographic_moe_soft_k3"):
        model_bf16 = build_neuro_brain_ablation(preset, **overrides).to(dtype=torch.bfloat16)
        b_bf16 = _synthetic_brain_batch(rng, n_assets=14, dtype=torch.bfloat16)
        out0 = model_bf16.forward_session(b_bf16)
        assert out0["pred_return"].dtype == torch.bfloat16
        assert torch.isfinite(out0["pred_return"]).all()
        assert torch.isfinite(out0["quantiles"]).all()
        y_bf16 = torch.randn(14, 6, dtype=torch.bfloat16) * 0.02
        st_bf16 = model_bf16.step_matured(
            out0["state"], matured_targets=y_bf16, matured_preds=out0["pred_return"].detach()
        )
        out1 = model_bf16.forward_session(b_bf16, state=st_bf16)
        rep_bf16 = model_bf16.compute_replay_loss(batch_size=4)
        assert torch.isfinite(out1["pred_return"]).all()
        assert torch.isfinite(rep_bf16).all()
        (out1["pred_return"].float().pow(2).mean() + rep_bf16.float()).backward()
        assert torch.isfinite(model_bf16.head.alpha_head.weight.grad).all()

    # 2) Seguridad FP16 en componentes críticos y caso límite N=1
    lat_fp16 = LateralInhibitionCompetition(hidden_dim=24).to(dtype=torch.float16)
    dend_fp16 = DendriticCompartmentLayer(hidden_dim=24, apical_dim=10, n_branches=4).to(
        dtype=torch.float16
    )
    x_fp16 = torch.randn(10, 24, dtype=torch.float16)
    c_fp16 = torch.randn(10, 10, dtype=torch.float16)
    o_lat, _ = lat_fp16(x_fp16)
    o_dend, _ = dend_fp16(x_fp16, c_fp16)
    assert o_lat.dtype == torch.float16 and torch.isfinite(o_lat).all()
    assert o_dend.dtype == torch.float16 and torch.isfinite(o_dend).all()

    model_fp32 = build_neuro_brain_ablation("full", **overrides)
    b_one = _synthetic_brain_batch(rng, n_assets=1, dtype=torch.float32)
    with torch.no_grad():
        out_one = model_fp32.forward_session(b_one)
    assert out_one["pred_return"].shape == (1, 6) and torch.isfinite(out_one["pred_return"]).all()

    # 2b) torch.amp.autocast(bfloat16) + step_matured fuera de autocast (escenario exacto bf16_amp)
    model_amp = build_neuro_brain_ablation("holographic_moe_soft_k3", **overrides)
    b_amp = _synthetic_brain_batch(rng, n_assets=12, dtype=torch.float32)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_amp = model_amp.forward_session(b_amp)
    assert torch.isfinite(out_amp["pred_return"]).all()
    assert torch.isfinite(out_amp["quantiles"]).all()
    y_amp = torch.randn(12, 6, dtype=torch.float32) * 0.02
    # step_matured fuera de autocast sobre pesos FP32 con estado BF16 del forward (#95)
    st_amp = model_amp.step_matured(
        out_amp["state"], matured_targets=y_amp, matured_preds=out_amp["pred_return"].detach()
    )
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_amp1 = model_amp.forward_session(b_amp, state=st_amp)
        rep_amp = model_amp.compute_replay_loss(batch_size=4)
    assert torch.isfinite(out_amp1["pred_return"]).all()
    assert torch.isfinite(rep_amp).all()
    (out_amp1["pred_return"].float().pow(2).mean() + rep_amp.float()).backward()
    assert torch.isfinite(model_amp.head.alpha_head.weight.grad).all()

    # 3) Auditoría AST estricta: CERO host syncs y CERO imports de CognitiveCore/ToolRouter
    mod_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mars_titan"
        / "models"
        / "mmarrs"
        / "neuro_brain_sota.py"
    )
    source = mod_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {"item", "cpu", "numpy", "tolist", "synchronize"}
    found_syncs: list[tuple[int, str]] = []
    forbidden_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in forbidden_calls:
                found_syncs.append((node.lineno, node.func.attr))
        if isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            for nm in names:
                if nm in ("TitansCognitiveCore", "ToolRouter", "DifferentiableQuantToolbox"):
                    forbidden_imports.append(nm)
    assert found_syncs == [], f"Sincronizaciones CPU↔GPU prohibidas: {found_syncs}"
    assert forbidden_imports == [], f"Imports prohibidos en Familia 2: {forbidden_imports}"


def test_p0_5_a_crisis_swr_replay_stores_nonzero_individual_asset_targets():
    """P0-5(a): SWR almacena top-K activos de alto |RPE| con targets reales (no media 0.0)."""
    torch.manual_seed(501)
    rng = np.random.default_rng(501)
    d_model, c_dim, n_h = 24, 10, 6
    n_assets = 40

    # 1) Directo en HippocampalSharpWaveReplay con targets market-residualized (media exacta 0.0!)
    replay = HippocampalSharpWaveReplay(
        hidden_dim=d_model, context_dim=c_dim, n_horizons=n_h, capacity=16
    )
    head = MultiHorizonGranularHead(hidden_dim=d_model, exposures_dim=0)

    # Buffer vacío: pérdida 0.0 exacta pero grafo autograd vivo + forward sin romper gradientes
    loss_empty = replay.replay_regularization_loss(head, batch_size=4)
    assert float(loss_empty.detach()) == 0.0
    assert loss_empty.requires_grad
    loss_empty.backward()
    assert head.alpha_head.weight.grad is not None and torch.isfinite(
        head.alpha_head.weight.grad
    ).all()
    head.zero_grad()

    h_empty_in = torch.randn(8, d_model, requires_grad=True)
    h_empty_out, _ = replay(h_empty_in)
    torch.testing.assert_close(h_empty_out, h_empty_in, atol=1e-6, rtol=1e-6)
    h_empty_out.pow(2).mean().backward()
    for name, p in replay.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Sin grad en vacío: {name}"
    replay.zero_grad()

    # Targets de crisis estrictamente centrados (media transversal == 0.0) con extremos +-0.12
    h_crisis = torch.randn(n_assets, d_model)
    ctx_crisis = torch.randn(n_assets, c_dim)
    y_crisis = torch.randn(n_assets, n_h) * 0.01
    y_crisis[3] = 0.12
    y_crisis[11] = -0.12
    y_crisis[27] = 0.10
    y_crisis[35] = -0.10
    y_crisis = y_crisis - y_crisis.mean(dim=0, keepdim=True)
    assert float(y_crisis.mean(dim=0).abs().max()) < 1e-6

    rpe_crisis = torch.full((n_assets,), 0.1)
    rpe_crisis[3] = 9.0
    rpe_crisis[11] = -8.5
    rpe_crisis[27] = 7.5
    rpe_crisis[35] = -7.0

    filled = replay.store_episode(
        h_crisis,
        ctx_crisis,
        y_crisis,
        surprise=torch.tensor(3.0),
        matured_rpe=rpe_crisis,
        top_k=4,
    )
    assert float(filled) == 4.0
    stored_tgts = replay.proto_target[replay.proto_filled]
    assert float(stored_tgts.abs().max()) > 0.09
    assert float(stored_tgts.abs().mean()) > 0.08

    # Entrenar head con replay_regularization_loss debe acercar predicciones a +-0.10 (no a 0!)
    opt = torch.optim.Adam(head.parameters(), lr=3e-2)
    for _ in range(40):
        opt.zero_grad()
        r_loss = replay.replay_regularization_loss(head, batch_size=4)
        r_loss.backward()
        opt.step()
    with torch.no_grad():
        pred_pair = head(torch.stack([h_crisis[3], h_crisis[11]], dim=0))["pred_return"][:, 0]
    assert float(pred_pair[0]) > 0.04 and float(pred_pair[1]) < -0.04

    # 2) Integración en StandaloneHolographicBrainModel.step_matured con targets media 0
    overrides = dict(
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
        n_cortical_areas=12,
        n_hidden_slots=4,
        n_hyperedges=6,
        n_dendritic_branches=6,
        plasticity_rank=12,
        replay_capacity=16,
        hrr_max_slots=16,
    )
    model = build_neuro_brain_ablation("full", **overrides)
    b_crisis = _synthetic_brain_batch(rng, n_assets=n_assets)
    out = model.forward_session(b_crisis)
    model.step_matured(
        out["state"],
        matured_targets=y_crisis,
        matured_preds=out["pred_return"].detach(),
    )
    assert model.replay_memory is not None
    m_stored = model.replay_memory.proto_target[model.replay_memory.proto_filled]
    assert float(m_stored.abs().max()) > 0.08


def test_p0_5_b_three_factor_ragged_asset_ids_alignment():
    """P0-5(b): alineación canónica `asset_ids` en paneles ragged `N_t != N_{t-1}`."""
    torch.manual_seed(502)
    d_model, rank = 24, 12
    plast = ThreeFactorDopaminePlasticity(hidden_dim=d_model, plasticity_rank=rank)
    st0 = plast.init_state(device=torch.device("cpu"), dtype=torch.float32)

    # Sesión t-1 con 10 activos [100..109]
    ids_prev = torch.tensor([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=torch.long)
    x_prev = torch.randn(10, d_model)
    h_prev = torch.randn(10, d_model)
    st_prev = plast.record_eligibility(x_prev, h_prev, st0, asset_ids=ids_prev)

    # Sobrevivientes [107, 102, 109, 100] + 2 IPOs [500, 501] (N_t=6 != N_prev=10, desordenado!)
    surv_idx = torch.tensor([7, 2, 9, 0], dtype=torch.long)
    ids_ragged = torch.tensor([107, 500, 102, 109, 501, 100], dtype=torch.long)
    rpe_surv = torch.tensor([2.0, -1.5, 1.2, -2.5], dtype=torch.float32)
    rpe_ragged = torch.tensor([2.0, 99.0, -1.5, 1.2, -99.0, -2.5], dtype=torch.float32)

    # Referencia exacta: solo los 4 sobrevivientes en orden canónico (con mismo ic_val explícito)
    ic_fixed = torch.tensor(0.05)
    st_ref_elig = plast.record_eligibility(
        x_prev[surv_idx], h_prev[surv_idx], st0, asset_ids=ids_prev[surv_idx]
    )
    # Igualar eligibility global para aislar la rama por activo
    st_ref_elig["eligibility"] = st_prev["eligibility"].clone()
    st_ref, _ = plast.consolidate(
        st_ref_elig,
        matured_rpe=rpe_surv,
        matured_ic=ic_fixed,
        asset_ids=ids_prev[surv_idx],
    )
    st_rag, _ = plast.consolidate(
        st_prev,
        matured_rpe=rpe_ragged,
        matured_ic=ic_fixed,
        asset_ids=ids_ragged,
    )
    torch.testing.assert_close(st_rag["fast_weights"], st_ref["fast_weights"], atol=1e-5, rtol=1e-5)

    # Panel completamente disjunto (cero intersección) no crashea y mantiene pesos finitos
    st_disj, _ = plast.consolidate(
        st_prev,
        matured_rpe=torch.tensor([1.0, -1.0, 0.5]),
        matured_ic=ic_fixed,
        asset_ids=torch.tensor([900, 901, 902], dtype=torch.long),
    )
    assert torch.isfinite(st_disj["fast_weights"]).all()

    # Integración end-to-end en StandaloneHolographicBrainModel con `asset_ids` ragged
    rng = np.random.default_rng(502)
    model = build_neuro_brain_ablation(
        "full",
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
        n_cortical_areas=12,
        n_hidden_slots=4,
        n_hyperedges=6,
        n_dendritic_branches=6,
        plasticity_rank=12,
        replay_capacity=12,
        hrr_max_slots=16,
    )
    b0 = _synthetic_brain_batch(rng, n_assets=18)
    b0["asset_ids"] = torch.arange(100, 118, dtype=torch.long)
    out0 = model.forward_session(b0)
    ids_mat = torch.tensor([115, 999, 102, 110, 100, 105, 888, 117, 103, 108], dtype=torch.long)
    y_mat = torch.randn(len(ids_mat), 6) * 0.03
    st1 = model.step_matured(
        out0["state"],
        matured_targets=y_mat,
        matured_preds=out0["pred_return"][: len(ids_mat)].detach(),
        asset_ids=ids_mat,
    )
    assert torch.isfinite(st1["plasticity_state"]["fast_weights"]).all()


def test_p0_5_c_cross_session_placebo_bijective_and_breaks_market_constant_context():
    """P0-5(c): placebo cross-session biyectivo para 7|N y rompe canales constantes de mercado."""
    torch.manual_seed(503)
    rng = np.random.default_rng(503)
    overrides = dict(
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
        n_cortical_areas=12,
        n_hidden_slots=4,
        n_hyperedges=6,
        n_dendritic_branches=6,
        plasticity_rank=12,
        replay_capacity=12,
        hrr_max_slots=16,
    )
    torch.manual_seed(503)
    m_soft = build_neuro_brain_ablation(
        "holographic_moe_soft_k3", moe_mode="soft", **overrides
    )
    torch.manual_seed(503)
    m_shuf = build_neuro_brain_ablation(
        "holographic_moe_soft_k3", moe_mode="shuffled_context", **overrides
    )
    m_shuf.load_state_dict(m_soft.state_dict())
    m_soft.eval()
    m_shuf.eval()

    # N=21 (divisible por 7!) con contexto 100% constante de mercado en cada sesión
    b0 = _synthetic_brain_batch(rng, n_assets=21)
    b1 = _synthetic_brain_batch(rng, n_assets=21)
    b0["high_snr_context"] = torch.linspace(-2.0, 2.0, 10).unsqueeze(0).expand(21, 10).clone()
    b1["high_snr_context"] = torch.linspace(3.0, -3.0, 10).unsqueeze(0).expand(21, 10).clone()

    with torch.no_grad():
        o0_soft = m_soft.forward_session(b0)
        o0_shuf = m_shuf.forward_session(b0)
        # Incluso en sesión 0 con contexto constante, placebo rompe alineación de canales
        assert not torch.allclose(o0_soft["pred_return"], o0_shuf["pred_return"], atol=1e-5)

        o1_soft = m_soft.forward_session(b1, state=o0_soft["state"])
        o1_shuf = m_shuf.forward_session(b1, state=o0_shuf["state"])
        # En sesión 1 con historial cross-session, el banco reemplaza el régimen de mercado
        assert "placebo_ctx_bank" in o0_shuf["state"]
        assert not torch.allclose(o1_soft["pred_return"], o1_shuf["pred_return"], atol=1e-5)

        # Determinismo bit a bit
        o1_shuf_rep = m_shuf.forward_session(b1, state=o0_shuf["state"])
        torch.testing.assert_close(
            o1_shuf["pred_return"], o1_shuf_rep["pred_return"], atol=0.0, rtol=0.0
        )


def test_p1_2_and_p2_spectral_hook_gated_lateral_horizon_mask_and_head_reuse(monkeypatch):
    """P1-2/P2: _SpectralLinear sin mutar buffers en forward, Lateral gate, horizon_mask y head."""
    torch.manual_seed(504)
    rng = np.random.default_rng(504)
    overrides = dict(
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
        n_cortical_areas=12,
        n_hidden_slots=4,
        n_hyperedges=6,
        n_dendritic_branches=6,
        plasticity_rank=12,
        replay_capacity=12,
        hrr_max_slots=16,
    )
    model = build_neuro_brain_ablation("full", **overrides)
    model.train()
    b0 = _synthetic_brain_batch(rng, n_assets=16)

    # 1) P1-2: forward() en train() NO muta buffers u/v de _SpectralLinear
    u_before = {
        k: v.clone() for k, v in model.pc_cycle.named_buffers() if k.endswith((".u", ".v"))
    }
    assert len(u_before) == 10
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    hook = model.register_spectral_post_step_hook(opt)

    out = model.forward_session(b0)
    for k, prev in u_before.items():
        cur = dict(model.pc_cycle.named_buffers())[k]
        torch.testing.assert_close(cur, prev, atol=0.0, rtol=0.0)

    # Tras optimizer.step(), el post-step hook SÍ actualiza los buffers espectrales u/v
    out["pred_return"].pow(2).mean().backward()
    opt.step()
    changed = any(
        not torch.allclose(dict(model.pc_cycle.named_buffers())[k], prev, atol=0.0, rtol=0.0)
        for k, prev in u_before.items()
    )
    assert changed
    hook.remove()

    # 2) P2: LateralInhibitionCompetition omite _offdiag_abs_corr en train() y lo ejecuta en eval()
    import mars_titan.models.mmarrs.neuro_brain_sota as nb_mod

    corr_calls = {"n": 0}
    orig_corr = nb_mod._offdiag_abs_corr

    def _counted_corr(x: torch.Tensor) -> torch.Tensor:
        corr_calls["n"] += 1
        return orig_corr(x)

    monkeypatch.setattr(nb_mod, "_offdiag_abs_corr", _counted_corr)
    lat = LateralInhibitionCompetition(hidden_dim=24)
    x_lat = torch.randn(16, 24)
    lat.train()
    lat(x_lat)
    assert corr_calls["n"] == 0
    lat.eval()
    lat(x_lat)
    assert corr_calls["n"] == 2

    # 3) P2: _structural_energy con horizon_mask=[1,0,0,0,0,0] ignora horizontes inactivos
    model.eval()
    h_test = torch.randn(16, 24)
    with torch.no_grad():
        pred_base = model.head(h_test, exposures=b0["exposures"])
        pred_mod = {k: v.clone() for k, v in pred_base.items()}
        pred_mod["pred_return"][:, 1:] += 10.0
        pred_mod["quantiles"][:, 1:, :] += 5.0
        pred_mod["vol_pred"][:, 1:] += 3.0
        h_mask_1d = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        e_base = model._structural_energy(
            pred_base, exposures=b0["exposures"], horizon_mask=h_mask_1d
        )
        e_mod = model._structural_energy(
            pred_mod, exposures=b0["exposures"], horizon_mask=h_mask_1d
        )
        torch.testing.assert_close(e_base, e_mod, atol=1e-6, rtol=1e-6)

    # 4) P2: forward_session con max_cycles=3 llama a self.head <= 4 veces (1 + max_cycles, no 7!)
    head_calls = {"n": 0}
    orig_head = model.head.forward

    def _counted_head(h_in: torch.Tensor, *, exposures: torch.Tensor | None = None):
        head_calls["n"] += 1
        return orig_head(h_in, exposures=exposures)

    monkeypatch.setattr(model.head, "forward", _counted_head)
    with torch.no_grad():
        out_via_call = model(b0)
    assert head_calls["n"] <= model.config.max_cycles + 1
    assert torch.isfinite(out_via_call["pred_return"]).all()


# =============================================================================
# Familia 2 v2 Computational Neuroscience SOTA Tests (#128 §2 + #133 P0-4 & B)
# =============================================================================


def _v2_test_overrides(hidden_dim: int = 24, replay_capacity: int = 16) -> dict[str, int]:
    return dict(
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=hidden_dim,
        n_cortical_areas=12,
        n_hidden_slots=4,
        n_hyperedges=6,
        n_dendritic_branches=6,
        plasticity_rank=12,
        replay_capacity=replay_capacity,
        hrr_max_slots=16,
    )


def test_v2_unitary_fhrr_exact_invertibility_zero_rfft_and_compile(monkeypatch):
    """P0-4: Real-pair unitary FHRR exact invertibility, 0 rfft calls, and torch.compile clean."""
    from mars_titan.models.mmarrs.neuro_brain_sota import (
        _fhrr_bind,
        _fhrr_unbind,
        _fhrr_unitary_pair_norm,
    )

    torch.manual_seed(601)
    # 1) Exact per-pair unit modulus + unitary invertibility across scales and D
    for d_dim in (24, 128, 256):
        a = torch.randn(32, d_dim)
        # Global L2-normalized role vector (where naive real-pair FHRR shrinks ~sqrt(D/2)^2!)
        b_global = torch.nn.functional.normalize(torch.randn(32, d_dim), dim=-1)
        b_u = _fhrr_unitary_pair_norm(b_global)
        assert b_u.shape == (32, d_dim // 2, 2)
        pair_norms = b_u.norm(dim=-1)
        torch.testing.assert_close(pair_norms, torch.ones_like(pair_norms), atol=1e-5, rtol=1e-5)

        bound = _fhrr_bind(a, b_global)
        unbound = _fhrr_unbind(bound, b_global)
        torch.testing.assert_close(bound.norm(dim=-1), a.norm(dim=-1), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(unbound, a, atol=1e-5, rtol=1e-5)

        # Broadcast 1D role vector [D] against [N, D]
        role_1d = torch.randn(d_dim) * 0.01
        torch.testing.assert_close(
            _fhrr_unbind(_fhrr_bind(a, role_1d), role_1d), a, atol=1e-5, rtol=1e-5
        )

    # Zero-pair edge case: falls back to identity phase (1, 0) with finite zero gradients
    a_z = torch.randn(8, 16, requires_grad=True)
    b_z = torch.zeros(8, 16, requires_grad=True)
    out_z = _fhrr_bind(a_z, b_z)
    torch.testing.assert_close(out_z, a_z, atol=1e-6, rtol=1e-6)
    out_z.pow(2).mean().backward()
    assert a_z.grad is not None and torch.isfinite(a_z.grad).all()
    assert b_z.grad is not None and torch.isfinite(b_z.grad).all()

    # 2) ZERO torch.fft.rfft / irfft calls in nb_v2_fhrr_lgn_pi & nb_v2_full
    def _forbid_fft(*_args, **_kwargs):
        raise AssertionError("torch.fft.rfft/irfft is FORBIDDEN when use_unitary_fhrr=True!")

    monkeypatch.setattr(torch.fft, "rfft", _forbid_fft)
    monkeypatch.setattr(torch.fft, "irfft", _forbid_fft)

    rng = np.random.default_rng(601)
    b0 = _synthetic_brain_batch(rng, n_assets=20)
    b1 = _synthetic_brain_batch(rng, n_assets=28)
    y0 = torch.tensor(rng.standard_normal((20, 6)) * 0.02, dtype=torch.float32)

    for preset in ("nb_v2_fhrr_lgn_pi", "nb_v2_full"):
        model = build_neuro_brain_ablation(preset, **_v2_test_overrides())
        assert model.config.use_unitary_fhrr is True
        model.eval()
        with torch.no_grad():
            o0 = model.forward_session(b0)
            st1 = model.step_matured(
                o0["state"], matured_targets=y0, matured_preds=o0["pred_return"].detach()
            )
            o1 = model.forward_session(b1, state=st1)
        assert torch.isfinite(o0["pred_return"]).all()
        assert torch.isfinite(o1["pred_return"]).all()
        assert float(st1["hrr_slots_used"]) == 2.0

    # 3) torch.compile(mode="default", dynamic=True) clean across dynamic N (N=20 -> N=28)
    m_comp = build_neuro_brain_ablation("nb_v2_full", **_v2_test_overrides())
    m_comp.eval()
    compiled = torch.compile(m_comp, mode="default", dynamic=True)
    with torch.no_grad():
        e0 = m_comp(b0)
        c0 = compiled(b0)
        e1 = m_comp(b1)
        c1 = compiled(b1)
    torch.testing.assert_close(c0["pred_return"], e0["pred_return"], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(c1["pred_return"], e1["pred_return"], atol=1e-3, rtol=1e-3)


def test_v2_lgn_dual_input_rank_level_preservation_and_nan_isolation():
    """LGN dual [clipped_z, masked_gaussian_rank] preserves level+rank and isolates NaN/Inf."""
    from mars_titan.models.mmarrs.neuro_brain_sota import _lgn_dual_transform

    torch.manual_seed(602)
    x_base = torch.randn(32, 12) * 0.5
    x_high_vol = x_base * 3.0  # 3x cross-session vol shock preserving exact ranks

    dual_base = _lgn_dual_transform(x_base)
    dual_high = _lgn_dual_transform(x_high_vol)
    assert dual_base.shape == (32, 24)
    # Clipped level half preserves cross-session volatility difference
    assert not torch.allclose(dual_base[:, :12], dual_high[:, :12], atol=1e-2)
    # Gaussian rank half is strictly invariant to positive monotone scaling
    torch.testing.assert_close(dual_base[:, 12:], dual_high[:, 12:], atol=1e-5, rtol=1e-5)

    # Absent rows filled with NaN/Inf have ZERO impact on present rows
    mask = torch.ones(32, dtype=torch.bool)
    mask[0::4] = False
    x_clean = x_base.clone()
    x_clean[~mask] = 0.0
    x_pois = x_base.clone()
    x_pois[0::8] = float("nan")
    x_pois[4::8] = float("inf")
    d_clean = _lgn_dual_transform(x_clean, present_row_mask=mask)
    d_pois = _lgn_dual_transform(x_pois, present_row_mask=mask)
    assert torch.isfinite(d_pois).all()
    torch.testing.assert_close(d_clean, d_pois, atol=0.0, rtol=0.0)
    torch.testing.assert_close(d_pois[~mask], torch.zeros_like(d_pois[~mask]), atol=0.0, rtol=0.0)

    # N=1 edge case and partial NaN inside present rows
    d_one = _lgn_dual_transform(torch.tensor([[2.5, float("nan"), -100.0]]))
    assert d_one.shape == (1, 6) and torch.isfinite(d_one).all()
    assert float(d_one[0, 1]) == 0.0 and float(d_one[0, 4]) == 0.0
    assert float(d_one[0, 2]) == -8.0

    # End-to-end bit-exact absent-modality NaN/Inf isolation in nb_v2_full
    rng = np.random.default_rng(602)
    b0 = _synthetic_brain_batch(rng, n_assets=20)
    m = build_neuro_brain_ablation("nb_v2_full", **_v2_test_overrides())
    m.eval()
    with torch.no_grad():
        o_clean = m.forward_session(b0)
        b_pois = {k: v.clone() for k, v in b0.items()}
        b_pois["news"] = torch.full_like(b_pois["news"], float("nan"))
        b_pois["fundamentals"][0::3, :] = float("inf")
        o_pois = m.forward_session(b_pois)
    assert torch.isfinite(o_pois["pred_return"]).all()
    torch.testing.assert_close(o_clean["pred_return"], o_pois["pred_return"], atol=0.0, rtol=0.0)

    # Partial NaN inside prices & physics_v4 still yields 100% finite outputs
    b_pnan = {k: v.clone() for k, v in b0.items()}
    b_pnan["prices"][2, 3] = float("nan")
    b_pnan["physics_v4"][5, 1] = float("inf")
    with torch.no_grad():
        o_pnan = m.forward_session(b_pnan)
    assert torch.isfinite(o_pnan["pred_return"]).all()


def test_v2_active_inference_precision_bounds_and_banach_contraction():
    """Active Inference Pi_{i,t} in [0.25, 1.0] varies per asset and preserves Banach < 0.90."""
    torch.manual_seed(603)
    rng = np.random.default_rng(603)
    model = build_neuro_brain_ablation("nb_v2_fhrr_lgn_pi", **_v2_test_overrides())
    assert model.config.use_active_inference_pi is True
    assert model.active_inference_pi is not None

    # Verify Larkum Apical Counterfactuals are CUT from PC loop per #133
    assert not hasattr(model, "apical_counterfactual")
    assert not hasattr(model.pc_cycle, "apical_counterfactual")

    b0 = _synthetic_brain_batch(rng, n_assets=24)
    b0["modality_age"] = torch.rand(24, 5) * 10.0
    # Make half the assets have full modalities and half have only prices
    b0["modality_mask"][:12, :] = 1.0
    b0["modality_mask"][12:, 1:] = 0.0

    # Zero forward buffer mutation of pc_err_var_ema
    ema_before = model.pc_err_var_ema.clone()
    model.eval()
    with torch.no_grad():
        out0 = model.forward_session(b0)
    torch.testing.assert_close(model.pc_err_var_ema, ema_before, atol=0.0, rtol=0.0)

    diag = out0["diagnostics"]
    assert float(diag["active_inference_pi_min"]) >= 0.25 - 1e-6
    assert float(diag["active_inference_pi_max"]) <= 1.0 + 1e-6
    assert float(diag["active_inference_pi_std"]) > 1e-4  # genuinely per-asset, NOT scalar!
    assert float(diag["lipschitz_upper_bound"]) < 0.90

    # step_matured updates pc_err_var_ema causally without NaN
    y0 = torch.tensor(rng.standard_normal((24, 6)) * 0.02, dtype=torch.float32)
    model.step_matured(
        out0["state"], matured_targets=y0, matured_preds=out0["pred_return"].detach()
    )
    assert torch.isfinite(model.pc_err_var_ema).all()
    assert not torch.allclose(model.pc_err_var_ema, ema_before, atol=1e-7)

    # Empirical Banach contraction < 0.90 with arbitrary per-asset Pi in [0.25, 1.0]
    x0 = torch.randn(24, 24)
    pi_2d = 0.25 + 0.75 * torch.rand(24, 24)
    with torch.no_grad():
        cached = model.pc_cycle.compute_spectral_weights(update_buffers=False)
        for _ in range(10):
            h_a = torch.randn(24, 24) * 2.0
            h_b = torch.randn(24, 24) * 2.0
            na, _ = model.pc_cycle(x0, h_a, pi_2d, cached_weights=cached)
            nb, _ = model.pc_cycle(x0, h_b, pi_2d, cached_weights=cached)
            ratio = (na - nb).norm(dim=-1) / (h_a - h_b).norm(dim=-1).clamp_min(1e-6)
            assert float(ratio.max()) < 0.90


def test_v2_homeostatic_scaling_causal_bcm_and_synaptic_tag_pt1():
    """Turrigiano homeostasis + causal t-1 detached BCM + synaptic tag W_tag inside PT1."""
    from mars_titan.models.mmarrs.neuro_brain_sota import TurrigianoHomeostaticScaling

    torch.manual_seed(604)
    d_model, rank = 24, 12

    # 1) TurrigianoHomeostaticScaling: causal, equivariant, rank-preserving, anti-collapse
    homeo = TurrigianoHomeostaticScaling(hidden_dim=d_model)
    x_ref = torch.randn(20, d_model)
    y0, rate_ema = homeo(x_ref, None)
    assert y0.shape == x_ref.shape and torch.isfinite(y0).all()
    # Sustained crisis activation collapse (rms ~ 0.05) drives gain > 1.25 (bounded <= 2.0)
    r_cur = rate_ema
    for _ in range(40):
        x_low = torch.randn(20, d_model) * 0.05
        y_low, r_cur = homeo(x_low, r_cur)
    gain_low = (y_low.norm() / x_low.norm().clamp_min(1e-6)).item()
    assert 1.2 < gain_low <= 2.0 + 1e-5
    # Permutation equivariance & rank preservation per channel
    perm = torch.randperm(20)
    y_p1, _ = homeo(x_ref, r_cur)
    y_p2, _ = homeo(x_ref[perm], r_cur)
    torch.testing.assert_close(y_p1[perm], y_p2, atol=1e-6, rtol=1e-6)
    assert torch.equal(x_ref.argsort(dim=0), y_p1.argsort(dim=0))

    # 2) ThreeFactorDopaminePlasticity with causal t-1 BCM + Synaptic Tag W_tag
    plast = ThreeFactorDopaminePlasticity(
        hidden_dim=d_model,
        plasticity_rank=rank,
        pt1_alpha_lo=0.05,
        pt1_theta_max=0.25,
        pt1_w_max=3.0,
        huber_delta=1.0,
        use_homeostatic_bcm=True,
    )
    st = plast.init_state(device=torch.device("cpu"), dtype=torch.float32)
    assert "bcm_theta" in st and "w_tag" in st

    # Strict causality: record_eligibility NEVER modifies fast_weights
    x_a = torch.randn(20, d_model)
    h_a = torch.randn(20, d_model)
    y_pre, _ = plast.read(x_a, st)
    st_el = plast.record_eligibility(x_a, h_a, st)
    torch.testing.assert_close(st_el["fast_weights"], torch.zeros(rank, rank), atol=0.0, rtol=0.0)
    y_post_el, _ = plast.read(x_a, st_el)
    torch.testing.assert_close(y_pre, y_post_el, atol=0.0, rtol=0.0)
    assert float(st_el["w_tag"].abs().sum()) > 0.0
    assert float(st_el["bcm_theta"].abs().sum()) > 0.0

    # BCM eliminates positive first-orthant DC bias under repeated constant positive input
    x_const = torch.full((20, d_model), 2.0)
    h_const = torch.full((20, d_model), 2.0)
    st_bcm = plast.init_state(device=torch.device("cpu"), dtype=torch.float32)
    for _ in range(30):
        st_bcm = plast.record_eligibility(x_const, h_const, st_bcm)
    # After BCM threshold catches up to constant v_n, new Hebbian DC tag decays toward 0
    assert float(st_bcm["w_tag"].abs().max()) < 0.05

    # Synaptic tag capture gate: high |RPE| writes much more strongly than near-zero noise RPE
    st_hi, d_hi = plast.consolidate(
        st_el, matured_rpe=torch.full((20,), 2.0), matured_ic=torch.tensor(0.25)
    )
    st_lo, d_lo = plast.consolidate(
        st_el, matured_rpe=torch.full((20,), 0.01), matured_ic=torch.tensor(0.001)
    )
    assert float(d_hi["tag_capture_gate"]) > float(d_lo["tag_capture_gate"]) * 2.0
    assert float(st_hi["fast_weights"].norm()) > float(st_lo["fast_weights"].norm()) * 3.0

    # 40-step 100x outlier storm stays strictly inside PT1 Lyapunov bounds
    st_storm = st_hi
    for step in range(40):
        sc = 100.0 if step % 2 == 0 else 1.0
        st_storm = plast.record_eligibility(
            torch.randn(20, d_model) * sc, torch.randn(20, d_model) * sc, st_storm
        )
        st_storm, d_s = plast.consolidate(
            st_storm,
            matured_rpe=torch.randn(20) * sc,
            matured_ic=torch.tensor(0.5 if step % 3 == 0 else -0.5),
        )
        assert float(d_s["alpha"]) >= 0.05 - 1e-6
        assert float(d_s["theta"]) <= 0.25 + 1e-6
        assert float(d_s["fast_weight_norm"]) <= 3.0 + 1e-4
        assert float(torch.linalg.matrix_norm(st_storm["w_tag"], ord="fro")) <= 3.0 + 1e-4
        assert torch.isfinite(st_storm["fast_weights"]).all()
        assert torch.isfinite(st_storm["bcm_theta"]).all()


def test_v2_three_strata_swr_dedup_retention_and_deterministic_gumbel_topk():
    """3-strata SWR (32/16/16) with branchless dedup + deterministic GPU Gumbel-TopK."""
    torch.manual_seed(605)
    d_model, c_dim, n_h = 24, 10, 6
    replay = HippocampalSharpWaveReplay(
        hidden_dim=d_model,
        context_dim=c_dim,
        n_horizons=n_h,
        capacity=64,
        use_stratified_swr=True,
    )
    assert (replay.cap_crisis, replay.cap_surprise, replay.cap_calm) == (32, 16, 16)
    head = MultiHorizonGranularHead(hidden_dim=d_model, exposures_dim=0)

    # Empty buffer: exact 0.0 loss with live autograd graph
    loss_empty = replay.replay_regularization_loss(head, batch_size=8)
    assert float(loss_empty.detach()) == 0.0 and loss_empty.requires_grad

    # 1) Branchless dedup: 10 near-identical crisis copies occupy ONLY 1 crisis slot!
    c1 = torch.nn.functional.normalize(torch.randn(d_model), dim=0) * 3.0
    for j in range(10):
        jitter = c1 + torch.randn(d_model) * 0.01
        replay.store_episode(
            jitter,
            torch.randn(c_dim),
            torch.full((n_h,), -0.08),
            surprise=torch.tensor(4.0),
            matured_rpe=torch.tensor(5.0 + 0.01 * j),
        )
    assert int(replay.proto_filled[:32].sum()) == 1

    # Add 2 orthogonal distinct crisis prototypes -> 3 crisis slots total
    c2 = torch.nn.functional.normalize(torch.randn(d_model), dim=0) * 3.0
    c2 = c2 - (c2 @ c1) / (c1 @ c1) * c1
    c3 = torch.nn.functional.normalize(torch.randn(d_model), dim=0) * 3.0
    c3 = c3 - (c3 @ c1) / (c1 @ c1) * c1 - (c3 @ c2) / (c2 @ c2) * c2
    for lat_c, tgt_v in ((c2, 0.09), (c3, -0.07)):
        replay.store_episode(
            lat_c,
            torch.randn(c_dim),
            torch.full((n_h,), tgt_v),
            surprise=torch.tensor(3.8),
            matured_rpe=torch.tensor(4.8),
        )
    assert int(replay.proto_filled[:32].sum()) == 3

    # 2) Store 20 surprise episodes -> fills Stratum 1 (16 slots), then 40 calm -> fills Stratum 2
    for _ in range(20):
        replay.store_episode(
            torch.randn(d_model) * 1.5,
            torch.randn(c_dim),
            torch.full((n_h,), 0.02),
            surprise=torch.tensor(1.5),
            matured_rpe=torch.tensor(0.3),
        )
    for _ in range(40):
        replay.store_episode(
            torch.randn(d_model) * 0.3,
            torch.zeros(c_dim),
            torch.full((n_h,), 0.002),
            surprise=torch.tensor(0.05),
            matured_rpe=torch.tensor(0.02),
        )

    # All 3 strata retain their prototypes simultaneously!
    assert int(replay.proto_filled[:32].sum()) >= 3  # crisis retained
    assert int(replay.proto_filled[32:48].sum()) == 16  # surprise retained (not flushed by calm!)
    assert int(replay.proto_filled[48:64].sum()) == 16  # calm anchors retained
    # Crisis prototypes c1, c2, c3 are STILL in crisis stratum [0:32]
    cr_lats = replay.proto_latent[:32][replay.proto_filled[:32]]
    for c_ref in (c1, c2, c3):
        assert float((cr_lats - c_ref.unsqueeze(0)).norm(dim=-1).min()) < 0.15

    # 3) Deterministic GPU Gumbel-TopK: identical for same seed, diverse across seeds
    b_s0_a = replay.sample_replay_batch(batch_size=8, sample_seed=0)
    b_s0_b = replay.sample_replay_batch(batch_size=8, sample_seed=0)
    torch.testing.assert_close(b_s0_a["latents"], b_s0_b["latents"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(b_s0_a["weights"], b_s0_b["weights"], atol=0.0, rtol=0.0)

    seen_diff = False
    for sd in range(1, 8):
        b_sd = replay.sample_replay_batch(batch_size=8, sample_seed=sd)
        if not torch.allclose(b_s0_a["latents"], b_sd["latents"], atol=1e-5):
            seen_diff = True
            break
    assert seen_diff, "Gumbel-TopK must sample diverse prototypes instead of fixed Top-8!"

    # Replay regularization loss is finite and backpropagates cleanly
    rep_loss = replay.replay_regularization_loss(head, batch_size=8)
    assert torch.isfinite(rep_loss) and float(rep_loss.detach()) > 0.0
    rep_loss.backward()
    assert head.alpha_head.weight.grad is not None and torch.isfinite(
        head.alpha_head.weight.grad
    ).all()


def test_v2_presets_gradients_audit_zero_host_syncs_and_bf16_n2600_smoke():
    """All 4 v2 presets + full grads + audit_zero_host_syncs == 0 + BF16 AMP N=2600 smoke."""
    from mars_titan.models.mmarrs.blackwell_kernels import audit_zero_host_syncs

    # 1) All 4 new v2 presets exist alongside all 9 existing v1 presets
    v2_presets = (
        "nb_v2_fhrr_lgn_pi",
        "nb_v2_full",
        "nb_v2_noic_twin",
        "nb_v2_moe_soft_k3",
    )
    for nm in v2_presets:
        assert nm in NEURO_BRAIN_ABLATIONS
    assert len(NEURO_BRAIN_ABLATIONS) >= 13

    torch.manual_seed(606)
    rng = np.random.default_rng(606)
    b0 = _synthetic_brain_batch(rng, n_assets=18)
    b0["asset_ids"] = torch.arange(100, 118, dtype=torch.long)
    b1 = _synthetic_brain_batch(rng, n_assets=15)
    b1["asset_ids"] = torch.arange(105, 120, dtype=torch.long)
    y0 = torch.tensor(rng.standard_normal((18, 6)) * 0.02, dtype=torch.float32)
    y0[1, 0] = float("nan")  # partial NaN resilience

    for preset in v2_presets:
        m = build_neuro_brain_ablation(preset, **_v2_test_overrides())
        assert m.config.noic_pinball_twin == (preset == "nb_v2_noic_twin")
        m.train()
        out0 = m.forward_session(b0)
        st1 = m.step_matured(
            out0["state"],
            matured_targets=y0,
            matured_preds=out0["pred_return"].detach(),
            asset_ids=b0["asset_ids"],
        )
        out1 = m.forward_session(b1, state=st1)
        rep_loss = m.compute_replay_loss(batch_size=4)
        total = (
            sum(c["pred_return"].pow(2).mean() for c in out1["cycle_outputs"])
            + out1["quantiles"].pow(2).mean()
            + out1["tb_logits"].pow(2).mean()
            + out1["turn_logits"].pow(2).mean()
            + out1["log_vol"].pow(2).mean()
            + out1["epistemic_error"].pow(2).mean()
            + rep_loss
        )
        total.backward()
        for p_name, p in m.named_parameters():
            assert p.grad is not None, f"Missing grad in {preset}: {p_name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {preset}: {p_name}"

    # 2) Official AST audit_zero_host_syncs == 0 on neuro_brain_sota.py
    nb_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mars_titan"
        / "models"
        / "mmarrs"
        / "neuro_brain_sota.py"
    )
    sync_audit = audit_zero_host_syncs([nb_path])
    assert sync_audit["total_findings"] == 0, f"Host sync findings: {sync_audit['findings']}"
    assert sync_audit["clean"] is True

    # 3) BF16 AMP + N=2600 smoke test (CUDA if available, else CPU)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_big = build_neuro_brain_ablation(
        "nb_v2_full", **_v2_test_overrides(hidden_dim=32, replay_capacity=64)
    ).to(dev)
    b_big = {k: v.to(dev) for k, v in _synthetic_brain_batch(rng, n_assets=2600).items()}
    y_big = torch.randn(2600, 6, device=dev, dtype=torch.float32) * 0.02
    with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16):
        out_big = m_big.forward_session(b_big)
    assert out_big["pred_return"].shape == (2600, 6)
    assert torch.isfinite(out_big["pred_return"]).all()
    assert torch.isfinite(out_big["quantiles"]).all()
    assert torch.all(out_big["quantiles"][..., 1:] > out_big["quantiles"][..., :-1])
    st_big = m_big.step_matured(
        out_big["state"],
        matured_targets=y_big,
        matured_preds=out_big["pred_return"].detach(),
    )
    with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16):
        out_big2 = m_big.forward_session(b_big, state=st_big)
        rep_big = m_big.compute_replay_loss(batch_size=8)
    assert torch.isfinite(out_big2["pred_return"]).all()
    assert torch.isfinite(rep_big).all()
    (out_big2["pred_return"].float().pow(2).mean() + rep_big.float()).backward()
    assert torch.isfinite(m_big.head.alpha_head.weight.grad).all()


def test_141_bptt_4step_multi_session_backward_no_inplace_version_error():
    """Regresión #141: bptt_steps=4 con step_matured entre forwards sin error in-place."""
    torch.manual_seed(607)
    rng = np.random.default_rng(607)
    for preset in ("standalone_k1", "full", "nb_v2_full", "nb_v2_moe_soft_k3"):
        model = build_neuro_brain_ablation(preset, **_v2_test_overrides())
        model.train()
        state = None
        losses: list[torch.Tensor] = []
        for step in range(4):
            n_a = 16 + (step % 3) * 2
            batch = _synthetic_brain_batch(rng, n_assets=n_a)
            batch["asset_ids"] = torch.arange(100, 100 + n_a, dtype=torch.long)
            out = model.forward_session(batch, state=state)
            rep_loss = model.compute_replay_loss(batch_size=4)
            step_loss = (
                out["pred_return"].pow(2).mean()
                + out["quantiles"].pow(2).mean()
                + out["tb_logits"].pow(2).mean()
                + out["turn_logits"].pow(2).mean()
                + out["log_vol"].pow(2).mean()
                + out["epistemic_error"].pow(2).mean()
                + 0.1 * rep_loss
            )
            losses.append(step_loss)
            y_mat = torch.tensor(rng.standard_normal((n_a, 6)) * 0.02, dtype=torch.float32)
            state = model.step_matured(
                out["state"],
                matured_targets=y_mat,
                matured_preds=out["pred_return"].detach(),
                asset_ids=batch["asset_ids"],
            )
        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        for p_name, p in model.named_parameters():
            assert p.grad is not None, f"Missing BPTT grad in {preset}: {p_name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite BPTT grad in {preset}: {p_name}"


def test_wave3c_nb_v2_full_runtime_zero_sync_and_high_snr_nan_guard(monkeypatch):
    """#185 live-CUDA 15->0 sync fix (0-D indexing banned) + #187 high_snr_context NaN guard."""
    from torch.utils._python_dispatch import TorchDispatchMode

    torch.manual_seed(608)
    rng = np.random.default_rng(608)

    class _ZeroSyncDispatchGuard(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.violations: list[str] = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            fname = str(func)
            if "_local_scalar_dense" in fname or "item" in fname:
                self.violations.append(f"scalar_sync:{fname}")
            # On CUDA, basic indexing (`buf[idx0d]`) with a 0-D LongTensor triggers DtoH sync!
            if (
                "aten.index.Tensor" in fname
                and len(args) >= 2
                and isinstance(args[1], (list, tuple))
            ):
                for idx_t in args[1]:
                    if (
                        isinstance(idx_t, torch.Tensor)
                        and idx_t.ndim == 0
                        and idx_t.dtype == torch.long
                    ):
                        self.violations.append(f"zero_dim_index:{fname}")
            if "aten.one_hot" in fname and len(args) >= 1:
                idx_t = args[0]
                if isinstance(idx_t, torch.Tensor) and idx_t.ndim == 0:
                    self.violations.append(f"zero_dim_one_hot:{fname}")
            return func(*args, **kwargs)

    # Build models BEFORE monkeypatching item/cpu
    m_full = build_neuro_brain_ablation("nb_v2_full", **_v2_test_overrides(replay_capacity=32))
    m_shuf = build_neuro_brain_ablation(
        "nb_v2_moe_soft_k3", moe_mode="shuffled_context", **_v2_test_overrides(replay_capacity=32)
    )
    b0 = _synthetic_brain_batch(rng, n_assets=20)
    b0["asset_ids"] = torch.arange(100, 120, dtype=torch.long)
    b1 = _synthetic_brain_batch(rng, n_assets=18)
    b1["asset_ids"] = torch.arange(102, 120, dtype=torch.long)
    y0 = torch.tensor(rng.standard_normal((20, 6)) * 0.02, dtype=torch.float32)

    def _forbid_sync(self, *_a, **_kw):
        raise AssertionError("Forbidden runtime host-sync (.item/.cpu/.tolist/.numpy)!")

    monkeypatch.setattr(torch.Tensor, "item", _forbid_sync)
    monkeypatch.setattr(torch.Tensor, "cpu", _forbid_sync)
    monkeypatch.setattr(torch.Tensor, "tolist", _forbid_sync)
    monkeypatch.setattr(torch.Tensor, "numpy", _forbid_sync)

    outputs_to_check: list[torch.Tensor] = []
    with _ZeroSyncDispatchGuard() as guard:
        for mdl in (m_full, m_shuf):
            mdl.train()
            o0 = mdl.forward_session(b0)
            st1 = mdl.step_matured(
                o0["state"],
                matured_targets=y0,
                matured_preds=o0["pred_return"].detach(),
                asset_ids=b0["asset_ids"],
            )
            o1 = mdl.forward_session(b1, state=st1)
            rep = mdl.compute_replay_loss(batch_size=4)
            assert mdl.replay_memory is not None
            mdl.replay_memory.store_episode(
                torch.randn(24),
                torch.randn(10),
                torch.randn(6) * 0.02,
                surprise=torch.tensor(2.0),
                matured_rpe=torch.tensor(3.0),
            )
            outputs_to_check.extend([o1["pred_return"], rep])

    assert guard.violations == [], f"Detected live-CUDA sync patterns: {guard.violations}"
    for t_chk in outputs_to_check:
        assert torch.isfinite(t_chk).all()

    # #187 NaN/Inf resilience in `high_snr_context` and `regime_vec`
    monkeypatch.undo()
    b_nan = _synthetic_brain_batch(rng, n_assets=16)
    b_nan["high_snr_context"][0, 0] = float("nan")
    b_nan["high_snr_context"][3, 2] = float("inf")
    b_nan["high_snr_context"][5, 4] = float("-inf")
    b_nan["regime_vec"] = torch.full((10,), float("nan"))
    m_full.eval()
    with torch.no_grad():
        o_nan = m_full.forward_session(b_nan)
    assert torch.isfinite(o_nan["pred_return"]).all()
    assert torch.isfinite(o_nan["alpha"]).all()
    assert torch.isfinite(o_nan["quantiles"]).all()


def test_wave3c_neuro_brain_opt_in_polar_alpha_head_and_norm_kind():
    """Opt-in PolarAlphaHead (P0-1a/b/c) & norm_kind ('rmsnorm'/'dyt') on Family 2."""
    from mars_titan.models.mmarrs.dl_training_sota import DynamicTanh, RMSNormSOTA

    torch.manual_seed(609)
    rng = np.random.default_rng(609)

    # Default config stays strictly backwards-compatible
    default_cfg = NeuroBrainConfig()
    assert default_cfg.use_polar_alpha_head is False
    assert default_cfg.norm_kind == "layernorm"

    b0 = _synthetic_brain_batch(rng, n_assets=20)
    b0["exposures"] = torch.randn(20, 4) * 5.0
    b0["realized_vol"] = torch.full((20,), 0.025)

    m_polar = build_neuro_brain_ablation(
        "nb_v2_full", use_polar_alpha_head=True, **_v2_test_overrides()
    )
    assert m_polar.polar_alpha_head is not None
    m_polar.train()
    out = m_polar.forward_session(b0)
    for key in ("alpha_raw", "alpha_u", "alpha_cal", "alpha_cal_for_cal", "alpha_cal_for_pplus"):
        assert key in out, f"Missing PolarAlphaHead key `{key}` in Family 2 output"
        assert torch.isfinite(out[key]).all()
    torch.testing.assert_close(out["alpha"], out["alpha_cal"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        out["alpha"].mean(dim=0), torch.zeros(6), atol=1e-5, rtol=1e-5
    )
    # #233 P0-1: Polar pred_return MUST equal pure zero-mean alpha_cal (no untrained factor_return!)
    torch.testing.assert_close(
        out["factor_return"], torch.zeros_like(out["alpha_cal"]), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(out["pred_return"], out["alpha_cal"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        out["pred_return"].mean(dim=0), torch.zeros(6), atol=1e-5, rtol=1e-5
    )

    # P0-1c cross-detachment: alpha_cal_for_cal trains ONLY gamma; alpha_cal_for_pplus detaches!
    m_polar.zero_grad()
    out["alpha_cal_for_cal"].pow(2).mean().backward(retain_graph=True)
    assert m_polar.polar_alpha_head.raw_gamma.grad is not None
    assert float(m_polar.polar_alpha_head.raw_gamma.grad.abs().sum()) > 0.0
    assert (
        m_polar.head.alpha_head.weight.grad is None
        or float(m_polar.head.alpha_head.weight.grad.abs().sum()) == 0.0
    )

    m_polar.zero_grad()
    out["alpha_cal_for_pplus"].pow(2).mean().backward()
    assert m_polar.head.alpha_head.weight.grad is not None
    assert float(m_polar.head.alpha_head.weight.grad.abs().sum()) > 0.0
    assert (
        m_polar.polar_alpha_head.raw_gamma.grad is None
        or float(m_polar.polar_alpha_head.raw_gamma.grad.abs().sum()) == 0.0
    )

    # norm_kind="rmsnorm" and "dyt"
    for nk, cls_exp in (("rmsnorm", RMSNormSOTA), ("dyt", DynamicTanh)):
        m_nk = build_neuro_brain_ablation(
            "nb_v2_full", norm_kind=nk, **_v2_test_overrides()
        )
        assert isinstance(m_nk.fusion_norm, cls_exp)
        assert isinstance(m_nk.head.norm, cls_exp)
        o_nk = m_nk.forward_session(b0)
        assert torch.isfinite(o_nk["pred_return"]).all()
        o_nk["pred_return"].pow(2).mean().backward()
        assert torch.isfinite(m_nk.head.alpha_head.weight.grad).all()

