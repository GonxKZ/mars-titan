"""Pruebas TDD de Familia 3 v2: `InducedFactorCrossAttention` y `LoopedCrossAssetTransformer`.

Verifica estrictamente las especificaciones de `#129` y el veredicto arbitral `#133` (A, B, C):
  1. Reducción de parámetros `~-52%` frente a `PCTRMRecursiveLoop` desenrollado (a `D=256, K=3`)
     e invariancia iso-parámetros `K=3` vs `K=1` (`< 0.5%` diferencia) con tope duro `1 <= K <= 4`.
  2. `InducedFactorCrossAttention` `O(N * R)` estrictamente invariante a `F` (`F=5, 9, 15, None`),
     equivariante ante permutaciones de activos, invariante al orden de columnas factoriales,
     con QK-RMSNorm y compuerta `0.05` `single_zero_v1` viva en paso 0.
  3. Ciclo humano Verify -> Measure -> Counterfactual Stress-Test -> Decide-or-Re-Loop:
     - El feedback contrafactual cambia limpiamente `z_{k+1}` frente a `use_counterfactual=False`
       y frente a `placebo_mode=True` con pesos idénticos.
     - Reutilización de `pred_eff`: `<= K + 1` llamadas a `head`.
     - Compuerta monótona por activo `[N]` (0% regresiones en `eval`) y parada adaptativa
       `AdaptiveLoopHaltGate` (activos tranquilos paran antes que activos estresados/conflictivos)
       con `ponder_kl_loss` 0-D diferenciable y los 10 diagnósticos 0-D sin syncs.
  4. Cero mutaciones de buffers en `forward()` (`_SpectralLinear` + `update_spectral_buffers`),
     contracción de Banach certificada `< 1.0`, `balance_bias` congelado en `eval()`, y gradientes
     finitos en todos los parámetros entrenables.
  5. `N = 2600` en `BF16 AMP`, `torch.compile(mode="default", dynamic=True)` en paneles irregulares
     y auditoría AST `audit_zero_host_syncs == 0` sobre `mor_loom_pc.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from mars_titan.models.mmarrs.blackwell_kernels import audit_zero_host_syncs
from mars_titan.models.mmarrs.counterfactual_sim import (
    CTX_LIQ_PCT,
    CTX_VOL_PCT,
    CTX_WASS,
)
from mars_titan.models.mmarrs.granular_heads import MultiHorizonGranularHead
from mars_titan.models.mmarrs.mor_loom_pc import (
    MAX_LOOP_STEPS,
    AdaptiveLoopHaltGate,
    FinancialEnergyVerifier,
    InducedFactorCrossAttention,
    LoopedCrossAssetTransformer,
    PCTRMRecursiveLoop,
)


def _make_ctx(
    n_assets: int,
    c_dim: int = 8,
    *,
    vol_pct: float | torch.Tensor = 0.5,
    liq_pct: float | torch.Tensor = 0.5,
    wass: float | torch.Tensor = 0.5,
) -> torch.Tensor:
    ctx = torch.zeros((n_assets, c_dim), dtype=torch.float32)
    ctx[:, CTX_WASS] = wass if isinstance(wass, torch.Tensor) else float(wass)
    ctx[:, CTX_VOL_PCT] = vol_pct if isinstance(vol_pct, torch.Tensor) else float(vol_pct)
    ctx[:, CTX_LIQ_PCT] = liq_pct if isinstance(liq_pct, torch.Tensor) else float(liq_pct)
    return ctx


class _CountingHead(nn.Module):
    """Envoltorio que cuenta llamadas exactas a `MultiHorizonGranularHead`."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.call_count = 0

    def forward(
        self, h: torch.Tensor, *, exposures: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        self.call_count += 1
        return self.inner(h, exposures=exposures)


def test_iso_param_weight_tying_reduction_and_hard_cap():
    """`K=3` vs `K=1` iso-parámetros (`<0.5%`), `-52%` vs `PCTRMRecursiveLoop` y tope `<=4`."""
    assert MAX_LOOP_STEPS == 4

    # 1) Comparación directa a D=256 frente a PCTRMRecursiveLoop desenrollado (K=3)
    old_unrolled = PCTRMRecursiveLoop(
        hidden_dim=256,
        context_dim=8,
        max_loops=3,
        use_moe=True,
        n_shared_experts=2,
        n_routed_experts=6,
    )
    new_k3 = LoopedCrossAssetTransformer(
        hidden_dim=256,
        context_dim=8,
        max_loops=3,
        use_cross_asset_attn=True,
        use_moe=True,
        n_shared_experts=2,
        n_routed_experts=6,
        use_counterfactual=True,
    )
    new_k1 = LoopedCrossAssetTransformer(
        hidden_dim=256,
        context_dim=8,
        max_loops=1,
        use_cross_asset_attn=True,
        use_moe=True,
        n_shared_experts=2,
        n_routed_experts=6,
        use_counterfactual=True,
    )

    p_old = sum(p.numel() for p in old_unrolled.parameters())
    p_k3 = sum(p.numel() for p in new_k3.parameters())
    p_k1 = sum(p.numel() for p in new_k1.parameters())

    # Reducción sustancial por weight-tying (~-52% vs PCTRMRecursiveLoop desenrollado)
    reduction = 1.0 - (p_k3 / float(p_old))
    assert reduction >= 0.45, f"Reducción insuficiente: {reduction:.3f} ({p_k3} vs {p_old})"
    # Iso-parámetros K=3 vs K=1: solo embeddings de paso + routers ligeros (< 0.5%)
    rel_diff = (p_k3 - p_k1) / float(p_k1)
    assert rel_diff < 0.005, f"K=3 rompe iso-parámetros frente a K=1: {rel_diff:.4%}"

    # 2) Validación de contratos y tope duro 1 <= max_loops <= 4
    with pytest.raises(ValueError):
        LoopedCrossAssetTransformer(hidden_dim=32, context_dim=8, max_loops=0)
    with pytest.raises(ValueError):
        LoopedCrossAssetTransformer(hidden_dim=32, context_dim=8, max_loops=5)
    with pytest.raises(ValueError):
        LoopedCrossAssetTransformer(hidden_dim=32, context_dim=8, lipschitz_bound=1.0)
    with pytest.raises(ValueError):
        LoopedCrossAssetTransformer(hidden_dim=4, context_dim=8, max_loops=2)


def test_induced_factor_cross_attention_f_invariant_equivariant_and_single_zero_gate():
    """`InducedFactorCrossAttention` es estrictamente invariante a F, equivariante y puerta 0.05."""
    torch.manual_seed(17)
    n_assets, d_model = 40, 32
    attn = InducedFactorCrossAttention(
        hidden_dim=d_model,
        n_landmarks=16,
        n_heads=4,
        context_dim=8,
        init_gate=0.05,
    )
    assert abs(float(attn.out_gate.detach()) - 0.05) < 1e-6

    h = torch.randn(n_assets, d_model, requires_grad=True)
    step_emb = torch.randn(d_model)
    regime_vec = torch.randn(8)

    # 1) Estrictamente invariante a F: F = 5, 9, 15 y None en la MISMA instancia sin cambiar pesos
    for f_dim in (5, 9, 15, None):
        exp = torch.randn(n_assets, f_dim) if f_dim is not None else None
        rmt = torch.randn(n_assets, 4) if f_dim is not None else None
        out, diag = attn(
            h.detach(),
            exposures=exp,
            rmt_loadings=rmt,
            step_emb=step_emb,
            regime_vec=regime_vec,
        )
        assert out.shape == (n_assets, d_model)
        assert torch.isfinite(out).all()
        assert diag["landmark_attn_entropy"].ndim == 0
        assert float(diag["landmark_attn_entropy"]) >= 0.0

    # 2) Equivarianza por permutación de activos e invarianza al orden de columnas de `exposures`
    exp9 = torch.randn(n_assets, 9)
    rmt4 = torch.randn(n_assets, 4)
    with torch.no_grad():
        out_base, _ = attn(
            h.detach(), exposures=exp9, rmt_loadings=rmt4, step_emb=step_emb, regime_vec=regime_vec
        )
        perm_assets = torch.randperm(n_assets)
        out_perm, _ = attn(
            h.detach()[perm_assets],
            exposures=exp9[perm_assets],
            rmt_loadings=rmt4[perm_assets],
            step_emb=step_emb,
            regime_vec=regime_vec,
        )
        torch.testing.assert_close(out_base[perm_assets], out_perm, atol=1e-4, rtol=1e-4)

        perm_cols = torch.randperm(9)
        out_col_perm, _ = attn(
            h.detach(),
            exposures=exp9[:, perm_cols],
            rmt_loadings=rmt4,
            step_emb=step_emb,
            regime_vec=regime_vec,
        )
        torch.testing.assert_close(out_base, out_col_perm, atol=1e-4, rtol=1e-4)

    # 3) `single_zero_v1` (gate=0.05): casi identidad al nacer y gradiente paso-0 vivo en todo
    out_g, _ = attn(h, exposures=exp9, rmt_loadings=rmt4, step_emb=step_emb, regime_vec=regime_vec)
    rel_delta = (out_g.detach() - h.detach()).norm() / h.detach().norm().clamp_min(1e-6)
    assert float(rel_delta) < 0.25
    out_g.pow(2).mean().backward()
    assert attn.out_gate.grad is not None and float(attn.out_gate.grad.abs()) > 0.0
    for name, p in attn.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Sin gradiente en {name}"


def test_counterfactual_feedback_modifies_z_next_vs_no_cf_and_placebo_and_head_budget():
    """Feedback CF altera `z_{k+1}` vs `no_cf` y `placebo_mode`, con `<= K+1` llamadas `head`."""
    torch.manual_seed(41)
    n_assets, d_model, c_dim, f_dim = 48, 32, 8, 6
    horizons = (1, 2, 5, 10, 21, 63)

    head_raw = MultiHorizonGranularHead(hidden_dim=d_model, horizons=horizons, exposures_dim=f_dim)
    verifier = FinancialEnergyVerifier(horizons=horizons)

    loop_cf = LoopedCrossAssetTransformer(
        hidden_dim=d_model,
        context_dim=c_dim,
        max_loops=3,
        use_cross_asset_attn=True,
        use_moe=True,
        n_shared_experts=2,
        n_routed_experts=4,
        use_counterfactual=True,
        placebo_mode=False,
    )
    loop_no_cf = LoopedCrossAssetTransformer(
        hidden_dim=d_model,
        context_dim=c_dim,
        max_loops=3,
        use_cross_asset_attn=True,
        use_moe=True,
        n_shared_experts=2,
        n_routed_experts=4,
        use_counterfactual=False,
        placebo_mode=False,
    )
    # Cargar exactamente los mismos pesos entrenables en ambos para comparación limpia
    loop_no_cf.load_state_dict(loop_cf.state_dict())

    x0 = torch.randn(n_assets, d_model)
    half = n_assets // 2
    vol_pct = torch.cat([torch.full((half,), 0.95), torch.full((n_assets - half,), 0.05)])
    liq_pct = torch.cat([torch.full((half,), 0.05), torch.full((n_assets - half,), 0.95)])
    ctx = _make_ctx(n_assets, c_dim, vol_pct=vol_pct, liq_pct=liq_pct, wass=2.5)
    exposures = torch.randn(n_assets, f_dim)
    rmt = torch.randn(n_assets, 4)
    conflict = torch.cat([torch.ones(half), torch.zeros(n_assets - half)])
    ood = torch.cat([torch.full((half,), 3.0), torch.zeros(n_assets - half)])
    h_mask = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    loop_cf.train()
    loop_no_cf.train()
    counting_head = _CountingHead(head_raw)
    with torch.no_grad():
        out_cf = loop_cf(
            x0,
            ctx,
            head=counting_head,
            verifier=verifier,
            exposures=exposures,
            rmt_loadings=rmt,
            ood_energy=ood,
            horizon_mask=h_mask,
            conflict=conflict,
            placebo_mode=False,
        )
        assert counting_head.call_count == 4  # Exactamente K + 1 = 4 llamadas a head en train!

        out_no_cf = loop_no_cf(
            x0,
            ctx,
            head=head_raw,
            verifier=verifier,
            exposures=exposures,
            rmt_loadings=rmt,
            ood_energy=ood,
            horizon_mask=h_mask,
            conflict=conflict,
        )
        out_pl = loop_cf(
            x0,
            ctx,
            head=head_raw,
            verifier=verifier,
            exposures=exposures,
            rmt_loadings=rmt,
            ood_energy=ood,
            horizon_mask=h_mask,
            conflict=conflict,
            placebo_mode=True,
        )

    # Paso 0 (k=0) arranca sin feedback previo -> idéntico; pasos k>=1 divergen limpiamente!
    torch.testing.assert_close(
        out_cf["step_outputs"][0]["alpha"],
        out_no_cf["step_outputs"][0]["alpha"],
        atol=1e-5,
        rtol=1e-5,
    )
    assert not torch.allclose(
        out_cf["step_outputs"][1]["alpha"], out_no_cf["step_outputs"][1]["alpha"], atol=1e-6
    )
    assert not torch.allclose(
        out_cf["step_outputs"][1]["alpha"], out_pl["step_outputs"][1]["alpha"], atol=1e-6
    )
    assert not torch.allclose(out_cf["final_hidden"], out_no_cf["final_hidden"], atol=1e-6)
    assert not torch.allclose(out_cf["final_hidden"], out_pl["final_hidden"], atol=1e-6)

    # Respeto estricto a horizon_mask (solo h=0 activo -> e_horizon == 0)
    torch.testing.assert_close(
        out_cf["energy"]["e_horizon"], torch.zeros(()), atol=1e-7, rtol=0.0
    )

    # Verificar los 10 diagnósticos 0-D requeridos sin syncs
    diag = out_cf["diagnostics"]
    required_0d = (
        "loop_k_used_mean",
        "loop_halt_at_k1_pct",
        "cf_fragility_k0",
        "cf_fragility_final",
        "cf_sign_flip_rate_final",
        "cf_worst_dd_final",
        "loop_monotonic_accept_rate",
        "loop_step_router_js_div",
        "landmark_attn_entropy",
        "ponder_kl_loss",
    )
    for key in required_0d:
        assert key in diag, f"Falta diagnóstico requerido `{key}`"
        val = diag[key]
        assert isinstance(val, torch.Tensor) and val.ndim == 0, f"`{key}` debe ser tensor 0-D"
        assert torch.isfinite(val).all(), f"`{key}` no es finito"

    # En `eval()` (`enable_early_exit=True`): <= K llamadas `head` y 0% regresiones monótonas
    loop_cf.eval()
    counting_head.call_count = 0
    with torch.no_grad():
        out_ev = loop_cf(
            x0,
            ctx,
            head=counting_head,
            verifier=verifier,
            exposures=exposures,
            rmt_loadings=rmt,
            ood_energy=ood,
            horizon_mask=h_mask,
            conflict=conflict,
            enable_early_exit=True,
        )
    assert counting_head.call_count <= 4
    e_traj = out_ev["diagnostics"]["energy_trajectory"]
    assert float(e_traj[-1]) <= float(e_traj[0]) + 1e-5


def test_adaptive_halt_gate_calm_vs_fragile_and_ponder_kl_gradients():
    """Activos tranquilos paran antes (`k≈1`) que activos frágiles/estresados (`k≈2..3`)."""
    torch.manual_seed(55)
    n_assets, d_model, c_dim = 64, 32, 8
    half = n_assets // 2

    head = MultiHorizonGranularHead(hidden_dim=d_model, exposures_dim=4)
    verifier = FinancialEnergyVerifier()
    loop = LoopedCrossAssetTransformer(
        hidden_dim=d_model,
        context_dim=c_dim,
        max_loops=3,
        use_cross_asset_attn=True,
        use_moe=True,
        n_shared_experts=2,
        n_routed_experts=4,
        use_counterfactual=True,
    )
    assert isinstance(loop.halt_gate, AdaptiveLoopHaltGate)

    # Mitad 0: tranquila (bajo riesgo estructural, baja vol_pct, alta liq, 0 conflicto, 0 OOD)
    # Mitad 1: frágil/estresada (alta vol_pct, ilíquida, conflicto=1, alto OOD)
    x0 = torch.randn(n_assets, d_model) * 0.5
    vol_pct = torch.cat([torch.full((half,), 0.05), torch.full((n_assets - half,), 0.98)])
    liq_pct = torch.cat([torch.full((half,), 0.98), torch.full((n_assets - half,), 0.02)])
    wass = torch.cat([torch.zeros(half), torch.full((n_assets - half,), 4.0)])
    ctx = _make_ctx(n_assets, c_dim, vol_pct=vol_pct, liq_pct=liq_pct, wass=wass)
    conflict = torch.cat([torch.zeros(half), torch.ones(n_assets - half)])
    ood = torch.cat([torch.zeros(half), torch.full((n_assets - half,), 8.0)])
    exposures = torch.randn(n_assets, 4)
    rmt = torch.randn(n_assets, 4)

    loop.train()
    out = loop(
        x0,
        ctx,
        head=head,
        verifier=verifier,
        exposures=exposures,
        rmt_loadings=rmt,
        ood_energy=ood,
        conflict=conflict,
    )
    diag = out["diagnostics"]
    exp_steps = diag["expected_steps_per_asset"]
    halt_dist = diag["halting_distribution"]
    ponder_kl = out["ponder_kl_loss"]

    assert halt_dist.shape == (n_assets, 3)
    torch.testing.assert_close(halt_dist.sum(dim=-1), torch.ones(n_assets), atol=1e-5, rtol=1e-5)
    assert float(exp_steps[half:].detach().mean()) > float(exp_steps[:half].detach().mean()) + 0.35
    assert float(halt_dist[:half, 0].detach().mean()) > (
        float(halt_dist[half:, 0].detach().mean()) + 0.20
    )
    assert float(ponder_kl.detach()) >= 0.0

    # Gradientes finitos en TODOS los parámetros entrenables de LoopedCrossAssetTransformer
    loss = (
        out["final_pred"]["pred_return"].pow(2).mean()
        + 0.1 * out["energy"]["total"]
        + 0.05 * ponder_kl
        + 0.01 * out["cf_fragility_loss"]
    )
    loss.backward()
    for name, p in loop.named_parameters():
        assert p.grad is not None, f"Parámetro sin gradiente en LoopedCrossAssetTransformer: {name}"
        assert torch.isfinite(p.grad).all(), f"Gradiente no finito en {name}"


def test_zero_forward_buffer_mutations_spectral_post_step_hook_and_eval_balance_freeze():
    """Cero mutaciones de buffers en `forward()`, hook post-step y `balance_bias` congelado."""
    torch.manual_seed(77)
    n_assets, d_model, c_dim = 32, 32, 8
    head = MultiHorizonGranularHead(hidden_dim=d_model, exposures_dim=4)
    verifier = FinancialEnergyVerifier()
    loop = LoopedCrossAssetTransformer(
        hidden_dim=d_model,
        context_dim=c_dim,
        max_loops=3,
        use_cross_asset_attn=True,
        use_moe=True,
        n_shared_experts=2,
        n_routed_experts=4,
        lipschitz_bound=0.88,
        use_counterfactual=True,
    )

    x0 = torch.randn(n_assets, d_model)
    ctx = _make_ctx(n_assets, c_dim, vol_pct=0.7, liq_pct=0.3, wass=1.5)
    exposures = torch.randn(n_assets, 4)

    # 1) Snapshot de TODOS los buffers antes de `forward()` (train y eval): cero mutación!
    bufs_before = {k: v.detach().clone() for k, v in loop.named_buffers()}
    loop.train()
    out_tr = loop(x0, ctx, head=head, verifier=verifier, exposures=exposures)
    for k, v in loop.named_buffers():
        torch.testing.assert_close(v, bufs_before[k], atol=0.0, rtol=0.0)

    loop.eval()
    with torch.no_grad():
        out_ev = loop(x0, ctx, head=head, verifier=verifier, exposures=exposures)
    for k, v in loop.named_buffers():
        torch.testing.assert_close(v, bufs_before[k], atol=0.0, rtol=0.0)

    # 2) Contracción certificada < lipschitz_bound < 1.0
    assert float(out_tr["diagnostics"]["max_spectral_norm"]) < 0.88
    assert float(out_ev["diagnostics"]["max_spectral_norm"]) < 0.88

    # 3) Hook post-step en optimizer actualiza buffers espectrales tras optimizer.step()
    loop.train()
    opt = torch.optim.AdamW(list(loop.parameters()) + list(head.parameters()), lr=5e-2)
    handle = loop.register_spectral_post_step_hook(opt)
    try:
        out_step = loop(x0, ctx, head=head, verifier=verifier, exposures=exposures)
        step_loss = (
            out_step["final_pred"]["pred_return"].pow(2).mean()
            + 0.05 * out_step["ponder_kl_loss"]
        )
        step_loss.backward()
        opt.step()
    finally:
        handle.remove()

    changed = any(
        not torch.allclose(v, bufs_before[k], atol=1e-7)
        for k, v in loop.named_buffers()
        if k.endswith(".u") or k.endswith(".v")
    )
    assert changed, "update_spectral_buffers() no actualizó los buffers espectrales tras step()"

    # 4) `balance_bias` se actualiza en train() y queda estrictamente congelado en eval()
    loads_list = out_tr["diagnostics"]["expert_loads"]
    assert len(loads_list) == 3
    bb_init = [r.balance_bias.detach().clone() for r in loop.step_routers]
    loop.eval()
    loop.update_balance_biases(loads_list, lr=0.2)
    for r, b0 in zip(loop.step_routers, bb_init, strict=True):
        torch.testing.assert_close(r.balance_bias, b0, atol=0.0, rtol=0.0)

    loop.train()
    skewed_loads = [torch.tensor([0.85, 0.05, 0.05, 0.05]) for _ in range(3)]
    loop.update_balance_biases(skewed_loads, lr=0.2)
    for r, b0 in zip(loop.step_routers, bb_init, strict=True):
        assert not torch.allclose(r.balance_bias, b0, atol=1e-5)

    # 5) Seguridad BPTT multi-sesión (4 sesiones + `update_balance_biases` + 1 backward)
    loop.zero_grad()
    head.zero_grad()
    bptt_loss = torch.zeros(())
    for _s_idx in range(4):
        out_s = loop(
            torch.randn(n_assets, d_model),
            ctx,
            head=head,
            verifier=verifier,
            exposures=exposures,
        )
        bptt_loss = bptt_loss + out_s["final_pred"]["pred_return"].pow(2).mean()
        loop.update_balance_biases(out_s["diagnostics"]["expert_loads"], lr=0.05)
    bptt_loss.backward()
    for name, p in loop.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"BPTT grad roto en {name}"


def test_n2600_bf16_amp_torch_compile_and_zero_host_syncs():
    """`N=2600` BF16 AMP, `torch.compile(default, dynamic=True)` y `audit_zero_host_syncs==0`."""
    src_path = (
        Path(__file__).resolve().parents[2]
        / "src/mars_titan/models/mmarrs/mor_loom_pc.py"
    )
    audit = audit_zero_host_syncs(src_path)
    assert audit["clean"] is True
    assert audit["total_findings"] == 0
    assert audit["forward_findings"] == 0

    torch.manual_seed(123)
    d_model, c_dim, f_dim, k_rmt = 32, 8, 9, 5
    head = MultiHorizonGranularHead(hidden_dim=d_model, exposures_dim=f_dim)
    verifier = FinancialEnergyVerifier()
    loop = LoopedCrossAssetTransformer(
        hidden_dim=d_model,
        context_dim=c_dim,
        max_loops=3,
        use_cross_asset_attn=True,
        n_landmarks=16,
        use_moe=True,
        n_shared_experts=2,
        n_routed_experts=4,
        use_counterfactual=True,
    )

    # 1) N = 2600 en BF16 AMP
    n_big = 2600
    x_big = torch.randn(n_big, d_model)
    ctx_big = _make_ctx(n_big, c_dim, vol_pct=0.75, liq_pct=0.25, wass=1.6)
    exp_big = torch.randn(n_big, f_dim)
    rmt_big = torch.randn(n_big, k_rmt)
    conf_big = torch.rand(n_big)

    loop.eval()
    with torch.no_grad(), torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_big = loop(
            x_big,
            ctx_big,
            head=head,
            verifier=verifier,
            exposures=exp_big,
            rmt_loadings=rmt_big,
            conflict=conf_big,
        )
    assert out_big["final_hidden"].shape == (n_big, d_model)
    assert torch.isfinite(out_big["final_hidden"]).all()
    assert torch.isfinite(out_big["final_pred"]["alpha"]).all()
    assert torch.isfinite(out_big["diagnostics"]["cf_fragility_final"]).all()

    # 2) `torch.compile(mode="default", dynamic=True)` en paneles irregulares
    class _CompiledBundle(nn.Module):
        def __init__(self, core: nn.Module, h_mod: nn.Module, v_mod: nn.Module) -> None:
            super().__init__()
            self.core = core
            self.h_mod = h_mod
            self.v_mod = v_mod

        def forward(
            self,
            x: torch.Tensor,
            ctx: torch.Tensor,
            exp: torch.Tensor,
            rmt: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            res = self.core(
                x,
                ctx,
                head=self.h_mod,
                verifier=self.v_mod,
                exposures=exp,
                rmt_loadings=rmt,
            )
            return (
                res["final_pred"]["alpha"],
                res["ponder_kl_loss"],
                res["diagnostics"]["cf_fragility_final"],
            )

    bundle = _CompiledBundle(loop, head, verifier)
    bundle.train()
    compiled = torch.compile(bundle, mode="default", dynamic=True)
    for n_dyn in (32, 67, 128):
        x_d = torch.randn(n_dyn, d_model)
        c_d = _make_ctx(n_dyn, c_dim, vol_pct=0.6, liq_pct=0.4, wass=1.2)
        e_d = torch.randn(n_dyn, f_dim)
        r_d = torch.randn(n_dyn, k_rmt)
        alpha_o, kl_o, cf_o = compiled(x_d, c_d, e_d, r_d)
        assert alpha_o.shape == (n_dyn, 6)
        assert torch.isfinite(alpha_o).all()
        assert torch.isfinite(kl_o).all()
        assert torch.isfinite(cf_o).all()


def test_wave3c_f3_diff_induced_factor_cross_attention_and_family3_only_guard():
    """#195 P0-3a: Opt-in DiffInducedFactorCrossAttention in Family 3 ONLY + lambda==0 control."""
    from mars_titan.models.mmarrs.config import MMARRSConfig
    from mars_titan.models.mmarrs.frontier_modules_sota import DiffInducedFactorCrossAttention
    from mars_titan.models.mmarrs.model import build_variation

    torch.manual_seed(321)
    n_assets, d_model, c_dim = 32, 32, 8
    head = MultiHorizonGranularHead(hidden_dim=d_model, exposures_dim=4)
    verifier = FinancialEnergyVerifier()

    # 1) Default keeps InducedFactorCrossAttention; opt-in swaps to DiffInducedFactorCrossAttention
    loop_default = LoopedCrossAssetTransformer(
        hidden_dim=d_model, context_dim=c_dim, max_loops=2, use_cross_asset_attn=True
    )
    assert isinstance(loop_default.cross_asset_attn, InducedFactorCrossAttention)

    torch.manual_seed(322)
    loop_diff = LoopedCrossAssetTransformer(
        hidden_dim=d_model,
        context_dim=c_dim,
        max_loops=2,
        use_cross_asset_attn=True,
        use_diff_xasset_attn=True,
        diff_xasset_zero_lambda_control=False,
    )
    torch.manual_seed(322)
    loop_zero = LoopedCrossAssetTransformer(
        hidden_dim=d_model,
        context_dim=c_dim,
        max_loops=2,
        use_cross_asset_attn=True,
        use_diff_xasset_attn=True,
        diff_xasset_zero_lambda_control=True,
    )
    loop_zero.load_state_dict(loop_diff.state_dict())
    assert isinstance(loop_diff.cross_asset_attn, DiffInducedFactorCrossAttention)
    assert isinstance(loop_zero.cross_asset_attn, DiffInducedFactorCrossAttention)

    x0 = torch.randn(n_assets, d_model)
    ctx = _make_ctx(n_assets, c_dim, vol_pct=0.6, liq_pct=0.4, wass=1.2)
    exp = torch.randn(n_assets, 4)
    rmt = torch.randn(n_assets, 4)

    loop_diff.train()
    loop_zero.train()
    out_d = loop_diff(x0, ctx, head=head, verifier=verifier, exposures=exp, rmt_loadings=rmt)
    out_z = loop_zero(x0, ctx, head=head, verifier=verifier, exposures=exp, rmt_loadings=rmt)

    assert torch.isfinite(out_d["final_hidden"]).all()
    assert torch.isfinite(out_z["final_hidden"]).all()
    assert not torch.allclose(out_d["final_hidden"], out_z["final_hidden"], atol=1e-5)
    assert "diff_lambda_mean" in out_d["diagnostics"]
    assert 0.0 < float(out_d["diagnostics"]["diff_lambda_mean"]) < 2.0
    assert float(out_z["diagnostics"]["diff_lambda_mean"]) == pytest.approx(0.0, abs=1e-7)

    # Full backward pass finite across all DiffInducedFactorCrossAttention params
    loss = out_d["final_pred"]["pred_return"].pow(2).mean() + 0.05 * out_d["ponder_kl_loss"]
    loss.backward()
    for p_name, p in loop_diff.cross_asset_attn.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Bad grad on {p_name}"

    # BF16 autocast + torch.compile(default, dynamic=True)
    loop_diff.eval()
    with torch.no_grad(), torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_bf16 = loop_diff(x0, ctx, head=head, verifier=verifier, exposures=exp, rmt_loadings=rmt)
    assert torch.isfinite(out_bf16["final_hidden"]).all()

    class _DiffBundle(nn.Module):
        def __init__(self, core: nn.Module, h_m: nn.Module, v_m: nn.Module) -> None:
            super().__init__()
            self.core = core
            self.h_m = h_m
            self.v_m = v_m

        def forward(self, x: torch.Tensor, c: torch.Tensor, e: torch.Tensor, r: torch.Tensor):
            res = self.core(x, c, head=self.h_m, verifier=self.v_m, exposures=e, rmt_loadings=r)
            return res["final_pred"]["alpha"], res["diagnostics"]["diff_lambda_mean"]

    compiled = torch.compile(_DiffBundle(loop_diff, head, verifier), mode="default", dynamic=True)
    for n_dyn in (24, 41):
        ao, lam = compiled(
            torch.randn(n_dyn, d_model),
            _make_ctx(n_dyn, c_dim, vol_pct=0.5, liq_pct=0.5, wass=1.0),
            torch.randn(n_dyn, 4),
            torch.randn(n_dyn, 4),
        )
        assert ao.shape == (n_dyn, 6) and torch.isfinite(ao).all() and torch.isfinite(lam).all()

    # Family 3 ONLY guard in MMARRSConfig & build_variation
    m_f3 = build_variation(
        "v14c_tied_xasset_cf",
        hidden_dim=24,
        use_diff_xasset_attn=True,
        allow_short_synthetic=True,
    )
    assert isinstance(m_f3.recursive_core.cross_asset_attn, DiffInducedFactorCrossAttention)

    with pytest.raises(ValueError, match="Family 3|mmarrs_titan"):
        MMARRSConfig(
            variation="v02_titans_neuromod_only",
            family="titans_original",
            use_diff_xasset_attn=True,
            allow_short_synthetic=True,
        )
    with pytest.raises(ValueError, match="Family 3|mmarrs_titan"):
        MMARRSConfig(
            variation="cog_v2_10tools_soft_k3",
            family="titans_cognitive_core",
            use_diff_xasset_attn=True,
            allow_short_synthetic=True,
        )

