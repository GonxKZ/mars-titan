"""Suite TDD para Family 1 (`titans_original` v2): `DualAxisTitansMemory` y `granular_heads` v2.

Cubre estrictamente los contratos aprobados de `#128` §1 y `#133` (P0-3, P0-6 y §B):
  1. `DualAxisTitansMemory`:
     - Recuperación asociativa exacta por sistema normal proximal FP32 (`solve_ex`, transpuesta).
     - Ortogonalización transversal CSOS y guardia anti-autotrampa (`val_i.detach()` en Axis 1).
     - Contracción de Lyapunov `||D^+||_F <= (1 - alpha_h) ||D||_F`, `kappa_2 <= 1 + theta_max`
       y estabilidad acotada `<= M_max` ante shocks 100x.
     - Rechazo redescendiente Tukey biweight (`c = 4.685`) sobre outliers `20*sigma` y máscaras CN.
     - Persistencia irregular `asset_ids`, dedup sesión 0 sin `torch.unique`, claves compatibles
       (`fast`, `fast_mom`, `medium`, `slow`, `asset_ids`, `M_mkt`, `M_mkt_slow`, `M_mkt_mom`,
       `last_ne_signal`), `compact_long_running_state` e interop con `long_running_harness`.
     - Gradientes BPTT vivos y finitos en `q_proj`, `k_proj`, `v_proj`, `out_proj`,
       `ach_gain` y `ne_gain`.
  2. `MultiHorizonGranularHead` (TabM `K=8` solo cabeza) + `MMARRSMultiTaskLoss` v2:
     - Retrocompatibilidad con `tabm_k=1`, `use_gaussian_rank_targets=False`, `w_stab_downside=0.0`.
     - Diversidad TabM `K=8` + supervisión vectorizada IC+Huber sin multiplicar FISTA/CRPS.
     - IC neutralizado con rangos gaussianos promediados en empates CN, NaNs, `liqw3`,
       `horizon_mask` y protección FM-7 (Huber/CRPS/Sharpe sobre targets crudos).
     - Semivarianza bajista `stab05` con EMA causal: penalidad cero en picos alcistas de IC.
  3. Seguridad bajo `autocast(bfloat16)` y auditoría estática `audit_zero_host_syncs == 0`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from mars_titan.models.mmarrs import granular_heads, neuromod_memory
from mars_titan.models.mmarrs.blackwell_kernels import audit_zero_host_syncs
from mars_titan.models.mmarrs.granular_heads import (
    MMARRSMultiTaskLoss,
    MultiHorizonGranularHead,
)
from mars_titan.models.mmarrs.long_running_harness import _sanitize_causal_state_masked
from mars_titan.models.mmarrs.neuromod_memory import (
    DualAxisTitansMemory,
    NeuromodulatedTitansMemory,
    compact_long_running_state,
)


def test_dual_axis_exact_matrix_solve_associative_retrieval_and_csos():
    """Axis 1 resuelve la ecuación normal proximal exacta, recupera k->M*k y elimina sesgo CSOS."""
    torch.manual_seed(101)
    hidden_dim, n_heads = 32, 4
    d_h = hidden_dim // n_heads
    # 2*d_h pares opuestos por bloque ortonormal -> G = K^T C K = (1/d_h) I y media(K) = 0
    n_blocks = 4
    n_assets = n_blocks * 2 * d_h
    mem = DualAxisTitansMemory(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        pt1_alpha_lo=0.02,
        pt1_theta_max=0.25,
        huber_delta=10.0,  # región lineal Huber para comprobar solución proximal exacta
        m_max=20.0,
        use_triton=False,
        use_fused_blackwell_kernels=False,
    )

    # Fijar q_proj == k_proj == I y v_proj == 0 para aislar la lectura asociativa pura q @ M^T
    with torch.no_grad():
        mem.k_proj.weight.copy_(torch.eye(hidden_dim))
        mem.k_proj.bias.zero_()
        mem.q_proj.weight.copy_(torch.eye(hidden_dim))
        mem.q_proj.bias.zero_()
        mem.v_proj.weight.zero_()
        mem.v_proj.bias.zero_()

    # Claves isotrópicas (+/- columnas ortonormales por cabeza) y mapa lineal verdadero M*
    k_blocks = []
    for _ in range(n_blocks):
        q_orth, _ = torch.linalg.qr(torch.randn(n_heads, d_h, d_h))  # [H, d_h, d_h]
        k_blocks.append(q_orth)
        k_blocks.append(-q_orth)
    k_unit = torch.cat(k_blocks, dim=1)  # [H, N, d_h], ||k_{i,h}||_2 = 1 y mean_i = 0
    x = k_unit.transpose(0, 1).contiguous().view(n_assets, hidden_dim)

    m_star = torch.randn(n_heads, d_h, d_h) * 0.4
    v_star_h = torch.bmm(k_unit, m_star.transpose(-1, -2))  # [H, N, d_h]
    target_hint = v_star_h.transpose(0, 1).contiguous().view(n_assets, hidden_dim)

    # 1) Verificación algebraica exacta del residuo de la ecuación normal proximal en FP32
    m_prev_probe = torch.randn(n_heads, d_h, d_h) * 0.2
    c_unif = torch.full((n_assets,), 1.0 / n_assets)
    alpha_h = torch.full((n_heads, 1, 1), 0.05)
    theta_h = torch.full((n_heads, 1, 1), 0.25)
    m_sol, cond_ub = mem._proximal_cohort_step(
        m_prev_probe, k_unit, v_star_h, c_unif, alpha_h, theta_h
    )
    assert float(cond_ub) <= 1.0 + mem.pt1_theta_max + 1e-6
    # Comprobar residuo directo: ||(I + theta K^T C K) M_sol^T - RHS||_F < 1e-5
    c_k = k_unit * c_unif.view(1, n_assets, 1)
    gram = torch.bmm(k_unit.transpose(-1, -2), c_k)
    eye = torch.eye(d_h).unsqueeze(0)
    lhs = torch.bmm(eye + theta_h * gram, m_sol.transpose(-1, -2))
    pred_p = torch.bmm(k_unit, m_prev_probe.transpose(-1, -2))
    e_raw = v_star_h - pred_p  # dentro de huber_delta=10 -> sin recorte
    e_tilde = e_raw - (c_unif.view(1, n_assets, 1) * e_raw).sum(dim=1, keepdim=True)
    rhs = (1.0 - alpha_h) * m_prev_probe.transpose(-1, -2) + theta_h * torch.bmm(
        c_k.transpose(-1, -2), pred_p + e_tilde
    )
    assert float((lhs - rhs).norm()) < 1e-5

    # 2) Iterar `update()` con el mapa lineal M*: `M_mkt` recupera la dirección exacta de M*
    st0 = mem.init_state(n_assets, device=x.device, dtype=torch.float32)
    st = st0
    for _ in range(60):
        st, diag_i = mem.update(
            x,
            st,
            ach_signal=torch.zeros(n_assets, 1),
            ne_signal=torch.tensor([-6.0]),
            target_hint=target_hint,
            detach_state=True,
        )
    assert "M_mkt" in st and st["M_mkt"].shape == (n_heads, d_h, d_h)
    assert "M_mkt_slow" in st and st["M_mkt_slow"].shape == (n_heads, d_h, d_h)
    assert "M_mkt_mom" in st and st["M_mkt_mom"].shape == (n_heads, d_h, d_h)
    assert "last_ne_signal" in st
    assert float(diag_i["cond_upper_bound"]) <= 1.0 + mem.pt1_theta_max + 1e-6

    q_test_raw = torch.randn(40, hidden_dim)
    q_test_h = F.normalize(q_test_raw.view(40, n_heads, d_h).transpose(0, 1), dim=-1, eps=1e-6)
    y_true_h = torch.bmm(q_test_h, m_star.transpose(-1, -2))
    y_hat_h = torch.bmm(q_test_h, st["M_mkt"].transpose(-1, -2))
    cos_sim = F.cosine_similarity(y_hat_h.reshape(-1), y_true_h.reshape(-1), dim=0)
    assert float(cos_sim) > 0.99, f"Recuperación asociativa M_mkt pobre: cos={float(cos_sim):.4f}"

    # 3) CSOS: añadir un sesgo constante de mercado `b` a todos los activos no altera M_mkt
    market_bias = torch.randn(1, hidden_dim) * 0.8
    st_clean, _ = mem.update(
        x,
        st0,
        ach_signal=torch.zeros(n_assets, 1),
        ne_signal=torch.tensor([-4.0]),
        target_hint=target_hint,
    )
    st_biased, _ = mem.update(
        x,
        st0,
        ach_signal=torch.zeros(n_assets, 1),
        ne_signal=torch.tensor([-4.0]),
        target_hint=target_hint + market_bias,
    )
    torch.testing.assert_close(st_clean["M_mkt"], st_biased["M_mkt"], atol=1e-5, rtol=1e-4)

    # 4) Guardia anti-autotrampa P0-3: backprop SOLO desde M_mkt jamás fluye hacia v_proj
    mem.zero_grad(set_to_none=True)
    hint_live = target_hint.clone().requires_grad_(True)
    st_grad, _ = mem.update(
        x,
        st0,
        ach_signal=torch.zeros(n_assets, 1),
        ne_signal=torch.tensor([0.0]),
        target_hint=hint_live,
        detach_state=False,
    )
    st_grad["M_mkt"].pow(2).sum().backward()
    assert mem.v_proj.weight.grad is None, "Autotrampa detectada: M_mkt retropropagó a v_proj!"
    assert mem.k_proj.weight.grad is not None and torch.isfinite(mem.k_proj.weight.grad).all()
    assert hint_live.grad is not None and float(hint_live.grad.abs().sum()) > 0.0


def test_dual_axis_lyapunov_contraction_and_100x_shock_stability():
    """Contracción exacta de Lyapunov ||D^+||_F <= (1 - alpha_h) ||D||_F y estabilidad 100x."""
    torch.manual_seed(202)
    hidden_dim, n_heads, n_assets = 24, 4, 32
    d_h = hidden_dim // n_heads
    mem = DualAxisTitansMemory(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        pt1_alpha_lo=0.08,
        pt1_eta_hi=0.95,
        pt1_theta_max=0.25,
        huber_delta=1.0,
        m_max=8.0,
        use_triton=False,
        use_fused_blackwell_kernels=False,
    )

    # 1) Contracción exacta sobre dos estados iniciales distintos (claves centradas, Huber amplio)
    mem_lin = DualAxisTitansMemory(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        pt1_alpha_lo=0.10,
        pt1_theta_max=0.25,
        huber_delta=50.0,
        m_max=50.0,
        use_triton=False,
        use_fused_blackwell_kernels=False,
    )
    with torch.no_grad():
        mem_lin.k_proj.weight.copy_(torch.eye(hidden_dim))
        mem_lin.k_proj.bias.zero_()
    x_half = torch.randn(n_assets // 2, hidden_dim)
    x_cent = torch.cat([x_half, -x_half], dim=0)
    hint = torch.randn(n_assets, hidden_dim) * 0.2
    st_a = mem_lin.init_state(n_assets, device=x_cent.device, dtype=torch.float32)
    st_b = mem_lin.init_state(n_assets, device=x_cent.device, dtype=torch.float32)
    st_a["M_mkt"] = torch.randn(n_heads, d_h, d_h) * 0.5
    st_b["M_mkt"] = st_a["M_mkt"] + torch.randn(n_heads, d_h, d_h) * 0.4

    diff_before = (st_a["M_mkt"] - st_b["M_mkt"]).norm(dim=(-2, -1))
    k_h = F.normalize(x_cent.view(n_assets, n_heads, d_h).transpose(0, 1), dim=-1, eps=1e-6)
    v_h = hint.view(n_assets, n_heads, d_h).transpose(0, 1)
    c_unif = torch.full((n_assets,), 1.0 / n_assets)
    alpha_h = torch.full((n_heads, 1, 1), 0.15)
    theta_h = torch.full((n_heads, 1, 1), 0.20)
    m_a_next, cond_ub = mem_lin._proximal_cohort_step(
        st_a["M_mkt"], k_h, v_h, c_unif, alpha_h, theta_h
    )
    m_b_next, _ = mem_lin._proximal_cohort_step(
        st_b["M_mkt"], k_h, v_h, c_unif, alpha_h, theta_h
    )
    diff_after = (m_a_next - m_b_next).norm(dim=(-2, -1))
    assert float(cond_ub) <= 1.0 + 0.20 + 1e-6
    assert torch.all(diff_after <= (1.0 - 0.15) * diff_before + 1e-5)

    # 2) Estabilidad acotada ante shocks 100x repetidos + asimetría ACh/NE + persistencia NE
    st = mem.init_state(n_assets, device=x_cent.device, dtype=torch.float32)
    for step in range(25):
        scale = 100.0 if step % 3 == 0 else 1.0
        x_s = torch.randn(n_assets, hidden_dim) * scale
        h_s = torch.randn(n_assets, hidden_dim) * scale
        st, d_step = mem.update(
            x_s,
            st,
            ach_signal=torch.ones(n_assets, 1),
            ne_signal=torch.tensor([2.0 if step % 3 == 0 else -1.0]),
            target_hint=h_s,
            detach_state=True,
        )
        assert float(d_step["alpha_min"]) >= mem.pt1_alpha_lo - 1e-6
        assert float(d_step["theta_max"]) <= mem.pt1_theta_max + 1e-6
        head_norms = st["M_mkt"].norm(dim=(-2, -1))
        assert torch.all(head_norms <= mem.m_max + 1e-5)

    for key in ("M_mkt", "M_mkt_slow", "M_mkt_mom", "fast", "medium", "slow"):
        assert torch.isfinite(st[key]).all(), f"NaN/Inf en estado {key} tras shocks 100x"

    # ACh alto amortigua escritura theta; NE alto acelera olvido alpha y persiste last_ne_signal
    x_probe = torch.randn(n_assets, hidden_dim)
    _, d_ach_lo = mem.update(
        x_probe, st, ach_signal=torch.zeros(n_assets, 1), ne_signal=torch.zeros(1)
    )
    _, d_ach_hi = mem.update(
        x_probe, st, ach_signal=torch.full((n_assets, 1), 6.0), ne_signal=torch.zeros(1)
    )
    assert float(d_ach_hi["theta_mean"]) < float(d_ach_lo["theta_mean"]) * 0.75

    st_calm, d_ne_lo = mem.update(
        x_probe, st, ach_signal=torch.zeros(n_assets, 1), ne_signal=torch.tensor([-3.0])
    )
    st_shock, d_ne_hi = mem.update(
        x_probe, st, ach_signal=torch.zeros(n_assets, 1), ne_signal=torch.tensor([4.0])
    )
    assert float(d_ne_hi["alpha_mean"]) > float(d_ne_lo["alpha_mean"]) + 0.05
    # `read()` con ne_signal=None lee `last_ne_signal` -> ne_slow_shift ya NO queda congelado!
    _, r_calm = mem.read(x_probe, st_calm, ne_signal=None)
    _, r_shock = mem.read(x_probe, st_shock, ne_signal=None)
    assert float(r_shock["ne_slow_shift"]) > float(r_calm["ne_slow_shift"]) + 0.4


def test_dual_axis_tukey_biweight_20sigma_outlier_rejection_and_write_mask():
    """Tukey biweight (c=4.685) redescinde outliers 20*sigma a peso exacto 0 en Axis 1 y Axis 2."""
    torch.manual_seed(303)
    hidden_dim, n_assets = 24, 48
    mem = DualAxisTitansMemory(
        hidden_dim=hidden_dim,
        n_heads=4,
        tukey_c=4.685,
        huber_delta=1.0,
        use_triton=False,
        use_fused_blackwell_kernels=False,
    )
    x = torch.randn(n_assets, hidden_dim) * 0.5
    hint_clean = torch.randn(n_assets, hidden_dim) * 0.25
    st0 = mem.init_state(n_assets, device=x.device, dtype=torch.float32)

    # Inyectar outlier 20*sigma (y luego 200*sigma) en el activo 0
    hint_20s = hint_clean.clone()
    hint_20s[0] = hint_20s[0] + 25.0
    hint_200s = hint_clean.clone()
    hint_200s[0] = hint_200s[0] + 250.0

    st_20s, d_20s = mem.update(
        x, st0, ach_signal=torch.zeros(n_assets, 1), ne_signal=torch.zeros(1), target_hint=hint_20s
    )
    st_200s, _ = mem.update(
        x, st0, ach_signal=torch.zeros(n_assets, 1), ne_signal=torch.zeros(1), target_hint=hint_200s
    )

    # Peso mínimo Tukey debe ser exactamente 0.0 para el outlier
    assert float(d_20s["tukey_min_weight"]) == pytest.approx(0.0, abs=1e-7)
    assert float(d_20s["tukey_zero_frac"]) >= 1.0 / n_assets - 1e-6
    # Al redescender a peso 0, outliers +25 y +250 dejan idénticos M_mkt y fast[0]==0
    torch.testing.assert_close(st_20s["M_mkt"], st_200s["M_mkt"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(st_20s["fast"][0], torch.zeros(hidden_dim), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(st_200s["fast"][0], torch.zeros(hidden_dim), atol=1e-6, rtol=1e-6)

    # Máscara CN de suspensión / limit-lock (`write_mask=0` en todos): seguro y finito
    mask_all_zero = torch.zeros(n_assets)
    st_halted, _ = mem.update(x, st_20s, write_mask=mask_all_zero, target_hint=hint_clean)
    assert torch.isfinite(st_halted["M_mkt"]).all()
    assert torch.isfinite(st_halted["fast"]).all()


def test_dual_axis_ragged_asset_ids_dedup_compat_keys_and_compaction():
    """Paneles ragged, dedup sesión 0, IPO con readout M_mkt, compactación y harness interop."""
    torch.manual_seed(404)
    hidden_dim = 16
    mem = DualAxisTitansMemory(
        hidden_dim=hidden_dim,
        n_heads=4,
        use_triton=False,
        use_fused_blackwell_kernels=False,
    )

    # 1) Sesión 0 con IDs duplicados: deduplicación sin `torch.unique` ni host-syncs
    dup_ids = torch.tensor([30, 10, 30, 20], dtype=torch.long)
    st_dup, _ = mem.update(
        torch.randn(4, hidden_dim),
        None,
        ach_signal=torch.ones(4, 1) * 0.2,
        ne_signal=torch.tensor([0.1]),
        target_hint=torch.randn(4, hidden_dim) * 0.5,
        asset_ids=dup_ids,
    )
    expected_keys = {
        "fast",
        "fast_mom",
        "medium",
        "slow",
        "asset_ids",
        "M_mkt",
        "M_mkt_slow",
        "M_mkt_mom",
        "last_ne_signal",
    }
    assert expected_keys.issubset(st_dup.keys())
    real_ids = st_dup["asset_ids"][st_dup["asset_ids"] < (1 << 61)]
    assert real_ids.tolist() == [10, 20, 30]

    # 2) Paneles ragged N=6 -> N=8: pausados preservan `slow`; IPO recibe readout M_mkt inmediato
    ids0 = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.long)
    st0, _ = mem.update(
        torch.randn(6, hidden_dim),
        None,
        ach_signal=torch.ones(6, 1) * 0.1,
        ne_signal=torch.tensor([0.0]),
        target_hint=torch.randn(6, hidden_dim) * 0.8,
        asset_ids=ids0,
    )
    idx50 = int((st0["asset_ids"] == 50).nonzero(as_tuple=False).item())
    slow50_before = st0["slow"][idx50].detach().clone()
    assert float(st0["M_mkt"].detach().norm()) > 1e-4

    ids1 = torch.tensor([90, 30, 70, 10, 100, 60, 80, 40], dtype=torch.long)
    x1 = torch.randn(8, hidden_dim)
    out_with_mkt, _ = mem.read(x1, st0, asset_ids=ids1)
    # Un IPO (posición 0, id=90) con M_mkt=0 vs M_mkt activo difiere por lectura asociativa Axis 1!
    st0_no_mkt = dict(st0)
    st0_no_mkt["M_mkt"] = torch.zeros_like(st0["M_mkt"])
    st0_no_mkt["M_mkt_slow"] = torch.zeros_like(st0["M_mkt_slow"])
    out_no_mkt, _ = mem.read(x1, st0_no_mkt, asset_ids=ids1)
    assert not torch.allclose(out_with_mkt[0], out_no_mkt[0], atol=1e-6)

    # Update en sesión 1: el activo pausado 50 preserva su banco lento bit-a-bit
    st1, _ = mem.update(
        x1,
        st0,
        ach_signal=torch.ones(8, 1) * 0.1,
        ne_signal=torch.tensor([0.2]),
        target_hint=torch.randn(8, hidden_dim) * 0.5,
        asset_ids=ids1,
    )
    idx50_after = int((st1["asset_ids"] == 50).nonzero(as_tuple=False).item())
    torch.testing.assert_close(st1["slow"][idx50_after], slow50_before, atol=1e-6, rtol=1e-6)

    # 3) `compact_long_running_state` atenúa `M_mkt_mom`, `fast_mom`, `fast` y preserva lentos
    comp = compact_long_running_state(st1, fast_decay=0.70, momentum_decay=0.50)
    torch.testing.assert_close(comp["fast"], st1["fast"] * 0.70, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(comp["fast_mom"], st1["fast_mom"] * 0.50, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(comp["M_mkt_mom"], st1["M_mkt_mom"] * 0.50, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(comp["medium"], st1["medium"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(comp["slow"], st1["slow"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(comp["M_mkt_slow"], st1["M_mkt_slow"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(comp["asset_ids"], st1["asset_ids"], atol=0.0, rtol=0.0)

    # 4) Interop con `_sanitize_causal_state_masked` de `long_running_harness.py` (rollback NaN)
    prev_outer = dict(titans_state=st0, last_asset_ids=ids0)
    corrupt_ts = dict(st1)
    corrupt_ts["M_mkt"] = torch.full_like(st1["M_mkt"], float("nan"))
    next_outer = dict(titans_state=corrupt_ts, last_asset_ids=ids1)
    san = _sanitize_causal_state_masked(prev_outer, next_outer, torch.tensor(False))
    torch.testing.assert_close(san["titans_state"]["M_mkt"], st0["M_mkt"], atol=1e-6, rtol=1e-6)
    assert torch.isfinite(san["titans_state"]["fast"]).all()

    # 5) Retrocompatibilidad intacta en `NeuromodulatedTitansMemory`
    legacy = NeuromodulatedTitansMemory(hidden_dim=hidden_dim, use_triton=False)
    l_st, _ = legacy.update(torch.randn(6, hidden_dim), None, asset_ids=ids0)
    l_out, _ = legacy.read(torch.randn(6, hidden_dim), l_st, asset_ids=ids0)
    assert l_out.shape == (6, hidden_dim) and torch.isfinite(l_out).all()


def test_dual_axis_bptt_live_gradients_all_params():
    """BPTT truncado (`detach_state=False`) da gradientes finitos no-nulos a TODOS los params."""
    torch.manual_seed(505)
    hidden_dim = 24
    mem = DualAxisTitansMemory(
        hidden_dim=hidden_dim,
        n_heads=4,
        use_triton=False,
        use_fused_blackwell_kernels=False,
    )
    state = None
    total_loss = torch.zeros(())
    for t in range(3):
        n_a = 12 + t * 2
        ids = torch.arange(t, t + n_a, dtype=torch.long)
        x = torch.randn(n_a, hidden_dim)
        out, _ = mem.read(x, state, asset_ids=ids)
        total_loss = total_loss + out.pow(2).mean()
        state, _ = mem.update(
            x,
            state,
            ach_signal=torch.rand(n_a, 1) * 0.5,
            ne_signal=torch.tensor([0.3 * (t - 1)]),
            target_hint=torch.randn(n_a, hidden_dim) * 0.5,
            asset_ids=ids,
            detach_state=False,
        )
    # Lectura final tras los 3 updates
    x_f = torch.randn(14, hidden_dim)
    out_f, _ = mem.read(x_f, state, asset_ids=torch.arange(2, 16, dtype=torch.long))
    (total_loss + out_f.pow(2).mean()).backward()

    required_params = (
        ("q_proj.weight", mem.q_proj.weight),
        ("k_proj.weight", mem.k_proj.weight),
        ("v_proj.weight", mem.v_proj.weight),
        ("out_proj.weight", mem.out_proj.weight),
        ("ach_gain", mem.ach_gain),
        ("ne_gain", mem.ne_gain),
    )
    for name, p in required_params:
        assert p.grad is not None, f"{name} recibió grad=None bajo BPTT!"
        assert torch.isfinite(p.grad).all(), f"{name} tiene grad no finito"
        assert float(p.grad.abs().sum()) > 0.0, f"{name} tiene grad exactamente cero"


def test_tabm_k8_head_only_diversity_per_member_loss_and_single_fista_crps(monkeypatch):
    """TabM K=8 en `alpha_head`: diversidad real, pérdida IC+Huber Kx, pero P+/CRPS solo 1x."""
    torch.manual_seed(606)
    n_assets, hidden_dim, n_exp = 32, 24, 4
    horizons = (1, 2, 5, 10, 21, 63)

    # 1) Retrocompatibilidad exacta tabm_k=1
    torch.manual_seed(607)
    head_k1 = MultiHorizonGranularHead(
        hidden_dim=hidden_dim, horizons=horizons, exposures_dim=n_exp
    )
    assert head_k1.tabm_k == 1
    assert head_k1.tabm_r is None and head_k1.tabm_s is None
    h_in = torch.randn(n_assets, hidden_dim)
    exp = torch.randn(n_assets, n_exp)
    out_k1 = head_k1(h_in, exposures=exp)
    assert out_k1["alpha_members"].shape == (n_assets, 1, len(horizons))
    torch.testing.assert_close(out_k1["alpha_members"].squeeze(1), out_k1["alpha"])

    # 2) TabM K=8 solo en alpha_head (+8*D + 8*H parámetros exactos)
    head_k8 = MultiHorizonGranularHead(
        hidden_dim=hidden_dim, horizons=horizons, exposures_dim=n_exp, tabm_k=8
    )
    n_p1 = sum(p.numel() for p in head_k1.parameters())
    n_p8 = sum(p.numel() for p in head_k8.parameters())
    assert n_p8 - n_p1 == 8 * hidden_dim + 8 * len(horizons)
    assert head_k8.tabm_r is not None and head_k8.tabm_s is not None
    assert 0.8 < float(head_k8.tabm_r.detach().mean()) < 1.2
    assert 0.8 < float(head_k8.tabm_s.detach().mean()) < 1.2

    out_k8 = head_k8(h_in, exposures=exp)
    am = out_k8["alpha_members"]
    assert am.shape == (n_assets, 8, len(horizons))
    # Cada miembro está centrado transversalmente y su media coincide con `alpha`
    torch.testing.assert_close(
        am.mean(dim=0), torch.zeros(8, len(horizons)), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(am.mean(dim=1), out_k8["alpha"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        out_k8["pred_return"], out_k8["factor_return"] + out_k8["alpha"], atol=1e-6, rtol=1e-6
    )
    # Diversidad no colapsada entre miembros
    assert float(am.std(dim=1).mean()) > 1e-5

    # 3) Supervisión por miembro: P+ FISTA y CRPS se ejecutan SOLO 1x (no 8x!),
    #    y los miembros k reciben gradientes distintos (sin colapso BatchEnsemble!)
    loss_fn = MMARRSMultiTaskLoss(
        horizons=horizons,
        use_fused_blackwell_kernels=True,
        use_gaussian_rank_targets=True,
    )
    fista_calls = 0
    crps_calls = 0
    orig_p_fwd = loss_fn.portfolio_layer.forward
    orig_crps = granular_heads.fused_pinball_crps_loss

    def _spy_p(*args, **kwargs):
        nonlocal fista_calls
        fista_calls += 1
        return orig_p_fwd(*args, **kwargs)

    def _spy_crps(*args, **kwargs):
        nonlocal crps_calls
        crps_calls += 1
        return orig_crps(*args, **kwargs)

    monkeypatch.setattr(loss_fn.portfolio_layer, "forward", _spy_p)
    monkeypatch.setattr(granular_heads, "fused_pinball_crps_loss", _spy_crps)

    y_ret = torch.randn(n_assets, len(horizons)) * 0.02
    ld = loss_fn(step_outputs=[out_k8], targets=dict(returns=y_ret), exposures=exp)
    assert fista_calls == 1, f"P+ FISTA se llamó {fista_calls} veces (debe ser 1!)"
    assert crps_calls == 1, f"CRPS se llamó {crps_calls} veces (debe ser 1!)"
    assert torch.isfinite(ld["total"])

    ld["total"].backward()
    assert head_k8.tabm_r.grad is not None and torch.isfinite(head_k8.tabm_r.grad).all()
    assert head_k8.tabm_s.grad is not None and torch.isfinite(head_k8.tabm_s.grad).all()
    # Los miembros k deben recibir gradientes distintos entre sí
    grad_spread_r = float(head_k8.tabm_r.grad.std(dim=0).mean())
    grad_spread_s = float(head_k8.tabm_s.grad.std(dim=0).mean())
    assert grad_spread_r > 1e-7 and grad_spread_s > 1e-7


def test_gaussian_rank_ic_cn_ties_nans_liqw3_horizon_mask_and_fm7_guard():
    """Gaussian IC maneja empates CN + NaNs + liqw3 + horizon_mask sin tocar Huber/CRPS/Sharpe."""
    torch.manual_seed(707)
    n_assets, n_h = 32, 6
    pred = torch.randn(n_assets, n_h, requires_grad=True)
    # Targets estilo CN: bloque empatado en +0.10 (limit-up), bloque en 0.0 (halts), y 2 NaNs
    y_cn = torch.randn(n_assets, n_h) * 0.02
    y_cn[:6, 0] = 0.10
    y_cn[6:12, 0] = 0.0
    y_cn[12:14, 0] = float("nan")
    exposures = torch.randn(n_assets, 4)
    liqw = torch.linspace(0.05, 1.0, n_assets).pow(3)  # liqw3
    h_mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    # 1) Finitud con NaNs + empates + liqw3 + horizon_mask
    loss_g = MMARRSMultiTaskLoss._neutralized_ic_loss(
        pred,
        y_cn,
        exposures=exposures,
        liquidity_weights=liqw,
        horizon_mask=h_mask,
        use_gaussian_rank_targets=True,
    )
    assert torch.isfinite(loss_g)
    loss_g.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    # Horizontes enmascarados reciben gradiente exactamente 0
    assert float(pred.grad[:, 2:].abs().sum()) == pytest.approx(0.0, abs=1e-7)
    # Filas NaN en horizonte 0 reciben gradiente 0 (sin contaminación)
    assert float(pred.grad[12:14, 0].abs().sum()) == pytest.approx(0.0, abs=1e-7)

    # 2) Invarianza a permutación dentro de empates CN (cero sesgo por orden de ticker)
    perm = torch.arange(n_assets)
    perm[:6] = torch.tensor([5, 3, 1, 4, 0, 2])  # permuta bloque limit-up empatado
    p_det = pred.detach()
    l_orig = MMARRSMultiTaskLoss._neutralized_ic_loss(
        p_det, y_cn, use_gaussian_rank_targets=True
    )
    l_perm = MMARRSMultiTaskLoss._neutralized_ic_loss(
        p_det[perm], y_cn[perm], use_gaussian_rank_targets=True
    )
    torch.testing.assert_close(l_orig, l_perm, atol=1e-5, rtol=1e-5)

    # 3) Invarianza a cola gorda extrema (multiplicar el máximo 50x no altera Gaussian IC)
    y_clean = torch.linspace(-0.05, 0.05, n_assets).unsqueeze(-1).expand(n_assets, n_h).clone()
    y_fat = y_clean.clone()
    y_fat[-1, :] = 2.50  # 50x outlier en el mayor retorno (mismo rango!)
    l_g_clean = MMARRSMultiTaskLoss._neutralized_ic_loss(
        p_det, y_clean, use_gaussian_rank_targets=True
    )
    l_g_fat = MMARRSMultiTaskLoss._neutralized_ic_loss(
        p_det, y_fat, use_gaussian_rank_targets=True
    )
    torch.testing.assert_close(l_g_clean, l_g_fat, atol=1e-5, rtol=1e-5)

    # 4) liqw3 cambia efectivamente el gradiente favoreciendo activos líquidos
    p_u = p_det.clone().requires_grad_(True)
    MMARRSMultiTaskLoss._neutralized_ic_loss(
        p_u, y_clean, use_gaussian_rank_targets=True
    ).backward()
    p_l = p_det.clone().requires_grad_(True)
    MMARRSMultiTaskLoss._neutralized_ic_loss(
        p_l, y_clean, liquidity_weights=liqw, use_gaussian_rank_targets=True
    ).backward()
    assert not torch.allclose(p_u.grad, p_l.grad, atol=1e-4)

    # 5) Guardia FM-7: Huber / CRPS / Sharpe siguen usando `y_h` crudo, NO rangos gaussianos
    head = MultiHorizonGranularHead(hidden_dim=16, exposures_dim=4)
    out = head(torch.randn(n_assets, 16), exposures=exposures)
    loss_v2 = MMARRSMultiTaskLoss(
        use_gaussian_rank_targets=True, use_fused_blackwell_kernels=False
    )
    ld_c = loss_v2(step_outputs=[out], targets=dict(returns=y_clean), exposures=exposures)
    ld_f = loss_v2(step_outputs=[out], targets=dict(returns=y_fat), exposures=exposures)
    assert not torch.allclose(ld_c["crps"], ld_f["crps"], atol=1e-4)
    assert not torch.allclose(ld_c["sharpe_turnover"], ld_f["sharpe_turnover"], atol=1e-4)


def test_downside_stab05_zero_penalty_on_upside_spikes_and_noic_twin():
    """`stab05` bajista nunca penaliza picos alcistas de IC, sí penaliza caídas, y respeta noic."""
    torch.manual_seed(808)
    n_assets, hidden_dim = 32, 16
    head = MultiHorizonGranularHead(hidden_dim=hidden_dim, exposures_dim=0, tabm_k=4)
    loss_fn = MMARRSMultiTaskLoss(
        w_stab_downside=0.05,
        stab_ema_decay=0.5,
        use_gaussian_rank_targets=True,
        use_fused_blackwell_kernels=False,
    )
    loss_fn.train()

    y = torch.linspace(-0.04, 0.04, n_assets).unsqueeze(-1).expand(n_assets, 6).clone()

    def _make_pred_with_corr(sign_strength: float) -> dict[str, torch.Tensor]:
        h_s = torch.randn(n_assets, hidden_dim)
        p = head(h_s)
        noise = torch.randn(n_assets, 6) * 0.01
        alpha_ctrl = (sign_strength * y + noise).detach().requires_grad_(True)
        p_mod = dict(p)
        p_mod["alpha"] = alpha_ctrl
        p_mod["pred_return"] = alpha_ctrl
        p_mod["alpha_members"] = alpha_ctrl.unsqueeze(1).expand(n_assets, 4, 6)
        return p_mod

    # Sesión 0 (inicializa EMA causal en torno a rho ~ +0.5): penalidad 0 al arrancar
    p0 = _make_pred_with_corr(0.5)
    ld0 = loss_fn(step_outputs=[p0], targets=dict(returns=y))
    assert float(ld0["stab_downside"]) == pytest.approx(0.0, abs=1e-7)
    ema_after_0 = float(loss_fn._ema_ic)
    assert ema_after_0 > 0.2

    # Sesión 1: PICO ALCISTA de IC (rho ~ +0.99 >> ema_after_0) -> penalidad EXACTAMENTE 0.0!
    p_up = _make_pred_with_corr(10.0)
    ld_up = loss_fn(step_outputs=[p_up], targets=dict(returns=y))
    assert float(ld_up["rho_ic"]) > ema_after_0 + 0.1
    assert float(ld_up["stab_downside"]) == pytest.approx(0.0, abs=1e-7)

    # Sesión 2: CAÍDA BAJISTA de IC (rho < ema) -> penalidad exacta = 0.5 * (ema - rho)^2
    ema_before_2 = float(loss_fn._ema_ic)
    p_dn = _make_pred_with_corr(-10.0)
    ld_dn = loss_fn(step_outputs=[p_dn], targets=dict(returns=y))
    rho_dn = float(ld_dn["rho_ic"])
    expected_pen = 0.5 * (ema_before_2 - rho_dn) ** 2
    assert float(ld_dn["stab_downside"]) == pytest.approx(expected_pen, rel=1e-4, abs=1e-6)
    assert float(ld_dn["stab_downside"]) > 0.1
    ld_dn["stab_downside"].backward()
    assert p_dn["alpha"].grad is not None and torch.isfinite(p_dn["alpha"].grad).all()
    assert float(p_dn["alpha"].grad.abs().sum()) > 0.0

    # Gemelo FM-7 `noic_pinball_twin=True`: fuerza w_stab_downside == 0.0
    loss_twin = MMARRSMultiTaskLoss(w_stab_downside=0.05, noic_pinball_twin=True)
    assert loss_twin.w_stab_downside == 0.0
    assert loss_twin.w_ic == 0.0
    assert loss_twin.w_sharpe == 0.0


def test_f1_v2_bf16_autocast_and_zero_host_syncs_ast_audit():
    """Seguridad BF16 autocast en DualAxis + TabM K=8 + Gaussian IC + stab05 y 0 host-syncs AST."""
    torch.manual_seed(909)
    n_assets, hidden_dim = 24, 32
    mem = DualAxisTitansMemory(
        hidden_dim=hidden_dim,
        n_heads=4,
        use_triton=False,
        use_fused_blackwell_kernels=False,
    )
    head = MultiHorizonGranularHead(hidden_dim=hidden_dim, exposures_dim=4, tabm_k=8)
    loss_fn = MMARRSMultiTaskLoss(
        use_gaussian_rank_targets=True,
        w_stab_downside=0.05,
        use_fused_blackwell_kernels=False,
    )

    x = torch.randn(n_assets, hidden_dim)
    exp = torch.randn(n_assets, 4)
    y = torch.randn(n_assets, 6) * 0.02
    liqw = torch.linspace(0.1, 1.0, n_assets).pow(3)
    ids = torch.arange(n_assets, dtype=torch.long)

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        st1, _ = mem.update(
            x,
            None,
            ach_signal=torch.ones(n_assets, 1) * 0.2,
            ne_signal=torch.tensor([0.2]),
            target_hint=torch.randn(n_assets, hidden_dim) * 0.3,
            asset_ids=ids,
            detach_state=False,
        )
        h_mem, _ = mem.read(x, st1, asset_ids=ids)
        pred = head(h_mem, exposures=exp)
        ld = loss_fn(
            step_outputs=[pred],
            targets=dict(returns=y),
            exposures=exp,
            liquidity_weights=liqw,
        )

    assert torch.isfinite(h_mem).all()
    assert torch.isfinite(pred["alpha_members"]).all()
    assert torch.isfinite(ld["total"])
    ld["total"].backward()
    for p in list(mem.parameters()) + list(head.parameters()):
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

    # Auditoría estática AST: CERO host-syncs en `neuromod_memory.py` y `granular_heads.py`
    audit = audit_zero_host_syncs(
        [Path(neuromod_memory.__file__), Path(granular_heads.__file__)]
    )
    assert audit["files_scanned"] == 2
    assert audit["forward_findings"] == 0, f"Forward host-syncs: {audit['findings']}"
    assert audit["total_findings"] == 0, f"Host-syncs detectados: {audit['findings']}"
    assert audit["clean"] is True


def test_f1_v2_4step_bptt_chunk_accumulation_auto_ne_and_edge_cases():
    """BPTT 4 pasos acumulados con stab05 + TabM K=8 + DualAxis (0 errores in-place) + bordes."""
    torch.manual_seed(1010)
    hidden_dim = 24
    mem = DualAxisTitansMemory(
        hidden_dim=hidden_dim,
        n_heads=4,
        use_triton=False,
        use_fused_blackwell_kernels=False,
    )
    head = MultiHorizonGranularHead(hidden_dim=hidden_dim, exposures_dim=4, tabm_k=8)
    loss_fn = MMARRSMultiTaskLoss(
        use_gaussian_rank_targets=True,
        w_stab_downside=0.05,
        stab_ema_decay=0.8,
        use_fused_blackwell_kernels=False,
    )
    loss_fn.train()

    # 1) Acumular 4 sesiones BPTT ragged (ne_signal=None -> EMA causal auto-NE) antes de backward()
    state = None
    chunk_loss = torch.zeros(())
    ne_trace = []
    for t in range(4):
        n_a = 16 + (t % 3) * 4
        ids = torch.arange(t * 2, t * 2 + n_a, dtype=torch.long)
        x = torch.randn(n_a, hidden_dim)
        exp = torch.randn(n_a, 4)
        y = torch.randn(n_a, 6) * 0.02
        h_mem, _ = mem.read(x, state, ne_signal=None, asset_ids=ids)
        pred = head(h_mem, exposures=exp)
        ld = loss_fn(step_outputs=[pred], targets=dict(returns=y), exposures=exp)
        chunk_loss = chunk_loss + ld["total"]
        # En t=3 inyectar shock de error maduro para verificar subida de auto-NE sin constante 0.015
        err_mag = torch.full((n_a,), 0.50 if t == 3 else 0.02)
        state, d_upd = mem.update(
            x,
            state,
            ach_signal=None,
            ne_signal=None,
            target_hint=torch.randn(n_a, hidden_dim) * 0.2,
            asset_ids=ids,
            matured_error=err_mag,
            detach_state=False,
        )
        ne_trace.append(float(d_upd["last_ne_signal"]))

    (chunk_loss / 4.0).backward()
    # Verificar 0 fallos in-place y gradientes finitos en TODOS los parámetros clave
    for p_name, p in (
        ("q_proj.weight", mem.q_proj.weight),
        ("k_proj.weight", mem.k_proj.weight),
        ("v_proj.weight", mem.v_proj.weight),
        ("out_proj.weight", mem.out_proj.weight),
        ("ach_gain", mem.ach_gain),
        ("ne_gain", mem.ne_gain),
        ("tabm_r", head.tabm_r),
        ("tabm_s", head.tabm_s),
    ):
        assert p is not None and p.grad is not None, f"{p_name} grad=None en BPTT 4-pasos"
        assert torch.isfinite(p.grad).all(), f"{p_name} grad no finito"
        assert float(p.grad.abs().sum()) > 0.0, f"{p_name} grad cero"
    # Auto-NE causal: neutro en t=0 y sube claramente ante shock de error en t=3
    assert ne_trace[0] == pytest.approx(0.0, abs=1e-6)
    assert ne_trace[3] > 2.0

    # 2) Bordes extremos: N=1 y N=0 en DualAxis, y horizonte todo-NaN en Gaussian IC
    st_n1, _ = mem.update(torch.randn(1, hidden_dim), None, detach_state=True)
    out_n1, _ = mem.read(torch.randn(1, hidden_dim), st_n1)
    assert torch.isfinite(out_n1).all()
    st_n0, _ = mem.update(torch.zeros(0, hidden_dim), st_n1, detach_state=True)
    out_n0, _ = mem.read(torch.zeros(0, hidden_dim), st_n0)
    assert out_n0.shape == (0, hidden_dim)

    p_nan = torch.randn(8, 2, requires_grad=True)
    y_all_nan = torch.full((8, 2), float("nan"))
    l_nan = MMARRSMultiTaskLoss._neutralized_ic_loss(
        p_nan, y_all_nan, use_gaussian_rank_targets=True
    )
    assert torch.isfinite(l_nan)
    l_nan.backward()
    assert p_nan.grad is not None and torch.isfinite(p_nan.grad).all()


def test_wave3c_granular_loss_per_cell_nan_masking_zero_sync():
    """#187: Per-cell NaN/Inf masking in Huber, CRPS (fused/unfused), TB, turn, vol, epist, P+."""
    torch.manual_seed(1111)
    n_assets, d_model, n_h = 24, 16, 6

    for use_fused in (False, True):
        for tabm_k in (1, 4):
            head = MultiHorizonGranularHead(hidden_dim=d_model, exposures_dim=4, tabm_k=tabm_k)
            loss_fn = MMARRSMultiTaskLoss(
                use_fused_blackwell_kernels=use_fused,
                use_gaussian_rank_targets=True,
            )
            h = torch.randn(n_assets, d_model, requires_grad=True)
            exp = torch.randn(n_assets, 4)
            pred = head(h, exposures=exp)

            y_ret = torch.randn(n_assets, n_h) * 0.02
            y_ret[0, 0] = float("nan")
            y_ret[1, 2] = float("inf")
            y_ret[2, 3] = float("-inf")
            y_ret[3, :] = float("nan")

            y_vol = torch.full((n_assets, n_h), 0.02)
            y_vol[0, 0] = float("nan")
            y_vol[4, 1] = float("inf")

            y_turn = (torch.rand(n_assets, n_h) > 0.5).float()
            y_turn[5, 0] = float("nan")

            comp = torch.randn(n_assets) * 0.01
            comp[6] = float("nan")

            ld = loss_fn(
                step_outputs=[pred],
                targets=dict(returns=y_ret, vol=y_vol, turning=y_turn),
                exposures=exp,
                companion_pred=comp,
            )
            for k_name, v in ld.items():
                if isinstance(v, torch.Tensor):
                    assert torch.isfinite(v).all(), (
                        f"Non-finite {k_name} (fused={use_fused}, k={tabm_k})"
                    )
            ld["total"].backward()
            assert h.grad is not None and torch.isfinite(h.grad).all()
            for p_name, p in head.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"Non-finite grad {p_name}"

    # Verify exact 0.0 gradient on NaN cell in direct leaf pred_return (Huber + CRPS + Epist)
    head1 = MultiHorizonGranularHead(hidden_dim=d_model, exposures_dim=0)
    base_p = head1(torch.randn(n_assets, d_model))
    pr_leaf = base_p["pred_return"].detach().clone().requires_grad_(True)
    q_leaf = base_p["quantiles"].detach().clone().requires_grad_(True)
    ep_leaf = base_p["epistemic_error"].detach().clone().requires_grad_(True)
    p_leaf = dict(base_p)
    p_leaf["pred_return"] = pr_leaf
    p_leaf["alpha"] = pr_leaf
    p_leaf["alpha_members"] = pr_leaf.unsqueeze(1)
    p_leaf["quantiles"] = q_leaf
    p_leaf["epistemic_error"] = ep_leaf

    y_partial = torch.randn(n_assets, n_h) * 0.02
    y_partial[0, 1] = float("nan")
    loss_leaf = MMARRSMultiTaskLoss(w_sharpe=0.0, use_fused_blackwell_kernels=False)
    ld_leaf = loss_leaf(step_outputs=[p_leaf], targets=dict(returns=y_partial))
    ld_leaf["total"].backward()
    assert float(pr_leaf.grad[0, 1].abs()) == pytest.approx(0.0, abs=1e-7)
    assert float(q_leaf.grad[0, 1].abs().sum()) == pytest.approx(0.0, abs=1e-7)
    assert float(ep_leaf.grad[0, 1].abs()) == pytest.approx(0.0, abs=1e-7)

    audit = audit_zero_host_syncs([Path(granular_heads.__file__)])
    assert audit["clean"] is True and audit["total_findings"] == 0


def test_wave3c_granular_loss_sota_terms_and_polar_p0_1_cross_detachment():
    """Opt-in TailListNet + VICReg + SupCon + PolarAlphaHead P0-1a/b/c cross-detachment & churn."""
    from mars_titan.models.mmarrs.frontier_modules_sota import PolarAlphaHead

    torch.manual_seed(1112)
    n_assets, d_model, n_h = 32, 24, 6

    # 1) Defaults remain 0.0 for backwards compatibility
    loss_default = MMARRSMultiTaskLoss()
    assert loss_default.w_listnet_tail == 0.0
    assert loss_default.w_latent_cov == 0.0
    assert loss_default.w_supcon == 0.0
    assert loss_default.w_churn == 0.0

    # 2) Opt-in SOTA terms + PolarAlphaHead P0-1a/b/c guards
    head = MultiHorizonGranularHead(hidden_dim=d_model, exposures_dim=4)
    polar = PolarAlphaHead(n_horizons=n_h, default_vol=0.02)
    loss_sota = MMARRSMultiTaskLoss(
        w_listnet_tail=0.5,
        w_latent_cov=0.1,
        w_supcon=0.1,
        w_polar_cal=1.0,
        w_churn=0.2,
        use_fused_blackwell_kernels=False,
    )

    h = torch.randn(n_assets, d_model, requires_grad=True)
    exp = torch.randn(n_assets, 4)
    y_ret = torch.randn(n_assets, n_h) * 0.02
    vol = torch.full((n_assets,), 0.02)
    liqw = torch.linspace(0.1, 1.0, n_assets)
    cur_ids = torch.arange(100, 100 + n_assets, dtype=torch.long)
    prev_ids = torch.arange(95, 95 + n_assets, dtype=torch.long)
    prev_u = torch.randn(n_assets, n_h)

    raw_pred = head(h, exposures=exp)
    p_out = polar(raw_pred["alpha"], vol=vol)
    step_pred = dict(raw_pred)
    step_pred["alpha_raw"] = raw_pred["alpha"]
    step_pred["alpha_u"] = p_out["u"]
    step_pred["alpha_cal"] = p_out["alpha_cal"]
    step_pred["alpha_cal_for_cal"] = p_out["alpha_cal_for_cal"]
    step_pred["alpha_cal_for_pplus"] = p_out["alpha_cal_for_pplus"]
    step_pred["alpha"] = p_out["alpha_cal"]
    step_pred["pred_return"] = raw_pred["factor_return"] + p_out["alpha_cal"]

    ld = loss_sota(
        step_outputs=[step_pred],
        targets=dict(returns=y_ret, vol=vol.unsqueeze(-1).expand(n_assets, n_h)),
        exposures=exp,
        liquidity_weights=liqw,
        final_hidden=h,
        prev_alpha_u=prev_u,
        cur_asset_ids=cur_ids,
        prev_asset_ids=prev_ids,
    )
    for k_req in ("listnet_tail", "latent_cov", "supcon", "polar_cal", "churn", "total"):
        assert k_req in ld, f"Missing loss key {k_req}"
        assert torch.isfinite(ld[k_req])
    assert float(ld["listnet_tail"].detach()) > 0.0
    assert float(ld["latent_cov"].detach()) > 0.0
    assert float(ld["supcon"].detach()) > 0.0
    assert float(ld["polar_cal"].detach()) > 0.0
    assert float(ld["churn"].detach()) > 0.0

    # P0-1c Check 1: polar_cal trains ONLY polar.raw_gamma, NOT head.alpha_head!
    head.zero_grad()
    polar.zero_grad()
    ld["polar_cal"].backward(retain_graph=True)
    assert polar.raw_gamma.grad is not None and float(polar.raw_gamma.grad.abs().sum()) > 0.0
    assert (
        head.alpha_head.weight.grad is None
        or float(head.alpha_head.weight.grad.abs().sum()) == 0.0
    )

    # P0-1c Check 2: sharpe_turnover receives alpha_cal_for_pplus -> 0 grad to polar.raw_gamma!
    head.zero_grad()
    polar.zero_grad()
    ld["sharpe_turnover"].backward(retain_graph=True)
    assert head.alpha_head.weight.grad is not None
    assert float(head.alpha_head.weight.grad.abs().sum()) > 0.0
    assert polar.raw_gamma.grad is None or float(polar.raw_gamma.grad.abs().sum()) == 0.0

    # P0-1a Check 3: ic + listnet_tail operate on alpha_u -> 0 grad to polar.raw_gamma!
    head.zero_grad()
    polar.zero_grad()
    (ld["ic"] + ld["listnet_tail"]).backward()
    assert head.alpha_head.weight.grad is not None
    assert float(head.alpha_head.weight.grad.abs().sum()) > 0.0
    assert polar.raw_gamma.grad is None or float(polar.raw_gamma.grad.abs().sum()) == 0.0


