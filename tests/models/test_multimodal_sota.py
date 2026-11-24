"""Pruebas TDD de `multimodal_sota.py`: fusión con conflicto, alineación SSL y cabeza Skew-t.

Verifica de forma estricta (FP32 estricto, BF16 y FP16 compatibles con RTX PRO 6000 `sm_120`):
  1. `CrossModalConflictFusion`: seguridad estricta de máscaras (mutar modalidades ausentes,
     incluso con `1e9`/`NaN`/`Inf`, cambia CERO bits en `fused_h`, `conflict_score` y
     diagnósticos), sensibilidad del detector de conflicto direccional/angular/grupal,
     decaimiento por frescura (`modality_age`), equivarianza por permutación y gradientes finitos.
  2. `SelfSupervisedMultimodalAlignmentLoss`: reconstrucción enmascarada sin fuga propia
     (leave-one-out estricto), alineación VICReg + Barlow Twins (`prices+charts` vs
     `fundamentals+macro+news`), detección real de colapso dimensional/colinealidad y seguridad
     ante modalidades parcialmente ausentes (incluyendo 1 sola modalidad presente, 0 presentes y
     `N=1`).
  3. `SkewStudentTDistributionHead` + `skew_student_t_nll`: dominios válidos (`scale > 0`,
     `df > 2.2`, `|skew| < 0.95`), paridad exacta con `scipy.stats.t` e integración numérica de
     VaR/CVaR, monotonicidad estricta del riesgo de cola (`CVaR_1% < VaR_1% < VaR_5%`,
     `CVaR_5% < VaR_5%`), cuantiles crecientes sin cruces, identidad exacta
     `aleatoric_var + epistemic_var == total_var`, acoplamiento epistémico con `conflict_score`
     y gradientes finitos.
  4. Auditoría AST estática: CERO sincronizaciones CPU↔GPU (`.item()`, `.cpu()`, `.numpy()`,
     `.tolist()`, `.synchronize()`) y uso nativo de `scaled_dot_product_attention`.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import integrate, stats

from mars_titan.models.mmarrs import multimodal_sota as ms_mod
from mars_titan.models.mmarrs.config import DEFAULT_HORIZONS, DEFAULT_QUANTILES
from mars_titan.models.mmarrs.multimodal_sota import (
    ASSET_CONTEXT_VIEW_INDICES,
    MODALITY_NAMES,
    CrossModalConflictFusion,
    SelfSupervisedMultimodalAlignmentLoss,
    SkewStudentTDistributionHead,
    skew_student_t_nll,
    skew_student_t_var_cvar,
)


def _skew_t_pdf_reference(
    y: float, loc: float, scale: float, df: float, skew: float
) -> float:
    """Densidad exacta de referencia Two-Piece Skew Student-t (Hansen 1994 / Fernández-Steel)."""
    z = (y - loc) / scale
    s = (1.0 + skew) if z >= 0.0 else (1.0 - skew)
    log_c = (
        math.lgamma(0.5 * (df + 1.0))
        - math.lgamma(0.5 * df)
        - 0.5 * math.log(math.pi * df)
    )
    return math.exp(
        log_c - math.log(scale) - 0.5 * (df + 1.0) * math.log1p(((z / s) ** 2) / df)
    )


def test_cross_modal_conflict_fusion_strict_mask_safety_zero_bit_leakage():
    """Mutar modalidades ausentes cambia CERO bits en `fused_h`, `conflict_score` y diagnósticos."""
    torch.manual_seed(11)
    n_assets, n_mod, d_model = 16, 5, 32
    assert MODALITY_NAMES == ("prices", "fundamentals", "macro", "news", "charts")

    fusion = CrossModalConflictFusion(hidden_dim=d_model, n_modalities=n_mod, n_heads=4)
    fusion.eval()

    tokens = torch.randn(n_assets, n_mod, d_model)
    mask = torch.ones(n_assets, n_mod)
    mask[:, 3] = 0.0  # noticias ausentes en toda la sesión
    mask[0::2, 1] = 0.0  # fundamentales ausentes en la mitad de los activos
    mask[0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])  # activo 0: solo 1 modalidad presente
    mask[1, :] = 0.0  # activo 1: todas las modalidades ausentes
    phys = torch.randn(n_assets, d_model)
    age = torch.rand(n_assets, n_mod) * 5.0

    with torch.no_grad():
        fused_clean, conf_clean, diag_clean = fusion(tokens, mask, phys, modality_age=age)

    assert fused_clean.shape == (n_assets, d_model)
    assert conf_clean.shape == (n_assets, 1)
    assert diag_clean["conflict_per_pair"].shape == (n_assets, 10)
    assert diag_clean["modality_votes"].shape == (n_assets, n_mod)
    assert diag_clean["modality_weights"].shape == (n_assets, n_mod)
    assert diag_clean["freshness"].shape == (n_assets, n_mod)
    assert diag_clean["group_conflict"].shape == (n_assets, 1)
    assert torch.isfinite(fused_clean).all()
    assert torch.isfinite(conf_clean).all()
    for name, tensor in diag_clean.items():
        assert isinstance(tensor, torch.Tensor), f"{name} debe ser Tensor en dispositivo"
        assert torch.isfinite(tensor).all(), f"Diagnóstico no finito: {name}"

    # Activo con 1 sola modalidad y activo con 0 modalidades: conflicto exactamente 0
    torch.testing.assert_close(conf_clean[0], torch.zeros(1), atol=0.0, rtol=0.0)
    torch.testing.assert_close(conf_clean[1], torch.zeros(1), atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        diag_clean["conflict_per_pair"][0], torch.zeros(10), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        diag_clean["conflict_per_pair"][1], torch.zeros(10), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        diag_clean["modality_votes"][:, 3], torch.zeros(n_assets), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        diag_clean["modality_weights"][:, 3], torch.zeros(n_assets), atol=0.0, rtol=0.0
    )

    # Envenenar modalidades ausentes con extremos numéricos, Inf y NaN -> CERO bits modificados
    absent = mask < 0.5
    for poison_val in (1e9, -1e9, float("inf"), float("nan")):
        tokens_poison = tokens.clone()
        tokens_poison[absent] = poison_val
        age_poison = age.clone()
        age_poison[absent] = poison_val
        with torch.no_grad():
            fused_p, conf_p, diag_p = fusion(
                tokens_poison, mask, phys, modality_age=age_poison
            )
        torch.testing.assert_close(fused_clean, fused_p, atol=0.0, rtol=0.0)
        torch.testing.assert_close(conf_clean, conf_p, atol=0.0, rtol=0.0)
        for key in diag_clean:
            torch.testing.assert_close(diag_clean[key], diag_p[key], atol=0.0, rtol=0.0)

    # Frescura: una modalidad obsoleta (age=60 días) recibe peso estrictamente menor que fresca
    with torch.no_grad():
        mask_full = torch.ones(n_assets, n_mod)
        age_fresh = torch.zeros(n_assets, n_mod)
        age_stale = torch.zeros(n_assets, n_mod)
        age_stale[:, 4] = 60.0
        _, _, d_fresh = fusion(tokens, mask_full, phys, modality_age=age_fresh)
        _, _, d_stale = fusion(tokens, mask_full, phys, modality_age=age_stale)
        assert float(d_stale["freshness"][:, 4].mean()) < 1e-2
        assert float(d_stale["modality_weights"][:, 4].mean()) < (
            float(d_fresh["modality_weights"][:, 4].mean()) * 0.25
        )

    # Gradientes finitos en train() y gradiente exactamente 0 en tokens ausentes
    fusion.train()
    tok_g = tokens.clone().requires_grad_(True)
    phys_g = phys.clone().requires_grad_(True)
    fused_g, conf_g, diag_g = fusion(tok_g, mask, phys_g, modality_age=age)
    loss = fused_g.pow(2).mean() + conf_g.pow(2).mean() + diag_g["conflict_per_pair"].pow(2).mean()
    loss.backward()
    for name, p in fusion.named_parameters():
        assert p.grad is not None, f"Parámetro sin gradiente: {name}"
        assert torch.isfinite(p.grad).all(), f"Gradiente no finito en {name}"
    assert tok_g.grad is not None and torch.isfinite(tok_g.grad).all()
    assert phys_g.grad is not None and torch.isfinite(phys_g.grad).all()
    torch.testing.assert_close(
        tok_g.grad[absent], torch.zeros_like(tok_g.grad[absent]), atol=0.0, rtol=0.0
    )


def test_cross_modal_conflict_fusion_detector_sensitivity_equivariance_and_low_precision():
    """Conflicto crece ante señales opuestas; equivariante y compatible BF16/FP16."""
    torch.manual_seed(23)
    n_assets, n_mod, d_model = 12, 5, 32
    fusion = CrossModalConflictFusion(hidden_dim=d_model, n_modalities=n_mod, n_heads=4)
    fusion.eval()

    mask = torch.ones(n_assets, n_mod)
    phys = torch.randn(n_assets, d_model)

    # Caso 1: vectores aleatorios alineados (+v en todas) vs opuestos (prices+charts=+v, resto=-v)
    base_dir = torch.randn(n_assets, 1, d_model)
    tokens_agree = base_dir.expand(n_assets, n_mod, d_model).clone()
    tokens_conflict = tokens_agree.clone()
    tokens_conflict[:, 1:4, :] = -tokens_conflict[:, 1:4, :]

    with torch.no_grad():
        _, conf_agree, diag_agree = fusion(tokens_agree, mask, phys)
        _, conf_conflict, diag_conflict = fusion(tokens_conflict, mask, phys)

    assert float(conf_agree.max()) < 1e-4
    assert float(conf_conflict.min()) > 0.25
    assert (conf_conflict > conf_agree + 0.2).all()
    assert (diag_conflict["group_conflict"] > diag_agree["group_conflict"] + 0.2).all()

    # Caso 2: vectores constantes positivos vs negativos (+1 vs -1) no colapsan por LayerNorm
    const_agree = torch.ones(n_assets, n_mod, d_model)
    const_conflict = const_agree.clone()
    const_conflict[:, 0, :] = 1.0
    const_conflict[:, 4, :] = 1.0
    const_conflict[:, 1:4, :] = -1.0
    with torch.no_grad():
        _, conf_c_agree, _ = fusion(const_agree, mask, phys)
        _, conf_c_conflict, _ = fusion(const_conflict, mask, phys)
    assert (conf_c_conflict > conf_c_agree + 0.2).all()

    # Caso 3: equivarianza estricta por permutación de activos
    perm = torch.randperm(n_assets)
    age = torch.rand(n_assets, n_mod) * 4.0
    with torch.no_grad():
        f1, c1, d1 = fusion(tokens_conflict, mask, phys, modality_age=age)
        fp, cp, dp = fusion(
            tokens_conflict[perm], mask[perm], phys[perm], modality_age=age[perm]
        )
    torch.testing.assert_close(f1[perm], fp, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(c1[perm], cp, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        d1["conflict_per_pair"][perm], dp["conflict_per_pair"], atol=1e-5, rtol=1e-5
    )

    # Caso 4: compatibilidad estricta BF16 y FP16 (forward + backward finitos)
    for dt in (torch.bfloat16, torch.float16):
        mod_lp = CrossModalConflictFusion(hidden_dim=d_model, n_modalities=n_mod, n_heads=4).to(
            dtype=dt
        )
        t_lp = torch.randn(n_assets, n_mod, d_model, dtype=dt, requires_grad=True)
        m_lp = mask.clone()
        m_lp[0, :] = 0.0
        m_lp[:, 2] = 0.0
        p_lp = torch.randn(n_assets, d_model, dtype=dt, requires_grad=True)
        a_lp = torch.rand(n_assets, n_mod, dtype=dt) * 10.0
        out_lp, conf_lp, diag_lp = mod_lp(t_lp, m_lp, p_lp, modality_age=a_lp)
        assert out_lp.dtype == dt
        assert torch.isfinite(out_lp).all()
        assert torch.isfinite(conf_lp).all()
        (out_lp.float().pow(2).mean() + conf_lp.float().pow(2).mean()).backward()
        assert t_lp.grad is not None and torch.isfinite(t_lp.grad).all()
        assert p_lp.grad is not None and torch.isfinite(p_lp.grad).all()
        for name, param in mod_lp.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), (
                f"Grad no finito en {dt} para {name}"
            )
        assert all(torch.isfinite(v).all() for v in diag_lp.values())


def test_self_supervised_alignment_loss_reconstruction_vicreg_barlow_and_missing_modalities():
    """SSL produce pérdidas finitas >= 0, sin fuga propia, detecta colapso y tolera máscaras."""
    torch.manual_seed(31)
    n_assets, n_mod, d_model = 24, 5, 32
    ssl = SelfSupervisedMultimodalAlignmentLoss(hidden_dim=d_model, n_modalities=n_mod, n_heads=4)

    tokens = torch.randn(n_assets, n_mod, d_model, requires_grad=True)
    mask = torch.ones(n_assets, n_mod)
    mask[0::3, 2] = 0.0  # macro parcialmente ausente
    mask[1::4, 3] = 0.0  # news parcialmente ausente
    phys = torch.randn(n_assets, d_model, requires_grad=True)

    losses = ssl(tokens, mask, physics_emb=phys)
    expected_keys = (
        "total",
        "reconstruction",
        "recon_cosine",
        "recon_mse",
        "alignment",
        "vicreg_invariance",
        "vicreg_variance",
        "vicreg_covariance",
        "barlow_twins",
        "valid_recon_ratio",
        "valid_pair_ratio",
    )
    for key in expected_keys:
        assert key in losses and isinstance(losses[key], torch.Tensor)
        assert torch.isfinite(losses[key]).all(), f"Pérdida no finita: {key}"
        val = float(losses[key].detach())
        assert val >= 0.0, f"Pérdida negativa en {key}: {val}"

    assert float(losses["reconstruction"].detach()) > 0.0
    assert float(losses["alignment"].detach()) > 0.0

    losses["total"].backward()
    for name, p in ssl.named_parameters():
        assert p.grad is not None, f"Parámetro SSL sin gradiente: {name}"
        assert torch.isfinite(p.grad).all(), f"Gradiente SSL no finito en {name}"
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert phys.grad is not None and torch.isfinite(phys.grad).all()

    # 1) Cero fuga propia en reconstrucción enmascarada (leave-one-out estricto)
    ssl.eval()
    with torch.no_grad():
        tok_base = torch.randn(n_assets, n_mod, d_model)
        mask_full = torch.ones(n_assets, n_mod)
        pred_base, valid_base = ssl.reconstruct_masked_modalities(
            tok_base, mask_full, phys.detach()
        )
        assert valid_base.all()
        for m in range(n_mod):
            tok_mut = tok_base.clone()
            tok_mut[:, m, :] = tok_mut[:, m, :] + 100.0
            pred_mut, _ = ssl.reconstruct_masked_modalities(tok_mut, mask_full, phys.detach())
            # Mutar modalidad `m` NO altera ni un bit de su propia reconstrucción `pred[:, m]`
            torch.testing.assert_close(
                pred_base[:, m, :], pred_mut[:, m, :], atol=0.0, rtol=0.0
            )

    # 2) Cero fuga de modalidades ausentes en las pérdidas SSL
    with torch.no_grad():
        tok_abs = tok_base.clone()
        tok_abs[mask < 0.5] = float("nan")
        l_clean = ssl(tok_base, mask, physics_emb=phys.detach())
        l_pois = ssl(tok_abs, mask, physics_emb=phys.detach())
        for key in expected_keys:
            torch.testing.assert_close(l_clean[key], l_pois[key], atol=0.0, rtol=0.0)

    # 3) Detección real de colapso dimensional (VICReg variance) y colinealidad (cov + Barlow)
    with torch.no_grad():
        diverse_tok = torch.randn(64, n_mod, d_model) * 2.0
        collapsed_tok = diverse_tok[:1].expand(64, n_mod, d_model).clone()
        mask_64 = torch.ones(64, n_mod)
        l_div = ssl(diverse_tok, mask_64)
        l_col = ssl(collapsed_tok, mask_64)
        assert float(l_col["vicreg_variance"]) > float(l_div["vicreg_variance"]) + 0.2

        # Embeddings con rango 1 (todas las dimensiones colineales) disparan covarianza/Barlow
        rank1_scalar = torch.randn(64, 1, 1) * 2.0
        rank1_dir = torch.randn(1, 1, d_model)
        rank1_tok = (rank1_scalar * rank1_dir).expand(64, n_mod, d_model).clone()
        l_r1 = ssl(rank1_tok, mask_64)
        assert float(l_r1["vicreg_covariance"]) > float(l_div["vicreg_covariance"])
        assert float(l_r1["barlow_twins"]) > float(l_div["barlow_twins"])

    # Validación de argumentos inválidos
    with pytest.raises(ValueError):
        SelfSupervisedMultimodalAlignmentLoss(hidden_dim=4)
    with pytest.raises(ValueError):
        CrossModalConflictFusion(hidden_dim=16, n_heads=3)
    with pytest.raises(ValueError):
        SkewStudentTDistributionHead(hidden_dim=16, min_df=2.0)
    with pytest.raises(ValueError):
        skew_student_t_var_cvar(
            torch.zeros(1), torch.ones(1), torch.full((1,), 5.0), torch.zeros(1), 1.5
        )

    # 4) Casos límite de máscaras: solo 1 modalidad presente, 0 presentes, y N=1
    ssl.train()
    edge_cases = [
        # Solo 1 modalidad presente por activo (variando qué modalidad)
        (
            torch.randn(10, n_mod, d_model, requires_grad=True),
            torch.eye(n_mod).repeat(2, 1),
        ),
        # Todas las modalidades ausentes
        (
            torch.randn(6, n_mod, d_model, requires_grad=True),
            torch.zeros(6, n_mod),
        ),
        # Solo vista A presente (prices + charts), vista B ausente
        (
            torch.randn(8, n_mod, d_model, requires_grad=True),
            torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0]]).repeat(8, 1),
        ),
        # Un único activo N=1 con todas las modalidades presentes
        (
            torch.randn(1, n_mod, d_model, requires_grad=True),
            torch.ones(1, n_mod),
        ),
    ]
    for tok_e, mask_e in edge_cases:
        ssl.zero_grad()
        out_e = ssl(tok_e, mask_e)
        for key in expected_keys:
            assert torch.isfinite(out_e[key]).all(), f"Edge case no finito en {key}"
            assert float(out_e[key].detach()) >= 0.0, f"Edge case negativo en {key}"
        out_e["total"].backward()
        assert tok_e.grad is not None and torch.isfinite(tok_e.grad).all()
        for name, p in ssl.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Grad edge case no finito en {name}"

    # 5) Compatibilidad BF16 y FP16 (forward + backward finitos)
    for dt in (torch.bfloat16, torch.float16):
        ssl_lp = SelfSupervisedMultimodalAlignmentLoss(
            hidden_dim=d_model, n_modalities=n_mod, n_heads=4
        ).to(dtype=dt)
        tok_lp = torch.randn(n_assets, n_mod, d_model, dtype=dt, requires_grad=True)
        out_lp = ssl_lp(tok_lp, mask)
        assert torch.isfinite(out_lp["total"]).all()
        assert float(out_lp["total"].detach()) >= 0.0
        out_lp["total"].backward()
        assert tok_lp.grad is not None and torch.isfinite(tok_lp.grad).all()


def test_skew_student_t_head_parameters_exact_nll_tail_risk_and_variance_decomposition():
    """SkewStudentTDistributionHead cumple dominios, NLL exacto, VaR/CVaR y aleatoric+epistemic."""
    torch.manual_seed(47)
    n_assets, d_model = 20, 32
    horizons = DEFAULT_HORIZONS
    n_h = len(horizons)

    head = SkewStudentTDistributionHead(
        hidden_dim=d_model,
        horizons=horizons,
        quantile_levels=DEFAULT_QUANTILES,
        n_ensemble=4,
    )

    # Incluir entradas extremas para estresar cotas de scale, df y skew
    h = torch.randn(n_assets, d_model)
    h[0] = 50.0
    h[1] = -50.0
    h.requires_grad_(True)
    conflict = torch.linspace(0.0, 1.0, n_assets).unsqueeze(-1)

    dist = head(h, conflict_score=conflict, compute_tail_risk=True)
    for key in (
        "loc",
        "mean",
        "scale",
        "df",
        "skew",
        "aleatoric_var",
        "epistemic_var",
        "total_var",
        "var_01",
        "var_05",
        "cvar_01",
        "cvar_05",
    ):
        assert key in dist and dist[key].shape == (n_assets, n_h)
        assert torch.isfinite(dist[key]).all(), f"Tensor no finito en cabeza Skew-t: {key}"

    # 1) Dominios paramétricos estrictos
    assert (dist["scale"] > 0.0).all()
    assert (dist["df"] > 2.2).all()
    assert (dist["skew"].abs() < 0.95).all()
    assert (dist["aleatoric_var"] > 0.0).all()
    assert (dist["epistemic_var"] > 0.0).all()

    # 2) Identidad exacta de descomposición aleatórica + epistémica == varianza total Skew-t
    torch.testing.assert_close(
        dist["aleatoric_var"] + dist["epistemic_var"],
        dist["total_var"],
        atol=1e-6,
        rtol=1e-5,
    )

    # 3) Monotonicidad estricta del riesgo de cola y 9 cuantiles sin cruces
    assert (dist["cvar_01"] < dist["var_01"]).all()
    assert (dist["var_01"] < dist["var_05"]).all()
    assert (dist["cvar_05"] < dist["var_05"]).all()
    assert (dist["cvar_01"] < dist["cvar_05"]).all()
    q = dist["quantiles"]
    assert q.shape == (n_assets, n_h, len(DEFAULT_QUANTILES))
    assert (q[..., 1:] > q[..., :-1]).all()
    torch.testing.assert_close(q[..., 0], dist["var_01"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(q[..., 1], dist["var_05"], atol=1e-6, rtol=1e-5)

    # 4) Mayor conflicto multimodal aumenta varianza epistémica sin alterar varianza aleatórica
    with torch.no_grad():
        d_low_c = head(h.detach(), conflict_score=torch.zeros(n_assets, 1))
        d_high_c = head(h.detach(), conflict_score=torch.ones(n_assets, 1))
        torch.testing.assert_close(
            d_low_c["aleatoric_var"], d_high_c["aleatoric_var"], atol=1e-7, rtol=1e-6
        )
        assert (d_high_c["epistemic_var"] > d_low_c["epistemic_var"]).all()

    # 5) Paridad exacta de `skew_student_t_nll` con `scipy.stats.t` cuando `skew == 0`
    y_ref = torch.randn(n_assets, n_h) * 0.03
    loc_t = dist["loc"].detach()
    scale_t = dist["scale"].detach()
    df_t = dist["df"].detach()
    zero_skew = torch.zeros_like(loc_t)
    nll_sym = skew_student_t_nll(y_ref, loc_t, scale_t, df_t, zero_skew, reduction="none")
    scipy_nll = -stats.t.logpdf(
        y_ref.numpy(), df=df_t.numpy(), loc=loc_t.numpy(), scale=scale_t.numpy()
    )
    torch.testing.assert_close(
        nll_sym, torch.tensor(scipy_nll, dtype=torch.float32), atol=1e-4, rtol=1e-4
    )

    # 6) Verificación por integración numérica de PDF, VaR y CVaR para sesgo negativo y positivo
    for df_v, skew_v in ((2.5, -0.75), (4.0, 0.0), (8.0, 0.80)):
        loc_v, scale_v = 0.002, 0.025
        area, _ = integrate.quad(
            _skew_t_pdf_reference, -np.inf, np.inf, args=(loc_v, scale_v, df_v, skew_v), limit=200
        )
        assert abs(area - 1.0) < 1e-5
        t_loc = torch.tensor(loc_v)
        t_sc = torch.tensor(scale_v)
        t_df = torch.tensor(df_v)
        t_sk = torch.tensor(skew_v)
        var01, cvar01 = skew_student_t_var_cvar(t_loc, t_sc, t_df, t_sk, 0.01)
        var05, cvar05 = skew_student_t_var_cvar(t_loc, t_sc, t_df, t_sk, 0.05)
        assert float(cvar01) < float(var01) < float(var05)
        assert float(cvar05) < float(var05)
        cdf05, _ = integrate.quad(
            _skew_t_pdf_reference,
            -np.inf,
            float(var05),
            args=(loc_v, scale_v, df_v, skew_v),
            limit=200,
        )
        cvar05_num, _ = integrate.quad(
            lambda y, lo=loc_v, sc=scale_v, df=df_v, sk=skew_v: (
                y * _skew_t_pdf_reference(y, lo, sc, df, sk)
            ),
            -np.inf,
            float(var05),
            limit=200,
        )
        cvar05_num /= 0.05
        assert abs(cdf05 - 0.05) < 5e-4
        assert abs(float(cvar05) - cvar05_num) / abs(cvar05_num) < 2e-3

    # 7) Gradientes finitos de NLL + pérdida evidencial en todos los parámetros
    loss_dict = head.loss(dist, y_ref)
    assert torch.isfinite(loss_dict["total"]).all()
    assert torch.isfinite(loss_dict["nll"]).all()
    assert torch.isfinite(loss_dict["evidential_reg"]).all()
    loss_dict["total"].backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"Parámetro sin gradiente en Skew-t: {name}"
        assert torch.isfinite(p.grad).all(), f"Gradiente no finito en {name}"
    assert h.grad is not None and torch.isfinite(h.grad).all()

    # 8) Compatibilidad BF16 y FP16 (dominios estrictos y gradientes finitos)
    for dt in (torch.bfloat16, torch.float16):
        head_lp = SkewStudentTDistributionHead(
            hidden_dim=d_model, horizons=horizons, n_ensemble=4
        ).to(dtype=dt)
        h_lp = torch.randn(n_assets, d_model, dtype=dt, requires_grad=True)
        c_lp = torch.rand(n_assets, 1, dtype=dt)
        y_lp = torch.randn(n_assets, n_h, dtype=dt) * 0.02
        d_lp = head_lp(h_lp, conflict_score=c_lp, compute_tail_risk=True)
        assert (d_lp["scale"].float() > 0.0).all()
        assert (d_lp["df"].float() > 2.2).all()
        assert (d_lp["skew"].float().abs() < 1.0).all()
        assert (d_lp["var_01"].float() < d_lp["var_05"].float()).all()
        assert (d_lp["cvar_05"].float() < d_lp["var_05"].float()).all()
        l_lp = head_lp.loss(d_lp, y_lp)
        assert torch.isfinite(l_lp["total"]).all()
        l_lp["total"].backward()
        assert h_lp.grad is not None and torch.isfinite(h_lp.grad).all()
        for name, p in head_lp.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (
                f"Grad no finito en {dt} para {name}"
            )


def test_end_to_end_multimodal_sota_pipeline_determinism_and_zero_host_syncs_ast():
    """Integración conjunta FP32 determinista + auditoría AST cero syncs CPU↔GPU y uso de SDPA."""
    torch.manual_seed(101)
    n_assets, d_model = 16, 32
    fusion = CrossModalConflictFusion(hidden_dim=d_model, n_modalities=5, n_heads=4)
    ssl = SelfSupervisedMultimodalAlignmentLoss(hidden_dim=d_model, n_modalities=5, n_heads=4)
    dist_head = SkewStudentTDistributionHead(
        hidden_dim=d_model, horizons=DEFAULT_HORIZONS, quantile_levels=DEFAULT_QUANTILES
    )

    tokens = torch.randn(n_assets, 5, d_model, requires_grad=True)
    mask = torch.ones(n_assets, 5)
    mask[0::3, 3] = 0.0
    phys = torch.randn(n_assets, d_model, requires_grad=True)
    age = torch.rand(n_assets, 5) * 3.0
    targets = torch.randn(n_assets, len(DEFAULT_HORIZONS)) * 0.02

    # Determinismo bit a bit en eval()
    fusion.eval()
    ssl.eval()
    dist_head.eval()
    with torch.no_grad():
        f1, c1, _ = fusion(tokens.detach(), mask, phys.detach(), modality_age=age)
        d1 = dist_head(f1, conflict_score=c1)
        s1 = ssl(tokens.detach(), mask, physics_emb=phys.detach())

        f2, c2, _ = fusion(tokens.detach(), mask, phys.detach(), modality_age=age)
        d2 = dist_head(f2, conflict_score=c2)
        s2 = ssl(tokens.detach(), mask, physics_emb=phys.detach())

    torch.testing.assert_close(f1, f2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(c1, c2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(d1["cvar_05"], d2["cvar_05"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(s1["total"], s2["total"], atol=0.0, rtol=0.0)

    # Paso conjunto train() con pérdida NLL + SSL
    fusion.train()
    ssl.train()
    dist_head.train()
    fused, conflict, _ = fusion(tokens, mask, phys, modality_age=age)
    dist = dist_head(fused, conflict_score=conflict)
    ssl_out = ssl(tokens, mask, physics_emb=phys)
    dist_loss = dist_head.loss(dist, targets)
    joint_loss = dist_loss["total"] + ssl_out["total"]
    assert torch.isfinite(joint_loss).all()
    joint_loss.backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert phys.grad is not None and torch.isfinite(phys.grad).all()

    # Auditoría AST estática estricta: CERO .item()/.cpu()/.numpy()/.tolist()/.synchronize()
    mod_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mars_titan"
        / "models"
        / "mmarrs"
        / "multimodal_sota.py"
    )
    source = mod_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"item", "cpu", "numpy", "tolist", "synchronize"}
    found_forbidden: list[tuple[int, str]] = []
    uses_sdpa = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in forbidden:
                found_forbidden.append((node.lineno, node.func.attr))
            if node.func.attr == "scaled_dot_product_attention":
                uses_sdpa = True
    assert found_forbidden == [], (
        f"Sincronizaciones CPU↔GPU prohibidas en multimodal_sota.py: {found_forbidden}"
    )
    assert uses_sdpa, "multimodal_sota.py debe usar F.scaled_dot_product_attention nativo"


def test_p1_5_cn_constant_macro_no_var_explosion_and_ssl_weight_scaling():
    """P1-5: macro constante excluido de view_b, CN sin explosión var_loss y escala ssl_weight."""
    torch.manual_seed(73)
    n_assets, n_mod, d_model = 64, 5, 32

    assert ASSET_CONTEXT_VIEW_INDICES == (1, 3)
    ssl = SelfSupervisedMultimodalAlignmentLoss(hidden_dim=d_model, n_modalities=n_mod, n_heads=4)
    assert ssl.view_b_indices == (1, 3)
    assert ssl.ssl_weight == pytest.approx(0.02)

    # 1) Sesión CN / sin fundamentales ni noticias: solo `prices` (0) + `macro` constante (2)
    prices = torch.randn(n_assets, d_model)
    const_macro = torch.randn(1, d_model).expand(n_assets, d_model).clone()
    tokens_cn = torch.zeros(n_assets, n_mod, d_model)
    tokens_cn[:, 0, :] = prices
    tokens_cn[:, 2, :] = const_macro
    tokens_cn.requires_grad_(True)

    mask_cn_zero = torch.zeros(n_assets, n_mod)
    mask_cn_zero[:, 0] = 1.0
    mask_cn_zero[:, 2] = 1.0  # macro constante presente en toda la sesión

    out_cn0 = ssl(tokens_cn, mask_cn_zero)
    assert torch.isfinite(out_cn0["total"]).all()
    assert float(out_cn0["vicreg_variance"].detach()) == 0.0
    assert float(out_cn0["alignment"].detach()) == 0.0
    assert float(out_cn0["total"].detach()) < 0.5
    out_cn0["total"].backward()
    assert tokens_cn.grad is not None and torch.isfinite(tokens_cn.grad).all()

    # 2) Sesión CN con exactamente 1 activo con noticias (< 2 activos con señal asset-varying)
    ssl.zero_grad()
    tokens_cn1 = tokens_cn.detach().clone()
    tokens_cn1[0, 3, :] = torch.randn(d_model)
    tokens_cn1.requires_grad_(True)
    mask_cn_one = mask_cn_zero.clone()
    mask_cn_one[0, 3] = 1.0  # solo 1 acción con news -> debe omitir/anular view_b sin explosión
    out_cn1 = ssl(tokens_cn1, mask_cn_one)
    assert float(out_cn1["vicreg_variance"].detach()) == 0.0
    assert float(out_cn1["alignment"].detach()) == 0.0
    assert float(out_cn1["total"].detach()) < 0.5
    out_cn1["total"].backward()
    assert tokens_cn1.grad is not None and torch.isfinite(tokens_cn1.grad).all()

    # 3) Escala `ssl_weight` configurable (0.02..0.05) y sobreescritura en forward()
    ssl.eval()
    tok_full = torch.randn(n_assets, n_mod, d_model)
    mask_full = torch.ones(n_assets, n_mod)
    with torch.no_grad():
        out_002 = ssl(tok_full, mask_full)
        out_005 = ssl(tok_full, mask_full, ssl_weight=0.05)
        torch.testing.assert_close(
            out_002["total"], out_002["raw_total"] * 0.02, atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            out_005["total"], out_002["raw_total"] * 0.05, atol=1e-6, rtol=1e-5
        )
        # No debe ahogar la pérdida principal (~1.5)
        assert float(out_002["total"]) < 2.0

    ssl_custom = SelfSupervisedMultimodalAlignmentLoss(
        hidden_dim=d_model, n_modalities=n_mod, n_heads=4, ssl_weight=0.04
    )
    assert ssl_custom.ssl_weight == pytest.approx(0.04)
    with pytest.raises(ValueError):
        SelfSupervisedMultimodalAlignmentLoss(hidden_dim=d_model, ssl_weight=-0.01)


def test_p1_5_skew_t_compute_tail_risk_skips_ppf_in_train_and_runs_in_eval(monkeypatch):
    """P1-5: train() omite Hill-396 `student_t_ppf` por defecto; eval() / True lo ejecuta."""
    torch.manual_seed(89)
    n_assets, d_model = 16, 32
    head = SkewStudentTDistributionHead(
        hidden_dim=d_model, horizons=DEFAULT_HORIZONS, quantile_levels=DEFAULT_QUANTILES
    )
    h = torch.randn(n_assets, d_model, requires_grad=True)
    conflict = torch.rand(n_assets, 1)
    targets = torch.randn(n_assets, len(DEFAULT_HORIZONS)) * 0.02

    ppf_calls = 0
    orig_ppf = ms_mod.student_t_ppf

    def _spy_ppf(p: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
        nonlocal ppf_calls
        ppf_calls += 1
        return orig_ppf(p, df)

    monkeypatch.setattr(ms_mod, "student_t_ppf", _spy_ppf)

    # 1) En train() con default compute_tail_risk=None -> CERO llamadas a student_t_ppf
    head.train()
    d_train = head(h, conflict_score=conflict)
    assert ppf_calls == 0
    for omitted in ("var_01", "var_05", "cvar_01", "cvar_05", "quantiles"):
        assert omitted not in d_train
    loss_tr = head.loss(d_train, targets)
    assert torch.isfinite(loss_tr["total"]).all()
    loss_tr["total"].backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()

    # 2) En train() con compute_tail_risk=True -> SÍ calcula VaR/CVaR y cuantiles
    ppf_calls = 0
    d_train_tr = head(h.detach(), conflict_score=conflict, compute_tail_risk=True)
    assert ppf_calls > 0
    for key in ("var_01", "var_05", "cvar_01", "cvar_05", "quantiles"):
        assert key in d_train_tr and torch.isfinite(d_train_tr[key]).all()
    assert (d_train_tr["var_01"] < d_train_tr["var_05"]).all()
    assert (d_train_tr["cvar_05"] < d_train_tr["var_05"]).all()

    # 3) En eval() con default compute_tail_risk=None -> SÍ calcula VaR/CVaR y cuantiles
    head.eval()
    ppf_calls = 0
    with torch.no_grad():
        d_eval = head(h.detach(), conflict_score=conflict)
    assert ppf_calls > 0
    for key in ("var_01", "var_05", "cvar_01", "cvar_05", "quantiles"):
        assert key in d_eval and torch.isfinite(d_eval[key]).all()

    # 4) En eval() con compute_tail_risk=False -> CERO llamadas a student_t_ppf y paridad exacta
    ppf_calls = 0
    with torch.no_grad():
        d_eval_fast = head(h.detach(), conflict_score=conflict, compute_tail_risk=False)
    assert ppf_calls == 0
    for omitted in ("var_01", "var_05", "cvar_01", "cvar_05", "quantiles"):
        assert omitted not in d_eval_fast
    for core_key in ("loc", "mean", "scale", "df", "skew", "aleatoric_var", "epistemic_var"):
        torch.testing.assert_close(d_eval[core_key], d_eval_fast[core_key], atol=0.0, rtol=0.0)


def test_p0_3_skew_t_nll_and_loss_sanitize_nan_targets_before_mask():
    """P0-3: NaN/Inf en targets (enmascarados o no) jamás producen `NaN * 0 = NaN` ni grad NaN."""
    torch.manual_seed(303)
    n_assets, d_model = 16, 32
    head = SkewStudentTDistributionHead(
        hidden_dim=d_model, horizons=DEFAULT_HORIZONS, quantile_levels=DEFAULT_QUANTILES
    )
    n_h = len(DEFAULT_HORIZONS)
    h = torch.randn(n_assets, d_model, requires_grad=True)
    conflict = torch.rand(n_assets, 1)
    dist = head(h, conflict_score=conflict)

    # Targets con NaN/+inf/-inf en posiciones enmascaradas (mask==0) y en posiciones activas
    targets = torch.randn(n_assets, n_h) * 0.02
    mask = torch.ones(n_assets, n_h)
    targets[0, 0] = float("nan")
    mask[0, 0] = 0.0  # clásico NaN * 0.0 -> antes moría al instante!
    targets[1, :] = float("nan")
    mask[1, :] = 0.0
    targets[2, 2] = float("inf")
    mask[2, 2] = 0.0
    targets[3, 1] = float("-inf")
    mask[3, 1] = 1.0  # incluso sin enmascarar, debe intersectarse con torch.isfinite

    for red in ("mean", "sum", "none"):
        nll = skew_student_t_nll(
            targets, dist["loc"], dist["scale"], dist["df"], dist["skew"], reduction=red, mask=mask
        )
        assert torch.isfinite(nll).all(), f"nll no finito en reduction={red}"
        if red == "none":
            assert float(nll[0, 0].detach()) == 0.0
            assert float(nll[3, 1].detach()) == 0.0

    # También con mask=None cuando hay targets NaN (activos excluidos de cotización)
    nll_nomask = skew_student_t_nll(
        targets, dist["loc"], dist["scale"], dist["df"], dist["skew"], reduction="mean", mask=None
    )
    assert torch.isfinite(nll_nomask).all()

    loss_d = head.loss(dist, targets, mask=mask)
    for k in ("total", "nll", "evidential_reg"):
        assert torch.isfinite(loss_d[k]).all(), f"loss[{k}] no finito con targets NaN"
    loss_d["total"].backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    for name, p in head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"grad NaN en {name}"

    # Sesión completamente NaN -> pérdida finita 0 y gradientes finitos 0
    head.zero_grad()
    h2 = torch.randn(8, d_model, requires_grad=True)
    dist2 = head(h2)
    loss_all_nan = head.loss(dist2, torch.full((8, n_h), float("nan")), mask=torch.zeros(8, n_h))
    assert torch.isfinite(loss_all_nan["total"]).all()
    loss_all_nan["total"].backward()
    assert h2.grad is not None and torch.isfinite(h2.grad).all()


def test_p0_3_student_t_ppf_no_nan_grad_near_c_mod_zero_singularity():
    """P0-3: backward de `student_t_ppf` cerca de `df=2.22, z≈3.13` (`c_mod==0`) jamás da NaN."""
    # Barrido denso incluyendo p≈8.7e-4 (z≈3.13) donde c_mod(z) cruza exactamente 0 para df=2.22
    ps = torch.cat(
        [
            torch.linspace(1e-5, 3e-3, 1024, dtype=torch.float32),
            torch.tensor([0.00086, 0.00087, 0.00088, 0.001, 0.005, 0.01, 0.05, 0.5, 0.99]),
        ]
    )
    df = torch.full_like(ps, 2.22).requires_grad_(True)
    out = ms_mod.student_t_ppf(ps, df)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert df.grad is not None
    assert torch.isfinite(df.grad).all(), "Singularity 0*inf -> NaN en backward de student_t_ppf!"

    # Verificar VaR/CVaR y cuantiles con df=2.22 y sesgos extremos [-0.94, +0.94]
    sk = torch.linspace(-0.94, 0.94, 64, dtype=torch.float32, requires_grad=True)
    df2 = torch.full_like(sk, 2.22).requires_grad_(True)
    loc = torch.zeros_like(sk).requires_grad_(True)
    scale = torch.full_like(sk, 0.02).requires_grad_(True)
    v01, cv01 = skew_student_t_var_cvar(loc, scale, df2, sk, 0.01)
    qs = ms_mod.skew_student_t_quantiles(loc, scale, df2, sk)
    assert torch.isfinite(v01).all() and torch.isfinite(cv01).all() and torch.isfinite(qs).all()
    (v01.sum() + cv01.sum() + qs.sum()).backward()
    for t_name, t in (("loc", loc), ("scale", scale), ("df", df2), ("skew", sk)):
        assert t.grad is not None and torch.isfinite(t.grad).all(), f"grad NaN en {t_name}"


def test_p0_2_skew_t_head_registers_taus_buffer_and_no_forward_tensor_alloc():
    """P0-2: `SkewStudentTDistributionHead` registra buffer de cuantiles `taus` sin H2D en fwd."""
    head = SkewStudentTDistributionHead(
        hidden_dim=32, horizons=DEFAULT_HORIZONS, quantile_levels=DEFAULT_QUANTILES
    )
    buf_names = dict(head.named_buffers())
    assert any("tau" in k or "quantile" in k for k in buf_names), (
        f"Se esperaba buffer de cuantiles registrado en SkewStudentTDistributionHead; "
        f"buffers actuales: {list(buf_names.keys())}"
    )

    if torch.cuda.is_available():
        head_gpu = head.cuda().eval()
        h_gpu = torch.randn(64, 32, device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            d_gpu = head_gpu(h_gpu, compute_tail_risk=True)
        assert d_gpu["quantiles"].is_cuda and torch.isfinite(d_gpu["quantiles"]).all()
        compiled = torch.compile(head_gpu, mode="default", dynamic=True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            d_c1 = compiled(torch.randn(48, 32, device="cuda"), compute_tail_risk=True)
            d_c2 = compiled(torch.randn(64, 32, device="cuda"), compute_tail_risk=True)
        assert torch.isfinite(d_c1["quantiles"]).all()
        assert torch.isfinite(d_c2["quantiles"]).all()

