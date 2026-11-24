"""Pruebas TDD para `dataset_enrichment_sota`: resiliencia ante ausencias y señales SOTA."""

from __future__ import annotations

import ast
import inspect
import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from mars_titan.data import dataset_enrichment_sota as des
from mars_titan.data.market_physics_v4_gpu import gaussian_rank_features
from mars_titan.models.mmarrs.blackwell_kernels import audit_zero_host_syncs
from mars_titan.models.mmarrs.model import build_family_model


def _synthetic_ohlcv_panel(
    rng: np.random.Generator,
    t_len: int = 160,
    n_assets: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drifts = rng.uniform(-0.0003, 0.0005, size=(1, n_assets))
    vols = rng.uniform(0.010, 0.025, size=(1, n_assets))
    rets = drifts + vols * rng.standard_normal((t_len, n_assets))
    base_px = rng.uniform(15.0, 200.0, size=(1, n_assets))
    close = base_px * np.exp(np.cumsum(rets, axis=0))
    open_ = close * np.exp(0.003 * rng.standard_normal((t_len, n_assets)))
    high = np.maximum(open_, close) * np.exp(np.abs(0.006 * rng.standard_normal((t_len, n_assets))))
    low = np.minimum(open_, close) * np.exp(-np.abs(0.006 * rng.standard_normal((t_len, n_assets))))
    vol_base = rng.uniform(11.0, 15.0, size=(1, n_assets))
    volume = np.exp(vol_base + 0.45 * rng.standard_normal((t_len, n_assets)))
    return open_, high, low, close, volume


def test_missingness_resilient_featurizer_anatomy_eb_imputation_and_dual_level_rank():
    """Verifica anatomía 3 canales, imputación EB jerárquica, nivel+rango y frescura."""
    torch.manual_seed(11)
    n_assets, n_feats = 60, 4

    # 3 sectores de 20 activos cada uno (one-hot exposures [N, 3])
    exposures = torch.zeros((n_assets, 3), dtype=torch.float32)
    exposures[0:20, 0] = 1.0
    exposures[20:40, 1] = 1.0
    exposures[40:60, 2] = 1.0

    x = torch.zeros((n_assets, n_feats), dtype=torch.float32)
    # Sector 0 alrededor de +3.0, Sector 1 alrededor de -3.0, Sector 2 alrededor de +0.5
    x[0:20, 0] = 3.0 + 0.15 * torch.randn(20)
    x[20:40, 0] = -3.0 + 0.15 * torch.randn(20)
    x[40:60, 0] = 0.5 + 0.15 * torch.randn(20)
    # Feature 1: sector 2 completamente ausente (debe caer al EB transversal -> ~0)
    x[0:20, 1] = 2.0 + 0.1 * torch.randn(20)
    x[20:40, 1] = 2.0 + 0.1 * torch.randn(20)
    x[40:60, 1] = float("nan")
    # Feature 2: 100% ausente (debe caer exactamente a 0.0)
    x[:, 2] = float("nan")
    # Feature 3: outlier extremo (+5000.0) + empates exactos
    x[:, 3] = torch.linspace(-1.0, 1.0, n_assets)
    x[10:15, 3] = 0.25
    x[0, 3] = 5000.0

    # Ocultar 1 activo en sector 0 y 1 activo en sector 1 en feature 0
    x[0, 0] = float("nan")
    x[20, 0] = float("nan")

    staleness_days = torch.zeros((n_assets, n_feats), dtype=torch.float32)
    staleness_days[1, 0] = 0.0
    staleness_days[2, 0] = 63.0
    staleness_days[3, 0] = 500.0
    staleness_days[0, 0] = 999.0  # ausente -> decay debe ser 0.0 exacto!

    featurizer = des.MissingnessResilientFeaturizer(
        winsor_sigma=5.0,
        clip_val=8.0,
        prior_strength_peer=5.0,
        prior_strength_cs=5.0,
        staleness_halflife_days=63.0,
    )
    out = featurizer(x, staleness=staleness_days, exposures=exposures)

    # Anatomía exacta de 3 canales y representación dual nivel + rango gaussiano
    assert out["sanitized_value"].shape == (n_assets, n_feats)
    assert out["availability_mask"].shape == (n_assets, n_feats)
    assert out["log1p_staleness_decay"].shape == (n_assets, n_feats)
    assert out["anatomy_3ch"].shape == (n_assets, 3 * n_feats)
    assert out["dual_level_rank"].shape == (n_assets, 2 * n_feats)
    assert out["features"].shape == (n_assets, 4 * n_feats)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            assert torch.isfinite(v).all(), f"Tensor {k} contiene valores no finitos"

    val = out["sanitized_value"]
    mask = out["availability_mask"]
    decay = out["log1p_staleness_decay"]
    grank = out["masked_gaussian_rank"]

    # 1) Jerarquía EB: ausente de sector 0 -> ~+2.3..+3.0; ausente de sector 1 -> ~-2.3..-3.0
    assert float(val[0, 0]) > 1.8
    assert float(val[20, 0]) < -1.8
    assert float(mask[0, 0]) == 0.0 and float(mask[1, 0]) == 1.0
    # 2) Sector 2 todo ausente en feat 1 -> cae hacia mediana transversal (~+2.0 con contracción EB)
    assert float(val[45, 1]) > 1.3
    # 3) Feature 2 100% ausente -> exactamente 0.0 en valor, máscara, decay y rango
    assert torch.equal(val[:, 2], torch.zeros(n_assets))
    assert torch.equal(mask[:, 2], torch.zeros(n_assets))
    assert torch.equal(decay[:, 2], torch.zeros(n_assets))
    assert torch.equal(grank[:, 2], torch.zeros(n_assets))
    # 4) Winsorización robusta de outlier extremo +5000.0 en feat 3
    assert float(val[0, 3]) <= 8.0
    assert float(val[0, 3]) < 5.0  # winsorizado a ~med + 5*MAD << 5000
    # 5) Frescura log1p monótona decreciente y 0.0 exacto en ausentes
    assert float(decay[1, 0]) == pytest.approx(1.0, abs=1e-5)
    assert float(decay[2, 0]) == pytest.approx(0.5, abs=1e-4)
    assert 0.0 < float(decay[3, 0]) < float(decay[2, 0])
    assert float(decay[0, 0]) == 0.0
    # 6) Rango gaussiano enmascarado: ausentes = 0.0 exacto; observados coinciden con reference
    assert float(grank[0, 0]) == 0.0
    assert float(grank[20, 0]) == 0.0
    ref_grank = gaussian_rank_features(
        torch.where(torch.isfinite(x), x, torch.zeros_like(x)),
        mask=torch.isfinite(x),
        clip_val=8.0,
    )
    torch.testing.assert_close(grank[:, 0], ref_grank[:, 0], atol=1e-4, rtol=1e-4)
    # 7) Preservación de nivel entre sesiones (3x vol) + invarianza de rango en [T, N, F]
    panel = torch.stack([0.20 + 0.04 * torch.randn(40, 2), 0.60 + 0.12 * torch.randn(40, 2)], dim=0)
    # Sesión 1 = 3x sesión 0 para probar preservación de nivel vs invarianza de rango
    panel[1] = 3.0 * panel[0]
    p_out = featurizer(panel)
    assert float(p_out["robust_winsorized_z"][1].mean()) == pytest.approx(
        3.0 * float(p_out["robust_winsorized_z"][0].mean()), rel=1e-3
    )
    torch.testing.assert_close(
        p_out["masked_gaussian_rank"][0], p_out["masked_gaussian_rank"][1], atol=1e-5, rtol=1e-5
    )

    # 8) Modo `robust_cs_z` evita saturación +8.0 en fundamentales US/CN log(Assets) in [20.9, 29.3]
    # y garantiza 0.0 exacto cuando n_obs == 1 (p. ej. 99.2% ausente en CN!)
    fund_log_assets = torch.linspace(20.9, 29.3, 50).unsqueeze(-1)
    fund_log_assets[0:10, 0] = float("nan")
    f_csz = des.MissingnessResilientFeaturizer(level_mode="robust_cs_z")(fund_log_assets)
    z_obs = f_csz["robust_winsorized_z"][10:, 0]
    assert float(z_obs.std()) > 0.5
    assert float(z_obs.abs().max()) < 8.0

    fund_single = torch.full((50, 1), float("nan"))
    fund_single[0, 0] = 23.76
    f_single = des.MissingnessResilientFeaturizer(level_mode="robust_cs_z")(fund_single)
    assert float(f_single["robust_winsorized_z"].abs().max()) == 0.0

    # Imputación por pares en `robust_cs_z` con desplazamiento +24.0 (sin sesgo de offset)
    fund_2sec = torch.full((40, 1), 26.0)
    fund_2sec[20:, 0] = 22.0
    fund_2sec[0, 0] = float("nan")
    fund_2sec[20, 0] = float("nan")
    sec_2 = torch.zeros((40, 2))
    sec_2[:20, 0] = 1.0
    sec_2[20:, 1] = 1.0
    f_2sec = des.MissingnessResilientFeaturizer(level_mode="robust_cs_z")(
        fund_2sec, exposures=sec_2
    )
    assert float(f_2sec["robust_winsorized_z"][0, 0]) > 0.5
    assert float(f_2sec["robust_winsorized_z"][20, 0]) < -0.5

    # 9) Entradas vacías T=0 / N=0 seguras
    empty_2d = torch.empty((0, 4), dtype=torch.float32)
    empty_out = featurizer(empty_2d)
    assert empty_out["features"].shape == (0, 16)


def test_missingness_resilient_featurizer_nan_inf_stress_zero_grad_leak_and_all_4_families():
    """Estrés 50%/90%/100% ausente con NaN/Inf: salida finita, 0 fuga de gradiente y 4 familias."""
    torch.manual_seed(42)
    n_assets, n_feats = 48, 8
    featurizer = des.MissingnessResilientFeaturizer()

    for drop_rate in (0.50, 0.90, 1.00):
        base = torch.randn((n_assets, n_feats), dtype=torch.float32)
        miss = torch.rand((n_assets, n_feats)) < drop_rate
        if drop_rate == 1.0:
            miss[:] = True
        raw = base.clone()
        # Mezclar NaN, +Inf y -Inf en las ranuras ausentes
        idx_flat = torch.arange(n_assets * n_feats).view(n_assets, n_feats)
        raw = torch.where(miss & (idx_flat % 3 == 0), torch.tensor(float("nan")), raw)
        raw = torch.where(miss & (idx_flat % 3 == 1), torch.tensor(float("inf")), raw)
        raw = torch.where(miss & (idx_flat % 3 == 2), torch.tensor(float("-inf")), raw)
        raw.requires_grad_(True)

        out = featurizer(raw, availability_mask=~miss)
        feats = out["features"]
        assert torch.isfinite(feats).all()
        assert float(feats.detach().min()) >= -featurizer.clip_val
        assert float(feats.detach().max()) <= featurizer.clip_val

        loss = (feats**2).sum()
        loss.backward()
        assert raw.grad is not None
        assert torch.isfinite(raw.grad).all()
        assert float(raw.grad[miss].abs().max()) == 0.0

    # Integración con las 4 familias independientes bajo 99.3% ausente en CN + 100% sin noticias
    n_s = 32
    exposures = torch.randn(n_s, 6)
    rmt = torch.randn(n_s, 4)
    mod_mask = torch.ones((n_s, 5), dtype=torch.float32)
    mod_mask[1:, 1] = 0.0  # 96.9% sin fundamentales (solo 1 activo presente)
    mod_mask[:, 3] = 0.0  # 100% sin noticias

    fund_dirty = torch.full((n_s, 10), float("nan"))
    fund_dirty[0] = torch.linspace(21.0, 27.0, 10)  # escala real CN log(CNY)
    news_dirty = torch.full((n_s, 8), float("inf"))
    raw_batch = dict(
        asset_ids=torch.arange(n_s, dtype=torch.long),
        prices=torch.randn(n_s, 12),
        fundamentals=fund_dirty,
        macro=torch.randn(1, 6).expand(n_s, 6).clone(),
        news=news_dirty,
        charts=torch.randn(n_s, 8),
        physics_v4=torch.randn(n_s, 10),
        high_snr_context=torch.randn(n_s, 6),
        exposures=exposures,
        rmt_loadings=rmt,
        modality_mask=mod_mask,
        modality_age=torch.tensor([0.0, 297.0, 1.0, 0.0, 0.0]).expand(n_s, 5).clone(),
    )
    featurizer.eval()
    clean_batch = featurizer.featurize_batch(raw_batch)
    for key in ("prices", "fundamentals", "macro", "news", "charts", "physics_v4"):
        assert torch.isfinite(clean_batch[key]).all(), f"NaN/Inf residual en {key}"

    dim_kw = dict(
        prices_dim=12,
        fundamentals_dim=10,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=10,
        context_dim=6,
        exposures_dim=6,
        rmt_dim=4,
        hidden_dim=32,
        max_loops=2,
        allow_short_synthetic=True,
    )
    family_variants = (
        ("titans_original", "titans_neuromod_sota"),
        ("holographic_brain", "nb_v2_full"),
        ("mmarrs_titan", "v13b_mmarrs_multimodal_sota"),
        ("titans_cognitive_core", "cog_v2_10tools_soft_k3"),
    )
    for fam, var in family_variants:
        model = build_family_model(var, family=fam, **dim_kw)
        model.eval()
        with torch.no_grad():
            res = model.forward_session(clean_batch)
        assert torch.isfinite(res["pred_return"]).all(), f"NaN en pred_return para {fam}/{var}"
        assert torch.isfinite(res["alpha"]).all(), f"NaN en alpha para {fam}/{var}"
        assert torch.isfinite(res["vol_pred"]).all(), f"NaN en vol_pred para {fam}/{var}"


def test_structured_modality_dropout_train_vs_eval_and_prices_preservation():
    """Dropout estructurado solo en train(), preserva precios siempre y sin NaN*0."""
    n_s = 64
    batch = dict(
        prices=torch.randn(n_s, 10),
        fundamentals=torch.full((n_s, 8), float("nan")),
        macro=torch.randn(n_s, 6),
        news=torch.randn(n_s, 8),
        charts=torch.randn(n_s, 8),
        modality_mask=torch.ones((n_s, 5), dtype=torch.float32),
        modality_age=torch.full((n_s, 5), 10.0, dtype=torch.float32),
    )
    batch["fundamentals"][:32] = torch.randn(32, 8)
    batch["modality_mask"][32:, 1] = 0.0

    dropper = des.StructuredModalityDropout(market_preset="CN")
    # En eval(): identidad bit a bit
    dropper.eval()
    eval_out = dropper(batch)
    assert torch.equal(eval_out["modality_mask"], batch["modality_mask"])
    assert torch.equal(eval_out["prices"], batch["prices"])

    # En train(): precios 100% intactos; auxiliares dropeadas y saneadas sin NaN
    dropper.train()
    gen = torch.Generator().manual_seed(123)
    dropped_counts = torch.zeros(4)
    for _ in range(12):
        tr_out = dropper(batch, generator=gen)
        assert torch.equal(tr_out["modality_mask"][:, 0], batch["modality_mask"][:, 0])
        assert torch.equal(tr_out["prices"], batch["prices"])
        assert torch.isfinite(tr_out["fundamentals"]).all()
        assert torch.isfinite(tr_out["news"]).all()
        assert torch.isfinite(tr_out["charts"]).all()
        # Donde modalidad fue apagada, tensor y edad deben ser exactamente 0.0
        for col, key in ((1, "fundamentals"), (2, "macro"), (3, "news"), (4, "charts")):
            off = tr_out["modality_mask"][:, col] < 0.5
            if off.any():
                assert float(tr_out[key][off].abs().max()) == 0.0
                assert float(tr_out["modality_age"][off, col].abs().max()) == 0.0
        dropped_counts += (tr_out["modality_mask"][:, 1:] < 0.5).float().mean(dim=0)

    # CN preset debe dropear fundamentales y noticias sustancialmente durante train
    assert float(dropped_counts[0] / 12.0) > 0.50
    assert float(dropped_counts[2] / 12.0) > 0.40


def test_enrich_full_dataset_sota_torch_economic_recovery_and_all_18_signals():
    """Recupera los 24 rasgos SOTA con fidelidad económica y enmascarado modal exacto."""
    assert des.ENRICHED_SOTA_DIM == len(des.ENRICHED_SOTA_NAMES) == 24
    idx = {name: i for i, name in enumerate(des.ENRICHED_SOTA_NAMES)}

    rng = np.random.default_rng(2026)
    t_len, n_assets = 120, 12
    rets = 0.008 * rng.standard_normal((t_len, n_assets))

    # Activo 0: difusión gaussiana limpia sin saltos
    rets[:, 0] = 0.008 * rng.standard_normal(t_len)
    # Activo 1: difusión suave + 2 saltos abruptos (+18%, -16%) en últimos 21 días
    rets[:, 1] = 0.004 * rng.standard_normal(t_len)
    rets[-10, 1] = 0.18
    rets[-4, 1] = -0.16
    # Activo 2: volatilidad con fuertes cambios de régimen (alto vol-of-vol 63d)
    vov_scale = np.where((np.arange(t_len) // 5) % 2 == 0, 0.003, 0.035)
    rets[:, 2] = vov_scale * rng.standard_normal(t_len)
    # Activo 3: semivarianza positiva (días alcistas grandes, bajistas pequeños)
    rets[-21:, 3] = np.where(np.arange(21) % 2 == 0, 0.030, -0.003)
    # Activo 4: semivarianza negativa (días bajistas grandes, alcistas pequeños)
    rets[-21:, 4] = np.where(np.arange(21) % 2 == 0, 0.003, -0.030)
    # Activo 5: tendencia unidireccional pura (+1%/día -> Kaufman ER ~ 1.0)
    rets[-21:, 5] = 0.010
    # Activo 6: zig-zag puro alternante (+1%, -1% -> Kaufman ER ~ 0.0)
    rets[-21:, 6] = np.where(np.arange(21) % 2 == 0, 0.010, -0.010)
    # Activo 7: acumulación con volumen anormal en días positivos
    rets[-21:, 7] = np.where(np.arange(21) % 2 == 0, 0.015, -0.015)
    # Activo 8: iliquidez asimétrica en días bajistas (volumen muy bajo al caer, alto al subir)
    rets[-21:, 8] = np.where(np.arange(21) % 2 == 0, 0.015, -0.015)

    close = 50.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = close * np.exp(0.001 * rng.standard_normal((t_len, n_assets)))
    # Ajustar rango H-L para activo 2 (vol-of-vol en rango)
    hl_mult = np.full((t_len, n_assets), 0.008, dtype=np.float64)
    hl_mult[:, 2] = vov_scale * 1.5
    high = np.maximum(open_, close) * np.exp(hl_mult)
    low = np.minimum(open_, close) * np.exp(-hl_mult)
    vol = np.full((t_len, n_assets), 1e6, dtype=np.float64)
    # Activo 7: volumen 10x en días alcistas, 0.2x en bajistas
    vol[-21:, 7] = np.where(rets[-21:, 7] > 0, 1e7, 2e5)
    # Activo 8: volumen 0.05x en días bajistas (alta iliquidez down), 5x en alcistas
    vol[-21:, 8] = np.where(rets[-21:, 8] < 0, 5e4, 5e6)

    # Exposures de 2 sectores: activos 0..5 sector 0, activos 6..11 sector 1
    exposures = np.zeros((t_len, n_assets, 2), dtype=np.float32)
    exposures[:, :6, 0] = 1.0
    exposures[:, 6:, 1] = 1.0

    # Fundamentales en formato real `encoded_v31_78` (26 vals, 26 obs, 26 log_age)
    # Activo 0: balance sólido y fresco; Activo 1: débil y muy antiguo; Activo 2: ausente (mask=0)
    fund_78 = np.zeros((t_len, n_assets, 78), dtype=np.float32)
    fund_mask = np.ones((t_len, n_assets), dtype=bool)
    fund_mask[:, 2] = False
    for a in range(n_assets):
        if a == 2:
            continue
        strength = 1.0 if a == 0 else (-1.0 if a == 1 else float(a - 5) * 0.1)
        # Slots 0..7 USD balance, 8..14 company ratios
        fund_78[:, a, 0] = 22.0 + strength  # Assets
        fund_78[:, a, 4] = 21.0 + 1.5 * strength  # Equity
        fund_78[:, a, 5] = 20.0 + 1.5 * strength  # Cash
        fund_78[:, a, 8] = 2.5 + strength  # current_ratio
        fund_78[:, a, 9] = 0.35 + 0.15 * strength  # working_capital_to_assets
        fund_78[:, a, 10] = 0.40 - 0.15 * strength  # liabilities_to_assets (menor es mejor)
        fund_78[:, a, 11] = 0.60 + 0.15 * strength  # equity_to_assets
        fund_78[:, a, 12] = 0.20 + 0.05 * strength  # receivables
        fund_78[:, a, 13] = 0.10 - 0.03 * strength  # payables
        fund_78[:, a, 14] = 0.70 - 0.20 * strength  # liabilities_to_positive_equity
        fund_78[:, a, 26 + np.r_[0, 4, 5, 8:15]] = 1.0
        age_d = 5.0 if a == 0 else (400.0 if a == 1 else 45.0)
        fund_78[:, a, 52 + np.r_[0, 4, 5, 8:15]] = np.log1p(age_d)

    # News sentiment, chart emb, macro (con ausentes explícitos en activo 2!)
    news_sent = rng.standard_normal((t_len, n_assets)).astype(np.float32)
    news_sent[-1, 5] = -3.0  # precio sube fuerte, noticia muy negativa -> divergencia positiva!
    news_mask = np.ones((t_len, n_assets), dtype=bool)
    news_mask[:, 2] = False
    news_sent[:, 2] = np.nan

    chart_emb = rng.standard_normal((t_len, n_assets, 16)).astype(np.float32)
    chart_mask = np.ones((t_len, n_assets), dtype=bool)
    chart_mask[:, 2] = False

    macro = rng.standard_normal((t_len, 12)).astype(np.float32)
    macro[-1] = 6.0  # shock macro fuerte al final
    macro_mask = np.ones((t_len, n_assets), dtype=bool)
    macro_mask[:, 2] = False

    out = des.enrich_full_dataset_sota_torch(
        torch.from_numpy(open_),
        torch.from_numpy(high),
        torch.from_numpy(low),
        torch.from_numpy(close),
        torch.from_numpy(vol),
        peer_exposures=torch.from_numpy(exposures),
        fundamentals=torch.from_numpy(fund_78),
        fundamentals_mask=torch.from_numpy(fund_mask),
        news=torch.from_numpy(news_sent),
        news_mask=torch.from_numpy(news_mask),
        charts=torch.from_numpy(chart_emb),
        charts_mask=torch.from_numpy(chart_mask),
        macro=torch.from_numpy(macro),
        macro_mask=torch.from_numpy(macro_mask),
    )

    assert out.shape == (t_len, n_assets, des.ENRICHED_SOTA_DIM)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert float(out.min()) >= -des.LIMIT
    assert float(out.max()) <= des.LIMIT

    # 1) Jump vs Continuous Vol (BNS)
    assert (
        float(out[-1, 1, idx["jump_var_ratio_21"]])
        > float(out[-1, 0, idx["jump_var_ratio_21"]]) + 0.35
    )
    assert float(out[-1, 2, idx["vol_of_vol_63"]]) > float(out[-1, 0, idx["vol_of_vol_63"]]) + 0.25
    assert float(out[-1, 3, idx["realized_skew_signed_21"]]) > 0.80
    assert float(out[-1, 4, idx["realized_skew_signed_21"]]) < -0.80

    # 2) Peer-residualized mom, Kaufman ER, signed abnormal turnover, Amihud asymmetry
    assert float(out[-1, 5, idx["kaufman_er_21"]]) > 0.90
    assert float(out[-1, 6, idx["kaufman_er_21"]]) < 0.20
    assert float(out[-1, 7, idx["signed_abn_turn_21"]]) > 0.30
    assert float(out[-1, 8, idx["amihud_asym_up_down_21"]]) > 0.15
    assert float(out[-1, 5, idx["peer_resid_mom_21"]]) > float(out[-1, 4, idx["peer_resid_mom_21"]])

    # 3) Cross-modal contradictions: presente recupera signo, ausente (activo 2) == 0.0 exacto!
    assert float(out[-1, 5, idx["price_vs_news_divergence"]]) > 0.8
    for cm_feat in (
        "price_vs_news_divergence",
        "price_vs_fund_quality_gap",
        "chart_vs_vol_exhaustion",
        "macro_stress_x_downside_beta",
        "pit_fund_quality",
        "pit_fund_value",
        "pit_fund_profitability",
    ):
        assert float(out[:, 2, idx[cm_feat]].abs().max()) == 0.0, f"{cm_feat} no fue 0.0 al faltar"

    # 4) PIT Fundamental composites: activo 0 (fuerte & fresco) > activo 1 (débil & obsoleto)
    assert float(out[-1, 0, idx["pit_fund_quality"]]) > float(out[-1, 1, idx["pit_fund_quality"]])
    assert float(out[-1, 0, idx["pit_fund_profitability"]]) > float(
        out[-1, 1, idx["pit_fund_profitability"]]
    )

    # 5) Sin modalidades auxiliares (solo OHLCV): todos los bloques cruzados/fundamentales == 0.0
    ohlcv_only = des.enrich_full_dataset_sota_torch(open_, high, low, close, vol)
    assert ohlcv_only.shape == (t_len, n_assets, des.ENRICHED_SOTA_DIM)
    assert torch.isfinite(ohlcv_only).all()
    for cm_feat in (
        "price_vs_news_divergence",
        "price_vs_fund_quality_gap",
        "chart_vs_vol_exhaustion",
        "macro_stress_x_downside_beta",
        "pit_fund_quality",
        "pit_fund_value",
        "pit_fund_profitability",
        "avail_surprise_21",
    ):
        assert float(ohlcv_only[:, :, idx[cm_feat]].abs().max()) == 0.0

    # 6) Soporte 1D [T] y vacío T=0
    out_1d = des.enrich_full_dataset_sota_torch(
        open_[:, 0], high[:, 0], low[:, 0], close[:, 0], vol[:, 0]
    )
    assert out_1d.shape == (t_len, des.ENRICHED_SOTA_DIM)
    assert torch.isfinite(out_1d).all()

    empty = torch.empty((0, 4), dtype=torch.float32)
    assert des.enrich_full_dataset_sota_torch(empty, empty, empty, empty, empty).shape == (
        0,
        4,
        des.ENRICHED_SOTA_DIM,
    )


def test_enrich_sota_bit_exact_causality_and_cn_zero_volume_halts():
    """Causalidad bit-a-bit estricta (diff==0.0 en <=t_cut) y seguridad en halts CN vol==0."""
    rng = np.random.default_rng(77)
    t_len, n_assets = 140, 14
    o, h, lo, c, v = _synthetic_ohlcv_panel(rng, t_len=t_len, n_assets=n_assets)
    exposures = rng.standard_normal((t_len, n_assets, 4)).astype(np.float32)
    fund = rng.standard_normal((t_len, n_assets, 15)).astype(np.float32)
    news = rng.standard_normal((t_len, n_assets, 12)).astype(np.float32)
    charts = rng.standard_normal((t_len, n_assets, 12)).astype(np.float32)
    macro = rng.standard_normal((t_len, 10)).astype(np.float32)
    t_cut = 95

    f1 = des.enrich_full_dataset_sota_torch(
        o,
        h,
        lo,
        c,
        v,
        peer_exposures=exposures,
        fundamentals=fund,
        news=news,
        charts=charts,
        macro=macro,
    )

    o2, h2, lo2, c2, v2 = o.copy(), h.copy(), lo.copy(), c.copy(), v.copy()
    exp2, fund2, news2, charts2, macro2 = (
        exposures.copy(),
        fund.copy(),
        news.copy(),
        charts.copy(),
        macro.copy(),
    )
    c2[t_cut + 1 :] *= 4.2
    h2[t_cut + 1 :] *= 5.0
    lo2[t_cut + 1 :] *= 0.2
    v2[t_cut + 1 :] *= 20.0
    exp2[t_cut + 1 :] = -9.0
    fund2[t_cut + 1 :] = 12.0
    news2[t_cut + 1 :] = -12.0
    charts2[t_cut + 1 :] = 15.0
    macro2[t_cut + 1 :] = -25.0

    f2 = des.enrich_full_dataset_sota_torch(
        o2,
        h2,
        lo2,
        c2,
        v2,
        peer_exposures=exp2,
        fundamentals=fund2,
        news=news2,
        charts=charts2,
        macro=macro2,
    )
    assert torch.equal(f1[: t_cut + 1], f2[: t_cut + 1])
    assert not torch.equal(f1[t_cut + 1 :], f2[t_cut + 1 :])

    # Causalidad bit-a-bit también en MissingnessResilientFeaturizer (3D causal_expanding_z)
    featurizer = des.MissingnessResilientFeaturizer(level_mode="causal_expanding_z")
    p1 = featurizer(torch.from_numpy(fund), exposures=torch.from_numpy(exposures))["features"]
    p2 = featurizer(torch.from_numpy(fund2), exposures=torch.from_numpy(exp2))["features"]
    assert torch.equal(p1[: t_cut + 1], p2[: t_cut + 1])

    # Estrés de halts CN: activo 0 100% suspendido (vol=0), activo 1 suspendido 30 días con salto
    v_cn = v.copy()
    c_cn = c.copy()
    v_cn[:, 0] = 0.0
    v_cn[50:80, 1] = 0.0
    c_cn[50:80, 1] = c_cn[49, 1]
    c_cn[80:, 1] *= 1.30  # salto al reanudar tras suspensión de 30 sesiones
    out_cn = des.enrich_full_dataset_sota_torch(o, h, lo, c_cn, v_cn)
    idx = {name: i for i, name in enumerate(des.ENRICHED_SOTA_NAMES)}
    assert torch.isfinite(out_cn).all()
    assert float(out_cn.min()) >= -des.LIMIT and float(out_cn.max()) <= des.LIMIT
    # Activo 0 totalmente suspendido debe quedar neutro (0.0) en BV, jump, vol-of-vol, Amihud, turn
    for col in (
        "bv_21",
        "jump_var_ratio_21",
        "vol_of_vol_63",
        "signed_abn_turn_21",
        "amihud_asym_up_down_21",
    ):
        assert float(out_cn[:, 0, idx[col]].abs().max()) == 0.0


def test_bf16_autocast_and_zero_host_syncs_audit():
    """Seguridad BF16 autocast + audit_zero_host_syncs == 0 en todo el módulo."""
    mod_path = Path(des.__file__).resolve()
    sync_rep = audit_zero_host_syncs(mod_path)
    assert sync_rep["files_scanned"] == 1
    assert sync_rep["total_findings"] == 0, f"Host-syncs detectados: {sync_rep['findings']}"
    assert sync_rep["forward_findings"] == 0
    assert sync_rep["clean"] is True

    # Verificar también cero `.nonzero()` y cero `torch.tensor(...)` en rutas calientes
    tree = ast.parse(inspect.getsource(des))
    hot_names = {
        "enrich_full_dataset_sota_torch",
        "MissingnessResilientFeaturizer",
        "StructuredModalityDropout",
    }
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in hot_names:
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and child.attr == "nonzero":
                    raise AssertionError(f"Llamada `.nonzero()` detectada en {node.name}")
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "tensor"
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "torch"
                ):
                    raise AssertionError(f"Copia H2D `torch.tensor(...)` en {node.name}")

    # BF16 autocast end-to-end
    rng = np.random.default_rng(314)
    o, h, lo, c, v = _synthetic_ohlcv_panel(rng, t_len=80, n_assets=20)
    o_bf, h_bf, lo_bf, c_bf, v_bf = (
        torch.from_numpy(a).to(torch.bfloat16) for a in (o, h, lo, c, v)
    )
    featurizer = des.MissingnessResilientFeaturizer()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        enr_bf = des.enrich_full_dataset_sota_torch(o_bf, h_bf, lo_bf, c_bf, v_bf)
        feat_bf = featurizer(enr_bf[-1])
    assert enr_bf.dtype == torch.float32
    assert torch.isfinite(enr_bf).all()
    assert torch.isfinite(feat_bf["features"]).all()


def test_audit_dataset_enrichment_quality_end_to_end():
    """Audita IC, fuga <0.10, redundancia, causalidad bit-exact y estrés 50/90/100%."""
    rng = np.random.default_rng(909)
    t_len, n_assets = 90, 32
    o, h, lo, c, v = _synthetic_ohlcv_panel(rng, t_len=t_len, n_assets=n_assets)
    # Retorno futuro estrictamente t+1 independiente (sin fuga)
    log_c = np.log(c)
    fwd_ret = np.zeros_like(log_c)
    fwd_ret[:-1] = log_c[1:] - log_c[:-1]
    fwd_ret[-1] = np.nan

    exposures = rng.standard_normal((t_len, n_assets, 4)).astype(np.float32)
    fund = rng.standard_normal((t_len, n_assets, 15)).astype(np.float32)
    news = rng.standard_normal((t_len, n_assets, 8)).astype(np.float32)
    charts = rng.standard_normal((t_len, n_assets, 8)).astype(np.float32)
    macro = rng.standard_normal((t_len, 8)).astype(np.float32)

    feats = des.enrich_full_dataset_sota_torch(
        o,
        h,
        lo,
        c,
        v,
        peer_exposures=exposures,
        fundamentals=fund,
        news=news,
        charts=charts,
        macro=macro,
    )
    rep = des.audit_dataset_enrichment_quality(
        feats,
        fwd_ret,
        feature_names=des.ENRICHED_SOTA_NAMES,
        ohlcv=(o, h, lo, c, v),
        modalities=dict(
            peer_exposures=exposures,
            fundamentals=fund,
            news=news,
            charts=charts,
            macro=macro,
        ),
    )

    assert rep["all_finite"] is True
    assert rep["leakage_screen_passed"] is True
    assert rep["max_abs_rank_ic"] < 0.10
    assert rep["max_abs_pearson_ic"] < 0.10
    assert rep["causality_check"]["bit_exact_causal"] is True
    assert rep["causality_check"]["max_abs_diff"] == 0.0
    assert rep["missingness_stress"]["all_passed"] is True
    assert set(rep["missingness_stress"]["levels"].keys()) == {"0.5", "0.9", "1.0"}
    for lvl_rep in rep["missingness_stress"]["levels"].values():
        assert lvl_rep["all_finite"] is True
        assert lvl_rep["within_bounds"] is True
        assert lvl_rep["max_nan_slot_grad"] == 0.0
    assert "missingness_coverage" in rep
    assert "ic_hac_t_stat" in rep

    # Inyectar un rasgo con fuga explícita (futuro t+1) -> la auditoría DEBE rechazarlo!
    leaky = feats.clone()
    leaky[:, :, 0] = torch.from_numpy(np.nan_to_num(fwd_ret, nan=0.0).astype(np.float32))
    rep_leak = des.audit_dataset_enrichment_quality(
        leaky, fwd_ret, feature_names=des.ENRICHED_SOTA_NAMES, run_stress=False
    )
    assert rep_leak["leakage_screen_passed"] is False
    assert rep_leak["max_abs_rank_ic"] > 0.10
    assert des.ENRICHED_SOTA_NAMES[0] in rep_leak["leakage_suspects"]


def test_n2600_speed_and_real_laboratorio_readonly_audit():
    """Rendimiento vectorizado en N=2600 y auditoría solo lectura de cachés reales <2024."""
    # 1) Velocidad vectorizada en N=2600 (CPU time inmune a contención de scheduler `nice -n 10`)
    n_large, f_dim = 2600, 46
    x_big = torch.randn((n_large, f_dim), dtype=torch.float32)
    x_big[::7] = float("nan")
    exp_big = torch.randn((n_large, 8), dtype=torch.float32)
    featurizer = des.MissingnessResilientFeaturizer()
    t0 = time.process_time()
    out_big = featurizer(x_big, exposures=exp_big)
    dt_feat = time.process_time() - t0
    assert out_big["features"].shape == (n_large, 4 * f_dim)
    assert torch.isfinite(out_big["features"]).all()
    assert dt_feat < 10.0, f"Featurizer demasiado lento en N=2600: {dt_feat:.3f}s CPU"

    rng = np.random.default_rng(2600)
    o, h, lo, c, v = (
        torch.from_numpy(a.astype(np.float32))
        for a in _synthetic_ohlcv_panel(rng, t_len=80, n_assets=n_large)
    )
    t1 = time.process_time()
    enr_big = des.enrich_full_dataset_sota_torch(o, h, lo, c, v)
    dt_enr = time.process_time() - t1
    assert enr_big.shape == (80, n_large, des.ENRICHED_SOTA_DIM)
    assert torch.isfinite(enr_big).all()
    assert dt_enr < 20.0, f"Enrichment demasiado lento en T=80,N=2600: {dt_enr:.3f}s CPU"

    # 2) Auditoría solo lectura sobre laboratorio real (< 2024 estrictamente!) si es legible
    lab_root = Path("/home/gpic/.local/state/mars-titan/laboratorio")
    for cache_name in ("x3-us-2013-tabla", "x3-cn-2013-real"):
        cdir = lab_root / cache_name
        if not (cdir / "X.npy").is_file() or not (cdir / "X3.npy").is_file():
            continue
        rep = des.audit_laboratorio_cache_readonly(cdir, max_rows_sample=50_000)
        assert rep["strictly_pre_2024"] is True
        assert rep["max_day_epoch"] < 19723  # 2024-01-01
        assert rep["n_sessions"] > 500
        # Alimentar una sesión real (con ~33-37% NaNs en X3!) al MissingnessResilientFeaturizer
        bounds = np.load(cdir / "bounds.npy", mmap_mode="r")
        s_idx = min(len(bounds) - 2, 1000)
        a, b = int(bounds[s_idx]), int(bounds[s_idx + 1])
        x3_sess = torch.from_numpy(np.array(np.load(cdir / "X3.npy", mmap_mode="r")[a:b]))
        res_real = featurizer(x3_sess)
        assert torch.isfinite(res_real["features"]).all()
        assert float(res_real["features"].abs().max()) <= featurizer.clip_val
        assert float(res_real["sanitized_value"].std()) > 1e-4


def test_adversarial_regressions_176_177_bf16_triplets_fx_halts_and_seal(tmp_path: Path):
    """Regresiones #176/#177: BF16+exposures, tripletas 78/420, invarianza FX, halts y sello."""
    # 1) P0-1: Cero `linalg.solve` (solo `solve_ex`) y paridad exacta FP32 bajo BF16 autocast
    src = inspect.getsource(des)
    assert "linalg.solve(" not in src, "Usar `linalg.solve_ex(..., check_errors=False)` sin sync"

    rng = np.random.default_rng(176)
    t_len, n_a = 80, 32
    o, h, lo, c, v = (
        torch.from_numpy(a.astype(np.float32)) for a in _synthetic_ohlcv_panel(rng, t_len, n_a)
    )
    # Exposures fuertemente correlacionadas (explotan en normal equations si bmm cae a BF16!)
    base_e = torch.randn(t_len, n_a, 1)
    corr_exp = torch.cat([base_e + 1e-2 * torch.randn(t_len, n_a, 1) for _ in range(4)], dim=-1)
    fund = torch.randn(t_len, n_a, 15)
    news = torch.randn(t_len, n_a, 8)
    charts = torch.randn(t_len, n_a, 8)
    macro = torch.randn(t_len, 10)

    enr_fp32 = des.enrich_full_dataset_sota_torch(
        o,
        h,
        lo,
        c,
        v,
        peer_exposures=corr_exp,
        fundamentals=fund,
        news=news,
        charts=charts,
        macro=macro,
    )
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        enr_amp = des.enrich_full_dataset_sota_torch(
            o,
            h,
            lo,
            c,
            v,
            peer_exposures=corr_exp,
            fundamentals=fund,
            news=news,
            charts=charts,
            macro=macro,
        )
    torch.testing.assert_close(enr_amp, enr_fp32, atol=1e-5, rtol=1e-5)

    # También featurizer con exposures bajo BF16 autocast debe coincidir con FP32
    feat_mod = des.MissingnessResilientFeaturizer()
    x_fp32 = torch.randn(n_a, 12)
    x_fp32[::5] = float("nan")
    f_fp32 = feat_mod(x_fp32, exposures=corr_exp[-1])["features"]
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        f_amp = feat_mod(x_fp32, exposures=corr_exp[-1])["features"]
    torch.testing.assert_close(f_amp, f_fp32, atol=1e-5, rtol=1e-5)

    # 2) P0-2: `featurize_batch` preserva tripletas `encoded_v31_78` (fund) y `420` (macro)
    n_b = 40
    f78 = torch.zeros(n_b, 78)
    # 10 activos observan slot 0 en escala real log(USD) ~ [21..26], resto ausentes (0.0 + obs=0)
    f78[:10, 0] = torch.linspace(21.0, 26.0, 10)
    f78[:10, 26 + 0] = 1.0
    f78[:10, 52 + 0] = torch.log1p(torch.linspace(5.0, 120.0, 10))
    m420 = torch.zeros(n_b, 420)
    m420[:, :50] = 1.5
    m420[:, 140 : 140 + 50] = 1.0
    m420[:, 280 : 280 + 50] = 2.0
    mm = torch.ones(n_b, 5)
    mm[10:, 1] = 0.0
    batch_tri = dict(
        prices=torch.randn(n_b, 12),
        fundamentals=f78,
        macro=m420,
        news=torch.randn(n_b, 8),
        charts=torch.randn(n_b, 8),
        exposures=torch.randn(n_b, 4),
        modality_mask=mm,
    )
    feat_mod.eval()
    cb = feat_mod.featurize_batch(batch_tri)
    f_v, f_o, f_l = torch.split(cb["fundamentals"], 26, dim=-1)
    # Máscaras siguen siendo estrictamente {0.0, 1.0} y `log_age` intacto en observados!
    assert torch.all((f_o == 0.0) | (f_o == 1.0))
    torch.testing.assert_close(f_o, f78[:, 26:52])
    torch.testing.assert_close(f_l[:10, 0], f78[:10, 52 + 0])
    assert float(f_l[10:].abs().max()) == 0.0
    # Valores observados bien centrados alrededor de 0 (no distorsionados por los 30 ceros!)
    assert abs(float(f_v[:10, 0].median())) < 0.25
    assert float(f_v[:10, 0].std()) > 0.5
    _, m_o, m_l = torch.split(cb["macro"], 140, dim=-1)
    assert torch.all((m_o == 0.0) | (m_o == 1.0))
    torch.testing.assert_close(m_l[:, :50], torch.full((n_b, 50), 2.0))

    # 3) P0-3: Invarianza FX exacta en `pit_fund_value` (USD slot 4 vs CNY slot 25)
    f_fx1 = torch.zeros(t_len, n_a, 78)
    # Mitad USD (slot 4), mitad CNY (slot 25)
    half = n_a // 2
    f_fx1[:, :half, 4] = torch.linspace(20.0, 24.0, half)
    f_fx1[:, :half, 26 + 4] = 1.0
    f_fx1[:, :half, 52 + 4] = 3.0
    f_fx1[:, half:, 25] = torch.linspace(20.0, 24.0, n_a - half)
    f_fx1[:, half:, 26 + 25] = 1.0
    f_fx1[:, half:, 52 + 25] = 3.0
    f_fx2 = f_fx1.clone()
    f_fx2[:, half:, 25] = f_fx2[:, half:, 25] + 1.95  # cambio FX log(7.0) solo en CNY

    idx = {name: i for i, name in enumerate(des.ENRICHED_SOTA_NAMES)}
    v1 = des.enrich_full_dataset_sota_torch(o, h, lo, c, v, fundamentals=f_fx1)
    v2 = des.enrich_full_dataset_sota_torch(o, h, lo, c, v, fundamentals=f_fx2)
    assert torch.equal(v1[:, :, idx["pit_fund_value"]], v2[:, :, idx["pit_fund_value"]])

    # 4) P1-1 + P1-2: mutación bool/age en auditoría + consistencia de halts en `_cum_ret_horizon`
    f_age = torch.full((t_len, n_a), 20.0)
    f_mask = torch.ones((t_len, n_a), dtype=torch.bool)
    fwd = torch.zeros((t_len, n_a))
    fwd[:-1] = torch.log(c[1:] / c[:-1])
    rep_bool = des.audit_dataset_enrichment_quality(
        enr_fp32,
        fwd,
        feature_names=des.ENRICHED_SOTA_NAMES,
        ohlcv=(o, h, lo, c, v),
        modalities=dict(fundamentals=fund, fundamentals_mask=f_mask, fundamentals_age=f_age),
        run_stress=False,
    )
    assert rep_bool["causality_check"]["bit_exact_causal"] is True
    assert rep_bool["causality_check"]["max_abs_diff"] == 0.0

    # Si t-5 estaba suspendido (vol=0), ret_5[t] debe quedar enmascarado (peer_resid_mom_5 == 0)
    v_halt = v.clone()
    v_halt[40, 0] = 0.0
    out_halt = des.enrich_full_dataset_sota_torch(o, h, lo, c, v_halt)
    assert float(out_halt[45, 0, idx["peer_resid_mom_5"]]) == 0.0

    # 5) #177: `audit_laboratorio_cache_readonly` rechaza `day.npy` desordenado con dato 2024+
    unsorted_days = np.asarray([19800, 15000, 16000], dtype=np.int32)
    np.save(tmp_path / "day.npy", unsorted_days)
    np.save(tmp_path / "X.npy", np.zeros((3, 4), dtype=np.float32))
    np.save(tmp_path / "y.npy", np.zeros((3,), dtype=np.float64))
    np.save(tmp_path / "bounds.npy", np.asarray([0, 3], dtype=np.int64))
    with pytest.raises(ValueError, match="2024"):
        des.audit_laboratorio_cache_readonly(tmp_path)


def test_enrich_sota_v2_24_signals_rank_vel_accel_vol_surge_cn_limit_halt_and_avail_surprise():
    """#216/#195-5B: 6 señales nuevas (18->24), causalidad bit-exact, BF16 FP32-master y CN."""
    new_six = (
        "rank_vel_5_21",
        "rank_accel_5_21_63",
        "peer_rev_5_x_idio_vol_surge",
        "limit_hit_streak_5",
        "post_halt_age_decay",
        "avail_surprise_21",
    )
    assert des.ENRICHED_SOTA_DIM == len(des.ENRICHED_SOTA_NAMES) == 24
    for name in new_six:
        assert name in des.ENRICHED_SOTA_NAMES
    idx = {name: i for i, name in enumerate(des.ENRICHED_SOTA_NAMES)}

    rng = np.random.default_rng(21601)
    t_len, n_assets = 130, 14
    rets = 0.006 * rng.standard_normal((t_len, n_assets))

    # Activo 0: rezagado fuerte en 21d (caída -21..-5), rebota en 5d -> rank_vel & rank_accel > 0
    rets[-63:-21, 0] = 0.004
    rets[-21:-5, 0] = -0.015
    rets[-5:, 0] = 0.020
    # Activo 1: líder fuerte en 21d (subida -21..-5), se gira en 5d -> rank_vel & rank_accel < 0
    rets[-63:-21, 1] = -0.004
    rets[-21:-5, 1] = 0.015
    rets[-5:, 1] = -0.020
    # Activo 2: caída 5d con surge fuerte de volatilidad bipower 5d vs 21d -> peer_rev*vol_surge > 0
    rets[:, 2] = 0.003 * rng.standard_normal(t_len)
    rets[-5:, 2] = np.array([-0.040, 0.025, -0.045, 0.020, -0.040])
    # Activo 3: caída 5d tranquila (misma vol baja) -> peer_rev*vol_surge <= 0
    rets[:, 3] = 0.009 * rng.standard_normal(t_len)
    rets[-5:, 3] = np.array([-0.008, -0.008, -0.008, -0.008, -0.008])
    # Activo 4: CN racha limit-up +10% (3 días seguidos en t=80..82) y halt en t=83
    rets[80:83, 4] = np.log(1.10)
    # Activo 5: CN racha limit-down -10% (2 días seguidos en t=80..81)
    rets[80:82, 5] = np.log(0.90)

    close = 50.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = close * np.exp(0.001 * rng.standard_normal((t_len, n_assets)))
    high = np.maximum(open_, close) * 1.008
    low = np.minimum(open_, close) * 0.992
    vol = np.full((t_len, n_assets), 1e6, dtype=np.float64)

    # Activo 4: halt en t=83 (vol=0) y reapertura en t=84,85,88
    vol[83, 4] = 0.0
    # Activo 6: pre-IPO vacío (c=NaN, v=0) en t=0..29, IPO en t=30 -> post_halt_age_decay == 0.0!
    close[:30, 6] = np.nan
    open_[:30, 6] = np.nan
    high[:30, 6] = np.nan
    low[:30, 6] = np.nan
    vol[:30, 6] = 0.0
    # Activo 7: fragmentado por halts alternos en últimos 21d -> bp_cnt < 3 -> vol_surge == 0.0!
    vol[-21::2, 7] = 0.0

    # Modalidades auxiliares para probar `avail_surprise_21`:
    # - Activo 8: ausente t=0..79, aparece cobertura en t=80 -> shock positivo fuerte!
    # - Activo 9: siempre presente t=0..79, apagón en t=80 -> shock negativo fuerte!
    # - Activo 10: constante siempre presente -> exacto 0.0!
    # - Activo 11: siempre ausente -> exacto 0.0!
    fund = rng.standard_normal((t_len, n_assets, 12)).astype(np.float32)
    news = rng.standard_normal((t_len, n_assets)).astype(np.float32)
    charts = rng.standard_normal((t_len, n_assets, 8)).astype(np.float32)
    f_mask = np.ones((t_len, n_assets), dtype=bool)
    n_mask = np.ones((t_len, n_assets), dtype=bool)
    c_mask = np.ones((t_len, n_assets), dtype=bool)
    for m in (f_mask, n_mask, c_mask):
        m[:80, 8] = False
        m[80:, 9] = False
        m[:, 11] = False

    out = des.enrich_full_dataset_sota_torch(
        torch.from_numpy(open_.astype(np.float32)),
        torch.from_numpy(high.astype(np.float32)),
        torch.from_numpy(low.astype(np.float32)),
        torch.from_numpy(close.astype(np.float32)),
        torch.from_numpy(vol.astype(np.float32)),
        fundamentals=torch.from_numpy(fund),
        fundamentals_mask=torch.from_numpy(f_mask),
        news=torch.from_numpy(news),
        news_mask=torch.from_numpy(n_mask),
        charts=torch.from_numpy(charts),
        charts_mask=torch.from_numpy(c_mask),
    )
    assert out.shape == (t_len, n_assets, 24)
    assert torch.isfinite(out).all()
    assert float(out.min()) >= -des.LIMIT and float(out.max()) <= des.LIMIT

    # 1) rank_vel_5_21 & rank_accel_5_21_63
    assert float(out[-1, 0, idx["rank_vel_5_21"]]) > 1.0
    assert float(out[-1, 1, idx["rank_vel_5_21"]]) < -1.0
    assert float(out[-1, 0, idx["rank_accel_5_21_63"]]) > 1.0
    assert float(out[-1, 1, idx["rank_accel_5_21_63"]]) < -1.0
    assert float(out[:21, :, idx["rank_vel_5_21"]].abs().max()) == 0.0
    assert float(out[:63, :, idx["rank_accel_5_21_63"]].abs().max()) == 0.0

    # 2) peer_rev_5_x_idio_vol_surge + guardia explícita bp_cnt >= 3
    assert float(out[-1, 2, idx["peer_rev_5_x_idio_vol_surge"]]) > (
        float(out[-1, 3, idx["peer_rev_5_x_idio_vol_surge"]]) + 0.05
    )
    assert float(out[-1, 7, idx["peer_rev_5_x_idio_vol_surge"]]) == 0.0

    # 3) limit_hit_streak_5 (CN 10% limit clustering + máscara en halt)
    assert float(out[82, 4, idx["limit_hit_streak_5"]]) == pytest.approx(3.0, abs=1e-5)
    assert float(out[83, 4, idx["limit_hit_streak_5"]]) == 0.0  # halted!
    assert float(out[81, 5, idx["limit_hit_streak_5"]]) == pytest.approx(-2.0, abs=1e-5)

    # 4) post_halt_age_decay (reapertura exacta exp(-d/5), 0 en halt, nunca-halt y pre-IPO!)
    assert float(out[:, 0, idx["post_halt_age_decay"]].abs().max()) == 0.0
    assert float(out[30, 6, idx["post_halt_age_decay"]]) == 0.0  # IPO != halt!
    assert float(out[83, 4, idx["post_halt_age_decay"]]) == 0.0  # durante halt
    assert float(out[84, 4, idx["post_halt_age_decay"]]) == pytest.approx(
        math.exp(-1.0 / 5.0), rel=1e-5
    )
    assert float(out[85, 4, idx["post_halt_age_decay"]]) == pytest.approx(
        math.exp(-2.0 / 5.0), rel=1e-5
    )
    assert float(out[88, 4, idx["post_halt_age_decay"]]) == pytest.approx(
        math.exp(-5.0 / 5.0), rel=1e-5
    )

    # 5) avail_surprise_21 (onset MNAR +, blackout -, constante == 0, sin aux == 0)
    assert float(out[80, 8, idx["avail_surprise_21"]]) > 0.75
    assert float(out[80, 9, idx["avail_surprise_21"]]) < -0.75
    assert float(out[:, 10, idx["avail_surprise_21"]].abs().max()) < 1e-6
    assert float(out[:, 11, idx["avail_surprise_21"]].abs().max()) == 0.0

    # 6) Causalidad bit-a-bit (diff == 0.0 en <= t_cut) mutando precios, halts y máscaras > t_cut
    t_cut = 86
    c_mut, v_mut = close.copy(), vol.copy()
    c_mut[t_cut + 1 :] *= 2.5
    v_mut[t_cut + 1 :, 0] = 0.0
    fm_mut = f_mask.copy()
    fm_mut[t_cut + 1 :] = ~fm_mut[t_cut + 1 :]
    out_mut = des.enrich_full_dataset_sota_torch(
        torch.from_numpy(open_.astype(np.float32)),
        torch.from_numpy(high.astype(np.float32)),
        torch.from_numpy(low.astype(np.float32)),
        torch.from_numpy(c_mut.astype(np.float32)),
        torch.from_numpy(v_mut.astype(np.float32)),
        fundamentals=torch.from_numpy(fund),
        fundamentals_mask=torch.from_numpy(fm_mut),
        news=torch.from_numpy(news),
        news_mask=torch.from_numpy(n_mask),
        charts=torch.from_numpy(charts),
        charts_mask=torch.from_numpy(c_mask),
    )
    assert torch.equal(out[: t_cut + 1], out_mut[: t_cut + 1])

    # 7) Paridad FP32 maestro bajo BF16 autocast en todas las 24 señales
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_bf16 = des.enrich_full_dataset_sota_torch(
            torch.from_numpy(open_.astype(np.float32)),
            torch.from_numpy(high.astype(np.float32)),
            torch.from_numpy(low.astype(np.float32)),
            torch.from_numpy(close.astype(np.float32)),
            torch.from_numpy(vol.astype(np.float32)),
            fundamentals=torch.from_numpy(fund),
            fundamentals_mask=torch.from_numpy(f_mask),
            news=torch.from_numpy(news),
            news_mask=torch.from_numpy(n_mask),
            charts=torch.from_numpy(charts),
            charts_mask=torch.from_numpy(c_mask),
        )
    torch.testing.assert_close(out_bf16, out, atol=1e-5, rtol=1e-5)

