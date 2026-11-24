"""Pruebas TDD de física de mercado y rasgos multimodales v4 (`market_physics_v4`)."""

from __future__ import annotations

import numpy as np
import pytest

from mars_titan.data import market_physics_v4 as mp4


def _synthetic_ohlcv(rng: np.random.Generator, n: int = 320):
    rets = 0.0003 + 0.015 * rng.standard_normal(n)
    close = 50.0 * np.exp(np.cumsum(rets))
    open_ = close * np.exp(0.004 * rng.standard_normal(n))
    high = np.maximum(open_, close) * np.exp(np.abs(0.007 * rng.standard_normal(n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(0.007 * rng.standard_normal(n)))
    volume = np.exp(12.0 + 0.6 * rng.standard_normal(n))
    return open_, high, low, close, volume


def test_asset_microstructure_v4_shape_finite_and_strictly_causal():
    """Mutando precios futuros tras t0, ninguna fila <= t0 puede cambiar bit a bit."""
    rng = np.random.default_rng(7)
    o, h, lo, c, v = _synthetic_ohlcv(rng, n=320)
    f1 = mp4.asset_microstructure_v4(o, h, lo, c, v)
    assert f1.shape == (320, len(mp4.ASSET_V4_NAMES))
    assert f1.dtype == np.float32
    assert np.isfinite(f1).all()

    # Mutar violentamente desde t0 + 1 en adelante
    t0 = 180
    o2, h2, lo2, c2, v2 = o.copy(), h.copy(), lo.copy(), c.copy(), v.copy()
    c2[t0 + 1 :] *= 3.5
    h2[t0 + 1 :] *= 4.0
    lo2[t0 + 1 :] *= 0.2
    v2[t0 + 1 :] *= 10.0
    f2 = mp4.asset_microstructure_v4(o2, h2, lo2, c2, v2)
    np.testing.assert_array_equal(f1[: t0 + 1], f2[: t0 + 1])
    # Y sí cambia después de t0
    assert not np.array_equal(f1[t0 + 1 :], f2[t0 + 1 :])


def test_path_signature_order2_distinguishes_event_ordering():
    """Las áreas de Lévy de orden 2 distinguen 'sube precio luego volumen' de lo inverso."""
    # Camino A: primero sube X, luego sube Y
    path_a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    # Camino B: primero sube Y, luego sube X (mismo punto inicial y final!)
    path_b = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    sig_a = mp4.path_signature_order2(path_a)
    sig_b = mp4.path_signature_order2(path_b)
    # Incrementos de orden 1 son idénticos
    np.testing.assert_allclose(sig_a[:2], sig_b[:2], atol=1e-12)
    # El área firmada de Lévy (orden 2 antisimétrica) tiene signo opuesto
    levy_a = sig_a[3] - sig_a[4]  # S^{0,1} - S^{1,0}
    levy_b = sig_b[3] - sig_b[4]
    assert levy_a == pytest.approx(1.0, abs=1e-12)
    assert levy_b == pytest.approx(-1.0, abs=1e-12)


def test_rmt_marchenko_pastur_separates_true_modes_from_noise():
    """RMT limpia una matriz con 3 factores reales y estima Absorption Ratio causal."""
    rng = np.random.default_rng(13)
    t_len, n_assets = 120, 40
    factors = rng.standard_normal((t_len, 3))
    loadings = rng.standard_normal((n_assets, 3))
    returns = factors @ loadings.T + 0.5 * rng.standard_normal((t_len, n_assets))

    res = mp4.rmt_clean_correlation(returns, top_k=5)
    assert res["n_signal_modes"] >= 3
    assert 0.0 < res["absorption_ratio"] <= 1.0
    assert res["spectral_entropy"] > 0.0
    assert res["clean_corr"].shape == (n_assets, n_assets)
    # Simétrica y definida positiva
    np.testing.assert_allclose(res["clean_corr"], res["clean_corr"].T, atol=1e-10)
    assert float(np.linalg.eigvalsh(res["clean_corr"]).min()) > 0.0
    assert res["loadings"].shape == (n_assets, 5)


def test_wasserstein_regime_shift_detects_volatility_break_causally():
    """La distancia de Wasserstein dispara ante cambio de régimen y es estrictamente causal."""
    rng = np.random.default_rng(21)
    t_len, n_assets = 120, 50
    panel = 0.01 * rng.standard_normal((t_len, n_assets))
    # Shock estructural en t >= 90 (volatilidad 5x y sesgo negativo)
    panel[90:] = -0.02 + 0.05 * rng.standard_normal((30, n_assets))

    w_dist = mp4.causal_wasserstein_shift(panel, short_window=5, long_window=40, maturity_lag=1)
    assert w_dist.shape == (t_len,)
    assert np.isfinite(w_dist).all()
    calm_mean = float(np.mean(w_dist[50:89]))
    shock_mean = float(np.mean(w_dist[93:105]))
    assert shock_mean > calm_mean * 3.0

    # Causalidad estricta: mutar t >= 80 no cambia w_dist[:81]
    panel_mut = panel.copy()
    panel_mut[80:] *= 20.0
    w_mut = mp4.causal_wasserstein_shift(panel_mut, short_window=5, long_window=40, maturity_lag=1)
    np.testing.assert_allclose(w_dist[:81], w_mut[:81], atol=1e-12)


def test_causal_lead_lag_spillover_recovers_lagged_leader_signal():
    """Un seguidor retardado 1 día respecto a líderes líquidos obtiene señal predictiva causal."""
    rng = np.random.default_rng(99)
    t_len, n_assets = 140, 30
    liq = np.linspace(10.0, 20.0, n_assets)  # últimos 6 son los líderes más líquidos
    liq_panel = np.broadcast_to(liq[None, :], (t_len, n_assets)).copy()
    returns = 0.01 * rng.standard_normal((t_len, n_assets))
    # Factor común captado primero por líderes líquidos (efecto Lo-MacKinlay / Hou)
    leader_factor = 0.015 * rng.standard_normal(t_len)
    returns[:, 24:] += leader_factor[:, None]
    leader_mean = returns[:, 24:].mean(axis=1)
    for t in range(1, t_len):
        returns[t, :24] += 0.8 * leader_mean[t - 1]

    spill = mp4.causal_lead_lag_spillover(
        returns, liq_panel, window=40, leader_frac=0.2, maturity_lag=1
    )
    assert spill.shape == (t_len, n_assets)
    assert np.isfinite(spill).all()
    # Correlación positiva fuerte fuera de muestra para seguidores en t >= 60
    corr = np.corrcoef(spill[60:, :24].ravel(), returns[60:, :24].ravel())[0, 1]
    assert corr > 0.50

    # Causalidad estricta: mutar returns[80:] no afecta spill[:81]
    ret_mut = returns.copy()
    ret_mut[80:] *= -50.0
    spill_mut = mp4.causal_lead_lag_spillover(
        ret_mut, liq_panel, window=40, leader_frac=0.2, maturity_lag=1
    )
    np.testing.assert_allclose(spill[:81], spill_mut[:81], atol=1e-12)


def test_panel_market_physics_and_high_snr_context_are_causal():
    """El pipeline completo de física de panel y contexto de alto SNR es finito y causal."""
    rng = np.random.default_rng(31)
    t_len, n_assets = 100, 36
    returns = 0.015 * rng.standard_normal((t_len, n_assets))
    liq = rng.normal(15.0, 1.5, size=(t_len, n_assets))
    vol = np.abs(0.015 + 0.005 * rng.standard_normal((t_len, n_assets)))
    macro = rng.standard_normal((t_len, 4))

    phys = mp4.panel_market_physics_v4(
        returns, liq, vol, macro=macro, rmt_window=40, maturity_lag=1
    )
    assert phys["rmt_absorption"].shape == (t_len,)
    assert phys["wasserstein"].shape == (t_len,)
    assert phys["lead_lag"].shape == (t_len, n_assets)
    assert phys["macro_surprises"].shape == (t_len, 4)
    assert phys["macro_betas"].shape == (t_len, n_assets, 4)
    assert phys["high_snr_context"].shape[0] == t_len
    assert phys["high_snr_context"].shape[1] == n_assets
    assert np.isfinite(phys["high_snr_context"]).all()

    # Verificar causalidad estricta en t = 75
    ret2 = returns.copy()
    ret2[75:] += 0.5
    phys2 = mp4.panel_market_physics_v4(ret2, liq, vol, macro=macro, rmt_window=40, maturity_lag=1)
    np.testing.assert_allclose(
        phys["high_snr_context"][:76], phys2["high_snr_context"][:76], atol=1e-10
    )


def test_multimodal_fusion_features_v4_handles_masks_novelty_and_macro_tailwind():
    """Fusión multimodal v4 calcula novedad, tailwind macro y respeta máscaras y causalidad."""
    rng = np.random.default_rng(123)
    t_len, n_assets = 45, 16
    news_emb = rng.standard_normal((t_len, n_assets, 8)).astype(np.float32)
    news_present = rng.random((t_len, n_assets)) > 0.6
    chart_emb = rng.standard_normal((t_len, n_assets, 8)).astype(np.float32)
    chart_present = np.ones((t_len, n_assets), dtype=bool)
    fund_vals = rng.standard_normal((t_len, n_assets, 6)).astype(np.float32)
    fund_present = np.ones((t_len, n_assets), dtype=bool)
    macro_surp = rng.standard_normal((t_len, 4)).astype(np.float64)
    macro_betas = rng.standard_normal((t_len, n_assets, 4)).astype(np.float64)

    mm = mp4.multimodal_fusion_features_v4(
        news_emb=news_emb,
        news_present=news_present,
        chart_emb=chart_emb,
        chart_present=chart_present,
        fund_vals=fund_vals,
        fund_present=fund_present,
        macro_surprises=macro_surp,
        macro_betas=macro_betas,
    )
    assert mm["features"].shape == (t_len, n_assets, len(mp4.MULTIMODAL_V4_NAMES))
    assert np.isfinite(mm["features"]).all()
    assert mm["modality_presence"].shape == (t_len, n_assets, 4)

    # Causalidad estricta: mutar noticias y macro en t >= 30 no altera t <= 29
    news2 = news_emb.copy()
    news2[30:] = -50.0
    macro2 = macro_surp.copy()
    macro2[30:] = 25.0
    mm2 = mp4.multimodal_fusion_features_v4(
        news_emb=news2,
        news_present=news_present,
        chart_emb=chart_emb,
        chart_present=chart_present,
        fund_vals=fund_vals,
        fund_present=fund_present,
        macro_surprises=macro2,
        macro_betas=macro_betas,
    )
    np.testing.assert_allclose(mm["features"][:30], mm2["features"][:30], atol=1e-6)


def test_asset_microstructure_v4_exact_rolling_pv_corr_skew_kurt_and_halt_masking():
    """pv_corr_21/skew_21/kurt_21 coinciden con ventanas directas y volume<=0 no satura Amihud."""
    from scipy import stats

    rng = np.random.default_rng(77)
    o, h, lo, c, v = _synthetic_ohlcv(rng, n=120)
    # Inyectar día de suspensión (volume == 0) con salto de cierre para verificar que no pega a 8.0
    v[50] = 0.0
    c[50] = c[49] * 1.04
    h[50] = max(o[50], c[50])
    lo[50] = min(o[50], c[50])

    feat = mp4.asset_microstructure_v4(o, h, lo, c, v)
    idx = {name: i for i, name in enumerate(mp4.ASSET_V4_NAMES)}

    # pv_corr_21 estrictamente en [-1, 1]
    pv = feat[:, idx["price_vol_corr_21"]]
    assert float(pv.min()) >= -1.0 - 1e-6
    assert float(pv.max()) <= 1.0 + 1e-6

    # En t=100 (sin halts en [80..100]), verificar exactitud frente a ventana directa [80..100]
    t_chk = 100
    log_c = np.log(c)
    ret_cc = np.zeros_like(log_c)
    ret_cc[1:] = log_c[1:] - log_c[:-1]
    log_v = np.log1p(v)
    r_win = ret_cc[t_chk - 20 : t_chk + 1]
    v_win = log_v[t_chk - 20 : t_chk + 1]
    assert feat[t_chk, idx["price_vol_corr_21"]] == pytest.approx(
        float(np.corrcoef(r_win, v_win)[0, 1]), abs=1e-5
    )
    assert feat[t_chk, idx["realized_skew_21"]] == pytest.approx(
        float(stats.skew(r_win, bias=True)), abs=1e-5
    )
    assert feat[t_chk, idx["realized_kurt_21"]] == pytest.approx(
        float(stats.kurtosis(r_win, fisher=True, bias=True)), abs=1e-5
    )

    # En t=50 (halt con volume==0), Amihud NO debe dispararse a LIMIT (8.0)
    assert float(feat[50, idx["amihud_up_21"]]) < 4.0
    # Activo completamente suspendido (volume==0 todos los días) -> Amihud neutral 0.0
    feat_halt = mp4.asset_microstructure_v4(o, h, lo, c, np.zeros_like(v))
    assert float(np.abs(feat_halt[:, idx["amihud_up_21"]]).max()) == 0.0
    assert float(np.abs(feat_halt[:, idx["amihud_down_21"]]).max()) == 0.0

    # Detección de splits sin ajustar (2-for-1: -50% overnight => |log(0.5)| = 0.693 > 0.55)
    assert mp4.detect_unadjusted_split_jumps(o, c)["n_jumps"] == 0
    o_split, c_split = o.copy(), c.copy()
    o_split[60:] *= 0.5
    c_split[60:] *= 0.5
    rep_split = mp4.detect_unadjusted_split_jumps(o_split, c_split)
    assert rep_split["n_jumps"] >= 1


def test_rmt_pairwise_overlap_avoids_shrinkage_on_halted_assets():
    """RMT divide por solape válido cnt_ij-1 y evita encoger correlaciones en halts."""
    rng = np.random.default_rng(51)
    t_len, n_assets = 63, 12
    r = 0.015 * rng.standard_normal((t_len, n_assets))
    # Activos 0 y 1 solo operan en las últimas 30 sesiones y son casi idénticos (rho ~ 1.0)
    base = 0.02 * rng.standard_normal(30)
    r[:33, 0] = np.nan
    r[:33, 1] = np.nan
    r[33:, 0] = base
    r[33:, 1] = base + 1e-4 * rng.standard_normal(30)
    # Activos 2 y 3 solo solapan 2 días (< min_overlap) -> correlación protegida en 0
    r[:-2, 2] = np.nan
    r[2:, 3] = np.nan

    res = mp4.rmt_clean_correlation(r, top_k=3, min_overlap=5)
    # Sin el sesgo (cnt-1)/(t_len-1), el modo común entre 0 y 1 mantiene correlación alta (> 0.85)
    assert float(res["raw_corr"][0, 1]) > 0.99
    assert float(res["raw_corr"][2, 3]) == 0.0


def test_causal_lead_lag_enforces_maturity_lag_on_leader_ret():
    """Lead-Lag usa leader_ret[t - maturity_lag] y rechaza maturity_lag < 1."""
    rng = np.random.default_rng(64)
    t_len, n_assets = 90, 20
    r = 0.015 * rng.standard_normal((t_len, n_assets))
    liq = rng.normal(14.0, 1.0, size=(t_len, n_assets))

    spill1 = mp4.causal_lead_lag_spillover(r, liq, window=30, maturity_lag=2)
    r_mut = r.copy()
    liq_mut = liq.copy()
    r_mut[70:] *= -40.0
    liq_mut[70:] += 100.0
    spill2 = mp4.causal_lead_lag_spillover(r_mut, liq_mut, window=30, maturity_lag=2)
    # Con maturity_lag=2, mutar t >= 70 NO puede alterar t <= 71 ([:72])
    np.testing.assert_array_equal(spill1[:72], spill2[:72])

    with pytest.raises(ValueError, match="maturity_lag"):
        mp4.causal_lead_lag_spillover(r, liq, maturity_lag=0)


