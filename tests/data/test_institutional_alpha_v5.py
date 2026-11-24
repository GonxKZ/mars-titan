"""Pruebas TDD para el motor institucional de factores v5 (`institutional_alpha_v5`)."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
import torch
from scipy import stats

from mars_titan.data import institutional_alpha_v5 as ia5


def _synthetic_ohlcv_panel(
    rng: np.random.Generator,
    t_len: int = 200,
    n_assets: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drifts = rng.uniform(-0.0002, 0.0006, size=(1, n_assets))
    vols = rng.uniform(0.010, 0.026, size=(1, n_assets))
    rets = drifts + vols * rng.standard_normal((t_len, n_assets))
    base_px = rng.uniform(20.0, 180.0, size=(1, n_assets))
    close = base_px * np.exp(np.cumsum(rets, axis=0))
    open_ = close * np.exp(0.004 * rng.standard_normal((t_len, n_assets)))
    high = np.maximum(open_, close) * np.exp(np.abs(0.006 * rng.standard_normal((t_len, n_assets))))
    low = np.minimum(open_, close) * np.exp(-np.abs(0.006 * rng.standard_normal((t_len, n_assets))))
    vol_base = rng.uniform(11.0, 15.0, size=(1, n_assets))
    volume = np.exp(vol_base + 0.5 * rng.standard_normal((t_len, n_assets)))
    return open_, high, low, close, volume


def test_institutional_v5_shape_bounds_1d_empty_and_nan_resilience():
    """Devuelve [T, N, 20] float32 finito en [-8, 8], tolera 1D, T=0, volumen nulo y NaNs."""
    assert len(ia5.INSTITUTIONAL_V5_NAMES) == 20
    rng = np.random.default_rng(101)
    o, h, lo, c, v = _synthetic_ohlcv_panel(rng, t_len=170, n_assets=8)
    v[:, 0] = 0.0
    c[30, 1] = np.nan
    o[45, 2] = h[45, 2] = lo[45, 2] = c[45, 2] = 50.0

    out = ia5.batched_institutional_alpha_v5_torch(
        torch.from_numpy(o),
        torch.from_numpy(h),
        torch.from_numpy(lo),
        torch.from_numpy(c),
        torch.from_numpy(v),
    )
    assert out.shape == (170, 8, 20)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert float(out.min()) >= -ia5.LIMIT
    assert float(out.max()) <= ia5.LIMIT
    # Sin calendario, la columna de calendario es exactamente 0
    cal_idx = ia5.INSTITUTIONAL_V5_NAMES.index("calendar_tom_dow_cycle")
    assert torch.equal(out[:, :, cal_idx], torch.zeros((170, 8), dtype=torch.float32))

    # Soporte 1D (un activo) con market_returns explícito
    mkt = rng.normal(0.0, 0.012, size=170)
    out_1d = ia5.batched_institutional_alpha_v5_torch(
        torch.from_numpy(o[:, 3]),
        torch.from_numpy(h[:, 3]),
        torch.from_numpy(lo[:, 3]),
        torch.from_numpy(c[:, 3]),
        torch.from_numpy(v[:, 3]),
        market_returns=torch.from_numpy(mkt),
    )
    assert out_1d.shape == (170, 20)
    assert torch.isfinite(out_1d).all()

    # T = 0 vacío
    empty = torch.empty((0, 4), dtype=torch.float32)
    out_empty = ia5.batched_institutional_alpha_v5_torch(empty, empty, empty, empty, empty)
    assert out_empty.shape == (0, 4, 20)
    assert out_empty.dtype == torch.float32

    # Seguridad bajo entradas bfloat16 (Blackwell AMP)
    o_bf, h_bf, lo_bf, c_bf, v_bf = (
        torch.from_numpy(x).to(torch.bfloat16) for x in (o, h, lo, c, v)
    )
    out_bf = ia5.batched_institutional_alpha_v5_torch(o_bf, h_bf, lo_bf, c_bf, v_bf)
    assert out_bf.shape == (170, 8, 20)
    assert out_bf.dtype == torch.float32
    assert torch.isfinite(out_bf).all()


def test_institutional_v5_strict_bit_for_bit_causality():
    """Mutar cualquier entrada en t > t0 nunca altera ni un solo bit en filas <= t0."""
    rng = np.random.default_rng(7)
    t_len, n_assets = 210, 10
    o, h, lo, c, v = _synthetic_ohlcv_panel(rng, t_len=t_len, n_assets=n_assets)
    mkt = 0.012 * rng.standard_normal(t_len)
    dow = np.arange(t_len) % 5
    dom = (np.arange(t_len) % 31) + 1
    t0 = 145

    f1 = ia5.batched_institutional_alpha_v5_torch(
        torch.from_numpy(o),
        torch.from_numpy(h),
        torch.from_numpy(lo),
        torch.from_numpy(c),
        torch.from_numpy(v),
        market_returns=torch.from_numpy(mkt),
        day_of_week=torch.from_numpy(dow),
        day_of_month=torch.from_numpy(dom),
    )

    o2, h2, lo2, c2, v2 = o.copy(), h.copy(), lo.copy(), c.copy(), v.copy()
    mkt2, dow2, dom2 = mkt.copy(), dow.copy(), dom.copy()
    c2[t0 + 1 :] *= 3.8
    h2[t0 + 1 :] *= 4.5
    lo2[t0 + 1 :] *= 0.25
    v2[t0 + 1 :] *= 15.0
    o2[t0 + 2] *= 1e4
    mkt2[t0 + 1 :] = -0.35
    dow2[t0 + 1 :] = 4
    dom2[t0 + 1 :] = 1

    f2 = ia5.batched_institutional_alpha_v5_torch(
        torch.from_numpy(o2),
        torch.from_numpy(h2),
        torch.from_numpy(lo2),
        torch.from_numpy(c2),
        torch.from_numpy(v2),
        market_returns=torch.from_numpy(mkt2),
        day_of_week=torch.from_numpy(dow2),
        day_of_month=torch.from_numpy(dom2),
    )
    assert torch.equal(f1[: t0 + 1], f2[: t0 + 1])
    assert not torch.equal(f1[t0 + 1 :], f2[t0 + 1 :])


def test_overnight_intraday_and_residual_momentum_economic_recovery():
    """Recupera tug-of-war overnight/intraday y momentum residual inmune a caída de mercado."""
    rng = np.random.default_rng(2025)
    t_len, n_assets = 180, 6
    mkt = -0.002 + 0.014 * rng.standard_normal(t_len)  # mercado bajista

    log_o = np.zeros((t_len, n_assets), dtype=np.float64)
    log_c = np.zeros((t_len, n_assets), dtype=np.float64)
    log_c[0] = math_log_base = np.log(100.0)
    log_o[0] = math_log_base

    for t in range(1, t_len):
        noise = 0.004 * rng.standard_normal(n_assets)
        # Activo 0: overnight +0.005, intraday -0.004 (fuerte tug-of-war positivo)
        # Activo 1: overnight -0.004, intraday +0.005 (tug-of-war negativo)
        # Activo 2: beta=1.8 + alfa puro positivo (+0.0035/día) en mercado bajista
        # Activo 3: beta=1.8 + alfa cero (solo beta)
        on = np.array([0.005, -0.004, 0.0015 + 0.9 * mkt[t], 0.9 * mkt[t], 0.0, 0.0]) + 0.5 * noise
        id_ = (
            np.array([-0.004, 0.005, 0.0020 + 0.9 * mkt[t], 0.9 * mkt[t], 0.0, 0.0]) + 0.5 * noise
        )
        log_o[t] = log_c[t - 1] + on
        log_c[t] = log_o[t] + id_

    o = np.exp(log_o)
    c = np.exp(log_c)
    h = np.maximum(o, c) * 1.005
    lo = np.minimum(o, c) * 0.995
    v = np.full((t_len, n_assets), 1e6)

    out = ia5.batched_institutional_alpha_v5_torch(
        torch.from_numpy(o),
        torch.from_numpy(h),
        torch.from_numpy(lo),
        torch.from_numpy(c),
        torch.from_numpy(v),
        market_returns=torch.from_numpy(mkt),
    )
    idx = {name: i for i, name in enumerate(ia5.INSTITUTIONAL_V5_NAMES)}

    # Tug-of-war overnight vs intraday (Lou-Polk-Skouras)
    assert float(out[-1, 0, idx["overnight_cum_21"]]) > 0.05
    assert float(out[-1, 0, idx["intraday_cum_21"]]) < -0.04
    assert float(out[-1, 0, idx["overnight_minus_intraday_63"]]) > 0.30
    assert float(out[-1, 1, idx["overnight_minus_intraday_63"]]) < -0.30

    # Residual Momentum (Blitz-Huij-Martens): activo 2 tiene fuerte Sharpe residual > activo 3
    assert float(out[-1, 2, idx["resid_mom_63"]]) > 2.0
    assert float(out[-1, 2, idx["resid_mom_126"]]) > 3.0
    assert float(out[-1, 2, idx["resid_mom_126"]]) > float(out[-1, 3, idx["resid_mom_126"]]) + 2.0


def test_lottery_tail_downside_beta_coskewness_and_pastor_stambaugh_recovery():
    """Verifica MAX/MIN, downside beta, asimetría de beta, co-skewness y reversión PS."""
    rng = np.random.default_rng(88)
    t_len, n_assets = 130, 6
    mkt = 0.015 * rng.standard_normal(t_len)
    rets = 0.005 * rng.standard_normal((t_len, n_assets))

    # Activo 0: lotería positiva en los últimos 21 días (3 saltos +12%)
    rets[-15, 0] = 0.12
    rets[-10, 0] = 0.11
    rets[-4, 0] = 0.13
    # Activo 1: cola izquierda severa en los últimos 21 días (3 desplomes -12%)
    rets[-16, 1] = -0.12
    rets[-9, 1] = -0.11
    rets[-3, 1] = -0.13
    # Activo 2: cóncavo -> downside beta = 2.0, upside beta = 0.4
    rets[:, 2] = np.where(mkt < 0.0, 2.0 * mkt, 0.4 * mkt) + 0.002 * rng.standard_normal(t_len)
    # Activo 3: convexo -> downside beta = 0.3, upside beta = 1.9
    rets[:, 3] = np.where(mkt < 0.0, 0.3 * mkt, 1.9 * mkt) + 0.002 * rng.standard_normal(t_len)

    # Activo 4: reversión Pastor-Stambaugh (r_t = -0.004 * sign(r_{t-1}) * log1p(dv_{t-1}))
    vol = np.full((t_len, n_assets), 1e5, dtype=np.float64)
    vol[-5:, 5] = 2e6  # Activo 5: shock de volumen 20x en últimos 5 días
    close = np.full((t_len, n_assets), 50.0, dtype=np.float64)
    for t in range(1, t_len):
        prev_r4 = rets[t - 1, 4]
        signed_ldv = np.sign(prev_r4) * np.log1p(close[t - 1, 4] * vol[t - 1, 4])
        rets[t, 4] = -0.004 * signed_ldv + 0.004 * rng.standard_normal()
        close[t] = close[t - 1] * np.exp(rets[t])

    open_ = close * 0.999
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997

    out = ia5.batched_institutional_alpha_v5_torch(
        torch.from_numpy(open_),
        torch.from_numpy(high),
        torch.from_numpy(low),
        torch.from_numpy(close),
        torch.from_numpy(vol),
        market_returns=torch.from_numpy(mkt),
    )
    idx = {name: i for i, name in enumerate(ia5.INSTITUTIONAL_V5_NAMES)}

    # Bali-Cakici-Whitelaw MAX / MIN / MAX + MIN
    assert float(out[-1, 0, idx["lottery_max_21"]]) == pytest.approx(0.12, abs=0.01)
    assert float(out[-1, 0, idx["lottery_tail_asym_21"]]) > 0.08
    assert float(out[-1, 1, idx["lottery_min_21"]]) == pytest.approx(-0.12, abs=0.01)
    assert float(out[-1, 1, idx["lottery_tail_asym_21"]]) < -0.08

    # Downside Beta & Beta Asymmetry & Co-Skewness (activos 2 y 3)
    assert float(out[-1, 2, idx["downside_beta_63"]]) == pytest.approx(2.0, abs=0.25)
    assert float(out[-1, 2, idx["beta_asym_63"]]) < -1.0
    assert float(out[-1, 3, idx["downside_beta_63"]]) == pytest.approx(0.3, abs=0.25)
    assert float(out[-1, 3, idx["beta_asym_63"]]) > 1.0
    # Verificación exacta de co-skewness Harvey-Siddique frente a fórmula directa en ventana 63
    r2_w = rets[-63:, 2]
    m_w = mkt[-63:]
    coskew_ref = float(
        np.mean((r2_w - r2_w.mean()) * (m_w - m_w.mean()) ** 2)
        / (np.std(r2_w) * np.var(m_w))
    )
    assert float(out[-1, 2, idx["coskewness_63"]]) == pytest.approx(coskew_ref, abs=1e-4)
    assert float(out[-1, 2, idx["coskewness_63"]]) < 0.0  # cóncavo -> co-skewness negativo
    assert float(out[-1, 3, idx["coskewness_63"]]) > 0.0  # convexo -> co-skewness positivo

    # Pastor-Stambaugh gamma < 0 en activo 4 y abnormal turnover shock > 1 en activo 5
    assert float(out[-1, 4, idx["ps_liquidity_gamma_21"]]) < -0.15
    assert float(out[-1, 5, idx["abnormal_dvol_shock_5_63"]]) > 1.2
    # Idiosyncratic vol mayor en activo 0 (con saltos de lotería) que en activo 2 (casi puro factor)
    assert float(out[-1, 0, idx["idio_vol_63"]]) > float(out[-1, 2, idx["idio_vol_63"]])

    # Persistencia de impacto en precio: régimen persistente vs alternante
    t_p = 60
    imp_regime_vol = np.full((t_p, 2), 1e5, dtype=np.float64)
    # Activo 0: bloque de iliquidez persistente (primeros 30 días alto vol, últimos 30 bajo vol)
    imp_regime_vol[30:, 0] = 2e3
    # Activo 1: iliquidez alternando día sí / día no (autocorrelación negativa)
    imp_regime_vol[::2, 1] = 2e3
    imp_regime_vol[1::2, 1] = 5e6
    r_p = np.full((t_p, 2), 0.015, dtype=np.float64)
    c_p = 50.0 * np.exp(np.cumsum(r_p, axis=0))
    out_p = ia5.batched_institutional_alpha_v5_torch(
        torch.from_numpy(c_p * 0.999),
        torch.from_numpy(c_p * 1.002),
        torch.from_numpy(c_p * 0.998),
        torch.from_numpy(c_p),
        torch.from_numpy(imp_regime_vol),
    )
    assert float(out_p[40, 0, idx["price_impact_persist_21"]]) > 0.5
    assert float(out_p[40, 1, idx["price_impact_persist_21"]]) < -0.5


def test_cross_sectional_regime_and_calendar_microstructure():
    """Verifica dispersión/asimetría transversal, ratio idiosincrático y ciclo calendario."""
    rng = np.random.default_rng(19)
    t_len, n_assets = 90, 20
    mkt = 0.02 * rng.standard_normal(t_len)
    # Mitad de sesiones dominadas por factor común, final con fuerte dispersión idiosincrática
    rets = mkt[:, None] + 0.002 * rng.standard_normal((t_len, n_assets))
    rets[-1] = mkt[-1] + 0.03 * rng.standard_normal(n_assets)
    rets[-1, 0] += 0.15  # sesgo transversal positivo marcado

    close = 50.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = close * 0.999
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = np.full((t_len, n_assets), 5e5)

    dow = np.zeros(t_len, dtype=np.int64)
    dom = np.full(t_len, 16, dtype=np.int64)  # mitad de mes + lunes -> mínimo
    dow[-1] = 4  # viernes
    dom[-1] = 1  # inicio de mes (Turn-of-Month) -> máximo

    out = ia5.batched_institutional_alpha_v5_torch(
        torch.from_numpy(open_),
        torch.from_numpy(high),
        torch.from_numpy(low),
        torch.from_numpy(close),
        torch.from_numpy(vol),
        market_returns=torch.from_numpy(mkt),
        day_of_week=torch.from_numpy(dow),
        day_of_month=torch.from_numpy(dom),
    )
    idx = {name: i for i, name in enumerate(ia5.INSTITUTIONAL_V5_NAMES)}

    assert float(out[-1, 0, idx["cs_return_dispersion"]]) > float(
        out[-5, 0, idx["cs_return_dispersion"]]
    )
    assert float(out[-1, 0, idx["cs_return_skewness"]]) > 1.0
    assert float(out[-1, 0, idx["idio_dispersion_ratio"]]) > 0.7
    assert float(out[-1, 0, idx["calendar_tom_dow_cycle"]]) == pytest.approx(1.0, abs=1e-4)
    assert float(out[-2, 0, idx["calendar_tom_dow_cycle"]]) < -0.9


def test_causal_feature_quality_audit_detects_signal_leakage_redundancy_collapse_and_nan():
    """Audita Rank IC (empate con scipy), IC_IR, fuga, redundancia, colapso y NaNs."""
    rng = np.random.default_rng(404)
    t_len, n_assets = 120, 100
    future_ret = 0.02 * rng.standard_normal((t_len, n_assets))

    # 6 rasgos sintéticos con patologías y señales controladas:
    # f0: señal genuina moderada (IC esperado ~0.06, estable y lejos de fuga >0.10)
    f0 = 0.06 * future_ret + 0.02 * rng.standard_normal((t_len, n_assets))
    # f1: copia casi idéntica de f0 (redundancia > 0.95)
    f1 = f0 + 0.002 * rng.standard_normal((t_len, n_assets))
    # f2: fuga severa de futuro (IC > 0.80 -> leakage suspect!)
    f2 = future_ret + 0.002 * rng.standard_normal((t_len, n_assets))
    # f3: ruido puro con empates (ties)
    f3 = np.round(rng.standard_normal((t_len, n_assets)), 1)
    # f4: varianza colapsada (constante)
    f4 = np.full((t_len, n_assets), 3.14)
    # f5: contaminación con NaN
    f5 = rng.standard_normal((t_len, n_assets))
    f5[5, 3] = np.nan

    feats = np.stack([f0, f1, f2, f3, f4, f5], axis=-1).astype(np.float32)
    names = ("alpha_clean", "alpha_dup", "leaky_fut", "noisy_ties", "const_dead", "has_nan")

    report = ia5.causal_feature_quality_audit(
        torch.from_numpy(feats),
        torch.from_numpy(future_ret.astype(np.float32)),
        feature_names=names,
        min_ic=0.02,
    )

    assert report["n_sessions"] == t_len
    assert report["n_assets"] == n_assets
    assert report["n_features"] == 6
    assert report["all_finite"] is False
    assert report["non_finite_counts"]["has_nan"] == 1
    assert report["non_finite_counts"]["alpha_clean"] == 0
    assert "const_dead" in report["variance_collapsed"]
    assert "leaky_fut" in report["leakage_suspects"]
    assert "alpha_clean" not in report["leakage_suspects"]
    assert "alpha_clean" in report["passing_min_ic"]
    assert report["ic_ir"]["alpha_clean"] > 0.25
    assert report["max_pairwise_corr"] > 0.95
    assert report["redundancy_passed"] is False
    assert any(
        {a, b} == {"alpha_clean", "alpha_dup"} for a, b, _ in report["redundant_pairs"]
    )
    assert report["overall_passed"] is False

    # Paridad exacta del Rank IC con scipy.stats.spearmanr (incluso con empates en f3!)
    ref_ics_f3 = [stats.spearmanr(feats[t, :, 3], future_ret[t]).statistic for t in range(t_len)]
    assert report["mean_rank_ic"]["noisy_ties"] == pytest.approx(
        float(np.mean(ref_ics_f3)), abs=1e-5
    )

    # Subconjunto limpio (solo f0 y f3) debe pasar overall_passed y redundancy_passed
    clean_rep = ia5.causal_feature_quality_audit(
        torch.from_numpy(feats[:, :, [0, 3]]),
        torch.from_numpy(future_ret.astype(np.float32)),
        feature_names=("alpha_clean", "noisy_ties"),
        min_ic=0.02,
    )
    assert clean_rep["all_finite"] is True
    assert clean_rep["variance_collapsed"] == []
    assert clean_rep["leakage_suspects"] == []
    assert clean_rep["redundancy_passed"] is True
    assert clean_rep["overall_passed"] is True


def test_institutional_v5_gpu_hot_paths_have_zero_host_sync_traps():
    """Audita vía AST que batched_institutional_alpha_v5_torch no tenga `.item`/`.cpu`/`nonzero`."""
    source = inspect.getsource(ia5)
    tree = ast.parse(source)
    hot_fns = {
        "batched_institutional_alpha_v5_torch",
        "_rolling_sum_2d",
        "_rolling_bivariate_stats",
        "_rolling_topk_mean_2d",
        "_cross_sectional_dispersion_skew",
    }
    forbidden_attrs = {"item", "cpu", "nonzero"}
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in hot_fns:
            found.add(node.name)
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and child.attr in forbidden_attrs:
                    raise AssertionError(
                        f"Trampa de sincronización host `{child.attr}` en {node.name}"
                    )
    assert "batched_institutional_alpha_v5_torch" in found


def test_institutional_v5_constant_series_6000_no_cancellation_and_masked_warmup_all_nan():
    """T=6000 constante no dispara cancelación FP64, warmup/all-NaN idio_vol=0 y t=0 enmascarado."""
    t_len, n_assets = 6000, 3
    # Activo 0: retorno diario constante +0.02 y volumen constante 1e5 -> varianza real = 0
    # Activo 1: todo NaN
    # Activo 2: serie aleatoria limpia
    rng = np.random.default_rng(6000)
    rets = np.zeros((t_len, n_assets), dtype=np.float64)
    rets[:, 0] = 0.02
    rets[:, 2] = 0.015 * rng.standard_normal(t_len)
    close = 50.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = close.copy()
    high = close * 1.001
    low = close * 0.999
    vol = np.full((t_len, n_assets), 1e5, dtype=np.float64)
    # Activo 0: dvol = close * vol = 1e5 constante + retorno diario constante +0.02
    vol[:, 0] = 1e5 / close[:, 0]
    close[:, 1] = np.nan
    open_[:, 1] = np.nan
    high[:, 1] = np.nan
    low[:, 1] = np.nan
    vol[:, 1] = np.nan

    mkt = 0.01 * rng.standard_normal(t_len)
    mkt[0] = 5.0  # valor extremo en t=0 no debe emparejarse con ret_cc[0]=0

    out = ia5.batched_institutional_alpha_v5_torch(
        torch.from_numpy(open_),
        torch.from_numpy(high),
        torch.from_numpy(low),
        torch.from_numpy(close),
        torch.from_numpy(vol),
        market_returns=torch.from_numpy(mkt),
    )
    idx = {name: i for i, name in enumerate(ia5.INSTITUTIONAL_V5_NAMES)}

    # Activo 0 constante en T=6000: cero cancelación espuria en PS gamma y resid_mom
    assert float(out[-1, 0, idx["ps_liquidity_gamma_21"]]) == 0.0
    assert float(out[-1, 0, idx["resid_mom_63"]]) == 0.0
    assert float(out[-1, 0, idx["resid_mom_126"]]) == 0.0

    # Warmup (primeros 14 días) y activo 1 todo-NaN tienen idio_vol_63 == 0.0 (nunca -4.14!)
    assert float(out[:14, 2, idx["idio_vol_63"]].abs().max()) == 0.0
    assert float(out[:, 1, idx["idio_vol_63"]].abs().max()) == 0.0

    # Mutar mkt[0] no debe cambiar ningún rasgo (t=0 enmascarado en valid_rm)
    mkt_alt = mkt.copy()
    mkt_alt[0] = -5.0
    out_alt = ia5.batched_institutional_alpha_v5_torch(
        torch.from_numpy(open_),
        torch.from_numpy(high),
        torch.from_numpy(low),
        torch.from_numpy(close),
        torch.from_numpy(vol),
        market_returns=torch.from_numpy(mkt_alt),
    )
    assert torch.equal(out, out_alt)


def test_causal_feature_quality_audit_exact_pairwise_corr_under_partial_nans():
    """Dos rasgos idénticos donde uno es NaN en la mitad de filas deben reportar corr == 1.0."""
    rng = np.random.default_rng(777)
    t_len, n_assets = 40, 30
    future_ret = 0.02 * rng.standard_normal((t_len, n_assets))
    base = rng.standard_normal((t_len, n_assets))
    # Añadir sesgo fuerte en la primera mitad para amplificar el error marginal vs pairwise
    base[:20] += 5.0
    half_nan = base.copy()
    half_nan[:20] = np.nan
    feats = np.stack([base, half_nan], axis=-1).astype(np.float32)

    rep = ia5.causal_feature_quality_audit(
        torch.from_numpy(feats),
        torch.from_numpy(future_ret.astype(np.float32)),
        feature_names=("full", "half_nan"),
        redundancy_threshold=0.70,
    )
    assert rep["corr_matrix"][0, 1] == pytest.approx(1.0, abs=1e-5)
    assert rep["max_pairwise_corr"] == pytest.approx(1.0, abs=1e-5)
    assert rep["redundancy_passed"] is False
    assert len(rep["redundant_pairs"]) == 1


def test_institutional_v5_zero_scalar_h2d_syncs_and_fp32_fp64_parity():
    """Cero torch.tensor() H2D ocultos y paridad FP32 (defecto) vs FP64 en CPU y CUDA."""
    source = inspect.getsource(ia5)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "batched_institutional_alpha_v5_torch",
            "_rolling_bivariate_stats",
            "_rolling_bivariate_63_fused",
            "_rolling_topk_mean_2d",
            "_cross_sectional_dispersion_skew",
        }:
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "tensor"
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "torch"
                ):
                    raise AssertionError(
                        f"Copia H2D/sync oculta `torch.tensor(...)` detectada en {node.name}"
                    )

    rng = np.random.default_rng(505)
    o, h, lo, c, v = _synthetic_ohlcv_panel(rng, t_len=190, n_assets=16)
    mkt = 0.012 * rng.standard_normal(190)
    dow = np.arange(190) % 5
    dom = (np.arange(190) % 31) + 1

    out_f32 = ia5.batched_institutional_alpha_v5_torch(
        o, h, lo, c, v, market_returns=mkt, day_of_week=dow, day_of_month=dom
    )
    out_f64 = ia5.batched_institutional_alpha_v5_torch(
        o,
        h,
        lo,
        c,
        v,
        market_returns=mkt,
        day_of_week=dow,
        day_of_month=dom,
        compute_dtype=torch.float64,
    )
    assert out_f32.dtype == torch.float32
    assert torch.isfinite(out_f32).all()
    torch.testing.assert_close(out_f32, out_f64, atol=2e-4, rtol=2e-4)

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        out_cuda = ia5.batched_institutional_alpha_v5_torch(
            torch.from_numpy(o).to(dev),
            torch.from_numpy(h).to(dev),
            torch.from_numpy(lo).to(dev),
            torch.from_numpy(c).to(dev),
            torch.from_numpy(v).to(dev),
            market_returns=torch.from_numpy(mkt).to(dev),
            day_of_week=torch.from_numpy(dow).to(dev),
            day_of_month=torch.from_numpy(dom).to(dev),
        )
        torch.testing.assert_close(out_cuda.cpu(), out_f64, atol=2e-4, rtol=2e-4)


def test_institutional_v5_financial_correctness_day0_halts_lagged_beta_and_audit():
    """Corrige Day-0 NaN, halts volume<=0 en impacto, beta_63[t-1] lag y cancelación en audit."""
    idx = {name: i for i, name in enumerate(ia5.INSTITUTIONAL_V5_NAMES)}

    # 1) Day-0 ret_cc[0]/ret_on[0] como NaN (no 0.0 espurio en lottery_min_21 durante t < 21)
    t_d0, n_d0 = 30, 3
    rets_pos = np.full((t_d0, n_d0), 0.03, dtype=np.float64)
    c_d0 = 50.0 * np.exp(np.cumsum(rets_pos, axis=0))
    o_d0 = c_d0 * 0.999
    h_d0 = c_d0 * 1.002
    l_d0 = c_d0 * 0.998
    v_d0 = np.full((t_d0, n_d0), 1e5, dtype=np.float64)
    out_d0 = ia5.batched_institutional_alpha_v5_torch(o_d0, h_d0, l_d0, c_d0, v_d0)
    # En t=5 (< 21), 5 retornos válidos todos +0.03 -> lottery_min_21 == +0.03 (jamás 0.02!)
    assert float(out_d0[5, 0, idx["lottery_min_21"]]) == pytest.approx(0.03, abs=1e-4)

    # 2) Días suspendidos (volume <= 0) enmascarados como NaN/neutrales en impacto y PS gamma
    rng = np.random.default_rng(919)
    t_h, n_h = 80, 4
    o_h, h_h, l_h, c_h, v_h = _synthetic_ohlcv_panel(rng, t_len=t_h, n_assets=n_h)
    # Activo 0: totalmente suspendido (volume == 0)
    v_h[:, 0] = 0.0
    # Activo 1 vs Activo 2: idénticos salvo 2 días de halt con salto de precio en Activo 1
    o_h[:, 1] = o_h[:, 2]
    h_h[:, 1] = h_h[:, 2]
    l_h[:, 1] = l_h[:, 2]
    c_h[:, 1] = c_h[:, 2]
    v_h[:, 1] = v_h[:, 2]
    v_h[50:52, 1] = 0.0
    c_h[50:52, 1] *= 1.08
    out_h = ia5.batched_institutional_alpha_v5_torch(o_h, h_h, l_h, c_h, v_h)
    assert float(out_h[:, 0, idx["price_impact_persist_21"]].abs().max()) == 0.0
    assert float(out_h[:, 0, idx["ps_liquidity_gamma_21"]].abs().max()) == 0.0
    # Con dvol=1.0 bug, dos días consecutivos con imp=8.0 disparaban persistencia a >0.95!
    assert abs(float(out_h[55, 1, idx["price_impact_persist_21"]])) < 0.85

    # 3) Lag de beta_63[t-1] en resid_daily: sin fuga contemporánea de beta_63[t]
    t_b, n_b = 85, 20
    mkt_b = 0.005 * rng.standard_normal(t_b)
    # En t=0..79 todos los activos tienen beta real = 0 respecto a mkt_b (solo ruido pequeño)
    rets_b = 0.002 * rng.standard_normal((t_b, n_b))
    # En t=80, shock enorme de mercado mkt_b[80]=0.50 con betas nuevas diversas b_i en [-1, 1]
    mkt_b[80] = 0.50
    b_new = np.linspace(-0.4, 0.4, n_b)
    rets_b[80] = b_new * mkt_b[80]
    c_b = 50.0 * np.exp(np.cumsum(rets_b, axis=0))
    out_b = ia5.batched_institutional_alpha_v5_torch(
        c_b * 0.999,
        c_b * 1.001,
        c_b * 0.998,
        c_b,
        np.full((t_b, n_b), 1e6),
        market_returns=mkt_b,
    )
    # Con beta_63[79] (~0) rezagada, resid_daily[80] ≈ rets_b[80] -> idio_dispersion_ratio ≈ 1.0!
    # Con beta_63[80] contemporánea (fuga), beta_63[80] absorbe b_new -> ratio colapsaría a ~0.0!
    assert float(out_b[80, 0, idx["idio_dispersion_ratio"]]) > 0.85

    # 4) Cero cancelación en causal_feature_quality_audit (offset 1e8 + NaNs parciales)
    t_a, n_a = 40, 25
    fut_a = 0.02 * rng.standard_normal((t_a, n_a))
    base_a = rng.standard_normal((t_a, n_a))
    f_huge_a = 1e8 + base_a
    f_huge_b = 1e8 + base_a + 0.05 * rng.standard_normal((t_a, n_a))
    # Caso adversarial: desplazamiento 1e8 en la primera mitad donde f_half es NaN
    f_shift = base_a.copy()
    f_shift[:20] += 1e8
    f_half = base_a.copy()
    f_half[:20] = np.nan
    feats_a = np.stack([f_huge_a, f_huge_b, f_shift, f_half], axis=-1)
    rep_a = ia5.causal_feature_quality_audit(
        torch.from_numpy(feats_a),
        torch.from_numpy(fut_a),
        feature_names=("huge_a", "huge_b", "shift_full", "half_nan"),
    )
    ref_01 = float(np.corrcoef(base_a.ravel(), (f_huge_b - 1e8).ravel())[0, 1])
    assert rep_a["corr_matrix"][0, 1] == pytest.approx(ref_01, abs=1e-4)
    assert rep_a["corr_matrix"][2, 3] == pytest.approx(1.0, abs=1e-4)


