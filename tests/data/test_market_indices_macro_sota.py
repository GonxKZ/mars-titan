"""Pruebas TDD para `market_indices_macro_sota`: índices institucionales, factores y macro PIT."""

from __future__ import annotations

import ast
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data import dataset_enrichment_sota as des
from mars_titan.data import market_indices_macro_sota as mims
from mars_titan.models.mmarrs.blackwell_kernels import audit_zero_host_syncs


def _synthetic_market_panel(
    rng: np.random.Generator,
    t_len: int = 180,
    n_assets: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Panel sintético OHLCV con factor de mercado común + retornos idiosincráticos."""
    mkt_ret = 0.0004 + 0.010 * rng.standard_normal((t_len, 1))
    betas = rng.uniform(0.5, 1.6, size=(1, n_assets))
    idio_vols = rng.uniform(0.008, 0.020, size=(1, n_assets))
    rets = betas * mkt_ret + idio_vols * rng.standard_normal((t_len, n_assets))
    base_px = rng.uniform(15.0, 200.0, size=(1, n_assets))
    close = base_px * np.exp(np.cumsum(rets, axis=0))
    open_ = close * np.exp(0.002 * rng.standard_normal((t_len, n_assets)))
    high = np.maximum(open_, close) * np.exp(np.abs(0.005 * rng.standard_normal((t_len, n_assets))))
    low = np.minimum(open_, close) * np.exp(-np.abs(0.005 * rng.standard_normal((t_len, n_assets))))
    vol_base = np.linspace(11.0, 16.0, n_assets)[None, :]
    volume = np.exp(vol_base + 0.35 * rng.standard_normal((t_len, n_assets)))
    return (
        open_.astype(np.float32),
        high.astype(np.float32),
        low.astype(np.float32),
        close.astype(np.float32),
        volume.astype(np.float32),
    )


def test_contracts_names_shapes_and_zero_duplicates_with_enriched_sota():
    """Contratos de nombres, dimensiones exactas y cero duplicados con `ENRICHED_SOTA_NAMES`."""
    assert mims.MARKET_REGIME_INDEX_DIM == len(mims.MARKET_REGIME_INDEX_NAMES) == 18
    assert mims.ASSET_INDEX_MACRO_DIM == len(mims.ASSET_INDEX_MACRO_NAMES) == 11
    assert mims.FACTOR_PORTFOLIO_DIM == len(mims.FACTOR_PORTFOLIO_NAMES) == 5
    assert mims.KEEP_ASSET_INDEX_MACRO_DIM == len(mims.KEEP_ASSET_INDEX_MACRO_NAMES) == 7
    assert len(mims.KEEP_ASSET_INDEX_MACRO_INDICES) == 7
    # Excluye explícitamente los redundantes (`stock_rel_to_vw_{5,21}`) y neutrales aislados (#224)
    for excluded in (
        "stock_rel_to_vw_5",
        "stock_rel_to_vw_21",
        "stock_turbulence_beta_63",
        "stock_macro_innov_beta_63",
    ):
        assert excluded not in mims.KEEP_ASSET_INDEX_MACRO_NAMES
    assert set(mims.KEEP_ASSET_INDEX_MACRO_NAMES) < set(mims.ASSET_INDEX_MACRO_NAMES)
    assert set(mims.KEEP_ASSET_INDEX_MACRO_NAMES).isdisjoint(set(des.ENRICHED_SOTA_NAMES))
    assert len(mims.MACRO_V31_140_NAMES) == 140
    assert len(set(mims.MACRO_V31_140_NAMES)) == 140

    all_new = (
        set(mims.MARKET_REGIME_INDEX_NAMES)
        | set(mims.ASSET_INDEX_MACRO_NAMES)
        | set(mims.FACTOR_PORTFOLIO_NAMES)
    )
    assert len(all_new) == 18 + 11 + 5
    assert all_new.isdisjoint(set(des.ENRICHED_SOTA_NAMES))

    rng = np.random.default_rng(101)
    o, h, lo, c, v = _synthetic_market_panel(rng, t_len=90, n_assets=20)
    out = mims.compute_market_indices_and_macro_sota_torch(o, h, lo, c, v)
    keep_sel = mims.select_keep_asset_index_macro_features(out["asset_index_macro_features"])
    assert keep_sel.shape == (90, 20, 7)
    for k_pos, orig_idx in enumerate(mims.KEEP_ASSET_INDEX_MACRO_INDICES):
        assert torch.equal(keep_sel[..., k_pos], out["asset_index_macro_features"][..., orig_idx])
    assert set(out.keys()) >= {
        "market_regime_indices",
        "asset_index_macro_features",
        "factor_returns",
        "market_ew_returns",
    }
    assert out["market_regime_indices"].shape == (90, 18)
    assert out["asset_index_macro_features"].shape == (90, 20, 11)
    assert out["factor_returns"].shape == (90, 5)
    assert out["market_ew_returns"].shape == (90,)
    for key, val in out.items():
        assert val.dtype == torch.float32, f"{key} no es float32"
        assert torch.isfinite(val).all(), f"{key} contiene NaN/Inf"
        assert float(val.min()) >= -mims.LIMIT
        assert float(val.max()) <= mims.LIMIT

    # Soporte 1D [T] y vacío T=0 / N=0
    out_1d = mims.compute_market_indices_and_macro_sota_torch(
        o[:, 0], h[:, 0], lo[:, 0], c[:, 0], v[:, 0]
    )
    assert out_1d["market_regime_indices"].shape == (90, 18)
    assert out_1d["asset_index_macro_features"].shape == (90, 11)
    assert out_1d["factor_returns"].shape == (90, 5)
    assert torch.isfinite(out_1d["asset_index_macro_features"]).all()

    empty = torch.empty((0, 6), dtype=torch.float32)
    out_empty = mims.compute_market_indices_and_macro_sota_torch(empty, empty, empty, empty, empty)
    assert out_empty["market_regime_indices"].shape == (0, 18)
    assert out_empty["asset_index_macro_features"].shape == (0, 6, 11)
    assert out_empty["factor_returns"].shape == (0, 5)


def test_market_breadth_dispersion_vol_turbulence_cn_stress_and_fx_invariance():
    """Fidelidad económica de amplitud, EW-VW, vol, Kritzman-Li, CN stress e invarianza FX."""
    m_idx = {name: i for i, name in enumerate(mims.MARKET_REGIME_INDEX_NAMES)}
    rng = np.random.default_rng(2026)
    t_len, n_assets = 260, 40

    # Fase 0 (0..139): mercado alcista amplio; Fase 1 (140..229): caída profunda bajo mínimos 252d;
    # Fase 2 (230..259): rally mega-cap con turbulencia + pánico vol corto + halts/límites CN
    rets = 0.005 * rng.standard_normal((t_len, n_assets))
    rets[20:140, :] += 0.006  # rally amplio -> breadth alto, pct_above_ma50/200 alto, NH > NL
    rets[140:230, :] -= 0.011  # bear profundo -> rompe mínimos de 252d (NL > NH)
    # Días 230..254: líderes mega-cap (activos 30..39, alto volumen) suben fuerte, small-caps caen
    rets[230:255, :20] -= 0.006
    rets[230:255, 30:] += 0.014
    # Últimos 5 días (255..259): shock violento de volatilidad y turbulencia en factores
    rets[255:260, :] = np.where(
        (np.arange(5)[:, None] % 2) == 0,
        -0.065,
        0.055,
    ) + 0.01 * rng.standard_normal((5, n_assets))
    # Shock extremo de dispersión & límite -10% + halts CN en el último bloque
    rets[258, :16] = -0.105

    close = 50.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = close * np.exp(0.001 * rng.standard_normal((t_len, n_assets)))
    high = np.maximum(open_, close) * np.exp(0.004)
    low = np.minimum(open_, close) * np.exp(-0.004)

    # Liquidez escalonada: activos 30..39 tienen 100x más volumen dólar que 0..19
    vol = np.full((t_len, n_assets), 1e5, dtype=np.float64)
    vol[:, 30:] = 2e7
    # Contracción de liquidez + suspensiones (volume=0) en días 255..259
    vol[255:260, :] *= 0.08
    vol[255:260, 16:24] = 0.0  # 8/40 = 20% halts!

    out = mims.compute_market_indices_and_macro_sota_torch(open_, high, low, close, vol)
    mkt = out["market_regime_indices"]

    # 1) Amplitud alcista (t=135) vs bajista (t=225)
    assert float(mkt[135, m_idx["mkt_ad_ema21"]]) > 0.35
    assert float(mkt[225, m_idx["mkt_ad_ema21"]]) < -0.35
    assert float(mkt[135, m_idx["mkt_pct_above_ma50"]]) > 0.85
    assert float(mkt[225, m_idx["mkt_pct_above_ma50"]]) < 0.15
    assert float(mkt[135, m_idx["mkt_pct_above_ma200"]]) > 0.80
    assert float(mkt[135, m_idx["mkt_nh_nl_spread_252"]]) > 0.20
    assert float(mkt[225, m_idx["mkt_nh_nl_spread_252"]]) < -0.20
    # Oscilador McClellan positivo en impulso alcista y negativo al girar a bear
    assert float(mkt[155, m_idx["mkt_mcclellan_osc"]]) < float(mkt[45, m_idx["mkt_mcclellan_osc"]])

    # 2) Liderazgo mega-cap (t=254): VW supera a EW => ew_vs_vw_spread < 0
    assert float(mkt[254, m_idx["mkt_ew_vs_vw_spread_5"]]) < -0.005
    assert float(mkt[254, m_idx["mkt_ew_vs_vw_spread_21"]]) < -0.02

    # 3) Shock de volatilidad, backwardation, turbulencia Kritzman-Li, iliquidez y stress CN
    assert (
        float(mkt[259, m_idx["mkt_synth_vix_21"]])
        > float(mkt[135, m_idx["mkt_synth_vix_21"]]) + 0.15
    )
    assert float(mkt[259, m_idx["mkt_vol_term_slope_5_63"]]) > 0.30
    # Turbulencia Kritzman-Li se dispara al estallar el choque (t=255 y pico t=258 >> calma t=135)
    assert (
        float(mkt[255, m_idx["mkt_turbulence_63"]])
        > float(mkt[135, m_idx["mkt_turbulence_63"]]) + 1.50
    )
    assert (
        float(mkt[255:260, m_idx["mkt_turbulence_63"]].max())
        > float(mkt[135, m_idx["mkt_turbulence_63"]]) + 3.00
    )
    assert (
        float(mkt[259, m_idx["mkt_liquidity_stress_21"]])
        > float(mkt[135, m_idx["mkt_liquidity_stress_21"]]) + 0.50
    )
    assert float(mkt[259, m_idx["mkt_halt_limit_stress_5"]]) > 0.20

    # 4) Invarianza a divisa FX (multiplicar precios por 7.25 USD->CNY preserva índices y señales)
    fx = 7.25
    out_fx = mims.compute_market_indices_and_macro_sota_torch(
        open_ * fx, high * fx, low * fx, close * fx, vol
    )
    torch.testing.assert_close(
        out_fx["factor_returns"], out["factor_returns"], atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        out_fx["market_regime_indices"], out["market_regime_indices"], atol=2e-2, rtol=1e-3
    )
    torch.testing.assert_close(
        out_fx["asset_index_macro_features"],
        out["asset_index_macro_features"],
        atol=2e-2,
        rtol=1e-3,
    )


def test_causal_factor_portfolios_and_asset_alpha_economic_recovery():
    """Portafolios factoriales rezagados (`t-1` x `t`) y recuperación de alfa `[T, N]`."""
    a_idx = {name: i for i, name in enumerate(mims.ASSET_INDEX_MACRO_NAMES)}
    f_idx = {name: i for i, name in enumerate(mims.FACTOR_PORTFOLIO_NAMES)}
    rng = np.random.default_rng(777)
    t_len, n_assets = 180, 30

    # Mercado con tendencia clara + volatilidad controlada
    r_mkt = 0.001 + 0.012 * rng.standard_normal((t_len, 1))
    # Inyectar días bajistas marcados para calibrar downside beta
    r_mkt[100:170:4, 0] = -0.025

    betas = np.ones((1, n_assets), dtype=np.float64)
    betas[0, 0] = 2.1  # activo 0: beta alta (2.1)
    betas[0, 1] = 0.25  # activo 1: beta defensiva baja (0.25)
    betas[0, 2] = 0.8  # activo 2: solo alto downside beta (cae fuerte cuando mercado < 0)

    idio = 0.004 * rng.standard_normal((t_len, n_assets))
    rets = betas * r_mkt + idio
    # Activo 2 sufre doble en días bajistas del mercado
    down_days = r_mkt[:, 0] < 0.0
    n_dn = int(down_days.sum())
    rets[down_days, 2] = 2.4 * r_mkt[down_days, 0] + 0.002 * rng.standard_normal(n_dn)
    # Activo 3 tiene fuerte retorno relativo positivo en últimos 21 días; activo 4 fuerte negativo
    rets[-21:, 3] += 0.012
    rets[-21:, 4] -= 0.012

    close = 40.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = close * np.exp(0.001 * rng.standard_normal((t_len, n_assets)))
    high = np.maximum(open_, close) * np.exp(0.004)
    low = np.minimum(open_, close) * np.exp(-0.004)
    vol = np.exp(np.linspace(11.0, 16.0, n_assets)[None, :]) * np.ones((t_len, 1))

    out = mims.compute_market_indices_and_macro_sota_torch(open_, high, low, close, vol)
    af = out["asset_index_macro_features"]
    fac = out["factor_returns"]

    # 1) Portafolios factoriales finitos, no colapsados y con t=0 == 0.0 exacto (sin retorno Day-0)
    assert float(fac[0].abs().max()) == 0.0
    assert float(fac[20:, f_idx["fac_mkt_vw"]].std()) > 0.005
    assert float(fac[20:, f_idx["fac_liq_smb"]].std()) > 0.001
    assert float(fac[70:, f_idx["fac_mom_umd"]].std()) > 0.001
    assert float(fac[20:, f_idx["fac_rev_st"]].std()) > 0.001
    assert float(fac[20:, f_idx["fac_bab_lowvol"]].std()) > 0.001

    # 2) Beta BAB (activo 0 > activo 1) y Downside Beta (activo 2 >> activo 1)
    assert float(af[-1, 0, a_idx["stock_bab_beta_63"]]) > (
        float(af[-1, 1, a_idx["stock_bab_beta_63"]]) + 0.8
    )
    assert float(af[-1, 2, a_idx["stock_downside_beta_63"]]) > (
        float(af[-1, 1, a_idx["stock_downside_beta_63"]]) + 0.8
    )

    # 3) Retorno relativo al índice ponderado por liquidez
    assert float(af[-1, 3, a_idx["stock_rel_to_vw_5"]]) > float(
        af[-1, 4, a_idx["stock_rel_to_vw_5"]]
    )
    assert float(af[-1, 3, a_idx["stock_rel_to_vw_21"]]) > float(
        af[-1, 4, a_idx["stock_rel_to_vw_21"]]
    ) + 0.15

    # 4) Factor momentum score & turbulence beta activos (no colapsados)
    assert float(af[80:, :, a_idx["stock_factor_mom_score_21"]].std()) > 1e-3
    assert float(af[90:, :, a_idx["stock_turbulence_beta_63"]].std()) > 1e-3
    assert float(af[80:, :, a_idx["stock_breadth_x_mom_21"]].std()) > 1e-3
    assert float(af[80:, :, a_idx["stock_ew_vw_tilt_21"]].std()) > 1e-4
    assert float(af[80:, :, a_idx["stock_vol_term_x_bab"]].std()) > 1e-3


def test_pit_macro_420_and_raw_composites_staleness_turnon_and_null_safety():
    """Macro PIT 420-triplet y raw: decaimiento por obsolescencia, alta limpia y nulidad segura."""
    m_idx = {name: i for i, name in enumerate(mims.MARKET_REGIME_INDEX_NAMES)}
    a_idx = {name: i for i, name in enumerate(mims.ASSET_INDEX_MACRO_NAMES)}
    rng = np.random.default_rng(420)
    t_len, n_assets = 160, 24
    o, h, lo, c, v = _synthetic_market_panel(rng, t_len=t_len, n_assets=n_assets)

    # 1) macro=None, 100%-NaN, mask=False y 420 con obs=0 -> exactamente 0.0 en los 6 rasgos macro
    macro_reg_cols = (
        "macro_innov_pc1",
        "macro_innov_pc2",
        "macro_dispersion_21",
        "macro_stress_composite",
    )
    macro_asset_cols = ("stock_macro_innov_beta_63", "stock_macro_shock_exposure")

    null_variants = [
        dict(macro=None, macro_mask=None),
        dict(macro=np.full((t_len, 12), np.nan, dtype=np.float32), macro_mask=None),
        dict(
            macro=rng.standard_normal((t_len, 12)).astype(np.float32),
            macro_mask=np.zeros((t_len, n_assets), dtype=bool),
        ),
        dict(macro=np.zeros((t_len, 420), dtype=np.float32), macro_mask=None),
    ]
    for kw in null_variants:
        res_null = mims.compute_market_indices_and_macro_sota_torch(o, h, lo, c, v, **kw)
        assert torch.isfinite(res_null["market_regime_indices"]).all()
        assert torch.isfinite(res_null["asset_index_macro_features"]).all()
        for cname in macro_reg_cols:
            assert float(res_null["market_regime_indices"][:, m_idx[cname]].abs().max()) == 0.0
        for cname in macro_asset_cols:
            assert (
                float(res_null["asset_index_macro_features"][:, :, a_idx[cname]].abs().max()) == 0.0
            )

    # 2) Macro real `encoded_v31_420` con alta a mitad de historia y shock hawkish/stress al final
    mac_420 = np.zeros((t_len, 420), dtype=np.float32)
    name_to_pos = {nm: i for i, nm in enumerate(mims.MACRO_V31_140_NAMES)}
    # Series diarias activas desde t=0 impulsadas por un factor macro hawkish común + ruido
    hawk_shocks = 0.02 * rng.standard_normal(t_len)
    hawk_level = np.cumsum(hawk_shocks)
    active_specs = (
        ("us_treasury_10y", 1.0),
        ("us_treasury_2y", 1.0),
        ("us_curve_10y_2y", -0.8),
        ("us_financial_conditions", 0.9),
        ("us_breakeven_10y", 0.4),
        ("us_broad_dollar", 0.8),
        ("wti_spot", -0.3),
        ("copper_global", -0.3),
    )
    for nm, load in active_specs:
        p = name_to_pos[nm]
        mac_420[:, p] = load * hawk_level + np.cumsum(0.008 * rng.standard_normal(t_len))
        mac_420[:, 140 + p] = 1.0
        mac_420[:, 280 + p] = 0.0  # fresco (0 días de retraso)

    # Serie que arranca a mitad de historia (t=80) con nivel alto +12.0 (ej. initial_claims / M2)
    # NO debe disparar un pico falso en t=80 porque cnt_lag < 10!
    p_late = name_to_pos["cn_m2_stock"]
    mac_420[80:, p_late] = 12.5 + 0.01 * rng.standard_normal(t_len - 80)
    mac_420[80:, 140 + p_late] = 1.0
    mac_420[80:, 280 + p_late] = np.log1p(15.0)

    # Inyectar un choque fuerte de endurecimiento/estrés en la última sesión
    for nm in ("us_treasury_10y", "us_treasury_2y", "us_financial_conditions", "us_broad_dollar"):
        mac_420[-1, name_to_pos[nm]] += 1.5
    mac_420[-1, name_to_pos["us_curve_10y_2y"]] -= 1.2

    # Comparar macro fresco vs macro muy obsoleto (log1p(500))
    mac_420_stale = mac_420.copy()
    mac_420_stale[:, 280:420] = np.log1p(500.0)

    # Hacer que el activo 0 suba con shocks macro hawkish y el activo 1 caiga con shocks hawkish
    r_base = np.diff(np.log(c), axis=0, prepend=np.log(c[:1]))
    r_adj = r_base.copy()
    r_adj[:, 0] = 0.80 * hawk_shocks + 0.003 * rng.standard_normal(t_len)
    r_adj[:, 1] = -0.80 * hawk_shocks + 0.003 * rng.standard_normal(t_len)
    c_mac = 50.0 * np.exp(np.cumsum(r_adj, axis=0)).astype(np.float32)
    o_mac = c_mac.copy()
    h_mac = (c_mac * 1.005).astype(np.float32)
    l_mac = (c_mac * 0.995).astype(np.float32)

    out_fresh = mims.compute_market_indices_and_macro_sota_torch(
        o_mac, h_mac, l_mac, c_mac, v, macro=mac_420, macro_layout="encoded_v31_420"
    )
    out_stale = mims.compute_market_indices_and_macro_sota_torch(
        o_mac, h_mac, l_mac, c_mac, v, macro=mac_420_stale, macro_layout="encoded_v31_420"
    )

    m_f = out_fresh["market_regime_indices"]
    m_s = out_stale["market_regime_indices"]
    a_f = out_fresh["asset_index_macro_features"]

    # El alta de `cn_m2_stock` en t=80 NO dispara explosión artificial
    assert abs(float(m_f[80, m_idx["macro_stress_composite"]])) < 2.5
    # El choque hawkish en t=-1 dispara estrés y PC1 frescos mucho más que obsoletos
    assert float(m_f[-1, m_idx["macro_stress_composite"]]) > 1.0
    assert float(m_f[-1, m_idx["macro_innov_pc1"]]) > 1.0
    assert abs(float(m_f[-1, m_idx["macro_innov_pc1"]])) > abs(
        float(m_s[-1, m_idx["macro_innov_pc1"]])
    )
    # Sensibilidad macro del activo 0 (+) vs activo 1 (-)
    assert float(a_f[-1, 0, a_idx["stock_macro_innov_beta_63"]]) > (
        float(a_f[-1, 1, a_idx["stock_macro_innov_beta_63"]]) + 0.3
    )
    assert float(a_f[-1, 0, a_idx["stock_macro_shock_exposure"]]) > (
        float(a_f[-1, 1, a_idx["stock_macro_shock_exposure"]]) + 0.3
    )


def test_3d_macro_mask_blocks_poisoned_asset_before_reduction_and_cache_reuse():
    """Regresión #226 P1-1/P2-2: macro 3D `[T, N, 420]` respeta `macro_mask` antes de promediar."""
    rng = np.random.default_rng(226)
    t_len, n_assets = 120, 16
    o, h, lo, c, v = _synthetic_market_panel(rng, t_len=t_len, n_assets=n_assets)

    mac_2d = np.zeros((t_len, 420), dtype=np.float32)
    mac_2d[:, :140] = np.cumsum(0.02 * rng.standard_normal((t_len, 140)), axis=0).astype(np.float32)
    mac_2d[:, 140:280] = 1.0
    mac_2d[:, 280:420] = 0.0

    # Expandir a 3D `[T, N, 420]` estilo `ZeroCopySessionStore`; ocultar activo 0 con `macro_mask`
    mac_clean_3d = np.broadcast_to(mac_2d[:, None, :], (t_len, n_assets, 420)).copy()
    mac_poison_3d = mac_clean_3d.copy()
    mac_poison_3d[:, 0, :140] = 1e6
    mac_poison_3d[:, 0, 140:280] = 1.0
    mac_poison_3d[:, 0, 280:420] = float("nan")

    mac_mask = np.ones((t_len, n_assets), dtype=bool)
    mac_mask[:, 0] = False

    out_clean = mims.compute_market_indices_and_macro_sota_torch(
        o, h, lo, c, v, macro=mac_clean_3d, macro_mask=mac_mask, macro_layout="encoded_v31_420"
    )
    out_poison = mims.compute_market_indices_and_macro_sota_torch(
        o, h, lo, c, v, macro=mac_poison_3d, macro_mask=mac_mask, macro_layout="encoded_v31_420"
    )
    for key in ("market_regime_indices", "asset_index_macro_features", "factor_returns"):
        assert torch.equal(out_clean[key], out_poison[key]), (
            f"Activo enmascarado contaminó `{key}` en macro 3D!"
        )
    # Caché por `(device, dtype)` reutilizada en llamadas repetidas (P2-2)
    assert len(mims._MACRO_V31_STATIC_CACHE) >= 1


def test_verifier_228_empty_n0_3d_and_feature_macro_masks_and_empty_audit():
    """Regresión #228 P1-1/P1-2/P2-1: `(T>0, N=0)` sin basura, máscaras 3D y audit vacío."""
    # 1) P1-1: `t_len > 0, n_assets == 0` debe devolver ceros finitos exactos (cero basura heap!)
    z50 = torch.zeros((5, 0), dtype=torch.float32)
    out_n0 = mims.compute_market_indices_and_macro_sota_torch(z50, z50, z50, z50, z50)
    assert out_n0["market_regime_indices"].shape == (5, 18)
    assert out_n0["asset_index_macro_features"].shape == (5, 0, 11)
    assert out_n0["factor_returns"].shape == (5, 5)
    assert out_n0["market_ew_returns"].shape == (5,)
    for k, v in out_n0.items():
        assert torch.isfinite(v).all(), f"{k} no finito en N=0"
        if v.numel() > 0:
            assert float(v.abs().max()) == 0.0, f"{k} contiene basura en N=0"

    # 2) P1-2: 3D `macro_mask [T, N, D]` y máscaras 2D por indicador `[T, 140]` / `[T, 420]`
    rng = np.random.default_rng(228)
    t_len, n_assets = 90, 10
    o, h, lo, c, v = _synthetic_market_panel(rng, t_len=t_len, n_assets=n_assets)

    mac_420 = np.zeros((t_len, 420), dtype=np.float32)
    rw_140 = np.cumsum(0.02 * rng.standard_normal((t_len, 140)), axis=0)
    mac_420[:, :140] = rw_140.astype(np.float32)
    mac_420[:, 140:280] = 1.0
    mac_420_3d = np.broadcast_to(mac_420[:, None, :], (t_len, n_assets, 420)).copy()
    mask_3d_420 = np.ones((t_len, n_assets, 420), dtype=bool)
    mask_3d_420[:, 0, :] = False
    out_3d_420 = mims.compute_market_indices_and_macro_sota_torch(
        o, h, lo, c, v, macro=mac_420_3d, macro_mask=mask_3d_420, macro_layout="encoded_v31_420"
    )
    assert torch.isfinite(out_3d_420["asset_index_macro_features"]).all()
    a_idx = {nm: i for i, nm in enumerate(mims.ASSET_INDEX_MACRO_NAMES)}
    b_col = a_idx["stock_macro_innov_beta_63"]
    assert float(out_3d_420["asset_index_macro_features"][:, 0, b_col].abs().max()) == 0.0

    mac_12_3d = rng.standard_normal((t_len, n_assets, 12)).astype(np.float32)
    mask_3d_12 = np.ones((t_len, n_assets, 12), dtype=bool)
    out_3d_12 = mims.compute_market_indices_and_macro_sota_torch(
        o, h, lo, c, v, macro=mac_12_3d, macro_mask=mask_3d_12
    )
    assert torch.isfinite(out_3d_12["asset_index_macro_features"]).all()

    # Máscaras 2D por indicador `[T, 140]` y `[T, 420]` emparejadas con `macro [T, 420]`
    mask_140 = np.ones((t_len, 140), dtype=bool)
    mask_140[:, 0] = False
    mac_420_p = mac_420.copy()
    mac_420_p[:, 0] = 1e6  # indicador 0 envenenado pero enmascarado por `mask_140`!
    out_m140_a = mims.compute_market_indices_and_macro_sota_torch(
        o, h, lo, c, v, macro=mac_420, macro_mask=mask_140, macro_layout="encoded_v31_420"
    )
    out_m140_b = mims.compute_market_indices_and_macro_sota_torch(
        o, h, lo, c, v, macro=mac_420_p, macro_mask=mask_140, macro_layout="encoded_v31_420"
    )
    assert torch.equal(out_m140_a["market_regime_indices"], out_m140_b["market_regime_indices"])

    mask_420 = np.ones((t_len, 420), dtype=bool)
    mask_420[:, 0] = False
    out_m420 = mims.compute_market_indices_and_macro_sota_torch(
        o, h, lo, c, v, macro=mac_420_p, macro_mask=mask_420, macro_layout="encoded_v31_420"
    )
    assert torch.equal(out_m140_a["market_regime_indices"], out_m420["market_regime_indices"])

    # 3) P2-1: `audit_market_indices_macro_value` seguro con paneles vacíos `T=0` o `N=0`
    for empty_shape in ((0, 6), (5, 0)):
        ze = torch.zeros(empty_shape, dtype=torch.float32)
        rep_e = mims.audit_market_indices_macro_value(ze, ze, ze, ze, ze, ze)
        assert rep_e["all_finite"] is True
        assert rep_e["within_bounds"] is True
        assert all(v == 0.0 for v in rep_e["max_abs_corr_vs_enriched_sota"].values())


def test_bit_exact_causality_bf16_parity_halts_and_zero_host_syncs():
    """Causalidad bit-a-bit (`diff==0.0`), paridad BF16 (`max_abs_diff==0.0`), halts y 0 syncs."""
    # 1) Auditoría estática AST: 0 host-syncs, 0 `linalg.solve(`, 0 `.nonzero()`, 0 `torch.tensor(`
    mod_path = Path(mims.__file__).resolve()
    sync_rep = audit_zero_host_syncs(mod_path)
    assert sync_rep["files_scanned"] == 1
    assert sync_rep["total_findings"] == 0, f"Host-syncs detectados: {sync_rep['findings']}"
    assert sync_rep["forward_findings"] == 0
    assert sync_rep["clean"] is True

    src = inspect.getsource(mims)
    assert "linalg.solve(" not in src, "Usar `torch.linalg.solve_ex(..., check_errors=False)`"
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith(
            ("compute_market_indices", "_")
        ):
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and child.attr == "nonzero":
                    raise AssertionError(f"`.nonzero()` detectado en {node.name}")
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "tensor"
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "torch"
                ):
                    raise AssertionError(f"`torch.tensor(...)` H2D detectado en {node.name}")

    # 2) Causalidad estricta bit-a-bit en OHLCV + macro 420-triplet
    rng = np.random.default_rng(31415)
    t_len, n_assets = 150, 24
    o, h, lo, c, v = _synthetic_market_panel(rng, t_len=t_len, n_assets=n_assets)
    mac_420 = np.zeros((t_len, 420), dtype=np.float32)
    mac_420[:, :140] = rng.standard_normal((t_len, 140)).astype(np.float32)
    mac_420[:, 140:280] = 1.0
    mac_420[:, 280:420] = np.log1p(rng.uniform(0.0, 20.0, size=(t_len, 140))).astype(np.float32)
    t_cut = 100

    res1 = mims.compute_market_indices_and_macro_sota_torch(o, h, lo, c, v, macro=mac_420)
    o2, h2, lo2, c2, v2, mac2 = o.copy(), h.copy(), lo.copy(), c.copy(), v.copy(), mac_420.copy()
    c2[t_cut + 1 :] *= 3.8
    h2[t_cut + 1 :] *= 4.5
    lo2[t_cut + 1 :] *= 0.25
    v2[t_cut + 1 :] *= 15.0
    mac2[t_cut + 1 :, :140] = -10.0
    mac2[t_cut + 1 :, 280:420] = 0.0

    res2 = mims.compute_market_indices_and_macro_sota_torch(o2, h2, lo2, c2, v2, macro=mac2)
    for key in (
        "market_regime_indices",
        "asset_index_macro_features",
        "factor_returns",
        "market_ew_returns",
    ):
        assert torch.equal(res1[key][: t_cut + 1], res2[key][: t_cut + 1]), (
            f"Fuga causal detectada en {key}!"
        )
        assert not torch.equal(res1[key][t_cut + 1 :], res2[key][t_cut + 1 :])

    # 3) Paridad exacta BF16 autocast (`max_abs_diff == 0.0` bajo Master FP32!)
    ot, ht, lt, ct, vt, mt = (torch.from_numpy(a) for a in (o, h, lo, c, v, mac_420))
    res_fp32 = mims.compute_market_indices_and_macro_sota_torch(ot, ht, lt, ct, vt, macro=mt)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        res_bf16 = mims.compute_market_indices_and_macro_sota_torch(ot, ht, lt, ct, vt, macro=mt)
    for key in ("market_regime_indices", "asset_index_macro_features", "factor_returns"):
        assert res_bf16[key].dtype == torch.float32
        assert float((res_bf16[key] - res_fp32[key]).abs().max()) == 0.0

    # 4) Estrés de activo 100% suspendido (`volume == 0` estilo CN)
    v_halt = v.copy()
    v_halt[:, 0] = 0.0
    res_halt = mims.compute_market_indices_and_macro_sota_torch(o, h, lo, c, v_halt, macro=mac_420)
    assert torch.isfinite(res_halt["market_regime_indices"]).all()
    assert torch.isfinite(res_halt["asset_index_macro_features"]).all()
    assert float(res_halt["asset_index_macro_features"][:, 0, :].abs().max()) == 0.0


def test_audit_market_indices_macro_value_end_to_end_leak_and_speed():
    """Auditoría integral de valor/ortogonalidad, rechazo de fugas y velocidad en N=1000."""
    rng = np.random.default_rng(9090)
    t_len, n_assets = 120, 32
    o, h, lo, c, v = _synthetic_market_panel(rng, t_len=t_len, n_assets=n_assets)
    mac = rng.standard_normal((t_len, 16)).astype(np.float32)

    log_c = np.log(c)
    fwd_ret = np.full_like(log_c, np.nan)
    fwd_ret[:-1] = log_c[1:] - log_c[:-1]

    rep = mims.audit_market_indices_macro_value(
        o,
        h,
        lo,
        c,
        v,
        fwd_ret,
        macro=mac,
        market_label="US_SYNTH",
    )
    assert rep["all_finite"] is True
    assert rep["within_bounds"] is True
    assert rep["leakage_screen_passed"] is True
    assert rep["max_abs_rank_ic"] < 0.10
    assert rep["max_abs_pearson_ic"] < 0.10
    assert rep["causality_check"]["bit_exact_causal"] is True
    assert rep["causality_check"]["max_abs_diff"] == 0.0
    assert rep["bf16_autocast_check"]["max_abs_diff"] == 0.0
    assert rep["macro_null_safety_check"]["passed"] is True
    assert rep["zero_host_syncs_clean"] is True
    assert set(rep["asset_feature_classifications"].keys()) == set(mims.ASSET_INDEX_MACRO_NAMES)
    assert set(rep["market_regime_classifications"].keys()) == set(mims.MARKET_REGIME_INDEX_NAMES)
    for cname in mims.MARKET_REGIME_INDEX_NAMES:
        expected_cls = "REGIME-CONDITIONING ONLY (FiLM/gating)"
        assert rep["market_regime_classifications"][cname] == expected_cls
    assert set(rep["max_abs_corr_vs_enriched_sota"].keys()) == set(mims.ASSET_INDEX_MACRO_NAMES)

    # Inyección explícita de fuga futura (y_{t+1} en un rasgo) -> debe reprobar la pantalla!
    leaky_asset = mims.compute_market_indices_and_macro_sota_torch(o, h, lo, c, v, macro=mac)[
        "asset_index_macro_features"
    ].clone()
    leaky_asset[:, :, 0] = torch.from_numpy(np.nan_to_num(fwd_ret, nan=0.0).astype(np.float32))
    rep_leak = mims.audit_market_indices_macro_value(
        o,
        h,
        lo,
        c,
        v,
        fwd_ret,
        macro=mac,
        precomputed_asset_features=leaky_asset,
        run_causality_check=False,
        run_bf16_check=False,
        run_macro_null_check=False,
    )
    assert rep_leak["leakage_screen_passed"] is False
    assert rep_leak["max_abs_rank_ic"] > 0.10

    # Velocidad vectorizada en T=100, N=1000 en CPU
    o_big, h_big, l_big, c_big, v_big = _synthetic_market_panel(rng, t_len=100, n_assets=1000)
    t0 = time.process_time()
    out_big = mims.compute_market_indices_and_macro_sota_torch(o_big, h_big, l_big, c_big, v_big)
    dt = time.process_time() - t0
    assert out_big["asset_index_macro_features"].shape == (100, 1000, 11)
    assert torch.isfinite(out_big["asset_index_macro_features"]).all()
    assert dt < 15.0, f"Demasiado lento en T=100,N=1000: {dt:.3f}s CPU"


def test_real_pre2024_us_and_cn_readonly_smoke():
    """Humo de solo lectura en historia real pre-2024 US y CN (estrictamente < 2024-01-01)."""
    mat_root = Path("/home/gonzalo/.local/state/mars-titan/history-2000-20261007/materialization")
    prep_root = mat_root / "prepared-v3.1" / "prepared"
    enc_root = mat_root / "encoded-history-v3.1" / "samples"
    if not (prep_root / "US").is_dir() or not (enc_root / "US").is_dir():
        pytest.skip("Directorio local de materialización pre-2024 no disponible")

    cfg = json.loads((mat_root / "encoded-history-v3.1" / "configuration.json").read_text())
    assert tuple(cfg["macro_indicators"]["US"]) == mims.MACRO_V31_140_NAMES
    assert tuple(cfg["macro_indicators"]["CN"]) == mims.MACRO_V31_140_NAMES

    for mkt in ("US", "CN"):
        tkr_dirs = sorted(p.name for p in (prep_root / mkt).iterdir() if p.is_dir())[:16]
        date_set: set[str] = set()
        px_tables: dict[
            str, tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        for tkr in tkr_dirs:
            pf = prep_root / mkt / tkr / "prices.parquet"
            if not pf.is_file():
                continue
            tbl = pq.read_table(pf, columns=["session", "open", "high", "low", "close", "volume"])
            s_all = [str(s)[:10] for s in tbl["session"].to_pylist()]
            keep = [i for i, s in enumerate(s_all) if "2022-01-01" <= s < "2024-01-01"]
            if len(keep) < 60:
                continue
            idx_k = np.asarray(keep, dtype=np.int64)
            s_k = [s_all[i] for i in keep]
            date_set.update(s_k)
            px_tables[tkr] = (
                s_k,
                tbl["open"].to_numpy(zero_copy_only=False)[idx_k],
                tbl["high"].to_numpy(zero_copy_only=False)[idx_k],
                tbl["low"].to_numpy(zero_copy_only=False)[idx_k],
                tbl["close"].to_numpy(zero_copy_only=False)[idx_k],
                tbl["volume"].to_numpy(zero_copy_only=False)[idx_k],
            )
        dates = sorted(date_set)
        assert max(dates) < "2024-01-01"
        d2i = {d: i for i, d in enumerate(dates)}
        tkrs = sorted(px_tables.keys())
        t_len, n_a = len(dates), len(tkrs)
        o_p = np.full((t_len, n_a), np.nan, dtype=np.float32)
        h_p = np.full((t_len, n_a), np.nan, dtype=np.float32)
        l_p = np.full((t_len, n_a), np.nan, dtype=np.float32)
        c_p = np.full((t_len, n_a), np.nan, dtype=np.float32)
        v_p = np.zeros((t_len, n_a), dtype=np.float32)
        for j, tkr in enumerate(tkrs):
            sk, oa, ha, la, ca, va = px_tables[tkr]
            di = np.asarray([d2i[d] for d in sk], dtype=np.int64)
            o_p[di, j] = oa
            h_p[di, j] = ha
            l_p[di, j] = la
            c_p[di, j] = ca
            v_p[di, j] = np.nan_to_num(va, nan=0.0)

        mac_p = np.full((t_len, 420), np.nan, dtype=np.float32)
        filled = np.zeros(t_len, dtype=bool)
        for tkr in tkrs:
            ef = enc_root / mkt / tkr / "samples.parquet"
            if not ef.is_file():
                continue
            etbl = pq.read_table(ef, columns=["session", "macro"])
            es = [str(s)[:10] for s in etbl["session"].to_pylist()]
            mac_raw = np.asarray(
                etbl["macro"].combine_chunks().values.to_numpy(zero_copy_only=False),
                dtype=np.float32,
            ).reshape(len(es), 420)
            for r_i, s_str in enumerate(es):
                d_i = d2i.get(s_str)
                if d_i is not None and not filled[d_i]:
                    mac_p[d_i] = mac_raw[r_i]
                    filled[d_i] = True
            if filled.all():
                break

        log_c = np.log(np.where(np.isfinite(c_p) & (c_p > 0), c_p, np.nan))
        fwd = np.full_like(c_p, np.nan)
        fwd[:-1] = log_c[1:] - log_c[:-1]
        rep = mims.audit_market_indices_macro_value(
            o_p, h_p, l_p, c_p, v_p, fwd, macro=mac_p, market_label=mkt
        )
        assert rep["all_finite"] is True
        assert rep["within_bounds"] is True
        assert rep["leakage_screen_passed"] is True
        assert rep["causality_check"]["bit_exact_causal"] is True
        assert rep["bf16_autocast_check"]["max_abs_diff"] == 0.0
        assert rep["macro_null_safety_check"]["passed"] is True
        assert rep["market_regime_stats"]["macro_innov_pc1"]["std"] > 1e-3
