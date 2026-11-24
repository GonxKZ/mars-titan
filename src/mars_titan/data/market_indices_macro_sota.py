"""Índices institucionales de mercado, portafolios factoriales y macro PIT derivados.

Diseñado como módulo complementario y estrictamente ortogonal a `dataset_enrichment_sota.py`
(cero duplicados frente a `ENRICHED_SOTA_NAMES`, cero host-syncs GPU `audit_zero_host_syncs == 0`,
precisión maestra FP32/FP64 dentro de `_autocast_disabled` bajo BF16 autocast, causalidad estricta
bit a bit `diff == 0.0` y separación total US/CN conforme a FM-8):

  1. `market_regime_indices [T, 18]` (`MARKET_REGIME_INDEX_NAMES`):
     Descriptores agregados de régimen a nivel de sesión para modulación FiLM / gating:
     - Amplitud e internos bursátiles: `mkt_ad_ema21`, `mkt_mcclellan_osc`, `mkt_pct_above_ma50`,
       `mkt_pct_above_ma200`, `mkt_nh_nl_spread_252`.
     - Concentración, dispersión y volatilidad sintética: `mkt_ew_vs_vw_spread_5`,
       `mkt_ew_vs_vw_spread_21`, `mkt_cs_dispersion_21`, `mkt_cs_skew_21`, `mkt_synth_vix_21`,
       `mkt_vol_term_slope_5_63`.
     - Turbulencia sistémica y estrés microestructural: `mkt_turbulence_63` (Kritzman-Li 2010
       sobre los 5 portafolios factoriales con `torch.linalg.solve_ex(..., check_errors=False)`),
       `mkt_liquidity_stress_21` (Amihud logarítmico invariante a divisa FX),
       `mkt_halt_limit_stress_5` (suspensiones + límites `±9.5%`, crítico en CN).
     - Compuestos macro PIT causales (`encoded_v31_420` con 6 pilares semánticos de los 140
       indicadores reales + PCA rodante causal por iteración de potencia sin cuSOLVER ni saltos
       de signo; o respaldo genérico `[T, M]`; exactamente `0.0` si `macro=None` o 100% ausente):
       `macro_innov_pc1`, `macro_innov_pc2`, `macro_dispersion_21`, `macro_stress_composite`.

  2. `factor_returns [T, 5]` (`FACTOR_PORTFOLIO_NAMES`):
     Portafolios miméticos estrictamente causales (`pesos_{t-1} * retorno_t`, cero fuga):
     `fac_mkt_vw`, `fac_liq_smb`, `fac_mom_umd`, `fac_rev_st`, `fac_bab_lowvol`.

  3. `asset_index_macro_features [T, N, 11]` (`ASSET_INDEX_MACRO_NAMES`, acotados en `[-8, 8]`):
     Señales transversales diferenciadas por activo para modelos de ranking `[T, N]`:
     `stock_rel_to_vw_5`, `stock_rel_to_vw_21`, `stock_factor_mom_score_21` (Ehsani-Linnainmaa
     2022), `stock_bab_beta_63`, `stock_downside_beta_63`, `stock_turbulence_beta_63`,
     `stock_breadth_x_mom_21`, `stock_ew_vw_tilt_21`, `stock_macro_innov_beta_63`,
     `stock_macro_shock_exposure`, `stock_vol_term_x_bab`.

  4. `audit_market_indices_macro_value(...)`:
     Auditoría empírica honesta de Rank-IC, t-stat HAC (Newey-West), correlación máxima frente a
     `ENRICHED_SOTA_NAMES`, clasificación explícita (`KEEP` / `REGIME-CONDITIONING ONLY` /
     `DROP/REDUNDANT` / `DROP/NEUTRAL`), causalidad bit-exact, paridad BF16, seguridad macro
     nulo/NaN y cero host-syncs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.nn import functional as F

from .dataset_enrichment_sota import (
    ENRICHED_SOTA_NAMES,
    LIMIT,
    _as_bool_tensor,
    _as_float_tensor,
    _autocast_disabled,
    _causal_affine_scan,
    _cross_sectional_pearson_series,
    _cross_sectional_rank_ic_series,
    _masked_median_lastdim,
    _masked_percentile_rank_2d,
    _newey_west_tstat_tensor,
    _normalize_optional_mask,
    _rolling_downside_beta_2d,
    _rolling_mean_cnt_2d,
    _rolling_sum_2d,
    _rolling_var_anchored_2d,
    _stable_calc_dtype,
    enrich_full_dataset_sota_torch,
)
from .institutional_alpha_v5 import causal_feature_quality_audit
from .market_physics_v4_gpu import gaussian_rank_features

FACTOR_PORTFOLIO_NAMES: tuple[str, ...] = (
    "fac_mkt_vw",
    "fac_liq_smb",
    "fac_mom_umd",
    "fac_rev_st",
    "fac_bab_lowvol",
)
FACTOR_PORTFOLIO_DIM: int = len(FACTOR_PORTFOLIO_NAMES)

MARKET_REGIME_INDEX_NAMES: tuple[str, ...] = (
    # 1) Equity Market Breadth & Internals
    "mkt_ad_ema21",
    "mkt_mcclellan_osc",
    "mkt_pct_above_ma50",
    "mkt_pct_above_ma200",
    "mkt_nh_nl_spread_252",
    # 2) Market Concentration, Dispersion & Synthetic Vol Index
    "mkt_ew_vs_vw_spread_5",
    "mkt_ew_vs_vw_spread_21",
    "mkt_cs_dispersion_21",
    "mkt_cs_skew_21",
    "mkt_synth_vix_21",
    "mkt_vol_term_slope_5_63",
    # 3) Systemic Turbulence & Microstructure Stress
    "mkt_turbulence_63",
    "mkt_liquidity_stress_21",
    "mkt_halt_limit_stress_5",
    # 4) Derived PIT Macro Composites
    "macro_innov_pc1",
    "macro_innov_pc2",
    "macro_dispersion_21",
    "macro_stress_composite",
)
MARKET_REGIME_INDEX_DIM: int = len(MARKET_REGIME_INDEX_NAMES)

ASSET_INDEX_MACRO_NAMES: tuple[str, ...] = (
    "stock_rel_to_vw_5",
    "stock_rel_to_vw_21",
    "stock_factor_mom_score_21",
    "stock_bab_beta_63",
    "stock_downside_beta_63",
    "stock_turbulence_beta_63",
    "stock_breadth_x_mom_21",
    "stock_ew_vw_tilt_21",
    "stock_macro_innov_beta_63",
    "stock_macro_shock_exposure",
    "stock_vol_term_x_bab",
)
ASSET_INDEX_MACRO_DIM: int = len(ASSET_INDEX_MACRO_NAMES)

# Subconjunto empíricamente validado como ortogonal (`KEEP`) en auditoría real pre-2024 US/CN:
# excluye `stock_rel_to_vw_{5,21}` (r=0.976-0.996 vs `peer_resid_mom_{5,21}`) y neutrales aislados
# (`stock_turbulence_beta_63`, `stock_macro_innov_beta_63`).
KEEP_ASSET_INDEX_MACRO_NAMES: tuple[str, ...] = (
    "stock_factor_mom_score_21",
    "stock_bab_beta_63",
    "stock_downside_beta_63",
    "stock_breadth_x_mom_21",
    "stock_ew_vw_tilt_21",
    "stock_macro_shock_exposure",
    "stock_vol_term_x_bab",
)
KEEP_ASSET_INDEX_MACRO_DIM: int = len(KEEP_ASSET_INDEX_MACRO_NAMES)
KEEP_ASSET_INDEX_MACRO_INDICES: tuple[int, ...] = tuple(
    ASSET_INDEX_MACRO_NAMES.index(nm) for nm in KEEP_ASSET_INDEX_MACRO_NAMES
)


def select_keep_asset_index_macro_features(
    asset_features: torch.Tensor,
) -> torch.Tensor:
    """Extrae el subconjunto ortogonal KEEP (`[..., 7]`) sin copias H2D ni host-syncs."""
    return torch.stack(
        [asset_features[..., i] for i in KEEP_ASSET_INDEX_MACRO_INDICES], dim=-1
    )

# Orden exacto de los 140 indicadores PIT reales en `encoded-history-v3.1/configuration.json`
# (idéntico en US y CN; bloque 420 = [values:140, observed:140, log_age:140])
MACRO_V31_140_NAMES: tuple[str, ...] = (
    "brent_spot",
    "brent_wti_spread",
    "cn_cpi_yoy_published",
    "cn_industrial_yoy_published",
    "cn_m2_stock",
    "cn_manufacturing_pmi",
    "cn_private_credit_gdp",
    "cn_private_credit_gdp_change_4q",
    "cn_real_effective_fx",
    "cn_tsf_flow",
    "cn_tsf_stock",
    "cny_per_usd",
    "cny_per_usd_change_21obs",
    "copper_global",
    "de_bond_10y",
    "ea_deposit_rate",
    "ea_real_effective_fx",
    "global_supply_pressure",
    "global_supply_pressure_change_1m",
    "jpy_per_usd",
    "us_bank_credit",
    "us_bank_credit_change_13w",
    "us_breakeven_10y",
    "us_breakeven_5y",
    "us_broad_dollar",
    "us_broad_dollar_change_21obs",
    "us_building_permits",
    "us_building_permits_yoy",
    "us_business_loans",
    "us_business_loans_yoy",
    "us_capacity_utilization",
    "us_capacity_utilization_change_1m",
    "us_consumer_loans",
    "us_consumer_loans_yoy",
    "us_continued_claims",
    "us_continued_claims_ma4",
    "us_core_cpi",
    "us_core_cpi_3m_annualized",
    "us_core_cpi_mom",
    "us_core_cpi_yoy",
    "us_core_pce_price",
    "us_core_pce_price_3m_annualized",
    "us_core_pce_price_mom",
    "us_core_pce_price_yoy",
    "us_cpi",
    "us_cpi_3m_annualized",
    "us_cpi_mom",
    "us_cpi_yoy",
    "us_curve_10y_2y",
    "us_curve_10y_3m",
    "us_curve_30y_10y",
    "us_curve_5y_2y",
    "us_effective_target_gap",
    "us_employment_ratio",
    "us_employment_ratio_change_1m",
    "us_factory_orders",
    "us_factory_orders_mom",
    "us_fed_assets",
    "us_fed_assets_change_4w",
    "us_fed_funds",
    "us_fed_funds_change_21d",
    "us_financial_conditions",
    "us_financial_conditions_change_4w",
    "us_financial_stress",
    "us_forward_inflation_5y5y_proxy",
    "us_hourly_wage",
    "us_hourly_wage_yoy",
    "us_housing_starts",
    "us_housing_starts_yoy",
    "us_industrial_production",
    "us_industrial_production_mom",
    "us_industrial_production_yoy",
    "us_initial_claims",
    "us_initial_claims_change_4w",
    "us_initial_claims_ma4",
    "us_job_openings",
    "us_job_openings_yoy",
    "us_m2",
    "us_m2_yoy",
    "us_manufacturing_hours",
    "us_manufacturing_hours_change_1m",
    "us_nominal_gdp",
    "us_nominal_gdp_yoy",
    "us_participation",
    "us_participation_change_1m",
    "us_payrolls",
    "us_payrolls_change_1m",
    "us_payrolls_yoy",
    "us_pce_price",
    "us_pce_price_3m_annualized",
    "us_pce_price_mom",
    "us_pce_price_yoy",
    "us_permits_starts_ratio",
    "us_ppi_all",
    "us_ppi_all_mom",
    "us_ppi_all_yoy",
    "us_ppi_final",
    "us_ppi_final_mom",
    "us_ppi_final_yoy",
    "us_quits_rate",
    "us_quits_rate_change_1m",
    "us_real_consumption",
    "us_real_consumption_mom",
    "us_real_curve_10y_5y",
    "us_real_disposable_income",
    "us_real_disposable_income_mom",
    "us_real_gdp",
    "us_real_gdp_qoq_annualized",
    "us_real_gdp_yoy",
    "us_real_government",
    "us_real_government_qoq",
    "us_real_investment",
    "us_real_investment_qoq",
    "us_real_retail_sales",
    "us_real_retail_sales_mom",
    "us_reserve_balances",
    "us_retail_sales",
    "us_retail_sales_mom",
    "us_retail_sales_yoy",
    "us_reverse_repo",
    "us_saving_rate",
    "us_saving_rate_change_1m",
    "us_sofr",
    "us_sofr_fedfunds_gap",
    "us_target_lower",
    "us_target_upper",
    "us_tips_10y",
    "us_tips_5y",
    "us_treasury_10y",
    "us_treasury_10y_change_21obs",
    "us_treasury_2y",
    "us_treasury_30y",
    "us_treasury_3m",
    "us_treasury_5y",
    "us_treasury_account",
    "us_unemployment",
    "us_unemployment_change_1m",
    "usd_per_eur",
    "wti_spot",
    "wti_spot_change_21obs",
)

_MACRO_V31_INDEX: dict[str, int] = {nm: i for i, nm in enumerate(MACRO_V31_140_NAMES)}

# 6 pilares económicos PIT derivados del esquema real `encoded-history-v3.1` (140 series)
MACRO_PILLAR_NAMES: tuple[str, ...] = (
    "rates_level",
    "yield_curve",
    "fin_stress_credit",
    "inflation_breakeven",
    "fx_dollar_cny",
    "commodities_growth_cn",
)

_MACRO_PILLAR_NAME_SPECS: tuple[tuple[tuple[str, float], ...], ...] = (
    # 0) rates_level (+ = tipos / yields más altos / hawkish)
    (
        ("us_treasury_10y", 1.0),
        ("us_treasury_2y", 1.0),
        ("us_treasury_5y", 1.0),
        ("us_treasury_30y", 1.0),
        ("us_treasury_3m", 1.0),
        ("us_treasury_10y_change_21obs", 1.0),
        ("us_fed_funds", 1.0),
        ("us_fed_funds_change_21d", 1.0),
        ("us_tips_10y", 1.0),
        ("us_tips_5y", 1.0),
        ("us_sofr", 1.0),
        ("de_bond_10y", 1.0),
        ("ea_deposit_rate", 1.0),
    ),
    # 1) yield_curve (+ = pendiente más empinada / expansión; - = inversión)
    (
        ("us_curve_10y_2y", 1.0),
        ("us_curve_10y_3m", 1.0),
        ("us_curve_30y_10y", 1.0),
        ("us_curve_5y_2y", 1.0),
        ("us_real_curve_10y_5y", 1.0),
    ),
    # 2) fin_stress_credit (+ = condiciones financieras más restrictivas / estrés crédito-laboral)
    (
        ("us_financial_conditions", 1.0),
        ("us_financial_conditions_change_4w", 1.0),
        ("us_financial_stress", 1.0),
        ("us_initial_claims", 1.0),
        ("us_initial_claims_change_4w", 1.0),
        ("us_continued_claims", 1.0),
        ("us_unemployment", 1.0),
        ("us_unemployment_change_1m", 1.0),
        ("us_sofr_fedfunds_gap", 1.0),
        ("global_supply_pressure", 1.0),
        ("global_supply_pressure_change_1m", 1.0),
        ("us_bank_credit_change_13w", -1.0),
        ("us_fed_assets_change_4w", -1.0),
    ),
    # 3) inflation_breakeven (+ = expectativas e inflación realizada al alza)
    (
        ("us_breakeven_10y", 1.0),
        ("us_breakeven_5y", 1.0),
        ("us_forward_inflation_5y5y_proxy", 1.0),
        ("us_cpi_yoy", 1.0),
        ("us_cpi_mom", 1.0),
        ("us_core_cpi_yoy", 1.0),
        ("us_core_cpi_mom", 1.0),
        ("us_core_pce_price_yoy", 1.0),
        ("us_pce_price_yoy", 1.0),
        ("us_ppi_all_yoy", 1.0),
        ("us_ppi_final_yoy", 1.0),
        ("cn_cpi_yoy_published", 1.0),
    ),
    # 4) fx_dollar_cny (+ = USD más fuerte / CNY-EUR-JPY más débiles -> apretón global)
    (
        ("us_broad_dollar", 1.0),
        ("us_broad_dollar_change_21obs", 1.0),
        ("cny_per_usd", 1.0),
        ("cny_per_usd_change_21obs", 1.0),
        ("jpy_per_usd", 1.0),
        ("usd_per_eur", -1.0),
        ("cn_real_effective_fx", -1.0),
        ("ea_real_effective_fx", -1.0),
    ),
    # 5) commodities_growth_cn (+ = reflación global, materias primas y actividad real US/CN)
    (
        ("brent_spot", 1.0),
        ("wti_spot", 1.0),
        ("wti_spot_change_21obs", 1.0),
        ("copper_global", 1.0),
        ("cn_manufacturing_pmi", 1.0),
        ("cn_industrial_yoy_published", 1.0),
        ("cn_tsf_flow", 1.0),
        ("cn_private_credit_gdp_change_4q", 1.0),
        ("us_industrial_production_yoy", 1.0),
        ("us_industrial_production_mom", 1.0),
        ("us_capacity_utilization_change_1m", 1.0),
        ("us_payrolls_change_1m", 1.0),
        ("us_real_gdp_yoy", 1.0),
        ("us_real_retail_sales_mom", 1.0),
        ("us_housing_starts_yoy", 1.0),
        ("us_building_permits_yoy", 1.0),
    ),
)

# Matriz dispersa fija `[140, 6]` preconstruida en NumPy (verificada en tiempo de carga)
_N_PILLARS: int = len(MACRO_PILLAR_NAMES)
_MACRO_PILLAR_SIGN_MATRIX_NP: np.ndarray = np.zeros((140, _N_PILLARS), dtype=np.float32)
_MACRO_PILLAR_ABS_MASK_NP: np.ndarray = np.zeros((140, _N_PILLARS), dtype=np.float32)
for _p_i, _spec in enumerate(_MACRO_PILLAR_NAME_SPECS):
    for _ind_name, _sgn in _spec:
        _col_i = _MACRO_V31_INDEX[_ind_name]
        _MACRO_PILLAR_SIGN_MATRIX_NP[_col_i, _p_i] = float(_sgn)
        _MACRO_PILLAR_ABS_MASK_NP[_col_i, _p_i] = 1.0

# Anclas semánticas unitarias deterministas para evitar saltos arbitrarios de signo en PCA rodante
_ANCHOR_V31_PC1_NP = np.asarray([1.0, -0.5, 1.0, 0.5, 0.7, -0.5], dtype=np.float64)
_ANCHOR_V31_PC1_NP = _ANCHOR_V31_PC1_NP / np.linalg.norm(_ANCHOR_V31_PC1_NP)
_ANCHOR_V31_PC2_NP = np.asarray([0.3, 0.7, -0.7, 0.5, -0.5, 1.0], dtype=np.float64)
_ANCHOR_V31_PC2_NP = _ANCHOR_V31_PC2_NP - (
    float(np.dot(_ANCHOR_V31_PC2_NP, _ANCHOR_V31_PC1_NP)) * _ANCHOR_V31_PC1_NP
)
_ANCHOR_V31_PC2_NP = _ANCHOR_V31_PC2_NP / np.linalg.norm(_ANCHOR_V31_PC2_NP)

# Cachés por `(device, dtype)` para evitar copias H2D repetidas en bucles CUDA (P2-2)
_MACRO_V31_STATIC_CACHE: dict[
    tuple[torch.device, torch.dtype],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_MACRO_RAW_ANCHOR_CACHE: dict[tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_macro_v31_static_tensors(
    device: torch.device,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (device, compute_dtype)
    cached = _MACRO_V31_STATIC_CACHE.get(key)
    if cached is None:
        sign_mat = torch.as_tensor(
            _MACRO_PILLAR_SIGN_MATRIX_NP, dtype=compute_dtype, device=device
        )
        mask_mat = torch.as_tensor(
            _MACRO_PILLAR_ABS_MASK_NP, dtype=compute_dtype, device=device
        )
        anchor1 = torch.as_tensor(_ANCHOR_V31_PC1_NP, dtype=torch.float64, device=device)
        anchor2 = torch.as_tensor(_ANCHOR_V31_PC2_NP, dtype=torch.float64, device=device)
        cached = (sign_mat, mask_mat, anchor1, anchor2)
        _MACRO_V31_STATIC_CACHE[key] = cached
    return cached


def _get_raw_macro_anchors(
    device: torch.device,
    k_p: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device, k_p)
    cached = _MACRO_RAW_ANCHOR_CACHE.get(key)
    if cached is None:
        a1_np = np.ones((k_p,), dtype=np.float64) / math.sqrt(float(k_p))
        a2_np = np.asarray([1.0 if i % 2 == 0 else -1.0 for i in range(k_p)], dtype=np.float64)
        a2_np = a2_np - float(np.dot(a2_np, a1_np)) * a1_np
        n2 = float(np.linalg.norm(a2_np))
        if n2 > 1e-12:
            a2_np = a2_np / n2
        anchor1 = torch.as_tensor(a1_np, dtype=torch.float64, device=device)
        anchor2 = torch.as_tensor(a2_np, dtype=torch.float64, device=device)
        cached = (anchor1, anchor2)
        _MACRO_RAW_ANCHOR_CACHE[key] = cached
    return cached


def _lag1(x: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """Desplaza estrictamente 1 sesión hacia adelante (`out[t] = x[t-1]`, `out[0] = fill`)."""
    t_len = x.shape[0]
    if t_len == 0:
        return x.clone()
    pad = torch.full((1, *x.shape[1:]), fill, dtype=x.dtype, device=x.device)
    if t_len == 1:
        return pad
    return torch.cat([pad, x[:-1]], dim=0)


def _lag_k(x: torch.Tensor, k: int, fill: float = 0.0) -> torch.Tensor:
    """Desplaza `k` sesiones causales (`out[t] = x[t-k]`, primeros `k` = `fill`)."""
    t_len = x.shape[0]
    if k <= 0 or t_len == 0:
        return x.clone()
    k_eff = min(k, t_len)
    pad = torch.full((k_eff, *x.shape[1:]), fill, dtype=x.dtype, device=x.device)
    if k_eff == t_len:
        return pad
    return torch.cat([pad, x[:-k_eff]], dim=0)


def _causal_ema_nd(x: torch.Tensor, valid: torch.Tensor, *, span: float) -> torch.Tensor:
    """EMA causal sobre eje temporal 0 con mantenimiento en huecos (`O(log T)`, 0 syncs)."""
    t_len = x.shape[0]
    if t_len == 0:
        return torch.zeros_like(x)
    alpha = 2.0 / (max(float(span), 1.0) + 1.0)
    decay = 1.0 - alpha
    v_bool = valid & torch.isfinite(x)
    x_safe = torch.where(v_bool, x, 0.0)
    seen_cnt = torch.cumsum(v_bool.to(torch.int64), dim=0)
    is_first = v_bool & (seen_cnt == 1)
    a_scan = torch.where(is_first, 0.0, torch.where(v_bool, decay, 1.0)).to(dtype=x.dtype)
    b_scan = torch.where(is_first, x_safe, torch.where(v_bool, alpha * x_safe, 0.0))
    ema = _causal_affine_scan(a_scan, b_scan)
    return torch.where(seen_cnt > 0, ema, 0.0)


def _rolling_beta_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    valid: torch.Tensor,
    window: int = 63,
    *,
    min_obs: int = 15,
    standardize_by_x_std: bool = False,
    vasicek_prior: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Beta rodante causal `cov(x, y) / var(x)` en bloques anclados FP64 (`O(T*N)`, 0 deriva)."""
    t_len, n_assets = y.shape
    dev, dtype = y.device, y.dtype
    out = torch.zeros((t_len, n_assets), dtype=dtype, device=dev)
    if t_len == 0 or n_assets == 0 or window <= 1:
        return out

    x_2d = x.unsqueeze(1) if x.ndim == 1 else x
    block_t = 256
    zero = torch.zeros((1, n_assets), dtype=torch.float64, device=dev)
    min_f = float(min_obs)

    for b0 in range(0, t_len, block_t):
        b1 = min(b0 + block_t, t_len)
        s0 = max(0, b0 - window + 1)
        m_blk = valid[s0:b1]
        x_blk = x_2d[s0:b1].to(torch.float64)
        y_blk = y[s0:b1].to(torch.float64)
        if x_blk.shape[1] == 1:
            ax = torch.where(
                m_blk[0:1].any(dim=1, keepdim=True) & torch.isfinite(x_blk[0:1]),
                x_blk[0:1],
                0.0,
            )
        else:
            ax = torch.where(m_blk[0:1], x_blk[0:1], 0.0)
        ay = torch.where(m_blk[0:1], y_blk[0:1], 0.0)

        xs = torch.where(m_blk, x_blk - ax, 0.0)
        ys = torch.where(m_blk, y_blk - ay, 0.0)
        mf = m_blk.to(torch.float64)

        c_cnt = torch.cat([zero, torch.cumsum(mf, dim=0)], dim=0)
        c_x = torch.cat([zero, torch.cumsum(xs, dim=0)], dim=0)
        c_y = torch.cat([zero, torch.cumsum(ys, dim=0)], dim=0)
        c_x2 = torch.cat([zero, torch.cumsum(xs * xs, dim=0)], dim=0)
        c_xy = torch.cat([zero, torch.cumsum(xs * ys, dim=0)], dim=0)

        l_blk = b1 - s0
        hi = torch.arange(b0 - s0 + 1, l_blk + 1, device=dev)
        lo = (hi - window).clamp_min(0)
        cnt = c_cnt[hi] - c_cnt[lo]
        denom = cnt.clamp_min(1.0)
        s1x = c_x[hi] - c_x[lo]
        s1y = c_y[hi] - c_y[lo]
        s2x = c_x2[hi] - c_x2[lo]
        sxy = c_xy[hi] - c_xy[lo]

        cx = (s2x - (s1x * s1x) / denom).clamp_min(0.0)
        cx = torch.where(cx <= 1e-12 * s2x, 0.0, cx)
        cxy = torch.where(cx == 0.0, 0.0, sxy - (s1x * s1y) / denom)

        ok_reg = (cnt >= min_f) & (cx > 1e-12)
        if standardize_by_x_std:
            # Retorno por choque de 1-sigma en x: cov(x,y) / std(x) = cxy / sqrt(cx * (cnt - 1))
            denom_std = torch.sqrt((cx * (cnt - 1.0).clamp_min(1.0)).clamp_min(1e-24))
            beta = torch.where(ok_reg, cxy / denom_std, 0.0)
        else:
            beta_raw = torch.where(ok_reg, cxy / (cx + 1e-10), 0.0)
            if vasicek_prior is not None:
                prior_mean, prior_n = vasicek_prior
                beta = torch.where(
                    ok_reg,
                    (cnt * beta_raw + prior_n * prior_mean) / (cnt + prior_n),
                    0.0,
                )
            else:
                beta = beta_raw
        out[b0:b1] = beta.to(dtype)
    return out


def _rolling_cov_lagged(
    x_tk: torch.Tensor,
    valid_t: torch.Tensor,
    window: int = 63,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Media y covarianza rodantes causales rezagadas `[t-window..t-1]` en FP64 (`O(T*K^2)`)."""
    t_len, k_dim = x_tk.shape
    dev = x_tk.device
    cnt_out = torch.zeros((t_len,), dtype=torch.float64, device=dev)
    mean_out = torch.zeros((t_len, k_dim), dtype=torch.float64, device=dev)
    cov_out = torch.zeros((t_len, k_dim, k_dim), dtype=torch.float64, device=dev)
    if t_len <= 1 or k_dim == 0 or window <= 1:
        return cnt_out, mean_out, cov_out

    block_t = 256
    z1 = torch.zeros((1,), dtype=torch.float64, device=dev)
    zk = torch.zeros((1, k_dim), dtype=torch.float64, device=dev)
    zkk = torch.zeros((1, k_dim, k_dim), dtype=torch.float64, device=dev)

    cnt_cur = torch.zeros((t_len,), dtype=torch.float64, device=dev)
    mean_cur = torch.zeros((t_len, k_dim), dtype=torch.float64, device=dev)
    cov_cur = torch.zeros((t_len, k_dim, k_dim), dtype=torch.float64, device=dev)

    for b0 in range(0, t_len, block_t):
        b1 = min(b0 + block_t, t_len)
        s0 = max(0, b0 - window + 1)
        m_blk = valid_t[s0:b1] & torch.isfinite(x_tk[s0:b1]).all(dim=-1)
        x_blk = x_tk[s0:b1].to(torch.float64)
        ax = torch.where(m_blk[0:1].unsqueeze(-1), x_blk[0:1], 0.0)
        xs = torch.where(m_blk.unsqueeze(-1), x_blk - ax, 0.0)
        outer = xs.unsqueeze(-1) * xs.unsqueeze(-2)  # [L, K, K]

        c_cnt = torch.cat([z1, torch.cumsum(m_blk.to(torch.float64), dim=0)], dim=0)
        c_s1 = torch.cat([zk, torch.cumsum(xs, dim=0)], dim=0)
        c_s2 = torch.cat([zkk, torch.cumsum(outer, dim=0)], dim=0)

        l_blk = b1 - s0
        hi = torch.arange(b0 - s0 + 1, l_blk + 1, device=dev)
        lo = (hi - window).clamp_min(0)
        cnt = c_cnt[hi] - c_cnt[lo]
        denom = cnt.clamp_min(1.0)
        s1 = c_s1[hi] - c_s1[lo]
        s2 = c_s2[hi] - c_s2[lo]
        mu_s = s1 / denom.unsqueeze(-1)
        css = s2 - mu_s.unsqueeze(-1) * s1.unsqueeze(-2)
        css = 0.5 * (css + css.transpose(-1, -2))
        cov_loc = torch.where(
            (cnt >= 2.0).view(-1, 1, 1),
            css / (cnt - 1.0).clamp_min(1.0).view(-1, 1, 1),
            0.0,
        )
        cnt_cur[b0:b1] = cnt
        mean_cur[b0:b1] = mu_s + ax
        cov_cur[b0:b1] = cov_loc

    # Rezagar estrictamente 1 sesión: estadísticas en `t` usan SOLO `[t-window..t-1]`
    cnt_out[1:] = cnt_cur[:-1]
    mean_out[1:] = mean_cur[:-1]
    cov_out[1:] = cov_cur[:-1]
    return cnt_out, mean_out, cov_out


def _rolling_prev_extrema_2d(
    log_c: torch.Tensor,
    traded: torch.Tensor,
    window: int = 251,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Máximo y mínimo causales de `log_c[t-window .. t-1]` y conteo previo (`O(T*N)`, 0 syncs)."""
    t_len, n_assets = log_c.shape
    dtype = log_c.dtype
    if t_len == 0 or n_assets == 0:
        z = torch.zeros_like(log_c)
        return z, z, z

    huge = 1e30
    prev_traded = _lag1(traded.to(dtype), fill=0.0) > 0.5
    prev_log_c = _lag1(log_c, fill=0.0)
    prev_cnt = _rolling_sum_2d(prev_traded.to(dtype), window, assume_finite=True)

    hi_in = torch.where(prev_traded, prev_log_c, -huge)
    lo_in = torch.where(prev_traded, -prev_log_c, -huge)

    # max_pool1d en `[N, 1, T + window - 1]` con padding causal a la izquierda
    pad_l = window - 1
    hi_n1t = hi_in.transpose(0, 1).unsqueeze(1).contiguous()
    lo_n1t = lo_in.transpose(0, 1).unsqueeze(1).contiguous()
    if pad_l > 0:
        hi_n1t = F.pad(hi_n1t, (pad_l, 0), value=-huge)
        lo_n1t = F.pad(lo_n1t, (pad_l, 0), value=-huge)
    prev_max = F.max_pool1d(hi_n1t, kernel_size=window, stride=1).squeeze(1).transpose(0, 1)
    prev_min = -F.max_pool1d(lo_n1t, kernel_size=window, stride=1).squeeze(1).transpose(0, 1)
    return prev_max, prev_min, prev_cnt


def _build_long_short_factor(
    signal_lag: torch.Tensor,
    ret_cc: torch.Tensor,
    valid_sig_lag: torch.Tensor,
    r_fin: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Portafolio dólar-neutral estrictamente causal: rangos `t-1` x retornos `t` (`[T]`)."""
    ok = valid_sig_lag & r_fin & torch.isfinite(signal_lag)
    cnt = ok.sum(dim=1)
    pct = _masked_percentile_rank_2d(signal_lag, ok)
    z = torch.where(ok, 2.0 * pct - 1.0, 0.0)
    w_pos = F.relu(z)
    w_neg = F.relu(-z)
    sum_pos = w_pos.sum(dim=1, keepdim=True)
    sum_neg = w_neg.sum(dim=1, keepdim=True)
    r_safe = torch.where(ok, ret_cc, 0.0)
    leg_pos = (w_pos * r_safe).sum(dim=1) / sum_pos.squeeze(1).clamp_min(1e-12)
    leg_neg = (w_neg * r_safe).sum(dim=1) / sum_neg.squeeze(1).clamp_min(1e-12)
    has_both = (cnt >= 4) & (sum_pos.squeeze(1) > 1e-12) & (sum_neg.squeeze(1) > 1e-12)
    return torch.where(has_both, leg_pos - leg_neg, 0.0).to(compute_dtype)


def _compute_pit_macro_composites(
    macro: torch.Tensor | np.ndarray | None,
    *,
    macro_mask: torch.Tensor | np.ndarray | None,
    t_len: int,
    n_assets: int,
    macro_layout: Literal["auto", "encoded_v31_420", "raw"],
    staleness_scale: float,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Calcula `(pc1, pc2, disp_21, stress, macro_sess_ok, asset_mask_tn)` PIT y 0-sync."""
    z_t = torch.zeros((t_len,), dtype=compute_dtype, device=device)
    f_t = torch.zeros((t_len,), dtype=torch.bool, device=device)
    f_tn = torch.zeros((t_len, n_assets), dtype=torch.bool, device=device)
    if macro is None or t_len == 0:
        return z_t, z_t, z_t, z_t, f_t, f_tn

    mac_t = _as_float_tensor(macro, dtype=compute_dtype, device=device)
    if mac_t.ndim == 1:
        mac_2d = mac_t.unsqueeze(1)  # [T, 1]
    elif mac_t.ndim == 2:
        mac_2d = mac_t  # [T, M]
    elif mac_t.ndim == 3:
        if macro_mask is not None:
            m_3d = _normalize_optional_mask(macro_mask, target_shape=mac_t.shape, device=device)
            fin_m = torch.isfinite(mac_t) & m_3d
        else:
            fin_m = torch.isfinite(mac_t)
        cnt_m = fin_m.to(compute_dtype).sum(dim=1).clamp_min(1.0)
        mac_2d = torch.where(fin_m, mac_t, 0.0).sum(dim=1) / cnt_m
        mac_2d = torch.where(fin_m.any(dim=1), mac_2d, float("nan"))
    else:
        raise ValueError(f"Forma inválida de macro: {tuple(mac_t.shape)}")

    if mac_2d.shape[0] != t_len or mac_2d.shape[1] == 0:
        if mac_2d.shape[0] != t_len:
            raise ValueError(f"macro.shape[0]={mac_2d.shape[0]} no coincide con T={t_len}")
        return z_t, z_t, z_t, z_t, f_t, f_tn

    d_m = mac_2d.shape[1]
    use_v31 = macro_layout == "encoded_v31_420" or (macro_layout == "auto" and d_m == 420)
    n_ind_eff = 140 if (use_v31 and d_m == 420) else d_m

    # Unificar `macro_mask` (1D `[T]`, 2D `[T, N]` o `[T, M]`/`[T, 140]`, 3D `[T, N, D]`)
    feat_mask_tm: torch.Tensor | None = None
    if macro_mask is not None:
        m_raw = _as_bool_tensor(macro_mask, device=device)
        if m_raw.ndim == 1:
            sess_mask = m_raw
            asset_mask_tn = m_raw.unsqueeze(1).expand(t_len, n_assets)
        elif m_raw.ndim == 3 and m_raw.shape[0] == t_len:
            sess_mask = m_raw.any(dim=(1, 2))
            m_tn_red = m_raw.any(dim=-1)
            asset_mask_tn = (
                m_tn_red.expand(t_len, n_assets)
                if m_tn_red.shape[1] in (1, n_assets)
                else sess_mask.unsqueeze(1).expand(t_len, n_assets)
            )
        elif m_raw.ndim == 2 and m_raw.shape[0] == t_len:
            k_m = m_raw.shape[1]
            is_feat_mask = (
                mac_t.ndim == 2
                and k_m != n_assets
                and k_m in (d_m, n_ind_eff)
            )
            if is_feat_mask:
                feat_mask_tm = (
                    m_raw[:, :140] if (use_v31 and d_m == 420 and k_m == 420) else m_raw
                )
                sess_mask = feat_mask_tm.any(dim=1)
                asset_mask_tn = sess_mask.unsqueeze(1).expand(t_len, n_assets)
            elif k_m in (1, n_assets):
                sess_mask = m_raw.any(dim=1)
                asset_mask_tn = m_raw.expand(t_len, n_assets)
            else:
                sess_mask = m_raw.any(dim=1)
                asset_mask_tn = sess_mask.unsqueeze(1).expand(t_len, n_assets)
        else:
            asset_mask_tn = _normalize_optional_mask(
                m_raw, target_shape=(t_len, n_assets), device=device
            )
            sess_mask = asset_mask_tn.any(dim=1)
    else:
        sess_mask = torch.ones((t_len,), dtype=torch.bool, device=device)
        asset_mask_tn = torch.ones((t_len, n_assets), dtype=torch.bool, device=device)

    if use_v31 and d_m == 420:
        vals, obs, lage = torch.split(mac_2d, 140, dim=1)  # [T, 140] cada uno
        valid_tm = (
            torch.isfinite(vals)
            & torch.isfinite(obs)
            & (obs > 0.5)
            & sess_mask.unsqueeze(1)
        )
        if feat_mask_tm is not None:
            valid_tm = valid_tm & feat_mask_tm
        lage_safe = torch.where(valid_tm & torch.isfinite(lage), lage, 0.0).clamp(0.0, 10.0)
        fresh_tm = torch.where(
            valid_tm,
            torch.exp(-lage_safe / max(staleness_scale, 1e-6)),
            0.0,
        )
    else:
        vals = mac_2d
        valid_tm = torch.isfinite(vals) & sess_mask.unsqueeze(1)
        if feat_mask_tm is not None:
            valid_tm = valid_tm & feat_mask_tm
        fresh_tm = valid_tm.to(compute_dtype)

    n_ind = vals.shape[1]

    # Innovaciones estacionarias I(0) estrictamente PIT/causales (`val_t - ema_10[t-1]`):
    # elimina correlaciones espurias de paseos aleatorios (Granger-Newbold) y picos al alta!
    ema_10 = _causal_ema_nd(vals, valid_tm, span=10.0)
    ema_10_lag = _lag1(ema_10, fill=0.0)
    obs_cnt_lag = _lag1(torch.cumsum(valid_tm.to(compute_dtype), dim=0), fill=0.0)
    raw_innov = torch.where(valid_tm & (obs_cnt_lag >= 1.0), vals - ema_10_lag, float("nan"))

    in_var_63, in_cnt_63 = _rolling_var_anchored_2d(raw_innov, 63)
    in_sq_m63, _ = _rolling_mean_cnt_2d(
        torch.where(torch.isfinite(raw_innov), raw_innov * raw_innov, float("nan")), 63
    )
    cnt_lag = _lag1(in_cnt_63, fill=0.0)
    sd_var_lag = torch.sqrt(_lag1(in_var_63, fill=0.0).clamp_min(0.0))
    sd_rms_lag = torch.sqrt(_lag1(in_sq_m63, fill=0.0).clamp_min(0.0))
    sd_lag = torch.maximum(sd_var_lag, sd_rms_lag)

    # Requerir >=10 innovaciones previas evita picos falsos cuando una serie arranca a mitad!
    innov_ok = valid_tm & torch.isfinite(raw_innov) & (cnt_lag >= 10.0) & (sd_lag > 1e-6)
    z_innov = torch.where(
        innov_ok,
        (torch.where(innov_ok, raw_innov, 0.0) / sd_lag.clamp_min(1e-6)).clamp(-5.0, 5.0)
        * fresh_tm,
        0.0,
    )  # [T, n_ind]

    any_innov_t = innov_ok.any(dim=1)  # [T]
    cnt_innov_t = innov_ok.to(compute_dtype).sum(dim=1)  # [T]

    # 1) Dispersión transversal entre indicadores macro (21d EMA)
    z_mean_t = torch.where(any_innov_t, z_innov.sum(dim=1) / cnt_innov_t.clamp_min(1.0), 0.0)
    z_diff = torch.where(innov_ok, z_innov - z_mean_t.unsqueeze(1), 0.0)
    z_disp_daily = torch.where(
        cnt_innov_t >= 2.0,
        torch.sqrt(((z_diff * z_diff).sum(dim=1) / cnt_innov_t.clamp_min(1.0)).clamp_min(0.0)),
        0.0,
    )
    macro_dispersion_21 = _causal_ema_nd(z_disp_daily, cnt_innov_t >= 2.0, span=21.0)
    macro_dispersion_21 = torch.where(any_innov_t, macro_dispersion_21, 0.0)

    # 2) Compresión a pilares (6 pilares económicos en `encoded_v31_420`, o bloques en raw)
    if use_v31 and n_ind == 140:
        sign_mat, mask_mat, anchor1, anchor2 = _get_macro_v31_static_tensors(device, compute_dtype)
        w_eff = torch.where(innov_ok, fresh_tm, 0.0)  # [T, 140]
        p_num = (z_innov * w_eff) @ sign_mat  # [T, 6]
        p_den = w_eff @ mask_mat  # [T, 6]
        pillars = torch.where(p_den > 1e-6, p_num / p_den.clamp_min(1e-6), 0.0)
        k_p = pillars.shape[1]
    else:
        k_p = min(n_ind, 6)
        block_ids = (torch.arange(n_ind, device=device) * k_p) // max(n_ind, 1)
        assign = F.one_hot(block_ids, num_classes=k_p).to(dtype=compute_dtype)  # [M, k_p]
        w_eff = innov_ok.to(compute_dtype)
        p_num = z_innov @ assign
        p_den = w_eff @ assign
        pillars = torch.where(p_den > 1e-6, p_num / p_den.clamp_min(1e-6), 0.0)
        anchor1, anchor2 = _get_raw_macro_anchors(device, k_p)

    # 3) PCA rodante causal `[t-63..t-1]` sobre matriz de correlación contraída Marchenko-Pastur
    #    (iteración de potencia sin cuSOLVER -> 0 syncs, inmune a ruido finito y rotación espuria)
    p_f64 = pillars.to(torch.float64)
    cnt_cov, _, cov_lag = _rolling_cov_lagged(p_f64, any_innov_t, window=63)
    eye_k = torch.eye(k_p, dtype=torch.float64, device=device).unsqueeze(0)
    diag_p = torch.diagonal(cov_lag, dim1=-2, dim2=-1).clamp_min(0.0)  # [T, k_p]
    inv_sd_p = torch.where(diag_p > 1e-8, torch.rsqrt(diag_p.clamp_min(1e-8)), 0.0)
    corr_lag = cov_lag * (inv_sd_p.unsqueeze(-1) * inv_sd_p.unsqueeze(-2))
    prior_mat = (
        1.60 * (anchor1.unsqueeze(-1) * anchor1.unsqueeze(-2))
        + 1.00 * (anchor2.unsqueeze(-1) * anchor2.unsqueeze(-2))
    ).unsqueeze(0)
    cov_reg = 0.50 * corr_lag + 0.50 * prior_mat + 1e-4 * eye_k  # [T, k_p, k_p]

    v1 = anchor1.unsqueeze(0).expand(t_len, k_p).clone()
    for _ in range(12):
        v1_next = torch.bmm(cov_reg, v1.unsqueeze(-1)).squeeze(-1)
        v1_norm = torch.linalg.vector_norm(v1_next, dim=-1, keepdim=True).clamp_min(1e-12)
        v1 = v1_next / v1_norm
    s1 = torch.where((v1 * anchor1.unsqueeze(0)).sum(dim=-1, keepdim=True) >= 0.0, 1.0, -1.0)
    v1 = v1 * s1

    has_pca = cnt_cov >= 15.0
    v1_eff = torch.where(has_pca.unsqueeze(-1), v1, anchor1.unsqueeze(0))
    pc1_f64 = (v1_eff * p_f64).sum(dim=-1)
    macro_innov_pc1 = torch.where(any_innov_t, pc1_f64.to(compute_dtype), 0.0)

    if k_p >= 2:
        lam1 = (v1 * torch.bmm(cov_reg, v1.unsqueeze(-1)).squeeze(-1)).sum(dim=-1)
        cov_def = cov_reg - lam1.view(t_len, 1, 1) * (v1.unsqueeze(-1) * v1.unsqueeze(-2))
        v2 = anchor2.unsqueeze(0).expand(t_len, k_p).clone()
        v2 = v2 - (v2 * v1).sum(dim=-1, keepdim=True) * v1
        v2 = v2 / torch.linalg.vector_norm(v2, dim=-1, keepdim=True).clamp_min(1e-12)
        for _ in range(12):
            v2_next = torch.bmm(cov_def, v2.unsqueeze(-1)).squeeze(-1)
            v2_next = v2_next - (v2_next * v1).sum(dim=-1, keepdim=True) * v1
            v2_norm = torch.linalg.vector_norm(v2_next, dim=-1, keepdim=True).clamp_min(1e-12)
            v2 = v2_next / v2_norm
        s2 = torch.where((v2 * anchor2.unsqueeze(0)).sum(dim=-1, keepdim=True) >= 0.0, 1.0, -1.0)
        v2 = v2 * s2
        v2_eff = torch.where(has_pca.unsqueeze(-1), v2, anchor2.unsqueeze(0))
        pc2_f64 = (v2_eff * p_f64).sum(dim=-1)
        macro_innov_pc2 = torch.where(any_innov_t, pc2_f64.to(compute_dtype), 0.0)
    else:
        macro_innov_pc2 = torch.zeros_like(macro_innov_pc1)

    # 4) Compuesto causal de estrés macro (direccional en `encoded_v31_420` + magnitud RMS)
    rms_z = torch.where(
        any_innov_t,
        torch.sqrt(((z_innov * z_innov).sum(dim=1) / cnt_innov_t.clamp_min(1.0)).clamp_min(0.0)),
        float("nan"),
    ).unsqueeze(1)
    rms_m, rms_c = _rolling_mean_cnt_2d(rms_z, 63)
    rms_v, _ = _rolling_var_anchored_2d(rms_z, 63)
    rms_m_lag = _lag1(rms_m, fill=0.0).squeeze(1)
    rms_c_lag = _lag1(rms_c, fill=0.0).squeeze(1)
    rms_sd_lag = torch.sqrt(_lag1(rms_v, fill=0.0).clamp_min(0.0)).squeeze(1)
    rms_val = torch.where(any_innov_t, rms_z.squeeze(1), 0.0)
    rms_zscore = torch.where(
        any_innov_t & (rms_c_lag >= 10.0) & (rms_sd_lag > 1e-6),
        ((rms_val - rms_m_lag) / rms_sd_lag.clamp_min(1e-6)).clamp(-4.0, 4.0),
        torch.where(any_innov_t, (rms_val - 1.0).clamp(-4.0, 4.0), 0.0),
    )

    if use_v31 and n_ind == 140:
        # Estrés: +fin_stress_credit - yield_curve + fx_dollar + |rates| + |infl| - growth + rms_z
        dir_stress = (
            0.35 * pillars[:, 2]
            - 0.25 * pillars[:, 1]
            + 0.25 * pillars[:, 4]
            + 0.20 * pillars[:, 0].abs()
            + 0.10 * pillars[:, 3].abs()
            - 0.15 * pillars[:, 5]
            + 0.35 * rms_zscore
        )
        macro_stress_composite = torch.where(any_innov_t, dir_stress, 0.0)
    else:
        macro_stress_composite = torch.where(any_innov_t, rms_zscore, 0.0)

    return (
        macro_innov_pc1,
        macro_innov_pc2,
        macro_dispersion_21,
        macro_stress_composite,
        any_innov_t,
        asset_mask_tn,
    )


def compute_market_indices_and_macro_sota_torch(
    open_: torch.Tensor | np.ndarray,
    high: torch.Tensor | np.ndarray,
    low: torch.Tensor | np.ndarray,
    close: torch.Tensor | np.ndarray,
    volume: torch.Tensor | np.ndarray,
    *,
    macro: torch.Tensor | np.ndarray | None = None,
    macro_mask: torch.Tensor | np.ndarray | None = None,
    macro_layout: Literal["auto", "encoded_v31_420", "raw"] = "auto",
    staleness_halflife_days: float = 63.0,
    compute_dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Calcula índices institucionales `[T, 18]`, factores `[T, 5]` y rasgos `[T, N, 11]`.

    Strictly causal bit a bit (`diff == 0.0`), 0 host-syncs GPU, precisión maestra FP32/FP64
    bajo `_autocast_disabled`, invariante a divisa FX y seguro ante `macro=None`/NaN y halts CN.
    """
    in_is_f64 = (isinstance(close, torch.Tensor) and close.dtype == torch.float64) or (
        isinstance(close, np.ndarray) and close.dtype == np.float64
    )
    prep_dtype = torch.float64 if (in_is_f64 or compute_dtype == torch.float64) else torch.float32
    dev = close.device if isinstance(close, torch.Tensor) else torch.device("cpu")

    with _autocast_disabled(dev):
        o_raw = _as_float_tensor(open_, dtype=prep_dtype, device=dev)
        h_raw = _as_float_tensor(high, dtype=prep_dtype, device=dev)
        l_raw = _as_float_tensor(low, dtype=prep_dtype, device=dev)
        c_raw = _as_float_tensor(close, dtype=prep_dtype, device=dev)
        v_raw = _as_float_tensor(volume, dtype=prep_dtype, device=dev)

        if not (o_raw.shape == h_raw.shape == l_raw.shape == c_raw.shape == v_raw.shape):
            raise ValueError("open_, high, low, close y volume deben tener la misma forma")
        if o_raw.ndim not in (1, 2):
            raise ValueError(f"Se esperan tensores 1D [T] o 2D [T, N]; recibido ndim={o_raw.ndim}")

        squeeze_1d = o_raw.ndim == 1
        if squeeze_1d:
            o_raw = o_raw.unsqueeze(1)
            h_raw = h_raw.unsqueeze(1)
            l_raw = l_raw.unsqueeze(1)
            c_raw = c_raw.unsqueeze(1)
            v_raw = v_raw.unsqueeze(1)

        t_len, n_assets = o_raw.shape
        if t_len == 0 or n_assets == 0:
            a_shape = (
                (t_len, ASSET_INDEX_MACRO_DIM)
                if squeeze_1d
                else (t_len, n_assets, ASSET_INDEX_MACRO_DIM)
            )
            return dict(
                market_regime_indices=torch.zeros(
                    (t_len, MARKET_REGIME_INDEX_DIM), dtype=torch.float32, device=dev
                ),
                asset_index_macro_features=torch.zeros(a_shape, dtype=torch.float32, device=dev),
                factor_returns=torch.zeros(
                    (t_len, FACTOR_PORTFOLIO_DIM), dtype=torch.float32, device=dev
                ),
                market_ew_returns=torch.zeros((t_len,), dtype=torch.float32, device=dev),
            )

        return _compute_market_indices_core(
            o_raw,
            h_raw,
            l_raw,
            c_raw,
            v_raw,
            squeeze_1d=squeeze_1d,
            macro=macro,
            macro_mask=macro_mask,
            macro_layout=macro_layout,
            staleness_halflife_days=staleness_halflife_days,
            compute_dtype=_stable_calc_dtype(compute_dtype),
        )


def _compute_market_indices_core(
    o_raw: torch.Tensor,
    h_raw: torch.Tensor,
    l_raw: torch.Tensor,
    c_raw: torch.Tensor,
    v_raw: torch.Tensor,
    *,
    squeeze_1d: bool,
    macro: torch.Tensor | np.ndarray | None,
    macro_mask: torch.Tensor | np.ndarray | None,
    macro_layout: Literal["auto", "encoded_v31_420", "raw"],
    staleness_halflife_days: float,
    compute_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    t_len, n_assets = o_raw.shape
    dev = o_raw.device

    # -------------------------------------------------------------------------
    # 0) Higiene OHLCV, máscara de activos operados y halts / límites CN
    # -------------------------------------------------------------------------
    px_ok = torch.isfinite(c_raw) & (c_raw > 0.0)
    traded = px_ok & torch.isfinite(v_raw) & (v_raw > 0.0)
    c = torch.where(px_ok, c_raw, 1e-12).clamp_min(1e-12)
    v = torch.where(traded, v_raw, 0.0).clamp_min(0.0)

    log_c_p = torch.log(c)
    log_c = log_c_p.to(dtype=compute_dtype)
    c_calc = c.to(dtype=compute_dtype)
    dvol = (c * v).clamp_min(0.0).to(dtype=compute_dtype)

    ret_cc = torch.full((t_len, n_assets), float("nan"), dtype=compute_dtype, device=dev)
    if t_len > 1:
        pair_traded = traded[1:] & traded[:-1]
        r_diff = (log_c_p[1:] - log_c_p[:-1]).to(dtype=compute_dtype)
        ret_cc[1:] = torch.where(pair_traded, r_diff, float("nan"))
    r_fin = torch.isfinite(ret_cc)
    r_safe = torch.where(r_fin, ret_cc, 0.0)
    abs_r = r_safe.abs()

    def _cum_ret_horizon(h_win: int) -> torch.Tensor:
        out_r = torch.full((t_len, n_assets), float("nan"), dtype=compute_dtype, device=dev)
        if t_len > h_win:
            ok_h = traded[h_win:] & traded[:-h_win]
            out_r[h_win:] = torch.where(
                ok_h, (log_c_p[h_win:] - log_c_p[:-h_win]).to(compute_dtype), float("nan")
            )
        return out_r

    ret_5 = _cum_ret_horizon(5)
    ret_21 = _cum_ret_horizon(21)

    # Liquidez rodante causal (FX-invariante vía log puro y raíz cuadrada proporcional)
    dvol_nan = torch.where(traded, dvol, float("nan"))
    dvol_m21, dvol_cnt21 = _rolling_mean_cnt_2d(dvol_nan, 21)
    log_dvol = torch.where(traded, torch.log(dvol.clamp_min(1e-12)), float("nan"))
    ldv_m21, ldv_cnt21 = _rolling_mean_cnt_2d(log_dvol, 21)

    # Volatilidad realizada 21d causal por activo (para BAB / low-vol)
    rv_cnt21 = _rolling_sum_2d(r_fin.to(compute_dtype), 21, assume_finite=True)
    rv_21 = torch.sqrt(
        (
            _rolling_sum_2d(torch.where(r_fin, r_safe * r_safe, 0.0), 21, assume_finite=True)
            / rv_cnt21.clamp_min(1.0)
        ).clamp_min(0.0)
    )

    # -------------------------------------------------------------------------
    # 1) Portafolios Factoriales Estrictamente Causales (`t-1` pesos x `t` ret)
    # -------------------------------------------------------------------------
    # Equal-weighted market return `[T]`
    cnt_r_t = r_fin.sum(dim=1)
    has_r_t = cnt_r_t > 0
    r_ew = torch.where(has_r_t, r_safe.sum(dim=1) / cnt_r_t.clamp_min(1).to(compute_dtype), 0.0)

    # fac_mkt_vw: ponderado por `sqrt(dvol_m21[t-1])` (100% FX-invariante y diversificado!)
    dvol_m21_lag = _lag1(dvol_m21, fill=0.0)
    dvol_ok_lag = _lag1((dvol_cnt21 >= 1.0).to(compute_dtype), fill=0.0) > 0.5
    w_vw_raw = torch.where(
        r_fin & dvol_ok_lag & (dvol_m21_lag > 0.0),
        torch.sqrt(dvol_m21_lag.clamp_min(0.0)),
        0.0,
    )
    w_vw_sum = w_vw_raw.sum(dim=1)
    has_vw = has_r_t & (w_vw_sum > 1e-12)
    fac_mkt_vw = torch.where(
        has_vw, (w_vw_raw * r_safe).sum(dim=1) / w_vw_sum.clamp_min(1e-12), r_ew
    )

    # fac_liq_smb: Small-Minus-Big (long baja liquidez, short alta liquidez en t-1)
    ldv_m21_lag = _lag1(ldv_m21, fill=0.0)
    ldv_ok_lag = _lag1((ldv_cnt21 >= 3.0).to(compute_dtype), fill=0.0) > 0.5
    fac_liq_smb = _build_long_short_factor(
        -ldv_m21_lag, ret_cc, ldv_ok_lag, r_fin, compute_dtype=compute_dtype
    )

    # fac_mom_umd: Momentum 63d con skip de 5d (`mean(ret_cc[t-68..t-6])`, continuo desde t>=21!)
    rm_63, rm_cnt63 = _rolling_mean_cnt_2d(ret_cc, 63)
    mom_sig_lag = _lag_k(rm_63, 6, fill=0.0)
    mom_ok_lag = _lag_k((rm_cnt63 >= 15.0).to(compute_dtype), 6, fill=0.0) > 0.5
    fac_mom_umd = _build_long_short_factor(
        mom_sig_lag, ret_cc, mom_ok_lag, r_fin, compute_dtype=compute_dtype
    )

    # fac_rev_st: Reversión de corto plazo 5d (long perdedores 5d en t-1, short ganadores)
    ret_5_lag = _lag1(ret_5, fill=float("nan"))
    rev_ok_lag = torch.isfinite(ret_5_lag)
    fac_rev_st = _build_long_short_factor(
        -torch.where(rev_ok_lag, ret_5_lag, 0.0),
        ret_cc,
        rev_ok_lag,
        r_fin,
        compute_dtype=compute_dtype,
    )

    # fac_bab_lowvol: Low-Vol / BAB (long baja vol 21d en t-1, short alta vol 21d)
    rv_21_lag = _lag1(rv_21, fill=0.0)
    rv_ok_lag = _lag1((rv_cnt21 >= 5.0).to(compute_dtype), fill=0.0) > 0.5
    fac_bab_lowvol = _build_long_short_factor(
        -rv_21_lag, ret_cc, rv_ok_lag, r_fin, compute_dtype=compute_dtype
    )

    factor_returns = torch.stack(
        [fac_mkt_vw, fac_liq_smb, fac_mom_umd, fac_rev_st, fac_bab_lowvol], dim=-1
    )  # [T, 5]

    # -------------------------------------------------------------------------
    # 2) Índices Institucionales a Nivel de Mercado (`market_regime_indices [T, 18]`)
    # -------------------------------------------------------------------------
    # 2a) Breadth & Internals: AD EMA21, McClellan Osc, % > MA50, % > MA200, NH-NL 252d
    n_up = (r_fin & (ret_cc > 0.0)).to(compute_dtype).sum(dim=1)
    n_dn = (r_fin & (ret_cc < 0.0)).to(compute_dtype).sum(dim=1)
    ad_daily = torch.where(has_r_t, (n_up - n_dn) / cnt_r_t.clamp_min(1).to(compute_dtype), 0.0)
    mkt_ad_ema21 = _causal_ema_nd(ad_daily, has_r_t, span=21.0)
    mkt_mcclellan_osc = _causal_ema_nd(ad_daily, has_r_t, span=19.0) - _causal_ema_nd(
        ad_daily, has_r_t, span=39.0
    )

    c_traded_nan = torch.where(traded, c_calc, float("nan"))
    sma_50, cnt_50 = _rolling_mean_cnt_2d(c_traded_nan, 50)
    sma_200, cnt_200 = _rolling_mean_cnt_2d(c_traded_nan, 200)
    ok_ma50 = traded & (cnt_50 >= 10.0)
    ok_ma200 = traded & (cnt_200 >= 20.0)
    n_ma50 = ok_ma50.to(compute_dtype).sum(dim=1)
    n_ma200 = ok_ma200.to(compute_dtype).sum(dim=1)
    mkt_pct_above_ma50 = torch.where(
        n_ma50 > 0.0,
        (ok_ma50 & (c_calc > sma_50)).to(compute_dtype).sum(dim=1) / n_ma50.clamp_min(1.0),
        0.5,
    )
    mkt_pct_above_ma200 = torch.where(
        n_ma200 > 0.0,
        (ok_ma200 & (c_calc > sma_200)).to(compute_dtype).sum(dim=1) / n_ma200.clamp_min(1.0),
        0.5,
    )

    prev_max_251, prev_min_251, prev_cnt_251 = _rolling_prev_extrema_2d(log_c, traded, window=251)
    elig_nhnl = traded & (prev_cnt_251 >= 20.0)
    n_elig_nhnl = elig_nhnl.to(compute_dtype).sum(dim=1)
    is_nh = elig_nhnl & (log_c >= prev_max_251 - 1e-7)
    is_nl = elig_nhnl & (log_c <= prev_min_251 + 1e-7)
    mkt_nh_nl_spread_252 = torch.where(
        n_elig_nhnl > 0.0,
        (is_nh.to(compute_dtype).sum(dim=1) - is_nl.to(compute_dtype).sum(dim=1))
        / n_elig_nhnl.clamp_min(1.0),
        0.0,
    )

    # 2b) Concentration, Dispersion, Skewness & Synthetic VIX + Term Structure
    ew_vs_vw_daily = (r_ew - fac_mkt_vw).unsqueeze(1)
    mkt_ew_vs_vw_spread_5 = _rolling_sum_2d(ew_vs_vw_daily, 5, assume_finite=True).squeeze(1)
    mkt_ew_vs_vw_spread_21 = _rolling_sum_2d(ew_vs_vw_daily, 21, assume_finite=True).squeeze(1)

    # Dispersión y asimetría transversal winsorizadas (inmunes a colas curtosis=175!)
    cs_med, cs_cnt = _masked_median_lastdim(r_safe, r_fin)
    abs_dev = torch.where(r_fin, (r_safe - cs_med.unsqueeze(1)).abs(), 0.0)
    cs_mad, _ = _masked_median_lastdim(abs_dev, r_fin)
    mad_sig = 1.4826 * cs_mad
    raw_mean = torch.where(r_fin, r_safe, 0.0).sum(dim=1) / cs_cnt.clamp_min(1.0)
    raw_diff = torch.where(r_fin, r_safe - raw_mean.unsqueeze(1), 0.0)
    raw_std = torch.sqrt(((raw_diff * raw_diff).sum(dim=1) / cs_cnt.clamp_min(1.0)).clamp_min(0.0))
    rob_scale = torch.where(mad_sig > 1e-6, mad_sig, raw_std).clamp_min(1e-6)
    lo_w = (cs_med - 4.0 * rob_scale).unsqueeze(1)
    hi_w = (cs_med + 4.0 * rob_scale).unsqueeze(1)
    r_win = torch.where(r_fin, r_safe.clamp(min=lo_w, max=hi_w), 0.0)
    win_mean = r_win.sum(dim=1) / cs_cnt.clamp_min(1.0)
    win_diff = torch.where(r_fin, r_win - win_mean.unsqueeze(1), 0.0)
    win_std = torch.sqrt(((win_diff * win_diff).sum(dim=1) / cs_cnt.clamp_min(1.0)).clamp_min(0.0))
    cs_disp_daily = torch.where(cs_cnt >= 3.0, win_std * math.sqrt(252.0), 0.0)
    mkt_cs_dispersion_21 = _causal_ema_nd(cs_disp_daily, cs_cnt >= 3.0, span=21.0)

    z_skew = torch.where(
        r_fin & (win_std.unsqueeze(1) > 1e-8),
        (win_diff / win_std.unsqueeze(1).clamp_min(1e-8)).clamp(-4.0, 4.0),
        0.0,
    )
    cs_skew_daily = torch.where(
        (cs_cnt >= 5.0) & (win_std > 1e-8),
        ((z_skew * z_skew * z_skew).sum(dim=1) / cs_cnt.clamp_min(1.0)).clamp(-4.0, 4.0),
        0.0,
    )
    mkt_cs_skew_21 = _causal_ema_nd(cs_skew_daily, (cs_cnt >= 5.0) & (win_std > 1e-8), span=21.0)

    # Synthetic VIX (21d vol anualizada de fac_mkt_vw) + pendiente term structure 5d/63d
    vw_nan = torch.where(has_r_t, fac_mkt_vw, float("nan")).unsqueeze(1)
    vw_var_5, vw_cnt_5 = _rolling_var_anchored_2d(vw_nan, 5)
    vw_var_21, vw_cnt_21 = _rolling_var_anchored_2d(vw_nan, 21)
    vw_var_63, vw_cnt_63 = _rolling_var_anchored_2d(vw_nan, 63)
    vol_5 = torch.sqrt(vw_var_5.squeeze(1).clamp_min(0.0) * 252.0)
    vol_21 = torch.sqrt(vw_var_21.squeeze(1).clamp_min(0.0) * 252.0)
    vol_63 = torch.sqrt(vw_var_63.squeeze(1).clamp_min(0.0) * 252.0)
    mkt_synth_vix_21 = torch.where(vw_cnt_21.squeeze(1) >= 5.0, vol_21, 0.0)
    mkt_vol_term_slope_5_63 = torch.where(
        (vw_cnt_5.squeeze(1) >= 3.0) & (vw_cnt_63.squeeze(1) >= 15.0),
        torch.log((vol_5 + 1e-3) / (vol_63 + 1e-3)),
        0.0,
    )

    # 2c) Systemic Turbulence (Kritzman-Li 2010), Liquidity Stress & CN Halt/Limit Stress
    fac_f64 = factor_returns.to(torch.float64)
    t_cnt_lag, t_mu_lag, t_cov_lag = _rolling_cov_lagged(fac_f64, has_r_t, window=63)
    eye_5 = torch.eye(FACTOR_PORTFOLIO_DIM, dtype=torch.float64, device=dev).unsqueeze(0)
    diag_5 = torch.diagonal(t_cov_lag, dim1=-2, dim2=-1)
    cov_reg_5 = 0.85 * t_cov_lag + 0.15 * torch.diag_embed(diag_5) + 1e-6 * eye_5
    delta_f = fac_f64 - t_mu_lag  # [T, 5]
    # `solve_ex(..., check_errors=False)` evita sincronización host cuSOLVER en CUDA
    sol_f = torch.linalg.solve_ex(cov_reg_5, delta_f.unsqueeze(-1), check_errors=False)[0].squeeze(
        -1
    )
    mahal_sq = ((delta_f * sol_f).sum(dim=-1) / float(FACTOR_PORTFOLIO_DIM)).clamp_min(0.0)
    mkt_turbulence_63 = torch.where(
        has_r_t & (t_cnt_lag >= 15.0),
        torch.log1p(mahal_sq).to(compute_dtype),
        0.0,
    )

    # Estrés de liquidez transversal Amihud logarítmico (100% estacionario e invariante a FX!)
    log_amihud = torch.where(
        traded & r_fin,
        torch.log(abs_r + 1e-6) - torch.log(dvol.clamp_min(1e-12)),
        0.0,
    )
    cs_amihud, cs_am_cnt = _masked_median_lastdim(log_amihud, traded & r_fin)
    am_ok = cs_am_cnt >= 2.0
    am_ema21 = _causal_ema_nd(cs_amihud, am_ok, span=21.0)
    am_ema63 = _causal_ema_nd(cs_amihud, am_ok, span=63.0)
    am_nan = torch.where(am_ok, cs_amihud, float("nan")).unsqueeze(1)
    am_var63, am_cnt63 = _rolling_var_anchored_2d(am_nan, 63)
    am_std63 = torch.sqrt(am_var63.squeeze(1).clamp_min(0.0))
    mkt_liquidity_stress_21 = torch.where(
        am_ok & (am_cnt63.squeeze(1) >= 15.0) & (am_std63 > 1e-6),
        (am_ema21 - am_ema63) / am_std63.clamp_min(1e-6),
        0.0,
    )

    # Estrés de suspensiones (`volume <= 0` tras cotizar) + límites `±9.5%` (enorme señal en CN!)
    traded_i = traded.to(torch.int64)
    has_traded_before = (torch.cumsum(traded_i, dim=0) - traded_i) > 0
    is_halt = (~traded) & (px_ok | has_traded_before)
    ret_simple = torch.where(r_fin, torch.expm1(r_safe.clamp(-10.0, 10.0)), 0.0)
    is_limit_hit = traded & r_fin & (ret_simple.abs() >= 0.095)
    active_univ = traded | is_halt
    n_active_univ = active_univ.to(compute_dtype).sum(dim=1)
    halt_lim_frac = torch.where(
        n_active_univ > 0.0,
        (is_halt | is_limit_hit).to(compute_dtype).sum(dim=1) / n_active_univ.clamp_min(1.0),
        0.0,
    )
    hl_mean5, _ = _rolling_mean_cnt_2d(
        torch.where(n_active_univ > 0.0, halt_lim_frac, float("nan")).unsqueeze(1), 5
    )
    mkt_halt_limit_stress_5 = hl_mean5.squeeze(1)

    # 2d) Compuestos Macro PIT Causales
    staleness_scale = math.log1p(max(staleness_halflife_days, 1.0)) / math.log(2.0)
    (
        macro_innov_pc1,
        macro_innov_pc2,
        macro_dispersion_21,
        macro_stress_composite,
        macro_sess_ok,
        macro_asset_mask_tn,
    ) = _compute_pit_macro_composites(
        macro,
        macro_mask=macro_mask,
        t_len=t_len,
        n_assets=n_assets,
        macro_layout=macro_layout,
        staleness_scale=staleness_scale,
        device=dev,
        compute_dtype=compute_dtype,
    )

    # -------------------------------------------------------------------------
    # 3) Señales Transversales Diferenciadas por Activo (`[T, N, 11]`)
    # -------------------------------------------------------------------------
    # 3a) Retorno relativo al índice ponderado por liquidez (`5d` y `21d`)
    vw_ret_5 = _rolling_sum_2d(fac_mkt_vw.unsqueeze(1), 5, assume_finite=True)
    vw_ret_21 = _rolling_sum_2d(fac_mkt_vw.unsqueeze(1), 21, assume_finite=True)
    stock_rel_to_vw_5 = torch.where(traded & torch.isfinite(ret_5), ret_5 - vw_ret_5, 0.0)
    stock_rel_to_vw_21 = torch.where(traded & torch.isfinite(ret_21), ret_21 - vw_ret_21, 0.0)

    # 3b) Betas factoriales rezagadas 63d + Factor Momentum (Ehsani & Linnainmaa 2022)
    t_ge_1 = (torch.arange(t_len, device=dev) >= 1).unsqueeze(1)
    valid_reg = t_ge_1 & r_fin & has_r_t.unsqueeze(1)

    # Beta de mercado BAB (k=0) con contracción Vasicek suave hacia 1.0, rezagada t-1
    beta_mkt_raw = _rolling_beta_2d(
        fac_mkt_vw,
        r_safe,
        valid_reg,
        window=63,
        min_obs=15,
        vasicek_prior=(1.0, 10.0),
    )
    stock_bab_beta_63 = torch.where(traded, _lag1(beta_mkt_raw, fill=0.0), 0.0)

    # Downside beta 63d (condicionado a `fac_mkt_vw < 0`), rezagada t-1
    db_raw = _rolling_downside_beta_2d(fac_mkt_vw.unsqueeze(1), ret_cc, valid_reg, window=63)
    stock_downside_beta_63 = torch.where(traded, _lag1(db_raw, fill=0.0), 0.0)

    # Momento de los 5 portafolios factoriales (vol-escalado causal 21d) dot beta_{i, k, t-1}
    fmom_acc = torch.zeros((t_len, n_assets), dtype=compute_dtype, device=dev)
    for k_i in range(FACTOR_PORTFOLIO_DIM):
        fk = factor_returns[:, k_i]
        fk_ok = has_r_t & (fk.abs() > 0.0) if k_i > 0 else has_r_t
        fk_ema21 = _causal_ema_nd(fk, fk_ok, span=21.0)
        fk_var63, fk_cnt63 = _rolling_var_anchored_2d(
            torch.where(fk_ok, fk, float("nan")).unsqueeze(1), 63
        )
        fk_sd63 = torch.sqrt(fk_var63.squeeze(1).clamp_min(0.0))
        fmom_k = torch.where(
            fk_ok & (fk_cnt63.squeeze(1) >= 10.0) & (fk_sd63 > 1e-6),
            (fk_ema21 / fk_sd63.clamp_min(1e-6)).clamp(-4.0, 4.0),
            0.0,
        )
        if k_i == 0:
            beta_k_lag = stock_bab_beta_63
        else:
            bk_raw = _rolling_beta_2d(
                fk, r_safe, t_ge_1 & r_fin & fk_ok.unsqueeze(1), window=63, min_obs=15
            )
            beta_k_lag = torch.where(traded, _lag1(bk_raw, fill=0.0), 0.0)
        fmom_acc = fmom_acc + beta_k_lag.clamp(-4.0, 4.0) * fmom_k.unsqueeze(1)
    stock_factor_mom_score_21 = torch.where(
        traded, fmom_acc / math.sqrt(float(FACTOR_PORTFOLIO_DIM)), 0.0
    )

    # 3c) Sensibilidad rezagada a innovaciones de turbulencia Kritzman-Li (`Δturbulence`)
    turb_lag1 = _lag1(mkt_turbulence_63, fill=0.0)
    delta_turb = mkt_turbulence_63 - turb_lag1
    turb_ok = has_r_t & (t_cnt_lag >= 16.0)
    turb_beta_raw = _rolling_beta_2d(
        delta_turb,
        r_safe,
        t_ge_1 & r_fin & turb_ok.unsqueeze(1),
        window=63,
        min_obs=15,
        standardize_by_x_std=True,
    )
    stock_turbulence_beta_63 = torch.where(traded, _lag1(turb_beta_raw, fill=0.0) * 100.0, 0.0)

    # 3d) Interacción amplitud de mercado x momentum 21d transversal
    ok_r21 = traded & torch.isfinite(ret_21)
    g_ret_21 = gaussian_rank_features(
        torch.where(ok_r21, ret_21, 0.0).unsqueeze(-1),
        mask=ok_r21.unsqueeze(-1),
        clip_val=LIMIT,
    ).squeeze(-1)
    breadth_regime = ((mkt_pct_above_ma50 - 0.5) * 2.0).unsqueeze(1)
    stock_breadth_x_mom_21 = torch.where(ok_r21, g_ret_21 * breadth_regime, 0.0)

    # 3e) Inclinación liquidez/tamaño x liderazgo EW vs VW 21d
    liq_valid = traded & (ldv_cnt21 >= 3.0)
    liq_pct = _masked_percentile_rank_2d(ldv_m21, liq_valid)
    liq_centered = 2.0 * liq_pct - 1.0
    ew_vw_scaled = (mkt_ew_vs_vw_spread_21 * 20.0).clamp(-4.0, 4.0).unsqueeze(1)
    stock_ew_vw_tilt_21 = torch.where(liq_valid, liq_centered * ew_vw_scaled, 0.0)

    # 3f) Sensibilidad y exposición causal a innovaciones macro PC1 (0.0 si falta macro!)
    mac_asset_ok = traded & macro_asset_mask_tn & macro_sess_ok.unsqueeze(1)
    mac_beta_raw = _rolling_beta_2d(
        macro_innov_pc1,
        r_safe,
        t_ge_1 & r_fin & mac_asset_ok,
        window=63,
        min_obs=15,
        standardize_by_x_std=True,
    )
    stock_macro_innov_beta_63 = torch.where(
        mac_asset_ok, _lag1(mac_beta_raw, fill=0.0) * 100.0, 0.0
    )
    stock_macro_shock_exposure = torch.where(
        mac_asset_ok,
        stock_macro_innov_beta_63.clamp(-4.0, 4.0) * macro_innov_pc1.clamp(-4.0, 4.0).unsqueeze(1),
        0.0,
    )

    # 3g) Interacción estructura temporal de volatilidad x factor defensivo BAB (-beta)
    bab_ok = traded & (stock_bab_beta_63.abs() > 1e-12)
    g_bab_def = -gaussian_rank_features(
        torch.where(bab_ok, stock_bab_beta_63, 0.0).unsqueeze(-1),
        mask=bab_ok.unsqueeze(-1),
        clip_val=LIMIT,
    ).squeeze(-1)
    stock_vol_term_x_bab = torch.where(
        bab_ok, g_bab_def * mkt_vol_term_slope_5_63.clamp(-4.0, 4.0).unsqueeze(1), 0.0
    )

    # -------------------------------------------------------------------------
    # 4) Empaquetado final finito y estrictamente acotado en `[-8, 8]` (FP32)
    # -------------------------------------------------------------------------
    market_regime_indices = torch.stack(
        [
            mkt_ad_ema21,
            mkt_mcclellan_osc,
            mkt_pct_above_ma50,
            mkt_pct_above_ma200,
            mkt_nh_nl_spread_252,
            mkt_ew_vs_vw_spread_5,
            mkt_ew_vs_vw_spread_21,
            mkt_cs_dispersion_21,
            mkt_cs_skew_21,
            mkt_synth_vix_21,
            mkt_vol_term_slope_5_63,
            mkt_turbulence_63,
            mkt_liquidity_stress_21,
            mkt_halt_limit_stress_5,
            macro_innov_pc1,
            macro_innov_pc2,
            macro_dispersion_21,
            macro_stress_composite,
        ],
        dim=-1,
    )
    asset_features = torch.stack(
        [
            stock_rel_to_vw_5,
            stock_rel_to_vw_21,
            stock_factor_mom_score_21,
            stock_bab_beta_63,
            stock_downside_beta_63,
            stock_turbulence_beta_63,
            stock_breadth_x_mom_21,
            stock_ew_vw_tilt_21,
            stock_macro_innov_beta_63,
            stock_macro_shock_exposure,
            stock_vol_term_x_bab,
        ],
        dim=-1,
    )

    def _finalize(t: torch.Tensor) -> torch.Tensor:
        return (
            torch.nan_to_num(t, nan=0.0, posinf=LIMIT, neginf=-LIMIT)
            .clamp(-LIMIT, LIMIT)
            .to(dtype=torch.float32)
        )

    mkt_out = _finalize(market_regime_indices)
    asset_out = _finalize(asset_features)
    fac_out = _finalize(factor_returns)
    ew_out = _finalize(r_ew)
    if squeeze_1d:
        asset_out = asset_out.squeeze(1)

    return dict(
        market_regime_indices=mkt_out,
        asset_index_macro_features=asset_out,
        factor_returns=fac_out,
        market_ew_returns=ew_out,
    )


# =============================================================================
# 4. Auditoría Empírica de Valor Predictivo, Ortogonalidad y Contratos de Rigor
# =============================================================================
def _pearson_corr_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    """Correlación de Pearson en FP64 sobre dos series 1D con posibles NaNs (0 host-sync attrs)."""
    ok = torch.isfinite(x) & torch.isfinite(y)
    n_obs = int(ok.sum())
    if n_obs < 10:
        return 0.0
    xs = torch.where(ok, x.to(torch.float64), 0.0)
    ys = torch.where(ok, y.to(torch.float64), 0.0)
    mx = xs.sum() / float(n_obs)
    my = ys.sum() / float(n_obs)
    dx = torch.where(ok, xs - mx, 0.0)
    dy = torch.where(ok, ys - my, 0.0)
    vx = float((dx * dx).sum())
    vy = float((dy * dy).sum())
    if vx <= 1e-18 or vy <= 1e-18:
        return 0.0
    return float((dx * dy).sum()) / math.sqrt(vx * vy)


def _cross_block_max_abs_corr(
    a_tnf: torch.Tensor,
    b_tng: torch.Tensor,
    a_names: Sequence[str],
    b_names: Sequence[str],
) -> tuple[dict[str, float], dict[str, str]]:
    """Correlación máxima absoluta por rasgo de `A [T,N,F]` contra `B [T,N,G]` en FP64."""
    t_len, n_assets, f_dim = a_tnf.shape
    g_dim = b_tng.shape[-1]
    if t_len == 0 or n_assets == 0 or f_dim == 0 or g_dim == 0:
        return {str(nm): 0.0 for nm in a_names}, {str(nm): "" for nm in a_names}
    n_rows = t_len * n_assets
    a_flat = a_tnf.reshape(n_rows, f_dim).to(torch.float64)
    b_flat = b_tng.reshape(n_rows, g_dim).to(torch.float64)
    a_fin = torch.isfinite(a_flat)
    b_fin = torch.isfinite(b_flat)
    a_c = torch.where(a_fin, a_flat, 0.0)
    b_c = torch.where(b_fin, b_flat, 0.0)
    a_cnt = a_fin.to(torch.float64).sum(dim=0).clamp_min(1.0)
    b_cnt = b_fin.to(torch.float64).sum(dim=0).clamp_min(1.0)
    a_mu = a_c.sum(dim=0, keepdim=True) / a_cnt.unsqueeze(0)
    b_mu = b_c.sum(dim=0, keepdim=True) / b_cnt.unsqueeze(0)
    da = torch.where(a_fin, a_flat - a_mu, 0.0)
    db = torch.where(b_fin, b_flat - b_mu, 0.0)
    va = (da * da).sum(dim=0).clamp_min(0.0)
    vb = (db * db).sum(dim=0).clamp_min(0.0)
    cov = da.T @ db  # [F, G]
    denom = torch.sqrt((va.unsqueeze(1) * vb.unsqueeze(0)).clamp_min(0.0))
    ok = denom > 1e-12
    corr = torch.where(ok, cov / denom.clamp_min(1e-12), 0.0).clamp(-1.0, 1.0)

    max_corr: dict[str, float] = {}
    best_match: dict[str, str] = {}
    for i, anm in enumerate(a_names):
        row_abs = corr[i].abs()
        best_j = int(torch.argmax(row_abs)) if g_dim > 0 else 0
        best_val = float(row_abs[best_j]) if g_dim > 0 else 0.0
        max_corr[anm] = best_val
        best_match[anm] = str(b_names[best_j]) if (g_dim > 0 and best_val > 0.0) else ""
    return max_corr, best_match


def audit_market_indices_macro_value(
    open_: torch.Tensor | np.ndarray,
    high: torch.Tensor | np.ndarray,
    low: torch.Tensor | np.ndarray,
    close: torch.Tensor | np.ndarray,
    volume: torch.Tensor | np.ndarray,
    future_returns: torch.Tensor | np.ndarray,
    *,
    macro: torch.Tensor | np.ndarray | None = None,
    macro_mask: torch.Tensor | np.ndarray | None = None,
    macro_layout: Literal["auto", "encoded_v31_420", "raw"] = "auto",
    peer_exposures: torch.Tensor | np.ndarray | None = None,
    fundamentals: torch.Tensor | np.ndarray | None = None,
    fundamentals_mask: torch.Tensor | np.ndarray | None = None,
    news: torch.Tensor | np.ndarray | None = None,
    news_mask: torch.Tensor | np.ndarray | None = None,
    charts: torch.Tensor | np.ndarray | None = None,
    charts_mask: torch.Tensor | np.ndarray | None = None,
    existing_enriched: torch.Tensor | np.ndarray | None = None,
    precomputed_asset_features: torch.Tensor | np.ndarray | None = None,
    market_label: str = "UNKNOWN",
    min_ic: float = 0.005,
    min_hac_t: float = 1.96,
    redundancy_threshold: float = 0.70,
    leakage_ic_threshold: float = 0.10,
    run_causality_check: bool = True,
    run_bf16_check: bool = True,
    run_macro_null_check: bool = True,
) -> dict[str, Any]:
    """Auditoría empírica completa de valor predictivo, ortogonalidad y garantías Blackwell.

    Clasifica honestamente cada rasgo transversal en:
      - `KEEP (high orthogonal alpha)`
      - `DROP/REDUNDANT` (correlación >= `redundancy_threshold` frente a `ENRICHED_SOTA_NAMES`)
      - `DROP/NEUTRAL` (sin evidencia significativa de IC en este régimen)
    y clasifica todos los descriptores de mercado en `REGIME-CONDITIONING ONLY (FiLM/gating)`
    reportando sus correlaciones predictivas temporales con volatilidad y dispersión futuras.
    """
    from mars_titan.models.mmarrs.blackwell_kernels import audit_zero_host_syncs

    o_t = _as_float_tensor(open_, dtype=torch.float32)
    dev = o_t.device
    h_t = _as_float_tensor(high, dtype=torch.float32, device=dev)
    l_t = _as_float_tensor(low, dtype=torch.float32, device=dev)
    c_t = _as_float_tensor(close, dtype=torch.float32, device=dev)
    v_t = _as_float_tensor(volume, dtype=torch.float32, device=dev)
    y_t = _as_float_tensor(future_returns, dtype=torch.float64, device=dev)
    if y_t.ndim == 3 and y_t.shape[-1] == 1:
        y_t = y_t.squeeze(-1)

    t_len, n_assets = c_t.shape
    out = compute_market_indices_and_macro_sota_torch(
        o_t,
        h_t,
        l_t,
        c_t,
        v_t,
        macro=macro,
        macro_mask=macro_mask,
        macro_layout=macro_layout,
    )
    mkt_reg = out["market_regime_indices"]  # [T, 18]
    asset_feat = (
        _as_float_tensor(precomputed_asset_features, dtype=torch.float32, device=dev)
        if precomputed_asset_features is not None
        else out["asset_index_macro_features"]
    )  # [T, N, 11]
    fac_ret = out["factor_returns"]  # [T, 5]

    all_finite = bool(
        torch.isfinite(mkt_reg).all()
        and torch.isfinite(asset_feat).all()
        and torch.isfinite(fac_ret).all()
    )
    within_bounds = bool(
        float(mkt_reg.abs().max()) <= LIMIT + 1e-5
        and float(asset_feat.abs().max()) <= LIMIT + 1e-5
        and float(fac_ret.abs().max()) <= LIMIT + 1e-5
    ) if (t_len > 0 and n_assets > 0) else True

    # 1) Calidad transversal de los 11 rasgos `[T, N, 11]` + HAC Newey-West t-stat
    base_rep = causal_feature_quality_audit(
        asset_feat,
        y_t,
        feature_names=ASSET_INDEX_MACRO_NAMES,
        min_ic=min_ic,
        leakage_ic_threshold=leakage_ic_threshold,
        redundancy_threshold=redundancy_threshold,
    )
    pearson_tf = _cross_sectional_pearson_series(asset_feat.to(torch.float64), y_t)
    rank_ic_tf = _cross_sectional_rank_ic_series(asset_feat.to(torch.float64), y_t)

    mean_pearson_ic: dict[str, float] = {}
    ic_hac_t_stat: dict[str, float] = {}
    active_nonzero_rate: dict[str, float] = {}
    total_slots = max(t_len * n_assets, 1)
    for i, nm in enumerate(ASSET_INDEX_MACRO_NAMES):
        col_p = pearson_tf[:, i]
        ok_p = torch.isfinite(col_p)
        n_ok_p = int(ok_p.sum())
        mean_pearson_ic[nm] = (
            float(torch.where(ok_p, col_p, 0.0).sum()) / float(max(n_ok_p, 1)) if n_ok_p else 0.0
        )
        ic_hac_t_stat[nm] = _newey_west_tstat_tensor(rank_ic_tf[:, i])
        f_col = asset_feat[:, :, i]
        nz_cnt = float((torch.isfinite(f_col) & (f_col.abs() > 1e-12)).sum())
        active_nonzero_rate[nm] = nz_cnt / float(total_slots)

    max_abs_rank_ic = max((abs(v) for v in base_rep["mean_rank_ic"].values()), default=0.0)
    max_abs_pearson_ic = max((abs(v) for v in mean_pearson_ic.values()), default=0.0)
    leakage_screen_passed = bool(
        max_abs_rank_ic < leakage_ic_threshold
        and max_abs_pearson_ic < leakage_ic_threshold
        and len(base_rep["leakage_suspects"]) == 0
    )

    # 2) Correlación máxima por pares contra los 24 rasgos existentes (`ENRICHED_SOTA_NAMES`)
    if existing_enriched is not None:
        enr_t = _as_float_tensor(existing_enriched, dtype=torch.float32, device=dev)
    else:
        enr_t = enrich_full_dataset_sota_torch(
            o_t,
            h_t,
            l_t,
            c_t,
            v_t,
            peer_exposures=peer_exposures,
            fundamentals=fundamentals,
            fundamentals_mask=fundamentals_mask,
            news=news,
            news_mask=news_mask,
            charts=charts,
            charts_mask=charts_mask,
            macro=macro,
            macro_mask=macro_mask,
        )
    max_corr_vs_enr, best_match_enr = _cross_block_max_abs_corr(
        asset_feat, enr_t, ASSET_INDEX_MACRO_NAMES, ENRICHED_SOTA_NAMES
    )

    # Clasificación honesta por rasgo transversal
    asset_feature_classifications: dict[str, str] = {}
    for nm in ASSET_INDEX_MACRO_NAMES:
        ic_v = abs(base_rep["mean_rank_ic"][nm])
        t_v = abs(ic_hac_t_stat[nm])
        c_enr = max_corr_vs_enr[nm]
        if c_enr >= redundancy_threshold:
            asset_feature_classifications[nm] = "DROP/REDUNDANT"
        elif ic_v >= min_ic and t_v >= min_hac_t:
            asset_feature_classifications[nm] = "KEEP (high orthogonal alpha)"
        else:
            asset_feature_classifications[nm] = "DROP/NEUTRAL"

    # 3) Diagnósticos de régimen temporal (FiLM/gating) sobre `MARKET_REGIME_INDEX_NAMES`
    vw = fac_ret[:, 0].to(torch.float64)
    next_vw = torch.full((t_len,), float("nan"), dtype=torch.float64, device=dev)
    fwd_vol5 = torch.full((t_len,), float("nan"), dtype=torch.float64, device=dev)
    fwd_disp5 = torch.full((t_len,), float("nan"), dtype=torch.float64, device=dev)

    # Dispersión futura transversal de `future_returns` (estrictamente t+1..t+5!)
    y_fin = torch.isfinite(y_t)
    y_cnt = y_fin.to(torch.float64).sum(dim=1).clamp_min(1.0)
    y_mu = torch.where(y_fin, y_t, 0.0).sum(dim=1, keepdim=True) / y_cnt.unsqueeze(1)
    y_diff = torch.where(y_fin, y_t - y_mu, 0.0)
    cs_disp_y = torch.where(
        y_cnt >= 3.0,
        torch.sqrt(((y_diff * y_diff).sum(dim=1) / y_cnt).clamp_min(0.0)),
        float("nan"),
    )
    if t_len > 1:
        next_vw[:-1] = vw[1:]
    if t_len > 5:
        vw_sq_fwd = vw[1:] * vw[1:]
        c_vsq = torch.cat(
            [torch.zeros((1,), dtype=torch.float64, device=dev), torch.cumsum(vw_sq_fwd, dim=0)],
            dim=0,
        )
        fwd_vol5[:-5] = torch.sqrt(((c_vsq[5:] - c_vsq[:-5]) / 5.0).clamp_min(0.0) * 252.0)

        # `y_t[t]` ya representa `r_{t+1}` -> `cs_disp_y[:-1]` alinea `fwd_disp5[t]` a `t+1..t+5`
        d_src = cs_disp_y[:-1]
        d_ok = torch.isfinite(d_src)
        c_dcnt = torch.cat(
            [
                torch.zeros((1,), dtype=torch.float64, device=dev),
                torch.cumsum(d_ok.to(torch.float64), dim=0),
            ],
            dim=0,
        )
        c_dsum = torch.cat(
            [
                torch.zeros((1,), dtype=torch.float64, device=dev),
                torch.cumsum(torch.where(d_ok, d_src, 0.0), dim=0),
            ],
            dim=0,
        )
        dc5 = c_dcnt[5:] - c_dcnt[:-5]
        ds5 = c_dsum[5:] - c_dsum[:-5]
        fwd_disp5[:-5] = torch.where(dc5 >= 3.0, ds5 / dc5.clamp_min(1.0), float("nan"))

    market_regime_stats: dict[str, dict[str, float]] = {}
    market_regime_classifications: dict[str, str] = {}
    for i, nm in enumerate(MARKET_REGIME_INDEX_NAMES):
        col = mkt_reg[:, i].to(torch.float64)
        mu_v = float(col.mean()) if t_len > 0 else 0.0
        sd_v = float(col.std(unbiased=False)) if t_len > 1 else 0.0
        nz_v = float((col.abs() > 1e-12).sum()) / float(max(t_len, 1))
        market_regime_stats[nm] = dict(
            mean=mu_v,
            std=sd_v,
            min=float(col.min()) if t_len > 0 else 0.0,
            max=float(col.max()) if t_len > 0 else 0.0,
            active_nonzero_rate=nz_v,
            ts_corr_next_mkt_ret=_pearson_corr_1d(col, next_vw),
            ts_corr_fwd5_mkt_vol=_pearson_corr_1d(col, fwd_vol5),
            ts_corr_fwd5_cs_disp=_pearson_corr_1d(col, fwd_disp5),
        )
        market_regime_classifications[nm] = "REGIME-CONDITIONING ONLY (FiLM/gating)"

    # 4) Estadísticas de los 5 portafolios factoriales
    factor_stats: dict[str, dict[str, float]] = {}
    for i, fnm in enumerate(FACTOR_PORTFOLIO_NAMES):
        fr = fac_ret[1:, i].to(torch.float64) if t_len > 1 else fac_ret[:, i].to(torch.float64)
        act = fr.abs() > 1e-12
        n_act = int(act.sum())
        if n_act >= 5:
            mu_d = float(fr.mean())
            sd_d = float(fr.std(unbiased=True))
            ann_ret = mu_d * 252.0
            ann_vol = sd_d * math.sqrt(252.0)
            sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
        else:
            ann_ret, ann_vol, sharpe = 0.0, 0.0, 0.0
        factor_stats[fnm] = dict(
            ann_return=ann_ret, ann_vol=ann_vol, sharpe=sharpe, active_days=float(n_act)
        )

    # 5) Causalidad bit-a-bit (`diff == 0.0`)
    causality_check: dict[str, Any] = dict(
        evaluated=False, bit_exact_causal=True, max_abs_diff=0.0, t_cut=0
    )
    if run_causality_check and t_len >= 6:
        t_cut = max(2, (2 * t_len) // 3)
        o_m, h_m, l_m, c_m, v_m = o_t.clone(), h_t.clone(), l_t.clone(), c_t.clone(), v_t.clone()
        c_m[t_cut + 1 :] = c_m[t_cut + 1 :] * 3.5
        h_m[t_cut + 1 :] = h_m[t_cut + 1 :] * 4.0
        l_m[t_cut + 1 :] = l_m[t_cut + 1 :] * 0.3
        v_m[t_cut + 1 :] = v_m[t_cut + 1 :] * 12.0
        mac_mut = None
        if macro is not None:
            mac_mut = _as_float_tensor(macro, dtype=torch.float32, device=dev).clone()
            if mac_mut.ndim >= 1 and mac_mut.shape[0] == t_len:
                if mac_mut.ndim >= 2 and mac_mut.shape[-1] == 420:
                    mac_mut[t_cut + 1 :, ..., :140] = -mac_mut[t_cut + 1 :, ..., :140] * 2.5 - 3.0
                else:
                    mac_mut[t_cut + 1 :] = -mac_mut[t_cut + 1 :] * 2.5 - 3.0
        out_mut = compute_market_indices_and_macro_sota_torch(
            o_m,
            h_m,
            l_m,
            c_m,
            v_m,
            macro=mac_mut,
            macro_mask=macro_mask,
            macro_layout=macro_layout,
        )
        diffs = [
            float((out[k][: t_cut + 1] - out_mut[k][: t_cut + 1]).abs().max())
            for k in ("market_regime_indices", "asset_index_macro_features", "factor_returns")
        ]
        exact_eq = all(
            torch.equal(out[k][: t_cut + 1], out_mut[k][: t_cut + 1])
            for k in ("market_regime_indices", "asset_index_macro_features", "factor_returns")
        )
        causality_check = dict(
            evaluated=True,
            bit_exact_causal=bool(exact_eq),
            max_abs_diff=max(diffs, default=0.0),
            t_cut=int(t_cut),
        )

    # 6) Paridad exacta BF16 autocast (`max_abs_diff == 0.0`)
    bf16_check: dict[str, Any] = dict(evaluated=False, max_abs_diff=0.0, passed=True)
    if run_bf16_check and t_len > 0 and n_assets > 0:
        amp_dev = "cuda" if dev.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=amp_dev, dtype=torch.bfloat16):
            out_bf = compute_market_indices_and_macro_sota_torch(
                o_t,
                h_t,
                l_t,
                c_t,
                v_t,
                macro=macro,
                macro_mask=macro_mask,
                macro_layout=macro_layout,
            )
        max_bf_diff = max(
            float((out_bf[k] - out[k]).abs().max())
            for k in ("market_regime_indices", "asset_index_macro_features", "factor_returns")
        )
        bf16_check = dict(
            evaluated=True,
            max_abs_diff=max_bf_diff,
            passed=bool(max_bf_diff == 0.0),
        )

    # 7) Seguridad `macro=None` y `macro=100%-NaN`
    macro_null_check: dict[str, Any] = dict(evaluated=False, passed=True, max_abs_residual=0.0)
    if run_macro_null_check and t_len > 0 and n_assets > 0:
        m_cols = [
            MARKET_REGIME_INDEX_NAMES.index(n)
            for n in MARKET_REGIME_INDEX_NAMES
            if n.startswith("macro_")
        ]
        a_cols = [
            ASSET_INDEX_MACRO_NAMES.index(n)
            for n in ("stock_macro_innov_beta_63", "stock_macro_shock_exposure")
        ]
        res_none = compute_market_indices_and_macro_sota_torch(o_t, h_t, l_t, c_t, v_t, macro=None)
        nan_mac = torch.full((t_len, 12), float("nan"), dtype=torch.float32, device=dev)
        res_nan = compute_market_indices_and_macro_sota_torch(
            o_t, h_t, l_t, c_t, v_t, macro=nan_mac
        )
        max_res = max(
            float(res_none["market_regime_indices"][:, m_cols].abs().max()),
            float(res_none["asset_index_macro_features"][:, :, a_cols].abs().max()),
            float(res_nan["market_regime_indices"][:, m_cols].abs().max()),
            float(res_nan["asset_index_macro_features"][:, :, a_cols].abs().max()),
        )
        all_fin_null = bool(
            torch.isfinite(res_none["market_regime_indices"]).all()
            and torch.isfinite(res_none["asset_index_macro_features"]).all()
            and torch.isfinite(res_nan["market_regime_indices"]).all()
            and torch.isfinite(res_nan["asset_index_macro_features"]).all()
        )
        macro_null_check = dict(
            evaluated=True,
            passed=bool(all_fin_null and max_res == 0.0),
            max_abs_residual=max_res,
        )

    sync_rep = audit_zero_host_syncs(Path(__file__).resolve())
    zero_host_syncs_clean = bool(sync_rep["total_findings"] == 0 and sync_rep["clean"])

    return dict(
        market_label=market_label,
        n_sessions=t_len,
        n_assets=n_assets,
        all_finite=all_finite,
        within_bounds=within_bounds,
        mean_rank_ic=base_rep["mean_rank_ic"],
        mean_pearson_ic=mean_pearson_ic,
        ic_ir=base_rep["ic_ir"],
        ic_hac_t_stat=ic_hac_t_stat,
        pct_positive_ic=base_rep["pct_positive_ic"],
        feature_stds=base_rep["feature_stds"],
        active_nonzero_rate=active_nonzero_rate,
        max_abs_rank_ic=max_abs_rank_ic,
        max_abs_pearson_ic=max_abs_pearson_ic,
        leakage_suspects=base_rep["leakage_suspects"],
        leakage_screen_passed=leakage_screen_passed,
        max_abs_corr_vs_enriched_sota=max_corr_vs_enr,
        most_correlated_enriched_feature=best_match_enr,
        max_pairwise_corr_within_new=base_rep["max_pairwise_corr"],
        redundant_pairs_within_new=base_rep["redundant_pairs"],
        asset_feature_classifications=asset_feature_classifications,
        market_regime_stats=market_regime_stats,
        market_regime_classifications=market_regime_classifications,
        factor_stats=factor_stats,
        causality_check=causality_check,
        bf16_autocast_check=bf16_check,
        macro_null_safety_check=macro_null_check,
        zero_host_syncs_clean=zero_host_syncs_clean,
        overall_passed=bool(
            all_finite
            and within_bounds
            and leakage_screen_passed
            and causality_check["bit_exact_causal"]
            and bf16_check["passed"]
            and macro_null_check["passed"]
            and zero_host_syncs_clean
        ),
    )


__all__ = [
    "ASSET_INDEX_MACRO_DIM",
    "ASSET_INDEX_MACRO_NAMES",
    "FACTOR_PORTFOLIO_DIM",
    "FACTOR_PORTFOLIO_NAMES",
    "KEEP_ASSET_INDEX_MACRO_DIM",
    "KEEP_ASSET_INDEX_MACRO_INDICES",
    "KEEP_ASSET_INDEX_MACRO_NAMES",
    "LIMIT",
    "MACRO_PILLAR_NAMES",
    "MACRO_V31_140_NAMES",
    "MARKET_REGIME_INDEX_DIM",
    "MARKET_REGIME_INDEX_NAMES",
    "audit_market_indices_macro_value",
    "compute_market_indices_and_macro_sota_torch",
    "select_keep_asset_index_macro_features",
]
