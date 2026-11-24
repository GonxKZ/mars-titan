"""Enriquecimiento SOTA y featurización resiliente a ausencias (`dataset_enrichment_sota`).

Diseñado para garantizar **cero pérdida de precisión, cero NaNs/Infs, cero fuga de gradiente y
causalidad estricta bit a bit** en las 4 familias independientes (`titans_original`,
`holographic_brain`, `mmarrs_titan`, `titans_cognitive_core`) sobre mercados `US` (`2000–2023`)
y `CN` (`2006–2023`), con **cero host-syncs GPU** (`audit_zero_host_syncs == 0`) y seguridad
completa bajo FP32 estricto y BF16 autocast en NVIDIA RTX PRO 6000 Blackwell (`sm_120`):

  1. `MissingnessResilientFeaturizer` + `StructuredModalityDropout`:
     - Anatomía exacta de 3 canales `(sanitized_value, availability_mask, log1p_staleness_decay)`.
     - Imputación Empírica Bayesiana Jerárquica estrictamente causal en sesión `t`:
       mediana de pares RMT/sector -> mediana transversal -> `0.0` (cero `NaN * 0`, cero fuga).
     - Preservador Dual Nivel + Rango Gaussiano con empates (`[robust_winsorized_z,
       masked_gaussian_rank]`, reutilizando `gaussian_rank_features` 0-sync).
     - `StructuredModalityDropout`: activo únicamente en `train()`, preserva siempre `prices` y
       emula los patrones reales de ausencia US/CN (p. ej. 99.2% fundamentales ausentes en CN,
       100% noticias ausentes en US 2000–2008 / CN 2006–2013).
  2. `enrich_full_dataset_sota_torch`:
     - 24 señales institucionales, cross-modales y multi-régimen ortogonales (`ENRICHED_SOTA_NAMES`,
       `ENRICHED_SOTA_DIM = 24`): descomposición Jump vs Volatilidad Continua de Barndorff-Nielsen
       & Shephard (`bv_21`, `jump_var_ratio_21`, `vol_of_vol_63`, `realized_skew_signed_21`), alfa
       peer-residualizada y de liquidez (`peer_resid_mom_{5,21,63}`, `liq_cond_rev_5`,
       `signed_abn_turn_21`, `kaufman_er_21`, `amihud_asym_up_down_21`), contradicciones
       cross-modales estrictamente enmascaradas (`price_vs_news_divergence`,
       `price_vs_fund_quality_gap`, `chart_vs_vol_exhaustion`, `macro_stress_x_downside_beta`),
       compuestos PIT fundamentales con decaimiento de obsolescencia (`pit_fund_quality`,
       `pit_fund_value`, `pit_fund_profitability`) y señales multi-régimen `US 2015` + `CN`
       (`rank_vel_5_21`, `rank_accel_5_21_63`, `peer_rev_5_x_idio_vol_surge`,
       `limit_hit_streak_5`, `post_halt_age_decay`, `avail_surprise_21`).
  3. `audit_dataset_enrichment_quality` & `audit_laboratorio_cache_readonly`:
     - Envuelve `causal_feature_quality_audit` (`institutional_alpha_v5.py`), aplica pantalla
       anti-fuga `max|corr(X_t, y_{t+1})| < 0.10`, calcula `t-stat` HAC (Newey-West), redundancia,
       cobertura de ausencias, causalidad bit a bit y prueba de estrés `50%/90%/100%` ausente con
       verificación de fuga de gradiente exactamente nula en ranuras `NaN`.
"""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .institutional_alpha_v5 import causal_feature_quality_audit
from .market_physics_v4_gpu import gaussian_rank_features

LIMIT: float = 8.0
_L2_CHUNK_TARGET_ELEMS: int = 4_000_000
_PRE_2024_EPOCH_DAY_LIMIT: int = 19723  # 2024-01-01 (estrictamente sellado!)

# Contratos de orden de modalidades observados en `encoded-history-v3.1` vs modelos MMARRS
MODEL_MODALITY_ORDER: tuple[str, ...] = ("prices", "fundamentals", "macro", "news", "charts")
ENCODED_MODALITY_ORDER: tuple[str, ...] = ("prices", "news", "charts", "fundamentals", "macro")
ENCODED_TO_MODEL_MODALITY_INDEX: tuple[int, ...] = (0, 3, 4, 1, 2)

# Perfiles empíricos medidos en US (2000-2023) y CN (2006-2023) estrictamente < 2024
US_EMPIRICAL_MISSINGNESS_PROFILE: dict[str, float] = {
    "p_drop_fundamentals": 0.45,
    "p_drop_macro": 0.05,
    "p_drop_news": 0.55,
    "p_drop_charts": 0.10,
    "p_session_blackout": 0.25,
    "p_feature_block_drop": 0.10,
    "measured_fund_missing_rate": 0.6333,
    "measured_news_missing_rate": 0.8228,
    "measured_zero_vol_halt_rate": 0.0226,
    "measured_return_excess_kurtosis": 175.31,
}

CN_EMPIRICAL_MISSINGNESS_PROFILE: dict[str, float] = {
    "p_drop_fundamentals": 0.75,
    "p_drop_macro": 0.05,
    "p_drop_news": 0.65,
    "p_drop_charts": 0.10,
    "p_session_blackout": 0.35,
    "p_feature_block_drop": 0.15,
    "measured_fund_missing_rate": 0.9922,
    "measured_news_missing_rate": 0.8816,
    "measured_zero_vol_halt_rate": 0.0382,
    "measured_return_excess_kurtosis": 52.68,
}

ENRICHED_SOTA_NAMES: tuple[str, ...] = (
    # 1) Jump vs Continuous Volatility Decomposition (Barndorff-Nielsen & Shephard)
    "bv_21",
    "jump_var_ratio_21",
    "vol_of_vol_63",
    "realized_skew_signed_21",
    # 2) Cross-Sectional Peer-Demeaned & Liquidity-Adjusted Alpha
    "peer_resid_mom_5",
    "peer_resid_mom_21",
    "peer_resid_mom_63",
    "liq_cond_rev_5",
    "signed_abn_turn_21",
    "kaufman_er_21",
    "amihud_asym_up_down_21",
    # 3) Cross-Modal Interaction & Contradiction Features (masked -> 0.0 if either absent)
    "price_vs_news_divergence",
    "price_vs_fund_quality_gap",
    "chart_vs_vol_exhaustion",
    "macro_stress_x_downside_beta",
    # 4) PIT Fundamental Composites (staleness-decayed, masked -> 0.0 if absent)
    "pit_fund_quality",
    "pit_fund_value",
    "pit_fund_profitability",
    # 5) Multi-Regime Rotation, Idiosyncratic Vol-Surge, CN Limit/Halt & MNAR Coverage (#195 #5B)
    "rank_vel_5_21",
    "rank_accel_5_21_63",
    "peer_rev_5_x_idio_vol_surge",
    "limit_hit_streak_5",
    "post_halt_age_decay",
    "avail_surprise_21",
)
ENRICHED_SOTA_DIM: int = len(ENRICHED_SOTA_NAMES)


def _as_float_tensor(
    x: torch.Tensor | np.ndarray,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        target_dev = x.device if device is None else device
        if x.dtype == dtype and x.device == target_dev:
            return x
        return x.to(device=target_dev, dtype=dtype)
    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    t = torch.as_tensor(np.asarray(x, dtype=np_dtype), dtype=dtype)
    return t.to(device=device) if device is not None else t


def _as_bool_tensor(
    x: torch.Tensor | np.ndarray,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x > 0.5 if x.is_floating_point() else x.to(dtype=torch.bool)
        return t.to(device=device) if (device is not None and t.device != device) else t
    t = torch.as_tensor(np.asarray(x, dtype=bool), dtype=torch.bool)
    return t.to(device=device) if device is not None else t


def _stable_calc_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _autocast_disabled(device: torch.device) -> AbstractContextManager[object]:
    """Desactiva autocast BF16/FP16 en cálculos maestros FP32/FP64 (evita fallos en bmm/solve)."""
    if device.type in ("cuda", "cpu"):
        return torch.amp.autocast(device_type=device.type, enabled=False)
    return contextlib.nullcontext()


def _masked_median_lastdim(
    vals: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mediana exacta sobre la última dimensión respetando `valid` (cero syncs, cero NaN*0)."""
    n_last = vals.shape[-1]
    calc_dtype = vals.dtype
    if n_last == 0:
        z = torch.zeros(vals.shape[:-1], dtype=calc_dtype, device=vals.device)
        return z, z

    # Sanear primero antes de ordenar (nunca propagar NaN/Inf al sort o gather)
    clean = torch.where(valid, vals, 0.0)
    sort_in = torch.where(valid, clean, float("inf"))
    sorted_vals = sort_in.sort(dim=-1).values
    cnt_i = valid.sum(dim=-1, keepdim=True)  # [..., 1] int64
    has_obs = cnt_i > 0
    idx_lo = ((cnt_i - 1) // 2).clamp(0, n_last - 1)
    idx_hi = (cnt_i // 2).clamp(0, n_last - 1)
    med_lo = sorted_vals.gather(-1, idx_lo)
    med_hi = sorted_vals.gather(-1, idx_hi)
    safe_lo = torch.where(has_obs, med_lo, 0.0)
    safe_hi = torch.where(has_obs, med_hi, 0.0)
    med = torch.where(has_obs, 0.5 * (safe_lo + safe_hi), 0.0).squeeze(-1)
    return med, cnt_i.squeeze(-1).to(dtype=calc_dtype)


def _masked_percentile_rank_2d(vals: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Percentil transversal tie-averaged en `[0, 1]` por sesión `[T, N]` (cero syncs)."""
    t_len, n_assets = vals.shape
    if t_len == 0 or n_assets == 0:
        return torch.zeros_like(vals)
    clean = torch.where(valid, vals, 0.0)
    masked = torch.where(valid, clean, float("inf"))
    ordered = masked.sort(dim=-1).values
    lower = torch.searchsorted(ordered, masked, right=False)
    upper = torch.searchsorted(ordered, masked, right=True)
    rank = 0.5 * (lower + upper + 1).to(vals.dtype)
    count = valid.sum(dim=-1, keepdim=True).to(vals.dtype).clamp_min(1.0)
    pct = ((rank - 0.5) / count).clamp(0.0, 1.0)
    return torch.where(valid, pct, 0.0)


def _rolling_sum_2d(
    x: torch.Tensor, window: int, *, assume_finite: bool = False
) -> torch.Tensor:
    """Suma rodante estrictamente causal O(T*N) sobre el eje temporal 0."""
    t_len, n_assets = x.shape
    if t_len == 0 or window <= 0:
        return torch.zeros_like(x)
    clean = x if assume_finite else torch.where(torch.isfinite(x), x, 0.0)
    zero = torch.zeros((1, n_assets), dtype=x.dtype, device=x.device)
    csum = torch.cat([zero, torch.cumsum(clean, dim=0)], dim=0)
    if window >= t_len:
        return csum[1:]
    out = csum[1:].clone()
    out[window:] = csum[window + 1 :] - csum[1 : t_len - window + 1]
    return out


def _rolling_mean_cnt_2d(x: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Media y conteo rodantes causales sobre entradas finitas `[T, N]`."""
    fin = torch.isfinite(x)
    cnt = _rolling_sum_2d(fin.to(x.dtype), window, assume_finite=True)
    tot = _rolling_sum_2d(torch.where(fin, x, 0.0), window, assume_finite=True)
    mean = torch.where(cnt > 0.0, tot / cnt.clamp_min(1.0), 0.0)
    return mean, cnt


def _rolling_var_anchored_2d(
    x: torch.Tensor,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Varianza rodante causal O(T*N) con anclaje causal por bloques en FP64 (cero deriva)."""
    t_len, n_assets = x.shape
    dev, dtype = x.device, x.dtype
    var_out = torch.zeros((t_len, n_assets), dtype=dtype, device=dev)
    cnt_out = torch.zeros((t_len, n_assets), dtype=dtype, device=dev)
    if t_len == 0 or n_assets == 0 or window <= 1:
        return var_out, cnt_out

    block_t = 256
    zero = torch.zeros((1, n_assets), dtype=torch.float64, device=dev)
    for b0 in range(0, t_len, block_t):
        b1 = min(b0 + block_t, t_len)
        s0 = max(0, b0 - window + 1)
        x_blk = x[s0:b1].to(torch.float64)
        m_blk = torch.isfinite(x_blk)
        ax = torch.where(m_blk[0:1], x_blk[0:1], 0.0)
        xs = torch.where(m_blk, x_blk - ax, 0.0)
        c_cnt = torch.cat([zero, torch.cumsum(m_blk.to(torch.float64), dim=0)], dim=0)
        c_s1 = torch.cat([zero, torch.cumsum(xs, dim=0)], dim=0)
        c_s2 = torch.cat([zero, torch.cumsum(xs * xs, dim=0)], dim=0)
        l_blk = b1 - s0
        hi = torch.arange(b0 - s0 + 1, l_blk + 1, device=dev)
        lo = (hi - window).clamp_min(0)
        cnt = c_cnt[hi] - c_cnt[lo]
        s1 = c_s1[hi] - c_s1[lo]
        s2 = c_s2[hi] - c_s2[lo]
        denom = cnt.clamp_min(1.0)
        css = (s2 - (s1 * s1) / denom).clamp_min(0.0)
        css = torch.where(css <= 1e-12 * s2, 0.0, css)
        v_loc = torch.where(cnt >= 2.0, css / (cnt - 1.0).clamp_min(1.0), 0.0)
        var_out[b0:b1] = v_loc.to(dtype)
        cnt_out[b0:b1] = cnt.to(dtype)
    return var_out, cnt_out


def _rolling_downside_beta_2d(
    r_mkt: torch.Tensor,
    r_asset: torch.Tensor,
    valid: torch.Tensor,
    window: int = 63,
) -> torch.Tensor:
    """Beta bajista causal (`r_mkt < 0`) en una pasada FP64 por bloques anclados O(T*N)."""
    t_len, n_assets = r_asset.shape
    dev, dtype = r_asset.device, r_asset.dtype
    out = torch.zeros((t_len, n_assets), dtype=dtype, device=dev)
    if t_len == 0 or n_assets == 0 or window <= 1:
        return out

    block_t = 256
    zero = torch.zeros((1, n_assets), dtype=torch.float64, device=dev)
    for b0 in range(0, t_len, block_t):
        b1 = min(b0 + block_t, t_len)
        s0 = max(0, b0 - window + 1)
        x_blk = r_mkt[s0:b1].to(torch.float64)
        y_blk = r_asset[s0:b1].to(torch.float64)
        m_dn = valid[s0:b1] & (x_blk < 0.0)
        if x_blk.shape[1] == 1:
            ax = torch.where(
                m_dn[0:1].any(dim=1, keepdim=True) & torch.isfinite(x_blk[0:1]),
                x_blk[0:1],
                0.0,
            )
        else:
            ax = torch.where(m_dn[0:1], x_blk[0:1], 0.0)
        ay = torch.where(m_dn[0:1], y_blk[0:1], 0.0)
        xs = torch.where(m_dn, x_blk - ax, 0.0)
        ys = torch.where(m_dn, y_blk - ay, 0.0)
        dn_f = m_dn.to(torch.float64)
        c_cnt = torch.cat([zero, torch.cumsum(dn_f, dim=0)], dim=0)
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
        beta = torch.where((cnt >= 5.0) & (cx > 1e-12), cxy / (cx + 1e-10), 0.0)
        out[b0:b1] = beta.to(dtype)
    return out


def _causal_affine_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Escaneo afín causal `s[t] = a[t] * s[t-1] + b[t]` en O(log T) sin bucles sobre T."""
    t_len = a.shape[0]
    if t_len <= 1:
        return b.clone()
    cur_a = a.clone()
    cur_b = b.clone()
    step = 1
    while step < t_len:
        a_prev = cur_a[:-step]
        b_prev = cur_b[:-step]
        a_curr = cur_a[step:]
        b_curr = cur_b[step:]
        new_b = a_curr * b_prev + b_curr
        new_a = a_curr * a_prev
        cur_b = torch.cat([cur_b[:step], new_b], dim=0)
        cur_a = torch.cat([cur_a[:step], new_a], dim=0)
        step *= 2
    return cur_b


def _peer_residualize_returns(
    ret: torch.Tensor,
    peer_exposures: torch.Tensor | None,
) -> torch.Tensor:
    """Residualiza retornos `[T, N]` por pares RMT/sector estrictamente en cada sesión `t`."""
    t_len, n_assets = ret.shape
    if t_len == 0 or n_assets == 0:
        return torch.zeros_like(ret)
    ok = torch.isfinite(ret)
    r_safe = torch.where(ok, ret, 0.0)

    if peer_exposures is None or peer_exposures.shape[-1] == 0 or n_assets < 3:
        cs_med, cnt = _masked_median_lastdim(r_safe, ok)
        has_cs = cnt >= 2.0
        return torch.where(ok & has_cs.unsqueeze(-1), r_safe - cs_med.unsqueeze(-1), 0.0)

    # Ridge OLS por lotes en sesión `t`: B_aug = [1, exposures], O(T*N*K^2), 0 syncs
    dtype = _stable_calc_dtype(ret.dtype)
    dev = ret.device
    with _autocast_disabled(dev):
        r_c = r_safe.to(dtype=dtype)
        exp_t = peer_exposures.to(device=dev, dtype=dtype)
        if exp_t.ndim == 2:
            exp_t = exp_t.unsqueeze(0).expand(t_len, n_assets, -1)
        k_dim = exp_t.shape[-1]
        ok_f = ok.to(dtype)
        b_clean = torch.where(torch.isfinite(exp_t) & ok.unsqueeze(-1), exp_t, 0.0)
        b_aug = torch.cat([ok_f.unsqueeze(-1), b_clean], dim=-1)  # [T, N, K+1]
        y_col = r_c.unsqueeze(-1)  # [T, N, 1]

        gram = torch.bmm(b_aug.transpose(1, 2), b_aug)  # [T, K+1, K+1]
        eye = torch.eye(k_dim + 1, device=dev, dtype=dtype).unsqueeze(0)
        cnt_t = ok_f.sum(dim=1).clamp_min(1.0).view(t_len, 1, 1)
        ridge = (1e-3 * (cnt_t / 100.0) + 1e-4) * eye
        rhs = torch.bmm(b_aug.transpose(1, 2), y_col)  # [T, K+1, 1]
        # `solve_ex(..., check_errors=False)` evita sincronización host cuSOLVER en CUDA
        coef = torch.linalg.solve_ex(gram + ridge, rhs, check_errors=False)[0]
        pred = torch.bmm(b_aug, coef).squeeze(-1)
        has_obs = ok_f.sum(dim=1, keepdim=True) >= 2.0
        res = torch.where(ok & has_obs, r_c - pred, 0.0)
    return res.to(dtype=ret.dtype)


def _normalize_optional_mask(
    mask: torch.Tensor | np.ndarray | None,
    *,
    target_shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Normaliza una máscara opcional booleana a `target_shape` (`[T, N, F]` o `[T, N]`)."""
    if mask is None:
        return torch.ones(target_shape, dtype=torch.bool, device=device)
    m = _as_bool_tensor(mask, device=device)
    while m.ndim < len(target_shape):
        m = m.unsqueeze(-1)
    if m.shape != target_shape:
        m = m.expand(target_shape)
    return m


# =============================================================================
# 1. StructuredModalityDropout (train-only, 0 host-syncs, prices always intact)
# =============================================================================
class StructuredModalityDropout(nn.Module):
    """Dropout estructurado de modalidades y bloques solo en `train()` preservando `prices`."""

    def __init__(
        self,
        *,
        p_drop_fundamentals: float = 0.40,
        p_drop_macro: float = 0.05,
        p_drop_news: float = 0.50,
        p_drop_charts: float = 0.10,
        p_session_blackout: float = 0.20,
        p_feature_block_drop: float = 0.10,
        n_feature_blocks: int = 4,
        market_preset: Literal["custom", "US", "CN"] = "custom",
    ) -> None:
        super().__init__()
        if market_preset == "US":
            prof = US_EMPIRICAL_MISSINGNESS_PROFILE
            p_drop_fundamentals = prof["p_drop_fundamentals"]
            p_drop_macro = prof["p_drop_macro"]
            p_drop_news = prof["p_drop_news"]
            p_drop_charts = prof["p_drop_charts"]
            p_session_blackout = prof["p_session_blackout"]
            p_feature_block_drop = prof["p_feature_block_drop"]
        elif market_preset == "CN":
            prof = CN_EMPIRICAL_MISSINGNESS_PROFILE
            p_drop_fundamentals = prof["p_drop_fundamentals"]
            p_drop_macro = prof["p_drop_macro"]
            p_drop_news = prof["p_drop_news"]
            p_drop_charts = prof["p_drop_charts"]
            p_session_blackout = prof["p_session_blackout"]
            p_feature_block_drop = prof["p_feature_block_drop"]
        elif market_preset != "custom":
            raise ValueError(
                f"market_preset inválido {market_preset!r}; válidos: ('custom', 'US', 'CN')"
            )

        probs = (
            p_drop_fundamentals,
            p_drop_macro,
            p_drop_news,
            p_drop_charts,
            p_session_blackout,
            p_feature_block_drop,
        )
        if any(not (0.0 <= float(p) <= 1.0) for p in probs):
            raise ValueError("Todas las probabilidades de dropout deben pertenecer a [0, 1]")
        if n_feature_blocks < 1:
            raise ValueError("n_feature_blocks debe ser >= 1")

        self.market_preset = market_preset
        self.p_drop_fundamentals = float(p_drop_fundamentals)
        self.p_drop_macro = float(p_drop_macro)
        self.p_drop_news = float(p_drop_news)
        self.p_drop_charts = float(p_drop_charts)
        self.p_session_blackout = float(p_session_blackout)
        self.p_feature_block_drop = float(p_feature_block_drop)
        self.n_feature_blocks = int(n_feature_blocks)

        aux_p = np.asarray(
            [
                self.p_drop_fundamentals,
                self.p_drop_macro,
                self.p_drop_news,
                self.p_drop_charts,
            ],
            dtype=np.float32,
        )
        self.register_buffer("_aux_drop_probs", torch.as_tensor(aux_p, dtype=torch.float32))

    def drop_feature_blocks(
        self,
        x: torch.Tensor,
        availability_mask: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Descarta bloques contiguos de rasgos solo en `train()` sin `NaN * 0` ni syncs."""
        if (not self.training) or self.p_feature_block_drop <= 0.0 or x.shape[-1] == 0:
            return x, availability_mask
        f_dim = x.shape[-1]
        n_blocks = min(self.n_feature_blocks, f_dim)
        dev = x.device
        block_ids = (torch.arange(f_dim, device=dev) * n_blocks) // f_dim  # [F] in 0..n_blocks-1
        lead_shape = x.shape[:-1]
        u_blk = torch.rand(
            (*lead_shape, n_blocks), device=dev, dtype=torch.float32, generator=generator
        )
        keep_blk = u_blk >= self.p_feature_block_drop
        keep_feat = keep_blk[..., block_ids]  # [..., F]
        mask_bool = (
            availability_mask > 0.5
            if availability_mask.is_floating_point()
            else availability_mask.to(dtype=torch.bool)
        )
        new_valid = mask_bool & keep_feat
        # Saneado previo evita NaN*0 incluso si x contiene NaN en posiciones descartadas
        x_out = torch.where(new_valid, x, torch.zeros_like(x))
        m_out = (
            new_valid.to(dtype=availability_mask.dtype)
            if availability_mask.is_floating_point()
            else new_valid
        )
        return x_out, m_out

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Aplica dropout estructurado en `train()`, identidad bit a bit en `eval()`."""
        out = dict(batch)
        if not self.training:
            return out
        if "prices" not in batch:
            return out

        prices = batch["prices"]
        n_assets = prices.shape[0]
        dev = prices.device
        aux_probs = self._aux_drop_probs.to(device=dev, dtype=torch.float32)  # [4]

        raw_mask = batch.get("modality_mask")
        if raw_mask is None:
            mask_dtype = prices.dtype
            orig_mask_bool = torch.ones((n_assets, 5), device=dev, dtype=torch.bool)
        else:
            mask_dtype = raw_mask.dtype
            orig_mask_bool = raw_mask.to(device=dev) > 0.5

        # Nunca apagar precios (columna 0)
        keep_prices = orig_mask_bool[:, 0:1]

        # Apagón a nivel de sesión (emula eras 2000-2008 sin noticias / CN sin fundamentales)
        sess_p = aux_probs * self.p_session_blackout
        u_sess = torch.rand((1, 4), device=dev, dtype=torch.float32, generator=generator)
        keep_sess = u_sess >= sess_p.unsqueeze(0)  # [1, 4]

        # Dropout por activo en las 4 modalidades auxiliares (fundamentals, macro, news, charts)
        u_asset = torch.rand((n_assets, 4), device=dev, dtype=torch.float32, generator=generator)
        keep_asset = u_asset >= aux_probs.unsqueeze(0)  # [N, 4]

        keep_aux = orig_mask_bool[:, 1:5] & keep_sess & keep_asset  # [N, 4]
        new_mask_bool = torch.cat([keep_prices, keep_aux], dim=1)  # [N, 5]
        out["modality_mask"] = new_mask_bool.to(dtype=mask_dtype)

        # Limpiar con `torch.where` los tensores auxiliares (jamás `x * mask` sobre NaN!)
        aux_keys = ((1, "fundamentals"), (2, "macro"), (3, "news"), (4, "charts"))
        for col, key in aux_keys:
            val = batch.get(key)
            if isinstance(val, torch.Tensor):
                t = val.to(device=dev)
                keep_col = new_mask_bool[:, col : col + 1]
                clean_t = torch.where(keep_col & torch.isfinite(t), t, torch.zeros_like(t))
                if key != "macro" and self.p_feature_block_drop > 0.0:
                    clean_t, _ = self.drop_feature_blocks(
                        clean_t, keep_col.expand_as(clean_t), generator=generator
                    )
                out[key] = clean_t

        raw_age = batch.get("modality_age")
        if isinstance(raw_age, torch.Tensor):
            age_t = raw_age.to(device=dev)
            out["modality_age"] = torch.where(
                new_mask_bool & torch.isfinite(age_t), age_t, torch.zeros_like(age_t)
            )
        return out


# =============================================================================
# 2. MissingnessResilientFeaturizer (GPU-batched, 0 host-syncs, BF16/FP32-safe)
# =============================================================================
class MissingnessResilientFeaturizer(nn.Module):
    """Featurizador causal resiliente a ausencias extremas para las 4 familias MMARRS/Titans."""

    def __init__(
        self,
        *,
        winsor_sigma: float = 5.0,
        clip_val: float = LIMIT,
        prior_strength_peer: float = 5.0,
        prior_strength_cs: float = 5.0,
        staleness_halflife_days: float = 63.0,
        max_staleness_days: float = 1000.0,
        level_mode: Literal[
            "winsorized_level", "robust_cs_z", "causal_expanding_z"
        ] = "winsorized_level",
        include_gaussian_rank: bool = True,
        apply_gaussian_rank_to_macro: bool = False,
        modality_dropout: StructuredModalityDropout | None = None,
    ) -> None:
        super().__init__()
        if winsor_sigma <= 0.0 or clip_val <= 0.0:
            raise ValueError("winsor_sigma y clip_val deben ser estrictamente positivos")
        if prior_strength_peer < 0.0 or prior_strength_cs < 0.0:
            raise ValueError("prior_strength_peer y prior_strength_cs deben ser >= 0")
        if staleness_halflife_days <= 0.0 or max_staleness_days <= 0.0:
            raise ValueError("staleness_halflife_days y max_staleness_days deben ser positivos")
        if level_mode not in ("winsorized_level", "robust_cs_z", "causal_expanding_z"):
            raise ValueError(f"level_mode inválido {level_mode!r}")

        self.winsor_sigma = float(winsor_sigma)
        self.clip_val = float(clip_val)
        self.prior_strength_peer = float(prior_strength_peer)
        self.prior_strength_cs = float(prior_strength_cs)
        self.staleness_halflife_days = float(staleness_halflife_days)
        self.max_staleness_days = float(max_staleness_days)
        self.staleness_scale = math.log1p(self.staleness_halflife_days) / math.log(2.0)
        self.level_mode = level_mode
        self.include_gaussian_rank = bool(include_gaussian_rank)
        self.apply_gaussian_rank_to_macro = bool(apply_gaussian_rank_to_macro)
        self.modality_dropout = modality_dropout

    def _hierarchical_eb_impute(
        self,
        x_win: torch.Tensor,
        valid: torch.Tensor,
        peer_exposures: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Imputación Empírica Bayesiana jerárquica estrictamente en sesión `t` (`[T, N, F]`).

        Jerarquía: mediana de pares RMT/sector -> mediana transversal -> `0.0`.
        Devuelve `(eb_target [T, N, F], cs_med [T, 1, F], cs_cnt [T, 1, F])`.
        """
        t_len, n_assets, f_dim = x_win.shape
        calc_dtype = x_win.dtype
        dev = x_win.device

        # Mover eje de activos N a la última dimensión: [T, F, N]
        # UN SOLO sort O(T*F*N log N) compartiendo `x_sorted_val` entre mediana CS y mediana peer!
        x_tfn = x_win.transpose(1, 2).contiguous()
        v_tfn = valid.transpose(1, 2).contiguous()
        x_clean = torch.where(v_tfn, x_tfn, 0.0)
        x_sort_in = torch.where(v_tfn, x_clean, float("inf"))
        x_sorted_val, ord_val = x_sort_in.sort(dim=-1)  # [T, F, N]

        cs_cnt_i = v_tfn.sum(dim=-1, keepdim=True)  # [T, F, 1]
        has_cs = cs_cnt_i > 0
        cs_lo_idx = ((cs_cnt_i - 1) // 2).clamp(0, max(n_assets - 1, 0))
        cs_hi_idx = (cs_cnt_i // 2).clamp(0, max(n_assets - 1, 0))
        cs_lo = torch.where(has_cs, x_sorted_val.gather(-1, cs_lo_idx), 0.0)
        cs_hi = torch.where(has_cs, x_sorted_val.gather(-1, cs_hi_idx), 0.0)
        cs_med_tf = torch.where(has_cs, 0.5 * (cs_lo + cs_hi), 0.0).squeeze(-1)  # [T, F]
        cs_cnt_tf = cs_cnt_i.squeeze(-1).to(dtype=calc_dtype)
        w_cs = cs_cnt_tf / (cs_cnt_tf + self.prior_strength_cs).clamp_min(1e-6)
        eb_cs_tf = w_cs * cs_med_tf  # [T, F] -> 0.0 exacto cuando cs_cnt == 0
        eb_cs_exp = eb_cs_tf.unsqueeze(1).expand(t_len, n_assets, f_dim)

        if peer_exposures is None or peer_exposures.shape[-1] == 0 or n_assets < 2:
            return eb_cs_exp, cs_med_tf.unsqueeze(1), cs_cnt_tf.unsqueeze(1)

        exp_t = peer_exposures.to(device=dev, dtype=calc_dtype)
        if exp_t.ndim == 2:
            exp_t = exp_t.unsqueeze(0).expand(t_len, n_assets, -1)
        exp_clean = torch.where(torch.isfinite(exp_t), exp_t, 0.0)

        # Polos positivos y negativos permiten tanto sectores one-hot (>=0) como cargas RMT firmadas
        poles = torch.cat([F.relu(exp_clean), F.relu(-exp_clean)], dim=-1)  # [T, N, 2K]
        n_groups = poles.shape[-1]
        pole_sum = poles.sum(dim=-1, keepdim=True)  # [T, N, 1]
        has_peer = pole_sum.squeeze(-1) > 1e-6  # [T, N]
        dom_group = poles.argmax(dim=-1)  # [T, N]

        # Reutilizar `x_sorted_val`/`ord_val` (0 sorts redundantes!): orden estable por grupo
        # contiguo + searchsorted extrae `[start_g, end_g)` y la mediana exacta de cada grupo!
        g_base = (
            torch.where(has_peer, dom_group, n_groups)
            .unsqueeze(1)
            .expand(t_len, f_dim, n_assets)
        )
        g_eff = torch.where(v_tfn, g_base, n_groups)  # [T, F, N] en 0..n_groups (n_groups=inválido)
        g_by_val = g_eff.gather(-1, ord_val)  # [T, F, N]
        g_sorted, ord_grp = g_by_val.sort(dim=-1, stable=True)  # [T, F, N]
        x_by_grp = x_sorted_val.gather(-1, ord_grp)  # [T, F, N], cada grupo g ordenado ascendente!

        g_keys = torch.arange(n_groups, device=dev, dtype=g_sorted.dtype).view(1, 1, n_groups)
        g_keys_exp = g_keys.expand(t_len, f_dim, n_groups).contiguous()
        starts = torch.searchsorted(g_sorted, g_keys_exp, right=False)  # [T, F, G]
        ends = torch.searchsorted(g_sorted, g_keys_exp, right=True)  # [T, F, G]
        cnt_i = ends - starts  # [T, F, G]
        has_g = cnt_i > 0
        idx_lo = (starts + (cnt_i - 1) // 2).clamp(0, n_assets - 1)
        idx_hi = (starts + cnt_i // 2).clamp(0, n_assets - 1)
        lo_v = torch.where(has_g, x_by_grp.gather(-1, idx_lo), 0.0)
        hi_v = torch.where(has_g, x_by_grp.gather(-1, idx_hi), 0.0)
        med_tfg = torch.where(has_g, 0.5 * (lo_v + hi_v), 0.0)  # [T, F, G]
        cnt_tfg = cnt_i.to(dtype=calc_dtype)
        w_tfg = cnt_tfg / (cnt_tfg + self.prior_strength_peer).clamp_min(1e-6)
        eb_tfg = w_tfg * med_tfg + (1.0 - w_tfg) * eb_cs_tf.unsqueeze(-1)  # [T, F, G]
        eb_group = eb_tfg.transpose(1, 2).contiguous()  # [T, G, F]

        weights_a = poles / pole_sum.clamp_min(1e-6)  # [T, N, 2K]
        peer_target = torch.bmm(weights_a, eb_group)  # [T, N, F]
        eb_target = torch.where(has_peer.unsqueeze(-1), peer_target, eb_cs_exp)
        return eb_target, cs_med_tf.unsqueeze(1), cs_cnt_tf.unsqueeze(1)

    def _compute_staleness_decay(
        self,
        valid: torch.Tensor,
        staleness: torch.Tensor | None,
        *,
        staleness_is_log1p: bool,
        calc_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calcula `(log1p_age, log1p_staleness_decay)` estrictamente en `[0, 1]` (0 si falta)."""
        dev = valid.device
        zeros = torch.zeros(valid.shape, dtype=calc_dtype, device=dev)
        if staleness is None:
            decay = torch.where(valid, torch.ones_like(zeros), zeros)
            return zeros, decay

        st = staleness.to(device=dev, dtype=calc_dtype)
        while st.ndim < valid.ndim:
            st = st.unsqueeze(-1)
        if st.shape != valid.shape:
            st = st.expand(valid.shape)

        st_ok = valid & torch.isfinite(st)
        st_safe = torch.where(st_ok, st, zeros).clamp_min(0.0)
        if staleness_is_log1p:
            max_log = math.log1p(self.max_staleness_days)
            log_age = st_safe.clamp(0.0, max_log)
        else:
            log_age = torch.log1p(st_safe.clamp(0.0, self.max_staleness_days))
        decay = torch.where(
            valid,
            torch.exp(-log_age / max(self.staleness_scale, 1e-6)),
            zeros,
        )
        return torch.where(valid, log_age, zeros), decay

    def forward(
        self,
        x: torch.Tensor,
        *,
        availability_mask: torch.Tensor | None = None,
        staleness: torch.Tensor | None = None,
        staleness_is_log1p: bool = False,
        exposures: torch.Tensor | None = None,
        rmt_loadings: torch.Tensor | None = None,
        level_mode: Literal[
            "winsorized_level", "robust_cs_z", "causal_expanding_z"
        ]
        | None = None,
        include_gaussian_rank: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """Transforma `[N, F]` o `[T, N, F]` en representación causal resiliente a ausencias."""
        if x.ndim not in (2, 3):
            raise ValueError(f"Se espera tensor 2D [N, F] o 3D [T, N, F]; recibido ndim={x.ndim}")
        orig_dtype = x.dtype
        calc_dtype = _stable_calc_dtype(orig_dtype)
        squeeze_2d = x.ndim == 2
        x3 = x.unsqueeze(0) if squeeze_2d else x
        t_len, n_assets, f_dim = x3.shape
        dev = x3.device

        eff_level_mode = self.level_mode if level_mode is None else level_mode
        eff_use_rank = (
            self.include_gaussian_rank if include_gaussian_rank is None else include_gaussian_rank
        )

        if t_len == 0 or n_assets == 0 or f_dim == 0:
            shape_base = (n_assets, f_dim) if squeeze_2d else (t_len, n_assets, f_dim)
            z = torch.zeros(shape_base, dtype=orig_dtype, device=dev)
            mult = 4 if eff_use_rank else 3
            zf = torch.zeros((*shape_base[:-1], f_dim * mult), dtype=orig_dtype, device=dev)
            return dict(
                sanitized_value=z,
                robust_winsorized_z=z,
                masked_gaussian_rank=z,
                availability_mask=z,
                log1p_age=z,
                log1p_staleness_decay=z,
                anatomy_3ch=torch.cat([z, z, z], dim=-1),
                dual_level_rank=torch.cat([z, z], dim=-1),
                features=zf,
                imputed_mask=torch.zeros(shape_base, dtype=torch.bool, device=dev),
            )

        # 1) Máscara combinada y saneamiento PREVIO (en precisión maestra FP32/FP64 sin autocast!)
        with _autocast_disabled(dev):
            x3_c = x3.to(dtype=calc_dtype)
            fin = torch.isfinite(x3_c)
            if availability_mask is not None:
                m_raw = (
                    availability_mask.unsqueeze(0)
                    if (squeeze_2d and availability_mask.ndim in (1, 2))
                    else availability_mask
                )
                m_bool = _normalize_optional_mask(m_raw, target_shape=x3_c.shape, device=dev)
                valid = fin & m_bool
            else:
                valid = fin

            # Saneamiento directo sobre x3_c ANTES de aritmética -> gradiente 0 exacto en ~valid
            x_safe = torch.where(valid, x3_c, torch.zeros_like(x3_c))

            # 2) Estadísticos robustos estrictamente en sesión `t` (detached anti-fuga de grad)
            x_det = x_safe.detach()
            x_tfn_det = x_det.transpose(1, 2).contiguous()
            v_tfn = valid.transpose(1, 2).contiguous()
            cs_med_tf, cs_cnt_tf = _masked_median_lastdim(x_tfn_det, v_tfn)  # [T, F]
            abs_dev = torch.where(v_tfn, (x_tfn_det - cs_med_tf.unsqueeze(-1)).abs(), 0.0)
            cs_mad_tf, _ = _masked_median_lastdim(abs_dev, v_tfn)
            mad_sigma_tf = 1.4826 * cs_mad_tf

            # Respaldo a desviación estándar enmascarada cuando MAD == 0 (rasgos dispersos)
            cnt_f = cs_cnt_tf.clamp_min(1.0)
            mean_tf = torch.where(v_tfn, x_tfn_det, 0.0).sum(dim=-1) / cnt_f
            diff_tf = torch.where(v_tfn, x_tfn_det - mean_tf.unsqueeze(-1), 0.0)
            std_tf = torch.sqrt(((diff_tf * diff_tf).sum(dim=-1) / cnt_f).clamp_min(0.0))
            scale_tf = torch.where(mad_sigma_tf > 1e-6, mad_sigma_tf, std_tf).clamp_min(1e-6)

            med_b = cs_med_tf.unsqueeze(1)  # [T, 1, F]
            scale_b = scale_tf.unsqueeze(1)  # [T, 1, F]
            lo_b = med_b - self.winsor_sigma * scale_b
            hi_b = med_b + self.winsor_sigma * scale_b

            # Winsorización diferenciable sobre observados (límites detached -> 0 grad en ausentes)
            x_win_obs = torch.where(
                valid, x_safe.clamp(min=lo_b, max=hi_b), torch.zeros_like(x_safe)
            )

            # 3) Nivel robusto + Imputación Empírica Bayesiana Jerárquica (peer -> CS -> 0.0)
            # En modos z-score, estandarizar observados ANTES de imputar garantiza invarianza a
            # desplazamientos (p. ej. log(CNY) ~ [20.9, 29.3] con n_obs=1 en CN sin explosión +8!)
            peer_list: list[torch.Tensor] = []
            for p_src in (exposures, rmt_loadings):
                if p_src is not None:
                    pt = p_src.to(device=dev, dtype=calc_dtype)
                    if squeeze_2d and pt.ndim == 2:
                        pt = pt.unsqueeze(0)
                    elif pt.ndim == 2:
                        pt = pt.unsqueeze(0).expand(t_len, n_assets, -1)
                    peer_list.append(pt)
            peer_exp = torch.cat(peer_list, dim=-1) if peer_list else None

            if eff_level_mode == "robust_cs_z":
                spread_ok = (cs_cnt_tf >= 2.0) & (scale_tf > 1e-6)
                obs_norm = torch.where(
                    valid & spread_ok.unsqueeze(1),
                    (x_win_obs - med_b) / scale_b.clamp_min(1e-6),
                    torch.zeros_like(x_win_obs),
                ).clamp(-self.clip_val, self.clip_val)
                eb_target, _, _ = self._hierarchical_eb_impute(
                    obs_norm.detach(), valid, peer_exposures=peer_exp
                )
                robust_z = torch.where(valid, obs_norm, eb_target).clamp(
                    -self.clip_val, self.clip_val
                )
            elif eff_level_mode == "causal_expanding_z":
                # Estandarización expansiva estrictamente causal sobre `<= t` (cero fuga futura!)
                has_t = (cs_cnt_tf >= 2.0) & (scale_tf > 1e-6)
                has_f = has_t.to(calc_dtype)
                cum_t = torch.cumsum(has_f, dim=0)
                denom_t = cum_t.clamp_min(1.0)
                exp_med = torch.cumsum(torch.where(has_t, cs_med_tf, 0.0), dim=0) / denom_t
                exp_scl = (
                    torch.cumsum(torch.where(has_t, scale_tf, 0.0), dim=0) / denom_t
                ).clamp_min(1e-6)
                has_any = (cum_t > 0.0).unsqueeze(1)
                obs_norm = torch.where(
                    valid & has_any,
                    (x_win_obs - exp_med.unsqueeze(1)) / exp_scl.unsqueeze(1),
                    torch.zeros_like(x_win_obs),
                ).clamp(-self.clip_val, self.clip_val)
                eb_target, _, _ = self._hierarchical_eb_impute(
                    obs_norm.detach(), valid, peer_exposures=peer_exp
                )
                robust_z = torch.where(valid, obs_norm, eb_target).clamp(
                    -self.clip_val, self.clip_val
                )
            else:
                # `winsorized_level`: preserva exactamente el nivel entre sesiones (p. ej. 3x vol!)
                eb_target, _, _ = self._hierarchical_eb_impute(
                    x_win_obs.detach(), valid, peer_exposures=peer_exp
                )
                robust_z = torch.where(valid, x_win_obs, eb_target).clamp(
                    -self.clip_val, self.clip_val
                )

            # 5) Rango Gaussiano transversal con empates (reutiliza `gaussian_rank_features`)
            if eff_use_rank:
                grank = gaussian_rank_features(
                    x_win_obs.detach(), mask=valid, clip_val=self.clip_val
                ).to(dtype=calc_dtype)
            else:
                grank = torch.zeros_like(robust_z)

            # 6) Canales de disponibilidad y decaimiento log1p de obsolescencia
            avail_f = valid.to(dtype=calc_dtype)
            st_in = (
                staleness.unsqueeze(0)
                if (staleness is not None and squeeze_2d and staleness.ndim in (1, 2))
                else staleness
            )
            log_age, staleness_decay = self._compute_staleness_decay(
                valid, st_in, staleness_is_log1p=staleness_is_log1p, calc_dtype=calc_dtype
            )

        # Empaquetado final en `orig_dtype` (BF16/FP32-safe)
        sanitized_out = robust_z.to(dtype=orig_dtype)
        grank_out = grank.to(dtype=orig_dtype)
        avail_out = avail_f.to(dtype=orig_dtype)
        age_out = log_age.to(dtype=orig_dtype)
        decay_out = staleness_decay.to(dtype=orig_dtype)
        imputed_mask = ~valid

        if squeeze_2d:
            sanitized_out = sanitized_out.squeeze(0)
            grank_out = grank_out.squeeze(0)
            avail_out = avail_out.squeeze(0)
            age_out = age_out.squeeze(0)
            decay_out = decay_out.squeeze(0)
            imputed_mask = imputed_mask.squeeze(0)

        anatomy_3ch = torch.cat([sanitized_out, avail_out, decay_out], dim=-1)
        dual_level_rank = torch.cat([sanitized_out, grank_out], dim=-1)
        features = (
            torch.cat([sanitized_out, grank_out, avail_out, decay_out], dim=-1)
            if eff_use_rank
            else anatomy_3ch
        )
        return dict(
            sanitized_value=sanitized_out,
            robust_winsorized_z=sanitized_out,
            masked_gaussian_rank=grank_out,
            availability_mask=avail_out,
            log1p_age=age_out,
            log1p_staleness_decay=decay_out,
            anatomy_3ch=anatomy_3ch,
            dual_level_rank=dual_level_rank,
            features=features,
            imputed_mask=imputed_mask,
        )

    def featurize_batch(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        generator: torch.Generator | None = None,
        fundamental_layout: Literal["auto", "encoded_v31_78", "raw"] = "auto",
        macro_layout: Literal["auto", "encoded_v31_420", "raw"] = "auto",
    ) -> dict[str, torch.Tensor]:
        """Limpia, imputa jerárquicamente y aplica dropout estructurado a un lote multimodal.

        Compatible directamente con las 4 familias independientes (`titans_original`,
        `holographic_brain`, `mmarrs_titan`, `titans_cognitive_core`): preserva exactamente las
        dimensiones de entrada esperadas (incluyendo tripletas `values_observed_log_age` de
        `encoded-history-v3.1`: 78 cols en fundamentales y 420 cols en macro) y añade telemetría
        de frescura `modality_freshness [N, 5]`.
        """
        work: dict[str, torch.Tensor] = (
            self.modality_dropout(batch, generator=generator)
            if self.modality_dropout is not None
            else dict(batch)
        )
        if "prices" not in work:
            return work

        prices = work["prices"]
        n_assets = prices.shape[0]
        dev, orig_dtype = prices.device, prices.dtype
        exposures = work.get("exposures")
        rmt_loadings = work.get("rmt_loadings")

        raw_mask = work.get("modality_mask")
        if raw_mask is None:
            mod_mask_bool = torch.ones((n_assets, 5), device=dev, dtype=torch.bool)
            work["modality_mask"] = mod_mask_bool.to(dtype=orig_dtype)
        else:
            mod_mask_bool = raw_mask.to(device=dev) > 0.5
            work["modality_mask"] = mod_mask_bool.to(dtype=raw_mask.dtype)

        raw_age = work.get("modality_age")
        if isinstance(raw_age, torch.Tensor):
            age_t = raw_age.to(device=dev)
            age_clean = torch.where(
                mod_mask_bool & torch.isfinite(age_t), age_t, torch.zeros_like(age_t)
            ).clamp_min(0.0)
            work["modality_age"] = age_clean
        else:
            age_clean = None

        max_log_age = math.log1p(self.max_staleness_days)

        def _featurize_triplet(
            val_t: torch.Tensor,
            col_mask: torch.Tensor,
            *,
            n_concepts: int,
            l_mode: Literal["winsorized_level", "robust_cs_z", "causal_expanding_z"],
            use_peers: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Procesa tripletas `[vals, obs, log_age]` sin z-scorear máscaras ni edades."""
            v_raw, obs_raw, lage_raw = torch.split(val_t, n_concepts, dim=-1)
            slot_ok = (
                torch.isfinite(obs_raw)
                & (obs_raw > 0.5)
                & torch.isfinite(v_raw)
                & col_mask.unsqueeze(-1)
            )
            lage_clean = torch.where(
                slot_ok & torch.isfinite(lage_raw),
                lage_raw.clamp(0.0, max_log_age),
                torch.zeros_like(lage_raw),
            )
            res_tri = self.forward(
                v_raw,
                availability_mask=slot_ok,
                staleness=lage_clean,
                staleness_is_log1p=True,
                exposures=exposures if use_peers else None,
                rmt_loadings=rmt_loadings if use_peers else None,
                level_mode=l_mode,
                include_gaussian_rank=False,
            )
            san_v = torch.where(
                col_mask.unsqueeze(-1),
                res_tri["sanitized_value"],
                torch.zeros_like(res_tri["sanitized_value"]),
            )
            obs_clean = slot_ok.to(dtype=val_t.dtype)
            decay = res_tri["log1p_staleness_decay"]
            obs_cnt = obs_clean.sum(dim=-1, keepdim=True).clamp_min(1.0)
            fresh_row = torch.where(
                slot_ok.any(dim=-1, keepdim=True),
                decay.sum(dim=-1, keepdim=True) / obs_cnt,
                torch.zeros((n_assets, 1), device=dev, dtype=val_t.dtype),
            )
            out_tri = torch.cat([san_v, obs_clean, lage_clean.to(dtype=val_t.dtype)], dim=-1)
            return out_tri, fresh_row

        # Políticas por modalidad (`include_gaussian_rank=False` evita 2 sorts redundantes/sesión)
        modality_specs: tuple[
            tuple[int, str, Literal["winsorized_level", "robust_cs_z"]], ...
        ] = (
            (0, "prices", "winsorized_level"),
            (1, "fundamentals", "robust_cs_z"),
            (2, "macro", "winsorized_level"),
            (3, "news", "winsorized_level"),
            (4, "charts", "winsorized_level"),
        )
        fresh_cols: list[torch.Tensor] = []
        for col, key, l_mode in modality_specs:
            val = work.get(key)
            col_mask = mod_mask_bool[:, col]
            col_age = age_clean[:, col : col + 1] if age_clean is not None else None
            if isinstance(val, torch.Tensor):
                is_fund_tri = (
                    key == "fundamentals"
                    and val.shape[-1] == 78
                    and fundamental_layout in ("auto", "encoded_v31_78")
                )
                is_macro_tri = (
                    key == "macro"
                    and val.shape[-1] == 420
                    and macro_layout in ("auto", "encoded_v31_420")
                )
                if is_fund_tri:
                    clean_tri, fresh_row = _featurize_triplet(
                        val, col_mask, n_concepts=26, l_mode="robust_cs_z", use_peers=True
                    )
                    work[key] = clean_tri
                    fresh_cols.append(fresh_row)
                elif is_macro_tri:
                    clean_tri, fresh_row = _featurize_triplet(
                        val, col_mask, n_concepts=140, l_mode="winsorized_level", use_peers=False
                    )
                    work[key] = clean_tri
                    fresh_cols.append(fresh_row)
                else:
                    res = self.forward(
                        val,
                        availability_mask=col_mask.unsqueeze(-1),
                        staleness=col_age,
                        exposures=exposures if key != "macro" else None,
                        rmt_loadings=rmt_loadings if key != "macro" else None,
                        level_mode=l_mode,
                        include_gaussian_rank=False,
                    )
                    san = res["sanitized_value"]
                    work[key] = torch.where(col_mask.unsqueeze(-1), san, torch.zeros_like(san))
                    fresh_cols.append(res["log1p_staleness_decay"].mean(dim=-1, keepdim=True))
            else:
                z_col = torch.zeros((n_assets, 1), device=dev, dtype=orig_dtype)
                fresh_cols.append(z_col)

        # Limpiar también `physics_v4` y `high_snr_context` si existen
        for extra_key in ("physics_v4", "high_snr_context"):
            ev = work.get(extra_key)
            if isinstance(ev, torch.Tensor):
                res_e = self.forward(
                    ev,
                    exposures=exposures if extra_key == "physics_v4" else None,
                    rmt_loadings=rmt_loadings if extra_key == "physics_v4" else None,
                    level_mode="winsorized_level",
                    include_gaussian_rank=False,
                )
                work[extra_key] = res_e["sanitized_value"]

        work["modality_freshness"] = torch.cat(fresh_cols, dim=-1)
        return work


# =============================================================================
# 3. enrich_full_dataset_sota_torch (24 señales institucionales & multi-régimen)
# =============================================================================
def _compute_pit_fundamental_composites(
    fundamentals: torch.Tensor | None,
    *,
    fundamentals_mask: torch.Tensor | None,
    fundamentals_age: torch.Tensor | None,
    log_close: torch.Tensor,
    fundamental_layout: Literal["auto", "encoded_v31_78", "raw"],
    staleness_scale: float,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construye `(quality, value, profitability, fund_valid, fund_freshness)` estrictamente PIT."""
    t_len, n_assets = log_close.shape
    dev = log_close.device
    zeros = torch.zeros((t_len, n_assets), dtype=compute_dtype, device=dev)
    false_m = torch.zeros((t_len, n_assets), dtype=torch.bool, device=dev)

    if fundamentals is None or fundamentals.shape[-1] == 0:
        return zeros, zeros, zeros, false_m, zeros

    f_t = fundamentals.to(device=dev, dtype=compute_dtype)
    if f_t.ndim == 2:
        f_t = f_t.unsqueeze(1)
    d_f = f_t.shape[-1]
    base_mask = _normalize_optional_mask(
        fundamentals_mask, target_shape=(t_len, n_assets), device=dev
    )
    use_v31 = fundamental_layout == "encoded_v31_78" or (fundamental_layout == "auto" and d_f == 78)

    if use_v31 and d_f == 78:
        f_vals, f_obs, f_lage = torch.split(f_t, 26, dim=-1)  # [T, N, 26] cada uno
        slot_ok = torch.isfinite(f_vals) & (f_obs > 0.5) & base_mask.unsqueeze(-1)
        lage_safe = torch.where(slot_ok & torch.isfinite(f_lage), f_lage, 0.0).clamp(0.0, 10.0)
        slot_fresh = torch.where(
            slot_ok, torch.exp(-lage_safe / max(staleness_scale, 1e-6)), 0.0
        )
        vals_safe = torch.where(slot_ok, f_vals, 0.0)
        g_slots = gaussian_rank_features(vals_safe, mask=slot_ok, clip_val=LIMIT).to(compute_dtype)

        def _weighted_slot_blend(specs: Sequence[tuple[int, float]]) -> torch.Tensor:
            num = torch.zeros((t_len, n_assets), dtype=compute_dtype, device=dev)
            den = torch.zeros((t_len, n_assets), dtype=compute_dtype, device=dev)
            for s_idx, sign in specs:
                ok_s = slot_ok[..., s_idx]
                w_s = slot_fresh[..., s_idx]
                num = num + torch.where(ok_s, sign * g_slots[..., s_idx] * w_s, 0.0)
                den = den + torch.where(ok_s, ok_s.to(compute_dtype), 0.0)
            return torch.where(den > 0.0, num / den.clamp_min(1.0), 0.0)

        # Calidad de balance: liquidez corriente, capital de trabajo, solvencia y patrimonio
        pit_quality = _weighted_slot_blend(
            ((8, 1.0), (9, 1.0), (10, -1.0), (11, 1.0), (25, 1.0), (24, -1.0))
        )
        # Rentabilidad / eficiencia operativa: efectivo vs activos, ciclo comercial y apalancamiento
        pit_prof = _weighted_slot_blend(
            ((5, 1.0), (0, -0.5), (12, 1.0), (13, -1.0), (14, -1.0), (25, 1.0), (23, -0.5))
        )
        # Valor contable relativo al nivel de precio: rankear SEPARADAMENTE por divisa (USD/CAD/CNY)
        # antes de promediar con frescura para eliminar sesgos de escala cambiaria (invarianza FX!)
        eq_slots = (4, 19, 25)
        val_num = torch.zeros((t_len, n_assets), dtype=compute_dtype, device=dev)
        val_cnt = torch.zeros((t_len, n_assets), dtype=compute_dtype, device=dev)
        for s_idx in eq_slots:
            ok_s = slot_ok[..., s_idx] & torch.isfinite(log_close)
            diff_s = torch.where(ok_s, vals_safe[..., s_idx] - log_close, 0.0)
            g_s = gaussian_rank_features(
                diff_s.unsqueeze(-1), mask=ok_s.unsqueeze(-1), clip_val=LIMIT
            ).squeeze(-1)
            val_num = val_num + torch.where(ok_s, g_s * slot_fresh[..., s_idx], 0.0)
            val_cnt = val_cnt + ok_s.to(compute_dtype)
        pit_value = torch.where(val_cnt > 0.0, val_num / val_cnt.clamp_min(1.0), 0.0)

        any_obs = slot_ok.any(dim=-1)
        obs_cnt = slot_ok.to(compute_dtype).sum(dim=-1).clamp_min(1.0)
        mean_fresh = torch.where(any_obs, slot_fresh.sum(dim=-1) / obs_cnt, 0.0)
        return pit_quality, pit_value, pit_prof, any_obs, mean_fresh

    # Formato genérico `[T, N, D_f]`
    slot_ok = torch.isfinite(f_t) & base_mask.unsqueeze(-1)
    any_obs = slot_ok.any(dim=-1)
    vals_safe = torch.where(slot_ok, f_t, 0.0)
    g_all = gaussian_rank_features(vals_safe, mask=slot_ok, clip_val=LIMIT).to(compute_dtype)

    if fundamentals_age is not None:
        age_t = fundamentals_age.to(device=dev, dtype=compute_dtype)
        if age_t.ndim == 3:
            age_t = age_t.mean(dim=-1)
        age_ok = any_obs & torch.isfinite(age_t)
        log_age = torch.log1p(torch.where(age_ok, age_t, 0.0).clamp(0.0, 1000.0))
        fresh = torch.where(any_obs, torch.exp(-log_age / max(staleness_scale, 1e-6)), 0.0)
    else:
        fresh = any_obs.to(compute_dtype)

    c1 = max(1, d_f // 3)
    c2 = max(c1, (2 * d_f) // 3)
    blk_slices = (
        slice(0, c1),
        slice(c1, c2) if c2 > c1 else slice(0, d_f),
        slice(c2, d_f) if c2 < d_f else slice(0, d_f),
    )

    comps: list[torch.Tensor] = []
    for sl in blk_slices:
        ok_b = slot_ok[..., sl]
        cnt_b = ok_b.to(compute_dtype).sum(dim=-1)
        avg_b = torch.where(ok_b, g_all[..., sl], 0.0).sum(dim=-1) / cnt_b.clamp_min(1.0)
        comps.append(torch.where(cnt_b > 0.0, avg_b * fresh, 0.0))
    return comps[0], comps[1], comps[2], any_obs, fresh


def _causal_embedding_drift(
    emb: torch.Tensor,
    valid: torch.Tensor,
    *,
    halflife: float = 10.0,
) -> torch.Tensor:
    """Deriva L2 causal `||e_t - ewma(e)_{t-1}|| / sqrt(D)` en O(log T) sin bucles sobre T."""
    t_len, n_assets, d_emb = emb.shape
    dev, dtype = emb.device, emb.dtype
    if t_len == 0 or n_assets == 0 or d_emb == 0:
        return torch.zeros((t_len, n_assets), dtype=dtype, device=dev)

    decay = math.pow(0.5, 1.0 / max(halflife, 1.0))
    inv_sqrt_d = 1.0 / math.sqrt(float(d_emb))
    out = torch.zeros((t_len, n_assets), dtype=dtype, device=dev)
    n_chunk = max(1, min(n_assets, _L2_CHUNK_TARGET_ELEMS // max(t_len * d_emb, 1)))
    for n0 in range(0, n_assets, n_chunk):
        n1 = min(n0 + n_chunk, n_assets)
        ec = emb[:, n0:n1]
        vc = valid[:, n0:n1]
        vc_col = vc.unsqueeze(-1)
        e_safe = torch.where(vc_col & torch.isfinite(ec), ec, 0.0)
        a_scan = torch.where(vc_col, decay, 1.0).to(dtype)
        b_scan = torch.where(vc_col, (1.0 - decay) * e_safe, 0.0)
        mem = _causal_affine_scan(a_scan, b_scan)
        d0 = torch.linalg.vector_norm(e_safe[0], dim=-1) * inv_sqrt_d
        out[0, n0:n1] = torch.where(vc[0], d0, 0.0)
        if t_len > 1:
            dt = torch.linalg.vector_norm(e_safe[1:] - mem[:-1], dim=-1) * inv_sqrt_d
            out[1:, n0:n1] = torch.where(vc[1:], dt, 0.0)
    return out


def _compute_avail_surprise_21(
    fund_valid: torch.Tensor | None,
    news_ok: torch.Tensor | None,
    ch_ok: torch.Tensor | None,
    *,
    traded: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Choque causal MNAR `avail_rate_t - ema_21(avail_rate)` sobre `(fund, news, charts)`."""
    t_len, n_assets = traded.shape
    dev = traded.device
    passed = [m for m in (fund_valid, news_ok, ch_ok) if m is not None]
    if not passed or t_len == 0 or n_assets == 0:
        return torch.zeros((t_len, n_assets), dtype=compute_dtype, device=dev)

    avail_sum = torch.zeros((t_len, n_assets), dtype=compute_dtype, device=dev)
    for m in passed:
        avail_sum = avail_sum + m.to(device=dev, dtype=compute_dtype)
    avail_rate = avail_sum / float(len(passed))

    alpha = 2.0 / 22.0
    decay = 1.0 - alpha
    t_is_0 = (torch.arange(t_len, device=dev) == 0).unsqueeze(1)
    a_ema = torch.where(t_is_0, 0.0, decay).to(dtype=compute_dtype).expand(t_len, n_assets)
    b_ema = torch.where(t_is_0, avail_rate, alpha * avail_rate)
    ema_21 = _causal_affine_scan(a_ema, b_ema)
    return torch.where(traded, avail_rate - ema_21, 0.0)


def enrich_full_dataset_sota_torch(
    open_: torch.Tensor | np.ndarray,
    high: torch.Tensor | np.ndarray,
    low: torch.Tensor | np.ndarray,
    close: torch.Tensor | np.ndarray,
    volume: torch.Tensor | np.ndarray,
    *,
    market_returns: torch.Tensor | np.ndarray | None = None,
    peer_exposures: torch.Tensor | np.ndarray | None = None,
    fundamentals: torch.Tensor | np.ndarray | None = None,
    fundamentals_mask: torch.Tensor | np.ndarray | None = None,
    fundamentals_age: torch.Tensor | np.ndarray | None = None,
    fundamental_layout: Literal["auto", "encoded_v31_78", "raw"] = "auto",
    news: torch.Tensor | np.ndarray | None = None,
    news_mask: torch.Tensor | np.ndarray | None = None,
    charts: torch.Tensor | np.ndarray | None = None,
    charts_mask: torch.Tensor | np.ndarray | None = None,
    macro: torch.Tensor | np.ndarray | None = None,
    macro_mask: torch.Tensor | np.ndarray | None = None,
    staleness_halflife_days: float = 63.0,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Calcula los 24 rasgos enriquecidos SOTA (`ENRICHED_SOTA_NAMES`) causales y 0 host-syncs.

    Acepta OHLCV `[T, N]` (devuelve `[T, N, 24]`) o 1D `[T]` (devuelve `[T, 24]`), `float32`
    estrictamente finito en `[-8, 8]`. Enmascara Day-0 y suspensiones (`volume <= 0`, críticas en
    CN `3.82%`), y garantiza exactamente `0.0` sin NaNs en interacciones cross-modales cuando falta
    cualquiera de las modalidades.
    """
    in_is_f64 = (isinstance(close, torch.Tensor) and close.dtype == torch.float64) or (
        isinstance(close, np.ndarray) and close.dtype == np.float64
    )
    prep_dtype = torch.float64 if (in_is_f64 or compute_dtype == torch.float64) else compute_dtype

    o_raw = _as_float_tensor(open_, dtype=prep_dtype)
    dev = o_raw.device
    h_raw = _as_float_tensor(high, dtype=prep_dtype, device=dev)
    l_raw = _as_float_tensor(low, dtype=prep_dtype, device=dev)
    c_raw = _as_float_tensor(close, dtype=prep_dtype, device=dev)
    v_raw = _as_float_tensor(volume, dtype=prep_dtype, device=dev)

    if not (o_raw.shape == h_raw.shape == l_raw.shape == c_raw.shape == v_raw.shape):
        raise ValueError("open_, high, low, close y volume deben tener exactamente la misma forma")
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
    n_feats = len(ENRICHED_SOTA_NAMES)
    if t_len == 0 or n_assets == 0:
        shape = (t_len, n_feats) if squeeze_1d else (t_len, n_assets, n_feats)
        return torch.empty(shape, dtype=torch.float32, device=dev)

    with _autocast_disabled(dev):
        return _enrich_full_dataset_sota_core(
            o_raw,
            h_raw,
            l_raw,
            c_raw,
            v_raw,
            squeeze_1d=squeeze_1d,
            market_returns=market_returns,
            peer_exposures=peer_exposures,
            fundamentals=fundamentals,
            fundamentals_mask=fundamentals_mask,
            fundamentals_age=fundamentals_age,
            fundamental_layout=fundamental_layout,
            news=news,
            news_mask=news_mask,
            charts=charts,
            charts_mask=charts_mask,
            macro=macro,
            macro_mask=macro_mask,
            staleness_halflife_days=staleness_halflife_days,
            compute_dtype=compute_dtype,
        )


def _enrich_full_dataset_sota_core(
    o_raw: torch.Tensor,
    h_raw: torch.Tensor,
    l_raw: torch.Tensor,
    c_raw: torch.Tensor,
    v_raw: torch.Tensor,
    *,
    squeeze_1d: bool,
    market_returns: torch.Tensor | np.ndarray | None,
    peer_exposures: torch.Tensor | np.ndarray | None,
    fundamentals: torch.Tensor | np.ndarray | None,
    fundamentals_mask: torch.Tensor | np.ndarray | None,
    fundamentals_age: torch.Tensor | np.ndarray | None,
    fundamental_layout: Literal["auto", "encoded_v31_78", "raw"],
    news: torch.Tensor | np.ndarray | None,
    news_mask: torch.Tensor | np.ndarray | None,
    charts: torch.Tensor | np.ndarray | None,
    charts_mask: torch.Tensor | np.ndarray | None,
    macro: torch.Tensor | np.ndarray | None,
    macro_mask: torch.Tensor | np.ndarray | None,
    staleness_halflife_days: float,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    t_len, n_assets = o_raw.shape
    dev = o_raw.device
    # Higiene OHLCV y máscara de días operados (halts `volume <= 0` y precios no finitos -> False)
    px_ok = torch.isfinite(c_raw) & (c_raw > 0.0)
    traded = px_ok & torch.isfinite(v_raw) & (v_raw > 0.0)
    o = torch.where(torch.isfinite(o_raw) & (o_raw > 0.0), o_raw, 1e-12).clamp_min(1e-12)
    c = torch.where(px_ok, c_raw, 1e-12).clamp_min(1e-12)
    h = torch.maximum(torch.where(torch.isfinite(h_raw), h_raw, o), torch.maximum(o, c))
    l_ = torch.minimum(
        torch.where(torch.isfinite(l_raw) & (l_raw > 0.0), l_raw, o).clamp_min(1e-12),
        torch.minimum(o, c),
    )
    v = torch.where(traded, v_raw, 0.0).clamp_min(0.0)

    log_h_p = torch.log(h)
    log_l_p = torch.log(l_)
    log_c_p = torch.log(c)
    dvol_p = (c * v).clamp_min(0.0)

    log_c = log_c_p.to(dtype=compute_dtype)
    hl = (log_h_p - log_l_p).clamp_min(0.0).to(dtype=compute_dtype)
    dvol = dvol_p.to(dtype=compute_dtype)

    # Retorno close-to-close en días consecutivos operados (Day-0 y halts -> NaN)
    ret_cc = torch.full((t_len, n_assets), float("nan"), dtype=compute_dtype, device=dev)
    if t_len > 1:
        pair_traded = traded[1:] & traded[:-1]
        r_diff = (log_c_p[1:] - log_c_p[:-1]).to(dtype=compute_dtype)
        ret_cc[1:] = torch.where(pair_traded, r_diff, float("nan"))

    # Retorno de mercado causal compacto `[T, 1]`
    if market_returns is not None:
        m_t = _as_float_tensor(market_returns, dtype=compute_dtype, device=dev)
        if m_t.ndim == 1:
            r_mkt = m_t.unsqueeze(1)
        elif m_t.ndim == 2 and m_t.shape[1] in (1, n_assets):
            r_mkt = m_t
        else:
            raise ValueError(f"Forma inválida de market_returns: {tuple(m_t.shape)}")
    else:
        fin_cc = torch.isfinite(ret_cc)
        cnt_m = fin_cc.sum(dim=1, keepdim=True)
        sum_m = torch.where(fin_cc, ret_cc, 0.0).sum(dim=1, keepdim=True)
        r_mkt = torch.where(
            cnt_m > 0, sum_m / cnt_m.clamp_min(1).to(compute_dtype), float("nan")
        )

    # -------------------------------------------------------------------------
    # 1) Jump vs Continuous Volatility Decomposition (Barndorff-Nielsen & Shephard)
    # -------------------------------------------------------------------------
    r_fin = torch.isfinite(ret_cc)
    r_sq = torch.where(r_fin, ret_cc * ret_cc, 0.0)
    rv_cnt = _rolling_sum_2d(r_fin.to(compute_dtype), 21, assume_finite=True)
    rv_daily = _rolling_sum_2d(r_sq, 21, assume_finite=True) / rv_cnt.clamp_min(1.0)

    abs_r = ret_cc.abs()
    bp_prod = torch.full_like(ret_cc, float("nan"))
    if t_len > 1:
        bp_ok = r_fin[1:] & r_fin[:-1]
        bp_prod[1:] = torch.where(bp_ok, abs_r[1:] * abs_r[:-1], float("nan"))
    bp_fin = torch.isfinite(bp_prod)
    bp_cnt = _rolling_sum_2d(bp_fin.to(compute_dtype), 21, assume_finite=True)
    bv_daily = (
        (0.5 * math.pi)
        * _rolling_sum_2d(torch.where(bp_fin, bp_prod, 0.0), 21, assume_finite=True)
        / bp_cnt.clamp_min(1.0)
    )
    bv_21 = torch.where(
        bp_cnt >= 5.0,
        torch.log(torch.sqrt(bv_daily.clamp_min(1e-10) * 252.0) * 100.0 + 1e-6),
        0.0,
    )
    jv_daily = (rv_daily - bv_daily).clamp_min(0.0)
    jump_var_ratio_21 = torch.where(
        (bp_cnt >= 5.0) & (rv_daily > 1e-12),
        jv_daily / (rv_daily + 1e-10),
        0.0,
    ).clamp(0.0, 1.0)

    # Vol-of-vol trimestral 63d sobre log-sigma Parkinson diario (solo días operados)
    park_var = (hl * hl) / (4.0 * math.log(2.0))
    daily_log_sig = torch.where(
        traded & torch.isfinite(hl), 0.5 * torch.log(park_var.clamp_min(1e-10)), float("nan")
    )
    vov_var63, vov_cnt63 = _rolling_var_anchored_2d(daily_log_sig, 63)
    vol_of_vol_63 = torch.where(vov_cnt63 >= 15.0, torch.sqrt(vov_var63.clamp_min(0.0)), 0.0)

    # Semivarianza firmada BNS/Bollerslev-Li-Zhao: (RS+ - RS-) / (RS+ + RS-)
    rs_pos = _rolling_sum_2d(
        torch.where(r_fin & (ret_cc > 0.0), r_sq, 0.0), 21, assume_finite=True
    )
    rs_neg = _rolling_sum_2d(
        torch.where(r_fin & (ret_cc < 0.0), r_sq, 0.0), 21, assume_finite=True
    )
    rs_tot = rs_pos + rs_neg
    realized_skew_signed_21 = torch.where(
        (rv_cnt >= 5.0) & (rs_tot > 1e-12),
        (rs_pos - rs_neg) / (rs_tot + 1e-10),
        0.0,
    ).clamp(-1.0, 1.0)

    # -------------------------------------------------------------------------
    # 2) Cross-Sectional Peer-Demeaned & Liquidity-Adjusted Alpha Signals
    # -------------------------------------------------------------------------
    exp_t = (
        _as_float_tensor(peer_exposures, dtype=compute_dtype, device=dev)
        if peer_exposures is not None
        else None
    )

    def _cum_ret_horizon(h_win: int) -> torch.Tensor:
        out_r = torch.full((t_len, n_assets), float("nan"), dtype=compute_dtype, device=dev)
        if t_len > h_win:
            # Exigir ambos extremos operados (`traded`) para excluir precios congelados de halts CN
            ok_h = traded[h_win:] & traded[:-h_win]
            out_r[h_win:] = torch.where(
                ok_h, (log_c_p[h_win:] - log_c_p[:-h_win]).to(compute_dtype), float("nan")
            )
        return out_r

    ret_5 = _cum_ret_horizon(5)
    ret_21 = _cum_ret_horizon(21)
    ret_63 = _cum_ret_horizon(63)

    peer_resid_mom_5 = _peer_residualize_returns(ret_5, exp_t)
    peer_resid_mom_21 = _peer_residualize_returns(ret_21, exp_t)
    peer_resid_mom_63 = _peer_residualize_returns(ret_63, exp_t)

    # Short-term liquidity-conditioned reversal ortogonalizado a `peer_resid_mom_5` (corr ~ 0):
    # `rev_5d * (2 * liq_pct - 1)` captura el término puro de interacción liquidez × reversión 5d
    log_dvol = torch.where(traded, torch.log1p(dvol), float("nan"))
    ldv_m21, ldv_cnt21 = _rolling_mean_cnt_2d(log_dvol, 21)
    liq_valid = traded & (ldv_cnt21 >= 3.0)
    liq_pct = _masked_percentile_rank_2d(ldv_m21, liq_valid)
    liq_centered = 2.0 * liq_pct - 1.0
    liq_cond_rev_5 = torch.where(
        liq_valid & torch.isfinite(ret_5), -peer_resid_mom_5 * liq_centered, 0.0
    )

    # Volume-clock / abnormal turnover-signed return (acumulación vs distribución)
    ldv_m63, ldv_cnt63 = _rolling_mean_cnt_2d(log_dvol, 63)
    abn_turn = torch.where(
        traded & r_fin & (ldv_cnt63 >= 5.0),
        log_dvol - ldv_m63,
        float("nan"),
    )
    signed_turn_daily = torch.where(
        torch.isfinite(abn_turn),
        torch.sign(ret_cc) * torch.tanh(abs_r / 0.02) * torch.tanh(abn_turn),
        float("nan"),
    )
    sat_m21, sat_cnt21 = _rolling_mean_cnt_2d(signed_turn_daily, 21)
    signed_abn_turn_21 = torch.where(sat_cnt21 >= 5.0, sat_m21, 0.0)

    # Kaufman Efficiency Ratio 21d
    net_move_21 = _rolling_sum_2d(torch.where(r_fin, ret_cc, 0.0), 21, assume_finite=True).abs()
    path_len_21 = _rolling_sum_2d(torch.where(r_fin, abs_r, 0.0), 21, assume_finite=True)
    kaufman_er_21 = torch.where(
        (rv_cnt >= 5.0) & (path_len_21 > 1e-10),
        net_move_21 / path_len_21.clamp_min(1e-10),
        0.0,
    ).clamp(0.0, 1.0)

    # Amihud Asymmetry (down-day vs up-day illiquidity ratio, Brennan-Huh-Subrahmanyam)
    illiq = torch.where(
        traded & r_fin,
        torch.log1p(abs_r / dvol.clamp_min(1.0) * 1e6),
        float("nan"),
    )
    up_ok = torch.isfinite(illiq) & (ret_cc > 0.0)
    dn_ok = torch.isfinite(illiq) & (ret_cc < 0.0)
    cnt_up = _rolling_sum_2d(up_ok.to(compute_dtype), 21, assume_finite=True)
    cnt_dn = _rolling_sum_2d(dn_ok.to(compute_dtype), 21, assume_finite=True)
    am_up = (
        _rolling_sum_2d(torch.where(up_ok, illiq, 0.0), 21, assume_finite=True)
        / cnt_up.clamp_min(1.0)
    )
    am_dn = (
        _rolling_sum_2d(torch.where(dn_ok, illiq, 0.0), 21, assume_finite=True)
        / cnt_dn.clamp_min(1.0)
    )
    amihud_asym_up_down_21 = torch.where(
        (cnt_up >= 2.0) & (cnt_dn >= 2.0),
        (am_dn - am_up) / (am_dn + am_up + 1e-6),
        0.0,
    ).clamp(-1.0, 1.0)

    # -------------------------------------------------------------------------
    # 3 & 4) PIT Fundamental Composites + Cross-Modal Interaction & Contradiction
    # -------------------------------------------------------------------------
    staleness_scale = math.log1p(max(staleness_halflife_days, 1.0)) / math.log(2.0)
    fund_t = (
        _as_float_tensor(fundamentals, dtype=compute_dtype, device=dev)
        if fundamentals is not None
        else None
    )
    fund_age_t = (
        _as_float_tensor(fundamentals_age, dtype=compute_dtype, device=dev)
        if fundamentals_age is not None
        else None
    )
    pit_fund_quality, pit_fund_value, pit_fund_profitability, fund_valid, fund_fresh = (
        _compute_pit_fundamental_composites(
            fund_t,
            fundamentals_mask=fundamentals_mask,
            fundamentals_age=fund_age_t,
            log_close=log_c,
            fundamental_layout=fundamental_layout,
            staleness_scale=staleness_scale,
            compute_dtype=compute_dtype,
        )
    )

    # Cross-modal 1: `price_vs_news_divergence` (0.0 exacto si falta noticias o precios)
    news_ok_opt: torch.Tensor | None = None
    if news is not None:
        news_t = _as_float_tensor(news, dtype=compute_dtype, device=dev)
        if squeeze_1d and news_t.ndim == 1:
            news_t = news_t.unsqueeze(1)
        n_m = _normalize_optional_mask(news_mask, target_shape=(t_len, n_assets), device=dev)
        if news_t.ndim == 2:
            news_ok = n_m & torch.isfinite(news_t)
            news_sig = torch.where(news_ok, news_t, 0.0)
        elif news_t.ndim == 3:
            fin_n = torch.isfinite(news_t)
            news_ok = n_m & fin_n.any(dim=-1)
            n_drift = _causal_embedding_drift(news_t, news_ok, halflife=5.0)
            n_dir = torch.where(fin_n, news_t, 0.0).mean(dim=-1)
            news_sig = torch.where(news_ok, torch.sign(n_dir) * n_drift, 0.0)
        else:
            raise ValueError(f"news debe ser 2D [T, N] o 3D [T, N, D]; recibido {news_t.shape}")
        news_ok_opt = news_ok
        both_pn = traded & torch.isfinite(ret_5) & news_ok
        g_r5 = gaussian_rank_features(
            torch.where(both_pn, ret_5, 0.0).unsqueeze(-1),
            mask=both_pn.unsqueeze(-1),
            clip_val=LIMIT,
        ).squeeze(-1)
        g_news = gaussian_rank_features(
            news_sig.unsqueeze(-1), mask=both_pn.unsqueeze(-1), clip_val=LIMIT
        ).squeeze(-1)
        price_vs_news_divergence = torch.where(both_pn, g_r5 - g_news, 0.0)
    else:
        price_vs_news_divergence = torch.zeros((t_len, n_assets), dtype=compute_dtype, device=dev)

    # Cross-modal 2: `price_vs_fund_quality_gap` (0.0 exacto si faltan fundamentales)
    both_pf = traded & torch.isfinite(ret_63) & fund_valid
    g_r63 = gaussian_rank_features(
        torch.where(both_pf, ret_63, 0.0).unsqueeze(-1),
        mask=both_pf.unsqueeze(-1),
        clip_val=LIMIT,
    ).squeeze(-1)
    price_vs_fund_quality_gap = torch.where(
        both_pf, g_r63 * fund_fresh - pit_fund_quality, 0.0
    )

    # Cross-modal 3: `chart_vs_vol_exhaustion` (0.0 exacto si faltan gráficos)
    ch_ok_opt: torch.Tensor | None = None
    if charts is not None:
        ch_t = _as_float_tensor(charts, dtype=compute_dtype, device=dev)
        if squeeze_1d and ch_t.ndim == 1:
            ch_t = ch_t.unsqueeze(1)
        c_m = _normalize_optional_mask(charts_mask, target_shape=(t_len, n_assets), device=dev)
        if ch_t.ndim == 2:
            ch_ok = c_m & torch.isfinite(ch_t)
            ch_drift = torch.where(ch_ok, ch_t, 0.0)
        elif ch_t.ndim == 3:
            ch_ok = c_m & torch.isfinite(ch_t).any(dim=-1)
            ch_drift = _causal_embedding_drift(ch_t, ch_ok, halflife=10.0)
        else:
            raise ValueError(f"charts debe ser 2D [T, N] o 3D [T, N, D]; recibido {ch_t.shape}")
        ch_ok_opt = ch_ok
        both_pc = traded & (bp_cnt >= 5.0) & ch_ok
        g_ch = gaussian_rank_features(
            ch_drift.unsqueeze(-1), mask=both_pc.unsqueeze(-1), clip_val=LIMIT
        ).squeeze(-1)
        g_jp = gaussian_rank_features(
            jump_var_ratio_21.unsqueeze(-1), mask=both_pc.unsqueeze(-1), clip_val=LIMIT
        ).squeeze(-1)
        chart_vs_vol_exhaustion = torch.where(both_pc, g_ch * g_jp, 0.0)
    else:
        chart_vs_vol_exhaustion = torch.zeros((t_len, n_assets), dtype=compute_dtype, device=dev)

    # Cross-modal 4: `macro_stress_x_downside_beta` (0.0 exacto si falta macro)
    if macro is not None:
        mac_t = _as_float_tensor(macro, dtype=compute_dtype, device=dev)
        if mac_t.ndim == 1:
            mac_2d = mac_t.unsqueeze(1)  # [T, 1]
        elif mac_t.ndim == 2:
            mac_2d = mac_t  # [T, D_m]
        elif mac_t.ndim == 3:
            fin_m = torch.isfinite(mac_t)
            cnt_m3 = fin_m.to(compute_dtype).sum(dim=1).clamp_min(1.0)
            mac_2d = torch.where(fin_m, mac_t, 0.0).sum(dim=1) / cnt_m3  # [T, D_m]
        else:
            raise ValueError(f"Forma inválida de macro: {tuple(mac_t.shape)}")

        mac_fin = torch.isfinite(mac_2d)
        mac_t_ok = mac_fin.any(dim=1, keepdim=True)  # [T, 1]
        m_mask_tn = _normalize_optional_mask(
            macro_mask, target_shape=(t_len, n_assets), device=dev
        )
        mac_valid_tn = traded & m_mask_tn & mac_t_ok.expand(t_len, n_assets)

        # Innovación macro estrictamente causal: ||m_t - media_63(m)_{t-1}|| / sqrt(D_m)
        m_mean63, m_cnt63 = _rolling_mean_cnt_2d(mac_2d, 63)
        m_mean_lag = torch.zeros_like(m_mean63)
        m_cnt_lag = torch.zeros_like(m_cnt63)
        if t_len > 1:
            m_mean_lag[1:] = m_mean63[:-1]
            m_cnt_lag[1:] = m_cnt63[:-1]
        m_diff = torch.where(mac_fin & (m_cnt_lag >= 5.0), mac_2d - m_mean_lag, 0.0)
        d_m = max(mac_2d.shape[1], 1)
        m_innov = torch.linalg.vector_norm(m_diff, dim=1, keepdim=True) / math.sqrt(float(d_m))
        m_innov = torch.where(mac_t_ok, m_innov, float("nan"))
        im_mean, im_cnt = _rolling_mean_cnt_2d(m_innov, 63)
        im_var, _ = _rolling_var_anchored_2d(m_innov, 63)
        im_std = torch.sqrt(im_var.clamp_min(0.0))
        macro_stress_z = torch.where(
            (im_cnt >= 5.0) & (im_std > 1e-8) & torch.isfinite(m_innov),
            (m_innov - im_mean) / im_std.clamp_min(1e-8),
            0.0,
        ).clamp(-4.0, 4.0)

        t_ge_1 = (torch.arange(t_len, device=dev) >= 1).unsqueeze(1)
        valid_db = t_ge_1 & r_fin & torch.isfinite(r_mkt)
        downside_beta_63 = _rolling_downside_beta_2d(r_mkt, ret_cc, valid_db, window=63)
        macro_stress_x_downside_beta = torch.where(
            mac_valid_tn,
            macro_stress_z.expand(t_len, n_assets) * downside_beta_63.clamp(-4.0, 4.0),
            0.0,
        )
    else:
        macro_stress_x_downside_beta = torch.zeros(
            (t_len, n_assets), dtype=compute_dtype, device=dev
        )

    # -------------------------------------------------------------------------
    # 5) Multi-Regime Rotation, Idiosyncratic Vol-Surge, CN Limit/Halt & MNAR (#195 #5B)
    # -------------------------------------------------------------------------
    # 5a) Velocidad y aceleración de rango transversal (1 sola llamada batcheada a gaussian_rank)
    ok_r5 = traded & torch.isfinite(ret_5)
    ok_r21 = traded & torch.isfinite(ret_21)
    ok_r63 = traded & torch.isfinite(ret_63)
    ret_h_stack = torch.stack(
        [
            torch.where(ok_r5, ret_5, 0.0),
            torch.where(ok_r21, ret_21, 0.0),
            torch.where(ok_r63, ret_63, 0.0),
        ],
        dim=-1,
    )
    ret_h_mask = torch.stack([ok_r5, ok_r21, ok_r63], dim=-1)
    g_ret_h = gaussian_rank_features(ret_h_stack, mask=ret_h_mask, clip_val=LIMIT).to(compute_dtype)
    g_ret_5, g_ret_21, g_ret_63 = g_ret_h[..., 0], g_ret_h[..., 1], g_ret_h[..., 2]
    rank_vel_5_21 = torch.where(ok_r5 & ok_r21, g_ret_5 - g_ret_21, 0.0)
    rank_accel_5_21_63 = torch.where(
        ok_r5 & ok_r21 & ok_r63,
        (g_ret_5 - g_ret_21) - (g_ret_21 - g_ret_63),
        0.0,
    )

    # 5b) Interacción reversión 5d × surge de volatilidad bipower 5d/21d (guardia bp_cnt >= 3!)
    bp_cnt_5 = _rolling_sum_2d(bp_fin.to(compute_dtype), 5, assume_finite=True)
    bv_daily_5 = (
        (0.5 * math.pi)
        * _rolling_sum_2d(torch.where(bp_fin, bp_prod, 0.0), 5, assume_finite=True)
        / bp_cnt_5.clamp_min(1.0)
    )
    vol_surge_ok = ok_r5 & (bp_cnt_5 >= 3.0) & (bp_cnt >= 3.0) & (bv_daily > 1e-12)
    bv_ratio_5_21 = torch.where(
        vol_surge_ok,
        torch.sqrt(bv_daily_5.clamp_min(0.0) / bv_daily.clamp_min(1e-12)).clamp(0.0, 10.0),
        0.0,
    )
    vs_cnt = vol_surge_ok.to(compute_dtype).sum(dim=1, keepdim=True)
    vs_mean = (
        torch.where(vol_surge_ok, bv_ratio_5_21, 0.0).sum(dim=1, keepdim=True)
        / vs_cnt.clamp_min(1.0)
    )
    vs_diff = torch.where(vol_surge_ok, bv_ratio_5_21 - vs_mean, 0.0)
    vs_std = torch.sqrt(
        ((vs_diff * vs_diff).sum(dim=1, keepdim=True) / vs_cnt.clamp_min(1.0)).clamp_min(0.0)
    )
    idio_vol_surge_z = torch.where(
        vol_surge_ok & (vs_cnt >= 2.0) & (vs_std > 1e-6),
        (bv_ratio_5_21 - vs_mean) / vs_std.clamp_min(1e-6),
        0.0,
    ).clamp(-4.0, 4.0)
    peer_rev_5_x_idio_vol_surge = torch.where(
        vol_surge_ok,
        -peer_resid_mom_5 * idio_vol_surge_z,
        0.0,
    )

    # 5c) Racha causal firmada de límites ±10% CN en 5d (umbral simple ±9.5%, enmascarado a traded)
    ret_simple = torch.where(r_fin, torch.expm1(ret_cc.clamp(-10.0, 10.0)), 0.0)
    hit_up = traded & r_fin & (ret_simple >= 0.095)
    hit_dn = traded & r_fin & (ret_simple <= -0.095)
    limit_hit_daily = hit_up.to(compute_dtype) - hit_dn.to(compute_dtype)
    limit_hit_streak_5 = torch.where(
        traded,
        _rolling_sum_2d(limit_hit_daily, 5, assume_finite=True),
        0.0,
    )

    # 5d) Decaimiento post-suspensión `exp(-days_since_halt / 5.0)` solo en sesiones reabiertas
    #     (pre-IPO sin precio ni historial previo NO cuenta como halt!)
    traded_i = traded.to(torch.int64)
    has_traded_before = (torch.cumsum(traded_i, dim=0) - traded_i) > 0
    is_halt = (~traded) & (px_ok | has_traded_before)
    t_idx = torch.arange(t_len, device=dev, dtype=torch.int64).unsqueeze(1).expand(t_len, n_assets)
    halt_t = torch.where(is_halt, t_idx, torch.full_like(t_idx, -1))
    last_halt_t = torch.cummax(halt_t, dim=0).values
    reopened = traded & (last_halt_t >= 0)
    days_since_halt = (t_idx - last_halt_t).clamp_min(1).to(compute_dtype)
    post_halt_age_decay = torch.where(
        reopened,
        torch.exp(-days_since_halt / 5.0),
        0.0,
    )

    # 5e) Choque MNAR de disponibilidad auxiliar `avail_rate_t - ema_21(avail_rate)`
    avail_surprise_21 = _compute_avail_surprise_21(
        fund_valid if fundamentals is not None else None,
        news_ok_opt,
        ch_ok_opt,
        traded=traded,
        compute_dtype=compute_dtype,
    )

    mat = torch.stack(
        [
            bv_21,
            jump_var_ratio_21,
            vol_of_vol_63,
            realized_skew_signed_21,
            peer_resid_mom_5,
            peer_resid_mom_21,
            peer_resid_mom_63,
            liq_cond_rev_5,
            signed_abn_turn_21,
            kaufman_er_21,
            amihud_asym_up_down_21,
            price_vs_news_divergence,
            price_vs_fund_quality_gap,
            chart_vs_vol_exhaustion,
            macro_stress_x_downside_beta,
            pit_fund_quality,
            pit_fund_value,
            pit_fund_profitability,
            rank_vel_5_21,
            rank_accel_5_21_63,
            peer_rev_5_x_idio_vol_surge,
            limit_hit_streak_5,
            post_halt_age_decay,
            avail_surprise_21,
        ],
        dim=-1,
    )
    mat = torch.nan_to_num(mat, nan=0.0, posinf=LIMIT, neginf=-LIMIT).clamp(-LIMIT, LIMIT)
    out = mat.to(dtype=torch.float32)
    return out.squeeze(1) if squeeze_1d else out


# =============================================================================
# 4. Auditoría de Calidad, Causalidad Bit-Exact, Fuga <0.10 y Estrés de Ausencia
# =============================================================================
def _newey_west_tstat_tensor(series: torch.Tensor, max_lag: int = 5) -> float:
    """Calcula t-stat HAC (Newey-West con núcleo Bartlett) sobre serie 1D con posibles NaNs."""
    ok = torch.isfinite(series)
    n_obs = int(ok.sum())
    if n_obs < 3:
        return 0.0
    # Compactar sin `.nonzero()` ni `.cpu()` usando orden estable por validez
    order = torch.argsort((~ok).to(torch.int8), stable=True)
    vals = series[order][:n_obs].to(torch.float64)
    mean_v = vals.mean()
    xc = vals - mean_v
    gamma0 = (xc * xc).sum() / float(n_obs)
    lag = min(max_lag, n_obs - 1)
    lrv = gamma0
    for k in range(1, lag + 1):
        w_k = 1.0 - float(k) / float(lag + 1)
        g_k = (xc[k:] * xc[:-k]).sum() / float(n_obs)
        lrv = lrv + 2.0 * w_k * g_k
    var_mean = float(lrv.clamp_min(0.0)) / float(n_obs)
    if var_mean <= 1e-24:
        return 0.0
    return float(mean_v) / math.sqrt(var_mean)


def _cross_sectional_pearson_series(
    feat: torch.Tensor,
    fut_ret: torch.Tensor,
) -> torch.Tensor:
    """Correlación de Pearson transversal por sesión `[T, F]` en FP64."""
    x_tfn = feat.permute(0, 2, 1).to(torch.float64)
    y_tfn = fut_ret.unsqueeze(1).expand_as(x_tfn).to(torch.float64)
    ok = torch.isfinite(x_tfn) & torch.isfinite(y_tfn)
    cnt = ok.sum(dim=-1, keepdim=True)
    denom = cnt.clamp_min(1).to(torch.float64)
    mx = torch.where(ok, x_tfn, 0.0).sum(dim=-1, keepdim=True) / denom
    my = torch.where(ok, y_tfn, 0.0).sum(dim=-1, keepdim=True) / denom
    dx = torch.where(ok, x_tfn - mx, 0.0)
    dy = torch.where(ok, y_tfn - my, 0.0)
    cov = (dx * dy).sum(dim=-1)
    vx = (dx * dx).sum(dim=-1)
    vy = (dy * dy).sum(dim=-1)
    valid_ic = (cnt.squeeze(-1) >= 3) & (vx > 1e-12) & (vy > 1e-12)
    return torch.where(valid_ic, cov / torch.sqrt((vx * vy).clamp_min(1e-24)), float("nan"))


def _cross_sectional_rank_ic_series(
    feat: torch.Tensor,
    fut_ret: torch.Tensor,
) -> torch.Tensor:
    """Serie temporal `[T, F]` de Rank IC (Spearman exacto con empates) en FP64."""
    x_tfn = feat.permute(0, 2, 1).to(torch.float64)
    y_tfn = fut_ret.unsqueeze(1).expand_as(x_tfn).to(torch.float64)
    ok = torch.isfinite(x_tfn) & torch.isfinite(y_tfn)
    cnt = ok.sum(dim=-1, keepdim=True)
    denom = cnt.clamp_min(1).to(torch.float64)

    def _mid_ranks(vals: torch.Tensor) -> torch.Tensor:
        n_a = vals.shape[-1]
        asc = torch.argsort(torch.where(ok, vals, float("inf")), dim=-1, stable=True)
        desc = torch.argsort(torch.where(ok, -vals, float("inf")), dim=-1, stable=True)
        pos = torch.arange(n_a, dtype=torch.float64, device=vals.device).view(1, 1, n_a)
        pos_exp = pos.expand_as(vals)
        r_asc = torch.zeros_like(vals)
        r_desc = torch.zeros_like(vals)
        r_asc.scatter_(-1, asc, pos_exp)
        r_desc.scatter_(-1, desc, pos_exp)
        return torch.where(ok, 0.5 * (r_asc + (denom - 1.0 - r_desc)), 0.0)

    rx = _mid_ranks(x_tfn)
    ry = _mid_ranks(y_tfn)
    mx = rx.sum(dim=-1, keepdim=True) / denom
    my = ry.sum(dim=-1, keepdim=True) / denom
    dx = torch.where(ok, rx - mx, 0.0)
    dy = torch.where(ok, ry - my, 0.0)
    cov = (dx * dy).sum(dim=-1)
    vx = (dx * dx).sum(dim=-1)
    vy = (dy * dy).sum(dim=-1)
    valid_ic = (cnt.squeeze(-1) >= 3) & (vx > 1e-12) & (vy > 1e-12)
    return torch.where(valid_ic, cov / torch.sqrt((vx * vy).clamp_min(1e-24)), float("nan"))


def audit_dataset_enrichment_quality(
    features: torch.Tensor | np.ndarray,
    future_returns: torch.Tensor | np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    min_ic: float = 0.005,
    leakage_ic_threshold: float = 0.10,
    redundancy_threshold: float = 0.70,
    min_std: float = 1e-6,
    ohlcv: tuple[
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray,
    ]
    | None = None,
    modalities: Mapping[str, Any] | None = None,
    causality_fn: Callable[[], tuple[torch.Tensor, torch.Tensor, int]] | None = None,
    run_stress: bool = True,
    clip_val: float = LIMIT,
) -> dict[str, Any]:
    """Auditoría integral SOTA de calidad de enriquecimiento, causalidad bit-exact y ausencias.

    Ejecuta `causal_feature_quality_audit` (`institutional_alpha_v5.py`), aplica pantalla de fuga
    `max|corr(X_t, y_{t+1})| < leakage_ic_threshold`, reporta IC, HAC t-stat (Newey-West),
    redundancia, cobertura de ausencias, tasa de outliers `|z| > 5`, causalidad bit a bit
    (`diff == 0.0` en `<= t_cut`) y estrés `50% / 90% / 100%` ausente con cero fuga de gradiente.
    """
    feat = _as_float_tensor(features, dtype=torch.float64)
    dev = feat.device
    y = _as_float_tensor(future_returns, dtype=torch.float64, device=dev)
    if y.ndim == 3 and y.shape[-1] == 1:
        y = y.squeeze(-1)

    t_len, n_assets, n_feats = feat.shape
    if feature_names is None:
        names = (
            ENRICHED_SOTA_NAMES
            if n_feats == len(ENRICHED_SOTA_NAMES)
            else tuple(f"feat_{i}" for i in range(n_feats))
        )
    else:
        names = tuple(str(n) for n in feature_names)

    base_rep = causal_feature_quality_audit(
        feat,
        y,
        feature_names=names,
        min_ic=min_ic,
        leakage_ic_threshold=leakage_ic_threshold,
        redundancy_threshold=redundancy_threshold,
        min_std=min_std,
    )

    # Pearson IC transversal + HAC Newey-West t-stat exacto sobre Rank IC por rasgo
    pearson_tf = _cross_sectional_pearson_series(feat, y)  # [T, F]
    rank_ic_tf = _cross_sectional_rank_ic_series(feat, y)  # [T, F]
    mean_pearson_ic: dict[str, float] = {}
    ic_hac_t_stat: dict[str, float] = {}
    missingness_coverage: dict[str, dict[str, float]] = {}
    outlier_rates_5sigma: dict[str, float] = {}

    total_slots = max(t_len * n_assets, 1)
    for i, nm in enumerate(names):
        col_p = pearson_tf[:, i]
        ok_p = torch.isfinite(col_p)
        n_ok_p = int(ok_p.sum())
        m_p = float(torch.where(ok_p, col_p, 0.0).sum()) / float(max(n_ok_p, 1)) if n_ok_p else 0.0
        mean_pearson_ic[nm] = m_p
        ic_hac_t_stat[nm] = _newey_west_tstat_tensor(rank_ic_tf[:, i])

        f_col = feat[:, :, i]
        fin_c = torch.isfinite(f_col)
        fin_rate = float(fin_c.sum()) / float(total_slots)
        nonzero_rate = float((fin_c & (f_col.abs() > 1e-12)).sum()) / float(total_slots)
        missingness_coverage[nm] = dict(
            finite_rate=fin_rate,
            missing_rate=1.0 - fin_rate,
            active_nonzero_rate=nonzero_rate,
        )
        n_fin = int(fin_c.sum())
        if n_fin >= 3:
            f_safe = torch.where(fin_c, f_col, 0.0)
            mu_c = f_safe.sum() / float(n_fin)
            diff_c = torch.where(fin_c, f_col - mu_c, 0.0)
            sd_c = math.sqrt(max(float((diff_c * diff_c).sum()) / float(n_fin), 0.0))
            out_r = (
                float((fin_c & (diff_c.abs() > 5.0 * sd_c)).sum()) / float(n_fin)
                if sd_c > 1e-12
                else 0.0
            )
        else:
            out_r = 0.0
        outlier_rates_5sigma[nm] = out_r

    max_abs_rank_ic = max((abs(v) for v in base_rep["mean_rank_ic"].values()), default=0.0)
    max_abs_pearson_ic = max((abs(v) for v in mean_pearson_ic.values()), default=0.0)
    leakage_screen_passed = bool(
        max_abs_rank_ic < leakage_ic_threshold
        and max_abs_pearson_ic < leakage_ic_threshold
        and len(base_rep["leakage_suspects"]) == 0
    )

    # 1) Verificación de causalidad bit a bit (`t > t_cut` mutado => diff == 0.0 en `<= t_cut`)
    causality_check: dict[str, Any] = dict(
        evaluated=False, bit_exact_causal=True, max_abs_diff=0.0, t_cut=0
    )
    if causality_fn is not None:
        a_out, b_out, t_cut = causality_fn()
        diff_pre = float((a_out[: t_cut + 1] - b_out[: t_cut + 1]).abs().max())
        causality_check = dict(
            evaluated=True,
            bit_exact_causal=bool(torch.equal(a_out[: t_cut + 1], b_out[: t_cut + 1])),
            max_abs_diff=diff_pre,
            t_cut=int(t_cut),
        )
    elif ohlcv is not None and t_len >= 4:
        o0, h0, l0, c0, v0 = (
            _as_float_tensor(arr, dtype=torch.float32, device=dev) for arr in ohlcv
        )
        t_cut = max(1, (2 * t_len) // 3)
        mod_kw = dict(modalities) if modalities is not None else {}
        f_orig = enrich_full_dataset_sota_torch(o0, h0, l0, c0, v0, **mod_kw)
        o_m, h_m, l_m, c_m, v_m = o0.clone(), h0.clone(), l0.clone(), c0.clone(), v0.clone()
        c_m[t_cut + 1 :] = c_m[t_cut + 1 :] * 3.5
        h_m[t_cut + 1 :] = h_m[t_cut + 1 :] * 4.0
        l_m[t_cut + 1 :] = l_m[t_cut + 1 :] * 0.3
        v_m[t_cut + 1 :] = v_m[t_cut + 1 :] * 12.0
        mod_mut: dict[str, Any] = {}
        for mk, mv in mod_kw.items():
            if isinstance(mv, (torch.Tensor, np.ndarray)):
                is_bool = (isinstance(mv, torch.Tensor) and mv.dtype == torch.bool) or (
                    isinstance(mv, np.ndarray) and mv.dtype == np.bool_
                )
                if is_bool:
                    mt = _as_bool_tensor(mv, device=dev).clone()
                    if mt.ndim >= 1 and mt.shape[0] == t_len:
                        mt[t_cut + 1 :] = ~mt[t_cut + 1 :]
                elif "age" in mk.lower():
                    mt = _as_float_tensor(mv, dtype=torch.float32, device=dev).clone()
                    if mt.ndim >= 1 and mt.shape[0] == t_len:
                        mt[t_cut + 1 :] = mt[t_cut + 1 :].clamp_min(0.0) + 100.0
                else:
                    mt = _as_float_tensor(mv, dtype=torch.float32, device=dev).clone()
                    if mt.ndim >= 1 and mt.shape[0] == t_len:
                        mt[t_cut + 1 :] = -mt[t_cut + 1 :] * 2.5 - 3.0
                mod_mut[mk] = mt
            else:
                mod_mut[mk] = mv
        f_mut = enrich_full_dataset_sota_torch(o_m, h_m, l_m, c_m, v_m, **mod_mut)
        diff_pre = float((f_orig[: t_cut + 1] - f_mut[: t_cut + 1]).abs().max())
        causality_check = dict(
            evaluated=True,
            bit_exact_causal=bool(torch.equal(f_orig[: t_cut + 1], f_mut[: t_cut + 1])),
            max_abs_diff=diff_pre,
            t_cut=int(t_cut),
        )

    # 2) Estrés de ausencias 50% / 90% / 100% + verificación de fuga de gradiente == 0.0
    missingness_stress: dict[str, Any] = dict(evaluated=False, all_passed=True, levels={})
    if run_stress and t_len > 0 and n_assets > 0:
        featurizer = MissingnessResilientFeaturizer(clip_val=clip_val).to(device=dev)
        featurizer.eval()
        sample_x = feat[-1].to(dtype=torch.float32)  # [N, F]
        gen = torch.Generator(device=dev).manual_seed(2026)
        levels_out: dict[str, dict[str, Any]] = {}
        all_ok = True
        for drop_p in (0.5, 0.9, 1.0):
            if drop_p >= 1.0:
                miss_m = torch.ones_like(sample_x, dtype=torch.bool)
            else:
                miss_m = torch.rand(sample_x.shape, device=dev, generator=gen) < drop_p
            dirty = torch.where(miss_m, float("nan"), sample_x).detach().requires_grad_(True)
            res_s = featurizer(dirty, availability_mask=~miss_m)
            out_s = res_s["features"]
            out_det = out_s.detach()
            fin_ok = bool(torch.isfinite(out_det).all())
            min_v = float(out_det.min()) if out_det.numel() else 0.0
            max_v = float(out_det.max()) if out_det.numel() else 0.0
            bounds_ok = bool(min_v >= -clip_val - 1e-5 and max_v <= clip_val + 1e-5)
            (out_s * out_s).sum().backward()
            grad = dirty.grad
            grad_fin = bool(grad is not None and torch.isfinite(grad).all())
            max_nan_grad = (
                float(torch.where(miss_m, grad.abs(), 0.0).max())
                if (grad is not None and bool(miss_m.any()))
                else 0.0
            )
            passed_lvl = fin_ok and bounds_ok and grad_fin and (max_nan_grad == 0.0)
            all_ok = all_ok and passed_lvl
            levels_out[str(drop_p)] = dict(
                all_finite=fin_ok,
                within_bounds=bounds_ok,
                min_val=min_v,
                max_val=max_v,
                grad_finite=grad_fin,
                max_nan_slot_grad=max_nan_grad,
                passed=passed_lvl,
            )
        missingness_stress = dict(evaluated=True, all_passed=all_ok, levels=levels_out)

    merged = dict(base_rep)
    merged.update(
        mean_pearson_ic=mean_pearson_ic,
        ic_hac_t_stat=ic_hac_t_stat,
        max_abs_rank_ic=max_abs_rank_ic,
        max_abs_pearson_ic=max_abs_pearson_ic,
        leakage_screen_passed=leakage_screen_passed,
        missingness_coverage=missingness_coverage,
        outlier_rates_5sigma=outlier_rates_5sigma,
        causality_check=causality_check,
        missingness_stress=missingness_stress,
        overall_passed=bool(
            base_rep["overall_passed"]
            and leakage_screen_passed
            and causality_check["bit_exact_causal"]
            and missingness_stress["all_passed"]
        ),
    )
    return merged


def audit_laboratorio_cache_readonly(
    cache_dir: str | Path,
    *,
    max_rows_sample: int = 200_000,
) -> dict[str, Any]:
    """Audita en modo estrictamente solo lectura una caché `< 2024` en `laboratorio/`."""
    cdir = Path(cache_dir)
    if not (cdir / "X.npy").is_file() or not (cdir / "day.npy").is_file():
        raise FileNotFoundError(f"Caché incompleta o inexistente: {cdir}")

    days = np.load(cdir / "day.npy", mmap_mode="r")
    max_day = int(np.max(days)) if days.size else 0
    min_day = int(np.min(days)) if days.size else 0
    if max_day >= _PRE_2024_EPOCH_DAY_LIMIT:
        raise ValueError(
            f"Partición 2024+ sellada detectada en {cdir} (max_day={max_day} >= "
            f"{_PRE_2024_EPOCH_DAY_LIMIT}): auditoría rechazada"
        )

    x_mm = np.load(cdir / "X.npy", mmap_mode="r")
    y_mm = np.load(cdir / "y.npy", mmap_mode="r")
    bounds = np.load(cdir / "bounds.npy", mmap_mode="r")
    sizes = np.diff(bounds)

    # Lectura contigua por bloques distribuidos cronológicamente (evita 50k page-faults aleatorios)
    n_rows = int(x_mm.shape[0])
    target_rows = min(n_rows, max(int(max_rows_sample), 1))
    n_slabs = min(16, max(1, target_rows // 256))
    slab_len = max(1, target_rows // n_slabs)
    max_start = max(0, n_rows - slab_len)
    starts = (
        np.linspace(0, max_start, num=n_slabs, dtype=np.int64)
        if n_slabs > 1
        else np.asarray([0], dtype=np.int64)
    )
    xs = np.concatenate(
        [np.asarray(x_mm[int(s) : int(s) + slab_len], dtype=np.float64) for s in starts], axis=0
    )
    ys = np.concatenate(
        [np.asarray(y_mm[int(s) : int(s) + slab_len], dtype=np.float64) for s in starts], axis=0
    )

    nan_rate_x = float(np.mean(~np.isfinite(xs)))
    zero_rate_x = float(np.mean(xs == 0.0))
    kurt_list: list[float] = []
    gt5_list: list[float] = []
    for j in range(xs.shape[1]):
        col = xs[:, j]
        ok = np.isfinite(col)
        if int(ok.sum()) > 10:
            v = col[ok]
            sd = float(np.std(v))
            if sd > 1e-12:
                z = (v - float(np.mean(v))) / sd
                kurt_list.append(float(np.mean(z**4) - 3.0))
                gt5_list.append(float(np.mean(np.abs(z) > 5.0)))

    x3_nan_rate: float | None = None
    x3_shape: tuple[int, ...] | None = None
    if (cdir / "X3.npy").is_file():
        x3_mm = np.load(cdir / "X3.npy", mmap_mode="r")
        x3_shape = tuple(int(d) for d in x3_mm.shape)
        x3s = np.concatenate(
            [np.asarray(x3_mm[int(s) : int(s) + slab_len], dtype=np.float64) for s in starts],
            axis=0,
        )
        x3_nan_rate = float(np.mean(~np.isfinite(x3s)))

    names_path = cdir / "names.json"
    n_named = 0
    if names_path.is_file():
        nj = json.loads(names_path.read_text(encoding="utf-8"))
        if isinstance(nj, dict):
            n_named = len(nj.get("names", nj.get("features", [])))
        elif isinstance(nj, list):
            n_named = len(nj)

    return dict(
        cache_dir=str(cdir),
        strictly_pre_2024=True,
        min_day_epoch=min_day,
        max_day_epoch=max_day,
        n_rows=int(x_mm.shape[0]),
        n_features_x=int(x_mm.shape[1]),
        n_named=n_named,
        n_sessions=int(len(bounds) - 1),
        session_size_min=int(sizes.min()) if sizes.size else 0,
        session_size_median=int(np.median(sizes)) if sizes.size else 0,
        session_size_max=int(sizes.max()) if sizes.size else 0,
        x_nan_rate=nan_rate_x,
        x_exact_zero_rate=zero_rate_x,
        x_median_excess_kurtosis=float(np.median(kurt_list)) if kurt_list else 0.0,
        x_max_excess_kurtosis=float(np.max(kurt_list)) if kurt_list else 0.0,
        x_max_gt5sigma_rate=float(np.max(gt5_list)) if gt5_list else 0.0,
        x3_shape=x3_shape,
        x3_nan_rate=x3_nan_rate,
        y_nan_rate=float(np.mean(~np.isfinite(ys))),
        y_std=float(np.nanstd(ys)),
    )


__all__ = [
    "CN_EMPIRICAL_MISSINGNESS_PROFILE",
    "ENCODED_MODALITY_ORDER",
    "ENCODED_TO_MODEL_MODALITY_INDEX",
    "ENRICHED_SOTA_DIM",
    "ENRICHED_SOTA_NAMES",
    "LIMIT",
    "MODEL_MODALITY_ORDER",
    "MissingnessResilientFeaturizer",
    "StructuredModalityDropout",
    "US_EMPIRICAL_MISSINGNESS_PROFILE",
    "audit_dataset_enrichment_quality",
    "audit_laboratorio_cache_readonly",
    "enrich_full_dataset_sota_torch",
]
