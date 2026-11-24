"""Física de mercado y rasgos multimodales causales v4 (`market_physics_v4`).

Extiende `long_horizon_v2` con microestructura diaria extraída de OHLCV, volatilidad rugosa
(Gatheral), diferenciación fraccionaria (López de Prado), firmas de trayectorias (Path Signatures
de orden 2), espectro de Teoría de Matrices Aleatorias (Marchenko-Pastur, Bouchaud), distancia
de transporte óptimo de Wasserstein para regímenes, grafo dirigido causal Lead-Lag, sorpresas y
betas macro dinámicas, y fusión multimodal con novedad semántica y máscaras de presencia.

Invariante estricta: toda fila en `t` depende únicamente de información disponible en `<= t`
(y en cálculos transversales con retornos realizados, estrictamente `<= t - maturity_lag`).
"""

from __future__ import annotations

import math

import numpy as np

LIMIT = 8.0

ASSET_V4_NAMES = (
    "vol_garman_klass_21",
    "vol_garman_klass_63",
    "vol_rogers_satchell_21",
    "vol_rogers_satchell_63",
    "vol_yang_zhang_21",
    "vol_yang_zhang_63",
    "rough_hurst_21",
    "rough_hurst_63",
    "vol_of_vol_21",
    "corwin_schultz_spread_21",
    "abdi_ranaldo_spread_21",
    "amihud_up_21",
    "amihud_down_21",
    "amihud_asym_21",
    "signed_volume_clv_5",
    "signed_volume_clv_21",
    "price_vol_corr_21",
    "realized_skew_21",
    "realized_kurt_21",
    "ffd_log_price_d035",
    "ffd_log_price_d050",
    "sig2_levy_pv_5",
    "sig2_levy_pv_21",
    "sig2_levy_pr_21",
)

MULTIMODAL_V4_NAMES = (
    "news_novelty",
    "news_presence",
    "news_staleness_log",
    "news_intensity_ewma",
    "chart_drift",
    "chart_presence",
    "fund_delta_norm",
    "fund_presence",
    "fund_staleness_log",
    "macro_tailwind",
    "macro_alignment_abs",
    "modality_completeness",
)


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x, nan=0.0), dtype=np.float64)])
    idx = np.arange(1, len(x) + 1)
    return c[idx] - c[np.maximum(idx - window, 0)]


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    valid = np.isfinite(x).astype(np.float64)
    cnt = _rolling_sum(valid, window)
    tot = _rolling_sum(np.where(np.isfinite(x), x, 0.0), window)
    return np.where(cnt > 0.0, tot / np.maximum(cnt, 1.0), 0.0)


def _rolling_var(x: np.ndarray, window: int) -> np.ndarray:
    valid = np.isfinite(x).astype(np.float64)
    cnt = _rolling_sum(valid, window)
    xv = np.where(np.isfinite(x), x, 0.0)
    s1 = _rolling_sum(xv, window)
    s2 = _rolling_sum(xv * xv, window)
    mean = s1 / np.maximum(cnt, 1.0)
    var = np.maximum(s2 / np.maximum(cnt, 1.0) - mean * mean, 0.0)
    return np.where(cnt >= 2.0, var * cnt / np.maximum(cnt - 1.0, 1.0), 0.0)


def _ffd_weights(d: float, thresh: float = 1e-4, max_k: int = 64) -> np.ndarray:
    """Pesos de diferenciación fraccionaria de ancho fijo (López de Prado, AFML cap. 5)."""
    w = [1.0]
    for k in range(1, max_k):
        wk = -w[-1] * (d - k + 1.0) / k
        if abs(wk) < thresh:
            break
        w.append(wk)
    return np.asarray(w, dtype=np.float64)


def _apply_ffd(series: np.ndarray, d: float) -> np.ndarray:
    """Convolución causal de diferenciación fraccionaria (sin fuga)."""
    w = _ffd_weights(d)
    n = len(series)
    out = np.zeros(n, dtype=np.float64)
    k = len(w)
    if n == 0:
        return out
    # Convolución causal: out[t] = sum_{j=0}^{k-1} w[j] * series[t - j]
    conv = np.convolve(series, w, mode="full")[:n]
    out[k - 1 :] = conv[k - 1 :]
    # Tipificación causal por desviación rodante de 63 sesiones
    std = np.sqrt(_rolling_var(out, 63))
    return np.where(std > 1e-12, out / np.maximum(std, 1e-12), 0.0)


def path_signature_order2(path: np.ndarray) -> np.ndarray:
    """Firma truncada exacta de orden 1 y 2 de una trayectoria `(L, D)`.

    Devuelve vector de dimensión `D + D*D`:
      - `[0:D]` incrementos de orden 1: `X_T - X_0`
      - `[D:]` matriz aplanada de integrales iteradas `S^{i,j}` (identidad de Chen).
    La parte antisimétrica `S^{i,j} - S^{j,i}` es el área firmada de Lévy.
    """
    x = np.asarray(path, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2:
        d = x.shape[1] if x.ndim == 2 else 1
        return np.zeros(d + d * d, dtype=np.float64)
    dx = np.diff(x, axis=0)  # (L-1, D)
    s1 = dx.sum(axis=0)
    # Estado acumulado antes de cada incremento (X_t - X_0)
    prefix = np.zeros_like(dx)
    if len(dx) > 1:
        prefix[1:] = np.cumsum(dx[:-1], axis=0)
    # S^{i,j} = sum_t prefix[t, i] * dx[t, j] + 0.5 * sum_t dx[t, i] * dx[t, j]
    s2 = prefix.T @ dx + 0.5 * (dx.T @ dx)
    return np.concatenate([s1, s2.ravel()])


def _rolling_levy_area(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Área firmada de Lévy normalizada en ventana causal `window` entre dos series `x` e `y`.

    Calculada en O(N) exacto usando sumas acumuladas:
      `Levy(t) = sum_{u=t-W+2..t} [(x_{u-1} - x_{t-W+1}) dy_u - (y_{u-1} - y_{t-W+1}) dx_u]`
    """
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    if n < window or window < 2:
        return out
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)
    dx[1:] = x[1:] - x[:-1]
    dy[1:] = y[1:] - y[:-1]
    x_prev = np.zeros(n, dtype=np.float64)
    y_prev = np.zeros(n, dtype=np.float64)
    x_prev[1:] = x[:-1]
    y_prev[1:] = y[:-1]

    # sum_{u} x_{u-1} dy_u - y_{u-1} dx_u  menos términos de anclaje al inicio de la ventana
    cross = x_prev * dy - y_prev * dx
    w_eff = window - 1
    c_cross = _rolling_sum(cross, w_eff)
    c_dx = _rolling_sum(dx, w_eff)
    c_dy = _rolling_sum(dy, w_eff)
    start_idx = np.maximum(np.arange(n) - w_eff, 0)
    raw = c_cross - x[start_idx] * c_dy + y[start_idx] * c_dx
    # Normalizar por energía de variación para invarianza de escala
    norm_x = np.sqrt(_rolling_sum(dx * dx, w_eff))
    norm_y = np.sqrt(_rolling_sum(dy * dy, w_eff))
    denom = norm_x * norm_y
    d_slice = denom[window - 1 :]
    out[window - 1 :] = np.where(
        d_slice > 1e-12, raw[window - 1 :] / np.maximum(d_slice, 1e-12), 0.0
    )
    return out


def _rough_hurst_local(log_vol: np.ndarray, window: int) -> np.ndarray:
    """Estimador causal del exponente de Hurst local de log-volatilidad (Gatheral et al.).

    Usa cociente de varianzas de incrementos en retardos q=1 y q=4:
      `H_hat = 0.5 * log(m_4 / m_1) / log(4)` acotado a `[0.02, 0.98]`.
    """
    n = len(log_vol)
    out = np.full(n, 0.5, dtype=np.float64)
    if n < max(window, 6):
        return out
    d1 = np.zeros(n, dtype=np.float64)
    d4 = np.zeros(n, dtype=np.float64)
    d1[1:] = (log_vol[1:] - log_vol[:-1]) ** 2
    d4[4:] = (log_vol[4:] - log_vol[:-4]) ** 2
    m1 = _rolling_mean(d1, window)
    m4 = _rolling_mean(d4, window)
    valid = (m1 > 1e-12) & (m4 > 1e-12)
    ratio = np.maximum(m4, 1e-12) / np.maximum(m1, 1e-12)
    h_est = np.where(valid, 0.5 * np.log(ratio) / math.log(4.0), 0.5)
    out[window:] = np.clip(h_est[window:], 0.02, 0.98)
    return out


def _rolling_window_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Correlación de Pearson causal exacta sobre la ventana `[t-window+1 .. t]` en `[-1, 1]`."""
    n = len(x)
    if n == 0 or window < 2:
        return np.zeros(n, dtype=np.float64)
    pad = np.full(window - 1, np.nan, dtype=np.float64)
    xw = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, x]), window)
    yw = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, y]), window)
    ok = np.isfinite(xw) & np.isfinite(yw)
    cnt = ok.sum(axis=1)
    denom_cnt = np.maximum(cnt[:, None], 1)
    mx = np.sum(np.where(ok, xw, 0.0), axis=1, keepdims=True) / denom_cnt
    my = np.sum(np.where(ok, yw, 0.0), axis=1, keepdims=True) / denom_cnt
    dx = np.where(ok, xw - mx, 0.0)
    dy = np.where(ok, yw - my, 0.0)
    sxx = np.sum(dx * dx, axis=1)
    syy = np.sum(dy * dy, axis=1)
    sxy = np.sum(dx * dy, axis=1)
    denom = np.sqrt(sxx * syy)
    return np.where(
        (cnt >= 2) & (denom > 1e-12),
        np.clip(sxy / np.maximum(denom, 1e-12), -1.0, 1.0),
        0.0,
    )


def _rolling_window_skew_kurt(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Asimetría y curtosis (exceso) centrales exactas sobre la ventana causal `[t-W+1 .. t]`."""
    n = len(x)
    if n == 0 or window < 3:
        z = np.zeros(n, dtype=np.float64)
        return z, z.copy()
    pad = np.full(window - 1, np.nan, dtype=np.float64)
    xw = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, x]), window)
    ok = np.isfinite(xw)
    cnt = ok.sum(axis=1)
    denom_cnt = np.maximum(cnt, 1)
    mx = np.sum(np.where(ok, xw, 0.0), axis=1, keepdims=True) / denom_cnt[:, None]
    dx = np.where(ok, xw - mx, 0.0)
    dx2 = dx * dx
    m2 = np.sum(dx2, axis=1) / denom_cnt
    m3 = np.sum(dx2 * dx, axis=1) / denom_cnt
    m4 = np.sum(dx2 * dx2, axis=1) / denom_cnt
    std = np.sqrt(np.maximum(m2, 0.0))
    valid = (cnt >= 3) & (std > 1e-12)
    skew = np.where(valid, m3 / np.maximum(std**3, 1e-36), 0.0)
    kurt = np.where(valid, m4 / np.maximum(m2 * m2, 1e-36) - 3.0, 0.0)
    return skew, kurt


def detect_unadjusted_split_jumps(
    open_: np.ndarray,
    close: np.ndarray,
    *,
    threshold: float = 0.55,
) -> dict[str, object]:
    """Audita saltos overnight extremos `|log(O_t / C_{t-1})| >= threshold` típicos de splits."""
    o = np.asarray(open_, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    if o.shape != c.shape or o.shape[0] < 2:
        return dict(n_jumps=0, max_abs_overnight_log_jump=0.0, threshold=float(threshold))
    valid = np.isfinite(o[1:]) & np.isfinite(c[:-1]) & (o[1:] > 0.0) & (c[:-1] > 0.0)
    jumps = np.where(
        valid,
        np.abs(np.log(np.maximum(o[1:], 1e-12) / np.maximum(c[:-1], 1e-12))),
        0.0,
    )
    flagged = jumps >= threshold
    return dict(
        n_jumps=int(flagged.sum()),
        max_abs_overnight_log_jump=float(jumps.max()) if jumps.size else 0.0,
        threshold=float(threshold),
    )


def asset_microstructure_v4(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Calcula los 24 rasgos causales de microestructura, volatilidad rugosa y Path Signatures.

    Contrato de entrada:
      - Las series OHLC deben estar **ajustadas por splits y dividendos** (use
        `detect_unadjusted_split_jumps` durante la ingesta para descartar saltos de split).
      - Los días suspendidos o sin negociación (`volume <= 0`) se enmascaran como NaN/neutrales
        en Amihud, volatilidades de rango y flujo de volumen para no saturar Amihud a `8.0`.
      - Todas las entradas son series 1D ordenadas cronológicamente para un activo.
      - Devuelve matriz `(N, 24)` en `float32`, finita y acotada a `[-8, 8]`.
    """
    o = np.maximum(np.asarray(open_, dtype=np.float64), 1e-12)
    h = np.maximum(np.asarray(high, dtype=np.float64), o)
    l_ = np.minimum(np.maximum(np.asarray(low, dtype=np.float64), 1e-12), o)
    c = np.maximum(np.asarray(close, dtype=np.float64), 1e-12)
    h = np.maximum(h, c)
    l_ = np.minimum(l_, c)
    v_raw = np.asarray(volume, dtype=np.float64)
    traded = np.isfinite(v_raw) & (v_raw > 0.0)
    v = np.maximum(v_raw, 0.0)
    n = len(c)
    if n == 0:
        return np.empty((0, len(ASSET_V4_NAMES)), dtype=np.float32)

    log_o = np.log(o)
    log_h = np.log(h)
    log_l = np.log(l_)
    log_c = np.log(c)
    log_v = np.log1p(v)
    log_v_traded = np.where(traded, log_v, np.nan)

    ret_cc = np.zeros(n, dtype=np.float64)
    ret_oc = log_c - log_o
    ret_co = np.zeros(n, dtype=np.float64)
    if n > 1:
        ret_cc[1:] = log_c[1:] - log_c[:-1]
        ret_co[1:] = log_o[1:] - log_c[:-1]

    u = log_h - log_o
    d = log_l - log_o
    hl = log_h - log_l
    co_c = log_c - log_o

    # 1) Garman-Klass, Rogers-Satchell y Yang-Zhang (días volume <= 0 enmascarados como NaN)
    gk_raw = np.maximum(0.5 * hl * hl - (2.0 * math.log(2.0) - 1.0) * co_c * co_c, 0.0)
    rs_raw = np.maximum(u * (u - co_c) + d * (d - co_c), 0.0)
    gk_daily = np.where(traded, gk_raw, np.nan)
    rs_daily = np.where(traded, rs_raw, np.nan)
    ret_oc_t = np.where(traded, ret_oc, np.nan)
    ret_co_t = np.where(traded, ret_co, np.nan)
    gk_21 = np.log(np.sqrt(np.maximum(_rolling_mean(gk_daily, 21), 1e-10) * 252.0) * 100.0 + 1e-6)
    gk_63 = np.log(np.sqrt(np.maximum(_rolling_mean(gk_daily, 63), 1e-10) * 252.0) * 100.0 + 1e-6)
    rs_21 = np.log(np.sqrt(np.maximum(_rolling_mean(rs_daily, 21), 1e-10) * 252.0) * 100.0 + 1e-6)
    rs_63 = np.log(np.sqrt(np.maximum(_rolling_mean(rs_daily, 63), 1e-10) * 252.0) * 100.0 + 1e-6)

    def _yz(win: int) -> np.ndarray:
        k_yz = 0.34 / (1.34 + (win + 1.0) / max(win - 1.0, 1.0))
        v_o = _rolling_var(ret_co_t, win)
        v_c = _rolling_var(ret_oc_t, win)
        v_rs = _rolling_mean(rs_daily, win)
        total = np.maximum(v_o + k_yz * v_c + (1.0 - k_yz) * v_rs, 1e-10)
        return np.log(np.sqrt(total * 252.0) * 100.0 + 1e-6)

    yz_21 = _yz(21)
    yz_63 = _yz(63)

    # 2) Volatilidad rugosa (Gatheral) + vol-of-vol
    daily_log_sigma = np.where(traded, 0.5 * np.log(np.maximum(gk_raw, 1e-10)), np.nan)
    hurst_21 = _rough_hurst_local(daily_log_sigma, 21)
    hurst_63 = _rough_hurst_local(daily_log_sigma, 63)
    vov_21 = np.sqrt(_rolling_var(daily_log_sigma, 21))

    # 3) Spread bid-ask implícito: Corwin-Schultz (2012) & Abdi-Ranaldo (2017)
    beta_cs = np.zeros(n, dtype=np.float64)
    gamma_cs = np.zeros(n, dtype=np.float64)
    if n > 1:
        beta_cs[1:] = hl[1:] ** 2 + hl[:-1] ** 2
        h2 = np.maximum(h[1:], h[:-1])
        l2 = np.minimum(l_[1:], l_[:-1])
        gamma_cs[1:] = np.log(np.maximum(h2 / np.maximum(l2, 1e-12), 1.0)) ** 2
    denom_cs = 3.0 - 2.0 * math.sqrt(2.0)
    alpha_cs = np.maximum(
        (np.sqrt(2.0 * beta_cs) - np.sqrt(beta_cs)) / denom_cs - np.sqrt(gamma_cs / denom_cs),
        0.0,
    )
    cs_daily = np.where(traded, 2.0 * np.tanh(0.5 * alpha_cs), np.nan)
    cs_21 = np.log1p(np.maximum(_rolling_mean(cs_daily, 21), 0.0) * 100.0)

    eta_mid = 0.5 * (log_h + log_l)
    ar_sq_raw = np.zeros(n, dtype=np.float64)
    if n > 1:
        ar_sq_raw[1:] = 4.0 * (log_c[:-1] - eta_mid[:-1]) * (log_c[:-1] - eta_mid[1:])
    ar_sq = np.where(traded, ar_sq_raw, np.nan)
    ar_21 = np.log1p(np.sqrt(np.maximum(_rolling_mean(ar_sq, 21), 0.0)) * 100.0)

    # 4) Amihud direccional: días volume <= 0 enmascarados como NaN/neutrales (jamás 8.0!)
    dollar_vol = np.maximum(c * v, 1.0)
    illiq = np.where(traded, np.abs(ret_cc) / dollar_vol * 1e6, np.nan)
    up_m = traded & (ret_cc > 0.0)
    dn_m = traded & (ret_cc < 0.0)
    amihud_up_21 = np.log1p(_rolling_mean(np.where(up_m, illiq, np.nan), 21))
    amihud_dn_21 = np.log1p(_rolling_mean(np.where(dn_m, illiq, np.nan), 21))
    amihud_asym_21 = amihud_dn_21 - amihud_up_21

    # 5) Proxy de flujo firmado (CLV * volumen) y correlación exacta precio-volumen en [-1, 1]
    clv = np.where((h - l_) > 1e-12, ((c - l_) - (h - c)) / np.maximum(h - l_, 1e-12), 0.0)
    signed_v = np.where(traded, clv * (log_v - _rolling_mean(log_v_traded, 63)), 0.0)
    sv_5 = _rolling_mean(signed_v, 5)
    sv_21 = _rolling_mean(signed_v, 21)
    pv_corr_21 = _rolling_window_corr(ret_cc, log_v_traded, 21)

    # 6) Asimetría y curtosis centrales exactas de la ventana (21d)
    skew_21, kurt_21 = _rolling_window_skew_kurt(ret_cc, 21)

    # 7) Diferenciación fraccionaria (López de Prado)
    ffd_035 = _apply_ffd(log_c, 0.35)
    ffd_050 = _apply_ffd(log_c, 0.50)

    # 8) Firmas de trayectoria de orden 2 (áreas firmadas de Lévy)
    sig_pv_5 = _rolling_levy_area(log_c, log_v, 5)
    sig_pv_21 = _rolling_levy_area(log_c, log_v, 21)
    sig_pr_21 = _rolling_levy_area(log_c, np.log1p(hl * 100.0), 21)

    cols = [
        gk_21,
        gk_63,
        rs_21,
        rs_63,
        yz_21,
        yz_63,
        hurst_21,
        hurst_63,
        vov_21,
        cs_21,
        ar_21,
        amihud_up_21,
        amihud_dn_21,
        amihud_asym_21,
        sv_5,
        sv_21,
        pv_corr_21,
        skew_21,
        kurt_21,
        ffd_035,
        ffd_050,
        sig_pv_5,
        sig_pv_21,
        sig_pr_21,
    ]
    mat = np.column_stack(cols)
    mat = np.nan_to_num(mat, nan=0.0, posinf=LIMIT, neginf=-LIMIT)
    return np.clip(mat, -LIMIT, LIMIT).astype(np.float32)


def rmt_clean_correlation(
    returns_window: np.ndarray,
    *,
    top_k: int = 5,
    min_overlap: int = 5,
) -> dict[str, object]:
    """Limpieza espectral por Teoría de Matrices Aleatorias (Marchenko-Pastur, Bouchaud).

    Usa solape válido por pares (`cnt_ij - 1`, con guardia `min_overlap`) para evitar sesgo de
    encogimiento en activos suspendidos o recién incorporados (IPOs), separa los autovalores
    colectivos reales por encima de `lambda_plus = (1 + sqrt(N/T))^2` y calcula cargas limpias.
    """
    r = np.asarray(returns_window, dtype=np.float64)
    t_len, n_assets = r.shape
    if t_len < 3 or n_assets < 2:
        eye = np.eye(n_assets, dtype=np.float64)
        return dict(
            n_signal_modes=0,
            absorption_ratio=0.0,
            spectral_entropy=0.0,
            raw_corr=eye,
            clean_corr=eye.copy(),
            loadings=np.zeros((n_assets, top_k), dtype=np.float64),
        )

    valid = np.isfinite(r)
    cnt = np.maximum(valid.sum(axis=0), 1)
    means = np.sum(np.where(valid, r, 0.0), axis=0) / cnt
    centered = np.where(valid, r - means[None, :], 0.0)
    stds = np.sqrt(np.sum(centered * centered, axis=0) / np.maximum(cnt - 1, 1))
    active = stds > 1e-12
    z = np.zeros_like(centered)
    z[:, active] = centered[:, active] / stds[None, active]

    v_f = valid.astype(np.float64)
    cnt_ij = v_f.T @ v_f
    min_ov = max(3, min(int(min_overlap), t_len))
    raw_corr = np.where(
        cnt_ij >= min_ov,
        np.clip((z.T @ z) / np.maximum(cnt_ij - 1.0, 1.0), -1.0, 1.0),
        0.0,
    )
    corr = 0.5 * (raw_corr + raw_corr.T)
    np.fill_diagonal(corr, 1.0)

    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    # Cota superior de Marchenko-Pastur (Laloux-Potters-Bouchaud)
    q_ratio = n_assets / max(float(t_len), 1.0)
    lambda_plus = (1.0 + math.sqrt(q_ratio)) ** 2
    signal_mask = eigvals > lambda_plus
    n_sig = int(signal_mask.sum())
    # Reestimar varianza del bulk si hay modos de señal fuertes
    if 0 < n_sig < n_assets:
        sigma_bulk2 = float(eigvals[~signal_mask].mean())
        lambda_plus_adj = sigma_bulk2 * lambda_plus
        signal_mask = eigvals > lambda_plus_adj
        n_sig = int(signal_mask.sum())

    # Recorte de autovalores de ruido preservando traza (eigenvalue clipping)
    cleaned_eigs = eigvals.copy()
    if n_sig < n_assets:
        noise_mean = max(float(eigvals[~signal_mask].mean()), 1e-4)
        cleaned_eigs[~signal_mask] = noise_mean
    cleaned_eigs = np.maximum(cleaned_eigs, 1e-4)
    clean_corr = (eigvecs * cleaned_eigs[None, :]) @ eigvecs.T
    clean_corr = 0.5 * (clean_corr + clean_corr.T)
    diag = np.sqrt(np.maximum(np.diag(clean_corr), 1e-8))
    clean_corr = clean_corr / (diag[:, None] * diag[None, :])
    np.fill_diagonal(clean_corr, 1.0)

    tr = float(eigvals.sum())
    k_abs = min(5, n_assets)
    absorption = float(eigvals[:k_abs].sum() / max(tr, 1e-12))
    probs = eigvals / max(tr, 1e-12)
    probs_pos = probs[probs > 1e-15]
    entropy = float(-np.sum(probs_pos * np.log(probs_pos)) / max(math.log(n_assets), 1e-12))

    # Cargas deterministas de los top_k modos (signo canónico)
    loadings = np.zeros((n_assets, top_k), dtype=np.float64)
    avail = min(top_k, n_assets)
    for j in range(avail):
        vec = eigvecs[:, j].copy()
        pivot = int(np.argmax(np.abs(vec)))
        if vec[pivot] < 0.0:
            vec = -vec
        loadings[:, j] = vec * math.sqrt(max(float(eigvals[j]), 0.0))

    return dict(
        n_signal_modes=n_sig,
        absorption_ratio=absorption,
        spectral_entropy=entropy,
        raw_corr=corr,
        clean_corr=clean_corr,
        loadings=loadings,
    )


def causal_wasserstein_shift(
    returns_panel: np.ndarray,
    *,
    short_window: int = 5,
    long_window: int = 40,
    maturity_lag: int = 1,
    n_quantiles: int = 64,
) -> np.ndarray:
    """Distancia 1D de Wasserstein estrictamente causal entre distribución reciente y previa.

    En la sesión `t`, compara las filas maduras `[t-lag-short+1 .. t-lag]` frente a la ventana
    anterior `[t-lag-short-long+1 .. t-lag-short]` mediante cuantiles uniformes.
    """
    r = np.asarray(returns_panel, dtype=np.float64)
    t_len = r.shape[0]
    out = np.zeros(t_len, dtype=np.float64)
    grid = np.linspace(0.01, 0.99, n_quantiles)
    for t in range(t_len):
        end_short = t - maturity_lag + 1  # exclusivo; última fila usada es t - maturity_lag
        start_short = end_short - short_window
        start_long = start_short - long_window
        if start_long < 0 or start_short <= start_long:
            continue
        s_vals = r[start_short:end_short]
        l_vals = r[start_long:start_short]
        s_fin = s_vals[np.isfinite(s_vals)]
        l_fin = l_vals[np.isfinite(l_vals)]
        if s_fin.size < 10 or l_fin.size < 20:
            continue
        qs = np.quantile(s_fin, grid)
        ql = np.quantile(l_fin, grid)
        out[t] = float(np.mean(np.abs(qs - ql)))
    return out


def causal_lead_lag_spillover(
    returns_panel: np.ndarray,
    liquidity_panel: np.ndarray,
    *,
    window: int = 40,
    leader_frac: float = 0.2,
    maturity_lag: int = 1,
) -> np.ndarray:
    """Señal causal dirigida Lead-Lag desde líderes líquidos (t-lag) hacia seguidores (t).

    Estima la beta retardada de cada activo sobre `leader_ret[tau - maturity_lag]` usando
    exclusivamente sesiones maduras `<= t - maturity_lag` y proyecta con `leader_ret[t - lag]`.
    """
    if maturity_lag < 1:
        raise ValueError("maturity_lag debe ser >= 1 para preservar observabilidad causal en t")
    r = np.asarray(returns_panel, dtype=np.float64)
    liq = np.asarray(liquidity_panel, dtype=np.float64)
    t_len, n_assets = r.shape
    out = np.zeros((t_len, n_assets), dtype=np.float64)
    n_leaders = max(1, int(round(n_assets * leader_frac)))

    # Retorno medio diario de los líderes definidos con liquidez del propio día (conocido al cierre)
    leader_ret = np.zeros(t_len, dtype=np.float64)
    for t in range(t_len):
        lt = liq[t]
        rt = r[t]
        ok = np.isfinite(lt) & np.isfinite(rt)
        if int(ok.sum()) < n_leaders:
            continue
        idx_ok = np.flatnonzero(ok)
        top_idx = idx_ok[np.argsort(lt[idx_ok], kind="stable")[-n_leaders:]]
        leader_ret[t] = float(np.mean(rt[top_idx]))

    for t in range(t_len):
        end_m = t - maturity_lag + 1  # última etiqueta madura es end_m - 1
        start_m = max(maturity_lag, end_m - window)
        if end_m - start_m < 10 or t - maturity_lag < 0:
            continue
        # Pares causales: x = leader_ret[tau - lag], y = r[tau, :] para tau in [start_m, end_m)
        x = leader_ret[start_m - maturity_lag : end_m - maturity_lag]
        y = r[start_m:end_m]
        xm = float(np.mean(x))
        xc = x - xm
        var_x = float(np.dot(xc, xc))
        if var_x <= 1e-12:
            continue
        valid = np.isfinite(y)
        cnt = np.maximum(valid.sum(axis=0), 1)
        ym = np.sum(np.where(valid, y, 0.0), axis=0) / cnt
        yc = np.where(valid, y - ym[None, :], 0.0)
        cov_xy = xc @ yc
        var_y = np.sum(yc * yc, axis=0)
        beta = cov_xy / (var_x + 1e-10)
        denom_c = np.sqrt(var_x * np.maximum(var_y, 1e-12))
        corr = np.where(var_y > 1e-12, cov_xy / np.maximum(denom_c, 1e-12), 0.0)
        # Contracción bayesiana James-Stein sobre t^2 para filtrar relaciones espurias
        r2 = np.clip(corr * corr, 0.0, 0.999)
        t2 = max(end_m - start_m - 2, 1) * r2 / np.maximum(1.0 - r2, 1e-6)
        shrink = np.clip(t2 / (t2 + 3.0), 0.0, 1.0)
        out[t] = shrink * beta * (leader_ret[t - maturity_lag] - xm)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def macro_surprises_and_betas(
    returns_panel: np.ndarray,
    macro: np.ndarray,
    *,
    halflife: float = 21.0,
    beta_window: int = 63,
    maturity_lag: int = 1,
    ridge_lambda: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sorpresas macro causales EWMA y betas macro dinámicas por activo."""
    r = np.asarray(returns_panel, dtype=np.float64)
    m = np.asarray(macro, dtype=np.float64)
    if m.ndim == 1:
        m = m[:, None]
    t_len, n_assets = r.shape
    k_mac = m.shape[1]
    surprises = np.zeros((t_len, k_mac), dtype=np.float64)
    decay = math.pow(0.5, 1.0 / max(halflife, 1.0))

    mean_ew = np.zeros(k_mac, dtype=np.float64)
    var_ew = np.ones(k_mac, dtype=np.float64)
    seen = np.zeros(k_mac, dtype=np.float64)
    for t in range(t_len):
        mt = m[t]
        fin = np.isfinite(mt)
        if t > 0:
            std = np.sqrt(np.maximum(var_ew, 1e-6))
            z_mac = np.clip((mt - mean_ew) / std, -6.0, 6.0)
            surprises[t] = np.where(fin & (seen >= 3.0), z_mac, 0.0)
        # Actualizar estado causal para t+1
        diff = np.where(fin, mt - mean_ew, 0.0)
        alpha = np.where(seen < 1.0, 1.0, 1.0 - decay)
        mean_ew = np.where(fin, mean_ew + alpha * diff, mean_ew)
        var_ew = np.where(fin, decay * var_ew + (1.0 - decay) * (diff * diff), var_ew)
        seen = seen + fin.astype(np.float64)

    betas = np.zeros((t_len, n_assets, k_mac), dtype=np.float64)
    eye = np.eye(k_mac, dtype=np.float64) * ridge_lambda
    for t in range(t_len):
        end_m = t - maturity_lag + 1
        start_m = max(0, end_m - beta_window)
        if end_m - start_m < max(k_mac + 4, 12):
            continue
        x = surprises[start_m:end_m]
        y = np.nan_to_num(r[start_m:end_m], nan=0.0)
        gram = x.T @ x + eye * (end_m - start_m) / 20.0
        rhs = x.T @ y  # (K, N)
        try:
            sol = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        betas[t] = np.clip(sol.T, -LIMIT, LIMIT)
    return surprises, betas


def panel_market_physics_v4(
    returns_panel: np.ndarray,
    liquidity_panel: np.ndarray,
    volatility_panel: np.ndarray,
    *,
    macro: np.ndarray | None = None,
    rmt_window: int = 63,
    top_k_rmt: int = 5,
    maturity_lag: int = 1,
    lag_vol_liq: bool = False,
) -> dict[str, np.ndarray]:
    """Calcula la física transversal completa y el descriptor de contexto de alto SNR para MoE.

    Contrato de observabilidad en `t`:
      - `maturity_lag >= 1` obligatorio: todos los cálculos dependientes de `returns_panel`
        (RMT, Wasserstein, Lead-Lag, betas macro, dispersión) usan estrictamente filas
        `<= t - maturity_lag`, nunca retornos objetivo futuros `y_t`.
      - `volatility_panel` y `liquidity_panel` deben ser observables al instante de decisión `t`;
        si provienen del cierre de `t`, pase `lag_vol_liq=True` para desplazar sus rangos
        `maturity_lag` sesiones.
    """
    if maturity_lag < 1:
        raise ValueError("maturity_lag debe ser >= 1 para preservar observabilidad causal en t")
    r = np.asarray(returns_panel, dtype=np.float64)
    liq = np.asarray(liquidity_panel, dtype=np.float64)
    vol = np.asarray(volatility_panel, dtype=np.float64)
    t_len, n_assets = r.shape

    rmt_abs = np.zeros(t_len, dtype=np.float64)
    rmt_ent = np.zeros(t_len, dtype=np.float64)
    rmt_nmodes = np.zeros(t_len, dtype=np.float64)
    rmt_loadings = np.zeros((t_len, n_assets, top_k_rmt), dtype=np.float64)
    cross_disp = np.zeros(t_len, dtype=np.float64)

    for t in range(t_len):
        end_m = t - maturity_lag + 1
        if end_m >= 2:
            prev = r[end_m - 1]
            pf = prev[np.isfinite(prev)]
            if pf.size >= 3:
                cross_disp[t] = float(np.std(pf))
        start_m = max(0, end_m - rmt_window)
        if end_m - start_m >= max(15, top_k_rmt + 2):
            res = rmt_clean_correlation(r[start_m:end_m], top_k=top_k_rmt)
            rmt_abs[t] = float(res["absorption_ratio"])
            rmt_ent[t] = float(res["spectral_entropy"])
            rmt_nmodes[t] = float(res["n_signal_modes"]) / max(float(n_assets), 1.0)
            rmt_loadings[t] = res["loadings"]

    wass = causal_wasserstein_shift(
        r, short_window=5, long_window=min(rmt_window, 40), maturity_lag=maturity_lag
    )
    lead_lag = causal_lead_lag_spillover(
        r, liq, window=min(rmt_window, 40), maturity_lag=maturity_lag
    )

    if macro is not None:
        m_surp, m_betas = macro_surprises_and_betas(
            r, macro, beta_window=rmt_window, maturity_lag=maturity_lag
        )
        macro_stress = np.sqrt(np.mean(m_surp * m_surp, axis=1))
    else:
        m_surp = np.zeros((t_len, 0), dtype=np.float64)
        m_betas = np.zeros((t_len, n_assets, 0), dtype=np.float64)
        macro_stress = np.zeros(t_len, dtype=np.float64)

    # Descriptor de contexto de ALTO SNR (baja varianza, jamás ruido crudo de retornos)
    vol_pct = np.zeros((t_len, n_assets), dtype=np.float64)
    liq_pct = np.zeros((t_len, n_assets), dtype=np.float64)
    for t in range(t_len):
        for src, dst in ((vol[t], vol_pct), (liq[t], liq_pct)):
            fin = np.isfinite(src)
            if int(fin.sum()) > 1:
                idx = np.flatnonzero(fin)
                order = np.argsort(src[idx], kind="stable")
                ranks = np.empty(len(idx), dtype=np.float64)
                ranks[order] = np.linspace(-1.0, 1.0, len(idx))
                dst[t, idx] = ranks
    if lag_vol_liq and maturity_lag > 0:
        for arr in (vol_pct, liq_pct):
            if maturity_lag < t_len:
                arr[maturity_lag:] = arr[:-maturity_lag].copy()
                arr[:maturity_lag] = 0.0
            else:
                arr[:] = 0.0

    high_snr_ctx = np.stack(
        [
            np.broadcast_to(rmt_abs[:, None], (t_len, n_assets)),
            np.broadcast_to(rmt_ent[:, None], (t_len, n_assets)),
            np.broadcast_to(rmt_nmodes[:, None], (t_len, n_assets)),
            np.broadcast_to(np.clip(wass * 50.0, 0.0, LIMIT)[:, None], (t_len, n_assets)),
            np.broadcast_to(np.clip(cross_disp * 25.0, 0.0, LIMIT)[:, None], (t_len, n_assets)),
            np.broadcast_to(np.clip(macro_stress, 0.0, LIMIT)[:, None], (t_len, n_assets)),
            vol_pct,
            liq_pct,
        ],
        axis=-1,
    ).astype(np.float32)

    return dict(
        rmt_absorption=rmt_abs,
        rmt_entropy=rmt_ent,
        rmt_loadings=rmt_loadings,
        wasserstein=wass,
        cross_dispersion=cross_disp,
        lead_lag=lead_lag,
        macro_surprises=m_surp,
        macro_betas=m_betas,
        high_snr_context=high_snr_ctx,
    )


def multimodal_fusion_features_v4(
    *,
    news_emb: np.ndarray,
    news_present: np.ndarray,
    chart_emb: np.ndarray,
    chart_present: np.ndarray,
    fund_vals: np.ndarray,
    fund_present: np.ndarray,
    macro_surprises: np.ndarray,
    macro_betas: np.ndarray,
    halflife: float = 21.0,
) -> dict[str, np.ndarray]:
    """Extrae rasgos causales de interacción multimodal: novedad, deriva visual y tailwind macro."""
    t_len, n_assets, d_news = news_emb.shape
    d_chart = chart_emb.shape[2]
    d_fund = fund_vals.shape[2]
    decay = math.pow(0.5, 1.0 / max(halflife, 1.0))

    news_p = np.asarray(news_present, dtype=bool)
    chart_p = np.asarray(chart_present, dtype=bool)
    fund_p = np.asarray(fund_present, dtype=bool)
    has_macro = (
        np.any(np.isfinite(macro_surprises), axis=1)
        if macro_surprises.shape[1] > 0
        else np.zeros(t_len, dtype=bool)
    )
    macro_p = np.broadcast_to(has_macro[:, None], (t_len, n_assets))

    news_mem = np.zeros((n_assets, d_news), dtype=np.float64)
    news_age = np.full(n_assets, 252.0, dtype=np.float64)
    news_int = np.zeros(n_assets, dtype=np.float64)
    chart_mem = np.zeros((n_assets, d_chart), dtype=np.float64)
    fund_mem = np.zeros((n_assets, d_fund), dtype=np.float64)
    fund_age = np.full(n_assets, 252.0, dtype=np.float64)

    out = np.zeros((t_len, n_assets, len(MULTIMODAL_V4_NAMES)), dtype=np.float32)

    for t in range(t_len):
        # 1) Novedad semántica de noticias (1 - coseno frente a EWMA previa)
        ne = np.nan_to_num(news_emb[t].astype(np.float64), nan=0.0)
        np_t = news_p[t]
        n_norm = np.linalg.norm(ne, axis=1)
        m_norm = np.linalg.norm(news_mem, axis=1)
        cos = np.sum(ne * news_mem, axis=1) / np.maximum(n_norm * m_norm, 1e-8)
        nov_raw = 1.0 - np.clip(cos, -1.0, 1.0)
        novelty = np.where(np_t & (m_norm > 1e-8), nov_raw, np_t.astype(np.float64))
        news_age = np.where(np_t, 0.0, np.minimum(news_age + 1.0, 252.0))
        news_int = decay * news_int + (1.0 - decay) * np_t.astype(np.float64)
        news_mem = np.where(np_t[:, None], decay * news_mem + (1.0 - decay) * ne, news_mem)

        # 2) Deriva visual de gráficos
        ce = np.nan_to_num(chart_emb[t].astype(np.float64), nan=0.0)
        cp_t = chart_p[t]
        c_dist = np.linalg.norm(ce - chart_mem, axis=1) / math.sqrt(max(d_chart, 1))
        c_drift = np.where(cp_t, c_dist, 0.0)
        chart_mem = np.where(cp_t[:, None], decay * chart_mem + (1.0 - decay) * ce, chart_mem)

        # 3) Cambio fundamental Point-in-Time y frescura
        fv = np.nan_to_num(fund_vals[t].astype(np.float64), nan=0.0)
        fp_t = fund_p[t]
        f_dist = np.linalg.norm(fv - fund_mem, axis=1) / math.sqrt(max(d_fund, 1))
        f_delta = np.where(fp_t, f_dist, 0.0)
        changed = fp_t & (f_delta > 1e-6)
        fund_age = np.where(changed, 0.0, np.minimum(fund_age + 1.0, 252.0))
        fund_mem = np.where(fp_t[:, None], fv, fund_mem)

        # 4) Macro tailwind: dot(asset_macro_beta_t, macro_surprise_t)
        if macro_surprises.shape[1] > 0:
            ms_t = np.nan_to_num(macro_surprises[t], nan=0.0)
            mb_t = np.nan_to_num(macro_betas[t], nan=0.0)
            tailwind = mb_t @ ms_t
            align_abs = np.abs(mb_t) @ np.abs(ms_t)
        else:
            tailwind = np.zeros(n_assets, dtype=np.float64)
            align_abs = np.zeros(n_assets, dtype=np.float64)

        completeness = (
            np_t.astype(np.float64)
            + cp_t.astype(np.float64)
            + fp_t.astype(np.float64)
            + macro_p[t].astype(np.float64)
        ) / 4.0

        row = np.column_stack(
            [
                novelty,
                np_t.astype(np.float64),
                np.log1p(news_age),
                news_int,
                c_drift,
                cp_t.astype(np.float64),
                f_delta,
                fp_t.astype(np.float64),
                np.log1p(fund_age),
                tailwind,
                align_abs,
                completeness,
            ]
        )
        out[t] = np.clip(np.nan_to_num(row, nan=0.0, posinf=LIMIT, neginf=-LIMIT), -LIMIT, LIMIT)

    presence = np.stack([news_p, chart_p, fund_p, macro_p], axis=-1).astype(np.float32)
    return dict(features=out, modality_presence=presence)
