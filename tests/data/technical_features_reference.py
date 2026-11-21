"""Segunda implementación de los rasgos técnicos `long_horizon_v2`, escrita desde su especificación.

No comparte código con `mars_titan.data.technical_features`: calcula cada ventana directamente,
sin sumas acumuladas. Sirve para contrastar las dos implementaciones (`test_technical_features`).
La escribió un segundo modelo (Gemini 4 Neon) a partir de la especificación en prosa de las 46
columnas, sin ver la implementación de referencia. Con precios reales de 150 activos (618.223
filas) la diferencia máxima entre ambas es de 2,6e-6.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

NOMBRES: tuple[str, ...] = (
    "alpha_percent",
    "alpha_t",
    "beta",
    "log_residual_volatility",
    "has_market_model",
    "return_5",
    "return_21",
    "return_63",
    "return_126",
    "return_252",
    "momentum_12_1",
    "log_volatility_252",
    "volatility_21_over_252",
    "volatility_63_over_252",
    "distance_to_high_252",
    "distance_to_low_252",
    "has_252_sessions",
    "log_dollar_volume_21",
    "dollar_volume_21_over_252",
    "dollar_volume_5_over_63",
    "abnormal_volume",
    "log_illiquidity_21",
    "return_1",
    "intraday_1",
    "overnight_1",
    "range_1",
    "intraday_21",
    "overnight_21",
    "maximum_return_21",
    "has_63_sessions",
    "intraday_mean_252_percent",
    "overnight_mean_252_percent",
    "intraday_5",
    "intraday_63",
    "intraday_252",
    "overnight_5",
    "overnight_63",
    "overnight_252",
    "realized_label_1",
    "realized_label_5",
    "realized_label_21",
    "alpha_21_percent",
    "alpha_63_percent",
    "untraded_63",
    "close_location_1",
    "close_location_5",
)

_IDX: dict[str, int] = {nombre: i for i, nombre in enumerate(NOMBRES)}
_MARKET_SXX_EPS: float = 2.220446049250313e-16
_MARKET_WINDOW: int = 252
_MARKET_MIN_PAIRS: int = 126
_CLIP_LIMIT: float = 8.0


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Suma móvil causal S(x, w)[t] = sum_{s=max(0, t-w+1)}^t x[s]."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    prefix = min(n, window - 1)
    if prefix > 0:
        out[:prefix] = np.cumsum(x[:prefix], dtype=np.float64)
    if n >= window:
        out[window - 1 :] = sliding_window_view(x, window).sum(axis=1)
    return out


def _rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    """Máximo móvil sobre exactamente `window` elementos; NaN antes de t = window - 1."""
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n >= window:
        out[window - 1 :] = sliding_window_view(x, window).max(axis=1)
    return out


def _rolling_min(x: np.ndarray, window: int) -> np.ndarray:
    """Mínimo móvil sobre exactamente `window` elementos; NaN antes de t = window - 1."""
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n >= window:
        out[window - 1 :] = sliding_window_view(x, window).min(axis=1)
    return out


def _vol(f: np.ndarray, window: int) -> np.ndarray:
    """Volatilidad poblacional causal sobre `window` retornos; NaN para t < window."""
    n = f.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= window:
        return out
    mean_f = _rolling_sum(f, window) / window
    mean_f2 = _rolling_sum(f * f, window) / window
    var = np.maximum(mean_f2 - mean_f * mean_f, 0.0)
    out[window:] = np.sqrt(var[window:])
    return out


def _compute_market_model(
    o: np.ndarray,
    c: np.ndarray,
    sessions: np.ndarray,
    calendar: np.ndarray,
    market: np.ndarray,
    out: np.ndarray,
) -> None:
    """Calcula y rellena en `out` las columnas del modelo de mercado (1..5 y 39..43)."""
    n_cal = calendar.shape[0]
    if n_cal == 0:
        return

    pos = np.searchsorted(calendar, sessions)
    in_cal = (pos < n_cal) & (calendar[np.minimum(pos, n_cal - 1)] == sessions)
    rows_in_cal = np.flatnonzero(in_cal)
    if rows_in_cal.size == 0:
        return
    cal_idx = pos[rows_in_cal]

    alineado = np.full(n_cal, np.nan, dtype=np.float64)
    o_in = o[rows_in_cal]
    c_in = c[rows_in_cal]
    valid_oc = np.isfinite(o_in) & np.isfinite(c_in) & (o_in > 0.0) & (c_in > 0.0)
    alineado[cal_idx[valid_oc]] = c_in[valid_oc] / o_in[valid_oc] - 1.0

    valid_pair = np.isfinite(alineado) & np.isfinite(market)
    n_pairs = _rolling_sum(valid_pair.astype(np.float64), _MARKET_WINDOW)

    xv = np.where(valid_pair, market, 0.0)
    yv = np.where(valid_pair, alineado, 0.0)

    sum_x = _rolling_sum(xv, _MARKET_WINDOW)
    sum_xx = _rolling_sum(xv * xv, _MARKET_WINDOW)
    sum_y = _rolling_sum(yv, _MARKET_WINDOW)
    sum_yy = _rolling_sum(yv * yv, _MARKET_WINDOW)
    sum_xy = _rolling_sum(xv * yv, _MARKET_WINDOW)

    with np.errstate(divide="ignore", invalid="ignore"):
        has_pairs = n_pairs > 0.0
        sxx = np.where(has_pairs, sum_xx - (sum_x * sum_x) / n_pairs, np.nan)
        sxy = np.where(has_pairs, sum_xy - (sum_x * sum_y) / n_pairs, np.nan)
        syy = np.where(has_pairs, sum_yy - (sum_y * sum_y) / n_pairs, np.nan)

        known = (n_pairs >= _MARKET_MIN_PAIRS) & (sxx > _MARKET_SXX_EPS)

        beta = np.where(known, sxy / sxx, np.nan)
        alfa = np.where(known, sum_y / n_pairs - beta * (sum_x / n_pairs), np.nan)
        residuo = np.where(
            known,
            np.sqrt(np.maximum(syy - beta * sxy, 0.0) / np.maximum(n_pairs - 2.0, 1.0)),
            np.nan,
        )

        realizada = np.zeros(n_cal, dtype=np.float64)
        if n_cal >= 2:
            raw_real = (alineado[1:] - alfa[:-1] - beta[:-1] * market[1:]) / residuo[:-1]
            realizada[1:] = np.where(np.isfinite(raw_real), raw_real, 0.0)

        raw_libre = alineado - beta * market
        libre = np.where(np.isfinite(raw_libre), raw_libre, 0.0)

    sc_real_5 = _rolling_sum(realizada, 5)
    sc_real_21 = _rolling_sum(realizada, 21)
    sc_libre_21 = _rolling_sum(libre, 21)
    sc_libre_63 = _rolling_sum(libre, 63)

    alpha_pct_in = alfa[cal_idx] * 100.0
    has_model_in = np.isfinite(alpha_pct_in)

    active_rows = rows_in_cal[has_model_in]
    active_cal = cal_idx[has_model_in]
    if active_rows.size == 0:
        return

    with np.errstate(divide="ignore", invalid="ignore"):
        out[active_rows, _IDX["alpha_percent"]] = alpha_pct_in[has_model_in]
        out[active_rows, _IDX["alpha_t"]] = alfa[active_cal] / (
            residuo[active_cal] / np.sqrt(n_pairs[active_cal])
        )
        out[active_rows, _IDX["beta"]] = beta[active_cal]
        out[active_rows, _IDX["log_residual_volatility"]] = np.log(residuo[active_cal] * 100.0)
        out[active_rows, _IDX["has_market_model"]] = 1.0

        out[active_rows, _IDX["realized_label_1"]] = realizada[active_cal]
        out[active_rows, _IDX["realized_label_5"]] = sc_real_5[active_cal] / np.sqrt(5.0)
        out[active_rows, _IDX["realized_label_21"]] = sc_real_21[active_cal] / np.sqrt(21.0)
        out[active_rows, _IDX["alpha_21_percent"]] = sc_libre_21[active_cal] / 21.0 * 100.0
        out[active_rows, _IDX["alpha_63_percent"]] = sc_libre_63[active_cal] / 63.0 * 100.0


def rasgos(
    prices: np.ndarray,
    sessions: np.ndarray,
    calendar: np.ndarray | None = None,
    market: np.ndarray | None = None,
) -> np.ndarray:
    """Calcula los 46 rasgos técnicos causales de un activo.

    Parámetros
    ----------
    prices : array-like de forma [filas, 5] float64
        Columnas: apertura (o), máximo (h), mínimo (l), cierre (c), volumen (v).
    sessions : array-like de forma [filas] datetime64[D]
        Fechas de sesión del activo en orden estrictamente creciente.
    calendar : array-like de forma [n] datetime64[D] o None
        Calendario creciente de sesiones del mercado.
    market : array-like de forma [n] float64 o None
        Rentabilidad apertura-a-cierre del factor de mercado por sesión del calendario.

    Retorna
    -------
    numpy.ndarray float32 de forma [filas, 46]
        Matriz de rasgos con NaN/inf sustituidos por 0 y valores acotados a [-8, 8].
    """
    prices_arr = np.asarray(prices, dtype=np.float64)
    if prices_arr.ndim != 2 or prices_arr.shape[1] != 5:
        raise ValueError(f"prices debe tener forma [filas, 5], recibido {prices_arr.shape}")

    n_rows = prices_arr.shape[0]
    sessions_arr = np.asarray(sessions, dtype="datetime64[D]")
    if sessions_arr.ndim != 1 or sessions_arr.shape[0] != n_rows:
        raise ValueError(f"sessions debe tener forma [{n_rows}], recibido {sessions_arr.shape}")

    cal_arr: np.ndarray | None = None
    mkt_arr: np.ndarray | None = None
    if calendar is not None and market is not None:
        cal_arr = np.asarray(calendar, dtype="datetime64[D]")
        mkt_arr = np.asarray(market, dtype=np.float64)
        if cal_arr.ndim != 1 or mkt_arr.ndim != 1 or cal_arr.shape[0] != mkt_arr.shape[0]:
            raise ValueError(
                "calendar y market deben ser vectores 1D de la misma longitud, "
                f"recibido {cal_arr.shape} y {mkt_arr.shape}"
            )

    if n_rows == 0:
        return np.zeros((0, len(NOMBRES)), dtype=np.float32)

    o = prices_arr[:, 0]
    h = prices_arr[:, 1]
    l = prices_arr[:, 2]  # noqa: E741
    c = prices_arr[:, 3]
    v = prices_arr[:, 4]

    out = np.full((n_rows, len(NOMBRES)), np.nan, dtype=np.float64)
    out[:, _IDX["has_market_model"]] = 0.0

    # Columnas 1..5 y 39..43: modelo de mercado
    if cal_arr is not None and mkt_arr is not None:
        _compute_market_model(o, c, sessions_arr, cal_arr, mkt_arr, out)

    with np.errstate(divide="ignore", invalid="ignore"):
        lc = np.log(c)

        d = np.empty(n_rows, dtype=np.float64)
        d[0] = np.nan
        d[1:] = lc[1:] - lc[:-1]
        f = np.where(np.isnan(d), 0.0, d)

        vol_21 = _vol(f, 21)
        vol_63 = _vol(f, 63)
        y_vol = _vol(f, 252)

        intradia = np.log(c / o)
        nocturno = np.empty(n_rows, dtype=np.float64)
        nocturno[0] = np.nan
        nocturno[1:] = np.log(o[1:] / c[:-1])

        ci = np.where(np.isfinite(intradia), intradia, 0.0)
        cn = np.where(np.isfinite(nocturno), nocturno, 0.0)

        dolares = c * np.maximum(v, 0.0)

        # Columnas 6..10: retornos normalizados por volatilidad anual
        for k in (5, 21, 63, 126, 252):
            if n_rows > k:
                out[k:, _IDX[f"return_{k}"]] = (lc[k:] - lc[:-k]) / (y_vol[k:] * np.sqrt(float(k)))

        # Columna 11: momentum 12-1
        if n_rows > 252:
            out[252:, _IDX["momentum_12_1"]] = (lc[252 - 21 : n_rows - 21] - lc[:-252]) / (
                y_vol[252:] * np.sqrt(231.0)
            )

        # Columnas 12..14: volatilidad y ratios de volatilidad
        out[:, _IDX["log_volatility_252"]] = np.log(y_vol * 100.0)
        out[:, _IDX["volatility_21_over_252"]] = np.log(vol_21 / y_vol)
        out[:, _IDX["volatility_63_over_252"]] = np.log(vol_63 / y_vol)

        # Columnas 15..16: distancia a máximo y mínimo de cierres en 252 sesiones
        out[:, _IDX["distance_to_high_252"]] = lc - np.log(_rolling_max(c, 252))
        out[:, _IDX["distance_to_low_252"]] = lc - np.log(_rolling_min(c, 252))

        # Columna 17: indicador de 252 retornos (t >= 252)
        idx_t = np.arange(n_rows)
        out[:, _IDX["has_252_sessions"]] = (idx_t >= 252).astype(np.float64)

        # Columnas 18..22: volumen en dólares, volumen anormal e iliquidez
        s_dol_5 = _rolling_sum(dolares, 5)
        s_dol_21 = _rolling_sum(dolares, 21)
        s_dol_63 = _rolling_sum(dolares, 63)
        s_dol_252 = _rolling_sum(dolares, 252)

        if n_rows > 20:
            out[20:, _IDX["log_dollar_volume_21"]] = (
                np.log(s_dol_21[20:] / 21.0) - np.log(1e7)
            ) / 3.0
        if n_rows > 251:
            out[251:, _IDX["dollar_volume_21_over_252"]] = np.log(
                (s_dol_21[251:] / 21.0) / (s_dol_252[251:] / 252.0)
            )
        if n_rows > 62:
            out[62:, _IDX["dollar_volume_5_over_63"]] = np.log(
                (s_dol_5[62:] / 5.0) / (s_dol_63[62:] / 63.0)
            )

        s_v_21 = _rolling_sum(v, 21)
        if n_rows > 20:
            out[20:, _IDX["abnormal_volume"]] = np.log(v[20:] / (s_v_21[20:] / 21.0))

        impacto = np.zeros(n_rows, dtype=np.float64)
        pos_dol = dolares > 0.0
        impacto[pos_dol] = np.abs(f[pos_dol]) / dolares[pos_dol]
        s_imp_21 = _rolling_sum(impacto, 21)
        if n_rows > 20:
            out[20:, _IDX["log_illiquidity_21"]] = (
                np.log(s_imp_21[20:] / 21.0) - np.log(1e-9)
            ) / 3.0

        # Columnas 23..30: rasgos diarios, intradía/nocturno 21d, máximo 21d e indicador 63
        out[:, _IDX["return_1"]] = d / y_vol
        out[:, _IDX["intraday_1"]] = intradia / y_vol
        out[:, _IDX["overnight_1"]] = nocturno / y_vol
        out[:, _IDX["range_1"]] = np.log(h / l) / y_vol

        s_ci_5 = _rolling_sum(ci, 5)
        s_ci_21 = _rolling_sum(ci, 21)
        s_ci_63 = _rolling_sum(ci, 63)
        s_ci_252 = _rolling_sum(ci, 252)

        s_cn_5 = _rolling_sum(cn, 5)
        s_cn_21 = _rolling_sum(cn, 21)
        s_cn_63 = _rolling_sum(cn, 63)
        s_cn_252 = _rolling_sum(cn, 252)

        if n_rows > 20:
            out[20:, _IDX["intraday_21"]] = (s_ci_21[20:] / 21.0) / y_vol[20:] * np.sqrt(21.0)
        if n_rows > 21:
            out[21:, _IDX["overnight_21"]] = (s_cn_21[21:] / 21.0) / y_vol[21:] * np.sqrt(21.0)

        out[:, _IDX["maximum_return_21"]] = _rolling_max(f, 21) / y_vol
        out[:, _IDX["has_63_sessions"]] = (idx_t >= 63).astype(np.float64)

        # Columnas 31..38: medias intradía y nocturna multi-horizonte
        if n_rows > 251:
            out[251:, _IDX["intraday_mean_252_percent"]] = (s_ci_252[251:] / 252.0) * 100.0
        if n_rows > 252:
            out[252:, _IDX["overnight_mean_252_percent"]] = (s_cn_252[252:] / 252.0) * 100.0

        for w, s_ci in ((5, s_ci_5), (63, s_ci_63), (252, s_ci_252)):
            if n_rows > w - 1:
                out[w - 1 :, _IDX[f"intraday_{w}"]] = (
                    (s_ci[w - 1 :] / float(w)) / y_vol[w - 1 :] * np.sqrt(float(w))
                )

        for w, s_cn in ((5, s_cn_5), (63, s_cn_63), (252, s_cn_252)):
            if n_rows > w:
                out[w:, _IDX[f"overnight_{w}"]] = (
                    (s_cn[w:] / float(w)) / y_vol[w:] * np.sqrt(float(w))
                )

        # Columnas 44..46: sesiones sin negociación y posición del cierre
        u = ((v <= 0.0) | ((o == c) & (h == l))).astype(np.float64)
        s_u_63 = _rolling_sum(u, 63)
        if n_rows > 62:
            out[62:, _IDX["untraded_63"]] = s_u_63[62:] / 63.0

        cl1 = np.zeros(n_rows, dtype=np.float64)
        hl_pos = h > l
        cl1[hl_pos] = (2.0 * c[hl_pos] - h[hl_pos] - l[hl_pos]) / (h[hl_pos] - l[hl_pos])
        out[:, _IDX["close_location_1"]] = cl1

        s_cl1_5 = _rolling_sum(cl1, 5)
        if n_rows > 4:
            out[4:, _IDX["close_location_5"]] = s_cl1_5[4:] / 5.0

    # Paso final: NaN/inf -> 0, acotar a [-8, 8], convertir a float32
    out = np.where(np.isfinite(out), out, 0.0)
    out = np.clip(out, -_CLIP_LIMIT, _CLIP_LIMIT)
    return out.astype(np.float32)
