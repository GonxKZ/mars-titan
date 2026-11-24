"""Evaluador Granular de 6 Dimensiones (32 métricas) para MARS-TITAN y MMARRS-TITAN.

Mide sin puntos ciegos:
  1. Valor puntual y precio (MAE, MedAE, RMSE, Huber, sMAPE, OOS R^2, Mincer-Zarnowitz).
  2. Ranking transversal y alfa puro (Rank IC, Pearson IC, ICIR, t_NW, IC neutralizado, spreads).
  3. Tendencias, Triple-Barrier (López de Prado) y puntos de giro.
  4. Incertidumbre, colas y calibración (CRPS, Pinball 9 cuantiles, Winkler, PIT KS, Brier/ECE,
     VaR/CVaR Kupiec y Christoffersen).
  5. Autopsia causal del error (descomposición mercado/factor/idiosincrático + salud de bucle/MoE).
  6. Economía real y anti-sobreajuste (Sharpe neto 1x/2x/3x con Almgren-Chriss, Sortino, Calmar,
     MaxDD, capacidad en M USD, Deflated Sharpe Ratio y CSCV/PBO).
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Literal

import numpy as np
from scipy import stats

from mars_titan.evaluation import factor_neutralization as fn
from mars_titan.evaluation import online_aggregator as oa

QUANTILE_LEVELS_9 = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
EULER_GAMMA = 0.5772156649015329
EXECUTION_MODES: tuple[str, ...] = ("close_to_close", "open_to_close_200pct")
MARKETS: tuple[str, ...] = ("US", "CN")
VAR_COV_MODES: tuple[str, ...] = ("auto", "factor_ewma", "ewma_portfolio", "independent")
DEFAULT_SURVIVORSHIP_DISCLOSURE = (
    "UNVERIFIED_COHORT_SURVIVORSHIP_CHECK_REQUIRED: co-gate with liquid top-third stratification."
)
_Z_05 = 1.6448536269514722


def triple_barrier_labels(
    returns_panel: np.ndarray,
    vol_panel: np.ndarray,
    *,
    up_mult: float = 2.0,
    down_mult: float = 2.0,
    horizon: int = 5,
) -> np.ndarray:
    """Etiquetas Triple-Barrier de López de Prado (`+1` barrera sup., `-1` inf., `0` vertical)."""
    r = np.asarray(returns_panel, dtype=np.float64)
    v = np.maximum(np.asarray(vol_panel, dtype=np.float64), 1e-6)
    t_len, n_assets = r.shape
    labels = np.zeros((t_len, n_assets), dtype=np.int8)
    for t in range(t_len):
        end = min(t + horizon, t_len)
        if end <= t:
            continue
        cum = np.cumsum(np.nan_to_num(r[t:end], nan=0.0), axis=0)  # (H_eff, N)
        up_thr = up_mult * v[t][None, :]
        dn_thr = -down_mult * v[t][None, :]
        hit_up = cum >= up_thr
        hit_dn = cum <= dn_thr
        any_up = hit_up.any(axis=0)
        any_dn = hit_dn.any(axis=0)
        first_up = np.where(any_up, np.argmax(hit_up, axis=0), horizon + 1)
        first_dn = np.where(any_dn, np.argmax(hit_dn, axis=0), horizon + 1)
        labels[t] = np.where(
            first_up < first_dn,
            1,
            np.where(first_dn < first_up, -1, 0),
        ).astype(np.int8)
    return labels


def discrete_crps(
    quantiles: np.ndarray,
    target: np.ndarray,
    *,
    levels: tuple[float, ...] = QUANTILE_LEVELS_9,
) -> dict[str, object]:
    """CRPS discreto exacto (2 * media de Pinball) y Pinball por nivel cuantílico."""
    q = np.asarray(quantiles, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    tau = np.asarray(levels, dtype=np.float64)
    resid = y[:, None] - q
    pinball = np.maximum(tau[None, :] * resid, (tau[None, :] - 1.0) * resid)
    by_level = {f"{lv:.2f}": float(pinball[:, i].mean()) for i, lv in enumerate(levels)}
    mean_pb = float(pinball.mean())
    return dict(
        crps=2.0 * mean_pb,
        pinball_mean=mean_pb,
        pinball_by_level=by_level,
    )


def kupiec_pof_test(violations: np.ndarray, alpha: float) -> dict[str, float]:
    """Prueba de proporción de fallos (POF) de Kupiec para backtest de VaR en cola `alpha`."""
    v = np.asarray(violations, dtype=bool)
    n = len(v)
    x = int(v.sum())
    if n == 0 or not (0.0 < alpha < 1.0):
        return dict(rate=0.0, lr_stat=0.0, p_value=1.0)
    p_hat = x / n
    eps = 1e-12
    p_c = min(max(p_hat, eps), 1.0 - eps)
    ll_null = (n - x) * math.log(1.0 - alpha) + x * math.log(alpha)
    ll_alt = (n - x) * math.log(1.0 - p_c) + x * math.log(p_c)
    lr = max(0.0, -2.0 * (ll_null - ll_alt))
    p_val = float(stats.chi2.sf(lr, df=1))
    return dict(rate=p_hat, lr_stat=lr, p_value=p_val)


def christoffersen_independence_test(violations: np.ndarray) -> dict[str, float]:
    """Prueba de independencia de agrupamiento de fallos VaR de Christoffersen (1998)."""
    v = np.asarray(violations, dtype=np.int64)
    if len(v) < 3:
        return dict(lr_ind=0.0, p_value=1.0)
    prev, curr = v[:-1], v[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    eps = 1e-12
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi_c = min(max(pi, eps), 1.0 - eps)
    pi0_c = min(max(pi0, eps), 1.0 - eps)
    pi1_c = min(max(pi1, eps), 1.0 - eps)
    ll_ind = (n00 + n10) * math.log(1.0 - pi_c) + (n01 + n11) * math.log(pi_c)
    ll_dep = (
        n00 * math.log(1.0 - pi0_c)
        + n01 * math.log(pi0_c)
        + n10 * math.log(1.0 - pi1_c)
        + n11 * math.log(pi1_c)
    )
    lr = max(0.0, -2.0 * (ll_ind - ll_dep))
    return dict(lr_ind=lr, p_value=float(stats.chi2.sf(lr, df=1)))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    *,
    n_trials: int = 1,
    benchmark_sr: float = 0.0,
    periods_per_year: float = 252.0,
    trial_sharpes: np.ndarray | list[float] | None = None,
    sr_std: float | None = None,
    sr_var: float | None = None,
    trial_sharpes_annualized: bool = False,
) -> dict[str, float]:
    """Deflated Sharpe Ratio exacto (Bailey & López de Prado 2014; paridad backtest_overfitting)."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 5 or float(np.std(r, ddof=1)) <= 1e-12:
        return dict(
            sr_annual=0.0,
            sr_threshold=0.0,
            dsr=0.5,
            sr_session=0.0,
            sr_threshold_session=0.0,
            sr_std_used=0.0,
            n_trials_effective=max(int(n_trials), 1),
        )
    sr_daily = float(np.mean(r) / np.std(r, ddof=1))
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # curtosis bruta (3 para normal)

    eff_trials = max(int(n_trials), 1)
    ts_clean = np.empty(0, dtype=np.float64)
    if trial_sharpes is not None:
        ts_arr = np.asarray(trial_sharpes, dtype=np.float64)
        ts_clean = ts_arr[np.isfinite(ts_arr)]
        if trial_sharpes_annualized:
            ts_clean = ts_clean / math.sqrt(periods_per_year)
        eff_trials = max(eff_trials, len(ts_clean))

    # Dispersión empírica entre ensayos sqrt(V[{SR_k}]) (paridad exacta backtest_overfitting.py)
    if sr_var is not None:
        sigma_sr = math.sqrt(max(float(sr_var), 0.0))
    elif sr_std is not None:
        sigma_sr = max(float(sr_std), 0.0)
    elif len(ts_clean) >= 2:
        sigma_sr = float(np.std(ts_clean, ddof=1))
    else:
        # Respaldo asintótico nulo cuando solo se aporta una serie sin distribución de ensayos
        sigma_sr = 1.0 / math.sqrt(max(n - 1, 1))

    if eff_trials > 1:
        z1 = float(stats.norm.ppf(1.0 - 1.0 / eff_trials))
        z2 = float(stats.norm.ppf(1.0 - 1.0 / (eff_trials * math.e)))
        sr_star = benchmark_sr / math.sqrt(periods_per_year) + sigma_sr * (
            (1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2
        )
    else:
        sr_star = benchmark_sr / math.sqrt(periods_per_year)

    denom_var = max(1.0 - skew * sr_daily + 0.25 * (kurt - 1.0) * sr_daily * sr_daily, 1e-6)
    z_stat = (sr_daily - sr_star) * math.sqrt(max(n - 1, 1)) / math.sqrt(denom_var)
    return dict(
        sr_annual=sr_daily * math.sqrt(periods_per_year),
        sr_threshold=sr_star * math.sqrt(periods_per_year),
        dsr=float(stats.norm.cdf(z_stat)),
        sr_session=sr_daily,
        sr_threshold_session=sr_star,
        sr_std_used=sigma_sr,
        n_trials_effective=eff_trials,
    )


def cscv_pbo(returns_matrix: np.ndarray, *, n_blocks: int = 8) -> dict[str, float]:
    """Probability of Backtest Overfitting por CSCV (Bailey et al., 2015)."""
    mat = np.asarray(returns_matrix, dtype=np.float64)
    t_len, n_strats = mat.shape
    if t_len < n_blocks * 4 or n_strats < 2 or n_blocks % 2 != 0:
        return dict(pbo=0.5, n_splits=0, logit_median=0.0)
    usable = (t_len // n_blocks) * n_blocks
    blocks = mat[:usable].reshape(n_blocks, usable // n_blocks, n_strats)
    half = n_blocks // 2
    logits = []
    for is_idx in itertools.combinations(range(n_blocks), half):
        is_set = set(is_idx)
        oos_idx = tuple(i for i in range(n_blocks) if i not in is_set)
        r_is = blocks[list(is_idx)].reshape(-1, n_strats)
        r_oos = blocks[list(oos_idx)].reshape(-1, n_strats)
        sr_is = np.mean(r_is, axis=0) / np.maximum(np.std(r_is, axis=0, ddof=1), 1e-12)
        sr_oos = np.mean(r_oos, axis=0) / np.maximum(np.std(r_oos, axis=0, ddof=1), 1e-12)
        best_is = int(np.argmax(sr_is))
        # Rango relativo de best_is en OOS
        rank_oos = float(np.sum(sr_oos <= sr_oos[best_is])) / (n_strats + 1.0)
        rank_oos = min(max(rank_oos, 1e-6), 1.0 - 1e-6)
        logits.append(math.log(rank_oos / (1.0 - rank_oos)))
    arr = np.asarray(logits, dtype=np.float64)
    return dict(
        pbo=float(np.mean(arr <= 0.0)),
        n_splits=len(logits),
        logit_median=float(np.median(arr)),
    )


def _newey_west_t(series: np.ndarray) -> float:
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return 0.0
    mean = float(np.mean(x))
    xc = x - mean
    lag = max(1, int(round(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    var = float(np.dot(xc, xc)) / n
    for k in range(1, min(lag + 1, n)):
        w = 1.0 - k / (lag + 1.0)
        cov = float(np.dot(xc[k:], xc[:-k])) / n
        var += 2.0 * w * cov
    se = math.sqrt(max(var, 1e-15) / n)
    return mean / se if se > 1e-15 else 0.0


def _randomized_quantile_pit(
    quantiles: np.ndarray,
    y: np.ndarray,
    levels: tuple[float, ...],
    *,
    seed: int = 0,
) -> np.ndarray:
    """PIT aleatorizado exacto (Czado-Gneiting-Held 2009): Uniform(0,1) exacto en oráculo."""
    q_sorted = np.sort(np.asarray(quantiles, dtype=np.float64), axis=1)
    y_arr = np.asarray(y, dtype=np.float64)
    edges = np.concatenate(([0.0], np.asarray(levels, dtype=np.float64), [1.0]))
    bin_idx = np.sum(q_sorted <= y_arr[:, None], axis=1)
    rng = np.random.default_rng(seed)
    lo = edges[bin_idx]
    hi = edges[bin_idx + 1]
    return lo + rng.random(len(y_arr)) * (hi - lo)


def _normalize_adv_usd(liquidity: np.ndarray | None, n: int) -> np.ndarray:
    """Normaliza liquidez a ADV en USD (acepta USD crudos >50, log-ADV [6..30] o z-score)."""
    if liquidity is None:
        return np.full(n, 1e7, dtype=np.float64)
    l_raw = np.nan_to_num(np.asarray(liquidity, dtype=np.float64), nan=0.0)
    l_med = float(np.median(l_raw)) if len(l_raw) else 0.0
    if l_med > 50.0:
        return np.clip(l_raw, 1e4, 1e12)
    if l_med >= 6.0:
        return np.exp(np.clip(l_raw, 8.0, 25.0))
    return 1e7 * np.exp(np.clip(l_raw, -4.0, 4.0))


def _compute_portfolio_var05_backtest(
    *,
    y_arr: np.ndarray,
    q05: np.ndarray,
    q50: np.ndarray,
    q95: np.ndarray | None,
    w_all: np.ndarray,
    bounds: list[tuple[int, int]],
    exposures: np.ndarray | None,
    factor_cov: np.ndarray | None,
    var_cov_mode: str,
    var_ewma_decay: float,
) -> tuple[np.ndarray, list[float], str]:
    """Backtest causal de VaR 5 % de cartera incorporando covarianza factorial o EWMA causal."""
    exp_arr: np.ndarray | None = None
    if exposures is not None:
        exp_arr = np.nan_to_num(np.asarray(exposures, dtype=np.float64), nan=0.0)
        if exp_arr.ndim == 1:
            exp_arr = exp_arr[:, None]

    fc_arr: np.ndarray | None = None
    if factor_cov is not None:
        fc_arr = np.asarray(factor_cov, dtype=np.float64)
        if fc_arr.ndim not in (2, 3):
            raise ValueError(
                f"factor_cov debe ser 2D (K,K) o 3D (T,K,K); recibido ndim={fc_arr.ndim}"
            )

    if var_cov_mode == "independent":
        resolved_mode = "independent"
    elif var_cov_mode == "ewma_portfolio":
        resolved_mode = "ewma_portfolio"
    elif exp_arr is not None and fc_arr is not None:
        resolved_mode = "explicit_factor"
    elif exp_arr is not None:
        resolved_mode = "causal_factor_ewma"
    else:
        resolved_mode = "causal_corr_ewma"

    # Detectar si `exposures` carece de columna constante para añadir intercepto de mercado
    add_intercept = False
    k_eff = 0
    if resolved_mode == "causal_factor_ewma" and exp_arr is not None:
        k_raw = exp_arr.shape[1]
        first_b = None
        for lo, hi in bounds:
            if hi - lo >= 2:
                first_b = exp_arr[lo:hi]
                break
        has_const = first_b is not None and bool(
            np.any((np.std(first_b, axis=0) < 1e-6) & (np.abs(np.mean(first_b, axis=0)) > 1e-6))
        )
        add_intercept = not has_const
        k_eff = k_raw + (1 if add_intercept else 0)

    # Estados causales (actualizados SOLO DESPUÉS de evaluar el breach de la sesión actual)
    n_obs_fac = 0
    ewma_ff = np.zeros((k_eff, k_eff), dtype=np.float64) if k_eff > 0 else np.empty((0, 0))
    ewma_noise = np.zeros((k_eff, k_eff), dtype=np.float64) if k_eff > 0 else np.empty((0, 0))
    ewma_pred_var = 0.0
    ewma_emp_var = 0.0

    n_obs_corr = 0
    ewma_num = 1.0
    ewma_den = 1.0
    max_eff_n = 1.0 / max(1.0 - var_ewma_decay, 1e-6)

    n_obs_port = 0
    ewma_port_var = 0.0

    port_viol_list: list[bool] = []
    port_excess_list: list[float] = []

    for s_idx, (lo, hi) in enumerate(bounds):
        if hi <= lo:
            continue
        wt_s = w_all[lo:hi]
        if float(np.sum(np.abs(wt_s))) <= 1e-12:
            wt_s = np.full(hi - lo, 1.0 / (hi - lo), dtype=np.float64)
        y_s = y_arr[lo:hi]
        r_p = float(np.dot(wt_s, y_s))
        c_s = q50[lo:hi]
        mu_p = float(np.dot(wt_s, c_s))
        sd_left = np.maximum(c_s - q05[lo:hi], 1e-8) / _Z_05
        if q95 is not None:
            sd_right = np.maximum(q95[lo:hi] - c_s, 1e-8) / _Z_05
        else:
            sd_right = sd_left
        sd_dir = np.where(wt_s >= 0.0, sd_left, sd_right)
        sigma_indep_sq = max(float(np.sum((wt_s * sd_dir) ** 2)), 1e-16)
        max_sigma = max(float(np.sum(np.abs(wt_s) * sd_dir)), 1e-8)

        b_aug: np.ndarray | None = None
        if resolved_mode == "explicit_factor" and exp_arr is not None and fc_arr is not None:
            b_s = exp_arr[lo:hi]
            sig_f = fc_arr if fc_arr.ndim == 2 else fc_arr[min(s_idx, fc_arr.shape[0] - 1)]
            if sig_f.shape != (b_s.shape[1], b_s.shape[1]):
                raise ValueError(
                    f"factor_cov shape {sig_f.shape} incompatible con exposures K={b_s.shape[1]}"
                )
            sig_f_sym = 0.5 * (sig_f + sig_f.T)
            bw = b_s.T @ wt_s
            port_fac_var = float(max(bw @ sig_f_sym @ bw, 0.0))
            asset_fac_var = np.maximum(np.einsum("ij,jk,ik->i", b_s, sig_f_sym, b_s), 0.0)
            idio_var = np.maximum(sd_dir * sd_dir - asset_fac_var, 0.0)
            port_idio_var = float(np.sum((wt_s * wt_s) * idio_var))
            sigma_p = math.sqrt(max(port_fac_var + port_idio_var, 1e-16))
        elif resolved_mode == "causal_factor_ewma" and exp_arr is not None:
            b_s = exp_arr[lo:hi]
            b_aug = (
                np.column_stack((b_s, np.ones(hi - lo, dtype=np.float64)))
                if add_intercept
                else b_s
            )
            if n_obs_fac == 0:
                sigma_p = math.sqrt(sigma_indep_sq)
            else:
                raw_cov = ewma_ff - ewma_noise
                raw_cov = 0.5 * (raw_cov + raw_cov.T)
                eigvals, eigvecs = np.linalg.eigh(raw_cov)
                sig_f_hat = (eigvecs * np.maximum(eigvals, 0.0)) @ eigvecs.T
                gamma = float(k_eff) / float(n_obs_fac + k_eff)
                sig_f_hat = (1.0 - gamma) * sig_f_hat + gamma * np.diag(np.diag(sig_f_hat))
                c_cal_sq = ewma_pred_var / max(ewma_emp_var, 1e-16)
                sig_f_scaled = c_cal_sq * sig_f_hat
                bw = b_aug.T @ wt_s
                port_fac_var = float(max(bw @ sig_f_scaled @ bw, 0.0))
                asset_fac_var = np.maximum(
                    np.einsum("ij,jk,ik->i", b_aug, sig_f_scaled, b_aug), 0.0
                )
                idio_var = np.maximum(sd_dir * sd_dir - asset_fac_var, 0.0)
                port_idio_var = float(np.sum((wt_s * wt_s) * idio_var))
                var_p = min(port_fac_var + port_idio_var, max_sigma * max_sigma)
                sigma_p = math.sqrt(max(var_p, 1e-16))
        elif resolved_mode == "causal_corr_ewma":
            sigma_indep = math.sqrt(sigma_indep_sq)
            eff_n = min(float(n_obs_corr), max_eff_n)
            kappa_sq = (eff_n * ewma_num + 1.0) / max(eff_n * ewma_den + 1.0, 1e-12)
            sigma_p = min(max(sigma_indep * math.sqrt(max(kappa_sq, 1e-6)), 1e-8), max_sigma)
        elif resolved_mode == "ewma_portfolio":
            eff_n = min(float(n_obs_port), max_eff_n)
            var_p = (eff_n * ewma_port_var + sigma_indep_sq) / (eff_n + 1.0)
            sigma_p = math.sqrt(max(var_p, 1e-16))
        else:
            sigma_p = math.sqrt(sigma_indep_sq)

        var05_p = mu_p - _Z_05 * sigma_p
        breached = r_p < var05_p
        port_viol_list.append(breached)
        if breached:
            port_excess_list.append(r_p - var05_p)

        # Actualizaciones causales post-observación de sesión `s_idx`
        if resolved_mode == "causal_factor_ewma" and b_aug is not None:
            e_s = y_s - c_s
            n_s = len(e_s)
            alpha_w = max(1.0 - var_ewma_decay, 1.0 / float(n_obs_fac + 1))
            pred_v_s = float(np.mean(sd_dir * sd_dir))
            emp_v_s = float(np.mean(e_s * e_s))
            ewma_pred_var = (1.0 - alpha_w) * ewma_pred_var + alpha_w * pred_v_s
            ewma_emp_var = (1.0 - alpha_w) * ewma_emp_var + alpha_w * emp_v_s
            if n_s > k_eff + 1:
                gram = b_aug.T @ b_aug
                ridge = 1e-4 * max(float(np.trace(gram)) / float(k_eff), 1.0)
                gram_reg = gram + ridge * np.eye(k_eff, dtype=np.float64)
                try:
                    f_hat = np.linalg.solve(gram_reg, b_aug.T @ e_s)
                    eps_hat = e_s - b_aug @ f_hat
                    gram_inv = np.linalg.inv(gram_reg)
                    meat = (b_aug.T * (eps_hat * eps_hat)[None, :]) @ b_aug
                    v_ols = gram_inv @ meat @ gram_inv
                    ff_s = np.outer(f_hat, f_hat)
                    ewma_ff = (1.0 - alpha_w) * ewma_ff + alpha_w * ff_s
                    ewma_noise = (1.0 - alpha_w) * ewma_noise + alpha_w * v_ols
                    n_obs_fac += 1
                except np.linalg.LinAlgError:
                    pass
        elif resolved_mode == "causal_corr_ewma":
            u_s = r_p - mu_p
            e_s = y_s - c_s
            z_p_sq = (u_s * u_s) / sigma_indep_sq
            z_asset_sq = float(np.mean((e_s * e_s) / np.maximum(sd_dir * sd_dir, 1e-16)))
            alpha_w = max(1.0 - var_ewma_decay, 1.0 / float(n_obs_corr + 1))
            ewma_num = (1.0 - alpha_w) * ewma_num + alpha_w * z_p_sq
            ewma_den = (1.0 - alpha_w) * ewma_den + alpha_w * z_asset_sq
            n_obs_corr += 1
        elif resolved_mode == "ewma_portfolio":
            u_s = r_p - mu_p
            alpha_w = max(1.0 - var_ewma_decay, 1.0 / float(n_obs_port + 1))
            ewma_port_var = (1.0 - alpha_w) * ewma_port_var + alpha_w * (u_s * u_s)
            n_obs_port += 1

    return np.asarray(port_viol_list, dtype=bool), port_excess_list, resolved_mode


def evaluate_granular_six_dimensions(
    *,
    y: np.ndarray,
    pred: np.ndarray,
    bounds: list[tuple[int, int]],
    quantiles: np.ndarray | None = None,
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS_9,
    probabilities: np.ndarray | None = None,
    volatility: np.ndarray | None = None,
    liquidity: np.ndarray | None = None,
    exposures: np.ndarray | None = None,
    portfolio_weights: np.ndarray | None = None,
    asset_ids: np.ndarray | None = None,
    loop_diagnostics: dict[str, float] | None = None,
    n_trials: int = 1,
    trial_sharpes: np.ndarray | list[float] | None = None,
    sr_std: float | None = None,
    sr_var: float | None = None,
    trial_returns_matrix: np.ndarray | None = None,
    spread_bps: float = 5.0,
    impact_eta: float = 0.15,
    aum_musd: float = 50.0,
    max_adv_participation: float = 0.05,
    execution_mode: Literal["close_to_close", "open_to_close_200pct"] = "close_to_close",
    market: Literal["US", "CN"] = "US",
    cn_stamp_tax_bps: float = 10.0,
    cn_limit_lock_threshold: float = 0.098,
    cn_limit_lock_slippage_bps: float = 50.0,
    survivorship_disclosure: str | None = None,
    pit_seed: int = 0,
    factor_cov: np.ndarray | None = None,
    var_cov_mode: Literal["auto", "factor_ewma", "ewma_portfolio", "independent"] = "auto",
    var_ewma_decay: float = 0.96,
) -> dict[str, dict[str, object]]:
    """Genera el tablero granular completo de 6 dimensiones (32+ métricas)."""
    if execution_mode not in EXECUTION_MODES:
        raise ValueError(
            f"execution_mode inválido {execution_mode!r}; válidos: {EXECUTION_MODES}"
        )
    if market not in MARKETS:
        raise ValueError(f"market inválido {market!r}; válidos: {MARKETS}")
    if var_cov_mode not in VAR_COV_MODES:
        raise ValueError(f"var_cov_mode inválido {var_cov_mode!r}; válidos: {VAR_COV_MODES}")
    if not (0.0 < float(var_ewma_decay) < 1.0):
        raise ValueError("var_ewma_decay debe pertenecer a (0, 1)")
    y_arr = np.asarray(y, dtype=np.float64)
    p_arr = np.asarray(pred, dtype=np.float64)
    n = len(y_arr)
    err = y_arr - p_arr

    # --- DIM 1: Valor puntual y precio reconstruido ---
    mae = float(np.mean(np.abs(err)))
    medae = float(np.median(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    huber_d = 0.02
    abs_e = np.abs(err)
    huber_terms = np.where(abs_e <= huber_d, 0.5 * err * err, huber_d * (abs_e - 0.5 * huber_d))
    huber = float(np.mean(huber_terms))
    # sMAPE sobre precio reconstruido P_hat / P_0 = exp(pred) vs P_1 / P_0 = exp(y)
    p_ratio = np.exp(np.clip(p_arr, -1.0, 1.0))
    y_ratio = np.exp(np.clip(y_arr, -1.0, 1.0))
    smape = float(np.mean(2.0 * np.abs(p_ratio - y_ratio) / np.maximum(p_ratio + y_ratio, 1e-8)))
    ss_err = float(np.sum(err * err))
    # Gu-Kelly-Xiu (2020) OOS R^2 estricto: benchmark cero `sum(y^2)` sin lookahead `mean(y)`
    ss_zero = float(np.sum(y_arr * y_arr))
    ss_centered = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    oos_r2 = 1.0 - ss_err / max(ss_zero, 1e-12)
    oos_r2_centered = 1.0 - ss_err / max(ss_centered, 1e-12)
    # Regresión Mincer-Zarnowitz: y = a + b * pred
    var_p = float(np.var(p_arr))
    if var_p > 1e-12:
        mz_slope = float(np.cov(p_arr, y_arr, ddof=0)[0, 1] / var_p)
        mz_intercept = float(np.mean(y_arr) - mz_slope * np.mean(p_arr))
    else:
        mz_slope, mz_intercept = 0.0, float(np.mean(y_arr))

    dim1 = dict(
        mae_bps=mae * 1e4,
        medae_bps=medae * 1e4,
        rmse_bps=rmse * 1e4,
        huber_bps=huber * 1e4,
        price_smape_pct=smape * 100.0,
        oos_r2=oos_r2,
        oos_r2_centered=oos_r2_centered,
        mz_slope=mz_slope,
        mz_intercept_bps=mz_intercept * 1e4,
    )

    # --- DIM 2: Ranking transversal y alfa puro ---
    rank_ics, pearson_ics, spreads_10 = [], [], []
    for lo, hi in bounds:
        if hi - lo < 10:
            continue
        ps, ys = p_arr[lo:hi], y_arr[lo:hi]
        ric = oa.rank_ic_vector(ps, ys)
        if math.isfinite(ric):
            rank_ics.append(ric)
        if float(np.std(ps)) > 1e-12 and float(np.std(ys)) > 1e-12:
            pearson_ics.append(float(np.corrcoef(ps, ys)[0, 1]))
        order = np.argsort(ps, kind="stable")
        k10 = max(1, (hi - lo) // 10)
        spreads_10.append(float(ys[order[-k10:]].mean() - ys[order[:k10]].mean()))

    ric_arr = np.asarray(rank_ics, dtype=np.float64)
    ric_mean = float(ric_arr.mean()) if len(ric_arr) else 0.0
    ric_std = float(ric_arr.std(ddof=1)) if len(ric_arr) > 1 else 0.0
    if exposures is not None:
        p_neut = fn.neutralize_sessions(p_arr, exposures, bounds, ridge_lambda=1.0, gamma=1.0)
        neut_ics = [
            oa.rank_ic_vector(p_neut[lo:hi], y_arr[lo:hi])
            for lo, hi in bounds
            if hi - lo >= 10
        ]
        neut_arr = np.asarray([v for v in neut_ics if math.isfinite(v)], dtype=np.float64)
        pure_ic = float(neut_arr.mean()) if len(neut_arr) else 0.0
        neut_sd = float(neut_arr.std(ddof=1)) if len(neut_arr) > 1 else 0.0
        pure_icir = float(pure_ic / neut_sd) if neut_sd > 1e-12 else 0.0
        pure_hit = float(np.mean(neut_arr > 0.0)) if len(neut_arr) else 0.0
    else:
        pure_ic = ric_mean
        pure_icir = ric_mean / ric_std if ric_std > 1e-12 else 0.0
        pure_hit = float(np.mean(ric_arr > 0.0)) if len(ric_arr) else 0.0

    dim2 = dict(
        sessions=len(ric_arr),
        rank_ic_mean=ric_mean,
        rank_ic_std=ric_std,
        rank_icir=ric_mean / ric_std if ric_std > 1e-12 else 0.0,
        rank_ic_t_nw=_newey_west_t(ric_arr),
        rank_ic_pos_rate=float(np.mean(ric_arr > 0.0)) if len(ric_arr) else 0.0,
        pearson_ic_mean=float(np.mean(pearson_ics)) if pearson_ics else 0.0,
        pure_alpha_ic_mean=pure_ic,
        pure_alpha_icir=pure_icir,
        pure_alpha_pos_rate=pure_hit,
        decile_spread_bps=float(np.mean(spreads_10)) * 1e4 if spreads_10 else 0.0,
    )

    # --- DIM 3: Tendencias, signo ponderado y puntos de giro ---
    sign_ok = (p_arr * y_arr) > 0.0
    sign_acc = float(np.mean(sign_ok))
    mag_w_acc = float(np.sum(np.abs(y_arr) * sign_ok) / max(float(np.sum(np.abs(y_arr))), 1e-12))
    # Correlación de tendencia de media de sesión (proxy de dirección agregada)
    sess_y = np.array([y_arr[lo:hi].mean() for lo, hi in bounds])
    sess_p = np.array([p_arr[lo:hi].mean() for lo, hi in bounds])
    turns_true = np.zeros(len(sess_y), dtype=bool)
    turns_pred = np.zeros(len(sess_y), dtype=bool)
    if len(sess_y) >= 3:
        turns_true[1:] = (sess_y[1:] * sess_y[:-1]) < 0.0
        turns_pred[1:] = (sess_p[1:] * sess_p[:-1]) < 0.0
    tp_hit = int(np.sum(turns_true & turns_pred))
    tp_rec = tp_hit / max(int(turns_true.sum()), 1)
    tp_prec = tp_hit / max(int(turns_pred.sum()), 1)
    dim3 = dict(
        sign_accuracy=sign_acc,
        magnitude_weighted_sign_acc=mag_w_acc,
        turning_point_recall=tp_rec,
        turning_point_precision=tp_prec,
    )

    # Pesos de cartera compartidos por Dim 4 (VaR temporal de cartera) y Dim 6 (economía real)
    if portfolio_weights is None:
        w_all = np.zeros(n, dtype=np.float64)
        for lo, hi in bounds:
            z = p_arr[lo:hi] - np.mean(p_arr[lo:hi])
            s = float(np.sum(np.abs(z)))
            if s > 1e-12:
                w_all[lo:hi] = z / s
    else:
        w_all = np.asarray(portfolio_weights, dtype=np.float64)

    # --- DIM 4: Incertidumbre, colas y calibración ---
    dim4: dict[str, object] = {}
    if quantiles is not None:
        q = np.asarray(quantiles, dtype=np.float64)
        crps_out = discrete_crps(q, y_arr, levels=quantile_levels)
        dim4["crps"] = crps_out["crps"]
        dim4["pinball_mean"] = crps_out["pinball_mean"]
        # Cobertura y Winkler en pares centrales disponibles (80 %, 90 %)
        lv_list = list(quantile_levels)
        for nom, lo_lv, hi_lv in ((0.80, 0.10, 0.90), (0.90, 0.05, 0.95)):
            if lo_lv in lv_list and hi_lv in lv_list:
                ql = q[:, lv_list.index(lo_lv)]
                qh = q[:, lv_list.index(hi_lv)]
                alpha = 1.0 - nom
                cov = float(np.mean((y_arr >= ql) & (y_arr <= qh)))
                winkler = float(
                    np.mean(
                        (qh - ql)
                        + (2.0 / alpha) * np.maximum(ql - y_arr, 0.0)
                        + (2.0 / alpha) * np.maximum(y_arr - qh, 0.0)
                    )
                )
                dim4[f"cov_{int(nom*100)}"] = cov
                dim4[f"winkler_{int(nom*100)}"] = winkler
        # PIT aleatorizado continuo exacto (evita p=0 espurio de 10 bins discretos)
        pit = _randomized_quantile_pit(q, y_arr, quantile_levels, seed=pit_seed)
        ks_stat, ks_p = stats.kstest(pit, "uniform")
        dim4["pit_ks_stat"] = float(ks_stat)
        dim4["pit_ks_p"] = float(ks_p)
        # VaR 5 % temporal de cartera (T sesiones) con covarianza factorial o EWMA causal
        if 0.05 in lv_list:
            q05 = q[:, lv_list.index(0.05)]
            q50 = q[:, lv_list.index(0.50)] if 0.50 in lv_list else p_arr
            q95 = q[:, lv_list.index(0.95)] if 0.95 in lv_list else None
            port_viol, port_excess_list, resolved_cov_mode = _compute_portfolio_var05_backtest(
                y_arr=y_arr,
                q05=q05,
                q50=q50,
                q95=q95,
                w_all=w_all,
                bounds=bounds,
                exposures=exposures,
                factor_cov=factor_cov,
                var_cov_mode=var_cov_mode,
                var_ewma_decay=float(var_ewma_decay),
            )
            kup = kupiec_pof_test(port_viol, 0.05)
            chr_t = christoffersen_independence_test(port_viol)
            dim4["var05_rate"] = kup["rate"]
            dim4["var05_n_obs"] = len(port_viol)
            dim4["var05_cov_mode"] = resolved_cov_mode
            dim4["asset_var05_rate"] = float(np.mean(y_arr < q05))
            dim4["kupiec_var05_p"] = kup["p_value"]
            dim4["christoffersen_var05_p"] = chr_t["p_value"]
            dim4["cvar05_excess_bps"] = (
                float(np.mean(port_excess_list)) * 1e4 if port_excess_list else 0.0
            )
    if probabilities is not None:
        pr = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        yb = (y_arr > 0.0).astype(np.float64)
        dim4["brier"] = float(np.mean((pr - yb) ** 2))
        dim4["log_loss"] = float(-np.mean(yb * np.log(pr) + (1.0 - yb) * np.log(1.0 - pr)))
        bins = np.clip((pr * 10).astype(np.int64), 0, 9)
        ece = 0.0
        for b_i in range(10):
            m = bins == b_i
            if m.any():
                ece += float(m.mean()) * abs(float(pr[m].mean()) - float(yb[m].mean()))
        dim4["ece"] = ece

    # --- DIM 5: Autopsia causal del error ---
    tot_var = max(float(np.var(err)), 1e-12)
    if exposures is not None:
        err_neut = fn.neutralize_sessions(err, exposures, bounds, ridge_lambda=0.1, gamma=1.0)
        idio_var = float(np.var(err_neut))
        fac_share = float(np.clip(1.0 - idio_var / tot_var, 0.0, 1.0))
    else:
        idio_var = tot_var
        fac_share = 0.0
    dim5: dict[str, object] = dict(
        total_error_std_bps=math.sqrt(tot_var) * 1e4,
        idio_error_std_bps=math.sqrt(max(idio_var, 0.0)) * 1e4,
        factor_var_share=fac_share,
    )
    if loop_diagnostics:
        dim5.update(loop_diagnostics)

    # --- DIM 6: Economía real con impacto Almgren-Chriss, modo ejecución, CN, DSR y PBO ---
    if volatility is None:
        vol_all = np.full(n, 0.02, dtype=np.float64)
    else:
        v_raw = np.nan_to_num(np.asarray(volatility, dtype=np.float64), nan=0.02)
        if float(np.min(v_raw)) < 0.0 or float(np.median(v_raw)) > 0.25:
            # Z-score de log-volatilidad -> mapear alrededor de 2 % diario
            vol_all = 0.02 * np.exp(0.5 * np.clip(v_raw, -3.0, 3.0))
        else:
            vol_all = np.clip(v_raw, 0.003, 0.25)

    adv_usd = _normalize_adv_usd(liquidity, n)
    aids_all = None if asset_ids is None else np.asarray(asset_ids)
    spread_rate = float(spread_bps) * 1e-4
    stamp_rate = (float(cn_stamp_tax_bps) * 1e-4) if market == "CN" else 0.0
    lock_slip_rate = (float(cn_limit_lock_slippage_bps) * 1e-4) if market == "CN" else 0.0
    aum_usd = float(aum_musd) * 1e6

    gross_rets: list[float] = []
    turnovers: list[float] = []
    exited_turnovers: list[float] = []
    linear_costs_1x: list[float] = []
    impact_costs_1x: list[float] = []
    stamp_costs_1x: list[float] = []
    max_parts: list[float] = []
    limit_lock_rates: list[float] = []

    w_prev_seg: np.ndarray | None = None
    prev_w_by_id: dict[Any, float] = {}
    prev_vol_by_id: dict[Any, float] = {}
    prev_adv_by_id: dict[Any, float] = {}

    for lo, hi in bounds:
        wt = w_all[lo:hi]
        yt = y_arr[lo:hi]
        vt = vol_all[lo:hi]
        adv_t = np.maximum(adv_usd[lo:hi], 1e4)
        gross_rets.append(float(np.dot(wt, yt)))

        locked = (np.abs(yt) >= cn_limit_lock_threshold) if market == "CN" else np.zeros(
            len(yt), dtype=bool
        )
        limit_lock_rates.append(float(np.mean(locked)) if len(locked) else 0.0)

        if execution_mode == "open_to_close_200pct":
            abs_w = np.abs(wt)
            to = float(2.0 * np.sum(abs_w))
            exited_to = 0.0
            leg_usd = abs_w * aum_usd
            part_raw = leg_usd / adv_t
            part = np.clip(part_raw, 0.0, 1.0)
            c_spread = float(2.0 * np.sum(abs_w) * spread_rate)
            c_stamp = float(np.sum(abs_w) * stamp_rate)
            c_lock = float(2.0 * np.sum(abs_w[locked]) * lock_slip_rate)
            c_lin = c_spread + c_stamp + c_lock
            c_imp = float(2.0 * np.sum(abs_w * impact_eta * vt * np.sqrt(part)))
            max_part = float(np.max(part_raw)) if len(part_raw) else 0.0
        else:
            if aids_all is not None:
                ids_t = aids_all[lo:hi]
                curr_keys = [
                    x.item() if isinstance(x, np.generic) else x for x in ids_t.tolist()
                ]
                w_prev_al = np.array(
                    [prev_w_by_id.get(aid, 0.0) for aid in curr_keys], dtype=np.float64
                )
                dw_act = np.abs(wt - w_prev_al)
                sell_act = np.maximum(w_prev_al - wt, 0.0)
                curr_set = set(curr_keys)
                ex_ids = [aid for aid in prev_w_by_id if aid not in curr_set]
                if ex_ids:
                    ex_w = np.array([prev_w_by_id[aid] for aid in ex_ids], dtype=np.float64)
                    ex_dw = np.abs(ex_w)
                    ex_sell = np.maximum(ex_w, 0.0)
                    ex_vol = np.array(
                        [prev_vol_by_id.get(aid, 0.02) for aid in ex_ids], dtype=np.float64
                    )
                    ex_adv = np.maximum(
                        np.array(
                            [prev_adv_by_id.get(aid, 1e7) for aid in ex_ids], dtype=np.float64
                        ),
                        1e4,
                    )
                    ex_part_raw = (ex_dw * aum_usd) / ex_adv
                    ex_part = np.clip(ex_part_raw, 0.0, 1.0)
                    exited_to = float(np.sum(ex_dw))
                    ex_c_spread = float(np.sum(ex_dw) * spread_rate)
                    ex_c_stamp = float(np.sum(ex_sell) * stamp_rate)
                    ex_c_imp = float(np.sum(ex_dw * impact_eta * ex_vol * np.sqrt(ex_part)))
                    ex_max_part = float(np.max(ex_part_raw))
                else:
                    exited_to = 0.0
                    ex_c_spread = 0.0
                    ex_c_stamp = 0.0
                    ex_c_imp = 0.0
                    ex_max_part = 0.0

                prev_w_by_id = {
                    aid: float(wv) for aid, wv in zip(curr_keys, wt, strict=True)
                }
                prev_vol_by_id = {
                    aid: float(vv) for aid, vv in zip(curr_keys, vt, strict=True)
                }
                prev_adv_by_id = {
                    aid: float(av) for aid, av in zip(curr_keys, adv_t, strict=True)
                }
            else:
                same_prev = w_prev_seg is not None and len(w_prev_seg) == len(wt)
                w_prev_al = w_prev_seg if same_prev else np.zeros_like(wt)
                dw_act = np.abs(wt - w_prev_al)
                sell_act = np.maximum(w_prev_al - wt, 0.0)
                exited_to = 0.0
                ex_c_spread = 0.0
                ex_c_stamp = 0.0
                ex_c_imp = 0.0
                ex_max_part = 0.0

            to = float(np.sum(dw_act)) + exited_to
            trade_usd = dw_act * aum_usd
            part_raw = trade_usd / adv_t
            part = np.clip(part_raw, 0.0, 1.0)
            c_spread = float(np.sum(dw_act) * spread_rate) + ex_c_spread
            c_stamp = float(np.sum(sell_act) * stamp_rate) + ex_c_stamp
            c_lock = float(np.sum(dw_act[locked]) * lock_slip_rate)
            c_lin = c_spread + c_stamp + c_lock
            c_imp = float(np.sum(dw_act * impact_eta * vt * np.sqrt(part))) + ex_c_imp
            act_max_part = float(np.max(part_raw)) if len(part_raw) else 0.0
            max_part = max(act_max_part, ex_max_part)

        turnovers.append(to)
        exited_turnovers.append(exited_to)
        linear_costs_1x.append(c_lin)
        impact_costs_1x.append(c_imp)
        stamp_costs_1x.append(c_stamp)
        max_parts.append(max_part)
        w_prev_seg = wt

    g_arr = np.asarray(gross_rets, dtype=np.float64)
    c_lin_arr = np.asarray(linear_costs_1x, dtype=np.float64)
    c_imp_arr = np.asarray(impact_costs_1x, dtype=np.float64)
    c_stamp_arr = np.asarray(stamp_costs_1x, dtype=np.float64)
    c_arr = c_lin_arr + c_imp_arr
    to_arr = np.asarray(turnovers, dtype=np.float64)
    ex_to_arr = np.asarray(exited_turnovers, dtype=np.float64)
    mp_arr = np.asarray(max_parts, dtype=np.float64)

    def _sr(r_s: np.ndarray) -> float:
        sd = float(np.std(r_s, ddof=1)) if len(r_s) > 1 else 0.0
        return float(np.mean(r_s) / sd * math.sqrt(252.0)) if sd > 1e-12 else 0.0

    net1 = g_arr - c_arr
    net2 = g_arr - 2.0 * c_arr
    net3 = g_arr - 3.0 * c_arr
    eq = np.cumsum(net1)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    dd = peak - eq
    max_dd = float(dd.max()) if len(dd) else 0.0
    down = net1[net1 < 0.0]
    down_ms = float(np.mean(down * down)) if len(down) > 0 else 0.0
    sortino = (
        float(np.mean(net1) / math.sqrt(down_ms) * math.sqrt(252.0))
        if down_ms > 1e-15
        else _sr(net1)
    )
    breakeven = float(np.mean(g_arr) / max(float(np.mean(to_arr)), 1e-8)) * 1e4

    # Capacidad exacta: separa coste lineal fijo (c_lin) de impacto sqrt(AUM) (c_imp) + tope ADV
    mean_g = max(float(np.mean(g_arr)), 0.0)
    mean_c_lin = float(np.mean(c_lin_arr)) if len(c_lin_arr) else 0.0
    mean_c_imp = float(np.mean(c_imp_arr)) if len(c_imp_arr) else 0.0
    mean_c_tot = float(np.mean(c_arr)) if len(c_arr) else 0.0
    alpha_headroom = max(0.5 * mean_g - mean_c_lin, 0.0)
    cap_alpha_musd = (
        float(aum_musd * ((alpha_headroom / max(mean_c_imp, 1e-12)) ** 2))
        if alpha_headroom > 0.0
        else 0.0
    )
    mean_max_part = float(np.mean(mp_arr)) if len(mp_arr) else 0.0
    cap_adv_musd = (
        float(aum_musd * (max_adv_participation / mean_max_part))
        if mean_max_part > 1e-12
        else 1e5
    )
    cap_musd = min(cap_alpha_musd, cap_adv_musd, 1e5)
    safety_margin = float(mean_g / max(mean_c_tot, 1e-12)) if mean_g > 0.0 else 0.0

    # DSR empírico + CSCV PBO cuando se aporta matriz de ensayos
    pbo_val: float | None = None
    eff_trial_sharpes = trial_sharpes
    if trial_returns_matrix is not None:
        mat_t = np.asarray(trial_returns_matrix, dtype=np.float64)
        pbo_val = float(cscv_pbo(mat_t)["pbo"])
        if eff_trial_sharpes is None and sr_std is None and sr_var is None and mat_t.ndim == 2:
            col_sd = np.std(mat_t, axis=0, ddof=1)
            valid_cols = col_sd > 1e-12
            if int(np.sum(valid_cols)) >= 2:
                eff_trial_sharpes = np.mean(mat_t[:, valid_cols], axis=0) / col_sd[valid_cols]

    dsr_out = deflated_sharpe_ratio(
        net1,
        n_trials=n_trials,
        trial_sharpes=eff_trial_sharpes,
        sr_std=sr_std,
        sr_var=sr_var,
    )

    dim6 = dict(
        gross_mean_bps=float(np.mean(g_arr)) * 1e4,
        gross_sharpe=_sr(g_arr),
        net_sharpe_1x=_sr(net1),
        net_sharpe_2x=_sr(net2),
        net_sharpe_3x=_sr(net3),
        sortino_1x=sortino,
        max_drawdown_1x=max_dd,
        calmar_1x=float(np.mean(net1) * 252.0 / max(max_dd, 1e-6)),
        turnover_mean=float(np.mean(to_arr)),
        exited_turnover_mean=float(np.mean(ex_to_arr)),
        linear_cost_bps=mean_c_lin * 1e4,
        impact_cost_bps=mean_c_imp * 1e4,
        safety_margin=safety_margin,
        breakeven_bps=breakeven,
        capacity_alpha_musd=min(cap_alpha_musd, 1e5),
        capacity_adv_cap_musd=min(cap_adv_musd, 1e5),
        capacity_musd=cap_musd,
        dsr=dsr_out["dsr"],
        n_trials=int(dsr_out["n_trials_effective"]),
        pbo=pbo_val,
        execution_mode=execution_mode,
        market=market,
        cn_stamp_tax_bps_cost=float(np.mean(c_stamp_arr)) * 1e4,
        cn_limit_lock_rate=float(np.mean(limit_lock_rates)) if limit_lock_rates else 0.0,
        cn_t1_violation=bool(market == "CN" and execution_mode == "open_to_close_200pct"),
        survivorship_disclosure=survivorship_disclosure or DEFAULT_SURVIVORSHIP_DISCLOSURE,
    )

    return dict(
        dim1_value=dim1,
        dim2_ranking=dim2,
        dim3_trend=dim3,
        dim4_uncertainty=dim4,
        dim5_error_autopsy=dim5,
        dim6_economics=dim6,
    )


def evaluate_liquid_top_third_stratification(
    *,
    y: np.ndarray,
    pred: np.ndarray,
    bounds: list[tuple[int, int]],
    liquidity: np.ndarray,
    top_fraction: float = 1.0 / 3.0,
    quantiles: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    volatility: np.ndarray | None = None,
    exposures: np.ndarray | None = None,
    asset_ids: np.ndarray | None = None,
    **eval_kwargs: Any,
) -> dict[str, object]:
    """Evalúa el tercio más líquido de cada sesión para co-validar sesgo de supervivencia."""
    if not (0.0 < float(top_fraction) <= 1.0):
        raise ValueError("top_fraction debe pertenecer a (0, 1]")
    y_arr = np.asarray(y, dtype=np.float64)
    p_arr = np.asarray(pred, dtype=np.float64)
    adv_usd = _normalize_adv_usd(liquidity, len(y_arr))

    keep_idx: list[int] = []
    new_bounds: list[tuple[int, int]] = []
    cursor = 0
    for lo, hi in bounds:
        m = hi - lo
        if m <= 0:
            continue
        k_top = max(1, int(math.ceil(m * float(top_fraction))))
        order = np.argsort(adv_usd[lo:hi], kind="stable")
        chosen = np.sort(order[-k_top:]) + lo
        keep_idx.extend(chosen.tolist())
        new_bounds.append((cursor, cursor + len(chosen)))
        cursor += len(chosen)

    idx = np.asarray(keep_idx, dtype=np.int64)
    rep = evaluate_granular_six_dimensions(
        y=y_arr[idx],
        pred=p_arr[idx],
        bounds=new_bounds,
        quantiles=None if quantiles is None else np.asarray(quantiles)[idx],
        probabilities=None if probabilities is None else np.asarray(probabilities)[idx],
        volatility=None if volatility is None else np.asarray(volatility)[idx],
        liquidity=adv_usd[idx],
        exposures=None if exposures is None else np.asarray(exposures)[idx],
        asset_ids=None if asset_ids is None else np.asarray(asset_ids)[idx],
        **eval_kwargs,
    )
    return dict(
        **rep,
        selected_rows=len(idx),
        selected_fraction=float(len(idx) / max(len(y_arr), 1)),
    )
