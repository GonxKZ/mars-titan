"""Neutralización causal de factores transversales por sesión (`CausalFactorNeutralizer`).

Motivación empírica (sesión 2013 `v10`, `fiabilidad/2013/resumen.md`):
- El IC líquido diario de S2 tiene media 0,062 pero desviación estándar diaria σ = 0,091 frente
  al suelo idiosincrático 1/√N ≈ 0,034 (N ≈ 860): el ~86 % de la varianza diaria del IC proviene
  de exposiciones factoriales no deseadas (liquidez, volatilidad, beta, momentum, PCA).
- Neutralizar ortogonalmente (con contracción Ridge) la señal contra exposiciones estrictamente
  causales conocidas en `t` reduce σ_IC y eleva el ICIR diario y la fracción de sesiones con IC > 0.

Contrato causal:
- Todas las exposiciones `B_t` son observables ANTES de la apertura de la sesión `t`.
- `causal_rolling_pca_exposures` extrae factores estadísticos exclusivamente de retornos ya maduros
  `tau <= t - maturity_lag` (nunca lee retornos contemporáneos ni futuros).
- Los hiperparámetros `(gamma, ridge_lambda)` se fijan exclusivamente en calibración.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mars_titan.evaluation import online_aggregator as oa


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _standardize_exposures(b_raw: np.ndarray) -> np.ndarray:
    """Imputa NaN/Inf con mediana de sesión y tipifica cada columna a media 0 y std 1."""
    b = np.asarray(b_raw, dtype=np.float64)
    if b.ndim == 1:
        b = b[:, None]
    n, k = b.shape
    out = np.zeros((n, k), dtype=np.float64)
    for j in range(k):
        col = b[:, j]
        finite = np.isfinite(col)
        if not np.any(finite):
            continue
        med = float(np.median(col[finite]))
        filled = np.where(finite, col, med)
        mean = float(np.mean(filled))
        centered = filled - mean
        std = float(np.std(centered))
        if std > 1e-12:
            out[:, j] = np.clip(centered / std, -5.0, 5.0)
    return out


def _session_z_f64(values: np.ndarray) -> np.ndarray:
    """Rangos medios tipificados en FP64 para una sesión."""
    ranks = oa.average_ranks(values)
    mean = float(np.mean(ranks))
    centered = ranks - mean
    std = float(np.std(centered))
    if not math.isfinite(std) or std <= 0.0:
        return np.zeros_like(ranks)
    return centered / std


def neutralize_sessions(
    score: np.ndarray,
    exposures: np.ndarray,
    bounds: list[tuple[int, int]],
    *,
    weights: np.ndarray | None = None,
    ridge_lambda: float = 1.0,
    gamma: float = 1.0,
    include_intercept: bool = True,
    rank_input: bool = False,
) -> np.ndarray:
    """Proyección ortogonal Ridge por sesión con mezcla parcial `gamma in [0, 1]`.

    Para cada sesión `[lo, hi)`:
      `b_hat = (B^T W B + lambda * P)^(-1) B^T W s`
      `s_perp = s - B @ b_hat`
      `s_out = (1 - gamma) * s_base + gamma * s_perp`
    donde `P` penaliza los factores pero no el intercepto (si `include_intercept=True`).
    """
    s_all = np.asarray(score, dtype=np.float64)
    b_all = np.asarray(exposures, dtype=np.float64)
    if b_all.ndim == 1:
        b_all = b_all[:, None]
    _require(len(s_all) == len(b_all), "score y exposures deben tener la misma longitud")
    _require(0.0 <= gamma <= 1.0, f"gamma debe estar en [0, 1]; se recibió {gamma!r}")
    _require(ridge_lambda >= 0.0 and math.isfinite(ridge_lambda), "ridge_lambda debe ser >= 0")

    w_all = None if weights is None else np.asarray(weights, dtype=np.float64)
    if w_all is not None:
        _require(len(w_all) == len(s_all), "weights debe tener la misma longitud que score")

    out = np.empty_like(s_all, dtype=np.float64)
    for lo, hi in bounds:
        n = hi - lo
        s_seg = _session_z_f64(s_all[lo:hi]) if rank_input else np.nan_to_num(s_all[lo:hi], nan=0.0)
        if gamma <= 0.0 or n <= 2 or b_all.shape[1] == 0:
            out[lo:hi] = s_seg
            continue
        b_std = _standardize_exposures(b_all[lo:hi])
        # Eliminar columnas degeneradas (varianza cero)
        keep_cols = np.any(b_std != 0.0, axis=0)
        if not np.any(keep_cols):
            out[lo:hi] = s_seg
            continue
        b_mat = b_std[:, keep_cols]
        if include_intercept:
            b_mat = np.column_stack([np.ones(n, dtype=np.float64), b_mat])
        p = b_mat.shape[1]

        if w_all is not None:
            w_seg = w_all[lo:hi]
            w_seg = np.where(np.isfinite(w_seg) & (w_seg > 0.0), w_seg, 0.0)
            w_sum = float(w_seg.sum())
            w_seg = (w_seg / w_sum * n) if w_sum > 0.0 else np.ones(n, dtype=np.float64)
            bw = b_mat * w_seg[:, None]
            gram = bw.T @ b_mat
            rhs = bw.T @ s_seg
        else:
            gram = b_mat.T @ b_mat
            rhs = b_mat.T @ s_seg

        # Regularización Ridge en factores (no en el intercepto cuando lambda > 0)
        if ridge_lambda > 0.0:
            pen = np.eye(p, dtype=np.float64) * (ridge_lambda * n / 100.0)
            if include_intercept:
                pen[0, 0] = 0.0
            gram = gram + pen
        try:
            beta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(gram, rhs, rcond=None)[0]

        s_perp = s_seg - b_mat @ beta
        if gamma >= 1.0:
            out[lo:hi] = s_perp
        else:
            # Mezclar en escala comparable (std unitaria) cuando rank_input o s_seg no trivial
            std_base = float(np.std(s_seg))
            std_perp = float(np.std(s_perp))
            if std_base > 1e-12 and std_perp > 1e-12:
                out[lo:hi] = (1.0 - gamma) * (s_seg / std_base) + gamma * (s_perp / std_perp)
            else:
                out[lo:hi] = (1.0 - gamma) * s_seg + gamma * s_perp
    return out


def causal_rolling_pca_exposures(
    returns_panel: np.ndarray,
    *,
    n_components: int = 5,
    window: int = 63,
    maturity_lag: int = 1,
    min_history: int = 20,
) -> np.ndarray:
    """Extrae `n_components` cargas factoriales PCA estrictamente causales por sesión.

    `returns_panel` tiene forma `(T, N)` (NaN permitido donde un activo no cotiza).
    Para la sesión `t`, solo usa `[max(0, t - maturity_lag - window + 1) .. t - maturity_lag]`,
    jamás toca `t` ni ninguna sesión posterior. Devuelve array `(T, N, n_components)` finito.
    """
    r = np.asarray(returns_panel, dtype=np.float64)
    _require(r.ndim == 2, "returns_panel debe ser 2D (sesiones, activos)")
    _require(n_components >= 1 and window >= 2 and maturity_lag >= 1, "Parámetros PCA inválidos")
    n_sess, n_assets = r.shape
    out = np.zeros((n_sess, n_assets, n_components), dtype=np.float64)
    k = min(n_components, n_assets)

    for t in range(n_sess):
        end_hist = t - maturity_lag + 1  # exclusivo; última fila usada es t - maturity_lag
        if end_hist < min_history:
            continue
        start_hist = max(0, end_hist - window)
        hist = r[start_hist:end_hist]
        # Centrar y estandarizar por activo usando solo la ventana histórica madura
        valid = np.isfinite(hist)
        counts = valid.sum(axis=0)
        active = counts >= max(5, min_history // 2)
        if int(active.sum()) <= k:
            continue
        sub = hist[:, active]
        sub_valid = valid[:, active]
        means = np.sum(np.where(sub_valid, sub, 0.0), axis=0) / np.maximum(counts[active], 1)
        centered = np.where(sub_valid, sub - means[None, :], 0.0)
        # SVD económica de la ventana (W x N_act) -> Vh[:k].T son las cargas factoriales
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        avail = min(k, vh.shape[0])
        # Fijar signo determinista (mayor componente absoluta positiva) entre sesiones
        loadings = vh[:avail].T.copy()
        for j in range(avail):
            pivot = int(np.argmax(np.abs(loadings[:, j])))
            if loadings[pivot, j] < 0.0:
                loadings[:, j] = -loadings[:, j]
        out[t, active, :avail] = loadings
    return out


@dataclass
class CausalFactorNeutralizer:
    """Neutralizador ajustado exclusivamente en calibración (elige `gamma` y `ridge_lambda`)."""

    gamma: float
    ridge_lambda: float
    include_intercept: bool
    rank_input: bool
    calibration_objective: float
    grid_results: list[dict[str, Any]]

    @classmethod
    def fit_calibration(
        cls,
        score: np.ndarray,
        target: np.ndarray,
        exposures: np.ndarray,
        bounds: list[tuple[int, int]],
        *,
        weights: np.ndarray | None = None,
        gammas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
        lambdas: tuple[float, ...] = (0.1, 1.0, 10.0),
        include_intercept: bool = True,
        rank_input: bool = True,
        objective: str = "icir",
        min_assets: int = 30,
        keep_mask: np.ndarray | None = None,
    ) -> CausalFactorNeutralizer:
        """Elige `(gamma, ridge_lambda)` en calibración maximizando ICIR, P(IC>0) o media."""
        _require(
            objective in ("icir", "mean", "positive_rate", "icir_with_floor"),
            f"Objetivo desconocido: {objective!r}",
        )
        y = np.asarray(target, dtype=np.float64)
        # Precalcular rangos del objetivo por sesión válida
        sess_info: list[tuple[int, int, np.ndarray | None, np.ndarray, float]] = []
        for lo, hi in bounds:
            if keep_mask is None:
                if hi - lo < min_assets:
                    continue
                y_seg = y[lo:hi]
                sub_idx = None
            else:
                local = np.flatnonzero(keep_mask[lo:hi])
                if len(local) < min_assets:
                    continue
                sub_idx = local
                y_seg = y[lo + local]
            if float(np.ptp(y_seg)) == 0.0:
                continue
            ly, n_y = oa._prep_target_ranks(y_seg)
            if n_y > 0.0:
                sess_info.append((lo, hi, sub_idx, ly, n_y))

        best_score = -1e9
        best_cfg = (0.0, lambdas[0])
        grid_results: list[dict[str, Any]] = []

        # Evaluar gamma=0 una sola vez (independiente de lambda)
        seen_zero = False
        for lam in lambdas:
            # Proyectar a gamma=1 una sola vez por lambda y mezclar para todos los gamma!
            s_base = neutralize_sessions(
                score,
                exposures,
                bounds,
                weights=weights,
                ridge_lambda=lam,
                gamma=0.0,
                include_intercept=include_intercept,
                rank_input=rank_input,
            )
            s_full = neutralize_sessions(
                score,
                exposures,
                bounds,
                weights=weights,
                ridge_lambda=lam,
                gamma=1.0,
                include_intercept=include_intercept,
                rank_input=rank_input,
            )
            for gamma in gammas:
                if gamma == 0.0:
                    if seen_zero:
                        continue
                    seen_zero = True
                    cand = s_base
                elif gamma == 1.0:
                    cand = s_full
                else:
                    cand = np.empty_like(s_base)
                    for lo, hi in bounds:
                        sb = s_base[lo:hi]
                        sf = s_full[lo:hi]
                        std_b, std_f = float(np.std(sb)), float(np.std(sf))
                        if std_b > 1e-12 and std_f > 1e-12:
                            cand[lo:hi] = (1.0 - gamma) * (sb / std_b) + gamma * (sf / std_f)
                        else:
                            cand[lo:hi] = (1.0 - gamma) * sb + gamma * sf

                ics = []
                for lo, hi, sub_idx, ly, n_y in sess_info:
                    seg = cand[lo:hi] if sub_idx is None else cand[lo + sub_idx]
                    v = oa._rank_ic_prepped(seg, ly, n_y)
                    if math.isfinite(v):
                        ics.append(v)
                arr = np.asarray(ics, dtype=np.float64)
                mean_ic = float(arr.mean()) if arr.size else -2.0
                std_ic = float(arr.std(ddof=1)) if arr.size > 1 else 1.0
                icir = mean_ic / std_ic if std_ic > 1e-12 else -2.0
                pos_rate = float(np.mean(arr > 0.0)) if arr.size else 0.0

                if objective == "icir":
                    obj_val = icir
                elif objective == "mean":
                    obj_val = mean_ic
                elif objective == "positive_rate":
                    obj_val = pos_rate
                else:
                    # ICIR solo si retiene al menos 85% del IC medio base
                    obj_val = icir if mean_ic > 0.0 else -1e6

                grid_results.append(
                    dict(
                        gamma=float(gamma),
                        ridge_lambda=float(lam),
                        mean_ic=mean_ic,
                        std_ic=std_ic,
                        icir=icir,
                        positive_rate=pos_rate,
                        objective=obj_val,
                    )
                )
                if obj_val > best_score + 1e-12:
                    best_score = obj_val
                    best_cfg = (float(gamma), float(lam))

        return cls(
            gamma=best_cfg[0],
            ridge_lambda=best_cfg[1],
            include_intercept=include_intercept,
            rank_input=rank_input,
            calibration_objective=float(best_score),
            grid_results=grid_results,
        )

    def transform(
        self,
        score: np.ndarray,
        exposures: np.ndarray,
        bounds: list[tuple[int, int]],
        *,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        return neutralize_sessions(
            score,
            exposures,
            bounds,
            weights=weights,
            ridge_lambda=self.ridge_lambda,
            gamma=self.gamma,
            include_intercept=self.include_intercept,
            rank_input=self.rank_input,
        )


def variance_decomposition(
    raw_ics: np.ndarray,
    neut_ics: np.ndarray,
    *,
    mean_assets: float,
) -> dict[str, Any]:
    """Descompone la varianza diaria del IC en componente idiosincrática y de riesgo factorial."""
    raw = np.asarray(raw_ics, dtype=np.float64)
    neut = np.asarray(neut_ics, dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    neut = neut[np.isfinite(neut)]
    idio_floor = 1.0 / math.sqrt(max(float(mean_assets), 1.0))
    raw_mean, raw_std = float(raw.mean()), float(raw.std(ddof=1))
    neut_mean, neut_std = float(neut.mean()), float(neut.std(ddof=1))
    raw_var, neut_var = raw_std * raw_std, neut_std * neut_std
    idio_var = idio_floor * idio_floor
    diff = neut - raw if len(neut) == len(raw) else None
    t_hac, se_hac, _ = (
        oa.newey_west_hac(diff) if diff is not None and len(diff) >= 2 else (None, None, 0)
    )
    return dict(
        sessions=int(len(neut)),
        idio_floor_std=idio_floor,
        raw_mean=raw_mean,
        raw_std=raw_std,
        raw_icir=raw_mean / raw_std if raw_std > 0 else None,
        raw_positive_rate=float(np.mean(raw > 0.0)),
        raw_factor_var_share=max(0.0, 1.0 - idio_var / raw_var) if raw_var > 0 else 0.0,
        neut_mean=neut_mean,
        neut_std=neut_std,
        neut_icir=neut_mean / neut_std if neut_std > 0 else None,
        neut_positive_rate=float(np.mean(neut > 0.0)),
        neut_factor_var_share=max(0.0, 1.0 - idio_var / neut_var) if neut_var > 0 else 0.0,
        std_reduction_pct=100.0 * (1.0 - neut_std / raw_std) if raw_std > 0 else 0.0,
        delta_mean=float(diff.mean()) if diff is not None else None,
        delta_t_hac=t_hac,
        delta_se_hac=se_hac,
    )
