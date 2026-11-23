"""Capa SOTA de post-entrenamiento: memoria multiescala por activo y combinador OAS-ICIR.

Combina mejoras causales probadas sobre las emisiones de modelos existentes sin reentrenar:
1. `multiscale_asset_residual_features` + `MultiScaleAssetMemory`:
   - Memoria causal multiescala de residuos por activo con semividas simultáneas (p. ej. 5, 21 y 63
     sesiones) y contracción bayesiana `mu_a(h) = sum_w_res / (sum_w + k_h)`.
   - Soporta tanto residuo de etiqueta (`y - p`) como residuo operable (`(y + alpha) - (k*p + alpha)`).
   - Sustituye la búsqueda discreta en 80 configuraciones por un ajuste continuo regularizado
     (no negativo con contracción Ridge) en calibración sobre 3 pesos de horizonte.
2. `oas_shrink_covariance` + `shrinkage_icir_weights`:
   - Contracción analítica Oracle Approximating Shrinkage (Chen et al., IEEE TSP 2010) sobre la
     covarianza transversal de modelos `Sigma_t`, evitando colinealidad al combinar múltiples
     semillas de Titans con árboles (LightGBM) y órganos de memoria ortogonales.
   - Pesos causales por sesión basados en ICIR contraído (Empirical-Bayes positive-part),
     anulando automáticamente placebos y repartiendo óptimamente entre familias ortogonales.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.evaluation import online_aggregator as oa
from mars_titan.memory import asset_residual_memory as arm

DEFAULT_HALF_LIVES = (5.0, 21.0, 63.0)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def multiscale_asset_residual_features(
    asset_ids: np.ndarray,
    prediction_at: np.ndarray,
    matures_at: np.ndarray,
    residuals: np.ndarray,
    *,
    half_lives: tuple[float, ...] = DEFAULT_HALF_LIVES,
    shrinkage_k: float = 10.0,
) -> np.ndarray:
    """Extrae `[N, H]` medias exponenciales contraídas de residuos maduros por activo.

    Contrato causal estricto:
    - Rechaza cualquier `prediction_at` o `matures_at >= FINAL_TEST_START_US`.
    - En la sesión `t`, el estado de cada activo solo integra observaciones con
      `matures_at <= t` (jamás usa residuos contemporáneos ni futuros).
    - El decaimiento se aplica por salto de sesiones observadas del activo (o por días
      transcurridos / 1 día), 100% vectorizado por sesión.
    """
    assets = np.asarray(asset_ids)
    times = np.asarray(prediction_at, dtype=np.int64)
    mats = np.asarray(matures_at, dtype=np.int64)
    res = np.asarray(residuals, dtype=np.float64)
    n = len(res)
    _require(
        len(assets) == n and len(times) == n and len(mats) == n,
        "Longitudes incompatibles en multiscale_asset_residual_features",
    )
    if n == 0:
        return np.empty((0, len(half_lives)), dtype=np.float64)
    _require(int(times.max()) < FINAL_TEST_START_US, "Violación del sello de 2024 en prediction_at")
    _require(int(mats.max()) < FINAL_TEST_START_US, "Violación del sello de 2024 en matures_at")
    _require(bool(np.all(mats > times)), "Todas las etiquetas deben madurar tras prediction_at")
    _require(
        len(half_lives) >= 1 and all(h > 0.0 and math.isfinite(h) for h in half_lives),
        "half_lives debe contener reales positivos finitos",
    )
    _require(shrinkage_k >= 0.0 and math.isfinite(shrinkage_k), "shrinkage_k debe ser >= 0")

    # Mapear asset_ids a enteros densos [0..A-1]
    _, inv_assets = np.unique(assets, return_inverse=True)
    n_assets = int(inv_assets.max()) + 1
    n_hl = len(half_lives)
    decays = np.asarray([math.pow(0.5, 1.0 / float(h)) for h in half_lives], dtype=np.float64)

    # Ordenar sesiones cronológicamente por prediction_at (manteniendo estabilidad)
    order = np.argsort(times, kind="mergesort")
    t_ord = times[order]
    change = np.r_[True, t_ord[1:] != t_ord[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], n]

    # Estado por activo: [n_assets, n_hl]
    sum_w = np.zeros((n_assets, n_hl), dtype=np.float64)
    sum_wr = np.zeros((n_assets, n_hl), dtype=np.float64)
    out = np.zeros((n, n_hl), dtype=np.float64)

    # Cola de cohortes pendientes por madurar: lista de (mat_us, a_idx, r_vals)
    pending: list[tuple[int, np.ndarray, np.ndarray]] = []
    k_val = float(shrinkage_k)

    for s, e in zip(starts, ends, strict=True):
        idx = order[s:e]
        now_us = int(t_ord[s])

        # Madurar todas las cohortes pendientes con mat_us <= now_us en orden de madurez
        if pending:
            ready = [item for item in pending if item[0] <= now_us]
            if ready:
                pending = [item for item in pending if item[0] > now_us]
                ready.sort(key=lambda item: item[0])
                for _, a_m, r_m in ready:
                    finite = np.isfinite(r_m)
                    if not np.any(finite):
                        continue
                    a_f = a_m[finite]
                    r_f = r_m[finite]
                    # Decaer solo activos actualizados en esta cohorte madura
                    sum_w[a_f] = sum_w[a_f] * decays[None, :] + 1.0
                    sum_wr[a_f] = sum_wr[a_f] * decays[None, :] + r_f[:, None]

        # Emitir memoria causal antes de encolar la cohorte actual
        a_cur = inv_assets[idx]
        denom = sum_w[a_cur] + k_val
        out[idx] = np.where(denom > 0.0, sum_wr[a_cur] / denom, 0.0)

        # Encolar esta cohorte para cuando madure
        mat_us = int(mats[idx].max())
        pending.append((mat_us, a_cur, res[idx]))

    return out


@dataclass
class MultiScaleAssetMemory:
    """Combinador lineal no negativo y regularizado de memorias de residuo multiescala."""

    weights: np.ndarray  # [H] pesos no negativos para cada horizonte
    scale: float
    calibration_ic: float
    base_calibration_ic: float

    @classmethod
    def fit_calibration(
        cls,
        parent_pred: np.ndarray,
        target: np.ndarray,
        features: np.ndarray,
        bounds: list[tuple[int, int]],
        *,
        ridge_lambda: float = 1e-2,
        scale_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0),
        min_assets: int = 30,
    ) -> MultiScaleAssetMemory:
        """Ajusta los pesos de horizonte y la escala global EXCLUSIVAMENTE en calibración."""
        p = np.asarray(parent_pred, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        f = np.asarray(features, dtype=np.float64)
        _require(f.ndim == 2 and len(p) == len(y) == len(f), "Dimensiones incompatibles")
        h = f.shape[1]

        # 1) Estimar dirección relativa de horizontes por Ridge no negativo (invariante a escala)
        res = y - p
        valid = np.isfinite(res) & np.all(np.isfinite(f), axis=1)
        if int(valid.sum()) > h:
            fv = f[valid]
            rv = res[valid]
            gram = (fv.T @ fv) / len(fv)
            scale_g = max(float(np.trace(gram)) / h, 1e-12)
            gram_reg = gram + (ridge_lambda * scale_g) * np.eye(h, dtype=np.float64)
            rhs = (fv.T @ rv) / len(fv)
            w_raw = np.linalg.solve(gram_reg, rhs)
            w_pos = np.maximum(w_raw, 0.0)
        else:
            w_pos = np.ones(h, dtype=np.float64) / h
        w_sum = float(w_pos.sum())
        if w_sum <= 1e-12:
            w_pos = np.ones(h, dtype=np.float64) / h
        else:
            # Normalizar pesos de horizonte al símplex para que `scale` esté en unidades 1:1 del residuo
            w_pos = w_pos / w_sum

        delta = f @ w_pos

        # 2) Ajustar escala global en calibración por IC medio de rangos (con 0.0 como respaldo seguro)
        sess_info: list[tuple[int, int, np.ndarray, float]] = []
        for lo, hi in bounds:
            if hi - lo < min_assets:
                continue
            y_seg = y[lo:hi]
            if float(np.ptp(y_seg)) == 0.0:
                continue
            ly, n_y = oa._prep_target_ranks(y_seg)
            if n_y > 0.0:
                sess_info.append((lo, hi, ly, n_y))

        def _eval_scale(sc: float) -> float:
            pred = p + sc * delta
            vals = [oa._rank_ic_prepped(pred[lo:hi], ly, n_y) for lo, hi, ly, n_y in sess_info]
            vals = [v for v in vals if math.isfinite(v)]
            return float(np.mean(vals)) if vals else -2.0

        base_ic = _eval_scale(0.0)
        best_sc = 0.0
        best_ic = base_ic
        for sc in scale_grid:
            val = _eval_scale(float(sc))
            if val > best_ic + 1e-12:
                best_ic = val
                best_sc = float(sc)

        return cls(
            weights=w_pos,
            scale=best_sc,
            calibration_ic=best_ic,
            base_calibration_ic=base_ic,
        )

    def correction(self, features: np.ndarray) -> np.ndarray:
        return self.scale * (np.asarray(features, dtype=np.float64) @ self.weights)

    def predict(self, parent_pred: np.ndarray, features: np.ndarray) -> np.ndarray:
        return np.asarray(parent_pred, dtype=np.float64) + self.correction(features)


def multiscale_rank_residual_features(
    asset_ids: np.ndarray,
    prediction_at: np.ndarray,
    matures_at: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    half_lives: tuple[float, ...] = DEFAULT_HALF_LIVES,
    shrinkage_k: float = 5.0,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Memoria multiescala causal de residuos gaussianos ortogonalizados por sesión (robusta a outliers).

    En cada sesión `t` (con filas válidas):
    1. Emite la z-score de la predicción `z_t` y las lecturas tipificadas `u_t[:, h]` de cada
       horizonte `h in half_lives`, usando SOLO cohortes maduras (`matures_at <= t`).
    2. Cuando madura la cohorte `t`, guarda el residuo de rango gaussiano ortogonalizado
       `e_t = g_t - (z_t · g_t / n) * z_t` (inmune a extremos de volatilidad en retornos crudos)
       y actualiza el estado EWMA con contracción bayesiana `m_a(h) = S_a(h) / (W_a(h) + k)`.
    Devuelve `(z_parent [N], u_features [N, H])`.
    """
    assets = np.asarray(asset_ids)
    times = np.asarray(prediction_at, dtype=np.int64)
    mats = np.asarray(matures_at, dtype=np.int64)
    pred = np.asarray(prediction, dtype=np.float64)
    targ = np.asarray(target, dtype=np.float64)
    n = len(targ)
    _require(
        len(assets) == n and len(times) == n and len(mats) == n and len(pred) == n,
        "Longitudes incompatibles en multiscale_rank_residual_features",
    )
    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty((0, len(half_lives)), dtype=np.float64)
    _require(int(times.max()) < FINAL_TEST_START_US, "Violación del sello de 2024 en prediction_at")
    _require(int(mats.max()) < FINAL_TEST_START_US, "Violación del sello de 2024 en matures_at")
    _require(bool(np.all(mats > times)), "Todas las etiquetas deben madurar tras prediction_at")

    valid = (
        np.isfinite(pred) & np.isfinite(targ)
        if valid_mask is None
        else (np.asarray(valid_mask, dtype=bool) & np.isfinite(pred) & np.isfinite(targ))
    )
    _, inv_assets = np.unique(assets, return_inverse=True)
    n_assets = int(inv_assets.max()) + 1
    n_hl = len(half_lives)
    decays = np.asarray([math.pow(0.5, 1.0 / float(h)) for h in half_lives], dtype=np.float64)

    order = np.argsort(times, kind="mergesort")
    t_ord = times[order]
    change = np.r_[True, t_ord[1:] != t_ord[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], n]

    sum_w = np.zeros((n_assets, n_hl), dtype=np.float64)
    sum_we = np.zeros((n_assets, n_hl), dtype=np.float64)
    z_out = np.zeros(n, dtype=np.float64)
    u_out = np.zeros((n, n_hl), dtype=np.float64)
    pending: list[tuple[int, np.ndarray, np.ndarray]] = []
    k_val = float(shrinkage_k)

    for s, e in zip(starts, ends, strict=True):
        idx = order[s:e]
        now_us = int(t_ord[s])

        if pending:
            ready = [item for item in pending if item[0] <= now_us]
            if ready:
                pending = [item for item in pending if item[0] > now_us]
                ready.sort(key=lambda item: item[0])
                for _, a_m, e_m in ready:
                    sum_w *= decays[None, :]
                    sum_we *= decays[None, :]
                    sum_w[a_m] += 1.0
                    sum_we[a_m] += e_m[:, None]

        v_loc = valid[idx]
        idx_v = idx[v_loc]
        if len(idx_v) < 2:
            continue
        z_seg = arm.zscore(pred[idx_v])
        z_out[idx_v] = z_seg

        a_cur = inv_assets[idx_v]
        denom = sum_w[a_cur] + k_val
        raw_m = np.where(sum_w[a_cur] > 0.0, sum_we[a_cur] / denom, 0.0)
        for h_i in range(n_hl):
            u_out[idx_v, h_i] = arm.zscore(raw_m[:, h_i])

        # Residuo gaussiano ortogonalizado para cuando madure esta sesión
        g_seg = arm.gaussian_ranks(targ[idx_v])
        e_seg = g_seg - (float(z_seg @ g_seg) / len(z_seg)) * z_seg
        mat_us = int(mats[idx_v].max())
        pending.append((mat_us, a_cur, e_seg))

    return z_out, u_out


@dataclass
class MultiScaleRankMemory:
    """Fusión multiescala regularizada sobre `z_parent + kappa * (u @ w)` ajustada en calibración."""

    weights: np.ndarray
    kappa: float
    calibration_ic: float
    base_calibration_ic: float

    @classmethod
    def fit_calibration(
        cls,
        z_parent: np.ndarray,
        u_features: np.ndarray,
        target: np.ndarray,
        bounds: list[tuple[int, int]],
        *,
        kappa_grid: tuple[float, ...] = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30),
        min_assets: int = 30,
    ) -> MultiScaleRankMemory:
        """Elige los pesos de horizonte (por IC medio positivo de cada horizonte en calibración) y `kappa`."""
        z = np.asarray(z_parent, dtype=np.float64)
        u = np.asarray(u_features, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        _require(u.ndim == 2 and len(z) == len(u) == len(y), "Dimensiones incompatibles")
        h = u.shape[1]

        sess_info: list[tuple[int, int, np.ndarray, float]] = []
        for lo, hi in bounds:
            if hi - lo < min_assets:
                continue
            y_seg = y[lo:hi]
            if float(np.ptp(y_seg)) == 0.0:
                continue
            ly, n_y = oa._prep_target_ranks(y_seg)
            if n_y > 0.0:
                sess_info.append((lo, hi, ly, n_y))

        # Pesos de horizonte por ICIR positivo en calibración (regularización natural sin sobreajuste)
        h_scores = np.zeros(h, dtype=np.float64)
        for h_i in range(h):
            ics_h = [
                oa._rank_ic_prepped(u[lo:hi, h_i], ly, n_y)
                for lo, hi, ly, n_y in sess_info
            ]
            ics_h = [v for v in ics_h if math.isfinite(v)]
            if len(ics_h) >= 2:
                arr_h = np.asarray(ics_h, dtype=np.float64)
                std_h = float(arr_h.std(ddof=1))
                h_scores[h_i] = max(float(arr_h.mean()) / std_h, 0.0) if std_h > 1e-12 else 0.0
        w_sum = float(h_scores.sum())
        w_pos = (h_scores / w_sum) if w_sum > 1e-12 else np.full(h, 1.0 / h, dtype=np.float64)
        u_blend = u @ w_pos

        def _eval_kappa(kap: float) -> float:
            pred = z + kap * u_blend
            vals = [oa._rank_ic_prepped(pred[lo:hi], ly, n_y) for lo, hi, ly, n_y in sess_info]
            vals = [v for v in vals if math.isfinite(v)]
            return float(np.mean(vals)) if vals else -2.0

        base_ic = _eval_kappa(0.0)
        best_kap = 0.0
        best_ic = base_ic
        for kap in kappa_grid:
            val = _eval_kappa(float(kap))
            if val > best_ic + 1e-12:
                best_ic = val
                best_kap = float(kap)
        return cls(weights=w_pos, kappa=best_kap, calibration_ic=best_ic, base_calibration_ic=base_ic)

    def signal(self, u_features: np.ndarray) -> np.ndarray:
        return self.kappa * (np.asarray(u_features, dtype=np.float64) @ self.weights)

    def predict(self, z_parent: np.ndarray, u_features: np.ndarray) -> np.ndarray:
        return np.asarray(z_parent, dtype=np.float64) + self.signal(u_features)


def oas_shrink_covariance(
    sample_cov: np.ndarray,
    *,
    n_obs: float,
) -> tuple[np.ndarray, float]:
    """Contracción analítica Oracle Approximating Shrinkage (Chen et al., IEEE TSP 2010).

    Contrae `S` hacia el objetivo esférico `mu * I` (`mu = tr(S) / p`):
        `Sigma_oas = (1 - rho) * S + rho * mu * I`
    Garantiza simetría y buen condicionamiento incluso con `p` cercano a `n_obs` o modelos colineales.
    """
    s = np.asarray(sample_cov, dtype=np.float64)
    s = 0.5 * (s + s.T)
    p = s.shape[0]
    if p <= 1:
        return s.copy(), 0.0
    tr_s = float(np.trace(s))
    mu = tr_s / p
    tr_s2 = float(np.sum(s * s))
    n = max(float(n_obs), 1.0)
    # Fórmula exacta de Chen et al. (2010), eq. (23)
    num = (1.0 - 2.0 / p) * tr_s2 + tr_s * tr_s
    den = (n + 1.0 - 2.0 / p) * (tr_s2 - (tr_s * tr_s) / p)
    if den <= 1e-15 or not math.isfinite(den):
        rho = 1.0
    else:
        rho = float(np.clip(num / den, 0.0, 1.0))
    eye = np.eye(p, dtype=np.float64)
    shrunk = (1.0 - rho) * s + rho * mu * eye
    # Piso mínimo en la diagonal para positividad estricta
    min_eig = float(np.linalg.eigvalsh(shrunk)[0])
    if min_eig <= 1e-12:
        shrunk = shrunk + (1e-10 - min_eig) * eye
    return shrunk, rho


def shrinkage_icir_weights(
    session_ics: np.ndarray,
    session_covs: np.ndarray,
    *,
    lam: float = 0.25,
    half_life: float | None = 63.0,
    session_times: np.ndarray | None = None,
    session_matures: np.ndarray | None = None,
    maturity_lag: int = 1,
    scored: np.ndarray | None = None,
    se_shrinkage: float = 1.0,
    cov_shrinkage: str = "oas",
    use_current_cov: bool = True,
    n_obs_cov: float = 200.0,
) -> np.ndarray:
    """Pesos causales por sesión combinando covarianza contraída OAS + ICIR contraído.

    En cada sesión `t` usa exclusivamente sesiones maduras `M(t)`:
    1. Estima `b_t` (media EWMA del IC) y su error estándar `SE_t` con `n_eff`.
    2. Aplica contracción Empirical-Bayes de parte positiva: `b_adj = max(b_t - se_shrinkage * SE_t, 0)`
       en el conjunto activo `{m : b_adj[m] > 0}`, anulando a cero expertos sin señal (placebos).
    3. Estima la matriz de covarianza de predicciones `Sigma_t` (actual o histórica madura) y la
       contrae con OAS (`cov_shrinkage="oas"`) + regularización `lam * I`.
    4. Resuelve el problema cuadrático con restricción de no negatividad sobre el conjunto activo
       y normaliza en el símplex.
    """
    ics = np.asarray(session_ics, dtype=np.float64)
    covs = np.asarray(session_covs, dtype=np.float64)
    n_sessions, n_models = ics.shape
    _require(covs.shape == (n_sessions, n_models, n_models), "Forma de covarianzas incompatible")
    _require(cov_shrinkage in ("oas", "none"), f"cov_shrinkage inválido: {cov_shrinkage!r}")
    _require(type(maturity_lag) is int and maturity_lag >= 1, "maturity_lag debe ser >= 1")

    times = (
        np.arange(n_sessions, dtype=np.int64)
        if session_times is None
        else np.asarray(session_times, dtype=np.int64)
    )
    sc = np.ones(n_sessions, dtype=bool) if scored is None else np.asarray(scored, dtype=bool)
    mat_arr = None
    if session_matures is not None:
        mat_arr = np.asarray(session_matures, dtype=np.int64)
        _require(
            mat_arr.shape == times.shape and bool(np.all(mat_arr > times)),
            "Toda etiqueta debe madurar estrictamente después de su instante de decisión",
        )

    ics_clean = np.nan_to_num(ics, nan=0.0)
    covs_clean = np.nan_to_num(covs, nan=0.0)
    uniform = np.full(n_models, 1.0 / n_models, dtype=np.float64)
    out = np.empty((n_sessions, n_models), dtype=np.float64)

    for t in range(n_sessions):
        if mat_arr is not None:
            matured = np.flatnonzero((mat_arr[:t] <= times[t]) & sc[:t])
        else:
            cutoff = t - maturity_lag
            matured = np.flatnonzero(sc[: cutoff + 1]) if cutoff >= 0 else np.empty(0, dtype=np.int64)
        if matured.size == 0:
            out[t] = uniform
            continue

        k = matured.size
        if half_life is not None:
            ages = np.arange(k - 1, -1, -1, dtype=np.float64)
            decay = np.power(2.0, -ages / float(half_life))
        else:
            decay = np.ones(k, dtype=np.float64)
        sw = float(decay.sum())
        norm_w = decay / sw
        n_eff = float((sw * sw) / np.sum(decay * decay))

        hist_ic = ics_clean[matured]
        b_mean = (norm_w[:, None] * hist_ic).sum(axis=0)
        if k >= 2:
            diff = hist_ic - b_mean[None, :]
            var = (norm_w[:, None] * (diff * diff)).sum(axis=0)
            se = np.sqrt(np.maximum(var, 0.0) / max(n_eff, 1.0))
        else:
            se = np.zeros(n_models, dtype=np.float64)

        b_adj = np.maximum(b_mean - se_shrinkage * se, 0.0)
        active = np.flatnonzero(b_adj > 0.0)
        if active.size == 0:
            out[t] = uniform
            continue

        if use_current_cov:
            cov_raw = covs_clean[t]
        else:
            cov_raw = np.tensordot(norm_w, covs_clean[matured], axes=(0, 0))

        sub_cov = cov_raw[np.ix_(active, active)]
        if cov_shrinkage == "oas":
            sub_cov, _ = oas_shrink_covariance(sub_cov, n_obs=n_obs_cov)

        sub_a = sub_cov + lam * np.eye(active.size, dtype=np.float64)
        sub_rhs = b_adj[active]
        try:
            raw_sub = np.linalg.solve(sub_a, sub_rhs)
        except np.linalg.LinAlgError:
            raw_sub = np.linalg.lstsq(sub_a, sub_rhs, rcond=None)[0]
        raw_sub = np.maximum(raw_sub, 0.0)
        total = float(raw_sub.sum())
        if total <= 0.0 or not math.isfinite(total):
            out[t] = uniform
        else:
            w_full = np.zeros(n_models, dtype=np.float64)
            w_full[active] = raw_sub / total
            out[t] = w_full
    return out
