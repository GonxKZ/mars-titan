"""Pruebas del pipeline SOTA de post-entrenamiento: memoria multiescala, OAS-ICIR y orquestación."""

from __future__ import annotations

import numpy as np
import pytest

from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.evaluation import enhanced_posttraining as ep
from mars_titan.evaluation import online_aggregator as oa

DAY = 86_400_000_000
START = 1_357_084_800_000_000


def _persistent_residual_panel(rng: np.random.Generator, *, sessions: int = 120, assets: int = 80):
    """Panel donde cada activo tiene un sesgo persistente rápido + lento en su residuo."""
    n = sessions * assets
    asset_ids = np.tile(np.arange(assets, dtype=np.int32), sessions)
    times = np.repeat(START + np.arange(sessions, dtype=np.int64) * DAY, assets)
    matures = times + DAY
    bounds = [(s * assets, (s + 1) * assets) for s in range(sessions)]

    slow_bias = rng.normal(0.0, 0.012, size=assets)
    fast_state = np.zeros(assets, dtype=np.float64)
    parent = np.empty(n, dtype=np.float64)
    target = np.empty(n, dtype=np.float64)
    alpha = rng.normal(0.0, 0.002, size=n)

    for s in range(sessions):
        lo, hi = bounds[s]
        fast_state = 0.80 * fast_state + 0.008 * rng.standard_normal(assets)
        base_sig = rng.standard_normal(assets) * 0.01
        parent[lo:hi] = base_sig
        target[lo:hi] = base_sig + slow_bias + fast_state + 0.012 * rng.standard_normal(assets)
    return dict(
        asset_ids=asset_ids,
        times=times,
        matures=matures,
        bounds=bounds,
        parent=parent,
        target=target,
        alpha=alpha,
    )


def test_multiscale_asset_memory_improves_ic_strictly_causally_and_respects_seal():
    """La memoria multiescala mejora el IC en evaluación, sin fuga y respetando sello 2024."""
    rng = np.random.default_rng(42)
    panel = _persistent_residual_panel(rng, sessions=120, assets=80)
    cal_end = 60 * 80

    # Extraer señales multiescala estrictamente causales
    features = ep.multiscale_asset_residual_features(
        panel["asset_ids"],
        panel["times"],
        panel["matures"],
        panel["target"] - panel["parent"],
        half_lives=(5.0, 21.0, 63.0),
        shrinkage_k=8.0,
    )
    assert features.shape == (len(panel["target"]), 3)
    assert np.all(features[:80] == 0.0)  # Primera sesión sin historia madura

    # Causalidad estricta: mutar targets desde sesión 40 no cambia features <= sesión 40
    res_mut = (panel["target"] - panel["parent"]).copy()
    res_mut[40 * 80 :] = 100.0
    features_mut = ep.multiscale_asset_residual_features(
        panel["asset_ids"],
        panel["times"],
        panel["matures"],
        res_mut,
        half_lives=(5.0, 21.0, 63.0),
        shrinkage_k=8.0,
    )
    np.testing.assert_array_equal(features[: 41 * 80], features_mut[: 41 * 80])
    assert not np.array_equal(features[41 * 80 :], features_mut[41 * 80 :])

    # Ajustar pesos de fusión SOLO en calibración (sesiones 0..59) y evaluar en 60..119
    mem = ep.MultiScaleAssetMemory.fit_calibration(
        panel["parent"][:cal_end],
        panel["target"][:cal_end],
        features[:cal_end],
        panel["bounds"][:60],
    )
    ev_bounds = [(lo - cal_end, hi - cal_end) for lo, hi in panel["bounds"][60:]]
    pred_ev = mem.predict(panel["parent"][cal_end:], features[cal_end:])
    ic_base = np.mean(
        [
            oa.rank_ic_vector(
                panel["parent"][cal_end + lo : cal_end + hi],
                panel["target"][cal_end + lo : cal_end + hi],
            )
            for lo, hi in ev_bounds
        ]
    )
    ic_mem = np.mean(
        [
            oa.rank_ic_vector(pred_ev[lo:hi], panel["target"][cal_end + lo : cal_end + hi])
            for lo, hi in ev_bounds
        ]
    )
    assert ic_mem > ic_base + 0.08

    # Sello 2024
    bad_t = panel["times"].copy()
    bad_t[-1] = FINAL_TEST_START_US
    with pytest.raises(ValueError, match="2024|test final|sello"):
        ep.multiscale_asset_residual_features(
            panel["asset_ids"], bad_t, bad_t + DAY, panel["target"] - panel["parent"]
        )


def test_shrinkage_icir_weights_handles_correlated_seeds_and_zeros_placebo():
    """OAS-ICIR reparte peso entre semillas correladas, rescata señal ortogonal y anula placebo."""
    rng = np.random.default_rng(9)
    n_sess, n_models = 150, 5
    # Modelos 0, 1, 2: 3 semillas muy correladas (rho ~ 0.92), IC ~ 0.06
    # Modelo 3: familia ortogonal (árboles), IC ~ 0.055
    # Modelo 4: placebo con IC ~ 0.0
    ics = np.column_stack(
        [
            rng.normal(0.060, 0.04, n_sess),
            rng.normal(0.060, 0.04, n_sess),
            rng.normal(0.058, 0.04, n_sess),
            rng.normal(0.055, 0.04, n_sess),
            rng.normal(0.000, 0.05, n_sess),
        ]
    )
    base_cov = np.array(
        [
            [1.00, 0.92, 0.92, 0.35, 0.00],
            [0.92, 1.00, 0.92, 0.35, 0.00],
            [0.92, 0.92, 1.00, 0.35, 0.00],
            [0.35, 0.35, 0.35, 1.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 1.00],
        ],
        dtype=np.float64,
    )
    covs = np.broadcast_to(base_cov, (n_sess, n_models, n_models)).copy()
    w = ep.shrinkage_icir_weights(ics, covs, half_life=63.0, se_shrinkage=1.0, cov_shrinkage="oas")
    assert w.shape == (n_sess, n_models)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(w >= 0.0)
    # En la segunda mitad, el placebo recibe peso casi cero y el árbol > cualquier semilla sola
    w_late = w[80:].mean(axis=0)
    assert w_late[4] < 0.03, f"Placebo no anulado: {w_late}"
    assert w_late[3] > w_late[0], (
        f"Familia ortogonal debe pesar más que una semilla repetida: {w_late}"
    )

    # Causalidad estricta: mutar IC de sesiones >= 100 no cambia pesos <= 100 (lag=1)
    ics_mut = ics.copy()
    ics_mut[100:] = -1.0
    w_mut = ep.shrinkage_icir_weights(
        ics_mut, covs, half_life=63.0, se_shrinkage=1.0, maturity_lag=1
    )
    np.testing.assert_array_equal(w[:101], w_mut[:101])


def test_oas_covariance_shrinkage_is_psd_and_well_conditioned():
    """La contracción OAS produce una matriz simétrica definida positiva bien condicionada."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((12, 8))  # p cercano a n -> muestra mal condicionada
    s_emp = (x.T @ x) / len(x)
    s_oas, alpha = ep.oas_shrink_covariance(s_emp, n_obs=len(x))
    assert 0.0 <= alpha <= 1.0
    np.testing.assert_allclose(s_oas, s_oas.T, atol=1e-14)
    eigvals = np.linalg.eigvalsh(s_oas)
    assert eigvals.min() > 0.0
    assert np.linalg.cond(s_oas) <= np.linalg.cond(s_emp) + 1e-9


def test_multiscale_rank_memory_is_strictly_causal_and_robust_to_outliers():
    """La memoria multiescala de rangos gaussianos resiste outliers extremos y es causal."""
    rng = np.random.default_rng(17)
    panel = _persistent_residual_panel(rng, sessions=100, assets=80)
    # Inyectar outliers extremos en target que arruinarían residuos crudos sin rangos
    target_noisy = panel["target"].copy()
    target_noisy[::97] *= 50.0
    cal_end = 50 * 80

    z1, u1 = ep.multiscale_rank_residual_features(
        panel["asset_ids"], panel["times"], panel["matures"], panel["parent"], target_noisy
    )
    t_mut = target_noisy.copy()
    t_mut[30 * 80 :] = -1000.0
    z2, u2 = ep.multiscale_rank_residual_features(
        panel["asset_ids"], panel["times"], panel["matures"], panel["parent"], t_mut
    )
    np.testing.assert_array_equal(u1[: 31 * 80], u2[: 31 * 80])
    assert not np.array_equal(u1[31 * 80 :], u2[31 * 80 :])

    mem = ep.MultiScaleRankMemory.fit_calibration(
        z1[:cal_end], u1[:cal_end], target_noisy[:cal_end], panel["bounds"][:50]
    )
    assert mem.kappa > 0.0
    ev_bounds = [(lo - cal_end, hi - cal_end) for lo, hi in panel["bounds"][50:]]
    pred_ev = mem.predict(z1[cal_end:], u1[cal_end:])
    ic_base = np.mean(
        [
            oa.rank_ic_vector(
                z1[cal_end + lo : cal_end + hi], target_noisy[cal_end + lo : cal_end + hi]
            )
            for lo, hi in ev_bounds
        ]
    )
    ic_mem = np.mean(
        [
            oa.rank_ic_vector(pred_ev[lo:hi], target_noisy[cal_end + lo : cal_end + hi])
            for lo, hi in ev_bounds
        ]
    )
    assert ic_mem > ic_base + 0.05
