"""Pruebas de neutralización causal de factores por sesión (`CausalFactorNeutralizer`)."""

from __future__ import annotations

import numpy as np

from mars_titan.evaluation import factor_neutralization as fn
from mars_titan.evaluation import online_aggregator as oa


def _factor_contaminated_panel(
    rng: np.random.Generator,
    *,
    sessions: int = 120,
    assets: int = 300,
    factor_tilt: float = 0.85,
    factor_vol: float = 0.035,
    idio_signal: float = 0.14,
):
    """Panel donde la señal tiene exposición espuria a 3 factores de estilo muy volátiles."""
    n = sessions * assets
    bounds = [(s * assets, (s + 1) * assets) for s in range(sessions)]
    exposures = rng.standard_normal((n, 3))
    alpha_true = rng.standard_normal(n)
    score = np.empty(n, dtype=np.float64)
    target = np.empty(n, dtype=np.float64)
    for _s, (lo, hi) in enumerate(bounds):
        b = exposures[lo:hi]
        f_ret = factor_vol * rng.standard_normal(3)
        # Señal contaminada por una inclinación fija hacia los factores de estilo
        tilt_vec = np.array([1.0, -0.8, 0.6]) * factor_tilt
        score[lo:hi] = (
            idio_signal * alpha_true[lo:hi] + b @ tilt_vec + 0.5 * rng.standard_normal(assets)
        )
        target[lo:hi] = 0.008 * alpha_true[lo:hi] + b @ f_ret + 0.015 * rng.standard_normal(assets)
    return dict(score=score, target=target, exposures=exposures, bounds=bounds)


def test_factor_neutralization_is_exact_orthogonal_projection_when_lam_zero():
    """Con lam=0 y gamma=1, la señal neutralizada tiene exposición exactamente 0 a B_t."""
    rng = np.random.default_rng(12)
    panel = _factor_contaminated_panel(rng, sessions=5, assets=100)
    neut = fn.neutralize_sessions(
        panel["score"],
        panel["exposures"],
        panel["bounds"],
        ridge_lambda=0.0,
        gamma=1.0,
        include_intercept=True,
    )
    for lo, hi in panel["bounds"]:
        b = np.column_stack(
            [np.ones(hi - lo), fn._standardize_exposures(panel["exposures"][lo:hi])]
        )
        # B^T s_neut debe ser ~0
        proj = b.T @ neut[lo:hi]
        np.testing.assert_allclose(proj, 0.0, atol=1e-10)


def test_factor_neutralization_crushes_daily_ic_variance_and_boosts_icir_and_hit_rate():
    """Neutralizar factores espurios reduce sigma_IC y eleva ICIR y P(IC > 0) fuera de muestra."""
    rng = np.random.default_rng(42)
    cal = _factor_contaminated_panel(rng, sessions=100, assets=300)
    ev = _factor_contaminated_panel(rng, sessions=150, assets=300)

    model = fn.CausalFactorNeutralizer.fit_calibration(
        cal["score"],
        cal["target"],
        cal["exposures"],
        cal["bounds"],
        gammas=(0.0, 0.25, 0.5, 0.75, 1.0),
        lambdas=(0.01, 1.0, 10.0),
        objective="icir",
    )
    assert model.gamma >= 0.5

    raw_ics = np.array(
        [oa.rank_ic_vector(ev["score"][lo:hi], ev["target"][lo:hi]) for lo, hi in ev["bounds"]]
    )
    neut_score = model.transform(ev["score"], ev["exposures"], ev["bounds"])
    neut_ics = np.array(
        [oa.rank_ic_vector(neut_score[lo:hi], ev["target"][lo:hi]) for lo, hi in ev["bounds"]]
    )

    decomp = fn.variance_decomposition(raw_ics, neut_ics, mean_assets=300)
    assert decomp["neut_std"] < decomp["raw_std"] * 0.75
    assert decomp["neut_icir"] > decomp["raw_icir"] * 1.35
    assert decomp["neut_positive_rate"] > decomp["raw_positive_rate"] + 0.08


def test_causal_neutralizer_preserves_pure_alpha_when_factor_is_genuine_signal():
    """Si los factores NO contaminan (son señal genuina), la calibración elige gamma cercano a 0."""
    rng = np.random.default_rng(99)
    sessions, assets = 100, 200
    n = sessions * assets
    bounds = [(s * assets, (s + 1) * assets) for s in range(sessions)]
    exposures = rng.standard_normal((n, 2))
    # El factor siempre tiene prima positiva constante => neutralizarlo destruiría señal
    score = exposures[:, 0] + 0.3 * rng.standard_normal(n)
    target = 0.01 * exposures[:, 0] + 0.02 * rng.standard_normal(n)

    model = fn.CausalFactorNeutralizer.fit_calibration(
        score, target, exposures, bounds, gammas=(0.0, 0.5, 1.0), lambdas=(0.1, 1.0)
    )
    assert model.gamma == 0.0


def test_causal_rolling_pca_exposures_strictly_respect_maturity_barrier():
    """Las exposiciones PCA rodantes en t solo leen retornos maduros en tau <= t - lag."""
    rng = np.random.default_rng(7)
    n_sessions, n_assets = 40, 50
    returns = rng.standard_normal((n_sessions, n_assets)) * 0.02
    # Inyectar un factor latente común fuerte
    f = rng.standard_normal((n_sessions, 2)) * 0.03
    loadings_true = rng.standard_normal((n_assets, 2))
    returns += f @ loadings_true.T

    exp1 = fn.causal_rolling_pca_exposures(returns, n_components=2, window=20, maturity_lag=2)
    returns_mut = returns.copy()
    # Mutar masivamente desde sesión 25 en adelante
    returns_mut[25:] = 500.0
    exp2 = fn.causal_rolling_pca_exposures(returns_mut, n_components=2, window=20, maturity_lag=2)

    # Hasta sesión 26 inclusive (25 + 2 = 27 es la primera que ve la sesión 25 madura), bit-exacto!
    np.testing.assert_array_equal(exp1[:27], exp2[:27])
    # A partir de 27 cambia
    assert not np.array_equal(exp1[27:], exp2[27:])
    # Las primeras sesiones sin historia suficiente son ceros finitos
    assert np.all(np.isfinite(exp1))
    assert np.all(exp1[:5] == 0.0)


def test_neutralize_sessions_handles_nans_and_weights_cleanly():
    """NaN en exposiciones o pesos se imputan causalmente a la mediana/cero sin propagar NaNs."""
    rng = np.random.default_rng(13)
    score = rng.standard_normal(200)
    exposures = rng.standard_normal((200, 3))
    exposures[0:10, 0] = np.nan
    exposures[50, 2] = np.inf
    weights = rng.uniform(0.5, 2.0, size=200)
    out = fn.neutralize_sessions(
        score, exposures, [(0, 100), (100, 200)], weights=weights, ridge_lambda=1.0, gamma=1.0
    )
    assert out.shape == score.shape
    assert np.all(np.isfinite(out))
