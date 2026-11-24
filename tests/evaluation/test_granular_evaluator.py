"""Pruebas TDD del Evaluador Granular de 6 Dimensiones (32 métricas) (`granular_evaluator`)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mars_titan.evaluation import backtest_overfitting as bo
from mars_titan.evaluation import granular_evaluator as ge


def _synthetic_eval_bundle(rng: np.random.Generator, sessions: int = 80, assets: int = 60):
    n = sessions * assets
    bounds = [(s * assets, (s + 1) * assets) for s in range(sessions)]
    vol = np.clip(0.01 + 0.01 * rng.random(n), 0.005, 0.05)
    liq = rng.normal(15.0, 2.0, size=n)
    alpha_true = rng.standard_normal(n)
    exposures = rng.standard_normal((n, 3))

    y = np.empty(n, dtype=np.float64)
    pred = np.empty(n, dtype=np.float64)
    market_ret = 0.01 * rng.standard_normal(sessions)
    for s, (lo, hi) in enumerate(bounds):
        f_ret = 0.008 * rng.standard_normal(3)
        y[lo:hi] = (
            0.004 * alpha_true[lo:hi]
            + exposures[lo:hi] @ f_ret
            + market_ret[s] * 0.3
            + vol[lo:hi] * rng.standard_normal(assets)
        )
        pred[lo:hi] = 0.004 * alpha_true[lo:hi] + 0.003 * rng.standard_normal(assets)

    # 9 cuantiles bien calibrados alrededor de pred
    z_q = np.array([-2.326, -1.645, -1.282, -0.674, 0.0, 0.674, 1.282, 1.645, 2.326])
    quantiles = pred[:, None] + vol[:, None] * z_q[None, :]
    probs = 1.0 / (1.0 + np.exp(-pred / np.maximum(vol, 1e-4)))
    return dict(
        y=y,
        pred=pred,
        quantiles=quantiles,
        probs=probs,
        vol=vol,
        liq=liq,
        exposures=exposures,
        bounds=bounds,
        market_ret=market_ret,
    )


def test_triple_barrier_labels_detect_upper_lower_and_vertical():
    """Triple-Barrier clasifica exactamente +1 (barrera superior), -1 (inferior) y 0 (expira)."""
    # Serie de retornos diarios (T=6, N=3)
    rets = np.array(
        [
            [0.02, -0.02, 0.001],
            [0.02, -0.02, -0.001],
            [-0.05, 0.05, 0.001],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    vol = np.full((6, 3), 0.01, dtype=np.float64)
    labels = ge.triple_barrier_labels(rets, vol, up_mult=2.5, down_mult=2.5, horizon=3)
    # En t=0: activo 0 acumula +0.04 en día 2 (> +0.025) -> +1
    # activo 1 acumula -0.04 en día 2 (< -0.025) -> -1
    # activo 2 nunca toca ±0.025 en 3 días -> 0
    assert labels[0, 0] == 1
    assert labels[0, 1] == -1
    assert labels[0, 2] == 0


def test_crps_and_pinball_are_strictly_better_for_calibrated_quantiles():
    """Cuantiles bien calibrados tienen menor CRPS y Pinball que cuantiles desplazados/estrechos."""
    rng = np.random.default_rng(12)
    b = _synthetic_eval_bundle(rng, sessions=30, assets=50)
    crps_good = ge.discrete_crps(b["quantiles"], b["y"], levels=ge.QUANTILE_LEVELS_9)
    crps_bad = ge.discrete_crps(b["quantiles"] + 0.04, b["y"], levels=ge.QUANTILE_LEVELS_9)
    assert crps_good["crps"] < crps_bad["crps"]
    assert crps_good["pinball_mean"] < crps_bad["pinball_mean"]
    assert len(crps_good["pinball_by_level"]) == 9


def test_deflated_sharpe_and_pbo_penalize_multiple_testing_and_overfit():
    """DSR baja cuando crecen n_trials y PBO sube con estrategias de ruido puro."""
    rng = np.random.default_rng(42)
    ret_good = 0.0015 + 0.008 * rng.standard_normal(500)
    dsr_1 = ge.deflated_sharpe_ratio(ret_good, n_trials=1)
    dsr_200 = ge.deflated_sharpe_ratio(ret_good, n_trials=200)
    assert dsr_1["dsr"] > dsr_200["dsr"]
    assert 0.0 <= dsr_200["dsr"] <= 1.0

    # PBO sobre 8 estrategias de ruido puro debe ser cercano a 0.5 (> 0.25)
    noise_matrix = rng.standard_normal((160, 8)) * 0.01
    pbo_noise = ge.cscv_pbo(noise_matrix, n_blocks=8)
    assert 0.0 <= pbo_noise["pbo"] <= 1.0
    assert pbo_noise["pbo"] > 0.2

    # Si una estrategia tiene señal muy fuerte real, PBO debe ser bajo (< 0.15)
    signal_matrix = noise_matrix.copy()
    signal_matrix[:, 0] += 0.008
    pbo_sig = ge.cscv_pbo(signal_matrix, n_blocks=8)
    assert pbo_sig["pbo"] < pbo_noise["pbo"]


def test_almgren_chriss_impact_and_granular_report_emit_all_six_dimensions():
    """evaluate_granular_six_dimensions devuelve las 6 dimensiones completas y finitas."""
    rng = np.random.default_rng(101)
    b = _synthetic_eval_bundle(rng, sessions=60, assets=60)
    # Construir pesos simples de cartera para evaluar Dim 6
    weights = np.zeros_like(b["pred"])
    for lo, hi in b["bounds"]:
        z = b["pred"][lo:hi] - b["pred"][lo:hi].mean()
        s = np.abs(z).sum()
        if s > 0:
            weights[lo:hi] = z / s

    report = ge.evaluate_granular_six_dimensions(
        y=b["y"],
        pred=b["pred"],
        bounds=b["bounds"],
        quantiles=b["quantiles"],
        quantile_levels=ge.QUANTILE_LEVELS_9,
        probabilities=b["probs"],
        volatility=b["vol"],
        liquidity=b["liq"],
        exposures=b["exposures"],
        portfolio_weights=weights,
        n_trials=10,
    )
    assert set(report.keys()) >= {
        "dim1_value",
        "dim2_ranking",
        "dim3_trend",
        "dim4_uncertainty",
        "dim5_error_autopsy",
        "dim6_economics",
    }
    assert report["dim1_value"]["mae_bps"] > 0.0
    assert np.isfinite(report["dim1_value"]["oos_r2"])
    assert report["dim2_ranking"]["rank_ic_mean"] > 0.05
    assert report["dim2_ranking"]["pure_alpha_ic_mean"] > 0.05
    assert 0.5 < report["dim3_trend"]["magnitude_weighted_sign_acc"] <= 1.0
    assert report["dim4_uncertainty"]["crps"] > 0.0
    assert "kupiec_var05_p" in report["dim4_uncertainty"]
    assert "pit_ks_stat" in report["dim4_uncertainty"]
    assert 0.0 <= report["dim5_error_autopsy"]["factor_var_share"] <= 1.0
    assert report["dim6_economics"]["net_sharpe_1x"] > 0.0
    assert report["dim6_economics"]["capacity_musd"] > 0.0
    assert 0.0 <= report["dim6_economics"]["dsr"] <= 1.0


def test_p0_1_ragged_panel_asset_ids_turnover_and_liquidation_cost():
    """P0-1: `asset_ids` alinea `w_prev` en paneles irregulares y cobra liquidación exacta."""
    # Sesión 0: activos [1, 2, 3, 4], pesos [+0.25, +0.25, -0.25, -0.25]
    # Sesión 1: baja activo 1 (liquidación exacta 0.25), entran [2, 3, 4] sin cambios + IPO 5 (0.0)
    # Sesión 2: solo [2, 3, 4] (N=3 != N_prev=4) sin cambios -> turnover sesión 2 = 0.0!
    asset_ids = np.array([1, 2, 3, 4, 2, 3, 4, 5, 2, 3, 4], dtype=np.int64)
    bounds = [(0, 4), (4, 8), (8, 11)]
    w = np.array([0.25, 0.25, -0.25, -0.25, 0.25, -0.25, -0.25, 0.0, 0.25, -0.25, -0.25])
    y = np.full(len(w), 0.001, dtype=np.float64)
    pred = w.copy() * 0.01
    vol = np.full(len(w), 0.02, dtype=np.float64)
    adv = np.full(len(w), 1e7, dtype=np.float64)

    rep_ids = ge.evaluate_granular_six_dimensions(
        y=y,
        pred=pred,
        bounds=bounds,
        portfolio_weights=w,
        volatility=vol,
        liquidity=adv,
        asset_ids=asset_ids,
        spread_bps=5.0,
        impact_eta=0.15,
        aum_musd=10.0,
    )
    # Turnovers exactos: s0 = 1.0 (entrada inicial), s1 = 0.25 (solo baja activo 1), s2 = 0.0
    expected_mean_to = (1.0 + 0.25 + 0.0) / 3.0
    assert rep_ids["dim6_economics"]["turnover_mean"] == pytest.approx(expected_mean_to, abs=1e-9)
    assert rep_ids["dim6_economics"]["exited_turnover_mean"] == pytest.approx(0.25 / 3.0, abs=1e-9)


def test_p0_5_exact_empirical_dsr_parity_with_backtest_overfitting_and_pbo_safety_margin():
    """P0-5: DSR con `trial_sharpes` coincide bit a bit con `backtest_overfitting.py` + PBO."""
    rng = np.random.default_rng(303)
    t_len, n_strats = 160, 12
    ret_mat = rng.standard_normal((t_len, n_strats)) * 0.008
    ret_mat[:, 0] += 0.0018

    trial_srs = np.array(
        [bo.sharpe_statistics(ret_mat[:, j])["sharpe"] for j in range(n_strats)], dtype=np.float64
    )
    emp_var = float(trial_srs.var(ddof=1))
    st0 = bo.sharpe_statistics(ret_mat[:, 0])
    bo_ref = bo.deflated_sharpe(
        st0["sharpe"],
        variance=emp_var,
        trials=n_strats,
        observations=st0["observations"],
        skewness=st0["skewness"],
        kurtosis=st0["kurtosis"],
    )
    ge_out = ge.deflated_sharpe_ratio(
        ret_mat[:, 0], n_trials=n_strats, trial_sharpes=trial_srs
    )
    assert ge_out["dsr"] == pytest.approx(bo_ref["probability"], abs=1e-10)
    assert ge_out["sr_threshold_session"] == pytest.approx(bo_ref["threshold"], abs=1e-10)

    # Integración en evaluate_granular_six_dimensions con trial_returns_matrix + safety_margin
    b = _synthetic_eval_bundle(rng, sessions=40, assets=40)
    rep = ge.evaluate_granular_six_dimensions(
        y=b["y"],
        pred=b["pred"],
        bounds=b["bounds"],
        volatility=b["vol"],
        liquidity=b["liq"],
        n_trials=n_strats,
        trial_returns_matrix=ret_mat,
    )
    dim6 = rep["dim6_economics"]
    assert 0.0 <= float(dim6["pbo"]) <= 1.0
    assert dim6["n_trials"] == n_strats
    assert dim6["safety_margin"] > 0.0
    expected_margin = dim6["gross_mean_bps"] / max(
        dim6["linear_cost_bps"] + dim6["impact_cost_bps"], 1e-12
    )
    assert dim6["safety_margin"] == pytest.approx(expected_margin, rel=1e-6)


def test_p0_5_exact_capacity_formula_and_adv_cap():
    """P0-5: capacidad separa spread lineal de impacto sqrt(AUM) y respeta tope 5 % ADV."""
    # 10 sesiones idénticas de 20 activos; solo 1 sesión de entrada y 9 de hold o open_to_close
    sessions, assets = 10, 20
    n = sessions * assets
    bounds = [(s * assets, (s + 1) * assets) for s in range(sessions)]
    w = np.tile(np.r_[np.full(10, 0.05), np.full(10, -0.05)], sessions)
    # Retorno bruto por sesión: dot(w, y) = 0.0040 (40 bps)
    y = np.tile(np.r_[np.full(10, 0.004), np.full(10, -0.004)], sessions)
    vol = np.full(n, 0.02, dtype=np.float64)
    adv_huge = np.full(n, 1e10, dtype=np.float64)  # ADV gigante para aislar fórmula analítica
    aum_musd = 50.0
    spread_bps = 5.0
    impact_eta = 0.15

    rep = ge.evaluate_granular_six_dimensions(
        y=y,
        pred=w * 0.01,
        bounds=bounds,
        portfolio_weights=w,
        volatility=vol,
        liquidity=adv_huge,
        execution_mode="open_to_close_200pct",
        spread_bps=spread_bps,
        impact_eta=impact_eta,
        aum_musd=aum_musd,
    )
    d6 = rep["dim6_economics"]
    mean_g = d6["gross_mean_bps"] * 1e-4
    c_lin = d6["linear_cost_bps"] * 1e-4
    c_imp = d6["impact_cost_bps"] * 1e-4
    expected_alpha_cap = aum_musd * (max(0.5 * mean_g - c_lin, 0.0) / max(c_imp, 1e-12)) ** 2
    assert d6["capacity_alpha_musd"] == pytest.approx(expected_alpha_cap, rel=1e-6)

    # Si spread lineal excede 0.5 * mean_g -> capacidad estrictamente 0.0
    rep_zero = ge.evaluate_granular_six_dimensions(
        y=y,
        pred=w * 0.01,
        bounds=bounds,
        portfolio_weights=w,
        volatility=vol,
        liquidity=adv_huge,
        execution_mode="open_to_close_200pct",
        spread_bps=25.0,  # 2 * 25 bps = 50 bps > 0.5 * 40 bps = 20 bps
        impact_eta=impact_eta,
        aum_musd=aum_musd,
    )
    assert rep_zero["dim6_economics"]["capacity_musd"] == 0.0

    # Con ADV pequeño ($1M), el tope part <= 5 % ADV restringe la capacidad
    adv_small = np.full(n, 1e6, dtype=np.float64)
    rep_cap = ge.evaluate_granular_six_dimensions(
        y=y,
        pred=w * 0.01,
        bounds=bounds,
        portfolio_weights=w,
        volatility=vol,
        liquidity=adv_small,
        execution_mode="open_to_close_200pct",
        spread_bps=1.0,
        impact_eta=0.05,
        aum_musd=aum_musd,
    )
    # Con |w_i| = 0.05 y ADV = $1M, trade = 0.05 * AUM <= 0.05 * $1M => AUM <= $1M
    assert rep_cap["dim6_economics"]["capacity_adv_cap_musd"] == pytest.approx(1.0, rel=1e-5)
    assert rep_cap["dim6_economics"]["capacity_musd"] <= 1.0 + 1e-5


def test_p0_5_execution_modes_cn_friction_survivorship_and_liquid_top_third():
    """P0-5: open_to_close_200pct, fricción CN (T+1/stamp/limit-lock), survivorship y top-1/3."""
    rng = np.random.default_rng(404)
    b = _synthetic_eval_bundle(rng, sessions=30, assets=60)
    # Crear señal persistente para que close_to_close tenga menor turnover que open_to_close_200pct
    base_sig = rng.standard_normal(60)
    pred_pers = np.concatenate(
        [0.004 * base_sig + 0.0005 * rng.standard_normal(60) for _ in range(30)]
    )
    y_mod = b["y"].copy()
    y_mod[0] = 0.105  # día bloqueado por límite superior CN (+10 %)
    adv_usd = np.exp(np.clip(b["liq"], 10.0, 22.0))

    rep_c2c_us = ge.evaluate_granular_six_dimensions(
        y=y_mod,
        pred=pred_pers,
        bounds=b["bounds"],
        volatility=b["vol"],
        liquidity=adv_usd,
        execution_mode="close_to_close",
        market="US",
    )
    rep_o2c_us = ge.evaluate_granular_six_dimensions(
        y=y_mod,
        pred=pred_pers,
        bounds=b["bounds"],
        volatility=b["vol"],
        liquidity=adv_usd,
        execution_mode="open_to_close_200pct",
        market="US",
    )
    rep_c2c_cn = ge.evaluate_granular_six_dimensions(
        y=y_mod,
        pred=pred_pers,
        bounds=b["bounds"],
        volatility=b["vol"],
        liquidity=adv_usd,
        execution_mode="close_to_close",
        market="CN",
        survivorship_disclosure=(
            "CN cohort filtered to active 2025 survivors; liquid top-third co-gated."
        ),
    )
    rep_o2c_cn = ge.evaluate_granular_six_dimensions(
        y=y_mod,
        pred=pred_pers,
        bounds=b["bounds"],
        volatility=b["vol"],
        liquidity=adv_usd,
        execution_mode="open_to_close_200pct",
        market="CN",
    )

    d6_c2c_us = rep_c2c_us["dim6_economics"]
    d6_o2c_us = rep_o2c_us["dim6_economics"]
    d6_c2c_cn = rep_c2c_cn["dim6_economics"]
    assert d6_o2c_us["turnover_mean"] == pytest.approx(2.0, abs=1e-6)
    assert d6_o2c_us["turnover_mean"] > d6_c2c_us["turnover_mean"] * 2.0
    assert d6_o2c_us["linear_cost_bps"] > d6_c2c_us["linear_cost_bps"] * 2.0

    # CN añade impuesto de timbre en ventas (10 bps) y detecta limit-lock + violación T+1 en O2C
    assert d6_c2c_cn["market"] == "CN"
    assert d6_c2c_cn["cn_stamp_tax_bps_cost"] > 0.0
    assert d6_c2c_cn["cn_limit_lock_rate"] > 0.0
    assert d6_c2c_cn["linear_cost_bps"] > d6_c2c_us["linear_cost_bps"]
    assert rep_c2c_cn["dim6_economics"]["cn_t1_violation"] is False
    assert rep_o2c_cn["dim6_economics"]["cn_t1_violation"] is True
    assert "survivors" in str(rep_c2c_cn["dim6_economics"]["survivorship_disclosure"])
    assert "survivorship_disclosure" in rep_c2c_us["dim6_economics"]

    # Helper de estratificación de tercio más líquido
    liq_slice = ge.evaluate_liquid_top_third_stratification(
        y=b["y"],
        pred=b["pred"],
        bounds=b["bounds"],
        liquidity=adv_usd,
        volatility=b["vol"],
    )
    assert liq_slice["selected_fraction"] == pytest.approx(1.0 / 3.0, abs=0.05)
    assert liq_slice["dim2_ranking"]["rank_ic_mean"] > 0.05
    assert np.isfinite(liq_slice["dim6_economics"]["net_sharpe_1x"])


def test_p0_5_dim4_oracle_randomized_pit_passes_ks_and_time_ordered_kupiec_christoffersen():
    """P0-5: PIT aleatorizado pasa KS (p>0.05) en oráculo y Kupiec/Christoffersen operan sobre T."""
    rng = np.random.default_rng(505)
    sessions, assets = 120, 40
    n = sessions * assets
    bounds = [(s * assets, (s + 1) * assets) for s in range(sessions)]
    vol = np.full(n, 0.02, dtype=np.float64)
    pred = rng.normal(0.0, 0.003, size=n)
    y_oracle = pred + vol * rng.standard_normal(n)
    z_q = stats.norm.ppf(ge.QUANTILE_LEVELS_9)
    q_oracle = pred[:, None] + vol[:, None] * z_q[None, :]

    rep_oracle = ge.evaluate_granular_six_dimensions(
        y=y_oracle,
        pred=pred,
        bounds=bounds,
        quantiles=q_oracle,
        quantile_levels=ge.QUANTILE_LEVELS_9,
        volatility=vol,
    )
    d4_o = rep_oracle["dim4_uncertainty"]
    assert d4_o["pit_ks_p"] > 0.05
    assert d4_o["var05_n_obs"] == sessions  # T=120 sesiones ordenadas, NO T*N=4800!
    assert d4_o["kupiec_var05_p"] > 0.05
    assert d4_o["christoffersen_var05_p"] > 0.05

    # Distribución mal calibrada (sesgada + estrecha) debe fallar PIT KS
    rep_bad = ge.evaluate_granular_six_dimensions(
        y=y_oracle,
        pred=pred,
        bounds=bounds,
        quantiles=(pred + 0.015)[:, None] + (0.4 * vol)[:, None] * z_q[None, :],
        quantile_levels=ge.QUANTILE_LEVELS_9,
        volatility=vol,
    )
    assert rep_bad["dim4_uncertainty"]["pit_ks_p"] < 0.01

    # Fallos VaR agrupados en el tiempo (12 sesiones consecutivas de crash) disparan Christoffersen
    y_cluster = y_oracle.copy()
    w_long = np.full(n, 1.0 / assets, dtype=np.float64)
    for s in range(40, 52):
        lo, hi = bounds[s]
        y_cluster[lo:hi] = pred[lo:hi] - 0.08
    rep_cluster = ge.evaluate_granular_six_dimensions(
        y=y_cluster,
        pred=pred,
        bounds=bounds,
        quantiles=q_oracle,
        quantile_levels=ge.QUANTILE_LEVELS_9,
        volatility=vol,
        portfolio_weights=w_long,
    )
    assert rep_cluster["dim4_uncertainty"]["var05_n_obs"] == sessions
    assert rep_cluster["dim4_uncertainty"]["christoffersen_var05_p"] < 0.01


def test_p1_5_gu_kelly_xiu_zero_benchmark_oos_r2_and_centered_diagnostic():
    """P1-5: `oos_r2` sigue Gu-Kelly-Xiu (denominador `sum(y^2)` sin lookahead `mean(y)`)."""
    rng = np.random.default_rng(707)
    sessions, assets = 40, 30
    n = sessions * assets
    bounds = [(s * assets, (s + 1) * assets) for s in range(sessions)]
    # Régimen alcista con media futura no nula (+200 bps): media futura != 0
    y = 0.02 + 0.015 * rng.standard_normal(n)

    # 1) Predictor nulo `pred == 0`: Gu-Kelly-Xiu OOS R² DEBE ser exactamente 0.0,
    #    mientras que el R² centrado (lookahead) es estrictamente negativo!
    rep_zero = ge.evaluate_granular_six_dimensions(
        y=y,
        pred=np.zeros(n, dtype=np.float64),
        bounds=bounds,
    )
    assert rep_zero["dim1_value"]["oos_r2"] == pytest.approx(0.0, abs=1e-12)
    assert rep_zero["dim1_value"]["oos_r2_centered"] < -0.5

    # 2) Predictor no trivial: comprobar fórmula exacta Gu-Kelly-Xiu y diagnóstico centrado
    pred = 0.015 + 0.4 * (y - 0.02) + 0.005 * rng.standard_normal(n)
    rep_pred = ge.evaluate_granular_six_dimensions(y=y, pred=pred, bounds=bounds)
    ss_err = float(np.sum((y - pred) ** 2))
    expected_gkx = 1.0 - ss_err / float(np.sum(y * y))
    expected_centered = 1.0 - ss_err / float(np.sum((y - np.mean(y)) ** 2))
    assert rep_pred["dim1_value"]["oos_r2"] == pytest.approx(expected_gkx, abs=1e-12)
    assert rep_pred["dim1_value"]["oos_r2_centered"] == pytest.approx(expected_centered, abs=1e-12)
    assert rep_pred["dim1_value"]["oos_r2"] > rep_pred["dim1_value"]["oos_r2_centered"]


def test_p1_5_correlated_factor_oracle_passes_kupiec_pof_without_masking_miscalibration():
    """P1-5: VaR de cartera usa covarianza factorial / EWMA causal y pasa Kupiec POF en oráculo."""
    rng = np.random.default_rng(808)
    sessions, assets, n_factors = 240, 60, 3
    n = sessions * assets
    bounds = [(s * assets, (s + 1) * assets) for s in range(sessions)]

    # Exposiciones B: factor 0 = beta de mercado ~ U(0.6, 1.4); factores 1..2 ~ N(0, 1)
    exposures = np.empty((n, n_factors), dtype=np.float64)
    exposures[:, 0] = rng.uniform(0.6, 1.4, size=n)
    exposures[:, 1:] = rng.standard_normal((n, n_factors - 1))

    sigma_f_diag = np.array([0.018, 0.012, 0.010], dtype=np.float64)
    factor_cov = np.diag(sigma_f_diag**2)
    idio_sd = rng.uniform(0.010, 0.016, size=n)

    # Media prevista + cartera con exposición residual a factores (long-only o sesgo beta)
    pred = rng.normal(0.0005, 0.002, size=n)
    weights = np.full(n, 1.0 / assets, dtype=np.float64)

    y = np.empty(n, dtype=np.float64)
    marg_sd = np.empty(n, dtype=np.float64)
    for lo, hi in bounds:
        b_s = exposures[lo:hi]
        f_ret = sigma_f_diag * rng.standard_normal(n_factors)
        eps = idio_sd[lo:hi] * rng.standard_normal(hi - lo)
        y[lo:hi] = pred[lo:hi] + b_s @ f_ret + eps
        fac_var_i = np.einsum("ij,jk,ik->i", b_s, factor_cov, b_s)
        marg_sd[lo:hi] = np.sqrt(fac_var_i + idio_sd[lo:hi] ** 2)

    z_q = stats.norm.ppf(ge.QUANTILE_LEVELS_9)
    q_oracle = pred[:, None] + marg_sd[:, None] * z_q[None, :]

    # 0) Confirmar que el modo legado `independent` subestima vol 5-10x y falla Kupiec POF!
    rep_indep = ge.evaluate_granular_six_dimensions(
        y=y,
        pred=pred,
        bounds=bounds,
        quantiles=q_oracle,
        exposures=exposures,
        portfolio_weights=weights,
        var_cov_mode="independent",
    )
    assert rep_indep["dim4_uncertainty"]["var05_rate"] > 0.20
    assert rep_indep["dim4_uncertainty"]["kupiec_var05_p"] < 1e-6

    # 1) Covarianza factorial explícita (`exposures` + `factor_cov`): Kupiec POF PASA (p > 0.05)
    rep_fac_exact = ge.evaluate_granular_six_dimensions(
        y=y,
        pred=pred,
        bounds=bounds,
        quantiles=q_oracle,
        exposures=exposures,
        factor_cov=factor_cov,
        portfolio_weights=weights,
    )
    d4_ex = rep_fac_exact["dim4_uncertainty"]
    assert d4_ex["var05_cov_mode"] == "explicit_factor"
    assert d4_ex["pit_ks_p"] > 0.05
    assert 0.02 <= float(d4_ex["var05_rate"]) <= 0.09
    assert float(d4_ex["kupiec_var05_p"]) > 0.05

    # 2) Solo `exposures` (sin `factor_cov` explícito): EWMA factorial causal PASA Kupiec POF
    rep_fac_causal = ge.evaluate_granular_six_dimensions(
        y=y,
        pred=pred,
        bounds=bounds,
        quantiles=q_oracle,
        exposures=exposures,
        portfolio_weights=weights,
    )
    d4_fc = rep_fac_causal["dim4_uncertainty"]
    assert d4_fc["var05_cov_mode"] == "causal_factor_ewma"
    assert 0.02 <= float(d4_fc["var05_rate"]) <= 0.09
    assert float(d4_fc["kupiec_var05_p"]) > 0.05

    # 3) Sin `exposures` ni `factor_cov`: EWMA causal de correlación PASA Kupiec POF
    rep_no_exp = ge.evaluate_granular_six_dimensions(
        y=y,
        pred=pred,
        bounds=bounds,
        quantiles=q_oracle,
        portfolio_weights=weights,
    )
    d4_ne = rep_no_exp["dim4_uncertainty"]
    assert d4_ne["var05_cov_mode"] == "causal_corr_ewma"
    assert 0.02 <= float(d4_ne["var05_rate"]) <= 0.09
    assert float(d4_ne["kupiec_var05_p"]) > 0.05

    # 4) Cuantiles severamente mal calibrados (0.35x demasiado estrechos) SIGUEN FALLANDO Kupiec!
    q_narrow = pred[:, None] + (0.35 * marg_sd)[:, None] * z_q[None, :]
    rep_bad_fac = ge.evaluate_granular_six_dimensions(
        y=y,
        pred=pred,
        bounds=bounds,
        quantiles=q_narrow,
        exposures=exposures,
        portfolio_weights=weights,
    )
    rep_bad_no_exp = ge.evaluate_granular_six_dimensions(
        y=y,
        pred=pred,
        bounds=bounds,
        quantiles=q_narrow,
        portfolio_weights=weights,
    )
    assert float(rep_bad_fac["dim4_uncertainty"]["kupiec_var05_p"]) < 0.01
    assert float(rep_bad_no_exp["dim4_uncertainty"]["kupiec_var05_p"]) < 0.01

    # 5) Causalidad estricta: mutar sesiones futuras no altera tasa VaR del prefijo
    cut = 120
    hi_cut = bounds[cut - 1][1]
    y_mut = y.copy()
    y_mut[hi_cut:] -= 0.50
    rep_pref_a = ge.evaluate_granular_six_dimensions(
        y=y[:hi_cut],
        pred=pred[:hi_cut],
        bounds=bounds[:cut],
        quantiles=q_oracle[:hi_cut],
        exposures=exposures[:hi_cut],
        portfolio_weights=weights[:hi_cut],
    )
    rep_pref_b = ge.evaluate_granular_six_dimensions(
        y=y_mut[:hi_cut],
        pred=pred[:hi_cut],
        bounds=bounds[:cut],
        quantiles=q_oracle[:hi_cut],
        exposures=exposures[:hi_cut],
        portfolio_weights=weights[:hi_cut],
    )
    assert rep_pref_a["dim4_uncertainty"]["var05_rate"] == rep_pref_b["dim4_uncertainty"][
        "var05_rate"
    ]
