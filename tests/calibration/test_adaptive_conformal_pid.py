"""Pruebas de calibración conforme normalizada localmente con controlador PID en línea."""

from __future__ import annotations

import numpy as np
import pytest

from mars_titan.calibration import adaptive_conformal_pid as pid
from mars_titan.environments.cohorts import FINAL_TEST_START_US

DAY = 86_400_000_000
START = 1_357_084_800_000_000  # 2013-01-02 UTC


def _heteroskedastic_panel(
    rng: np.random.Generator,
    *,
    sessions: int = 80,
    assets: int = 200,
    vol_regime_shift_at: int | None = None,
    shift_mult: float = 2.0,
    base_scale: float = 0.015,
):
    """Panel heterocedástico con 10 deciles de volatilidad y cambio de régimen opcional."""
    n = sessions * assets
    times = np.repeat(START + np.arange(sessions, dtype=np.int64) * DAY, assets)
    matures = times + DAY
    groups = np.full(n, "US")
    # Volatilidad conocida en t abarca 10x entre deciles (0.25x a 2.5x)
    vol = rng.lognormal(mean=np.log(base_scale), sigma=0.75, size=n)
    true_sigma = vol.copy()
    if vol_regime_shift_at is not None:
        shift_mask = np.repeat(np.arange(sessions) >= vol_regime_shift_at, assets)
        true_sigma[shift_mask] *= shift_mult
    mu = 0.001 * rng.standard_normal(n)
    y = mu + true_sigma * rng.standard_normal(n)
    # Cuantiles iniciales infradispersos (0.7x del real) y proporcionales a la vol conocida
    # niveles: (0.025, 0.1, 0.5, 0.9, 0.975)
    z = np.array([-1.96, -1.28155, 0.0, 1.28155, 1.96], dtype=np.float64) * 0.70
    q = mu[:, None] + vol[:, None] * z[None, :]
    return dict(
        target=y,
        quantiles=q,
        groups=groups,
        prediction_at=times,
        matures_at=matures,
        volatility=vol,
    )


def test_normalized_cqr_eliminates_volatility_decile_collapse():
    """La CQR normalizada localmente debe mantener cobertura homogénea en el peor decil de vol."""
    rng = np.random.default_rng(42)
    cal = _heteroskedastic_panel(rng, sessions=40, assets=250)
    ev = _heteroskedastic_panel(rng, sessions=60, assets=250)

    # CQR aditiva estática (control): colapsa en el decil de máxima volatilidad
    lo_cal, hi_cal = cal["quantiles"][:, 1], cal["quantiles"][:, 3]
    add_scores = np.maximum(lo_cal - cal["target"], cal["target"] - hi_cal)
    q_add = float(np.quantile(add_scores, 0.80))
    lo_ev, hi_ev = ev["quantiles"][:, 1], ev["quantiles"][:, 3]
    cov_add = (ev["target"] >= lo_ev - q_add) & (ev["target"] <= hi_ev + q_add)

    # CQR normalizada con PID (kp=0, ki=0 => estática normalizada pura para aislar la normalización)
    ctrl = pid.NormalizedConformalPID.start(
        cal["target"],
        cal["quantiles"],
        cal["groups"],
        levels=(0.025, 0.1, 0.5, 0.9, 0.975),
        nominals=(0.8, 0.95),
        scale_mode="width",
        kp=0.0,
        ki=0.0,
        kd=0.0,
    )
    emitted, _, _ = pid.replay_normalized_pid(
        ctrl,
        ev["quantiles"],
        ev["groups"],
        ev["prediction_at"],
        ev["target"],
        ev["matures_at"],
    )
    cov_norm = (ev["target"] >= emitted[:, 1]) & (ev["target"] <= emitted[:, 3])

    # Comparar peor decil de volatilidad
    deciles = np.digitize(ev["volatility"], np.quantile(ev["volatility"], np.linspace(0.1, 0.9, 9)))
    worst_add = min(float(cov_add[deciles == d].mean()) for d in range(10))
    worst_norm = min(float(cov_norm[deciles == d].mean()) for d in range(10))

    assert worst_add < 0.70, f"El panel debe reproducir el colapso aditivo (fue {worst_add:.3f})"
    assert worst_norm > 0.77, (
        f"CQR normalizada debe estabilizar el peor decil (fue {worst_norm:.3f})"
    )
    assert abs(float(cov_norm.mean()) - 0.80) < 0.02
    # Además, la anchura mediana (activos tranquilos) es más estrecha y el Winkler/Interval Score
    # propio (que penaliza la falta de cobertura en activos volátiles) es estrictamente menor!
    alpha = 0.20
    lo_a, hi_a = lo_ev - q_add, hi_ev + q_add
    lo_n, hi_n = emitted[:, 1], emitted[:, 3]
    y = ev["target"]
    is_add = float(
        np.mean(
            (hi_a - lo_a)
            + (2.0 / alpha) * np.maximum(lo_a - y, 0.0)
            + (2.0 / alpha) * np.maximum(y - hi_a, 0.0)
        )
    )
    is_norm = float(
        np.mean(
            (hi_n - lo_n)
            + (2.0 / alpha) * np.maximum(lo_n - y, 0.0)
            + (2.0 / alpha) * np.maximum(y - hi_n, 0.0)
        )
    )
    # En activos tranquilos (decil 0 de vol) evita el inflado aditivo, y el Winkler/Interval Score
    # propio (que penaliza la falta de cobertura en activos volátiles) es estrictamente menor!
    assert float(np.mean((hi_n - lo_n)[deciles == 0])) < float(np.mean((hi_a - lo_a)[deciles == 0]))
    assert is_norm < is_add


def test_pid_controller_recovers_coverage_after_volatility_regime_shift():
    """Ante un salto de volatilidad 2x no observado en vol_21, el PID recupera cobertura ~0.80."""
    rng = np.random.default_rng(7)
    cal = _heteroskedastic_panel(rng, sessions=30, assets=200)
    ev = _heteroskedastic_panel(
        rng, sessions=120, assets=200, vol_regime_shift_at=20, shift_mult=2.0
    )

    static_ctrl = pid.NormalizedConformalPID.start(
        cal["target"],
        cal["quantiles"],
        cal["groups"],
        levels=(0.025, 0.1, 0.5, 0.9, 0.975),
        nominals=(0.8, 0.95),
        scale_mode="width",
        kp=0.0,
        ki=0.0,
        kd=0.0,
    )
    adaptive_ctrl = pid.NormalizedConformalPID.start(
        cal["target"],
        cal["quantiles"],
        cal["groups"],
        levels=(0.025, 0.1, 0.5, 0.9, 0.975),
        nominals=(0.8, 0.95),
        scale_mode="width",
        kp=0.08,
        ki=0.15,
        c_sat=2.0,
        kd=0.0,
    )
    q_static, _, _ = pid.replay_normalized_pid(
        static_ctrl,
        ev["quantiles"],
        ev["groups"],
        ev["prediction_at"],
        ev["target"],
        ev["matures_at"],
    )
    q_pid, emissions, updates = pid.replay_normalized_pid(
        adaptive_ctrl,
        ev["quantiles"],
        ev["groups"],
        ev["prediction_at"],
        ev["target"],
        ev["matures_at"],
    )
    # Evaluar en las últimas 60 sesiones post-cambio de régimen
    post = slice(60 * 200, None)
    cov_static = float(
        np.mean(
            (ev["target"][post] >= q_static[post, 1]) & (ev["target"][post] <= q_static[post, 3])
        )
    )
    cov_pid = float(
        np.mean((ev["target"][post] >= q_pid[post, 1]) & (ev["target"][post] <= q_pid[post, 3]))
    )
    assert cov_static < 0.65
    assert abs(cov_pid - 0.80) < 0.04, f"PID debe recuperar 0.80 tras el shock (fue {cov_pid:.3f})"
    assert len(emissions) == 120 and len(updates) == 119


def test_strict_causality_with_delayed_maturity_and_final_test_seal():
    """Mutaciones futuras no cambian emisiones pasadas, y >= 2024 es rechazado."""
    rng = np.random.default_rng(11)
    cal = _heteroskedastic_panel(rng, sessions=20, assets=50)
    ev = _heteroskedastic_panel(rng, sessions=15, assets=50)
    # Retardo de madurez de 3 días
    ev["matures_at"] = ev["prediction_at"] + 3 * DAY

    c1 = pid.NormalizedConformalPID.start(
        cal["target"], cal["quantiles"], cal["groups"], kp=0.1, ki=0.1
    )
    c2 = pid.NormalizedConformalPID.start(
        cal["target"], cal["quantiles"], cal["groups"], kp=0.1, ki=0.1
    )
    y_mut = ev["target"].copy()
    # Mutar masivamente las sesiones >= 5
    y_mut[5 * 50 :] = 999.0
    out1, _, _ = pid.replay_normalized_pid(
        c1, ev["quantiles"], ev["groups"], ev["prediction_at"], ev["target"], ev["matures_at"]
    )
    out2, _, _ = pid.replay_normalized_pid(
        c2, ev["quantiles"], ev["groups"], ev["prediction_at"], y_mut, ev["matures_at"]
    )
    # Hasta la sesión 7 inclusive (5 + 3 = 8 es la primera que ve la sesión 5 madura),
    # las emisiones DEBEN ser bit-exactas!
    np.testing.assert_array_equal(out1[: 8 * 50], out2[: 8 * 50])
    # A partir de la sesión 8 ya divergen
    assert not np.array_equal(out1[8 * 50 :], out2[8 * 50 :])

    # Sello 2024
    bad_times = ev["prediction_at"].copy()
    bad_times[-1] = FINAL_TEST_START_US
    c3 = pid.NormalizedConformalPID.start(cal["target"], cal["quantiles"], cal["groups"])
    with pytest.raises(ValueError, match="2024|test final|sello"):
        pid.replay_normalized_pid(
            c3, ev["quantiles"], ev["groups"], bad_times, ev["target"], bad_times + DAY
        )


def test_export_restore_roundtrip_is_bit_exact_and_monotonic_quantiles():
    """Exportar/restaurar preserva estado SHA-256 y emisiones bit a bit; cuantiles no se cruzan."""
    rng = np.random.default_rng(19)
    cal = _heteroskedastic_panel(rng, sessions=20, assets=60)
    ev = _heteroskedastic_panel(rng, sessions=20, assets=60)

    c_full = pid.NormalizedConformalPID.start(
        cal["target"], cal["quantiles"], cal["groups"], kp=0.08, ki=0.12, kd=0.05
    )
    c_split = pid.NormalizedConformalPID.start(
        cal["target"], cal["quantiles"], cal["groups"], kp=0.08, ki=0.12, kd=0.05
    )
    out_full, _, _ = pid.replay_normalized_pid(
        c_full, ev["quantiles"], ev["groups"], ev["prediction_at"], ev["target"], ev["matures_at"]
    )

    mid = 10 * 60
    part1, _, _ = pid.replay_normalized_pid(
        c_split,
        ev["quantiles"][:mid],
        ev["groups"][:mid],
        ev["prediction_at"][:mid],
        ev["target"][:mid],
        ev["matures_at"][:mid],
    )
    state = c_split.export()
    restored = pid.NormalizedConformalPID.restore(state)
    assert restored.state_sha256() == c_split.state_sha256()
    part2, _, _ = pid.replay_normalized_pid(
        restored,
        ev["quantiles"][mid:],
        ev["groups"][mid:],
        ev["prediction_at"][mid:],
        ev["target"][mid:],
        ev["matures_at"][mid:],
    )
    np.testing.assert_array_equal(out_full, np.vstack([part1, part2]))
    assert restored.state_sha256() == c_full.state_sha256()
    # Monotonía de cuantiles en todas las filas emitidas
    assert np.all(np.diff(out_full, axis=1) >= 0.0)


def test_volatility_and_hybrid_scale_modes_work_causally():
    """Los modos `volatility` y `hybrid` aceptan escala externa y mantienen cobertura."""
    rng = np.random.default_rng(23)
    cal = _heteroskedastic_panel(rng, sessions=30, assets=120)
    ev = _heteroskedastic_panel(rng, sessions=30, assets=120)
    for mode in ("width", "volatility", "hybrid"):
        ctrl = pid.NormalizedConformalPID.start(
            cal["target"],
            cal["quantiles"],
            cal["groups"],
            scale_mode=mode,
            cal_scales=cal["volatility"],
            kp=0.05,
            ki=0.08,
        )
        emitted, _, _ = pid.replay_normalized_pid(
            ctrl,
            ev["quantiles"],
            ev["groups"],
            ev["prediction_at"],
            ev["target"],
            ev["matures_at"],
            scales=ev["volatility"],
        )
        assert emitted.shape == ev["quantiles"].shape
        assert np.all(np.isfinite(emitted))
        assert np.all(np.diff(emitted, axis=1) >= 0.0)
        cov80 = float(np.mean((ev["target"] >= emitted[:, 1]) & (ev["target"] <= emitted[:, 3])))
        assert abs(cov80 - 0.80) < 0.04


def test_mondrian_stratified_normalized_pid_equalizes_all_vol_deciles():
    """Mondrian por decil de volatilidad + normalización local + PID iguala todos los deciles."""
    rng = np.random.default_rng(31)
    cal = _heteroskedastic_panel(rng, sessions=40, assets=250)
    ev = _heteroskedastic_panel(
        rng, sessions=80, assets=250, vol_regime_shift_at=20, shift_mult=1.5
    )
    cal_dec = np.digitize(
        cal["volatility"], np.quantile(cal["volatility"], np.linspace(0.1, 0.9, 9))
    )
    ev_dec = np.digitize(ev["volatility"], np.quantile(cal["volatility"], np.linspace(0.1, 0.9, 9)))
    g_cal = pid.make_stratified_groups(cal["groups"], cal_dec)
    g_ev = pid.make_stratified_groups(ev["groups"], ev_dec)

    ctrl = pid.NormalizedConformalPID.start(
        cal["target"], cal["quantiles"], g_cal, scale_mode="width", kp=0.10, ki=0.20
    )
    emitted, _, _ = pid.replay_normalized_pid(
        ctrl, ev["quantiles"], g_ev, ev["prediction_at"], ev["target"], ev["matures_at"]
    )
    cov80 = (ev["target"] >= emitted[:, 1]) & (ev["target"] <= emitted[:, 3])
    # En la segunda mitad post-shock (sesiones 40..79), el peor decil converge cerca de 0.80
    post = slice(40 * 250, None)
    worst_dec = min(
        float(cov80[post][ev_dec[post] == d].mean()) for d in range(10) if np.any(ev_dec[post] == d)
    )
    assert worst_dec > 0.74, f"Peor decil Mondrian-NPID demasiado bajo: {worst_dec:.3f}"
    assert abs(float(cov80[post].mean()) - 0.80) < 0.03
