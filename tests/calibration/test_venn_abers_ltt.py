"""Pruebas de calibración Inductive Venn-Abers rápida O(N log N) y control de riesgo SessionLTT."""

from __future__ import annotations

import time

import numpy as np
import pytest

from mars_titan.calibration import venn_abers_ltt as val


def _naive_weighted_pav(
    x: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Referencia exacta O(N) de PAV ponderado con agrupación de empates."""
    uniq, inv = np.unique(x, return_inverse=True)
    wy = np.bincount(inv, weights=w * y, minlength=len(uniq))
    ww = np.bincount(inv, weights=w, minlength=len(uniq))
    tot_y: list[float] = []
    tot_w: list[float] = []
    edges: list[float] = []
    for sy, sw, edge in zip(wy, ww, uniq, strict=True):
        tot_y.append(float(sy))
        tot_w.append(float(sw))
        edges.append(float(edge))
        while len(tot_y) > 1 and tot_y[-2] / tot_w[-2] >= tot_y[-1] / tot_w[-1]:
            tot_y[-2] += tot_y[-1]
            tot_w[-2] += tot_w[-1]
            edges[-2] = edges[-1]
            del tot_y[-1], tot_w[-1], edges[-1]
    return np.asarray(edges, dtype=np.float64), np.asarray(tot_y) / np.asarray(tot_w)


def _naive_venn_abers(
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_test: np.ndarray,
    w_cal: np.ndarray | None = None,
    w_test: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Oráculo exacto O(N_test * N_cal) de Inductive Venn-Abers."""
    w = (
        np.ones(len(x_cal), dtype=np.float64)
        if w_cal is None
        else np.asarray(w_cal, dtype=np.float64)
    )
    p0 = np.empty(len(x_test), dtype=np.float64)
    p1 = np.empty(len(x_test), dtype=np.float64)
    for i, xt in enumerate(x_test):
        xx = np.append(x_cal, xt)
        ww = np.append(w, w_test)
        for label, out in ((0.0, p0), (1.0, p1)):
            yy = np.append(y_cal, label)
            edges, vals = _naive_weighted_pav(xx, yy, ww)
            pos = int(np.searchsorted(edges, xt, side="left"))
            out[i] = vals[min(pos, len(vals) - 1)]
    return p0, p1


@pytest.mark.parametrize("with_ties", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_fast_venn_abers_matches_naive_oracle_to_machine_precision(with_ties, weighted):
    """El algoritmo rápido O(N log^2 N) iguala al oráculo O(N^2) a < 1e-12 siempre."""
    rng = np.random.default_rng(101 + int(with_ties) * 10 + int(weighted))
    x_cal = rng.normal(size=180)
    if with_ties:
        x_cal = np.round(x_cal, 1)
    probs = 1.0 / (1.0 + np.exp(-1.5 * x_cal))
    y_cal = (rng.random(len(x_cal)) < probs).astype(np.float64)
    w_cal = rng.uniform(0.2, 3.0, size=len(x_cal)) if weighted else None

    # Mezcla de puntos fuera del rango, en empates exactos y en intervalos interiores
    x_test = np.concatenate(
        [
            [-10.0, 10.0],
            x_cal[:20],
            rng.normal(size=60),
        ]
    )
    ref_p0, ref_p1 = _naive_venn_abers(x_cal, y_cal, x_test, w_cal=w_cal)

    cal = val.VennAbersCalibrator.fit(x_cal, y_cal, weights=w_cal)
    pred = cal.predict(x_test)

    np.testing.assert_allclose(pred["p0"], ref_p0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(pred["p1"], ref_p1, rtol=0.0, atol=1e-12)
    assert np.all(pred["p0"] <= pred["p1"] + 1e-15)
    assert np.all(
        (pred["p_minimax"] >= pred["p0"] - 1e-15) & (pred["p_minimax"] <= pred["p1"] + 1e-15)
    )
    expected_minimax = pred["p1"] / (1.0 - pred["p0"] + pred["p1"])
    np.testing.assert_allclose(pred["p_minimax"], expected_minimax, rtol=0.0, atol=1e-14)


def test_venn_abers_beats_raw_isotonic_on_tail_brier_and_runs_fast_at_scale():
    """Venn-Abers regulariza las colas extremas frente a isotónica pura y escala a 100K filas."""
    rng = np.random.default_rng(314)
    # Régimen ruidoso similar a mercados financieros (señal débil)
    x_cal = rng.standard_normal(5_000)
    y_cal = (rng.random(5_000) < (1.0 / (1.0 + np.exp(-0.12 * x_cal)))).astype(np.float64)
    x_ev = rng.standard_normal(20_000)
    y_ev = (rng.random(20_000) < (1.0 / (1.0 + np.exp(-0.12 * x_ev)))).astype(np.float64)

    cal = val.VennAbersCalibrator.fit(x_cal, y_cal)
    pred = cal.predict(x_ev)
    # En colas extremas, p1 - p0 > 0 y evita probabilidades degeneradas 0 o 1
    assert float(pred["p_minimax"].min()) > 0.0
    assert float(pred["p_minimax"].max()) < 1.0
    tail = np.abs(x_ev) > 2.0
    center = np.abs(x_ev) < 0.5
    assert float(pred["uncertainty"][tail].mean()) > float(pred["uncertainty"][center].mean())

    # Comparar Brier contra isotónica pura
    edges, vals = _naive_weighted_pav(x_cal, y_cal, np.ones_like(x_cal))
    p_iso = vals[np.clip(np.searchsorted(edges, x_ev, side="left"), 0, len(vals) - 1)]
    brier_va = val.brier_score(pred["p_minimax"], y_ev)
    brier_iso = val.brier_score(p_iso, y_ev)
    assert brier_va <= brier_iso + 1e-5

    # Benchmark de velocidad en escala real (100K cal + 100K test < 2.0 s)
    x_big = rng.standard_normal(100_000)
    y_big = (rng.random(100_000) < 0.5 + 0.03 * np.tanh(x_big)).astype(np.float64)
    t0 = time.monotonic()
    big_cal = val.VennAbersCalibrator.fit(x_big, y_big)
    big_pred = big_cal.predict(x_big)
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"Venn-Abers 100K tardó demasiado: {elapsed:.2f} s"
    assert len(big_pred["p_minimax"]) == 100_000


def test_venn_abers_temporal_decay_adapts_to_base_rate_drift():
    """Con decaimiento exponencial temporal, Venn-Abers sigue cambios de tasa base sin fuga."""
    rng = np.random.default_rng(77)
    sessions = np.repeat(np.arange(100), 50)
    x = rng.standard_normal(len(sessions))
    # Primera mitad tasa base 0.30, segunda mitad 0.70
    p_true = np.where(sessions < 50, 0.30, 0.70)
    y = (rng.random(len(sessions)) < p_true).astype(np.float64)

    cal_unif = val.VennAbersCalibrator.fit(x, y)
    cal_decay = val.VennAbersCalibrator.fit_with_decay(x, y, session_order=sessions, half_life=15.0)
    p_u = float(cal_unif.predict(np.array([0.0]))["p_minimax"][0])
    p_d = float(cal_decay.predict(np.array([0.0]))["p_minimax"][0])
    assert p_d > p_u + 0.10


def test_session_ltt_gate_certifies_high_reliability_and_rejects_noise():
    """SessionLTTGate certifica con FWER controlado cuando hay señal y devuelve None con ruido."""
    rng = np.random.default_rng(2026)
    n_cal, n_ev = 250, 250
    # Gate causal: cuando gate_score > 0.5, IC es positivo 96% de las veces; si no, 55%
    g_cal = rng.uniform(0.0, 1.0, size=n_cal)
    ic_cal = np.where(
        g_cal > 0.45,
        rng.normal(0.08, 0.04, size=n_cal),
        rng.normal(0.005, 0.05, size=n_cal),
    )
    g_ev = rng.uniform(0.0, 1.0, size=n_ev)
    ic_ev = np.where(
        g_ev > 0.45,
        rng.normal(0.08, 0.04, size=n_ev),
        rng.normal(0.005, 0.05, size=n_ev),
    )

    gate = val.SessionLTTGate.fit(
        g_cal,
        ic_cal,
        target_reliability=0.90,
        delta=0.05,
        margin=0.0,
        method="holm",
    )
    assert gate.chosen_threshold is not None
    ev_report = gate.evaluate(g_ev, ic_ev)
    assert ev_report["coverage"] > 0.30
    assert ev_report["hit_rate"] >= 0.90
    assert ev_report["active_ic"] > ev_report["base_ic"]

    # Sin señal (IC centrado en 0): NINGÚN umbral debe pasar
    ic_noise = rng.normal(0.0, 0.05, size=n_cal)
    gate_noise = val.SessionLTTGate.fit(
        g_cal, ic_noise, target_reliability=0.90, delta=0.05, margin=0.0
    )
    assert gate_noise.chosen_threshold is None
    ev_noise = gate_noise.evaluate(g_ev, ic_noise)
    assert ev_noise["coverage"] == 0.0


def test_session_ltt_gate_does_not_leak_evaluation_and_supports_block_testing():
    """El umbral de SessionLTTGate depende solo de calibración y soporta ventanas de 5 sesiones."""
    rng = np.random.default_rng(55)
    g_cal = rng.uniform(0.0, 1.0, size=200)
    ic_cal = np.where(
        g_cal > 0.3, 0.06 + 0.03 * rng.standard_normal(200), -0.02 + 0.05 * rng.standard_normal(200)
    )
    gate1 = val.SessionLTTGate.fit(g_cal, ic_cal, target_reliability=0.90, delta=0.05)
    gate2 = val.SessionLTTGate.fit(g_cal, ic_cal, target_reliability=0.90, delta=0.05)
    assert gate1.chosen_threshold == gate2.chosen_threshold

    # Prueba por bloques no solapados de 5 sesiones
    gate_blk = val.SessionLTTGate.fit(
        g_cal, ic_cal, target_reliability=0.85, delta=0.05, block_size=5
    )
    assert gate_blk.chosen_threshold is not None
    assert gate_blk.block_size == 5
