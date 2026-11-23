"""Órganos nuevos de memoria episódica: banco denso, memoria holográfica delta y traza por activo.

Comprueba para cada órgano: cuenta independiente en casos pequeños, causalidad estricta (sin
etiquetas de la misma sesión, no maduras ni futuras y con 2024 sellado), determinismo con semilla,
placebo sin señal donde el órgano real sí la recupera, estabilidad numérica en FP32/FP64 y con
predicciones constantes, pausa/restauración exacta y capacidad de la memoria holográfica.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.memory import episodic_organs as eo
from mars_titan.memory import neighbour_residual_memory as nrm
from mars_titan.memory import view_keys as vk

DAY = 86_400_000_000
START = 1_333_310_700_000_000  # abril de 2012 en microsegundos UTC


def _average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    first = np.empty(len(values), dtype=bool)
    first[:1] = True
    first[1:] = ordered[1:] != ordered[:-1]
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = ((starts + 1 + ends) / 2.0)[np.cumsum(first) - 1]
    return ranks


def _rank_corr(left, right):
    rl = _average_ranks(left)
    rr = _average_ranks(right)
    rl -= rl.mean()
    rr -= rr.mean()
    denom = math.sqrt(float(rl @ rl) * float(rr @ rr))
    return float(rl @ rr) / denom if denom > 0 else 0.0


def _mean_session_ic(reads, target, prediction_at, *, start_session=1):
    times = np.unique(prediction_at)
    ics = []
    for t in times[start_session:]:
        rows = prediction_at == t
        ics.append(_rank_corr(reads[rows], target[rows]))
    return float(np.mean(ics))


# -------------------------------------------------------------------------------------------------
# 1. Banco denso con temperatura (media ponderada y dispersión ponderada)
# -------------------------------------------------------------------------------------------------


def test_dense_bank_matches_independent_hand_calculation_for_mean_and_dispersion():
    """Cuenta independiente a mano de media y dispersión ponderadas para dos temperaturas."""
    # Tres sesiones, 2 activos por sesión, 1 modelo; clave 2D sin normalizar para comprobar coseno.
    keys = np.array(
        [
            # Sesión 0 (madura en t=1)
            [3.0, 0.0],  # unit: [1, 0], residual = clip(0.05 - 0.01, -0.05, 0.05) = +0.04
            [
                0.0,
                4.0,
            ],  # unit: [0, 1], residual = clip(-0.10 - 0.02, -0.05, 0.05) = -0.05 (clipped)
            # Sesión 1 (madura en t=2)
            [1.0, 1.0],  # unit: [1/√2, 1/√2], residual = clip(0.01 - 0.03, -0.05, 0.05) = -0.02
            [1.0, 0.0],  # unit: [1, 0], residual = clip(0.03 - 0.01, -0.05, 0.05) = +0.02
            # Sesión 2 (consulta sobre banco de sesiones 0 y 1; k=3 de 4 filas)
            [1.0, 0.0],  # cosenos: [1.0, 0.0, 1/√2, 1.0] -> top 3 son idx 0, 3, 2
            [0.0, 0.0],  # clave nula -> cosenos [0, 0, 0, 0] -> top 3 más antiguos: idx 0, 1, 2
        ],
        dtype=np.float64,
    )
    preds = np.array([[0.01], [0.02], [0.03], [0.01], [0.0], [0.0]], dtype=np.float64)
    targets = np.array([0.05, -0.10, 0.01, 0.03, 0.0, 0.0], dtype=np.float64)
    times = np.array([START, START, START + DAY, START + DAY, START + 2 * DAY, START + 2 * DAY])
    matures = times + DAY
    markets = np.array(["US"] * 6)

    config = eo.DenseBankConfig(window=2, k=3, temperatures=(0.05, 0.2), clip=0.05)
    memory = eo.DenseBankMemory(config, models=1)
    means, disps, p_means, p_disps = eo.replay_dense_bank(
        memory,
        markets=markets,
        prediction_at=times,
        matures_at=matures,
        keys=keys,
        predictions=preds,
        target=targets,
    )
    assert p_means is None and p_disps is None

    # Sesión 0: banco vacío -> 0.0 exacto en media y dispersión.
    for tau in (0.05, 0.2):
        np.testing.assert_array_equal(means[tau][:2], np.zeros((2, 1)))
        np.testing.assert_array_equal(disps[tau][:2], np.zeros((2, 1)))

    # Sesión 2, fila 4: top 3 son (idx 0: cos=1, v=+0.04), (idx 3: cos=1, v=+0.02),
    # (idx 2: cos=1/√2, v=-0.02).
    cos_row4 = np.array([1.0, 1.0, 1.0 / math.sqrt(2.0)])
    val_row4 = np.array([0.04, 0.02, -0.02])
    # Sesión 2, fila 5 (clave nula): todos los cosenos son 0, empates van a los 3 más antiguos
    # (idx 0: +0.04, idx 1: -0.05, idx 2: -0.02) con pesos uniformes 1/3.
    val_row5 = np.array([0.04, -0.05, -0.02])
    expected_mean5 = float(np.mean(val_row5))
    expected_disp5 = math.sqrt(float(np.mean((val_row5 - expected_mean5) ** 2)))

    for tau in (0.05, 0.2):
        w = np.exp((cos_row4 - cos_row4.max()) / tau)
        w /= w.sum()
        mu4 = float(w @ val_row4)
        disp4 = math.sqrt(float(w @ ((val_row4 - mu4) ** 2)))
        assert means[tau][4, 0] == pytest.approx(mu4, abs=1e-13)
        assert disps[tau][4, 0] == pytest.approx(disp4, abs=1e-13)
        assert means[tau][5, 0] == pytest.approx(expected_mean5, abs=1e-13)
        assert disps[tau][5, 0] == pytest.approx(expected_disp5, abs=1e-13)

    # Temperatura menor concentra más peso en los dos vecinos de coseno 1 (media más positiva).
    assert means[0.05][4, 0] > means[0.2][4, 0]


def test_dense_bank_causality_blocks_same_session_immature_and_future_labels():
    """Cambiar etiquetas no maduras, de la misma sesión o futuras no cambia lecturas anteriores."""
    rng = np.random.default_rng(11)
    sessions, assets = 10, 16
    rows = sessions * assets
    raw = rng.normal(size=(rows, 8)).astype(np.float32)
    keys = nrm.keys_from_inputs(raw, nrm.projection(8, 6, seed=3))
    data = dict(
        markets=np.array(["US"] * rows),
        prediction_at=np.repeat(START + np.arange(sessions) * DAY, assets).astype(np.int64),
        matures_at=np.repeat(START + (np.arange(sessions) + 2) * DAY, assets).astype(np.int64),
        keys=keys,
        predictions=rng.normal(scale=0.01, size=(rows, 2)),
        target=rng.normal(scale=0.02, size=rows),
    )
    config = eo.DenseBankConfig(window=5, k=12, temperatures=(0.05, 0.2), clip=0.1)
    base_m, base_d, base_pm, base_pd = eo.replay_dense_bank(
        eo.DenseBankMemory(config, models=2), **data, placebo_seed=7
    )

    # Corte en la sesión 6: las etiquetas de sesiones 5 y 6 aún NO maduraron en t=6 (lag=2).
    cut_t = START + 6 * DAY
    changed = {k: np.array(v, copy=True) for k, v in data.items()}
    immature_or_future = changed["matures_at"] > cut_t
    future_rows = changed["prediction_at"] > cut_t
    changed["target"][immature_or_future] = rng.normal(
        scale=0.5, size=int(immature_or_future.sum())
    )
    changed["predictions"][future_rows] = rng.normal(scale=0.5, size=(int(future_rows.sum()), 2))
    changed["keys"][future_rows] = changed["keys"][future_rows][::-1]

    pert_m, pert_d, pert_pm, pert_pd = eo.replay_dense_bank(
        eo.DenseBankMemory(config, models=2), **changed, placebo_seed=7
    )
    early = data["prediction_at"] <= cut_t
    for tau in config.temperatures:
        np.testing.assert_array_equal(pert_m[tau][early], base_m[tau][early])
        np.testing.assert_array_equal(pert_d[tau][early], base_d[tau][early])
        np.testing.assert_array_equal(pert_pm[tau][early], base_pm[tau][early])
        np.testing.assert_array_equal(pert_pd[tau][early], base_pd[tau][early])
        assert not np.array_equal(pert_m[tau][~early], base_m[tau][~early])

    # Frontera exacta de madurez: matures_at == prediction_at sí se lee; +1 microsegundo no.
    short = {k: v[: 2 * assets].copy() for k, v in data.items()}
    short["matures_at"] = short["prediction_at"] + DAY
    m_on, d_on, _, _ = eo.replay_dense_bank(eo.DenseBankMemory(config, models=2), **short)
    assert np.any(m_on[0.05][assets:] != 0.0) and np.any(d_on[0.05][assets:] > 0.0)
    late = dict(short, matures_at=short["matures_at"] + 1)
    m_late, d_late, _, _ = eo.replay_dense_bank(eo.DenseBankMemory(config, models=2), **late)
    np.testing.assert_array_equal(m_late[0.05][assets:], np.zeros((assets, 2)))
    np.testing.assert_array_equal(d_late[0.05][assets:], np.zeros((assets, 2)))


def test_dense_bank_recovers_planted_signal_and_uncertainty_while_placebo_has_no_signal():
    """Recupera señal de residuos por vecinos e incertidumbre real; el placebo queda sin señal."""
    rng = np.random.default_rng(19)
    sessions, assets = 35, 32
    rows = sessions * assets
    # Dos ejes latentes: el primero da la dirección del residuo; el segundo controla el ruido.
    latent = rng.normal(size=rows)
    high_noise = (np.arange(rows) % 2) == 1
    keys = np.column_stack(
        [
            latent,
            np.where(high_noise, 2.0, -2.0),
            rng.normal(scale=0.15, size=(rows, 4)),
        ]
    ).astype(np.float32)
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)
    parent = rng.normal(scale=0.01, size=(rows, 1))
    noise_scale = np.where(high_noise, 0.04, 0.002)
    target = parent[:, 0] + 0.03 * np.tanh(latent) + rng.normal(scale=noise_scale)
    times = np.repeat(START + np.arange(sessions) * DAY, assets).astype(np.int64)

    config = eo.DenseBankConfig(window=20, k=32, temperatures=(0.05, 0.1, 0.2), clip=0.15)
    means, disps, p_means, _ = eo.replay_dense_bank(
        eo.DenseBankMemory(config, models=1),
        markets=np.array(["US"] * rows),
        prediction_at=times,
        matures_at=times + DAY,
        keys=keys,
        predictions=parent,
        target=target,
        placebo_seed=123,
    )
    residual = target - parent[:, 0]
    real_ic = _mean_session_ic(means[0.1][:, 0], residual, times, start_session=5)
    placebo_ic = _mean_session_ic(p_means[0.1][:, 0], residual, times, start_session=5)
    assert real_ic > 0.5
    assert abs(placebo_ic) < 0.08
    # La dispersión ponderada separa las filas ruidosas de las precisas.
    late = times >= START + 5 * DAY
    assert disps[0.1][late & high_noise, 0].mean() > 2.5 * disps[0.1][late & ~high_noise, 0].mean()


def test_dense_bank_determinism_fp32_fp64_constant_predictions_validation_and_restore():
    """Determinismo, FP32/FP64, hilos, predicción constante, límites y pausa/restauración."""
    rng = np.random.default_rng(23)
    sessions, assets = 8, 140
    rows = sessions * assets
    raw = rng.normal(size=(rows, 6)).astype(np.float32)
    keys32 = nrm.keys_from_inputs(raw, nrm.projection(6, 5, seed=9))
    data32 = dict(
        markets=np.array(["US"] * rows),
        prediction_at=np.repeat(START + np.arange(sessions) * DAY, assets).astype(np.int64),
        matures_at=np.repeat(START + (np.arange(sessions) + 1) * DAY, assets).astype(np.int64),
        keys=keys32,
        predictions=np.full((rows, 2), 0.005, dtype=np.float64),  # predicción constante
        target=rng.normal(scale=0.01, size=rows),
    )
    config = eo.DenseBankConfig(window=4, k=16, temperatures=(0.05, 0.2), clip=0.1)
    m1, d1, pm1, pd1 = eo.replay_dense_bank(
        eo.DenseBankMemory(config, models=2, threads=1), **data32, placebo_seed=5
    )
    m3, d3, pm3, pd3 = eo.replay_dense_bank(
        eo.DenseBankMemory(config, models=2, threads=3), **data32, placebo_seed=5
    )
    _, _, pm_other, _ = eo.replay_dense_bank(
        eo.DenseBankMemory(config, models=2, threads=1), **data32, placebo_seed=6
    )
    for tau in config.temperatures:
        assert np.isfinite(m1[tau]).all() and np.isfinite(d1[tau]).all() and np.all(d1[tau] >= 0.0)
        np.testing.assert_array_equal(m1[tau], m3[tau])
        np.testing.assert_array_equal(d1[tau], d3[tau])
        np.testing.assert_array_equal(pm1[tau], pm3[tau])
        np.testing.assert_array_equal(pd1[tau], pd3[tau])
        assert not np.array_equal(pm1[tau][assets:], pm_other[tau][assets:])

    # FP64 coincide con FP32 dentro del redondeo de cosenos FP32.
    data64 = dict(data32, keys=keys32.astype(np.float64))
    m64, d64, _, _ = eo.replay_dense_bank(eo.DenseBankMemory(config, models=2), **data64)
    for tau in config.temperatures:
        np.testing.assert_allclose(m64[tau], m1[tau], atol=1e-5)
        np.testing.assert_allclose(d64[tau], d1[tau], atol=1e-5)

    # Pausa y restauración a mitad de recorrido: idéntico bit a bit.
    cut = 4 * assets
    head = {k: v[:cut] for k, v in data32.items()}
    tail = {k: v[cut:] for k, v in data32.items()}
    mem = eo.DenseBankMemory(config, models=2)
    hm, hd, hpm, _ = eo.replay_dense_bank(mem, **head, placebo_seed=5, flush=False)
    restored = eo.DenseBankMemory.restore(mem.export())
    tm, td, tpm, _ = eo.replay_dense_bank(
        restored,
        **tail,
        placebo_seed=5,
        pending_targets=dict(
            markets=head["markets"], prediction_at=head["prediction_at"], target=head["target"]
        ),
    )
    for tau in config.temperatures:
        np.testing.assert_array_equal(np.concatenate([hm[tau], tm[tau]]), m1[tau])
        np.testing.assert_array_equal(np.concatenate([hd[tau], td[tau]]), d1[tau])
        np.testing.assert_array_equal(np.concatenate([hpm[tau], tpm[tau]]), pm1[tau])

    # Estado manipulado rechazado.
    tampered = mem.export()
    tampered["markets"]["US"]["bank"][0]["values"][0, 0] += 0.01
    with pytest.raises(ValueError, match="huella"):
        eo.DenseBankMemory.restore(tampered)

    # Límites de configuración y test final de 2024 sellado.
    for bad in (
        dict(window=0, k=8, temperatures=(0.1,)),
        dict(window=127, k=8, temperatures=(0.1,)),
        dict(window=5, k=0, temperatures=(0.1,)),
        dict(window=5, k=257, temperatures=(0.1,)),
        dict(window=5, k=8, temperatures=()),
        dict(window=5, k=8, temperatures=(0.2, 0.1)),
        dict(window=5, k=8, temperatures=(0.0,)),
        dict(window=5, k=8, temperatures=(0.1,), clip=-1.0),
    ):
        with pytest.raises(ValueError):
            eo.DenseBankConfig(**bad)

    data_2024 = dict(
        data32,
        prediction_at=data32["prediction_at"] + (FINAL_TEST_START_US - START),
        matures_at=data32["matures_at"] + (FINAL_TEST_START_US - START),
    )
    with pytest.raises(ValueError, match="desarrollo"):
        eo.replay_dense_bank(eo.DenseBankMemory(config, models=2), **data_2024)


# -------------------------------------------------------------------------------------------------
# 2. Memoria holográfica reducida con regla delta
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_holographic_unitary_vectors_and_fft_binding_unbinding_are_exact(dtype):
    """Los vectores base son unitarios en frecuencia y correlación(k, conv(k, v)) == v."""
    dim = 1024
    tol = 1e-5 if dtype == torch.float32 else 1e-13
    a = eo.unitary_base_vector(dim, 42, "asset", "US/A001", dtype=dtype)
    r = eo.unitary_base_vector(dim, 42, "regime", "calm_up", dtype=dtype)
    v = eo.unitary_base_vector(dim, 42, "value", "residual", dtype=dtype)
    assert a.dtype == dtype and a.shape == (dim,)
    # Módulo unitario en cada frecuencia y norma L2 unitaria.
    for vec in (a, r, v):
        spectrum = torch.fft.rfft(vec, n=dim)
        torch.testing.assert_close(
            spectrum.abs(), torch.ones_like(spectrum.real), rtol=0.0, atol=tol
        )
        assert float(torch.linalg.vector_norm(vec)) == pytest.approx(1.0, abs=tol)
    # Composición por convolución circular conserva unitariedad.
    key = eo.circular_convolution(a, r)
    key_spec = torch.fft.rfft(key, n=dim)
    torch.testing.assert_close(key_spec.abs(), torch.ones_like(key_spec.real), rtol=0.0, atol=tol)
    # Recuperación exacta por correlación circular: (k ★ (k ⊛ v)) == v.
    bound = eo.circular_convolution(key, v)
    unbound = eo.circular_correlation(key, bound)
    torch.testing.assert_close(unbound, v, rtol=0.0, atol=tol)
    assert float(torch.dot(unbound, v)) == pytest.approx(1.0, abs=tol)
    # Determinismo con semilla: misma semilla da igualdad exacta; distinta da vector casi ortogonal.
    torch.testing.assert_close(a, eo.unitary_base_vector(dim, 42, "asset", "US/A001", dtype=dtype))
    other = eo.unitary_base_vector(dim, 43, "asset", "US/A001", dtype=dtype)
    assert abs(float(torch.dot(a, other))) < 0.15


def test_holographic_delta_matches_independent_hand_calculation_with_decay_and_delta_rule():
    """Verifica a mano superposición, regla delta (sin reescribir lo ya predicho) y decaimiento."""
    dim = 1024
    # Caso A: 1 par escrito dos veces con decay=1.0, rate=1.0 -> la segunda escritura tiene delta=0.
    cfg_nod = eo.HolographicDeltaConfig(
        dim=dim, decay=1.0, rate=1.0, seed=7, entity="asset", clip=1.0, dtype="float64"
    )
    mem = eo.HolographicDeltaMemory(cfg_nod, models=1)
    r0, _ = mem.emit(
        "US",
        START,
        np.array(["US/A"]),
        np.array(["calm_up"]),
        np.array([[0.01]]),
        matures_at=START + DAY,
    )
    assert r0[0, 0] == 0.0
    mem.mature("US", START, np.array([0.06]), at=START + DAY)  # residuo = +0.05
    r1, _ = mem.emit(
        "US",
        START + DAY,
        np.array(["US/A"]),
        np.array(["calm_up"]),
        np.array([[0.01]]),
        matures_at=START + 2 * DAY,
    )
    assert r1[0, 0] == pytest.approx(0.05, abs=1e-13)
    norm_after_first = float(torch.linalg.vector_norm(mem._markets["US"].memory))
    # Escribir el mismo residuo +0.05 otra vez: error = 0.05 - 0.05 = 0 -> la memoria no duplica.
    mem.mature("US", START + DAY, np.array([0.06]), at=START + 2 * DAY)
    r2, _ = mem.emit(
        "US",
        START + 2 * DAY,
        np.array(["US/A"]),
        np.array(["calm_up"]),
        np.array([[0.0]]),
        matures_at=START + 3 * DAY,
    )
    assert r2[0, 0] == pytest.approx(0.05, abs=1e-13)
    assert float(torch.linalg.vector_norm(mem._markets["US"].memory)) == pytest.approx(
        norm_after_first, abs=1e-13
    )
    # Sobrescribir el mismo par con residuo -0.03 -> lectura exacta -0.03.
    mem.mature("US", START + 2 * DAY, np.array([-0.03]), at=START + 3 * DAY)
    r3, _ = mem.emit(
        "US",
        START + 3 * DAY,
        np.array(["US/A"]),
        np.array(["calm_up"]),
        np.array([[0.0]]),
        matures_at=START + 4 * DAY,
    )
    assert r3[0, 0] == pytest.approx(-0.03, abs=1e-13)
    # Residuo extremo (+5.0) con clip=1.0 debe recortarse exactamente a +1.0.
    mem.mature("US", START + 3 * DAY, np.array([5.0]), at=START + 4 * DAY)
    r4, _ = mem.emit(
        "US",
        START + 4 * DAY,
        np.array(["US/A"]),
        np.array(["calm_up"]),
        np.array([[0.0]]),
        matures_at=START + 5 * DAY,
    )
    assert r4[0, 0] == pytest.approx(1.0, abs=1e-13)

    # Caso B: cuenta independiente paso a paso con decay=0.8 y rate=0.5 y 2 activos en una sesión.
    cfg = eo.HolographicDeltaConfig(
        dim=dim, decay=0.8, rate=0.5, seed=11, entity="asset", clip=0.5, dtype="float64"
    )
    mem_b = eo.HolographicDeltaMemory(cfg, models=1)
    assets = np.array(["US/A", "US/B"])
    regimes = np.array(["calm_up", "calm_up"])
    preds = np.array([[0.0], [0.0]])
    y0 = np.array([0.4, -0.2])
    y1 = np.array([0.2, 0.3])
    mem_b.emit("US", START, assets, regimes, preds, matures_at=START + DAY)
    mem_b.mature("US", START, y0, at=START + DAY)
    read1, _ = mem_b.emit("US", START + DAY, assets, regimes, preds, matures_at=START + 2 * DAY)
    mem_b.mature("US", START + DAY, y1, at=START + 2 * DAY)
    read2, _ = mem_b.emit("US", START + 2 * DAY, assets, regimes, preds, matures_at=START + 3 * DAY)

    # Referencia independiente construida a mano con rfft/irfft.
    u_val = eo.unitary_base_vector(dim, 11, "value", "residual", dtype=torch.float64)
    r_vec = eo.unitary_base_vector(dim, 11, "regime", "calm_up", dtype=torch.float64)
    a_vecs = [
        eo.unitary_base_vector(dim, 11, "asset", name, dtype=torch.float64) for name in assets
    ]
    keys_ref = [eo.circular_convolution(av, r_vec) for av in a_vecs]
    bound_ref = [eo.circular_convolution(kv, u_val) for kv in keys_ref]
    m_ref = torch.zeros(dim, dtype=torch.float64)
    for y_step, actual in ((y0, read1), (y1, read2)):
        m_ref = m_ref * 0.8
        for idx in range(2):
            pred = float(torch.dot(eo.circular_correlation(keys_ref[idx], m_ref), u_val))
            m_ref = m_ref + 0.5 * (float(y_step[idx]) - pred) * bound_ref[idx]
        expected = np.array(
            [float(torch.dot(eo.circular_correlation(kv, m_ref), u_val)) for kv in keys_ref]
        )
        np.testing.assert_allclose(actual[:, 0], expected, atol=1e-13)


def test_holographic_delta_separates_regimes_and_supports_sector_keys_order_invariantly():
    """El mismo activo en regímenes distintos no colisiona; modo sector promedia por sector."""
    dim = 1024
    cfg_asset = eo.HolographicDeltaConfig(
        dim=dim, decay=1.0, rate=1.0, seed=13, entity="asset", clip=1.0
    )
    mem = eo.HolographicDeltaMemory(cfg_asset, models=1)
    # US/A en calm_up tiene residuo +0.08; en turbulent_down tiene residuo -0.06.
    mem.emit(
        "US",
        START,
        np.array(["US/A"]),
        np.array(["calm_up"]),
        np.array([[0.0]]),
        matures_at=START + DAY,
    )
    mem.mature("US", START, np.array([0.08]), at=START + DAY)
    mem.emit(
        "US",
        START + DAY,
        np.array(["US/A"]),
        np.array(["turbulent_down"]),
        np.array([[0.0]]),
        matures_at=START + 2 * DAY,
    )
    mem.mature("US", START + DAY, np.array([-0.06]), at=START + 2 * DAY)
    out, _ = mem.emit(
        "US",
        START + 2 * DAY,
        np.array(["US/A", "US/A"]),
        np.array(["calm_up", "turbulent_down"]),
        np.array([[0.0], [0.0]]),
        matures_at=START + 3 * DAY,
    )
    # Con 2 pares en dim=1024, el ruido de interferencia es ~1/√1024 ≈ 0.03 * 0.08 ≈ 0.002.
    assert abs(out[0, 0] - 0.08) < 0.01
    assert abs(out[1, 0] - (-0.06)) < 0.01

    # Modo sector: varios activos comparten sector; el resultado no depende del orden de las filas.
    cfg_sec = eo.HolographicDeltaConfig(
        dim=dim, decay=1.0, rate=1.0, seed=17, entity="sector", clip=1.0
    )
    sectors = np.array(["US/Tech", "US/Tech", "US/Health", "US/Health"])
    regimes = np.array(["calm_up"] * 4)
    preds = np.zeros((4, 1))
    y = np.array([0.10, 0.02, -0.04, -0.08])  # medias de sector: Tech=+0.06, Health=-0.06
    m1 = eo.HolographicDeltaMemory(cfg_sec, models=1)
    m1.emit("US", START, sectors, regimes, preds, matures_at=START + DAY)
    m1.mature("US", START, y, at=START + DAY)
    q1, _ = m1.emit(
        "US",
        START + DAY,
        np.array(["US/Tech", "US/Health"]),
        np.array(["calm_up", "calm_up"]),
        np.zeros((2, 1)),
        matures_at=START + 2 * DAY,
    )
    # Permutando las filas de entrada en la sesión 0 debe salir exactamente lo mismo.
    order = np.array([2, 0, 3, 1])
    m2 = eo.HolographicDeltaMemory(cfg_sec, models=1)
    m2.emit("US", START, sectors[order], regimes[order], preds[order], matures_at=START + DAY)
    m2.mature("US", START, y[order], at=START + DAY)
    q2, _ = m2.emit(
        "US",
        START + DAY,
        np.array(["US/Tech", "US/Health"]),
        np.array(["calm_up", "calm_up"]),
        np.zeros((2, 1)),
        matures_at=START + 2 * DAY,
    )
    np.testing.assert_allclose(q1, q2, atol=1e-14)
    assert abs(q1[0, 0] - 0.06) < 0.01
    assert abs(q1[1, 0] - (-0.06)) < 0.01


def test_holographic_delta_causality_blocks_same_session_immature_and_future_labels():
    """Causalidad estricta en lectura real y placebo de la memoria holográfica."""
    rng = np.random.default_rng(31)
    sessions, assets = 10, 12
    rows = sessions * assets
    names = np.tile([f"US/A{i:02d}" for i in range(assets)], sessions)
    regime_cycle = ["calm_up", "turbulent_down", "calm_down", "turbulent_up"]
    regimes = np.repeat([regime_cycle[s % 4] for s in range(sessions)], assets)
    data = dict(
        markets=np.array(["US"] * rows),
        prediction_at=np.repeat(START + np.arange(sessions) * DAY, assets).astype(np.int64),
        matures_at=np.repeat(START + (np.arange(sessions) + 2) * DAY, assets).astype(np.int64),
        entities=names,
        regimes=regimes,
        predictions=rng.normal(scale=0.01, size=(rows, 2)),
        target=rng.normal(scale=0.02, size=rows),
    )
    cfg = eo.HolographicDeltaConfig(
        dim=1024, decay=0.95, rate=0.5, seed=5, entity="asset", clip=0.1
    )
    base, base_p = eo.replay_holographic_delta(
        eo.HolographicDeltaMemory(cfg, models=2, placebo_seed=11), **data
    )
    cut_t = START + 6 * DAY
    changed = {k: np.array(v, copy=True) for k, v in data.items()}
    immature_or_future = changed["matures_at"] > cut_t
    future_rows = changed["prediction_at"] > cut_t
    changed["target"][immature_or_future] = rng.normal(
        scale=0.5, size=int(immature_or_future.sum())
    )
    changed["predictions"][future_rows] = rng.normal(scale=0.5, size=(int(future_rows.sum()), 2))
    pert, pert_p = eo.replay_holographic_delta(
        eo.HolographicDeltaMemory(cfg, models=2, placebo_seed=11), **changed
    )
    early = data["prediction_at"] <= cut_t
    np.testing.assert_array_equal(pert[early], base[early])
    np.testing.assert_array_equal(pert_p[early], base_p[early])
    assert not np.array_equal(pert[~early], base[~early])

    # Frontera exacta: matures_at == now se lee; matures_at == now + 1 no.
    short = {k: v[: 2 * assets].copy() for k, v in data.items()}
    short["matures_at"] = short["prediction_at"] + DAY
    short["regimes"] = np.full(2 * assets, "calm_up")
    r_on, _ = eo.replay_holographic_delta(eo.HolographicDeltaMemory(cfg, models=2), **short)
    assert np.any(r_on[assets:] != 0.0)
    r_late, _ = eo.replay_holographic_delta(
        eo.HolographicDeltaMemory(cfg, models=2), **dict(short, matures_at=short["matures_at"] + 1)
    )
    np.testing.assert_array_equal(r_late[assets:], np.zeros((assets, 2)))


@pytest.mark.parametrize("entity_mode", ["asset", "sector"])
def test_holographic_delta_recovers_regime_conditional_signal_while_placebo_has_no_signal(
    entity_mode,
):
    """El órgano real recupera residuos dependientes del régimen (activo y sector); placebo ~ 0."""
    rng = np.random.default_rng(37 if entity_mode == "asset" else 41)
    sessions, assets = 60, 30
    rows = sessions * assets
    if entity_mode == "asset":
        entities = np.tile([f"US/A{i:02d}" for i in range(assets)], sessions)
        group_ids = np.tile(np.arange(assets), sessions)
        n_groups = assets
    else:
        n_groups = 10
        sec_names = [f"US/Sec{i % n_groups}" for i in range(assets)]
        entities = np.tile(sec_names, sessions)
        group_ids = np.tile(np.arange(assets) % n_groups, sessions)
    # Dos regímenes alternos con signo opuesto del residuo por entidad: una traza sin régimen
    # vería media ~0, pero la clave compuesta entidad ⊛ régimen lo recupera con claridad.
    session_regimes = np.array(
        ["calm_up" if s % 2 == 0 else "turbulent_down" for s in range(sessions)]
    )
    regimes = np.repeat(session_regimes, assets)
    sign_reg = np.where(regimes == "calm_up", 1.0, -1.0)
    base_effect = rng.normal(scale=0.04, size=n_groups)
    parent = rng.normal(scale=0.01, size=(rows, 1))
    target = parent[:, 0] + sign_reg * base_effect[group_ids] + rng.normal(scale=0.008, size=rows)
    times = np.repeat(START + np.arange(sessions) * DAY, assets).astype(np.int64)

    cfg = eo.HolographicDeltaConfig(
        dim=1024, decay=0.95, rate=0.5, seed=29, entity=entity_mode, clip=0.2
    )
    residual = target - parent[:, 0]
    placebo_ics = []
    real_ic = None
    for pseed in (101, 102, 103, 104, 105):
        reads, placebo = eo.replay_holographic_delta(
            eo.HolographicDeltaMemory(cfg, models=1, placebo_seed=pseed),
            markets=np.array(["US"] * rows),
            prediction_at=times,
            matures_at=times + DAY,
            entities=entities,
            regimes=regimes,
            predictions=parent,
            target=target,
        )
        if real_ic is None:
            real_ic = _mean_session_ic(reads[:, 0], residual, times, start_session=6)
        placebo_ics.append(_mean_session_ic(placebo[:, 0], residual, times, start_session=6))
    assert real_ic > 0.70
    assert abs(float(np.mean(placebo_ics))) < 0.08
    assert max(abs(v) for v in placebo_ics) < 0.20


def test_holographic_delta_capacity_degrades_when_superposing_more_pairs_than_dimension():
    """Al superponer más pares que la dimensión D, la correlación cae y el MSE crece."""
    rng = np.random.default_rng(53)

    def measure(dim, pairs):
        cfg = eo.HolographicDeltaConfig(
            dim=dim, decay=1.0, rate=1.0, seed=19, entity="asset", clip=10.0, dtype="float64"
        )
        mem = eo.HolographicDeltaMemory(cfg, models=1)
        entities = np.array([f"US/P{i:04d}" for i in range(pairs)])
        regimes = np.array([f"reg_{i % 4}" for i in range(pairs)])
        values = rng.normal(scale=1.0, size=pairs)
        mem.emit("US", START, entities, regimes, np.zeros((pairs, 1)), matures_at=START + DAY)
        mem.mature("US", START, values, at=START + DAY)
        recovered, _ = mem.emit(
            "US", START + DAY, entities, regimes, np.zeros((pairs, 1)), matures_at=START + 2 * DAY
        )
        rec = recovered[:, 0]
        corr = float(np.corrcoef(values, rec)[0, 1])
        mse = float(np.mean((rec - values) ** 2))
        return corr, mse

    corr_low, mse_low = measure(1024, 64)
    corr_mid, mse_mid = measure(1024, 512)
    corr_over, mse_over = measure(1024, 2048)
    assert corr_low > 0.90
    assert corr_low > corr_mid > corr_over
    assert mse_low < mse_mid < mse_over
    assert corr_over < 0.55

    # Con carga fija P=1024, mayor dimensión (4096 frente a 1024) mejora la recuperación.
    corr_1024, mse_1024 = measure(1024, 1024)
    corr_4096, mse_4096 = measure(4096, 1024)
    assert corr_4096 > corr_1024 + 0.15
    assert mse_4096 < mse_1024 * 0.5


def test_holographic_delta_determinism_fp32_autocast_constant_predictions_validation_and_restore():
    """FP32/FP64, sin autocast, predicciones constantes, pausa/restauración y validaciones."""
    rng = np.random.default_rng(59)
    sessions, assets = 6, 10
    rows = sessions * assets
    data = dict(
        markets=np.array(["US"] * rows),
        prediction_at=np.repeat(START + np.arange(sessions) * DAY, assets).astype(np.int64),
        matures_at=np.repeat(START + (np.arange(sessions) + 1) * DAY, assets).astype(np.int64),
        entities=np.tile([f"US/A{i}" for i in range(assets)], sessions),
        regimes=np.repeat(["calm_up", "turbulent_down"] * 3, assets),
        predictions=np.full((rows, 2), 0.01, dtype=np.float64),  # predicción constante
        target=rng.normal(scale=0.02, size=rows),
    )
    cfg64 = eo.HolographicDeltaConfig(dim=1024, decay=0.9, rate=0.5, seed=7, dtype="float64")
    cfg32 = eo.HolographicDeltaConfig(dim=1024, decay=0.9, rate=0.5, seed=7, dtype="float32")
    r64, p64 = eo.replay_holographic_delta(
        eo.HolographicDeltaMemory(cfg64, models=2, placebo_seed=9), **data
    )
    r64_again, p64_again = eo.replay_holographic_delta(
        eo.HolographicDeltaMemory(cfg64, models=2, placebo_seed=9), **data
    )
    r32, p32 = eo.replay_holographic_delta(
        eo.HolographicDeltaMemory(cfg32, models=2, placebo_seed=9), **data
    )
    assert np.isfinite(r64).all() and np.isfinite(p64).all()
    assert np.isfinite(r32).all() and np.isfinite(p32).all()
    np.testing.assert_array_equal(r64, r64_again)
    np.testing.assert_array_equal(p64, p64_again)
    np.testing.assert_allclose(r32, r64, atol=1e-5)
    np.testing.assert_allclose(p32, p64, atol=1e-5)

    # Pausa y restauración a mitad de recorrido: real y placebo bit a bit idénticos.
    cut = 3 * assets
    head = {k: v[:cut] for k, v in data.items()}
    tail = {k: v[cut:] for k, v in data.items()}
    mem = eo.HolographicDeltaMemory(cfg64, models=2, placebo_seed=9)
    hr, hp = eo.replay_holographic_delta(mem, **head, flush=False)
    restored = eo.HolographicDeltaMemory.restore(mem.export())
    tr, tp = eo.replay_holographic_delta(
        restored,
        **tail,
        pending_targets=dict(
            markets=head["markets"], prediction_at=head["prediction_at"], target=head["target"]
        ),
    )
    np.testing.assert_array_equal(np.concatenate([hr, tr]), r64)
    np.testing.assert_array_equal(np.concatenate([hp, tp]), p64)

    # Estado manipulado rechazado.
    tampered = mem.export()
    tampered["markets"]["US"]["memory"][0, 0] += 0.1
    with pytest.raises(ValueError, match="huella"):
        eo.HolographicDeltaMemory.restore(tampered)

    # Autocast rechazado explícitamente.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            eo.unitary_base_vector(1024, 0, "asset", "US/A")

    # Configuraciones fuera de rango rechazadas.
    for bad in (
        dict(dim=512, decay=0.9, seed=0),
        dict(dim=4097, decay=0.9, seed=0),
        dict(dim=1024, decay=0.0, seed=0),
        dict(dim=1024, decay=1.01, seed=0),
        dict(dim=1024, decay=0.9, seed=-1),
        dict(dim=1024, decay=0.9, seed=0, entity="country"),
        dict(dim=1024, decay=0.9, seed=0, rate=0.0),
        dict(dim=1024, decay=0.9, seed=0, rate=1.5),
        dict(dim=1024, decay=0.9, seed=0, clip=-0.1),
        dict(dim=1024, decay=0.9, seed=0, dtype="float16"),
    ):
        with pytest.raises(ValueError):
            eo.HolographicDeltaConfig(**bad)

    # 2024 sellado.
    with pytest.raises(ValueError, match="desarrollo"):
        eo.HolographicDeltaMemory(cfg64, models=1).emit(
            "US",
            FINAL_TEST_START_US,
            np.array(["US/A"]),
            np.array(["calm_up"]),
            np.zeros((1, 1)),
            matures_at=FINAL_TEST_START_US + DAY,
        )


# -------------------------------------------------------------------------------------------------
# 3. Traza por activo (media exponencial de residuos maduros del padre)
# -------------------------------------------------------------------------------------------------


def test_asset_trace_matches_independent_hand_calculation_with_half_life_clip_and_gaps():
    """Cuenta independiente a mano de EWMA por activo con semivida, recorte y huecos."""
    h = 2.0
    d = 0.5 ** (1.0 / h)
    cfg = eo.AssetTraceConfig(half_life=h, clip=0.05)
    mem = eo.AssetResidualTrace(cfg, models=1)

    # Sesión 0: US/A (residuo clip(0.08-0.01)=+0.05), US/B (residuo -0.02).
    r0, _ = mem.emit(
        "US", START, np.array(["US/A", "US/B"]), np.array([[0.01], [0.0]]), matures_at=START + DAY
    )
    np.testing.assert_array_equal(r0, np.zeros((2, 1)))
    mem.mature("US", START, np.array([0.08, -0.02]), at=START + DAY)

    # Sesión 1: solo US/B (US/A ausente); lectura de US/B es -0.02; residuo nuevo +0.04.
    r1, _ = mem.emit(
        "US", START + DAY, np.array(["US/B"]), np.array([[0.01]]), matures_at=START + 2 * DAY
    )
    assert r1[0, 0] == pytest.approx(-0.02, abs=1e-15)
    mem.mature("US", START + DAY, np.array([0.05]), at=START + 2 * DAY)  # residuo +0.04

    # Sesión 2: US/A vuelve (2 sesiones después de su última escritura), US/B presente, US/C nuevo.
    r2, _ = mem.emit(
        "US",
        START + 2 * DAY,
        np.array(["US/A", "US/B", "US/C"]),
        np.zeros((3, 1)),
        matures_at=START + 3 * DAY,
    )
    expected_b2 = (d * (-0.02) + 0.04) / (d + 1.0)
    assert r2[0, 0] == pytest.approx(0.05, abs=1e-15)  # US/A conserva su media durante su ausencia
    assert r2[1, 0] == pytest.approx(expected_b2, abs=1e-15)
    assert r2[2, 0] == 0.0  # US/C sin historia lee cero
    mem.mature("US", START + 2 * DAY, np.array([-0.03, 0.01, 0.02]), at=START + 3 * DAY)

    # Sesión 3: US/A tuvo delta=2 sesiones entre sesión 0 y sesión 2 -> peso viejo d^2.
    r3, _ = mem.emit(
        "US",
        START + 3 * DAY,
        np.array(["US/A", "US/B", "US/C"]),
        np.zeros((3, 1)),
        matures_at=START + 4 * DAY,
    )
    expected_a3 = ((d**2) * 0.05 + (-0.03)) / ((d**2) + 1.0)
    expected_b3 = ((d**2) * (-0.02) + d * 0.04 + 0.01) / (d**2 + d + 1.0)
    assert r3[0, 0] == pytest.approx(expected_a3, abs=1e-15)
    assert r3[1, 0] == pytest.approx(expected_b3, abs=1e-15)
    assert r3[2, 0] == pytest.approx(0.02, abs=1e-15)

    # Ausencia prolongada (1200 sesiones con h=1, donde 0.5**1200 == 0.0 en FP64): no subdesborda.
    mem_long = eo.AssetResidualTrace(eo.AssetTraceConfig(half_life=1.0), models=1)
    mem_long.emit("US", START, np.array(["US/OLD"]), np.zeros((1, 1)), matures_at=START + DAY)
    mem_long.mature("US", START, np.array([0.07]), at=START + DAY)
    for s in range(1, 1201):
        at = START + 2 * s * DAY
        mem_long.emit("US", at, np.array(["US/OTHER"]), np.zeros((1, 1)), matures_at=at + DAY)
        mem_long.mature("US", at, np.array([0.01]), at=at + DAY)
    mean_old, weight_old = mem_long.read("US", np.array(["US/OLD"]))
    assert weight_old[0] > 0.0 and mean_old[0, 0] == pytest.approx(0.07, abs=1e-15)
    # Al regresar tras 1200 semividas, el peso viejo decae a 0 y la media pasa al nuevo residuo.
    at_back = START + 2 * 1202 * DAY
    mem_long.emit("US", at_back, np.array(["US/OLD"]), np.zeros((1, 1)), matures_at=at_back + DAY)
    mem_long.mature("US", at_back, np.array([-0.04]), at=at_back + DAY)
    mean_back, weight_back = mem_long.read("US", np.array(["US/OLD"]))
    assert weight_back[0] == pytest.approx(1.0, abs=1e-15)
    assert mean_back[0, 0] == pytest.approx(-0.04, abs=1e-15)


def test_asset_trace_causality_blocks_same_session_immature_future_and_other_markets():
    """Causalidad estricta, frontera de madurez y aislamiento entre mercados."""
    rng = np.random.default_rng(61)
    sessions, assets = 10, 12
    rows = sessions * assets
    data = dict(
        markets=np.array(["US"] * rows),
        assets=np.tile([f"US/A{i:02d}" for i in range(assets)], sessions),
        prediction_at=np.repeat(START + np.arange(sessions) * DAY, assets).astype(np.int64),
        matures_at=np.repeat(START + (np.arange(sessions) + 2) * DAY, assets).astype(np.int64),
        predictions=rng.normal(scale=0.01, size=(rows, 2)),
        target=rng.normal(scale=0.02, size=rows),
    )
    cfg = eo.AssetTraceConfig(half_life=5.0, clip=0.1)
    base, base_p = eo.replay_asset_trace(
        eo.AssetResidualTrace(cfg, models=2), **data, placebo_seed=19
    )
    cut_t = START + 6 * DAY
    changed = {k: np.array(v, copy=True) for k, v in data.items()}
    immature_or_future = changed["matures_at"] > cut_t
    future_rows = changed["prediction_at"] > cut_t
    changed["target"][immature_or_future] = rng.normal(
        scale=0.5, size=int(immature_or_future.sum())
    )
    changed["predictions"][future_rows] = rng.normal(scale=0.5, size=(int(future_rows.sum()), 2))
    pert, pert_p = eo.replay_asset_trace(
        eo.AssetResidualTrace(cfg, models=2), **changed, placebo_seed=19
    )
    early = data["prediction_at"] <= cut_t
    np.testing.assert_array_equal(pert[early], base[early])
    np.testing.assert_array_equal(pert_p[early], base_p[early])
    assert not np.array_equal(pert[~early], base[~early])

    # Frontera exacta: matures_at == prediction_at sí se lee; +1 microsegundo no.
    short = {k: v[: 2 * assets].copy() for k, v in data.items()}
    short["matures_at"] = short["prediction_at"] + DAY
    r_on, _ = eo.replay_asset_trace(eo.AssetResidualTrace(cfg, models=2), **short)
    assert np.any(r_on[assets:] != 0.0)
    r_late, _ = eo.replay_asset_trace(
        eo.AssetResidualTrace(cfg, models=2), **dict(short, matures_at=short["matures_at"] + 1)
    )
    np.testing.assert_array_equal(r_late[assets:], np.zeros((assets, 2)))

    # Aislamiento entre mercados: añadir CN no cambia ni un bit de US.
    cn = dict(
        markets=np.array(["CN"] * rows),
        assets=np.tile([f"CN/C{i:02d}" for i in range(assets)], sessions),
        prediction_at=data["prediction_at"],
        matures_at=data["matures_at"],
        predictions=rng.normal(scale=0.05, size=(rows, 2)),
        target=rng.normal(scale=0.05, size=rows),
    )
    joint = {k: np.concatenate([data[k], cn[k]]) for k in data}
    joint_reads, joint_p = eo.replay_asset_trace(
        eo.AssetResidualTrace(cfg, models=2), **joint, placebo_seed=19
    )
    np.testing.assert_array_equal(joint_reads[:rows], base)
    np.testing.assert_array_equal(joint_p[:rows], base_p)


def test_asset_trace_recovers_planted_asset_residual_while_placebo_has_no_signal():
    """Traza real recupera residuo persistente por activo; su placebo queda sin señal."""
    rng = np.random.default_rng(67)
    sessions, assets = 35, 32
    rows = sessions * assets
    persistent = rng.normal(scale=0.03, size=assets)
    codes = np.tile(np.arange(assets), sessions)
    parent = rng.normal(scale=0.01, size=(rows, 1))
    target = parent[:, 0] + persistent[codes] + rng.normal(scale=0.015, size=rows)
    times = np.repeat(START + np.arange(sessions) * DAY, assets).astype(np.int64)
    reads, placebo = eo.replay_asset_trace(
        eo.AssetResidualTrace(eo.AssetTraceConfig(half_life=10.0, clip=0.15), models=1),
        markets=np.array(["US"] * rows),
        assets=np.tile([f"US/A{i:02d}" for i in range(assets)], sessions),
        prediction_at=times,
        matures_at=times + DAY,
        predictions=parent,
        target=target,
        placebo_seed=77,
    )
    residual = target - parent[:, 0]
    real_ic = _mean_session_ic(reads[:, 0], residual, times, start_session=5)
    placebo_ic = _mean_session_ic(placebo[:, 0], residual, times, start_session=5)
    assert real_ic > 0.70
    assert abs(placebo_ic) < 0.08


def test_asset_trace_determinism_constant_predictions_validation_and_restore():
    """Determinismo, predicción constante, pausa/restauración, validación y 2024 sellado."""
    rng = np.random.default_rng(71)
    sessions, assets = 8, 12
    rows = sessions * assets
    data = dict(
        markets=np.array(["US"] * rows),
        assets=np.tile([f"US/A{i:02d}" for i in range(assets)], sessions),
        prediction_at=np.repeat(START + np.arange(sessions) * DAY, assets).astype(np.int64),
        matures_at=np.repeat(START + (np.arange(sessions) + 1) * DAY, assets).astype(np.int64),
        predictions=np.full((rows, 2), -0.002, dtype=np.float64),
        target=rng.normal(scale=0.02, size=rows),
    )
    cfg = eo.AssetTraceConfig(half_life=7.0, clip=0.1)
    r1, p1 = eo.replay_asset_trace(eo.AssetResidualTrace(cfg, models=2), **data, placebo_seed=11)
    r2, p2 = eo.replay_asset_trace(eo.AssetResidualTrace(cfg, models=2), **data, placebo_seed=11)
    _, p_other = eo.replay_asset_trace(
        eo.AssetResidualTrace(cfg, models=2), **data, placebo_seed=12
    )
    assert np.isfinite(r1).all() and np.isfinite(p1).all()
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(p1, p2)
    assert not np.array_equal(p1[assets:], p_other[assets:])

    # Pausa y restauración a mitad de recorrido.
    cut = 4 * assets
    head = {k: v[:cut] for k, v in data.items()}
    tail = {k: v[cut:] for k, v in data.items()}
    mem = eo.AssetResidualTrace(cfg, models=2)
    hr, hp = eo.replay_asset_trace(mem, **head, placebo_seed=11, flush=False)
    restored = eo.AssetResidualTrace.restore(mem.export())
    tr, tp = eo.replay_asset_trace(
        restored,
        **tail,
        placebo_seed=11,
        pending_targets=dict(
            markets=head["markets"],
            assets=head["assets"],
            prediction_at=head["prediction_at"],
            target=head["target"],
        ),
    )
    np.testing.assert_array_equal(np.concatenate([hr, tr]), r1)
    np.testing.assert_array_equal(np.concatenate([hp, tp]), p1)

    tampered = copy.deepcopy(mem.export())
    tampered["markets"]["US"]["total"][0, 0] += 1.0
    with pytest.raises(ValueError, match="huella"):
        eo.AssetResidualTrace.restore(tampered)

    for bad in (
        dict(half_life=0.0),
        dict(half_life=-3.0),
        dict(half_life=float("nan")),
        dict(half_life=5.0, clip=0.0),
        dict(half_life=5.0, clip=-1.0),
    ):
        with pytest.raises(ValueError):
            eo.AssetTraceConfig(**bad)

    with pytest.raises(ValueError, match="desarrollo"):
        eo.AssetResidualTrace(cfg, models=1).emit(
            "US",
            FINAL_TEST_START_US,
            np.array(["US/A"]),
            np.zeros((1, 1)),
            matures_at=FINAL_TEST_START_US + DAY,
        )


# -------------------------------------------------------------------------------------------------
# 4. Regímenes causales a partir de las columnas escaladas de view_keys
# -------------------------------------------------------------------------------------------------


def test_session_regimes_classify_four_quadrants_order_invariantly_and_causally():
    """Clasifica los cuatro regímenes (tendencia × volatilidad) sin depender de orden ni futuro."""
    cols = len(vk.COLUMNS)
    idx_ret = vk.COLUMNS.index("return_63")
    idx_vol = vk.COLUMNS.index("volatility_21_over_252")
    raw = np.zeros((14, cols), dtype=np.float32)
    # 4 sesiones de 3 activos + 1 sesión pequeña de 2 activos (< min_assets=3 -> unclassified)
    specs = [
        (0, 3, +0.5, -0.4, "calm_up"),
        (3, 6, -0.3, -0.1, "calm_down"),
        (6, 9, +0.2, +0.8, "turbulent_up"),
        (9, 12, -0.7, +0.6, "turbulent_down"),
        (12, 14, +0.5, +0.5, "unclassified"),
    ]
    slices = []
    for i, (start, end, ret, vol, _) in enumerate(specs):
        raw[start:end, idx_ret] = ret
        raw[start:end, idx_vol] = vol
        slices.append(("evaluation", "US", START + i * DAY, start, end))
    regimes = eo.session_regimes_from_inputs(raw, slices, min_assets=3)
    for start, end, _, _, label in specs:
        assert list(regimes[start:end]) == [label] * (end - start)

    # Cambiar filas de sesiones posteriores no cambia el régimen de sesiones anteriores.
    changed = raw.copy()
    changed[6:, :] = -9.0
    after = eo.session_regimes_from_inputs(changed, slices, min_assets=3)
    np.testing.assert_array_equal(after[:6], regimes[:6])

    # Un particionado incompleto de filas debe rechazarse en lugar de dejar etiquetas sin escribir.
    with pytest.raises(ValueError, match="filas"):
        eo.session_regimes_from_inputs(raw, slices[:-1], min_assets=3)


def test_dense_bank_evicts_sessions_older_than_window_and_isolates_markets():
    """El banco denso expulsa las sesiones fuera de `window` y aísla cada mercado."""
    # window=1: al consultar en sesión 2, solo existe sesión 1 en el banco (sesión 0 expulsada).
    keys = np.array(
        [
            [1.0, 0.0],  # sesión 0: residuo +0.09 (debe desaparecer al entrar sesión 1)
            [0.0, 1.0],  # sesión 1: residuo -0.03
            [1.0, 0.0],  # sesión 2: consulta con clave [1, 0] -> solo puede leer sesión 1 (-0.03)
        ],
        dtype=np.float64,
    )
    preds = np.zeros((3, 1), dtype=np.float64)
    targets = np.array([0.09, -0.03, 0.0], dtype=np.float64)
    times = np.array([START, START + DAY, START + 2 * DAY], dtype=np.int64)
    matures = times + DAY
    markets = np.array(["US", "US", "US"])

    cfg = eo.DenseBankConfig(window=1, k=4, temperatures=(0.1,), clip=0.1)
    means, disps, _, _ = eo.replay_dense_bank(
        eo.DenseBankMemory(cfg, models=1),
        markets=markets,
        prediction_at=times,
        matures_at=matures,
        keys=keys,
        predictions=preds,
        target=targets,
    )
    assert means[0.1][1, 0] == pytest.approx(0.09, abs=1e-14)
    assert means[0.1][2, 0] == pytest.approx(-0.03, abs=1e-14)
    assert disps[0.1][2, 0] == pytest.approx(0.0, abs=1e-14)

    # Aislamiento entre mercados: añadir CN con residuos opuestos no cambia ni un bit de US.
    cn_keys = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    cn_targets = np.array([-0.09, 0.09, 0.0], dtype=np.float64)
    joint_m, joint_d, _, _ = eo.replay_dense_bank(
        eo.DenseBankMemory(cfg, models=1),
        markets=np.concatenate([markets, np.array(["CN", "CN", "CN"])]),
        prediction_at=np.concatenate([times, times]),
        matures_at=np.concatenate([matures, matures]),
        keys=np.vstack([keys, cn_keys]),
        predictions=np.zeros((6, 1), dtype=np.float64),
        target=np.concatenate([targets, cn_targets]),
    )
    np.testing.assert_array_equal(joint_m[0.1][:3], means[0.1])
    np.testing.assert_array_equal(joint_d[0.1][:3], disps[0.1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_holographic_fft_primitives_support_odd_dim_and_finite_gradients_including_zero(dtype):
    """Dimensión impar unitaria y gradientes finitos en FP32/FP64 incluso en entradas nulas."""
    dim = 1025
    tol = 1e-5 if dtype == torch.float32 else 1e-13
    u = eo.unitary_base_vector(dim, 17, "asset", "US/ODD", dtype=dtype)
    v = eo.unitary_base_vector(dim, 17, "value", "residual", dtype=dtype)
    spectrum = torch.fft.rfft(u, n=dim)
    torch.testing.assert_close(spectrum.abs(), torch.ones_like(spectrum.real), rtol=0.0, atol=tol)
    assert float(torch.linalg.vector_norm(u)) == pytest.approx(1.0, abs=tol)
    recovered = eo.circular_correlation(u, eo.circular_convolution(u, v))
    torch.testing.assert_close(recovered, v, rtol=0.0, atol=tol)

    # Gradientes finitos con entradas nulas/constantes y con entradas unitarias.
    for left_data, right_data in (
        (torch.zeros(dim, dtype=dtype), torch.zeros(dim, dtype=dtype)),
        (torch.full((dim,), 0.25, dtype=dtype), torch.full((dim,), -0.25, dtype=dtype)),
        (u.detach().clone(), v.detach().clone()),
    ):
        left = left_data.clone().requires_grad_(True)
        right = right_data.clone().requires_grad_(True)
        bound = eo.circular_convolution(left, right)
        unbound = eo.circular_correlation(left, bound)
        loss = (unbound**2).sum() + bound.sum()
        loss.backward()
        assert torch.isfinite(left.grad).all() and torch.isfinite(right.grad).all()


def test_holographic_delta_isolates_markets_and_respects_unclipped_mode_and_placebo_seeds():
    """Aislamiento entre mercados, modo clip=None y placebos distintos por semilla."""
    cfg = eo.HolographicDeltaConfig(
        dim=1024, decay=0.95, rate=1.0, seed=3, entity="asset", clip=None, dtype="float64"
    )
    times = np.array([START, START, START + DAY, START + DAY], dtype=np.int64)
    matures = times + DAY
    entities = np.array(["US/A", "US/B", "US/A", "US/B"])
    regimes = np.array(["calm_up"] * 4)
    preds = np.zeros((4, 1), dtype=np.float64)
    targets = np.array([2.5, -1.5, 0.0, 0.0], dtype=np.float64)  # > 0.1 sin recorte

    us_only, us_p1 = eo.replay_holographic_delta(
        eo.HolographicDeltaMemory(cfg, models=1, placebo_seed=10),
        markets=np.array(["US"] * 4),
        prediction_at=times,
        matures_at=matures,
        entities=entities,
        regimes=regimes,
        predictions=preds,
        target=targets,
    )
    # Con clip=None, el residuo 2.5 se recupera cerca de 2.5 (no recortado a 0.1).
    assert abs(us_only[2, 0] - 2.5) < 0.2
    assert abs(us_only[3, 0] - (-1.5)) < 0.2

    # Añadir CN con las mismas claves de régimen y residuos gigantes no toca ni un bit de US.
    joint, joint_p1 = eo.replay_holographic_delta(
        eo.HolographicDeltaMemory(cfg, models=1, placebo_seed=10),
        markets=np.array(["US"] * 4 + ["CN"] * 4),
        prediction_at=np.concatenate([times, times]),
        matures_at=np.concatenate([matures, matures]),
        entities=np.concatenate([entities, np.array(["CN/A", "CN/B", "CN/A", "CN/B"])]),
        regimes=np.concatenate([regimes, regimes]),
        predictions=np.zeros((8, 1), dtype=np.float64),
        target=np.concatenate([targets, np.array([50.0, -50.0, 0.0, 0.0])]),
    )
    np.testing.assert_array_equal(joint[:4], us_only)
    np.testing.assert_array_equal(joint_p1[:4], us_p1)

    # Semillas distintas producen placebos distintos en un panel con suficientes filas.
    many_e = np.tile([f"US/A{i}" for i in range(12)], 3)
    many_r = np.full(36, "calm_up")
    many_t = np.repeat([START, START + DAY, START + 2 * DAY], 12).astype(np.int64)
    rng = np.random.default_rng(73)
    many_y = rng.normal(scale=0.1, size=36)
    _, p_a = eo.replay_holographic_delta(
        eo.HolographicDeltaMemory(cfg, models=1, placebo_seed=1),
        markets=np.array(["US"] * 36),
        prediction_at=many_t,
        matures_at=many_t + DAY,
        entities=many_e,
        regimes=many_r,
        predictions=np.zeros((36, 1)),
        target=many_y,
    )
    _, p_b = eo.replay_holographic_delta(
        eo.HolographicDeltaMemory(cfg, models=1, placebo_seed=2),
        markets=np.array(["US"] * 36),
        prediction_at=many_t,
        matures_at=many_t + DAY,
        entities=many_e,
        regimes=many_r,
        predictions=np.zeros((36, 1)),
        target=many_y,
    )
    assert not np.array_equal(p_a[12:], p_b[12:])


def test_all_three_organs_hold_2023_decisions_maturing_in_2024_and_reject_contract_violations():
    """Decisiones de 2023 que maduran en 2024 quedan pendientes sin escribirse; guardas intactas."""
    last = FINAL_TEST_START_US - DAY // 2
    matures_2024 = last + DAY
    assert last < FINAL_TEST_START_US <= matures_2024
    times = np.array([last - DAY, last], dtype=np.int64)
    matures = np.array([last, matures_2024], dtype=np.int64)
    markets = np.array(["US", "US"])
    assets = np.array(["US/A", "US/A"])
    regimes = np.array(["calm_up", "calm_up"])
    keys = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    preds = np.zeros((2, 1), dtype=np.float64)
    targets = np.array([0.05, 0.09], dtype=np.float64)

    dense = eo.DenseBankMemory(eo.DenseBankConfig(window=4, k=4, temperatures=(0.1,)), models=1)
    eo.replay_dense_bank(
        dense,
        markets=markets,
        prediction_at=times,
        matures_at=matures,
        keys=keys,
        predictions=preds,
        target=targets,
    )
    assert dense.pending("US") == [(int(last), int(matures_2024))]
    before = dense.export()["state_sha256"]
    with pytest.raises(ValueError, match="desarrollo"):
        dense.mature("US", int(last), np.array([0.09]), at=int(matures_2024))
    with pytest.raises(ValueError, match="retroceder"):
        dense.emit("US", int(last - 1), keys[:1], preds[:1], matures_at=int(last))
    with pytest.raises(ValueError, match="finita"):
        dense.emit("US", int(last + 1), keys[:1], np.array([[np.nan]]), matures_at=int(last + DAY))
    assert dense.export()["state_sha256"] == before

    holo = eo.HolographicDeltaMemory(eo.HolographicDeltaConfig(dim=1024, decay=0.95, seed=1), 1)
    eo.replay_holographic_delta(
        holo,
        markets=markets,
        prediction_at=times,
        matures_at=matures,
        entities=assets,
        regimes=regimes,
        predictions=preds,
        target=targets,
    )
    assert holo.pending("US") == [(int(last), int(matures_2024))]
    before_h = holo.export()["state_sha256"]
    with pytest.raises(ValueError, match="desarrollo"):
        holo.mature("US", int(last), np.array([0.09]), at=int(matures_2024))
    with pytest.raises(ValueError, match="retroceder"):
        holo.emit("US", int(last - 1), assets[:1], regimes[:1], preds[:1], matures_at=int(last))
    assert holo.export()["state_sha256"] == before_h

    trace = eo.AssetResidualTrace(eo.AssetTraceConfig(half_life=5.0), 1)
    eo.replay_asset_trace(
        trace,
        markets=markets,
        assets=assets,
        prediction_at=times,
        matures_at=matures,
        predictions=preds,
        target=targets,
    )
    assert trace.pending("US") == [(int(last), int(matures_2024))]
    before_t = trace.export()["state_sha256"]
    with pytest.raises(ValueError, match="desarrollo"):
        trace.mature("US", int(last), np.array([0.09]), at=int(matures_2024))
    with pytest.raises(ValueError, match="retroceder"):
        trace.emit("US", int(last - 1), assets[:1], preds[:1], matures_at=int(last))
    assert trace.export()["state_sha256"] == before_t
