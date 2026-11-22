"""Memoria por vecinos del residuo maduro: definición del laboratorio, causalidad y contratos."""

import numpy as np
import pytest
from scipy.special import ndtri

from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.memory import neighbour_residual_memory as nrm

DAY = 86_400_000_000
START = 1_333_310_700_000_000
CONFIG = nrm.NeighbourConfig(window=3, k=7, tau=0.2, clip=3.0)


def panel(*, sessions=10, assets=30, inputs=10, dim=8, models=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = sessions * assets
    raw = rng.normal(size=(rows, inputs)).astype(np.float32)
    matrix = nrm.projection(inputs, dim, seed=5)
    return dict(
        markets=np.array(["US"] * rows),
        prediction_at=np.repeat(START + np.arange(sessions) * DAY, assets).astype(np.int64),
        matures_at=np.repeat(START + (np.arange(sessions) + 1) * DAY, assets).astype(np.int64),
        keys=nrm.keys_from_inputs(raw, matrix),
        predictions=rng.normal(size=(rows, models)),
        target=rng.normal(size=rows) * 0.01,
    )


def run(data, config=CONFIG, **options):
    memory = nrm.NeighbourMemory(config, models=data["predictions"].shape[1])
    reads, placebo = nrm.replay(memory, **data, **options)
    return reads, placebo, memory


def zscore(v):
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0.0


def lab_reference(data, config, assets, exclude_own=False):
    """La lectura del laboratorio tabular escrita a mano: banco de las W sesiones anteriores.

    Con `exclude_own`, cada consulta descarta las filas de su propio activo (la fila r del
    panel es el activo r % assets).
    """
    sessions = len(data["target"]) // assets
    models = data["predictions"].shape[1]
    keys, target = data["keys"], data["target"]
    values = []
    for s in range(sessions):
        rows = slice(s * assets, (s + 1) * assets)
        order = np.argsort(target[rows], kind="mergesort")
        ranks = np.empty(assets)
        ranks[order] = np.arange(1, assets + 1)
        g = ndtri((ranks - 0.5) / assets)
        g = g - g.mean()
        columns = []
        for m in range(models):
            z = zscore(data["predictions"][rows, m])
            columns.append(np.clip(g - (z @ g / assets) * z, -config.clip, config.clip))
        values.append(np.stack(columns, axis=1))
    reads = np.zeros((len(target), models))
    for s in range(1, sessions):
        first = max(0, s - config.window)
        bank = keys[first * assets : s * assets].astype(np.float64)
        value = np.concatenate(values[first:s])
        query = keys[s * assets : (s + 1) * assets].astype(np.float64)
        sims = query @ bank.T
        k = min(config.k, len(bank))
        for i in range(assets):
            if exclude_own:
                sims[i, np.arange(len(bank)) % assets == i] = -np.inf
            top = np.argsort(-sims[i], kind="stable")[:k]
            w = np.exp((sims[i, top] - sims[i, top].max()) / config.tau)
            w /= w.sum()
            reads[s * assets + i] = w @ value[top]
    return reads


def test_reads_equal_the_tabular_lab_definition():
    data = panel(sessions=8, assets=30)
    reads, _, _ = run(data)
    # Cosenos en FP32 frente a la referencia en FP64: diferencias de redondeo de orden 1e-6.
    np.testing.assert_allclose(reads, lab_reference(data, CONFIG, 30), rtol=0, atol=2e-5)


def test_future_labels_keys_and_predictions_never_change_an_earlier_read():
    data = panel(sessions=8, assets=30, seed=1)
    base, base_placebo, _ = run(data, placebo_seed=3)
    cut = 5 * 30
    changed = {k: np.array(v, copy=True) for k, v in data.items()}
    changed["target"][cut - 30 :] += 1.0
    changed["keys"][cut:] = changed["keys"][cut:][::-1]
    changed["predictions"][cut:] *= -3.0
    reads, placebo, _ = run(changed, placebo_seed=3)
    np.testing.assert_array_equal(reads[:cut], base[:cut])
    np.testing.assert_array_equal(placebo[:cut], base_placebo[:cut])
    assert not np.array_equal(reads[cut:], base[cut:])


def test_a_label_that_matures_at_the_decision_is_read_and_a_later_one_is_not():
    data = panel(sessions=3, assets=30, seed=2)
    reads, _, _ = run(data)
    assert np.any(reads[30:60] != 0)
    late = dict(data, matures_at=data["matures_at"] + 1)
    reads, _, _ = run(late)
    np.testing.assert_array_equal(reads[30:60], np.zeros((30, 2)))


def test_an_empty_bank_reads_zero_and_a_small_bank_uses_every_row():
    data = panel(sessions=2, assets=5, seed=3)
    config = nrm.NeighbourConfig(window=3, k=50, tau=0.2, clip=3.0)
    reads, _, _ = run(data, config)
    np.testing.assert_array_equal(reads[:5], np.zeros((5, 2)))
    np.testing.assert_allclose(reads, lab_reference(data, config, 5), rtol=0, atol=2e-5)


def test_the_placebo_permutes_the_bank_values_and_is_reproducible():
    data = panel(sessions=6, assets=30, seed=4)
    reads, placebo, _ = run(data, placebo_seed=9)
    _, again, _ = run(data, placebo_seed=9)
    _, other, _ = run(data, placebo_seed=10)
    np.testing.assert_array_equal(placebo, again)
    assert not np.array_equal(placebo[30:], reads[30:])
    assert not np.array_equal(placebo[30:], other[30:])
    # Con k igual al banco entero y temperatura enorme, permutar los valores solo cambia el
    # orden de la suma de la media (redondeo).
    flat = nrm.NeighbourConfig(window=3, k=1000, tau=1e9, clip=3.0)
    reads, placebo, _ = run(data, flat, placebo_seed=9)
    np.testing.assert_allclose(placebo, reads, rtol=0, atol=1e-9)


def test_pause_and_restore_mid_stream_change_nothing():
    data = panel(sessions=8, assets=30, seed=5)
    whole, whole_placebo, _ = run(data, placebo_seed=4)
    head = slice(0, 4 * 30)
    first = {k: v[head] for k, v in data.items()}
    memory = nrm.NeighbourMemory(CONFIG, models=2)
    reads_a, placebo_a = nrm.replay(memory, **first, placebo_seed=4, flush=False)
    restored = nrm.NeighbourMemory.restore(memory.export())
    rest = {k: v[4 * 30 :] for k, v in data.items()}
    reads_b, placebo_b = nrm.replay(
        restored, **rest, placebo_seed=4, pending_targets=dict(
            markets=first["markets"], prediction_at=first["prediction_at"], target=first["target"]
        ),
    )
    np.testing.assert_array_equal(np.concatenate([reads_a, reads_b]), whole)
    np.testing.assert_array_equal(np.concatenate([placebo_a, placebo_b]), whole_placebo)


def test_a_tampered_state_is_rejected():
    data = panel(sessions=4, assets=30, seed=6)
    _, _, memory = run(data)
    state = memory.export()
    state["markets"]["US"]["bank"][0]["values"][0, 0] += 1.0
    with pytest.raises(ValueError, match="huella"):
        nrm.NeighbourMemory.restore(state)


def test_non_finite_values_fail_with_their_coordinates():
    data = panel(sessions=3, assets=30, seed=7)
    data["predictions"][31, 1] = np.nan
    with pytest.raises(ValueError, match=str(START + DAY)):
        run(data)
    data = panel(sessions=3, assets=30, seed=7)
    data["target"][2] = np.inf
    with pytest.raises(ValueError, match=str(START)):
        run(data)


def test_cohorts_mature_in_order_and_the_clock_never_goes_back():
    data = panel(sessions=3, assets=30, seed=8)
    memory = nrm.NeighbourMemory(CONFIG, models=2)
    for s in range(2):
        rows = slice(s * 30, (s + 1) * 30)
        memory.emit(
            "US", START + s * DAY, data["keys"][rows], data["predictions"][rows],
            matures_at=START + (s + 1) * DAY,
        )
    with pytest.raises(ValueError, match="fuera de orden"):
        memory.mature("US", START + DAY, data["target"][30:60], at=START + 3 * DAY)
    with pytest.raises(ValueError, match="todavía no es madura"):
        memory.mature("US", START, data["target"][:30], at=START + DAY - 1)
    with pytest.raises(ValueError, match="retroceder"):
        memory.emit(
            "US", START, data["keys"][:30], data["predictions"][:30], matures_at=START + DAY
        )
    with pytest.raises(ValueError, match="después de su decisión"):
        memory.emit(
            "US", START + 2 * DAY, data["keys"][60:], data["predictions"][60:],
            matures_at=START + 2 * DAY,
        )


def test_stored_residuals_are_clipped_like_the_lab():
    data = panel(sessions=6, assets=30, seed=11)
    tight = nrm.NeighbourConfig(window=3, k=7, tau=0.2, clip=0.5)
    reads, _, _ = run(data, tight)
    np.testing.assert_allclose(reads, lab_reference(data, tight, 30), rtol=0, atol=2e-5)
    loose, _, _ = run(data)
    assert np.abs(reads).max() <= 0.5 + 1e-12 < np.abs(loose).max()


def test_a_cohort_cannot_mature_before_the_last_emission_of_its_market():
    data = panel(sessions=3, assets=30, seed=12)
    memory = nrm.NeighbourMemory(CONFIG, models=2)
    for s in range(3):
        rows = slice(s * 30, (s + 1) * 30)
        memory.emit(
            "US", START + s * DAY, data["keys"][rows], data["predictions"][rows],
            matures_at=START + (s + 1) * DAY,
        )
    before = memory.export()["state_sha256"]
    with pytest.raises(ValueError, match="retroceder"):
        memory.mature("US", START, data["target"][:30], at=START + DAY)
    assert memory.export()["state_sha256"] == before


def test_the_final_test_stays_closed():
    data = panel(sessions=2, assets=30, seed=9)
    shift = FINAL_TEST_START_US - START
    data["prediction_at"] = data["prediction_at"] + shift
    data["matures_at"] = data["matures_at"] + shift
    with pytest.raises(ValueError, match="desarrollo"):
        run(data)


def test_a_decision_of_2023_whose_label_matures_in_2024_is_read_but_never_written():
    """H01 de la auditoría de Gemini.

    La etiqueta de la última sesión de 2023 pertenece a 2023 por su decisión, pero madura en el
    test final. Su cohorte se emite y ninguna escritura la usa, igual que en la memoria por
    activo. Con datos de la campaña no llega a ocurrir: la vista excluye esas etiquetas
    (target_after_cutoff).
    """
    data = panel(sessions=3, assets=30, seed=9)
    shift = FINAL_TEST_START_US - START - 2 * DAY - DAY // 2
    data["prediction_at"] = data["prediction_at"] + shift
    data["matures_at"] = data["matures_at"] + shift
    last, matures = int(data["prediction_at"][-1]), int(data["matures_at"][-1])
    assert last < FINAL_TEST_START_US <= matures
    reads, _, memory = run(data)
    first_two = {key: value[:60] for key, value in data.items()}
    np.testing.assert_array_equal(reads[:60], run(first_two)[0])
    assert memory.pending("US") == [(last, matures)]
    assert nrm.NeighbourMemory.restore(memory.export()).pending("US") == [(last, matures)]
    with pytest.raises(ValueError, match="desarrollo"):
        memory.mature("US", last, data["target"][-30:], at=matures)


def test_projection_keys_are_unit_rows_and_reproducible():
    rng = np.random.default_rng(10)
    raw = rng.normal(size=(20, 6)).astype(np.float32)
    raw[3] = 0.0
    matrix = nrm.projection(6, 4, seed=1)
    np.testing.assert_array_equal(matrix, nrm.projection(6, 4, seed=1))
    assert set(np.unique(matrix).tolist()) == {-1.0, 1.0}
    keys = nrm.keys_from_inputs(raw, matrix)
    assert keys.dtype == np.float32
    norms = np.linalg.norm(keys, axis=1)
    np.testing.assert_allclose(np.delete(norms, 3), 1.0, atol=1e-6)
    np.testing.assert_array_equal(keys[3], np.zeros(4, dtype=np.float32))


@pytest.mark.parametrize(
    "options",
    [
        dict(window=0, k=5, tau=0.2),
        dict(window=3, k=0, tau=0.2),
        dict(window=3, k=5, tau=0.0),
        dict(window=3, k=5, tau=0.2, clip=-1.0),
        dict(window=3, k=5, tau=float("nan")),
    ],
)
def test_an_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        nrm.NeighbourConfig(**options)


def asset_codes(data, assets):
    """Código del activo de cada fila del panel: la fila r es el activo r % assets."""
    return np.tile(np.arange(assets), len(data["target"]) // assets).astype(np.int64)


EXCLUDING = nrm.NeighbourConfig(window=3, k=7, tau=0.2, clip=3.0, exclude_own=True)


@pytest.mark.parametrize("exclude_own", [False, True])
def test_ties_at_the_kth_neighbour_go_to_the_oldest_rows_as_in_the_stable_lab_order(exclude_own):
    data = panel(sessions=6, assets=30, seed=4)
    # Claves canónicas: los cosenos valen 0 o 1 exactos y cada consulta empata con decenas de
    # filas del banco (las de los activos que comparten su clave), más de las k que lee.
    keys = np.zeros_like(data["keys"])
    keys[np.arange(len(keys)), np.arange(len(keys)) % 3] = 1.0
    data["keys"] = keys
    config = EXCLUDING if exclude_own else CONFIG
    options = dict(assets=asset_codes(data, 30)) if exclude_own else {}
    reads, _, _ = run(data, config, **options)
    expected = lab_reference(data, config, 30, exclude_own=exclude_own)
    np.testing.assert_allclose(reads, expected, rtol=0, atol=2e-5)


@pytest.mark.parametrize(
    ("size", "k", "levels"),
    [(10, 7, None), (600, 256, None), (1003, 7, None), (1003, 7, 3), (5000, 256, 4)],
)
def test_the_selection_of_neighbours_is_exactly_the_stable_order_of_the_lab(size, k, levels):
    rng = np.random.default_rng(size + k)
    sims = rng.normal(size=(6, size)).astype(np.float32)
    if levels is not None:
        # Pocos valores distintos: empates en el k-ésimo coseno.
        sims = np.round(sims * levels / 3).astype(np.float32)
    sims[1, : size // 2] = -np.inf  # filas del propio activo excluidas
    sims[2, : size - 3] = -np.inf  # menos de k candidatas finitas
    expected = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    np.testing.assert_array_equal(nrm._top(sims, k), expected)


@pytest.mark.parametrize("exclude_own", [False, True])
def test_reads_do_not_depend_on_the_number_of_reading_threads(exclude_own):
    # 300 activos: tres bloques de consultas por cohorte, que se reparten entre los hilos.
    data = panel(sessions=6, assets=300, seed=8)
    config = EXCLUDING if exclude_own else CONFIG
    options = dict(assets=asset_codes(data, 300)) if exclude_own else {}
    single, single_placebo, plain = run(data, config, placebo_seed=4, **options)
    memory = nrm.NeighbourMemory(config, models=2, threads=3)
    threaded, threaded_placebo = nrm.replay(memory, **data, placebo_seed=4, **options)
    np.testing.assert_array_equal(threaded, single)
    np.testing.assert_array_equal(threaded_placebo, single_placebo)
    # Los hilos no forman parte del estado.
    assert memory.export()["state_sha256"] == plain.export()["state_sha256"]


def test_the_reading_threads_are_a_small_positive_integer():
    for threads in (0, 1.5, True, nrm.MAX_THREADS + 1):
        with pytest.raises(ValueError, match="hilos"):
            nrm.NeighbourMemory(CONFIG, models=2, threads=threads)
    saved = nrm.NeighbourMemory(CONFIG, models=2).export()
    assert nrm.NeighbourMemory.restore(saved, threads=2).threads == 2


def test_excluding_the_own_asset_reads_only_the_rows_of_other_assets():
    data = panel(sessions=8, assets=30, seed=11)
    reads, placebo, _ = run(data, EXCLUDING, assets=asset_codes(data, 30), placebo_seed=2)
    expected = lab_reference(data, EXCLUDING, 30, exclude_own=True)
    np.testing.assert_allclose(reads, expected, rtol=0, atol=2e-5)
    # Con todas las filas el propio activo entra en sus vecinos y la lectura cambia.
    plain, _, _ = run(data)
    assert not np.allclose(reads[30:], plain[30:])
    assert placebo is not None and not np.array_equal(placebo[30:], reads[30:])


def test_the_exclusion_needs_the_asset_of_every_row_and_only_with_it():
    data = panel(sessions=3, assets=5, seed=12)
    with pytest.raises(ValueError, match="activo de cada fila"):
        run(data, EXCLUDING)
    with pytest.raises(ValueError, match="activo de cada fila"):
        run(data, EXCLUDING, assets=np.zeros(4, dtype=np.int64))
    with pytest.raises(ValueError, match="activo de cada fila"):
        run(data, EXCLUDING, assets=asset_codes(data, 5).astype(np.float64))
    with pytest.raises(ValueError, match="solo con la exclusión"):
        run(data, assets=asset_codes(data, 5))


def test_a_query_whose_bank_holds_only_its_own_asset_reads_zero():
    config = nrm.NeighbourConfig(window=3, k=2, tau=0.2, clip=3.0, exclude_own=True)
    memory = nrm.NeighbourMemory(config, models=1)
    keys = nrm.keys_from_inputs(np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32))
    memory.emit(
        "US",
        START,
        keys,
        np.array([[0.1], [0.2], [0.3]]),
        matures_at=START + DAY,
        assets=np.array([7, 7, 7]),
    )
    memory.mature("US", START, np.array([0.03, -0.01, 0.02]), at=START + DAY)
    reads, placebo = memory.emit(
        "US",
        START + DAY,
        keys[:2],
        np.array([[0.1], [0.2]]),
        matures_at=START + 2 * DAY,
        assets=np.array([7, 8]),
        placebo_seed=1,
    )
    assert np.isfinite(reads).all() and np.isfinite(placebo).all()
    assert reads[0, 0] == 0.0 and placebo[0, 0] == 0.0
    assert reads[1, 0] != 0.0


def test_the_exclusion_survives_a_pause_and_restore():
    data = panel(sessions=8, assets=30, seed=14)
    codes = asset_codes(data, 30)
    whole, _, _ = run(data, EXCLUDING, assets=codes)
    memory = nrm.NeighbourMemory(EXCLUDING, models=2)
    head, tail = slice(0, 4 * 30), slice(4 * 30, None)
    first = {k: v[head] for k, v in data.items()}
    reads_a, _ = nrm.replay(memory, **first, assets=codes[head], flush=False)
    restored = nrm.NeighbourMemory.restore(memory.export())
    rest = {k: v[tail] for k, v in data.items()}
    reads_b, _ = nrm.replay(
        restored,
        **rest,
        assets=codes[tail],
        pending_targets=dict(
            markets=first["markets"], prediction_at=first["prediction_at"], target=first["target"]
        ),
    )
    np.testing.assert_array_equal(np.concatenate([reads_a, reads_b]), whole)


def test_the_identity_names_the_exclusion_only_when_it_is_declared():
    assert CONFIG.identity() == dict(window=3, k=7, tau=0.2, clip=3.0)
    assert EXCLUDING.identity() == dict(window=3, k=7, tau=0.2, clip=3.0, exclude_own=True)
    with pytest.raises(ValueError, match="exclude_own"):
        nrm.NeighbourConfig(window=3, k=7, tau=0.2, exclude_own=1)


def test_a_restored_state_of_an_exclusion_needs_the_asset_of_every_row():
    data = panel(sessions=4, assets=30, seed=15)
    memory = nrm.NeighbourMemory(EXCLUDING, models=2)
    nrm.replay(memory, **data, assets=asset_codes(data, 30), flush=False)
    payload = memory.export()
    nrm.NeighbourMemory.restore(payload)
    for part in ("bank", "pending"):
        broken = memory.export()
        broken["markets"]["US"][part][0].pop("assets")
        broken["state_sha256"] = nrm._state_sha256(broken)
        with pytest.raises(ValueError, match="no es coherente"):
            nrm.NeighbourMemory.restore(broken)


def test_reading_a_cohort_reuses_the_buffers_of_the_earlier_ones():
    """Con el banco lleno, una cohorte se lee sin pedir memoria del tamaño de los bloques.

    La matriz de similitudes de un bloque de consultas frente al banco (128 × 9.000 cosenos FP32,
    4,6 MB aquí; 87 MB con el banco real de 63 cohortes) y la máscara de candidatas se escriben en
    búferes que se reservan una vez y se reutilizan. Pedirlos de nuevo en cada bloque cuesta
    fallos de página y tiempo de sistema (110 millones de fallos en el origen 2013).
    """
    import tracemalloc

    assets, sessions = 3000, 6
    data = panel(sessions=sessions, assets=assets, seed=12)
    first = {key: value[: (sessions - 1) * assets] for key, value in data.items()}
    last = {key: value[(sessions - 1) * assets :] for key, value in data.items()}
    memory = nrm.NeighbourMemory(CONFIG, models=2)
    nrm.replay(memory, **first, flush=False)
    # Solo la cohorte pendiente (la última de `first`) necesita su etiqueta al reanudar.
    pending = slice((sessions - 2) * assets, (sessions - 1) * assets)
    earlier = {key: data[key][pending] for key in ("markets", "prediction_at", "target")}
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        reads, _ = nrm.replay(memory, **last, flush=False, pending_targets=earlier)
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()
    similarities = nrm.CHUNK * CONFIG.window * assets * 4
    assert peak < similarities / 2, f"{peak} bytes nuevos al leer una cohorte"
    whole, _, _ = run(data)
    np.testing.assert_array_equal(reads, whole[(sessions - 1) * assets :])
