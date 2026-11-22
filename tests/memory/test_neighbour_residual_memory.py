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


def lab_reference(data, config, assets):
    """La lectura del laboratorio tabular escrita a mano: banco de las W sesiones anteriores."""
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
