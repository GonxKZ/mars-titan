"""Memoria por activo del residuo madurado (H1-b) como etapa posterior sobre emisiones.

Los paneles son sintéticos y no proceden de ningún modelo. Se comprueba la causalidad por
perturbación del futuro, la identidad con la definición del laboratorio tabular (retraso de
una sesión), el placebo que lee la memoria de otro activo, la pausa y reanudación sin pérdida,
los casos límite y que cada guarda rechaza lo que debe rechazar.
"""

import numpy as np
import pytest
from scipy.special import ndtri

from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.memory import asset_residual_memory as arm

DAY = 86_400_000_000
START = 1_356_000_000_000_000  # diciembre de 2012 en microsegundos UTC


def panel(sessions=30, assets=12, *, seed=0, markets=("US",), lag=1, missing=()):
    """Sesiones diarias con activos fijos por mercado y madurez a lag sesiones.

    missing lista pares (sesión, activo) que no cotizan. Las predicciones llevan una parte de
    la etiqueta y otra persistente de cada activo, así que la memoria tiene algo que aprender.
    """
    rng = np.random.default_rng(seed)
    rows = []
    persistent = {m: rng.normal(0, 1, assets) for m in markets}
    for s in range(sessions):
        for market in markets:
            for a in range(assets):
                if (s, a) in missing:
                    continue
                target = persistent[market][a] * 0.5 + rng.normal()
                rows.append(
                    dict(
                        market=market,
                        asset=f"{market}/A{a:03d}",
                        at=START + s * DAY,
                        matures=START + (s + lag) * DAY,
                        prediction=0.3 * target + rng.normal(),
                        target=target,
                    )
                )
    return rows


def columns(rows):
    return dict(
        markets=np.array([r["market"] for r in rows]),
        assets=np.array([r["asset"] for r in rows]),
        prediction_at=np.array([r["at"] for r in rows], dtype=np.int64),
        matures_at=np.array([r["matures"] for r in rows], dtype=np.int64),
        prediction=np.array([r["prediction"] for r in rows]),
        target=np.array([r["target"] for r in rows]),
    )


CONFIG = arm.AssetMemoryConfig(half_life=5.0, value="residual")


def replay(data, config=CONFIG, kappas=(0.0, 0.1, 0.3), **options):
    memory = arm.AssetResidualMemory(config)
    return arm.replay(memory, kappas=kappas, **data, **options), memory


def zscore(values):
    spread = values.std()
    return (values - values.mean()) / spread if spread > 0 else values * 0.0


def gauss_ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    g = ndtri((ranks - 0.5) / len(values))
    return g - g.mean()


def lab_reference(data, half_life, kappa, value="residual", permutation=None):
    """La memoria por activo del laboratorio tabular (tab_lab.py) con retraso de una sesión."""
    names = sorted(set(data["assets"].tolist()))
    index = {name: i for i, name in enumerate(names)}
    times = np.unique(data["prediction_at"])
    decay = 0.5 ** (1.0 / half_life)
    memory, weight = np.zeros(len(names)), np.zeros(len(names))
    out, previous = np.zeros(len(data["target"])), None
    for t in times:
        rows = np.flatnonzero(data["prediction_at"] == t)
        ids = np.array([index[a] for a in data["assets"][rows]])
        z = zscore(data["prediction"][rows])
        if previous is not None:
            old_ids, old_z, old_y = previous
            g = gauss_ranks(old_y)
            e = g - (float(old_z @ g) / len(old_z)) * old_z
            v = {"residual": e, "label": g, "sign": np.sign(e)}[value]
            memory *= decay
            weight *= decay
            memory[old_ids] += v
            weight[old_ids] += 1.0
        read_ids = ids if permutation is None else permutation[ids]
        read = np.where(
            weight[read_ids] > 0, memory[read_ids] / np.maximum(weight[read_ids], 1e-12), 0.0
        )
        out[rows] = z + kappa * zscore(read)
        previous = (ids, z, data["target"][rows])
    return out


def test_emissions_equal_the_tabular_lab_definition_with_a_one_session_delay():
    data = columns(panel())
    emitted, _ = replay(data, kappas=(0.1,))
    expected = lab_reference(data, 5.0, 0.1)
    np.testing.assert_allclose(emitted[0.1], expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("value", ["label", "sign"])
def test_the_other_stored_values_follow_the_lab_definition(value):
    data = columns(panel(seed=3))
    emitted, _ = replay(data, arm.AssetMemoryConfig(half_life=21.0, value=value), kappas=(0.2,))
    np.testing.assert_allclose(
        emitted[0.2], lab_reference(data, 21.0, 0.2, value=value), rtol=0, atol=1e-12
    )


def test_kappa_zero_emits_the_session_zscore_of_the_parent():
    data = columns(panel(seed=1))
    emitted, _ = replay(data, kappas=(0.0,))
    for t in np.unique(data["prediction_at"]):
        rows = data["prediction_at"] == t
        np.testing.assert_array_equal(emitted[0.0][rows], zscore(data["prediction"][rows]))


def test_future_labels_and_inputs_never_change_an_earlier_emission():
    data = columns(panel(sessions=40, seed=2, lag=2))
    emitted, _ = replay(data)
    cut = START + 25 * DAY
    changed = {key: value.copy() for key, value in data.items()}
    rng = np.random.default_rng(99)
    future_label = changed["matures_at"] > cut
    future_input = changed["prediction_at"] > cut
    changed["target"][future_label] = rng.normal(size=future_label.sum()) * 50
    changed["prediction"][future_input] = rng.normal(size=future_input.sum()) * 50
    perturbed, _ = replay(changed)
    early = data["prediction_at"] <= cut
    for kappa in emitted:
        np.testing.assert_array_equal(perturbed[kappa][early], emitted[kappa][early])
    assert not np.array_equal(perturbed[0.3][~early], emitted[0.3][~early])


def test_a_label_that_matures_exactly_at_the_decision_is_read_and_a_later_one_is_not():
    data = columns(panel(sessions=6, seed=4))
    memory = arm.AssetResidualMemory(CONFIG)
    first = data["prediction_at"] == START
    memory.emit("US", START, data["assets"][first], data["prediction"][first], kappas=(0.1,),
                matures_at=START + DAY)
    with pytest.raises(ValueError, match="todavía no es madura"):
        memory.mature("US", START, data["target"][first], at=START + DAY - 1)
    memory.mature("US", START, data["target"][first], at=START + DAY)
    assert memory.matured("US") == 1


def test_the_placebo_reads_the_memory_of_another_asset():
    data = columns(panel(seed=5))
    names = sorted(set(data["assets"].tolist()))
    permutation = np.random.default_rng(7).permutation(len(names))
    mapping = {names[i]: names[permutation[i]] for i in range(len(names))}
    emitted, _ = replay(data, kappas=(0.2,), read_map=mapping)
    np.testing.assert_allclose(
        emitted[0.2], lab_reference(data, 5.0, 0.2, permutation=permutation), rtol=0, atol=1e-12
    )
    honest, _ = replay(data, kappas=(0.2,))
    assert not np.array_equal(honest[0.2], emitted[0.2])


def test_pause_and_restore_mid_stream_change_nothing():
    data = columns(panel(sessions=30, seed=6, lag=2))
    whole, memory = replay(data)
    cut = START + 13 * DAY
    head = {key: value[data["prediction_at"] <= cut] for key, value in data.items()}
    tail = {key: value[data["prediction_at"] > cut] for key, value in data.items()}
    first = arm.AssetResidualMemory(CONFIG)
    early = arm.replay(first, kappas=(0.0, 0.1, 0.3), **head, flush=False)
    state = first.export()
    restored = arm.AssetResidualMemory.restore(state)
    assert restored.state_sha256() == first.state_sha256()
    # Las etiquetas de la cabeza que maduran durante la cola llegan con la cola.
    late = arm.replay(restored, kappas=(0.0, 0.1, 0.3), **tail, pending_targets=head)
    for kappa in whole:
        joined = np.empty_like(whole[kappa])
        joined[data["prediction_at"] <= cut] = early[kappa]
        joined[data["prediction_at"] > cut] = late[kappa]
        np.testing.assert_array_equal(joined, whole[kappa])
    assert restored.state_sha256() == memory.state_sha256()


def test_a_tampered_state_is_rejected():
    data = columns(panel(sessions=8, seed=8, lag=2))
    _, memory = replay(data, flush=False)
    state = memory.export()
    state["markets"]["US"]["weight"][0] += 1.0
    with pytest.raises(ValueError, match="huella"):
        arm.AssetResidualMemory.restore(state)


def test_a_newly_listed_asset_reads_an_empty_memory_without_breaking_the_session():
    missing = {(s, 3) for s in range(10)}
    data = columns(panel(sessions=14, seed=9, missing=missing))
    emitted, memory = replay(data, kappas=(0.2,))
    assert np.isfinite(emitted[0.2]).all()
    np.testing.assert_allclose(emitted[0.2], lab_reference(data, 5.0, 0.2), rtol=0, atol=1e-12)
    mean, weight = memory.read("US", np.array(["US/A003", "US/A999"]))
    assert weight[1] == 0 and mean[1] == 0


def test_constant_predictions_and_constant_reads_give_zero_scores():
    data = columns(panel(sessions=5, seed=10))
    data["prediction"][:] = 1.5
    emitted, _ = replay(data, kappas=(0.3,))
    first = data["prediction_at"] == START
    np.testing.assert_array_equal(emitted[0.3][first], np.zeros(first.sum()))
    assert np.isfinite(emitted[0.3]).all()


def test_a_nan_prediction_fails_with_its_coordinates():
    data = columns(panel(sessions=4, seed=11))
    data["prediction"][5] = np.nan
    with pytest.raises(ValueError, match="US/A005") as raised:
        replay(data)
    assert str(START) in str(raised.value)


def test_a_nan_target_fails_with_its_coordinates_when_it_matures():
    data = columns(panel(sessions=4, seed=12))
    data["target"][14] = np.inf
    with pytest.raises(ValueError, match="US/A002") as raised:
        replay(data)
    assert str(START + DAY) in str(raised.value)


def test_cohorts_mature_once_in_emission_order_and_the_clock_never_goes_back():
    data = columns(panel(sessions=3, seed=13))
    memory = arm.AssetResidualMemory(CONFIG)
    for s in range(2):
        rows = data["prediction_at"] == START + s * DAY
        memory.emit(
            "US", START + s * DAY, data["assets"][rows], data["prediction"][rows],
            kappas=(0.0,), matures_at=START + (s + 1) * DAY,
        )
    second = data["prediction_at"] == START + DAY
    with pytest.raises(ValueError, match="fuera de orden"):
        memory.mature("US", START + DAY, data["target"][second], at=START + 3 * DAY)
    first = data["prediction_at"] == START
    memory.mature("US", START, data["target"][first], at=START + 3 * DAY)
    with pytest.raises(ValueError, match="retroceder"):
        memory.emit(
            "US", START + DAY, data["assets"][second], data["prediction"][second],
            kappas=(0.0,), matures_at=START + 2 * DAY,
        )


def test_a_repeated_asset_in_a_session_is_rejected():
    data = columns(panel(sessions=2, seed=14))
    rows = np.flatnonzero(data["prediction_at"] == START)
    assets = data["assets"][rows].copy()
    assets[1] = assets[0]
    memory = arm.AssetResidualMemory(CONFIG)
    with pytest.raises(ValueError, match="repetido"):
        memory.emit(
            "US", START, assets, data["prediction"][rows], kappas=(0.0,), matures_at=START + DAY
        )


def test_a_label_that_matures_before_its_decision_is_rejected():
    data = columns(panel(sessions=2, seed=16))
    data["matures_at"][:] = data["prediction_at"]
    with pytest.raises(ValueError, match="después de su decisión"):
        replay(data)


def test_emit_rejects_a_label_that_matures_at_its_own_decision():
    data = columns(panel(sessions=1, seed=17))
    memory = arm.AssetResidualMemory(CONFIG)
    with pytest.raises(ValueError, match="después de su decisión"):
        memory.emit(
            "US", START, data["assets"], data["prediction"], kappas=(0.0,), matures_at=START
        )


def test_a_cohort_cannot_mature_before_the_last_emission_of_its_market():
    data = columns(panel(sessions=3, seed=18))
    memory = arm.AssetResidualMemory(CONFIG)
    for s in range(3):
        rows = data["prediction_at"] == START + s * DAY
        memory.emit(
            "US", START + s * DAY, data["assets"][rows], data["prediction"][rows],
            kappas=(0.0,), matures_at=START + (s + 1) * DAY,
        )
    first = data["prediction_at"] == START
    before = memory.state_sha256()
    with pytest.raises(ValueError, match="retroceder"):
        memory.mature("US", START, data["target"][first], at=START + DAY)
    assert memory.state_sha256() == before


def test_markets_decay_and_learn_separately():
    data = columns(panel(sessions=12, seed=15, markets=("US", "CN")))
    emitted, memory = replay(data, kappas=(0.2,))
    us = data["markets"] == "US"
    alone = {key: value[us] for key, value in data.items()}
    only_us, _ = replay(alone, kappas=(0.2,))
    np.testing.assert_array_equal(emitted[0.2][us], only_us[0.2])
    assert memory.matured("CN") == memory.matured("US") == 12


def test_the_final_test_stays_closed():
    memory = arm.AssetResidualMemory(CONFIG)
    with pytest.raises(ValueError, match="desarrollo"):
        memory.emit(
            "US", FINAL_TEST_START_US, np.array(["US/A"]), np.array([0.1]),
            kappas=(0.0,), matures_at=FINAL_TEST_START_US + DAY,
        )


@pytest.mark.parametrize(
    "config",
    [
        dict(half_life=0.0, value="residual"),
        dict(half_life=float("nan"), value="residual"),
        dict(half_life=5.0, value="otra"),
    ],
)
def test_an_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        arm.AssetMemoryConfig(**config)
