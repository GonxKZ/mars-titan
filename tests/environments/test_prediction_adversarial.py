"""Agentes guionizados que intentan obtener etiquetas o créditos sin señal legítima."""

import copy
import importlib
import json

import numpy as np
import pytest

from mars_titan.environments.actions import ActionGrid

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("gymnasium") is None,
    reason="Requiere el extra reinforcement",
)

SHAPES = dict(prices=(2, 5), news=(3,), charts=(2,), fundamentals=(4,), macro=(2,))
SOURCE = "a" * 64
# Valores reconocibles para buscarlos en cualquier salida del entorno.
SECRET = (0.0371234567, -0.0456789123, 0.0612345678)


def cohort(at, *, maturity, targets, ids=("US/B", "US/A"), scale=1.0):
    count = len(ids)
    return dict(
        prediction_at=at,
        asset_ids=list(ids),
        available_at=np.full(count, at),
        target_available_at=np.full(count, maturity),
        target=np.asarray(targets, dtype=np.float64),
        inputs={
            name: np.full((count, *shape), scale, dtype=np.float32)
            for name, shape in SHAPES.items()
        },
    )


def rows(*, first=SECRET[:2], later=SECRET[1:]):
    return [
        cohort(10, maturity=40, targets=first),
        cohort(20, maturity=50, targets=later),
        cohort(30, maturity=60, targets=(0.01, -0.01)),
    ]


class Source:
    """Fuente indexada que declara su partición como las fuentes del proyecto."""

    def __init__(self, cohorts, partition="train"):
        self.cohorts, self.partition = cohorts, partition

    def __call__(self, position):
        return self.cohorts[position] if position < len(self.cohorts) else None


def environment(cohorts=None, *, partition="train", declared=None):
    grid = ActionGrid.fit(np.linspace(-0.1, 0.1, 101), source_sha256=SOURCE, partition="train")
    prediction = importlib.import_module("mars_titan.environments.prediction")
    return prediction.CausalPredictionEnv(
        Source(rows() if cohorts is None else cohorts, declared or partition),
        source_sha256=SOURCE,
        grid=grid,
        shapes=SHAPES,
        max_assets=3,
        partition=partition,
    )


def leaked(value, secrets):
    """Buscar un objetivo en arrays, números y texto JSON de cualquier salida."""
    if isinstance(value, np.ndarray):
        return any(
            np.isclose(value.astype(np.float64), s, rtol=0, atol=1e-12).any() for s in secrets
        )
    if isinstance(value, dict):
        return any(leaked(item, secrets) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(leaked(item, secrets) for item in value)
    if isinstance(value, float):
        return any(abs(value - s) <= 1e-12 for s in secrets)
    return False


def test_oracle_cannot_read_immature_targets_from_observation_info_or_snapshot():
    env = environment()
    observed, info = env.reset(seed=42)
    exposed = [observed, info, env.snapshot()]
    observed, reward, _, _, info = env.step(np.array([10, 10, 10]))
    exposed += [observed, reward, info, env.snapshot()]
    observed, reward, _, _, info = env.step(np.array([10, 10, 10]))
    exposed += [observed, reward, info, env.snapshot()]
    # Ninguna etiqueta madura antes del instante 40. Las tres siguen ocultas.
    assert env.snapshot()["clock"] == 30 and env.snapshot()["pending"]
    assert not leaked(exposed, SECRET)
    text = json.dumps(env.snapshot())
    assert all(repr(s) not in text for s in SECRET)
    assert all("target" not in event for event in env.snapshot()["pending"])
    with pytest.raises(NotImplementedError):
        env.render()


@pytest.mark.parametrize("perturbed", ["pending_label", "future_cohort", "future_inputs"])
def test_future_perturbations_do_not_change_emitted_history(perturbed):
    base, changed = rows(), rows()
    if perturbed == "pending_label":
        changed[0]["target"] = np.array([0.09, -0.09])
    if perturbed == "future_cohort":
        changed[2] = cohort(
            30, maturity=60, targets=(0.08, 0.07, 0.06), ids=("US/C", "US/A", "US/B")
        )
    if perturbed == "future_inputs":
        changed[2]["inputs"]["news"][:] = 9
    one, two = environment(base), environment(changed)
    first, second = [one.reset(seed=1)], [two.reset(seed=1)]
    # El primer paso entrega la cohorte de 20. El segundo ya revela la cohorte de 30.
    steps = 1 if perturbed != "pending_label" else 2
    for _ in range(steps):
        first.append(one.step(np.array([3, 17, 0])))
        second.append(two.step(np.array([3, 17, 0])))
    for a, b in zip(first, second, strict=True):
        observed_a, observed_b = a[0], b[0]
        for name in observed_a:
            np.testing.assert_array_equal(observed_a[name], observed_b[name])
        assert a[1:] == b[1:]


def test_snapshot_cannot_inject_or_keep_targets_and_rereads_source_labels():
    env = environment()
    env.reset(seed=42)
    env.step(np.array([10, 10, 0]))
    state = env.snapshot()
    expected = env.step(np.array([10, 10, 0]))

    restored = environment()
    restored.restore(copy.deepcopy(state))
    result = restored.step(np.array([10, 10, 0]))
    assert result[1:] == expected[1:]

    injected = copy.deepcopy(state)
    injected["pending"][0]["target"] = SECRET[0]
    with pytest.raises(ValueError, match="identidad y maduración"):
        environment().restore(injected)

    legacy = copy.deepcopy(state)
    legacy["schema_version"] = 1
    with pytest.raises(ValueError, match="identidad o los límites"):
        environment().restore(legacy)

    relabelled = rows()
    relabelled[0]["target"] = np.array([0.09, -0.09])
    with pytest.raises(ValueError, match="cohorte de origen"):
        environment(relabelled).restore(copy.deepcopy(state))

    moved = copy.deepcopy(state)
    moved["pending"][0]["position"] = 1
    with pytest.raises(ValueError):
        environment().restore(moved)

    # Adelantar o retrasar una maduración cambiaría cuándo se entrega el crédito.
    delayed = copy.deepcopy(state)
    delayed["pending"][0]["target_available_at"] = 45
    with pytest.raises(ValueError, match="cohorte de origen"):
        environment().restore(delayed)


def test_source_partition_must_match_environment_partition():
    # Diciembre de 2022 pertenece a la validación walk-forward, aunque sea anterior a 2023.
    december = 1_669_852_800_000_000
    validation = [cohort(december, maturity=december + 10, targets=(0.01, 0.02))]
    with pytest.raises(ValueError, match="otra partición"):
        environment(validation, partition="train", declared="validation")


def test_reset_discards_pending_credits_explicitly_and_carries_no_state():
    env = environment()
    env.reset(seed=42)
    env.reset(seed=42)
    assert env.snapshot()["abandoned"] == dict(episodes=0, pending_credits=0)
    env.step(np.array([0, 20, 0]))
    observed, info = env.reset(seed=42)
    assert info["pending"] == 0
    assert env.snapshot()["abandoned"] == dict(episodes=1, pending_credits=2)

    fresh = environment()
    fresh.reset(seed=42)
    for action in ([10, 10, 0], [5, 15, 0], [0, 20, 0]):
        a, b = env.step(np.array(action)), fresh.step(np.array(action))
        assert a[1:] == b[1:]
    assert env.done and fresh.done
    env.reset(seed=42)
    assert env.snapshot()["abandoned"] == dict(episodes=1, pending_credits=2)


@pytest.mark.parametrize(
    "action",
    [
        np.array([10.0, 10.0, 0.0]),
        np.array([np.nan, 10, 0]),
        np.array([True, False, True]),
        np.array([10, 10]),
        np.array([10, 21, 0]),
        np.array([[10, 10, 0]]),
        np.array(["10", "10", "0"]),
        [10, 10, 0.5],
    ],
)
def test_invalid_or_non_integer_actions_fail_before_any_transition(action):
    env = environment()
    env.reset(seed=42)
    before = env.snapshot()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.snapshot() == before


def test_padding_actions_and_scripted_policies_cannot_create_positive_reward():
    env = environment()
    env.reset(seed=42)
    padded = environment()
    padded.reset(seed=42)
    rewards = []
    for step in range(3):
        a = env.step(np.array([step, 20 - step, 0]))
        b = padded.step(np.array([step, 20 - step, 20]))
        assert a[1:] == b[1:]
        rewards.append(a[1])
        assert all(event["reward"] <= 0 for event in a[4]["matured"])
    assert all(reward <= 0 for reward in rewards)
    # Las tres cohortes reciben un único crédito por activo, también en el último paso.
    assert env.snapshot()["pending"] == []
