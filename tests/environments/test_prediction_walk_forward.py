"""Entorno predictivo con los cortes de una ventana walk-forward declarada en su recibo."""

import copy
import importlib

import numpy as np
import pytest

from mars_titan.environments.actions import ActionGrid
from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.evaluation.splits import PARTITIONS
from tests.environments.walk_forward_fixture import microseconds, window

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("gymnasium") is None,
    reason="Requiere el extra reinforcement",
)

SHAPES = dict(prices=(2, 5), news=(3,), charts=(2,), fundamentals=(4,), macro=(2,))
SOURCE = "a" * 64
DAY = 86_400_000_000


def cohort(at, maturity, targets=(0.01, -0.02), ids=("US/A", "US/B")):
    count = len(ids)
    return dict(
        prediction_at=at,
        asset_ids=list(ids),
        available_at=np.full(count, at),
        target_available_at=np.full(count, maturity),
        target=np.asarray(targets, dtype=np.float64),
        inputs={
            name: np.full((count, *shape), 0.5, dtype=np.float32) for name, shape in SHAPES.items()
        },
    )


class Source:
    def __init__(self, cohorts, partition, bounds=None):
        self.cohorts, self.partition = cohorts, partition
        if bounds is not None:
            self.market_bounds = bounds

    def __call__(self, position):
        return self.cohorts[position] if position < len(self.cohorts) else None


def inside(partition, offsets=(1, 2, 3)):
    """Cohortes diarias desde el inicio del tramo, con la etiqueta del día siguiente."""
    start, _ = window().segment(partition)
    return [cohort(start + k * DAY, start + (k + 1) * DAY) for k in offsets]


def environment(cohorts, partition="evaluation", *, declared=None, bounds=None, **options):
    grid = ActionGrid.fit(np.linspace(-0.1, 0.1, 101), source_sha256=SOURCE, partition="train")
    prediction = importlib.import_module("mars_titan.environments.prediction")
    return prediction.CausalPredictionEnv(
        Source(cohorts, declared or partition, bounds),
        source_sha256=SOURCE,
        grid=grid,
        shapes=SHAPES,
        max_assets=3,
        partition=partition,
        window=options.pop("window", window()),
        **options,
    )


def run(env, action=10):
    env.reset(seed=3)
    rewards, observations = [], []
    while not env.done:
        observation, reward, *_ = env.step(np.full(3, action))
        rewards.append(reward)
        observations.append(observation)
    return rewards, observations


@pytest.mark.parametrize("partition", PARTITIONS)
def test_each_window_segment_admits_its_own_cohorts(partition):
    env = environment(inside(partition), partition)
    start, end = window().segment(partition)
    rewards, _ = run(env)
    assert len(rewards) == 3 and all(np.isfinite(rewards))
    assert (env.start, env.end) == (start, end)
    assert env.identity["walk_forward"] == window().identity(partition)
    assert env.identity["walk_forward"]["receipt_sha256"] == window().sha256


@pytest.mark.parametrize("partition", PARTITIONS)
def test_cohorts_before_after_or_with_labels_crossing_the_segment_end_are_rejected(partition):
    start, end = window().segment(partition)
    for bad in (
        cohort(start - DAY, start + DAY),
        cohort(end, end + DAY),
        # La purga por intervalo: una etiqueta que madura en la frontera queda fuera.
        cohort(end - DAY, end),
    ):
        env = environment([*inside(partition, (1,)), bad], partition)
        env.reset()
        # En la evaluación de 2023 el final del tramo es el test sellado, que ya rechaza la cohorte.
        message = "ventana walk-forward|periodo de desarrollo|fecha o población"
        with pytest.raises(ValueError, match=message):
            env.step(np.zeros(3, dtype=np.int64))
        with pytest.raises(ValueError, match=message):
            environment([bad], partition).reset()


def test_no_cohort_or_label_from_2024_reaches_the_evaluation_segment():
    start, end = window().segment("evaluation")
    assert end == FINAL_TEST_START_US
    env = environment([cohort(end - DAY, end - 1)], "evaluation")
    env.reset()
    _, reward, done, *_ = env.step(np.full(3, 10))
    assert done and np.isfinite(reward)
    with pytest.raises(ValueError):
        environment([cohort(end - DAY, end + DAY)], "evaluation").reset()


def test_legacy_identity_and_cuts_are_unchanged_without_a_window():
    grid = ActionGrid.fit(np.linspace(-0.1, 0.1, 101), source_sha256=SOURCE, partition="train")
    prediction = importlib.import_module("mars_titan.environments.prediction")
    legacy = prediction.CausalPredictionEnv(
        Source([cohort(10, 20)], "train"),
        source_sha256=SOURCE,
        grid=grid,
        shapes=SHAPES,
        max_assets=3,
    )
    assert "walk_forward" not in legacy.identity and legacy.window is None
    with pytest.raises(ValueError, match="presupuesto"):
        prediction.CausalPredictionEnv(
            Source([cohort(10, 20)], "evaluation"),
            source_sha256=SOURCE,
            grid=grid,
            shapes=SHAPES,
            max_assets=3,
            partition="evaluation",
        )


def test_the_source_cannot_declare_other_cuts_or_partition_than_the_receipt():
    start, end = window().segment("validation")
    accepted = environment(inside("validation"), "validation", bounds={"US": (start, end, end)})
    assert accepted.start == start
    for bounds in ({"US": (start, end + DAY, end + DAY)}, {"US": (start, end, end - DAY)}):
        with pytest.raises(ValueError, match="otros cortes"):
            environment(inside("validation"), "validation", bounds=bounds)
    with pytest.raises(ValueError, match="otra partición"):
        environment(inside("validation"), "validation", declared="train")
    with pytest.raises(ValueError, match="presupuesto"):
        environment(inside("validation"), "validation", window=dict(window().bounds))


def test_future_cohorts_do_not_change_emitted_observations_or_rewards():
    base = inside("evaluation", range(1, 6))
    altered = copy.deepcopy(base)
    altered[4]["target"][:] = (0.09, -0.09)
    altered[4]["inputs"]["news"][:] = 7.0
    first, observed = run(environment(base))
    second, other = run(environment(altered))
    # El quinto bloque solo influye a partir de su propia observación y su crédito.
    assert first[:3] == second[:3]
    for a, b in zip(observed[:3], other[:3], strict=True):
        assert all(np.array_equal(a[key], b[key]) for key in a)


def test_snapshot_restores_inside_the_window_and_rejects_another_receipt():
    env = environment(inside("evaluation", range(1, 5)))
    env.reset(seed=1)
    env.step(np.full(3, 8))
    state = env.snapshot()
    restored = environment(inside("evaluation", range(1, 5)))
    restored.restore(state)
    assert restored.position == env.position and restored.clock == env.clock
    other = environment(
        inside("evaluation", range(1, 5)), window=window(until=microseconds("2022-12-01"))
    )
    with pytest.raises(ValueError, match="identidad"):
        other.restore(state)
