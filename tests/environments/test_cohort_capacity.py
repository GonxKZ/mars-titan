"""La cohorte de mayor tamaño medido entra completa y con un presupuesto explícito."""

import importlib

import numpy as np
import pytest

from mars_titan.environments.actions import ActionGrid
from mars_titan.environments.cohorts import MAX_COHORT_ASSETS, read_cohort, shapes_contract

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("gymnasium") is None,
    reason="Requiere el extra reinforcement",
)

# Máximo medido en la población preparada desde 2000: 4.200 activos US el 6-11-2023.
MEASURED_MAXIMUM = 4200
HISTORICAL = dict(prices=(64, 5), news=(384,), charts=(512,), fundamentals=(45,), macro=(420,))
SMALL = dict(prices=(2, 5), news=(1,), charts=(1,), fundamentals=(1,), macro=(1,))
SOURCE = "a" * 64


def cohort(count, at=10, maturity=20, shapes=SMALL):
    ids = [f"US/A{i:05d}" for i in range(count)]
    return dict(
        prediction_at=at,
        asset_ids=ids,
        available_at=np.full(count, at),
        target_available_at=np.full(count, maturity),
        target=np.linspace(-0.05, 0.05, count),
        inputs={
            name: np.zeros((count, *shape), dtype=np.float32) for name, shape in shapes.items()
        },
    )


def test_historical_shapes_fit_the_measured_and_maximum_cohorts_within_budget():
    per_asset = sum(int(np.prod(shape)) for shape in HISTORICAL.values()) * 4 + 1
    assert per_asset == 6725
    shapes_contract(HISTORICAL, MEASURED_MAXIMUM, 64 * 1024**2)
    shapes_contract(HISTORICAL, MAX_COHORT_ASSETS, 64 * 1024**2)
    assert MAX_COHORT_ASSETS * per_asset <= 64 * 1024**2
    with pytest.raises(ValueError, match="presupuesto"):
        shapes_contract(HISTORICAL, MAX_COHORT_ASSETS, MAX_COHORT_ASSETS * per_asset - 1)
    with pytest.raises(ValueError, match="presupuesto del entorno"):
        shapes_contract(SMALL, MAX_COHORT_ASSETS + 1, 64 * 1024**2)


def test_every_asset_of_the_largest_measured_session_is_observed_and_credited():
    rows = [cohort(MEASURED_MAXIMUM)]
    checked = read_cohort(rows[0], SMALL, MEASURED_MAXIMUM, 64 * 1024**2)
    assert len(checked["asset_ids"]) == MEASURED_MAXIMUM
    grid = ActionGrid.fit(np.linspace(-0.1, 0.1, 101), source_sha256=SOURCE, partition="train")
    prediction = importlib.import_module("mars_titan.environments.prediction")
    env = prediction.CausalPredictionEnv(
        lambda position: rows[position] if position < len(rows) else None,
        source_sha256=SOURCE,
        grid=grid,
        shapes=SMALL,
        max_assets=MEASURED_MAXIMUM,
    )
    observed, info = env.reset(seed=0)
    assert observed["active"].sum() == MEASURED_MAXIMUM == len(info["asset_ids"])
    _, reward, done, _, info = env.step(np.full(MEASURED_MAXIMUM, 10, dtype=np.int64))
    credited = {event["asset_id"] for event in info["matured"]}
    assert done and credited == set(checked["asset_ids"]) and np.isfinite(reward)


def test_action_grid_accepts_a_full_cohort_of_distributions_and_targets():
    grid = ActionGrid.fit(np.linspace(-0.1, 0.1, 101), source_sha256=SOURCE, partition="train")
    probabilities = np.full((MAX_COHORT_ASSETS, 21), 1 / 21)
    targets = np.zeros(MAX_COHORT_ASSETS)
    assert grid.median(probabilities).shape == (MAX_COHORT_ASSETS,)
    assert grid.expected_loss(probabilities, targets).shape == (MAX_COHORT_ASSETS,)
    assert grid.saturation(targets)["samples"] == MAX_COHORT_ASSETS
    with pytest.raises(ValueError):
        grid.saturation(np.zeros(MAX_COHORT_ASSETS + 1))


def test_cohort_above_the_contract_fails_instead_of_dropping_assets():
    with pytest.raises(ValueError, match="población"):
        read_cohort(cohort(MEASURED_MAXIMUM + 1), SMALL, MEASURED_MAXIMUM, 64 * 1024**2)
