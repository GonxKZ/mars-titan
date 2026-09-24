"""Coherencia temporal, contable y multimodal de mundos identificados como ficticios."""

import numpy as np
import pytest

from mars_titan.episodes.worlds import WorldConfig, generate_world


def test_world_is_reproducible_and_targets_follow_prices():
    config = WorldConfig(assets=6, sessions=40, context=8, seed=42)
    world = generate_world(config)
    repeated = generate_world(config)
    np.testing.assert_array_equal(world.prices, repeated.prices)
    assert world.identity == repeated.identity
    at = 12
    cohort = world.cohort(at)
    expected = world.prices[at + 1, :, 3] / world.prices[at + 1, :, 0] - 1
    expected -= world.market_returns[at + 1] * world.market_loading[at + 1]
    np.testing.assert_allclose(cohort["target"], expected, rtol=0, atol=1e-14)
    assert (cohort["available_at"] <= cohort["prediction_at"]).all()
    assert (cohort["target_available_at"] > cohort["prediction_at"]).all()
    assert set(cohort["inputs"]) == {"prices", "news", "charts", "fundamentals", "macro"}


def test_future_changes_do_not_change_current_modalities():
    world = generate_world(WorldConfig(assets=4, sessions=40, context=8, seed=44))
    before = world.cohort(12)
    world.prices[13:] *= 3
    after = world.cohort(12)
    for key in before["inputs"]:
        np.testing.assert_array_equal(before["inputs"][key], after["inputs"][key])
    np.testing.assert_allclose(world.assets, world.liabilities + world.equity)
    assert (world.fundamental_available_at >= world.fundamental_period_end).all()
    assert all("ficticia" in text for text in world.events(12))


def test_null_and_known_signal_scenarios_have_distinct_mechanisms():
    world = generate_world(WorldConfig(assets=32, sessions=512, context=8, seed=42, signal=0.01))
    null = generate_world(WorldConfig(assets=32, sessions=512, context=8, seed=42, signal=0))
    at = np.arange(8, 511)
    residual = world.prices[at + 1, :, 3] / world.prices[at + 1, :, 0] - 1
    residual -= world.market_returns[at + 1, None] * world.market_loading[at + 1, None]
    residual_null = null.prices[at + 1, :, 3] / null.prices[at + 1, :, 0] - 1
    residual_null -= null.market_returns[at + 1, None] * null.market_loading[at + 1, None]
    np.testing.assert_allclose(residual - residual_null, 0.01 * world.signal[at], atol=1e-14)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"assets": 0},
        {"sessions": 1000000},
        {"signal": float("nan")},
        {"partition": "test"},
        {"context": 1000},
    ],
)
def test_world_rejects_invalid_or_unbounded_configurations(kwargs):
    with pytest.raises(ValueError):
        generate_world(WorldConfig(**kwargs))
