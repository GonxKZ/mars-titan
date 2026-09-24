"""Acciones financieras comunes, señales temporales y pausas recuperables."""

import numpy as np
import pytest

from mars_titan.episodes.worlds import WorldConfig, generate_world
from mars_titan.simulation.environment import FinancialEnv, RecoverablePause
from mars_titan.simulation.market import MarketTape


def tape():
    world = generate_world(WorldConfig(assets=4, sessions=20, context=8))
    return MarketTape.from_world(world, lambda inputs: inputs["news"][:, 0] * 0.002)


def test_financial_actions_persist_and_pause_restores_without_a_transition():
    source = tape()
    first, second = FinancialEnv(source), FinancialEnv(source)
    obs, _ = first.reset(seed=42)
    assert first.action_space.n == 6
    assert np.isfinite(obs).all()
    first.step(5)
    first.pause()
    checkpoint = first.snapshot()
    with pytest.raises(RecoverablePause):
        first.step(0)
    assert first.snapshot() == checkpoint
    second.restore(checkpoint)
    first.resume()
    second.resume()
    a, b = first.step(0), second.step(0)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1:] == b[1:]


def test_end_of_tape_truncates_without_forced_liquidation():
    env = FinancialEnv(tape(), cost_bps=0)
    env.reset(seed=42)
    for _ in range(len(env.tape) - 1):
        _, reward, terminated, truncated, info = env.step(5)
        assert np.isfinite(reward)
    assert not terminated and truncated
    assert info["reason"] == "episode_limit"
    assert info["reward_valid"]


def test_real_market_requires_adjustment_and_corporate_action_evidence():
    source = tape()
    with pytest.raises(ValueError, match="procedencia"):
        MarketTape(
            source.prices,
            source.close_times,
            source.assets,
            source.scores,
            domain="real",
            currency="USD",
        )


def test_real_market_does_not_invent_prediction_availability():
    source = tape()
    with pytest.raises(ValueError, match="predicciones"):
        MarketTape(
            source.prices,
            source.close_times,
            source.assets,
            source.scores,
            domain="real",
            currency="USD",
            open_times=source.open_times,
            audit={
                "price_basis": "unadjusted",
                "corporate_actions_complete": True,
                "evidence_sha256": "a" * 64,
            },
        )


def test_asset_permutation_preserves_actions_and_accounting():
    source = tape()
    order = np.array([3, 1, 0, 2])
    permuted = MarketTape(
        source.prices[:, order],
        source.close_times,
        [source.assets[i] for i in order],
        source.scores[:, order],
        domain="synthetic",
        currency="USD",
    )
    first, second = FinancialEnv(source), FinancialEnv(permuted)
    np.testing.assert_array_equal(first.reset()[0], second.reset()[0])
    for action in (5, 0, 1):
        a, b = first.step(action), second.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]


def test_restore_rejects_a_reopened_terminal_state():
    env = FinancialEnv(tape(), cost_bps=0)
    env.reset(seed=42)
    for _ in range(len(env.tape) - 1):
        env.step(0)
    state = env.snapshot()
    state["done"] = False
    with pytest.raises(ValueError, match="final"):
        FinancialEnv(env.tape, cost_bps=0).restore(state)


def test_fractional_future_timestamp_cannot_be_rounded_into_the_past():
    source = tape()
    with pytest.raises(ValueError, match="enteros"):
        MarketTape(
            source.prices,
            source.close_times,
            source.assets,
            source.scores,
            domain="synthetic",
            currency="USD",
            prediction_times=source.close_times.astype(float) + 0.5,
        )
