"""Paridad de decisiones y recuperación a través de la interfaz C."""

import numpy as np
import pytest
import torch

from mars_titan.episodes.worlds import WorldConfig, generate_world
from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.native_runtime import NativeLibrary
from mars_titan.simulation.training import FinancialTrainer, TrainConfig


def tape():
    world = generate_world(WorldConfig(assets=16, sessions=28, context=8))
    return MarketTape.from_world(world, lambda inputs: inputs["news"][:, 0] * 0.002)


def test_native_environment_matches_every_observation_and_valuation():
    source = tape()
    reference = FinancialEnv(source)
    native = FinancialEnv(source, backend="native")
    np.testing.assert_array_equal(reference.reset(seed=42)[0], native.reset(seed=42)[0])
    actions = np.random.default_rng(43).integers(0, 6, len(source) - 1)
    for action in actions:
        expected, actual = reference.step(action), native.step(action)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == pytest.approx(expected[1], abs=1e-12)
        assert actual[2:] == expected[2:]
    assert native.book.execution_counts["native_steps"] == len(source) - 1


@pytest.mark.parametrize("algorithm", ["ppo", "double_dqn"])
def test_native_training_recovers_exactly(tmp_path, algorithm):
    config = TrainConfig(
        total_steps=12,
        batch_size=2,
        rollout_steps=4,
        ppo_epochs=2,
        replay_capacity=16,
        warmup_steps=2,
        target_interval=2,
        checkpoint_steps=4,
    )

    def trainer():
        return FinancialTrainer(
            FinancialEnv(tape(), backend="native"),
            algorithm,
            config,
            seed=42,
            device="cpu",
            diagnostic=True,
        )

    full, partial = trainer(), trainer()
    full.run(tmp_path / "full")
    partial.run(tmp_path / "resumed", stop_after=5)
    resumed = trainer()
    resumed.run(tmp_path / "resumed", resume=True)
    for key, value in full.network.state_dict().items():
        torch.testing.assert_close(value, resumed.network.state_dict()[key], rtol=0, atol=0)
    assert full.env.snapshot() == resumed.env.snapshot()


def test_unaligned_numpy_memory_is_rejected_before_casting():
    frame = np.ndarray((2, 5), dtype=np.float64, buffer=bytearray(81), offset=1)
    assert not frame.flags.aligned
    with pytest.raises(ValueError):
        NativeLibrary.frame(frame, 2)
