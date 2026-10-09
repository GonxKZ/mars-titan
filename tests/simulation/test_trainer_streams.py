"""El entrenador separa sus flujos aleatorios del mundo y rechaza cierres ausentes.

Las pruebas solo construyen el entrenador. No avanzan transiciones ni llaman al optimizador.
"""

import numpy as np
import pytest

from mars_titan.episodes.worlds import WorldConfig, generate_world
from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.training import FinancialTrainer, TrainConfig

SEED = 42


def environment(*, missing=False):
    world = generate_world(WorldConfig(assets=4, sessions=15, context=8, seed=SEED))
    tape = MarketTape.from_world(world, lambda x: 0.002 * x["news"][:, 0])
    if missing:
        prices = np.array(tape.prices)
        prices[3, 1, 3] = np.nan
        tape = MarketTape(
            prices, tape.close_times, tape.assets, tape.scores, domain="synthetic", currency="USD"
        )
    return FinancialEnv(tape)


def trainer(algorithm, **options):
    config = TrainConfig(total_steps=8, batch_size=2, rollout_steps=4, replay_capacity=16)
    return FinancialTrainer(
        environment(**options), algorithm, config, seed=SEED, device="cpu", diagnostic=True
    )


@pytest.mark.parametrize("algorithm", ["ppo", "double_dqn"])
def test_exploration_stream_differs_from_the_world_generator_with_the_same_seed(algorithm):
    first, second = trainer(algorithm), trainer(algorithm)
    world = np.random.default_rng(SEED)
    draws = first.rng.random(256)
    # Antes, la exploración repetía exactamente los números que generaron el mundo.
    assert not np.array_equal(draws, world.random(256))
    assert np.abs(np.corrcoef(draws, np.random.default_rng(SEED).random(256))[0, 1]) < 0.3
    np.testing.assert_array_equal(second.rng.random(256), draws)
    assert first.identity["rng_streams"] == dict(
        numpy="seed_sequence_spawn_v1", torch="manual_seed"
    )
    assert first.step_count == first.updates == 0


@pytest.mark.parametrize("algorithm", ["ppo", "double_dqn"])
def test_training_rejects_tapes_whose_missing_closes_could_censor_losses(algorithm):
    with pytest.raises(ValueError, match="completa"):
        trainer(algorithm, missing=True)
