"""Rechazo de estados y transiciones incompletos sin perder el estado confirmado."""

import copy
import random

import numpy as np
import pytest
import torch

from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.portfolio import CorporateAction
from mars_titan.simulation.training import FinancialTrainer, TrainConfig
from tests.simulation.native_library import NATIVE_BACKEND


def market(*, actions=()):
    prices = np.empty((17, 1, 5), dtype=np.float64)
    closes = 10 + np.arange(17) * 0.1
    prices[:, 0, :4] = closes[:, None]
    prices[:, 0, 4] = 1000
    return MarketTape(
        prices,
        np.arange(1, 34, 2, dtype=np.int64),
        ["A"],
        np.full((17, 1), 0.01),
        domain="synthetic",
        currency="USD",
        actions=actions,
    )


def assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected, strict=True):
            assert_state_equal(first, second)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    ("algorithm", "corrupt"),
    [
        ("ppo", "network"),
        ("ppo", "optimizer"),
        ("ppo", "numpy_rng"),
        ("ppo", "sampling_rng"),
        ("ppo", "rng"),
        ("ppo", "episodes"),
        ("ppo", "environment"),
        ("double_dqn", "target"),
        ("double_dqn", "replay"),
        ("double_dqn", "rng"),
    ],
)
def test_rejected_checkpoint_preserves_complete_trainer_state(algorithm, corrupt):
    cuda_initialized = torch.cuda.is_initialized()
    config = TrainConfig(
        total_steps=16,
        batch_size=2,
        rollout_steps=4,
        ppo_epochs=1,
        replay_capacity=16,
        warmup_steps=2,
        target_interval=2,
    )
    trainer = FinancialTrainer(
        FinancialEnv(market()), algorithm, config, seed=42, device="cpu", diagnostic=True
    )
    for _ in range(5):
        trainer.advance()
    confirmed = trainer.snapshot()
    observation = trainer.observation.copy()
    for _ in range(3):
        trainer.advance()
    damaged = trainer.snapshot()
    trainer.restore(confirmed)
    damaged["rng"]["python"] = random.Random(91).getstate()
    if corrupt in {"network", "target"}:
        damaged[corrupt][next(iter(damaged[corrupt]))] = torch.ones(1)
    elif corrupt == "optimizer":
        damaged["optimizer"]["param_groups"] = []
    elif corrupt == "numpy_rng":
        damaged["numpy_rng"] = {"bit_generator": "invalid"}
    elif corrupt == "sampling_rng":
        damaged["sampling_rng"] = torch.zeros(1, dtype=torch.uint8)
    elif corrupt == "rng":
        damaged["rng"]["torch"] = torch.zeros(1, dtype=torch.uint8)
    elif corrupt == "episodes":
        del damaged["episodes"]
    elif corrupt == "environment":
        del damaged["environment"]["rng_seed"]
    else:
        damaged["replay"]["data"]["action"][0] = 6

    with pytest.raises((ValueError, RuntimeError, KeyError)):
        trainer.restore(damaged)

    assert_state_equal(trainer.snapshot(), confirmed)
    np.testing.assert_array_equal(trainer.observation, observation)
    # Otras pruebas de la suite pueden haber abierto CUDA antes de este diagnóstico CPU.
    assert torch.cuda.is_initialized() == cuda_initialized


@pytest.mark.parametrize("effective_at", [2, 4])
def test_market_rejects_duplicate_corporate_action_ids_before_trading(effective_at):
    actions = (
        CorporateAction("same", "A", "split", 2, 2.0, verified=True),
        CorporateAction("same", "A", "split", effective_at, 2.0, verified=True),
    )
    with pytest.raises(ValueError, match="duplicada"):
        market(actions=actions)


def test_market_rejects_corporate_actions_outside_its_calendar():
    action = CorporateAction("closed", "A", "split", 3, 2.0, verified=True)
    with pytest.raises(ValueError, match="calendario"):
        market(actions=(action,))


@pytest.mark.parametrize("backend", ["python", NATIVE_BACKEND])
def test_failed_corporate_action_preserves_pending_orders_and_cursor(backend):
    action = CorporateAction("overflow", "A", "split", 4, 1e308, verified=True)
    env = FinancialEnv(market(actions=(action,)), backend=backend)
    env.reset(seed=42)
    env.step(5)
    confirmed = env.snapshot()
    book = env.book

    with pytest.raises((ValueError, OverflowError, FloatingPointError)):
        env.step(1)

    assert env.snapshot() == confirmed
    assert env.book is book


@pytest.mark.parametrize("backend", ["python", NATIVE_BACKEND])
def test_failed_observation_preserves_the_next_successful_transition(backend, monkeypatch):
    env = FinancialEnv(market(), backend=backend)
    env.reset(seed=42)
    confirmed = env.snapshot()
    book = env.book
    reference = FinancialEnv(env.tape, backend=backend)
    reference.restore(copy.deepcopy(confirmed))

    def fail():
        raise RuntimeError("Fallo al construir la observación")

    with monkeypatch.context() as patch:
        patch.setattr(env, "_observation", fail)
        with pytest.raises(RuntimeError, match="observación"):
            env.step(5)

    assert env.snapshot() == confirmed
    assert env.book is book
    actual, expected = env.step(5), reference.step(5)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]
    assert env.snapshot() == reference.snapshot()
