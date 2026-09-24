"""Recuperación exacta de actualizaciones pequeñas, solo como diagnóstico en CPU."""

import pytest
import torch

from mars_titan.episodes.worlds import WorldConfig, generate_world
from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.training import FinancialTrainer, TrainConfig


def environment():
    world = generate_world(WorldConfig(assets=4, sessions=15, context=8))
    return FinancialEnv(MarketTape.from_world(world, lambda x: 0.002 * x["news"][:, 0]))


@pytest.mark.parametrize("algorithm", ["ppo", "double_dqn"])
def test_resume_preserves_next_update_and_final_parameters(tmp_path, algorithm):
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
    full = FinancialTrainer(
        environment(), algorithm, config, seed=42, device="cpu", diagnostic=True
    )
    result = full.run(tmp_path / "full")
    partial = FinancialTrainer(
        environment(), algorithm, config, seed=42, device="cpu", diagnostic=True
    )
    assert partial.run(tmp_path / "resumed", stop_after=3)["status"] == "paused"
    restored = FinancialTrainer(
        environment(), algorithm, config, seed=42, device="cpu", diagnostic=True
    )
    resumed = restored.run(tmp_path / "resumed", resume=True)
    assert result["global_step"] == resumed["global_step"] == 12
    for name, value in full.network.state_dict().items():
        torch.testing.assert_close(value, restored.network.state_dict()[name], rtol=0, atol=0)
    assert full.env.snapshot() == restored.env.snapshot()


def test_cpu_is_never_an_implicit_fallback():
    with pytest.raises(ValueError):
        FinancialTrainer(environment(), "ppo", TrainConfig(), seed=42, device="cpu")


def test_training_does_not_admit_validation_data():
    world = generate_world(WorldConfig(assets=4, sessions=15, context=8, partition="validation"))
    env = FinancialEnv(MarketTape.from_world(world, lambda x: 0.002 * x["news"][:, 0]))
    with pytest.raises(ValueError):
        FinancialTrainer(
            env, "ppo", TrainConfig(total_steps=8), seed=42, device="cpu", diagnostic=True
        )


def test_failure_before_first_step_can_resume(tmp_path, monkeypatch):
    config = TrainConfig(total_steps=4, batch_size=2, rollout_steps=4, replay_capacity=8)
    trainer = FinancialTrainer(environment(), "ppo", config, seed=42, device="cpu", diagnostic=True)
    original = trainer.advance

    def fail():
        raise OSError("Interrupción de prueba")

    monkeypatch.setattr(trainer, "advance", fail)
    with pytest.raises(OSError):
        trainer.run(tmp_path / "failed")
    monkeypatch.setattr(trainer, "advance", original)
    assert trainer.run(tmp_path / "failed", resume=True)["global_step"] == 4


def test_network_and_replay_do_not_depend_on_default_dtype():
    from mars_titan.simulation.algorithms import FinancialNetwork
    from mars_titan.simulation.replay import Replay

    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert next(FinancialNetwork(8).parameters()).dtype == torch.float32
        assert Replay(8, 8).data["observation"].dtype == torch.float32
    finally:
        torch.set_default_dtype(original)
