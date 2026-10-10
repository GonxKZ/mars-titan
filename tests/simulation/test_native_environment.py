"""Paridad de decisiones y recuperación a través de la interfaz C."""

import numpy as np
import pytest
import torch

from mars_titan.episodes.worlds import WorldConfig, generate_world
from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.native_runtime import NativeLibrary, library_path, load_library
from mars_titan.simulation.training import FinancialTrainer, TrainConfig
from tests.simulation import native_library
from tests.simulation.native_library import requires_native_library, simulator_path


def tape():
    world = generate_world(WorldConfig(assets=16, sessions=28, context=8))
    return MarketTape.from_world(world, lambda inputs: inputs["news"][:, 0] * 0.002)


@requires_native_library
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


@requires_native_library
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


def test_a_declared_library_is_never_treated_as_missing(tmp_path, monkeypatch):
    declared = tmp_path / "libmars_titan_simulation.so"
    monkeypatch.setenv("MARS_TITAN_NATIVE_LIBRARY", str(declared))
    assert library_path() == str(declared)
    assert library_path(tmp_path / "explicit.so") == tmp_path / "explicit.so"
    with pytest.raises(FileNotFoundError):
        load_library()


def resolved_simulator():
    # Una omisión dentro de la prueba la marcaría como omitida y escondería el defecto.
    try:
        return simulator_path()
    except pytest.skip.Exception as error:
        pytest.fail(f"La resolución de mars-titan-sim se omitió: {error}")


def test_a_declared_simulator_is_never_treated_as_missing(tmp_path, monkeypatch):
    declared = tmp_path / "mars-titan-sim"
    monkeypatch.setenv("MARS_TITAN_SIM_EXECUTABLE", str(declared))
    monkeypatch.setattr(native_library, "SIMULATOR", tmp_path / "compiled")
    (tmp_path / "compiled").write_text("")
    with pytest.raises(FileNotFoundError):
        resolved_simulator()
    declared.write_text("")
    assert resolved_simulator() == declared


def test_without_declaration_the_compiled_simulator_is_used_or_the_test_is_skipped(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MARS_TITAN_SIM_EXECUTABLE", raising=False)
    compiled = tmp_path / "mars-titan-sim"
    monkeypatch.setattr(native_library, "SIMULATOR", compiled)
    with pytest.raises(pytest.skip.Exception, match="MARS_TITAN_SIM_EXECUTABLE"):
        simulator_path()
    compiled.write_text("")
    assert resolved_simulator() == compiled
