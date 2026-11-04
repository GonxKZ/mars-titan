"""Rangos de entrada en el lector episódico, con parámetros del padre congelados."""

import pytest
import torch

from mars_titan.models.predictive_adaptation import attach_adapters
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.session_ic_objective import SessionICObjective
from mars_titan.models.titans.session_ranks import SESSION_RANK_INPUTS
from mars_titan.posttraining import chronological_matrix
from mars_titan.posttraining.readout_adapters import ReadoutAdapterTrainer
from mars_titan.training import mars_titan_run
from tests.posttraining.test_readout_adapters import MATRIX, named, spec
from tests.training.test_financial_run import RecordingOptimizer, StopAtStep
from tests.training.test_financial_run_session_objective import streams
from tests.training.test_mars_titan_run import build, reader, recipe
from tests.training.test_mars_titan_run import native as native


@pytest.fixture(autouse=True)
def permitted_doubles(learning_doubles):
    """El recorrido usa el registrador. El gancho global sigue impidiendo pasos de PyTorch."""


@pytest.fixture(autouse=True)
def explicit_backend():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return streams(tmp_path_factory.mktemp("ranked-reader"))


def parent(data, **changes):
    config = FinancialConfig(
        data["train"].specification(),
        variant="mac_online",
        hidden_size=32,
        session_rank_inputs=SESSION_RANK_INPUTS,
        **changes,
    )
    return FinancialPredictor(config, dtype=torch.float64).eval().requires_grad_(False)


@pytest.mark.parametrize("admission", ["m0", "m1"])
def test_ranked_reader_runs_and_recovers_without_changing_the_parent(
    data, tmp_path, native, admission
):
    plan = recipe(block_rows=8, loss="mae")
    direct = build(
        data, tmp_path / "direct", native, model=parent(data), plan=plan, admission=admission
    )
    fingerprint = direct.predictor._parameter_id
    expected = direct.run()
    assert expected["status"] == "completed"
    assert direct.predictor._parameter_id == fingerprint
    resumed = build(
        data, tmp_path / "resumed", native, model=parent(data), plan=plan, admission=admission
    )
    stop = StopAtStep(2)
    stop.trainer = resumed
    paused = resumed.run(stop=stop)
    assert paused["status"] == "paused"
    restored = resumed.run(resume=True)
    assert restored["status"] == "completed"
    left, right = [], []
    direct.evaluate(data["validation"], rows=left)
    resumed.evaluate(data["validation"], rows=right)
    assert left and left == right
    assert resumed.predictor._parameter_id == fingerprint


def test_ranked_reader_rejects_a_fragmented_session(data, tmp_path, native):
    runner = build(
        data, tmp_path / "split", native, model=parent(data), plan=recipe(block_rows=2, loss="mae")
    )
    with pytest.raises(ValueError, match="sesión completa"):
        runner.run()


def test_reader_rejects_an_objective_that_would_reinterpret_the_parent_score(
    data, tmp_path, native
):
    with pytest.raises(ValueError, match="objetivo de salida"):
        build(
            data,
            tmp_path / "ic",
            native,
            model=parent(data, session_ic_objective=SessionICObjective()),
        )


def adapter_runner(data, output, native, *, core_rows=None, update_instants=3):
    predictor = parent(data)
    attach_adapters(
        predictor, chronological_matrix.titans_targets(MATRIX, spec("core")["core"]), seed=7
    )
    predictor._seal_parameters()
    extension, codec = reader(data, "m1")
    extension.requires_grad_(False)
    plan = recipe(block_rows=8, loss="mae", update_instants=update_instants)
    return ReadoutAdapterTrainer(
        predictor,
        extension,
        plan,
        posttraining=dict(kind="technical_check", arm="core"),
        core_rows=core_rows,
        admission="m1",
        retention=mars_titan_run.retention_config(plan, "m1"),
        native=native,
        codec=codec,
        train=data["train"],
        validation=data["validation"],
        output=output,
        world="fixture",
        fold="0",
        optimizer_factory=RecordingOptimizer,
        audit=True,
    )


@pytest.mark.parametrize("update_instants", [1, 3])
def test_ranked_core_adapters_repeat_complete_sessions_including_unlabelled_rows(
    data, tmp_path, native, update_instants
):
    complete = adapter_runner(data, tmp_path / "complete", native, update_instants=update_instants)
    bounded = adapter_runner(
        data, tmp_path / "bounded", native, core_rows=8, update_instants=update_instants
    )
    assert complete.run()["status"] == bounded.run()["status"] == "completed"
    left, right = named(complete), named(bounded)
    assert left and len(left) == len(right)
    for first, second in zip(left, right, strict=True):
        for name, gradient in first.items():
            if gradient is None:
                assert second[name] is None
            else:
                torch.testing.assert_close(second[name], gradient, rtol=1e-9, atol=1e-12)
    fragmented = adapter_runner(
        data, tmp_path / "fragmented", native, core_rows=2, update_instants=update_instants
    )
    with pytest.raises(ValueError, match="repetir todos los flujos"):
        fragmented.run()
