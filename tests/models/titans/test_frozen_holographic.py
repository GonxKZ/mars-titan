"""H1, H2 y H3 en el consumidor congelado, sin entrenamiento ni optimizador."""

import io
from dataclasses import replace

import pytest
import torch
from test_frozen_financial import bank
from test_frozen_financial import frozen_backend as frozen_backend
from test_holographic_network import H1, configs, raw_batch, setup

from mars_titan.models.titans import holographic_network
from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer, _execution_signature
from tests.models.titans.test_episodic_readout import CONTEXT, reader


def frozen(config=H1, dtype=torch.float64):
    model, batch = setup(dtype=dtype, holographic_network=config)
    model.eval().requires_grad_(False)
    return FrozenFinancialConsumer(model), batch


@pytest.mark.parametrize("config", configs(), ids=lambda c: f"{c.kind}/{c.areas}")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_preparation_matches_the_predictor_and_keeps_one_memory_update(config, dtype):
    engine, batch = frozen(config, dtype)
    state = engine.predictor.initial_state(batch.flow_ids)
    expected = engine.predictor.prepare(batch, state)
    rng = torch.get_rng_state().clone()
    actual = engine.prepare(batch, state, context_id=CONTEXT)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    assert actual.next_state.observed_steps.tolist() == [1] * len(batch.flow_ids)
    assert actual.next_state.mac.memory.steps.tolist() == [1] * len(batch.flow_ids)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not actual.point_predictions.requires_grad
    assert all(p.grad is None for p in engine.predictor.parameters())


@pytest.mark.parametrize("k", [1, 2, 4])
def test_readout_refinements_do_not_repeat_the_mac_update(k):
    engine, batch = frozen()
    extension = reader(refinements=k).eval().requires_grad_(False)
    consumer = FrozenFinancialConsumer(engine.predictor, readout=extension)
    state = engine.predictor.initial_state(batch.flow_ids)
    result = consumer.prepare(batch, state, context_id=CONTEXT, snapshot=bank(batch))
    assert len(result.readout.reads) == k
    assert result.next_state.mac.memory.steps.tolist() == [1] * len(batch.flow_ids)


def test_checkpoint_recovery_preserves_the_next_prediction_and_consumer_identity():
    engine, batch = frozen()
    archive = io.BytesIO()
    torch.save(engine.predictor.state_dict(), archive)
    clone, _ = setup(holographic_network=H1)
    archive.seek(0)
    clone.load_state_dict(torch.load(archive, weights_only=True, map_location="cpu"))
    clone.eval().requires_grad_(False)
    recovered = FrozenFinancialConsumer(clone)
    assert recovered.identity() == engine.identity()
    a = engine.prepare(batch, engine.predictor.initial_state(batch.flow_ids), context_id=CONTEXT)
    b = recovered.prepare(batch, clone.initial_state(batch.flow_ids), context_id=CONTEXT)
    torch.testing.assert_close(a.point_predictions, b.point_predictions, rtol=0, atol=0)


def test_holographic_registration_is_explicit_for_each_consumer_recipe():
    engine, _ = frozen()
    models = (("predictor", engine.predictor),)
    with pytest.raises(ValueError):
        _execution_signature(models)


def test_a_changed_configuration_is_detected_before_preparation():
    engine, _ = frozen()
    network = engine.predictor.holographic_network
    network.config = replace(network.config, local_gain=0.4)
    with pytest.raises(ValueError):
        engine.verify(strong=False)


def test_installed_graph_configuration_must_match_the_predictor_declaration():
    engine, _ = frozen()
    network = engine.predictor.holographic_network
    network.config = replace(network.config, local_gain=0.4)
    with pytest.raises(ValueError):
        FrozenFinancialConsumer(engine.predictor)


@pytest.mark.parametrize("name", ["hidden_size", "max_batch"])
def test_installed_graph_dimensions_must_match_the_predictor_declaration(name):
    engine, _ = frozen()
    network = engine.predictor.holographic_network
    setattr(network, name, getattr(network, name) + 1)
    with pytest.raises(ValueError):
        FrozenFinancialConsumer(engine.predictor)


@pytest.mark.parametrize("name", ["hidden_size", "max_batch"])
def test_changed_graph_dimensions_are_detected_by_the_cheap_check(name):
    engine, _ = frozen()
    network = engine.predictor.holographic_network
    setattr(network, name, getattr(network, name) + 1)
    with pytest.raises(ValueError):
        engine.verify(strong=False)


@pytest.mark.parametrize("k", [None, 1, 2, 4])
def test_area_predictions_are_exposed_with_their_pre_readout_scope(k):
    engine, batch = frozen(replace(H1, area_predictions=0.5))
    extension = None if k is None else reader(refinements=k).eval().requires_grad_(False)
    engine = FrozenFinancialConsumer(engine.predictor, readout=extension)
    state = engine.predictor.initial_state(batch.flow_ids)
    expected = engine.predictor.prepare(batch, state)
    result = engine.prepare(
        batch, state, context_id=CONTEXT, snapshot=None if k is None else bank(batch)
    )
    torch.testing.assert_close(
        result.pre_readout_area_predictions, expected.area_predictions, rtol=0, atol=0
    )
    torch.testing.assert_close(result.pre_readout_area_votes.votes, expected.area_votes.votes)
    assert not result.pre_readout_area_predictions.requires_grad
    assert engine.identity()["area_predictions_scope"] == "pre_episodic_readout"
    warmup = engine.prepare(batch, state, context_id=CONTEXT, warmup=True)
    assert warmup.pre_readout_area_predictions is None and warmup.pre_readout_area_votes is None


def test_raw_buffer_changes_require_the_strong_boundary_check():
    engine, _ = frozen()
    before = engine.predictor.holographic_network.edges._version
    engine.predictor.holographic_network.edges.data.logical_not_()
    assert engine.predictor.holographic_network.edges._version == before
    engine.verify(strong=False)
    with pytest.raises(ValueError):
        engine.verify(strong=True)


def test_holographic_source_is_registered_only_when_present_and_checked_strongly(
    monkeypatch, tmp_path
):
    engine, _ = frozen()
    source = "mars_titan.models.titans.holographic_network"
    assert source in engine.identity()["implementation"]
    engine.verify(strong=True)
    plain, _ = setup()
    plain.eval().requires_grad_(False)
    assert source not in FrozenFinancialConsumer(plain).identity()["implementation"]
    changed = tmp_path / "different_holographic_source.py"
    changed.write_text("# Fuente ficticia distinta para comprobar la huella.\n")
    monkeypatch.setattr(holographic_network, "__file__", str(changed))
    with pytest.raises(ValueError):
        engine.verify(strong=True)


@pytest.mark.cuda
@pytest.mark.parametrize("config", configs(), ids=lambda c: f"{c.kind}/{c.areas}")
def test_cuda_fp32_consumer_matches_its_predictor_without_a_cpu_fallback(config):
    if not torch.cuda.is_available():
        pytest.skip("Esta comprobación necesita cuda:0, no admite una sustitución CPU")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        pytest.fail("La comprobación CUDA necesita FP32 estricto con TF32 desactivado")
    model, _ = setup(dtype=torch.float32, holographic_network=config)
    model.to("cuda:0").eval().requires_grad_(False)
    batch = DecisionBatch.from_corpus(
        raw_batch(), model.config.inputs, device="cuda:0", dtype=torch.float32
    )
    engine = FrozenFinancialConsumer(model)
    state = model.initial_state(batch.flow_ids)
    expected = model.prepare(batch, state)
    result = engine.prepare(batch, state, context_id=CONTEXT)
    torch.testing.assert_close(result.point_predictions, expected.point_predictions, rtol=0, atol=0)
    assert result.point_predictions.device == torch.device("cuda:0")
