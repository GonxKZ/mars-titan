"""Integración del codificador de ventana sin alterar el estado exterior de Titans."""

import copy
from dataclasses import replace

import pytest
import torch
from test_financial_adapter import api, raw_batch, setup


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_disabled_encoder_keeps_checkpoint_rng_and_predictions(device):
    original, batch = setup("transformer_direct")
    explicit = api().FinancialPredictor(
        replace(original.config, temporal_encoder=None), dtype=torch.float64, device=device
    )
    original.to(device)
    batch = api().DecisionBatch.from_corpus(
        raw_batch(), original.config.inputs, dtype=torch.float64, device=device
    )
    assert explicit.config.identity() == original.config.identity()
    assert "temporal_encoder" not in explicit.config.identity()
    explicit.load_state_dict(copy.deepcopy(original.state_dict()))
    left = original.prepare(batch, original.initial_state(batch.flow_ids))
    right = explicit.prepare(batch, explicit.initial_state(batch.flow_ids))
    torch.testing.assert_close(left.point_predictions, right.point_predictions, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["kronlstm_v1", "vector_delta_v1"])
@pytest.mark.parametrize("variant", ["transformer_direct", "mac_online"])
def test_kron_encoder_preserves_shared_initialization_and_separates_identity(kind, variant):
    original, batch = setup(variant)
    before = torch.random.get_rng_state().clone()
    config = replace(original.config, temporal_encoder=kind)
    model = api().FinancialPredictor(config, dtype=torch.float64)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert config.identity()["temporal_encoder"] == kind
    assert model.get_extra_state()["backbone_contract"]["state_policy"] == "reset_per_window"
    for name, value in original.named_parameters():
        if not name.startswith("price_encoder."):
            torch.testing.assert_close(dict(model.named_parameters())[name], value, rtol=0, atol=0)
    state = model.initial_state(batch.flow_ids)
    result = model.prepare(batch, state, differentiable=True)
    assert result.point_predictions.shape == (2,)
    assert state.observed_steps.tolist() == [0, 0]
    assert result.next_state.observed_steps.tolist() == [1, 1]
    gradient = torch.autograd.grad(
        result.point_predictions.sum(), model.price_encoder.projection.weight
    )[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
    restored = api().FinancialPredictor(config, dtype=torch.float64)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    actual = restored.prepare(batch, restored.initial_state(batch.flow_ids))
    torch.testing.assert_close(actual.point_predictions, result.point_predictions, rtol=0, atol=0)
    with pytest.raises(ValueError):
        original.load_state_dict(model.state_dict())


@pytest.mark.parametrize("kind", [False, "kron", "", 1])
def test_unknown_temporal_encoder_fails_at_configuration(kind):
    original, _ = setup("transformer_direct")
    with pytest.raises(ValueError, match="codificador|temporal_encoder"):
        replace(original.config, temporal_encoder=kind)


@pytest.mark.parametrize("kind", ["kronlstm_v1", "vector_delta_v1"])
def test_declared_encoder_reaches_the_existing_predictor_factory(kind):
    from mars_titan.training.search_cases import checked_predictor_changes

    assert checked_predictor_changes({"temporal_encoder": kind}) == {"temporal_encoder": kind}


def test_active_encoder_adds_its_real_sources_to_the_frozen_manifest():
    from mars_titan.models.titans.frozen_financial import _implementation

    baseline = _implementation()
    active = _implementation(kronlstm=True)
    expected = {
        "mars_titan.models.kronlstm.rule",
        "mars_titan.models.kronlstm.price_encoder",
    }
    assert expected.isdisjoint(baseline)
    assert expected <= active.keys()
    assert all(len(active[name]) == 64 for name in expected)


def test_shared_price_input_projection_keeps_the_paired_initial_weights():
    original, _ = setup("transformer_direct")
    model = api().FinancialPredictor(
        replace(original.config, temporal_encoder="kronlstm_v1"), dtype=torch.float64
    )
    for name, value in original.price_encoder.projection.named_parameters():
        actual = dict(model.price_encoder.projection.named_parameters())[name]
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
        assert actual.data_ptr() != value.data_ptr()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("tamper", ["rule", "norm", "convolution"])
def test_frozen_encoder_verifies_predictions_and_rejects_changed_computation(device, tamper):
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

    original, _ = setup("transformer_direct")
    config = replace(original.config, temporal_encoder="kronlstm_v1")
    model = api().FinancialPredictor(config, dtype=torch.float64, device=device)
    model.eval().requires_grad_(False)
    batch = api().DecisionBatch.from_corpus(
        raw_batch(), config.inputs, dtype=torch.float64, device=device
    )
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        frozen = FrozenFinancialConsumer(model)
        state = model.initial_state(batch.flow_ids)
        expected = model.prepare(batch, state)
        actual = frozen.prepare(batch, state, context_id="c" * 64)
        torch.testing.assert_close(
            actual.point_predictions, expected.point_predictions, rtol=0, atol=0
        )
        assert state.observed_steps.tolist() == [0, 0]
        assert actual.next_state.observed_steps.tolist() == [1, 1]
        if tamper == "rule":
            model.price_encoder.blocks[0].rule = "vector"
        elif tamper == "norm":
            model.price_encoder.blocks[0].attn_norm.eps = 0.1
        else:
            model.price_encoder.blocks[0].qkv_conv1d.padding = (1,)
        with pytest.raises(ValueError, match="cálculo|identidad"):
            frozen.verify(strong=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
