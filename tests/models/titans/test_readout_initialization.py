"""Arranque del refinador en el padre, sin cambiar la ruta anterior ni el RNG."""

import io

import pytest
import torch
from test_episodic_readout import CODEC, CONTEXT, api, source


def build(
    initialization="pytorch_default", *, device="cpu", dtype=torch.float64, steps=1, mode="bank"
):
    return api().EpisodicReadout(
        api().EpisodicReadoutConfig(
            CODEC,
            hidden_size=32,
            refinements=steps,
            mode=mode,
            refinement_initialization=initialization,
        ),
        device=device,
        dtype=dtype,
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("steps", [1, 2, 4])
@pytest.mark.parametrize(("mode", "size"), [("bank", 3), ("bank", 0), ("no_bank", 0)])
def test_zero_refinement_preserves_the_parent_and_has_a_trainable_first_layer(
    device, dtype, steps, mode, size
):
    model = build("zero_refinement", device=device, dtype=dtype, steps=steps, mode=mode)
    data = (
        api().EpisodeSnapshot.create(
            **source(size),
            cutoff=10,
            codec_id=CODEC,
            context_id=CONTEXT,
            device=device,
            dtype=dtype,
        )
        if mode == "bank"
        else None
    )
    state = torch.linspace(-0.8, 0.9, 64, device=device, dtype=dtype).reshape(2, 32)
    result = model(state, data, context_id=CONTEXT, cutoff=10, differentiable=True)
    assert torch.equal(result.state, state)
    result.state.square().sum().backward()
    for parameter in model.refinement.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    for projection in (model.query_projection, model.value_projection):
        gradient = projection.weight.grad
        assert gradient is None or torch.count_nonzero(gradient) == 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.refinement.weight.add_(0.01)
    changed = model(state, data, context_id=CONTEXT, cutoff=10, differentiable=True)
    changed.state.square().sum().backward()
    assert not torch.equal(changed.state, state)
    if size:
        assert torch.count_nonzero(model.value_projection.weight.grad) > 0
        assert torch.count_nonzero(model.query_projection.weight.grad) > 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_reader_keeps_the_frozen_parent_quantiles_and_prediction(device, dtype):
    from dataclasses import replace

    from test_financial_adapter import raw_batch, setup

    from mars_titan.models.titans.financial import DecisionBatch, FinancialPredictor

    original, _ = setup()
    parent = (
        FinancialPredictor(
            replace(original.config, head="quantile_head_v1"), device=device, dtype=dtype
        )
        .eval()
        .requires_grad_(False)
    )
    batch = DecisionBatch.from_corpus(raw_batch(), parent.config.inputs, device=device, dtype=dtype)
    prepared = parent.prepare(batch, parent.initial_state(batch.flow_ids))
    model = build("zero_refinement", device=device, dtype=dtype)
    data = api().EpisodeSnapshot.create(
        **source(3),
        cutoff=batch.prediction_at[0],
        codec_id=CODEC,
        context_id=CONTEXT,
        device=device,
        dtype=dtype,
    )
    result = api().apply_episodic_readout(
        prepared, parent.head, model, data, context_id=CONTEXT, cutoff=batch.prediction_at[0]
    )
    assert torch.equal(result.point_predictions, prepared.point_predictions)
    assert torch.equal(result.quantiles, prepared.quantiles)


def test_initialization_keeps_projection_weights_rng_and_legacy_identity():
    before = torch.get_rng_state().clone()
    legacy = api().EpisodicReadout(
        api().EpisodicReadoutConfig(CODEC, hidden_size=32), dtype=torch.float64
    )
    explicit = build()
    zero = build("zero_refinement")
    assert torch.equal(torch.get_rng_state(), before)
    assert legacy.get_extra_state() == explicit.get_extra_state()
    assert "refinement_initialization" not in legacy.config.identity()
    assert zero.config.identity()["refinement_initialization"] == "zero_refinement"
    for name, value in legacy.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, explicit.state_dict()[name])
            if not name.startswith("refinement."):
                assert torch.equal(value, zero.state_dict()[name])


def test_zero_initialized_reader_recovers_its_changed_weights_and_rejects_another_contract():
    model = build("zero_refinement")
    with torch.no_grad():
        model.refinement.bias.fill_(0.125)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    recovered = build("zero_refinement")
    recovered.load_state_dict(saved)
    assert torch.equal(recovered.refinement.bias, model.refinement.bias)
    with pytest.raises(ValueError, match="configuración|contrato"):
        build().load_state_dict(saved)


@pytest.mark.parametrize("value", [None, True, "zero", [], 0])
def test_unknown_initialization_is_rejected(value):
    with pytest.raises(ValueError, match="inicialización"):
        build(value)
