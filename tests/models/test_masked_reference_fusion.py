"""Fusión con presencia en las referencias, contrastada en CPU y sin optimizadores."""

import pytest
import torch

from mars_titan.data import input_policy
from mars_titan.models.baselines.multimodal import (
    MODALITIES,
    PRESENCE_FUSION,
    STRICT_FUSION,
    MultimodalReference,
)
from tests.models.titans.test_financial_adapter import DIMENSIONS
from tests.models.titans.test_financial_adapter import setup as financial_setup

KINDS = ("rnn", "lstm", "gru", "dlinear", "transformer")


def build(kind, *, fusion=PRESENCE_FUSION, layers=1, seed=7):
    torch.manual_seed(seed)
    options = (
        dict(transformer=dict(heads=2, feedforward_multiplier=2)) if kind == "transformer" else {}
    )
    return MultimodalReference(
        kind,
        DIMENSIONS,
        context=16,
        hidden_size=32,
        layers=layers,
        dropout=0.0,
        mask_fusion=fusion,
        **options,
    ).double()


def values(batch=4, *, seed=11):
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(
            (batch, 16, width) if name == "prices" else (batch, width),
            generator=generator,
            dtype=torch.float64,
        )
        for name, width in DIMENSIONS.items()
    }


def presence(batch=4):
    mask = torch.ones(batch, len(MODALITIES), dtype=torch.bool)
    mask[1, [1, 3]] = False
    mask[2, 4] = False
    mask[3, [1, 3, 4]] = False
    return mask


def zero_absent(inputs, mask):
    result = {name: value.clone() for name, value in inputs.items()}
    for index, name in enumerate(MODALITIES):
        if name != "prices":
            result[name][~mask[:, index]] = 0
    return result


def test_order_matches_the_presence_contract_of_the_reader_and_titans():
    assert MODALITIES == input_policy.MODALITIES


def test_reference_reproduces_the_financial_predictor_fusion_exactly():
    predictor, batch = financial_setup("transformer_direct")
    reference = MultimodalReference(
        "transformer",
        DIMENSIONS,
        context=64,
        hidden_size=32,
        layers=1,
        dropout=0.0,
        mask_fusion=PRESENCE_FUSION,
    ).double()
    shared = {"price_encoder", "encoders", "fusion", "head"}
    reference.load_state_dict(
        {k: v for k, v in predictor.state_dict().items() if k.split(".", 1)[0] in shared},
        strict=True,
    )
    assert not batch.presence.all()
    prepared = predictor.prepare(batch, predictor.initial_state(batch.flow_ids))
    # El modo coincide con el predictor para usar la misma ruta de TransformerEncoderLayer.
    reference.train(predictor.training)
    with torch.no_grad():
        actual = reference(batch.inputs, batch.presence)
    torch.testing.assert_close(actual, prepared.point_predictions, rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_absent_blocks_receive_no_gradient_and_their_fill_is_irrelevant(kind):
    model = build(kind)
    mask = presence()
    inputs = {name: value.requires_grad_() for name, value in zero_absent(values(), mask).items()}
    output = model(inputs, mask)
    output.square().sum().backward()
    for index, name in enumerate(MODALITIES[1:], 1):
        gradient = inputs[name].grad
        assert torch.count_nonzero(gradient[~mask[:, index]]) == 0
        assert gradient[mask[:, index]].abs().sum() > 0
    filled = {name: value.detach().clone() for name, value in inputs.items()}
    for index, name in enumerate(MODALITIES[1:], 1):
        filled[name][~mask[:, index]] = 1e6 * torch.randn_like(filled[name][~mask[:, index]])
    with torch.no_grad():
        torch.testing.assert_close(model(filled, mask), output.detach(), rtol=0, atol=0)


def test_an_always_absent_block_does_not_update_its_projection():
    model = build("gru")
    mask = presence()
    mask[:, 1] = False
    model(zero_absent(values(), mask), mask).sum().backward()
    news = list(model.encoders["news"].parameters())
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in news)
    assert model.encoders["macro"][0].weight.grad.abs().sum() > 0
    assert model.fusion[0].weight.grad[:, -len(MODALITIES) :].abs().sum() > 0


def test_presence_bits_enter_the_first_fusion_layer_after_the_masked_projections():
    model = build("lstm", layers=2)
    observed = {}
    handle = model.fusion[0].register_forward_pre_hook(
        lambda module, arguments: observed.update(value=arguments[0].detach().clone())
    )
    mask = presence()
    try:
        model(values(), mask)
    finally:
        handle.remove()
    fused = observed["value"]
    assert fused.shape == (4, 5 * 32 + 5)
    assert model.fusion[3].in_features == 32
    torch.testing.assert_close(fused[:, -5:], mask.double(), rtol=0, atol=0)
    for index in range(1, len(MODALITIES)):
        block = fused[:, index * 32 : (index + 1) * 32]
        assert torch.count_nonzero(block[~mask[:, index]]) == 0
    assert model.configuration["mask_fusion"] == PRESENCE_FUSION


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("layers", [1, 2])
def test_disabled_option_keeps_the_strict_construction_and_contract(kind, layers):
    explicit, default = build(kind, fusion=STRICT_FUSION, layers=layers), None
    torch.manual_seed(7)
    options = (
        dict(transformer=dict(heads=2, feedforward_multiplier=2)) if kind == "transformer" else {}
    )
    default = MultimodalReference(
        kind, DIMENSIONS, context=16, hidden_size=32, layers=layers, dropout=0.0, **options
    ).double()
    assert explicit.state_dict().keys() == default.state_dict().keys()
    for key, value in default.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(explicit.state_dict()[key], value)
    assert default.fusion[0].in_features == 5 * 32
    assert "mask_fusion" not in default.configuration
    inputs = values()
    with torch.no_grad():
        torch.testing.assert_close(explicit(inputs), default(inputs), rtol=0, atol=0)
    with pytest.raises(ValueError, match="presencia"):
        default(inputs, presence())


@pytest.mark.parametrize(
    "fault",
    [
        None,
        lambda mask: mask.double(),
        lambda mask: mask[:, :4],
        lambda mask: mask[:3],
        lambda mask: mask.tolist(),
    ],
)
def test_presence_fusion_requires_aligned_boolean_bits(fault):
    model = build("dlinear")
    mask = presence()
    with pytest.raises(ValueError, match="presencia"):
        model(values(), None if fault is None else fault(mask))


@pytest.mark.parametrize("fusion", ["presence", "", None, True])
def test_unknown_fusion_is_rejected_before_allocating(fusion):
    with pytest.raises(ValueError, match="fusión"):
        build("gru", fusion=fusion)
