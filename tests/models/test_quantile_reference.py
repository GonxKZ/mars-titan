"""Referencias multimodales con `quantile_head_v1`, en CPU y sin optimizadores."""

import json

import pytest
import torch

from mars_titan.models.baselines.multimodal import (
    MASK_FUSIONS,
    PRESENCE_FUSION,
    SCALAR_HEAD,
    MultimodalReference,
)
from mars_titan.models.quantile_head import (
    MEDIAN_INDEX,
    QUANTILE_HEAD,
    QuantileHead,
    median,
    pinball_loss,
)
from tests.models.test_masked_reference_fusion import KINDS, presence, values
from tests.models.titans.test_financial_adapter import DIMENSIONS


def build(kind, *, head=None, fusion=PRESENCE_FUSION, layers=1, seed=7):
    torch.manual_seed(seed)
    options = (
        dict(transformer=dict(heads=2, feedforward_multiplier=2)) if kind == "transformer" else {}
    )
    if head is not None:
        options["head"] = head
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


def arguments(fusion):
    return (values(), presence()) if fusion == PRESENCE_FUSION else (values(),)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fusion", MASK_FUSIONS)
def test_quantile_variant_shares_the_scalar_trunk_and_emits_ordered_levels(kind, fusion):
    scalar, quantile = build(kind, fusion=fusion), build(kind, head=QUANTILE_HEAD, fusion=fusion)
    trunk = {k: v for k, v in scalar.state_dict().items() if not k.startswith("head.")}
    other = {k: v for k, v in quantile.state_dict().items() if not k.startswith("head.")}
    assert trunk.keys() == other.keys()
    for key, value in trunk.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other[key]), key
    assert isinstance(quantile.head, QuantileHead) and quantile.head.weight.shape == (5, 32)
    assert type(scalar.head) is torch.nn.Linear and scalar.head.weight.shape == (1, 32)
    with torch.no_grad():
        encoded = quantile.encode(*arguments(fusion))
        output = quantile(*arguments(fusion))
    assert output.shape == (4, 5) and torch.isfinite(output).all()
    assert (output.diff(dim=1) >= 0).all()
    torch.testing.assert_close(output, quantile.head(encoded), rtol=0, atol=0)
    torch.testing.assert_close(encoded, scalar.encode(*arguments(fusion)), rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_median_equals_the_scalar_output_with_the_same_median_weights(kind):
    scalar, quantile = build(kind), build(kind, head=QUANTILE_HEAD)
    with torch.no_grad():
        quantile.head.weight[MEDIAN_INDEX] = scalar.head.weight[0]
        quantile.head.bias[MEDIAN_INDEX] = scalar.head.bias[0]
        point = scalar(*arguments(PRESENCE_FUSION))
        output = quantile(*arguments(PRESENCE_FUSION))
    torch.testing.assert_close(median(output), point, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("layers", [1, 2])
def test_scalar_default_and_explicit_scalar_head_are_identical(kind, layers):
    default = build(kind, layers=layers)
    explicit = build(kind, head=SCALAR_HEAD, layers=layers)
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key, value in default.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, explicit.state_dict()[key])
    assert "head" not in default.configuration
    assert json.dumps(default.configuration, sort_keys=True) == json.dumps(
        explicit.configuration, sort_keys=True
    )
    with torch.no_grad():
        output = default(*arguments(PRESENCE_FUSION))
        torch.testing.assert_close(explicit(*arguments(PRESENCE_FUSION)), output, rtol=0, atol=0)
        encoded = default.encode(*arguments(PRESENCE_FUSION))
        torch.testing.assert_close(output, default.head(encoded).squeeze(-1), rtol=0, atol=0)
    assert output.shape == (4,)


def test_only_the_quantile_configuration_declares_its_head():
    scalar, quantile = build("gru"), build("gru", head=QUANTILE_HEAD)
    expected = {**scalar.configuration, "head": QUANTILE_HEAD}
    assert quantile.configuration == expected


@pytest.mark.parametrize("kind", KINDS)
def test_pinball_gradients_reach_head_trunk_and_observed_modalities(kind):
    model = build(kind, head=QUANTILE_HEAD)
    inputs = {name: value.requires_grad_() for name, value in values().items()}
    mask = presence()
    target = torch.linspace(-0.02, 0.02, 4, dtype=torch.float64)
    loss = pinball_loss(model(inputs, mask), target)
    loss.backward()
    assert model.head.weight.grad.abs().sum() > 0
    assert model.head.bias.grad[MEDIAN_INDEX] != 0
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for name, p in model.named_parameters()
        if not name.startswith("encoders.")
    )
    assert inputs["prices"].grad.abs().sum() > 0
    # Las filas sin noticias no deben recibir gradiente por esa modalidad.
    absent = ~mask[:, 1]
    assert torch.count_nonzero(inputs["news"].grad[absent]) == 0
    assert inputs["news"].grad[~absent].abs().sum() > 0


@pytest.mark.parametrize("head", ["quantile", "", None, True, "quantile_head_v2"])
def test_unknown_heads_are_rejected_before_allocating(head):
    with pytest.raises(ValueError, match="cabeza"):
        MultimodalReference("gru", DIMENSIONS, context=16, hidden_size=32, head=head)


def test_state_dict_cannot_load_across_heads():
    scalar, quantile = build("dlinear"), build("dlinear", head=QUANTILE_HEAD)
    with pytest.raises(RuntimeError, match="size mismatch"):
        quantile.load_state_dict(scalar.state_dict())
    with pytest.raises(RuntimeError, match="size mismatch"):
        scalar.load_state_dict(quantile.state_dict())
