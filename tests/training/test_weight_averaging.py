"""Media exponencial de pesos de F2 con valores fijados a mano, sin optimizador."""

import math

import pytest
import torch
from torch import nn

from mars_titan.training.weight_averaging import WEIGHT_EMA, ShadowWeights, WeightAverage, applied


def model():
    torch.manual_seed(0)
    return nn.Linear(3, 2, dtype=torch.float64)


def test_each_update_follows_the_declared_recursion():
    layer = model()
    named = dict(layer.named_parameters())
    shadow = ShadowWeights(WeightAverage(decay=0.9), named)
    expected = {name: value.detach().clone() for name, value in named.items()}
    for step in range(3):
        with torch.no_grad():
            for value in named.values():
                value.add_(0.5 * (step + 1))
        shadow.update(named)
        for name, value in named.items():
            expected[name] = 0.9 * expected[name] + 0.1 * value.detach()
    for name, value in zip(shadow.names, shadow.shadow, strict=True):
        torch.testing.assert_close(value, expected[name], rtol=1e-14, atol=1e-15)
    assert shadow.updates == 3


def test_applied_swaps_the_weights_and_restores_them_even_after_an_error():
    layer = model()
    named = dict(layer.named_parameters())
    shadow = ShadowWeights(WeightAverage(decay=0.5), named)
    own = {name: value.detach().clone() for name, value in named.items()}
    with torch.no_grad():
        layer.weight.add_(1.0)
    shadow.update(named)
    moved = layer.weight.detach().clone()
    with applied(named, shadow.emitted()):
        torch.testing.assert_close(layer.weight, 0.5 * own["weight"] + 0.5 * moved)
    assert torch.equal(layer.weight, moved)
    with pytest.raises(RuntimeError), applied(named, shadow.emitted()):
        raise RuntimeError("fallo durante la validación")
    assert torch.equal(layer.weight, moved) and torch.equal(layer.bias, own["bias"])
    with pytest.raises(ValueError, match="parámetros del predictor"):
        with applied(named, {"other": torch.zeros(2)}):
            pass
    with pytest.raises(ValueError, match="parámetros ajustados"):
        ShadowWeights(WeightAverage(), {})


def test_state_round_trips_and_rejects_another_configuration():
    layer = model()
    named = dict(layer.named_parameters())
    shadow = ShadowWeights(WeightAverage(), named)
    with torch.no_grad():
        layer.bias.add_(2.0)
    shadow.update(named)
    other = ShadowWeights(WeightAverage(), named)
    other.load_state_dict(shadow.state_dict())
    assert other.updates == 1
    assert all(torch.equal(a, b) for a, b in zip(other.shadow, shadow.shadow, strict=True))
    with pytest.raises(ValueError):
        ShadowWeights(WeightAverage(decay=0.99), named).load_state_dict(shadow.state_dict())
    broken = shadow.state_dict()
    broken["shadow"][0] = broken["shadow"][0].float()
    with pytest.raises(ValueError):
        other.load_state_dict(broken)


def test_identity_reports_the_half_life():
    identity = WeightAverage(decay=0.5).identity()
    assert identity["name"] == WEIGHT_EMA and math.isclose(identity["half_life_steps"], 1.0)


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.1, True, float("nan")])
def test_invalid_decays_are_rejected(decay):
    with pytest.raises(ValueError):
        WeightAverage(decay=decay)
