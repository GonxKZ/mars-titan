"""Rutas de gradiente de mediana y dispersión, con idéntica función de inferencia."""

import importlib

import pytest
import torch
from torch.nn import functional as F

from mars_titan.models.quantile_head import QuantileHead, pinball_loss

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]


def routing(median=True, spreads=True):
    return importlib.import_module("mars_titan.models.quantile_routing").QuantileRouting(
        median_from_tails=median, encoder_from_spreads=spreads
    )


def manual(value, weight, bias, median_feedback, spread_feedback):
    raw = F.linear(value, weight, bias)
    auxiliary = raw if spread_feedback else F.linear(value.detach(), weight, bias)
    point = raw[:, 2]
    location = point if median_feedback else point.detach()
    low = location - F.softplus(auxiliary[:, 1])
    high = location + F.softplus(auxiliary[:, 3])
    return torch.stack(
        (low - F.softplus(auxiliary[:, 0]), low, point, high, high + F.softplus(auxiliary[:, 4])),
        dim=-1,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "median_feedback,spread_feedback", [(True, True), (False, True), (True, False), (False, False)]
)
def test_routed_head_preserves_values_and_matches_the_declared_partial_derivatives(
    device, dtype, median_feedback, spread_feedback
):
    head = QuantileHead(
        3, device=device, dtype=dtype, routing=routing(median_feedback, spread_feedback)
    )
    value = torch.tensor(
        [[1.0, -2.0, 0.5], [-0.4, 0.6, 0.3]], device=device, dtype=dtype, requires_grad=True
    )
    target = torch.tensor([0.2, -0.1], device=device, dtype=dtype)
    expected = manual(value, head.weight, head.bias, median_feedback, spread_feedback)
    actual = head(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    original = manual(value, head.weight, head.bias, True, True)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    variables = (value, head.weight, head.bias)
    got = torch.autograd.grad(pinball_loss(actual, target), variables)
    want = torch.autograd.grad(pinball_loss(expected, target), variables)
    for left, right in zip(got, want, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(head(value), original, rtol=0, atol=0)


def test_complete_isolation_preserves_primary_feedback_and_keeps_spread_weights_trainable():
    head = QuantileHead(1, dtype=torch.float64, routing=routing(False, False))
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[0.0], [10.0], [1.0], [0.0], [0.0]]))
        head.bias.copy_(torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    x = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.2], dtype=torch.float64)
    q = head(x)
    gx, gw, gb = torch.autograd.grad(pinball_loss(q, y), (x, head.weight, head.bias))
    torch.testing.assert_close(gx, torch.full_like(x, 0.1))
    torch.testing.assert_close(gb[2], torch.tensor(0.1, dtype=gb.dtype))
    assert gb[[0, 1, 3, 4]].abs().sum() > 0
    # El enrutamiento es un gradiente sustituto. La diferencia finita de los valores
    # conserva -0,575 y no tiene por qué coincidir con el gradiente enrutado de 0,1.
    epsilon = 1e-6
    with torch.no_grad():
        finite = (pinball_loss(head(x + epsilon), y) - pinball_loss(head(x - epsilon), y)) / (
            2 * epsilon
        )
    torch.testing.assert_close(
        finite, torch.tensor(-0.575, dtype=finite.dtype), rtol=1e-8, atol=1e-8
    )
    assert abs(finite.item() - gx.item()) > 0.5
    assert torch.equal(gw, torch.zeros_like(gw))


@pytest.mark.parametrize("device", DEVICES)
def test_the_full_routing_control_has_the_original_graph_parameters_and_rng(device):
    torch.manual_seed(51)
    first = QuantileHead(4, device=device, dtype=torch.float64)
    rng = torch.random.get_rng_state()
    second = QuantileHead(4, device=device, dtype=torch.float64, routing=routing())
    # Se emparejan los pesos, sin hacer pasar dos inicializaciones aleatorias por una.
    second.load_state_dict(first.state_dict())
    torch.random.set_rng_state(rng)
    before = torch.random.get_rng_state()
    x = torch.arange(12, device=device, dtype=torch.float64).reshape(3, 4).requires_grad_()
    outputs = [head(x) for head in (first, second)]
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    gradients = [
        torch.autograd.grad(q.sum(), (x, *head.parameters()))
        for q, head in zip(outputs, (first, second), strict=True)
    ]
    for left, right in zip(*gradients, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.equal(before, torch.random.get_rng_state())
    assert set(first.state_dict()) == set(second.state_dict()) == {"weight", "bias"}


@pytest.mark.parametrize("changes", [dict(median_from_tails=1), dict(encoder_from_spreads=None)])
def test_non_boolean_routing_is_rejected(changes):
    cls = importlib.import_module("mars_titan.models.quantile_routing").QuantileRouting
    with pytest.raises(ValueError):
        cls(**changes)


def test_head_rejects_an_unidentified_routing_contract():
    with pytest.raises(ValueError):
        QuantileHead(4, routing={})


@pytest.mark.parametrize("name", ["quantile_head", "quantile_routing"])
def test_producer_and_frozen_consumer_bind_the_routing_implementation(tmp_path, monkeypatch, name):
    from mars_titan.models.titans.frozen_financial import _implementation
    from mars_titan.training.titans_walk_forward import _code

    module = importlib.import_module("mars_titan.models." + name)
    path = tmp_path / (name + ".py")
    path.write_text("revision one\n")
    monkeypatch.setattr(module, "__file__", str(path))
    first = (_code(), _implementation())
    path.write_text("revision two\n")
    second = (_code(), _implementation())
    key = "mars_titan.models." + name
    assert all(left[key] != right[key] for left, right in zip(first, second, strict=True))
