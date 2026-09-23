"""Gradientes de política y controles de igual capacidad sobre padres congelados."""

import importlib

import numpy as np
import pytest
import torch

from mars_titan.data.embeddings import require_cuda


def module():
    return importlib.import_module("mars_titan.models.predictive_adaptation")


def grid(device):
    return torch.linspace(-0.1, 0.1, 21, dtype=torch.float64, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_enumerating_actions_matches_the_exact_expected_gradient(device):
    if device == "cuda:0":
        require_cuda()
    engine = module()
    values = grid(device)
    center = torch.tensor([-0.015, 0.012], dtype=torch.float64, device=device, requires_grad=True)
    target = torch.tensor([0.023, -0.004], dtype=torch.float64, device=device)
    probabilities = engine.gaussian_log_probabilities(center, values, 0.02).exp().detach()
    exact = engine.objective(center, target, values, 0.02, "expected")[0].sum()
    expected_gradient = torch.autograd.grad(exact, center, retain_graph=True)[0]
    enumerated = torch.zeros_like(center)
    for action in range(21):
        actions = torch.full((2,), action, dtype=torch.int64, device=device)
        loss, _ = engine.objective(center, target, values, 0.02, "reinforce", actions=actions)
        gradient = torch.autograd.grad(loss.sum(), center, retain_graph=True)[0]
        enumerated += probabilities[:, action] * gradient
    torch.testing.assert_close(enumerated, expected_gradient, rtol=1e-11, atol=1e-11)


def test_gaussian_policy_keeps_gradient_outside_grid_where_laplace_is_flat():
    values = grid("cpu")
    center = torch.tensor([0.13], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0.0], dtype=torch.float64)
    loss, _ = module().objective(center, target, values, 0.05, "expected")
    gradient = torch.autograd.grad(loss.sum(), center)[0]
    assert gradient.item() > 1e-4


def test_supervised_control_has_same_loss_scale_and_sampling_is_reproducible():
    device = require_cuda()
    values = grid(device)
    center = torch.tensor([0.0, 0.02], device=device, dtype=torch.float64)
    target = torch.tensor([0.04, -0.03], device=device, dtype=torch.float64)
    supervised, actions = module().objective(center, target, values, 0.02, "mae")
    assert actions is None
    torch.testing.assert_close(
        supervised, torch.tensor([2.0, 2.5], dtype=torch.float64, device=device)
    )
    rng = torch.Generator(device=device).manual_seed(42)
    before = rng.get_state()
    first = module().objective(center, target, values, 0.02, "reinforce", generator=rng)
    rng.set_state(before)
    second = module().objective(center, target, values, 0.02, "reinforce", generator=rng)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def test_zero_residual_preserves_parent_float64_then_learns_all_features():
    device = require_cuda()
    engine = module()
    model = engine.LinearResidualPolicy(np.zeros(6), np.ones(6), target_scale=0.02).to(device)
    features = torch.ones(3, 6, device=device, dtype=torch.float32)
    parent = torch.tensor([0.01234567890123] * 3, device=device, dtype=torch.float64)
    assert torch.equal(model(features, parent), parent)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, _ = engine.objective(
        model(features, parent), torch.zeros_like(parent), grid(device), 0.02, "mae"
    )
    loss.mean().backward()
    assert torch.all(model.correction.weight.grad != 0)
    optimizer.step()
    assert torch.all(model(features, parent) < parent)


def test_adapter_does_not_backpropagate_into_parent_or_input_encoders():
    device = require_cuda()
    model = module().LinearResidualPolicy(np.zeros(3), np.ones(3), target_scale=0.02).to(device)
    features = torch.ones(2, 3, device=device, requires_grad=True)
    parent = torch.ones(2, device=device, dtype=torch.float64, requires_grad=True)
    model(features, parent).sum().backward()
    assert parent.grad is None and features.grad is None
    assert model.correction.bias.grad is not None


def test_reward_labels_are_not_trainable_parameters():
    target = torch.tensor([0.01], dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="etiquetas"):
        module().objective(
            torch.zeros(1, dtype=torch.float64), target, grid("cpu"), 0.02, "expected"
        )


def test_parameter_precision_does_not_depend_on_global_default_dtype():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = module().LinearResidualPolicy(np.zeros(3), np.ones(3), target_scale=0.02)
        assert model.correction.weight.dtype == torch.float32
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("case", ["nonfinite", "bad_grid", "bad_action", "shape", "scale"])
def test_invalid_policy_inputs_fail_explicitly(case):
    center = torch.tensor([0.0], dtype=torch.float64)
    target = torch.tensor([0.01], dtype=torch.float64)
    values, scale, actions = grid("cpu"), 0.02, torch.tensor([10])
    if case == "nonfinite":
        center[0] = torch.nan
    if case == "bad_grid":
        values = values[:-1]
    if case == "bad_action":
        actions[0] = 21
    if case == "shape":
        target = target.repeat(2)
    if case == "scale":
        scale = 0.0
    with pytest.raises(ValueError):
        module().objective(center, target, values, scale, "reinforce", actions=actions)
