"""Votos de H1 en FP32, con recuperación y gradientes, sin pasos de optimizador."""

import io
from dataclasses import replace

import pytest
import torch
from test_area_predictions import VOTE_PARAMETERS, moved_votes
from test_holographic_network import H1, module_inputs, network
from torch.nn.attention import SDPBackend, sdpa_kernel

from mars_titan.training.kernel_policy import FP32_STRICT, apply_kernel_policy

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


@pytest.fixture(autouse=True)
def strict_precision():
    previous = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.mha.get_fastpath_enabled(),
    )
    apply_kernel_policy(FP32_STRICT)
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]
        torch.backends.mha.set_fastpath_enabled(previous[3])


def inputs_on(device):
    blocks, presence, inputs, encoders = module_inputs()
    return (
        blocks.to(device=device, dtype=torch.float32),
        presence.to(device),
        {name: value.to(device=device, dtype=torch.float32) for name, value in inputs.items()},
        encoders.to(device=device, dtype=torch.float32),
    )


def read_with_gradients(module, device):
    features, votes = module.forward_with_votes(*inputs_on(device))
    (features.square().mean() + votes.shift.square().mean()).backward()
    gradients = {}
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        gradient = parameter.grad.detach().cpu()
        assert torch.isfinite(gradient).all(), name
        gradients[name] = gradient
    for name in VOTE_PARAMETERS:
        value = gradients[name]
        assert value is not None and torch.isfinite(value).all(), name
        assert torch.count_nonzero(value) > 0, name
    return features.detach().cpu(), votes.shift.detach().cpu(), gradients


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("areas", [5, 19, 38])
def test_nonzero_votes_and_gradients_survive_recovery_on_the_device(device, areas):
    config = replace(H1, areas=areas, area_predictions=0.5)
    reference = network(config).float()
    moved_votes(reference)
    checkpoint = io.BytesIO()
    torch.save(reference.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = network(config).to(device=device, dtype=torch.float32)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    expected = read_with_gradients(reference, "cpu")
    actual = read_with_gradients(restored, device)
    # Tolerancias fijadas antes de CUDA, como en las comparaciones de #749.
    for observed, wanted in zip(actual[:2], expected[:2], strict=True):
        torch.testing.assert_close(observed, wanted, rtol=1e-4, atol=1e-5)
    assert actual[2].keys() == expected[2].keys()
    for name, wanted in expected[2].items():
        observed = actual[2][name]
        torch.testing.assert_close(observed, wanted, rtol=1e-3, atol=1e-5, msg=name)


@pytest.mark.parametrize("device", DEVICES)
def test_a_row_without_areas_has_zero_shift_and_finite_gradients_on_the_device(device):
    module = network(replace(H1, areas=19, area_predictions=0.5)).to(
        device=device, dtype=torch.float32
    )
    moved_votes(module)
    _, states, alive = module._run(*inputs_on(device))
    alive = alive.clone()
    alive[0] = False
    votes = module.area_votes(states, alive)
    assert votes.shift[0] == 0 and not votes.weights[0].any() and not votes.votes[0].any()
    votes.shift.square().sum().backward()
    parameters = dict(module.named_parameters())
    for name in VOTE_PARAMETERS:
        gradient = parameters[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_initial_votes_preserve_the_task_and_shared_gradients_on_the_device(device, weight):
    plain = network(H1).to(device=device, dtype=torch.float32)
    voting = network(replace(H1, area_predictions=weight)).to(device=device, dtype=torch.float32)
    values = inputs_on(device)
    expected = plain(*values)
    actual, votes = voting.forward_with_votes(*values)
    expected.square().mean().backward()
    (actual + votes.shift[:, None]).square().mean().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not votes.shift.any()
    parameters = dict(voting.named_parameters())
    for name, parameter in plain.named_parameters():
        observed = parameters[name].grad
        if parameter.grad is None:
            assert observed is None, name
        else:
            torch.testing.assert_close(observed, parameter.grad, rtol=0, atol=0, msg=name)
