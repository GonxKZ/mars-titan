"""Recorte por filas con extremos finitos, ceros y derivadas conservadas."""

import pytest
import torch

from mars_titan.models.titans import MemoryConfig, MemoryStability, NeuralMemory
from mars_titan.models.titans.neural_memory import clip_rows


@pytest.mark.parametrize(
    ("dtype", "magnitude", "limit"),
    [
        (torch.float32, 1e19, 1.0),
        (torch.float64, 1e200, 1.0),
        (torch.float32, 1e-20, 1e-20),
        (torch.float64, 1e-200, 1e-200),
        (torch.float64, 1e200, 1e-200),
        (torch.float32, 1e19, 7.0),
        (torch.float64, 1e200, 7.0),
    ],
)
def test_finite_extremes_keep_the_direction_and_requested_norm(dtype, magnitude, limit):
    gradient = torch.tensor([[3 * magnitude, 4 * magnitude]], dtype=dtype)
    original = gradient.clone()
    result = clip_rows(gradient, limit)
    torch.testing.assert_close(result / limit, torch.tensor([[0.6, 0.8]], dtype=dtype))
    assert torch.isfinite(result).all()
    assert result.dtype == dtype and result.device == gradient.device
    assert torch.equal(gradient, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unclipped_rows_preserve_exact_values_and_signed_zero(dtype):
    gradient = torch.tensor([[0.3, -0.4], [-0.0, 0.0]], dtype=dtype, requires_grad=True)
    result = clip_rows(gradient, 1.0)
    assert torch.equal(result, gradient)
    assert torch.equal(torch.signbit(result), torch.signbit(gradient))
    derivative = torch.autograd.grad(result.sum(), gradient, create_graph=True)[0]
    torch.testing.assert_close(derivative, torch.ones_like(gradient), rtol=0, atol=0)


@pytest.mark.parametrize("magnitude", [1.0, 1e19, 1e200])
def test_large_rows_retain_the_hand_derived_first_derivative(magnitude):
    value = torch.tensor([[3 * magnitude, 4 * magnitude]], dtype=torch.float64)
    value.requires_grad_()
    derivative = torch.autograd.grad(clip_rows(value, 1.0).sum(), value)[0]
    # Para g = (3s, 4s), d(sum(g / ||g||))/dg = (0,032, -0,024) / s.
    torch.testing.assert_close(
        derivative * magnitude,
        torch.tensor([[0.032, -0.024]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    "values", [[[0.0, 0.0]], [[0.3, -0.4]], [[3.0, 4.0]], [[3.0, 0.0]], [[3.0, 3.0]]]
)
def test_first_and_second_derivatives_match_finite_differences(values):
    gradient = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: clip_rows(value, 1.0), (gradient,))
    assert torch.autograd.gradgradcheck(lambda value: clip_rows(value, 1.0), (gradient,))


def test_strided_rows_are_clipped_independently():
    values = torch.tensor([[3e200, 9.0, 4e200, 9.0], [0.3, 9.0, -0.4, 9.0]], dtype=torch.float64)
    gradient = values[:, ::2]
    before = values.clone()
    result = clip_rows(gradient, 1.0)
    torch.testing.assert_close(result, torch.tensor([[0.6, 0.8], [0.3, -0.4]], dtype=torch.float64))
    assert torch.equal(values, before)


@pytest.mark.parametrize(("dtype", "limit"), [(torch.float32, 1e-20), (torch.float64, 1e-200)])
def test_tiny_limit_keeps_zero_rows_and_their_derivatives_finite(dtype, limit):
    gradient = torch.zeros(1, 2, dtype=dtype, requires_grad=True)
    first = torch.autograd.grad(clip_rows(gradient, limit).sum(), gradient, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), gradient)[0]
    torch.testing.assert_close(first, torch.ones_like(gradient), rtol=0, atol=0)
    torch.testing.assert_close(second, torch.zeros_like(gradient), rtol=0, atol=0)


def associative_update(dtype, scale, device):
    model = NeuralMemory(
        MemoryConfig(
            dim=2, depth=1, normalize_qk=False, stability=MemoryStability(gradient_clip=1)
        ),
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        model.initial_weights[0].copy_(torch.eye(2, dtype=dtype, device=device))
        model.key_projection.weight.copy_(torch.eye(2, dtype=dtype, device=device))
        for projection in (
            model.value_projection,
            model.alpha_projection,
            model.eta_projection,
            model.theta_projection,
        ):
            projection.weight.zero_()
    shared = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    state = model.initial_state(1)
    following = model.update(
        torch.tensor([[[3 * scale, 4 * scale]]], dtype=dtype, device=device), state
    )
    # Cada fila de 2·residuo·clave tiene dirección (0,6, 0,8). Con logits nulos θ = 0,05.
    expected_momentum = torch.tensor([[[-0.03, -0.04], [-0.03, -0.04]]], dtype=dtype, device=device)
    torch.testing.assert_close(following.momentum[0], expected_momentum)
    torch.testing.assert_close(following.weights[0], 0.499 * state.weights[0] + expected_momentum)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, shared[name], rtol=0, atol=0)
    assert following.steps.item() == 1 and state.steps.item() == 0


@pytest.mark.parametrize(("dtype", "scale"), [(torch.float32, 1e9), (torch.float64, 1e100)])
def test_associative_update_keeps_finite_surprise_instead_of_erasing_it(dtype, scale):
    associative_update(dtype, scale, "cpu")


def test_unrepresentable_clip_is_rejected_before_initializing_memory():
    config = MemoryConfig(dim=2, stability=MemoryStability(gradient_clip=1e-200))
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="precisión"):
        NeuralMemory(config, dtype=torch.float32)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_precision_conversion_cannot_silently_remove_the_clip():
    config = MemoryConfig(dim=2, stability=MemoryStability(gradient_clip=1e-200))
    model = NeuralMemory(config, dtype=torch.float64).float()
    with pytest.raises(ValueError, match="precisión"):
        model.initial_state(1)


def test_direct_clipping_rejects_an_unrepresentable_limit():
    with pytest.raises(ValueError, match="precisión"):
        clip_rows(torch.zeros(1, 2), 1e-200)


@pytest.mark.cuda
@pytest.mark.parametrize("magnitude", [1.0, 1e19])
def test_cuda_float32_uses_the_same_clipping_contract(magnitude):
    gradient = torch.tensor([[3 * magnitude, 4 * magnitude]], dtype=torch.float32, device="cuda:0")
    result = clip_rows(gradient, 1.0)
    torch.testing.assert_close(result.cpu(), torch.tensor([[0.6, 0.8]], dtype=torch.float32))
    assert result.device == torch.device("cuda:0")


@pytest.mark.cuda
def test_cuda_associative_update_preserves_finite_surprise():
    associative_update(torch.float32, 1e9, "cuda:0")
