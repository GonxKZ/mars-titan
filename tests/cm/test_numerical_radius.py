"""Contrastes numéricos de operadores, sin construir ni entrenar modelos."""

import math

import pytest
import torch

from mars_titan.cm.numerical_radius import numerical_radius_estimates, radius_penalty


def test_zero_and_degenerate_maxima_have_finite_values_and_gradients():
    for value in (0.0, 0.5):
        matrix = (value * torch.eye(3, dtype=torch.float64)).requires_grad_()
        result = numerical_radius_estimates(matrix)
        assert result.grid_lower_estimate.item() == pytest.approx(value)
        penalty = radius_penalty(result, threshold=0.1)
        penalty.backward()
        assert torch.isfinite(matrix.grad).all()
        assert torch.isfinite(result.spectral_norm)


@pytest.mark.parametrize("angle_block_size", [1, 7, 64])
def test_normal_matrix_and_angular_correction(angle_block_size):
    values = torch.tensor([1.2 * complex(math.cos(0.3), math.sin(0.3)), -0.5j])
    matrix = torch.diag(values.to(torch.complex128))
    result = numerical_radius_estimates(matrix, grid_size=12, angle_block_size=angle_block_size)
    actual = values.abs().max().item()
    assert result.grid_lower_estimate.item() <= actual + 1e-7
    assert result.angular_corrected_estimate.item() >= actual - 1e-7
    assert result.spectral_norm.item() == pytest.approx(actual)
    correction = 2 * torch.linalg.matrix_norm(matrix, ord="fro") * math.sin(math.pi / 24)
    torch.testing.assert_close(
        result.angular_corrected_estimate, result.grid_lower_estimate + correction
    )


def test_nilpotent_and_non_normal_separate_the_three_quantities():
    matrix = torch.tensor([[0.0, 4.0], [0.0, 0.0]], dtype=torch.float64)
    result = numerical_radius_estimates(matrix)
    assert torch.linalg.eigvals(matrix).abs().max().item() == 0
    assert result.grid_lower_estimate.item() == pytest.approx(2)
    assert result.spectral_norm.item() == pytest.approx(4)
    shifted = numerical_radius_estimates(matrix + torch.eye(2))
    assert shifted.grid_lower_estimate.item() == pytest.approx(3)
    assert shifted.spectral_norm.item() > 3


def test_alternating_operators_grow_despite_small_pointwise_numerical_radius():
    first = torch.tensor([[0.0, 1.5], [0.0, 0.0]], dtype=torch.float64)
    second = first.T
    for matrix in (first, second):
        result = numerical_radius_estimates(matrix)
        assert result.grid_lower_estimate.item() == pytest.approx(0.75)
        assert torch.linalg.eigvals(matrix).abs().max().item() == 0
        assert result.spectral_norm.item() == pytest.approx(1.5)
    product = second @ first
    torch.testing.assert_close(product, torch.diag(torch.tensor([0.0, 2.25], dtype=torch.float64)))
    assert torch.linalg.matrix_norm(torch.linalg.matrix_power(product, 4), ord=2) > 25


def test_batches_match_individual_matrices_without_changing_rng_or_input():
    matrix = torch.tensor([[[1.2, 0.4], [-0.3, 0.7]], [[0.2, 0.6], [0.1, -0.4]]])
    original = matrix.clone()
    state = torch.random.get_rng_state().clone()
    result = numerical_radius_estimates(matrix, grid_size=17, angle_block_size=3)
    expected = torch.stack(
        [numerical_radius_estimates(item, grid_size=17).grid_lower_estimate for item in matrix]
    )
    torch.testing.assert_close(result.grid_lower_estimate, expected)
    torch.testing.assert_close(matrix, original)
    assert torch.equal(state, torch.random.get_rng_state())
    assert result.grid_lower_estimate.dtype == torch.float64
    assert result.grid_lower_estimate.device == matrix.device


def test_complex_gradcheck_away_from_ties_and_float32_gradient_preserved():
    matrix = torch.tensor(
        [[1.4 + 0.2j, 0.3 - 0.1j], [0.1 + 0.4j, -0.2 + 0.1j]],
        dtype=torch.complex128,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        lambda value: radius_penalty(numerical_radius_estimates(value, grid_size=13), 0.1),
        (matrix,),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )
    low_precision = torch.tensor([[1.4, 0.3], [0.1, -0.2]], requires_grad=True)
    radius_penalty(numerical_radius_estimates(low_precision), 0.1).backward()
    assert low_precision.grad.dtype == torch.float32
    assert torch.isfinite(low_precision.grad).all()
    assert low_precision.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "measure", ["grid_lower_estimate", "angular_corrected_estimate", "spectral_norm"]
)
def test_penalty_is_per_operator_squared_hinge(measure):
    result = numerical_radius_estimates(torch.tensor([[[0.2]], [[2.0]]]))
    value = getattr(result, measure)
    torch.testing.assert_close(
        radius_penalty(result, 0.9, measure=measure), torch.relu(value - 0.9).square()
    )


@pytest.mark.parametrize(
    "matrix",
    [
        torch.ones(2, 3),
        torch.ones(0, 0),
        torch.ones(17, 1, 1),
        torch.ones(257, 257),
        torch.ones(2, 2, dtype=torch.int64),
        torch.tensor([[float("nan")]]),
        torch.tensor([[float("inf")]]),
    ],
)
def test_invalid_matrices_are_rejected(matrix):
    with pytest.raises((TypeError, ValueError)):
        numerical_radius_estimates(matrix)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid_size": 1},
        {"grid_size": 1025},
        {"grid_size": True},
        {"angle_block_size": 0},
        {"max_estimated_bytes": 1},
    ],
)
def test_invalid_grid_or_memory_budget_is_rejected(kwargs):
    with pytest.raises(ValueError):
        numerical_radius_estimates(torch.eye(2), **kwargs)


def test_saved_autograd_memory_is_included_before_solver(monkeypatch):
    matrix = torch.eye(8, dtype=torch.float64)
    reference = numerical_radius_estimates(matrix, grid_size=64)
    assert reference.estimated_saved_bytes == 0
    enabled = numerical_radius_estimates(matrix.requires_grad_(), grid_size=64)
    assert enabled.estimated_saved_bytes > 0
    calls = []
    monkeypatch.setattr(torch.linalg, "eigvalsh", lambda *args, **kwargs: calls.append(True))
    with pytest.raises(ValueError, match="memoria"):
        numerical_radius_estimates(matrix, max_estimated_bytes=reference.estimated_forward_bytes)
    assert calls == []


@pytest.mark.parametrize("threshold", [True, -1, float("nan"), float("inf")])
def test_invalid_penalty_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        radius_penalty(numerical_radius_estimates(torch.eye(2)), threshold)


def test_unknown_penalty_measure_rejected():
    with pytest.raises(ValueError):
        radius_penalty(numerical_radius_estimates(torch.eye(2)), 1, measure="certified_bound")


@pytest.mark.parametrize(
    "order,grid,block,batch",
    [(1, 2, 1, 1), (1, 17, 1, 1), (1, 128, 1, 1), (1, 128, 8, 1), (1, 128, 1, 16), (2, 17, 3, 1)],
)
def test_autograd_saved_tensors_fit_declared_estimate(order, grid, block, batch):
    saved = []
    matrix = torch.eye(order, dtype=torch.complex128).repeat(batch, 1, 1).requires_grad_()

    def pack(tensor):
        saved.append(tensor.numel() * tensor.element_size())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        result = numerical_radius_estimates(matrix, grid_size=grid, angle_block_size=block)
    assert sum(saved) <= result.estimated_saved_bytes


def test_different_block_sizes_preserve_values_and_smooth_gradients():
    gradients = []
    values = []
    for block in (1, 4, 17):
        matrix = torch.tensor([[1.4, 0.3], [0.1, -0.2]], dtype=torch.float64, requires_grad=True)
        result = numerical_radius_estimates(matrix, grid_size=17, angle_block_size=block)
        penalty = radius_penalty(result, 0.1)
        gradients.append(torch.autograd.grad(penalty, matrix)[0])
        values.append(penalty.detach())
    for index in (1, 2):
        torch.testing.assert_close(values[index], values[0])
        torch.testing.assert_close(gradients[index], gradients[0])


def test_penalty_overflow_is_explicit():
    from dataclasses import replace

    result = numerical_radius_estimates(torch.eye(1))
    large = replace(result, angular_corrected_estimate=torch.tensor(1e200, dtype=torch.float64))
    with pytest.raises(ValueError, match="finito"):
        radius_penalty(large, 0)


def test_non_tensor_and_finite_input_with_nonfinite_output_rejected():
    with pytest.raises(TypeError):
        numerical_radius_estimates([[1.0]])
    with pytest.raises(ValueError, match="no finito"):
        numerical_radius_estimates(torch.full((2, 2), 1e308, dtype=torch.float64))
