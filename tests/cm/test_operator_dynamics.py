"""Operador fijo y dinámica variable separados del diagnóstico puntual, sin modelos."""

import math

import pytest
import torch

from mars_titan.cm.numerical_radius import numerical_radius_estimates
from mars_titan.cm.operator_dynamics import fixed_operator_powers, variable_products

A1 = torch.tensor([[0.0, 1.5], [0.0, 0.0]], dtype=torch.float64)
A2 = torch.tensor([[0.0, 0.0], [1.5, 0.0]], dtype=torch.float64)


def test_alternating_pair_is_pointwise_small_but_its_products_grow():
    """El contraejemplo de la issue: radios numéricos 0,75 y autovalor 2,25 del producto."""
    sequence = torch.stack([A1, A2] * 4)
    result = variable_products(sequence, threshold=1.0, grid_size=64)
    torch.testing.assert_close(
        result.pointwise.grid_lower_estimate, torch.full((8,), 0.75, dtype=torch.float64)
    )
    assert (result.pointwise.angular_corrected_estimate < 1).all()
    assert (result.pointwise.spectral_norm == 1.5).all()
    expected = [
        1.5 * 2.25**k if t % 2 == 0 else 2.25 ** (k + 1)
        for t, k in zip(range(8), [0, 0, 1, 1, 2, 2, 3, 3], strict=True)
    ]
    torch.testing.assert_close(result.product_norms, torch.tensor(expected, dtype=torch.float64))
    torch.testing.assert_close(
        result.product_spectral_radii[1::2],
        torch.tensor([2.25**k for k in range(1, 5)], dtype=torch.float64),
    )
    assert result.pointwise_below_but_product_above


def test_each_member_of_the_pair_meets_the_fixed_operator_expression():
    for matrix in (A1, A2):
        result = fixed_operator_powers(matrix, (1, 2, 3), grid_size=64)
        torch.testing.assert_close(result.norms, torch.tensor([1.5, 0.0, 0.0], dtype=torch.float64))
        assert (result.norms <= result.bound).all() and result.contracting_estimate
    # El producto repetido sí es un operador fijo, pero no contractivo.
    product = fixed_operator_powers(A2 @ A1, (1, 2, 4))
    torch.testing.assert_close(
        product.norms, torch.tensor([2.25, 2.25**2, 2.25**4], dtype=torch.float64)
    )
    assert not product.contracting_estimate and (product.norms <= product.bound).all()


@pytest.mark.parametrize("seed", range(6))
def test_fixed_operator_powers_stay_below_twice_the_estimate_power(seed):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn((6, 6), generator=generator, dtype=torch.float64)
    matrix = 0.9 * matrix / torch.linalg.matrix_norm(matrix, ord=2)
    result = fixed_operator_powers(matrix, (1, 2, 5, 9))
    estimate = result.estimates.angular_corrected_estimate
    torch.testing.assert_close(
        result.bound, 2 * estimate ** torch.tensor([1.0, 2.0, 5.0, 9.0], dtype=torch.float64)
    )
    assert (result.norms <= result.bound * (1 + 1e-12)).all()
    for power, norm in zip(result.powers, result.norms, strict=True):
        explicit = torch.linalg.matrix_norm(torch.linalg.matrix_power(matrix, power), ord=2)
        torch.testing.assert_close(norm, explicit, rtol=0, atol=0)


def test_a_repeated_contracting_operator_is_not_flagged():
    matrix = torch.tensor([[0.5, 0.2], [0.0, 0.4]], dtype=torch.float64)
    result = variable_products(matrix.expand(5, 2, 2))
    assert not result.pointwise_below_but_product_above
    explicit = torch.eye(2, dtype=torch.float64)
    for step, norm in enumerate(result.product_norms):
        explicit = matrix @ explicit
        torch.testing.assert_close(norm, torch.linalg.matrix_norm(explicit, ord=2))
        assert norm <= 2 * result.pointwise.angular_corrected_estimate[0] ** (step + 1)


def test_an_expanding_operator_is_not_the_counterexample():
    """Productos por encima del umbral con lecturas puntuales también por encima."""
    result = variable_products((1.5 * torch.eye(2, dtype=torch.float64)).expand(3, 2, 2))
    assert (result.pointwise.angular_corrected_estimate > 1).all()
    assert (result.product_norms > 1).all()
    assert not result.pointwise_below_but_product_above


def test_pointwise_reading_is_the_heuristic_diagnostic_in_blocks_of_sixteen():
    generator = torch.Generator().manual_seed(9)
    sequence = 0.3 * torch.randn((40, 3, 3), generator=generator, dtype=torch.float64)
    result = variable_products(sequence, grid_size=17)
    for start in (0, 16, 32):
        block = numerical_radius_estimates(sequence[start : start + 16], grid_size=17)
        torch.testing.assert_close(
            result.pointwise.angular_corrected_estimate[start : start + 16],
            block.angular_corrected_estimate,
            rtol=0,
            atol=0,
        )
    assert len(result.product_norms) == 40


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: fixed_operator_powers(A1, (0,)), "potencias"),
        (lambda: fixed_operator_powers(A1, (2, 1)), "potencias"),
        (lambda: fixed_operator_powers(A1, [1]), "potencias"),
        (lambda: fixed_operator_powers(A1, (65,)), "potencias"),
        (lambda: fixed_operator_powers(A1.to(torch.complex128), (1,)), "reales"),
        (lambda: fixed_operator_powers(torch.zeros((2, 3)), (1,)), "cuadrados"),
        (lambda: fixed_operator_powers(torch.full((2, 2), math.nan), (1,)), "NaN"),
        (lambda: variable_products(A1), "cuadrados"),
        (lambda: variable_products(torch.zeros((257, 2, 2))), "cuadrados"),
        (lambda: variable_products(torch.stack([A1, A2]), threshold=0), "umbral"),
    ],
)
def test_invalid_operators_and_arguments_are_rejected(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_inputs_are_not_modified_and_no_rng_is_consumed():
    sequence = torch.stack([A1, A2])
    original, state = sequence.clone(), torch.random.get_rng_state().clone()
    variable_products(sequence)
    fixed_operator_powers(A1, (1, 2))
    assert torch.equal(sequence, original)
    assert torch.equal(state, torch.random.get_rng_state())
