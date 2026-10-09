"""Jacobianos completos de MAC y del refinamiento episódico en dimensiones pequeñas.

Contrastan la compresión RᵀJR de C con J denso, la forma `I + η D_z f` de cada
refinamiento y las diferencias finitas en FP64. No construyen optimizadores.
"""

import pytest
import torch
from test_episodic_readout_fixed_episodes import random_reader, random_snapshot
from test_local_control import evaluate, fixture
from torch.nn.attention import SDPBackend, sdpa_kernel

from mars_titan.cm.operator_dynamics import variable_products
from mars_titan.models.titans import MACConfig, MemoryConfig, TitansMAC
from mars_titan.models.titans.transition_jacobian import (
    fast_state_jacobian,
    fast_state_point,
    fast_state_trajectory,
    fast_state_transition,
    refinement_jacobians,
)

CONTEXT = "b" * 64


def central_difference(function, point, direction, step=1e-6):
    return (function(point + step * direction) - function(point - step * direction)) / (2 * step)


@pytest.mark.parametrize("dimension", [2, 4])
def test_compressed_operator_is_the_projection_of_the_dense_jacobian(dimension):
    mac, control, token, state = fixture(dimension=dimension, rank=3)
    with sdpa_kernel(SDPBackend.MATH):
        _, state = mac(token, state)
    result = evaluate(control, mac, token, state)
    jacobian = fast_state_jacobian(mac, token, state)
    order = 2 * mac.config.memory.depth * dimension**2
    assert jacobian.shape == (order, order)
    torch.testing.assert_close(
        result.operators[0], control.basis.T @ jacobian @ control.basis, rtol=1e-11, atol=1e-12
    )


def test_dense_jacobian_matches_central_differences_in_fp64():
    mac, _, token, state = fixture(dimension=2)
    jacobian = fast_state_jacobian(mac, token, state)
    point = fast_state_point(state).detach()
    transition = fast_state_transition(mac, token, state)
    generator = torch.Generator().manual_seed(5)
    for _ in range(3):
        direction = torch.randn(point.shape, generator=generator, dtype=torch.float64)
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            numeric = central_difference(transition, point, direction)
        torch.testing.assert_close(jacobian @ direction, numeric, rtol=1e-6, atol=1e-8)


def test_trajectory_advances_with_the_ordinary_transition_and_feeds_variable_products():
    mac, _, _, state = fixture(dimension=2)
    tokens = torch.linspace(-0.4, 0.5, 12, dtype=torch.float64).reshape(1, 6, 2)
    jacobians, final = fast_state_trajectory(mac, tokens, state)
    assert jacobians.shape == (6, 16, 16)
    expected = state
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        for index in range(6):
            _, expected = mac(tokens[:, index : index + 1], expected)
    torch.testing.assert_close(fast_state_point(final), fast_state_point(expected), rtol=0, atol=0)
    assert final.memory.steps.tolist() == [6]
    products = variable_products(jacobians)
    explicit = torch.eye(16, dtype=torch.float64)
    for step, norm in enumerate(products.product_norms):
        explicit = jacobians[step] @ explicit
        torch.testing.assert_close(norm, torch.linalg.matrix_norm(explicit, ord=2))


def test_dense_jacobian_is_limited_to_small_orders_single_flows_and_online_memory():
    large = TitansMAC(
        MACConfig(memory=MemoryConfig(dim=16, max_tokens=1, max_batch=4), max_segment=1),
        dtype=torch.float64,
    )
    token = torch.zeros((1, 1, 16), dtype=torch.float64)
    with pytest.raises(ValueError, match="256"):
        fast_state_jacobian(large, token, large.initial_state(1))
    mac, _, token, _ = fixture(dimension=2)
    with pytest.raises(ValueError, match="un flujo"):
        fast_state_jacobian(mac, token.expand(2, 1, 2), mac.initial_state(2))
    with pytest.raises(ValueError, match="trayectoria"):
        fast_state_trajectory(mac, token.reshape(1, 2), mac.initial_state(1))


def step_parts(model, state, base, snapshot, positions):
    """f(z) = tanh(W[z, base, read(z), presencia] + b) escrita aparte del lector."""
    read, _ = model._read(state, snapshot, CONTEXT, 10, positions)
    joined = torch.cat((state, base, read.values, read.presence.to(state.dtype)), dim=-1)
    return model.refinement(joined).tanh()


@pytest.mark.parametrize("episodes", ["per_step", "first_read"])
def test_each_refinement_operator_is_identity_plus_gate_times_the_state_derivative(episodes):
    model = random_reader(11, refinements=4, neighbors=3, episode_selection=episodes)
    snapshot = random_snapshot(6, torch.float64)
    generator = torch.Generator().manual_seed(2)
    base = torch.randn((1, 32), generator=generator, dtype=torch.float64)
    steps = refinement_jacobians(model, base, snapshot, context_id=CONTEXT, cutoff=10)
    assert steps.shape == (4, 32, 32)
    gate = model.step_logit.detach().sigmoid()
    state, first = base, None
    for jacobian in steps:
        with torch.no_grad():
            _, chosen = model._read(state, snapshot, CONTEXT, 10, first)
        if episodes == "first_read" and first is None:
            first = chosen
        positions = first if episodes == "first_read" else chosen
        derivative = torch.autograd.functional.jacobian(
            lambda value, p=positions: step_parts(model, value, base, snapshot, p), state
        ).reshape(32, 32)
        torch.testing.assert_close(
            jacobian, torch.eye(32, dtype=torch.float64) + gate * derivative, rtol=1e-12, atol=1e-14
        )
        direction = torch.randn((1, 32), generator=generator, dtype=torch.float64)
        numeric = central_difference(
            lambda value, p=positions: value + gate * step_parts(model, value, base, snapshot, p),
            state.detach(),
            direction,
        )
        torch.testing.assert_close((jacobian @ direction.T).T, numeric, rtol=1e-6, atol=1e-8)
        with torch.no_grad():
            state = state + gate * step_parts(model, state, base, snapshot, positions)
    # El recorrido de los Jacobianos termina en el mismo estado que el lector.
    with torch.no_grad():
        result = model(base, snapshot, context_id=CONTEXT, cutoff=10)
    torch.testing.assert_close(state, result.state, rtol=0, atol=0)


def test_refinement_without_episodes_still_has_the_gated_residual_form():
    model = random_reader(4, refinements=2, mode="no_bank")
    base = torch.full((1, 32), 0.1, dtype=torch.float64)
    steps = refinement_jacobians(model, base)
    gate = model.step_logit.detach().sigmoid()
    weight = model.refinement.weight.detach()[:, :32]
    state = base
    for jacobian in steps:
        joined = torch.cat((state, base, torch.zeros_like(state), torch.zeros((1, 1))), dim=-1)
        slope = 1 - model.refinement(joined).detach().tanh().square()
        expected = torch.eye(32, dtype=torch.float64) + gate * slope.T * weight
        torch.testing.assert_close(jacobian, expected, rtol=1e-12, atol=1e-14)
        with torch.no_grad():
            state = state + gate * model.refinement(joined).tanh()


def test_refinement_jacobians_reject_batches_and_other_models():
    model = random_reader(4, refinements=2, mode="no_bank")
    with pytest.raises(ValueError, match="una fila"):
        refinement_jacobians(model, torch.zeros((2, 32), dtype=torch.float64))
    with pytest.raises(ValueError, match="lector"):
        refinement_jacobians(torch.nn.Linear(2, 2), torch.zeros((1, 32), dtype=torch.float64))
