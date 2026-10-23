"""Parada residual acotada del puente regional, sin ajuste de parámetros."""

import importlib
import random
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from tests.proposals.test_mature_proposals import no_optimizer_steps as no_optimizer_steps

BACKENDS = ("adaptive_reference_full", "adaptive_compact")
DTYPES = (torch.float32, torch.float64)


def api():
    return importlib.import_module("mars_titan.models.titans.bridge_recurrence")


def tensor(values, dtype=torch.float64):
    assert torch.cuda.is_available(), "La prueba requiere CUDA, sin alternativa CPU"
    return torch.tensor(values, device="cuda:0", dtype=dtype)


def mix(local, graph, weights, available, backend, *, atol=0.1, rtol=0.0, gain=0.5):
    module = api()
    return module.adaptive_mix(
        local,
        graph,
        weights,
        available,
        steps=4,
        gain=gain,
        stopping=module.BridgeStoppingConfig(atol, rtol, backend=backend),
    )


def explicit_prefix(local, graph, weights, steps):
    """Referencia escalar independiente para una dinámica con todas las regiones presentes."""
    current = local
    batch, regions, width = local.shape
    for _ in range(steps):
        rows = []
        for row in range(batch):
            nodes = []
            for receiver in range(regions):
                message = [
                    sum(
                        graph[row, receiver, sender] * current[row, sender, coordinate]
                        for sender in range(regions)
                    )
                    for coordinate in range(width)
                ]
                transformed = torch.stack(
                    [
                        sum(
                            message[source] * weights[receiver, source, destination]
                            for source in range(width)
                        )
                        for destination in range(width)
                    ]
                )
                nodes.append(0.5 * current[row, receiver] + 0.5 * torch.tanh(transformed))
            rows.append(torch.stack(nodes))
        current = torch.stack(rows)
    return current


def test_residual_stopping_api_is_available_in_call():
    module = importlib.import_module("mars_titan.models.titans.bridge_recurrence")
    for name in ("BridgeStoppingConfig", "AdaptiveMixResult", "adaptive_mix"):
        assert callable(getattr(module, name, None)), f"Falta la API {name}"


def test_none_keeps_the_original_recurrence_identity_and_config_is_frozen():
    module = api()
    expected = dict(
        schema_version=1,
        kind="regional_mix_tanh_v1",
        steps=4,
        max_node_updates_per_row=1216,
        weights="receiver_region_value_reused_at_every_step",
        update="synchronous_fixed_graph_mixture_tanh",
    )
    assert module.BridgeRecurrenceConfig(4, 1216).identity() == expected
    assert module.BridgeRecurrenceConfig(4, 1216, stopping=None).identity() == expected
    stopping = module.BridgeStoppingConfig(0.1, 0.0)
    assert stopping.backend == "adaptive_reference_full"
    active = module.BridgeRecurrenceConfig(4, 1216, stopping=stopping).identity()
    assert active != expected and "stopping" in active
    with pytest.raises(FrozenInstanceError):
        stopping.atol = 0.2


@pytest.mark.parametrize(
    "atol,rtol,backend",
    [
        (True, 0.0, "adaptive_reference_full"),
        (0.0, False, "adaptive_reference_full"),
        (-1.0, 0.0, "adaptive_reference_full"),
        (0.0, -1.0, "adaptive_reference_full"),
        (float("inf"), 0.0, "adaptive_reference_full"),
        (0.0, float("nan"), "adaptive_reference_full"),
        ("0.1", 0.0, "adaptive_reference_full"),
        (0.0, 0.0, "full"),
    ],
)
def test_stopping_rejects_invalid_tolerances_and_backend(atol, rtol, backend):
    with pytest.raises(ValueError):
        api().BridgeStoppingConfig(atol, rtol, backend=backend)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_manual_prefix_selection_values_and_selected_branch_gradients(dtype, backend):
    module = api()
    local = tensor([[[0.1]], [[0.3]], [[1.0]]], dtype).requires_grad_()
    graph = tensor([[[1.0]]] * 3, dtype)
    weights = tensor([[[0.0]]], dtype).requires_grad_()
    available = tensor([[True]] * 3, torch.bool)
    before = tuple(value.detach().clone() for value in (local, graph, weights, available))
    result = module.adaptive_mix(
        local,
        graph,
        weights,
        available,
        steps=4,
        gain=0.5,
        stopping=module.BridgeStoppingConfig(0.1, 0.0, backend=backend),
    )
    assert isinstance(result, module.AdaptiveMixResult)
    tolerance = (
        dict(rtol=1e-6, atol=1e-7) if dtype == torch.float32 else dict(rtol=1e-13, atol=1e-14)
    )
    torch.testing.assert_close(
        result.values, tensor([[[0.05]], [[0.075]], [[0.0625]]], dtype), **tolerance
    )
    assert result.selected_steps.tolist() == [1, 2, 4]
    assert result.evaluated_steps.tolist() == ([4, 4, 4] if backend == BACKENDS[0] else [1, 2, 4])
    assert result.criterion_met.tolist() == [True, True, True]
    with pytest.raises(FrozenInstanceError):
        result.selected_steps = result.evaluated_steps
    for value, expected_dtype in (
        (result.selected_steps, torch.int64),
        (result.evaluated_steps, torch.int64),
        (result.criterion_met, torch.bool),
    ):
        assert value.shape == (3,) and value.dtype == expected_dtype
        assert value.device == local.device and not value.requires_grad
    local_gradient, weight_gradient = torch.autograd.grad(result.values.sum(), (local, weights))
    torch.testing.assert_close(
        local_gradient, tensor([[[0.5]], [[0.25]], [[0.0625]]], dtype), **tolerance
    )
    torch.testing.assert_close(weight_gradient, tensor([[[0.45]]], dtype), **tolerance)
    for value, saved in zip((local, graph, weights, available), before, strict=True):
        assert torch.equal(value, saved)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "previous,current,atol,rtol",
    [
        ([1.0], [0.5], 0.5, 0.0),
        ([1.0], [0.5], 0.0, 0.5),
        ([0.25], [1.0], 0.0, 0.8),
        ([0.0, 0.0], [0.08, 0.08], 0.1, 0.0),
    ],
    ids=("absolute-boundary", "previous-scale", "current-scale", "linf-not-l2"),
)
def test_public_predicate_has_inclusive_linf_and_symmetric_scale(previous, current, atol, rtol):
    before = tensor([[previous]]).requires_grad_()
    after = tensor([[current]]).requires_grad_()
    result = api().residual_stopping(
        before, after, tensor([[True]], torch.bool), atol=atol, rtol=rtol
    )
    assert result.tolist() == [True]
    assert result.shape == (1,) and result.dtype == torch.bool
    assert result.device == before.device and not result.requires_grad


@pytest.mark.cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_fp32_dynamics_use_the_fp64_predicate_without_rounding_the_threshold(backend):
    local = tensor([[[0.2]]], torch.float32)
    graph = tensor([[[1.0]]], torch.float32)
    weights = tensor([[[0.0]]], torch.float32)
    available = tensor([[True]], torch.bool)
    assert float((local.double() * 0.5).item()) > 0.1
    result = mix(local, graph, weights, available, backend)
    assert result.selected_steps.tolist() == [2]
    assert result.evaluated_steps.tolist() == ([4] if backend == BACKENDS[0] else [2])
    assert result.criterion_met.tolist() == [True]
    torch.testing.assert_close(result.values, tensor([[[0.05]]], torch.float32), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_tolerances_require_exact_equality_of_representable_present_coordinates(dtype):
    previous = tensor([[[1.0, 0.0]], [[0.0, -0.0]]], dtype)
    current = tensor([[[1.0, torch.finfo(dtype).tiny]], [[-0.0, 0.0]]], dtype)
    result = api().residual_stopping(
        previous, current, tensor([[True], [True]], torch.bool), atol=0.0, rtol=0.0
    )
    assert result.tolist() == [False, True]


@pytest.mark.cuda
@pytest.mark.parametrize("atol", [0.0, 5e-309])
def test_huge_coordinate_cannot_hide_a_tiny_change_when_relative_tolerance_is_zero(atol):
    previous = tensor([[[1e308, 1e-308]], [[1e308, 0.0]]])
    current = tensor([[[1e308, 2e-308]], [[1e308, 0.0]]])
    assert current[0, 0, 1] != previous[0, 0, 1]
    result = api().residual_stopping(
        previous, current, tensor([[True], [True]], torch.bool), atol=atol, rtol=0.0
    )
    assert result.tolist() == [False, True]


@pytest.mark.cuda
@pytest.mark.parametrize("present", [True, False], ids=("overflow", "masked-before-subtraction"))
def test_fp64_delta_overflow_is_rejected_only_for_present_coordinates(present):
    largest = torch.finfo(torch.float64).max
    previous, current = tensor([[[largest]]]), tensor([[[-largest]]])
    assert torch.isfinite(previous).all() and torch.isfinite(current).all()
    if present:
        with pytest.raises(ValueError):
            api().residual_stopping(
                previous, current, tensor([[True]], torch.bool), atol=0.0, rtol=0.0
            )
    else:
        result = api().residual_stopping(
            previous, current, tensor([[False]], torch.bool), atol=0.0, rtol=0.0
        )
        assert result.tolist() == [True]


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_finite_extreme_tolerances_do_not_require_an_overflowing_threshold(dtype, backend):
    local, graph = tensor([[[2.0]]], dtype), tensor([[[1.0]]], dtype)
    weights, available = tensor([[[0.0]]], dtype), tensor([[True]], torch.bool)
    for atol, rtol in ((1e308, 1e308), (0.0, 1e308), (1e308, 0.0)):
        stopping = api().BridgeStoppingConfig(atol, rtol, backend=backend)
        assert stopping.atol == atol and stopping.rtol == rtol
        result = mix(local, graph, weights, available, backend, atol=atol, rtol=rtol)
        assert result.selected_steps.tolist() == [1] and result.criterion_met.tolist() == [True]
        assert result.evaluated_steps.tolist() == ([4] if backend == BACKENDS[0] else [1])
        torch.testing.assert_close(result.values, tensor([[[1.0]]], dtype), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_absent_regions_do_not_change_the_criterion_and_all_absent_selects_one(dtype, backend):
    local = tensor([[[0.1], [1e20]], [[2.0], [3.0]]], dtype).requires_grad_()
    graph = torch.eye(2, dtype=dtype, device="cuda:0").repeat(2, 1, 1)
    weights = torch.zeros((2, 1, 1), dtype=dtype, device="cuda:0")
    available = tensor([[True, False], [False, False]], torch.bool)
    result = mix(local, graph, weights, available, backend)
    assert result.selected_steps.tolist() == [1, 1]
    assert result.evaluated_steps.tolist() == ([4, 4] if backend == BACKENDS[0] else [1, 1])
    assert result.criterion_met.tolist() == [True, True]
    torch.testing.assert_close(
        result.values, tensor([[[0.05], [0.0]], [[0.0], [0.0]]], dtype), rtol=0, atol=0
    )
    gradient = torch.autograd.grad(result.values.sum(), local)[0]
    torch.testing.assert_close(
        gradient, tensor([[[0.5], [0.0]], [[0.0], [0.0]]], dtype), rtol=0, atol=0
    )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_nonfinite_inputs_and_parameters_cannot_be_hidden_in_absent_regions(dtype, backend):
    available = tensor([[True, False]], torch.bool)
    for broken in ("state", "graph", "weights"):
        local = tensor([[[0.1], [0.2]]], dtype)
        graph = torch.eye(2, dtype=dtype, device="cuda:0")[None]
        weights = torch.zeros((2, 1, 1), dtype=dtype, device="cuda:0")
        if broken == "state":
            local[0, 1, 0] = float("nan")
        elif broken == "graph":
            graph[0, 1, 1] = float("nan")
        else:
            weights[1, 0, 0] = float("inf")
        with pytest.raises(ValueError):
            mix(local, graph, weights, available, backend)
    for invalid in (float("nan"), float("inf")):
        previous = tensor([[[0.1], [invalid]]], dtype)
        current = tensor([[[0.05], [0.0]]], dtype)
        for before, after in ((previous, current), (current, previous)):
            with pytest.raises(ValueError):
                api().residual_stopping(before, after, available, atol=0.1, rtol=0.0)


@pytest.mark.cuda
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("atol", [0.08, 0.14], ids=("compare-four-to-three", "never-select-three"))
def test_fourth_prefix_compares_to_the_immediate_previous_step_and_never_selects_three(
    backend, atol
):
    result = mix(
        tensor([[[1.0]]]),
        tensor([[[1.0]]]),
        tensor([[[0.0]]]),
        tensor([[True]], torch.bool),
        backend,
        atol=atol,
    )
    assert result.selected_steps.tolist() == result.evaluated_steps.tolist() == [4]
    assert result.criterion_met.tolist() == [True]
    torch.testing.assert_close(result.values, tensor([[[0.0625]]]), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_selected_prefix_matches_scalar_reference_gradients_permutations_blocks_and_rng(
    dtype, backend, monkeypatch
):
    basis = tensor([[[1.0, -0.3], [0.5, -0.2]]], dtype)
    local = (tensor([0.1, 0.3, 1.0, 0.35], dtype)[:, None, None] * basis).requires_grad_()
    graph = tensor([[[0.75, 0.25], [0.2, 0.8]]] * 4, dtype)
    weights = tensor([[[0.12, 0.03], [-0.04, 0.16]], [[-0.05, 0.07], [0.02, 0.1]]], dtype)
    weights.requires_grad_()
    available = torch.ones((4, 2), dtype=torch.bool, device="cuda:0")
    saved = tuple(value.detach().clone() for value in (local, graph, weights, available))
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()

    def forbidden_module(*args, **kwargs):
        pytest.fail("La recurrencia pura no debe ejecutar MAC ni otro nn.Module")

    with monkeypatch.context() as patch:
        patch.setattr(torch.nn.Module, "_call_impl", forbidden_module)
        result = mix(local, graph, weights, available, backend)
    assert result.selected_steps.tolist() == [1, 2, 4, 2]
    reference_local = local.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    reference = torch.cat(
        [
            explicit_prefix(
                reference_local[row : row + 1], graph[row : row + 1], reference_weights, steps
            )
            for row, steps in enumerate((1, 2, 4, 2))
        ]
    )
    tolerance = (
        dict(rtol=3e-5, atol=3e-7) if dtype == torch.float32 else dict(rtol=3e-12, atol=3e-13)
    )
    torch.testing.assert_close(result.values, reference, **tolerance)
    actual_gradients = torch.autograd.grad(result.values.square().sum(), (local, weights))
    expected_gradients = torch.autograd.grad(
        reference.square().sum(), (reference_local, reference_weights)
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, **tolerance)
    detached = tuple(value.detach() for value in (local, graph, weights, available))
    order = tensor([2, 0, 3, 1], torch.int64)
    permuted = mix(detached[0][order], detached[1][order], detached[2], detached[3][order], backend)
    torch.testing.assert_close(permuted.values, result.values[order], **tolerance)
    assert torch.equal(permuted.selected_steps, result.selected_steps[order])
    parts = [
        mix(
            detached[0][start:end],
            detached[1][start:end],
            detached[2],
            detached[3][start:end],
            backend,
        )
        for start, end in ((0, 1), (1, 3), (3, 4))
    ]
    torch.testing.assert_close(
        torch.cat([part.values for part in parts]), result.values, **tolerance
    )
    assert torch.equal(torch.cat([part.selected_steps for part in parts]), result.selected_steps)
    assert torch.equal(torch.cat([part.evaluated_steps for part in parts]), result.evaluated_steps)
    assert torch.equal(torch.cat([part.criterion_met for part in parts]), result.criterion_met)
    for value, original in zip((local, graph, weights, available), saved, strict=True):
        assert torch.equal(value, original)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(0))
    assert python_rng == random.getstate()
    for before, after in zip(numpy_rng, np.random.get_state(), strict=True):
        assert np.array_equal(before, after)


@pytest.mark.cuda
def test_zero_residual_does_not_certify_contraction_or_the_global_derivative():
    local = tensor([[[0.0]]]).requires_grad_()
    graph, weights, available = tensor([[[1.0]]]), tensor([[[2.0]]]), tensor([[True]], torch.bool)
    for steps in (1, 2, 4):
        fixed = api().nonlinear_mix(local, graph, weights, available, steps=steps, gain=0.25)
        assert fixed.item() == 0.0
        gradient = torch.autograd.grad(fixed.sum(), local)[0]
        assert gradient.item() == 1.25**steps
    result = mix(local, graph, weights, available, BACKENDS[0], atol=0.0, rtol=0.0, gain=0.25)
    assert result.selected_steps.tolist() == [1] and result.evaluated_steps.tolist() == [4]
    assert result.criterion_met.tolist() == [True]
    assert torch.autograd.grad(result.values.sum(), local)[0].item() == 1.25


@pytest.mark.cuda
def test_full_checks_executed_discarded_steps_while_compact_stops_before_them():
    largest = torch.finfo(torch.float64).max
    local = tensor([[[1.0, 1.0]]])
    graph = tensor([[[1.0]]])
    weights = tensor([[[largest, -largest], [-largest, largest / 2]]])
    available = tensor([[True]], torch.bool)
    compact = mix(local, graph, weights, available, BACKENDS[1], atol=2.0, gain=1.0)
    assert compact.selected_steps.tolist() == compact.evaluated_steps.tolist() == [1]
    assert compact.criterion_met.tolist() == [True]
    torch.testing.assert_close(compact.values, tensor([[[0.0, -1.0]]]), rtol=0, atol=0)
    with pytest.raises(ValueError):
        mix(local, graph, weights, available, BACKENDS[0], atol=2.0, gain=1.0)
