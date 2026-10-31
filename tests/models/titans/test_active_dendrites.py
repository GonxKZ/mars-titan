"""Referencia matemática del operador dendrítico, sin ajustes de parámetros."""

import io
from importlib import import_module
from importlib.util import find_spec

import pytest
import torch


def layer(*, hidden=3, context=2, segments=2, seed=4, dtype=torch.float64):
    return import_module("mars_titan.models.titans.active_dendrites").ActiveDendrites(
        hidden, context, segments=segments, seed=seed, dtype=dtype
    )


def active_layer():
    result = layer()
    with torch.no_grad():
        result.segment_weights.copy_(
            torch.tensor([[[1.0, 0], [-3.0, 0]], [[0, 2.0], [0, -0.5]], [[-1.0, 0], [1.0, 0]]])
        )
        result.gain.copy_(torch.tensor([0.7, -0.4, 1.2]))
    return result


def test_the_contextual_operator_is_available_as_its_own_component():
    assert find_spec("mars_titan.models.titans.active_dendrites") is not None


def test_signed_absolute_winner_matches_an_independent_scalar_reference():
    module = active_layer()
    values = torch.tensor([[2.0, -3.0, 0.5], [-1.0, 4.0, -2.0]], dtype=torch.float64)
    context = torch.tensor([[1.0, 1.0], [-0.5, 0.4]], dtype=torch.float64)
    expected = torch.empty_like(values)
    for row in range(2):
        for unit in range(3):
            responses = [
                sum(
                    module.segment_weights[unit, segment, j].item() * context[row, j].item()
                    for j in range(2)
                )
                for segment in range(2)
            ]
            winner = max(range(2), key=lambda j: abs(responses[j]))
            d = torch.tensor(responses[winner], dtype=torch.float64)
            expected[row, unit] = values[row, unit] * (
                1 + module.gain[unit].tanh() * (2 * d.sigmoid() - 1)
            )
    torch.testing.assert_close(module(values, context), expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_initial_gain_and_zero_context_preserve_exact_values(dtype):
    module = layer(dtype=dtype)
    values = torch.tensor([[1.5, -2.0, 0]], dtype=dtype)
    context = torch.tensor([[0.5, -1.0]], dtype=dtype)
    assert torch.equal(module(values, context), values)
    with torch.no_grad():
        module.gain.fill_(1.5)
    assert torch.equal(module(values, torch.zeros_like(context)), values)


def test_constructor_uses_its_own_rng_and_repeats_the_same_weights():
    before = torch.get_rng_state().clone()
    first, same, other = layer(), layer(), layer(seed=5)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(first.segment_weights, same.segment_weights)
    assert not torch.equal(first.segment_weights, other.segment_weights)
    assert first.segment_weights.abs().sum() > 0
    assert first.gain.count_nonzero() == 0


def test_initial_gain_receives_a_gradient_without_changing_the_weights():
    module = layer()
    before = {k: v.detach().clone() for k, v in module.state_dict().items()}
    values = torch.ones((2, 3), dtype=torch.float64)
    context = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    module(values, context).sum().backward()
    assert module.gain.grad.abs().sum() > 0
    assert module.segment_weights.grad.count_nonzero() == 0
    assert all(torch.equal(before[k], v) for k, v in module.state_dict().items())


def test_ties_choose_the_first_segment_and_only_winners_receive_gradients():
    module = active_layer()
    module(
        torch.ones((1, 3), dtype=torch.float64), torch.ones((1, 2), dtype=torch.float64)
    ).sum().backward()
    gradients = module.segment_weights.grad
    assert gradients[0, 1].abs().sum() > 0 and gradients[0, 0].count_nonzero() == 0
    assert gradients[1, 0].abs().sum() > 0 and gradients[1, 1].count_nonzero() == 0
    assert gradients[2, 0].abs().sum() > 0 and gradients[2, 1].count_nonzero() == 0


def test_gradients_match_finite_differences_away_from_selection_boundaries():
    module = active_layer()
    values = torch.tensor([[0.7, -0.3, 1.1]], dtype=torch.float64, requires_grad=True)
    context = torch.tensor([[0.6, -0.8]], dtype=torch.float64, requires_grad=True)
    # La tercera unidad tiene empate deliberado. Sus dos respuestas comparten valor
    # absoluto para todo el contexto y el desempate siempre elige el primer segmento.
    assert torch.autograd.gradcheck(module, (values, context), atol=1e-6, rtol=1e-4)


def test_output_is_pointwise_bounded_without_claiming_a_full_jacobian_bound():
    module = active_layer()
    values = torch.tensor([[2.0, -3.0, 0]], dtype=torch.float64)
    for scale in (-1e4, 0.0, 1e4):
        context = torch.tensor([[scale, -scale]], dtype=torch.float64)
        actual = module(values, context)
        assert (actual.abs() <= 2 * values.abs()).all()
        assert torch.equal(actual.sign(), values.sign())


def test_rows_are_independent_and_inputs_are_not_modified():
    module = active_layer()
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    context = torch.tensor([[1.0, 2.0], [-2.0, 1.0]], dtype=torch.float64)
    original = values.clone(), context.clone()
    full = module(values, context)
    assert torch.equal(module(values.flip(0), context.flip(0)).flip(0), full)
    assert torch.equal(
        torch.cat([module(values[i : i + 1], context[i : i + 1]) for i in range(2)]), full
    )
    assert torch.equal(values, original[0]) and torch.equal(context, original[1])


def test_large_gains_still_respect_the_pointwise_bound():
    module = active_layer()
    with torch.no_grad():
        module.gain.fill_(1000.0)
    values = torch.ones(1, 3, dtype=torch.float64)
    context = torch.full((1, 2), 1000.0, dtype=torch.float64)
    assert (module(values, context).abs() <= 2 * values.abs()).all()


def test_empty_rows_and_weights_only_roundtrip():
    module = active_layer()
    assert module(
        torch.empty(0, 3, dtype=torch.float64), torch.empty(0, 2, dtype=torch.float64)
    ).shape == (0, 3)
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    buffer.seek(0)
    restored = layer()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    values, context = torch.ones(1, 3, dtype=torch.float64), torch.ones(1, 2, dtype=torch.float64)
    assert torch.equal(restored(values, context), module(values, context))
    assert restored.identity() == module.identity()


@pytest.mark.parametrize(
    "changes",
    [
        dict(hidden=0),
        dict(context=True),
        dict(segments=0),
        dict(segments=65),
        dict(seed=-1),
        dict(dtype=torch.float16),
        dict(hidden=4096, context=4096),
    ],
)
def test_invalid_geometry_or_precision_is_rejected_before_allocation(changes):
    with pytest.raises(ValueError):
        layer(**changes)


@pytest.mark.parametrize("bad", ["rank", "width", "rows", "dtype", "nan", "inf", "layout"])
def test_invalid_context_is_rejected(bad):
    module = active_layer()
    values = torch.ones(2, 3, dtype=torch.float64)
    context = torch.ones(2, 2, dtype=torch.float64)
    if bad == "rank":
        context = context[0]
    if bad == "width":
        context = torch.ones(2, 4, dtype=torch.float64)
    if bad == "rows":
        context = context[:1]
    if bad == "dtype":
        context = context.float()
    if bad == "nan":
        context[0, 0] = float("nan")
    if bad == "inf":
        context[0, 0] = float("inf")
    if bad == "layout":
        context = context.to_sparse()
    with pytest.raises(ValueError):
        module(values, context)


def test_overflow_is_rejected_instead_of_being_hidden_by_the_initial_zero_gain():
    module = layer()
    with torch.no_grad():
        module.segment_weights.fill_(torch.finfo(torch.float64).max)
    with pytest.raises(ValueError):
        module(torch.ones(1, 3, dtype=torch.float64), torch.full((1, 2), 2.0, dtype=torch.float64))


def test_context_depending_on_the_value_can_amplify_the_full_derivative_above_two():
    module = layer(hidden=1, context=1, segments=1)
    with torch.no_grad():
        module.segment_weights.fill_(20.0)
        module.gain.fill_(2.0)
    value = torch.tensor([[0.12]], dtype=torch.float64, requires_grad=True)
    output = module(value, value)
    derivative = torch.autograd.grad(output.sum(), value)[0]
    assert output.abs().item() <= 2 * value.abs().item()
    assert derivative.item() > 2


def test_response_budget_is_checked_before_materializing_the_projection():
    module = layer(hidden=128, context=1, segments=64)
    with pytest.raises(ValueError, match="presupuesto"):
        module(torch.ones(129, 128, dtype=torch.float64), torch.ones(129, 1, dtype=torch.float64))


@pytest.mark.parametrize("bad", ["type", "rank", "width", "dtype", "nan", "layout"])
def test_invalid_representations_are_rejected(bad):
    module = active_layer()
    values = torch.ones(2, 3, dtype=torch.float64)
    if bad == "type":
        values = [[1.0, 1.0, 1.0]]
    if bad == "rank":
        values = values[0]
    if bad == "width":
        values = values[:, :2]
    if bad == "dtype":
        values = values.float()
    if bad == "nan":
        values[0, 0] = float("nan")
    if bad == "layout":
        values = values.to_sparse()
    with pytest.raises(ValueError):
        module(values, torch.ones(2, 2, dtype=torch.float64))


@pytest.mark.parametrize(
    "bad", ["half", "geometry", "gain_dtype", "gain_nan", "segments_nan", "autocast"]
)
def test_incompatible_or_nonfinite_parameters_are_rejected(bad):
    module = layer()
    values = torch.ones(1, 3, dtype=torch.float64)
    context = torch.ones(1, 2, dtype=torch.float64)
    if bad == "half":
        module.half()
    if bad == "geometry":
        module.segments += 1
    if bad == "gain_dtype":
        module.gain = torch.nn.Parameter(module.gain.float())
    if bad == "gain_nan":
        with torch.no_grad():
            module.gain.fill_(float("nan"))
    if bad == "segments_nan":
        with torch.no_grad():
            module.segment_weights.fill_(float("nan"))
    with pytest.raises(ValueError):
        if bad == "autocast":
            with torch.autocast("cpu", dtype=torch.bfloat16):
                module(values, context)
        else:
            module(values, context)


@pytest.mark.cuda
def test_cuda_fp32_matches_the_cpu_reference_and_has_finite_gradients():
    reference = active_layer().float()
    module = active_layer().to(device="cuda:0", dtype=torch.float32)
    values = torch.tensor([[2.0, -3.0, 1.0]], dtype=torch.float32)
    context = torch.tensor([[0.5, -1.0]], dtype=torch.float32)
    actual = module(values.cuda(), context.cuda())
    torch.testing.assert_close(actual.cpu(), reference(values, context), rtol=2e-6, atol=2e-6)
    actual.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in module.parameters())
