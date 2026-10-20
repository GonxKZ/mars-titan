"""Derivadas locales de la mediana y las colas, sin actualizar el predictor."""

import importlib

import pytest
import torch

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]


def api():
    return importlib.import_module("mars_titan.evaluation.quantile_feedback")


def example(device="cpu", dtype=torch.float64):
    x = torch.zeros((1, 1), device=device, dtype=dtype, requires_grad=True)
    weight = torch.tensor(
        [[0.0], [10.0], [1.0], [0.0], [0.0]], device=device, dtype=dtype, requires_grad=True
    )
    bias = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], device=device, dtype=dtype, requires_grad=True)
    target = torch.tensor([0.2], device=device, dtype=dtype, requires_grad=True)
    return x, weight, bias, target


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spread_feedback_can_oppose_and_dominate_the_primary_gradient(device, dtype):
    inputs = example(device, dtype)
    before = [x.detach().clone() for x in inputs]
    result = api().feedback(*inputs)
    # Los signos de los cinco residuos dan g_mediana=0,1 y g_colas=-0,675.
    torch.testing.assert_close(
        result["primary"], torch.full((1, 1), 0.1, device=device, dtype=dtype)
    )
    torch.testing.assert_close(
        result["auxiliary"], torch.full((1, 1), -0.675, device=device, dtype=dtype)
    )
    torch.testing.assert_close(
        result["combined"], torch.full((1, 1), -0.575, device=device, dtype=dtype)
    )
    assert (result["primary"] * result["combined"]).sum() < 0
    for value, old in zip(inputs, before, strict=True):
        torch.testing.assert_close(value, old, rtol=0, atol=0)
        assert value.grad is None
    assert all(not value.requires_grad for value in result.values())


def test_symmetric_tails_can_cancel_without_a_conflict():
    x, weight, bias, target = example()
    weight = torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0]], dtype=x.dtype)
    bias = torch.zeros(5, dtype=x.dtype)
    result = api().feedback(x, weight, bias, target)
    torch.testing.assert_close(result["auxiliary"], torch.zeros_like(x), rtol=0, atol=1e-16)
    torch.testing.assert_close(result["combined"], result["primary"], rtol=0, atol=1e-16)


def test_a_frozen_input_stays_frozen_even_when_called_inside_no_grad():
    values = [value.detach() for value in example()]
    with torch.no_grad():
        result = api().feedback(*values)
    assert all(not value.requires_grad and value.grad is None for value in values)
    torch.testing.assert_close(result["primary"], torch.full_like(values[0], 0.1))


@pytest.mark.parametrize("device", DEVICES)
def test_batched_derivatives_use_the_same_mean_loss_and_preserve_rng(device):
    x, weight, bias, target = example(device)
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state(device) if device == "cuda:0" else None
    one = api().feedback(x, weight, bias, target)
    many = api().feedback(x.repeat(3, 1), weight, bias, target.repeat(3))
    for name in ("primary", "auxiliary", "combined"):
        torch.testing.assert_close(many[name], one[name].repeat(3, 1) / 3)
    torch.testing.assert_close(many["combined"], many["primary"] + many["auxiliary"])
    assert torch.equal(cpu, torch.random.get_rng_state())
    if cuda is not None:
        assert torch.equal(cuda, torch.cuda.get_rng_state(device))


def test_total_gradient_agrees_with_finite_differences_away_from_pinball_ties():
    from mars_titan.models.quantile_head import ordered_quantiles, pinball_loss

    x, weight, bias, target = example()
    actual = api().feedback(x, weight, bias, target)["combined"]
    epsilon = 1e-6
    losses = [
        pinball_loss(
            ordered_quantiles(torch.nn.functional.linear(x + sign * epsilon, weight, bias)), target
        )
        for sign in (-1, 1)
    ]
    torch.testing.assert_close(
        actual.squeeze(), (losses[1] - losses[0]) / (2 * epsilon), rtol=1e-8, atol=1e-8
    )


@pytest.mark.parametrize(
    "fault",
    ["nan", "inf", "weight", "target", "empty", "width", "half", "batch_limit", "width_limit"],
)
def test_invalid_inputs_are_rejected_before_derivation(fault):
    values = list(example())
    if fault in ("nan", "inf"):
        values[0] = torch.full_like(values[0], float(fault))
    elif fault == "weight":
        values[1] = values[1][:4]
    elif fault == "target":
        values[3] = values[3][:0]
    elif fault == "empty":
        values[0] = values[0][:0]
    elif fault == "width":
        values[0] = torch.zeros((1, 2), dtype=torch.float64)
    elif fault == "half":
        values = [x.half() for x in values]
    elif fault == "batch_limit":
        values[0] = torch.zeros((4097, 1), dtype=torch.float64)
    else:
        values[0] = torch.zeros((1, 4097), dtype=torch.float64)
    with pytest.raises(ValueError):
        api().feedback(*values)


@pytest.mark.parametrize(
    "fault",
    [
        "not_tensor",
        "mixed_dtype",
        "sparse",
        "inference",
        "projection_overflow",
        "gradient_overflow",
    ],
)
def test_incompatible_execution_modes_and_overflow_are_explicit(fault):
    values = list(example())
    if fault == "not_tensor":
        values[0] = None
    elif fault == "mixed_dtype":
        values[3] = values[3].float()
    elif fault == "sparse":
        values[0] = values[0].to_sparse()
    elif fault == "projection_overflow":
        values[0] = torch.full((1, 1), 1e308, dtype=torch.float64)
    elif fault == "gradient_overflow":
        values[1] = torch.tensor(
            [[1.7e308], [1.7e308], [-1.7e308], [-1.7e308], [-1.7e308]], dtype=torch.float64
        )
        values[2] = torch.tensor([20.0, 20.0, 0.0, 20.0, 20.0], dtype=torch.float64)
        values[3] = torch.tensor([100.0], dtype=torch.float64)
    if fault == "inference":
        with torch.inference_mode(), pytest.raises(ValueError):
            api().feedback(*values)
    else:
        with pytest.raises(ValueError):
            api().feedback(*values)
