"""Selección del catálogo con coeficientes dados, sin ajustar parámetros."""

import importlib
from copy import deepcopy

import pytest
import torch


def api():
    return importlib.import_module("mars_titan.models.titans.mechanism_policy")


def policy(*, dtype=torch.float64, device="cpu", weight=0.5, minimum=0.0):
    width = len(api().FEATURES)
    point = torch.zeros((width + 1, 12), dtype=dtype, device=device)
    interval = torch.zeros((width + 1, 36), dtype=dtype, device=device)
    point[-1, 1] = 2
    interval[-1, 4] = 3
    return api().MechanismPolicy(
        mean=torch.zeros(width, dtype=dtype, device=device),
        scale=torch.ones(width, dtype=dtype, device=device),
        point_coefficients=point,
        interval_coefficients=interval,
        support_min=torch.full((width,), -10, dtype=dtype, device=device),
        support_max=torch.full((width,), 10, dtype=dtype, device=device),
        costs=torch.zeros(36, dtype=dtype, device=device),
        interval_weight=weight,
        min_gain=minimum,
    )


def inputs(model):
    return torch.zeros((2, len(api().FEATURES)), dtype=model.mean.dtype, device=model.mean.device)


def allowed(model):
    return torch.ones((2, 36), dtype=torch.bool, device=model.mean.device)


def test_interval_utility_does_not_change_the_shared_point_gain():
    model = policy()
    result = model(inputs(model), allowed(model))
    assert result.action_ids.tolist() == [4, 4]
    assert result.point_gains[0, 3:6].tolist() == [2, 2, 2]
    assert result.interval_gains[0, 3:6].tolist() == [0, 3, 0]
    assert result.scores[0, 3:6].tolist() == [2, 3.5, 2]
    without_intervals = policy(weight=0)
    assert without_intervals(inputs(model), allowed(model)).action_ids.tolist() == [3, 3]


def test_unavailable_actions_are_excluded_before_the_choice_and_keep_their_scores():
    model = policy()
    availability = allowed(model)
    availability[0, 4] = False
    result = model(inputs(model), availability)
    assert result.action_ids.tolist() == [3, 4]
    assert result.scores[0, 4].item() == 3.5
    assert not result.available[0, 4]
    assert availability[1, 4]


def test_declared_cost_and_minimum_gain_can_keep_the_baseline():
    model = policy(minimum=2)
    model.costs[4] = 2
    assert model(inputs(model), allowed(model)).action_ids.tolist() == [0, 0]
    assert model(inputs(model), allowed(model)).scores[0, 0].item() == 0


def test_marginal_support_fallback_does_not_evaluate_an_overflowing_unused_regression():
    model = policy()
    model.point_coefficients[0, 1] = 1e308
    values = torch.full_like(inputs(model), 1e308)
    result = model(values, allowed(model))
    assert result.action_ids.tolist() == [0, 0]
    assert result.supported.tolist() == [False, False]
    assert torch.isfinite(result.scores).all()


def test_supported_overflow_is_rejected_instead_of_becoming_a_choice():
    model = policy()
    model.point_coefficients[0, 1] = 1e308
    values = inputs(model)
    values[0, 0] = 10
    with pytest.raises(ValueError, match="finit"):
        model(values, allowed(model))


def test_selection_is_row_independent_and_does_not_consume_rng_or_retain_autograd():
    cpu_rng = torch.get_rng_state().clone()
    model = policy()
    model.point_coefficients[0, 1] = 2
    values = inputs(model)
    values[:, 0] = torch.tensor([-2, 2], dtype=values.dtype)
    values.requires_grad_()
    result = model(values, allowed(model))
    assert result.action_ids.tolist() == [0, 4]
    other = model(values.flip(0), allowed(model).flip(0))
    assert torch.equal(other.action_ids, result.action_ids.flip(0))
    assert torch.equal(other.scores, result.scores.flip(0))
    for index in range(2):
        separate = model(values[index : index + 1], allowed(model)[:1])
        assert torch.equal(separate.scores[0], result.scores[index])
    assert not result.scores.requires_grad
    assert not tuple(model.parameters())
    assert torch.equal(torch.get_rng_state(), cpu_rng)


@pytest.mark.parametrize(
    "fault", ["features", "availability", "baseline", "dtype", "empty", "many"]
)
def test_invalid_observations_and_availability_do_not_emit_an_action(fault):
    model = policy()
    values, mask = inputs(model), allowed(model)
    if fault == "features":
        values[0, 0] = float("nan")
    elif fault == "availability":
        mask = mask.to(torch.int64)
    elif fault == "baseline":
        mask[0, 0] = False
    elif fault == "dtype":
        values = values.float()
    elif fault == "empty":
        values, mask = values[:0], mask[:0]
    else:
        values, mask = values.repeat(2, 1), mask.repeat(2, 1)
    with pytest.raises(ValueError):
        model(values, mask)


@pytest.mark.parametrize(
    "fault", ["missing", "extra", "identity", "shape", "dtype", "nan", "scale", "baseline"]
)
def test_rejected_checkpoint_does_not_copy_any_buffer(fault):
    model = policy()
    before = deepcopy(model.state_dict())
    payload = deepcopy(before)
    payload["point_coefficients"][-1, 2] = 8
    if fault == "missing":
        payload.pop("costs")
    elif fault == "extra":
        payload["extra"] = torch.zeros(1)
    elif fault == "identity":
        payload["_extra_state"]["interval_weight"] = 4
    elif fault == "shape":
        payload["interval_coefficients"] = payload["interval_coefficients"][:, :35]
    elif fault == "dtype":
        payload["costs"] = payload["costs"].float()
    elif fault == "nan":
        payload["costs"][4] = float("nan")
    elif fault == "scale":
        payload["scale"][0] = 0
    else:
        payload["point_coefficients"][-1, 0] = 1
    with pytest.raises(ValueError):
        model.load_state_dict(payload)
    for name, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, before[name])
        else:
            assert value == before[name]


def test_checkpoint_recovers_the_next_selection_and_complete_score_vector():
    model = policy()
    payload = deepcopy(model.state_dict())
    payload["costs"][4] = 3
    model.load_state_dict(payload)
    restored = policy()
    restored.load_state_dict(model.state_dict())
    expected, actual = model(inputs(model), allowed(model)), restored(inputs(model), allowed(model))
    assert expected.action_ids.tolist() == [3, 3]
    assert torch.equal(actual.action_ids, expected.action_ids)
    assert torch.equal(actual.scores, expected.scores)


@pytest.mark.parametrize(
    "field,value", [("interval_weight", -1), ("min_gain", True), ("min_gain", float("inf"))]
)
def test_invalid_utility_scales_are_rejected_before_constructing_a_policy(field, value):
    model = policy()
    arguments = {name: tensor for name, tensor in model.named_buffers()}
    arguments.update(interval_weight=0.5, min_gain=0.0)
    arguments[field] = value
    with pytest.raises(ValueError):
        api().MechanismPolicy(**arguments)


@pytest.mark.parametrize(
    "fault",
    [
        "scale",
        "cost",
        "point_anchor",
        "interval_anchor",
        "cost_anchor",
        "bounds",
        "dtype",
        "not_tensor",
    ],
)
def test_invalid_regression_buffers_cannot_define_a_policy(fault):
    model = policy()
    arguments = {name: tensor.clone() for name, tensor in model.named_buffers()}
    if fault == "scale":
        arguments["scale"][0] = 0
    elif fault == "cost":
        arguments["costs"][1] = -1
    elif fault == "point_anchor":
        arguments["point_coefficients"][0, 0] = 1
    elif fault == "interval_anchor":
        arguments["interval_coefficients"][0, 0] = 1
    elif fault == "cost_anchor":
        arguments["costs"][0] = 1
    elif fault == "bounds":
        arguments["support_min"][0] = 1
    elif fault == "dtype":
        arguments["scale"] = arguments["scale"].float()
    else:
        arguments["costs"] = [0] * 36
    with pytest.raises(ValueError):
        api().MechanismPolicy(**arguments, interval_weight=0.5, min_gain=0.0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_autocast_is_rejected_before_changing_the_declared_precision(device):
    if device == "cuda:0":
        assert torch.cuda.is_available()
        torch.cuda.set_device(device)
    model = policy(dtype=torch.float32, device=device)
    backend = "cuda" if device == "cuda:0" else "cpu"
    with torch.autocast(backend), pytest.raises(ValueError, match="autocast"):
        model(inputs(model), allowed(model))


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_checkpoint_restores_the_next_cuda_decision(dtype):
    assert torch.cuda.is_available()
    torch.cuda.set_device("cuda:0")
    cpu = policy(dtype=dtype)
    cpu.costs[4] = 3
    gpu = policy(dtype=dtype, device="cuda:0")
    gpu.load_state_dict(cpu.state_dict())
    expected, actual = cpu(inputs(cpu), allowed(cpu)), gpu(inputs(gpu), allowed(gpu))
    assert actual.action_ids.cpu().tolist() == [3, 3]
    torch.testing.assert_close(actual.scores.cpu(), expected.scores)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_matches_cpu_and_preserves_the_cuda_rng(dtype):
    assert torch.cuda.is_available()
    torch.cuda.set_device("cuda:0")
    rng = torch.cuda.get_rng_state(0).clone()
    cpu = policy(dtype=dtype)
    gpu = policy(dtype=dtype, device="cuda:0")
    expected, actual = cpu(inputs(cpu), allowed(cpu)), gpu(inputs(gpu), allowed(gpu))
    assert torch.equal(actual.action_ids.cpu(), expected.action_ids)
    torch.testing.assert_close(actual.scores.cpu(), expected.scores, rtol=1e-6, atol=1e-6)
    assert torch.equal(torch.cuda.get_rng_state(0), rng)
