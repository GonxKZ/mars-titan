"""Coherencia y elegibilidad con gradientes conocidos, sin objetivos financieros."""

import importlib

import pytest
import torch


def api():
    return importlib.import_module("mars_titan.models.titans.plasticity")


def configuration(**changes):
    module = importlib.import_module("mars_titan.models.titans.plasticity_config")
    return module.MemoryPlasticity(**changes)


def test_first_update_matches_weighted_moments_and_local_eligibility():
    config = configuration(
        beta=0.5,
        trace_decay=0.5,
        epsilon=1e-6,
        write_floor=0.2,
        capture="eligibility",
        capture_floor=0.1,
    )
    gradient = torch.tensor([[[2.0, 0.0], [0.0, 0.0]]], dtype=torch.float64)
    original = api().initial_history(gradient)
    result = api().advance(config, gradient, original, torch.tensor([0]))
    expected_mean = gradient / 2
    expected_energy = torch.tensor([[[2.0], [0.0]]], dtype=torch.float64)
    expected_coherence = torch.tensor([[[1 / (1 + 1e-6)], [0.0]]])
    expected_tag = torch.tensor([[[0.5 * 4 / (4 + 1e-6)], [0.0]]], dtype=torch.float64)
    torch.testing.assert_close(result.history[0], expected_mean)
    torch.testing.assert_close(result.history[1], expected_energy)
    torch.testing.assert_close(result.history[2], expected_tag)
    torch.testing.assert_close(
        result.coherence, expected_coherence.to(torch.float64), rtol=1e-6, atol=0
    )
    torch.testing.assert_close(result.write_gain, 0.2 + 0.8 * result.coherence)
    torch.testing.assert_close(
        result.capture_gain, 0.1 + 0.9 * expected_tag * result.coherence.mean(1, keepdim=True)
    )
    assert all(torch.count_nonzero(value) == 0 for value in original)
    assert len({value.data_ptr() for value in original}) == 3


def test_persistent_and_alternating_gradients_have_different_coherence():
    config = configuration(beta=0.9)
    gradient = torch.ones(2, 3, 3, dtype=torch.float64)
    history = api().initial_history(gradient)
    for index in range(60):
        current = gradient.clone()
        current[1] *= (-1) ** index
        result = api().advance(config, current, history, torch.full((2,), index))
        history = result.history
    assert result.coherence[0].min() > 0.999
    assert result.coherence[1].max() < 0.01
    torch.testing.assert_close(history[1][0], history[1][1])
    assert result.write_gain[0].min() > result.write_gain[1].max()


def test_constant_controls_keep_the_declared_gains_and_rng():
    config = configuration(
        write="constant", capture="constant", write_constant=0.25, capture_constant=0.75
    )
    gradient = torch.arange(8, dtype=torch.float64).reshape(2, 2, 2)
    rng = torch.random.get_rng_state().clone()
    result = api().advance(
        config, gradient, api().initial_history(gradient), torch.zeros(2, dtype=torch.int64)
    )
    assert torch.equal(result.write_gain, torch.full((2, 2, 1), 0.25, dtype=torch.float64))
    assert torch.equal(result.capture_gain, torch.full((2, 2, 1), 0.75, dtype=torch.float64))
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("beta", [0.0, 0.5, 0.999999999])
def test_zero_gradients_are_finite_and_do_not_create_an_eligibility_tag(beta):
    config = configuration(beta=beta, capture="eligibility")
    gradient = torch.zeros(1, 2, 2, dtype=torch.float64, requires_grad=True)
    result = api().advance(config, gradient, api().initial_history(gradient), torch.tensor([0]))
    assert torch.count_nonzero(result.coherence) == 0
    assert torch.count_nonzero(result.history[2]) == 0
    loss = sum(value.sum() for value in result.history) + result.write_gain.sum()
    (derivative,) = torch.autograd.grad(loss, gradient)
    assert torch.isfinite(derivative).all()


def test_tag_persists_and_decays_when_the_next_gradient_is_zero():
    config = configuration(capture="eligibility", trace_decay=0.5)
    gradient = torch.ones(1, 2, 2, dtype=torch.float64)
    first = api().advance(config, gradient, api().initial_history(gradient), torch.tensor([0]))
    following = api().advance(config, torch.zeros_like(gradient), first.history, torch.tensor([1]))
    torch.testing.assert_close(following.history[2], first.history[2] * 0.5)
    assert following.capture_gain.min() > config.capture_floor


def test_float64_gradients_agree_with_finite_differences():
    config = configuration(capture="eligibility", epsilon=0.1)
    gradient = torch.tensor([[[0.2, 0.1], [0.3, -0.4]]], dtype=torch.float64, requires_grad=True)
    mean = torch.full_like(gradient, 0.1, requires_grad=True)
    energy = torch.full((1, 2, 1), 2.0, dtype=torch.float64, requires_grad=True)
    tag = torch.full((1, 2, 1), 0.3, dtype=torch.float64, requires_grad=True)

    def function(g, m, v, z):
        result = api().advance(config, g, (m, v, z), torch.tensor([3]))
        return (*result.history, result.write_gain, result.capture_gain)

    assert torch.autograd.gradcheck(function, (gradient, mean, energy, tag))


@pytest.mark.parametrize(
    "field,value",
    [
        ("write", "bayesian"),
        ("capture", "certified"),
        ("beta", 1.0),
        ("beta", -0.1),
        ("trace_decay", 1.0),
        ("epsilon", 0.0),
        ("epsilon", float("nan")),
        ("write_floor", -0.1),
        ("capture_floor", 1.1),
        ("write_constant", float("inf")),
        ("capture_constant", -0.1),
        ("beta", True),
    ],
)
def test_invalid_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        configuration(**{field: value})


@pytest.mark.parametrize("fault", ["nan", "overflow", "negative_energy", "tag", "steps"])
def test_invalid_update_does_not_modify_its_input(fault):
    config = configuration()
    gradient = torch.ones(1, 2, 2, dtype=torch.float64)
    history = api().initial_history(gradient)
    steps = torch.tensor([0])
    if fault == "nan":
        gradient[0, 0, 0] = float("nan")
    elif fault == "overflow":
        gradient.fill_(1e200)
    elif fault == "negative_energy":
        history[1].fill_(-1)
    elif fault == "tag":
        history[2].fill_(1.1)
    else:
        steps.fill_(-1)
    before = tuple(value.clone() for value in history)
    with pytest.raises(ValueError):
        api().advance(config, gradient, history, steps)
    assert all(torch.equal(a, b) for a, b in zip(before, history, strict=True))


def test_rows_are_independent_of_batch_order():
    config = configuration(capture="eligibility")
    gradient = torch.arange(27, dtype=torch.float64).reshape(3, 3, 3) / 20
    history = api().initial_history(gradient)
    steps = torch.tensor([0, 2, 7])
    order = torch.tensor([2, 0, 1])
    direct = api().advance(config, gradient, history, steps)
    permuted = api().advance(
        config, gradient[order], tuple(x[order] for x in history), steps[order]
    )
    for a, b in zip(
        (*direct.history, direct.write_gain, direct.capture_gain),
        (*permuted.history, permuted.write_gain, permuted.capture_gain),
        strict=True,
    ):
        torch.testing.assert_close(a[order], b, rtol=0, atol=0)


def test_overflowing_denominators_are_not_hidden_by_a_finite_zero_ratio():
    config = configuration(beta=0.5, epsilon=torch.finfo(torch.float64).max)
    gradient = torch.ones(1, 2, 2, dtype=torch.float64)
    history = api().initial_history(gradient)
    history[1].fill_(torch.finfo(torch.float64).max)
    with pytest.raises(ValueError, match="NaN|infinito"):
        api().advance(config, gradient, history, torch.tensor([2]))


def test_strided_autograd_gradients_preserve_values_without_requiring_a_copy():
    gradient = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3).transpose(1, 2)
    assert not gradient.is_contiguous()
    history = api().initial_history(gradient)
    assert all(value.is_contiguous() for value in history)
    steps = torch.zeros(2, dtype=torch.int64)
    expected = api().advance(configuration(), gradient.contiguous(), history, steps)
    actual = api().advance(configuration(), gradient, history, steps)
    for a, b in zip(
        (*expected.history, expected.write_gain, expected.capture_gain),
        (*actual.history, actual.write_gain, actual.capture_gain),
        strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    "fault",
    ["config", "rectangular", "half", "steps_type", "history_length", "epsilon_subnormal"],
)
def test_invalid_frontier_contracts_are_rejected_before_computation(fault):
    config = configuration()
    gradient = torch.ones(1, 2, 2, dtype=torch.float32)
    history = api().initial_history(gradient)
    steps = torch.tensor([0])
    if fault == "config":
        config = {}
    elif fault == "rectangular":
        gradient = torch.ones(1, 2, 3)
    elif fault == "half":
        gradient = gradient.half()
    elif fault == "steps_type":
        steps = steps.float()
    elif fault == "history_length":
        history = history[:2]
    elif fault == "epsilon_subnormal":
        config = configuration(epsilon=1e-45)
    with pytest.raises(ValueError):
        api().advance(config, gradient, history, steps)


def test_autocast_cannot_silently_change_plasticity_precision():
    gradient = torch.ones(1, 2, 2)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        api().advance(configuration(), gradient, api().initial_history(gradient), torch.tensor([0]))
