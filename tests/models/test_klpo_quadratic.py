"""Identidades cuadráticas con parámetros fijos, sin optimizador ni ajuste."""

import importlib
import math

import pytest
import torch

from mars_titan.models.klpo import behavior_log_probabilities, token_loss
from mars_titan.models.predictive_adaptation import gaussian_log_probabilities


def engine():
    return importlib.import_module("mars_titan.models.klpo_quadratic")


def example(dtype=torch.float64):
    values = torch.linspace(-0.1, 0.1, 21, dtype=dtype)
    parent = torch.tensor([-0.05, 0.0, 0.08], dtype=dtype)
    logq = behavior_log_probabilities(parent, values, 0.03, 1e-6).to(dtype)
    targets = torch.tensor([-0.01, 0.02, 0.06], dtype=dtype)
    rewards = -(values[None, :] - targets[:, None]).abs() / 0.03
    return logq, rewards, values, 0.03, 0.3


def evaluate(terms, centers):
    return terms.constant + terms.linear * centers + terms.curvature * centers.square() / 2


def test_symmetric_case_has_hand_computed_coefficients():
    logq = torch.full((1, 3), -math.log(3), dtype=torch.float64)
    terms = engine().quadratic_terms(
        logq,
        torch.tensor([[-1.0, 0.0, -1.0]], dtype=torch.float64),
        torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64),
        1.0,
        2.0,
    )
    torch.testing.assert_close(terms.constant, torch.tensor([0.0]).double(), atol=1e-28, rtol=0)
    torch.testing.assert_close(terms.linear, torch.tensor([0.0]).double(), atol=1e-14, rtol=0)
    torch.testing.assert_close(
        terms.curvature, torch.tensor([4 / 3], dtype=torch.float64), atol=1e-14, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_coefficients_match_exact_value_gradient_and_hessian(dtype):
    logq, rewards, values, scale, beta = example(dtype)
    centers = torch.tensor([-0.04, 0.015, 0.1], dtype=torch.float64, requires_grad=True)
    terms = engine().quadratic_terms(logq, rewards, values, scale, beta)
    actual = evaluate(terms, centers)
    expected = token_loss(
        gaussian_log_probabilities(centers, values, scale), logq, rewards, beta, "klpo_exact"
    )[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    gradient = torch.autograd.grad(actual.sum(), centers, create_graph=True)[0]
    reference = torch.autograd.grad(expected.sum(), centers, create_graph=True)[0]
    torch.testing.assert_close(gradient, reference, rtol=1e-12, atol=1e-12)
    hessian = torch.autograd.grad(gradient.sum(), centers)[0]
    reference_hessian = torch.autograd.grad(reference.sum(), centers)[0]
    torch.testing.assert_close(hessian, reference_hessian, rtol=1e-12, atol=1e-12)
    assert all(value.dtype == torch.float64 for value in terms)


def test_coefficients_match_finite_differences_without_parameter_updates():
    logq, rewards, values, scale, beta = example()
    centers = torch.tensor([-0.04, 0.015, 0.1], dtype=torch.float64)
    terms = engine().quadratic_terms(logq, rewards, values, scale, beta)
    losses = [
        token_loss(
            gaussian_log_probabilities(centers + offset, values, scale),
            logq,
            rewards,
            beta,
            "klpo_exact",
        )[0]
        for offset in (-1e-6, 1e-6)
    ]
    torch.testing.assert_close(
        (losses[1] - losses[0]) / 2e-6,
        terms.linear + terms.curvature * centers,
        rtol=1e-7,
        atol=1e-7,
    )


def test_terms_freeze_data_preserve_storage_and_do_not_consume_rng():
    logq, rewards, values, scale, beta = example()
    tensors = (logq, rewards, values)
    before = [value.clone() for value in tensors]
    for value in tensors:
        value.requires_grad_()
    rng = torch.get_rng_state().clone()
    terms = engine().quadratic_terms(logq, rewards, values, scale, beta)
    assert all(not value.requires_grad and value.grad_fn is None for value in terms)
    assert torch.equal(rng, torch.get_rng_state())
    for actual, expected in zip(tensors, before, strict=True):
        assert torch.equal(actual, expected)
    terms.constant.add_(1)
    for actual, expected in zip(tensors, before, strict=True):
        assert torch.equal(actual, expected)


def test_row_order_partition_and_reward_shift_preserve_terms():
    logq, rewards, values, scale, beta = example()
    first = engine().quadratic_terms(logq, rewards, values, scale, beta)
    shifted = engine().quadratic_terms(logq, rewards + 7, values, scale, beta)
    for left, right in zip(first, shifted, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
    for indices in ([2, 0, 1], [0], [1, 2]):
        subset = engine().quadratic_terms(logq[indices], rewards[indices], values, scale, beta)
        for actual, expected in zip(subset, first, strict=True):
            torch.testing.assert_close(actual, expected[indices], rtol=0, atol=0)


def test_smaller_exact_objective_can_increase_center_absolute_error():
    values = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    epsilon, beta = 1e-6, 0.1
    # Tres acciones para la identidad escalar, sin cambiar la rejilla vigente de 21.
    logq = torch.logaddexp(
        torch.log_softmax(-values.square() / 2, 0) + math.log1p(-epsilon),
        torch.full_like(values, math.log(epsilon / 3)),
    )[None, :]
    rewards = -(values - 0.2).abs()[None, :]
    terms = engine().quadratic_terms(logq, rewards, values, 1.0, beta)
    q = logq.exp()
    variance = (q * values.square()).sum()
    covariance = (q * values * rewards).sum()
    torch.testing.assert_close(
        covariance / variance, torch.tensor(0.2, dtype=torch.float64), rtol=1e-12, atol=1e-12
    )
    centers = torch.tensor([0.0, 2.0], dtype=torch.float64, requires_grad=True)
    logp = torch.log_softmax(centers[:, None] * values - values.square() / 2, 1)
    exact = token_loss(logp, logq.expand(2, -1), rewards.expand(2, -1), beta, "klpo_exact")[0]
    torch.testing.assert_close(evaluate(terms, centers), exact, rtol=1e-12, atol=1e-12)
    gradient = torch.autograd.grad(exact.sum(), centers)[0]
    assert gradient[1].abs() < 1e-14
    assert terms.curvature.item() > 0 and exact[1] < exact[0]
    torch.testing.assert_close((centers - 0.2).abs(), torch.tensor([0.2, 1.8], dtype=torch.float64))


def test_emitted_discrete_median_is_separate_from_adapter_center():
    from mars_titan.environments.actions import ActionGrid
    from mars_titan.models.predictive_adaptation import LinearResidualPolicy

    # Construcción técnica con corrección nula, sin fit ni cambios de parámetros.
    with torch.random.fork_rng(devices=[]):
        adapter = LinearResidualPolicy([0.0], [1.0], target_scale=1.0)
    before = {name: value.clone() for name, value in adapter.state_dict().items()}
    parent = torch.tensor([0.0, 2.0], dtype=torch.float64)
    centers = adapter(torch.zeros(2, 1), parent)
    torch.testing.assert_close(centers, parent, rtol=0, atol=0)
    values = torch.arange(-10, 11, dtype=torch.float64) / 10
    grid = ActionGrid(values.numpy(), 1.0, "0" * 64, 1)
    probabilities = gaussian_log_probabilities(centers, values, 1.0).exp().detach().numpy()
    medians = torch.from_numpy(grid.median(probabilities))
    torch.testing.assert_close(medians, torch.tensor([0.0, 0.6], dtype=torch.float64))
    assert (medians[1] - 0.2).abs() > (medians[0] - 0.2).abs()
    logq = behavior_log_probabilities(torch.zeros(2), values, 1.0, 1e-6)
    rewards = -(values - 0.2).abs()[None, :].expand(2, -1)
    loss = token_loss(
        gaussian_log_probabilities(centers, values, 1.0), logq, rewards, 0.1, "klpo_exact"
    )[0]
    assert loss[1] < loss[0]
    for name, value in adapter.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize(
    "invalid",
    [
        "beta_zero",
        "beta_bool",
        "scale_zero",
        "scale_nan",
        "grid_shape",
        "grid_order",
        "grid_nan",
        "logq_nan",
        "normalization",
        "support",
        "rewards_shape",
        "rewards_layout",
        "integer",
        "fp16",
        "overflow",
    ],
)
def test_invalid_or_unrepresentable_quadratic_inputs_fail(invalid):
    logq, rewards, values, scale, beta = example()
    if invalid == "beta_zero":
        beta = 0
    elif invalid == "beta_bool":
        beta = True
    elif invalid == "scale_zero":
        scale = 0
    elif invalid == "scale_nan":
        scale = float("nan")
    elif invalid == "grid_shape":
        values = values[:-1]
    elif invalid == "grid_order":
        values[1] = values[0]
    elif invalid == "grid_nan":
        values[0] = torch.nan
    elif invalid == "logq_nan":
        logq[0, 0] = torch.nan
    elif invalid == "normalization":
        logq += 1
    elif invalid == "support":
        logq[:] = -torch.inf
        logq[:, 0] = 0
    elif invalid == "rewards_shape":
        rewards = rewards[:1]
    elif invalid == "rewards_layout":
        rewards = rewards.to_sparse()
    elif invalid == "integer":
        rewards = rewards.long()
    elif invalid == "fp16":
        rewards = rewards.half()
    else:
        rewards[0, 0] = 1e308
        rewards[0, 1] = -1e308
    with pytest.raises(ValueError):
        engine().quadratic_terms(logq, rewards, values, scale, beta)


def test_element_budget_precedes_promotion_and_moment_buffers(monkeypatch):
    module = engine()

    def forbidden(*args, **kwargs):
        raise AssertionError("Se alcanzó la conversión antes de comprobar el presupuesto")

    monkeypatch.setattr(module, "_inputs", forbidden)
    logq = torch.tensor([-math.log(4096)]).expand(4096, 4096)
    rewards = torch.zeros(1).expand(4096, 4096)
    with pytest.raises(ValueError, match="presupuesto"):
        module.quadratic_terms(logq, rewards, torch.arange(4096).float(), 1, 0.3)
