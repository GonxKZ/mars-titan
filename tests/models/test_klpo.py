"""Identidades de KLPO en problemas sintéticos con solución comprobable."""

import importlib
import itertools

import pytest
import torch


def engine():
    return importlib.import_module("mars_titan.models.klpo")


def example():
    logits = torch.tensor([[0.4, -0.7, 0.2]], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[0.15, 0.35, 0.5]], dtype=torch.float64)
    rewards = torch.tensor([[-1.3, -0.2, -0.8]], dtype=torch.float64)
    return logits, q.log(), rewards


def test_exact_variance_matches_finite_differences_and_enumerated_full_gradient():
    logits, logq, rewards = example()
    logp = logits.log_softmax(1)
    exact = engine().token_loss(logp, logq, rewards, 0.3, "klpo_exact")[0].sum()
    gradient = torch.autograd.grad(exact, logits, retain_graph=True)[0]
    enumerated = torch.zeros_like(logits)
    for a in range(3):
        loss = engine().token_loss(
            logp, logq, rewards, 0.3, "klpo_full", actions=torch.tensor([a])
        )[0]
        enumerated += (
            logq.exp()[0, a] * torch.autograd.grad(loss.sum(), logits, retain_graph=True)[0]
        )
    torch.testing.assert_close(gradient, enumerated, atol=1e-14, rtol=1e-13)
    for a in range(3):
        losses = []
        for sign in (-1, 1):
            shifted = logits.detach().clone()
            shifted[0, a] += sign * 1e-6
            h = rewards - 0.3 * (shifted.log_softmax(1) - logq)
            centered = h - (logq.exp() * h).sum(1, keepdim=True)
            losses.append((logq.exp() * centered.square()).sum() / 0.6)
        assert gradient[0, a].item() == pytest.approx(
            ((losses[1] - losses[0]) / 2e-6).item(), abs=1e-9
        )


def test_independent_mc_enumeration_matches_exact_but_reusing_action_does_not():
    logits, logq, rewards = example()
    logp, q = logits.log_softmax(1), logq.exp()
    expected = torch.autograd.grad(
        engine().token_loss(logp, logq, rewards, 0.3, "klpo_exact")[0].sum(),
        logits,
        retain_graph=True,
    )[0]
    enumerated, dependent = torch.zeros_like(logits), torch.zeros_like(logits)
    for a, v in itertools.product(range(3), repeat=2):
        loss = engine().token_loss(
            logp,
            logq,
            rewards,
            0.3,
            "klpo_mc",
            actions=torch.tensor([a]),
            auxiliaries=torch.tensor([[v]]),
            auxiliary_samples=1,
        )[0]
        gradient = torch.autograd.grad(loss.sum(), logits, retain_graph=True)[0]
        enumerated += q[0, a] * q[0, v] * gradient
        if a == v:
            dependent += q[0, a] * gradient
    torch.testing.assert_close(enumerated, expected, atol=1e-14, rtol=1e-13)
    assert dependent.abs().max() == 0 and expected.abs().max() > 0.01


def test_gibbs_minimum_and_reward_shift_invariance():
    logits, logq, rewards = example()
    gradients = []
    for shift in (0.0, 9.0):
        loss = engine().token_loss(logits.log_softmax(1), logq, rewards + shift, 0.3, "klpo_exact")[
            0
        ]
        gradients.append(torch.autograd.grad(loss.sum(), logits)[0])
    torch.testing.assert_close(*gradients, atol=1e-14, rtol=1e-13)
    optimum = (logq + rewards / 0.3).detach().requires_grad_()
    loss = engine().token_loss(optimum.log_softmax(1), logq, rewards, 0.3, "klpo_exact")[0]
    assert loss.item() < 1e-28
    assert torch.autograd.grad(loss.sum(), optimum)[0].abs().max() < 1e-14


def test_behavior_has_explicit_full_support_for_extreme_parent():
    values = torch.linspace(-0.1, 0.1, 21, dtype=torch.float64)
    parent = torch.tensor([-1e4, 1e4], dtype=torch.float64, requires_grad=True)
    logq = engine().behavior_log_probabilities(parent, values, 0.02, 1e-6)
    q = logq.exp()
    assert not logq.requires_grad and torch.isfinite(logq).all()
    assert q.min() >= 1e-6 / 21 * (1 - 1e-14)
    torch.testing.assert_close(q.sum(1), torch.ones(2, dtype=torch.float64))
    assert q[0, 0] > 0.999 and q[1, -1] > 0.999


def test_action_rng_is_independent_of_mc_budget_and_restores_exactly():
    logits, logq, rewards = example()
    args = logits.log_softmax(1).repeat(8, 1), logq.repeat(8, 1), rewards.repeat(8, 1), 0.3
    action_rng, auxiliary_rng = torch.Generator().manual_seed(42), torch.Generator().manual_seed(43)
    saved = action_rng.get_state(), auxiliary_rng.get_state()
    first = engine().token_loss(
        *args,
        "klpo_mc",
        generator=action_rng,
        auxiliary_generator=auxiliary_rng,
        auxiliary_samples=128,
    )
    after = action_rng.get_state(), auxiliary_rng.get_state()
    for rng, state in zip((action_rng, auxiliary_rng), saved, strict=True):
        rng.set_state(state)
    second = engine().token_loss(
        *args,
        "klpo_mc",
        generator=action_rng,
        auxiliary_generator=auxiliary_rng,
        auxiliary_samples=128,
    )
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert torch.equal(action_rng.get_state(), after[0]) and torch.equal(
        auxiliary_rng.get_state(), after[1]
    )
    action_rng.set_state(saved[0])
    small = engine().token_loss(
        *args,
        "klpo_mc",
        generator=action_rng,
        auxiliary_generator=auxiliary_rng,
        auxiliary_samples=1,
    )
    assert torch.equal(first[1], small[1])


def test_historical_sampler_and_rewards_never_receive_gradients():
    logits, logq, rewards = example()
    logq.requires_grad_()
    rewards.requires_grad_()
    loss = engine().token_loss(logits.log_softmax(1), logq, rewards, 0.3, "klpo_exact")[0]
    loss.sum().backward()
    assert logits.grad is not None and logq.grad is None and rewards.grad is None


@pytest.mark.parametrize(
    "invalid",
    ["nan", "unnormalized", "zero_support", "beta", "empty", "action", "auxiliary", "rng", "mode"],
)
def test_invalid_distributions_or_draws_fail_explicitly(invalid):
    logits, logq, rewards = example()
    logp, beta, mode, options = logits.log_softmax(1), 0.3, "klpo_full", {}
    if invalid == "nan":
        rewards[0, 0] = torch.nan
    elif invalid == "unnormalized":
        logq = logq + 1
    elif invalid == "zero_support":
        logq = torch.tensor([[0.0, -torch.inf, -torch.inf]], dtype=torch.float64)
    elif invalid == "beta":
        beta = 0
    elif invalid == "empty":
        logp, logq, rewards = logp[:0], logq[:0], rewards[:0]
    elif invalid == "action":
        options["actions"] = torch.tensor([3])
    elif invalid == "auxiliary":
        mode, options = (
            "klpo_mc",
            dict(actions=torch.tensor([0]), auxiliaries=torch.tensor([[3]]), auxiliary_samples=1),
        )
    elif invalid == "rng":
        rng = torch.Generator()
        mode, options = "klpo_mc", dict(generator=rng, auxiliary_generator=rng)
    else:
        mode = "ppo"
    with pytest.raises(ValueError):
        engine().token_loss(logp, logq, rewards, beta, mode, **options)


def test_exact_objective_learns_synthetic_linear_signal_without_financial_data():
    from mars_titan.models.predictive_adaptation import gaussian_log_probabilities

    x = torch.linspace(-1, 1, 32, dtype=torch.float64)
    targets, parent = 0.03 * x, torch.zeros_like(x)
    values = torch.linspace(-0.1, 0.1, 21, dtype=torch.float64)
    logq = engine().behavior_log_probabilities(parent, values, 0.03, 1e-6)
    rewards = -(values[None, :] - targets[:, None]).abs() / 0.03
    weight = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([weight], lr=0.02)
    before = None
    for _ in range(80):
        optimizer.zero_grad()
        logp = gaussian_log_probabilities(0.03 * weight * x, values, 0.03)
        loss = engine().token_loss(logp, logq, rewards, 0.3, "klpo_exact")[0].mean()
        before = loss.item() if before is None else before
        loss.backward()
        optimizer.step()
    assert loss.item() < before * 0.9
    assert weight.item() > 0
