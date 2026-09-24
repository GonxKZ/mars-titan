"""Cálculos de referencia para PPO y Double DQN, sin ejecutar una campaña."""

import numpy as np
import torch

from mars_titan.simulation.algorithms import double_targets, generalized_advantage, ppo_objective


def test_double_dqn_selects_online_action_and_bootstraps_truncations():
    online = torch.tensor([[5.0, 1.0], [5.0, 1.0]])
    target = torch.tensor([[2.0, 10.0], [2.0, 10.0]])
    result = double_targets(
        torch.tensor([1.0, 1.0]), torch.tensor([False, True]), online, target, 0.5
    )
    torch.testing.assert_close(result, torch.tensor([2.0, 1.0]))
    assert not result.requires_grad


def test_gae_bootstraps_truncation_but_does_not_cross_reset():
    advantages, returns = generalized_advantage(
        np.array([1.0, 100.0]),
        np.array([2.0, 0.0]),
        np.array([4.0, 0.0]),
        np.array([False, True]),
        np.array([True, False]),
        gamma=0.5,
        lam=1,
    )
    np.testing.assert_array_equal(advantages, [1, 100])
    np.testing.assert_array_equal(returns, [3, 100])


def test_ppo_clipping_respects_advantage_sign():
    old = torch.zeros(2)
    new = torch.log(torch.tensor([1.3, 1.3], requires_grad=True))
    loss = ppo_objective(new, old, torch.tensor([1.0, -1.0]), clip=0.2)
    torch.testing.assert_close(loss, torch.tensor(0.05))
    loss.backward()
