"""Objetivos y redes auxiliares de los comparadores financieros."""

import numpy as np
import torch
from torch import nn


class FinancialNetwork(nn.Module):
    """Dos capas de 64 unidades y seis salidas, con valor adicional para PPO."""

    def __init__(self, observations, *, value_head=False):
        super().__init__()
        if type(observations) is not int or not 1 <= observations <= 32768:
            raise ValueError("La observación financiera excede el presupuesto")
        self.body = nn.Sequential(
            nn.Linear(observations, 64, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(64, 64, dtype=torch.float32),
            nn.Tanh(),
        )
        self.actions = nn.Linear(64, 6, dtype=torch.float32)
        self.value = nn.Linear(64, 1, dtype=torch.float32) if value_head else None

    def forward(self, observations):
        features = self.body(observations)
        scores = self.actions(features)
        return (scores, self.value(features).squeeze(-1)) if self.value is not None else scores


def double_targets(rewards, terminated, online_next, target_next, gamma):
    with torch.no_grad():
        selected = online_next.argmax(dim=1, keepdim=True)
        values = target_next.gather(1, selected).squeeze(1)
        return rewards + gamma * (~terminated).to(rewards.dtype) * values


def ppo_objective(log_probabilities, old_log_probabilities, advantages, *, clip=0.2):
    ratio = torch.exp(log_probabilities - old_log_probabilities)
    return -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages).mean()


def generalized_advantage(
    rewards, values, next_values, terminated, truncated, *, gamma=0.99, lam=0.95
):
    arrays = [np.asarray(x) for x in (rewards, values, next_values, terminated, truncated)]
    if any(x.shape != arrays[0].shape or x.ndim != 1 for x in arrays) or not len(arrays[0]):
        raise ValueError("El recorrido PPO no conserva sus dimensiones")
    rewards, values, next_values, terminated, truncated = arrays
    result = np.empty(len(rewards), dtype=np.float64)
    carry = 0.0
    for i in range(len(result) - 1, -1, -1):
        delta = rewards[i] + gamma * (not terminated[i]) * next_values[i] - values[i]
        carry = delta + gamma * lam * (not (terminated[i] or truncated[i])) * carry
        result[i] = carry
    return result, result + values
