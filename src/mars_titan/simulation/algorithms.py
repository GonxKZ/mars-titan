"""Objetivos y redes auxiliares de los comparadores financieros."""

from numbers import Real

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
    """Acumula GAE en el eje temporal de [tiempo] o [tiempo, entorno].

    Una truncación conserva el bootstrap y corta el arrastre del siguiente episodio.
    Las entradas deben ser finitas, incluso en transiciones terminales.
    """
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not 0 <= value <= 1
        for value in (gamma, lam)
    ):
        raise ValueError("gamma y lambda deben ser números finitos entre cero y uno")
    arrays = [np.asarray(x) for x in (rewards, values, next_values, terminated, truncated)]
    if (
        arrays[0].ndim not in (1, 2)
        or not arrays[0].size
        or any(x.shape != arrays[0].shape for x in arrays)
    ):
        raise ValueError("El recorrido PPO no conserva sus dimensiones")
    if any(x.dtype.kind != "b" for x in arrays[3:]):
        raise ValueError("Las máscaras de terminación y truncación deben ser booleanas")
    if any(x.dtype.kind not in "iuf" for x in arrays[:3]):
        raise ValueError("Las recompensas y valores deben ser números reales")
    if any(not np.isfinite(x).all() for x in arrays[:3]):
        raise ValueError("Las recompensas y valores deben ser finitos")
    gamma, lam = float(gamma), float(lam)
    terminated, truncated = arrays[3:]
    with np.errstate(over="raise", invalid="raise"):
        rewards, values, next_values = (x.astype(np.float64, copy=False) for x in arrays[:3])
        result = rewards + gamma * (~terminated) * next_values - values
        continuation = gamma * lam * (~(terminated | truncated))
        for i in range(len(result) - 2, -1, -1):
            result[i] += continuation[i] * result[i + 1]
        return result, result + values
