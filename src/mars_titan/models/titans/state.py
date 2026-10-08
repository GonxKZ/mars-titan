"""Estado explícito por flujo, sin almacenamiento compartido mutable."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class NeuralMemoryState:
    weights: tuple[Tensor, ...]
    momentum: tuple[Tensor, ...]
    steps: Tensor
    config_id: str


@dataclass(frozen=True)
class MACState:
    memory: NeuralMemoryState
    config_id: str


def check_differentiable(differentiable: bool) -> None:
    if type(differentiable) is not bool:
        raise ValueError("differentiable debe ser booleano")


def check_finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contiene NaN o infinito")


def copy_state(state: NeuralMemoryState, *, differentiable: bool) -> NeuralMemoryState:
    def copy(value: Tensor) -> Tensor:
        return value.clone() if differentiable else value.detach().clone()

    return NeuralMemoryState(
        tuple(copy(w) for w in state.weights),
        tuple(copy(m) for m in state.momentum),
        state.steps.clone(),
        state.config_id,
    )


def require_payload(payload: object, fields: set[str]) -> dict:
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("El estado serializado tiene campos incompatibles")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("La versión del estado serializado no está admitida")
    return payload
