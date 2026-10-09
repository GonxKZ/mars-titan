"""Estado explícito por flujo, sin almacenamiento compartido mutable."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class NeuralMemoryState:
    weights: tuple[Tensor, ...]
    momentum: tuple[Tensor, ...]
    steps: Tensor
    config_id: str
    # Ventanas causales de claves y valores proyectados, [lote, K − 1, dim]. Vacío sin
    # convolución, que es el estado anterior a la sección 4.4.
    convolution: tuple[Tensor, ...] = ()


@dataclass(frozen=True)
class MACState:
    memory: NeuralMemoryState
    config_id: str
    # Ventana causal de las consultas proyectadas, que calcula MAC. Vacío sin convolución.
    convolution: tuple[Tensor, ...] = ()


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
        tuple(copy(c) for c in state.convolution),
    )


def map_memory_rows(
    state: NeuralMemoryState, transform: Callable[[Tensor], Tensor]
) -> NeuralMemoryState:
    """Aplicar la misma operación por filas de flujo a todos los tensores del estado."""
    return replace(
        state,
        weights=tuple(transform(value) for value in state.weights),
        momentum=tuple(transform(value) for value in state.momentum),
        steps=transform(state.steps),
        convolution=tuple(transform(value) for value in state.convolution),
    )


def map_mac_rows(state: MACState, transform: Callable[[Tensor], Tensor]) -> MACState:
    return replace(
        state,
        memory=map_memory_rows(state.memory, transform),
        convolution=tuple(transform(value) for value in state.convolution),
    )


def mac_tensors(state: MACState) -> tuple[Tensor, ...]:
    """Tensores por flujo en orden fijo: pesos, momentum, pasos y ventanas de k, v y q."""
    memory = state.memory
    return (
        *memory.weights,
        *memory.momentum,
        memory.steps,
        *memory.convolution,
        *state.convolution,
    )


def join_mac_rows(states: Sequence[MACState]) -> MACState:
    """Concatenar los flujos de varios estados del mismo contrato en el orden recibido."""
    first = states[0]
    if any(
        state.config_id != first.config_id
        or state.memory.config_id != first.memory.config_id
        or len(mac_tensors(state)) != len(mac_tensors(first))
        for state in states
    ):
        raise ValueError("Solo se unen estados MAC del mismo contrato")

    def join(get):
        return torch.cat([get(state) for state in states])

    memory = first.memory
    return MACState(
        NeuralMemoryState(
            tuple(join(lambda s, i=i: s.memory.weights[i]) for i in range(len(memory.weights))),
            tuple(join(lambda s, i=i: s.memory.momentum[i]) for i in range(len(memory.momentum))),
            join(lambda s: s.memory.steps),
            memory.config_id,
            tuple(
                join(lambda s, i=i: s.memory.convolution[i]) for i in range(len(memory.convolution))
            ),
        ),
        first.config_id,
        tuple(join(lambda s, i=i: s.convolution[i]) for i in range(len(first.convolution))),
    )


def require_payload(payload: object, fields: set[str]) -> dict:
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("El estado serializado tiene campos incompatibles")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("La versión del estado serializado no está admitida")
    return payload
