"""Estadísticas causales por fila de la memoria y ganancias acotadas de plasticidad."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .plasticity_config import MemoryPlasticity
from .state import check_finite, require_true


@dataclass(frozen=True)
class PlasticityStep:
    history: tuple[Tensor, Tensor, Tensor]
    write_gain: Tensor
    capture_gain: Tensor
    coherence: Tensor


def initial_history(reference: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Media con la forma de los pesos y dos escalares independientes por fila."""
    shape = (*reference.shape[:-1], 1)
    return (
        reference.new_zeros(reference.shape),
        reference.new_zeros(shape),
        reference.new_zeros(shape),
    )


def validate_history(history, reference: Tensor) -> None:
    if type(history) is not tuple or len(history) != 3:
        raise ValueError("La plasticidad necesita media, energía y elegibilidad")
    for value, shape in zip(
        history,
        (reference.shape, (*reference.shape[:-1], 1), (*reference.shape[:-1], 1)),
        strict=True,
    ):
        if (
            not isinstance(value, Tensor)
            or value.shape != shape
            or value.dtype != reference.dtype
            or value.device != reference.device
            or value.layout != torch.strided
            or not value.is_contiguous()
        ):
            raise ValueError(
                "El estado de plasticidad tiene forma, tipo o dispositivo incompatible"
            )
        check_finite(value, "El estado de plasticidad")
    require_true((history[1] >= 0).all(), "La energía de plasticidad debe ser no negativa")
    require_true(
        ((history[2] >= 0) & (history[2] <= 1)).all(),
        "La elegibilidad de plasticidad debe estar entre cero y uno",
    )


def _validate(config, gradient, history, steps):
    if not isinstance(config, MemoryPlasticity):
        raise ValueError("Se necesita MemoryPlasticity")
    if (
        not isinstance(gradient, Tensor)
        or gradient.ndim != 3
        or gradient.shape[0] < 1
        or gradient.shape[1] < 1
        or gradient.shape[1] != gradient.shape[2]
        or gradient.dtype not in (torch.float32, torch.float64)
        or gradient.layout != torch.strided
    ):
        raise ValueError("El gradiente de plasticidad debe ser un lote de matrices cuadradas")
    if torch.is_autocast_enabled(gradient.device.type):
        raise ValueError("La plasticidad exige precisión explícita, sin autocast")
    info = torch.finfo(gradient.dtype)
    if not info.tiny <= config.epsilon <= info.max:
        raise ValueError("epsilon debe ser representable como número normal en la precisión usada")
    if (
        not isinstance(steps, Tensor)
        or steps.dtype != torch.int64
        or steps.shape != gradient.shape[:1]
        or steps.device != gradient.device
    ):
        raise ValueError("Los pasos de plasticidad tienen forma, tipo o dispositivo incompatible")
    require_true(
        ((steps >= 0) & (steps < torch.iinfo(torch.int64).max)).all(),
        "Los pasos de plasticidad están fuera del dominio",
    )
    check_finite(gradient, "El gradiente de plasticidad")
    validate_history(history, gradient)


def _gain(mode: str, signal: Tensor, floor: float, constant: float) -> Tensor:
    if mode == "none":
        return torch.ones_like(signal)
    if mode == "constant":
        return torch.full_like(signal, constant)
    return floor + (1 - floor) * signal


def advance(
    config: MemoryPlasticity, gradient: Tensor, history: tuple[Tensor, ...], steps: Tensor
) -> PlasticityStep:
    """Calcular un paso sin modificar entradas ni consultar objetivos financieros.

    La saturación entre cero y uno forma parte del algoritmo. Los NaN, infinitos
    y desbordamientos se rechazan antes de saturar. epsilon tiene unidades de
    gradiente al cuadrado y no representa una varianza inferida.
    """
    _validate(config, gradient, history, steps)
    return _advance_validated(config, gradient, history, steps)


def _advance_validated(config, gradient, history, steps):
    """Paso interno tras validar configuración, precisión, estado y contador en la frontera.

    NeuralMemory comprueba esa frontera antes de recorrer sus tokens. Los estados
    siguientes proceden de esta recurrencia. Se conservan todas las comprobaciones de
    valores intermedios, incluidos los desbordamientos de denominadores.
    """
    previous_mean, previous_energy, previous_tag = history
    squared = gradient.square().sum(dim=-1, keepdim=True)
    mean = config.beta * previous_mean + (1 - config.beta) * gradient
    energy = config.beta * previous_energy + (1 - config.beta) * squared
    mass = (
        torch.ones_like(steps, dtype=gradient.dtype)
        if config.beta == 0
        else -torch.expm1((steps.to(gradient.dtype) + 1) * math.log(config.beta))
    )
    mass = mass[:, None, None]
    corrected_energy = energy / mass
    coherence_denominator = mass * energy + config.epsilon
    activity_denominator = corrected_energy + config.epsilon
    for value in (
        squared,
        mean,
        energy,
        corrected_energy,
        coherence_denominator,
        activity_denominator,
    ):
        check_finite(value, "La actualización de plasticidad")
    raw_coherence = mean.square().sum(dim=-1, keepdim=True) / coherence_denominator
    raw_activity = squared / activity_denominator
    check_finite(raw_coherence, "La coherencia de plasticidad")
    check_finite(raw_activity, "La actividad de plasticidad")
    coherence = raw_coherence.clamp(0, 1)
    activity = raw_activity.clamp(0, 1)
    tag = config.trace_decay * previous_tag + (1 - config.trace_decay) * activity
    capture_signal = tag * coherence.mean(dim=1, keepdim=True)
    write_gain = _gain(config.write, coherence, config.write_floor, config.write_constant)
    capture_gain = _gain(
        config.capture, capture_signal, config.capture_floor, config.capture_constant
    )
    for value in (tag, write_gain, capture_gain):
        check_finite(value, "La ganancia de plasticidad")
    return PlasticityStep((mean, energy, tag), write_gain, capture_gain, coherence)
