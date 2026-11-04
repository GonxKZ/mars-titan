"""Pesos con que se valida y se emite un ajuste: media exponencial de F2 (#749).

Tras cada paso del optimizador, cada parámetro ajustado θ actualiza su sombra

    θ̄ ← δ θ̄ + (1 − δ) θ,    con θ̄ = θ al empezar,

mediante `torch.optim.swa_utils.get_ema_multi_avg_fn`, la función pública de PyTorch que usa
`AveragedModel`. No se copia el predictor entero. Solo se guardan las sombras de los
parámetros que ajusta el optimizador, en su mismo dispositivo y tipo.

La validación y la selección usan las sombras. El ajuste sigue con los pesos propios, que
`applied` restaura por copia exacta al terminar de evaluar. El estado elegido guarda las
sombras como pesos del modelo, de modo que quien lo cargue predice con la media sin conocer
este módulo.

Fuente: Polyak y Juditsky (1992), Acceleration of stochastic approximation by averaging,
SIAM Journal on Control and Optimization 30(4), https://doi.org/10.1137/0330046, para el
promedio de iterados. La forma exponencial es la de PyTorch. La semivida en pasos es
log(0,5)/log(δ).
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch.optim.swa_utils import get_ema_multi_avg_fn

WEIGHT_EMA = "weight_ema_v1"


@dataclass(frozen=True)
class WeightAverage:
    # Retención de la sombra en cada paso. 0,999 equivale a una semivida de 693 pasos.
    decay: float = 0.999

    def __post_init__(self):
        if (
            isinstance(self.decay, bool)
            or not isinstance(self.decay, (int, float))
            or not math.isfinite(self.decay)
            or not 0.0 < self.decay < 1.0
        ):
            raise ValueError("La media exponencial necesita 0 < δ < 1")
        object.__setattr__(self, "decay", float(self.decay))

    def identity(self):
        return dict(
            schema_version=1,
            name=WEIGHT_EMA,
            decay=self.decay,
            half_life_steps=math.log(0.5) / math.log(self.decay),
            update="torch_swa_utils_ema_after_each_optimizer_step",
            scope="trainable_parameters_of_the_optimizer",
            validation_and_selection="shadow_weights",
            training="own_weights",
        )


class ShadowWeights:
    """Sombras de los parámetros ajustados, en el orden de sus nombres."""

    def __init__(self, config, named):
        if type(config) is not WeightAverage or not named:
            raise ValueError("Las sombras necesitan su configuración y parámetros ajustados")
        self.config, self.names = config, sorted(named)
        self.shadow = [named[name].detach().clone() for name in self.names]
        self.updates = 0
        self._update = get_ema_multi_avg_fn(config.decay)

    def update(self, named):
        """Incorporar los pesos tras un paso. Se llama una vez por paso confirmado."""
        self._update(self.shadow, [named[name].detach() for name in self.names], self.updates)
        self.updates += 1

    def emitted(self):
        """Pesos con que se valida y se emite, por nombre."""
        return dict(zip(self.names, self.shadow, strict=True))

    def state_dict(self):
        return dict(
            schema_version=1,
            configuration=self.config.identity(),
            names=list(self.names),
            shadow=[value.detach().cpu().clone() for value in self.shadow],
            updates=self.updates,
        )

    def load_state_dict(self, state):
        if (
            type(state) is not dict
            or state.get("schema_version") != 1
            or state.get("configuration") != self.config.identity()
            or state.get("names") != self.names
            or type(state.get("updates")) is not int
            or state["updates"] < 0
            or len(state.get("shadow", ())) != len(self.shadow)
        ):
            raise ValueError("Las sombras guardadas no corresponden a este ajuste")
        for target, value in zip(self.shadow, state["shadow"], strict=True):
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError("Una sombra guardada no conserva su forma o su tipo")
            target.copy_(value)
        self.updates = state["updates"]


@contextmanager
def applied(named, emitted):
    """Sustituir temporalmente los pesos propios por `emitted` y restaurarlos por copia exacta."""
    if not emitted or set(emitted) - set(named):
        raise ValueError("Los pesos emitidos deben corresponder a parámetros del predictor")
    own = {name: named[name].detach().clone() for name in emitted}
    try:
        with torch.no_grad():
            for name, value in emitted.items():
                named[name].copy_(value)
        yield
    finally:
        with torch.no_grad():
            for name, value in own.items():
                named[name].copy_(value)
