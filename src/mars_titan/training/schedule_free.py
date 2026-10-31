"""AdamW sin calendario de tasa para F3 de #749 (`schedule_free_v1`).

Usa `schedulefree.AdamWScheduleFree` 1.4.1, de Defazio et al. (2024), The Road Less
Scheduled, arXiv:2405.15682, fijado en el extra `optimizers`. El optimizador mantiene tres
secuencias: z, el paso de tipo Adam; x, la media ponderada de los z; e
y = (1 − β₁) z + β₁ x, el punto donde se calcula el gradiente. Durante el ajuste, los
parámetros del predictor contienen y y el estado del optimizador guarda z.

La validación, la selección y la emisión usan x, como `optimizer.eval()`. Aquí x se calcula
fuera de sitio con lerp(y, z, 1 − 1/β₁), la misma operación que hace `eval()`, y los pesos
y se restauran por copia exacta. Con `eval()` y `train()` en su sitio, cada ida y vuelta
redondea y, y guardar un estado a mitad de época movería el ajuste según el reloj. El
optimizador permanece siempre en modo de ajuste.

Con la tasa constante del control, sin calentamiento, y los valores por defecto r = 0 y
weight_lr_power = 2, el peso de cada paso es c_{k+1} = 1/(k + 1) y x es la media uniforme
de los z, la de Polyak y Juditsky (1992). El predictor solo normaliza con LayerNorm y no
tiene estadísticas de lote que recalcular en x.
"""

import math
from dataclasses import dataclass
from importlib import metadata

import torch

SCHEDULE_FREE = "schedule_free_v1"
VERSION = "1.4.1"
# Valores de la biblioteca que no declara la receta. La tasa y el decaimiento son los suyos.
BETA2, EPS, R, WEIGHT_LR_POWER = 0.999, 1e-8, 0.0, 2.0


@dataclass(frozen=True)
class ScheduleFree:
    # Pasos de calentamiento lineal de la tasa. 0 conserva la tasa constante del control.
    warmup_steps: int = 0
    # Interpolación entre y y z. Con β₁ = 0,9, x pesa 0,9 en el punto del gradiente.
    beta1: float = 0.9

    def __post_init__(self):
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps <= 1_000_000:
            raise ValueError("El calentamiento es un número entero de pasos no negativo")
        if (
            isinstance(self.beta1, bool)
            or not isinstance(self.beta1, (int, float))
            or not math.isfinite(self.beta1)
            or not 0.0 < self.beta1 < 1.0
        ):
            raise ValueError("β₁ debe estar en (0, 1)")
        object.__setattr__(self, "beta1", float(self.beta1))

    def identity(self):
        return dict(
            schema_version=1,
            name=SCHEDULE_FREE,
            library=f"schedulefree=={VERSION}",
            optimizer="schedulefree.AdamWScheduleFree",
            betas=[self.beta1, BETA2],
            eps=EPS,
            warmup_steps=self.warmup_steps,
            r=R,
            weight_lr_power=WEIGHT_LR_POWER,
            learning_rate_and_weight_decay="those_of_the_recipe",
            validation_and_selection="averaged_point_x_computed_out_of_place",
            training="gradient_point_y",
        )

    def build(self, groups, *, learning_rate, weight_decay):
        """Optimizador en modo de ajuste, con la versión fijada de la biblioteca."""
        if metadata.version("schedulefree") != VERSION:
            raise ValueError(f"F3 necesita schedulefree=={VERSION}")
        from schedulefree import AdamWScheduleFree

        optimizer = AdamWScheduleFree(
            groups,
            lr=learning_rate,
            betas=(self.beta1, BETA2),
            eps=EPS,
            weight_decay=weight_decay,
            warmup_steps=self.warmup_steps,
            r=R,
            weight_lr_power=WEIGHT_LR_POWER,
        )
        # Sin estado todavía, train() solo cambia el modo. Nunca se vuelve a eval().
        optimizer.train()
        return optimizer


def is_schedule_free(optimizer):
    """Comprobar el tipo sin importar la biblioteca, que solo se instala con su extra."""
    kind = type(optimizer)
    return kind.__module__.startswith("schedulefree.") and kind.__name__ == "AdamWScheduleFree"


class AveragedPoint:
    """Punto x de cada parámetro ajustado, sin modificar los pesos y del ajuste."""

    def __init__(self, optimizer, named):
        if not named:
            raise ValueError("El punto promediado necesita los parámetros ajustados")
        self.optimizer = optimizer
        self.names = {id(value): name for name, value in named.items()}

    def emitted(self):
        result = {}
        for group in self.optimizer.param_groups:
            if not group["train_mode"]:
                raise ValueError("El optimizador sin calendario debe seguir en modo de ajuste")
            beta1 = group["betas"][0]
            for value in group["params"]:
                y = value.detach()
                z = self.optimizer.state[value].get("z")
                # Sin estado, antes del primer paso, x = y = z.
                result[self.names[id(value)]] = (
                    y.clone() if z is None else torch.lerp(y, z.to(y.device), 1 - 1 / beta1)
                )
        if set(result) != set(self.names.values()):
            raise ValueError("El optimizador no cubre exactamente los parámetros ajustados")
        return result
