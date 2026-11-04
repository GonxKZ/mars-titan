"""Contracción y ruido al empezar una continuación (F3 de #749, `shrink_perturb_v1`).

Antes del primer paso de una ejecución nueva, cada parámetro ajustado θ pasa a

    θ ← λ θ + s θ₀,

con θ₀ el mismo parámetro de una inicialización nueva del predictor. Así el ruido sigue la
distribución de inicialización de cada tipo de parámetro, como en Ash y Adams (2020), On
warm-starting neural network training, NeurIPS 33, arXiv:1910.08475, sección 4. El artículo
usa, por ejemplo, λ = 0,6 y s = 0,01 en el aprendizaje en línea.

La semilla de θ₀ se deriva con `SeedSequence` de la semilla declarada y de la del ajuste, así
que cada semilla de ajuste recibe su propia inicialización nueva y la media entre semillas
también promedia el ruido. La inicialización se construye con el generador de la
configuración del predictor y no consume el RNG global del ajuste. Una reanudación no vuelve
a aplicar la transformación porque el estado guardado ya la contiene.

λ = 1 y s = 0 es la continuación sin cambios, el control. λ < 1 con s = 0 solo contrae y
λ = 1 con s > 0 solo perturba. Son contrastes distintos con su propia identidad.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

SHRINK_PERTURB = "shrink_perturb_v1"


def _fraction(value, name, low, high):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} debe estar en [{low}, {high}]")
    return float(value)


@dataclass(frozen=True)
class ShrinkPerturb:
    shrink: float = 0.6
    noise: float = 0.01
    # Semilla declarada de θ₀. La de cada ajuste se deriva de ella y de la semilla del ajuste.
    seed: int = 1

    def __post_init__(self):
        object.__setattr__(self, "shrink", _fraction(self.shrink, "λ", 0.0, 1.0))
        object.__setattr__(self, "noise", _fraction(self.noise, "s", 0.0, 1.0))
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("La semilla de θ₀ debe ser un entero de 32 bits")
        if self.shrink == 0.0:
            raise ValueError("Con λ = 0 se descartaría al padre entero")

    def identity(self):
        return dict(
            schema_version=1,
            name=SHRINK_PERTURB,
            shrink=self.shrink,
            noise=self.noise,
            seed=self.seed,
            fresh_seed="seed_sequence_of_the_declared_seed_and_the_fit_seed",
            noise_source="same_parameter_of_a_fresh_predictor_initialization",
            scope="trainable_parameters_of_the_optimizer",
            applied="once_before_the_first_step_of_a_new_run",
        )

    def fresh_seed(self, fit_seed):
        """Semilla de θ₀ de un ajuste, distinta en cada semilla de ajuste y reproducible."""
        if type(fit_seed) is not int or not 0 <= fit_seed < 2**32:
            raise ValueError("La semilla del ajuste debe ser un entero de 32 bits")
        return int(np.random.SeedSequence([self.seed, fit_seed]).generate_state(1)[0])


@torch.no_grad()
def shrink_perturb(named, fresh, config):
    """Aplicar θ ← λθ + sθ₀ en su sitio a los parámetros `named` con los de `fresh`."""
    if type(config) is not ShrinkPerturb or not named or set(named) - set(fresh):
        raise ValueError("La contracción necesita su configuración y θ₀ de cada parámetro")
    for name in sorted(named):
        value, initial = named[name], fresh[name]
        if initial.shape != value.shape or initial.dtype != value.dtype:
            raise ValueError(f"θ₀ no conserva la forma o el tipo de {name}")
        value.mul_(config.shrink).add_(initial.to(value.device), alpha=config.noise)
