"""Orden de visita de las cohortes reales en cada época del ajuste.

Cada época recorre todas las cohortes reales de entrenamiento en la permutación del
generador `[seed, epoch]`. El postentrenamiento solo con datos reales usa este orden sin
depender de los módulos de aumento, que lo amplían con bloques adicionales tras las
cohortes reales y con el mismo generador.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Visit:
    arm: str
    episode: int
    cohort: int
    reset: bool


def real_order(rng, count):
    """Todas las cohortes reales una vez, en la permutación del generador de la época."""
    return [Visit("real", -1, int(i), True) for i in rng.permutation(count)]


def validation_order(count):
    """Validación real en orden cronológico, sin permutar."""
    return [Visit("real", -1, i, True) for i in range(count)]
