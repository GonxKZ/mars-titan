"""Objetivos puntuales de control sin cambiar la cabeza de cinco salidas.

Las cuatro colas siguen presentes, pero no reciben señal de ajuste. Sus valores
no constituyen cuantiles aprendidos. Estos controles no sustituyen una cabeza
escalar y no están activados en las recetas históricas.
"""

import torch
from torch.nn import functional

from mars_titan.models.quantile_head import LEVELS, PINBALL, median

POINT_MAE = "quantile_point_mae_v1"
POINT_COMPONENT = "quantile_point_component_v1"
POINT_OBJECTIVES = (POINT_MAE, POINT_COMPONENT)
QUANTILE_OBJECTIVES = (PINBALL, *POINT_OBJECTIVES)


def point_objective(prediction, target, kind):
    """L1 de la mediana o su contribución L1/10 a la media de cinco pinball.

    Devuelve una media escalar con el mismo dtype y dispositivo. No corrige
    valores no finitos. El consumidor debe rechazar datos o gradientes inválidos.
    """
    if not isinstance(kind, str) or kind not in POINT_OBJECTIVES:
        raise ValueError("El objetivo puntual necesita un nombre de control explícito")
    if (
        not isinstance(prediction, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or prediction.ndim != 2
        or target.ndim != 1
        or prediction.shape != (len(target), len(LEVELS))
        or len(target) == 0
    ):
        raise ValueError("El control necesita cinco salidas por fila y etiquetas no vacías")
    if (
        prediction.dtype != target.dtype
        or prediction.device != target.device
        or not prediction.dtype.is_floating_point
    ):
        raise ValueError("Salidas y etiquetas deben compartir precisión real y dispositivo")
    loss = functional.l1_loss(median(prediction), target)
    return loss / 10 if kind == POINT_COMPONENT else loss
