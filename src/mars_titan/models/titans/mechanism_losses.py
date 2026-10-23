"""Pérdidas del catálogo sobre alternativas emitidas y etiquetas ya admitidas.

El interval score sigue la ecuación 43 de Gneiting y Raftery (2007).
Recibir tensores no acredita que la etiqueta haya madurado. Su colector debe
verificar esa procedencia antes de llamar y conservar alternativas no observadas.
"""

from dataclasses import dataclass

import torch

from .mechanism_policy import ACTIONS, POINT_ACTIONS
from .state import check_finite, deferred_checks, require_true


@dataclass(frozen=True)
class MechanismLosses:
    point_errors: torch.Tensor
    interval_scores: torch.Tensor
    coverage: torch.Tensor
    widths: torch.Tensor

    def gains(self):
        """Ganancias firmadas frente a off/k1/none, incluidas las negativas."""
        return (
            self.point_errors[:, :1] - self.point_errors,
            self.interval_scores[:, :1] - self.interval_scores,
        )


@torch.no_grad()
def mechanism_losses(quantiles, targets):
    """Una pérdida por alternativa y fila, sin promediar activos como sesiones.

    El orden de intervalos es 80 % y 95 %, con extremos incluidos para cobertura.
    No se aceptan alternativas ausentes como ceros. Las 36 deben estar presentes.
    La agregación por sesión y entre mercados corresponde al protocolo externo.
    """
    if (
        not isinstance(quantiles, torch.Tensor)
        or quantiles.layout != torch.strided
        or quantiles.ndim != 3
        or quantiles.shape[1:] != (len(ACTIONS), 5)
        or not 1 <= len(quantiles) <= 8192
        or quantiles.dtype not in (torch.float32, torch.float64)
        or not isinstance(targets, torch.Tensor)
        or targets.layout != torch.strided
        or targets.shape != (len(quantiles),)
        or targets.dtype != quantiles.dtype
        or targets.device != quantiles.device
    ):
        raise ValueError("Las alternativas y etiquetas no corresponden al contrato")
    if torch.is_autocast_enabled(quantiles.device.type):
        raise ValueError("Las pérdidas necesitan FP32 o FP64 explícitos, sin autocast")
    with deferred_checks():
        check_finite(quantiles, "Los cuantiles de las alternativas")
        check_finite(targets, "Las etiquetas admitidas")
        require_true(
            (quantiles[..., 1:] >= quantiles[..., :-1]).all(),
            "Los cuantiles de las alternativas están cruzados",
        )
        medians = quantiles[..., 2].reshape(len(quantiles), len(POINT_ACTIONS), 3)
        require_true(
            (medians == medians[..., :1]).all(),
            "Las calibraciones deben conservar la misma mediana por contexto y ruta",
        )
        point = (medians[..., 0] - targets[:, None]).abs()
        lower, upper = quantiles[..., [1, 0]], quantiles[..., [3, 4]]
        widths = upper - lower
        y = targets[:, None, None]
        penalty = quantiles.new_tensor([10.0, 40.0])
        scores = widths + penalty * ((lower - y).clamp_min(0) + (y - upper).clamp_min(0))
        interval = scores.mean(-1)
        coverage = (lower <= y) & (y <= upper)
        check_finite(point, "Los errores puntuales")
        check_finite(widths, "Las anchuras de los intervalos")
        check_finite(interval, "Los interval scores")
    return MechanismLosses(point, interval, coverage, widths)
