"""Adaptador residual común y tres objetivos predictivos con información completa."""

import math

import numpy as np
import torch
from torch import nn


def _policy_inputs(centers, values, scale):
    if (
        not isinstance(centers, torch.Tensor)
        or centers.ndim != 1
        or not 1 <= len(centers) <= 4096
        or not centers.is_floating_point()
        or not isinstance(values, torch.Tensor)
        or values.shape != (21,)
        or not values.is_floating_point()
        or values.device != centers.device
        or type(scale) not in (int, float)
        or not math.isfinite(scale)
        or scale <= 0
        or not torch.isfinite(centers).all()
        or not torch.isfinite(values).all()
        or not torch.all(torch.diff(values) > 0)
    ):
        raise ValueError("Los centros, acciones o escala de la política no son válidos")
    return centers.double(), values.double()


def gaussian_log_probabilities(centers, values, scale):
    """Calcular logits gaussianos sin el término común mu², con anchura fija."""
    centers, values = _policy_inputs(centers, values, scale)
    z, m = values / scale, centers / scale
    logits = z[None, :] * m[:, None] - 0.5 * z.square()[None, :]
    result = torch.log_softmax(logits, dim=1)
    if not torch.isfinite(result).all():
        raise ValueError("Los logits han desbordado la representación float64")
    return result


def objective(centers, targets, values, scale, mode, *, actions=None, generator=None):
    """Devolver pérdidas por fila. REINFORCE minimiza costes, no su negativo."""
    centers, values = _policy_inputs(centers, values, scale)
    if (
        mode not in {"reinforce", "expected", "mae"}
        or not isinstance(targets, torch.Tensor)
        or targets.shape != centers.shape
        or targets.device != centers.device
        or not targets.is_floating_point()
        or targets.requires_grad
        or not torch.isfinite(targets).all()
        or (mode != "reinforce" and actions is not None)
    ):
        raise ValueError("La modalidad de ajuste o las etiquetas no son válidas")
    targets = targets.double()
    if mode == "mae":
        loss = (centers - targets).abs() / scale
    else:
        log_probability = gaussian_log_probabilities(centers, values, scale)
        probability = log_probability.exp()
        costs = (values[None, :] - targets[:, None]).abs() / scale
        expected = (probability * costs).sum(dim=1)
        if mode == "expected":
            loss = expected
        else:
            if actions is None:
                actions = torch.multinomial(probability, 1, generator=generator).squeeze(1)
            if (
                not isinstance(actions, torch.Tensor)
                or actions.shape != centers.shape
                or actions.device != centers.device
                or actions.dtype not in {torch.int32, torch.int64}
                or (actions < 0).any()
                or (actions >= 21).any()
            ):
                raise ValueError("Las acciones muestreadas no pertenecen a la rejilla")
            selected = actions.long()[:, None]
            advantage = costs.gather(1, selected).squeeze(1).detach() - expected.detach()
            loss = advantage * log_probability.gather(1, selected).squeeze(1)
    if not torch.isfinite(loss).all():
        raise ValueError("La pérdida de adaptación no es finita")
    return loss, actions


class LinearResidualPolicy(nn.Module):
    """Ajustar una corrección común sin modificar las predicciones del padre almacenadas."""

    def __init__(self, mean, scale, *, target_scale):
        super().__init__()
        mean, scale = np.asarray(mean), np.asarray(scale)
        if (
            mean.ndim != 1
            or not 1 <= len(mean) <= 16384
            or scale.shape != mean.shape
            or np.iscomplexobj(mean)
            or np.iscomplexobj(scale)
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or (scale <= 0).any()
            or type(target_scale) not in (int, float)
            or not math.isfinite(target_scale)
            or target_scale <= 0
        ):
            raise ValueError("La normalización y la escala de retorno no son válidas")
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float64))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float64))
        self.target_scale = float(target_scale)
        self.correction = nn.Linear(len(mean), 1, dtype=torch.float32)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, features, parent):
        if (
            features.ndim != 2
            or features.shape[1] != len(self.mean)
            or not 1 <= len(features) <= 4096
            or parent.shape != (len(features),)
            or features.device != self.mean.device
            or parent.device != self.mean.device
            or not features.is_floating_point()
            or not parent.is_floating_point()
            or not torch.isfinite(features).all()
            or not torch.isfinite(parent).all()
        ):
            raise ValueError("Las entradas y el padre no forman un lote válido del adaptador")
        standardized = ((features.detach().double() - self.mean) / self.scale).float()
        if not torch.isfinite(standardized).all():
            raise ValueError("La estandarización del adaptador ha desbordado float32")
        result = (
            parent.detach().double()
            + self.target_scale * self.correction(standardized).squeeze(1).double()
        )
        if not torch.isfinite(result).all():
            raise ValueError("El centro predictivo del adaptador no es finito")
        return result
