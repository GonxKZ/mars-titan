"""Separar las derivadas locales de mediana y colas sobre la representación final.

Es un diagnóstico de la cabeza, no de todo el gradiente del predictor. No actualiza
parámetros ni estima el efecto de una trayectoria de entrenamiento alternativa.
"""

import torch
from torch.nn import functional as F

from mars_titan.models.quantile_head import MEDIAN_INDEX, ordered_quantiles, pinball_loss


def feedback(encoded, weight, bias, target):
    """Devolver cuantiles y tres gradientes [B, D], sin conservar grafos históricos.

    La tarea principal es el término de la mediana de la pérdida común: L1/10.
    La parte auxiliar contiene los otros cuatro términos. Los tensores recibidos
    se separan de sus grafos y no se modifican ni reciben `.grad`.
    """
    tensors = (encoded, weight, bias, target)
    if (
        any(not isinstance(value, torch.Tensor) for value in tensors)
        or encoded.ndim != 2
        or not 1 <= encoded.shape[0] <= 4096
        or not 1 <= encoded.shape[1] <= 4096
        or encoded.dtype not in (torch.float32, torch.float64)
        or str(encoded.device) not in ("cpu", "cuda:0")
        or weight.shape != (5, encoded.shape[1])
        or bias.shape != (5,)
        or target.shape != (encoded.shape[0],)
        or any(value.dtype != encoded.dtype or value.device != encoded.device for value in tensors)
        or any(value.layout != torch.strided for value in tensors)
        or torch.is_inference_mode_enabled()
    ):
        raise ValueError("La cabeza necesita B×D, 5×D, 5 y B en FP32/FP64, con B y D hasta 4096")
    if any(not torch.isfinite(value).all().item() for value in tensors):
        raise ValueError("El diagnóstico no admite entradas NaN o Inf")
    with torch.enable_grad():
        local = encoded.detach().requires_grad_(True)
        raw = F.linear(local, weight.detach(), bias.detach())
        quantiles = ordered_quantiles(raw)
        if not torch.isfinite(quantiles).all().item():
            raise ValueError("La proyección de la cabeza produjo NaN o Inf")
        truth = target.detach()
        primary = F.l1_loss(quantiles[:, MEDIAN_INDEX], truth) / 10
        total = pinball_loss(quantiles, truth)
        primary_gradient = torch.autograd.grad(primary, local, retain_graph=True)[0]
        auxiliary_gradient = torch.autograd.grad(total - primary, local, retain_graph=True)[0]
        combined_gradient = torch.autograd.grad(total, local)[0]
    result = dict(
        quantiles=quantiles.detach(),
        primary=primary_gradient.detach(),
        auxiliary=auxiliary_gradient.detach(),
        combined=combined_gradient.detach(),
    )
    if any(not torch.isfinite(value).all().item() for value in result.values()):
        raise ValueError("La derivada local produjo NaN o Inf")
    return result
