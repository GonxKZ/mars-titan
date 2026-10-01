"""KLPO por tokens para una decisión discreta y su control de varianza exacta.

Ecuaciones 3.13 y 3.16 de Zhang et al. (2026), revisión del 20 de septiembre.
La política histórica permanece fija y los auxiliares se sortean independientemente.
"""

import math

import torch

from .predictive_adaptation import gaussian_log_probabilities

MODES = ("klpo_full", "klpo_mc", "klpo_exact")


def behavior_log_probabilities(parent, values, scale, epsilon):
    """Mezclar el padre congelado con una masa uniforme explícita, solo en q."""
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("La exploración del muestreador debe estar entre cero y uno")
    logq = gaussian_log_probabilities(parent.detach(), values, scale)
    return torch.logaddexp(
        logq + math.log1p(-epsilon), torch.full_like(logq, math.log(epsilon / len(values)))
    )


def _inputs(logp, logq, rewards, beta):
    if (
        type(beta) not in (int, float)
        or not math.isfinite(beta)
        or beta <= 0
        or not isinstance(logp, torch.Tensor)
        or logp.ndim != 2
        or not 1 <= logp.shape[0] <= 4096
        or not 2 <= logp.shape[1] <= 4096
    ):
        raise ValueError("KLPO requiere un lote acotado y beta positiva")
    for value in (logp, logq, rewards):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != logp.shape
            or value.device != logp.device
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Las distribuciones y recompensas de KLPO no son válidas")
    logp, logq, rewards = logp.double(), logq.detach().double(), rewards.detach().double()
    for value in (logp, logq):
        if not torch.allclose(value.logsumexp(1), torch.zeros_like(value[:, 0]), atol=1e-7, rtol=0):
            raise ValueError("KLPO necesita logaritmos de probabilidades normalizadas")
    # Multinomial normaliza sus pesos. La corrección debe usar esa misma q.
    logq = logq - logq.logsumexp(1, keepdim=True)
    if (logq.exp() == 0).any():
        raise ValueError("El muestreador necesita soporte positivo representable")
    return logp, logq, rewards


def _indices(value, shape, logp):
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != shape
        or value.device != logp.device
        or value.dtype not in (torch.int32, torch.int64)
        or (value < 0).any()
        or (value >= logp.shape[1]).any()
    ):
        raise ValueError("Los índices muestreados no pertenecen a la política")
    return value.long()


def token_loss(
    logp,
    logq,
    rewards,
    beta,
    mode,
    *,
    actions=None,
    auxiliaries=None,
    generator=None,
    auxiliary_generator=None,
    auxiliary_samples=128,
):
    """Devolver pérdida por fila y acciones. No dividir por el número de acciones.

    Full y MC son sustitutos de gradiente, no valores de la varianza exacta.
    Los índices explícitos permiten contrastar la esperanza por enumeración.
    """
    logp, logq, rewards = _inputs(logp, logq, rewards, beta)
    if (
        mode not in MODES
        or type(auxiliary_samples) is not int
        or not 1 <= auxiliary_samples <= 4096
        or (mode != "klpo_mc" and auxiliaries is not None)
        or (mode == "klpo_exact" and actions is not None)
    ):
        raise ValueError("La variante de KLPO o su presupuesto auxiliar no son válidos")
    q = logq.exp()
    h = rewards - beta * (logp - logq)
    if mode == "klpo_exact":
        centered = h - (q * h).sum(1, keepdim=True)
        loss = (q * centered.square()).sum(1) / (2 * beta)
    else:
        if actions is None:
            actions = torch.multinomial(q, 1, generator=generator).squeeze(1)
        actions = _indices(actions, (len(logp),), logp)
        if mode == "klpo_full":
            correction = (q * logp).sum(1)
        else:
            if auxiliaries is None:
                if auxiliary_generator is None or auxiliary_generator is generator:
                    raise ValueError("Los sorteos auxiliares requieren un RNG independiente")
                auxiliaries = torch.multinomial(
                    q, auxiliary_samples, replacement=True, generator=auxiliary_generator
                )
            auxiliaries = _indices(auxiliaries, (len(logp), auxiliary_samples), logp)
            correction = logp.gather(1, auxiliaries).mean(1)
        coefficient = h.gather(1, actions[:, None]).squeeze(1).detach()
        loss = -coefficient * (logp.gather(1, actions[:, None]).squeeze(1) - correction)
    if not torch.isfinite(loss).all():
        raise ValueError("La pérdida de KLPO ha desbordado la representación numérica")
    return loss, actions
