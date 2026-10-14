"""Normas de gradientes, pesos y actualizaciones por grupo de parámetros.

Los grupos siguen la misma regla en todas las familias: la clave del módulo (por ejemplo
`predictor` o `readout`) y el primer componente del nombre del parámetro, como
`predictor.memory` o `readout.bank`. Así no hace falta declarar grupos por familia y cada
parámetro entrenable cae en un único grupo.

Las normas se acumulan en float64 en el dispositivo de cada parámetro y se entregan al
registrador como escalares de tensor, de modo que no hay sincronización hasta el volcado.
La razón de actualización `||W_t - W_{t-1}|| / ||W_{t-1}||` necesita una copia de los pesos
antes del paso, que `UpdateProbe` solo toma en los pasos que tocan registro.
"""

import torch

from .recorder import trace_name


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def parameter_groups(modules):
    """Parámetros entrenables por clave de módulo y primer componente de su nombre.

    La GRU candidata nativa devuelve un diccionario en `named_parameters()`, en lugar del
    iterador de pares de PyTorch.
    """
    grouped = {}
    for key, module in modules.items():
        named = module.named_parameters()
        for name, parameter in named.items() if isinstance(named, dict) else named:
            if parameter.requires_grad:
                group = trace_name(f"{key}.{name.split('.', 1)[0]}")
                grouped.setdefault(group, []).append((name, parameter))
    return grouped


def _l2(tensors):
    norms = [torch.linalg.vector_norm(t.detach(), dtype=torch.float64) for t in tensors]
    return torch.linalg.vector_norm(torch.stack(norms)) if norms else None


def record_gradients(recorder, grouped):
    """Norma L2 del gradiente y de los pesos de cada grupo."""
    if not recorder.active:
        return
    for group, parameters in grouped.items():
        gradients = [p.grad for _, p in parameters if p.grad is not None]
        if gradients:
            recorder.scalar("grad_l2", _l2(gradients), group=group)
        recorder.scalar("weight_l2", _l2([p for _, p in parameters]), group=group)


class UpdateProbe:
    """Copia los pesos antes del paso y mide la actualización relativa después."""

    def __init__(self):
        self._before = None

    def before(self, grouped):
        self._before = {
            group: [p.detach().clone() for _, p in parameters]
            for group, parameters in grouped.items()
        }

    def after(self, recorder, grouped):
        _require(self._before is not None, "Falta la copia anterior al paso")
        before, self._before = self._before, None
        for group, parameters in grouped.items():
            old = before[group]
            delta = _l2([p.detach() - o for (_, p), o in zip(parameters, old, strict=True)])
            recorder.scalar("update_ratio", delta / _l2(old).clamp(min=1e-300), group=group)
