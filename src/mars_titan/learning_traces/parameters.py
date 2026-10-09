"""Normas de gradientes, pesos y actualizaciones por grupo de parámetros.

Los grupos se declaran por prefijos de nombre (por ejemplo `memory.` para los pesos de la
memoria neuronal o `head.` para la cabeza). Cada parámetro entrenable pertenece a un único
grupo. Los que no coinciden con ningún prefijo van a `other`, y el informe lo muestra.

Las normas se acumulan en float64 en el dispositivo de cada parámetro y se entregan al
registrador como escalares de tensor, de modo que no hay sincronización hasta el volcado.
La razón de actualización `||W_t - W_{t-1}|| / ||W_{t-1}||` necesita una copia de los pesos
antes del paso, que `UpdateProbe` solo toma en los pasos que tocan registro.
"""

import torch

OTHER = "other"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def group_parameters(named_parameters, groups):
    """Repartir los parámetros entrenables en grupos disjuntos por prefijo."""
    _require(isinstance(groups, dict) and OTHER not in groups, "Grupos de parámetros inválidos")
    result = {name: [] for name in groups}
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        matches = [g for g, prefixes in groups.items() if name.startswith(tuple(prefixes))]
        _require(len(matches) <= 1, f"El parámetro {name} cae en varios grupos: {matches}")
        result.setdefault(matches[0] if matches else OTHER, []).append((name, parameter))
    return {name: values for name, values in result.items() if values}


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
