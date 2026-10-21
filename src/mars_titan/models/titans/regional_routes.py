"""Rutas regionales con posiciones comunes y proyecciones recalculadas por lectura."""

from dataclasses import dataclass

import torch

from .episodic_readout import EpisodicReadout
from .refinement_routes import ROUTES, _check_differentiation, _checked_base
from .regional_episode_readout import RegionalEpisodeRead
from .state import deferred_checks

REFRESH_AT = {"k1": (), "k2c": (), "k2r": (1,), "k4cc": (), "k4cr": (2,), "k4rc": (1,)}


@dataclass(frozen=True)
class RegionalRouteResult:
    state: torch.Tensor
    reads: tuple[RegionalEpisodeRead, ...]
    refinements: int
    searches: int
    value_projections: int


def run_route_pair(
    model, base, snapshot=None, *, route, context_id=None, cutoff=None, differentiable=False
):
    """Compartir K1 y ejecutar solo la ruta solicitada, sin avanzar MAC ni escribir episodios.

    C conserva posiciones y R las renueva al empezar su bloque. Se recalculan los
    valores proyectados en cada lectura. C no designa aquí el factor de CM-v1.
    """
    if (
        type(model) is not EpisodicReadout
        or model.config.evidence_bridge is None
        or type(route) is not str
        or route not in ROUTES
    ):
        raise ValueError("La ruta regional necesita un lector y una acción identificados")
    _check_differentiation(differentiable)
    with deferred_checks(), torch.set_grad_enabled(differentiable):
        base = _checked_base(base)
        base = base if differentiable else base.detach()
        model._validate_read(base, snapshot, context_id, cutoff, refinements=int(route[1]))
        state, positions, reads, searches = base, None, (), 0
        refresh = REFRESH_AT[route]
        populated = snapshot is not None and snapshot.count > 0
        for step in range(int(route[1])):
            if step in refresh:
                positions = None
            searches += int(populated and positions is None)
            read, positions, _ = model._read_parts(
                state, snapshot, context_id, cutoff, positions=positions, repeated=True
            )
            state = model._step(state, base, read)
            reads += (read,)
            result = RegionalRouteResult(
                state, reads, step + 1, searches, step + 1 if populated else 0
            )
            if step == 0:
                first = result
        return first, result
