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


@dataclass(frozen=True)
class RegionalRouteCollection:
    routes: dict[str, RegionalRouteResult]
    refinements: int
    searches: int
    value_projections: int


@dataclass(frozen=True, kw_only=True)
class AdaptiveRegionalRouteCollection(RegionalRouteCollection):
    executed_reads: tuple[RegionalEpisodeRead, ...]


@dataclass(frozen=True)
class _Node:
    result: RegionalRouteResult
    positions: torch.Tensor | None


class _Decision:
    """Compartir prefijos y posiciones dentro de una llamada, sin cachear valores regionales."""

    def __init__(self, model, base, snapshot, context_id, cutoff, steps):
        if type(model) is not EpisodicReadout or model.config.evidence_bridge is None:
            raise ValueError("La ruta regional necesita un lector identificado")
        model._validate_read(base, snapshot, context_id, cutoff, refinements=steps)
        self.model, self.base, self.snapshot = model, base, snapshot
        self.context_id, self.cutoff = context_id, cutoff
        self.populated = snapshot is not None and snapshot.count > 0
        self.steps = self.searches = 0
        recurrence = model.config.evidence_bridge.recurrence
        self.executed_reads = (
            [] if recurrence is not None and recurrence.stopping is not None else None
        )

    def extend(self, parent=None, *, refresh=False, steps=1):
        state = self.base if parent is None else parent.result.state
        reads = () if parent is None else parent.result.reads
        positions = None if parent is None or refresh else parent.positions
        searches = 0 if parent is None else parent.result.searches
        for _ in range(steps):
            searching = int(self.populated and positions is None)
            read, positions, _ = self.model._read_parts(
                state,
                self.snapshot,
                self.context_id,
                self.cutoff,
                positions=positions,
                repeated=True,
            )
            state = self.model._step(state, self.base, read)
            reads += (read,)
            if self.executed_reads is not None:
                self.executed_reads.append(read)
            searches += searching
            self.steps += 1
            self.searches += searching
        return _Node(
            RegionalRouteResult(
                state, reads, len(reads), searches, len(reads) if self.populated else 0
            ),
            positions,
        )


def run_route_pair(
    model, base, snapshot=None, *, route, context_id=None, cutoff=None, differentiable=False
):
    """Compartir K1 y ejecutar solo la ruta solicitada, sin avanzar MAC ni escribir episodios.

    C conserva posiciones y R las renueva al empezar su bloque. Se recalculan los
    valores proyectados en cada lectura. C no designa aquí el factor de CM-v1.
    """
    if type(route) is not str or route not in ROUTES:
        raise ValueError("La ruta regional necesita un lector y una acción identificados")
    _check_differentiation(differentiable)
    with deferred_checks(), torch.set_grad_enabled(differentiable):
        base = _checked_base(base)
        base = base if differentiable else base.detach()
        decision = _Decision(model, base, snapshot, context_id, cutoff, int(route[1]))
        node = decision.extend()
        first = node.result
        for step in range(1, int(route[1])):
            node = decision.extend(node, refresh=step in REFRESH_AT[route])
        return first, node.result


def collect_routes(
    model, base, snapshot=None, *, context_id=None, cutoff=None, differentiable=False
):
    """Recoger seis rutas con nueve refinamientos y tres búsquedas si hay episodios.

    Cada nodo vuelve a proyectar los valores regionales. Compartir un prefijo común
    evita repetir su cálculo, sin convertir las vistas en bancos persistentes.
    """
    _check_differentiation(differentiable)
    with deferred_checks(), torch.set_grad_enabled(differentiable):
        base = _checked_base(base)
        base = base if differentiable else base.detach()
        decision = _Decision(model, base, snapshot, context_id, cutoff, 9)
        first = decision.extend()
        retained = decision.extend(first)
        renewed = decision.extend(first, refresh=True)
        nodes = (
            first,
            retained,
            renewed,
            decision.extend(retained, steps=2),
            decision.extend(retained, refresh=True, steps=2),
            decision.extend(renewed, steps=2),
        )
        fields = (
            dict(zip(ROUTES, (node.result for node in nodes), strict=True)),
            decision.steps,
            decision.searches,
            decision.steps if decision.populated else 0,
        )
        if decision.executed_reads is not None:
            return AdaptiveRegionalRouteCollection(
                *fields, executed_reads=tuple(decision.executed_reads)
            )
        return RegionalRouteCollection(*fields)
