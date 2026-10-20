"""Rutas de refinamiento y consulta sobre una decisión ya preparada.

C conserva episodios y proyecciones, pero recalcula la atención. R selecciona
otros episodios antes de refinar. Ninguna ruta escribe ni avanza el núcleo.
"""

from dataclasses import dataclass

import torch

from .episodic_readout import EpisodicReadout
from .state import check_differentiable, check_finite, deferred_checks

# Los dos últimos refinamientos forman un bloque. No hay salida K3 ni ruta RR.
ROUTES = ("k1", "k2c", "k2r", "k4cc", "k4cr", "k4rc")
_ACTIONS = {
    "k1": (),
    "k2c": (False,),
    "k2r": (True,),
    "k4cc": (False, False),
    "k4cr": (False, True),
    "k4rc": (True, False),
}


@dataclass(frozen=True)
class RouteResult:
    state: torch.Tensor
    reads: tuple
    refinements: int
    searches: int
    value_projections: int


@dataclass(frozen=True)
class RouteCollection:
    routes: dict[str, RouteResult]
    refinements: int
    searches: int
    value_projections: int


@dataclass(frozen=True)
class _Node:
    result: RouteResult
    cache: object


class _Decision:
    """Cachés locales a una llamada. No se pueden reanudar ni reutilizar fuera de ella."""

    def __init__(self, model, base, snapshot, context_id, cutoff, steps):
        if type(model) is not EpisodicReadout:
            raise ValueError("La ruta necesita un lector episódico identificado")
        # El presupuesto corresponde a la ruta, no al K declarado en el checkpoint.
        model._validate_read(base, snapshot, context_id, cutoff, refinements=steps)
        self.model, self.base, self.snapshot = model, base, snapshot
        self.context_id, self.cutoff = context_id, cutoff
        self.steps = self.searches = self.value_projections = 0
        check_finite(model.step_logit, "La puerta de refinamiento")

    def extend(self, parent=None, *, refresh=False, steps=1, base=None):
        base = self.base if base is None else base
        state = base if parent is None else parent.result.state
        reads = () if parent is None else parent.result.reads
        cache = None if parent is None or refresh else parent.cache
        prior_searches = 0 if parent is None else parent.result.searches
        performed = 0
        for _ in range(steps):
            searching = cache is None and self.snapshot is not None and self.snapshot.count > 0
            read, _, cache = self.model._read_parts(
                state, self.snapshot, self.context_id, self.cutoff, repeated=True, cached=cache
            )
            state = self.model._step(state, base, read)
            reads += (read,)
            self.steps += 1
            performed += int(searching)
        self.searches += performed
        self.value_projections += performed
        return _Node(
            RouteResult(
                state, reads, len(reads), prior_searches + performed, prior_searches + performed
            ),
            cache,
        )


def _check_differentiation(differentiable):
    check_differentiable(differentiable)
    if differentiable and torch.is_inference_mode_enabled():
        raise ValueError("inference_mode no permite la diferenciación solicitada")


def _checked_base(base):
    if not isinstance(base, torch.Tensor):
        raise ValueError("El estado de trabajo debe ser un tensor")
    return base


def run_route(
    model, base, snapshot=None, *, route, context_id=None, cutoff=None, differentiable=False
):
    """Ejecutar una ruta identificada sin calcular las alternativas descartadas."""
    return run_route_pair(
        model,
        base,
        snapshot,
        route=route,
        context_id=context_id,
        cutoff=cutoff,
        differentiable=differentiable,
    )[1]


def run_route_pair(
    model, base, snapshot=None, *, route, context_id=None, cutoff=None, differentiable=False
):
    """Devolver K1 y la ruta final compartiendo su prefijo, sin otra transición del núcleo."""
    if type(route) is not str or route not in ROUTES:
        raise ValueError("La ruta debe ser k1, k2c, k2r, k4cc, k4cr o k4rc")
    _check_differentiation(differentiable)
    with deferred_checks(), torch.set_grad_enabled(differentiable):
        base = _checked_base(base)
        base = base if differentiable else base.detach()
        decision = _Decision(model, base, snapshot, context_id, cutoff, int(route[1]))
        node = decision.extend()
        first = node.result
        for index, refresh in enumerate(_ACTIONS[route]):
            node = decision.extend(node, refresh=refresh, steps=1 if index == 0 else 2)
        return first, node.result


def collect_routes(
    model, base, snapshot=None, *, context_id=None, cutoff=None, differentiable=False
):
    """Calcular seis alternativas con prefijos compartidos para comparaciones de desarrollo.

    La colección necesita nueve refinamientos y, con banco, tres búsquedas.
    No representa el coste de ejecutar una única ruta ni el de una política.
    """
    _check_differentiation(differentiable)
    with deferred_checks(), torch.set_grad_enabled(differentiable):
        base = _checked_base(base)
        base = base if differentiable else base.detach()
        decision = _Decision(model, base, snapshot, context_id, cutoff, 9)
        first = decision.extend()
        cached = decision.extend(first)
        renewed = decision.extend(first, refresh=True)
        nodes = (
            first,
            cached,
            renewed,
            decision.extend(cached, steps=2),
            decision.extend(cached, refresh=True, steps=2),
            decision.extend(renewed, steps=2),
        )
        return RouteCollection(
            dict(zip(ROUTES, (node.result for node in nodes), strict=True)),
            decision.steps,
            decision.searches,
            decision.value_projections,
        )
