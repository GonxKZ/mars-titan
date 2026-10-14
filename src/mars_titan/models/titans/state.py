"""Estado explícito por flujo, sin almacenamiento compartido mutable."""

import math
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class NeuralMemoryState:
    weights: tuple[Tensor, ...]
    momentum: tuple[Tensor, ...]
    steps: Tensor
    config_id: str
    # Ventanas causales de claves y valores proyectados, [lote, K − 1, dim]. Vacío sin
    # convolución, que es el estado anterior a la sección 4.4.
    convolution: tuple[Tensor, ...] = ()
    # Niveles ocultos de PT5, dos por capa en orden (capa, nivel), [lote, dim, dim]. Vacío
    # sin memoria multiescala.
    levels: tuple[Tensor, ...] = ()


@dataclass(frozen=True)
class MACState:
    memory: NeuralMemoryState
    config_id: str
    # Ventana causal de las consultas proyectadas, que calcula MAC. Vacío sin convolución.
    convolution: tuple[Tensor, ...] = ()


def check_differentiable(differentiable: bool) -> None:
    if type(differentiable) is not bool:
        raise ValueError("differentiable debe ser booleano")


class _PendingChecks:
    """Comprobaciones de un tramo de cálculo en CUDA, resueltas con una sola sincronización.

    Cada comprobación guarda su indicador como tensor del dispositivo y su mensaje en orden
    de programa. Un indicador booleano debe ser cierto y uno real, la norma infinito de un
    tensor, debe ser finito. Un mismo tensor sin cambios solo se comprueba una vez, y se
    conserva su referencia para que su identificador no se reutilice dentro del tramo.
    """

    def __init__(self) -> None:
        self.flags: list[Tensor] = []
        self.messages: list[str] = []
        self.seen: dict[tuple[int, int], Tensor] = {}

    def add(self, flag: Tensor, message: str) -> None:
        self.flags.append(flag)
        self.messages.append(message)

    def first_failure(self) -> str | None:
        """Mensaje de la primera comprobación fallida en orden de programa, o None.

        Los indicadores se apilan por tipo y pasan al anfitrión con una sola copia. No se
        descartan, así que un CUDA Graph que los captura puede volver a leerlos tras cada
        repetición.
        """
        if not self.flags:
            return None
        groups: dict[torch.dtype, list[int]] = {}
        for index, flag in enumerate(self.flags):
            groups.setdefault(flag.dtype, []).append(index)
        parts, order = [], []
        for dtype, indices in groups.items():
            stacked = torch.stack([self.flags[index] for index in indices])
            parts.append(stacked if dtype == torch.bool else torch.isfinite(stacked))
            order.extend(indices)
        values = (parts[0] if len(parts) == 1 else torch.cat(parts)).tolist()
        passed = [True] * len(self.flags)
        for index, ok in zip(order, values, strict=True):
            passed[index] = ok
        for ok, message in zip(passed, self.messages, strict=True):
            if not ok:
                return message
        return None

    def resolve(self) -> None:
        message = self.first_failure()
        self.flags, self.messages = [], []
        if message is not None:
            raise ValueError(message)


_PENDING: ContextVar[_PendingChecks | None] = ContextVar("titans_pending_checks", default=None)


@contextmanager
def deferred_checks():
    """Diferir las comprobaciones de datos en CUDA hasta el final del bloque `with`.

    El error es el mismo `ValueError` con el mismo mensaje que la comprobación inmediata y
    se lanza antes de devolver nada, solo que al final del tramo en lugar de en su línea.
    Si el tramo lanza otro error, antes se resuelven las comprobaciones previas, como si se
    hubieran hecho en su sitio. Dentro de otro tramo diferido no abre uno nuevo: sus
    comprobaciones se resuelven al final del exterior. Los tensores que recuerda para no
    repetirlas solo se conservan mientras dura el tramo interior, así que un recorrido largo
    no retiene los temporales de cada paso hasta el final.
    """
    outer = _PENDING.get()
    if outer is not None:
        seen, outer.seen = outer.seen, {}
        try:
            yield
        finally:
            outer.seen = seen
        return
    pending = _PendingChecks()
    token = _PENDING.set(pending)
    try:
        yield
    except BaseException as error:
        _PENDING.reset(token)
        try:
            pending.resolve()
        except ValueError as earlier:
            raise earlier from error
        raise
    _PENDING.reset(token)
    pending.resolve()


def require_true(flag: Tensor, message: str) -> None:
    """Exigir un booleano escalar del cálculo, diferido en CUDA dentro de `deferred_checks`."""
    pending = _PENDING.get()
    if pending is None or flag.device.type != "cuda":
        if not bool(flag):
            if pending is not None:
                pending.resolve()
            raise ValueError(message)
        return
    pending.add(flag, message)


def finite_indicator(value: Tensor) -> Tensor:
    """Indicador de finitud de un tensor en su dispositivo, sin esperar al resultado.

    `isfinite(value).all()` lanza cinco núcleos en coma flotante (igualdad, valor absoluto,
    desigualdad, producto y reducción) y crea tres temporales del tamaño del tensor. Para
    un tensor real no vacío basta su norma infinito, max |x|, que es una sola reducción sin
    temporales. Es finita si y solo si lo son todos los valores, porque el máximo propaga
    NaN e infinito y, si no los hay, es uno de los valores. Se calcula en el tipo del tensor,
    sin redondeo, y sin grafo de autograd. El resto de tensores devuelve el booleano de
    `isfinite(value).all()`.
    """
    if value.is_floating_point() and value.numel():
        return torch.linalg.vector_norm(value.detach(), float("inf"))
    return torch.isfinite(value).all()


def is_finite(indicator: Tensor) -> bool:
    """Leer un indicador de `finite_indicator`, con una sincronización si está en CUDA."""
    if indicator.dtype == torch.bool:
        return bool(indicator)
    return math.isfinite(indicator.item())


def check_finite(value: Tensor, name: str) -> None:
    require_finite(value, f"{name} contiene NaN o infinito")


def require_finite(value: Tensor, message: str) -> None:
    """Exigir valores finitos con su propio mensaje, diferido en CUDA dentro de `deferred_checks`.

    Dentro del tramo, un mismo tensor sin cambios solo se comprueba una vez. Si falla, el
    primer mensaje registrado es también el que daría la comprobación inmediata.
    """
    pending = _PENDING.get()
    if pending is not None and value.device.type == "cuda":
        key = (id(value), value._version)
        if key in pending.seen:
            return
        pending.seen[key] = value
        pending.add(finite_indicator(value), message)
        return
    if not is_finite(finite_indicator(value)):
        if pending is not None:
            pending.resolve()
        raise ValueError(message)


def copy_state(state: NeuralMemoryState, *, differentiable: bool) -> NeuralMemoryState:
    def copy(value: Tensor) -> Tensor:
        return value.clone() if differentiable else value.detach().clone()

    return NeuralMemoryState(
        tuple(copy(w) for w in state.weights),
        tuple(copy(m) for m in state.momentum),
        state.steps.clone(),
        state.config_id,
        tuple(copy(c) for c in state.convolution),
        tuple(copy(level) for level in state.levels),
    )


def map_memory_rows(
    state: NeuralMemoryState, transform: Callable[[Tensor], Tensor]
) -> NeuralMemoryState:
    """Aplicar la misma operación por filas de flujo a todos los tensores del estado."""
    return replace(
        state,
        weights=tuple(transform(value) for value in state.weights),
        momentum=tuple(transform(value) for value in state.momentum),
        steps=transform(state.steps),
        convolution=tuple(transform(value) for value in state.convolution),
        levels=tuple(transform(value) for value in state.levels),
    )


def map_mac_rows(state: MACState, transform: Callable[[Tensor], Tensor]) -> MACState:
    return replace(
        state,
        memory=map_memory_rows(state.memory, transform),
        convolution=tuple(transform(value) for value in state.convolution),
    )


def mac_tensors(state: MACState) -> tuple[Tensor, ...]:
    """Tensores por flujo en orden fijo: pesos, momentum, pasos, ventanas de k, v y q y,
    con PT5, los niveles ocultos al final."""
    memory = state.memory
    return (
        *memory.weights,
        *memory.momentum,
        memory.steps,
        *memory.convolution,
        *state.convolution,
        *memory.levels,
    )


def join_mac_rows(states: Sequence[MACState]) -> MACState:
    """Concatenar los flujos de varios estados del mismo contrato en el orden recibido."""
    first = states[0]
    if any(
        state.config_id != first.config_id
        or state.memory.config_id != first.memory.config_id
        or len(mac_tensors(state)) != len(mac_tensors(first))
        for state in states
    ):
        raise ValueError("Solo se unen estados MAC del mismo contrato")

    def join(get):
        return torch.cat([get(state) for state in states])

    memory = first.memory
    return MACState(
        NeuralMemoryState(
            tuple(join(lambda s, i=i: s.memory.weights[i]) for i in range(len(memory.weights))),
            tuple(join(lambda s, i=i: s.memory.momentum[i]) for i in range(len(memory.momentum))),
            join(lambda s: s.memory.steps),
            memory.config_id,
            tuple(
                join(lambda s, i=i: s.memory.convolution[i]) for i in range(len(memory.convolution))
            ),
            tuple(join(lambda s, i=i: s.memory.levels[i]) for i in range(len(memory.levels))),
        ),
        first.config_id,
        tuple(join(lambda s, i=i: s.convolution[i]) for i in range(len(first.convolution))),
    )


def require_payload(payload: object, fields: set[str]) -> dict:
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("El estado serializado tiene campos incompatibles")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("La versión del estado serializado no está admitida")
    return payload
