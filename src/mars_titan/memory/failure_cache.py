"""Reutilizar un libro decodificado durante el evento, sin compartir datos mutables."""

import sys
from copy import copy

from .failure_context import FailureContext
from .failure_ledger import FailureLedger, _Emission, _Failure

_SLOT = "failure_decoded_cache"


def fork(book):
    """Copiar contenedores y registros, compartiendo solo sus bytes y escalares inmutables."""
    if type(book) is not FailureLedger:
        raise ValueError("La copia S2 necesita un libro identificado")
    emissions, failures = {}, {}

    def emission(value):
        if id(value) not in emissions:
            cloned = copy(value)
            object.__setattr__(cloned, "context", copy(value.context))
            emissions[id(value)] = cloned
        return emissions[id(value)]

    def failure(value):
        if id(value) not in failures:
            cloned = copy(value)
            object.__setattr__(cloned, "emission", emission(value.emission))
            failures[id(value)] = cloned
        return failures[id(value)]

    result = copy(book)
    result._pending = {key: emission(value) for key, value in book._pending.items()}
    result._last = dict(book._last)
    result._retained = tuple(failure(value) for value in book._retained)
    result._snapshot = tuple(failure(value) for value in book._snapshot)
    return result


def retained_bytes(book, maximum, *, reference_key=None):
    """Contabilidad de objetos alcanzables, con salida temprana al exceder el presupuesto."""
    total = sys.getsizeof(book) + sys.getsizeof(vars(book)) + 512
    stack = [value for key, value in vars(book).items() if key not in {"layout", "configuration"}]
    if reference_key is not None:
        stack.append(reference_key)
    seen = set()
    while stack and total <= maximum:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        total += sys.getsizeof(value)
        if type(value) is dict:
            stack.extend(value.keys())
            stack.extend(value.values())
        elif type(value) is tuple:
            stack.extend(value)
        elif type(value) in (FailureContext, _Emission, _Failure):
            stack.append(vars(value))
    return total


def _key(session, reference):
    if (
        type(reference) is not dict
        or any(type(key) is not str for key in reference)
        or any(type(value) not in (int, str) for value in reference.values())
    ):
        return None
    return (
        session.contract_id,
        session._failure_producer.layout.fingerprint(),
        session._failure_config.ledger.fingerprint(),
        session._failure_config.decoded_cache_bytes,
        tuple(sorted(reference.items())),
    )


def lookup(session, reference):
    """Comprobar el archivo en cada acierto y devolver una copia independiente."""
    event = getattr(session, "_inflight", None)
    if type(event) is not dict or session._failure_config.decoded_cache_bytes == 0:
        return None
    cached = event.get(_SLOT)
    if cached is None:
        return None
    key = _key(session, reference)
    if key is None or key != cached[0]:
        return None
    session.artifacts._bytes(reference, identity=session.contract_id, kind="failure_memory")
    return fork(cached[1])


def remember(session, reference, book):
    """Guardar una sola entrada validada dentro de la vida del evento."""
    event = getattr(session, "_inflight", None)
    if type(event) is not dict or session._failure_config.decoded_cache_bytes == 0:
        return
    event.pop(_SLOT, None)
    maximum = session._failure_config.decoded_cache_bytes
    if retained_bytes(book, maximum) > maximum:
        return
    key = _key(session, reference)
    cloned = fork(book)
    if retained_bytes(cloned, maximum, reference_key=key) <= maximum:
        event[_SLOT] = key, cloned
