"""Persistencia de S2 con contextos únicos e instantánea recuperable por corte."""

import hashlib

from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.episodic_snapshot import _digest_id

from .failure_context import FailureContext
from .failure_ledger import (
    FailureLedger,
    _clock,
    _decision,
    _Emission,
    _Failure,
    _key,
    _number,
    _order,
)

_FIELDS = {
    "schema_version",
    "layout_id",
    "configuration_id",
    "clock",
    "counts",
    "last",
    "emissions",
    "failures",
    "pending",
    "retained",
    "snapshot",
    "state_sha256",
}


def _digest(payload):
    description = {name: value for name, value in payload.items() if name != "state_sha256"}
    description["emissions"] = [
        dict(context_id=row["context"]["context_id"], prediction=row["prediction"])
        for row in payload["emissions"]
    ]
    return hashlib.sha256(canonical(description).encode()).hexdigest()


def export(book):
    failures = {r.emission.context.context_id: r for r in (*book._retained, *book._snapshot)}
    emissions = {e.context.context_id: e for e in book._pending.values()}
    emissions.update({key: r.emission for key, r in failures.items()})
    payload = dict(
        schema_version=1,
        layout_id=book.layout.fingerprint(),
        configuration_id=book.configuration.fingerprint(),
        clock=book.clock,
        counts=book.counts(),
        last=[list(row) for row in sorted(book._last.items())],
        emissions=[
            dict(context=emissions[key].context.export(), prediction=emissions[key].prediction)
            for key in sorted(emissions)
        ],
        failures=[
            dict(
                context_id=key,
                target=failures[key].target,
                available_at=failures[key].available_at,
                received_at=failures[key].received_at,
            )
            for key in sorted(failures)
        ],
        pending=sorted(e.context.context_id for e in book._pending.values()),
        retained=[r.emission.context.context_id for r in book._retained],
        snapshot=[r.emission.context.context_id for r in book._snapshot],
    )
    return dict(payload, state_sha256=_digest(payload))


def _list(value, maximum, name):
    if type(value) is not list or len(value) > maximum:
        raise ValueError(f"La lista S2 de {name} no conserva su límite")
    return value


def _unique_identifiers(value, maximum, name):
    value = _list(value, maximum, name)
    for key in value:
        _digest_id(key, name)
    if len(set(value)) != len(value):
        raise ValueError(f"La lista S2 de {name} repite una identidad")
    return value


def _restore_last(book, payload):
    rows = _list(payload, book.configuration.max_flows, "flujos")
    keys = []
    for row in rows:
        if type(row) is not list or len(row) != 2:
            raise ValueError("El cursor S2 no conserva su flujo e instante")
        key = _decision(*row)
        if book.clock is None or key[1] > book.clock:
            raise ValueError("El cursor S2 procede del futuro")
        keys.append(key)
    if keys != sorted(keys) or len(dict(keys)) != len(keys):
        raise ValueError("Los cursores S2 no son únicos y canónicos")
    book._last = dict(keys)


def _restore_emissions(book, payload):
    maximum = book.configuration.max_pending + 2 * book.configuration.capacity
    rows = _list(payload, maximum, "emisiones")
    emissions = {}
    decisions = set()
    fit = max(book.configuration.threshold_fitted_until, book.configuration.source_fitted_until)
    for row in rows:
        if type(row) is not dict or set(row) != {"context", "prediction"}:
            raise ValueError("La emisión S2 recuperada no conserva su esquema")
        context = FailureContext.restore(book.layout, row["context"])
        if (
            book.clock is None
            or not fit < context.prediction_at <= book.clock
            or book._last.get(context.flow_id, -1) < context.prediction_at
            or context.context_id in emissions
            or _key(context) in decisions
        ):
            raise ValueError("La emisión S2 no concuerda con su origen, cursor o corte")
        emissions[context.context_id] = _Emission(
            context, _number(row["prediction"]), book.configuration.cache_keys
        )
        decisions.add(_key(context))
    if list(emissions) != sorted(emissions):
        raise ValueError("Las emisiones S2 no conservan su orden canónico")
    return emissions


def _restore_failures(book, payload, emissions):
    rows = _list(payload, 2 * book.configuration.capacity, "fallos")
    failures = {}
    for row in rows:
        if type(row) is not dict or set(row) != {
            "context_id",
            "target",
            "available_at",
            "received_at",
        }:
            raise ValueError("El fallo S2 recuperado no conserva su esquema")
        key = row["context_id"]
        _digest_id(key, "fallo")
        if key not in emissions or key in failures:
            raise ValueError("El fallo S2 repite o no encuentra su emisión")
        emission = emissions[key]
        target = _number(row["target"])
        available, received = _clock(row["available_at"]), _clock(row["received_at"], terminal=True)
        if not emission.context.prediction_at < available <= received <= book.clock:
            raise ValueError("El fallo S2 recuperado no había madurado en su recepción")
        error = _number(target - emission.prediction)
        if abs(error) <= book.configuration.threshold:
            raise ValueError("El fallo S2 no supera el umbral declarado")
        failures[key] = _Failure(emission, target, available, received)
    if list(failures) != sorted(failures):
        raise ValueError("Los fallos S2 no conservan su orden canónico")
    return failures


def _restore_members(book, payload, emissions, failures):
    config = book.configuration
    pending = _unique_identifiers(payload["pending"], config.max_pending, "pendientes")
    retained = _unique_identifiers(payload["retained"], config.capacity, "retenidos")
    snapshot = _unique_identifiers(payload["snapshot"], config.capacity, "instantánea")
    if (
        pending != sorted(pending)
        or set(pending) & set(failures)
        or set(pending) | set(failures) != set(emissions)
        or set(retained) | set(snapshot) != set(failures)
    ):
        raise ValueError("Las referencias S2 no concilian con las emisiones y fallos únicos")
    book._pending = {_key(emissions[key].context): emissions[key] for key in pending}
    book._retained = tuple(failures[key] for key in retained)
    book._snapshot = tuple(failures[key] for key in snapshot)
    ordered = sorted(failures.values(), key=_order)
    expected = tuple(ordered[-config.capacity :])
    previous = tuple(r for r in ordered if r.received_at < book.clock)[-config.capacity :]
    if book._retained != expected or book._snapshot != previous:
        raise ValueError("La retención o la publicación de la instantánea S2 no son coherentes")


def _restore_counts(book, counts):
    distinct_failures = {r.emission.context.context_id for r in (*book._retained, *book._snapshot)}
    if (
        type(counts) is not dict
        or set(counts) != set(book.counts())
        or any(type(v) is not int or not 0 <= v < 2**63 for v in counts.values())
        or (book.clock is None and any(counts.values()))
        or counts["pending"] != len(book._pending)
        or counts["issued"] != counts["resolved"] + counts["censored"] + counts["pending"]
        or not len(distinct_failures) <= counts["admitted"] <= counts["resolved"]
        or counts["issued"] < len(book._last)
    ):
        raise ValueError("Los contadores S2 no concilian con las emisiones")
    for name in ("issued", "resolved", "censored", "admitted"):
        setattr(book, f"_{name}", counts[name])


def restore(layout, configuration, payload):
    book = FailureLedger(layout, configuration)
    if (
        type(payload) is not dict
        or set(payload) != _FIELDS
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["layout_id"] != layout.fingerprint()
        or payload["configuration_id"] != configuration.fingerprint()
    ):
        raise ValueError("El libro S2 recuperado no conserva su contrato")
    if payload["clock"] is not None:
        book.begin(payload["clock"], settlement=True)
    _restore_last(book, payload["last"])
    emissions = _restore_emissions(book, payload["emissions"])
    failures = _restore_failures(book, payload["failures"], emissions)
    _restore_members(book, payload, emissions, failures)
    _restore_counts(book, payload["counts"])
    book._budget(book._pending, book._retained, book._last)
    if book.export() != payload:
        raise ValueError("La huella del estado S2 no conserva el contenido publicado")
    return book
