"""Recuperación del banco E2 con validación semántica previa al checksum."""

import hashlib

from .spaced_utility import _canonical, _copy, _integer, _number, _require
from .spaced_utility_state import _fields


def checksum(state):
    data = {key: value for key, value in state.items() if key != "memory"}
    return hashlib.sha256(state["memory"] + _canonical(data).encode()).hexdigest()


def _identity(value, expected):
    if isinstance(expected, dict):
        _fields(value, expected, "La identidad del banco E2 no conserva sus campos")
        for key, item in expected.items():
            _identity(value[key], item)
    else:
        _require(
            type(value) is type(expected) and value == expected,
            "El banco E2 pertenece a otro contrato",
        )


def _rng(value):
    _fields(
        value, ("bit_generator", "state", "has_uint32", "uinteger"), "El RNG E2 está incompleto"
    )
    _fields(value["state"], ("state", "inc"), "El estado PCG64 E2 está incompleto")
    _require(
        value["bit_generator"] == "PCG64"
        and _integer(value["has_uint32"], 0, 1)
        and _integer(value["uinteger"], 0, 2**32 - 1)
        and all(_integer(v, 0, 2**128 - 1) for v in value["state"].values())
        and value["state"]["inc"] % 2 == 1,
        "El RNG E2 no conserva sus tipos o dominio",
    )


def _ids(values, maximum):
    _require(
        type(values) is list
        and len(values) <= maximum
        and all(_integer(i, 1) for i in values)
        and values == sorted(set(values)),
        "Los IDs del recibo E2 deben ser positivos, únicos y crecientes",
    )
    return values


def _receipt(run, value):
    metadata = dict(run._memory.snapshot_metadata())
    if value is None:
        _require(
            metadata == dict(seen=0, last_id=0, confirmed_at=0), "Falta el recibo del banco E2"
        )
        return
    _fields(
        value,
        (
            "before_seen",
            "after_seen",
            "before_last_id",
            "confirmed_at",
            "old_ids",
            "new_ids",
            "retained_ids",
            "protected_ids",
            "frontier_ids",
            "ranking",
            "evidence_sha256",
            "ledger_usage",
            "estimated_working_bytes",
        ),
        "El recibo E2 no conserva su esquema",
    )
    for key in (
        "before_seen",
        "after_seen",
        "before_last_id",
        "confirmed_at",
        "estimated_working_bytes",
    ):
        _require(_integer(value[key]), "Los contadores del recibo E2 no son válidos")
    cap = run.config.capacity
    old = _ids(value["old_ids"], cap)
    new = _ids(value["new_ids"], 8192)
    retained = _ids(value["retained_ids"], cap)
    protected = _ids(value["protected_ids"], cap)
    frontier = _ids(value["frontier_ids"], cap)
    _require(
        1 <= len(new) == value["after_seen"] - value["before_seen"]
        and (
            (value["before_seen"] == value["before_last_id"] == 0)
            or 0 < value["before_seen"] <= value["before_last_id"]
        )
        and len(old) == min(value["before_seen"], cap)
        and all(i <= value["before_last_id"] for i in old)
        and new[0] > value["before_last_id"]
        and metadata
        == dict(seen=value["after_seen"], last_id=new[-1], confirmed_at=value["confirmed_at"])
        and retained == [r.id for r in run.records()]
        and len(retained) == min(value["after_seen"], cap)
        and set(retained) <= set(old + new),
        "El recibo E2 no concuerda con los registros y cursores nativos",
    )
    _usage(run, value, old, new)
    _decision(run, value, old, new, retained, protected, frontier)


def _usage(run, value, old, new):
    from .spaced_retention import working_bytes

    cap = run.config.capacity
    usage = value["ledger_usage"]
    _fields(
        usage,
        ("episodes", "pending_sessions", "pending_rows", "references"),
        "Falta el uso del ledger E2",
    )
    limits = dict(
        episodes=cap,
        pending_sessions=run.evidence_config.max_pending_sessions,
        pending_rows=run.evidence_config.max_pending_rows,
        references=run.evidence_config.max_references,
    )
    _require(
        all(_integer(usage[k], 0, limit) for k, limit in limits.items())
        and usage["episodes"] == len(old)
        and bool(usage["pending_sessions"]) == bool(usage["pending_rows"])
        and usage["references"] <= 64 * usage["pending_rows"]
        and working_bytes(len(old) + len(new), usage) == value["estimated_working_bytes"]
        and value["estimated_working_bytes"] <= run.config.max_working_bytes,
        "El recibo E2 no conserva los límites de memoria estimados",
    )
    digest = value["evidence_sha256"]
    _require(
        type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
        "La huella de evidencia E2 no es válida",
    )


def _ranking(run, value, old):
    ranking = value["ranking"]
    _require(
        type(ranking) is list and len(ranking) <= len(old), "El ranking E2 excede su capacidad"
    )
    recipients, sources = set(), set()
    for item in ranking:
        _require(
            type(item) is list and len(item) == 3,
            "Una utilidad E2 necesita receptor, fuente y ganancia",
        )
        target, source, gain = item
        _require(
            _integer(target, 1)
            and _integer(source, 1)
            and target in old
            and source in old
            and target not in recipients
            and source not in sources
            and type(gain) is float
            and _number(gain) > run.evidence_config.positive_gain_margin
            and (run.config.policy == "permuted_utility" or target == source),
            "El ranking E2 no conserva fuentes y utilidades admisibles",
        )
        recipients.add(target)
        sources.add(source)
    _require(
        ranking == sorted(ranking, key=lambda item: (-item[2], item[0])),
        "El ranking E2 no está ordenado",
    )
    return ranking


def _decision(run, value, old, new, retained, protected, frontier):
    cap = run.config.capacity
    ranking = _ranking(run, value, old)
    clients = sorted(old + new)
    overflow = len(clients) > cap
    original = run.config.policy == "reservoir"
    _require(
        frontier == (new[-run.config.frontier :] if overflow and not original else []),
        "La frontera E2 no coincide con las admisiones",
    )
    utility = run.config.policy in ("utility", "permuted_utility") and overflow
    _require(
        (utility or not ranking)
        and protected == sorted(i[0] for i in ranking[: cap - len(frontier)]),
        "La protección E2 no corresponde al ranking",
    )
    if not overflow:
        _require(
            retained == clients and not ranking, "Un banco E2 sin desborde debe conservar todo"
        )
    elif run.config.policy in ("utility", "permuted_utility", "recent"):
        anchors = set(protected + frontier)
        fill = sorted((i for i in clients if i not in anchors), reverse=True)[: cap - len(anchors)]
        _require(
            retained == sorted([*anchors, *fill]), "La selección E2 no sigue su política declarada"
        )
    elif run.config.policy == "uniform_frontier":
        _require(set(frontier) <= set(retained), "La selección uniforme E2 pierde su frontera")


def restore(run, payload):
    _fields(
        payload,
        ("schema_version", "identity", "memory", "rng", "receipt", "sha256"),
        "El estado del banco E2 no conserva sus campos",
    )
    _require(
        type(payload["schema_version"]) is int and payload["schema_version"] == 1,
        "La versión del banco E2 no está admitida",
    )
    _identity(payload["identity"], run.identity())
    _require(
        type(payload["memory"]) is bytes and len(payload["memory"]) <= 2 * 1024**2,
        "El archivo nativo E2 excede su presupuesto",
    )
    _rng(payload["rng"])
    run._memory.restore_bytes(payload["memory"])
    _receipt(run, payload["receipt"])
    state = {k: v for k, v in payload.items() if k != "sha256"}
    _require(checksum(state) == payload["sha256"], "La huella del banco E2 ha cambiado")
    run._rng.bit_generator.state = _copy(payload["rng"])
    _require(
        _canonical(run._rng.bit_generator.state) == _canonical(payload["rng"]),
        "El RNG E2 cambió de tipos",
    )
    run._receipt = _copy(payload["receipt"])
    return run
