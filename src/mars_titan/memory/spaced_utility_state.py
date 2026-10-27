"""Recuperación semántica del registro E2, además de su huella de contenido."""

import hashlib

from mars_titan.models.titans.episodic_snapshot import _digest_id

from .spaced_utility import (
    DAY_US,
    _canonical,
    _gains,
    _integer,
    _number,
    _require,
    _rows,
)


def _fields(value, expected, message):
    _require(
        isinstance(value, dict) and len(value) == len(expected) and set(value) == set(expected),
        message,
    )


def _identity(value, run):
    _fields(value, ("codec_id", "config"), "La identidad E2 no conserva sus campos")
    expected = run.config.identity()
    _fields(value["config"], expected, "La configuración E2 no conserva sus campos")
    _require(
        value["codec_id"] == run.codec_id
        and all(
            type(value["config"][key]) is type(item) and value["config"][key] == item
            for key, item in expected.items()
        ),
        "El estado E2 corresponde a otra configuración",
    )


def _checksum(state, expected):
    _require(
        hashlib.sha256(_canonical(state).encode()).hexdigest() == expected,
        "La huella del estado E2 ha cambiado",
    )


def _pairs(values, maximum, *, at):
    _require(
        isinstance(values, list) and len(values) <= maximum, "Las identidades E2 exceden su límite"
    )
    result = {}
    for pair in values:
        _require(
            isinstance(pair, list)
            and len(pair) == 2
            and _integer(pair[0], 1)
            and _integer(pair[1])
            and pair[1] <= at
            and pair[0] not in result,
            "Las identidades o maduraciones E2 no son válidas",
        )
        result[pair[0]] = pair[1]
    _require(list(result) == sorted(result), "Las identidades E2 deben estar ordenadas")
    return result


def _episodes(values, config, clock, completed):
    _require(
        isinstance(values, list) and len(values) <= config.capacity,
        "La evidencia E2 excede la capacidad",
    )
    result = {}
    for pair in values:
        _require(
            isinstance(pair, list)
            and len(pair) == 2
            and _integer(pair[0], 1)
            and pair[0] not in result,
            "La evidencia E2 repite episodios",
        )
        identifier, score = pair
        _fields(
            score,
            (
                "maturity_at",
                "sessions",
                "mean_gain",
                "positive_sessions",
                "first_positive_day",
                "last_positive_day",
            ),
            "La evidencia E2 tiene otro esquema",
        )
        n, positive = score["sessions"], score["positive_sessions"]
        first, last = score["first_positive_day"], score["last_positive_day"]
        _require(
            _integer(score["maturity_at"])
            and score["maturity_at"] <= clock
            and _integer(n, 0, completed)
            and _integer(positive, 0, n),
            "Los recuentos de evidencia E2 no concuerdan con el reloj",
        )
        mean = _number(score["mean_gain"])
        _require(n != 0 or mean == 0, "Un episodio E2 sin observaciones tiene media cero")
        if positive == 0:
            _require(first is last is None, "Un episodio E2 sin evidencia positiva no tiene fechas")
        else:
            _require(
                _integer(first)
                and _integer(last)
                and score["maturity_at"] // DAY_US <= first <= last <= clock // DAY_US
                and (positive != 1 or first == last),
                "Los días positivos E2 no corresponden a sus sesiones",
            )
        result[identifier] = dict(score, mean_gain=mean)
    _require(list(result) == sorted(result), "La evidencia E2 debe estar ordenada")
    return result


def _status(value, row, *, decision, clock):
    if value is None:
        return None
    _require(isinstance(value, list) and len(value) == 3, "El resultado guardado E2 no es válido")
    kind, available, target = value
    _require(
        kind in ("label", "excluded", "finalized")
        and _integer(available)
        and decision <= available <= clock,
        "El resultado E2 queda fuera de su corte",
    )
    if kind == "label":
        earliest = row["label_not_before"]
        _require(
            earliest is not None and decision < earliest <= available,
            "La etiqueta E2 guardada no ha madurado",
        )
        target = _number(target)
        _gains(row, target)
    else:
        _require(target is None, "Una censura E2 no contiene etiqueta")
    return kind, available, target


def _sessions(values, config, clock, last_issued):
    _require(
        isinstance(values, list) and len(values) <= config.max_pending_sessions,
        "Las sesiones pendientes E2 exceden su límite",
    )
    sessions, total, references, pending, observed, censored = {}, 0, 0, 0, 0, 0
    for value in values:
        _fields(
            value,
            ("market", "decision_at", "snapshot_sha256", "maturity", "rows"),
            "La sesión guardada E2 tiene otro esquema",
        )
        market, at = value["market"], value["decision_at"]
        _require(
            market in ("US", "CN") and _integer(at) and at <= last_issued.get(market, -1),
            "La sesión pendiente E2 no tiene emisión acreditada",
        )
        key = market, at
        _require(key not in sessions, "La sesión pendiente E2 está repetida")
        _digest_id(value["snapshot_sha256"], "El snapshot E2 guardado")
        eligible = _pairs(value["maturity"], config.capacity, at=at)
        raw = value["rows"]
        _require(
            isinstance(raw, list) and all(isinstance(r, dict) and "status" in r for r in raw),
            "Las filas E2 guardadas necesitan su estado",
        )
        normalized_market, rows = _rows(
            [{k: v for k, v in r.items() if k != "status"} for r in raw], at, eligible
        )
        _require(normalized_market == market, "La sesión E2 mezcla mercados")
        for row in raw:
            current = rows[row["flow_id"]]
            current["status"] = _status(row["status"], current, decision=at, clock=clock)
            status = current["status"]
            pending += status is None
            observed += status is not None and status[0] == "label"
            censored += status is not None and status[0] != "label"
        _require(
            any(r["status"] is None for r in rows.values()),
            "Una sesión E2 ya completa debió publicarse",
        )
        used = {i for r in rows.values() for i in r["deleted_points"]}
        _require(used == set(eligible), "Las referencias E2 no corresponden a sus episodios usados")
        total += len(rows)
        references += sum(len(r["deleted_points"]) for r in rows.values())
        _require(
            total <= config.max_pending_rows and references <= config.max_references,
            "El estado pendiente E2 excede sus presupuestos",
        )
        sessions[key] = dict(snapshot_sha256=value["snapshot_sha256"], maturity=eligible, rows=rows)
    return sessions, pending, observed, censored


def restore(run, payload):
    _fields(
        payload,
        (
            "schema_version",
            "identity",
            "clock",
            "issued",
            "resolved",
            "censored",
            "completed_sessions",
            "last_issued",
            "episodes",
            "sessions",
            "sha256",
        ),
        "El estado E2 no conserva su esquema",
    )
    state = {k: v for k, v in payload.items() if k != "sha256"}
    # Las estructuras y sus límites se validan antes de serializarlas para la huella.
    # Así, un valor profundo en un campo escalar no fuerza una recursión sin límite.
    _identity(state["identity"], run)
    _require(
        type(state["schema_version"]) is int and state["schema_version"] == 1,
        "El estado E2 corresponde a otra configuración o versión",
    )
    for name in ("issued", "resolved", "censored", "completed_sessions"):
        _require(_integer(state[name]), "Los contadores E2 no son válidos")
    clock, last = state["clock"], state["last_issued"]
    if clock is None:
        _require(
            all(
                state[name] == 0
                for name in ("issued", "resolved", "censored", "completed_sessions")
            )
            and last == {}
            and state["episodes"] == state["sessions"] == [],
            "El estado inicial E2 contiene actividad",
        )
        _checksum(state, payload["sha256"])
        return run
    _require(
        _integer(clock)
        and isinstance(last, dict)
        and all(k in ("US", "CN") and _integer(v) and v <= clock for k, v in last.items())
        and (bool(last) == (state["issued"] > 0)),
        "El cursor de emisiones E2 no es válido",
    )
    episodes = _episodes(state["episodes"], run.config, clock, state["completed_sessions"])
    sessions, pending, observed, censored = _sessions(state["sessions"], run.config, clock, last)
    _require(
        state["issued"] == state["resolved"] + state["censored"] + pending
        and observed <= state["resolved"]
        and censored <= state["censored"]
        and state["completed_sessions"] <= state["resolved"] + state["censored"],
        "Los contadores E2 no concuerdan con sus pendientes",
    )
    _checksum(state, payload["sha256"])
    run.clock, run._last_issued, run._episodes, run._sessions = (
        clock,
        dict(last),
        episodes,
        sessions,
    )
    run.issued, run.resolved, run.censored = (state[k] for k in ("issued", "resolved", "censored"))
    run.completed_sessions = state["completed_sessions"]
    return run
