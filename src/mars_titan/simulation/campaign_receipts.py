"""Leer recibos adaptativos desde un componente operativo independiente del runtime."""

import hashlib
import json
import re
from datetime import datetime
from itertools import pairwise
from pathlib import Path

MAX_BYTES = 32 * 1024**2
MAX_CASES = 75
STATUSES = frozenset(
    {"pending", "queued", "running", "waiting", "paused", "blocked", "failed", "completed"}
)
STAGES = frozenset({"pilot", "main", "auxiliary", "audit"})
AUXILIARY = ("ppo_recent_aux", "ppo_replay_aux")
ROW_FIELDS = frozenset(
    {
        "path",
        "stage",
        "variant",
        "seed",
        "status",
        "config_sha256",
        "planned_transitions",
        "transitions",
    }
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "El recibo repite un campo JSON")
        result[key] = value
    return result


def _read(path, maximum=MAX_BYTES, *, with_hash=False):
    _require(
        not any(part.is_symlink() for part in (path, *path.parents)) and path.is_file(),
        "Falta un archivo regular del recibo",
    )
    with path.open("rb") as source:
        payload = source.read(maximum + 1)
    _require(0 < len(payload) <= maximum, "El recibo está vacío o supera su presupuesto")
    try:
        result = json.loads(payload, object_pairs_hook=_unique)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("El recibo no contiene JSON legible") from error
    _require(isinstance(result, dict), "El recibo debe ser un objeto JSON")
    return (result, hashlib.sha256(payload).hexdigest()) if with_hash else result


def _digest(value):
    try:
        encoded = json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
    except (ValueError, RecursionError) as error:
        raise ValueError("El recibo no conserva un contenido JSON finito") from error
    return hashlib.sha256(encoded.encode()).hexdigest()


def _hash(value):
    _require(
        isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None,
        "Falta una huella SHA256 válida",
    )
    return value


def _integer(value, *, maximum=2**63 - 1):
    _require(type(value) is int and 0 <= value <= maximum, "El recuento no es un entero acotado")
    return value


def _status(value):
    _require(
        isinstance(value, str) and value in STATUSES, "El recibo contiene un estado desconocido"
    )
    return value


def _schema(value):
    _require(
        type(value.get("schema_version")) is int and value["schema_version"] == 1,
        "La versión del recibo no es compatible",
    )


def _timestamp(value):
    _require(isinstance(value, str) and len(value) <= 64, "Falta la fecha de publicación")
    result = datetime.fromisoformat(value)
    _require(result.tzinfo is not None, "La publicación debe declarar su zona horaria")
    return result


def _journal(path):
    envelope = _read(path)
    _require(
        set(envelope) == {"payload", "sha256"} and isinstance(envelope["payload"], dict),
        "El diario no contiene un sobre coherente",
    )
    _require(
        _hash(envelope["sha256"]) == _digest(envelope["payload"]),
        "La huella del diario no coincide con su contenido",
    )
    return envelope["payload"], envelope["sha256"]


def _identity(value):
    _schema(value)
    settings, base = value.get("settings"), value.get("base_configuration")
    _require(
        value.get("kind") == "adaptive_campaign"
        and isinstance(settings, dict)
        and isinstance(base, dict),
        "La identidad no pertenece a una campaña adaptativa",
    )
    _require(
        all(item.get("final_test_opened") is False for item in (value, settings, base)),
        "La identidad no acredita la reserva cerrada",
    )
    _require(
        type(settings.get("schema_version")) is int and settings["schema_version"] in (1, 2),
        "La versión de la configuración no está reconocida",
    )
    variants, seeds = settings.get("variants"), settings.get("seeds")
    _require(
        isinstance(variants, list)
        and 1 <= len(variants) <= MAX_CASES
        and all(isinstance(v, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", v) for v in variants)
        and len(set(variants)) == len(variants)
        and not set(variants).intersection(AUXILIARY)
        and isinstance(seeds, list)
        and 1 <= len(seeds) <= MAX_CASES
        and all(type(seed) is int and 0 <= seed <= 2**63 - 1 for seed in seeds)
        and len(set(seeds)) == len(seeds),
        "La identidad no declara variantes y semillas únicas",
    )
    return variants, seeds


def _projection(case):
    stage, variant, seed = case.get("stage"), case.get("variant"), case.get("seed")
    _require(
        isinstance(stage, str)
        and stage in STAGES
        and isinstance(variant, str)
        and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", variant) is not None
        and type(seed) is int
        and 0 <= seed <= 2**63 - 1,
        "El caso no conserva etapa, variante y semilla",
    )
    _require(
        case.get("id") == f"{stage}-{variant}-{seed}"
        and case.get("output") == f"{stage}/{variant}-{seed}",
        "El caso contiene una identidad o ruta ajena",
    )
    status = _status(case.get("status"))
    planned = _integer(case.get("transitions"))
    completed = _integer(case.get("confirmed_transitions", 0))
    _require(
        (
            planned == completed == 0
            if stage == "audit"
            else 0 <= completed <= planned and planned > 0
        ),
        "Las transiciones no concilian el presupuesto del caso",
    )
    if status == "completed":
        _hash(case.get("receipt_sha256"))
    return dict(
        path=case["output"],
        stage=stage,
        variant=variant,
        seed=seed,
        status=status,
        config_sha256=_hash(case.get("config_sha256")),
        planned_transitions=planned,
        transitions=completed,
    )


def _training_completion(case, identity, root):
    if case["status"] != "completed" or case["stage"] == "audit":
        return
    confirmed, planned = case.get("confirmed_transitions", 0), case["transitions"]
    schema = identity["settings"]["schema_version"]
    reason = case.get("stopping_reason")
    if confirmed == planned:
        _require(
            reason == "budget_exhausted" if schema == 2 else reason in (None, "budget_exhausted"),
            "El caso completo no acredita el agotamiento de su presupuesto",
        )
        return
    _require(
        schema == 2 and case["stage"] in {"main", "auxiliary"} and reason == "early_stop",
        "El caso completo no acredita sus transiciones o una parada permitida",
    )
    native, digest = _read(root / case["output"] / "run.json", 8 * 1024**2, with_hash=True)
    _require(digest == case["receipt_sha256"], "El recibo de parada temprana no conserva su huella")
    selection = identity["base_configuration"].get("selection")
    _require(
        isinstance(selection, dict) and selection.get("early_stopping") is True,
        "La configuración no admite parada temprana",
    )
    minimum = _integer(selection.get("min_transitions"))
    patience = _integer(selection.get("patience"))
    _require(
        minimum < confirmed < planned
        and patience > 0
        and _integer(case.get("optimizer_steps")) > 0,
        "La parada no acredita transiciones mínimas y actualizaciones",
    )
    _require(
        type(native.get("schema_version")) is int
        and native["schema_version"] == 3
        and native.get("kind") == "native_ppo"
        and native.get("status") == "completed"
        and type(native.get("seed")) is int
        and native["seed"] == case["seed"]
        and native.get("agent_variant") == case["variant"]
        and native.get("device") == "cuda:0"
        and native.get("diagnostic") is False
        and native.get("parent_frozen") is True
        and native.get("final_test_opened") is False
        and native.get("identity_sha256") == _hash(case.get("training_identity_sha256"))
        and _integer(native.get("transitions")) == confirmed
        and _integer(native.get("total_steps")) == planned
        and _integer(native.get("optimizer_steps")) == case["optimizer_steps"]
        and native.get("stopping_reason") == "early_stop"
        and native.get("selection") == dict(selection, policy="greedy_argmax"),
        "El recibo no acredita identidad, reserva y contadores de la parada",
    )
    history = native.get("evaluation_cursors")
    _require(
        isinstance(history, list) and 2 <= len(history) <= 4096,
        "Faltan las evaluaciones completas de la parada",
    )
    cursors = [_integer(cursor, maximum=confirmed) for cursor in history]
    interval = _integer(identity["settings"].get("evaluation_transitions"))
    environments = _integer(identity["base_configuration"].get("environments"))
    _require(
        interval > 0
        and environments > 0
        and cursors[0] == 0
        and cursors[-1] == confirmed
        and all(
            interval <= current - previous < interval + environments
            for previous, current in pairwise(cursors)
        )
        and _integer(native.get("evaluations")) == len(cursors),
        "Los cursores no conservan la programación de validaciones",
    )
    best = native.get("best")
    _require(isinstance(best, dict), "Falta el cursor seleccionado")
    best_cursor = _integer(best.get("transitions"), maximum=confirmed)
    stale = _integer(native.get("stale_evaluations"))
    _require(
        best_cursor in cursors
        and stale >= patience
        and stale == sum(cursor > max(minimum, best_cursor) for cursor in cursors),
        "La parada no acredita su paciencia con evaluaciones completas",
    )


def _state(value, identity, root):
    _schema(value)
    variants, seeds = _identity(identity)
    _require(
        value.get("identity_sha256") == _digest(identity), "El diario pertenece a otra identidad"
    )
    status = _status(value.get("status"))
    phase = value.get("phase")
    _require(
        isinstance(phase, str) and phase in STAGES | {"pilot_complete", "completed"},
        "La fase de la campaña es desconocida",
    )
    _timestamp(value.get("updated_at"))
    cases = value.get("cases")
    _require(
        isinstance(cases, list)
        and 1 <= len(cases) <= MAX_CASES
        and all(isinstance(case, dict) for case in cases),
        "El diario excede su límite de casos",
    )
    rows = [_projection(case) for case in cases]
    _require(len({row["path"] for row in rows}) == len(rows), "El diario repite un caso")
    active = value.get("active_case")
    _require(
        active is None
        or any(case["id"] == active and case["status"] != "completed" for case in cases),
        "El caso activo no corresponde a trabajo pendiente",
    )
    _require(
        type(value.get("audit_opened")) is bool and type(value.get("budget_complete")) is bool,
        "Falta el estado de auditoría o del presupuesto",
    )
    if value.get("freeze_sha256") is not None:
        _hash(value["freeze_sha256"])
    gate = value.get("gate")
    _require(
        gate is None or isinstance(gate, dict) and type(gate.get("enabled")) is bool,
        "La decisión auxiliar no tiene un estado válido",
    )
    enabled = gate is not None and gate["enabled"]
    present = {row["stage"] for row in rows}
    _require(
        "pilot" in present and ("auxiliary" not in present or enabled),
        "El plan no conserva las etapas declaradas",
    )
    if "main" in present:
        choice = value.get("choice")
        _require(
            isinstance(choice, dict) and _integer(choice.get("transitions")) > 0,
            "Falta el presupuesto principal elegido",
        )
    if "audit" in present:
        _require(
            "main" in present and value["audit_opened"] and value.get("freeze_sha256") is not None,
            "La auditoría no conserva la selección congelada",
        )
    for stage in present:
        allowed = (
            AUXILIARY
            if stage == "auxiliary"
            else (*variants, *AUXILIARY)
            if stage == "audit" and enabled
            else variants
        )
        expected = {(variant, seed) for variant in allowed for seed in seeds}
        actual = {(row["variant"], row["seed"]) for row in rows if row["stage"] == stage}
        _require(
            actual == expected if status == "completed" else actual <= expected,
            "La etapa no conserva sus casos declarados",
        )
    for case in cases:
        _training_completion(case, identity, root)
    if status == "completed":
        required = {"pilot", "main", "audit"} | ({"auxiliary"} if enabled else set())
        _require(
            phase == "completed"
            and present == required
            and gate is not None
            and value["budget_complete"]
            and active is None
            and all(row["status"] == "completed" for row in rows),
            "La campaña completa conserva etapas o casos pendientes",
        )
    return rows


def _run(value, state, identity):
    _schema(value)
    _require(
        value.get("kind") == "adaptive_campaign"
        and value.get("identity_sha256") == _digest(identity),
        "El informe pertenece a otra identidad",
    )
    _require(value.get("final_test_opened") is False, "El informe no acredita la reserva cerrada")
    _require(
        value.get("parent_frozen") is True
        and value.get("domain") == "synthetic"
        and value.get("analysis_domain") == "technical",
        "El informe no conserva padre y dominio",
    )
    _status(value.get("status"))
    phase = value.get("phase")
    _require(
        isinstance(phase, str) and phase in STAGES | {"pilot_complete", "completed"},
        "La fase del informe es desconocida",
    )
    count = _integer(value.get("completed_cases"), maximum=MAX_CASES)
    for field in ("budget_complete", "audit_opened", "selection_frozen"):
        _require(type(value.get(field)) is bool, "Falta una declaración del informe")
    published, committed = _timestamp(value.get("updated_at")), _timestamp(state["updated_at"])
    # La igualdad vincula publicaciones. El reloj de pared puede retroceder.
    expected_count = sum(case["status"] == "completed" for case in state["cases"])
    _require(count <= expected_count, "El informe atribuye casos todavía no confirmados")
    if published == committed:
        _require(
            all(
                value[field] == state[field]
                for field in ("status", "phase", "budget_complete", "audit_opened")
            )
            and count == expected_count
            and value["selection_frozen"] == (state.get("freeze_sha256") is not None),
            "El informe no concilia su generación con el diario",
        )
        return True
    return False


def _registry(value, state, rows):
    _schema(value)
    _require(value.get("kind") == "adaptive_campaign", "El registro no es adaptativo")
    _status(value.get("status"))
    registered = value.get("runs")
    planned = _integer(value.get("planned_runs"), maximum=MAX_CASES)
    _require(
        isinstance(registered, list) and 1 <= len(registered) == planned <= len(rows),
        "El plan del registro no concilia sus filas",
    )
    current = {row["path"]: row for row in rows}
    seen = set()
    for row in registered:
        _require(isinstance(row, dict) and set(row) == ROW_FIELDS, "La fila no conserva su esquema")
        path = row["path"]
        _require(
            isinstance(path, str) and path in current and path not in seen,
            "El registro contiene rutas ajenas o filas duplicadas",
        )
        seen.add(path)
        expected = current[path]
        _status(row["status"])
        _require(
            type(row["seed"]) is int
            and _integer(row["planned_transitions"]) == expected["planned_transitions"]
            and all(
                row[field] == expected[field] for field in ROW_FIELDS - {"status", "transitions"}
            )
            and _integer(row["transitions"]) <= expected["transitions"]
            and (row["status"] != "completed" or row == expected),
            "El registro altera un caso o atribuye progreso futuro",
        )
    return value["status"] == state["status"] and registered == rows


def read_adaptive_receipt(path, expected=None):
    """Validar una generación publicada. BlockingIOError indica publicación aún incompleta."""
    path = Path(path)
    _require(path.name == "registry.json", "El lector adaptativo necesita registry.json")
    for _ in range(3):
        state, snapshot_hash = _journal(path.with_name("campaign.json"))
        identity = _read(path.with_name("identity.json"))
        report, registry = _read(path.with_name("run.json")), _read(path)
        _, after = _journal(path.with_name("campaign.json"))
        if snapshot_hash == after:
            break
    else:
        raise BlockingIOError("El diario cambió durante las lecturas acotadas")
    rows = _state(state, identity, path.parent)
    complete_report = _run(report, state, identity)
    complete_registry = _registry(registry, state, rows)
    if not complete_report or not complete_registry:
        raise BlockingIOError("La publicación del diario todavía no concilia informe y registro")
    if expected is not None:
        _require(
            _integer(expected, maximum=MAX_CASES) == len(rows),
            "El plan no coincide con el esperado",
        )
    return dict(
        kind="adaptive_campaign_receipt",
        status=state["status"],
        phase=state["phase"],
        completed_runs=sum(row["status"] == "completed" for row in rows),
        planned_runs=len(rows),
        identity_sha256=state["identity_sha256"],
        snapshot_sha256=snapshot_hash,
        final_test_opened=report["final_test_opened"],
    )
