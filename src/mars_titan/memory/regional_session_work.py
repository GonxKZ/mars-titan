"""Admitir el recorrido regional completo sin confundir reserva y visitas observadas."""

import hashlib
import json

from mars_titan.models.titans.regional_execution import regional_work_per_row

from .regional_work import MAX_INTEGER, RESOURCES, RegionalWorkLedger


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def configuration(consumer, limits, max_attempts):
    predictor = getattr(consumer, "predictor", None)
    program = getattr(predictor, "regional_execution", None)
    if program is None:
        if limits is not None:
            raise ValueError("Las cuotas regionales requieren un programa regional activo")
        return None
    if (
        type(limits) is not dict
        or set(limits) != RESOURCES
        or any(type(value) is not int or not 0 <= value <= MAX_INTEGER for value in limits.values())
        or type(max_attempts) is not int
        or not 1 <= max_attempts <= 32
    ):
        raise ValueError("El programa regional necesita nueve cuotas y un límite de intentos")
    return dict(
        schema_version=1,
        kind="financial_regional_work_v1",
        program_id=_digest(program.get_extra_state()),
        limits=dict(sorted(limits.items())),
        max_attempts=max_attempts,
        admission="whole_cohort_worst_case_before_predictions_without_refund",
        observed_work="separate_execution_trace_unknown_after_abandonment",
    )


def ledger(session):
    config = session._regional_config
    return RegionalWorkLedger(
        session.output,
        artifacts=session.artifacts,
        contract_id=session.contract_id,
        model_id=session.model_id,
        program_id=config["program_id"],
        limits=config["limits"],
        max_attempts=config["max_attempts"],
    )


def latest(session, snapshot=None):
    snapshot = session.snapshot() if snapshot is None else snapshot
    head = json.loads(session.native.read_blob(str(session.output / "latest.json"), 65536))
    if head["generation"] != snapshot["cursor"]:
        raise ValueError("La reserva regional no corresponde a la cabeza nativa")
    bundle = session._bundle(snapshot["state"])
    return dict(
        generation=head["generation"],
        cursor=snapshot["cursor"],
        checkpoint_sha256=head["checkpoint_sha256"],
        reservation_id=bundle["regional_work"]["reservation_id"] if bundle is not None else None,
    )


def bounds(consumer, rows, *, collection, kind):
    units = dict.fromkeys(sorted(RESOURCES), 0)
    if kind == "settlement":
        return units
    model, count = consumer.predictor, len(rows)
    has_context = model.session_context is not None
    collecting = collection and kind != "warmup"
    core = count * (2 + int(has_context) if collecting else 1)
    per_row = regional_work_per_row(model.config.regional_execution, model.regional_fusion.config)
    units.update(
        core_observations=core,
        context_queries=core if has_context else 0,
        selector_decisions=core,
        local_node_evaluations=per_row["local_node_evaluations"] * core,
        summary_evaluations=per_row["summary_evaluations"] * core,
        mac_observations=core if model.mac is not None else 0,
        context_commits=len({row.flow_id.split("/", 1)[0] for row in rows}) if has_context else 0,
    )
    reader = consumer.readout
    if reader is not None and kind != "warmup":
        if collecting:
            # La colección admite hasta 34/18 y la emisión hasta cuatro visitas adicionales.
            refinements, searches = 38, 22
        elif hasattr(consumer, "catalog"):
            budget = consumer.catalog.budget
            refinements, searches = budget.max_reader_steps, budget.max_reader_searches
        else:
            refinements = reader.config.refinements
            searches = 1 if reader.config.episode_selection == "first_read" else refinements
        units.update(
            reader_refinements=count * refinements,
            reader_searches=count * searches if reader.config.mode != "no_bank" else 0,
        )
    return units


def reserve(session, head, rows, *, kind, cutoff):
    receipt = session._regional_work.reserve(
        cutoff,
        {name: head[name] for name in ("generation", "cursor", "checkpoint_sha256")},
        _digest([row.metadata() for row in rows]),
        bounds(
            session.consumer, rows, collection=session._collection_config is not None, kind=kind
        ),
        event_kind=kind,
    )
    return session._regional_work.archive_payload(receipt["reservation_id"])


def check_link(link):
    if (
        type(link) is not dict
        or set(link) != {"reservation_id", "reference", "cohort", "receipts"}
        or type(link["reservation_id"]) is not str
        or len(link["reservation_id"]) != 64
        or any(character not in "0123456789abcdef" for character in link["reservation_id"])
        or type(link["reference"]) is not dict
        or type(link["cohort"]) is not dict
        or type(link["receipts"]) is not list
    ):
        raise ValueError("El enlace regional no conserva su reserva y referencia")


def verify_bundle(session, bundle, control, rows, *, event_kind=None):
    link = bundle["regional_work"]
    check_link(link)
    if control.get("regional_work") != link:
        raise ValueError("El control y el bundle no enlazan la misma reserva regional")
    inflight = session._inflight
    if (
        inflight is not None
        and "regional_work" in inflight
        and bundle["generation"] == inflight["regional_head"]["generation"] + 1
        and link != inflight["regional_work"]
    ):
        raise ValueError("La propuesta no conserva la reserva del evento en curso")
    session._regional_work.verify_archive_payload(
        link,
        generation=bundle["generation"],
        cutoff=bundle["cutoff"],
        rows_digest=_digest(rows),
        event_kind=event_kind,
    )


def compact(session):
    """Usar el bloque histórico ya confirmado como raíz del resumen regional."""
    snapshot = session.snapshot()
    bundle = session._bundle(snapshot["state"])
    if bundle is None:
        return
    archive = bundle["mechanism_history"]["chunks"][-1]["reference"]
    session._regional_work.compact(
        latest(session, snapshot),
        archive=dict(
            reference=archive,
            generation=bundle["generation"],
            reservation_id=bundle["regional_work"]["reservation_id"],
        ),
    )
