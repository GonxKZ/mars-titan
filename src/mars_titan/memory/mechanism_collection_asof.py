"""Reconstruir resultados desde un prefijo nativo explícito, sin recibos posteriores."""

from mars_titan.calibration.cohort_feedback import CohortOutcome
from mars_titan.models.titans.config import bounded_integer
from mars_titan.models.titans.financial_inputs import FINAL_TEST_US
from mars_titan.models.titans.mechanism_selection import digest

from .mechanism_collection_record import CollectedMechanisms
from .mechanism_supervision import MechanismSupervision
from .native_mechanism_collection import PACKET, _key, _require


class _Collection:
    def __init__(self, entry, packet):
        self.entry, self.packet = entry, packet
        self.rows = packet.payload()["rows"]
        self.supervision = MechanismSupervision.start(packet)
        self.censored, self.resolutions, self.seen = [], [], set()

    def receive(self, pairs, generation, sha):
        indices, outcomes = zip(*pairs, strict=True)
        watermark = max(o.available_at for o in outcomes)
        self.supervision = self.supervision.receive(
            outcomes, at=max(watermark, self.supervision.record()["clock"] or watermark)
        )
        self.resolutions.append(
            dict(generation=generation, row_indices=list(indices), native_record_sha256=sha)
        )

    def record(self):
        return dict(
            generation=self.entry["generation"],
            collection=self.packet.record(),
            supervision=self.supervision.record(),
            censored=self.censored,
            resolutions=self.resolutions,
            packet_reference=dict(self.entry["packet"]),
        )


def _receive(session, states, lookup, record, generation, sha):
    received = {}
    for kind in ("feedback", "excluded", "finalized"):
        for item in record[kind]:
            p = item["prediction"]
            match = lookup.get((p["asset"], p["decision_at"]))
            if match is None:
                continue
            state, index = match
            _require(index not in state.seen, "El prefijo repite el resultado de una predicción")
            state.seen.add(index)
            if kind == "finalized":
                _require(
                    record["close_phase"]
                    and item["closed_at"] == record["cutoff"]
                    and item["closed_at"] == session.phase.close_at,
                    "La censura no conserva su cierre nativo",
                )
                state.censored.append(
                    dict(
                        row_index=index,
                        generation=generation,
                        closed_at=item["closed_at"],
                        native_record_sha256=sha,
                    )
                )
            else:
                row = state.rows[index]
                outcome = CohortOutcome(
                    row["flow_id"].split("/", 1)[0],
                    row["prediction_at"],
                    row["sample_id"],
                    item["available_at"] if kind == "feedback" else record["cutoff"],
                    value=item["target"] if kind == "feedback" else None,
                    status="label" if kind == "feedback" else "excluded",
                    evidence_sha256=None
                    if kind == "feedback"
                    else item["evidence"]["evidence_sha256"],
                )
                received.setdefault(state.entry["generation"], []).append((index, outcome))
    for key, pairs in received.items():
        states[key].receive(pairs, generation, sha)


def _check_event_order(previous_at, previous_kind, event, before):
    _require(
        previous_at <= event["cutoff"] < before,
        "El registro retrocede o alcanza el corte histórico exclusivo",
    )
    _require(
        event["cutoff"] != previous_at
        or (event["kind"] == "settlement" and previous_kind in ("warmup", "decision")),
        "El registro repite el corte sin una liquidación posterior admisible",
    )


def _prefix(session, bundle, generations, *, through_generation, before, max_records, max_bytes):
    _require(bundle is not None, "No hay una recogida confirmada")
    bounded_integer(before, "corte histórico exclusivo", 1, FINAL_TEST_US)
    _require(
        type(generations) is tuple and 1 <= len(generations) <= 256,
        "La ventana necesita una tupla de hasta 256 generaciones explícitas",
    )
    for generation in generations:
        bounded_integer(generation, "generación de recogida", 1, bundle["generation"])
    _require(
        list(generations) == sorted(set(generations)),
        "Las generaciones de la ventana se repiten o desordenan",
    )
    bounded_integer(
        through_generation, "última generación histórica", generations[-1], bundle["generation"]
    )
    bounded_integer(max_records, "máximo de registros históricos", 1, 256)
    bounded_integer(max_bytes, "máximo de bytes históricos", 1, 1024**3)
    _require(
        through_generation - generations[0] + 1 <= max_records,
        "El prefijo supera sus registros",
    )
    binding = session._binding
    manifest = bundle["mechanism_collections"]
    binding.collections.index(
        session, manifest, generation=bundle["generation"], cutoff=bundle["cutoff"]
    )
    chosen = set(generations)
    selected = [entry for entry in manifest["entries"] if entry["generation"] in chosen]
    _require(
        len(selected) == len(generations), "La generación no contiene una recogida de decisión"
    )
    _require(
        all(entry["cutoff"] < before for entry in selected),
        "La recogida no es anterior al corte histórico",
    )
    total, states, lookup, records = 0, {}, {}, []
    for entry in selected:
        total += entry["packet"]["bytes"]
        _require(total < max_bytes, "Los paquetes agotan los bytes del prefijo histórico")
        packet = CollectedMechanisms.restore(session._read(entry["packet"], PACKET))
        state = _Collection(entry, packet)
        states[entry["generation"]] = state
        for i, row in enumerate(state.rows):
            key = _key(row)
            _require(key not in lookup, "Las colecciones repiten una predicción")
            lookup[key] = state, i
    previous, previous_kind = selected[0]["cutoff"] - 1, None
    for current in range(generations[0], through_generation + 1):
        event = binding.history.read_event(bundle["mechanism_history"], current)
        _check_event_order(previous, previous_kind, event, before)
        _require(total < max_bytes, "El prefijo agota sus bytes antes del registro nativo")
        record, sha, size = binding._read_record(
            session, current, maximum=min(max_bytes - total, 64 * 1024**2)
        )
        _require(
            sha == event["native_record_sha256"], "El registro histórico cambió su huella nativa"
        )
        binding._check_payload(event["payload"], record)
        binding._check_record(
            dict(event["payload"], observations=record["observations_count"]),
            record,
            cutoff=event["cutoff"],
        )
        _require(
            event["kind"] == record["event_kind"],
            "El tipo del evento histórico contradice su registro nativo",
        )
        if current in states:
            state = states[current]
            entry = state.entry
            _require(
                record["event_kind"] == "decision"
                and record["cutoff"] == entry["cutoff"]
                and event["payload"]["collection_packet"] == entry["packet"],
                "La colección no enlaza su emisión nativa",
            )
            binding.collections.check_anchor(state.packet, event["payload"])
        _receive(session, states, lookup, record, current, sha)
        total += size
        records.append(dict(generation=current, cutoff=record["cutoff"], sha256=sha, bytes=size))
        previous, previous_kind = record["cutoff"], record["event_kind"]
    return states, records, total


def collection_at(
    session, bundle, generation, *, through_generation, before, max_records, max_bytes
):
    """Conservar el formato de una vista individual desde el mismo recorrido causal."""
    states, records, total = _prefix(
        session,
        bundle,
        (generation,),
        through_generation=through_generation,
        before=before,
        max_records=max_records,
        max_bytes=max_bytes,
    )
    state = states[generation]
    packet, censored = state.packet, state.censored
    result = dict(
        collection=packet.record(),
        supervision=state.supervision.record(),
        censored=censored,
        as_of=dict(
            schema_version=1,
            kind="native_mechanism_view_v1",
            contract_id=session.contract_id,
            model_id=session.model_id,
            collection_generation=generation,
            through_generation=through_generation,
            before=before,
            native_records=records,
            resolutions=state.resolutions,
            source_bytes=total,
            packet_reference=dict(state.entry["packet"]),
        ),
    )
    result["as_of"]["view_sha256"] = digest(
        dict(
            collection_sha256=packet.sha256,
            supervision=result["supervision"],
            censored=censored,
            as_of=result["as_of"],
        )
    )
    return result


def collections_at(
    session, bundle, generations, *, through_generation, before, max_records, max_bytes
):
    """Una procedencia común, sin repetir registros ni recibos posteriores por colección."""
    states, records, total = _prefix(
        session,
        bundle,
        generations,
        through_generation=through_generation,
        before=before,
        max_records=max_records,
        max_bytes=max_bytes,
    )
    result = dict(
        collections=[state.record() for state in states.values()],
        as_of=dict(
            schema_version=1,
            kind="native_mechanism_window_v1",
            contract_id=session.contract_id,
            model_id=session.model_id,
            collection_generations=list(generations),
            through_generation=through_generation,
            before=before,
            native_records=records,
            source_bytes=total,
        ),
    )
    result["as_of"]["window_sha256"] = digest(
        dict(
            collections=[
                dict(
                    collection_sha256=state.packet.sha256,
                    **{k: v for k, v in item.items() if k != "collection"},
                )
                for state, item in zip(states.values(), result["collections"], strict=True)
            ],
            as_of=result["as_of"],
        )
    )
    return result
