"""Reconstruir resultados desde un prefijo nativo explícito, sin recibos posteriores."""

from mars_titan.calibration.cohort_feedback import CohortOutcome
from mars_titan.models.titans.config import bounded_integer
from mars_titan.models.titans.financial_inputs import FINAL_TEST_US
from mars_titan.models.titans.mechanism_selection import digest

from .mechanism_collection_record import CollectedMechanisms
from .mechanism_supervision import MechanismSupervision
from .native_mechanism_collection import PACKET, _key, _require


def collection_at(
    session, bundle, generation, *, through_generation, before, max_records, max_bytes
):
    """Corte exclusivo y generación incluidos por contrato, sin descubrirlos con etiquetas.

    El presupuesto de bytes cubre el paquete y los registros nativos. Los bloques del
    historial conservan los límites propios del archivo y se verifican por separado.
    """
    _require(bundle is not None, "No hay una recogida confirmada")
    bounded_integer(before, "corte histórico exclusivo", 1, FINAL_TEST_US)
    bounded_integer(generation, "generación de recogida", 1, bundle["generation"])
    bounded_integer(
        through_generation, "última generación histórica", generation, bundle["generation"]
    )
    bounded_integer(max_records, "máximo de registros históricos", 1, 256)
    bounded_integer(max_bytes, "máximo de bytes históricos", 1, 1024**3)
    _require(through_generation - generation + 1 <= max_records, "El prefijo supera sus registros")
    binding = session._binding
    manifest = bundle["mechanism_collections"]
    binding.collections.index(
        session, manifest, generation=bundle["generation"], cutoff=bundle["cutoff"]
    )
    selected = [entry for entry in manifest["entries"] if entry["generation"] == generation]
    _require(len(selected) == 1, "La generación no contiene una recogida de decisión")
    entry = selected[0]
    _require(entry["cutoff"] < before, "La recogida no es anterior al corte histórico")
    total = entry["packet"]["bytes"]
    _require(total < max_bytes, "El paquete agota los bytes del prefijo histórico")
    packet = CollectedMechanisms.restore(session._read(entry["packet"], PACKET))
    rows = packet.payload()["rows"]
    lookup = {_key(row): i for i, row in enumerate(rows)}
    supervision, censored, seen, records = MechanismSupervision.start(packet), [], set(), []
    resolutions = []
    previous = entry["cutoff"] - 1
    for current in range(generation, through_generation + 1):
        event = binding.history.read_event(bundle["mechanism_history"], current)
        _require(
            previous < event["cutoff"] < before, "El registro alcanza el futuro del corte histórico"
        )
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
        if current == generation:
            _require(
                record["event_kind"] == "decision"
                and record["cutoff"] == entry["cutoff"]
                and event["payload"]["collection_packet"] == entry["packet"],
                "La colección no enlaza su emisión nativa",
            )
            binding.collections.check_anchor(packet, event["payload"])
        received, indices = [], []
        for kind in ("feedback", "excluded", "finalized"):
            for item in record[kind]:
                p = item["prediction"]
                index = lookup.get((p["asset"], p["decision_at"]))
                if index is None:
                    continue
                _require(index not in seen, "El prefijo repite el resultado de una predicción")
                seen.add(index)
                if kind == "finalized":
                    _require(
                        record["close_phase"]
                        and item["closed_at"] == record["cutoff"]
                        and item["closed_at"] == session.phase.close_at,
                        "La censura no conserva su cierre nativo",
                    )
                    censored.append(
                        dict(
                            row_index=index,
                            generation=current,
                            closed_at=item["closed_at"],
                            native_record_sha256=sha,
                        )
                    )
                else:
                    row = rows[index]
                    indices.append(index)
                    received.append(
                        CohortOutcome(
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
                    )
        if received:
            watermark = max(o.available_at for o in received)
            supervision = supervision.receive(
                received, at=max(watermark, supervision.record()["clock"] or watermark)
            )
            resolutions.append(
                dict(generation=current, row_indices=indices, native_record_sha256=sha)
            )
        total += size
        records.append(dict(generation=current, cutoff=record["cutoff"], sha256=sha, bytes=size))
        previous = record["cutoff"]
    result = dict(
        collection=packet.record(),
        supervision=supervision.record(),
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
            resolutions=resolutions,
            source_bytes=total,
            packet_reference=entry["packet"],
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
