"""Archivo acotado de alternativas y resultados bajo la cabeza nativa de la sesión."""

from dataclasses import asdict, dataclass

import numpy as np

from mars_titan.calibration.cohort_feedback import CohortOutcome
from mars_titan.models.titans.config import FingerprintOnce, bounded_integer
from mars_titan.models.titans.financial_inputs import FINAL_TEST_US
from mars_titan.models.titans.mechanism_policy import ACTIONS

from .mechanism_collection_record import CollectedMechanisms
from .mechanism_history import HistoryCapacityError, _digest
from .mechanism_supervision import MechanismSupervision

PACKET = "mechanism_collection"
RECEIPT = "mechanism_receipt"
KIND = "native_mechanism_collections_v1"


@dataclass(frozen=True)
class MechanismCollectionConfig(FingerprintOnce):
    """Agotar el archivo detiene la propuesta, sin retirar evidencia confirmada."""

    max_decisions: int = 64

    def __post_init__(self):
        bounded_integer(self.max_decisions, "decisiones recogidas", 1, 256)

    def identity(self):
        return dict(
            kind=KIND,
            **asdict(self),
            anchor_tolerance={"torch.float64": 1e-12, "torch.float32": 1e-5},
            closure="native_censoring_without_label_or_changed_timestamp",
        )


def _key(row):
    return row["flow_id"], row["prediction_at"]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class NativeMechanismCollections:
    """Referencias directas, pendientes indexados y lectura de una decisión por vez.

    El índice no incluye cuantiles ni etiquetas. El receptor guarda la procedencia de
    cada resultado en registros nativos. Las censuras administrativas tienen su propio
    estado y no se pasan como etiquetas a la supervisión de desarrollo.
    """

    def __init__(self, config):
        _require(type(config) is MechanismCollectionConfig, "Configuración de recogida inválida")
        self.config = config

    def empty(self):
        return dict(schema_version=1, kind=KIND, config_id=self.config.fingerprint(), entries=[])

    def index(self, session, manifest, *, generation, cutoff):
        _require(
            type(manifest) is dict
            and set(manifest) == set(self.empty())
            and type(manifest["schema_version"]) is int
            and manifest["schema_version"] == 1
            and manifest["kind"] == KIND
            and manifest["config_id"] == self.config.fingerprint()
            and type(manifest["entries"]) is list,
            "El índice de recogida no conserva su contrato",
        )
        bounded_integer(
            len(manifest["entries"]), "decisiones archivadas", 0, self.config.max_decisions
        )
        previous, clock, pending = 0, 0, set()
        for entry in manifest["entries"]:
            _require(
                type(entry) is dict
                and set(entry) == {"generation", "cutoff", "packet", "receipt", "pending"},
                "La decisión archivada no conserva sus referencias",
            )
            bounded_integer(entry["generation"], "generación recogida", previous + 1, generation)
            bounded_integer(
                entry["cutoff"], "corte de recogida", clock + 1, min(cutoff, FINAL_TEST_US - 1)
            )
            previous, clock = entry["generation"], entry["cutoff"]
            for name, kind in (("packet", PACKET), ("receipt", RECEIPT)):
                ref = entry[name]
                header = session.artifacts._header(session.contract_id, kind)
                _require(
                    type(ref) is dict
                    and set(ref) == set(header) | {"name", "bytes", "sha256"}
                    and all(type(ref[k]) is type(v) and ref[k] == v for k, v in header.items()),
                    "Una referencia de recogida pertenece a otro contrato",
                )
                _digest(ref["sha256"])
                _require(ref["name"] == ref["sha256"] + ".bin", "Nombre de recogida inválido")
                bounded_integer(ref["bytes"], "bytes de recogida", 1, session.artifacts.max_bytes)
            _require(
                type(entry["pending"]) is list and len(entry["pending"]) <= 8192,
                "La cola de recogida supera su límite",
            )
            _require(
                all(
                    type(k) is list
                    and len(k) == 2
                    and type(k[0]) is str
                    and type(k[1]) is int
                    and k[1] == entry["cutoff"]
                    for k in entry["pending"]
                ),
                "El índice pendiente no conserva flujo y corte",
            )
            keys = [tuple(k) for k in entry["pending"]]
            _require(
                keys == sorted(set(keys)) and not pending.intersection(keys),
                "La recogida repite o desordena pendientes",
            )
            pending.update(keys)
        _require(len(pending) <= 32768, "La recogida supera la cola financiera")
        return pending

    @staticmethod
    def check_anchor(packet, control):
        body = packet.payload()
        rows = body["rows"]
        fields = (
            "flow_id",
            "sample_id",
            "prediction_at",
            "input_available_at",
            "input_sha256",
            "matures_at",
        )
        _require(
            [{k: r[k] for k in fields} for r in rows]
            == [{k: r[k] for k in fields} for r in control["rows"]],
            "La colección no conserva las observaciones emitidas",
        )
        observation = body["observation"]["observation"]
        _require(
            observation["context_id"] == control["context_id"]
            and observation["snapshot_sha256"] == control["snapshot_sha256"]
            and observation["consumer_id"] == control["consumer_id"],
            "La colección no conserva el consumidor o snapshot emitido",
        )
        tolerance = {"torch.float64": 1e-12, "torch.float32": 1e-5}[body["output_dtype"]]
        for group, anchor in zip(body["groups"], body["anchor"], strict=True):
            emitted = [r for r in control["rows"] if r["flow_id"].startswith(group["market"] + "/")]
            _require(
                all(r["requested"] == anchor["requested"] for r in emitted),
                "La colección cambió la acción ancla solicitada",
            )
            candidate = group["candidates"][ACTIONS.index(anchor["applied"])]
            _require(candidate["status"] != "unavailable", "Falta la salida del ancla")
            _require(
                np.allclose(
                    candidate["quantiles"],
                    [r["quantiles"] for r in emitted],
                    rtol=tolerance,
                    atol=tolerance,
                ),
                "Las alternativas no reproducen la emisión ancla dentro de su tolerancia",
            )

    def _load(self, session, entry, manifest):
        packet = CollectedMechanisms.restore(session._read(entry["packet"], PACKET))
        receipt = session._read(entry["receipt"], RECEIPT)
        _require(
            type(receipt) is dict
            and set(receipt)
            == {
                "schema_version",
                "creation_record_sha256",
                "supervision",
                "resolutions",
                "censored",
            }
            and type(receipt["schema_version"]) is int
            and receipt["schema_version"] == 1,
            "El recibo no conserva su esquema",
        )
        record, digest = session._binding._record(session, entry["generation"])
        event = session._binding.read_event(session, manifest, entry["generation"])
        _require(
            receipt["creation_record_sha256"] == digest
            and record["event_kind"] == "decision"
            and record["cutoff"] == entry["cutoff"]
            and event["payload"]["collection_packet"] == entry["packet"],
            "La colección no enlaza su emisión nativa",
        )
        self.check_anchor(packet, event["payload"])
        supervision = MechanismSupervision.restore(packet, receipt["supervision"])
        self._provenance(
            session,
            entry,
            packet,
            receipt,
            generation=manifest["generation"],
            cutoff=manifest["cutoff"],
        )
        outcomes = receipt["supervision"]["outcomes"]
        censored = {r["row_index"] for r in receipt["censored"]}
        expected = [
            list(_key(row))
            for i, row in enumerate(packet.payload()["rows"])
            if outcomes[i] is None and i not in censored
        ]
        _require(expected == entry["pending"], "El índice contradice los resultados pendientes")
        return packet, supervision, receipt

    @staticmethod
    def _provenance(session, entry, packet, receipt, *, generation, cutoff):
        rows, targets = packet.payload()["rows"], receipt["supervision"]["outcomes"]
        watermarks = [target["available_at"] for target in targets if target is not None]
        _require(
            receipt["supervision"]["clock"] == (max(watermarks) if watermarks else None),
            "La supervisión cambió su watermark de disponibilidad",
        )
        seen = set()
        _require(
            type(receipt["resolutions"]) is list
            and len(receipt["resolutions"]) <= len(rows)
            and type(receipt["censored"]) is list
            and len(receipt["censored"]) <= len(rows),
            "El recibo supera sus resoluciones",
        )
        for resolution in receipt["resolutions"]:
            _require(
                type(resolution) is dict
                and set(resolution) == {"generation", "native_record_sha256", "row_indices"}
                and type(resolution["row_indices"]) is list
                and bool(resolution["row_indices"]),
                "Falta la procedencia nativa de las etiquetas",
            )
            bounded_integer(
                resolution["generation"],
                "generación del resultado",
                entry["generation"],
                generation,
            )
            record, digest = session._binding._record(session, resolution["generation"])
            _require(
                digest == resolution["native_record_sha256"] and record["cutoff"] <= cutoff,
                "El resultado nativo cambió o pertenece al futuro",
            )
            available = {}
            for native in record["feedback"]:
                p = native["prediction"]
                available[p["asset"], p["decision_at"]] = dict(
                    available_at=native["available_at"],
                    value=native["target"],
                    status="label",
                    evidence_sha256=None,
                )
            for native in record["excluded"]:
                p = native["prediction"]
                available[p["asset"], p["decision_at"]] = dict(
                    available_at=record["cutoff"],
                    value=None,
                    status="excluded",
                    evidence_sha256=native["evidence"]["evidence_sha256"],
                )
            for index in resolution["row_indices"]:
                bounded_integer(index, "fila resuelta", 0, len(rows) - 1)
                _require(
                    index not in seen
                    and _key(rows[index]) in available
                    and targets[index] == available[_key(rows[index])],
                    "El recibo no conserva el resultado nativo de la predicción",
                )
                seen.add(index)
        _require(
            seen == {i for i, target in enumerate(targets) if target is not None},
            "Una etiqueta no conserva procedencia nativa",
        )
        for item in receipt["censored"]:
            _require(
                type(item) is dict
                and set(item) == {"row_index", "generation", "closed_at", "native_record_sha256"},
                "La censura no conserva su procedencia",
            )
            index = item["row_index"]
            bounded_integer(index, "fila censurada", 0, len(rows) - 1)
            bounded_integer(
                item["generation"], "generación de cierre", entry["generation"], generation
            )
            record, digest = session._binding._record(session, item["generation"])
            matches = [
                n
                for n in record["finalized"]
                if (n["prediction"]["asset"], n["prediction"]["decision_at"]) == _key(rows[index])
            ]
            _require(
                index not in seen
                and targets[index] is None
                and digest == item["native_record_sha256"]
                and len(matches) == 1
                and record["close_phase"]
                and record["cutoff"] <= cutoff
                and item["closed_at"] == session.phase.close_at
                and matches[0]["closed_at"] == item["closed_at"],
                "La censura no corresponde al cierre nativo",
            )
            seen.add(index)

    def resolve(
        self,
        session,
        bundle,
        *,
        record,
        record_sha256,
        packet_reference,
        outcomes,
        excluded,
        finalized,
    ):
        _require(
            record["generation"] == bundle["generation"] and record["cutoff"] == bundle["cutoff"],
            "La recepción no corresponde a la generación nativa",
        )
        manifest = bundle["mechanism_collections"]
        self.index(session, manifest, generation=bundle["generation"], cutoff=bundle["cutoff"])
        entries = [dict(entry) for entry in manifest["entries"]]
        if packet_reference is not None:
            if len(entries) >= self.config.max_decisions:
                raise HistoryCapacityError(
                    "La recogida agota su límite de decisiones sin borrar evidencia"
                )
            packet = CollectedMechanisms.restore(session._read(packet_reference, PACKET))
            receipt = dict(
                schema_version=1,
                creation_record_sha256=record_sha256,
                supervision=MechanismSupervision.start(packet).record(),
                resolutions=[],
                censored=[],
            )
            entries.append(
                dict(
                    generation=bundle["generation"],
                    cutoff=bundle["cutoff"],
                    packet=packet_reference,
                    receipt=session._stage(receipt, RECEIPT),
                    pending=[list(_key(r)) for r in packet.payload()["rows"]],
                )
            )
        changes = {}
        for item in (*outcomes, *excluded, *finalized):
            p = item.prediction
            changes[p.asset, p.decision_at] = item
        consumed = set()
        for entry in entries:
            keys = set(map(tuple, entry["pending"]))
            wanted = keys.intersection(changes)
            if not wanted:
                continue
            packet, supervision, receipt = self._load(session, entry, bundle)
            received, indices = [], []
            for i, row in enumerate(packet.payload()["rows"]):
                key = _key(row)
                if key not in wanted:
                    continue
                item = changes[key]
                if hasattr(item, "closed_at"):
                    receipt["censored"].append(
                        dict(
                            row_index=i,
                            generation=bundle["generation"],
                            closed_at=item.closed_at,
                            native_record_sha256=record_sha256,
                        )
                    )
                else:
                    target = (
                        dict(value=item.label.value)
                        if hasattr(item, "label")
                        else dict(status="excluded", evidence_sha256=item.evidence.evidence_sha256)
                    )
                    received.append(
                        CohortOutcome(
                            row["flow_id"].split("/", 1)[0],
                            row["prediction_at"],
                            row["sample_id"],
                            item.label.available_at if hasattr(item, "label") else bundle["cutoff"],
                            **target,
                        )
                    )
                    indices.append(i)
                consumed.add(key)
            if received:
                # Este reloj es el watermark de disponibilidades recibidas. El momento
                # de recepción real queda en el registro nativo, incluido el cierre.
                clock = max(o.available_at for o in received)
                previous = supervision.record()["clock"]
                supervision = supervision.receive(received, at=max(clock, previous or clock))
                receipt["supervision"] = supervision.record()
                receipt["resolutions"].append(
                    dict(
                        generation=bundle["generation"],
                        native_record_sha256=record_sha256,
                        row_indices=indices,
                    )
                )
            entry["receipt"] = session._stage(receipt, RECEIPT)
            entry["pending"] = [list(k) for k in sorted(keys - wanted)]
        _require(consumed == set(changes), "Un resultado nativo no enlaza la recogida pendiente")
        return dict(manifest, entries=entries)

    def verify(self, session, bundle, *, strong=False):
        manifest = bundle["mechanism_collections"]
        pending = self.index(
            session, manifest, generation=bundle["generation"], cutoff=bundle["cutoff"]
        )
        actual = {_key(row) for row in session._read(bundle["pending"], "pending")["rows"]}
        _require(pending == actual, "Las colas nativa y de recogida no coinciden")
        current = session._read(bundle["control"], "control")["collection_packet"]
        selected = [e for e in manifest["entries"] if e["generation"] == bundle["generation"]]
        _require(
            (selected[0]["packet"] if selected else None) == current,
            "El bundle perdió su colección actual",
        )
        for entry in manifest["entries"]:
            if strong or entry["pending"] or entry["generation"] == bundle["generation"]:
                self._load(session, entry, bundle)

    def read(self, session, bundle, generation):
        _require(bundle is not None, "No hay una recogida confirmada")
        bounded_integer(generation, "generación de recogida", 1, bundle["generation"])
        manifest = bundle["mechanism_collections"]
        self.index(session, manifest, generation=bundle["generation"], cutoff=bundle["cutoff"])
        selected = [entry for entry in manifest["entries"] if entry["generation"] == generation]
        _require(len(selected) == 1, "La generación no contiene una recogida de decisión")
        packet, supervision, receipt = self._load(session, selected[0], bundle)
        return packet, supervision, receipt["censored"]

    def live_references(self, session, bundle):
        manifest = bundle["mechanism_collections"]
        self.index(session, manifest, generation=bundle["generation"], cutoff=bundle["cutoff"])
        return [entry[name] for entry in manifest["entries"] for name in ("packet", "receipt")]
