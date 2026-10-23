"""Leer una trayectoria nativa completa, sin colección ni alternativas reconstruidas.

Este lector interno consume el enlace verificado de FinancialSession. Los sellos
comprueban coherencia, sin autenticar una falsificación completa del archivo local.
El tamaño de un snapshot no archivado permanece desconocido, aunque exista su hash.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from mars_titan.memory.financial_mechanisms import MechanismBinding
from mars_titan.memory.financial_session import FinancialPhase, FinancialSession, _check_callbacks
from mars_titan.memory.mechanism_collection_asof import _check_event_order
from mars_titan.models.titans.config import bounded_integer
from mars_titan.models.titans.financial_inputs import FINAL_TEST_US, FLOW_ID

from .mature_review import _loss, _quantiles, encoded, identifier, require

MAX_TRAJECTORY_BYTES = 8 * 1024**2
MAX_SOURCE_BYTES = 64 * 1024**2
MAX_ROWS = 8192
MAX_RECORDS = 256
MAX_INTEGER = 2**63 - 1
KIND = "native_mature_trajectory_v1"
BODY_FIELDS = {
    "schema_version",
    "kind",
    "phase",
    "consumer_id",
    "parameter_id",
    "source_contract_id",
    "through_generation",
    "head_sha256",
    "before",
    "native_records",
    "logical_source_bytes",
    "rows",
}
ROW_FIELDS = {
    "flow_id",
    "sample_id",
    "prediction_at",
    "input_available_at",
    "input_sha256",
    "matures_at",
    "snapshot_sha256",
    "bank_entries",
    "prediction_id",
    "emitted_quantiles",
    "status",
    "target",
    "available_at",
    "native_received_at",
    "receipt_generation",
    "native_record_sha256",
    "emitted_point_loss",
}
PREDICTION_FIELDS = {"id", "asset", "task", "horizon", "generation", "decision_at", "value"}
PHASE_FIELDS = {"partition", "warmup_start", "decision_start", "decision_end", "close_at"}
RECORD_FIELDS = {"generation", "cutoff", "sha256", "bytes"}


def _finite(value, name):
    require(type(value) is float and math.isfinite(value), f"{name} debe ser un float finito")


def _header(body):
    require(type(body) is dict and set(body) == BODY_FIELDS, "La trayectoria cambió su esquema")
    require(
        type(body["schema_version"]) is int
        and body["schema_version"] == 1
        and body["kind"] == KIND,
        "La trayectoria no conserva su versión y clase",
    )
    bounded_integer(body["before"], "corte de trayectoria", 1, FINAL_TEST_US)
    phase = body["phase"]
    require(type(phase) is dict and set(phase) == PHASE_FIELDS, "La fase cambió su esquema")
    require(type(phase["partition"]) is str, "La partición necesita un nombre explícito")
    FinancialPhase(**phase)
    require(
        phase["partition"] in ("train", "validation") and phase["close_at"] < body["before"],
        "La trayectoria debe ser una fase permitida cerrada antes del corte",
    )
    for name in ("consumer_id", "parameter_id", "source_contract_id", "head_sha256"):
        identifier(body[name], name)
    count = body["through_generation"]
    bounded_integer(count, "generaciones de trayectoria", 1, MAX_RECORDS)
    require(
        type(body["native_records"]) is list and len(body["native_records"]) == count,
        "Falta el prefijo nativo completo",
    )
    previous, used, records = phase["warmup_start"], 0, {}
    for generation, record in enumerate(body["native_records"], 1):
        require(
            type(record) is dict and set(record) == RECORD_FIELDS, "Un registro cambió su esquema"
        )
        bounded_integer(record["generation"], "generación nativa", generation, generation)
        bounded_integer(record["cutoff"], "corte nativo", previous, phase["close_at"])
        bounded_integer(record["bytes"], "bytes nativos", 1, MAX_SOURCE_BYTES)
        identifier(record["sha256"], "El registro nativo")
        records[generation] = record
        previous = record["cutoff"]
        used += record["bytes"]
    require(
        records[count]["sha256"] == body["head_sha256"]
        and records[count]["cutoff"] == phase["close_at"],
        "La trayectoria no conserva la cabeza cerrada",
    )
    bounded_integer(body["logical_source_bytes"], "bytes lógicos", used, MAX_SOURCE_BYTES)
    return phase, records


def _row(row, phase, records, before, cutoffs):
    require(
        type(row) is dict and set(row) == ROW_FIELDS, "La fila de trayectoria cambió su esquema"
    )
    flow, at = row["flow_id"], row["prediction_at"]
    require(type(flow) is str and FLOW_ID.fullmatch(flow) is not None, "El flujo no es válido")
    bounded_integer(at, "decisión emitida", phase["decision_start"], phase["decision_end"] - 1)
    require(
        row["sample_id"] == f"{flow}/{at}" and at in cutoffs,
        "La fila no corresponde a una emisión del prefijo",
    )
    bounded_integer(row["input_available_at"], "disponibilidad de inputs", 0, at)
    identifier(row["input_sha256"], "Los inputs")
    # prediction_id es un string SHA-256 generado por el ejecutor nativo.
    identifier(row["prediction_id"], "La predicción nativa")
    _quantiles(row["emitted_quantiles"])
    if row["snapshot_sha256"] is None:
        bounded_integer(row["bank_entries"], "episodios del snapshot ausente", 0, 0)
    else:
        identifier(row["snapshot_sha256"], "El snapshot")
        if row["bank_entries"] is not None:
            bounded_integer(row["bank_entries"], "episodios del snapshot", 0, 1024)
    maturity = row["matures_at"]
    if maturity is not None:
        bounded_integer(maturity, "madurez declarada", at + 1, MAX_INTEGER)
    status = row["status"]
    require(
        type(status) is str and status in {"label", "pending", "excluded", "censored"},
        "Estado desconocido",
    )
    if status == "pending":
        require(
            row["available_at"]
            is row["native_received_at"]
            is row["receipt_generation"]
            is row["native_record_sha256"]
            is None,
            "La fila pendiente inventa una recepción",
        )
    else:
        generation = row["receipt_generation"]
        require(
            type(generation) is int and generation in records, "La recepción perdió su generación"
        )
        native = records[generation]
        require(
            row["native_received_at"] == native["cutoff"]
            and row["native_record_sha256"] == native["sha256"]
            and at <= native["cutoff"] < before,
            "La recepción no corresponde al registro nativo",
        )
        if status == "censored":
            require(
                row["available_at"] is None and native["cutoff"] == phase["close_at"],
                "La censura no corresponde al cierre",
            )
        else:
            minimum = maturity if status == "label" else at
            require(type(minimum) is int, "La etiqueta no declara su madurez")
            bounded_integer(
                row["available_at"], "disponibilidad del resultado", minimum, native["cutoff"]
            )
    if status == "label":
        _finite(row["target"], "La etiqueta")
        _finite(row["emitted_point_loss"], "La pérdida")
        require(
            row["emitted_point_loss"] == _loss(row["emitted_quantiles"][2], row["target"]),
            "La pérdida no corresponde a la mediana nativa exacta",
        )
    else:
        require(
            row["target"] is row["emitted_point_loss"] is None,
            "Una fila sin etiqueta declara pérdida",
        )


def _validate(body):
    phase, records = _header(body)
    require(
        type(body["rows"]) is list and len(body["rows"]) <= MAX_ROWS,
        "La trayectoria supera sus filas",
    )
    samples, predictions = set(), set()
    cutoffs = {item["cutoff"] for item in records.values()}
    for row in body["rows"]:
        _row(row, phase, records, body["before"], cutoffs)
        require(
            row["sample_id"] not in samples and row["prediction_id"] not in predictions,
            "La trayectoria repite una muestra o predicción",
        )
        samples.add(row["sample_id"])
        predictions.add(row["prediction_id"])


@dataclass(frozen=True, init=False)
class MatureTrajectory:
    """Emisiones y resultados de un prefijo completo, con copias sin alias mutables.

    bank_entries cuenta el snapshot consultado, no el tamaño actual del banco.
    Si solo consta la huella del snapshot, su conteo permanece en None.
    """

    _json: str
    _identity: str

    def __init__(self, body):
        content = encoded(body, MAX_TRAJECTORY_BYTES)
        _validate(body)
        object.__setattr__(self, "_json", content)
        object.__setattr__(self, "_identity", hashlib.sha256(content.encode()).hexdigest())

    def record(self):
        require(type(self._json) is str, "La trayectoria no conserva su JSON")
        content = self._json.encode()
        require(
            len(content) <= MAX_TRAJECTORY_BYTES
            and hashlib.sha256(content).hexdigest() == self._identity,
            "La trayectoria no conserva su sello",
        )
        body = json.loads(self._json)
        _validate(body)
        return dict(body, trajectory_id=self._identity)


def _prediction(value, session, generation=None):
    require(
        type(value) is dict and set(value) == PREDICTION_FIELDS, "La predicción cambió su esquema"
    )
    identifier(value["id"], "La predicción nativa")
    require(
        type(value["asset"]) is str
        and FLOW_ID.fullmatch(value["asset"]) is not None
        and value["task"] == session.task
        and type(value["horizon"]) is int
        and value["horizon"] == session.horizon,
        "La predicción pertenece a otro flujo u objetivo",
    )
    bounded_integer(value["generation"], "generación de predicción", 1, MAX_RECORDS)
    bounded_integer(value["decision_at"], "corte de predicción", 1, FINAL_TEST_US - 1)
    if generation is not None:
        require(value["generation"] == generation, "La predicción pertenece a otra generación")
    _finite(value["value"], "La emisión nativa")
    return value


def _emissions(event, native, session, predictions, samples):
    result = []
    if event["kind"] != "decision":
        return result
    control = event["payload"]
    by_flow = {
        row["asset"]: _prediction(row, session, event["generation"])
        for row in native["predictions"]
    }
    for emitted in control["rows"]:
        flow = emitted["flow_id"]
        prediction = by_flow[flow]
        identifier_ = prediction["id"]
        require(
            identifier_ not in predictions and emitted["sample_id"] not in samples,
            "La fuente repite una emisión",
        )
        _quantiles(emitted["quantiles"])
        require(
            prediction["value"] == emitted["point"] == emitted["quantiles"][2],
            "La predicción no conserva la mediana nativa exacta",
        )
        snapshot = control["snapshot_sha256"]
        row = dict(
            **{
                name: emitted[name]
                for name in (
                    "flow_id",
                    "sample_id",
                    "prediction_at",
                    "input_available_at",
                    "input_sha256",
                    "matures_at",
                )
            },
            snapshot_sha256=snapshot,
            bank_entries=0 if snapshot is None else None,
            prediction_id=identifier_,
            emitted_quantiles=list(emitted["quantiles"]),
            status="pending",
            target=None,
            available_at=None,
            native_received_at=None,
            receipt_generation=None,
            native_record_sha256=None,
            emitted_point_loss=None,
        )
        predictions[identifier_] = (row, prediction)
        samples.add(row["sample_id"])
        result.append(row)
    return result


def _receive(native, digest, session, predictions):
    for kind in ("feedback", "excluded", "finalized"):
        require(type(native[kind]) is list, "El resultado nativo necesita una lista")
        for item in native[kind]:
            require(
                type(item) is dict and type(item.get("prediction")) is dict,
                "Falta la predicción del resultado",
            )
            prediction = _prediction(item["prediction"], session)
            require(
                prediction["id"] in predictions,
                "El resultado pertenece a una predicción no emitida",
            )
            row, original = predictions[prediction["id"]]
            require(
                prediction == original
                and row["status"] == "pending"
                and native["generation"] >= original["generation"]
                and native["cutoff"] >= row["prediction_at"],
                "La recepción cambió su predicción o repite un resultado",
            )
            row.update(
                native_received_at=native["cutoff"],
                receipt_generation=native["generation"],
                native_record_sha256=digest,
            )
            if kind == "feedback":
                require(
                    type(item.get("revision")) is int and item["revision"] == 0,
                    "Revisión nativa no admitida",
                )
                _finite(item.get("target"), "La etiqueta nativa")
                _finite(item.get("error"), "El error nativo")
                require(
                    item["error"] == item["target"] - original["value"],
                    "El error contradice la emisión",
                )
                row.update(
                    status="label",
                    target=item["target"],
                    available_at=item["available_at"],
                    emitted_point_loss=_loss(row["emitted_quantiles"][2], item["target"]),
                )
            elif kind == "excluded":
                evidence = item.get("evidence")
                require(
                    type(evidence) is dict
                    and evidence.get("asset") == original["asset"]
                    and evidence.get("decision_at") == original["decision_at"],
                    "La exclusión pertenece a otra decisión",
                )
                identifier(evidence.get("evidence_sha256"), "La exclusión nativa")
                row.update(status="excluded", available_at=native["cutoff"])
            else:
                require(
                    native["close_phase"] is True
                    and item.get("closed_at") == native["cutoff"] == session.phase.close_at,
                    "La finalización no conserva el cierre nativo",
                )
                row["status"] = "censored"


def read_mature_trajectory(session, *, before, max_records=MAX_RECORDS, max_bytes=MAX_SOURCE_BYTES):
    """Consumir el prefijo cerrado completo mediante el enlace nativo verificado."""
    bounded_integer(before, "corte exclusivo", 1, FINAL_TEST_US)
    bounded_integer(max_records, "máximo de registros", 1, MAX_RECORDS)
    bounded_integer(max_bytes, "máximo de bytes", 1, MAX_SOURCE_BYTES)
    require(type(session) is FinancialSession, "La fuente necesita una FinancialSession real")
    _check_callbacks(session)
    binding = session._binding
    require(type(binding) is MechanismBinding, "La sesión no tiene el enlace de mecanismos")
    phase = session.phase
    require(
        phase.partition in ("train", "validation") and phase.close_at < before,
        "Partición o corte de trayectoria prohibidos",
    )
    consumer = session.consumer
    consumer.verify(strong=True)
    identity = consumer.identity()
    snapshot = session.snapshot()
    require(snapshot["phase_closed"] is True, "La trayectoria necesita una fase cerrada")
    count = snapshot["cursor"]
    bounded_integer(count, "registros del prefijo", 1, max_records)
    rows, records, predictions, samples = [], [], {}, set()
    used, previous_at, previous_kind, previous_sha = 0, None, None, None
    for generation in range(1, count + 1):
        require(used < max_bytes, "La trayectoria agotó sus bytes antes del siguiente registro")
        native, digest, size = binding._read_record(session, generation, maximum=max_bytes - used)
        used += size
        event = session.mechanism_event(generation)
        require(used < max_bytes, "No quedan bytes para el evento verificado")
        used += len(encoded(event, max_bytes - used).encode())
        require(
            event["generation"] == generation
            and event["kind"] == native["event_kind"]
            and event["native_record_sha256"] == digest
            and native["identity_sha256"] == snapshot["identity_sha256"]
            and (previous_sha is None or native["previous_sha256"] == previous_sha),
            "La generación no conserva el enlace y la cadena nativa",
        )
        _check_event_order(
            event["cutoff"] - 1 if previous_at is None else previous_at,
            previous_kind,
            event,
            before,
        )
        binding._check_payload(event["payload"], native)
        binding._check_record(
            dict(event["payload"], observations=native["observations_count"]),
            native,
            cutoff=event["cutoff"],
        )
        require(
            generation == count or native["close_phase"] is False,
            "La trayectoria conserva registros posteriores al cierre",
        )
        rows.extend(_emissions(event, native, session, predictions, samples))
        require(len(rows) <= MAX_ROWS, "La trayectoria supera sus filas")
        _receive(native, digest, session, predictions)
        records.append(
            dict(generation=generation, cutoff=event["cutoff"], sha256=digest, bytes=size)
        )
        previous_at, previous_kind, previous_sha = event["cutoff"], event["kind"], digest
    require(
        previous_sha == snapshot["record_sha256"]
        and native["close_phase"] is True
        and previous_at == phase.close_at,
        "La última generación no corresponde a la cabeza cerrada",
    )
    consumer.verify(strong=True)
    _check_callbacks(session)
    require(
        consumer.identity() == identity and session.snapshot() == snapshot,
        "El consumidor o la fuente cambiaron durante la lectura",
    )
    return MatureTrajectory(
        dict(
            schema_version=1,
            kind=KIND,
            phase=asdict(phase),
            consumer_id=consumer.model_id,
            parameter_id=consumer.predictor._parameter_id,
            source_contract_id=session.contract_id,
            through_generation=count,
            head_sha256=snapshot["record_sha256"],
            before=before,
            native_records=records,
            logical_source_bytes=used,
            rows=rows,
        )
    )
