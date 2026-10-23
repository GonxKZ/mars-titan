"""Señales de emisiones maduras sobre vistas y eventos ya verificados.

Las dos huellas esperadas proceden del coordinador tras leer el archivo nativo.
Su coherencia no autentica al coordinador ni impide un rollback local completo.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass

from mars_titan.memory.mechanism_collection_record import CollectedMechanisms
from mars_titan.memory.mechanism_history import EVENT_FIELDS
from mars_titan.memory.mechanism_supervision import MechanismSupervision
from mars_titan.memory.native_mechanism_collection import NativeMechanismCollections
from mars_titan.models.titans.config import bounded_integer
from mars_titan.models.titans.financial_inputs import FINAL_TEST_US, FLOW_ID
from mars_titan.models.titans.mechanism_catalog import MechanismAction

MAX_SOURCE_BYTES = 64 * 1024**2
MAX_REVIEW_BYTES = 1024**2
MAX_REVIEW_ROWS = 8192
MAX_INTEGER = 2**63 - 1
KIND = "mature_native_review_v1"
REVIEW_FIELDS = {
    "schema_version",
    "kind",
    "review_at",
    "parent_consumer_id",
    "window_sha256",
    "events_sha256",
    "contract_id",
    "through_generation",
    "collection_generations",
    "native_records",
    "rows",
    "mature_rows",
    "conditional_evidence",
    "selection_performed",
    "deployment_performed",
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
    "requested",
    "emitted",
    "emitted_quantiles",
    "status",
    "target",
    "available_at",
    "native_received_at",
    "receipt_generation",
    "native_record_sha256",
    "emitted_point_loss",
    "reference_point_loss",
    "emitted_point_gain",
    "candidate_point_losses",
    "candidate_point_gains",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identifier(value, name):
    require(
        type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{name} necesita una huella SHA-256",
    )
    return value


def encoded(value, maximum, *, pretty=False):
    """Acotar el JSON durante su codificación, antes de materializarlo completo."""
    encoder = json.JSONEncoder(
        sort_keys=not pretty,
        separators=None if pretty else (",", ":"),
        indent=2 if pretty else None,
        allow_nan=False,
    )
    parts, size = [], int(pretty)
    try:
        for part in encoder.iterencode(value):
            size += len(part.encode())
            require(size <= maximum, "El contenido JSON supera su presupuesto de bytes")
            parts.append(part)
    except (TypeError, OverflowError, RecursionError) as error:
        raise ValueError("El contenido no conserva un JSON finito y acotado") from error
    return "".join(parts) + ("\n" if pretty else "")


def fingerprint(value, maximum=MAX_REVIEW_BYTES):
    return hashlib.sha256(encoded(value, maximum).encode()).hexdigest()


def _stamp(window, review_at, expected):
    require(type(window) is dict and set(window) == {"collections", "as_of"}, "Falta la ventana")
    stamp, collections = window["as_of"], window["collections"]
    fields = {
        "schema_version",
        "kind",
        "contract_id",
        "model_id",
        "collection_generations",
        "through_generation",
        "before",
        "native_records",
        "source_bytes",
        "window_sha256",
    }
    require(type(stamp) is dict and set(stamp) == fields, "La ventana cambió su esquema")
    require(
        type(stamp["schema_version"]) is int
        and stamp["schema_version"] == 1
        and stamp["kind"] == "native_mechanism_window_v1"
        and type(stamp["before"]) is int
        and stamp["before"] == review_at,
        "La ventana no corresponde al corte exclusivo de revisión",
    )
    for name in ("contract_id", "model_id", "window_sha256"):
        identifier(stamp[name], name)
    bounded_integer(stamp["source_bytes"], "bytes de fuente", 1, MAX_SOURCE_BYTES)
    require(type(collections) is list and 1 <= len(collections) <= 256, "Colecciones sin límite")
    metadata = []
    for item in collections:
        require(
            type(item) is dict
            and set(item)
            == {
                "generation",
                "collection",
                "supervision",
                "censored",
                "resolutions",
                "packet_reference",
            },
            "Una colección histórica cambió su esquema",
        )
        require(type(item["collection"]) is dict, "Falta el paquete de colección")
        bounded_integer(item["generation"], "generación de colección", 1, MAX_INTEGER)
        metadata.append(
            dict(
                collection_sha256=item["collection"].get("sha256"),
                **{key: value for key, value in item.items() if key != "collection"},
            )
        )
    body = dict(
        collections=metadata,
        as_of={key: value for key, value in stamp.items() if key != "window_sha256"},
    )
    require(
        stamp["window_sha256"] == expected == fingerprint(body, MAX_SOURCE_BYTES),
        "La huella de ventana no corresponde a la esperada",
    )
    return stamp, collections


def _native_records(stamp, review_at):
    generations = stamp["collection_generations"]
    require(
        type(generations) is list
        and generations
        and len(generations) <= 256
        and all(type(value) is int and value > 0 for value in generations)
        and generations == sorted(set(generations)),
        "Las generaciones de colección no son únicas y ordenadas",
    )
    bounded_integer(stamp["through_generation"], "fin de prefijo", generations[-1], MAX_INTEGER)
    count = stamp["through_generation"] - generations[0] + 1
    records = stamp["native_records"]
    require(type(records) is list and 1 <= count == len(records) <= 256, "Prefijo sin límite")
    previous, result, used = 0, {}, 0
    for generation, record in enumerate(records, generations[0]):
        require(
            type(record) is dict and set(record) == {"generation", "cutoff", "sha256", "bytes"},
            "El registro nativo cambió su esquema",
        )
        bounded_integer(record["generation"], "generación nativa", generation, generation)
        bounded_integer(record["cutoff"], "recepción nativa", max(1, previous), review_at - 1)
        bounded_integer(record["bytes"], "bytes nativos", 1, MAX_SOURCE_BYTES)
        identifier(record["sha256"], "El registro nativo")
        previous = record["cutoff"]
        used += record["bytes"]
        require(used <= stamp["source_bytes"], "El prefijo contradice sus bytes de fuente")
        result[generation] = record
    return result


def _decision_events(events, stamp, records, expected):
    require(type(events) in (tuple, list) and 1 <= len(events) <= 256, "Eventos sin límite")
    require(
        fingerprint(events, MAX_SOURCE_BYTES) == expected,
        "La huella de los eventos no corresponde a la esperada",
    )
    result = {}
    for event in events:
        require(type(event) is dict and set(event) == EVENT_FIELDS, "El evento cambió su esquema")
        generation = event["generation"]
        require(type(generation) is int and generation in records, "Evento fuera del prefijo")
        require(generation not in result, "El prefijo repite un evento de decisión")
        native = records[generation]
        require(
            type(event["schema_version"]) is int
            and event["schema_version"] == 1
            and event["kind"] == "decision"
            and event["model_id"] == stamp["model_id"]
            and type(event["cutoff"]) is int
            and event["cutoff"] == native["cutoff"]
            and event["native_record_sha256"] == native["sha256"]
            and type(event["payload"]) is dict,
            "El evento no enlaza su decisión nativa",
        )
        result[generation] = event
    require(set(result) == set(stamp["collection_generations"]), "Faltan decisiones del prefijo")
    return result


def _receipts(item, outcomes, records):
    links = item["resolutions"]
    require(type(links) is list and len(links) <= len(outcomes), "Recepciones sin límite")
    result = {}
    for link in links:
        require(
            type(link) is dict
            and set(link)
            == {
                "generation",
                "row_indices",
                "native_record_sha256",
            },
            "La recepción cambió su esquema",
        )
        generation = link["generation"]
        require(type(generation) is int and generation in records, "Recepción fuera del prefijo")
        native = records[generation]
        require(link["native_record_sha256"] == native["sha256"], "La recepción cambió de huella")
        indices = link["row_indices"]
        require(type(indices) is list and indices, "La recepción no identifica filas")
        for index in indices:
            bounded_integer(index, "fila recibida", 0, len(outcomes) - 1)
            require(
                index not in result and outcomes[index] is not None, "Recepción repetida o vacía"
            )
            result[index] = native
    require(
        set(result) == {index for index, target in enumerate(outcomes) if target is not None},
        "Los resultados no conservan todas sus recepciones nativas",
    )
    return result


def _censored(item, outcomes, records, received):
    require(type(item["censored"]) is list, "Falta el estado de censura")
    result = {}
    for row in item["censored"]:
        require(
            type(row) is dict
            and set(row)
            == {
                "row_index",
                "generation",
                "closed_at",
                "native_record_sha256",
            },
            "La censura cambió su esquema",
        )
        index, generation = row["row_index"], row["generation"]
        bounded_integer(index, "fila censurada", 0, len(outcomes) - 1)
        require(
            type(generation) is int
            and generation in records
            and index not in received
            and index not in result
            and outcomes[index] is None,
            "La censura repite o contradice un resultado",
        )
        native = records[generation]
        require(
            row["closed_at"] == native["cutoff"]
            and row["native_record_sha256"] == native["sha256"],
            "La censura no conserva su cierre nativo",
        )
        result[index] = native
    return result


def _loss(prediction, target):
    result = abs(prediction - target)
    require(math.isfinite(result), "La pérdida puntual madura no es finita")
    return result


def _quantiles(values):
    require(
        type(values) is list
        and len(values) == 5
        and all(type(value) is float and math.isfinite(value) for value in values)
        and values == sorted(values),
        "La emisión necesita cinco cuantiles nativos finitos y ordenados",
    )


def _row(metadata, emitted, target, receipt, censored, group, local, observation, review_at):
    quantiles = emitted["quantiles"]
    _quantiles(quantiles)
    require(
        emitted["point"] == quantiles[2],
        "El punto emitido no conserva la mediana nativa exacta",
    )
    result = dict(
        **metadata,
        snapshot_sha256=observation["snapshot_sha256"],
        bank_entries=observation["bank_entries"],
        requested=emitted["requested"],
        emitted=group["anchor"],
        emitted_quantiles=list(quantiles),
        status="censored" if censored is not None else "pending",
        target=None,
        available_at=None,
        native_received_at=None,
        receipt_generation=None,
        native_record_sha256=None,
        emitted_point_loss=None,
        reference_point_loss=None,
        emitted_point_gain=None,
        candidate_point_losses=[None] * 12,
        candidate_point_gains=[None] * 12,
    )
    native = receipt if receipt is not None else censored
    if native is not None:
        result.update(
            native_received_at=native["cutoff"],
            receipt_generation=native["generation"],
            native_record_sha256=native["sha256"],
        )
    if target is None:
        return result
    result.update(status=target["status"], available_at=target["available_at"])
    require(target["available_at"] <= native["cutoff"], "La recepción precede a la disponibilidad")
    if target["status"] != "label":
        return result
    bounded_integer(metadata["matures_at"], "madurez", metadata["prediction_at"] + 1, review_at - 1)
    bounded_integer(target["available_at"], "disponibilidad", metadata["matures_at"], review_at - 1)
    value = target["value"]
    result.update(target=value, emitted_point_loss=_loss(quantiles[2], value))
    for index, candidate in enumerate(group["candidates"][::3]):
        if candidate["status"] == "observed":
            result["candidate_point_losses"][index] = _loss(candidate["quantiles"][local][2], value)
    reference = result["candidate_point_losses"][0]
    result["reference_point_loss"] = reference
    if reference is not None:
        gain = reference - result["emitted_point_loss"]
        require(math.isfinite(gain), "La ganancia puntual madura no es finita")
        result["emitted_point_gain"] = gain
        result["candidate_point_gains"] = [
            None if loss is None else reference - loss for loss in result["candidate_point_losses"]
        ]
        require(
            all(gain is None or math.isfinite(gain) for gain in result["candidate_point_gains"]),
            "Una ganancia condicional madura no es finita",
        )
    return result


def _collection_rows(item, event, records, parent, review_at):
    packet = CollectedMechanisms.restore(item["collection"])
    body, control = packet.payload(), event["payload"]
    observation = body["observation"]["observation"]
    require(
        observation["consumer_id"] == parent == control["consumer_id"]
        and observation["cutoff"] == event["cutoff"],
        "La colección pertenece a otro padre o decisión",
    )
    NativeMechanismCollections.check_anchor(packet, control)
    targets = MechanismSupervision.restore(packet, item["supervision"]).record()["outcomes"]
    received = _receipts(item, targets, records)
    censored = _censored(item, targets, records, received)
    native_rows = {row["sample_id"]: row for row in control["rows"]}
    groups = {
        group["market"]: dict(group, anchor=anchor["applied"])
        for group, anchor in zip(body["groups"], body["anchor"], strict=True)
    }
    positions = dict.fromkeys(groups, 0)
    result = []
    for index, metadata in enumerate(body["rows"]):
        market = metadata["flow_id"].split("/", 1)[0]
        bounded_integer(metadata["prediction_at"], "decisión emitida", 1, review_at - 1)
        result.append(
            _row(
                metadata,
                native_rows[metadata["sample_id"]],
                targets[index],
                received.get(index),
                censored.get(index),
                groups[market],
                positions[market],
                observation,
                review_at,
            )
        )
        positions[market] += 1
    return result


def _review_loss(row):
    losses, gains = row["candidate_point_losses"], row["candidate_point_gains"]
    require(
        type(losses) is list and type(gains) is list and len(losses) == len(gains) == 12,
        "Faltan las pérdidas condicionales de la revisión",
    )
    require(
        all(
            value is None or (type(value) is float and math.isfinite(value) and value >= 0)
            for value in losses
        ),
        "Las pérdidas condicionales no son finitas o desconocidas",
    )
    if row["status"] != "label":
        require(
            row["target"]
            is row["emitted_point_loss"]
            is row["reference_point_loss"]
            is row["emitted_point_gain"]
            is None
            and all(value is None for value in (*losses, *gains)),
            "Una fila sin etiqueta no puede declarar pérdida o ganancia",
        )
        return
    target = row["target"]
    require(type(target) is float and math.isfinite(target), "La etiqueta debe ser finita")
    loss = _loss(row["emitted_quantiles"][2], target)
    require(
        type(row["emitted_point_loss"]) is float
        and row["emitted_point_loss"] == loss
        and (
            row["reference_point_loss"] is None
            if losses[0] is None
            else type(row["reference_point_loss"]) is float
            and row["reference_point_loss"] == losses[0]
        ),
        "La pérdida no corresponde a la mediana emitida y la referencia declarada",
    )
    reference = losses[0]
    expected = None if reference is None else reference - loss
    require(
        (
            row["emitted_point_gain"] is None
            if expected is None
            else type(row["emitted_point_gain"]) is float and row["emitted_point_gain"] == expected
        ),
        "La ganancia emitida no conserva su signo",
    )
    for candidate, gain in zip(losses, gains, strict=True):
        expected = None if reference is None or candidate is None else reference - candidate
        require(
            (
                gain is None
                if expected is None
                else type(gain) is float and math.isfinite(gain) and gain == expected
            ),
            "Una ganancia condicional contradice su pérdida",
        )


def _review_row(row, records, cutoffs, review_at):
    require(type(row) is dict and set(row) == ROW_FIELDS, "La fila de revisión cambió su esquema")
    require(
        type(row["flow_id"]) is str
        and FLOW_ID.fullmatch(row["flow_id"]) is not None
        and row["prediction_at"] in cutoffs
        and row["sample_id"] == f"{row['flow_id']}/{row['prediction_at']}",
        "La fila de revisión no identifica una decisión del prefijo",
    )
    bounded_integer(row["prediction_at"], "decisión revisada", 1, review_at - 1)
    bounded_integer(row["input_available_at"], "entrada revisada", 0, row["prediction_at"])
    identifier(row["input_sha256"], "Los inputs")
    bounded_integer(row["bank_entries"], "episodios de la revisión", 0, 1024)
    if row["snapshot_sha256"] is None:
        require(row["bank_entries"] == 0, "Falta el snapshot del banco no vacío")
    else:
        identifier(row["snapshot_sha256"], "El snapshot de la revisión")
    for name in ("requested", "emitted"):
        require(
            type(row[name]) is str and len(row[name].split("/")) == 3, "Falta la acción revisada"
        )
        MechanismAction(*row[name].split("/"))
    _quantiles(row["emitted_quantiles"])
    status = row["status"]
    require(
        type(status) is str and status in {"label", "pending", "excluded", "censored"},
        "Estado de fila desconocido",
    )
    if row["matures_at"] is not None:
        bounded_integer(
            row["matures_at"], "madurez revisada", row["prediction_at"] + 1, MAX_INTEGER
        )
    if status == "pending":
        require(
            row["available_at"]
            is row["native_received_at"]
            is row["receipt_generation"]
            is row["native_record_sha256"]
            is None,
            "La fila pendiente declara una recepción inexistente",
        )
    else:
        generation = row["receipt_generation"]
        require(type(generation) is int and generation in records, "La fila perdió su recepción")
        native = records[generation]
        require(
            row["native_received_at"] == native["cutoff"]
            and row["native_record_sha256"] == native["sha256"]
            and native["cutoff"] >= row["prediction_at"],
            "La fila contradice su recepción nativa",
        )
        if status == "censored":
            require(row["available_at"] is None, "La censura no es una etiqueta disponible")
        else:
            minimum = row["matures_at"] if status == "label" else row["prediction_at"]
            require(type(minimum) is int, "La etiqueta no tiene madurez explícita")
            bounded_integer(
                row["available_at"], "disponibilidad revisada", minimum, native["cutoff"]
            )
    _review_loss(row)


def _validate_review(body):
    require(type(body) is dict and set(body) == REVIEW_FIELDS, "La revisión cambió su esquema")
    require(
        type(body["schema_version"]) is int
        and body["schema_version"] == 1
        and body["kind"] == KIND
        and body["selection_performed"] is body["deployment_performed"] is False
        and body["conditional_evidence"]
        == "issued_trajectory_with_anchored_conditional_alternatives",
        "La revisión no conserva su alcance de evidencia",
    )
    bounded_integer(body["review_at"], "corte de revisión", 1, FINAL_TEST_US)
    for name in ("parent_consumer_id", "window_sha256", "events_sha256", "contract_id"):
        identifier(body[name], name)
    records = _native_records(dict(body, source_bytes=MAX_SOURCE_BYTES), body["review_at"])
    values = body["rows"]
    require(
        type(values) is list and 1 <= len(values) <= MAX_REVIEW_ROWS, "Filas de revisión sin límite"
    )
    cutoffs = {records[generation]["cutoff"] for generation in body["collection_generations"]}
    samples, snapshots = set(), {}
    for row in values:
        _review_row(row, records, cutoffs, body["review_at"])
        require(row["sample_id"] not in samples, "La revisión repite una muestra")
        samples.add(row["sample_id"])
        before = snapshots.setdefault(row["prediction_at"], row["snapshot_sha256"])
        require(before == row["snapshot_sha256"], "Una decisión cambió de snapshot entre filas")
    count = sum(row["status"] == "label" for row in values)
    bounded_integer(body["mature_rows"], "filas maduras", count, count)


@dataclass(frozen=True, init=False)
class MatureReview:
    """Copia sellada de una señal, sin afirmar autenticación externa de sus fuentes."""

    _json: str
    _identity: str

    def __init__(self, body):
        _validate_review(body)
        content = encoded(body, MAX_REVIEW_BYTES)
        object.__setattr__(self, "_json", content)
        object.__setattr__(self, "_identity", hashlib.sha256(content.encode()).hexdigest())

    def record(self):
        require(
            hashlib.sha256(self._json.encode()).hexdigest() == self._identity,
            "La revisión no conserva su sello",
        )
        body = json.loads(self._json)
        _validate_review(body)
        return dict(body, review_id=self._identity)

    @property
    def review_id(self):
        self.record()
        return self._identity


def build_mature_review(
    window,
    decision_events,
    *,
    review_at,
    parent_consumer_id,
    expected_window_sha256,
    expected_events_sha256,
):
    """Revisar un prefijo anterior al corte, sin predictor, solver o selección."""
    bounded_integer(review_at, "corte de revisión", 1, FINAL_TEST_US)
    parent = identifier(parent_consumer_id, "El padre")
    identifier(expected_window_sha256, "La ventana esperada")
    identifier(expected_events_sha256, "Los eventos esperados")
    stamp, collections = _stamp(window, review_at, expected_window_sha256)
    records = _native_records(stamp, review_at)
    events = _decision_events(decision_events, stamp, records, expected_events_sha256)
    require(
        [item["generation"] for item in collections] == stamp["collection_generations"],
        "Las colecciones no conservan sus generaciones ordenadas",
    )
    rows, samples = [], set()
    for item in collections:
        values = _collection_rows(item, events[item["generation"]], records, parent, review_at)
        require(len(rows) + len(values) <= MAX_REVIEW_ROWS, "La señal supera sus filas")
        for row in values:
            require(row["sample_id"] not in samples, "La señal repite una emisión")
            samples.add(row["sample_id"])
        rows.extend(values)
    return MatureReview(
        dict(
            schema_version=1,
            kind=KIND,
            review_at=review_at,
            parent_consumer_id=parent,
            window_sha256=expected_window_sha256,
            events_sha256=expected_events_sha256,
            contract_id=stamp["contract_id"],
            through_generation=stamp["through_generation"],
            collection_generations=list(stamp["collection_generations"]),
            native_records=list(stamp["native_records"]),
            rows=rows,
            mature_rows=sum(row["status"] == "label" for row in rows),
            conditional_evidence="issued_trajectory_with_anchored_conditional_alternatives",
            selection_performed=False,
            deployment_performed=False,
        )
    )
