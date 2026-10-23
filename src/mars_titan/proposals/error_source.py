"""Marco completo de una fase cerrada, sin ejecutar ni reconstruir el predictor."""

import hashlib
import json
from dataclasses import asdict, dataclass

from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.models.titans.config import bounded_integer
from mars_titan.models.titans.financial_inputs import FLOW_ID
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.proposals.error_rule import ErrorProposalPlan
from mars_titan.proposals.mature_review import (
    MAX_REVIEW_BYTES,
    MAX_REVIEW_ROWS,
    MAX_SOURCE_BYTES,
    MatureReview,
    build_mature_review,
    encoded,
    fingerprint,
    identifier,
    require,
)

EVIDENCE_FIELDS = {
    "schema_version",
    "kind",
    "plan_id",
    "registry_id",
    "review_id",
    "parent_consumer_id",
    "source_contract_id",
    "phase",
    "source_head",
    "all_events_sha256",
    "decision_generations",
    "logical_source_bytes",
    "frame",
}


def _evidence_phase(value, review):
    require(
        type(value) is dict
        and set(value)
        == {"partition", "warmup_start", "decision_start", "decision_end", "close_at"},
        "La fase de procedencia cambió su esquema",
    )
    require(
        type(value["partition"]) is str and value["partition"] in ("train", "validation"),
        "La evidencia necesita una fase de entrenamiento o validación",
    )
    phase = FinancialPhase(**value)
    records = {record["generation"]: record for record in review["native_records"]}
    require(
        phase.close_at < review["review_at"]
        and phase.close_at == review["native_records"][-1]["cutoff"]
        and all(
            phase.decision_start <= records[generation]["cutoff"] < phase.decision_end
            for generation in review["collection_generations"]
        )
        and all(
            phase.decision_start <= row["prediction_at"] < phase.decision_end
            for row in review["rows"]
        ),
        "La fase no conserva las decisiones o el cierre de la revisión",
    )


def _evidence_flows(values):
    require(
        type(values) is list
        and len(values) <= MAX_REVIEW_ROWS
        and all(type(flow) is str and FLOW_ID.fullmatch(flow) is not None for flow in values)
        and values == sorted(set(values)),
        "El marco necesita flujos únicos, ordenados y acotados",
    )
    return values


def _evidence_frame(value, review):
    require(
        type(value) is dict
        and set(value) == {"sample_ids", "sha256", "observed_flow_ids", "absent_flow_ids"},
        "El marco de procedencia cambió su esquema",
    )
    samples = sorted(row["sample_id"] for row in review["rows"])
    require(
        type(value["sample_ids"]) is list
        and len(value["sample_ids"]) == len(samples)
        and all(type(sample) is str for sample in value["sample_ids"])
        and value["sample_ids"] == samples,
        "El marco no contiene exactamente las emisiones revisadas",
    )
    identifier(value["sha256"], "La huella del marco")
    require(value["sha256"] == fingerprint(samples), "La huella del marco cambió sus muestras")
    observed = _evidence_flows(value["observed_flow_ids"])
    absent = _evidence_flows(value["absent_flow_ids"])
    require(
        observed == sorted({row["flow_id"] for row in review["rows"]})
        and not set(observed).intersection(absent)
        and len(observed) + len(absent) <= MAX_REVIEW_ROWS,
        "El marco contradice sus flujos observados o ausentes",
    )


def _validate_evidence(review, metadata):
    """Contrastar coherencia interna, sin autenticar las fuentes del coordinador."""
    require(type(review) is MatureReview, "Falta una revisión madura tipada")
    require(
        type(metadata) is dict and set(metadata) == EVIDENCE_FIELDS,
        "La procedencia de la evidencia cambió su esquema",
    )
    require(
        type(metadata["schema_version"]) is int
        and metadata["schema_version"] == 1
        and type(metadata["kind"]) is str
        and metadata["kind"] == "closed_parent_error_evidence_v1",
        "La evidencia no conserva su tipo o versión",
    )
    for name in (
        "plan_id",
        "registry_id",
        "review_id",
        "parent_consumer_id",
        "source_contract_id",
        "all_events_sha256",
    ):
        identifier(metadata[name], name)
    body = review.record()
    require(
        metadata["review_id"] == body["review_id"]
        and metadata["parent_consumer_id"] == body["parent_consumer_id"]
        and metadata["source_contract_id"] == body["contract_id"],
        "La procedencia perdió su revisión, padre o contrato",
    )
    _evidence_phase(metadata["phase"], body)
    head = metadata["source_head"]
    require(
        type(head) is dict and set(head) == {"generation", "record_sha256"},
        "La cabeza de procedencia cambió su esquema",
    )
    bounded_integer(
        head["generation"],
        "generación de cabeza",
        body["through_generation"],
        body["through_generation"],
    )
    identifier(head["record_sha256"], "La cabeza nativa")
    require(
        head["record_sha256"] == body["native_records"][-1]["sha256"],
        "La cabeza no corresponde al último registro de la revisión",
    )
    generations = metadata["decision_generations"]
    require(
        type(generations) is list
        and len(generations) == len(body["collection_generations"])
        and all(type(generation) is int for generation in generations)
        and generations == body["collection_generations"],
        "La procedencia cambió las generaciones de decisión",
    )
    bounded_integer(
        metadata["logical_source_bytes"], "bytes lógicos de fuente", 1, MAX_SOURCE_BYTES
    )
    native_bytes = sum(record["bytes"] for record in body["native_records"])
    require(
        metadata["logical_source_bytes"] >= native_bytes,
        "La procedencia declara menos bytes que su prefijo nativo",
    )
    _evidence_frame(metadata["frame"], body)


@dataclass(frozen=True, init=False)
class ErrorProposalEvidence:
    """Revisión y procedencia selladas. No autentica un archivo local falsificado entero."""

    review: MatureReview
    _json: str
    _sha256: str

    def __init__(self, review, metadata):
        _validate_evidence(review, metadata)
        content = encoded(metadata, MAX_REVIEW_BYTES)
        object.__setattr__(self, "review", review)
        object.__setattr__(self, "_json", content)
        object.__setattr__(self, "_sha256", hashlib.sha256(content.encode()).hexdigest())

    def record(self):
        require(type(self._json) is str, "La evidencia perdió su contenido JSON")
        identifier(self._sha256, "El sello de evidencia")
        content = self._json.encode()
        require(len(content) <= MAX_REVIEW_BYTES, "La evidencia supera sus bytes")
        require(
            hashlib.sha256(content).hexdigest() == self._sha256,
            "La evidencia no conserva su sello",
        )
        result = json.loads(self._json)
        _validate_evidence(self.review, result)
        return dict(result, evidence_id=self._sha256)

    @property
    def plan_id(self):
        return self.record()["plan_id"]

    @property
    def registry_id(self):
        return self.record()["registry_id"]


def _parent(session, registry, plan):
    require(type(session) is FinancialSession, "La fuente necesita una FinancialSession real")
    require(type(session.consumer) is FrozenMechanismConsumer, "Falta el consumidor de mecanismos")
    consumer = session.consumer
    consumer.verify(strong=True)
    identity = consumer.identity()
    require(
        session.contract_id == plan.source_contract_id
        and consumer.model_id == plan.parent_consumer_id == fingerprint(identity)
        and session.phase.partition == plan.source_partition
        and session.phase.decision_start == plan.generation_start
        and session.phase.close_at < plan.generation_end,
        "La fuente no corresponde al contrato, padre, partición o cortes declarados",
    )
    require(
        identity["fixed_controls"] is True
        and "selector" not in identity
        and identity["calibration"] is None
        and "local_control" not in identity["predictor"],
        "El registro inicial no representa selector, calibración o control local",
    )
    baseline = registry.resolve(registry.baseline_id).identity()
    reader = identity["readout"]
    require(
        baseline["predictor"] == identity["predictor"]["configuration"]
        and baseline["readout"] == (None if reader is None else reader["configuration"])
        and baseline["dtype"] == identity["predictor"]["dtype"]
        and (reader is None or reader["dtype"] == baseline["dtype"])
        and baseline["action"] == identity["action"]
        and baseline["catalog"] == identity["catalog"],
        "La receta de referencia no corresponde al consumidor real",
    )
    return consumer


def _complete_events(session, snapshot, plan):
    count = snapshot["cursor"]
    require(
        type(count) is int and 1 <= count <= plan.max_source_records,
        "La fuente supera el límite de generaciones o está vacía",
    )
    events, decisions, samples, used = [], [], [], 0
    for generation in range(1, count + 1):
        event = session.mechanism_event(generation)
        used += len(encoded(event, plan.max_source_bytes).encode())
        require(used <= plan.max_source_bytes, "Los eventos exceden los bytes de fuente")
        require(
            event["cutoff"] < plan.generation_end, "La fuente contiene un evento posterior al corte"
        )
        events.append(event)
        if event["kind"] == "decision" and event["payload"]["rows"]:
            require(plan.generation_start <= event["cutoff"], "La decisión precede al tramo")
            decisions.append(event)
            samples.extend(row["sample_id"] for row in event["payload"]["rows"])
    require(
        bool(decisions) and len(samples) == len(set(samples)),
        "El marco está vacío o repite emisiones",
    )
    return events, decisions, samples, used


def prepare_error_review(session, *, registry, plan):
    """Leer todas las emisiones de la fase cerrada y conservar sus recepciones reales."""
    require(type(plan) is ErrorProposalPlan, "Falta el plan tipado de generación")
    plan.check_registry(registry)
    consumer = _parent(session, registry, plan)
    before = session.snapshot()
    require(before["phase_closed"] is True, "La fase de generación debe estar cerrada")
    events, decisions, samples, used = _complete_events(session, before, plan)
    require(used < plan.max_source_bytes, "No quedan bytes de fuente para leer las colecciones")
    generations = tuple(event["generation"] for event in decisions)
    window = session.mechanism_collections_at(
        generations,
        through_generation=before["cursor"],
        before=plan.generation_end,
        max_records=plan.max_source_records,
        max_bytes=plan.max_source_bytes - used,
    )
    used += window["as_of"]["source_bytes"]
    require(used <= plan.max_source_bytes, "La revisión excede los bytes acumulados de fuente")
    review = build_mature_review(
        window,
        decisions,
        review_at=plan.generation_end,
        parent_consumer_id=plan.parent_consumer_id,
        expected_window_sha256=window["as_of"]["window_sha256"],
        expected_events_sha256=fingerprint(decisions, plan.max_source_bytes),
    )
    rows = review.record()["rows"]
    require(
        sorted(samples) == sorted(row["sample_id"] for row in rows),
        "La revisión omite o añade emisiones del marco completo",
    )
    observed = sorted({row["flow_id"] for row in rows})
    require(set(observed) <= set(plan.flow_ids), "La fuente contiene flujos fuera del universo")
    consumer.verify(strong=True)
    require(session.snapshot() == before, "La fuente cambió durante la revisión")
    frame = dict(
        sample_ids=sorted(samples),
        sha256=fingerprint(sorted(samples)),
        observed_flow_ids=observed,
        absent_flow_ids=[flow for flow in plan.flow_ids if flow not in observed],
    )
    metadata = dict(
        schema_version=1,
        kind="closed_parent_error_evidence_v1",
        plan_id=plan.plan_id,
        registry_id=registry.registry_id,
        review_id=review.review_id,
        parent_consumer_id=plan.parent_consumer_id,
        source_contract_id=session.contract_id,
        phase=asdict(session.phase),
        source_head=dict(generation=before["cursor"], record_sha256=before["record_sha256"]),
        all_events_sha256=fingerprint(events, plan.max_source_bytes),
        decision_generations=list(generations),
        logical_source_bytes=used,
        frame=frame,
    )
    return ErrorProposalEvidence(review, metadata)
