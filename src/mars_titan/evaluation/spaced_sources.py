"""Comprobar informes y condiciones compartidas de los controles E2 completados."""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import outside_source, sha256
from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.training.spaced_window import KIND, _digest, _verify_outputs

PARTITIONS = ("validation", "calibration", "evaluation")
WINDOW_KINDS = dict(standard=KIND, regional="spaced_regional_utility_frozen_window_v1")
_REQUEST = {
    "schema_version",
    "kind",
    "arm",
    "device",
    "view_sha256",
    "anchor_run_sha256",
    "anchor_selected_sha256",
    "source_manifest_sha256",
    "study_sha256",
    "study",
    "protocol_sha256",
    "code",
    "native_sha256",
}
_IDENTITY = {
    "schema_version",
    "kind",
    "request",
    "selected",
    "phases",
    "indices",
    "expected_rows",
    "retention",
    "evidence_enabled",
    "state_transfer",
    "state_reset",
    "interrupted_phase",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _phase(value, partition):
    _require(
        type(value) is dict
        and set(value)
        == {"partition", "warmup_start", "decision_start", "decision_end", "close_at"}
        and value["partition"] == partition
        and all(
            type(value[k]) is int
            for k in ("warmup_start", "decision_start", "decision_end", "close_at")
        )
        and 0
        <= value["warmup_start"]
        <= value["decision_start"]
        < value["decision_end"]
        <= value["close_at"]
        <= FINAL_TEST_START_US,
        "Una fase E2 no conserva sus fronteras de desarrollo",
    )


def _selected(value, request, contract):
    _require(
        type(value) is dict
        and type(value.get("schema_version")) is int
        and value["schema_version"] == 1
        and value.get("kind")
        == (
            "spaced_selected_regional_reader_v1"
            if contract == "regional"
            else "spaced_selected_reader_v1"
        )
        and value.get("state_transfer") == "shared_parameters_only"
        and value.get("reader_contract", "standard") == contract
        and value.get("view_sha256") == request["view_sha256"]
        and type(value.get("seed")) is int
        and 0 <= value["seed"] < 2**32
        and all(_sha(value.get(k)) for k in ("consumer_id", "variant_sha256", "readout_sha256"))
        and type(value.get("parent")) is dict
        and all(_sha(value["parent"].get(k)) for k in ("run_id", "run_sha256", "parameters_sha256"))
        and type(value.get("anchor")) is dict
        and _sha(value["anchor"].get("run_id"))
        and value["anchor"].get("run_sha256") == request["anchor_run_sha256"]
        and value["anchor"].get("selected_sha256") == request["anchor_selected_sha256"],
        "El lector E2 no conserva su selección y procedencia",
    )


def _header(document, study, digest, arm):
    kind = WINDOW_KINDS[study.reader_contract]
    _require(
        type(document) is dict
        and type(document.get("schema_version")) is int
        and document["schema_version"] == 1
        and document.get("kind") == kind
        and document.get("status") == "completed"
        and document.get("final_test_opened") is False,
        "Se necesitan ventanas E2 completadas de desarrollo",
    )
    request, identity = document.get("request"), document.get("identity")
    _require(
        type(request) is dict
        and set(request) == _REQUEST
        and type(request["schema_version"]) is int
        and request["schema_version"] == 1
        and request["kind"] == kind
        and request["arm"] == arm
        and request["device"] in ("cpu", "cuda:0")
        and request["study_sha256"] == digest
        and request["study"] == study.identity()
        and all(
            _sha(request[k])
            for k in (
                "view_sha256",
                "anchor_run_sha256",
                "anchor_selected_sha256",
                "source_manifest_sha256",
                "protocol_sha256",
                "native_sha256",
            )
        )
        and type(request["code"]) is dict
        and bool(request["code"])
        and all(type(k) is str and _sha(v) for k, v in request["code"].items()),
        "La declaración del estudio E2 no coincide con la ventana",
    )
    _require(
        type(identity) is dict
        and set(identity)
        == _IDENTITY | ({"reader_contract"} if study.reader_contract == "regional" else set())
        and type(identity["schema_version"]) is int
        and identity["schema_version"] == 1
        and identity["kind"] == kind
        and identity["request"] == request
        and identity.get("reader_contract", "standard") == study.reader_contract
        and identity["state_transfer"] == "selected_shared_parameters_only"
        and identity["state_reset"] == "each_phase_empty_fast_memory_bank_and_evidence"
        and identity["interrupted_phase"] == "repeat_from_start"
        and identity["evidence_enabled"] is (arm != "baseline")
        and document.get("run_id") == _digest(identity),
        "La identidad o el estado inicial E2 no corresponden al informe",
    )
    _selected(identity["selected"], request, study.reader_contract)
    for key in ("phases", "indices", "expected_rows"):
        _require(
            type(identity[key]) is dict and set(identity[key]) == set(PARTITIONS),
            "La ventana E2 pierde fases o índices",
        )
    for name in PARTITIONS:
        _phase(identity["phases"][name], name)
        _require(_sha(identity["indices"][name]), "La identidad del índice E2 no es válida")
        rows = identity["expected_rows"][name]
        _require(
            type(rows) is int and 0 < rows <= study.max_phase_rows,
            "Las filas E2 exceden el presupuesto o no están declaradas",
        )
    retention = study.retention_for(arm)
    if retention is None:
        retention = RetentionConfig(
            capacity=study.retention.capacity,
            seed=study.retention.seed,
            frontier=min(8, study.retention.capacity),
        )
    _require(
        identity["retention"] == asdict(retention),
        "La política E2 no corresponde al control declarado",
    )
    shared = {
        k: v for k, v in identity.items() if k not in ("request", "retention", "evidence_enabled")
    }
    shared["request"] = {k: v for k, v in request.items() if k != "arm"}
    return shared


@dataclass(frozen=True)
class SpacedRunSource:
    arm: str
    report_path: Path
    report: dict
    report_sha256: str
    prediction_path: Path
    record: dict
    phase: dict

    def provenance(self):
        return dict(
            run_id=self.report["run_id"],
            report_sha256=self.report_sha256,
            prediction=self.record,
            identity=self.report["identity"],
        )

    def verify(self):
        _require(
            sha256(self.report_path) == self.report_sha256,
            "El informe E2 cambió durante el análisis",
        )
        _verify_outputs(self.report_path.parent, self.report)
        _require(
            self.prediction_path.stat().st_size == self.record["bytes"],
            "El tamaño de las predicciones E2 ha cambiado",
        )


def load_sources(reports, study, study_sha256, partition, output):
    """Validar todos los controles antes de puntuar, sin cargar modelos ni estados."""
    _require(
        type(reports) is dict and set(reports) == set(study.arms),
        "Faltan controles E2 o se añadieron otros",
    )
    sources, common = [], None
    for arm in study.arms:
        path = Path(reports[arm])
        safe_destination(path)
        outside_source(path.parent, output)
        document, digest = read_manifest(path, maximum=16 * 1024**2)
        shared = _header(document, study, study_sha256, arm)
        _require(
            common is None or shared == common,
            "Los controles E2 no comparten las mismas condiciones",
        )
        common = shared
        _verify_outputs(path.parent, document)
        record = document["predictions"][partition]
        _require(
            type(record.get("rows")) is int
            and record["rows"] == document["identity"]["expected_rows"][partition]
            and type(record.get("bytes")) is int
            and record["bytes"] >= 12
            and type(record.get("metrics")) is dict,
            "Las predicciones E2 no concilian con las filas esperadas",
        )
        predictions = path.parent / record["path"]
        _require(
            predictions.is_file() and predictions.stat().st_size == record["bytes"],
            "El análisis necesita la tabla original de predicciones E2 y su tamaño correcto",
        )
        sources.append(
            SpacedRunSource(
                arm,
                path,
                document,
                digest,
                predictions,
                record,
                document["identity"]["phases"][partition],
            )
        )
    return sources
