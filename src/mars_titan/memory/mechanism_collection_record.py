"""Colección inmutable de predicciones alternativas, sin etiquetas ni publicación propia."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from mars_titan.calibration.online_conformal import _instant, _require
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.episodic_snapshot import _device, _digest_id
from mars_titan.models.titans.financial_inputs import FLOW_ID
from mars_titan.models.titans.mechanism_catalog import (
    MechanismAction,
    MechanismAvailability,
    MechanismCatalog,
)
from mars_titan.models.titans.mechanism_observation import MechanismObservation
from mars_titan.models.titans.mechanism_policy import ACTIONS
from mars_titan.models.titans.mechanism_selection import _calibration_failed, digest

KIND = "conditional_mechanism_collection_v1"
PROTOCOL = "one_market_deviation_from_frozen_anchor_v1"
MAX_BYTES = 48 * 1024**2
ROW_FIELDS = {"flow_id", "sample_id", "prediction_at", "input_available_at", "input_sha256"}
CALIBRATION_WORK = {
    "online_attempts",
    "online_failures",
    "online_created_proposals",
    "online_selected_proposals",
    "online_discarded_proposals",
    "static_cohorts",
}
WORK = {
    "core_observations",
    "mac_online_observations",
    "reader_refinements",
    "reader_searches",
    "reader_value_projections",
    "context_passes",
    "shared_route_batches",
    "individual_route_batches",
    "calibration_attempts",
    "calibration_created_proposals",
    "calibration_discarded_proposals",
}


def action(key):
    _require(type(key) is str and key in ACTIONS, "La acción no pertenece al catálogo")
    return MechanismAction(*key.split("/"))


def matrix(values, size):
    _require(
        type(values) is list
        and len(values) == size
        and all(type(row) is list and len(row) == 5 for row in values)
        and all(type(value) is float for row in values for value in row),
        "Una alternativa necesita cinco cuantiles por fila",
    )
    array = np.asarray(values, dtype=np.float64)
    _require(
        np.isfinite(array).all() and (array[:, 1:] >= array[:, :-1]).all(),
        "Los cuantiles alternativos deben ser finitos y ordenados",
    )
    return array


def _candidate(item, index, *, catalog, available, market, size, anchors):
    _require(
        type(item) is dict
        and set(item)
        == {
            "requested",
            "emitted",
            "status",
            "reasons",
            "quantiles",
            "joint_actions",
            "calibration_work",
            "cross_returns",
        },
        "La alternativa no conserva su esquema",
    )
    requested = action(item["requested"])
    _require(requested.key == ACTIONS[index], "Las alternativas cambian de orden")
    resolved = catalog.resolve(requested, available, rows=size)
    if resolved.reasons:
        _require(
            item["status"] == "unavailable"
            and item["reasons"] == list(resolved.reasons)
            and item["quantiles"] is None
            and item["emitted"] is None
            and item["joint_actions"] is None
            and item["calibration_work"] is None
            and item["cross_returns"] == [],
            "Una acción no admisible no tiene predicción ni pérdida atribuible",
        )
        return None
    joint = {name: (requested.key if name == market else key) for name, key in anchors.items()}
    _require(item["joint_actions"] == joint, "La alternativa cambia otra acción ancla")
    work = item["calibration_work"]
    _require(type(work) is dict and set(work) == CALIBRATION_WORK, "Falta el trabajo de PT2")
    failed = _calibration_failed(work, [action(key) for key in joint.values()])
    _require(
        type(work["static_cohorts"]) is int
        and work["static_cohorts"]
        == sum(action(key).calibration == "static" for key in joint.values()),
        "El trabajo de calibración estática no concilia",
    )
    fallback = failed and requested.calibration == "online"
    emitted = MechanismAction(requested.context, "k1", "none") if fallback else requested
    _require(
        item["status"] == ("fallback" if fallback else "observed")
        and item["emitted"] == emitted.key
        and item["reasons"] == (["pending_calibration_bytes"] if fallback else []),
        "La colección confunde la acción solicitada con la emitida",
    )
    cross = [
        dict(
            market=name,
            requested=key,
            emitted=MechanismAction(action(key).context, "k1", "none").key,
            reasons=["pending_calibration_bytes"],
        )
        for name, key in joint.items()
        if name != market and failed and action(key).calibration == "online"
    ]
    _require(item["cross_returns"] == cross, "La colección oculta retornos de otros mercados")
    return matrix(item["quantiles"], size)


def _work(body, catalog):
    work = body["work"]
    _require(
        type(work) is dict
        and set(work) == WORK
        and all(type(v) is int and v >= 0 for v in work.values()),
        "La colección necesita contadores enteros de trabajo",
    )
    groups = {g["market"]: g for g in body["groups"]}
    parent = {
        market: not catalog.resolve(
            MechanismAction("parent"),
            MechanismAvailability(**group["availability"]),
            rows=sum(r["flow_id"].startswith(market + "/") for r in body["rows"]),
        ).reasons
        for market, group in groups.items()
    }
    observations, passes = 0, 0
    for block in body["blocks"]:
        count = 1 + int(any(parent[flow.split("/", 1)[0]] for flow in block["flow_ids"]))
        observations += count * len(block["flow_ids"])
        passes += count
    _require(
        work["core_observations"] == observations
        and work["context_passes"] == passes
        and work["mac_online_observations"] in (0, observations)
        and work["reader_refinements"] <= 34 * len(body["rows"])
        and work["reader_searches"] <= 18 * len(body["rows"])
        and work["shared_route_batches"] <= passes
        and work["individual_route_batches"] <= 6 * passes,
        "El trabajo no corresponde a los bloques y contextos ejecutados",
    )
    if catalog.reader_contract == "shared_evidence_bridge_v1":
        populated = body["observation"]["observation"]["bank_entries"] > 0
        _require(
            work["reader_value_projections"] == (work["reader_refinements"] if populated else 0),
            "La colección regional debe contar las proyecciones de cada lectura",
        )
    for target, source in (
        ("calibration_attempts", "online_attempts"),
        ("calibration_created_proposals", "online_created_proposals"),
        ("calibration_discarded_proposals", "online_discarded_proposals"),
    ):
        total = sum(
            c["calibration_work"][source]
            for g in groups.values()
            for c in g["candidates"]
            if c["calibration_work"] is not None
        )
        _require(work[target] == total, "La colección omite trabajo de propuestas PT2")


def validate(body):
    _require(
        type(body) is dict
        and set(body)
        == {
            "schema_version",
            "kind",
            "protocol",
            "output_dtype",
            "execution_device",
            "observation",
            "catalog",
            "rows",
            "anchor",
            "prior_state_sha256",
            "producer_sha256",
            "blocks",
            "groups",
            "work",
        },
        "La colección no conserva su esquema completo",
    )
    _require(
        type(body["schema_version"]) is int
        and body["schema_version"] == 1
        and body["kind"] == KIND
        and body["protocol"] == PROTOCOL,
        "La colección pertenece a otro protocolo",
    )
    observed = MechanismObservation.restore(body["observation"]).record()["observation"]
    _require(
        body["output_dtype"] in ("torch.float32", "torch.float64")
        and type(body["execution_device"]) is str
        and str(_device(body["execution_device"])) == body["execution_device"],
        "La colección necesita precisión y dispositivo explícitos",
    )
    for key in ("prior_state_sha256", "producer_sha256"):
        _digest_id(body[key], key)
    catalog = MechanismCatalog.restore(body["catalog"])
    rows = body["rows"]
    _require(type(rows) is list and 1 <= len(rows) <= 8192, "La colección necesita filas acotadas")
    for row in rows:
        _require(
            type(row) is dict and set(row) == ROW_FIELDS | {"matures_at"}, "Falta metadata de fila"
        )
        _require(
            type(row["flow_id"]) is str and FLOW_ID.fullmatch(row["flow_id"]) is not None,
            "La colección necesita un activo identificado",
        )
        _require(
            type(row["sample_id"]) is str and 0 < len(row["sample_id"].encode()) <= 128,
            "La colección necesita una muestra identificada",
        )
        _require(
            row["sample_id"] == f"{row['flow_id']}/{observed['cutoff']}",
            "La muestra no identifica el activo y el corte",
        )
        _digest_id(row["input_sha256"], "Los inputs")
        _require(
            row["prediction_at"] == observed["cutoff"]
            and _instant(row["input_available_at"], "La entrada") <= observed["cutoff"],
            "Una fila no estaba disponible en el corte",
        )
        if row["matures_at"] is not None:
            _require(
                _instant(row["matures_at"], "La madurez") > observed["cutoff"],
                "La madurez no puede preceder a la emisión",
            )
    flows = [r["flow_id"] for r in rows]
    _require(
        flows == sorted(set(flows))
        and digest(flows) == observed["flow_ids_sha256"]
        and digest([{k: r[k] for k in sorted(ROW_FIELDS)} for r in rows])
        == observed["rows_sha256"],
        "La colección no contiene exactamente las observaciones selladas",
    )
    markets = [g["market"] for g in observed["groups"]]
    _require(type(body["anchor"]) is list and len(body["anchor"]) == len(markets), "Falta el ancla")
    anchors = {}
    for source, market in zip(body["anchor"], markets, strict=True):
        _require(
            type(source) is dict
            and set(source) == {"market", "requested", "applied", "reasons"}
            and source["market"] == market,
            "El ancla no conserva sus mercados",
        )
        action(source["requested"])
        anchors[market] = action(source["applied"]).key
    _require(
        type(body["groups"]) is list and len(body["groups"]) == len(markets), "Falta una cohorte"
    )
    for group, market, original in zip(body["groups"], markets, observed["groups"], strict=True):
        _require(
            type(group) is dict
            and set(group) == {"market", "availability", "candidates"}
            and group["market"] == market
            and type(group["candidates"]) is list
            and len(group["candidates"]) == 36,
            "La cohorte no conserva las 36 alternativas",
        )
        available = MechanismAvailability(**group["availability"])
        anchor = catalog.resolve(
            action(body["anchor"][markets.index(market)]["requested"]),
            available,
            rows=original["rows"],
        )
        _require(
            anchor.applied.key == anchors[market]
            and list(anchor.reasons) == body["anchor"][markets.index(market)]["reasons"],
            "El ancla contradice la disponibilidad",
        )
        count = sum(flow.startswith(market + "/") for flow in flows)
        _require(count == original["rows"], "La cohorte no conserva sus activos")
        medians = {}
        for index, candidate in enumerate(group["candidates"]):
            values = _candidate(
                candidate,
                index,
                catalog=catalog,
                available=available,
                market=market,
                size=count,
                anchors=anchors,
            )
            if values is not None and candidate["status"] == "observed":
                key = index // 3
                if key in medians:
                    _require(
                        np.array_equal(values[:, 2], medians[key]),
                        "Una calibración cambió la mediana de su contexto y ruta",
                    )
                medians[key] = values[:, 2]
        for candidate in group["candidates"]:
            if candidate["status"] == "fallback":
                baseline = group["candidates"][
                    0 if candidate["requested"].startswith("off/") else 18
                ]
                _require(
                    baseline["status"] == "observed"
                    and candidate["quantiles"] == baseline["quantiles"],
                    "El retorno no conserva los cuantiles K1 de su contexto",
                )
    _require(
        type(body["blocks"]) is list and 1 <= len(body["blocks"]) <= len(rows), "Faltan bloques"
    )
    seen = []
    for block in body["blocks"]:
        _require(
            type(block) is dict
            and set(block) == {"flow_ids", "fast_state_sha256"}
            and type(block["flow_ids"]) is list
            and block["flow_ids"],
            "Bloque incompleto",
        )
        _digest_id(block["fast_state_sha256"], "El estado rápido anterior")
        seen.extend(block["flow_ids"])
    _require(
        sorted(seen) == flows and len(set(seen)) == len(seen), "Los bloques repiten u omiten filas"
    )
    _work(body, catalog)


@dataclass(frozen=True, init=False)
class CollectedMechanisms:
    """Bytes canónicos acotados, compatibles con un artefacto sellado de SessionArtifacts."""

    _bytes: bytes
    sha256: str

    def __init__(self, body):
        validate(body)
        encoded = canonical(body).encode()
        _require(len(encoded) <= MAX_BYTES, "La colección supera 48 MiB")
        object.__setattr__(self, "_bytes", encoded)
        object.__setattr__(self, "sha256", hashlib.sha256(encoded).hexdigest())

    def payload(self):
        return json.loads(self._bytes)

    def record(self):
        return dict(payload_utf8=self._bytes, sha256=self.sha256)

    @classmethod
    def restore(cls, record):
        _require(
            type(record) is dict
            and set(record) == {"payload_utf8", "sha256"}
            and type(record["payload_utf8"]) is bytes
            and len(record["payload_utf8"]) <= MAX_BYTES,
            "Registro de colección inválido",
        )
        _require(
            hashlib.sha256(record["payload_utf8"]).hexdigest() == record["sha256"],
            "La huella de la colección cambió",
        )
        value = cls(json.loads(record["payload_utf8"]))
        _require(
            value._bytes == record["payload_utf8"], "La colección no conserva codificación canónica"
        )
        return value
