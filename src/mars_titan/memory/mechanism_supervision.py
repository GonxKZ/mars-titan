"""Recepción parcial de etiquetas maduras y pérdidas condicionales, sin ajuste."""

import json
from dataclasses import dataclass

import torch

from mars_titan.calibration.cohort_feedback import CohortOutcome
from mars_titan.calibration.cohort_feedback_state import validate_feedback
from mars_titan.calibration.online_conformal import _instant, _require
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.episodic_snapshot import _device
from mars_titan.models.titans.mechanism_selection import digest
from mars_titan.models.titans.state import check_finite, deferred_checks

from .mechanism_collection_record import CollectedMechanisms

KIND = "conditional_mechanism_supervision_v1"
MAX_BYTES = 8 * 1024**2


def _nonnegative_mean(value, dims):
    """Promediar magnitudes no negativas sin desbordar una suma finita evitable."""
    scale = value.amax(dims, keepdim=True)
    divisor = torch.where(scale > 0, scale, torch.ones_like(scale))
    return ((value / divisor).mean(dims, keepdim=True) * scale).squeeze(dims)


def _validate(collection, clock, outcomes):
    _require(
        type(collection) is CollectedMechanisms, "La supervisión necesita una colección sellada"
    )
    rows = collection.payload()["rows"]
    _require(type(outcomes) is list and len(outcomes) == len(rows), "Faltan estados de resultados")
    if clock is not None:
        _require(
            _instant(clock, "La recepción") >= rows[0]["prediction_at"],
            "El reloj precede a la colección",
        )
    normalized = []
    for row, target in zip(rows, outcomes, strict=True):
        if target is None:
            normalized.append(None)
            continue
        _require(
            type(target) is dict
            and set(target) == {"available_at", "value", "status", "evidence_sha256"}
            and clock is not None,
            "Resultado sin esquema o reloj",
        )
        if target["status"] == "label":
            _require(row["matures_at"] is not None, "Falta una madurez explícita de la etiqueta")
        value = validate_feedback(
            dict(target, matures_at=row["matures_at"]), row["prediction_at"], clock
        )
        normalized.append(dict(target, value=value, available_at=int(target["available_at"])))
    return normalized


@dataclass(frozen=True, init=False)
class MechanismSupervision:
    """Estado nuevo por recepción. Una exclusión no se convierte en una pérdida cero."""

    collection: CollectedMechanisms
    _json: str

    def __init__(self, collection, *, clock, outcomes):
        outcomes = _validate(collection, clock, outcomes)
        body = dict(
            schema_version=1,
            kind=KIND,
            collection_sha256=collection.sha256,
            clock=clock,
            outcomes=outcomes,
        )
        encoded = canonical(body)
        _require(len(encoded.encode()) <= MAX_BYTES, "La supervisión supera su presupuesto")
        object.__setattr__(self, "collection", collection)
        object.__setattr__(self, "_json", encoded)

    @classmethod
    def start(cls, collection):
        _require(type(collection) is CollectedMechanisms, "Falta la colección de predicciones")
        return cls(collection, clock=None, outcomes=[None] * len(collection.payload()["rows"]))

    def record(self):
        body = json.loads(self._json)
        return dict(body, state_sha256=digest(body))

    @classmethod
    def restore(cls, collection, record):
        _require(
            type(record) is dict
            and set(record)
            == {"schema_version", "kind", "collection_sha256", "clock", "outcomes", "state_sha256"},
            "La supervisión no conserva todos sus campos",
        )
        body = {key: value for key, value in record.items() if key != "state_sha256"}
        _require(
            type(record["schema_version"]) is int
            and record["schema_version"] == 1
            and record["kind"] == KIND
            and record["collection_sha256"] == collection.sha256
            and record["state_sha256"] == digest(body),
            "La supervisión pertenece a otra colección o cambió",
        )
        return cls(collection, clock=record["clock"], outcomes=record["outcomes"])

    def receive(self, outcomes, *, at):
        body = json.loads(self._json)
        at = _instant(at, "La recepción")
        _require(body["clock"] is None or at >= body["clock"], "El reloj no puede retroceder")
        rows = self.collection.payload()["rows"]
        _require(
            type(outcomes) in (tuple, list) and len(outcomes) <= len(rows), "Resultados sin límite"
        )
        expected = {
            (row["flow_id"].split("/", 1)[0], row["prediction_at"], row["sample_id"]): i
            for i, row in enumerate(rows)
        }
        updated = body["outcomes"]
        for outcome in outcomes:
            _require(type(outcome) is CohortOutcome, "Se necesita un resultado tipado de cohorte")
            key = outcome.group, outcome.prediction_at, outcome.sample_id
            _require(key in expected, "El resultado pertenece a otra predicción")
            index = expected[key]
            _require(updated[index] is None, "El resultado de esta predicción ya se recibió")
            updated[index] = dict(
                available_at=outcome.available_at,
                value=outcome.value,
                status=outcome.status,
                evidence_sha256=outcome.evidence_sha256,
            )
        return MechanismSupervision(self.collection, clock=at, outcomes=updated)

    @torch.no_grad()
    def metrics(self, *, device):
        """MAE e interval score por cohorte, sobre las mismas filas maduras disponibles.

        Usa FP64 y un dispositivo explícito. No ajusta parámetros ni pondera mercados
        o fechas como réplicas independientes. La agregación temporal es externa.
        """
        device = _device(device)
        body, labels = self.collection.payload(), json.loads(self._json)["outcomes"]
        result = []
        for group in body["groups"]:
            indices = [
                i
                for i, row in enumerate(body["rows"])
                if row["flow_id"].startswith(group["market"] + "/")
            ]
            valid = [
                local
                for local, index in enumerate(indices)
                if labels[index] is not None and labels[index]["status"] == "label"
            ]
            metric = dict(
                collection_sha256=self.collection.sha256,
                market=group["market"],
                prediction_at=body["observation"]["observation"]["cutoff"],
                unit="market_decision_cohort_mean_over_mature_rows",
                dtype="torch.float64",
                mature_rows=len(valid),
                pending_rows=sum(labels[i] is None for i in indices),
                excluded_rows=sum(
                    labels[i] is not None and labels[i]["status"] != "label" for i in indices
                ),
                mature_sample_ids=[body["rows"][indices[j]]["sample_id"] for j in valid],
                point_losses=[None] * 12,
                point_gains=[None] * 12,
                interval_losses=[None] * 36,
                interval_gains=[None] * 36,
                coverage=[None] * 36,
                widths=[None] * 36,
                candidate_status=[c["status"] for c in group["candidates"]],
            )
            if valid:
                observed = [
                    i for i, c in enumerate(group["candidates"]) if c["status"] == "observed"
                ]
                q = torch.tensor(
                    [[group["candidates"][i]["quantiles"][j] for j in valid] for i in observed],
                    dtype=torch.float64,
                    device=device,
                )
                y = torch.tensor(
                    [labels[indices[j]]["value"] for j in valid], dtype=torch.float64, device=device
                )
                with deferred_checks():
                    point = (q[..., 2] - y).abs()
                    lower, upper = q[..., [1, 0]], q[..., [3, 4]]
                    widths = upper - lower
                    score = widths + q.new_tensor([10.0, 40.0]) * (
                        (lower - y[None, :, None]).clamp_min(0)
                        + (y[None, :, None] - upper).clamp_min(0)
                    )
                    coverage = ((lower <= y[None, :, None]) & (y[None, :, None] <= upper)).double()
                    check_finite(point, "Los errores puntuales maduros")
                    check_finite(score, "Los interval scores maduros")
                    values = torch.cat(
                        (
                            _nonnegative_mean(point, (1,))[:, None],
                            _nonnegative_mean(score, (1, 2))[:, None],
                            coverage.mean(1),
                            _nonnegative_mean(widths, (1,)),
                        ),
                        dim=1,
                    )
                    check_finite(values, "Las pérdidas condicionales maduras")
                packed = values.cpu().tolist()
                baseline = packed[observed.index(0)]
                for index, values in zip(observed, packed, strict=True):
                    if index % 3 == 0:
                        metric["point_losses"][index // 3] = values[0]
                        metric["point_gains"][index // 3] = baseline[0] - values[0]
                    metric["interval_losses"][index] = values[1]
                    metric["interval_gains"][index] = baseline[1] - values[1]
                    metric["coverage"][index], metric["widths"][index] = values[2:4], values[4:6]
            result.append(metric)
        return result
