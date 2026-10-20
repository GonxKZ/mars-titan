"""Propuestas aisladas de PT2 con resultados parciales de las cohortes emitidas.

La publicación y la verificación de las evidencias externas corresponden al ejecutor
de sesión. Este componente comprueba identidades, formatos, tiempos y presupuestos.
No calcula pérdidas de etiquetas ausentes ni garantiza cobertura de una selección.
"""

import copy
from dataclasses import dataclass

import numpy as np

from .cohort_feedback_state import (
    KIND,
    MAX_ROWS,
    POLICY,
    identifier,
    limits,
    packed_calibrator,
    restore_payload,
    seal,
    validate_feedback,
)
from .online_conformal import OnlineConformal, _instant, _require


@dataclass(frozen=True)
class CohortOutcome:
    group: str
    prediction_at: int
    sample_id: str
    available_at: int
    value: float | None = None
    status: str = "label"
    evidence_sha256: str | None = None


@dataclass(frozen=True)
class EmissionProposal:
    state: "CohortFeedback"
    quantiles: np.ndarray
    emission: dict


@dataclass(frozen=True)
class FeedbackProposal:
    state: "CohortFeedback"
    settlements: tuple[dict, ...]


class CohortFeedback:
    """Snapshots de calibración en CPU. emit y feedback devuelven propuestas nuevas."""

    def __init__(self, calibrator, cohorts, *, max_rows, max_bytes, clock, cancelled_cohorts=0):
        self._calibrator, self._cohorts = calibrator, cohorts
        self._max_rows, self._max_bytes = max_rows, max_bytes
        self._clock, self._cancelled = clock, cancelled_cohorts

    @classmethod
    def start(cls, calibrator, *, max_rows=MAX_ROWS, max_bytes=8 * 1024**2):
        limits(max_rows, max_bytes)
        _require(type(calibrator) is OnlineConformal, "Se necesita OnlineConformal identificado")
        payload = packed_calibrator(calibrator)
        _require(
            all(not market["pending"] for market in payload["markets"].values()),
            "No se puede iniciar sin las identidades de cohortes ya pendientes",
        )
        clocks = [
            market["clock"] for market in payload["markets"].values() if market["clock"] is not None
        ]
        state = cls(
            OnlineConformal.restore(calibrator.export()),
            [],
            max_rows=max_rows,
            max_bytes=max_bytes,
            clock=max(clocks) if clocks else None,
        )
        return cls.restore(state.export())

    @property
    def pending_rows(self):
        return sum(len(item["rows"]) for item in self._cohorts)

    def corrections(self, group):
        return self._calibrator.corrections(group)

    def _fork(self):
        return type(self)(
            OnlineConformal.restore(self._calibrator.export()),
            copy.deepcopy(self._cohorts),
            max_rows=self._max_rows,
            max_bytes=self._max_bytes,
            clock=self._clock,
            cancelled_cohorts=self._cancelled,
        )

    def _time(self, at):
        at = _instant(at, "El corte del feedback")
        _require(
            self._clock is None or at >= self._clock, "El reloj del feedback no puede retroceder"
        )
        return at

    def emit(self, group, prediction_at, sample_ids, quantiles, matures_at):
        """Proponer una cohorte completa, ordenada por identidad, sin leer resultados."""
        prediction_at = self._time(prediction_at)
        identifier(group)
        _require(
            type(sample_ids) in (tuple, list)
            and type(matures_at) in (tuple, list)
            and 0 < len(sample_ids) == len(matures_at)
            and len(sample_ids) + self.pending_rows <= self._max_rows,
            "La cohorte no conserva sus filas o supera el presupuesto pendiente",
        )
        names = [identifier(name) for name in sample_ids]
        _require(len(set(names)) == len(names), "Una cohorte no puede repetir una identidad")
        due = [_instant(value, "La madurez de fila") for value in matures_at]
        _require(
            all(value > prediction_at for value in due),
            "La etiqueta debe madurar después de emitir",
        )
        _require(
            isinstance(quantiles, np.ndarray)
            and quantiles.shape == (len(names), 5)
            and quantiles.dtype in (np.float32, np.float64),
            "La emisión necesita una matriz NumPy de cinco cuantiles y precisión explícita",
        )
        order = sorted(range(len(names)), key=names.__getitem__)
        proposal = self._fork()
        calibrated, emission = proposal._calibrator.emit(
            group, prediction_at, quantiles[order], max(due)
        )
        proposal._cohorts.append(
            dict(
                group=group,
                prediction_at=prediction_at,
                rows=[
                    dict(
                        sample_id=names[i],
                        matures_at=due[i],
                        available_at=None,
                        value=None,
                        status=None,
                        evidence_sha256=None,
                    )
                    for i in order
                ],
            )
        )
        proposal._cohorts.sort(key=lambda item: (item["prediction_at"], item["group"]))
        proposal._clock = prediction_at
        proposal.export()
        aligned = np.empty_like(calibrated)
        aligned[order] = calibrated
        return EmissionProposal(proposal, aligned, emission)

    def feedback(self, outcomes, *, at):
        """Proponer recepción parcial y resolver cohortes en orden, sin publicar cambios.

        Las exclusiones necesitan una huella cuya autenticidad valida el ejecutor.
        Una cohorte excluida se retira entera tras su madurez, aunque falten otras filas.
        Los resultados posteriores de esas filas siguen siendo responsabilidad de las
        métricas del ejecutor y ya no se envían a esta cola de calibración.
        """
        at = self._time(at)
        _require(
            type(outcomes) in (tuple, list) and len(outcomes) <= self.pending_rows,
            "El lote de resultados supera las filas pendientes",
        )
        proposal = self._fork()
        positions = {
            (item["group"], item["prediction_at"], row["sample_id"]): row
            for item in proposal._cohorts
            for row in item["rows"]
        }
        for outcome in outcomes:
            _require(type(outcome) is CohortOutcome, "Se necesita un resultado de cohorte tipado")
            key = (
                identifier(outcome.group),
                _instant(outcome.prediction_at, "La decisión"),
                identifier(outcome.sample_id),
            )
            _require(key in positions, "El resultado no pertenece a una cohorte pendiente")
            row = positions[key]
            _require(row["available_at"] is None, "El resultado ya se recibió para esta fila")
            candidate = dict(
                row,
                available_at=outcome.available_at,
                value=outcome.value,
                status=outcome.status,
                evidence_sha256=outcome.evidence_sha256,
            )
            candidate["value"] = validate_feedback(candidate, key[1], at)
            candidate["available_at"] = int(outcome.available_at)
            row.update(candidate)
        proposal._clock = at
        settlements = proposal._drain(at)
        proposal.export()
        return FeedbackProposal(proposal, tuple(settlements))

    def _drain(self, at):
        retained, settlements, blocked = [], [], set()
        for item in self._cohorts:
            group, prediction_at, rows = item["group"], item["prediction_at"], item["rows"]
            exclusions = [row for row in rows if row["status"] in ("excluded", "finalized")]
            unresolved = [row["sample_id"] for row in rows if row["available_at"] is None]
            if (
                group in blocked
                or max(row["matures_at"] for row in rows) > at
                or (unresolved and not exclusions)
            ):
                blocked.add(group)
                retained.append(item)
                continue
            if exclusions:
                record = self._calibrator.cancel(group, prediction_at, at=at)
                self._cancelled += 1
            else:
                record = dict(
                    self._calibrator.mature(
                        group, prediction_at, np.array([row["value"] for row in rows]), at=at
                    ),
                    status="matured",
                    rows=len(rows),
                )
            labels = [row for row in rows if row["status"] == "label"]
            settlements.append(
                dict(
                    record,
                    labels=copy.deepcopy(labels),
                    exclusions=copy.deepcopy(exclusions),
                    unresolved_ids=unresolved,
                )
            )
        self._cohorts = retained
        return settlements

    def export(self):
        """Estado JSON acotado y versionado, sin compartir matrices ni metadatos."""
        return seal(
            dict(
                schema_version=1,
                kind=KIND,
                policy=POLICY,
                max_rows=self._max_rows,
                max_bytes=self._max_bytes,
                clock=self._clock,
                cancelled_cohorts=self._cancelled,
                calibrator=packed_calibrator(self._calibrator),
                cohorts=copy.deepcopy(self._cohorts),
            )
        )

    def state_sha256(self):
        return self.export()["state_sha256"]

    @classmethod
    def restore(cls, payload):
        calibrator = restore_payload(payload)
        return cls(
            calibrator,
            copy.deepcopy(payload["cohorts"]),
            max_rows=payload["max_rows"],
            max_bytes=payload["max_bytes"],
            clock=payload["clock"],
            cancelled_cohorts=payload["cancelled_cohorts"],
        )
