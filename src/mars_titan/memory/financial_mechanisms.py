"""Enlace de propuestas de mecanismos con la generación financiera existente."""

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from mars_titan.calibration import conformal_quantiles as cqr
from mars_titan.calibration.cohort_feedback import CohortOutcome
from mars_titan.models.quantile_head import LEVELS
from mars_titan.models.titans.config import bounded_integer, canonical
from mars_titan.models.titans.financial_inputs import FINAL_TEST_US
from mars_titan.models.titans.mechanism_selection import check_trace

from . import mechanism_history, mechanism_observations
from .financial_consumers import TitansBinding
from .mechanism_history import MechanismHistory, MechanismHistoryConfig

KIND = "titans_mechanisms_v1"
FIELDS = ("mechanisms", "mechanism_history")
TRACE_FIELDS = (
    "context_id",
    "consumer_id",
    "parameter_id",
    "snapshot_sha256",
    "levels",
    "calibration_record",
    "rows",
    "work",
)
PAYLOAD_FIELDS = set(TRACE_FIELDS) | {
    "auxiliary_sha256",
    "feedback_watermark",
    "received_prediction_ids",
    "settlements",
    "phase_closed",
}


def _sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _prediction(value):
    return dict(
        id=value.id,
        asset=value.asset,
        task=value.task.name,
        horizon=value.task.horizon,
        generation=value.generation,
        decision_at=value.decision_at,
        value=value.value,
    )


class MechanismBinding(TitansBinding):
    """M0/M1 con contexto y calibración confirmados por el mismo ejecutor nativo."""

    kind = KIND

    def __init__(self, consumer, codec, retention, admission):
        if admission not in {"m0", "m1"}:
            raise ValueError("El enlace de mecanismos requiere admisión M0 o M1")
        super().__init__(consumer, codec, retention, admission)
        control = consumer.predictor.local_control
        if control is not None and control.config.mode != "disabled":
            raise ValueError("La combinación durable con control local C aún no está comprobada")
        self.max_batch = consumer.max_batch

    def code(self):
        result = dict(
            financial_mechanisms=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            mechanism_history=hashlib.sha256(
                Path(mechanism_history.__file__).read_bytes()
            ).hexdigest(),
        )
        if self.consumer.policy is not None:
            result["mechanism_observations"] = hashlib.sha256(
                Path(mechanism_observations.__file__).read_bytes()
            ).hexdigest()
        return result

    def _trace_fields(self):
        return TRACE_FIELDS + (("selection",) if self.consumer.policy is not None else ())

    @staticmethod
    def configuration(config):
        config = MechanismHistoryConfig() if config is None else config
        if type(config) is not MechanismHistoryConfig:
            raise ValueError("El archivo de mecanismos necesita su configuración identificada")
        return config

    def initialize(self, session, config):
        self.history = MechanismHistory(
            session.artifacts,
            identity=session.contract_id,
            model_id=session.model_id,
            config=config,
        )

    def restore(self, session, bundle):
        return (
            self.consumer.initial_state()
            if bundle is None
            else self.consumer.restore_state(session._read(bundle["mechanisms"], "mechanisms"))
        )

    def previous(self, session, bundle):
        return dict(
            mechanisms=session._stage(
                self.consumer.export_state(self.restore(session, bundle)), "mechanisms"
            )
            if bundle is None
            else bundle["mechanisms"],
            mechanism_history=self.history.empty()
            if bundle is None
            else bundle["mechanism_history"],
        )

    def settlement_control(self):
        result = dict(
            observations=0,
            mac_updates=0,
            refinements=0,
            reevaluations=0,
            group_estimated_bytes=0,
            context_id=None,
            consumer_id=self.consumer.model_id,
            parameter_id=self.consumer.predictor._parameter_id,
            snapshot_sha256=None,
            levels=[],
            calibration_record=None,
            rows=[],
            work=dict(
                core_observations=0,
                mac_online_observations=0,
                reader_refinements=0,
                reader_searches=0,
                context_updates_proposed=0,
            ),
        )
        if self.consumer.policy is not None:
            result["selection"] = None
        return result

    def prepare_event(
        self, session, rows, fast, snapshot, *, context_id, warmup, batch_rows, auxiliary
    ):
        flows = tuple(row.flow_id for row in rows)
        maturity = (
            None
            if warmup
            else tuple(
                session.prefixes.expected_maturity(row.flow_id, row.prediction_at) for row in rows
            )
        )
        selection = {}
        if self.consumer.policy is not None and not warmup:
            selection["observation"] = mechanism_observations.extract_observation(
                rows,
                consumer=self.consumer,
                state=auxiliary,
                flow_ids=flows,
                cutoff=rows[0].prediction_at,
                context_id=context_id,
                snapshot=snapshot,
            )
        event = self.consumer.begin(
            auxiliary,
            flow_ids=flows,
            cutoff=rows[0].prediction_at,
            context_id=context_id,
            matures_at=maturity,
            snapshot=snapshot,
            warmup=warmup,
            **selection,
        )
        model = self.consumer.predictor
        for start in range(0, len(rows), batch_rows):
            batch = session._decision_batch(rows[start : start + batch_rows])
            unknown = tuple(flow for flow in batch.flow_ids if flow not in fast["references"])
            initial = model.initial_state(unknown) if unknown else None
            previous = session._fast_store.gather(fast, batch.flow_ids, initial_state=initial)
            prepared = self.consumer.prepare(batch, previous, event)
            fast = session._fast_store.replace(fast, previous, prepared.next_state)
            event = prepared.event
        emission = self.consumer.finish(event)
        metadata = {row.flow_id: row.metadata() for row in rows}
        due = dict(zip(flows, maturity, strict=True)) if maturity is not None else {}
        records = []
        for index, row in enumerate(emission.rows):
            record = json.loads(canonical(dict(row)))
            record.update(metadata[row["flow_id"]])
            record["matures_at"] = due.get(row["flow_id"])
            record["quantiles"] = (
                emission.quantiles[index].tolist() if emission.quantiles is not None else None
            )
            records.append(record)
        calibration = None
        selected = {row["flow_id"].split("/", 1)[0]: row["calibration"] for row in records}
        if any(mode != "none" for mode in selected.values()):
            calibration = json.loads(canonical(self.consumer._static))
            calibration["groups"] = {
                group: value
                for group, value in calibration["groups"].items()
                if selected.get(group, "none") != "none"
            }
            for group, mode in selected.items():
                if mode == "online":
                    calibration["groups"][group] = dict(
                        intervals={
                            key: dict(correction=value)
                            for key, value in auxiliary.calibration.corrections(group).items()
                        }
                    )
        work = dict(emission.work)
        control = dict(
            observations=len(rows),
            mac_updates=work["mac_online_observations"],
            refinements=work["reader_refinements"],
            reevaluations=0,
            group_estimated_bytes=0,
            context_id=context_id,
            consumer_id=self.consumer.model_id,
            parameter_id=model._parameter_id,
            snapshot_sha256=snapshot._digest if snapshot is not None else None,
            levels=list(LEVELS) if emission.quantiles is not None else [],
            calibration_record=calibration,
            rows=records,
            work=work,
            auxiliary=self.consumer.export_state(emission.state),
        )
        if self.consumer.policy is not None:
            control["selection"] = (
                json.loads(emission.selection_json) if emission.selection_json is not None else None
            )
        aligned = (
            dict(zip(emission.flow_ids, emission.point_predictions, strict=True))
            if not warmup
            else {}
        )
        return [float(aligned[flow]) for flow in flows] if not warmup else [], fast, control

    def _record(self, session, generation):
        raw = session.native.read_blob(
            str(session.output / f"record-{generation}.json"), 64 * 1024**2
        )
        record = json.loads(raw)
        if (
            type(record) is not dict
            or type(record.get("schema_version")) is not int
            or record["schema_version"] != 2
            or record.get("generation") != generation
            or record.get("prefix_policy_sha256") != session.prefixes.policy_id
        ):
            raise ValueError("El registro nativo no corresponde a la generación de mecanismos")
        return record, hashlib.sha256(raw).hexdigest()

    def _feedback(self, session, auxiliary, metadata, outcomes, excluded, *, cutoff, kind):
        pending = (
            auxiliary.calibration.unresolved_keys()
            if auxiliary.calibration is not None
            else frozenset()
        )
        received, ids = [], []
        for item in (*outcomes, *excluded):
            prediction = item.prediction
            row = metadata[prediction.asset, prediction.decision_at]
            group = prediction.asset.split("/", 1)[0]
            key = group, prediction.decision_at, row["sample_id"]
            if key not in pending:
                continue
            if hasattr(item, "label"):
                result = CohortOutcome(
                    group,
                    prediction.decision_at,
                    row["sample_id"],
                    item.label.available_at,
                    value=item.label.value,
                )
            else:
                result = CohortOutcome(
                    group,
                    prediction.decision_at,
                    row["sample_id"],
                    cutoff,
                    status="excluded",
                    evidence_sha256=item.evidence.evidence_sha256,
                )
            received.append(result)
            ids.append(prediction.id)
        if kind == "warmup":
            if received:
                raise ValueError("El calentamiento no puede recibir feedback de mecanismos")
            return auxiliary, (), ids, None
        # El límite administrativo es exclusivo. No se cambia la fecha de ningún dato.
        watermark = min(cutoff, FINAL_TEST_US - 1)
        if auxiliary.cutoff is None:
            if received:
                raise ValueError("El estado sin emisiones no puede recibir resultados")
            auxiliary = replace(auxiliary, cutoff=watermark)
        auxiliary, settlements = self.consumer.feedback(
            auxiliary, received, at=watermark, settlement=kind == "settlement"
        )
        return auxiliary, settlements, ids, watermark

    def resolve(self, session, bundle, original_rows, outcomes, excluded, finalized):
        control = session._read(bundle["control"], "control")
        auxiliary = self.consumer.restore_state(session._read(bundle["mechanisms"], "mechanisms"))
        record, digest = self._record(session, bundle["generation"])
        before = session._inflight["mechanism_before"]
        if (
            record["identity_sha256"] != before["identity_sha256"]
            or record["previous_sha256"] != before["record_sha256"]
            or record["state_before_sha256"] != _sha(before["state"])
            or record["close_phase"] != session._inflight["close_phase"]
        ):
            raise ValueError("El registro no conserva el prefijo de estado confirmado")
        self._check_record(control, record, cutoff=bundle["cutoff"])
        expected = [
            dict(
                id=item.id,
                prediction=_prediction(item.prediction),
                revision=item.label.revision,
                available_at=item.label.available_at,
                target=item.label.value,
                error=item.label.value - item.prediction.value,
            )
            for item in outcomes
        ]
        if (
            record["feedback"] != expected
            or (
                [
                    (row["prediction"], row["evidence"]["evidence_sha256"])
                    for row in record["excluded"]
                ]
                != [
                    (_prediction(item.prediction), item.evidence.evidence_sha256)
                    for item in excluded
                ]
            )
            or (
                [(row["prediction"], row["closed_at"]) for row in record["finalized"]]
                != [(_prediction(item.prediction), item.closed_at) for item in finalized]
            )
        ):
            raise ValueError("El registro contradice los resultados validados por el nativo")
        metadata = {(row["flow_id"], row["prediction_at"]): row for row in original_rows}
        auxiliary, settlements, received, watermark = self._feedback(
            session,
            auxiliary,
            metadata,
            outcomes,
            excluded,
            cutoff=bundle["cutoff"],
            kind=record["event_kind"],
        )
        if record["close_phase"]:
            auxiliary = self.consumer.close_phase(
                auxiliary, at=bundle["cutoff"], evidence_sha256=digest
            )
        reference = session._stage(self.consumer.export_state(auxiliary), "mechanisms")
        payload = {name: control[name] for name in self._trace_fields()}
        payload.update(
            auxiliary_sha256=reference["sha256"],
            feedback_watermark=watermark,
            received_prediction_ids=received,
            settlements=list(settlements),
            phase_closed=record["close_phase"],
        )
        event = dict(
            schema_version=1,
            model_id=session.model_id,
            generation=bundle["generation"],
            cutoff=bundle["cutoff"],
            kind=record["event_kind"],
            native_record_sha256=digest,
            payload=payload,
        )
        return dict(
            bundle,
            mechanisms=reference,
            mechanism_history=self.history.append(bundle["mechanism_history"], event),
        )

    def _check_record(self, control, record, *, cutoff):
        rows = control["rows"]
        warmup = record["event_kind"] == "warmup"
        if (
            record["cutoff"] != cutoff
            or record["observations_count"] != control["observations"]
            or len(rows) != control["observations"]
            or control["consumer_id"] != self.consumer.model_id
            or control["parameter_id"] != self.consumer.predictor._parameter_id
        ):
            raise ValueError("La traza no concilia con el modelo, corte y observaciones nativas")
        flows = [row["flow_id"] for row in rows]
        if flows != sorted(set(flows)) or len(flows) > 8192:
            raise ValueError("La traza repite o desordena los flujos emitidos")
        predictions = {row["asset"]: row for row in record["predictions"]}
        expected = set() if warmup else set(flows)
        if set(predictions) != expected or len(predictions) != len(record["predictions"]):
            raise ValueError("Las salidas nativas no son las de la traza completa")
        for row in rows:
            if row["prediction_at"] != cutoff or row["input_available_at"] > cutoff:
                raise ValueError("La traza contiene una observación ajena al corte")
            if warmup:
                if row["point"] is not None or row["quantiles"] is not None:
                    raise ValueError("El calentamiento no emite predicciones financieras")
                continue
            point = row["point"]
            native = predictions[row["flow_id"]]
            if (
                type(point) is not float
                or not math.isfinite(point)
                or native["value"] != point
                or native["decision_at"] != cutoff
            ):
                raise ValueError("El punto archivado no es la predicción nativa emitida")
        self._check_quantiles(control)
        if self.consumer.policy is not None:
            trace = control.get("selection")
            if warmup or not rows:
                if trace is not None:
                    raise ValueError("El calentamiento y settlement no eligen mecanismos")
            else:
                check_trace(
                    self.consumer,
                    trace,
                    rows=rows,
                    cutoff=cutoff,
                    context_id=control["context_id"],
                    snapshot_sha256=control["snapshot_sha256"],
                    work=control["work"],
                )
                if (
                    trace["observation"]["observation"]["producer_sha256"]
                    != self.code()["mechanism_observations"]
                ):
                    raise ValueError("La observación procede de otro extractor")

    @staticmethod
    def _check_quantiles(control):
        for row in control["rows"]:
            if row["quantiles"] is None:
                if row["raw_quantiles"] is not None:
                    raise ValueError("Faltan los cuantiles emitidos de una salida con intervalos")
                continue
            for name in ("quantiles", "raw_quantiles", "k1_quantiles"):
                values = np.asarray(row[name], dtype=np.float64)
                if (
                    values.shape != (5,)
                    or not np.isfinite(values).all()
                    or (values[1:] < values[:-1]).any()
                ):
                    raise ValueError("Los cuantiles archivados no son finitos y ordenados")
            if (
                control["levels"] != list(LEVELS)
                or row["quantiles"][2] != row["point"]
                or row["raw_quantiles"][2] != row["point"]
            ):
                raise ValueError("La traza no conserva los niveles o la mediana emitida")
        calibrated = [row for row in control["rows"] if row["calibration"] != "none"]
        if calibrated:
            raw = np.asarray([row["raw_quantiles"] for row in calibrated], dtype=np.float64)
            groups = np.asarray([row["flow_id"].split("/", 1)[0] for row in calibrated])
            emitted, _ = cqr.apply_conformal_quantiles(control["calibration_record"], raw, groups)
            saved = np.asarray([row["quantiles"] for row in calibrated], dtype=np.float64)
            if emitted.tobytes() != saved.tobytes():
                raise ValueError("Los cuantiles no conservan la corrección usada al emitir")
        for row in control["rows"]:
            if row["calibration"] == "none" and row["quantiles"] != row["raw_quantiles"]:
                raise ValueError("La ruta sin calibración cambió sus cuantiles")

    def _check_payload(self, payload, record):
        fields = PAYLOAD_FIELDS | ({"selection"} if self.consumer.policy is not None else set())
        if type(payload) is not dict or set(payload) != fields:
            raise ValueError("El archivo no conserva el esquema de la traza financiera")
        expected_watermark = (
            None if record["event_kind"] == "warmup" else min(record["cutoff"], FINAL_TEST_US - 1)
        )
        ids = payload["received_prediction_ids"]
        native_ids = {row["prediction"]["id"] for row in (*record["feedback"], *record["excluded"])}
        if (
            type(ids) is not list
            or any(type(value) is not str for value in ids)
            or len(ids) != len(set(ids))
            or not set(ids) <= native_ids
            or payload["feedback_watermark"] != expected_watermark
            or type(payload["phase_closed"]) is not bool
            or payload["phase_closed"] != record["close_phase"]
            or type(payload["settlements"]) is not list
        ):
            raise ValueError("El archivo contradice el feedback y el cierre nativos")

    def verify(self, session, bundle, observations):
        auxiliary = self.restore(session, bundle)
        manifest = bundle["mechanism_history"]
        self.history.verify(manifest)
        if (
            manifest["events"] != bundle["generation"]
            or manifest["cutoff"] != bundle["cutoff"]
            or auxiliary.cutoff != bundle["cutoff"]
        ):
            raise ValueError("El archivo y el auxiliar no corresponden a la misma generación")
        control = session._read(bundle["control"], "control")
        event = self.history.read_event(manifest, bundle["generation"])
        record, digest = self._record(session, bundle["generation"])
        self._check_record(control, record, cutoff=bundle["cutoff"])
        payload = event["payload"]
        self._check_payload(payload, record)
        if (
            event["native_record_sha256"] != digest
            or payload["auxiliary_sha256"] != bundle["mechanisms"]["sha256"]
            or any(payload[name] != control[name] for name in self._trace_fields())
            or payload["phase_closed"] != (auxiliary.terminal is not None)
        ):
            raise ValueError("El archivo contradice la emisión o el estado auxiliar confirmado")
        fields = ("flow_id", "sample_id", "prediction_at", "input_available_at", "input_sha256")
        if [{name: row[name] for name in fields} for row in control["rows"]] != observations:
            raise ValueError("La traza no conserva las observaciones verificadas de la generación")

    def read_event(self, session, bundle, generation):
        if bundle is None:
            raise ValueError("No hay eventos de mecanismos confirmados")
        bounded_integer(generation, "generación de mecanismos", 1, bundle["generation"])
        event = self.history.read_event(bundle["mechanism_history"], generation)
        record, digest = self._record(session, generation)
        self._check_payload(event["payload"], record)
        control = dict(event["payload"], observations=record["observations_count"])
        self._check_record(control, record, cutoff=event["cutoff"])
        if digest != event["native_record_sha256"]:
            raise ValueError("El registro histórico no conserva su huella nativa")
        return event
