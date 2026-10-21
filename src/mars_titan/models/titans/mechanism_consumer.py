"""Consumidor congelado de contexto, rutas y calibración con propuestas por cohorte.

No publica generaciones. El enlace financiero debe confirmar su estado auxiliar junto
con el estado rápido y la cola nativa. Las acciones explícitas son controles fijos,
no un controlador aprendido ni una reproducción de un mecanismo cerebral.
"""

import hashlib
import importlib
import inspect
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from mars_titan.calibration.cohort_feedback import CohortFeedback
from mars_titan.calibration.online_conformal import _instant, _require
from mars_titan.models.quantile_head import QuantileHead

from .config import bounded_integer, canonical
from .episodic_readout import EpisodicReadout
from .episodic_snapshot import EpisodeSnapshot
from .financial import FinancialPredictor, _tensor_digest
from .financial_inputs import DecisionBatch
from .frozen_financial import _execution_signature, _implementation, _numerics
from .mechanism_calibration import _reserve_fits, _static_record, calibrate_event
from .mechanism_catalog import MechanismAction, MechanismAvailability, MechanismCatalog
from .mechanism_event import (
    MechanismBlock,
    MechanismEmission,
    MechanismEvent,
    MechanismState,
    frozen_array,
    frozen_row,
)
from .refinement_policy import _output
from .refinement_routes import run_route_pair
from .session_context import SessionContextState

_SOURCES = (
    __name__,
    "mars_titan.models.titans.mechanism_calibration",
    "mars_titan.models.titans.mechanism_event",
    "mars_titan.models.titans.mechanism_catalog",
    "mars_titan.models.titans.refinement_routes",
    "mars_titan.models.titans.refinement_policy",
    "mars_titan.calibration.cohort_feedback",
    "mars_titan.calibration.cohort_feedback_state",
    "mars_titan.calibration.online_conformal",
    "mars_titan.calibration.conformal_quantiles",
)


def _code():
    return _implementation() | {
        name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
        for name in _SOURCES
    }


def _digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _reader_digest(reader):
    return (
        None
        if reader is None
        else _digest(
            {
                name: _tensor_digest(value)
                for name, value in (*reader.named_parameters(), *reader.named_buffers())
            }
        )
    )


def _proof(value):
    _require(
        type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "Se necesita una identidad SHA-256",
    )


def _methods(instance):
    result = []
    for name, method in inspect.getmembers(FrozenMechanismConsumer, inspect.isfunction):
        if name in vars(instance) or method.__module__ != __name__:
            raise ValueError("El consumidor no admite métodos sustituidos")
        result.append((name, id(method), id(method.__code__)))
    return tuple(result)


class FrozenMechanismConsumer:
    """Preparar bloques en GPU sin retener sus estados rápidos durante toda la cohorte."""

    def __init__(
        self,
        predictor,
        *,
        readout=None,
        calibration=None,
        action=None,
        catalog=None,
    ):
        if type(self) is not FrozenMechanismConsumer or type(predictor) is not FinancialPredictor:
            raise ValueError("El consumidor necesita un predictor financiero identificado")
        action = MechanismAction() if action is None else action
        catalog = MechanismCatalog() if catalog is None else catalog
        if type(action) is not MechanismAction or type(catalog) is not MechanismCatalog:
            raise ValueError("El consumidor necesita acción y catálogo identificados")
        if readout is not None and (
            type(readout) is not EpisodicReadout
            or readout.config.refinements != 1
            or readout.config.hidden_size != predictor.config.hidden_size
            or readout.step_logit.device != predictor.head.weight.device
            or readout.step_logit.dtype != predictor.head.weight.dtype
        ):
            raise ValueError(
                "El lector debe ser el control K1 y compartir dimensión, precisión y dispositivo"
            )
        if calibration is not None and (
            type(calibration) is not CohortFeedback or calibration.pending_rows
        ):
            raise ValueError("El calibrador inicial no puede tener cohortes pendientes")
        self.predictor, self.readout, self.action, self.catalog = (
            predictor,
            readout,
            action,
            catalog,
        )
        self.max_batch = min(
            predictor.config.max_batch,
            readout.config.max_batch if readout else predictor.config.max_batch,
        )
        self._initial_calibration = calibration.export() if calibration is not None else None
        if calibration is not None:
            if self._initial_calibration["clock"] is not None or not _reserve_fits(calibration):
                raise ValueError("El calibrador de fase necesita reinicio y reserva para feedback")
        self._static = (
            _static_record(self._initial_calibration) if calibration is not None else None
        )
        self._models = (("predictor", predictor),) + ((("readout", readout),) if readout else ())
        self._signature, modes = _execution_signature(self._models)
        predictor.verify_parameter_identity()
        self._reader_id, self._code, self._flags = _reader_digest(readout), _code(), _numerics()
        self._identity = dict(
            schema_version=1,
            recipe="frozen_mechanism_cohort_proposals_v1",
            predictor=predictor.get_extra_state(),
            predictor_parameters_sha256=predictor._parameter_id,
            readout=readout.get_extra_state() if readout else None,
            reader_sha256=self._reader_id,
            action=asdict(action),
            catalog=catalog.identity(),
            calibration=_digest(self._initial_calibration) if calibration is not None else None,
            static_calibration=_digest(self._static),
            implementation=self._code,
            numerics=self._flags,
            modules=modes,
            online_policy="only_cohorts_emitted_online_predict_all_then_feedback_v1",
            phase_end="archive_censored_pending_without_maturity_or_cross_phase_reuse",
            fixed_controls=True,
            new_learned_selector=False,
            planning_batch=self.max_batch,
        )
        self.model_id = _digest(self._identity)
        self._methods = _methods(self)

    def identity(self):
        return json.loads(canonical(self._identity))

    def verify(self, *, strong=True):
        if type(strong) is not bool:
            raise ValueError("La verificación necesita un modo booleano explícito")
        if type(self.action) is not MechanismAction or type(self.catalog) is not MechanismCatalog:
            raise ValueError("El catálogo o la acción cambiaron de tipo")
        if (
            _methods(self) != self._methods
            or self.max_batch != self._identity["planning_batch"]
            or self.predictor is not self._models[0][1]
            or self.readout is not (self._models[1][1] if len(self._models) > 1 else None)
        ):
            raise ValueError("El consumidor cambió sus componentes o su plan de ejecución")
        signature, _ = _execution_signature(self._models)
        if (
            signature != self._signature
            or _numerics() != self._flags
            or (
                self.predictor.get_extra_state() != self._identity["predictor"]
                or (self.readout.get_extra_state() if self.readout else None)
                != self._identity["readout"]
                or asdict(self.action) != self._identity["action"]
                or self.catalog.identity() != self._identity["catalog"]
            )
        ):
            raise ValueError("El cálculo cambió respecto al consumidor identificado")
        if strong:
            if (
                _digest(self._static) != self._identity["static_calibration"]
                or (
                    _digest(self._initial_calibration)
                    if self._initial_calibration is not None
                    else None
                )
                != self._identity["calibration"]
            ):
                raise ValueError("El calibrador estático cambió respecto de su identidad")
            self.predictor.verify_parameter_identity()
            if _reader_digest(self.readout) != self._reader_id or _code() != self._code:
                raise ValueError("El código o los parámetros cambiaron durante la cohorte")

    def initial_state(self):
        self.verify()
        return MechanismState(
            self.model_id,
            None,
            self.predictor.initial_context() if self.predictor.session_context else None,
            CohortFeedback.restore(self._initial_calibration)
            if self._initial_calibration is not None
            else None,
        )

    def _check_state(self, state, *, active=True):
        if type(state) is not MechanismState or state.model_id != self.model_id:
            raise ValueError("El estado auxiliar pertenece a otra identidad")
        if active and state.terminal is not None:
            raise ValueError("La fase está cerrada y sus pendientes solo son archivo censurado")
        if state.cutoff is not None:
            bounded_integer(state.cutoff, "corte auxiliar", 0, 1_704_067_200_000_000)
        if (state.context is None) != (self.predictor.session_context is None):
            raise ValueError("El estado no conserva el contexto del padre")
        if state.context is not None:
            if type(state.context) is not SessionContextState:
                raise ValueError("El contexto auxiliar tiene un tipo ajeno al predictor")
            self.predictor.session_context.validate(state.context, self.predictor._parameter_id)
            if any(value is not None for value in state.context.pending_tokens):
                raise ValueError("El estado auxiliar aún tiene una sesión sin completar")
            if any(
                at >= 0 and (state.cutoff is None or at > state.cutoff)
                for at in state.context.committed_at
            ):
                raise ValueError("El contexto confirmado procede del futuro")
        if (state.calibration is None) != (self._initial_calibration is None):
            raise ValueError("El estado no conserva el calibrador del padre")
        if state.calibration is not None:
            if type(state.calibration) is not CohortFeedback:
                raise ValueError("El calibrador auxiliar no tiene un tipo admitido")
            payload = state.calibration.export()
            if payload["calibrator"]["header"] != self._initial_calibration["calibrator"][
                "header"
            ] or (
                payload["max_rows"] != self._initial_calibration["max_rows"]
                or payload["max_bytes"] != self._initial_calibration["max_bytes"]
                or (
                    payload["clock"] is not None
                    and (state.cutoff is None or payload["clock"] > state.cutoff)
                )
            ):
                raise ValueError("El calibrador pertenece a otra fuente, presupuesto o corte")

    def begin(
        self,
        state,
        *,
        flow_ids,
        cutoff,
        context_id,
        matures_at,
        action=None,
        snapshot=None,
        warmup=False,
    ):
        self.verify()
        self._check_state(state)
        cutoff = _instant(cutoff, "La decisión de mecanismos")
        _proof(context_id)
        action = self.action if action is None else action
        if type(action) is not MechanismAction or type(warmup) is not bool:
            raise ValueError("La cohorte necesita una acción y modo explícitos")
        if type(flow_ids) not in (tuple, list):
            raise ValueError("La cohorte necesita una lista o tupla de filas acotada")
        bounded_integer(len(flow_ids), "filas del evento de mecanismos", 1, 8192)
        flows = tuple(flow_ids)
        self.predictor._check_flows(flows)
        if state.cutoff is not None and cutoff <= state.cutoff:
            raise ValueError("La decisión debe ser posterior al estado auxiliar")
        if warmup:
            if matures_at is not None:
                raise ValueError("El calentamiento no recibe fechas de etiquetas")
            maturity = {}
            action = MechanismAction(action.context, "k1", "none")
        else:
            if type(matures_at) is not tuple or len(matures_at) != len(flows):
                raise ValueError("La cohorte necesita una madurez explícita por fila")
            maturity = {
                flow: _instant(at, "La madurez de fila") if at is not None else None
                for flow, at in zip(flows, matures_at, strict=True)
            }
            if any(at is not None and at <= cutoff for at in maturity.values()):
                raise ValueError("La madurez debe ser posterior a la emisión")
        if warmup and snapshot is not None:
            raise ValueError("El calentamiento no consulta episodios")
        if snapshot is not None:
            if type(snapshot) is not EpisodeSnapshot:
                raise ValueError("La instantánea no tiene un tipo admitido")
            snapshot.verify()
        flows = tuple(sorted(flows))
        groups = tuple(sorted({flow.split("/", 1)[0] for flow in flows}))
        known_maturity = {
            group: all(
                maturity.get(flow) is not None for flow in flows if flow.startswith(group + "/")
            )
            for group in groups
        }
        planning_rows = min(len(flows), self.max_batch)
        reader_limit, populated = 4, False
        if self.readout is not None and not warmup:
            with torch.no_grad():
                base = self.predictor.head.weight.new_zeros(
                    (planning_rows, self.predictor.config.hidden_size)
                )
                self.readout._validate_read(base, snapshot, context_id, cutoff, refinements=1)
            count = snapshot.count if snapshot is not None else 0
            reader_limit = max(
                k
                for k in (1, 2, 4)
                if self.readout.estimated_bytes(planning_rows, count, refinements=k)
                <= self.readout.config.max_working_bytes
            )
            populated = count > 0
        elif snapshot is not None:
            raise ValueError("No hay un lector que admita la instantánea")
        available_online = {}
        for group in groups:
            rows = sum(flow.startswith(group + "/") for flow in flows)
            available_online[group] = (
                not warmup
                and state.calibration is not None
                and group in self._static["groups"]
                and state.calibration.can_emit(group, rows=rows)
            )
        if state.calibration is not None:
            payload = state.calibration.export()
            eligible_groups = {g for g in groups if available_online[g] and known_maturity[g]}
            online_rows = sum(flow.split("/", 1)[0] in eligible_groups for flow in flows)
            if payload["max_rows"] < online_rows + state.calibration.pending_rows:
                available_online = dict.fromkeys(groups, False)
        plans = []
        for group in groups:
            rows = sum(flow.startswith(group + "/") for flow in flows)
            availability = MechanismAvailability(
                context=self.predictor.session_context is not None,
                reader=self.readout is not None and not warmup,
                quantiles=type(self.predictor.head) is QuantileHead,
                static=self._static is not None and group in self._static["groups"],
                online=state.calibration is not None,
                online_capacity=available_online[group],
                online_maturity=known_maturity[group],
                context_capacity=len(groups) <= self.catalog.budget.max_context_markets,
                nonempty_bank=populated,
                reader_limit=reader_limit,
            )
            resolved = self.catalog.resolve(action, availability, rows=rows)
            plans.append((group, resolved))
        return MechanismEvent(
            self.model_id,
            context_id,
            cutoff,
            flows,
            tuple(maturity[flow] for flow in flows) if not warmup else (),
            tuple(plans),
            state.context,
            state.calibration,
            snapshot,
            warmup,
        )

    def prepare(self, batch, state, event):
        self.verify(strong=False)
        if type(batch) is not DecisionBatch:
            raise ValueError("El bloque de mecanismos no es una entrada financiera validada")
        if type(event) is not MechanismEvent or event.model_id != self.model_id:
            raise ValueError("La propuesta pertenece a otro consumidor")
        processed = {row["flow_id"] for row in event.rows}
        if any(flow in processed for flow in batch.flow_ids):
            raise ValueError("La cohorte repite una fila ya preparada")
        if not set(batch.flow_ids) <= set(event.flow_ids) or any(
            at != event.cutoff for at in batch.prediction_at
        ):
            raise ValueError("El bloque no pertenece a la cohorte declarada")
        bounded_integer(len(batch.flow_ids), "lote del consumidor", 1, self.max_batch)
        plans = dict(event.plans)
        options = {}
        if self.predictor.session_context is not None:
            options = dict(
                context=event.context,
                context_selection=tuple(
                    plans[flow.split("/", 1)[0]].applied.context == "parent"
                    for flow in batch.flow_ids
                ),
            )
        prepared = self.predictor.prepare(batch, state, differentiable=False, **options)
        if event.warmup:
            rows = tuple(
                frozen_row(
                    flow_id=flow,
                    sample_id=sample,
                    point=None,
                    raw_quantiles=None,
                    k1_point=None,
                    k1_quantiles=None,
                    requested=plans[flow.split("/", 1)[0]].requested.key,
                    context=plans[flow.split("/", 1)[0]].applied.context,
                    executed_route="k1",
                    emitted_route="k1",
                    calibration="none",
                    reasons=(),
                    reader_steps=0,
                    reader_searches=0,
                    discarded_reader_steps=0,
                )
                for flow, sample in zip(batch.flow_ids, batch.sample_ids, strict=True)
            )
            return MechanismBlock(
                prepared.next_state,
                replace(event, context=prepared.next_context, rows=event.rows + rows),
            )
        rows = []
        selected_routes = tuple(
            plans[flow.split("/", 1)[0]].applied.route for flow in batch.flow_ids
        )
        for route in sorted(set(selected_routes)):
            indices = [i for i, value in enumerate(selected_routes) if value == route]
            index = torch.tensor(indices, dtype=torch.int64, device=prepared.working_state.device)
            base = prepared.working_state.index_select(0, index)
            first = last = base
            steps = searches = 0
            if self.readout is not None and not event.warmup:
                initial, final = run_route_pair(
                    self.readout,
                    base,
                    event.snapshot,
                    route=route,
                    context_id=event.context_id,
                    cutoff=event.cutoff,
                )
                first, last, steps, searches = (
                    initial.state,
                    final.state,
                    final.refinements,
                    final.searches,
                )
            if self.readout is None:
                initial_points = prepared.point_predictions.index_select(0, index)
                initial_quantiles = (
                    prepared.quantiles.index_select(0, index)
                    if prepared.quantiles is not None
                    else None
                )
                points, final_quantiles = initial_points, initial_quantiles
            else:
                with torch.no_grad():
                    initial_points, initial_quantiles = _output(self.predictor.head, first)
                    points, final_quantiles = (
                        (initial_points, initial_quantiles)
                        if first is last
                        else _output(self.predictor.head, last)
                    )
            quantiles = final_quantiles is not None
            original = initial_quantiles if quantiles else initial_points
            output = final_quantiles if quantiles else points
            raw = output.detach().cpu().tolist()
            k1 = original.detach().cpu().tolist()
            point_values, initial_values = (
                points.detach().cpu().tolist(),
                initial_points.detach().cpu().tolist(),
            )
            for local, position in enumerate(indices):
                plan = plans[batch.flow_ids[position].split("/", 1)[0]]
                rows.append(
                    frozen_row(
                        flow_id=batch.flow_ids[position],
                        sample_id=batch.sample_ids[position],
                        point=point_values[local],
                        raw_quantiles=tuple(raw[local]) if quantiles else None,
                        k1_point=initial_values[local],
                        k1_quantiles=tuple(k1[local]) if quantiles else None,
                        requested=plan.requested.key,
                        context=plan.applied.context,
                        executed_route=plan.applied.route,
                        emitted_route=plan.applied.route,
                        calibration=plan.applied.calibration,
                        reasons=plan.reasons,
                        reader_steps=steps,
                        reader_searches=searches,
                        discarded_reader_steps=0,
                    )
                )
        return MechanismBlock(
            prepared.next_state,
            replace(event, context=prepared.next_context, rows=event.rows + tuple(rows)),
        )

    def finish(self, event):
        self.verify()
        if type(event) is not MechanismEvent or event.model_id != self.model_id:
            raise ValueError("La propuesta pertenece a otro consumidor")
        if event.snapshot is not None:
            event.snapshot.verify()
        by_flow = {row["flow_id"]: dict(row) for row in event.rows}
        if len(event.rows) != len(event.flow_ids) or set(by_flow) != set(event.flow_ids):
            raise ValueError("La cohorte no está completa")
        rows = [by_flow[flow] for flow in event.flow_ids]
        work = dict(
            core_observations=len(rows),
            mac_online_observations=len(rows)
            if self.predictor.config.variant == "mac_online"
            else 0,
            reader_refinements=sum(row["reader_steps"] for row in rows),
            reader_searches=sum(row["reader_searches"] for row in rows),
            context_updates_proposed=sum(
                tokens is not None for tokens in event.context.pending_tokens
            )
            if event.context is not None
            else 0,
        )
        if self.predictor.regional_fusion is not None:
            work.update(self.predictor.regional_fusion.operation_counts(len(rows)))
        context = (
            self.predictor.commit_context(event.context) if event.context is not None else None
        )
        calibration = event.calibration
        if event.warmup:
            return MechanismEmission(
                MechanismState(self.model_id, event.cutoff, context, calibration),
                event.flow_ids,
                None,
                None,
                None,
                tuple(frozen_row(**row) for row in rows),
                frozen_row(**work),
            )
        calibration, raw, output, calibration_work = calibrate_event(event, rows, self._static)
        work.update(calibration_work)
        points = frozen_array([row["point"] for row in rows])
        if not np.isfinite(points).all():
            raise ValueError("La predicción emitida contiene valores no finitos")
        if output is not None and not np.array_equal(output[:, 2], points):
            raise ValueError("La calibración cambió la mediana realmente emitida")
        return MechanismEmission(
            MechanismState(self.model_id, event.cutoff, context, calibration),
            event.flow_ids,
            points,
            frozen_array(output) if output is not None else None,
            frozen_array(raw) if raw is not None else None,
            tuple(frozen_row(**row) for row in rows),
            frozen_row(**work),
        )

    def feedback(self, state, outcomes, *, at, settlement=False):
        self.verify()
        self._check_state(state)
        at = _instant(at, "El corte del feedback")
        if (
            type(settlement) is not bool
            or state.cutoff is None
            or at < state.cutoff
            or (not settlement and at != state.cutoff)
        ):
            raise ValueError("El feedback debe seguir a la emisión o declarar settlement")
        if state.calibration is None:
            _require(not outcomes, "No hay cohortes online a las que enviar estos resultados")
            return replace(state, cutoff=at), ()
        proposed = state.calibration.feedback(outcomes, at=at)
        return replace(state, cutoff=at, calibration=proposed.state), proposed.settlements

    def close_phase(self, state, *, at, evidence_sha256):
        self.verify()
        self._check_state(state)
        _proof(evidence_sha256)
        bounded_integer(
            at,
            "cierre administrativo",
            state.cutoff if state.cutoff is not None else 0,
            1_704_067_200_000_000,
        )
        terminal = frozen_row(
            status="phase_closed",
            closed_at=at,
            evidence_sha256=evidence_sha256,
            censored_rows=state.calibration.pending_rows if state.calibration is not None else 0,
            archived_calibration_sha256=state.calibration.state_sha256()
            if state.calibration is not None
            else None,
        )
        return replace(state, cutoff=at, terminal=terminal)

    def export_state(self, state):
        self.verify()
        self._check_state(state, active=False)
        return dict(
            schema_version=1,
            model_id=self.model_id,
            cutoff=state.cutoff,
            context=self.predictor.export_context(state.context)
            if state.context is not None
            else None,
            calibration=state.calibration.export() if state.calibration is not None else None,
            terminal=dict(state.terminal) if state.terminal is not None else None,
        )

    def restore_state(self, payload):
        self.verify()
        if (
            type(payload) is not dict
            or set(payload)
            != {"schema_version", "model_id", "cutoff", "context", "calibration", "terminal"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["model_id"] != self.model_id
        ):
            raise ValueError("El estado auxiliar no corresponde a este consumidor")
        state = MechanismState(
            self.model_id,
            payload["cutoff"],
            self.predictor.restore_context(payload["context"])
            if payload["context"] is not None
            else None,
            CohortFeedback.restore(payload["calibration"])
            if payload["calibration"] is not None
            else None,
            frozen_row(**payload["terminal"]) if payload["terminal"] is not None else None,
        )
        self._check_state(state, active=False)
        if state.terminal is not None:
            expected = self.close_phase(
                replace(state, terminal=None),
                at=state.cutoff,
                evidence_sha256=state.terminal.get("evidence_sha256"),
            )
            if dict(expected.terminal) != dict(state.terminal):
                raise ValueError("El archivo terminal no conserva sus cohortes censuradas")
        return state
