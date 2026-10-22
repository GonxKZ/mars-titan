"""Alternativas condicionales desde un prefijo común, sin confirmar sus estados."""

import hashlib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from mars_titan.data.input_policy import MODALITIES, masked_inputs
from mars_titan.models.quantile_head import QuantileHead
from mars_titan.models.titans import refinement_routes, regional_routes
from mars_titan.models.titans.financial import _tensor_digest
from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.models.titans.mechanism_calibration import calibrate_event
from mars_titan.models.titans.mechanism_catalog import MechanismAction
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.mechanism_policy import ACTIONS
from mars_titan.models.titans.mechanism_selection import digest
from mars_titan.models.titans.refinement_policy import _output
from mars_titan.models.titans.refinement_routes import ROUTES

from . import mechanism_collection_record
from .episodic_session import DecisionRow
from .mechanism_collection_record import (
    KIND,
    PROTOCOL,
    REGIONAL_KIND,
    REGIONAL_WORK,
    WORK,
    CollectedMechanisms,
)
from .mechanism_observations import _row_features, extract_observation


def state_digest(value):
    """Identificar estados con tensores sin conservar sus copias CPU ni su grafo."""

    def visit(item):
        if isinstance(item, torch.Tensor):
            return {"tensor_sha256": _tensor_digest(item)}
        if type(item) is dict:
            return {key: visit(child) for key, child in item.items()}
        if type(item) in (tuple, list):
            return [visit(child) for child in item]
        return item

    return digest(visit(value))


class MechanismCollector:
    """Procesar bloques desde los mismos estados anteriores y sellar las salidas CPU.

    Cambia una acción de un mercado cada vez. Las otras conservan el ancla. No
    confirma pesos rápidos, contexto, calibrador o memoria y no recibe etiquetas.
    La publicación durable del paquete corresponde al coordinador de su ejecución.
    """

    def __init__(self, consumer, state, *, rows, cutoff, context_id, matures_at, snapshot=None):
        if (
            type(consumer) is not FrozenMechanismConsumer
            or type(consumer.predictor.head) is not QuantileHead
        ):
            raise ValueError("La colección requiere el consumidor identificado y cinco cuantiles")
        self.consumer = consumer
        if type(rows) not in (tuple, list) or not 1 <= len(rows) <= 8192:
            raise ValueError("La colección necesita filas explícitas y acotadas")
        self.observation = extract_observation(
            rows,
            consumer=consumer,
            state=state,
            flow_ids=tuple(r.flow_id for r in rows),
            cutoff=cutoff,
            context_id=context_id,
            snapshot=snapshot,
        )
        supplied = tuple(rows)
        self._metadata = {r.flow_id: r.metadata() for r in supplied}
        flows = tuple(r.flow_id for r in supplied)
        options = {"observation": self.observation} if consumer.policy is not None else {}
        self._event = consumer.begin(
            state,
            flow_ids=flows,
            cutoff=cutoff,
            context_id=context_id,
            matures_at=matures_at,
            snapshot=snapshot,
            **options,
        )
        self._availability = dict(self._event.availability)
        self._state = state
        self._before = state_digest(consumer.export_state(state))
        self._blocks, self._raw, self._closed = [], {}, False
        program = consumer.predictor.regional_execution
        self._regional = (
            program.trace_descriptor(consumer.predictor.regional_fusion)
            if program is not None
            else None
        )
        self._work = dict.fromkeys(
            sorted(WORK | (REGIONAL_WORK if self._regional is not None else set())), 0
        )
        self._anchor = [
            dict(
                market=market,
                requested=plan.requested.key,
                applied=plan.applied.key,
                reasons=list(plan.reasons),
            )
            for market, plan in self._event.plans
        ]
        self._preflight_node_updates()

    def _preflight_node_updates(self):
        """Reservar el peor caso de colección y emisión antes de cualquier MAC.

        La unión contempla cualquier partición de filas. Compartir prefijos reduce el
        trabajo real, pero no renueva ni amplía esta admisión conservadora del evento.
        """
        reader = self.consumer.readout
        if reader is None or not reader.recurrence_work(1):
            return
        reads = 0
        for context in ("off", "parent"):
            routes = self._routes(context)
            reads += sum(
                int(route[1])
                for route in ROUTES
                if any(route in allowed for allowed in routes.values())
            )
        episodes = self._event.snapshot.count if self._event.snapshot is not None else 0
        for _, plan in self._event.plans:
            reader.preflight_node_updates(reads + plan.reader_steps, episodes)

    def emission_event(self):
        """Reutilizar en la emisión el ancla que ya reservó colección y acción final."""
        self._verify_prefix()
        return self._event

    def _verify_prefix(self):
        if type(self.consumer) is not FrozenMechanismConsumer:
            raise ValueError("El consumidor de la colección cambió de tipo")
        self.consumer.verify()
        if (
            self._event.context is not self._state.context
            or self._event.calibration is not self._state.calibration
            or state_digest(self.consumer.export_state(self._state)) != self._before
        ):
            raise ValueError("El estado del prefijo cambió durante la colección")
        if self._event.snapshot is not None:
            self._event.snapshot.verify()

    def _routes(self, context):
        return {
            market: tuple(
                route
                for route in ROUTES
                if not self.consumer.catalog.resolve(
                    MechanismAction(context, route),
                    available,
                    rows=sum(f.startswith(market + "/") for f in self._event.flow_ids),
                ).reasons
            )
            for market, available in self._availability.items()
        }

    @torch.no_grad()
    def add(self, rows, fast_state):
        self._verify_prefix()
        if (
            self._closed
            or type(rows) not in (tuple, list)
            or not 1 <= len(rows) <= self.consumer.max_batch
        ):
            raise ValueError("La colección necesita un bloque abierto y acotado")
        if any(type(r) is not DecisionRow for r in rows):
            raise ValueError("La colección solo procesa filas admitidas")
        if any(
            r.flow_id in self._raw or r.metadata() != self._metadata.get(r.flow_id) for r in rows
        ):
            raise ValueError("El bloque repite o cambia una observación de la cohorte")
        flows = tuple(r.flow_id for r in rows)
        if len(set(flows)) != len(flows):
            raise ValueError("El bloque repite un activo")
        model = self.consumer.predictor
        # El mismo hash declarado no basta si se han sustituido los bytes de entrada.
        for row in rows:
            _row_features(row, model.config.inputs, self._event.cutoff)
        raw = dict(
            inputs={name: np.stack([r.inputs[name] for r in rows]) for name in MODALITIES},
            presence=np.stack([r.presence for r in rows]),
            sample_ids=[r.sample_id for r in rows],
            prediction_at=np.array([r.prediction_at for r in rows], dtype="datetime64[us]"),
            input_available_at=np.array(
                [r.input_available_at for r in rows], dtype="datetime64[us]"
            ),
        )
        input_presence = (
            raw["presence"].tolist()
            if self._regional is not None and self._regional["schema_version"] == 2
            else None
        )
        if not masked_inputs(model.config.inputs.input_policy):
            raw.pop("presence")
        batch = DecisionBatch.from_corpus(
            raw, model.config.inputs, device=model.head.weight.device, dtype=model.head.weight.dtype
        )
        before = state_digest(model.export_state(fast_state))
        proposed = {flow: {} for flow in flows}
        work = dict.fromkeys(self._work, 0)
        regional_calls = []
        for context in ("off", "parent"):
            routes = self._routes(context)
            if not any(routes[flow.split("/", 1)[0]] for flow in flows):
                continue
            options = {}
            if model.session_context is not None:
                options = dict(
                    context=self._event.context,
                    context_selection=tuple(
                        context == "parent" and bool(routes[f.split("/", 1)[0]]) for f in flows
                    ),
                )
            prepared = model.prepare(batch, fast_state, differentiable=False, **options)
            if self._regional is not None:
                traces = model.regional_execution.export_trace(
                    prepared.regional_trace, model.regional_fusion
                )
                if input_presence is not None and any(
                    trace["availability"][:5] != present
                    for trace, present in zip(traces, input_presence, strict=True)
                ):
                    raise ValueError("La disponibilidad regional cambió las presencias de entrada")
                for trace, flow in zip(traces, flows, strict=True):
                    selected = context == "parent" and bool(routes[flow.split("/", 1)[0]])
                    if not selected and trace["context_norm"] != 0:
                        raise ValueError("Una alternativa sin consulta conserva contexto regional")
                    work["regional_local_node_evaluations"] += trace["work"][
                        "local_node_evaluations"
                    ]
                    work["regional_summary_evaluations"] += trace["work"]["summary_evaluations"]
                    work["regional_program_selection_rows"] += 1
                    work["regional_context_query_rows"] += int(selected)
                regional_calls.append(
                    dict(
                        context=context,
                        rows=[
                            dict(flow_id=flow, trace=trace)
                            for flow, trace in zip(flows, traces, strict=True)
                        ],
                    )
                )
            work["core_observations"] += len(flows)
            work["mac_online_observations"] += (
                len(flows) if model.config.variant == "mac_online" else 0
            )
            work["context_passes"] += 1
            union = tuple(
                route for route in ROUTES if any(route in routes[f.split("/", 1)[0]] for f in flows)
            )
            outputs = {}
            if self.consumer.readout is None:
                outputs["k1"] = prepared.quantiles.detach().cpu().tolist()
            else:
                reader, base = self.consumer.readout, prepared.working_state
                route_api = (
                    regional_routes
                    if reader.config.evidence_bridge is not None
                    else refinement_routes
                )
                count = self._event.snapshot.count if self._event.snapshot is not None else 0
                if (
                    len(union) == 6
                    and reader.estimated_bytes(len(flows), count, refinements=9)
                    <= reader.config.max_working_bytes
                ):
                    collected = route_api.collect_routes(
                        reader,
                        base,
                        self._event.snapshot,
                        context_id=self._event.context_id,
                        cutoff=self._event.cutoff,
                    )
                    outputs = {
                        route: _output(model.head, value.state)[1].detach().cpu().tolist()
                        for route, value in collected.routes.items()
                    }
                    work["reader_refinements"] += collected.refinements * len(flows)
                    work["reader_searches"] += collected.searches * len(flows)
                    work["reader_value_projections"] += collected.value_projections * len(flows)
                    work["shared_route_batches"] += 1
                    del collected
                else:
                    for route in union:
                        _, final = route_api.run_route_pair(
                            reader,
                            base,
                            self._event.snapshot,
                            route=route,
                            context_id=self._event.context_id,
                            cutoff=self._event.cutoff,
                        )
                        outputs[route] = _output(model.head, final.state)[1].detach().cpu().tolist()
                        work["reader_refinements"] += final.refinements * len(flows)
                        work["reader_searches"] += final.searches * len(flows)
                        work["reader_value_projections"] += final.value_projections * len(flows)
                        work["individual_route_batches"] += 1
                        del final
            for index, flow in enumerate(flows):
                proposed[flow].update(
                    {
                        f"{context}/{route}": outputs[route][index]
                        for route in routes[flow.split("/", 1)[0]]
                    }
                )
            del prepared
        if state_digest(model.export_state(fast_state)) != before:
            raise ValueError("Una alternativa modificó el estado rápido de partida")
        self._verify_prefix()
        # Solo se incorporan bloques completos. Un error anterior no deja medias filas.
        self._raw.update(proposed)
        block = dict(flow_ids=list(flows), fast_state_sha256=before)
        if self._regional is not None:
            block["regional_execution"] = regional_calls
        self._blocks.append(block)
        for name, value in work.items():
            self._work[name] += value

    def _rows(self, plans):
        result = []
        for flow in self._event.flow_ids:
            plan = plans[flow.split("/", 1)[0]]
            key = f"{plan.applied.context}/{plan.applied.route}"
            values = self._raw[flow][key]
            k1 = self._raw[flow][f"{plan.applied.context}/k1"]
            result.append(
                dict(
                    flow_id=flow,
                    sample_id=self._metadata[flow]["sample_id"],
                    point=values[2],
                    raw_quantiles=list(values),
                    k1_point=k1[2],
                    k1_quantiles=list(k1),
                    requested=plan.requested.key,
                    context=plan.applied.context,
                    executed_route=plan.applied.route,
                    emitted_route=plan.applied.route,
                    calibration=plan.applied.calibration,
                    reasons=plan.reasons,
                    reader_steps=plan.reader_steps,
                    reader_searches=plan.reader_searches,
                    discarded_reader_steps=0,
                )
            )
        return result

    def finish(self):
        self._verify_prefix()
        if self._closed or set(self._raw) != set(self._event.flow_ids):
            raise ValueError("La colección está cerrada o todavía le faltan filas")
        anchor = dict(self._event.plans)
        groups = []
        work = dict(self._work)
        for market, available in self._availability.items():
            indices = [
                i for i, flow in enumerate(self._event.flow_ids) if flow.startswith(market + "/")
            ]
            candidates = []
            for key in ACTIONS:
                requested = MechanismAction(*key.split("/"))
                plan = self.consumer.catalog.resolve(requested, available, rows=len(indices))
                item = dict(
                    requested=key,
                    emitted=None,
                    status="unavailable",
                    reasons=list(plan.reasons),
                    quantiles=None,
                    joint_actions=None,
                    calibration_work=None,
                    cross_returns=[],
                )
                if not plan.reasons:
                    plans = dict(anchor, **{market: plan})
                    rows = self._rows(plans)
                    candidate = replace(self._event, plans=tuple(plans.items()))
                    _, _, output, calibration_work = calibrate_event(
                        candidate, rows, self.consumer._static
                    )
                    selected = rows[indices[0]]
                    item.update(
                        status="fallback" if selected["reasons"] else "observed",
                        emitted=MechanismAction(
                            selected["context"], selected["emitted_route"], selected["calibration"]
                        ).key,
                        reasons=list(selected["reasons"]),
                        quantiles=output[indices].tolist(),
                        joint_actions={name: p.applied.key for name, p in plans.items()},
                        calibration_work=calibration_work,
                        cross_returns=[
                            dict(
                                market=name,
                                requested=p.applied.key,
                                emitted=MechanismAction(p.applied.context, "k1", "none").key,
                                reasons=["pending_calibration_bytes"],
                            )
                            for name, p in plans.items()
                            if name != market
                            and any(
                                r["flow_id"].startswith(name + "/")
                                and "pending_calibration_bytes" in r["reasons"]
                                for r in rows
                            )
                        ],
                    )
                    for source, target in (
                        ("online_attempts", "calibration_attempts"),
                        ("online_created_proposals", "calibration_created_proposals"),
                        ("online_discarded_proposals", "calibration_discarded_proposals"),
                    ):
                        work[target] += calibration_work[source]
                candidates.append(item)
            groups.append(
                dict(market=market, availability=asdict(available), candidates=candidates)
            )
        body = dict(
            schema_version=2 if self._regional is not None else 1,
            kind=REGIONAL_KIND if self._regional is not None else KIND,
            protocol=PROTOCOL,
            output_dtype=str(self.consumer.predictor.head.weight.dtype),
            execution_device=str(self.consumer.predictor.head.weight.device),
            observation=self.observation.record(),
            catalog=self.consumer.catalog.identity(),
            rows=[
                dict(self._metadata[flow], matures_at=due)
                for flow, due in zip(self._event.flow_ids, self._event.matures_at, strict=True)
            ],
            anchor=self._anchor,
            prior_state_sha256=self._before,
            producer_sha256=digest(
                {
                    str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__), Path(mechanism_collection_record.__file__))
                }
            ),
            blocks=sorted(self._blocks, key=lambda b: b["flow_ids"]),
            groups=groups,
            work=work,
        )
        if self._regional is not None:
            body["regional_execution"] = self._regional
        packet = CollectedMechanisms(body)
        self._closed = True
        self._raw.clear()
        return packet
