"""Recorrer M0/M1 con rutas acotadas y recoger objetivos tras su maduración."""

import hashlib
from dataclasses import asdict, dataclass

import torch

from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.episodic_readout import EpisodicPrediction
from mars_titan.models.titans.frozen_financial import _numerics
from mars_titan.models.titans.refinement_policy import (
    RoutePolicy,
    _output,
    apply_routes,
    route_features,
)
from mars_titan.models.titans.refinement_routes import ROUTES, collect_routes, run_route
from mars_titan.training.mars_titan_run import MarsTitanInference


@dataclass(frozen=True)
class _Work:
    refinements: int
    searches: int
    value_projections: int


def _model_digest(model):
    if model is None:
        return None
    identity = {
        name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in (*model.named_parameters(), *model.named_buffers())
    }
    identity["contract"] = model.get_extra_state()
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


class RefinementInference(MarsTitanInference):
    """Un avance MAC por observación y selección de rutas sobre un banco M0/M1.

    La colección es un instrumento de desarrollo separado de la política. Sus
    etiquetas no llegan al cálculo de ninguna ruta y cada traza se publica al
    madurar. El recorrido base conserva calendario, admisión y reinicio de fases.
    """

    def __init__(self, *args, mode, route=None, policy=None, trace=None, **kwargs):
        if mode not in ("collect", "fixed", "policy"):
            raise ValueError("El modo debe ser collect, fixed o policy")
        if (mode == "fixed" and route not in ROUTES) or (mode != "fixed" and route is not None):
            raise ValueError("La ruta fija no corresponde al modo declarado")
        if (mode == "policy") != (type(policy) is RoutePolicy):
            raise ValueError("El modo de política necesita un controlador identificado")
        if mode == "collect" and not callable(trace):
            raise ValueError("La colección necesita un destino explícito para las trazas maduras")
        if mode != "collect" and trace is not None:
            raise ValueError("Las trazas alternativas solo pertenecen a la colección")
        super().__init__(*args, **kwargs)
        if self.admission not in ("m0", "m1"):
            raise ValueError("Las rutas compartidas requieren M0 o M1, sin admisión por error")
        self.mode, self.route, self.policy, self.trace = mode, route, policy, trace
        if policy is not None and policy.mean.device != self.device:
            raise ValueError("La política debe estar en el dispositivo del predictor")
        self.pending_traces = {}
        self._models = (self.predictor, self.readout, policy)
        self._verify_frozen()
        self._digests = tuple(_model_digest(model) for model in self._models)
        self._work_totals = _Work(0, 0, 0)
        self._route_counts = dict.fromkeys(ROUTES, 0)
        self._contract_json = canonical(self.identity())

    def _verify_frozen(self):
        for model in self._models:
            if model is not None and (
                any(module.training for module in model.modules())
                or any(
                    value.requires_grad or value.grad is not None for value in model.parameters()
                )
            ):
                raise ValueError("El núcleo, lector y política deben permanecer congelados")

    def identity(self):
        return dict(
            schema_version=1,
            family="bounded_refinement_routes_v1",
            mode=self.mode,
            route=self.route,
            model_digests=list(self._digests),
            admission=self.admission,
            inherited_readout_recipe=self.recipe.identity(),
            retention=asdict(self.retention) if self.retention is not None else None,
            codec=self.codec.fingerprint(),
            world=self.world,
            fold=self.fold,
            numerics=_numerics(),
            routes=list(ROUTES),
            max_pending_traces=32768,
            phase_state="restart_with_past_inputs_only_warmup",
        )

    def _new_pass(self, partition):
        if self.pending_traces:
            raise ValueError("Hay trazas de una fase incompleta. Crear un recorrido nuevo")
        self._work_totals = _Work(0, 0, 0)
        self._route_counts = dict.fromkeys(ROUTES, 0)
        return super()._new_pass(partition)

    def _collect(self, prepared, batch, phase, snapshot, context):
        collected = collect_routes(
            self.readout,
            prepared.working_state,
            snapshot,
            context_id=context,
            cutoff=batch.prediction_at[0],
        )
        outputs = [_output(self.predictor.head, collected.routes[route].state) for route in ROUTES]
        features, _, _ = route_features(
            prepared.working_state, collected.routes["k1"], self.predictor.head
        )
        predictions = torch.stack([output[0] for output in outputs], dim=1).cpu().tolist()
        quantiles = (
            torch.stack([output[1] for output in outputs], dim=1).cpu().tolist()
            if self.quantiles
            else [None] * len(batch.flow_ids)
        )
        features = features.cpu().tolist()
        if len(self.pending_traces) + len(batch.flow_ids) > 32768:
            raise ValueError("Las trazas pendientes superan el límite de filas")
        for index, key in enumerate(zip(batch.flow_ids, batch.prediction_at, strict=True)):
            if key in self.pending_traces:
                raise ValueError("La emisión ya tiene una traza pendiente")
            self.pending_traces[key] = dict(
                flow_id=key[0],
                decision_at=key[1],
                input_available_at=batch.input_available_at[index],
                partition=phase.partition,
                snapshot_context=context,
                features=features[index],
                predictions=predictions[index],
                quantiles=quantiles[index],
            )
        size = len(batch.flow_ids)
        work = _Work(
            collected.refinements * size,
            collected.searches * size,
            collected.value_projections * size,
        )
        return EpisodicPrediction(outputs[0][0], work, outputs[0][1])

    def _predict_block(self, prepared, batch, phase, snapshot, context):
        options = dict(context_id=context, cutoff=batch.prediction_at[0])
        if self.mode == "collect":
            result = self._collect(prepared, batch, phase, snapshot, context)
            for route in ROUTES:
                self._route_counts[route] += len(batch.flow_ids)
        elif self.mode == "fixed":
            route = run_route(
                self.readout, prepared.working_state, snapshot, route=self.route, **options
            )
            point, quantiles = _output(self.predictor.head, route.state)
            size = len(batch.flow_ids)
            work = _Work(
                route.refinements * size, route.searches * size, route.value_projections * size
            )
            result = EpisodicPrediction(point, work, quantiles)
            self._route_counts[self.route] += size
        else:
            chosen = apply_routes(
                self.readout,
                prepared.working_state,
                self.predictor.head,
                snapshot,
                policy=self.policy,
                **options,
            )
            routes = [ROUTES[code] for code in chosen.route_codes]
            searches = (
                sum(1 + int("r" in route) for route in routes)
                if snapshot is not None and snapshot.count
                else 0
            )
            work = _Work(sum(int(route[1]) for route in routes), searches, searches)
            for route in routes:
                self._route_counts[route] += 1
            result = EpisodicPrediction(chosen.point_predictions, work, chosen.quantiles)
        previous, work = self._work_totals, result.readout
        self._work_totals = _Work(
            previous.refinements + work.refinements,
            previous.searches + work.searches,
            previous.value_projections + work.value_projections,
        )
        return result

    def _refinement_work(self, result, size):
        return result.readout.refinements

    def _labels(self, run, phase, event, *, train, rows=None):
        if train:
            raise ValueError("El recorrido de rutas no ajusta el núcleo ni el lector")
        if self.mode == "collect":
            for flow, decision_at, target in event.labels:
                if event.at <= decision_at:
                    raise ValueError("La etiqueta de la traza no ha madurado después de decidir")
                key = (flow, decision_at)
                if key not in self.pending_traces:
                    raise ValueError("La etiqueta no tiene una traza emitida")
                record = self.pending_traces.pop(key)
                self.trace(dict(record, maturity_at=event.at, target=target))
        super()._labels(run, phase, event, train=False, rows=rows)

    def _close(self, run):
        self.pending_traces.clear()
        super()._close(run)

    def evaluate(self, source, *, stop=None, rows=None):
        self._verify_frozen()
        if canonical(self.identity()) != self._contract_json:
            raise ValueError("La receta, banco o ruta cambiaron desde la identificación")
        if tuple(_model_digest(model) for model in self._models) != self._digests:
            raise ValueError("Los parámetros o contratos cambiaron desde la selección")
        result = super().evaluate(source, stop=stop, rows=rows)
        self._verify_frozen()
        if canonical(self.identity()) != self._contract_json:
            raise ValueError("El recorrido modificó su receta, banco o ruta")
        if tuple(_model_digest(model) for model in self._models) != self._digests:
            raise ValueError("El recorrido modificó parámetros o contratos")
        return dict(
            result,
            searches=self._work_totals.searches,
            value_projections=self._work_totals.value_projections,
            route_counts=dict(self._route_counts),
            route_count_scope="computed_alternatives"
            if self.mode == "collect"
            else "executed_routes",
            route_count_denominator="all_predictions_including_unresolved",
        )
