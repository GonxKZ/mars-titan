"""Recorrido cronológico de `FinancialPredictor` con BPTT truncado y etiquetas maduras.

El optimizador ajusta los parámetros compartidos, la memoria persistente de MAC y los
pesos rápidos iniciales aprendidos. Los pesos rápidos y el momentum de cada flujo
solo cambian mediante la regla asociativa de Titans al observar entradas. El estado
de trabajo de cada predicción se descarta después de usarla.
"""

import hashlib
import importlib
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from mars_titan.budget_training import validate_loss
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.session_metrics import SessionErrors
from mars_titan.memory.financial_observations import FinancialObservationSource
from mars_titan.models.quantile_head import PINBALL, QUANTILE_HEAD, pinball_loss
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.financial import VARIANTS, FinancialPredictor, FinancialState
from mars_titan.models.titans.financial_inputs import DecisionBatch, validated_cpu_batch
from mars_titan.models.titans.frozen_financial import _implementation, _numerics
from mars_titan.models.titans.state import MACState, NeuralMemoryState

from .checkpoints import (
    StopRequest,
    capture_rng,
    load_training_state,
    restore_rng,
    save_training_state,
)
from .learning_hold import require_learning_allowed
from .selection import (
    VALIDATION_PLATEAU,
    advance_selection,
    initial_selection,
    validate_selection,
)

RECIPE = "titans_financial_chronological_v1"
_OWN_MODULES = (
    "mars_titan.training.financial_run",
    "mars_titan.memory.financial_observations",
    "mars_titan.training.checkpoints",
    "mars_titan.training.selection",
    "mars_titan.evaluation.session_metrics",
)


# Marca de una predicción del tramo cuyo grafo se recalcula al actualizar.
_REPLAY = object()


def _default_selection():
    return dict(metric="session_mae", patience=3, min_delta=0.0)


@dataclass(frozen=True)
class ChronologicalRecipe:
    """Hiperparámetros declarados antes de ejecutar, comunes a los controles emparejados."""

    truncation: int = 8
    loss: str = "mae"
    huber_delta: float = 0.01
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    epochs: int = 20
    selection: dict = field(default_factory=_default_selection)
    block_rows: int = 128
    checkpoint_updates: int = 256
    checkpoint_seconds: float = 900.0
    # None conserva el grafo de todos los flujos del tramo. Un entero recalcula el tramo
    # por bloques de hasta ese número de flujos y acumula sus gradientes antes del paso.
    accumulation_rows: int | None = None

    def __post_init__(self):
        # pinball solo corresponde a `quantile_head_v1`. huber_delta conserva su validación.
        validate_loss("mae" if self.loss == PINBALL else self.loss, self.huber_delta)
        validate_selection(self.selection, epochs=self.epochs)
        numbers = (self.learning_rate, self.weight_decay, self.checkpoint_seconds)
        if (
            any(type(v) is not int or not 1 <= v <= m for v, m in self._bounded())
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers)
            or not 0 < self.learning_rate <= 1
            or self.weight_decay < 0
            or self.checkpoint_seconds <= 0
            or (
                self.accumulation_rows is not None
                and (
                    type(self.accumulation_rows) is not int
                    or not 1 <= self.accumulation_rows <= 256
                )
            )
            or (
                self.max_grad_norm is not None
                and (
                    type(self.max_grad_norm) not in (int, float)
                    or not math.isfinite(self.max_grad_norm)
                    or self.max_grad_norm <= 0
                )
            )
        ):
            raise ValueError(
                "La receta necesita truncamiento, presupuesto, optimizador y parada válidos"
            )

    def _bounded(self):
        return (
            (self.truncation, 256),
            (self.epochs, 1000),
            (self.block_rows, 256),
            (self.checkpoint_updates, 1_000_000),
        )

    def identity(self):
        fields = asdict(self)
        # Sin acumulación se conserva literalmente la identidad anterior y su huella.
        if fields["accumulation_rows"] is None:
            fields.pop("accumulation_rows")
        else:
            fields["gradient_accumulation"] = "flow_blocks_replayed_from_segment_start_v1"
        return dict(
            schema_version=1,
            recipe=RECIPE,
            **fields,
            optimizer="AdamW",
            truncation_unit="decision_instants_per_differentiable_segment",
            label_rule="loss_only_after_maturity_event_then_next_segment_update",
            loss_reduction="mean_over_matured_labels_of_segment",
            fast_state="reset_each_pass_then_warmup",
            selection_metric_definition="SessionErrors.session_mae_on_issued_predictions",
        )


def load_recipe(path):
    """Leer la receta declarada antes de ejecutar, con sus variantes y su estado.

    `walk_forward` es opcional y solo lo interpreta `training.titans_walk_forward`.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise ValueError("La receta no es un archivo regular de hasta 64 KiB")
    document = json.loads(path.read_text(encoding="utf-8"))
    fields = {"schema_version", "recipe_name", "status", "variants", "pairing_source"}
    if (
        not isinstance(document, dict)
        or set(document) - {"walk_forward"} != fields | {"predictor", "recipe", "pending"}
        or document["schema_version"] != 1
        or document["recipe_name"] != RECIPE
        or tuple(document["variants"]) != VARIANTS
        or document["pairing_source"] not in VARIANTS
    ):
        raise ValueError("La receta no conserva su esquema, variantes y emparejamiento")
    return ChronologicalRecipe(**document["recipe"]), document


def parameter_roles(predictor):
    """Separar el ajuste externo por función. Los estados por flujo no son parámetros."""
    roles = dict(shared=[], persistent_memory=[], initial_fast_weights=[])
    for name, _ in predictor.named_parameters():
        if name == "mac.persistent":
            roles["persistent_memory"].append(name)
        elif name.startswith("mac.memory.initial_weights."):
            roles["initial_fast_weights"].append(name)
        else:
            roles["shared"].append(name)
    return roles


def _stack(states):
    """Unir filas por flujo conservando el grafo del tramo diferenciable."""
    first = states[0]
    mac = None
    if first.mac is not None:
        memory = first.mac.memory

        def join(get):
            return torch.cat([get(state.mac.memory) for state in states])

        mac = MACState(
            NeuralMemoryState(
                tuple(join(lambda m, i=i: m.weights[i]) for i in range(len(memory.weights))),
                tuple(join(lambda m, i=i: m.momentum[i]) for i in range(len(memory.momentum))),
                join(lambda m: m.steps),
                memory.config_id,
            ),
            first.mac.config_id,
        )
    return FinancialState(
        first.config_id,
        first.parameter_id,
        tuple(flow for state in states for flow in state.flow_ids),
        tuple(item for state in states for item in state.last_sample_ids),
        tuple(item for state in states for item in state.last_prediction_at),
        torch.cat([state.observed_steps for state in states]),
        mac,
    )


def _split(state, *, detach=False, parameter_id=None):
    """Separar filas. detach corta el BPTT y parameter_id confirma el sellado posterior."""

    def take(value, row):
        piece = value[row : row + 1]
        return piece.detach() if detach else piece

    for row, flow in enumerate(state.flow_ids):
        mac = state.mac
        if mac is not None:
            memory = mac.memory
            mac = MACState(
                NeuralMemoryState(
                    tuple(take(w, row) for w in memory.weights),
                    tuple(take(m, row) for m in memory.momentum),
                    memory.steps[row : row + 1],
                    memory.config_id,
                ),
                mac.config_id,
            )
        yield (
            flow,
            FinancialState(
                state.config_id,
                parameter_id or state.parameter_id,
                (flow,),
                (state.last_sample_ids[row],),
                (state.last_prediction_at[row],),
                state.observed_steps[row : row + 1],
                mac,
            ),
        )


def _compatible(expected, supplied):
    """Las vistas de ajuste y validación comparten fuente y representación, no índice."""
    left, right = expected.identity(), supplied.identity()
    left.pop("view_sha256")
    right.pop("view_sha256")
    return canonical(left) == canonical(right)


class Paused(Exception):
    """Solicitud de parada atendida en una barrera ya confirmada."""


@dataclass
class _Pass:
    """Estado de un recorrido. Solo flows, pending, errores y contadores se persisten."""

    flows: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)
    errors: SessionErrors = field(default_factory=SessionErrors)
    counters: dict = field(
        default_factory=lambda: dict(
            observations=0,
            warmup_observations=0,
            predictions=0,
            labels=0,
            labels_in_loss=0,
            labels_without_graph=0,
            updates=0,
            segments=0,
            loss_sum=0.0,
            unresolved=0,
        )
    )
    graphs: dict = field(default_factory=dict)
    # Inferencia: `rows.append((flujo, decisión, predicción, objetivo, cuantiles))`.
    rows: object = None
    levels: dict = field(default_factory=dict)
    predictions: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    used: list = field(default_factory=list)
    instants: int = 0
    # Con acumulación: lotes del tramo en orden y estado de cada flujo al empezarlo.
    # None indica un flujo nuevo dentro del tramo. Se vacían en cada actualización.
    segment: list = field(default_factory=list)
    starts: dict = field(default_factory=dict)


class ChronologicalTrainer:
    """Ajustar un control de Titans-MAC recorriendo instantes de decisión en orden.

    En cada evento se aplican primero las etiquetas maduras de ese instante. Si el
    tramo diferenciable ya contiene `truncation` instantes de decisión, se actualiza
    con esas etiquetas y se cortan los grafos. Después se predicen las entradas del
    instante con los parámetros vigentes.
    """

    def __init__(
        self,
        predictor,
        recipe,
        *,
        train,
        validation,
        output,
        optimizer_factory=None,
        pairing=None,
        audit=False,
    ):
        if type(predictor) is not FinancialPredictor or type(recipe) is not ChronologicalRecipe:
            raise ValueError("El entrenador necesita el predictor financiero y su receta")
        if predictor.local_control is not None:
            raise ValueError("C pertenece al factorial CM-v1 y no a este entrenador")
        self.quantiles = predictor.config.head == QUANTILE_HEAD
        if self.quantiles != (recipe.loss == PINBALL):
            raise ValueError("La cabeza de cuantiles se ajusta solo y siempre con pinball")
        if torch.backends.mha.get_fastpath_enabled() or torch.is_inference_mode_enabled():
            raise ValueError("El recorrido exige fastpath=False declarado y sin inference_mode")
        if (
            type(train) is not FinancialObservationSource
            or type(validation) is not FinancialObservationSource
            or train.phase.partition != "train"
            or validation.phase.partition != "validation"
            or train.dataset.identity != validation.dataset.identity
            or validation.phase.decision_start < train.phase.decision_end
        ):
            raise ValueError("Se necesitan fases de ajuste y validación ordenadas del mismo corpus")
        specification = predictor.config.inputs
        if any(
            not _compatible(specification, source.specification()) for source in (train, validation)
        ):
            raise ValueError("Las vistas no conservan la entrada del predictor")
        if (
            recipe.block_rows > predictor.config.max_batch
            or (recipe.accumulation_rows or 0) > predictor.config.max_batch
        ):
            raise ValueError("El bloque de activos supera el lote del predictor")
        self.device = predictor.head.weight.device
        if str(self.device) not in {"cpu", "cuda:0"}:
            raise ValueError("El recorrido admite cpu o cuda:0 explícitos")
        if pairing is not None and (
            not isinstance(pairing, dict)
            or pairing.get("target_after") != predictor._parameter_id
            or pairing.get("target_variant") != predictor.config.variant
            or pairing.get("runtime_state_transferred") is not False
        ):
            raise ValueError("El recibo de emparejamiento no corresponde a estos parámetros")
        self.predictor, self.recipe = predictor, recipe
        self.train, self.validation = train, validation
        self.output, self.audit = Path(output), [] if audit else None
        for protected in (*train.dataset.roots.values(), train.path.parent, validation.path.parent):
            outside_source(protected, self.output)
            outside_source(self.output, protected)
        self.roles = parameter_roles(predictor)
        named = dict(predictor.named_parameters())
        groups = [
            dict(params=[named[name] for name in names], role=role)
            for role, names in self.roles.items()
            if names
        ]
        factory = optimizer_factory or (
            lambda values: torch.optim.AdamW(
                values, lr=recipe.learning_rate, weight_decay=recipe.weight_decay
            )
        )
        self.optimizer = factory(groups)
        listed = [id(p) for group in self.optimizer.param_groups for p in group["params"]]
        if len(listed) != len(set(listed)) or set(listed) != {id(p) for p in named.values()}:
            raise ValueError("El optimizador debe cubrir exactamente el ajuste externo")
        self.identity = dict(
            schema_version=1,
            recipe=recipe.identity(),
            predictor=predictor.config.identity(),
            dtype=str(predictor.head.weight.dtype),
            device=str(self.device),
            initial_parameters_sha256=predictor._parameter_id,
            pairing_sha256=None
            if pairing is None
            else hashlib.sha256(canonical(pairing).encode()).hexdigest(),
            parameter_roles=self.roles,
            optimizer=type(self.optimizer).__module__ + "." + type(self.optimizer).__qualname__,
            dataset_sha256=train.dataset.identity,
            sources={
                name: dict(index_sha256=source.identity, phase=asdict(source.phase))
                for name, source in (("train", train), ("validation", validation))
            },
            implementation=self._code(),
            numerics=_numerics(),
            final_test_opened=False,
        )
        self.run_id = hashlib.sha256(canonical(self.identity).encode()).hexdigest()
        self.global_step, self.selection, self.history, self.train_metrics = 0, None, [], None

    @staticmethod
    def _code():
        own = {name: sha256(Path(importlib.import_module(name).__file__)) for name in _OWN_MODULES}
        return {**_implementation(), **own}

    def _check_runtime(self):
        if (
            self._code() != self.identity["implementation"]
            or _numerics() != self.identity["numerics"]
        ):
            raise ValueError("El código o la configuración numérica cambiaron durante el recorrido")

    def _loss(self, prediction, target):
        """Pérdida del tramo. Con la cabeza de cuantiles, `prediction` es [etiquetas, 5]."""
        functional = torch.nn.functional
        if self.recipe.loss == PINBALL:
            return pinball_loss(prediction, target)
        if self.recipe.loss == "mae":
            return functional.l1_loss(prediction, target)
        if self.recipe.loss == "mse":
            return functional.mse_loss(prediction, target)
        return functional.huber_loss(prediction, target, delta=self.recipe.huber_delta)

    def _observe(self, run, source, event, *, differentiable):
        """Predecir cada bloque desde el estado previo de sus flujos, sin efectos cruzados."""
        predictor, specification = self.predictor, self.predictor.config.inputs
        warmup = event.at < source.phase.decision_start
        grad = differentiable and not warmup
        # La acumulación emite con el mismo cálculo y corta el grafo tras cada bloque.
        accumulate = grad and self.recipe.accumulation_rows is not None
        if grad and run.instants == 0 and predictor.config.variant == "mac_frozen":
            self._anchor_frozen(run.flows)
        for raw in event.inputs:
            batch = DecisionBatch.from_validated(
                validated_cpu_batch(raw, specification),
                device=self.device,
                dtype=predictor.head.weight.dtype,
            )
            new = tuple(flow for flow in batch.flow_ids if flow not in run.flows)
            if new:
                run.flows.update(_split(predictor.initial_state(new, differentiable=grad)))
            if accumulate:
                for flow in batch.flow_ids:
                    if flow not in run.starts:
                        run.starts[flow] = None if flow in new else run.flows[flow]
                run.segment.append(batch)
            state = _stack([run.flows[flow] for flow in batch.flow_ids])
            with torch.set_grad_enabled(grad):
                prepared = predictor.prepare(batch, state, differentiable=grad)
            run.flows.update(_split(prepared.next_state, detach=accumulate))
            size = len(batch.flow_ids)
            run.counters["observations"] += size
            if warmup:
                run.counters["warmup_observations"] += size
                continue
            values = prepared.point_predictions.detach().cpu().tolist()
            # El error y la selección usan la mediana emitida. La pérdida usa los cinco niveles.
            graphs = prepared.quantiles if self.quantiles else prepared.point_predictions
            levels = (
                prepared.quantiles.detach().cpu().tolist()
                if run.rows is not None and self.quantiles
                else None
            )
            for row, (flow, at) in enumerate(zip(batch.flow_ids, batch.prediction_at, strict=True)):
                run.pending[flow, at] = values[row]
                if levels is not None:
                    run.levels[flow, at] = levels[row]
                if grad:
                    run.graphs[flow, at] = _REPLAY if accumulate else graphs[row]
                if self.audit is not None:
                    self.audit.append(("prediction", source.phase.partition, flow, at, values[row]))
            run.counters["predictions"] += size
        if event.inputs and not warmup:
            run.instants += 1

    def _anchor_frozen(self, states):
        """Sin escrituras, la memoria de cada flujo es M0. Cada tramo lee el M0 vigente."""
        flows, size = list(states), self.predictor.config.max_batch
        for start in range(0, len(flows), size):
            chunk = flows[start : start + size]
            memory = self.predictor.mac.initial_state(len(chunk), differentiable=True).memory
            for row, flow in enumerate(chunk):
                state = states[flow]
                fresh = NeuralMemoryState(
                    tuple(w[row : row + 1] for w in memory.weights),
                    tuple(m[row : row + 1] for m in memory.momentum),
                    memory.steps[row : row + 1],
                    memory.config_id,
                )
                states[flow] = replace(state, mac=replace(state.mac, memory=fresh))

    def _labels(self, run, event, *, train):
        """Resolver etiquetas maduras contra la predicción emitida y su grafo vigente."""
        markets, moments, errors = [], [], []
        for flow, decision_at, value in event.labels:
            key = flow, decision_at
            if key not in run.pending:
                raise ValueError("El label no tiene una predicción emitida pendiente")
            issued = run.pending.pop(key)
            markets.append(flow.split("/", 1)[0])
            moments.append(decision_at)
            errors.append(issued - value)
            run.counters["labels"] += 1
            if self.audit is not None:
                phase = "train" if train else "validation"
                self.audit.append(("label", phase, flow, decision_at, event.at))
            graph = run.graphs.pop(key, None)
            if run.rows is not None:
                run.rows.append((flow, decision_at, issued, value, run.levels.pop(key, None)))
            if not train:
                continue
            if graph is None:
                run.counters["labels_without_graph"] += 1
                continue
            run.predictions.append(graph)
            run.targets.append(value)
            run.used.append((flow, decision_at, event.at))
        for start in range(0, len(errors), 4096):
            run.errors.update(
                markets[start : start + 4096],
                moments[start : start + 4096],
                errors[start : start + 4096],
            )

    def _backward(self, run):
        """Gradiente de la pérdida media del tramo con los grafos conservados."""
        prediction = torch.stack(run.predictions)
        target = torch.tensor(run.targets, dtype=prediction.dtype, device=prediction.device)
        loss = self._loss(prediction, target)
        if not torch.isfinite(loss).item():
            raise ValueError("La pérdida del tramo no es finita")
        loss.backward()
        return loss

    def _replay(self, run):
        """Recalcular el tramo por bloques de flujos y acumular el gradiente de la misma media.

        Los flujos son independientes dados los parámetros, que no cambian dentro del tramo.
        Cada bloque parte del estado de sus flujos al empezar el tramo, recorre sus lotes en
        el orden original y pondera su pérdida media por su fracción de etiquetas. La suma
        de los bloques es la pérdida media del tramo.
        """
        predictor, size = self.predictor, self.recipe.accumulation_rows
        targets = {
            (flow, at): value for (flow, at, _), value in zip(run.used, run.targets, strict=True)
        }
        order = list(dict.fromkeys(flow for flow, _, _ in run.used))
        total = 0.0
        for start in range(0, len(order), size):
            group = order[start : start + size]
            states = {flow: run.starts[flow] for flow in group if run.starts[flow] is not None}
            if predictor.config.variant == "mac_frozen":
                self._anchor_frozen(states)
            members, graphs = set(group), {}
            for batch in run.segment:
                rows = [i for i, flow in enumerate(batch.flow_ids) if flow in members]
                if not rows:
                    continue
                batch = batch.select(rows)
                new = tuple(flow for flow in batch.flow_ids if flow not in states)
                if new:
                    states.update(_split(predictor.initial_state(new, differentiable=True)))
                state = _stack([states[flow] for flow in batch.flow_ids])
                with torch.enable_grad():
                    prepared = predictor.prepare(batch, state, differentiable=True)
                states.update(_split(prepared.next_state))
                outputs = prepared.quantiles if self.quantiles else prepared.point_predictions
                for row, key in enumerate(zip(batch.flow_ids, batch.prediction_at, strict=True)):
                    if key in targets:
                        graphs[key] = outputs[row]
            keys = [(flow, at) for flow, at, _ in run.used if flow in members]
            if any(key not in graphs for key in keys):
                raise ValueError("La repetición del tramo no reproduce sus predicciones")
            prediction = torch.stack([graphs[key] for key in keys])
            target = torch.tensor(
                [targets[key] for key in keys], dtype=prediction.dtype, device=prediction.device
            )
            loss = self._loss(prediction, target) * (len(keys) / len(run.used))
            if not torch.isfinite(loss).item():
                raise ValueError("La pérdida del tramo no es finita")
            loss.backward()
            total += float(loss.detach())
        return torch.tensor(total)

    def _update(self, run, at):
        """Un paso con las etiquetas maduras del tramo y truncamiento de todos los flujos."""
        if run.predictions:
            loss = self._replay(run) if self.recipe.accumulation_rows else self._backward(run)
            torch.nn.utils.clip_grad_norm_(
                self.predictor.parameters(),
                self.recipe.max_grad_norm or math.inf,
                error_if_nonfinite=True,
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            # El paso es la única modificación admitida de los parámetros. Se sella de nuevo.
            self.predictor._seal_parameters()
            self.global_step += 1
            run.counters["updates"] += 1
            run.counters["labels_in_loss"] += len(run.predictions)
            run.counters["loss_sum"] += float(loss.detach()) * len(run.predictions)
            if self.audit is not None:
                self.audit.append(("update", at, self.global_step, tuple(run.used)))
        parameter_id = self.predictor._parameter_id
        for state in list(run.flows.values()):
            run.flows.update(_split(state, detach=True, parameter_id=parameter_id))
        run.graphs.clear()
        run.predictions.clear()
        run.targets.clear()
        run.used.clear()
        run.segment.clear()
        run.starts.clear()
        run.instants = 0
        run.counters["segments"] += 1

    def _close(self, run):
        run.counters["unresolved"] = len(run.pending)
        run.flows.clear()
        run.pending.clear()
        run.graphs.clear()
        run.levels.clear()
        run.segment.clear()
        run.starts.clear()

    @staticmethod
    def _metrics(run):
        summary = run.errors.summary()
        counters = dict(run.counters)
        labels = counters["labels_in_loss"]
        counters["mean_loss"] = counters.pop("loss_sum") / labels if labels else None
        return dict(
            samples=summary["samples"],
            session_count=summary["session_count"],
            session_mae=summary["session_mae"],
            session_mse=summary["session_mse"],
            **counters,
        )

    def evaluate(self, source, *, stop=None, rows=None):
        """Validación temporal con parámetros congelados y memoria rápida reiniciada.

        Con `rows`, cada etiqueta resuelta añade su predicción emitida y sus cuantiles.
        """
        predictor, run = self.predictor, _Pass(rows=rows)
        predictor.eval()
        with torch.no_grad():
            for event in source.batched_events(block_rows=self.recipe.block_rows):
                if stop is not None and stop.requested:
                    raise Paused
                self._labels(run, event, train=False)
                if event.inputs:
                    self._observe(run, source, event, differentiable=False)
                if event.close_phase:
                    self._close(run)
        if run.counters["labels"] == 0:
            raise ValueError("La validación no contiene etiquetas maduras")
        return self._metrics(run)

    def predict_partition(self, source, rows, *, stop=None):
        """Recorrer un tramo posterior al ajuste con la misma regla que la validación.

        La memoria rápida empieza en su estado inicial, observa el calentamiento de la fase
        sin etiquetas y avanza en orden. Las etiquetas solo se comparan con lo emitido.
        """
        if (
            type(source) is not FinancialObservationSource
            or source.phase.partition not in ("validation", "calibration", "evaluation")
            or source.dataset.identity != self.train.dataset.identity
            or source.phase.decision_start < self.train.phase.decision_end
            or not _compatible(self.predictor.config.inputs, source.specification())
        ):
            raise ValueError("El tramo no pertenece al corpus y la entrada del ajuste")
        if not hasattr(rows, "append"):
            raise ValueError("La inferencia necesita un destino de filas")
        return self.evaluate(source, stop=stop, rows=rows)

    def _train_pass(self, run, cursor, stop, save):
        predictor, source = self.predictor, self.train
        predictor.train()
        start, stage = cursor["event"], cursor["stage"]
        events = source.batched_events(start_cursor=start, block_rows=self.recipe.block_rows)
        since, last = 0, time.perf_counter()
        for index, event in enumerate(events, start):
            if not (index == start and stage == "inputs"):
                self._labels(run, event, train=True)
                decision = bool(event.inputs) and event.at >= source.phase.decision_start
                if (decision and run.instants == self.recipe.truncation) or event.close_phase:
                    before = self.global_step
                    self._update(run, event.at)
                    since += self.global_step - before
                    if event.close_phase:
                        self._close(run)
                        return self._metrics(run)
                    if (
                        stop.requested
                        or since >= self.recipe.checkpoint_updates
                        or time.perf_counter() - last >= self.recipe.checkpoint_seconds
                    ):
                        save(dict(cursor, event=index, stage="inputs"), run)
                        since, last = 0, time.perf_counter()
                        if stop.requested:
                            raise Paused
            if event.inputs:
                self._observe(run, source, event, differentiable=True)
        raise ValueError("El recorrido de ajuste terminó sin su cierre declarado")

    def _export(self, run):
        if run.graphs or run.segment or run.starts:
            raise ValueError("Solo se confirma un recorrido en la barrera posterior a un paso")
        predictor, size = self.predictor, self.predictor.config.max_batch
        flows = sorted(run.flows)
        fast = [
            predictor.export_state_cpu(_stack([run.flows[f] for f in flows[i : i + size]]))
            for i in range(0, len(flows), size)
        ]
        keys = sorted(run.pending)
        return dict(
            fast=fast,
            pending=dict(
                flows=[flow for flow, _ in keys],
                decision_at=torch.tensor([at for _, at in keys], dtype=torch.int64),
                values=torch.tensor([run.pending[key] for key in keys], dtype=torch.float64),
            ),
            errors=[[m, t, *values] for (m, t), values in sorted(run.errors.sessions.items())],
            counters=dict(run.counters),
        )

    def _restore(self, payload):
        run = _Pass()
        for item in payload["fast"]:
            run.flows.update(_split(self.predictor.restore_state(item, device=self.device)))
        pending = payload["pending"]
        run.pending = {
            (flow, int(at)): float(value)
            for flow, at, value in zip(
                pending["flows"],
                pending["decision_at"].tolist(),
                pending["values"].tolist(),
                strict=True,
            )
        }
        run.errors.sessions = {(m, t): [c, a, s] for m, t, c, a, s in payload["errors"]}
        run.counters = dict(payload["counters"])
        return run

    def _load_best(self, report):
        """Dejar el predictor con el estado seleccionado que acredita el informe."""
        state = load_training_state(
            self.output / "checkpoints",
            expected_identity=self.identity,
            selection="best",
            expected_sha256=report["best_checkpoint"]["sha256"],
        )
        self.predictor.load_state_dict(state["model"])
        return state

    def run(self, *, resume=False, stop=None):
        """Recorrer épocas hasta la paciencia declarada o el presupuesto fijo."""
        output, checkpoints = self.output, self.output / "checkpoints"
        if isinstance(self.optimizer, torch.optim.Optimizer):
            require_learning_allowed("ChronologicalTrainer.run de Titans-MAC")
        if type(resume) is not bool or output.is_symlink() or output.exists() != resume:
            raise ValueError("Usa una ejecución nueva o solicita continuar una existente")
        stop = stop or StopRequest()
        report_path = output / "run.json"
        cursor, run = dict(epoch=0, phase="validation"), None
        if resume:
            report = _read_report(report_path)
            if report["identity"] != self.identity:
                raise ValueError("La identidad o configuración de la ejecución ha cambiado")
            state = load_training_state(checkpoints, expected_identity=self.identity)
            self.predictor.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"], str(self.device))
            self.global_step, cursor = state["global_step"], state["cursor"]
            self.selection, self.history = state["selection"], state["history"]
            self.train_metrics = state["train_metrics"]
            if state["run"] is not None:
                self.predictor.train()
                run = self._restore(state["run"])
            if report["status"] == "completed":
                self._load_best(report)
                return report
        else:
            output.mkdir(parents=True)
            report = dict(
                schema_version=1,
                kind="titans_financial_chronological_run",
                run_id=self.run_id,
                identity=self.identity,
                status="running",
                started_at_utc=datetime.now(UTC).isoformat(),
                final_test_opened=False,
                attempts=[],
            )

        def save(position, current=None, *, best=False):
            self._check_runtime()
            state = dict(
                global_step=self.global_step,
                cursor=position,
                model=self.predictor.state_dict(),
                optimizer=self.optimizer.state_dict(),
                rng=capture_rng(str(self.device)),
                selection=self.selection,
                history=self.history,
                train_metrics=self.train_metrics,
                run=None if current is None else self._export(current),
            )
            path = save_training_state(checkpoints, state, identity=self.identity, best=best)
            reference = dict(path=str(path.relative_to(output)), sha256=sha256(path))
            report.update(recovery_checkpoint=reference, global_step=self.global_step)
            if best:
                report["best_checkpoint"] = reference
            report.update(selection=self.selection, history=self.history, cursor=position)
            atomic_json(report_path, report)

        if not resume:
            save(cursor)
        started = time.perf_counter()
        try:
            while cursor["phase"] != "done":
                epoch = cursor["epoch"]
                if cursor["phase"] == "validation":
                    metrics = self.evaluate(self.validation, stop=stop)
                    options = self.recipe.selection
                    self.selection = (
                        initial_selection(metrics["session_mae"], options)
                        if epoch == 0
                        else advance_selection(
                            self.selection, metrics["session_mae"], epoch, options
                        )
                    )
                    self.history.append(
                        dict(
                            epoch=epoch,
                            global_step=self.global_step,
                            train=self.train_metrics,
                            validation=metrics,
                        )
                    )
                    self.train_metrics = None
                    # Con presupuesto fijo, should_stop nunca corta y se registra plateau_epoch.
                    finished = epoch >= self.recipe.epochs or self.selection["should_stop"]
                    cursor = (
                        dict(epoch=epoch, phase="done")
                        if finished
                        else dict(epoch=epoch, phase="train", event=0, stage="start")
                    )
                    save(cursor, best=self.selection["last_improved"])
                    if stop.requested and not finished:
                        raise Paused
                    continue
                self.train_metrics = self._train_pass(run or _Pass(), cursor, stop, save)
                run = None
                cursor = dict(epoch=epoch + 1, phase="validation")
                save(cursor)
            self._load_best(report)
            report.update(
                status="completed",
                stopped_early=cursor["epoch"] < self.recipe.epochs,
                stopping=self.recipe.selection.get("stopping", VALIDATION_PLATEAU),
                plateau_epoch=self.selection.get("plateau_epoch"),
                best_epoch=self.selection["best_epoch"],
                best_score=self.selection["best_score"],
                finished_at_utc=datetime.now(UTC).isoformat(),
            )
            return report
        except Paused:
            report["status"] = "paused"
            return report
        except BaseException as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            report["attempts"].append(dict(seconds=time.perf_counter() - started))
            atomic_json(report_path, report)


def _read_report(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024**2:
        raise ValueError("El informe de la ejecución no es regular o supera 16 MiB")
    return json.loads(path.read_text())
