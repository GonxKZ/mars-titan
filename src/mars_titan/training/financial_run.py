"""Recorrido cronológico de `FinancialPredictor` con BPTT truncado y etiquetas maduras.

El optimizador ajusta los parámetros compartidos, la memoria persistente de MAC y los
pesos rápidos iniciales aprendidos. Los pesos rápidos y el momentum de cada flujo
solo cambian mediante la regla asociativa de Titans al observar entradas. El estado
de trabajo de cada predicción se descarta después de usarla.

Con el control local C de CM-v1, el modo disabled identifica la B del factorial (SDPA
Math y la misma base) y el modo penalty suma al objetivo del tramo la media de los
términos de sus grupos lógicos medidos, sin añadir pasos. Con `accumulation_rows`, cada
bloque de flujos añade además sus términos de C divididos por el número de grupos del
tramo. El diagnóstico se rechaza.
"""

import hashlib
import importlib
import json
import math
import time
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from mars_titan import nvtx_ranges
from mars_titan.budget_training import validate_loss
from mars_titan.data.input_policy import MODALITIES
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.session_metrics import SessionErrors
from mars_titan.learning_traces.hooks import observing, traced_update
from mars_titan.memory.financial_observations import FinancialObservationSource
from mars_titan.models.predictive_adaptation import is_adapter_name
from mars_titan.models.quantile_head import PINBALL, QUANTILE_HEAD, pinball_loss
from mars_titan.models.titans.auxiliary_horizons import auxiliary_errors
from mars_titan.models.titans.config import MAX_BLOCK_ROWS, canonical
from mars_titan.models.titans.financial import VARIANTS, FinancialPredictor, FinancialState
from mars_titan.models.titans.financial_inputs import (
    DecisionBatch,
    device_tensor,
    validated_cpu_batches,
)
from mars_titan.models.titans.frozen_financial import _implementation, _numerics
from mars_titan.models.titans.state import (
    deferred_checks,
    join_mac_rows,
    map_mac_rows,
    require_true,
)

from . import anchored_decay, distillation
from .anchored_decay import ANCHORS
from .checkpoints import (
    StopRequest,
    capture_rng,
    load_training_state,
    restore_rng,
    save_training_state,
)
from .common_epoch import chosen_state
from .horizon_labels import HorizonLabels
from .input_pipeline import prepared
from .kernel_policy import declared_policy, require_policy, valid_precision
from .learning_hold import require_learning_allowed
from .quantile_objectives import POINT_OBJECTIVES, QUANTILE_OBJECTIVES, point_objective
from .schedule_free import AveragedPoint, ScheduleFree, is_schedule_free
from .selection import (
    AWAIT,
    CHRONOLOGICAL_MODES,
    COMMON_EPOCH,
    FINISH,
    VALIDATION_PLATEAU,
    advance_selection,
    awaiting,
    bind_joint_epoch,
    epoch_decision,
    initial_selection,
    validate_selection,
)
from .shrink_perturb import ShrinkPerturb, shrink_perturb
from .weight_averaging import ShadowWeights, WeightAverage, applied

RECIPE = "titans_financial_chronological_v1"
_OWN_MODULES = (
    "mars_titan.training.financial_run",
    "mars_titan.memory.financial_observations",
    "mars_titan.training.checkpoints",
    "mars_titan.training.selection",
    "mars_titan.training.common_epoch",
    "mars_titan.training.quantile_objectives",
    "mars_titan.training.horizon_labels",
    "mars_titan.evaluation.session_metrics",
)
# F2 y F3 de #749, en el orden de la receta. Su código entra en la identidad si se declaran.
_WEIGHT_OPTIONS = dict(
    weight_average=WeightAverage, schedule_free=ScheduleFree, shrink_perturb=ShrinkPerturb
)
_WEIGHT_MODULES = dict(
    weight_average="mars_titan.training.weight_averaging",
    schedule_free="mars_titan.training.schedule_free",
    shrink_perturb="mars_titan.training.shrink_perturb",
)


# Marca de una predicción del tramo cuyo grafo se recalcula al actualizar.
_REPLAY = object()
# Contadores de F5. Solo aparecen si el predictor declara las cabezas auxiliares.
_AUXILIARY_COUNTERS = (
    "auxiliary_labels",
    "auxiliary_labels_in_loss",
    "auxiliary_labels_discarded",
    "auxiliary_loss_sum",
)


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
    # None conserva la configuración numérica del proceso. Una precisión declarada aplica
    # `kernel_policy` al construir el recorrido y entra en su identidad.
    precision: str | None = None
    # None conserva el decaimiento de AdamW hacia cero. `initial_parameters` lo ancla al
    # estado con que empieza el ajuste, que en el postentrenamiento es el padre (#444).
    weight_decay_anchor: str | None = None
    # None ajusta solo con las etiquetas. Una fracción w suma la distancia a las salidas de un
    # profesor en las filas de ajuste con peso w y deja 1 − w a la tarea (#57).
    distillation_weight: float | None = None
    # None valida con los pesos del ajuste. Un objeto declara la media exponencial de F2
    # (#749), con la que se valida, se elige y se emite.
    weight_average: dict | None = None
    # None ajusta con AdamW. Un objeto declara AdamW sin calendario de F3 (#749).
    schedule_free: dict | None = None
    # None empieza con los pesos recibidos. Un objeto declara la contracción y el ruido de
    # F3 (#749), aplicados una vez antes del primer paso de una ejecución nueva.
    shrink_perturb: dict | None = None

    def __post_init__(self):
        # Los controles puntuales conservan la cabeza y validan los mismos hiperparámetros.
        validate_loss("mae" if self.loss in QUANTILE_OBJECTIVES else self.loss, self.huber_delta)
        validate_selection(self.selection, epochs=self.epochs, modes=CHRONOLOGICAL_MODES)
        distillation.checked_weight(self.distillation_weight)
        if self.loss in POINT_OBJECTIVES and self.distillation_weight is not None:
            raise ValueError("El control de solo mediana no admite destilación de las colas")
        numbers = (self.learning_rate, self.weight_decay, self.checkpoint_seconds)
        if (
            any(type(v) is not int or not 1 <= v <= m for v, m in self._bounded())
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers)
            or not 0 < self.learning_rate <= 1
            or self.weight_decay < 0
            or self.checkpoint_seconds <= 0
            or not valid_precision(self.precision)
            or self.weight_decay_anchor not in ANCHORS
            or (
                self.accumulation_rows is not None
                and (
                    type(self.accumulation_rows) is not int
                    or not 1 <= self.accumulation_rows <= MAX_BLOCK_ROWS
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
        self.weight_options()

    def weight_options(self):
        """Configuraciones de F2 y F3 sobre los pesos del ajuste, o None si no se declaran."""
        options = {}
        for name, kind in _WEIGHT_OPTIONS.items():
            value = getattr(self, name)
            if value is not None and (type(value) is not dict or set(value) != set(asdict(kind()))):
                raise ValueError(f"{name} debe declarar toda su configuración")
            options[name] = None if value is None else kind(**value)
        if options["weight_average"] is not None and options["schedule_free"] is not None:
            raise ValueError("La media exponencial y el ajuste sin calendario promedian dos veces")
        if self.weight_decay_anchor is not None and (
            options["schedule_free"] is not None or options["shrink_perturb"] is not None
        ):
            raise ValueError("El decaimiento anclado solo se combina con AdamW sin contracción")
        return options

    def _bounded(self):
        return (
            (self.truncation, 256),
            (self.epochs, 1000),
            (self.block_rows, MAX_BLOCK_ROWS),
            (self.checkpoint_updates, 1_000_000),
        )

    def identity(self):
        fields = asdict(self)
        # Sin acumulación se conserva literalmente la identidad anterior y su huella.
        if fields["accumulation_rows"] is None:
            fields.pop("accumulation_rows")
        else:
            fields["gradient_accumulation"] = "flow_blocks_replayed_from_segment_start_v1"
        if fields["precision"] is None:
            fields.pop("precision")
        if fields["weight_decay_anchor"] is None:
            fields.pop("weight_decay_anchor")
        if fields["distillation_weight"] is None:
            fields.pop("distillation_weight")
        else:
            fields["distillation_distance"] = distillation.DISTANCE
        # F2 y F3 solo aparecen si se declaran, con la identidad de su configuración.
        options = self.weight_options()
        for name, option in options.items():
            if option is None:
                fields.pop(name)
            else:
                fields[name] = option.identity()
        return dict(
            schema_version=1,
            recipe=RECIPE,
            **fields,
            optimizer="AdamW" if options["schedule_free"] is None else "AdamWScheduleFree",
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
    """Separar el ajuste externo por función. Los estados por flujo no son parámetros.

    Las correcciones de un adaptador del postentrenamiento forman su propio papel. Sin
    adaptadores el diccionario conserva literalmente sus tres papeles anteriores.
    """
    roles = dict(shared=[], persistent_memory=[], initial_fast_weights=[])
    for name, _ in predictor.named_parameters():
        if is_adapter_name(name):
            roles.setdefault("adapters", []).append(name)
        elif name == "mac.persistent":
            roles["persistent_memory"].append(name)
        elif name.startswith("mac.memory.initial_weights."):
            roles["initial_fast_weights"].append(name)
        else:
            roles["shared"].append(name)
    return roles


class _Block:
    """Estado por bloques producido por una llamada, con los contadores ya en Python."""

    __slots__ = ("state", "steps")

    def __init__(self, state):
        self.state = state
        self.steps = state.observed_steps.tolist()


def _detached(state, parameter_id=None):
    """El mismo estado sin grafo. parameter_id confirma el sellado posterior a un paso."""
    mac = state.mac
    if mac is not None:
        # Todos los tensores del estado MAC, también las ventanas de la convolución causal.
        mac = map_mac_rows(mac, lambda value: value.detach())
    return replace(state, parameter_id=parameter_id or state.parameter_id, mac=mac)


class FlowStates:
    """Estado rápido de cada flujo, guardado en el bloque que lo produjo.

    Cada flujo apunta a su fila dentro del último estado por bloques que lo actualizó.
    `gather` devuelve ese mismo bloque si las filas pedidas lo cubren completo y en orden, y
    si no concatena los tramos contiguos de cada bloque de origen. Separar el estado en una
    vista por fila creaba un nodo de autograd por fila, cuyo backward rellenaba con ceros un
    gradiente del tamaño del bloque, y el coste crecía con el cuadrado de las filas. Los
    valores y los gradientes de cada fila no cambian: solo se copian o se reutilizan.
    """

    def __init__(self, rows=None):
        # Cada flujo apunta a su bloque y a su fila dentro de él. El diccionario conserva el
        # orden en que aparecen los flujos, que es el que usan los recorridos.
        self._rows = {} if rows is None else dict(rows)

    def __contains__(self, flow):
        return flow in self._rows

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __bool__(self):
        return bool(self._rows)

    def clear(self):
        self._rows.clear()

    def values(self):
        """Estados por bloques que conservan al menos un flujo vigente, sin repetirlos."""
        blocks = {id(block): block for block, _ in self._rows.values()}
        return [block.state for block in blocks.values()]

    def handle(self, flow):
        return self._rows[flow]

    def subset(self, flows):
        return FlowStates((flow, self._rows[flow]) for flow in flows)

    def steps(self, flow):
        block, row = self._rows[flow]
        return block.steps[row]

    def put(self, state, *, detach=False):
        """Registrar cada fila de un estado por bloques como la vigente de su flujo."""
        block = _Block(_detached(state) if detach else state)
        for row, flow in enumerate(state.flow_ids):
            self._rows[flow] = (block, row)

    def detach(self, parameter_id=None):
        """Cortar el BPTT de todos los flujos. parameter_id confirma el sellado posterior."""
        replaced = {}
        for flow, (block, row) in self._rows.items():
            fresh = replaced.get(id(block))
            if fresh is None:
                fresh = replaced[id(block)] = _Block(_detached(block.state, parameter_id))
            self._rows[flow] = (fresh, row)

    def gather(self, flows):
        """Estado por bloques de `flows` en ese orden, conservando el grafo del tramo."""
        entries = [self._rows[flow] for flow in flows]
        first = entries[0][0]
        if len(entries) == len(first.steps) and all(
            block is first and row == index for index, (block, row) in enumerate(entries)
        ):
            return first.state
        runs = []
        for block, row in entries:
            if runs and runs[-1][0] is block and runs[-1][2] == row:
                runs[-1][2] += 1
            else:
                runs.append([block, row, row + 1])

        state, mac = first.state, None
        if state.mac is not None:
            parts = [
                map_mac_rows(block.state.mac, lambda value, a=start, b=stop: value[a:b])
                for block, start, stop in runs
            ]
            mac = parts[0] if len(parts) == 1 else join_mac_rows(parts)
        return FinancialState(
            state.config_id,
            state.parameter_id,
            tuple(flows),
            tuple(block.state.last_sample_ids[row] for block, row in entries),
            tuple(block.state.last_prediction_at[row] for block, row in entries),
            torch.tensor([block.steps[row] for block, row in entries], dtype=torch.int64),
            mac,
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

    flows: FlowStates = field(default_factory=FlowStates)
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
    # Con acumulación: lotes del tramo en orden, con el plan de C de su evento, y estado de
    # cada flujo al empezarlo. None indica un flujo nuevo dentro del tramo. Se vacían en
    # cada actualización.
    segment: list = field(default_factory=list)
    starts: dict = field(default_factory=dict)
    # Con la penalización C, un elemento por grupo lógico medido del tramo: su término con
    # grafo o, con acumulación, la tupla de sus flujos medidos, que la repetición recalcula.
    penalties: list = field(default_factory=list)
    # Con PT4, el contexto de mercado del recorrido. Entre eventos no guarda nada pendiente.
    context: object = None
    # Con F5, el registro de etiquetas auxiliares del recorrido de ajuste.
    horizons: object = None


class ChronologicalInference:
    """Recorrer las fases de un predictor por instantes con la política de memoria declarada.

    Cada recorrido empieza con el estado inicial de cada flujo, observa el calentamiento de
    la fase sin etiquetas y emite cada predicción antes de conocer su resultado. El ajuste
    añade a este recorrido el grafo, la pérdida y el paso del optimizador.
    """

    def __init__(self, predictor, recipe, *, audit=False):
        if type(predictor) is not FinancialPredictor or type(recipe) is not ChronologicalRecipe:
            raise ValueError("El entrenador necesita el predictor financiero y su receta")
        control = predictor.local_control
        if control is not None and control.config.mode == "diagnostic":
            raise ValueError("El diagnóstico C pertenece a la sesión congelada, no al ajuste")
        # Solo la penalización necesita un plan de C por evento. disabled conserva SDPA Math.
        self.penalized = control is not None and control.config.mode == "penalty"
        # La repetición por bloques de la acumulación no reconstruye el contexto de PT4.
        self.uses_context = predictor.session_context is not None
        if self.uses_context and recipe.accumulation_rows is not None:
            raise ValueError("PT4 no se combina con la acumulación por bloques")
        self.quantiles = predictor.config.head == QUANTILE_HEAD
        if self.quantiles != (recipe.loss in QUANTILE_OBJECTIVES):
            raise ValueError(
                "La cabeza de cuantiles requiere pinball o un control puntual identificado"
            )
        # Un objetivo de salida de #749 sustituye la pérdida MAE de la cabeza escalar, que es
        # su control declarado. La destilación supone las salidas escalares o de cuantiles.
        self.objective = predictor.output_objective
        if self.objective is not None and (
            recipe.loss != "mae" or recipe.distillation_weight is not None
        ):
            raise ValueError("El objetivo de salida sustituye la pérdida mae sin destilación")
        # F1 y A4 son otros objetivos de salida. Solo se distinguen para rechazar el
        # postentrenamiento, que todavía no tiene un contraste declarado para ellos.
        self.undeclared_posttraining = (
            predictor.config.laplace_scale_head is not None
            or predictor.config.opponent_heads is not None
        )
        # Las filas emitidas llevan cinco niveles con la cabeza de cuantiles o con F1.
        self.emits_levels = self.quantiles or hasattr(self.objective, "levels")
        # F5 solo cambia el ajuste. La inferencia y la validación no lo usan.
        self.horizons = predictor.config.auxiliary_horizons
        if torch.backends.mha.get_fastpath_enabled() or torch.is_inference_mode_enabled():
            raise ValueError("El recorrido exige fastpath=False declarado y sin inference_mode")
        if (
            recipe.block_rows > predictor.config.max_batch
            or (recipe.accumulation_rows or 0) > predictor.config.max_batch
        ):
            raise ValueError("El bloque de activos supera el lote del predictor")
        self.device = predictor.head.weight.device
        if str(self.device) not in {"cpu", "cuda:0"}:
            raise ValueError("El recorrido admite cpu o cuda:0 explícitos")
        self.predictor, self.recipe = predictor, recipe
        self.audit = [] if audit else None
        # La política se aplica antes de calcular nada y antes de registrar `_numerics`.
        self.kernel_policy = declared_policy(recipe.precision)

    def _observe(self, run, source, event, *, differentiable, blocks=None):
        """Predecir cada bloque desde el estado previo de sus flujos, sin efectos cruzados.

        `blocks` es el resultado de validar los bloques del evento con `prepared`, adelantado
        en el hilo del prefetch. Sin él, los bloques se validan aquí.
        """
        predictor, specification = self.predictor, self.predictor.config.inputs
        warmup = event.at < source.phase.decision_start
        grad = differentiable and not warmup
        # La acumulación emite con el mismo cálculo y corta el grafo tras cada bloque.
        accumulate = grad and self.recipe.accumulation_rows is not None
        if grad and run.instants == 0 and predictor.config.variant == "mac_frozen":
            self._anchor_frozen(run.flows)
        with nvtx_ranges.phase("titans.validate"):
            if blocks is None:
                validated = validated_cpu_batches(event.inputs, specification)
            else:
                validated = blocks.result()
        with nvtx_ranges.phase("titans.control_plan"):
            plan = self._control_plan(run, source, event, validated) if self.penalized else {}
        measured, emitted = [], []
        ledger = None
        if differentiable and self.horizons is not None:
            if run.horizons is None:
                run.horizons = HorizonLabels(self.horizons)
                for key in _AUXILIARY_COUNTERS:
                    run.counters.setdefault(key, 0)
            ledger = run.horizons
            ledger.observe(flow.split("/", 1)[0] for cpu in validated for flow in cpu.flow_ids)
        # Las operaciones por sesión solo admiten el instante completo en un único bloque.
        complete = len(validated) == 1
        if self.uses_context and run.context is None:
            run.context = predictor.initial_context(differentiable=grad)
        context = dict(context=run.context) if self.uses_context else {}
        # Las comprobaciones de finitud de todos los bloques del evento se resuelven juntas
        # al final, antes de volver al recorrido, que es el único que aplica pasos, guarda
        # checkpoints o cierra recibos. Así no se espera a la GPU tras cada bloque.
        with deferred_checks():
            for cpu in validated:
                batch = DecisionBatch.from_validated(
                    cpu, device=self.device, dtype=predictor.head.weight.dtype
                )
                new = tuple(flow for flow in batch.flow_ids if flow not in run.flows)
                if new:
                    run.flows.put(predictor.initial_state(new, differentiable=grad))
                if accumulate:
                    for flow in batch.flow_ids:
                        if flow not in run.starts:
                            # La repetición parte del estado sin grafo de cada flujo al empezar.
                            run.starts[flow] = None if flow in new else run.flows.handle(flow)
                    run.segment.append((batch, plan, complete))
                state = run.flows.gather(batch.flow_ids)
                with nvtx_ranges.phase("titans.prepare"), torch.set_grad_enabled(grad):
                    prepared = predictor.prepare(
                        batch,
                        state,
                        differentiable=grad,
                        session_complete=complete,
                        **plan,
                        **context,
                    )
                if self.uses_context:
                    # Los bloques siguientes del evento leen el mismo contexto confirmado.
                    context = dict(context=prepared.next_context)
                result = prepared.local_control
                if result is not None:
                    # Con acumulación el término emitido solo cuenta: su grafo retendría el lote.
                    term = result.penalty.detach() if accumulate else result.penalty
                    measured.append((term, result.flow_ids))
                run.flows.put(prepared.next_state, detach=accumulate)
                size = len(batch.flow_ids)
                run.counters["observations"] += size
                if warmup:
                    run.counters["warmup_observations"] += size
                    continue
                # El error y la selección usan la mediana emitida. La pérdida usa los cinco niveles.
                # `unbind` deja un solo nodo de autograd para las filas del bloque.
                graphs = self._graphs(prepared).unbind()
                if grad:
                    for row, key in enumerate(
                        zip(batch.flow_ids, batch.prediction_at, strict=True)
                    ):
                        run.graphs[key] = _REPLAY if accumulate else graphs[row]
                emitted.append((batch, prepared))
                run.counters["predictions"] += size
                if ledger is not None:
                    ledger.remember(cpu)
            if self.uses_context:
                # La sesión se escribe una vez, después de emitir todas sus predicciones.
                with nvtx_ranges.phase("titans.context"), torch.set_grad_enabled(grad):
                    run.context = predictor.commit_context(context["context"], differentiable=grad)
            with nvtx_ranges.phase("titans.emit"):
                self._emit(run, source, emitted)
            if measured:
                with nvtx_ranges.phase("titans.penalty"):
                    self._penalty(run, measured, keep=grad, replay=accumulate)
        if event.inputs and not warmup:
            run.instants += 1

    def _graphs(self, prepared):
        """Salidas por fila que entran en la pérdida del tramo."""
        if self.quantiles:
            return prepared.quantiles
        if self.objective is not None:
            return prepared.outputs
        return prepared.point_predictions

    def _levels(self, prepared):
        if self.quantiles:
            return prepared.quantiles.detach()
        return self.objective.levels(prepared.outputs)

    def _emit(self, run, source, emitted):
        """Registrar las predicciones emitidas del evento con una sola copia al anfitrión.

        Copiar cada bloque obligaba a esperar a la GPU antes de preparar el siguiente.
        Las predicciones solo se consultan al llegar sus etiquetas, en eventos posteriores.
        """
        if not emitted:
            return
        values = torch.cat(
            [prepared.point_predictions.detach() for _, prepared in emitted]
        ).tolist()
        levels = None
        if run.rows is not None and self.emits_levels:
            levels = torch.cat([self._levels(prepared) for _, prepared in emitted]).tolist()
        keys = [
            key
            for batch, _ in emitted
            for key in zip(batch.flow_ids, batch.prediction_at, strict=True)
        ]
        for index, (flow, at) in enumerate(keys):
            run.pending[flow, at] = values[index]
            if levels is not None:
                run.levels[flow, at] = levels[index]
            if self.audit is not None:
                self.audit.append(("prediction", source.phase.partition, flow, at, values[index]))

    def _control_plan(self, run, source, event, batches):
        """Fijar la selección de C sobre todos los flujos del evento antes de dividirlo.

        El grupo lógico es el evento completo y el contexto identifica parámetros, tramo e
        instante, así que dividir en bloques o reanudar no cambia los flujos medidos.
        """
        flows = tuple(flow for batch in batches for flow in batch.flow_ids)
        steps = torch.tensor(
            [run.flows.steps(f) if f in run.flows else 0 for f in flows],
            dtype=torch.int64,
            device="cpu",
        )
        context = hashlib.sha256(
            canonical(
                dict(
                    schema_version=1,
                    group="chronological_decision_event_v1",
                    parameters=self.predictor._parameter_id,
                    index=source.identity,
                    partition=source.phase.partition,
                    at=event.at,
                )
            ).encode()
        ).hexdigest()
        selection = self.predictor.local_control.select_flows(flows, steps, context_id=context)
        return dict(control_selection=selection, control_context_id=context)

    @staticmethod
    def _penalty(run, measured, *, keep, replay=False):
        """Sumar los bloques `(término, flujos)` del grupo. Su suma es la media declarada.

        Con `replay` el tramo conserva los flujos medidos del grupo en lugar del término,
        que `_replay` vuelve a calcular con grafo en el bloque de cada flujo.
        """
        total = sum(term for term, _ in measured)
        counters = run.counters
        for key in ("control_groups", "control_groups_kept", "control_flows"):
            counters.setdefault(key, 0)
        counters.setdefault("control_penalty_sum", 0.0)
        counters["control_groups"] += 1
        counters["control_flows"] += sum(len(flows) for _, flows in measured)
        counters["control_penalty_sum"] += float(total.detach())
        if keep:
            counters["control_groups_kept"] += 1
            run.penalties.append(
                tuple(flow for _, flows in measured for flow in flows) if replay else total
            )

    def _anchor_frozen(self, states):
        """Sin escrituras, la memoria de cada flujo es M0. Cada tramo lee el M0 vigente."""
        flows, size = list(states), self.predictor.config.max_batch
        for start in range(0, len(flows), size):
            chunk = flows[start : start + size]
            memory = self.predictor.mac.initial_state(len(chunk), differentiable=True).memory
            state = states.gather(chunk)
            states.put(replace(state, mac=replace(state.mac, memory=memory)))

    def _labels(self, run, event, *, train):
        """Resolver etiquetas maduras contra la predicción emitida y su grafo vigente."""
        markets, moments, errors = [], [], []
        for flow, decision_at, value in event.labels:
            key = flow, decision_at
            if key not in run.pending:
                raise ValueError("El label no tiene una predicción emitida pendiente")
            issued = run.pending.pop(key)
            if train and run.horizons is not None:
                run.counters["auxiliary_labels"] += run.horizons.mature(flow, decision_at, value)
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

    def _close(self, run):
        run.counters["unresolved"] = len(run.pending)
        run.flows.clear()
        run.context = None
        run.horizons = None
        run.pending.clear()
        run.graphs.clear()
        run.levels.clear()
        run.segment.clear()
        run.starts.clear()
        run.penalties.clear()

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

    def predict(self, source, rows, *, stop=None):
        """Predecir un tramo medido con la misma regla que la validación del ajuste.

        La memoria rápida empieza en su estado inicial, observa el calentamiento de la fase
        sin etiquetas y avanza en orden. Las etiquetas solo se comparan con lo emitido.
        """
        if (
            type(source) is not FinancialObservationSource
            or source.phase.partition not in ("validation", "calibration", "evaluation")
            or not _compatible(self.predictor.config.inputs, source.specification())
        ):
            raise ValueError("El tramo no pertenece a la entrada del predictor")
        if not hasattr(rows, "append"):
            raise ValueError("La inferencia necesita un destino de filas")
        return self.evaluate(source, stop=stop, rows=rows)


class ChronologicalTrainer(ChronologicalInference):
    """Ajustar un control de Titans-MAC recorriendo instantes de decisión en orden.

    En cada evento se aplican primero las etiquetas maduras de ese instante. Si el
    tramo diferenciable ya contiene `truncation` instantes de decisión, se actualiza
    con esas etiquetas y se cortan los grafos. Después se predicen las entradas del
    instante con los parámetros vigentes.
    """

    # Gancho opcional para las trazas de #448. Se llama como `trace(event, modules)` después
    # de cada validación completa y no debe cambiar el cálculo ni el RNG. Si vale None, el
    # recorrido no llama a nada y es el mismo de antes bit a bit.
    trace = None
    # Trazas por paso con el contrato común (`learning_traces.hooks.TrainingTraces`). Con
    # None, `traced_update` y `observing` son bloques vacíos.
    traces = None
    # Puntuación de cada validación con la parada `common_epoch` (#746), que la exige. La
    # asigna la ventana con un `common_epoch.EpochValidation`.
    epoch_scorer = None

    def _trace_modules(self):
        """Módulos del recorrido para las trazas. Solo se agrupan los parámetros ajustables."""
        return dict(predictor=self.predictor)

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
        posttraining=None,
        teacher=None,
        audit=False,
    ):
        super().__init__(predictor, recipe, audit=audit)
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
        if pairing is not None and (
            not isinstance(pairing, dict)
            or pairing.get("target_after") != predictor._parameter_id
            or pairing.get("target_variant") != predictor.config.variant
            or pairing.get("runtime_state_transferred") is not False
        ):
            raise ValueError("El recibo de emparejamiento no corresponde a estos parámetros")
        # Las salidas del profesor solo entran en la pérdida y deben ser de este tramo (#57).
        if (recipe.distillation_weight is None) != (teacher is None) or (
            teacher is not None
            and (
                type(teacher) is not distillation.TeacherTargets
                or teacher.quantiles != self.quantiles
                or teacher.source != distillation.bound_source(train)
            )
        ):
            raise ValueError("La destilación necesita las salidas del profesor de este tramo")
        self.teacher = teacher
        self.train, self.validation = train, validation
        self.output = Path(output)
        for protected in (*train.dataset.roots.values(), train.path.parent, validation.path.parent):
            outside_source(protected, self.output)
            outside_source(self.output, protected)
        if posttraining is not None and (
            not isinstance(posttraining, dict) or pairing is not None or not posttraining
        ):
            raise ValueError("El postentrenamiento se declara como un objeto sin emparejamiento")
        if posttraining is not None and (self.undeclared_posttraining or self.horizons is not None):
            raise ValueError("F1, F5 y A4 no tienen todavía un postentrenamiento declarado")
        options = recipe.weight_options()
        self.weight_average = options["weight_average"]
        self.schedule_free = options["schedule_free"]
        self.shrink = options["shrink_perturb"]
        # Con F2 o F3 se emiten otros pesos que los del ajuste y el estado guarda ambos.
        self.emits_other_weights = self.weight_average is not None or self.schedule_free is not None
        if posttraining is not None and any(option is not None for option in options.values()):
            raise ValueError("F2 y F3 no tienen todavía un postentrenamiento declarado")
        if self.shrink is not None and (
            predictor.local_control is not None or self.shrink.seed == predictor.config.seed
        ):
            raise ValueError("F3 contrae sin control local y con θ₀ de otra semilla")
        self.roles = parameter_roles(predictor)
        if "adapters" in self.roles and posttraining is None:
            raise ValueError("Un predictor con adaptadores solo se ajusta en el postentrenamiento")
        named = dict(predictor.named_parameters())
        frozen = []
        if posttraining is not None:
            # Solo se ajusta lo que requiere gradiente: los adaptadores o, en la continuación
            # completa, todo el ajuste externo. Los tensores originales quedan congelados.
            frozen = [name for name, value in named.items() if not value.requires_grad]
            self.roles = {
                role: [name for name in names if named[name].requires_grad]
                for role, names in self.roles.items()
            }
        groups = [
            dict(params=[named[name] for name in names], role=role)
            for role, names in self.roles.items()
            if names
        ]
        if optimizer_factory is None and self.schedule_free is not None:

            def optimizer_factory(values):
                return self.schedule_free.build(
                    values, learning_rate=recipe.learning_rate, weight_decay=recipe.weight_decay
                )

        factory = optimizer_factory or anchored_decay.recipe_factory(recipe)
        self.optimizer = factory(groups)
        listed = [id(p) for group in self.optimizer.param_groups for p in group["params"]]
        self.trainable = {name: named[name] for names in self.roles.values() for name in names}
        trainable = {id(value) for value in self.trainable.values()}
        if len(listed) != len(set(listed)) or set(listed) != trainable or not trainable:
            raise ValueError("El optimizador debe cubrir exactamente el ajuste externo")
        if isinstance(self.optimizer, torch.optim.Optimizer) and is_schedule_free(
            self.optimizer
        ) != (self.schedule_free is not None):
            raise ValueError("El optimizador no es el que declara la receta para F3")
        # Con F2 la sombra se crea al empezar o reanudar, después de la contracción de F3.
        self.shadow = None
        self.averaged_point = (
            None if self.schedule_free is None else AveragedPoint(self.optimizer, self.trainable)
        )
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
        if self.kernel_policy is not None:
            # Sin precisión declarada la identidad conserva su forma anterior.
            self.identity["kernel_policy"] = self.kernel_policy
        control = predictor.local_control
        if control is not None:
            # Sin control local, la identidad conserva literalmente su forma anterior.
            self.identity["local_control"] = dict(
                configuration=control.config.identity(),
                basis_sha256=control.get_extra_state()["basis_sha256"],
                objective=(
                    "task_mean_plus_mean_of_measured_group_penalties_of_segment"
                    if self.penalized
                    else "task_only"
                ),
                group="chronological_decision_event_v1",
                penalty_without_labels="discarded_without_step",
            )
            if self.penalized and recipe.accumulation_rows is not None:
                # Combinación nueva: no altera la identidad de ninguna ejecución anterior.
                self.identity["local_control"]["penalty_accumulation"] = (
                    "measured_flows_replayed_in_their_flow_block_over_segment_groups_v1"
                )
        if posttraining is not None:
            # Sin postentrenamiento la identidad conserva literalmente su forma anterior.
            self.identity.update(posttraining=posttraining, frozen_parameters=frozen)
        if teacher is not None:
            # Sin destilación la identidad conserva literalmente su forma anterior.
            self.identity["distillation"] = dict(
                teacher.identity,
                weight=recipe.distillation_weight,
                implementation_sha256=sha256(Path(distillation.__file__)),
            )
        anchored = anchored_decay.describe(self.optimizer, recipe.weight_decay_anchor)
        if anchored is not None:
            # Solo el decaimiento anclado añade su código. Sin él la identidad no cambia.
            self.identity["anchored_decay"] = anchored
        self.run_id = hashlib.sha256(canonical(self.identity).encode()).hexdigest()
        self.global_step, self.selection, self.history, self.train_metrics = 0, None, [], None

    def _code(self):
        own = {name: sha256(Path(importlib.import_module(name).__file__)) for name in _OWN_MODULES}
        if self.predictor.config.regional_variant is not None:
            name = "mars_titan.models.titans.regional_variants"
            own[name] = sha256(Path(importlib.import_module(name).__file__))
        # C5 solo entra en la identidad cuando está activa, igual que las variantes regionales.
        if self.predictor.config.cross_asset_attention is not None:
            name = "mars_titan.models.titans.cross_asset_attention"
            own[name] = sha256(Path(importlib.import_module(name).__file__))
        # H4 de #783 añade el módulo del predictor y el de las medidas de la ventana.
        if self.predictor.config.asset_groups is not None:
            for name in (
                "mars_titan.models.titans.asset_groups",
                "mars_titan.data.asset_groups",
            ):
                own[name] = sha256(Path(importlib.import_module(name).__file__))
        # H1 de #783 también, con el catálogo macro del que salen sus familias.
        if self.predictor.config.holographic_network is not None:
            module = importlib.import_module("mars_titan.models.titans.holographic_network")
            own[module.__name__] = sha256(Path(module.__file__))
            own["data/catalogs/macro-indicators.csv"] = sha256(module.CATALOG)
        for option, name in _WEIGHT_MODULES.items():
            # F2 y F3 solo añaden su código si la receta los declara.
            if getattr(self.recipe, option) is not None:
                own[name] = sha256(Path(importlib.import_module(name).__file__))
        return {
            **_implementation(kronlstm=self.predictor.config.temporal_encoder is not None),
            **own,
        }

    def _check_runtime(self):
        if (
            self._code() != self.identity["implementation"]
            or _numerics() != self.identity["numerics"]
        ):
            raise ValueError("El código o la configuración numérica cambiaron durante el recorrido")
        require_policy(self.recipe.precision, self.kernel_policy)

    def _task_loss(self, prediction, target, keys):
        """Pérdida del tramo. Con la cabeza de cuantiles, `prediction` es [etiquetas, 5].

        Con un objetivo de salida, `prediction` son sus salidas y `keys` los pares (flujo,
        decisión) de cada fila, que identifican su sesión (mercado e instante).
        """
        if self.objective is not None:
            sessions = [(flow.split("/", 1)[0], at) for flow, at in keys]
            return self.objective.loss(prediction, target, sessions)
        return self._loss(prediction, target)

    def _loss(self, prediction, target):
        """Pérdida anterior sin objetivo por sesión, con su interfaz de dos tensores."""
        functional = torch.nn.functional
        if self.recipe.loss == PINBALL:
            return pinball_loss(prediction, target)
        if self.recipe.loss in POINT_OBJECTIVES:
            return point_objective(prediction, target, self.recipe.loss)
        if self.recipe.loss == "mae":
            return functional.l1_loss(prediction, target)
        if self.recipe.loss == "mse":
            return functional.mse_loss(prediction, target)
        return functional.huber_loss(prediction, target, delta=self.recipe.huber_delta)

    def _distilled(self, loss, prediction, keys):
        """Objetivo del tramo: la tarea o, con profesor, su mezcla con la distancia a él."""
        if self.teacher is None:
            return loss
        teacher = self.teacher.values(keys, device=prediction.device, dtype=prediction.dtype)
        return distillation.objective(loss, prediction, teacher, self.recipe.distillation_weight)

    def _backward(self, run):
        """Gradiente de la pérdida media del tramo con los grafos conservados.

        Con profesor, la tarea se mezcla con la distancia a sus salidas. Con la penalización
        C, el objetivo suma la media de los términos de los grupos medidos en el tramo. La
        pérdida devuelta sigue siendo la de la tarea.
        """
        prediction = torch.stack(run.predictions)
        target = torch.tensor(run.targets, dtype=prediction.dtype, device=prediction.device)
        keys = [(flow, at) for flow, at, _ in run.used]
        loss = self._task_loss(prediction, target, keys)
        objective = self._distilled(loss, prediction, keys)
        if run.penalties:
            objective = objective + torch.stack(run.penalties).mean()
        if not torch.isfinite(objective).item():
            raise ValueError("La pérdida del tramo no es finita")
        objective.backward()
        return loss

    def _replay(self, run):
        """Recalcular el tramo por bloques de flujos y acumular el gradiente del mismo objetivo.

        Los flujos son independientes dados los parámetros, que no cambian dentro del tramo.
        Cada bloque parte del estado de sus flujos al empezar el tramo, recorre sus lotes en
        el orden original y pondera su pérdida media por su fracción de etiquetas. La suma
        de los bloques es la pérdida media del tramo.

        Con la penalización C, cada lote se repite con el plan de su evento y el bloque suma
        los términos de sus flujos medidos divididos por el número de grupos del tramo. El
        término de un flujo solo depende de los parámetros, de su token y del valor
        desacoplado de su estado rápido, y cada evento ya divide por todos sus flujos
        medidos, así que la suma de los bloques es la media de los grupos. Los flujos medidos
        sin etiquetas en el tramo también se repiten.
        """
        predictor, size = self.predictor, self.recipe.accumulation_rows
        targets = {
            (flow, at): value for (flow, at, _), value in zip(run.used, run.targets, strict=True)
        }
        controlled = [flow for flows in run.penalties for flow in flows]
        order = list(dict.fromkeys([*(flow for flow, _, _ in run.used), *controlled]))
        sessions = predictor.session_level or (
            self.objective is not None and not self.objective.row_decomposable
        )
        if sessions:
            # Las operaciones por sesión repiten cada instante entero, también las filas sin
            # etiqueta en el tramo, y su pérdida no es una suma de términos por bloque.
            order = list(
                dict.fromkeys([*order, *(f for batch, _, _ in run.segment for f in batch.flow_ids)])
            )
            if len(order) > size:
                raise ValueError("Las operaciones por sesión necesitan repetir todos los flujos")
        total, replayed = 0.0, 0
        for start in range(0, len(order), size):
            group = order[start : start + size]
            states = FlowStates(
                (flow, run.starts[flow]) for flow in group if run.starts[flow] is not None
            )
            # Sin grafo: la repetición no reutiliza nada del cálculo emitido.
            states.detach()
            if predictor.config.variant == "mac_frozen":
                self._anchor_frozen(states)
            # Las comprobaciones de todos los lotes del grupo y la de su pérdida se resuelven
            # juntas antes de su backward, en lugar de esperar a la GPU tras cada lote.
            with deferred_checks():
                members, graphs, penalties = set(group), {}, []
                for batch, plan, whole in run.segment:
                    rows = [i for i, flow in enumerate(batch.flow_ids) if flow in members]
                    if not rows:
                        continue
                    # Solo la selección de todas las filas de un instante completo lo sigue siendo.
                    complete = whole and len(rows) == len(batch.flow_ids)
                    batch = batch.select(rows)
                    new = tuple(flow for flow in batch.flow_ids if flow not in states)
                    if new:
                        states.put(predictor.initial_state(new, differentiable=True))
                    state = states.gather(batch.flow_ids)
                    with torch.enable_grad():
                        prepared = predictor.prepare(
                            batch, state, differentiable=True, session_complete=complete, **plan
                        )
                    states.put(prepared.next_state)
                    if prepared.local_control is not None:
                        penalties.append(prepared.local_control.penalty)
                        replayed += len(prepared.local_control.flow_ids)
                    outputs = self._graphs(prepared).unbind()
                    for row, key in enumerate(
                        zip(batch.flow_ids, batch.prediction_at, strict=True)
                    ):
                        if key in targets:
                            graphs[key] = outputs[row]
                keys = [(flow, at) for flow, at, _ in run.used if flow in members]
                if any(key not in graphs for key in keys):
                    raise ValueError("La repetición del tramo no reproduce sus predicciones")
                terms = []
                if keys:
                    prediction = torch.stack([graphs[key] for key in keys])
                    target = torch.tensor(
                        [targets[key] for key in keys],
                        dtype=prediction.dtype,
                        device=prediction.device,
                    )
                    task = self._task_loss(prediction, target, keys)
                    share = len(keys) / len(run.used)
                    loss = task * share
                    total += float(loss.detach())
                    if self.teacher is not None:
                        loss = self._distilled(task, prediction, keys) * share
                    terms.append(loss)
                if penalties:
                    terms.append(sum(penalties) / len(run.penalties))
                objective = sum(terms[1:], terms[0])
                require_true(torch.isfinite(objective), "La pérdida del tramo no es finita")
            objective.backward()
        if replayed != len(controlled):
            raise ValueError("La repetición del tramo no reproduce los flujos medidos de C")
        return torch.tensor(total)

    def _auxiliary(self, run):
        """Sumar al gradiente el término de F5 con las etiquetas auxiliares entregadas.

        El token de cada decisión se calcula una vez, con los parámetros vigentes y las
        entradas guardadas al emitirla, para todos sus horizontes maduros. Se procesa por
        bloques de hasta `accumulation_rows` decisiones o del lote del predictor. Cada bloque
        divide su suma por todas las etiquetas auxiliares del tramo, así que la suma de los
        bloques es su media. Con peso cero no se calcula nada y el gradiente es el de la
        referencia.
        """
        ready, config = run.horizons.take(), self.horizons
        labels = sum(len(heads) for _, heads, _ in ready)
        run.counters["auxiliary_labels_in_loss"] += labels
        if not ready or config.weight == 0:
            return
        predictor = self.predictor
        dtype, limit = predictor.head.weight.dtype, predictor.config.max_batch
        size = min(self.recipe.accumulation_rows or limit, limit)
        total = 0.0
        for start in range(0, len(ready), size):
            values, heads, targets = zip(*ready[start : start + size], strict=True)
            inputs = {
                name: device_tensor(
                    np.stack([row[name] for row in values]),
                    self.device,
                    torch.bool if name == "presence" else dtype,
                )
                for name in (*MODALITIES, "presence")
            }
            presence = inputs.pop("presence")
            # Una fila del token por decisión y una entrada de la pérdida por horizonte.
            rows = torch.tensor(
                [index for index, names in enumerate(heads) for _ in names], device=self.device
            )
            heads = torch.tensor([head for names in heads for head in names], device=self.device)
            targets = torch.tensor(
                [value for row in targets for value in row], dtype=dtype, device=self.device
            )
            with deferred_checks():
                outputs = predictor.auxiliary_heads(predictor.fused_tokens(inputs, presence))
                errors = auxiliary_errors(
                    outputs.index_select(0, rows), targets, heads, config.horizons
                )
                term = config.weight * errors.sum() / labels
                require_true(torch.isfinite(term), "El término auxiliar no es finito")
            term.backward()
            total += float(term.detach())
        run.counters["auxiliary_loss_sum"] += total

    def _update(self, run, at):
        """Un paso con las etiquetas maduras del tramo y truncamiento de todos los flujos."""
        if run.predictions:
            with nvtx_ranges.phase("titans.backward"):
                loss = self._replay(run) if self.recipe.accumulation_rows else self._backward(run)
                if run.horizons is not None:
                    self._auxiliary(run)
            if run.penalties:
                run.counters["control_groups_in_objective"] = run.counters.get(
                    "control_groups_in_objective", 0
                ) + len(run.penalties)
            with traced_update(self.traces, self.global_step + 1, self.optimizer, loss) as probe:
                with nvtx_ranges.phase("titans.clip"):
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.predictor.parameters(),
                        self.recipe.max_grad_norm or math.inf,
                        error_if_nonfinite=True,
                    )
                    probe.clip(norm, self.recipe.max_grad_norm)
                with nvtx_ranges.phase("titans.optimizer"):
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.shadow is not None:
                        self.shadow.update(self.trainable)
                    # El paso es la única modificación admitida de los parámetros. Se sella
                    # de nuevo.
                    self.predictor._seal_parameters()
            self.global_step += 1
            run.counters["updates"] += 1
            run.counters["labels_in_loss"] += len(run.predictions)
            run.counters["loss_sum"] += float(loss.detach()) * len(run.predictions)
            if self.audit is not None:
                self.audit.append(("update", at, self.global_step, tuple(run.used)))
        elif run.penalties:
            # Sin etiquetas maduras no hay paso: C no añade actualizaciones al presupuesto.
            run.counters["control_groups_discarded"] = run.counters.get(
                "control_groups_discarded", 0
            ) + len(run.penalties)
        if run.horizons is not None and run.horizons.ready:
            # Igual que C, F5 no añade pasos: sin etiquetas principales se descartan.
            discarded = run.horizons.take()
            run.counters["auxiliary_labels_discarded"] += sum(len(row[1]) for row in discarded)
        parameter_id = self.predictor._parameter_id
        with nvtx_ranges.phase("titans.truncate"):
            run.flows.detach(parameter_id)
            if run.context is not None:
                run.context = self.predictor.detach_context(run.context, parameter_id)
        run.penalties.clear()
        run.graphs.clear()
        run.predictions.clear()
        run.targets.clear()
        run.used.clear()
        run.segment.clear()
        run.starts.clear()
        run.instants = 0
        run.counters["segments"] += 1

    def predict_partition(self, source, rows, *, stop=None):
        """Predecir un tramo posterior al ajuste del mismo corpus con `predict`."""
        if (
            type(source) is not FinancialObservationSource
            or source.dataset.identity != self.train.dataset.identity
            or source.phase.decision_start < self.train.phase.decision_end
        ):
            raise ValueError("El tramo no pertenece al corpus y la entrada del ajuste")
        return self.predict(source, rows, stop=stop)

    def _train_pass(self, run, cursor, stop, save):
        predictor, source = self.predictor, self.train
        predictor.train()
        specification = predictor.config.inputs
        # Con prefetch, el hilo productor lee el evento siguiente y valida sus bloques mientras
        # este hilo calcula el actual. La validación no depende del estado del ajuste.
        events = nvtx_ranges.iterate(
            "titans.read",
            prepared(
                source.batched_events(
                    start_cursor=cursor["event"], block_rows=self.recipe.block_rows
                ),
                lambda event: validated_cpu_batches(event.inputs, specification),
                source.dataset.pipeline.prefetch_batches,
            ),
        )
        with closing(events):
            return self._train_events(run, cursor, stop, save, events)

    def _train_events(self, run, cursor, stop, save, events):
        source, start, stage = self.train, cursor["event"], cursor["stage"]
        since, last = 0, time.perf_counter()
        for index, (event, blocks) in enumerate(events, start):
            if not (index == start and stage == "inputs"):
                with nvtx_ranges.phase("titans.labels"):
                    self._labels(run, event, train=True)
                decision = bool(event.inputs) and event.at >= source.phase.decision_start
                if (decision and run.instants == self.recipe.truncation) or event.close_phase:
                    if event.close_phase and run.horizons is not None:
                        run.horizons.flush()
                    before = self.global_step
                    with nvtx_ranges.phase("titans.update"):
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
                        with nvtx_ranges.phase("titans.checkpoint"):
                            save(dict(cursor, event=index, stage="inputs"), run)
                        since, last = 0, time.perf_counter()
                        if stop.requested:
                            raise Paused
            if event.inputs:
                with (
                    nvtx_ranges.phase("titans.observe"),
                    observing(self.traces, self.global_step + 1),
                ):
                    self._observe(run, source, event, differentiable=True, blocks=blocks)
        raise ValueError("El recorrido de ajuste terminó sin su cierre declarado")

    def _export(self, run):
        if run.graphs or run.segment or run.starts or run.penalties:
            raise ValueError("Solo se confirma un recorrido en la barrera posterior a un paso")
        predictor, size = self.predictor, self.predictor.config.max_batch
        flows = sorted(run.flows)
        fast = [
            predictor.export_state_cpu(run.flows.gather(flows[i : i + size]))
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
            **(
                dict(context=None if run.context is None else predictor.export_context(run.context))
                if self.uses_context
                else {}
            ),
            **(
                dict(horizons=None if run.horizons is None else run.horizons.export())
                if self.horizons is not None
                else {}
            ),
        )

    def _restore(self, payload):
        if self.uses_context != ("context" in payload):
            raise ValueError("El recorrido guardado no corresponde al contexto del predictor")
        if (self.horizons is not None) != ("horizons" in payload):
            raise ValueError("El recorrido guardado no corresponde a las cabezas auxiliares")
        run = _Pass()
        for item in payload["fast"]:
            run.flows.put(self.predictor.restore_state(item, device=self.device))
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
        if self.uses_context and payload["context"] is not None:
            run.context = self.predictor.restore_context(payload["context"])
        if self.horizons is not None and payload["horizons"] is not None:
            run.horizons = HorizonLabels.restore(self.horizons, payload["horizons"])
        return run

    def _emission(self):
        """Origen de los pesos emitidos con F2 o F3, o None si se emiten los del ajuste."""
        return self.shadow if self.shadow is not None else self.averaged_point

    @contextmanager
    def _emitted_weights(self):
        """Validar con los pesos que se eligen y se emiten, y volver después a los del ajuste."""
        source = self._emission()
        if source is None:
            yield
            return
        try:
            with applied(dict(self.predictor.named_parameters()), source.emitted()):
                self.predictor._seal_parameters()
                yield
        finally:
            self.predictor._seal_parameters()

    def _model_states(self):
        """Pesos del estado guardado, con los emitidos en `model`.

        Con F2 o F3, `trained_model` guarda además los del ajuste, que son los que se
        recuperan al reanudar. Los consumidores del estado elegido leen `model`.
        """
        model = self.predictor.state_dict()
        source = self._emission()
        if source is None:
            return dict(model=model)
        states = dict(model=model, trained_model=self.predictor.state_dict())
        for name, value in source.emitted().items():
            model[name] = value.detach().clone()
        if self.shadow is not None:
            states["weight_average"] = self.shadow.state_dict()
        return states

    def _start_weights(self, report, state=None):
        """Aplicar F3 a una ejecución nueva y crear o recuperar la sombra de F2."""
        if state is None and self.shrink is not None:
            before = self.predictor._parameter_id
            fresh = type(self.predictor)(
                replace(self.predictor.config, seed=self.shrink.seed),
                device=self.device,
                dtype=self.predictor.head.weight.dtype,
            )
            shrink_perturb(self.trainable, dict(fresh.named_parameters()), self.shrink)
            self.predictor._seal_parameters()
            report["shrink_perturb"] = dict(
                configuration=self.shrink.identity(),
                parameters_before_sha256=before,
                fresh_parameters_sha256=fresh._parameter_id,
                parameters_after_sha256=self.predictor._parameter_id,
            )
        if self.weight_average is not None:
            self.shadow = ShadowWeights(self.weight_average, self.trainable)
            if state is not None:
                self.shadow.load_state_dict(state["weight_average"])

    def _load_best(self, report, directory=None):
        """Dejar el predictor con el estado seleccionado que acredita el informe.

        `directory` lee los estados de otra carpeta con la misma identidad, como la del
        ajuste confirmado que se regenera desde una salida nueva.
        """
        state = load_training_state(
            Path(directory or self.output / "checkpoints"),
            expected_identity=self.identity,
            selection="best",
            expected_sha256=report["best_checkpoint"]["sha256"],
        )
        self.predictor.load_state_dict(state["model"])
        return state

    def load_epoch(self, report, epoch):
        """Dejar cargado el estado fijado de una época evaluada con la época común."""
        state = load_training_state(
            self.output / "checkpoints",
            expected_identity=self.identity,
            selection="pinned",
            expected_sha256=report["epoch_states"][str(epoch)]["sha256"],
        )
        self.predictor.load_state_dict(state["model"])
        return state

    def run(self, *, resume=False, stop=None, joint_epoch=None):
        """Recorrer épocas hasta la paciencia declarada o el presupuesto fijo.

        Con la meseta conjunta, el ajuste espera en su primera meseta (`AWAIT`) hasta que se
        reanuda con la época común del grupo (`joint_epoch`). Con la época común, espera tras
        la última época y `joint_epoch` elige el estado fijado de una de ellas.
        """
        output, checkpoints = self.output, self.output / "checkpoints"
        if isinstance(self.optimizer, torch.optim.Optimizer):
            require_learning_allowed("ChronologicalTrainer.run de Titans-MAC")
        if type(resume) is not bool or output.is_symlink() or output.exists() != resume:
            raise ValueError("Usa una ejecución nueva o solicita continuar una existente")
        common = self.recipe.selection.get("stopping") == COMMON_EPOCH
        if common and self.epoch_scorer is None:
            raise ValueError("La época común necesita puntuar cada validación con su IC")
        stop = stop or StopRequest()
        report_path = output / "run.json"
        cursor, run = dict(epoch=0, phase="validation"), None
        if resume:
            report = _read_report(report_path)
            if report["identity"] != self.identity:
                raise ValueError("La identidad o configuración de la ejecución ha cambiado")
            state = load_training_state(checkpoints, expected_identity=self.identity)
            # Con F2 o F3 se reanuda con los pesos del ajuste, no con los emitidos.
            trained = "trained_model" if self.emits_other_weights else "model"
            self.predictor.load_state_dict(state[trained])
            self.optimizer.load_state_dict(state["optimizer"])
            self._start_weights(report, state)
            restore_rng(state["rng"], str(self.device))
            self.global_step, cursor = state["global_step"], state["cursor"]
            self.selection, self.history = state["selection"], state["history"]
            self.train_metrics = state["train_metrics"]
            if self.traces is not None:
                self.traces.restore(state.get("traces"))
            if state["run"] is not None:
                self.predictor.train()
                run = self._restore(state["run"])
            # Una ejecución ya conjunta solo continúa o se confirma con su misma época común.
            bind_joint_epoch(report, joint_epoch, self.recipe.selection, self.recipe.epochs)
            if report["status"] == "completed":
                self._load_best(report)
                return report
        else:
            output.mkdir(parents=True)
            if self.traces is not None:
                self.traces.restore(None)
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
            bind_joint_epoch(report, joint_epoch, self.recipe.selection, self.recipe.epochs)
            self._start_weights(report)

        def save(position, current=None, *, best=False, pin=None):
            # `pin` es la época evaluada cuyo estado se conserva hasta fijar la época común.
            self._check_runtime()
            state = dict(
                global_step=self.global_step,
                cursor=position,
                **self._model_states(),
                optimizer=self.optimizer.state_dict(),
                rng=capture_rng(str(self.device)),
                selection=self.selection,
                history=self.history,
                train_metrics=self.train_metrics,
                run=None if current is None else self._export(current),
            )
            if self.traces is not None:
                state["traces"] = self.traces.checkpoint()
            path = save_training_state(
                checkpoints, state, identity=self.identity, best=best, pin=pin is not None
            )
            reference = dict(path=str(path.relative_to(output)), sha256=sha256(path))
            report.update(recovery_checkpoint=reference, global_step=self.global_step)
            if best:
                report["best_checkpoint"] = reference
            if pin is not None:
                report.setdefault("epoch_states", {})[str(pin)] = reference
            report.update(selection=self.selection, history=self.history, cursor=position)
            atomic_json(report_path, report)

        if not resume:
            save(cursor)
        started = time.perf_counter()
        try:
            options = self.recipe.selection
            while cursor["phase"] != "done":
                epoch = cursor["epoch"]
                if cursor["phase"] == "train" and cursor["stage"] == "start":
                    # Al empezar una época, el ajuste espera al grupo o
                    # termina si ya está en la época conjunta.
                    decision = epoch_decision(
                        self.selection, options, self.recipe.epochs, joint_epoch
                    )
                    if decision == AWAIT:
                        report.update(awaiting(self.selection, self.recipe.epochs))
                        return report
                    if decision == FINISH:
                        cursor = dict(epoch=epoch, phase="done")
                        save(cursor)
                        continue
                if cursor["phase"] == "validation":
                    # La época común puntúa las filas de esta misma validación desde la época 1.
                    rows = self.epoch_scorer.rows() if common and epoch >= 1 else None
                    with self._emitted_weights():
                        metrics = self.evaluate(self.validation, stop=stop, rows=rows)
                    if rows is not None:
                        metrics.update(self.epoch_scorer.score(epoch, rows, metrics))
                    if self.trace is not None or self.traces is not None:
                        event = dict(
                            kind="validation",
                            epoch=epoch,
                            global_step=self.global_step,
                            score=metrics["session_mae"],
                        )
                        if self.traces is not None:
                            self.traces.validation(event)
                        if self.trace is not None:
                            self.trace(event, self._trace_modules())
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
                    # Con presupuesto fijo o con meseta conjunta, should_stop
                    # nunca detiene el ajuste por sí solo.
                    decision = epoch_decision(
                        self.selection, options, self.recipe.epochs, joint_epoch
                    )
                    finished = decision == FINISH
                    cursor = (
                        dict(epoch=epoch, phase="done")
                        if finished
                        else dict(epoch=epoch, phase="train", event=0, stage="start")
                    )
                    # Con AWAIT, el cursor ya confirmado espera al grupo al empezar la época.
                    save(
                        cursor,
                        best=self.selection["last_improved"],
                        pin=epoch if rows is not None else None,
                    )
                    if stop.requested and not finished:
                        raise Paused
                    continue
                self.train_metrics = self._train_pass(run or _Pass(), cursor, stop, save)
                run = None
                cursor = dict(epoch=epoch + 1, phase="validation")
                save(cursor)
            chosen = (
                chosen_state(
                    checkpoints, self.identity, report, self.history, self.selection, joint_epoch
                )
                if common
                else dict(
                    best_epoch=self.selection["best_epoch"],
                    best_score=self.selection["best_score"],
                )
            )
            report.update(chosen)
            self._load_best(report)
            report.update(
                status="completed",
                stopped_early=cursor["epoch"] < self.recipe.epochs,
                stopping=self.recipe.selection.get("stopping", VALIDATION_PLATEAU),
                plateau_epoch=self.selection.get("plateau_epoch"),
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
