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
from mars_titan.memory import training_event_cache
from mars_titan.memory.mature_reliability import ReliabilitySignal
from mars_titan.models.predictive_adaptation import is_adapter_name
from mars_titan.models.quantile_head import PINBALL, QUANTILE_HEAD, median, pinball_loss
from mars_titan.models.titans.auxiliary_horizons import auxiliary_errors
from mars_titan.models.titans.axial_time_region import join_history_rows, map_history_rows
from mars_titan.models.titans.config import MAX_BLOCK_ROWS, canonical
from mars_titan.models.titans.financial import VARIANTS, FinancialPredictor, FinancialState
from mars_titan.models.titans.financial_inputs import (
    FINAL_TEST_US,
    DecisionBatch,
    device_tensor,
    validated_cpu_batches,
)
from mars_titan.models.titans.frozen_financial import (
    _implementation,
    _numerics,
    declared_sources,
)
from mars_titan.models.titans.session_ic_objective import (
    known_offset,
    parse_session_ic_term,
    session_row_weights,
)
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
from .determinism import (
    declared_determinism,
    valid_determinism,
)
from .horizon_labels import HorizonLabels
from .input_pipeline import prepared
from .kernel_policy import declared_policy, require_policy, valid_precision
from .learning_hold import require_learning_allowed
from .quantile_objectives import (
    POINT_COMPONENT,
    POINT_OBJECTIVES,
    QUANTILE_OBJECTIVES,
    point_objective,
)
from .schedule_free import AveragedPoint, ScheduleFree, is_schedule_free
from .selection import (
    AWAIT,
    CHRONOLOGICAL_MODES,
    COMMON_EPOCH,
    DECAY_LR,
    FINISH,
    SESSION_RANK_IC,
    VALIDATION_PLATEAU,
    advance_selection,
    apply_lr_decay,
    awaiting,
    bind_joint_epoch,
    build_stopping_report,
    canonical_selection,
    emit_average_count,
    epoch_decision,
    initial_selection,
    lr_decay_config,
    metric_definition,
    retained_epochs,
    selected_emit_epochs,
    selection_score,
    smoothing_window,
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
# Variantes de #751 cuyo código solo entra en la identidad cuando el predictor las declara.
_PREDICTOR_MODULES = dict(
    regional_variant="mars_titan.models.titans.regional_variants",
    cross_asset_attention="mars_titan.models.titans.cross_asset_attention",
    axial_time_region="mars_titan.models.titans.axial_time_region",
    fractal_shared_operator="mars_titan.models.titans.fractal_shared_operator",
    asset_groups="mars_titan.models.titans.asset_groups",
)


# Marca de una predicción del tramo cuyo grafo se recalcula al actualizar.
_REPLAY = object()
# --- factor_memory ---
# Marca de estado factorial inicial todavía no capturado al principio del tramo.
_UNSET = object()
# --- fin factor_memory ---
# Contadores de F5. Solo aparecen si el predictor declara las cabezas auxiliares.
_AUXILIARY_COUNTERS = (
    "auxiliary_labels",
    "auxiliary_labels_in_loss",
    "auxiliary_labels_discarded",
    "auxiliary_loss_sum",
)
# Año juliano exacto en microsegundos (365,25 días) y cotas de la semivida causal por recencia.
RECENCY_YEAR_US = 31_557_600_000_000
MIN_RECENCY_HALF_LIFE_YEARS = 0.25
MAX_RECENCY_HALF_LIFE_YEARS = 100.0
# Orden canónico de los grupos funcionales para el recorte de gradiente por rol.
CLIP_GROUP_ORDER = ("technical_head", "mac", "backbone_fusion")


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
    # None conserva el determinismo del proceso. `torch_deterministic_algorithms_v1` activa el
    # modo determinista de PyTorch y de cuDNN al construir el recorrido y entra en su
    # identidad (#794).
    determinism: str | None = None
    # None ajusta con la pérdida declarada. Un objeto le suma el IC por sesión del punto
    # emitido, con su peso y signo contrario (`session_ic_term_v1`), sin cambiar la cabeza.
    session_ic_term: dict | None = None
    # False conserva el recorte global único. True recorta por separado cada grupo funcional
    # (`technical_head`, `mac`, `backbone_fusion`) al umbral `max_grad_norm`.
    grad_clip_by_role: bool = False
    # None pondera por igual todos los segmentos. Un positivo H (en años) modula el paso efectivo
    # del optimizador con semivida causal H anclada en `train.phase.decision_end`.
    recency_half_life_years: float | None = None
    # False emite cada tramo con el grafo que la repetición descarta. True lo emite sin grafo
    # cuando el tramo se repite (`accumulation_rows`): mismos valores con menos trabajo.
    observe_without_graph: bool = False

    def __post_init__(self):
        # Los controles puntuales conservan la cabeza y validan los mismos hiperparámetros.
        validate_loss("mae" if self.loss in QUANTILE_OBJECTIVES else self.loss, self.huber_delta)
        validate_selection(
            self.selection,
            epochs=self.epochs,
            modes=CHRONOLOGICAL_MODES,
            initial_reported=True,
            adaptive=True,
        )
        distillation.checked_weight(self.distillation_weight)
        if self.loss in POINT_OBJECTIVES and self.distillation_weight is not None:
            raise ValueError("El control de solo mediana no admite destilación de las colas")
        if type(self.grad_clip_by_role) is not bool:
            raise ValueError("grad_clip_by_role debe ser un booleano")
        if self.grad_clip_by_role and self.max_grad_norm is None:
            raise ValueError("grad_clip_by_role necesita un max_grad_norm positivo")
        if type(self.observe_without_graph) is not bool:
            raise ValueError("observe_without_graph debe ser un booleano")
        if self.observe_without_graph and self.accumulation_rows is None:
            raise ValueError("observe_without_graph solo se aplica con accumulation_rows")
        if self.recency_half_life_years is not None:
            if (
                isinstance(self.recency_half_life_years, bool)
                or not isinstance(self.recency_half_life_years, (int, float))
                or not math.isfinite(self.recency_half_life_years)
                or not MIN_RECENCY_HALF_LIFE_YEARS
                <= self.recency_half_life_years
                <= MAX_RECENCY_HALF_LIFE_YEARS
            ):
                raise ValueError(
                    "recency_half_life_years debe ser None o un número finito en [0.25, 100]"
                )
            object.__setattr__(self, "recency_half_life_years", float(self.recency_half_life_years))
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
        if not valid_determinism(self.determinism):
            raise ValueError("El determinismo de la receta es None o una política admitida")
        if self.session_ic() is not None and self.distillation_weight is not None:
            raise ValueError("El término de IC por sesión no se combina con la destilación")
        options = self.weight_options()
        if self.recency_half_life_years is not None and options["schedule_free"] is not None:
            raise ValueError("La recencia por tasa efectiva no se combina con schedule_free")
        if lr_decay_config(self.selection) is not None and options["schedule_free"] is not None:
            raise ValueError("La reducción de tasa no se combina con schedule_free")

    def session_ic(self):
        """Término de IC por sesión de la receta, o None si no lo declara."""
        if self.session_ic_term is not None and type(self.session_ic_term) is not dict:
            raise ValueError("session_ic_term debe declarar weight y minimum_rows, o ser None")
        return parse_session_ic_term(self.session_ic_term)

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
        fields["selection"] = canonical_selection(self.selection)
        # Las opciones nuevas de parada solo entran en la identidad cuando no valen su defecto.
        if smoothing_window(self.selection) > 1:
            fields["smoothing_rule"] = "centered_moving_average_truncated_at_phase_boundaries_v1"
        if lr_decay_config(self.selection) is not None:
            fields["learning_rate_decay_rule"] = (
                "multiply_lr_restore_best_model_and_optimizer_reset_patience_v1"
            )
        if emit_average_count(self.selection) > 1:
            fields["emit_weight_averaging"] = (
                "shared_and_adapter_weights_float64_mean_memories_and_online_state_from_best_epoch_v1"
            )
        # Sin acumulación se conserva literalmente la identidad anterior y su huella.
        if fields["accumulation_rows"] is None:
            fields.pop("accumulation_rows")
        else:
            fields["gradient_accumulation"] = "flow_blocks_replayed_from_segment_start_v1"
        # Sin la opción no aparece ningún campo nuevo y la identidad anterior se conserva.
        if fields.pop("observe_without_graph"):
            fields["emission"] = "without_graph_when_the_segment_is_replayed_v1"
        if fields["precision"] is None:
            fields.pop("precision")
        if fields["determinism"] is None:
            fields.pop("determinism")
        if fields["weight_decay_anchor"] is None:
            fields.pop("weight_decay_anchor")
        if fields["distillation_weight"] is None:
            fields.pop("distillation_weight")
        else:
            fields["distillation_distance"] = distillation.DISTANCE
        # El término de IC por sesión solo aparece si se declara, con su identidad.
        term = self.session_ic()
        if term is None:
            fields.pop("session_ic_term")
        else:
            fields["session_ic_term"] = term.identity()
        # F2 y F3 solo aparecen si se declaran, con la identidad de su configuración.
        options = self.weight_options()
        for name, option in options.items():
            if option is None:
                fields.pop(name)
            else:
                fields[name] = option.identity()
        # Sin recorte por grupo ni recencia se conserva literalmente la identidad anterior.
        if not fields["grad_clip_by_role"]:
            fields.pop("grad_clip_by_role")
        else:
            fields["grad_clip_rule"] = (
                "independent_l2_per_component_group_technical_head_mac_backbone_fusion_v1"
            )
        if fields["recency_half_life_years"] is None:
            fields.pop("recency_half_life_years")
        else:
            fields["recency_weighting"] = dict(
                schema_version=1,
                half_life_years=float(self.recency_half_life_years),
                year_microseconds=RECENCY_YEAR_US,
                anchor="train_phase_decision_end",
                normalization="closed_form_continuous_mean_over_train_phase_v1",
                segment_timestamp="mean_weight_over_matured_label_decision_at_v1",
                application="effective_learning_rate_after_clip_v1",
            )
        return dict(
            schema_version=1,
            recipe=RECIPE,
            **fields,
            optimizer="AdamW" if options["schedule_free"] is None else "AdamWScheduleFree",
            truncation_unit="decision_instants_per_differentiable_segment",
            label_rule="loss_only_after_maturity_event_then_next_segment_update",
            loss_reduction="mean_over_matured_labels_of_segment",
            fast_state="reset_each_pass_then_warmup",
            selection_metric_definition=metric_definition(self.selection),
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


def clip_parameter_groups(predictor, trainable=None):
    """Particionar los parámetros ajustables en grupos funcionales disjuntos y exhaustivos.

    Orden determinista heredado de `predictor.named_parameters()`:
    - `technical_head`: rama técnica, cabeza y objetivos o cabezas de salida.
    - `mac`: memoria rápida, atención MAC y puertas de fiabilidad.
    - `backbone_fusion`: codificadores multimodales, fusión y operadores de entrada/sesión.
    """
    allowed = None if trainable is None else set(trainable)
    buckets = {group: [] for group in CLIP_GROUP_ORDER}
    for name, _ in predictor.named_parameters():
        if allowed is not None and name not in allowed:
            continue
        if name.startswith(("technical_branch.", "head.", "output_objective.", "auxiliary_heads.")):
            buckets["technical_head"].append(name)
        elif name.startswith(("mac.", "reliability_gates.", "cross_sectional_rank_memory.")):
            buckets["mac"].append(name)
        # --- factor_memory ---
        elif name.startswith("factor_memory."):
            buckets["mac"].append(name)
        # --- fin factor_memory ---
        # --- contrastive_memory ---
        elif name.startswith("contrastive_memory."):
            buckets["mac"].append(name)
        # --- fin contrastive_memory ---
        # --- ic_titans_mlp ---
        elif name.startswith("ic_titans_mlp."):
            buckets["mac"].append(name)
        # --- fin ic_titans_mlp ---
        # --- lead_lag_memory ---
        elif name.startswith("lead_lag_memory."):
            buckets["mac"].append(name)
        # --- fin lead_lag_memory ---
        # --- three_factor_consolidation ---
        elif name.startswith("three_factor_consolidation."):
            buckets["mac"].append(name)
        # --- fin three_factor_consolidation ---
        else:
            buckets["backbone_fusion"].append(name)
    groups = {group: names for group, names in buckets.items() if names}
    expected = {name for name, _ in predictor.named_parameters()} if allowed is None else allowed
    assigned = [name for names in groups.values() for name in names]
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise ValueError("La partición de recorte por grupo debe cubrir cada parámetro una vez")
    return groups


def recency_segment_weight(decision_timestamps, *, decision_start, decision_end, half_life_years):
    """Peso causal por recencia del segmento normalizado a media continua 1 en el tramo de ajuste.

    Ancla estrictamente en `decision_end` del tramo de ajuste (nunca en validación, calibración
    ni evaluación) y rechaza cualquier fecha fuera de `[decision_start, decision_end)` o en 2024+.
    """
    if (
        isinstance(half_life_years, bool)
        or not isinstance(half_life_years, (int, float))
        or not math.isfinite(half_life_years)
        or not MIN_RECENCY_HALF_LIFE_YEARS <= half_life_years <= MAX_RECENCY_HALF_LIFE_YEARS
    ):
        raise ValueError("half_life_years debe ser un número finito en [0.25, 100]")
    if (
        type(decision_start) is not int
        or type(decision_end) is not int
        or not 0 < decision_start < decision_end <= FINAL_TEST_US
    ):
        raise ValueError("Los límites del tramo de ajuste deben ser causales y anteriores a 2024")
    timestamps = list(decision_timestamps)
    if not timestamps:
        raise ValueError("El segmento necesita al menos una etiqueta madura con fecha")
    rate = math.log(2.0) / float(half_life_years)
    span_years = (decision_end - decision_start) / RECENCY_YEAR_US
    decay_span = rate * span_years
    mean_weight = -math.expm1(-decay_span) / decay_span if decay_span > 0.0 else 1.0
    total = 0.0
    for at in timestamps:
        if type(at) is not int or not decision_start <= at < decision_end or at >= FINAL_TEST_US:
            raise ValueError(
                "La fecha de decisión queda fuera del tramo causal de ajuste o abre 2024"
            )
        age_years = (decision_end - at) / RECENCY_YEAR_US
        total += math.exp(-rate * age_years)
    weight = (total / len(timestamps)) / mean_weight
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("El peso por recencia calculado no es finito y positivo")
    return float(weight)


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
    regions = state.regions
    if regions is not None:
        # El historial de C2 corta su grafo en el mismo punto que MAC.
        regions = map_history_rows(regions, lambda value: value.detach())
    return replace(state, parameter_id=parameter_id or state.parameter_id, mac=mac, regions=regions)


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

        state, mac, regions = first.state, None, None
        if state.mac is not None:
            parts = [
                map_mac_rows(block.state.mac, lambda value, a=start, b=stop: value[a:b])
                for block, start, stop in runs
            ]
            mac = parts[0] if len(parts) == 1 else join_mac_rows(parts)
        if state.regions is not None:
            pieces = [
                map_history_rows(block.state.regions, lambda value, a=start, b=stop: value[a:b])
                for block, start, stop in runs
            ]
            regions = pieces[0] if len(pieces) == 1 else join_history_rows(pieces)
        return FinancialState(
            state.config_id,
            state.parameter_id,
            tuple(flows),
            tuple(block.state.last_sample_ids[row] for block, row in entries),
            tuple(block.state.last_prediction_at[row] for block, row in entries),
            torch.tensor([block.steps[row] for block, row in entries], dtype=torch.int64),
            mac,
            regions,
        )


def _compatible(expected, supplied):
    """Las vistas de ajuste y validación comparten fuente y representación, no índice."""
    left, right = expected.identity(), supplied.identity()
    left.pop("view_sha256")
    right.pop("view_sha256")
    return canonical(left) == canonical(right)


class Paused(Exception):
    """Solicitud de parada atendida en una barrera ya confirmada."""


class _SessionRankIC:
    """Acumular el IC de rangos de Spearman por sesión durante el ajuste en O(sesiones activas).

    Cada sesión se cierra en cuanto maduran todas sus predicciones pendientes del recorrido
    (o al cierre del tramo) con exactamente la misma definición que
    `scale_diagnostics.session_correlations(..., rank=True)`.
    """

    def __init__(self, min_assets):
        self.min_assets = int(min_assets)
        self.pending_counts = {}
        self.open_sessions = {}
        self.defined_values = []

    def observe(self, market, decision_at):
        key = (market, int(decision_at))
        self.pending_counts[key] = self.pending_counts.get(key, 0) + 1

    def _finish_session(self, key):
        pair = self.open_sessions.pop(key, None)
        if pair is None:
            return
        preds, targets = pair
        if len(preds) < self.min_assets:
            return
        observed = np.asarray(targets, dtype=np.float64)
        predicted = np.asarray(preds, dtype=np.float64)
        if np.ptp(observed) == 0 or np.ptp(predicted) == 0:
            return
        from scipy.stats import rankdata

        value = float(np.corrcoef(rankdata(observed), rankdata(predicted))[0, 1])
        if math.isfinite(value):
            self.defined_values.append(value)

    def resolve(self, market, decision_at, prediction, target):
        key = (market, int(decision_at))
        entry = self.open_sessions.get(key)
        if entry is None:
            entry = ([], [])
            self.open_sessions[key] = entry
        entry[0].append(float(prediction))
        entry[1].append(float(target))
        remaining = self.pending_counts.get(key, 0) - 1
        if remaining <= 0:
            self.pending_counts.pop(key, None)
            self._finish_session(key)
        else:
            self.pending_counts[key] = remaining

    def close(self):
        for key in sorted(self.open_sessions):
            self._finish_session(key)
        self.open_sessions.clear()
        self.pending_counts.clear()

    def summary(self):
        return dict(
            session_rank_ic=(
                float(np.asarray(self.defined_values, dtype=np.float64).mean())
                if self.defined_values
                else None
            ),
            session_rank_ic_sessions=len(self.defined_values),
        )

    def export(self):
        return dict(
            defined=torch.tensor(self.defined_values, dtype=torch.float64),
            pending=[[m, t, c] for (m, t), c in sorted(self.pending_counts.items())],
            open=[
                [
                    m,
                    t,
                    torch.tensor(preds, dtype=torch.float64),
                    torch.tensor(targets, dtype=torch.float64),
                ]
                for (m, t), (preds, targets) in sorted(self.open_sessions.items())
            ],
        )

    @classmethod
    def restore(cls, min_assets, payload):
        acc = cls(min_assets)
        acc.defined_values = [float(v) for v in payload["defined"].tolist()]
        acc.pending_counts = {(m, int(t)): int(c) for m, t, c in payload["pending"]}
        acc.open_sessions = {
            (m, int(t)): (preds.tolist(), targets.tolist())
            for m, t, preds, targets in payload["open"]
        }
        return acc


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
    # Con acumulación: lotes del tramo en orden, con el plan de C de su evento y la señal
    # de #791 que leyeron, y estado de cada flujo al empezarlo. None indica un flujo nuevo
    # dentro del tramo. Se vacían en cada actualización.
    segment: list = field(default_factory=list)
    starts: dict = field(default_factory=dict)
    # Con la penalización C, un elemento por grupo lógico medido del tramo: su término con
    # grafo o, con acumulación, la tupla de sus flujos medidos, que la repetición recalcula.
    penalties: list = field(default_factory=list)
    # Con PT4, el contexto de mercado del recorrido. Entre eventos no guarda nada pendiente.
    context: object = None
    # Con F5, el registro de etiquetas auxiliares del recorrido de ajuste.
    horizons: object = None
    # Con #791, la señal de fiabilidad madura del recorrido.
    reliability: object = None
    # Con CRRM, el estado de la memoria transversal de rango y su instantánea para replay.
    rank_memory: object = None
    rank_memory_start: object = None
    rank_memory_labels: list = field(default_factory=list)
    # --- factor_memory ---
    # Con LFM-SW, la base factorial por mercado y su fotografía al inicio del tramo.
    factor_memory: object = None
    factor_start: object = _UNSET
    # --- fin factor_memory ---
    # --- contrastive_memory ---
    # Con Idea 5 (InfoNCE-Titans), el estado por mercado y su copia al inicio del tramo.
    contrastive: object = None
    contrastive_start: object = None
    # --- fin contrastive_memory ---
    # --- ic_titans_mlp ---
    ic_mlp_memory: object = None
    ic_mlp_memory_start: object = None
    ic_mlp_memory_labels: list = field(default_factory=list)
    # --- fin ic_titans_mlp ---
    # --- lead_lag_memory ---
    lead_lag_memory: object = None
    lead_lag_memory_start: object = None
    lead_lag_memory_labels: list = field(default_factory=list)
    # --- fin lead_lag_memory ---
    # --- three_factor_consolidation ---
    three_factor_state: object = None
    three_factor_start: object = None
    three_factor_labels: list = field(default_factory=list)
    # --- fin three_factor_consolidation ---
    # Con métrica `session_rank_ic`, acumulador en flujo del IC de rangos de entrenamiento.
    rank_ic: object = None


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
        # El término de IC por sesión se suma a la pérdida declarada de la cabeza, que un
        # objetivo de salida ya sustituye por la suya.
        self.session_ic = recipe.session_ic()
        if self.session_ic is not None and self.objective is not None:
            raise ValueError("El término de IC por sesión no se combina con un objetivo de salida")
        # Con liquidez, cada fila lleva a la pérdida su columna del bloque del gráfico.
        self.liquidity = self.session_ic is not None and self.session_ic.weighted
        declares_liquidity = (
            self.session_ic is not None and self.session_ic.liquidity_column is not None
        )
        removed = predictor.config.removed_modalities or ()
        if declares_liquidity and any(name in removed for name in ("charts", "charts_technical")):
            raise ValueError("La ponderación por liquidez requiere la modalidad charts sin retirar")
        if declares_liquidity and (
            -self.session_ic.liquidity_column > predictor.config.inputs.dimensions["charts"]
        ):
            raise ValueError("La columna de liquidez no pertenece al bloque del gráfico")
        # Con el desplazamiento conocido, cada fila lleva además el suyo a la pérdida.
        self.offset = self.session_ic is not None and self.session_ic.shifted
        if self.offset and any(name in removed for name in ("charts", "charts_technical")):
            raise ValueError("El desplazamiento conocido requiere la modalidad charts sin retirar")
        if self.offset and (
            -self.session_ic.offset_column > predictor.config.inputs.dimensions["charts"]
        ):
            raise ValueError("La columna del desplazamiento no pertenece al bloque del gráfico")
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
        # #791 cambia el ajuste y la inferencia, así que los dos recorridos llevan su señal.
        self.reliability = predictor.config.mature_reliability
        # CRRM cambia ajuste e inferencia y consume etiquetas maduras en ambos recorridos.
        self.uses_rank_memory = predictor.cross_sectional_rank_memory is not None
        # --- factor_memory ---
        self.uses_factor_memory = predictor.factor_memory is not None
        # --- fin factor_memory ---
        # --- contrastive_memory ---
        self.uses_contrastive = predictor.contrastive_memory is not None
        # --- fin contrastive_memory ---
        # --- ic_titans_mlp ---
        self.uses_ic_mlp_memory = predictor.ic_titans_mlp is not None
        # --- fin ic_titans_mlp ---
        # --- lead_lag_memory ---
        self.uses_lead_lag_memory = predictor.lead_lag_memory is not None
        # --- fin lead_lag_memory ---
        # --- three_factor_consolidation ---
        self.uses_three_factor = predictor.three_factor_consolidation is not None
        # --- fin three_factor_consolidation ---
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
        # La política se aplica antes de calcular nada y antes de registrar `_numerics`, que
        # guarda también las banderas del determinismo.
        self.kernel_policy = declared_policy(recipe.precision)
        self.determinism = declared_determinism(recipe.determinism)

    def _signal(self, run):
        """Señal de #791 del recorrido, creada al usarla por primera vez, o None."""
        if self.reliability is not None and run.reliability is None:
            run.reliability = ReliabilitySignal(self.reliability)
        return run.reliability

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
        # Con `observe_without_graph`, la emisión de un tramo que se repite no construye el
        # grafo que la repetición descartaría. Los valores emitidos y los estados no cambian.
        graph = grad and not (accumulate and self.recipe.observe_without_graph)
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
            run.context = predictor.initial_context(differentiable=graph)
        if self.uses_rank_memory and run.rank_memory is None:
            run.rank_memory = predictor.initial_rank_memory()
        if self.uses_rank_memory and accumulate and run.rank_memory_start is None:
            run.rank_memory_start = predictor.detach_rank_memory(run.rank_memory)
        # --- ic_titans_mlp ---
        if self.uses_ic_mlp_memory and run.ic_mlp_memory is None:
            run.ic_mlp_memory = predictor.initial_ic_mlp_memory()
        if self.uses_ic_mlp_memory and accumulate and run.ic_mlp_memory_start is None:
            run.ic_mlp_memory_start = predictor.detach_ic_mlp_memory(run.ic_mlp_memory)
        # --- fin ic_titans_mlp ---
        # --- lead_lag_memory ---
        if self.uses_lead_lag_memory and run.lead_lag_memory is None:
            run.lead_lag_memory = predictor.initial_lead_lag_memory()
        if self.uses_lead_lag_memory and accumulate and run.lead_lag_memory_start is None:
            run.lead_lag_memory_start = predictor.detach_lead_lag_memory(run.lead_lag_memory)
        # --- fin lead_lag_memory ---
        # --- three_factor_consolidation ---
        if self.uses_three_factor and run.three_factor_state is None:
            run.three_factor_state = predictor.initial_three_factor_state()
        if self.uses_three_factor and accumulate and run.three_factor_start is None:
            run.three_factor_start = predictor.detach_three_factor_state(run.three_factor_state)
        # --- fin three_factor_consolidation ---
        context = dict(context=run.context) if self.uses_context else {}
        # --- factor_memory ---
        if self.uses_factor_memory:
            if accumulate and run.factor_start is _UNSET:
                run.factor_start = run.factor_memory
            if run.factor_memory is None:
                run.factor_memory = predictor.initial_factor_memory(differentiable=graph)
        factor = dict(factor_memory=run.factor_memory) if self.uses_factor_memory else {}
        # --- fin factor_memory ---
        # --- contrastive_memory ---
        contrastive = None
        if self.uses_contrastive:
            if run.contrastive is None:
                run.contrastive = predictor.initial_contrastive_memory(differentiable=graph)
            if accumulate and run.contrastive_start is None:
                run.contrastive_start = predictor.detach_contrastive_memory(run.contrastive)
            contrastive = run.contrastive
        # --- fin contrastive_memory ---
        signal, surprises = self._signal(run), []
        # Las comprobaciones de finitud de todos los bloques del evento se resuelven juntas
        # al final, antes de volver al recorrido, que es el único que aplica pasos, guarda
        # checkpoints o cierra recibos. Así no se espera a la GPU tras cada bloque.
        with deferred_checks():
            for cpu in validated:
                batch = DecisionBatch.from_validated(
                    cpu, device=self.device, dtype=predictor.head.weight.dtype
                )
                options = dict(plan)
                if signal is not None:
                    # Todos los bloques del evento leen la misma señal, la de sus etiquetas.
                    options["reliability"] = torch.tensor(
                        signal.values(batch.flow_ids),
                        dtype=predictor.head.weight.dtype,
                        device=self.device,
                    )
                new = tuple(flow for flow in batch.flow_ids if flow not in run.flows)
                if new:
                    run.flows.put(predictor.initial_state(new, differentiable=graph))
                if accumulate:
                    for flow in batch.flow_ids:
                        if flow not in run.starts:
                            # La repetición parte del estado sin grafo de cada flujo al empezar.
                            run.starts[flow] = None if flow in new else run.flows.handle(flow)
                    run.segment.append((batch, options, complete))
                state = run.flows.gather(batch.flow_ids)
                if grad and predictor.config.variant == "mac_frozen":
                    # Sin escrituras, cada bloque lee directamente el M0 vigente sin fragmentar
                    # `run.flows` en cortes `SliceBackward0` de complejidad O(R·F·D²).
                    memory = predictor.mac.initial_state(
                        len(batch.flow_ids), differentiable=graph
                    ).memory
                    state = replace(state, mac=replace(state.mac, memory=memory))
                prepare_options = (
                    dict(options, rank_memory=run.rank_memory) if self.uses_rank_memory else options
                )
                # --- factor_memory ---
                if self.uses_factor_memory:
                    prepare_options = dict(prepare_options, **factor)
                # --- fin factor_memory ---
                # --- contrastive_memory ---
                if self.uses_contrastive:
                    prepare_options = dict(prepare_options, contrastive=contrastive)
                # --- fin contrastive_memory ---
                # --- ic_titans_mlp ---
                if self.uses_ic_mlp_memory:
                    prepare_options = dict(prepare_options, ic_mlp_memory=run.ic_mlp_memory)
                # --- fin ic_titans_mlp ---
                # --- lead_lag_memory ---
                if self.uses_lead_lag_memory:
                    prepare_options = dict(prepare_options, lead_lag_memory=run.lead_lag_memory)
                # --- fin lead_lag_memory ---
                # --- three_factor_consolidation ---
                if self.uses_three_factor:
                    prepare_options = dict(
                        prepare_options, three_factor_state=run.three_factor_state
                    )
                # --- fin three_factor_consolidation ---
                with nvtx_ranges.phase("titans.prepare"), torch.set_grad_enabled(graph):
                    prepared = predictor.prepare(
                        batch,
                        state,
                        differentiable=graph,
                        session_complete=complete,
                        **prepare_options,
                        **context,
                    )
                if self.uses_rank_memory and not warmup:
                    run.rank_memory = predictor.stage_rank_memory(
                        run.rank_memory,
                        batch.flow_ids,
                        batch.prediction_at,
                        prepared.rank_memory_keys,
                        prepared.point_predictions,
                        differentiable=graph and not accumulate,
                    )
                # --- ic_titans_mlp ---
                if self.uses_ic_mlp_memory and not warmup:
                    run.ic_mlp_memory = predictor.stage_ic_mlp_memory(
                        run.ic_mlp_memory,
                        batch.flow_ids,
                        batch.prediction_at,
                        prepared.ic_mlp_keys,
                        differentiable=graph and not accumulate,
                    )
                # --- fin ic_titans_mlp ---
                # --- lead_lag_memory ---
                if self.uses_lead_lag_memory and not warmup:
                    run.lead_lag_memory = predictor.stage_lead_lag_memory(
                        run.lead_lag_memory,
                        batch.flow_ids,
                        batch.prediction_at,
                        prepared.lead_lag_memberships,
                        prepared.point_predictions,
                        differentiable=graph and not accumulate,
                    )
                # --- fin lead_lag_memory ---
                # --- three_factor_consolidation ---
                if self.uses_three_factor:
                    run.three_factor_state = predictor.stage_three_factor_state(
                        run.three_factor_state,
                        batch.flow_ids,
                        batch.prediction_at,
                        prepared.three_factor_emission,
                        prepared.point_predictions,
                        record_pending=not warmup,
                        differentiable=graph and not accumulate,
                    )
                # --- fin three_factor_consolidation ---
                if self.uses_context:
                    # Los bloques siguientes del evento leen el mismo contexto confirmado.
                    context = dict(context=prepared.next_context)
                # --- factor_memory ---
                if self.uses_factor_memory:
                    next_factor = prepared.next_factor_memory
                    run.factor_memory = (
                        predictor.detach_factor_memory(next_factor) if accumulate else next_factor
                    )
                    factor = dict(factor_memory=run.factor_memory)
                # --- fin factor_memory ---
                # --- contrastive_memory ---
                if self.uses_contrastive:
                    contrastive = prepared.next_contrastive
                # --- fin contrastive_memory ---
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
                graphs = self._graphs(prepared, batch).unbind()
                if grad:
                    for row, key in enumerate(
                        zip(batch.flow_ids, batch.prediction_at, strict=True)
                    ):
                        run.graphs[key] = _REPLAY if accumulate else graphs[row]
                emitted.append((batch, prepared))
                if prepared.surprise is not None:
                    surprises.append((batch.flow_ids, prepared.surprise))
                run.counters["predictions"] += size
                if ledger is not None:
                    ledger.remember(cpu)
            if self.uses_context:
                # La sesión se escribe una vez, después de emitir todas sus predicciones.
                with nvtx_ranges.phase("titans.context"), torch.set_grad_enabled(graph):
                    run.context = predictor.commit_context(context["context"], differentiable=graph)
            # --- contrastive_memory ---
            if self.uses_contrastive:
                # La memoria contrastiva escribe la sesión entera tras emitir sus predicciones.
                with nvtx_ranges.phase("titans.contrastive"), torch.set_grad_enabled(graph):
                    committed = predictor.commit_contrastive_memory(
                        contrastive, differentiable=graph
                    )
                    run.contrastive = (
                        predictor.detach_contrastive_memory(committed) if accumulate else committed
                    )
            # --- fin contrastive_memory ---
            with nvtx_ranges.phase("titans.emit"):
                self._emit(run, source, emitted)
            if surprises:
                # El control de sorpresa la registra al emitir y la leen los eventos siguientes.
                losses = torch.cat([loss for _, loss in surprises]).tolist()
                signal.observe([flow for flows, _ in surprises for flow in flows], losses)
            if measured:
                with nvtx_ranges.phase("titans.penalty"):
                    self._penalty(run, measured, keep=grad, replay=accumulate)
        if event.inputs and not warmup:
            run.instants += 1

    def _graphs(self, prepared, batch):
        """Salidas por fila que entran en la pérdida del tramo.

        Con el término ponderado por liquidez, cada fila añade al final su liquidez recibida,
        sin gradiente: así llega a la pérdida con su predicción, también en la repetición por
        bloques, sin más estado que guardar. Con el desplazamiento conocido de la etiqueta,
        añade después el suyo, de la misma forma.
        """
        if self.quantiles:
            outputs = prepared.quantiles
        elif self.objective is not None:
            return prepared.outputs
        else:
            outputs = prepared.point_predictions
        if not self.liquidity and not self.offset:
            return outputs
        charts = batch.inputs["charts"]
        columns = [outputs if outputs.ndim == 2 else outputs.unsqueeze(1)]
        if self.liquidity:
            column = charts[:, self.session_ic.liquidity_column].detach()
            if self.session_ic.top_third_weighted:
                # Donde falta la modalidad charts, la liquidez no es conocida y pesa 1.
                present = batch.presence[:, MODALITIES.index("charts")]
                column = torch.where(present, column, torch.full_like(column, float("nan")))
            columns.append(column)
        if self.offset:
            term = self.session_ic
            columns.append(known_offset(charts[:, term.offset_column], term.offset_scale))
        columns[1:] = [column.to(outputs.dtype).unsqueeze(1) for column in columns[1:]]
        return torch.cat(columns, dim=1)

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
            if run.rank_ic is not None:
                run.rank_ic.observe(flow.split("/", 1)[0], at)
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
        markets, moments, errors, matured, rank_matured = [], [], [], [], []
        # --- ic_titans_mlp ---
        ic_mlp_matured = []
        # --- fin ic_titans_mlp ---
        # --- lead_lag_memory ---
        lead_lag_matured = []
        # --- fin lead_lag_memory ---
        # --- three_factor_consolidation ---
        three_factor_matured = []
        # --- fin three_factor_consolidation ---
        for flow, decision_at, value in event.labels:
            key = flow, decision_at
            if key not in run.pending:
                raise ValueError("El label no tiene una predicción emitida pendiente")
            issued = run.pending.pop(key)
            if self.reliability is not None:
                matured.append((flow, issued, value))
            if self.uses_rank_memory:
                rank_matured.append((flow, decision_at, value))
            # --- ic_titans_mlp ---
            if self.uses_ic_mlp_memory:
                ic_mlp_matured.append((flow, decision_at, value))
            # --- fin ic_titans_mlp ---
            # --- lead_lag_memory ---
            if self.uses_lead_lag_memory:
                lead_lag_matured.append((flow, decision_at, value))
            # --- fin lead_lag_memory ---
            # --- three_factor_consolidation ---
            if self.uses_three_factor:
                three_factor_matured.append((flow, decision_at, value))
            # --- fin three_factor_consolidation ---
            if train and run.horizons is not None:
                run.counters["auxiliary_labels"] += run.horizons.mature(flow, decision_at, value)
            market = flow.split("/", 1)[0]
            markets.append(market)
            moments.append(decision_at)
            errors.append(issued - value)
            if train and run.rank_ic is not None:
                run.rank_ic.resolve(market, decision_at, issued, value)
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
        signal = self._signal(run)
        if signal is not None:
            signal.settle(matured)
        if self.uses_rank_memory and rank_matured:
            if run.rank_memory is None:
                run.rank_memory = self.predictor.initial_rank_memory()
            accumulate = train and self.recipe.accumulation_rows is not None
            if accumulate and run.segment:
                run.rank_memory_labels.append((len(run.segment), tuple(rank_matured)))
            run.rank_memory = self.predictor.settle_rank_memory(
                run.rank_memory,
                rank_matured,
                differentiable=train and not accumulate,
                train=train,
            )
        # --- ic_titans_mlp ---
        if self.uses_ic_mlp_memory and ic_mlp_matured:
            if run.ic_mlp_memory is None:
                run.ic_mlp_memory = self.predictor.initial_ic_mlp_memory()
            accumulate = train and self.recipe.accumulation_rows is not None
            if accumulate and run.segment:
                run.ic_mlp_memory_labels.append((len(run.segment), tuple(ic_mlp_matured)))
            run.ic_mlp_memory = self.predictor.settle_ic_mlp_memory(
                run.ic_mlp_memory,
                ic_mlp_matured,
                differentiable=train and not accumulate,
                train=train,
            )
        # --- fin ic_titans_mlp ---
        # --- lead_lag_memory ---
        if self.uses_lead_lag_memory and lead_lag_matured:
            if run.lead_lag_memory is None:
                run.lead_lag_memory = self.predictor.initial_lead_lag_memory()
            accumulate = train and self.recipe.accumulation_rows is not None
            if accumulate and run.segment:
                run.lead_lag_memory_labels.append((len(run.segment), tuple(lead_lag_matured)))
            run.lead_lag_memory = self.predictor.settle_lead_lag_memory(
                run.lead_lag_memory,
                lead_lag_matured,
                differentiable=train and not accumulate,
                train=train,
            )
        # --- fin lead_lag_memory ---
        # --- three_factor_consolidation ---
        if self.uses_three_factor and three_factor_matured:
            if run.three_factor_state is None:
                run.three_factor_state = self.predictor.initial_three_factor_state()
            accumulate = train and self.recipe.accumulation_rows is not None
            if accumulate and run.segment:
                run.three_factor_labels.append((len(run.segment), tuple(three_factor_matured)))
            run.three_factor_state = self.predictor.settle_three_factor_state(
                run.three_factor_state,
                three_factor_matured,
                differentiable=train and not accumulate,
                train=train,
            )
        # --- fin three_factor_consolidation ---
        for start in range(0, len(errors), 4096):
            run.errors.update(
                markets[start : start + 4096],
                moments[start : start + 4096],
                errors[start : start + 4096],
            )

    def _close(self, run):
        if run.rank_ic is not None:
            run.rank_ic.close()
        run.counters["unresolved"] = len(run.pending)
        run.flows.clear()
        run.context = None
        run.horizons = None
        run.reliability = None
        run.rank_memory = None
        run.rank_memory_start = None
        run.rank_memory_labels.clear()
        # --- factor_memory ---
        run.factor_memory = None
        run.factor_start = _UNSET
        # --- fin factor_memory ---
        # --- contrastive_memory ---
        run.contrastive = None
        run.contrastive_start = None
        # --- fin contrastive_memory ---
        # --- ic_titans_mlp ---
        run.ic_mlp_memory = None
        run.ic_mlp_memory_start = None
        run.ic_mlp_memory_labels.clear()
        # --- fin ic_titans_mlp ---
        # --- lead_lag_memory ---
        run.lead_lag_memory = None
        run.lead_lag_memory_start = None
        run.lead_lag_memory_labels.clear()
        # --- fin lead_lag_memory ---
        # --- three_factor_consolidation ---
        run.three_factor_state = None
        run.three_factor_start = None
        run.three_factor_labels.clear()
        # --- fin three_factor_consolidation ---
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
        result = dict(
            samples=summary["samples"],
            session_count=summary["session_count"],
            session_mae=summary["session_mae"],
            session_mse=summary["session_mse"],
            **counters,
        )
        if run.rank_ic is not None:
            result.update(run.rank_ic.summary())
        return result

    def evaluate(self, source, *, stop=None, rows=None):
        """Validación temporal con parámetros congelados y memoria rápida reiniciada.

        Con `rows`, cada etiqueta resuelta añade su predicción emitida y sus cuantiles.
        """
        predictor, run = self.predictor, _Pass(rows=rows)
        predictor.eval()
        specification = predictor.config.inputs
        # Como en el ajuste: con prefetch, el hilo productor lee el evento siguiente y sus hilos
        # de preparación validan los bloques mientras este hilo calcula el actual.
        events = nvtx_ranges.iterate(
            "titans.read",
            prepared(
                source.batched_events(block_rows=self.recipe.block_rows),
                lambda event: validated_cpu_batches(event.inputs, specification),
                source.dataset.pipeline.prefetch_batches,
                source.dataset.pipeline.prepare_workers,
            ),
        )
        with torch.no_grad(), closing(events):
            for event, blocks in events:
                if stop is not None and stop.requested:
                    raise Paused
                self._labels(run, event, train=False)
                if event.inputs:
                    self._observe(run, source, event, differentiable=False, blocks=blocks)
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
            not training_event_cache.supported(source)
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
            not training_event_cache.supported(train)
            or not training_event_cache.supported(validation)
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
            predictor.local_control is not None
            or self.shrink.fresh_seed(predictor.config.seed) == predictor.config.seed
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
        self.clip_groups = (
            clip_parameter_groups(predictor, self.trainable) if recipe.grad_clip_by_role else None
        )
        self._clip_param_groups = (
            None
            if self.clip_groups is None
            else [[self.trainable[name] for name in names] for names in self.clip_groups.values()]
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
        if self.determinism is not None:
            self.identity["determinism"] = self.determinism
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
        if self.clip_groups is not None:
            # Sin recorte por grupo, la identidad conserva literalmente su forma anterior.
            self.identity["grad_clip_groups"] = self.clip_groups
        self.run_id = hashlib.sha256(canonical(self.identity).encode()).hexdigest()
        self.global_step, self.selection, self.history, self.train_metrics = 0, None, [], None
        self.track_train_rank_ic = recipe.selection["metric"] == SESSION_RANK_IC
        self.lr_decays = []
        self.epoch_lrs = {}

    def _code(self):
        own = {name: sha256(Path(importlib.import_module(name).__file__)) for name in _OWN_MODULES}
        own.update(training_event_cache.source_implementation(getattr(self, "train", None)))
        from mars_titan.models.titans import host_finite

        if host_finite.enabled():
            own[host_finite.__name__] = sha256(Path(host_finite.__file__))
        for option, name in _PREDICTOR_MODULES.items():
            if getattr(self.predictor.config, option) is not None:
                own[name] = sha256(Path(importlib.import_module(name).__file__))
        # H1 de #783 también, con el catálogo macro del que salen sus familias.
        if self.predictor.config.holographic_network is not None:
            module = importlib.import_module("mars_titan.models.titans.holographic_network")
            own[module.__name__] = sha256(Path(module.__file__))
            own["data/catalogs/macro-indicators.csv"] = sha256(module.CATALOG)
        # C3 agrupa sus sectores con el módulo común de C4, que también entra en la identidad.
        if self.predictor.config.fractal_shared_operator is not None:
            name = "mars_titan.data.correlation_regions"
            own[name] = sha256(Path(importlib.import_module(name).__file__))
        # H4 de #783 añade también el módulo de las medidas de la ventana.
        if self.predictor.config.asset_groups is not None:
            name = "mars_titan.data.asset_groups"
            own[name] = sha256(Path(importlib.import_module(name).__file__))
        # #791 añade el acumulador de su señal. Sus puertas ya están en `_implementation`,
        # porque `financial` importa siempre su configuración.
        if self.predictor.config.mature_reliability is not None:
            name = "mars_titan.memory.mature_reliability"
            own[name] = sha256(Path(importlib.import_module(name).__file__))
        for option, name in _WEIGHT_MODULES.items():
            # F2 y F3 solo añaden su código si la receta los declara.
            if getattr(self.recipe, option) is not None:
                own[name] = sha256(Path(importlib.import_module(name).__file__))
        # El determinismo de #794 también, con el módulo que lo activa y lo comprueba.
        if self.recipe.determinism is not None:
            name = "mars_titan.training.determinism"
            own[name] = sha256(Path(importlib.import_module(name).__file__))
        return {
            **_implementation(
                kronlstm=self.predictor.config.temporal_encoder is not None,
                declared=declared_sources(self.predictor.config),
            ),
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
        liquidity = offset = None
        if self.offset:
            # La última columna es el desplazamiento conocido de la fila.
            offset, prediction = prediction[:, -1], prediction[:, :-1]
        if self.liquidity:
            # La última que queda es la liquidez de la fila; el resto, sus salidas.
            liquidity, prediction = prediction[:, -1], prediction[:, :-1]
        if (self.offset or self.liquidity) and not self.quantiles:
            prediction = prediction.squeeze(1)
        sessions = (
            [(flow.split("/", 1)[0], at) for flow, at in keys]
            if self.session_ic is not None
            else None
        )
        row_weights = (
            session_row_weights(liquidity, sessions, self.session_ic.liquidity_weight)
            if self.session_ic is not None and self.session_ic.top_third_weighted
            else None
        )
        loss = self._loss(prediction, target, weights=row_weights)
        if self.session_ic is not None:
            # El punto es el que se emite: la mediana con cuantiles o la salida escalar.
            point = median(prediction) if self.quantiles else prediction
            loss = loss + self.session_ic.term(point, target, sessions, liquidity, offset)
        return loss

    def _loss(self, prediction, target, *, weights=None):
        """Pérdida anterior sin objetivo por sesión, con su interfaz de dos tensores."""
        functional = torch.nn.functional
        if weights is None:
            if self.recipe.loss == PINBALL:
                return pinball_loss(prediction, target)
            if self.recipe.loss in POINT_OBJECTIVES:
                return point_objective(prediction, target, self.recipe.loss)
            if self.recipe.loss == "mae":
                return functional.l1_loss(prediction, target)
            if self.recipe.loss == "mse":
                return functional.mse_loss(prediction, target)
            return functional.huber_loss(prediction, target, delta=self.recipe.huber_delta)
        if self.recipe.loss == PINBALL:
            per_row = pinball_loss(prediction, target, reduction="none")
        elif self.recipe.loss in POINT_OBJECTIVES:
            per_row = functional.l1_loss(median(prediction), target, reduction="none")
            if self.recipe.loss == POINT_COMPONENT:
                per_row = per_row / 10
        elif self.recipe.loss == "mae":
            per_row = functional.l1_loss(prediction, target, reduction="none")
        elif self.recipe.loss == "mse":
            per_row = functional.mse_loss(prediction, target, reduction="none")
        else:
            per_row = functional.huber_loss(
                prediction, target, reduction="none", delta=self.recipe.huber_delta
            )
        weights = weights.to(per_row.dtype)
        return (weights * per_row).sum() / weights.sum()

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
        with deferred_checks():
            prediction = torch.stack(run.predictions)
            target = torch.tensor(run.targets, dtype=prediction.dtype, device=prediction.device)
            keys = [(flow, at) for flow, at, _ in run.used]
            loss = self._task_loss(prediction, target, keys)
            objective = self._distilled(loss, prediction, keys)
            if run.penalties:
                objective = objective + torch.stack(run.penalties).mean()
            require_true(torch.isfinite(objective), "La pérdida del tramo no es finita")
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
        sessions = (
            predictor.session_level
            or (self.objective is not None and not self.objective.row_decomposable)
            # El término de IC por sesión necesita todas las etiquetas de cada sesión en un bloque.
            or self.session_ic is not None
        )
        # --- contrastive_memory ---
        if self.uses_contrastive:
            sessions = True
        # --- fin contrastive_memory ---
        dropout = (
            predictor.config.modality_dropout is not None
            or predictor.config.fusion_dropout is not None
        )
        if sessions or dropout:
            # Las operaciones por sesión y el dropout sembrado por lote repiten cada instante
            # entero en un único bloque para conservar exactamente la huella y sus máscaras.
            order = list(
                dict.fromkeys([*order, *(f for batch, _, _ in run.segment for f in batch.flow_ids)])
            )
            if len(order) > size:
                raise ValueError(
                    "Las operaciones por sesión y el dropout por lote necesitan repetir "
                    "todos los flujos"
                )
        total, replayed = 0.0, 0
        for start in range(0, len(order), size):
            group = order[start : start + size]
            states = FlowStates(
                (flow, run.starts[flow]) for flow in group if run.starts[flow] is not None
            )
            # Sin grafo: la repetición no reutiliza nada del cálculo emitido.
            states.detach()
            # Las comprobaciones de todos los lotes del grupo y la de su pérdida se resuelven
            # juntas antes de su backward, en lugar de esperar a la GPU tras cada lote.
            with deferred_checks():
                members, graphs, batch_outputs, penalties = set(group), {}, [], []
                loss_to_record = None
                rank_state = run.rank_memory_start if self.uses_rank_memory else None
                rank_by_pos = {}
                if self.uses_rank_memory:
                    for position, items in run.rank_memory_labels:
                        rank_by_pos.setdefault(position, []).append(items)
                # --- factor_memory ---
                factor_state = None
                if self.uses_factor_memory:
                    factor_state = (
                        predictor.initial_factor_memory(differentiable=True)
                        if run.factor_start in (None, _UNSET)
                        else predictor.detach_factor_memory(run.factor_start)
                    )
                # --- fin factor_memory ---
                # --- contrastive_memory ---
                contrastive = (
                    predictor.detach_contrastive_memory(
                        run.contrastive_start, reattach_initial=True
                    )
                    if self.uses_contrastive
                    else None
                )
                # --- fin contrastive_memory ---
                # --- ic_titans_mlp ---
                ic_mlp_state = run.ic_mlp_memory_start if self.uses_ic_mlp_memory else None
                ic_mlp_by_pos = {}
                if self.uses_ic_mlp_memory:
                    for position, items in run.ic_mlp_memory_labels:
                        ic_mlp_by_pos.setdefault(position, []).append(items)
                # --- fin ic_titans_mlp ---
                # --- lead_lag_memory ---
                lead_lag_state = run.lead_lag_memory_start if self.uses_lead_lag_memory else None
                lead_lag_by_pos = {}
                if self.uses_lead_lag_memory:
                    for position, items in run.lead_lag_memory_labels:
                        lead_lag_by_pos.setdefault(position, []).append(items)
                # --- fin lead_lag_memory ---
                # --- three_factor_consolidation ---
                three_factor_state = run.three_factor_start if self.uses_three_factor else None
                three_factor_by_pos = {}
                if self.uses_three_factor:
                    for position, items in run.three_factor_labels:
                        three_factor_by_pos.setdefault(position, []).append(items)
                # --- fin three_factor_consolidation ---
                for seg_index, (batch, plan, whole) in enumerate(run.segment):
                    if self.uses_rank_memory:
                        for items in rank_by_pos.get(seg_index, ()):
                            rank_state = predictor.settle_rank_memory(
                                rank_state, items, differentiable=True, train=True
                            )
                    # --- ic_titans_mlp ---
                    if self.uses_ic_mlp_memory:
                        for items in ic_mlp_by_pos.get(seg_index, ()):
                            ic_mlp_state = predictor.settle_ic_mlp_memory(
                                ic_mlp_state, items, differentiable=True, train=True
                            )
                    # --- fin ic_titans_mlp ---
                    # --- lead_lag_memory ---
                    if self.uses_lead_lag_memory:
                        for items in lead_lag_by_pos.get(seg_index, ()):
                            lead_lag_state = predictor.settle_lead_lag_memory(
                                lead_lag_state, items, differentiable=True, train=True
                            )
                    # --- fin lead_lag_memory ---
                    # --- three_factor_consolidation ---
                    if self.uses_three_factor:
                        for items in three_factor_by_pos.get(seg_index, ()):
                            three_factor_state = predictor.settle_three_factor_state(
                                three_factor_state, items, differentiable=True, train=True
                            )
                    # --- fin three_factor_consolidation ---
                    rows = [i for i, flow in enumerate(batch.flow_ids) if flow in members]
                    if not rows:
                        continue
                    # Solo la selección de todas las filas de un instante completo lo sigue siendo.
                    complete = whole and len(rows) == len(batch.flow_ids)
                    if len(rows) != len(batch.flow_ids) or rows != list(range(len(batch.flow_ids))):
                        batch = batch.select(rows)
                    if "reliability" in plan:
                        plan = dict(plan, reliability=plan["reliability"][rows])
                    new = tuple(flow for flow in batch.flow_ids if flow not in states)
                    if new:
                        states.put(predictor.initial_state(new, differentiable=True))
                    state = states.gather(batch.flow_ids)
                    if predictor.config.variant == "mac_frozen":
                        memory = predictor.mac.initial_state(
                            len(batch.flow_ids), differentiable=True
                        ).memory
                        state = replace(state, mac=replace(state.mac, memory=memory))
                    # --- contrastive_memory ---
                    if contrastive is not None and predictor.contrastive_memory.needs_commit(
                        contrastive, batch.flow_ids, batch.prediction_at
                    ):
                        contrastive = predictor.commit_contrastive_memory(
                            contrastive, differentiable=True
                        )
                    # --- fin contrastive_memory ---
                    replay_plan = (
                        dict(plan, rank_memory=rank_state) if self.uses_rank_memory else plan
                    )
                    # --- factor_memory ---
                    if self.uses_factor_memory:
                        replay_plan = dict(replay_plan, factor_memory=factor_state)
                    # --- fin factor_memory ---
                    # --- contrastive_memory ---
                    if contrastive is not None:
                        replay_plan = dict(replay_plan, contrastive=contrastive)
                    # --- fin contrastive_memory ---
                    # --- ic_titans_mlp ---
                    if self.uses_ic_mlp_memory:
                        replay_plan = dict(replay_plan, ic_mlp_memory=ic_mlp_state)
                    # --- fin ic_titans_mlp ---
                    # --- lead_lag_memory ---
                    if self.uses_lead_lag_memory:
                        replay_plan = dict(replay_plan, lead_lag_memory=lead_lag_state)
                    # --- fin lead_lag_memory ---
                    # --- three_factor_consolidation ---
                    if self.uses_three_factor:
                        replay_plan = dict(replay_plan, three_factor_state=three_factor_state)
                    # --- fin three_factor_consolidation ---
                    with torch.enable_grad():
                        prepared = predictor.prepare(
                            batch,
                            state,
                            differentiable=True,
                            session_complete=complete,
                            **replay_plan,
                        )
                    if self.uses_rank_memory:
                        rank_state = predictor.stage_rank_memory(
                            rank_state,
                            batch.flow_ids,
                            batch.prediction_at,
                            prepared.rank_memory_keys,
                            prepared.point_predictions,
                            differentiable=True,
                        )
                    # --- factor_memory ---
                    if self.uses_factor_memory:
                        factor_state = prepared.next_factor_memory
                    # --- fin factor_memory ---
                    # --- contrastive_memory ---
                    if contrastive is not None:
                        contrastive = prepared.next_contrastive
                    # --- fin contrastive_memory ---
                    # --- ic_titans_mlp ---
                    if self.uses_ic_mlp_memory:
                        ic_mlp_state = predictor.stage_ic_mlp_memory(
                            ic_mlp_state,
                            batch.flow_ids,
                            batch.prediction_at,
                            prepared.ic_mlp_keys,
                            differentiable=True,
                        )
                    # --- fin ic_titans_mlp ---
                    # --- lead_lag_memory ---
                    if self.uses_lead_lag_memory:
                        lead_lag_state = predictor.stage_lead_lag_memory(
                            lead_lag_state,
                            batch.flow_ids,
                            batch.prediction_at,
                            prepared.lead_lag_memberships,
                            prepared.point_predictions,
                            differentiable=True,
                        )
                    # --- fin lead_lag_memory ---
                    # --- three_factor_consolidation ---
                    if self.uses_three_factor:
                        three_factor_state = predictor.stage_three_factor_state(
                            three_factor_state,
                            batch.flow_ids,
                            batch.prediction_at,
                            prepared.three_factor_emission,
                            prepared.point_predictions,
                            record_pending=True,
                            differentiable=True,
                        )
                    # --- fin three_factor_consolidation ---
                    states.put(prepared.next_state)
                    if prepared.local_control is not None:
                        penalties.append(prepared.local_control.penalty)
                        replayed += len(prepared.local_control.flow_ids)
                    outputs = self._graphs(prepared, batch)
                    batch_index = len(batch_outputs)
                    batch_outputs.append(outputs)
                    for row, key in enumerate(
                        zip(batch.flow_ids, batch.prediction_at, strict=True)
                    ):
                        if key in targets:
                            graphs[key] = (batch_index, row)
                keys = [(flow, at) for flow, at, _ in run.used if flow in members]
                if any(key not in graphs for key in keys):
                    raise ValueError("La repetición del tramo no reproduce sus predicciones")
                terms = []
                if keys:
                    chunks = []
                    pos = 0
                    n_keys = len(keys)
                    while pos < n_keys:
                        b_idx, first_row = graphs[keys[pos]]
                        run_rows = [first_row]
                        pos += 1
                        while pos < n_keys and graphs[keys[pos]][0] == b_idx:
                            run_rows.append(graphs[keys[pos]][1])
                            pos += 1
                        source = batch_outputs[b_idx]
                        if (
                            run_rows[0] == 0
                            and len(run_rows) == source.shape[0]
                            and run_rows[-1] == len(run_rows) - 1
                            and all(
                                run_rows[i] == i for i in range(1, len(run_rows))
                            )
                        ):
                            chunks.append(source)
                        elif (
                            run_rows[-1] - run_rows[0] + 1 == len(run_rows)
                            and all(
                                run_rows[i] == run_rows[0] + i for i in range(1, len(run_rows))
                            )
                        ):
                            chunks.append(source[run_rows[0] : run_rows[-1] + 1])
                        else:
                            index = (
                                torch.tensor(run_rows, dtype=torch.int64, pin_memory=True).to(
                                    device=source.device, non_blocking=True
                                )
                                if source.is_cuda
                                else torch.tensor(run_rows, dtype=torch.int64, device=source.device)
                            )
                            chunks.append(source.index_select(0, index))
                    prediction = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
                    target_values = [targets[key] for key in keys]
                    target = (
                        torch.tensor(target_values, dtype=prediction.dtype, pin_memory=True).to(
                            device=prediction.device, non_blocking=True
                        )
                        if prediction.is_cuda
                        else torch.tensor(
                            target_values, dtype=prediction.dtype, device=prediction.device
                        )
                    )
                    task = self._task_loss(prediction, target, keys)
                    share = len(keys) / len(run.used)
                    loss = task * share
                    loss_to_record = loss.detach()
                    if self.teacher is not None:
                        loss = self._distilled(task, prediction, keys) * share
                    terms.append(loss)
                if penalties:
                    terms.append(sum(penalties) / len(run.penalties))
                objective = sum(terms[1:], terms[0])
                require_true(torch.isfinite(objective), "La pérdida del tramo no es finita")
            objective.backward()
            if loss_to_record is not None:
                total += float(loss_to_record)
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

    def _clip_gradients(self):
        """Recortar gradientes de forma global o por grupo funcional con finitud atómica."""
        if not self.recipe.grad_clip_by_role:
            return torch.nn.utils.clip_grad_norm_(
                self.predictor.parameters(),
                self.recipe.max_grad_norm or math.inf,
                error_if_nonfinite=True,
            )
        # Comprobar finitud en todos los grupos antes de escalar ninguno in-place.
        norms = [
            torch.nn.utils.get_total_norm(
                [p.grad for p in params if p.grad is not None], error_if_nonfinite=True
            ).to(self.device)
            for params in self._clip_param_groups
        ]
        for params, group_norm in zip(self._clip_param_groups, norms, strict=True):
            torch.nn.utils.clip_grads_with_norm_(params, self.recipe.max_grad_norm, group_norm)
        return torch.stack(norms).max()

    def _recency_weight(self, used):
        """Peso causal por recencia del segmento a partir de `decision_at`, o None si apagado."""
        half_life = self.recipe.recency_half_life_years
        if half_life is None:
            return None
        phase = self.train.phase
        return recency_segment_weight(
            (decision_at for _, decision_at, _ in used),
            decision_start=phase.decision_start,
            decision_end=phase.decision_end,
            half_life_years=half_life,
        )

    def _step_optimizer(self, recency_weight=None):
        """Aplicar un paso del optimizador, modulando la tasa efectiva si hay recencia activa."""
        if recency_weight is None:
            self.optimizer.step()
            return
        base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        for group, base in zip(self.optimizer.param_groups, base_lrs, strict=True):
            group["lr"] = base * recency_weight
        try:
            self.optimizer.step()
        finally:
            for group, base in zip(self.optimizer.param_groups, base_lrs, strict=True):
                group["lr"] = base

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
            recency_weight = self._recency_weight(run.used)
            with traced_update(self.traces, self.global_step + 1, self.optimizer, loss) as probe:
                with nvtx_ranges.phase("titans.clip"):
                    norm = self._clip_gradients()
                    probe.clip(norm, self.recipe.max_grad_norm)
                with nvtx_ranges.phase("titans.optimizer"):
                    self._step_optimizer(recency_weight)
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
            if run.rank_memory is not None:
                run.rank_memory = self.predictor.detach_rank_memory(run.rank_memory, parameter_id)
            # --- factor_memory ---
            if run.factor_memory is not None:
                run.factor_memory = self.predictor.detach_factor_memory(
                    run.factor_memory, parameter_id
                )
            # --- fin factor_memory ---
            # --- contrastive_memory ---
            if run.contrastive is not None:
                run.contrastive = self.predictor.detach_contrastive_memory(
                    run.contrastive, parameter_id
                )
            # --- fin contrastive_memory ---
            # --- ic_titans_mlp ---
            if run.ic_mlp_memory is not None:
                run.ic_mlp_memory = self.predictor.detach_ic_mlp_memory(
                    run.ic_mlp_memory, parameter_id
                )
            # --- fin ic_titans_mlp ---
            # --- lead_lag_memory ---
            if run.lead_lag_memory is not None:
                run.lead_lag_memory = self.predictor.detach_lead_lag_memory(
                    run.lead_lag_memory, parameter_id
                )
            # --- fin lead_lag_memory ---
            # --- three_factor_consolidation ---
            if run.three_factor_state is not None:
                run.three_factor_state = self.predictor.detach_three_factor_state(
                    run.three_factor_state, parameter_id
                )
            # --- fin three_factor_consolidation ---
        run.penalties.clear()
        run.graphs.clear()
        run.predictions.clear()
        run.targets.clear()
        run.used.clear()
        run.segment.clear()
        run.starts.clear()
        run.rank_memory_start = None
        run.rank_memory_labels.clear()
        # --- factor_memory ---
        run.factor_start = _UNSET
        # --- fin factor_memory ---
        # --- contrastive_memory ---
        run.contrastive_start = None
        # --- fin contrastive_memory ---
        # --- ic_titans_mlp ---
        run.ic_mlp_memory_start = None
        run.ic_mlp_memory_labels.clear()
        # --- fin ic_titans_mlp ---
        # --- lead_lag_memory ---
        run.lead_lag_memory_start = None
        run.lead_lag_memory_labels.clear()
        # --- fin lead_lag_memory ---
        # --- three_factor_consolidation ---
        run.three_factor_start = None
        run.three_factor_labels.clear()
        # --- fin three_factor_consolidation ---
        run.instants = 0
        run.counters["segments"] += 1

    def predict_partition(self, source, rows, *, stop=None):
        """Predecir un tramo posterior al ajuste del mismo corpus con `predict`."""
        if (
            not training_event_cache.supported(source)
            or source.dataset.identity != self.train.dataset.identity
            or source.phase.decision_start < self.train.phase.decision_end
        ):
            raise ValueError("El tramo no pertenece al corpus y la entrada del ajuste")
        return self.predict(source, rows, stop=stop)

    def _train_pass(self, run, cursor, stop, save):
        predictor, source = self.predictor, self.train
        predictor.train()
        predictor.set_fit_epoch(cursor["epoch"])
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
                source.dataset.pipeline.prepare_workers,
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
        # --- three_factor_consolidation ---
        if run.three_factor_start is not None or run.three_factor_labels:
            raise ValueError("Solo se confirma un recorrido en la barrera posterior a un paso")
        # --- fin three_factor_consolidation ---
        if (
            run.graphs
            or run.segment
            or run.starts
            or run.penalties
            or run.rank_memory_start is not None
            or run.rank_memory_labels
        ):
            raise ValueError("Solo se confirma un recorrido en la barrera posterior a un paso")
        # --- factor_memory ---
        if run.factor_start is not _UNSET:
            raise ValueError("Solo se confirma un recorrido en la barrera posterior a un paso")
        # --- fin factor_memory ---
        # --- contrastive_memory ---
        if run.contrastive_start is not None:
            raise ValueError("Solo se confirma un recorrido en la barrera posterior a un paso")
        # --- fin contrastive_memory ---
        # --- ic_titans_mlp ---
        if run.ic_mlp_memory_start is not None or run.ic_mlp_memory_labels:
            raise ValueError("Solo se confirma un recorrido en la barrera posterior a un paso")
        # --- fin ic_titans_mlp ---
        # --- lead_lag_memory ---
        if run.lead_lag_memory_start is not None or run.lead_lag_memory_labels:
            raise ValueError("Solo se confirma un recorrido en la barrera posterior a un paso")
        # --- fin lead_lag_memory ---
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
            **(
                dict(reliability=None if run.reliability is None else run.reliability.export())
                if self.reliability is not None
                else {}
            ),
            **(
                dict(
                    rank_memory=None
                    if run.rank_memory is None
                    else predictor.export_rank_memory(run.rank_memory)
                )
                if self.uses_rank_memory
                else {}
            ),
            # --- factor_memory ---
            **(
                dict(
                    factor_memory=None
                    if run.factor_memory is None
                    else predictor.export_factor_memory(run.factor_memory)
                )
                if self.uses_factor_memory
                else {}
            ),
            # --- fin factor_memory ---
            # --- contrastive_memory ---
            **(
                dict(
                    contrastive=None
                    if run.contrastive is None
                    else predictor.export_contrastive_memory(run.contrastive)
                )
                if self.uses_contrastive
                else {}
            ),
            # --- fin contrastive_memory ---
            # --- ic_titans_mlp ---
            **(
                dict(
                    ic_mlp_memory=None
                    if run.ic_mlp_memory is None
                    else predictor.export_ic_mlp_memory(run.ic_mlp_memory)
                )
                if self.uses_ic_mlp_memory
                else {}
            ),
            # --- fin ic_titans_mlp ---
            # --- lead_lag_memory ---
            **(
                dict(
                    lead_lag_memory=None
                    if run.lead_lag_memory is None
                    else predictor.export_lead_lag_memory(run.lead_lag_memory)
                )
                if self.uses_lead_lag_memory
                else {}
            ),
            # --- fin lead_lag_memory ---
            # --- three_factor_consolidation ---
            **(
                dict(
                    three_factor_state=None
                    if run.three_factor_state is None
                    else predictor.export_three_factor_state(run.three_factor_state)
                )
                if self.uses_three_factor
                else {}
            ),
            # --- fin three_factor_consolidation ---
            **(
                dict(rank_ic=None if run.rank_ic is None else run.rank_ic.export())
                if self.track_train_rank_ic
                else {}
            ),
        )

    def _restore(self, payload):
        if self.uses_context != ("context" in payload):
            raise ValueError("El recorrido guardado no corresponde al contexto del predictor")
        if (self.horizons is not None) != ("horizons" in payload):
            raise ValueError("El recorrido guardado no corresponde a las cabezas auxiliares")
        if (self.reliability is not None) != ("reliability" in payload):
            raise ValueError("El recorrido guardado no corresponde a la señal de fiabilidad")
        if self.uses_rank_memory != ("rank_memory" in payload):
            raise ValueError(
                "El recorrido guardado no corresponde a la memoria transversal de rango"
            )
        # --- factor_memory ---
        if self.uses_factor_memory != ("factor_memory" in payload):
            raise ValueError("El recorrido guardado no corresponde a la memoria factorial")
        # --- fin factor_memory ---
        # --- contrastive_memory ---
        if self.uses_contrastive != ("contrastive" in payload):
            raise ValueError(
                "El recorrido guardado no corresponde a la memoria contrastiva del predictor"
            )
        # --- fin contrastive_memory ---
        # --- ic_titans_mlp ---
        if self.uses_ic_mlp_memory != ("ic_mlp_memory" in payload):
            raise ValueError("El recorrido guardado no corresponde a la memoria IC-Titans-MLP")
        # --- fin ic_titans_mlp ---
        # --- lead_lag_memory ---
        if self.uses_lead_lag_memory != ("lead_lag_memory" in payload):
            raise ValueError(
                "El recorrido guardado no corresponde a la memoria rápida de desfase entre grupos"
            )
        # --- fin lead_lag_memory ---
        # --- three_factor_consolidation ---
        if self.uses_three_factor != ("three_factor_state" in payload):
            raise ValueError(
                "El recorrido guardado no corresponde a la consolidación de tres factores"
            )
        # --- fin three_factor_consolidation ---
        if self.track_train_rank_ic != ("rank_ic" in payload):
            raise ValueError("El recorrido guardado no corresponde al IC de rangos del ajuste")
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
        if self.reliability is not None and payload["reliability"] is not None:
            run.reliability = ReliabilitySignal.restore(self.reliability, payload["reliability"])
        if self.uses_rank_memory and payload["rank_memory"] is not None:
            run.rank_memory = self.predictor.restore_rank_memory(payload["rank_memory"])
        # --- factor_memory ---
        if self.uses_factor_memory and payload["factor_memory"] is not None:
            run.factor_memory = self.predictor.restore_factor_memory(payload["factor_memory"])
        # --- fin factor_memory ---
        # --- contrastive_memory ---
        if self.uses_contrastive and payload["contrastive"] is not None:
            run.contrastive = self.predictor.restore_contrastive_memory(payload["contrastive"])
        # --- fin contrastive_memory ---
        # --- ic_titans_mlp ---
        if self.uses_ic_mlp_memory and payload["ic_mlp_memory"] is not None:
            run.ic_mlp_memory = self.predictor.restore_ic_mlp_memory(payload["ic_mlp_memory"])
        # --- fin ic_titans_mlp ---
        # --- lead_lag_memory ---
        if self.uses_lead_lag_memory and payload["lead_lag_memory"] is not None:
            run.lead_lag_memory = self.predictor.restore_lead_lag_memory(payload["lead_lag_memory"])
        # --- fin lead_lag_memory ---
        # --- three_factor_consolidation ---
        if self.uses_three_factor and payload["three_factor_state"] is not None:
            run.three_factor_state = self.predictor.restore_three_factor_state(
                payload["three_factor_state"]
            )
        # --- fin three_factor_consolidation ---
        if self.track_train_rank_ic and payload["rank_ic"] is not None:
            run.rank_ic = _SessionRankIC.restore(
                self.recipe.selection["min_assets"], payload["rank_ic"]
            )
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
            seed = self.shrink.fresh_seed(self.predictor.config.seed)
            fresh = type(self.predictor)(
                replace(self.predictor.config, seed=seed),
                device=self.device,
                dtype=self.predictor.head.weight.dtype,
            )
            shrink_perturb(self.trainable, dict(fresh.named_parameters()), self.shrink)
            self.predictor._seal_parameters()
            report["shrink_perturb"] = dict(
                configuration=self.shrink.identity(),
                fresh_seed=seed,
                parameters_before_sha256=before,
                fresh_parameters_sha256=fresh._parameter_id,
                parameters_after_sha256=self.predictor._parameter_id,
            )
        if self.weight_average is not None:
            self.shadow = ShadowWeights(self.weight_average, self.trainable)
            if state is not None:
                self.shadow.load_state_dict(state["weight_average"])

    def _new_train_pass(self):
        return _Pass(
            rank_ic=(
                _SessionRankIC(self.recipe.selection["min_assets"])
                if self.track_train_rank_ic
                else None
            )
        )

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

    def _load_best_for_resume(self, report):
        """Restaurar pesos de ajuste (y sombra F2 si procede) del mejor punto de control."""
        state = load_training_state(
            self.output / "checkpoints",
            expected_identity=self.identity,
            selection="best",
            expected_sha256=report["best_checkpoint"]["sha256"],
        )
        trained = "trained_model" if self.emits_other_weights else "model"
        self.predictor.load_state_dict(state[trained])
        if self.shadow is not None:
            self.shadow.load_state_dict(state["weight_average"])
        return state

    def _average_emitted_model(self, checkpoints, report, epochs):
        """Promediar pesos compartidos (y adaptadores) de `epochs` manteniendo memorias de la mejor.

        Los tensores en `roles["shared"]` y `roles["adapters"]` se acumulan en float64 en el
        orden determinista de `epochs`, se dividen por `len(epochs)` y se devuelven al tipo del
        parámetro. `mac.persistent` (`persistent_memory`), `mac.memory.initial_weights.*`
        (`initial_fast_weights`), todos los búferes, `_extra_state` y el estado en línea se
        toman literalmente de `epochs[0]` (la mejor época), sin promediar.
        """
        best_ep = epochs[0]
        loaded = {}
        for ep in epochs:
            ref = report.get("epoch_states", {}).get(str(ep))
            if ref is None:
                raise ValueError(f"Falta el estado fijado de la época {ep} para promediar pesos")
            loaded[ep] = load_training_state(
                checkpoints,
                expected_identity=self.identity,
                selection="pinned",
                expected_sha256=ref["sha256"],
            )
        averaged_roles = ["shared", *(["adapters"] if "adapters" in self.roles else [])]
        best_only_roles = [
            role for role in ("persistent_memory", "initial_fast_weights") if self.roles.get(role)
        ]
        averaged_names = {name for role in averaged_roles for name in self.roles.get(role, [])}
        best_state = loaded[best_ep]
        best_model = best_state["model"]
        averaged_model = {}
        count = len(epochs)
        for name, value in best_model.items():
            if name in averaged_names and isinstance(value, torch.Tensor):
                total = torch.zeros_like(value, dtype=torch.float64)
                for ep in epochs:
                    total.add_(loaded[ep]["model"][name].to(dtype=torch.float64))
                mean_value = (total / count).to(dtype=value.dtype)
                if not torch.isfinite(mean_value).all():
                    raise ValueError(f"El promedio de pesos produjo valores no finitos en {name}")
                averaged_model[name] = mean_value
            elif isinstance(value, torch.Tensor):
                averaged_model[name] = value.detach().clone()
            else:
                averaged_model[name] = value
        metadata = dict(
            mode="average",
            requested=emit_average_count(self.recipe.selection),
            epochs=list(epochs),
            best_epoch=best_ep,
            averaged_roles=averaged_roles,
            best_only_roles=best_only_roles,
            memories_and_online_state_epoch=best_ep,
            source_checkpoints={str(ep): dict(report["epoch_states"][str(ep)]) for ep in epochs},
        )
        return averaged_model, best_state, metadata

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
        la última época y `joint_epoch` elige el estado fijado de una de ellas. Con opciones
        adaptativas (`smoothing`, `learning_rate_decay`, `emit`) conserva los estados fijados
        necesarios, aplica reducciones deterministas y reanudables, promedia los pesos
        compartidos de las k mejores épocas sin promediar memorias ni estado en línea, y
        guarda el informe de parada junto a los resultados.
        """
        output, checkpoints = self.output, self.output / "checkpoints"
        if isinstance(self.optimizer, torch.optim.Optimizer):
            require_learning_allowed("ChronologicalTrainer.run de Titans-MAC")
        if type(resume) is not bool or output.is_symlink() or output.exists() != resume:
            raise ValueError("Usa una ejecución nueva o solicita continuar una existente")
        options = self.recipe.selection
        common = options.get("stopping") == COMMON_EPOCH
        if common and self.epoch_scorer is None:
            raise ValueError("La época común necesita puntuar cada validación con su IC")
        # Con el IC de rangos como métrica, cada validación se puntúa también con él.
        ranked = options["metric"] == SESSION_RANK_IC
        if ranked and self.epoch_scorer is None:
            raise ValueError("El IC de rangos por sesión necesita puntuar cada validación")
        window = smoothing_window(options)
        decay_cfg = lr_decay_config(options)
        emit_k = emit_average_count(options)
        keep_epoch_states = window > 1 or emit_k > 1
        if decay_cfg is not None and any(
            "lr" not in group
            or type(group["lr"]) not in (int, float)
            or not math.isfinite(group["lr"])
            or group["lr"] <= 0
            for group in self.optimizer.param_groups
        ):
            raise ValueError(
                "learning_rate_decay requiere una tasa finita y positiva en cada grupo"
            )
        stop = stop or StopRequest()
        report_path = output / "run.json"
        cursor, run = dict(epoch=0, phase="validation"), None
        self.lr_decays, self.epoch_lrs = [], {}
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
            if decay_cfg is not None:
                self.lr_decays = list(state.get("lr_decays", []))
                self.epoch_lrs = dict(state.get("epoch_lrs", {}))
            if self.traces is not None:
                self.traces.restore(state.get("traces"))
            if state["run"] is not None:
                self.predictor.train()
                self.predictor.set_fit_epoch(cursor["epoch"])
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

        def save(
            position,
            current=None,
            *,
            best=False,
            pin=None,
            keep_pinned=None,
            best_sha256=None,
            retained=None,
            promoted_epoch=None,
        ):
            # `pin` es la época evaluada cuyo estado se conserva para época común, suavizado o
            # promedio de pesos. `keep_pinned` y `best_sha256` actualizan `latest.json` de forma
            # atómica en la misma escritura.
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
            if decay_cfg is not None:
                state["lr_decays"] = list(self.lr_decays)
                state["epoch_lrs"] = dict(self.epoch_lrs)
            if self.traces is not None:
                state["traces"] = self.traces.checkpoint()
            path = save_training_state(
                checkpoints,
                state,
                identity=self.identity,
                best=best,
                pin=pin is not None,
                keep_pinned=keep_pinned,
                best_sha256=best_sha256,
            )
            reference = dict(path=str(path.relative_to(output)), sha256=sha256(path))
            report.update(recovery_checkpoint=reference, global_step=self.global_step)
            if best:
                report["best_checkpoint"] = reference
            elif promoted_epoch is not None:
                report["best_checkpoint"] = dict(report["epoch_states"][str(promoted_epoch)])
            if pin is not None:
                report.setdefault("epoch_states", {})[str(pin)] = reference
            if retained is not None and "epoch_states" in report:
                allowed = {str(ep) for ep in retained}
                report["epoch_states"] = {
                    key: value for key, value in report["epoch_states"].items() if key in allowed
                }
            report.update(selection=self.selection, history=self.history, cursor=position)
            atomic_json(report_path, report)

        if not resume:
            save(cursor)
        started = time.perf_counter()
        try:
            while cursor["phase"] != "done":
                epoch = cursor["epoch"]
                if cursor["phase"] == "lr_decay":
                    factor = decay_cfg["factor"]
                    old_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
                    new_lrs = [lr * factor for lr in old_lrs]
                    if any(not math.isfinite(lr) or lr <= 0 for lr in new_lrs):
                        raise ValueError(
                            "La tasa de aprendizaje reducida dejó de ser finita y positiva"
                        )
                    best_state = self._load_best_for_resume(report)
                    self.optimizer.load_state_dict(best_state["optimizer"])
                    for group, lr in zip(self.optimizer.param_groups, new_lrs, strict=True):
                        group["lr"] = lr
                    restored_epoch = self.selection["best_epoch"]
                    self.selection = apply_lr_decay(self.selection, options)
                    self.lr_decays.append(
                        dict(
                            reduction=self.selection["reductions_done"],
                            at_epoch=epoch,
                            restored_epoch=restored_epoch,
                            factor=factor,
                            previous_learning_rate=old_lrs[0],
                            new_learning_rate=new_lrs[0],
                        )
                    )
                    cursor = dict(epoch=epoch, phase="train", event=0, stage="start")
                    save(cursor)
                    if stop.requested:
                        raise Paused
                    continue
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
                    # El IC de rangos puntúa todas, también la del estado inicial.
                    scored = ranked or (common and epoch >= 1)
                    rows = self.epoch_scorer.rows() if scored else None
                    with self._emitted_weights():
                        metrics = self.evaluate(self.validation, stop=stop, rows=rows)
                    if rows is not None:
                        metrics.update(self.epoch_scorer.score(epoch, rows, metrics))
                    score = selection_score(options, metrics)
                    if self.trace is not None or self.traces is not None:
                        event = dict(
                            kind="validation",
                            epoch=epoch,
                            global_step=self.global_step,
                            score=score,
                        )
                        if self.traces is not None:
                            self.traces.validation(event)
                        if self.trace is not None:
                            self.trace(event, self._trace_modules())
                    self.selection = (
                        initial_selection(score, options)
                        if epoch == 0
                        else advance_selection(self.selection, score, epoch, options)
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
                    if finished:
                        cursor = dict(epoch=epoch, phase="done")
                    elif decision == DECAY_LR:
                        cursor = dict(epoch=epoch, phase="lr_decay")
                    else:
                        cursor = dict(epoch=epoch, phase="train", event=0, stage="start")
                    if keep_epoch_states:
                        needed = retained_epochs(self.selection, options)
                        existing = report.get("epoch_states", {})
                        keep_digests = {
                            existing[str(ep)]["sha256"] for ep in needed if str(ep) in existing
                        }
                        pin_epoch = epoch if epoch in needed else None
                        best_ep = self.selection["best_epoch"]
                        if self.selection["best_score"] is None or best_ep == epoch:
                            best_flag = True
                            promote_sha = None
                            promoted_ep = None
                        else:
                            best_flag = False
                            target_ref = existing.get(str(best_ep))
                            current_best = report.get("best_checkpoint")
                            if (
                                target_ref is not None
                                and current_best is not None
                                and current_best["sha256"] != target_ref["sha256"]
                            ):
                                promote_sha = target_ref["sha256"]
                                promoted_ep = best_ep
                            else:
                                promote_sha = None
                                promoted_ep = None
                        save(
                            cursor,
                            best=best_flag,
                            pin=pin_epoch,
                            keep_pinned=keep_digests,
                            best_sha256=promote_sha,
                            retained=needed,
                            promoted_epoch=promoted_ep,
                        )
                    else:
                        # Con AWAIT, el cursor ya confirmado espera al grupo al empezar la época.
                        save(
                            cursor,
                            best=self.selection["last_improved"],
                            pin=epoch if common and rows is not None else None,
                        )
                    if stop.requested and not finished:
                        raise Paused
                    continue
                if decay_cfg is not None:
                    self.epoch_lrs[str(epoch + 1)] = float(self.optimizer.param_groups[0]["lr"])
                self.train_metrics = self._train_pass(
                    run or self._new_train_pass(), cursor, stop, save
                )
                run = None
                cursor = dict(epoch=epoch + 1, phase="validation")
                save(cursor)

            emitted_epochs = selected_emit_epochs(self.selection, options)
            if common:
                chosen = chosen_state(
                    checkpoints, self.identity, report, self.history, self.selection, joint_epoch
                )
                report.update(chosen)
                self._load_best(report)
                emitted_epochs = [chosen["best_epoch"]]
            elif emit_k > 1 and len(emitted_epochs) > 1:
                averaged_model, best_state, emit_meta = self._average_emitted_model(
                    checkpoints, report, emitted_epochs
                )
                self.predictor.load_state_dict(averaged_model)
                final_state = dict(
                    global_step=self.global_step,
                    cursor=cursor,
                    model=averaged_model,
                    optimizer=best_state["optimizer"],
                    rng=capture_rng(str(self.device)),
                    selection=self.selection,
                    history=self.history,
                    train_metrics=self.train_metrics,
                    run=None,
                )
                if self.emits_other_weights and "trained_model" in best_state:
                    final_state["trained_model"] = best_state["trained_model"]
                if self.shadow is not None and "weight_average" in best_state:
                    final_state["weight_average"] = best_state["weight_average"]
                if decay_cfg is not None:
                    final_state["lr_decays"] = list(self.lr_decays)
                    final_state["epoch_lrs"] = dict(self.epoch_lrs)
                if self.traces is not None:
                    final_state["traces"] = self.traces.checkpoint()
                saved_path = save_training_state(
                    checkpoints, final_state, identity=self.identity, best=True
                )
                reference = dict(
                    path=str(saved_path.relative_to(output)), sha256=sha256(saved_path)
                )
                chosen = dict(
                    best_checkpoint=reference,
                    recovery_checkpoint=reference,
                    best_epoch=emitted_epochs[0],
                    best_score=self.selection["best_raw_score"],
                    averaged_epochs=list(emitted_epochs),
                    emit=emit_meta,
                )
                if window > 1:
                    chosen["best_smoothed_score"] = self.selection["best_score"]
                report.update(chosen)
            else:
                if window > 1:
                    chosen = dict(
                        best_epoch=self.selection["best_epoch"],
                        best_score=self.selection["best_raw_score"],
                        best_smoothed_score=self.selection["best_score"],
                    )
                else:
                    chosen = dict(
                        best_epoch=self.selection["best_epoch"],
                        best_score=self.selection["best_score"],
                    )
                report.update(chosen)
                self._load_best(report)
            if decay_cfg is not None:
                report["learning_rate_reductions"] = list(self.lr_decays)
                lr_map = {int(k): float(v) for k, v in self.epoch_lrs.items()}
            else:
                lr_map = {
                    entry["epoch"]: float(self.recipe.learning_rate)
                    for entry in self.history
                    if entry["epoch"] >= 1
                }
            stopped_early = cursor["epoch"] < self.recipe.epochs
            stopping_report = build_stopping_report(
                self.history,
                self.selection,
                options,
                epochs=self.recipe.epochs,
                stopped_early=stopped_early,
                emitted_epochs=emitted_epochs,
                lr_decays=self.lr_decays,
                learning_rates=lr_map,
                best_epoch=report["best_epoch"],
            )
            stopping_path = output / "stopping_report.json"
            atomic_json(stopping_path, stopping_report)
            report.update(
                status="completed",
                stopped_early=stopped_early,
                stopping=self.recipe.selection.get("stopping", VALIDATION_PLATEAU),
                plateau_epoch=self.selection.get("plateau_epoch"),
                stopping_report=stopping_report,
                stopping_report_file=dict(path=stopping_path.name, sha256=sha256(stopping_path)),
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
