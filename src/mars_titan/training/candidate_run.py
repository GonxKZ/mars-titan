"""Recorrido cronológico de la GRU candidata con banco episódico y etiquetas maduras.

El optimizador de PyTorch ajusta los parámetros del módulo nativo `Candidate` a través
del enlace `_episodic_native`, que comparte tensores y autograd con esta instalación de
PyTorch. Las proyecciones fijas del codec no son parámetros. La GRU reinicia su estado en
cada ventana, así que no hay BPTT entre instantes: cada predicción depende de los
parámetros vigentes, de su ventana y de una instantánea constante del banco.

En cada evento se resuelven primero las etiquetas maduras contra la predicción emitida,
después se actualiza si el tramo está completo y por último se predicen las entradas del
instante. Las predicciones leen el banco confirmado antes de admitir las etiquetas de ese
mismo evento, como `FinancialSession`, y la admisión se aplica al terminar el evento.
"""

import hashlib
import importlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from torch.utils.checkpoint import checkpoint

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.session_metrics import SessionErrors
from mars_titan.evaluation.splits import PARTITIONS, stopping_rule
from mars_titan.memory.candidate_bank import (
    CandidateBankConfig,
    CandidateEpisodeBank,
    CandidateEpisodes,
)
from mars_titan.memory.financial_observations import FinancialObservationSource
from mars_titan.models.candidate.episode_codec import FrozenCandidateCodec
from mars_titan.models.candidate.frozen_consumer import _numerics
from mars_titan.models.candidate.input_adapter import CandidateInputAdapter
from mars_titan.models.quantile_head import (
    CONTRACT,
    MEDIAN_INDEX,
    PINBALL,
    QUANTILE_COLUMNS,
    median,
    pinball_loss,
)
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.financial_inputs import DecisionBatch, validated_cpu_batch

from .checkpoints import (
    StopRequest,
    capture_rng,
    load_training_state,
    restore_rng,
    save_training_state,
)
from .financial_run import _compatible
from .learning_hold import require_learning_allowed
from .selection import VALIDATION_PLATEAU, advance_selection, initial_selection, validate_selection

RECIPE = "candidate_gru_chronological_v1"
ADMISSIONS = ("m0", "m1")
# L1 sobre la mediana es la salida escalar de retroceso que declara #22.
LOSSES = (PINBALL, "mae")
HELDOUT = ("calibration", "evaluation")
_ROLES = ("encoder", "episodic_read", "refiner", "head")
_OWN_MODULES = (
    "mars_titan.training.candidate_run",
    "mars_titan.memory.candidate_bank",
    "mars_titan.memory.financial_observations",
    "mars_titan.training.checkpoints",
    "mars_titan.training.selection",
    "mars_titan.evaluation.session_metrics",
    "mars_titan.models.quantile_head",
)
_ROW_CHUNK = 65_536


def _default_selection():
    return dict(metric="session_mae", patience=5, min_delta=1e-05, stopping="fixed_budget")


@dataclass(frozen=True)
class CandidateRecipe:
    """Hiperparámetros declarados antes de ejecutar. `update_instants` no es BPTT."""

    admission: str = "m1"
    refinements: int = 1
    bank_capacity: int = 1024
    bank_seed: int = 73
    update_instants: int = 8
    loss: str = PINBALL
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    epochs: int = 30
    selection: dict = field(default_factory=_default_selection)
    block_rows: int = 128
    # None conserva un único backward por tramo. Un entero propaga cada evento al madurar
    # sus etiquetas, en grupos de bloques de predicción completos de hasta ese número de filas.
    accumulation_rows: int | None = None
    # Guardar solo las entradas de cada bloque y repetir su forward durante el backward.
    recompute: bool = False
    checkpoint_updates: int = 256
    checkpoint_seconds: float = 900.0

    def __post_init__(self):
        validate_selection(self.selection, epochs=self.epochs)
        integers = (
            (self.update_instants, 1, 256),
            (self.epochs, 1, 1000),
            (self.block_rows, 1, 256),
            (self.checkpoint_updates, 1, 1_000_000),
            (self.bank_capacity, 1, 8192),
            (self.bank_seed, 0, 2**64 - 1),
        )
        numbers = (self.learning_rate, self.weight_decay, self.checkpoint_seconds)
        clip, rows = self.max_grad_norm, self.accumulation_rows
        if (
            self.admission not in ADMISSIONS
            or type(self.refinements) is not int
            or self.refinements not in (1, 2, 4)
            or type(self.recompute) is not bool
            or self.loss not in LOSSES
            or any(type(v) is not int or not low <= v <= high for v, low, high in integers)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers)
            or not 0 < self.learning_rate <= 1
            or self.weight_decay < 0
            or self.checkpoint_seconds <= 0
            or (
                rows is not None
                and (type(rows) is not int or not self.block_rows <= rows <= 65_536)
            )
            or (
                clip is not None
                and (type(clip) not in (int, float) or not math.isfinite(clip) or clip <= 0)
            )
        ):
            raise ValueError(
                "La receta necesita banco M0/M1, K = 1, 2 o 4, pérdida, presupuesto, "
                "optimizador, acumulación, recomputación y checkpoints válidos"
            )

    def identity(self):
        return dict(
            schema_version=1,
            recipe=RECIPE,
            **asdict(self),
            optimizer="AdamW",
            update_unit="decision_instants_per_segment_no_bptt",
            label_rule="loss_only_after_maturity_event_then_next_segment_update",
            loss_reduction=(
                "mean_over_matured_labels_of_segment"
                if self.accumulation_rows is None
                else "row_sum_backward_at_maturity_by_whole_prediction_blocks_"
                "then_gradient_divided_by_segment_labels"
            ),
            bank_rule="snapshot_before_event_labels_then_admit_mature_labels",
            bank_reset="each_pass_starts_empty",
            refinements_counted_as_memory_updates=False,
            selection_metric_definition="SessionErrors.session_mae_on_issued_medians",
        )


def load_recipe(path, *, variant):
    """Leer la receta, su variante y la regla de selección del protocolo referenciado."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise ValueError("La receta no es un archivo regular de hasta 64 KiB")
    document = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "schema_version",
        "recipe_name",
        "status",
        "arm",
        "protocol",
        "model",
        "recipe",
        "variants",
        "principal",
        "pending",
    }
    if (
        not isinstance(document, dict)
        or set(document) != fields
        or document["schema_version"] != 1
        or document["recipe_name"] != RECIPE
        or not isinstance(document["variants"], dict)
        or document["principal"] not in document["variants"]
        or variant not in document["variants"]
    ):
        raise ValueError("La receta no conserva su esquema, variantes y brazo principal")
    protocol_path = (path.parent / document["protocol"]).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rule = stopping_rule(protocol)
    options = {**document["recipe"], **document["variants"][variant]}
    recipe = CandidateRecipe(**options)
    if recipe.epochs != rule["max_epochs"] or recipe.selection != {
        key: value for key, value in rule.items() if key != "max_epochs"
    }:
        raise ValueError("La selección o el presupuesto no coinciden con el protocolo")
    if document["model"].get("seeds") != protocol["seeds"]:
        raise ValueError("Las semillas del brazo no son las del protocolo")
    return recipe, dict(document, protocol_sha256=sha256(protocol_path))


def parameter_roles(model, admission):
    """Separar parámetros por función. Con M0 la lectura episódica no recibe gradiente."""
    roles = {role: [] for role in _ROLES}
    for name in model.named_parameters():
        if name.startswith(("query_", "value_")):
            roles["episodic_read"].append(name)
        elif name.startswith("update_") or name == "step_logit":
            roles["refiner"].append(name)
        elif name.startswith("head_"):
            roles["head"].append(name)
        else:
            roles["encoder"].append(name)
    inert = roles.pop("episodic_read") if admission == "m0" else []
    return roles, inert


def _parameters_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters().items()}


def _load_parameters(model, values):
    """Copiar en el módulo vigente para conservar las referencias del optimizador."""
    current = model.named_parameters()
    if not isinstance(values, dict) or list(values) != list(current):
        raise ValueError("El estado no contiene exactamente los parámetros del candidato")
    for name, value in values.items():
        target = current[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != target.shape
            or value.dtype != target.dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("Un parámetro guardado cambia forma, precisión o finitud")
    with torch.no_grad():
        for name, value in values.items():
            current[name].copy_(value.to(current[name].device))


def _adapter_contract(identity, *, window=True):
    """La identidad del adaptador sin sus pesos ni su modo, que cambian al ajustar.

    Sin `window` se quitan también la huella de la vista y la del índice de entrada, las
    dos únicas partes que cambian al aplicar el estado a otra ventana de la misma edición.
    """
    contract = {k: v for k, v in identity.items() if k not in ("parameters_sha256", "training")}
    if not window and isinstance(contract.get("input_specification"), dict):
        contract["input_specification"] = {
            k: v
            for k, v in contract["input_specification"].items()
            if k not in ("source_sha256", "view_sha256")
        }
    return contract


def _read_report(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024**2:
        raise ValueError("El informe de la ejecución no es regular o supera 16 MiB")
    return json.loads(path.read_text())


class _Pause(Exception):
    """Solicitud de parada atendida en una barrera ya confirmada."""


@dataclass
class _Pending:
    """Predicción emitida a la espera de su etiqueta. key y value solo existen con M1."""

    issued: float
    sample: str | None = None
    quantiles: np.ndarray | None = None
    available: int | None = None
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    block: int | None = None


def _counters():
    return dict(
        observations=0,
        warmup_observations=0,
        predictions=0,
        labels=0,
        labels_in_loss=0,
        labels_without_graph=0,
        admitted=0,
        updates=0,
        segments=0,
        loss_sum=0.0,
        unresolved=0,
    )


@dataclass
class _Pass:
    """Estado de un recorrido. Se persisten pendientes, banco, admisión, errores y contadores."""

    bank: CandidateEpisodeBank | None = None
    staged: CandidateEpisodes | None = None
    memory: object = None
    pending: dict = field(default_factory=dict)
    errors: SessionErrors = field(default_factory=SessionErrors)
    counters: dict = field(default_factory=_counters)
    graphs: dict = field(default_factory=dict)
    predictions: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    outstanding: dict = field(default_factory=dict)
    used: list = field(default_factory=list)
    instants: int = 0
    next_block: int = 0
    accumulated: int = 0
    accumulated_loss: object = 0.0


class _PredictionRows:
    """Filas del contrato de predicciones en bloques Arrow compactos."""

    def __init__(self, dtype):
        self.dtype, self.tables = dtype, []
        self._reset()

    def _reset(self):
        self.columns = dict(sample=[], flow=[], at=[], target=[], quantiles=[])

    def add(self, entry, flow, at, target):
        for name, value in zip(
            ("sample", "flow", "at", "target", "quantiles"),
            (entry.sample, flow, at, target, entry.quantiles),
            strict=True,
        ):
            self.columns[name].append(value)
        if len(self.columns["at"]) >= _ROW_CHUNK:
            self.flush()

    def flush(self):
        columns = self.columns
        if not columns["at"]:
            return
        levels = np.stack(columns["quantiles"]).astype(self.dtype, copy=False)
        table = {
            "sample_id": columns["sample"],
            "asset_id": columns["flow"],
            "market": [flow.split("/", 1)[0] for flow in columns["flow"]],
            "prediction_at": pa.array(
                np.asarray(columns["at"], dtype=np.int64), type=pa.timestamp("us", tz="UTC")
            ),
            "target": np.asarray(columns["target"], dtype=np.float64),
            # Los mismos bits que la mediana emitida y que su columna de nivel.
            "prediction": levels[:, MEDIAN_INDEX],
            "zero": np.zeros(len(columns["at"]), dtype=np.float64),
        }
        table.update(zip(QUANTILE_COLUMNS, levels.T, strict=True))
        self.tables.append(pa.table(table))
        self._reset()


class CandidateChronologicalPredictor:
    """Recorrer fases declaradas con parámetros congelados, banco vacío y etiquetas maduras.

    Es la parte del recorrido que no ajusta nada. El entrenador la usa para validar y
    predecir fuera del ajuste, y la variante B la usa para predecir una ventana posterior
    con el estado elegido en su ancla. El banco solo admite claves y valores del codec
    fijo con etiquetas maduras. K repite lectura y refinamiento sobre la misma instantánea.
    """

    def __init__(
        self, adapter, recipe, *, sources, output, world="candidate_gru", fold="0", audit=False
    ):
        if type(adapter) is not CandidateInputAdapter or type(recipe) is not CandidateRecipe:
            raise ValueError("El recorrido necesita el adaptador identificado y su receta")
        if torch.is_inference_mode_enabled():
            raise ValueError("El recorrido no admite inference_mode")
        if adapter.specification.input_policy != HISTORICAL_MASKED:
            raise ValueError("El recorrido solo admite la política histórica con máscaras")
        sources = dict(sources)
        if (
            not sources
            or not set(sources) <= set(PARTITIONS)
            or any(type(s) is not FinancialObservationSource for s in sources.values())
            or any(s.phase.partition != name for name, s in sources.items())
            or len({s.dataset.identity for s in sources.values()}) != 1
        ):
            raise ValueError("Las fases deben ser las particiones declaradas del mismo corpus")
        # Las particiones del mismo corpus son disjuntas y ordenadas por construcción.
        if any(not _compatible(adapter.specification, s.specification()) for s in sources.values()):
            raise ValueError("Las vistas no conservan la entrada del candidato")
        model = adapter.model
        if any(not isinstance(text, str) or not 0 < len(text) <= 128 for text in (world, fold)):
            raise ValueError("El ámbito del banco necesita mundo y fold explícitos")
        parameter = model.named_parameters()["head_weight"]
        self.device, self.dtype = str(parameter.device), parameter.dtype
        if self.device not in {"cpu", "cuda:0"}:
            raise ValueError("El recorrido admite cpu o cuda:0 explícitos")
        if self.device == "cuda:0" and not torch.cuda.is_available():
            raise RuntimeError("cuda:0 no está disponible y no se cambia de dispositivo")
        self.adapter, self.model, self.recipe = adapter, model, recipe
        self.native, self.specification = adapter.native, adapter.specification
        self.sources = sources
        self.world, self.fold = world, fold
        self.output, self.audit = Path(output), [] if audit else None
        dataset = next(iter(sources.values())).dataset
        for protected in (*dataset.roots.values(), *(s.path.parent for s in sources.values())):
            outside_source(protected, self.output)
            outside_source(self.output, protected)
        self.codec = FrozenCandidateCodec(adapter) if recipe.admission == "m1" else None
        self._empty = model.empty_memory()
        model.eval()

    def _new_bank(self, partition):
        if self.codec is None:
            return None
        return CandidateEpisodeBank(
            self.native,
            CandidateBankConfig(capacity=self.recipe.bank_capacity, seed=self.recipe.bank_seed),
            codec_id=self.codec.fingerprint(),
            representation_id=self.model.representation_id(),
            dtype=self.dtype,
            world=self.world,
            partition=partition,
            fold=self.fold,
        )

    def _memory(self, run):
        """Instantánea del banco confirmado, común a todos los bloques del evento."""
        if run.memory is None:
            if run.bank is None:
                run.memory = self._empty
            else:
                view = run.bank.read_view()
                run.memory = self.model.snapshot(
                    view["keys"].to(self.device),
                    view["values"].to(self.device),
                    view["returns"].to(self.device),
                    view["ids"],
                    self.model.representation_id(),
                )
        return run.memory

    def _forward(self, batch, memory, *, grad):
        with torch.inference_mode(False), torch.no_grad():
            tensors = DecisionBatch.from_validated(batch, device=self.device, dtype=self.dtype)
        inputs = self.native.CandidateInputs(
            *(tensors.inputs[name] for name in MODALITIES), tensors.presence
        )
        refinements = self.recipe.refinements
        if grad and self.recipe.recompute:
            # Mismos parámetros, instantánea y K en la repetición. La salida emitida no cambia.
            return checkpoint(
                lambda: self.model.forward(inputs, memory, refinements).quantiles,
                use_reentrant=False,
            )
        with torch.set_grad_enabled(grad):
            return self.model.forward(inputs, memory, refinements).quantiles

    def _observe(self, run, source, event, *, train):
        """Predecir cada bloque con la instantánea previa a las etiquetas del evento."""
        warmup = event.at < source.phase.decision_start
        partition = source.phase.partition
        for raw in event.inputs:
            batch = validated_cpu_batch(raw, self.specification)
            size = len(batch.flow_ids)
            run.counters["observations"] += size
            if warmup:
                # La GRU no conserva estado entre ventanas. El calentamiento no predice.
                run.counters["warmup_observations"] += size
                continue
            seen = 0 if run.bank is None else run.bank.seen
            quantiles = self._forward(batch, self._memory(run), grad=train)
            block, run.next_block = run.next_block, run.next_block + 1
            if train:
                run.outstanding[block] = size
            emitted = quantiles.detach()
            issued = emitted[:, MEDIAN_INDEX].cpu().tolist()
            levels = None if train else emitted.cpu().numpy()
            encoded = None if self.codec is None else self.codec.encode(batch)
            for row, (flow, at) in enumerate(zip(batch.flow_ids, batch.prediction_at, strict=True)):
                if (flow, at) in run.pending:
                    raise ValueError("La decisión ya tiene una predicción pendiente")
                entry = _Pending(issued[row])
                if not train:
                    entry.sample, entry.quantiles = batch.sample_ids[row], levels[row]
                if encoded is not None:
                    entry.available = batch.input_available_at[row]
                    entry.key = torch.from_numpy(encoded.keys[row].copy())
                    entry.value = torch.from_numpy(encoded.values[row].copy())
                run.pending[flow, at] = entry
                if train:
                    entry.block = block
                    run.graphs[flow, at] = quantiles[row]
                if self.audit is not None:
                    self.audit.append(("prediction", partition, flow, at, issued[row], seen))
            run.counters["predictions"] += size
        if event.inputs and not warmup:
            run.instants += 1

    def _labels(self, run, source, event, *, train, rows=None):
        """Resolver etiquetas maduras contra la predicción emitida y preparar su admisión."""
        markets, moments, errors, admitted = [], [], [], []
        partition = source.phase.partition
        for flow, decision_at, value in event.labels:
            key = flow, decision_at
            entry = run.pending.pop(key, None)
            if entry is None:
                raise ValueError("El label no tiene una predicción emitida pendiente")
            markets.append(flow.split("/", 1)[0])
            moments.append(decision_at)
            errors.append(entry.issued - value)
            run.counters["labels"] += 1
            if self.audit is not None:
                self.audit.append(("label", partition, flow, decision_at, event.at))
            if run.bank is not None:
                admitted.append((decision_at, flow, entry, value))
            if rows is not None:
                rows.add(entry, flow, decision_at, value)
            graph = run.graphs.pop(key, None)
            if not train:
                continue
            if graph is None:
                run.counters["labels_without_graph"] += 1
                continue
            run.outstanding[entry.block] -= 1
            if not run.outstanding[entry.block]:
                del run.outstanding[entry.block]
            run.predictions.append(graph)
            run.targets.append(value)
            run.blocks.append(entry.block)
            run.used.append((flow, decision_at, event.at))
        for start in range(0, len(errors), 4096):
            run.errors.update(
                markets[start : start + 4096],
                moments[start : start + 4096],
                errors[start : start + 4096],
            )
        if admitted:
            run.staged = self._episodes(run.bank, admitted, event.at)

    def _episodes(self, bank, admitted, at):
        """Mismo orden canónico e IDs que el ejecutor: maduración, decisión y flujo."""
        admitted.sort(key=lambda item: (item[0], item[1]))
        first = bank.seen + 1
        return CandidateEpisodes(
            ids=torch.arange(first, first + len(admitted), dtype=torch.int64),
            keys=torch.stack([entry.key for _, _, entry, _ in admitted]),
            values=torch.stack([entry.value for _, _, entry, _ in admitted]),
            times=torch.tensor(
                [[decision, entry.available, at] for decision, _, entry, _ in admitted],
                dtype=torch.int64,
            ),
            labels=torch.tensor([value for *_, value in admitted], dtype=torch.float64),
        )

    def _admit(self, run, source, at):
        """Publicar en el banco las etiquetas del evento después de sus predicciones."""
        if run.staged is None:
            return
        episodes, run.staged = run.staged, None
        run.bank = run.bank.propose(episodes, confirmed_at=at)
        run.memory = None
        run.counters["admitted"] += len(episodes.ids)
        if self.audit is not None:
            self.audit.append(
                ("admit", source.phase.partition, at, tuple(episodes.ids.tolist()), run.bank.seen)
            )

    @staticmethod
    def _close(run):
        run.counters["unresolved"] = len(run.pending)
        run.pending.clear()
        run.graphs.clear()
        run.staged = run.memory = None

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

    def evaluate(self, source, *, stop=None, destination=None):
        """Recorrido con parámetros congelados, banco reiniciado y etiquetas maduras."""
        if all(source is not known for known in self.sources.values()):
            raise ValueError("La evaluación solo recorre las fases declaradas")
        self.model.eval()
        run = _Pass(bank=self._new_bank(source.phase.partition))
        dtype = np.float64 if self.dtype == torch.float64 else np.float32
        rows = None if destination is None else _PredictionRows(dtype)
        parameters = self.model.named_parameters().values()
        flags = [value.requires_grad for value in parameters]
        try:
            # La GRU de ATen en CPU no da los mismos bits si sus pesos requieren gradiente,
            # aunque no se registre el grafo. Así coincide con FrozenCandidateConsumer.
            for value in parameters:
                value.requires_grad_(False)
            with torch.no_grad():
                for event in source.batched_events(block_rows=self.recipe.block_rows):
                    if stop is not None and stop.requested:
                        raise _Pause
                    self._labels(run, source, event, train=False, rows=rows)
                    if event.inputs:
                        self._observe(run, source, event, train=False)
                    self._admit(run, source, event.at)
                    if event.close_phase:
                        self._close(run)
        finally:
            for value, flag in zip(parameters, flags, strict=True):
                value.requires_grad_(flag)
        if run.counters["labels"] == 0:
            raise ValueError("La fase no contiene etiquetas maduras")
        if rows is not None:
            rows.flush()
            atomic_parquet_batches(Path(destination), rows.tables)
        return self._metrics(run)


class CandidateChronologicalTrainer(CandidateChronologicalPredictor):
    """Ajustar la GRU candidata recorriendo instantes de decisión en orden.

    Solo cambian los parámetros mediante el optimizador. El banco es estado del recorrido,
    se reinicia en cada pasada y solo admite claves y valores del codec fijo con etiquetas
    maduras. K repite lectura y refinamiento sobre la misma instantánea y no escribe.
    """

    def __init__(
        self,
        adapter,
        recipe,
        *,
        train,
        validation,
        output,
        heldout=None,
        world="candidate_gru",
        fold="0",
        optimizer_factory=None,
        audit=False,
    ):
        heldout = {} if heldout is None else dict(heldout)
        if not set(heldout) <= set(HELDOUT):
            raise ValueError("Las fases deben ser las particiones declaradas del mismo corpus")
        sources = dict(train=train, validation=validation, **heldout)
        super().__init__(
            adapter, recipe, sources=sources, output=output, world=world, fold=fold, audit=audit
        )
        self.train, self.validation, self.heldout = train, validation, heldout
        model = self.model
        self.roles, self.inert = parameter_roles(model, recipe.admission)
        named = model.named_parameters()
        groups = [
            dict(params=[named[name] for name in names], role=role)
            for role, names in self.roles.items()
        ]
        self.trainable = [value for group in groups for value in group["params"]]
        if not all(value.requires_grad for value in self.trainable):
            raise ValueError("Los parámetros ajustables del candidato deben requerir gradiente")
        factory = optimizer_factory or (
            lambda values: torch.optim.AdamW(
                values, lr=recipe.learning_rate, weight_decay=recipe.weight_decay
            )
        )
        self.optimizer = factory(groups)
        listed = [id(p) for group in self.optimizer.param_groups for p in group["params"]]
        if len(listed) != len(set(listed)) or set(listed) != {id(p) for p in self.trainable}:
            raise ValueError("El optimizador debe cubrir exactamente los parámetros ajustables")
        self.identity = dict(
            schema_version=1,
            recipe=recipe.identity(),
            adapter=adapter.identity(),
            codec=None if self.codec is None else self.codec.identity(),
            output_head=dict(CONTRACT),
            dtype=str(self.dtype),
            device=self.device,
            parameter_roles=self.roles,
            inert_parameters=self.inert,
            optimizer=type(self.optimizer).__module__ + "." + type(self.optimizer).__qualname__,
            dataset_sha256=train.dataset.identity,
            sources={
                name: dict(index_sha256=s.identity, phase=asdict(s.phase))
                for name, s in sources.items()
            },
            bank_scope=dict(world=world, fold=fold),
            implementation=self._code(),
            numerics=_numerics(),
            final_test_opened=False,
        )
        self.run_id = hashlib.sha256(canonical(self.identity).encode()).hexdigest()
        self.global_step, self.selection, self.history, self.train_metrics = 0, None, [], None

    @staticmethod
    def _code():
        return {name: sha256(Path(importlib.import_module(name).__file__)) for name in _OWN_MODULES}

    def _check_runtime(self):
        if (
            self._code() != self.identity["implementation"]
            or _numerics() != self.identity["numerics"]
        ):
            raise ValueError("El código o la configuración numérica cambiaron durante el recorrido")

    def _loss(self, quantiles, target, *, reduction="mean"):
        if self.recipe.loss == PINBALL:
            if reduction == "mean":
                return pinball_loss(quantiles, target)
            return pinball_loss(quantiles, target, reduction="none").sum()
        return torch.nn.functional.l1_loss(median(quantiles), target, reduction=reduction)

    def _stacked(self, run, rows):
        quantiles = torch.stack([run.predictions[row] for row in rows])
        target = [run.targets[row] for row in rows]
        return quantiles, torch.tensor(target, dtype=quantiles.dtype, device=quantiles.device)

    def _accumulate(self, run, at):
        """Propagar las etiquetas que maduran en el evento, por grupos de bloques completos.

        Cada grupo suma la pérdida de sus filas y el paso del tramo divide el gradiente por el
        total de etiquetas. Equivale a ponderar cada grupo por sus filas sobre ese total. Las
        predicciones conservan su instantánea y sus parámetros, así que solo cambia el orden
        de las sumas. Un bloque nunca se reparte entre grupos, para no recorrer su grafo dos
        veces dentro del evento.
        """
        if not run.predictions:
            return
        blocks = {}
        for row, block in enumerate(run.blocks):
            blocks.setdefault(block, []).append(row)
        groups = [([], set())]
        for block, rows in blocks.items():
            if groups[-1][0] and len(groups[-1][0]) + len(rows) > self.recipe.accumulation_rows:
                groups.append(([], set()))
            groups[-1][0].extend(rows)
            groups[-1][1].add(block)
        for rows, members in groups:
            loss = self._loss(*self._stacked(run, rows), reduction="sum")
            # Un bloque con filas aún sin etiqueta conserva su grafo hasta que maduren o hasta
            # el paso del tramo. Sin recomputación retiene sus activaciones completas.
            loss.backward(retain_graph=not members.isdisjoint(run.outstanding))
            run.accumulated += len(rows)
            run.accumulated_loss = run.accumulated_loss + loss.detach()
        if self.audit is not None:
            self.audit.append(("accumulate", at, tuple(len(rows) for rows, _ in groups)))
        run.predictions.clear()
        run.targets.clear()
        run.blocks.clear()

    def _update(self, run, at):
        """Un paso con las etiquetas maduras del tramo. Después se descartan los grafos."""
        if self.recipe.accumulation_rows is None:
            labels = len(run.predictions)
            if labels:
                loss = self._loss(*self._stacked(run, range(labels)))
                if not torch.isfinite(loss).item():
                    raise ValueError("La pérdida del tramo no es finita")
                loss.backward()
                total = float(loss.detach()) * labels
        else:
            labels = run.accumulated
            if labels:
                # Una sola sincronización por tramo, como sin acumulación.
                total = float(run.accumulated_loss)
                if not math.isfinite(total):
                    raise ValueError("La pérdida del tramo no es finita")
                for value in self.trainable:
                    if value.grad is not None:
                        value.grad.div_(labels)
        if labels:
            torch.nn.utils.clip_grad_norm_(
                self.trainable, self.recipe.max_grad_norm or math.inf, error_if_nonfinite=True
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            run.counters["updates"] += 1
            run.counters["labels_in_loss"] += labels
            run.counters["loss_sum"] += total
            if self.audit is not None:
                self.audit.append(("update", at, self.global_step, tuple(run.used)))
        run.graphs.clear()
        run.outstanding.clear()
        run.predictions.clear()
        run.targets.clear()
        run.blocks.clear()
        run.used.clear()
        run.accumulated, run.accumulated_loss = 0, 0.0
        run.instants = 0
        run.counters["segments"] += 1

    def _train_pass(self, run, cursor, stop, save):
        source = self.train
        self.model.train()
        start, stage = cursor["event"], cursor["stage"]
        events = source.batched_events(start_cursor=start, block_rows=self.recipe.block_rows)
        since, last = 0, time.perf_counter()
        for index, event in enumerate(events, start):
            if not (index == start and stage == "inputs"):
                self._labels(run, source, event, train=True)
                if self.recipe.accumulation_rows is not None:
                    self._accumulate(run, event.at)
                decision = bool(event.inputs) and event.at >= source.phase.decision_start
                if (decision and run.instants == self.recipe.update_instants) or event.close_phase:
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
                            raise _Pause
            if event.inputs:
                self._observe(run, source, event, train=True)
            self._admit(run, source, event.at)
        raise ValueError("El recorrido de ajuste terminó sin su cierre declarado")

    def _export(self, run):
        """Barrera tras un paso: sin grafos vivos y con la admisión del evento sin aplicar."""
        if run.graphs or run.predictions or run.instants or run.accumulated:
            raise ValueError("La barrera solo admite tramos ya actualizados")
        keys = sorted(run.pending)
        entries = [run.pending[key] for key in keys]
        pending = dict(
            flows=[flow for flow, _ in keys],
            decision_at=torch.tensor([at for _, at in keys], dtype=torch.int64),
            issued=torch.tensor([e.issued for e in entries], dtype=torch.float64),
        )
        if run.bank is not None:
            pending.update(
                available_at=torch.tensor([e.available for e in entries], dtype=torch.int64),
                keys=torch.stack([e.key for e in entries])
                if entries
                else torch.empty((0, 128), dtype=self.dtype),
                values=torch.stack([e.value for e in entries])
                if entries
                else torch.empty((0, 256), dtype=self.dtype),
            )
        staged = run.staged
        return dict(
            pending=pending,
            bank=None if run.bank is None else run.bank.snapshot(),
            staged=None if staged is None else asdict(staged),
            errors=[[m, t, *values] for (m, t), values in sorted(run.errors.sessions.items())],
            counters=dict(run.counters),
        )

    def _restore(self, payload):
        bank = self._new_bank("train")
        run = _Pass(bank=None if bank is None else bank.restore(payload["bank"]))
        pending = payload["pending"]
        for row, (flow, at, issued) in enumerate(
            zip(
                pending["flows"],
                pending["decision_at"].tolist(),
                pending["issued"].tolist(),
                strict=True,
            )
        ):
            entry = _Pending(float(issued))
            if bank is not None:
                entry.available = int(pending["available_at"][row])
                entry.key = pending["keys"][row].clone()
                entry.value = pending["values"][row].clone()
            run.pending[flow, int(at)] = entry
        if payload["staged"] is not None:
            run.staged = CandidateEpisodes(**payload["staged"])
        run.errors.sessions = {(m, t): [c, a, s] for m, t, c, a, s in payload["errors"]}
        run.counters = dict(payload["counters"])
        return run

    def _predict(self, stop):
        """Predicciones fuera del ajuste del estado seleccionado, con el contrato común."""
        result = {}
        for name, source in (("validation", self.validation), *self.heldout.items()):
            path = self.output / f"{name}-predictions.parquet"
            metrics = self.evaluate(source, stop=stop, destination=path)
            result[name] = dict(
                path=path.name, sha256=sha256(path), bytes=path.stat().st_size, metrics=metrics
            )
        return result

    def run(self, *, resume=False, stop=None):
        """Recorrer épocas con presupuesto fijo, conservar el mejor estado y predecir."""
        require_learning_allowed("el entrenador cronológico de la GRU candidata")
        output, checkpoints = self.output, self.output / "checkpoints"
        if type(resume) is not bool or output.is_symlink() or output.exists() != resume:
            raise ValueError("Usa una ejecución nueva o solicita continuar una existente")
        stop = stop or StopRequest()
        report_path = output / "run.json"
        cursor, run = dict(epoch=0, phase="validation"), None
        if resume:
            report = _read_report(report_path)
            if report["identity"] != self.identity:
                raise ValueError("La identidad o configuración de la ejecución ha cambiado")
            if report["status"] == "completed":
                return report
            state = load_training_state(checkpoints, expected_identity=self.identity)
            _load_parameters(self.model, state["model"])
            if self.model.parameter_fingerprint() != state["parameters_sha256"]:
                raise ValueError("Los parámetros recuperados no conservan su huella")
            self.optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"], self.device)
            self.global_step, cursor = state["global_step"], state["cursor"]
            self.selection, self.history = state["selection"], state["history"]
            self.train_metrics = state["train_metrics"]
            if state["run"] is not None:
                run = self._restore(state["run"])
        else:
            output.mkdir(parents=True)
            report = dict(
                schema_version=1,
                kind="candidate_gru_chronological_run",
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
                model=_parameters_cpu(self.model),
                parameters_sha256=self.model.parameter_fingerprint(),
                optimizer=self.optimizer.state_dict(),
                rng=capture_rng(self.device),
                selection=self.selection,
                history=self.history,
                train_metrics=self.train_metrics,
                run=None if current is None else self._export(current),
            )
            path = save_training_state(checkpoints, state, identity=self.identity, best=best)
            reference = dict(path=str(path.relative_to(output)), sha256=sha256(path))
            report.update(recovery_checkpoint=reference, global_step=self.global_step)
            if best:
                report["best_checkpoint"] = dict(
                    reference, parameters_sha256=state["parameters_sha256"]
                )
            report.update(selection=self.selection, history=self.history, cursor=position)
            atomic_json(report_path, report)

        if not resume:
            save(cursor)
        started = time.perf_counter()
        try:
            while cursor["phase"] != "predictions":
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
                    finished = epoch >= self.recipe.epochs or self.selection["should_stop"]
                    cursor = (
                        dict(epoch=epoch, phase="predictions")
                        if finished
                        else dict(epoch=epoch, phase="train", event=0, stage="start")
                    )
                    save(cursor, best=self.selection["last_improved"])
                    if stop.requested and not finished:
                        raise _Pause
                    continue
                self.train_metrics = self._train_pass(
                    run or _Pass(bank=self._new_bank("train")), cursor, stop, save
                )
                run = None
                cursor = dict(epoch=epoch + 1, phase="validation")
                save(cursor)
            best = load_training_state(
                checkpoints,
                expected_identity=self.identity,
                selection="best",
                expected_sha256=report["best_checkpoint"]["sha256"],
            )
            _load_parameters(self.model, best["model"])
            if self.model.parameter_fingerprint() != best["parameters_sha256"]:
                raise ValueError("El estado seleccionado no conserva su huella")
            report["predictions"] = self._predict(stop)
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
        except _Pause:
            report["status"] = "paused"
            return report
        except BaseException as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            report["attempts"].append(dict(seconds=time.perf_counter() - started))
            atomic_json(report_path, report)


def restore_selected(output, adapter, *, carried=False):
    """Cargar el mejor estado de una ejecución completa en un adaptador del mismo contrato.

    El adaptador resultante puede congelarse para `FrozenCandidateConsumer`. Con
    `carried` el adaptador puede pertenecer a otra ventana de la misma edición, con la
    misma representación, configuración, codec, binario y código.
    """
    output = Path(output)
    report = _read_report(output / "run.json")
    identity = report.get("identity", {})
    window = not carried
    if report.get("status") != "completed" or _adapter_contract(
        adapter.identity(), window=window
    ) != _adapter_contract(identity.get("adapter", {}), window=window):
        raise ValueError("La ejecución no está completa o pertenece a otro adaptador")
    selected = report["best_checkpoint"]
    state = load_training_state(
        output / "checkpoints",
        expected_identity=identity,
        selection="best",
        expected_sha256=selected["sha256"],
    )
    _load_parameters(adapter.model, state["model"])
    if adapter.model.parameter_fingerprint() != selected["parameters_sha256"]:
        raise ValueError("El estado seleccionado no conserva su huella")
    return selected["parameters_sha256"]
