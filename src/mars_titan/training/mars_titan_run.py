"""Recorrido cronológico de MARS-TITAN: Titans-MAC congelado, banco episódico y lector ajustado.

El padre `mac_online` seleccionado no cambia. Sus parámetros están congelados y su memoria
rápida avanza con la regla asociativa de Titans en cada observación, igual que en el brazo
Titans-MAC. El optimizador ajusta solo `EpisodicReadout`, que refina el estado de trabajo de
MAC con episodios maduros y aplica la cabeza del padre. El estado de trabajo se descarta tras
cada predicción, así que no hay BPTT entre instantes.

En cada evento se resuelven primero las etiquetas maduras contra la predicción emitida,
después se actualiza si el tramo está completo y por último se predicen las entradas del
instante con la instantánea del banco confirmada antes de esas etiquetas. La admisión del
evento se publica al final, como en `FinancialSession`. Los IDs de episodio siguen el orden
canónico de decisión y flujo del ejecutor nativo. El banco se reinicia en cada recorrido.

Las predicciones del tramo no conservan su grafo. Al actualizar se repite el lector sobre cada
bloque emitido del tramo con su misma instantánea y estado de trabajo, se exige que la
repetición reproduzca exactamente lo emitido y se acumula el gradiente de la pérdida media.
"""

import hashlib
import importlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.session_metrics import SessionErrors
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.financial_consumers import (
    bank_retention,
    bank_snapshot,
    episodic_bank,
    mature_record,
)
from mars_titan.memory.financial_observations import FinancialObservationSource
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.memory.write_policy import MatureErrorConfig
from mars_titan.models.quantile_head import PINBALL, QUANTILE_HEAD, pinball_loss
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.episodic_readout import EpisodicReadout, apply_episodic_readout
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch, validated_cpu_batch
from mars_titan.models.titans.frozen_financial import _implementation, _numerics

from .checkpoints import (
    StopRequest,
    capture_rng,
    load_training_state,
    restore_rng,
    save_training_state,
)
from .financial_run import Paused, _compatible, _read_report, _split, _stack
from .learning_hold import require_learning_allowed
from .selection import (
    FIXED_BUDGET,
    VALIDATION_PLATEAU,
    advance_selection,
    initial_selection,
    validate_selection,
)

RECIPE = "mars_titan_episodic_readout_chronological_v1"
# Hiperparámetros del optimizador que puede variar un caso, los mismos que en Titans-MAC.
SEARCHED = ("learning_rate", "max_grad_norm")
CASE_NAME = re.compile(r"[a-z][a-z0-9_-]{0,31}")
ADMISSIONS = ("m0", "m1", "m2")
LOSSES = (PINBALL, "mae")
_OWN_MODULES = (
    "mars_titan.training.mars_titan_run",
    "mars_titan.training.financial_run",
    "mars_titan.memory.financial_observations",
    "mars_titan.memory.financial_consumers",
    "mars_titan.memory.retention_bank",
    "mars_titan.memory.write_policy",
    "mars_titan.memory.episodic_codec",
    "mars_titan.training.checkpoints",
    "mars_titan.training.selection",
    "mars_titan.evaluation.session_metrics",
)


def _default_selection():
    return dict(metric="session_mae", patience=5, min_delta=1e-05, stopping=FIXED_BUDGET)


@dataclass(frozen=True)
class ReadoutRecipe:
    """Hiperparámetros declarados antes de ejecutar, comunes a los brazos emparejados.

    La combinación de componentes (escritura, K y modo de episodios) pertenece a la variante.
    `update_instants` cuenta instantes de decisión por tramo y no es BPTT.
    """

    update_instants: int = 8
    loss: str = PINBALL
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    epochs: int = 30
    selection: dict = field(default_factory=_default_selection)
    block_rows: int = 128
    neighbors: int = 8
    temperature: float = 1.0
    bank_capacity: int = 1024
    bank_seed: int = 73
    max_working_bytes: int = 128 * 1024**2
    checkpoint_updates: int = 256
    checkpoint_seconds: float = 900.0

    def __post_init__(self):
        validate_selection(self.selection, epochs=self.epochs)
        integers = (
            (self.update_instants, 1, 256),
            (self.epochs, 1, 1000),
            (self.block_rows, 1, 256),
            (self.neighbors, 1, 8),
            (self.bank_capacity, 8, 1024),
            (self.bank_seed, 0, 2**64 - 1),
            (self.max_working_bytes, 1024, 128 * 1024**2),
            (self.checkpoint_updates, 1, 1_000_000),
        )
        numbers = (
            self.learning_rate,
            self.weight_decay,
            self.temperature,
            self.checkpoint_seconds,
        )
        clip = self.max_grad_norm
        if (
            self.loss not in LOSSES
            or any(type(v) is not int or not low <= v <= high for v, low, high in integers)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers)
            or not 0 < self.learning_rate <= 1
            or self.weight_decay < 0
            or not 1e-4 <= self.temperature <= 100
            or self.checkpoint_seconds <= 0
            or (
                clip is not None
                and (type(clip) not in (int, float) or not math.isfinite(clip) or clip <= 0)
            )
        ):
            raise ValueError(
                "La receta del lector necesita pérdida, tramo, banco, presupuesto, "
                "optimizador y checkpoints válidos"
            )

    def identity(self):
        return dict(
            schema_version=1,
            recipe=RECIPE,
            **asdict(self),
            optimizer="AdamW",
            trained_parameters="episodic_readout_only_parent_frozen",
            update_unit="decision_instants_per_segment_no_bptt",
            label_rule="loss_only_after_maturity_event_then_next_segment_update",
            loss_reduction="mean_over_matured_labels_of_segment",
            gradient="emitted_blocks_recomputed_at_update_with_bitwise_check",
            bank_rule="snapshot_before_event_labels_then_admit_mature_labels",
            bank_reset="each_pass_starts_empty",
            episode_ids="canonical_decision_then_flow",
            fast_state="parent_mac_online_reset_each_pass_then_warmup",
            refinements_counted_as_memory_updates=False,
            selection_metric_definition="SessionErrors.session_mae_on_issued_predictions",
        )


def load_recipe(path):
    """Leer la receta del lector con su esquema, estado y casos de búsqueda declarados.

    Cada caso de `walk_forward.search_cases` sustituye los mismos hiperparámetros del
    optimizador (`SEARCHED`), que entonces no aparecen en `recipe`, como en Titans-MAC.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise ValueError("La receta no es un archivo regular de hasta 64 KiB")
    document = json.loads(path.read_text(encoding="utf-8"))
    fields = {"schema_version", "recipe_name", "status", "recipe", "walk_forward", "pending"}
    if (
        not isinstance(document, dict)
        or set(document) != fields
        or document["schema_version"] != 1
        or document["recipe_name"] != RECIPE
        or not isinstance(document["recipe"], dict)
        or not isinstance(document["walk_forward"], dict)
        or set(document["walk_forward"]) != {"search_cases"}
    ):
        raise ValueError("La receta del lector no conserva su esquema")
    cases = document["walk_forward"]["search_cases"]
    valid = isinstance(cases, dict) and all(
        isinstance(case, dict) and case for case in cases.values()
    )
    keys = {frozenset(case) for case in cases.values()} if valid else set()
    if not (
        valid
        and 1 <= len(cases) <= 3
        and all(isinstance(name, str) and CASE_NAME.fullmatch(name) for name in cases)
        and len(keys) == 1
        and next(iter(keys)) <= set(SEARCHED)
        and not next(iter(keys)) & set(document["recipe"])
        and len({canonical(case) for case in cases.values()}) == len(cases)
    ):
        raise ValueError(
            "Los casos de búsqueda deben ser de uno a tres, distintos, con nombre válido y "
            "sustituir los mismos hiperparámetros del optimizador, ausentes de la receta base"
        )
    for case in cases.values():
        ReadoutRecipe(**(document["recipe"] | case))
    return document


def case_recipe(document, search_case):
    """Receta del lector para el caso de búsqueda elegido."""
    cases = document["walk_forward"]["search_cases"]
    if not isinstance(search_case, str) or search_case not in cases:
        raise ValueError("Elige uno de los casos de búsqueda que declara la receta del lector")
    return ReadoutRecipe(**(document["recipe"] | cases[search_case]))


def retention_config(recipe, admission, *, policy="reservoir", **options):
    """Configuración del banco de cada escritura. M0 no construye banco.

    `options` son los límites de la retención con centros fijos de M1, como la frontera.
    """
    if admission == "m0":
        return None
    return bank_retention(
        admission, capacity=recipe.bank_capacity, seed=recipe.bank_seed, policy=policy, **options
    )


def parameter_roles(readout, admission):
    """Separar parámetros del lector. Con M0 la lectura de episodios no recibe gradiente."""
    roles = dict(episodic_read=[], refiner=[])
    for name, _ in readout.named_parameters():
        if name.startswith(("query_projection.", "value_projection.")):
            roles["episodic_read"].append(name)
        else:
            roles["refiner"].append(name)
    inert = roles.pop("episodic_read") if admission == "m0" else []
    return roles, inert


def _parameters_digest(module):
    values = {
        name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in module.named_parameters()
    }
    return hashlib.sha256(canonical(values).encode()).hexdigest()


def _pack_bank(snapshot):
    """Sustituir los archivos nativos por tensores uint8 admitidos por la carga restringida."""

    def pack(value):
        return torch.frombuffer(bytearray(value), dtype=torch.uint8).clone()

    if "memory" in snapshot:
        return dict(snapshot, memory=pack(snapshot["memory"]))
    return dict(snapshot, indices={role: pack(raw) for role, raw in snapshot["indices"].items()})


def _unpack_bank(payload):
    def unpack(value):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError("El archivo del banco no conserva sus bytes")
        return bytes(value.numpy())

    if "memory" in payload:
        return dict(payload, memory=unpack(payload["memory"]))
    return dict(payload, indices={role: unpack(raw) for role, raw in payload["indices"].items()})


@dataclass
class _Entry:
    """Predicción emitida a la espera de su etiqueta. key y value solo existen con banco."""

    issued: float
    levels: list | None = None
    available: int | None = None
    key: np.ndarray | None = None
    value: np.ndarray | None = None
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
        mac_updates=0,
        refinements=0,
        updates=0,
        segments=0,
        loss_sum=0.0,
        unresolved=0,
    )


@dataclass
class _Pass:
    """Estado de un recorrido. Se persisten flujos, pendientes, banco, admisión y errores."""

    flows: dict = field(default_factory=dict)
    bank: object = None
    staged: list = field(default_factory=list)
    pending: dict = field(default_factory=dict)
    errors: SessionErrors = field(default_factory=SessionErrors)
    counters: dict = field(default_factory=_counters)
    # Bloques emitidos en el tramo vigente, con lo necesario para repetir el lector.
    blocks: dict = field(default_factory=dict)
    used: list = field(default_factory=list)
    instants: int = 0
    next_block: int = 0


class MarsTitanInference:
    """Recorrer fases con el padre congelado, el lector y el banco de la escritura declarada.

    Es la parte del recorrido que no ajusta nada. Valida, predice los tramos medidos y, en la
    variante B, predice una ventana posterior con el estado elegido en su ancla.
    """

    def __init__(
        self,
        predictor,
        readout,
        recipe,
        *,
        admission,
        retention,
        native,
        codec,
        world,
        fold,
        audit=False,
    ):
        if (
            type(predictor) is not FinancialPredictor
            or type(readout) is not EpisodicReadout
            or type(recipe) is not ReadoutRecipe
        ):
            raise ValueError("El recorrido necesita el padre, el lector y su receta")
        config = predictor.config
        control = predictor.local_control
        if config.variant != "mac_online" or (
            control is not None and control.config.mode != "disabled"
        ):
            # La B de CM-v1 llega con C en modo disabled. La penalización solo ajusta el núcleo.
            raise ValueError("El lector parte de Titans-MAC mac_online, con C solo en disabled")
        if predictor.training or any(p.requires_grad for p in predictor.parameters()):
            raise ValueError("El padre Titans-MAC debe llegar congelado, en eval y sin gradientes")
        if torch.backends.mha.get_fastpath_enabled() or torch.is_inference_mode_enabled():
            raise ValueError("El recorrido exige fastpath=False declarado y sin inference_mode")
        self.quantiles = config.head == QUANTILE_HEAD
        if self.quantiles != (recipe.loss == PINBALL):
            raise ValueError("La cabeza de cuantiles se ajusta solo y siempre con pinball")
        if admission not in ADMISSIONS:
            raise ValueError("M3 no tiene definición acreditada. Se admiten M0, M1 y M2")
        if (admission == "m0") != (readout.config.mode == "no_bank") or (
            (admission == "m0") != (retention is None)
            or (admission == "m1" and type(retention) is not RetentionConfig)
            or (admission == "m2" and type(retention) is not MatureErrorConfig)
        ):
            raise ValueError("M0 usa el control no_bank sin banco y M1/M2 su banco declarado")
        if (
            type(codec) is not FrozenEpisodeCodec
            or codec.identity()["input_specification"] != config.inputs.identity()
            or readout.config.codec_id != codec.fingerprint()
        ):
            raise ValueError("El codec no comparte la entrada del padre y del lector")
        weight = predictor.head.weight
        if (
            readout.config.hidden_size != config.hidden_size
            or readout.step_logit.dtype != weight.dtype
            or readout.step_logit.device != weight.device
            or readout.config.neighbors != recipe.neighbors
            or readout.config.temperature != recipe.temperature
            or readout.config.max_working_bytes != recipe.max_working_bytes
            or recipe.block_rows > min(config.max_batch, readout.config.max_batch)
        ):
            raise ValueError("El lector no corresponde al padre o a la receta declarada")
        self.device, self.dtype = weight.device, weight.dtype
        if str(self.device) not in {"cpu", "cuda:0"}:
            raise ValueError("El recorrido admite cpu o cuda:0 explícitos")
        if any(not isinstance(text, str) or not 0 < len(text) <= 128 for text in (world, fold)):
            raise ValueError("El ámbito del banco necesita mundo y fold explícitos")
        if admission != "m0" and not hasattr(native, "MemoryRecord"):
            raise ValueError("El banco necesita el enlace episódico nativo")
        self.predictor, self.readout, self.recipe = predictor, readout, recipe
        self.admission, self.retention, self.native, self.codec = (
            admission,
            retention,
            native,
            codec,
        )
        self.world, self.fold = world, fold
        self.audit = [] if audit else None

    # El contexto de una instantánea no cambia el cálculo. Solo la vincula a su evento.
    def _context(self, partition, at):
        return hashlib.sha256(
            canonical([self.world, self.fold, partition, at]).encode()
        ).hexdigest()

    def _new_bank(self, partition):
        if self.admission == "m0":
            return None
        return episodic_bank(
            self.native,
            self.retention,
            self.admission,
            codec_id=self.codec.fingerprint(),
            world=self.world,
            partition=partition,
            fold=self.fold,
        )

    def _observe(self, run, phase, event, *, train):
        """Predecir los bloques del instante con la instantánea previa a sus etiquetas."""
        predictor, specification = self.predictor, self.predictor.config.inputs
        warmup = event.at < phase.decision_start
        snapshot = context = None
        if not warmup:
            context = self._context(phase.partition, event.at)
            if run.bank is not None:
                snapshot = bank_snapshot(
                    run.bank,
                    codec_id=self.codec.fingerprint(),
                    context_id=context,
                    cutoff=event.at,
                    dtype=self.dtype,
                    device=self.device,
                )
        seen = 0 if run.bank is None else run.bank.seen
        for raw in event.inputs:
            cpu = validated_cpu_batch(raw, specification)
            batch = DecisionBatch.from_validated(cpu, device=self.device, dtype=self.dtype)
            new = tuple(flow for flow in batch.flow_ids if flow not in run.flows)
            if new:
                run.flows.update(_split(predictor.initial_state(new)))
            state = _stack([run.flows[flow] for flow in batch.flow_ids])
            with torch.no_grad():
                prepared = predictor.prepare(batch, state)
            run.flows.update(_split(prepared.next_state))
            size = len(batch.flow_ids)
            run.counters["observations"] += size
            run.counters["mac_updates"] += size
            if warmup:
                run.counters["warmup_observations"] += size
                continue
            # Solo se conserva lo que la repetición del lector necesita, sin el estado MAC.
            slim = replace(prepared, next_state=None, detached_tokens=None)
            if train and (
                self.readout.estimated_bytes(
                    size, 0 if snapshot is None else snapshot.count, differentiable=True
                )
                > self.readout.config.max_working_bytes
            ):
                raise ValueError("La repetición diferenciable supera el presupuesto del lector")
            with torch.no_grad():
                result = apply_episodic_readout(
                    slim,
                    predictor.head,
                    self.readout,
                    snapshot,
                    context_id=context,
                    cutoff=event.at,
                )
            issued = result.point_predictions.detach()
            values = issued.cpu().tolist()
            levels = (
                result.quantiles.detach().cpu().tolist()
                if self.quantiles and not train
                else [None] * size
            )
            encoded = None if run.bank is None else self.codec.encode(cpu)
            block = None
            if train:
                block, run.next_block = run.next_block, run.next_block + 1
                run.blocks[block] = dict(
                    prepared=slim,
                    snapshot=snapshot,
                    context=context,
                    cutoff=event.at,
                    issued=issued,
                    keys=tuple(zip(batch.flow_ids, batch.prediction_at, strict=True)),
                )
            for row, (flow, at) in enumerate(zip(batch.flow_ids, batch.prediction_at, strict=True)):
                if (flow, at) in run.pending:
                    raise ValueError("La decisión ya tiene una predicción pendiente")
                entry = _Entry(values[row], levels[row], block=block)
                if encoded is not None:
                    entry.available = batch.input_available_at[row]
                    entry.key = encoded.key_inputs[row].copy()
                    entry.value = encoded.values[row].copy()
                run.pending[flow, at] = entry
                if self.audit is not None:
                    self.audit.append(("prediction", phase.partition, flow, at, values[row], seen))
            run.counters["predictions"] += size
            run.counters["refinements"] += size * self.readout.config.refinements
        if event.inputs and not warmup:
            run.instants += 1

    def _labels(self, run, phase, event, *, train, rows=None):
        """Resolver etiquetas maduras contra lo emitido y preparar su admisión."""
        markets, moments, errors = [], [], []
        for flow, decision_at, value in event.labels:
            entry = run.pending.pop((flow, decision_at), None)
            if entry is None:
                raise ValueError("El label no tiene una predicción emitida pendiente")
            markets.append(flow.split("/", 1)[0])
            moments.append(decision_at)
            errors.append(entry.issued - value)
            run.counters["labels"] += 1
            if self.audit is not None:
                self.audit.append(("label", phase.partition, flow, decision_at, event.at))
            if run.bank is not None:
                run.staged.append((decision_at, flow, entry, value))
            if rows is not None:
                rows.append((flow, decision_at, entry.issued, value, entry.levels))
            if not train:
                continue
            if entry.block not in run.blocks:
                run.counters["labels_without_graph"] += 1
                continue
            run.used.append((flow, decision_at, entry.block, value))
        for start in range(0, len(errors), 4096):
            run.errors.update(
                markets[start : start + 4096],
                moments[start : start + 4096],
                errors[start : start + 4096],
            )

    def _admit(self, run, phase, at):
        """Publicar en el banco las etiquetas del evento después de sus predicciones."""
        if not run.staged:
            return
        staged, run.staged = sorted(run.staged, key=lambda item: item[:2]), []
        first = run.bank.seen + 1
        records = [
            mature_record(
                self.native,
                identifier=first + offset,
                decision_at=decision_at,
                available_at=entry.available,
                maturity_at=at,
                key=entry.key.tolist(),
                value=entry.value.tolist(),
                label=value,
            )
            for offset, (decision_at, _, entry, value) in enumerate(staged)
        ]
        options = dict(confirmed_at=at)
        if self.admission == "m2":
            # Error financiero maduro de la predicción emitida, como en `FinancialSession`.
            options["errors"] = {
                record.id: value - entry.issued
                for record, (_, _, entry, value) in zip(records, staged, strict=True)
            }
        run.bank = run.bank.propose(records, **options)
        run.counters["admitted"] += len(records)
        if self.audit is not None:
            ids = tuple(record.id for record in records)
            self.audit.append(("admit", phase.partition, at, ids, run.bank.seen))

    @staticmethod
    def _close(run):
        run.counters["unresolved"] = len(run.pending)
        run.flows.clear()
        run.pending.clear()
        run.blocks.clear()
        run.used.clear()
        run.staged = []

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

    def _pass(self, phase, events, *, stop=None, rows=None):
        """Recorrer una fase sin ajustar, con banco vacío y memoria rápida inicial."""
        self.readout.eval()
        run = _Pass(bank=self._new_bank(phase.partition))
        with torch.no_grad():
            for event in events:
                if stop is not None and stop.requested:
                    raise Paused
                self._labels(run, phase, event, train=False, rows=rows)
                if event.inputs:
                    self._observe(run, phase, event, train=False)
                self._admit(run, phase, event.at)
                if event.close_phase:
                    self._close(run)
        if run.counters["labels"] == 0:
            raise ValueError("La fase no contiene etiquetas maduras")
        return self._metrics(run)

    def evaluate(self, source, *, stop=None, rows=None):
        """Recorrido con parámetros congelados. Con `rows`, una fila por etiqueta resuelta."""
        if type(source) is not FinancialObservationSource or not _compatible(
            self.predictor.config.inputs, source.specification()
        ):
            raise ValueError("El tramo no pertenece a la entrada del padre")
        events = source.batched_events(block_rows=self.recipe.block_rows)
        return self._pass(source.phase, events, stop=stop, rows=rows)


class ReadoutTrainer(MarsTitanInference):
    """Ajustar el lector episódico sobre el padre congelado recorriendo instantes en orden."""

    def __init__(
        self,
        predictor,
        readout,
        recipe,
        *,
        admission,
        retention,
        native,
        codec,
        train,
        validation,
        output,
        world,
        fold,
        optimizer_factory=None,
        audit=False,
    ):
        super().__init__(
            predictor,
            readout,
            recipe,
            admission=admission,
            retention=retention,
            native=native,
            codec=codec,
            world=world,
            fold=fold,
            audit=audit,
        )
        if (
            type(train) is not FinancialObservationSource
            or type(validation) is not FinancialObservationSource
            or train.phase.partition != "train"
            or validation.phase.partition != "validation"
            or train.dataset.identity != validation.dataset.identity
            or validation.phase.decision_start < train.phase.decision_end
        ):
            raise ValueError("Se necesitan fases de ajuste y validación ordenadas del mismo corpus")
        if any(
            not _compatible(predictor.config.inputs, source.specification())
            for source in (train, validation)
        ):
            raise ValueError("Las vistas no conservan la entrada del padre")
        self.train, self.validation = train, validation
        self.output = Path(output)
        for protected in (*train.dataset.roots.values(), train.path.parent, validation.path.parent):
            outside_source(protected, self.output)
            outside_source(self.output, protected)
        self.roles, self.inert = parameter_roles(readout, admission)
        named = dict(readout.named_parameters())
        groups = [
            dict(params=[named[name] for name in names], role=role)
            for role, names in self.roles.items()
        ]
        self.trainable = [value for group in groups for value in group["params"]]
        if not all(value.requires_grad for value in self.trainable):
            raise ValueError("Los parámetros ajustables del lector deben requerir gradiente")
        factory = optimizer_factory or (
            lambda values: torch.optim.AdamW(
                values, lr=recipe.learning_rate, weight_decay=recipe.weight_decay
            )
        )
        self.optimizer = factory(groups)
        listed = [id(p) for group in self.optimizer.param_groups for p in group["params"]]
        if len(listed) != len(set(listed)) or set(listed) != {id(p) for p in self.trainable}:
            raise ValueError("El optimizador debe cubrir exactamente los parámetros del lector")
        self.identity = dict(
            schema_version=1,
            recipe=recipe.identity(),
            parent=dict(
                predictor=predictor.get_extra_state(),
                parameters_sha256=predictor._parameter_id,
                state="selected_checkpoint_frozen",
            ),
            readout=readout.get_extra_state(),
            initial_readout_sha256=_parameters_digest(readout),
            codec=codec.identity(),
            admission=admission,
            retention=None if retention is None else asdict(retention),
            bank_scope=dict(world=world, fold=fold),
            native_sha256=None if admission == "m0" else native.binary_sha256,
            parameter_roles=self.roles,
            inert_parameters=self.inert,
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
        # La identidad se compara con su copia JSON del informe, así que se normaliza aquí.
        self.identity = json.loads(canonical(self.identity))
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
        self.predictor.verify_parameter_identity()
        if self.predictor._parameter_id != self.identity["parent"]["parameters_sha256"]:
            raise ValueError("El padre congelado cambió durante el ajuste del lector")

    def _loss(self, result, target):
        """Pinball de los cinco niveles con `quantile_head_v1` o L1 de la salida escalar."""
        if self.recipe.loss == PINBALL:
            return pinball_loss(result.quantiles, target)
        return torch.nn.functional.l1_loss(result.point_predictions, target)

    def _update(self, run, at):
        """Repetir cada bloque del tramo con etiquetas y acumular la pérdida media del tramo."""
        total = len(run.used)
        if total:
            matured = {}
            for flow, decision_at, block, value in run.used:
                matured.setdefault(block, {})[flow, decision_at] = value
            loss_sum = 0.0
            for block in sorted(matured):
                record = run.blocks[block]
                result = apply_episodic_readout(
                    record["prepared"],
                    self.predictor.head,
                    self.readout,
                    record["snapshot"],
                    context_id=record["context"],
                    cutoff=record["cutoff"],
                    differentiable=True,
                )
                if not torch.equal(result.point_predictions.detach(), record["issued"]):
                    raise ValueError("La repetición del bloque no reproduce su predicción emitida")
                labels = matured[block]
                positions = [i for i, key in enumerate(record["keys"]) if key in labels]
                index = torch.tensor(positions, dtype=torch.int64, device=self.device)
                target = torch.tensor(
                    [labels[record["keys"][i]] for i in positions],
                    dtype=self.dtype,
                    device=self.device,
                )
                selected = replace(
                    result,
                    point_predictions=result.point_predictions.index_select(0, index),
                    quantiles=None
                    if result.quantiles is None
                    else result.quantiles.index_select(0, index),
                )
                loss = self._loss(selected, target) * (len(positions) / total)
                if not torch.isfinite(loss).item():
                    raise ValueError("La pérdida del tramo no es finita")
                loss.backward()
                loss_sum += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(
                self.trainable, self.recipe.max_grad_norm or math.inf, error_if_nonfinite=True
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            run.counters["updates"] += 1
            run.counters["labels_in_loss"] += total
            run.counters["loss_sum"] += loss_sum * total
            if self.audit is not None:
                used = tuple((flow, decision_at) for flow, decision_at, _, _ in run.used)
                self.audit.append(("update", at, self.global_step, used))
        run.blocks.clear()
        run.used.clear()
        run.instants = 0
        run.counters["segments"] += 1

    def _train_pass(self, run, cursor, stop, save):
        source, phase = self.train, self.train.phase
        self.readout.train()
        start, stage = cursor["event"], cursor["stage"]
        events = source.batched_events(start_cursor=start, block_rows=self.recipe.block_rows)
        since, last = 0, time.perf_counter()
        for index, event in enumerate(events, start):
            if not (index == start and stage == "inputs"):
                self._labels(run, phase, event, train=True)
                decision = bool(event.inputs) and event.at >= phase.decision_start
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
                            raise Paused
            if event.inputs:
                self._observe(run, phase, event, train=True)
            self._admit(run, phase, event.at)
        raise ValueError("El recorrido de ajuste terminó sin su cierre declarado")

    def _export(self, run):
        """Barrera tras un paso: sin bloques vivos y con la admisión del evento sin aplicar."""
        if run.blocks or run.used or run.instants:
            raise ValueError("La barrera solo admite tramos ya actualizados")
        predictor, size = self.predictor, self.predictor.config.max_batch
        flows = sorted(run.flows)
        fast = [
            predictor.export_state_cpu(_stack([run.flows[f] for f in flows[i : i + size]]))
            for i in range(0, len(flows), size)
        ]
        keys = sorted(run.pending)
        entries = [run.pending[key] for key in keys]
        pending = dict(
            flows=[flow for flow, _ in keys],
            decision_at=torch.tensor([at for _, at in keys], dtype=torch.int64),
            issued=torch.tensor([e.issued for e in entries], dtype=torch.float64),
        )
        staged = sorted(run.staged, key=lambda item: item[:2])
        if run.bank is not None:
            pending.update(self._features(entries))
        return dict(
            fast=fast,
            pending=pending,
            bank=None if run.bank is None else _pack_bank(run.bank.snapshot()),
            staged=dict(
                flows=[flow for _, flow, _, _ in staged],
                decision_at=torch.tensor([at for at, *_ in staged], dtype=torch.int64),
                issued=torch.tensor([e.issued for _, _, e, _ in staged], dtype=torch.float64),
                labels=torch.tensor([v for *_, v in staged], dtype=torch.float64),
                **(self._features([e for _, _, e, _ in staged]) if run.bank is not None else {}),
            ),
            errors=[[m, t, *values] for (m, t), values in sorted(run.errors.sessions.items())],
            counters=dict(run.counters),
            next_block=run.next_block,
        )

    @staticmethod
    def _features(entries):
        width = (len(entries), 64)
        return dict(
            available_at=torch.tensor([e.available for e in entries], dtype=torch.int64),
            keys=torch.from_numpy(np.stack([e.key for e in entries]))
            if entries
            else torch.empty(width, dtype=torch.float32),
            values=torch.from_numpy(np.stack([e.value for e in entries]))
            if entries
            else torch.empty(width, dtype=torch.float32),
        )

    @staticmethod
    def _entries(payload, banked):
        entries = []
        for row, issued in enumerate(payload["issued"].tolist()):
            entry = _Entry(float(issued))
            if banked:
                entry.available = int(payload["available_at"][row])
                entry.key = payload["keys"][row].numpy().copy()
                entry.value = payload["values"][row].numpy().copy()
            entries.append(entry)
        return entries

    def _restore(self, payload):
        run = _Pass(bank=self._new_bank("train"))
        if (payload["bank"] is None) != (run.bank is None):
            raise ValueError("El estado del banco no corresponde a la escritura declarada")
        if run.bank is not None:
            run.bank = run.bank.restore(_unpack_bank(payload["bank"]))
        for item in payload["fast"]:
            run.flows.update(_split(self.predictor.restore_state(item, device=self.device)))
        banked = run.bank is not None
        pending = payload["pending"]
        for flow, at, entry in zip(
            pending["flows"],
            pending["decision_at"].tolist(),
            self._entries(pending, banked),
            strict=True,
        ):
            run.pending[flow, int(at)] = entry
        staged = payload["staged"]
        run.staged = [
            (int(at), flow, entry, float(value))
            for flow, at, entry, value in zip(
                staged["flows"],
                staged["decision_at"].tolist(),
                self._entries(staged, banked),
                staged["labels"].tolist(),
                strict=True,
            )
        ]
        run.errors.sessions = {(m, t): [c, a, s] for m, t, c, a, s in payload["errors"]}
        run.counters = dict(payload["counters"])
        run.next_block = payload["next_block"]
        return run

    def _readout_state(self):
        return {
            key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for key, value in self.readout.state_dict().items()
        }

    def _load_best(self, report):
        state = load_training_state(
            self.output / "checkpoints",
            expected_identity=self.identity,
            selection="best",
            expected_sha256=report["best_checkpoint"]["sha256"],
        )
        self.readout.load_state_dict(state["model"])
        if _parameters_digest(self.readout) != state["readout_sha256"]:
            raise ValueError("El lector seleccionado no conserva su huella")
        return state

    def run(self, *, resume=False, stop=None):
        """Recorrer épocas con la regla del protocolo y dejar cargado el mejor lector."""
        require_learning_allowed("el ajuste del lector episódico de MARS-TITAN")
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
            state = load_training_state(checkpoints, expected_identity=self.identity)
            self.readout.load_state_dict(state["model"])
            if _parameters_digest(self.readout) != state["readout_sha256"]:
                raise ValueError("Los parámetros recuperados del lector no conservan su huella")
            self.optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"], str(self.device))
            self.global_step, cursor = state["global_step"], state["cursor"]
            self.selection, self.history = state["selection"], state["history"]
            self.train_metrics = state["train_metrics"]
            if state["run"] is not None:
                run = self._restore(state["run"])
            if report["status"] == "completed":
                self._load_best(report)
                return report
        else:
            output.mkdir(parents=True)
            report = dict(
                schema_version=1,
                kind="mars_titan_readout_chronological_run",
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
                model=self._readout_state(),
                readout_sha256=_parameters_digest(self.readout),
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
                report["best_checkpoint"] = dict(reference, readout_sha256=state["readout_sha256"])
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
                self.train_metrics = self._train_pass(
                    run or _Pass(bank=self._new_bank("train")), cursor, stop, save
                )
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
