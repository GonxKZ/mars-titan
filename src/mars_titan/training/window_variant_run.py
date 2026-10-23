"""Ajuste experimental de ventanas cortas, independiente de las campañas canónicas."""

import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from mars_titan.budget_training import seed_run, validate_loss
from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.embeddings import require_cuda
from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES
from mars_titan.data.price_windows import gate_price_window
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.session_metrics import SessionErrors
from mars_titan.models.baselines.multimodal import fuse_modalities, validate_architecture
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import FinancialInputSpec

from .checkpoints import (
    StopRequest,
    capture_rng,
    load_training_state,
    restore_rng,
    save_training_state,
)
from .checkpoints import (
    _index as _checkpoint_index,
)
from .checkpoints import (
    _intact as _checkpoint_intact,
)
from .cohort_contract import representation_identity
from .corpus_inputs import CorpusDataset
from .kernel_policy import FP32_STRICT, declared_policy, require_policy
from .learning_hold import require_learning_allowed
from .reference_run import (
    _SOURCES as _REFERENCE_SOURCES,
)
from .reference_run import (
    _evaluate,
    _forward,
    _metrics,
    _Pause,
    _session_table,
    _statistics,
    _update,
    row_loss,
)
from .selection import FIXED_BUDGET, advance_selection, initial_selection, validate_selection

KIND = "independent_window_variant_v1"
_DATES = ("train_start", "train_end", "validation_start", "validation_end")
_DUST_FIELDS = {
    "population",
    "chunk_size",
    "sigma",
    "seed",
    "max_forward_calls",
    "max_workspace_bytes",
}
_SOURCE_FILES = (
    "training/window_variant_run.py",
    "training/reference_run.py",
    "training/checkpoints.py",
    "training/selection.py",
    "training/kernel_policy.py",
    "training/corpus_inputs.py",
    "training/cohort_contract.py",
    "training/temporal_corpus.py",
    "training/temporal_contract.py",
    "training/edition_subsets.py",
    "training/edition_catalog.py",
    "evaluation/splits.py",
    "evaluation/session_metrics.py",
    "models/titans/financial.py",
    "models/titans/financial_inputs.py",
    "models/titans/config.py",
    "models/titans/state.py",
    "models/baselines/multimodal.py",
    "models/baselines/transformer.py",
    "data/input_policy.py",
    "data/price_windows.py",
    "data/temporal.py",
    "data/storage.py",
    "budget_training.py",
)


def _selection():
    return dict(metric="session_mae", stopping=FIXED_BUDGET, patience=3, min_delta=0.0)


def _integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require_precision(policy):
    require_policy(FP32_STRICT, policy)
    if any(torch.is_autocast_enabled(device) for device in ("cpu", "cuda")):
        raise ValueError("La precisión fp32_strict no admite autocast en CPU ni CUDA")


@dataclass(frozen=True)
class WindowVariantRecipe:
    """Receta cerrada y previa al ajuste, con intervalos efectivos y presupuesto fijo."""

    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    temporal_encoder: str | None = None
    estimator: str = "backprop"
    optimizer: str = "adamw"
    seed: int = 42
    epochs: int = 2
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    loss: str = "mae"
    huber_delta: float = 0.01
    max_grad_norm: float | None = 1.0
    hidden_size: int = 64
    layers: int = 1
    max_seconds: float = 240.0
    max_output_bytes: int = 512 * 1024**2
    max_cuda_bytes: int = 6 * 1024**3
    cache_bytes: int = 64 * 1024**2
    checkpoint_steps: int = 64
    selection: dict = field(default_factory=_selection)
    dust: dict | None = None
    final_test_access: bool = False

    def __post_init__(self):
        moments = []
        for name in _DATES:
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError("Las fechas de la ventana deben ser fechas ISO explícitas")
            moment = date.fromisoformat(value)
            if moment.isoformat() != value or moment.day != 1:
                raise ValueError("Los límites de la ventana deben ser comienzos de mes")
            moments.append(moment)
        start, train_end, validation_start, end = moments
        months = (end.year - validation_start.year) * 12 + end.month - validation_start.month
        if not (
            date(2000, 1, 1) <= start < train_end == validation_start < end <= date(2024, 1, 1)
            and (end - start).days <= 366
            and 1 <= months <= 3
        ):
            raise ValueError(
                "La ventana efectiva necesita hasta 366 días en total y reserva de 2024"
            )
        if (
            self.temporal_encoder not in (None, "kronlstm_v1", "vector_delta_v1")
            or self.estimator not in ("backprop", "dust")
            or self.optimizer not in ("adamw", "sgd")
            or type(self.final_test_access) is not bool
            or self.final_test_access
            or not _integer(self.seed, 0, 2**32 - 1)
            or not _integer(self.epochs, 1, 1000)
            or not _integer(self.batch_size, 1, 256)
            or not _integer(self.checkpoint_steps, 1, 1_000_000)
            or not _integer(self.max_output_bytes, 1, 1024**3)
            or not _integer(self.max_cuda_bytes, 1, 6 * 1024**3)
            or not _integer(self.cache_bytes, 0, 4096 * 1024**2)
            or not _positive(self.learning_rate)
            or self.learning_rate > 1
            or type(self.weight_decay) not in (int, float)
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
            or not _positive(self.max_seconds)
            or self.max_seconds > 14400
            or (self.max_grad_norm is not None and not _positive(self.max_grad_norm))
        ):
            raise ValueError("La receta de ventana no conserva sus variantes y presupuestos")
        validate_architecture(self.hidden_size, self.layers, 0.0)
        validate_loss(self.loss, self.huber_delta)
        validate_selection(self.selection, epochs=self.epochs)
        if self.selection.get("stopping") != FIXED_BUDGET:
            raise ValueError(
                "Los controles de ventana requieren presupuesto fijo de actualizaciones"
            )
        if self.estimator == "dust":
            if self.temporal_encoder is not None or not isinstance(self.dust, dict):
                raise ValueError(
                    "Dust requiere TransformerDirect escalar y configuración explícita"
                )
            if set(self.dust) != _DUST_FIELDS:
                raise ValueError("La configuración Dust no tiene exactamente sus campos")
            options = self.dust
            if (
                not _integer(options["population"], 2, 4096)
                or not _integer(options["chunk_size"], 1, options["population"])
                or not _positive(options["sigma"])
                or not _integer(options["seed"], 0, 2**32 - 1)
                or not _integer(options["max_forward_calls"], 1, 1_000_000)
                or not _integer(options["max_workspace_bytes"], 1, 1024**3)
            ):
                raise ValueError("La población, el ruido y los límites Dust no son válidos")
        elif self.dust is not None:
            raise ValueError("Backpropagation no admite una configuración Dust oculta")
        object.__setattr__(self, "selection", deepcopy(self.selection))
        object.__setattr__(self, "dust", deepcopy(self.dust))

    def identity(self):
        return asdict(self)


def _recipe(value):
    if type(value) is not WindowVariantRecipe:
        raise ValueError("El runner necesita WindowVariantRecipe explícita")
    return WindowVariantRecipe(**asdict(value))


def prepare_window_variant_view(parent, catalog, output, recipe):
    """Componer subconjunto anual y vista existente sin alterar la edición histórica."""
    from .edition_subsets import materialize
    from .temporal_corpus import prepare_temporal_corpus

    recipe = _recipe(recipe)
    start, end = date.fromisoformat(recipe.train_start), date.fromisoformat(recipe.validation_end)
    if start.month != 1:
        raise ValueError("El subconjunto anual requiere inicio de entrenamiento en enero")
    parent, catalog, output = map(Path, (parent, catalog, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError("La vista experimental ya tiene una salida")
    for source in (parent, catalog):
        outside_source(source, output)
    validation = date.fromisoformat(recipe.validation_start)
    validation_months = (end.year - validation.year) * 12 + end.month - validation.month
    definition = dict(
        schema_version=1,
        kind="edition_subset_definition",
        name=f"window-{start.year}",
        markets=["US"],
        years=dict(first=start.year, last=(end - timedelta(days=1)).year),
        labels="usable",
    )
    protocol = dict(
        schema_version=2,
        market="US",
        train_start="2000-01-01",
        first_validation_start=recipe.validation_start,
        final_test_start="2024-01-01",
        final_test_end="2025-01-01",
        minimum_train_months=3,
        validation_months=validation_months,
        calibration_months=1,
        evaluation_months=1,
        step_months=120,
        primary_metric="session_mae",
        seeds=[recipe.seed],
        purge="label_interval",
        selection={**recipe.selection, "max_epochs": recipe.epochs},
    )
    from mars_titan.evaluation.splits import build_folds

    if len(build_folds(protocol)) != 1:
        raise ValueError("El protocolo experimental debe declarar una única ventana")
    output.mkdir(parents=True)
    atomic_json(output / "subset-definition.json", definition)
    atomic_json(output / "protocol.json", protocol)
    materialize(catalog, output / "subset-definition.json", output / "subset")
    prepared = prepare_temporal_corpus(
        parent,
        output / "protocol.json",
        None,
        None,
        output / "views",
        recover_annual_boundaries=True,
        input_policy=HISTORICAL_MASKED,
        subset=output / "subset/manifest.json",
    )
    if len(prepared["folds"]) != 1:
        raise ValueError("La preparación experimental debe producir una única ventana")
    manifest = output / "views/fold-000/manifest.json"
    atomic_json(
        output / "preparation.json",
        dict(
            status="prepared",
            manifest=str(manifest.resolve()),
            manifest_sha256=sha256(manifest),
            recipe=recipe.identity(),
            final_test_opened=False,
        ),
    )
    return manifest


class _WindowModel(nn.Module):
    """Ruta limpia del predictor completo con un estado nuevo en cada ventana."""

    def __init__(self, predictor, policy):
        super().__init__()
        self.predictor = predictor
        self.policy = policy
        self.forward_work = dict(clean_forward_calls=0, clean_row_evaluations=0)

    def forward(self, inputs, presence=None):
        _require_precision(self.policy)
        model = self.predictor
        self.forward_work["clean_forward_calls"] += 1
        self.forward_work["clean_row_evaluations"] += len(inputs["prices"])
        model._check_parameters()
        price, checks = model.price_encoder.encode_last(gate_price_window(inputs["prices"]))
        for value, description in checks:
            if not torch.isfinite(value).all():
                raise ValueError(description)
        token = fuse_modalities(price, inputs, model.encoders, model.fusion, presence)
        emitted = model.head(token).squeeze(-1)
        if not torch.isfinite(emitted).all():
            raise ValueError("La salida de la ventana contiene valores no finitos")
        return emitted


class _Budget:
    def __init__(self, began, seconds, stop):
        self.began, self.seconds, self.stop = began, seconds, stop

    @property
    def requested(self):
        return self.stop.requested or self.expired

    @property
    def expired(self):
        return time.perf_counter() - self.began >= self.seconds


def _sources(recipe):
    root = Path(__file__).resolve().parents[1]
    names = set(_SOURCE_FILES) | set(_REFERENCE_SOURCES)
    if recipe.temporal_encoder is not None:
        names.update(
            "models/kronlstm/" + name for name in ("__init__.py", "rule.py", "price_encoder.py")
        )
    if recipe.estimator == "dust":
        names.add("models/titans/frozen_financial.py")
        modules = sorted((root / "training/dust").glob("*.py"))
        if not modules or len(modules) > 32:
            raise ValueError("Falta un paquete Dust acotado e identificado")
        names.update(str(path.relative_to(root)) for path in modules)
    result = {name: sha256(root / name) for name in sorted(names)}
    result.update(
        {
            "project/" + name: sha256(root.parents[1] / name)
            for name in ("uv.lock", "pyproject.toml")
        }
    )
    return result


def _row_digest(digest, batch):
    for sample in batch["sample_ids"]:
        digest.update(sample.encode() + b"\n")


def _population(dataset, recipe, budget):
    """Admitir todas las filas reales antes de construir o ajustar el predictor."""
    if dataset.temporal is None or dataset.manifest.get("final_test_opened") is not False:
        raise ValueError("La ventana necesita una vista temporal con la reserva final cerrada")
    if not dataset.manifest.get("data_subset"):
        raise ValueError("La ventana histórica necesita un subconjunto declarado")
    if set(dataset.temporals) != {"US"}:
        raise ValueError("La primera variante independiente requiere un único calendario US")
    records, first = {}, None
    for partition in ("train", "validation"):
        start, end = (
            (recipe.train_start, recipe.train_end)
            if partition == "train"
            else (recipe.validation_start, recipe.validation_end)
        )
        low, high = np.datetime64(start, "us"), np.datetime64(end, "us")
        count, ids, inputs, targets = 0, hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        for batch in dataset.batches(
            partition=partition, batch_size=recipe.batch_size, epoch=0, seed=recipe.seed
        ):
            if budget.requested:
                raise _Pause()
            if np.any(batch["prediction_at"] < low) or np.any(batch["prediction_at"] >= high):
                raise ValueError("Una fila real no pertenece al intervalo efectivo de la receta")
            if first is None:
                first = batch
            count += len(batch["target"])
            _row_digest(ids, batch)
            for name in MODALITIES:
                values = batch["inputs"][name]
                inputs.update(
                    name.encode() + _canonical(list(values.shape)).encode() + values.tobytes()
                )
            inputs.update(batch["presence"].tobytes())
            targets.update(batch["target"].tobytes())
        if count <= 0 or count != dataset.manifest["counts"][partition]:
            raise ValueError("La ventana efectiva no reconcilia todas las filas declaradas")
        records[partition] = dict(
            samples=count,
            sample_ids_sha256=ids.hexdigest(),
            inputs_sha256=inputs.hexdigest(),
            targets_sha256=targets.hexdigest(),
        )
    return first, records


def _output_bytes(output, maximum):
    total, count, folders = 0, 0, [output]
    while folders:
        for item in folders.pop().iterdir():
            count += 1
            if count > 256 or item.is_symlink():
                raise ValueError("La salida supera el número de archivos o contiene un enlace")
            if item.is_dir():
                folders.append(item)
            else:
                total += item.stat().st_size
            if total > maximum:
                raise ValueError("La salida excede el presupuesto de bytes de la receta")
    return total


@contextmanager
def _locked_output(output, resume):
    if output.is_symlink():
        raise ValueError("La salida de ajuste no admite enlaces simbólicos")
    if resume:
        if not output.is_dir():
            raise ValueError("Falta la salida confirmada que se quiere recuperar")
    else:
        output.mkdir(parents=True, exist_ok=False)
    descriptor = os.open(output / ".run.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def _optimizer(predictor, recipe):
    decay, protected = [], []
    for parameter in predictor.parameters():
        (protected if getattr(parameter, "_no_weight_decay", False) else decay).append(parameter)
    groups = [dict(params=decay, weight_decay=recipe.weight_decay)]
    if protected:
        groups.append(dict(params=protected, weight_decay=0.0))
    constructor = torch.optim.AdamW if recipe.optimizer == "adamw" else torch.optim.SGD
    return constructor(groups, lr=recipe.learning_rate)


def _state_bytes(predictor, recipe, population):
    parameter_bytes = sum(value.numel() * value.element_size() for value in predictor.parameters())
    multiplier = 12 if recipe.optimizer == "adamw" else 4
    prediction_bytes = 512 * sum(record["samples"] for record in population.values())
    estimate = multiplier * parameter_bytes + 3 * 1024**2 + prediction_bytes
    if estimate > recipe.max_output_bytes:
        raise ValueError("El presupuesto de salida no admite recuperación y predicciones completas")


def _read_record(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024**2:
        raise ValueError("Falta un recibo regular dentro de su presupuesto")
    return json.loads(path.read_text())


def _has_runtime(output):
    return (output / "config.json").exists() or (output / "checkpoints/latest.json").exists()


def _session_artifact(predictions, destination, *, expected_samples, expected_sha256, budget):
    """Resumir las filas seleccionadas ya emitidas, sin repetir el predictor ni el lector."""
    if sha256(predictions) != expected_sha256:
        raise ValueError("Han cambiado las predicciones antes de resumir las sesiones")
    errors, zero = SessionErrors(max_sessions=366), SessionErrors(max_sessions=366)
    count = 0
    with pq.ParquetFile(predictions) as stream:
        for batch in stream.iter_batches(
            batch_size=4096, columns=["market", "prediction_at", "target", "prediction"]
        ):
            if budget.requested:
                raise _Pause()
            markets = np.asarray(batch.column(0).to_pylist())
            times = batch.column(1).cast(pa.int64()).to_numpy(zero_copy_only=False)
            targets = batch.column(2).to_numpy(zero_copy_only=False).astype(np.float64)
            values = batch.column(3).to_numpy(zero_copy_only=False).astype(np.float64)
            errors.update(markets, times, values - targets)
            zero.update(markets, times, -targets)
            count += batch.num_rows
    if count != expected_samples or sha256(predictions) != expected_sha256:
        raise ValueError("El resumen por sesión no concilia las predicciones confirmadas")
    if budget.requested:
        raise _Pause()
    atomic_parquet_batches(destination, [_session_table(errors, zero)])
    return dict(path=destination.name, sha256=sha256(destination))


def _verify_completed_artifacts(output, report):
    for category in ("predictions", "session_metrics"):
        references = report.get(category)
        if not isinstance(references, dict) or set(references) != {"train", "validation"}:
            raise ValueError("Faltan las predicciones o las métricas por sesión confirmadas")
        for reference in references.values():
            if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
                raise ValueError("Un artefacto confirmado no conserva su referencia")
            path = output / reference["path"]
            if (
                path.parent != output
                or path.is_symlink()
                or not path.is_file()
                or sha256(path) != reference["sha256"]
            ):
                raise ValueError("Ha cambiado una predicción o métrica por sesión confirmada")


def _predictor(dataset, first, recipe, device):
    dimensions = {name: int(first["inputs"][name].shape[-1]) for name in MODALITIES}
    specification = FinancialInputSpec(
        source_sha256=dataset.temporal.contract["parent_sha256"],
        view_sha256=dataset.identity,
        representation=representation_identity(
            dataset.manifest["representation"], input_policy=HISTORICAL_MASKED
        ),
        dimensions=dimensions,
        input_policy=HISTORICAL_MASKED,
    )
    configuration = FinancialConfig(
        specification,
        hidden_size=recipe.hidden_size,
        layers=recipe.layers,
        seed=recipe.seed,
        max_batch=max(256, recipe.batch_size),
        temporal_encoder=recipe.temporal_encoder,
    )
    return FinancialPredictor(configuration, device=str(device), dtype=torch.float32)


def _dust(predictor, recipe):
    if recipe.estimator != "dust":
        return None
    from .dust import DustConfig, DustEstimator, FinancialDustAdapter

    return DustEstimator(
        FinancialDustAdapter(predictor),
        DustConfig(**recipe.dust),
        loss=recipe.loss,
        huber_delta=recipe.huber_delta,
    )


def _update_window(model, estimator, optimizer, batch, recipe, device):
    predictor = model.predictor
    model.train()
    optimizer.zero_grad(set_to_none=True)
    target_source = torch.from_numpy(batch["target"]).to(device)
    target = target_source.to(dtype=predictor.head.weight.dtype)
    if estimator is None:
        emitted = _forward(model, batch, device)
        loss = row_loss(recipe.identity(), emitted, emitted, target, False).mean()
        if not torch.isfinite(loss):
            raise ValueError("La pérdida de la ventana no es finita")
        loss.backward()
    else:
        inputs = {
            name: torch.from_numpy(value).to(device) for name, value in batch["inputs"].items()
        }
        presence = torch.from_numpy(batch["presence"]).to(device)
        estimate = estimator.estimate(inputs, presence, target)
        parameters = dict(predictor.named_parameters())
        if set(estimate.gradients) != set(parameters):
            raise ValueError("Las señales Dust no cubren todos los parámetros del predictor")
        for name, parameter in parameters.items():
            gradient = estimate.gradients[name]
            if (
                gradient.shape != parameter.shape
                or gradient.device != parameter.device
                or gradient.dtype != parameter.dtype
            ):
                raise ValueError("La señal Dust no conserva forma, precisión y dispositivo")
            parameter.grad = gradient.detach()
        loss = estimate.loss
        with torch.no_grad():
            emitted = _forward(model, batch, device)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("La pérdida del estimador debe ser un escalar finito")
    parameters = list(predictor.parameters())
    if any(
        parameter.grad is None or not torch.isfinite(parameter.grad).all()
        for parameter in parameters
    ):
        raise ValueError("Los gradientes del predictor están ausentes o no son finitos")
    if recipe.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(parameters, recipe.max_grad_norm, error_if_nonfinite=True)
    _require_precision(model.policy)
    optimizer.step()
    _require_precision(model.policy)
    if any(not torch.isfinite(parameter).all() for parameter in parameters):
        raise ValueError("La actualización produjo parámetros no finitos")
    predictor._seal_parameters()
    return emitted.detach(), target_source.detach(), float(loss.detach())


def run_window_variant(manifest, output, recipe, resume=False, stop=None):
    """Ajustar únicamente la vista corta admitida y guardar cada barrera recuperable."""
    require_learning_allowed("el ajuste experimental de ventanas independientes")
    began = time.perf_counter()
    recipe = _recipe(recipe)
    policy = declared_policy(FP32_STRICT)
    _require_precision(policy)
    if type(resume) is not bool or (stop is not None and not isinstance(stop, StopRequest)):
        raise ValueError("La recuperación y la parada necesitan sus tipos explícitos")
    stop = StopRequest() if stop is None else stop
    budget = _Budget(began, recipe.max_seconds, stop)
    manifest, output = Path(manifest), Path(output)
    dataset = CorpusDataset(
        manifest, cache_bytes=recipe.cache_bytes, input_policy=HISTORICAL_MASKED
    )
    for source in (manifest, *dataset.roots.values()):
        outside_source(source, output)
    sources = _sources(recipe)
    admission = dict(
        manifest_sha256=dataset.identity,
        recipe=recipe.identity(),
        sources=sources,
        precision=FP32_STRICT,
        kernel_policy=policy,
    )
    previous = (
        _read_record(output / "report.json") if resume and (output / "report.json").exists() else {}
    )
    admission_resume = previous.get("stage") == "admission" and not _has_runtime(output)
    if admission_resume and previous.get("identity") != admission:
        raise ValueError("La admisión pendiente no conserva la vista, las fuentes y la receta")
    try:
        first, population = _population(dataset, recipe, budget)
    except _Pause:
        with _locked_output(output, resume):
            previous = (
                _read_record(output / "report.json") if (output / "report.json").exists() else {}
            )
            runtime = resume and _has_runtime(output)
            checkpoint, step = None, 0
            if runtime:
                directory = output / "checkpoints"
                confirmed = next(
                    (
                        row
                        for row in _checkpoint_index(directory)["latest"]
                        if _checkpoint_intact(directory, row)
                    ),
                    None,
                )
                step = confirmed["global_step"] if confirmed is not None else None
                checkpoint = (
                    dict(path="checkpoints/" + confirmed["name"], sha256=confirmed["sha256"])
                    if confirmed is not None
                    else None
                )
            attempts = [] if runtime else previous.get("attempts", [])
            attempts.append(
                dict(
                    status="partial" if budget.expired else "paused",
                    elapsed_seconds=time.perf_counter() - began,
                )
            )
            report = dict(
                status="partial" if budget.expired else "paused",
                stage="admission",
                stop_reason="timeout" if budget.expired else "requested",
                identity=admission,
                global_step=step,
                checkpoint=checkpoint,
                final_test_opened=False,
                attempts=attempts,
            )
            # La readmisión no sustituye el estado confirmado ni un resultado completado.
            destination = "admission-attempt.json" if runtime else "report.json"
            _output_bytes(output, recipe.max_output_bytes)
            atomic_json(output / destination, report)
            _output_bytes(output, recipe.max_output_bytes)
            return report
    device = require_cuda(max_bytes=recipe.max_cuda_bytes)
    if str(device) != "cuda:0":
        raise RuntimeError("La ventana experimental requiere cuda:0 explícita")
    seed_run(recipe.seed)
    torch.cuda.reset_peak_memory_stats(0)
    predictor = _predictor(dataset, first, recipe, device)
    model = _WindowModel(predictor, policy)
    optimizer, estimator = _optimizer(predictor, recipe), _dust(predictor, recipe)
    _state_bytes(predictor, recipe, population)
    identity = dict(
        schema_version=1,
        kind=KIND,
        experimental=True,
        final_test_access=False,
        manifest_sha256=dataset.identity,
        recipe=recipe.identity(),
        sources=sources,
        configuration=predictor.config.identity(),
        population=population,
        initial_parameters_sha256=predictor._parameter_id,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=str(device),
        gpu=torch.cuda.get_device_name(0),
        dtype="torch.float32",
        precision=FP32_STRICT,
        kernel_policy=policy,
        numpy=np.__version__,
        pyarrow=importlib.metadata.version("pyarrow"),
        exchange_calendars=importlib.metadata.version("exchange-calendars"),
        numerics=dict(
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cpu_threads=torch.get_num_threads(),
        ),
        estimator=estimator.identity() if estimator is not None else None,
    )
    sealed = _canonical(identity)
    config = dict(identity=identity, recipe=recipe.identity())
    checkpoint_directory = output / "checkpoints"
    state = dict(
        global_step=0, epoch=0, cursor=None, selection=None, history=[], statistics=_statistics()
    )
    attempts = []
    with _locked_output(output, resume):
        if resume and (output / "report.json").exists():
            previous = _read_record(output / "report.json")
            admission_resume = previous.get("stage") == "admission" and not _has_runtime(output)
            if admission_resume and previous.get("identity") != admission:
                raise ValueError("La admisión pendiente cambió antes de recuperar")
        if resume and not admission_resume:
            if _read_record(output / "config.json") != config:
                raise ValueError(
                    "La configuración, las fuentes o la vista cambiaron antes de recuperar"
                )
            attempts = previous.get("attempts", [])
            state = load_training_state(checkpoint_directory, expected_identity=identity)
            predictor.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"], str(device))
            model.forward_work = deepcopy(state["forward_work"])
            if estimator is not None:
                estimator.load_state_dict(state["estimator"])
            if previous.get("status") == "completed":
                load_training_state(
                    checkpoint_directory, expected_identity=identity, selection="best"
                )
                _require_precision(policy)
                _verify_completed_artifacts(output, previous)
                return previous
        else:
            attempts = previous.get("attempts", [])
            atomic_json(output / "config.json", config)

        def save(*, best=False):
            _require_precision(policy)
            if _sources(recipe) != sources or _canonical(identity) != sealed:
                raise ValueError(
                    "Las fuentes o la configuración sellada cambiaron durante el ajuste"
                )
            _output_bytes(output, recipe.max_output_bytes)
            state.update(
                model=predictor.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=capture_rng(str(device)),
                estimator=estimator.state_dict() if estimator is not None else None,
                forward_work=deepcopy(model.forward_work),
            )
            path = save_training_state(checkpoint_directory, state, identity=identity, best=best)
            _output_bytes(output, recipe.max_output_bytes)
            return dict(path=str(path.relative_to(output)), sha256=sha256(path))

        reference = save()
        report = dict(
            status="running",
            identity=identity,
            device=str(device),
            final_test_opened=False,
            samples={partition: record["samples"] for partition, record in population.items()},
            parameters=sum(parameter.numel() for parameter in predictor.parameters()),
            evaluation_scope=dict(
                train="in_sample", validation="used_for_selection", final_test="closed"
            ),
            global_step=state["global_step"],
            epochs=state["history"],
            selection=state["selection"],
            checkpoint=reference,
            attempts=attempts,
            predictions={},
            session_metrics={},
        )
        try:
            if state["selection"] is None:
                initial = _evaluate(
                    model, dataset, recipe.batch_size, device=str(device), stop=budget
                )
                state["selection"] = initial_selection(initial["session_mae"], recipe.selection)
                state["initial_validation"] = initial
                reference = save(best=True)
            while state["epoch"] < recipe.epochs:
                statistics = state["statistics"]
                for batch in dataset.batches(
                    partition="train",
                    batch_size=recipe.batch_size,
                    epoch=state["epoch"],
                    seed=recipe.seed,
                    cursor=state["cursor"],
                ):
                    if budget.requested:
                        raise _Pause()
                    started = time.perf_counter()
                    emitted, target, _ = _update_window(
                        model, estimator, optimizer, batch, recipe, device
                    )
                    _update(statistics, emitted, target)
                    statistics["elapsed_seconds"] += time.perf_counter() - started
                    state["cursor"], state["global_step"] = (
                        batch["confirmed_cursor"],
                        state["global_step"] + 1,
                    )
                    if state["global_step"] % recipe.checkpoint_steps == 0 or budget.requested:
                        reference = save()
                    if budget.requested:
                        raise _Pause()
                if statistics["samples"] != population["train"]["samples"]:
                    raise ValueError("La época no ha consumido toda la vista de entrenamiento")
                validation = _evaluate(
                    model, dataset, recipe.batch_size, device=str(device), stop=budget
                )
                state["epoch"] += 1
                state["selection"] = advance_selection(
                    state["selection"], validation["session_mae"], state["epoch"], recipe.selection
                )
                state["history"].append(
                    dict(epoch=state["epoch"], train=_metrics(statistics), validation=validation)
                )
                state["cursor"], state["statistics"] = None, _statistics()
                reference = save(best=state["selection"]["last_improved"])
            best = load_training_state(
                checkpoint_directory, expected_identity=identity, selection="best"
            )
            predictor.load_state_dict(best["model"])
            metrics = {}
            for partition in ("train", "validation"):
                destination = output / f"{partition}-predictions.parquet"
                metrics[partition] = _evaluate(
                    model,
                    dataset,
                    recipe.batch_size,
                    device=str(device),
                    partition=partition,
                    destination=destination,
                    stop=budget,
                )
                report["predictions"][partition] = dict(
                    path=destination.name, sha256=sha256(destination)
                )
            for partition, prediction in report["predictions"].items():
                report["session_metrics"][partition] = _session_artifact(
                    output / prediction["path"],
                    output / f"{partition}-sessions.parquet",
                    expected_samples=population[partition]["samples"],
                    expected_sha256=prediction["sha256"],
                    budget=budget,
                )
            report.update(status="completed", metrics=metrics)
        except _Pause:
            if state["epoch"] < recipe.epochs:
                reference = save()
            report.update(
                status="partial" if budget.expired else "paused",
                stop_reason="timeout" if budget.expired else "requested",
            )
        attempts.append(
            dict(
                elapsed_seconds=time.perf_counter() - began,
                status=report["status"],
                peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(0),
                peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(0),
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            )
        )
        report.update(
            global_step=state["global_step"],
            epochs=state["history"],
            selection=state["selection"],
            checkpoint=reference,
            initial_validation=state.get("initial_validation"),
            estimator=estimator.state_dict()["totals"] if estimator is not None else None,
            work={**model.forward_work, "optimizer_updates": state["global_step"]},
        )
        _output_bytes(output, recipe.max_output_bytes)
        _require_precision(policy)
        atomic_json(output / "report.json", report)
        _output_bytes(output, recipe.max_output_bytes)
        return report
