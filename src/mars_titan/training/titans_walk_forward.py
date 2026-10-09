"""Una ventana walk-forward de Titans-MAC: fases, ajuste, mejor estado y predicciones por fila.

Contrato por ventana para el orquestador de la campaña con máscaras:

- Entradas: la vista de la ventana (supervisión con su contrato temporal v2 y la edición
  desde 2000 con máscaras), el protocolo v2 que la generó, el identificador de ventana,
  la receta cronológica con su sección `walk_forward`, la variante y la semilla.
- Salida: un directorio con `run.json` y, al completarse, un Parquet por fila para
  validación, calibración y evaluación con las columnas de `training.reference_run`
  (y los cinco cuantiles si la receta usa `quantile_head_v1`). `run.json` declara
  `final_test_opened=false` y `predictions[tramo] = {path, sha256, rows, bytes, metrics}`.
- Reanudación: la misma llamada continúa desde el último checkpoint coherente del ajuste
  o desde el primer tramo sin confirmar. Una ventana completada con la misma petición se
  devuelve después de comprobar sus huellas, sin abrir fuentes ni repetir cálculos.

Política de memoria, común a las cuatro variantes: cada recorrido parte del estado rápido
inicial y de una cola vacía. El ajuste empieza en el inicio de su tramo. Validación,
calibración y evaluación observan antes las entradas de los `warmup_months` previos a su
tramo, sin etiquetas ni predicciones emitidas, y avanzan después en orden. Solo
`mac_online` escribe en la memoria, con la pérdida asociativa de Titans. Las etiquetas se
comparan con la predicción emitida cuando maduran y nunca modifican el estado.
"""

import hashlib
import importlib
import math
import os
import time
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.splits import build_folds, stopping_rule
from mars_titan.memory.financial_observations import (
    FinancialObservationSource,
    prepare_observation_index,
)
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.financial import (
    VARIANTS,
    FinancialConfig,
    FinancialPredictor,
    copy_paired_parameters,
)
from mars_titan.models.titans.financial_inputs import FINAL_TEST_US

from .checkpoints import StopRequest
from .corpus_inputs import CorpusDataset
from .financial_run import ChronologicalTrainer, Paused, load_recipe
from .learning_hold import require_learning_allowed
from .temporal_contract import temporal_contracts

KIND = "titans_walk_forward_window"
PREDICTED = ("validation", "calibration", "evaluation")
MEMORY_POLICY = "reset_each_pass_then_input_warmup_v1"
DTYPES = {"float32": torch.float32, "float64": torch.float64}
_ROWS_PER_TABLE = 65_536
_CODE = (
    "mars_titan.training.titans_walk_forward",
    "mars_titan.training.financial_run",
    "mars_titan.memory.financial_observations",
    "mars_titan.models.titans.financial",
    "mars_titan.models.titans.mac",
    "mars_titan.models.titans.neural_memory",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def memory_policy(warmup_months):
    """Declarar la política de estado de las cuatro variantes en todos los recorridos."""
    return dict(
        name=MEMORY_POLICY,
        reset="initial_fast_state_and_empty_queue_at_each_pass",
        train="from_fold_train_start_without_warmup",
        warmup="inputs_in_the_months_before_the_measured_partition_without_labels",
        warmup_months=warmup_months,
        advance="chronological_associative_update_written_only_by_mac_online",
        labels="compared_with_the_issued_prediction_after_maturity_never_written",
        mature_financial_errors="not_admitted_bank_disabled",
    )


def walk_forward_options(document):
    """Leer la sección `walk_forward` de la receta. Hoy solo declara el calentamiento."""
    options = document.get("walk_forward")
    _require(
        isinstance(options, dict)
        and set(options) == {"warmup_months"}
        and type(options["warmup_months"]) is int
        and 0 <= options["warmup_months"] <= 60,
        "La receta necesita walk_forward.warmup_months entero entre 0 y 60",
    )
    return options


def _micros(day):
    return int(np.datetime64(day, "us").astype(np.int64))


def _months_before(day, months):
    start = date.fromisoformat(day)
    position = start.year * 12 + start.month - 1 - months
    return date(position // 12, position % 12 + 1, 1).isoformat()


def window_phases(fold, warmup_months):
    """Fases del ajuste y de los tres tramos medidos, con el calentamiento acotado."""
    origin, train_end = (_micros(day) for day in fold["train"])
    phases = {"train": FinancialPhase("train", origin, origin, train_end, train_end)}
    for name in PREDICTED:
        start, end = (_micros(day) for day in fold[name])
        warmup = max(origin, _micros(_months_before(fold[name][0], warmup_months)))
        phases[name] = FinancialPhase(name, warmup, start, end, end)
    return phases


def _protocol(path, window, seed):
    protocol, digest = read_manifest(Path(path), 1024**2)
    rule = stopping_rule(protocol)
    folds = {fold["id"]: fold for fold in build_folds(protocol)}
    _require(window in folds, "La ventana no pertenece al protocolo")
    _require(seed in protocol["seeds"], "La semilla no está declarada en el protocolo")
    return protocol, digest, rule, folds[window]


def _recipe(path, rule):
    recipe, document = load_recipe(path)
    selection = {key: value for key, value in rule.items() if key != "max_epochs"}
    _require(
        recipe.epochs == rule["max_epochs"] and recipe.selection == selection,
        "La receta no aplica la regla de selección y parada del protocolo",
    )
    return recipe, document, walk_forward_options(document)


def _check_view(dataset, protocol, fold):
    """Exigir la ventana y el protocolo declarados en cada mercado de la vista."""
    contracts = temporal_contracts(dataset.manifest, input_policy=HISTORICAL_MASKED)
    common = {key: value for key, value in protocol.items() if key != "market"}
    _require(
        contracts
        and protocol["market"] in contracts
        and contracts[protocol["market"]]["protocol"] == protocol
        and all(
            {key: value for key, value in contract["protocol"].items() if key != "market"} == common
            and contract["fold"] == fold
            for contract in contracts.values()
        ),
        "La vista no conserva el protocolo y la ventana declarados",
    )
    counts = dataset.manifest["counts"]
    _require(
        dataset.manifest.get("final_test_opened") is False
        and all(type(counts.get(name)) is int and counts[name] > 0 for name in counts)
        and set(counts) == {"train", *PREDICTED},
        "La vista necesita filas en sus cuatro tramos y la reserva final cerrada",
    )


def _sources(dataset, phases, root):
    """Preparar o reutilizar un índice por fase. El nombre incluye vista y fase."""
    sources = {}
    for name, phase in phases.items():
        key = canonical(dict(view=dataset.identity, phase=asdict(phase)))
        folder = root / f"{name}-{hashlib.sha256(key.encode()).hexdigest()[:16]}"
        manifest = prepare_observation_index(dataset, folder, phase=phase, resume=folder.exists())
        sources[name] = FinancialObservationSource(dataset, manifest)
    return sources


def _predictor(document, specification, variant, seed, device):
    """Construir la variante con los parámetros iniciales copiados del emparejamiento."""
    options = {key: value for key, value in document["predictor"].items() if key != "dtype"}
    dtype = DTYPES[document["predictor"]["dtype"]]

    def build(name):
        config = FinancialConfig(specification, variant=name, seed=seed, **options)
        return FinancialPredictor(config, device=device, dtype=dtype)

    target = build(variant)
    if variant == document["pairing_source"]:
        return target, None
    return target, copy_paired_parameters(build(document["pairing_source"]), target)


def _code():
    return {name: sha256(Path(importlib.import_module(name).__file__)) for name in _CODE}


def _request(view, protocol_sha, window, recipe, variant, seed, device):
    """Petición verificable sin abrir la vista: huellas de sus archivos y del código."""
    return dict(
        view_sha256=sha256(Path(view)),
        protocol_sha256=protocol_sha,
        window=window,
        recipe_sha256=sha256(Path(recipe)),
        variant=variant,
        seed=seed,
        device=device,
        code=_code(),
    )


class PredictionRows:
    """Acumular filas resueltas y convertirlas por bloques al esquema común."""

    def __init__(self, quantiles):
        self.quantiles, self.buffer, self.tables, self.count = quantiles, [], [], 0

    def append(self, record):
        flow, at, _, _, levels = record
        if not 0 <= at < FINAL_TEST_US or (levels is not None) != self.quantiles:
            raise ValueError("La fila queda fuera de la edición o no conserva su salida")
        self.buffer.append(record)
        if len(self.buffer) >= _ROWS_PER_TABLE:
            self._flush()

    def _flush(self):
        if not self.buffer:
            return
        flows, moments, predictions, targets, levels = zip(*self.buffer, strict=True)
        columns = dict(
            sample_id=[f"{flow}/{at}" for flow, at in zip(flows, moments, strict=True)],
            asset_id=list(flows),
            market=[flow.split("/", 1)[0] for flow in flows],
            prediction_at=pa.array(moments, type=pa.timestamp("us", tz="UTC")),
            target=pa.array(targets, type=pa.float64()),
            prediction=pa.array(predictions, type=pa.float64()),
            zero=pa.array(np.zeros(len(flows)), type=pa.float64()),
        )
        if self.quantiles:
            matrix = np.asarray(levels, dtype=np.float64)
            columns.update(zip(QUANTILE_COLUMNS, matrix.T, strict=True))
        self.tables.append(pa.table(columns))
        self.count += len(flows)
        self.buffer.clear()

    def finish(self):
        self._flush()
        return self.tables


def _verify(output, report):
    for record in report["predictions"].values():
        path = output / record["path"]
        safe_destination(path)
        _require(sha256(path) == record["sha256"], "Han cambiado las predicciones confirmadas")


def run_titans_window(
    view,
    protocol,
    window,
    recipe,
    *,
    variant,
    seed,
    output,
    device="cuda:0",
    indices=None,
    stop=None,
    optimizer_factory=None,
):
    """Ajustar una variante en una ventana y escribir sus predicciones por fila.

    La protección del aprendizaje se comprueba antes de leer ninguna fuente.
    `optimizer_factory` solo existe para comprobar el bucle con un optimizador que no
    modifica pesos.
    """
    require_learning_allowed("run_titans_window de Titans-MAC")
    _require(variant in VARIANTS, "La variante no pertenece a los controles de Titans-MAC")
    _require(type(seed) is int and 0 <= seed < 2**32, "La semilla no es válida")
    _require(device in ("cpu", "cuda:0"), "El dispositivo debe ser cpu o cuda:0 explícitos")
    view, output = Path(view), Path(output)
    protocol_document, protocol_sha, rule, fold = _protocol(protocol, window, seed)
    chronological, document, options = _recipe(recipe, rule)
    request = _request(view, protocol_sha, window, recipe, variant, seed, device)
    safe_destination(output)
    report_path = output / "run.json"
    if report_path.exists():
        report, _ = read_manifest(report_path, 16 * 1024**2)
        _require(report.get("request") == request, "La petición de la ventana ha cambiado")
        if report["status"] == "completed":
            _verify(output, report)
            return report
    else:
        _require(
            not output.exists() or not any(output.iterdir()),
            "La salida contiene datos de otra ejecución",
        )
    if device == "cuda:0":
        from mars_titan.data.embeddings import require_cuda

        require_cuda()
        _require(
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"},
            "Configura CUBLAS_WORKSPACE_CONFIG antes de iniciar PyTorch",
        )
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    _check_view(dataset, protocol_document, fold)
    for protected in (*dataset.roots.values(), view.parent):
        outside_source(protected, output)
    phases = window_phases(fold, options["warmup_months"])
    sources = _sources(dataset, phases, Path(indices) if indices else output / "indices")
    specification = sources["train"].specification()
    predictor, pairing = _predictor(document, specification, variant, seed, device)
    trainer = ChronologicalTrainer(
        predictor,
        chronological,
        train=sources["train"],
        validation=sources["validation"],
        output=output / "fit",
        optimizer_factory=optimizer_factory,
        pairing=pairing,
    )
    identity = dict(
        schema_version=1,
        kind=KIND,
        request=request,
        window=fold,
        stopping_rule=rule,
        memory_policy=memory_policy(options["warmup_months"]),
        phases={name: asdict(phase) for name, phase in phases.items()},
        indices={name: source.identity for name, source in sources.items()},
        fit_run_id=trainer.run_id,
        expected_rows={name: dataset.manifest["counts"][name] for name in PREDICTED},
        final_test_opened=False,
    )
    run_id = hashlib.sha256(canonical(identity).encode()).hexdigest()
    if report_path.exists():
        _require(report["identity"] == identity, "La identidad de la ventana ha cambiado")
        _verify(output, report)
    else:
        output.mkdir(parents=True, exist_ok=True)
        report = dict(
            schema_version=1,
            kind=KIND,
            run_id=run_id,
            request=request,
            identity=identity,
            status="running",
            started_at_utc=datetime.now(UTC).isoformat(),
            final_test_opened=False,
            predictions={},
            attempts=[],
        )
        atomic_json(report_path, report)
    stop = stop or StopRequest()
    started = time.perf_counter()
    try:
        fit = trainer.run(resume=(output / "fit").exists(), stop=stop)
        if fit["status"] != "completed":
            report["status"] = "paused"
            return report
        report["fit"] = dict(
            path="fit/run.json",
            sha256=sha256(output / "fit/run.json"),
            run_id=trainer.run_id,
            best_epoch=fit["best_epoch"],
            best_score=fit["best_score"],
            best_checkpoint=dict(
                fit["best_checkpoint"], path="fit/" + fit["best_checkpoint"]["path"]
            ),
            plateau_epoch=fit["plateau_epoch"],
            global_step=trainer.global_step,
        )
        for name in PREDICTED:
            if name in report["predictions"]:
                continue
            if stop.requested:
                raise Paused
            rows = PredictionRows(trainer.quantiles)
            metrics = trainer.predict_partition(sources[name], rows, stop=stop)
            path = output / f"{name}-predictions.parquet"
            written = atomic_parquet_batches(path, rows.finish())
            _require(
                written == rows.count == metrics["labels"] == identity["expected_rows"][name],
                f"Las predicciones de {name} no concilian con la población de la vista",
            )
            if name == "validation":
                best, recomputed = fit["best_score"], metrics["session_mae"]
                report["validation_selection_check"] = dict(
                    best_score=best,
                    recomputed=recomputed,
                    absolute_difference=abs(best - recomputed),
                )
                _require(
                    math.isclose(best, recomputed, rel_tol=1e-6, abs_tol=1e-12),
                    "La validación del mejor estado no reproduce la puntuación seleccionada",
                )
            report["predictions"][name] = dict(
                path=path.name,
                sha256=sha256(path),
                rows=written,
                bytes=path.stat().st_size,
                metrics=metrics,
            )
            atomic_json(report_path, report)
        report.update(status="completed", finished_at_utc=datetime.now(UTC).isoformat())
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
