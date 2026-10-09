"""Una ventana walk-forward de Titans-MAC: fases, ajuste, mejor estado y predicciones por fila.

Contrato por ventana para el orquestador de la campaña con máscaras:

- Entradas: la vista de la ventana (supervisión con su contrato temporal v2 y la edición
  desde 2000 con máscaras), el protocolo v2 que la generó, el identificador de ventana,
  la receta cronológica con su sección `walk_forward`, la variante, la semilla y, si la
  receta declara casos de búsqueda, el caso elegido.
- Salida: un directorio con `run.json` y, al completarse, un Parquet por fila para
  validación, calibración y evaluación con las columnas de `training.reference_run`
  (y los cinco cuantiles si la receta usa `quantile_head_v1`). `run.json` declara
  `final_test_opened=false` y `predictions[tramo] = {path, sha256, rows, bytes, metrics}`.
- Reanudación: la misma llamada continúa desde el último checkpoint coherente del ajuste
  o desde el primer tramo sin confirmar. Una ventana completada con la misma petición se
  devuelve después de comprobar sus huellas, sin abrir fuentes ni repetir cálculos.

Campaña con máscaras: `titans_fit` y `titans_carry` son los ejecutores que registra
`training.masked_campaign` para los ajustes y las predicciones trasladadas de la variante
B. `carry_titans` aplica el estado elegido en la ventana ancla sin ajustar nada. Los
dos ejecutores declaran fastpath=False durante su trabajo, como la orden de una
ventana, y restauran después el estado del proceso.

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
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
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
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.financial import (
    VARIANTS,
    FinancialConfig,
    FinancialPredictor,
    copy_paired_parameters,
)
from mars_titan.models.titans.financial_inputs import FINAL_TEST_US
from mars_titan.models.titans.local_control import MACProjectionConfig

from .candidate_walk_forward import check_view_rows
from .checkpoints import StopRequest
from .corpus_inputs import CorpusDataset
from .financial_run import ChronologicalRecipe, ChronologicalTrainer, Paused, load_recipe
from .learning_hold import require_learning_allowed
from .search_cases import case_options, checked_search_cases
from .selection import AWAIT, campaign_rule, with_rule
from .temporal_contract import temporal_contracts
from .walk_forward_phases import PREDICTED, checked_warmup, window_phases

KIND = "titans_walk_forward_window"
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
    """Leer la sección `walk_forward`: calentamiento y casos de búsqueda opcionales.

    Cada caso de `search_cases` sustituye los mismos hiperparámetros de `SEARCHED`, que
    entonces no aparecen en `recipe`. Así ningún valor base queda sin usar o sin elegir.
    """
    options = document.get("walk_forward")
    _require(
        isinstance(options, dict)
        and {"warmup_months"} <= set(options) <= {"warmup_months", "search_cases"},
        "La receta necesita walk_forward.warmup_months entero entre 0 y 60",
    )
    checked_warmup(options["warmup_months"])
    if "search_cases" in options:
        checked_search_cases(options["search_cases"], document["recipe"], ChronologicalRecipe)
    return options


def case_recipe(document, search_case):
    """Receta cronológica del caso elegido. Sin casos declarados, la de la receta."""
    cases = walk_forward_options(document).get("search_cases")
    return ChronologicalRecipe(**case_options(document["recipe"], cases, search_case))


def view_protocol(view):
    """Protocolo v2 que declara el contrato temporal de la vista, sin abrir sus datos.

    En una vista conjunta todos los mercados comparten regla y ventana. Se toma el del
    primer mercado en orden alfabético para que la petición no dependa del orden.
    """
    manifest, _ = read_manifest(Path(view), 64 * 1024**2)
    contracts = temporal_contracts(manifest, input_policy=HISTORICAL_MASKED)
    _require(contracts, "La vista no declara un contrato temporal")
    return contracts[sorted(contracts)[0]]["protocol"]


def _protocol(protocol, window, seed):
    """Aceptar la ruta del protocolo o su documento. La huella es la de su JSON canónico."""
    if not isinstance(protocol, dict):
        protocol, _ = read_manifest(Path(protocol), 1024**2)
    digest = hashlib.sha256(canonical(protocol).encode()).hexdigest()
    rule = stopping_rule(protocol)
    folds = {fold["id"]: fold for fold in build_folds(protocol)}
    _require(window in folds, "La ventana no pertenece al protocolo")
    _require(seed in protocol["seeds"], "La semilla no está declarada en el protocolo")
    return protocol, digest, rule, folds[window]


def _recipe(path, rule, search_case=None, stopping=None):
    """Receta del caso con la regla del protocolo o, si la campaña la declara, su parada."""
    _, document = load_recipe(path)
    recipe = case_recipe(document, search_case)
    selection = {key: value for key, value in rule.items() if key != "max_epochs"}
    _require(
        recipe.epochs == rule["max_epochs"] and recipe.selection == selection,
        "La receta no aplica la regla de selección y parada del protocolo",
    )
    return (
        with_rule(recipe, campaign_rule(rule, stopping)),
        document,
        walk_forward_options(document),
    )


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


def control_config(local_control):
    """Contrato del control local C de la petición. Solo existe en `mac_online`."""
    if local_control is None:
        return None
    _require(isinstance(local_control, dict), "El control local se declara como un objeto")
    return MACProjectionConfig(**local_control)


def _predictor(document, specification, variant, seed, device, local_control=None):
    """Construir la variante con los parámetros iniciales copiados del emparejamiento.

    Con el control C, la variante es `mac_online` y coincide con la fuente del emparejamiento.
    """
    options = {key: value for key, value in document["predictor"].items() if key != "dtype"}
    dtype = DTYPES[document["predictor"]["dtype"]]
    control = control_config(local_control)
    _require(
        control is None or variant == document["pairing_source"],
        "El control C necesita que mac_online sea la fuente del emparejamiento",
    )

    def build(name):
        config = FinancialConfig(specification, variant=name, seed=seed, **options)
        return FinancialPredictor(config, local_control=control, device=device, dtype=dtype)

    target = build(variant)
    if variant == document["pairing_source"]:
        return target, None
    return target, copy_paired_parameters(build(document["pairing_source"]), target)


def _code():
    return {name: sha256(Path(importlib.import_module(name).__file__)) for name in _CODE}


def _request(
    view,
    protocol_sha,
    window,
    recipe,
    variant,
    seed,
    device,
    search_case=None,
    local_control=None,
    stopping=None,
):
    """Petición verificable sin abrir la vista: huellas de sus archivos y del código.

    El caso de búsqueda solo aparece si la receta los declara, el control C solo si se pide
    y la regla de parada solo si la campaña declara una parada temprana, de modo que las
    demás peticiones conservan su forma anterior.
    """
    request = dict(
        view_sha256=sha256(Path(view)),
        protocol_sha256=protocol_sha,
        window=window,
        recipe_sha256=sha256(Path(recipe)),
        variant=variant,
        seed=seed,
        device=device,
        code=_code(),
    )
    if search_case is not None:
        request["search_case"] = search_case
    if local_control is not None:
        request["local_control"] = asdict(control_config(local_control))
    if stopping is not None:
        request["stopping_rule"] = dict(stopping)
    return request


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


def checked_tables(rows, metrics, dataset, partition):
    """Tablas de un tramo con exactamente las filas y objetivos de la vista.

    Se exige además que el destino reciba cada etiqueta puntuada. Reutiliza la comprobación
    por mercado, activo e instante de la GRU candidata.
    """
    tables = rows.finish()
    _require(
        tables and rows.count == metrics["labels"],
        f"Las predicciones de {partition} no concilian con las etiquetas puntuadas",
    )
    check_view_rows(pa.concat_tables(tables), dataset, partition)
    return tables


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
    search_case=None,
    local_control=None,
    stopping=None,
    joint_epoch=None,
):
    """Ajustar una variante en una ventana y escribir sus predicciones por fila.

    La protección del aprendizaje se comprueba antes de leer ninguna fuente.
    `optimizer_factory` solo existe para comprobar el bucle con un optimizador que no
    modifica pesos. `search_case` elige uno de los casos de búsqueda de la receta, y es
    obligatorio si la receta los declara. `local_control` declara el control C de CM-v1
    (disabled para su B o penalty) y solo se admite con `mac_online`. `stopping` es la
    parada temprana que declara la campaña, con la misma métrica que el protocolo. Con la
    meseta conjunta la ventana devuelve `awaiting_joint_stop` en la primera meseta y se
    completa al reanudarla con la época común del grupo (`joint_epoch`).
    """
    require_learning_allowed("run_titans_window de Titans-MAC")
    _require(variant in VARIANTS, "La variante no pertenece a los controles de Titans-MAC")
    _require(
        local_control is None or variant == "mac_online",
        "El control local C solo se declara sobre mac_online",
    )
    control_config(local_control)
    _require(type(seed) is int and 0 <= seed < 2**32, "La semilla no es válida")
    _require(device in ("cpu", "cuda:0"), "El dispositivo debe ser cpu o cuda:0 explícitos")
    view, output = Path(view), Path(output)
    protocol_document, protocol_sha, rule, fold = _protocol(protocol, window, seed)
    chronological, document, options = _recipe(recipe, rule, search_case, stopping)
    rule = campaign_rule(rule, stopping)
    request = _request(
        view,
        protocol_sha,
        window,
        recipe,
        variant,
        seed,
        device,
        search_case,
        local_control,
        stopping,
    )
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
    predictor, pairing = _predictor(
        document, specification, variant, seed, device, request.get("local_control")
    )
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
        recipe=document,
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
        fit = trainer.run(resume=(output / "fit").exists(), stop=stop, joint_epoch=joint_epoch)
        if fit["status"] == AWAIT:
            report.update(status=AWAIT, individual_stop_epoch=fit["individual_stop_epoch"])
            return report
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
            joint_stop_epoch=fit.get("joint_stop_epoch"),
            global_step=trainer.global_step,
        )
        # Estado elegido del que salen las predicciones. Lo usa el recibo de la campaña.
        report["checkpoint"] = report["fit"]["best_checkpoint"]
        for name in PREDICTED:
            if name in report["predictions"]:
                continue
            if stop.requested:
                raise Paused
            rows = PredictionRows(trainer.quantiles)
            metrics = trainer.predict_partition(sources[name], rows, stop=stop)
            tables = checked_tables(rows, metrics, dataset, name)
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
            # Solo se escribe un tramo ya conciliado. Ningún archivo queda sin confirmar.
            path = output / f"{name}-predictions.parquet"
            written = atomic_parquet_batches(path, tables)
            _require(written == rows.count, f"El Parquet de {name} no conserva sus filas")
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


CARRY_KIND = "titans_carried_predictions"
CARRIED = ("calibration", "evaluation")
# Campos del caso que la campaña declara para cada brazo de Titans-MAC.
CASE_FIELDS = {"recipe", "recipe_sha256", "variant", "seed", "search_case"}
# La regla de parada solo aparece si la campaña declara una parada temprana.
STOPPING_FIELD = "stopping_rule"


def carried_memory_policy(warmup_months):
    """Política de la variante B: parámetros elegidos del ancla y memoria rápida reiniciada."""
    return dict(
        memory_policy(warmup_months),
        parameters="anchor_selected_state_without_further_fitting",
        fast_state="never_transferred_from_the_anchor_reset_at_each_pass",
    )


def _carried_parameters(model, target):
    """Cargar el estado elegido del ancla en el predictor de la ventana trasladada.

    Solo pueden cambiar las huellas de la vista y del índice de entrada. Representación,
    dimensiones, política, contexto, variante, semilla y arquitectura deben coincidir.
    """
    expected = target.get_extra_state()

    def portable(extra):
        configuration = dict(extra["configuration"])
        inputs = dict(configuration.pop("inputs"))
        inputs.pop("source_sha256")
        inputs.pop("view_sha256")
        return canonical(dict(extra, configuration=dict(configuration, inputs=inputs), training=0))

    anchor = model["_extra_state"]
    _require(
        portable(anchor) == portable(expected),
        "El estado del ancla no corresponde a la entrada ni a la arquitectura del predictor",
    )
    target.load_state_dict({**model, "_extra_state": dict(expected, training=anchor["training"])})


def _new_destination(output, protected):
    safe_destination(output)
    for source in protected:
        outside_source(source, output)
        outside_source(output, source)
    _require(
        not output.exists() and not output.is_symlink(),
        "Las predicciones trasladadas necesitan un directorio nuevo",
    )


def carry_titans(
    anchor, anchor_view, view, output, *, device="cuda:0", stop=None, modality_ablation=None
):
    """Predecir una ventana posterior con el estado elegido en la ventana ancla.

    Es la pieza de la variante B. No ajusta parámetros ni selección. Cada tramo trasladado
    empieza con la memoria rápida inicial y su propio calentamiento de entradas. Con
    `modality_ablation` predice solo la evaluación, también en la propia ventana del ancla,
    y el calentamiento y el tramo leen las mismas entradas ablacionadas, así que la memoria
    rápida también ve la ausencia.
    """
    require_learning_allowed("la predicción trasladada de Titans-MAC")
    from .carried_predictions import ablation_record, carried_window, predicted_partitions
    from .checkpoints import load_training_state
    from .financial_run import ChronologicalInference

    started = time.perf_counter()
    anchor, anchor_view, view, output = (Path(v) for v in (anchor, anchor_view, view, output))
    _require(device in ("cpu", "cuda:0"), "El dispositivo debe ser cpu o cuda:0 explícitos")
    report, report_sha = read_manifest(anchor / "run.json", 16 * 1024**2)
    _require(
        report.get("kind") == KIND
        and report.get("status") == "completed"
        and report.get("final_test_opened") is False
        and isinstance(report.get("checkpoint"), dict),
        "El ancla no es una ventana de Titans-MAC completada",
    )
    _require(
        sha256(anchor_view) == report["request"]["view_sha256"],
        "La vista del ancla no es la de su ajuste",
    )
    _verify(anchor, report)
    identity = report["identity"]
    document, request = identity["recipe"], identity["request"]
    options = walk_forward_options(document)
    # El recorrido congelado usa el truncamiento y los bloques del caso elegido en el ancla.
    chronological = case_recipe(document, request.get("search_case"))
    anchor_manifest, _ = read_manifest(anchor_view, 64 * 1024**2)
    if device == "cuda:0":
        from mars_titan.data.embeddings import require_cuda

        require_cuda()
        _require(
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"},
            "Configura CUBLAS_WORKSPACE_CONFIG antes de iniciar PyTorch",
        )
    dataset = CorpusDataset(
        view, input_policy=HISTORICAL_MASKED, modality_ablation=modality_ablation
    )
    anchor_fold, fold, age = carried_window(
        anchor_manifest,
        dataset.manifest,
        input_policy=HISTORICAL_MASKED,
        same_window=modality_ablation is not None,
    )
    _check_view(dataset, view_protocol(view), fold)
    _new_destination(output, (*dataset.roots.values(), view.parent, anchor))
    phases = window_phases(fold, options["warmup_months"])
    partitions = predicted_partitions(modality_ablation)
    sources = _sources(dataset, {name: phases[name] for name in partitions}, output / "indices")
    options_predictor = {k: v for k, v in document["predictor"].items() if k != "dtype"}
    predictor = FinancialPredictor(
        FinancialConfig(
            sources[partitions[0]].specification(),
            variant=request["variant"],
            seed=request["seed"],
            **options_predictor,
        ),
        local_control=control_config(request.get("local_control")),
        device=device,
        dtype=DTYPES[document["predictor"]["dtype"]],
    )
    fit, _ = read_manifest(anchor / "fit/run.json", 16 * 1024**2)
    state = load_training_state(
        anchor / "fit/checkpoints",
        expected_identity=fit["identity"],
        selection="best",
        expected_sha256=report["checkpoint"]["sha256"],
    )
    _carried_parameters(state["model"], predictor)
    inference = ChronologicalInference(predictor, chronological)
    predictions = {}
    for name in partitions:
        rows = PredictionRows(inference.quantiles)
        metrics = inference.predict(sources[name], rows, stop=stop)
        tables = checked_tables(rows, metrics, dataset, name)
        path = output / f"{name}-predictions.parquet"
        written = atomic_parquet_batches(path, tables)
        _require(written == rows.count, f"El Parquet de {name} no conserva sus filas")
        predictions[name] = dict(
            path=path.name,
            sha256=sha256(path),
            rows=written,
            bytes=path.stat().st_size,
            metrics=metrics,
        )
    receipt = dict(
        schema_version=1,
        kind=CARRY_KIND,
        status="completed",
        anchor=dict(
            run_sha256=report_sha,
            checkpoint_sha256=report["checkpoint"]["sha256"],
            view_sha256=report["request"]["view_sha256"],
            fold=anchor_fold,
            variant=request["variant"],
            seed=request["seed"],
            **({"search_case": request["search_case"]} if "search_case" in request else {}),
        ),
        view_sha256=sha256(view),
        fold=fold,
        months_since_anchor_information=age,
        memory_policy=carried_memory_policy(options["warmup_months"]),
        phases={name: asdict(phases[name]) for name in partitions},
        indices={name: source.identity for name, source in sources.items()},
        device=device,
        code=_code(),
        predictions=predictions,
        final_test_opened=False,
        scientific_training_started=False,
        seconds=time.perf_counter() - started,
        finished_at_utc=datetime.now(UTC).isoformat(),
        **ablation_record(modality_ablation),
    )
    atomic_json(output / "carry.json", receipt)
    return receipt


@contextmanager
def unfused_attention():
    """Declarar fastpath=False durante un trabajo de Titans-MAC y restaurar el estado previo.

    El recorrido cronológico lo exige. La orden de una ventana lo fija para todo el
    proceso. En la campaña, los demás brazos conservan la configuración del proceso.
    """
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def _campaign_case(run):
    case = run.case
    _require(
        isinstance(case, dict)
        and set(case) - {STOPPING_FIELD} == CASE_FIELDS
        and case["seed"] == run.job["seed"]
        and run.policy == HISTORICAL_MASKED,
        "El trabajo no declara un caso de Titans-MAC de la campaña con máscaras",
    )
    _require(
        sha256(Path(case["recipe"])) == case["recipe_sha256"],
        "La receta de Titans-MAC cambió después de planificar la campaña",
    )
    return case


def titans_fit(run, *, device="cuda:0", optimizer_factory=None):
    """Ejecutor de ajuste para `training.masked_campaign`, con su `JobRun`.

    Ajusta la variante y semilla del caso en la ventana del trabajo y devuelve el informe
    con las predicciones por fila y el estado elegido. La campaña reserva la GPU.
    """
    from .masked_campaign import Paused as CampaignPaused

    case = _campaign_case(run)
    with unfused_attention():
        report = run_titans_window(
            run.view,
            view_protocol(run.view),
            run.job["window"],
            case["recipe"],
            variant=case["variant"],
            seed=case["seed"],
            output=run.folder,
            device=device,
            stop=run.stop,
            optimizer_factory=optimizer_factory,
            search_case=case["search_case"],
            stopping=case.get(STOPPING_FIELD),
            joint_epoch=run.joint_epoch,
        )
    if report["status"] == "paused":
        raise CampaignPaused
    _require(
        report["status"] in ("completed", AWAIT)
        and report["request"]["view_sha256"] == run.view_sha256,
        "La ventana de Titans-MAC no confirma la vista del trabajo",
    )
    return report


def titans_carry(run, *, device="cuda:0"):
    """Ejecutor de predicción trasladada para `training.masked_campaign` (variante B)."""
    from .masked_campaign import Paused as CampaignPaused

    try:
        with unfused_attention():
            return carry_titans(
                run.anchor["folder"],
                run.anchor["view"],
                run.view,
                run.folder,
                device=device,
                stop=run.stop,
            )
    except Paused as error:
        raise CampaignPaused from error
