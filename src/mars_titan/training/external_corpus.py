"""Referencia XGBoost CUDA recuperable, con lecturas acotadas del corpus supervisado."""

import argparse
import fcntl
import math
import os
import re
import resource
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import (
    INPUT_POLICIES,
    STRICT_INPUTS,
    masked_inputs,
    policy_identity,
)
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.models.baselines.boosting_selection import BoostingSelection, ValidationCache
from mars_titan.models.baselines.external_boosting import (
    ExternalBoostingModel,
    _libraries,
    available_ram_bytes,
    external_cache_plan,
    fit_external_boosting,
    free_disk_bytes,
)

from .checkpoints import StopRequest
from .corpus_inputs import CorpusDataset
from .learning_hold import require_learning_allowed
from .tabular_corpus import _matrix, _predict, feature_order

# Opciones del recorrido que no son parámetros del ajuste externo.
_READER_OPTIONS = {"batch_size", "max_validation_cache_bytes", "input_policy"}


class _Paused(Exception):
    """Parada confirmada tras guardar una ronda completa."""


def _identity(dataset, options, cp, xgb):
    root = Path(__file__).parents[1]
    policy = options.get("input_policy", STRICT_INPUTS)
    names = (
        "training/external_corpus.py",
        "training/tabular_corpus.py",
        "training/corpus_inputs.py",
        "training/temporal_corpus.py",
        "evaluation/splits.py",
        "evaluation/split_readiness.py",
        "training/cohort_contract.py",
        "models/baselines/external_boosting.py",
        "models/baselines/boosting_selection.py",
        "models/baselines/inputs.py",
        "evaluation/session_metrics.py",
        "data/streaming.py",
        "data/batches.py",
        "data/storage.py",
        "data/cohort_files.py",
        "data/embeddings.py",
    ) + (("data/input_policy.py",) if masked_inputs(policy) else ())
    return dict(
        manifest_sha256=dataset.identity,
        cohort=dataset.cohort,
        **policy_identity(policy),
        options=options,
        xgboost=xgb.__version__,
        cupy=cp.__version__,
        numpy=np.__version__,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        cuda_driver=cp.cuda.runtime.driverGetVersion(),
        code={name: sha256(root / name) for name in names},
    )


def _load(output, checkpoint, rows):
    if not isinstance(checkpoint, dict) or not re.fullmatch(
        r"checkpoints/attempt-\d{4}-round-\d{4}\.ubj", checkpoint.get("path", "")
    ):
        raise ValueError("El recibo no contiene una ruta de modelo confirmada")
    path = output / checkpoint["path"]
    safe_destination(path)
    return ExternalBoostingModel.load(path, checkpoint["sha256"], training_rows=rows)


def _prune_selection(output, report):
    """Retirar solo modelos propios tras confirmar el recibo que conserva sus sustitutos."""
    recent = report["recovery_checkpoints"]
    if (
        not isinstance(recent, list)
        or not 1 <= len(recent) <= 2
        or recent[-1] != report["recovery_checkpoint"]
    ):
        raise ValueError("La recuperación no conserva uno o dos checkpoints recientes")
    retained = recent + [report["checkpoint"]]
    for record in retained:
        if not isinstance(record, dict) or not re.fullmatch(
            r"checkpoints/attempt-\d{4}-round-\d{4}\.ubj", record.get("path", "")
        ):
            raise ValueError("La retención solo admite rutas propias de modelos confirmados")
        path = output / record["path"]
        safe_destination(path)
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError("No se retiran modelos sin confirmar todos sus sustitutos")
    keep = {record["path"] for record in retained}
    directory = output / "checkpoints"
    for path in directory.iterdir():
        if (
            re.fullmatch(r"attempt-\d{4}-round-\d{4}\.ubj", path.name)
            and str(path.relative_to(output)) not in keep
        ):
            safe_destination(path)
            path.unlink()


def _confirm_selection(output, report, model):
    state = model.audit["selection"]
    options = report["identity"]["options"]
    BoostingSelection(options["selection"], options["rounds"], state=state)
    count = model.booster.num_boosted_rounds()
    if state["completed_rounds"] != count:
        raise ValueError("La recuperación necesita el booster de la última evaluación completa")
    path = output / "checkpoints" / f"attempt-{len(report['attempts']):04}-round-{count:04}.ubj"
    record = dict(path=str(path.relative_to(output)), sha256=model.save(path), rounds=count)
    recent = (report.get("recovery_checkpoints", []) + [record])[-2:]
    best = record if state["selected_round"] == count else report["checkpoint"]
    if (
        best is None
        or state["selected_round"] != count
        and (report.get("selected_round") != state["selected_round"])
    ):
        raise ValueError("El mejor modelo no tiene un checkpoint confirmado")
    confirmed = dict(
        report,
        checkpoint=best,
        recovery_checkpoint=record,
        recovery_checkpoints=recent,
        fitted_rows=model.training_rows,
        consumed_training_rows=model.training_rows * count,
        completed_rounds=count,
        selected_round=state["selected_round"],
        stop_reason=state["stop_reason"],
        selection=dict(state),
        audit=model.audit,
    )
    atomic_json(output / "run.json", confirmed)
    report.clear()
    report.update(confirmed)
    _prune_selection(output, report)


def run_external_reference(
    manifest,
    output,
    *,
    rounds=100,
    max_depth=4,
    max_bin=128,
    learning_rate=0.05,
    seed=42,
    batch_size=256,
    max_batch_bytes=64 * 1024**2,
    max_host_cache_bytes=16 * 1024**3,
    on_host=True,
    checkpoint_interval=10,
    selection=None,
    max_validation_cache_bytes=16 * 1024**3,
    resume=False,
    stop=None,
    input_policy=STRICT_INPUTS,
    max_disk_cache_bytes=None,
):
    """Recorrer todas las filas admitidas sin abrir el test ni reducir la población.

    Antes de crear la salida se estima la caché con la población declarada y se
    falla si no cabe en los presupuestos de RAM o disco ni en lo disponible.
    """
    require_learning_allowed("el ajuste XGBoost del corpus")
    if type(batch_size) is not int or not 1 <= batch_size <= 4096 or type(resume) is not bool:
        raise ValueError("El lote o el modo de recuperación no son válidos")
    if input_policy not in INPUT_POLICIES:
        raise ValueError("La política de entradas no está admitida")
    masked = masked_inputs(input_policy)
    if masked and not on_host and max_disk_cache_bytes is None:
        raise ValueError("La edición con máscaras necesita un presupuesto de disco explícito")
    if selection is not None:
        BoostingSelection(selection, rounds)
        if (
            type(max_validation_cache_bytes) is not int
            or not 1 <= max_validation_cache_bytes <= 32 * 1024**3
        ):
            raise ValueError("El presupuesto de caché de validación no es válido")
    dataset, output = CorpusDataset(Path(manifest), input_policy=input_policy), Path(output)
    if min(dataset.manifest["counts"].values()) < 1:
        raise ValueError("Se necesitan entrenamiento y validación no vacíos")
    for protected in (*dataset.roots.values(), Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)
    safe_destination(output)
    if output.exists() and not resume:
        raise ValueError("La referencia necesita una salida nueva o recuperación explícita")
    if resume and not (output / "run.json").is_file():
        raise ValueError("No existe un informe recuperable")
    first = next(dataset.batches(partition="train", batch_size=batch_size, epoch=0, seed=0))
    features = _matrix(first, np.float32, presence=masked).shape[1]
    validation_rows = dataset.manifest["counts"]["validation"]
    validation_bytes = (
        # Valores float32, objetivo y fecha de 8 bytes, mercado <U2 y cabeceras por bloque.
        min(
            max_validation_cache_bytes,
            validation_rows * (features * 4 + 24) + 4096 * math.ceil(validation_rows / batch_size),
        )
        if selection is not None
        else 0
    )
    plan = external_cache_plan(
        rows=dataset.manifest["counts"]["train"],
        features=features,
        max_bin=max_bin,
        on_host=on_host,
        max_host_cache_bytes=max_host_cache_bytes,
        max_disk_cache_bytes=max_disk_cache_bytes,
        available_ram=available_ram_bytes(),
        free_disk=free_disk_bytes(output),
        other_disk_bytes=validation_bytes,
    )
    cp, xgb = _libraries()
    options = dict(
        rounds=rounds,
        max_depth=max_depth,
        max_bin=max_bin,
        learning_rate=learning_rate,
        seed=seed,
        batch_size=batch_size,
        max_batch_bytes=max_batch_bytes,
        max_host_cache_bytes=max_host_cache_bytes,
        on_host=on_host,
        checkpoint_interval=checkpoint_interval,
    )
    if selection is not None:
        options["selection"] = dict(selection)
        options["max_validation_cache_bytes"] = max_validation_cache_bytes
    if masked:
        options["input_policy"] = input_policy
    if max_disk_cache_bytes is not None:
        options["max_disk_cache_bytes"] = max_disk_cache_bytes
    identity = _identity(dataset, options, cp, xgb)
    output.mkdir(parents=True, exist_ok=resume)
    lock = os.open(output / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = (
            read_manifest(output / "run.json", 8 * 1024**2)[0]
            if resume
            else dict(
                schema_version=2 if selection is not None else 1,
                model="xgboost_external_cuda",
                device="cuda:0",
                identity=identity,
                samples=dataset.manifest["counts"],
                scope=dataset.manifest["scope"],
                cohort_complete=dataset.manifest["cohort_complete"],
                final_test_opened=False,
                feature_order=feature_order(input_policy),
                fitted_rows=0,
                completed_rounds=0,
                status="pending",
                checkpoint=None,
                attempts=[],
            )
        )
        if report.get("identity") != identity:
            raise ValueError("La identidad de datos, código o configuración ha cambiado")
        if selection is not None:
            if report.get("schema_version") != 2:
                raise ValueError("La selección necesita un recibo de la edición declarada")
            if report["checkpoint"]:
                _prune_selection(output, report)
        parent = (
            _load(
                output,
                report["recovery_checkpoint"] if selection is not None else report["checkpoint"],
                report["samples"]["train"],
            )
            if report["checkpoint"]
            else None
        )
        if selection is not None and parent is not None:
            state = BoostingSelection(selection, rounds, state=report["selection"]).state
            if (
                parent.audit.get("selection") != state
                or parent.audit["rounds"] != state["completed_rounds"]
            ):
                raise ValueError(
                    "El booster y la selección no pertenecen al mismo punto confirmado"
                )
        if report["status"] == "completed":
            if parent is None or (
                selection is None
                and parent.audit["rounds"] != rounds
                or selection is not None
                and not report["selection"]["stop_reason"]
            ):
                raise ValueError("La referencia terminada no conserva todas sus rondas")
            if selection is not None:
                selected = _load(output, report["checkpoint"], report["samples"]["train"])
                if selected.audit["rounds"] != report["selected_round"]:
                    raise ValueError("El modelo seleccionado no conserva su ronda")
            for partition in ("train", "validation"):
                item = report["predictions"][partition]
                path = output / f"{partition}-predictions.parquet"
                safe_destination(path)
                if item["path"] != path.name or sha256(path) != item["sha256"]:
                    raise ValueError("Una predicción terminada no conserva su integridad")
            return report
        if len(report["attempts"]) >= 9999:
            raise ValueError("Se ha alcanzado el límite de intentos de recuperación")
        return _execute(dataset, output, report, parent, cp, xgb, stop or StopRequest(), plan=plan)
    finally:
        os.close(lock)


def _execute(dataset, output, report, parent, cp, xgb, stop, plan=None):
    options = report["identity"]["options"]
    batch_size = options["batch_size"]
    presence = masked_inputs(options.get("input_policy", STRICT_INPUTS))
    started = time.perf_counter()
    attempt = dict(
        started_at_utc=datetime.now(UTC).isoformat(),
        observed_device_used_bytes_max=0,
        memory_plan=plan,
    )
    report["attempts"].append(attempt)
    report["status"] = "running"

    def observed_memory():
        free, total = cp.cuda.runtime.memGetInfo()
        attempt["observed_device_used_bytes_max"] = max(
            attempt["observed_device_used_bytes_max"], total - free
        )

    def confirm(model):
        if _identity(dataset, options, cp, xgb) != report["identity"]:
            raise ValueError("El código científico ha cambiado durante el ajuste")
        count = model.booster.num_boosted_rounds()
        if "selection" in options:
            state = model.audit["selection"]
            if (
                state["selected_round"] == count
                or count % options["checkpoint_interval"] == 0
                or state["stop_reason"] is not None
                or stop.requested
            ):
                observed_memory()
                _confirm_selection(output, report, model)
            return stop.requested
        path = output / "checkpoints" / f"attempt-{len(report['attempts']):04}-round-{count:04}.ubj"
        digest = model.save(path)
        report.update(
            checkpoint=dict(path=str(path.relative_to(output)), sha256=digest),
            fitted_rows=model.training_rows,
            completed_rounds=count,
            audit=model.audit,
        )
        observed_memory()
        atomic_json(output / "run.json", report)
        if stop.requested:
            raise _Paused

    def factory():
        for batch in dataset.batches(partition="train", batch_size=batch_size, epoch=0, seed=0):
            if stop.requested:
                raise _Paused
            yield _matrix(batch, np.float32, presence=presence), batch["target"]

    def validation():
        for batch in dataset.batches(
            partition="validation", batch_size=batch_size, epoch=0, seed=0
        ):
            if stop.requested:
                raise _Paused
            yield (
                _matrix(batch, np.float32, presence=presence),
                batch["target"],
                batch["market"],
                batch["prediction_at"],
            )

    try:
        observed_memory()
        atomic_json(output / "run.json", report)
        if stop.requested:
            raise _Paused
        fit_start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="external-", dir=output) as temporary:
            validation_options = {}
            if "selection" in options:
                cached = ValidationCache(
                    validation,
                    Path(temporary) / "validation",
                    expected_rows=report["samples"]["validation"],
                    max_bytes=options["max_validation_cache_bytes"],
                )
                attempt["validation_cache_bytes"] = cached.bytes
                validation_options = dict(
                    validation_factory=cached,
                    validation_rows=report["samples"]["validation"],
                    stop_requested=lambda: stop.requested,
                )
            model = fit_external_boosting(
                factory,
                Path(temporary) / "pages",
                expected_rows=report["samples"]["train"],
                **{key: value for key, value in options.items() if key not in _READER_OPTIONS},
                resume=parent,
                checkpoint=confirm,
                **validation_options,
            )
        attempt["fit_seconds"] = time.perf_counter() - fit_start
        if "selection" in options:
            attempt["replayed_rounds"] = model.audit.get("replayed_rounds", 0)
            attempt["replay_seconds"] = model.audit.get("replay_seconds", 0.0)
            attempt["replayed_training_rows"] = (
                attempt["replayed_rounds"] * report["samples"]["train"]
            )
        if stop.requested:
            raise _Paused
        if "selection" not in options and report["completed_rounds"] != options["rounds"]:
            confirm(model)
        restored = _load(output, report["checkpoint"], report["samples"]["train"])
        predictions = {}
        for partition in ("train", "validation"):
            if stop.requested:
                raise _Paused
            path = output / f"{partition}-predictions.parquet"
            metrics = _predict(
                model,
                restored,
                dataset,
                partition,
                batch_size,
                path,
                dtype=np.float32,
                presence=presence,
            )
            predictions[partition] = dict(path=path.name, sha256=sha256(path), metrics=metrics)
        if "selection" in options and (
            not report["selection"]["stop_reason"]
            or report["selected_round"] != restored.booster.num_boosted_rounds()
            or predictions["validation"]["metrics"]["session_mae"]
            != report["selection"]["best_session_mae"]
        ):
            raise ValueError("La evaluación final no coincide con el mejor modelo seleccionado")
        if _identity(dataset, options, cp, xgb) != report["identity"]:
            raise ValueError("La identidad cambió durante la evaluación")
        report.update(status="completed", predictions=predictions, restored_predictions_equal=True)
    except _Paused:
        report["status"] = "paused"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        observed_memory()
        attempt.update(
            status=report["status"],
            total_seconds=time.perf_counter() - started,
            process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            finished_at_utc=datetime.now(UTC).isoformat(),
            memory_scope=(
                "Muestras de toda la GPU, incluidas otras aplicaciones. No es el pico del proceso."
            ),
        )
        atomic_json(output / "run.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-bin", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batch-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--max-host-cache-bytes", type=int, default=16 * 1024**3)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--disk-cache", action="store_true")
    parser.add_argument("--max-disk-cache-bytes", type=int)
    parser.add_argument("--input-policy", choices=INPUT_POLICIES, default=STRICT_INPUTS)
    parser.add_argument("--resume", action="store_true")
    options = vars(parser.parse_args())
    options["on_host"] = not options.pop("disk_cache")
    with StopRequest() as stop:
        result = run_external_reference(**options, stop=stop)
    print(f"Estado: {result['status']}. Rondas confirmadas: {result['completed_rounds']}")


if __name__ == "__main__":
    main()
