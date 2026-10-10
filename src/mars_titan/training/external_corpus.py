"""Referencia XGBoost CUDA recuperable, con lecturas acotadas del corpus supervisado."""

import argparse
import atexit
import fcntl
import math
import os
import re
import resource
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
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
from mars_titan.models.baselines.boosting_selection import BoostingSelection, ResidentValidation
from mars_titan.models.baselines.external_boosting import (
    ExternalBoostingModel,
    _libraries,
    available_ram_bytes,
    build_external_matrix,
    directory_bytes,
    external_cache_plan,
    fit_external_boosting,
    free_disk_bytes,
)

from .checkpoints import StopRequest
from .corpus_inputs import CorpusDataset
from .learning_hold import require_learning_allowed
from .reference_run import FULL_TRAIN_VALIDATION, PREDICTION_RETENTIONS
from .tabular_corpus import _Errors, _matrix, _predict, feature_order, retained_partitions

# Opciones del recorrido que no son parámetros del ajuste externo.
_READER_OPTIONS = {
    "batch_size",
    "max_validation_cache_bytes",
    "input_policy",
    "prediction_retention",
}


def _partitions(options):
    """Las opciones sin el campo conservan la retención anterior de ajuste y validación."""
    return retained_partitions(options.get("prediction_retention", FULL_TRAIN_VALIDATION))


# Validación residente en la GPU: el resto queda en RAM y se copia en cada ronda. La
# reserva deja sitio a las páginas, gradientes e histogramas del ajuste en 8 GB.
VALIDATION_DEVICE_BYTES = 2 * 1024**3
DEVICE_RESERVE_BYTES = 3 * 1024**3


@dataclass
class _TrainRows:
    """Mercado, instante y objetivo float64 de cada fila de la matriz, en su orden.

    Se registran en la primera pasada de la construcción y permiten resumir el ajuste
    con las predicciones de la matriz cuantizada, sin volver a leer el corpus.
    """

    expected: int
    china: np.ndarray | None = None
    moments: np.ndarray | None = None
    target: np.ndarray | None = None
    parts: list | None = None

    def start(self):
        """Registrar solo la primera pasada completa."""
        recording = self.target is None
        if recording:
            self.parts = []
        return recording

    def add(self, batch):
        self.parts.append(
            (
                np.asarray(batch["market"]) == "CN",
                np.asarray(batch["prediction_at"], dtype="datetime64[us]").astype(np.int64),
                np.array(batch["target"], dtype=np.float64),
            )
        )

    def finish(self):
        china, moments, target = (
            np.concatenate(values) for values in zip(*self.parts, strict=True)
        )
        if len(target) != self.expected:
            raise ValueError("Las claves de la matriz no conservan la población")
        self.china, self.moments, self.target, self.parts = china, moments, target, None


@dataclass
class _TrainingMatrix:
    matrix: object
    rows: _TrainRows

    def close(self):
        self.matrix.close()


class SharedWindow:
    """Matriz cuantizada y validación de una ventana, vivas mientras su clave no cambie.

    Las configuraciones con el mismo `max_bin`, filas y lotes comparten los mismos cortes
    y páginas, y todas comparten la validación. Solo hay una matriz viva por proceso y su
    uso es exclusivo durante cada ajuste. `release` libera páginas, RAM y GPU.

    Entre procesos, un cerrojo de archivo junto al directorio compartido lo reserva para
    el proceso que conserva la matriz. Otro proceso, como una ranura GPU con un proceso
    por trabajo, espera a que se libere en vez de borrar páginas ajenas o duplicar el pico
    de disco. El sistema suelta el cerrojo si el proceso muere, y el siguiente borra sus
    restos antes de construir.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.matrix_key = self.matrix = self.directory = self.handle = None
        self.validation_key = self.validation = None
        self.constructions = 0

    def reclaimable_disk_bytes(self):
        with self.lock:
            return self.matrix.matrix.disk_bytes() if self.matrix else 0

    def resident_bytes(self):
        with self.lock:
            rows = self.matrix.rows.expected * 17 if self.matrix else 0
            return rows + (self.validation.bytes if self.validation else 0)

    @contextmanager
    def training_matrix(self, key, directory, build, *, stopped=None):
        """Ceder la matriz de la clave, construyéndola en `directory` si no existe.

        Mientras otro proceso reserva el directorio se espera, y `stopped` permite atender
        una parada durante la espera.
        """
        with self.lock:
            built = self.matrix_key != key
            if built:
                self._release_matrix()
                directory = Path(directory)
                safe_destination(directory)
                self._reserve(directory, stopped)
                try:
                    if directory.exists():
                        # Restos de un proceso terminado: el cerrojo impide que alguien los use.
                        shutil.rmtree(directory)
                    self.matrix = build(directory)
                except BaseException:
                    if directory.exists():
                        shutil.rmtree(directory)
                    self._unreserve()
                    raise
                self.matrix_key, self.directory = key, directory
                self.constructions += 1
            yield self.matrix, built

    def _reserve(self, directory, stopped):
        directory.parent.mkdir(parents=True, exist_ok=True)
        handle = (directory.parent / f"{directory.name}.lock").open("a")
        try:
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if stopped is not None and stopped():
                        raise _Paused from None
                    time.sleep(1.0)
        except BaseException:
            handle.close()
            raise
        self.handle = handle

    def _unreserve(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def discard_stale(self, directory):
        """Borrar las páginas de un proceso terminado si nadie reserva el directorio."""
        directory = Path(directory)
        with self.lock:
            if not directory.exists():
                return 0
            with (directory.parent / f"{directory.name}.lock").open("a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return 0
                safe_destination(directory)
                size = directory_bytes(directory)
                shutil.rmtree(directory)
                return size

    def resident_validation(self, key, build):
        with self.lock:
            if self.validation_key != key:
                self._release_validation()
                self.validation = build()
                self.validation_key = key
            return self.validation

    def _release_matrix(self):
        if self.matrix is not None:
            self.matrix.close()
        self._unreserve()
        self.matrix_key = self.matrix = self.directory = None

    def _release_validation(self):
        if self.validation is not None:
            self.validation.release()
        self.validation_key = self.validation = None

    def release(self, *, blocking=True):
        """Liberar todo. Sin bloqueo, no espera a un ajuste que esté usando la matriz."""
        if not self.lock.acquire(blocking=blocking):
            return False
        try:
            self._release_matrix()
            self._release_validation()
            return True
        finally:
            self.lock.release()


SHARED = SharedWindow()
atexit.register(SHARED.release)


def _device_budget(cp):
    free, _ = cp.cuda.runtime.memGetInfo()
    return max(0, min(VALIDATION_DEVICE_BYTES, free - DEVICE_RESERVE_BYTES))


def _matrix_metrics(model, training, batch_size):
    """Resumir el ajuste con las predicciones de la matriz, con los lotes del lector.

    Los árboles hist dividen en cortes de la propia matriz y el predictor de ELLPACK
    devuelve el límite inferior de cada bin, así que cada fila toma las mismas ramas que
    con sus float32. Los lotes de `batch_size` filas reproducen la acumulación de
    `_predict` sobre el corpus.
    """
    prediction, rows = model.predict_matrix(training.matrix), training.rows
    errors = _Errors()
    for start in range(0, rows.expected, batch_size):
        end = start + batch_size
        errors.add(
            prediction[start:end],
            rows.target[start:end],
            np.where(rows.china[start:end], "CN", "US"),
            rows.moments[start:end],
        )
    return errors.summary()


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
    prediction_retention=FULL_TRAIN_VALIDATION,
    shared_directory=None,
):
    """Recorrer todas las filas admitidas sin abrir el test ni reducir la población.

    Con la retención reservada se escriben validación, calibración y evaluación por
    fila del modelo seleccionado y el ajuste solo se resume con las predicciones de la
    matriz cuantizada.

    Antes de crear la salida se estima la caché con la población declarada y se
    falla si no cabe en los presupuestos de RAM o disco ni en lo disponible.

    Con `shared_directory`, la matriz y la validación de la ventana se conservan en
    `SHARED` para las configuraciones siguientes. Es una opción de rendimiento: no
    forma parte de la identidad y el modelo es el mismo que sin compartir.
    """
    require_learning_allowed("el ajuste XGBoost del corpus")
    if type(batch_size) is not int or not 1 <= batch_size <= 4096 or type(resume) is not bool:
        raise ValueError("El lote o el modo de recuperación no son válidos")
    if input_policy not in INPUT_POLICIES:
        raise ValueError("La política de entradas no está admitida")
    retained_partitions(prediction_retention)
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
    shared = SHARED if shared_directory is not None else None
    # Lo que ya ocupa la ventana compartida se reutiliza o se libera antes de construir.
    plan = external_cache_plan(
        rows=dataset.manifest["counts"]["train"],
        features=features,
        max_bin=max_bin,
        on_host=on_host,
        max_host_cache_bytes=max_host_cache_bytes,
        max_disk_cache_bytes=max_disk_cache_bytes,
        available_ram=available_ram_bytes() + (shared.resident_bytes() if shared else 0),
        free_disk=free_disk_bytes(output) + (shared.reclaimable_disk_bytes() if shared else 0),
        resident_bytes=validation_bytes,
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
    if prediction_retention != FULL_TRAIN_VALIDATION:
        options["prediction_retention"] = prediction_retention
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
            if set(report["predictions"]) != set(_partitions(options)):
                raise ValueError("Las predicciones terminadas no siguen la retención declarada")
            for partition in _partitions(options):
                item = report["predictions"][partition]
                path = output / f"{partition}-predictions.parquet"
                safe_destination(path)
                if item["path"] != path.name or sha256(path) != item["sha256"]:
                    raise ValueError("Una predicción terminada no conserva su integridad")
            return report
        if len(report["attempts"]) >= 9999:
            raise ValueError("Se ha alcanzado el límite de intentos de recuperación")
        return _execute(
            dataset,
            output,
            report,
            parent,
            cp,
            xgb,
            stop or StopRequest(),
            plan=plan,
            shared=shared,
            shared_directory=None if shared is None else Path(shared_directory),
        )
    finally:
        os.close(lock)


@contextmanager
def _training_matrix(shared, key, directory, build, stopped=None):
    """Matriz propia, liberada al terminar, o cedida por la ventana compartida."""
    if shared is None:
        training = build(directory)
        try:
            yield training, True
        finally:
            training.close()
        return
    with shared.training_matrix(key, directory, build, stopped=stopped) as (training, built):
        yield training, built


def _remove_stale_temporaries(output):
    """Un intento interrumpido puede dejar páginas y validación en su directorio temporal."""
    removed = 0
    for path in output.glob("external-*"):
        if path.is_dir() and not path.is_symlink():
            safe_destination(path)
            removed += sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            shutil.rmtree(path)
    return removed


def _execute(
    dataset, output, report, parent, cp, xgb, stop, plan=None, shared=None, shared_directory=None
):
    options = report["identity"]["options"]
    batch_size = options["batch_size"]
    policy = options.get("input_policy", STRICT_INPUTS)
    presence = masked_inputs(policy)
    started = time.perf_counter()
    attempt = dict(
        started_at_utc=datetime.now(UTC).isoformat(),
        observed_device_used_bytes_max=0,
        memory_plan=plan,
        stale_temporary_bytes_removed=_remove_stale_temporaries(output),
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

    rows = _TrainRows(report["samples"]["train"])

    def factory():
        recording = rows.start()
        for batch in dataset.batches(partition="train", batch_size=batch_size, epoch=0, seed=0):
            if stop.requested:
                raise _Paused
            if recording:
                rows.add(batch)
            yield _matrix(batch, np.float32, presence=presence), batch["target"]
        if recording:
            rows.finish()

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

    construction = dict(
        expected_rows=report["samples"]["train"],
        max_bin=options["max_bin"],
        max_batch_bytes=options["max_batch_bytes"],
        max_host_cache_bytes=options["max_host_cache_bytes"],
        on_host=options["on_host"],
        max_disk_cache_bytes=options.get("max_disk_cache_bytes"),
    )
    # Las filas, su orden y sus lotes quedan fijados por el manifiesto, la política y el lote.
    source = (dataset.identity, policy, batch_size)
    matrix_key = (*source, tuple(sorted(construction.items())))

    def build_matrix(directory):
        return _TrainingMatrix(build_external_matrix(factory, directory, **construction), rows)

    def build_validation():
        return ResidentValidation(
            validation,
            expected_rows=report["samples"]["validation"],
            max_bytes=options["max_validation_cache_bytes"],
        )

    resident = None
    try:
        observed_memory()
        atomic_json(output / "run.json", report)
        if stop.requested:
            raise _Paused
        fit_start = time.perf_counter()
        with (
            tempfile.TemporaryDirectory(prefix="external-", dir=output) as temporary,
            _training_matrix(
                shared,
                matrix_key,
                shared_directory or Path(temporary) / "pages",
                build_matrix,
                lambda: stop.requested,
            ) as (training, built),
        ):
            attempt["matrix"] = dict(
                shared=shared is not None,
                constructed=built,
                construction_passes=list(training.matrix.iterator.completed),
                disk_bytes=training.matrix.disk_bytes(),
            )
            validation_options = {}
            if "selection" in options:
                resident = (
                    shared.resident_validation(
                        (*source, options["max_validation_cache_bytes"]), build_validation
                    )
                    if shared
                    else build_validation()
                )
                attempt["validation_cache_bytes"] = resident.bytes
                attempt["validation_device_bytes"] = resident.place(_device_budget(cp))
                validation_options = dict(
                    validation_factory=resident,
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
                matrix=training.matrix,
                **validation_options,
            )
            if resident is not None:
                # True si las rondas sumaron solo el árbol nuevo, False si se usó la completa.
                attempt["validation_leaf_sums"] = resident.incremental
                resident.release_device()
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
            for partition in _partitions(options):
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
            if "train" not in predictions:
                # Las particiones reservadas ya han comparado el modelo recargado.
                report["train_metrics"] = _matrix_metrics(restored, training, batch_size)
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
        if resident is not None:
            resident.release_device()
            if shared is None:
                resident.release()
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
    parser.add_argument(
        "--prediction-retention", choices=PREDICTION_RETENTIONS, default=FULL_TRAIN_VALIDATION
    )
    parser.add_argument("--resume", action="store_true")
    options = vars(parser.parse_args())
    options["on_host"] = not options.pop("disk_cache")
    with StopRequest() as stop:
        result = run_external_reference(**options, stop=stop)
    print(f"Estado: {result['status']}. Rondas confirmadas: {result['completed_rounds']}")


if __name__ == "__main__":
    main()
