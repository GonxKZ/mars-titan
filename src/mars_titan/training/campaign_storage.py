"""Presupuesto de disco de la campaña con máscaras, liberación al confirmar y guardia.

La campaña escribe por ajuste tablas por fila, puntos de control, índices de observaciones
y cachés. Este módulo reúne tres piezas que no leen datos ni ajustan modelos:

- ``job_footprint`` calcula los bytes que un trabajo conserva al terminar y los que ocupa
  solo mientras se ejecuta, a partir de los recuentos de su vista y de una declaración de
  almacenamiento medida (``configs/baselines/historical-masked-campaign-storage.json``).
  ``plan_peak`` recorre el plan en orden y da el máximo de ocupación simultánea.
- ``release_confirmed`` borra, después de escribir el recibo de un trabajo, lo que ningún
  consumidor vuelve a leer: los índices de observaciones, que se reconstruyen desde la
  vista, y los puntos de control de recuperación. Conserva el estado elegido, que es el
  que leen los hijos, los traslados y las etapas posteriores.
- ``DiskGuard`` rechaza lanzar si el pico proyectado no cabe en el espacio libre menos el
  margen declarado, no admite un trabajo que no quepa y convierte un espacio libre por
  debajo del margen en una parada recuperable en la siguiente barrera de cada ejecutor.
"""

import json
import math
import shutil
import threading
import time
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json

STORAGE_KIND = "historical_masked_campaign_storage"
HELD_OUT = ("validation", "calibration", "evaluation")
# Escritor de predicciones de cada modelo de `masked_campaign.EXECUTORS`.
WRITERS = {
    "neural": "neural",
    "ridge": "ridge",
    "xgboost": "xgboost",
    "episodic_gru": "episodic_gru",
    "titans_mac": "titans",
    "mars_titan": "titans",
    "cm_v1_core": "titans",
    "cm_v1": "titans",
}
# Carpeta de puntos de control dentro del intento y modelos con índice de observaciones.
CHECKPOINTS = {
    "neural": "checkpoints",
    "episodic_gru": "run/checkpoints",
    "titans_mac": "fit/checkpoints",
    "mars_titan": "fit/checkpoints",
    "cm_v1_core": "fit/checkpoints",
    "cm_v1": "fit/checkpoints",
}
INDEXED = ("episodic_gru", "titans_mac", "mars_titan", "cm_v1_core", "cm_v1")
_FIELDS = {
    "schema_version",
    "kind",
    "status",
    "margin_bytes",
    "check_seconds",
    "release_on_confirmation",
    "prediction_row_bytes",
    "job_report_bytes",
    "train_sessions_bytes",
    "state_bytes",
    "retained_states",
    "recovery_state_extra_bytes",
    "flow_state_bytes",
    "index",
    "xgboost",
    "measurement",
}


class DiskBudgetError(ValueError):
    """El pico proyectado no cabe en el disco con el margen declarado."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def load_storage(path):
    """Validar la declaración de almacenamiento antes de usarla en ninguna proyección."""
    path = Path(path)
    document, digest = read_manifest(path, 256 * 1024)
    _require(
        isinstance(document, dict)
        and set(document) == _FIELDS
        and document["schema_version"] == 1
        and document["kind"] == STORAGE_KIND
        and document["status"] == "declared_before_launch"
        and type(document["margin_bytes"]) is int
        and document["margin_bytes"] >= 1024**3
        and _positive(document["check_seconds"])
        and 0 < document["check_seconds"] <= 300
        and type(document["release_on_confirmation"]) is bool,
        "La declaración de almacenamiento no cumple su contrato",
    )
    rows = document["prediction_row_bytes"]
    writers = set(WRITERS.values())
    _require(
        isinstance(rows, dict)
        and set(rows) == writers
        and all(
            isinstance(rows[w], dict)
            and set(rows[w]) == set(HELD_OUT)
            and all(_positive(v) and v > 0 for v in rows[w].values())
            for w in writers
        ),
        "Faltan bytes por fila medidos de algún escritor o tramo",
    )
    for name in ("state_bytes", "retained_states", "recovery_state_extra_bytes"):
        _require(
            isinstance(document[name], dict)
            and set(document[name]) == set(WRITERS)
            and all(type(v) is int and v >= 0 for v in document[name].values()),
            f"{name} debe declarar un entero por modelo",
        )
    index, boosting = document["index"], document["xgboost"]
    _require(
        isinstance(index, dict)
        and set(index) == {"bytes_per_row", "build_bytes_per_row", "warmup_partition"}
        and _positive(index["bytes_per_row"])
        and _positive(index["build_bytes_per_row"])
        and index["warmup_partition"] in HELD_OUT
        and isinstance(boosting, dict)
        and set(boosting) == {"features", "max_bin"}
        and all(type(v) is int and v > 0 for v in boosting.values())
        and all(
            type(document[name]) is int and document[name] >= 0
            for name in ("job_report_bytes", "train_sessions_bytes", "flow_state_bytes")
        )
        and isinstance(document["measurement"], dict),
        "Los índices, XGBoost o los tamaños fijos no están declarados",
    )
    return dict(document, path=str(path.resolve()), sha256=digest)


def view_counts(views):
    """Filas por tramo y flujos de cada ámbito y ventana desde `masked_campaign.scope_views`.

    Los flujos son los activos de la vista: el estado a mitad de época de la familia Titans
    guarda memoria rápida y cola pendiente por flujo.
    """
    counts = {}
    for scope, value in views.items():
        counts[scope] = {}
        for window, record in value["windows"].items():
            manifest, _ = read_manifest(Path(record["path"]), 64 * 1024**2)
            counts[scope][window] = dict(record["counts"], flows=len(manifest["assets"]))
    return counts


def _case_bins(job, boosting):
    """Contenedores del caso de búsqueda, o el máximo declarado si el caso aún no existe."""
    case = job.get("case")
    return (
        case.get("max_bin", boosting["max_bin"]) if isinstance(case, dict) else boosting["max_bin"]
    )


def index_rows(counts, warmup, *, partitions=HELD_OUT, train=True):
    """Filas de los índices de un trabajo: muestra y etiqueta por fila de cada fase.

    Las fases reservadas añaden las muestras de calentamiento, aproximadas por el tramo
    declarado de doce meses cuando el escritor calienta y por cero cuando no.
    """
    extra = counts[warmup] if warmup else 0
    rows = sum(2 * counts[p] + extra for p in partitions)
    return rows + (2 * counts["train"] if train else 0)


def job_footprint(job, counts, storage, *, release=None, prediction_bytes=None):
    """Bytes que un trabajo conserva al confirmarse y bytes extra mientras se ejecuta.

    `prediction_bytes(job, partition)` sustituye los bytes por fila declarados por una
    medida exacta. `release` usa por defecto la política declarada. Un traslado solo
    escribe calibración y evaluación y no tiene estados propios ni caché de XGBoost.
    """
    release = storage["release_on_confirmation"] if release is None else release
    model = job["model"]
    _require(model in WRITERS, f"No hay huella declarada para el modelo {model}")
    writer = WRITERS[model]
    carry = job.get("kind") == "carry"
    partitions = HELD_OUT[1:] if carry else HELD_OUT
    measured = prediction_bytes or (
        lambda _, partition: counts[partition] * storage["prediction_row_bytes"][writer][partition]
    )
    tables = {partition: int(measured(job, partition)) for partition in partitions}
    state = 0 if carry else storage["state_bytes"][model]
    kept = storage["retained_states"][model]
    releasable = release and model in CHECKPOINTS
    retained = dict(
        predictions=sum(tables.values()),
        reports=storage["job_report_bytes"],
        sessions=storage["train_sessions_bytes"] if model == "neural" and not carry else 0,
        states=state * (min(kept, 1) if releasable else kept),
        indices=0,
    )
    transient = dict(
        recovery=state * max(kept - 1, 0) if releasable else 0,
        mid_epoch=0 if carry else storage["recovery_state_extra_bytes"][model],
        writing=max(tables.values()),
        indices=0,
        cache=0,
    )
    if model in INDEXED:
        index = storage["index"]
        warmup = None if model == "episodic_gru" else index["warmup_partition"]
        rows = index_rows(counts, warmup, partitions=partitions, train=not carry)
        built = int(rows * index["bytes_per_row"])
        largest = max(2 * counts[p] for p in ((*partitions, "train") if not carry else partitions))
        transient["indices"] = int(largest * index["build_bytes_per_row"])
        if release:
            transient["indices"] += built
        else:
            retained["indices"] = built
    if model in ("titans_mac", "mars_titan", "cm_v1_core", "cm_v1") and not carry:
        transient["mid_epoch"] += 2 * counts.get("flows", 0) * storage["flow_state_bytes"]
    if model == "xgboost" and not carry:
        boosting = storage["xgboost"]
        # Páginas ELLPACK densas: ⌈log₂ max_bin⌉ bits por valor, medidos con filas reales.
        # La validación reside en RAM y en la GPU, sin caché en disco.
        bits = (int(_case_bins(job, boosting)) - 1).bit_length()
        transient["cache"] = math.ceil(counts["train"] * boosting["features"] * bits / 8)
    return dict(
        retained=retained,
        transient=transient,
        retained_bytes=sum(retained.values()),
        transient_bytes=sum(transient.values()),
    )


def plan_peak(jobs, counts, storage, *, done=(), release=None, prediction_bytes=None):
    """Recorrer el plan en orden y dar el pico de ocupación nueva de los trabajos pendientes.

    Los trabajos confirmados ya ocupan el disco y no suman. Cada trabajo pendiente suma lo
    que conserva, y en su turno también lo que ocupa mientras se ejecuta.
    """
    done = set(done)
    cumulative, peak, worst = 0, 0, None
    for job in jobs:
        if job["id"] in done:
            continue
        footprint = job_footprint(
            job,
            counts[job["scope"]][job["window"]],
            storage,
            release=release,
            prediction_bytes=prediction_bytes,
        )
        moment = cumulative + footprint["retained_bytes"] + footprint["transient_bytes"]
        if moment > peak:
            peak, worst = moment, job["id"]
        cumulative += footprint["retained_bytes"]
    return dict(peak_bytes=peak, retained_bytes=cumulative, peak_job=worst)


def free_bytes(path, usage=shutil.disk_usage):
    """Bytes que puede escribir un usuario sin privilegios en el sistema de `path`."""
    path = Path(path)
    while not path.exists():
        path = path.parent
    return usage(path).free


class DiskGuard:
    """Margen de disco declarado para lanzar, admitir trabajos y pausar sin romper nada.

    `watch` envuelve la solicitud de parada de la campaña. Los ejecutores ya consultan
    `requested` en sus barreras seguras y guardan un estado recuperable al verla. La
    consulta del disco se limita a una cada `check_seconds`. Cada trabajo admitido reserva
    su huella completa hasta `settle`, así que varios trabajos simultáneos no cuentan dos
    veces con el mismo espacio libre.
    """

    def __init__(
        self, path, margin_bytes, *, check_seconds=5, usage=shutil.disk_usage, clock=time.monotonic
    ):
        _require(type(margin_bytes) is int and margin_bytes > 0, "El margen debe ser positivo")
        self.path, self.margin, self.interval = Path(path), margin_bytes, check_seconds
        self.usage, self.clock = usage, clock
        self.low, self.checked, self.last_free = False, None, None
        self.reserved, self.lock = {}, threading.Lock()

    def free(self):
        self.last_free = free_bytes(self.path, self.usage)
        return self.last_free

    def require_launch(self, projection):
        """Rechazar el lanzamiento si el pico de lo pendiente no cabe sobre el margen."""
        free = self.free()
        if projection["peak_bytes"] + self.margin > free:
            raise DiskBudgetError(
                f"El pico proyectado de {projection['peak_bytes']} bytes en "
                f"{projection['peak_job']} más el margen de {self.margin} bytes supera los "
                f"{free} bytes libres. Libera espacio o declara una retención más compacta"
            )
        return dict(free_bytes=free, margin_bytes=self.margin, **projection)

    def admits(self, need_bytes, key=None):
        """Un trabajo solo empieza si lo que ocupará, sumado a lo reservado, deja el margen.

        Con `key`, el trabajo admitido reserva `need_bytes` hasta `settle(key)`.
        """
        with self.lock:
            free = self.free() - sum(self.reserved.values())
            admitted = free - need_bytes >= self.margin
            if admitted and key is not None:
                self.reserved[key] = need_bytes
            return admitted

    def settle(self, key):
        """El trabajo terminó: lo que ocupa ya figura en el espacio libre del disco."""
        with self.lock:
            self.reserved.pop(key, None)

    def below_margin(self):
        now = self.clock()
        if self.checked is None or now - self.checked >= self.interval:
            self.checked = now
            self.low = self.free() < self.margin
        return self.low

    def watch(self, stop):
        return _GuardedStop(stop, self)

    def state(self):
        return dict(
            free_bytes=self.last_free,
            margin_bytes=self.margin,
            below_margin=self.low,
            reserved_bytes=sum(self.reserved.values()),
        )


class _GuardedStop:
    """Solicitud de parada que también se activa con el disco por debajo del margen."""

    def __init__(self, stop, guard):
        self.stop, self.guard = stop, guard

    @property
    def requested(self):
        return bool(self.stop.requested) or self.guard.below_margin()


def _size(path):
    if path.is_symlink() or not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_blocks * 512
    return sum(p.lstat().st_blocks * 512 for p in path.rglob("*") if not p.is_symlink())


def release_confirmed(folder, model):
    """Liberar lo que no vuelve a leerse de un intento cuyo recibo ya está escrito.

    Borra los índices de observaciones y los puntos de control de recuperación, y deja el
    estado elegido, los informes y las tablas por fila. Escribe `released.json` con lo
    liberado y es idempotente.
    """
    folder = Path(folder)
    safe_destination(folder)
    released = {}
    indices = folder / "indices"
    if model in INDEXED and indices.is_dir() and not indices.is_symlink():
        released["indices"] = _size(indices)
        shutil.rmtree(indices)
    if model in CHECKPOINTS:
        from .checkpoints import release_recovery_states

        directory = folder / CHECKPOINTS[model]
        if directory.is_dir():
            released["recovery_states"] = release_recovery_states(directory)
    if model == "xgboost":
        # Restos de páginas de una parada forzada. La ejecución confirmada no los usa.
        for path in folder.glob("external-*"):
            if path.is_dir() and not path.is_symlink():
                released.setdefault("external_caches", 0)
                released["external_caches"] += _size(path)
                shutil.rmtree(path)
    record = folder / "released.json"
    if released and any(released.values()):
        previous = json.loads(record.read_text()) if record.is_file() else {}
        merged = {k: previous.get(k, 0) + released.get(k, 0) for k in {*previous, *released}}
        atomic_json(record, merged)
    return released


def projection_report(jobs, counts, storage, done=()):
    """Resumen de la proyección para el recibo de la campaña."""
    projection = plan_peak(jobs, counts, storage, done=done)
    return dict(
        storage_sha256=storage["sha256"],
        pending_jobs=sum(job["id"] not in set(done) for job in jobs),
        **projection,
    )


def confirmed_ids(output, jobs):
    """Trabajos con recibo escrito, sin verificarlos: solo sirven para no contarlos."""
    output = Path(output)
    return {job["id"] for job in jobs if (output / "jobs" / job["id"] / "receipt.json").is_file()}
