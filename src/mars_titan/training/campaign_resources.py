"""Recursos declarados por trabajo de la campaña y admisión acotada de GPU, CPU y RAM.

La declaración de ejecución es un archivo aparte de la campaña. Fija las ranuras GPU, los
trabajadores CPU, los presupuestos de VRAM y RAM del anfitrión, CUDA MPS y la tubería de
lectura, y asigna a cada modelo del ejecutor su dispositivo, su VRAM y su RAM estimadas,
con valores propios por ámbito si hacen falta. No forma parte de la identidad científica:
cambiarla no cambia qué datos ve un trabajo ni sus salidas, solo cuántos se ejecutan a la
vez. Un trabajo cuya estimación supera el presupuesto se rechaza antes de empezar la
campaña.

Sin declaración, la campaña conserva su ejecución anterior: los trabajos CUDA de uno en
uno en el mismo proceso y los CPU con los `cpu_workers` de la sección tabular.
"""

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json

from .input_pipeline import PipelineOptions

EXECUTION_KIND = "historical_masked_campaign_execution"
DEVICES = ("cpu", "cuda")
MAX_GPU_SLOTS = 4
MAX_CPU_WORKERS = 8
MAX_THREADS_PER_JOB = 8
MIB = 1024**2
# Memoria de un proceso CUDA fuera del asignador de PyTorch: contexto, cuBLAS y cuDNN.
CONTEXT_MIB = 512
# Reintentos de un trabajo que agota su VRAM, cada uno con una reserva mayor.
MAX_OOM_RETRIES = 2
OBSERVED_KIND = "historical_masked_campaign_observed_resources"
_FIELDS = {"schema_version", "kind", "gpu", "host", "pipeline", "models", "notes"}
_GPU = {"slots", "vram_budget_mib", "mps", "mps_pipe_directory"}
_HOST = {"cpu_workers", "threads_per_job", "ram_budget_mib", "reserve_mib"}
_MODEL = {"device", "vram_mib", "host_mib", "scopes", "arms", "campaign_process"}
_SIZES = {"vram_mib", "host_mib"}
_PIPELINE = {"decode_workers", "prefetch_batches", "group_cache_mib"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _mib(value, label, *, lower=0, upper=64 * 1024):
    _require(
        type(value) is int and lower <= value <= upper,
        f"{label} debe ser un entero de MiB entre {lower} y {upper}",
    )
    return value * MIB


@dataclass(frozen=True)
class JobResources:
    """Dispositivo, VRAM y RAM que un trabajo declara antes de ejecutarse.

    `campaign_process` ejecuta el trabajo en un hilo del proceso de la campaña aunque haya
    ranuras, para que conserve el estado que comparte con los siguientes del mismo modelo
    (la matriz cuantizada de XGBoost o las estadísticas de Ridge de una ventana). Ese
    proceso no acota la VRAM del trabajo, así que su declaración debe ser una cota medida.
    """

    device: str
    vram_bytes: int = 0
    host_bytes: int = 0
    campaign_process: bool = False


@dataclass(frozen=True)
class Execution:
    """Concurrencia y presupuestos de una ejecución de la campaña."""

    gpu_slots: int = 1
    cpu_workers: int = 1
    threads_per_job: int | None = None
    vram_budget_bytes: int | None = None
    host_budget_bytes: int | None = None
    host_reserve_bytes: int = 0
    mps: bool = False
    mps_pipe_directory: str | None = None
    pipeline: PipelineOptions | None = None
    models: dict = field(default_factory=dict)
    path: str | None = None
    sha256: str | None = None

    @property
    def isolated(self):
        """Con más de una ranura, cada trabajo GPU corre en su propio proceso."""
        return self.gpu_slots > 1

    def resources(self, job, default_device):
        """Recursos del trabajo: los de su brazo, modelo y ámbito, o su dispositivo sin más."""
        declared = self.models.get(job["model"])
        if declared is None:
            return JobResources(device=default_device)
        declared = declared["arms"].get(job.get("arm"), declared)
        return declared["scopes"].get(job["scope"], declared["default"])

    def record(self):
        return dict(
            path=self.path,
            sha256=self.sha256,
            gpu_slots=self.gpu_slots,
            cpu_workers=self.cpu_workers,
            threads_per_job=self.threads_per_job,
            vram_budget_bytes=self.vram_budget_bytes,
            host_budget_bytes=self.host_budget_bytes,
            host_reserve_bytes=self.host_reserve_bytes,
            mps=self.mps,
            pipeline=None if self.pipeline is None else self.pipeline.environment(),
        )


def _model(name, value, scopes):
    _require(
        isinstance(value, dict) and set(value) <= _MODEL and {"device", "host_mib"} <= set(value),
        f"Los recursos de {name} no cumplen su contrato",
    )
    device = value["device"]
    _require(device in DEVICES, f"{name} declara un dispositivo desconocido")
    campaign_process = value.get("campaign_process", False)
    _require(
        type(campaign_process) is bool and (device == "cuda" or not campaign_process),
        f"{name} solo puede ejecutarse en el proceso de la campaña si es CUDA",
    )

    def resources(label, *entries):
        # Cada entrada sustituye a las anteriores: modelo, ámbito, brazo y brazo en el ámbito.
        sizes = {}
        for entry in entries:
            sizes.update(entry)
        vram, host = sizes.get("vram_mib"), sizes["host_mib"]
        if device == "cuda":
            _require(vram is not None, f"{label} necesita su VRAM estimada")
            vram_bytes = _mib(vram, f"La VRAM de {label}", lower=1)
        else:
            _require(vram in (None, 0), f"{label} es CPU y no declara VRAM")
            vram_bytes = 0
        host_bytes = _mib(host, f"La RAM de {label}", lower=1)
        return JobResources(device, vram_bytes, host_bytes, campaign_process)

    def scoped(entry, label):
        overrides = entry.get("scopes", {})
        _require(
            isinstance(overrides, dict)
            and set(overrides) <= set(scopes)
            and all(
                isinstance(sizes, dict) and sizes and set(sizes) <= _SIZES
                for sizes in overrides.values()
            ),
            f"Los ámbitos de {label} no pertenecen a la campaña o no declaran VRAM o RAM",
        )
        return overrides

    base = {key: value[key] for key in _SIZES & set(value)}
    model_scopes = scoped(value, name)
    arms = value.get("arms", {})
    _require(
        isinstance(arms, dict)
        and all(
            isinstance(entry, dict) and entry and set(entry) <= _SIZES | {"scopes"}
            for entry in arms.values()
        ),
        f"Los brazos de {name} deben declarar VRAM, RAM o ámbitos",
    )
    declared = dict(
        default=resources(name, base),
        scopes={
            scope: resources(f"{name} en {scope}", base, sizes)
            for scope, sizes in model_scopes.items()
        },
        arms={},
    )
    # Un brazo hereda lo del modelo en cada ámbito y lo sustituye con lo suyo. Sus recursos
    # se resuelven aquí para todos los ámbitos y un error aparece al cargar la declaración.
    for arm, entry in arms.items():
        own, arm_scopes = {key: entry[key] for key in _SIZES & set(entry)}, scoped(entry, arm)
        declared["arms"][arm] = dict(
            default=resources(f"{name}/{arm}", base, own),
            scopes={
                scope: resources(
                    f"{name}/{arm} en {scope}",
                    base,
                    model_scopes.get(scope, {}),
                    own,
                    arm_scopes.get(scope, {}),
                )
                for scope in scopes
            },
        )
    return declared


def load_execution(path, *, scopes=("US", "CN", "US+CN")):
    """Validar una declaración de ejecución. Las comprobaciones contra el plan van aparte."""
    path = Path(path)
    document, digest = read_manifest(path, 256 * 1024)
    _require(
        isinstance(document, dict)
        and _FIELDS - {"notes"} <= set(document) <= _FIELDS
        and document["schema_version"] == 1
        and document["kind"] == EXECUTION_KIND,
        "La declaración de ejecución no cumple su contrato",
    )
    gpu, host, pipeline = document["gpu"], document["host"], document["pipeline"]
    _require(
        isinstance(gpu, dict)
        and {"slots", "vram_budget_mib", "mps"} <= set(gpu) <= _GPU
        and type(gpu["slots"]) is int
        and 1 <= gpu["slots"] <= MAX_GPU_SLOTS
        and type(gpu["mps"]) is bool
        and (not gpu["mps"] or gpu["slots"] > 1)
        and (gpu.get("mps_pipe_directory") is None or gpu["mps"]),
        f"Las ranuras GPU deben estar entre 1 y {MAX_GPU_SLOTS}, y MPS solo se declara con "
        "varias ranuras",
    )
    mps_directory = gpu.get("mps_pipe_directory")
    _require(
        mps_directory is None
        or (isinstance(mps_directory, str) and Path(mps_directory).is_absolute()),
        "El directorio de MPS debe ser una ruta absoluta",
    )
    _require(
        isinstance(host, dict)
        and set(host) == _HOST
        and type(host["cpu_workers"]) is int
        and 1 <= host["cpu_workers"] <= MAX_CPU_WORKERS
        and type(host["threads_per_job"]) is int
        and 1 <= host["threads_per_job"] <= MAX_THREADS_PER_JOB,
        f"Los trabajadores CPU deben estar entre 1 y {MAX_CPU_WORKERS} y los hilos por "
        f"trabajo entre 1 y {MAX_THREADS_PER_JOB}",
    )
    _require(
        isinstance(pipeline, dict) and set(pipeline) == _PIPELINE,
        "La tubería de lectura declara decode_workers, prefetch_batches y group_cache_mib",
    )
    models = document["models"]
    _require(isinstance(models, dict) and models, "La declaración necesita recursos por modelo")
    return Execution(
        gpu_slots=gpu["slots"],
        cpu_workers=host["cpu_workers"],
        threads_per_job=host["threads_per_job"],
        vram_budget_bytes=_mib(gpu["vram_budget_mib"], "El presupuesto de VRAM", lower=256),
        host_budget_bytes=_mib(host["ram_budget_mib"], "El presupuesto de RAM", lower=1024),
        host_reserve_bytes=_mib(host["reserve_mib"], "La reserva de RAM"),
        mps=gpu["mps"],
        mps_pipe_directory=mps_directory,
        pipeline=PipelineOptions(**pipeline),
        models={name: _model(name, value, scopes) for name, value in models.items()},
        path=str(path.resolve()),
        sha256=digest,
    )


def check_plan(execution, jobs, executors):
    """Exigir orden topológico, recursos para cada trabajo y que cada uno quepa solo.

    Sin ranuras la ejecución es en serie, así que un trabajo anterior a su dependencia
    dejaría la campaña detenida.
    """
    seen, arms = set(), {}
    for job in jobs:
        arms.setdefault(job["model"], set()).add(job.get("arm"))
        _require(
            all(dependency in seen for dependency in job["depends"]),
            f"{job['id']} aparece en el plan antes que alguna de sus dependencias",
        )
        seen.add(job["id"])
        key = job["model"], job["kind"]
        _require(key in executors, f"No hay ejecutor para {job['id']}")
        if not execution.models:
            continue
        _require(job["model"] in execution.models, f"La ejecución no declara {job['model']}")
        resources = execution.resources(job, executors[key]["device"])
        _require(
            resources.device == executors[key]["device"],
            f"{job['id']} declara {resources.device} y su ejecutor usa {executors[key]['device']}",
        )
        if resources.device == "cuda" and execution.vram_budget_bytes is not None:
            _require(
                resources.vram_bytes <= execution.vram_budget_bytes,
                f"{job['id']} estima {resources.vram_bytes // MIB} MiB de VRAM y supera el "
                f"presupuesto de {execution.vram_budget_bytes // MIB} MiB",
            )
        if execution.host_budget_bytes is not None:
            _require(
                resources.host_bytes <= execution.host_budget_bytes,
                f"{job['id']} estima {resources.host_bytes // MIB} MiB de RAM y supera el "
                f"presupuesto de {execution.host_budget_bytes // MIB} MiB",
            )
    for name, declared in execution.models.items():
        unknown = sorted(set(declared["arms"]) - arms.get(name, set()))
        _require(not unknown, f"{name} declara brazos que no tiene en el plan: {unknown}")


def available_host_bytes():
    """`MemAvailable` del anfitrión, la memoria que se puede reservar sin intercambio."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("No se puede leer MemAvailable")


class ResourcePool:
    """Admisión de trabajos por ranuras, VRAM y RAM declaradas.

    Un trabajo entra si su dispositivo tiene una ranura libre, la suma de VRAM y RAM
    declaradas de los trabajos en curso más la suya cabe en el presupuesto, y la memoria
    disponible del anfitrión cubre su RAM más la reserva. Sin trabajos en curso de su
    dispositivo, la ranura y la VRAM siempre admiten uno, porque `check_plan` ya comprobó
    que cabe solo.
    """

    def __init__(self, execution, *, available=available_host_bytes):
        self.execution, self.available = execution, available
        self.running = {"cpu": [], "cuda": []}
        # Contexto CUDA que conserva el proceso de la campaña tras su primer trabajo CUDA.
        self.resident = 0

    def hold_context(self):
        """Contar desde ahora el contexto CUDA del proceso de la campaña.

        Mientras corre un trabajo de ese proceso, su declaración ya incluye el contexto.
        """
        self.resident = CONTEXT_MIB * MIB

    def _slots(self, device):
        return self.execution.gpu_slots if device == "cuda" else self.execution.cpu_workers

    def waits_for_campaign(self, resources):
        """Si el trabajo espera al único hilo de los trabajos del proceso de la campaña."""
        return resources.campaign_process and any(r.campaign_process for r in self.running["cuda"])

    def admits(self, resources):
        execution, running = self.execution, self.running
        if len(running[resources.device]) >= self._slots(resources.device):
            return False
        if self.waits_for_campaign(resources):
            return False
        busy = [item for items in running.values() for item in items]
        if resources.device == "cuda" and execution.vram_budget_bytes is not None:
            jobs = [*running["cuda"], resources]
            resident = 0 if any(r.campaign_process for r in jobs) else self.resident
            if sum(r.vram_bytes for r in jobs) + resident > execution.vram_budget_bytes:
                return False
        if execution.host_budget_bytes is not None and (
            sum(r.host_bytes for r in busy) + resources.host_bytes > execution.host_budget_bytes
        ):
            return False
        # Con trabajos en curso, su memoria ya reservada cuenta en MemAvailable. El primero
        # también necesita la reserva, salvo que no declare RAM (ejecución anterior).
        return not resources.host_bytes or (
            self.available() >= resources.host_bytes + execution.host_reserve_bytes
        )

    def acquire(self, resources):
        _require(self.admits(resources), "El trabajo no cabe en los recursos libres")
        self.running[resources.device].append(resources)

    def release(self, resources):
        self.running[resources.device].remove(resources)


def legacy_execution(campaign):
    """Ejecución anterior a la declaración: una ranura GPU en el proceso de la campaña."""
    return Execution(
        gpu_slots=1,
        cpu_workers=campaign["tabular"]["cpu_workers"],
        pipeline=None,
        models={},
    )


def environment(execution):
    """Variables de la tubería, los hilos y MPS que hereda cada proceso de un trabajo."""
    values = {} if execution.pipeline is None else execution.pipeline.environment()
    if execution.threads_per_job is not None:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            values[name] = str(execution.threads_per_job)
    if execution.mps:
        directory = execution.mps_pipe_directory or os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
        _require(directory, "MPS necesita CUDA_MPS_PIPE_DIRECTORY o mps_pipe_directory")
        values["CUDA_MPS_PIPE_DIRECTORY"] = directory
    return values


class PeakEstimates:
    """VRAM de cada tipo de trabajo: la declarada o la observada, la mayor de las dos.

    El tipo es el modelo, el brazo y el ámbito. Cuando un trabajo termina en su proceso, su
    pico reservado por el asignador más `CONTEXT_MIB` sustituye a la estimación si es mayor,
    de modo que los siguientes trabajos del mismo tipo se admiten con lo observado. Nunca baja
    de lo declarado, que se mide en la ventana más poblada. Si un trabajo agota su VRAM, la
    estimación sube la mitad (al menos 1 GiB) sin pasar del presupuesto y el trabajo se
    repite, como mucho `MAX_OOM_RETRIES` veces. El estado se guarda de forma atómica en
    `path` para que una reanudación lo conserve.
    """

    def __init__(self, execution, path):
        self.execution, self.path = execution, Path(path)
        self.observed, self.retries = {}, {}
        if self.path.exists():
            document = json.loads(self.path.read_text())
            _require(
                isinstance(document, dict)
                and document.get("kind") == OBSERVED_KIND
                and isinstance(document.get("observed"), dict)
                and all(
                    isinstance(entry, dict) and type(entry.get("vram_bytes")) is int
                    for entry in document["observed"].values()
                ),
                "El registro de recursos observados no cumple su contrato",
            )
            self.observed = document["observed"]

    @staticmethod
    def key(job):
        # Los brazos de un mismo modelo pueden diferir mucho en VRAM, así que cada uno corrige
        # solo su propia estimación.
        return f"{job['model']}/{job.get('arm')}/{job['scope']}"

    def resources(self, job, default_device):
        declared = self.execution.resources(job, default_device)
        entry = self.observed.get(self.key(job))
        if declared.device != "cuda" or entry is None:
            return declared
        vram = max(declared.vram_bytes, entry["vram_bytes"])
        if self.execution.vram_budget_bytes is not None:
            vram = min(vram, self.execution.vram_budget_bytes)
        return replace(declared, vram_bytes=vram)

    def _save(self, job, vram_bytes, **fields):
        key = self.key(job)
        entry = dict(self.observed.get(key, {}))
        entry.update(fields, vram_bytes=max(vram_bytes, entry.get("vram_bytes", 0)))
        self.observed[key] = entry
        atomic_json(self.path, dict(kind=OBSERVED_KIND, observed=self.observed))

    def observe(self, job, usage):
        """Registrar el pico reservado de un trabajo terminado en su proceso."""
        peak = usage.get("peak_vram_reserved_bytes")
        if type(peak) is int and peak > 0:
            self._save(job, peak + CONTEXT_MIB * MIB, peak_reserved_bytes=peak)

    def grow(self, job, current):
        """Subir la estimación tras agotar la VRAM. Falso si no quedan reintentos o margen."""
        count = self.retries.get(job["id"], 0)
        budget = self.execution.vram_budget_bytes
        if count >= MAX_OOM_RETRIES or (budget is not None and current.vram_bytes >= budget):
            return False
        grown = max(current.vram_bytes * 3 // 2, current.vram_bytes + 1024 * MIB)
        if budget is not None:
            grown = min(grown, budget)
        self.retries[job["id"]] = count + 1
        oom = self.observed.get(self.key(job), {}).get("oom", 0) + 1
        self._save(job, grown, oom=oom)
        return True
