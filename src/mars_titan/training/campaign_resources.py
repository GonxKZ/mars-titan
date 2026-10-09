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

import os
from dataclasses import dataclass, field
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest

from .input_pipeline import PipelineOptions

EXECUTION_KIND = "historical_masked_campaign_execution"
DEVICES = ("cpu", "cuda")
MAX_GPU_SLOTS = 4
MAX_CPU_WORKERS = 8
MIB = 1024**2
_FIELDS = {"schema_version", "kind", "gpu", "host", "pipeline", "models", "notes"}
_GPU = {"slots", "vram_budget_mib", "mps", "mps_pipe_directory"}
_HOST = {"cpu_workers", "ram_budget_mib", "reserve_mib"}
_MODEL = {"device", "vram_mib", "host_mib", "scopes"}
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
    """Dispositivo, VRAM y RAM que un trabajo declara antes de ejecutarse."""

    device: str
    vram_bytes: int = 0
    host_bytes: int = 0


@dataclass(frozen=True)
class Execution:
    """Concurrencia y presupuestos de una ejecución de la campaña."""

    gpu_slots: int = 1
    cpu_workers: int = 1
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
        """Recursos del trabajo: los de su modelo y ámbito, o su dispositivo sin estimación."""
        declared = self.models.get(job["model"])
        if declared is None:
            return JobResources(device=default_device)
        return declared["scopes"].get(job["scope"], declared["default"])

    def record(self):
        return dict(
            path=self.path,
            sha256=self.sha256,
            gpu_slots=self.gpu_slots,
            cpu_workers=self.cpu_workers,
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

    def resources(entry, label):
        vram = entry.get("vram_mib", value.get("vram_mib"))
        host = entry.get("host_mib", value["host_mib"])
        if device == "cuda":
            _require(vram is not None, f"{label} necesita su VRAM estimada")
            vram_bytes = _mib(vram, f"La VRAM de {label}", lower=1)
        else:
            _require(vram in (None, 0), f"{label} es CPU y no declara VRAM")
            vram_bytes = 0
        return JobResources(device, vram_bytes, _mib(host, f"La RAM de {label}", lower=1))

    overrides = value.get("scopes", {})
    _require(
        isinstance(overrides, dict)
        and set(overrides) <= set(scopes)
        and all(
            isinstance(entry, dict) and entry and set(entry) <= {"vram_mib", "host_mib"}
            for entry in overrides.values()
        ),
        f"Los ámbitos de {name} no pertenecen a la campaña o no declaran VRAM o RAM",
    )
    return dict(
        default=resources({}, name),
        scopes={
            scope: resources(entry, f"{name} en {scope}") for scope, entry in overrides.items()
        },
    )


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
        and 1 <= host["cpu_workers"] <= MAX_CPU_WORKERS,
        f"Los trabajadores CPU deben estar entre 1 y {MAX_CPU_WORKERS}",
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
    seen = set()
    for job in jobs:
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

    def _slots(self, device):
        return self.execution.gpu_slots if device == "cuda" else self.execution.cpu_workers

    def admits(self, resources):
        execution, running = self.execution, self.running
        if len(running[resources.device]) >= self._slots(resources.device):
            return False
        busy = [item for items in running.values() for item in items]
        if (
            resources.device == "cuda"
            and execution.vram_budget_bytes is not None
            and sum(r.vram_bytes for r in running["cuda"]) + resources.vram_bytes
            > execution.vram_budget_bytes
        ):
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
    """Variables de la tubería y de MPS que hereda cada proceso de un trabajo."""
    values = {} if execution.pipeline is None else execution.pipeline.environment()
    if execution.mps:
        directory = execution.mps_pipe_directory or os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
        _require(directory, "MPS necesita CUDA_MPS_PIPE_DIRECTORY o mps_pipe_directory")
        values["CUDA_MPS_PIPE_DIRECTORY"] = directory
    return values
