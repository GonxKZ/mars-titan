"""Identidad de la plataforma en la que corre una ejecución de MARS-TITAN.

Registra lo que distingue una máquina de otra al comparar resultados: sistema, arquitectura,
procesador, memoria, PyTorch y cada GPU NVIDIA. Usa lo que ya resuelven las herramientas
existentes, sin detección propia: `platform`, `lscpu` de util-linux, `/proc/meminfo`,
`nvidia-smi` y PyTorch. La capacidad CPU es la que elige PyTorch para sus núcleos.

Las GPU salen de `nvidia-smi` y no de `torch.cuda`, porque consultar las propiedades con
PyTorch crea un contexto CUDA en el proceso. La campaña cuenta ese contexto en su VRAM solo
cuando ejecuta su primer trabajo CUDA, así que leer la identidad no debe crearlo.

La parte `comparable` define una misma plataforma a efectos de comparar resultados, y su
huella es `platform_sha256`. El controlador NVIDIA, el núcleo, la memoria total y el
identificador físico de cada GPU se registran pero no entran en la huella. Cambian con una
actualización o con otro ejemplar del mismo modelo sin cambiar las bibliotecas numéricas del
lock ni los núcleos que se ejecutan.

Bajo qemu-user el intérprete informa de la arquitectura emulada, mientras que `lscpu` es un
programa del anfitrión y describe el procesador real. Si no coinciden, la plataforma queda
marcada como emulada y ningún perfil de hardware la admite.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

KIND = "mars_titan_platform"
SCHEMA_VERSION = 1
# Valor por defecto que pide detectar PyTorch, distinto de None, que indica su ausencia.
_DETECT = object()
_GPU_FIELDS = ("name", "compute_cap", "memory.total", "driver_version", "uuid", "pci.bus_id")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _lscpu_fields(entries):
    # lscpu --json anida los núcleos de cada fabricante en `children`, como los dos tipos
    # de núcleo Arm de la GB10.
    for entry in entries:
        yield entry["field"].rstrip(":"), entry.get("data")
        yield from _lscpu_fields(entry.get("children", ()))


def cpu_record(*, run=subprocess.run, torch=None):
    """Arquitectura y modelos del procesador según `lscpu`, más la capacidad de PyTorch."""
    # Con LC_ALL=C los nombres de campo no dependen del idioma del sistema.
    result = run(
        ["lscpu", "--json"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    fields = list(_lscpu_fields(json.loads(result.stdout)["lscpu"]))
    architecture = [value for name, value in fields if name == "Architecture"]
    models = [value for name, value in fields if name == "Model name"]
    _require(len(architecture) == 1 and models, "lscpu no informa de arquitectura y modelo")
    return dict(
        architecture=architecture[0],
        models=list(dict.fromkeys(models)),
        logical=os.cpu_count(),
        capability=None if torch is None else torch.backends.cpu.get_cpu_capability(),
    )


def cpu_name(*, run=subprocess.run):
    """Modelos del procesador en una línea, como los registran los informes de medida.

    Sustituye a leer `model name` de /proc/cpuinfo, que no existe en aarch64.
    """
    return " + ".join(cpu_record(run=run)["models"])


def memory_total_bytes(meminfo=Path("/proc/meminfo")):
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise ValueError("No se puede leer MemTotal")


def _optional(value):
    # nvidia-smi escribe [N/A] en los campos que una GPU no tiene, como la memoria propia
    # de una GPU con memoria unificada.
    return None if value.startswith("[") else value


def gpu_records(*, run=subprocess.run, which=shutil.which):
    """GPU NVIDIA de la máquina según `nvidia-smi`. Lista vacía si no hay controlador."""
    if which("nvidia-smi") is None:
        return []
    result = run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(_GPU_FIELDS)}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    gpus = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        _require(len(values) == len(_GPU_FIELDS), "nvidia-smi devolvió otra lista de campos")
        name, capability, memory, driver, uuid, bus = values
        major, _, minor = capability.partition(".")
        memory = _optional(memory)
        gpus.append(
            dict(
                name=name,
                compute_capability=[int(major), int(minor)],
                memory_total_mib=None if memory is None else int(memory),
                driver=_optional(driver),
                uuid=uuid,
                pci_bus_id=bus,
            )
        )
    return gpus


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def torch_record(torch):
    if torch is None:
        return None
    cuda = torch.version.cuda
    return dict(
        version=torch.__version__,
        cuda=cuda,
        # Consultar la versión de cuDNN carga la biblioteca, sin crear un contexto CUDA.
        cudnn=torch.backends.cudnn.version() if cuda else None,
    )


def comparable(record):
    """Lo que debe coincidir para comparar resultados de dos ejecuciones."""
    torch = record["torch"]
    return dict(
        system=record["system"],
        machine=record["machine"],
        emulated=record["emulated"],
        cpu_models=record["cpu"]["models"],
        cpu_capability=record["cpu"]["capability"],
        torch=None if torch is None else [torch["version"], torch["cuda"], torch["cudnn"]],
        gpus=[[gpu["name"], gpu["compute_capability"]] for gpu in record["gpus"]],
    )


def platform_sha256(record):
    payload = json.dumps(comparable(record), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def platform_identity(*, cpu=None, gpus=None, torch=_DETECT, memory=None):
    """Identidad de esta máquina. Los argumentos sustituyen detecciones en las pruebas.

    `torch=None` describe un entorno sin PyTorch.
    """
    torch = _torch() if torch is _DETECT else torch
    cpu = cpu_record(torch=torch) if cpu is None else cpu
    record = dict(
        schema_version=SCHEMA_VERSION,
        kind=KIND,
        system=platform.system(),
        machine=platform.machine(),
        kernel=platform.release(),
        python=platform.python_version(),
        emulated=cpu["architecture"] != platform.machine(),
        cpu=cpu,
        memory_total_bytes=memory_total_bytes() if memory is None else memory,
        torch=torch_record(torch),
        gpus=gpu_records() if gpus is None else gpus,
    )
    record["platform_sha256"] = platform_sha256(record)
    return record


def read_identity(record):
    """Validar una identidad registrada y comprobar que su huella corresponde a su contenido."""
    _require(
        isinstance(record, dict)
        and record.get("kind") == KIND
        and record.get("schema_version") == SCHEMA_VERSION,
        "La identidad de plataforma no cumple su contrato",
    )
    try:
        digest = platform_sha256(record)
    except (KeyError, TypeError) as error:
        raise ValueError("La identidad de plataforma está incompleta") from error
    _require(
        record.get("platform_sha256") == digest,
        "La huella de la plataforma no corresponde a su contenido",
    )
    return record
