"""Perfiles de hardware declarados y su comprobación contra la plataforma detectada.

Cada máquina admitida tiene un perfil en `configs/hardware/<nombre>.json`. El perfil se elige
por su nombre, en la declaración de ejecución de la campaña o en la orden de comprobación, y
nunca se deduce de lo detectado. Antes de empezar se compara con la identidad de la
plataforma y cualquier diferencia detiene la ejecución.

El perfil fija también los límites de MARS-TITAN en esa máquina. Con memoria dedicada hay un
límite para la RAM del anfitrión y otro para la VRAM. Con memoria unificada, como en la GB10,
un único límite cubre las dos, porque la GPU reserva de la misma memoria que el anfitrión y
allí corre además una aplicación de producción. En ese caso el proceso debe correr dentro de
un cgroup cuyo `memory.max` no supere el límite, para que sea el sistema quien lo haga
cumplir. Que las reservas del controlador de la GPU cuenten en ese cgroup está pendiente de
comprobar en la máquina real, así que el presupuesto de VRAM de la declaración de ejecución
también se suma dentro del mismo límite.
"""

import os
import re
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest

from .platform_identity import read_identity

KIND = "mars_titan_hardware_profile"
# Perfil elegido para las órdenes que no tienen una declaración de ejecución propia.
PROFILE_ENV = "MARS_TITAN_HARDWARE_PROFILE"
PROFILES = Path(__file__).resolve().parents[3] / "configs" / "hardware"
MIB = 1024**2
_FIELDS = {"schema_version", "kind", "name", "machine", "gpu", "memory", "cpu_threads", "notes"}
_GPU = {"name", "compute_capability"}
_MEMORY = {
    "dedicated": {"model", "host_mib", "device_mib"},
    "unified": {"model", "total_mib"},
}
_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive(value):
    return type(value) is int and value > 0


def load_profile(name, *, folder=PROFILES):
    """Perfil declarado con ese nombre, validado y con la huella de su archivo."""
    _require(isinstance(name, str) and _NAME.fullmatch(name), "El nombre del perfil no es válido")
    path = Path(folder) / f"{name}.json"
    _require(path.is_file(), f"No existe el perfil de hardware {name}")
    profile, digest = read_manifest(path, 64 * 1024)
    _require(isinstance(profile, dict), f"El perfil de hardware {name} no es un objeto")
    gpu, memory = profile.get("gpu"), profile.get("memory")
    _require(
        set(profile) - {"notes"} == _FIELDS - {"notes"}
        and profile["schema_version"] == 1
        and profile["kind"] == KIND
        and profile["name"] == name
        and profile["machine"] in ("x86_64", "aarch64")
        and isinstance(gpu, dict)
        and set(gpu) == _GPU
        and isinstance(gpu["name"], str)
        and isinstance(gpu["compute_capability"], list)
        and len(gpu["compute_capability"]) == 2
        and all(type(v) is int and v >= 0 for v in gpu["compute_capability"])
        and isinstance(memory, dict)
        and set(memory) == _MEMORY.get(memory.get("model"), set())
        and all(_positive(memory[key]) for key in memory if key != "model")
        and _positive(profile["cpu_threads"]),
        f"El perfil de hardware {name} no cumple su contrato",
    )
    return dict(profile, path=str(path.resolve()), sha256=digest)


def declared_profile(environ=os.environ):
    """Perfil que nombra MARS_TITAN_HARDWARE_PROFILE. Sin él, la orden no empieza."""
    name = environ.get(PROFILE_ENV)
    _require(name, f"Declara el perfil de hardware en {PROFILE_ENV}, por ejemplo rtx4070-laptop")
    return load_profile(name)


def hardware_mismatches(profile, identity):
    """Diferencias entre la máquina de un perfil y una identidad de plataforma."""
    identity = read_identity(identity)
    found = []
    if identity["emulated"]:
        found.append("la plataforma está emulada")
    if identity["machine"] != profile["machine"]:
        found.append(f"arquitectura {identity['machine']} en lugar de {profile['machine']}")
    expected = profile["gpu"]
    if not identity["gpus"]:
        found.append("no hay ninguna GPU NVIDIA")
    memory = profile["memory"]
    for gpu in identity["gpus"]:
        if gpu["name"] != expected["name"]:
            found.append(f"GPU {gpu['name']} en lugar de {expected['name']}")
        if gpu["compute_capability"] != expected["compute_capability"]:
            found.append(
                f"capacidad {gpu['compute_capability']} en lugar de "
                f"{expected['compute_capability']}"
            )
        if memory["model"] == "dedicated" and (
            gpu["memory_total_mib"] is None or gpu["memory_total_mib"] < memory["device_mib"]
        ):
            found.append(f"la GPU no tiene los {memory['device_mib']} MiB propios del perfil")
    host_mib = memory["host_mib"] if memory["model"] == "dedicated" else memory["total_mib"]
    if identity["memory_total_bytes"] < host_mib * MIB:
        found.append(f"la máquina tiene menos de {host_mib} MiB de memoria")
    if identity["cpu"]["logical"] < profile["cpu_threads"]:
        found.append(f"la máquina tiene menos de {profile['cpu_threads']} hilos de CPU")
    return found


def cgroup_memory_limit(cgroup=Path("/proc/self/cgroup"), root=Path("/sys/fs/cgroup")):
    """Límite de memoria más estricto del cgroup v2 del proceso y sus padres, o None."""
    lines = [line for line in cgroup.read_text().splitlines() if line.startswith("0::")]
    if len(lines) != 1:
        return None
    limits = []
    folder = root / lines[0][3:].lstrip("/")
    for current in (folder, *folder.parents):
        limit = current / "memory.max"
        if limit.is_file() and (value := limit.read_text().strip()) != "max":
            limits.append(int(value))
        if current == root:
            break
    return min(limits, default=None)


def check_profile(profile, identity, *, memory_limit=None):
    """Detener la ejecución si la plataforma no es la del perfil o no respeta su límite."""
    found = hardware_mismatches(profile, identity)
    memory = profile["memory"]
    if memory["model"] == "unified":
        limit = (memory_limit or cgroup_memory_limit)()
        if limit is None or limit > memory["total_mib"] * MIB:
            found.append(
                f"la memoria unificada exige un cgroup con memory.max de {memory['total_mib']} "
                "MiB como mucho"
            )
    _require(
        not found, f"La plataforma no corresponde al perfil {profile['name']}: {'; '.join(found)}"
    )


def check_budgets(profile, *, vram_bytes, host_bytes, threads):
    """Exigir que los presupuestos de una declaración de ejecución quepan en el perfil."""
    memory = profile["memory"]
    vram_bytes = vram_bytes or 0
    host_bytes = host_bytes or 0
    if memory["model"] == "dedicated":
        _require(
            vram_bytes <= memory["device_mib"] * MIB and host_bytes <= memory["host_mib"] * MIB,
            f"Los presupuestos superan los {memory['device_mib']} MiB de VRAM o los "
            f"{memory['host_mib']} MiB de RAM del perfil {profile['name']}",
        )
    else:
        _require(
            vram_bytes + host_bytes <= memory["total_mib"] * MIB,
            f"VRAM y RAM superan juntas los {memory['total_mib']} MiB de memoria unificada "
            f"del perfil {profile['name']}",
        )
    _require(
        threads <= profile["cpu_threads"],
        f"La ejecución usa {threads} hilos y el perfil {profile['name']} admite "
        f"{profile['cpu_threads']}",
    )


def same_platform(records, *, unrecorded=None):
    """Plataforma común de las ejecuciones que entran en una comparación.

    `records` asocia cada ejecución con su identidad, o con None si se hizo antes de que se
    registrara. Todas las registradas deben compartir huella. Las anteriores solo se admiten
    con una regla declarada, `unrecorded`, el perfil de hardware al que se atribuyen, y
    entonces cada identidad registrada debe corresponder también a ese perfil.
    """
    _require(records, "No hay ejecuciones que comparar")
    recorded = {label: read_identity(r) for label, r in records.items() if r is not None}
    missing = sorted(label for label, record in records.items() if record is None)
    digests = {record["platform_sha256"] for record in recorded.values()}
    _require(
        len(digests) <= 1,
        "La comparación mezcla plataformas distintas: "
        + ", ".join(sorted(f"{k}={r['platform_sha256'][:12]}" for k, r in recorded.items())),
    )
    if missing:
        _require(
            unrecorded is not None,
            f"{len(missing)} ejecuciones no registran su plataforma, la primera {missing[0]}. "
            "Declara el perfil de hardware al que se atribuyen para admitirlas",
        )
        for label, record in recorded.items():
            found = hardware_mismatches(unrecorded, record)
            _require(
                not found,
                f"{label} no corresponde al perfil {unrecorded['name']} atribuido a las "
                f"ejecuciones sin plataforma: {'; '.join(found)}",
            )
    return dict(
        platform_sha256=next(iter(digests), None),
        recorded=len(recorded),
        unrecorded=len(missing),
        unrecorded_profile=unrecorded["name"] if missing else None,
    )
