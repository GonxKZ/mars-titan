"""Coste observado de un intento de la campaña, sin muestreo en el recorrido de cálculo."""

import importlib.metadata
import os
import platform
import re
import resource
import subprocess
import sys
import threading
import time

SITE_ENV = "MARS_TITAN_SITE"
_SITE = re.compile(r"[a-z0-9][a-z0-9.-]{0,63}")
_DRIVER = re.compile(r"[0-9A-Za-z._-]{1,64}")


def _site():
    """Etiqueta del equipo que ejecuta la campaña, declarada por quien la lanza.

    Distingue en los resultados dónde se ajustó cada modelo (por ejemplo, la GPU de Colab
    frente a la DGX). Sin la variable no hay etiqueta. Una etiqueta mal escrita detiene el
    intento al empezar, antes de gastar cómputo en un resultado sin procedencia.
    """
    label = os.environ.get(SITE_ENV)
    if label is None:
        return None
    if _SITE.fullmatch(label) is None:
        raise ValueError(
            f"{SITE_ENV} es una etiqueta de 1 a 64 caracteres: minúsculas, cifras, puntos y guiones"
        )
    return label


def _gpu_driver():
    """Versión del controlador de la GPU 0, o None si no se puede leer."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.split()
    return lines[0] if len(lines) == 1 and _DRIVER.fullmatch(lines[0]) else None


def _optional(read):
    try:
        return read()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _versions():
    versions = {"python": platform.python_version()}
    for package in ("torch", "numpy", "pyarrow", "lightgbm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _cuda(device):
    result = dict(
        cuda_allocated_peak_bytes=None,
        cuda_reserved_peak_bytes=None,
        cuda_scope="pytorch_allocator",
        gpu_name=None,
        cuda_reason="El intento no utiliza CUDA",
        # Procedencia del acelerador: arquitectura, memoria, controlador y bibliotecas.
        gpu_capability=None,
        gpu_total_bytes=None,
        gpu_driver=None,
        cuda_runtime=None,
        cudnn=None,
    )
    if device == "cpu":
        return result
    if device.startswith("opencl:"):
        result["cuda_reason"] = "OpenCL no utiliza el asignador CUDA de PyTorch"
        return result
    torch = sys.modules.get("torch")
    result["cuda_reason"] = "CUDA no se inició en este proceso"
    try:
        if torch is not None and torch.cuda.is_initialized():
            # Son picos del asignador de PyTorch, no de toda la memoria del controlador.
            allocated = int(torch.cuda.max_memory_allocated(device))
            reserved = int(torch.cuda.max_memory_reserved(device))
            name = torch.cuda.get_device_name(device)
            result.update(
                cuda_allocated_peak_bytes=allocated,
                cuda_reserved_peak_bytes=reserved,
                gpu_name=name,
                cuda_reason=None,
            )
    except (AttributeError, RuntimeError) as error:
        result["cuda_reason"] = f"{type(error).__name__}: {str(error)[:300]}"
        return result
    if result["cuda_reason"] is None:
        # Cada dato se lee por separado: uno desconocido no borra los demás ni los picos.
        result.update(
            gpu_capability=_optional(
                lambda: [int(v) for v in torch.cuda.get_device_capability(device)]
            ),
            gpu_total_bytes=_optional(
                lambda: int(torch.cuda.get_device_properties(device).total_memory)
            ),
            gpu_driver=_gpu_driver(),
            cuda_runtime=_optional(lambda: torch.version.cuda),
            cudnn=_optional(lambda: torch.backends.cudnn.version()),
        )
    return result


class AttemptMetrics:
    """Tiempo de este intento y pico de memoria de su proceso independiente.

    La CPU suma tiempo de sus hilos y puede superar el tiempo de pared. RSS es el pico de
    toda la vida del proceso, incluida la importación. Cada trabajo de la cola tiene su
    proceso. No se mide energía ni se deduce de muestras aisladas de potencia.
    """

    def __init__(self):
        self.site = _site()
        self.started = time.perf_counter()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self.cpu_started = usage.ru_utime + usage.ru_stime

    def finish(self, device):
        """Recoger los contadores sin iniciar CUDA ni sincronizar kernels adicionales."""
        if device not in ("cpu", "cuda:0", "opencl:0"):
            raise ValueError("El dispositivo debe ser cpu, cuda:0 u opencl:0 explícitos")
        wall = time.perf_counter() - self.started
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux expresa ru_maxrss en KiB. El motor usa Linux, también dentro de WSL2.
        return dict(
            wall_seconds=wall,
            cpu_seconds=usage.ru_utime + usage.ru_stime - self.cpu_started,
            peak_rss_bytes=int(usage.ru_maxrss) * 1024,
            rss_scope="process_lifetime",
            device=device,
            site=self.site,
            platform=dict(system=platform.system(), machine=platform.machine()),
            versions=_versions(),
            energy_joules=None,
            energy_reason="No se dispone de una medida integrada de energía",
            **_cuda(device),
        )


def _process_gpu_bytes(pid):
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=2,
        check=True,
    )
    if len(result.stdout) > 65536:
        raise ValueError("La lista de procesos GPU supera el límite")
    for line in result.stdout.splitlines():
        process, value = (part.strip() for part in line.split(","))
        if process == str(pid) and value.isdecimal():
            return int(value) * 1024**2
    return None


class ProcessGpuMemory:
    """Máximo muestreado del proceso en la GPU 0, con espacio constante y fuera del cálculo.

    El contador incluye OpenCL y el contexto del controlador. No es una medida continua ni
    sustituye el contador de asignaciones de PyTorch. La energía sigue sin medirse.
    """

    def __init__(self):
        self.pid, self.peak, self.samples = os.getpid(), None, 0
        self.error, self.thread = None, None
        self.stop = threading.Event()

    def _sample(self):
        try:
            value = _process_gpu_bytes(self.pid)
            if value is not None:
                self.peak = value if self.peak is None else max(value, self.peak)
                self.samples += 1
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.error = f"{type(error).__name__}: {str(error)[:200]}"

    def _work(self):
        while not self.stop.is_set():
            self._sample()
            self.stop.wait(1)

    def start(self):
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()
        return self

    def finish(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(3)
        return dict(
            gpu_memory_peak_sampled_bytes=self.peak,
            gpu_memory_samples=self.samples,
            gpu_memory_period_seconds=1,
            gpu_memory_scope="nvidia_smi_process_sampled",
            gpu_memory_reason=None
            if self.samples
            else self.error or "No hubo muestras disponibles",
        )
