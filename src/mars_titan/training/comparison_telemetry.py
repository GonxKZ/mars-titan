"""Coste observado de un intento de la campaña, sin muestreo en el recorrido de cálculo."""

import importlib.metadata
import platform
import resource
import sys
import time


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
    )
    if device == "cpu":
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


class AttemptMetrics:
    """Tiempo de este intento y pico de memoria de su proceso independiente.

    La CPU suma tiempo de sus hilos y puede superar el tiempo de pared. RSS es el pico de
    toda la vida del proceso, incluida la importación. Cada trabajo de la cola tiene su
    proceso. No se mide energía ni se deduce de muestras aisladas de potencia.
    """

    def __init__(self):
        self.started = time.perf_counter()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self.cpu_started = usage.ru_utime + usage.ru_stime

    def finish(self, device):
        """Recoger los contadores sin iniciar CUDA ni sincronizar kernels adicionales."""
        if device not in ("cpu", "cuda:0"):
            raise ValueError("El dispositivo debe ser cpu o cuda:0 explícitos")
        wall = time.perf_counter() - self.started
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux expresa ru_maxrss en KiB. El motor usa Linux, también dentro de WSL2.
        return dict(
            wall_seconds=wall,
            cpu_seconds=usage.ru_utime + usage.ru_stime - self.cpu_started,
            peak_rss_bytes=int(usage.ru_maxrss) * 1024,
            rss_scope="process_lifetime",
            device=device,
            platform=dict(system=platform.system(), machine=platform.machine()),
            versions=_versions(),
            energy_joules=None,
            energy_reason="No se dispone de una medida integrada de energía",
            **_cuda(device),
        )
