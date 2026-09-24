"""Admisión de una única carga científica CUDA, con márgenes de RAM y VRAM."""

import csv
import fcntl
import io
import os
import resource
import subprocess
from pathlib import Path

GIB = 1024**3


def check_host_memory(max_ram=12 * GIB):
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > max_ram:
        raise MemoryError("El proceso supera el presupuesto de RAM")


class GpuLease:
    def __init__(self, *, max_ram=12 * GIB, max_vram=6 * GIB, reserve_vram=GIB):
        if not 0 < max_ram <= 12 * GIB or not 0 < max_vram <= 6 * GIB or reserve_vram < GIB // 2:
            raise ValueError("El presupuesto excede los límites experimentales")
        self.max_ram, self.max_vram, self.reserve = max_ram, max_vram, reserve_vram
        self.handle = None
        self.record = None
        self.original_setter = None

    def __enter__(self):
        import torch

        runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/mars-titan-{os.getuid()}"))
        runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.handle = (runtime / "mars-titan-scientific-gpu.lock").open("a")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            processes = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA no está disponible. No se cambia a CPU")
            for pid, _name in csv.reader(io.StringIO(processes.stdout), skipinitialspace=True):
                if int(pid) != os.getpid():
                    raise RuntimeError("Hay otra carga de cómputo activa en CUDA")
            free, total = torch.cuda.mem_get_info(0)
            budget = min(self.max_vram, free - self.reserve)
            if budget < 256 * 1024**2:
                raise RuntimeError("No hay VRAM libre suficiente para conservar el margen")
            self.original_setter = torch.cuda.set_per_process_memory_fraction
            limit = budget / total

            def bounded_fraction(fraction, device=None):
                nonlocal limit
                if device not in (None, 0, "cuda:0", torch.device("cuda:0")):
                    raise ValueError("La carga científica solo admite cuda:0")
                limit = min(limit, fraction)
                self.original_setter(limit, device)

            # Los consumidores anteriores no pueden ampliar el presupuesto de esta ejecución.
            torch.cuda.set_per_process_memory_fraction = bounded_fraction
            bounded_fraction(limit, 0)
            self.record = dict(
                device="cuda:0",
                gpu=torch.cuda.get_device_name(0),
                torch=str(torch.__version__),
                cuda=torch.version.cuda,
                free_vram_bytes=free,
                max_vram_bytes=budget,
                max_ram_bytes=self.max_ram,
                reserved_margin_bytes=self.reserve,
            )
            self.check()
            return self
        except BaseException:
            self.__exit__()
            raise

    def check(self):
        import torch

        check_host_memory(self.max_ram)
        if self.record and torch.cuda.memory_allocated(0) > self.record["max_vram_bytes"]:
            raise MemoryError("El proceso supera el presupuesto de VRAM")

    def __exit__(self, *_args):
        if self.original_setter:
            import torch

            torch.cuda.set_per_process_memory_fraction = self.original_setter
            self.original_setter = None
        if self.handle:
            self.handle.close()
            self.handle = None
