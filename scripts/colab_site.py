#!/usr/bin/env python3
"""Ficha del equipo que ejecuta una campaña: máquina, GPU, controlador y bibliotecas.

Un modelo ajustado en otra máquina no es comparable bit a bit con uno de la DGX. Esta ficha
deja escrito dónde se ajustó cada estado de campaña, para el informe y para el artículo, y
permite ver de un vistazo en qué difieren dos equipos antes de comparar sus resultados.

Uso:
    colab_site.py --output equipo.json [--disk /content]
    colab_site.py --compare dgx.json colab.json

La ficha no lleva nombres de usuario, rutas personales ni el nombre de la máquina.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PACKAGES = ("torch", "triton", "numpy", "pyarrow", "scipy", "lightgbm")
SITE_ENV = "MARS_TITAN_SITE"
# Lo que decide si dos equipos pueden dar los mismos bits. El resto es informativo.
NUMERIC_FIELDS = (
    ("platform", "machine"),
    ("gpu", "name"),
    ("gpu", "capability"),
    ("gpu", "driver"),
    ("torch", "cuda_runtime"),
    ("torch", "cudnn"),
    ("packages", "torch"),
    ("packages", "triton"),
    ("packages", "numpy"),
    ("python",),
)


def _first_line(path, prefix):
    try:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if line.startswith(prefix):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _gpu(run=subprocess.run):
    """Nombre, memoria y controlador de la GPU 0 según nvidia-smi, o None si no hay."""
    try:
        result = run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 3 or not fields[0] or not fields[2]:
        return None
    # Una GPU de memoria unificada, como la GB10, no declara memoria propia.
    memory = int(fields[1]) if fields[1].isdecimal() else None
    return dict(name=fields[0], memory_mib=memory, driver=fields[2], capability=None)


def _torch():
    """Lo que PyTorch dice de su compilación, sin iniciar CUDA si no está disponible."""
    torch = sys.modules.get("torch")
    if torch is None:
        try:
            import torch
        except ImportError:
            return None, None
    details = dict(
        cuda_runtime=torch.version.cuda,
        cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        cuda_available=bool(torch.cuda.is_available()),
        matmul_precision=torch.get_float32_matmul_precision(),
        allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
    )
    capability = None
    if details["cuda_available"]:
        capability = [int(value) for value in torch.cuda.get_device_capability(0)]
    return details, capability


def collect(*, disk=None, site=None, run=subprocess.run, with_torch=True):
    """Ficha del equipo actual."""
    memory = _first_line("/proc/meminfo", "MemTotal")
    document = dict(
        schema_version=1,
        kind="machine_site_v1",
        site=site if site is not None else os.environ.get(SITE_ENV),
        collected_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        platform=dict(
            system=platform.system(), machine=platform.machine(), release=platform.release()
        ),
        cpu=dict(model=_first_line("/proc/cpuinfo", "model name"), logical=os.cpu_count()),
        memory_kib=int(memory.split()[0]) if memory and memory.split()[0].isdecimal() else None,
        python=platform.python_version(),
        packages={},
        gpu=_gpu(run),
        torch=None,
        environment={
            name: os.environ.get(name)
            for name in ("CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")
        },
    )
    for package in PACKAGES:
        try:
            document["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            document["packages"][package] = None
    if with_torch:
        details, capability = _torch()
        document["torch"] = details
        if document["gpu"] is not None:
            document["gpu"]["capability"] = capability
    if disk is not None:
        usage = shutil.disk_usage(disk)
        document["disk"] = dict(total_bytes=usage.total, free_bytes=usage.free)
    return document


def _get(document, path):
    value = document
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def differences(first, second):
    """Campos numéricamente relevantes en los que dos equipos no coinciden."""
    result = []
    for path in NUMERIC_FIELDS:
        left, right = _get(first, path), _get(second, path)
        if left != right:
            result.append(dict(field=".".join(path), first=left, second=right))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--output", type=Path, help="Fichero donde guardar la ficha")
    parser.add_argument("--disk", type=Path, help="Carpeta cuyo disco se anota")
    parser.add_argument("--site", help=f"Etiqueta del equipo (por defecto, {SITE_ENV})")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("A", "B"))
    args = parser.parse_args(argv)
    if args.compare:
        first, second = (json.loads(path.read_text(encoding="utf-8")) for path in args.compare)
        rows = differences(first, second)
        for row in rows:
            print(f"{row['field']}: {row['first']!r} frente a {row['second']!r}")
        print(
            "Los dos equipos coinciden en lo que decide los bits."
            if not rows
            else f"{len(rows)} diferencias: los resultados no son comparables bit a bit."
        )
        return 0
    document = collect(disk=args.disk, site=args.site)
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
