"""Rutas explícitas para los archivos de la medición, fuera de la copia del código."""

import os
from pathlib import Path


def workdir():
    raw = os.environ.get("MARS_TITAN_BENCHMARK_WORK")
    if not raw or not Path(raw).is_absolute():
        raise ValueError("MARS_TITAN_BENCHMARK_WORK necesita un directorio absoluto de trabajo")
    path = Path(raw).resolve(strict=True)
    if not path.is_dir() or path.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("El directorio del ensayo debe existir fuera del repositorio")
    return path
