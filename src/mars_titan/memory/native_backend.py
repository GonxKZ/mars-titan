"""Carga explícita de un único enlace nativo y comprobación de su identidad."""

import importlib.util
import os
from pathlib import Path

import torch

from mars_titan.data.storage import sha256

_loaded = None


def _signature(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def load_native(path=None):
    """Cargar el enlace compilado previamente, sin descargas ni compilación implícita."""
    global _loaded
    selected = path or os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    if not selected:
        raise FileNotFoundError("Falta la ruta del enlace episódico nativo compilado")
    selected = Path(selected).resolve(strict=True)
    if _loaded is not None:
        previous, signature, module = _loaded
        if selected != previous or _signature(selected) != signature:
            raise RuntimeError("El enlace episódico cambió. Inicia otro proceso")
        return module
    signature = _signature(selected)
    digest = sha256(selected)
    specification = importlib.util.spec_from_file_location("_episodic_native", selected)
    if specification is None or specification.loader is None:
        raise ValueError("La ruta no contiene un módulo nativo de Python")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if (
        module.abi_version != 1
        or module.torch_version != torch.__version__
        or _signature(selected) != signature
    ):
        raise ValueError("El enlace nativo no corresponde al contrato o al runtime instalado")
    module.binary_sha256 = digest
    _loaded = selected, signature, module
    return module
