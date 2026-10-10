"""Lectura acotada de archivos regulares para el observatorio.

El servidor en directo y el recolector leen archivos que escriben otros procesos. Ninguno
sigue enlaces, sale de su raíz ni lee más bytes de los permitidos.
"""

import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path


def utc_text(timestamp):
    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")


def open_regular(root, relative, maximum):
    """Abrir un archivo regular dentro de `root` sin seguir enlaces y con tamaño acotado."""
    root = Path(root).resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise PermissionError(relative)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise PermissionError(relative)
        return os.fdopen(fd, "rb"), info
    except BaseException:
        os.close(fd)
        raise


def read_json(root, relative, maximum):
    stream, info = open_regular(root, relative, maximum)
    with stream:
        body = stream.read(maximum + 1)
    if len(body) > maximum:
        raise ValueError("El archivo ha crecido durante la lectura")
    return json.loads(body), info


def signature(path):
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError:
        return None
    return info.st_ino, info.st_size, info.st_mtime_ns
