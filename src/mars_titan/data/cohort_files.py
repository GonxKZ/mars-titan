"""Lecturas de manifiestos ligadas a sus bytes y destinos sin enlaces intermedios."""

import hashlib
import json


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("El manifiesto contiene una clave duplicada")
        result[key] = value
    return result


def read_manifest(path, maximum=64 * 1024**2):
    if path.is_symlink() or not path.is_file():
        raise ValueError("El manifiesto no es un archivo regular")
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError("El manifiesto supera su presupuesto")
    return json.loads(payload, object_pairs_hook=_unique), hashlib.sha256(payload).hexdigest()


def safe_destination(path):
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("El destino incluye un enlace intermedio")
