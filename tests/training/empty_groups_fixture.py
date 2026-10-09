"""Grupos Parquet sin filas en muestras técnicas, como los de la edición histórica v3."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mars_titan.data.storage import sha256


def regroup(path, size, empty=()):
    """Reescribir con grupos de `size` filas y un grupo vacío antes de cada índice de `empty`.

    Un índice igual al número de grupos con filas añade el grupo vacío al final.
    """
    table = pq.read_table(path)
    chunks = [table.slice(start, size) for start in range(0, table.num_rows, size)]
    with pq.ParquetWriter(path, table.schema) as writer:
        for index in range(len(chunks) + 1):
            for _ in range(list(empty).count(index)):
                writer.write_table(table.slice(0, 0))
            if index < len(chunks):
                writer.write_table(chunks[index])
    metadata = pq.ParquetFile(path).metadata
    return [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]


def regroup_samples(manifest, size, layout=None):
    """Reagrupar las muestras de todos los activos y actualizar sus huellas.

    `layout` recibe el número de grupos con filas y devuelve dónde insertar grupos vacíos.
    Sin `layout`, el archivo queda compacto con los mismos grupos con filas.
    """
    manifest = Path(manifest)
    meta = json.loads(manifest.read_text())
    sizes = {}
    for asset in meta["assets"]:
        folder = Path(meta["roots"]["samples"]) / asset["market"] / asset["symbol"]
        path = folder / "samples.parquet"
        groups = -(-pq.ParquetFile(path).metadata.num_rows // size)
        sizes[f"{asset['market']}/{asset['symbol']}"] = regroup(
            path, size, () if layout is None else layout(groups)
        )
        asset["samples_sha256"] = sha256(path)
    manifest.write_text(json.dumps(meta))
    return sizes


def final_empty(groups):
    """El caso real: un único grupo vacío después del último grupo con filas."""
    return [groups]


def inner_empty(groups):
    """Grupos vacíos intermedios y final, que desplazan los índices físicos posteriores."""
    return [1, 1, groups]


def canonical(value):
    """Forma comparable de lotes y cursores, sin la huella propia de cada manifiesto."""
    if isinstance(value, np.ndarray):
        return value.dtype.str, value.shape, value.tobytes()
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items() if k != "manifest_sha256"}
    if isinstance(value, list | tuple):
        return [canonical(v) for v in value]
    return value
