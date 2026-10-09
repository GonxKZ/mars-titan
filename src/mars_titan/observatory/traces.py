"""Paquetes de trazas de aprendizaje para el observatorio.

Las trazas de #448 pueden tener millones de puntos. Copiarlas como listas JSON obligaría
al navegador a analizar decenas de megabytes de texto, así que cada paquete separa un
manifiesto JSON pequeño de un bloque binario con columnas little-endian. El navegador
crea vistas tipadas sobre ese bloque sin copiarlo.

El bloque se nombra por su huella y el manifiesto se sustituye después de escribirlo.
Un lector que llegue entre las dos escrituras ve el paquete anterior completo o el
nuevo completo. NaN codifica una observación ausente y nunca se interpola.
"""

import hashlib
import json
import math
import os
import re
import sys
import tempfile
from array import array
from pathlib import Path

from mars_titan.data.storage import atomic_json

SCHEMA_VERSION = 1
KIND = "mars_titan_learning_traces"
GROUPS = {"optimization", "titans", "episodic", "rl", "session", "adapters", "modalities"}
X_UNITS = {"optimizer_step", "epoch", "session"}
PROVENANCES = {"measured", "fixture"}
NAME = re.compile(r"[A-Za-z0-9][\w.-]{0,80}")
IDENTIFIER = re.compile(r"[a-z][a-z0-9_.]{0,95}")
MAX_BUNDLE_BYTES = 256 * 1024**2


def _column(values, code):
    column = array(code, values)
    if sys.byteorder == "big":
        column.byteswap()
    return column.tobytes()


def _increasing(x, label):
    previous = -math.inf
    for value in x:
        if not math.isfinite(value) or value <= previous:
            raise ValueError(f"{label}: el eje x debe ser finito y estrictamente creciente")
        previous = value


class _Blob:
    """Bloque binario con desplazamientos alineados a 8 bytes para Float64Array."""

    def __init__(self):
        self.parts, self.size = [], 0

    def add(self, payload, dtype, length):
        reference = dict(offset=self.size, length=length, dtype=dtype)
        padding = -len(payload) % 8
        self.parts.append(payload + b"\0" * padding)
        self.size += len(payload) + padding
        return reference


def _text(value, label, maximum=120):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label}: texto ausente o demasiado largo")
    return value


def write_trace_bundle(
    folder,
    name,
    *,
    run_id,
    attempt_id,
    model_id,
    provenance,
    x_unit,
    series=(),
    matrices=(),
    cadence=None,
    max_bytes=MAX_BUNDLE_BYTES,
):
    """Escribir un paquete y registrarlo en `index.json`.

    Cada serie es un diccionario con `id`, `group`, `label`, `unit`, `x` e `y`. Cada matriz
    añade `rows` y `values` en orden de filas, con una fila por capa o componente.
    """
    folder = Path(folder)
    if not NAME.fullmatch(name):
        raise ValueError("Nombre de paquete no admitido")
    if provenance not in PROVENANCES or x_unit not in X_UNITS:
        raise ValueError("Procedencia o unidad del eje no admitidas")
    blob, manifest_series, manifest_matrices, seen = _Blob(), [], [], set()

    def header(item, label):
        identifier = item.get("id")
        if (
            not isinstance(identifier, str)
            or not IDENTIFIER.fullmatch(identifier)
            or identifier in seen
        ):
            raise ValueError(f"{label}: identificador ausente, repetido o no admitido")
        seen.add(identifier)
        if item.get("group") not in GROUPS:
            raise ValueError(f"{label}: grupo no admitido")
        return dict(
            id=identifier,
            group=item["group"],
            label=_text(item.get("label"), label),
            unit=_text(item.get("unit", "sin unidad"), label, 40),
        )

    for index, item in enumerate(series):
        label = f"series[{index}]"
        x, y = [float(v) for v in item["x"]], [float(v) for v in item["y"]]
        _increasing(x, label)
        if len(x) != len(y) or not x:
            raise ValueError(f"{label}: x e y necesitan la misma longitud no nula")
        if any(math.isinf(v) for v in y):
            raise ValueError(f"{label}: los valores infinitos no son observaciones")
        manifest_series.append(
            dict(
                header(item, label),
                x=blob.add(_column(x, "d"), "f64", len(x)),
                y=blob.add(_column(y, "f"), "f32", len(y)),
            )
        )
    for index, item in enumerate(matrices):
        label = f"matrices[{index}]"
        x = [float(v) for v in item["x"]]
        rows = [_text(row, label, 40) for row in item["rows"]]
        values = [float(v) for v in item["values"]]
        _increasing(x, label)
        if not rows or not x or len(values) != len(rows) * len(x):
            raise ValueError(f"{label}: la matriz necesita filas × puntos valores")
        if any(math.isinf(v) for v in values):
            raise ValueError(f"{label}: los valores infinitos no son observaciones")
        scale = item.get("range")
        if scale is not None and (
            len(scale) != 2 or not all(map(math.isfinite, scale)) or scale[0] >= scale[1]
        ):
            raise ValueError(f"{label}: rango de color no válido")
        manifest_matrices.append(
            dict(
                header(item, label),
                rows=rows,
                range=list(scale) if scale is not None else None,
                x=blob.add(_column(x, "d"), "f64", len(x)),
                values=blob.add(_column(values, "f"), "f32", len(values)),
            )
        )
    if blob.size > max_bytes:
        raise ValueError("El paquete supera su presupuesto de bytes")
    body = b"".join(blob.parts)
    digest = hashlib.sha256(body).hexdigest()
    blob_name = f"{name}-{digest[:16]}.bin"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / f"{name}.json"
    previous = None
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text()).get("blob")
    if not (folder / blob_name).is_file():
        fd, temporary = tempfile.mkstemp(prefix=f".{blob_name}.", dir=folder)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, folder / blob_name)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    manifest = dict(
        schema_version=SCHEMA_VERSION,
        kind=KIND,
        name=name,
        run_id=_text(run_id, "run_id", 96),
        attempt_id=_text(attempt_id, "attempt_id", 96),
        model_id=_text(model_id, "model_id", 96),
        provenance=provenance,
        x_unit=x_unit,
        cadence=cadence,
        blob=blob_name,
        blob_bytes=len(body),
        blob_sha256=digest,
        series=manifest_series,
        matrices=manifest_matrices,
    )
    atomic_json(manifest_path, manifest)
    index_path = folder / "index.json"
    index = (
        json.loads(index_path.read_text())
        if index_path.is_file()
        else dict(schema_version=1, bundles=[])
    )
    entries = {entry["name"]: entry for entry in index["bundles"]}
    entries[name] = dict(
        name=name,
        manifest=manifest_path.name,
        run_id=run_id,
        attempt_id=attempt_id,
        model_id=model_id,
        provenance=provenance,
    )
    atomic_json(
        index_path, dict(schema_version=1, bundles=[entries[key] for key in sorted(entries)])
    )
    if previous and previous != blob_name and NAME.fullmatch(previous.removesuffix(".bin")):
        (folder / previous).unlink(missing_ok=True)
    return manifest
