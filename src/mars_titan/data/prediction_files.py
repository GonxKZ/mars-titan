"""Tablas de predicciones por fila compactadas o liberadas, con lectura y huellas comprobadas.

Un ajuste escribe validación, calibración y evaluación con claves, objetivo y decimales.
Todos los ajustes de una ventana comparten las mismas filas y objetivos, así que las claves
y el objetivo se guardan una sola vez por tramo en una tabla común con dirección de
contenido. Cada ajuste conserva solo lo que no se puede reconstruir: sus decimales y, si su
orden no es el común, el índice de cada fila. La predicción puntual o el centro que repiten
los bits de la mediana y el control nulo que vale cero positivo se reconstruyen. Un decimal
de float64 cuyos valores vuelven de float32 con los mismos bits se guarda en float32.

Junto a las tablas, ``predictions-retention.json`` registra para cada archivo su huella
original, su número de filas, la huella de su contenido bit a bit en el orden escrito y en
el orden común y su estado:

- ``compacted``: el archivo original se sustituyó por sus decimales y la tabla común.
  ``read`` devuelve exactamente la tabla original, con su esquema, su orden y sus bits.
- ``released``: no queda ninguna tabla. La huella de contenido permite comprobar una
  regeneración desde el estado conservado del ajuste.

Los consumidores no cambian sus registros: siguen pidiendo la ruta y la huella del archivo
original, y ``verify`` y ``read`` resuelven el estado. Cada operación conserva siempre una
forma legible del archivo: primero escribe y comprueba lo nuevo, después publica el registro
y por último borra lo anterior. Un bloqueo por carpeta serializa los registros.
"""

import base64
import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .storage import atomic_json, sha256

RETENTION_FILE = "predictions-retention.json"
KEY_COLUMNS = ("sample_id", "asset_id", "market", "prediction_at")
SHARED_COLUMNS = (*KEY_COLUMNS, "target")
MEDIAN = "quantile_0500"
# Columnas que pueden repetir los bits de la mediana y entonces no se guardan.
MEDIAN_COPIES = ("prediction", "center")
ROW_INDEX = "row"
PRESENT, COMPACTED, RELEASED = "present", "compacted", "released"
LARGE_GROUP_ROWS = 1 << 20
ZSTD_LEVEL = 3
_SCHEMA_KEY = b"mars_titan.original_schema"
_MAX_RETENTION_BYTES = 16 * 1024**2


class PredictionsReleased(ValueError):
    """Las filas de este archivo se liberaron y solo pueden regenerarse."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_order(table):
    """Orden común de un tramo: bloques por activo y, dentro de cada uno, por instante."""
    return np.lexsort(
        (
            table["prediction_at"].cast(pa.int64()).to_numpy(),
            table["asset_id"].to_numpy(zero_copy_only=False).astype(str),
        )
    )


def shared_rows(table):
    """Tabla común de un tramo: claves y objetivo en el orden común."""
    return table.select(list(SHARED_COLUMNS)).take(canonical_order(table))


def _float_bits(array):
    width = {2: np.uint16, 4: np.uint32, 8: np.uint64}[array.type.byte_width]
    return array.fill_null(0).to_numpy(zero_copy_only=False).view(width)


def same_table(left, right):
    """Mismo esquema y mismos valores, con los decimales comparados bit a bit."""
    if not left.schema.equals(right.schema) or left.num_rows != right.num_rows:
        return False
    for name in left.column_names:
        a, b = left[name], right[name]
        if pa.types.is_floating(a.type):
            if not a.is_null().equals(b.is_null()):
                return False
            if not np.array_equal(_float_bits(a), _float_bits(b)):
                return False
        elif not a.equals(b):
            return False
    return True


def _column_bytes(array):
    """Máscara de nulos y bytes de los valores de una columna, sin depender de sus trozos."""
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    nulls = array.is_null().to_numpy(zero_copy_only=False).astype(np.uint8).tobytes()
    kind = array.type
    if pa.types.is_floating(kind):
        bits = _float_bits(array)
        return nulls, bits.astype(bits.dtype.newbyteorder("<"))
    if pa.types.is_boolean(kind):
        return nulls, array.fill_null(False).to_numpy(zero_copy_only=False).astype("u1")
    if pa.types.is_integer(kind) or pa.types.is_timestamp(kind):
        return nulls, array.cast(pa.int64()).fill_null(0).to_numpy().astype("<i8")
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        text = array.cast(pa.large_string()).fill_null("")
        offsets = np.frombuffer(text.buffers()[1], dtype="<i8")
        offsets = offsets[text.offset : text.offset + len(text) + 1]
        data = text.buffers()[2]
        body = b"" if data is None else data.to_pybytes()[offsets[0] : offsets[-1]]
        return nulls, np.diff(offsets).tobytes() + body
    raise ValueError(f"Tipo de columna sin huella de contenido: {kind}")


def content_digest(table):
    """Huella del esquema y de los bits de cada columna en el orden de la tabla."""
    digest = hashlib.sha256()
    for field in table.schema:
        digest.update(f"{field.name}\0{field.type}\0{field.nullable}\n".encode())
    digest.update(f"rows={table.num_rows}\n".encode())
    for name in table.column_names:
        for part in _column_bytes(table[name]):
            part = part if isinstance(part, bytes) else part.tobytes()
            digest.update(len(part).to_bytes(8, "little"))
            digest.update(part)
    return digest.hexdigest()


def canonical_digest(table):
    """Huella del contenido en el orden común, que no depende del orden de escritura."""
    return content_digest(table.take(canonical_order(table)))


def _large_options(schema):
    floats = [field.name for field in schema if pa.types.is_floating(field.type)]
    return dict(
        compression="zstd",
        compression_level=ZSTD_LEVEL,
        use_dictionary=[name for name in schema.names if name not in floats] or False,
        use_byte_stream_split=floats or False,
    )


def write_large(table, path):
    """Escribir en grupos grandes con decimales separados por bytes, de forma atómica."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        pq.write_table(table, name, row_group_size=LARGE_GROUP_ROWS, **_large_options(table.schema))
        with open(name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return path


def _median_copy(table, name):
    return (
        name in table.column_names
        and MEDIAN in table.column_names
        and table.schema.field(name).type == table.schema.field(MEDIAN).type
        and same_table(table.select([name]), table.select([MEDIAN]).rename_columns([name]))
    )


def _positive_zero(table):
    if "zero" not in table.column_names or table.schema.field("zero").type != pa.float64():
        return False
    column = table["zero"]
    return column.null_count == 0 and not _float_bits(column).any()


def _rebuilt(table):
    """Columnas que se reconstruyen sin guardarlas."""
    names = [name for name in MEDIAN_COPIES if _median_copy(table, name)]
    return names + (["zero"] if _positive_zero(table) else [])


def _narrow(column):
    """Un float64 cuyos valores vuelven de float32 con los mismos bits se guarda en float32."""
    if column.type != pa.float64():
        return column
    narrow = column.cast(pa.float32(), safe=False)
    back = narrow.cast(pa.float64())
    same = same_table(pa.table({"v": back}), pa.table({"v": column}))
    return narrow if same else column


def split(table, rows):
    """Decimales propios de un ajuste y su índice de fila sobre la tabla común `rows`."""
    _require(
        set(SHARED_COLUMNS) <= set(table.column_names),
        "La tabla no tiene las claves y el objetivo del contrato",
    )
    order = canonical_order(table)
    _require(
        same_table(table.select(list(SHARED_COLUMNS)).take(order), rows),
        "Las claves u objetivos no son los de la tabla común",
    )
    skipped = set(SHARED_COLUMNS) | set(_rebuilt(table))
    columns = {name: _narrow(table[name]) for name in table.column_names if name not in skipped}
    _require(columns, "La tabla no tiene decimales propios")
    _require(ROW_INDEX not in columns, f"La columna {ROW_INDEX} está reservada")
    if not np.array_equal(order, np.arange(len(order))):
        index = np.empty(len(order), dtype=np.int64)
        index[order] = np.arange(len(order))
        columns[ROW_INDEX] = pa.array(index, pa.int32() if len(order) < 2**31 else pa.int64())
    encoded = base64.b64encode(table.schema.remove_metadata().serialize().to_pybytes())
    return pa.table(columns).replace_schema_metadata({_SCHEMA_KEY: encoded})


def restore(rows, own):
    """Reconstruir la tabla original desde la tabla común y los decimales propios."""
    metadata = own.schema.metadata or {}
    _require(_SCHEMA_KEY in metadata, "Los decimales no declaran el esquema original")
    schema = pa.ipc.read_schema(pa.py_buffer(base64.b64decode(metadata[_SCHEMA_KEY])))
    _require(own.num_rows == rows.num_rows, "Los decimales no tienen las filas de la tabla común")
    if ROW_INDEX in own.column_names:
        rows = rows.take(own[ROW_INDEX])
    columns = {}
    for field in schema:
        if field.name in SHARED_COLUMNS:
            columns[field.name] = rows[field.name]
        elif field.name in own.column_names:
            columns[field.name] = own[field.name].cast(field.type)
        elif field.name in MEDIAN_COPIES and MEDIAN in own.column_names:
            columns[field.name] = own[MEDIAN].cast(field.type)
        elif field.name == "zero":
            columns[field.name] = pa.array(np.zeros(rows.num_rows), field.type)
        else:
            raise ValueError(f"No se puede reconstruir la columna {field.name}")
    return pa.table(columns, schema=schema)


@contextlib.contextmanager
def _locked(folder):
    """Serializar los cambios del registro de retención de una carpeta."""
    descriptor = os.open(
        Path(folder) / f".{RETENTION_FILE}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _retention(folder):
    path = Path(folder) / RETENTION_FILE
    if not path.is_file():
        return {}
    _require(path.stat().st_size <= _MAX_RETENTION_BYTES, f"{path} supera su presupuesto")
    document = json.loads(path.read_text())
    _require(
        isinstance(document, dict)
        and document.get("kind") == "prediction_retention"
        and isinstance(document.get("files"), dict),
        f"{path} no es un registro de retención de predicciones",
    )
    return document["files"]


def _publish(folder, files):
    atomic_json(
        Path(folder) / RETENTION_FILE,
        dict(schema_version=1, kind="prediction_retention", files=files),
    )


def entry(path):
    """Registro de retención de un archivo de predicciones, o nada si sigue intacto."""
    path = Path(path)
    return _retention(path.parent).get(path.name)


def _regular(path, digest):
    return path.is_file() and not path.is_symlink() and sha256(path) == digest


def _rows_path(folder, record):
    path = Path(folder) / record["path"]
    _require(_regular(path, record["sha256"]), f"La tabla común {record['path']} ha cambiado")
    return path


def verify(path, expected_sha256, *, label=None):
    """Estado comprobado de un archivo de predicciones: present, compacted o released.

    `label` antepone a cualquier rechazo el artefacto que se estaba confirmando.
    """
    if label is None:
        return _verify(Path(path), expected_sha256)
    try:
        return _verify(Path(path), expected_sha256)
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from error


def _verify(path, expected_sha256):
    if _regular(path, expected_sha256):
        return PRESENT
    record = entry(path)
    _require(record is not None, f"La huella de {path.name} no coincide")
    _require(record["sha256"] == expected_sha256, f"{path.name} no es el archivo registrado")
    _require(not path.exists(), f"{path.name} existe con otra huella")
    if record["state"] == COMPACTED:
        compact = record["compact"]
        _require(
            _regular(path.parent / compact["path"], compact["sha256"]),
            f"Los decimales de {path.name} han cambiado",
        )
        _rows_path(path.parent, compact["rows"])
        return COMPACTED
    _require(record["state"] == RELEASED, f"Estado desconocido para {path.name}")
    return RELEASED


def read(path, expected_sha256, columns=None):
    """Tabla original de un archivo presente o compactado, con las columnas pedidas."""
    path = Path(path)
    state = verify(path, expected_sha256)
    if state == RELEASED:
        raise PredictionsReleased(
            f"Las filas de {path.name} se liberaron. Regenéralas desde el estado conservado"
        )
    if state == PRESENT:
        names = pq.read_schema(path).names
    else:
        compact = entry(path)["compact"]
        own = pq.read_table(path.parent / compact["path"], use_threads=False)
        rows = pq.read_table(_rows_path(path.parent, compact["rows"]), use_threads=False)
        table = restore(rows, own)
        names = table.column_names
    _require(columns is None or set(columns) <= set(names), f"Faltan columnas en {path.name}")
    if state == PRESENT:
        return pq.read_table(
            path, columns=None if columns is None else list(columns), use_threads=False
        )
    return table if columns is None else table.select(list(columns))


def describe(table):
    """Filas y huellas de contenido de una tabla, en su orden y en el común."""
    return dict(
        rows=table.num_rows,
        content_sha256=content_digest(table),
        canonical_sha256=canonical_digest(table),
    )


def write_rows(table, folder):
    """Escribir la tabla común de un tramo con dirección de contenido, una sola vez."""
    rows = shared_rows(table)
    path = Path(folder) / f"rows-{content_digest(rows)}.parquet"
    if not path.is_file():
        write_large(rows, path)
    _require(
        same_table(pq.read_table(path, use_threads=False), rows),
        f"La tabla común {path.name} no se relee igual",
    )
    return path


def _decimals(path):
    return path.with_name(f"{path.stem}.decimals.parquet")


def compact(path, expected_sha256, rows_folder):
    """Sustituir un archivo por sus decimales sobre la tabla común de su tramo.

    Comprueba antes de borrar que la reconstrucción repite la tabla original bit a bit.
    Es idempotente: un archivo ya compactado devuelve su registro.
    """
    path = Path(path)
    with _locked(path.parent):
        state = verify(path, expected_sha256)
        if state == COMPACTED:
            return entry(path)
        _require(state == PRESENT, f"{path.name} ya no tiene filas")
        table = pq.read_table(path, use_threads=False)
        rows_path = write_rows(table, rows_folder)
        rows = pq.read_table(rows_path, use_threads=False)
        own_path = _decimals(path)
        write_large(split(table, rows), own_path)
        restored = restore(rows, pq.read_table(own_path, use_threads=False))
        _require(same_table(restored, table), f"La compactación de {path.name} no es reversible")
        record = dict(
            sha256=expected_sha256,
            bytes=path.stat().st_size,
            **describe(table),
            state=COMPACTED,
            compact=dict(
                path=own_path.name,
                sha256=sha256(own_path),
                bytes=own_path.stat().st_size,
                rows=dict(path=os.path.relpath(rows_path, path.parent), sha256=sha256(rows_path)),
            ),
        )
        files = _retention(path.parent)
        files[path.name] = record
        _publish(path.parent, files)
        path.unlink()
        return record


def release(path, expected_sha256, **provenance):
    """Borrar las filas de un archivo presente o compactado y conservar sus huellas.

    `provenance` añade al registro lo necesario para regenerarlo, como el trabajo y la
    etapa. La tabla común no se borra aquí, porque la comparten otros ajustes.
    """
    path = Path(path)
    with _locked(path.parent):
        state = verify(path, expected_sha256)
        if state == RELEASED:
            # Un corte tras publicar el registro deja los decimales, que ya nadie lee.
            _decimals(path).unlink(missing_ok=True)
            return entry(path)
        files = _retention(path.parent)
        if state == PRESENT:
            table = pq.read_table(path, use_threads=False)
            record = dict(sha256=expected_sha256, bytes=path.stat().st_size, **describe(table))
            doomed = path
        else:
            previous = files[path.name]
            record = {k: v for k, v in previous.items() if k not in ("state", "compact")}
            doomed = path.parent / previous["compact"]["path"]
        record = dict(record, state=RELEASED, **provenance)
        files[path.name] = record
        _publish(path.parent, files)
        doomed.unlink()
        return record


def rows_references(folder):
    """Tablas comunes a las que apunta algún archivo compactado de una carpeta."""
    return {
        (Path(folder) / record["compact"]["rows"]["path"]).resolve()
        for record in _retention(folder).values()
        if record["state"] == COMPACTED
    }


def matches(path, expected_sha256, table):
    """Comprobar que una tabla regenerada repite bit a bit el contenido registrado.

    Para un archivo presente o compactado se compara con su tabla. Para uno liberado, con
    la huella de contenido en el orden escrito que se registró al liberarlo.
    """
    state = verify(path, expected_sha256)
    if state == RELEASED:
        record = entry(path)
        return (
            table.num_rows == record["rows"] and content_digest(table) == record["content_sha256"]
        )
    return same_table(read(path, expected_sha256), table)
