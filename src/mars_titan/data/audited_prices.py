"""Reutilización de precios auditados con huellas, calendario y corte comprobados."""

import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .batches import _check_string_width
from .cohort_files import read_manifest, safe_destination
from .storage import sha256
from .temporal import aware

MAX_PRICE_ROWS = 200_000
MAX_PRICE_BYTES = 64 * 1024**2
COLUMNS = ("open", "high", "low", "close", "volume", "session", "available_at")


def audit_catalog(path):
    safe_destination(Path(path))
    manifest, identity = read_manifest(Path(path), MAX_PRICE_BYTES)
    if not isinstance(manifest.get("files"), dict) or not isinstance(
        manifest.get("details_root"), str
    ):
        raise ValueError("El estado de precios auditados no identifica sus archivos")
    root, records = Path(manifest["details_root"]), {}
    safe_destination(root)
    root = root.resolve()
    for row in manifest["files"].values():
        market, symbol = row.get("market"), row.get("symbol")
        if (
            market not in {"US", "CN"}
            or not isinstance(symbol, str)
            or symbol in {".", ".."}
            or not re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", symbol)
            or (market, symbol) in records
        ):
            raise ValueError("El estado de precios repite o no identifica un activo")
        artifact = row.get("artifacts", {}).get("prices.parquet", {})
        records[market, symbol] = dict(
            path=str(root / market / symbol / "prices.parquet"),
            sha256=artifact.get("sha256"),
            rows=artifact.get("rows"),
            source_sha256=row.get("source_sha256"),
        )
    return records, identity


def confirm_audited_prices(record, source_hash):
    """Confirmar la fuente ya validada sin volver a descomprimir sus valores."""
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "sha256", "rows", "source_sha256"}
        or record["source_sha256"] != source_hash
        or type(record["rows"]) is not int
        or not 0 <= record["rows"] <= MAX_PRICE_ROWS
        or not isinstance(record["path"], str)
    ):
        raise ValueError("Los precios auditados no corresponden al activo o a su presupuesto")
    path = Path(record["path"])
    safe_destination(path)
    if not path.is_file() or path.stat().st_size > MAX_PRICE_BYTES:
        raise ValueError("Los precios auditados superan el presupuesto de archivo de 64 MiB")
    if sha256(path) != record["sha256"]:
        raise ValueError("La huella de los precios auditados ha cambiado")
    return path


def read_audited_prices(record, source_hash, clock, cutoff):
    return _read_audited_columns(record, source_hash, clock, cutoff, COLUMNS[:5])


def read_audited_factor(record, source_hash, clock, cutoff):
    """Leer solo los precios de apertura/cierre que necesita el residual."""
    return _read_audited_columns(record, source_hash, clock, cutoff, ("open", "close"))


def _read_audited_columns(record, source_hash, clock, cutoff, value_columns):
    columns = (*value_columns, "session", "available_at")
    path = confirm_audited_prices(record, source_hash)
    with pq.ParquetFile(path, read_dictionary=["session"]) as file:
        if file.metadata.num_rows != record["rows"] or not set(columns) <= set(
            file.schema_arrow.names
        ):
            raise ValueError("Los precios auditados no conservan su esquema o recuento")
        if any(
            file.metadata.row_group(i).total_byte_size > MAX_PRICE_BYTES
            for i in range(file.num_row_groups)
        ):
            raise ValueError("Un grupo de precios auditados supera el presupuesto de 64 MiB")
        if any(
            not (
                pa.types.is_floating(file.schema_arrow.field(name).type)
                or pa.types.is_integer(file.schema_arrow.field(name).type)
            )
            for name in value_columns
        ):
            raise ValueError("Los precios auditados no contienen columnas numéricas")
        stamp_type = file.schema_arrow.field("available_at").type
        if not pa.types.is_timestamp(stamp_type) or not stamp_type.tz:
            raise ValueError("La disponibilidad auditada necesita zona horaria")
        parts, decoded, selected_rows, previous = [], 0, 0, None
        for group in range(file.num_row_groups):
            count = 0
            for batch in file.iter_batches(
                batch_size=256, row_groups=[group], columns=["session"], use_threads=False
            ):
                if batch.nbytes > MAX_PRICE_BYTES:
                    raise ValueError("El lote de sesiones auditadas supera el presupuesto")
                column = batch.column(0)
                _check_string_width(column, 10, "sesión")
                for day in column.to_pylist():
                    if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
                        raise ValueError("Las sesiones auditadas no son fechas válidas")
                    if previous is not None and day <= previous:
                        raise ValueError("Las sesiones auditadas están repetidas o desordenadas")
                    previous = day
                    count += day <= cutoff
            if not count:
                continue
            selected_rows += count
            # Solo campos numéricos y fechas de diez caracteres ya comprobadas.
            # El prefijo evita decodificar valores de las filas reservadas.
            if decoded + count * 80 > MAX_PRICE_BYTES:
                raise ValueError("Los precios auditados superan el presupuesto decodificado")
            batch = next(
                file.iter_batches(
                    batch_size=count, row_groups=[group], columns=list(columns), use_threads=False
                )
            )
            decoded += batch.nbytes
            if decoded > MAX_PRICE_BYTES:
                raise ValueError("Los precios auditados superan 64 MiB")
            parts.append(pa.Table.from_batches([batch]))
        table = (
            pa.concat_tables(parts)
            if parts
            else pa.Table.from_batches(
                [], schema=pa.schema([file.schema_arrow.field(name) for name in columns])
            )
        )
    frame = table.to_pandas()
    if len(frame):
        values = frame[list(value_columns)].to_numpy(dtype=np.float64)
        if value_columns == COLUMNS[:5]:
            o, h, lo, c, v = values.T
            invalid = (
                (o <= 0)
                | (lo <= 0)
                | (c <= 0)
                | (h < lo)
                | (h < o)
                | (h < c)
                | (lo > o)
                | (lo > c)
                | (v < 0)
            )
        else:
            invalid = (values <= 0).any(axis=1)
        if not np.isfinite(values).all() or np.any(invalid):
            raise ValueError("Los precios auditados contienen OHLCV inválido")
        for day, available in zip(frame.session, frame.available_at, strict=True):
            if aware(available) != clock.decision(day):
                raise ValueError(
                    "El calendario auditado no coincide con la disponibilidad de precios"
                )
    if sha256(path) != record["sha256"]:
        raise ValueError("Los precios auditados cambiaron durante la lectura")
    return (
        frame,
        dict(
            mode="audited_parquet",
            source_sha256=source_hash,
            artifact_sha256=record["sha256"],
            accepted=len(frame),
        ),
        record["rows"] - selected_rows,
    )
