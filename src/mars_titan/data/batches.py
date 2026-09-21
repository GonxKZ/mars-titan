"""Acceso macro por grupos y confirmación atómica de particiones acotadas."""

import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def read_bounded_table(path: Path, *, max_rows: int = 100_000, max_bytes: int = 64 * 1024**2):
    if min(max_rows, max_bytes) < 1:
        raise ValueError("El presupuesto de lectura debe ser positivo")
    with pq.ParquetFile(path) as file:
        meta = file.metadata
        if meta.num_rows > max_rows or any(
            meta.row_group(i).total_byte_size > max_bytes for i in range(meta.num_row_groups)
        ):
            raise ValueError("La partición supera el presupuesto de lectura")
        batches, size = [], 0
        for batch in file.iter_batches(batch_size=64, use_threads=False):
            size += batch.nbytes
            if size > max_bytes:
                raise ValueError("La tabla decodificada supera el presupuesto de lectura")
            batches.append(batch)
        return pa.Table.from_batches(batches, schema=file.schema_arrow)


def atomic_parquet_batches(path: Path, tables) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    writer, count = None, 0
    try:
        for table in tables:
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            count += table.num_rows
        if writer is None:
            pq.write_table(pa.table({}), temporary, compression="zstd")
        else:
            writer.close()
            writer = None
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if writer is not None:
            writer.close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    return count


class MacroContexts:
    """Índice temporal de grupos Parquet, con una caché de dos grupos como máximo."""

    columns = ["prediction_at", "indicator_id", "value", "available_at", "unit"]

    def __init__(self, path: Path, *, max_group_bytes: int = 64 * 1024**2):
        if max_group_bytes < 1:
            raise ValueError("El presupuesto macro debe ser positivo")
        self.file = pq.ParquetFile(path, read_dictionary=["indicator_id", "unit"])
        self.cache = OrderedDict()
        self.ranges = []
        self.max_group_bytes = max_group_bytes
        try:
            index = self.file.schema_arrow.get_field_index("prediction_at")
            if index < 0 or not set(self.columns) <= set(self.file.schema_arrow.names):
                raise ValueError("Faltan columnas del contexto macro")
            for number in range(self.file.num_row_groups):
                group = self.file.metadata.row_group(number)
                if group.total_byte_size > max_group_bytes or group.num_rows > 100_000:
                    raise ValueError("El grupo macro supera el presupuesto de lectura")
                stats = group.column(index).statistics
                if stats is None or not stats.has_min_max or stats.null_count:
                    raise ValueError("El índice macro necesita fechas y estadísticas válidas")
                self.ranges.append((stats.min, stats.max))
            indicators = set()
            for batch in self.file.iter_batches(
                batch_size=256, columns=["indicator_id"], use_threads=False
            ):
                _check_string_width(batch.column(0), 128, "indicador")
                indicators.update(batch.column(0).to_pylist())
                if None in indicators or len(indicators) > 1024:
                    raise ValueError("El catálogo macro no es válido o supera el límite")
            self.indicators = sorted(indicators)
            if not self.indicators:
                raise ValueError("El panel macro está vacío")
        except BaseException:
            self.file.close()
            raise

    @property
    def cached_row_groups(self) -> int:
        return len(self.cache)

    def at(self, decision) -> list[dict]:
        result = []
        for number, (first, last) in enumerate(self.ranges):
            if not first <= decision <= last:
                continue
            if number not in self.cache:
                rows, size = [], 0
                for batch in self.file.iter_batches(
                    batch_size=64, row_groups=[number], columns=self.columns, use_threads=False
                ):
                    _check_string_width(batch.column("unit"), 256, "unidad")
                    _check_string_width(batch.column("indicator_id"), 128, "indicador")
                    size += batch.nbytes
                    if size > self.max_group_bytes:
                        raise ValueError("El grupo macro decodificado supera el presupuesto")
                    rows.extend(batch.to_pylist())
                self.cache[number] = rows
                if len(self.cache) > 2:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(number)
            result.extend(row for row in self.cache[number] if row["prediction_at"] == decision)
            if len(result) > 1024:
                raise ValueError("La decisión macro supera el límite de indicadores")
        return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()
        self.cache.clear()


def _check_string_width(column, limit: int, field: str) -> None:
    values = column.dictionary if pa.types.is_dictionary(column.type) else column
    maximum = pc.max(pc.utf8_length(values)).as_py()
    if maximum is not None and maximum > limit:
        raise ValueError(f"El campo {field} supera la longitud permitida")
