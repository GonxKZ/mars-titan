"""Contextos temporales sin acumular cuerpos ni recalcular todo el historial."""

from collections import OrderedDict
from itertools import groupby
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .samples import macro_vector
from .storage import sha256
from .temporal import aware


class NewsWindows:
    """Consultar intervalos inclusivos con una caché de dos grupos Arrow."""

    columns = (
        "available_at",
        "content_hash",
        "event_id",
        "content_kind",
        "text",
        "cohort_id",
        "availability_rule",
    )

    def __init__(self, path, *, max_group_bytes=16 * 1024**2):
        if type(max_group_bytes) is not int or max_group_bytes < 1:
            raise ValueError("El presupuesto de grupos debe ser positivo")
        self.file = pq.ParquetFile(path, read_dictionary=["text"])
        self.cache, self.ranges = OrderedDict(), []
        self.max_group_bytes = max_group_bytes
        try:
            schema = self.file.schema_arrow
            if not set(self.columns) <= set(schema.names):
                raise ValueError("Faltan columnas del texto preparado")
            dtype = schema.field("available_at").type
            if not pa.types.is_timestamp(dtype) or dtype.tz is None:
                raise ValueError("La disponibilidad textual requiere zona horaria")
            index, previous = schema.get_field_index("available_at"), None
            for group in range(self.file.num_row_groups):
                metadata = self.file.metadata.row_group(group)
                if metadata.total_byte_size > max_group_bytes:
                    raise ValueError("El grupo de noticias supera el presupuesto")
                if not metadata.num_rows:
                    continue
                stats = metadata.column(index).statistics
                if not stats or not stats.has_min_max or stats.null_count:
                    raise ValueError("Las noticias requieren estadísticas temporales completas")
                if previous is not None and stats.min < previous:
                    raise ValueError("Los grupos de noticias no están ordenados")
                previous = stats.max
                self.ranges.append((stats.min, stats.max, group))
        except BaseException:
            self.file.close()
            raise

    @property
    def cached_row_groups(self):
        return len(self.cache)

    def between(self, start, end):
        start, end = aware(start), aware(end)
        if start > end:
            raise ValueError("El intervalo textual está invertido")
        for first, last, number in self.ranges:
            if first > end:
                break
            if last < start:
                continue
            if number not in self.cache:
                table = self.file.read_row_group(
                    number, columns=list(self.columns), use_threads=False
                )
                if table.nbytes > self.max_group_bytes:
                    raise ValueError("El grupo textual decodificado supera el presupuesto")
                times = table["available_at"].to_pylist()
                if any(a > b for a, b in zip(times, times[1:], strict=False)):
                    raise ValueError("Las filas de noticias no están ordenadas")
                self.cache[number] = table
                if len(self.cache) > 2:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(number)
            table = self.cache[number]
            selected = table.filter(
                pc.and_(
                    pc.greater_equal(table["available_at"], start),
                    pc.less_equal(table["available_at"], end),
                )
            )
            for batch in selected.to_batches(max_chunksize=1):
                yield from batch.to_pylist()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()
        self.cache.clear()


class FactCursor:
    """Avanzar por publicaciones sin volver a recorrer todos los hechos por decisión."""

    def __init__(self, rows):
        self.rows = iter(
            sorted(
                (MappingProxyType(dict(row)) for row in rows),
                key=lambda r: aware(r["available_at"]),
            )
        )
        self.next = next(self.rows, None)
        self.values, self.ranks, self.ambiguous = {}, {}, set()
        self.last = None

    def at(self, cutoff):
        cutoff = aware(cutoff)
        if self.last is not None and cutoff < self.last:
            raise ValueError("El cursor contable no puede retroceder")
        self.last = cutoff
        while self.next is not None and self.next["available_at"] <= cutoff:
            row = self.next
            concept = row["concept"]
            rank = (row["period_end"], row["available_at"], row["period_start"] or "")
            if concept not in self.ranks or rank > self.ranks[concept]:
                self.values[concept], self.ranks[concept] = row, rank
                self.ambiguous.discard(concept)
            elif rank == self.ranks[concept] and row["value"] != self.values[concept]["value"]:
                self.ambiguous.add(concept)
            self.next = next(self.rows, None)
        return {key: value for key, value in self.values.items() if key not in self.ambiguous}


class MacroVectors:
    """Calcular una vez los vectores macro y compartirlos entre los activos."""

    def __init__(self, path, *, max_bytes=64 * 1024**2, cutoff_year=2023):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("El presupuesto macro debe ser positivo")
        self.path = Path(path)
        self.sha256 = sha256(self.path)
        self.index, self.available, self.indicators = {}, [], []
        with pq.ParquetFile(path) as file:
            columns = ["prediction_at", "indicator_id", "value", "available_at"]
            if not set(columns) <= set(file.schema_arrow.names):
                raise ValueError("Faltan columnas del contexto macro")
            for name in ("prediction_at", "available_at"):
                dtype = file.schema_arrow.field(name).type
                if not pa.types.is_timestamp(dtype) or not dtype.tz:
                    raise ValueError("Las fechas macro deben incluir zona horaria")
            names = set()
            for batch in file.iter_batches(
                batch_size=4096, columns=["indicator_id"], use_threads=False
            ):
                for name in batch.column(0).to_pylist():
                    if not isinstance(name, str) or not 1 <= len(name) <= 128:
                        raise ValueError("El identificador macro no es válido")
                    names.add(name)
                if len(names) > 1024:
                    raise ValueError("El catálogo macro supera el presupuesto")
            self.indicators = sorted(names)
            width = 3 * len(names)
            if not width:
                raise ValueError("El contexto macro está vacío")
            capacity = file.metadata.num_rows // len(names)
            if capacity * (width * 4 + 256) > max_bytes:
                raise ValueError("Los vectores e índices macro superan el presupuesto")
            self.values = np.empty((capacity, width), dtype=np.float32)

            def records():
                for batch in file.iter_batches(batch_size=1024, columns=columns, use_threads=False):
                    yield from batch.to_pylist()

            previous = None
            for moment, group in groupby(records(), key=lambda r: r["prediction_at"]):
                moment = aware(moment)
                if previous is not None and moment <= previous:
                    raise ValueError("Las decisiones macro no están ordenadas")
                previous = moment
                rows = []
                for row in group:
                    if len(rows) >= len(self.indicators):
                        raise ValueError(
                            "La decisión macro repite indicadores o excede su presupuesto"
                        )
                    rows.append(row)
                if sorted(row["indicator_id"] for row in rows) != self.indicators:
                    raise ValueError("La decisión macro no conserva todos sus indicadores")
                if moment.year > cutoff_year:
                    continue
                if not any(row["value"] is not None for row in rows):
                    continue
                vector, available = macro_vector(rows, moment)
                position = len(self.index)
                if position >= capacity:
                    raise ValueError("Las decisiones macro exceden el presupuesto declarado")
                self.values[position] = vector
                self.index[moment] = position
                self.available.append(available)
        self.values.flags.writeable = False
        self.nbytes = self.values.nbytes
        if sha256(self.path) != self.sha256:
            raise ValueError("El contexto macro cambió durante su lectura")
        self.signature = self.path.stat()

    def verify(self):
        """Reutilizar la lectura solo mientras siga identificando el mismo archivo."""
        current = self.path.stat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(current, field) != getattr(self.signature, field) for field in fields):
            if sha256(self.path) != self.sha256:
                raise ValueError("El contexto macro ha cambiado desde su lectura")
            self.signature = current

    def at(self, moment):
        position = self.index.get(aware(moment))
        return None if position is None else (self.values[position], self.available[position])
