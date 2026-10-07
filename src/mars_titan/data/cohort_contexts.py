"""Contextos temporales sin acumular cuerpos ni recalcular todo el historial."""

import hashlib
import json
from collections import OrderedDict
from itertools import groupby
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .batches import _check_string_width
from .cohort_files import safe_destination
from .input_policy import STRICT_INPUTS, masked_inputs, numeric_observations, policy_identity
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
    dictionary_columns = ("text",)

    def __init__(self, path, *, max_group_bytes=16 * 1024**2):
        if type(max_group_bytes) is not int or max_group_bytes < 1:
            raise ValueError("El presupuesto de grupos debe ser positivo")
        self.file = pq.ParquetFile(path, read_dictionary=list(self.dictionary_columns))
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

    def __init__(
        self,
        path,
        *,
        max_bytes=64 * 1024**2,
        cutoff_year=2023,
        input_policy=STRICT_INPUTS,
        indicators=None,
    ):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("El presupuesto macro debe ser positivo")
        masked = masked_inputs(input_policy)
        if type(cutoff_year) is not int or cutoff_year > 2023:
            raise ValueError("El contexto macro debe mantener cerrada la reserva desde 2024")
        self.input_policy, self.cutoff_year = input_policy, cutoff_year
        declared = None if indicators is None else list(indicators)
        if declared is not None and (
            not 1 <= len(declared) <= 1024
            or len(set(declared)) != len(declared)
            or any(not isinstance(name, str) or not 1 <= len(name) <= 128 for name in declared)
        ):
            raise ValueError("El catálogo macro declarado no es válido")
        self.path = None if path is None else Path(path)
        self.index, self.available, self.indicators = {}, [], []
        self.missing = []
        if path is None:
            if not masked or declared is None:
                raise ValueError("Una fuente macro ausente necesita política histórica y catálogo")
            self.indicators = sorted(declared)
            self.empty = np.zeros(3 * len(declared), dtype=np.float32)
            self.empty.flags.writeable = False
            self.values = np.empty((0, len(self.empty)), dtype=np.float32)
            self.values.flags.writeable = False
            self.nbytes = self.empty.nbytes
            if self.nbytes > max_bytes:
                raise ValueError("El vector macro ausente supera el presupuesto")
            self.sha256 = hashlib.sha256(
                json.dumps(
                    dict(**policy_identity(input_policy), indicators=self.indicators, source=None),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            return
        safe_destination(self.path)
        if self.path.stat().st_size > max_bytes:
            raise ValueError("El archivo macro supera el presupuesto")
        self.sha256 = sha256(self.path)
        with pq.ParquetFile(path) as metadata:
            dictionaries = [
                name
                for name in ("indicator_id", "missing_reason")
                if name in metadata.schema_arrow.names
            ]
        with pq.ParquetFile(path, read_dictionary=dictionaries) as file:
            if any(
                file.metadata.row_group(i).total_byte_size > max_bytes
                for i in range(file.num_row_groups)
            ):
                raise ValueError("Un grupo macro supera el presupuesto")
            columns = ["prediction_at", "indicator_id", "value", "available_at"]
            if not set(columns) <= set(file.schema_arrow.names):
                raise ValueError("Faltan columnas del contexto macro")
            for name in ("prediction_at", "available_at"):
                dtype = file.schema_arrow.field(name).type
                if not pa.types.is_timestamp(dtype) or not dtype.tz:
                    raise ValueError("Las fechas macro deben incluir zona horaria")
            if masked and "missing_reason" in file.schema_arrow.names:
                columns.append("missing_reason")
            names, moments, previous_time = set(), 0, None
            current_names = set()
            for batch in file.iter_batches(
                batch_size=4096,
                columns=["indicator_id", "prediction_at"] if masked else ["indicator_id"],
                use_threads=False,
            ):
                if batch.nbytes > max_bytes:
                    raise ValueError("El lote de identificadores macro supera el presupuesto")
                _check_string_width(batch.column(0), 128, "indicador")
                times = batch.column(1).to_pylist() if masked else None
                for index, name in enumerate(batch.column(0).to_pylist()):
                    if not isinstance(name, str) or not 1 <= len(name) <= 128:
                        raise ValueError("El identificador macro no es válido")
                    names.add(name)
                    if masked:
                        stamp = aware(times[index])
                        if previous_time is not None and stamp < previous_time:
                            raise ValueError("Las decisiones macro no están ordenadas")
                        if stamp != previous_time:
                            current_names.clear()
                            if stamp.year <= cutoff_year:
                                moments += 1
                        if name in current_names:
                            raise ValueError("La decisión macro repite indicadores")
                        current_names.add(name)
                        previous_time = stamp
                        if moments > max_bytes // 256:
                            raise ValueError("Los índices macro superan el presupuesto")
                if len(names) > 1024:
                    raise ValueError("El catálogo macro supera el presupuesto")
            if declared is not None and (
                not names <= set(declared) or (not masked and names != set(declared))
            ):
                raise ValueError("El panel no corresponde al catálogo macro declarado")
            self.indicators = sorted(declared if declared is not None else names)
            width = 3 * len(self.indicators)
            if not width:
                raise ValueError("El contexto macro está vacío")
            capacity = moments if masked else file.metadata.num_rows // len(names)
            row_bytes = width * 4 + (len(self.indicators) * 8 + 512 if masked else 256)
            if capacity * row_bytes > max_bytes:
                raise ValueError("Los vectores e índices macro superan el presupuesto")
            self.values = np.empty((capacity, width), dtype=np.float32)

            def records():
                for batch in file.iter_batches(batch_size=1024, columns=columns, use_threads=False):
                    if batch.nbytes > max_bytes:
                        raise ValueError("El lote macro supera el presupuesto")
                    _check_string_width(batch.column("indicator_id"), 128, "indicador")
                    if "missing_reason" in columns:
                        _check_string_width(
                            batch.column("missing_reason"), 2048, "causa de ausencia"
                        )
                    yield from batch.to_pylist()

            previous, reason_bytes = None, 0
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
                found = [row["indicator_id"] for row in rows]
                if len(set(found)) != len(found) or not set(found) <= set(self.indicators):
                    raise ValueError("La decisión macro repite indicadores o cambia su catálogo")
                if not masked and sorted(found) != self.indicators:
                    raise ValueError("La decisión macro no conserva todos sus indicadores")
                if moment.year > cutoff_year:
                    continue
                if not masked and not any(row["value"] is not None for row in rows):
                    continue
                if masked:
                    by_id = {row["indicator_id"]: row for row in rows}
                    rows = [
                        by_id.get(name, dict(indicator_id=name, value=None, available_at=None))
                        for name in self.indicators
                    ]
                    causes = numeric_observations(rows, moment)[3]
                    reason_bytes += sum(
                        len(reason.encode()) + 64 for reason in causes if reason is not None
                    )
                    if capacity * row_bytes + reason_bytes > max_bytes:
                        raise ValueError("Las causas de ausencia macro superan el presupuesto")
                    self.missing.append(causes)
                vector, available = macro_vector(rows, moment, input_policy=input_policy)
                position = len(self.index)
                if position >= capacity:
                    raise ValueError("Las decisiones macro exceden el presupuesto declarado")
                self.values[position] = vector
                self.index[moment] = position
                self.available.append(available)
        self.values.flags.writeable = False
        self.empty = np.zeros(width, dtype=np.float32)
        self.empty.flags.writeable = False
        self.nbytes = self.values.nbytes
        if sha256(self.path) != self.sha256:
            raise ValueError("El contexto macro cambió durante su lectura")
        self.signature = self.path.stat()

    def verify(self):
        """Reutilizar la lectura solo mientras siga identificando el mismo archivo."""
        if self.path is None:
            return
        current = self.path.stat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(current, field) != getattr(self.signature, field) for field in fields):
            if sha256(self.path) != self.sha256:
                raise ValueError("El contexto macro ha cambiado desde su lectura")
            self.signature = current

    def at(self, moment):
        moment = aware(moment)
        if moment.year > self.cutoff_year:
            return None
        position = self.index.get(moment)
        if position is None and masked_inputs(self.input_policy):
            return self.empty, None
        return None if position is None else (self.values[position], self.available[position])

    def missing_at(self, moment):
        position = self.index.get(aware(moment))
        if not masked_inputs(self.input_policy):
            raise ValueError("La política estricta no publica causas de ausencia por celda")
        if position is None:
            return ["source_missing" if self.path is None else "missing_session"] * len(
                self.indicators
            )
        return list(self.missing[position])
