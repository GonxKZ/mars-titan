"""Admisión por sesión de todos los indicadores del catálogo macro.

La tabla conserva la evidencia del cálculo previo. Esta comprobación no reconstruye
versiones históricas ni recalcula fórmulas cuyos rezagos no contiene el panel diario.
"""

import csv
import ctypes
import hashlib
import json
import os
import resource
import tempfile
import time
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .macro import _catalog, _exclusion
from .preparation import atomic_parquet
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

_COLUMNS = (
    "prediction_at",
    "indicator_id",
    "value",
    "available_at",
    "period_start",
    "missing_reason",
    "source_hashes",
    "unit",
    "seasonal_adjustment",
)
_MAX_GROUP_BYTES = 64 * 1024**2
_MAX_GROUP_ROWS = 100_000
_MAX_MISSING_REASONS = 4096
_TEXT_LIMITS = {
    "indicator_id": 128,
    "period_start": 10,
    "missing_reason": 2048,
    "unit": 256,
    "seasonal_adjustment": 256,
    "source_hashes": 64,
}


def _read_catalog(path: Path) -> dict:
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError("El catálogo macro supera el presupuesto de lectura")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not 1 <= len(rows) <= 1024:
        raise ValueError("El catálogo necesita entre 1 y 1024 indicadores")
    required = {"id", "kind", "unit", "frequency", "input_ids", "formula", "vintage_policy"}
    for row in rows:
        if not required <= row.keys() or any(row[key] is None for key in required):
            raise ValueError("Faltan campos del contrato en el catálogo macro")
        if not row["id"].strip() or len(row["id"]) > 128 or not row["unit"].strip():
            raise ValueError("Identificador o unidad no válidos en el catálogo macro")
        if row["kind"] == "derived" and row["vintage_policy"] != "DERIVE_FROM_ASOF_VINTAGES":
            raise ValueError("El derivado necesita versiones históricas en el catálogo")
    return _catalog(rows)[0]


def _check_schema(schema: pa.Schema) -> None:
    if not set(_COLUMNS) <= set(schema.names) or len(schema.names) != len(set(schema.names)):
        raise ValueError("Faltan columnas del contrato macro o hay columnas duplicadas")
    for name in ("prediction_at", "available_at"):
        field = schema.field(name).type
        if not pa.types.is_timestamp(field) or field.tz != "UTC":
            raise ValueError("Las fechas macro deben ser timestamps UTC")
    for name in ("indicator_id", "period_start", "missing_reason", "unit", "seasonal_adjustment"):
        if not _string_type(schema.field(name).type):
            raise ValueError(f"El campo macro {name} debe contener texto")
    value = schema.field("value").type
    hashes = schema.field("source_hashes").type
    if not (pa.types.is_floating(value) or pa.types.is_integer(value)):
        raise ValueError("El valor macro debe ser numérico")
    if not pa.types.is_list(hashes) or not _string_type(hashes.value_type):
        raise ValueError("La procedencia macro debe contener una lista de huellas SHA-256")


def _string_type(kind) -> bool:
    return pa.types.is_string(kind) or (
        pa.types.is_dictionary(kind) and pa.types.is_string(kind.value_type)
    )


def _bool(values) -> np.ndarray:
    return pc.fill_null(values, False).to_numpy(zero_copy_only=False)


def _present(values) -> np.ndarray:
    return _bool(pc.greater(pc.utf8_length(pc.utf8_trim_whitespace(values)), 0))


def _selected_batches(file: pq.ParquetFile, first: datetime, stop: datetime):
    """Leer fechas primero y materializar únicamente el prefijo anterior al corte."""
    time_index = next(
        index
        for index in range(len(file.schema))
        if file.schema.column(index).path == "prediction_at"
    )
    for number in range(file.num_row_groups):
        group = file.metadata.row_group(number)
        if (
            group.num_rows > _MAX_GROUP_ROWS
            or group.total_byte_size > _MAX_GROUP_BYTES
            or any(group.column(i).num_values > 1_000_000 for i in range(group.num_columns))
        ):
            raise ValueError("El grupo macro supera el presupuesto de lectura")
        stats = group.column(time_index).statistics
        if stats and stats.has_min_max and (stats.max < first or stats.min >= stop):
            continue
        moments = file.read_row_group(number, columns=["prediction_at"], use_threads=False)
        column = moments["prediction_at"]
        if column.null_count:
            raise ValueError("La decisión macro no puede ser nula")
        selected = np.flatnonzero(
            _bool(pc.and_(pc.greater_equal(column, first), pc.less(column, stop)))
        )
        if not len(selected):
            continue
        last = int(selected[-1]) + 1
        if pc.any(pc.greater_equal(column.slice(0, last), stop)).as_py():
            raise ValueError("El orden macro mezclaría valores posteriores al corte temporal")
        # Las longitudes del formato acotan también índices y offsets de diccionarios.
        projected_bytes = (
            group.total_byte_size
            + 128 * group.num_rows
            + 8 * sum(group.column(i).num_values for i in range(group.num_columns))
        )
        if projected_bytes > _MAX_GROUP_BYTES:
            raise ValueError("El prefijo macro supera el presupuesto de lectura")
        batch = next(
            file.iter_batches(
                batch_size=last, row_groups=[number], columns=list(_COLUMNS), use_threads=False
            )
        )
        if batch.num_rows != last or batch.nbytes > _MAX_GROUP_BYTES:
            raise ValueError("El prefijo macro está incompleto o excede el presupuesto")
        for offset in range(0, len(selected), 512):
            decoded = _bounded_decode(batch.take(pa.array(selected[offset : offset + 512])))
            yield decoded, decoded.nbytes, number


def _bounded_decode(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Acotar texto y expansión de diccionarios antes de construir arrays densos."""
    expanded = batch.nbytes
    for name, limit in _TEXT_LIMITS.items():
        column = pc.list_flatten(batch[name]) if name == "source_hashes" else batch[name]
        if pa.types.is_dictionary(column.type):
            lengths = pc.take(pc.binary_length(column.dictionary), column.indices)
            expanded += (pc.sum(lengths).as_py() or 0) + 4 * len(column)
        else:
            lengths = pc.binary_length(column)
        maximum = pc.max(lengths).as_py()
        if maximum is not None and maximum > limit:
            raise ValueError(f"El campo {name} supera el presupuesto de texto")
    if expanded > _MAX_GROUP_BYTES:
        raise ValueError("El lote macro decodificado supera el presupuesto")
    fields = [
        pa.field(
            field.name,
            pa.list_(pa.string())
            if field.name == "source_hashes"
            else pa.string()
            if field.name in _TEXT_LIMITS
            else field.type,
        )
        for field in batch.schema
    ]
    return batch.cast(pa.schema(fields))


def _publish_directory(source: Path, destination: Path) -> None:
    """Confirmar el directorio sin sustituir un destino creado por otra ejecución."""
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise OSError("La publicación atómica requiere renameat2 con RENAME_NOREPLACE")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    # AT_FDCWD y RENAME_NOREPLACE evitan la carrera entre exists y rename.
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _invalid_rows(batch: pa.RecordBatch, entries: list[dict], indices: np.ndarray) -> dict:
    value, available = batch["value"], batch["available_at"]
    derived = np.array([entry["kind"] == "derived" for entry in entries])[indices]
    allowed = np.array(
        [entry["kind"] == "derived" or _exclusion(entry) is None for entry in entries]
    )
    units = pc.take(pa.array([entry["unit"] for entry in entries]), pa.array(indices))
    unit_valid = _present(batch["unit"]) & (~derived | _bool(pc.equal(batch["unit"], units)))
    hashes = batch["source_hashes"]
    flattened = pc.list_flatten(hashes)
    bad_hashes = ~_bool(pc.match_substring_regex(flattened, "^[0-9a-fA-F]{64}$"))
    parents = pc.list_parent_indices(hashes).to_numpy(zero_copy_only=False)
    bad_provenance = np.bincount(parents[bad_hashes], minlength=batch.num_rows) > 0
    bad_provenance |= ~_bool(pc.greater(pc.list_value_length(hashes), 0))
    periods = pc.strptime(batch["period_start"], format="%Y-%m-%d", unit="us", error_is_null=True)
    period_valid = _bool(pc.equal(pc.strftime(periods, format="%Y-%m-%d"), batch["period_start"]))
    period_valid &= _bool(pc.greater_equal(pc.year(periods), 1))
    period_valid &= _bool(
        pc.less_equal(pc.cast(periods, pa.date32()), pc.cast(available, pa.date32()))
    )
    return {
        "missing_value": _bool(pc.is_null(value)),
        "nonfinite_value": _bool(pc.and_(pc.is_valid(value), pc.invert(pc.is_finite(value)))),
        "missing_availability": _bool(pc.is_null(available)),
        "future_availability": _bool(pc.greater(available, batch["prediction_at"])),
        "declared_missing": _bool(pc.is_valid(batch["missing_reason"])),
        "invalid_unit": ~unit_valid,
        "invalid_provenance": bad_provenance,
        "missing_historical_adjustment": ~derived & ~_present(batch["seasonal_adjustment"]),
        "invalid_period": ~period_valid,
        "catalog_source_excluded": ~allowed[indices],
    }


def assess_macro_completeness(
    source: Path,
    catalog_path: Path,
    output: Path,
    *,
    start: str,
    end: str,
    market: str,
) -> dict:
    """Publicar sesiones con un único valor válido por cada ID del catálogo.

    El periodo incluye ambos días. Cuenta también las sesiones sin filas del panel.
    Una salida existente no se sustituye. El informe y el Parquet se confirman juntos
    en un directorio nuevo y externo al origen, después de verificar sus huellas.
    """
    started = time.perf_counter()
    source, catalog_path = (
        Path(source).resolve(strict=True),
        Path(catalog_path).resolve(strict=True),
    )
    output = Path(output)
    outside_source(source.parent, output)
    outside_source(catalog_path.parent, output)
    if output.exists():
        raise FileExistsError(f"La salida de admisión ya existe: {output}")
    first_day, last_day = date.fromisoformat(start), date.fromisoformat(end)
    if last_day < first_day or (last_day - first_day).days > 366 * 50:
        raise ValueError("El periodo macro está invertido o supera cincuenta años")
    first = datetime.combine(first_day, datetime.min.time(), UTC)
    stop = datetime.combine(last_day + timedelta(days=1), datetime.min.time(), UTC)
    source_hash, catalog_hash = sha256(source), sha256(catalog_path)
    catalog = _read_catalog(catalog_path)
    identifiers = sorted(catalog)
    entries = [catalog[name] for name in identifiers]
    clock = MarketClock(
        market,
        (first_day - timedelta(days=7)).isoformat(),
        (last_day + timedelta(days=7)).isoformat(),
    )
    decisions = [moment for moment in clock.decisions if first <= moment < stop]
    if len(decisions) * len(entries) > 8_000_000:
        raise ValueError("Las decisiones y el catálogo exceden el presupuesto de admisión")
    moments = pa.array(decisions, type=pa.timestamp("us", tz="UTC"))
    shape = len(decisions), len(entries)
    counts, valid_counts = np.zeros(shape, dtype=np.int32), np.zeros(shape, dtype=np.int32)
    max_available = np.full(len(decisions), np.iinfo(np.int64).min, dtype=np.int64)
    invalid_counts, missing_reasons = Counter(), Counter()
    rows = zero_rows = largest_batch = decoded_groups = 0
    last_group = -1
    metadata = pq.read_metadata(source)
    _check_schema(metadata.schema.to_arrow_schema())
    dictionary_columns = [
        metadata.schema.column(index).path
        for index in range(len(metadata.schema))
        if metadata.schema.column(index).path.split(".")[0] in _TEXT_LIMITS
    ]
    with pq.ParquetFile(source, metadata=metadata, read_dictionary=dictionary_columns) as file:
        available_type = file.schema_arrow.field("available_at").type
        source_rows = file.metadata.num_rows
        if source_rows > np.iinfo(np.int32).max:
            raise ValueError("El panel macro excede el presupuesto de filas")
        for batch, decoded_bytes, group_number in _selected_batches(file, first, stop):
            indicator_indices = pc.index_in(batch["indicator_id"], value_set=pa.array(identifiers))
            decision_indices = pc.index_in(batch["prediction_at"], value_set=moments)
            if indicator_indices.null_count:
                raise ValueError("El panel contiene un indicador ajeno al catálogo")
            if decision_indices.null_count:
                raise ValueError("La decisión macro no coincide con una sesión del reloj")
            indicator_indices = indicator_indices.to_numpy(zero_copy_only=False)
            decision_indices = decision_indices.to_numpy(zero_copy_only=False)
            invalid = _invalid_rows(batch, entries, indicator_indices)
            valid = ~np.logical_or.reduce(list(invalid.values()))
            np.add.at(counts, (decision_indices, indicator_indices), 1)
            np.add.at(valid_counts, (decision_indices[valid], indicator_indices[valid]), 1)
            availability = pc.cast(batch["available_at"], pa.int64())
            availability = pc.fill_null(availability, 0).to_numpy(zero_copy_only=False)
            np.maximum.at(max_available, decision_indices[valid], availability[valid])
            invalid_counts.update(
                {name: int(mask.sum()) for name, mask in invalid.items() if mask.any()}
            )
            reasons = Counter(pc.drop_null(batch["missing_reason"]).to_pylist())
            if len(missing_reasons.keys() | reasons.keys()) > _MAX_MISSING_REASONS:
                raise ValueError("Los motivos de ausencia superan el presupuesto")
            missing_reasons.update(reasons)
            zero_rows += int(np.count_nonzero(valid & _bool(pc.equal(batch["value"], 0))))
            rows += batch.num_rows
            if group_number != last_group:
                decoded_groups += 1
                last_group = group_number
            largest_batch = max(largest_batch, decoded_bytes)
    if sha256(source) != source_hash or sha256(catalog_path) != catalog_hash:
        raise ValueError("El origen o el catálogo cambió durante la comprobación macro")
    admitted_cells = (counts == 1) & (valid_counts == 1)
    complete = admitted_cells.all(axis=1)
    admitted = pa.table(
        {
            "prediction_at": moments.filter(pa.array(complete)),
            "macro_available_at": pa.array(max_available[complete], type=available_type),
            "indicator_count": pa.array(
                np.full(int(complete.sum()), len(entries)), type=pa.int32()
            ),
        }
    )
    period = {
        "market": market,
        "start": start,
        "end": end,
        "decisions": [t.isoformat() for t in decisions],
    }
    report = {
        "schema_version": 1,
        "policy": "all_catalog_indicators_valid",
        "market": market,
        "start": start,
        "end": end,
        "source_sha256": source_hash,
        "catalog_sha256": catalog_hash,
        "period_sha256": hashlib.sha256(json.dumps(period, sort_keys=True).encode()).hexdigest(),
        "required_indicator_ids": identifiers,
        "required_indicator_count": len(entries),
        "source_rows": source_rows,
        "selected_rows": rows,
        "total_decisions": len(decisions),
        "complete_decisions": int(complete.sum()),
        "excluded_decisions": int((~complete).sum()),
        "population_ready": bool(complete.any()),
        "missing_by_indicator": dict(
            zip(identifiers, (~admitted_cells).sum(axis=0).tolist(), strict=True)
        ),
        "absent_rows_by_indicator": dict(
            zip(identifiers, (counts == 0).sum(axis=0).tolist(), strict=True)
        ),
        "duplicate_indicator_decisions": int((counts > 1).any(axis=1).sum()),
        "invalid_rows_by_reason": dict(sorted(invalid_counts.items())),
        "declared_missing_reasons": dict(sorted(missing_reasons.items())),
        "zero_value_rows": zero_rows,
        "complete_decisions_path": "complete-decisions.parquet",
        "unit_policy": "historical_native_raw_units_and_catalog_derived_units",
        "provenance_policy": "nonempty_sha256_list_from_asof_materialization",
        "formula_recalculation": False,
        "decoded_row_groups": decoded_groups,
        "largest_decoded_batch_bytes": largest_batch,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "admission"
        stage.mkdir()
        atomic_parquet(stage / "complete-decisions.parquet", admitted)
        report["complete_decisions_sha256"] = sha256(stage / "complete-decisions.parquet")
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(stage / "report.json", report)
        _publish_directory(stage, output)
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return report
