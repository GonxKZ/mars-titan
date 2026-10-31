"""Vistas analíticas acotadas de fallos maduros del checkpoint S2 actual.

La retención no es el historial completo. Filtrar un checkpoint futuro tampoco
reconstruye la selección que existía antes. Los pendientes nunca se exportan.
"""

import json
import os
import tempfile
from pathlib import Path

import duckdb
import pyarrow as pa

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.models.titans.config import bounded_integer, canonical

from .failure_context import UNKNOWN_PUBLICATION
from .failure_ledger import FailureLedger

MAX_BATCH_ROWS = 512
_SCOPES = {"snapshot": "snapshot_before_feedback", "retained": "retained_after_feedback"}


def _schema(metadata):
    instant = pa.timestamp("us", tz="UTC")
    return pa.schema(
        [
            pa.field("context_id", pa.string(), nullable=False),
            pa.field("flow_id", pa.string(), nullable=False),
            pa.field("market", pa.string(), nullable=False),
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("input_sha256", pa.string(), nullable=False),
            pa.field("action", pa.string(), nullable=False),
            pa.field("prediction_at", instant, nullable=False),
            pa.field("input_available_at", instant, nullable=False),
            pa.field("available_at", instant, nullable=False),
            pa.field("received_at", instant, nullable=False),
            pa.field("issued_prediction", pa.float64(), nullable=False),
            pa.field("target", pa.float64(), nullable=False),
            pa.field("error", pa.float64(), nullable=False),
            pa.field("severity", pa.float64(), nullable=False),
            pa.field("key_inputs", pa.list_(pa.field("element", pa.float64())), nullable=False),
            pa.field("macro", pa.list_(pa.field("element", pa.float32())), nullable=False),
            pa.field("macro_available_at", pa.list_(pa.field("element", instant)), nullable=False),
            pa.field("presence", pa.list_(pa.field("element", pa.bool_())), nullable=False),
            pa.field("area_activity", pa.list_(pa.field("element", pa.bool_())), nullable=False),
        ],
        metadata={b"failure_memory": canonical(metadata).encode()},
    )


def _row(record, threshold):
    emission, context = record.emission, record.emission.context
    return dict(
        context_id=context.context_id,
        flow_id=context.flow_id,
        market=context.flow_id.split("/", 1)[0],
        sample_id=context.sample_id,
        input_sha256=context.input_sha256,
        action=context.action,
        prediction_at=context.prediction_at,
        input_available_at=context.input_available_at,
        available_at=record.available_at,
        received_at=record.received_at,
        issued_prediction=emission.prediction,
        target=record.target,
        error=record.target - emission.prediction,
        severity=1 - threshold / record.severity,
        key_inputs=context.key_inputs.tolist(),
        macro=context.macro.tolist(),
        macro_available_at=[
            None if value == UNKNOWN_PUBLICATION else value
            for value in context.macro_available_at.tolist()
        ],
        presence=context.presence.tolist(),
        area_activity=context.area_activity.tolist(),
    )


def _batches(records, threshold, schema, batch_rows):
    if not records:
        yield pa.RecordBatch.from_pylist([], schema=schema)
    for start in range(0, len(records), batch_rows):
        yield pa.RecordBatch.from_pylist(
            [_row(record, threshold) for record in records[start : start + batch_rows]],
            schema=schema,
        )


def failure_reader(book, *, scope="snapshot", as_of=None, batch_rows=128):
    """Preparar una vista inmutable, por lotes, del conjunto maduro elegido."""
    if type(book) is not FailureLedger or type(scope) is not str or scope not in _SCOPES:
        raise ValueError("La exportación S2 necesita un libro y un alcance explícitos")
    bounded_integer(batch_rows, "filas por lote analítico", 1, MAX_BATCH_ROWS)
    if as_of is not None and (type(as_of) is not int or as_of != book.clock):
        raise ValueError(
            "Otro instante necesita su propio checkpoint, no un filtrado retrospectivo"
        )
    payload = book.export()
    verified = FailureLedger.restore(book.layout, book.configuration, payload)
    records = verified._snapshot if scope == "snapshot" else verified._retained
    metadata = dict(
        schema_version=1,
        kind="mature_failure_analytics_v1",
        state_sha256=payload["state_sha256"],
        configuration=verified.configuration.identity(),
        layout=verified.layout.identity(),
        as_of_us=verified.clock,
        scope=_SCOPES[scope],
        exported=len(records),
        pending_excluded=verified.counts()["pending"],
        admitted_total=verified.counts()["admitted"],
        history_complete=False,
        macro_dates="provided_or_null_never_inferred_from_age",
    )
    schema = _schema(metadata)
    return pa.RecordBatchReader.from_batches(
        schema, _batches(records, verified.configuration.threshold, schema, batch_rows)
    )


def write_failures(book, directory, *, scope="snapshot", as_of=None, batch_rows=128):
    """Escribir un archivo por huella y alcance, conservando resultados existentes."""
    reader = failure_reader(book, scope=scope, as_of=as_of, batch_rows=batch_rows)
    metadata = json.loads(reader.schema.metadata[b"failure_memory"])
    path = Path(directory) / f"failure-{scope}-{metadata['state_sha256']}.parquet"
    if path.exists():
        reader.close()
        raise FileExistsError("La exportación S2 ya existe y no se sobrescribe")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        atomic_parquet_batches(
            Path(temporary), (pa.Table.from_batches([batch]) for batch in reader)
        )
        # link falla si otro escritor publicó la misma huella entre tanto.
        os.link(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        reader.close()
        os.unlink(temporary)
    return path


def failure_counts(book, *, scope="snapshot", as_of=None, batch_rows=128):
    """Conteos del conjunto exportado, sin convertirlos en métricas del predictor completo."""
    reader = failure_reader(book, scope=scope, as_of=as_of, batch_rows=batch_rows)
    options = dict(
        threads=1,
        memory_limit="64MiB",
        max_temp_directory_size="0B",
        autoinstall_known_extensions=False,
        autoload_known_extensions=False,
    )
    try:
        with duckdb.connect(config=options) as connection:
            connection.register("mature_failure_memory", reader)
            rows = connection.execute(
                "SELECT market,action,count(*) AS eligible_failures,"
                "min(epoch_us(available_at)),max(epoch_us(available_at)) "
                "FROM mature_failure_memory GROUP BY market,action ORDER BY market,action"
            ).fetchall()
    finally:
        reader.close()
    fields = ("market", "action", "eligible_failures", "first_available_at", "last_available_at")
    return [dict(zip(fields, row, strict=True)) for row in rows]
