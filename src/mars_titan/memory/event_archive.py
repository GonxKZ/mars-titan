"""Codificar y verificar eventos financieros por sesión en un archivo Arrow IPC acotado."""

import hashlib
import os
import shutil
import struct
from collections import deque
from contextlib import closing
from itertools import islice

import numpy as np
import pyarrow as pa

from mars_titan.data.cohort_files import safe_destination
from mars_titan.data.input_policy import MODALITIES
from mars_titan.data.storage import outside_source, sha256
from mars_titan.memory.financial_observations import ObservationEvent
from mars_titan.models.titans.financial_inputs import validated_cpu_batches
from mars_titan.training.input_pipeline import prepared

SCHEMA = pa.schema(
    [
        ("at", pa.int64()),
        ("closed", pa.bool_()),
        ("n", pa.int32()),
        ("ids", pa.list_(pa.string())),
        ("markets", pa.list_(pa.string())),
        ("available", pa.binary()),
        ("prediction", pa.binary()),
        ("presence", pa.binary()),
        *((name, pa.binary()) for name in MODALITIES),
        ("label_flows", pa.list_(pa.string())),
        ("label_at", pa.list_(pa.int64())),
        ("label_values", pa.list_(pa.float64())),
    ]
)
BLOCK_FIELDS = {"sample_ids", "market", "input_available_at", "prediction_at", "presence", "inputs"}
# Cota de una copia. La vista de EE. UU. (4.198 activos, ~7 KiB de entradas por fila) ocupa del
# orden de 50 a 100 GB comprimida en las ventanas tardías: con 16 GiB la DGX volvía a Parquet en
# cada época. La reserva de disco de training_event_cache sigue protegiendo el volumen.
MAX_BYTES = 192 * 1024**3


class ArchiveBudgetError(ValueError):
    """La copia opcional supera su presupuesto de disco."""


class InvalidArchive(ValueError):
    """La copia local ya no corresponde a su recibo."""


def _input_bytes(block):
    if block is None:
        return dict(
            n=0,
            ids=[],
            markets=[],
            **dict.fromkeys(("available", "prediction", "presence", *MODALITIES), b""),
        )
    if set(block) != BLOCK_FIELDS or set(block["inputs"]) != set(MODALITIES):
        raise ValueError("El bloque contiene otros campos de entrada")
    return dict(
        n=len(block["sample_ids"]),
        ids=block["sample_ids"],
        markets=block["market"],
        available=block["input_available_at"].astype("<M8[us]", copy=False).tobytes(),
        prediction=block["prediction_at"].astype("<M8[us]", copy=False).tobytes(),
        presence=block["presence"].tobytes(),
        **{name: block["inputs"][name].astype("<f4", copy=False).tobytes() for name in MODALITIES},
    )


def _encode(event):
    labels = tuple(zip(*event.labels, strict=True)) if event.labels else ([], [], [])
    rows = []
    for index, block in enumerate(event.inputs or (None,)):
        flows, decisions, targets = labels if index == 0 else ([], [], [])
        rows.append(
            dict(
                at=event.at,
                closed=event.close_phase,
                **_input_bytes(block),
                label_flows=flows,
                label_at=decisions,
                label_values=targets,
            )
        )
    return pa.RecordBatch.from_pylist(rows, schema=SCHEMA)


def _decode(batch, spec, block_rows, at, stored_rows):
    first, *rest = batch.to_pylist()
    labels = tuple(zip(first["label_flows"], first["label_at"], first["label_values"], strict=True))
    blocks = []
    for i, row in enumerate((first, *rest)):
        n = row["n"]
        if (
            row["at"] != at
            or row["closed"] != first["closed"]
            or (i and any(row[k] for k in ("label_flows", "label_at", "label_values")))
        ):
            raise ValueError("El lote mezcla instantes o repite etiquetas")
        if not 0 <= n <= stored_rows or len(row["ids"]) != n or len(row["markets"]) != n:
            raise ValueError("La longitud del bloque ha cambiado")
        if n == 0:
            continue
        inputs = {}
        for name in MODALITIES:
            shape = (
                (n, spec.context, spec.dimensions[name])
                if name == "prices"
                else (n, spec.dimensions[name])
            )
            inputs[name] = (
                np.frombuffer(row[name], dtype="<f4").astype(np.float32, copy=False).reshape(shape)
            )
        dates = np.frombuffer(row["prediction"], dtype="<M8[us]").astype(
            "datetime64[us]", copy=False
        )
        availability = np.frombuffer(row["available"], dtype="<M8[us]").astype(
            "datetime64[us]", copy=False
        )
        presence = np.frombuffer(row["presence"], dtype=np.bool_).reshape(n, len(MODALITIES))
        for begin in range(0, n, block_rows):
            part = slice(begin, begin + block_rows)
            blocks.append(
                dict(
                    sample_ids=row["ids"][part],
                    market=row["markets"][part],
                    prediction_at=dates[part],
                    input_available_at=availability[part],
                    presence=presence[part],
                    inputs={name: values[part] for name, values in inputs.items()},
                )
            )
    return ObservationEvent(at, tuple(blocks), labels, first["closed"])


def _original(source, start, count, block):
    reference = getattr(source, "reference_batched_events", source.batched_events)
    with closing(reference(start_cursor=start, block_rows=block)) as events:
        yield from islice(events, count)
    source._confirm()


def write(source, spec, path, *, start, count, block, max_bytes, verify_rows=False, reserve=0):
    """Guardar un recorrido acotado en un archivo nuevo y devolver su recibo."""
    if not (
        0 <= start < start + count <= len(source.metadata["groups"]) and 0 < max_bytes <= MAX_BYTES
    ):
        raise ValueError("El tramo o el presupuesto de disco no es válido")
    safe_destination(path)
    for root in (*source.dataset.roots.values(), source.path.parent, source.dataset.path.parent):
        outside_source(root, path)
    options = pa.ipc.IpcWriteOptions(compression="lz4", use_threads=False)
    reference = None
    with path.open("xb") as stream:
        try:
            with (
                pa.ipc.new_file(stream, SCHEMA, options=options) as writer,
                closing(_original(source, start, count, block)) as events,
            ):

                def writing():
                    check_at = 0
                    for event in events:
                        writer.write_batch(_encode(event))
                        size = stream.tell()
                        if size > max_bytes:
                            raise ArchiveBudgetError("La copia supera el presupuesto de disco")
                        if reserve and size >= check_at:
                            if shutil.disk_usage(path.parent).free < reserve:
                                raise ArchiveBudgetError("La copia agotaría la reserva de disco")
                            check_at = size + 64 * 1024**2
                        yield event

                if verify_rows:
                    reference = fingerprint(
                        writing(), spec, source.dataset.pipeline.prefetch_batches
                    )
                else:
                    deque(writing(), maxlen=0)
            if stream.tell() > max_bytes:
                raise ArchiveBudgetError("El archivo completo supera el presupuesto de disco")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    stamp = dict(
        dataset=source.dataset.identity,
        index=source.identity,
        spec=spec.fingerprint(),
        start=start,
        events=count,
        block=block,
        bytes=path.stat().st_size,
        sha256=sha256(path),
    )
    if verify_rows:
        actual = fingerprint(
            replay(source, spec, path, stamp, cursor=start, block=block),
            spec,
            source.dataset.pipeline.prefetch_batches,
        )
        if actual != reference:
            raise InvalidArchive("La copia no reproduce las entradas y etiquetas del origen")
        stamp["verified"] = actual
    return stamp


def _signature(path):
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_ctime_ns


def verified_signature(path, stamp):
    safe_destination(path)
    before = _signature(path)
    if (
        not 0 < stamp["bytes"] == before[2] <= MAX_BYTES
        or sha256(path) != stamp["sha256"]
        or _signature(path) != before
    ):
        raise InvalidArchive("Ha cambiado la huella del archivo de trabajo")
    return before


def replay(source, spec, path, stamp, *, cursor, block, signature=None):
    """Reproducir eventos verificados con el tamaño de bloque que pide el consumidor."""
    if (
        block <= 0
        or stamp["block"] % block
        or not stamp["start"] <= cursor <= stamp["start"] + stamp["events"]
    ):
        raise ValueError("El cursor o el tamaño de bloque no corresponde a esta copia")
    safe_destination(path)
    if (stamp["dataset"], stamp["index"], stamp["spec"]) != (
        source.dataset.identity,
        source.identity,
        spec.fingerprint(),
    ):
        raise ValueError("La copia pertenece a otro origen")
    before = verified_signature(path, stamp) if signature is None else signature
    source._confirm()
    with pa.memory_map(str(path), "r") as stream:
        reader = pa.ipc.open_file(stream, options=pa.ipc.IpcReadOptions(use_threads=False))
        if reader.num_record_batches != stamp["events"] or reader.schema != SCHEMA:
            raise ValueError("El esquema o el número de eventos ha cambiado")
        for i in range(cursor - stamp["start"], stamp["events"]):
            at, count = source.metadata["groups"][stamp["start"] + i]
            event = _decode(reader.get_batch(i), spec, block, at, stamp["block"])
            if _signature(path) != before:
                raise InvalidArchive("El archivo cambió durante la lectura")
            if (
                sum(len(b["sample_ids"]) for b in event.inputs)
                + len(event.labels)
                + event.close_phase
                != count
            ):
                raise ValueError("El número de filas no coincide con el índice")
            yield event
    if _signature(path) != before:
        raise InvalidArchive("El archivo cambió durante la lectura")
    source._confirm()


def fingerprint(events, spec, depth=0):
    """Identificar todas las entradas validadas y las etiquetas en su orden de maduración."""
    digest = hashlib.sha256()
    rows = labels = 0
    with closing(prepared(events, lambda e: validated_cpu_batches(e.inputs, spec), depth)) as items:
        for event, task in items:
            digest.update(
                struct.pack(
                    "<q?II", event.at, event.close_phase, len(event.inputs), len(event.labels)
                )
            )
            for batch in task.result():
                rows += len(batch.sample_ids)
                digest.update(bytes.fromhex(batch.input_digest))
            if event.labels:
                flows, moments, targets = zip(*event.labels, strict=True)
                digest.update(b"\0".join(flow.encode() for flow in flows) + b"\0")
                digest.update(np.asarray(moments, dtype="<i8").tobytes())
                digest.update(np.asarray(targets, dtype="<f8").tobytes())
                labels += len(targets)
    return dict(rows=rows, labels=labels, digest=digest.hexdigest())
