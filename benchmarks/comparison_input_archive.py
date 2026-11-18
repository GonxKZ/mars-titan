"""Comparar el lector cronológico con la repetición de las mismas entradas desde Arrow IPC."""

import argparse
import hashlib
import json
import resource
import struct
import tempfile
import time
from contextlib import closing
from itertools import islice
from pathlib import Path

import numpy as np
import pyarrow as pa

from mars_titan.data.cohort_files import safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES
from mars_titan.data.storage import outside_source, sha256
from mars_titan.memory.financial_observations import FinancialObservationSource, ObservationEvent
from mars_titan.models.titans.financial_inputs import validated_cpu_batches
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.input_pipeline import background, prepared

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
MAX_BYTES = 16 * 1024**3


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
    with closing(source.batched_events(start_cursor=start, block_rows=block)) as events:
        yield from islice(events, count)
    source._confirm()


def write(source, spec, path, *, start, count, block, max_bytes):
    """Guardar un recorrido acotado en un archivo nuevo y devolver su recibo."""
    if not (
        0 <= start < start + count <= len(source.metadata["groups"]) and 0 < max_bytes <= MAX_BYTES
    ):
        raise ValueError("El tramo o el presupuesto de disco no es válido")
    safe_destination(path)
    for root in (*source.dataset.roots.values(), source.path.parent, source.dataset.path.parent):
        outside_source(root, path)
    options = pa.ipc.IpcWriteOptions(compression="lz4", use_threads=False)
    with path.open("xb") as stream:
        try:
            with (
                pa.ipc.new_file(stream, SCHEMA, options=options) as writer,
                closing(_original(source, start, count, block)) as events,
            ):
                for event in events:
                    writer.write_batch(_encode(event))
                    if stream.tell() > max_bytes:
                        raise ValueError("La copia supera el presupuesto de disco")
            if stream.tell() > max_bytes:
                raise ValueError("El archivo completo supera el presupuesto de disco")
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    return dict(
        dataset=source.dataset.identity,
        index=source.identity,
        spec=spec.fingerprint(),
        start=start,
        events=count,
        block=block,
        bytes=path.stat().st_size,
        sha256=sha256(path),
    )


def replay(source, spec, path, stamp, *, cursor, block):
    """Reproducir eventos verificados con el tamaño de bloque que pide el consumidor."""
    if (
        block <= 0
        or stamp["block"] % block
        or not stamp["start"] <= cursor <= stamp["start"] + stamp["events"]
    ):
        raise ValueError("El cursor o el tamaño de bloque no corresponde a esta copia")
    safe_destination(path)
    before = path.stat()
    if (stamp["dataset"], stamp["index"], stamp["spec"]) != (
        source.dataset.identity,
        source.identity,
        spec.fingerprint(),
    ):
        raise ValueError("La copia pertenece a otro origen")
    if not 0 < stamp["bytes"] == before.st_size <= MAX_BYTES or sha256(path) != stamp["sha256"]:
        raise ValueError("Ha cambiado la huella del archivo de trabajo")
    source._confirm()
    with pa.memory_map(str(path), "r") as stream:
        reader = pa.ipc.open_file(stream, options=pa.ipc.IpcReadOptions(use_threads=False))
        if reader.num_record_batches != stamp["events"] or reader.schema != SCHEMA:
            raise ValueError("El esquema o el número de eventos ha cambiado")
        for i in range(cursor - stamp["start"], stamp["events"]):
            at, count = source.metadata["groups"][stamp["start"] + i]
            event = _decode(reader.get_batch(i), spec, block, at, stamp["block"])
            if (
                sum(len(b["sample_ids"]) for b in event.inputs)
                + len(event.labels)
                + event.close_phase
                != count
            ):
                raise ValueError("El número de filas no coincide con el índice")
            yield event
    end = path.stat()
    if (before.st_dev, before.st_ino, before.st_ctime_ns) != (
        end.st_dev,
        end.st_ino,
        end.st_ctime_ns,
    ):
        raise ValueError("El archivo cambió durante la lectura")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument(
        "work", type=Path, help="Directorio existente en un disco con espacio para la copia"
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--events", type=int, help="Por defecto, hasta el cierre de la fase")
    parser.add_argument("--block-rows", type=int, default=1024)
    args = parser.parse_args()
    dataset = CorpusDataset(args.view, input_policy=HISTORICAL_MASKED)
    source = FinancialObservationSource(dataset, args.index)
    if source.phase.partition not in {"train", "validation"}:
        raise ValueError("El ensayo necesita ajuste o validación de desarrollo")
    spec = source.specification()
    count = len(source.metadata["groups"]) - args.start if args.events is None else args.events
    depth = dataset.pipeline.prefetch_batches
    with tempfile.TemporaryDirectory(dir=args.work) as temp:
        path = Path(temp) / "events.arrow"
        before = time.perf_counter()
        stamp = write(
            source,
            spec,
            path,
            start=args.start,
            count=count,
            block=args.block_rows,
            max_bytes=MAX_BYTES,
        )
        report = dict(archive=stamp, build_seconds=time.perf_counter() - before, passes=[])
        for mode in ("original", "preparada", "preparada", "original"):
            before = time.perf_counter()
            if mode == "original":
                events = _original(source, args.start, count, args.block_rows)
            else:
                events = replay(source, spec, path, stamp, cursor=args.start, block=args.block_rows)
                if depth:
                    events = background(events, depth)
            measured = fingerprint(events, spec, depth)
            seconds = time.perf_counter() - before
            if report["passes"] and report["passes"][0]["data"] != measured:
                raise ValueError("Las filas, etiquetas o entradas no coinciden")
            report["passes"].append(
                dict(
                    mode=mode,
                    seconds=seconds,
                    data=measured,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                )
            )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
