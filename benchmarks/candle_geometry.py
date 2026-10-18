"""Medir representaciones sobre las mismas ventanas, sin aprendizaje.

`measure` compara geometría y PNG con tinta en CPU. CUDA se solicita explícitamente y añade
ResNet18 con la reserva GPU existente. `compare` coteja por lotes las mismas filas y entradas
restantes, incluida la máscara por sesión. No prepara objetivos ni inicia ajustes.
"""

import argparse
import hashlib
import importlib
import io
import json
import platform
import resource
import time
import zipfile
from contextlib import nullcontext
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import PIL

from mars_titan.data.candle_geometry import geometry_chart, geometry_spec
from mars_titan.data.candle_geometry_edition import compare_representations
from mars_titan.data.charts import chart_png
from mars_titan.data.pretraining_free_encoders import chart_vectors
from mars_titan.data.prices import check_ordering_rtol
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.hardware.platform_identity import cpu_name

MAX_BYTES = 16 * 1024**2


def code_identity():
    names = (
        "mars_titan.data.candle_geometry",
        "mars_titan.data.charts",
        "mars_titan.data.pretraining_free_encoders",
        "mars_titan.data.embeddings",
        "mars_titan.data.prices",
    )
    return dict(
        benchmark=sha256(Path(__file__)),
        **{name: sha256(Path(importlib.import_module(name).__file__)) for name in names},
    )


def windows(samples, path=None, *, ordering_rtol=0.0):
    """Hasta 256 ventanas y 16 MiB sin comprimir. El archivo no acredita disponibilidad."""
    check_ordering_rtol(ordering_rtol)
    if type(samples) is not int or not 1 <= samples <= 256:
        raise ValueError("El número de ventanas debe pertenecer a [1, 256]")
    if path is None:
        random = np.random.default_rng(665)
        opening = np.exp(random.normal(3, 0.1, (samples, 64)))
        close = opening * np.exp(random.normal(0, 0.02, opening.shape))
        values = np.stack(
            (opening, np.maximum(opening, close) * 1.03, np.minimum(opening, close) * 0.97, close),
            axis=-1,
        )
        values[::4] = 10.0
        present = np.ones((samples, 64), dtype=bool)
        present[1::3, 10] = False
        values[~present] = 0.0
        source = dict(kind="synthetic_technical_fixture", seed=665)
    else:
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
            raise ValueError("Las ventanas necesitan un archivo regular de hasta 16 MiB")
        with path.open("rb") as stream:
            payload = stream.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            raise ValueError("El archivo de ventanas supera 16 MiB")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if (
                len(entries) != 2
                or {entry.filename for entry in entries} != {"windows.npy", "presence.npy"}
                or sum(entry.file_size for entry in entries) > MAX_BYTES
            ):
                raise ValueError("El NPZ no conserva dos arrays acotados de ventanas y presencia")
        with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
            values, present = stored["windows"], stored["presence"]
        source = dict(kind="provided_windows", source_sha256=hashlib.sha256(payload).hexdigest())
    if (
        values.ndim != 3
        or values.shape[1:] != (64, 4)
        or not 1 <= len(values) <= 256
        or values.dtype.kind not in "fiu"
        or present.shape != values.shape[:2]
        or present.dtype != np.bool_
    ):
        raise ValueError("Se requieren de 1 a 256 ventanas OHLC de 64 sesiones y sus máscaras")
    if (values[~present] != 0).any():
        raise ValueError("Las posiciones ausentes deben conservar relleno cero")
    records = [(value[mask], mask) for value, mask in zip(values, present, strict=True)]
    digest = hashlib.sha256()
    for value, mask in records:
        content, _ = geometry_chart(
            value, end_index=len(value) - 1, present=mask, ordering_rtol=ordering_rtol
        )
        digest.update(content.encode())
    return records, dict(
        **source,
        samples=len(records),
        context_sessions=64,
        absent_slots=int((~present).sum()),
        input_bytes=values.nbytes + present.nbytes,
        inputs_sha256=digest.hexdigest(),
        ordering_rtol=ordering_rtol,
    )


def _geometry(records, *, ordering_rtol=0.0):
    return [
        geometry_chart(value, end_index=len(value) - 1, present=mask, ordering_rtol=ordering_rtol)
        for value, mask in records
    ]


def _images(records, encoder=None, *, ordering_rtol=0.0):
    results = []
    for offset in range(0, len(records), 8):
        pngs = [
            chart_png(value, end_index=len(value) - 1, present=mask, ordering_rtol=ordering_rtol)
            for value, mask in records[offset : offset + 8]
        ]
        vectors = chart_vectors(pngs) if encoder is None else encoder.images(pngs)
        results.extend(
            (hashlib.sha256(png).hexdigest(), vector)
            for png, vector in zip(pngs, vectors, strict=True)
        )
    return results


def _fingerprint(result):
    digest = hashlib.sha256()
    for content, vector in result:
        if vector.shape != (512,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
            raise ValueError("El codificador no conserva un vector FP32 finito de 512 valores")
        digest.update(content.encode())
        digest.update(vector.tobytes())
    return digest.hexdigest()


def _measure(variants, records, repeats, warmup, synchronize):
    hashes, costs = {}, {name: [] for name in variants}
    for name, run in variants.items():
        hashes[name] = _fingerprint(run(records))
        for _ in range(warmup):
            run(records)
    for repetition in range(repeats):
        order = list(variants) if repetition % 2 == 0 else list(reversed(variants))
        for name in order:
            if synchronize is not None:
                synchronize()
            started = time.perf_counter()
            result = variants[name](records)
            if synchronize is not None:
                synchronize()
            costs[name].append(time.perf_counter() - started)
            if _fingerprint(result) != hashes[name]:
                raise ValueError("El benchmark no reproduce los mismos bits entre repeticiones")
    return {
        name: dict(
            seconds=seconds,
            batch_milliseconds={
                f"p{q}": float(np.percentile(seconds, q) * 1000) for q in (50, 95, 99)
            },
            mean_milliseconds_per_window=float(np.mean(seconds) * 1000 / len(records)),
            rows_per_second=float(len(records) / np.median(seconds)),
            output_sha256=hashes[name],
            reproducible=True,
        )
        for name, seconds in costs.items()
    }


def measure(*, samples=64, repeats=9, warmup=2, device="cpu", windows_path=None, ordering_rtol=0.0):
    """Tiempo incluye dibujo, representación y hash de entrada. El hash de salida queda fuera."""
    check_ordering_rtol(ordering_rtol)
    if (
        type(samples) is not int
        or not 1 <= samples <= 256
        or type(repeats) is not int
        or not 1 <= repeats <= 100
        or type(warmup) is not int
        or not 0 <= warmup <= 10
        or device not in ("cpu", "cuda")
    ):
        raise ValueError(
            "El lote, las repeticiones, el calentamiento o el dispositivo no son válidos"
        )
    started = time.perf_counter()
    sources = code_identity()
    records, workload = windows(samples, windows_path, ordering_rtol=ordering_rtol)
    variants = {
        "geometry": partial(_geometry, ordering_rtol=ordering_rtol),
        "ink": partial(_images, ordering_rtol=ordering_rtol),
    }
    gpu, peak = None, None
    encoder_identity = None
    synchronize = None
    lease = nullcontext()
    if device == "cuda":
        from mars_titan.training.experiment_resources import GpuLease

        lease = GpuLease()
    with lease as admitted:
        if device == "cuda":
            import torch

            from mars_titan.data.embeddings import FrozenEncoders

            torch.cuda.reset_peak_memory_stats(0)
            encoder = FrozenEncoders(image_batch_size=8, word_embedding_placement="cpu")
            encoder_identity = encoder.spec
            variants["resnet"] = partial(_images, encoder=encoder, ordering_rtol=ordering_rtol)
            synchronize = partial(torch.cuda.synchronize, 0)
            gpu = admitted.record
        results = _measure(variants, records, repeats, warmup, synchronize)
        if device == "cuda":
            admitted.check()
            peak = dict(
                allocated=torch.cuda.max_memory_allocated(0),
                reserved=torch.cuda.max_memory_reserved(0),
            )
    if code_identity() != sources:
        raise ValueError("El código cambió durante la medida")
    return dict(
        schema_version=1,
        measured_at=datetime.now(UTC).isoformat(),
        workload=workload,
        geometry=geometry_spec(),
        output_bytes_per_row=512 * 4,
        optimizer_steps=0,
        hardware=dict(cpu=cpu_name(), platform=platform.platform()),
        versions=dict(
            python=platform.python_version(), numpy=np.__version__, pillow=PIL.__version__
        ),
        repetitions=repeats,
        warmup=warmup,
        reference_passes_per_variant=1,
        image_batch_size=8,
        results=results,
        cuda=gpu,
        cuda_framework_peak_bytes=peak,
        encoder=encoder_identity,
        source_sha256=sources,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        measure_total_seconds=time.perf_counter() - started,
        benchmark_sha256=sha256(Path(__file__)),
        unmeasured=["energy", "monetary_cost", "transfer_bytes", "predictive_quality"],
        limits=[
            "Los percentiles corresponden al lote completo entre repeticiones, no a cada fila.",
            "El pico RSS pertenece al proceso, no a cada representación por separado.",
            "La preparación CUDA incluye el codificador congelado de texto, aunque aquí no se usa.",
            "La procedencia temporal de las ventanas requiere otra auditoría.",
        ],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cost = commands.add_parser("measure")
    cost.add_argument("--samples", type=int, default=64)
    cost.add_argument("--windows", type=Path)
    cost.add_argument("--repeats", type=int, default=9)
    cost.add_argument("--warmup", type=int, default=2)
    cost.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    cost.add_argument("--ordering-rtol", type=float, default=0.0)
    cost.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--previous", type=Path, required=True)
    comparison.add_argument("--current", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "measure":
        if args.device == "cuda":
            from mars_titan.data.embeddings import strict_fp32

            strict_fp32()
        result = measure(
            samples=args.samples,
            repeats=args.repeats,
            warmup=args.warmup,
            device=args.device,
            windows_path=args.windows,
            ordering_rtol=args.ordering_rtol,
        )
        atomic_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    result = compare_representations(args.previous, args.current, args.output)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
