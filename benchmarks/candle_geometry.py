"""Medir representaciones sobre las mismas ventanas, sin aprendizaje.

`measure` compara geometría y PNG con tinta en CPU. CUDA se solicita explícitamente y añade
ResNet18 con la reserva GPU existente. `compare` coteja por lotes las mismas filas y entradas
restantes, incluida la máscara por sesión. No prepara objetivos ni inicia ajustes.
"""

import argparse
import hashlib
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
import pyarrow.parquet as pq

from mars_titan.data.candle_geometry import geometry_chart, geometry_spec
from mars_titan.data.charts import chart_png
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.pretraining_free_encoders import chart_vectors
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.hardware.platform_identity import cpu_name

MAX_BYTES = 16 * 1024**2


def _input_configuration(configuration):
    # El contraste solo cambia imágenes. La caché y las huellas de código quedan registradas
    # por separado, mientras el contrato efectivo de texto y el de las entradas deben coincidir.
    result = {k: v for k, v in configuration.items() if k not in {"code", "cache_charts"}}
    result["encoders"] = {
        k: v
        for k, v in configuration["encoders"].items()
        if k not in {"image_model", "image_policy", "image_url", "image_weights_sha256"}
    }
    return result


def _compare_samples(paths):
    with pq.ParquetFile(paths[0]) as before, pq.ParquetFile(paths[1]) as after:
        if (
            before.schema_arrow != after.schema_arrow
            or before.metadata.num_rows != after.metadata.num_rows
        ):
            raise ValueError("Las representaciones no conservan las mismas filas y columnas")
        for file in (before, after):
            if file.metadata.num_rows > 200_000 or any(
                file.metadata.row_group(i).total_byte_size > 64 * 1024**2
                for i in range(file.num_row_groups)
            ):
                raise ValueError("Las muestras superan el presupuesto de comparación")
        columns = [
            name for name in before.schema_arrow.names if name not in {"charts", "chart_hash"}
        ]
        streams = [
            file.iter_batches(batch_size=128, columns=columns, use_threads=False)
            for file in (before, after)
        ]
        count = 0
        for left, right in zip(*streams, strict=True):
            if max(left.nbytes, right.nbytes) > MAX_BYTES:
                raise ValueError("El lote decodificado supera el presupuesto de comparación")
            if not left.equals(right):
                raise ValueError("Han cambiado filas, orden o entradas distintas de las velas")
            count += left.num_rows
        return count


def compare_representations(previous, current, output):
    """Comparar las mismas muestras y su preparación, sin reconstruir ventanas sin máscara."""
    roots = (Path(previous), Path(current))
    output = Path(output)
    for root in roots:
        outside_source(root, output)
    configurations = [read_manifest(root / "configuration.json") for root in roots]
    if _input_configuration(configurations[0][0]) != _input_configuration(configurations[1][0]):
        raise ValueError("La comparación necesita la misma preparación y los mismos contratos")
    manifests = [read_manifest(root / "manifest.json") for root in roots]
    if any(manifest[0].get("cohort_complete") is not True for manifest in manifests):
        raise ValueError("La comparación necesita dos ediciones completas")
    assets = [sorted((a["market"], a["symbol"]) for a in item[0]["assets"]) for item in manifests]
    if assets[0] != assets[1] or len(set(assets[0])) != len(assets[0]):
        raise ValueError("Las representaciones no conservan la misma población de activos")
    count = 0
    for market, symbol in assets[0]:
        folders = [root / "samples" / market / symbol for root in roots]
        for root, folder in zip(roots, folders, strict=True):
            safe_destination(folder)
            if not folder.resolve().is_relative_to(root.resolve()):
                raise ValueError("Un activo queda fuera de la edición")
        receipts = [read_manifest(folder / "manifest.json")[0] for folder in folders]
        paths = [folder / "samples.parquet" for folder in folders]
        expected = [receipt["samples_sha256"] for receipt in receipts]
        if [sha256(path) for path in paths] != expected:
            raise ValueError("Las muestras no coinciden con sus huellas confirmadas")
        count += _compare_samples(paths)
        if [sha256(path) for path in paths] != expected:
            raise ValueError("Las muestras han cambiado durante la comparación")
    if any(count != item[0]["samples"] for item in manifests):
        raise ValueError("El recuento leído no coincide con el manifiesto")
    result = dict(
        schema_version=1,
        assets=len(assets[0]),
        samples=count,
        same_inputs=True,
        different_columns_allowed=["charts", "chart_hash"],
        configuration_sha256=[item[1] for item in configurations],
        manifest_sha256=[item[1] for item in manifests],
        batch_rows=128,
        max_batch_bytes=MAX_BYTES,
    )
    safe_destination(output)
    atomic_json(output / "summary.json", result)
    return result


def windows(samples, path=None):
    """Hasta 256 ventanas y 16 MiB sin comprimir. El archivo no acredita disponibilidad."""
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
        content, _ = geometry_chart(value, end_index=len(value) - 1, present=mask)
        digest.update(content.encode())
    return records, dict(
        **source,
        samples=len(records),
        context_sessions=64,
        absent_slots=int((~present).sum()),
        input_bytes=values.nbytes + present.nbytes,
        inputs_sha256=digest.hexdigest(),
    )


def _geometry(records):
    return [
        geometry_chart(value, end_index=len(value) - 1, present=mask) for value, mask in records
    ]


def _images(records, encoder=None):
    results = []
    for offset in range(0, len(records), 8):
        pngs = [
            chart_png(value, end_index=len(value) - 1, present=mask)
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


def measure(*, samples=64, repeats=9, warmup=2, device="cpu", windows_path=None):
    """Tiempo incluye dibujo, representación y hash de entrada. El hash de salida queda fuera."""
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
    records, workload = windows(samples, windows_path)
    variants = {"geometry": _geometry, "ink": _images}
    gpu, peak = None, None
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
            variants["resnet"] = partial(_images, encoder=encoder)
            synchronize = partial(torch.cuda.synchronize, 0)
            gpu = admitted.record
        results = _measure(variants, records, repeats, warmup, synchronize)
        if device == "cuda":
            admitted.check()
            peak = dict(
                allocated=torch.cuda.max_memory_allocated(0),
                reserved=torch.cuda.max_memory_reserved(0),
            )
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
    cost.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--previous", type=Path, required=True)
    comparison.add_argument("--current", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "measure":
        result = measure(
            samples=args.samples,
            repeats=args.repeats,
            warmup=args.warmup,
            device=args.device,
            windows_path=args.windows,
        )
        atomic_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    result = compare_representations(args.previous, args.current, args.output)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
