"""Medir lectura y construcción de lotes sobre un corpus técnico reproducible."""

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import time
import types
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.storage import sha256
from mars_titan.training import corpus_inputs

DIMENSIONS = dict(news=384, charts=512, fundamentals=45, macro=420)
READER_PATH = "src/mars_titan/training/corpus_inputs.py"
CASES = {"small": (4, 64, 16, 1), "dense": (16, 1024, 256, 1), "sparse": (16, 1024, 256, 16)}


def prepare_corpus(output, *, assets, rows, group_rows, stride):
    """Generar todas las filas y marcar las exclusiones sin alterar sus posiciones."""
    if (
        any(type(v) is not int or v < 1 for v in (assets, rows, group_rows, stride))
        or assets > 128
        or rows > 8192
        or assets * rows > 131_072
        or group_rows > rows
        or stride > rows
    ):
        raise ValueError("La carga supera los límites del benchmark")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    roots = {name: output / name for name in ("prepared", "samples", "labels")}
    context, entries = 64, []
    price_times = np.datetime64("2010-01-01T16:00", "us") + np.arange(
        rows + context
    ) * np.timedelta64(1, "m")
    ends = np.arange(context - 1, context - 1 + rows)
    moments = price_times[ends] + np.timedelta64(5, "m")
    accepted = np.arange(rows) % stride == 0
    timestamp = pa.timestamp("us", tz="UTC")
    for asset in range(assets):
        symbol, rng = f"A{asset:04d}", np.random.default_rng(42 + asset)
        folders = {name: root / "US" / symbol for name, root in roots.items()}
        for folder in folders.values():
            folder.mkdir(parents=True)
        close = 100 + asset + np.arange(rows + context) / 100
        prices = pa.table(
            dict(
                open=close - 0.1,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=rng.uniform(100, 10_000, len(close)),
                available_at=pa.array(price_times.astype(np.int64), type=timestamp),
            )
        )
        available = pa.array((moments - np.timedelta64(1, "m")).astype(np.int64), type=timestamp)
        samples = {
            "prediction_at": pa.array(moments.astype(np.int64), type=timestamp),
            "price_end_index": ends,
            "input_availability": pa.StructArray.from_arrays(
                [available] * 5, names=["prices", *DIMENSIONS]
            ),
        }
        for name, width in DIMENSIONS.items():
            values = rng.standard_normal((rows, width), dtype=np.float32)
            samples[name] = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), width)
        labels = pa.table(
            dict(
                sample_row=np.arange(rows),
                prediction_at=pa.array(moments.astype(np.int64), type=timestamp),
                target_available_at=pa.array(
                    (moments + np.timedelta64(1, "m")).astype(np.int64),
                    type=timestamp,
                    mask=~accepted,
                ),
                target=pa.array(rng.normal(0, 0.01, rows), mask=~accepted),
                partition=["train" if valid else None for valid in accepted],
                reason=["accepted" if valid else "insufficient_history" for valid in accepted],
            )
        )
        paths = {
            "prices": folders["prepared"] / "prices.parquet",
            "samples": folders["samples"] / "samples.parquet",
            "labels": folders["labels"] / "labels.parquet",
        }
        for name, table in (("prices", prices), ("samples", pa.table(samples)), ("labels", labels)):
            pq.write_table(table, paths[name], row_group_size=group_rows)
        entries.append(
            dict(
                market="US",
                symbol=symbol,
                counts=dict(train=int(accepted.sum()), validation=0),
                **{name + "_sha256": sha256(path) for name, path in paths.items()},
            )
        )
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                schema_version=1,
                kind="corpus_supervision",
                domain="technical",
                scope="development_snapshot",
                cohort_complete=False,
                context_sessions=context,
                roots={key: str(path) for key, path in roots.items()},
                assets=entries,
                counts=dict(train=assets * int(accepted.sum()), validation=0),
                final_test_opened=False,
            ),
            sort_keys=True,
        )
    )
    return manifest


def reader(revision):
    if revision == "current":
        return corpus_inputs.CorpusDataset, sha256(Path(corpus_inputs.__file__))
    if not re.fullmatch(r"[a-f0-9]{7,40}", revision):
        raise ValueError("La referencia necesita un commit hexadecimal fijo")
    source = subprocess.check_output(["git", "show", f"{revision}:{READER_PATH}"])
    module = types.ModuleType("mars_titan.training._corpus_reader_reference")
    module.__package__ = "mars_titan.training"
    module.__file__ = f"{revision}:{READER_PATH}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.CorpusDataset, hashlib.sha256(source).hexdigest()


def traverse(dataset, batch_size, *, fingerprint):
    digest, rows, batches, size = hashlib.sha256(), 0, 0, 0
    for batch in dataset.batches(partition="train", batch_size=batch_size, epoch=2, seed=42):
        rows += len(batch["target"])
        batches += 1
        for name in (*DIMENSIONS, "prices"):
            values = batch["inputs"][name]
            if not values.flags.c_contiguous:
                raise ValueError("El lector no entrega lotes contiguos")
            size += values.nbytes
            if fingerprint:
                digest.update(name.encode())
                digest.update(values.tobytes())
        if fingerprint:
            for name in (
                "target",
                "weight",
                "prediction_at",
                "target_available_at",
                "input_available_at",
            ):
                digest.update(name.encode())
                digest.update(batch[name].tobytes())
            digest.update(
                json.dumps(
                    {name: batch[name] for name in ("sample_ids", "market", "confirmed_cursor")},
                    sort_keys=True,
                ).encode()
            )
    if rows != dataset.manifest["counts"]["train"]:
        raise ValueError("La medida no ha recorrido todas las filas admitidas")
    return dict(rows=rows, batches=batches, input_bytes=size, stream_sha256=digest.hexdigest())


def peak_rss():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("No se ha podido medir VmHWM")


def measure(manifest, *, batch_size, revision):
    implementation, code_hash = reader(revision)
    started = time.perf_counter()
    dataset = implementation(manifest)
    initialization = time.perf_counter() - started
    warmup_started = time.perf_counter()
    checked = traverse(dataset, batch_size, fingerprint=True)
    warmup = time.perf_counter() - warmup_started
    started = time.perf_counter()
    timed = traverse(dataset, batch_size, fingerprint=False)
    elapsed = time.perf_counter() - started
    if any(timed[key] != checked[key] for key in ("rows", "batches", "input_bytes")):
        raise ValueError("Las pasadas no recorren el mismo volumen")
    return dict(
        **checked,
        initialization_seconds=initialization,
        reader_seconds=elapsed,
        initialization_plus_reader_seconds=initialization + elapsed,
        warmup_and_fingerprint_seconds=warmup,
        samples_per_second=checked["rows"] / elapsed,
        peak_rss_bytes=peak_rss(),
        source_sha256=dataset.identity,
        code_sha256=code_hash,
    )


def profile_reader(manifest, *, batch_size, revision):
    """Perfilar otra pasada con reloj de CPU, sin mezclarla con el tiempo sin instrumentar."""
    import cProfile
    import pstats

    implementation, code_hash = reader(revision)
    dataset = implementation(manifest)
    traverse(dataset, batch_size, fingerprint=False)
    profiler = cProfile.Profile(timer=time.process_time)
    profiler.enable()
    volume = traverse(dataset, batch_size, fingerprint=False)
    profiler.disable()
    statistics = pstats.Stats(profiler)
    functions = []
    for (filename, line, name), (_, calls, own, cumulative, _) in sorted(
        statistics.stats.items(), key=lambda item: item[1][3], reverse=True
    )[:25]:
        functions.append(
            dict(
                file=Path(filename).name,
                line=line,
                function=name,
                calls=calls,
                self_seconds=own,
                cumulative_seconds=cumulative,
            )
        )
    return dict(
        rows=volume["rows"],
        batches=volume["batches"],
        code_sha256=code_hash,
        source_sha256=dataset.identity,
        profiled_process_seconds=statistics.total_tt,
        timer="cProfile_process_time",
        functions=functions,
    )


def benchmark(output, reference, repetitions, baseline_only=False):
    if not 2 <= repetitions <= 10:
        raise ValueError("La comparación necesita entre dos y diez repeticiones")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", reference], text=True).strip()
    measurements, workloads = [], []
    load_before = list(os.getloadavg())
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for name, (assets, rows, groups, stride) in CASES.items():
        manifest = prepare_corpus(
            output / name, assets=assets, rows=rows, group_rows=groups, stride=stride
        )
        metadata = json.loads(manifest.read_text())
        source_bytes = sum(
            (Path(metadata["roots"][root]) / asset["market"] / asset["symbol"] / filename)
            .stat()
            .st_size
            for asset in metadata["assets"]
            for root, filename in (
                ("prepared", "prices.parquet"),
                ("samples", "samples.parquet"),
                ("labels", "labels.parquet"),
            )
        )
        workloads.append(
            dict(
                case=name,
                assets=assets,
                source_rows=assets * rows,
                admitted_rows=metadata["counts"]["train"],
                group_rows=groups,
                accepted_stride=stride,
                source_parquet_bytes=source_bytes,
            )
        )
        for batch_size in (32, 512):
            expected = None
            for repeat in range(repetitions):
                modes = ["reference"] if baseline_only else ["reference", "candidate"]
                if repeat % 2:
                    modes.reverse()
                for mode in modes:
                    command = [
                        "uv",
                        "run",
                        "--no-sync",
                        "python",
                        str(Path(__file__).resolve()),
                        "--worker",
                        str(manifest),
                        "--batch-size",
                        str(batch_size),
                        "--reference",
                        revision if mode == "reference" else "current",
                    ]
                    started = time.perf_counter()
                    process = subprocess.run(
                        command, env=environment, capture_output=True, text=True, check=True
                    )
                    result = json.loads(process.stdout)
                    result.update(
                        case=name,
                        mode=mode,
                        repeat=repeat,
                        batch_size=batch_size,
                        process_seconds=time.perf_counter() - started,
                    )
                    expected = expected or result["stream_sha256"]
                    if result["stream_sha256"] != expected:
                        raise ValueError("El contenido, orden o cursor difiere de la referencia")
                    measurements.append(result)
                    (output / "progress.json").write_text(json.dumps(measurements, indent=2))
    summary = []
    for name in CASES:
        for batch_size in (32, 512):
            for mode in ("reference", "candidate"):
                selected = [
                    row
                    for row in measurements
                    if (row["case"], row["batch_size"], row["mode"]) == (name, batch_size, mode)
                ]
                if not selected:
                    continue
                times = [row["reader_seconds"] for row in selected]
                summary.append(
                    dict(
                        case=name,
                        batch_size=batch_size,
                        mode=mode,
                        median_seconds=statistics.median(times),
                        minimum_seconds=min(times),
                        maximum_seconds=max(times),
                        std_seconds=statistics.stdev(times),
                        median_peak_rss_bytes=statistics.median(
                            row["peak_rss_bytes"] for row in selected
                        ),
                    )
                )
    report = dict(
        schema_version=1,
        domain="technical",
        measured_at=datetime.now(UTC).isoformat(),
        reference_revision=revision,
        benchmark_sha256=sha256(Path(__file__)),
        python=platform.python_version(),
        numpy=np.__version__,
        pyarrow=pa.__version__,
        platform=platform.platform(),
        cpu=next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        logical_cpus=os.cpu_count(),
        load_average_before=load_before,
        load_average_after=list(os.getloadavg()),
        workloads=workloads,
        shapes={**DIMENSIONS, "prices": [64, 5]},
        repetitions=repetitions,
        warmup_full_passes=1,
        numerical_threads=1,
        peak_vram_bytes=None,
        cpu_gpu_transfer_bytes=None,
        energy_joules=None,
        monetary_cost=None,
        final_test_opened=False,
        summary=summary,
        measurements=measurements,
        limits=[
            "Datos sintéticos técnicos. No se ejecutan modelos ni GPU.",
            "La lectura medida incluye validación y construcción de todos los lotes.",
            "Inicialización, calentamiento y huella de salida se registran por separado.",
            "El tiempo de proceso incluye importaciones y todas esas pasadas.",
            "VmHWM incluye importaciones y calentamiento del proceso independiente.",
            "Caché del sistema caliente y aplicaciones ajenas sin detener.",
        ],
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", default="7455556")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.worker:
        operation = profile_reader if args.profile else measure
        print(
            json.dumps(operation(args.worker, batch_size=args.batch_size, revision=args.reference))
        )
    elif args.output:
        result = benchmark(args.output, args.reference, args.repetitions, args.baseline_only)
        print(json.dumps(result["summary"], indent=2))
    else:
        parser.error("Se requiere --output")


if __name__ == "__main__":
    main()
