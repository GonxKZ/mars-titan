"""Comparación alternada de B2 con una revisión anterior, sin aprendizaje del predictor."""

import argparse
import hashlib
import importlib
import importlib.util
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import torch
from area_memory_cost import cohort, filled

from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name

REFERENCE = "3e89cc5dff0a22303ea5d213e40fe1df179142cd"
VARIANTS = {
    "bridge": {},
    "zero_gain": {"gain": 0.0},
    "shuffled": {"targets": "shuffled_within_cohort"},
}
SOURCE_MODULES = ("associative_memory", "area_memory", "area_query_bridge")


def modules(package):
    return {name: importlib.import_module(f"{package}.memory.{name}") for name in SOURCE_MODULES}


def reference_modules(commit, directory):
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("La referencia necesita el SHA completo de un commit local")
    process = subprocess.Popen(
        ["git", "archive", "--format=tar", commit, "--", "src/mars_titan"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        size, count = 0, 0
        with tarfile.open(fileobj=process.stdout, mode="r|") as stream:
            for member in stream:
                size, count = size + member.size, count + 1
                if size > 64 * 1024**2 or count > 4096:
                    raise ValueError("La copia de referencia supera el presupuesto de código")
                stream.extract(member, directory, filter="data")
        if process.wait(timeout=30) != 0:
            raise ValueError("No se pudo leer la revisión de referencia")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()
    package = "_paired_area_reference"
    root = Path(directory) / "src/mars_titan"
    specification = importlib.util.spec_from_file_location(
        package, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[package] = module
    specification.loader.exec_module(module)
    return modules(package)


def correction(implementation, options):
    return implementation["associative_memory"].MatureCorrection(
        implementation["associative_memory"].AssociativeMemoryConfig(
            "proximal", rate=0.05, forgetting=0.01
        ),
        areas=implementation["area_memory"].AreaLayout(),
        query_bridge=implementation["area_query_bridge"].AreaQueryBridgeConfig(**options),
    )


def source_hashes(implementation):
    return {
        f"src/mars_titan/memory/{name}.py": hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
        for name, module in implementation.items()
    }


def summary(samples, rows):
    ordered = sorted(samples)
    return dict(
        samples_seconds=samples,
        p50_ms=statistics.median(samples) * 1000,
        p95_ms=ordered[round(0.95 * (len(ordered) - 1))] * 1000,
        p99_ms=ordered[round(0.99 * (len(ordered) - 1))] * 1000,
        rows_per_second=rows / statistics.median(samples),
    )


def cycle(read, write):
    return read(), write()


def compare(before, after, rows, warmup, repeats):
    references = [
        filled(value, torch.Generator().manual_seed(20261023)) for value in (before, after)
    ]
    feedback = [
        cohort(
            rows,
            state.writes + 1,
            10_000_000,
            value,
            torch.Generator().manual_seed(20261024 + rows),
        )
        for state, value in zip(references, (before, after), strict=True)
    ]
    operations = {}
    updated = []
    for name, state, data in zip(("reference", "current"), references, feedback, strict=True):
        read = partial(state.read, data.keys)
        write = partial(state.write, data, cutoff=10_000_000)
        operations[f"{name}_read"] = read
        operations[f"{name}_write"] = write
        operations[f"{name}_cycle"] = partial(cycle, read, write)
        updated.append(write())
    if (
        updated[0].export()["matrix_sha256"] != updated[1].export()["matrix_sha256"]
        or not torch.equal(updated[0].read(feedback[0].keys), updated[1].read(feedback[1].keys))
        or before.identity() != after.identity()
    ):
        raise ValueError("La implementación actual no conserva la referencia bit a bit")
    del updated
    for _ in range(warmup):
        for operation in operations.values():
            operation()
    samples = {name: [] for name in operations}
    for index in range(repeats):
        order = list(operations)
        if index % 2:
            order.reverse()
        for name in order:
            started = time.perf_counter()
            operations[name]()
            samples[name].append(time.perf_counter() - started)
    return dict(
        parity="bitwise",
        numeric_state_bytes=[state.numeric_state_bytes for state in references],
        timings={name: summary(values, rows) for name, values in samples.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-commit", default=REFERENCE)
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not 0 <= args.warmup <= 10
        or not 1 <= args.repeats <= 100
        or not 1 <= len(args.rows) <= 4
        or any(not 1 <= rows <= 8192 for rows in args.rows)
        or len(set(args.rows)) != len(args.rows)
    ):
        raise ValueError("Los tamaños y las repeticiones exceden el presupuesto de la medida")
    if args.output.exists():
        raise FileExistsError("La medida necesita un archivo nuevo")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started, load = time.perf_counter(), os.getloadavg()
    current = modules("mars_titan")
    with tempfile.TemporaryDirectory(
        prefix="paired-reference-", dir=args.output.parent
    ) as directory:
        reference = reference_modules(args.reference_commit, directory)
        results = {
            f"{name}/{rows}": compare(
                correction(reference, options),
                correction(current, options),
                rows,
                args.warmup,
                args.repeats,
            )
            for name, options in VARIANTS.items()
            for rows in args.rows
        }
        receipt = dict(
            schema_version=1,
            kind="paired_proximal_memory_cost",
            measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
            reference_commit=args.reference_commit,
            reference_sources_sha256=source_hashes(reference),
            current_sources_sha256=source_hashes(current),
            paired_source_sha256=hashlib.sha256(
                Path("src/mars_titan/memory/paired_proximal.py").read_bytes()
            ).hexdigest(),
            benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            workload=(
                "Lectura y escritura de una cohorte sintética sobre el mismo snapshot, "
                "sin pasos de optimizador."
            ),
            shapes=dict(key_size=64, areas=4, feature_size=128, rows=args.rows),
            warmup=args.warmup,
            repeats=args.repeats,
            order="Orden de las seis operaciones invertido en las repeticiones impares.",
            quantiles="p50 como mediana, p95/p99 como estadístico de orden más cercano con round.",
            environment=dict(
                cpu=cpu_name(),
                python=platform.python_version(),
                torch=str(torch.__version__),
                threads=torch.get_num_threads(),
                load_average_at_start=load,
                load_average_at_end=os.getloadavg(),
                max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                cuda_initialized=torch.cuda.is_initialized(),
            ),
            cases=results,
            main_wall_seconds=time.perf_counter() - started,
            limits=[
                "El RSS incluye las dos implementaciones y el entorno de benchmark, "
                "no su diferencia de pico.",
                "p95 y p99 son estadísticos de orden de las muestras, no garantías de servicio.",
                "El ciclo no incluye el predictor CUDA, codificación o persistencia de una sesión.",
            ],
        )
        atomic_json(args.output, receipt)


if __name__ == "__main__":
    main()
