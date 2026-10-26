"""Medir en CPU la lectura y la escritura de B1, de sus controles y de B6 con la clave del codec.

Las claves y los valores son sintéticos con semilla fija y solo fijan las formas: clave de 64
coordenadas, valor escalar y cohortes de 64, 256 y 1.024 resultados por evento. La criba usa 256
activos de EE. UU., así que 256 es el caso principal. Se ejecutan lecturas y escrituras de
prueba, sin optimizador ni evaluación predictiva. Cada caso parte de un estado ya escrito,
con tres calentamientos y veinte repeticiones, y registra percentiles y bytes del estado.
Nadaraya-Watson se mide con su capacidad llena, el peor caso de su lectura.
"""

import argparse
import hashlib
import os
import platform
import resource
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.memory.area_memory import AreaLayout
from mars_titan.memory.area_query_bridge import AreaQueryBridgeConfig
from mars_titan.memory.associative_memory import (
    AssociativeMemoryConfig,
    MatureCorrection,
)
from mars_titan.memory.kernel_memory import KernelMemoryConfig

SIZES = (64, 256, 1024)
PROXIMAL = AssociativeMemoryConfig("proximal", rate=0.05, forgetting=0.01)
CASES = {
    "b6_linear_codec": MatureCorrection(PROXIMAL),
    "area_memory": MatureCorrection(PROXIMAL, areas=AreaLayout(areas=4, shifts=1)),
    "single_area_same_coordinates": MatureCorrection(
        PROXIMAL, areas=AreaLayout(areas=1, shifts=4, permutation="identity")
    ),
    "identical_areas": MatureCorrection(
        PROXIMAL, areas=AreaLayout(areas=4, shifts=1, permutation="identity")
    ),
    "shuffled_labels": MatureCorrection(
        PROXIMAL, areas=AreaLayout(areas=4, shifts=1, labels="shuffled_within_cohort")
    ),
    "nadaraya_watson": MatureCorrection(
        KernelMemoryConfig(bandwidth=0.1, forgetting=0.01, capacity=65536)
    ),
}
CASES.update(
    {
        name: MatureCorrection(PROXIMAL, areas=AreaLayout(), query_bridge=config)
        for name, config in {
            "query_bridge": AreaQueryBridgeConfig(),
            "query_bridge_zero_gain": AreaQueryBridgeConfig(gain=0),
            "query_bridge_shuffled": AreaQueryBridgeConfig(targets="shuffled_within_cohort"),
            "query_bridge_two_steps": AreaQueryBridgeConfig(steps=2),
            "query_bridge_four_steps": AreaQueryBridgeConfig(steps=4),
        }.items()
    }
)


def cohort(rows, start, at, correction, generator):
    keys = torch.randn((rows, 64), dtype=torch.float64, generator=generator)
    keys = keys / torch.linalg.vector_norm(keys, dim=1, keepdim=True)
    values = [0.01 * value for value in torch.randn(rows, generator=generator).tolist()]
    return correction.feedback(
        ids=list(range(start, start + rows)),
        decision_at=[at - 5] * rows,
        available_at=[at] * rows,
        keys=keys,
        values=values,
    )


def timed(function, warmup, repeats):
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    ordered = sorted(samples)
    return dict(
        p50_ms=statistics.median(samples) * 1e3,
        p95_ms=ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))] * 1e3,
        p99_ms=ordered[min(len(ordered) - 1, round(0.99 * (len(ordered) - 1)))] * 1e3,
        min_ms=ordered[0] * 1e3,
        repeats=repeats,
        samples_ms=[value * 1e3 for value in samples],
    )


def filled(correction, generator):
    """Estado tras escrituras de 256 resultados: una para las áreas y la capacidad para NW."""
    memory = correction.empty()
    writes = 1 if correction.memory.rule != "nadaraya_watson" else 65536 // 256
    for index in range(writes):
        memory = memory.write(
            cohort(256, memory.writes + 1, 10 * (index + 1), correction, generator),
            cutoff=10 * (index + 1),
        )
    return memory


def state_bytes(payload):
    if isinstance(payload, torch.Tensor):
        return payload.nbytes
    if isinstance(payload, dict):
        return sum(state_bytes(value) for value in payload.values())
    if isinstance(payload, list):
        return sum(state_bytes(value) for value in payload)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 <= args.warmup <= 10 or not 1 <= args.repeats <= 100:
        raise ValueError("El calentamiento y las repeticiones exceden los límites de la medida")
    if args.output.exists():
        raise FileExistsError("La medida necesita un recibo nuevo")
    started, load_start = time.perf_counter(), os.getloadavg()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    cases = {}
    for name, correction in CASES.items():
        generator = torch.Generator().manual_seed(20261023)
        memory = filled(correction, generator)
        for rows in SIZES:
            # El mismo lote de consulta en todos los modelos, separado de la carga de memoria.
            query_generator = torch.Generator().manual_seed(20261024 + rows)
            later = cohort(rows, memory.writes + 1, 10_000_000, correction, query_generator)
            cases[f"{name}/{rows}"] = dict(
                write=timed(
                    lambda later=later, memory=memory: memory.write(later, cutoff=10_000_000),
                    args.warmup,
                    args.repeats,
                ),
                read=timed(
                    lambda keys=later.keys, memory=memory: memory.read(keys),
                    args.warmup,
                    args.repeats,
                ),
            )
        serialized = state_bytes(memory.export())
        cases[f"{name}/serialized_tensor_bytes"] = serialized
        cases[f"{name}/numeric_state_bytes"] = getattr(memory, "numeric_state_bytes", serialized)
        cases[f"{name}/matrix_elements"] = memory.matrix.numel()
    receipt = dict(
        schema_version=1,
        kind="redundant_area_memory_cost",
        issue=750,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        sources_sha256={
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (
                "src/mars_titan/memory/area_memory.py",
                "src/mars_titan/memory/area_query_bridge.py",
                "src/mars_titan/memory/kernel_memory.py",
                "src/mars_titan/memory/associative_memory.py",
                "benchmarks/area_memory_cost.py",
            )
        },
        data="Claves y valores sintéticos de semilla fija para lecturas y escrituras técnicas.",
        shapes=dict(key_size=64, value_size=1, cohorts=list(SIZES), main_cohort=256),
        corrections={name: correction.identity() for name, correction in CASES.items()},
        environment=dict(
            cpu=cpu_name(),
            torch=torch.__version__,
            torch_threads=torch.get_num_threads(),
            python=platform.python_version(),
            load_average_at_start=load_start,
            load_average_at_end=os.getloadavg(),
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        ),
        not_measured=[
            "GPU, porque B1 y sus controles trabajan en CPU y FP64 como B6",
            "gradientes, porque B1 no tiene parámetros entrenables y el padre llega congelado",
            "energía, sin instrumento",
        ],
        cases=cases,
        total_wall_seconds=time.perf_counter() - started,
        quantiles="p50 como mediana, p95/p99 como estadístico de orden más cercano",
        limits=[
            "Con veinte repeticiones p99 es casi el máximo observado y no define un SLO.",
            "Los bytes numéricos no incluyen objetos Python ni picos temporales de factorización.",
            "Se igualan coordenadas de coeficientes, no tiempo de cómputo ni todos los metadatos.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, receipt)


if __name__ == "__main__":
    main()
