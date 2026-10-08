"""Mide mecanismos C/M con datos sintéticos CPU, sin modelos ni aprendizaje."""

import json
import math
import os
import platform
import resource
import statistics
import time
import tracemalloc
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

from mars_titan.cm.medoids import select_medoids
from mars_titan.cm.numerical_radius import numerical_radius_estimates, radius_penalty


def _latencies(call, repetitions):
    call()
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        result = call()
        times.append(time.perf_counter() - start)
    return result, {
        "seconds": times,
        "median_seconds": statistics.median(times),
        "minimum_seconds": min(times),
        "maximum_seconds": max(times),
        "stdev_seconds": statistics.stdev(times),
    }


def _operator(matrix, gradient):
    result = numerical_radius_estimates(matrix)
    if gradient:
        torch.autograd.grad(radius_penalty(result, 0.9).sum(), matrix)
    return result


def benchmark():
    """Devuelve tiempos internos, buffers, RSS máximo y límites de la medida."""
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(812)
    measurements = []
    for order, batch, gradient in [(16, 1, False), (32, 2, True), (64, 1, False)]:
        source = rng.standard_normal((batch, order, order)) / math.sqrt(order)
        matrix = torch.tensor(source, dtype=torch.float64, device="cpu", requires_grad=gradient)
        result, times = _latencies(partial(_operator, matrix, gradient), 5)
        measurements.append(
            {
                "mechanism": "C",
                "order": order,
                "batch": batch,
                "grid_size": 64,
                "angle_block_size": 8,
                "gradient": gradient,
                **times,
                "estimated_forward_bytes": result.estimated_forward_bytes,
                "estimated_saved_bytes": result.estimated_saved_bytes,
            }
        )
    for n, f, dimensions, capacity in [(256, 16, 4, 3), (1024, 32, 8, 4), (4096, 64, 8, 4)]:
        clients = rng.standard_normal((n, dimensions))
        candidates = rng.standard_normal((f, dimensions))
        client_ids = [f"e{i:06d}" for i in range(n)]
        candidate_ids = [f"f{i:06d}" for i in range(f)]
        call = partial(select_medoids, clients, client_ids, candidates, candidate_ids, capacity)
        result, times = _latencies(call, 3)
        tracemalloc.start()
        call()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if peak > result.estimated_peak_bytes:
            raise RuntimeError("El pico rastreado supera la estimación de buffers propios")
        reference = math.fsum(
            min(math.dist(row, candidates[index]) for index in result.candidate_indices)
            for row in clients
        )
        if not math.isclose(reference, result.objective, rel_tol=1e-14):
            raise ArithmeticError("El coste no coincide con la referencia independiente")
        measurements.append(
            {
                "mechanism": "M",
                "clients": n,
                "candidates": f,
                "dimensions": dimensions,
                "capacity": capacity,
                **times,
                "result": asdict(result),
                "tracemalloc_peak_bytes": peak,
                "distance_pairs_per_second": result.distance_pairs / times["median_seconds"],
                "input_coordinate_bytes": clients.nbytes + candidates.nbytes,
                "independent_objective": reference,
            }
        )
    cpu = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "device": "cpu",
        "cpu": cpu,
        "platform": platform.platform(),
        "seed": 812,
        "threads": {
            "torch": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "measurements": measurements,
        "warmup_calls_per_shape": 1,
        "workload_wall_seconds": time.perf_counter() - started,
        "process_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "process_peak_rss_bytes": usage.ru_maxrss * 1024,
        "limitations": [
            "Script para Linux. No controla otras aplicaciones del equipo.",
            "Las latencias internas excluyen importaciones y el calentamiento.",
            "El tiempo de carga incluye calentamientos y referencias, pero excluye importaciones.",
            "El tiempo CPU y el RSS máximo corresponden al proceso, incluidas importaciones.",
            "tracemalloc registra asignaciones de Python y NumPy, no todos los workspaces nativos.",
            "No se midieron energía, GPU, VRAM, transferencias ni coste económico.",
            "No se compara una implementación sustituida ni se afirma una aceleración.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), ensure_ascii=False, indent=2))
