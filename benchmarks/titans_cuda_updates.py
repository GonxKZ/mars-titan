"""Comparar los kernels de escritura MAC y de vecinos con la referencia en la GPU local.

Uso desde la raíz: PYTHONPATH=src python benchmarks/titans_cuda_updates.py salida.json
Opcionalmente --bank acepta las columnas de un banco de ensayo con claves ya normalizadas.
Estos tiempos solo incluyen la operación indicada, no un entrenamiento completo.
"""

import argparse
import json
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import torch

from mars_titan.models.titans import cuda_kernels


def compare(reference, candidate, iterations):
    for operation in (reference, candidate, candidate, reference):
        operation()
    timings = {"reference": [], "candidate": []}
    memory = {label: [] for label in timings}
    for label in ("reference", "candidate", "candidate", "reference") * 3:
        operation = reference if label == "reference" else candidate
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize()
        timings[label].append(1e6 * (time.perf_counter() - start) / iterations)
        memory[label].append(torch.cuda.max_memory_allocated() - before)
    return {
        "microseconds": timings,
        "peak_extra_bytes": memory,
        "median_speedup": statistics.median(timings["reference"])
        / statistics.median(timings["candidate"]),
    }


def neighbor_case(scores, iterations):
    def reference():
        return torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :8]

    def candidate():
        return cuda_kernels.stable_neighbors(scores, 8)

    if not torch.equal(reference(), candidate()):
        raise ValueError("La selección cambió los índices o su orden")
    return compare(reference, candidate, iterations)


def write_case(rows, iterations, generator):
    def normal(*shape):
        return torch.randn(shape, device="cuda", generator=generator)

    weight, previous, gradient = (normal(rows, 64, 64) for _ in range(3))
    alpha, eta, theta = normal(rows, 64, 1), normal(rows, 1, 1), normal(rows, 1, 1)

    def reference():
        velocity = eta * previous - theta * gradient
        return (1 - alpha) * weight + velocity, velocity

    def candidate():
        result = cuda_kernels.try_memory_write(
            (weight,), (previous,), (gradient,), (alpha, eta, theta), differentiable=False
        )
        if result is None:
            raise RuntimeError("La GPU o la disposición no utiliza la escritura fusionada")
        return result[0][0], result[1][0]

    if not all(
        torch.equal(a.view(torch.int32), b.view(torch.int32))
        for a, b in zip(reference(), candidate(), strict=True)
    ):
        raise ValueError("La escritura cambió los bits del peso o del momentum")
    return compare(reference, candidate, iterations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--iterations", type=int, default=32)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations debe ser positivo")
    if not torch.cuda.is_available() or not cuda_kernels._supported_device(torch.device("cuda:0")):
        parser.error("El ensayo necesita la ruta CUDA de Titans admitida en este dispositivo")
    if args.output.exists():
        parser.error("Elige una salida nueva para conservar cada medida")
    torch.cuda.set_device(0)
    generator = torch.Generator(device="cuda").manual_seed(42)
    if args.bank:
        keys = torch.load(args.bank, map_location="cpu", weights_only=True)["keys"].to(
            device="cuda", dtype=torch.float64
        )
        if keys.ndim != 2 or keys.shape[1] != 64 or not 32 <= keys.shape[0] <= 1024:
            parser.error("Se espera una matriz de entre 32 y 1.024 claves de 64 coordenadas")
    else:
        keys = torch.randn((1024, 64), dtype=torch.float64, device="cuda", generator=generator)
    scores = {}
    for rows in (1, 32, 128, 512):
        query = torch.randn((rows, 64), dtype=torch.float64, device="cuda", generator=generator)
        full = query @ keys.T
        for columns in sorted({min(keys.shape[0], n) for n in (32, 128, 512, 1024)}):
            scores[f"{rows}x{columns}"] = neighbor_case(
                full[:, :columns].contiguous(), args.iterations
            )
    result = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "torch": str(torch.__version__),
        "triton": version("triton"),
        "iterations_per_measurement": args.iterations,
        "stable_neighbors": scores,
        "memory_write": {
            rows: write_case(rows, args.iterations, generator)
            for rows in (1, 8, 64, 128, 512, 1024)
        },
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
