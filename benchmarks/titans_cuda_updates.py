"""Comparar los kernels de escritura MAC y de vecinos con la referencia en la GPU local.

Uso desde la raíz: PYTHONPATH=src python benchmarks/titans_cuda_updates.py salida.json
Opcionalmente --bank acepta las columnas de un banco de ensayo con claves ya normalizadas.
Estos tiempos solo incluyen la operación indicada, no un entrenamiento completo.
"""

import argparse
import json
import statistics
import time
from contextlib import nullcontext
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

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


def reference_write(weight, previous, gradient, alpha, eta, theta):
    velocity = eta * previous - theta * gradient
    return (1 - alpha) * weight + velocity, velocity


def check_probe(device):
    """Cotejar los empates y el backward antes de cronometrar otra arquitectura."""
    values = [float("nan"), -0.0, 0.0, float("-inf"), float("inf"), 1.0, 1.0]
    scores = torch.tensor((values * 5)[:33], device=device, dtype=torch.float64).repeat(3, 1)
    scores[1].fill_(float("-inf"))
    scores[2].fill_(1.0)
    scores[2, -1] = torch.nextafter(scores[2, -1], scores.new_tensor(2.0))
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    for count in (1, 8):
        if not torch.equal(cuda_kernels.stable_neighbors(scores, count), order[:, :count]):
            raise ValueError("El orden de los vecinos empatados o casi iguales ha cambiado")

    generator = torch.Generator(device=device).manual_seed(120)
    dims = [(3, 64, 64)] * 3 + [(3, 64, 1), (3, 1, 1), (3, 1, 1)]

    def sample(shape):
        return torch.randn(shape, dtype=torch.float32, device=device, generator=generator)

    inputs = [sample(shape) for shape in dims]
    upstream = [sample(dims[0]), sample(dims[0])]
    directions = [sample(shape) for shape in dims]

    def derivatives(fused):
        args = [x.detach().clone().requires_grad_() for x in inputs]
        if fused:
            result = cuda_kernels.try_memory_write(
                (args[0],), (args[1],), (args[2],), args[3:], differentiable=True
            )
            if result is None:
                raise RuntimeError("La sonda no ejecutó la escritura CUDA")
            outputs = result[0][0], result[1][0]
        else:
            outputs = reference_write(*args)
        first = torch.autograd.grad(outputs, args, upstream, create_graph=True)
        loss = sum((g * v).sum() for g, v in zip(first, directions, strict=True))
        second = torch.autograd.grad(loss, args)
        return (*outputs, *first, *second)

    expected, actual = derivatives(False), derivatives(True)
    for index, (a, b) in enumerate(zip(expected, actual, strict=True)):
        if not torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)):
            raise ValueError(
                f"Cambian los bits de la escritura o sus derivadas en la salida {index}"
            )
    return dict(stable_index_cases=2, fp32_outputs=2, first_derivatives=6, second_derivatives=6)


def write_case(rows, iterations, generator):
    def normal(*shape):
        return torch.randn(shape, device="cuda", generator=generator)

    weight, previous, gradient = (normal(rows, 64, 64) for _ in range(3))
    alpha, eta, theta = normal(rows, 64, 1), normal(rows, 1, 1), normal(rows, 1, 1)

    def reference():
        return reference_write(weight, previous, gradient, alpha, eta, theta)

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


def benchmark_dispatch(device, probe):
    """La excepción opcional vive solo durante el microbenchmark y se limita a una GPU."""
    if device.type != "cuda" or torch.version.cuda is None or find_spec("triton") is None:
        raise ValueError("La comparación requiere CUDA de NVIDIA y Triton")
    if cuda_kernels._supported_device(device):
        return nullcontext()
    if not probe:
        raise ValueError("Esta GPU solo admite el ensayo exploratorio con --probe-device")
    return patch.object(cuda_kernels, "_supported_device", lambda current: current == device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument(
        "--probe-device",
        action="store_true",
        help="Medir solo en este proceso otra GPU NVIDIA que disponga de Triton",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations debe ser positivo")
    if not torch.cuda.is_available():
        parser.error("El ensayo necesita una GPU CUDA disponible")
    device = torch.device("cuda:0")
    supported = cuda_kernels._supported_device(device)
    try:
        dispatch = benchmark_dispatch(device, args.probe_device)
    except ValueError as exc:
        parser.error(str(exc))
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
    with dispatch:
        probe_checks = check_probe(device)
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
            "production_supported": supported,
            "probe_checks": probe_checks,
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
