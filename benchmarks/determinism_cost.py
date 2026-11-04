"""Medir el coste del determinismo de PyTorch y si dos ejecuciones iguales coinciden (#794).

Cada modo corre en procesos nuevos, porque cuBLAS lee `CUBLAS_WORKSPACE_CONFIG` al crear su
primer handle y el modo determinista cambia los núcleos de todo el proceso:

- `process_default`: FP32 estricto, sin el modo determinista de PyTorch ni el espacio de
  trabajo fijo de cuBLAS, como un ajuste que no declara `determinism`.
- `torch_deterministic_algorithms_v1`: lo mismo con `apply_determinism` y
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Cada proceso construye la referencia MAC en línea de la receta histórica con máscaras con las
entradas sintéticas de `objective_variants_cost.py`, que tienen las formas de la edición v3.1.
Recorre T instantes con grafo, calcula el error absoluto frente a objetivos sintéticos y hace
backward. Tras W calentamientos repite el recorrido R veces y guarda la huella SHA-256 de las
predicciones y de los gradientes de cada repetición. Cada modo se lanza en dos procesos, y el
recibo compara las huellas dentro de cada proceso, entre los dos procesos de un modo y entre
modos. En CUDA cada proceso toma la GPU exclusiva de `GpuLease`. No hay optimizador ni pasos,
y al final se comprueba que los parámetros no han cambiado.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmarks.objective_variants_cost import BASE, batches, specification
from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.training.determinism import TORCH_DETERMINISTIC, apply_determinism
from mars_titan.training.experiment_resources import GpuLease
from mars_titan.training.kernel_policy import FP32_STRICT, apply_kernel_policy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = "process_default"
MODES = (DEFAULT, TORCH_DETERMINISTIC)
PROCESSES = 2
GIB = 1024**3
LIMITS = dict(rows=(1, 1024), instants=(1, 16), repetitions=(2, 20), warmup=(0, 5), threads=(1, 8))


def _digest(tensors):
    digest = hashlib.sha256()
    for name, value in tensors:
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _summary(values):
    return dict(
        median=statistics.median(values), minimum=min(values), maximum=max(values), samples=values
    )


def graph_pass(model, inputs, targets, synchronize):
    """Un recorrido con grafo y su backward, con la huella de predicciones y gradientes."""
    synchronize()
    started = time.perf_counter()
    state = model.initial_state(inputs[0].flow_ids, differentiable=True)
    points = []
    for batch in inputs:
        result = model.prepare(batch, state, differentiable=True, session_complete=True)
        state = result.next_state
        points.append(result.point_predictions)
    point = torch.cat(points)
    torch.nn.functional.l1_loss(point, torch.cat(targets)).backward()
    synchronize()
    elapsed = time.perf_counter() - started
    gradients = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
    if not all(torch.isfinite(grad).all() for _, grad in gradients):
        raise ValueError("La medida encontró un gradiente no finito")
    digests = dict(predictions=_digest([("point", point)]), gradients=_digest(gradients))
    for value in model.parameters():
        value.grad = None
    return elapsed, digests


def worker(mode, device, *, rows, instants, repetitions, warmup, threads):
    """Medir un modo en este proceso y devolver sus tiempos y huellas."""
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(False)
    policy = apply_kernel_policy(FP32_STRICT)
    # Fuera del modo determinista se conserva lo que PyTorch hace por defecto.
    determinism = apply_determinism(mode) if mode == TORCH_DETERMINISTIC else None
    lease = GpuLease(max_ram=6 * GIB, max_vram=6 * GIB) if device == "cuda:0" else None
    if lease is not None:
        lease.__enter__()
    try:
        synchronize = torch.cuda.synchronize if lease is not None else (lambda: None)
        spec = specification()
        inputs = batches(spec, rows, instants, device=device)
        generator = torch.Generator().manual_seed(11)
        targets = [
            (torch.randn(rows, generator=generator) * 0.015).to(device) for _ in range(instants)
        ]
        config = FinancialConfig(spec, variant="mac_online", seed=42, **BASE)
        model = FinancialPredictor(config, dtype=torch.float32).to(device).train()
        before = _digest(model.named_parameters())
        with sdpa_kernel(SDPBackend.MATH):
            for _ in range(warmup):
                graph_pass(model, inputs, targets, synchronize)
            runs = [graph_pass(model, inputs, targets, synchronize) for _ in range(repetitions)]
        if _digest(model.named_parameters()) != before:
            raise RuntimeError("La medida ha modificado parámetros")
        record = dict(
            mode=mode,
            device=device,
            kernel_policy=policy,
            determinism=determinism,
            numerics=dict(
                deterministic=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            ),
            graph_seconds=_summary([elapsed for elapsed, _ in runs]),
            digests=[digests for _, digests in runs],
            parameters_unchanged=True,
            versions=dict(torch=torch.__version__, cuda=torch.version.cuda),
        )
        if lease is not None:
            lease.check()
            record.update(
                gpu=torch.cuda.get_device_name(0),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                admission=dict(lease.record),
            )
        return record
    finally:
        if lease is not None:
            lease.__exit__()


def launch(mode, device, options, scratch, index):
    """Ejecutar un modo en un proceso nuevo con su entorno de cuBLAS."""
    environment = dict(os.environ)
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    if mode == TORCH_DETERMINISTIC:
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    result = Path(scratch) / f"{mode}-{index}.json"
    command = [sys.executable, "-m", "benchmarks.determinism_cost", "--worker", mode]
    command += ["--device", device, "--result", str(result)]
    for name, value in options.items():
        command += [f"--{name}", str(value)]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return json.loads(result.read_text(encoding="utf-8"))


def _same(values):
    return len({json.dumps(value, sort_keys=True) for value in values}) == 1


def compare(records):
    """Coincidencia de las huellas dentro de cada proceso, entre procesos y entre modos."""
    modes = {
        mode: dict(
            within_process=all(_same(run["digests"]) for run in runs),
            across_processes=_same([run["digests"][0] for run in runs]),
        )
        for mode, runs in records.items()
    }
    default, fixed = (records[mode][0]["digests"][0] for mode in MODES)
    medians = [
        statistics.median(
            value for run in records[mode] for value in run["graph_seconds"]["samples"]
        )
        for mode in MODES
    ]
    return dict(
        modes=modes,
        same_predictions_across_modes=default["predictions"] == fixed["predictions"],
        same_gradients_across_modes=default["gradients"] == fixed["gradients"],
        median_graph_seconds=dict(zip(MODES, medians, strict=True)),
        cost_ratio_of_medians=medians[1] / medians[0],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--instants", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--worker", choices=MODES)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    options = {name: getattr(args, name) for name in LIMITS}
    if any(not low <= options[name] <= high for name, (low, high) in LIMITS.items()):
        raise SystemExit("Las formas, repeticiones o hilos exceden el presupuesto de la medida")
    if args.worker is not None:
        if args.result is None or args.result.exists():
            raise SystemExit("El proceso de medida necesita un resultado nuevo")
        record = worker(args.worker, args.device, **options)
        args.result.write_text(json.dumps(record), encoding="utf-8")
        return
    if args.output is None or args.output.exists():
        raise SystemExit("La medida necesita un recibo nuevo")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    load = os.getloadavg()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Los resultados de cada proceso van junto a la salida y no en /tmp, que es memoria.
    with tempfile.TemporaryDirectory(dir=args.output.parent) as scratch:
        records = {
            mode: [launch(mode, args.device, options, scratch, index) for index in range(PROCESSES)]
            for mode in MODES
        }
    sources = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in (
            "src/mars_titan/training/determinism.py",
            "src/mars_titan/training/kernel_policy.py",
            "benchmarks/determinism_cost.py",
            "benchmarks/objective_variants_cost.py",
        )
    }
    comparison = compare(records)
    receipt = dict(
        schema_version=1,
        kind="determinism_cost",
        issue=794,
        recorded_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=commit,
        sources_sha256=sources,
        scope=(
            "Forward con grafo de T instantes de la referencia MAC en línea, error absoluto frente "
            "a objetivos sintéticos y backward, en dos procesos por modo. Entradas sintéticas con "
            "las formas de la edición v3.1. Sin optimizador ni pasos."
        ),
        device=args.device,
        options=options,
        environment=dict(
            cpu=cpu_name(),
            cpu_count=os.cpu_count(),
            python=platform.python_version(),
            load_average_at_start=load,
            load_average_at_end=os.getloadavg(),
            concurrent_load="otras sesiones comparten la máquina, ver las cargas medias",
        ),
        not_measured=[
            "el recorrido cronológico completo con lectura de datos, que ejecutaría la criba",
            "energía, sin instrumento adecuado",
        ],
        comparison=comparison,
        processes=records,
    )
    atomic_json(args.output, receipt)
    print(json.dumps(comparison, ensure_ascii=False))


if __name__ == "__main__":
    main()
