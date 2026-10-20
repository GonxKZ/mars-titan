"""Coste de captura local en escrituras completas de memoria, sin optimizador."""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from mars_titan.models.titans.config import MemoryConfig, MemoryMultiscale
from mars_titan.models.titans.financial import MEMORY_PROJECTIONS, PAPER_PROJECTIONS
from mars_titan.models.titans.neural_memory import NeuralMemory
from mars_titan.models.titans.plasticity_config import MemoryPlasticity
from mars_titan.training.gpu_supervisor import read_gpu


def digest(model):
    result = hashlib.sha256()
    for name, value in model.named_parameters():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def run(output):
    if output.exists():
        raise FileExistsError("La medida necesita un archivo nuevo")
    admission = read_gpu()
    if admission.compute_pids or admission.free_mib < 6144:
        raise RuntimeError("Se requiere GPU libre de otras cargas de cómputo y 6 GiB disponibles")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible. No hay alternativa implícita CPU")
    torch.cuda.set_device("cuda:0")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Se requiere la configuración determinista declarada de cuBLAS")
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    records = []
    variants = dict(
        capture_global=MemoryPlasticity(
            write="none", capture="eligibility", gradient_contract="scalar_controls_v2"
        ),
        capture_local=MemoryPlasticity(
            write="none",
            capture="eligibility",
            gradient_contract="scalar_controls_v2",
            capture_scope="row_local",
        ),
        capture_constant=MemoryPlasticity(
            write="none", capture="constant", gradient_contract="scalar_controls_v2"
        ),
    )
    names = list(variants)
    for dtype in (torch.float32, torch.float64):
        for size in (64, 256):
            generator = torch.Generator(device="cuda:0").manual_seed(841)
            tokens = (
                torch.randn(size, 4, 64, generator=generator, device="cuda:0", dtype=dtype) / 10
            )
            for graph in (False, True):
                for repeat in range(5):
                    offset = repeat % len(names)
                    for name in names[offset:] + names[:offset]:
                        config = MemoryConfig(
                            64,
                            depth=2,
                            max_batch=256,
                            max_tokens=4,
                            multiscale=MemoryMultiscale(),
                            plasticity=variants[name],
                            max_state_bytes=256 * 1024**2,
                            **MEMORY_PROJECTIONS[PAPER_PROJECTIONS],
                        )
                        construction = time.perf_counter()
                        model = NeuralMemory(config, device="cuda:0", dtype=dtype)
                        initial = model.initial_state(size)
                        torch.cuda.synchronize()
                        construction = time.perf_counter() - construction
                        before = digest(model)
                        for _ in range(3):
                            warm = model.update(tokens, initial, differentiable=graph)
                            if graph:
                                warm_loss = sum(
                                    value.square().mean() for value in (*warm.weights, *warm.levels)
                                )
                                warm_loss.backward()
                                for parameter in model.parameters():
                                    parameter.grad = None
                                del parameter
                                del warm_loss
                            del warm
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        allocated = torch.cuda.memory_allocated()
                        begin = time.perf_counter()
                        state = model.update(tokens, initial, differentiable=graph)
                        torch.cuda.synchronize()
                        forward = time.perf_counter() - begin
                        backward = None
                        if graph:
                            begin = time.perf_counter()
                            loss = sum(
                                value.square().mean() for value in (*state.weights, *state.levels)
                            )
                            loss.backward()
                            torch.cuda.synchronize()
                            backward = time.perf_counter() - begin
                            if not all(
                                p.grad is None or torch.isfinite(p.grad).all()
                                for p in model.parameters()
                            ):
                                raise ValueError("El backward no produce gradientes finitos")
                            del loss
                        peak = torch.cuda.max_memory_allocated()
                        reserved = torch.cuda.max_memory_reserved()
                        if peak > 6 * 1024**3:
                            raise RuntimeError("El caso excedió 6 GiB de asignación CUDA")
                        if before != digest(model):
                            raise RuntimeError("La medida modificó los parámetros compartidos")
                        persistent = sum(
                            value.numel() * value.element_size()
                            for value in (
                                *initial.weights,
                                *initial.momentum,
                                *initial.levels,
                                *initial.modulation,
                                *initial.convolution,
                                initial.steps,
                            )
                        )
                        records.append(
                            dict(
                                variant=name,
                                dtype=str(dtype),
                                rows=size,
                                graph=graph,
                                repeat=repeat,
                                configuration=config.identity(),
                                construction_seconds=construction,
                                forward_four_writes_seconds=forward,
                                backward_seconds=backward,
                                allocated_before_bytes=allocated,
                                peak_allocated_bytes=peak,
                                peak_reserved_bytes=reserved,
                                persistent_state_bytes=persistent,
                                parameter_sha256=before,
                                parameters_unchanged=True,
                            )
                        )
                        del model, initial, state
                        torch.cuda.empty_cache()
            del tokens
    summary = {}
    for dtype in ("torch.float32", "torch.float64"):
        if len({value["parameter_sha256"] for value in records if value["dtype"] == dtype}) != 1:
            raise RuntimeError("Las variantes no comparten la inicialización")
        for size in (64, 256):
            for graph in (False, True):
                for name in names:
                    items = [
                        v
                        for v in records
                        if (v["dtype"], v["rows"], v["graph"], v["variant"])
                        == (dtype, size, graph, name)
                    ]
                    seconds = [
                        v["forward_four_writes_seconds"] + (v["backward_seconds"] or 0)
                        for v in items
                    ]
                    summary[f"{dtype}/{size}/{'backward' if graph else 'inference'}/{name}"] = dict(
                        p50_ms=float(np.quantile(seconds, 0.5)) * 1000,
                        p95_ms=float(np.quantile(seconds, 0.95)) * 1000,
                        p99_ms=float(np.quantile(seconds, 0.99)) * 1000,
                        rows_per_second=4 * size * len(seconds) / sum(seconds),
                        peak_allocated_bytes=max(v["peak_allocated_bytes"] for v in items),
                        persistent_state_bytes=items[0]["persistent_state_bytes"],
                        repetitions=len(items),
                    )
    result = dict(
        schema_version=1,
        kind="local_capture_memory_cuda_cost",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hashes={
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "src/mars_titan/models/titans/plasticity.py",
                "src/mars_titan/models/titans/plasticity_config.py",
                "src/mars_titan/models/titans/neural_memory.py",
            )
        },
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        gpu=torch.cuda.get_device_name(0),
        device="cuda:0",
        admission=asdict(admission),
        warmup_trajectories=3,
        warmup_writes_per_trajectory=4,
        measured_writes_per_trajectory=4,
        condition_order="rotation",
        repetitions=5,
        internal_seconds=time.perf_counter() - started,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        optimizer_steps=0,
        final_test_opened=False,
        summaries=summary,
        records=records,
        limits=[
            "Fixtures técnicos y pesos iniciales, sin datos financieros.",
            "Mide cuatro escrituras completas de memoria y backward cuando se declara. "
            "No mide el predictor financiero ni su canalización de datos.",
            "La inicialización se registra aparte. Los p95/p99 sobre cinco trayectorias "
            "no estiman colas raras.",
            "No hay paridad esperada entre variantes. La paridad numérica CPU/CUDA "
            "y la recuperación tienen pruebas técnicas separadas.",
            "No se mide energía ni volumen de transferencias. "
            "El escritorio gráfico permanece activo.",
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            dict(cases=len(summary), trajectories=len(records), seconds=result["internal_seconds"])
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
