"""Coste técnico de observar escrituras, con formas declaradas y sin ajustar parámetros."""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from mars_titan.learning_traces.hooks import TrainingTraces, observing
from mars_titan.learning_traces.memory import MemoryObserver
from mars_titan.learning_traces.recorder import TraceConfig, TraceRecorder
from mars_titan.models.titans.config import MemoryConfig, MemoryMultiscale
from mars_titan.models.titans.neural_memory import NeuralMemory
from mars_titan.models.titans.plasticity_config import MemoryPlasticity
from mars_titan.training.gpu_supervisor import read_gpu


def digest(module):
    h = hashlib.sha256()
    for name, value in module.named_parameters():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_memory(model, tokens, state, shared):
    return model.update_shared(tokens, state) if shared else model.update(tokens, state)


def main(output):
    assert not output.exists()
    assert not read_gpu().compute_pids, "Hay otro proceso de cómputo CUDA"
    assert torch.cuda.is_available(), "Se necesita CUDA, sin alternativa implícita CPU"
    assert torch.cuda.get_device_name(0).startswith("NVIDIA GeForce RTX 4070")
    torch.cuda.set_device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    )
    processes = []
    for line in query.splitlines():
        pid, name, used = [value.strip() for value in line.split(",")]
        processes.append(
            dict(pid=int(pid), process_name=Path(name).name, used_memory_reported=used)
        )
    started = time.perf_counter()
    output.mkdir(parents=True)
    results = []
    conditions = ("off", "every_step", "every_16")
    for size in (64, 256):
        for shared in (False, True):
            for plasticity in (False, True):
                key = (
                    f"{size}-"
                    + ("shared" if shared else "flows")
                    + ("-plasticity" if plasticity else "-plain")
                )
                config = MemoryConfig(
                    64,
                    depth=2,
                    max_batch=256,
                    max_tokens=1,
                    parameter_seed=42,
                    plasticity=MemoryPlasticity(
                        capture="eligibility", gradient_contract="scalar_controls_v2"
                    )
                    if plasticity
                    else None,
                    multiscale=MemoryMultiscale() if plasticity else None,
                )
                model = NeuralMemory(config, device="cuda:0", dtype=torch.float32)
                state = model.initial_state(1 if shared else size)
                generator = torch.Generator(device="cuda:0").manual_seed(20261020)
                tokens = (
                    torch.randn(
                        (size, 64) if shared else (size, 1, 64),
                        generator=generator,
                        device="cuda:0",
                    )
                    * 0.1
                )
                before = digest(model)

                reference = state
                for _ in range(16):
                    reference = write_memory(model, tokens, reference, shared)
                timings = {c: [] for c in conditions}
                volume = {c: 0 for c in conditions}
                peaks = {c: 0 for c in conditions}
                flushing = {c: [] for c in conditions}
                totals = {c: [] for c in conditions}
                for repeat in range(3):
                    for condition in conditions[repeat:] + conditions[:repeat]:
                        traces = observer = None
                        if condition != "off":
                            recorder = TraceRecorder(
                                output / key / f"{condition}-{repeat}",
                                TraceConfig(1 if condition == "every_step" else 16, 1 << 20),
                                {
                                    "family": "memory_write_fixture",
                                    "configuration": config.identity(),
                                },
                            )
                            observer = MemoryObserver({"core": model})
                            traces = TrainingTraces(recorder, {"core": model}, observers=[observer])
                        result = None
                        current = state
                        for _ in range(3):
                            write_memory(model, tokens, state, shared)
                        torch.cuda.synchronize(0)
                        torch.cuda.reset_peak_memory_stats(0)
                        block_start = time.perf_counter()
                        current = state
                        for step in range(1, 17):
                            begin = time.perf_counter()
                            with observing(traces, step):
                                result = write_memory(model, tokens, current, shared)
                            current = result
                            if traces is not None and traces.due(step):
                                observer.record(recorder)
                            torch.cuda.synchronize(0)
                            timings[condition].append(time.perf_counter() - begin)
                        begin = time.perf_counter()
                        if traces is not None:
                            recorder.flush(16, "technical")
                            volume[condition] += recorder.bytes
                        torch.cuda.synchronize(0)
                        flushing[condition].append(time.perf_counter() - begin)
                        totals[condition].append(time.perf_counter() - block_start)
                        peaks[condition] = max(peaks[condition], torch.cuda.max_memory_allocated(0))
                        for name in ("weights", "momentum", "levels", "modulation"):
                            for a, b in zip(
                                getattr(reference, name), getattr(result, name), strict=True
                            ):
                                assert torch.equal(a, b), (key, condition, name)
                        del a, b
                        assert torch.equal(reference.steps, result.steps)
                        assert model.observer is None
                assert before == digest(model)
                result = dict(
                    case=key,
                    rows=size,
                    shared=shared,
                    plasticity=plasticity,
                    dim=64,
                    depth=2,
                    numerically_equal=True,
                    parameters_unchanged=True,
                    conditions={
                        c: dict(
                            samples=len(timings[c]),
                            latency_seconds=dict(
                                zip(
                                    ("p50", "p95", "p99"),
                                    map(float, np.quantile(timings[c], [0.5, 0.95, 0.99])),
                                    strict=True,
                                )
                            ),
                            total_block_seconds=totals[c],
                            flush_seconds=flushing[c],
                            trace_bytes=volume[c],
                            peak_cuda_allocated_bytes=peaks[c],
                        )
                        for c in conditions
                    },
                )
                results.append(result)
                print(json.dumps({"completed": key}), flush=True)
                del model, state, tokens, reference, result, current, traces, observer, recorder
    report = dict(
        kind="memory_write_fixture_trace_cost_v1",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_files={
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "src/mars_titan/models/titans/neural_memory.py",
                "src/mars_titan/learning_traces/memory.py",
            )
        },
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        dtype="float32",
        tf32=False,
        threads=torch.get_num_threads(),
        concurrent_compute_processes=processes,
        python=platform.python_version(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        warmup=3,
        measured_steps=16,
        repetitions=3,
        condition_order="rotation",
        results=results,
        process_internal_seconds=time.perf_counter() - started,
        optimizer_steps=0,
        final_test_opened=False,
        limits=[
            "Fixtures técnicos con parámetros iniciales, sin datos financieros.",
            "Cada repetición recorre 16 escrituras desde el mismo estado inicial, "
            "con los mismos tokens. Se sincroniza cada llamada completa. "
            "No mide la canalización financiera completa.",
            "Los percentiles describen 48 llamadas por condición y caso. "
            "No certifican latencia extrema.",
            "El volcado se registra aparte y ya está incluido en los tiempos de bloque.",
            "Los bytes de trazas suman tres repeticiones. El escritorio permanece activo.",
            "No se miden energía, volumen total de transferencias ni memoria por asignación. "
            "No se recomienda una cadencia para toda la campaña.",
        ],
    )
    (output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args().output)
