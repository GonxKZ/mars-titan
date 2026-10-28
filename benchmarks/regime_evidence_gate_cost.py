"""Coste CPU de la puerta E4 y de la escritura proximal que regula, con valores de prueba."""

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from mars_titan.memory.associative_memory import (
    AssociativeMemory,
    AssociativeMemoryConfig,
    MatureFeedback,
)
from mars_titan.memory.regime_evidence_gate import MAX_ROWS, RegimeEvidenceGate, RegimeGateConfig
from mars_titan.training.regime_gate_inference import RegimeGateCorrectionInference

MODES = ("none", "cusum", "constant")


def cost_case(rows, mode, *, repeats=7, warmup=2):
    """Medir puerta, escritura de A, exportación, JSON y recuperación del mismo evento."""
    if (
        type(rows) is not int
        or not 1 <= rows <= MAX_ROWS
        or mode not in MODES
        or type(repeats) is not int
        or not 1 <= repeats <= 30
        or type(warmup) is not int
        or not 0 <= warmup <= 5
    ):
        raise ValueError(
            "La medida necesita 1–8.192 filas, un modo, 1–30 repeticiones y 0–5 calentamientos"
        )
    flows = [f"{'CN' if index % 4 == 0 else 'US'}/A{index}" for index in range(rows)]

    def cohort(at, factor):
        return tuple(
            (flow, at, at + 1, 0.0, (1 + index % 31) * factor * 0.001)
            for index, flow in enumerate(flows)
        )

    config = (
        RegimeGateConfig(min_history=1)
        if mode != "constant"
        else RegimeGateConfig(mode="constant", constant=0.5, min_history=1)
    )
    gate = RegimeEvidenceGate(config)
    for at in (1, 3):
        _, gate = gate.step(cohort(at, 1.0), cutoff=at + 1)
    # El evento medido aumenta el error un 50 % y la puerta del modo cusum se abre en parte.
    candidates = cohort(5, 1.5)
    generator = torch.Generator().manual_seed(0)
    keys = torch.randn(rows, 64, generator=generator, dtype=torch.float64)
    keys = keys / torch.linalg.vector_norm(keys, dim=1, keepdim=True)
    feedback = MatureFeedback(
        ids=torch.arange(1, rows + 1),
        decision_at=torch.full((rows,), 5),
        available_at=torch.full((rows,), 6),
        keys=keys,
        values=torch.tensor([[row[4]] for row in candidates], dtype=torch.float64),
        weights=torch.full((rows,), 1.0 / rows, dtype=torch.float64),
    )
    memory = AssociativeMemory(AssociativeMemoryConfig("proximal", rate=0.25))
    staged = [(5, flow, None, None, None) for flow in flows]

    def cycle():
        started = time.perf_counter()
        if mode == "none":
            written, state, encoded = memory.write(feedback, cutoff=6), None, b""
        else:
            decision, state = gate.step(candidates, cutoff=6)
            # Se llama al método real de E4 con las puertas de la decisión medida.
            holder = SimpleNamespace(_gates=decision.by_market())
            written = RegimeGateCorrectionInference._write_memory(
                holder, memory, feedback, staged, 6
            )
            encoded = json.dumps(state.export(), sort_keys=True, allow_nan=False).encode()
            restored = RegimeEvidenceGate.restore(config, json.loads(encoded))
        elapsed = time.perf_counter() - started
        if state is not None and restored.export() != state.export():
            raise RuntimeError("La medida alteró el estado E4 recuperado")
        return elapsed, written, state, len(encoded)

    _, expected, expected_state, _ = cycle()
    samples, byte_count = [], 0
    for index in range(warmup + repeats):
        elapsed, written, state, byte_count = cycle()
        if not torch.equal(written.matrix, expected.matrix) or (
            state is not None and state.export() != expected_state.export()
        ):
            raise RuntimeError("Las repeticiones no reproducen la misma escritura")
        if index >= warmup:
            samples.append(elapsed)
    tracemalloc.start()
    try:
        cycle()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    median = statistics.median(samples)
    return dict(
        rows=rows,
        mode=mode,
        event_gates=None if mode == "none" else gate.step(candidates, cutoff=6)[0].by_market(),
        warmup=warmup,
        repeats=repeats,
        samples_seconds=samples,
        median_seconds=median,
        minimum_seconds=min(samples),
        maximum_seconds=max(samples),
        rows_per_second_at_median=rows / median,
        serialized_state_bytes=byte_count,
        separate_traced_cycle_peak_bytes=peak,
        memory_sha256=hashlib.sha256(expected.matrix.numpy().tobytes()).hexdigest(),
        exact_repetition=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 256, 1024, 8192])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if len(args.sizes) > 4 or len(set(args.sizes)) != len(args.sizes):
        parser.error("La medida admite hasta cuatro tamaños distintos")
    started = time.perf_counter()
    cases = [
        cost_case(size, mode, repeats=args.repeats, warmup=args.warmup)
        for size in args.sizes
        for mode in MODES
    ]
    root = Path(__file__).parents[1] / "src/mars_titan"
    sources = ("memory/regime_evidence_gate.py", "training/regime_gate_inference.py")
    print(
        json.dumps(
            dict(
                schema_version=1,
                kind="regime_evidence_gate_fixture_cost",
                recorded_at_utc=datetime.now(UTC).isoformat(),
                device="cpu",
                scientific_data=False,
                python=platform.python_version(),
                torch=torch.__version__,
                torch_threads=torch.get_num_threads(),
                platform=platform.platform(),
                machine=platform.machine(),
                logical_cpus=os.cpu_count(),
                load_average=os.getloadavg(),
                source_sha256={
                    name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
                },
                benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                total_seconds=time.perf_counter() - started,
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                cases=cases,
                limits=[
                    "Son errores y claves de prueba. No hay ajustes ni resultados financieros.",
                    "La latencia incluye la puerta, la escritura proximal de A con clave 64, "
                    "la exportación, el JSON y la recuperación del estado E4.",
                    "El modo none solo escribe A, como B6 sin puerta.",
                    "Las comprobaciones de igualdad quedan fuera del intervalo interno.",
                    "tracemalloc usa un ciclo adicional separado y no mide toda la RAM.",
                    "El RSS es absoluto. No se mide GPU, energía ni la sesión completa.",
                    "Siete repeticiones no acreditan una estimación estable de p95/p99.",
                ],
            ),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
