"""Coste CPU del selector E3 y su recuperación JSON con errores literales de prueba."""

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

from mars_titan.memory.salience_budget import (
    MAX_ROWS,
    RULES,
    MatureWrite,
    MatureWriteBudget,
    SalienceBudgetConfig,
)


def cost_case(rows, rule, *, repeats=7, warmup=2):
    """Medir selección, exportación, JSON y recuperación sobre el mismo prefijo fijado."""
    if (
        type(rows) is not int
        or not 1 <= rows <= MAX_ROWS
        or type(repeats) is not int
        or not 1 <= repeats <= 30
        or type(warmup) is not int
        or not 0 <= warmup <= 5
    ):
        raise ValueError("La medida necesita 1–8.192 filas, 1–30 repeticiones y 0–5 calentamientos")
    config = SalienceBudgetConfig(rule=rule, writes_per_market=32, min_history=2)
    run = MatureWriteBudget(config)

    def cohort(at):
        return tuple(
            MatureWrite(
                f"{'CN' if index % 4 == 0 else 'US'}/A{index}",
                at,
                at + 1,
                0.001 * ((index % 3) - 1),
                (1 + index % 31) * (at % 3 + 1) * 0.001,
            )
            for index in range(rows)
        )

    for at in (1, 3):
        _, run = run.select(cohort(at), cutoff=at + 1)
    candidates = cohort(5)
    expected, following = run.select(candidates, cutoff=6)
    expected_state = following.export()

    def cycle():
        started = time.perf_counter()
        selection, state = run.select(candidates, cutoff=6)
        encoded = json.dumps(state.export(), sort_keys=True, allow_nan=False).encode()
        restored = MatureWriteBudget.restore(config, json.loads(encoded))
        elapsed = time.perf_counter() - started
        if selection != expected or restored.export() != expected_state:
            raise RuntimeError("La medida alteró las identidades o el estado recuperado")
        return elapsed, len(encoded)

    samples, byte_count = [], None
    for index in range(warmup + repeats):
        elapsed, byte_count = cycle()
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
        rule=rule,
        selected=len(expected.selected),
        warmup=warmup,
        repeats=repeats,
        samples_seconds=samples,
        median_seconds=median,
        minimum_seconds=min(samples),
        maximum_seconds=max(samples),
        candidates_per_second_at_median=rows / median,
        serialized_state_bytes=byte_count,
        separate_traced_cycle_peak_bytes=peak,
        state_sha256=expected_state["sha256"],
        exact_recovery=True,
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
        cost_case(size, rule, repeats=args.repeats, warmup=args.warmup)
        for size in args.sizes
        for rule in RULES
    ]
    for size in args.sizes:
        if len({item["selected"] for item in cases if item["rows"] == size}) != 1:
            raise RuntimeError("Los controles no conservan el mismo cupo")
    source = Path(__file__).parents[1] / "src/mars_titan/memory/salience_budget.py"
    print(
        json.dumps(
            dict(
                schema_version=1,
                kind="salience_budget_fixture_cost",
                recorded_at_utc=datetime.now(UTC).isoformat(),
                device="cpu",
                scientific_data=False,
                python=platform.python_version(),
                platform=platform.platform(),
                machine=platform.machine(),
                logical_cpus=os.cpu_count(),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                total_seconds=time.perf_counter() - started,
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                cases=cases,
                limits=[
                    "Son candidatos con errores literales. No hay ajustes ni resultados "
                    "financieros.",
                    "La latencia incluye selección, exportación, JSON y recuperación, "
                    "sin publicación de banco.",
                    "Las verificaciones de igualdad quedan fuera del intervalo interno "
                    "y dentro del proceso.",
                    "tracemalloc usa un ciclo adicional separado y no mide toda la RAM "
                    "del proceso.",
                    "El RSS es absoluto. No se mide GPU, energía, sesiones completas "
                    "ni colas de etiquetas.",
                    "Siete repeticiones no acreditan una estimación estable de las colas p95/p99.",
                ],
            ),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
