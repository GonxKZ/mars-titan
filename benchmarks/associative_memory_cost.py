"""Medir en CPU la escritura y la lectura de B6 con las reglas delta, proximal y kalman.

Las claves y los valores son sintéticos con semilla fija y solo fijan las formas: d = 64,
m = 1 y cohortes de 16 a 8.192 resultados, el máximo que admite una escritura. No se ajusta
ni se evalúa nada. Cada caso escribe desde el mismo estado, con tres calentamientos y veinte
repeticiones, y registra percentiles, el tamaño del estado exportado y la carga del equipo.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from mars_titan.memory.associative_memory import (
    AssociativeMemory,
    AssociativeMemoryConfig,
    KalmanNoise,
    MatureFeedback,
)

SIZES = (16, 256, 1024, 8192)
NOISE = dict(
    process_noise=1e-5, observation_noise=1e-4, cohort_correlation=0.1, prior_variance=1e-5
)
RULES = {
    "delta": dict(rule="delta", rate=0.5, forgetting=0.01),
    "proximal": dict(rule="proximal", rate=1.0, forgetting=0.01),
    "kalman": dict(rule="kalman", kalman=KalmanNoise(**NOISE)),
    "kalman_huber": dict(rule="kalman", kalman=KalmanNoise(**NOISE, huber_threshold=3.0)),
}


def cohort(rows, rule, generator):
    keys = torch.randn((rows, 64), dtype=torch.float64, generator=generator)
    keys = keys / torch.linalg.vector_norm(keys, dim=1, keepdim=True)
    values = 0.01 * torch.randn((rows, 1), dtype=torch.float64, generator=generator)
    available = torch.arange(10, 10 + rows, dtype=torch.int64)
    return MatureFeedback(
        ids=torch.arange(1, rows + 1, dtype=torch.int64),
        decision_at=available - 5,
        available_at=available,
        keys=keys,
        values=values,
        weights=torch.full((rows,), 1.0 / rows, dtype=torch.float64)
        if rule == "proximal"
        else None,
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
        min_ms=ordered[0] * 1e3,
        repeats=repeats,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    cases = {}
    for name, options in RULES.items():
        config = AssociativeMemoryConfig(key_size=64, value_size=1, **options)
        generator = torch.Generator().manual_seed(20261010)
        memory = AssociativeMemory(config)
        # Un estado ya escrito evita medir la primera escritura desde cero como caso típico.
        memory = memory.write(cohort(256, config.rule, generator), cutoff=1_000_000)
        for rows in SIZES:
            feedback = cohort(rows, config.rule, generator)
            later = MatureFeedback(
                ids=feedback.ids + 10_000,
                decision_at=feedback.decision_at + 10_000,
                available_at=feedback.available_at + 10_000,
                keys=feedback.keys,
                values=feedback.values,
                weights=feedback.weights,
            )
            written = timed(
                lambda later=later, memory=memory: memory.write(later, cutoff=2_000_000),
                args.warmup,
                args.repeats,
            )
            read = timed(
                lambda keys=feedback.keys, memory=memory: memory.read(keys),
                args.warmup,
                args.repeats,
            )
            entry = dict(write=written, read=read)
            if config.rule == "kalman":
                entry["variance"] = timed(
                    lambda keys=feedback.keys, memory=memory: memory.variance(keys),
                    args.warmup,
                    args.repeats,
                )
            cases[f"{name}/{rows}"] = entry
        payload = memory.export()
        cases[f"{name}/state_bytes"] = sum(
            value.nbytes for value in payload.values() if isinstance(value, torch.Tensor)
        )
    receipt = dict(
        schema_version=1,
        kind="associative_memory_cost",
        issue=455,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        sources_sha256={
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (
                "src/mars_titan/memory/associative_memory.py",
                "benchmarks/associative_memory_cost.py",
            )
        },
        data="sintéticos con semilla fija, solo para fijar formas. Nada se ajusta ni evalúa",
        shapes=dict(key_size=64, value_size=1, cohorts=list(SIZES)),
        noise=NOISE,
        environment=dict(
            cpu=next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            ),
            torch=torch.__version__,
            torch_threads=torch.get_num_threads(),
            python=platform.python_version(),
            load_average_at_start=os.getloadavg(),
        ),
        not_measured=["GPU, porque B6 trabaja en CPU y FP64", "energía, sin instrumento"],
        cases=cases,
    )
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + "\n")
    for name, entry in cases.items():
        if isinstance(entry, dict):
            print(name, round(entry["write"]["p50_ms"], 3), round(entry["write"]["p95_ms"], 3))


if __name__ == "__main__":
    main()
