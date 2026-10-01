"""Medir lectura Parquet y simulación completa con concurrencia acotada."""

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.episodes.storage import EpisodeSource, write_world
from mars_titan.episodes.worlds import WorldConfig, generate_world
from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.evaluation import evaluate, fixed_policy
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.storage import read_tape, write_tape


def io_counters():
    values = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":")
        values[key] = int(value)
    return values


def pipeline(root, workers):
    manifest = root / "episodes" / "manifest.json"
    with EpisodeSource(manifest) as source:
        count = len(source)
        samples = sum(row[1] for row in source.index)

    def read_positions(positions):
        values, latencies, decoded = [], [], 0
        with EpisodeSource(manifest, max_cache_bytes=8 * 1024**2) as source:
            for position in positions:
                start = time.perf_counter()
                cohort = source(int(position))
                digest = hashlib.sha256()
                for name, inputs in sorted(cohort["inputs"].items()):
                    digest.update(name.encode())
                    digest.update(inputs.tobytes())
                    decoded += inputs.nbytes
                values.append(
                    (int(position), 0.002 * cohort["inputs"]["news"][:, 0], digest.hexdigest())
                )
                latencies.append(time.perf_counter() - start)
        return values, latencies, decoded

    before = io_counters()
    started = time.perf_counter()
    ranges = np.array_split(np.arange(count), workers)
    if workers == 1:
        batches = [read_positions(ranges[0])]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batches = list(pool.map(read_positions, ranges))
    read_seconds = time.perf_counter() - started
    template = read_tape(root / "market")
    scores = template.scores.copy()
    fingerprints, latencies, decoded = [], [], 0
    for values, durations, volume in batches:
        decoded += volume
        latencies.extend(durations)
        for position, prediction, fingerprint in values:
            scores[position] = prediction
            fingerprints.append(fingerprint)
    np.testing.assert_array_equal(scores, template.scores)
    tape = MarketTape(
        template.prices,
        template.close_times,
        template.assets,
        scores,
        domain="synthetic",
        currency="USD",
        partition="train",
        parent_id=template.identity["parent_id"],
        source_identity=template.identity["source"],
        open_times=template.open_times,
    )
    result = evaluate(FinancialEnv(tape), fixed_policy("rebalance_50"))
    elapsed = time.perf_counter() - started
    after = io_counters()
    return dict(
        workers=workers,
        samples=samples,
        cohort_count=count,
        pipeline_seconds=elapsed,
        samples_per_second=samples / elapsed,
        parquet_and_input_seconds=read_seconds,
        simulation_seconds=result["elapsed_seconds"],
        decoded_input_bytes=decoded,
        cohort_latency_seconds=dict(
            zip(
                ("p50", "p95", "p99"),
                np.quantile(latencies, [0.5, 0.95, 0.99]).tolist(),
                strict=True,
            )
        ),
        io_delta={key: after[key] - before[key] for key in before},
        input_sha256=hashlib.sha256("".join(fingerprints).encode()).hexdigest(),
        financial_validation=result["financial_validation"],
        process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        peak_vram_bytes=None,
        cpu_gpu_transfer_bytes=None,
        energy_joules=None,
        cost_currency=None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        start = time.perf_counter()
        pipeline(args.work, args.worker)
        warmup = time.perf_counter() - start
        result = pipeline(args.work, args.worker)
        result["warmup_seconds"] = warmup
        print(json.dumps(result))
        return
    if args.output is None or not 2 <= args.repetitions <= 10:
        raise ValueError("El benchmark necesita salida y entre dos y diez repeticiones")
    args.work.mkdir(parents=True, exist_ok=False)
    config = WorldConfig(assets=128, sessions=256, context=64, seed=4242)
    world = generate_world(config)
    started = time.perf_counter()
    write_world(world, args.work / "episodes")
    write_tape(
        MarketTape.from_world(world, lambda inputs: 0.002 * inputs["news"][:, 0]),
        args.work / "market",
    )
    preparation_seconds = time.perf_counter() - started
    runs = []
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    for workers in (1, 2, 4, 8):
        for _ in range(args.repetitions):
            start = time.perf_counter()
            completed = subprocess.run(
                [sys.executable, __file__, "--work", str(args.work), "--worker", str(workers)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
            )
            measured = json.loads(completed.stdout)
            measured["total_process_seconds"] = time.perf_counter() - start
            runs.append(measured)
    reference = runs[0]
    for result in runs:
        if (
            result["input_sha256"] != reference["input_sha256"]
            or result["financial_validation"] != reference["financial_validation"]
        ):
            raise ValueError("La concurrencia altera entradas o resultados contables")
    profiler = cProfile.Profile()
    profiler.runcall(pipeline, args.work, 1)
    stats = pstats.Stats(profiler)
    profile = []
    for (file, line, function), values in sorted(
        stats.stats.items(), key=lambda entry: entry[1][3], reverse=True
    ):
        if "mars_titan" in file or "episode_pipeline" in file:
            profile.append(
                dict(
                    file=Path(file).name,
                    line=line,
                    function=function,
                    calls=values[1],
                    self_seconds=values[2],
                    cumulative_seconds=values[3],
                )
            )
            if len(profile) >= 12:
                break
    summary = []
    for workers in (1, 2, 4, 8):
        group = [row for row in runs if row["workers"] == workers]
        times = [row["pipeline_seconds"] for row in group]
        summary.append(
            dict(
                workers=workers,
                median_seconds=float(np.median(times)),
                min_seconds=min(times),
                max_seconds=max(times),
                std_seconds=float(np.std(times, ddof=1)),
                median_peak_rss_bytes=float(
                    np.median([row["process_lifetime_peak_rss_bytes"] for row in group])
                ),
            )
        )
    process_info = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    cpu_info = (
        subprocess.check_output(["rg", "-m", "1", "^model name", "/proc/cpuinfo"], text=True)
        .strip()
        .split(":", 1)[1]
        .strip()
    )
    report = dict(
        schema_version=1,
        measured_at=datetime.now(UTC).isoformat(),
        domain="technical",
        final_test_opened=False,
        workload=dict(
            assets=128,
            sessions=256,
            context=64,
            observations="analytic",
            admitted_modalities=list(world.shapes),
        ),
        hardware=dict(
            cpu=cpu_info,
            architecture=platform.machine(),
            os=platform.platform(),
            other_cuda_processes=len(process_info.stdout.splitlines())
            if process_info.returncode == 0
            else None,
        ),
        software=dict(
            python=platform.python_version(), numpy=np.__version__, pyarrow=pa.__version__
        ),
        concurrency=dict(
            worker_options=[1, 2, 4, 8],
            pending_tasks="como máximo un bloque por trabajador",
            arrow_threads=False,
            numerical_threads=1,
            cache_bytes_per_worker=8 * 1024**2,
        ),
        preparation_seconds=preparation_seconds,
        repetitions=args.repetitions,
        warmup_runs_per_process=1,
        fastest_measured_workers=min(summary, key=lambda row: row["median_seconds"])["workers"],
        code_sha256={
            str(path): sha256(path)
            for path in (
                Path(__file__).relative_to(Path.cwd()),
                Path("src/mars_titan/episodes/storage.py"),
                Path("src/mars_titan/simulation/portfolio.py"),
                Path("src/mars_titan/simulation/environment.py"),
                Path("src/mars_titan/simulation/evaluation.py"),
            )
        },
        summary=summary,
        runs=runs,
        profile=profile,
        limits=[
            "Carga sintética analítica, sin codificadores ni entrenamiento neuronal.",
            "Caché del sistema caliente, sin vaciar cachés ni detener otros procesos.",
            "El tiempo total incluye importaciones y calentamiento. El recorrido los excluye.",
            "No se mide GPU, energía ni coste monetario. El perfil usa otra ejecución.",
        ],
    )
    atomic_json(args.output, report)
    print(
        json.dumps(
            dict(fastest_measured_workers=report["fastest_measured_workers"], summary=summary)
        )
    )


if __name__ == "__main__":
    main()
