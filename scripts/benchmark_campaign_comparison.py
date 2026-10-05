"""Medir el análisis completo con y sin reutilizar la política inicial."""

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.storage import atomic_json, outside_source, sha256


def scientific_outputs(directory):
    names = ("methods.csv", "folds.csv", "intervals.csv", "session-errors.parquet")
    signatures = {name: sha256(directory / name) for name in names}
    with (directory / "cases.csv").open() as stream:
        cases = list(csv.DictReader(stream))
    for case in cases:
        case.pop("checked_seconds")
    return signatures, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "completion", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not 3 <= args.repetitions <= 10:
        parser.error("Se necesitan entre 3 y 10 repeticiones")
    for source in (args.reference, args.completion):
        outside_source(source, args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    environment = os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    measurements, expected = [], None
    for repetition in range(args.repetitions + 1):
        # Se alterna el orden para no favorecer siempre a la misma condición.
        for cache in (0, 8) if repetition % 2 == 0 else (8, 0):
            name = f"cache-{cache}-run-{repetition}"
            output = args.output / name
            timing = args.output / f"{name}.time"
            began = time.perf_counter()
            result = subprocess.run(
                [
                    "/usr/bin/time",
                    "-f",
                    "%U %S %M",
                    "-o",
                    str(timing),
                    sys.executable,
                    "-m",
                    "mars_titan.evaluation.campaign_comparison",
                    "--reference",
                    str(args.reference),
                    "--completion",
                    str(args.completion),
                    "--output",
                    str(output),
                    "--initial-cache-mib",
                    str(cache),
                ],
                env=environment,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
            elapsed = time.perf_counter() - began
            if result.returncode:
                raise RuntimeError(result.stderr[-16000:])
            actual = scientific_outputs(output)
            if expected is None:
                expected = actual
            if actual != expected:
                raise ValueError("La caché cambia un resultado científico")
            user, system, rss = map(float, timing.read_text().split())
            report = json.loads((output / "comparison.json").read_text())
            measurements.append(
                dict(
                    cache_mib=cache,
                    repetition=repetition,
                    warmup=repetition == 0,
                    process_wall_seconds=elapsed,
                    user_seconds=user,
                    system_seconds=system,
                    peak_rss_bytes=int(rss) * 1024,
                    resources=report["resources"],
                )
            )
            print(f"{name}: {elapsed:.3f} s", flush=True)
    medians = {
        str(cache): statistics.median(
            row["process_wall_seconds"]
            for row in measurements
            if row["cache_mib"] == cache and not row["warmup"]
        )
        for cache in (0, 8)
    }
    cpu = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    atomic_json(
        args.output / "benchmark.json",
        dict(
            measured_at_utc=datetime.now(UTC).isoformat(),
            cpu=cpu,
            platform=platform.platform(),
            python=platform.python_version(),
            versions=report["versions"],
            source_sha256=report["analysis_source_sha256"],
            measurements=measurements,
            median_seconds=medians,
            speedup=medians["0"] / medians["8"],
            scientific_outputs_exact=True,
            thread_limits={"OPENBLAS_NUM_THREADS": 1, "OMP_NUM_THREADS": 1},
            gpu_used=False,
            energy_joules=None,
            monetary_cost=None,
        ),
    )


if __name__ == "__main__":
    main()
