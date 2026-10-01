"""Comparar el ejecutable C++ y la referencia con precios y políticas idénticos."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mars_titan.data.storage import atomic_json, sha256


def executable_peak_rss():
    with Path("/proc/self/status").open() as stream:
        for line in stream.read(32768).splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("El sistema no informa VmHWM")


def python_reference(source, output):
    from mars_titan.simulation.environment import FinancialEnv
    from mars_titan.simulation.evaluation import evaluate, fixed_policy
    from mars_titan.simulation.storage import read_tape

    started = time.perf_counter()
    tape = read_tape(source)
    results = {}
    for policy in ("cash", "hold_initial", "rebalance_50"):
        for cost in (0, 10, 25):
            report = evaluate(FinancialEnv(tape, cost_bps=cost), fixed_policy(policy))
            results[f"{policy}-cost-{cost}"] = report["financial_validation"]
    elapsed = time.perf_counter() - started
    atomic_json(
        output,
        dict(
            metrics=results,
            internal_seconds=elapsed,
            process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            executable_peak_rss_bytes=executable_peak_rss(),
        ),
    )


def _metrics_cpp(folder):
    return {
        f"{policy}-cost-{cost}": json.loads(
            (folder / f"{policy}-cost-{cost}" / "run.json").read_text()
        )["financial_validation"]
        for policy in ("cash", "hold_initial", "rebalance_50")
        for cost in (0, 10, 25)
    }


def _sizes(path):
    names = subprocess.check_output(
        ["rg", "--files", "--hidden", "--no-ignore", str(path)], text=True
    ).splitlines()
    return sum(Path(name).stat().st_size for name in names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--python-worker", type=Path)
    args = parser.parse_args()
    if args.python_worker:
        python_reference(args.python_worker, args.output)
        return
    if args.binary is None or args.work is None or not 3 <= args.repetitions <= 10:
        raise ValueError("El benchmark necesita binario, carpeta nueva y entre 3 y 10 repeticiones")
    binary = args.binary.resolve(strict=True)
    args.work.mkdir(parents=True, exist_ok=False)
    from mars_titan.episodes.worlds import WorldConfig, generate_world
    from mars_titan.simulation.market import MarketTape
    from mars_titan.simulation.storage import write_tape

    world = generate_world(
        WorldConfig(assets=128, sessions=256, context=64, seed=1042, partition="validation")
    )
    tape = MarketTape.from_world(world, lambda inputs: 0.002 * inputs["news"][:, 0])
    source = args.work / "input"
    write_tape(tape, source)
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    timings, expected, maximum_error = [], None, 0.0
    native_identity = None
    initial_load = os.getloadavg()
    for implementation, workers in (("python", 1), ("cpp", 1), ("cpp", 2), ("cpp", 4), ("cpp", 8)):
        for repetition in range(args.repetitions + 1):
            destination = args.work / f"{implementation}-{workers}-{repetition}"
            result_path = destination.with_suffix(".json")
            command = (
                [
                    sys.executable,
                    __file__,
                    "--python-worker",
                    str(source),
                    "--output",
                    str(result_path),
                ]
                if implementation == "python"
                else [
                    str(binary),
                    "--input",
                    str(source),
                    "--output",
                    str(destination),
                    "--compare",
                    "--workers",
                    str(workers),
                    "--diagnostic",
                ]
            )
            started = time.perf_counter()
            result = subprocess.run(
                command, check=True, capture_output=True, text=True, env=environment, timeout=120
            )
            total = time.perf_counter() - started
            if implementation == "python":
                report = json.loads(result_path.read_text())
                metrics = report["metrics"]
                internal, peak = report["internal_seconds"], report["executable_peak_rss_bytes"]
                written = result_path.stat().st_size
            else:
                report = json.loads((destination / "comparison.json").read_text())
                metrics = _metrics_cpp(destination)
                internal, peak = report["total_seconds"], report["executable_peak_rss_bytes"]
                native_identity = report["identity"]
                written = _sizes(destination)
            if expected is None:
                expected = metrics
            for name, values in metrics.items():
                for key, value in values.items():
                    reference = expected[name][key]
                    if type(reference) in (int, float):
                        np.testing.assert_allclose(value, reference, rtol=1e-12, atol=1e-10)
                        maximum_error = max(maximum_error, abs(value - reference))
                    elif value != reference:
                        raise ValueError("El ejecutable cambia el resultado o su admisión")
            if repetition:
                timings.append(
                    dict(
                        implementation=implementation,
                        workers=workers,
                        repetition=repetition,
                        total_process_seconds=total,
                        internal_seconds=internal,
                        process_peak_rss_bytes=peak,
                        persisted_bytes=written,
                    )
                )
            if result.stderr:
                raise RuntimeError("El proceso produjo diagnósticos durante la medición")
    summary = []
    for implementation, workers in (("python", 1), ("cpp", 1), ("cpp", 2), ("cpp", 4), ("cpp", 8)):
        group = [
            row
            for row in timings
            if (row["implementation"], row["workers"]) == (implementation, workers)
        ]
        values = [row["total_process_seconds"] for row in group]
        summary.append(
            dict(
                implementation=implementation,
                workers=workers,
                median_seconds=statistics.median(values),
                minimum_seconds=min(values),
                maximum_seconds=max(values),
                std_seconds=statistics.stdev(values),
                portfolio_decisions_per_second=9 * 192 / statistics.median(values),
                asset_decisions_per_second=9 * 192 * 128 / statistics.median(values),
                median_peak_rss_bytes=statistics.median(
                    row["process_peak_rss_bytes"] for row in group
                ),
            )
        )
    report = dict(
        schema_version=1,
        measured_at=datetime.now(UTC).isoformat(),
        domain="technical",
        workload=dict(
            assets=128,
            sessions=193,
            decisions=192,
            scenarios=9,
            parent="fixed_signal_reference",
            input_file_bytes=(source / "market.parquet").stat().st_size,
        ),
        source_sha256=sha256(source / "manifest.json"),
        executable_sha256=sha256(binary),
        benchmark_sha256=sha256(Path(__file__)),
        financial_metrics_sha256=hashlib.sha256(
            json.dumps(expected, sort_keys=True).encode()
        ).hexdigest(),
        cpu=subprocess.check_output(["rg", "-m", "1", "^model name", "/proc/cpuinfo"], text=True)
        .split(":", 1)[1]
        .strip(),
        platform=platform.platform(),
        python=platform.python_version(),
        libraries={name: importlib.metadata.version(name) for name in ("numpy", "pyarrow")},
        native_identity=native_identity,
        logical_cpus=os.cpu_count(),
        load_average_before=initial_load,
        load_average_after=os.getloadavg(),
        peak_rss_method="linux_proc_self_status_VmHWM_after_exec",
        repetitions=args.repetitions,
        warmup_per_configuration=1,
        summary=summary,
        runs=timings,
        parity=dict(
            relative_tolerance=1e-12,
            absolute_tolerance=1e-10,
            maximum_absolute_error=maximum_error,
            all_scenarios=True,
        ),
        peak_vram_bytes=None,
        cpu_gpu_transfer_bytes=None,
        energy_joules=None,
        monetary_cost=None,
        final_test_opened=False,
        limits=[
            "La referencia Python usa evaluate y persiste un resumen, sin checkpoints intermedios.",
            "C++ incluye checkpoints, bloqueo y recibos en su tiempo interno.",
            "El tiempo total incluye arranque, bibliotecas, lectura y salida de datos sintéticos.",
            "La caché del sistema permanece caliente. No se detienen otras aplicaciones.",
            "VmHWM mide el máximo observado por Linux desde exec, compartido por todos los hilos.",
            "No se mide entrenamiento neuronal, GPU, energía ni coste monetario.",
        ],
    )
    atomic_json(args.output, report)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
