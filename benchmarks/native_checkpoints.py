"""Medir dos ejecutables completos sobre las mismas cintas y estados finales."""

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

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.episodes.worlds import WorldConfig, generate_world
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.storage import write_tape


def confirmed_results(destination):
    results = {}
    summary = json.loads((destination / "comparison.json").read_text())
    if summary["status"] != "completed":
        raise ValueError("La comparación no ha terminado")
    for item in summary["runs"]:
        report = json.loads((destination / item["path"]).read_text())
        folder = destination / item["path"]
        state = json.loads((folder.parent / report["checkpoint"]["path"]).read_text())
        results[item["name"]] = {
            "metrics": report["financial_validation"],
            "snapshot": state["snapshot"],
            "last_event": state["last_event"],
        }
    return summary, hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()


def benchmark(reference, candidate, work, repetitions):
    work.mkdir(parents=True, exist_ok=False)
    binaries = {
        "reference": reference.resolve(strict=True),
        "candidate": candidate.resolve(strict=True),
    }
    rows, workload, identities = [], [], {}
    for label, assets, sessions, context in (
        ("small", 16, 64, 16),
        ("representative", 128, 256, 64),
        ("large", 512, 320, 64),
    ):
        world = generate_world(
            WorldConfig(
                assets=assets, sessions=sessions, context=context, seed=1042, partition="validation"
            )
        )
        tape = MarketTape.from_world(world, lambda inputs: 0.002 * inputs["news"][:, 0])
        source = work / label / "input"
        write_tape(tape, source)
        workload.append(
            dict(
                name=label,
                assets=assets,
                decisions=sessions - context,
                file_bytes=(source / "market.parquet").stat().st_size,
            )
        )
        del tape, world
        expected = None
        for workers in (1, 8):
            for repeat in range(repetitions + 1):
                modes = (
                    ("reference", "candidate") if repeat % 2 == 0 else ("candidate", "reference")
                )
                for mode in modes:
                    destination = work / label / f"{mode}-{workers}-{repeat}"
                    command = [
                        str(binaries[mode]),
                        "--input",
                        str(source),
                        "--output",
                        str(destination),
                        "--compare",
                        "--workers",
                        str(workers),
                        "--diagnostic",
                    ]
                    started = time.perf_counter()
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=120,
                        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
                    )
                    total = time.perf_counter() - started
                    if result.stderr:
                        raise RuntimeError("El ejecutable produjo diagnósticos")
                    report, fingerprint = confirmed_results(destination)
                    identities[mode] = {
                        k: v
                        for k, v in report["identity"].items()
                        if k.startswith("native_")
                        or k in {"compiler_id", "compiler_version", "build_type"}
                    }
                    if expected is None:
                        expected = fingerprint
                    if fingerprint != expected:
                        raise ValueError("Cambian métricas, estado confirmado o último evento")
                    if repeat:
                        rows.append(
                            dict(
                                workload=label,
                                mode=mode,
                                workers=workers,
                                repetition=repeat,
                                total_process_seconds=total,
                                internal_seconds=report["total_seconds"],
                                peak_rss_bytes=report["executable_peak_rss_bytes"],
                                results_sha256=fingerprint,
                            )
                        )
    summaries = []
    for case in workload:
        for workers in (1, 8):
            for mode in binaries:
                matching = [
                    r
                    for r in rows
                    if (r["workload"], r["workers"], r["mode"]) == (case["name"], workers, mode)
                ]
                times = [r["total_process_seconds"] for r in matching]
                summaries.append(
                    dict(
                        workload=case["name"],
                        mode=mode,
                        workers=workers,
                        median_seconds=statistics.median(times),
                        minimum_seconds=min(times),
                        maximum_seconds=max(times),
                        std_seconds=statistics.stdev(times),
                        median_peak_rss_bytes=statistics.median(
                            r["peak_rss_bytes"] for r in matching
                        ),
                    )
                )
    return dict(
        schema_version=1,
        domain="technical",
        measured_at=datetime.now(UTC).isoformat(),
        cpu=subprocess.check_output(["rg", "-m", "1", "^model name", "/proc/cpuinfo"], text=True)
        .split(":", 1)[1]
        .strip(),
        platform=platform.platform(),
        executable_sha256={k: sha256(v) for k, v in binaries.items()},
        source_identities=identities,
        code_sha256=sha256(Path(__file__)),
        workload=workload,
        repetitions=repetitions,
        warmup_per_case=1,
        summary=summaries,
        runs=rows,
        exact_parity=True,
        final_test_opened=False,
        limits=[
            "Tiempo total de procesos con Parquet, contabilidad y checkpoints.",
            "Orden alternado y caché caliente, sin detener otras aplicaciones.",
            "La memoria utiliza VmHWM después de exec, compartida por los hilos.",
            "No se mide GPU, energía ni calidad predictiva.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if not 3 <= args.repetitions <= 10:
        raise ValueError("Se requieren entre tres y diez repeticiones")
    result = benchmark(args.reference, args.candidate, args.work, args.repetitions)
    atomic_json(args.output, result)
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
