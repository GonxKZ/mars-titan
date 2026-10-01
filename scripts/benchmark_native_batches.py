"""Comparar el recorrido contable nativo con 1, 2, 4 y 8 trabajadores."""

import argparse
import json
import platform
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.storage import atomic_json, sha256

SHAPES = ((16, 8, 64), (256, 64, 128), (2048, 128, 128), (4096, 64, 128))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, choices=range(1, 10), default=3)
    parser.add_argument("--concurrent-workload", required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    backend = binary.with_name("libmars_titan_simulation.so")
    if not binary.is_file() or args.output.exists():
        raise ValueError("Se necesita un ejecutable y una salida nueva")
    hardware = (
        json.loads(subprocess.check_output(["lscpu", "-J"], text=True, timeout=30))
        if shutil.which("lscpu")
        else None
    )
    report = dict(
        schema_version=1,
        measured_at=datetime.now(UTC).isoformat(),
        domain="technical",
        binary_sha256=sha256(binary),
        backend_sha256=sha256(backend) if backend.is_file() else None,
        benchmark_source_sha256=sha256(
            Path(__file__).resolve().parents[1] / "native" / "benchmarks" / "batch_rollout.cpp"
        ),
        hardware=hardware,
        platform=platform.platform(),
        concurrent_workload=args.concurrent_workload,
        repetitions=args.repetitions,
        warmup="Un paso seguido de reset, incluido en tiempo total y excluido del rollout.",
        limits=[
            "La cinta compartida no produce trayectorias independientes "
            "ni resultados de aprendizaje.",
            "No se miden energía, coste monetario, asignaciones ni transferencias GPU.",
            "VmHWM mide RSS del proceso en Linux, cero indica que no está disponible.",
        ],
        records=[],
    )
    for environments, assets, sessions in SHAPES:
        expected_checksum = None
        for repetition in range(args.repetitions):
            modes = ("reference", 1, 2, 4, 8)
            offset = repetition % len(modes)
            for mode in modes[offset:] + modes[:offset]:
                command = [
                    str(binary),
                    "--environments",
                    str(environments),
                    "--assets",
                    str(assets),
                    "--sessions",
                    str(sessions),
                ]
                command += ["--reference"] if mode == "reference" else ["--workers", str(mode)]
                started = time.perf_counter()
                process = subprocess.run(
                    command, check=True, capture_output=True, text=True, timeout=120
                )
                process_seconds = time.perf_counter() - started
                record = json.loads(process.stdout)
                record.update(process_seconds=process_seconds, repetition=repetition)
                if expected_checksum is None:
                    expected_checksum = record["checksum"]
                if record["checksum"] != expected_checksum:
                    raise ValueError("Las observaciones o recompensas difieren entre ejecuciones")
                report["records"].append(record)
                atomic_json(args.output, report)
                print(
                    f"Entornos={environments}, activos={assets}, modo={mode}, "
                    f"repetición={repetition + 1}, tiempo={record['total_seconds']:.6f} s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
