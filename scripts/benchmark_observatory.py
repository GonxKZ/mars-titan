"""Reproduce en Linux una exportación sintética en CPU, sin entrenamientos ni red."""

import hashlib
import json
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "scripts/export_observatory.py"
STAMP = "2020-01-02T12:00:00Z"
LIMITS = {
    "max_runs": 128,
    "max_file_bytes": 262144,
    "max_output_bytes": 8388608,
    "timeout_seconds": 5,
}


def build_fixture(directory):
    """Crea 32 estados nuevos y calcula su huella por ruta relativa, NUL y contenido."""
    directory.mkdir()
    common = {
        "schema_version": 1,
        "attempt_id": "synthetic-attempt-01",
        "model_id": "M3",
        "variant_id": "synthetic-benchmark-v1",
        "status": "completed",
        "phase": "validation",
        "heartbeat_at": STAMP,
        "started_at": STAMP,
        "updated_at": STAMP,
        "completed_steps": 500,
        "total_steps": 500,
        "metrics": {"loss": 0.03, "mae": 0.02},
        "history": [
            {"step": step, "recorded_at": STAMP, "loss": 0.03, "mae": 0.02} for step in range(500)
        ],
    }
    digest, size = hashlib.sha256(), 0
    for index in range(32):
        run_id = f"synthetic-run-{index:03d}"
        relative = Path(run_id) / "status.json"
        body = (json.dumps(common | {"run_id": run_id}, sort_keys=True) + "\n").encode()
        (directory / run_id).mkdir()
        (directory / relative).write_bytes(body)
        digest.update(relative.as_posix().encode() + b"\0" + body)
        size += len(body)
    return {
        "runs": 32,
        "history_points_per_run": 500,
        "input_bytes": size,
        "sha256": digest.hexdigest(),
    }


def measure_exports(source, output):
    """Mide cinco CLI reales y captura RSS antes de consultar metadatos con subprocess."""
    command = [sys.executable, str(EXPORTER), "--input-dir", str(source), "--output", str(output)]
    for name, value in LIMITS.items():
        command.extend(("--" + name.replace("_", "-"), str(value)))
    rss_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    elapsed, failures, output_bytes = [], [], None
    for attempt in range(1, 6):
        start = time.perf_counter()
        code, reason = None, None
        try:
            process = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=15,
            )
            code = process.returncode
            duration = time.perf_counter() - start
            if code:
                reason = "export_failed"
            else:
                document = json.loads(output.read_bytes())
                runs = document["runs"]
                if len(runs) != 32 or any(len(run["history"]) != 500 for run in runs):
                    raise ValueError("La salida del caso de prueba no es la esperada")
                output_bytes = output.stat().st_size
                elapsed.append(duration)
        except subprocess.TimeoutExpired:
            reason = "subprocess_timeout"
        except (OSError, ValueError, KeyError, TypeError):
            reason = "invalid_output_or_io_error"
        if reason:
            failures.append(
                {
                    "attempt": attempt,
                    "returncode": code,
                    "reason": reason,
                    "elapsed_seconds": time.perf_counter() - start,
                }
            )
    return {
        "repetitions": 5,
        "elapsed_seconds": elapsed,
        "median_seconds": statistics.median(elapsed) if elapsed else None,
        "children_rss_before_kib": rss_before,
        "max_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "output_bytes": output_bytes,
        "failed_attempts": failures,
    }


def provenance():
    """Recoge procedencia después de terminar la medida de los procesos hijos."""
    cpu = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    paths = ["scripts/export_observatory.py", "scripts/benchmark_observatory.py"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal", "--", *paths],
        cwd=ROOT,
        text=True,
        timeout=5,
    )
    return {
        "commit": commit.strip(),
        "measured_sources_clean": not dirty.strip(),
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_model": cpu,
        },
        "sha256": {
            "exporter": hashlib.sha256(EXPORTER.read_bytes()).hexdigest(),
            "generator": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }


def main():
    if sys.platform != "linux":
        raise SystemExit("Este benchmark requiere Linux y /proc.")
    with tempfile.TemporaryDirectory(prefix="mars-observatory-benchmark-") as temporary:
        root = Path(temporary)
        fixture = build_fixture(root / "runs")
        measurements = measure_exports(root / "runs", root / "public.json")
    report = (
        provenance()
        | measurements
        | {
            "measurement": "synthetic_cpu_export_not_training",
            "recorded_at": datetime.now(UTC).isoformat(),
            "command": "uv run --locked python scripts/benchmark_observatory.py",
            "fixture": fixture,
            "limits": LIMITS | {"subprocess_timeout_seconds": 15},
            "timing_scope": "CLI con arranque Python, sin comprobación posterior del JSON",
            "rss_scope": "Máximo RUSAGE_CHILDREN en KiB antes de consultar metadatos",
            "fixture_hash_scope": "SHA-256: ruta relativa UTF-8, NUL y bytes JSON, por run_id",
        }
    )
    report["sha256"]["fixture"] = fixture["sha256"]
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if measurements["failed_attempts"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
