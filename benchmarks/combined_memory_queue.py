"""Ejecutar y reanudar el catálogo dentro de su presupuesto y un plazo absoluto."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.learning_hold import require_learning_allowed

from .combined_memory_study import configuration, remaining_seconds, verify_prepared


def manifest(work, job):
    return work / "runs" / job["id"] / ("window.json" if job["kind"] == "candidate" else "run.json")


def status(work, job):
    path = manifest(work, job)
    return json.loads(path.read_text()).get("status") if path.exists() else None


def checkpoint(work, job):
    path = work / "runs" / job["id"] / "fit/checkpoints/latest.json"
    if not path.exists():
        return None
    recent = json.loads(path.read_text()).get("latest", [])
    return recent[0]["sha256"] if recent else None


def launch_job(work, runtime, job, timeout, log):
    """Un proceso aislado. La reserva GPU se adquiere al lanzar la cola completa."""
    environment = dict(os.environ, PYTHONPATH=f"{runtime / 'src'}:{runtime}")
    command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "benchmarks.combined_memory_study",
        "run",
        "--work",
        str(work),
        "--job",
        job["id"],
    ]
    started, timed_out = time.monotonic(), False
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=runtime,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        code = process.returncode
    return dict(exit_code=code, timeout=timed_out, wall_seconds=time.monotonic() - started)


def run_queue(work, runtime, deadline, *, launch=launch_job):
    require_learning_allowed("la cola acotada de combinaciones")
    if deadline.tzinfo is None:
        raise ValueError("El plazo necesita una zona horaria explícita")
    work, runtime = Path(work).resolve(), Path(runtime).resolve()
    receipt = verify_prepared(work)
    with (work / ".queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Ya hay una cola ejecutando este estudio") from error
        return _run_locked(work, runtime, deadline, receipt, launch)


def _run_locked(work, runtime, deadline, receipt, launch):
    folder = work / "execution"
    folder.mkdir(exist_ok=True)
    events = []

    def emit(**fields):
        record = dict(at_utc=datetime.now(UTC).isoformat(), **fields)
        events.append(record)
        with (folder / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    for job in receipt["jobs"]:
        if status(work, job) == "completed":
            continue
        if "parent" in job:
            parent = next(j for j in receipt["jobs"] if j["id"] == job["parent"])
            if status(work, parent) != "completed":
                emit(event="pending_dependency", job=job["id"], parent=job["parent"])
                continue
        while True:
            left = min(remaining_seconds(work), (deadline - datetime.now(UTC)).total_seconds())
            if left <= 0:
                emit(event="budget_exhausted", job=job["id"])
                return dict(status="budget_exhausted", events=events)
            before = checkpoint(work, job)
            measurements = set((work / "measurements").glob("*.json"))
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            label = job["id"].replace("/", "_") + "-" + stamp
            emit(event="started", job=job["id"], remaining_seconds=left, checkpoint=before)
            began = time.monotonic()
            try:
                result = launch(
                    work,
                    runtime,
                    job,
                    min(configuration()["max_job_seconds"] + 45, left),
                    folder / (label + ".log"),
                )
            except BaseException:
                if not set((work / "measurements").glob("*.json")) - measurements:
                    (work / "measurements").mkdir(exist_ok=True)
                    atomic_json(
                        work / "measurements" / (label + "-interrupted.json"),
                        dict(
                            job=job,
                            status="interrupted",
                            process_seconds=time.monotonic() - began,
                            optimizer_steps=None,
                            prepared_sha256=sha256(work / "prepared.json"),
                            final_test_opened=False,
                        ),
                    )
                emit(event="interrupted", job=job["id"])
                raise
            current = status(work, job)
            if not set((work / "measurements").glob("*.json")) - measurements:
                (work / "measurements").mkdir(exist_ok=True)
                atomic_json(
                    work / "measurements" / (label + "-controller.json"),
                    dict(
                        job=job,
                        status="terminated" if result["timeout"] else "failed_before_receipt",
                        process_seconds=result["wall_seconds"],
                        optimizer_steps=None,
                        prepared_sha256=sha256(work / "prepared.json"),
                        final_test_opened=False,
                    ),
                )
            emit(event="finished", job=job["id"], status=current, **result)
            if result["timeout"]:
                return dict(status="budget_exhausted", events=events)
            if result["exit_code"]:
                raise RuntimeError(f"Falló {job['id']}. Consultar {folder / (label + '.log')}")
            if current == "completed":
                break
            if current != "paused" or checkpoint(work, job) == before:
                return dict(status="paused_without_progress", events=events)
            emit(event="resume_pending", job=job["id"], checkpoint=checkpoint(work, job))
    pending = [j["id"] for j in receipt["jobs"] if status(work, j) != "completed"]
    return dict(status="incomplete" if pending else "completed", pending=pending, events=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--deadline-utc", required=True, type=datetime.fromisoformat)
    args = parser.parse_args()
    result = run_queue(args.work, args.runtime, args.deadline_utc)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "events"}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
