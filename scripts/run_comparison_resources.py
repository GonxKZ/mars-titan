"""Ejecutar una cola congelada con un presupuesto de RAM por trabajo."""

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

GIB = 1024**3
RESERVE_GIB = 4


def memory_limit(runner):
    if runner == "tabular":
        return 12
    if runner in {"fit", "reader", "ridge", "rule"}:
        return 8
    raise ValueError("El ejecutor no tiene un presupuesto de RAM declarado")


def available_memory(path=Path("/proc/meminfo")):
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields[:1] == ["MemAvailable:"]:
            if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdecimal():
                break
            return int(fields[1]) * 1024
    raise ValueError("No se puede comprobar la RAM disponible")


def memory_ready(limit_gib, available):
    if limit_gib not in {8, 12} or type(available) is not int or available < 0:
        raise ValueError("El límite o la memoria disponible no son válidos")
    return available >= (limit_gib + RESERVE_GIB) * GIB


def runtime_path(runtime):
    root = Path(runtime).resolve()
    if not (root / "src/mars_titan/training/comparison_campaign.py").is_file():
        raise ValueError("El runtime no contiene el controlador de la comparación")
    source = str(root / "src")
    sys.path.insert(0, source)
    os.environ["PYTHONPATH"] = source
    return root


def load_campaign(runtime):
    root = runtime_path(runtime)
    module = importlib.import_module("mars_titan.training.comparison_campaign")
    expected = root / "src/mars_titan/training/comparison_campaign.py"
    if Path(module.__file__).resolve() != expected:
        raise ValueError("Se ha cargado un controlador distinto del runtime solicitado")
    return module


class Status:
    def __init__(self, path, job, limit):
        self.path, self.job, self.limit = Path(path), job, limit
        self.key, self.saved = None, float("-inf")

    def save(self, phase, *, available=None, pid=None, returncode=None):
        key, now = (phase, pid, returncode), time.monotonic()
        if key == self.key and now - self.saved < 15:
            return
        record = dict(
            observed_at_utc=datetime.now(UTC).isoformat(),
            job=self.job,
            phase=phase,
            limit_gib=self.limit,
            reserve_gib=RESERVE_GIB,
            available_bytes=available,
            pid=pid,
            returncode=returncode,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".resource-status-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        if key != self.key:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        self.key, self.saved = key, now


def running_pid(record, owned):
    path = Path(record)
    if not path.exists():
        return None
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("El registro del trabajo excede el límite de lectura")
    data = json.loads(path.read_text())
    attempts = data.get("attempts", [])
    pid = attempts[-1].get("pid") if attempts else None
    return pid if data.get("status") == "running" and pid in owned else None


def run_worker(
    launcher,
    limit_gib,
    command,
    *,
    status,
    record,
    job,
    stop,
    probe=available_memory,
    poll_seconds=1,
    stop_timeout=600,
):
    from mars_titan.training import gpu_supervisor as processes

    if not command or not Path(launcher).is_file():
        raise ValueError("Falta el comando o el lanzador de recursos")
    state = Status(status, job, limit_gib)
    child, owned, generation = None, {}, None
    parent = os.getppid()
    try:
        while not stop.requested and os.getppid() == parent:
            available = probe()
            if memory_ready(limit_gib, available):
                break
            state.save("waiting_memory", available=available)
            time.sleep(poll_seconds)
        if stop.requested or os.getppid() != parent:
            state.save("paused")
            return 3
        # memslot vuelve a comprobar la memoria después de adquirir el cerrojo de GPU.
        child = subprocess.Popen(
            [str(launcher), "gpu", "--max", f"{limit_gib}G", "--", *command],
            start_new_session=True,
        )
        root = processes._process_info(child.pid)
        if root is None:
            raise RuntimeError("No se puede acreditar el proceso de recursos")
        generation = root[2]
        owned = {child.pid: generation}
        while not stop.requested and os.getppid() == parent:
            owned = processes._owned_processes(child.pid, owned, generation)
            if processes._exited(child):
                code, forced, _ = processes._stop_child(child, stop_timeout, owned, generation)
                child = None
                code = 75 if forced else code if code >= 0 else 128 - code
                phase = "completed" if code == 0 else "paused" if code == 3 else "failed"
                state.save(phase, returncode=code)
                return code
            pid = running_pid(record, owned)
            available = probe()
            phase = "running" if pid else "queued_resources"
            if not pid and not memory_ready(limit_gib, available):
                phase = "waiting_memory"
            state.save(phase, available=available, pid=pid)
            time.sleep(poll_seconds)
        code, forced, _ = processes._stop_child(child, stop_timeout, owned, generation)
        child = None
        state.save("failed" if forced else "paused", returncode=code)
        return 75 if forced else 3
    except BaseException:
        state.save("failed")
        raise
    finally:
        if child is not None:
            processes._stop_child(child, stop_timeout, owned, generation)


def make_queue(campaign, args):
    declaration, digest = campaign.load_declaration(args.declaration)
    layout = campaign.Layout(args.state, args.views)
    if not campaign.authorized(
        layout, digest, commit=campaign.code_commit(), changes=campaign.code_changes()
    ):
        raise ValueError("El runtime y la declaración no tienen un permiso coincidente")

    class ResourceQueue(campaign.Queue):
        def _command(self, job, *, preflight=False):
            original = super()._command(job, preflight=preflight)
            return [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--runtime",
                str(args.runtime),
                "--resource-launcher",
                str(args.resource_launcher),
                "--limit-gib",
                str(memory_limit(job.runner)),
                "--status",
                str(layout.state / "resource-status.json"),
                "--record",
                str(layout.record(job.key)),
                "--job",
                job.key,
                "--",
                *original,
            ]

    return ResourceQueue(
        declaration, layout, args.machine, workers=1, declaration_path=args.declaration
    )


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    worker = values[:1] == ["worker"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--resource-launcher", type=Path, required=True)
    if worker:
        parser.add_argument("--limit-gib", type=int, choices=(8, 12), required=True)
        parser.add_argument("--status", type=Path, required=True)
        parser.add_argument("--record", type=Path, required=True)
        parser.add_argument("--job", required=True)
        parser.add_argument("command", nargs=argparse.REMAINDER)
        args = parser.parse_args(values[1:])
        runtime_path(args.runtime)
        from mars_titan.training.gpu_supervisor import StopFlag

        stop = StopFlag()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop.request)
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        return run_worker(
            args.resource_launcher,
            args.limit_gib,
            command,
            status=args.status,
            record=args.record,
            job=args.job,
            stop=stop,
        )
    parser.add_argument("--machine", choices=("references",), required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--views", type=Path)
    parser.add_argument("--declaration", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args(values)
    args.runtime = args.runtime.resolve()
    args.resource_launcher = args.resource_launcher.resolve(strict=True)
    for field in ("state", "views", "declaration", "results"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.resolve())
    campaign = load_campaign(args.runtime)
    queue = make_queue(campaign, args)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, queue.request_stop)
    result = queue.run()
    campaign._print_summary(result)
    pending = any(campaign.job_state(queue.layout, job) != campaign.DONE for job in queue.jobs)
    if pending:
        return 3
    if args.results:
        command = [
            sys.executable,
            "-m",
            "mars_titan.training.comparison_campaign",
            "export-results",
            "--machine",
            args.machine,
            "--state",
            str(queue.layout.state),
            "--destination",
            str(args.results),
        ]
        if args.declaration:
            command += ["--declaration", str(args.declaration)]
        return subprocess.run(command, cwd=args.runtime, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
