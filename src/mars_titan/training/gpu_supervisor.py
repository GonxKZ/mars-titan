"""Supervisar una cola externa con admisión CUDA y parada recuperable."""

import argparse
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class GpuSnapshot:
    total_mib: int
    free_mib: int
    compute_pids: tuple[int, ...]


@dataclass
class StopFlag:
    requested: bool = False

    def request(self, *_args):
        self.requested = True


def parse_gpu(xml):
    """Leer exclusivamente la GPU 0, sin interpretar unidades desconocidas."""
    if len(xml) > 1024**2:
        raise ValueError("El informe de GPU supera el límite de 1 MiB")
    try:
        devices = ET.fromstring(xml).findall("gpu")
    except ET.ParseError as error:
        raise ValueError("El controlador no ha devuelto XML válido") from error
    if len(devices) != 1:
        raise ValueError("La consulta debe identificar una única GPU")
    gpu = devices[0]
    values = []
    for field in ("total", "free"):
        text = gpu.findtext(f"fb_memory_usage/{field}", "").strip()
        if not re.fullmatch(r"\d+ MiB", text):
            raise ValueError("La memoria de GPU no está disponible en MiB")
        values.append(int(text.split()[0]))
    total, free = values
    if not 0 <= free <= total or total <= 0:
        raise ValueError("La memoria libre no concuerda con el total")
    processes = gpu.find("processes")
    if processes is None or (processes.text or "").strip():
        raise ValueError("El controlador no informa de los procesos de GPU")
    compute = []
    for process in processes.findall("process_info"):
        pid = process.findtext("pid", "").strip()
        kind = process.findtext("type", "").strip()
        if not pid.isdecimal() or int(pid) <= 0 or kind not in {"C", "G", "C+G"}:
            raise ValueError("El controlador no identifica un proceso de GPU")
        if kind == "C":
            compute.append(int(pid))
    return GpuSnapshot(total, free, tuple(compute))


def read_gpu():
    result = subprocess.run(
        ["nvidia-smi", "-q", "-x", "-i", "0"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return parse_gpu(result.stdout)


def memory_reason(snapshot, required_mib, child_group=None):
    if snapshot.free_mib < required_mib:
        return "memoria_insuficiente"
    for pid in snapshot.compute_pids:
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            return "otro_proceso_cuda"
        if child_group is None or group != child_group:
            return "otro_proceso_cuda"
    return None


class _Status:
    def __init__(self, path):
        self.path = path
        self.last = None
        self.saved_at = float("-inf")

    def save(self, status, starts, pauses, reason=None, snapshot=None, returncode=None):
        now = time.monotonic()
        key = status, reason, starts, pauses, returncode
        if key == self.last and now - self.saved_at < 15:
            return
        record = dict(
            schema_version=1,
            observed_at=datetime.now(UTC).isoformat(),
            status=status,
            starts=starts,
            pauses=pauses,
            reason=reason,
            returncode=returncode,
            free_mib=snapshot.free_mib if snapshot else None,
        )
        descriptor, temporary = tempfile.mkstemp(prefix=".gpu-status-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w") as output:
                json.dump(record, output, ensure_ascii=False, allow_nan=False)
                output.write("\n")
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        if key != self.last:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        self.last, self.saved_at = key, now


def _stop_child(child, timeout):
    """Enviar señales solo al grupo creado por este supervisor."""
    deadline = time.monotonic() + timeout
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return child.wait(), False
    try:
        code = child.wait(timeout=timeout)
        while time.monotonic() < deadline:
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                return code, False
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return child.wait(), True


def supervise(
    command,
    state_path,
    *,
    min_free_mib=6656,
    reserve_mib=768,
    poll_seconds=5,
    cooldown_seconds=30,
    pause_timeout=600,
    max_pauses=3,
    probe=read_gpu,
    stop=None,
):
    """Esperar memoria, ejecutar y pausar. Un fallo del comando requiere revisión."""
    integers = (min_free_mib, reserve_mib, max_pauses)
    times = (poll_seconds, cooldown_seconds, pause_timeout)
    if (
        not command
        or any(not isinstance(s, str) or not s for s in command)
        or any(type(n) is not int or n <= 0 for n in integers)
        or reserve_mib >= min_free_mib
        or max_pauses > 100
        or any(type(n) not in (int, float) or not math.isfinite(n) for n in times)
        or not 0.01 <= poll_seconds <= 60
        or not 0 <= cooldown_seconds <= 3600
        or not 0.01 <= pause_timeout <= 600
    ):
        raise ValueError("El comando o los presupuestos de supervisión no son válidos")
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = os.open(path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    child = None
    stop = stop or StopFlag()
    status = _Status(path)
    starts, pauses, ready_at = 0, 0, 0.0
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not stop.requested:
            if child is not None and child.poll() is not None:
                code = child.returncode
                _, forced = _stop_child(child, pause_timeout)
                child = None
                if forced:
                    status.save(
                        "blocked", starts, pauses, "descendiente_no_finalizado", returncode=code
                    )
                    return 75
                status.save("completed" if code == 0 else "failed", starts, pauses, returncode=code)
                return code if code >= 0 else 128 - code
            snapshot = None
            try:
                snapshot = probe()
                reason = memory_reason(
                    snapshot, reserve_mib if child else min_free_mib, child.pid if child else None
                )
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                reason = "lectura_gpu_fallida"
            if child is None:
                if snapshot and snapshot.total_mib < min_free_mib:
                    status.save(
                        "blocked", starts, pauses, "presupuesto_supera_dispositivo", snapshot
                    )
                    return 2
                if not reason and time.monotonic() >= ready_at and not stop.requested:
                    child = subprocess.Popen(command, start_new_session=True)
                    starts += 1
                    status.save("running", starts, pauses, snapshot=snapshot)
                else:
                    status.save(
                        "waiting", starts, pauses, reason or "espera_entre_intentos", snapshot
                    )
            elif reason:
                status.save("pausing", starts, pauses, reason, snapshot)
                code, forced = _stop_child(child, pause_timeout)
                child = None
                pauses += 1
                if forced or code != 0 or pauses >= max_pauses:
                    status.save(
                        "blocked",
                        starts,
                        pauses,
                        "pausa_no_recuperable" if forced or code else "limite_de_pausas",
                        snapshot,
                        code,
                    )
                    return 75
                ready_at = time.monotonic() + cooldown_seconds
                status.save("waiting", starts, pauses, reason, snapshot)
            else:
                status.save("running", starts, pauses, snapshot=snapshot)
            time.sleep(poll_seconds)
        status.save("stopping", starts, pauses)
        if child is not None:
            code, forced = _stop_child(child, pause_timeout)
            child = None
            if forced or code != 0:
                status.save("failed", starts, pauses, "parada_no_confirmada", returncode=code)
                return 75
        status.save("stopped", starts, pauses)
        return 0
    finally:
        if child is not None:
            _stop_child(child, pause_timeout)
        os.close(lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--min-free-mib", type=int, default=6656)
    parser.add_argument("--reserve-mib", type=int, default=768)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--cooldown-seconds", type=float, default=30)
    parser.add_argument("--pause-timeout", type=float, default=600)
    parser.add_argument("--max-pauses", type=int, default=3)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    stop = StopFlag()
    handlers = {sig: signal.signal(sig, stop.request) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        return supervise(
            command,
            args.state,
            min_free_mib=args.min_free_mib,
            reserve_mib=args.reserve_mib,
            poll_seconds=args.poll_seconds,
            cooldown_seconds=args.cooldown_seconds,
            pause_timeout=args.pause_timeout,
            max_pauses=args.max_pauses,
            stop=stop,
        )
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
