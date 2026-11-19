"""Admisión conservadora de parejas medidas en PC2, sobre el controlador científico fijado."""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
from datetime import UTC, datetime

MACHINE = "mars_titan"
GPU = "NVIDIA GeForce RTX 5050 Laptop GPU"
WARMUP_RELEASE = "e23a44034c8584682397d3d5f7ac76dbc7bd8e12"
PACKED_RELEASE = "f52cc3c332ba358e7afc1772e2038f05505c2285"
PAIRS = {frozenset(("transformer", "dlinear")), frozenset(("transformer", "mars_titan_m1"))}


def eligible_pair(primary, ready):
    for other in ready[1:]:
        if other.stage != primary.stage or frozenset((primary.model, other.model)) not in PAIRS:
            continue
        # Esta pareja se midió sobre la misma vista. Dos lectores de datos distintos pueden
        # ocupar más RAM. El ensayo de Transformer y M1 sí usó dos vistas independientes.
        if "dlinear" in (primary.model, other.model) and other.year != primary.year:
            continue
        return other
    return None


def gpu_free_mib():
    try:
        result = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"],
                text=True,
                timeout=5,
            )
            .strip()
            .splitlines()
        )
        if len(result) == 1:
            name, free = result[0].rsplit(",", 1)
            if name.strip() == GPU:
                return int(free)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def host_free_mib():
    try:
        result = subprocess.check_output(
            [
                "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)",
            ],
            text=True,
            timeout=6,
        )
        return int(result.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def admission(available, per_job, gpu_mib=None, host_mib=None):
    if available is None or available < 2 * per_job:
        return "ram_wsl"
    if gpu_mib is None or gpu_mib < 5632:
        return "vram"
    if host_mib is None or host_mib < 2560:
        return "ram_windows"
    return "par"


def reason(campaign, per_job):
    free = campaign._available_memory_gib()
    if free is None or free < 2 * per_job:
        return "ram_wsl"
    return admission(free, per_job, gpu_free_mib(), host_free_mib())


def watch_resources(campaign, queue, ended, pressure):
    ticks = 0
    host = None
    while not ended.wait(2):
        try:
            available, gpu = campaign._available_memory_gib(), gpu_free_mib()
            if ticks % 8 == 0:
                host = host_free_mib()
            ticks += 1
            low = available is None or available < 2 or gpu is None or gpu < 1024
            low = low or host is None or host < 1280
        except (OSError, ValueError):
            low = True
        if low:
            pressure.set()
            queue.request_stop()
            return


def note(layout, **values):
    stamp = datetime.now(UTC).isoformat()
    with (layout.state / "pc2-pairs.jsonl").open("a", encoding="utf-8") as stream:
        print(json.dumps(dict(at_utc=stamp, **values), ensure_ascii=False), file=stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assess", action="store_true")
    args = parser.parse_args(argv)
    from mars_titan.training import comparison_campaign as c

    decl, digest = c.load_declaration()
    layout = c.Layout()
    per_job = decl["budget"]["memory_gib"][MACHINE]
    commit, changes = c.code_commit(), c.code_changes()
    if (
        commit not in (WARMUP_RELEASE, PACKED_RELEASE)
        or changes is not None
        or not c.authorized(layout, digest, commit=commit, changes=changes)
    ):
        raise SystemExit("No coincide la versión científica o su permiso")
    policy = reason(c, per_job)
    # Las parejas fueron medidas con el calentamiento residente. En la revisión que
    # también empaqueta la entrada, la cola oficial usa su presupuesto de un trabajador.
    if commit == PACKED_RELEASE and policy == "par":
        policy = "serie_entradas_verificadas"
    if args.assess:
        print(
            json.dumps(
                dict(admission=policy, per_job_gib=per_job, available=c._available_memory_gib())
            )
        )
        return 0
    if policy == "par" and any(
        ok is False for _, ok, _ in c.doctor(decl, layout, MACHINE, digest=digest, workers=2)
    ):
        policy = "doctor"
    note(
        layout, mode=policy, available_gib=c._available_memory_gib(), required_pair_gib=2 * per_job
    )
    if policy != "par":
        # La ruta de serie se delega al controlador original.
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-m",
                "mars_titan.training.comparison_campaign",
                "run",
                "--machine",
                MACHINE,
                "--device",
                "cuda:0",
            ],
        )
    plan = c.plan(decl, MACHINE)
    control = c.Queue(decl, layout, MACHINE, workers=1)
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        control.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    preflight_failures = control.preflight()
    allow_pairs = not preflight_failures
    while not stopped and not control._deadline_reached() and not control._disk_exhausted():
        records = {j.key: c.read_record(layout, j.key) for j in plan}
        ready = [j for j in plan if c.job_state(layout, j, records) in (c.PENDING, c.PAUSED)]
        if not ready:
            break
        group = [ready[0]]
        peer = eligible_pair(group[0], ready) if allow_pairs else None
        if peer is not None and reason(c, per_job) == "par":
            group.append(peer)
        control = c.Queue(decl, layout, MACHINE, workers=len(group))
        control.jobs = group
        if stopped:
            break
        note(layout, jobs=[j.key for j in group], workers=len(group))
        ended, pressure = threading.Event(), threading.Event()
        watcher = None
        if len(group) == 2:
            watcher = threading.Thread(target=watch_resources, args=(c, control, ended, pressure))
            watcher.start()
        try:
            control.run(preflight=False)
        finally:
            ended.set()
            if watcher is not None:
                watcher.join()
        if pressure.is_set():
            note(layout, mode="serial_after_pressure")
            allow_pairs = False
        elif control.stop.is_set():
            break
        after = [c.read_record(layout, j.key) for j in group]
        if any(r.get("status") == c.FAILED for r in after):
            allow_pairs = False
        if all(
            len(a["attempts"]) == len(records[j.key]["attempts"])
            for j, a in zip(group, after, strict=True)
        ):
            break
    return 0 if all(c.job_state(layout, j) == c.DONE for j in plan) else 3


if __name__ == "__main__":
    raise SystemExit(main())
