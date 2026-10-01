"""Comprobar la salud de una campaña externa sin iniciar ni detener cargas."""

import argparse
import fcntl
import json
import math
import os
import re
import select
import stat
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

MAX_JSON_BYTES = 256 * 1024
MAX_PROCESSES = 128
PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "Result",
    "MainPID",
    "InvocationID",
    "ControlGroup",
    "ExecMainStartTimestampMonotonic",
    "ExecMainStatus",
)
COUNTERS = ("completed_runs", "completed_parents", "global_step", "completed_steps", "partitions")
MESSAGES = {
    "service_failed": "El servicio de la campaña ha fallado. Revisar su journal.",
    "supervisor_failed": "El supervisor ha registrado un fallo o un bloqueo.",
    "stopped": "La campaña se ha detenido sin un resumen final completado.",
    "observation_failed": "No se ha podido consultar el servicio de la campaña.",
    "possible_inactivity": "No se observa avance de CPU, I/O ni contadores científicos.",
}


def _read(path, limit):
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("La entrada no es un archivo regular")
        content = source.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f"La entrada supera el límite de {limit} bytes")
    return content, metadata


def _json(path):
    content, metadata = _read(path, MAX_JSON_BYTES)
    try:
        value = json.loads(content)
    except RecursionError as error:
        raise ValueError("La entrada JSON tiene demasiados niveles") from error
    if not isinstance(value, dict):
        raise ValueError("La entrada JSON debe ser un objeto")
    return value, metadata


def _number(value):
    if type(value) is int:
        return 0 <= value <= 2**63 - 1
    return type(value) is float and math.isfinite(value) and 0 <= value <= 2**63 - 1


def read_report(path):
    """Leer solo campos operativos. Las fechas no cuentan como progreso."""
    value, metadata = _json(path)
    report = {"path": str(path), "mtime": metadata.st_mtime}
    for key in ("status", "phase", "reason", "returncode"):
        if value.get(key) is not None and not isinstance(value[key], (str, int)):
            raise ValueError(f"El campo {key} tiene un tipo no admitido")
        if isinstance(value.get(key), (str, int)):
            if isinstance(value[key], str) and len(value[key]) > 256:
                raise ValueError(f"El campo {key} supera el límite de 256 caracteres")
            report[key] = value[key]
    for key in COUNTERS:
        if _number(value.get(key)):
            report[key] = value[key]
        elif key in value and not (key == "partitions" and isinstance(value[key], dict)):
            raise ValueError(f"El contador {key} no es un número acotado")
    partitions = value.get("partitions")
    if isinstance(partitions, dict):
        report["partitions"] = len(partitions.keys() & {"train", "validation"})
    runs = value.get("runs", [])
    if isinstance(runs, list):
        report["training"] = any(
            isinstance(run, dict) and run.get("status") == "running" for run in runs
        )
    report["training"] = report.get("training", False) or value.get("phase") in {
        "train",
        "training",
    }
    observed = value.get("observed_at")
    if isinstance(observed, str):
        parsed = datetime.fromisoformat(observed)
        if parsed.tzinfo is None:
            raise ValueError("La fecha del supervisor no tiene zona horaria")
        report["observed_at"] = parsed.timestamp()
    return report


def _command(arguments, *, timeout=5):
    child = subprocess.Popen(
        arguments,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=os.environ | {"LC_ALL": "C"},
    )
    output = bytearray()
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([child.stdout], [], [], remaining)[0]:
                raise subprocess.TimeoutExpired(arguments, timeout)
            chunk = os.read(child.stdout.fileno(), 65537 - len(output))
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > 65536:
                raise ValueError("La consulta supera el límite de 64 KiB")
        code = child.wait(timeout=max(0, deadline - time.monotonic()))
        if code:
            raise subprocess.CalledProcessError(code, arguments)
        return output.decode()
    finally:
        child.stdout.close()
        if child.poll() is None:
            child.kill()
        child.wait()


def read_service(name):
    properties = dict(
        line.split("=", 1)
        for line in _command(
            [
                "systemctl",
                "--user",
                "show",
                "--no-pager",
                "--property=" + ",".join(PROPERTIES),
                name,
            ]
        ).splitlines()
        if "=" in line
    )
    if properties.get("LoadState") == "not-found":
        raise ValueError("El servicio configurado no existe")
    if properties.get("ActiveState") not in {
        "active",
        "inactive",
        "failed",
        "activating",
        "deactivating",
        "reloading",
    }:
        raise ValueError("systemd no devuelve un estado de servicio reconocido")
    return dict(
        name=name,
        active=properties["ActiveState"],
        sub=properties.get("SubState", ""),
        result=properties.get("Result", ""),
        pid=int(properties.get("MainPID", 0)),
        invocation=properties.get("InvocationID", ""),
        cgroup=properties.get("ControlGroup", ""),
        started_monotonic=int(properties.get("ExecMainStartTimestampMonotonic", 0)) / 1_000_000,
        exit_status=int(properties.get("ExecMainStatus", 0)),
    )


def _process(pid, proc_root, exclude_supervisor=False):
    text = _read(proc_root / str(pid) / "stat", 16384)[0].decode()
    command = text[text.index("(") + 1 : text.rindex(")")]
    if command == "nvidia-smi":
        return None, None
    if exclude_supervisor:
        arguments = _read(proc_root / str(pid) / "cmdline", 4096)[0].split(b"\0")
        for argument in arguments:
            if argument == b"--":
                break
            if argument.rsplit(b"/", 1)[-1] in {
                b"gpu_supervisor.py",
                b"mars_titan.training.gpu_supervisor",
            }:
                return None, None
    fields = text[text.rindex(")") + 2 :].split()
    process = dict(
        start_ticks=int(fields[19]),
        cpu_ticks=int(fields[11]) + int(fields[12]),
        io_bytes=None,
    )
    try:
        counters = dict(
            line.split(": ", 1)
            for line in _read(proc_root / str(pid) / "io", 4096)[0].decode().splitlines()
        )
        process["io_bytes"] = sum(
            int(counters[key]) for key in ("rchar", "wchar", "read_bytes", "write_bytes")
        )
    except (FileNotFoundError, ProcessLookupError):
        raise
    except (OSError, ValueError, KeyError) as error:
        return process, f"PID {pid} io: {error}"
    return process, None


def read_resources(
    service,
    *,
    exclude_supervisor=False,
    cgroup_root=Path("/sys/fs/cgroup"),
    proc_root=Path("/proc"),
):
    resources = dict(processes={})
    errors = []
    group = service.get("cgroup", "")
    if not group:
        return resources, errors
    folder = (cgroup_root / group.lstrip("/")).resolve()
    if not folder.is_relative_to(cgroup_root.resolve()):
        raise ValueError("El grupo de procesos sale de la raíz cgroup")
    for name, key in (("cpu.stat", "cpu_usec"), ("memory.current", "memory_bytes")):
        try:
            text = _read(folder / name, 8192)[0].decode()
            resources[key] = (
                int(dict(line.split() for line in text.splitlines())["usage_usec"])
                if name == "cpu.stat"
                else int(text)
            )
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"{name}: {error}")
    try:
        lines = _read(folder / "io.stat", 16384)[0].decode().splitlines()
        counters = [dict(part.split("=", 1) for part in line.split()[1:]) for line in lines]
        resources.update(
            read_bytes=sum(int(item.get("rbytes", 0)) for item in counters),
            write_bytes=sum(int(item.get("wbytes", 0)) for item in counters),
        )
    except (OSError, ValueError) as error:
        errors.append(f"io.stat: {error}")
    try:
        pids = _read(folder / "cgroup.procs", 8192)[0].decode().split()
        if len(pids) > MAX_PROCESSES:
            raise ValueError("El cgroup supera el límite de 128 procesos")
        for value in pids:
            pid = int(value)
            if exclude_supervisor and pid == service["pid"]:
                continue
            try:
                process, io_error = _process(pid, proc_root, exclude_supervisor)
                if process:
                    resources["processes"][str(pid)] = process
                if io_error:
                    errors.append(io_error)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, ValueError, KeyError, IndexError) as error:
                errors.append(f"PID {pid}: {error}")
    except (OSError, ValueError) as error:
        errors.append(f"cgroup: {error}")
    return resources, errors


def collect(
    service_name, *, guard=None, summary=None, preparation=None, activities=(), probe_gpu=False
):
    sample = dict(
        timestamp=time.time(),
        monotonic=time.monotonic(),
        boot_id="",
        service_name=service_name,
        service=None,
        resources={"processes": {}},
        guard=None,
        summary=None,
        preparation=None,
        activities=[],
        gpu_compute_pids=None,
        errors=[],
    )
    try:
        sample["boot_id"] = _read("/proc/sys/kernel/random/boot_id", 128)[0].decode().strip()
        sample["service"] = read_service(service_name)
        resources, errors = read_resources(sample["service"], exclude_supervisor=guard is not None)
        sample["resources"], sample["errors"] = resources, errors
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        sample["errors"].append(str(error))
    for key, paths in (
        ("guard", [guard]),
        ("summary", [summary]),
        ("preparation", [preparation]),
        ("activities", activities),
    ):
        for path in paths:
            if path is None:
                continue
            try:
                record = read_report(path)
                if key == "activities":
                    sample[key].append(record)
                else:
                    sample[key] = record
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as error:
                sample["errors"].append(f"{path}: {error}")
    if probe_gpu:
        try:
            pids = _command(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
            ).split()
            sample["gpu_compute_pids"] = sorted(
                set(map(int, pids)) & set(map(int, sample["resources"]["processes"]))
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            sample["errors"].append(f"Consulta GPU: {error}")
    return sample


def _identity(sample):
    service = sample.get("service") or {}
    return (
        sample.get("boot_id"),
        *(service.get(key) for key in ("name", "invocation", "pid", "started_monotonic")),
    )


def _service_name(sample):
    return sample.get("service_name") or (sample.get("service") or {}).get("name")


def _reports(sample):
    return {
        "summary": sample.get("summary") or {},
        "preparation": sample.get("preparation") or {},
        **{record["path"]: record for record in sample.get("activities", []) if "path" in record},
    }


def _progress(current, previous):
    if current.get("path") != previous.get("path"):
        return False
    return any(
        _number(current.get(key)) and _number(previous.get(key)) and current[key] > previous[key]
        for key in COUNTERS
    ) or (current.get("status") == "completed" and previous.get("status") != "completed")


def activity_since(sample, previous):
    result = dict(
        cpu_ticks=0, io_bytes=0, semantic_progress=False, baseline_reset=True, comparable=False
    )
    if not previous or _identity(sample) != _identity(previous["sample"]):
        return result
    before = previous["sample"]
    result["baseline_reset"] = False
    processes = sample["resources"]["processes"]
    older = before["resources"]["processes"]
    for pid, process in processes.items():
        old = older.get(pid)
        if old is None or old["start_ticks"] != process["start_ticks"]:
            result["baseline_reset"] = True
            continue
        for counter in ("cpu_ticks", "io_bytes"):
            if not _number(process.get(counter)) or not _number(old.get(counter)):
                continue
            result["comparable"] = True
            delta = process[counter] - old[counter]
            if delta < 0:
                result["baseline_reset"] = True
            else:
                result[counter] += delta
    older_reports = _reports(before)
    result["semantic_progress"] = any(
        _progress(current, older_reports.get(key, {})) for key, current in _reports(sample).items()
    )
    return result


def _current_report(record, sample):
    service = sample["service"]
    started = sample["timestamp"] - (sample["monotonic"] - service["started_monotonic"])
    return bool(record) and record.get("mtime", 0) >= started - 1


def _guard(sample, *, terminal=False):
    record = sample.get("guard") or {}
    started = sample["timestamp"] - (sample["monotonic"] - sample["service"]["started_monotonic"])
    observed = record.get("observed_at", 0)
    age = sample["timestamp"] - observed
    if observed < started - 1 or age < 0 or (age > 90 and not terminal):
        return {}
    return record


def _phase(sample, previous, activity):
    service = sample.get("service")
    if service is None:
        return "observation_failed"
    if service["active"] == "failed":
        return "service_failed"
    summary = sample.get("summary") or {}
    if service["active"] == "inactive":
        if (
            summary.get("status") == "completed"
            or _guard(sample, terminal=True).get("status") == "completed"
        ):
            return "completed"
        return "stopped" if previous and previous.get("has_run") else "not_started"
    guard = _guard(sample)
    if guard.get("status") in {"failed", "blocked"}:
        return "supervisor_failed"
    if guard.get("status") == "waiting":
        return "waiting_gpu"
    if service["active"] in {"activating", "deactivating"} or guard.get("status") in {
        "pausing",
        "stopping",
    }:
        return "transitioning"
    preparation = sample.get("preparation") or {}
    if preparation.get("status") == "running" and _current_report(preparation, sample):
        return "preparing_cpu"
    for record in sample.get("activities", []):
        if (
            record.get("status") == "running"
            and record.get("training")
            and _current_report(record, sample)
        ):
            return "training"
    if (
        previous
        and _progress(summary, previous["sample"].get("summary") or {})
        and activity["semantic_progress"]
    ):
        return "training" if summary.get("status") == "running" else "running_unknown"
    return "running_unknown"


def assess(sample, previous=None, *, idle_seconds=900):
    """Separar fase, actividad observable y avisos sin interpretar GPU ociosa."""
    if previous:
        before = previous["sample"]
        names = (_service_name(before), _service_name(sample))
        same_service = None in names or names[0] == names[1]
        if before.get("boot_id") != sample.get("boot_id") or not same_service:
            previous = None
    activity = activity_since(sample, previous)
    phase = _phase(sample, previous, activity)
    now = sample["monotonic"]
    last_activity = previous.get("last_activity_monotonic", now) if previous else now
    moved = activity["cpu_ticks"] > 0 or activity["io_bytes"] > 0 or activity["semantic_progress"]
    if activity["baseline_reset"] or moved or phase in {"waiting_gpu", "transitioning"}:
        last_activity = now
    health, alert = "ok", None
    if phase in {"service_failed", "supervisor_failed", "observation_failed", "stopped"}:
        health, alert = "failed" if phase != "stopped" else "stopped", phase
    elif phase == "waiting_gpu":
        health = "waiting"
    elif phase in {"preparing_cpu", "training", "running_unknown"}:
        if previous and not activity["comparable"] and not moved and not activity["baseline_reset"]:
            health, last_activity = "unknown", now
        elif not moved and now - last_activity >= idle_seconds:
            health, alert = "possible_inactivity", "possible_inactivity"
    service = sample.get("service") or {}
    alert_key = (
        f"{alert}:{_service_name(sample)}:{service.get('invocation')}:{service.get('result')}"
        if alert
        else None
    )
    return dict(
        schema_version=1,
        observed_at=datetime.fromtimestamp(sample["timestamp"], UTC).isoformat(),
        phase=phase,
        health=health,
        sample=sample,
        activity=activity,
        last_activity_monotonic=last_activity,
        has_run=bool(
            service.get("active") in {"active", "reloading", "deactivating"}
            or (previous and previous.get("has_run"))
        ),
        alert_key=alert_key,
        notify=bool(alert_key and (not previous or previous.get("alert_key") != alert_key)),
        message=MESSAGES.get(alert),
    )


def atomic_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=".campaign-health-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output, ensure_ascii=False, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def check(args):
    args.state.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        args.state.with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous, previous_error = None, None
        try:
            previous = _json(args.state)[0]
            if (
                previous.get("schema_version") != 1
                or not isinstance(previous.get("sample"), dict)
                or not isinstance(previous.get("phase"), str)
                or not isinstance(previous.get("health"), str)
                or not _number(previous.get("last_activity_monotonic"))
            ):
                raise ValueError("El estado anterior no tiene el formato esperado")
        except FileNotFoundError:
            pass
        except (ValueError, OSError) as error:
            previous, previous_error = None, str(error)
        sample = collect(
            args.service,
            guard=args.guard,
            summary=args.summary,
            preparation=args.preparation,
            activities=args.activity,
            probe_gpu=args.probe_gpu,
        )
        if previous_error:
            sample["errors"].append(f"Estado anterior: {previous_error}")
        try:
            state = assess(sample, previous, idle_seconds=args.idle_seconds)
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            if previous is None:
                raise
            sample["errors"].append(f"Estado anterior incompatible: {error}")
            previous = None
            state = assess(sample, idle_seconds=args.idle_seconds)
        atomic_json(args.state, state)
        if state["notify"]:
            print(f"<4>AVISO MARS-TITAN: {state['message']} Estado: {args.state}", flush=True)
            if args.notify:
                try:
                    subprocess.run(
                        ["notify-send", "--urgency=critical", "MARS-TITAN", state["message"]],
                        check=True,
                        timeout=5,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except (OSError, subprocess.SubprocessError) as error:
                    print(f"<4>No se pudo mostrar el aviso de escritorio: {error}", flush=True)
        elif not previous or (state["phase"], state["health"]) != (
            previous["phase"],
            previous["health"],
        ):
            print(
                json.dumps(
                    {key: state[key] for key in ("observed_at", "phase", "health")},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return state
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default="mars-titan-scientific-campaign.service")
    parser.add_argument("--state", type=Path, required=True)
    for key in ("guard", "summary", "preparation"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--activity", type=Path, action="append", default=[])
    parser.add_argument("--idle-seconds", type=int, default=900)
    parser.add_argument("--probe-gpu", action="store_true")
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    if (
        not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", args.service)
        or not 300 <= args.idle_seconds <= 86400
        or len(args.activity) > 8
    ):
        parser.error("El servicio o los límites de supervisión no son válidos")
    inputs = [args.guard, args.summary, args.preparation, *args.activity]
    if any(path and path.resolve() == args.state.resolve() for path in inputs):
        parser.error("El estado del monitor no puede sustituir una entrada")
    try:
        check(args)
    except BlockingIOError:
        return 0
    except (OSError, ValueError) as error:
        print(f"<3>No se pudo guardar la comprobación de salud: {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
