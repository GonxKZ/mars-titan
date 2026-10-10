"""Admitir PPO o KLPO nativo con una única carga GPU o un diagnóstico CPU explícito.

El mismo lanzador sirve a mars-titan-ppo y a mars-titan-klpo (``--binary``). Los esquemas
adaptativos 2 a 4 y la configuración KLPO admiten catálogos grandes y auditoría separada.
El esquema 4 y KLPO usan cintas reconstruidas, así que también su evaluación se detiene con
la protección local del aprendizaje antes de lanzar el binario.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.gpu_supervisor import memory_reason, read_gpu
from mars_titan.training.learning_hold import require_learning_allowed

MIB = 1024**2
RESERVE_MIB = 1024
MINIMUM_BUDGET_MIB = 256
MAXIMUM_BUDGET_MIB = 6144
PAUSE_TIMEOUT = 60
KILL_TIMEOUT = 5
GPU_PROBE_TIMEOUT = 5
LOCK_NAME = "mars-titan-scientific-gpu.lock"
DEFAULT_BINARY = (
    Path(__file__).resolve().parents[1] / "build/native/native-ppo-release/mars-titan-ppo"
)
KLPO_KIND = "native_klpo_terminal"
RECONSTRUCTED_SCHEMA = 4


def catalog_config(document):
    """Configuración con catálogo de fuentes y auditoría separada."""
    return document.get("schema_version") in (2, 3, RECONSTRUCTED_SCHEMA) or (
        document.get("kind") == KLPO_KIND
    )


def reconstructed_config(document):
    """Configuración que ajusta o evalúa sobre cintas reconstruidas del histórico."""
    return (
        document.get("schema_version") == RECONSTRUCTED_SCHEMA or document.get("kind") == KLPO_KIND
    )


class GpuWaiting(RuntimeError):
    """La GPU no admite todavía el proceso solicitado."""


def boot_id():
    path = Path("/proc/sys/kernel/random/boot_id")
    value = path.read_text().strip()
    if not value or len(value) > 128:
        raise ValueError("No se puede identificar el arranque del sistema")
    return value


def command_identity(command):
    paths = {
        "--config",
        "--output",
        "--train-tape",
        "--validation-tape",
        "--audit-run",
        "--audit-tape",
    }
    ignored = {
        "--gpu-lease-fd",
        "--vram-budget-bytes",
        "--vram-total-bytes",
        "--stop-after",
        "--parent-pid",
    }
    selected = []
    index = 0
    while index < len(command):
        if command[index] in ignored:
            index += 2
        elif command[index] == "--resume":
            index += 1
        elif command[index] in paths:
            selected.extend((command[index], str(Path(command[index + 1]).resolve())))
            index += 2
        else:
            selected.append(command[index])
            index += 1
    config = Path(command[command.index("--config") + 1])
    identity = dict(
        command=selected, binary_sha256=sha256(Path(command[0])), config_sha256=sha256(config)
    )
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@contextmanager
def watch_lock(path):
    if path is None:
        yield
        return
    safe_destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
        ):
            raise ValueError("El bloqueo de vigilancia no es un archivo privado regular")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise GpuWaiting("Otra instancia mantiene la vigilancia de este caso") from error
        yield
    finally:
        os.close(descriptor)


class ActiveWatch:
    """Contar tiempo observado y reservas por interrupciones sin sumar tiempo apagado."""

    def __init__(self, command, path, poll_seconds):
        self.path = path
        self.last_tick = None
        self.reserve = PAUSE_TIMEOUT + KILL_TIMEOUT + GPU_PROBE_TIMEOUT + max(poll_seconds, 0.1)
        self.current_boot = boot_id()
        identity = command_identity(command)
        self.record = dict(
            schema_version=2,
            identity_sha256=identity,
            status="ready",
            starts=0,
            pauses=0,
            active_seconds=0.0,
            budget_seconds=0.0,
            uncertain_attempts=0,
            reserved_seconds=0.0,
            unobserved_reserve_seconds=self.reserve,
            child_active=False,
            budget_complete=True,
            boot_id=self.current_boot,
            owner_pid=os.getpid(),
            child_pid=None,
            parent_death_signal="SIGKILL" if "--parent-pid" in command else None,
            observed_at=datetime.now(UTC).isoformat(),
            started_monotonic=None,
            last_observed_monotonic=None,
            reason=None,
            returncode=None,
        )
        if path is not None and path.exists():
            previous, _ = read_manifest(path, 64 * 1024)
            self._validate(previous, identity)
            self.record = previous
            if previous["child_active"]:
                if previous.get("parent_death_signal") != "SIGKILL":
                    self.record.update(
                        status="blocked",
                        budget_complete=False,
                        reason="unbounded_unobserved_child",
                        returncode=1,
                    )
                    self.save()
                    raise RuntimeError(
                        "Hay un hijo anterior sin observación ni vínculo acreditado con su padre"
                    )
                self.record["budget_seconds"] += previous["unobserved_reserve_seconds"]
                self.record["reserved_seconds"] += previous["unobserved_reserve_seconds"]
                self.record["uncertain_attempts"] += 1
                self.record.update(child_active=False, status="interrupted", child_pid=None)
            if not self.record["budget_complete"]:
                raise RuntimeError("El presupuesto de una ejecución anterior necesita revisión")
            self.record.update(
                boot_id=self.current_boot,
                owner_pid=os.getpid(),
                unobserved_reserve_seconds=self.reserve,
                parent_death_signal="SIGKILL" if "--parent-pid" in command else None,
            )

    @staticmethod
    def _validate(record, identity):
        if record.get("schema_version") != 2 or record.get("identity_sha256") != identity:
            raise ValueError("La identidad de vigilancia no corresponde a este comando")
        for key in (
            "active_seconds",
            "budget_seconds",
            "reserved_seconds",
            "unobserved_reserve_seconds",
        ):
            value = record.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("El contador de vigilancia no es finito y no negativo")
        for key in ("starts", "pauses", "uncertain_attempts"):
            if type(record.get(key)) is not int or not 0 <= record[key] <= 1_000_000:
                raise ValueError("Los intentos de vigilancia no están acotados")
        if (
            type(record.get("child_active")) is not bool
            or type(record.get("budget_complete")) is not bool
            or not isinstance(record.get("boot_id"), str)
            or not math.isclose(
                record["budget_seconds"],
                record["active_seconds"] + record["reserved_seconds"],
                abs_tol=1e-6,
            )
            or record["unobserved_reserve_seconds"] > 180
        ):
            raise ValueError("La vigilancia no conserva el contrato de contabilidad")

    def save(self):
        self.record["observed_at"] = datetime.now(UTC).isoformat()
        if self.path is not None:
            atomic_json(self.path, self.record)

    def begin(self):
        now = time.monotonic()
        self.last_tick = now
        self.record["starts"] += 1
        self.record.update(
            status="running",
            child_active=True,
            owner_pid=os.getpid(),
            started_monotonic=now,
            last_observed_monotonic=now,
            child_pid=None,
            reason=None,
            returncode=None,
        )
        self.save()

    def observe(self, status="running", reason=None, snapshot=None):
        now = time.monotonic()
        if self.last_tick is not None:
            elapsed = max(0.0, now - self.last_tick)
            self.record["active_seconds"] += elapsed
            self.record["budget_seconds"] += elapsed
            self.last_tick = now
        self.record.update(
            status=status,
            reason=reason,
            last_observed_monotonic=now,
            free_mib=snapshot.free_mib if snapshot else None,
        )
        self.save()

    def finish(self, code, reason=None, snapshot=None):
        phase = {0: "completed", 2: "paused", 3: "waiting", 4: "budget_exhausted"}.get(
            code, "failed"
        )
        self.observe(phase, reason, snapshot)
        self.record.update(
            child_active=False, child_pid=None, returncode=code, recoverable_checkpoint=code == 2
        )
        self.record["pauses"] += int(code == 2)
        self.last_tick = None
        self.save()


@contextmanager
def gpu_admission():
    """Mantener el bloqueo compartido con el hijo hasta que termine."""
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/mars-titan-{os.getuid()}"))
    if not runtime.is_absolute():
        raise ValueError("El directorio de ejecución debe ser absoluto")
    runtime.mkdir(mode=0o700, exist_ok=True)
    directory = runtime.lstat()
    if (
        not stat.S_ISDIR(directory.st_mode)
        or directory.st_uid != os.getuid()
        or directory.st_mode & 0o022
    ):
        raise ValueError("El directorio de ejecución no pertenece exclusivamente al usuario")
    path = runtime / LOCK_NAME
    descriptor = os.open(
        path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, 0o600
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
        ):
            raise ValueError("El bloqueo GPU debe ser un archivo regular privado del usuario")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise GpuWaiting("Otra carga científica mantiene el bloqueo GPU") from error
        # Las versiones anteriores creaban el bloqueo con umask. Solo su dueño,
        # dentro del directorio protegido y tras adquirirlo, normaliza los permisos.
        os.fchmod(descriptor, 0o600)
        current = path.lstat()
        if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise RuntimeError("El archivo de bloqueo GPU cambió durante la admisión")
        snapshot = read_gpu()
        reason = memory_reason(snapshot, RESERVE_MIB + MINIMUM_BUDGET_MIB)
        if reason == "memoria_insuficiente":
            raise GpuWaiting(
                "La GPU no permite reservar 1 GiB y asignar al menos 256 MiB al proceso"
            )
        if reason or snapshot.compute_pids:
            raise GpuWaiting("La GPU está ocupada por otro proceso de cálculo")
        budget = (
            min(MAXIMUM_BUDGET_MIB, snapshot.free_mib - RESERVE_MIB, snapshot.total_mib * 3 // 4)
            * MIB
        )
        if budget < MINIMUM_BUDGET_MIB * MIB:
            raise GpuWaiting("El presupuesto admitido no alcanza 256 MiB")
        yield descriptor, budget, snapshot.total_mib * MIB
    finally:
        os.close(descriptor)


def signal_child(child, signum):
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass


def stop_child(child, signum):
    """Dar tiempo al checkpoint y recoger únicamente el proceso que se ha creado."""
    signal_child(child, signum)
    try:
        return child.wait(timeout=PAUSE_TIMEOUT)
    except subprocess.TimeoutExpired:
        print("Error: El proceso nativo no confirmó la pausa dentro del plazo", file=sys.stderr)
        signal_child(child, signal.SIGKILL)
        child.wait(timeout=KILL_TIMEOUT)
        return 75


def run_child(
    command, descriptor=None, *, poll_seconds=5, state=None, active_seconds_limit=None, watch=None
):
    requested = None
    reason = None
    snapshot = None
    watch = watch or ActiveWatch(command, state, poll_seconds)
    if active_seconds_limit is not None and active_seconds_limit <= watch.reserve:
        watch.finish(4, "active_budget_exhausted")
        return 4

    def request(signum, _frame):
        nonlocal requested
        if requested is None:
            requested = signum

    handlers = {sig: signal.signal(sig, request) for sig in (signal.SIGINT, signal.SIGTERM)}
    child = None
    try:
        watch.begin()
        child = subprocess.Popen(
            command,
            pass_fds=() if descriptor is None else (descriptor,),
            start_new_session=True,
        )
        watch.record["child_pid"] = child.pid
        watch.save()
        started = watch.record["started_monotonic"]
        next_probe = time.monotonic() + poll_seconds
        while True:
            if (
                active_seconds_limit is not None
                and time.monotonic() - started >= active_seconds_limit - watch.reserve
            ):
                requested = signal.SIGTERM
                reason = "active_budget_exhausted"
            if requested is not None:
                watch.observe("pausing", reason, snapshot)
                code = stop_child(child, requested)
                break
            try:
                code = child.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pass
            if descriptor is not None and time.monotonic() >= next_probe:
                try:
                    snapshot = read_gpu()
                    reason = memory_reason(snapshot, RESERVE_MIB, child.pid)
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                    reason = "lectura_gpu_fallida"
                if reason:
                    watch.observe("pausing", reason, snapshot)
                    code = stop_child(child, signal.SIGTERM)
                    break
                watch.observe(snapshot=snapshot)
                next_probe = time.monotonic() + poll_seconds
            elif descriptor is None and time.monotonic() >= next_probe:
                watch.observe()
                next_probe = time.monotonic() + poll_seconds
        code = code if code >= 0 else 128 - code
        watch.finish(code, reason, snapshot)
        return code
    except BaseException:
        if child is not None and child.poll() is None:
            stop_child(child, signal.SIGTERM)
        watch.finish(1, "launcher_failed")
        raise
    finally:
        try:
            if child is not None and child.poll() is None:
                stop_child(child, signal.SIGTERM)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-tape", type=Path, action="append", default=[])
    parser.add_argument("--validation-tape", type=Path, action="append", default=[])
    parser.add_argument("--audit-run", type=Path)
    parser.add_argument("--audit-tape", type=Path, action="append", default=[])
    parser.add_argument(
        "--evaluation-cost",
        type=float,
        action="append",
        default=[],
        help="Coste en pb de la evaluación separada. Repetible y solo con --audit-run",
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--watch-state", type=Path)
    parser.add_argument("--active-seconds-limit", type=float)
    parser.add_argument(
        "--diagnostic", action="store_true", help="Diagnóstico CPU de hasta 32 transiciones"
    )
    args = parser.parse_args(argv)
    try:
        document, _ = read_manifest(args.config, 4 * MIB)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    maximum_sources = 512 if catalog_config(document) else 12
    if max(len(args.train_tape), len(args.validation_tape), len(args.audit_tape)) > maximum_sources:
        parser.error(f"Se admiten como máximo {maximum_sources} fuentes por partición")
    if args.audit_run is not None:
        if (
            not catalog_config(document)
            or not args.audit_tape
            or args.train_tape
            or args.validation_tape
        ):
            parser.error(
                "La auditoría necesita esquema 2, 3 o 4 o KLPO, --audit-run y --audit-tape, "
                "sin fuentes de entrenamiento"
            )
    elif args.audit_tape or not args.train_tape or not args.validation_tape:
        parser.error("El entrenamiento necesita --train-tape y --validation-tape")
    if args.evaluation_cost and (
        args.audit_run is None
        or len(args.evaluation_cost) > 16
        or any(not math.isfinite(cost) or not 0 <= cost <= 1000 for cost in args.evaluation_cost)
        or any(a >= b for a, b in zip(args.evaluation_cost, args.evaluation_cost[1:], strict=False))
    ):
        parser.error(
            "Los costes de evaluación solo acompañan a --audit-run y deben ser crecientes, "
            "finitos y estar entre 0 y 1000 pb"
        )
    if args.stop_after is not None and not 0 <= args.stop_after <= 2**63 - 1:
        parser.error("La parada debe ser un número entero no negativo de 64 bits")
    if not math.isfinite(args.poll_seconds) or not 0.01 <= args.poll_seconds <= 60:
        parser.error("La consulta de recursos debe estar entre 0,01 y 60 segundos")
    if args.active_seconds_limit is not None and (
        not math.isfinite(args.active_seconds_limit)
        or not 0 <= args.active_seconds_limit <= 168 * 3600
    ):
        parser.error("El límite activo debe estar entre cero y 168 horas")
    if args.watch_state is not None:
        watched = args.watch_state.resolve()
        if watched == args.config.resolve() or any(
            watched.is_relative_to(path.resolve())
            for path in [
                args.output,
                *args.train_tape,
                *args.validation_tape,
                *args.audit_tape,
                *([args.audit_run] if args.audit_run else []),
            ]
        ):
            parser.error("El estado de vigilancia debe quedar fuera de las fuentes y de la salida")
    if args.audit_run is None:
        algorithm = "KLPO" if document.get("kind") == KLPO_KIND else "PPO"
        require_learning_allowed(f"el entrenamiento {algorithm} nativo")
    elif reconstructed_config(document):
        require_learning_allowed("la evaluación nativa sobre cintas reconstruidas")
    try:
        binary = args.binary.resolve(strict=True)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("El binario nativo debe ser un archivo ejecutable")
        command = [str(binary), "--config", str(args.config), "--output", str(args.output)]
        for option, sources in (
            ("--train-tape", args.train_tape),
            ("--validation-tape", args.validation_tape),
            ("--audit-tape", args.audit_tape),
        ):
            for source in sources:
                command.extend((option, str(source)))
        if args.audit_run is not None:
            command.extend(("--audit-run", str(args.audit_run)))
        for cost in args.evaluation_cost:
            command.extend(("--evaluation-cost", repr(cost)))
        if args.resume:
            command.append("--resume")
        if args.stop_after is not None:
            command.extend(("--stop-after", str(args.stop_after)))
        command.extend(("--device", "cpu" if args.diagnostic else "cuda:0"))
        command.extend(("--parent-pid", str(os.getpid())))
        if args.diagnostic:
            command.append("--diagnostic")
        with watch_lock(args.watch_state):
            watch = ActiveWatch(command, args.watch_state, args.poll_seconds)
            if args.active_seconds_limit is not None and args.active_seconds_limit <= watch.reserve:
                watch.finish(4, "active_budget_exhausted")
                return 4
            if args.diagnostic:
                print("Diagnóstico explícito en CPU, limitado a 32 transiciones", file=sys.stderr)
                return run_child(
                    command,
                    watch=watch,
                    poll_seconds=args.poll_seconds,
                    active_seconds_limit=args.active_seconds_limit,
                )
            try:
                with gpu_admission() as (descriptor, budget, total):
                    return run_child(
                        [
                            *command,
                            "--gpu-lease-fd",
                            str(descriptor),
                            "--vram-budget-bytes",
                            str(budget),
                            "--vram-total-bytes",
                            str(total),
                        ],
                        descriptor,
                        watch=watch,
                        poll_seconds=args.poll_seconds,
                        active_seconds_limit=args.active_seconds_limit,
                    )
            except GpuWaiting:
                watch.finish(3, "gpu_not_admitted")
                raise
    except GpuWaiting as error:
        print(f"Espera: {error}", file=sys.stderr)
        return 3
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
