"""Admitir PPO nativo con una única carga GPU o un diagnóstico CPU explícito."""

import argparse
import fcntl
import os
import signal
import stat
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from mars_titan.training.gpu_supervisor import memory_reason, read_gpu

MIB = 1024**2
RESERVE_MIB = 1024
MINIMUM_BUDGET_MIB = 256
MAXIMUM_BUDGET_MIB = 6144
PAUSE_TIMEOUT = 60
KILL_TIMEOUT = 5
LOCK_NAME = "mars-titan-scientific-gpu.lock"
DEFAULT_BINARY = (
    Path(__file__).resolve().parents[1] / "build/native/native-ppo-release/mars-titan-ppo"
)


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
            or metadata.st_mode & 0o022
        ):
            raise ValueError("El bloqueo GPU debe ser un archivo regular privado del usuario")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Otra carga científica mantiene el bloqueo GPU") from error
        current = path.lstat()
        if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise RuntimeError("El archivo de bloqueo GPU cambió durante la admisión")
        snapshot = read_gpu()
        reason = memory_reason(snapshot, RESERVE_MIB + MINIMUM_BUDGET_MIB)
        if reason == "memoria_insuficiente":
            raise RuntimeError(
                "La GPU no permite reservar 1 GiB y asignar al menos 256 MiB al proceso"
            )
        if reason or snapshot.compute_pids:
            raise RuntimeError("La GPU está ocupada por otro proceso de cálculo")
        budget = (
            min(MAXIMUM_BUDGET_MIB, snapshot.free_mib - RESERVE_MIB, snapshot.total_mib * 3 // 4)
            * MIB
        )
        if budget < MINIMUM_BUDGET_MIB * MIB:
            raise RuntimeError("El presupuesto admitido no alcanza 256 MiB")
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


def run_child(command, descriptor=None):
    requested = None

    def request(signum, _frame):
        nonlocal requested
        if requested is None:
            requested = signum

    handlers = {sig: signal.signal(sig, request) for sig in (signal.SIGINT, signal.SIGTERM)}
    child = None
    try:
        child = subprocess.Popen(
            command,
            pass_fds=() if descriptor is None else (descriptor,),
            start_new_session=True,
        )
        while True:
            if requested is not None:
                code = stop_child(child, requested)
                break
            try:
                code = child.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                continue
        return code if code >= 0 else 128 - code
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
    parser.add_argument("--train-tape", type=Path, action="append", required=True)
    parser.add_argument("--validation-tape", type=Path, action="append", required=True)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument(
        "--diagnostic", action="store_true", help="Diagnóstico CPU de hasta 32 transiciones"
    )
    args = parser.parse_args(argv)
    if len(args.train_tape) > 12 or len(args.validation_tape) > 12:
        parser.error("Se admiten como máximo 12 fuentes por partición")
    if args.stop_after is not None and not 0 <= args.stop_after <= 2**63 - 1:
        parser.error("La parada debe ser un número entero no negativo de 64 bits")
    try:
        binary = args.binary.resolve(strict=True)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("El binario nativo debe ser un archivo ejecutable")
        command = [str(binary), "--config", str(args.config), "--output", str(args.output)]
        for option, sources in (
            ("--train-tape", args.train_tape),
            ("--validation-tape", args.validation_tape),
        ):
            for source in sources:
                command.extend((option, str(source)))
        if args.resume:
            command.append("--resume")
        if args.stop_after is not None:
            command.extend(("--stop-after", str(args.stop_after)))
        if args.diagnostic:
            print("Diagnóstico explícito en CPU, limitado a 32 transiciones", file=sys.stderr)
            return run_child([*command, "--device", "cpu", "--diagnostic"])
        with gpu_admission() as (descriptor, budget, total):
            return run_child(
                [
                    *command,
                    "--device",
                    "cuda:0",
                    "--gpu-lease-fd",
                    str(descriptor),
                    "--vram-budget-bytes",
                    str(budget),
                    "--vram-total-bytes",
                    str(total),
                ],
                descriptor,
            )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
