"""Exporta una instantánea pública del observatorio desde estados locales."""

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import stat
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

MODELS = [
    {"id": "B0", "name": "Residual cero y media histórica", "kind": "baseline"},
    {"id": "B1", "name": "Ridge con variables históricas", "kind": "baseline"},
    {"id": "B2", "name": "Boosting tabular compacto", "kind": "baseline"},
    {"id": "B3", "name": "GRU compacta", "kind": "baseline"},
    {"id": "B6", "name": "Memoria asociativa con regla delta", "kind": "baseline"},
    {"id": "M0", "name": "Codificador sin memoria", "kind": "ablation"},
    {"id": "M1", "name": "Memoria global y escritura uniforme", "kind": "ablation"},
    {"id": "M2", "name": "Escritura por error maduro", "kind": "ablation"},
    {"id": "M3", "name": "Sorpresa, régimen e incertidumbre", "kind": "candidate"},
]
METRICS = (
    "mae",
    "mse",
    "rank_ic",
    "coverage_80",
    "coverage_95",
    "loss",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "vram_peak_mib",
    "ram_peak_mib",
    "elapsed_seconds",
    "samples_per_second",
)
STATUSES = {"queued", "running", "paused", "completed", "failed", "cancelled"}
PHASES = {"prepare", "train", "validation", "calibration", "test", "evaluation"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
MAX_HISTORY = 500
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def identifier(value, *, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("Identificador público inválido.")
    return value


def integer(value):
    if value is not None and (type(value) is not int or not 0 <= value <= 2**53 - 1):
        raise ValueError("Contador inválido.")
    return value


def number(value, name):
    if value is None:
        return None
    if type(value) not in (int, float) or not -1e308 <= value <= 1e308:
        raise ValueError("Métrica no numérica o no finita.")
    if not math.isfinite(value):
        raise ValueError("Métrica no finita.")
    if name == "rank_ic" and not -1 <= value <= 1:
        raise ValueError("Rank IC fuera de rango.")
    if name.startswith("coverage_") and not 0 <= value <= 1:
        raise ValueError("Cobertura fuera de rango.")
    if name not in {"rank_ic", "loss"} and value < 0:
        raise ValueError("Métrica negativa no permitida.")
    return value


def timestamp(value, now):
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError("Fecha inválida.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("Fecha inválida.") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("La fecha necesita zona horaria.")
    parsed = parsed.astimezone(UTC)
    if parsed > now:
        raise ValueError("La fecha está en el futuro respecto a la exportación.")
    return parsed.isoformat().replace("+00:00", "Z")


def optional_choice(value, choices):
    if value is not None and (not isinstance(value, str) or value not in choices):
        raise ValueError("Estado o fase no admitidos.")
    return value


def object_value(value):
    if not isinstance(value, dict):
        raise ValueError("Se esperaba un objeto JSON.")
    return value


def check_order(earlier, later):
    if earlier and later and datetime.fromisoformat(earlier) > datetime.fromisoformat(later):
        raise ValueError("Fechas fuera de orden.")


def comparison_group(raw, phase, fold):
    contract = raw.get("comparison_contract")
    if contract is None or phase is None or fold is None:
        return None
    object_value(contract)
    identity = {
        key: identifier(contract.get(key))
        for key in ("target", "universe", "splits", "evaluation_regime")
    }
    identity.update(phase=phase, fold=fold)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return "cg-" + hashlib.sha256(encoded).hexdigest()


def public_history(raw, run, now, deadline):
    if not isinstance(raw, list):
        raise ValueError("Historial inválido.")
    result = []
    previous_step = -1
    previous_time = None
    for point in raw:
        check_deadline(deadline)
        object_value(point)
        if point.get("attempt_id", run["attempt_id"]) != run["attempt_id"]:
            raise ValueError("El historial mezcla intentos.")
        if point.get("phase", run["phase"]) != run["phase"]:
            raise ValueError("El historial mezcla fases.")
        step = integer(point.get("step"))
        recorded_at = timestamp(point.get("recorded_at"), now)
        if step is None or recorded_at is None or step <= previous_step:
            raise ValueError("Historial sin fecha o pasos estrictamente crecientes.")
        if run["completed_steps"] is not None and step > run["completed_steps"]:
            raise ValueError("Historial posterior al progreso confirmado.")
        check_order(previous_time, recorded_at)
        check_order(run["started_at"], recorded_at)
        check_order(recorded_at, run["updated_at"])
        previous_step, previous_time = step, recorded_at
        result.append(
            {
                "step": step,
                "recorded_at": recorded_at,
                "loss": number(point.get("loss"), "loss"),
                "mae": number(point.get("mae"), "mae"),
            }
        )
    return result[-MAX_HISTORY:]


def public_checkpoint(raw, run, now):
    """Valida el punto guardado y su coherencia con el progreso confirmado."""
    checkpoint = object_value(raw)
    resumable = checkpoint.get("resumable")
    if resumable is not None and type(resumable) is not bool:
        raise ValueError("Indicador de recuperación inválido.")
    result = {
        "step": integer(checkpoint.get("step")),
        "saved_at": timestamp(checkpoint.get("saved_at"), now),
        "resumable": resumable,
    }
    saved_step = result["step"]
    if resumable and (saved_step is None or result["saved_at"] is None):
        raise ValueError("La recuperación necesita un paso y una fecha de checkpoint observados.")
    if saved_step is not None and run["completed_steps"] is not None:
        if saved_step > run["completed_steps"]:
            raise ValueError("Checkpoint posterior al progreso confirmado.")
    check_order(result["saved_at"], run["updated_at"])
    return result


def public_run(raw, directory, now, release_test, deadline):
    object_value(raw)
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("Versión de esquema no admitida.")
    run = {key: identifier(raw.get(key)) for key in ("run_id", "attempt_id", "model_id")}
    if run["run_id"] != directory:
        raise ValueError("El identificador no coincide con su directorio.")
    if run["model_id"] not in {model["id"] for model in MODELS}:
        raise ValueError("Modelo fuera del catálogo.")
    run["variant_id"] = identifier(raw.get("variant_id"), optional=True)
    run["status"] = optional_choice(raw.get("status"), STATUSES)
    run["phase"] = optional_choice(raw.get("phase"), PHASES)
    for key in ("heartbeat_at", "started_at", "updated_at"):
        run[key] = timestamp(raw.get(key), now)
    check_order(run["started_at"], run["updated_at"])
    check_order(run["started_at"], run["heartbeat_at"])
    for key in ("completed_steps", "total_steps", "epoch", "max_epochs", "seed"):
        run[key] = integer(raw.get(key))
    for current, total in (("completed_steps", "total_steps"), ("epoch", "max_epochs")):
        if run[current] is not None and run[total] is not None and run[current] > run[total]:
            raise ValueError("Progreso superior al total declarado.")
    fold = raw.get("fold")
    run["fold"] = integer(fold) if type(fold) is int else identifier(fold, optional=True)
    run["comparison_group"] = comparison_group(raw, run["phase"], run["fold"])
    metrics = object_value(raw.get("metrics", {}))
    run["metrics"] = {key: number(metrics.get(key), key) for key in METRICS}
    run["history"] = public_history(raw.get("history", []), run, now, deadline)
    run["checkpoint"] = public_checkpoint(raw.get("checkpoint", {}), run, now)
    is_test = run["phase"] in {"test", "evaluation"}
    if is_test and release_test and run["status"] not in TERMINAL_STATUSES:
        raise ValueError("Liberar el test exige una ejecución terminada.")
    run["test_released"] = bool(is_test and release_test)
    if run["phase"] is None or (is_test and not release_test):
        run["metrics"] = dict.fromkeys(METRICS)
        run["history"] = []
    return run


def check_deadline(deadline):
    if time.monotonic() > deadline:
        raise ValueError("Se ha agotado el presupuesto de tiempo de exportación.")


def open_directory(path, *, create=False):
    """Abre cada componente sin seguir enlaces, manteniendo descriptores del padre."""
    parts = Path(path).absolute().parts
    descriptor = os.open(parts[0], DIRECTORY_FLAGS)
    try:
        for part in parts[1:]:
            try:
                following = os.open(part, DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, dir_fd=descriptor)
                following = os.open(part, DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Clave JSON duplicada.")
        result[key] = value
    return result


def reject_constant(_value):
    raise ValueError("Constante JSON no finita.")


def read_status(directory_fd, max_file_bytes):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor = os.open("status.json", flags, dir_fd=directory_fd)
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_file_bytes:
            raise ValueError("El estado no es un archivo regular acotado.")
        body = handle.read(max_file_bytes + 1)
    if len(body) > max_file_bytes:
        raise ValueError("El estado supera el límite de bytes.")
    try:
        raw = json.loads(body, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("JSON inválido, duplicado o excesivamente profundo.") from None
    pending = [(raw, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 16:
            raise ValueError("El JSON supera el límite de profundidad.")
        children = (
            item.values() if isinstance(item, dict) else item if isinstance(item, list) else ()
        )
        pending.extend((child, depth + 1) for child in children)
    return raw


def collect_runs(input_dir, now, release_test, max_runs, max_file_bytes, deadline):
    try:
        root_fd = open_directory(input_dir)
    except FileNotFoundError:
        return []
    runs = []
    try:
        with os.scandir(root_fd) as entries:
            for count, entry in enumerate(entries, start=1):
                check_deadline(deadline)
                if count > max_runs * 4:
                    raise ValueError("Demasiadas entradas en el directorio.")
                if entry.is_symlink():
                    raise ValueError("No se admiten enlaces simbólicos.")
                if not entry.is_dir(follow_symlinks=False):
                    continue
                identifier(entry.name)
                run_fd = os.open(entry.name, DIRECTORY_FLAGS, dir_fd=root_fd)
                try:
                    try:
                        raw = read_status(run_fd, max_file_bytes)
                    except FileNotFoundError:
                        continue
                finally:
                    os.close(run_fd)
                if len(runs) >= max_runs:
                    raise ValueError("Demasiadas ejecuciones para una instantánea.")
                runs.append(public_run(raw, entry.name, now, release_test, deadline))
    finally:
        os.close(root_fd)
    return sorted(runs, key=lambda run: (run["run_id"], run["attempt_id"]))


def atomic_write(output, document, deadline, max_output_bytes):
    payload = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if len(payload) > max_output_bytes:
        raise ValueError("La salida supera el límite de bytes.")
    check_deadline(deadline)
    parent_fd = open_directory(output.parent, create=True)
    temporary = ".observatory-" + secrets.token_hex(16) + ".json"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            mode=0o644,
            dir_fd=parent_fd,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        check_deadline(deadline)
        os.replace(temporary, output.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        finally:
            os.close(parent_fd)


def export_snapshot(
    input_dir,
    output,
    *,
    now=None,
    release_test=False,
    max_runs=128,
    max_file_bytes=262144,
    max_output_bytes=8388608,
    timeout_seconds=5.0,
):
    """Valida todo antes de reemplazar la instantánea pública, sin leer los eventos privados."""
    if type(max_runs) is not int or not 1 <= max_runs <= 128:
        raise ValueError("El límite de ejecuciones debe estar entre 1 y 128.")
    if type(max_file_bytes) is not int or not 1 <= max_file_bytes <= 262144:
        raise ValueError("El límite de bytes debe estar entre 1 y 262144.")
    if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= 8388608:
        raise ValueError("El límite de salida debe estar entre 1 y 8388608 bytes.")
    if type(release_test) is not bool:
        raise ValueError("La liberación del test debe ser una decisión booleana explícita.")
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 30:
        raise ValueError("El presupuesto de tiempo debe ser positivo y no superar 30 segundos.")
    deadline = time.monotonic() + timeout_seconds
    now = datetime.now(UTC) if now is None else now
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("La fecha de exportación necesita zona horaria.")
    now = now.astimezone(UTC)
    input_dir = Path(os.path.abspath(input_dir))
    output = Path(os.path.abspath(output))
    if output.is_relative_to(input_dir):
        raise ValueError("La salida no puede estar dentro del directorio de estados privados.")
    try:
        runs = collect_runs(input_dir, now, release_test, max_runs, max_file_bytes, deadline)
    except OSError:
        raise ValueError("No se pueden leer los estados locales sin seguir enlaces.") from None
    notes = []
    if any(
        run["status"] == "running"
        and (
            run["heartbeat_at"] is None
            or (now - datetime.fromisoformat(run["heartbeat_at"])).total_seconds() > 180
        )
        for run in runs
    ):
        notes.append("Hay ejecuciones sin latido reciente. Su actividad no está confirmada.")
    if any(run["phase"] is None for run in runs):
        notes.append("Hay ejecuciones sin fase identificada. Sus métricas permanecen ocultas.")
    document = {
        "schema_version": 1,
        "project": "MARS-TITAN",
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "source_status": "available" if runs else "no_runs_registered",
        "poll_interval_seconds": 60,
        "stale_after_seconds": 180,
        "models": MODELS,
        "runs": runs,
        "notes": notes,
    }
    atomic_write(output, document, deadline, max_output_bytes)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--release-test", action="store_true", help="Liberar test de runs terminados."
    )
    parser.add_argument("--max-runs", type=int, default=128)
    parser.add_argument("--max-file-bytes", type=int, default=262144)
    parser.add_argument("--max-output-bytes", type=int, default=8388608)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    try:
        export_snapshot(
            args.input_dir,
            args.output,
            release_test=args.release_test,
            max_runs=args.max_runs,
            max_file_bytes=args.max_file_bytes,
            max_output_bytes=args.max_output_bytes,
            timeout_seconds=args.timeout_seconds,
        )
    except ValueError as error:
        print(f"Exportación cancelada: {error}", file=sys.stderr)
        return 1
    except OSError:
        print("Exportación cancelada por un error de acceso o escritura.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
