"""Recolección incremental, saneamiento e historial paginado de campañas."""

import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.storage import atomic_json

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
STATUS = {
    "pending": "queued",
    "interrupted": "paused",
    "running": "running",
    "completed": "completed",
    "failed": "failed",
    "paused": "paused",
    "cancelled": "cancelled",
    "blocked": "blocked",
    "queued": "queued",
}
KINDS = {
    name: label
    for name, label in (
        ("rnn", "RNN"),
        ("gru", "GRU"),
        ("lstm", "LSTM"),
        ("dlinear", "DLinear"),
        ("ridge", "Ridge"),
        ("xgboost", "XGBoost"),
        ("mlp", "MLP"),
        ("boosting", "Boosting"),
        ("zero", "Residual cero"),
        ("adaptation", "Adaptación predictiva"),
        ("unknown", "Modelo no identificado"),
    )
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[\w.-]{1,96}", value, re.ASCII):
        raise ValueError("Identificador de campaña inválido")
    return value


def finite(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def utc(value):
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("La fecha observada necesita zona horaria")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def planned_runs(kind, config, *, parents=1):
    """Contar el diseño fijo sin importar entrenadores ni reservar CUDA."""
    if kind == "neural":
        arms = sum(len(config["pooled_weightings"]) if a == "US+CN" else 1 for a in config["arms"])
        seeds = len(config["finalist_seeds"])
        return (
            arms
            * len(config["models"])
            * (12 + seeds - 1 + seeds * len(config["posttraining_losses"]))
        )
    if kind == "tabular":
        return (
            len(config["ridge_alphas"])
            + len(config["depths"]) * len(config["bins"]) * len(config["rates"])
            + len(config["finalist_seeds"])
            - 1
        )
    if kind == "adaptation":
        return (
            parents
            * len(config["seeds"])
            * sum(
                len(config["betas"]) if mode.startswith("klpo_") else 1 for mode in config["modes"]
            )
        )
    raise ValueError("Tipo de diseño desconocido")


def safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("La fuente debe permanecer dentro de su raíz")
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError("No se admiten enlaces en las fuentes")
    return path


def lock_held(folder):
    for name in (".lock", ".queue.lock", ".study.lock"):
        try:
            fd = os.open(folder / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            continue
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("El bloqueo no es un archivo regular")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
        finally:
            os.close(fd)
    return False


class Collector:
    """Guardar fuentes pequeñas y registros públicos en una transacción local."""

    def __init__(self, root, cache, *, max_files=4096, max_bytes=64 * 1024**2):
        self.root = Path(root).absolute()
        self.max_files, self.max_bytes = max_files, max_bytes
        self.bytes_read = 0
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS sources (path TEXT PRIMARY KEY, signature TEXT, body TEXT)"
        )
        self.db.execute("CREATE TABLE IF NOT EXISTS runs (key TEXT PRIMARY KEY, body TEXT)")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.db.close()

    def read(self, path):
        path = safe_path(self.root, path.relative_to(self.root))
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return None
        with os.fdopen(fd, "rb") as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 2 * 1024**2:
                raise ValueError("La fuente no es un JSON regular de hasta 2 MiB")
            signature = str(
                (metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)
            )
            cached = self.db.execute(
                "SELECT signature, body FROM sources WHERE path=?", (str(path),)
            ).fetchone()
            if cached and signature == cached[0]:
                return json.loads(cached[1])
            body = handle.read(2 * 1024**2 + 1)
            after = os.fstat(handle.fileno())
        self.bytes_read += len(body)
        if (
            self.bytes_read > self.max_bytes
            or len(body) > 2 * 1024**2
            or (after.st_size, after.st_mtime_ns) != (metadata.st_size, metadata.st_mtime_ns)
        ):
            raise ValueError("La fuente ha cambiado durante la lectura o supera el presupuesto")
        try:
            raw = json.loads(body)
            if not isinstance(raw, dict):
                raise ValueError()
            json.dumps(raw, allow_nan=False)
        except (ValueError, RecursionError, UnicodeError):
            raise ValueError("Recibo JSON inválido") from None
        self.db.execute(
            "INSERT OR REPLACE INTO sources VALUES (?, ?, ?)", (str(path), signature, body.decode())
        )
        return raw

    def collect(self, sources, *, now=None):
        now = utc(now or datetime.now(UTC).isoformat())
        self.bytes_read = 0
        campaigns, seen = [], set()
        with self.db:
            for source in sources:
                name = identifier(source["id"])
                if name in seen or source["domain"] not in {"real", "synthetic", "technical"}:
                    raise ValueError("Campaña duplicada o dominio desconocido")
                seen.add(name)
                folder = safe_path(self.root, source["path"])
                summary = self.read(folder / "summary.json") or {}
                planned = summary.get("planned_runs")
                if planned is not None and (
                    type(planned) is not int or not 0 <= planned <= 2**53 - 1
                ):
                    raise ValueError("El recuento previsto necesita un entero acotado")
                if "configuration" in source:
                    configuration = self.read(safe_path(self.root, source["configuration"]))
                    planned = planned_runs(
                        source["kind"], configuration, parents=source.get("parents", 1)
                    )
                    if summary.get("planned_runs", planned) != planned:
                        raise ValueError("El recuento no coincide con la configuración")
                live = folder.is_dir() and lock_held(folder)
                try:
                    records = self._campaign(source, folder, summary, now, live)
                except (TypeError, KeyError, AttributeError, IndexError) as error:
                    raise ValueError("La estructura del recibo no cumple su contrato") from error
                for record in records:
                    key = record["run_id"] + ":" + record["attempt_id"]
                    self.db.execute(
                        "INSERT OR REPLACE INTO runs VALUES (?, ?)",
                        (key, json.dumps(record, allow_nan=False)),
                    )
                campaigns.append(
                    dict(
                        id=name,
                        domain=source["domain"],
                        planned_runs=planned,
                        status=STATUS.get(summary.get("status"), "queued"),
                        dependencies=[identifier(x) for x in source.get("dependencies", [])],
                    )
                )
            stored = self.db.execute(
                "SELECT count(*), coalesce(sum(length(body)),0) FROM sources"
            ).fetchone()
            if stored[0] > self.max_files or stored[1] > self.max_bytes:
                raise ValueError("El registro de fuentes supera su presupuesto")
            runs = [json.loads(row[0]) for row in self.db.execute("SELECT body FROM runs")]
            if len(runs) > self.max_files:
                raise ValueError("El historial supera su presupuesto")
            for campaign in campaigns:
                members = [r for r in runs if r["metadata"]["campaign"] == campaign["id"]]
                latest = {r["run_id"]: r for r in sorted(members, key=lambda r: r["attempt_id"])}
                campaign["counts"] = dict(Counter(r["status"] for r in latest.values()))
                campaign["registered_runs"] = len(latest)
                if campaign["planned_runs"] is None:
                    campaign["planned_runs"] = len(latest)
                campaign["counts"]["not_started"] = max(0, campaign["planned_runs"] - len(latest))
                if campaign["dependencies"] and not members:
                    campaign["status"] = "blocked"
            runs.sort(key=lambda r: (r["status"] != "running", r["run_id"], r["attempt_id"]))
        return dict(
            schema_version=2,
            project="MARS-TITAN",
            generated_at=now,
            source_status="available" if runs else "no_runs_registered",
            poll_interval_seconds=60,
            stale_after_seconds=900,
            campaigns=campaigns,
            models=[dict(id=k, name=v, kind="baseline") for k, v in KINDS.items()],
            runs=runs,
            notes=[
                "El test final permanece sellado. MARS-TITAN no está implementado.",
                "Las curvas sin fechas originales utilizan épocas. "
                "La observación del proceso no acredita nuevo progreso.",
                "El historial conserva comprobaciones técnicas y campañas con poblaciones "
                "distintas. Los grupos de comparación dependen de sus fuentes.",
            ],
        )

    def _campaign(self, source, folder, summary, now, live):
        tasks = {}
        for item in summary.get("runs", []):
            attempts = item.get("attempts") or [item]
            for index, attempt in enumerate(attempts, 1):
                if "path" in attempt or "report_path" in attempt:
                    relative = (
                        Path(attempt["report_path"])
                        if "report_path" in attempt
                        else Path(attempt["path"]) / "run.json"
                    )
                    safe_path(folder, relative)
                    tasks[str(relative)] = {
                        **item,
                        **attempt,
                        "case": item.get("case", item.get("options", {})),
                        "attempt_id": f"attempt-{index:04d}" if item.get("attempts") else "legacy",
                    }
        if folder.exists():
            for path in folder.rglob("run.json"):
                safe_path(folder, path.relative_to(folder))
                if len(tasks) >= self.max_files:
                    raise ValueError("Demasiados recibos en la campaña")
                tasks.setdefault(str(path.relative_to(folder)), {})
        records = []
        for relative, task in tasks.items():
            report_path = safe_path(folder, relative)
            report = self.read(report_path)
            if report is not None and report.get("status") not in STATUS:
                raise ValueError("El recibo existente no declara un estado válido")
            if (
                report is None
                and self.db.execute(
                    "SELECT 1 FROM sources WHERE path=?", (str(report_path),)
                ).fetchone()
            ):
                continue
            report = report or {}
            checkpoint = self.read(report_path.parent / "checkpoints/latest.json") or {}
            record = public_run(source, task, report, checkpoint, relative, report_path, now, live)
            validate_record(record, now)
            records.append(record)
        return records


def public_run(source, task, report, checkpoint, relative, report_path, now, live):
    identity = report.get("identity", {})
    case = identity.get("case", task.get("case", {}))
    kind = case.get("kind", task.get("kind", report.get("model", report.get("kind", "unknown"))))
    mode = case.get("mode")
    model = "adaptation" if mode else kind if kind in KINDS else "unknown"
    status = STATUS.get(report.get("status", task.get("status")))
    # Un recibo de error del coordinador puede ser posterior al último punto del hijo.
    if task.get("status") in {"failed", "paused", "interrupted"}:
        status = STATUS[task["status"]]
    epochs = report.get("epochs", [])
    epochs = epochs if isinstance(epochs, list) else []
    if len(epochs) > 500:
        raise ValueError("Demasiadas épocas en un recibo")
    measures = report.get("predictions", {}).get("validation", {}).get("metrics", {})
    if not measures:
        measures = task.get("metrics", {}).get(kind, {})
    if not measures and epochs:
        measures = epochs[-1].get("validation", {})
    measures = validation_metrics(measures, mode)
    history = [
        dict(
            step=e["epoch"],
            recorded_at=None,
            loss=validation_metrics(e.get("validation", {}), mode)["loss"],
            mae=validation_metrics(e.get("validation", {}), mode)["mae"],
        )
        for e in epochs
    ]
    metrics = dict(measures)
    metrics["elapsed_seconds"] = finite(report.get("total_seconds", report.get("attempt_seconds")))
    for key, raw_key in (
        ("ram_peak_mib", "process_lifetime_peak_rss_bytes"),
        ("vram_peak_mib", "peak_vram_allocated_bytes"),
    ):
        value = finite(report.get(raw_key))
        metrics[key] = value / 1024**2 if value is not None else None
    attempts = report.get("attempts", [])
    if attempts:
        metrics["elapsed_seconds"] = sum(finite(a.get("seconds")) or 0 for a in attempts)
        peaks = [finite(a.get("peak_vram_allocated_bytes")) for a in attempts]
        observed = [p for p in peaks if p is not None]
        metrics["vram_peak_mib"] = max(observed) / 1024**2 if observed else None
    updated = utc(report.get("finished_at_utc"))
    if updated is None and report_path.exists():
        updated = utc(datetime.fromtimestamp(report_path.stat().st_mtime, UTC).isoformat())
    if updated is not None and updated > now:
        raise ValueError("Recibo con fecha posterior a la observación")
    saved = checkpoint.get("latest", [])
    step = finite(report.get("global_step"))
    saved_step = finite(saved[0].get("global_step")) if saved else None
    manifest = identity.get("manifest_sha256", report.get("manifest_sha256"))
    run_key = task.get("id", str(Path(relative).parent))
    name = source["id"] + "-" + digest(run_key)[:20]
    stage = task.get("stage", "adaptation" if mode else "train")
    method = mode or ("supervised_continuation" if stage == "posttraining" else "initial_training")
    if method not in {
        "initial_training",
        "supervised_continuation",
        "reinforce",
        "expected",
        "mae",
        "klpo_full",
        "klpo_mc",
        "klpo_exact",
    }:
        raise ValueError("Método sin contrato público")
    samples = report.get("samples", report.get("counts", task.get("samples", {})))
    fingerprint = digest(
        {
            "manifest": manifest,
            "domain": source["domain"],
            "weighting": identity.get("weighting"),
            "metric": "row_mae",
        }
    )
    return dict(
        run_id=name,
        attempt_id=task.get("attempt_id", "legacy"),
        model_id=model,
        variant_id=digest(case)[:16] if case else None,
        status=status,
        phase="validation",
        started_at=utc(report.get("started_at_utc")),
        updated_at=updated,
        heartbeat_at=now if live and status == "running" else None,
        completed_steps=max(step or 0, saved_step or 0) or None,
        total_steps=None,
        epoch=epochs[-1]["epoch"] if epochs else None,
        max_epochs=finite(case.get("epochs")),
        seed=finite(case.get("seed")),
        fold=task.get("arm") if task.get("arm") in {"US", "CN", "US+CN"} else "unidentified",
        comparison_group=fingerprint if manifest else None,
        metrics=metrics,
        history=history,
        checkpoint=dict(step=saved_step, saved_at=None, resumable=None),
        test_released=False,
        metadata=dict(
            campaign=source["id"],
            domain=source["domain"],
            method=method,
            configuration_sha256=digest(case) if case else None,
            source_sha256=digest(report),
            history_axis="epoch",
            progress_time_source="receipt_mtime",
            train_rows=finite(samples.get("train")),
            validation_rows=finite(samples.get("validation")),
            parent=digest(task["parent"]) if task.get("parent") else None,
            error_type=identifier(task["error_type"]) if task.get("error_type") else None,
        ),
    )


def validation_metrics(measures, mode):
    measures = measures.get("median", {}) if mode else measures
    values = {key: measures.get(key) for key in METRICS}
    for key in ("mae", "mse"):
        if values[key] is None:
            values[key] = measures.get("diagnostic_" + key)
    values["loss"] = measures.get("objective_loss", values["loss"])
    if isinstance(values["loss"], str):
        values["loss"] = None
    for key, value in values.items():
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("La métrica debe ser numérica y finita")
        low = -1 if key == "rank_ic" else -math.inf if key == "loss" else 0
        high = 1 if key == "rank_ic" or key.startswith("coverage_") else math.inf
        if not low <= value <= high:
            raise ValueError("La métrica está fuera de rango")
    return values


def validate_record(record, now):
    """Rechazar progreso y cronología que el navegador no pueda interpretar."""
    counters = [
        record[key] for key in ("completed_steps", "total_steps", "epoch", "max_epochs", "seed")
    ]
    counters += [point["step"] for point in record["history"]]
    counters += [record["checkpoint"]["step"]]
    counters += [record["metadata"][key] for key in ("train_rows", "validation_rows")]
    if any(
        value is not None and (type(value) is not int or not 0 <= value <= 2**53 - 1)
        for value in counters
    ):
        raise ValueError("El progreso necesita contadores enteros acotados")
    if record["epoch"] is not None and record["max_epochs"] is not None:
        if record["epoch"] > record["max_epochs"]:
            raise ValueError("Las épocas superan el presupuesto declarado")
    previous = -1
    for point in record["history"]:
        if point["step"] <= previous:
            raise ValueError("Las épocas no son estrictamente crecientes")
        previous = point["step"]
        if record["completed_steps"] is not None and point["step"] > record["completed_steps"]:
            raise ValueError("La época supera el progreso confirmado")
    for key in ("started_at", "updated_at", "heartbeat_at"):
        if record[key] is not None and datetime.fromisoformat(record[key]) > datetime.fromisoformat(
            now
        ):
            raise ValueError("La fecha del registro está en el futuro")
    if record["started_at"] and record["updated_at"]:
        if datetime.fromisoformat(record["started_at"]) > datetime.fromisoformat(
            record["updated_at"]
        ):
            raise ValueError("El informe precede a su inicio")


def write_pages(snapshot, output, *, page_size=64):
    """Confirmar primero páginas inmutables y después el índice público."""
    if type(page_size) is not int or not 1 <= page_size <= 128:
        raise ValueError("El tamaño de página debe estar entre 1 y 128")
    output = Path(output)
    base = {k: v for k, v in snapshot.items() if k != "runs"}
    pages = []
    for start in range(page_size, len(snapshot["runs"]), page_size):
        runs = snapshot["runs"][start : start + page_size]
        document = dict(base, runs=runs)
        document.pop("campaigns", None)
        timestamps = [
            r[k] for r in runs for k in ("updated_at", "heartbeat_at", "started_at") if r[k]
        ]
        document["generated_at"] = max(
            timestamps, key=datetime.fromisoformat, default=snapshot["generated_at"]
        )
        name = f"pages/{digest(document)}.json"
        if not (output / name).exists():
            atomic_json(output / name, document)
        pages.append(name)
    index = dict(
        base,
        runs=snapshot["runs"][:page_size],
        pagination=dict(total_runs=len(snapshot["runs"]), page_size=page_size, pages=pages),
    )
    atomic_json(output / "observatory.json", index)
    return index
