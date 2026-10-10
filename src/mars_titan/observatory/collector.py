"""Recolección incremental, saneamiento e historial paginado de campañas."""

import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.storage import atomic_json
from mars_titan.training.reference_design import candidate_indices

from .activities import (
    ADAPTIVE_VARIANTS,
    CONDITIONS,
    FINANCIAL,
    PREDICTIVE,
    classify,
    financial_validation,
    native_adaptation,
)
from .bounded_files import signature
from .window_campaigns import STAGES as WINDOW_STAGE_SHAPES
from .window_campaigns import campaign_state

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
    "session_mae",
)
# Medidas opcionales por época que el navegador lee si existen. Un valor ausente,
# negativo o no finito se publica como null para no rechazar recibos anteriores.
EPOCH_MEASURES = (
    ("train_mae", "train", "mae"),
    ("train_samples_per_second", "train", "samples_per_second"),
    ("train_seconds", "train", "elapsed_seconds"),
    ("session_mae", "validation", "session_mae"),
    ("validation_seconds", "validation", "elapsed_seconds"),
)
STATUS = {
    "pending": "queued",
    "waiting": "queued",
    "interrupted": "paused",
    "running": "running",
    "completed": "completed",
    "failed": "failed",
    "paused": "paused",
    "cancelled": "cancelled",
    "blocked": "blocked",
    "queued": "queued",
}

ADAPTIVE_STAGES = {
    "pilot": "piloto",
    "main": "comparacion",
    "auxiliary": "consolidacion",
    "audit": "auditoria",
}


def adaptive_task(item, relative):
    stage, model, seed = item.get("stage"), item.get("variant"), item.get("seed")
    configuration = item.get("config_sha256")
    if (
        stage not in ADAPTIVE_STAGES
        or model not in ADAPTIVE_VARIANTS
        or type(seed) is not int
        or seed not in {42, 43, 44}
        or str(relative) != f"{stage}/{model}-{seed}/run.json"
        or not isinstance(configuration, str)
        or not re.fullmatch(r"[a-f0-9]{64}", configuration)
    ):
        raise ValueError(
            "El registro adaptativo no conserva etapa, modelo, semilla o configuración"
        )
    return dict(
        kind=model,
        activity="evaluation" if stage == "audit" else "rl",
        phase="evaluation" if stage == "audit" else "train",
        variant_id=ADAPTIVE_STAGES[stage],
        native_registry=True,
    )


def frozen_evaluation_tasks(summary, maximum):
    """Recoger solo los informes confirmados por la selección congelada."""
    identity, runs = summary["identity"], summary["runs"]
    jobs = {job["id"]: job for job in identity["jobs"]}
    if (
        type(summary.get("schema_version")) is not int
        or summary["schema_version"] != 1
        or summary.get("final_test_opened") is not False
        or identity.get("partitions") != ["calibration", "evaluation"]
        or len(jobs) != len(identity["jobs"])
        or len(jobs) > maximum
        or not isinstance(runs, dict)
        or summary.get("planned_runs") != len(jobs)
        or summary.get("completed_runs") != len(runs)
        or summary.get("status") == "completed"
        and len(runs) != len(jobs)
    ):
        raise ValueError("El resumen de evaluación no conserva la selección congelada")
    tasks = {}
    for key, saved in runs.items():
        if key not in jobs or saved.get("path") != f"runs/{key}/run.json":
            raise ValueError("El registro de evaluación no corresponde a su trabajo previsto")
        tasks[saved["path"]] = dict(saved, id=key, job=jobs[key])
    return tasks


def frozen_evaluation_task(task, report):
    """Identificar la evaluación sin heredar el modo utilizado para entrenar."""
    family = report.get("family")
    family = {"xgboost_external_cuda": "xgboost"}.get(family, family)
    if (
        type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
        or report.get("job") != task["job"]
        or report.get("status") != "completed"
        or report.get("final_test_opened") is not False
        or family not in {"rnn", "lstm", "gru", "dlinear", "ridge", "xgboost"}
        or set(report.get("predictions", {})) != {"calibration", "evaluation"}
        or report.get("identity") not in (None, {})
        or report.get("activity", "evaluation") != "evaluation"
        or report.get("phase", "evaluation") != "evaluation"
    ):
        raise ValueError("El recibo no conserva el contrato de evaluación congelada")
    return dict(
        activity="evaluation",
        phase="evaluation",
        case={"kind": family, "seed": report["case"].get("seed")},
        variant_id=identifier(task["id"].replace("/", ".")),
    )


KINDS = {
    name: label
    for name, label in (
        ("rnn", "RNN"),
        ("gru", "GRU"),
        ("lstm", "LSTM"),
        ("dlinear", "DLinear"),
        ("ridge", "Ridge"),
        ("xgboost", "XGBoost"),
        ("transformer_compact", "Transformer compacto"),
        ("titans_mac", "Titans-MAC"),
        ("gru_episodic", "GRU episódica"),
        ("mars_titan", "MARS-TITAN"),
        ("cm_v1", "CM-v1"),
        ("mlp", "MLP"),
        ("boosting", "Boosting"),
        ("zero", "Residual cero"),
        ("adaptation", "Adaptación predictiva"),
        ("factor_world", "Generador de mundos sintéticos"),
        ("ppo", "PPO"),
        ("klpo", "KLPO"),
        ("double_dqn", "Double DQN"),
        ("ppo_window", "PPO con ventana temporal"),
        ("ppo_gru", "PPO con GRU"),
        ("ppo_episodic", "PPO con memoria episódica"),
        ("ppo_hmm", "PPO con HMM"),
        ("ppo_episodic_hmm", "PPO con memoria episódica y HMM"),
        ("ppo_recent_aux", "PPO con consolidación reciente"),
        ("ppo_replay_aux", "PPO con consolidación histórica"),
        ("simulator", "Simulador financiero"),
        ("cash", "Mantener efectivo"),
        ("hold_initial", "Conservar posiciones iniciales"),
        ("rebalance_50", "Reequilibrar al 50 %"),
        ("equal_weight_monthly", "Pesos iguales con reequilibrio mensual"),
        ("market_index", "Índice de mercado"),
        ("financial_comparison", "Resumen de comparación financiera"),
        ("adaptive_comparison", "Campaña de adaptación RL"),
        ("unknown", "Modelo no identificado"),
    )
}

# Categoría de cada modelo del catálogo. Los demás son referencias predictivas.
MODEL_KINDS = {
    "factor_world": "generator",
    "simulator": "simulation",
    "cash": "financial_baseline",
    "hold_initial": "financial_baseline",
    "rebalance_50": "financial_baseline",
    "equal_weight_monthly": "financial_baseline",
    "market_index": "financial_baseline",
    "financial_comparison": "summary",
    "adaptive_comparison": "summary",
    "klpo": "reinforcement",
    "titans_mac": "memory",
    "gru_episodic": "memory",
    "mars_titan": "memory",
    "cm_v1": "memory",
}
WINDOW_SOURCE = "window_campaign"
WINDOW_STAGES = {stage for stage, _ in WINDOW_STAGE_SHAPES.values()}
# Con 13.029 trabajos y sus mapas de actualizaciones y de la cadena, el resumen de
# adaptadores de A se estima en unos 2 MiB.
WINDOW_SUMMARY_BYTES = 8 * 1024**2
WINDOW_JOBS = 20_000
WINDOW_GRACE_SECONDS = 3600
WINDOW_REFRESH_SECONDS = 60


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
    if not isinstance(config, dict):
        raise ValueError("Falta una configuración de campaña válida")
    if kind == "neural":
        arms = sum(len(config["pooled_weightings"]) if a == "US+CN" else 1 for a in config["arms"])
        seeds = len(config["finalist_seeds"])
        candidates = len(candidate_indices(config))
        return (
            arms
            * len(config["models"])
            * (candidates + seeds - 1 + seeds * len(config["posttraining_losses"]))
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
    if kind in {"paired_posttraining", "financial"}:
        version = config.get("schema_version")
        versions = {1, 2} if kind == "paired_posttraining" else {1}
        if (
            type(version) is not int
            or version not in versions
            or config.get("final_test_opened") is not False
        ):
            raise ValueError("El diseño debe declarar su versión y mantener cerrado el test")
        if version == 2 and config.get("parent_seed_policy") != "matching":
            raise ValueError("La versión 2 requiere padres emparejados por semilla")
        fields = (
            ("seeds", "conditions", "modes", "neural_controls")
            if kind == "paired_posttraining"
            else ("seeds", "algorithms")
        )
        counts = {}
        for field in fields:
            values = config.get(field)
            item_type = int if field == "seeds" else str
            if (
                not isinstance(values, list)
                or not (0 if field == "neural_controls" else 1) <= len(values) <= 128
                or any(type(value) is not item_type for value in values)
                or len(set(values)) != len(values)
                or field == "seeds"
                and any(not 0 <= value < 2**32 for value in values)
            ):
                raise ValueError("Las listas del diseño deben ser únicas, válidas y acotadas")
            if item_type is str:
                for value in values:
                    identifier(value)
            counts[field] = len(values)
        if kind == "financial":
            return counts["seeds"] * counts["algorithms"]
        if (
            not isinstance(parents, dict)
            or not 1 <= len(parents) <= 64
            or any(
                not isinstance(family, str) or family not in {"neural", "tabular"}
                for family in parents.values()
            )
        ):
            raise ValueError("Cada padre debe declarar una familia neuronal o tabular")
        for parent in parents:
            identifier(parent)
        objectives = (
            len(parents) * counts["modes"]
            + sum(family == "neural" for family in parents.values()) * counts["neural_controls"]
        )
        return counts["seeds"] * counts["conditions"] * objectives
    raise ValueError("Tipo de diseño desconocido")


def _check_sha256(body, expected):
    if (
        not isinstance(expected, str)
        or not re.fullmatch(r"[a-f0-9]{64}", expected)
        or hashlib.sha256(body).hexdigest() != expected
    ):
        raise ValueError("La configuración no conserva su huella confirmada")


def _report_paths(folder, maximum):
    pending, entries_seen = [folder], 0
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                entries_seen += 1
                if entries_seen > maximum * 16:
                    raise ValueError("El recorrido de informes supera su presupuesto")
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in {"private", "checkpoints", "trace"}:
                        pending.append(Path(entry.path))
                elif entry.name == "run.json":
                    yield safe_path(folder, Path(entry.path).relative_to(folder))


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
    for name in (".lock", ".queue.lock", ".study.lock", ".run.lock", ".campaign.lock"):
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
        self.window_cache = {}
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS sources (path TEXT PRIMARY KEY, signature TEXT, body TEXT)"
        )
        self.db.execute("CREATE TABLE IF NOT EXISTS runs (key TEXT PRIMARY KEY, body TEXT)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS run_origins "
            "(campaign TEXT, path TEXT, task_id TEXT, run_id TEXT, attempt_id TEXT, "
            "provisional INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(campaign, path))"
        )
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(run_origins)")}
        for name, declaration in (("attempt_id", "TEXT"), ("provisional", "INTEGER DEFAULT 0")):
            if name not in columns:
                self.db.execute(f"ALTER TABLE run_origins ADD COLUMN {name} {declaration}")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS run_origin_attempt "
            "ON run_origins(campaign, run_id, attempt_id)"
        )

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.db.close()

    def read(self, path, *, expected_sha256=None):
        relative = path.relative_to(self.root)
        if "private" in relative.parts:
            raise ValueError("No se leen los estados privados de las ejecuciones")
        path = safe_path(self.root, relative)
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
                if expected_sha256 is not None:
                    _check_sha256(cached[1].encode(), expected_sha256)
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
        if expected_sha256 is not None:
            _check_sha256(body, expected_sha256)
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
        if now is not None:
            now = utc(now)
            if now is None:
                raise ValueError("La fecha de observación no es válida")
        self.bytes_read = 0
        campaigns, windows, seen = [], [], set()
        with self.db:
            for source in sources:
                name = identifier(source["id"])
                if name in seen or source["domain"] not in {"real", "synthetic", "technical"}:
                    raise ValueError("Campaña duplicada o dominio desconocido")
                seen.add(name)
                folder = safe_path(self.root, source["path"])
                if source["kind"] == WINDOW_SOURCE:
                    windows.append(self._window_campaign(source, folder))
                    continue
                summary_path = safe_path(folder, source.get("summary", "summary.json"))
                cached = self.db.execute(
                    "SELECT body FROM sources WHERE path=?", (str(summary_path),)
                ).fetchone()
                if cached:
                    # El resumen anterior acredita los nombres que pueden acabar de cambiar.
                    self._run_identities(
                        source, self._tasks(folder, json.loads(cached[0]), discover=False)
                    )
                summary = self.read(summary_path) or {}
                planned = summary.get("planned_runs")
                if planned is not None and (
                    type(planned) is not int or not 0 <= planned <= 2**53 - 1
                ):
                    raise ValueError("El recuento previsto necesita un entero acotado")
                if "configuration" in source:
                    configuration = self.read(
                        safe_path(self.root, source["configuration"]),
                        expected_sha256=source.get("configuration_sha256"),
                    )
                    if configuration is None and "configuration_snapshot" in source:
                        if "configuration_sha256" not in source:
                            raise ValueError("La copia de configuración necesita una huella fijada")
                        configuration = self.read(
                            safe_path(self.root, source["configuration_snapshot"]),
                            expected_sha256=source["configuration_sha256"],
                        )
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
            origins = self.db.execute(
                "SELECT count(*), coalesce(sum(length(campaign)+length(path)+"
                "coalesce(length(task_id),0)+length(run_id)+coalesce(length(attempt_id),0)+1),0) "
                "FROM run_origins"
            ).fetchone()
            if (
                stored[0] > self.max_files
                or origins[0] > self.max_files
                or stored[1] + origins[1] > self.max_bytes
            ):
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
                if campaign["dependencies"] and not members and campaign["status"] == "queued":
                    campaign["status"] = "blocked"
            runs.sort(key=lambda r: (r["status"] != "running", r["run_id"], r["attempt_id"]))
        return dict(
            schema_version=2,
            project="MARS-TITAN",
            generated_at=now or utc(datetime.now(UTC).isoformat()),
            source_status="available" if runs else "no_runs_registered",
            poll_interval_seconds=60,
            stale_after_seconds=900,
            campaigns=campaigns,
            models=[
                dict(
                    id=k,
                    name=v,
                    kind="reinforcement"
                    if k in ADAPTIVE_VARIANTS
                    else MODEL_KINDS.get(k, "baseline"),
                )
                for k, v in KINDS.items()
            ],
            runs=runs,
            window_campaigns=windows,
            notes=[
                "El test final permanece sellado.",
                "Las campañas por ventanas publican el estado de cada trabajo y la fecha de su "
                "recibo, sin las métricas de sus resúmenes.",
                "Las curvas sin fechas originales utilizan épocas. "
                "La observación del proceso no acredita nuevo progreso.",
                "El historial conserva comprobaciones técnicas y campañas con poblaciones "
                "distintas. Los grupos de comparación dependen de sus fuentes.",
                "Los entornos de adaptación RL son escenarios técnicos con 3 conceptos macro "
                "simulados. No equivalen al corpus real con 140 indicadores.",
            ],
        )

    def _window_campaign(self, source, folder):
        """Estado de una campaña por ventanas, o su ausencia si todavía no hay resumen.

        Sus trabajos no pasan a registros: la campaña A tiene decenas de miles y el sitio
        los muestra como una matriz de estados que se publica en un documento aparte.
        """
        declared = dict(
            id=source["id"],
            domain=source["domain"],
            stage=source.get("stage"),
            configuration=source.get("configuration"),
        )
        if declared["stage"] not in WINDOW_STAGES or not isinstance(declared["configuration"], str):
            raise ValueError("La campaña por ventanas debe declarar su etapa y su configuración")
        if not safe_path(self.root, declared["configuration"]).is_file():
            raise ValueError("Falta la configuración declarada de la campaña por ventanas")
        summary = folder / "summary.json"
        if summary.is_symlink():
            raise ValueError("No se admiten enlaces en las fuentes")
        found = signature(summary)
        if found is None:
            return dict(declared, state=None)
        # Con el mismo resumen solo pueden cambiar los intentos abiertos, que se releen
        # como mucho cada WINDOW_REFRESH_SECONDS. Recorrer 13.029 trabajos cuesta unos
        # 0,1 s, demasiado para repetirlo cada 15 s sin cambios.
        cached = self.window_cache.get(source["id"])
        now = time.monotonic()
        if cached and cached[0] == (found, folder) and now - cached[1] < WINDOW_REFRESH_SECONDS:
            return dict(declared, state=cached[2])
        self.bytes_read += found[1]
        if self.bytes_read > self.max_bytes:
            raise ValueError("La fuente supera el presupuesto de lectura")
        state = campaign_state(
            source["id"], folder, max_bytes=WINDOW_SUMMARY_BYTES, max_jobs=WINDOW_JOBS
        )
        if state["stage"] != declared["stage"]:
            raise ValueError("El resumen no corresponde a la etapa declarada de la campaña")
        self.window_cache[source["id"]] = ((found, folder), now, state)
        return dict(declared, state=state)

    def _tasks(self, folder, summary, *, discover):
        frozen = summary.get("kind") == "frozen_temporal_evaluation"
        tasks = frozen_evaluation_tasks(summary, self.max_files) if frozen else {}
        adaptive = summary.get("kind") == "adaptive_campaign"
        if adaptive and summary.get("schema_version") != 1:
            raise ValueError("La versión del registro adaptativo no está admitida")
        runs = [] if frozen else summary.get("runs", [])
        if len(runs) > self.max_files:
            raise ValueError("Demasiados recibos en el resumen de campaña")
        if isinstance(runs, dict):
            runs = [{**record, "id": key} for key, record in runs.items()]
        for item in runs:
            attempts = item.get("attempts") or [item]
            for index, attempt in enumerate(attempts, 1):
                if "path" in attempt or "report_path" in attempt:
                    relative = Path(attempt.get("report_path", attempt.get("path")))
                    if "report_path" not in attempt and relative.name != "run.json":
                        relative /= "run.json"
                    if set(relative.parts[:-1]) & {"private", "checkpoints"}:
                        raise ValueError("El recibo no puede estar dentro de los estados privados")
                    safe_path(folder, relative)
                    if str(relative) in tasks:
                        raise ValueError("Dos entradas del resumen reclaman el mismo recibo")
                    tasks[str(relative)] = {
                        **item,
                        **attempt,
                        "case": item.get("case", item.get("options", {})),
                        "attempt_id": identifier(attempt.get("attempt_id", f"attempt-{index:04d}"))
                        if item.get("attempts")
                        else item.get("attempt_id", "legacy"),
                        "attempt_declared": bool(item.get("attempts")) or "attempt_id" in item,
                    }
                    if adaptive:
                        tasks[str(relative)].update(adaptive_task(item, relative))
        if discover and folder.exists() and not frozen:
            for path in _report_paths(folder, self.max_files):
                relative = str(path.relative_to(folder))
                if adaptive and relative == "run.json":
                    continue
                if relative not in tasks and len(tasks) >= self.max_files:
                    raise ValueError("Demasiados recibos en la campaña")
                tasks.setdefault(relative, {})
        return tasks

    def _run_identities(self, source, tasks):
        """Vincular rutas y trabajos sin inferir identidad de sus métricas o fechas."""
        campaign = source["id"]
        origins = {
            path: (task_id, run_id)
            for path, task_id, run_id in self.db.execute(
                "SELECT path, task_id, run_id FROM run_origins WHERE campaign=?", (campaign,)
            )
        }
        existing = {
            key.rsplit(":", 1)[0]
            for (key,) in self.db.execute(
                "SELECT key FROM runs WHERE key >= ? AND key < ?",
                (campaign + "-", campaign + "."),
            )
        }
        groups = {}
        for path, task in tasks.items():
            group = ("task", task["id"]) if "id" in task else ("path", path)
            groups.setdefault(group, []).append(path)
        by_task = {}
        for old_task, run_id in origins.values():
            if old_task is not None:
                by_task.setdefault(old_task, set()).add(run_id)
        prefix = str(safe_path(self.root, source["path"])) + "/"
        receipt_paths = set(tasks) | set(origins)
        receipt_paths.update(
            path[len(prefix) :]
            for (path,) in self.db.execute(
                "SELECT path FROM sources WHERE path >= ? AND path < ?",
                (prefix, prefix[:-1] + "0"),
            )
            if path.endswith("/run.json")
        )
        route_owners = {}
        for path in receipt_paths:
            alias = campaign + "-" + digest(str(Path(path).parent))[:20]
            route_owners.setdefault(alias, set()).add(path)
        identities = {}
        claimed = {run_id: (run_id, True) for _, run_id in origins.values()}
        for (kind, value), paths in groups.items():
            task_id = value if kind == "task" else None
            bound = {origins[path][1] for path in paths if path in origins}
            bound.update(by_task.get(task_id, ()))
            if len(bound) > 1:
                raise ValueError("El trabajo contradice asociaciones de identidad existentes")
            aliases = {campaign + "-" + digest(str(Path(path).parent))[:20] for path in paths}
            declared = campaign + "-" + digest(task_id)[:20] if task_id is not None else None
            if declared is not None:
                aliases.add(declared)
            known = aliases & existing
            if bound:
                target = next(iter(bound))
            elif declared in existing:
                target = declared
            elif len(known) == 1:
                target = next(iter(known))
            elif len(known) > 1:
                raise ValueError("Falta una identidad confirmada para conciliar estos orígenes")
            else:
                target = declared or next(iter(aliases))
            for alias in known:
                if any(
                    path not in paths and origins.get(path, (None, None))[1] != target
                    for path in route_owners.get(alias, ())
                ):
                    raise ValueError("El alias también identifica un recibo de otro origen")
            for alias in known | {target}:
                if alias in claimed and (
                    claimed[alias][0] != target or not (bound and claimed[alias][1])
                ):
                    raise ValueError("Una identidad pertenece a dos orígenes distintos")
                claimed[alias] = (target, bool(bound))
            restored = {
                path: self._cached_attempt(source, path, tasks[path], target)
                for path in paths
                if path not in origins
            }
            for alias in known - {target}:
                self._reconcile_run(campaign, alias, target)
            for path in paths:
                self.db.execute(
                    "INSERT INTO run_origins (campaign, path, task_id, run_id) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(campaign, path) DO UPDATE SET "
                    "task_id=coalesce(excluded.task_id,task_id), run_id=excluded.run_id",
                    (campaign, path, task_id, target),
                )
                if path in restored:
                    attempt, provisional = restored[path]
                    self.db.execute(
                        "UPDATE run_origins SET attempt_id=?, provisional=? "
                        "WHERE campaign=? AND path=?",
                        (attempt, int(provisional), campaign, path),
                    )
                identities[path] = target
        return identities

    def _cached_attempt(self, source, path, task, run_id):
        """Recuperar el intento antiguo desde su recibo cacheado y la clave de esa ruta."""
        cached = self.db.execute(
            "SELECT body FROM sources WHERE path=?",
            (str(self.root / source["path"] / path),),
        ).fetchone()
        if cached is None:
            return None, False
        report = json.loads(cached[0])
        discovered = source["id"] + "-" + digest(str(Path(path).parent))[:20]
        if (
            "attempt_id" not in report
            and self.db.execute(
                "SELECT 1 FROM runs WHERE key=?", (discovered + ":legacy",)
            ).fetchone()
        ):
            return "legacy", True
        attempt = report.get("attempt_id", task.get("attempt_id", "legacy"))
        identifier(attempt)
        if self.db.execute("SELECT 1 FROM runs WHERE key=?", (run_id + ":" + attempt,)).fetchone():
            return attempt, "attempt_id" not in report and not task.get("attempt_declared")
        return None, False

    def _reconcile_run(self, campaign, alias, target):
        """Conservar intentos sin sustituto y preferir el registro de la identidad confirmada."""
        keys = self.db.execute(
            "SELECT key FROM runs WHERE key >= ? AND key < ?", (alias + ":", alias + ";")
        ).fetchall()
        for (key,) in keys:
            self._move_record(campaign, key, target + ":" + key.rsplit(":", 1)[1])

    def _move_record(self, campaign, key, target):
        source = self.db.execute("SELECT body FROM runs WHERE key=?", (key,)).fetchone()
        if source is None or key == target:
            return
        destination = self.db.execute("SELECT body FROM runs WHERE key=?", (target,)).fetchone()
        for stored_key, stored in ((key, source), (target, destination)):
            if stored is None:
                continue
            record = json.loads(stored[0])
            if (
                record["metadata"]["campaign"] != campaign
                or stored_key != record["run_id"] + ":" + record["attempt_id"]
            ):
                raise ValueError("El alias no conserva su campaña e identidad originales")
        if destination is None:
            record = json.loads(source[0])
            record["run_id"], record["attempt_id"] = target.rsplit(":", 1)
            self.db.execute(
                "INSERT INTO runs VALUES (?, ?)", (target, json.dumps(record, allow_nan=False))
            )
        self.db.execute("DELETE FROM runs WHERE key=?", (key,))

    def _attempt_identity(self, source, path, task, report, run_id):
        """Confirmar el nombre de un intento provisional solo con su ruta acreditada."""
        campaign = source["id"]
        previous, provisional = self.db.execute(
            "SELECT attempt_id, provisional FROM run_origins WHERE campaign=? AND path=?",
            (campaign, path),
        ).fetchone()
        if "attempt_id" in report:
            attempt, inferred = report["attempt_id"], False
        elif task.get("attempt_declared"):
            attempt, inferred = task["attempt_id"], False
        else:
            attempt = previous or "legacy"
            inferred = provisional if previous else True
        identifier(attempt)
        if self.db.execute(
            "SELECT 1 FROM run_origins WHERE campaign=? AND run_id=? AND attempt_id=? AND path!=?",
            (campaign, run_id, attempt, path),
        ).fetchone():
            raise ValueError("Dos rutas declaran la misma identidad de intento")
        promotion = (
            previous == "legacy"
            and provisional
            and attempt != previous
            and task.get("attempt_declared")
            and attempt == task["attempt_id"]
        )
        if (
            previous != attempt
            and not promotion
            and self.db.execute(
                "SELECT 1 FROM runs WHERE key=?", (run_id + ":" + attempt,)
            ).fetchone()
        ):
            raise ValueError("El intento existente carece de una asociación con esta ruta")
        if promotion:
            self._move_record(campaign, run_id + ":legacy", run_id + ":" + attempt)
        self.db.execute(
            "UPDATE run_origins SET attempt_id=?, provisional=? WHERE campaign=? AND path=?",
            (attempt, int(inferred), campaign, path),
        )
        return attempt

    def _campaign(self, source, folder, summary, now, live):
        frozen = summary.get("kind") == "frozen_temporal_evaluation"
        tasks = self._tasks(folder, summary, discover=True)
        identities = self._run_identities(source, tasks)
        records = []
        for relative, task in tasks.items():
            report_path = safe_path(folder, relative)
            report = self.read(
                report_path, expected_sha256=(task.get("sha256") or "") if frozen else None
            )
            if frozen:
                task.update(frozen_evaluation_task(task, report or {}))
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
            if (
                report
                and task.get("native_registry")
                and (
                    report.get("model") != task["kind"]
                    or report.get("activity") != task["activity"]
                    or report.get("seed") != task["seed"]
                )
            ):
                raise ValueError("El recibo nativo no corresponde a su caso registrado")
            recovery = report.get("checkpoint", {})
            checkpoint = (
                {}
                if frozen or any(key in recovery for key in ("step", "saved_at", "resumable"))
                else self.read(report_path.parent / "checkpoints/latest.json") or {}
            )
            record = public_run(source, task, report, checkpoint, relative, report_path, now, live)
            record["run_id"] = identities[relative]
            record["attempt_id"] = self._attempt_identity(
                source, relative, task, report, record["run_id"]
            )
            validate_record(record, now)
            records.append(record)
        return records


def _resources(report, phase):
    attempts = report.get("attempts") or []
    if not isinstance(attempts, list) or any(not isinstance(row, dict) for row in attempts):
        raise ValueError("Los intentos deben conservar una lista de medidas")
    records = [report, *attempts]
    if native_adaptation(report):
        resource = report.get("resources", {})
        field = (
            "executable_peak_rss_bytes"
            if resource.get("ram_peak_method") == "procfs_VmHWM"
            else "process_lifetime_peak_rss_bytes"
        )
        records.append(
            {
                field: resource.get("ram_peak_bytes"),
                "peak_vram_allocated_bytes": resource.get("vram_peak_bytes"),
            }
        )

    def peak(field):
        values = [finite(row.get(field)) for row in records]
        observed = [value for value in values if value is not None]
        return max(observed) / 1024**2 if observed else None

    elapsed = finite(
        report.get("total_seconds", report.get("attempt_seconds", report.get("elapsed_seconds")))
    )
    timings = [finite(row.get("total_seconds", row.get("seconds"))) for row in attempts]
    observed = [value for value in timings if value is not None]
    if observed:
        elapsed = sum(observed)
        if not math.isfinite(elapsed):
            raise ValueError("El tiempo acumulado supera el rango numérico")
    executable = peak("executable_peak_rss_bytes")
    lifetime = peak("process_lifetime_peak_rss_bytes")
    scope = (
        "executable"
        if executable is not None
        else "process_lifetime"
        if lifetime is not None
        else None
    )
    metadata = dict(
        ram_peak_scope=scope,
        executable_peak_rss_mib=executable,
        process_lifetime_peak_rss_mib=lifetime,
    )
    resources = dict(
        elapsed_seconds=elapsed,
        ram_peak_mib=executable if executable is not None else lifetime,
        vram_peak_mib=peak("peak_vram_allocated_bytes"),
    )
    if phase in {"test", "evaluation"}:
        metadata = dict.fromkeys(metadata)
        resources = dict.fromkeys(resources)
    return resources, metadata


def public_run(source, task, report, checkpoint, relative, report_path, now, live):
    identity = report.get("identity", {})
    case = identity.get(
        "case", identity.get("model_config", identity.get("config", task.get("case", {})))
    )
    if report.get("domain", source["domain"]) != source["domain"]:
        raise ValueError("La procedencia del informe no coincide con su campaña")
    activity, method = classify(report, task, case)
    phase = report.get(
        "phase",
        task.get("phase")
        or {"synthetic_generation": "prepare", "rl": "train"}.get(activity, "validation"),
    )
    kind = case.get("kind", task.get("kind", report.get("model", report.get("kind", "unknown"))))
    mode = case.get("mode")
    model = (
        "adaptation"
        if activity == "predictive_adaptation"
        else kind
        if kind in KINDS
        else "unknown"
    )
    status = STATUS.get(report.get("status", task.get("status")))
    # Un recibo de error del coordinador puede ser posterior al último punto del hijo.
    if task.get("status") in {"failed", "paused", "interrupted", "blocked", "cancelled", "waiting"}:
        status = STATUS[task["status"]]
    epochs = report.get("epochs", [])
    epochs = epochs if isinstance(epochs, list) else []
    if len(epochs) > 500:
        raise ValueError("Demasiadas épocas en un recibo")
    predictive = activity in PREDICTIVE and phase not in {"test", "evaluation"}
    measures = (
        report.get("predictions", {}).get("validation", {}).get("metrics", {}) if predictive else {}
    )
    if not measures:
        measures = task.get("metrics", {}).get(kind, {})
    if not measures and epochs:
        measures = epochs[-1].get("validation", {})
    measures = validation_metrics(measures if predictive else {}, mode)
    history = [
        dict(
            step=e["epoch"],
            recorded_at=None,
            loss=validation_metrics(e.get("validation", {}), mode)["loss"],
            mae=validation_metrics(e.get("validation", {}), mode)["mae"],
            **epoch_measures(e, mode),
        )
        for e in epochs
        if predictive
    ]
    resources, memory = _resources(report, phase)
    metrics = dict(measures, **resources)
    observed_time = (
        report.get("finished_at_utc") or report.get("updated_at_utc") or report.get("updated_at")
    )
    updated = utc(observed_time)
    if updated is None and report_path.exists():
        updated = utc(datetime.fromtimestamp(report_path.stat().st_mtime, UTC).isoformat())
    # Una escritura puede confirmarse después de comenzar el recorrido del recolector.
    now = now or utc(datetime.now(UTC).isoformat())
    if updated is not None and datetime.fromisoformat(updated) > datetime.fromisoformat(now):
        raise ValueError("Recibo con fecha posterior a la observación")
    saved = checkpoint.get("latest", [])
    recovery = report.get("checkpoint", {})
    step = report.get("global_step")
    saved_step = saved[0].get("global_step") if saved else recovery.get("step")
    progress = [value for value in (step, saved_step) if value is not None]
    saved_at = utc(recovery.get("saved_at"))
    resumable = recovery.get("resumable")
    environment = identity.get("environment", {})
    if not environment and activity in FINANCIAL:
        environment = {
            key: case.get(key, identity.get(key))
            for key in ("capital", "participation", "allocation", "score_scale", "ruin_penalty")
        }
    manifest = identity.get(
        "manifest_sha256",
        report.get("manifest_sha256", identity.get("tape_sha256", environment.get("tape_sha256"))),
    )
    if manifest is None:
        manifest = identity.get("dataset", {}).get("validation_sha256")
    parent = identity.get("parent")
    parent_model = parent.get("model", kind) if isinstance(parent, dict) else kind
    parent_model = {"xgboost_external_cuda": "xgboost"}.get(parent_model, parent_model)
    if activity not in {"predictive_adaptation", "supervised_continuation"} or parent_model not in {
        "rnn",
        "lstm",
        "gru",
        "dlinear",
        "ridge",
        "xgboost",
    }:
        parent_model = None
    condition = case.get("condition")
    if condition not in CONDITIONS:
        condition = None
    currency = report.get("currency")
    if currency is not None and (
        not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency)
    ):
        raise ValueError("La moneda necesita un código público de tres letras")
    run_key = task.get("id", str(Path(relative).parent))
    name = source["id"] + "-" + digest(run_key)[:20]
    samples = report.get("samples", report.get("counts", task.get("samples", {})))
    fingerprint = digest(
        {
            "manifest": manifest,
            "domain": source["domain"],
            "activity": activity,
            "objective": identity.get("objective", case.get("objective", report.get("objective"))),
            "weighting": identity.get("weighting"),
            "metric": "financial_return" if activity in FINANCIAL else "row_mae",
            "currency": currency,
            "cost_bps": report.get(
                "cost_bps", identity.get("cost_bps", environment.get("cost_bps"))
            ),
            "partition": report.get("partition"),
            "financial_conditions": environment
            if environment
            else {key: identity.get(key) for key in ("capital", "participation", "allocation")},
        }
    )
    return dict(
        run_id=name,
        attempt_id=identifier(report.get("attempt_id", task.get("attempt_id", "legacy"))),
        model_id=model,
        activity=activity,
        variant_id=task.get("variant_id") or (digest(case)[:16] if case else None),
        status=status,
        phase=phase,
        started_at=utc(report.get("started_at_utc") or report.get("started_at")),
        updated_at=updated,
        heartbeat_at=now if live and status == "running" else None,
        completed_steps=max(progress) if progress else None,
        total_steps=report.get(
            "total_steps",
            task.get("planned_transitions")
            if task.get("native_registry") and activity == "rl"
            else None,
        ),
        epoch=epochs[-1]["epoch"] if epochs and predictive else None,
        max_epochs=finite(case.get("epochs")),
        seed=case.get("seed", report.get("seed", task.get("seed"))),
        fold=task.get("arm") if task.get("arm") in {"US", "CN", "US+CN"} else None,
        comparison_group=fingerprint if manifest else None,
        metrics=metrics,
        history=history,
        checkpoint=dict(step=saved_step, saved_at=saved_at, resumable=resumable),
        financial_validation=None
        if phase in {"test", "evaluation"}
        else financial_validation(report, activity),
        test_released=False,
        metadata=dict(
            campaign=source["id"],
            domain=source["domain"],
            method=method,
            parent_frozen=report.get("parent_frozen"),
            currency=currency,
            condition=condition,
            parent_model=parent_model,
            configuration_sha256=task.get("config_sha256")
            if task.get("native_registry")
            else digest(case)
            if case
            else None,
            source_sha256=digest(report),
            history_axis="epoch" if predictive else "none",
            **selection_summary(report, predictive),
            progress_time_source="receipt_timestamp" if observed_time else "receipt_mtime",
            train_rows=finite(samples.get("train")),
            validation_rows=finite(samples.get("validation")),
            parent=digest(parent or task["parent"]) if parent or task.get("parent") else None,
            error_type=identifier(task["error_type"]) if task.get("error_type") else None,
            **memory,
        ),
    )


def _primary(measures, mode):
    if mode and isinstance(measures, dict):
        measures = measures.get("median", measures if measures.get("primary") == "median" else {})
    return measures if isinstance(measures, dict) else {}


def epoch_measures(epoch, mode):
    """Copiar las medidas opcionales de una época sin calcular ninguna nueva."""
    sections = dict(
        train=_primary(epoch.get("train"), None), validation=_primary(epoch.get("validation"), mode)
    )
    return {name: finite(sections[section].get(key)) for name, section, key in EPOCH_MEASURES}


def selection_summary(report, predictive):
    """Mejor época y parada temprana declaradas por el entrenador, solo en validación."""
    selection = report.get("selection") if predictive else None
    selection = selection if isinstance(selection, dict) else {}
    best = selection.get("best_epoch")
    stopped = report.get("stopped_early") if predictive else None
    return dict(
        best_epoch=best if type(best) is int and 0 <= best <= 2**53 - 1 else None,
        stopped_early=stopped if type(stopped) is bool else None,
    )


def validation_metrics(measures, mode):
    if mode:
        measures = measures.get("median", measures if measures.get("primary") == "median" else {})
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
    now = now or utc(datetime.now(UTC).isoformat())
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
    if record["phase"] not in {
        "prepare",
        "train",
        "validation",
        "calibration",
        "test",
        "evaluation",
    }:
        raise ValueError("Fase sin contrato público")
    if record["total_steps"] is not None and record["completed_steps"] is not None:
        if record["completed_steps"] > record["total_steps"]:
            raise ValueError("El progreso supera el total declarado")
    recovery = record["checkpoint"]
    for flag in (recovery["resumable"], record["metadata"]["parent_frozen"]):
        if flag is not None and type(flag) is not bool:
            raise ValueError("La recuperación y el padre congelado requieren booleanos")
    if recovery["resumable"] and (recovery["step"] is None or recovery["saved_at"] is None):
        raise ValueError("No se acredita recuperación sin paso y fecha guardados")
    if recovery["saved_at"] and datetime.fromisoformat(
        recovery["saved_at"]
    ) > datetime.fromisoformat(record["updated_at"] or now):
        raise ValueError("El checkpoint es posterior al informe")
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
    base = {k: v for k, v in snapshot.items() if k not in {"runs", "window_campaigns"}}
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
    windows = [_window_entry(entry, output) for entry in snapshot.get("window_campaigns", [])]
    index = dict(
        base,
        runs=snapshot["runs"][:page_size],
        pagination=dict(total_runs=len(snapshot["runs"]), page_size=page_size, pages=pages),
        window_campaigns=windows,
    )
    atomic_json(output / "observatory.json", index)
    _prune_windows(output, {entry["path"] for entry in windows})
    return index


def _prune_windows(output, current, *, grace=WINDOW_GRACE_SECONDS):
    """Borrar los documentos de ventanas que ya no enumera el índice.

    Cada confirmación cambia la matriz y su huella, así que sin limpieza la salida crecería
    con cada trabajo. Se espera `grace` segundos desde la última escritura, porque una
    publicación en curso o un navegador con el índice anterior todavía pueden leerlos.
    """
    folder = output / "windows"
    if not folder.is_dir():
        return
    limit = time.time() - grace
    for path in folder.glob("*.json"):
        name = f"windows/{path.name}"
        if name not in current and path.lstat().st_mtime < limit:
            path.unlink(missing_ok=True)


def _window_entry(entry, output):
    """Entrada del índice de una campaña por ventanas y su documento inmutable.

    El índice lleva los recuentos para elegir campaña sin descargar nada más. La matriz va
    en `windows/<huella>.json`, que el navegador solo pide al mostrar esa campaña.
    """
    state = entry["state"]
    declared = {key: entry[key] for key in ("id", "domain", "stage", "configuration")}
    if state is None:
        return dict(declared, path=None, status=None, updated_at=None, jobs=0, done=0, attempts=0)
    name = f"windows/{digest(state)}.json"
    if not (output / name).exists():
        atomic_json(output / name, state)
    states = Counter(cell[4] for cell in state["cells"])
    return dict(
        declared,
        path=name,
        status=state["status"],
        updated_at=state["updated_at"],
        jobs=len(state["cells"]),
        done=states["done"],
        attempts=states["attempt"],
    )
