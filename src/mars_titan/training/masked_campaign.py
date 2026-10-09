"""Ejecutar la campaña con máscaras de la edición desde 2000 y publicar sus fuentes.

La ejecución comprueba el bloqueo de aprendizaje antes de cada trabajo, confirma un
recibo por trabajo con huellas, tramos y filas, no repite los trabajos confirmados con
la misma identidad y rehace los incompletos. Sin declaración de ejecución, los trabajos
CUDA se ejecutan de uno en uno en este proceso bajo una única reserva de la GPU y los
trabajos CPU con la concurrencia declarada. Una declaración de ejecución
(``training.campaign_resources``) puede fijar varias ranuras GPU: cada trabajo GPU corre
entonces en su propio proceso con su VRAM acotada (``training.campaign_slots``), y la
admisión respeta las dependencias del plan, las ranuras y la VRAM y la RAM declaradas.
Los recibos y el resumen se escriben solo desde este proceso, de uno en uno.
Cuando una semilla de un brazo tiene su predictor elegido en una ventana, se escribe el
recibo walk-forward de cada mercado con el contrato de ``environments.walk_forward_receipt``.
Al final se escribe el manifiesto de fuentes de cada ámbito que consume
``evaluation.walk_forward_comparison``. El plan y sus variantes están en
``training.campaign_plan``.

Con una declaración de almacenamiento (``training.campaign_storage``), la ejecución no
empieza si el pico proyectado de lo pendiente no cabe sobre el margen, no admite un
trabajo que no quepa, pausa de forma recuperable si el espacio libre cae por debajo del
margen y libera tras cada recibo los índices y los estados de recuperación.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.environments.walk_forward_receipt import (
    RECEIPT_KIND as WINDOW_RECEIPT_KIND,
)
from mars_titan.environments.walk_forward_receipt import prediction_fingerprint, read_window_receipt
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.evaluation.splits import PARTITIONS

from .campaign_plan import (
    CARRY,
    FIT,
    HELDOUT_RETENTION,
    NEURAL,
    QUANTILE_HEAD,
    _arm_specs,
    _require,
    arm_output,
    check_campaign,
    count_jobs,
    load_campaign,
    plan_campaign,
)
from .campaign_resources import (
    Execution,
    PeakEstimates,
    ResourcePool,
    check_plan,
    environment,
    legacy_execution,
    load_execution,
)
from .campaign_storage import (
    DiskGuard,
    confirmed_ids,
    job_footprint,
    load_storage,
    projection_report,
    release_confirmed,
    view_counts,
)
from .corpus_inputs import DIGEST_CACHE_ENV
from .learning_hold import LearningHoldError, require_learning_allowed

RUN_KIND = "historical_masked_campaign_run"
# Posiciones del plan que una ranura libre puede adelantar a un trabajo que espera.
BACKFILL_JOBS = 16
# Picos de VRAM observados por tipo de trabajo, para admitir los siguientes.
OBSERVED_RESOURCES = "observed-resources.json"
# Huellas de los archivos de las vistas que comparten los trabajos de la campaña.
DIGESTS = "file-digests.json"
RECEIPT_KIND = "masked_campaign_job"
# Tramos que lee la comparación: calibración común y evaluación.
COMPARED = ("calibration", "evaluation")
MAX_ATTEMPTS = 32


class Paused(Exception):
    """Parada solicitada en una barrera confirmada de un trabajo."""


class DiskPaused(Paused):
    """Parada antes de empezar un trabajo que no cabe sobre el margen de disco."""


def _views_report(directory, scope, policy):
    report, digest = read_manifest(directory / "report.json", 1024**2)
    joint = len(comparison.SCOPES[scope]) > 1
    _require(
        isinstance(report, dict)
        and report.get("input_policy") == policy
        and report.get("status") == "temporal_views_prepared"
        and report.get("final_test_opened") is False
        and (report.get("kind") == "joint_temporal_views") == joint
        and report.get("schema_version") == (3 if joint else 2)
        and isinstance(report.get("folds"), list),
        f"Las vistas de {scope} no son vistas preparadas con la política de la campaña",
    )
    return report, digest


def scope_views(directory, scope, campaign):
    """Validar las vistas de un ámbito frente a la comparación y devolver sus huellas."""
    directory = Path(directory)
    policy = campaign["input_policy"]
    resolved = campaign["comparison_config"]["resolved_scopes"][scope]
    report, digest = _views_report(directory, scope, policy)
    _require(
        [row.get("id") for row in report["folds"]] == list(resolved["windows"]),
        f"Las vistas de {scope} no contienen exactamente las ventanas del protocolo",
    )
    windows, edition = {}, None
    for row in report["folds"]:
        _require(row.get("has_all_partitions") is True, f"{row['id']} tiene un tramo vacío")
        record = dict(path=f"{row['id']}/manifest.json", sha256=row.get("manifest_sha256"))
        view, current = comparison._view(
            directory, record, resolved["windows"][row["id"]], resolved, policy
        )
        _require(edition is None or current == edition, "Las ventanas mezclan ediciones")
        edition = current
        manifest, _ = read_manifest(directory / record["path"], 8 * 1024**2)
        counts = manifest.get("counts")
        _require(
            isinstance(counts, dict)
            and set(counts) == set(PARTITIONS)
            and all(type(v) is int and v > 0 for v in counts.values())
            and counts == row.get("counts"),
            f"{row['id']} no conserva sus recuentos declarados",
        )
        windows[row["id"]] = dict(
            path=str((directory / record["path"]).resolve()), sha256=view, counts=counts
        )
    return dict(report_sha256=digest, edition=edition, windows=windows)


def prepare_views(path, parent, output):
    """Preparar las vistas de cada ámbito desde la supervisión histórica, sin ajustar nada."""
    from .joint_temporal_corpus import prepare_joint_temporal_corpus
    from .reference_campaign import campaign_views
    from .temporal_corpus import prepare_temporal_corpus

    campaign = load_campaign(path)
    policy = campaign["input_policy"]
    parent, output = Path(parent), Path(output)
    declared = campaign["comparison_config"]
    folder = Path(campaign["comparison_path"]).parent
    prepared = {}
    for scope in campaign["scopes"]:
        destination = output / scope
        protocols = {
            market: (folder / name).resolve()
            for market, name in declared["scopes"][scope]["protocols"].items()
        }
        if not destination.exists():
            if len(protocols) > 1:
                prepare_joint_temporal_corpus(
                    parent,
                    {market: dict(protocol=value) for market, value in protocols.items()},
                    destination,
                    input_policy=policy,
                    recover_annual_boundaries=True,
                )
            else:
                market = next(iter(protocols))
                projection = output / "parents" / f"{market}.json"
                view = campaign_views(parent, [market], input_policy=policy)[market]
                if projection.exists():
                    _require(
                        read_manifest(projection, 8 * 1024**2)[0] == view,
                        f"La proyección de {market} no corresponde a la supervisión indicada",
                    )
                else:
                    atomic_json(projection, view)
                prepare_temporal_corpus(
                    projection,
                    protocols[market],
                    None,
                    None,
                    destination,
                    input_policy=policy,
                    recover_annual_boundaries=True,
                )
        prepared[scope] = scope_views(destination, scope, campaign)
    return prepared


def _rows_digest(table):
    """Huella de las filas y objetivos, independiente del orden de lectura."""
    market = table["market"].to_numpy(zero_copy_only=False).astype(str)
    asset = table["asset_id"].to_numpy(zero_copy_only=False).astype(str)
    moment = table["prediction_at"].cast(pa.int64()).to_numpy()
    target = table["target"].to_numpy().astype("<f8")
    order = np.lexsort((moment, asset, market))
    digest = hashlib.sha256()
    for values in (market[order], asset[order]):
        digest.update("\n".join(values.tolist()).encode())
        digest.update(b"\0")
    digest.update(moment[order].astype("<i8").tobytes())
    digest.update(target[order].tobytes())
    return digest.hexdigest()


def _fingerprints(table, markets):
    """Huella del recibo walk-forward de las predicciones de cada mercado con filas."""
    market = table["market"].to_numpy(zero_copy_only=False).astype(str)
    moment = table["prediction_at"].cast(pa.int64()).to_numpy()
    asset = table["asset_id"].to_numpy(zero_copy_only=False).astype(str)
    score = table["prediction"].to_numpy()
    result = {}
    for name in markets:
        rows = market == name
        if rows.any():
            count, digest = prediction_fingerprint(moment[rows], asset[rows], score[rows])
            result[name] = dict(rows=count, sha256=digest)
    return result


def _group(job):
    """Brazo, semilla y ventana: la unidad que recibe un predictor elegido."""
    return job["scope"], job["window"], job["arm"], job["seed"]


@dataclass(frozen=True)
class JobRun:
    """Lo que necesita un ejecutor: trabajo, caso resuelto, vista, destino y ancla.

    `parent` solo existe en los ajustes que parten de otro predictor elegido en la misma
    ventana y semilla, como el lector de MARS-TITAN sobre Titans-MAC o un brazo de CM-v1
    sobre su núcleo.
    """

    job: dict
    case: dict | None
    view: Path
    view_sha256: str
    folder: Path
    policy: str
    batch_size: int
    checkpoint_seconds: float
    stop: object
    anchor: dict | None = None
    parent: dict | None = None


def _neural_fit(run):
    from .reference_run import run_reference_case

    report = run_reference_case(
        run.view,
        run.folder,
        run.case,
        batch_size=run.batch_size,
        checkpoint_seconds=run.checkpoint_seconds,
        resume=run.folder.exists(),
        stop=run.stop,
        input_policy=run.policy,
        prediction_retention=HELDOUT_RETENTION,
    )
    if report["status"] == "paused":
        raise Paused
    _require(
        report["status"] == "completed"
        and report["identity"]["manifest_sha256"] == run.view_sha256
        and report["identity"]["case"] == run.case,
        "La referencia no confirma la vista y el caso del trabajo",
    )
    return report


def _ridge_fit(run):
    from .tabular_corpus import run_tabular_reference

    return run_tabular_reference(
        run.view, run.folder, kind="ridge", **run.case, prediction_retention=HELDOUT_RETENTION
    )


def _xgboost_fit(run):
    from .external_corpus import run_external_reference

    report = run_external_reference(
        run.view,
        run.folder,
        resume=(run.folder / "run.json").is_file(),
        stop=run.stop,
        prediction_retention=HELDOUT_RETENTION,
        **run.case,
    )
    if report["status"] == "paused":
        raise Paused
    return report


def _episodic_gru(run, **options):
    from .candidate_walk_forward import run_job

    report = run_job(run, **options)
    if report["status"] == "paused":
        raise Paused
    return report


def _carry(run):
    from .carried_predictions import carry_reference, carry_tabular

    options = dict(batch_size=run.batch_size, input_policy=run.policy)
    sources = (run.anchor["folder"], run.anchor["view"], run.view, run.folder)
    if run.job["model"] == "neural":
        return carry_reference(*sources, stop=run.stop, **options)
    return carry_tabular(*sources, kind=run.job["model"], **options)


def _titans_fit(run):
    from .titans_walk_forward import titans_fit

    return titans_fit(run)


def _titans_carry(run):
    from .titans_walk_forward import titans_carry

    return titans_carry(run)


def _mars_titan_fit(run):
    from .mars_titan_walk_forward import mars_titan_fit

    return mars_titan_fit(run)


def _mars_titan_carry(run):
    from .mars_titan_walk_forward import mars_titan_carry

    return mars_titan_carry(run)


def _cm_v1_core_fit(run):
    from .cm_v1_factorial import cm_v1_core_fit

    return cm_v1_core_fit(run)


def _cm_v1_fit(run):
    from .cm_v1_factorial import cm_v1_fit

    return cm_v1_fit(run)


def _cm_v1_carry(run):
    from .cm_v1_factorial import cm_v1_carry

    return cm_v1_carry(run)


# Ejecutores por modelo y tipo, con su dispositivo y si reanudan el último intento.
EXECUTORS = {
    ("neural", FIT): dict(run=_neural_fit, device="cuda", resumable=True, report="run.json"),
    ("ridge", FIT): dict(run=_ridge_fit, device="cuda", resumable=False, report="run.json"),
    ("xgboost", FIT): dict(run=_xgboost_fit, device="cuda", resumable=True, report="run.json"),
    ("neural", CARRY): dict(run=_carry, device="cuda", resumable=False, report="carry.json"),
    ("ridge", CARRY): dict(run=_carry, device="cuda", resumable=False, report="carry.json"),
    ("xgboost", CARRY): dict(run=_carry, device="cuda", resumable=False, report="carry.json"),
    ("episodic_gru", FIT): dict(
        run=_episodic_gru, device="cuda", resumable=True, report="window.json"
    ),
    ("episodic_gru", CARRY): dict(
        run=_episodic_gru, device="cuda", resumable=False, report="carry.json"
    ),
    ("titans_mac", FIT): dict(run=_titans_fit, device="cuda", resumable=True, report="run.json"),
    ("titans_mac", CARRY): dict(
        run=_titans_carry, device="cuda", resumable=False, report="carry.json"
    ),
    ("mars_titan", FIT): dict(
        run=_mars_titan_fit, device="cuda", resumable=True, report="run.json"
    ),
    ("mars_titan", CARRY): dict(
        run=_mars_titan_carry, device="cuda", resumable=False, report="carry.json"
    ),
    ("cm_v1_core", FIT): dict(
        run=_cm_v1_core_fit, device="cuda", resumable=True, report="run.json"
    ),
    ("cm_v1", FIT): dict(run=_cm_v1_fit, device="cuda", resumable=True, report="run.json"),
    ("cm_v1", CARRY): dict(run=_cm_v1_carry, device="cuda", resumable=False, report="carry.json"),
}


def _code():
    root = Path(__file__).parents[1]
    names = (
        "training/masked_campaign.py",
        "training/campaign_plan.py",
        "training/carried_predictions.py",
        "training/reference_design.py",
        "environments/walk_forward_receipt.py",
        "evaluation/walk_forward_comparison.py",
        "evaluation/splits.py",
    )
    return {name: sha256(root / name) for name in names}


def _identity(campaign, views):
    return dict(
        schema_version=1,
        kind=RUN_KIND,
        campaign_sha256=campaign["sha256"],
        comparison_sha256=campaign["comparison_config"]["sha256"],
        tabular_sha256=campaign["tabular"]["sha256"],
        input_policy=campaign["input_policy"],
        stopping_rule=campaign["rule"],
        variant=campaign["variant"],
        views={
            scope: dict(
                report_sha256=record["report_sha256"],
                windows={w: v["sha256"] for w, v in record["windows"].items()},
            )
            for scope, record in views.items()
        },
        code=_code(),
        final_test_opened=False,
    )


class _Campaign:
    """Estado confirmado de la campaña y verificación de cada recibo."""

    def __init__(self, campaign, views, output, identity, executors, stop, jobs=(), disk=None):
        self.campaign, self.views, self.output = campaign, views, output
        self.identity, self.executors, self.stop = identity, executors, stop
        # Guardia, recuentos y declaración de almacenamiento, o nada sin declaración.
        self.disk = disk
        self.identity_sha256 = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        self.receipts = {}
        # Picos de memoria de los trabajos que corrieron en su propio proceso.
        self.usage = {}
        # Huella de filas y objetivos por ámbito, ventana y tramo, común a todos los brazos.
        self.rows = {}
        self.groups = {}
        for job in jobs:
            self.groups.setdefault(_group(job), []).append(job["id"])

    def same_rows(self, job, receipt):
        """Exigir que cada trabajo de una ventana evalúe las mismas filas que el primero."""
        for partition, record in receipt["predictions"].items():
            key = (job["scope"], job["window"], partition)
            expected = self.rows.setdefault(key, (job["id"], record["rows_sha256"]))
            _require(
                expected[1] == record["rows_sha256"],
                f"{job['id']} no evalúa las mismas filas ni objetivos de {partition} "
                f"que {expected[0]}",
            )

    def folder(self, job):
        return self.output / "jobs" / job["id"]

    def selected(self, scope, window, arm, seed):
        """Recibo elegido para una semilla: ganador de búsqueda, finalista o traslado."""
        prefix = f"{scope}/{window}/{arm}/"
        found = {k: v for k, v in self.receipts.items() if k.startswith(prefix)}
        carry = found.get(f"{prefix}carry-s{seed}")
        if carry is not None:
            return f"{prefix}carry-s{seed}", carry
        finalist = found.get(f"{prefix}finalist-s{seed}")
        if finalist is not None:
            return f"{prefix}finalist-s{seed}", finalist
        searches = [(k, v) for k, v in found.items() if k.startswith(prefix + "search-")]
        _require(searches, f"Falta el ajuste seleccionado de {prefix}")
        key, receipt = min(searches, key=lambda item: (item[1]["score"], item[0]))
        _require(receipt["identity"]["seed"] == seed, f"{prefix} no tiene la semilla {seed}")
        return key, receipt

    def resolve(self, job):
        """Caso y ancla del trabajo a partir de los recibos de sus dependencias."""
        _require(
            all(dep in self.receipts for dep in job["depends"]),
            f"{job['id']} depende de trabajos sin confirmar",
        )
        parent = self.parent_of(job)
        origin = (
            {} if parent is None else dict(parent=parent["job"], parent_sha256=parent["sha256"])
        )
        if job["stage"] == "search":
            return job["case"], None, origin
        if job["stage"] == "finalist":
            # El ganador sale de las búsquedas propias, nunca de las del padre.
            own = f"{job['scope']}/{job['window']}/{job['arm']}/search-"
            key, winner = min(
                ((dep, self.receipts[dep]) for dep in job["depends"] if dep.startswith(own)),
                key=lambda item: (item[1]["score"], item[0]),
            )
            case = winner["identity"]["case"] | dict(seed=job["seed"])
            return case, None, dict(source=key, source_sha256=winner["sha256"], **origin)
        key, receipt = self.selected(job["scope"], job["anchor"], job["arm"], job["seed"])
        anchor = dict(
            folder=self.output / receipt["attempt"],
            view=Path(self.views[job["scope"]]["windows"][job["anchor"]]["path"]),
            job=key,
            sha256=receipt["sha256"],
        )
        return None, anchor, dict(source=key, source_sha256=receipt["sha256"])

    def parent_of(self, job):
        """Predictor elegido del que parte un ajuste con padre en su ventana y semilla."""
        if job.get("parent") is None or job["kind"] != FIT:
            return None
        key, receipt = self.selected(job["scope"], job["window"], job["parent"], job["seed"])
        return dict(
            folder=self.output / receipt["attempt"],
            job=key,
            sha256=receipt["sha256"],
            checkpoint_sha256=receipt["parent"]["sha256"],
        )

    def job_identity(self, job, case, sources):
        view = self.views[job["scope"]]["windows"][job["window"]]
        fields = ("id", "scope", "window", "arm", "family", "model", "stage", "kind", "seed")
        return dict(
            campaign_identity_sha256=self.identity_sha256,
            **{key: job[key] for key in fields},
            anchor=job["anchor"],
            case=case,
            view_sha256=view["sha256"],
            sources=sources,
        )

    def confirmed(self, job, identity):
        """Leer un recibo existente y comprobar identidad y artefactos."""
        path = self.folder(job) / "receipt.json"
        if not path.is_file():
            return None
        receipt, digest = read_manifest(path, 8 * 1024**2)
        _require(
            receipt.get("identity") == identity,
            f"El trabajo confirmado {job['id']} cambió de identidad",
        )
        for record in [receipt["report"], *receipt["predictions"].values()]:
            _require(
                sha256(self.output / record["path"]) == record["sha256"],
                f"Un artefacto confirmado de {job['id']} ha cambiado",
            )
        self.same_rows(job, receipt)
        return dict(receipt, sha256=digest)

    def attempt(self, job, resumable):
        folder = self.folder(job)
        safe_destination(folder)
        attempts = sorted(p.name for p in folder.glob("attempt-*")) if folder.exists() else []
        if resumable and attempts:
            return folder / attempts[-1]
        _require(len(attempts) < MAX_ATTEMPTS, f"{job['id']} alcanzó el límite de intentos")
        return folder / f"attempt-{len(attempts) + 1:04d}"

    def prepare(self, job):
        """Resolver caso, identidad y destino antes de ejecutar un trabajo pendiente."""
        case, anchor, sources = self.resolve(job)
        identity = self.job_identity(job, case, sources)
        receipt = self.confirmed(job, identity)
        if receipt is not None:
            return None, receipt
        executor = self.executors[job["model"], job["kind"]]
        section = self.campaign["neural" if job["family"] == NEURAL else "tabular"]
        view = self.views[job["scope"]]["windows"][job["window"]]
        run = JobRun(
            job=job,
            case=case,
            view=Path(view["path"]),
            view_sha256=view["sha256"],
            folder=self.attempt(job, executor["resumable"]),
            policy=self.campaign["input_policy"],
            batch_size=section["batch_size"],
            checkpoint_seconds=self.campaign["neural"]["checkpoint_seconds"],
            stop=self.stop,
            anchor=anchor,
            parent=self.parent_of(job),
        )
        return (run, identity), None

    def confirm(self, job, run, identity, report):
        """Comprobar predicciones, tramos, filas y puntuación, y escribir el recibo."""
        executor = self.executors[job["model"], job["kind"]]
        resolved = self.campaign["comparison_config"]["resolved_scopes"][job["scope"]]
        window = resolved["windows"][job["window"]]
        view = self.views[job["scope"]]["windows"][job["window"]]
        quantile = arm_output(self.campaign, job["arm"]) == QUANTILE_HEAD
        columns = comparison.COLUMNS + (comparison.QUANTILE_COLUMNS if quantile else ())
        _require(report.get("final_test_opened") is False, f"{job['id']} abre la reserva final")
        predictions = {}
        for partition in COMPARED:
            record = report["predictions"][partition]
            path = run.folder / record["path"]
            safe_destination(path)
            _require(sha256(path) == record["sha256"], f"{job['id']} cambió {partition}")
            table = comparison._read_predictions(dict(path=path, sha256=record["sha256"]), columns)
            label = f"{job['id']} ({partition})"
            comparison._check_segment(table, window, partition, resolved["markets"], label)
            _require(
                table.num_rows == view["counts"][partition],
                f"{label}: {table.num_rows} filas frente a {view['counts'][partition]} de la vista",
            )
            predictions[partition] = dict(
                path=str(path.relative_to(self.output)),
                sha256=record["sha256"],
                rows=table.num_rows,
                rows_sha256=_rows_digest(table),
                markets=_fingerprints(table, resolved["markets"]),
            )
        score = None
        if job["kind"] == FIT:
            score = report["predictions"]["validation"]["metrics"]["session_mae"]
            _require(
                type(score) in (int, float) and math.isfinite(score) and score >= 0,
                f"{job['id']} no tiene un MAE de validación finito",
            )
        report_path = run.folder / executor["report"]
        receipt = dict(
            parent=self.parent(job, identity, report),
            schema_version=1,
            kind=RECEIPT_KIND,
            status="completed",
            identity=identity,
            attempt=str(run.folder.relative_to(self.output)),
            report=dict(path=str(report_path.relative_to(self.output)), sha256=sha256(report_path)),
            score=score,
            predictions=predictions,
            final_test_opened=False,
            confirmed_at_utc=datetime.now(UTC).isoformat(),
        )
        self.same_rows(job, receipt)
        path = self.folder(job) / "receipt.json"
        atomic_json(path, receipt)
        return dict(receipt, sha256=sha256(path))

    def parent(self, job, identity, report):
        """Predictor ajustado del que salen las predicciones: el propio o el del ancla."""
        if job["kind"] == FIT:
            checkpoint = report.get("checkpoint")
            _require(isinstance(checkpoint, dict), f"{job['id']} no declara su estado elegido")
            return dict(id=job["id"], sha256=checkpoint["sha256"])
        source = identity["sources"]["source"]
        parent = self.receipts[source]["parent"]
        _require(
            report.get("anchor", {}).get("checkpoint_sha256") == parent["sha256"],
            f"{job['id']} no parte del estado elegido en {source}",
        )
        return dict(parent)

    def _disk_need(self, job):
        _, counts, storage = self.disk
        footprint = job_footprint(job, counts[job["scope"]][job["window"]], storage)
        return footprint["retained_bytes"] + footprint["transient_bytes"]

    def fits(self, job):
        """Si el trabajo cabe ahora sobre el margen de disco, sin reservar nada."""
        return self.disk is None or self.disk[0].admits(self._disk_need(job))

    def admit(self, job):
        """Empezar un trabajo solo si lo que ocupará deja intacto el margen de disco."""
        if self.disk is None:
            return
        need = self._disk_need(job)
        if not self.disk[0].admits(need, job["id"]):
            raise DiskPaused(f"{job['id']} necesita {need} bytes sobre el margen de disco")

    def release(self, job, receipt):
        """Liberar lo que nadie vuelve a leer de un intento con su recibo ya escrito."""
        if self.disk is None:
            return
        if self.disk[2]["release_on_confirmation"]:
            release_confirmed(self.output / receipt["attempt"], job["model"])
        self.disk[0].settle(job["id"])

    def record(self, job, receipt):
        """Guardar el recibo y, si completa su grupo, publicar los recibos de ventana."""
        self.release(job, receipt)
        self.receipts[job["id"]] = receipt
        group = _group(job)
        # Los auxiliares, como los núcleos de CM-v1, no publican recibo de ventana.
        compared = job["arm"] in self.campaign["comparison_config"]["arms"]
        if compared and all(key in self.receipts for key in self.groups[group]):
            self.publish(*group)

    def publish(self, scope, window, arm, seed):
        """Recibo walk-forward por mercado del predictor elegido para la semilla y ventana.

        La calibración común usa el tramo anterior a la evaluación y la purga obliga a que
        sus etiquetas maduren antes del final del tramo. Por eso la última etiqueta usada se
        acota con el microsegundo anterior a la evaluación, también en una ventana trasladada.
        """
        _, receipt = self.selected(scope, window, arm, seed)
        resolved = self.campaign["comparison_config"]["resolved_scopes"][scope]
        folder = self.output / "windows" / scope / window / arm / f"seed-{seed}"
        for market in resolved["markets"]:
            record = dict(
                kind=WINDOW_RECEIPT_KIND,
                schema_version=1,
                protocol=resolved["protocols"][market],
                fold=resolved["windows"][window],
                parent=receipt["parent"],
                labels_used_until=0,
                predictions={
                    partition: value["markets"][market]
                    for partition, value in receipt["predictions"].items()
                    if market in value["markets"]
                },
            )
            start, _ = read_window_receipt(record).segment("evaluation")
            record["labels_used_until"] = start - 1
            read_window_receipt(record)
            path = folder / f"{market}.json"
            if path.is_file():
                _require(
                    read_manifest(path, 1024**2)[0] == record,
                    f"El recibo de {path.relative_to(self.output)} no corresponde a sus trabajos",
                )
            else:
                atomic_json(path, record)


def _summary(output, identity, jobs, receipts, status, **extra):
    """Resumen confirmado de la campaña. Solo lo escribe el proceso de la campaña."""
    planned = Counter(job["kind"] for job in jobs)
    done = Counter(job["kind"] for job in jobs if job["id"] in receipts)
    summary = dict(
        schema_version=1,
        kind=RUN_KIND,
        status=status,
        identity_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        planned=dict(training_jobs=planned[FIT], prediction_jobs=planned[CARRY]),
        completed=dict(training_jobs=done[FIT], prediction_jobs=done[CARRY]),
        jobs={job["id"]: job["id"] in receipts for job in jobs},
        final_test_opened=False,
        updated_at_utc=datetime.now(UTC).isoformat(),
        **extra,
    )
    atomic_json(output / "summary.json", summary)
    return summary


def run_campaign(
    path, views, output, *, executors=None, lease=None, stop=None, storage=None, execution=None
):
    """Ejecutar o reanudar la campaña. Los ejecutores y la reserva se pueden sustituir.

    `storage` es la ruta de la declaración de almacenamiento. Con ella se comprueba el
    pico proyectado antes de escribir nada y la ejecución vigila el margen de disco.
    `execution` es la ruta de una declaración de ejecución o una `Execution`. Sin ella se
    conserva la ejecución anterior. La declaración no entra en la identidad de la campaña,
    así que una campaña puede reanudarse con otra concurrencia.
    """
    from .checkpoints import StopRequest

    require_learning_allowed("la campaña con máscaras")
    campaign = load_campaign(path)
    jobs = plan_campaign(campaign)
    count_jobs(campaign, jobs)
    if execution is None:
        execution = legacy_execution(campaign)
    elif not isinstance(execution, Execution):
        execution = load_execution(execution, scopes=tuple(campaign["scopes"]))
    _require(
        isinstance(views, dict) and set(views) == set(campaign["scopes"]),
        "Se necesitan las vistas de exactamente los ámbitos de la campaña",
    )
    checked = {scope: scope_views(Path(views[scope]), scope, campaign) for scope in views}
    output = Path(output)
    safe_destination(output)
    for directory in views.values():
        outside_source(Path(directory), output)
        outside_source(output, Path(directory))
    outside_source(Path("dataset"), output)
    executors = dict(EXECUTORS if executors is None else executors)
    _require(set(executors) == set(EXECUTORS), "Faltan ejecutores para algún modelo")
    check_plan(execution, jobs, executors)
    identity = _identity(campaign, checked)
    disk, launch = None, None
    if storage is not None:
        declared = load_storage(storage)
        guard = DiskGuard(output, declared["margin_bytes"], check_seconds=declared["check_seconds"])
        counts = view_counts(checked)
        launch = guard.require_launch(
            projection_report(jobs, counts, declared, confirmed_ids(output, jobs))
        )
        disk = (guard, counts, declared)
    output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = output / "campaign.json"
        if marker.exists():
            _require(
                read_manifest(marker, 8 * 1024**2)[0] == identity,
                "La salida pertenece a otra campaña, vista o código",
            )
        else:
            _require(
                all(p.name in {".lock", "summary.json"} for p in output.iterdir()),
                "La salida sin identidad contiene artefactos ajenos",
            )
            atomic_json(marker, identity)
        state = _Campaign(campaign, checked, output, identity, executors, None, jobs, disk)
        uses_gpu = any(executors[j["model"], j["kind"]]["device"] == "cuda" for j in jobs)
        default_lease = _slot_lease(execution) if execution.isolated else _gpu_lease
        reservation = (lease or default_lease)() if uses_gpu else nullcontext()
        signals = StopRequest() if stop is None else nullcontext(stop)
        pause = None
        record = dict(execution=execution.record())

        def disk_report():
            if disk is None:
                return {}
            pending = {} if pause is None else dict(pause=pause)
            return dict(disk=dict(launch=launch, guard=disk[0].state(), **pending))

        _summary(output, identity, jobs, state.receipts, "running", **record, **disk_report())
        try:
            with (
                signals as requested,
                _environment(execution, output),
                reservation,
                ThreadPoolExecutor(execution.cpu_workers) as pool,
            ):
                state.stop = requested if disk is None else disk[0].watch(requested)
                status = _execute(state, jobs, pool, execution)
        except Paused as error:
            status = "paused"
            if isinstance(error, DiskPaused):
                pause = str(error)
            elif disk is not None and disk[0].low:
                pause = "El espacio libre bajó del margen durante un trabajo"
        except LearningHoldError as error:
            _summary(
                output,
                identity,
                jobs,
                state.receipts,
                "blocked",
                error=str(error),
                **record,
                **disk_report(),
            )
            raise
        except BaseException as error:
            _summary(
                output,
                identity,
                jobs,
                state.receipts,
                "failed",
                error=str(error),
                **record,
                **disk_report(),
            )
            raise
        return _summary(
            output,
            identity,
            jobs,
            state.receipts,
            status,
            **record,
            **disk_report(),
            usage=state.usage,
        )
    finally:
        os.close(descriptor)


def _gpu_lease():
    from .experiment_resources import GpuLease

    return GpuLease()


class _environment:
    """Variables de la tubería durante la campaña, restauradas al terminar.

    Las huellas de los archivos de las vistas se comparten en `file-digests.json` de la
    salida, de modo que cada trabajo, en este proceso o en su ranura, no vuelve a leer
    completos los archivos que ya comprobó otro con la misma firma de stat.
    """

    def __init__(self, execution, output):
        self.values, self.previous = environment(execution), {}
        if DIGEST_CACHE_ENV not in os.environ:
            self.values[DIGEST_CACHE_ENV] = str(Path(output).resolve() / DIGESTS)

    def __enter__(self):
        for name, value in self.values.items():
            self.previous[name] = os.environ.get(name)
            os.environ[name] = value
        return self

    def __exit__(self, *_):
        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _slot_lease(execution):
    def lease():
        from .campaign_slots import SlotLease

        return SlotLease(execution)

    return lease


def _launch_order(pending, execution, estimates, executors, ready, fits):
    """Trabajos que se intentan lanzar, en orden.

    Sin ranuras aisladas, solo el primero pendiente, como en la ejecución en serie. Con
    ranuras, los `BACKFILL_JOBS` primeros del plan de mayor a menor VRAM estimada, de modo
    que los grandes no esperan detrás de los pequeños y estos rellenan lo que queda. Con la
    misma VRAM se conserva el orden del plan. Si el primer trabajo listo de esa ventana no
    cabe (`fits`), su dispositivo queda reservado para él: no se intenta ningún otro trabajo
    de ese dispositivo hasta que los que están en curso le dejen sitio. Así un trabajo
    grande no espera indefinidamente detrás de los pequeños que lo adelantan.
    """
    if not execution.isolated:
        return pending[:1]
    window = pending[:BACKFILL_JOBS]

    def device(job):
        return executors[job["model"], job["kind"]]["device"]

    def size(job):
        return estimates.resources(job, device(job)).vram_bytes

    order = sorted(window, key=size, reverse=True)
    head = next((job for job in window if ready(job)), None)
    if head is None or fits(head):
        return order
    return [job for job in order if job is head or device(job) != device(head)]


class _Running:
    """Trabajo lanzado: su plan, su ejecución, su identidad y sus recursos."""

    def __init__(self, job, run, identity, resources):
        self.job, self.run, self.identity, self.resources = job, run, identity, resources


def _execute(state, jobs, pool, execution):
    """Recorrer el plan por orden de prioridad respetando dependencias y recursos.

    Un trabajo se lanza cuando sus dependencias están confirmadas y `ResourcePool` lo
    admite. Si no cabe, se prueban los siguientes del plan. Sin ranuras aisladas, cada
    trabajo CUDA se ejecuta en este proceso hasta terminar, como antes. Con ranuras, cada
    trabajo GPU corre en su proceso y este bucle solo lanza, espera y confirma. Una parada o
    un fallo detienen los lanzamientos, piden a los trabajos en curso que se detengan en su
    siguiente barrera y esperan a todos antes de terminar.
    """
    from .campaign_slots import (
        SlotProcess,
        SlotTask,
        executor_name,
        new_event,
        run_fields,
        slot_environment,
        strict_fp32,
        wait,
    )

    admission = ResourcePool(execution)
    estimates = PeakEstimates(execution, state.output / OBSERVED_RESOURCES)
    pending, threads, slots = list(jobs), {}, {}
    # Trabajos CUDA declarados en el proceso de la campaña: uno a la vez, en un hilo.
    in_process = ThreadPoolExecutor(1) if execution.isolated else None
    event = new_event() if execution.isolated else None
    failure, paused = None, False

    def finish(entry, report):
        admission.release(entry.resources)
        state.record(entry.job, state.confirm(entry.job, entry.run, entry.identity, report))
        _summary(
            state.output,
            state.identity,
            jobs,
            state.receipts,
            "running",
            execution=execution.record(),
        )

    def collect(block):
        nonlocal failure, paused
        handles = list(slots)
        if block and (threads or slots):
            futures = list(threads)
            if futures and not handles:
                wait_futures(futures, return_when=FIRST_COMPLETED, timeout=0.5)
            elif handles:
                wait(handles, 0.05 if futures else 0.5)
        for future in [future for future in threads if future.done()]:
            entry = threads.pop(future)
            try:
                report = future.result()
            except Paused:
                admission.release(entry.resources)
                paused = True
                continue
            except BaseException as error:  # Se lanza al terminar los demás trabajos.
                admission.release(entry.resources)
                failure = failure or error
                continue
            try:
                finish(entry, report)
            except BaseException as error:
                failure = failure or error
        for handle in handles:
            result = handle.poll()
            if result is None:
                continue
            entry = slots.pop(handle)
            status, value, usage = result[:3]
            state.usage[entry.job["id"]] = usage
            estimates.observe(entry.job, usage)
            if status == "completed":
                try:
                    finish(entry, value)
                except BaseException as error:
                    failure = failure or error
            else:
                admission.release(entry.resources)
                if status == "paused":
                    paused = True
                elif value["type"] == "OutOfMemoryError" and estimates.grow(
                    entry.job, entry.resources
                ):
                    # Solo falló este proceso: se repite desde su intento con más VRAM.
                    pending.insert(0, entry.job)
                else:
                    failure = failure or RuntimeError(
                        f"{entry.job['id']} falló en su proceso: {value['type']}: "
                        f"{value['message']}\n{result[3] if len(result) > 3 else ''}"
                    )

    def ready(job):
        return all(dep in state.receipts for dep in job["depends"])

    def admitted(job):
        resources = estimates.resources(job, state.executors[job["model"], job["kind"]]["device"])
        # Con trabajos en curso, uno que no cabe en disco espera a que liberen su reserva.
        fits = admission.admits(resources) and (not (threads or slots) or state.fits(job))
        return resources, fits

    def reserves(job):
        """Si el trabajo no cabe por recursos que otros lanzamientos le seguirían quitando.

        Esperar al hilo de la campaña no reserva el dispositivo: las ranuras siguen llenándose
        hasta que ese hilo quede libre.
        """
        resources, fits = admitted(job)
        return not fits and not admission.waits_for_campaign(resources)

    def launch():
        """Lanzar los trabajos listos en orden. Devuelve si alguno empezó o se confirmó.

        Sin ranuras aisladas solo se considera el primer trabajo pendiente, como en la
        ejecución en serie. Con ranuras se adelantan como mucho `BACKFILL_JOBS` posiciones
        del plan, para ocupar una ranura libre sin alejarse del orden declarado.
        """
        progressed = False
        order = _launch_order(
            pending, execution, estimates, state.executors, ready, lambda job: not reserves(job)
        )
        for job in order:
            if failure is not None or paused or state.stop.requested:
                break
            if not ready(job):
                continue
            resources, fits = admitted(job)
            if not fits:
                continue
            executor = state.executors[job["model"], job["kind"]]
            prepared, receipt = state.prepare(job)
            pending.remove(job)
            progressed = True
            if receipt is not None:
                state.record(job, receipt)
                continue
            require_learning_allowed(f"el trabajo {job['id']}")
            state.admit(job)
            run, identity = prepared
            entry = _Running(job, run, identity, resources)
            admission.acquire(resources)
            if resources.device == "cpu":
                threads[pool.submit(executor["run"], run)] = entry
            elif execution.isolated and resources.campaign_process:
                strict_fp32()
                admission.hold_context()
                threads[in_process.submit(executor["run"], run)] = entry
            elif execution.isolated:
                lock = state.folder(job) / ".job.lock"
                lock.parent.mkdir(parents=True, exist_ok=True)
                task = SlotTask(
                    executor=executor_name(executor["run"]),
                    run=run_fields(run),
                    environment=slot_environment(execution, resources),
                    vram_bytes=resources.vram_bytes,
                    lock=str(lock),
                )
                slots[SlotProcess(task, event)] = entry
            else:
                try:
                    if resources.device == "cuda":
                        strict_fp32()
                    report = executor["run"](run)
                except BaseException:
                    admission.release(resources)
                    raise
                finish(entry, report)
                # El plan vuelve a recorrerse desde el principio, como en la ejecución en serie.
                return True
        return progressed

    try:
        while pending or threads or slots:
            collect(block=False)
            stopping = failure is not None or paused or state.stop.requested
            if stopping:
                if event is not None:
                    event.set()
                if not (threads or slots):
                    break
                collect(block=True)
                continue
            if pending and launch():
                continue
            if threads or slots:
                collect(block=True)
                continue
            if pending:
                blocked = pending[0]
                raise RuntimeError(
                    f"{blocked['id']} no puede empezar: sus dependencias no se confirmaron "
                    "o sus recursos declarados no caben en la memoria libre"
                )
    except BaseException:
        if event is not None:
            event.set()
        while threads or slots:
            collect(block=True)
        raise
    finally:
        if in_process is not None:
            in_process.shutdown()
    if failure is not None:
        raise failure
    if paused or state.stop.requested:
        raise Paused
    _require(len(state.receipts) == len(jobs), "La campaña no confirmó todos sus trabajos")
    return "completed"


def _confirmed_state(path, views, output):
    """Reconstruir los recibos confirmados de una campaña sin ejecutar ningún trabajo."""
    campaign = load_campaign(path)
    checked = {scope: scope_views(Path(views[scope]), scope, campaign) for scope in views}
    identity = _identity(campaign, checked)
    marker = Path(output) / "campaign.json"
    _require(
        marker.is_file() and read_manifest(marker, 8 * 1024**2)[0] == identity,
        "La salida no corresponde a esta campaña, sus vistas o su código",
    )
    return campaign, _Campaign(campaign, checked, Path(output), identity, EXECUTORS, None)


def write_sources(path, views, output, scope, *, comparison_path=None):
    """Escribir el manifiesto de fuentes de un ámbito y validarlo con la comparación.

    Sin `comparison_path` se valida con la comparación de la campaña, que incluye los
    brazos de las familias sin entrenador conectado. Una comparación declarada con un
    subconjunto de brazos permite evaluar los brazos ya producidos.
    """
    campaign, state = _confirmed_state(path, views, output)
    _require(scope in campaign["scopes"], "El ámbito no pertenece a la campaña")
    validation = comparison.load_config(
        Path(comparison_path or campaign["comparison_path"]).resolve()
    )
    produced = {spec["arm"]: spec for spec in _arm_specs(campaign)}
    wanted = {
        name: arm for name, arm in validation["arms"].items() if arm["output"] != "zero_control"
    }
    missing = sorted(set(wanted) - set(produced))
    _require(
        not missing,
        f"Faltan productores para {', '.join(missing)}. Declara una comparación con los "
        "brazos disponibles o conecta su entrenador",
    )
    jobs = [job for job in plan_campaign(campaign) if job["scope"] == scope]
    for job in jobs:
        case, _, sources = state.resolve(job)
        receipt = state.confirmed(job, state.job_identity(job, case, sources))
        _require(receipt is not None, f"Falta confirmar {job['id']} antes de publicar fuentes")
        state.receipts[job["id"]] = receipt
    windows = state.views[scope]["windows"]
    folder = Path(output) / "sources"
    arms, rows = {}, {}
    for name, arm in wanted.items():
        _require(
            sorted(arm["seeds"]) == sorted(produced[name]["seeds"]),
            f"La comparación declara otras semillas para {name}",
        )
        arms[name] = {}
        for seed in arm["seeds"]:
            entries = arms[name][str(seed)] = {}
            for window in windows:
                key, receipt = state.selected(scope, window, name, seed)
                entry = dict(
                    input_policy=campaign["input_policy"], view_sha256=windows[window]["sha256"]
                )
                for partition, record in receipt["predictions"].items():
                    rows.setdefault((window, partition), {})[key] = record["rows_sha256"]
                    # Una salida puntual no tiene cuantiles que calibrar.
                    if partition == "evaluation" or arm["output"] == QUANTILE_HEAD:
                        entry[partition] = dict(
                            path=os.path.relpath(Path(output) / record["path"], folder),
                            sha256=record["sha256"],
                        )
                entries[window] = entry
    for (window, partition), digests in rows.items():
        _require(
            len(set(digests.values())) == 1,
            f"{len(set(digests.values()))} conjuntos de filas distintos en {window} ({partition})",
        )
    manifest = dict(
        schema_version=1,
        kind=comparison.SOURCES_KIND,
        scope=scope,
        input_policy=campaign["input_policy"],
        windows={
            w: dict(view=dict(path=v["path"], sha256=v["sha256"])) for w, v in windows.items()
        },
        arms=arms,
    )
    destination = folder / f"{scope}.json"
    candidate = folder / f".{scope}.candidate.json"
    safe_destination(destination)
    atomic_json(candidate, manifest)
    try:
        comparison.load_sources(candidate, validation, scope)
    except BaseException:
        candidate.unlink()
        raise
    os.replace(candidate, destination)
    return destination


def _views_argument(values):
    views = {}
    for value in values or []:
        scope, _, directory = value.partition("=")
        _require(scope in comparison.SCOPES and directory, "Usa --views ÁMBITO=DIRECTORIO")
        _require(scope not in views, f"El ámbito {scope} aparece dos veces")
        views[scope] = Path(directory)
    return views


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Validar y contar trabajos sin leer datos")
    prepare = commands.add_parser("prepare", help="Preparar las vistas de cada ámbito")
    execute = commands.add_parser("run", help="Ejecutar o reanudar la campaña")
    sources = commands.add_parser("sources", help="Publicar el manifiesto de fuentes")
    for command in (check, prepare, execute, sources):
        command.add_argument("--campaign", type=Path, required=True)
    prepare.add_argument("--parent", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    for command in (execute, sources):
        command.add_argument("--views", action="append", required=True)
        command.add_argument("--output", type=Path, required=True)
    execute.add_argument(
        "--storage",
        type=Path,
        required=True,
        help="Declaración de almacenamiento con el margen de disco y los bytes medidos",
    )
    execute.add_argument(
        "--execution",
        type=Path,
        help="Declaración de ejecución con ranuras GPU, trabajadores CPU, memoria y lectura",
    )
    sources.add_argument("--scope", choices=tuple(comparison.SCOPES), required=True)
    sources.add_argument("--comparison", type=Path)
    args = parser.parse_args(argv)
    if args.command == "check":
        result = check_campaign(args.campaign)
    elif args.command == "prepare":
        prepared = prepare_views(args.campaign, args.parent, args.output)
        result = {
            scope: dict(record, windows=len(record["windows"]))
            for scope, record in prepared.items()
        }
    elif args.command == "run":
        result = run_campaign(
            args.campaign,
            _views_argument(args.views),
            args.output,
            storage=args.storage,
            execution=args.execution,
        )
        result.pop("jobs")
    else:
        destination = write_sources(
            args.campaign,
            _views_argument(args.views),
            args.output,
            args.scope,
            comparison_path=args.comparison,
        )
        result = dict(sources=str(destination), sha256=sha256(destination))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status", "completed") in {"completed", "checked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
