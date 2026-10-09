"""Ejecutar la campaña con máscaras de la edición desde 2000 y publicar sus fuentes.

La ejecución comprueba el bloqueo de aprendizaje antes de cada trabajo, confirma un
recibo por trabajo con huellas, tramos y filas, no repite los trabajos confirmados con
la misma identidad y rehace los incompletos. Los trabajos CUDA se ejecutan de uno en
uno bajo una única reserva de la GPU y los trabajos CPU con la concurrencia declarada.
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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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

from . import campaign_numerics, campaign_schedule
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
    launch_blockers,
    load_campaign,
    plan_campaign,
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
from .learning_hold import LearningHoldError, require_learning_allowed

RUN_KIND = "historical_masked_campaign_run"
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


def _eligibility(campaign, scope):
    """Protocolos de elegibilidad por mercado que la comparación declara para un ámbito."""
    declared = campaign["comparison_config"]
    design = declared.get(comparison.JOINT_FIELD)
    if design is None or design["joint_scope"] != scope:
        return None
    folder = Path(campaign["comparison_path"]).parent
    return {
        market: (folder / name).resolve() for market, name in design["market_eligibility"].items()
    }


def prepare_views(path, parent, output, *, scopes=None):
    """Preparar las vistas de cada ámbito desde la supervisión histórica, sin ajustar nada.

    `scopes` limita la preparación a algunos ámbitos de la campaña. Un ámbito con
    elegibilidad por mercado en la comparación la aplica al preparar la unión.
    """
    from .joint_temporal_corpus import prepare_joint_temporal_corpus
    from .reference_campaign import campaign_views
    from .temporal_corpus import prepare_temporal_corpus

    campaign = load_campaign(path)
    policy = campaign["input_policy"]
    parent, output = Path(parent), Path(output)
    declared = campaign["comparison_config"]
    folder = Path(campaign["comparison_path"]).parent
    scopes = list(campaign["scopes"]) if scopes is None else list(scopes)
    _require(
        scopes and len(set(scopes)) == len(scopes) and set(scopes) <= set(campaign["scopes"]),
        "Los ámbitos que se preparan deben ser de la campaña",
    )
    prepared = {}
    for scope in scopes:
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
                    eligibility=_eligibility(campaign, scope),
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
        # Solo la versión 2 declara la precisión, así que la identidad de la 1 no cambia.
        **({"numerics": campaign["numerics"]} if campaign.get("numerics") else {}),
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
        numerics = self.campaign.get("numerics")
        _require(
            numerics is None or receipt.get("numerics") == numerics,
            f"El recibo de {job['id']} registra otra precisión numérica",
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
        if self.campaign.get("numerics"):
            # En cada trabajo, antes de que su ejecutor cree modelos.
            campaign_numerics.apply(self.campaign["numerics"])
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
        numerics = self.campaign.get("numerics")
        if numerics:
            campaign_numerics.require_job(numerics, job["id"], report)
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
        if numerics:
            receipt["numerics"] = campaign_numerics.current()
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

    def admit(self, job):
        """Empezar un trabajo solo si lo que ocupará deja intacto el margen de disco."""
        if self.disk is None:
            return
        guard, counts, storage = self.disk
        footprint = job_footprint(job, counts[job["scope"]][job["window"]], storage)
        need = footprint["retained_bytes"] + footprint["transient_bytes"]
        if not guard.admits(need, job["id"]):
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
        # Un mercado sin filas en algún tramo comparado de la ventana no tiene recibo.
        for market in [
            market
            for market in resolved["markets"]
            if all(market in value["markets"] for value in receipt["predictions"].values())
        ]:
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
    path, views, output, *, executors=None, lease=None, stop=None, storage=None, window=None
):
    """Ejecutar o reanudar la campaña. Los ejecutores y la reserva se pueden sustituir.

    `storage` es la ruta de la declaración de almacenamiento. Con ella se comprueba el
    pico proyectado antes de escribir nada y la ejecución vigila el margen de disco.
    `window` limita la ejecución a una ventana de campaña (`campaign_schedule`) y a las
    dependencias que tenga en ventanas anteriores. El resumen describe entonces esa ventana.
    """
    from .checkpoints import StopRequest

    require_learning_allowed("la campaña con máscaras")
    campaign = load_campaign(path)
    blockers = launch_blockers(campaign)
    _require(not blockers, "La campaña no se puede lanzar: " + "; ".join(blockers))
    jobs = plan_campaign(campaign)
    count_jobs(campaign, jobs)
    scope_of_run = {}
    if window is not None:
        jobs = campaign_schedule.window_jobs(campaign, jobs, window)
        scope_of_run = dict(window=window)
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
        reservation = (lease or _gpu_lease)() if uses_gpu else nullcontext()
        signals = StopRequest() if stop is None else nullcontext(stop)
        workers = campaign["tabular"]["cpu_workers"]
        pause = None

        def disk_report():
            if disk is None:
                return dict(scope_of_run)
            pending = {} if pause is None else dict(pause=pause)
            return dict(disk=dict(launch=launch, guard=disk[0].state(), **pending), **scope_of_run)

        _summary(output, identity, jobs, state.receipts, "running", **disk_report())
        try:
            with signals as requested, reservation, ThreadPoolExecutor(workers) as pool:
                state.stop = requested if disk is None else disk[0].watch(requested)
                status = _execute(state, jobs, pool, workers)
        except Paused as error:
            status = "paused"
            if isinstance(error, DiskPaused):
                pause = str(error)
            elif disk is not None and disk[0].low:
                pause = "El espacio libre bajó del margen durante un trabajo"
        except LearningHoldError as error:
            _summary(
                output, identity, jobs, state.receipts, "blocked", error=str(error), **disk_report()
            )
            raise
        except BaseException as error:
            _summary(
                output, identity, jobs, state.receipts, "failed", error=str(error), **disk_report()
            )
            raise
        return _summary(output, identity, jobs, state.receipts, status, **disk_report())
    finally:
        os.close(descriptor)


def _gpu_lease():
    from .experiment_resources import GpuLease

    return GpuLease()


def _execute(state, jobs, pool, workers):
    """Recorrer el plan en orden. CUDA de uno en uno y CPU con concurrencia acotada."""
    running = {}

    def collect(done):
        for future in done:
            job, run, identity = running.pop(future)
            state.record(job, state.confirm(job, run, identity, future.result()))

    for job in jobs:
        while any(dep not in state.receipts for dep in job["depends"]) and running:
            collect(wait(running, return_when=FIRST_COMPLETED).done)
        if state.stop.requested:
            raise Paused
        prepared, receipt = state.prepare(job)
        if receipt is not None:
            state.record(job, receipt)
            continue
        require_learning_allowed(f"el trabajo {job['id']}")
        state.admit(job)
        run, identity = prepared
        executor = state.executors[job["model"], job["kind"]]
        if executor["device"] == "cpu":
            while len(running) >= workers:
                collect(wait(running, return_when=FIRST_COMPLETED).done)
            running[pool.submit(executor["run"], run)] = (job, run, identity)
            continue
        state.record(job, state.confirm(job, run, identity, executor["run"](run)))
    while running:
        collect(wait(running, return_when=FIRST_COMPLETED).done)
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
    subconjunto de brazos permite evaluar los brazos ya producidos. Un brazo prestado del
    ámbito conjunto apunta a las predicciones del brazo conjunto en la ventana emparejada,
    que la comparación restringe a las filas del mercado del ámbito.
    """
    campaign, state = _confirmed_state(path, views, output)
    _require(scope in campaign["scopes"], "El ámbito no pertenece a la campaña")
    validation = comparison.load_config(
        Path(comparison_path or campaign["comparison_path"]).resolve()
    )
    _require(scope in validation["resolved_scopes"], "La comparación no declara el ámbito")
    resolved = validation["resolved_scopes"][scope]
    borrowed, joint = resolved["borrowed"], resolved.get("joint_scope")
    produced = {spec["arm"]: spec for spec in _arm_specs(campaign, scope)}
    wanted = {
        name: arm for name, arm in resolved["arms"].items() if arm["output"] != "zero_control"
    }
    missing = sorted(set(wanted) - set(produced) - set(borrowed))
    _require(
        not missing,
        f"Faltan productores para {', '.join(missing)}. Declara una comparación con los "
        "brazos disponibles o conecta su entrenador",
    )
    jobs = [job for job in plan_campaign(campaign) if job["scope"] == scope]
    origins = {}
    if borrowed:
        _require(joint in campaign["scopes"], f"La campaña no ajusta el ámbito {joint}")
        origins = {spec["arm"]: spec for spec in _arm_specs(campaign, joint)}
        _require(
            set(borrowed.values()) <= set(origins),
            f"Los brazos prestados no se ajustan en {joint}",
        )
        # Los trabajos conjuntos de los brazos prestados y de los padres de los que parten.
        closure, pending = set(), list(borrowed.values())
        while pending:
            arm = pending.pop()
            if arm not in closure:
                closure.add(arm)
                pending += [origins[arm]["parent"]] if origins[arm]["parent"] else []
        jobs = [
            job
            for job in plan_campaign(campaign)
            if job["scope"] == joint and job["arm"] in closure
        ] + jobs
    for job in jobs:
        case, _, sources = state.resolve(job)
        receipt = state.confirmed(job, state.job_identity(job, case, sources))
        _require(receipt is not None, f"Falta confirmar {job['id']} antes de publicar fuentes")
        state.receipts[job["id"]] = receipt
    windows = state.views[scope]["windows"]
    folder = Path(output) / "sources"
    arms, rows = {}, {}
    for name, arm in wanted.items():
        origin = borrowed.get(name)
        spec = origins[origin] if origin else produced[name]
        _require(
            sorted(arm["seeds"]) == sorted(spec["seeds"]),
            f"La comparación declara otras semillas para {name}",
        )
        arms[name] = {}
        for seed in arm["seeds"]:
            entries = arms[name][str(seed)] = {}
            for window in windows:
                if origin:
                    pair = resolved["joint_windows"][window]
                    key, receipt = state.selected(joint, pair, origin, seed)
                    view_sha256 = state.views[joint]["windows"][pair]["sha256"]
                else:
                    key, receipt = state.selected(scope, window, name, seed)
                    view_sha256 = windows[window]["sha256"]
                entry = dict(input_policy=campaign["input_policy"], view_sha256=view_sha256)
                for partition, record in receipt["predictions"].items():
                    # Las filas de un brazo prestado incluyen el otro mercado del conjunto.
                    if not origin:
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

    def window_entry(window, value):
        entry = dict(view=dict(path=value["path"], sha256=value["sha256"]))
        if borrowed:
            pair = state.views[joint]["windows"][resolved["joint_windows"][window]]
            entry["joint_view"] = dict(path=pair["path"], sha256=pair["sha256"])
        return entry

    manifest = dict(
        schema_version=1,
        kind=comparison.SOURCES_KIND,
        scope=scope,
        input_policy=campaign["input_policy"],
        windows={w: window_entry(w, v) for w, v in windows.items()},
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
    views = commands.add_parser("views", help="Validar vistas ya preparadas sin ajustar nada")
    for command in (check, prepare, execute, sources, views):
        command.add_argument("--campaign", type=Path, required=True)
    prepare.add_argument("--parent", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--scope", action="append", choices=tuple(comparison.SCOPES))
    for command in (execute, sources, views):
        command.add_argument("--views", action="append", required=True)
    for command in (execute, sources):
        command.add_argument("--output", type=Path, required=True)
    execute.add_argument(
        "--storage",
        type=Path,
        required=True,
        help="Declaración de almacenamiento con el margen de disco y los bytes medidos",
    )
    execute.add_argument("--window", help="Ventana de campaña que se ejecuta, con sus fases base")
    sources.add_argument("--scope", choices=tuple(comparison.SCOPES), required=True)
    sources.add_argument("--comparison", type=Path)
    args = parser.parse_args(argv)
    if args.command == "check":
        result = check_campaign(args.campaign)
    elif args.command == "prepare":
        prepared = prepare_views(args.campaign, args.parent, args.output, scopes=args.scope)
        result = {
            scope: dict(record, windows=len(record["windows"]))
            for scope, record in prepared.items()
        }
    elif args.command == "views":
        campaign = load_campaign(args.campaign)
        checked = {
            scope: scope_views(directory, scope, campaign)
            for scope, directory in _views_argument(args.views).items()
        }
        _require(set(checked) == set(campaign["scopes"]), "Faltan vistas de algún ámbito")
        result = dict(
            status="checked",
            views={scope: dict(r, windows=len(r["windows"])) for scope, r in checked.items()},
        )
    elif args.command == "run":
        result = run_campaign(
            args.campaign,
            _views_argument(args.views),
            args.output,
            storage=args.storage,
            window=args.window,
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
