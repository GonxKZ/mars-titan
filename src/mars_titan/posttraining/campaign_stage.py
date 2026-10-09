"""Etapa de postentrenamiento de la campaña con máscaras: la matriz por ventana walk-forward.

La etapa parte de una campaña base confirmada (`training.masked_campaign`). En cada
ventana reentrenada, el padre de un brazo neuronal y una semilla es el estado elegido
de ese brazo en esa ventana: el ganador de la búsqueda o el finalista de la semilla.
Los casos de la matriz se ajustan solo con el tramo de ajuste de la ventana, se
seleccionan con su validación, con el padre elegible en la época cero, y predicen
calibración y evaluación con el esquema común de la comparación y los cinco cuantiles.

Variante B: en una ventana intermedia no se ajusta nada. Cada caso aplica el estado
seleccionado en la ventana ancla, con el mismo padre del ancla que la campaña base
traslada a esa ventana, a la calibración y la evaluación de la ventana trasladada.

La declaración fija cómo se leen las cohortes de ajuste y validación de los brazos
neuronales. `view_blocks` las lee desde la vista por bloques con un presupuesto de memoria
y solo guarda el índice de cada ventana. `ordered_corpus` prepara la copia ordenada en
Parquet y declara si se retira al confirmar los ajustes de la ventana. Las dos lecturas
dan los mismos lotes.

Cada trabajo confirma un recibo con su identidad, huellas, filas y objetivos, que deben
coincidir con los de la campaña base en la misma ventana, y escribe el recibo
walk-forward de cada mercado. El padre congelado no genera trabajos: sus predicciones
son las de la campaña base. Los trabajos confirmados no se repiten y los pendientes se
reanudan desde su punto de control.
"""

import argparse
import fcntl
import gc
import hashlib
import json
import os
from collections import Counter
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.environments.walk_forward_receipt import (
    RECEIPT_KIND as WINDOW_RECEIPT_KIND,
)
from mars_titan.environments.walk_forward_receipt import read_window_receipt
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.models.quantile_head import MEDIAN_INDEX, QUANTILE_COLUMNS, QUANTILE_HEAD
from mars_titan.training import masked_campaign
from mars_titan.training.campaign_plan import (
    CARRY,
    DECLARED,
    FIT,
    load_campaign,
    plan_campaign,
    schedule,
)
from mars_titan.training.carried_predictions import carried_window
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.learning_hold import LearningHoldError, require_learning_allowed

from . import adapter_matrix
from .matrix_runs import (
    MatrixParent,
    MatrixWindow,
    index_manifest,
    predict_heldout,
    release_ordered,
)
from .parents import load_parent

STAGE_KIND = "historical_masked_posttraining_stage"
RUN_KIND = "historical_masked_posttraining_stage_run"
RECEIPT_KIND = "masked_posttraining_job"
KEEP, RELEASE = "keep", "release_after_window_fits"
ORDERED, BLOCKS = "ordered_corpus", "view_blocks"
# Memoria de las filas de un bloque, sin el proceso, el modelo ni las cachés del lector.
BLOCK_BYTES = (256 * 1024**2, 16 * 1024**3)
_FIELDS = {
    "schema_version",
    "kind",
    "status",
    "name",
    "campaign",
    "matrix",
    "scopes",
    "arms",
    "cohort_reading",
    "limits",
    "final_test_opened",
}
_LIMITS = {"max_training_jobs", "max_prediction_jobs"}
COMPARED = masked_campaign.COMPARED


class Paused(Exception):
    """Parada solicitada o ajuste pendiente en una barrera confirmada."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _names(values, allowed, label):
    _require(
        isinstance(values, list)
        and values
        and len(set(values)) == len(values)
        and all(value in allowed for value in values),
        f"{label} deben ser distintos y pertenecer a la campaña",
    )
    return values


def load_stage(path):
    """Validar la etapa, su campaña y su matriz sin leer datos."""
    path = Path(path)
    config, digest = read_manifest(path, 1024**2)
    _require(
        isinstance(config, dict)
        and set(config) == _FIELDS
        and config["schema_version"] == 1
        and config["kind"] == STAGE_KIND
        and config["status"] == DECLARED
        and config["final_test_opened"] is False
        and isinstance(config["name"], str)
        and comparison._name(config["name"].replace("-", "_")),
        "La etapa de postentrenamiento no cumple su contrato",
    )
    _reading(config["cohort_reading"])
    base = path.parent
    campaign = load_campaign((base / config["campaign"]).resolve())
    matrix_path = (base / config["matrix"]).resolve()
    matrix, matrix_sha256 = adapter_matrix.read_matrix(matrix_path)
    # Todos los brazos neuronales de la campaña emiten cuantiles y se ajustan con pinball.
    adapter_matrix.objectives(matrix, QUANTILE_HEAD)
    _require(
        matrix["input_policy"] == campaign["input_policy"]
        and matrix["budget"]["batch_size"] == campaign["neural"]["batch_size"],
        "La matriz debe compartir la política y el lote de la campaña",
    )
    scopes = _names(config["scopes"], campaign["scopes"], "Los ámbitos")
    _require(
        scopes == [scope for scope in campaign["scopes"] if scope in scopes],
        "Los ámbitos siguen el orden de la campaña",
    )
    neural = campaign["neural"]["arms"]
    arms = _names(config["arms"], neural, "Los brazos neuronales")
    declared = campaign["comparison_config"]["arms"]
    _require(
        all(sorted(declared[arm]["seeds"]) == sorted(matrix["budget"]["seeds"]) for arm in arms),
        "Cada semilla de la matriz parte del padre elegido con esa semilla",
    )
    limits = config["limits"]
    _require(
        isinstance(limits, dict)
        and set(limits) == _LIMITS
        and all(type(v) is int and 0 <= v <= 100_000 for v in limits.values()),
        "Los límites de trabajos deben ser enteros declarados",
    )
    return dict(
        config,
        sha256=digest,
        path=str(path.resolve()),
        campaign=campaign,
        matrix=matrix,
        matrix_path=str(matrix_path),
        matrix_sha256=matrix_sha256,
        families={arm: neural[arm] for arm in arms},
    )


def _reading(value):
    """Lectura de las cohortes: corpus ordenado con su retención o bloques de la vista."""
    ordered = (
        isinstance(value, dict)
        and value.keys() == {"source", "retention"}
        and value["source"] == ORDERED
        and value["retention"] in (KEEP, RELEASE)
    )
    blocks = (
        isinstance(value, dict)
        and value.keys() == {"source", "max_block_bytes"}
        and value["source"] == BLOCKS
        and type(value["max_block_bytes"]) is int
        and BLOCK_BYTES[0] <= value["max_block_bytes"] <= BLOCK_BYTES[1]
    )
    _require(ordered or blocks, "La lectura de cohortes de la etapa no es válida")
    return value


def arm_name(base_arm, point):
    """Nombre del brazo postentrenado, válido para la comparación walk-forward."""
    return f"{base_arm}__{point.replace('+', '_')}"


def plan_stage(stage):
    """Enumerar ajustes y traslados por ámbito, ventana, brazo base, semilla y caso."""
    campaign, matrix = stage["campaign"], stage["matrix"]
    jobs = []
    for scope in stage["scopes"]:
        folds = list(campaign["comparison_config"]["resolved_scopes"][scope]["windows"].values())
        for row in schedule(folds, campaign["period"]):
            for base_arm, family in stage["families"].items():
                cases = adapter_matrix.cases(
                    matrix, stage["matrix_sha256"], family, head=QUANTILE_HEAD
                )
                for item in cases:
                    seed, point = item["case"]["seed"], item["id"].split("/", 1)[1]
                    arm = arm_name(base_arm, point)
                    kind = FIT if row["trained"] else CARRY
                    jobs.append(
                        dict(
                            id=f"{scope}/{row['window']}/{arm}/{kind}-s{seed}",
                            scope=scope,
                            window=row["window"],
                            anchor=row["anchor"],
                            arm=arm,
                            base_arm=base_arm,
                            family=family,
                            point=point,
                            control=item["control"],
                            seed=seed,
                            kind=kind,
                            case=item["case"],
                            depends=[]
                            if row["trained"]
                            else [f"{scope}/{row['anchor']}/{arm}/{FIT}-s{seed}"],
                        )
                    )
    _require(len({job["id"] for job in jobs}) == len(jobs), "El plan contiene trabajos repetidos")
    return jobs


def count_stage(stage, jobs=None):
    """Contar ajustes y traslados por ámbito, brazo y semilla y aplicar los límites."""
    jobs = plan_stage(stage) if jobs is None else jobs
    scopes = {}
    for scope in stage["scopes"]:
        selected = [job for job in jobs if job["scope"] == scope]
        arms = {}
        for job in selected:
            entry = arms.setdefault(job["arm"], {}).setdefault(
                str(job["seed"]), dict(fit=0, carry=0)
            )
            entry[job["kind"]] += 1
        windows = {}
        for job in selected:
            windows[job["window"]] = job["kind"] == FIT
        scopes[scope] = dict(
            windows=len(windows),
            retrained_windows=[window for window, trained in windows.items() if trained],
            carried_windows=sum(not trained for trained in windows.values()),
            training_jobs=sum(job["kind"] == FIT for job in selected),
            prediction_jobs=sum(job["kind"] == CARRY for job in selected),
            arms=arms,
        )
    totals = dict(
        training_jobs=sum(job["kind"] == FIT for job in jobs),
        prediction_jobs=sum(job["kind"] == CARRY for job in jobs),
    )
    limits = stage["limits"]
    for kind, limit in (
        ("training_jobs", "max_training_jobs"),
        ("prediction_jobs", "max_prediction_jobs"),
    ):
        _require(
            totals[kind] <= limits[limit],
            f"La etapa prevé {totals[kind]} trabajos ({kind}) y supera el límite "
            f"declarado {limit}={limits[limit]}",
        )
    return dict(scopes=scopes, **totals)


def check_stage(path):
    """Validar, planificar y contar sin leer datos, reservar la GPU ni ajustar."""
    stage = load_stage(path)
    campaign = stage["campaign"]
    return dict(
        status="checked",
        name=stage["name"],
        variant=campaign["variant"],
        stage_sha256=stage["sha256"],
        campaign_sha256=campaign["sha256"],
        matrix_sha256=stage["matrix_sha256"],
        input_policy=campaign["input_policy"],
        objectives=adapter_matrix.objectives(stage["matrix"], QUANTILE_HEAD),
        excluded_controls=adapter_matrix.excluded_controls(stage["matrix"], QUANTILE_HEAD),
        cohort_reading=stage["cohort_reading"],
        counts=count_stage(stage),
        scientific_training_started=False,
        final_test_opened=False,
    )


def _code():
    root = Path(__file__).parents[1]
    names = (
        "posttraining/campaign_stage.py",
        "posttraining/matrix_runs.py",
        "environments/view_cohorts.py",
        "posttraining/adapter_matrix.py",
        "posttraining/run.py",
        "posttraining/heldout.py",
        "posttraining/evaluation.py",
        "posttraining/inputs.py",
        "posttraining/parents.py",
        "models/predictive_adaptation.py",
        "models/quantile_head.py",
        "training/masked_campaign.py",
        "training/campaign_plan.py",
        "training/carried_predictions.py",
        "environments/walk_forward_receipt.py",
        "evaluation/walk_forward_comparison.py",
    )
    return {name: sha256(root / name) for name in names}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _base_receipts(base, campaign, stage):
    """Confirmar los trabajos base de los ámbitos y brazos de la etapa, en orden del plan."""
    for job in plan_campaign(campaign):
        if job["scope"] not in stage["scopes"] or job["arm"] not in stage["families"]:
            continue
        case, _, sources = base.resolve(job)
        receipt = base.confirmed(job, base.job_identity(job, case, sources))
        _require(receipt is not None, f"Falta confirmar {job['id']} en la campaña base")
        base.receipts[job["id"]] = receipt


def _ordered_quantiles(table, label):
    """Exigir cinco niveles no decrecientes y la mediana como predicción, con los mismos bits."""
    levels = np.stack([table[name].to_numpy() for name in QUANTILE_COLUMNS], axis=1)
    _require(
        np.isfinite(levels).all() and (np.diff(levels, axis=1) >= 0).all(),
        f"{label}: los cuantiles no son finitos o no conservan su orden",
    )
    _require(
        np.array_equal(table["prediction"].to_numpy(), levels[:, MEDIAN_INDEX]),
        f"{label}: la predicción no es la mediana de la cabeza",
    )


class _Stage:
    """Estado confirmado de la etapa: ventanas, padres, recibos y recibos walk-forward."""

    def __init__(self, stage, base, campaign_output, output, identity, *, device, lease, stop):
        self.stage, self.base, self.output = stage, base, output
        self.campaign = stage["campaign"]
        self.campaign_output, self.identity = campaign_output, identity
        self.identity_sha256 = _digest(identity)
        self.device, self.lease, self.stop = device, lease, stop
        self.policy = self.campaign["input_policy"]
        self.batch_size = stage["matrix"]["budget"]["batch_size"]
        self.receipts, self.budgets, self.released = {}, {}, {}
        self.window_key = self.parent_key = self.dataset_key = None
        self.window = self.parent = self.dataset = None

    # Contextos: una ventana, un padre y una vista abiertos como máximo.
    def close_parent(self):
        if self.parent is not None:
            self.parent.close()
        self.parent_key = self.parent = None
        gc.collect()

    def close_window(self):
        self.close_parent()
        if self.window is not None:
            self.window.close()
        self.window_key = self.window = None

    def close(self):
        self.close_window()
        self.dataset_key = self.dataset = None

    def view(self, scope, window):
        return self.base.views[scope]["windows"][window]

    def window_folder(self, scope, window):
        return self.output / "windows-data" / scope / window

    def open_window(self, scope, window):
        if self.window_key != (scope, window):
            self.close_window()
            view = self.view(scope, window)
            reading = self.stage["cohort_reading"]
            self.window = MatrixWindow(
                view["path"],
                self.window_folder(scope, window),
                encoding=view["sha256"],
                input_policy=self.policy,
                batch_size=self.batch_size,
                stop=self.stop,
                max_block_bytes=reading.get("max_block_bytes"),
            )
            self.window_key = (scope, window)
            _require(
                self.window.source_sha256 == view["sha256"],
                "El corpus ordenado no corresponde a la vista de la ventana",
            )
        return self.window

    def open_dataset(self, scope, window):
        if self.dataset_key != (scope, window):
            self.dataset = CorpusDataset(
                Path(self.view(scope, window)["path"]), input_policy=self.policy
            )
            self.dataset_key = (scope, window)
        return self.dataset

    def base_parent(self, scope, window, base_arm, seed):
        """Trabajo base elegido para la semilla en la ventana y su informe verificado."""
        key, receipt = self.base.selected(scope, window, base_arm, seed)
        report = self.campaign_output / receipt["report"]["path"]
        _require(
            sha256(report) == receipt["report"]["sha256"],
            f"El informe del padre {key} ha cambiado",
        )
        return key, receipt, report

    def open_parent(self, job):
        key = (job["scope"], job["window"], job["base_arm"], job["seed"])
        if self.parent_key != key:
            self.close_parent()
            window = self.open_window(job["scope"], job["window"])
            _, _, report = self.base_parent(*key)
            folder = self.window_folder(job["scope"], job["window"])
            self.parent = MatrixParent(
                window,
                report,
                folder / "parents" / job["base_arm"] / f"seed-{job['seed']}",
                matrix=self.stage["matrix"],
                digest=self.stage["matrix_sha256"],
                seed=job["seed"],
                device=self.device,
                lease=self.lease,
                stop=self.stop,
            )
            self.parent_key = key
            self.budgets["/".join(map(str, key))] = self.parent.budget
        return self.parent

    def job_identity(self, job):
        window = job["window"] if job["kind"] == FIT else job["anchor"]
        key, receipt, _ = self.base_parent(job["scope"], window, job["base_arm"], job["seed"])
        identity = dict(
            stage_identity_sha256=self.identity_sha256,
            **{
                name: job[name]
                for name in (
                    "id",
                    "scope",
                    "window",
                    "anchor",
                    "arm",
                    "base_arm",
                    "family",
                    "point",
                    "control",
                    "seed",
                    "kind",
                    "case",
                )
            },
            view_sha256=self.view(job["scope"], job["window"])["sha256"],
            parent=dict(
                job=key,
                receipt_sha256=receipt["sha256"],
                checkpoint_sha256=receipt["parent"]["sha256"],
            ),
            anchor_fit=None,
        )
        if job["kind"] == CARRY:
            (anchor,) = job["depends"]
            _require(anchor in self.receipts, f"{job['id']} depende de {anchor}, sin confirmar")
            identity["anchor_fit"] = dict(
                job=anchor, receipt_sha256=self.receipts[anchor]["sha256"]
            )
        return identity

    def folder(self, job):
        return self.output / "jobs" / job["id"]

    def confirmed(self, job, identity):
        path = self.folder(job) / "receipt.json"
        if not path.is_file():
            return None
        receipt, digest = read_manifest(path, 8 * 1024**2)
        _require(receipt.get("identity") == identity, f"{job['id']} cambió de identidad")
        for record in [receipt["run"], *receipt["predictions"].values()]:
            _require(
                sha256(self.output / record["path"]) == record["sha256"],
                f"Un artefacto confirmado de {job['id']} ha cambiado",
            )
        return dict(receipt, sha256=digest)

    def fit(self, job, folder):
        """Ajustar o recuperar el caso con el padre de la ventana y predecir los tramos."""
        parent = self.open_parent(job)
        rows = [row for row in parent.rows if row["id"] == f"seed-{job['seed']}/{job['point']}"]
        _require(
            len(rows) == 1 and rows[0]["case"] == job["case"],
            f"{job['id']} no corresponde al plan de su padre",
        )
        row = rows[0]
        report = parent.run(
            row,
            folder / "run",
            checkpoint_seconds=self.campaign["neural"]["checkpoint_seconds"],
            stop=self.stop,
        )
        if report["status"] != "completed":
            raise Paused
        run_path = folder / "run" / "run.json"
        predictions = predict_heldout(
            parent.parent,
            run_path,
            report,
            self.open_dataset(job["scope"], job["window"]),
            folder,
            device=self.device,
            batch_size=self.batch_size,
            stop=self.stop,
        )
        return dict(
            run=run_path,
            predictions=predictions,
            parent=dict(id=job["id"], sha256=report["checkpoint"]["sha256"]),
            score=report["predictions"]["validation"]["metrics"]["session_mae"],
            updates=report["global_step"],
            selection=report["selection"],
        )

    def carry(self, job, folder):
        """Aplicar sin ajuste el estado del ancla, con el padre del ancla, a otra ventana."""
        self.close_window()
        scope, anchor = job["scope"], job["anchor"]
        (dependency,) = job["depends"]
        fitted = self.receipts[dependency]
        _, anchor_receipt, anchor_report = self.base_parent(
            scope, anchor, job["base_arm"], job["seed"]
        )
        # La campaña base traslada a esta ventana el mismo padre elegido en el ancla.
        _, carried, _ = self.base_parent(scope, job["window"], job["base_arm"], job["seed"])
        _require(
            carried["parent"] == anchor_receipt["parent"],
            f"{job['id']}: la campaña base no traslada el padre del ancla",
        )
        anchor_view, view = (
            read_manifest(Path(self.view(scope, name)["path"]), 8 * 1024**2)[0]
            for name in (anchor, job["window"])
        )
        carried_window(anchor_view, view, input_policy=self.policy)
        data = self.window_folder(scope, anchor)
        population = (
            index_manifest(data)
            if self.stage["cohort_reading"]["source"] == BLOCKS
            else data / "ordered" / "manifest.json"
        )
        diagnostic = self.device == "cpu"
        parent = load_parent(
            population, anchor_report, device=self.device, diagnostic=diagnostic, lease=self.lease
        )
        run_path = self.output / fitted["run"]["path"]
        report, digest = read_manifest(run_path, 8 * 1024**2)
        _require(digest == fitted["run"]["sha256"], f"El ajuste del ancla de {job['id']} cambió")
        predictions = predict_heldout(
            parent,
            run_path,
            report,
            self.open_dataset(scope, job["window"]),
            folder,
            device=self.device,
            batch_size=self.batch_size,
            stop=self.stop,
        )
        return dict(
            run=run_path,
            predictions=predictions,
            parent=dict(fitted["parent"]),
            score=None,
            updates=0,
            selection=report["selection"],
        )

    def confirm(self, job, identity, folder, result):
        """Comprobar tramos, filas, objetivos y cuantiles, y escribir el recibo del trabajo."""
        resolved = self.campaign["comparison_config"]["resolved_scopes"][job["scope"]]
        window = resolved["windows"][job["window"]]
        view = self.view(job["scope"], job["window"])
        columns = comparison.COLUMNS + QUANTILE_COLUMNS
        predictions = {}
        for partition in COMPARED:
            record = result["predictions"][partition]
            path = folder / record["path"]
            safe_destination(path)
            label = f"{job['id']} ({partition})"
            table = comparison._read_predictions(dict(path=path, sha256=record["sha256"]), columns)
            comparison._check_segment(table, window, partition, resolved["markets"], label)
            _require(
                table.num_rows == view["counts"][partition],
                f"{label}: {table.num_rows} filas frente a {view['counts'][partition]} de la vista",
            )
            _ordered_quantiles(table, label)
            rows = masked_campaign._rows_digest(table)
            source, expected = self.base.rows[(job["scope"], job["window"], partition)]
            _require(
                rows == expected,
                f"{label}: no evalúa las mismas filas ni objetivos que {source} de la campaña base",
            )
            predictions[partition] = dict(
                path=str(path.relative_to(self.output)),
                sha256=record["sha256"],
                rows=table.num_rows,
                rows_sha256=rows,
                markets=masked_campaign._fingerprints(table, resolved["markets"]),
            )
        run_path = result["run"]
        receipt = dict(
            schema_version=1,
            kind=RECEIPT_KIND,
            status="completed",
            identity=identity,
            run=dict(path=str(run_path.relative_to(self.output)), sha256=sha256(run_path)),
            parent=result["parent"],
            score=result["score"],
            updates=result["updates"],
            selection=result["selection"],
            predictions=predictions,
            final_test_opened=False,
            confirmed_at_utc=datetime.now(UTC).isoformat(),
        )
        path = self.folder(job) / "receipt.json"
        atomic_json(path, receipt)
        return dict(receipt, sha256=sha256(path))

    def publish(self, job, receipt):
        """Recibo walk-forward por mercado del brazo postentrenado, con el contrato de #390."""
        resolved = self.campaign["comparison_config"]["resolved_scopes"][job["scope"]]
        folder = (
            self.output
            / "windows"
            / job["scope"]
            / job["window"]
            / job["arm"]
            / f"seed-{job['seed']}"
        )
        for market in resolved["markets"]:
            record = dict(
                kind=WINDOW_RECEIPT_KIND,
                schema_version=1,
                protocol=resolved["protocols"][market],
                fold=resolved["windows"][job["window"]],
                parent=receipt["parent"],
                labels_used_until=0,
                predictions={
                    partition: value["markets"][market]
                    for partition, value in receipt["predictions"].items()
                    if market in value["markets"]
                },
            )
            # Ajuste, selección y calibración común usan etiquetas maduras antes de evaluar.
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

    def release(self, jobs, job):
        """Retirar la copia ordenada de una ventana cuando todos sus ajustes están confirmados."""
        key = f"{job['scope']}/{job['window']}"
        if self.stage["cohort_reading"].get("retention") != RELEASE or key in self.released:
            return
        pending = [
            item["id"]
            for item in jobs
            if item["kind"] == FIT
            and (item["scope"], item["window"]) == (job["scope"], job["window"])
            and item["id"] not in self.receipts
        ]
        if pending:
            return
        self.close_window()
        manifest = self.window_folder(job["scope"], job["window"]) / "ordered" / "manifest.json"
        if manifest.is_file():
            self.released[key] = release_ordered(manifest)

    def execute(self, jobs):
        """Recorrer el plan en orden y confirmar cada trabajo."""
        for job in jobs:
            if self.stop.requested:
                raise Paused
            identity = self.job_identity(job)
            receipt = self.confirmed(job, identity)
            if receipt is None:
                require_learning_allowed(f"el trabajo {job['id']}")
                folder = self.folder(job)
                safe_destination(folder)
                folder.mkdir(parents=True, exist_ok=True)
                try:
                    result = (self.fit if job["kind"] == FIT else self.carry)(job, folder)
                except InterruptedError as error:
                    raise Paused from error
                receipt = self.confirm(job, identity, folder, result)
            self.receipts[job["id"]] = receipt
            self.publish(job, receipt)
            if job["kind"] == FIT:
                self.release(jobs, job)
        return "completed"


def _identity(stage, views):
    campaign = stage["campaign"]
    return dict(
        schema_version=1,
        kind=RUN_KIND,
        stage_sha256=stage["sha256"],
        campaign_sha256=campaign["sha256"],
        matrix_sha256=stage["matrix_sha256"],
        input_policy=campaign["input_policy"],
        variant=campaign["variant"],
        views={
            scope: {window: value["sha256"] for window, value in record["windows"].items()}
            for scope, record in views.items()
        },
        code=_code(),
        final_test_opened=False,
    )


def _summary(output, identity, jobs, state, status, **extra):
    planned = Counter(job["kind"] for job in jobs)
    done = Counter(job["kind"] for job in jobs if job["id"] in state.receipts)
    summary = dict(
        schema_version=1,
        kind=RUN_KIND,
        status=status,
        identity_sha256=_digest(identity),
        planned=dict(training_jobs=planned[FIT], prediction_jobs=planned[CARRY]),
        completed=dict(training_jobs=done[FIT], prediction_jobs=done[CARRY]),
        updates={job_id: receipt["updates"] for job_id, receipt in state.receipts.items()},
        budgets=state.budgets,
        released_ordered_copies=state.released,
        jobs={job["id"]: job["id"] in state.receipts for job in jobs},
        final_test_opened=False,
        updated_at_utc=datetime.now(UTC).isoformat(),
        **extra,
    )
    atomic_json(output / "summary.json", summary)
    return summary


def _gpu_lease():
    from mars_titan.training.experiment_resources import GpuLease

    return GpuLease()


def run_stage(path, views, campaign_output, output, *, lease=None, stop=None, device="cuda:0"):
    """Ejecutar o reanudar la etapa sobre una campaña base confirmada.

    `lease` sustituye la reserva de la GPU y `device="cpu"` limita la ejecución a los
    diagnósticos de hasta 5000 filas de `run_case`. La protección del aprendizaje se
    comprueba antes de abrir fuentes y antes de cada trabajo pendiente.
    """
    from mars_titan.training.checkpoints import StopRequest

    require_learning_allowed("la etapa de postentrenamiento de la campaña")
    stage = load_stage(path)
    jobs = plan_stage(stage)
    count_stage(stage, jobs)
    campaign = stage["campaign"]
    _require(device in ("cpu", "cuda:0"), "El dispositivo debe ser cpu o cuda:0")
    _require(
        isinstance(views, dict) and set(views) == set(campaign["scopes"]),
        "Se necesitan las vistas de todos los ámbitos de la campaña base",
    )
    views = {scope: Path(value) for scope, value in views.items()}
    campaign_output, output = Path(campaign_output), Path(output)
    safe_destination(output)
    for protected in (*views.values(), campaign_output, Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)
    _, base = masked_campaign._confirmed_state(campaign["path"], views, campaign_output)
    _base_receipts(base, campaign, stage)
    identity = _identity(stage, base.views)
    output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = output / "stage.json"
        if marker.exists():
            _require(
                read_manifest(marker, 8 * 1024**2)[0] == identity,
                "La salida pertenece a otra etapa, campaña, matriz, vista o código",
            )
        else:
            _require(
                all(p.name in {".lock", "summary.json"} for p in output.iterdir()),
                "La salida sin identidad contiene artefactos ajenos",
            )
            atomic_json(marker, identity)
        signals = StopRequest() if stop is None else nullcontext(stop)
        reservation = (lease or _gpu_lease)()
        state = _Stage(
            stage, base, campaign_output, output, identity, device=device, lease=None, stop=None
        )
        _summary(output, identity, jobs, state, "running")
        try:
            with signals as state.stop, reservation as state.lease:
                try:
                    status = state.execute(jobs)
                finally:
                    state.close()
        except Paused:
            status = "paused"
        except LearningHoldError as error:
            _summary(output, identity, jobs, state, "blocked", error=str(error))
            raise
        except BaseException as error:
            _summary(output, identity, jobs, state, "failed", error=str(error))
            raise
        return _summary(output, identity, jobs, state, status)
    finally:
        os.close(descriptor)


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
    execute = commands.add_parser("run", help="Ejecutar o reanudar la etapa")
    for command in (check, execute):
        command.add_argument("--stage", type=Path, required=True)
    execute.add_argument("--views", action="append", required=True)
    execute.add_argument("--campaign-output", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        result = check_stage(args.stage)
    else:
        result = run_stage(
            args.stage, _views_argument(args.views), args.campaign_output, args.output
        )
        result.pop("jobs")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"completed", "checked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
