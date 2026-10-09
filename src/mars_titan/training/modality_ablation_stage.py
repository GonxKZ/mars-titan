"""Etapa de ablación de modalidades: volver a predecir la evaluación sin reentrenar.

La etapa parte de una campaña base confirmada (`training.masked_campaign`). Para cada
ámbito, ventana, brazo comparado que la campaña produce, semilla y variante declarada en
la sección `modality_ablation` de la comparación, carga el estado elegido de ese brazo y
semilla y predice la evaluación de la ventana con la variante aplicada en la lectura
(`data.modality_ablation`). En una ventana reentrenada el estado es el elegido en ella y en
una ventana trasladada de la variante B, el del ancla que la campaña base traslada. Se
reutiliza el traslado de cada familia con `modality_ablation`, así que no se ajustan pesos,
normalizadores, selección ni calibración.

Titans-MAC, MARS-TITAN, CM-v1 y la GRU candidata tienen estado en línea. Su predicción
enmascarada recorre el calentamiento y el tramo con las mismas entradas ablacionadas, de
modo que la memoria rápida y el banco episódico también ven la ausencia. Cada tramo empieza
con la memoria inicial, igual que la predicción original.

No ajusta nada, pero es una evaluación científica: la protección del aprendizaje vigente
la retiene antes de crear salidas y antes de cada trabajo pendiente. Durante la ejecución,
un gancho global rechaza cualquier paso de un optimizador de PyTorch antes de modificar
pesos. Cada trabajo confirma un recibo con su identidad, el estado de partida, las huellas
y las filas, que deben ser las de la campaña base en la misma ventana. `sources` publica el
manifiesto que lee `evaluation.walk_forward_comparison` con `--ablation-sources`.
"""

import argparse
import fcntl
import hashlib
import json
import os
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data import prediction_files
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.modality_ablation import VARIANTS, ablation_identity
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation import modality_ablation as analysis
from mars_titan.evaluation import walk_forward_comparison as comparison

from . import masked_campaign
from .campaign_plan import (
    DECLARED,
    NEURAL,
    QUANTILE_HEAD,
    _arm_specs,
    arm_output,
    load_campaign,
    plan_campaign,
    schedule,
)
from .campaign_storage import release_confirmed
from .learning_hold import LearningHoldError, hold_path, learning_blocked

STAGE_KIND = "historical_masked_modality_ablation_stage"
RUN_KIND = "historical_masked_modality_ablation_stage_run"
RECEIPT_KIND = "masked_modality_ablation_job"
PARTITION = "evaluation"
MAX_ATTEMPTS = 32
_FIELDS = {
    "schema_version",
    "kind",
    "status",
    "name",
    "campaign",
    "scopes",
    "limits",
    "final_test_opened",
}


class Paused(Exception):
    """Parada solicitada en una barrera confirmada de un trabajo."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def require_evaluation_allowed(action):
    """La etapa no ajusta nada, pero la protección vigente también retiene las evaluaciones."""
    if learning_blocked():
        raise LearningHoldError(
            f"Bloqueo vigente: {action} es una evaluación científica y no se ejecuta mientras "
            f"{hold_path()} no declare training_allowed verdadero"
        )


@contextmanager
def forbid_optimizer_steps():
    """Rechazar cualquier paso de un optimizador de PyTorch mientras dura la etapa."""
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    def forbid(optimizer, args, kwargs):
        raise RuntimeError(
            f"La ablación de modalidades no admite pasos de {type(optimizer).__name__}"
        )

    handle = register_optimizer_step_pre_hook(forbid)
    try:
        yield
    finally:
        handle.remove()


def load_stage(path):
    """Validar la etapa, su campaña y la declaración de la ablación sin leer datos."""
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
        "La etapa de ablación no cumple su contrato",
    )
    campaign = load_campaign((path.parent / config["campaign"]).resolve())
    declared = campaign["comparison_config"]
    _require(
        comparison.ABLATION_FIELD in declared,
        "La comparación de la campaña no declara la ablación de modalidades",
    )
    scopes = config["scopes"]
    _require(
        isinstance(scopes, list)
        and scopes
        and len(set(scopes)) == len(scopes)
        and scopes == [scope for scope in campaign["scopes"] if scope in scopes],
        "Los ámbitos deben ser de la campaña y seguir su orden",
    )
    limits = config["limits"]
    _require(
        isinstance(limits, dict)
        and set(limits) == {"max_prediction_jobs"}
        and type(limits["max_prediction_jobs"]) is int
        and 0 <= limits["max_prediction_jobs"] <= 1_000_000,
        "El límite de predicciones debe ser un entero declarado",
    )
    # Brazos comparados con productor en la campaña. Los auxiliares no se comparan.
    specs = [spec for spec in _arm_specs(campaign) if not spec["helper"]]
    return dict(
        config,
        sha256=digest,
        path=str(path.resolve()),
        campaign=campaign,
        declaration=declared[comparison.ABLATION_FIELD],
        specs=specs,
    )


def plan_stage(stage):
    """Una predicción por ámbito, ventana, brazo, semilla y variante, en orden del plan."""
    campaign, jobs = stage["campaign"], []
    for scope in stage["scopes"]:
        folds = list(campaign["comparison_config"]["resolved_scopes"][scope]["windows"].values())
        for row in schedule(folds, campaign["period"]):
            for spec in stage["specs"]:
                for seed in spec["seeds"]:
                    for variant in VARIANTS:
                        jobs.append(
                            dict(
                                id=f"{scope}/{row['window']}/{spec['arm']}/{variant}-s{seed}",
                                scope=scope,
                                window=row["window"],
                                anchor=row["anchor"],
                                arm=spec["arm"],
                                family=spec["family"],
                                model=spec["model"],
                                seed=seed,
                                variant=variant,
                            )
                        )
    _require(len({job["id"] for job in jobs}) == len(jobs), "El plan contiene trabajos repetidos")
    return jobs


def count_stage(stage, jobs=None):
    """Contar predicciones por ámbito, brazo, semilla y variante y aplicar el límite."""
    jobs = plan_stage(stage) if jobs is None else jobs
    scopes = {}
    for scope in stage["scopes"]:
        selected = [job for job in jobs if job["scope"] == scope]
        arms = {}
        for job in selected:
            seeds = arms.setdefault(job["arm"], {}).setdefault(str(job["seed"]), {})
            seeds[job["variant"]] = seeds.get(job["variant"], 0) + 1
        scopes[scope] = dict(
            windows=len({job["window"] for job in selected}),
            prediction_jobs=len(selected),
            arms=arms,
        )
    limit = stage["limits"]["max_prediction_jobs"]
    _require(
        len(jobs) <= limit,
        f"La etapa prevé {len(jobs)} predicciones y supera el límite declarado {limit}",
    )
    return dict(
        scopes=scopes,
        training_jobs=0,
        prediction_jobs=len(jobs),
        variants=dict(Counter(job["variant"] for job in jobs)),
    )


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
        comparison_sha256=campaign["comparison_config"]["sha256"],
        input_policy=campaign["input_policy"],
        declaration=stage["declaration"],
        arms={spec["arm"]: spec["seeds"] for spec in stage["specs"]},
        counts=count_stage(stage),
        optimizer_steps=0,
        scientific_training_started=False,
        final_test_opened=False,
    )


@dataclass(frozen=True)
class AblationRun:
    """Lo que necesita un ejecutor: trabajo, vista, destino, estado de partida y variante."""

    job: dict
    view: Path
    view_sha256: str
    folder: Path
    anchor: dict
    policy: str
    batch_size: int
    stop: object

    @property
    def variant(self):
        return self.job["variant"]


def _neural(run):
    from .carried_predictions import carry_reference
    from .reference_run import _Pause

    try:
        return carry_reference(
            run.anchor["folder"],
            run.anchor["view"],
            run.view,
            run.folder,
            batch_size=run.batch_size,
            input_policy=run.policy,
            stop=run.stop,
            modality_ablation=run.variant,
        )
    except _Pause as error:
        raise Paused from error


def _tabular(run):
    from .carried_predictions import carry_tabular

    return carry_tabular(
        run.anchor["folder"],
        run.anchor["view"],
        run.view,
        run.folder,
        kind=run.job["model"],
        batch_size=run.batch_size,
        input_policy=run.policy,
        modality_ablation=run.variant,
    )


def _episodic(run):
    from .candidate_walk_forward import carry_window

    report = carry_window(
        run.anchor["folder"],
        run.anchor["view"],
        run.view,
        run.folder,
        parent_id=run.anchor["job"],
        device="cuda:0",
        stop=run.stop,
        modality_ablation=run.variant,
    )
    if report["status"] == "paused":
        raise Paused
    return report


def _chronological(carry):
    """Ejecutor de una familia cronológica, con fastpath=False solo durante el trabajo."""

    def run_carry(run):
        from .financial_run import Paused as ChronologicalPaused
        from .titans_walk_forward import unfused_attention

        try:
            with unfused_attention():
                return carry()(
                    run.anchor["folder"],
                    run.anchor["view"],
                    run.view,
                    run.folder,
                    device="cuda:0",
                    stop=run.stop,
                    modality_ablation=run.variant,
                )
        except ChronologicalPaused as error:
            raise Paused from error

    return run_carry


def _titans():
    from .titans_walk_forward import carry_titans

    return carry_titans


def _mars_titan():
    from .mars_titan_walk_forward import carry_mars_titan

    return carry_mars_titan


def _cm_v1():
    from .cm_v1_factorial import carry_cm_v1

    return carry_cm_v1


# Ejecutor por modelo de la campaña. Todos reutilizan el traslado de su familia.
EXECUTORS = {
    "neural": _neural,
    "ridge": _tabular,
    "xgboost": _tabular,
    "episodic_gru": _episodic,
    "titans_mac": _chronological(_titans),
    "mars_titan": _chronological(_mars_titan),
    "cm_v1": _chronological(_cm_v1),
}


def _code():
    root = Path(__file__).parents[1]
    names = (
        "training/modality_ablation_stage.py",
        "data/modality_ablation.py",
        "evaluation/modality_ablation.py",
        "training/corpus_inputs.py",
        "training/carried_predictions.py",
        "training/masked_campaign.py",
        "training/campaign_plan.py",
    )
    return {name: sha256(root / name) for name in names}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _base_receipts(base, campaign, stage, pairs=None):
    """Confirmar los trabajos base de los ámbitos de la etapa, en el orden del plan.

    `pairs` limita la confirmación a esos pares (ámbito, ventana) al ejecutar una ventana.
    """
    for job in plan_campaign(campaign):
        if job["scope"] not in stage["scopes"]:
            continue
        if pairs is not None and (job["scope"], job["window"]) not in pairs:
            continue
        case, _, sources = base.resolve(job)
        receipt = base.confirmed(job, base.job_identity(job, case, sources))
        _require(receipt is not None, f"Falta confirmar {job['id']} en la campaña base")
        base.receipts[job["id"]] = receipt


class _Stage:
    """Estado confirmado de la etapa: recibos y comprobación de cada predicción."""

    def __init__(self, stage, base, output, identity, executors, stop):
        self.stage, self.base, self.output = stage, base, output
        self.campaign = stage["campaign"]
        self.identity, self.identity_sha256 = identity, _digest(identity)
        self.executors, self.stop = executors, stop
        self.receipts = {}

    def folder(self, job):
        return self.output / "jobs" / job["id"]

    def view(self, scope, window):
        return self.base.views[scope]["windows"][window]

    def job_identity(self, job):
        """Estado de partida: el elegido en el ancla, que la campaña usa en esta ventana."""
        key, receipt = self.base.selected(job["scope"], job["anchor"], job["arm"], job["seed"])
        used, window = self.base.selected(job["scope"], job["window"], job["arm"], job["seed"])
        _require(
            window["parent"] == receipt["parent"],
            f"{job['id']}: la campaña base no predice esta ventana con el estado del ancla",
        )
        return dict(
            stage_identity_sha256=self.identity_sha256,
            **{
                name: job[name]
                for name in (
                    "id",
                    "scope",
                    "window",
                    "anchor",
                    "arm",
                    "family",
                    "model",
                    "seed",
                    "variant",
                )
            },
            modality_ablation=ablation_identity(job["variant"]),
            view_sha256=self.view(job["scope"], job["window"])["sha256"],
            source=dict(job=key, receipt_sha256=receipt["sha256"]),
            window_source=dict(job=used, receipt_sha256=window["sha256"]),
            parent=dict(receipt["parent"]),
        )

    def confirmed(self, job, identity):
        path = self.folder(job) / "receipt.json"
        if not path.is_file():
            return None
        receipt, digest = read_manifest(path, 8 * 1024**2)
        _require(receipt.get("identity") == identity, f"{job['id']} cambió de identidad")
        _require(
            sha256(self.output / receipt["report"]["path"]) == receipt["report"]["sha256"],
            f"Un artefacto confirmado de {job['id']} ha cambiado",
        )
        record = receipt["prediction"]
        prediction_files.verify(
            self.output / record["path"],
            record["sha256"],
            label=f"Un artefacto confirmado de {job['id']} ha cambiado",
        )
        return dict(receipt, sha256=digest)

    def attempt(self, job):
        folder = self.folder(job)
        safe_destination(folder)
        attempts = sorted(p.name for p in folder.glob("attempt-*")) if folder.exists() else []
        _require(len(attempts) < MAX_ATTEMPTS, f"{job['id']} alcanzó el límite de intentos")
        return folder / f"attempt-{len(attempts) + 1:04d}"

    def prepare(self, job, identity):
        view = self.view(job["scope"], job["window"])
        source = self.base.receipts[identity["source"]["job"]]
        section = self.campaign["neural" if job["family"] == NEURAL else "tabular"]
        return AblationRun(
            job=job,
            view=Path(view["path"]),
            view_sha256=view["sha256"],
            folder=self.attempt(job),
            anchor=dict(
                folder=self.base.output / source["attempt"],
                view=Path(self.view(job["scope"], job["anchor"])["path"]),
                job=identity["source"]["job"],
            ),
            policy=self.campaign["input_policy"],
            batch_size=section["batch_size"],
            stop=self.stop,
        )

    def confirm(self, job, identity, run, report):
        """Comprobar la ablación, el estado de partida, el tramo y las filas, y confirmar."""
        resolved = self.campaign["comparison_config"]["resolved_scopes"][job["scope"]]
        label = f"{job['id']} ({PARTITION})"
        _require(report.get("final_test_opened") is False, f"{job['id']} abre la reserva final")
        _require(
            report.get("modality_ablation") == identity["modality_ablation"],
            f"{job['id']} no declara la ablación pedida",
        )
        _require(
            report.get("anchor", {}).get("checkpoint_sha256") == identity["parent"]["sha256"],
            f"{job['id']} no parte del estado elegido en {identity['source']['job']}",
        )
        _require(
            set(report.get("predictions", {})) == {PARTITION},
            f"{job['id']} solo debe predecir la evaluación",
        )
        record = report["predictions"][PARTITION]
        path = run.folder / record["path"]
        safe_destination(path)
        _require(sha256(path) == record["sha256"], f"{job['id']} cambió su evaluación")
        quantile = arm_output(self.campaign, job["arm"]) == QUANTILE_HEAD
        columns = comparison.COLUMNS + (comparison.QUANTILE_COLUMNS if quantile else ())
        table = comparison._read_predictions(dict(path=path, sha256=record["sha256"]), columns)
        window = resolved["windows"][job["window"]]
        comparison._check_segment(table, window, PARTITION, resolved["markets"], label)
        expected = self.view(job["scope"], job["window"])["counts"][PARTITION]
        _require(table.num_rows == expected, f"{label}: {table.num_rows} filas frente a {expected}")
        rows = masked_campaign._rows_digest(table)
        source, digest = self.base.rows[(job["scope"], job["window"], PARTITION)]
        _require(
            rows == digest,
            f"{label}: no evalúa las mismas filas ni objetivos que {source} de la campaña base",
        )
        report_path = next(
            run.folder / name for name in ("carry.json",) if (run.folder / name).is_file()
        )
        receipt = dict(
            schema_version=1,
            kind=RECEIPT_KIND,
            status="completed",
            identity=identity,
            attempt=str(run.folder.relative_to(self.output)),
            report=dict(path=str(report_path.relative_to(self.output)), sha256=sha256(report_path)),
            prediction=dict(
                path=str(path.relative_to(self.output)),
                sha256=record["sha256"],
                rows=table.num_rows,
                rows_sha256=rows,
            ),
            optimizer_steps=0,
            final_test_opened=False,
            confirmed_at_utc=datetime.now(UTC).isoformat(),
        )
        target = self.folder(job) / "receipt.json"
        atomic_json(target, receipt)
        # Con el recibo escrito, los índices del calentamiento ya no se leen.
        release_confirmed(run.folder, job["model"])
        return dict(receipt, sha256=sha256(target))

    def execute(self, jobs):
        """Recorrer el plan en orden y confirmar cada predicción pendiente."""
        for job in jobs:
            if self.stop.requested:
                raise Paused
            identity = self.job_identity(job)
            receipt = self.confirmed(job, identity)
            if receipt is None:
                require_evaluation_allowed(f"la ablación {job['id']}")
                run = self.prepare(job, identity)
                report = self.executors[job["model"]](run)
                receipt = self.confirm(job, identity, run, report)
            self.receipts[job["id"]] = receipt
        return "completed"


def _identity(stage, views):
    campaign = stage["campaign"]
    return dict(
        schema_version=1,
        kind=RUN_KIND,
        stage_sha256=stage["sha256"],
        campaign_sha256=campaign["sha256"],
        comparison_sha256=campaign["comparison_config"]["sha256"],
        input_policy=campaign["input_policy"],
        variant=campaign["variant"],
        declaration=stage["declaration"],
        views={
            scope: {window: value["sha256"] for window, value in record["windows"].items()}
            for scope, record in views.items()
        },
        code=_code(),
        final_test_opened=False,
    )


def _summary(output, identity, jobs, state, status, **extra):
    done = [job for job in jobs if job["id"] in state.receipts]
    summary = dict(
        schema_version=1,
        kind=RUN_KIND,
        status=status,
        identity_sha256=_digest(identity),
        planned=dict(training_jobs=0, prediction_jobs=len(jobs)),
        completed=dict(training_jobs=0, prediction_jobs=len(done)),
        variants=dict(Counter(job["variant"] for job in done)),
        optimizer_steps=0,
        jobs={job["id"]: job["id"] in state.receipts for job in jobs},
        final_test_opened=False,
        updated_at_utc=datetime.now(UTC).isoformat(),
        **extra,
    )
    atomic_json(output / "summary.json", summary)
    return summary


def _gpu_lease():
    from .experiment_resources import GpuLease

    return GpuLease()


def _opened(path, views, campaign_output, output, pairs=None):
    """Etapa, campaña base confirmada y destino comprobados, sin crear nada."""
    stage = load_stage(path)
    campaign = stage["campaign"]
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
    _base_receipts(base, campaign, stage, pairs)
    return stage, base, output


def run_stage(path, views, campaign_output, output, *, executors=None, lease=None, stop=None):
    """Ejecutar o reanudar la etapa sobre una campaña base confirmada.

    `executors` sustituye los ejecutores por modelo y `lease`, la reserva de la GPU. La
    protección se comprueba antes de abrir fuentes y antes de cada trabajo pendiente.
    """
    from .checkpoints import StopRequest

    require_evaluation_allowed("la etapa de ablación de modalidades")
    stage = load_stage(path)
    jobs = plan_stage(stage)
    count_stage(stage, jobs)
    executors = dict(EXECUTORS if executors is None else executors)
    _require(set(executors) == set(EXECUTORS), "Faltan ejecutores para algún modelo")
    stage, base, output = _opened(path, views, campaign_output, output)
    identity = _identity(stage, base.views)
    output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = output / "stage.json"
        if marker.exists():
            _require(
                read_manifest(marker, 8 * 1024**2)[0] == identity,
                "La salida pertenece a otra etapa, campaña, comparación, vista o código",
            )
        else:
            _require(
                all(p.name in {".lock", "summary.json"} for p in output.iterdir()),
                "La salida sin identidad contiene artefactos ajenos",
            )
            atomic_json(marker, identity)
        state = _Stage(stage, base, output, identity, executors, None)
        signals = StopRequest() if stop is None else nullcontext(stop)
        reservation = (lease or _gpu_lease)()
        _summary(output, identity, jobs, state, "running")
        try:
            with signals as state.stop, reservation, forbid_optimizer_steps():
                status = state.execute(jobs)
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


def write_sources(
    path, views, campaign_output, output, scope, *, comparison_path=None, window=None
):
    """Escribir el manifiesto de predicciones enmascaradas de un ámbito y validarlo.

    Sin `comparison_path` se valida con la comparación de la campaña. Una comparación con
    un subconjunto de brazos publica solo esos brazos, como `masked_campaign.write_sources`.
    Con `window` se publica solo esa ventana en `sources/windows/<ventana>/`.
    """
    pairs = None
    if window is not None:
        # La ventana y los anclas de los que parten sus predicciones enmascaradas.
        planned = [
            job
            for job in plan_stage(load_stage(path))
            if (job["scope"], job["window"]) == (scope, window)
        ]
        pairs = {(scope, name) for job in planned for name in (job["window"], job["anchor"])}
    stage, base, output = _opened(path, views, campaign_output, output, pairs)
    _require(scope in stage["scopes"], "El ámbito no pertenece a la etapa")
    identity = _identity(stage, base.views)
    _require(
        (output / "stage.json").is_file()
        and read_manifest(output / "stage.json", 8 * 1024**2)[0] == identity,
        "La salida no corresponde a esta etapa, su campaña, sus vistas o su código",
    )
    validation = comparison.load_config(
        Path(comparison_path or stage["campaign"]["comparison_path"]).resolve()
    )
    if window is not None:
        validation = comparison.restrict_windows(validation, scope, [window])
    _require(
        validation.get(comparison.ABLATION_FIELD) == stage["declaration"],
        "La comparación de validación no declara la misma ablación",
    )
    produced = {spec["arm"]: spec for spec in stage["specs"]}
    wanted = {
        name: arm
        for name, arm in validation["arms"].items()
        if arm["output"] != comparison.ZERO_CONTROL
    }
    missing = sorted(set(wanted) - set(produced))
    _require(not missing, f"Faltan productores para {', '.join(missing)}")
    state = _Stage(stage, base, output, identity, EXECUTORS, None)
    windows = base.views[scope]["windows"]
    folder = output / "sources"
    if window is not None:
        windows, folder = {window: windows[window]}, folder / "windows" / window
    variants = {}
    for job in plan_stage(stage):
        if job["scope"] != scope or job["arm"] not in wanted or job["window"] not in windows:
            continue
        _require(
            job["seed"] in wanted[job["arm"]]["seeds"],
            f"La comparación declara otras semillas para {job['arm']}",
        )
        receipt = state.confirmed(job, state.job_identity(job))
        _require(receipt is not None, f"Falta confirmar {job['id']} antes de publicar fuentes")
        record = receipt["prediction"]
        seeds = variants.setdefault(job["variant"], {}).setdefault(job["arm"], {})
        seeds.setdefault(str(job["seed"]), {})[job["window"]] = dict(
            evaluation=dict(
                path=os.path.relpath(output / record["path"], folder), sha256=record["sha256"]
            )
        )
    manifest = dict(
        schema_version=1,
        kind=analysis.SOURCES_KIND,
        scope=scope,
        input_policy=stage["campaign"]["input_policy"],
        comparison_sha256=validation["sha256"],
        windows={window: value["sha256"] for window, value in windows.items()},
        variants=variants,
    )
    destination = folder / f"{scope}.json"
    candidate = folder / f".{scope}.candidate.json"
    safe_destination(destination)
    atomic_json(candidate, manifest)
    resolved = validation["resolved_scopes"][scope]
    try:
        comparison._ablation_sources(
            candidate,
            validation,
            dict(scope=scope, views=manifest["windows"], windows=resolved["windows"]),
        )
    except BaseException:
        candidate.unlink()
        raise
    os.replace(candidate, destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Validar y contar predicciones sin leer datos")
    execute = commands.add_parser("run", help="Ejecutar o reanudar la etapa")
    sources = commands.add_parser("sources", help="Publicar el manifiesto de un ámbito")
    for command in (check, execute, sources):
        command.add_argument("--stage", type=Path, required=True)
    for command in (execute, sources):
        command.add_argument("--views", action="append", required=True)
        command.add_argument("--campaign-output", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    sources.add_argument("--scope", choices=tuple(comparison.SCOPES), required=True)
    sources.add_argument("--comparison", type=Path)
    args = parser.parse_args(argv)
    if args.command == "check":
        result = check_stage(args.stage)
    elif args.command == "run":
        views = masked_campaign._views_argument(args.views)
        result = run_stage(args.stage, views, args.campaign_output, args.output)
        result.pop("jobs")
    else:
        destination = write_sources(
            args.stage,
            masked_campaign._views_argument(args.views),
            args.campaign_output,
            args.output,
            args.scope,
            comparison_path=args.comparison,
        )
        result = dict(sources=str(destination), sha256=sha256(destination))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status", "completed") in {"completed", "checked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
