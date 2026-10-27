"""Diagnóstico E2 por ventanas con pesos congelados y confirmación de fases."""

import hashlib
import importlib
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.session_metrics import SessionErrors
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.memory.spaced_study import load_study

from .checkpoints import StopRequest
from .corpus_inputs import CorpusDataset
from .experiment_resources import GpuLease, check_host_memory
from .financial_run import Paused
from .learning_hold import require_learning_allowed
from .mars_titan_correction import _parent_specification
from .mars_titan_run import MarsTitanInference
from .mars_titan_walk_forward import _cuda, _native, _parent
from .prefix_eligibility import PrefixTargetVerifier
from .spaced_inference import SpacedInference
from .spaced_selected import load_selected_reader
from .titans_walk_forward import (
    PREDICTED,
    PredictionRows,
    _check_view,
    _protocol,
    _require,
    _sources,
    _verify,
    checked_tables,
    unfused_attention,
    view_protocol,
    walk_forward_options,
    window_phases,
)
from .window_indices import index_root

KIND = "spaced_utility_frozen_window_v1"
_CODE = (
    "mars_titan.training.spaced_window",
    "mars_titan.memory.spaced_study",
    "mars_titan.training.spaced_selected",
    "mars_titan.training.spaced_inference",
    "mars_titan.training.mars_titan_run",
    "mars_titan.training.mars_titan_walk_forward",
    "mars_titan.memory.spaced_session",
    "mars_titan.memory.spaced_retention",
    "mars_titan.memory.spaced_utility",
    "mars_titan.memory.spaced_utility_readout",
)


def _digest(value):
    from mars_titan.models.titans.config import canonical

    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _code():
    return {name: sha256(Path(importlib.import_module(name).__file__)) for name in _CODE}


def _spent(report):
    attempts = report.get("attempts")
    _require(
        type(attempts) is list
        and len(attempts) < 1000
        and all(
            type(v) is dict
            and type(v.get("seconds")) in (int, float)
            and math.isfinite(v["seconds"])
            and v["seconds"] >= 0
            for v in attempts
        ),
        "El historial de intentos no conserva un coste válido",
    )
    return sum(v["seconds"] for v in attempts)


class _Budget:
    def __init__(self, stop, seconds, *, lease=None):
        self.stop, self.seconds, self.lease = stop, seconds, lease
        self.started = time.monotonic()

    @property
    def requested(self):
        if self.lease is None:
            check_host_memory()
        else:
            self.lease.check()
        return self.stop.requested or time.monotonic() - self.started >= self.seconds


def _verify_outputs(output, report):
    _require(type(report.get("predictions")) is dict, "El informe no conserva sus fases")
    _require(set(report["predictions"]) <= set(PREDICTED), "El informe añade fases desconocidas")
    for name, record in report["predictions"].items():
        _require(type(record) is dict, "La fase no conserva su registro de archivos")
        _require(record.get("path") == f"{name}-predictions.parquet", "La fase cambió de ruta")
        sessions = record.get("sessions")
        _require(
            type(sessions) is dict and sessions.get("path") == f"{name}-session-metrics.parquet",
            "La fase no conserva sus métricas por sesión",
        )
        path = output / sessions["path"]
        safe_destination(path)
        _require(
            path.is_file() and sha256(path) == sessions.get("sha256"),
            "Las métricas por sesión han cambiado",
        )
        evidence = record.get("evidence")
        if report["request"]["arm"] == "baseline":
            _require(evidence is None, "La referencia no debe inventar evidencia E2")
        else:
            _require(
                type(evidence) is dict and evidence.get("path") == f"{name}-evidence.json",
                "La fase no conserva su evidencia E2",
            )
            path = output / evidence["path"]
            safe_destination(path)
            _, digest = read_manifest(path, 16 * 1024**2)
            _require(digest == evidence.get("sha256"), "La evidencia E2 ha cambiado")
    _verify(output, report)
    if report["status"] == "completed":
        _require(set(report["predictions"]) == set(PREDICTED), "La ventana completada pierde fases")


def _session_table(tables, metrics):
    errors = SessionErrors()
    for table in tables:
        for batch in table.to_batches(max_chunksize=4096):
            errors.update(
                batch["market"].to_pylist(),
                batch["prediction_at"].cast(pa.int64()).to_numpy(),
                batch["prediction"].to_numpy() - batch["target"].to_numpy(),
            )
    measured = errors.summary()
    _require(
        measured["samples"] == metrics["labels"]
        and measured["session_count"] == metrics["session_count"]
        and all(
            math.isclose(measured[key], metrics[key], rel_tol=1e-12, abs_tol=1e-14)
            for key in ("session_mae", "session_mse")
        ),
        "Las filas emitidas no reproducen las métricas por sesión",
    )
    entries = sorted(errors.sessions.items())
    return pa.table(
        dict(
            market=[market for (market, _), _ in entries],
            prediction_at=pa.array(
                [at for (_, at), _ in entries], type=pa.timestamp("us", tz="UTC")
            ),
            samples=[v[0] for _, v in entries],
            mae=[v[1] / v[0] for _, v in entries],
            mse=[v[2] / v[0] for _, v in entries],
            rmse=[math.sqrt(v[2] / v[0]) for _, v in entries],
        )
    )


def _prepare(view, anchor, source_manifest, study_path, arm, output, device, indices, holder):
    study, study_sha = load_study(study_path)
    retention = study.retention_for(arm)
    anchor_report, anchor_sha = read_manifest(anchor / "run.json", 16 * 1024**2)
    request = anchor_report.get("request", {})
    _require(type(request) is dict, "El modelo elegido no conserva su petición")
    seed = request.get("seed")
    _, protocol_sha, _, fold = _protocol(view_protocol(view), request.get("window"), seed)
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    _check_view(dataset, view_protocol(view), fold)
    _require(
        request.get("view_sha256") == dataset.identity, "El estado elegido pertenece a otra vista"
    )
    parent_path = Path(request["parent"]["path"])
    parent, _ = _parent(parent_path, view, seed)
    options = walk_forward_options(parent["identity"]["recipe"], bounded_fit=True)
    phases = window_phases(fold, options["warmup_months"], options.get("fit_months"))
    encoded = {name: asdict(phase) for name, phase in phases.items()}
    _require(
        encoded == parent["identity"]["phases"] == anchor_report["identity"]["phases"],
        "El diagnóstico debe conservar las fases originales del estado elegido",
    )
    _require(
        all(dataset.manifest["counts"][name] <= study.max_phase_rows for name in PREDICTED),
        "Una fase supera las filas permitidas, no se recortan muestras",
    )
    _, original_sha = read_manifest(source_manifest, 8 * 1024**2)
    native = _native("m1")
    proposal = dict(
        schema_version=1,
        kind=KIND,
        arm=arm,
        device=device,
        view_sha256=dataset.identity,
        anchor_run_sha256=anchor_sha,
        anchor_selected_sha256=anchor_report["checkpoint"]["sha256"],
        source_manifest_sha256=original_sha,
        study_sha256=study_sha,
        study=study.identity(),
        protocol_sha256=protocol_sha,
        code=_code(),
        native_sha256=native.binary_sha256,
    )
    safe_destination(output)
    report_path = output / "run.json"
    if report_path.exists():
        report, _ = read_manifest(report_path, 16 * 1024**2)
        _require(
            type(report.get("schema_version")) is int
            and report["schema_version"] == 1
            and report.get("status") in ("running", "paused", "completed", "failed")
            and (report.get("status") != "completed" or type(report.get("identity")) is dict)
            and report.get("kind") == KIND
            and report.get("request") == proposal
            and report.get("final_test_opened") is False,
            "La petición o el estado del diagnóstico E2 ha cambiado",
        )
        _verify_outputs(output, report)
    else:
        _require(
            not output.exists() or not any(output.iterdir()), "La salida pertenece a otra ejecución"
        )
        for path in (*dataset.roots.values(), view.parent, anchor, parent_path, Path("dataset")):
            outside_source(path, output)
            outside_source(output, path)
        output.mkdir(parents=True, exist_ok=True)
        report = dict(
            schema_version=1,
            kind=KIND,
            request=proposal,
            identity=None,
            run_id=None,
            status="running",
            predictions={},
            attempts=[],
            started_at_utc=datetime.now(UTC).isoformat(),
            final_test_opened=False,
        )
        atomic_json(report_path, report)
    _spent(report)
    holder.update(report=report, destination=report_path)
    root = Path(indices) if indices is not None else index_root(dataset, output)
    for path in (*dataset.roots.values(), view.parent, anchor, parent_path):
        outside_source(path, root)
    sources = _sources(dataset, {name: phases[name] for name in PREDICTED}, root)
    specification = _parent_specification(sources["validation"], parent)
    loaded = load_selected_reader(anchor, view, specification, device=device)
    _require(
        study.retention.capacity == loaded.recipe.bank_capacity
        and study.retention.seed == loaded.recipe.bank_seed,
        "La capacidad y semilla deben coincidir con la referencia del lector",
    )
    prefixes = PrefixTargetVerifier(dataset, source_manifest=source_manifest)
    common = dict(
        native=native, codec=loaded.codec, world="spaced_utility_frozen_study_v1", fold=fold["id"]
    )
    if retention is None:
        inference = MarsTitanInference(
            loaded.consumer.predictor,
            loaded.consumer.readout,
            loaded.recipe,
            admission="m1",
            retention=RetentionConfig(
                capacity=study.retention.capacity,
                seed=study.retention.seed,
                frontier=min(8, study.retention.capacity),
            ),
            **common,
        )
    else:
        inference = SpacedInference(
            loaded.consumer, loaded.recipe, retention=retention, prefixes=prefixes, **common
        )
    identity = dict(
        schema_version=1,
        kind=KIND,
        request=proposal,
        selected=loaded.identity(),
        phases={name: asdict(phases[name]) for name in PREDICTED},
        indices={name: source.identity for name, source in sources.items()},
        expected_rows={name: dataset.manifest["counts"][name] for name in PREDICTED},
        retention=asdict(inference.retention),
        evidence_enabled=arm != "baseline",
        state_transfer="selected_shared_parameters_only",
        state_reset="each_phase_empty_fast_memory_bank_and_evidence",
        interrupted_phase="repeat_from_start",
    )
    _require(
        report["identity"] is None or report["identity"] == identity,
        "La identidad del diagnóstico cambió al reconstruirlo",
    )
    report.update(identity=identity, run_id=_digest(identity))
    return study, dataset, sources, inference, report


def run_spaced_window(
    view, anchor, source_manifest, study, *, arm, output, device="cuda:0", indices=None, stop=None
):
    """Ejecutar un único control del estudio. La guarda precede a fuentes, GPU y outputs."""
    require_learning_allowed("la ventana congelada de retención E2")
    _require(device in ("cpu", "cuda:0"), "El dispositivo debe ser cpu o cuda:0 explícitos")
    started = time.monotonic()
    lease = GpuLease() if device == "cuda:0" else nullcontext()
    with lease as resource, unfused_attention():
        _cuda(device)
        paths = tuple(Path(path) for path in (view, anchor, source_manifest, study, output))
        holder = {}
        try:
            config, dataset, sources, inference, report = _prepare(
                *paths[:4], arm, paths[4], device, indices, holder
            )
        except BaseException as error:
            if holder:
                failed = holder["report"]
                failed.update(status="failed", error_type=type(error).__name__, error=str(error))
                failed["attempts"].append(dict(seconds=time.monotonic() - started))
                atomic_json(holder["destination"], failed)
            raise
        destination = paths[4] / "run.json"
        if report["status"] == "completed":
            return report
        remaining = config.max_seconds - _spent(report)
        budget = _Budget(
            stop or StopRequest(),
            remaining - (time.monotonic() - started),
            lease=resource if device == "cuda:0" else None,
        )
        report["status"] = "running"
        atomic_json(destination, report)
        try:
            for name in PREDICTED:
                if name in report["predictions"]:
                    continue
                if budget.requested:
                    raise Paused
                rows = PredictionRows(inference.quantiles)
                metrics = inference.evaluate(sources[name], stop=budget, rows=rows)
                tables = checked_tables(rows, metrics, dataset, name)
                path = paths[4] / f"{name}-predictions.parquet"
                count = atomic_parquet_batches(path, tables)
                _require(count == rows.count, "La escritura de predicciones pierde filas")
                sessions_path = paths[4] / f"{name}-session-metrics.parquet"
                session_count = atomic_parquet_batches(
                    sessions_path, [_session_table(tables, metrics)]
                )
                evidence = None
                if arm != "baseline":
                    target = paths[4] / f"{name}-evidence.json"
                    atomic_json(target, inference.spaced_state())
                    evidence = dict(path=target.name, sha256=sha256(target))
                report["predictions"][name] = dict(
                    path=path.name,
                    sha256=sha256(path),
                    rows=count,
                    bytes=path.stat().st_size,
                    metrics=metrics,
                    evidence=evidence,
                    sessions=dict(
                        path=sessions_path.name, sha256=sha256(sessions_path), rows=session_count
                    ),
                )
                atomic_json(destination, report)
            report.update(status="completed", finished_at_utc=datetime.now(UTC).isoformat())
        except Paused:
            report["status"] = "paused"
        except BaseException as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            report["attempts"].append(dict(seconds=time.monotonic() - started))
            atomic_json(destination, report)
        return report
