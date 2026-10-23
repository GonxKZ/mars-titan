"""Contrastar recetas con sesiones propias y confirmar solo su destino experimental."""

import fcntl
import hashlib
import math
import os
import re
import resource
import stat
import time
from contextlib import contextmanager
from dataclasses import asdict
from itertools import islice
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.financial_observations import (
    FinancialObservationSource,
    run_observation_source,
)
from mars_titan.memory.financial_session import FinancialSession
from mars_titan.memory.regional_session_work import configuration as regional_configuration
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.memory.write_policy import CompositeScoreConfig, MatureErrorConfig
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.proposals.candidate_runtime import materialize_registered_candidate
from mars_titan.proposals.error_proposer import _bindings
from mars_titan.proposals.error_rule import _finite, _point_statistics
from mars_titan.proposals.mature_review import encoded, fingerprint, identifier, require
from mars_titan.proposals.trajectory_reader import MatureTrajectory, read_mature_trajectory
from mars_titan.training.prefix_eligibility import PrefixTargetVerifier

MAX_STATE_BYTES = 4 * 1024**2
MAX_TRAJECTORY_BYTES = 8 * 1024**2
MAX_LOG_BYTES = 128 * 1024**2
MAX_ATTEMPTS = 2
INITIALIZATION = "paired_parent_weights_fresh_state_v1"
FRAME_FIELDS = (
    "flow_id",
    "sample_id",
    "prediction_at",
    "input_available_at",
    "input_sha256",
    "matures_at",
)


def _seal(value):
    body = {key: part for key, part in value.items() if key != "state_sha256"}
    return dict(body, state_sha256=fingerprint(body, MAX_STATE_BYTES))


def _publish(path, value):
    encoded(value, MAX_STATE_BYTES, pretty=True)
    atomic_json(path, value)


def _initial_state(identity, contrast_id):
    return dict(
        schema_version=1,
        identity=identity,
        identity_id=contrast_id,
        status="running",
        arms=[
            dict(configuration_id=item, status="planned", attempts=[], receipt=None)
            for item in identity["configurations"]
        ],
        result=None,
    )


def _recover_bootstrap(output):
    with os.scandir(output) as entries:
        paths = [Path(entry.path) for entry in islice(entries, 19)]
    require(len(paths) <= 18, "El arranque conserva demasiados artefactos temporales")
    temporary = []
    for path in paths:
        if path.name in (".lock", "identity.json"):
            continue
        info = path.lstat()
        require(
            re.fullmatch(r"\.(?:identity|state)\.json\.[a-z0-9_]{8}", path.name) is not None
            and stat.S_ISREG(info.st_mode)
            and info.st_size <= MAX_STATE_BYTES,
            "Falta el estado confirmado o existe un temporal ajeno al arranque",
        )
        temporary.append(path)
    for path in temporary:
        path.unlink()


@contextmanager
def _locked(output, *, resume):
    require(type(resume) is bool, "resume debe ser booleano")
    safe_destination(output)
    require(
        output.exists() == resume, "El contraste necesita salida nueva o recuperación explícita"
    )
    if not resume:
        output.mkdir(parents=True)
    require(output.is_dir(), "La salida del contraste no es un directorio")
    descriptor = os.open(
        output / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
    )
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode), "El bloqueo no es regular")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def _source(
    source, plan, parent, prefixes, retention, native, block_rows, admission, regional_work_limits
):
    require(type(source) is FinancialObservationSource, "Falta la fuente de observaciones real")
    require(type(parent) is FrozenMechanismConsumer, "Falta el consumidor padre congelado")
    expected_retention = {
        "m0": RetentionConfig,
        "m1": RetentionConfig,
        "m2": MatureErrorConfig,
        "m3": CompositeScoreConfig,
    }
    require(
        type(prefixes) is PrefixTargetVerifier
        and type(admission) is str
        and admission in expected_retention
        and type(retention) is expected_retention[admission],
        "Faltan los prefijos o la retención identificados",
    )
    require(
        type(block_rows) is int and 1 <= block_rows <= parent.max_batch,
        "El bloque físico no cabe en el consumidor",
    )
    require(
        type(admission) is str and admission in ("m0", "m1", "m2", "m3"),
        "La política de escritura no está admitida",
    )
    source._confirm()
    parent.verify(strong=True)
    retention.__post_init__()
    regional_configuration(parent, regional_work_limits, MAX_ATTEMPTS)
    phase = source.phase
    require(
        phase.partition in ("train", "validation")
        and phase.decision_start == plan.selection_start
        and phase.close_at < plan.selection_end
        and parent.model_id == plan.parent_consumer_id
        and source.dataset.identity == parent.predictor.config.inputs.source_sha256
        and prefixes.source_id == source.dataset.identity,
        "La selección no corresponde al intervalo, corpus y padre declarados",
    )
    require(
        len(source.metadata["groups"]) <= plan.max_source_records,
        "La selección supera sus generaciones declaradas",
    )
    identifier(native.binary_sha256, "El enlace nativo")
    return source.specification()


def _identity(
    ledger,
    plan,
    evidence,
    source,
    parent,
    parent_codec,
    prefixes,
    retention,
    native,
    block_rows,
    admission,
    selection_min_gain,
    regional_work_limits,
):
    _, state, requests = _bindings(ledger, plan, evidence)
    require(
        len(state["attempts"]) == len(requests)
        and all(item["status"] != "reserved" for item in state["attempts"]),
        "La generación debe terminar antes de iniciar la selección",
    )
    proposed = {
        item["configuration_id"] for item in state["attempts"] if item["status"] == "proposed"
    }
    configurations = [
        ledger.registry.baseline_id,
        *[candidate for candidate in plan.candidate_ids if candidate in proposed],
    ]
    modules = tuple(
        Path(__file__).with_name(name)
        for name in (
            "trajectory_contrast.py",
            "candidate_runtime.py",
            "trajectory_reader.py",
            "error_rule.py",
            "error_proposer.py",
            "error_source.py",
            "mature_review.py",
            "ledger.py",
            "contracts.py",
        )
    )
    return dict(
        schema_version=1,
        kind="registered_trajectory_contrast_identity_v1",
        plan_id=plan.plan_id,
        registry_id=ledger.registry.registry_id,
        evidence_id=evidence.record()["evidence_id"],
        review_id=evidence.review.review_id,
        proposal_receipts=[item["receipt_sha256"] for item in state["attempts"]],
        configurations=configurations,
        baseline_id=ledger.registry.baseline_id,
        parent_consumer_id=parent.model_id,
        parent_parameter_id=parent.predictor._parameter_id,
        parent_codec_id=parent_codec.fingerprint(),
        source=dict(
            index_id=source.identity, dataset_id=source.dataset.identity, phase=asdict(source.phase)
        ),
        prefix_id=prefixes.policy_id,
        retention=asdict(retention),
        admission=admission,
        block_rows=block_rows,
        native_sha256=native.binary_sha256,
        initialization=INITIALIZATION,
        selection_min_gain=selection_min_gain,
        regional_work_limits=regional_work_limits,
        max_execution_attempts_per_arm=MAX_ATTEMPTS,
        max_native_log_bytes_per_arm=MAX_LOG_BYTES,
        code={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in modules},
        evaluation_executed=False,
        canonical_default_changed=False,
    )


def _read_state(output, identity):
    value, _ = read_manifest(output / "state.json", MAX_STATE_BYTES)
    require(
        type(value) is dict
        and set(value)
        == {"schema_version", "identity", "identity_id", "status", "arms", "result", "state_sha256"}
        and value["schema_version"] == 1
        and type(value["schema_version"]) is int
        and value["identity"] == identity
        and value["identity_id"] == fingerprint(identity)
        and value == _seal(value),
        "El contraste cambió su identidad o estado",
    )
    require(
        type(value["status"]) is str
        and value["status"] in ("running", "completed", "inconclusive")
        and (
            value["result"] is None
            if value["status"] == "running"
            else type(value["result"]) is dict
        )
        and type(value["arms"]) is list
        and all(type(entry) is dict for entry in value["arms"])
        and [entry.get("configuration_id") for entry in value["arms"]]
        == identity["configurations"],
        "El contraste cambió el orden de las recetas",
    )
    for entry in value["arms"]:
        require(
            type(entry) is dict
            and set(entry) == {"configuration_id", "status", "attempts", "receipt"}
            and entry["status"] in ("planned", "running", "failed", "completed")
            and type(entry["attempts"]) is list
            and len(entry["attempts"]) <= MAX_ATTEMPTS,
            "Un brazo cambió su estado o cuota de ejecución",
        )
        for attempt in entry["attempts"]:
            _check_attempt(attempt)
        require(
            all(item["status"] in ("interrupted", "failed") for item in entry["attempts"][:-1]),
            "Un intento previo no conserva su fallo o interrupción",
        )
        if entry["status"] == "planned":
            require(
                not entry["attempts"] and entry["receipt"] is None,
                "Un brazo pendiente declara ejecución previa",
            )
        else:
            require(
                bool(entry["attempts"])
                and (
                    entry["attempts"][-1]["status"] == entry["status"]
                    or entry["status"] == "completed"
                    and len(entry["attempts"]) == MAX_ATTEMPTS
                    and entry["attempts"][-1]["status"] in ("failed", "interrupted")
                    and type(entry["receipt"]) is dict
                    and entry["receipt"].get("cost", {}).get("native_closed_reconciliation") is True
                ),
                "El brazo contradice su último intento",
            )
            require(
                (
                    type(entry["receipt"]) is dict
                    if entry["status"] == "completed"
                    else entry["receipt"] is None
                ),
                "El brazo perdió su recibo terminal",
            )
    if value["status"] != "running":
        require(
            all(entry["status"] == "completed" for entry in value["arms"])
            and value["result"].get("status") == value["status"],
            "El contraste terminal conserva brazos pendientes o un resultado incompatible",
        )
    return value


def _check_attempt(attempt):
    require(
        type(attempt) is dict
        and set(attempt) == {"status", "elapsed_seconds", "error"}
        and type(attempt["status"]) is str
        and attempt["status"] in ("running", "interrupted", "failed", "completed"),
        "El intento cambió su esquema o estado",
    )
    if attempt["status"] in ("running", "interrupted"):
        require(attempt["elapsed_seconds"] is None, "Un corte sin recibo no tiene un coste medido")
    else:
        value = attempt["elapsed_seconds"]
        require(
            type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "Un intento no conserva su coste medido",
        )
    if attempt["status"] in ("running", "completed"):
        require(attempt["error"] is None, "Un intento sin fallo declara un error")
    else:
        require(
            type(attempt["error"]) is str and 1 <= len(attempt["error"].encode()) <= 1024,
            "Falta el motivo acotado del intento interrumpido",
        )


def _stored_trajectory(output, entry, *, authoritative, initialization, plan):
    receipt = entry["receipt"]
    require(type(receipt) is dict, "Falta el recibo de la trayectoria terminada")
    relative = f"trajectories/{entry['configuration_id']}.json"
    require(receipt["trajectory_relative_path"] == relative, "La trayectoria sale de su destino")
    path = output / relative
    safe_destination(path)
    body, signature = read_manifest(path, MAX_TRAJECTORY_BYTES)
    require(signature == receipt["trajectory_file_sha256"], "La trayectoria cambió sus bytes")
    trajectory_id = body.pop("trajectory_id")
    trajectory = MatureTrajectory(body)
    result = trajectory.record()
    require(
        result["trajectory_id"] == trajectory_id == receipt["trajectory_id"],
        "La trayectoria cambió su identidad",
    )
    require(
        result == authoritative.record(),
        "La trayectoria derivada no corresponde a las emisiones nativas actuales",
    )
    require(
        receipt["configuration_id"] == entry["configuration_id"]
        and receipt["effective_configuration_id"] == initialization["effective_configuration_id"]
        and receipt["consumer_id"] == result["consumer_id"]
        and receipt["parameter_id"] == result["parameter_id"]
        and receipt["initialization"] == initialization
        and receipt["statistics"] == _statistics(authoritative, plan),
        "El recibo no corresponde a los pesos, emisiones o estadísticas verificadas",
    )
    return trajectory


def _statistics(trajectory, plan):
    body = trajectory.record()
    return _point_statistics(
        body["rows"],
        flow_ids=plan.flow_ids,
        generation_start=plan.selection_start,
        generation_end=plan.selection_end,
        market_weights=plan.market_weights,
        min_sessions_per_market=plan.min_sessions_per_market,
        min_excess_mae=plan.min_excess_mae,
    )


@contextmanager
def _arm_session(
    output,
    entry,
    *,
    ledger,
    parent,
    parent_codec,
    specification,
    source,
    prefixes,
    retention,
    native,
    block_rows,
    admission,
    contrast_id,
    regional_work_limits,
):
    runtime = materialize_registered_candidate(
        ledger.registry,
        entry["configuration_id"],
        parent=parent,
        parent_codec=parent_codec,
        input_spec=specification,
        device="cuda:0",
    )
    directory = output / "arms" / entry["configuration_id"]
    with FinancialSession(
        directory,
        native=native,
        consumer=runtime.consumer,
        codec=runtime.codec,
        prefixes=prefixes,
        retention=retention,
        phase=source.phase,
        admission=admission,
        world="registered_contrast_v1",
        fold=contrast_id[:16],
        block_rows=block_rows,
        max_log_bytes=MAX_LOG_BYTES,
        mechanism_collection=None,
        regional_work_limits=regional_work_limits,
        regional_max_attempts=MAX_ATTEMPTS,
        resume=directory.exists(),
    ) as session:
        yield runtime, session


def _audit_arm(output, entry, *, plan, **options):
    require(
        (output / "arms" / entry["configuration_id"]).is_dir(),
        "Falta la sesión nativa de un brazo terminado",
    )
    with _arm_session(output, entry, **options) as (runtime, session):
        require(
            session.snapshot()["phase_closed"] is True
            and session.snapshot()["cursor"] == len(options["source"].metadata["groups"]),
            "El brazo terminado no conserva su cierre nativo completo",
        )
        trajectory = read_mature_trajectory(
            session,
            before=plan.selection_end,
            max_records=plan.max_source_records,
            max_bytes=plan.max_source_bytes,
        )
        return trajectory, runtime.record()


def _run_arm(
    output,
    entry,
    *,
    ledger,
    plan,
    parent,
    parent_codec,
    specification,
    source,
    prefixes,
    retention,
    native,
    block_rows,
    admission,
    contrast_id,
    regional_work_limits,
    audit_only=False,
):
    start = time.perf_counter()
    if audit_only:
        require(
            (output / "arms" / entry["configuration_id"]).is_dir(),
            "La cuota está agotada y no existe una sesión nativa que reconciliar",
        )
    with _arm_session(
        output,
        entry,
        ledger=ledger,
        parent=parent,
        parent_codec=parent_codec,
        specification=specification,
        source=source,
        prefixes=prefixes,
        retention=retention,
        native=native,
        block_rows=block_rows,
        admission=admission,
        contrast_id=contrast_id,
        regional_work_limits=regional_work_limits,
    ) as (runtime, session):
        initialized = time.perf_counter()
        if audit_only:
            require(
                session.snapshot()["phase_closed"] is True
                and session.snapshot()["cursor"] == len(source.metadata["groups"]),
                "La cuota está agotada y la sesión nativa no está cerrada y completa",
            )
        else:
            run_observation_source(source, session)
        executed = time.perf_counter()
        require(
            session.snapshot()["phase_closed"] is True
            and session.snapshot()["cursor"] == len(source.metadata["groups"]),
            "La trayectoria no completó toda la cronología de selección",
        )
        trajectory = read_mature_trajectory(
            session,
            before=plan.selection_end,
            max_records=plan.max_source_records,
            max_bytes=plan.max_source_bytes,
        )
        runtime.consumer.verify(strong=True)
        statistics = _statistics(trajectory, plan)
        body = trajectory.record()
        work = {}
        for generation in range(1, session.snapshot()["cursor"] + 1):
            counters = session.mechanism_event(generation)["payload"]["work"]
            require(
                all(type(value) is int and value >= 0 for value in counters.values()),
                "La traza de trabajo contiene un contador no entero o negativo",
            )
            for name, value in counters.items():
                work[name] = work.get(name, 0) + value
        path = output / "trajectories" / f"{entry['configuration_id']}.json"
        path.parent.mkdir(exist_ok=True)
        encoded(body, MAX_TRAJECTORY_BYTES, pretty=True)
        atomic_json(path, body)
        import torch

        from mars_titan.data.storage import sha256

        receipt = dict(
            configuration_id=entry["configuration_id"],
            effective_configuration_id=runtime.effective_configuration.configuration_id,
            consumer_id=runtime.consumer.model_id,
            parameter_id=runtime.consumer.predictor._parameter_id,
            trajectory_id=body["trajectory_id"],
            trajectory_relative_path=path.relative_to(output).as_posix(),
            trajectory_file_sha256=sha256(path),
            initialization=runtime.record(),
            statistics=statistics,
            cost=dict(
                initialization_seconds=initialized - start,
                execution_seconds=None if audit_only else executed - initialized,
                native_closed_reconciliation=audit_only,
                readback_seconds=time.perf_counter() - executed,
                logical_source_bytes=body["logical_source_bytes"],
                native_generations=session.snapshot()["cursor"],
                native_work=work,
                shared_parameter_training=False,
                process_ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                cumulative_torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                cumulative_torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                memory_scope="cumulative_process_peaks_not_exclusive_arm_allocations",
                energy=None,
                cpu_gpu_transfer_bytes=None,
            ),
        )
    return receipt, trajectory


def _compare(trajectories, arms, identity, plan):
    reference = sorted(trajectories[0].record()["rows"], key=lambda row: row["sample_id"])
    frame = [{key: row[key] for key in FRAME_FIELDS} for row in reference]
    for trajectory in trajectories[1:]:
        rows = sorted(trajectory.record()["rows"], key=lambda row: row["sample_id"])
        require(
            [{key: row[key] for key in FRAME_FIELDS} for row in rows] == frame,
            "Las trayectorias no comparten el marco completo de emisiones e inputs",
        )
        for base, variant in zip(reference, rows, strict=True):
            if base["status"] == variant["status"] == "label":
                require(
                    base["target"] == variant["target"], "Las trayectorias cambiaron sus etiquetas"
                )
    complete = all(entry["statistics"]["status"] == "complete" for entry in arms)
    baseline = arms[0]["statistics"]["parent_mae"] if complete else None
    rows, selected, best = {}, identity["baseline_id"] if complete else None, baseline
    for entry in arms:
        identifier_ = entry["configuration_id"]
        mae = entry["statistics"]["parent_mae"] if complete else None
        delta = None if mae is None else mae - baseline
        rows[identifier_] = dict(mae=mae, delta_mae=delta, gain=None if delta is None else -delta)
        if complete and mae < best and baseline - mae > identity["selection_min_gain"]:
            best, selected = mae, identifier_
    return dict(
        status="complete" if complete else "incomplete",
        by_configuration_id=rows,
        selected_configuration_id=selected,
        reason="minimum_mae_strict_margin_baseline_on_tie" if complete else "incomplete_support",
    )


def run_registered_contrast(
    output,
    *,
    ledger,
    plan,
    evidence,
    parent,
    parent_codec,
    source,
    prefixes,
    retention,
    native,
    admission="m1",
    block_rows=64,
    selection_min_gain=0.0,
    regional_work_limits=None,
    resume=False,
):
    """Recorrer S con pesos emparejados y estados nuevos, sin ajustar ni ejecutar E."""
    _bindings(ledger, plan, evidence)
    _finite(selection_min_gain, "El margen de selección")
    require(type(parent_codec) is FrozenEpisodeCodec, "Falta el codec del padre")
    specification = _source(
        source,
        plan,
        parent,
        prefixes,
        retention,
        native,
        block_rows,
        admission,
        regional_work_limits,
    )
    identity = _identity(
        ledger,
        plan,
        evidence,
        source,
        parent,
        parent_codec,
        prefixes,
        retention,
        native,
        block_rows,
        admission,
        selection_min_gain,
        regional_work_limits,
    )
    contrast_id = fingerprint(identity)
    output = Path(output)
    for protected in (
        source.path.parent,
        source.dataset.path.parent,
        *source.dataset.roots.values(),
    ):
        outside_source(Path(protected), output)
        outside_source(output, Path(protected))
    with _locked(output, resume=resume):
        if resume:
            if (output / "identity.json").exists():
                saved, _ = read_manifest(output / "identity.json", MAX_STATE_BYTES)
                require(saved == identity, "El contraste no corresponde a su protocolo original")
            if (output / "state.json").exists():
                state = _read_state(output, identity)
            else:
                _recover_bootstrap(output)
                state = _initial_state(identity, contrast_id)
                _publish(output / "state.json", _seal(state))
        else:
            _publish(output / "identity.json", identity)
            state = _initial_state(identity, contrast_id)
            _publish(output / "state.json", _seal(state))
        trajectories = []
        for entry in state["arms"]:
            options = dict(
                ledger=ledger,
                parent=parent,
                parent_codec=parent_codec,
                specification=specification,
                source=source,
                prefixes=prefixes,
                retention=retention,
                native=native,
                block_rows=block_rows,
                admission=admission,
                contrast_id=contrast_id,
                regional_work_limits=regional_work_limits,
            )
            if entry["status"] != "completed" and len(entry["attempts"]) == MAX_ATTEMPTS:
                entry["receipt"], authoritative = _run_arm(
                    output, entry, plan=plan, audit_only=True, **options
                )
                if entry["status"] == "running":
                    entry["attempts"][-1].update(
                        status="interrupted",
                        error="Proceso interrumpido después del cierre nativo y antes del recibo",
                    )
                entry["status"] = "completed"
                _publish(output / "state.json", _seal(state))
                initialization = entry["receipt"]["initialization"]
            elif entry["status"] != "completed":
                require(
                    len(entry["attempts"]) < MAX_ATTEMPTS,
                    "La cuota de ejecución del brazo está agotada",
                )
                if entry["status"] == "running":
                    entry["attempts"][-1].update(
                        status="interrupted",
                        error="Proceso interrumpido antes de confirmar el recibo de coste",
                    )
                entry["status"] = "running"
                attempt = dict(status="running", elapsed_seconds=None, error=None)
                entry["attempts"].append(attempt)
                _publish(output / "state.json", _seal(state))
                started = time.perf_counter()
                try:
                    entry["receipt"], authoritative = _run_arm(output, entry, plan=plan, **options)
                except BaseException as error:
                    reason = (
                        (type(error).__name__ + ": " + str(error))
                        .encode()[:1024]
                        .decode(errors="ignore")
                    )
                    attempt.update(
                        status="failed",
                        elapsed_seconds=time.perf_counter() - started,
                        error=reason,
                    )
                    entry["status"] = "failed"
                    _publish(output / "state.json", _seal(state))
                    raise
                attempt.update(status="completed", elapsed_seconds=time.perf_counter() - started)
                entry["status"] = "completed"
                _publish(output / "state.json", _seal(state))
                initialization = entry["receipt"]["initialization"]
            else:
                authoritative, initialization = _audit_arm(output, entry, plan=plan, **options)
            trajectories.append(
                _stored_trajectory(
                    output,
                    entry,
                    authoritative=authoritative,
                    initialization=initialization,
                    plan=plan,
                )
            )
        comparison = _compare(
            trajectories, [entry["receipt"] for entry in state["arms"]], identity, plan
        )
        source._confirm()
        parent.verify(strong=True)
        confirmation = None
        if comparison["status"] == "complete":
            confirmation = dict(
                schema_version=1,
                kind="experimental_next_phase_confirmation_v1",
                contrast_id=contrast_id,
                expected_parent_consumer_id=parent.model_id,
                selected_configuration_id=comparison["selected_configuration_id"],
                trajectory_ids=[item.record()["trajectory_id"] for item in trajectories],
                evaluation_start=plan.evaluation_start,
                evaluation_end=plan.evaluation_end,
                evaluation_executed=False,
                canonical_default_changed=False,
            )
            confirmation["confirmation_id"] = fingerprint(confirmation)
        result = dict(
            schema_version=1,
            kind="registered_trajectory_contrast_v1",
            status="completed" if comparison["status"] == "complete" else "inconclusive",
            contrast_id=contrast_id,
            plan_id=plan.plan_id,
            registry_id=ledger.registry.registry_id,
            baseline_id=ledger.registry.baseline_id,
            initialization=INITIALIZATION,
            source=identity["source"],
            arms=[entry["receipt"] for entry in state["arms"]],
            comparison=comparison,
            confirmation=confirmation,
            evaluation_executed=False,
        )
        source._confirm()
        parent.verify(strong=True)
        if state["result"] is not None:
            require(state["result"] == result, "La recuperación cambió el resultado confirmado")
        if confirmation is not None:
            destination = output / "confirmation.json"
            if destination.exists():
                require(
                    read_manifest(destination, MAX_STATE_BYTES)[0] == confirmation,
                    "La confirmación existente pertenece a otro padre o resultado",
                )
            else:
                _publish(destination, confirmation)
        else:
            require(
                not (output / "confirmation.json").exists(),
                "El contraste inconcluyente conserva una confirmación incompatible",
            )
        state.update(status=result["status"], result=result)
        if not (output / "identity.json").exists():
            _publish(output / "identity.json", identity)
        _publish(output / "state.json", _seal(state))
        return result
