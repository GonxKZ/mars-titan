"""Selección por validación y evaluación posterior de una política de rutas congelada."""

import math
import time
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.batches import atomic_parquet_batches, read_bounded_table
from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.models.quantile_head import LEVELS, QUANTILE_COLUMNS
from mars_titan.training.learning_hold import require_learning_allowed

from .refinement_study import configuration
from .refinement_study_traces import completed_phase, reconstruct


def session_weights(times):
    """Cada instante tiene peso total uno, con sus activos unidos en el mismo grupo US."""
    values = np.asarray(times)
    if values.ndim != 1 or values.dtype.kind not in "iu" or not len(values):
        raise ValueError("Los pesos necesitan instantes enteros de un tramo no vacío")
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def best_candidate(records):
    if not records or any(
        type(row.get(field)) not in (float, int) or not math.isfinite(row[field]) or row[field] < 0
        for row in records
        for field in ("session_mae", "mean_refinements")
    ):
        raise ValueError("Los candidatos necesitan error y trabajo finitos y no negativos")
    return min(records, key=lambda row: (row["session_mae"], row["mean_refinements"]))


def save_policy(path, policy):
    payload = dict(
        identity=policy.get_extra_state(),
        buffers={name: value.detach().cpu().tolist() for name, value in policy.named_buffers()},
        dtype="float64",
    )
    if path.exists():
        if read_manifest(path)[0] != payload:
            raise ValueError("La política existente no coincide con el ajuste reproducido")
    else:
        atomic_json(path, payload)


def load_policy(path, *, device):
    import torch

    from mars_titan.models.titans.refinement_policy import RoutePolicy

    value = read_manifest(path)[0]
    if value.get("dtype") != "float64":
        raise ValueError("La política necesita la precisión FP64 del ajuste ridge")
    model = RoutePolicy(
        **{
            name: torch.tensor(data, device=device, dtype=torch.float64)
            for name, data in value["buffers"].items()
        },
        min_gain=value["identity"]["min_gain"],
    )
    if model.get_extra_state() != value["identity"]:
        raise ValueError("La identidad de la política no conserva su contrato")
    return model


def _traces(folder, partition):
    receipt = read_manifest(folder / partition / "phase.json")[0]
    if receipt.get("status") != "completed" or receipt["identity"]["partition"] != partition:
        raise ValueError("La colección no ha confirmado la fase solicitada")
    completed_phase(folder / partition, receipt["identity"])
    table = read_bounded_table(folder / partition / "traces.parquet", max_rows=131072)
    if len(table) != receipt["rows"]:
        raise ValueError("Las trazas no conservan su número de filas")
    return table, receipt


def _sources(work, names):
    from mars_titan.training.corpus_inputs import CorpusDataset
    from mars_titan.training.titans_walk_forward import _sources as build_sources

    dataset = CorpusDataset(work / "fold-018/view.json", input_policy=HISTORICAL_MASKED)
    staged = read_manifest(work / "staged.json")[0]
    phases = {name: FinancialPhase(**staged["phases"][name]) for name in names}
    return dataset, build_sources(dataset, phases, work / "indices/policy")


def predict_phase(work, job, dataset, source, output, *, policy, parent_identity, stop, device):
    """Reproducir la política efectiva, incluidos los grupos que siguen rutas distintas."""
    from mars_titan.training.titans_walk_forward import PredictionRows, checked_tables

    reader_job = dict(job, parent=job["reader"])
    engine, origin = reconstruct(
        work, reader_job, source, mode="policy", policy=policy, device=device
    )
    identity = dict(
        prepared_sha256=sha256(work / "prepared.json"),
        source=source.identity,
        origin=origin,
        selection=parent_identity,
    )
    previous = completed_phase(output, identity)
    if previous is not None:
        return previous
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "phase.json", dict(status="running", identity=identity))
    rows = PredictionRows(engine.quantiles)
    started = time.perf_counter()
    metrics = engine.evaluate(source, stop=stop, rows=rows)
    elapsed = time.perf_counter() - started
    tables = checked_tables(rows, metrics, dataset, source.phase.partition)
    path = output / "predictions.parquet"
    count = atomic_parquet_batches(path, tables)
    if count != metrics["labels"] or metrics["updates"] != 0:
        raise ValueError("La fase de predicción no conserva sus filas o modifica parámetros")
    result = dict(
        status="completed",
        identity=identity,
        rows=count,
        metrics=metrics,
        process_seconds=elapsed,
        throughput_scored_rows_per_second=count / elapsed,
        timing_scope="full_phase_including_input_warmup_prediction_and_maturity",
        artifacts={"predictions.parquet": sha256(path)},
        shared_parameters_unchanged=True,
        final_test_opened=False,
        completed_at_utc=datetime.now(UTC).isoformat(),
    )
    atomic_json(output / "phase.json", result)
    return result


def select_policy(work, job, *, stop=None, device="cuda:0"):
    """Ajustar solo con trazas nuevas y seleccionar con la ejecución real de validación."""
    require_learning_allowed("el ajuste y selección de la puerta de refinamientos")
    import torch

    from mars_titan.training.refinement_policy_fit import fit_policy

    folder, output = work / "runs" / job["parent"], work / "runs" / job["id"]
    train, training = _traces(folder, "train")
    _, validation = _traces(folder, "validation")
    identity = dict(
        prepared_sha256=sha256(work / "prepared.json"),
        job=job,
        train_phase_sha256=sha256(folder / "train/phase.json"),
        validation_phase_sha256=sha256(folder / "validation/phase.json"),
    )
    previous = completed_phase(output, identity)
    if previous is not None:
        return previous
    if (
        training["identity"]["origin"]["reader_run_sha256"]
        != validation["identity"]["origin"]["reader_run_sha256"]
    ):
        raise ValueError("El ajuste y validación no conservan el mismo lector")
    dataset, sources = _sources(work, ("validation",))
    staged = read_manifest(work / "staged.json")[0]
    phase, proof = staged["phases"]["train"], staged["fit_rows_proof"]
    features = torch.tensor(train["features"].to_pylist(), device=device, dtype=torch.float64)
    predictions = torch.tensor(train["predictions"].to_pylist(), device=device, dtype=torch.float64)
    times = train["decision_at"].to_numpy()
    common = dict(
        features=features,
        predictions=predictions,
        targets=torch.tensor(train["target"].to_numpy(), device=device, dtype=torch.float64),
        decision_at=torch.tensor(times, device=device, dtype=torch.int64),
        maturity_at=torch.tensor(train["maturity_at"].to_numpy(), device=device, dtype=torch.int64),
        session_weights=torch.tensor(session_weights(times), device=device, dtype=torch.float64),
        parent_selected_before=proof["parent_labels_mature_until"],
        fit_start=phase["decision_start"],
        fit_end=phase["close_at"],
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "phase.json", dict(status="running", identity=identity))
    spec = configuration()["policy"]
    candidates = []
    for ridge in spec["ridge"]:
        for min_gain in spec["min_gain"]:
            for cost in spec["cost_per_extra_refinement"]:
                if stop is not None and stop.requested:
                    from mars_titan.training.financial_run import Paused

                    raise Paused
                identifier = f"candidate-{len(candidates):02d}"
                local = output / identifier
                extra_search = spec["cost_per_extra_search"]
                costs = [
                    cost,
                    cost + extra_search,
                    3 * cost,
                    3 * cost + extra_search,
                    3 * cost + extra_search,
                ]
                policy, fitted = fit_policy(
                    **common,
                    ridge=ridge,
                    costs=torch.tensor(costs, dtype=torch.float64, device=device),
                    min_gain=min_gain,
                )
                save_policy(local / "policy.json", policy)
                atomic_json(local / "fit.json", fitted)
                record = predict_phase(
                    work,
                    job,
                    dataset,
                    sources["validation"],
                    local / "validation",
                    policy=policy,
                    parent_identity=dict(
                        candidate=identifier, policy_sha256=sha256(local / "policy.json")
                    ),
                    stop=stop,
                    device=device,
                )
                metrics = record["metrics"]
                candidates.append(
                    dict(
                        id=identifier,
                        session_mae=metrics["session_mae"],
                        mean_refinements=metrics["refinements"] / metrics["predictions"],
                        mean_searches=metrics["searches"] / metrics["predictions"],
                        ridge=ridge,
                        min_gain=min_gain,
                        costs=costs,
                        policy_sha256=sha256(local / "policy.json"),
                        validation_sha256=sha256(local / "validation/phase.json"),
                    )
                )
    selected = best_candidate(candidates)
    artifacts = {}
    for row in candidates:
        for suffix in (
            "policy.json",
            "fit.json",
            "validation/phase.json",
            "validation/predictions.parquet",
        ):
            path = output / row["id"] / suffix
            artifacts[str(path.relative_to(output))] = sha256(path)
    receipt = dict(
        status="completed",
        identity=identity,
        candidates=candidates,
        selected=selected,
        selection_partition="validation",
        calibration_used_for_selection=False,
        evaluation_used_for_selection=False,
        artifacts=artifacts,
        final_test_opened=False,
        completed_at_utc=datetime.now(UTC).isoformat(),
    )
    atomic_json(output / "phase.json", receipt)
    return receipt


def fit_calibrator(output, calibration_path):
    from mars_titan.calibration.conformal_quantiles import fit_conformal_quantiles

    calibration = read_bounded_table(calibration_path, max_rows=131072)
    spec = configuration()["calibration"]
    q = np.column_stack([calibration[name].to_numpy() for name in QUANTILE_COLUMNS])
    record = fit_conformal_quantiles(
        calibration["target"].to_numpy(),
        q,
        np.asarray(calibration["market"].to_pylist()),
        levels=LEVELS,
        nominals=spec["nominals"],
        min_rows=spec["min_rows"],
    )
    record.update(calibration_predictions_sha256=sha256(calibration_path))
    path = output / "calibrator.json"
    if path.exists() and read_manifest(path)[0] != record:
        raise ValueError("El calibrador confirmado no coincide con su tramo de ajuste")
    if not path.exists():
        atomic_json(path, record)
    return record


def apply_calibrator(output, record, evaluation_path):
    from mars_titan.calibration.conformal_quantiles import apply_conformal_quantiles

    evaluation = read_bounded_table(evaluation_path, max_rows=131072)
    q_eval = np.column_stack([evaluation[name].to_numpy() for name in QUANTILE_COLUMNS])
    calibrated, widened = apply_conformal_quantiles(
        record, q_eval, np.asarray(evaluation["market"].to_pylist())
    )
    for index, name in enumerate(QUANTILE_COLUMNS):
        evaluation = evaluation.set_column(
            evaluation.schema.get_field_index(name), name, pa.array(calibrated[:, index])
        )
    atomic_parquet_batches(output / "calibrated-evaluation.parquet", [evaluation])
    return dict(calibrator=record, widened_rows=widened)


def evaluate_selected(work, job, *, stop=None, device="cuda:0"):
    """Abrir calibración y desarrollo de 2023 solo tras verificar el sello de selección."""
    require_learning_allowed("la evaluación de desarrollo de la puerta seleccionada")
    selected_folder = work / "runs" / job["parent"]
    seal = read_manifest(selected_folder / "phase.json")[0]
    if (
        seal.get("status") != "completed"
        or seal.get("selection_partition") != "validation"
        or seal.get("calibration_used_for_selection") is not False
        or seal.get("evaluation_used_for_selection") is not False
    ):
        raise ValueError("La selección de la política no está confirmada solo con validación")
    completed_phase(selected_folder, seal["identity"])
    if any(
        seal["identity"]["job"][field] != job[field] for field in ("reader", "projection", "seed")
    ):
        raise ValueError("El trabajo no conserva el padre y la semilla de la selección")
    policy_path = selected_folder / seal["selected"]["id"] / "policy.json"
    if sha256(policy_path) != seal["selected"]["policy_sha256"]:
        raise ValueError("La política elegida cambió después de la selección")
    policy = load_policy(policy_path, device=device)
    dataset, sources = _sources(work, ("calibration", "evaluation"))
    output = work / "runs" / job["id"]
    selected_sha = sha256(selected_folder / "phase.json")
    results = {}
    chosen_identity = dict(selection_sha256=selected_sha, policy_sha256=sha256(policy_path))
    results["calibration"] = predict_phase(
        work,
        job,
        dataset,
        sources["calibration"],
        output / "calibration",
        policy=policy,
        parent_identity=chosen_identity,
        stop=stop,
        device=device,
    )
    record = fit_calibrator(output, output / "calibration/predictions.parquet")
    results["evaluation"] = predict_phase(
        work,
        job,
        dataset,
        sources["evaluation"],
        output / "evaluation",
        policy=policy,
        parent_identity=dict(chosen_identity, calibrator_sha256=sha256(output / "calibrator.json")),
        stop=stop,
        device=device,
    )
    calibrated = apply_calibrator(
        output,
        record,
        output / "evaluation/predictions.parquet",
    )
    from .memory_microstudy_analysis import score_table

    scores = score_table(pq.read_table(output / "calibrated-evaluation.parquet"))
    atomic_parquet_batches(output / "sessions.parquet", [scores.to_table()])
    result = dict(
        schema_version=1,
        status="completed",
        job=job,
        selection_sha256=selected_sha,
        phases=results,
        calibration=calibrated,
        scores=scores.summary(),
        artifacts={
            name: sha256(output / name)
            for name in ("calibrator.json", "calibrated-evaluation.parquet", "sessions.parquet")
        },
        scientific_level="exploratory_development",
        final_test_opened=False,
    )
    atomic_json(output / "run.json", result)
    return result
