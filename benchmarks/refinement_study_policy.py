"""Selección por validación y evaluación posterior de una política de rutas congelada."""

import math
import time
from datetime import UTC, datetime
from pathlib import Path

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
from .refinement_study_capacity import MAX_TRACE_ROWS
from .refinement_study_traces import completed_phase, reconstruct


def selection_group(work, job):
    """Verificar ambos lectores de un núcleo antes de abrir sus resultados posteriores."""
    from mars_titan.posttraining.chronological_windows import _arm_parts

    from .refinement_study import jobs
    from .refinement_study_traces import reader_components

    selected = [
        item
        for item in jobs()
        if item["kind"] == "select"
        and (item["projection"], item["seed"]) == (job["projection"], job["seed"])
    ]
    ids = [item["id"] for item in selected]
    if {item["depth"] for item in selected} != {1, 4} or (
        "selection_jobs" in job and job["selection_jobs"] != ids
    ):
        raise ValueError("El grupo necesita los sellos K1 y K4 del mismo núcleo y semilla")
    paths = [work / "runs" / item["id"] / "phase.json" for item in selected]
    if any(not path.is_file() for path in paths):
        raise ValueError("Falta un sello de selección del grupo")
    expected_core = (work / "runs" / f"{job['projection']}-{job['seed']}" / "core").resolve()
    parent_view = work / "fold-017/view.json"
    digests = {}
    for item, path in zip(selected, paths, strict=True):
        seal = read_manifest(path)[0]
        if (
            seal.get("status") != "completed"
            or seal.get("selection_partition") != "validation"
            or seal.get("calibration_used_for_selection") is not False
            or seal.get("evaluation_used_for_selection") is not False
            or seal["identity"]["job"] != item
            or seal["identity"]["prepared_sha256"] != sha256(work / "prepared.json")
        ):
            raise ValueError("El sello no conserva la selección declarada solo con validación")
        completed_phase(path.parent, seal["identity"])
        arm = work / "runs" / item["reader"]
        reader, reader_sha, _, core, parent = _arm_parts(arm, parent_view)
        if (
            core.resolve() != expected_core
            or reader["request"]["seed"] != job["seed"]
            or reader["request"]["components"] != reader_components(item["depth"])
            or parent["request"]["seed"] != job["seed"]
            or parent["request"]["view_sha256"] != sha256(parent_view)
            or parent["request"]["recipe_sha256"]
            != sha256(work / "recipes" / f"{job['projection']}.json")
        ):
            raise ValueError("Los sellos no comparten el núcleo, proyección y semilla esperados")
        for partition in ("train", "validation"):
            trace_path = work / "runs" / item["parent"] / partition / "phase.json"
            trace = read_manifest(trace_path)[0]
            if (
                sha256(trace_path) != seal["identity"][f"{partition}_phase_sha256"]
                or trace["identity"]["origin"]["reader_run_sha256"] != reader_sha
            ):
                raise ValueError("El sello no conserva las trazas del lector elegido")
        digests[item["id"]] = sha256(path)
    return dict(
        scope="projection_and_seed",
        projection=job["projection"],
        seed=job["seed"],
        core_run_sha256=sha256(expected_core / "run.json"),
        selections=digests,
    )


def prior_parent_exposure(work, job):
    """La evaluación de 2022 ya publicada por el contrato original del núcleo y lector."""
    reader = read_manifest(work / "runs" / job["reader"] / "run.json")[0]
    parent = read_manifest(Path(reader["request"]["parent"]["path"]) / "run.json")[0]
    return dict(
        parent_fold="fold-017",
        published_evaluation_interval=["2022-01-01", "2023-01-01"],
        core_evaluation_sha256=parent["predictions"]["evaluation"]["sha256"],
        reader_evaluation_sha256=reader["predictions"]["evaluation"]["sha256"],
        evaluation_reports_used_for_policy_fit_or_selection=False,
        raw_2022_data_unread=False,
    )


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
    import torch

    from mars_titan.models.titans.refinement_controls import ThresholdRoutePolicy

    threshold = type(policy) is ThresholdRoutePolicy
    if not threshold and any(value.dtype != torch.float64 for _, value in policy.named_buffers()):
        raise ValueError("El ajuste ridge debe conservar sus buffers FP64")
    payload = dict(
        identity=policy.get_extra_state(),
        buffers={name: value.detach().cpu().tolist() for name, value in policy.named_buffers()},
        dtype=None if threshold else "float64",
    )
    if path.exists():
        if read_manifest(path)[0] != payload:
            raise ValueError("La política existente no coincide con el ajuste reproducido")
    else:
        atomic_json(path, payload)


def load_policy(path, *, device):
    import torch

    from mars_titan.models.titans.refinement_controls import ThresholdRoutePolicy
    from mars_titan.models.titans.refinement_policy import RoutePolicy

    value = read_manifest(path)[0]
    if value["identity"].get("policy") == "strict_feature_threshold_after_k1_v1":
        if value.get("dtype") is not None or value.get("buffers") != {}:
            raise ValueError("El control por umbral no contiene buffers ajustados")
        model = ThresholdRoutePolicy(
            **{key: value["identity"][key] for key in ("feature_name", "cutoff", "route")}
        )
    else:
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


def route_table(traces, route):
    """Una alternativa registrada, con las mismas identidades y etiquetas maduras."""
    from mars_titan.models.titans.refinement_routes import ROUTES

    if route not in ROUTES:
        raise ValueError("La ruta no pertenece a la colección")
    position = ROUTES.index(route)
    flows, times = traces["flow_id"].to_pylist(), traces["decision_at"].to_pylist()
    predictions = np.asarray(traces["predictions"].to_pylist(), dtype=np.float64)
    columns = dict(
        sample_id=[f"{flow}/{at}" for flow, at in zip(flows, times, strict=True)],
        asset_id=flows,
        market=[flow.split("/", 1)[0] for flow in flows],
        prediction_at=pa.array(times, type=pa.timestamp("us", tz="UTC")),
        prediction=predictions[:, position],
        target=traces["target"],
        zero=np.zeros(len(traces)),
    )
    quantiles = traces["quantiles"].to_pylist()
    if any(value is not None for value in quantiles):
        if any(value is None for value in quantiles):
            raise ValueError("La colección mezcla salidas con y sin cuantiles")
        values = np.asarray(quantiles, dtype=np.float64)[:, position]
        columns.update(zip(QUANTILE_COLUMNS, values.T, strict=True))
    return pa.table(columns)


def _traces(folder, partition):
    receipt = read_manifest(folder / partition / "phase.json")[0]
    if receipt.get("status") != "completed" or receipt["identity"]["partition"] != partition:
        raise ValueError("La colección no ha confirmado la fase solicitada")
    completed_phase(folder / partition, receipt["identity"])
    table = read_bounded_table(folder / partition / "traces.parquet", max_rows=MAX_TRACE_ROWS)
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


def predict_phase(
    work,
    job,
    dataset,
    source,
    output,
    *,
    policy=None,
    mode="policy",
    route=None,
    schedule=None,
    parent_identity,
    stop,
    device,
):
    """Reproducir la política efectiva, incluidos los grupos que siguen rutas distintas."""
    from mars_titan.training.titans_walk_forward import PredictionRows, checked_tables

    reader_job = dict(job, parent=job["reader"])
    engine, origin = reconstruct(
        work,
        reader_job,
        source,
        mode=mode,
        policy=policy,
        route=route,
        schedule=schedule,
        device=device,
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
    from mars_titan.training.refinement_inference import _model_digest

    before = tuple(_model_digest(model) for model in (engine.predictor, engine.readout, policy))
    started = time.perf_counter()
    metrics = engine.evaluate(source, stop=stop, rows=rows)
    elapsed = time.perf_counter() - started
    if before != tuple(
        _model_digest(model) for model in (engine.predictor, engine.readout, policy)
    ):
        raise ValueError("La fase modificó parámetros o contratos de los modelos")
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
    validation_table, validation = _traces(folder, "validation")
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
                        validation_route_counts=metrics["route_counts"],
                    )
                )
    selected = best_candidate(candidates)
    thresholds = select_thresholds(
        work, job, output, train, dataset, sources["validation"], stop=stop, device=device
    )
    from mars_titan.models.titans.refinement_routes import ROUTES

    target = validation_table["target"].to_numpy()
    weights = session_weights(validation_table["decision_at"].to_numpy())
    predictions = np.asarray(validation_table["predictions"].to_pylist(), dtype=np.float64)
    fixed = [
        dict(
            id=route,
            session_mae=float(np.average(np.abs(predictions[:, index] - target), weights=weights)),
            mean_refinements=int(route[1]),
        )
        for index, route in enumerate(ROUTES)
    ]
    baseline = np.asarray(validation_table["features"].to_pylist())[:, 0]
    scalars = [
        dict(
            alpha=alpha,
            session_mae=float(np.average(np.abs(alpha * baseline - target), weights=weights)),
        )
        for alpha in configuration()["scalar_contraction"]
    ]
    scalar = min(scalars, key=lambda row: row["session_mae"])
    artifacts = {}
    for row in candidates + thresholds:
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
        controls=dict(
            threshold=best_candidate(thresholds),
            threshold_candidates=thresholds,
            hashed=dict(
                counts=[int(selected["validation_route_counts"][route]) for route in ROUTES],
                seed=job["seed"] + configuration()["hash_seed_offset"],
                frequency_scope="all_validation_predictions_including_unresolved",
            ),
            fixed_route=best_candidate(fixed),
            fixed_candidates=fixed,
            scalar=scalar,
            scalar_candidates=scalars,
        ),
        selection_partition="validation",
        calibration_used_for_selection=False,
        evaluation_used_for_selection=False,
        prior_parent_exposure=prior_parent_exposure(work, job),
        artifacts=artifacts,
        final_test_opened=False,
        completed_at_utc=datetime.now(UTC).isoformat(),
    )
    atomic_json(output / "phase.json", receipt)
    return receipt


def select_thresholds(work, job, output, train, dataset, source, *, stop, device):
    """Umbrales disponibles después de K1, elegidos en una familia separada del ridge."""
    from mars_titan.models.titans.refinement_controls import FEATURES, ThresholdRoutePolicy

    spec = configuration()["threshold_control"]
    values = np.asarray(train["features"].to_pylist(), dtype=np.float64)[
        :, FEATURES.index(spec["feature"])
    ]
    records = []
    for fraction in spec["fit_quantiles"]:
        cutoff = float(np.quantile(values, fraction, method="linear"))
        for route in spec["routes"]:
            identifier = f"threshold-{len(records):02d}"
            local = output / identifier
            policy = ThresholdRoutePolicy(feature_name=spec["feature"], cutoff=cutoff, route=route)
            save_policy(local / "policy.json", policy)
            fitted = dict(
                feature=spec["feature"],
                fit_quantile=fraction,
                cutoff=cutoff,
                route=route,
                threshold_source="train_only",
            )
            atomic_json(local / "fit.json", fitted)
            receipt = predict_phase(
                work,
                job,
                dataset,
                source,
                local / "validation",
                policy=policy,
                parent_identity=dict(
                    candidate=identifier, policy_sha256=sha256(local / "policy.json")
                ),
                stop=stop,
                device=device,
            )
            metrics = receipt["metrics"]
            records.append(
                dict(
                    id=identifier,
                    session_mae=metrics["session_mae"],
                    mean_refinements=metrics["refinements"] / metrics["predictions"],
                    policy_sha256=sha256(local / "policy.json"),
                    validation_sha256=sha256(local / "validation/phase.json"),
                    **fitted,
                )
            )
    return records


def fit_calibrator(output, calibration_path):
    from mars_titan.calibration.conformal_quantiles import fit_conformal_quantiles

    calibration = read_bounded_table(calibration_path, max_rows=MAX_TRACE_ROWS)
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

    evaluation = read_bounded_table(evaluation_path, max_rows=MAX_TRACE_ROWS)
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
    """Ejecutar las fases posteriores tras verificar los dos sellos del grupo."""
    require_learning_allowed("la evaluación de desarrollo de la puerta seleccionada")
    group = selection_group(work, job)
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
    from .refinement_study_controls import reference_paths

    references = reference_paths(work, job)
    dataset, sources = _sources(work, ("calibration", "evaluation"))
    output = work / "runs" / job["id"]
    selected_sha = sha256(selected_folder / "phase.json")
    results = {}
    chosen_identity = dict(
        selection_sha256=selected_sha, policy_sha256=sha256(policy_path), selection_group=group
    )
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
    from .refinement_study_controls import calibrate_references, compare_controls, evaluate_controls

    controls = evaluate_controls(
        work, job, dataset, sources, output, selected_folder, seal, stop=stop, device=device
    )
    calibrate_references(output, references, seal["controls"]["scalar"]["alpha"])
    comparison = compare_controls(output, seal)
    from .memory_microstudy_analysis import score_table

    scores = score_table(pq.read_table(output / "calibrated-evaluation.parquet"))
    atomic_parquet_batches(output / "sessions.parquet", [scores.to_table()])
    result = dict(
        schema_version=1,
        status="completed",
        job=job,
        selection_sha256=selected_sha,
        selection_group=group,
        phases=results,
        calibration=calibrated,
        controls=controls,
        comparison=comparison,
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
