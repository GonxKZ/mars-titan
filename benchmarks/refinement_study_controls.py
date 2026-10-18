"""Reproducir controles fijados y comparar errores emparejados por sesión."""

import time

import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation.paired_comparisons import compare_series, delta
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.models.titans.refinement_controls import HashedRoutes
from mars_titan.models.titans.refinement_routes import ROUTES
from mars_titan.training.learning_hold import require_learning_allowed

from .memory_microstudy_analysis import score_table
from .refinement_study import configuration
from .refinement_study_policy import (
    _sources,
    apply_calibrator,
    fit_calibrator,
    load_policy,
    predict_phase,
)
from .refinement_study_traces import completed_phase


def frozen_reference(work, job, *, device="cuda:0", stop=None):
    """El predictor sin lector o M0 con lector propio, trasladados sin ajuste adicional."""
    from mars_titan.posttraining.chronological_windows import (
        _arm_parts,
        _best,
        _completed,
        _predictor,
        _readout_models,
        _write,
    )
    from mars_titan.training.financial_run import ChronologicalInference
    from mars_titan.training.mars_titan_run import MarsTitanInference
    from mars_titan.training.mars_titan_walk_forward import _carried_readout
    from mars_titan.training.refinement_inference import _model_digest
    from mars_titan.training.titans_walk_forward import KIND, PredictionRows, _verify, case_recipe

    require_learning_allowed("la referencia congelada del estudio de refinamientos")
    parent, output = work / "runs" / job["parent"], work / "runs" / job["id"]
    parent_view, view = work / "fold-017/view.json", work / "fold-018/view.json"
    path = output / "frozen.json"
    if path.exists():
        result = read_manifest(path)[0]
        if (
            result.get("final_test_opened") is not False
            or result["view_sha256"] != sha256(view)
            or result["parent"]["run_sha256"] != sha256(parent / "run.json")
        ):
            raise ValueError("La referencia congelada no conserva su padre y ventana")
        _verify(output, result)
        return dict(result, status="completed")
    dataset, sources = _sources(work, ("validation", "calibration", "evaluation"))
    if job["kind"] == "reference":
        report, digest = _completed(parent, (KIND,))
        _, state = _best(parent, report)
        document, request = report["identity"]["recipe"], report["request"]
        model = (
            _predictor(
                document,
                sources["validation"].specification(),
                request,
                device,
                state["model"],
                carried=True,
            )
            .eval()
            .requires_grad_(False)
        )
        inference = ChronologicalInference(model, case_recipe(document, request.get("search_case")))
        models = (model,)
    elif job["kind"] == "m0_reference":
        report, digest, family, core, core_report = _arm_parts(parent, parent_view)
        frozen, variant, codec, readout, recipe = _readout_models(
            family,
            report,
            core,
            core_report,
            sources["validation"].specification(),
            device,
            carried=True,
        )
        if variant.admission != "m0":
            raise ValueError("La referencia M0 necesita su propio lector sin banco")
        _, state = _best(parent, report)
        _carried_readout(state["model"], readout)
        readout.eval().requires_grad_(False)
        inference = MarsTitanInference(
            frozen,
            readout,
            recipe,
            admission="m0",
            retention=None,
            native=None,
            codec=codec,
            world=family.world,
            fold="fold-018",
        )
        models = (frozen, readout)
    else:
        raise ValueError("El trabajo no declara una referencia conocida")
    if (
        report["request"]["view_sha256"] != sha256(parent_view)
        or report["request"]["seed"] != job["seed"]
    ):
        raise ValueError("La referencia cambió su vista o semilla")
    model_digests = tuple(_model_digest(model) for model in models)
    predictions = {}
    for name, source in sources.items():
        local = output / name
        identity = dict(
            prepared_sha256=sha256(work / "prepared.json"),
            parent_sha256=digest,
            job=job,
            source=source.identity,
            models=list(model_digests),
        )
        receipt = completed_phase(local, identity)
        if receipt is None:
            atomic_json(local / "phase.json", dict(status="running", identity=identity))
            rows = PredictionRows(inference.quantiles)
            started = time.perf_counter()
            metrics = (
                inference.predict(source, rows, stop=stop)
                if job["kind"] == "reference"
                else inference.evaluate(source, rows=rows, stop=stop)
            )
            elapsed = time.perf_counter() - started
            if model_digests != tuple(_model_digest(model) for model in models):
                raise ValueError("La referencia modificó sus parámetros")
            written = _write(local, name, rows, metrics, dataset)
            receipt = dict(
                status="completed",
                identity=identity,
                prediction=written,
                process_seconds=elapsed,
                artifacts={written["path"]: written["sha256"]},
            )
            atomic_json(local / "phase.json", receipt)
        predictions[name] = dict(
            receipt["prediction"], path=name + "/" + receipt["prediction"]["path"]
        )
    result = dict(
        kind="refinement_study_frozen_reference",
        status="completed",
        parent=dict(
            path=str(parent.resolve()),
            run_sha256=digest,
            checkpoint_sha256=report["checkpoint"]["sha256"],
        ),
        predictions=predictions,
        view_sha256=sha256(view),
        final_test_opened=False,
        shared_parameters_unchanged=True,
    )
    atomic_json(path, result)
    return dict(result, status="completed")


def reference_paths(work, job):
    stem = f"{job['projection']}-{job['seed']}"
    result = {}
    for name, directory in (("core", "reference"), ("m0", "m0-reference")):
        folder = work / "runs" / stem / directory
        receipt = read_manifest(folder / "frozen.json")[0]
        parent = work / "runs" / stem / ("core" if name == "core" else "m0-reader")
        if (
            receipt.get("final_test_opened") is not False
            or receipt["view_sha256"] != sha256(work / "fold-018/view.json")
            or receipt["parent"]["path"] != str(parent.resolve())
            or receipt["parent"]["run_sha256"] != sha256(parent / "run.json")
        ):
            raise ValueError("La referencia no conserva el padre, semilla o vista declarados")
        for record in receipt["predictions"].values():
            if sha256(folder / record["path"]) != record["sha256"]:
                raise ValueError("Las predicciones de referencia han cambiado")
        result[name] = (folder, receipt)
    return result


def calibrate_references(output, references, scalar):
    """Calibrar cada referencia y la contracción fijada solo con su validación."""
    for name, (folder, receipt) in references.items():
        scales = ((name, 1.0), ("scalar", scalar)) if name == "core" else ((name, 1.0),)
        for control, alpha in scales:
            local = output / "controls" / control
            for phase in ("calibration", "evaluation"):
                table = pq.read_table(folder / receipt["predictions"][phase]["path"])
                for column in ("prediction", *QUANTILE_COLUMNS):
                    values = table[column].to_numpy() * alpha
                    table = table.set_column(
                        table.schema.get_field_index(column), column, pa.array(values)
                    )
                atomic_parquet_batches(local / f"{phase}-predictions.parquet", [table])
                if phase == "calibration":
                    calibrator = fit_calibrator(local, local / "calibration-predictions.parquet")
            apply_calibrator(local, calibrator, local / "evaluation-predictions.parquet")
            atomic_json(
                local / "reference.json",
                dict(source_sha256=sha256(folder / "frozen.json"), scale=alpha),
            )


def evaluate_controls(work, job, dataset, sources, output, selected_folder, seal, *, stop, device):
    """Comparar las seis rutas, un umbral, un hash y la consulta original por paso."""
    declarations = [(route, dict(mode="fixed", route=route)) for route in ROUTES]
    threshold_path = selected_folder / seal["controls"]["threshold"]["id"] / "policy.json"
    threshold = load_policy(threshold_path, device=device)
    hashed = seal["controls"]["hashed"]
    declarations.extend(
        [
            ("threshold", dict(mode="policy", policy=threshold)),
            (
                "hashed",
                dict(mode="hashed", schedule=HashedRoutes(tuple(hashed["counts"]), hashed["seed"])),
            ),
            ("per_step_k4", dict(mode="per_step")),
        ]
    )
    selected_sha = sha256(selected_folder / "phase.json")
    receipt = {}
    for name, arguments in declarations:
        local = output / "controls" / name
        identity = dict(selection_sha256=selected_sha, control=name)
        calibration = predict_phase(
            work,
            job,
            dataset,
            sources["calibration"],
            local / "calibration",
            parent_identity=identity,
            stop=stop,
            device=device,
            **arguments,
        )
        calibrator = fit_calibrator(local, local / "calibration/predictions.parquet")
        evaluation = predict_phase(
            work,
            job,
            dataset,
            sources["evaluation"],
            local / "evaluation",
            parent_identity=dict(identity, calibrator_sha256=sha256(local / "calibrator.json")),
            stop=stop,
            device=device,
            **arguments,
        )
        apply_calibrator(local, calibrator, local / "evaluation/predictions.parquet")
        receipt[name] = dict(calibration=calibration, evaluation=evaluation)
    return receipt


def compare_controls(output, seal):
    """Diferencias de MAE sobre la misma población, con bloques de fechas y costes separados."""
    table = pq.read_table(output / "calibrated-evaluation.parquet")
    policy = score_table(table)
    scores = dict(policy=policy, zero=score_table(table, zero=True))
    for name in (*ROUTES, "threshold", "hashed", "per_step_k4", "core", "m0", "scalar"):
        scored = score_table(
            pq.read_table(output / "controls" / name / "calibrated-evaluation.parquet"),
            policy.cohort_sha256,
        )
        scores[name] = scored
    for name, scored in scores.items():
        atomic_parquet_batches(
            output / "comparison" / f"{name}-sessions.parquet", [scored.to_table()]
        )
    series = {name: scored.series("mae") for name, scored in scores.items()}
    selected_fixed = seal["controls"]["fixed_route"]["id"]
    contrasts = {
        f"policy_minus_{name}": delta(name, "policy")
        for name in dict.fromkeys(
            (
                "k1",
                selected_fixed,
                "zero",
                "threshold",
                "hashed",
                "per_step_k4",
                "core",
                "m0",
                "scalar",
            )
        )
    }
    config = configuration()
    comparisons = {}
    for block in (config["bootstrap"]["block_length"], *config["sensitivity_block_lengths"]):
        comparisons[str(block)] = compare_series(
            series, contrasts, **(config["bootstrap"] | dict(block_length=block))
        )
    result = dict(
        status="completed",
        kind="paired_refinement_development_comparison",
        scientific_level="exploratory_development",
        cohort_sha256=policy.cohort_sha256,
        selected_fixed_route=selected_fixed,
        scores={name: scored.summary() for name, scored in scores.items()},
        comparisons=comparisons,
        missing_controls=[],
        limits=[
            "El contraste corresponde a una semilla y al lector declarado. "
            "No confirma generalización.",
            "Las frecuencias por hash se fijaron en validación. "
            "No garantizan el mismo coste en evaluación.",
            "K4 por paso cambia solo el calendario de consulta del lector elegido. "
            "No se ha vuelto a ajustar.",
            "Los intervalos se corrigen dentro de esta familia de contrastes, "
            "no entre todos los estudios.",
        ],
        final_test_opened=False,
    )
    atomic_json(output / "comparison/results.json", result)
    return result
