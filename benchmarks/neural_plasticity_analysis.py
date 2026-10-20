"""Analizar todas las configuraciones del piloto sobre las mismas sesiones."""

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from benchmarks.memory_microstudy_analysis import score_table
from benchmarks.neural_plasticity_study import FILES, verify_prepared
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.paired_comparisons import compare_series, delta, interaction
from mars_titan.training.learning_hold import require_learning_allowed
from mars_titan.training.titans_walk_forward import CHANGES, _protocol, _verify


def checked_scores(table, cohort=None):
    if "zero" not in table.column_names or np.any(table["zero"].to_numpy() != 0):
        raise ValueError("El control residual cero debe ser cero en todas sus filas")
    return score_table(table, cohort)


def secondary_contrasts(names):
    result = {f"{name}_minus_zero": delta("zero", name) for name in names}
    pairs = [
        ("mac_original", "write_coherence"),
        ("fixed_write", "write_coherence"),
        ("cascade_original", "cascade_write"),
        ("cascade_original", "cascade_capture"),
        ("cascade_write", "cascade_combined"),
        ("cascade_capture", "cascade_combined"),
        ("cascade_fixed_write", "cascade_write"),
        ("cascade_fixed_capture", "cascade_capture"),
        ("cascade_fixed_both", "cascade_combined"),
        ("cascade_original", "cascade_unity"),
    ]
    for base, candidate in pairs:
        result[f"{candidate}_minus_{base}"] = delta(base, candidate)
    result["interaction"] = interaction(
        "cascade_original", "cascade_write", "cascade_capture", "cascade_combined"
    )
    return result


def key_competition_contrasts(names):
    """Separar centrado, selección de entrada, coordenadas fijas y referencia cero."""
    result = {f"{name}_minus_zero": delta("zero", name) for name in names}
    for candidate in names:
        if candidate != "original":
            result[f"{candidate}_minus_original"] = delta("original", candidate)
    for base, candidate in (
        ("fixed", "competitive"),
        ("centered_fixed", "centered_competitive"),
        ("centered", "centered_competitive"),
        ("competitive", "centered_competitive"),
    ):
        result[f"{candidate}_minus_{base}"] = delta(base, candidate)
    result["interaction"] = interaction(
        "original", "centered", "competitive", "centered_competitive"
    )
    return result


def quantile_routing_contrasts(names):
    """Comparar cada ruta de realimentación, sus componentes y el control cero."""
    result = {f"{name}_minus_zero": delta("zero", name) for name in names}
    for candidate in names:
        if candidate != "original":
            result[f"{candidate}_minus_original"] = delta("original", candidate)
    for base in ("median_isolated", "encoder_isolated"):
        result[f"both_isolated_minus_{base}"] = delta(base, "both_isolated")
    result["interaction"] = interaction(
        "original", "median_isolated", "encoder_isolated", "both_isolated"
    )
    return result


def interpret(interval, margin):
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("El margen debe ser positivo y finito")
    if interval is None:
        return "intervalo_no_disponible"
    if (
        len(interval) != 2
        or not all(math.isfinite(x) for x in interval)
        or interval[0] > interval[1]
    ):
        raise ValueError("El intervalo no es válido")
    lower, upper = interval
    if upper < -margin:
        return "menor_error_en_desarrollo"
    if lower > 0:
        return "mayor_error_en_desarrollo"
    if upper < 0:
        return "inconcluyente_respecto_al_margen"
    return "inconcluyente"


def unity_control(reference, candidate, reference_state, candidate_state):
    """Comprobar el control unitario por observación y por tensor compartido."""
    import torch

    left = reference.sort_by("sample_id")
    right = candidate.sort_by("sample_id")
    tensors = [
        {
            key: value
            for key, value in state["state"]["model"].items()
            if isinstance(value, torch.Tensor)
        }
        for state in (reference_state, candidate_state)
    ]
    matching = (
        bool(tensors[0])
        and tensors[0].keys() == tensors[1].keys()
        and all(
            value.dtype == tensors[1][key].dtype and torch.equal(value, tensors[1][key])
            for key, value in tensors[0].items()
        )
    )
    predictions = left.equals(right, check_metadata=False)
    initial = (
        reference_state["identity"]["initial_parameters_sha256"]
        == candidate_state["identity"]["initial_parameters_sha256"]
    )
    return dict(
        passed=predictions and matching and initial,
        predictions_equal=predictions,
        shared_tensors_equal=matching,
        initial_parameters_equal=initial,
        shared_tensors=len(tensors[0]),
        tolerance=dict(absolute=0, relative=0),
    )


def read_run(work, prepared, job):
    folder = work / "runs" / job["id"]
    path = folder / "run.json"
    report = json.loads(path.read_text())
    request = report["request"]
    _, protocol_digest, _, fold = _protocol(
        work / FILES["protocol"], prepared["configuration"]["fold"], job["seed"]
    )
    if (
        report.get("status") != "completed"
        or report.get("final_test_opened") is not False
        or request["seed"] != job["seed"]
        or request["device"] != "cuda:0"
        or request["variant"] != "mac_online"
        or request["window"] != prepared["configuration"]["fold"]
        or request.get(CHANGES) != (job["changes"] or None)
        or request["view_sha256"] != prepared["artifacts"][FILES["view"]]
        or request["protocol_sha256"] != protocol_digest
        or request["recipe_sha256"] != prepared["artifacts"][FILES["recipe"]]
    ):
        raise ValueError(f"La ejecución {job['id']} no corresponde al piloto completado")
    if fold != dict(id=prepared["configuration"]["fold"], **prepared["configuration"]["window"]):
        raise ValueError("El protocolo no conserva los intervalos declarados")
    for module, digest in request["code"].items():
        if prepared["code"].get("src/" + module.replace(".", "/") + ".py") != digest:
            raise ValueError("La ejecución tiene otra revisión de código")
    for record in report["predictions"].values():
        if not (folder / record["path"]).resolve().is_relative_to(folder.resolve()):
            raise ValueError("Una predicción está fuera de su ejecución")
    _verify(folder, report)
    checkpoint = (folder / report["checkpoint"]["path"]).resolve()
    if (
        not checkpoint.is_relative_to(folder.resolve())
        or sha256(checkpoint) != report["checkpoint"]["sha256"]
    ):
        raise ValueError("El checkpoint seleccionado ha cambiado")
    prediction = report["predictions"]["evaluation"]
    table = pq.read_table(folder / prediction["path"])
    if len(table) != prepared["counts"]["evaluation"]:
        raise ValueError("La ejecución no conserva todas las filas declaradas")
    times = table["prediction_at"].cast("int64").to_numpy()
    first, end = prepared["configuration"]["window"]["evaluation"]
    start_us = np.datetime64(first, "us").astype(np.int64)
    end_us = np.datetime64(end, "us").astype(np.int64)
    if np.any(times < start_us) or np.any(times >= end_us) or end > "2024-01-01":
        raise ValueError("Las predicciones están fuera del desarrollo declarado")
    return report, table, sha256(path)


def execution_cost(work, prepared):
    plan = json.loads((work / "execution-plan.json").read_text())
    if plan["prepared_sha256"] != sha256(work / "prepared.json"):
        raise ValueError("El plan de ejecución no corresponde a la preparación")
    paths = [(path, True) for path in (work / "launches").glob("*.json")]
    for entry in plan.get("prior_receipts", []):
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("Un coste histórico ha cambiado")
        paths.append((path, False))
    records, identifiers, completed = [], set(), set()
    for path, current in paths:
        raw = json.loads(path.read_text())
        value = raw["wrapper_seconds"]
        if (
            raw.get("closed") is not True
            or raw.get("global_hold_unchanged") is not True
            or raw.get("final_test_opened") is not False
            or raw["id"] in identifiers
            or type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("Los costes no conservan intentos cerrados y distintos")
        identifiers.add(raw["id"])
        internal = raw.get("internal")
        if internal is None:
            if (
                raw["status"] != "failed"
                or not isinstance(raw.get("error"), str)
                or not raw["error"]
            ):
                raise ValueError("Un lanzamiento sin recibo interno necesita un fallo explícito")
        elif not isinstance(internal, dict) or internal.get("closure_confirmed") is not True:
            raise ValueError("Un intento interno no tiene cierre confirmado")
        if current and raw["status"] == "completed":
            if raw["prepared_sha256"] != plan["prepared_sha256"]:
                raise ValueError("Hay resultados completados de otra preparación")
            completed.add(raw["job"]["id"])
        records.append(
            dict(
                id=raw["id"],
                job_id=raw["job"]["id"],
                status=raw["status"],
                current_study=current,
                source_commit=raw["source_commit"],
                wrapper_seconds=value,
                internal_receipt_available=internal is not None,
                internal_seconds=internal["actual_seconds"] if internal is not None else None,
                optimizer_steps=internal["optimizer_steps"] if internal is not None else None,
                peak_rss_bytes=internal["peak_rss_bytes"] if internal is not None else None,
                peak_cuda_allocated_bytes=(internal or {}).get("peak_cuda_allocated_bytes"),
                peak_cuda_reserved_bytes=(internal or {}).get("peak_cuda_reserved_bytes"),
                receipt_sha256=sha256(path),
            )
        )
    if completed != {job["id"] for job in prepared["jobs"]}:
        raise ValueError("Faltan recibos de ejecución de alguna configuración")
    return dict(
        attempts=records,
        total_wrapper_seconds=sum(r["wrapper_seconds"] for r in records),
        timing_relation="Los tiempos internos están incluidos en los externos. No sumarlos.",
    )


def analyze(work, output):
    require_learning_allowed("el análisis del piloto neuronal")
    started = time.perf_counter()
    work, output = Path(work).resolve(), Path(output).resolve()
    outside_source(work, output)
    if output.exists():
        raise FileExistsError("El análisis necesita un directorio nuevo")
    prepared = verify_prepared(work)
    config = prepared["configuration"]
    validity_config = config.get("validity_control", {})
    control_names = (
        validity_config.get("reference", "cascade_original"),
        validity_config.get("unit", "cascade_unity"),
    )
    if len(config["seeds"]) != 1:
        raise ValueError("Este análisis describe un piloto de una semilla")
    models, scores, series, controls, cohort, steps = {}, {}, {}, {}, None, set()
    for job in prepared["jobs"]:
        run, table, digest = read_run(work, prepared, job)
        name = job["arm"]
        if name in models:
            raise ValueError("No se pueden convertir semillas en observaciones temporales")
        current = checked_scores(table, cohort)
        cohort = current.cohort_sha256
        if not math.isclose(
            current.summary()["point"]["mae"],
            run["predictions"]["evaluation"]["metrics"]["session_mae"],
            rel_tol=1e-6,
            abs_tol=1e-12,
        ):
            raise ValueError("El MAE por sesión no reproduce el del ejecutor")
        models[name] = dict(
            summary=current.summary(),
            job_id=job["id"],
            seed=job["seed"],
            run_sha256=digest,
            checkpoint_sha256=run["checkpoint"]["sha256"],
            predictions_sha256=run["predictions"]["evaluation"]["sha256"],
            best_epoch=run["fit"]["best_epoch"],
            optimizer_steps=run["fit"]["global_step"],
        )
        steps.add(run["fit"]["global_step"])
        if name in control_names:
            controls[name] = (run, table, work / "runs" / job["id"])
        scores[name], series[name] = current, current.series("mae")
        if "zero" not in scores:
            zero = score_table(table, cohort, zero=True)
            scores["zero"], series["zero"] = zero, zero.series("mae")
    if len(steps) != 1:
        raise ValueError("Las configuraciones no conservan las mismas actualizaciones")
    import torch

    pair = [controls[name] for name in control_names]
    payloads = [
        torch.load(folder / run["checkpoint"]["path"], map_location="cpu", weights_only=True)
        for run, _, folder in pair
    ]
    validity = unity_control(pair[0][1], pair[1][1], *payloads)
    validity["phase_predictions_equal"] = {}
    for phase in ("validation", "calibration", "evaluation"):
        tables = [
            pq.read_table(folder / run["predictions"][phase]["path"]).sort_by("sample_id")
            for run, _, folder in pair
        ]
        validity["phase_predictions_equal"][phase] = tables[0].equals(
            tables[1], check_metadata=False
        )
    validity["passed"] = validity["passed"] and all(validity["phase_predictions_equal"].values())
    primary = compare_series(
        series,
        {"primary": delta(config["primary_reference"], config["primary_candidate"])},
        **config["bootstrap"],
        sensitivity_block_lengths=config["sensitivity_block_lengths"],
    )
    contrasts, result_kind = {
        "key_competition_v1": (key_competition_contrasts, "key_competition_development_results"),
        "quantile_routing_v1": (quantile_routing_contrasts, "quantile_routing_development_results"),
    }.get(
        prepared.get("study_version"),
        (secondary_contrasts, "neural_plasticity_development_results"),
    )
    secondary = compare_series(
        series,
        contrasts(list(models)),
        **config["bootstrap"],
        sensitivity_block_lengths=config["sensitivity_block_lengths"],
    )
    costs = execution_cost(work, prepared)
    output.mkdir(parents=True)
    for name, values in scores.items():
        pq.write_table(values.to_table(), output / f"{name}-sessions.parquet", compression="zstd")
    report = dict(
        schema_version=1,
        kind=result_kind,
        scientific_level="single_seed_reused_development_pilot",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        script_sha256=sha256(Path(__file__)),
        prepared_sha256=sha256(work / "prepared.json"),
        configuration=config,
        models=models,
        zero=scores["zero"].summary(),
        primary=primary,
        unity_control=validity,
        primary_numeric_interpretation=interpret(
            primary["contrasts"][0]["interval"], config["minimum_relevant_mae"]
        ),
        primary_interpretation=(
            interpret(primary["contrasts"][0]["interval"], config["minimum_relevant_mae"])
            if validity["passed"]
            else "atribucion_bloqueada_por_control_unitario"
        ),
        secondary=secondary,
        costs=costs,
        analysis_seconds=time.perf_counter() - started,
        final_test_opened=False,
        uncertainty="Cuantiles del modelo sin recalibración posterior en este análisis",
        limits=[
            "Una semilla y desarrollo reutilizado. No demuestra superioridad general.",
            "Los contrastes secundarios son exploratorios y conservan intervalos simultáneos.",
            "Las sesiones se remuestrean por bloques. Las filas y las configuraciones "
            "no son réplicas temporales.",
            "La comparación iguala actualizaciones, no tiempo de cómputo.",
        ],
    )
    atomic_json(output / "results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.work, args.output)
    print(
        json.dumps(dict(models=len(result["models"]), analysis_seconds=result["analysis_seconds"]))
    )


if __name__ == "__main__":
    main()
