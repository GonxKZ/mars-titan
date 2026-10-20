"""Comparar errores de varias semillas sin multiplicar las observaciones temporales."""

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks.memory_microstudy_analysis import score_table
from benchmarks.neural_plasticity_analysis import (
    checked_scores,
    execution_cost,
    interpret,
    point_objective_contrasts,
    quantile_routing_contrasts,
    read_run,
    unity_control,
)
from benchmarks.neural_plasticity_study import verify_prepared
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.forecast_panel import SessionSeries, weighted_session_mean
from mars_titan.evaluation.paired_comparisons import compare_series, delta
from mars_titan.training.learning_hold import require_learning_allowed


def average_seeds(items, seeds):
    """Promediar MAE por sesión y registrar su dispersión entre las semillas declaradas."""
    if (
        not isinstance(seeds, list)
        or not 2 <= len(seeds) <= 8
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or set(items) != set(seeds)
    ):
        raise ValueError("Las semillas deben ser distintas, completas e identificadas")
    values = [items[seed] for seed in seeds]
    first = values[0]
    if (
        any(
            item.metric != "mae"
            or not item.loss
            or item.cohort_sha256 != first.cohort_sha256
            or item.markets != first.markets
            or any(
                not np.array_equal(getattr(item, name), getattr(first, name))
                for name in ("session_market", "session_period", "defined")
            )
            for item in values
        )
        or not first.defined.any()
    ):
        raise ValueError("Cada semilla necesita la misma población, sesiones y máscara de MAE")
    means = [
        weighted_session_mean(v.values, v.defined, v.session_market, len(v.markets), "session")
        for v in values
    ]
    series = SessionSeries.average(values)
    return series, dict(
        seeds=seeds,
        sessions=len(series.values),
        mean_session_mae=math.fsum(means) / len(means),
        seed_session_mae={str(seed): value for seed, value in zip(seeds, means, strict=True)},
        seed_sample_std=float(np.std(means, ddof=1)),
        seed_min=float(min(means)),
        seed_max=float(max(means)),
    )


def control_pair(work, reference, unit):
    """Contrastar pesos, inicialización y tres fases dentro de una misma semilla."""
    import torch

    pair = [(run, table, work / "runs" / job["id"]) for job, run, table in (reference, unit)]
    payloads = [
        torch.load(folder / run["checkpoint"]["path"], map_location="cpu", weights_only=True)
        for run, _, folder in pair
    ]
    result = unity_control(pair[0][1], pair[1][1], *payloads)
    result["phase_predictions_equal"] = {}
    for phase in ("validation", "calibration", "evaluation"):
        tables = [
            pq.read_table(folder / run["predictions"][phase]["path"]).sort_by("sample_id")
            for run, _, folder in pair
        ]
        result["phase_predictions_equal"][phase] = tables[0].equals(tables[1], check_metadata=False)
    result["passed"] = result["passed"] and all(result["phase_predictions_equal"].values())
    return result


def paired_initialization(work, records, seeds):
    """Verificar la identidad inicial por semilla, sin afirmar paridad tras el ajuste.

    Las rutas de checkpoints ya deben haber superado read_run. Solo se carga uno
    cada vez y no se conservan sus tensores en el resultado.
    """
    import torch

    by_seed = {}
    for seed in seeds:
        identities = {}
        for (name, current_seed), (job, run, _) in records.items():
            if current_seed != seed:
                continue
            path = work / "runs" / job["id"] / run["checkpoint"]["path"]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            digest = payload.get("identity", {}).get("initial_parameters_sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("El checkpoint no identifica sus parámetros iniciales")
            identities[name] = digest
            del payload
        if len(identities) < 2:
            raise ValueError("Faltan controles emparejados de la inicialización")
        by_seed[str(seed)] = dict(passed=len(set(identities.values())) == 1, identities=identities)
    return dict(
        kind="paired_initial_parameters",
        passed=all(value["passed"] for value in by_seed.values()),
        by_seed=by_seed,
    )


def analyze(work, output):
    require_learning_allowed("el análisis multisemilla del piloto neuronal")
    started = time.perf_counter()
    work, output = Path(work).resolve(), Path(output).resolve()
    outside_source(work, output)
    if output.exists():
        raise FileExistsError("El análisis necesita un directorio nuevo")
    prepared = verify_prepared(work)
    if prepared.get("study_version") not in (
        "quantile_routing_v2",
        "quantile_point_v1",
        "local_capture_v1",
    ):
        raise ValueError(
            "El analizador requiere un protocolo multisemilla de cuantiles identificado"
        )
    point_control = prepared["study_version"] == "quantile_point_v1"
    local_capture = prepared["study_version"] == "local_capture_v1"
    config = prepared["configuration"]
    seeds = config["seeds"]
    arms = list(dict.fromkeys(job["arm"] for job in prepared["jobs"]))
    expected = {(arm, seed) for arm in arms for seed in seeds}
    actual = [(job["arm"], job["seed"]) for job in prepared["jobs"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("El catálogo debe conservar todos los pares de configuración y semilla")
    runs, scores, controls, cohort, steps, zero = {}, {}, {}, None, set(), None
    control_names = {} if local_capture else config["validity_control"]
    for job in prepared["jobs"]:
        run, table, digest = read_run(work, prepared, job)
        current = checked_scores(table, cohort)
        cohort = current.cohort_sha256
        if not math.isclose(
            current.summary()["point"]["mae"],
            run["predictions"]["evaluation"]["metrics"]["session_mae"],
            rel_tol=1e-6,
            abs_tol=1e-12,
        ):
            raise ValueError("El MAE por sesión no reproduce el del ejecutor")
        if job["id"] in runs:
            raise ValueError("Un identificador de trabajo está repetido")
        scores[job["arm"], job["seed"]] = current
        steps.add(run["fit"]["global_step"])
        runs[job["id"]] = dict(
            arm=job["arm"],
            seed=job["seed"],
            run_sha256=digest,
            checkpoint_sha256=run["checkpoint"]["sha256"],
            predictions_sha256=run["predictions"]["evaluation"]["sha256"],
            best_epoch=run["fit"]["best_epoch"],
            optimizer_steps=run["fit"]["global_step"],
            summary=current.summary(),
        )
        if local_capture or job["arm"] in control_names.values():
            controls[job["arm"], job["seed"]] = job, run, table
        if zero is None:
            zero = score_table(table, cohort, zero=True)
    if len(steps) != 1:
        raise ValueError("Las configuraciones no conservan las mismas actualizaciones")
    if local_capture:
        validity = paired_initialization(work, controls, seeds)
    else:
        by_seed = {
            str(seed): control_pair(
                work,
                controls[control_names["reference"], seed],
                controls[control_names["unit"], seed],
            )
            for seed in seeds
        }
        validity = dict(passed=all(value["passed"] for value in by_seed.values()), by_seed=by_seed)
    models, series = {}, {"zero": zero.series("mae")}
    for arm in arms:
        series[arm], models[arm] = average_seeds(
            {seed: scores[arm, seed].series("mae") for seed in seeds}, seeds
        )
    options = dict(
        **config["bootstrap"], sensitivity_block_lengths=config["sensitivity_block_lengths"]
    )
    primary = compare_series(
        series,
        {"primary": delta(config["primary_reference"], config["primary_candidate"])},
        **options,
    )
    if local_capture:
        if set(arms) != {"capture_global", "capture_local", "capture_constant"}:
            raise ValueError("El contraste local necesita sus tres controles registrados")
        contrasts = {f"{arm}_minus_zero": delta("zero", arm) for arm in arms}
        contrasts.update(
            local_minus_constant=delta("capture_constant", "capture_local"),
            global_minus_constant=delta("capture_constant", "capture_global"),
        )
    else:
        contrasts = (
            point_objective_contrasts(arms) if point_control else quantile_routing_contrasts(arms)
        )
    secondary = compare_series(series, contrasts, **options)
    costs = execution_cost(work, prepared)
    paired = {
        str(seed): (
            models[config["primary_candidate"]]["seed_session_mae"][str(seed)]
            - models[config["primary_reference"]]["seed_session_mae"][str(seed)]
        )
        for seed in seeds
    }
    result = dict(
        schema_version=1,
        kind=(
            "local_capture_multiseed_development_results"
            if local_capture
            else "quantile_point_objective_multiseed_development_results"
            if point_control
            else "quantile_routing_multiseed_development_results"
        ),
        scientific_level="multi_seed_reused_development_pilot",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        script_sha256=sha256(Path(__file__)),
        prepared_sha256=sha256(work / "prepared.json"),
        configuration=config,
        models=models,
        runs=runs,
        zero=zero.summary(),
        primary=primary,
        secondary=secondary,
        **{("initialization_control" if local_capture else "unity_control"): validity},
        seed_variation=dict(
            paired_delta=paired,
            paired_sample_std=float(np.std(list(paired.values()), ddof=1)),
            scope="Dispersión entre estas semillas. El intervalo temporal usa su promedio fijo.",
        ),
        primary_numeric_interpretation=interpret(
            primary["contrasts"][0]["interval"], config["minimum_relevant_mae"]
        ),
        primary_interpretation=(
            interpret(primary["contrasts"][0]["interval"], config["minimum_relevant_mae"])
            if validity["passed"]
            else "atribucion_bloqueada_por_inicializacion"
            if local_capture
            else "atribucion_bloqueada_por_control_unitario"
        ),
        costs=costs,
        analysis_seconds=time.perf_counter() - started,
        final_test_opened=False,
        limits=[
            "Desarrollo reutilizado. No confirma superioridad general ni sobre el estado del arte.",
            "Se promedian errores por sesión. No se evalúa un ensemble de predicciones.",
            "El bootstrap temporal mantiene fijas las semillas observadas "
            "y no estima otras nuevas.",
            "La variación entre semillas se informa por separado, sin multiplicar las fechas.",
            "Se igualan actualizaciones, no tiempo. Los secundarios tienen intervalos simultáneos.",
        ],
    )
    if point_control:
        result["objective_controls"] = config["uncertainty_training"]
        result["limits"].append(
            "Las colas de los controles puntuales no se ajustan. Sus valores no constituyen "
            "cuantiles aprendidos y no permiten atribuir una mejora de incertidumbre."
        )
    output.mkdir(parents=True)
    for (arm, seed), values in scores.items():
        pq.write_table(
            values.to_table(), output / f"{arm}-{seed}-sessions.parquet", compression="zstd"
        )
    pq.write_table(zero.to_table(), output / "zero-sessions.parquet", compression="zstd")
    for name, values in series.items():
        if name != "zero":
            pq.write_table(
                pa.table(
                    dict(
                        session_market=values.session_market,
                        session_period=values.session_period,
                        mae=values.values,
                        defined=values.defined,
                    )
                ),
                output / f"{name}-mean-sessions.parquet",
                compression="zstd",
            )
    atomic_json(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.work, args.output)
    print(json.dumps(dict(models=len(result["models"]), runs=len(result["runs"]))))


if __name__ == "__main__":
    main()
