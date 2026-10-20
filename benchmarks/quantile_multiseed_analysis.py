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


def analyze(work, output):
    require_learning_allowed("el análisis multisemilla del piloto de cuantiles")
    started = time.perf_counter()
    work, output = Path(work).resolve(), Path(output).resolve()
    outside_source(work, output)
    if output.exists():
        raise FileExistsError("El análisis necesita un directorio nuevo")
    prepared = verify_prepared(work)
    if prepared.get("study_version") != "quantile_routing_v2":
        raise ValueError("El analizador requiere el protocolo multisemilla de cuantiles v2")
    config = prepared["configuration"]
    seeds = config["seeds"]
    arms = list(dict.fromkeys(job["arm"] for job in prepared["jobs"]))
    expected = {(arm, seed) for arm in arms for seed in seeds}
    actual = [(job["arm"], job["seed"]) for job in prepared["jobs"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("El catálogo debe conservar todos los pares de configuración y semilla")
    runs, scores, controls, cohort, steps, zero = {}, {}, {}, None, set(), None
    control_names = config["validity_control"]
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
        if job["arm"] in control_names.values():
            controls[job["arm"], job["seed"]] = job, run, table
        if zero is None:
            zero = score_table(table, cohort, zero=True)
    if len(steps) != 1:
        raise ValueError("Las configuraciones no conservan las mismas actualizaciones")
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
    secondary = compare_series(series, quantile_routing_contrasts(arms), **options)
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
        kind="quantile_routing_multiseed_development_results",
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
        unity_control=validity,
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
