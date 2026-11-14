"""Analizar resultados de los dos equipos usando solo sus ZIP, sin modelos ni dataset."""

import argparse
import csv
import math
import platform
import time
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.comparison_campaign import DONE, HISTORY, MACHINES, MAGNITUDE_RUNNERS, plan

from . import comparison_metrics as metrics
from . import comparison_protocol as protocol
from .comparison_bundle import ResultsBundle
from .comparison_inference import compare_family
from .forecast_panel import DAY_MICROSECONDS, SessionSeries


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _merge(bundles):
    first = bundles[0].manifest
    jobs = {}
    for bundle in bundles:
        current = bundle.manifest
        _require(
            current["provenance"]["training_commit"] == first["provenance"]["training_commit"],
            "Los paquetes proceden de distinto código o commit",
        )
        _require(
            current["provenance"] == first["provenance"],
            "Los paquetes no comparten declaración y cambios del código",
        )
        _require(
            current["cohorts"] == first["cohorts"], "Los paquetes no comparten las mismas cohortes"
        )
        for key, row in current["jobs"].items():
            _require(key not in jobs, f"El trabajo {key} está repetido en los paquetes")
            jobs[key] = bundle, row
    return jobs


def _scope(declaration, stage):
    _require(stage in ("all", "primary", "history"), "La etapa no es válida")
    origins = [
        row
        for row in declaration["origins"]
        if stage == "all" or (row["role"] == HISTORY) == (stage == "history")
    ]
    _require(origins, "La etapa no contiene orígenes")
    years = {row["year"] for row in origins}
    expected = {
        job.key: job
        for machine in MACHINES
        for job in plan(declaration, machine)
        if job.year in years
    }
    return origins, expected


def _resource_row(bundle, key, descriptor):
    record = bundle.read_json(descriptor["record"])
    attempts = record.get("attempts", [])

    def total(name):
        values = [row.get(name) for row in attempts]
        return (
            math.fsum(values)
            if values
            and all(isinstance(v, (float, int)) and math.isfinite(v) and v >= 0 for v in values)
            else None
        )

    resource = [row["resources"] for row in attempts if isinstance(row.get("resources"), dict)]

    def peak(name):
        values = [row[name] for row in resource if row.get(name) is not None]
        return max(values) if values else None

    whole = total("process_wall_seconds")
    cpu = [row.get("cpu_seconds") for row in resource]
    return dict(
        job=key,
        model=descriptor["model"],
        origin=descriptor["year"],
        seed=descriptor["seed"],
        machine=bundle.manifest["machine"],
        status=descriptor["status"],
        attempts=len(attempts),
        wall_seconds=whole if whole is not None else total("seconds"),
        wall_scope="full_process" if whole is not None else "reported_attempts_legacy",
        cpu_seconds=math.fsum(cpu)
        if len(resource) == len(attempts) and cpu and all(v is not None for v in cpu)
        else None,
        peak_rss_bytes=peak("peak_rss_bytes"),
        cuda_allocated_peak_bytes=peak("cuda_allocated_peak_bytes"),
        cuda_reserved_peak_bytes=peak("cuda_reserved_peak_bytes"),
        measured_attempts=len(resource),
        gpu_names=sorted({row["gpu_name"] for row in resource if row.get("gpu_name")}),
        energy_joules=None,
        energy_reason="No hay una medida integrada de energía",
        failures=[
            {k: row[k] for k in ("status", "error_type", "error", "traceback") if k in row}
            for row in attempts
            if row.get("error")
        ],
        log=descriptor.get("log"),
        parent_job=descriptor["needs"],
    )


def _curves(bundle, key, job):
    base = dict(job=key, model=job["model"], origin=job["year"])
    rows = []
    if job["fit_report"] is not None:
        fit = bundle.read_json(job["fit_report"])
        previous = -1
        for point in fit.get("history", []):
            epoch = point["epoch"]
            _require(
                type(epoch) is int and epoch > previous, "El historial repite o desordena épocas"
            )
            previous = epoch
            train, validation = point.get("train") or {}, point.get("validation") or {}
            rows.append(
                dict(
                    base,
                    seed=job["seed"],
                    axis="epoch",
                    step=epoch,
                    global_step=point.get("global_step"),
                    train_mae=train.get("session_mae"),
                    train_mse=train.get("session_mse"),
                    train_loss=train.get("mean_loss"),
                    validation_mae=validation.get("session_mae"),
                    validation_ic=None,
                    selected=epoch == fit.get("best_epoch"),
                )
            )
    for seed, name in job["seed_reports"].items():
        report = bundle.read_json(name)
        previous = 0
        for point in report.get("validation_history", []):
            values, defined = (
                np.asarray(point["values"], dtype=np.float64),
                np.asarray(point["defined"]),
            )
            _require(
                values.shape == defined.shape
                and defined.dtype == np.bool_
                and np.isfinite(values).all(),
                "La curva tabular no tiene valores y máscaras válidos",
            )
            rounds = point["rounds"]
            _require(type(rounds) is int and rounds > previous, "La curva tabular desordena rondas")
            previous = rounds
            rows.append(
                dict(
                    base,
                    seed=int(seed),
                    axis="round",
                    step=rounds,
                    train_loss=None,
                    train_mae=None,
                    train_mse=None,
                    validation_mae=None,
                    validation_ic=float(values[defined].mean()) if defined.any() else None,
                    validation_sessions=int(defined.sum()),
                    selected=rounds == report["rounds"],
                )
            )
    for name in job["trial_reports"].values():
        report = bundle.read_json(name)
        rows.append(
            dict(
                base,
                seed=None,
                axis="alpha",
                step=report["alpha"],
                train_mae=(report.get("train_metrics") or {}).get("session_mae"),
                validation_mae=report["predictions"]["validation"]["metrics"]["session_mae"],
                validation_ic=None,
                selected=None,
            )
        )
    return rows


def _save_sessions(output, model, year, seed, partition, view, measured):
    relative = Path("sessions") / model / str(year) / str(seed) / f"{partition}-{view}.parquet"
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(measured.sessions, path, compression="zstd")
    return relative.as_posix()


def _pipeline_costs(resources):
    """Contar cada trabajo una vez por cadena, sin sumar cadenas como coste de campaña."""
    indexed = {row["job"]: row for row in resources}
    for row in resources:
        chain, key, seen = [], row["job"], set()
        while key is not None and key in indexed:
            _require(key not in seen, "La cadena de dependencias contiene un ciclo")
            seen.add(key)
            chain.append(key)
            key = indexed[key]["parent_job"]
        times = [indexed[item]["wall_seconds"] for item in chain]
        row["pipeline_cost_jobs"] = list(reversed(chain))
        row["pipeline_wall_seconds"] = (
            math.fsum(times) if key is None and all(t is not None for t in times) else None
        )
        row["pipeline_cost_reason"] = (
            "Incluye los intentos del padre y del lector. "
            "No sumar entre lectores que comparten padre"
        )


def _measure_jobs(jobs, selected, output):
    per_seed, cache, reference, zero, times, resources, curves = [], {}, {}, {}, {}, [], []
    for key in sorted(selected):
        if key not in jobs:
            continue
        bundle, job = jobs[key]
        resources.append(_resource_row(bundle, key, job))
        curves.extend(_curves(bundle, key, job))
        if job["status"] != DONE:
            continue
        year, model = job["year"], job["model"]
        cohort = bundle.manifest["cohorts"][str(year)]
        for seed, tables in sorted(job["tables"].items()):
            row = dict(
                job=key,
                model=model,
                origin=year,
                seed=seed,
                partitions={},
                calibration_factors=None,
            )
            for partition in ("calibration", "validation", "evaluation"):
                panel = metrics.panel_from_table(bundle.read_table(tables[partition]))
                metrics.check_partition(panel, cohort, partition)
                identity = panel.cohort_sha256
                population = year, partition
                _require(
                    reference.setdefault(population, identity) == identity,
                    "Las filas u objetivos no coinciden entre modelos y semillas",
                )
                if partition == "calibration":
                    row["calibration_factors"] = metrics.fit_scale(panel)
                raw = metrics.measure(panel, magnitude=job["runner"] in MAGNITUDE_RUNNERS)
                part = dict(raw=raw.summary)
                part["raw_sessions"] = _save_sessions(
                    output, model, year, seed, partition, "raw", raw
                )
                if population not in zero:
                    zero[population] = metrics.measure(metrics.zero_panel(panel), magnitude=True)
                    _save_sessions(
                        output, "zero", year, "shared", partition, "raw", zero[population]
                    )
                part["zero_mae"] = zero[population].summary["point"]["mae"]
                if partition == "evaluation":
                    adjusted = metrics.calibrated_panel(panel, row["calibration_factors"])
                    calibrated = metrics.measure(adjusted, magnitude=True, view="calibrated")
                    part["calibrated"] = calibrated.summary
                    part["calibrated_sessions"] = _save_sessions(
                        output, model, year, seed, partition, "calibrated", calibrated
                    )
                    cache.setdefault((model, year), {})[seed] = dict(
                        raw.series, **calibrated.series
                    )
                    times[year] = (
                        raw.sessions["prediction_at"]
                        .to_numpy()
                        .astype("datetime64[us]")
                        .astype(np.int64)
                    )
                row["partitions"][partition] = part
            per_seed.append(row)
    _pipeline_costs(resources)
    return per_seed, cache, zero, times, resources, curves


def _mean(series):
    return float(np.mean(series.values[series.defined])) if series.defined.any() else None


def _model_metrics(declaration, origins, cache, zero):
    models, averaged, csv_rows = {}, {}, []
    for model, configuration in declaration["models"].items():
        required = (
            {"shared"}
            if configuration["runner"] in ("ridge", "rule")
            else {str(s) for s in declaration["seeds"]}
        )
        by_origin, pooled = {}, {}
        for origin in origins:
            year = origin["year"]
            cells = cache.get((model, year), {})
            if set(cells) != required:
                by_origin[str(year)] = dict(
                    complete=False, missing_seeds=sorted(required - set(cells))
                )
                continue
            names = set.intersection(*(set(cell) for cell in cells.values()))
            combined = {
                name: SessionSeries.average([cell[name] for cell in cells.values()])
                for name in names
            }
            averaged[model, year] = combined
            values = {
                name: _mean(combined[name]) if name in combined else None
                for name in protocol.METRICS
            }
            values["zero_mae"] = zero[year, "evaluation"].summary["point"]["mae"]
            for name in ("mae_raw", "mae_calibrated"):
                values[name + "_relative_percent"] = (
                    None
                    if values[name] is None or values["zero_mae"] == 0
                    else 100 * (values[name] / values["zero_mae"] - 1)
                )
            values.update(complete=True, seed_count=len(cells))
            by_origin[str(year)] = values
            csv_rows.append(dict(model=model, origin=year, **values))
        if all(row["complete"] for row in by_origin.values()):
            for metric in protocol.METRICS:
                items = [averaged[model, o["year"]].get(metric) for o in origins]
                valid = [(s.values[s.defined]) for s in items if s is not None]
                pooled[metric] = (
                    float(np.mean(np.concatenate(valid)))
                    if valid and sum(map(len, valid))
                    else None
                )
            pooled["zero_mae"] = float(
                np.mean(
                    np.concatenate(
                        [zero[o["year"], "evaluation"].series["mae_raw"].values for o in origins]
                    )
                )
            )
        models[model] = dict(by_origin=by_origin, pooled=pooled)
    for origin in origins:
        year = origin["year"]
        if (year, "evaluation") in zero:
            base = zero[year, "evaluation"].series["mae_raw"]
            averaged["zero", year] = dict(
                mae_raw=base, mae_calibrated=replace(base, metric="mae_calibrated")
            )
    return models, averaged, csv_rows


def _comparisons(options, origins, averaged, times):
    groups = {f"origin_{o['year']}": [o["year"]] for o in origins}
    for role in sorted({o["role"] for o in origins}):
        groups[role] = [o["year"] for o in origins if o["role"] == role]
    groups["all_selected_origins"] = [o["year"] for o in origins]
    result = {}
    settings = options["bootstrap"]
    for group, years in groups.items():
        years = sorted(years)
        days = np.concatenate([times[year] // DAY_MICROSECONDS for year in years])
        strata = np.concatenate([np.full(len(times[year]), year) for year in years])
        group_result = dict(origins=years, metrics={})
        for metric in options["metrics"]:
            families = {}
            for family, pairs in options["families"].items():
                members, excluded = [], []
                for variant, control in pairs:
                    label = f"{variant}/{control}"
                    if any(
                        metric not in averaged[model, year]
                        for model in (variant, control)
                        for year in years
                    ):
                        excluded.append(
                            dict(
                                name=label,
                                reason="La métrica no está definida para estos modelos",
                            )
                        )
                        continue
                    values, defined = [], []
                    for year in years:
                        a, b = averaged[variant, year][metric], averaged[control, year][metric]
                        _require(
                            a.cohort_sha256 == b.cohort_sha256, "El contraste no comparte población"
                        )
                        values.append(a.values - b.values)
                        defined.append(a.defined & b.defined)
                    members.append(
                        dict(
                            name=label,
                            values=np.concatenate(values),
                            defined=np.concatenate(defined),
                            direction="less" if metric.startswith("mae") else "greater",
                        )
                    )
                if not members:
                    families[family] = dict(
                        available=False, excluded=excluded, reason="No hay contrastes aplicables"
                    )
                    continue
                main = compare_family(
                    members, days, strata, block_length=settings["block_days"], settings=settings
                )
                sensitivity = {
                    str(length): compare_family(
                        members, days, strata, block_length=length, settings=settings
                    )
                    for length in settings["sensitivity_block_days"]
                }
                families[family] = dict(main=main, sensitivity=sensitivity, excluded=excluded)
            group_result["metrics"][metric] = families
        result[group] = group_result
    return result


def _write_outputs(output, report, summary_rows, curves, plots):
    _csv(
        output / "metrics.csv",
        summary_rows,
        [
            "model",
            "origin",
            "seed_count",
            *protocol.METRICS,
            "zero_mae",
            "mae_raw_relative_percent",
            "mae_calibrated_relative_percent",
        ],
    )
    _csv(
        output / "resources.csv",
        report["resources"],
        [
            "job",
            "model",
            "origin",
            "seed",
            "machine",
            "status",
            "attempts",
            "wall_seconds",
            "wall_scope",
            "parent_job",
            "pipeline_wall_seconds",
            "pipeline_cost_reason",
            "cpu_seconds",
            "peak_rss_bytes",
            "cuda_allocated_peak_bytes",
            "cuda_reserved_peak_bytes",
            "measured_attempts",
            "energy_joules",
            "energy_reason",
        ],
    )
    _csv(
        output / "learning-curves.csv",
        curves,
        [
            "job",
            "model",
            "origin",
            "seed",
            "axis",
            "step",
            "global_step",
            "train_mae",
            "train_mse",
            "train_loss",
            "validation_mae",
            "validation_ic",
            "validation_sessions",
            "selected",
        ],
    )
    atomic_json(output / "learning-curves.json", curves)
    if plots:
        from .comparison_plots import write_plots

        report["plots"] = write_plots(output / "plots", report, curves)
    else:
        report["plots"] = []
    incomplete = (
        "La matriz está completa para la etapa seleccionada."
        if report["complete"]
        else "La matriz está incompleta. Se describe sin inferencia de comparación."
    )
    (output / "README.md").write_text(
        "# Resultados de la campaña de comparación\n\n" + incomplete + "\n\n"
        "`report.json` contiene las métricas por semilla, partición y modelo, "
        "los contrastes y la procedencia. `metrics.csv` resume las medias por sesión "
        "de cada origen. `sessions/` conserva los estadísticos por sesión. "
        "`learning-curves.csv` y `learning-curves.json` conservan las curvas registradas. "
        "`resources.csv` describe los intentos y sus costes.\n\n"
        "Los valores vacíos no se sustituyen por cero. Los MAE brutos de controles "
        "de orden no son comparables. El cambio porcentual del MAE es una variación "
        "del error, no una rentabilidad. Las correlaciones indefinidas mantienen su máscara. "
        "Las semillas se promedian después de medir sus errores, sin formar "
        "un conjunto de predicciones.\n\n"
        "Los intervalos remuestrean bloques dentro de cada origen. "
        "2023 fue reutilizado y no es un test independiente. Los años anteriores a 2018 "
        "se interpretan retrospectivamente. La energía y las latencias no registradas "
        "siguen sin medir. El coste de cada lector se declara por separado del padre. "
        "Su coste completo incluye también el ajuste del padre.\n\n"
        "Este análisis no ejecuta modelos ni acredita rentabilidad de una política financiera.\n",
        encoding="utf-8",
    )


def analyze(paths, output, *, stage="all", allow_partial=False, plots=True):
    """Reunir uno o dos paquetes. La inferencia requiere todos los trabajos de su alcance."""
    _require(1 <= len(paths) <= 2, "Se necesitan uno o dos paquetes de resultados")
    output = Path(output)
    _require(not output.exists(), "El directorio de resultados ya existe")
    started = time.perf_counter()
    with ExitStack() as stack, threadpool_limits(limits=2):
        bundles = [stack.enter_context(ResultsBundle(path)) for path in sorted(paths, key=str)]
        jobs = _merge(bundles)
        declaration = bundles[0].declaration
        origins, expected = _scope(declaration, stage)
        missing = {
            key: jobs[key][1]["status"] if key in jobs else "missing_archive"
            for key in expected
            if key not in jobs or jobs[key][1]["status"] != DONE
        }
        _require(
            allow_partial or not missing,
            f"Faltan trabajos completos para el análisis: {len(missing)}",
        )
        supplied = declaration.get("analysis")
        options = protocol.checked(
            supplied or protocol.declaration(declaration["models"]), declaration["models"]
        )
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(
            output / "analysis-status.json",
            dict(status="running", started_at_utc=datetime.now(UTC).isoformat()),
        )
        try:
            per_seed, cache, zero, times, resources, curves = _measure_jobs(jobs, expected, output)
            models, averaged, rows = _model_metrics(declaration, origins, cache, zero)
            report = dict(
                schema_version=1,
                kind="comparison_campaign_offline_analysis_v1",
                complete=not missing,
                missing_jobs=missing,
                scope=dict(
                    stage=stage,
                    origins=[o["year"] for o in origins],
                    models=list(declaration["models"]),
                    seeds=declaration["seeds"],
                    expected_jobs=len(expected),
                ),
                provenance=dict(
                    bundles[0].manifest["provenance"],
                    archives=[
                        dict(name=b.path.name, sha256=sha256(b.path), machine=b.manifest["machine"])
                        for b in bundles
                    ],
                ),
                protocol=options,
                analysis_implementation=dict(
                    files={
                        path.name: sha256(path)
                        for name in (
                            "comparison_campaign_analysis",
                            "comparison_bundle",
                            "comparison_inference",
                            "comparison_metrics",
                            "comparison_protocol",
                            "comparison_plots",
                            "forecast_panel",
                            "forecast_scores",
                            "scale_diagnostics",
                            "paired_comparisons",
                            "variant_screening_analysis",
                        )
                        for path in (Path(__file__).with_name(name + ".py"),)
                    },
                    python=platform.python_version(),
                    packages={
                        name: version(name)
                        for name in ("numpy", "pyarrow", "scipy", "statsmodels", "matplotlib")
                    },
                ),
                analysis_specification_source="archive"
                if supplied is not None
                else "legacy_code_default",
                per_seed=per_seed,
                models=models,
                resources=resources,
                comparisons={} if missing else _comparisons(options, origins, averaged, times),
                inference_reason="La matriz está incompleta" if missing else None,
            )
            _write_outputs(output, report, rows, curves, plots)
            report["analysis_wall_seconds"] = time.perf_counter() - started
            report["finished_at_utc"] = datetime.now(UTC).isoformat()
            atomic_json(output / "report.json", report)
            atomic_json(
                output / "analysis-status.json", dict(status="completed", complete=not missing)
            )
            return report
        except BaseException as error:
            atomic_json(
                output / "analysis-status.json",
                dict(status="failed", error=f"{type(error).__name__}: {str(error)[:1000]}"),
            )
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--bundle",
        type=Path,
        action="append",
        required=True,
        help="ZIP de resultados de cada equipo",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Directorio nuevo para el análisis"
    )
    parser.add_argument("--stage", choices=("all", "primary", "history"), default="all")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Solo descripción si faltan trabajos, sin inferencia",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    report = analyze(
        args.bundle,
        args.output,
        stage=args.stage,
        allow_partial=args.allow_partial,
        plots=not args.no_plots,
    )
    print(f"Análisis escrito en {args.output}. Matriz completa: {report['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
