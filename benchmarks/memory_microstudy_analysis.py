"""Resumir el microensayo con las métricas y contrastes por sesión del proyecto."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation.forecast_panel import ForecastPanel, SessionSeries
from mars_titan.evaluation.forecast_scores import score_sessions
from mars_titan.evaluation.paired_comparisons import compare_series, delta, interaction
from mars_titan.models.quantile_head import LEVELS, QUANTILE_COLUMNS

from .memory_microstudy import CONFIG, configuration, jobs


def score_table(table, expected_cohort=None, *, zero=False):
    columns = (
        None if zero or not set(QUANTILE_COLUMNS) <= set(table.column_names) else QUANTILE_COLUMNS
    )
    panel = ForecastPanel.from_arrow(
        table,
        prediction="zero" if zero else "prediction",
        markets=("US",),
        quantile_columns=columns,
        levels=LEVELS if columns else None,
    )
    if expected_cohort is not None and panel.cohort_sha256 != expected_cohort:
        raise ValueError("La población o los objetivos no coinciden con la referencia")
    return score_sessions(panel)


def seed_means(series):
    """Promediar las tres semillas por fecha, sin convertirlas en fechas nuevas."""
    result = {}
    for name in ("mac_online", "pt1_init", "pt1"):
        keys = [f"{name}-{seed}" for seed in configuration()["seeds"]]
        if all(key in series for key in keys):
            result[name + "-mean"] = SessionSeries.average([series[key] for key in keys])
    return result


def contrasts(series):
    result = {f"{name}_minus_zero": delta("zero", name) for name in series if name != "zero"}
    pairs = [
        ("mac_online-mean", "pt1-mean"),
        ("pt1_init-mean", "pt1-mean"),
        ("mac_online-mean", "pt1_init-mean"),
        ("mac_frozen-42", "mac_online-42"),
        ("mac_disabled-42", "mac_online-42"),
        ("mac_online-42", "mac_paper-42"),
    ]
    pairs += [("cm_v1_b-42", name) for name in ("cm_v1_bc-42", "cm_v1_bm-42", "cm_v1_bcm-42")]
    pairs += [
        ("mac_online-42" if name == "post-full_continuation-42" else "frozen_parent-42", name)
        for name in series
        if name.startswith(("post-", "mars_"))
    ]
    for base, variant in pairs:
        if base in series and variant in series:
            result[f"{variant}_minus_{base}"] = delta(base, variant)
    factorial = ("cm_v1_b-42", "cm_v1_bc-42", "cm_v1_bm-42", "cm_v1_bcm-42")
    if all(name in series for name in factorial):
        result["cm_interaction"] = interaction(*factorial)
    return result


def analyze(work, output):
    work, output = Path(work), Path(output)
    if output.exists():
        raise FileExistsError("El análisis necesita un destino nuevo")
    prepared = json.loads((work / "prepared.json").read_text())
    if (
        prepared["configuration_sha256"] != sha256(CONFIG)
        or prepared["configuration"] != configuration()
        or prepared["jobs"] != jobs()
    ):
        raise ValueError("El análisis no usa la configuración preregistrada")
    output.mkdir(parents=True)
    models, series, pending, cohort, zero_added = {}, {}, [], None, False
    for job in prepared["jobs"]:
        folder = work / "runs" / job["id"]
        manifest = folder / ("window.json" if job["kind"] == "candidate" else "run.json")
        if not manifest.exists():
            pending.append(dict(job=job["id"], reason="Sin manifiesto final"))
            continue
        run = json.loads(manifest.read_text())
        if run.get("status") != "completed":
            pending.append(dict(job=job["id"], reason=run.get("status", "Sin estado")))
            continue
        if run.get("final_test_opened") is not False:
            raise ValueError("La ejecución no conserva la reserva final")
        record = run["predictions"]["evaluation"]
        path = (folder / record["path"]).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path) != record["sha256"]:
            raise ValueError("Las predicciones no corresponden a su ejecución")
        table = pq.read_table(path)
        if len(table) != prepared["counts"]["evaluation"]:
            raise ValueError("La evaluación no conserva todas las filas declaradas")
        scores = score_table(table, cohort)
        cohort = scores.cohort_sha256
        pq.write_table(scores.to_table(), output / (job["id"] + "-sessions.parquet"))
        models[job["id"]] = dict(
            summary=scores.summary(),
            run_sha256=sha256(manifest),
            predictions_sha256=sha256(path),
            seed=job["seed"],
            kind=job["kind"],
        )
        series[job["id"]] = scores.series("mae")
        if not zero_added:
            zero = score_table(table, cohort, zero=True)
            models["zero"] = dict(summary=zero.summary(), kind="zero_residual")
            series["zero"] = zero.series("mae")
            pq.write_table(zero.to_table(), output / "zero-sessions.parquet")
            zero_added = True
    if not models:
        raise ValueError("No hay ninguna ejecución completada que analizar")
    if any(name.startswith(("post-", "mars_")) for name in models):
        frozen_path = work / "frozen-parent/reference.json"
        frozen = json.loads(frozen_path.read_text())
        parent = json.loads((work / "runs/mac_online-42/run.json").read_text())
        if (
            frozen.get("control") != "frozen_parent"
            or frozen.get("optimizer_steps") != 0
            or frozen.get("shared_parameters_unchanged") is not True
            or frozen.get("checkpoint_sha256") != parent["checkpoint"]["sha256"]
        ):
            raise ValueError("La referencia congelada no conserva el padre seleccionado")
        prediction = (frozen_path.parent / frozen["predictions"]["path"]).resolve()
        if (
            not prediction.is_relative_to(frozen_path.parent.resolve())
            or sha256(prediction) != frozen["predictions"]["sha256"]
        ):
            raise ValueError("La referencia congelada no conserva sus predicciones")
        scores = score_table(pq.read_table(prediction), cohort)
        models["frozen_parent-42"] = dict(
            summary=scores.summary(),
            kind="frozen_runtime_parent",
            run_sha256=sha256(frozen_path),
            predictions_sha256=sha256(prediction),
        )
        series["frozen_parent-42"] = scores.series("mae")
        pq.write_table(scores.to_table(), output / "frozen_parent-42-sessions.parquet")
    series.update(seed_means(series))
    for name in series:
        if name.endswith("-mean"):
            keys = [f"{name.removesuffix('-mean')}-{seed}" for seed in configuration()["seeds"]]
            values = [models[key]["summary"]["point"]["mae"] for key in keys]
            models[name] = dict(
                kind="mean_of_three_seeds_by_session",
                seeds=configuration()["seeds"],
                mae=float(np.mean(values)),
                seed_mae=values,
                seed_standard_deviation=float(np.std(values, ddof=1)),
                sessions=len(series[name].values),
            )
    comparison = compare_series(series, contrasts(series), **configuration()["bootstrap"])
    report = dict(
        schema_version=1,
        kind="memory_microstudy_results",
        scientific_level="exploratory_microstudy",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        prepared_sha256=sha256(work / "prepared.json"),
        configuration_sha256=sha256(CONFIG),
        script_sha256=sha256(Path(__file__)),
        models=models,
        comparisons=comparison,
        pending=pending,
        final_test_opened=False,
        limits=[
            "Una época, cuatro activos y una ventana. "
            "No estima convergencia ni superioridad general.",
            "Los intervalos temporales y la dispersión entre semillas "
            "describen fuentes distintas de variación.",
            "La fase de postentrenamiento añade actualizaciones al padre, "
            "cuyo coste no desaparece.",
        ],
    )
    atomic_json(output / "results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.work, args.output)
    print(
        json.dumps(
            dict(models=len(report["models"]), pending=report["pending"]), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
