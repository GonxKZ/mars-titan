"""Comparación declarada de retención E2 con parámetros seleccionados comunes."""

import math
import platform
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import safe_destination
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.memory.spaced_study import load_study
from mars_titan.models.quantile_head import LEVELS, QUANTILE_COLUMNS
from mars_titan.training.learning_hold import require_learning_allowed

from .forecast_files import score_prediction_file
from .paired_comparisons import compare_series, delta
from .spaced_sources import PARTITIONS, load_sources


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _audit_sessions(source, scores):
    measured = scores.to_table().select(["market", "prediction_at", "samples", "mae", "mse"])
    stored = pq.read_table(source.report_path.parent / source.record["sessions"]["path"])
    expected = {"market", "prediction_at", "samples", "mae", "mse", "rmse"}
    _require(
        set(stored.column_names) == expected and len(stored) == len(measured),
        "La tabla de métricas por sesión E2 no conserva sus columnas y filas",
    )
    for name in expected:
        dtype = stored.schema.field(name).type
        valid = (
            (pa.types.is_string(dtype) or pa.types.is_large_string(dtype))
            if name == "market"
            else dtype == pa.timestamp("us", "UTC")
            if name == "prediction_at"
            else dtype == pa.int64()
            if name == "samples"
            else dtype == pa.float64()
        )
        _require(
            valid and stored[name].null_count == 0,
            "La tabla de sesiones E2 tiene tipos o ausencias inválidos",
        )
    order = [("market", "ascending"), ("prediction_at", "ascending")]
    stored, measured = stored.sort_by(order), measured.sort_by(order)
    _require(
        stored.select(["market", "prediction_at", "samples"]).equals(
            measured.select(["market", "prediction_at", "samples"])
        ),
        "Las sesiones E2 no corresponden a sus predicciones",
    )
    for name in ("mae", "mse", "rmse"):
        reference = (
            np.sqrt(measured["mse"].to_numpy()) if name == "rmse" else measured[name].to_numpy()
        )
        _require(
            np.allclose(stored[name].to_numpy(), reference, rtol=1e-12, atol=1e-14),
            "Las métricas guardadas no reproducen las predicciones E2",
        )
    metrics = source.record["metrics"]
    _require(
        type(source.record["sessions"].get("rows")) is int
        and source.record["sessions"]["rows"] == len(stored)
        and type(metrics.get("labels")) is int
        and metrics["labels"] == int(sum(scores.samples))
        and type(metrics.get("session_count")) is int
        and metrics["session_count"] == len(stored),
        "El resumen E2 pierde etiquetas o sesiones",
    )
    point = scores.summary()["point"]
    for field, name in (("session_mae", "mae"), ("session_mse", "mse")):
        value = metrics.get(field)
        _require(
            type(value) in (int, float)
            and math.isfinite(value)
            and math.isclose(value, point[name], rel_tol=1e-12, abs_tol=1e-14),
            "El resumen E2 no conserva el promedio por sesión",
        )


def _score_run(source, analysis, temporary, batch_size):
    names = pq.read_schema(source.prediction_path).names
    present = set(names) & set(QUANTILE_COLUMNS)
    _require(
        not present or present == set(QUANTILE_COLUMNS),
        "Las columnas de cuantiles E2 están incompletas",
    )
    quantiles = {source.arm: (QUANTILE_COLUMNS, LEVELS)} if present else None
    scores, resources = score_prediction_file(
        source.prediction_path,
        {source.arm: "prediction", "residual_zero": None},
        rows=source.record["rows"],
        minimum=analysis.rank_ic_min_assets,
        temporary=temporary,
        batch_size=batch_size,
        quantiles=quantiles,
    )
    chosen = scores[source.arm]
    _require(
        source.phase["decision_start"]
        <= int(chosen.session_time.min())
        <= int(chosen.session_time.max())
        < source.phase["decision_end"],
        "Las predicciones E2 quedan fuera de su fase declarada",
    )
    _audit_sessions(source, chosen)
    return scores, resources


def _contrasts(arms, *, include_zero):
    result = {
        f"{arm}_minus_baseline": delta("baseline", arm)
        for arm in arms
        if arm not in ("baseline", "utility")
    }
    result.update(
        {
            f"utility_minus_{arm}": delta(arm, "utility")
            for arm in arms
            if arm not in ("baseline", "utility")
        }
    )
    if include_zero:
        result.update(
            {
                f"{arm}_minus_residual_zero": delta("residual_zero", arm)
                for arm in ("baseline", "utility")
            }
        )
    return result


def _assessment(comparison, margin):
    interval = comparison["contrasts"][0]["interval"]
    if interval is None:
        return "inconcluyente"
    if interval[1] < -margin:
        return "menor_error_en_esta_ventana"
    if interval[0] > margin:
        return "mayor_error_en_esta_ventana"
    if -margin <= interval[0] <= interval[1] <= margin:
        return "intervalo_dentro_del_margen"
    return "inconcluyente"


def _comparisons(scores, study):
    options = study.analysis.bootstrap_options()
    primary = compare_series(
        {name: scores[name].series("mae") for name in ("baseline", "utility")},
        {study.analysis.primary_contrast: delta("baseline", "utility")},
        **options,
    )
    result = {}
    metrics = ["mae", "mse", "direction_accuracy", "rank_ic"]
    if scores["baseline"].levels is not None:
        metrics += ["pinball", "sign_brier"]
    for metric in metrics:
        quantile = metric in ("pinball", "sign_brier")
        series = {
            name: value.series(metric)
            for name, value in scores.items()
            if not quantile or name != "residual_zero"
        }
        result[metric] = compare_series(
            series, _contrasts(study.arms, include_zero=not quantile), **options
        )
    return dict(
        comparison=primary,
        min_relevant_mae=study.analysis.min_relevant_mae,
        assessment=_assessment(primary, study.analysis.min_relevant_mae),
    ), result


def analyze_spaced_runs(reports, study, *, partition, output, batch_size=4096):
    """Comparar una ventana y semilla E2, solo con permiso científico y declaración previa."""
    require_learning_allowed("la comparación científica por sesiones de E2")
    _require(partition in PARTITIONS, "El tramo E2 no pertenece al desarrollo permitido")
    _require(type(batch_size) is int and 1 <= batch_size <= 8192, "El lote Arrow E2 no es válido")
    study_path, output = Path(study), Path(output)
    safe_destination(study_path)
    safe_destination(output)
    _require(not output.exists(), "La salida de comparación E2 ya existe")
    study, study_sha = load_study(study_path)
    _require(
        study.analysis is not None,
        "La comparación E2 necesita un análisis declarado antes de la ventana",
    )
    sources = load_sources(reports, study, study_sha, partition, output)
    started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".e2-comparison-", dir=output.parent) as folder:
        scores, resources = {}, {}
        cohort, levels = None, None
        for source in sources:
            current, usage = _score_run(source, study.analysis, Path(folder), batch_size)
            value = current[source.arm]
            _require(
                cohort is None or cohort == value.cohort_sha256,
                "Los controles E2 no comparten la misma población",
            )
            if scores:
                _require(
                    levels == value.levels,
                    "Los controles E2 no comparten la misma salida de cuantiles",
                )
            cohort, levels = value.cohort_sha256, value.levels
            scores[source.arm] = value
            scores.setdefault("residual_zero", current["residual_zero"])
            resources[source.arm] = usage
        primary, comparisons = _comparisons(scores, study)
        for source in sources:
            source.verify()
        _require(sha256(study_path) == study_sha, "El estudio E2 cambió durante el análisis")
        tables = [
            s.to_table().append_column("condition", pa.array([name] * len(s.mae)))
            for name, s in scores.items()
        ]
        table = pa.concat_tables(tables, promote_options="default")
        output.mkdir()
        path = output / "sessions.parquet"
        pq.write_table(table, path, compression="zstd")
        result = dict(
            schema_version=1,
            kind="spaced_frozen_session_comparison_v1",
            status="completed",
            final_test_opened=False,
            reader_contract=study.reader_contract,
            partition=partition,
            study_sha256=study_sha,
            analysis=study.analysis.identity(),
            cohort_sha256=cohort,
            source={s.arm: s.provenance() for s in sources},
            resources=resources,
            primary=primary,
            comparisons=comparisons,
            scores={
                name: s.summary(market_weighting=study.analysis.market_weighting)
                for name, s in scores.items()
            },
            sessions=dict(path=path.name, sha256=sha256(path), rows=len(table)),
            software=dict(
                python=platform.python_version(),
                numpy=np.__version__,
                pyarrow=pa.__version__,
                duckdb=duckdb.__version__,
            ),
            code={
                name: sha256(Path(__file__).with_name(name + ".py"))
                for name in (
                    "spaced_comparison",
                    "spaced_sources",
                    "forecast_files",
                    "forecast_panel",
                    "forecast_scores",
                    "paired_comparisons",
                )
            },
            seconds=time.perf_counter() - started,
            limits=[
                "Una ventana y semilla no acreditan generalización ni superioridad.",
                "Se comparan políticas de retención con los mismos parámetros "
                "ya elegidos, sin reentrenar cada control.",
                "La identidad fija el análisis antes de emitir en el controlador. "
                "Los hashes no autentican un entrenamiento ni una fecha independiente.",
                "La disponibilidad histórica de las entradas originales requiere "
                "su auditoría separada.",
                "El contraste primario usa su intervalo marginal. Los secundarios "
                "tienen corrección simultánea dentro de cada métrica, "
                "sin corrección conjunta entre métricas.",
                "El margen de desarrollo está declarado y no es un umbral financiero validado.",
                "Los cuantiles proceden de las salidas existentes. El predictor "
                "residual cero no tiene una cabeza de incertidumbre.",
            ],
        )
        atomic_json(output / "comparison.json", result)
    return result
