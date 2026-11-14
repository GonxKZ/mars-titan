"""Métricas de predicciones archivadas con el contrato estadístico del proyecto."""

import math
import re
from dataclasses import dataclass, replace

import numpy as np
import pyarrow as pa

from . import scale_diagnostics as scale
from .forecast_panel import ForecastPanel, SessionSeries
from .forecast_scores import score_sessions

_QUANTILE = re.compile(r"quantile_(\d{4})")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def panel_from_table(table):
    """Leer puntos y cuantiles sin construir ningún modelo."""
    names = table.column_names
    _require(len(names) == len(set(names)), "La tabla repite columnas")
    quantiles = []
    for name in names:
        if name.startswith("quantile_"):
            match = _QUANTILE.fullmatch(name)
            _require(match is not None, "Un nivel de cuantil no tiene el formato declarado")
            quantiles.append((int(match.group(1)) / 1000, name))
    quantiles.sort()
    return ForecastPanel.from_arrow(
        table,
        quantile_columns=[name for _, name in quantiles] if quantiles else None,
        levels=[level for level, _ in quantiles] if quantiles else None,
        markets=("US",),
    )


def check_partition(panel, cohort, partition):
    """Comprobar filas y límites temporales con los metadatos archivados de la cohorte."""
    expected = cohort["selection"]["counts"][partition]
    begin, end = (
        int(np.datetime64(value, "us").astype(np.int64)) for value in cohort["fold"][partition]
    )
    _require(panel.rows == expected, "Las filas no coinciden con la cohorte declarada")
    _require(
        np.all((panel.prediction_at >= begin) & (panel.prediction_at < end)),
        "Hay predicciones fuera del tramo temporal declarado",
    )


def fit_scale(calibration):
    """Estimar λ solo con la partición de calibración que recibe el consumidor."""
    return scale.market_shrinkage(calibration, calibration.prediction)


def calibrated_panel(evaluation, factors):
    """Aplicar λ al punto. La campaña no declara una transformación de sus cuantiles."""
    predicted = scale.apply_shrinkage(evaluation, evaluation.prediction, factors)
    predicted.setflags(write=False)
    return replace(evaluation, prediction=predicted, quantiles=None, levels=None)


def zero_panel(panel):
    """Referencia cero sobre exactamente las mismas filas y objetivos."""
    predicted = np.zeros(panel.rows)
    predicted.setflags(write=False)
    return replace(panel, prediction=predicted, quantiles=None, levels=None)


@dataclass(frozen=True)
class PredictionMetrics:
    """Resumen y estadísticos por sesión. No conserva los paneles de todas las filas."""

    summary: dict
    sessions: pa.Table
    series: dict


def measure(panel, *, magnitude, view="raw"):
    """Puntuar una lectura sin cambiar su población ni confundir correlación indefinida con cero.

    `magnitude=False` corresponde a los controles de orden. Sus puntuaciones brutas no son
    retornos, por lo que solo se conservan las correlaciones y los diagnósticos de amplitud.
    Las métricas del punto calibrado se obtienen en otra llamada con `view="calibrated"`.
    """
    _require(view in ("raw", "calibrated"), "La lectura debe ser raw o calibrated")
    _require(
        view != "calibrated" or panel.quantiles is None,
        "La calibración del punto no recalibra cuantiles",
    )
    with np.errstate(over="raise", invalid="raise"):
        scored = score_sessions(panel, rank_ic_min_assets=3)
        pearson, defined = scale.session_correlations(panel, panel.prediction, min_assets=3)
        summary = scored.summary(market_weighting="session")
        pearson_summary = dict(
            mean=math.fsum(pearson[defined]) / int(defined.sum()) if defined.any() else None,
            defined_sessions=int(defined.sum()),
            undefined_sessions=int((~defined).sum()),
            session_std=float(np.std(pearson[defined], ddof=1)) if defined.sum() > 1 else None,
            method="pearson_within_session",
        )
        summary.update(
            view=view,
            magnitude_comparable=magnitude,
            magnitude_reason=None
            if magnitude
            else "El control predice un orden, no una magnitud del residuo",
            pearson_ic=pearson_summary,
            prediction_diagnostics=dict(
                weighting="row",
                mean=float(np.mean(panel.prediction)),
                standard_deviation=float(np.std(panel.prediction)),
                minimum=float(np.min(panel.prediction)),
                maximum=float(np.max(panel.prediction)),
                exact_zero_fraction=float(np.mean(panel.prediction == 0)),
                target_standard_deviation=float(np.std(panel.target)),
            ),
        )
    sessions = scored.to_table().append_column("pearson_ic", pa.array(pearson, mask=~defined))
    series = {}
    if view == "raw":
        series["rank_ic"] = scored.series("rank_ic")
        series["pearson_ic"] = SessionSeries(
            "pearson_ic",
            False,
            panel.cohort_sha256,
            panel.markets,
            scored.session_market,
            scored.session_period,
            pearson,
            defined,
        )
    if magnitude:
        for name in ("mae", "mse"):
            metric = f"{name}_{view}"
            series[metric] = replace(scored.series(name), metric=metric)
    else:
        for point in (summary["point"], *summary["by_market"].values()):
            for name in point:
                if name not in ("sessions", "rank_ic", "rank_ic_sessions"):
                    point[name] = None
        summary.update(
            row_weighted=None,
            direction=None,
            selective_direction=None,
            quantiles=None,
            quantiles_reason="Un control de orden no declara una distribución del residuo",
        )
        sessions = sessions.select(
            ["market", "prediction_at", "samples", "rank_ic", "rank_ic_status", "pearson_ic"]
        )
    if view == "calibrated":
        summary["quantiles_reason"] = (
            "La calibración se aplica solo al punto, no a los cuantiles originales"
        )
    return PredictionMetrics(summary, sessions, series)
