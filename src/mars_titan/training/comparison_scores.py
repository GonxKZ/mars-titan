"""Medidas de seguimiento de un trabajo terminado de la campaña de comparación (#836).

Un modelo puede ordenar bien los activos de una sesión y aun así perder contra la predicción
cero, porque con un residuo de media casi nula toda amplitud sin señal suma error. El MAE sin
más no distingue ese caso del de un modelo que no ha aprendido nada. Este módulo lee las tablas
de predicción que deja un trabajo y calcula, con el diagnóstico de escala del proyecto
(`evaluation.scale_diagnostics`), lo que la tabla de resultados enseña mientras la campaña
avanza:

- El MAE por sesión del modelo y el de la predicción cero con las mismas filas.
- El MAE por sesión de λ·predicción, con λ elegido en la partición de calibración, que es
  anterior a la evaluación, y aplicado sin cambios al año evaluado.
- El IC por sesión, de Pearson y de rangos, entre la predicción y el residuo.

No ajusta nada y no interviene en la selección de ningún modelo. Es un resumen para seguir la
campaña. El análisis con incertidumbre se hace al cerrarla, sobre las mismas tablas.
"""

import math
from pathlib import Path

import numpy as np

PARTITIONS = ("validation", "calibration", "evaluation")
MARKETS = ("US",)
_COLUMNS = ["sample_id", "market", "prediction_at", "target", "prediction"]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def prediction_tables(output, report):
    """Tablas de predicción de un trabajo terminado, un grupo por cada semilla que ajustó.

    Los ajustes, los lectores, Ridge y la regla declaran sus tres tablas en `predictions`.
    El control tabular ajusta todas las semillas en un trabajo y deja las tablas de cada una
    junto a su informe.
    """
    output = Path(output)
    predictions = report.get("predictions")
    try:
        if isinstance(predictions, dict):
            groups = [{name: output / predictions[name]["path"] for name in PARTITIONS}]
        else:
            groups = [
                {
                    name: (output / record["path"]).parent / f"{name}-predictions.parquet"
                    for name in PARTITIONS
                }
                for _, record in sorted(report.get("seeds", {}).items())
            ]
    except (KeyError, TypeError):
        # Un informe que no declara alguna de sus tablas no se puede medir.
        groups = []
    _require(
        groups and all(path.is_file() for group in groups for path in group.values()),
        "El trabajo no conserva sus tablas de predicción",
    )
    return groups


def _panel(path):
    import pyarrow.parquet as pq

    from mars_titan.evaluation.forecast_panel import ForecastPanel

    return ForecastPanel.from_arrow(pq.read_table(path, columns=_COLUMNS), markets=MARKETS)


def _mean(values):
    return math.fsum(values) / len(values)


def table_scores(tables, *, magnitude):
    """Medidas de un grupo de tablas (validación, calibración y evaluación) de una semilla.

    `magnitude` indica si la predicción es una magnitud del residuo. La regla y el control
    tabular predicen un orden, así que su MAE sin calibrar no es comparable y queda vacío.
    """
    from mars_titan.evaluation import scale_diagnostics as sd

    calibration, evaluation = _panel(tables["calibration"]), _panel(tables["evaluation"])
    measured = sd.diagnose_predictor(
        calibration,
        evaluation,
        fit_prediction=calibration.prediction,
        prediction=evaluation.prediction,
    )
    series = measured["series"]
    return dict(
        sessions=int(evaluation.sessions),
        rows=int(evaluation.rows),
        cohort_sha256=evaluation.cohort_sha256,
        zero=float(np.mean(series["zero"])),
        raw=float(np.mean(series["raw"])) if magnitude else None,
        calibrated=float(np.mean(series["shrunk"])),
        scale=measured["shrinkage"].get(MARKETS[0]),
        pearson_ic=measured["mean_pearson_ic"],
        rank_ic=measured["mean_rank_ic"],
    )


def job_scores(output, report, *, magnitude):
    """Medidas de un trabajo: las de su tabla o la media de las de sus semillas.

    Todas las semillas de un trabajo predicen las mismas filas, así que comparten sesiones,
    filas y predicción cero. Si no fuera así el trabajo no sería comparable y se rechaza.
    """
    scores = [
        table_scores(tables, magnitude=magnitude) for tables in prediction_tables(output, report)
    ]
    first = scores[0]
    _require(
        all(item["cohort_sha256"] == first["cohort_sha256"] for item in scores),
        "Las semillas del trabajo no predicen las mismas filas",
    )
    result = dict(
        seeds=len(scores),
        sessions=first["sessions"],
        rows=first["rows"],
        cohort_sha256=first["cohort_sha256"],
        zero=first["zero"],
    )
    for name in ("raw", "calibrated", "scale", "pearson_ic", "rank_ic"):
        values = [item[name] for item in scores]
        result[name] = None if any(value is None for value in values) else _mean(values)
    return result
