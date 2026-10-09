"""Segunda implementación de las métricas por sesión, escrita desde su especificación.

La comparación walk-forward calcula sus métricas con `evaluation.forecast_scores`, que
opera con índices de sesión y `bincount`. Este módulo las recalcula con agrupaciones de
pandas a partir de `docs/research/metrics.md`, sin importar ese código. Si las dos
implementaciones coinciden sobre las mismas predicciones crudas, un error de agregación,
de convención o de orden de columnas tendría que estar repetido en las dos para pasar
desapercibido.

Una sesión es el par (mercado, instante de decisión). Las convenciones son las declaradas:

- Dirección: los objetivos nulos no se juzgan y una predicción nula cuenta como fallo.
- Rank IC: correlación de Pearson de los rangos medios. Sin valor por
  `insufficient_assets`, `constant_target` o `constant_prediction`, en ese orden.
- Cuantiles: pinball por nivel, frecuencia de `y <= q`, intervalos centrales con extremos
  incluidos, anchura, puntuación de intervalo de Gneiting y Raftery (2007) y probabilidad
  implícita de subida con la regla
  `piecewise_linear_cdf_at_zero_flat_beyond_extreme_levels_v1`.

Las columnas de salida tienen los nombres de la tabla de sesiones de la comparación para
poder cotejarlas columna a columna.
"""

import math

import numpy as np
import pandas as pd

RANK_IC_STATUS = ("defined", "insufficient_assets", "constant_target", "constant_prediction")
SIGN_BINS = 10
COUNT_COLUMNS = (
    "samples",
    "direction_eligible",
    "direction_calls",
    "direction_hits",
    "positive_targets",
    "negative_targets",
    "up_calls",
    "down_calls",
    "up_hits",
    "down_hits",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def interval_pairs(levels):
    """Pares (tau, 1 - tau) con tau < 0,5, del intervalo más estrecho al más ancho."""
    pairs = [
        (low, high)
        for low in levels
        for high in levels
        if low < 0.5 and math.isclose(low + high, 1.0, abs_tol=1e-12)
    ]
    return sorted(pairs, key=lambda pair: pair[1] - pair[0])


def up_probability(quantiles, levels):
    """1 − F(0) con F lineal entre los puntos (q_j, tau_j) y plana fuera de los extremos.

    Se recorre cada fila buscando el último nivel cuyo cuantil no supera el cero. Si no hay
    ninguno, F(0) es el primer nivel. Si lo son todos, el último. En otro caso se interpola
    entre ese punto y el siguiente, que es estrictamente positivo. Con un cuantil igual a
    cero el término de interpolación vale cero y F toma el nivel de ese cuantil, que es el
    límite por la derecha.
    """
    quantiles = np.asarray(quantiles, dtype=np.float64)
    levels = list(levels)
    probability = np.empty(len(quantiles))
    for row, values in enumerate(quantiles):
        last = -1
        for index, value in enumerate(values):
            if value <= 0.0:
                last = index
        if last == -1:
            cdf = levels[0]
        elif last == len(levels) - 1:
            cdf = levels[-1]
        else:
            low, high = values[last], values[last + 1]
            cdf = levels[last] + (levels[last + 1] - levels[last]) * (0.0 - low) / (high - low)
        probability[row] = 1.0 - cdf
    return probability


def _rank_ic(group, min_assets):
    if len(group) < min_assets:
        return math.nan, "insufficient_assets"
    if group["target"].nunique() == 1:
        return math.nan, "constant_target"
    if group["prediction"].nunique() == 1:
        return math.nan, "constant_prediction"
    target = group["target"].rank(method="average").to_numpy()
    prediction = group["prediction"].rank(method="average").to_numpy()
    target, prediction = target - target.mean(), prediction - prediction.mean()
    denominator = math.sqrt(float(np.dot(target, target)) * float(np.dot(prediction, prediction)))
    return float(np.dot(target, prediction)) / denominator, "defined"


def session_scores(frame, *, levels=None, quantile_columns=None, rank_ic_min_assets=3):
    """Métricas por sesión de un panel con columnas `market`, `prediction_at`, `target` y
    `prediction`, y los cuantiles si se indican `levels` y `quantile_columns`.

    `prediction_at` son microsegundos UTC enteros. Devuelve un DataFrame ordenado por
    mercado e instante con una fila por sesión.
    """
    required = {"market", "prediction_at", "target", "prediction"}
    _require(required <= set(frame.columns), f"Faltan columnas: {sorted(required - set(frame))}")
    _require(
        (levels is None) == (quantile_columns is None),
        "Los niveles y las columnas de cuantiles se indican juntos",
    )
    _require(type(rank_ic_min_assets) is int and rank_ic_min_assets >= 3, "Mínimo de activos")
    data = pd.DataFrame(
        dict(
            market=frame["market"].astype(str).to_numpy(),
            prediction_at=np.asarray(frame["prediction_at"], dtype=np.int64),
            target=np.asarray(frame["target"], dtype=np.float64),
            prediction=np.asarray(frame["prediction"], dtype=np.float64),
        )
    )
    _require(np.isfinite(data[["target", "prediction"]].to_numpy()).all(), "Valores no finitos")
    target, prediction = data["target"], data["prediction"]
    error = prediction - target
    eligible = target != 0
    calls = eligible & (prediction != 0)
    data["abs_error"] = error.abs()
    data["sq_error"] = error * error
    data["direction_eligible"] = eligible.astype(np.int64)
    data["direction_calls"] = calls.astype(np.int64)
    data["direction_hits"] = (calls & (np.sign(prediction) == np.sign(target))).astype(np.int64)
    data["positive_targets"] = (target > 0).astype(np.int64)
    data["negative_targets"] = (target < 0).astype(np.int64)
    data["up_calls"] = (eligible & (prediction > 0)).astype(np.int64)
    data["down_calls"] = (eligible & (prediction < 0)).astype(np.int64)
    data["up_hits"] = (eligible & (prediction > 0) & (target > 0)).astype(np.int64)
    data["down_hits"] = (eligible & (prediction < 0) & (target < 0)).astype(np.int64)
    if levels is not None:
        levels = tuple(float(level) for level in levels)
        _require(len(levels) == len(quantile_columns), "Un nombre de columna por nivel")
        q = np.column_stack(
            [np.asarray(frame[name], dtype=np.float64) for name in quantile_columns]
        )
        _require(np.isfinite(q).all(), "Cuantiles no finitos")
        y = target.to_numpy()
        for j, level in enumerate(levels):
            u = y - q[:, j]
            data[f"pinball_{level}"] = np.where(u >= 0, level * u, (level - 1.0) * u)
            data[f"below_{level}"] = (y <= q[:, j]).astype(np.float64)
        for low_level, high_level in interval_pairs(levels):
            low, high = q[:, levels.index(low_level)], q[:, levels.index(high_level)]
            nominal = f"{round(high_level - low_level, 12):g}"
            alpha = 2.0 * low_level
            outside = np.where(y < low, low - y, 0.0) + np.where(y > high, y - high, 0.0)
            data[f"coverage_{nominal}"] = ((low <= y) & (y <= high)).astype(np.float64)
            data[f"width_{nominal}"] = high - low
            data[f"interval_score_{nominal}"] = (high - low) + (2.0 / alpha) * outside
        probability = up_probability(q, levels)
        up = (y > 0).astype(np.float64)
        weight = eligible.to_numpy().astype(np.float64)
        data["sign_weight"] = weight
        data["sign_squared"] = weight * (probability - up) ** 2
        bins = np.minimum(np.floor(probability * SIGN_BINS).astype(np.int64), SIGN_BINS - 1)
        for b in range(SIGN_BINS):
            inside = weight * (bins == b)
            data[f"bin_rows_{b}"] = inside
            data[f"bin_probability_{b}"] = inside * probability
            data[f"bin_up_{b}"] = inside * up
    keys = ["market", "prediction_at"]
    grouped = data.groupby(keys, sort=True)
    result = grouped.size().rename("samples").to_frame()
    result["mae"] = grouped["abs_error"].mean()
    result["mse"] = grouped["sq_error"].mean()
    for name in COUNT_COLUMNS[1:]:
        result[name] = grouped[name].sum()
    ranks = grouped[["target", "prediction"]].apply(
        lambda group: pd.Series(_rank_ic(group, rank_ic_min_assets), index=["ic", "status"])
    )
    result["rank_ic"] = ranks["ic"].astype(np.float64)
    result["rank_ic_status"] = ranks["status"]
    if levels is not None:
        quantile_names = [
            name
            for name in data.columns
            if name.startswith(("pinball_", "below_", "coverage_", "width_", "interval_score_"))
        ]
        for name in quantile_names:
            result[name] = grouped[name].mean()
        weight = grouped["sign_weight"].sum()
        result["sign_brier"] = (grouped["sign_squared"].sum() / weight).where(weight > 0)
        for prefix in ("bin_rows", "bin_probability", "bin_up"):
            result[f"sign_{prefix}"] = list(
                np.column_stack(
                    [grouped[f"{prefix}_{b}"].sum().to_numpy() for b in range(SIGN_BINS)]
                )
            )
    return result.reset_index()


def session_mean(values, defined, markets, weighting):
    """Media entre sesiones definidas con ponderación `session` o `market`."""
    values = np.asarray(values, dtype=np.float64)
    defined = np.asarray(defined, dtype=bool)
    markets = np.asarray(markets)
    _require(weighting in ("session", "market"), "Ponderación desconocida")
    if not defined.any():
        return None
    if weighting == "session":
        return float(values[defined].mean())
    means = [
        float(values[defined & (markets == name)].mean()) for name in np.unique(markets[defined])
    ]
    return float(np.mean(means))
