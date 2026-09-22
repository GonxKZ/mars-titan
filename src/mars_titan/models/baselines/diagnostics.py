"""Diagnóstico de regresión sin interpretar el error como rentabilidad."""

from datetime import date

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(values)
        if np.iscomplexobj(array):
            raise ValueError(f"{name} debe contener valores reales")
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} debe ser un vector de valores reales") from error
    if array.ndim != 1 or not array.size:
        raise ValueError(f"{name} debe ser un vector unidimensional no vacío")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} debe contener valores finitos")
    return array


def _aligned_vectors(target: ArrayLike, prediction: ArrayLike, reference: ArrayLike | None):
    target = _vector(target, "target")
    prediction = _vector(prediction, "prediction")
    reference = np.zeros_like(target) if reference is None else _vector(reference, "reference")
    if target.size != prediction.size or target.size != reference.size:
        raise ValueError("target, prediction y reference deben tener la misma longitud")
    return target, prediction, reference


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def _pearson(target: NDArray[np.float64], prediction: NDArray[np.float64]) -> float:
    target = target / np.max(np.abs(target))
    prediction = prediction / np.max(np.abs(prediction))
    target = target - np.mean(target)
    prediction = prediction - np.mean(prediction)
    value = np.dot(target / np.linalg.norm(target), prediction / np.linalg.norm(prediction))
    return float(np.clip(value, -1, 1))


def _correlations(target: NDArray[np.float64], prediction: NDArray[np.float64]):
    reason = None
    if target.size < 2:
        reason = "Se necesitan al menos dos observaciones"
    elif np.all(target == target[0]):
        reason = "El objetivo es constante"
    elif np.all(prediction == prediction[0]):
        reason = "La predicción es constante"
    if reason is not None:
        return {
            "pearson": None,
            "pearson_reason": reason,
            "spearman": None,
            "spearman_reason": reason,
        }
    return {
        "pearson": _pearson(target, prediction),
        "pearson_reason": None,
        "spearman": _pearson(_average_ranks(target), _average_ranks(prediction)),
        "spearman_reason": None,
    }


def _direction(target: NDArray[np.float64], prediction: NDArray[np.float64]):
    nonzero_target = target != 0
    active = nonzero_target & (prediction != 0)
    evaluated_n = int(np.count_nonzero(active))
    nonzero_target_n = int(np.count_nonzero(nonzero_target))
    correct_n = int(np.count_nonzero(active & (np.sign(target) == np.sign(prediction))))
    return {
        "accuracy": correct_n / evaluated_n if evaluated_n else None,
        "accuracy_reason": None
        if evaluated_n
        else "No hay pares con objetivo y predicción no nulos",
        "evaluated_n": evaluated_n,
        "correct_n": correct_n,
        "nonzero_target_n": nonzero_target_n,
        "target_zero_n": int(target.size) - nonzero_target_n,
        "prediction_zero_n": int(np.count_nonzero(prediction == 0)),
        "abstained_nonzero_target_n": nonzero_target_n - evaluated_n,
        "coverage": evaluated_n / nonzero_target_n if nonzero_target_n else None,
        "coverage_reason": None if nonzero_target_n else "No hay objetivos no nulos",
    }


def point_diagnostics(
    target: ArrayLike, prediction: ArrayLike, reference: ArrayLike | None = None
) -> dict[str, object]:
    """Calcula errores por fila y una comparación pareada con una referencia.

    El sesgo es predicción menos objetivo. Las diferencias pareadas son el
    error del modelo menos el de la referencia, por lo que un valor negativo
    favorece al modelo. La habilidad relativa es uno menos el cociente de
    errores. La referencia predice cero salvo que se proporcione otra.

    La dirección se evalúa cuando objetivo y predicción son distintos de cero.
    La cobertura divide esos pares entre los objetivos no nulos. Una predicción
    cero cuenta como abstención. Los cuantiles usan interpolación lineal.
    Las métricas indefinidas devuelven None y su motivo. Un desbordamiento
    numérico genera ValueError para evitar un resultado JSON no finito.
    """
    reference_kind = "zero" if reference is None else "provided"
    target, prediction, reference = _aligned_vectors(target, prediction, reference)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            error = prediction - target
            absolute_error = np.abs(error)
            squared_error = np.square(error)
            reference_absolute_error = np.abs(reference - target)
            reference_squared_error = np.square(reference_absolute_error)
            mae, mse = float(np.mean(absolute_error)), float(np.mean(squared_error))
            reference_mae = float(np.mean(reference_absolute_error))
            reference_mse = float(np.mean(reference_squared_error))
            quantiles = np.quantile(absolute_error, [0.5, 0.9, 0.95, 0.99], method="linear")
            r2, r2_reason = None, "El objetivo es constante"
            if not np.all(target == target[0]):
                scale = np.max(np.abs(target))
                r2 = float(1 - np.mean(np.square(error / scale)) / np.var(target / scale))
                r2_reason = None
            result = {
                "n": int(target.size),
                "mae": mae,
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "bias": float(np.mean(error)),
                "median_absolute_error": float(quantiles[0]),
                "q90_absolute_error": float(quantiles[1]),
                "q95_absolute_error": float(quantiles[2]),
                "q99_absolute_error": float(quantiles[3]),
                "max_absolute_error": float(np.max(absolute_error)),
                "quantile_method": "linear",
                "r2": r2,
                "r2_reason": r2_reason,
                **_correlations(target, prediction),
                "direction": _direction(target, prediction),
                "reference": {
                    "kind": reference_kind,
                    "mae": reference_mae,
                    "mse": reference_mse,
                    "mae_skill": float(1 - np.divide(mae, reference_mae))
                    if reference_mae
                    else None,
                    "mae_skill_reason": None if reference_mae else "El MAE de referencia es cero",
                    "mse_skill": float(1 - np.divide(mse, reference_mse))
                    if reference_mse
                    else None,
                    "mse_skill_reason": None if reference_mse else "El MSE de referencia es cero",
                    "paired_mae_difference": float(
                        np.mean(absolute_error - reference_absolute_error)
                    ),
                    "paired_mse_difference": float(
                        np.mean(squared_error - reference_squared_error)
                    ),
                },
            }
    except FloatingPointError as error:
        raise ValueError("Los errores exceden el rango numérico de float64") from error
    return result


def _date_groups(dates: ArrayLike, n: int):
    try:
        dates = np.asarray(dates)
        if dates.ndim != 1 or dates.size != n:
            raise ValueError("dates debe ser un vector de la misma longitud que target")
        if dates.dtype.kind not in "MOSU" or (
            dates.dtype.kind == "O"
            and any(not isinstance(value, (str, bytes, date, np.datetime64)) for value in dates)
        ):
            raise ValueError("dates debe contener fechas")
        days = np.asarray(dates, dtype="datetime64[D]")
        if np.any(np.isnat(days)):
            raise ValueError("dates contiene fechas ausentes")
    except (TypeError, ValueError) as error:
        raise ValueError("dates debe contener fechas válidas y una por cada observación") from error
    return np.unique(days, return_inverse=True, return_counts=True)


def paired_date_bootstrap(
    dates: ArrayLike,
    target: ArrayLike,
    prediction: ArrayLike,
    reference: ArrayLike | None = None,
    *,
    block_length: int = 5,
    repetitions: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    """Estima un intervalo exploratorio del MAE del modelo menos la referencia.

    Ordena las fechas y remuestrea bloques móviles de fechas consecutivas
    observadas. Cada fecha seleccionada conserva todas sus filas. Concatena
    los bloques y trunca el último hasta obtener el número original de fechas.
    Las medias siguen ponderadas por fila, también con paneles desequilibrados.
    Las entradas deben identificar fechas civiles coherentes entre sí.

    Usa los percentiles 2,5 y 97,5 de las diferencias pareadas. No presupone
    independencia entre activos de la misma fecha ni corrige la selección de
    modelos. Limita las repeticiones a 10 000 y las fechas remuestreadas a diez
    millones. Mantiene en memoria un remuestreo cada vez, sin copiar sus filas.
    """
    for name, value, lower, upper in (
        ("block_length", block_length, 1, None),
        ("repetitions", repetitions, 2, 10_000),
        ("seed", seed, 0, None),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < lower
            or (upper is not None and value > upper)
        ):
            raise ValueError(f"{name} debe ser un entero dentro de sus límites")
    block_length, repetitions, seed = int(block_length), int(repetitions), int(seed)
    reference_kind = "zero" if reference is None else "provided"
    target, prediction, reference = _aligned_vectors(target, prediction, reference)
    unique_dates, inverse, counts = _date_groups(dates, target.size)
    n_dates = int(unique_dates.size)
    if repetitions * n_dates > 10_000_000:
        raise ValueError("El remuestreo supera el límite de diez millones de fechas")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            differences = np.abs(prediction - target) - np.abs(reference - target)
            result = {
                "metric": "paired_mae_difference",
                "estimate": float(np.mean(differences)),
                "interval": None,
                "interval_reason": "Se necesitan más fechas distintas que la longitud de bloque",
                "confidence_level": 0.95,
                "method": "moving_date_blocks_percentile",
                "quantile_method": "linear",
                "block_length": block_length,
                "repetitions": repetitions,
                "completed_repetitions": 0,
                "seed": seed,
                "n_dates": n_dates,
                "n": int(target.size),
                "reference_kind": reference_kind,
                "warning": (
                    "Intervalo exploratorio condicionado a estas fechas y a la longitud de bloque. "
                    "No corrige la selección de modelos ni demuestra rentabilidad."
                ),
            }
            if block_length >= n_dates:
                return result
            date_sums = np.bincount(inverse, weights=differences)
            if not np.all(np.isfinite(date_sums)):
                raise FloatingPointError("Las sumas por fecha no son finitas")
            rng = np.random.default_rng(seed)
            n_blocks = (n_dates + block_length - 1) // block_length
            offsets = np.arange(block_length)
            samples = np.empty(repetitions)
            for index in range(repetitions):
                starts = rng.integers(0, n_dates - block_length + 1, size=n_blocks)
                selected = (starts[:, None] + offsets).ravel()[:n_dates]
                samples[index] = np.sum(date_sums[selected]) / np.sum(counts[selected])
            lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
            result["interval"] = {"lower": float(lower), "upper": float(upper)}
            result["interval_reason"] = None
            result["completed_repetitions"] = repetitions
    except FloatingPointError as error:
        raise ValueError("Los errores exceden el rango numérico de float64") from error
    return result
