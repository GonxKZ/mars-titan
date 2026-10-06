"""Aciertos e intervalos empíricos de retornos residuales, no de subidas brutas del activo.

La dependencia temporal impide atribuir una garantía de cobertura a estos
cuantiles. El llamante debe separar calibración y evaluación después de
congelar la selección del modelo. Este módulo no lee datos ni elige modelos.
"""

import math

import numpy as np

MAX_ROWS = 1_000_000
TARGET_KIND = "residual_return"
_LEVELS = {0.9: (9, 10), 0.95: (19, 20)}
_INSUFFICIENT = "El rango requerido supera el número de observaciones de calibración"
_CALIBRATION_FIELDS = {
    "schema_version",
    "kind",
    "partition",
    "target_kind",
    "confidence",
    "coverage_guaranteed",
    "calibration_samples",
    "order_statistic",
    "radius",
    "reason",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _vectors(target, prediction):
    values = []
    for source in (target, prediction):
        raw = np.asarray(source)
        _require(
            raw.ndim == 1 and raw.size <= MAX_ROWS and raw.dtype.kind in "fiu",
            "Las entradas deben ser vectores reales de hasta un millón de filas",
        )
        if raw.dtype.kind in "iu":
            _require(
                np.all(raw <= 2**53) and (raw.dtype.kind == "u" or np.all(raw >= -(2**53))),
                "Un entero pierde precisión al convertirlo a float64",
            )
        try:
            with np.errstate(over="raise", invalid="raise"):
                array = raw.astype(np.float64, copy=False)
        except FloatingPointError as error:
            raise ValueError("Las entradas desbordan float64") from error
        _require(np.isfinite(array).all(), "Las entradas contienen valores no finitos")
        values.append(array)
    _require(values[0].shape == values[1].shape, "Las entradas no tienen la misma forma")
    return values


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _directions(target, prediction):
    positive, negative = target > 0, target < 0
    eligible = positive | negative
    positive_calls, negative_calls = (prediction > 0) & eligible, (prediction < 0) & eligible

    def count(values):
        return int(np.count_nonzero(values))

    correct_positive, correct_negative = (
        count(positive_calls & positive),
        count(negative_calls & negative),
    )
    positives, negatives = count(positive), count(negative)
    calls_positive, calls_negative = count(positive_calls), count(negative_calls)
    eligible_count, calls = positives + negatives, calls_positive + calls_negative
    correct = correct_positive + correct_negative
    nonzero = count(prediction != 0)
    return dict(
        target_kind=TARGET_KIND,
        samples=len(target),
        zero_targets=len(target) - eligible_count,
        eligible_targets=eligible_count,
        positive_targets=positives,
        negative_targets=negatives,
        calls=calls,
        positive_calls=calls_positive,
        negative_calls=calls_negative,
        abstentions=eligible_count - calls,
        correct_calls=correct,
        correct_positive=correct_positive,
        correct_negative=correct_negative,
        call_coverage=_ratio(calls, eligible_count),
        conditional_accuracy=_ratio(correct, calls),
        direction_accuracy=_ratio(correct, eligible_count),
        positive_precision=_ratio(correct_positive, calls_positive),
        negative_precision=_ratio(correct_negative, calls_negative),
        positive_recall=_ratio(correct_positive, positives),
        negative_recall=_ratio(correct_negative, negatives),
        nonzero_predictions=nonzero,
        nonzero_prediction_fraction=_ratio(nonzero, len(target)),
    )


def directional_diagnostics(target, prediction):
    """Contar aciertos de signo excluyendo objetivos nulos y conservando abstenciones.

    ``call_coverage`` usa los objetivos no nulos. Una predicción cero es una
    abstención y cuenta sin acierto en ``direction_accuracy``. La fracción
    ``nonzero_prediction_fraction`` usa todas las filas, incluidos objetivos cero.
    Un signo positivo del retorno residual no implica una subida bruta del activo.
    """
    return _directions(*_vectors(target, prediction))


def _order(count, confidence):
    _require(
        type(confidence) in (int, float) and confidence in _LEVELS,
        "Solo se admiten niveles 0,90 y 0,95",
    )
    numerator, denominator = _LEVELS[confidence]
    return ((count + 1) * numerator + denominator - 1) // denominator


def _calibration(record):
    _require(
        isinstance(record, dict) and set(record) == _CALIBRATION_FIELDS,
        "El calibrador no conserva sus campos",
    )
    _require(
        type(record["schema_version"]) is int
        and record["schema_version"] == 1
        and record["kind"] == "symmetric_absolute_error"
        and record["partition"] == "calibration"
        and record["target_kind"] == TARGET_KIND
        and record["coverage_guaranteed"] is False
        and type(record["calibration_samples"]) is int
        and 0 <= record["calibration_samples"] <= MAX_ROWS,
        "El calibrador no conserva su origen, nivel o población",
    )
    count = record["calibration_samples"]
    order = _order(count, record["confidence"])
    _require(
        type(record["order_statistic"]) is int and record["order_statistic"] == order,
        "El rango del calibrador no coincide con su población",
    )
    radius = record["radius"]
    if order > count:
        _require(
            radius is None and record["reason"] == _INSUFFICIENT,
            "Una calibración insuficiente no puede declarar un radio",
        )
    else:
        _require(type(radius) in (int, float), "El radio debe ser un número real")
        try:
            radius = float(radius)
        except OverflowError as error:
            raise ValueError("El radio desborda float64") from error
        _require(
            math.isfinite(radius)
            and 0 <= radius <= np.finfo(np.float64).max / 2
            and record["reason"] is None,
            "El radio o la anchura del intervalo no son finitos y representables",
        )
    return radius


def calibrate_absolute_error(target, prediction, *, confidence, partition="calibration"):
    """Fijar un radio con el error absoluto de calibración, sin interpolación.

    Se toma el orden ceil((n + 1) × confidence). Si supera n, el radio queda
    indefinido. El registro describe un cuantil empírico y no acredita por sí
    solo la separación temporal ni la procedencia de los vectores recibidos.
    """
    _require(partition == "calibration", "El radio solo se ajusta con calibración")
    target, prediction = _vectors(target, prediction)
    order = _order(len(target), confidence)
    try:
        with np.errstate(over="raise", invalid="raise"):
            errors = np.abs(target - prediction)
    except FloatingPointError as error:
        raise ValueError("El error absoluto desborda float64") from error
    defined = order <= len(errors)
    result = dict(
        schema_version=1,
        kind="symmetric_absolute_error",
        partition=partition,
        target_kind=TARGET_KIND,
        confidence=float(confidence),
        coverage_guaranteed=False,
        calibration_samples=len(errors),
        order_statistic=order,
        radius=float(np.partition(errors, order - 1)[order - 1]) if defined else None,
        reason=None if defined else _INSUFFICIENT,
    )
    _calibration(result)
    return result


def interval_diagnostics(target, prediction, calibration):
    """Evaluar el radio congelado y llamar un signo solo si el intervalo excluye cero.

    La cobertura incluye objetivos nulos y ambos extremos del intervalo. Sus
    diagnósticos direccionales excluyen los objetivos nulos. No se vuelve a
    calcular el radio con las etiquetas evaluadas.
    """
    target, prediction = _vectors(target, prediction)
    radius = _calibration(calibration)
    result = dict(
        target_kind=TARGET_KIND,
        confidence=calibration["confidence"],
        coverage_guaranteed=False,
        samples=len(target),
        intervals=0,
        covered=None,
        coverage=None,
        mean_width=None,
        direction=None,
        reason=calibration["reason"],
    )
    if radius is None:
        return result
    try:
        with np.errstate(over="raise", invalid="raise"):
            lower, upper = prediction - radius, prediction + radius
    except FloatingPointError as error:
        raise ValueError("Los extremos del intervalo desbordan float64") from error
    covered = int(np.count_nonzero((target >= lower) & (target <= upper)))
    calls = np.zeros_like(prediction)
    calls[lower > 0], calls[upper < 0] = 1.0, -1.0
    result.update(
        intervals=len(target),
        covered=covered,
        coverage=_ratio(covered, len(target)),
        mean_width=2 * radius if len(target) else None,
        direction=_directions(target, calls),
        reason=None if len(target) else "No hay observaciones de evaluación",
    )
    return result
