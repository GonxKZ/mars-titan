"""Calibración común de intervalos de cuantiles por regresión cuantílica conformalizada.

Sigue CQR (Romano, Patterson y Candès, 2019) con su puntuación simétrica. Para el
intervalo central de nivel nominal 1 − α con extremos emitidos q_bajo y q_alto:

    E_i = max(q_bajo,i − y_i, y_i − q_alto,i),
    Q = estadístico de orden ceil((n + 1)(1 − α)) de las E_i del tramo de calibración,
    intervalo calibrado = [q_bajo − Q, q_alto + Q].

Q negativo estrecha un intervalo demasiado ancho y Q positivo ensancha uno
sobreconfiado. Cada grupo declarado (el mercado) recibe su propia corrección. Si
el orden pedido supera n o el grupo no alcanza el mínimo declarado, la corrección
queda indefinida con su motivo y nunca se sustituye por cero.

La corrección solo usa filas de calibración y queda congelada en un registro con
huella antes de evaluar. La garantía de cobertura de CQR exige intercambiabilidad,
que la dependencia temporal no asegura, por eso el registro declara
``coverage_guaranteed=False``. La mediana no cambia, así que el MAE tampoco.
"""

import hashlib
import json
from fractions import Fraction

import numpy as np

from mars_titan.evaluation.forecast_panel import central_intervals

KIND = "conformal_quantile_calibration"
METHOD = "cqr_symmetric_score_v1"
PARTITION = "calibration"
# Los extremos solo se ensanchan hacia la mediana y el intervalo interior. Así se
# conserva el orden sin estrechar ningún intervalo respecto a su corrección CQR.
ORDER_RULE = "widen_to_median_and_inner_interval"
_MAX_GROUPS = 16


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def interval_pairs(levels, nominals):
    """Pares centrales declarados, del más estrecho al más ancho, y posición de la mediana."""
    levels = tuple(float(level) for level in levels)
    _require(
        1 < len(levels) <= 32
        and all(0 < level < 1 for level in levels)
        and all(a < b for a, b in zip(levels, levels[1:], strict=False)),
        "Los niveles deben ser crecientes y estar en (0, 1)",
    )
    _require(0.5 in levels, "La calibración necesita la mediana entre los niveles")
    pairs = [
        (round(levels[upper] - levels[lower], 12), lower, upper)
        for lower, upper in central_intervals(levels)
    ]
    nominals = tuple(float(value) for value in nominals)
    _require(
        nominals == tuple(nominal for nominal, _, _ in pairs),
        "Las coberturas declaradas deben coincidir con todos los pares centrales emitidos",
    )
    used = {index for _, lower, upper in pairs for index in (lower, upper)}
    _require(
        used | {levels.index(0.5)} == set(range(len(levels))),
        "Todos los niveles salvo la mediana deben pertenecer a un intervalo calibrado",
    )
    return levels, pairs, levels.index(0.5)


def _arrays(target, quantiles, groups, width):
    target = np.asarray(target)
    quantiles = np.asarray(quantiles)
    groups = np.asarray(groups)
    rows = len(target) if target.ndim == 1 else -1
    _require(
        rows >= 1
        and target.dtype.kind == "f"
        and quantiles.dtype.kind == "f"
        and quantiles.shape == (rows, width)
        and groups.shape == (rows,)
        and groups.dtype.kind == "U",
        "Se necesitan objetivos, cuantiles [filas, niveles] y grupos de texto alineados",
    )
    target, quantiles = target.astype(np.float64), quantiles.astype(np.float64)
    _require(
        np.isfinite(target).all() and np.isfinite(quantiles).all(),
        "Los objetivos o cuantiles contienen NaN o infinitos",
    )
    crossed = int(np.count_nonzero(np.any(np.diff(quantiles, axis=1) < 0, axis=1)))
    _require(crossed == 0, f"Hay {crossed} filas con cuantiles cruzados")
    return target, quantiles, groups


def _order(rows, nominal):
    fraction = Fraction(str(nominal))
    return -(-(rows + 1) * fraction.numerator // fraction.denominator)


def fit_conformal_quantiles(target, quantiles, groups, *, levels, nominals, min_rows):
    """Ajustar una corrección por grupo e intervalo con filas de calibración solamente.

    Devuelve un registro serializable. ``calibration_coverage`` es la cobertura en
    las mismas filas de calibración, sin corregir y corregida, y sirve para
    comprobar el ajuste, no como estimación de la cobertura futura.
    """
    levels, pairs, _ = interval_pairs(levels, nominals)
    _require(
        type(min_rows) is int and 1 <= min_rows <= 10_000_000,
        "El mínimo de filas debe ser un entero positivo",
    )
    target, quantiles, groups = _arrays(target, quantiles, groups, len(levels))
    names = sorted(set(groups.tolist()))
    _require(len(names) <= _MAX_GROUPS and all(names), "Los grupos deben ser pocos y no vacíos")
    record = dict(
        schema_version=1,
        kind=KIND,
        method=METHOD,
        partition=PARTITION,
        levels=list(levels),
        nominals=[nominal for nominal, _, _ in pairs],
        min_rows=min_rows,
        order_rule=ORDER_RULE,
        coverage_guaranteed=False,
        groups={},
    )
    for name in names:
        selected = groups == name
        rows = int(np.count_nonzero(selected))
        corrections = {}
        for nominal, lower, upper in pairs:
            scores = np.maximum(
                quantiles[selected, lower] - target[selected],
                target[selected] - quantiles[selected, upper],
            )
            order = _order(rows, nominal)
            entry = dict(
                lower_level=levels[lower],
                upper_level=levels[upper],
                order_statistic=order,
                correction=None,
                reason=None,
                raw_calibration_coverage=float(np.mean(scores <= 0)),
                calibration_coverage=None,
            )
            if rows < min_rows:
                entry["reason"] = "El grupo no alcanza el mínimo declarado de filas"
            elif order > rows:
                entry["reason"] = "El orden requerido supera las filas de calibración"
            else:
                correction = float(np.partition(scores, order - 1)[order - 1])
                entry.update(
                    correction=correction,
                    calibration_coverage=float(np.mean(scores <= correction)),
                )
            corrections[f"{nominal:g}"] = entry
        record["groups"][name] = dict(rows=rows, intervals=corrections)
    return record


def calibrator_sha256(record):
    """Huella del registro congelado, independiente del orden de sus claves."""
    raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def undefined_groups(record, groups):
    """Grupos presentes sin corrección definida en algún intervalo."""
    return sorted(
        name
        for name in set(np.asarray(groups).tolist())
        if name not in record["groups"]
        or any(
            entry["correction"] is None for entry in record["groups"][name]["intervals"].values()
        )
    )


def apply_conformal_quantiles(record, quantiles, groups):
    """Aplicar el registro congelado sin observar objetivos.

    Devuelve los cuantiles calibrados y las filas cuyo extremo se ensanchó para
    conservar el orden, por intervalo nominal.
    """
    _require(
        isinstance(record, dict)
        and record.get("schema_version") == 1
        and record.get("kind") == KIND
        and record.get("method") == METHOD
        and record.get("order_rule") == ORDER_RULE
        and record.get("coverage_guaranteed") is False,
        "El calibrador no conserva su método y contrato",
    )
    levels, pairs, median = interval_pairs(record["levels"], record["nominals"])
    rows = len(np.asarray(groups))
    _, quantiles, groups = _arrays(np.zeros(rows), quantiles, groups, len(levels))
    missing = undefined_groups(record, groups)
    _require(not missing, f"No hay corrección definida para los grupos {missing}")
    names = sorted(record["groups"])
    codes = np.searchsorted(names, groups)
    calibrated = quantiles.copy()
    inner_low = inner_high = quantiles[:, median]
    adjusted = {}
    for nominal, lower, upper in pairs:
        key = f"{nominal:g}"
        correction = np.array(
            [record["groups"][name]["intervals"][key]["correction"] for name in names]
        )[codes]
        low = quantiles[:, lower] - correction
        high = quantiles[:, upper] + correction
        widened = (low > inner_low) | (high < inner_high)
        low, high = np.minimum(low, inner_low), np.maximum(high, inner_high)
        calibrated[:, lower], calibrated[:, upper] = low, high
        inner_low, inner_high = low, high
        adjusted[key] = int(np.count_nonzero(widened))
    return calibrated, adjusted
