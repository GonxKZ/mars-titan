"""Recalibración periódica de la CQR con filas maduras, alternativa trivial de PT2.

Es la comparación sencilla que declara la calibración en línea (#454): en lugar de seguir el
cuantil de la puntuación cohorte a cohorte, se vuelve a ajustar la CQR estática cada cierto
número de sesiones de cada mercado con las filas cuya etiqueta ya ha madurado. Si iguala a
la versión en línea, el seguimiento por cohortes no aporta nada sobre algo más simple.

Cada mercado sigue sus propias sesiones. Las primeras `every_sessions` usan la CQR estática
de la partición de calibración, la misma corrección inicial que la versión en línea. Al
empezar cada bloque siguiente, en el instante t de su primera sesión, se reúnen las filas de
calibración y de evaluación del mercado con `matures_at <= t` y `prediction_at >= t -
lookback`. Si alcanzan el mínimo de filas, la CQR se ajusta con ellas y se aplica a todo el
bloque. Si no, el bloque conserva la corrección anterior. Una etiqueta que madura después
de t no interviene en ninguna corrección emitida en t o antes.
"""

import numpy as np

from mars_titan.calibration import conformal_quantiles as cqr
from mars_titan.environments.cohorts import FINAL_TEST_START_US

METHOD = "cqr_periodic_refit_matured_rows_v1"
DAY_MICROSECONDS = 86_400_000_000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rows(part, quantile_width, label):
    """Columnas alineadas de un tramo: objetivo, cuantiles, mercado, instante y madurez."""
    target = np.asarray(part["target"], dtype=np.float64)
    quantiles = np.asarray(part["quantiles"], dtype=np.float64)
    groups = np.asarray(part["groups"])
    prediction_at = np.asarray(part["prediction_at"])
    matures_at = np.asarray(part["matures_at"])
    rows = len(target)
    _require(
        rows >= 1
        and target.shape == (rows,)
        and quantiles.shape == (rows, quantile_width)
        and groups.shape == (rows,)
        and groups.dtype.kind == "U"
        and prediction_at.shape == matures_at.shape == (rows,)
        and prediction_at.dtype.kind in "iu"
        and matures_at.dtype.kind in "iu"
        and bool(np.all(matures_at > prediction_at)),
        f"El tramo de {label} necesita columnas alineadas y madurez posterior a cada predicción",
    )
    # Igual que PT2, la alternativa periódica no lee el test reservado de 2024.
    _require(
        bool(np.all(matures_at < FINAL_TEST_START_US)),
        f"El tramo de {label} entra en el test reservado de 2024",
    )
    return target, quantiles, groups, prediction_at.astype(np.int64), matures_at.astype(np.int64)


def replay_periodic_refit(record, calibration, evaluation, *, every_sessions, lookback_days):
    """Emitir la evaluación con la CQR reajustada cada `every_sessions` sesiones de su mercado.

    `record` es la CQR estática de la ventana, ajustada con la partición de calibración, y
    fija niveles, intervalos y mínimo de filas. `calibration` y `evaluation` son
    diccionarios con `target`, `quantiles`, `groups`, `prediction_at` y `matures_at` en
    microsegundos UTC. Devuelve los cuantiles emitidos en el orden de la evaluación y un
    registro de cada reajuste.
    """
    _require(
        type(every_sessions) is int and every_sessions >= 1,
        "El reajuste necesita un número entero positivo de sesiones",
    )
    _require(
        type(lookback_days) is int and lookback_days >= 1,
        "La ventana de filas maduras necesita un número entero positivo de días",
    )
    width = len(record["levels"])
    pool = _rows(calibration, width, "calibración")
    target, quantiles, groups, prediction_at, matures_at = _rows(evaluation, width, "evaluación")
    missing = cqr.undefined_groups(record, groups)
    _require(not missing, f"La CQR estática no define una corrección inicial para {missing}")
    _require(
        int(pool[3].max()) < int(prediction_at.min()),
        "La calibración debe preceder a la evaluación",
    )
    # Las filas que pueden entrar en un reajuste: calibración y evaluación del mismo mercado.
    current_rows = (target, quantiles, groups, prediction_at, matures_at)
    known = tuple(np.concatenate(pair) for pair in zip(pool, current_rows, strict=True))
    lookback = lookback_days * DAY_MICROSECONDS
    emitted = np.empty_like(quantiles)
    refits = []
    for name in sorted(set(groups.tolist())):
        current = dict(record, groups={name: record["groups"][name]})
        market = np.flatnonzero(groups == name)
        sessions = np.unique(prediction_at[market])
        for start in range(0, len(sessions), every_sessions):
            at = int(sessions[start])
            if start:
                selected = (known[2] == name) & (known[4] <= at) & (known[3] >= at - lookback)
                rows = int(np.count_nonzero(selected))
                used = False
                if rows:
                    fitted = cqr.fit_conformal_quantiles(
                        known[0][selected],
                        known[1][selected],
                        known[2][selected],
                        levels=record["levels"],
                        nominals=record["nominals"],
                        min_rows=record["min_rows"],
                    )
                    # Sin el mínimo de filas el bloque conserva la corrección anterior.
                    used = not cqr.undefined_groups(fitted, [name])
                    if used:
                        current = fitted
                refits.append(
                    dict(
                        group=name,
                        at=at,
                        rows=rows,
                        used=used,
                        calibrator_sha256=cqr.calibrator_sha256(current),
                    )
                )
            block = sessions[start : start + every_sessions]
            members = market[np.isin(prediction_at[market], block)]
            calibrated, _ = cqr.apply_conformal_quantiles(
                current, quantiles[members], groups[members]
            )
            emitted[members] = calibrated
    return emitted, refits
