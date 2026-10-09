"""Cintas de ajuste, validación y evaluación de una política por ventana walk-forward.

Una política solo ve predicciones fuera de muestra. Por eso cada cinta se monta con el
tramo de evaluación de una ventana del predictor y con el recibo de esa ventana, cuyo
`labels_used_until` es anterior a la primera decisión del tramo. La política de la
ventana k se ajusta con las evaluaciones de las `train_windows` ventanas anteriores a
su validación, valida con la evaluación de la ventana k - 1 y evalúa la de la ventana k.
Toda la información que usa la política termina antes de la primera decisión evaluada.

El universo de activos se fija con datos de ajuste y validación: los activos admitidos en
todos esos tramos, con alguna predicción en la validación, ordenados por la mediana del
efectivo negociado (cierre por volumen) de la validación. Un activo del universo que la
cinta de evaluación excluye deja esa evaluación como fallida, sin cambiar el universo con
información posterior.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa

from mars_titan.environments.walk_forward_receipt import WalkForwardWindow
from mars_titan.evaluation import walk_forward_comparison as comparison

from .reconstructed_tape import build_reconstructed_tape

ROLES = ("train", "validation", "evaluation")
SEGMENT = "evaluation"
UNIVERSE_RULE = "median_traded_value_in_validation_v1"
# Motivo de los episodios de un predictor que no emitió filas del mercado en un tramo.
NO_PREDICTIONS = "predictor_without_predictions"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bounds(fold):
    """Tramo de evaluación de una ventana del predictor en microsegundos UTC."""
    return tuple(int(np.datetime64(day, "us").astype(np.int64)) for day in fold[SEGMENT])


def policy_windows(folds, train_windows):
    """Ventanas de política con sus tramos de ajuste y validación, en orden.

    `folds` son las ventanas del protocolo en orden. La primera ventana de política es
    la primera con `train_windows` evaluaciones anteriores a su validación.
    """
    _require(
        type(train_windows) is int and 1 <= train_windows <= 12,
        "La política necesita entre 1 y 12 ventanas de ajuste",
    )
    rows = []
    for index in range(train_windows + 1, len(folds)):
        row = dict(
            window=folds[index]["id"],
            train=[fold["id"] for fold in folds[index - 1 - train_windows : index - 1]],
            validation=folds[index - 1]["id"],
        )
        check_order(row, {fold["id"]: fold for fold in folds})
        rows.append(row)
    _require(rows, "El protocolo no tiene ventanas suficientes para ajustar y validar políticas")
    return rows


def check_order(row, folds):
    """Exigir tramos consecutivos y que ajuste y validación terminen antes de evaluar."""
    segments = [_bounds(folds[name]) for name in (*row["train"], row["validation"], row["window"])]
    _require(
        all(start < end for start, end in segments)
        and all(a[1] <= b[0] for a, b in zip(segments, segments[1:], strict=False)),
        f"La política de {row['window']} usaría información posterior a su evaluación",
    )


def policy_schedule(rows, period, folds):
    """Asignar a cada ventana de política su ancla, como la variante de la campaña base.

    Con `period` 1 cada ventana ajusta su política. Con un periodo mayor, se ajusta en la
    primera ventana y cada `period` ventanas, y las intermedias evalúan sin ajuste la
    política seleccionada en su ancla, cuya validación termina antes de su evaluación.
    """
    _require(type(period) is int and period >= 1, "El periodo de reajuste no es válido")
    result = []
    for index, row in enumerate(rows):
        anchor = rows[index - index % period]
        _require(
            _bounds(folds[anchor["validation"]])[1] <= _bounds(folds[row["window"]])[0],
            f"La política trasladada a {row['window']} vería información posterior",
        )
        result.append(dict(row, anchor=anchor["window"], trained=anchor is row))
    return result


def segment_predictions(path, digest, market):
    """Puntuaciones emitidas de un mercado: instante, activo y mediana, con su huella.

    Se leen con el lector de la comparación, que comprueba huella, tipos y ausencias, para
    que un cambio del formato de las predicciones toque un único punto.
    """
    table = comparison._read_predictions(
        dict(path=Path(path), sha256=digest), ("asset_id", "market", "prediction_at", "prediction")
    )
    rows = table["market"].to_numpy(zero_copy_only=False).astype(str) == market
    return dict(
        prediction_at=table["prediction_at"].cast(pa.int64()).to_numpy()[rows],
        asset_id=table["asset_id"].to_numpy(zero_copy_only=False).astype(str)[rows],
        score=table["prediction"].to_numpy()[rows],
    )


def build_segment_tape(edition, window, values, *, market, role, lag, symbols=None):
    """Cinta del tramo de evaluación de una ventana del predictor para un papel de la política."""
    _require(isinstance(window, WalkForwardWindow) and role in ROLES, "Papel o recibo no válidos")
    start, _ = window.segment(SEGMENT)
    # El recibo ya lo exige. Se repite aquí porque es la garantía de la que depende la etapa.
    _require(window.labels_used_until < start, "El predictor vio etiquetas posteriores al tramo")
    return build_reconstructed_tape(
        edition,
        [window],
        [values],
        market=market,
        partition="train" if role == "train" else "validation",
        dividend_payment_lag_sessions=lag,
        segment=SEGMENT,
        symbols=symbols,
    )


def admission(tape):
    """Activos admitidos en un tramo, su efectivo negociado mediano y si tienen predicción."""
    value = tape.prices[:, :, 3] * np.nan_to_num(tape.prices[:, :, 4], nan=0.0)
    predicted = np.isfinite(tape.scores).any(axis=0)
    return {
        asset: dict(traded_value=float(np.median(value[:, i])), predicted=bool(predicted[i]))
        for i, asset in enumerate(tape.assets)
    }


def select_universe(train, validation, max_assets):
    """Universo común de una política con la regla `median_traded_value_in_validation_v1`.

    `train` es la lista de admisiones de los tramos de ajuste y `validation` la del tramo
    de validación. Solo intervienen datos anteriores a la evaluación.
    """
    _require(type(max_assets) is int and 1 <= max_assets <= 4096, "El universo admite de 1 a 4096")
    candidates = set(validation).intersection(*(set(item) for item in train))
    ranked = sorted(
        (asset for asset in candidates if validation[asset]["predicted"]),
        key=lambda asset: (-validation[asset]["traded_value"], asset),
    )
    _require(ranked, "Ningún activo cumple la regla del universo en ajuste y validación")
    return sorted(ranked[:max_assets])
