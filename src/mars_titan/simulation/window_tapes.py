"""Cintas de ajuste, validación y evaluación de una política por ventana walk-forward.

Una política solo ve predicciones fuera de muestra. Por eso cada cinta se monta con el
tramo de evaluación de una ventana del predictor y con el recibo de esa ventana, cuyo
`labels_used_until` es anterior a la primera decisión del tramo. La política de la
ventana k se ajusta con las evaluaciones de las ventanas anteriores a su validación, valida
con la evaluación de la ventana k - 1 y evalúa la de la ventana k. Con la regla en
expansión usa todas esas evaluaciones, desde la primera ventana del protocolo y hasta un
máximo declarado de las más recientes, y con la regla fija solo las `minimum` más
recientes. La primera ventana de política es la primera con `minimum` evaluaciones
anteriores a su validación, en las dos reglas. Toda la información que usa la política
termina antes de la primera decisión evaluada.

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

from .market import RECONSTRUCTED
from .reconstructed_tape import SOURCE_KIND, build_reconstructed_tape

ROLES = ("train", "validation", "evaluation")
# Tramos de ajuste de una política: las evaluaciones fuera de muestra anteriores a su
# validación, todas hasta `maximum` (las más recientes) o solo las `minimum` más recientes.
EXPANDING = "expanding_prior_evaluations_v1"
FIXED = "fixed_prior_evaluations_v1"
TRAIN_RULES = (EXPANDING, FIXED)
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


def train_rule(value):
    """Regla de ajuste declarada: `(regla, mínimo, máximo)` de evaluaciones anteriores.

    La regla en expansión toma todas las evaluaciones anteriores a la validación hasta
    `maximum`, las más recientes. La fija toma siempre `minimum`, así que su máximo coincide
    con el mínimo y solo ella lo admite.
    """
    _require(
        isinstance(value, dict)
        and set(value) == {"rule", "minimum", "maximum"}
        and value["rule"] in TRAIN_RULES
        and type(value["minimum"]) is int
        and type(value["maximum"]) is int
        and 1 <= value["minimum"] <= 12
        and value["minimum"] <= value["maximum"]
        and (value["maximum"] == value["minimum"]) == (value["rule"] == FIXED),
        "La política declara su regla y entre 1 y 12 ventanas de ajuste como mínimo, con un "
        "máximo que solo coincide con el mínimo en la regla fija",
    )
    return value["rule"], value["minimum"], value["maximum"]


def policy_windows(folds, train_windows):
    """Ventanas de política con sus tramos de ajuste y validación, en orden.

    `folds` son las ventanas del protocolo en orden y `train_windows` declara la regla, el
    mínimo y el máximo de evaluaciones de ajuste. La primera ventana de política es la
    primera con `minimum` evaluaciones anteriores a su validación, y ninguna se ajusta con
    más de `maximum`.
    """
    _, minimum, maximum = train_rule(train_windows)
    rows = []
    for index in range(minimum + 1, len(folds)):
        row = dict(
            window=folds[index]["id"],
            train=[fold["id"] for fold in folds[max(0, index - 1 - maximum) : index - 1]],
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


def require_real_tape(tape, edition_id, label):
    """Exigir una cinta real de la edición verificada al empezar la etapa.

    Una política de la campaña solo aprende, se selecciona y se evalúa sobre precios reales
    reconstruidos: dominio `real`, origen en la edición sin ajustar, base reconstruida con
    su contrato (que `MarketTape` comprueba al crearla) y la misma identidad de edición que
    la etapa verificó y declaró. Una cinta sintética o de otra edición detiene el trabajo
    antes de lanzar ningún ejecutor.
    """
    audit = tape.identity.get("audit") or {}
    source = tape.identity.get("source") or {}
    _require(
        tape.domain == "real"
        and source.get("kind") == SOURCE_KIND
        and audit.get("price_basis") == RECONSTRUCTED
        and audit.get("edition_id") == edition_id,
        f"La cinta {label} no es una cinta real de la edición declarada",
    )
    return tape


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
