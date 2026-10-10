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

El universo se elige en cada tramo con datos anteriores a su primera decisión, de modo que
incluye las empresas que después dejan de cotizar. En cada tramo son los `max_assets`
activos con mayor mediana del efectivo negociado (cierre por volumen) en las sesiones de
clasificación anteriores al tramo. En los tramos de ajuste y validación, que la política
usa como historia, los candidatos cumplen las condiciones de la cinta del tramo y tienen
alguna predicción del predictor del universo en él. Una baja dentro del tramo no excluye a
nadie. En la evaluación los candidatos son los que cotizaban al empezar y tenían
predicciones en la validación del ancla, sin mirar la propia evaluación. Un activo del
universo que la cinta de evaluación excluye por la calidad de sus filas deja esa
evaluación como fallida.

La política observa un diseño fijo de activos, la unión ordenada de los universos de sus
tramos. En cada tramo, los activos del diseño que no forman parte de su universo quedan sin
precios ni predicciones. Una ventana intermedia, evaluada sin reajuste con la política de
su ancla, elige su universo dentro del diseño del ancla.
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
UNIVERSE_RULE = "point_in_time_median_traded_value_v2"
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


def build_segment_tape(
    edition, window, values, *, market, role, lag, listing_status, symbols=None, universe=None
):
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
        listing_status=listing_status,
        segment=SEGMENT,
        symbols=symbols,
        universe=universe,
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


def universe_rule(value):
    """Regla declarada del universo: `(máximo de activos, sesiones de clasificación)`."""
    _require(
        isinstance(value, dict)
        and set(value) == {"rule", "max_assets", "ranking_sessions"}
        and value["rule"] == UNIVERSE_RULE
        and type(value["max_assets"]) is int
        and 1 <= value["max_assets"] <= 4096
        and type(value["ranking_sessions"]) is int
        and 20 <= value["ranking_sessions"] <= 756,
        f"El universo declara {UNIVERSE_RULE}, de 1 a 4096 activos y de 20 a 756 sesiones",
    )
    return value["max_assets"], value["ranking_sessions"]


def layout_bound(universe, train_windows):
    """Máximo del diseño de una política: universos disjuntos en todos sus tramos.

    El diseño une los universos de las ventanas de ajuste, la validación y la evaluación,
    así que nunca supera `max_assets` por ese número de tramos ni el límite de 4096 activos.
    """
    max_assets, _ = universe_rule(universe)
    return min(4096, max_assets * (train_rule(train_windows)[2] + 2))


def covered(values):
    """Activos con alguna predicción finita en las predicciones de un tramo."""
    scores = np.asarray(values["score"], dtype=np.float64)
    assets = np.asarray(values["asset_id"]).astype(str)
    return set(assets[np.isfinite(scores)].tolist())


def select_universe(census, covered, max_assets, *, evaluation, within=None):
    """Universo de un tramo: los `max_assets` candidatos con mayor efectivo mediano previo.

    `census` es el de `reconstructed_tape.census` para el tramo y `covered` los activos con
    predicciones del predictor del universo: las del propio tramo en ajuste y validación y
    las de la validación del ancla en una evaluación. En ajuste y validación un candidato
    cumple las condiciones de la cinta del tramo. En una evaluación basta con que cotizara al
    empezar, porque esas condiciones usan filas posteriores. `within` limita la elección al
    diseño de un ancla. Un activo sin efectivo previo positivo no es candidato. El empate se
    resuelve por la clave del activo y el resultado se ordena.
    """
    _require(type(max_assets) is int and 1 <= max_assets <= 4096, "El universo admite de 1 a 4096")
    ranked = sorted(
        (
            asset
            for asset, row in census.items()
            if (row["listed"] if evaluation else row["reason"] is None)
            and asset in covered
            and row["value"] > 0
            and (within is None or asset in within)
        ),
        key=lambda asset: (-census[asset]["value"], asset),
    )
    _require(ranked, "Ningún activo cumple la regla del universo en el tramo")
    return sorted(ranked[:max_assets])
