"""Abstención neutral `neutral_abstention_v1` (A2 del estudio de variantes, #742 y #748).

La fase 0 (#743) indica que las predicciones de los pilotos son demasiado amplias para lo
que ordenan. Esta etapa no cambia el modelo ni sus pesos. Emite la mediana solo cuando el
intervalo central calibrado excluye el cero y emite cero en otro caso:

    l_i = q_bajo,i − Q_g,   u_i = q_alto,i + Q_g,
    ŷ_i = m_i si l_i > 0 o u_i < 0,   ŷ_i = 0 en otro caso.

Q_g es la corrección CQR del mercado g (`calibration.conformal_quantiles`), ajustada solo con
la partición de calibración (G) y congelada antes de abrir la evaluación. El intervalo
declarado por defecto es [q0,1, q0,9], de cobertura nominal 0,8. Es la misma afirmación de
signo que `forecast_scores` cuenta como `commitments`. La regla de orden de la CQR mantiene la
mediana dentro del intervalo calibrado, así que la predicción emitida tiene su mismo signo.

Los controles se abstienen en el mismo número de filas de cada sesión que la variante. La
cobertura coincide así sesión a sesión y solo cambia qué filas quedan en cero:

- `random`: filas al azar, con un generador propio por semilla del modelo.
- `small_prediction`: las de menor |mediana|.
- `past_volatility`: las de mayor volatilidad pasada del residuo del activo, la raíz de la
  media de los cuadrados de sus últimas etiquetas ya maduras al emitir.
- `seed_disagreement`: las de mayor desviación típica de la mediana entre semillas, que
  emiten todas en el mismo instante.

Los empates y las puntuaciones no definidas se resuelven por la posición canónica del panel,
con las no definidas al final. Si la variante no supera a estos controles, el intervalo
calibrado no decide mejor que una regla trivial dónde abstenerse y la variante se descarta.
"""

import hashlib
import json

import numpy as np

from mars_titan.calibration import conformal_quantiles as cqr
from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.evaluation import abstention_coverage, forecast_scores
from mars_titan.evaluation.forecast_panel import ForecastPanel, central_intervals

IDENTITY = "neutral_abstention_v1"
SCHEMA_VERSION = 1
CONTROLS = ("random", "small_prediction", "past_volatility", "seed_disagreement")
# Configuración declarada por defecto. El mínimo de filas es el de las comparaciones
# walk-forward y la ventana de volatilidad la de la mediana pasada de la fase 0.
DEFAULT = dict(
    nominal=0.8,
    min_rows=1000,
    random_seed=20261023,
    volatility_window=60,
    volatility_min_labels=20,
)
_RECORD = {
    "schema_version",
    "identity",
    "config",
    "fitted_until",
    "calibrator",
    "calibrator_sha256",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def checked_config(config):
    """Validar la configuración declarada de la etapa y devolver una copia."""
    _require(
        isinstance(config, dict) and set(config) == set(DEFAULT),
        f"La abstención neutral declara exactamente {sorted(DEFAULT)}",
    )
    window = config["volatility_window"]
    _require(
        type(config["nominal"]) is float and 0 < config["nominal"] < 1,
        "La cobertura nominal debe ser un decimal en (0, 1)",
    )
    _require(
        type(config["min_rows"]) is int and 1 <= config["min_rows"] <= 10_000_000,
        "El mínimo de filas de la CQR debe ser un entero positivo",
    )
    _require(
        type(config["random_seed"]) is int and 0 <= config["random_seed"] < 2**32,
        "La semilla del control aleatorio debe ser un entero de 32 bits",
    )
    _require(
        type(window) is int
        and 1 <= window <= 10_000
        and type(config["volatility_min_labels"]) is int
        and 1 <= config["volatility_min_labels"] <= window,
        "La ventana de volatilidad y su mínimo de etiquetas no son válidos",
    )
    return dict(config)


def _instants(values, rows, name):
    values = np.asarray(values)
    _require(
        values.shape == (rows,) and values.dtype.kind in "iu",
        f"{name} debe tener una fila por predicción en microsegundos UTC",
    )
    _require(
        values.dtype.kind != "u" or np.all(values <= np.iinfo(np.int64).max),
        f"{name} supera el rango de microsegundos UTC con signo",
    )
    values = values.astype(np.int64)
    _require(np.all(values < FINAL_TEST_START_US), f"{name} entra en el test reservado de 2024")
    return values


def _median(panel):
    """Mediana emitida, que debe ser la predicción puntual del modelo."""
    _require(
        isinstance(panel, ForecastPanel) and panel.levels is not None and 0.5 in panel.levels,
        "La abstención neutral necesita un panel con cuantiles y mediana",
    )
    median = panel.quantiles[:, panel.levels.index(0.5)]
    _require(
        np.array_equal(panel.prediction, median),
        "La predicción puntual del panel no es su mediana",
    )
    return median


def _groups(panel):
    return np.asarray(panel.markets)[panel.market]


def fit(panel, matures_at, config=None):
    """Ajustar la CQR por mercado con la partición de calibración y congelar la etapa.

    `matures_at` es la madurez de cada etiqueta en el orden canónico del panel. Su máximo
    queda en el registro y ninguna emisión anterior podrá usar esta calibración.
    """
    config = checked_config(DEFAULT if config is None else config)
    _median(panel)
    matures_at = _instants(matures_at, panel.rows, "La madurez de calibración")
    _require(
        np.all(matures_at > panel.prediction_at),
        "Cada etiqueta debe madurar después de su predicción",
    )
    panel.interval(config["nominal"])
    nominals = [
        round(panel.levels[upper] - panel.levels[lower], 12)
        for lower, upper in central_intervals(panel.levels)
    ]
    calibrator = cqr.fit_conformal_quantiles(
        panel.target,
        panel.quantiles,
        _groups(panel),
        levels=panel.levels,
        nominals=nominals,
        min_rows=config["min_rows"],
    )
    return dict(
        schema_version=SCHEMA_VERSION,
        identity=IDENTITY,
        config=config,
        fitted_until=int(matures_at.max()),
        calibrator=calibrator,
        calibrator_sha256=cqr.calibrator_sha256(calibrator),
    )


def checked_record(record):
    """Comprobar un registro congelado, también tras leerlo de JSON."""
    _require(
        isinstance(record, dict)
        and set(record) == _RECORD
        and record["schema_version"] == SCHEMA_VERSION
        and record["identity"] == IDENTITY
        and type(record["fitted_until"]) is int
        and record["fitted_until"] < FINAL_TEST_START_US,
        "El registro de la abstención neutral no conserva su contrato",
    )
    checked_config(record["config"])
    _require(
        cqr.calibrator_sha256(record["calibrator"]) == record["calibrator_sha256"],
        "La calibración del registro no coincide con su huella",
    )
    return record


def stage_sha256(record):
    """Huella de la etapa congelada: configuración, calibración y madurez del ajuste."""
    return hashlib.sha256(_canonical(checked_record(record)).encode()).hexdigest()


def apply(record, panel):
    """Predicción emitida y filas abstenidas de un tramo posterior, sin leer sus objetivos."""
    record = checked_record(record)
    median = _median(panel)
    _require(
        tuple(record["calibrator"]["levels"]) == panel.levels,
        "El panel no emite los niveles calibrados",
    )
    _require(
        int(panel.prediction_at.min()) >= record["fitted_until"],
        "La calibración usa etiquetas que maduran después de una emisión evaluada",
    )
    calibrated, _ = cqr.apply_conformal_quantiles(
        record["calibrator"], panel.quantiles, _groups(panel)
    )
    lower, upper = panel.interval(record["config"]["nominal"])
    claim = (calibrated[:, lower] > 0) | (calibrated[:, upper] < 0)
    return np.where(claim, median, 0.0), ~claim


def matched_abstention(panel, abstained, score, *, eligible=None):
    """Abstenerse en tantas filas de cada sesión como `abstained`, las de mayor `score`.

    Una puntuación NaN no está definida y queda detrás de todas las definidas. Los empates
    siguen el orden canónico de las filas. La ordenación por sesión es la de
    `forecast_scores`, medida un 30 % más rápida que `np.lexsort` con un millón de filas.
    Si se declara `eligible`, las demás filas siempre se abstienen y no consumen el
    presupuesto de las elegibles. Así una mediana nula no reduce las llamadas de un control.
    """
    abstained = np.asarray(abstained)
    score = np.asarray(score, dtype=np.float64)
    _require(
        abstained.shape == score.shape == (panel.rows,) and abstained.dtype == bool,
        "La abstención y la puntuación deben tener una fila por predicción",
    )
    _require(not np.isinf(score).any(), "La puntuación de un control no puede ser infinita")
    if eligible is not None:
        eligible = np.asarray(eligible)
        _require(
            eligible.shape == (panel.rows,) and eligible.dtype == bool,
            "La elegibilidad del control debe ser una máscara por fila",
        )
        abstained = abstained & eligible
    counts = np.bincount(panel.session[abstained], minlength=panel.sessions)
    key = np.where(np.isnan(score), np.inf, -score)
    order, position = forecast_scores._within_session_order(panel, key)
    if eligible is not None and not eligible.all():
        active = eligible[order]
        prefix = np.cumsum(active)
        before = prefix[panel.session_starts] - active[panel.session_starts]
        position = prefix - np.repeat(before, panel.session_samples) - 1
    rank = np.empty(panel.rows, dtype=np.int64)
    rank[order] = position
    chosen = rank < counts[panel.session]
    return chosen if eligible is None else chosen | ~eligible


def _rolling_rms(values, window):
    """RMS móvil en O(n), sin restar acumulaciones dominadas por datos ya descartados.

    Una ventana ocupa como mucho dos bloques de anchura `window`. Se unen las normas de
    su sufijo izquierdo y su prefijo derecho con `hypot`. Los prefijos iniciales solo
    usan su propio bloque. Escalar antes de acumular evita elevar retornos grandes al cuadrado.
    """
    if not len(values):
        return np.empty(0, dtype=np.float64)
    scaled = np.pad(np.abs(values) / np.sqrt(window), (0, -len(values) % window)).reshape(
        -1, window
    )
    prefix = np.hypot.accumulate(scaled, axis=1).ravel()
    suffix = np.hypot.accumulate(scaled[:, ::-1], axis=1)[:, ::-1].ravel()
    last = np.arange(len(values))
    first = np.maximum(last + 1 - window, 0)
    count = last + 1 - first
    crossing = first // window != last // window
    norms = prefix[: len(values)].copy()
    norms[crossing] = np.hypot(suffix[first[crossing]], prefix[last[crossing]])
    result = norms / np.sqrt(count / window)
    _require(np.isfinite(result).all(), "La volatilidad calculada no es finita")
    return result


def past_volatility(assets, prediction_at, history, *, window, min_labels):
    """Raíz de la media de los cuadrados de las últimas etiquetas maduras de cada activo.

    `history` es un diccionario con `asset_id`, `target` y `matures_at` de las etiquetas que
    puede leer la ventana: sus particiones anteriores y la propia evaluación. Para una fila
    emitida en t solo cuentan las del mismo activo con madurez <= t, las `window` más
    recientes por madurez. Con menos de `min_labels` el valor es NaN.
    """
    _require(
        type(window) is int
        and 1 <= window <= 10_000
        and type(min_labels) is int
        and 1 <= min_labels <= window,
        "La ventana de volatilidad y su mínimo de etiquetas no son válidos",
    )
    assets = np.asarray(assets, dtype=str)
    rows = len(assets)
    prediction_at = _instants(prediction_at, rows, "Los instantes de emisión")
    labels = np.asarray(history["asset_id"], dtype=str)
    target = np.asarray(history["target"], dtype=np.float64)
    matures_at = _instants(history["matures_at"], len(labels), "La madurez del historial")
    _require(
        assets.ndim == labels.ndim == 1 and target.shape == labels.shape,
        "El historial necesita activo, objetivo y madurez alineados",
    )
    _require(np.isfinite(target).all(), "El historial contiene objetivos no finitos")
    result = np.full(rows, np.nan)
    if not rows:
        return result
    order = np.lexsort((matures_at, labels))
    labels, target, matures_at = labels[order], target[order], matures_at[order]
    _require(
        not np.any((labels[1:] == labels[:-1]) & (matures_at[1:] == matures_at[:-1])),
        "Un activo tiene dos etiquetas que maduran en el mismo instante",
    )
    queries = np.argsort(assets, kind="stable")
    names = assets[queries]
    starts = np.flatnonzero(np.r_[True, names[1:] != names[:-1]])
    for begin, end in zip(starts, np.r_[starts[1:], rows], strict=True):
        selected = queries[begin:end]
        low = np.searchsorted(labels, names[begin], side="left")
        high = np.searchsorted(labels, names[begin], side="right")
        seen = np.searchsorted(matures_at[low:high], prediction_at[selected], side="right")
        defined = np.minimum(seen, window) >= min_labels
        rms = _rolling_rms(target[low:high], window)
        result[selected[defined]] = rms[seen[defined] - 1]
    return result


def seed_disagreement(panels):
    """Desviación típica de la mediana entre semillas, fila a fila, sobre la misma población."""
    _require(
        isinstance(panels, (list, tuple)) and len(panels) >= 2,
        "El desacuerdo entre semillas necesita al menos dos semillas",
    )
    first = panels[0].cohort_sha256
    _require(
        all(panel.cohort_sha256 == first for panel in panels),
        "Las semillas no evalúan las mismas filas",
    )
    return np.column_stack([_median(panel) for panel in panels]).std(axis=1)


def control_scores(record, panel, *, stream, volatility, disagreement):
    """Puntuación de cada control. Una fila con mayor puntuación se abstiene antes."""
    config = checked_record(record)["config"]
    _require(type(stream) is int and stream >= 0, "El flujo aleatorio es la semilla del modelo")
    generator = np.random.default_rng(np.random.SeedSequence([config["random_seed"], stream]))
    return dict(
        random=generator.random(panel.rows),
        small_prediction=-np.abs(_median(panel)),
        past_volatility=np.asarray(volatility, dtype=np.float64),
        seed_disagreement=np.asarray(disagreement, dtype=np.float64),
    )


def emit(record, panel, *, stream, volatility, disagreement):
    """Predicciones de la variante y de sus controles, con su cobertura de llamadas.

    `volatility` y `disagreement` son las puntuaciones de `past_volatility` y
    `seed_disagreement` en el orden canónico del panel, y `stream` la semilla del modelo.
    Las predicciones no leen objetivos. La cobertura de llamadas del registro sí los usa,
    con la definición de `abstention_coverage`.
    """
    prediction, abstained = apply(record, panel)
    median = _median(panel)
    scores = control_scores(
        record, panel, stream=stream, volatility=volatility, disagreement=disagreement
    )
    outputs = {IDENTITY: prediction}
    eligible = median != 0
    for name in CONTROLS:
        chosen = matched_abstention(panel, abstained, scores[name], eligible=eligible)
        outputs[name] = np.where(chosen, 0.0, median)
    eligible = int(np.count_nonzero(panel.target != 0))
    coverage = {
        name: abstention_coverage.call_coverage(
            np.count_nonzero((values != 0) & (panel.target != 0)), eligible
        )
        for name, values in outputs.items()
    }
    summary = dict(
        identity=IDENTITY,
        stage_sha256=stage_sha256(record),
        rows=panel.rows,
        sessions=panel.sessions,
        abstained_rows=int(np.count_nonzero(abstained)),
        undefined_volatility_rows=int(np.count_nonzero(np.isnan(scores["past_volatility"]))),
        call_coverage=coverage,
    )
    return outputs, summary
