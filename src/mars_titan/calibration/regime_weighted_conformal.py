"""Conformal ponderado por contextos parecidos `regime_weighted_conformal_v1` (A3, #748).

Sigue la idea de HopCPT (Auer, Gauch, Klotz y Hochreiter, NeurIPS 2023,
https://arxiv.org/abs/2303.12783v2): los errores pasados cuyo contexto de emisión se parece
al actual pesan más al calcular el cuantil del error. Aquí no se entrena ninguna red. La
lectura es la de una memoria de Hopfield moderna con proyecciones identidad sobre contextos
estandarizados en G:

    p_ij = softmax_j(−‖z_i − z_j‖² / (2h²)),
    Q_i  = inf{e : Σ_j p_ij · 1{E_j ≤ e} ≥ 1 − a},
    [q_bajo,i − Q_i, q_alto,i + Q_i] con la regla de orden de la CQR.

E_j es la puntuación CQR de una fila pasada cuya etiqueta ya maduró en el instante t de la
emisión, del mismo mercado y emitida como mucho `lookback_days` antes. El contexto de una
fila usa solo lo disponible al emitirla:

- el logaritmo de la anchura emitida q0,9 − q0,1;
- el logaritmo de la raíz de la media de los cuadrados del objetivo residual del mercado,
  con las etiquetas que maduraron en los últimos `regime_days` días.

Los cuantiles calibrados pueden pasar después a la abstención de A2 con
`neutral_abstention.emit_calibrated`.

La media y la desviación típica de cada rasgo se fijan con las filas de calibración (G). Hay
dos controles con el mismo conjunto pasado y la misma regla de cuantil: `uniform`, con pesos
iguales, y `shuffled`, que baraja los contextos del conjunto pasado en cada sesión. Barajar
los contextos conserva en cada fila el mismo conjunto de pesos y solo rompe su relación con
los errores. Si la variante no mejora a los dos controles, la similitud de contexto no aporta.
"""

import hashlib
import json
import math

import numpy as np

from mars_titan.calibration import conformal_quantiles as cqr
from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.evaluation.forecast_panel import ForecastPanel, central_intervals

IDENTITY = "regime_weighted_conformal_v1"
SCHEMA_VERSION = 1
WEIGHTINGS = ("context", "uniform", "shuffled")
DAY_US = 86_400_000_000
# Intervalo cuya anchura entra en el contexto: el central más estrecho, [q0,1, q0,9].
CONTEXT_NOMINAL = 0.8
DEFAULT = dict(bandwidth=0.5, lookback_days=120, regime_days=30, min_rows=1000, seed=20261027)
# Suelo de las magnitudes antes del logaritmo y margen del cuantil ponderado. La suma
# acumulada y secuencial de n pesos iguales se desvía del nivel exacto hasta unos n · 1e-16,
# 3e-12 con 30.000 filas. El margen queda muy por debajo del paso 1/n con el máximo de
# filas admitido (1e-7).
FLOOR = 1e-12
TOLERANCE = 1e-9
# Desviación mínima de un rasgo en G, en unidades logarítmicas. Por debajo el rasgo es
# constante salvo redondeo (la anchura (m + s) − (m − s) no da exactamente 2s) y
# estandarizarlo convertiría ese redondeo en distancias.
MIN_DEVIATION = 1e-9
# Elementos de la matriz de pesos que se forman a la vez, 4 MiB por temporal en float64. Con
# bloques mayores los temporales salen de la caché y cada sesión tarda varias veces más (ver
# docs/experiments/regime-weighted-conformal-v1.md).
MAX_BLOCK = 524_288
_RECORD = ("schema_version", "identity", "config", "levels", "nominals", "scale", "fitted_until")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_config(config):
    """Validar la configuración declarada y devolver una copia."""
    _require(
        isinstance(config, dict) and set(config) == set(DEFAULT),
        f"El conformal ponderado declara exactamente {sorted(DEFAULT)}",
    )
    _require(
        type(config["bandwidth"]) is float and 0 < config["bandwidth"] <= 100,
        "La anchura de banda debe ser un decimal positivo",
    )
    for name in ("lookback_days", "regime_days"):
        _require(
            type(config[name]) is int and 1 <= config[name] <= 3660,
            f"{name} debe ser un número de días entre 1 y 3.660",
        )
    _require(
        type(config["min_rows"]) is int and 1 <= config["min_rows"] <= 10_000_000,
        "El mínimo de filas pasadas debe ser un entero positivo",
    )
    _require(
        type(config["seed"]) is int and 0 <= config["seed"] < 2**32,
        "La semilla de los pesos barajados debe ser un entero de 32 bits",
    )
    return dict(config)


def _instants(values, rows, name):
    values = np.asarray(values)
    _require(
        values.shape == (rows,) and values.dtype.kind == "i",
        f"{name} necesita una fila por predicción en microsegundos UTC",
    )
    values = values.astype(np.int64)
    _require(np.all(values < FINAL_TEST_START_US), f"{name} entra en el test reservado de 2024")
    return values


def _rms(values):
    """Raíz de la media de los cuadrados, escalando antes para no desbordar al elevar."""
    peak = float(np.max(np.abs(values)))
    if peak == 0:
        return 0.0
    return peak * math.sqrt(float(np.mean(np.square(values / peak))))


def market_regime(markets, prediction_at, history, *, days):
    """RMS del objetivo residual de cada mercado con etiquetas maduras en (t − días, t].

    `history` contiene `market`, `target` y `matures_at` de las etiquetas que puede leer la
    ventana. Sin ninguna etiqueta madura en ese intervalo el valor es NaN.
    """
    markets = np.asarray(markets, dtype=str)
    rows = len(markets)
    prediction_at = _instants(prediction_at, rows, "Los instantes de emisión")
    labels = np.asarray(history["market"], dtype=str)
    target = np.asarray(history["target"], dtype=np.float64)
    matures_at = _instants(history["matures_at"], len(labels), "La madurez del historial")
    _require(
        markets.ndim == 1 and target.shape == labels.shape and np.isfinite(target).all(),
        "El historial necesita mercado, objetivo finito y madurez alineados",
    )
    result = np.full(rows, np.nan)
    for name in np.unique(markets):
        own = labels == name
        order = np.argsort(matures_at[own], kind="stable")
        moments, values = matures_at[own][order], target[own][order]
        selected = np.flatnonzero(markets == name)
        for at in np.unique(prediction_at[selected]):
            low = np.searchsorted(moments, at - days * DAY_US, side="right")
            high = np.searchsorted(moments, at, side="right")
            if high > low:
                result[selected[prediction_at[selected] == at]] = _rms(values[low:high])
    return result


def contexts(panel, history, *, days):
    """Contexto de cada fila en su emisión: log de la anchura y log del régimen del mercado."""
    _require(
        isinstance(panel, ForecastPanel) and panel.levels is not None,
        "El contexto necesita un panel con cuantiles",
    )
    lower, upper = panel.interval(CONTEXT_NOMINAL)
    width = panel.quantiles[:, upper] - panel.quantiles[:, lower]
    regime = market_regime(
        np.asarray(panel.markets)[panel.market], panel.prediction_at, history, days=days
    )
    return np.column_stack(
        [
            np.log(np.maximum(width, FLOOR)),
            np.log(np.where(np.isnan(regime), np.nan, np.maximum(regime, FLOOR))),
        ]
    )


def weighted_quantile(scores, weights, level):
    """inf{e : Σ_j p_j · 1{E_j ≤ e} ≥ nivel} para cada columna de pesos normalizados.

    `weights` tiene forma [puntuaciones, filas evaluadas] y cada columna suma uno. Las
    puntuaciones empatadas se acumulan en el orden estable de la ordenación.
    """
    order = np.argsort(scores, kind="stable")
    return _sorted_quantile(scores[order], weights[order], level)


def _sorted_quantile(scores, weights, level):
    # Las puntuaciones van en el primer eje para que la suma acumulada recorra filas
    # contiguas. Los pesos no son negativos, así que la suma no decrece y el primer índice que
    # alcanza el nivel es el número de posiciones que quedan por debajo.
    below = np.cumsum(weights, axis=0) < level - TOLERANCE
    _require(not below[-1].any(), "Los pesos de una fila no alcanzan el nivel pedido")
    return scores[np.count_nonzero(below, axis=0)]


def fit(calibration, matures_at, history, config=None):
    """Fijar con las filas de calibración (G) la escala de los rasgos del contexto."""
    config = checked_config(DEFAULT if config is None else config)
    matures_at = _instants(matures_at, calibration.rows, "La madurez de calibración")
    _require(
        np.all(matures_at > calibration.prediction_at),
        "Cada etiqueta debe madurar después de su predicción",
    )
    levels, pairs, _ = cqr.interval_pairs(calibration.levels, _nominals(calibration.levels))
    features = contexts(calibration, history, days=config["regime_days"])
    defined = ~np.isnan(features).any(axis=1)
    _require(defined.sum() >= 2, "La calibración no tiene contextos definidos suficientes")
    mean = features[defined].mean(axis=0)
    deviation = features[defined].std(axis=0)
    _require(
        np.all(deviation >= MIN_DEVIATION), "Un rasgo del contexto es constante en la calibración"
    )
    return dict(
        schema_version=SCHEMA_VERSION,
        identity=IDENTITY,
        config=config,
        levels=list(levels),
        nominals=[nominal for nominal, _, _ in pairs],
        scale=dict(mean=mean.tolist(), deviation=deviation.tolist()),
        fitted_until=int(matures_at.max()),
    )


def _nominals(levels):
    return [round(levels[upper] - levels[lower], 12) for lower, upper in central_intervals(levels)]


def checked_record(record):
    """Comprobar un registro congelado, también tras leerlo de JSON."""
    _require(
        isinstance(record, dict)
        and set(record) == set(_RECORD)
        and record["schema_version"] == SCHEMA_VERSION
        and record["identity"] == IDENTITY
        and type(record["fitted_until"]) is int
        and record["fitted_until"] < FINAL_TEST_START_US,
        "El registro del conformal ponderado no conserva su contrato",
    )
    checked_config(record["config"])
    scale = record["scale"]
    _require(
        isinstance(scale, dict)
        and set(scale) == {"mean", "deviation"}
        and len(scale["mean"]) == len(scale["deviation"]) == 2
        and all(math.isfinite(value) for value in scale["mean"])
        and all(math.isfinite(value) and value >= MIN_DEVIATION for value in scale["deviation"]),
        "La escala del contexto no es válida",
    )
    cqr.interval_pairs(record["levels"], record["nominals"])
    return record


def stage_sha256(record):
    """Huella del registro congelado."""
    raw = json.dumps(checked_record(record), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _rows(record, panel, matures_at, history, label):
    """Mercado, instantes, contexto estandarizado y puntuaciones CQR de un tramo."""
    _require(
        isinstance(panel, ForecastPanel) and panel.levels == tuple(record["levels"]),
        f"El tramo de {label} no emite los niveles del registro",
    )
    matures_at = _instants(matures_at, panel.rows, f"La madurez de {label}")
    _require(
        np.all(matures_at > panel.prediction_at),
        f"Cada etiqueta de {label} debe madurar después de su predicción",
    )
    _, pairs, _ = cqr.interval_pairs(record["levels"], record["nominals"])
    features = contexts(panel, history, days=record["config"]["regime_days"])
    scale = record["scale"]
    standardized = (features - np.asarray(scale["mean"])) / np.asarray(scale["deviation"])
    scores = np.column_stack(
        [
            np.maximum(
                panel.quantiles[:, lower] - panel.target, panel.target - panel.quantiles[:, upper]
            )
            for _, lower, upper in pairs
        ]
    )
    return dict(
        market=np.asarray(panel.markets)[panel.market],
        prediction_at=panel.prediction_at,
        matures_at=matures_at,
        context=standardized,
        scores=scores,
    )


def _weights(now, keys, bandwidth):
    """Pesos softmax de −‖z_i − z_j‖² / (2h²) con forma [claves, filas evaluadas].

    Se resta el máximo de cada columna antes de la exponencial para que un contexto lejano de
    todos los pasados no deje todos los pesos en cero.
    """
    distance = np.square(keys[:, 0, None] - now[:, 0])
    for column in range(1, keys.shape[1]):
        distance += np.square(keys[:, column, None] - now[:, column])
    logits = distance / -(2 * bandwidth * bandwidth)
    logits -= logits.max(axis=0)
    weights = np.exp(logits, out=logits)
    weights /= weights.sum(axis=0)
    return weights


def replay(record, calibration, evaluation, history, *, weighting):
    """Cuantiles calibrados de la evaluación, sesión a sesión y mercado a mercado.

    `calibration` y `evaluation` son pares (panel, madurez en su orden canónico). Una fila
    entra en el conjunto pasado de la sesión t solo si su etiqueta maduró como mucho en t, es
    del mismo mercado y se emitió como mucho `lookback_days` antes. Devuelve los cuantiles
    calibrados en el orden canónico de la evaluación y un resumen sin objetivos.
    """
    record = checked_record(record)
    _require(weighting in WEIGHTINGS, f"La ponderación debe ser una de {WEIGHTINGS}")
    config = record["config"]
    panel = evaluation[0]
    past = _rows(record, *calibration, history, "calibración")
    current = _rows(record, *evaluation, history, "evaluación")
    _require(
        int(panel.prediction_at.min()) >= record["fitted_until"],
        "La escala usa etiquetas que maduran después de una emisión evaluada",
    )
    pool = {key: np.concatenate([past[key], current[key]]) for key in past}
    defined = ~np.isnan(pool["context"]).any(axis=1)
    _, pairs, median = cqr.interval_pairs(record["levels"], record["nominals"])
    corrections = np.empty((panel.rows, len(pairs)))
    effective = np.empty(panel.rows)
    sizes = []
    lookback = config["lookback_days"] * DAY_US
    for code, name in enumerate(panel.markets):
        own = (pool["market"] == name) & defined
        selected = np.flatnonzero(panel.market == code)
        for session, at in enumerate(np.unique(panel.prediction_at[selected])):
            now = selected[panel.prediction_at[selected] == at]
            _require(
                not np.isnan(current["context"][now]).any(),
                "Una fila evaluada no tiene contexto definido en su emisión",
            )
            mask = own & (pool["matures_at"] <= at) & (pool["prediction_at"] >= at - lookback)
            count = int(np.count_nonzero(mask))
            _require(
                count >= config["min_rows"],
                f"{name} no tiene {config['min_rows']} errores maduros antes de {at}",
            )
            sizes.append(count)
            keys = pool["context"][mask]
            if weighting == "shuffled":
                sequence = np.random.SeedSequence([config["seed"], code, session])
                keys = keys[np.random.default_rng(sequence).permutation(count)]
            orders = [np.argsort(pool["scores"][mask, k], kind="stable") for k in range(len(pairs))]
            sorted_scores = [pool["scores"][mask, k][order] for k, order in enumerate(orders)]
            if weighting == "uniform":
                weights = np.full((count, 1), 1.0 / count)
                for k, (nominal, _, _) in enumerate(pairs):
                    corrections[now, k] = _sorted_quantile(sorted_scores[k], weights, nominal)[0]
                effective[now] = count
                continue
            block = max(1, MAX_BLOCK // count)
            for start in range(0, len(now), block):
                rows = now[start : start + block]
                weights = _weights(current["context"][rows], keys, config["bandwidth"])
                for k, (nominal, _, _) in enumerate(pairs):
                    corrections[rows, k] = _sorted_quantile(
                        sorted_scores[k], weights[orders[k]], nominal
                    )
                effective[rows] = 1.0 / np.square(weights).sum(axis=0)
    calibrated, adjusted = cqr.widen_in_order(panel.quantiles, pairs, median, corrections)
    summary = dict(
        identity=IDENTITY,
        weighting=weighting,
        stage_sha256=stage_sha256(record),
        rows=panel.rows,
        sessions=len(sizes),
        past_rows=dict(min=min(sizes), max=max(sizes)),
        effective_rows=dict(min=float(effective.min()), mean=float(effective.mean())),
        order_adjusted_rows=adjusted,
    )
    return calibrated, summary
