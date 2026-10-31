"""Diagnóstico de escala, ordenación y contracción de predicciones ya guardadas.

Con un residuo de media cercana a cero, predecir cero es la máxima contracción y toda
amplitud sin señal suma error. Una predicción x que ordena bien pero es demasiado amplia
mejora con λ·x y λ < 1. Este módulo mide la ordenación por sesión, la pendiente de
calibración y el λ que minimiza el MAE por sesión en una partición anterior, y lo aplica
sin cambios en la evaluación. No ajusta ningún modelo. La evaluación solo elige λ en la
cota con información futura, que se informa aparte y nunca como resultado.

Todas las medias dan el mismo peso a cada sesión (mercado e instante), igual que el MAE
por sesión de las comparaciones. Los intervalos usan el bootstrap circular por bloques de
días UTC de ``paired_comparisons`` y su corrección por máximo estudentizado.
"""

import math
from collections import deque
from statistics import NormalDist

import numpy as np
from scipy.stats import rankdata

from mars_titan.evaluation.paired_comparisons import circular_block_counts, family_intervals


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def session_weights(panel):
    """Peso 1/n de cada fila de una sesión con n activos, en el orden canónico del panel."""
    return np.repeat(1.0 / panel.session_samples, panel.session_samples)


def session_mae(panel, prediction):
    """MAE de cada sesión para una predicción alineada con las filas del panel."""
    error = np.abs(np.asarray(prediction, dtype=np.float64) - panel.target)
    return np.add.reduceat(error, panel.session_starts) / panel.session_samples


def shrinkage_factor(target, prediction, weights):
    """Menor λ en [0, 1] que minimiza Σ w·|y − λ·x|, o None si todas las x son cero.

    Con x ≠ 0, |y − λx| = |x|·|y/x − λ|. La función es convexa y su mínimo sin restricción
    es la mediana de y/x ponderada con w·|x|. Recortar ese mínimo a [0, 1] da el mínimo
    restringido. Entre varios mínimos se toma el menor, el más próximo a predecir cero.
    """
    target, prediction, weights = (
        np.asarray(v, dtype=np.float64) for v in (target, prediction, weights)
    )
    _require(
        target.shape == prediction.shape == weights.shape and target.ndim == 1,
        "Objetivo, predicción y pesos deben ser vectores alineados",
    )
    _require(
        np.isfinite(target).all() and np.isfinite(prediction).all() and np.all(weights > 0),
        "Los valores deben ser finitos y los pesos positivos",
    )
    active = prediction != 0
    if not active.any():
        return None
    ratio = target[active] / prediction[active]
    order = np.argsort(ratio, kind="stable")
    cumulative = np.cumsum((weights[active] * np.abs(prediction[active]))[order])
    median = ratio[order][np.searchsorted(cumulative, cumulative[-1] / 2, side="left")]
    return float(np.clip(median, 0.0, 1.0))


def market_shrinkage(panel, prediction):
    """λ de cada mercado del panel con el MAE por sesión como objetivo."""
    prediction = np.asarray(prediction, dtype=np.float64)
    weights = session_weights(panel)
    result = {}
    for code, market in enumerate(panel.markets):
        rows = panel.market == code
        if rows.any():
            result[market] = shrinkage_factor(panel.target[rows], prediction[rows], weights[rows])
    return result


def apply_shrinkage(panel, prediction, factors):
    """Multiplicar cada fila por el λ de su mercado. Un mercado sin λ predice cero."""
    scale = np.zeros(panel.rows)
    for code, market in enumerate(panel.markets):
        factor = factors.get(market)
        scale[panel.market == code] = 0.0 if factor is None else factor
    return np.asarray(prediction, dtype=np.float64) * scale


def calibration_slope(panel, prediction):
    """Pendiente sin constante de y sobre x con pesos por sesión, o None si x es nula.

    Es el factor que minimiza el error cuadrático de b·x. Si vale 0,25, la predicción
    tiene cuatro veces la amplitud que le correspondería con su correlación.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    weights = session_weights(panel)
    denominator = math.fsum(weights * prediction * prediction)
    if denominator == 0:
        return None
    return math.fsum(weights * prediction * panel.target) / denominator


def session_correlations(panel, prediction, *, rank=False, min_assets=3):
    """Correlación de Pearson, o de rangos, entre predicción y objetivo en cada sesión.

    Devuelve los valores y una máscara. Una sesión con menos de ``min_assets`` filas o
    con predicción u objetivo constantes no tiene correlación definida y vale cero.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    values = np.zeros(panel.sessions)
    defined = np.zeros(panel.sessions, dtype=bool)
    for session, (start, count) in enumerate(
        zip(panel.session_starts, panel.session_samples, strict=True)
    ):
        observed = panel.target[start : start + count]
        predicted = prediction[start : start + count]
        if count < min_assets or np.ptp(observed) == 0 or np.ptp(predicted) == 0:
            continue
        if rank:
            observed, predicted = rankdata(observed), rankdata(predicted)
        value = float(np.corrcoef(observed, predicted)[0, 1])
        if math.isfinite(value):
            values[session], defined[session] = value, True
    return values, defined


def past_median(assets, sessions, target, *, lag, window, min_labels):
    """Mediana de las últimas etiquetas del mismo activo con al menos ``lag`` sesiones de retraso.

    ``sessions`` numera las sesiones del mercado en orden temporal. Con lag = 2, la fila de
    la sesión t solo ve etiquetas emitidas en t − 2 o antes, que ya habían madurado. Con
    menos de ``min_labels`` etiquetas la fila predice cero.
    """
    assets, sessions = np.asarray(assets), np.asarray(sessions, dtype=np.int64)
    target = np.asarray(target, dtype=np.float64)
    _require(
        assets.shape == sessions.shape == target.shape and target.ndim == 1,
        "Activos, sesiones y etiquetas deben ser vectores alineados",
    )
    _require(
        type(lag) is int and lag >= 1 and type(window) is int and 1 <= min_labels <= window,
        "El retraso, la ventana y el mínimo de etiquetas no son válidos",
    )
    result = np.zeros(len(target))
    order = np.lexsort((sessions, assets))
    starts = np.flatnonzero(np.r_[True, assets[order][1:] != assets[order][:-1]])
    for begin, end in zip(starts, np.r_[starts[1:], len(order)], strict=True):
        rows = order[begin:end]
        _require(
            np.all(np.diff(sessions[rows]) > 0), "Un activo repite sesión en el control pasado"
        )
        recent, pending = deque(maxlen=window), 0
        for row in rows:
            while sessions[rows[pending]] <= sessions[row] - lag:
                recent.append(target[rows[pending]])
                pending += 1
            if len(recent) >= min_labels:
                result[row] = float(np.median(recent))
    return result


def block_family(values, session_period, *, block_length, replicates, seed, confidence):
    """Medias por columna con intervalos marginales y simultáneos por bloques de días UTC.

    ``values`` tiene una fila por sesión y una columna por contraste, todas definidas.
    Cada réplica remuestrea días completos con los mismos bloques para todas las columnas.
    """
    values = np.asarray(values, dtype=np.float64)
    period = np.asarray(session_period, dtype=np.int64)
    _require(
        values.ndim == 2 and values.shape[0] == len(period) and values.shape[1] >= 1,
        "La familia necesita una fila por sesión",
    )
    _require(np.isfinite(values).all(), "La familia contiene valores no finitos")
    periods = int(period.max()) + 1
    _require(1 <= block_length < periods, "Se necesitan más días que la longitud del bloque")
    estimate = values.mean(axis=0)
    counts = circular_block_counts(np.random.default_rng(seed), replicates, periods, block_length)
    weights = counts[:, period].astype(np.float64)
    draws = (weights @ values) / weights.sum(axis=1, keepdims=True)
    family = family_intervals(estimate, draws, confidence)
    return dict(
        estimate=estimate,
        lower=family["lower"],
        upper=family["upper"],
        joint_lower=family["joint_lower"],
        joint_upper=family["joint_upper"],
        critical=family["critical"],
    )


def minimum_detectable_effect(differential, *, candidates, alpha, power):
    """Diferencia media por sesión detectable con Bonferroni unilateral y autocorrelación AR(1).

    El tamaño efectivo es n(1 − ρ)/(1 + ρ) con la autocorrelación de orden uno ρ recortada
    a [0, 0,99]. Devuelve el efecto en unidades de la serie y en desviaciones típicas. Es una
    aproximación optimista frente al intervalo simultáneo bilateral que decide en la familia.
    """
    differential = np.asarray(differential, dtype=np.float64)
    _require(differential.ndim == 1 and len(differential) >= 3, "La serie es demasiado corta")
    spread = float(np.std(differential, ddof=1))
    centred = differential - differential.mean()
    energy = float(np.dot(centred, centred))
    rho = float(np.dot(centred[1:], centred[:-1]) / energy) if energy > 0 else 0.0
    clipped = min(max(rho, 0.0), 0.99)
    effective = len(differential) * (1 - clipped) / (1 + clipped)
    normal = NormalDist()
    effects = {}
    for count in candidates:
        z = normal.inv_cdf(1 - alpha / count) + normal.inv_cdf(power)
        effects[str(count)] = dict(
            standardized=z / math.sqrt(effective), absolute=z * spread / math.sqrt(effective)
        )
    return dict(
        standard_deviation=spread,
        autocorrelation=rho,
        effective_sessions=effective,
        effects=effects,
    )


def ensure_before(earlier, later):
    """Exigir que la partición que elige λ termine antes de que empiece la evaluación."""
    _require(
        int(earlier.prediction_at.max()) < int(later.prediction_at.min()),
        "La partición que elige λ debe ser anterior a la evaluación",
    )


def _mean_or_none(values, defined):
    selected = values[defined]
    return math.fsum(selected) / len(selected) if len(selected) else None


def diagnose_predictor(fit, evaluation, *, fit_prediction, prediction):
    """Medidas de un predictor con λ elegido en ``fit`` y aplicado sin cambios a ``evaluation``."""
    ensure_before(fit, evaluation)
    factors = market_shrinkage(fit, fit_prediction)
    shrunk = apply_shrinkage(evaluation, prediction, factors)
    bound = market_shrinkage(evaluation, prediction)
    pearson, pearson_defined = session_correlations(evaluation, prediction)
    rank, rank_defined = session_correlations(evaluation, prediction, rank=True)
    zero = session_mae(evaluation, np.zeros(evaluation.rows))
    raw = session_mae(evaluation, prediction)
    defined_pearson = pearson[pearson_defined]
    return dict(
        series=dict(
            raw=raw,
            shrunk=session_mae(evaluation, shrunk),
            bound=session_mae(evaluation, apply_shrinkage(evaluation, prediction, bound)),
            pearson=pearson,
            pearson_defined=pearson_defined,
            zero=zero,
        ),
        shrinkage=factors,
        shrinkage_with_future_information=bound,
        fit_slope=calibration_slope(fit, fit_prediction),
        evaluation_slope=calibration_slope(evaluation, prediction),
        mean_pearson_ic=_mean_or_none(pearson, pearson_defined),
        mean_rank_ic=_mean_or_none(rank, rank_defined),
        pearson_icir=(
            float(defined_pearson.mean() / np.std(defined_pearson, ddof=1))
            if len(defined_pearson) > 2 and np.std(defined_pearson, ddof=1) > 0
            else None
        ),
        undefined_ic_sessions=int(np.sum(~pearson_defined)),
        prediction_scale=float(np.median(np.abs(prediction))),
        target_scale=float(np.median(np.abs(evaluation.target))),
    )
