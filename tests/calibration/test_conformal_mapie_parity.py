"""Paridad de la CQR estática con `ConformalizedQuantileRegressor` de MAPIE 1.5.0.

MAPIE recibe tres estimadores ya ajustados (inferior, superior y mediana) que solo
devuelven cuantiles guardados, y se ejecuta grupo a grupo porque su CQR no tiene grupos.
Se compara con `symmetric_correction=True`, que usa la misma puntuación
E = max(q_bajo − y, y − q_alto) que `fit_conformal_quantiles`. No se ajusta ningún modelo.

Las dos implementaciones toman un estadístico de orden distinto en algunos tamaños de
calibración, lo que se documenta aquí con su regla exacta:

- La propia usa el orden ⌈(n + 1)(1 − α)⌉ de Romano, Patterson y Candès (2019), en
  aritmética racional, y deja la corrección indefinida si supera n.
- MAPIE calcula `np.quantile(E, (1 − α)(1 + 1/n), method="higher")`, cuyo índice es
  ⌈(1 − α)(1 + 1/n)(n − 1)⌉ empezando en cero. Cuando (n + 1)(1 − α) es entero toma el
  estadístico siguiente, más conservador. Con 0,8 ocurre si n + 1 es múltiplo de 5 y con
  0,95 si es múltiplo de 20.
- MAPIE rechaza n < max(1/α, 1/(1 − α)), con α = 1 − cobertura calculada en decimal. La
  propia ya está definida en n = 4 con 0,8 y en n = 19 con 0,95, donde el orden es n y
  la corrección es la puntuación máxima. MAPIE rechaza esos dos tamaños.

En los demás casos las correcciones, y por tanto los intervalos, coinciden bit a bit.
"""

import math
import warnings
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from mars_titan.calibration.conformal_quantiles import (
    apply_conformal_quantiles,
    fit_conformal_quantiles,
)
from mars_titan.models.quantile_head import LEVELS
from tests.suite_support import reference_module

pytestmark = pytest.mark.external_reference

NOMINALS = (0.8, 0.95)
# Índices (inferior, superior) del intervalo central de cada cobertura en LEVELS.
PAIRS = {0.8: (1, 3), 0.95: (0, 4)}
MEDIAN = LEVELS.index(0.5)


@pytest.fixture(scope="module")
def regressor():
    mapie = reference_module("mapie")
    assert mapie.__version__ == "1.5.0"
    return reference_module("mapie.regression").ConformalizedQuantileRegressor


class Stored:
    """Estimador ya ajustado que devuelve los cuantiles guardados de cada fila pedida."""

    fitted_ = True

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def fit(self, X, y):
        raise AssertionError("Un estimador prefit no debe ajustarse")

    def predict(self, X):
        return self.values[np.asarray(X, dtype=np.int64)[:, 0]]


def mapie_interval(regressor, calibration, target, evaluation, nominal):
    """Límites y mediana de MAPIE para las filas de evaluación de un grupo."""
    lower, upper = PAIRS[nominal]
    values = np.concatenate([calibration, evaluation])
    estimators = [Stored(values[:, index]) for index in (lower, upper, MEDIAN)]
    model = regressor(estimators, confidence_level=nominal, prefit=True)
    model.conformalize(np.arange(len(calibration))[:, None], target)
    rows = len(calibration) + np.arange(len(evaluation))
    points, bounds = model.predict_interval(rows[:, None], symmetric_correction=True)
    return bounds[:, 0, 0], bounds[:, 1, 0], points


def quantiles(rng, rows):
    center = rng.normal(0.0, 0.01, rows)
    spread = np.abs(rng.normal(0.02, 0.005, rows))
    offsets = np.array([-1.96, -1.2816, 0.0, 1.2816, 1.96])
    return center[:, None] + spread[:, None] * offsets[None, :]


def integer_order(rows, nominal):
    return ((rows + 1) * Fraction(str(nominal))).denominator == 1


def test_per_group_corrections_and_intervals_match_exactly(regressor):
    rng = np.random.default_rng(29)
    sizes = {"US": (400, 120), "CN": (257, 90)}
    # En estos tamaños (n + 1)(1 − α) no es entero y los dos órdenes coinciden.
    assert not any(integer_order(n, nominal) for n, _ in sizes.values() for nominal in NOMINALS)
    parts = {name: quantiles(rng, n + m) for name, (n, m) in sizes.items()}
    # Objetivos con más dispersión que los cuantiles en US y menos en CN: una corrección
    # ensancha y la otra estrecha.
    noise = {"US": 0.035, "CN": 0.012}
    targets = {name: rng.normal(0.0, noise[name], len(parts[name])) for name in parts}
    calibration = {name: slice(0, n) for name, (n, _) in sizes.items()}
    evaluation = {name: slice(n, None) for name, (n, _) in sizes.items()}
    names = list(sizes)
    record = fit_conformal_quantiles(
        np.concatenate([targets[name][calibration[name]] for name in names]),
        np.concatenate([parts[name][calibration[name]] for name in names]),
        np.concatenate([[name] * sizes[name][0] for name in names]),
        levels=LEVELS,
        nominals=list(NOMINALS),
        min_rows=1,
    )
    held_out = np.concatenate([parts[name][evaluation[name]] for name in names])
    groups = np.concatenate([[name] * sizes[name][1] for name in names])
    calibrated, adjusted = apply_conformal_quantiles(record, held_out, groups)
    # MAPIE no reordena intervalos anidados, así que el caso normal no debe necesitarlo.
    assert adjusted == {"0.8": 0, "0.95": 0}
    signs = set()
    for name in names:
        rows = groups == name
        for nominal in NOMINALS:
            entry = record["groups"][name]["intervals"][f"{nominal:g}"]
            signs.add(np.sign(entry["correction"]))
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                low, high, median = mapie_interval(
                    regressor,
                    parts[name][calibration[name]],
                    targets[name][calibration[name]],
                    parts[name][evaluation[name]],
                    nominal,
                )
            lower, upper = PAIRS[nominal]
            np.testing.assert_array_equal(calibrated[rows, lower], low)
            np.testing.assert_array_equal(calibrated[rows, upper], high)
            np.testing.assert_array_equal(calibrated[rows, MEDIAN], median)
    assert signs == {-1.0, 1.0}


def separated_scores(rows, rng):
    """Calibración con puntuaciones distintas y conocidas: E_i = s_i con q = (−1, 1).

    Las s_i son múltiplos de 2⁻⁸, así que |y| − 1 = s_i es exacto en coma flotante.
    """
    scores = rng.permutation(np.arange(1, rows + 1) / 256.0)
    signs = np.where(np.arange(rows) % 2 == 0, 1.0, -1.0)
    target = signs * (1.0 + scores)
    calibration = np.tile([-1.0, -1.0, 0.0, 1.0, 1.0], (rows, 1))
    return target, calibration, np.sort(scores)


@pytest.mark.parametrize("nominal", NOMINALS)
def test_small_calibration_sets_follow_the_documented_rules(regressor, nominal):
    """Recorre n = 1 a 120 y clasifica cada tamaño según la regla del docstring."""
    rng = np.random.default_rng(3)
    # MAPIE pasa la cobertura a α en decimal, así que α vale exactamente 0,2 o 0,05.
    alpha = float(Decimal("1") - Decimal(str(nominal)))
    refused = {n for n in range(1, 121) if n < max(1 / alpha, 1 / (1 - alpha))}
    observed = dict(undefined=[], refused_by_mapie=[], one_rank_higher=[], equal=[])
    for rows in range(1, 121):
        target, calibration, scores = separated_scores(rows, rng)
        record = fit_conformal_quantiles(
            target,
            calibration,
            np.array(["US"] * rows),
            levels=LEVELS,
            nominals=list(NOMINALS),
            min_rows=1,
        )
        entry = record["groups"]["US"]["intervals"][f"{nominal:g}"]
        order = entry["order_statistic"]
        assert order == math.ceil((rows + 1) * Fraction(str(nominal)))
        evaluation = np.zeros((1, len(LEVELS)))
        try:
            _, high, _ = mapie_interval(regressor, calibration, target, evaluation, nominal)
        except ValueError as error:
            assert "Number of samples of the score is too low" in str(error)
            mapie_rank = None
        else:
            mapie_rank = int(np.flatnonzero(scores == high[0])[0])
        if entry["correction"] is None:
            assert order > rows and mapie_rank is None
            assert entry["reason"] == "El orden requerido supera las filas de calibración"
            observed["undefined"].append(rows)
            continue
        assert entry["correction"] == scores[order - 1]
        if mapie_rank is None:
            assert rows in refused and order == rows
            observed["refused_by_mapie"].append(rows)
        elif integer_order(rows, nominal):
            assert mapie_rank == order
            observed["one_rank_higher"].append(rows)
        else:
            assert mapie_rank == order - 1
            observed["equal"].append(rows)
    undefined, refused_only, period = {
        0.8: ([1, 2, 3], [4], 5),
        0.95: (list(range(1, 19)), [19], 20),
    }[nominal]
    higher = [n for n in range(1, 121) if (n + 1) % period == 0 and n not in refused]
    others = set(undefined) | set(refused_only) | set(higher)
    assert observed == dict(
        undefined=undefined,
        refused_by_mapie=refused_only,
        one_rank_higher=higher,
        equal=[n for n in range(1, 121) if n not in others],
    )
