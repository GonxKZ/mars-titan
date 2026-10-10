"""Paridad de las puntuaciones de cuantiles e intervalos con scoringrules 0.11.0.

Compara con el backend `numpy` de scoringrules las tres implementaciones del proyecto:
`pinball_loss` de la cabeza de cuantiles, `score_sessions` de la comparación walk-forward y
`session_scores`, la reimplementación independiente de la auditoría. Los cuantiles y
objetivos son valores fijos escritos en la prueba, no salen de ningún modelo.

Tolerancia declarada antes de ejecutar, elemento a elemento en FP64:
|a − b| ≤ 1e-12 + 1e-14 · |b|. El término absoluto es el pedido para los valores de
rentabilidad. El relativo solo actúa por encima de 100, en las filas extremas: la puntuación
de intervalo suma términos no negativos con a lo sumo cinco redondeos, así que las dos
implementaciones pueden diferir como mucho en unos 5 · 2⁻⁵³ ≈ 5,6·10⁻¹⁶ relativos. Un
cambio de convención, como invertir el signo de la pinball o no dividir la penalización
entre α/2, mueve el valor en un factor de orden uno.

La relación con el CRPS es la de Bracher et al. (2021): con los pesos canónicos, la
puntuación de intervalo ponderada de los K intervalos centrales y la mediana es dos veces
la pinball media de sus 2K + 1 niveles, que es también la aproximación del CRPS por
cuantiles de scoringrules. No es el CRPS de la distribución completa.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from mars_titan.calibration.conformal_quantiles import (
    apply_conformal_quantiles,
    fit_conformal_quantiles,
)
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.forecast_scores import score_sessions
from mars_titan.integrity.independent_scores import session_scores
from mars_titan.models.quantile_head import LEVELS, pinball_loss
from tests.suite_support import reference_module

pytestmark = pytest.mark.external_reference

ATOL, RTOL = 1e-12, 1e-14
# Pares centrales (inferior, superior) y su cobertura nominal, del más estrecho al más ancho.
PAIRS = ((1, 3, "0.8"), (0, 4, "0.95"))
COLUMNS = [f"q{index}" for index in range(len(LEVELS))]
DAY = np.datetime64("2023-03-01T20:00:00", "us")


@pytest.fixture(scope="module")
def sr():
    module = reference_module("scoringrules")
    assert module.__version__ == "0.11.0"
    return module


def close(actual, expected, label):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected)
    assert actual.shape == expected.shape, label
    difference = np.abs(actual - expected)
    bound = ATOL + RTOL * np.abs(expected)
    assert bool(np.all(difference <= bound)), f"{label}: máximo {difference.max():.3e}"


def regular_rows(rng, rows):
    target = rng.normal(0.0, 0.02, rows)
    center = rng.normal(0.0, 0.01, rows)
    spread = np.abs(rng.normal(0.02, 0.005, rows))
    offsets = np.array([-1.96, -1.2816, 0.0, 1.2816, 1.96])
    return target, center[:, None] + spread[:, None] * offsets[None, :]


def edge_rows():
    """Filas límite: empates entre cuantiles, objetivo sobre un cuantil o un extremo,
    objetivo muy lejos del intervalo y magnitudes grandes."""
    q = np.array([-0.03, -0.01, 0.002, 0.015, 0.04])
    rows = [
        (0.0, np.zeros(5)),  # masa puntual con el objetivo sobre ella
        (0.01, np.zeros(5)),  # masa puntual con el objetivo por encima
        (-0.01, np.zeros(5)),  # y por debajo
        (0.0, np.array([-0.02, -0.02, -0.02, 0.01, 0.01])),  # colas empatadas
        (0.03, np.array([-0.02, 0.0, 0.0, 0.0, 0.03])),  # centro empatado, objetivo en el extremo
    ]
    rows += [(float(value), q) for value in q]  # objetivo exactamente sobre cada cuantil
    rows += [(-5.0, q), (5.0, q)]  # muy lejos por debajo y por encima del intervalo del 95 %
    rows += [(-0.02, q), (0.02, q)]  # dentro del 95 % y fuera del 80 %
    big = 1e6 * q
    rows += [(-3e7, big), (2.5e7, big), (1e4, big)]  # magnitudes grandes
    target = np.array([row[0] for row in rows])
    return target, np.stack([row[1] for row in rows])


@pytest.fixture(scope="module")
def fixture():
    target, quantiles = regular_rows(np.random.default_rng(13), 400)
    edge_target, edge_quantiles = edge_rows()
    target = np.concatenate([target, edge_target])
    quantiles = np.concatenate([quantiles, edge_quantiles])
    assert np.all(np.diff(quantiles, axis=1) >= 0)
    return target, quantiles


def panel(target, quantiles, sessions):
    """Panel de US con las sesiones indicadas, en el mismo orden que las filas."""
    times = DAY + sessions.astype("timedelta64[D]")
    return ForecastPanel.from_columns(
        np.array([f"r{index:05d}" for index in range(len(target))]),
        np.array(["US"] * len(target)),
        times,
        target,
        quantiles[:, 2],
        quantiles=quantiles,
        levels=LEVELS,
    )


def frame(target, quantiles, sessions):
    times = (DAY + sessions.astype("timedelta64[D]")).astype("datetime64[us]").astype(np.int64)
    columns = dict(market="US", prediction_at=times, target=target, prediction=quantiles[:, 2])
    return pd.DataFrame(columns | {name: quantiles[:, j] for j, name in enumerate(COLUMNS)})


def their_pinball(sr, target, quantiles):
    """Pinball por fila y nivel con quantile_score de scoringrules."""
    levels = np.array(LEVELS)[None, :]
    return sr.quantile_score(target[:, None], quantiles, levels, backend="numpy")


def their_interval(sr, target, quantiles, lower, upper):
    """Puntuación del intervalo central [q_inferior, q_superior] con α = 2 τ_inferior."""
    alpha = 2 * LEVELS[lower]
    return sr.interval_score(
        target, quantiles[:, lower], quantiles[:, upper], alpha, backend="numpy"
    )


def session_means(values, sessions):
    counts = np.bincount(sessions)
    if values.ndim == 1:
        return np.bincount(sessions, weights=values) / counts
    return np.column_stack([np.bincount(sessions, weights=column) / counts for column in values.T])


def test_training_pinball_matches_the_mean_quantile_score(sr, fixture):
    target, quantiles = fixture
    ours = pinball_loss(
        torch.from_numpy(quantiles), torch.from_numpy(target), reduction="none"
    ).numpy()
    close(ours, their_pinball(sr, target, quantiles).mean(axis=1), "pinball_loss")


@pytest.mark.parametrize("rows_per_session", [1, 4])
def test_walk_forward_scores_match_per_level_and_per_interval(sr, fixture, rows_per_session):
    target, quantiles = fixture
    sessions = np.arange(len(target)) // rows_per_session
    scores = score_sessions(panel(target, quantiles, sessions))
    close(scores.pinball, session_means(their_pinball(sr, target, quantiles), sessions), "pinball")
    assert [nominal for nominal, _, _ in scores.intervals] == [0.8, 0.95]
    for index, (lower, upper, _) in enumerate(PAIRS):
        expected = session_means(their_interval(sr, target, quantiles, lower, upper), sessions)
        close(scores.interval_score[:, index], expected, f"interval_score {index}")


def test_independent_scores_match_per_level_and_per_interval(sr, fixture):
    target, quantiles = fixture
    sessions = np.arange(len(target))
    result = session_scores(
        frame(target, quantiles, sessions), levels=LEVELS, quantile_columns=COLUMNS
    )
    pinball = their_pinball(sr, target, quantiles)
    for j, level in enumerate(LEVELS):
        close(result[f"pinball_{level}"].to_numpy(), pinball[:, j], f"pinball_{level}")
    for lower, upper, nominal in PAIRS:
        expected = their_interval(sr, target, quantiles, lower, upper)
        close(result[f"interval_score_{nominal}"].to_numpy(), expected, nominal)


def canonical_wis(sr, target, quantiles):
    """WIS de Bracher et al. (2021) con w_0 = 1/2 y w_k = α_k / 2 a partir de interval_score."""
    total = 0.5 * np.abs(target - quantiles[:, 2])
    for lower, upper, _ in PAIRS:
        total = total + LEVELS[lower] * their_interval(sr, target, quantiles, lower, upper)
    return total / (len(PAIRS) + 0.5)


def test_twice_the_mean_pinball_is_the_quantile_crps_and_the_canonical_wis(sr, fixture):
    target, quantiles = fixture
    ours = (
        2
        * pinball_loss(
            torch.from_numpy(quantiles), torch.from_numpy(target), reduction="none"
        ).numpy()
    )
    close(sr.crps_quantile(target, quantiles, np.array(LEVELS), backend="numpy"), ours, "crps")
    close(canonical_wis(sr, target, quantiles), ours, "wis")


def test_scoringrules_0_11_numpy_wis_adds_the_median_instead_of_its_error(sr, fixture):
    """Defecto conocido de la versión fijada, por eso la WIS se compone con interval_score.

    En 0.11.0 el backend numpy suma w_0 · m en lugar de w_0 · |y − m|
    (https://github.com/frazane/scoringrules/issues/140). Si una versión posterior lo
    corrige, esta prueba falla y la nota de `metrics.md` debe revisarse.
    """
    target, quantiles = fixture
    lower = quantiles[:, [lower for lower, _, _ in PAIRS]]
    upper = quantiles[:, [upper for _, upper, _ in PAIRS]]
    alpha = np.array([2 * LEVELS[lower] for lower, _, _ in PAIRS])
    theirs = sr.weighted_interval_score(
        target, quantiles[:, 2], lower, upper, alpha, backend="numpy"
    )
    shift = 0.5 * (quantiles[:, 2] - np.abs(target - quantiles[:, 2])) / (len(PAIRS) + 0.5)
    close(theirs, canonical_wis(sr, target, quantiles) + shift, "wis 0.11.0")
    assert np.abs(shift).max() > 1e-3


def test_interval_scores_after_static_cqr_match(sr, fixture):
    """La puntuación de los intervalos corregidos por la CQR estática también coincide."""
    target, quantiles = fixture
    rows = len(target)
    groups = np.where(np.arange(rows) % 3 == 0, "CN", "US")
    calibration = np.arange(rows) < rows // 2
    record = fit_conformal_quantiles(
        target[calibration],
        quantiles[calibration],
        groups[calibration],
        levels=LEVELS,
        nominals=[0.8, 0.95],
        min_rows=20,
    )
    calibrated, _ = apply_conformal_quantiles(record, quantiles[~calibration], groups[~calibration])
    evaluated = target[~calibration]
    assert not np.array_equal(calibrated, quantiles[~calibration])
    scores = score_sessions(panel(evaluated, calibrated, np.arange(len(evaluated))))
    for index, (lower, upper, _) in enumerate(PAIRS):
        expected = their_interval(sr, evaluated, calibrated, lower, upper)
        close(scores.interval_score[:, index], expected, f"cqr interval_score {index}")
