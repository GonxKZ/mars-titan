import ast
import math
from pathlib import Path

import numpy as np
import pytest

from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation.forecast_panel import ForecastPanel

DAY = 86_400_000_000


def _panel(target, prediction, *, sessions, start=0, assets=None):
    """Panel de un mercado con ``sessions`` días consecutivos y el mismo número de activos."""
    target = np.asarray(target, dtype=np.float64)
    count = len(target) // sessions
    assets = assets or [f"A{i}" for i in range(count)]
    times = np.repeat(np.arange(start, start + sessions) * DAY, count)
    identities = [f"{asset}/{time}" for time, asset in zip(times, assets * sessions, strict=True)]
    return ForecastPanel.from_columns(
        identities, np.array(["US"] * len(target)), times, target, np.asarray(prediction, float)
    )


def _objective(target, prediction, weights, factor):
    return math.fsum(weights * np.abs(target - factor * prediction))


@pytest.mark.parametrize("seed", range(6))
def test_shrinkage_factor_matches_a_dense_grid(seed):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=400)
    target = 0.3 * signal + rng.normal(size=400)
    prediction = rng.uniform(0.5, 6.0) * signal
    weights = rng.uniform(0.2, 2.0, size=400)
    factor = sd.shrinkage_factor(target, prediction, weights)
    assert 0.0 <= factor <= 1.0
    best = _objective(target, prediction, weights, factor)
    grid = np.linspace(0, 1, 2001)
    assert best <= min(_objective(target, prediction, weights, g) for g in grid) + 1e-9


def test_shrinkage_factor_recovers_a_known_amplitude():
    target = np.array([1.0, -2.0, 0.5, 3.0])
    assert sd.shrinkage_factor(target, 4 * target, np.ones(4)) == pytest.approx(0.25)


def test_shrinkage_factor_takes_the_smallest_minimizer_and_clips():
    # Con dos filas de igual peso, cualquier λ entre 0,2 y 0,6 es mínimo: se elige 0,2.
    target, prediction = np.array([0.2, 0.6]), np.ones(2)
    assert sd.shrinkage_factor(target, prediction, np.ones(2)) == pytest.approx(0.2)
    assert sd.shrinkage_factor(-np.abs(target), prediction, np.ones(2)) == 0.0
    assert sd.shrinkage_factor(10 * target, prediction, np.ones(2)) == 1.0
    assert sd.shrinkage_factor(target, np.zeros(2), np.ones(2)) is None


def test_shrinkage_factor_rejects_misaligned_or_invalid_inputs():
    with pytest.raises(ValueError, match="alineados"):
        sd.shrinkage_factor(np.ones(3), np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="finitos"):
        sd.shrinkage_factor(np.array([np.nan, 1.0]), np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="positivos"):
        sd.shrinkage_factor(np.ones(2), np.ones(2), np.array([1.0, 0.0]))


def test_market_shrinkage_weights_each_session_equally():
    # La primera sesión tiene un activo y la segunda tres. Con pesos por fila ganaría la
    # segunda, con pesos por sesión ambas pesan lo mismo y el mínimo es el menor de los dos.
    times = np.array([0, DAY, DAY, DAY])
    panel = ForecastPanel.from_columns(
        ["a", "b", "c", "d"],
        np.array(["US"] * 4),
        times,
        np.array([0.1, 0.9, 0.9, 0.9]),
        np.ones(4),
    )
    assert sd.market_shrinkage(panel, panel.prediction) == {"US": pytest.approx(0.1)}


def test_apply_shrinkage_predicts_zero_without_a_factor():
    panel = _panel([1.0, 2.0], [3.0, 4.0], sessions=2)
    assert np.array_equal(sd.apply_shrinkage(panel, panel.prediction, {"US": None}), [0, 0])
    assert np.allclose(sd.apply_shrinkage(panel, panel.prediction, {"US": 0.5}), [1.5, 2.0])


def test_calibration_slope_measures_excess_amplitude():
    target = np.array([0.5, -1.0, 2.0, 0.25])
    panel = _panel(target, 4 * target, sessions=2)
    assert sd.calibration_slope(panel, panel.prediction) == pytest.approx(0.25)
    assert sd.calibration_slope(panel, np.zeros(4)) is None


def test_session_correlations_mark_undefined_sessions():
    target = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0]
    prediction = [1.0, 2.0, 3.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0]
    panel = _panel(target, prediction, sessions=3)
    values, defined = sd.session_correlations(panel, panel.prediction, rank=True)
    assert defined.tolist() == [True, True, False]
    assert values.tolist() == pytest.approx([1.0, -1.0, 0.0])
    pearson, _ = sd.session_correlations(panel, panel.prediction)
    assert 0.9 < pearson[0] < 1.0


def test_past_median_never_reads_the_last_matured_lag():
    # La etiqueta enorme de la sesión 4 solo puede entrar a partir de la sesión 6.
    sessions = np.arange(8)
    target = np.array([1.0, 1.0, 1.0, 1.0, 1000.0, 1.0, 1.0, 1.0])
    result = sd.past_median(np.array(["A"] * 8), sessions, target, lag=2, window=3, min_labels=2)
    assert result.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    result = sd.past_median(np.array(["A"] * 8), sessions, target, lag=2, window=1, min_labels=1)
    assert result[6] == 1000.0 and result[5] == 1.0


def test_past_median_keeps_assets_apart_and_rejects_repeated_sessions():
    assets = np.array(["A", "B", "A", "B", "A", "B"])
    sessions = np.array([0, 0, 1, 1, 2, 2])
    target = np.array([1.0, -5.0, 2.0, -6.0, 3.0, -7.0])
    result = sd.past_median(assets, sessions, target, lag=1, window=5, min_labels=1)
    assert result.tolist() == [0.0, 0.0, 1.0, -5.0, 1.5, -5.5]
    with pytest.raises(ValueError, match="repite sesión"):
        sd.past_median(
            np.array(["A", "A"]), np.array([0, 0]), np.ones(2), lag=1, window=2, min_labels=1
        )


def test_block_family_is_reproducible_and_contains_the_estimate():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(60, 3)) + np.array([0.0, 0.5, -0.5])
    period = np.arange(60)
    options = dict(block_length=5, replicates=400, seed=7, confidence=0.95)
    first = sd.block_family(values, period, **options)
    second = sd.block_family(values, period, **options)
    for key in ("estimate", "lower", "upper", "joint_lower", "joint_upper"):
        assert np.array_equal(first[key], second[key])
    assert np.allclose(first["estimate"], values.mean(axis=0))
    assert np.all(first["joint_lower"] <= first["lower"])
    assert np.all(first["joint_upper"] >= first["upper"])
    assert first["joint_lower"][1] > 0 and first["joint_upper"][2] < 0
    with pytest.raises(ValueError, match="más días"):
        sd.block_family(values, period, **{**options, "block_length": 60})


def test_minimum_detectable_effect_matches_the_published_approximation():
    # Con 250 sesiones independientes se detectan unas 0,21 y 0,24 desviaciones típicas.
    alternating = np.tile([1.0, -1.0], 125)
    result = sd.minimum_detectable_effect(alternating, candidates=[8, 32], alpha=0.05, power=0.8)
    assert result["autocorrelation"] < 0
    assert result["effective_sessions"] == 250
    assert result["effects"]["8"]["standardized"] == pytest.approx(0.211, abs=0.002)
    assert result["effects"]["32"]["standardized"] == pytest.approx(0.240, abs=0.002)
    flat = sd.minimum_detectable_effect(np.zeros(10), candidates=[8], alpha=0.05, power=0.8)
    assert flat["standard_deviation"] == 0 and flat["effects"]["8"]["absolute"] == 0


def test_positive_autocorrelation_reduces_the_effective_sessions():
    rng = np.random.default_rng(3)
    noise = rng.normal(size=400)
    series = np.convolve(noise, np.ones(5) / 5, mode="same")
    result = sd.minimum_detectable_effect(series, candidates=[8], alpha=0.05, power=0.8)
    assert result["autocorrelation"] > 0.3
    assert result["effective_sessions"] < 400 / 2


def test_diagnose_predictor_chooses_lambda_without_the_evaluation():
    rng = np.random.default_rng(11)
    signal = rng.normal(size=(2, 20 * 16))
    fit_target = 0.2 * signal[0] + rng.normal(size=signal.shape[1])
    fit = _panel(fit_target, 3 * signal[0], sessions=20)
    evaluation_target = 0.2 * signal[1] + rng.normal(size=signal.shape[1])
    evaluation = _panel(evaluation_target, 3 * signal[1], sessions=20, start=40)
    result = sd.diagnose_predictor(
        fit, evaluation, fit_prediction=fit.prediction, prediction=evaluation.prediction
    )
    # Cambiar las etiquetas de evaluación no mueve el λ elegido, solo la cota.
    altered = _panel(-evaluation_target, 3 * signal[1], sessions=20, start=40)
    again = sd.diagnose_predictor(
        fit, altered, fit_prediction=fit.prediction, prediction=altered.prediction
    )
    assert result["shrinkage"] == again["shrinkage"]
    assert again["shrinkage_with_future_information"] == {"US": 0.0}
    assert 0 < result["shrinkage"]["US"] < 0.5
    assert result["series"]["shrunk"].mean() < result["series"]["raw"].mean()
    assert result["series"]["bound"].mean() <= result["series"]["shrunk"].mean() + 1e-12
    with pytest.raises(ValueError, match="anterior a la evaluación"):
        sd.diagnose_predictor(
            evaluation, fit, fit_prediction=evaluation.prediction, prediction=fit.prediction
        )


def test_scale_diagnostics_do_not_import_pytorch():
    path = Path(sd.__file__)
    modules = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (
            node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")]
        )
    }
    assert "torch" not in modules


def test_ensure_before_rejects_overlapping_partitions():
    first = _panel([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], sessions=1)
    same_day = _panel([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], sessions=1)
    with pytest.raises(ValueError, match="anterior"):
        sd.ensure_before(first, same_day)
    sd.ensure_before(first, _panel([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], sessions=1, start=1))


def test_session_correlations_need_three_assets():
    two = _panel([1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 2.0, 1.0], sessions=2)
    assert not sd.session_correlations(two, two.prediction)[1].any()
    three = _panel([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], sessions=1)
    assert sd.session_correlations(three, three.prediction)[1].all()


def test_icir_uses_the_sample_deviation():
    target = np.tile([1.0, 2.0, 3.0, 4.0], 4)
    prediction = np.concatenate(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 4.0, 3.0], [2.0, 1.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
    )
    fit = _panel(target, prediction, sessions=4)
    evaluation = _panel(target, prediction, sessions=4, start=10)
    result = sd.diagnose_predictor(
        fit, evaluation, fit_prediction=fit.prediction, prediction=evaluation.prediction
    )
    values = result["series"]["pearson"]
    assert result["pearson_icir"] == pytest.approx(values.mean() / np.std(values, ddof=1))


def test_calibration_slope_weights_sessions_and_not_rows():
    # La sesión de un activo pesa lo mismo que la de tres: la pendiente no es la de filas.
    times = np.array([0, DAY, DAY, DAY])
    panel = ForecastPanel.from_columns(
        ["a", "b", "c", "d"],
        np.array(["US"] * 4),
        times,
        np.array([2.0, 1.0, 1.0, 1.0]),
        np.ones(4),
    )
    assert sd.calibration_slope(panel, panel.prediction) == pytest.approx(1.5)


def test_effective_sessions_clip_strong_autocorrelation():
    series = np.arange(2000, dtype=np.float64)
    result = sd.minimum_detectable_effect(series, candidates=[8], alpha=0.05, power=0.8)
    assert result["autocorrelation"] > 0.99
    assert result["effective_sessions"] == pytest.approx(2000 * 0.01 / 1.99)


def test_block_family_matches_explicit_circular_blocks():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(30, 2))
    period = np.arange(30)
    result = sd.block_family(values, period, block_length=4, replicates=50, seed=9, confidence=0.9)
    from mars_titan.evaluation.paired_comparisons import circular_block_indices

    index = circular_block_indices(np.random.default_rng(9), 50, 30, 4)
    draws = values[index].mean(axis=1)
    lower, upper = np.quantile(draws, [0.05, 0.95], axis=0, method="linear")
    assert np.allclose(result["lower"], lower) and np.allclose(result["upper"], upper)
    wider = sd.block_family(values, period, block_length=1, replicates=50, seed=9, confidence=0.9)
    assert not np.allclose(wider["lower"], result["lower"])


def test_shrunk_series_applies_the_fitted_factor():
    rng = np.random.default_rng(2)
    signal = rng.normal(size=(2, 10 * 8))
    fit = _panel(0.3 * signal[0] + rng.normal(size=80), 2 * signal[0], sessions=10)
    evaluation = _panel(0.3 * signal[1] + rng.normal(size=80), 2 * signal[1], sessions=10, start=20)
    result = sd.diagnose_predictor(
        fit, evaluation, fit_prediction=fit.prediction, prediction=evaluation.prediction
    )
    factor = sd.market_shrinkage(fit, fit.prediction)["US"]
    expected = sd.session_mae(evaluation, factor * evaluation.prediction)
    assert np.array_equal(result["series"]["shrunk"], expected)
    bound = result["shrinkage_with_future_information"]["US"]
    assert np.array_equal(
        result["series"]["bound"], sd.session_mae(evaluation, bound * evaluation.prediction)
    )
