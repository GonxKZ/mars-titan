"""Métricas por sesión con valores calculados a mano y propiedades de agregación."""

import json
import math

import numpy as np
import pytest
from scipy.stats import spearmanr

from mars_titan.evaluation import forecast_scores
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.forecast_scores import score_sessions, selective_risk
from mars_titan.evaluation.session_metrics import SessionErrors

LEVELS = (0.025, 0.1, 0.5, 0.9, 0.975)
DAY = np.datetime64("2023-03-01T00:00:00", "us")


def at(day, hour=20):
    return DAY + np.timedelta64(day, "D") + np.timedelta64(hour, "h")


def build(target, prediction, sessions, markets=None, **options):
    target = np.asarray(target, dtype=np.float64)
    markets = np.array(["US"] * len(target)) if markets is None else np.asarray(markets)
    hours = np.where(markets == "US", 20, 7)
    times = np.array([at(day, hour) for day, hour in zip(sessions, hours, strict=True)])
    return ForecastPanel.from_columns(
        np.array([f"r{i:04d}" for i in range(len(target))]),
        markets,
        times,
        target,
        np.asarray(prediction, dtype=np.float64),
        **options,
    )


def random_panel(rng, rows=600, days=12, quantiles=True):
    sessions = rng.integers(0, days, rows)
    markets = np.where(rng.random(rows) < 0.4, "CN", "US")
    target = np.round(rng.normal(size=rows), 1)
    prediction = np.round(0.4 * target + rng.normal(size=rows), 1)
    options = {}
    if quantiles:
        width = rng.uniform(0.1, 2.0, rows)
        offsets = np.array([-1.96, -1.28, 0.0, 1.28, 1.96])
        options = dict(quantiles=prediction[:, None] + width[:, None] * offsets, levels=LEVELS)
    return build(target, prediction, sessions, markets, **options)


def test_session_mean_gives_each_session_the_same_weight_by_hand():
    # Errores absolutos 1 y 3 en una sesión y 6 en otra.
    result = score_sessions(build([0, 0, 0], [1, -3, 6], [0, 0, 1])).summary()
    assert result["point"]["mae"] == 4.0
    assert result["row_weighted"]["mae"] == pytest.approx(10 / 3)
    assert result["point"]["mse"] == (5 + 36) / 2
    assert result["point"]["rmse"] == math.sqrt(20.5)
    assert result["row_weighted"]["mse"] == pytest.approx(46 / 3)
    assert result["rows"] == 3 and result["sessions"] == 2 and result["periods"] == 2


def test_market_weighting_is_declared_and_changes_only_the_aggregation():
    # US tiene dos sesiones (MAE 1 y 3) y CN una (MAE 8).
    panel = build([0, 0, 0], [1, 3, 8], [0, 1, 0], ["US", "US", "CN"])
    scores = score_sessions(panel)
    by_session = scores.summary(market_weighting="session")
    by_market = scores.summary(market_weighting="market")
    assert by_session["point"]["mae"] == 4.0 and by_market["point"]["mae"] == 5.0
    assert by_session["by_market"]["US"]["mae"] == by_market["by_market"]["US"]["mae"] == 2.0
    assert by_session["by_market"]["CN"]["mae"] == 8.0
    assert by_market["market_weighting"] == "market"


def test_session_errors_accumulator_and_panel_scores_agree():
    rng = np.random.default_rng(11)
    panel = random_panel(rng, quantiles=False)
    accumulator = SessionErrors()
    for start in range(0, panel.rows, 97):
        part = slice(start, start + 97)
        accumulator.update(
            np.asarray(panel.markets)[panel.market[part]],
            panel.prediction_at[part],
            panel.prediction[part] - panel.target[part],
        )
    expected = accumulator.summary()
    result = score_sessions(panel).summary()
    assert result["sessions"] == expected["session_count"]
    assert result["point"]["mae"] == pytest.approx(expected["session_mae"], rel=1e-12)
    assert result["point"]["mse"] == pytest.approx(expected["session_mse"], rel=1e-12)
    for market in ("US", "CN"):
        reference = expected["by_market_session"][market]
        assert result["by_market"][market]["mae"] == pytest.approx(reference["mae"], rel=1e-12)


def test_row_order_and_asset_order_inside_sessions_do_not_change_any_output():
    rng = np.random.default_rng(3)
    panel = random_panel(rng)
    order = rng.permutation(panel.rows)
    shuffled = ForecastPanel.from_columns(
        panel.row_id.take(order),
        np.asarray(panel.markets)[panel.market][order],
        panel.prediction_at[order],
        panel.target[order],
        panel.prediction[order],
        quantiles=panel.quantiles[order],
        levels=LEVELS,
    )
    assert score_sessions(panel).summary() == score_sessions(shuffled).summary()
    assert selective_risk(panel).summary() == selective_risk(shuffled).summary()


def test_perfect_prediction_has_zero_error_and_unit_rank_and_direction():
    target = [0.3, -0.1, 0.2, 0.5, -0.4]
    result = score_sessions(build(target, target, [0] * 5)).summary()["point"]
    assert result["mae"] == result["mse"] == 0
    assert result["rank_ic"] == 1.0 and result["direction_accuracy"] == 1.0


def test_reversed_ranks_give_minus_one():
    result = score_sessions(build([1, 2, 3, 4], [9, 7, 5, 3], [0] * 4)).summary()
    assert result["point"]["rank_ic"] == -1.0


def test_rank_ic_matches_scipy_average_ranks_with_ties_inside_each_session():
    rng = np.random.default_rng(5)
    panel = random_panel(rng, rows=900, days=6, quantiles=False)
    scores = score_sessions(panel)
    for session, start in enumerate(panel.session_starts):
        rows = slice(start, start + panel.session_samples[session])
        expected = spearmanr(panel.target[rows], panel.prediction[rows]).statistic
        assert scores.rank_ic[session] == pytest.approx(expected, abs=1e-13)


def test_undefined_rank_ic_is_counted_by_cause_and_never_averaged_as_zero():
    target = [1, 2, 0.5, 0.5, 0.5, 1, 2, 3, 1, 2, 3]
    prediction = [1, 2, 1, 2, 3, 4, 4, 4, 1, 2, 3]
    sessions = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    scores = score_sessions(build(target, prediction, sessions))
    assert scores.rank_ic_status.tolist() == [1, 2, 3, 0]
    summary = scores.summary()
    assert summary["rank_ic"]["sessions"] == dict(
        defined=1, insufficient_assets=1, constant_target=1, constant_prediction=1
    )
    assert summary["point"]["rank_ic"] == 1.0 and summary["point"]["rank_ic_sessions"] == 1
    table = scores.to_table()
    assert table["rank_ic"].null_count == 3
    assert table["rank_ic_status"].to_pylist()[:3] == [
        "insufficient_assets",
        "constant_target",
        "constant_prediction",
    ]
    stricter = score_sessions(build(target, prediction, sessions), rank_ic_min_assets=4)
    assert stricter.summary()["point"]["rank_ic"] is None
    # Los valores no definidos se exportan como null, nunca como NaN.
    json.dumps(stricter.summary(), allow_nan=False)
    with pytest.raises(ValueError, match="al menos 3"):
        score_sessions(build(target, prediction, sessions), rank_ic_min_assets=2)


def test_direction_excludes_zero_targets_and_counts_zero_predictions_as_misses():
    # Sesión 0: acierto, fallo, objetivo cero excluido y abstención. Sesión 1: dos aciertos.
    target = [2, -1, 0, 3, -1, 1]
    prediction = [1, 1, 5, 0, -2, 4]
    summary = score_sessions(build(target, prediction, [0, 0, 0, 0, 1, 1])).summary()
    assert summary["point"]["direction_accuracy"] == pytest.approx((1 / 3 + 1) / 2)
    assert summary["point"]["conditional_direction_accuracy"] == pytest.approx((1 / 2 + 1) / 2)
    direction = summary["direction"]
    assert direction["eligible_rows"] == 5 and direction["calls"] == 4 and direction["hits"] == 3
    assert direction["zero_target_rows"] == 1 and direction["zero_prediction_rows"] == 1
    assert direction["call_coverage"] == 4 / 5 and direction["undefined_sessions"] == 0
    only_zero = score_sessions(build([0, 0], [1, -1], [0, 0])).summary()
    assert only_zero["point"]["direction_accuracy"] is None
    assert only_zero["direction"]["undefined_sessions"] == 1


def quantile_panel(target, rows):
    return build(
        target,
        [row[2] for row in rows],
        [0] * len(target),
        quantiles=np.asarray(rows, dtype=np.float64),
        levels=LEVELS,
    )


def test_pinball_coverage_width_and_level_frequency_by_hand():
    rows = [[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]]
    scores = score_sessions(quantile_panel([1.0, -3.0], rows))
    quantiles = scores.summary()["quantiles"]
    # tau = 0,1: y = 1 sobre q = -1 cuesta 0,1·2. y = -3 bajo q = -1 cuesta 0,9·2.
    assert quantiles["pinball"]["0.1"] == pytest.approx((0.2 + 1.8) / 2)
    # tau = 0,5: la mitad del error absoluto de la mediana.
    assert quantiles["pinball"]["0.5"] == pytest.approx(0.5 * (1 + 3) / 2)
    # tau = 0,975 con q = 2: ambos objetivos quedan por debajo y cuestan 0,025·|y − q|.
    assert quantiles["pinball"]["0.975"] == pytest.approx((0.025 * 1 + 0.025 * 5) / 2)
    assert quantiles["level_frequency"]["0.1"] == 0.5
    assert quantiles["level_frequency"]["0.975"] == 1.0
    assert quantiles["level_frequency_gap"]["0.975"] == pytest.approx(0.025)
    narrow, wide = quantiles["intervals"]
    assert (narrow["nominal"], narrow["lower_level"], narrow["upper_level"]) == (0.8, 0.1, 0.9)
    assert narrow["coverage"] == 0.5 and narrow["width"] == 2.0
    assert narrow["coverage_gap"] == pytest.approx(-0.3)
    assert wide["coverage"] == 0.5 and wide["width"] == 4.0
    assert quantiles["point_equals_median"] is True


def test_interval_bounds_are_inclusive_and_full_or_empty_coverage_is_exact():
    rows = [[-2, -1, 0, 1, 2]] * 3
    covered = score_sessions(quantile_panel([-1.0, 0.0, 1.0], rows)).summary()["quantiles"]
    assert covered["intervals"][0]["coverage"] == 1.0
    outside = score_sessions(quantile_panel([-5.0, 5.0, 3.0], rows)).summary()["quantiles"]
    assert outside["intervals"][0]["coverage"] == outside["intervals"][1]["coverage"] == 0.0


def test_committed_sign_errors_count_confident_intervals_that_miss_the_sign():
    rows = [
        [0.1, 0.2, 0.3, 0.4, 0.5],  # afirma positivo y falla
        [-0.5, -0.4, -0.3, -0.2, -0.1],  # afirma negativo y acierta
        [-0.2, -0.1, 0.0, 0.1, 0.2],  # no afirma signo
        [0.1, 0.2, 0.3, 0.4, 0.5],  # afirma positivo con objetivo cero: no se juzga
    ]
    quantiles = score_sessions(quantile_panel([-0.2, -1.0, 0.5, 0.0], rows)).summary()["quantiles"]
    narrow = quantiles["intervals"][0]
    assert narrow["sign_commitment"] == 3 / 4
    assert narrow["committed_rows"] == 3 and narrow["committed_wrong_rows"] == 1
    assert narrow["committed_sign_error"] == 1 / 2


def test_point_models_report_absent_quantile_metrics_with_a_reason():
    scores = score_sessions(build([1, 2, 3], [1, 2, 2], [0, 0, 0]))
    summary = scores.summary()
    assert summary["quantiles"] is None and summary["quantiles_reason"]
    with pytest.raises(ValueError, match="no emite cuantiles"):
        scores.series("pinball")
    with pytest.raises(ValueError, match="no emite cuantiles"):
        selective_risk(build([1, 2, 3], [1, 2, 2], [0, 0, 0]))
    shifted = score_sessions(quantile_panel([1.0], [[-2, -1, 0.5, 1, 2]]))
    assert shifted.summary()["quantiles"]["point_equals_median"] is True
    no_median = build([1.0], [0.0], [0], quantiles=np.array([[-1.0, 1.0]]), levels=(0.1, 0.9))
    assert score_sessions(no_median).summary()["quantiles"]["point_equals_median"] is None


def test_within_session_order_matches_lexsort_including_ties():
    rng = np.random.default_rng(9)
    panel = random_panel(rng, rows=1200, days=20, quantiles=False)
    values = np.round(rng.normal(size=panel.rows), 1)
    order, position = forecast_scores._within_session_order(panel, values)
    assert np.array_equal(order, np.lexsort((values, panel.session)))
    assert np.array_equal(position, np.arange(panel.rows) - panel.session_starts[panel.session])


def test_selective_risk_keeps_the_narrowest_intervals_within_each_session_by_hand():
    widths = np.array([4.0, 1.0, 3.0, 2.0])
    errors = np.array([10.0, 1.0, 3.0, 2.0])
    offsets = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    panel = build(errors, np.zeros(4), [0] * 4, quantiles=widths[:, None] * offsets, levels=LEVELS)
    result = selective_risk(panel, coverages=(0.25, 0.5, 1.0)).summary()
    assert [point["mae"] for point in result["curve"]] == [1.0, 1.5, 4.0]
    assert [point["oracle_mae"] for point in result["curve"]] == [1.0, 1.5, 4.0]
    assert [point["retained_rows"] for point in result["curve"]] == [1, 2, 4]
    assert result["full_mae"] == 4.0 and result["score"] == "central_interval_width_0.8"
    reversed_errors = build(
        errors[::-1].copy(),
        np.zeros(4),
        [0] * 4,
        quantiles=widths[:, None] * offsets,
        levels=LEVELS,
    )
    # Las dos anchuras menores corresponden ahora a los errores 3 y 10.
    worse = selective_risk(reversed_errors, coverages=(0.5,)).summary()["curve"][0]
    assert worse["mae"] == (3.0 + 10.0) / 2 and worse["oracle_mae"] == 1.5


def test_selective_counts_use_exact_ceilings_and_canonical_tie_breaks():
    panel = build(
        np.arange(10.0), np.zeros(10), [0] * 10, quantiles=np.zeros((10, 5)), levels=LEVELS
    )
    result = selective_risk(panel, coverages=(0.7, 0.71))
    assert result.retained[0].tolist() == [7, 8]
    # Con anchuras iguales se conservan las primeras filas canónicas (r0000 a r0006).
    assert result.mae[0, 0] == np.mean(np.arange(7.0))
    series = result.series(0.7)
    assert series.metric == "selective_mae@0.7" and series.loss
    with pytest.raises(ValueError, match="no calculada"):
        result.series(0.5)
    for coverages in ((0.5, 0.5), (0.0, 1.0), (1.2,), ()):
        with pytest.raises(ValueError, match="crecientes"):
            selective_risk(panel, coverages=coverages)
    with pytest.raises(ValueError, match="intervalo central"):
        selective_risk(panel, nominal=0.5)


def test_series_carry_masks_and_loss_semantics():
    scores = score_sessions(build([1, 2, 0.5, 0.5, 0.5], [1, 2, 1, 2, 3], [0, 0, 1, 1, 1]))
    rank = scores.series("rank_ic")
    assert not rank.loss and rank.defined.tolist() == [False, False]
    assert rank.values.tolist() == [0.0, 0.0]
    mae = scores.series("mae")
    assert mae.loss and mae.metric == "mae" and mae.defined.all()
    with pytest.raises(ValueError, match="no admitida"):
        scores.series("rmse")
