"""Métricas de tablas guardadas, sin inferencia ni acceso a pesos."""

from dataclasses import replace

import numpy as np
import pytest

from mars_titan.evaluation import comparison_metrics as metrics
from mars_titan.evaluation.forecast_panel import ForecastPanel, SessionSeries
from tests.evaluation.comparison_results_fixtures import prediction_table


def panel(tmp_path, partition="evaluation", scale=1):
    import pyarrow.parquet as pq

    path = tmp_path / f"{partition}.parquet"
    prediction_table(path, ["A", "B", "C"], 2019, partition, scale)
    return metrics.panel_from_table(pq.read_table(path))


def test_saved_quantiles_produce_point_interval_and_probability_metrics(tmp_path):
    source = panel(tmp_path)
    measured = metrics.measure(source, magnitude=True)

    assert measured.summary["point"]["mae"] == pytest.approx(
        np.abs(source.prediction - source.target).mean()
    )
    assert measured.summary["point"]["rmse"] >= measured.summary["point"]["mae"]
    assert measured.summary["quantiles"]["levels"] == [0.025, 0.1, 0.5, 0.9, 0.975]
    assert len(measured.summary["quantiles"]["intervals"]) == 2
    assert measured.summary["quantiles"]["sign_probability"]["brier"] is not None
    assert "pearson_ic" in measured.sessions.column_names
    assert measured.series["pearson_ic"].defined.all()


def test_rank_only_controls_do_not_report_a_comparable_raw_magnitude(tmp_path):
    measured = metrics.measure(panel(tmp_path), magnitude=False)

    assert measured.summary["point"]["mae"] is None
    assert measured.summary["magnitude_reason"]
    assert measured.summary["point"]["rank_ic"] is not None
    assert "mae" not in measured.sessions.column_names
    assert "mae_raw" not in measured.series


def test_calibration_is_fitted_before_evaluation_and_does_not_recalibrate_quantiles(tmp_path):
    calibration = panel(tmp_path, "calibration")
    evaluation = panel(tmp_path)

    factors = metrics.fit_scale(calibration)
    calibrated = metrics.calibrated_panel(evaluation, factors)
    measured = metrics.measure(calibrated, magnitude=True)

    assert factors == {"US": pytest.approx(0.5)}
    assert measured.summary["point"]["mae"] > 0
    assert measured.summary["point"]["mae"] == pytest.approx(np.abs(evaluation.target).mean())
    assert calibrated.levels is None and calibrated.quantiles is None
    np.testing.assert_array_equal(evaluation.prediction, panel(tmp_path).prediction)


def test_seed_average_is_an_average_of_errors_and_not_an_ensemble():
    first = ForecastPanel.from_columns(
        ["a", "b", "c"], ["US"] * 3, np.array([1, 1, 1]), np.zeros(3), np.ones(3), markets=("US",)
    )
    second = replace(first, prediction=-first.prediction)
    one = metrics.measure(first, magnitude=True).series["mae_raw"]
    two = metrics.measure(second, magnitude=True).series["mae_raw"]

    average = SessionSeries.average([one, two])

    assert average.values.tolist() == [1.0]
    assert np.mean([first.prediction, second.prediction], axis=0).tolist() == [0.0] * 3


def test_constant_predictions_keep_undefined_correlations_explicit(tmp_path):
    source = panel(tmp_path)
    zero = replace(source, prediction=np.zeros(source.rows), quantiles=None, levels=None)
    measured = metrics.measure(zero, magnitude=True)

    assert measured.summary["pearson_ic"]["mean"] is None
    assert measured.summary["point"]["rank_ic"] is None
    assert measured.series["pearson_ic"].defined.sum() == 0
    assert measured.summary["direction"]["call_coverage"] == 0


def test_prediction_rows_and_dates_must_match_the_declared_partition(tmp_path):
    source = panel(tmp_path)
    cohort = dict(
        selection=dict(counts=dict(evaluation=source.rows)),
        fold=dict(evaluation=["2019-01-01", "2020-01-01"]),
    )
    metrics.check_partition(source, cohort, "evaluation")
    cohort["selection"]["counts"]["evaluation"] -= 1
    with pytest.raises(ValueError, match="filas"):
        metrics.check_partition(source, cohort, "evaluation")
    cohort["selection"]["counts"]["evaluation"] += 1
    cohort["fold"]["evaluation"] = ["2020-01-01", "2021-01-01"]
    with pytest.raises(ValueError, match="temporal"):
        metrics.check_partition(source, cohort, "evaluation")


def test_malformed_quantile_levels_are_not_silently_ignored(tmp_path):
    import pyarrow.parquet as pq

    path = tmp_path / "bad.parquet"
    prediction_table(path, ["A", "B", "C"], 2019, "evaluation", 1)
    table = pq.read_table(path)
    names = ["quantile_bad" if n == "quantile_0025" else n for n in table.column_names]
    with pytest.raises(ValueError, match="cuantil"):
        metrics.panel_from_table(table.rename_columns(names))
