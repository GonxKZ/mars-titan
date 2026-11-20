"""Métricas de tablas guardadas, sin inferencia ni acceso a pesos."""

from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

from mars_titan.evaluation import comparison_metrics as metrics
from mars_titan.evaluation.forecast_panel import ForecastPanel, SessionSeries
from tests.evaluation.comparison_results_fixtures import prediction_table


def panel(tmp_path, partition="evaluation", scale=1):
    import pyarrow.parquet as pq

    path = tmp_path / f"{partition}.parquet"
    prediction_table(path, ["A", "B", "C"], 2019, partition, scale)
    return metrics.panel_from_table(pq.read_table(path))


def identified_rows():
    return pa.table(
        dict(
            sample_id=["US/A/1", "US/B/1", "US/A/2"],
            asset_id=["US/A", "US/B", "US/A"],
            market=["US"] * 3,
            prediction_at=pa.array([1, 1, 2], type=pa.timestamp("us", tz="UTC")),
            target=[0.01, -0.03, 0.04],
            prediction=[0.0] * 3,
        )
    )


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("asset_id", ["US/A", "US/A", "US/A"]),
        ("asset_id", ["US/B", "US/A", "US/A"]),
        ("asset_id", ["US/A", None, "US/A"]),
        ("sample_id", ["US/A/1", "US/B/1", "US/A/3"]),
    ],
)
def test_samples_keep_their_asset_and_decision_time(name, values):
    table = identified_rows()
    changed = table.set_column(table.schema.get_field_index(name), name, pa.array(values))
    with pytest.raises(ValueError, match="activo"):
        metrics.panel_from_table(changed)


def test_legacy_rows_recover_the_asset_only_from_an_unambiguous_key():
    table = identified_rows().drop(["asset_id"])
    index = table.schema.get_field_index("sample_id")
    for invalid in ("opaca", "US//1", "US/A/01", "CN/A/1", "US/A/B/1", "US/A/3"):
        changed = table.set_column(index, "sample_id", pa.array([invalid, "US/B/1", "US/A/2"]))
        with pytest.raises(ValueError, match="activo"):
            metrics.panel_from_table(changed)


@pytest.mark.parametrize(
    "invalid", ["CN/A", "A", "US/", "US/A/B", "US/.", "US/..", "US/" + "A" * 65]
)
@pytest.mark.parametrize("explicit", [True, False])
def test_an_invalid_flow_is_rejected_even_if_the_sample_id_repeats_it(invalid, explicit):
    table = identified_rows()
    table = table.set_column(0, "sample_id", pa.array([f"{invalid}/1", "US/B/1", "US/A/2"]))
    table = table.set_column(1, "asset_id", pa.array([invalid, "US/B", "US/A"]))
    if not explicit:
        table = table.drop(["asset_id"])
    with pytest.raises(ValueError, match="activo"):
        metrics.panel_from_table(table)


def test_reordered_chunks_and_legacy_rows_keep_the_existing_population_digest():
    table = identified_rows()
    previous = ForecastPanel.from_arrow(table, markets=("US",)).cohort_sha256
    reordered = pa.concat_tables([table.slice(2), table.slice(0, 2)])
    for stored in (table, reordered, reordered.drop(["asset_id"])):
        panel = metrics.panel_from_table(stored)
        assert panel.rows == 3 and panel.cohort_sha256 == previous


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
