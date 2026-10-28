"""Lectura de fixtures Parquet por sesiones completas, sin modelos ni aprendizaje."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.forecast_scores import score_sessions


@pytest.fixture
def forecast_file(tmp_path):
    rows = []
    for day in range(67):
        for market, size in (("US", 4), ("CN", 1 if day % 2 else 3)):
            for asset in range(size):
                target = 0.02 * asset - 0.01 + day * 0.0001
                prediction = target + (0.04 if market == "US" else 0.01)
                rows.append(
                    dict(
                        sample_id=f"{market}/{day}/{asset}",
                        market=market,
                        prediction_at=1_600_000_000_000_000 + day * 86_400_000_000,
                        target=target,
                        prediction=prediction,
                        intact=prediction,
                        q_low=prediction - 0.1,
                        q_mid=prediction,
                        q_high=prediction + 0.1,
                    )
                )
    table = pa.Table.from_pylist(rows)
    table = table.set_column(
        2, "prediction_at", table["prediction_at"].cast(pa.timestamp("us", "UTC"))
    )
    table = table.take(np.random.Generator(np.random.PCG64(5)).permutation(len(table)))
    path = tmp_path / "predictions.parquet"
    pq.write_table(table, path, row_group_size=7)
    return path, table


def read(forecast_file, tmp_path, **kwargs):
    from mars_titan.evaluation.forecast_files import score_prediction_file

    path, table = forecast_file
    options = dict(rows=len(table), minimum=3, temporary=tmp_path, batch_size=7)
    options.update(kwargs)
    return score_prediction_file(path, {"model": "prediction", "zero": None}, **options)


@pytest.mark.parametrize("batch_size", [1, 7, 4096])
def test_complete_sessions_match_full_panel_across_arrow_batches(
    forecast_file, tmp_path, batch_size
):
    scores, resources = read(forecast_file, tmp_path, batch_size=batch_size)
    full = score_sessions(ForecastPanel.from_arrow(forecast_file[1])).summary()
    measured = scores["model"].summary()
    full.pop("cohort_sha256")
    measured.pop("cohort_sha256")
    assert measured == full
    assert measured["point"]["mae"] == pytest.approx(0.025)
    assert resources["sessions"] == 134
    assert resources["largest_session_rows"] == 4
    assert scores["model"].cohort_sha256 == scores["zero"].cohort_sha256
    assert scores["zero"].summary()["point"]["rank_ic"] is None


def test_partitioning_does_not_change_the_population_identity(forecast_file, tmp_path):
    first, _ = read(forecast_file, tmp_path, batch_size=1)
    second, _ = read(forecast_file, tmp_path, batch_size=64)
    assert first["model"].cohort_sha256 == second["model"].cohort_sha256
    np.testing.assert_array_equal(first["model"].mae, second["model"].mae)


def test_existing_quantiles_are_scored_without_adding_them_to_zero(forecast_file, tmp_path):
    scores, _ = read(
        forecast_file,
        tmp_path,
        quantiles={"model": (("q_low", "q_mid", "q_high"), (0.1, 0.5, 0.9))},
    )
    expected = score_sessions(
        ForecastPanel.from_arrow(
            forecast_file[1],
            quantile_columns=("q_low", "q_mid", "q_high"),
            levels=(0.1, 0.5, 0.9),
        )
    )
    np.testing.assert_array_equal(scores["model"].pinball, expected.pinball)
    assert scores["model"].levels == (0.1, 0.5, 0.9)
    assert scores["zero"].levels is None


@pytest.mark.parametrize("problem", ["future", "duplicate", "nan", "null", "crossed", "inf_mse"])
def test_invalid_rows_are_rejected_after_sorting(forecast_file, tmp_path, problem):
    path, table = forecast_file
    options = {}
    if problem == "duplicate":
        table = pa.concat_tables([table, table.slice(0, 1)])
    else:
        column = {"future": "prediction_at", "crossed": "q_low"}.get(problem, "prediction")
        values = table[column].to_pylist()
        values[0] = {
            "future": pa.scalar(1_704_067_200_000_000, pa.timestamp("us", "UTC")).as_py(),
            "nan": float("nan"),
            "null": None,
            "crossed": 99.0,
            "inf_mse": 1e308,
        }[problem]
        table = table.set_column(
            table.column_names.index(column), column, pa.array(values, type=table[column].type)
        )
        if problem == "crossed":
            options["quantiles"] = {"model": (("q_low", "q_mid", "q_high"), (0.1, 0.5, 0.9))}
    pq.write_table(table, path, row_group_size=7)
    with pytest.raises(ValueError):
        read((path, table), tmp_path, **options)


@pytest.mark.parametrize("problem", ["rows", "sessions", "empty_points", "bad_column", "batch"])
def test_contract_limits_prevent_partial_scores(forecast_file, tmp_path, problem):
    from mars_titan.evaluation.forecast_files import score_prediction_file

    path, table = forecast_file
    points = {"model": "prediction"}
    options = dict(rows=len(table), minimum=3, temporary=tmp_path, batch_size=7)
    if problem == "rows":
        options["rows"] += 1
    elif problem == "sessions":
        options["max_sessions"] = 1
    elif problem == "empty_points":
        points = {}
    elif problem == "bad_column":
        points["model"] = 'prediction" FROM other'
    else:
        options["batch_size"] = 0
    with pytest.raises(ValueError):
        score_prediction_file(path, points, **options)


def test_equality_constraint_checks_emitted_points(forecast_file, tmp_path):
    read(forecast_file, tmp_path, equal_points=(("prediction", "intact"),))
    path, table = forecast_file
    values = table["intact"].to_pylist()
    values[0] = 99.0
    table = table.set_column(table.column_names.index("intact"), "intact", pa.array(values))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="emisi[oó]n|coincid"):
        read((path, table), tmp_path, equal_points=(("prediction", "intact"),))


def test_same_size_different_targets_change_the_population_fingerprint(forecast_file, tmp_path):
    first, _ = read(forecast_file, tmp_path)
    path, table = forecast_file
    values = table["target"].to_pylist()
    values[0] += 0.001
    changed = table.set_column(table.column_names.index("target"), "target", pa.array(values))
    other = tmp_path / "changed.parquet"
    pq.write_table(changed, other)
    second, _ = read((other, changed), tmp_path)
    assert first["model"].cohort_sha256 != second["model"].cohort_sha256


def test_incomplete_footer_does_not_enter_the_parquet_reader(tmp_path):
    from mars_titan.evaluation.forecast_files import score_prediction_file

    path = tmp_path / "truncated.parquet"
    path.write_bytes(b"PAR1")
    with pytest.raises(ValueError, match="Parquet"):
        score_prediction_file(
            path, {"model": "prediction"}, rows=1, minimum=3, temporary=Path(tmp_path)
        )


def test_quantile_width_overflow_is_rejected(forecast_file, tmp_path):
    path, table = forecast_file
    for column, value in (("q_low", -1e308), ("q_high", 1e308)):
        table = table.set_column(
            table.column_names.index(column), column, pa.array([value] * len(table))
        )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="finito|desborda"):
        read(
            (path, table),
            tmp_path,
            quantiles={"model": (("q_low", "q_mid", "q_high"), (0.1, 0.5, 0.9))},
        )
