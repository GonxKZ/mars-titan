"""Contrastes por sesión, control de la reserva y remuestreo pareado."""

import hashlib
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.evaluation.prediction_statistics import (
    block_intervals,
    initial_policy,
    resampled_means,
    review_predictions,
)


def panel(**changes):
    values = dict(
        sample_id=["b", "a", "c"],
        asset_id=["US/B", "US/A", "US/A"],
        market=["US", "US", "US"],
        prediction_at=[datetime(2023, 1, day, tzinfo=UTC) for day in (3, 3, 4)],
        target=[1.0, 3.0, 6.0],
        prediction=[0.0, 0.0, 0.0],
        parent=[1.0, 3.0, 0.0],
        zero=[0.0, 0.0, 0.0],
        center=[1.0, 3.0, 6.0],
    )
    values.update(changes)
    values["prediction_at"] = pa.array(values["prediction_at"], pa.timestamp("us", tz="UTC"))
    return pa.table(values)


def source(tmp_path, table, **changes):
    path = tmp_path / "evaluation.parquet"
    pq.write_table(table, path)
    values = dict(
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        partition="evaluation",
        bounds=["2023-01-01", "2023-02-01"],
        declared_metrics=None,
        metadata={},
    )
    values.update(changes)
    return values


def test_sessions_receive_equal_weight_with_unbalanced_asset_counts(tmp_path):
    result = review_predictions(source(tmp_path, panel()))
    score = result["metrics"]["prediction"]
    assert score["samples"] == 3
    assert score["session_count"] == 2
    assert score["mae"] == pytest.approx(10 / 3)
    assert score["session_mae"] == 4
    assert score["session_mse"] == 20.5
    assert result["metrics"]["parent"]["session_mae"] == 3
    assert result["metrics"]["center"]["session_mae"] == 0
    assert result["sessions"]["mae_prediction"].to_pylist() == [2, 6]
    assert result["diagnostics"]["rank_ic_valid_sessions"] == 0
    assert result["diagnostics"]["mean_session_rank_ic"] is None


def test_initial_cache_reuses_only_identical_inputs_and_respects_its_byte_limit(monkeypatch):
    from mars_titan.evaluation import prediction_statistics as module

    calls = []
    original = module.initial_policy

    def record(parent, grid):
        calls.append(len(parent))
        return original(parent, grid)

    monkeypatch.setattr(module, "initial_policy", record)
    grid = dict(
        schema_version=1,
        values=np.linspace(-1, 1, 21).tolist(),
        scale=0.2,
        source_sha256="a" * 64,
        training_samples=10,
    )
    parent = np.array([0.1, 0.2])
    cache = module.InitialPolicyCache(max_bytes=parent.nbytes)
    first = cache.get(parent, grid)
    np.testing.assert_array_equal(first, original(parent, grid))
    assert cache.get(parent.copy(), dict(grid)) is first
    assert len(calls) == 1
    assert not first.flags.writeable
    cache.get(parent + 0.1, grid)
    cache.get(parent, grid)
    assert len(calls) == 3
    assert cache.bytes_used <= parent.nbytes
    cache.get(parent, grid | {"scale": 0.3})
    assert len(calls) == 4
    disabled = module.InitialPolicyCache(max_bytes=0)
    disabled.get(parent, grid)
    disabled.get(parent, grid)
    assert len(calls) == 6 and disabled.bytes_used == 0


def test_market_sessions_are_not_merged_by_timestamp(tmp_path):
    table = panel(
        market=["US", "US", "CN"],
        asset_id=["US/B", "US/A", "CN/A"],
        prediction_at=[datetime(2023, 1, 3, tzinfo=UTC)] * 3,
    )
    assert review_predictions(source(tmp_path, table))["metrics"]["prediction"]["session_mae"] == 4


def test_asset_permutation_preserves_pairing_and_all_metrics(tmp_path):
    first = review_predictions(source(tmp_path, panel()))
    permuted = panel().take(pa.array([2, 0, 1]))
    second = review_predictions(source(tmp_path, permuted), cohort=first["cohort"])
    assert second["metrics"] == first["metrics"]
    assert second["sessions"].equals(first["sessions"])
    np.testing.assert_array_equal(second["parent"], first["parent"])


@pytest.mark.parametrize(
    "column,values", [("sample_id", ["b", "b", "c"]), ("asset_id", ["US/A"] * 3)]
)
def test_duplicate_sample_or_asset_session_is_rejected(tmp_path, column, values):
    with pytest.raises(ValueError):
        review_predictions(source(tmp_path, panel(**{column: values})))


def test_target_or_population_changes_invalidate_a_pair(tmp_path):
    original = review_predictions(source(tmp_path, panel()))["cohort"]
    for changes in ({"target": [1.0, 3.0, 7.0]}, {"asset_id": ["US/C", "US/A", "US/A"]}):
        with pytest.raises(ValueError):
            review_predictions(source(tmp_path, panel(**changes)), cohort=original)


@pytest.mark.parametrize("changes", [{"prediction": [0.0, np.inf, 0.0]}, {"zero": [0.0, 1.0, 0.0]}])
def test_nonfinite_predictions_and_nonzero_zero_control_are_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        review_predictions(source(tmp_path, panel(**changes)))


def test_wrong_hash_and_declared_metric_are_rejected(tmp_path):
    data = source(tmp_path, panel())
    with pytest.raises(ValueError):
        review_predictions(data | {"sha256": "0" * 64})
    declared = review_predictions(data)["metrics"]
    declared["prediction"]["session_mae"] = 0
    with pytest.raises(ValueError):
        review_predictions(data | {"declared_metrics": declared})


def test_reserved_dates_are_rejected_before_loading_targets(tmp_path, monkeypatch):
    data = source(tmp_path, panel(prediction_at=[datetime(2024, 1, 3, tzinfo=UTC)] * 3))
    original = pq.ParquetFile

    class ObservedFile:
        def __init__(self, *args, **kwargs):
            self.file = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.file, name)

        def read(self, columns, **kwargs):
            assert "target" not in columns, "Se intentó leer una etiqueta reservada"
            return self.file.read(columns=columns, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", ObservedFile)
    with pytest.raises(ValueError):
        review_predictions(data)


def test_rank_ic_is_computed_per_session_and_constant_cases_remain_undefined(tmp_path):
    dates = [datetime(2023, 1, day, tzinfo=UTC) for day in (3, 3, 3, 4, 4, 4)]
    table = pa.table(
        dict(
            sample_id=list("abcdef"),
            asset_id=["US/A", "US/B", "US/C"] * 2,
            market=["US"] * 6,
            prediction_at=pa.array(dates, pa.timestamp("us", tz="UTC")),
            target=[1.0, 2.0, 3.0] * 2,
            prediction=[1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            parent=[0.0] * 6,
            zero=[0.0] * 6,
        )
    )
    result = review_predictions(source(tmp_path, table))
    assert result["sessions"]["rank_ic"].to_pylist() == pytest.approx([1, -1])
    assert result["diagnostics"]["rank_ic_valid_sessions"] == 2
    assert result["diagnostics"]["mean_session_rank_ic"] == pytest.approx(0)


def test_resampling_uses_shared_blocks_and_truncates_the_last_block():
    effects = np.array([[1, 10], [3, 30], [5, 50], [7, 70], [9, 90]], dtype=np.float64)
    starts = np.array([[0, 3, 1], [2, 2, 0]], dtype=np.int64)
    np.testing.assert_allclose(
        resampled_means(effects, starts, 2), [[4.6, 46], [5, 50]], atol=1e-14
    )


def test_constant_effect_and_seed_recovery_have_the_expected_intervals():
    values = np.tile([0.1, -0.2], (17, 1))
    result = block_intervals(values, block_length=5, repetitions=200, seed=42)
    np.testing.assert_allclose(result["estimate"], [0.1, -0.2])
    np.testing.assert_allclose(result["lower"], [0.1, -0.2])
    np.testing.assert_allclose(result["upper"], [0.1, -0.2])
    assert result == block_intervals(values, block_length=5, repetitions=200, seed=42)


def test_insufficient_sessions_do_not_create_a_zero_width_interval():
    result = block_intervals(np.ones((3, 2)), block_length=3, repetitions=200, seed=42)
    assert result["lower"] is None and result["upper"] is None
    assert result["reason"]


@pytest.mark.parametrize(
    "options", [{"block_length": 0}, {"repetitions": True}, {"repetitions": 10001}, {"seed": -1}]
)
def test_bootstrap_rejects_invalid_and_unbounded_work(options):
    with pytest.raises(ValueError):
        block_intervals(np.ones((7, 2)), **options)


def test_initial_policy_uses_the_frozen_grid_and_saturates_at_its_endpoints():
    grid = dict(
        schema_version=1,
        values=np.linspace(-1, 1, 21).tolist(),
        scale=0.1,
        source_sha256="a" * 64,
        training_samples=10,
    )
    np.testing.assert_array_equal(initial_policy(np.array([-100.0, 0.0, 100.0]), grid), [-1, 0, 1])


def test_complex_effects_are_not_silently_projected_to_real_values():
    values = np.ones((7, 2), dtype=np.complex128) * (1 + 1j)
    with pytest.raises(ValueError):
        block_intervals(values)
    with pytest.raises(ValueError):
        resampled_means(values, np.zeros((2, 2), dtype=np.int64), 4)


def test_overflow_cannot_leave_an_infinite_estimate_in_the_result():
    with pytest.raises(ValueError):
        block_intervals(np.full((3, 2), 1e308), block_length=3)


def test_resampling_does_not_modify_the_callers_random_state():
    state = np.random.get_state()
    block_intervals(np.arange(21, dtype=np.float64).reshape(7, 3), repetitions=31, seed=7)
    after = np.random.get_state()
    assert after[0] == state[0]
    np.testing.assert_array_equal(after[1], state[1])
    assert after[2:] == state[2:]
