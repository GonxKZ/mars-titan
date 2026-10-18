"""El resumen del microensayo respeta sesiones, identidades y semillas completas."""

import numpy as np
import pyarrow as pa
import pytest

from benchmarks.memory_microstudy_analysis import contrasts, score_table, seed_means
from benchmarks.memory_microstudy_state import Scalars, replay, same_predictions
from mars_titan.training.learning_hold import LearningHoldError


def table():
    return pa.table(
        dict(
            sample_id=["A0", "B0", "A1"],
            asset_id=["A", "B", "A"],
            market=["US"] * 3,
            prediction_at=pa.array([0, 0, 86_400_000_000], type=pa.timestamp("us", tz="UTC")),
            target=[1.0, 1.0, 10.0],
            prediction=[0.0, 0.0, 0.0],
        )
    )


def test_session_mean_is_not_the_global_row_mean():
    scores = score_table(table())
    assert scores.summary()["point"]["mae"] == 5.5
    assert scores.summary()["row_weighted"]["mae"] == 4.0


def test_pairing_accepts_permutations_but_rejects_changed_targets():
    first = score_table(table())
    second = score_table(table().take([2, 0, 1]), first.cohort_sha256)
    np.testing.assert_array_equal(first.mae, second.mae)
    changed = table().set_column(4, "target", pa.array([1.0, 2.0, 10.0]))
    with pytest.raises(ValueError, match="población"):
        score_table(changed, first.cohort_sha256)


def test_missing_seeds_are_not_silently_averaged_as_a_complete_comparison():
    series = score_table(table()).series("mae")
    assert seed_means({"pt1-42": series, "pt1-43": series}) == {}
    result = seed_means({f"pt1-{seed}": series for seed in (42, 43, 44)})
    np.testing.assert_array_equal(result["pt1-mean"].values, series.values)
    assert len(result["pt1-mean"].values) == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_state_statistics_reject_nonfinite_values(value):
    with pytest.raises(ValueError, match="finito"):
        Scalars().scalar("norm", value)


def test_state_statistics_cannot_silently_overwrite_a_metric():
    records = Scalars()
    records.scalar("norm", 1.0)
    with pytest.raises(ValueError, match="duplicada"):
        records.scalar("norm", 2.0)


def test_state_replay_respects_the_hold_before_opening_any_files(tmp_path, monkeypatch):
    hold = tmp_path / "hold.json"
    hold.write_text('{"training_allowed": false}')
    monkeypatch.setenv("MARS_TITAN_TRAINING_HOLD", str(hold))
    with pytest.raises(LearningHoldError):
        replay(tmp_path / "absent", "mac_online-42", tmp_path / "report.json")


def test_replay_alignment_compares_values_bitwise_after_ordering_rows():
    original = table()
    assert same_predictions(original, original.take([2, 0, 1]))
    perturbed = original.set_column(5, "prediction", pa.array([np.nextafter(0.0, 1.0), 0.0, 0.0]))
    assert not same_predictions(original, perturbed)


def test_adapter_effect_uses_the_frozen_runtime_control_not_a_different_gradient_mode():
    series = dict.fromkeys(
        ("zero", "mac_online-42", "frozen_parent-42", "post-head-42", "post-full_continuation-42")
    )
    result = contrasts(series)
    assert result["post-head-42_minus_frozen_parent-42"] == {
        "post-head-42": 1.0,
        "frozen_parent-42": -1.0,
    }
    assert result["post-full_continuation-42_minus_mac_online-42"] == {
        "post-full_continuation-42": 1.0,
        "mac_online-42": -1.0,
    }
