"""Ventanas temporales comunes, maduración y reserva final sin selección aleatoria."""

import importlib
from datetime import UTC, datetime

import numpy as np
import pytest

from mars_titan.data.temporal import MarketClock


def module():
    try:
        return importlib.import_module("mars_titan.evaluation.splits")
    except ModuleNotFoundError:
        pytest.fail("Falta el contrato de validación temporal")


def configuration():
    return dict(
        schema_version=1,
        market="US",
        train_start="2019-01-01",
        first_validation_start="2022-01-01",
        validation_months=6,
        calibration_months=3,
        evaluation_months=3,
        step_months=3,
        minimum_train_months=36,
        gap_sessions=1,
        final_test_start="2024-01-01",
        final_test_end="2025-01-01",
        primary_metric="session_mae",
        seeds=[42, 43, 44],
    )


def micros(*values):
    return np.array(
        [int(datetime.fromisoformat(v).replace(tzinfo=UTC).timestamp() * 1e6) for v in values],
        dtype=np.int64,
    )


def test_expanding_windows_have_disjoint_evaluations_and_reserve_test():
    folds = module().build_folds(configuration())
    assert len(folds) == 5
    assert folds[0]["train"] == ["2019-01-01", "2022-01-01"]
    assert folds[0]["validation"] == ["2022-01-01", "2022-07-01"]
    assert folds[0]["calibration"] == ["2022-07-01", "2022-10-01"]
    assert folds[0]["evaluation"] == ["2022-10-01", "2023-01-01"]
    assert folds[-1]["evaluation"] == ["2023-10-01", "2024-01-01"]
    assert all(
        a["evaluation"][1] == b["evaluation"][0] for a, b in zip(folds[:-1], folds[1:], strict=True)
    )
    assert all(f["train"][0] == "2019-01-01" for f in folds)


@pytest.mark.parametrize(
    "change",
    [
        dict(step_months=1),
        dict(gap_sessions=-1),
        dict(validation_months=0),
        dict(first_validation_start="2020-01-01"),
        dict(final_test_end="2023-01-01"),
        dict(first_validation_start="2022-01-02"),
        dict(seeds=[42, 42]),
        dict(primary_metric="test_mae"),
        dict(market="other"),
        dict(unknown=1),
    ],
)
def test_invalid_or_leaky_protocol_is_rejected(change):
    with pytest.raises(ValueError):
        module().build_folds(configuration() | change)


def test_labels_crossing_frontier_and_last_training_session_are_excluded():
    config = configuration()
    fold = module().build_folds(config)[0]
    clock = MarketClock("US", "2021-12-01", "2024-01-08")
    dates = ["2021-12-29", "2021-12-30", "2021-12-30", "2021-12-31", "2022-01-03"]
    prediction = np.array([int(clock.decision(day).timestamp() * 1e6) for day in dates])
    maturity = np.array(
        [
            int(clock.decision(day).timestamp() * 1e6)
            for day in ["2021-12-30", "2021-12-31", "2022-01-03", "2022-01-03", "2022-01-04"]
        ]
    )
    result = module().assign_partitions(
        prediction, prediction.copy(), maturity, fold, clock, config
    )
    assert result["partition"].tolist() == ["train", "train", "excluded", "excluded", "validation"]
    assert result["reason"][2] == "label_crosses_boundary"
    assert result["reason"][3] == "session_gap"


def test_future_inputs_missing_macro_and_test_rows_never_enter_development():
    config = configuration()
    clock = MarketClock("US", "2021-12-01", "2025-01-07")
    pred = np.array(
        [
            int(clock.decision(day).timestamp() * 1e6)
            for day in ["2022-01-04", "2022-01-05", "2024-01-02"]
        ]
    )
    available = pred.copy()
    available[0] += 1
    result = module().assign_partitions(
        pred,
        available,
        pred + 86400 * 1000000,
        module().build_folds(config)[0],
        clock,
        config,
        eligible=np.array([True, False, True]),
    )
    assert result["partition"].tolist() == ["excluded", "excluded", "test_reserved"]
    assert result["reason"].tolist() == [
        "future_inputs",
        "incomplete_inputs",
        "final_test_reserved",
    ]


def test_asset_permutation_does_not_change_assignments_and_same_session_uses_same_window():
    config = configuration()
    clock = MarketClock("US", "2021-12-01", "2024-01-08")
    pred = np.array(
        [
            int(clock.decision(day).timestamp() * 1e6)
            for day in ["2022-03-11", "2022-03-14", "2022-03-14", "2022-07-05", "2022-10-04"]
        ]
    )
    args = (module().build_folds(config)[0], clock, config)
    baseline = module().assign_partitions(pred, pred, pred + 86400 * 1000000, *args)
    order = np.array([4, 2, 0, 3, 1])
    permuted = module().assign_partitions(
        pred[order], pred[order], pred[order] + 86400 * 1000000, *args
    )
    assert permuted["partition"].tolist() == baseline["partition"][order].tolist()
    assert baseline["partition"].tolist() == ["validation"] * 3 + ["calibration", "evaluation"]


def test_inserting_future_rows_cannot_change_earlier_membership():
    config = configuration()
    clock = MarketClock("US", "2021-01-01", "2025-01-08")
    pred = micros("2021-06-01T20:05:00", "2022-06-01T20:05:00")
    args = (module().build_folds(config)[0], clock, config)
    before = module().assign_partitions(pred, pred, pred + 86400 * 1000000, *args)
    extended = np.concatenate([pred, micros("2024-06-03T20:05:00")])
    after = module().assign_partitions(extended, extended, extended + 86400 * 1000000, *args)
    assert before["partition"].tolist() == after["partition"][:2].tolist()


@pytest.mark.parametrize("defect", ["dimensions", "dtype", "maturity", "clock"])
def test_malformed_time_metadata_is_rejected(defect):
    config = configuration()
    clock = MarketClock("US", "2021-01-01", "2025-01-08")
    pred = micros("2021-06-01T20:05:00")
    available = pred.copy()
    maturity = pred + 86400 * 1000000
    if defect == "dimensions":
        available = np.array([], dtype=np.int64)
    if defect == "dtype":
        pred = pred.astype(float)
    if defect == "maturity":
        maturity = pred.copy()
    if defect == "clock":
        pred = pred + 1
    with pytest.raises(ValueError):
        module().assign_partitions(
            pred, available, maturity, module().build_folds(config)[0], clock, config
        )


def test_label_at_exact_boundary_is_not_mature_in_the_previous_partition():
    config = configuration()
    config["gap_sessions"] = 0
    clock = MarketClock("US", "2021-12-01", "2024-01-08")
    prediction = micros("2021-12-30T21:05:00")
    maturity = micros("2022-01-01T00:00:00")
    result = module().assign_partitions(
        prediction, prediction, maturity, module().build_folds(config)[0], clock, config
    )
    assert result["partition"].tolist() == ["excluded"]


def test_two_session_margin_excludes_an_already_mature_training_example():
    config = configuration()
    config["gap_sessions"] = 2
    clock = MarketClock("US", "2021-12-01", "2024-01-08")
    prediction = micros("2021-12-30T21:05:00")
    maturity = micros("2021-12-31T21:05:00")
    result = module().assign_partitions(
        prediction, prediction, maturity, module().build_folds(config)[0], clock, config
    )
    assert result["partition"].tolist() == ["excluded"]
    assert result["reason"].tolist() == ["session_gap"]
