"""Contrato mínimo de los recibos walk-forward que leen los entornos de refuerzo."""

import copy
import json

import numpy as np
import pytest

from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.environments.walk_forward_receipt import (
    prediction_fingerprint,
    read_window_receipt,
)
from mars_titan.evaluation.splits import PARTITIONS, build_folds
from tests.environments.walk_forward_fixture import ROOT, fold, microseconds, protocol, receipt

VALUES = dict(
    prediction_at=np.array([30, 10, 20, 10], dtype=np.int64),
    asset_id=["US/B", "US/A", "US/A", "US/B"],
    score=np.array([0.3, 0.1, 0.2, -0.1]),
)


def test_valid_receipt_exposes_the_protocol_bounds_in_microseconds():
    window = read_window_receipt(receipt("US"))
    expected = fold("US")
    assert window.market == "US" and window.fold == expected["id"] == "fold-018"
    for name in PARTITIONS:
        start, end = window.segment(name)
        assert (start, end) == tuple(microseconds(day) for day in expected[name])
    assert window.segment("evaluation")[1] == FINAL_TEST_START_US
    assert window.labels_used_until == microseconds("2023-01-01") - 1
    assert window.identity("calibration")["partition"] == "calibration"
    with pytest.raises(ValueError, match="tramo"):
        window.segment("test")


@pytest.mark.parametrize("index", [0, 5, -1])
def test_every_protocol_window_is_admitted_and_ends_before_2024(index):
    for market in ("US", "CN"):
        window = read_window_receipt(receipt(market, index=index))
        assert all(window.segment(name)[1] <= FINAL_TEST_START_US for name in PARTITIONS)


def mutated(change):
    value = receipt("US")
    change(value)
    return value


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(kind="other"),
        lambda r: r.update(schema_version=True),
        lambda r: r.update(extra=1),
        lambda r: r.pop("parent"),
        # Protocolo v1, con margen fijo, o con el test reservado desplazado.
        lambda r: r["protocol"].update(schema_version=1),
        lambda r: r["protocol"].update(final_test_start="2025-01-01"),
        lambda r: r["protocol"].update(evaluation_months=6),
    ],
)
def test_receipts_without_a_v2_protocol_and_2024_reserve_are_rejected(change):
    with pytest.raises(ValueError):
        read_window_receipt(mutated(change))


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["fold"].update(evaluation=["2023-01-01", "2024-02-01"]),
        lambda r: r["fold"].update(validation=["2022-04-01", "2022-11-01"]),
        lambda r: r["fold"].update(id="fold-999"),
        lambda r: r["fold"].update(train=["1999-01-01", "2022-04-01"]),
    ],
)
def test_hand_written_bounds_that_differ_from_the_protocol_are_rejected(change):
    with pytest.raises(ValueError, match="ventana no pertenece"):
        read_window_receipt(mutated(change))


@pytest.mark.parametrize(
    "until", [microseconds("2023-01-01"), microseconds("2023-06-01"), -1, 1.0, None]
)
def test_labels_used_on_or_after_the_evaluation_start_are_rejected(until):
    value = receipt("US")
    value["labels_used_until"] = until
    with pytest.raises(ValueError, match="etiqueta usada"):
        read_window_receipt(value)


@pytest.mark.parametrize(
    "predictions",
    [
        {"test": {"rows": 1, "sha256": "a" * 64}},
        {"evaluation": {"rows": 0, "sha256": "a" * 64}},
        {"evaluation": {"rows": 1, "sha256": "A" * 64}},
        {"evaluation": {"rows": 1, "sha256": "a" * 64, "path": "x"}},
        [],
    ],
)
def test_prediction_records_need_rows_and_a_fingerprint(predictions):
    value = receipt("US")
    value["predictions"] = predictions
    with pytest.raises(ValueError, match="huellas"):
        read_window_receipt(value)


@pytest.mark.parametrize(
    "parent",
    [{"id": "", "sha256": "b" * 64}, {"id": "p", "sha256": "x"}, {"id": "p"}, None],
)
def test_the_window_parent_must_be_identified(parent):
    value = receipt("US")
    value["parent"] = parent
    with pytest.raises(ValueError, match="predictor"):
        read_window_receipt(value)


def test_receipt_digest_changes_with_any_field():
    base = read_window_receipt(receipt("US"))
    later = read_window_receipt(receipt("US", until=microseconds("2022-12-30")))
    other = copy.deepcopy(receipt("US"))
    other["parent"]["id"] = "another"
    assert len({base.sha256, later.sha256, read_window_receipt(other).sha256}) == 3
    assert base.sha256 == read_window_receipt(receipt("US")).sha256


def test_fingerprint_ignores_row_order_but_not_values_or_keys():
    rows, digest = prediction_fingerprint(**VALUES)
    order = [3, 1, 0, 2]
    shuffled = {key: np.asarray(value)[order] for key, value in VALUES.items()}
    assert prediction_fingerprint(**shuffled) == (rows, digest) and rows == 4
    changed = dict(VALUES, score=VALUES["score"] + np.array([0, 0, 0, 1e-12]))
    moved = dict(VALUES, prediction_at=VALUES["prediction_at"] + np.array([0, 0, 0, 1]))
    renamed = dict(VALUES, asset_id=["US/B", "US/A", "US/A", "US/C"])
    for variant in (changed, moved, renamed):
        assert prediction_fingerprint(**variant)[1] != digest


@pytest.mark.parametrize(
    "values",
    [
        dict(VALUES, asset_id=["US/B", "US/A", "US/A", "US/A"]),
        dict(VALUES, score=np.array([0.3, np.nan, 0.2, 0.1])),
        dict(VALUES, prediction_at=VALUES["prediction_at"].astype(float)),
        dict(VALUES, asset_id=["US/B", "US/A", "US/A\nUS/X", "US/B"]),
        dict(prediction_at=np.array([], dtype=np.int64), asset_id=[], score=np.array([])),
    ],
)
def test_fingerprint_rejects_duplicates_non_finite_scores_and_ambiguous_keys(values):
    with pytest.raises(ValueError):
        prediction_fingerprint(**values)


def test_cn_and_us_protocols_share_the_2023_window_dates():
    us, cn = read_window_receipt(receipt("US")), read_window_receipt(receipt("CN"))
    assert us.bounds == cn.bounds and us.fold != cn.fold and protocol("CN")["market"] == "CN"


def test_a_v1_protocol_with_a_fixed_session_gap_is_rejected():
    # La versión 1 purga con un margen de sesiones que los límites del recibo no representan.
    path = ROOT / "configs/evaluation/historical-masked-us-walk-forward.json"
    legacy = json.loads(path.read_text())
    value = receipt("US")
    value["protocol"], value["fold"] = legacy, build_folds(legacy)[-1]
    value["labels_used_until"] = microseconds(value["fold"]["evaluation"][0]) - 1
    with pytest.raises(ValueError, match="protocolo walk-forward v2"):
        read_window_receipt(value)
