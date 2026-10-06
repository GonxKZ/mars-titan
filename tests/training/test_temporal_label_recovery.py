"""Recuperación explícita del corte anual antiguo sin saltar las purgas nuevas."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.macro_coverage import assess_macro_completeness
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.temporal_corpus import prepare_temporal_corpus
from tests.training.test_temporal_corpus import inputs  # noqa: F401


@pytest.fixture
def annual_inputs(inputs, tmp_path):  # noqa: F811 - pytest inyecta la fixture importada.
    parent, protocol, macro, _ = inputs
    meta = json.loads(parent.read_text())
    clock = MarketClock("US", "2022-11-01", "2024-02-01")
    prediction, maturity = clock.decision("2022-12-30"), clock.decision("2023-01-03")
    samples = Path(meta["roots"]["samples"]) / "US/A0000/samples.parquet"
    table = pq.read_table(samples)
    rows = table.to_pylist()
    previous = rows[1]["prediction_at"]
    rows[1]["prediction_at"] = prediction
    rows[1]["input_availability"] = {name: prediction for name in rows[1]["input_availability"]}
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), samples, row_group_size=3)
    labels = Path(meta["roots"]["labels"]) / "US/A0000/labels.parquet"
    table = pq.read_table(labels)
    rows = table.to_pylist()
    rows[1].update(
        prediction_at=prediction,
        target_available_at=maturity,
        target=0.01,
        partition=None,
        reason="target_crosses_partition_boundary",
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), labels)
    meta["assets"][0].update(
        samples_sha256=sha256(samples),
        labels_sha256=sha256(labels),
        counts=dict(train=1, validation=7),
    )
    meta["counts"] = dict(train=1, validation=7)
    parent.write_text(json.dumps(meta))
    table = pq.read_table(macro)
    rows = table.to_pylist()
    for row in rows:
        if row["prediction_at"] == previous:
            row.update(prediction_at=prediction, available_at=prediction)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), macro)
    admission = tmp_path / "annual-admission"
    assess_macro_completeness(
        macro,
        tmp_path / "catalog/catalog.csv",
        admission,
        market="US",
        start="2022-11-01",
        end="2023-12-31",
    )
    return parent, protocol, macro, admission / "report.json"


def _label(output):
    path = output / "fold-000/manifest.json"
    reader = CorpusDataset(path)
    batches = list(reader.batches(partition="train", batch_size=1, epoch=0, seed=42))
    table = pq.read_table(reader.roots["labels"] / "US/A0000/labels.parquet")
    return table.to_pylist()[1], batches, reader.manifest


def test_annual_target_is_preserved_by_default_and_recovered_only_when_requested(
    annual_inputs, tmp_path
):
    original_hashes = [sha256(path) for path in annual_inputs]
    legacy, recovered = tmp_path / "legacy", tmp_path / "recovered"
    prepare_temporal_corpus(*annual_inputs, legacy)
    old, old_batches, _ = _label(legacy)
    assert old["reason"] == "target_crosses_partition_boundary"
    assert old["target"] is None
    assert len(old_batches) == 1
    report = prepare_temporal_corpus(*annual_inputs, recovered, recover_annual_boundaries=True)
    row, batches, view = _label(recovered)
    assert row["target"] == 0.01
    assert row["partition"] == "train"
    assert row["reason"] == "accepted"
    assert row["source_reason"] == "target_crosses_partition_boundary"
    assert len(batches) == 2
    assert report["recovered_annual_candidates"] == 1
    assert report["folds"][0]["recovered_annual_labels"] == 1
    assert view["label_admission"]["recover_annual_boundaries"] is True
    assert original_hashes == [sha256(path) for path in annual_inputs]
    assert report["final_test_opened"] is False


def test_recovered_label_still_obeys_new_partition_boundary(annual_inputs, tmp_path):
    parent, protocol, macro, admission = annual_inputs
    config = json.loads(protocol.read_text())
    config.update(first_validation_start="2023-01-01", minimum_train_months=1)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(config))
    output = tmp_path / "purged"
    report = prepare_temporal_corpus(
        parent, protocol, macro, admission, output, recover_annual_boundaries=True
    )
    row, _, _ = _label(output)
    assert row["reason"] == "session_gap"
    assert row["partition"] is None
    assert row["target"] is None
    assert row["source_reason"] == "target_crosses_partition_boundary"
    assert report["folds"][0]["recovered_annual_labels"] == 0


@pytest.mark.parametrize("mutation", ["null", "nan", "past", "wrong_year", "non_session", "test"])
def test_corrupt_annual_targets_cannot_be_readmitted(annual_inputs, tmp_path, mutation):
    parent = annual_inputs[0]
    meta = json.loads(parent.read_text())
    path = Path(meta["roots"]["labels"]) / "US/A0000/labels.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    row = rows[1]
    clock = MarketClock("US", "2022-11-01", "2024-02-01")
    if mutation in {"null", "nan"}:
        row["target"] = None if mutation == "null" else float("nan")
    else:
        row["target_available_at"] = {
            "past": row["prediction_at"],
            "wrong_year": clock.decision("2022-12-29"),
            "non_session": clock.decision("2023-01-04"),
            "test": clock.decision("2024-01-03"),
        }[mutation]
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    meta["assets"][0]["labels_sha256"] = sha256(path)
    parent.write_text(json.dumps(meta))
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="anual"):
        prepare_temporal_corpus(*annual_inputs, output, recover_annual_boundaries=True)
    assert not output.exists()
