"""Vistas temporales que comparten modalidades y exigen el contexto macro completo."""

import csv
import importlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.macro_coverage import assess_macro_completeness
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.test_corpus_inputs import corpus


@pytest.fixture
def inputs(tmp_path):
    parent = corpus(tmp_path / "parent", assets=1, rows=10)
    meta = json.loads(parent.read_text())
    clock = MarketClock("US", "2022-11-01", "2024-02-01")
    days = [
        "2022-11-15",
        "2023-05-30",
        "2023-05-31",
        "2023-06-15",
        "2023-07-14",
        "2023-08-15",
        "2023-09-15",
        "2023-10-16",
        "2023-12-20",
        "2024-01-03",
    ]
    times = [clock.decision(day) for day in days]
    maturity = [clock.decisions[clock.decisions.index(t) + 1] for t in times]
    samples = Path(meta["roots"]["samples"]) / "US/A0000/samples.parquet"
    rows = pq.read_table(samples).to_pylist()
    for row, moment in zip(rows, times, strict=True):
        row["prediction_at"] = moment
        row["input_availability"] = {
            name: moment for name in ("prices", "news", "charts", "fundamentals", "macro")
        }
    rows[7]["input_availability"]["news"] = maturity[7]
    pq.write_table(pa.Table.from_pylist(rows), samples, row_group_size=3)
    labels = Path(meta["roots"]["labels"]) / "US/A0000/labels.parquet"
    table = pa.table(
        dict(
            sample_row=np.arange(10),
            prediction_at=times,
            target_available_at=maturity,
            target=[i / 100 if i != 9 else None for i in range(10)],
            partition=["train"] + ["validation"] * 8 + [None],
            reason=["accepted"] * 9 + ["final_test_reserved"],
        )
    )
    pq.write_table(table, labels)
    meta["assets"][0].update(
        samples_sha256=sha256(samples),
        labels_sha256=sha256(labels),
        counts={"train": 1, "validation": 8},
    )
    meta["counts"] = dict(train=1, validation=8)
    meta["final_test_opened"] = False
    parent.write_text(json.dumps(meta))
    with Path("data/catalogs/macro-indicators.csv").open() as stream:
        catalog = [r for r in csv.DictReader(stream) if r["id"] in {"us_cpi", "us_unemployment"}]
    catalog_path = tmp_path / "catalog/catalog.csv"
    catalog_path.parent.mkdir()
    with catalog_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=catalog[0])
        writer.writeheader()
        writer.writerows(catalog)
    macro_rows = []
    for index, moment in enumerate(times[:-1]):
        for entry in catalog:
            macro_rows.append(
                dict(
                    prediction_at=moment,
                    indicator_id=entry["id"],
                    value=None if index == 4 else 10.0,
                    available_at=moment,
                    period_start="2022-10-01",
                    missing_reason="missing_source_value" if index == 4 else None,
                    source_hashes=["a" * 64],
                    unit="published",
                    seasonal_adjustment="published",
                )
            )
    macro = tmp_path / "macro-source/macro.parquet"
    macro.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(macro_rows), macro, row_group_size=4)
    admission = tmp_path / "admission"
    assess_macro_completeness(
        macro, catalog_path, admission, market="US", start="2022-11-01", end="2023-12-31"
    )
    protocol = Path("configs/evaluation/strict-macro-walk-forward.json")
    return parent, protocol, macro, admission / "report.json"


def prepare(inputs, output):
    return importlib.import_module("mars_titan.training.temporal_corpus").prepare_temporal_corpus(
        *inputs, output
    )


def test_views_keep_modalities_use_new_macro_and_visit_temporal_partitions(inputs, tmp_path):
    report = prepare(inputs, tmp_path / "views")
    assert len(report["folds"]) == 4
    manifest = tmp_path / "views/fold-000/manifest.json"
    meta = json.loads(manifest.read_text())
    original = json.loads(inputs[0].read_text())
    assert meta["roots"]["samples"] == original["roots"]["samples"]
    assert meta["assets"][0]["samples_sha256"] == original["assets"][0]["samples_sha256"]
    assert meta["counts"] == dict(train=2, validation=1, calibration=1, evaluation=1)
    reader = CorpusDataset(manifest)
    for partition, count in meta["counts"].items():
        batches = list(reader.batches(partition=partition, batch_size=1, epoch=0, seed=42))
        assert len(batches) == count
        assert all(np.all(b["inputs"]["macro"][:, 2:4] == 1) for b in batches)
        assert all(np.allclose(b["inputs"]["macro"][:, :2], np.log1p(10)) for b in batches)
        assert all(np.all(b["input_available_at"] <= b["prediction_at"]) for b in batches)
    labels = pq.read_table(Path(meta["roots"]["labels"]) / "US/A0000/labels.parquet").to_pylist()
    assert labels[2]["reason"] == "session_gap"
    assert labels[4]["reason"] == "incomplete_inputs"
    assert labels[7]["reason"] == "future_inputs"
    assert labels[9]["target"] is None
    assert labels[9]["reason"] == "final_test_reserved"
    assert report["scientific_training_started"] is False


def test_view_recovers_exact_batch_order_and_rejects_changed_macro(inputs, tmp_path):
    prepare(inputs, tmp_path / "views")
    path = tmp_path / "views/fold-000/manifest.json"
    reader = CorpusDataset(path)
    batches = list(reader.batches(partition="train", batch_size=1, epoch=0, seed=42))
    resumed = list(
        reader.batches(
            partition="train", batch_size=1, epoch=0, seed=42, cursor=batches[0]["confirmed_cursor"]
        )
    )
    assert resumed[0]["sample_ids"] == batches[1]["sample_ids"]
    inputs[2].write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        list(reader.batches(partition="train", batch_size=1, epoch=0, seed=42))


def test_view_refuses_missing_modality_and_reserved_test_selection(inputs, tmp_path):
    parent = json.loads(inputs[0].read_text())
    path = Path(parent["roots"]["samples"]) / "US/A0000/samples.parquet"
    table = pq.read_table(path).drop(["charts"])
    pq.write_table(table, path)
    parent["assets"][0]["samples_sha256"] = sha256(path)
    inputs[0].write_text(json.dumps(parent))
    with pytest.raises(ValueError):
        prepare(inputs, tmp_path / "views")
    assert not (tmp_path / "views").exists()


def test_source_exclusion_reason_is_preserved_without_truncation(inputs, tmp_path):
    meta = json.loads(inputs[0].read_text())
    path = Path(meta["roots"]["labels"]) / "US/A0000/labels.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[8].update(partition=None, target=None, reason="target_crosses_partition_boundary")
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    meta["assets"][0].update(labels_sha256=sha256(path), counts=dict(train=1, validation=7))
    meta["counts"] = dict(train=1, validation=7)
    inputs[0].write_text(json.dumps(meta))
    prepare(inputs, tmp_path / "views")
    view = CorpusDataset(tmp_path / "views/fold-000/manifest.json")
    assert len(list(view.batches(partition="train", batch_size=1, epoch=0, seed=42))) == 2
    label = pq.read_table(view.roots["labels"] / "US/A0000/labels.parquet").to_pylist()[8]
    assert label["reason"] == "target_crosses_partition_boundary"


def test_forged_train_partition_is_rejected_even_if_counts_and_hashes_are_updated(inputs, tmp_path):
    prepare(inputs, tmp_path / "views")
    path = tmp_path / "views/fold-000/manifest.json"
    view = json.loads(path.read_text())
    labels_path = Path(view["roots"]["labels"]) / "US/A0000/labels.parquet"
    table = pq.read_table(labels_path)
    rows = table.to_pylist()
    rows[6]["partition"] = "train"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), labels_path)
    view["assets"][0]["labels_sha256"] = sha256(labels_path)
    view["assets"][0]["counts"].update(train=3, evaluation=0)
    view["counts"].update(train=3, evaluation=0)
    path.write_text(json.dumps(view))
    with pytest.raises(ValueError, match="partición"):
        list(CorpusDataset(path).batches(partition="train", batch_size=1, epoch=0, seed=42))


def test_preparation_preserves_existing_edition_and_parent_change_invalidates_reader(
    inputs, tmp_path
):
    output = tmp_path / "views"
    prepare(inputs, output)
    before = (output / "fold-000/manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare(inputs, output)
    assert (output / "fold-000/manifest.json").read_bytes() == before
    reader = CorpusDataset(output / "fold-000/manifest.json")
    parent = json.loads(inputs[0].read_text())
    parent["changed"] = True
    inputs[0].write_text(json.dumps(parent))
    with pytest.raises(ValueError, match="fuente"):
        list(reader.batches(partition="train", batch_size=1, epoch=0, seed=42))


def test_campaign_view_keeps_calibration_and_evaluation_counts(inputs, tmp_path):
    prepare(inputs, tmp_path / "views")
    campaign = importlib.import_module("mars_titan.training.reference_campaign")
    path = tmp_path / "views/fold-000/manifest.json"
    view = campaign.campaign_views(path, ["US"])["US"]
    assert view["counts"] == dict(train=2, validation=1, calibration=1, evaluation=1)


def test_scientific_identity_covers_temporal_transforms_and_calendar(monkeypatch):
    runner = importlib.import_module("mars_titan.training.reference_run")
    monkeypatch.setattr(runner.torch.cuda, "get_device_name", lambda device: "GPU de prueba")
    identity = runner.scientific_identity()
    assert {
        "training/temporal_corpus.py",
        "evaluation/splits.py",
        "evaluation/split_readiness.py",
        "data/cohort_contexts.py",
        "data/samples.py",
        "data/temporal.py",
    } <= identity["code"].keys()
    assert identity["exchange_calendars"]


def test_view_cannot_replace_the_parent_macro_catalog_with_another_catalog(inputs, tmp_path):
    parent = json.loads(inputs[0].read_text())
    parent["representation"] = {"macro_indicators": ["us_cpi", "unverified_indicator"]}
    inputs[0].write_text(json.dumps(parent))
    with pytest.raises(ValueError, match="conceptos macro"):
        prepare(inputs, tmp_path / "views")
