"""Conservar las ventanas verificadas al preparar ajustes posteriores."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.environments.corpus_source import ParquetCohortSource, prepare_causal_corpus
from mars_titan.episodes.parents import ParentCache
from mars_titan.posttraining.inputs import PairedInputs
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.test_temporal_corpus import inputs as inputs
from tests.training.test_temporal_corpus import prepare


def ordered_view(inputs, tmp_path):
    prepare(inputs, tmp_path / "views")
    source = tmp_path / "views/fold-000/manifest.json"
    destination = tmp_path / "ordered"
    report = prepare_causal_corpus(source, destination, batch_size=2)
    return source, destination / "manifest.json", report


def test_temporal_posttraining_keeps_all_train_rows_and_excludes_reserved_partitions(
    inputs, tmp_path
):
    source, ordered, report = ordered_view(inputs, tmp_path)
    assert report["counts"] == dict(train=2, validation=1, calibration=1, evaluation=1)
    assert set(report["partitions"]) == {"train", "validation"}
    assert report["grid"]["training_samples"] == 2
    assert report["source_manifest"]["sha256"] == sha256(source)
    expected = CorpusDataset(source)
    with (
        ParquetCohortSource(ordered, partition="train") as train,
        ParquetCohortSource(ordered, partition="validation") as validation,
        ParentCache(
            tmp_path / "parent.sqlite",
            "a" * 64,
            {"fixture": "zero"},
            lambda x: np.zeros(len(x["prices"])),
        ) as parent,
    ):
        data = PairedInputs(train, validation, parent)
        for partition in ("train", "validation"):
            reference = {
                key: (batch["target"][index], batch["inputs"]["macro"][index])
                for batch in expected.batches(partition=partition, batch_size=2, epoch=0, seed=0)
                for index, key in enumerate(batch["sample_ids"])
            }
            actual = list(data.batches(partition=partition, condition="real", batch_size=2))
            assert sum(len(batch["target"]) for batch in actual) == report["counts"][partition]
            for batch in actual:
                for index, key in enumerate(batch["sample_ids"]):
                    target, macro = reference[key]
                    assert batch["target"][index] == target
                    np.testing.assert_array_equal(batch["inputs"]["macro"][index], macro)
        assert any(row[0] >= 1_672_531_200_000_000 for row in train.index)
        with pytest.raises(ValueError):
            list(data.batches(partition="evaluation", condition="real", batch_size=2))


@pytest.mark.parametrize("field", ["path", "sha256", "missing"])
def test_temporal_ordered_source_rejects_a_changed_supervision_binding(inputs, tmp_path, field):
    source, ordered, report = ordered_view(inputs, tmp_path)
    if field == "path":
        report["source_manifest"][field] = str(tmp_path / "views/fold-001/manifest.json")
    elif field == "sha256":
        report["source_manifest"][field] = "f" * 64
    else:
        del report["source_manifest"]
    ordered.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="origen|supervisión|huella"):
        ParquetCohortSource(ordered, partition="train")


def test_temporal_source_rejects_a_cohort_in_the_purge_gap(inputs, tmp_path):
    _, ordered, _ = ordered_view(inputs, tmp_path)
    with ParquetCohortSource(ordered, partition="train") as source:
        metadata = json.loads(ordered.read_text())
        metadata["partitions"]["train"]["cohorts"][-1][0] = source.bounds[2]
    ordered.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="índice|cohorte"):
        ParquetCohortSource(ordered, partition="train")


def test_reserved_targets_do_not_change_training_grid_or_exported_rows(inputs, tmp_path):
    source, _, before = ordered_view(inputs, tmp_path)
    meta = json.loads(source.read_text())
    labels = Path(meta["roots"]["labels"]) / "US/A0000/labels.parquet"
    table = pq.read_table(labels)
    rows = table.to_pylist()
    changed = 0
    for row in rows:
        if row["partition"] in {"calibration", "evaluation"}:
            row["target"] += 1_000_000
            changed += 1
    assert changed == 2
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), labels)
    meta["assets"][0]["labels_sha256"] = sha256(labels)
    source.write_text(json.dumps(meta))
    after = prepare_causal_corpus(source, tmp_path / "changed-ordered", batch_size=2)
    assert before["source_sha256"] != after["source_sha256"]
    assert {k: v for k, v in before["grid"].items() if k != "source_sha256"} == {
        k: v for k, v in after["grid"].items() if k != "source_sha256"
    }
    for partition in ("train", "validation"):
        assert before["partitions"][partition] == after["partitions"][partition]


def test_temporal_preparation_resumes_after_a_complete_partition(inputs, tmp_path, monkeypatch):
    from mars_titan.environments import corpus_source

    prepare(inputs, tmp_path / "views")
    source = tmp_path / "views/fold-000/manifest.json"
    output = tmp_path / "ordered"
    stop = StopRequest()
    original = corpus_source._partition

    def finish_partition(*args):
        result = original(*args)
        stop.requested = True
        return result

    with monkeypatch.context() as patch:
        patch.setattr(corpus_source, "_partition", finish_partition)
        paused = prepare_causal_corpus(source, output, batch_size=2, stop=stop)
    assert paused["status"] == "paused"
    assert set(paused["partitions"]) == {"train"}
    completed = prepare_causal_corpus(source, output, batch_size=2, resume=True)
    assert completed["status"] == "completed"
    assert completed["partitions"]["train"] == paused["partitions"]["train"]
    assert completed["source_manifest"] == paused["source_manifest"]
    assert prepare_causal_corpus(source, output, batch_size=2, resume=True) == completed


def test_parent_population_checks_include_unread_temporal_partitions(inputs, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mars_titan.environments.actions import ActionGrid
    from mars_titan.posttraining import run
    from mars_titan.posttraining.inputs import fit_normalization
    from mars_titan.posttraining.queue import read_design

    _, ordered, report = ordered_view(inputs, tmp_path)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(run, "require_device", lambda *args: None)
    with (
        ParquetCohortSource(ordered, partition="train") as train,
        ParquetCohortSource(ordered, partition="validation") as validation,
        ParentCache(
            tmp_path / "parent.sqlite", "a" * 64, "zero", lambda x: np.zeros(len(x["prices"]))
        ) as cache,
    ):
        data = PairedInputs(train, validation, cache)
        parent = SimpleNamespace(
            identity=dict(
                checkpoint_sha256="a" * 64,
                source_sha256=train.source_sha256,
                counts=dict(report["counts"]),
            )
        )
        case = read_design("configs/baselines/real-continuations-v2.json")[1]("gru")[0]["case"]
        args = dict(
            parent=parent,
            batch_size=2,
            device="cuda:0",
            diagnostic=False,
            lease=None,
            checkpoint_seconds=60,
            max_updates=None,
        )
        grid = ActionGrid.from_dict(report["grid"])
        norm = fit_normalization(data, batch_size=2)
        assert run._validate_run(data, case, grid, norm, **args)[1]["rows"] == 2
        parent.identity["counts"]["evaluation"] += 1
        with pytest.raises(ValueError, match="población"):
            run._validate_run(data, case, grid, norm, **args)
