"""Identidad y alineación completa de predicciones padre, sin duplicar modalidades."""

import importlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.training.tabular_corpus import run_tabular_reference
from tests.environments.test_corpus_source import fixture

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("duckdb", "gymnasium")),
    reason="Requiere los extras data y reinforcement",
)


def module():
    return importlib.import_module("mars_titan.training.predictive_parents")


def setup(tmp_path):
    from mars_titan.environments.corpus_source import prepare_causal_corpus

    original = fixture(tmp_path / "data")
    ordered, parent = tmp_path / "ordered", tmp_path / "parent"
    prepare_causal_corpus(original, ordered)
    report = run_tabular_reference(original, parent, kind="ridge", batch_size=5)
    return ordered / "manifest.json", parent / "run.json", report


def test_parent_cache_aligns_every_prediction_without_copying_modalities(tmp_path):
    from mars_titan.environments.corpus_source import ParquetCohortSource

    ordered, parent, report = setup(tmp_path)
    output = tmp_path / "cache"
    cached = module().prepare_parent_cache(ordered, parent, output)
    assert cached["status"] == "completed" and cached["model"] == "ridge"
    assert cached["counts"] == {"train": 12, "validation": 6}
    assert cached["checkpoint_sha256"] == report["checkpoint"]["sha256"]
    for partition in ("train", "validation"):
        expected = {
            row["sample_id"]: row["prediction"]
            for row in pq.read_table(
                parent.parent / report["predictions"][partition]["path"],
            ).to_pylist()
        }
        with ParquetCohortSource(ordered, partition=partition) as source:
            aligned = module().ParentPredictions(output / "manifest.json", source)
            offset = 0
            for index in range(len(source)):
                cohort = source(index)
                wanted = [
                    expected[f"{asset}/{cohort['prediction_at']}"] for asset in cohort["asset_ids"]
                ]
                np.testing.assert_array_equal(aligned.values(offset, len(wanted)), wanted)
                offset += len(wanted)
            assert aligned.dtype == np.dtype("float64")
            assert aligned.payload_bytes == cached["counts"][partition] * 8
            aligned.close()
    before = sha256(output / "manifest.json")
    assert module().prepare_parent_cache(ordered, parent, output, resume=True) == cached
    assert sha256(output / "manifest.json") == before


@pytest.mark.parametrize(
    "problem",
    ["missing", "duplicate", "wrong_target", "wrong_asset", "wrong_time", "wrong_market", "nan"],
)
def test_parent_join_rejects_incomplete_duplicate_or_different_labels(tmp_path, problem):
    ordered, parent, report = setup(tmp_path)
    path = parent.parent / report["predictions"]["train"]["path"]
    rows = pq.read_table(path).to_pylist()
    if problem == "missing":
        rows = rows[:-1]
    if problem == "duplicate":
        rows[-1] = rows[0]
    if problem == "wrong_target":
        rows[0]["target"] += 0.1
    if problem == "wrong_asset":
        rows[0]["asset_id"] = "US/WRONG"
    if problem == "wrong_time":
        rows[0]["prediction_at"] = rows[0]["prediction_at"].replace(year=2020)
    if problem == "wrong_market":
        rows[0]["market"] = "CN"
    if problem == "nan":
        rows[0]["prediction"] = float("nan")
    pq.write_table(pa.Table.from_pylist(rows), path)
    report["predictions"]["train"]["sha256"] = sha256(path)
    parent.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        module().prepare_parent_cache(ordered, parent, tmp_path / "cache")
    assert not (tmp_path / "cache/manifest.json").exists()


@pytest.mark.parametrize("problem", ["manifest", "checkpoint", "status", "test", "model"])
def test_parent_contract_must_match_completed_reference(tmp_path, problem):
    ordered, parent, report = setup(tmp_path)
    if problem == "manifest":
        report["manifest_sha256"] = "b" * 64
    if problem == "checkpoint":
        report["checkpoint"]["sha256"] = "b" * 64
    if problem == "status":
        report["status"] = "running"
    if problem == "test":
        report["final_test_opened"] = True
    if problem == "model":
        report["model"] = "mars-titan"
    parent.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        module().prepare_parent_cache(ordered, parent, tmp_path / "cache")


def test_cached_values_are_read_only_and_changed_cache_is_rejected(tmp_path):
    from mars_titan.environments.corpus_source import ParquetCohortSource

    ordered, parent, _ = setup(tmp_path)
    output = tmp_path / "cache"
    report = module().prepare_parent_cache(ordered, parent, output)
    with ParquetCohortSource(ordered, partition="train") as source:
        aligned = module().ParentPredictions(output / "manifest.json", source)
        values = aligned.values(0, 1)
        with pytest.raises(ValueError):
            values[0] = 2.0
        path = output / report["partitions"]["train"]["path"]
        values = np.load(path).copy()
        values[0] += 1
        with path.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        with pytest.raises(ValueError, match="cambiado|huella"):
            aligned.values(0, 1)
        aligned.close()


def test_parent_cache_recovers_confirmed_partition_after_interruption(tmp_path, monkeypatch):
    engine = module()
    ordered, parent, _ = setup(tmp_path)
    output = tmp_path / "cache"
    original = engine._align

    def interrupted(source, *args, **kwargs):
        if source.partition == "validation":
            raise InterruptedError("Interrupción de la segunda partición")
        return original(source, *args, **kwargs)

    monkeypatch.setattr(engine, "_align", interrupted)
    with pytest.raises(InterruptedError):
        engine.prepare_parent_cache(ordered, parent, output)
    partial = json.loads((output / "progress.json").read_text())
    first = output / partial["partitions"]["train"]["path"]
    signature = (first.stat().st_ino, first.stat().st_mtime_ns, sha256(first))
    monkeypatch.setattr(engine, "_align", original)
    assert (
        engine.prepare_parent_cache(ordered, parent, output, resume=True)["status"] == "completed"
    )
    assert signature == (first.stat().st_ino, first.stat().st_mtime_ns, sha256(first))
