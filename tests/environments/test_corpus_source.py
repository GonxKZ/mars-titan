"""Paso del corpus supervisado a cohortes Parquet sin cambiar muestras ni etiquetas."""

import importlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.test_corpus_inputs import corpus
from tests.training.test_corpus_targets import audited_edition
from tests.training.test_reference_run import training_corpus

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("duckdb", "gymnasium")),
    reason="Requiere los extras data y reinforcement",
)


def fixture(tmp_path):
    manifest = training_corpus(tmp_path)
    meta = json.loads(manifest.read_text())
    for asset in meta["assets"]:
        path = (
            Path(meta["roots"]["samples"]) / asset["market"] / asset["symbol"] / "samples.parquet"
        )
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row["input_availability"] = {
                name: row["prediction_at"]
                for name in (
                    "prices",
                    "news",
                    "charts",
                    "fundamentals",
                    "macro",
                )
            }
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
        asset["samples_sha256"] = sha256(path)
    manifest.write_text(json.dumps(meta))
    return manifest


def module():
    return importlib.import_module("mars_titan.environments.corpus_source")


def test_parquet_cohorts_preserve_entire_population_and_exact_features(tmp_path):
    manifest = fixture(tmp_path / "data")
    output = tmp_path / "ordered"
    result = module().prepare_causal_corpus(manifest, output, batch_size=5)
    assert result["status"] == "completed" and result["final_test_opened"] is False
    assert result["counts"] == {"train": 12, "validation": 6}
    dataset = CorpusDataset(manifest)
    for partition, expected_cohorts in (("train", 6), ("validation", 3)):
        expected = {}
        for batch in dataset.batches(partition=partition, batch_size=5, epoch=0, seed=0):
            for i, key in enumerate(batch["sample_ids"]):
                expected[key] = (
                    batch["target"][i],
                    {name: value[i] for name, value in batch["inputs"].items()},
                )
        with module().ParquetCohortSource(output / "manifest.json", partition=partition) as source:
            assert len(source) == expected_cohorts
            ids = []
            for position in range(len(source)):
                cohort = source(position)
                assert cohort["asset_ids"] == sorted(cohort["asset_ids"])
                for i, key in enumerate(cohort["asset_ids"]):
                    sample = f"{key}/{cohort['prediction_at']}"
                    ids.append(sample)
                    assert cohort["target"][i] == expected[sample][0]
                    for name, values in cohort["inputs"].items():
                        np.testing.assert_array_equal(values[i], expected[sample][1][name])
            assert set(ids) == set(expected) and len(ids) == len(expected)
            assert source(len(source)) is None
            with pytest.raises(ValueError):
                source(len(source) + 1)
    assert result["grid"]["training_samples"] == 12


def test_missing_availability_does_not_become_a_fabricated_timestamp(tmp_path):
    manifest = training_corpus(tmp_path / "data")
    with pytest.raises(ValueError, match="disponibilidad"):
        module().prepare_causal_corpus(manifest, tmp_path / "ordered")
    assert not (tmp_path / "ordered/manifest.json").exists()


def test_version_two_provenance_survives_export_without_external_verification_claim(tmp_path):
    from mars_titan.training.corpus_targets import prepare_corpus_targets

    materialized, prepared = audited_edition(tmp_path / "original")
    samples = tmp_path / "original/samples/US/AAA/samples.parquet"
    rows = pq.read_table(samples).to_pylist()
    for row in rows:
        row["input_availability"] = {
            name: row["prediction_at"]
            for name in (
                "prices",
                "news",
                "charts",
                "fundamentals",
                "macro",
            )
        }
    pq.write_table(pa.Table.from_pylist(rows), samples)
    receipt = samples.with_name("manifest.json")
    info = json.loads(receipt.read_text())
    info["samples_sha256"] = sha256(samples)
    receipt.write_text(json.dumps(info))
    prepare_corpus_targets(materialized, prepared, tmp_path / "labels")
    output = tmp_path / "ordered"
    report = module().prepare_causal_corpus(tmp_path / "labels/manifest.json", output)
    assert report["cohort_id"] == "original_audited"
    assert report["news_content_policy"] == "source_audited_not_external"
    with module().ParquetCohortSource(output / "manifest.json", partition="train") as source:
        assert source.cohort_id == "original_audited"
        assert source.news_content_policy == "source_audited_not_external"
        assert len(source(0)["asset_ids"]) == 1
    report["cohort_id"], report["news_content_policy"] = (
        "externally_verified",
        "verified_full_articles",
    )
    (output / "manifest.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="cohorte|procedencia"):
        with module().ParquetCohortSource(output / "manifest.json", partition="train") as source:
            source(0)


def test_completed_preparation_is_immutable_when_resumed(tmp_path):
    manifest = fixture(tmp_path / "data")
    output = tmp_path / "ordered"
    result = module().prepare_causal_corpus(manifest, output)
    before = {name: sha256(output / name) for name in ("manifest.json", "progress.json")}
    assert module().prepare_causal_corpus(manifest, output, resume=True) == result
    assert before == {name: sha256(output / name) for name in before}


def test_completed_progress_can_publish_missing_final_manifest_on_resume(tmp_path, monkeypatch):
    engine = module()
    manifest = fixture(tmp_path / "data")
    output = tmp_path / "ordered"
    original = engine.atomic_json

    def interrupt(path, value):
        if path.name == "manifest.json":
            raise OSError("Interrupción antes de publicar el manifiesto final")
        return original(path, value)

    monkeypatch.setattr(engine, "atomic_json", interrupt)
    with pytest.raises(OSError):
        engine.prepare_causal_corpus(manifest, output)
    assert json.loads((output / "progress.json").read_text())["status"] == "completed"
    monkeypatch.setattr(engine, "atomic_json", original)
    result = engine.prepare_causal_corpus(manifest, output, resume=True)
    assert result["status"] == "completed" and (output / "manifest.json").is_file()


def test_producer_repackages_groups_to_the_reader_budget(tmp_path, monkeypatch):
    engine = module()
    monkeypatch.setattr(engine, "MAX_BLOCK_BYTES", 4096)
    output = tmp_path / "ordered"
    engine.prepare_causal_corpus(fixture(tmp_path / "data"), output, batch_size=1)
    with engine.ParquetCohortSource(
        output / "manifest.json", partition="train", max_cache_bytes=4096
    ) as source:
        assert sum(len(source(i)["asset_ids"]) for i in range(len(source))) == 12


def test_real_legacy_availability_columns_are_supported_without_inventing_macro_dates(tmp_path):
    manifest = fixture(tmp_path / "data")
    meta = json.loads(manifest.read_text())
    for asset in meta["assets"]:
        path = (
            Path(meta["roots"]["samples"]) / asset["market"] / asset["symbol"] / "samples.parquet"
        )
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row["macro_available_at"] = row["input_availability"].pop("macro")
        pq.write_table(pa.Table.from_pylist(rows), path)
        asset["samples_sha256"] = sha256(path)
    manifest.write_text(json.dumps(meta))
    assert module().prepare_causal_corpus(manifest, tmp_path / "ordered")["status"] == "completed"


def test_macro_values_without_any_date_cannot_be_exported_as_causal(tmp_path):
    manifest = fixture(tmp_path / "data")
    meta = json.loads(manifest.read_text())
    path = Path(meta["roots"]["samples"]) / "US/A0000/samples.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["input_availability"]["macro"] = None
    pq.write_table(pa.Table.from_pylist(rows), path)
    meta["assets"][0]["samples_sha256"] = sha256(path)
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="disponibilidad"):
        module().prepare_causal_corpus(manifest, tmp_path / "ordered")


def test_file_changed_while_hashing_cannot_be_accepted_as_original(tmp_path, monkeypatch):
    engine = module()
    output = tmp_path / "ordered"
    result = engine.prepare_causal_corpus(fixture(tmp_path / "data"), output)
    changed = output / result["partitions"]["train"]["path"]
    real_hash = engine.sha256
    modified = False

    def race(path):
        nonlocal modified
        digest = real_hash(path)
        if Path(path) == changed and not modified:
            modified = True
            rows = pq.read_table(path).to_pylist()
            rows[0]["news"][0] += 1
            pq.write_table(pa.Table.from_pylist(rows), path)
        return digest

    monkeypatch.setattr(engine, "sha256", race)
    with pytest.raises(ValueError, match="cambiado|huella"):
        engine.ParquetCohortSource(output / "manifest.json", partition="train")


def test_cohort_reader_rejects_target_maturity_outside_its_partition(tmp_path):
    output = tmp_path / "ordered"
    module().prepare_causal_corpus(fixture(tmp_path / "data"), output)
    meta = json.loads((output / "manifest.json").read_text())
    old = output / meta["partitions"]["train"]["path"]
    rows = pq.read_table(old).to_pylist()
    rows[0]["target_available_at"] = 1_672_531_200_000_000
    changed = tmp_path / "changed.parquet"
    pq.write_table(pa.Table.from_pylist(rows), changed)
    digest = sha256(changed)
    path = output / f"train-{digest}.parquet"
    changed.rename(path)
    meta["partitions"]["train"].update(
        path=path.name, sha256=digest, size_bytes=path.stat().st_size
    )
    (output / "manifest.json").write_text(json.dumps(meta))
    with module().ParquetCohortSource(output / "manifest.json", partition="train") as source:
        with pytest.raises(ValueError, match="partición"):
            source(0)


def test_partition_receipt_resumes_without_rewriting_confirmed_parquet(tmp_path, monkeypatch):
    engine = module()
    manifest = fixture(tmp_path / "data")
    output = tmp_path / "ordered"
    stop = StopRequest()
    original = engine.atomic_json

    def pause(path, value):
        original(path, value)
        if path.name == "progress.json" and "train" in value.get("partitions", {}):
            stop.request_stop()

    monkeypatch.setattr(engine, "atomic_json", pause)
    partial = engine.prepare_causal_corpus(manifest, output, stop=stop)
    assert partial["status"] == "paused" and set(partial["partitions"]) == {"train"}
    train = output / partial["partitions"]["train"]["path"]
    before = (train.stat().st_ino, train.stat().st_mtime_ns, sha256(train))
    monkeypatch.setattr(engine, "atomic_json", original)
    result = engine.prepare_causal_corpus(manifest, output, resume=True)
    assert result["status"] == "completed"
    assert before == (train.stat().st_ino, train.stat().st_mtime_ns, sha256(train))
    with pytest.raises(ValueError):
        engine.prepare_causal_corpus(manifest, output)


def test_reader_rejects_changed_file_and_snapshot_restores_real_parquet_cursor(tmp_path):
    from mars_titan.environments.actions import ActionGrid
    from mars_titan.environments.prediction import CausalPredictionEnv

    output = tmp_path / "ordered"
    result = module().prepare_causal_corpus(fixture(tmp_path / "data"), output)
    with module().ParquetCohortSource(output / "manifest.json", partition="train") as source:
        grid = ActionGrid.from_dict(result["grid"])
        env = CausalPredictionEnv(
            source,
            source_sha256=source.source_sha256,
            grid=grid,
            shapes=source.shapes,
            max_assets=source.max_assets,
        )
        env.reset(seed=42)
        env.step(np.full(source.max_assets, 10))
        state = env.snapshot()
        expected = env.step(np.full(source.max_assets, 10))
        env.restore(state)
        observed = env.step(np.full(source.max_assets, 10))
        assert expected[1:] == observed[1:]
        for name in observed[0]:
            np.testing.assert_array_equal(observed[0][name], expected[0][name])
        path = output / result["partitions"]["train"]["path"]
        path.write_bytes(b"broken")
        with pytest.raises(ValueError, match="cambiado|huella"):
            source(0)


def test_cohorts_cross_parquet_groups_without_losing_rows_or_unbounded_cache(tmp_path):
    manifest = corpus(tmp_path / "data", assets=3, rows=3000, group_size=257)
    meta = json.loads(manifest.read_text())
    for asset in meta["assets"]:
        for kind in ("samples", "labels"):
            path = Path(meta["roots"][kind]) / asset["market"] / asset["symbol"] / f"{kind}.parquet"
            rows = pq.read_table(path).to_pylist()
            for i, row in enumerate(rows):
                if i >= 2000:
                    row["prediction_at"] = row["prediction_at"].replace(year=2023)
                    if kind == "labels":
                        row["partition"] = "validation"
                        row["target_available_at"] = row["target_available_at"].replace(year=2023)
                if kind == "samples":
                    row["input_availability"] = {
                        name: row["prediction_at"]
                        for name in (
                            "prices",
                            "news",
                            "charts",
                            "fundamentals",
                            "macro",
                        )
                    }
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=257)
            asset[kind + "_sha256"] = sha256(path)
        asset["counts"] = dict(train=2000, validation=1000)
    meta["counts"] = dict(train=6000, validation=3000)
    manifest.write_text(json.dumps(meta))
    output = tmp_path / "ordered"
    module().prepare_causal_corpus(manifest, output, batch_size=512)
    with module().ParquetCohortSource(output / "manifest.json", partition="train") as source:
        assert source.file.num_row_groups >= 3
        observed = 0
        for i in range(len(source)):
            row = source(i)
            observed += len(row["asset_ids"])
            assert row["prediction_at"] == source.index[i][0]
            assert len(source.cache) <= 2 and source.cache_bytes <= source.max_cache_bytes
        assert observed == 6000
    with module().ParquetCohortSource(
        output / "manifest.json", partition="train", max_cache_bytes=1
    ) as source:
        with pytest.raises(ValueError, match="presupuesto"):
            source(0)
