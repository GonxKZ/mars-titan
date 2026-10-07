"""Lectura conjunta de ventanas locales sin perder sesiones ni mezclar calendarios."""

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.temporal_fixture import temporal_fixture


def joint_fixture(root, fold="fold-000", *, extra_days=()):
    local = {
        market: temporal_fixture(root / market, market, extra_days=extra_days)
        for market in ("US", "CN")
    }
    metas = {
        market: json.loads(
            (fixture.parent if fold is None else fixture.views / fold / "manifest.json").read_text()
        )
        for market, fixture in local.items()
    }
    destination = root / "joint"
    roots = {name: destination / name for name in ("prepared", "samples", "labels")}
    for market, meta in metas.items():
        for name, target in roots.items():
            shutil.copytree(Path(meta["roots"][name]) / market, target / market)
    meta = copy.deepcopy(metas["US"])
    meta.pop("temporal_view", None)
    meta.update(
        roots={name: str(path) for name, path in roots.items()},
        markets=["CN", "US"],
        assets=[asset for value in metas.values() for asset in value["assets"]],
        counts={p: sum(value["counts"][p] for value in metas.values()) for p in meta["counts"]},
    )
    if fold is not None:
        meta["temporal_views"] = {market: value["temporal_view"] for market, value in metas.items()}
    manifest = destination / "manifest.json"
    atomic_json(manifest, meta)
    return SimpleNamespace(manifest=manifest, local=local, metadata=meta, fold=fold)


def market_sources(fixture):
    result = {}
    for market, local in fixture.local.items():
        view = json.loads((local.views / "fold-000/manifest.json").read_text())["temporal_view"]
        result[market] = dict(
            protocol=local.protocol, macro=Path(view["macro_path"]), admission=local.admission
        )
    return result


def test_joint_producer_preserves_labels_and_publishes_consistent_local_views(tmp_path):
    import importlib

    try:
        module = importlib.import_module("mars_titan.training.joint_temporal_corpus")
    except ModuleNotFoundError:
        pytest.fail("Falta el productor de ventanas conjuntas")
    fixture = joint_fixture(tmp_path, fold=None)
    output = tmp_path / "joint-views"
    report = module.prepare_joint_temporal_corpus(fixture.manifest, market_sources(fixture), output)
    assert report["schema_version"] == 2
    assert len(report["folds"]) == 10
    for fold in report["folds"]:
        combined = CorpusDataset(output / fold["id"] / "manifest.json")
        assert set(combined.temporals) == {"US", "CN"}
        for market, local in fixture.local.items():
            projected = CorpusDataset(output / "markets" / market / fold["id"] / "manifest.json")
            original = CorpusDataset(local.views / fold["id"] / "manifest.json")
            assert projected.manifest["counts"] == original.manifest["counts"]
            for asset in projected.assets:
                assert (
                    combined._file(asset, "labels").read_bytes()
                    == original._file(asset, "labels").read_bytes()
                )
        assert combined.manifest["counts"] == fold["counts"]
        assert (
            sum(fold["market_counts"][market]["evaluation"] for market in ("US", "CN"))
            == fold["counts"]["evaluation"]
        )
    with pytest.raises(FileExistsError):
        module.prepare_joint_temporal_corpus(fixture.manifest, market_sources(fixture), output)


def test_joint_campaign_preflight_keeps_one_budget_and_both_admissions(tmp_path):
    from mars_titan.training.joint_temporal_corpus import prepare_joint_temporal_corpus
    from mars_titan.training.real_campaign import prepare_campaign

    fixture = joint_fixture(tmp_path, fold=None)
    views = tmp_path / "joint-views"
    prepare_joint_temporal_corpus(fixture.manifest, market_sources(fixture), views)
    plan = json.loads(Path("configs/baselines/convergence-temporal-search-us.json").read_text())
    plan["arms"] = ["US+CN"]
    config = tmp_path / "joint-search.json"
    atomic_json(config, plan)
    args = SimpleNamespace(
        views=views,
        encoded=fixture.local["US"].encoded,
        neural_config=config,
        tabular_config=Path("configs/baselines/tabular-convergence-us.json"),
        post_config=Path("configs/baselines/real-matched-posttraining.json"),
        references=tmp_path / "references",
        completion=tmp_path / "completion",
        state_dir=tmp_path / "state",
    )
    identity, stages = prepare_campaign(args)
    assert len(identity["admissions"]) == 2
    assert [row["planned"] for row in stages] == [400, 3380, 1890]
    assert set(identity["neural"]["protocols_sha256"]) == {"US", "CN"}


def test_joint_publication_failure_keeps_sources_and_allows_a_new_attempt(tmp_path, monkeypatch):
    from mars_titan.training import joint_temporal_corpus as module

    fixture = joint_fixture(tmp_path, fold=None)
    original = fixture.manifest.read_bytes()
    writer = module.atomic_json

    def fail_report(path, value):
        if value.get("kind") == "joint_temporal_views":
            raise OSError("corte técnico antes de publicar")
        return writer(path, value)

    output = tmp_path / "views-final"
    monkeypatch.setattr(module, "atomic_json", fail_report)
    with pytest.raises(OSError):
        module.prepare_joint_temporal_corpus(fixture.manifest, market_sources(fixture), output)
    assert not output.exists() and fixture.manifest.read_bytes() == original
    monkeypatch.setattr(module, "atomic_json", writer)
    module.prepare_joint_temporal_corpus(fixture.manifest, market_sources(fixture), output)
    assert CorpusDataset(output / "fold-000/manifest.json").manifest["counts"]["train"] == 12


def test_joint_resume_preserves_rows_and_revalidates_both_calendars(tmp_path):
    fixture = joint_fixture(tmp_path)
    dataset = CorpusDataset(fixture.manifest)
    complete = list(dataset.batches(partition="train", batch_size=5, epoch=1, seed=43))
    resumed = list(
        dataset.batches(
            partition="train",
            batch_size=5,
            epoch=1,
            seed=43,
            cursor=complete[1]["confirmed_cursor"],
        )
    )
    assert [key for batch in resumed for key in batch["sample_ids"]] == [
        key for batch in complete[2:] for key in batch["sample_ids"]
    ]
    path = fixture.local["CN"].admission
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="fuente|admisión"):
        list(dataset.batches(partition="train", batch_size=5, epoch=1, seed=43))


def test_joint_heldout_evaluates_every_local_key_with_no_calendar_intersection(tmp_path):
    import pyarrow.parquet as pq

    from mars_titan.posttraining.heldout import evaluate_partition
    from mars_titan.training.checkpoints import StopRequest

    fixture = joint_fixture(tmp_path, fold="fold-008", extra_days=("2023-10-02",))
    dataset = CorpusDataset(fixture.manifest)

    class Parent:
        def predict(self, inputs):
            return np.zeros(len(inputs["news"]), dtype=np.float64)

    for partition in ("calibration", "evaluation"):
        output = tmp_path / f"{partition}.parquet"
        scores = evaluate_partition(
            dataset, Parent(), partition, output, stop=StopRequest(), device="cpu", batch_size=3
        )
        records = pq.read_table(output).to_pylist()
        expected = rows(dataset, partition)
        assert {row["sample_id"] for row in records} == expected.keys()
        assert scores["prediction"]["samples"] == len(expected)
        assert set(scores["prediction"]["by_market_session"]) == {"US", "CN"}
        assert all(row["target"] == expected[row["sample_id"]][0] for row in records)


def rows(dataset, partition):
    result = {}
    for batch in dataset.batches(partition=partition, batch_size=3, epoch=0, seed=42):
        for index, key in enumerate(batch["sample_ids"]):
            assert key not in result
            result[key] = (
                batch["target"][index],
                {name: values[index] for name, values in batch["inputs"].items()},
            )
    return result


def test_joint_reader_preserves_the_disjoint_union_of_each_local_partition(tmp_path):
    fixture = joint_fixture(tmp_path)
    combined = CorpusDataset(fixture.manifest, cache_bytes=1024**2)
    assert combined.partitions == ("train", "validation", "calibration", "evaluation")
    for partition in combined.partitions:
        expected = {}
        for local in fixture.local.values():
            expected.update(
                rows(CorpusDataset(local.views / fixture.fold / "manifest.json"), partition)
            )
        actual = rows(combined, partition)
        assert actual.keys() == expected.keys()
        for key in actual:
            assert actual[key][0] == expected[key][0]
            for modality in actual[key][1]:
                np.testing.assert_array_equal(actual[key][1][modality], expected[key][1][modality])


def test_joint_reader_and_ordering_keep_exclusive_holidays_and_local_cutoffs(tmp_path):
    from mars_titan.environments.corpus_source import ParquetCohortSource, prepare_causal_corpus

    fixture = joint_fixture(
        tmp_path,
        fold="fold-008",
        extra_days=("2023-07-04", "2023-09-27", "2023-09-28", "2023-10-02"),
    )
    dataset = CorpusDataset(fixture.manifest)
    assigned = {}
    for partition in dataset.partitions:
        for batch in dataset.batches(partition=partition, batch_size=3, epoch=0, seed=42):
            for market, moment in zip(batch["market"], batch["prediction_at"], strict=True):
                assigned[(market, str(moment)[:10])] = partition
    assert assigned[("CN", "2023-07-04")] == "train"
    assert ("US", "2023-07-04") not in assigned
    assert assigned[("US", "2023-10-02")] == "calibration"
    assert ("CN", "2023-10-02") not in assigned
    assert assigned[("US", "2023-09-28")] == "validation"
    assert ("CN", "2023-09-28") not in assigned
    output = tmp_path / "ordered"
    prepare_causal_corpus(fixture.manifest, output)
    with ParquetCohortSource(output / "manifest.json", partition="validation") as source:
        observed = {}
        for index in range(len(source)):
            block = source(index)
            day = str(np.datetime64(block["prediction_at"], "us"))[:10]
            for asset in block["asset_ids"]:
                observed[(asset.split("/")[0], day)] = True
        assert ("US", "2023-09-28") in observed
        assert ("CN", "2023-09-28") not in observed


def test_single_and_joint_temporal_contracts_cannot_coexist(tmp_path):
    fixture = joint_fixture(tmp_path)
    fixture.metadata["temporal_view"] = fixture.metadata["temporal_views"]["US"]
    # La vista US basta para demostrar que el campo conjunto no puede ignorarse.
    fixture.metadata["assets"] = fixture.metadata["assets"][:1]
    fixture.metadata["counts"] = fixture.metadata["assets"][0]["counts"]
    atomic_json(fixture.manifest, fixture.metadata)
    with pytest.raises(ValueError):
        CorpusDataset(fixture.manifest)


@pytest.mark.parametrize(
    "fault",
    ["missing_market", "unknown_market", "different_gap", "different_fold", "different_seeds"],
)
def test_joint_contract_rejects_missing_or_misaligned_market_evidence(tmp_path, fault):
    fixture = joint_fixture(tmp_path)
    contracts = fixture.metadata["temporal_views"]
    if fault == "missing_market":
        contracts.pop("CN")
    elif fault == "unknown_market":
        contracts["other"] = contracts["CN"]
    elif fault == "different_gap":
        contracts["CN"]["protocol"]["gap_sessions"] = 2
    elif fault == "different_seeds":
        contracts["CN"]["protocol"]["seeds"] = [43]
    else:
        contracts["CN"]["fold"]["id"] = "fold-009"
    atomic_json(fixture.manifest, fixture.metadata)
    with pytest.raises(ValueError):
        CorpusDataset(fixture.manifest)


def test_joint_ordering_and_parent_cache_preserve_both_markets(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from mars_titan.environments.corpus_source import ParquetCohortSource, prepare_causal_corpus
    from mars_titan.training.predictive_inputs import PredictiveDataset
    from mars_titan.training.predictive_parents import prepare_parent_cache

    fixture = joint_fixture(tmp_path)
    dataset = CorpusDataset(fixture.manifest)
    ordered = tmp_path / "ordered"
    prepared = prepare_causal_corpus(fixture.manifest, ordered, batch_size=3)
    assert prepared["counts"] == dataset.manifest["counts"]
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "model.bin").write_bytes(b"recibo tecnico sin pesos cargables")
    predictions = {}
    expected = {}
    for partition in ("train", "validation"):
        records = []
        for batch in dataset.batches(partition=partition, batch_size=3, epoch=0, seed=42):
            for i, key in enumerate(batch["sample_ids"]):
                records.append(
                    dict(
                        sample_id=key,
                        asset_id="/".join(key.split("/")[:2]),
                        market=batch["market"][i],
                        prediction_at=batch["prediction_at"][i],
                        target=batch["target"][i],
                        prediction=batch["target"][i] + 0.02,
                    )
                )
        path = parent / f"{partition}.parquet"
        table = pa.Table.from_pylist(records)
        table = table.set_column(
            table.schema.get_field_index("prediction_at"),
            "prediction_at",
            table["prediction_at"].cast(pa.timestamp("us", tz="UTC")),
        )
        pq.write_table(table, path)
        predictions[partition] = dict(path=path.name, sha256=sha256(path))
        expected[partition] = {r["sample_id"]: r["target"] for r in records}
        with ParquetCohortSource(ordered / "manifest.json", partition=partition) as source:
            observed = [asset for i in range(len(source)) for asset in source(i)["asset_ids"]]
            assert len(observed) == len(records)
            assert {asset.split("/")[0] for asset in observed} == {"US", "CN"}
    atomic_json(
        parent / "run.json",
        dict(
            status="completed",
            final_test_opened=False,
            model="ridge",
            scope="full_corpus",
            cohort_complete=True,
            manifest_sha256=sha256(fixture.manifest),
            samples=prepared["counts"],
            checkpoint=dict(path="model.bin", sha256=sha256(parent / "model.bin")),
            predictions=predictions,
        ),
    )
    cache = tmp_path / "parent-cache"
    prepare_parent_cache(ordered / "manifest.json", parent / "run.json", cache)
    with PredictiveDataset(ordered / "manifest.json", cache / "manifest.json") as adapted:
        for partition in ("train", "validation"):
            observed = {}
            for batch in adapted.batches(partition=partition, batch_size=3, epoch=0, seed=42):
                for i, key in enumerate(batch["sample_ids"]):
                    assert batch["weight"][i] == 1
                    assert batch["parent"][i] == batch["target"][i] + 0.02
                    observed[key] = batch["target"][i]
            assert observed == expected[partition]
