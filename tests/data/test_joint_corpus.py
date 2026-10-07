"""Unión de dos mercados sin cambiar observaciones, unidades ni etiquetas."""

import importlib
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.chinese_samples import CNY_CONCEPTS
from mars_titan.data.cohort_news import COHORT_POLICIES
from mars_titan.data.currency_samples import COMMON_CONCEPTS
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.cohort_contract import representation_hash


def engine():
    try:
        return importlib.import_module("mars_titan.data.joint_corpus")
    except ModuleNotFoundError:
        pytest.fail("Falta la materialización del corpus conjunto")


def source(root, market, *, complete=True):
    """Crear una edición técnica de un activo con dos decisiones y 140 indicadores."""
    concepts = list(COMMON_CONCEPTS if market == "US" else CNY_CONCEPTS)
    size = len(concepts)
    moments = [datetime(2022, 6, 15, tzinfo=UTC), datetime(2023, 6, 15, tzinfo=UTC)]
    roots = {k: root / k for k in ("prepared", "samples", "labels")}
    paths = {
        k: roots["prepared" if k == "prices" else k] / market / "SAME" / f"{k}.parquet"
        for k in ("prices", "samples", "labels")
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            dict(
                open=[10.0] * 64,
                high=[11.0] * 64,
                low=[9.0] * 64,
                close=[10.5] * 64,
                volume=[100.0] * 64,
                available_at=[
                    datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(64)
                ],
            )
        ),
        paths["prices"],
    )
    numeric = np.array([list(range(1, size + 1)) + [1] * size + [0.5] * size] * 2, dtype=np.float32)
    table = pa.table(
        dict(
            cohort_id=["original_audited"] * 2,
            prediction_at=moments,
            price_end_index=[63, 63],
            news=pa.array([[1.0] * 384] * 2, type=pa.list_(pa.float32(), 384)),
            charts=pa.array([[2.0] * 512] * 2, type=pa.list_(pa.float32(), 512)),
            fundamentals=pa.FixedSizeListArray.from_arrays(pa.array(numeric.ravel()), 3 * size),
            macro=pa.array(
                [[3.0] * 140 + [1.0] * 140 + [0.5] * 140] * 2, type=pa.list_(pa.float32(), 420)
            ),
            input_availability=[
                dict.fromkeys(("prices", "news", "charts", "fundamentals", "macro"), t)
                for t in moments
            ],
        )
    )
    pq.write_table(table, paths["samples"], row_group_size=1)
    pq.write_table(
        pa.table(
            dict(
                cohort_id=["original_audited"] * 2,
                sample_row=[0, 1],
                prediction_at=moments,
                target_available_at=[t + timedelta(days=1) for t in moments],
                target=[0.01, -0.02],
                partition=["train", "validation"],
                reason=["accepted", "accepted"],
            )
        ),
        paths["labels"],
    )
    representation = dict(
        fundamental_concepts=concepts,
        macro_indicators=[f"macro{i:03}" for i in range(140)],
        encoders={"kind": "generated_test_vectors"},
        representation_code={"fixture": "a" * 64},
        text_aggregation="float64_sum_float32_mean_all_admitted_events",
        context_sessions=64,
        news_lookback_sessions=5,
    )
    asset = dict(
        market=market,
        symbol="SAME",
        samples=2,
        counts=dict(train=1, validation=1),
        cohort_id="original_audited",
        representation_sha256=representation_hash(representation),
        **{k + "_sha256": sha256(path) for k, path in paths.items()},
    )
    meta = dict(
        schema_version=2,
        kind="corpus_supervision",
        technical_fixture=True,
        cohort_id="original_audited",
        news_content_policy=COHORT_POLICIES["original_audited"],
        context_sessions=64,
        roots={k: str(p) for k, p in roots.items()},
        assets=[asset],
        counts=asset["counts"],
        scope="full_corpus" if complete else "development_snapshot",
        cohort_complete=complete,
        coverage=[dict(market=market, symbol="SAME", state="encoded", samples=2)],
        candidate_count=1,
        samples=2,
        failed_assets=0,
        markets=[market],
        representation=representation,
        final_test_opened=False,
        configuration={},
        market_factors={market: {"fixture": True}},
    )
    path = root / "manifest.json"
    atomic_json(path, meta)
    return path, paths, numeric


def pair(tmp_path):
    return {m: source(tmp_path / m, m, complete=m == "US") for m in ("US", "CN")}


def test_union_preserves_all_observations_and_separates_native_accounting_channels(tmp_path):
    from mars_titan.training.corpus_inputs import CorpusDataset

    parents = pair(tmp_path)
    output = tmp_path / "joint"
    report = engine().prepare_joint_corpus({m: p[0] for m, p in parents.items()}, output)
    assert report["samples"] == 4 and report["assets"] == 2
    assert report["cohort_complete"] is False and report["scope"] == "development_snapshot"
    manifest = output / "supervised/manifest.json"
    dataset = CorpusDataset(manifest)
    assert dataset.manifest["counts"] == dict(train=2, validation=2)
    assert dataset.manifest["representation"]["fundamental_concepts"] == [
        *COMMON_CONCEPTS,
        *CNY_CONCEPTS,
    ]
    for market, (_, paths, numeric) in parents.items():
        destination = output / "encoded/samples" / market / "SAME/samples.parquet"
        original = pq.read_table(paths["samples"])
        result = pq.read_table(destination)
        assert result.drop(["fundamentals"]).equals(original.drop(["fundamentals"]))
        width, offset = (23, 0) if market == "US" else (3, 23)
        actual = np.asarray(result["fundamentals"].to_pylist(), dtype=np.float32)
        expected = np.zeros((2, 78), dtype=np.float32)
        for block in range(3):
            expected[:, block * 26 + offset : block * 26 + offset + width] = numeric[
                :, block * width : (block + 1) * width
            ]
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        for kind, prefix in (("prices", "prepared"), ("labels", "supervised/labels")):
            copied = output / prefix / market / "SAME" / (kind + ".parquet")
            assert sha256(copied) == sha256(paths[kind])
            assert copied.stat().st_ino != paths[kind].stat().st_ino
    for partition in ("train", "validation"):
        rows = list(dataset.batches(partition=partition, batch_size=4, epoch=0, seed=42))
        assert sum(len(b["target"]) for b in rows) == 2
        assert all(b["inputs"]["fundamentals"].shape[1] == 78 for b in rows)


@pytest.mark.parametrize("fault", ["encoder", "macro", "taxonomy", "open_test", "missing_market"])
def test_incompatible_sources_fail_before_publishing_an_edition(tmp_path, fault):
    parents = pair(tmp_path)
    path = parents["CN"][0]
    meta = json.loads(path.read_text())
    if fault == "encoder":
        meta["representation"]["encoders"] = {"kind": "different"}
    elif fault == "macro":
        meta["representation"]["macro_indicators"][0] = "different"
    elif fault == "taxonomy":
        meta["representation"]["fundamental_concepts"][2] = "us-gaap:StockholdersEquity:CNY"
    elif fault == "open_test":
        meta["final_test_opened"] = True
    else:
        meta["markets"] = ["US"]
    meta["assets"][0]["representation_sha256"] = representation_hash(meta["representation"])
    atomic_json(path, meta)
    output = tmp_path / "joint"
    with pytest.raises(ValueError):
        engine().prepare_joint_corpus({m: p[0] for m, p in parents.items()}, output)
    assert not (output / "report.json").exists()


def test_completed_edition_reuses_bytes_and_rejects_changed_artifacts(tmp_path):
    parents = pair(tmp_path)
    sources = {m: p[0] for m, p in parents.items()}
    output = tmp_path / "joint"
    first = engine().prepare_joint_corpus(sources, output)
    path = output / "encoded/samples/CN/SAME/samples.parquet"
    before = sha256(path), path.stat().st_mtime_ns
    assert engine().prepare_joint_corpus(dict(reversed(list(sources.items()))), output) == first
    assert (sha256(path), path.stat().st_mtime_ns) == before
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError):
        engine().prepare_joint_corpus(sources, output)


@pytest.mark.parametrize(
    "field,value", [("samples", 100), ("status", "paused"), ("final_test_opened", True)]
)
def test_completed_receipt_cannot_change_population_or_completion(tmp_path, field, value):
    parents = pair(tmp_path)
    sources = {m: p[0] for m, p in parents.items()}
    output = tmp_path / "joint"
    engine().prepare_joint_corpus(sources, output)
    path = output / "report.json"
    report = json.loads(path.read_text())
    report[field] = value
    atomic_json(path, report)
    with pytest.raises(ValueError):
        engine().prepare_joint_corpus(sources, output)


def test_recovery_keeps_confirmed_assets_and_rejects_a_changed_asset_receipt(tmp_path, monkeypatch):
    parents = pair(tmp_path)
    sources = {m: p[0] for m, p in parents.items()}
    output = tmp_path / "joint"
    api = engine()
    original = api._project
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(str(args[0]))
        if len(calls) == 2:
            raise InterruptedError("Corte técnico entre activos")
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "_project", interrupted)
    with pytest.raises(InterruptedError):
        api.prepare_joint_corpus(sources, output)
    confirmed = output / "encoded/samples/US/SAME/samples.parquet"
    before = sha256(confirmed), confirmed.stat().st_mtime_ns
    marker = confirmed.with_name("projection.json")
    saved = marker.read_bytes()
    changed = json.loads(saved)
    changed["asset"]["samples"] = 100
    atomic_json(marker, changed)
    with pytest.raises(ValueError):
        api.prepare_joint_corpus(sources, output)
    assert not (output / "encoded/samples/CN/SAME/samples.parquet").exists()
    marker.write_bytes(saved)
    monkeypatch.setattr(api, "_project", original)
    assert api.prepare_joint_corpus(sources, output)["samples"] == 4
    assert (sha256(confirmed), confirmed.stat().st_mtime_ns) == before


def test_missing_projection_digest_cannot_turn_validation_into_an_unchecked_read(
    tmp_path, monkeypatch
):
    parents = pair(tmp_path)
    sources = {m: p[0] for m, p in parents.items()}
    output = tmp_path / "joint"
    api = engine()
    original = api._project

    def interrupted(source_path, *args, **kwargs):
        if source_path == parents["CN"][1]["samples"]:
            raise InterruptedError("Corte antes de la segunda proyección")
        return original(source_path, *args, **kwargs)

    monkeypatch.setattr(api, "_project", interrupted)
    with pytest.raises(InterruptedError):
        api.prepare_joint_corpus(sources, output)
    marker = output / "encoded/samples/US/SAME/projection.json"
    record = json.loads(marker.read_text())
    record["asset"]["samples_sha256"] = None
    atomic_json(marker, record)
    with pytest.raises(ValueError):
        api.prepare_joint_corpus(sources, output)


def test_input_budget_is_checked_before_hashing_or_creating_output(tmp_path, monkeypatch):
    parents = pair(tmp_path)
    api = engine()
    monkeypatch.setattr(api, "_LIMIT", 1)
    monkeypatch.setattr(
        api,
        "CorpusDataset",
        lambda *_args, **_kwargs: pytest.fail("Se abrió el lector antes de comprobar el tamaño"),
    )
    output = tmp_path / "joint"
    with pytest.raises(ValueError, match="64 MiB"):
        api.prepare_joint_corpus({m: p[0] for m, p in parents.items()}, output)
    assert not output.exists()
