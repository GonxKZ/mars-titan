import importlib
import json
import shutil
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from tests.data.test_budget_targets import fixture_prices


def function():
    module = importlib.import_module("mars_titan.training.corpus_inputs")
    result = getattr(module, "prepare_corpus_targets", None)
    assert result is not None, "Falta la preparación de etiquetas por activo"
    return result


def materialized(tmp_path):
    clock, stock, market = fixture_prices()
    stock.loc[131, "close"] += 2.0
    prepared, samples = tmp_path / "prepared", tmp_path / "samples"
    folder, encoded = prepared / "US/AAA", samples / "US/AAA"
    folder.mkdir(parents=True)
    encoded.mkdir(parents=True)
    for frame in (stock, market):
        frame["high"], frame["low"], frame["volume"] = 110.0, 90.0, 100.0
    pq.write_table(pa.Table.from_pandas(stock, preserve_index=False), folder / "prices.parquet")
    factor = tmp_path / "market.parquet"
    pq.write_table(pa.Table.from_pandas(market, preserve_index=False), factor)
    dates = [
        clock.decisions[130],
        clock.decision("2022-12-30"),
        clock.decision("2023-01-03"),
        clock.decisions[-1],
        datetime(2024, 1, 2, tzinfo=UTC),
    ]
    indexes = {t: i for i, t in enumerate(clock.decisions)}
    records = [
        {
            "prediction_at": t,
            "price_end_index": indexes.get(t, len(stock) - 1),
            "news": [1.0, 2.0],
            "charts": [3.0],
            "fundamentals": [4.0],
            "macro": [5.0],
        }
        for t in dates
    ]
    pq.write_table(pa.Table.from_pylist(records), encoded / "samples.parquet")
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "market": "US",
                "symbol": "AAA",
                "fingerprint": "fixture",
                "news_content_policy": "verified_full_articles",
                "artifacts": {"prices.parquet": sha256(folder / "prices.parquet")},
            }
        )
    )
    (encoded / "manifest.json").write_text(
        json.dumps(
            {
                "market": "US",
                "symbol": "AAA",
                "prepared_fingerprint": "fixture",
                "news_content_policy": "verified_full_articles",
                "context_sessions": 64,
                "samples_sha256": sha256(encoded / "samples.parquet"),
            }
        )
    )
    manifest = tmp_path / "materialized.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "materialized_corpus",
                "scope": "development_snapshot",
                "cohort_complete": False,
                "context_sessions": 64,
                "samples_root": str(samples),
                "calendar_start": {"US": "2021-01-01"},
                "assets": [{"market": "US", "symbol": "AAA"}],
                "market_factors": {
                    "US": {
                        "market": "US",
                        "symbol": "SPY",
                        "prices_path": str(factor),
                        "prices_sha256": sha256(factor),
                    }
                },
            }
        )
    )
    return manifest, prepared


def test_targets_match_known_residual_and_account_for_every_sample(tmp_path):
    manifest, prepared = materialized(tmp_path)
    output = tmp_path / "supervised"
    result = function()(manifest, prepared, output)
    assert result["counts"] == {"train": 1, "validation": 1}
    labels = pq.read_table(output / "labels/US/AAA/labels.parquet").to_pylist()
    assert len(labels) == 5
    assert labels[0]["target"] == pytest.approx(0.02, abs=1e-14)
    assert labels[1]["reason"] == "target_crosses_partition_boundary"
    assert labels[-1]["reason"] == "final_test_reserved"
    assert labels[-1]["target"] is None
    assert result["scope"] == "development_snapshot" and result["cohort_complete"] is False
    module = importlib.import_module("mars_titan.training.corpus_inputs")
    batches = list(
        module.supervised_batches(
            output / "manifest.json", partition="train", batch_size=8, epoch=0, seed=42
        )
    )
    assert len(batches) == 1 and len(batches[0]["target"]) == 1


def test_existing_targets_are_reused_but_changes_fail_closed(tmp_path):
    manifest, prepared = materialized(tmp_path)
    output = tmp_path / "supervised"
    function()(manifest, prepared, output)
    path = output / "labels/US/AAA/labels.parquet"
    before = path.stat().st_mtime_ns
    assert function()(manifest, prepared, output)["reused_assets"] == 1
    assert path.stat().st_mtime_ns == before
    data = json.loads(manifest.read_text())
    data["context_sessions"] = 32
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="configuración|edición"):
        function()(manifest, prepared, output)


def test_reusing_labels_preserves_manifest_identity_and_confirmed_cursor(tmp_path):
    manifest, prepared = materialized(tmp_path)
    output = tmp_path / "supervised"
    function()(manifest, prepared, output)
    path = output / "manifest.json"
    dataset = importlib.import_module("mars_titan.training.corpus_inputs").CorpusDataset(path)
    options = dict(partition="train", batch_size=1, epoch=0, seed=42)
    cursor = next(dataset.batches(**options))["confirmed_cursor"]
    digest = sha256(path)
    function()(manifest, prepared, output)
    assert sha256(path) == digest
    assert list(dataset.batches(**options, cursor=cursor)) == []


@pytest.mark.parametrize("package", ["numpy", "pandas", "exchange_calendars"])
def test_changed_label_dependency_cannot_reuse_prior_assets(tmp_path, monkeypatch, package):
    manifest, prepared = materialized(tmp_path)
    output = tmp_path / "supervised"
    function()(manifest, prepared, output)
    dependency = importlib.import_module(package)
    monkeypatch.setattr(dependency, "__version__", "changed-environment")
    with pytest.raises(ValueError, match="configuración|edición"):
        function()(manifest, prepared, output)


def test_changed_array_implementation_cannot_reuse_labels(tmp_path, monkeypatch):
    manifest, prepared = materialized(tmp_path)
    output = tmp_path / "supervised"
    function()(manifest, prepared, output)
    target_module = importlib.import_module("mars_titan.training.corpus_targets")
    original = target_module.sha256

    def changed(path):
        return "changed-array-code" if path.name == "residual_arrays.py" else original(path)

    monkeypatch.setattr(target_module, "sha256", changed)
    with pytest.raises(ValueError, match="configuración|edición"):
        function()(manifest, prepared, output)


def test_reference_and_numpy_backends_produce_identical_label_artifacts(tmp_path):
    manifest, prepared = materialized(tmp_path)
    module = importlib.import_module("mars_titan.training.corpus_targets")
    hashes = []
    for backend in ("reference", "numpy"):
        output = tmp_path / backend
        result = module.prepare_corpus_targets(manifest, prepared, output, backend=backend)
        assert result["configuration"]["backend"] == backend
        hashes.append(result["assets"][0]["labels_sha256"])
    assert hashes[0] == hashes[1]
    with pytest.raises(ValueError, match="configuración|edición"):
        module.prepare_corpus_targets(manifest, prepared, tmp_path / "numpy", backend="reference")


def test_unknown_target_backend_does_not_create_output(tmp_path):
    manifest, prepared = materialized(tmp_path)
    module = importlib.import_module("mars_titan.training.corpus_targets")
    output = tmp_path / "unknown"
    with pytest.raises(ValueError, match="motor"):
        module.prepare_corpus_targets(manifest, prepared, output, backend="auto")
    assert not output.exists()


def test_factor_from_another_market_is_rejected_before_writing(tmp_path):
    manifest, prepared = materialized(tmp_path)
    data = json.loads(manifest.read_text())
    data["market_factors"]["US"]["market"] = "CN"
    manifest.write_text(json.dumps(data))
    output = tmp_path / "supervised"
    with pytest.raises(ValueError, match="mercado"):
        function()(manifest, prepared, output)
    assert not output.exists()


def test_unreviewed_news_cannot_become_supervised_research_data(tmp_path):
    manifest, prepared = materialized(tmp_path)
    path = prepared / "US/AAA/manifest.json"
    metadata = json.loads(path.read_text())
    metadata["news_content_policy"] = "not_reviewed"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="noticias"):
        function()(manifest, prepared, tmp_path / "supervised")


def test_partial_label_write_is_recovered_before_publishing_manifest(tmp_path, monkeypatch):
    manifest, prepared = materialized(tmp_path)
    output = tmp_path / "supervised"
    function()
    target_module = importlib.import_module("mars_titan.training.corpus_targets")
    original = target_module.atomic_json

    def interrupt(path, value):
        if path.name == "receipt.json":
            raise KeyboardInterrupt
        original(path, value)

    monkeypatch.setattr(target_module, "atomic_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        function()(manifest, prepared, output)
    assert not (output / "manifest.json").exists()
    monkeypatch.setattr(target_module, "atomic_json", original)
    assert function()(manifest, prepared, output)["counts"] == {"train": 1, "validation": 1}


def test_unrelated_existing_output_is_not_adopted_or_overwritten(tmp_path):
    manifest, prepared = materialized(tmp_path)
    output = tmp_path / "supervised"
    output.mkdir()
    original = output / "notes.txt"
    original.write_text("Conservar")
    with pytest.raises(ValueError, match="directorio"):
        function()(manifest, prepared, output)
    assert original.read_text() == "Conservar"
    assert not (output / "configuration.json").exists()


def audited_edition(tmp_path):
    manifest, prepared = materialized(tmp_path)
    meta = json.loads(manifest.read_text())
    cohort = "original_audited"
    policy = "source_audited_not_external"
    meta.update(schema_version=2, cohort_id=cohort, news_content_policy=policy)
    meta.update(
        candidate_count=1,
        samples=5,
        failed_assets=0,
        coverage=[dict(market="US", symbol="AAA", state="encoded", samples=5)],
    )
    meta["assets"][0]["cohort_id"] = cohort
    folder = tmp_path / "samples/US/AAA"
    sample_path = folder / "samples.parquet"
    table = pq.read_table(sample_path)
    pq.write_table(table.append_column("cohort_id", pa.array([cohort] * len(table))), sample_path)
    for path in (prepared / "US/AAA/manifest.json", folder / "manifest.json"):
        item = json.loads(path.read_text())
        item.update(cohort_id=cohort, news_content_policy=policy)
        if path.parent == folder:
            item["samples_sha256"] = sha256(sample_path)
            item.update(
                fundamental_concepts=["Assets"],
                macro_indicators=["inflation"],
                encoders=dict(version=1),
                representation_code={"fixture": "0" * 64},
                text_aggregation="test_mean",
                news_lookback_sessions=5,
            )
        path.write_text(json.dumps(item))
    manifest.write_text(json.dumps(meta))
    return manifest, prepared


def test_audited_cohort_survives_labels_and_batches_without_external_claim(tmp_path):
    manifest, prepared = audited_edition(tmp_path)
    output = tmp_path / "supervised"
    result = function()(manifest, prepared, output)
    assert result["schema_version"] == 2
    assert result["cohort_id"] == "original_audited"
    assert result["news_content_policy"] == "source_audited_not_external"
    assert all(a["cohort_id"] == result["cohort_id"] for a in result["assets"])
    labels = pq.read_table(output / "labels/US/AAA/labels.parquet")
    assert labels["cohort_id"].to_pylist() == ["original_audited"] * 5
    dataset = importlib.import_module("mars_titan.training.corpus_inputs").CorpusDataset(
        output / "manifest.json"
    )
    batch = next(dataset.batches(partition="train", batch_size=8, epoch=0, seed=42))
    assert batch["cohort_id"] == "original_audited"
    assert batch["target"].tolist() == pytest.approx([0.02], abs=1e-8)


@pytest.mark.parametrize("where", ["manifest", "asset", "prepared", "encoded", "sample"])
def test_mixed_cohort_identity_cannot_enter_supervision(tmp_path, where):
    manifest, prepared = audited_edition(tmp_path)
    meta = json.loads(manifest.read_text())
    if where in {"manifest", "asset"}:
        target = meta if where == "manifest" else meta["assets"][0]
        target["cohort_id"] = "externally_verified"
        manifest.write_text(json.dumps(meta))
    elif where in {"prepared", "encoded"}:
        path = (prepared if where == "prepared" else tmp_path / "samples") / "US/AAA/manifest.json"
        item = json.loads(path.read_text())
        item["cohort_id"] = "externally_verified"
        path.write_text(json.dumps(item))
    else:
        path = tmp_path / "samples/US/AAA/samples.parquet"
        table = pq.read_table(path)
        rows = table.to_pylist()
        rows[0]["cohort_id"] = "externally_verified"
        pq.write_table(pa.Table.from_pylist(rows), path)
        receipt = path.parent / "manifest.json"
        item = json.loads(receipt.read_text())
        item["samples_sha256"] = sha256(path)
        receipt.write_text(json.dumps(item))
    output = tmp_path / "supervised"
    with pytest.raises(ValueError, match="cohorte|noticias"):
        function()(manifest, prepared, output)
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("artifact", ["samples", "labels"])
def test_reader_rejects_rehashed_rows_from_another_cohort(tmp_path, artifact):
    manifest, prepared = audited_edition(tmp_path)
    output = tmp_path / "supervised"
    function()(manifest, prepared, output)
    path = (
        tmp_path / "samples" if artifact == "samples" else output / "labels"
    ) / f"US/AAA/{artifact}.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["cohort_id"] = "externally_verified"
    pq.write_table(pa.Table.from_pylist(rows), path)
    meta_path = output / "manifest.json"
    meta = json.loads(meta_path.read_text())
    meta["assets"][0][artifact + "_sha256"] = sha256(path)
    meta_path.write_text(json.dumps(meta))
    dataset = importlib.import_module("mars_titan.training.corpus_inputs").CorpusDataset(meta_path)
    with pytest.raises(ValueError, match="cohorte"):
        list(dataset.batches(partition="train", batch_size=8, epoch=0, seed=42))


def test_partial_asset_list_cannot_claim_complete_encoded_coverage(tmp_path):
    manifest, prepared = audited_edition(tmp_path)
    meta = json.loads(manifest.read_text())
    meta.update(
        scope="full_corpus",
        cohort_complete=True,
        candidate_count=2,
        samples=10,
        failed_assets=0,
        coverage=[
            dict(market="US", symbol="AAA", state="encoded", samples=5),
            dict(market="US", symbol="BBB", state="encoded", samples=5),
        ],
    )
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="cobertura|población|candidatos"):
        function()(manifest, prepared, tmp_path / "supervised")
    assert not (tmp_path / "supervised/manifest.json").exists()


@pytest.mark.parametrize(
    "change", [dict(cohort_id="externally_verified"), dict(counts={"train": 99, "validation": 1})]
)
def test_reused_label_receipt_cannot_relabel_cohort(tmp_path, change):
    manifest, prepared = audited_edition(tmp_path)
    output = tmp_path / "supervised"
    function()(manifest, prepared, output)
    before = sha256(output / "manifest.json")
    receipt = output / "labels/US/AAA/receipt.json"
    receipt.write_text(json.dumps({**json.loads(receipt.read_text()), **change}))
    with pytest.raises(ValueError, match="cohorte|recibo"):
        function()(manifest, prepared, output)
    assert sha256(output / "manifest.json") == before


@pytest.mark.parametrize("field", ["fundamental_concepts", "macro_indicators", "encoders"])
def test_equal_width_different_feature_meanings_cannot_share_corpus(tmp_path, field):
    manifest, prepared = audited_edition(tmp_path)
    for root in (prepared, tmp_path / "samples"):
        shutil.copytree(root / "US/AAA", root / "US/BBB")
        path = root / "US/BBB/manifest.json"
        item = json.loads(path.read_text())
        item["symbol"] = "BBB"
        if root.name == "samples":
            item.update(
                fundamental_concepts=["Assets"],
                macro_indicators=["inflation"],
                encoders=dict(version=1),
            )
            item[field] = dict(version=2) if field == "encoders" else ["other_feature"]
        path.write_text(json.dumps(item))
    path = tmp_path / "samples/US/AAA/manifest.json"
    item = json.loads(path.read_text())
    item.update(
        fundamental_concepts=["Assets"], macro_indicators=["inflation"], encoders=dict(version=1)
    )
    path.write_text(json.dumps(item))
    meta = json.loads(manifest.read_text())
    meta["assets"].append(dict(market="US", symbol="BBB", cohort_id="original_audited"))
    meta.update(
        candidate_count=2,
        samples=10,
        failed_assets=0,
        coverage=[dict(market="US", symbol=s, state="encoded", samples=5) for s in ("AAA", "BBB")],
    )
    manifest.write_text(json.dumps(meta))
    with pytest.raises(
        ValueError, match="representaci|semántica|codificador|conceptos|indicadores"
    ):
        function()(manifest, prepared, tmp_path / "supervised")
    assert not (tmp_path / "supervised/manifest.json").exists()
