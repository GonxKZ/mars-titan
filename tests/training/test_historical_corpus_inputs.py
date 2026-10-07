"""Lectura y etiquetas de fixtures históricos con ausencias, sin modelos."""

import hashlib
import json
from datetime import timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES, policy_identity
from mars_titan.data.storage import sha256
from mars_titan.training.cohort_contract import cohort_identity, representation_hash
from mars_titan.training.corpus_inputs import CorpusDataset, supervised_batches
from mars_titan.training.corpus_targets import prepare_corpus_targets
from tests.training.test_corpus_targets import audited_edition


def dump(path, value):
    path.write_text(json.dumps(value))


def historical_edition(tmp_path):
    manifest, prepared = audited_edition(tmp_path)
    meta = json.loads(manifest.read_text())
    identity = policy_identity(HISTORICAL_MASKED)
    meta.update(
        identity,
        schema_version=3,
        samples=4,
        candidate_count=2,
        training_ready=False,
        final_test_opened=False,
    )
    meta["coverage"][0]["samples"] = 4
    meta["coverage"].append(dict(market="US", symbol="NO_PRICES", state="missing_required_prices"))
    source = tmp_path / "samples/US/AAA"
    path = source / "samples.parquet"
    rows = pq.read_table(path).to_pylist()[:4]
    for row in rows:
        row.update(
            news=[0.0, 0.0],
            fundamentals=[0.0, 0.0, 0.0],
            macro=[0.0, 0.0, 0.0],
            presence=[True, False, True, False, False],
            news_count=0,
            input_availability={
                name: row["prediction_at"] if name in {"prices", "charts"} else None
                for name in MODALITIES
            },
        )
    stamp = pa.timestamp("us", tz="UTC")
    table = pa.Table.from_pylist(rows)
    table = table.set_column(
        table.schema.get_field_index("input_availability"),
        "input_availability",
        pa.array(
            [row["input_availability"] for row in rows],
            type=pa.struct([(name, stamp) for name in MODALITIES]),
        ),
    )
    pq.write_table(table, path, row_group_size=2)
    for receipt in (prepared / "US/AAA/manifest.json", source / "manifest.json"):
        item = json.loads(receipt.read_text())
        item.update(identity, schema_version=4)
        if receipt.parent == source:
            item["samples_sha256"] = sha256(path)
        dump(receipt, item)
    dump(manifest, meta)
    return manifest, prepared


def supervised(tmp_path):
    manifest, prepared = historical_edition(tmp_path)
    output = tmp_path / "supervised"
    prepare_corpus_targets(manifest, prepared, output, input_policy=HISTORICAL_MASKED)
    return output / "manifest.json"


def read_all(path, **kwargs):
    return list(
        supervised_batches(
            path,
            partition="train",
            batch_size=1,
            epoch=0,
            seed=42,
            input_policy=HISTORICAL_MASKED,
            **kwargs,
        )
    )


def change_samples(tmp_path, meta_path, change):
    path = tmp_path / "samples/US/AAA/samples.parquet"
    table = pq.read_table(path)
    changed = change(table)
    pq.write_table(changed, path, row_group_size=2)
    meta = json.loads(meta_path.read_text())
    meta["assets"][0]["samples_sha256"] = sha256(path)
    dump(meta_path, meta)


def change_rows(table, change):
    rows = table.to_pylist()
    for row in rows:
        change(row)
    return pa.Table.from_pylist(rows, schema=table.schema)


def test_historical_supervision_requires_opt_in_before_creating_output(tmp_path):
    manifest, prepared = historical_edition(tmp_path)
    output = tmp_path / "strict"
    with pytest.raises(ValueError):
        prepare_corpus_targets(manifest, prepared, output)
    assert not output.exists()


def test_historical_absences_keep_labels_ids_and_cursor(tmp_path):
    path = supervised(tmp_path)
    meta = json.loads(path.read_text())
    assert meta["schema_version"] == 3
    assert meta["input_policy"] == HISTORICAL_MASKED
    assert meta["mask_contract"] == policy_identity(HISTORICAL_MASKED)["mask_contract"]
    assert meta["training_ready"] is False
    assert meta["samples"] == 4
    assert meta["candidate_count"] == 2
    assert meta["coverage"][-1]["state"] == "missing_required_prices"
    with pytest.raises(ValueError):
        CorpusDataset(path)
    batches = read_all(path)
    assert len(batches) == 1
    batch = batches[0]
    assert batch["sample_ids"]
    assert batch["presence"].dtype == np.bool_
    assert batch["presence"].tolist() == [[True, False, True, False, False]]
    assert "presence" not in batch["inputs"]
    assert batch["target"].tolist() == pytest.approx([0.02], abs=1e-14)
    assert np.all(batch["input_available_at"] <= batch["prediction_at"])
    assert read_all(path, cursor=batch["confirmed_cursor"]) == []
    labels = pq.read_table(tmp_path / "supervised/labels/US/AAA/labels.parquet").to_pylist()
    assert [row["sample_row"] for row in labels] == list(range(4))
    assert labels[1]["reason"] == "target_crosses_partition_boundary"
    assert labels[-1]["reason"] == "target_after_cutoff"


def test_masked_labels_equal_the_strict_reference_for_common_samples(tmp_path):
    strict_root, masked_root = tmp_path / "strict", tmp_path / "masked"
    strict_root.mkdir()
    masked_root.mkdir()
    manifest, prepared = audited_edition(strict_root)
    prepare_corpus_targets(manifest, prepared, strict_root / "supervised")
    supervised(masked_root)
    strict = pq.read_table(strict_root / "supervised/labels/US/AAA/labels.parquet").slice(0, 4)
    masked = pq.read_table(masked_root / "supervised/labels/US/AAA/labels.parquet")
    assert strict.equals(masked)


def test_observed_zero_keeps_presence_in_numeric_blocks(tmp_path):
    path = supervised(tmp_path)

    def observe(row):
        row["presence"][3:] = [True, True]
        for name in ("fundamentals", "macro"):
            row[name] = [0.0, 1.0, 0.0]
            row["input_availability"][name] = row["prediction_at"]

    change_samples(tmp_path, path, lambda table: change_rows(table, observe))
    batch = read_all(path)[0]
    assert batch["presence"].tolist() == [[True, False, True, True, True]]
    assert batch["inputs"]["fundamentals"][0].tolist() == [0.0, 1.0, 0.0]


@pytest.mark.parametrize(
    "change",
    [
        lambda row: row["news"].__setitem__(0, 1.0),
        lambda row: row["fundamentals"].__setitem__(0, 1.0),
        lambda row: row["macro"].__setitem__(1, 1.0),
        lambda row: row["macro"].__setitem__(1, 0.5),
        lambda row: row["fundamentals"].__setitem__(2, 1.0),
        lambda row: row["news"].__setitem__(0, float("nan")),
        lambda row: row["macro"].__setitem__(0, float("inf")),
        lambda row: row["presence"].__setitem__(0, False),
        lambda row: row["presence"].__setitem__(2, False),
        lambda row: row["input_availability"].__setitem__("news", row["prediction_at"]),
        lambda row: row["input_availability"].__setitem__("prices", None),
        lambda row: row["input_availability"].__setitem__(
            "charts", row["prediction_at"] + timedelta(seconds=1)
        ),
    ],
)
def test_inconsistent_absence_or_time_is_rejected(tmp_path, change):
    path = supervised(tmp_path)
    change_samples(tmp_path, path, lambda table: change_rows(table, change))
    with pytest.raises(ValueError):
        read_all(path)


@pytest.mark.parametrize("column", ["presence", "input_availability", "news_count"])
def test_historical_metadata_cannot_be_omitted(tmp_path, column):
    path = supervised(tmp_path)
    change_samples(tmp_path, path, lambda table: table.drop_columns(column))
    with pytest.raises(ValueError):
        read_all(path)


def test_presence_requires_boolean_elements(tmp_path):
    path = supervised(tmp_path)
    change_samples(
        tmp_path,
        path,
        lambda table: table.set_column(
            table.schema.get_field_index("presence"),
            "presence",
            pa.array([[1, 0, 1, 0, 0]] * len(table)),
        ),
    )
    with pytest.raises(ValueError):
        read_all(path)


def test_masked_vectors_match_the_declared_catalog_lengths(tmp_path):
    path = supervised(tmp_path)
    meta = json.loads(path.read_text())
    meta["representation"]["macro_indicators"].append("another")
    meta["assets"][0]["representation_sha256"] = representation_hash(
        meta["representation"], input_policy=HISTORICAL_MASKED
    )
    dump(path, meta)
    with pytest.raises(ValueError):
        read_all(path)


@pytest.mark.parametrize(
    "change",
    [
        lambda meta: meta["mask_contract"].update(version=True),
        lambda meta: meta["mask_contract"]["order"].reverse(),
        lambda meta: meta.pop("mask_contract"),
        lambda meta: meta.update(input_policy="other"),
        lambda meta: meta.update(schema_version=True),
        lambda meta: meta["coverage"][-1].update(state="missing_modalities"),
    ],
)
def test_historical_contract_is_exact(tmp_path, change):
    manifest, _ = historical_edition(tmp_path)
    meta = json.loads(manifest.read_text())
    change(meta)
    with pytest.raises(ValueError):
        cohort_identity(meta, input_policy=HISTORICAL_MASKED)


def test_historical_identity_contains_policy_without_changing_strict_hash(tmp_path):
    manifest, _ = historical_edition(tmp_path)
    receipt = json.loads((tmp_path / "samples/US/AAA/manifest.json").read_text())
    strict_fields = (
        "fundamental_concepts",
        "macro_indicators",
        "encoders",
        "representation_code",
        "text_aggregation",
        "context_sessions",
        "news_lookback_sessions",
    )
    strict = {name: receipt[name] for name in strict_fields}
    expected = hashlib.sha256(json.dumps(strict, sort_keys=True).encode()).hexdigest()
    assert representation_hash(strict) == expected
    historical = representation_hash(receipt, input_policy=HISTORICAL_MASKED)
    assert historical != expected
    with pytest.raises(ValueError):
        representation_hash(receipt)
    assert (
        cohort_identity(json.loads(manifest.read_text()), input_policy=HISTORICAL_MASKED)
        == "original_audited"
    )


def test_resume_keeps_all_historical_ids_and_presence_across_groups(tmp_path):
    manifest, prepared = historical_edition(tmp_path)
    samples = tmp_path / "samples/US/AAA/samples.parquet"
    table = pq.read_table(samples)
    template = table.to_pylist()[0]
    prices = pq.read_table(prepared / "US/AAA/prices.parquet").to_pylist()
    rows = []
    for index in range(130, 151):
        row = dict(template)
        row["prediction_at"] = prices[index]["available_at"]
        row["price_end_index"] = index
        row["presence"] = [True, False, True, False, index % 2 == 0]
        row["macro"] = [0.0, float(row["presence"][-1]), 0.0]
        row["input_availability"] = {
            name: row["prediction_at"] if row["presence"][position] else None
            for position, name in enumerate(MODALITIES)
        }
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), samples, row_group_size=4)
    receipt_path = samples.parent / "manifest.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["samples_sha256"] = sha256(samples)
    dump(receipt_path, receipt)
    meta = json.loads(manifest.read_text())
    meta["samples"] = meta["coverage"][0]["samples"] = len(rows)
    dump(manifest, meta)
    output = tmp_path / "supervised"
    prepare_corpus_targets(manifest, prepared, output, input_policy=HISTORICAL_MASKED)
    path = output / "manifest.json"
    options = dict(partition="train", batch_size=5, epoch=2, seed=42)
    dataset = CorpusDataset(
        path, input_policy=HISTORICAL_MASKED, cache_bytes=1024**2, cache_sample_tables=True
    )
    batches = list(dataset.batches(**options))
    ids = [key for batch in batches for key in batch["sample_ids"]]
    assert len(ids) == len(set(ids)) == len(rows)
    expected_presence = {
        np.datetime64(row["prediction_at"].replace(tzinfo=None), "us"): row["presence"]
        for row in rows
    }
    for batch in batches:
        for moment, presence in zip(batch["prediction_at"], batch["presence"], strict=True):
            assert presence.tolist() == expected_presence[moment]
    before = sha256(path), path.stat().st_mtime_ns
    assert (
        prepare_corpus_targets(manifest, prepared, output, input_policy=HISTORICAL_MASKED)[
            "reused_assets"
        ]
        == 1
    )
    assert before == (sha256(path), path.stat().st_mtime_ns)
    resumed = list(dataset.batches(**options, cursor=batches[1]["confirmed_cursor"]))
    assert [key for batch in resumed for key in batch["sample_ids"]] == ids[10:]
    assert dataset.cached_bytes <= 1024**2


@pytest.mark.parametrize("field", ["training_ready", "final_test_opened"])
def test_historical_corpus_does_not_claim_scientific_admission(tmp_path, field):
    manifest, _ = historical_edition(tmp_path)
    meta = json.loads(manifest.read_text())
    meta[field] = True
    with pytest.raises(ValueError):
        cohort_identity(meta, input_policy=HISTORICAL_MASKED)


@pytest.mark.parametrize("version", [True, False, 3.0])
def test_boolean_and_float_versions_never_enter_historical_manifests(tmp_path, version):
    manifest, _ = historical_edition(tmp_path)
    meta = json.loads(manifest.read_text())
    meta["schema_version"] = version
    with pytest.raises(ValueError):
        cohort_identity(meta, input_policy=HISTORICAL_MASKED)


def test_historical_temporal_view_needs_a_separate_consumer_contract(tmp_path):
    path = supervised(tmp_path)
    meta = json.loads(path.read_text())
    meta["temporal_view"] = {}
    dump(path, meta)
    with pytest.raises(ValueError, match="temporal"):
        CorpusDataset(path, input_policy=HISTORICAL_MASKED)


@pytest.mark.parametrize(
    "mask", [None, [True, False, True, False], [True, None, True, False, False]]
)
def test_presence_has_exactly_five_known_booleans(tmp_path, mask):
    path = supervised(tmp_path)
    change_samples(
        tmp_path,
        path,
        lambda table: table.set_column(
            table.schema.get_field_index("presence"),
            "presence",
            pa.array([mask] * len(table), type=pa.list_(pa.bool_())),
        ),
    )
    with pytest.raises(ValueError):
        read_all(path)


def test_catalog_change_prevents_reusing_old_target_identity(tmp_path):
    manifest, prepared = historical_edition(tmp_path)
    output = tmp_path / "supervised"
    prepare_corpus_targets(manifest, prepared, output, input_policy=HISTORICAL_MASKED)
    path = tmp_path / "samples/US/AAA/manifest.json"
    receipt = json.loads(path.read_text())
    receipt["macro_indicators"] = ["renamed"]
    dump(path, receipt)
    with pytest.raises(ValueError, match="artefactos|identidad|representaciones"):
        prepare_corpus_targets(manifest, prepared, output, input_policy=HISTORICAL_MASKED)


def test_price_only_producer_reaches_supervision_and_reader_without_models(tmp_path):
    from mars_titan.data.cohort_preparation import prepare_cohort_asset
    from mars_titan.data.corpus_encoding import encode_corpus
    from tests.data.test_budget_targets import fixture_prices
    from tests.data.test_cohort_samples import Encoders

    clock, stock, market = fixture_prices()
    stock, market = stock.iloc[:160].copy(), market.iloc[:160].copy()
    for frame in (stock, market):
        frame["high"], frame["low"], frame["volume"] = 110.0, 90.0, 100.0
    raw = tmp_path / "raw"
    raw.mkdir()
    prices = raw / "prices.csv"
    stock.rename(
        columns={
            "session": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )[["Date", "Open", "High", "Low", "Close", "Volume"]].to_csv(prices, index=False)
    asset = dict(
        market="US",
        symbol="AAA",
        hashes={"prices.csv": sha256(prices)},
        paths=dict(prices=["prices.csv"], news=[], fundamentals=[], charts=[]),
    )
    prepared = tmp_path / "prepared"
    prepare_cohort_asset(
        raw,
        prepared,
        asset,
        clock,
        cohort="original_audited",
        reviews={},
        input_policy=HISTORICAL_MASKED,
    )
    preparation = tmp_path / "preparation.json"
    dump(
        preparation,
        dict(
            **policy_identity(HISTORICAL_MASKED),
            schema_version=2,
            kind="prepared_cohort",
            status="completed",
            cohort_id="original_audited",
            candidate_count=2,
            failed_assets=0,
            prepared_root=str(prepared),
            assets=[
                dict(
                    market="US",
                    symbol="AAA",
                    state="prepared",
                    manifest_sha256=sha256(prepared / "US/AAA/manifest.json"),
                ),
                dict(market="US", symbol="NO_PRICES", state="missing_required_prices"),
            ],
        ),
    )
    factor = tmp_path / "factor.parquet"
    pq.write_table(pa.Table.from_pandas(market, preserve_index=False), factor)
    encoded = tmp_path / "encoded"
    result = encode_corpus(
        preparation,
        encoded,
        macros={},
        encoders=Encoders(),
        clocks={"US": clock},
        company_factors=False,
        fundamental_concepts=["Assets"],
        input_policy=HISTORICAL_MASKED,
        macro_indicators=["inflation"],
        market_factors={
            "US": dict(
                market="US", symbol="FACTOR", prices_path=str(factor), prices_sha256=sha256(factor)
            )
        },
    )
    assert result["samples"] == 97
    assert result["failed_assets"] == 0
    output = tmp_path / "supervised"
    report = prepare_corpus_targets(
        encoded / "manifest.json", prepared, output, input_policy=HISTORICAL_MASKED
    )
    labels = pq.read_table(output / "labels/US/AAA/labels.parquet").to_pylist()
    assert len(labels) == 97
    assert sum(row["reason"] == "insufficient_history" for row in labels) == 62
    assert sum(row["reason"] == "missing_next_session" for row in labels) == 1
    assert report["counts"] == {"train": 34, "validation": 0}
    batches = list(
        supervised_batches(
            output / "manifest.json",
            partition="train",
            batch_size=9,
            epoch=0,
            seed=42,
            input_policy=HISTORICAL_MASKED,
        )
    )
    assert len({key for batch in batches for key in batch["sample_ids"]}) == 34
    assert all(
        np.array_equal(
            batch["presence"], np.tile([True, False, True, False, False], (len(batch["target"]), 1))
        )
        for batch in batches
    )
    assert report["coverage"][-1]["state"] == "missing_required_prices"


@pytest.mark.parametrize("count", [1, -1, 0.0, None])
def test_absent_news_has_a_known_zero_event_count(tmp_path, count):
    path = supervised(tmp_path)
    change_samples(
        tmp_path,
        path,
        lambda table: table.set_column(
            table.schema.get_field_index("news_count"), "news_count", pa.array([count] * len(table))
        ),
    )
    with pytest.raises(ValueError):
        read_all(path)


def test_present_news_needs_an_admitted_event_even_with_zero_embedding(tmp_path):
    path = supervised(tmp_path)

    def observed(row):
        row["presence"][1] = True
        row["input_availability"]["news"] = row["prediction_at"]

    change_samples(tmp_path, path, lambda table: change_rows(table, observed))
    with pytest.raises(ValueError):
        read_all(path)


@pytest.mark.parametrize("value", ["0.0", False])
def test_numeric_vectors_do_not_accept_text_or_booleans(tmp_path, value):
    path = supervised(tmp_path)
    change_samples(
        tmp_path,
        path,
        lambda table: table.set_column(
            table.schema.get_field_index("news"), "news", pa.array([[value, value]] * len(table))
        ),
    )
    with pytest.raises(ValueError):
        read_all(path)


@pytest.mark.parametrize("name", ["fundamentals", "macro"])
def test_present_numeric_block_contains_an_observed_concept(tmp_path, name):
    path = supervised(tmp_path)

    def claim(row):
        row["presence"][MODALITIES.index(name)] = True
        row["input_availability"][name] = row["prediction_at"]

    change_samples(tmp_path, path, lambda table: change_rows(table, claim))
    with pytest.raises(ValueError, match="conceptos"):
        read_all(path)


def test_missing_price_candidate_cannot_disappear_from_coverage(tmp_path):
    manifest, prepared = historical_edition(tmp_path)
    meta = json.loads(manifest.read_text())
    meta["coverage"].pop()
    dump(manifest, meta)
    with pytest.raises(ValueError, match="cobertura|candidatos"):
        prepare_corpus_targets(
            manifest, prepared, tmp_path / "supervised", input_policy=HISTORICAL_MASKED
        )


def test_reserved_sample_is_rejected_before_decoding_its_vectors(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    path = supervised(tmp_path)
    change_samples(
        tmp_path,
        path,
        lambda table: change_rows(
            table, lambda row: row.update(prediction_at=datetime(2024, 1, 2, tzinfo=UTC))
        ),
    )
    original = pq.ParquetFile.read_row_group

    def guarded(file, group, columns=None, **kwargs):
        if columns and "news" in columns:
            raise AssertionError("No deben decodificarse las entradas de una decisión reservada")
        return original(file, group, columns=columns, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", guarded)
    with pytest.raises(ValueError, match="corte"):
        read_all(path)
