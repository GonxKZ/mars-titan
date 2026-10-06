"""Edición de monedas separadas con vectores y objetivos de origen recuperables."""

import csv
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.cohort_contexts import MacroVectors
from mars_titan.data.cohort_samples import materialize_cohort_asset
from mars_titan.data.embeddings import EmbeddingCache
from mars_titan.data.storage import sha256
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.corpus_targets import prepare_corpus_targets
from tests.data.test_budget_targets import fixture_prices
from tests.data.test_cohort_samples import Encoders
from tests.data.test_company_factors import balance


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, default=str))


@pytest.fixture
def currency_corpus(tmp_path):
    clock, stock, market = fixture_prices()
    stock.loc[stock.session == "2023-01-03", "close"] += 2
    for frame in (stock, market):
        frame["high"], frame["low"], frame["volume"] = 120.0, 80.0, 100.0
    root = tmp_path / "original"
    prepared, samples = root / "prepared", root / "samples"
    market_path = root / "SPY.parquet"
    root.mkdir()
    pq.write_table(pa.Table.from_pandas(market, preserve_index=False), market_path)
    moments = [
        clock.decision(day)
        for day in (
            "2022-12-27",
            "2022-12-28",
            "2022-12-29",
            "2022-12-30",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
            "2023-01-06",
        )
    ]
    identifiers = sorted(
        r["id"] for r in csv.DictReader(Path("data/catalogs/macro-indicators.csv").open())
    )
    macro_path = root / "macro.parquet"
    old_macro = pa.Table.from_pylist(
        [
            dict(prediction_at=t, indicator_id=name, value=1.0, available_at=moments[0])
            for t in moments
            for name in identifiers
        ]
    )
    pq.write_table(old_macro, macro_path)
    cache_path = root / "embeddings.sqlite"
    cache = EmbeddingCache(cache_path)
    receipts = {}
    for symbol, currency in (("A", "USD"), ("CADX", "CAD")):
        folder = prepared / "US" / symbol
        (folder / "news").mkdir(parents=True)
        pq.write_table(pa.Table.from_pandas(stock, preserve_index=False), folder / "prices.parquet")
        facts = [
            dict(
                r,
                concept=r["concept"].replace(":USD", ":" + currency),
                unit=currency,
                period_end="2022-09-30",
                filed="2022-10-03",
                available_at=clock.decision("2022-10-04"),
            )
            for r in balance()
        ]
        pq.write_table(pa.Table.from_pylist(facts), folder / "fundamentals.parquet")
        news = [
            dict(
                available_at=t,
                content_hash=hashlib.sha256((symbol + str(i)).encode()).hexdigest(),
                event_id=str(i),
                content_kind="article_candidate",
                text="Noticia " + symbol,
                cohort_id="original_audited",
                availability_rule="source_timestamp",
            )
            for i, t in enumerate((moments[0], moments[4]))
        ]
        pq.write_table(pa.Table.from_pylist(news), folder / "news/news.parquet")
        pq.write_table(
            pa.table({"reason": pa.array([], type=pa.string())}), folder / "news/excluded.parquet"
        )
        write_json(folder / "news/manifest.json", dict(counts=dict(accepted=2)))
        names = (
            "prices.parquet",
            "fundamentals.parquet",
            "news/news.parquet",
            "news/excluded.parquet",
            "news/manifest.json",
        )
        write_json(
            folder / "manifest.json",
            dict(
                schema_version=3,
                market="US",
                symbol=symbol,
                cohort_id="original_audited",
                news_content_policy="source_audited_not_external",
                training_ready=False,
                fingerprint=hashlib.sha256(symbol.encode()).hexdigest(),
                counts=dict(prices=len(stock), fundamentals=len(facts), news=len(news)),
                policy=dict(
                    cutoff="2023-12-31",
                    calendar=hashlib.sha256(
                        "|".join(t.isoformat() for t in clock.decisions).encode()
                    ).hexdigest(),
                ),
                artifacts={name: sha256(folder / name) for name in names},
            ),
        )
        receipts[symbol] = materialize_cohort_asset(
            folder,
            samples / "US" / symbol,
            clock,
            MacroVectors(macro_path),
            Encoders(),
            cache,
            cohort="original_audited",
            context=2,
        )
    cache.close()
    encoded = root / "encoded.json"
    coverage = [
        dict(
            market="US",
            symbol=symbol,
            state="encoded",
            samples=r["samples"],
            fingerprint=r["fingerprint"],
        )
        for symbol, r in receipts.items()
    ]
    write_json(
        encoded,
        dict(
            schema_version=2,
            kind="materialized_corpus",
            cohort_id="original_audited",
            news_content_policy="source_audited_not_external",
            scope="full_corpus",
            cohort_complete=True,
            context_sessions=2,
            samples_root=str(samples),
            calendar_start=dict(US="2021-01-01"),
            assets=[dict(market="US", symbol="A", cohort_id="original_audited")],
            coverage=coverage,
            candidate_count=2,
            samples=receipts["A"]["samples"],
            failed_assets=0,
            markets=["US"],
            market_factors=dict(
                US=dict(
                    market="US",
                    symbol="SPY",
                    prices_path=str(market_path),
                    prices_sha256=sha256(market_path),
                )
            ),
            final_test_opened=False,
        ),
    )
    prepare_corpus_targets(encoded, prepared, root / "supervised")
    selected = [moments[i] for i in (1, 2, 3, 4, 6)]
    new_macro = root / "new-macro.parquet"
    table = pa.Table.from_pylist(
        [
            dict(prediction_at=t, indicator_id=name, value=2.0, available_at=t)
            for t in moments
            for name in identifiers
        ]
    )
    pq.write_table(table, new_macro)
    admission = root / "admission"
    admission.mkdir()
    complete = admission / "complete-decisions.parquet"
    pq.write_table(
        pa.table(
            dict(
                prediction_at=pa.array(selected),
                macro_available_at=pa.array(selected),
                indicator_count=pa.array([140] * len(selected), type=pa.int32()),
            )
        ),
        complete,
    )
    catalog = Path("data/catalogs/macro-indicators.csv").resolve()
    write_json(
        admission / "report.json",
        dict(
            schema_version=1,
            market="US",
            required_indicator_count=140,
            required_indicator_ids=identifiers,
            source_sha256=sha256(new_macro),
            catalog_sha256=sha256(catalog),
            complete_decisions=len(selected),
            complete_decisions_path=complete.name,
            complete_decisions_sha256=sha256(complete),
            start="2022-12-27",
            end="2023-01-06",
        ),
    )
    return dict(
        parent_manifest=root / "supervised/manifest.json",
        macro_path=new_macro,
        admission_path=admission / "report.json",
        catalog_path=catalog,
        clock=clock,
        source_cache=cache_path,
    ), selected


def prepare(fixture, output, **kwargs):
    options, _ = fixture
    module = importlib.import_module("mars_titan.data.currency_samples")
    return module.prepare_currency_corpus(output=output, **options, **kwargs)


def test_cpu_projection_is_pending_until_cad_finishes_and_resumes_without_recopying(
    currency_corpus, tmp_path
):
    output = tmp_path / "edition"
    report = prepare(currency_corpus, output)
    assert report["status"] == "requires_encoders"
    assert report["pending_cad"] == ["CADX"]
    assert not (output / "encoded/manifest.json").exists()
    assert not (output / "supervised/manifest.json").exists()
    path = output / "encoded/samples/US/A/samples.parquet"
    before = (sha256(path), path.stat().st_mtime_ns)
    completed = prepare(currency_corpus, output, encoders=Encoders())
    assert completed["status"] == "completed"
    assert before == (sha256(path), path.stat().st_mtime_ns)
    dataset = CorpusDataset(output / "supervised/manifest.json")
    assert dataset.manifest["configuration"]["source_manifest_sha256"] == sha256(
        output / "encoded/manifest.json"
    )
    assert len(dataset.manifest["representation"]["fundamental_concepts"]) == 23
    assert {a["symbol"] for a in dataset.assets} == {"A", "CADX"}
    for part in ("train", "validation"):
        for batch in dataset.batches(partition=part, batch_size=3, epoch=0, seed=42):
            assert batch["inputs"]["fundamentals"].shape[-1] == 69


def test_usd_projection_preserves_vectors_labels_and_year_boundary(currency_corpus, tmp_path):
    options, selected = currency_corpus
    parent = json.loads(options["parent_manifest"].read_text())
    originals = {key: Path(value) / "US/A" for key, value in parent["roots"].items()}
    source = pq.read_table(originals["samples"] / "samples.parquet")
    labels = pq.read_table(originals["labels"] / "labels.parquet")
    output = tmp_path / "edition"
    prepare(currency_corpus, output, encoders=Encoders())
    new = pq.read_table(output / "encoded/samples/US/A/samples.parquet")
    new_labels = pq.read_table(output / "supervised/labels/US/A/labels.parquet")
    indices = new["source_sample_row"].to_numpy()
    assert new["prediction_at"].to_pylist() == selected
    for name in ("news", "charts", "price_end_index", "chart_hash", "news_set_sha256"):
        assert new[name].to_pylist() == source[name].take(pa.array(indices)).to_pylist()
    for name in ("target", "target_available_at", "reason", "partition", "prediction_at"):
        assert new_labels[name].to_pylist() == labels[name].take(pa.array(indices)).to_pylist()
    old_values = np.array(
        source["fundamentals"].take(pa.array(indices)).to_pylist(), dtype=np.float32
    ).reshape(-1, 3, 15)
    new_values = np.array(new["fundamentals"].to_pylist(), dtype=np.float32).reshape(-1, 3, 23)
    np.testing.assert_array_equal(new_values[:, :, :15], old_values)
    assert not new_values[:, :, 15:].any()
    assert "target_crosses_partition_boundary" in new_labels["reason"].to_pylist()
    for t, values, available in zip(
        new["prediction_at"].to_pylist(),
        new["macro"].to_pylist(),
        new["macro_available_at"].to_pylist(),
        strict=True,
    ):
        expected = MacroVectors(options["macro_path"]).at(t)
        np.testing.assert_array_equal(values, expected[0])
        assert available == expected[1]


def test_cad_levels_remain_cad_and_all_new_inputs_have_real_availability(currency_corpus, tmp_path):
    output = tmp_path / "edition"
    prepare(currency_corpus, output, encoders=Encoders())
    table = pq.read_table(output / "encoded/samples/US/CADX/samples.parquet")
    values = np.array(table["fundamentals"].to_pylist(), dtype=np.float32).reshape(-1, 3, 23)
    assert not values[:, 1, :8].any()
    assert values[:, 1, 15].all()
    np.testing.assert_allclose(values[:, 0, 15], np.log1p(100), rtol=1e-7)
    assert values[:, 1, 8:15].all()
    for row in table.to_pylist():
        assert set(row["input_availability"]) == {
            "prices",
            "news",
            "fundamentals",
            "charts",
            "macro",
        }
        assert all(v <= row["prediction_at"] for v in row["input_availability"].values())
        assert row["prediction_at"].year < 2024
    labels = pq.read_table(output / "supervised/labels/US/CADX/labels.parquet").to_pylist()
    annual = next(r for r in labels if r["reason"] == "target_crosses_partition_boundary")
    assert annual["target"] == pytest.approx(0.02, abs=1e-12)
    assert annual["target_available_at"].date().isoformat() == "2023-01-03"


def test_invalid_symbol_cannot_escape_the_edition_root(currency_corpus, tmp_path):
    options, _ = currency_corpus
    parent = json.loads(options["parent_manifest"].read_text())
    parent["assets"][0]["symbol"] = parent["coverage"][0]["symbol"] = "../../outside"
    write_json(options["parent_manifest"], parent)
    with pytest.raises(ValueError, match="identidad|símbolo"):
        prepare(currency_corpus, tmp_path / "edition")
    assert not (tmp_path / "edition").exists()


def test_changed_encoder_or_source_never_completes_an_edition(currency_corpus, tmp_path):
    prepare(currency_corpus, tmp_path / "edition")
    changed = Encoders()
    changed.spec = changed.spec | {"version": 2}
    with pytest.raises(ValueError, match="codificador|encoder"):
        prepare(currency_corpus, tmp_path / "edition", encoders=changed)
    assert not (tmp_path / "edition/supervised/manifest.json").exists()


def test_an_usd_asset_outside_admission_keeps_typed_empty_artifacts(currency_corpus, tmp_path):
    options, _ = currency_corpus
    parent = json.loads(options["parent_manifest"].read_text())
    folder = Path(parent["roots"]["samples"]) / "US/A"
    label_path = Path(parent["roots"]["labels"]) / "US/A/labels.parquet"
    pq.write_table(
        pq.read_table(folder / "samples.parquet").slice(0, 1), folder / "samples.parquet"
    )
    pq.write_table(pq.read_table(label_path).slice(0, 1), label_path)
    receipt = json.loads((folder / "manifest.json").read_text())
    receipt.update(samples=1, samples_sha256=sha256(folder / "samples.parquet"))
    write_json(folder / "manifest.json", receipt)
    parent["assets"][0].update(
        samples=1,
        samples_sha256=receipt["samples_sha256"],
        labels_sha256=sha256(label_path),
        counts=dict(train=1, validation=0),
    )
    parent["coverage"][0]["samples"] = parent["samples"] = 1
    parent["counts"] = dict(train=1, validation=0)
    write_json(options["parent_manifest"], parent)
    output = tmp_path / "edition"
    prepare(currency_corpus, output, encoders=Encoders())
    labels = pq.read_table(output / "supervised/labels/US/A/labels.parquet")
    assert len(labels) == 0
    assert {"sample_row", "target", "reason", "source_sample_row"} <= set(labels.column_names)
    dataset = CorpusDataset(output / "supervised/manifest.json")
    assert [a["symbol"] for a in dataset.assets] == ["CADX"]


def test_sample_receipt_cannot_misattribute_the_encoder(currency_corpus, tmp_path):
    options, _ = currency_corpus
    parent = json.loads(options["parent_manifest"].read_text())
    path = Path(parent["roots"]["samples"]) / "US/A/manifest.json"
    receipt = json.loads(path.read_text())
    receipt["encoders"] = receipt["encoders"] | {"version": 99}
    write_json(path, receipt)
    with pytest.raises(ValueError, match="representación|encoder|codificador"):
        prepare(currency_corpus, tmp_path / "edition")


def test_new_cache_cannot_overwrite_the_original(currency_corpus, tmp_path):
    options, _ = currency_corpus
    before = sha256(options["source_cache"])
    with pytest.raises(ValueError, match="caché|destino|origen"):
        prepare(
            currency_corpus,
            tmp_path / "edition",
            cache_path=options["source_cache"],
            encoders=Encoders(),
        )
    assert sha256(options["source_cache"]) == before


def test_information_view_records_both_currency_dependencies(currency_corpus, tmp_path):
    from mars_titan.training.information_inputs import corpus_view

    output = tmp_path / "edition"
    prepare(currency_corpus, output, encoders=Encoders())
    dataset = CorpusDataset(output / "supervised/manifest.json")
    view = corpus_view(dataset, macro_catalog=currency_corpus[0]["catalog_path"])
    ratio = next(
        v
        for v in view.manifest["variables"]
        if v["name"] == "fundamentals/company:current_ratio:ratio"
    )
    assert set(ratio["dependencies"]) == {
        "fundamentals/us-gaap:AssetsCurrent:USD",
        "fundamentals/us-gaap:LiabilitiesCurrent:USD",
        "fundamentals/us-gaap:AssetsCurrent:CAD",
        "fundamentals/us-gaap:LiabilitiesCurrent:CAD",
    }
    assert "CAD" in ratio["history"]


def test_published_manifests_satisfy_the_real_encoder_consumer(currency_corpus, tmp_path):
    from mars_titan.posttraining.preparation import encoder_contract

    output = tmp_path / "edition"
    prepare(currency_corpus, output, encoders=Encoders())
    contract = encoder_contract(
        output / "supervised/manifest.json", output / "encoded/manifest.json"
    )
    assert contract["encoders"] == Encoders.spec
    assert contract["context"] == 2


class CachedEncoders:
    """Acreditar el codificador de los vectores sin permitir una inferencia nueva."""

    spec = Encoders.spec

    def text(self, text):
        raise RuntimeError("Falta el texto en la caché, no se permite codificar")

    def images(self, images):
        raise RuntimeError("Falta el gráfico en la caché, no se permite codificar")


def test_cache_only_edition_reuses_both_cache_sources_without_reencoding(currency_corpus, tmp_path):
    import sqlite3
    from contextlib import closing

    first, second = tmp_path / "first", tmp_path / "second"
    prepare(currency_corpus, first, encoders=Encoders())
    prepare(currency_corpus, second)
    sources = [first / "embeddings.sqlite", currency_corpus[0]["source_cache"]]
    hashes = {path: sha256(path) for path in sources}
    with closing(sqlite3.connect(sources[0].resolve().as_uri() + "?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(second / "embeddings.sqlite")) as destination:
            source.backup(destination)
    result = prepare(currency_corpus, second, encoders=CachedEncoders())
    assert result["status"] == "completed"
    for symbol in ("A", "CADX"):
        for relative in (
            f"encoded/samples/US/{symbol}/samples.parquet",
            f"supervised/labels/US/{symbol}/labels.parquet",
        ):
            assert pq.read_table(first / relative).equals(pq.read_table(second / relative))
    receipt = json.loads((second / "encoded/samples/US/CADX/manifest.json").read_text())
    assert receipt["currency_projection"]["source"]["kind"] == "cached_cad_materialization"
    assert receipt["currency_projection"]["source"]["neural_inference_executed"] is False
    assert hashes == {path: sha256(path) for path in sources}


def test_cache_only_mode_fails_closed_when_a_representation_is_missing(currency_corpus, tmp_path):
    with pytest.raises(RuntimeError, match="caché"):
        prepare(currency_corpus, tmp_path / "edition", encoders=CachedEncoders())
    assert not (tmp_path / "edition/encoded/manifest.json").exists()


@pytest.mark.parametrize("field", ["counts", "excluded_reasons"])
def test_resume_checks_receipt_counts_against_verified_label_rows(currency_corpus, tmp_path, field):
    output = tmp_path / "edition"
    prepare(currency_corpus, output, encoders=Encoders())
    path = output / "supervised/labels/US/A/receipt.json"
    receipt = json.loads(path.read_text())
    if field == "counts":
        assert receipt["counts"] == {"train": 2, "validation": 2}
        receipt["counts"]["train"] += 1
        receipt["counts"]["validation"] -= 1
    else:
        receipt["excluded_reasons"] = {}
    write_json(path, receipt)
    before = sha256(output / "supervised/manifest.json")
    with pytest.raises(ValueError, match="recibo|recuento"):
        prepare(currency_corpus, output, encoders=Encoders())
    assert sha256(output / "supervised/manifest.json") == before
