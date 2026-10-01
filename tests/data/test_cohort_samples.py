"""Materialización por ventanas con recuperación y procedencia explícita."""

import hashlib
import importlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.cohort_contexts import MacroVectors
from mars_titan.data.cohort_preparation import prepare_cohort_asset
from mars_titan.data.embeddings import EmbeddingCache
from mars_titan.data.samples import eligible_samples
from mars_titan.data.storage import sha256
from tests.data.test_cohort_preparation import fixture as original_fixture


def materialize(*args, **kwargs):
    try:
        module = importlib.import_module("mars_titan.data.cohort_samples")
    except ModuleNotFoundError:
        pytest.fail("Falta la materialización por ventanas de la cohorte")
    return module.materialize_cohort_asset(*args, **kwargs)


class Encoders:
    """Codificador numérico de prueba, sin modelos, red ni afirmaciones científicas."""

    spec = {"purpose": "unit_test", "device": "fixture", "version": 1}

    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def text(self, text):
        return np.full(384, len(text) / 100, dtype=np.float32)

    def images(self, pngs):
        self.calls += 1
        if self.fail and self.calls == 2:
            raise RuntimeError("Interrupción de prueba")
        return np.stack(
            [np.full(512, hashlib.sha256(p).digest()[0] / 255, dtype=np.float32) for p in pngs]
        )


def fixture(tmp_path):
    raw, asset, clock = original_fixture(tmp_path)
    days = [d for d in clock.days if "2023-07-03" <= d.isoformat() <= "2023-07-14"]
    (raw / "prices.csv").write_text(
        "Date,Open,High,Low,Close,Volume\n"
        + "".join(
            f"{d},{10 + i},{12 + i},{9 + i},{11 + i + (i % 2) * 0.3},100\n"
            for i, d in enumerate(days)
        )
    )
    asset["hashes"]["prices.csv"] = sha256(raw / "prices.csv")
    root = tmp_path / "prepared"
    prepare_cohort_asset(raw, root, asset, clock, cohort="original_audited", reviews={})
    macro_path = tmp_path / "macro.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    prediction_at=clock.decision(str(d)),
                    indicator_id="observed",
                    available_at=clock.decision(str(days[0])),
                    value=1.0,
                    unit="ratio",
                )
                for d in days
            ]
        ),
        macro_path,
    )
    return root / "US/A", clock, MacroVectors(macro_path)


def run(source, output, clock, macro, cache, **kwargs):
    return materialize(
        source,
        output,
        clock,
        macro,
        kwargs.pop("encoders", Encoders()),
        cache,
        cohort="original_audited",
        context=2,
        company_factors=False,
        batch_rows=2,
        **kwargs,
    )


def test_windows_match_reference_and_store_four_modalities_with_cohort(tmp_path):
    source, clock, macro = fixture(tmp_path)
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        output = tmp_path / "encoded"
        report = run(source, output, clock, macro, cache)
        table = pq.read_table(output / "samples.parquet")
        rows = table.to_pylist()
        news = pq.read_table(source / "news/news.parquet").to_pylist()
        reference = list(
            eligible_samples(
                pq.read_table(source / "prices.parquet").to_pandas(),
                news,
                pq.read_table(source / "fundamentals.parquet").to_pylist(),
                clock,
                context=2,
            )
        )
        assert [r["session"] for r in rows] == [r["session"] for r in reference]
        assert len(rows) == 5
        assert rows[0]["session"] == "2023-07-06"
        assert report["samples"] == len(rows)
        assert report["cohort_id"] == "original_audited"
        assert report["news_content_policy"] == "source_audited_not_external"
        assert report["training_ready"] is False
        assert report["reused"] is False
        assert table.schema.field("news").type == pa.list_(pa.float32(), 384)
        assert table.schema.field("charts").type == pa.list_(pa.float32(), 512)
        for row, expected in zip(rows, reference, strict=True):
            assert row["cohort_id"] == "original_audited"
            assert row["price_end_index"] == expected["price_end_index"]
            np.testing.assert_allclose(row["fundamentals"], expected["fundamentals"], rtol=1e-6)
            np.testing.assert_array_equal(row["news"], Encoders().text(news[0]["text"]))
            assert row["news_count"] == 1
            assert row["news_kind_counts"] == dict(
                article_candidate=1, summary=0, verified_full_article=0
            )
            assert all(t <= row["prediction_at"] for t in row["input_availability"].values())
        again = run(source, output, clock, macro, cache)
        assert again["reused"] is True
        assert again["samples_sha256"] == report["samples_sha256"]
    finally:
        cache.close()


def test_interrupted_asset_recovers_cached_vectors_without_publishing_partial_table(tmp_path):
    source, clock, macro = fixture(tmp_path)
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        output = tmp_path / "interrupted"
        with pytest.raises(RuntimeError, match="Interrupción"):
            run(source, output, clock, macro, cache, encoders=Encoders(fail=True))
        assert not (output / "manifest.json").exists()
        assert not (output / "samples.parquet").exists()
        resumed = run(source, output, clock, macro, cache)
        baseline = run(source, tmp_path / "reference", clock, macro, cache)
        assert resumed["samples_sha256"] == baseline["samples_sha256"]
        assert resumed["cache_hits"]["news"] >= 1
    finally:
        cache.close()


def test_receipt_rejects_changed_cohort_counts_or_readiness(tmp_path):
    source, clock, macro = fixture(tmp_path)
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        output = tmp_path / "encoded"
        run(source, output, clock, macro, cache)
        path = output / "manifest.json"
        original = path.read_text()
        for change in (
            dict(cohort_id="externally_verified"),
            dict(samples=100),
            dict(training_ready=True),
        ):
            path.write_text(json.dumps({**json.loads(original), **change}))
            with pytest.raises(ValueError, match="recibo|identidad|recuento"):
                run(source, output, clock, macro, cache)
        path.write_text(original)
        (source / "news/news.parquet").write_bytes(b"Artefacto cambiado")
        with pytest.raises(ValueError, match="huella|artefacto"):
            run(source, output, clock, macro, cache)
    finally:
        cache.close()


def test_missing_macro_produces_typed_empty_partition_and_explicit_exclusions(tmp_path):
    source, clock, macro = fixture(tmp_path)
    macro.index.clear()
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        output = tmp_path / "encoded"
        report = run(source, output, clock, macro, cache)
        table = pq.read_table(output / "samples.parquet")
        assert report["samples"] == 0
        assert report["excluded_reasons"]["missing_macro"] == 5
        assert "prediction_at" in table.schema.names
        assert table.schema.field("news").type == pa.list_(pa.float32(), 384)
    finally:
        cache.close()


def test_changed_macro_file_cannot_reuse_receipt(tmp_path):
    source, clock, macro = fixture(tmp_path)
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        output = tmp_path / "encoded"
        run(source, output, clock, macro, cache)
        rows = pq.read_table(macro.path).to_pylist()
        rows[0]["value"] = 99.0
        pq.write_table(pa.Table.from_pylist(rows), macro.path)
        with pytest.raises(ValueError, match="macro|cambiado"):
            run(source, output, clock, macro, cache)
    finally:
        cache.close()
