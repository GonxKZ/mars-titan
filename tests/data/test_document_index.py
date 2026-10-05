"""Índices documentales sobre noticias preparadas y cachés reales, sin codificadores."""

import importlib
import json
import sqlite3
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.cohort_news import write_cohort_news
from mars_titan.data.cohort_samples import _text_window
from mars_titan.data.embeddings import EmbeddingCache
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock

ENCODER = "a" * 64
START = datetime(2023, 7, 1, tzinfo=UTC)
END = datetime(2023, 7, 8, tzinfo=UTC)


def api():
    try:
        return importlib.import_module("mars_titan.data.document_index")
    except ModuleNotFoundError:
        pytest.fail("Falta el índice de documentos anterior a la agregación")


def fixture(tmp_path, *, future=False):
    clock = MarketClock("US", "2023-01-01", "2024-01-01")
    source = tmp_path / "raw"
    source.mkdir(parents=True)
    cache_path, manifests = tmp_path / "cache.sqlite", {}
    cache = EmbeddingCache(cache_path)
    try:
        for symbol in ("A", "B"):
            rows = [
                dict(Date=day, Article_title=title, Article=body, Url=url, Stock_symbol=symbol)
                for day, title, body, url in [
                    ("2023-07-05", "Documento uno", "Texto uno", "https://example.org/1"),
                    ("2023-07-06", "Documento dos", "Texto dos", "https://example.org/2"),
                ]
            ]
            if future:
                rows.append(dict(Date="2023-07-10", Article="Futuro", Stock_symbol=symbol))
            path = source / f"{symbol}.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            prepared = tmp_path / "prepared" / symbol
            write_cohort_news(
                source,
                [path.name],
                prepared,
                symbol=symbol,
                clock=clock,
                cohort="original_audited",
                batch_rows=1,
            )
            manifests[f"US/{symbol}"] = prepared / "manifest.json"
            for index, row in enumerate(pq.read_table(prepared / "news.parquet").to_pylist()):
                identity = dict(
                    encoder=ENCODER,
                    kind="news",
                    content=row["content_hash"],
                    policy=row["availability_rule"],
                )
                cache.put(identity, np.full(384, index * 2 + 1, dtype=np.float32))
    finally:
        cache.close()
    return manifests, cache_path


def prepare(tmp_path, *, future=False, **kwargs):
    sources, cache = fixture(tmp_path, future=future)
    output = tmp_path / "index"
    report = api().prepare_document_index(
        sources, output, cache_path=cache, encoder_sha256=ENCODER, batch_rows=1, **kwargs
    )
    return output, cache, sources, report


def test_mentions_remain_distinct_and_mean_reuses_existing_vectors(tmp_path):
    output, cache, sources, report = prepare(tmp_path)
    with api().DocumentIndex(output / "manifest.json", cache_path=cache) as reader:
        rows = [r for b in reader.batches("US/A", START, END, batch_size=1) for r in b["documents"]]
        other = [r for b in reader.batches("US/B", START, END) for r in b["documents"]]
        assert len(rows) == len(other) == 2
        assert {r["document_id"] for r in rows} == {r["document_id"] for r in other}
        assert {r["mention_id"] for r in rows}.isdisjoint(r["mention_id"] for r in other)
        assert rows[0]["source_file"] == "A.jsonl"
        assert rows[0]["line"] == 1
        assert rows[0]["historical_body_version_verified"] is False
        result = reader.compare("US/A", START, END, selected=1)
        np.testing.assert_array_equal(result["mean"], np.full(384, 2.0))
        np.testing.assert_array_equal(result["selected_mean"], np.full(384, 3.0))
        assert result["examined_mentions"] == result["unique_documents"] == 2
        assert result["encoded_documents"] == 0
        assert result["selected_mentions"] == [rows[-1]["mention_id"]]
    reference_rows = pq.read_table(sources["US/A"].parent / "news.parquet").to_pylist()
    db = EmbeddingCache(cache)
    try:
        expected = _text_window(
            reference_rows, "original_audited", ENCODER, None, db, Counter(), Counter()
        )
    finally:
        db.close()
    np.testing.assert_array_equal(result["mean"], expected["news"])
    assert report["mentions"] == 4 and report["documents"] == 2


def test_future_suffix_does_not_change_available_rows_or_selection(tmp_path):
    outputs = [
        prepare(tmp_path / name, future=future)
        for name, future in (("prefix", False), ("extended", True))
    ]
    results = []
    for output, cache, _, _ in outputs:
        with api().DocumentIndex(output / "manifest.json", cache_path=cache) as reader:
            results.append(reader.compare("US/A", START, END, selected=1))
    for key in ("mean", "selected_mean", "selected_mentions", "window_sha256"):
        np.testing.assert_array_equal(results[0][key], results[1][key])


def test_cursor_resumes_confirmed_prefix_and_rejects_another_window(tmp_path):
    output, cache, _, _ = prepare(tmp_path)
    with api().DocumentIndex(output / "manifest.json", cache_path=cache) as reader:
        all_batches = list(reader.batches("US/A", START, END, batch_size=1))
        cursor = all_batches[0]["confirmed_cursor"]
        remaining = list(reader.batches("US/A", START, END, batch_size=1, cursor=cursor))
        assert remaining == all_batches[1:]
        assert (
            list(
                reader.batches(
                    "US/A", START, END, batch_size=1, cursor=all_batches[-1]["confirmed_cursor"]
                )
            )
            == []
        )
        with pytest.raises(ValueError, match="cursor"):
            list(reader.batches("US/B", START, END, batch_size=1, cursor=cursor))
        with pytest.raises(ValueError, match="cursor"):
            list(
                reader.batches(
                    "US/A", START, END, batch_size=1, cursor=cursor | {"prefix_sha256": "0" * 64}
                )
            )


def test_publication_is_reusable_and_interruption_cannot_publish_partial_index(
    tmp_path, monkeypatch
):
    sources, cache = fixture(tmp_path)
    output = tmp_path / "index"
    original = api().atomic_parquet_batches

    def interrupted(path, tables):
        next(iter(tables))
        raise InterruptedError("Corte antes de publicar")

    monkeypatch.setattr(api(), "atomic_parquet_batches", interrupted)
    with pytest.raises(InterruptedError):
        api().prepare_document_index(sources, output, cache_path=cache, encoder_sha256=ENCODER)
    assert not output.exists()
    monkeypatch.setattr(api(), "atomic_parquet_batches", original)
    first = api().prepare_document_index(sources, output, cache_path=cache, encoder_sha256=ENCODER)
    second = api().prepare_document_index(sources, output, cache_path=cache, encoder_sha256=ENCODER)
    assert second == first | {"reused": True}


@pytest.mark.parametrize("damage", ["missing", "checksum", "replaced", "width", "encoder"])
def test_incompatible_or_corrupt_cache_is_never_recoded(tmp_path, damage):
    output, cache, sources, _ = prepare(tmp_path)
    with sqlite3.connect(cache) as db:
        if damage == "missing":
            db.execute("DELETE FROM embeddings")
        elif damage == "checksum":
            db.execute("UPDATE embeddings SET checksum='broken'")
    if damage in {"width", "replaced"}:
        db = EmbeddingCache(cache)
        try:
            for row in pq.read_table(sources["US/A"].parent / "news.parquet").to_pylist():
                db.put(
                    dict(
                        encoder=ENCODER,
                        kind="news",
                        content=row["content_hash"],
                        policy=row["availability_rule"],
                    ),
                    np.zeros(2 if damage == "width" else 384, dtype=np.float32),
                )
        finally:
            db.close()
    with pytest.raises(ValueError, match="caché|representación|codificador"):
        with api().DocumentIndex(
            output / "manifest.json",
            cache_path=cache,
            encoder_sha256="b" * 64 if damage == "encoder" else ENCODER,
        ) as reader:
            reader.compare("US/A", START, END, selected=1)


def test_ambiguous_duplicate_and_changed_content_are_rejected(tmp_path):
    sources, cache = fixture(tmp_path)
    path = sources["US/A"].parent / "news.parquet"
    rows = pq.read_table(path).to_pylist()
    rows.insert(1, rows[0] | {"availability_rule": "another_rule"})
    pq.write_table(pa.Table.from_pylist(rows), path)
    receipt = json.loads(sources["US/A"].read_text())
    receipt["artifacts"]["news.parquet"] = sha256(path)
    receipt["counts"] = dict(records=3, accepted=3, excluded=0)
    sources["US/A"].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="duplicad|caché"):
        api().prepare_document_index(
            sources, tmp_path / "index", cache_path=cache, encoder_sha256=ENCODER
        )
    rows = [rows[0] | {"text": "Cuerpo modificado sin cambiar su huella"}, rows[-1]]
    pq.write_table(pa.Table.from_pylist(rows), path)
    receipt["artifacts"]["news.parquet"] = sha256(path)
    receipt["counts"] = dict(records=2, accepted=2, excluded=0)
    sources["US/A"].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="contenido|huella"):
        api().prepare_document_index(
            sources, tmp_path / "index", cache_path=cache, encoder_sha256=ENCODER
        )


def test_query_and_parquet_budgets_reject_excess_before_returning_comparison(tmp_path):
    output, cache, _, _ = prepare(tmp_path)
    with pytest.raises(ValueError, match="presupuesto"):
        api().DocumentIndex(output / "manifest.json", cache_path=cache, max_group_bytes=1)
    with api().DocumentIndex(output / "manifest.json", cache_path=cache) as reader:
        with pytest.raises(ValueError, match="presupuesto"):
            reader.compare("US/A", START, END, selected=1, max_mentions=1)
        with pytest.raises(ValueError, match="intervalo"):
            list(reader.batches("US/A", END, START))
        assert reader.compare("US/A", END, END, selected=1)["mean"] is None
        assert reader.cached_row_groups <= 2


def test_index_corruption_and_missing_read_only_cache_fail_without_creation(tmp_path):
    output, cache, _, _ = prepare(tmp_path)
    (output / "documents.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="huella|cambiado"):
        api().DocumentIndex(output / "manifest.json", cache_path=cache)
    missing = tmp_path / "missing.sqlite"
    with pytest.raises((ValueError, sqlite3.OperationalError)):
        EmbeddingCache(missing, read_only=True)
    assert not missing.exists()


def test_read_only_embedding_access_does_not_allow_writes_or_large_blobs(tmp_path):
    path = tmp_path / "cache.sqlite"
    cache = EmbeddingCache(path)
    cache.put({"encoder": "test"}, np.ones(10, dtype=np.float32))
    cache.close()
    before = path.read_bytes()
    cache = EmbeddingCache(path, read_only=True)
    try:
        with pytest.raises(ValueError, match="presupuesto"):
            cache.get({"encoder": "test"}, max_bytes=4)
        with pytest.raises(sqlite3.OperationalError):
            cache.put({"encoder": "test"}, np.zeros(10, dtype=np.float32))
        np.testing.assert_array_equal(cache.get({"encoder": "test"}), np.ones(10))
    finally:
        cache.close()
    assert path.read_bytes() == before


def test_query_limits_cover_other_assets_and_decoded_document_bytes(tmp_path):
    output, cache, _, _ = prepare(tmp_path)
    with api().DocumentIndex(output / "manifest.json", cache_path=cache) as reader:
        with pytest.raises(ValueError, match="presupuesto"):
            list(reader.batches("US/A", START, END, max_scan_rows=1))
        with pytest.raises(ValueError, match="presupuesto"):
            list(reader.batches("US/A", START, END, max_batch_bytes=100))
        all_rows = list(reader.batches("US/A", START, END))
        assert sum(len(b["documents"]) for b in all_rows) == 2


def test_cli_prepares_and_measures_the_same_window_without_changing_sources(tmp_path):
    sources, cache = fixture(tmp_path)
    before = {str(path): sha256(path) for path in sources.values()}
    output = tmp_path / "index"
    command = [sys.executable, "-m", "mars_titan.data.document_cli"]
    created = subprocess.run(
        command
        + [
            "prepare",
            "--source",
            f"US/A={sources['US/A']}",
            "--source",
            f"US/B={sources['US/B']}",
            "--cache",
            str(cache),
            "--encoder",
            ENCODER,
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(created.stdout)["mentions"] == 4
    measured = subprocess.run(
        command
        + [
            "compare",
            "--index",
            str(output / "manifest.json"),
            "--cache",
            str(cache),
            "--asset",
            "US/A",
            "--start",
            START.isoformat(),
            "--end",
            END.isoformat(),
            "--selected",
            "1",
            "--repeats",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(measured.stdout)
    assert report["comparison"]["mean"] == [2.0] * 384
    assert report["comparison"]["selected_mean"] == [3.0] * 384
    assert report["measurement"]["repetitions"] == 2
    assert report["measurement"]["peak_rss_bytes"] > 0
    assert report["measurement"]["wall_seconds"] > 0
    assert before == {str(path): sha256(path) for path in sources.values()}


def test_executable_example_uses_separate_synthetic_sources_and_shared_cache(tmp_path):
    output = tmp_path / "control"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/example_document_index.py",
            "--output",
            str(output),
            "--assets",
            "2",
            "--documents",
            "8",
            "--repeats",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["comparison"]["examined_mentions"] == 8
    assert report["comparison"]["encoded_documents"] == 0
    assert report["comparison"]["mean"] == report["comparison"]["selected_mean"]
    assert json.loads((output / "control.json").read_text())["historical_evidence"] is False


def test_damaged_staged_parquet_is_rejected_before_publishing_manifest(tmp_path, monkeypatch):
    sources, cache = fixture(tmp_path)
    output = tmp_path / "index"
    original = api().atomic_parquet_batches

    def damaged(path, tables):
        count = original(path, tables)
        path.write_bytes(b"truncated")
        return count

    monkeypatch.setattr(api(), "atomic_parquet_batches", damaged)
    with pytest.raises(ValueError, match="Parquet|regular"):
        api().prepare_document_index(sources, output, cache_path=cache, encoder_sha256=ENCODER)
    assert not output.exists()
