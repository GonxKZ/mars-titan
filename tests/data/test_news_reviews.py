"""Admisión de cuerpos completos contrastados, sin sustituirlos por titulares."""

import hashlib
import importlib
import json

import pytest

from mars_titan.data.news import read_news
from mars_titan.data.temporal import MarketClock


def fixture(tmp_path):
    raw = {
        "Date": "2024-07-05",
        "Stock_symbol": "A",
        "Article_title": "Resultados de A",
        "Article": "Artículo completo.\nPromoción.\nDeclaración del autor.",
        "Url": "https://example.org/article",
    }
    line = json.dumps(raw)
    source = tmp_path / "A.jsonl"
    source.write_text(line + "\n")
    digest = hashlib.sha256(line.encode()).hexdigest()
    body = raw["Article"]
    spans = [[0, body.index("\nPromoción")], [body.index("Declaración"), len(body)]]
    selected = "\n".join(body[start:end] for start, end in spans)
    review = {
        "source_record_hash": digest,
        "status": "verified_full_article",
        "symbol": "A",
        "source_date": raw["Date"],
        "source_url": raw["Url"],
        "evidence_url": "https://publisher.example.org/article",
        "checked_at": "2026-09-20T00:00:00+00:00",
        "body_spans": spans,
        "body_sha256": hashlib.sha256(selected.encode()).hexdigest(),
        "historical_version_verified": False,
        "note": "Cuerpo completo y declaración contrastados, promoción excluida.",
    }
    return source, raw, review, selected


def test_only_reviewed_full_bodies_are_admitted_without_changing_the_source(tmp_path):
    source, raw, review, body = fixture(tmp_path)
    original = source.read_bytes()
    clock = MarketClock("US", "2024-01-01", "2025-01-01")
    rows, excluded = read_news(source, "A", clock, reviews={})
    assert not rows
    assert excluded[0]["reason"] == "content_unreviewed"
    rows, excluded = read_news(source, "A", clock, reviews={review["source_record_hash"]: review})
    assert not excluded
    assert rows[0]["text"] == raw["Article_title"] + "\n" + body
    assert rows[0]["content_review"] == "verified_full_article"
    assert rows[0]["historical_body_version_verified"] is False
    assert rows[0]["available_at"] == clock.decision("2024-07-08")
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "status,reason", [("rejected", "content_rejected"), ("unverifiable", "content_unverifiable")]
)
def test_rejected_or_unverifiable_bodies_do_not_enter_strict_data(tmp_path, status, reason):
    source, _, review, _ = fixture(tmp_path)
    review["status"] = status
    rows, excluded = read_news(
        source,
        "A",
        MarketClock("US", "2024-01-01", "2025-01-01"),
        reviews={review["source_record_hash"]: review},
    )
    assert rows == []
    assert excluded[0]["reason"] == reason


@pytest.mark.parametrize(
    "field,value",
    [
        ("body_sha256", "0" * 64),
        ("body_spans", [[0, 9999]]),
        ("body_spans", [[10, 20], [0, 5]]),
        ("symbol", "B"),
        ("source_date", "2024-07-04"),
        ("source_url", "https://example.org/another"),
    ],
)
def test_review_cannot_be_reused_for_different_content_or_provenance(tmp_path, field, value):
    source, _, review, _ = fixture(tmp_path)
    review[field] = value
    rows, excluded = read_news(
        source,
        "A",
        MarketClock("US", "2024-01-01", "2025-01-01"),
        reviews={review["source_record_hash"]: review},
    )
    assert not rows
    assert excluded[0]["reason"] == "content_review_mismatch"


def test_manifest_rejects_duplicate_reviews_and_missing_evidence(tmp_path):
    _, _, review, _ = fixture(tmp_path)
    try:
        load = importlib.import_module("mars_titan.data.news_reviews").load_reviews
    except ModuleNotFoundError:
        pytest.fail("Falta la lectura validada de revisiones de noticias")
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps({"schema_version": 1, "reviews": [review, review]}))
    with pytest.raises(ValueError):
        load(path)
    review["evidence_url"] = ""
    path.write_text(json.dumps({"schema_version": 1, "reviews": [review]}))
    with pytest.raises(ValueError):
        load(path)


def test_strict_audit_keeps_review_metadata_in_typed_parquet(tmp_path):
    import pyarrow.parquet as pq

    from mars_titan.data.news_audit import audit_news_panel

    source = tmp_path / "source"
    source.mkdir()
    _, _, review, _ = fixture(source)
    panel = {"market": "US", "assets": [{"symbol": "A", "paths": {"news": ["A.jsonl"]}}]}
    report = audit_news_panel(
        source,
        panel,
        tmp_path / "derived",
        MarketClock("US", "2024-01-01", "2025-01-01"),
        reviews={review["source_record_hash"]: review},
    )
    assert report["admission_scope"] == "verified_full_editorial_content"
    assert report["admitted_content_reviewed"] is True
    assert report["accepted"] == 1
    assert report["historical_body_version_verified"] is False
    table = pq.read_table(tmp_path / "derived/US/A/news.parquet")
    assert table["content_review"].to_pylist() == ["verified_full_article"]
    assert table["reviewed_body_sha256"].to_pylist() == [review["body_sha256"]]


def test_preparation_with_empty_review_registry_cannot_keep_unreviewed_news(tmp_path):
    from mars_titan.data.preparation import prepare_asset

    source = tmp_path / "source"
    source.mkdir()
    fixture(source)
    (source / "A.csv").write_text("Date,Open,High,Low,Close,Volume\n2024-07-08,10,12,9,11,1\n")
    (source / "facts.json").write_text('{"filings":[]}')
    asset = {
        "symbol": "A",
        "paths": {"prices": ["A.csv"], "news": ["A.jsonl"], "fundamentals": ["facts.json"]},
    }
    result = prepare_asset(
        source,
        tmp_path / "prepared",
        asset,
        MarketClock("US", "2024-01-01", "2025-01-01"),
        news_reviews={},
    )
    assert result["news_content_policy"] == "verified_full_articles"
    assert result["counts"]["news"] == 0
    assert result["news_rejections"][0]["reason"] == "content_unreviewed"
    assert result["training_ready"] is False


@pytest.mark.parametrize("body", [None, "", "   "])
def test_title_or_summary_cannot_substitute_for_a_full_article(tmp_path, body):
    source, raw, review, _ = fixture(tmp_path)
    raw.update(Article=body, summary="Resumen disponible")
    line = json.dumps(raw)
    source.write_text(line + "\n")
    review["source_record_hash"] = hashlib.sha256(line.encode()).hexdigest()
    rows, rejected = read_news(
        source,
        "A",
        MarketClock("US", "2024-01-01", "2025-01-01"),
        reviews={review["source_record_hash"]: review},
    )
    assert not rows
    assert rejected[0]["reason"] == "missing_full_article"


def test_prepare_cli_uses_verified_review_registry_by_default(tmp_path, monkeypatch):
    import sys

    from mars_titan.data import cli, news_reviews, preparation

    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"market": "US", "assets": [{"symbol": "A"}]}))
    calls = []
    registry = {"review": "synthetic_wiring_fixture"}

    def load(path):
        calls.append(path.as_posix())
        return registry

    def prepare(source, destination, asset, clock, *, news_reviews=None):
        assert news_reviews is registry
        return {"symbol": "A", "counts": {}, "elapsed_seconds": 0, "reused": False}

    monkeypatch.setattr(news_reviews, "load_reviews", load)
    monkeypatch.setattr(preparation, "prepare_asset", prepare)
    monkeypatch.setattr(sys, "argv", ["mars-data", "prepare", "--panel", str(panel)])
    cli.main()
    assert calls == ["data/manifests/news-reviews.json"]


def test_reviewed_slice_cannot_reduce_a_full_body_to_whitespace(tmp_path):
    source, raw, review, _ = fixture(tmp_path)
    position = raw["Article"].index("\n")
    review["body_spans"] = [[position, position + 1]]
    review["body_sha256"] = hashlib.sha256(b"\n").hexdigest()
    rows, rejected = read_news(
        source,
        "A",
        MarketClock("US", "2024-01-01", "2025-01-01"),
        reviews={review["source_record_hash"]: review},
    )
    assert not rows
    assert rejected[0]["reason"] == "content_review_mismatch"


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_record_hash", "not-a-hash"),
        ("status", "unknown"),
        ("note", ""),
        ("checked_at", "2026-09-20T00:00:00"),
        ("evidence_url", "file:///tmp/article"),
        ("body_sha256", "invalid"),
        ("historical_version_verified", "false"),
        ("body_spans", []),
        ("body_spans", [[False, 10]]),
        ("body_spans", [[0, 10], [5, 12]]),
    ],
)
def test_invalid_review_configuration_fails_before_scanning_news(tmp_path, field, value):
    from mars_titan.data.news_reviews import load_reviews

    _, _, review, _ = fixture(tmp_path)
    review[field] = value
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps({"schema_version": 1, "reviews": [review]}))
    with pytest.raises(ValueError):
        load_reviews(path)
