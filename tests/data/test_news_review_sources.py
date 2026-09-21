"""Comprobar que cada revisión corresponde al registro original completo."""

import hashlib
import json

import pytest

from mars_titan.data import news_reviews


def source_and_reviews(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    raw = {
        "Article": "Artículo completo de prueba.",
        "Date": "2023-06-01",
        "Stock_symbol": "A",
        "Url": "https://example.org/a",
    }
    line = json.dumps(raw)
    (source / "A.jsonl").write_text(line + "\n")
    digest = hashlib.sha256(line.encode()).hexdigest()
    review = {
        "source_record_hash": digest,
        "source_file": "A.jsonl",
        "source_row": 1,
        "symbol": "A",
        "source_date": raw["Date"],
        "source_url": raw["Url"],
        "status": "verified_full_article",
        "note": "Caso sintético.",
        "checked_at": "2026-09-21T00:00:00+00:00",
        "evidence_url": "https://example.org/source",
        "body_spans": [[0, len(raw["Article"])]],
        "body_sha256": hashlib.sha256(raw["Article"].encode()).hexdigest(),
        "historical_version_verified": False,
    }
    return source, {digest: review}


def checker():
    check = getattr(news_reviews, "check_review_sources", None)
    assert callable(check), "Falta comprobar las revisiones contra sus registros originales"
    return check


def test_review_source_check_reconciles_status_without_changing_input(tmp_path):
    source, reviews = source_and_reviews(tmp_path)
    original = (source / "A.jsonl").read_bytes()
    report = checker()(source, reviews)
    assert report["reviews"] == 1
    assert report["statuses"] == {"verified_full_article": 1}
    assert report["by_asset_year"] == [
        {"symbol": "A", "year": 2023, "status": "verified_full_article", "records": 1}
    ]
    assert report["source_hashes"]["A.jsonl"] == hashlib.sha256(original).hexdigest()
    assert (source / "A.jsonl").read_bytes() == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_row", 2),
        ("source_row", True),
        ("symbol", "B"),
        ("source_file", "../A.jsonl"),
        ("body_sha256", "0" * 64),
    ],
)
def test_invalid_review_provenance_fails_instead_of_counting_verification(tmp_path, field, value):
    source, reviews = source_and_reviews(tmp_path)
    next(iter(reviews.values()))[field] = value
    with pytest.raises(ValueError):
        checker()(source, reviews)


def test_changed_original_is_not_covered_by_old_review(tmp_path):
    source, reviews = source_and_reviews(tmp_path)
    (source / "A.jsonl").write_text('{"Article":"otro"}\n')
    with pytest.raises(ValueError):
        checker()(source, reviews)


def test_line_change_invalidates_review_even_when_body_and_metadata_stay_equal(tmp_path):
    source, reviews = source_and_reviews(tmp_path)
    path = source / "A.jsonl"
    row = json.loads(path.read_text())
    row["Article_title"] = "Un título añadido después de revisar"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="registro original"):
        checker()(source, reviews)
