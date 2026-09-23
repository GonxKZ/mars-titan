"""Cohortes editoriales distintas con registros y salidas comprobables."""

import hashlib
import importlib
import json

import pyarrow.parquet as pq
import pytest

from mars_titan.data.news import read_news
from mars_titan.data.temporal import MarketClock
from tests.data.test_news_reviews import fixture


def write(*args, **kwargs):
    try:
        module = importlib.import_module("mars_titan.data.cohort_news")
    except ModuleNotFoundError:
        pytest.fail("Falta la preparación de noticias por cohorte")
    return module.write_cohort_news(*args, **kwargs)


def source_rows(tmp_path, rows):
    root = tmp_path / "source"
    root.mkdir(exist_ok=True)
    path = root / "A.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return root


def article(day="2023-07-05", **changes):
    return dict(
        Date=day,
        Stock_symbol="A",
        Article_title="Resultados de A",
        Article="La empresa A publica sus resultados.",
        Url="https://example.org/a",
        **changes,
    )


def options(cohort="original_audited"):
    return dict(
        symbol="A",
        clock=MarketClock("US", "2023-01-01", "2025-01-01"),
        cohort=cohort,
        cutoff="2023-12-31",
    )


def test_original_and_external_cohorts_do_not_share_admission_claims(tmp_path):
    source = source_rows(tmp_path, [article()])
    raw = (source / "A.jsonl").read_bytes()
    original = write(source, ["A.jsonl"], tmp_path / "original", **options())
    strict = write(
        source, ["A.jsonl"], tmp_path / "strict", **options("externally_verified"), reviews={}
    )
    assert original["counts"] == {"records": 1, "accepted": 1, "excluded": 0}
    assert strict["counts"] == {"records": 1, "accepted": 0, "excluded": 1}
    row = pq.read_table(tmp_path / "original/news.parquet").to_pylist()[0]
    assert row["content_review"] == "source_audited_not_external"
    assert row["content_kind"] == "article_candidate"
    assert row["available_at"] == options()["clock"].decision("2023-07-06")
    assert row["published_at"] is None
    assert (source / "A.jsonl").read_bytes() == raw


def test_strict_content_preserves_the_existing_reviewed_reader(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path, _, review, _ = fixture(source)
    clock = MarketClock("US", "2024-01-01", "2025-01-01")
    reviews = {review["source_record_hash"]: review}
    expected, errors = read_news(path, "A", clock, reviews=reviews)
    assert not errors
    write(
        source,
        [path.name],
        tmp_path / "output",
        symbol="A",
        clock=clock,
        cohort="externally_verified",
        cutoff="2024-12-31",
        reviews=reviews,
    )
    actual = pq.read_table(tmp_path / "output/news.parquet").to_pylist()[0]
    for name in (
        "text",
        "event_id",
        "content_hash",
        "available_at",
        "published_at",
        "content_review",
    ):
        assert actual[name] == expected[0][name]


def test_original_cannot_restore_a_confirmed_rejection(tmp_path):
    source = source_rows(tmp_path, [article()])
    raw = (source / "A.jsonl").read_text().rstrip("\r\n")
    review = dict(
        status="rejected",
        symbol="A",
        source_date="2023-07-05",
        source_url="https://example.org/a",
        note="Contenido ajeno comprobado",
    )
    report = write(
        source,
        ["A.jsonl"],
        tmp_path / "output",
        **options(),
        reviews={hashlib.sha256(raw.encode()).hexdigest(): review},
    )
    assert report["counts"]["accepted"] == 0
    assert report["reasons"]["content_rejected"] == 1


def test_summary_and_unzoned_time_remain_explicit_in_original_cohort(tmp_path):
    source = source_rows(
        tmp_path, [dict(datetime="2023-07-05 10:30:00", title="Resumen", summary="Texto original")]
    )
    report = write(source, ["A.jsonl"], tmp_path / "output", **options())
    row = pq.read_table(tmp_path / "output/news.parquet").to_pylist()[0]
    assert report["counts"]["accepted"] == 1
    assert row["content_kind"] == "summary"
    assert row["association_evidence"] == "inventory_file_assignment"
    assert row["published_at"] is None
    assert row["source_date"] == "2023-07-05 10:30:00"
    assert row["availability_rule"] == "declared_date_without_verified_timezone"


def test_duplicates_and_late_records_are_counted_without_unbounded_lists(tmp_path):
    source = source_rows(
        tmp_path, [article("2023-07-06"), article(), article(), article("2024-01-02")]
    )
    report = write(source, ["A.jsonl"], tmp_path / "output", **options(), batch_rows=1)
    assert report["counts"] == {"records": 4, "accepted": 2, "excluded": 2}
    assert report["reasons"] == {"duplicate": 1, "reserved": 1}
    rows = pq.read_table(tmp_path / "output/news.parquet").to_pylist()
    assert [r["source_date"] for r in rows] == ["2023-07-05", "2023-07-06"]
    assert "news_rejections" not in report
    assert pq.read_table(tmp_path / "output/excluded.parquet").num_rows == 2


def test_changed_source_or_cohort_cannot_reuse_a_confirmed_output(tmp_path):
    source = source_rows(tmp_path, [article()])
    output = tmp_path / "output"
    first = write(source, ["A.jsonl"], output, **options())
    assert write(source, ["A.jsonl"], output, **options())["reused"] is True
    assert first["reused"] is False
    with pytest.raises(ValueError, match="edición|configuración|identidad"):
        write(source, ["A.jsonl"], output, **options("externally_verified"), reviews={})
    with (source / "A.jsonl").open("a") as stream:
        stream.write(json.dumps(article("2023-07-06")) + "\n")
    with pytest.raises(ValueError, match="edición|configuración|identidad"):
        write(source, ["A.jsonl"], output, **options())


def test_oversized_record_is_quarantined_and_next_record_is_processed(tmp_path):
    source = source_rows(tmp_path, [dict(Article="x" * 4096), article()])
    report = write(source, ["A.jsonl"], tmp_path / "output", **options(), max_record_bytes=1024)
    assert report["counts"] == {"records": 2, "accepted": 1, "excluded": 1}
    assert report["reasons"] == {"record_too_large": 1}
    assert pq.read_table(tmp_path / "output/news.parquet")["line"].to_pylist() == [2]


def test_invalid_cohort_and_source_paths_fail_before_reading(tmp_path):
    source = source_rows(tmp_path, [article()])
    with pytest.raises(ValueError, match="cohorte"):
        write(source, ["A.jsonl"], tmp_path / "bad", **options("anything"))
    with pytest.raises(ValueError, match="origen|ruta"):
        write(source, ["../elsewhere.jsonl"], tmp_path / "bad", **options())
    assert not (tmp_path / "bad").exists()


def test_incomplete_receipt_cannot_claim_a_reusable_artifact(tmp_path):
    source = source_rows(tmp_path, [article()])
    output = tmp_path / "output"
    write(source, ["A.jsonl"], output, **options())
    path = output / "manifest.json"
    receipt = json.loads(path.read_text())
    receipt["artifacts"] = {}
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="recibo"):
        write(source, ["A.jsonl"], output, **options())


def test_symbol_mismatch_and_malformed_offset_are_never_inferred(tmp_path):
    wrong = article()
    wrong["Stock_symbol"] = "B"
    source = source_rows(tmp_path, [wrong, article("2023-07-05T10:30:00+24:00")])
    receipt = write(source, ["A.jsonl"], tmp_path / "output", **options())
    assert receipt["counts"]["accepted"] == 0
    assert receipt["reasons"] == {"symbol_mismatch": 1, "invalid_record": 1}


def test_symlinked_receipt_is_not_read_or_overwritten(tmp_path):
    source = source_rows(tmp_path, [article()])
    output = tmp_path / "output"
    output.mkdir()
    other = tmp_path / "private.json"
    other.write_text("{}")
    (output / "configuration.json").symlink_to(other)
    with pytest.raises(ValueError, match="enlace"):
        write(source, ["A.jsonl"], output, **options())
    assert other.read_text() == "{}"


def test_declared_day_before_cutoff_does_not_admit_later_availability(tmp_path):
    source = source_rows(tmp_path, [article("2023-12-29")])
    report = write(source, ["A.jsonl"], tmp_path / "output", **options())
    assert report["counts"]["accepted"] == 0
    assert report["reasons"] == {"availability_after_cutoff": 1}


def test_second_read_must_match_the_indexed_raw_bytes(tmp_path, monkeypatch):
    source = source_rows(tmp_path, [article()])
    module = importlib.import_module("mars_titan.data.cohort_news")
    original = module._read_source

    def inconsistent(*args, **kwargs):
        for item in original(*args, **kwargs):
            changed = list(item)
            changed[3] = bytes(32)
            yield tuple(changed)

    monkeypatch.setattr(module, "_read_source", inconsistent)
    with pytest.raises(ValueError, match="registro|lecturas|huella"):
        write(source, ["A.jsonl"], tmp_path / "output", **options())
    assert not (tmp_path / "output/manifest.json").exists()
