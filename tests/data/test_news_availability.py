"""Admisión y procedencia de noticias sin inferir horas ni vínculos ausentes."""

import hashlib
import json
from datetime import UTC, datetime

import pytest

from mars_titan.data.news import read_news
from mars_titan.data.temporal import MarketClock


@pytest.fixture
def clock():
    return MarketClock("US", "2024-01-01", "2025-01-01")


def write_news(tmp_path, rows, name="AAPL.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def row(date="2024-07-05", **extra):
    return {
        "Date": date,
        "Stock_symbol": "AAPL",
        "Article_title": "Resultados de Apple",
        "Article": "Apple presenta sus resultados.",
        "Url": "https://example.org/a",
        **extra,
    }


def test_filename_cannot_replace_missing_symbol_evidence(tmp_path, clock):
    source = row()
    source.pop("Stock_symbol")
    path = write_news(tmp_path, [source])
    accepted, rejected = read_news(path, "AAPL", clock)
    assert accepted == []
    assert rejected[0]["reason"] == "missing_symbol_evidence"
    assert rejected[0]["line"] == 1
    assert rejected[0]["source_file"] == str(path)
    assert (
        rejected[0]["source_record_hash"]
        == hashlib.sha256(path.read_text().rstrip("\n").encode()).hexdigest()
    )


@pytest.mark.parametrize("date", ["2024-11-03T01:30:00", "2024-07-05T20:01:00-00:00"])
def test_unknown_timezone_is_not_inferred_from_market(tmp_path, clock, date):
    path = write_news(tmp_path, [row(date)])
    accepted, rejected = read_news(path, "AAPL", clock)
    assert accepted == []
    assert rejected[0]["reason"] == "unverified_timezone"


def test_date_only_preserves_unknown_event_time_and_measures_lag_sensitivity(tmp_path, clock):
    path = write_news(tmp_path, [row(language="en")])
    one, _ = read_news(path, "AAPL", clock)
    two, _ = read_news(path, "AAPL", clock, date_only_lag=2)
    assert one[0]["event_at"] is None
    assert one[0]["source_date"] == "2024-07-05"
    assert one[0]["language"] == "en"
    assert one[0]["available_at"] == datetime(2024, 7, 8, 20, 5, tzinfo=UTC)
    assert two[0]["available_at"] == datetime(2024, 7, 9, 20, 5, tzinfo=UTC)
    assert one[0]["event_id"] == two[0]["event_id"]


@pytest.mark.parametrize("lag", [0, -1, True, 1.5])
def test_invalid_lag_fails_before_reading_source(tmp_path, clock, lag):
    with pytest.raises(ValueError, match="retardo"):
        read_news(tmp_path / "missing.jsonl", "AAPL", clock, date_only_lag=lag)


def test_timestamp_after_decision_is_not_visible_at_that_decision(tmp_path, clock):
    path = write_news(
        tmp_path, [row("2024-07-05T16:04:00-04:00"), row("2024-07-05T16:06:00-04:00")]
    )
    accepted, rejected = read_news(path, "AAPL", clock)
    assert rejected == []
    assert [r["event_at"] for r in accepted] == [
        datetime(2024, 7, 5, 20, 4, tzinfo=UTC),
        datetime(2024, 7, 5, 20, 6, tzinfo=UTC),
    ]
    assert sum(r["available_at"] <= clock.decision("2024-07-05") for r in accepted) == 1
    assert sum(r["available_at"] <= clock.decision("2024-07-08") for r in accepted) == 2


def test_explicit_offsets_resolve_repeated_local_hour_and_crossed_date(tmp_path, clock):
    path = write_news(
        tmp_path,
        [
            row("2024-11-03T01:30:00-04:00"),
            row("2024-11-03T01:30:00-05:00"),
            row("2024-07-06T02:00:00+09:00"),
        ],
    )
    accepted, rejected = read_news(path, "AAPL", clock)
    assert rejected == []
    assert [r["event_at"] for r in accepted] == [
        datetime(2024, 7, 5, 17, tzinfo=UTC),
        datetime(2024, 11, 3, 5, 30, tzinfo=UTC),
        datetime(2024, 11, 3, 6, 30, tzinfo=UTC),
    ]


def test_equivalent_timestamps_deduplicate_but_later_revision_stays_separate(tmp_path, clock):
    path = write_news(
        tmp_path,
        [
            row("2024-07-05T16:00:00-04:00"),
            row("2024-07-05T20:00:00Z"),
            row("2024-07-08T20:00:00Z", Article="Apple actualiza sus cifras."),
        ],
    )
    accepted, rejected = read_news(path, "AAPL", clock)
    assert len(accepted) == 2
    assert [r["reason"] for r in rejected] == ["duplicate"]
    assert accepted[0]["event_id"] != accepted[1]["event_id"]
    assert accepted[0]["content_hash"] != accepted[1]["content_hash"]
    original, _ = read_news(
        write_news(tmp_path, [row("2024-07-05T16:00:00-04:00")], "copy.jsonl"), "AAPL", clock
    )
    assert original[0]["event_id"] == accepted[0]["event_id"]
    assert original[0]["content_hash"] == accepted[0]["content_hash"]


def test_submicrosecond_timestamp_is_not_rounded_into_admission(tmp_path, clock):
    path = write_news(tmp_path, [row("2024-07-05T20:05:00.0000001Z")])
    accepted, rejected = read_news(path, "AAPL", clock)
    assert accepted == []
    assert rejected[0]["reason"] == "unsupported_timestamp_precision"


@pytest.mark.parametrize("offset", ["-00:00:00.0000001", "-00:00:00.5", "+00:00:30"])
def test_nonstandard_offsets_cannot_silently_change_publication_precision(tmp_path, clock, offset):
    path = write_news(tmp_path, [row(f"2024-07-05T20:05:00.000000{offset}")])
    accepted, rejected = read_news(path, "AAPL", clock)
    assert accepted == []
    assert rejected[0]["reason"] in {"unsupported_timestamp_precision", "unverified_timezone"}


def test_timestamp_without_seconds_is_not_promoted_to_exact_seconds(tmp_path, clock):
    path = write_news(tmp_path, [row("2024-07-05T20:05Z")])
    accepted, rejected = read_news(path, "AAPL", clock)
    assert accepted == []
    assert rejected[0]["reason"] == "unsupported_timestamp_precision"
