"""Admisión limitada a conceptos y documentos oficiales identificados."""

import csv
from pathlib import Path

import pytest

from mars_titan.data.macro import _exclusion, calculate_macro
from mars_titan.data.temporal import MarketClock


def entry():
    with Path("data/catalogs/macro-indicators.csv").open() as stream:
        row = next(r for r in csv.DictReader(stream) if r["id"] == "cn_cpi_yoy_published")
    row.update(
        vintage_policy="DATED_OFFICIAL_RELEASES",
        verification_status="verified_official_release_archive",
        availability_rule="OFFICIAL_RELEASE_BOUND_THEN_NEXT_SESSION",
    )
    return row


def observation():
    return dict(
        indicator_id="cn_cpi_yoy_published",
        period_start="2023-01-01",
        reference_period_end="2023-01-31",
        value=2.1,
        realtime_start="2023-02-13",
        realtime_end="9999-12-31",
        declared_publication_date="2023-02-11",
        archive_path_date="2023-02-13",
        source_timezone="Asia/Shanghai",
        source_hash="b" * 64,
        source_url="https://www.stats.gov.cn/english/PressRelease/202302/t20230213_1902713.html",
        source_title="Consumer Prices for January 2023",
        availability_policy="latest_declared_or_archive_day_then_next_session",
        publication_timestamp_verified=False,
        native_unit="percent_yoy",
        seasonal_adjustment="published_yoy",
        missing_reason=None,
    )


def test_exact_release_contract_admits_concept_without_inventing_api_identifier():
    assert entry()["series_id"] == "no identifier verified"
    assert _exclusion(entry()) is None
    clock = MarketClock("US", "2023-02-10", "2023-02-15")
    rows = calculate_macro([observation()], [entry()], clock)
    assert [r["value"] for r in rows] == [None, None, 2.1, 2.1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "unknown"),
        ("provider", "other"),
        ("frequency", "Q"),
        ("unit", "index"),
        ("series_id", "invented"),
        ("source_url", "https://example.org"),
        ("verification_status", "unchecked"),
    ],
)
def test_changed_identity_cannot_bypass_catalog_gate(field, value):
    row = entry()
    row[field] = value
    assert _exclusion(row) is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("realtime_start", "2023-02-10"),
        ("declared_publication_date", "2023-01-15"),
        ("source_timezone", "UTC"),
        ("publication_timestamp_verified", True),
        ("native_unit", "index"),
        ("source_url", "https://example.org"),
        ("archive_path_date", "2023-02-12"),
        ("reference_period_end", "2023-01-30"),
    ],
)
def test_release_record_cannot_invent_date_precision_unit_or_source(field, value):
    row = observation()
    row[field] = value
    with pytest.raises(ValueError):
        calculate_macro([row], [entry()], MarketClock("US", "2023-02-10", "2023-02-15"))
