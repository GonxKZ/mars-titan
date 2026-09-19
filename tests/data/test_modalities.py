"""Verificar adaptadores de fuentes reales con ejemplos pequeños explícitos."""

import importlib
import json
from datetime import UTC, datetime

import pytest

from mars_titan.data.temporal import MarketClock


def module(name):
    try:
        return importlib.import_module(f"mars_titan.data.{name}")
    except ModuleNotFoundError:
        pytest.fail(f"Adaptador pendiente: {name}")


@pytest.fixture
def clock():
    return MarketClock("US", "2023-01-01", "2025-12-31")


def test_price_audit_rejects_impossible_ohlc_and_duplicate_sessions(tmp_path, clock):
    path = tmp_path / "aapl.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
        "2024-07-01,10,12,9,11,0,0,0\n"
        "2024-07-02,10,9,8,11,100,0,0\n"
        "2024-07-03,10,12,9,11,100,0,0\n"
        "2024-07-03,10,12,9,12,100,0,0\n"
    )
    frame, audit = module("prices").read_prices(path, clock)
    assert list(frame["session"]) == ["2024-07-01"]
    assert frame.iloc[0]["volume"] == 0
    assert audit["rows"] == 4
    assert audit["excluded"] == 3
    assert audit["invalid_ohlc"] == 1
    assert audit["duplicate_session_rows"] == 2


def test_news_uses_symbol_metadata_and_lag_then_deduplicates(tmp_path, clock):
    path = tmp_path / "AAPL.jsonl"
    rows = [
        {
            "Date": "2024-07-04",
            "Article": "Apple reports its results.",
            "Url": "https://example.org/a",
            "Stock_symbol": "aapl",
        },
        {
            "Date": "2024-07-04",
            "Article": "Apple reports its results.",
            "Url": "https://example.org/a",
            "Stock_symbol": "aapl",
        },
        {
            "Date": "2024-07-04",
            "Article": "Unrelated company.",
            "Url": "https://example.org/b",
            "Stock_symbol": "msft",
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    accepted, rejected = module("news").read_news(path, "AAPL", clock)
    assert len(accepted) == 1
    assert accepted[0]["available_at"] == datetime(2024, 7, 5, 20, 5, tzinfo=UTC)
    assert accepted[0]["published_at"] is None
    assert sorted(r["reason"] for r in rejected) == ["duplicate", "symbol_mismatch"]


def test_news_invalid_nan_and_empty_text_are_explicit(tmp_path, clock):
    path = tmp_path / "AAPL.jsonl"
    path.write_text('{"Date":"2024-07-04","Article":NaN,"Stock_symbol":"aapl"}\n')
    rows, rejected = module("news").read_news(path, "AAPL", clock)
    assert rows == []
    assert rejected[0]["reason"] == "missing_text"


def test_fundamentals_use_inner_filed_preserve_revisions_and_deduplicate(tmp_path, clock):
    facts = {
        "us-gaap": {
            "Assets": {
                "units": {
                    "USD": [
                        {"end": "2024-03-31", "val": 100, "filed": "2024-05-02", "accn": "one"},
                        {"end": "2024-03-31", "val": 105, "filed": "2024-07-05", "accn": "two"},
                        {"end": "2024-06-30", "val": 110, "accn": "unknown"},
                    ]
                }
            }
        }
    }
    path = tmp_path / "facts.json"
    path.write_text(
        json.dumps(
            {
                "cik": "123",
                "filings": [
                    {"filing_date": "2000-01-01", "facts": facts},
                    {"filing_date": "2025-01-01", "facts": facts},
                ],
            }
        )
    )
    rows, audit = module("fundamentals").read_fundamentals([path], "US", clock)
    assert len(rows) == 2
    assert audit["missing_publication"] == 2
    assert audit["duplicates"] == 2
    assert {r["value"] for r in rows} == {100, 105}
    assert rows[1]["available_at"] == datetime(2024, 7, 8, 20, 5, tzinfo=UTC)
    snapshot = module("fundamentals").snapshot(rows, datetime(2024, 6, 1, tzinfo=UTC))
    assert snapshot["us-gaap:Assets:USD"]["value"] == 100


def test_china_period_end_never_becomes_publication(tmp_path):
    path = tmp_path / "income.jsonl"
    path.write_text('[{"ts_code":"000001.SZ","end_date":"20240331","revenue":100}]')
    clock = MarketClock("CN", "2023-01-01", "2025-12-31")
    rows, audit = module("fundamentals").read_fundamentals([path], "CN", clock)
    assert rows == []
    assert audit["missing_publication"] == 1


def test_conflicting_fact_with_same_identity_is_not_arbitrarily_selected(tmp_path, clock):
    facts = [
        {"end": "2024-03-31", "val": v, "filed": "2024-05-02", "accn": "same"} for v in [100, 101]
    ]
    path = tmp_path / "facts.json"
    path.write_text(
        json.dumps({"filings": [{"facts": {"us-gaap": {"Assets": {"units": {"USD": facts}}}}}]})
    )
    rows, audit = module("fundamentals").read_fundamentals([path], "US", clock)
    assert rows == []
    assert audit["ambiguous_facts"] == 1
