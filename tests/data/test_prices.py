"""Exclusiones localizables y hechos corporativos conservados sin reparación."""

import pandas as pd
import pytest

from mars_titan.data.prices import read_prices
from mars_titan.data.temporal import MarketClock


@pytest.fixture
def clock():
    return MarketClock("US", "2023-01-01", "2025-01-01")


def test_price_details_reconcile_overlapping_reasons_and_preserve_source_rows(tmp_path, clock):
    path = tmp_path / "A.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2024-07-01,10,12,9,11,0\n"
        "2024-07-02,10,12,9,11,1\n"
        "2024-07-02,10,9,8,11,-1\n"
        "2024-07-04,10,12,9,11,1\n"
        ",10,12,9,11,1\n"
    )
    prices, audit = read_prices(path, clock, include_details=True)
    assert list(prices["session"]) == ["2024-07-01"]
    assert audit["rows"] == audit["accepted"] + audit["excluded"] == 5
    excluded = audit["details"]["exclusions"]
    assert [row["source_row"] for row in excluded] == [2, 3, 4, 5]
    assert excluded[1]["reasons"] == ["duplicate_session", "invalid_ohlc"]
    assert excluded[2]["reasons"] == ["non_session"]
    assert excluded[3]["reasons"] == ["invalid_date"]
    assert excluded[3]["source_date"] is None
    assert audit["details"]["coverage"][0] == {
        "year": 2024,
        "first_observed_session": "2024-07-01",
        "last_observed_session": "2024-07-04",
        "observed_rows": 4,
        "accepted_rows": 1,
        "excluded_rows": 3,
        "expected_sessions_within_observed_span": 3,
        "absent_sessions_within_observed_span": 1,
    }


def test_split_and_dividend_are_evidence_not_instructions_to_readjust_prices(tmp_path, clock):
    path = tmp_path / "A.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
        "2024-07-01,10,12,9,11,10,0,2\n"
        "2024-07-02,10,12,9,11,10,0.5,0\n"
        "2024-07-03,10,12,9,11,10,unknown,0\n"
    )
    prices, audit = read_prices(path, clock, include_details=True)
    assert prices["open"].to_list() == [10, 10, 10]
    assert prices["close"].to_list() == [11, 11, 11]
    actions = audit["details"]["corporate_actions"]
    assert [row["source_row"] for row in actions] == [1, 2, 3]
    assert actions[0]["stock_splits"] == "2"
    assert actions[1]["dividends"] == "0.5"
    assert actions[2]["values_valid"] is False
    assert actions[2]["dividends"] == "unknown"
    assert audit["corporate_action_columns"] == ["Dividends", "Stock Splits"]
    assert audit["accepted"] == 3


def test_detailed_inspection_matches_normal_reader_without_filling_missing_sessions(
    tmp_path, clock
):
    path = tmp_path / "A.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume\n2024-07-01,10,12,9,11,1\n2024-07-03,10,12,9,11,1\n"
    )
    plain, _ = read_prices(path, clock)
    detailed, audit = read_prices(path, clock, include_details=True)
    pd.testing.assert_frame_equal(plain, detailed)
    assert len(detailed) == 2
    assert audit["details"]["coverage"][0]["absent_sessions_within_observed_span"] == 1
    assert audit["corporate_action_columns"] == []
    assert audit["details"]["corporate_actions"] == []


@pytest.mark.parametrize("rows", ["", ",10,12,9,11,1\n", "invalid,10,12,9,11,1\n"])
def test_empty_or_unusable_dates_have_no_invented_coverage(tmp_path, clock, rows):
    path = tmp_path / "A.csv"
    path.write_text("Date,Open,High,Low,Close,Volume\n" + rows)
    prices, audit = read_prices(path, clock, include_details=True)
    assert prices.empty
    assert audit["first_session"] is None
    assert audit["last_session"] is None
    assert audit["details"]["coverage"] == []
    assert audit["invalid_date_rows"] == audit["rows"]


def test_noncanonical_and_missing_dates_do_not_invent_sessions_or_duplicates(tmp_path, clock):
    path = tmp_path / "A.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2024-1-2,10,12,9,11,1\n"
        "2024-01-03,10,12,9,11,1\n"
        ",10,12,9,11,1\n"
        ",10,12,9,11,1\n"
    )
    prices, audit = read_prices(path, clock, include_details=True)
    assert prices["session"].to_list() == ["2024-01-03"]
    assert audit["first_session"] == audit["last_session"] == "2024-01-03"
    assert audit["invalid_date_rows"] == 3
    assert audit["duplicate_session_rows"] == 0
    assert audit["non_session_rows"] == 0
    assert [row["reasons"] for row in audit["details"]["exclusions"]] == [["invalid_date"]] * 3
    assert audit["details"]["coverage"][0]["expected_sessions_within_observed_span"] == 1


def test_non_session_aggregate_matches_exclusion_reasons_with_unknown_dates(tmp_path, clock):
    path = tmp_path / "A.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume\n2024-07-04,10,12,9,11,1\n,10,12,9,11,1\n,10,12,9,11,1\n"
    )
    _, audit = read_prices(path, clock, include_details=True)
    assert (
        audit["non_session_rows"]
        == sum("non_session" in row["reasons"] for row in audit["details"]["exclusions"])
        == 1
    )
    assert audit["duplicate_session_rows"] == 0
