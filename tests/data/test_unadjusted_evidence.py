"""Lectores de evidencia externa y contrastes de cierres y acciones corporativas."""

import json

import numpy as np
import pytest

from mars_titan.data.unadjusted_evidence import (
    close_tolerance,
    compare_closes,
    compare_events,
    read_alphavantage_daily,
    read_alphavantage_dividends,
    read_alphavantage_splits,
    read_eodhd_daily,
    read_eodhd_dividends,
    read_eodhd_splits,
    read_sse_daily,
)


def test_sse_reader_requires_the_declared_complete_series():
    body = {
        "code": "600000",
        "total": 2,
        "kline": [[20230104, 7, 7.2, 6.9, 7.01, 10], [20230103, 7, 7.1, 6.9, 7.0, 9]],
    }
    sessions, close = read_sse_daily(json.dumps(body).encode())
    assert sessions.astype(str).tolist() == ["2023-01-03", "2023-01-04"]
    np.testing.assert_allclose(close, [7.0, 7.01])
    body["total"] = 3
    with pytest.raises(ValueError, match="completa"):
        read_sse_daily(json.dumps(body).encode())
    body = {"total": 2, "kline": [[20230103, 7, 7, 7, 7.0, 1], [20230103, 7, 7, 7, 7.0, 1]]}
    with pytest.raises(ValueError, match="repite"):
        read_sse_daily(json.dumps(body).encode())


def test_vendor_readers_reject_non_positive_values_and_parse_ratios():
    eod = [{"date": "2020-08-28", "close": 499.23}, {"date": "2020-08-31", "close": 129.04}]
    sessions, close = read_eodhd_daily(json.dumps(eod).encode())
    assert close.tolist() == [499.23, 129.04]
    with pytest.raises(ValueError, match="no positivos"):
        read_eodhd_daily(json.dumps([{"date": "2020-08-28", "close": 0}]).encode())
    daily = {"Time Series (Daily)": {"2020-01-03": {"4. close": "134.34"}}}
    assert read_alphavantage_daily(json.dumps(daily).encode())[1].tolist() == [134.34]
    with pytest.raises(ValueError, match="serie diaria"):
        read_alphavantage_daily(json.dumps({"Note": "limit"}).encode())
    splits = [
        {"date": "2020-08-31", "split": "4.000000/1.000000"},
        {"date": "2023-01-03", "split": "1/20"},
    ]
    np.testing.assert_allclose(read_eodhd_splits(json.dumps(splits).encode())[1], [4.0, 0.05])
    dividends = [{"date": "2012-08-09", "value": 0.0946, "unadjustedValue": 2.65}]
    assert read_eodhd_dividends(json.dumps(dividends).encode())[1].tolist() == [2.65]
    av = {"data": [{"ex_dividend_date": "2023-11-09", "amount": "1.66"}]}
    assert read_alphavantage_dividends(json.dumps(av).encode())[1].tolist() == [1.66]
    av = {"data": [{"effective_date": "1999-05-27", "split_factor": "2.0000"}]}
    assert read_alphavantage_splits(json.dumps(av).encode())[1].tolist() == [2.0]


def test_close_comparison_separates_exact_rounded_and_mismatched_sessions():
    sessions = np.array(["2000-01-03", "2000-01-04", "2000-01-05", "2024-01-02"], "datetime64[D]")
    ours = np.array([111.9375, 102.5, np.nan, 50.0])
    reference = (
        np.array(
            ["2000-01-03", "2000-01-04", "2000-01-05", "2000-01-06", "2024-01-02"], "datetime64[D]"
        ),
        np.array([111.94, 102.51, 99.0, 98.0, 50.0]),
    )
    result = compare_closes(sessions, ours, *reference, start="2000-01-01", end="2023-12-31")
    assert result["common_sessions"] == 3 and result["defined"] == 2
    assert result["exact"] == 0 and result["within_half_tick"] == 1
    assert result["mismatch_sessions"] == ["2000-01-04"]
    assert result["reference_only_sessions"] == 1 and result["provider_only_sessions"] == 0
    np.testing.assert_allclose(close_tolerance([0.5, 10.0]), [0.00005 + 1e-6, 0.005 + 2e-5])


def test_event_comparison_reports_both_sides_and_value_mismatches():
    result = compare_events(
        ["2012-08-09", "2012-11-07", "2013-02-07"],
        [2.65, 2.65, 2.65],
        ["2012-08-09", "2012-11-07", "2013-05-09"],
        [2.65, 2.60, 3.05],
        start="2012-01-01",
        end="2013-03-01",
        rtol=1e-4,
    )
    assert result["same_date"] == 2 and result["same_value"] == 1
    assert result["value_mismatch_dates"] == ["2012-11-07"]
    assert result["provider_only_dates"] == ["2013-02-07"]
    assert result["reference_only_dates"] == []
