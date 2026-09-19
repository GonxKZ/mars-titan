"""Casos de calendario y barrera de admisión multimodal."""

import importlib
from datetime import UTC, datetime

import pytest


def temporal_module():
    try:
        return importlib.import_module("mars_titan.data.temporal")
    except ModuleNotFoundError:
        pytest.fail("El contrato temporal todavía no está implementado")


def test_date_only_uses_next_session_not_calendar_day():
    clock = temporal_module().MarketClock("US", "2024-01-01", "2024-12-31")
    # Independence Day is followed by Friday, then a weekend.
    assert clock.date_available("2024-07-04") == datetime(2024, 7, 5, 20, 5, tzinfo=UTC)
    assert clock.date_available("2024-07-05") == datetime(2024, 7, 8, 20, 5, tzinfo=UTC)
    assert clock.date_available("2024-07-04", lag=2) == datetime(2024, 7, 8, 20, 5, tzinfo=UTC)


def test_close_changes_with_dst_and_half_days():
    clock = temporal_module().MarketClock("US", "2024-01-01", "2024-12-31")
    assert clock.decision("2024-03-08").hour == 21
    assert clock.decision("2024-03-11").hour == 20
    assert clock.decision("2024-11-29") == datetime(2024, 11, 29, 18, 5, tzinfo=UTC)


def test_four_real_modalities_and_macro_are_required():
    module = temporal_module()
    cutoff = datetime(2024, 7, 8, 20, 5, tzinfo=UTC)
    observed = {name: cutoff for name in ("prices", "news", "fundamentals", "charts", "macro")}
    assert module.admission_errors(observed, cutoff) == []
    assert module.admission_errors({**observed, "fundamentals": None}, cutoff) == [
        "fundamentals:missing_availability"
    ]
    assert module.admission_errors({**observed, "macro": None}, cutoff) == [
        "macro:missing_availability"
    ]
    future = datetime(2024, 7, 9, tzinfo=UTC)
    assert module.admission_errors({**observed, "news": future}, cutoff) == [
        "news:future_information"
    ]


def test_naive_times_and_invalid_lag_fail_fast():
    module = temporal_module()
    with pytest.raises(ValueError, match="timezone"):
        module.admission_errors({}, datetime(2024, 1, 1))
    with pytest.raises(ValueError, match="lag"):
        module.MarketClock("US", "2024-01-01", "2024-12-31").date_available("2024-01-01", 0)
