"""Calendario de madurez del enlace financiero, sin consultar retornos futuros."""

import pandas as pd
import pytest
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation

from mars_titan.data.temporal import MarketClock


@pytest.mark.parametrize(
    "decision,expected",
    [
        ("2021-01-15", "2021-01-19T21:05:00Z"),
        ("2021-03-12", "2021-03-15T20:05:00Z"),
        ("2021-11-24", "2021-11-26T18:05:00Z"),
        ("2023-12-29", None),
    ],
)
def test_maturity_uses_the_next_admitted_session_and_price_availability(
    four_flow_source, decision, expected, monkeypatch
):
    verifier = four_flow_source["prefixes"]

    def forbidden(*args, **kwargs):
        raise AssertionError("El calendario no puede consultar retornos futuros")

    monkeypatch.setattr(verifier, "_returns", forbidden)
    at = int(
        pd.Timestamp(MarketClock("US", "2021-01-01", "2023-12-31").decision(decision)).value // 1000
    )
    actual = verifier.expected_maturity("US/T0000", at)
    assert actual == (None if expected is None else int(pd.Timestamp(expected).value // 1000))


@pytest.mark.parametrize(
    "flow,at",
    [("US/unknown", 1), ("US/T0000", True), ("US/T0000", 1), ("US/T0000", 1_704_067_200_000_000)],
)
def test_maturity_rejects_foreign_flows_non_sessions_and_the_reserved_period(
    four_flow_source, flow, at
):
    with pytest.raises(ValueError):
        four_flow_source["prefixes"].expected_maturity(flow, at)
