"""Ejecución de apertura a cierre en la sesión siguiente sobre una edición sintética.

La edición es un fixture con el formato real (``unadjusted_edition_fixture``). No contiene
precios de ningún proveedor y nada se ajusta ni se evalúa con datos de mercado.
"""

import numpy as np
import pytest

from mars_titan.data.temporal import MarketClock
from mars_titan.simulation.session_prices import REASONS, SessionPrices
from tests.simulation.unadjusted_edition_fixture import Asset, write_edition

US = [
    Asset("AAA", overrides={10: dict(open=20.0, close=21.0)}),
    Asset(
        "BBB",
        unverified=(30,),
        zero_volume=(31,),
        missing=(32,),
        off_grid_open=(33,),
        events=((40, 0.25, 2.0),),
    ),
]
CN = [
    Asset(
        "600000.SS",
        base=10.0,
        overrides={
            9: dict(open=10.0, close=10.0),
            10: dict(open=11.0, close=11.0),
            11: dict(open=10.0, close=10.0),
            12: dict(open=9.0, close=9.5),
            19: dict(open=10.0, close=10.0),
            20: dict(open=10.45, close=10.2),
            21: dict(open=10.0, close=10.0),
            180: dict(open=10.0, close=10.0),
        },
        events=((20, 0.5, 0),),
    ),
    Asset("688001.SS", base=30.0, overrides={9: dict(open=30.0, close=30.0)}),
    Asset("900901.SS", base=1.0),
]


@pytest.fixture(scope="module")
def edition(tmp_path_factory):
    root = tmp_path_factory.mktemp("edition")
    write_edition(root, {"US": US, "CN": CN})
    return root


def decisions(market, positions):
    clock = MarketClock(market, "2023-01-01", "2023-12-31")
    return np.array(
        [int(clock.decisions[i].timestamp()) * 1_000_000 for i in positions], dtype=np.int64
    )


def run(edition, market, assets, positions):
    prices = SessionPrices(edition, market, "2023-01-01", "2023-12-31")
    return prices.executions(np.array(assets, dtype=object), decisions(market, positions))


def test_the_next_session_open_and_close_are_the_exact_traded_prices(edition):
    result = run(edition, "US", ["US/AAA"], [9])
    assert result["open"][0] == 20.0 and result["close"][0] == 21.0
    assert result["long_entry"][0] and result["short_entry"][0]
    assert REASONS[result["reason"][0]] == "executable"
    assert result["buy_tax"][0] == result["sell_tax"][0] == 0


def test_rows_without_a_tradable_open_never_execute_and_keep_their_reason(edition):
    result = run(edition, "US", ["US/BBB"] * 5 + ["US/ZZZ"], [29, 30, 31, 32, 39, 9])
    reasons = [REASONS[code] for code in result["reason"]]
    assert reasons == [
        "unverified",
        "no_volume",
        "no_row",
        "off_grid_open",
        "ambiguous_event",
        "no_row",
    ]
    assert not result["long_entry"].any() and not result["short_entry"].any()
    assert np.isnan(result["open"]).all() and np.isnan(result["close"]).all()


def test_china_limit_up_blocks_buys_and_limit_down_blocks_short_sales(edition):
    result = run(edition, "CN", ["CN/600000.SS"] * 2, [9, 11])
    # Referencia 10,00: límites 11,00 y 9,00 en el tablero principal.
    assert not result["long_entry"][0] and result["short_entry"][0]
    assert result["blocked_long"][0] and result["exit_limit_short"][0]
    assert result["long_entry"][1] and not result["short_entry"][1]
    assert result["blocked_short"][1] and not result["exit_limit_long"][1]


def test_the_reference_subtracts_the_dividend_paid_at_that_open(edition):
    # Cierre anterior 10,00 y dividendo 0,50: referencia 9,50 y límite superior 10,45.
    result = run(edition, "CN", ["CN/600000.SS"], [19])
    assert result["open"][0] == 10.45 and result["blocked_long"][0]
    assert result["short_entry"][0]


def test_stamp_duty_follows_its_dates_and_star_bands_are_wider(edition):
    result = run(edition, "CN", ["CN/600000.SS", "CN/600000.SS", "CN/688001.SS"], [9, 179, 9])
    assert result["buy_tax"].tolist() == [0.0, 0.0, 0.0]
    clock = MarketClock("CN", "2023-01-01", "2023-12-31")
    assert clock.days[180].isoformat() >= "2023-08-28"
    assert result["sell_tax"].tolist() == [0.001, 0.0005, 0.001]
    assert result["long_entry"][2] and result["short_entry"][2]


def test_boards_without_accredited_rules_do_not_trade(edition):
    result = run(edition, "CN", ["CN/900901.SS"], [9])
    assert REASONS[result["reason"][0]] == "board_without_rules"
    assert not result["long_entry"][0]


def test_decisions_off_the_calendar_or_without_a_next_session_are_rejected(edition):
    prices = SessionPrices(edition, "US", "2023-01-01", "2023-12-31")
    moment = decisions("US", [9])
    with pytest.raises(ValueError, match="calendario"):
        prices.executions(np.array(["US/AAA"], dtype=object), moment + 1)
    last = decisions("US", [len(MarketClock("US", "2023-01-01", "2023-12-31").days) - 1])
    with pytest.raises(ValueError, match="2023"):
        prices.executions(np.array(["US/AAA"], dtype=object), last)


def test_identity_names_the_edition_and_the_rules(edition):
    prices = SessionPrices(edition, "CN", "2023-01-01", "2023-12-31")
    identity = prices.identity()
    assert identity["market_rules"] == "cn_a_share_v1"
    assert identity["price_basis"] == "unadjusted_reconstructed"
    assert len(identity["edition_id"]) == 64
