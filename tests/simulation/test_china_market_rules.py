"""Reglas de acciones A con casos contables conocidos y fechas de vigencia."""

import hashlib
import json
import math

import numpy as np
import pytest

from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.market_rules import beijing_day, board, china_a_share_instrument
from mars_titan.simulation.portfolio import CorporateAction, Instrument, Period, Portfolio, Quote

HOUR = 3_600_000_000
RATE = 0.001


def session(day, *, close=False):
    """Apertura a las 9:30 y cierre a las 15:00 de Pekín."""
    return beijing_day(*day) + (15 * HOUR if close else 9 * HOUR + HOUR // 2)


def book(asset="600000.SS", *, position=0.0, close=10.0, day=(2021, 3, 2), cost_bps=10):
    """Cartera iniciada al cierre previo a la apertura de ``day``, con una posición dada."""
    instrument = china_a_share_instrument(asset)
    portfolio = Portfolio(
        {asset: instrument}, {"CNY": 1_000_000}, cost_bps=cost_bps, participation=1
    )
    portfolio.start(session(day) - 19 * HOUR, {asset: Quote(close, close, 1e9)})
    if position:
        # Una posición impar procede, por ejemplo, de acciones liberadas.
        snapshot = portfolio.snapshot()
        state = snapshot["state"]
        state["positions"][asset] = float(position)
        state["cash"]["CNY"] -= position * close
        snapshot["state_sha256"] = hashlib.sha256(
            json.dumps(state, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        portfolio.restore(snapshot)
    return portfolio


def execute(portfolio, asset, target, open_price, *, day=(2021, 3, 2), actions=()):
    portfolio.submit({asset: target}, decision_at=portfolio.clock)
    return portfolio.advance(
        session(day),
        session(day, close=True),
        {asset: Quote(open_price, open_price, 1e9)},
        actions=actions,
    )


@pytest.mark.parametrize(
    "asset,kind",
    [
        ("600000.SS", "main"),
        ("CN/601318.SS", "main"),
        ("688981.SS", "star"),
        ("000001.SZ", "main"),
        ("002594.SZ", "main"),
        ("300750.SZ", "chinext"),
        ("600000.SH", "main"),
    ],
)
def test_board_comes_from_the_code_and_exchange(asset, kind):
    assert board(asset) == kind


@pytest.mark.parametrize("asset", ["900901.SS", "200002.SZ", "830799.BJ", "AAPL", "600000"])
def test_boards_without_verified_rules_are_rejected(asset):
    with pytest.raises(ValueError):
        china_a_share_instrument(asset)


def test_main_board_buys_round_lots_and_sells_the_odd_balance_at_once():
    portfolio = book()
    result = execute(portfolio, "600000.SS", 1_050, 10.0)
    assert [t["quantity"] for t in result["trades"]] == [1_000]
    for held, target, sold in ((150, 0, 150), (150, 100, 50), (150, 20, 100), (250, 100, 150)):
        portfolio = book(position=held)
        result = execute(portfolio, "600000.SS", target, 10.0)
        assert [t["quantity"] for t in result["trades"]] == [-sold]


def test_star_orders_need_two_hundred_shares_and_small_balances_leave_whole():
    asset = "688981.SS"
    assert execute(book(asset), asset, 150, 10.0)["trades"] == []
    assert [t["quantity"] for t in execute(book(asset), asset, 257, 10.0)["trades"]] == [257]
    for held, target, sold in ((150, 50, 0), (150, 0, 150), (300, 50, 250), (300, 150, 0)):
        result = execute(book(asset, position=held), asset, target, 10.0)
        assert [t["quantity"] for t in result["trades"]] == ([-sold] if sold else [])


@pytest.mark.parametrize(
    "open_price,target,reason", [(11.0, 1_000, "limit_up"), (9.0, 0, "limit_down")]
)
def test_opening_at_the_daily_limit_does_not_fill_in_that_direction(open_price, target, reason):
    portfolio = book(position=500)
    result = execute(portfolio, "600000.SS", target, open_price)
    assert result["trades"] == [] and result["unfilled"] == [dict(asset="600000.SS", reason=reason)]
    assert portfolio.orders["600000.SS"]["target"] == target
    inside = execute(book(position=500), "600000.SS", target, 10.99 if target else 9.01)
    assert inside["trades"]


def test_limit_price_rounds_half_up_to_the_cent():
    instrument = china_a_share_instrument("600000.SS")
    at = session((2021, 3, 2))
    assert instrument.limits(10.15, at) == (11.17, 9.14)
    assert instrument.limits(10.05, at) == (11.06, 9.05)
    assert instrument.limits(None, at) is None


@pytest.mark.parametrize("day,blocked", [((2020, 8, 21), True), ((2020, 8, 24), False)])
def test_chinext_band_widens_on_the_registration_reform(day, blocked):
    asset = "300750.SZ"
    result = execute(book(asset, day=day), asset, 1_000, 11.5, day=day)
    assert (result["trades"] == []) is blocked


def test_star_band_starts_with_the_board():
    instrument = china_a_share_instrument("688981.SS")
    assert instrument.limits(10.0, session((2019, 7, 19))) is None
    assert instrument.limits(10.0, session((2019, 7, 22))) == (12.0, 8.0)


def test_limits_use_the_ex_rights_reference_after_splits_and_dividends():
    asset = "600000.SS"
    day = (2021, 3, 2)
    split = CorporateAction("split", asset, "split", session(day), 2.0, verified=True)
    result = execute(book(asset, position=1_000, close=20.0), asset, 0, 9.5, actions=[split])
    assert [t["quantity"] for t in result["trades"]] == [-2_000]
    dividend = CorporateAction(
        "dividend",
        asset,
        "dividend",
        session(day),
        0.5,
        pay_at=session(day, close=True),
        verified=True,
    )
    result = execute(book(asset, position=1_000), asset, 0, 8.6, actions=[dividend])
    assert result["trades"]


@pytest.mark.parametrize(
    "day,side,tax",
    [
        ((2007, 6, 1), "buy", 0.003),
        ((2008, 5, 5), "buy", 0.001),
        ((2008, 9, 22), "buy", 0.0),
        ((2023, 8, 25), "sell", 0.001),
        ((2023, 8, 28), "sell", 0.0005),
    ],
)
def test_stamp_duty_follows_its_dated_schedule(day, side, tax):
    portfolio = book(position=1_000 if side == "sell" else 0, day=day)
    result = execute(portfolio, "600000.SS", 0 if side == "sell" else 1_000, 10.0, day=day)
    trade = result["trades"][0]
    assert trade["cost"] == pytest.approx(abs(trade["quantity"]) * 10.0 * (RATE + tax), rel=1e-12)


def test_same_session_round_trips_are_impossible():
    portfolio = book()
    execute(portfolio, "600000.SS", 1_000, 10.0)
    portfolio.submit({"600000.SS": 0}, decision_at=portfolio.clock)
    # La venta solo puede ejecutarse en una apertura posterior al cierre de la compra.
    with pytest.raises(ValueError, match="posterior"):
        portfolio.advance(
            session((2021, 3, 2)),
            session((2021, 3, 2), close=True),
            {"600000.SS": Quote(10, 10, 1e9)},
        )


def test_rules_are_part_of_the_identity_and_absent_by_default():
    assert Instrument("CNY").identity() == dict(currency="CNY", lot=1)
    identity = china_a_share_instrument("600000.SS").identity()
    assert identity["rules"] == "cn_a_share_v1_main" and identity["lot"] == 100
    with pytest.raises(ValueError, match="identidad"):
        Instrument("CNY", taxes=(Period(0, 1, sell=0.001),))
    with pytest.raises(ValueError, match="ordenados"):
        Instrument("CNY", price_limits=(Period(5, 6, band=0.1), Period(0, 10, band=0.1)), rules="x")


def test_environment_applies_declared_rules_and_identifies_them():
    assets = ["600000.SS", "688981.SS"]
    days = [(2021, 3, 1), (2021, 3, 2), (2021, 3, 3)]
    prices = np.tile([10.0, 10.0, 10.0, 10.0, 1e7], (3, 2, 1))
    tape = MarketTape(
        prices,
        [session(day, close=True) for day in days],
        assets,
        np.full((3, 2), 0.01),
        domain="synthetic",
        currency="CNY",
        open_times=[session(day) for day in days],
    )
    rules = {asset: china_a_share_instrument(asset) for asset in assets}
    plain, ruled = (
        FinancialEnv(tape, capital=100_000),
        FinancialEnv(tape, capital=100_000, instruments=rules),
    )
    assert "instruments_sha256" in ruled.identity and "instruments_sha256" not in plain.identity
    ruled.reset(seed=0)
    _, _, _, _, info = ruled.step(5)
    assert all(t["quantity"] % 100 == 0 for t in info["trades"] if t["asset"] == "600000.SS")
    assert all(math.isfinite(t["cost"]) for t in info["trades"])


def test_buying_with_all_cash_reserves_the_stamp_duty_of_both_sides():
    portfolio = book(day=(2007, 6, 1))
    result = execute(portfolio, "600000.SS", 1_000_000, 10.0, day=(2007, 6, 1))
    quantity = result["trades"][0]["quantity"]
    assert quantity == 99_600 and portfolio.cash["CNY"] >= 0
    assert result["costs"]["CNY"] == pytest.approx(quantity * 10.0 * (RATE + 0.003))
