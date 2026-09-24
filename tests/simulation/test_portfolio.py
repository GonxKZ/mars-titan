"""Casos contables con cantidades y precios conocidos."""

import hashlib
import json

import pytest

from mars_titan.simulation.portfolio import CorporateAction, Instrument, Portfolio, Quote


def book(*, cost_bps=0, currencies=False):
    instruments = {"A": Instrument("USD"), "B": Instrument("CNY" if currencies else "USD")}
    cash = {"USD": 2000, "CNY": 500} if currencies else {"USD": 2000}
    portfolio = Portfolio(instruments, cash, cost_bps=cost_bps, participation=0.1)
    portfolio.start(1, {"A": Quote(10, 10, 10000), "B": Quote(20, 20, 10000)})
    return portfolio


def test_next_open_execution_persistent_positions_and_two_sided_costs():
    portfolio = book(cost_bps=100)
    portfolio.submit({"A": 100}, decision_at=1)
    assert portfolio.positions == {}
    first = portfolio.advance(2, 3, {"A": Quote(10, 12, 10000), "B": Quote(20, 20, 10000)})
    assert portfolio.cash["USD"] == 990
    assert portfolio.positions["A"] == 100
    assert first["nav"]["USD"] == 2190
    second = portfolio.advance(4, 5, {"A": Quote(12, 13, 10000), "B": Quote(20, 20, 10000)})
    assert second["trades"] == []
    portfolio.submit({"A": 0}, decision_at=5)
    end = portfolio.advance(6, 7, {"A": Quote(13, 13, 10000), "B": Quote(20, 20, 10000)})
    assert portfolio.cash["USD"] == 2277
    assert end["costs"]["USD"] == 23
    assert portfolio.positions == {}


def test_split_and_delayed_dividend_conserve_net_worth():
    portfolio = book()
    portfolio.submit({"A": 100}, decision_at=1)
    portfolio.advance(2, 3, {"A": Quote(10, 10, 10000)})
    split = CorporateAction("split-1", "A", "split", 4, 2, verified=True)
    result = portfolio.advance(4, 5, {"A": Quote(5, 5, 20000)}, actions=[split])
    assert portfolio.positions["A"] == 200
    assert result["nav"]["USD"] == 2000
    dividend = CorporateAction("dividend-1", "A", "dividend", 6, 0.5, pay_at=10, verified=True)
    result = portfolio.advance(6, 7, {"A": Quote(4.5, 4.5, 20000)}, actions=[dividend])
    assert result["nav"]["USD"] == 2000
    assert portfolio.cash["USD"] == 1000
    portfolio.advance(8, 9, {"A": Quote(4.5, 4.5, 20000)})
    paid = portfolio.advance(10, 11, {"A": Quote(4.5, 4.5, 20000)})
    assert portfolio.cash["USD"] == 1100
    assert paid["nav"]["USD"] == 2000


def test_missing_quotes_leave_positions_and_pending_orders_explicit():
    portfolio = book()
    portfolio.submit({"A": 100}, decision_at=1)
    missing = portfolio.advance(2, 3, {"A": Quote(None, 10, 10000)})
    assert not missing["trades"]
    assert missing["unfilled"][0]["reason"] == "missing_open"
    portfolio.advance(4, 5, {"A": Quote(10, 10, 10000)})
    missing = portfolio.advance(6, 7, {"A": Quote(10, None, 10000)})
    assert missing["nav"]["USD"] is None
    assert portfolio.positions["A"] == 100
    assert portfolio.cash["USD"] == 1000


def test_currencies_never_finance_each_other_and_cash_never_turns_negative():
    portfolio = book(currencies=True)
    portfolio.submit({"A": 300, "B": 100}, decision_at=1)
    result = portfolio.advance(2, 3, {"A": Quote(10, 10, 10000), "B": Quote(20, 20, 10000)})
    assert portfolio.positions == {"A": 200, "B": 25}
    assert portfolio.cash == {"USD": 0, "CNY": 0}
    assert result["nav"] == {"USD": 2000, "CNY": 500}


def test_snapshot_restores_pending_orders_and_dividend_entitlements():
    portfolio = book()
    portfolio.submit({"A": 100}, decision_at=1)
    portfolio.advance(2, 3, {"A": Quote(10, 10, 10000)})
    state = portfolio.snapshot()
    restored = book()
    restored.restore(state)
    arguments = (4, 5, {"A": Quote(10, 11, 10000)})
    assert portfolio.advance(*arguments) == restored.advance(*arguments)
    assert portfolio.snapshot() == restored.snapshot()
    state["state"]["cash"]["USD"] = -1
    with pytest.raises(ValueError):
        restored.restore(state)


def test_invalid_time_and_unverified_actions_do_not_mutate_book():
    portfolio = book()
    state = portfolio.snapshot()
    with pytest.raises(ValueError):
        portfolio.submit({"A": 1}, decision_at=2)
    with pytest.raises(ValueError):
        portfolio.advance(1, 2, {"A": Quote(10, 10, 10000)})
    with pytest.raises(ValueError):
        portfolio.advance(
            2, 3, {"A": Quote(5, 5, 10000)}, actions=[CorporateAction("x", "A", "split", 2, 2)]
        )
    assert portfolio.snapshot() == state


def test_exact_budget_is_not_rejected_by_float_roundoff():
    prices = {"A": 450.94, "B": 371.43, "C": 926.84}
    portfolio = Portfolio({a: Instrument("USD") for a in prices}, {"USD": 1749.21}, cost_bps=0)
    quotes = {a: Quote(p, p, 100000) for a, p in prices.items()}
    portfolio.start(1, quotes)
    portfolio.submit(dict.fromkeys(prices, 1), decision_at=1)
    result = portfolio.advance(2, 3, quotes)
    assert portfolio.positions == dict.fromkeys(prices, 1)
    assert portfolio.cash["USD"] == 0
    assert result["nav"]["USD"] == pytest.approx(1749.21, rel=1e-15)


@pytest.mark.parametrize("change", ["nav", "future_order"])
def test_rehashed_state_still_requires_accounting_and_temporal_consistency(change):
    portfolio = book()
    portfolio.submit({"A": 1}, decision_at=1)
    state = portfolio.snapshot()
    if change == "nav":
        state["state"]["nav"]["USD"] = 5000
    else:
        state["state"]["orders"]["A"]["decision_at"] = 100
    state["state_sha256"] = hashlib.sha256(
        json.dumps(state["state"], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    with pytest.raises(ValueError):
        portfolio.restore(state)


def test_pending_order_uses_newly_observed_liquidity_at_next_open():
    portfolio = Portfolio({"A": Instrument("USD")}, {"USD": 2000}, cost_bps=0)
    portfolio.start(1, {"A": Quote(10, 10, None)})
    portfolio.submit({"A": 50}, decision_at=1)
    first = portfolio.advance(2, 3, {"A": Quote(10, 10, 10000)})
    assert not first["trades"]
    assert first["unfilled"][0]["reason"] == "unknown_liquidity"
    second = portfolio.advance(4, 5, {"A": Quote(10, 10, 0)})
    assert second["trades"][0]["quantity"] == 50
