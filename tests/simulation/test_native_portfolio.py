"""Paridad contable del núcleo C++ con la referencia Python."""

import numpy as np
import pytest

from mars_titan.simulation.native_portfolio import NativePortfolio
from mars_titan.simulation.portfolio import CorporateAction, Instrument, Portfolio, Quote


def assert_same(reference, native):
    assert reference.positions == native.positions
    assert reference.orders == native.orders
    for name in ("cash", "nav"):
        left, right = getattr(reference, name), getattr(native, name)
        assert left.keys() == right.keys()
        for currency in left:
            if left[currency] is None:
                assert right[currency] is None
            else:
                assert right[currency] == pytest.approx(left[currency], rel=1e-12, abs=1e-10)


def test_native_roundoff_tolerance_does_not_erase_large_finite_cash():
    capital = float(np.finfo(np.float64).max)
    portfolio = NativePortfolio({"A": Instrument("USD")}, {"USD": capital})
    quotes = {"A": Quote(1, 1, 100)}
    portfolio.start(1, quotes)
    portfolio.advance(2, 3, quotes)
    assert portfolio.cash["USD"] == capital


def test_failed_frame_copy_keeps_the_confirmed_native_state():
    class FailingCopy(np.ndarray):
        def copy(self, *args, **kwargs):
            raise MemoryError("Fallo de asignación de prueba")

    portfolio = NativePortfolio({"A": Instrument("USD")}, {"USD": 1000}, cost_bps=0)
    portfolio.start(1, {"A": Quote(10, 10, 1000)})
    portfolio.submit({"A": 5}, decision_at=1)
    before = portfolio.snapshot()
    frame = np.array([[10.0, 10.0, 10.0, 10.0, 1000.0]]).view(FailingCopy)
    with pytest.raises(MemoryError):
        portfolio.advance_frame(2, 3, frame)
    assert portfolio.snapshot() == before


def test_unfilled_receipts_keep_the_reference_order():
    instruments = {a: Instrument("USD") for a in ("A", "B")}
    receipts = []
    for kind in (Portfolio, NativePortfolio):
        portfolio = kind(instruments, {"USD": 10}, cost_bps=0)
        portfolio.start(1, {a: Quote(20, 20, 1000) for a in instruments})
        portfolio.submit({"A": 2, "B": 2}, decision_at=1)
        receipts.append(
            portfolio.advance(2, 3, {"A": Quote(20, 20, 1000), "B": Quote(None, 20, 1000)})
        )
    assert receipts[0] == receipts[1]


def test_missing_valuations_keep_the_order_of_persistent_positions():
    instruments = {"A": Instrument("USD"), "B": Instrument("CNY")}
    receipts = []
    for kind in (Portfolio, NativePortfolio):
        portfolio = kind(instruments, {"USD": 1000, "CNY": 1000}, cost_bps=0)
        quotes = {a: Quote(10, 10, 10000) for a in instruments}
        portfolio.start(1, quotes)
        portfolio.submit({"A": 2, "B": 2}, decision_at=1)
        portfolio.advance(2, 3, quotes)
        receipts.append(portfolio.advance(4, 5, {a: Quote(10, None, 10000) for a in instruments}))
    assert receipts[0] == receipts[1]


def test_failed_restore_keeps_the_destination_state(monkeypatch):
    source = NativePortfolio({"A": Instrument("USD")}, {"USD": 1000}, cost_bps=0)
    source.start(1, {"A": Quote(10, 10, 1000)})
    source.submit({"A": 5}, decision_at=1)
    source.advance(2, 3, {"A": Quote(10, 10, 1000)})
    destination = NativePortfolio({"A": Instrument("USD")}, {"USD": 1000}, cost_bps=0)
    before = destination.snapshot()

    def fail(*args):
        raise MemoryError("Fallo de asignación de prueba")

    monkeypatch.setattr(destination, "_frame", fail)
    with pytest.raises(MemoryError):
        destination.restore(source.snapshot())
    assert destination.snapshot() == before


def test_failed_receipt_creation_does_not_publish_new_balances(monkeypatch):
    portfolio = NativePortfolio({"A": Instrument("USD")}, {"USD": 1000}, cost_bps=0)
    portfolio.start(1, {"A": Quote(10, 10, 1000)})
    portfolio.submit({"A": 5}, decision_at=1)
    before = portfolio.snapshot()

    def fail(*args):
        raise MemoryError("Fallo de recibo de prueba")

    monkeypatch.setattr(portfolio, "_result", fail)
    with pytest.raises(MemoryError):
        portfolio.advance(2, 3, {"A": Quote(10, 10, 1000)})
    assert portfolio.snapshot() == before


def test_native_orders_costs_and_corporate_events_match_reference():
    instruments = {"A": Instrument("USD"), "B": Instrument("CNY")}
    reference = Portfolio(instruments, {"USD": 2000, "CNY": 500}, cost_bps=10)
    native = NativePortfolio(instruments, {"USD": 2000, "CNY": 500}, cost_bps=10)
    first = {"A": Quote(10, 10, 10000), "B": Quote(20, 20, 10000)}
    for book in (reference, native):
        book.start(1, first)
        book.submit({"A": 100, "B": 100}, decision_at=1)
        book.advance(2, 3, first)
    assert_same(reference, native)
    actions = [CorporateAction("split", "A", "split", 4, 2, verified=True)]
    prices = {"A": Quote(5, 5, 20000), "B": Quote(20, 20, 10000)}
    for book in (reference, native):
        book.advance(4, 5, prices, actions=actions)
        book.advance(6, 7, prices)
        book.submit({"A": 0, "B": 0}, decision_at=7)
        book.advance(8, 9, prices)
    assert_same(reference, native)
    assert native.execution_counts == {"native_steps": 3, "reference_event_steps": 1}


@pytest.mark.parametrize("assets", [1, 16, 128])
def test_random_portfolios_keep_quantities_orders_and_valuation(assets):
    rng = np.random.default_rng(42)
    instruments = {f"A{i:03}": Instrument("USD" if i % 2 == 0 else "CNY") for i in range(assets)}
    reference = Portfolio(instruments, {"USD": 10000, "CNY": 10000}, cost_bps=25)
    native = NativePortfolio(instruments, {"USD": 10000, "CNY": 10000}, cost_bps=25)
    for step in range(20):
        prices = rng.uniform(1, 100, assets)
        quotes = {
            a: Quote(float(p), float(p * 1.01), float(rng.integers(0, 10000)))
            for a, p in zip(instruments, prices, strict=True)
        }
        if step == 0:
            for book in (reference, native):
                book.start(1, quotes)
        else:
            for book in (reference, native):
                book.advance(step * 2, step * 2 + 1, quotes)
        targets = {a: float(rng.integers(0, 400)) for a in instruments}
        for book in (reference, native):
            book.submit(targets, decision_at=step * 2 + 1)
        assert_same(reference, native)
    restored = NativePortfolio(instruments, {"USD": 10000, "CNY": 10000}, cost_bps=25)
    restored.restore(native.snapshot())
    assert restored.snapshot() == native.snapshot()
    assert restored.advance(40, 41, quotes) == native.advance(40, 41, quotes)
