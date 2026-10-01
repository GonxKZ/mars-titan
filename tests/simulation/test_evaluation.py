"""Resultados contables independientes del objetivo de ajuste."""

import numpy as np
import pytest

from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.evaluation import evaluate
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.portfolio import CorporateAction


def tape(*, missing=False, ruined=False):
    prices = np.array([[[10, 10, 10, 10, 1e6]], [[10, 12, 10, 12, 1e6]], [[12, 12, 9, 9, 1e6]]])
    if missing:
        prices[-1, 0, 3] = np.nan
    return MarketTape(
        prices,
        [100, 200, 300],
        ["A"],
        np.ones((3, 1)),
        domain="synthetic",
        currency="USD",
        partition="validation",
        open_times=[50, 150, 250],
        actions=[CorporateAction("loss", "A", "writeoff", 250, 0, verified=True)] if ruined else (),
    )


def test_passive_valuation_includes_costs_without_final_liquidation():
    env = FinancialEnv(tape(), capital=1000, cost_bps=0)
    report = evaluate(env, lambda observation, step: 5 if step == 0 else 0)
    assert report["financial_validation"]["net_return"] == pytest.approx(-0.1)
    assert report["financial_validation"]["max_drawdown"] == pytest.approx(0.25)
    assert report["financial_validation"]["completed"] is True
    assert env.book.positions == {"A": 100}


def test_missing_held_close_is_an_incomplete_result():
    report = evaluate(FinancialEnv(tape(missing=True), capital=1000, cost_bps=0), lambda *_: 5)
    metrics = report["financial_validation"]
    assert metrics["net_return"] is None
    assert metrics["max_drawdown"] is None
    assert metrics["completed"] is False
    assert metrics["invalid_reason"] == "missing_close"


def test_ruin_is_an_observed_result_with_explicit_reward_penalty():
    report = evaluate(FinancialEnv(tape(ruined=True), capital=1000, cost_bps=0), lambda *_: 5)
    metrics = report["financial_validation"]
    assert metrics["net_return"] == -1
    assert metrics["max_drawdown"] == 1
    assert metrics["completed"] is True
    assert metrics["invalid_reason"] == "ruined"
    assert report["ruin_reward_penalty"] == -20
