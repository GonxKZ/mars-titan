import math

import numpy as np
import pytest

from mars_titan.evaluation import financial_metrics as metrics


def test_equity_metrics_match_hand_computed_values():
    nav = [100.0, 110.0, 99.0, 108.9]
    result = metrics.equity_metrics(nav, market="US", turnover=1.5, costs=0.25)
    returns = [0.1, -0.1, 0.1]
    mean = sum(returns) / 3
    deviation = math.sqrt(sum((r - mean) ** 2 for r in returns) / 2)
    downside = math.sqrt(0.01 / 3)
    assert result["sessions"] == 3
    assert result["cumulative_return"] == pytest.approx(0.089)
    assert result["cumulative_return_percent"] == pytest.approx(8.9)
    assert result["annualized_return"] == pytest.approx(1.089 ** (252 / 3) - 1)
    assert result["log_growth"] == pytest.approx(math.log(1.089))
    assert result["volatility"] == pytest.approx(deviation * math.sqrt(252))
    assert result["sharpe"] == pytest.approx(mean / deviation * math.sqrt(252))
    assert result["sortino"] == pytest.approx(mean / downside * math.sqrt(252))
    assert result["max_drawdown"] == pytest.approx(0.1)
    assert (result["turnover"], result["costs"], result["ruined"]) == (1.5, 0.25, False)


def test_cash_has_no_risk_ratios_and_ruin_ends_at_minus_one():
    cash = metrics.equity_metrics([100.0] * 5, market="US")
    assert (cash["volatility"], cash["sharpe"], cash["sortino"]) == (0.0, None, None)
    assert cash["max_drawdown"] == 0
    ruined = metrics.equity_metrics([100.0, 50.0, 0.0], market="US")
    assert ruined["ruined"] and ruined["annualized_return"] == -1 and ruined["log_growth"] is None
    assert ruined["max_drawdown"] == 1


@pytest.mark.parametrize(
    "nav",
    [[100.0], [100.0, 0.0, 50.0], [100.0, -1.0], [100.0, math.nan], [0.0, 1.0], [[1.0, 2.0]]],
)
def test_invalid_equity_series_fail(nav):
    with pytest.raises(ValueError):
        metrics.equity_metrics(nav, market="US")


def test_ratios_do_not_depend_on_the_initial_capital():
    rng = np.random.default_rng(3)
    nav = 1000 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300))
    nav = np.concatenate(([1000.0], nav))
    small = metrics.equity_metrics(nav, market="US")
    large = metrics.equity_metrics(nav * 37.5, market="US")
    for key in ("cumulative_return", "annualized_return", "sharpe", "sortino", "max_drawdown"):
        assert small[key] == pytest.approx(large[key], rel=1e-12)


def test_drawdown_depends_on_the_order_of_sessions():
    assert metrics.max_drawdown([100, 50, 100, 200]) == pytest.approx(0.5)
    assert metrics.max_drawdown([100, 200, 100, 50]) == pytest.approx(0.75)


def test_china_annualizes_with_its_own_sessions_per_year():
    nav = [100.0, 110.0, 99.0, 108.9]
    china = metrics.equity_metrics(nav, market="CN")
    assert china["annualized_return"] == pytest.approx(1.089 ** (243 / 3) - 1)
    us = metrics.equity_metrics(nav, market="US")
    assert china["sharpe"] == pytest.approx(us["sharpe"] * math.sqrt(243 / 252))
    with pytest.raises(ValueError, match="sesiones por año"):
        metrics.equity_metrics(nav, market="HK")


def _series(seed, drift, sessions=250):
    rng = np.random.default_rng(seed)
    return rng.normal(drift, 0.01, sessions)


def test_block_bootstrap_is_paired_reproducible_and_flags_a_sure_difference():
    base = _series(1, 0.0)
    returns = dict(klpo=base + 0.002, ppo=base.copy(), noise=_series(2, 0.0))
    first = metrics.block_bootstrap(
        returns,
        market="US",
        base="ppo",
        block_length=10,
        replicates=400,
        seed=7,
        sensitivity=(5, 400),
    )
    second = metrics.block_bootstrap(
        returns,
        market="US",
        base="ppo",
        block_length=10,
        replicates=400,
        seed=7,
        sensitivity=(5, 400),
    )
    assert first == second
    mean = first["differences"]["mean_session_return"]
    # Una diferencia constante no varía entre réplicas emparejadas.
    assert mean["klpo"]["estimate"] == pytest.approx(0.002)
    assert mean["klpo"]["interval"] == pytest.approx([0.002, 0.002])
    assert mean["klpo"]["simultaneous_excludes_zero"] is True
    assert mean["noise"]["simultaneous_excludes_zero"] in (True, False)
    interval = first["arms"]["ppo"]["sharpe"]["interval"]
    assert interval[0] < first["arms"]["ppo"]["sharpe"]["estimate"] < interval[1]
    assert first["multiplicity"]["family_size"] == 2
    assert first["sensitivity"][1]["reason"] is not None
    assert first["sensitivity"][1]["arms"]["ppo"]["sharpe"]["interval"] is None


def test_block_bootstrap_identical_series_have_zero_differences():
    base = _series(4, 0.001)
    result = metrics.block_bootstrap(
        dict(a=base, b=base.copy()), market="US", base="a", block_length=5, replicates=50, seed=1
    )
    row = result["differences"]["sharpe"]["b"]
    assert row["estimate"] == 0 and row["simultaneous_interval"] == [0.0, 0.0]
    assert row["simultaneous_excludes_zero"] is False


def test_block_bootstrap_keeps_a_ruin_inside_the_replicates_that_draw_it():
    returns = _series(6, 0.0, 40)
    returns[-1] = -1.0
    result = metrics.block_bootstrap(
        dict(a=returns, b=np.zeros(40)),
        market="US",
        base="b",
        block_length=3,
        replicates=200,
        seed=2,
    )
    assert result["arms"]["a"]["annualized_return"]["estimate"] == -1
    interval = result["arms"]["a"]["annualized_return"]["interval"]
    assert interval[1] == -1 or interval[0] == -1


@pytest.mark.parametrize(
    "change",
    [
        dict(block_length=0),
        dict(replicates=5),
        dict(seed=-1),
        dict(confidence=0.4),
        dict(base="missing"),
        dict(market="HK"),
    ],
)
def test_block_bootstrap_rejects_undeclared_parameters(change):
    options = dict(market="US", base="a", block_length=5, replicates=50, seed=1) | change
    with pytest.raises(ValueError):
        metrics.block_bootstrap(dict(a=np.zeros(20), b=np.zeros(20)), **options)


def test_block_bootstrap_rejects_unpaired_or_impossible_returns():
    with pytest.raises(ValueError):
        metrics.block_bootstrap(
            dict(a=np.zeros(20), b=np.zeros(19)),
            market="US",
            base="a",
            block_length=5,
            replicates=50,
            seed=1,
        )
    with pytest.raises(ValueError):
        metrics.block_bootstrap(
            dict(a=np.full(20, -1.5), b=np.zeros(20)),
            market="US",
            base="a",
            block_length=5,
            replicates=50,
            seed=1,
        )
