"""Las convenciones financieras comunes dan las mismas cifras en los dos informes.

La cartera larga y corta (`long_short`) y el informe de políticas (`financial_metrics`)
calculan sus estadísticos con `financial_conventions`. Sobre la misma serie de retornos
las cifras deben coincidir bit a bit, también las réplicas del bootstrap.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mars_titan.evaluation import financial_conventions as conventions
from mars_titan.evaluation import financial_metrics as metrics
from mars_titan.evaluation import long_short
from mars_titan.evaluation.paired_comparisons import circular_block_indices

ROOT = Path(__file__).resolve().parents[2]
# Estadísticos que publican los dos informes, con su nombre en cada uno.
SHARED = dict(
    cumulative_return="cumulative_return",
    annualized_return="annualized_return",
    mean_session_return="mean_net_return",
    volatility="volatility",
    sharpe="sharpe",
    max_drawdown="max_drawdown",
)
RESAMPLED = ("annualized_return", "volatility", "sharpe", "mean_session_return")


def _nav(seed, sessions=300):
    rng = np.random.default_rng(seed)
    return 1000 * np.concatenate(([1.0], np.cumprod(1 + rng.normal(0.0004, 0.012, sessions))))


DECLARED = [
    path
    for path in sorted(ROOT.glob("configs/evaluation/*.json"))
    if "long_short" in json.loads(path.read_text())
]


@pytest.mark.parametrize("path", DECLARED, ids=lambda path: path.name)
def test_declared_long_short_sessions_are_the_common_ones(path):
    section = json.loads(path.read_text())["long_short"]
    assert section["annualization_sessions"] == conventions.SESSIONS_PER_YEAR
    assert conventions.SESSIONS_PER_YEAR == {"US": 252, "CN": 243}


@pytest.mark.parametrize("market", ["US", "CN"])
def test_point_statistics_are_identical_in_both_reports(market):
    nav = _nav(3)
    returns = metrics.session_returns(nav)
    policies = metrics.equity_metrics(nav, market=market)
    portfolio = long_short.statistics(
        returns[:, None], np.zeros((len(returns), 1)), conventions.sessions_per_year(market)
    )
    for ours, theirs in SHARED.items():
        assert policies[ours] == float(portfolio[theirs][0]), ours


@pytest.mark.parametrize("market", ["US", "CN"])
def test_bootstrap_intervals_are_identical_in_both_reports(market):
    returns = {arm: metrics.session_returns(_nav(seed)) for arm, seed in (("a", 5), ("b", 6))}
    options = dict(block_length=7, replicates=96, seed=11)
    policies = metrics.block_bootstrap(returns, market=market, base="a", **options)
    matrix = np.column_stack(list(returns.values()))[None]
    estimate, draws, reason = long_short.bootstrap(
        matrix,
        np.zeros(matrix.shape[1:]),
        sessions_per_year=conventions.sessions_per_year(market),
        **options,
    )
    assert reason is None
    for ours in RESAMPLED:
        theirs = SHARED[ours]
        for column, arm in enumerate(returns):
            level = long_short.level_intervals(
                estimate[theirs][0, column : column + 1],
                draws[theirs][:, 0, column : column + 1],
                policies["confidence"],
            )[0]
            assert policies["arms"][arm][ours] == level, (ours, arm)


def test_resampling_draws_fixed_chunks_from_one_seeded_generator():
    rng = np.random.default_rng(23)
    expected = np.concatenate([circular_block_indices(rng, size, 50, 4) for size in (32, 32, 6)])
    found = list(conventions.resamples(50, block_length=4, replicates=70, seed=23))
    assert [offset for offset, _ in found] == [0, 32, 64]
    assert np.array_equal(np.concatenate([index for _, index in found]), expected)


def test_a_replicate_that_keeps_every_session_in_order_matches_the_point_estimate():
    returns = np.stack([metrics.session_returns(_nav(seed, 120)) for seed in (1, 2)], axis=-1)
    point = conventions.statistics(returns, 252)
    replicate = conventions.resampled(returns, 252, np.arange(120)[None])
    for name in conventions.STATISTICS:
        assert np.allclose(replicate[name][0], point[name], rtol=1e-12, atol=0), name


def test_identical_and_constant_series_stay_exact_in_every_replicate():
    base = metrics.session_returns(_nav(4, 80))
    returns = np.stack([base, base, np.full(80, 0.001)], axis=-1)
    _, index = next(conventions.resamples(80, block_length=5, replicates=32, seed=2))
    replicate = conventions.resampled(returns, 252, index)
    for name in conventions.STATISTICS:
        assert np.array_equal(replicate[name][:, 0], replicate[name][:, 1]), name
    # Una serie constante no tiene dispersión en ninguna réplica ni un Sharpe definido.
    assert (replicate["volatility"][:, 2] == 0).all()
    assert np.isnan(replicate["sharpe"][:, 2]).all()
    point = conventions.statistics(returns, 252)
    assert math.isnan(point["sharpe"][2]) and math.isnan(point["sortino"][2])


def test_a_ruin_only_reaches_the_replicates_that_draw_it():
    returns = np.full((30, 1), 0.01)
    returns[12] = -1.2
    point = conventions.statistics(returns, 243)
    assert point["cumulative_return"][0] == -1 and point["max_drawdown"][0] == 1
    # La segunda réplica salta la sesión 12 y repite la primera.
    index = np.stack([np.arange(30), np.r_[np.arange(12), np.arange(13, 30), 0]])
    replicate = conventions.resampled(returns, 243, index)
    assert replicate["annualized_return"][0, 0] == -1 and replicate["max_drawdown"][0, 0] == 1
    assert replicate["annualized_return"][1, 0] == pytest.approx(1.01**243 - 1)
    assert replicate["max_drawdown"][1, 0] == 0


def test_an_undeclared_market_or_resampling_parameter_is_rejected():
    with pytest.raises(ValueError, match="sesiones por año"):
        conventions.sessions_per_year("HK")
    for options in (
        dict(block_length=0, replicates=10, seed=1),
        dict(block_length=3, replicates=0, seed=1),
        dict(block_length=3, replicates=10, seed=-1),
        dict(block_length=20, replicates=10, seed=1),
    ):
        with pytest.raises(ValueError):
            next(conventions.resamples(20, **options))
