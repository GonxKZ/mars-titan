"""Cartera larga y corta por cuartiles con resultados conocidos, sin datos de mercado."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mars_titan.evaluation import long_short
from mars_titan.evaluation.forecast_panel import ForecastPanel

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/evaluation/historical-masked-2000-comparison.json"
DAY = np.datetime64("2023-03-01T21:05:00", "us")
OPTIONS = dict(fraction=0.25, min_assets=4, exposure=dict(long=0.5, short=0.5))


def panel(predictions, sessions=None):
    predictions = np.asarray(predictions, dtype=np.float64)
    sessions = np.zeros(len(predictions), dtype=int) if sessions is None else np.asarray(sessions)
    times = DAY + sessions.astype("timedelta64[D]")
    return ForecastPanel.from_columns(
        np.array([f"r{i:03d}" for i in range(len(predictions))]),
        np.array(["US"] * len(predictions)),
        times,
        np.zeros(len(predictions)),
        predictions,
    )


def execution(rows, **changes):
    value = dict(
        open=np.full(rows, 100.0),
        close=np.full(rows, 100.0),
        long_entry=np.ones(rows, dtype=bool),
        short_entry=np.ones(rows, dtype=bool),
        buy_tax=np.zeros(rows),
        sell_tax=np.zeros(rows),
        exit_limit_long=np.zeros(rows, dtype=bool),
        exit_limit_short=np.zeros(rows, dtype=bool),
    )
    value.update(changes)
    return value


def test_the_repository_declares_the_protocol_rule():
    section = json.loads(CONFIG.read_text())["long_short"]
    assert long_short.declaration(section) is section
    assert section["cost_bps_per_side"] == [0, 5, 10, 20]
    assert section["exposure"] == {"long": 0.5, "short": 0.5} and section["fraction"] == 0.25


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.update(fraction=0.6),
        lambda s: s.update(min_assets=2),
        lambda s: s.update(cost_bps_per_side=[5, 0]),
        lambda s: s.update(cost_bps_per_side=[0, 0.5]),
        lambda s: s.update(holding="next_session_open_to_next_open"),
        lambda s: s.update(unfilled="renormalize"),
        lambda s: s.update(ties="random"),
        lambda s: s.update(market_rules={"US": "none", "CN": "none"}),
        lambda s: s.update(statistics=["sharpe"]),
        lambda s: s.update(exposure={"long": 1.0}),
        lambda s: s.update(annualization_sessions={"US": 252}),
        lambda s: s.update(views="pooled"),
        lambda s: s.pop("seeds"),
        lambda s: s.update(extra=True),
    ],
)
def test_declarations_that_change_the_rule_are_rejected(change):
    section = copy.deepcopy(json.loads(CONFIG.read_text())["long_short"])
    change(section)
    with pytest.raises(ValueError):
        long_short.declaration(section)


def test_extremes_by_hand_with_eight_assets():
    weights = long_short.quartile_weights(panel([5, 1, 7, 3, 8, 2, 6, 4]), **OPTIONS)
    # k = 2: largos 8 y 7, cortos 1 y 2, con 0,5 / 2 cada uno.
    assert weights.tolist() == [0, -0.25, 0.25, 0, 0.25, -0.25, 0, 0]


def test_ties_crossing_the_boundary_stay_out_and_constant_predictions_abstain():
    # k = 2. El 7 entra, los tres 6 empatados en la frontera no. Los dos 1 sí son cortos.
    weights = long_short.quartile_weights(panel([7, 6, 6, 6, 1, 1, 3, 4]), **OPTIONS)
    assert weights.tolist() == [0.25, 0, 0, 0, -0.25, -0.25, 0, 0]
    assert not long_short.quartile_weights(panel(np.zeros(8)), **OPTIONS).any()


def test_sessions_below_the_minimum_do_not_trade_and_sessions_are_independent():
    values = [1, 2, 3] + [4, 3, 2, 1, 9, 8, 7, 6]
    weights = long_short.quartile_weights(panel(values, [0, 0, 0] + [1] * 8), **OPTIONS)
    assert not weights[:3].any()
    assert weights[3:].tolist() == [0, 0, -0.25, -0.25, 0.25, 0.25, 0, 0]
    # El orden de las filas de entrada no cambia la selección.
    shuffled = np.array(values)[::-1]
    again = long_short.quartile_weights(panel(shuffled, ([1] * 8 + [0, 0, 0])), **OPTIONS)
    assert sorted(again.tolist()) == sorted(weights.tolist())


def test_book_by_hand_with_costs_taxes_and_unfilled_orders():
    scores = panel([5, 1, 7, 3, 8, 2, 6, 4])
    weights = long_short.quartile_weights(scores, **OPTIONS)
    close = np.full(8, 100.0)
    # Filas canónicas r002 (7) y r004 (8) largas, r001 (1) y r005 (2) cortas.
    close[[2, 4, 1, 5]] = [110.0, 95.0, 90.0, 104.0]
    run = execution(8, close=close, long_entry=np.array([1, 1, 1, 1, 0, 1, 1, 1], dtype=bool))
    run["buy_tax"][:] = 0.001
    run["sell_tax"][:] = 0.002
    book = long_short.session_book(scores, weights, run)
    # Solo r002 entra largo (r004 queda en efectivo): 0,25 · 0,10. Cortas: 0,25 · (0,10 − 0,04).
    gross = 0.25 * 0.10 + 0.25 * 0.10 + 0.25 * (-0.04)
    traded = 0.25 * (1 + 1.1) + 0.25 * (1 + 0.9) + 0.25 * (1 + 1.04)
    taxes = 0.25 * (0.001 + 1.1 * 0.002) + 0.25 * (0.002 + 0.9 * 0.001)
    taxes += 0.25 * (0.002 + 1.04 * 0.001)
    assert book["gross_return"][0] == pytest.approx(gross, abs=1e-15)
    assert book["traded"][0] == pytest.approx(traded, abs=1e-15)
    assert book["taxes"][0] == pytest.approx(taxes, abs=1e-15)
    assert book["long_exposure"][0] == 0.25 and book["short_exposure"][0] == 0.5
    assert book["selected_long"][0] == 2 and book["filled_long"][0] == 1
    net = long_short.net_returns(book, [0, 10])
    assert net[0, 0] == pytest.approx(gross - taxes, abs=1e-15)
    assert net[1, 0] == pytest.approx(gross - 0.001 * traded - taxes, abs=1e-15)


def test_fixed_returns_give_the_exact_sharpe_volatility_and_drawdown():
    returns = np.array([0.01, -0.02, 0.03, 0.01, -0.01, 0.02])
    stats = long_short.statistics(returns[:, None], np.full((6, 1), 2.0), 252)
    mean, sd = returns.mean(), returns.std(ddof=1)
    assert stats["mean_net_return"][0] == pytest.approx(mean, rel=1e-14)
    assert stats["volatility"][0] == pytest.approx(sd * math.sqrt(252), rel=1e-14)
    assert stats["sharpe"][0] == pytest.approx(mean / sd * math.sqrt(252), rel=1e-12)
    wealth = np.cumprod(1 + returns)
    assert stats["cumulative_return"][0] == pytest.approx(wealth[-1] - 1, rel=1e-12)
    assert stats["annualized_return"][0] == pytest.approx(wealth[-1] ** (252 / 6) - 1, rel=1e-10)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    assert stats["max_drawdown"][0] == pytest.approx(np.max(1 - wealth / peak), rel=1e-12)
    assert stats["max_drawdown"][0] == pytest.approx(0.02, rel=1e-12)
    assert stats["turnover"][0] == 2.0


def test_flat_and_ruined_series_have_declared_values():
    stats = long_short.statistics(np.zeros((5, 1)), np.zeros((5, 1)), 252)
    assert math.isnan(stats["sharpe"][0]) and stats["max_drawdown"][0] == 0
    assert stats["cumulative_return"][0] == 0 and stats["volatility"][0] == 0
    ruined = long_short.statistics(np.array([[0.1], [-1.0], [0.5]]), np.ones((3, 1)), 252)
    assert ruined["cumulative_return"][0] == -1 and ruined["max_drawdown"][0] == 1


def test_bootstrap_with_identical_arms_collapses_their_contrasts():
    rng = np.random.default_rng(3)
    base = rng.normal(0.001, 0.01, 200)
    returns = np.stack([base, base, base + 0.002], axis=-1)[None]
    traded = np.ones((200, 3))
    estimate, draws, reason = long_short.bootstrap(
        returns, traded, sessions_per_year=252, block_length=5, replicates=200, seed=7
    )
    assert reason is None and draws["sharpe"].shape == (200, 1, 3)
    family = {"b-a": {"b": 1.0, "a": -1.0}, "c-a": {"c": 1.0, "a": -1.0}}
    mean = long_short.contrasts(
        estimate["mean_net_return"][0],
        draws["mean_net_return"][:, 0],
        ["a", "b", "c"],
        family,
        0.95,
    )
    same, shifted = mean["contrasts"]
    assert same["estimate"] == 0 and same["interval"] == [0.0, 0.0]
    # Un desplazamiento constante se estima exacto y su intervalo no tiene anchura.
    assert shifted["estimate"] == pytest.approx(0.002, rel=1e-12)
    assert shifted["interval"][0] == pytest.approx(0.002, rel=1e-9)
    assert shifted["interval"][1] == pytest.approx(0.002, rel=1e-9)


def test_paired_resampling_uses_the_same_sessions_for_every_arm_and_cost():
    rng = np.random.default_rng(5)
    returns = rng.normal(0, 0.01, (2, 120, 2))
    estimate, draws, _ = long_short.bootstrap(
        returns, np.ones((120, 2)), sessions_per_year=252, block_length=4, replicates=64, seed=11
    )
    again = long_short.bootstrap(
        returns, np.ones((120, 2)), sessions_per_year=252, block_length=4, replicates=64, seed=11
    )[1]
    for name in long_short.STATISTICS:
        assert np.array_equal(draws[name], again[name])
    # Restar un coste constante a todas las sesiones baja la media en esa cantidad en cada réplica.
    shifted = returns.copy()
    shifted[1] = shifted[0] - 0.0005
    _, moved, _ = long_short.bootstrap(
        shifted, np.ones((120, 2)), sessions_per_year=252, block_length=4, replicates=64, seed=11
    )
    gap = moved["mean_net_return"][:, 0] - moved["mean_net_return"][:, 1]
    assert np.allclose(gap, 0.0005, atol=1e-15)


def test_contrasts_with_an_undefined_arm_are_reported_and_left_out_of_the_family():
    estimate = np.array([np.nan, 1.0, 2.0])
    draws = np.column_stack([np.full(50, np.nan), np.linspace(0.5, 1.5, 50), np.linspace(1, 3, 50)])
    family = {"a-b": {"a": 1.0, "b": -1.0}, "c-b": {"c": 1.0, "b": -1.0}}
    result = long_short.contrasts(estimate, draws, ["a", "b", "c"], family, 0.9)
    assert result["family_size"] == 1
    undefined, defined = result["contrasts"]
    assert undefined["estimate"] is None and undefined["reason"]
    assert defined["estimate"] == 1.0 and defined["simultaneous_interval"] is not None


def test_short_series_report_why_there_is_no_interval():
    estimate, draws, reason = long_short.bootstrap(
        np.zeros((1, 10, 1)),
        np.zeros((10, 1)),
        sessions_per_year=252,
        block_length=16,
        replicates=20,
        seed=1,
    )
    assert draws is None and "bloque" in reason
    assert long_short.level_intervals(estimate["mean_net_return"][0], None, 0.95)[0] == dict(
        estimate=0.0, interval=None
    )


def test_the_extremes_round_down_to_whole_assets():
    # Diez activos: k = floor(2,5) = 2, no 3.
    weights = long_short.quartile_weights(panel(np.arange(10.0)), **OPTIONS)
    assert np.count_nonzero(weights > 0) == 2 and np.count_nonzero(weights < 0) == 2
    assert weights[weights > 0].tolist() == [0.25, 0.25]


def test_drawdown_counts_losses_from_the_initial_capital():
    stats = long_short.statistics(np.array([[-0.1], [0.05]]), np.ones((2, 1)), 252)
    assert stats["max_drawdown"][0] == pytest.approx(0.1)


def test_each_replicate_keeps_the_order_of_its_resampled_sessions():
    from mars_titan.evaluation.paired_comparisons import circular_block_indices

    rng = np.random.default_rng(8)
    returns = rng.normal(0, 0.02, (1, 60, 2))
    traded = rng.uniform(1, 2, (60, 2))
    _, draws, _ = long_short.bootstrap(
        returns, traded, sessions_per_year=252, block_length=5, replicates=40, seed=13
    )
    index = circular_block_indices(np.random.default_rng(13), 40, 60, 5)
    for replicate in (0, 17, 39):
        expected = long_short.statistics(
            returns[:, index[replicate], :], traded[index[replicate]][None], 252
        )
        for name in long_short.STATISTICS:
            assert np.allclose(draws[name][replicate], expected[name], equal_nan=True), name
