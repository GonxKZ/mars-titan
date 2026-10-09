"""Referencias sin predicción: cartera 1/N reequilibrada e índice de mercado comprado."""

import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.simulation import index_benchmark
from mars_titan.simulation.environment import ALLOCATIONS, FinancialEnv
from mars_titan.simulation.evaluation import (
    REBALANCE_SESSIONS,
    REFERENCE_ALLOCATIONS,
    evaluate,
    fixed_policy,
)
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.native_runtime import library_path
from mars_titan.simulation.portfolio import CorporateAction

SESSIONS = 50
ASSETS = ["A", "B", "C", "D"]


def tape(*, scores=None, writeoff=None, gap=True):
    rng = np.random.default_rng(11)
    closes = 20 * np.exp(np.cumsum(rng.normal(0, 0.01, (SESSIONS, len(ASSETS))), axis=0))
    prices = np.stack([closes, closes * 1.01, closes * 0.99, closes, np.full_like(closes, 1e7)], 2)
    if gap:
        # C no tiene cierre valorado en la sesión 21, la del segundo reequilibrio. Una cinta
        # reconstruida no lo permite, pero la regla de reparto no debe depender de ello.
        prices[21, 2, :4] = np.nan
    if scores is None:
        scores = np.tile([0.02, -0.01, np.nan, 0.005], (SESSIONS, 1))
    times = 100 * np.arange(1, SESSIONS + 1)
    actions = ()
    if writeoff is not None:
        at = int(times[30] - 50)
        actions = [CorporateAction("gone", writeoff, "writeoff", at, 0, verified=True)]
    return MarketTape(
        prices,
        times,
        ASSETS,
        scores,
        domain="synthetic",
        currency="USD",
        open_times=times - 50,
        actions=actions,
    )


def environment(source, allocation, **options):
    return FinancialEnv(source, capital=100_000, cost_bps=10, allocation=allocation, **options)


def test_the_default_allocation_keeps_the_identity_of_previous_environments():
    env = FinancialEnv(tape())
    assert env.allocation == ALLOCATIONS[0]
    assert env.identity["allocation"] == "positive_top_quartile_equal_weight"
    other = FinancialEnv(tape(), allocation=ALLOCATIONS[1])
    assert other.identity["allocation"] == "valid_assets_equal_weight"
    with pytest.raises(ValueError, match="composición"):
        FinancialEnv(tape(), allocation="best_in_hindsight")


def test_equal_weight_splits_among_every_valued_asset_and_ignores_scores():
    env = environment(tape(), ALLOCATIONS[1])
    env.reset(seed=1)
    targets = env._targets(1.0)
    nav = env.book.nav["USD"]
    # Las puntuaciones negativas o ausentes no excluyen a B ni a C.
    assert set(targets) == set(ASSETS)
    for i, asset in enumerate(ASSETS):
        assert targets[asset] == pytest.approx(nav / 4 / env.tape.prices[0, i, 3], rel=1e-15)
    ranked = environment(tape(), ALLOCATIONS[0])
    ranked.reset(seed=1)
    # La regla común solo compra el cuartil superior de puntuaciones positivas.
    assert ranked._targets(1.0).keys() == {"A"}


def test_equal_weight_does_not_depend_on_any_prediction():
    rng = np.random.default_rng(5)
    shuffled = rng.normal(0, 0.02, (SESSIONS, len(ASSETS)))
    shuffled[rng.random(shuffled.shape) < 0.3] = np.nan
    policy = fixed_policy("equal_weight_monthly")
    first = evaluate(environment(tape(), ALLOCATIONS[1]), policy)
    second = evaluate(environment(tape(scores=shuffled), ALLOCATIONS[1]), policy)
    assert first["equity"]["nav"] == second["equity"]["nav"]
    assert first["financial_validation"] == second["financial_validation"]


def test_the_monthly_portfolio_rebalances_every_21_sessions_and_holds_in_between():
    policy = fixed_policy("equal_weight_monthly")
    actions = [policy(None, step) for step in range(SESSIONS - 1)]
    assert REBALANCE_SESSIONS == 21
    assert [step for step, action in enumerate(actions) if action == 5] == [0, 21, 42]
    assert set(actions) == {0, 5}
    assert fixed_policy("market_index")(None, 0) == 5
    assert {fixed_policy("market_index")(None, step) for step in range(1, 30)} == {0}
    assert set(REFERENCE_ALLOCATIONS) == {
        "cash",
        "hold_initial",
        "rebalance_50",
        "equal_weight_monthly",
        "market_index",
    }


def test_a_rebalance_skips_an_asset_without_a_valued_close_and_a_retired_asset():
    env = environment(tape(writeoff="D"), ALLOCATIONS[1])
    env.reset(seed=1)
    for _ in range(21):
        env.step(0)
    # Sesión 21: C no tiene cierre valorado y queda fuera del reparto.
    assert env.cursor == 21
    targets = env._targets(1.0)
    assert {asset for asset, value in targets.items() if value > 0} == {"A", "B", "D"}
    for _ in range(21):
        env.step(0)
    # Sesión 42: D se dio de baja en la sesión 30 y C vuelve a estar valorado.
    assert "D" in env.book.retired
    targets = env._targets(1.0)
    assert {asset for asset, value in targets.items() if value > 0} == {"A", "B", "C"}


@pytest.mark.skipif(library_path() is None, reason="Falta la biblioteca nativa de simulación")
@pytest.mark.parametrize("reference", ["equal_weight_monthly", "market_index"])
def test_native_accounting_reproduces_the_references_without_predictions(reference):
    allocation = REFERENCE_ALLOCATIONS[reference]
    source = tape(writeoff="D", gap=False)
    ours = evaluate(environment(source, allocation), fixed_policy(reference))
    theirs = evaluate(environment(source, allocation, backend="native"), fixed_policy(reference))
    assert ours["financial_validation"]["completed"] is True
    np.testing.assert_allclose(theirs["equity"]["nav"], ours["equity"]["nav"], rtol=1e-12)
    for key in ("net_return", "costs", "turnover", "max_drawdown"):
        assert theirs["financial_validation"][key] == pytest.approx(
            ours["financial_validation"][key], rel=1e-12, abs=1e-12
        )


# Índice chino calculado con niveles diarios


def us(day, hour):
    return int(np.datetime64(f"{day}T{hour}", "us").astype(np.int64))


# Cierres de cuatro sesiones de Shanghái: 15:05 en Pekín son las 07:05 UTC.
CLOSES = [us(day, "07:05") for day in ("2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06")]


def write_levels(path, rows):
    table = pa.table(
        dict(
            session=pa.array([row[0] for row in rows], pa.string()),
            open=pa.array([row[1] for row in rows], pa.float64()),
            close=pa.array([row[2] for row in rows], pa.float64()),
        )
    )
    pq.write_table(table, path)
    return index_benchmark.read_levels(path, sha256(path))


def test_the_index_buys_at_the_next_open_and_values_each_close(tmp_path):
    levels = write_levels(
        tmp_path / "levels.parquet",
        [
            ("2022-12-30", 3800.0, 3870.0),
            ("2023-01-03", 3880.0, 3900.0),
            ("2023-01-04", 3910.0, 3950.0),
            ("2023-01-05", 3960.0, 3940.0),
            ("2023-01-06", 3930.0, 4000.0),
        ],
    )
    record, reason = index_benchmark.index_episode(
        levels, CLOSES, market="CN", capital=1_000_000, cost_bps=10
    )
    assert reason is None
    units = 1_000_000 / (3910.0 * 1.001)
    assert record["equity"]["nav"] == pytest.approx(
        [1_000_000, units * 3950, units * 3940, units * 4000], rel=1e-15
    )
    assert record["equity"]["close_times"] == CLOSES
    assert record["net_return"] == pytest.approx(units * 4000 / 1_000_000 - 1, rel=1e-14)
    assert record["liquidated_net_return"] == pytest.approx(
        units * 4000 * 0.999 / 1_000_000 - 1, rel=1e-14
    )
    assert record["costs"] == pytest.approx(units * 3910 * 0.001, rel=1e-12)
    assert record["turnover"] == pytest.approx(1 / 1.001, rel=1e-14)
    assert record["max_drawdown"] == pytest.approx(1 - 3940 / 3950, rel=1e-12)
    assert record["steps"] == 3


def test_missing_levels_keep_the_last_close_and_delay_the_purchase(tmp_path):
    levels = write_levels(
        tmp_path / "levels.parquet",
        [
            ("2023-01-03", 3880.0, 3900.0),
            # Sin apertura el 4 de enero: la compra pasa a la apertura del 5.
            ("2023-01-04", math.nan, 3950.0),
            ("2023-01-05", 3960.0, 3940.0),
        ],
    )
    record, _ = index_benchmark.index_episode(levels, CLOSES, market="CN", capital=1000, cost_bps=0)
    units = 1000 / 3960.0
    # El 6 de enero no tiene nivel y conserva el último cierre conocido.
    assert record["equity"]["nav"] == pytest.approx([1000, 1000, units * 3940, units * 3940])


def test_future_levels_do_not_change_earlier_closes(tmp_path):
    rows = [
        ("2023-01-03", 3880.0, 3900.0),
        ("2023-01-04", 3910.0, 3950.0),
        ("2023-01-05", 3960.0, 3940.0),
        ("2023-01-06", 3930.0, 4000.0),
    ]
    base = write_levels(tmp_path / "a.parquet", rows)
    changed = write_levels(tmp_path / "b.parquet", [*rows[:3], ("2023-01-06", 1.0, 1.0)])
    first, _ = index_benchmark.index_episode(base, CLOSES, market="CN", capital=1, cost_bps=5)
    second, _ = index_benchmark.index_episode(changed, CLOSES, market="CN", capital=1, cost_bps=5)
    assert first["equity"]["nav"][:3] == second["equity"]["nav"][:3]
    assert first["equity"]["nav"][3] != second["equity"]["nav"][3]


def test_an_episode_without_levels_is_reported_as_unavailable(tmp_path):
    later = write_levels(
        tmp_path / "later.parquet",
        [("2023-01-04", 3910.0, 3950.0), ("2023-01-05", 3960.0, 3940.0)],
    )
    assert index_benchmark.index_episode(later, CLOSES, market="CN", capital=1, cost_bps=0) == (
        None,
        index_benchmark.UNAVAILABLE,
    )
    no_open = write_levels(
        tmp_path / "no-open.parquet",
        [("2023-01-03", 3880.0, 3900.0), ("2023-01-04", math.nan, 3950.0)],
    )
    assert index_benchmark.index_episode(no_open, CLOSES, market="CN", capital=1, cost_bps=0)[1]


def test_levels_are_read_only_with_their_declared_digest(tmp_path):
    path = tmp_path / "levels.parquet"
    write_levels(path, [("2023-01-03", 3880.0, 3900.0), ("2023-01-04", 3910.0, 3950.0)])
    with pytest.raises(ValueError, match="huella"):
        index_benchmark.read_levels(path, "0" * 64)
    for rows, match in (
        ([("2023-01-04", 1.0, 1.0), ("2023-01-03", 1.0, 1.0)], "crecientes"),
        ([("2023-01-03", 1.0, math.inf), ("2023-01-04", 1.0, 1.0)], "positivos"),
        ([("2023-01-03", 1.0, -1.0), ("2023-01-04", 1.0, 1.0)], "positivos"),
    ):
        with pytest.raises(ValueError, match=match):
            write_levels(path, rows)


def test_local_sessions_follow_the_time_zone_of_each_market():
    # 21:05 UTC del 3 de enero es el cierre de Nueva York de ese día y las 05:05 del 4 en Pekín.
    late = us("2023-01-03", "21:05")
    assert str(index_benchmark.local_sessions([late], "US")[0]) == "2023-01-03"
    assert str(index_benchmark.local_sessions([late], "CN")[0]) == "2023-01-04"
    assert str(index_benchmark.local_sessions(CLOSES[:1], "CN")[0]) == "2023-01-03"
