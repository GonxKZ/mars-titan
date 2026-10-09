"""Políticas guionizadas que intentan obtener recompensa financiera sin señal legítima."""

import math

import numpy as np
import pytest

from mars_titan.episodes.worlds import WorldConfig, generate_world
from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.evaluation import evaluate
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.portfolio import CorporateAction
from tests.simulation.native_library import NATIVE_BACKEND

DAY = 86_400_000_000
START = 946_684_800_000_000 + 16 * 3_600_000_000
BACKENDS = ("python", NATIVE_BACKEND)
EVIDENCE = {
    "price_basis": "unadjusted",
    "corporate_actions_complete": True,
    "evidence_sha256": "a" * 64,
}


def custom(prices, scores=None, *, actions=(), **options):
    prices = np.asarray(prices, dtype=np.float64)
    sessions, assets = prices.shape[:2]
    return MarketTape(
        prices,
        START + np.arange(sessions, dtype=np.int64) * DAY,
        [f"US/A{i}" for i in range(assets)],
        np.full((sessions, assets), 0.01) if scores is None else scores,
        domain=options.pop("domain", "synthetic"),
        currency="USD",
        actions=actions,
        **options,
    )


def flat(sessions=6, assets=2):
    return custom(np.tile([100.0, 100.0, 100.0, 100.0, 1e6], (sessions, assets, 1)))


def world_tape():
    world = generate_world(WorldConfig(assets=4, sessions=24, context=8))
    return MarketTape.from_world(world, lambda inputs: inputs["news"][:, 0] * 0.002)


def run(env, policy):
    observation, _ = env.reset(seed=7)
    observations, results, step = [observation], [], 0
    while not env.done:
        result = env.step(policy(step))
        observations.append(result[0])
        results.append(result)
        step += 1
    return observations, results


POLICIES = {
    "hold": lambda step: 0,
    "cash": lambda step: 1,
    "maximum_exposure": lambda step: 5,
    "maximum_turnover": lambda step: 5 if step % 2 == 0 else 1,
    "buy_last_session": lambda step: 5 if step >= 3 else 1,
}


@pytest.mark.parametrize("name", sorted(POLICIES))
@pytest.mark.parametrize("backend", BACKENDS)
def test_scripted_policies_cannot_earn_on_a_flat_market_with_default_costs(name, backend):
    env = FinancialEnv(flat(), backend=backend)
    assert env.cost_bps == 10
    _, results = run(env, POLICIES[name])
    rewards = [result[1] for result in results]
    assert all(result[4]["reward_valid"] for result in results)
    assert sum(rewards) <= 0 and max(rewards) <= 0
    trades = sum(len(result[4]["trades"]) for result in results)
    # Sin movimiento de precios, toda operación cuesta y la inactividad vale cero.
    assert (sum(rewards) < 0) == (trades > 0)
    if name in {"hold", "cash"}:
        assert rewards == [0.0] * len(rewards)


@pytest.mark.parametrize("name", sorted(POLICIES))
@pytest.mark.parametrize("backend", BACKENDS)
def test_rewards_telescope_without_terminal_or_reset_bonus(name, backend):
    env = FinancialEnv(world_tape(), backend=backend)
    first = run(env, POLICIES[name])[1]
    total = math.fsum(result[1] for result in first)
    final = first[-1][4]["nav"]["USD"]
    assert first[-1][3] and first[-1][4]["reason"] == "episode_limit"
    assert total == pytest.approx(math.log(final) - math.log(env.capital), rel=0, abs=1e-12)
    # Un reinicio no arrastra efectivo, posiciones ni recompensas del episodio anterior.
    second = run(env, POLICIES[name])[1]
    assert [r[1:] for r in second] == [r[1:] for r in first]


@pytest.mark.parametrize("backend", BACKENDS)
def test_future_perturbation_does_not_change_past_observations_or_rewards(backend):
    base = world_tape()
    cut = 6
    prices, scores = np.array(base.prices), np.array(base.scores)
    prices[cut + 1 :, :, :4] *= 1.5
    prices[cut + 1 :, :, 4] *= 3
    scores[cut + 1 :] = -scores[cut + 1 :]
    changed = MarketTape(
        prices,
        base.close_times,
        base.assets,
        scores,
        domain="synthetic",
        currency="USD",
    )
    actions = [5, 0, 3, 1, 4, 5, 2, 0, 5, 1]
    one, two = FinancialEnv(base, backend=backend), FinancialEnv(changed, backend=backend)
    observations = [one.reset(seed=1)[0]], [two.reset(seed=1)[0]]
    for step, action in enumerate(actions):
        a, b = one.step(action), two.step(action)
        if step < cut:
            # La recompensa del paso t solo usa la apertura y el cierre de t + 1.
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1:] == b[1:]
        observations[0].append(a[0])
        observations[1].append(b[0])
    for left, right in zip(observations[0][: cut + 1], observations[1][: cut + 1], strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("backend", BACKENDS)
def test_orders_fill_at_the_next_open_not_at_the_decision_close(backend):
    prices = np.array(
        [
            [[100.0, 100.0, 100.0, 100.0, 1e6]],
            [[150.0, 150.0, 150.0, 150.0, 1e6]],
            [[150.0, 150.0, 150.0, 150.0, 1e6]],
        ]
    )
    env = FinancialEnv(custom(prices), backend=backend)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(5)
    # La orden se dimensiona con el cierre de 100 y se ejecuta con la apertura siguiente.
    assert [trade["price"] for trade in info["trades"]] == [150.0]
    quantity = info["trades"][0]["quantity"]
    assert quantity == math.floor(10_000 / (150 * 1.001))
    assert env.book.cash["USD"] >= 0
    assert reward == pytest.approx(math.log(info["nav"]["USD"] / 10_000))
    assert reward < 0


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("backend", BACKENDS)
def test_random_action_sequences_never_borrow_short_or_exceed_net_worth(seed, backend):
    env = FinancialEnv(world_tape(), backend=backend)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    while not env.done:
        _, reward, _, _, info = env.step(int(rng.integers(6)))
        assert math.isfinite(reward)
        cash, nav = env.book.cash["USD"], info["nav"]["USD"]
        assert cash >= 0 and nav > 0
        assert all(quantity > 0 for quantity in env.book.positions.values())
        invested = sum(
            quantity * env.tape.prices[env.cursor, env.tape.assets.index(asset), 3]
            for asset, quantity in env.book.positions.items()
        )
        assert invested <= nav * (1 + 1e-12)


@pytest.mark.parametrize(
    "action", [float("nan"), 5.0, True, np.bool_(True), -1, 6, np.array([5]), "5", None, 2**70]
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_invalid_actions_fail_without_changing_the_confirmed_state(action, backend):
    env = FinancialEnv(world_tape(), backend=backend)
    env.reset(seed=0)
    env.step(5)
    before = env.snapshot()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.snapshot() == before


@pytest.mark.parametrize("backend", BACKENDS)
def test_missing_close_of_a_held_asset_is_invalid_and_cannot_hide_a_loss(backend):
    prices = np.tile([100.0, 100.0, 100.0, 100.0, 1e6], (4, 2, 1))
    prices[2, 0, 3] = np.nan
    env = FinancialEnv(custom(prices), backend=backend)
    env.reset(seed=0)
    env.step(5)
    _, reward, terminated, truncated, info = env.step(0)
    assert (reward, terminated, truncated) == (0.0, False, True)
    assert info["reward_valid"] is False and info["reason"] == "missing_close"
    report = evaluate(FinancialEnv(custom(prices), backend=backend), lambda *_: 5)
    assert report["financial_validation"]["completed"] is False
    assert report["financial_validation"]["net_return"] is None

    # Una baja acreditada sí se contabiliza como pérdida.
    written = custom(
        np.tile([100.0, 100.0, 100.0, 100.0, 1e6], (4, 2, 1)),
        actions=[
            CorporateAction(
                "w", "US/A0", "writeoff", START + 2 * DAY - 23_400_000_000, 0, verified=True
            )
        ],
    )
    env = FinancialEnv(written, backend=backend)
    env.reset(seed=0)
    env.step(5)
    _, reward, _, _, info = env.step(0)
    assert info["reward_valid"] and reward < math.log(0.6)


def test_real_tapes_reject_predictions_fitted_after_the_decision():
    base = world_tape()
    options = dict(
        domain="real",
        currency="USD",
        open_times=base.open_times,
        prediction_times=base.close_times,
    )
    times = [int(t) for t in base.close_times]
    for fits in (None, times[:-1], [t + 1 for t in times], [float(t) for t in times]):
        with pytest.raises(ValueError, match="ajuste anterior"):
            MarketTape(
                base.prices,
                base.close_times,
                base.assets,
                base.scores,
                audit=dict(EVIDENCE, prediction_fit_ends=fits),
                **options,
            )
    # Un padre ajustado con toda la partición produce puntuaciones dentro de muestra.
    in_sample = [times[-1]] * len(times)
    with pytest.raises(ValueError, match="ajuste anterior"):
        MarketTape(
            base.prices,
            base.close_times,
            base.assets,
            base.scores,
            audit=dict(EVIDENCE, prediction_fit_ends=in_sample),
            **options,
        )
    walk_forward = [max(0, t - 2 * DAY) for t in times]
    tape = MarketTape(
        base.prices,
        base.close_times,
        base.assets,
        base.scores,
        audit=dict(EVIDENCE, prediction_fit_ends=walk_forward),
        **options,
    )
    assert tape.identity["audit"]["prediction_fit_ends"] == walk_forward


def test_ending_invested_reports_the_unpaid_exit_cost_without_changing_rewards():
    invested = evaluate(FinancialEnv(flat()), lambda *_: 5)
    cash = evaluate(FinancialEnv(flat()), lambda *_: 1)
    value = sum(invested["ending_positions"].values()) * 100.0
    exit_costs = value * 10 / 10000
    assert invested["terminal_liquidation"]["estimated_costs"] == pytest.approx(exit_costs)
    gap = (
        invested["financial_validation"]["net_return"]
        - invested["terminal_liquidation"]["net_return"]
    )
    assert gap == pytest.approx(exit_costs / 10_000)
    assert cash["terminal_liquidation"] == dict(
        basis="final_close_minus_cost_bps", estimated_costs=0.0, net_return=0.0
    )
    assert invested["terminal_liquidation"]["net_return"] < 0


def test_corporate_action_before_the_first_decision_is_rejected():
    prices = np.tile([100.0, 100.0, 100.0, 100.0, 1e6], (4, 2, 1))
    first_open = START - 23_400_000_000
    with pytest.raises(ValueError, match="primera decisión"):
        custom(
            prices,
            actions=[CorporateAction("w", "US/A0", "writeoff", first_open, 0, verified=True)],
        )


@pytest.mark.parametrize("decision", [1, 2, 5, 7])
@pytest.mark.parametrize("backend", BACKENDS)
def test_order_quantities_do_not_depend_on_the_execution_bar_or_later(decision, backend):
    base = world_tape()
    prices, scores = np.array(base.prices), np.array(base.scores)
    # La apertura de ejecución se conserva. Cierre, volumen y predicciones futuras cambian.
    prices[decision + 1 :, :, 1:4] *= 1.3
    prices[decision + 1 :, :, 4] *= 5
    prices[decision + 2 :, :, 0] *= 1.3
    scores[decision + 1 :] = -scores[decision + 1 :]
    changed = MarketTape(
        prices, base.close_times, base.assets, scores, domain="synthetic", currency="USD"
    )
    one, two = FinancialEnv(base, backend=backend), FinancialEnv(changed, backend=backend)
    one.reset(seed=3)
    two.reset(seed=3)
    actions = [5, 2, 4, 1, 5, 3, 0, 5]
    for step in range(decision + 1):
        a, b = one.step(actions[step]), two.step(actions[step])
    assert a[4]["trades"] and a[4]["trades"] == b[4]["trades"]
    assert a[4]["unfilled"] == b[4]["unfilled"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_costs_are_charged_on_every_buy_and_sell(backend):
    env = FinancialEnv(flat(), backend=backend)
    _, results = run(env, POLICIES["maximum_turnover"])
    trades = [trade for result in results for trade in result[4]["trades"]]
    assert {trade["quantity"] > 0 for trade in trades} == {True, False}
    for trade in trades:
        assert trade["cost"] == pytest.approx(abs(trade["quantity"] * trade["price"]) * 0.001)
    costs = results[-1][4]["costs"]["USD"]
    assert costs == pytest.approx(results[-1][4]["turnover"]["USD"] * 0.001)
    assert results[-1][4]["nav"]["USD"] == pytest.approx(env.capital - costs)
