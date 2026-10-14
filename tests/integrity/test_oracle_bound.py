"""Cota con información futura de un paso sobre cintas escritas en la prueba, sin aprender."""

import math

import numpy as np
import pytest

from mars_titan.integrity import oracle_bound
from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.evaluation import evaluate
from mars_titan.simulation.portfolio import CorporateAction
from tests.simulation.test_financial_adversarial import DAY, START, custom, flat


def walk(sessions=60, assets=8, seed=11):
    """Precios con aperturas y cierres distintos y volumen amplio. No proceden de ningún modelo."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (sessions, assets)), axis=0))
    open_ = close * np.exp(rng.normal(0, 0.01, (sessions, assets)))
    high, low = np.maximum(open_, close) * 1.01, np.minimum(open_, close) * 0.99
    volume = np.full((sessions, assets), 1e7)
    return np.stack([open_, high, low, close, volume], axis=2)


def test_scores_are_the_return_of_the_holding_each_decision_opens():
    prices = walk(sessions=5, assets=2)
    scores = oracle_bound.oracle_scores(custom(prices))
    # La tenencia va de la apertura siguiente a la posterior y la última, al cierre final.
    np.testing.assert_array_equal(scores[:-2], prices[2:, :, 0] / prices[1:-1, :, 0] - 1)
    np.testing.assert_array_equal(scores[-2], prices[-1, :, 3] / prices[-1, :, 0] - 1)
    assert np.isnan(scores[-1]).all()


def test_a_later_open_changes_only_the_holdings_that_cross_it():
    prices = walk(sessions=8, assets=3)
    changed = prices.copy()
    changed[6, :, 0] *= 1.5
    before = oracle_bound.oracle_scores(custom(prices))
    after = oracle_bound.oracle_scores(custom(changed))
    np.testing.assert_array_equal(before[:4], after[:4])
    assert not np.allclose(before[4], after[4]) and not np.allclose(before[5], after[5])
    np.testing.assert_array_equal(before[6], after[6])


def test_corporate_actions_inside_a_holding_enter_its_return():
    prices = np.tile([100.0, 100.0, 100.0, 100.0, 1e6], (4, 3, 1))
    # A0 se desdobla dos por una en la apertura de la sesión 2 y su precio se reduce a la mitad.
    prices[2:, 0, :4] = 50.0
    opening = [START + session * DAY - 23_400_000_000 for session in range(4)]
    actions = [
        CorporateAction("d", "US/A1", "dividend", opening[2], 1.0, opening[3], verified=True),
        CorporateAction("s", "US/A0", "split", opening[2], 2.0, verified=True),
        CorporateAction("w", "US/A2", "writeoff", opening[3], 0, verified=True),
    ]
    scores = oracle_bound.oracle_scores(custom(prices, actions=actions))
    # Decisión 0: tenencia de la apertura 1 a la 2, con el desdoblamiento y el dividendo.
    np.testing.assert_array_equal(scores[0], [0.0, 101 / 100 - 1, 0.0])
    # Decisión 1: la baja sin recuperación cae dentro de la tenencia de A2, que lo pierde todo.
    np.testing.assert_array_equal(scores[1], [0.0, 0.0, -1.0])


def test_a_priced_delisting_pays_its_amount_and_an_unpriced_one_has_no_return():
    prices = np.tile([100.0, 100.0, 100.0, 100.0, 1e6], (4, 3, 1))
    prices[2:, :2] = np.nan
    scores = np.full(prices.shape[:2], 0.01)
    scores[2:, :2] = np.nan
    opening = [START + session * DAY - 23_400_000_000 for session in range(4)]
    actions = [
        CorporateAction("p", "US/A0", "delisting", opening[2], 150.0, opening[3], verified=True),
        CorporateAction("u", "US/A1", "unpriced_delisting", opening[2], 0, verified=True),
    ]
    growth = oracle_bound.holding_growth(custom(prices, scores, actions=actions))
    # El cobro de la baja entra en la tenencia aunque el activo ya no tenga precio.
    assert growth[0, 0] == 150 / 100 and growth[0, 2] == 1.0
    assert np.isnan(growth[0, 1]) and np.isnan(growth[1:, :2]).all()


def test_a_missing_open_is_valued_at_its_close_and_cannot_be_bought():
    prices = walk(sessions=6, assets=2)
    prices[3, 0, 0] = np.nan
    growth = oracle_bound.holding_growth(custom(prices))
    # La apertura ausente no se negocia: el cierre de su sesión reparte la tenencia en dos.
    assert growth[1, 0] == prices[3, 0, 3] / prices[2, 0, 0]
    assert growth[2, 0] == prices[4, 0, 0] / prices[3, 0, 3]
    assert growth[1, 0] * growth[2, 0] == pytest.approx(prices[4, 0, 0] / prices[2, 0, 0])
    scores = oracle_bound.oracle_scores(custom(prices))
    assert scores[1, 0] == growth[1, 0] - 1 and np.isnan(scores[2, 0])
    np.testing.assert_array_equal(scores[:-1, 1], growth[:-1, 1] - 1)


def test_the_exact_bound_adds_the_best_holding_or_cash_of_each_decision():
    prices = walk(sessions=30, assets=4)
    growth = np.concatenate([prices[2:, :, 0], prices[-1:, :, 3]]) / prices[1:, :, 0]
    expected = np.log(np.maximum(growth.max(axis=1), 1.0)).sum()
    assert oracle_bound.exact_log_growth(custom(prices)) == pytest.approx(expected, rel=1e-14)
    assert oracle_bound.exact_log_growth(flat(sessions=8, assets=3)) == 0.0
    # Una decisión sin ningún activo valorado solo puede quedarse en caja.
    prices[5:7] = np.nan
    expected = np.log(np.maximum(np.nan_to_num(growth, nan=1.0).max(axis=1), 1.0))
    expected[3:6] = 0.0
    assert oracle_bound.exact_log_growth(custom(prices)) == pytest.approx(expected.sum())


def test_the_oracle_tape_has_its_own_identity_and_is_marked():
    tape = custom(walk())
    oracle = oracle_bound.oracle_tape(tape)
    assert oracle.sha256 != tape.sha256
    assert oracle.identity["parent_id"] == oracle_bound.ORACLE_PARENT
    assert oracle.identity["source"]["excluded_from_comparisons"] is True
    assert oracle.identity["source"]["oracle_of"] == tape.sha256


def test_the_oracle_cannot_earn_on_a_flat_market():
    result = oracle_bound.bound(flat(sessions=8, assets=3))
    assert result["log_growth"]["oracle"] == 0.0
    assert result["policies"]["cash"]["net_return"] == 0.0
    # Puntuaciones positivas constantes: compra a precio plano y paga costes.
    assert result["log_growth"]["tape_scores"] < 0
    assert result["fraction_of_oracle"] is None and not result["suspicious"]
    assert result["log_growth"]["exact"] == 0.0


def test_noise_scores_stay_far_below_the_oracle():
    prices = walk()
    noise = np.random.default_rng(3).normal(0, 0.01, prices.shape[:2])
    result = oracle_bound.bound(custom(prices, noise))
    assert result["excluded_from_comparisons"] is True
    assert result["log_growth"]["oracle"] > 0.5
    assert result["log_growth"]["oracle"] > result["log_growth"]["tape_scores"]
    assert not result["suspicious"]


def test_scores_with_future_information_are_flagged():
    prices = walk()
    leaked = oracle_bound.oracle_scores(custom(prices))
    result = oracle_bound.bound(custom(prices, np.nan_to_num(leaked, nan=0.0)))
    assert result["fraction_of_oracle"] == pytest.approx(1.0)
    assert result["suspicious"]


def concentrated(tape, count):
    """Puntuaciones con información futura que solo dejan positivos los mejores activos."""
    future = oracle_bound.oracle_scores(tape)
    order = np.argsort(-np.nan_to_num(future, nan=-np.inf), axis=1, kind="stable")
    marked = np.full(future.shape, -1.0)
    np.put_along_axis(marked, order[:, :count], 1.0, axis=1)
    marked[~(future > 0)] = -1.0
    marked[np.isnan(tape.scores)] = np.nan
    return marked


def episode_growth(tape, scores, **options):
    env = FinancialEnv(custom(tape.prices, scores, actions=tape.actions), **options)
    result = evaluate(env, oracle_bound.full_exposure)["financial_validation"]
    return math.log1p(result["net_return"])


def test_a_policy_that_concentrates_exceeds_the_rule_but_not_the_exact_bound():
    # Caso de la revisión de #584: el mejor activo de cada decisión supera a la política con
    # información futura del cuartil, que no es un máximo, y sigue por debajo de la exacta.
    tape = custom(walk(60, 8))
    result = oracle_bound.bound(tape, env_options=dict(cost_bps=0))
    top = episode_growth(tape, concentrated(tape, 1), cost_bps=0)
    assert top > result["log_growth"]["oracle"]
    assert top <= result["log_growth"]["exact"]


def premium_delisting():
    """A0 se compra a 100 y se da de baja en la apertura siguiente con un cobro de 150."""
    prices = np.tile([100.0, 100.0, 100.0, 100.0, 1e6], (5, 2, 1))
    prices[2:, 0] = np.nan
    scores = np.full(prices.shape[:2], -0.01)
    scores[0, 0], scores[2:, 0] = 0.01, np.nan
    opening = [START + session * DAY - 23_400_000_000 for session in range(5)]
    action = CorporateAction(
        "p", "US/A0", "delisting", opening[2], 150.0, opening[3], verified=True
    )
    return custom(prices, scores, actions=[action])


def missing_open():
    """A0 no abre en la sesión 2 y la posición comprada en la 1 llega a la apertura 3."""
    prices = np.tile([100.0, 100.0, 100.0, 100.0, 1e6], (5, 2, 1))
    prices[2, 0] = [np.nan, 130.0, 110.0, 120.0, 1e6]
    prices[3:, 0, :4] = 150.0
    scores = np.full(prices.shape[:2], -0.01)
    scores[:, 0] = 0.01
    return custom(prices, scores)


@pytest.mark.parametrize("build", [premium_delisting, missing_open])
def test_holdings_through_delistings_and_missing_opens_stay_inside_the_exact_bound(build):
    tape = build()
    growth = episode_growth(tape, tape.scores, cost_bps=0, participation=1.0)
    exact = oracle_bound.exact_log_growth(tape)
    # El episodio gana de verdad con esa tenencia, y la cota la recoge.
    assert growth > 0.3
    assert growth <= exact + oracle_bound.EXACT_TOLERANCE


@pytest.mark.parametrize("seed", range(4))
def test_no_policy_without_costs_exceeds_the_exact_bound(seed):
    rng = np.random.default_rng(seed)
    tape = custom(walk(sessions=40, assets=6, seed=seed))
    exact = oracle_bound.exact_log_growth(tape)
    future = np.nan_to_num(oracle_bound.oracle_scores(tape), nan=0.0)
    candidates = [concentrated(tape, 1), concentrated(tape, 2), future]
    candidates.append(rng.normal(0, 0.01, future.shape))
    for scores in candidates:
        for participation in (0.01, 1.0):
            growth = episode_growth(tape, scores, cost_bps=0, participation=participation)
            assert growth <= exact + oracle_bound.EXACT_TOLERANCE
