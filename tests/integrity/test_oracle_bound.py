"""Cota con información futura de un paso sobre cintas escritas en la prueba, sin aprender."""

import numpy as np
import pytest

from mars_titan.integrity import oracle_bound
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
    # Decisión 1: la baja cae dentro de la tenencia de A2, que queda sin puntuación.
    np.testing.assert_array_equal(scores[1, :2], [0.0, 0.0])
    assert np.isnan(scores[1, 2])


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
