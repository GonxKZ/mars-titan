"""Regla de momentum y reversión de los controles comunes de la criba (#746).

Las ventanas se construyen a mano con los canales de la v3.1, así que cada componente tiene un
valor conocido. La regla no ajusta nada y las pruebas no ejecutan ningún optimizador.
"""

import numpy as np
import pytest

from mars_titan.data.price_windows import CONTEXT_SESSIONS, PRICE_WINDOW_CHANNELS
from mars_titan.models.baselines import momentum_reversal as rule

CONFIG = dict(identity=rule.IDENTITY, reversal_sessions=5, combination=rule.COMBINATION)
LAST, LAG = CONTEXT_SESSIONS - 1, CONTEXT_SESSIONS - 6


def windows(closes, *, seed=3):
    """Ventanas con el cierre de cada fila en t − 5 y en t y ruido en los demás canales."""
    generator = np.random.default_rng(seed)
    rows = len(closes)
    prices = generator.normal(size=(rows, CONTEXT_SESSIONS, len(PRICE_WINDOW_CHANNELS)))
    prices[:, :, rule.PRESENT] = 1.0
    prices[:, 0, rule.CLOSE] = 0.0
    for row, (lagged, last) in enumerate(closes):
        prices[row, LAG, rule.CLOSE], prices[row, LAST, rule.CLOSE] = lagged, last
    return prices.astype(np.float32)


def reference(values, sessions):
    """Rangos centrados escritos fila a fila: posición media entre los iguales de la sesión."""
    result = []
    for row, value in enumerate(values):
        peers = values[sessions == sessions[row]]
        if len(peers) == 1:
            result.append(0.0)
            continue
        below, equal = np.sum(peers < value), np.sum(peers == value)
        result.append((below + (equal - 1) / 2) / (len(peers) - 1) - 0.5)
    return np.asarray(result)


def test_the_components_read_the_close_of_the_decision_and_of_k_sessions_before():
    prices = windows([(0.10, 0.25), (-0.05, -0.20)])
    parts = rule.components(prices, 5)
    reversal, reversal_defined = parts["reversal"]
    momentum, momentum_defined = parts["momentum"]
    np.testing.assert_allclose(reversal, [-(0.25 - 0.10), -(-0.20 + 0.05)], rtol=1e-6)
    np.testing.assert_allclose(momentum, [0.10, -0.05], rtol=1e-6)
    assert reversal_defined.all() and momentum_defined.all()
    # El resto de canales y de posiciones no entra en ningún componente.
    other = prices.copy()
    other[:, :, [0, 1, 2, 4]] += 7.0
    other[:, 1:LAG, rule.CLOSE] -= 3.0
    other[:, LAG + 1 : LAST, rule.CLOSE] += 2.0
    for name, (values, _) in rule.components(other, 5).items():
        np.testing.assert_array_equal(values, parts[name][0])


def test_the_reversal_horizon_is_the_declared_number_of_sessions():
    prices = windows([(0.0, 0.3)])
    prices[0, LAST - 2, rule.CLOSE] = 0.2
    parts = rule.components(prices, 2)
    np.testing.assert_allclose(parts["reversal"][0], [-(0.3 - 0.2)], rtol=1e-6)
    np.testing.assert_allclose(parts["momentum"][0], [0.2], rtol=1e-6)


def test_an_absent_position_leaves_out_the_components_that_need_it():
    prices = windows([(0.1, 0.2), (0.3, 0.1), (0.2, 0.4)])
    prices[0, LAG, :] = 0.0
    prices[1, 20, :] = 0.0
    prices[2, LAST, :] = 0.0
    parts = rule.components(prices, 5)
    assert parts["reversal"][1].tolist() == [False, True, False]
    assert parts["momentum"][1].tolist() == [False, True, True]


def test_the_ranks_are_centered_within_each_session():
    values = np.array([3.0, 1.0, 2.0, 2.0, 5.0, -1.0, 4.0])
    sessions = np.array([0, 0, 0, 0, 1, 1, 2])
    defined = np.ones(7, dtype=bool)
    ranks = rule.centered_ranks(values, defined, sessions)
    np.testing.assert_allclose(ranks, reference(values, sessions), rtol=0, atol=1e-15)
    np.testing.assert_allclose(ranks, [0.5, -0.5, 0.0, 0.0, 0.5, -0.5, 0.0], rtol=0, atol=1e-15)
    # Una fila sin valor no ocupa rango ni recibe otro que cero.
    defined[0] = False
    ranks = rule.centered_ranks(values, defined, sessions)
    assert ranks[0] == 0.0
    np.testing.assert_allclose(ranks[1:4], [-0.5, 0.25, 0.25], rtol=0, atol=1e-15)


def test_each_session_is_ranked_alone_and_the_order_of_its_rows_does_not_matter():
    generator = np.random.default_rng(8)
    values, sessions = generator.normal(size=40), np.repeat(np.arange(4), 10)
    defined = generator.random(40) > 0.2
    ranks = rule.centered_ranks(values, defined, sessions)
    np.testing.assert_allclose(
        ranks[defined], reference(values[defined], sessions[defined]), rtol=0, atol=1e-15
    )
    order = generator.permutation(40)
    shuffled = rule.centered_ranks(values[order], defined[order], sessions[order])
    np.testing.assert_array_equal(shuffled, ranks[order])
    # Cambiar otra sesión no mueve los rangos de la primera.
    moved = values.copy()
    moved[10:] *= -3.0
    np.testing.assert_array_equal(rule.centered_ranks(moved, defined, sessions)[:10], ranks[:10])


def test_the_prediction_is_the_mean_of_the_defined_components():
    prices = windows([(0.1, 0.2), (0.3, 0.1), (-0.2, -0.1), (0.0, 0.5), (0.25, 0.3)])
    prices[3, LAG, :] = 0.0
    prices[4, LAST, :] = 0.0
    sessions = np.zeros(5, dtype=np.int64)
    prediction = rule.predict(prices, sessions, CONFIG)
    parts = rule.components(prices, 5)
    expected = np.zeros(5)
    for values, defined in parts.values():
        expected += rule.centered_ranks(values, defined, sessions)
    expected[:3] /= 2
    np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-15)
    # La fila 3 no tiene ningún componente definido y vale cero.
    assert prediction[3] == 0.0
    # Sin la sesión de la decisión, la fila 4 solo tiene momentum, el tercero de cuatro, y la
    # media no lo divide entre los dos componentes.
    np.testing.assert_allclose(prediction[4], 1 / 6, rtol=0, atol=1e-15)
    # La fila 1 cae en la última semana y llega más alta a t − 5, así que encabeza los dos
    # componentes.
    assert prediction.dtype == np.float64 and np.argmax(prediction[:3]) == 1


def test_a_falling_week_ranks_above_a_rising_one_with_the_same_momentum():
    prices = windows([(0.0, -0.3), (0.0, 0.3)])
    prediction = rule.predict(prices, np.zeros(2, dtype=np.int64), CONFIG)
    assert prediction[0] > prediction[1]


@pytest.mark.parametrize(
    "config",
    [
        dict(CONFIG, identity="momentum_v2"),
        dict(CONFIG, reversal_sessions=0),
        dict(CONFIG, reversal_sessions=CONTEXT_SESSIONS - 1),
        dict(CONFIG, reversal_sessions=5.0),
        dict(CONFIG, combination="sum"),
        dict(CONFIG, extra=1),
        "momentum_reversal_v1",
    ],
)
def test_the_declaration_is_checked(config):
    with pytest.raises(ValueError, match="sesiones de reversión"):
        rule.checked_config(config)


def test_windows_outside_the_contract_are_refused():
    prices = windows([(0.1, 0.2)])
    with pytest.raises(ValueError, match="64 sesiones"):
        rule.components(prices[:, 1:], 5)
    with pytest.raises(ValueError, match="64 sesiones"):
        rule.components(prices[..., :5], 5)
    broken = prices.copy()
    broken[0, 3, 1] = np.nan
    with pytest.raises(ValueError, match="no finitos"):
        rule.components(broken, 5)
    broken = prices.copy()
    broken[0, 3, rule.PRESENT] = 0.5
    with pytest.raises(ValueError, match="cero o uno"):
        rule.components(broken, 5)
    with pytest.raises(ValueError, match="su sesión"):
        rule.predict(prices, np.zeros(2), CONFIG)
