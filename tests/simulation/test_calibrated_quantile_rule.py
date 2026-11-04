"""Regla fija sobre cuantiles calibrados y no operar, G2 (#748).

Las cintas son sintéticas y escritas a mano. Nada aprende ni actualiza parámetros.
"""

import numpy as np
import pytest

from mars_titan.evaluation import neutral_abstention as na
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.models.quantile_head import LEVELS
from mars_titan.simulation import calibrated_quantile_rule as rule
from mars_titan.simulation.environment import ACTIONS, ALLOCATIONS, FinancialEnv
from mars_titan.simulation.evaluation import evaluate
from mars_titan.simulation.market import MarketTape

OFFSETS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


def tape(scores, missing=()):
    """Cinta con cierres que suben un 1 % por sesión y volumen suficiente para ejecutar.

    `missing` lista pares (sesión, activo) sin cierre valorado.
    """
    scores = np.asarray(scores, dtype=np.float64)
    sessions, assets = scores.shape
    close = 10 * 1.01 ** np.arange(sessions)
    prices = np.empty((sessions, assets, 5))
    prices[:, :, :4] = close[:, None, None]
    prices[:, :, 4] = 1e9
    for session, asset in missing:
        prices[session, asset, 3] = np.nan
    times = 100 * (np.arange(sessions) + 1)
    return MarketTape(
        prices,
        times,
        [f"A{index}" for index in range(assets)],
        scores,
        domain="synthetic",
        currency="USD",
        partition="validation",
        open_times=times - 50,
    )


def recorded(policy):
    actions = []

    def wrapped(observation, step):
        actions.append(policy(observation, step))
        return actions[-1]

    return wrapped, actions


def test_the_rule_invests_only_when_some_asset_has_a_positive_claim():
    scores = [[0.0, 0.0], [np.nan, -0.01], [0.0, 0.02], [0.0, 0.0], [0.03, 0.0]]
    policy, actions = recorded(rule.rule_policy)
    evaluate(FinancialEnv(tape(scores), capital=1000, cost_bps=0), policy)
    # Una decisión por sesión salvo la última, que no tiene apertura siguiente.
    assert actions == [rule.CASH, rule.CASH, rule.INVEST, rule.CASH]
    assert ACTIONS[rule.INVEST] == 1.0 and ACTIONS[rule.CASH] == 0.0


def test_a_claim_on_an_asset_without_a_valued_close_does_not_invest():
    scores = [[0.0, 0.0], [0.02, 0.0], [0.0, 0.0]]
    policy, actions = recorded(rule.rule_policy)
    evaluate(FinancialEnv(tape(scores, missing=[(1, 0)]), capital=1000, cost_bps=0), policy)
    assert actions == [rule.CASH, rule.CASH]


def test_the_exposure_goes_only_to_assets_with_an_upward_claim():
    scores = np.zeros((4, 8))
    scores[1, [2, 5]] = [0.01, 0.03]
    scores[1, 6] = -0.05
    env = FinancialEnv(tape(scores), capital=10_000, cost_bps=0, participation=1.0)
    observation, _ = env.reset(seed=42)
    for step in range(2):
        observation, *_ = env.step(rule.rule_policy(observation, step))
    # Con dos afirmaciones positivas el cuartil superior es la mayor de ellas.
    assert set(env.book.positions) == {"A5"}


def test_no_trade_keeps_the_capital_in_cash():
    scores = np.full((5, 3), 0.02)
    reports = rule.compare(FinancialEnv(tape(scores), capital=1000, cost_bps=10))
    assert reports["no_trade"]["financial_validation"]["net_return"] == 0
    rising = reports[rule.IDENTITY]["financial_validation"]["net_return"]
    assert rising > 0
    again = rule.compare(FinancialEnv(tape(scores), capital=1000, cost_bps=10))
    for name, report in reports.items():
        report.pop("elapsed_seconds"), again[name].pop("elapsed_seconds")
        assert again[name] == report
    other = FinancialEnv(tape(scores), allocation=ALLOCATIONS[1])
    with pytest.raises(ValueError, match="cuartil superior"):
        rule.compare(other)


def panel(rng, start, sessions, assets, shift=0.0):
    rows = sessions * assets
    at = start + np.repeat(np.arange(sessions), assets) * 86_400_000_000
    ids = [f"US/A{index % assets}/{t}" for index, t in enumerate(at)]
    median = rng.normal(shift, 0.01, rows)
    spread = rng.uniform(0.002, 0.01, rows)
    target = median + rng.normal(0, 0.005, rows)
    result = ForecastPanel.from_columns(
        ids,
        np.full(rows, "US"),
        at,
        target,
        median,
        quantiles=median[:, None] + spread[:, None] * OFFSETS,
        levels=LEVELS,
    )
    return result, result.prediction_at + 86_400_000_000


def test_the_rule_reads_the_a2_claims_of_the_calibrated_intervals():
    rng = np.random.default_rng(8)
    start = 1_577_836_800_000_000
    calibration = panel(rng, start, 30, 4)
    evaluation, _ = panel(rng, start + 40 * 86_400_000_000, 12, 4, shift=0.004)
    record = na.fit(*calibration, na.DEFAULT | dict(min_rows=20))
    emitted, abstained = na.apply(record, evaluation)
    # El panel ordena por instante e identidad, así que cada sesión es una fila de la cinta.
    scores = emitted.reshape(12, 4)
    assert 0 < abstained.sum() < evaluation.rows
    policy, actions = recorded(rule.rule_policy)
    evaluate(FinancialEnv(tape(scores), capital=1000, cost_bps=0), policy)
    expected = [rule.INVEST if (row > 0).any() else rule.CASH for row in scores[:-1]]
    assert actions == expected
    assert rule.INVEST in actions and rule.CASH in actions


@pytest.mark.parametrize("observation", [np.zeros(7), np.zeros(15), np.zeros(1), np.zeros((2, 8))])
def test_observations_without_the_environment_layout_are_rejected(observation):
    with pytest.raises(ValueError, match="formato"):
        rule.rule_policy(observation, 0)


def test_the_rule_reads_the_observation_layout_of_the_environment():
    # La regla repite el número de rasgos por activo sin importarlo de `environment.py`,
    # porque ese archivo forma parte de la identidad de las ejecuciones de políticas y de las
    # etapas de la campaña. Esta prueba falla si el entorno cambia el formato.
    for assets in (1, 3, 8):
        env = FinancialEnv(tape(np.zeros((3, assets))))
        assert env.observation_space.shape == (rule.FEATURES_PER_ASSET * assets + 2,)
    scores = np.zeros((3, 3))
    scores[0, 1] = 0.02
    observation, _ = FinancialEnv(tape(scores, missing=[(0, 2)])).reset(seed=42)
    features = observation[:-2].reshape(3, rule.FEATURES_PER_ASSET)
    assert features[1, rule.SCORE] > 0 and features[0, rule.SCORE] == 0
    assert features[:, rule.VALID].tolist() == [1.0, 1.0, 0.0]


def test_a_ruined_or_unvalued_book_stays_in_cash():
    # La observación de una cartera sin patrimonio es todo ceros.
    assert rule.rule_policy(np.zeros(6 * 3 + 2, dtype=np.float32), 0) == rule.CASH
