"""Prueba de humo con la edición real, solo lectura y para ejecución local explícita.

Se activa declarando ``MARS_TITAN_UNADJUSTED_EDITION`` con la ruta de la edición. Construye
cintas de 2023 con pocos activos y puntuaciones sintéticas, que no proceden de ningún modelo,
y recorre los entornos con acciones fijas. La cinta china se compara también entre los motores
Python y nativo. No aprende ni evalúa políticas.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.evaluation import evaluate, fixed_policy
from mars_titan.simulation.market_rules import china_a_share_instrument
from mars_titan.simulation.reconstructed_tape import build_reconstructed_tape
from tests.simulation.native_library import requires_native_library
from tests.simulation.unadjusted_edition_fixture import evaluation_window, predictions

EDITION = os.environ.get("MARS_TITAN_UNADJUSTED_EDITION")
pytestmark = pytest.mark.skipif(
    EDITION is None, reason="Declara MARS_TITAN_UNADJUSTED_EDITION para leer la edición real"
)
ASSETS = {
    "US": ["AAPL", "IBM", "MSFT", "JNJ", "XOM"],
    "CN": ["600519.SS", "600239.SS", "000001.SZ", "300750.SZ", "688981.SS"],
}


@pytest.mark.parametrize("market", ["US", "CN"])
def test_real_edition_builds_a_2023_tape_and_fixed_actions_keep_the_accounting(market):
    values = predictions(market, ASSETS[market], score=lambda k, i: 0.01)
    tape, report = build_reconstructed_tape(
        Path(EDITION),
        [evaluation_window(market, values)],
        [values],
        market=market,
        partition="validation",
        dividend_payment_lag_sessions=0,
        symbols=ASSETS[market],
    )
    assert report["assets"] + sum(report["exclusions"].values()) == len(ASSETS[market])
    assert np.isfinite(tape.prices[:, :, 3]).all() and len(tape) > 200
    rules = (
        {asset: china_a_share_instrument(asset) for asset in tape.assets}
        if market == "CN"
        else None
    )
    env = FinancialEnv(tape, instruments=rules)
    env.reset(seed=0)
    trades = 0
    while not env.done:
        decision = env.cursor
        _, reward, _, _, info = env.step((5, 0, 0, 1)[decision % 4])
        assert np.isfinite(reward) and info["reward_valid"]
        for trade in info["trades"]:
            i = tape.assets.index(trade["asset"])
            assert trade["price"] == tape.prices[decision + 1, i, 0]
            assert tape.prices[decision, i, 4] > 0
            if market == "CN" and trade["quantity"] > 0:
                assert trade["quantity"] % rules[trade["asset"]].lot == 0
            trades += 1
    assert trades > 0
    print(report)


@requires_native_library
def test_real_chinese_tape_has_the_same_trajectory_in_the_native_engine():
    values = predictions("CN", ASSETS["CN"], score=lambda k, i: 0.01 * ((i + k) % 5 - 1))
    tape, _ = build_reconstructed_tape(
        Path(EDITION),
        [evaluation_window("CN", values)],
        [values],
        market="CN",
        partition="validation",
        dividend_payment_lag_sessions=2,
        symbols=ASSETS["CN"],
    )
    rules = {asset: china_a_share_instrument(asset) for asset in tape.assets}
    for plan in ((5, 0, 0, 1), (5, 1), (3, 5, 2, 0, 4)):
        reference = FinancialEnv(tape, capital=1_000_000, instruments=rules)
        native = FinancialEnv(tape, capital=1_000_000, instruments=rules, backend="native")
        np.testing.assert_array_equal(reference.reset(seed=0)[0], native.reset(seed=0)[0])
        reasons = set()
        while not reference.done:
            action = plan[reference.cursor % len(plan)]
            expected, actual = reference.step(action), native.step(action)
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1:] == expected[1:]
            assert native.book.snapshot()["state"] == reference.book.snapshot()["state"]
            reasons.update(entry["reason"] for entry in expected[4]["unfilled"])
        assert native.done and native.book.execution_counts["native_steps"] > 0
        print(plan, native.book.execution_counts, sorted(reasons))


@requires_native_library
def test_dvn_keeps_the_2009_window_with_its_last_close_valued_at_the_previous_trade():
    # DVN no tiene fila el 31 de diciembre de 2009 y vuelve a negociar el 4 de enero de 2010.
    # Antes se excluía y anulaba la evaluación de fold-004 para todos los predictores.
    symbols = ["DVN", "XOM", "IBM"]
    # DVN va primero en orden alfabético y tiene la mayor puntuación, así que se mantiene.
    values = predictions(
        "US", symbols, start="2009-01-01", end="2009-12-31", score=lambda k, i: 0.01 * (3 - i)
    )
    tape, report = build_reconstructed_tape(
        Path(EDITION),
        [evaluation_window("US", values, index=4)],
        [values],
        market="US",
        partition="validation",
        dividend_payment_lag_sessions=0,
        symbols=symbols,
    )
    assert tape.identity["audit"]["walk_forward"][0]["fold"] == "fold-004"
    assert report["excluded"] == {} and "US/DVN" in tape.assets
    assert report["last_session"] == "2009-12-31"
    assert report["counts"]["final_sessions_without_row"] == 1
    dvn = tape.assets.index("US/DVN")
    assert np.isnan(tape.prices[-1, dvn, [0, 4]]).all()
    assert tape.prices[-1, dvn, 3] == tape.prices[-2, dvn, 3]
    for policy in ("hold_initial", "rebalance_50"):
        results = [
            evaluate(FinancialEnv(tape, capital=1_000_000, backend=backend), fixed_policy(policy))
            for backend in ("python", "native")
        ]
        assert results[0]["financial_validation"]["completed"] is True
        assert results[0]["financial_validation"] == results[1]["financial_validation"]
        assert results[0]["equity"]["nav"] == results[1]["equity"]["nav"]
        assert results[0]["ending_positions"].get("US/DVN", 0) > 0
        print(policy, results[0]["financial_validation"])
