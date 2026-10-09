"""Prueba de humo con la edición real, solo lectura y para ejecución local explícita.

Se activa declarando ``MARS_TITAN_UNADJUSTED_EDITION`` con la ruta de la edición. Construye
cintas de 2023 con pocos activos y puntuaciones sintéticas constantes, que no proceden de
ningún modelo, y recorre los entornos con acciones fijas. No aprende ni evalúa políticas.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.market_rules import china_a_share_instrument
from mars_titan.simulation.reconstructed_tape import build_reconstructed_tape
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
