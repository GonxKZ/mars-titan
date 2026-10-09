"""Las cintas y carteras admiten la población US medida sin omitir activos."""

import numpy as np
import pytest

from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.market import MAX_TAPE_CELLS, MarketTape
from mars_titan.simulation.portfolio import MAX_INSTRUMENTS, Instrument, Portfolio
from mars_titan.simulation.storage import read_tape, write_tape

DAY = 86_400_000_000
START = 946_684_800_000_000 + 16 * 3_600_000_000
# 4.202 activos US distintos entre 2000 y 2023 y 4.200 en la sesión más poblada.
UNIVERSE = 4202
SESSIONS_2023 = 251


def tape(assets, sessions, *, broadcast=True):
    row = np.array([100.0, 100.0, 100.0, 100.0, 1e6])
    prices = np.broadcast_to(row, (sessions, assets, 5))
    scores = np.broadcast_to(np.linspace(-0.01, 0.01, assets), (sessions, assets))
    if not broadcast:
        prices, scores = np.array(prices), np.array(scores)
    return MarketTape(
        prices,
        START + np.arange(sessions, dtype=np.int64) * DAY,
        [f"US/A{i:05d}" for i in range(assets)],
        scores,
        domain="synthetic",
        currency="USD",
    )


def test_a_year_of_the_measured_universe_fits_one_tape_within_its_memory_budget():
    market = tape(UNIVERSE, SESSIONS_2023)
    cells = UNIVERSE * SESSIONS_2023
    assert 1_048_576 < cells <= MAX_TAPE_CELLS
    assert market.prices.nbytes + market.scores.nbytes == 48 * cells
    assert 48 * MAX_TAPE_CELLS == 96 * 1024**2


@pytest.mark.parametrize(
    "assets,sessions", [(MAX_INSTRUMENTS + 1, 2), (MAX_INSTRUMENTS, MAX_TAPE_CELLS // 8192 + 1)]
)
def test_tapes_above_the_contract_fail_before_copying(assets, sessions):
    with pytest.raises(ValueError, match="contrato"):
        tape(assets, sessions)


def test_python_environment_observes_and_trades_the_whole_universe():
    env = FinancialEnv(tape(UNIVERSE, 3), capital=1e9, participation=1)
    observation, _ = env.reset(seed=0)
    assert observation.shape == (6 * UNIVERSE + 2,)
    assert np.count_nonzero(observation[5 : 6 * UNIVERSE : 6]) == UNIVERSE
    _, reward, _, _, info = env.step(5)
    # El cuartil superior de las predicciones positivas recibe órdenes completas.
    positive = sum(1 for value in np.linspace(-0.01, 0.01, UNIVERSE) if value > 0)
    assert len(info["trades"]) == -(-positive // 4) and np.isfinite(reward)
    assert Portfolio({f"A{i}": Instrument("USD") for i in range(MAX_INSTRUMENTS)}, {"USD": 1})


def test_native_backend_rejects_the_universe_explicitly():
    with pytest.raises(ValueError, match="4096"):
        FinancialEnv(tape(UNIVERSE, 3), backend="native")


def test_storage_round_trip_keeps_every_asset(tmp_path):
    market = tape(UNIVERSE, 3, broadcast=False)
    write_tape(market, tmp_path / "tape")
    restored = read_tape(tmp_path / "tape")
    assert restored.assets == market.assets and restored.sha256 == market.sha256
