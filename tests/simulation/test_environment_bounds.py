"""Cotas del entorno que ninguna política puede superar, sin aprender.

Las cintas son sintéticas e identificadas como tales. Las políticas siguen reglas fijas o
eligen al azar con semilla, y ninguna ajusta parámetros. La cota se comprueba con números
calculados a mano, con políticas que conocen el rendimiento de la sesión siguiente y con
cientos de episodios en las contabilidades Python y nativa.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from mars_titan.integrity.oracle_bound import oracle_tape
from mars_titan.simulation import environment_bounds as bounds
from mars_titan.simulation.campaign_stage import episode
from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.evaluation import evaluate, fixed_policy
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.portfolio import DELISTING, CorporateAction
from tests.simulation.native_library import NATIVE_BACKEND

OPEN, CLOSE, VOLUME = 0, 3, 4
TIMES = dict(close=100, open=50, step=100)


def tape(rows, *, actions=(), scores=None):
    """Cinta sintética con filas [sesión][activo] = (apertura, cierre, volumen)."""
    rows = np.asarray(rows, dtype=float)
    sessions, assets = rows.shape[:2]
    prices = np.full((sessions, assets, 5), np.nan)
    prices[:, :, OPEN] = rows[:, :, 0]
    prices[:, :, 1] = np.fmax(rows[:, :, 0], rows[:, :, 1])
    prices[:, :, 2] = np.fmin(rows[:, :, 0], rows[:, :, 1])
    prices[:, :, CLOSE] = rows[:, :, 1]
    prices[:, :, VOLUME] = rows[:, :, 2]
    return MarketTape(
        prices,
        [TIMES["close"] + TIMES["step"] * index for index in range(sessions)],
        [chr(ord("A") + index) for index in range(assets)],
        # Sin cierre no hay predicción, como exige la cinta tras una baja.
        np.where(np.isnan(rows[:, :, 1]), np.nan, 1.0) if scores is None else scores,
        domain="synthetic",
        currency="USD",
        partition="validation",
        open_times=[TIMES["open"] + TIMES["step"] * index for index in range(sessions)],
        actions=actions,
    )


def at(session):
    return TIMES["open"] + TIMES["step"] * session


def action(identifier, asset, kind, session, value, pay=None):
    return CorporateAction(
        identifier,
        asset,
        kind,
        at(session),
        value,
        pay_at=None if pay is None else at(pay),
        verified=True,
    )


# A gana con el salto nocturno y B paga un dividendo y después hace un split 2 por 1.
ROWS = [
    [(10, 10, 1000), (20, 20, 2000)],
    [(11, 12, 1000), (19, 21, 2000)],
    [(12, 11, 1000), (10.5, 11, 2000)],
]
ACTIONS = (action("div", "B", "dividend", 1, 1.0, pay=2), action("split", "B", "split", 2, 2.0))


def test_the_bound_takes_the_best_overnight_gap_then_the_best_intraday_return():
    growth = bounds.session_growth(tape(ROWS, actions=ACTIONS))
    # Sesión 1: mejor salto 11/10 (A) y mejor intradía 21/19 (B). Mantener A da 12/10, menos
    # que vender A en la apertura y comprar B. Sesión 2: el split deja el salto de B en uno.
    assert growth == pytest.approx([1.1 * 21 / 19, 11 / 10.5], rel=1e-15)


def test_the_capacity_follows_the_previous_volume_and_the_split_at_the_open():
    capacity = bounds.session_capacity(tape(ROWS, actions=ACTIONS), participation=0.01)
    # 1 % del volumen del cierre anterior, escalado por el split, al precio de apertura.
    assert capacity == pytest.approx([10 * 11 + 20 * 19, 10 * 12 + 2 * 20 * 10.5])
    limits = bounds.tape_bounds(tape(ROWS, actions=ACTIONS), participation=0.01, capital=1000)
    assert limits["turnover"] == pytest.approx([0.49, 1.03])


def test_retirements_value_the_position_with_their_payment_or_with_nothing():
    rows = [
        [(10, 10, 1000), (4, 4, 1000)],
        [(10, 10, 1000), (math.nan, math.nan, math.nan)],
    ]
    delisted = tape(rows, actions=(action("out", "B", DELISTING, 1, 5.0, pay=1),))
    assert bounds.session_growth(delisted) == pytest.approx([1.25])
    assert bounds.session_capacity(delisted, participation=0.01) == pytest.approx([100.0])
    lost = tape(rows, actions=(action("loss", "B", "writeoff", 1, 0),))
    assert bounds.session_growth(lost) == pytest.approx([1.0])


def test_an_unknown_corporate_action_stops_the_bound():
    known = tape(ROWS)

    class Unknown:
        prices, close_times, open_times = known.prices, known.close_times, known.open_times
        assets = known.assets
        actions = [SimpleNamespace(kind="rights_issue", asset="A", effective_at=at(1), value=1)]

        def __len__(self):
            return len(self.close_times)

    with pytest.raises(ValueError, match="rights_issue"):
        bounds.session_growth(Unknown())


def world(seed, *, assets=5, sessions=60):
    """Cinta aleatoria con saltos, dividendos, splits, una baja con cobro y otra sin valor."""
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(0, 0.03, (sessions, assets)), axis=0))
    gap = np.exp(rng.normal(0, 0.02, (sessions, assets)))
    open_ = np.vstack([close[:1], close[:-1] * gap[1:]])
    volume = rng.uniform(2e3, 5e4, (sessions, assets))
    actions = [
        action("div-a", "A", "dividend", 10, 0.4, pay=12),
        action("split-b", "B", "split", 20, 2.0),
        action("div-c", "C", "dividend", 30, 0.3, pay=30),
        action("out-d", "D", DELISTING, 40, float(close[39, 3] * 1.3), pay=45),
        action("loss-e", "E", "writeoff", 50, 0),
    ]
    open_[20:, 1] /= 2
    close[20:, 1] /= 2
    for column, start in ((3, 40), (4, 50)):
        open_[start:, column] = close[start:, column] = volume[start:, column] = np.nan
    rows = np.stack([open_, close, volume], axis=2)
    scores = rng.normal(size=(sessions, assets))
    scores[np.isnan(close)] = np.nan
    return tape(rows, actions=tuple(actions), scores=scores)


def random_policy(seed):
    rng = np.random.default_rng(seed)
    return lambda _observation, _step: int(rng.integers(0, 6))


POLICIES = {
    "cash": lambda seed: fixed_policy("cash"),
    "full": lambda seed: lambda _observation, _step: 5,
    "alternate": lambda seed: lambda _observation, step: 5 * (step % 2),
    "random": random_policy,
}
BACKENDS = ("python", NATIVE_BACKEND)


def episodes(source, backend, *, cost_bps, capital=1e6):
    for name, factory in POLICIES.items():
        for seed in range(3 if name == "random" else 1):
            env = FinancialEnv(source, backend=backend, capital=capital, cost_bps=cost_bps)
            yield name, episode(cost_bps, evaluate(env, factory(seed), seed=seed))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed", range(4))
def test_no_policy_exceeds_the_bounds_even_knowing_the_next_session(backend, seed):
    source = world(seed)
    limits = bounds.tape_bounds(source, participation=0.01, capital=1e6)
    # La política con información futura conoce el rendimiento de apertura a cierre de la
    # sesión siguiente. La cinta sin costes y con capital pequeño hace la cota más exigente.
    candidates = [source, oracle_tape(source)]
    checked = 0
    for candidate in candidates:
        for cost in (0.0, 10.0):
            for capital in (1e3, 1e6):
                limits = bounds.tape_bounds(candidate, participation=0.01, capital=capital)
                for name, record in episodes(candidate, backend, cost_bps=cost, capital=capital):
                    if record["status"] == "failed":
                        continue
                    assert bounds.episode_defect(record, limits) is None, name
                    checked += 1
    assert checked >= 40


def test_the_bounds_are_reached_when_nothing_else_limits_the_policy():
    # Sin salto nocturno, una subida del 10 % en la sesión y liquidez de sobra: invertirlo
    # todo en la apertura alcanza la cota de crecimiento salvo el redondeo.
    rows = [[(10, 10, 1e9)], [(10, 11, 1e9)], [(11, 11, 1e9)]]
    source = tape(rows)
    env = FinancialEnv(source, capital=1000, cost_bps=0)
    record = episode(0.0, evaluate(env, lambda _observation, _step: 5))
    limits = bounds.tape_bounds(source, participation=0.01, capital=1000)
    nav = record["equity"]["nav"]
    assert nav[1] / nav[0] == pytest.approx(limits["growth"][0], rel=1e-12)
    # Con poco volumen, la misma política agota la participación de cada apertura.
    scarce = tape([[(10, 10, 300)], [(10, 11, 300)], [(11, 11, 300)]])
    env = FinancialEnv(scarce, capital=1000, cost_bps=0)
    record = episode(0.0, evaluate(env, lambda _observation, _step: 5))
    limits = bounds.tape_bounds(scarce, participation=0.01, capital=1000)
    assert record["turnover"] == pytest.approx(limits["turnover"][-1], rel=1e-12)


def honest(source):
    env = FinancialEnv(source, capital=1e6, cost_bps=10)
    return episode(10.0, evaluate(env, lambda _observation, _step: 5))


def test_a_jump_beyond_the_bound_or_an_impossible_rotation_is_a_defect():
    source = world(0)
    limits = bounds.tape_bounds(source, participation=0.01, capital=1e6)
    record = honest(source)
    assert bounds.episode_defect(record, limits) is None
    nav = list(record["equity"]["nav"])
    nav[5:] = [value * limits["growth"][4] * 1.01 for value in nav[5:]]
    jumped = dict(record, equity=dict(record["equity"], nav=nav))
    assert "cota con información futura" in bounds.episode_defect(jumped, limits)
    assert "sesión 5" in bounds.episode_defect(jumped, limits)
    busy = dict(record, turnover=limits["turnover"][record["steps"] - 1] * 1.01)
    assert "capacidad de la participación" in bounds.episode_defect(busy, limits)
    shifted = dict(
        record, equity=dict(record["equity"], close_times=record["equity"]["close_times"][1:])
    )
    assert "cierres de su cinta" in bounds.episode_defect(shifted, limits)


def test_bounds_reject_an_invalid_participation_or_capital():
    for participation, capital in ((0, 1e6), (1.5, 1e6), (0.01, 0), (0.01, math.inf)):
        with pytest.raises(ValueError, match="participación y un capital"):
            bounds.tape_bounds(world(0), participation=participation, capital=capital)
