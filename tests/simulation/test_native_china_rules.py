"""Reglas de acciones A en el motor nativo, con paridad exacta frente a la referencia Python.

Las cintas son sintéticas y se identifican como tales. Una se construye con el formato real de
la edición reconstruida (aperturas en los límites con el residuo de la reconstrucción, fuera de
rejilla, suspensiones, filas ausentes y eventos) y otra recorre los cambios de banda y de timbre
de 2008 a 2023. Las acciones son fijas o pseudoaleatorias. No hay aprendizaje.
"""

import json
import math
import subprocess
from collections import Counter

import numpy as np
import pytest

from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.evaluation import evaluate, fixed_policy
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.market_rules import (
    beijing_day,
    china_a_share_instrument,
    tape_instruments,
)
from mars_titan.simulation.native_runtime import RULES_CONTRACT, load_library
from mars_titan.simulation.portfolio import CorporateAction, Instrument, Period
from mars_titan.simulation.reconstructed_tape import build_reconstructed_tape
from mars_titan.simulation.storage import instruments_from_manifest, read_tape, write_tape
from tests.simulation.native_library import requires_native_library
from tests.simulation.native_library import simulator_path as simulator
from tests.simulation.unadjusted_edition_fixture import (
    Asset,
    evaluation_window,
    listing_status,
    predictions,
    write_edition,
)

HOUR = 3_600_000_000
CAPITAL = 1_000_000
ASSETS = ["CN/000001.SZ", "CN/300750.SZ", "CN/600000.SS", "CN/688981.SS"]
# Sesiones a ambos lados de cada cambio de timbre y de banda comprobado en fuentes primarias.
DAYS = [
    (2008, 4, 22),
    (2008, 4, 23),
    (2008, 4, 24),
    (2008, 4, 25),
    (2008, 9, 17),
    (2008, 9, 18),
    (2008, 9, 19),
    (2008, 9, 22),
    (2019, 7, 18),
    (2019, 7, 19),
    (2019, 7, 22),
    (2019, 7, 23),
    (2020, 8, 19),
    (2020, 8, 20),
    (2020, 8, 21),
    (2020, 8, 24),
    (2020, 8, 25),
    (2020, 8, 26),
    (2023, 8, 23),
    (2023, 8, 24),
    (2023, 8, 25),
    (2023, 8, 28),
    (2023, 8, 29),
    (2023, 8, 30),
]
OPENINGS = (
    "limit_up",
    "limit_down",
    "inside_up",
    "inside_down",
    "gap_up",
    "gap_down",
    "ordinary",
    "ordinary",
    "ordinary",
    "suspended",
    "missing_open",
    "off_grid",
)


def opening(day):
    return beijing_day(*day) + 9 * HOUR + HOUR // 2


def closing(day):
    return beijing_day(*day) + 15 * HOUR


def rules(assets):
    return {asset: china_a_share_instrument(asset) for asset in assets}


def cents(value):
    return round(value * 100) / 100


def dated_tape(seed, *, partition="train"):
    """Cinta CNY sintética con aperturas en los límites vigentes de cada fecha y tablero."""
    rng = np.random.default_rng(seed)
    instruments = rules(ASSETS)
    count = len(ASSETS)
    prices = np.full((len(DAYS), count, 5), np.nan)
    previous = np.array([12.34, 48.6, 10.0, 25.05])
    actions = []
    for t, day in enumerate(DAYS):
        at = opening(day)
        for i, asset in enumerate(ASSETS):
            close_before = previous[i]
            if t == 0:
                prices[t, i] = (close_before, close_before, close_before, close_before, 2e7)
                continue
            limits = instruments[asset].limits(float(close_before), at)
            upper, lower = limits or (cents(close_before * 1.3), cents(close_before * 0.7))
            kind = OPENINGS[rng.integers(len(OPENINGS))]
            volume = float(rng.integers(5, 50) * 1e6)
            if kind == "suspended":
                prices[t, i] = (np.nan, np.nan, np.nan, close_before, 0.0)
                continue
            opened = {
                "limit_up": upper,
                "limit_down": lower,
                "inside_up": upper - 0.01,
                "inside_down": lower + 0.01,
                "gap_up": cents(upper * 1.02),
                "gap_down": cents(lower * 0.98),
                "ordinary": cents(close_before * (1 + rng.uniform(-0.04, 0.04))),
                "missing_open": np.nan,
                "off_grid": close_before * (1 + rng.uniform(-0.03, 0.03)) + 0.003,
            }[kind]
            reference = close_before if np.isnan(opened) else opened
            close = cents(min(max(reference * (1 + rng.uniform(-0.03, 0.03)), lower), upper))
            prices[t, i] = (opened, max(reference, close), min(reference, close), close, volume)
            previous[i] = close
    # Dividendo con la apertura en el límite ajustado, split con resto impar y split de STAR.
    dividend_day, split_day, star_day = 5, 9, 15
    actions.append(
        CorporateAction(
            "600000-dividend",
            "CN/600000.SS",
            "dividend",
            opening(DAYS[dividend_day]),
            0.3,
            pay_at=opening(DAYS[dividend_day + 2]),
            verified=True,
        )
    )
    reference = prices[dividend_day - 1, 2, 3] - 0.3
    adjusted = instruments["CN/600000.SS"].limits(float(reference), opening(DAYS[dividend_day]))
    prices[dividend_day, 2, :4] = (adjusted[1], adjusted[1], adjusted[1], adjusted[1])
    prices[dividend_day, 2, 4] = 3e7
    for name, asset, day, ratio in (
        ("000001-split", "CN/000001.SZ", split_day, 1.25),
        ("688981-split", "CN/688981.SS", star_day, 1.5),
    ):
        actions.append(
            CorporateAction(name, asset, "split", opening(DAYS[day]), ratio, verified=True)
        )
    scores = rng.normal(0.004, 0.01, (len(DAYS), count))
    tape = MarketTape(
        prices,
        [closing(day) for day in DAYS],
        ASSETS,
        scores,
        domain="synthetic",
        currency="CNY",
        partition=partition,
        open_times=[opening(day) for day in DAYS],
        actions=sorted(actions, key=lambda a: (a.effective_at, a.asset, a.kind != "dividend")),
        parent_id="synthetic_china_rules_fixture",
    )
    return tape, instruments


def plans(seed, steps):
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 6, steps),
        np.resize([5, 1], steps),
        np.resize([5, 0, 0, 2, 1, 4], steps),
    ]


def assert_parity(tape, instruments, actions, *, capital=CAPITAL, seed=3):
    """Paso a paso: observación, recompensa, final, info y estado completo de la cartera."""
    reference = FinancialEnv(tape, capital=capital, instruments=instruments)
    native = FinancialEnv(tape, capital=capital, instruments=instruments, backend="native")
    np.testing.assert_array_equal(reference.reset(seed=seed)[0], native.reset(seed=seed)[0])
    seen = Counter()
    for action in actions:
        if reference.done:
            break
        expected, actual = reference.step(int(action)), native.step(int(action))
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]
        assert native.book.snapshot()["state"] == reference.book.snapshot()["state"]
        info = expected[4]
        seen.update(entry["reason"] for entry in info["unfilled"])
        for trade in info["trades"]:
            seen["buy" if trade["quantity"] > 0 else "sell"] += 1
            lot = instruments[trade["asset"]].lot
            if trade["quantity"] < 0 and -trade["quantity"] % lot:
                seen["odd_lot_sale"] += 1
    assert reference.done == native.done
    return seen, native


@requires_native_library
@pytest.mark.parametrize("seed", [11, 12, 13])
def test_dated_bands_and_stamp_duty_match_python_step_by_step(seed):
    tape, instruments = dated_tape(seed)
    seen = Counter()
    for actions in plans(seed, len(tape) - 1):
        result, native = assert_parity(tape, instruments, actions)
        seen += result
        assert native.book.execution_counts["native_steps"] > 0
    # La paridad no es vacía: hay compras, ventas y órdenes bloqueadas en ambos límites.
    assert seen["buy"] and seen["sell"]
    assert seen["limit_up"] + seen["limit_down"] > 0


@requires_native_library
def test_all_rule_paths_are_exercised_across_the_dated_fixtures():
    seen = Counter()
    for seed in range(11, 21):
        tape, instruments = dated_tape(seed)
        for actions in plans(seed, len(tape) - 1):
            seen += assert_parity(tape, instruments, actions)[0]
    for reason in ("limit_up", "limit_down", "missing_open", "cash_liquidity_or_lot_limit"):
        assert seen[reason] > 0, reason
    assert seen["odd_lot_sale"] > 0


@requires_native_library
def test_chinext_band_widens_in_the_native_engine_on_its_reform_date():
    asset = "CN/300750.SZ"
    days = [(2020, 8, 20), (2020, 8, 21), (2020, 8, 24)]
    prices = np.full((3, 1, 5), np.nan)
    prices[0, 0] = (50.0, 50.0, 50.0, 50.0, 1e8)
    prices[1, 0] = (55.0, 55.0, 55.0, 50.0, 1e8)
    prices[2, 0] = (55.0, 55.0, 55.0, 55.0, 1e8)
    tape = MarketTape(
        prices,
        [closing(day) for day in days],
        [asset],
        np.full((3, 1), 0.01),
        domain="synthetic",
        currency="CNY",
        open_times=[opening(day) for day in days],
    )
    instruments = rules([asset])
    env = FinancialEnv(tape, capital=CAPITAL, instruments=instruments, backend="native")
    env.reset(seed=0)
    blocked = env.step(5)[4]
    assert blocked["trades"] == [] and blocked["unfilled"] == [dict(asset=asset, reason="limit_up")]
    # El 24 de agosto la banda es del 20 % y la misma apertura del 10 % ya se ejecuta.
    filled = env.step(0)[4]
    assert [t["quantity"] % 100 for t in filled["trades"]] == [0]
    assert env.book.execution_counts["native_steps"] == 2


EDITION_ASSETS = [
    # Aperturas en ambos límites con el residuo de la reconstrucción, suspensión y huecos.
    Asset(
        "600000.SS",
        base=10.0,
        zero_volume=(60, 61),
        missing=(80,),
        off_grid_open=(90,),
        overrides={
            39: dict(close=10.00),
            40: dict(open=11.00 * (1 - 5e-7) / (1 + 4e-7), close=11.00),
            44: dict(close=11.00),
            45: dict(open=9.90 * (1 + 5e-7) / (1 + 4e-7), close=9.95),
        },
    ),
    Asset("300750.SZ", base=50.0, overrides={120: dict(close=50.0), 121: dict(open=60.0)}),
    # Split con resto impar, dividendo y evento ambiguo en la misma sesión.
    Asset("000001.SZ", base=12.0, events=((100, 0.046154, 1.25), (150, 0.2, 0))),
    Asset("688981.SS", base=40.0, events=((170, 0.1, 1.5),)),
    # Apertura en el límite del 5 % dentro del tramo ST y salto del 17,6 % en un día sin límite.
    Asset(
        "600519.SS",
        base=1700.0,
        overrides={
            46: dict(close=1700.00),
            47: dict(open=1785.00 * (1 - 5e-7) / (1 + 4e-7), close=1785.00),
            117: dict(close=1700.00),
            118: dict(open=2000.00, close=2000.00),
        },
    ),
]


@pytest.fixture(scope="module")
def edition(tmp_path_factory):
    root = tmp_path_factory.mktemp("edition")
    write_edition(root, {"CN": EDITION_ASSETS})
    return root


# Estado de cotización de la fixture. Los tramos ST y S de 600519.SS reducen su banda al 5 %
# en parte de 2023 y dos días quedan sin límite, así que la paridad recorre esas bandas.
STATUS = {
    "CN/600519.SS": dict(
        special_treatment=[["2023-02-01", "2023-05-04"]],
        share_reform_pending=[["2023-06-01", "2023-07-03"]],
        limit_free_days=["2023-07-03", "2023-09-12"],
    ),
}


def reconstructed(root, *, score=None, lag=2, status=None):
    symbols = [asset.symbol for asset in EDITION_ASSETS]
    values = predictions("CN", symbols, score=score)
    tape, _ = build_reconstructed_tape(
        root,
        [evaluation_window("CN", values)],
        [values],
        market="CN",
        partition="validation",
        dividend_payment_lag_sessions=lag,
        listing_status=listing_status(root, china=STATUS if status is None else status),
    )
    return tape, tape_instruments(tape)


@requires_native_library
@pytest.mark.parametrize("plan", range(3))
def test_reconstructed_chinese_tape_matches_python_step_by_step(edition, plan):
    tape, instruments = reconstructed(edition)
    actions = plans(plan, len(tape) - 1)[plan]
    seen, native = assert_parity(tape, instruments, actions)
    counts = native.book.execution_counts
    assert counts["native_steps"] > counts["reference_event_steps"] > 0
    assert seen["buy"] and seen["sell"]


@requires_native_library
def test_reconstructed_limit_opens_block_in_the_native_engine(edition):
    # Solo 600000.SS tiene predicción positiva, como en la prueba de la referencia Python.
    tape, instruments = reconstructed(edition, score=lambda k, i: 0.05 if i == 2 else -0.01)
    plan = {39: 5, 41: 5, 42: 0, 43: 0, 44: 1}
    actions = [plan.get(k, 1) for k in range(46)]
    seen, native = assert_parity(tape, instruments, actions)
    assert seen["limit_up"] >= 1 and seen["limit_down"] >= 1
    assert native.book.positions["CN/600000.SS"] % 100 == 0


@requires_native_library
def test_listing_status_bands_block_and_free_orders_in_both_engines(edition):
    # Solo 600519.SS tiene predicción positiva. Su apertura del 2023-03-16 queda en el límite
    # del 5 % del tramo ST y la del 2023-07-03, sin límite, sube un 17,6 % y se ejecuta.
    tape, instruments = reconstructed(edition, score=lambda k, i: 0.05 if i == 3 else -0.01)
    limits = instruments["CN/600519.SS"].price_limits
    assert {period.band for period in limits} == {0.05, 0.10}
    plan = {46: 5, 47: 1, 117: 5}
    actions = [plan.get(k, 1) for k in range(118)]
    seen, native = assert_parity(tape, instruments, actions)
    assert seen["limit_up"] >= 1 and seen["buy"] >= 1
    assert native.book.positions["CN/600519.SS"] > 0
    # Con las reglas del tablero, sin estado, la primera apertura sí se ejecutaría.
    board = rules(tape.assets)
    upper, _ = board["CN/600519.SS"].limits(1700.0 * (1 + 4e-7), tape.open_times[47])
    assert upper == 1870.0


@requires_native_library
def test_identity_declares_the_native_rule_contract_only_with_rules():
    tape, instruments = dated_tape(11)
    native = FinancialEnv(tape, instruments=instruments, backend="native")
    python = FinancialEnv(tape, instruments=instruments)
    assert native.identity["native_market_rules"] == RULES_CONTRACT
    assert "native_market_rules" not in python.identity
    assert native.identity["instruments_sha256"] == python.identity["instruments_sha256"]
    plain = MarketTape(
        tape.prices, tape.close_times, tape.assets, tape.scores, domain="synthetic", currency="CNY"
    )
    assert "native_market_rules" not in FinancialEnv(plain, backend="native").identity


@requires_native_library
def test_native_limits_reproduce_the_python_decimal_rounding():
    library = load_library()
    rng = np.random.default_rng(7)
    values = np.concatenate(
        [
            rng.integers(1, 1_000_000, 4000) / 100,
            (rng.integers(1, 1_000_000, 4000) + 0.5) / 100,
            rng.integers(1, 10_000_000, 4000) / 1000,
            rng.uniform(0.001, 5000, 4000),
            10 ** rng.uniform(-4, 7, 2000),
            [10.15, 10.05, 0.01, 0.004, 0.005, 1e-9, 12345678.905, 9.995, 2.675],
        ]
    )
    at = beijing_day(2021, 3, 2)
    for band in (0.05, 0.1, 0.2, 0.15, 0.123, 0.0625):
        instrument = Instrument("CNY", price_limits=(Period(0, 2**62, band=band),), rules="x")
        for value in values:
            assert library.price_limits(float(value), band) == instrument.limits(float(value), at)
    assert library.price_limits(None, 0.1) is None
    assert library.price_limits(-1.0, 0.1) is None
    assert library.price_limits(10.0, 0.0) is None
    for band in (1.0, -0.1, 1e-11, float("nan")):
        with pytest.raises(ValueError):
            library.price_limits(10.0, band)


def test_schema_two_manifests_carry_and_restore_the_rules(tmp_path):
    tape, instruments = dated_tape(11)
    write_tape(tape, tmp_path / "ruled", instruments=instruments)
    write_tape(tape, tmp_path / "plain")
    ruled = json.loads((tmp_path / "ruled/manifest.json").read_text())
    plain = json.loads((tmp_path / "plain/manifest.json").read_text())
    assert ruled["schema_version"] == 2 and plain["schema_version"] == 1
    assert "instruments" not in plain
    assert instruments_from_manifest(ruled, tape.assets) == instruments
    assert read_tape(tmp_path / "ruled").sha256 == read_tape(tmp_path / "plain").sha256
    broken = dict(ruled, schema_version=1)
    with pytest.raises(ValueError, match="versión"):
        instruments_from_manifest(broken, tape.assets)
    changed = json.loads(json.dumps(ruled))
    changed["instruments"][ASSETS[0]]["lot"] = 0
    with pytest.raises(ValueError):
        instruments_from_manifest(changed, tape.assets)
    with pytest.raises(ValueError, match="moneda"):
        write_tape(tape, tmp_path / "other", instruments={a: Instrument("USD") for a in ASSETS})
    assert not (tmp_path / "other").exists()


def run_simulator(source, output, *arguments):
    return subprocess.run(
        [str(simulator()), "--input", str(source), "--output", str(output), *arguments],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize("policy", ["cash", "hold_initial", "rebalance_50"])
@pytest.mark.parametrize("seed", [11, 14, 17])
def test_cpp_session_applies_the_same_rules_as_python(tmp_path, policy, seed):
    # La sesión C++ que usan PPO y KLPO nativos aplica también los eventos de la apertura.
    tape, instruments = dated_tape(seed, partition="validation")
    write_tape(tape, tmp_path / "input", instruments=instruments)
    result = run_simulator(
        tmp_path / "input",
        tmp_path / "output",
        "--policy",
        policy,
        "--capital",
        str(CAPITAL),
        "--diagnostic",
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "output/run.json").read_text())
    expected = evaluate(
        FinancialEnv(tape, capital=CAPITAL, instruments=instruments), fixed_policy(policy)
    )
    assert report["financial_validation"] == expected["financial_validation"]
    identity = json.loads((tmp_path / "output/identity.json").read_text())
    assert identity["market_rules"] == RULES_CONTRACT
    if policy == "rebalance_50":
        # Sin reglas la misma cinta da otra contabilidad: la comparación no es trivial.
        plain = evaluate(FinancialEnv(tape, capital=CAPITAL), fixed_policy(policy))
        assert plain["financial_validation"] != expected["financial_validation"]


def test_cpp_reader_rejects_rules_without_their_manifest_version(tmp_path):
    tape, instruments = dated_tape(11, partition="validation")
    write_tape(tape, tmp_path / "input", instruments=instruments)
    path = tmp_path / "input/manifest.json"
    manifest = json.loads(path.read_text())
    for change in (
        dict(schema_version=1),
        dict(instruments={**manifest["instruments"], ASSETS[0]: {"currency": "USD", "lot": 1}}),
        dict(instruments={k: v for k, v in manifest["instruments"].items() if k != ASSETS[0]}),
    ):
        path.write_text(json.dumps({**manifest, **change}))
        result = run_simulator(tmp_path / "input", tmp_path / "output", "--diagnostic")
        assert result.returncode == 1 and result.stderr.startswith("Error: "), result.stderr
        assert not (tmp_path / "output").exists()
    write_tape(tape, tmp_path / "plain")
    plain = run_simulator(tmp_path / "plain", tmp_path / "plain-output", "--diagnostic")
    assert plain.returncode == 0, plain.stderr
    identity = json.loads((tmp_path / "plain-output/identity.json").read_text())
    assert "market_rules" not in identity


@requires_native_library
def test_kernel_without_rules_is_identical_to_v1_and_rejects_invalid_rules():
    from mars_titan.simulation.native_portfolio import NativePortfolio
    from mars_titan.simulation.native_runtime import RULES

    tape, instruments = dated_tape(12)
    plain = NativePortfolio({a: Instrument("CNY", lot=100) for a in ASSETS}, {"CNY": CAPITAL})
    plain.start(int(tape.close_times[0]), tape.quotes(0))
    plain.submit({a: 1234.0 for a in ASSETS}, decision_at=plain.clock)
    library = plain.library
    frame = np.ascontiguousarray(tape.prices[1])
    outputs = []
    for rules_array in (None, np.zeros(len(ASSETS), dtype=RULES)):
        if rules_array is not None:
            rules_array["reference"] = np.nan
        library.step(plain, frame, int(tape.open_times[1]), int(tape.close_times[1]), rules_array)
        outputs.append(
            (
                plain._next_positions.tobytes(),
                plain._next_accounts.tobytes(),
                plain._trades.tobytes(),
            )
        )
    assert outputs[0] == outputs[1]
    for field, value in (
        ("band", 1.0),
        ("buy_tax", -0.1),
        ("sell_tax", 1.0),
        ("odd_lot_exit", 2),
        ("reserved", 1),
        ("reference", math.inf),
        ("minimum_order", -1.0),
    ):
        invalid = np.zeros(len(ASSETS), dtype=RULES)
        invalid["reference"] = np.nan
        invalid[field][0] = value
        with pytest.raises(ValueError, match="reglas"):
            library.step(plain, frame, int(tape.open_times[1]), int(tape.close_times[1]), invalid)
    # Una segunda ejecución antes del siguiente cierre violaría T+1 y el núcleo la rechaza.
    with pytest.raises(ValueError, match="tiempos"):
        library.step(plain, frame, plain.clock, int(tape.close_times[1]), None)
