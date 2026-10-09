"""Cinta real desde la edición reconstruida: filas verificadas, suspensiones y cortes."""

import copy
import dataclasses
import hashlib
import json
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.market import RECONSTRUCTED_CONTRACT, MarketTape
from mars_titan.simulation.market_rules import china_a_share_instrument
from mars_titan.simulation.reconstructed_tape import build_reconstructed_tape, read_edition
from tests.environments.walk_forward_fixture import microseconds
from tests.simulation.native_library import requires_native_library
from tests.simulation.unadjusted_edition_fixture import (
    Asset,
    evaluation_window,
    predictions,
    tape_days,
    write_edition,
)

US = [
    # Dividendo, split posterior y un evento en la primera sesión, que no se aplica.
    Asset("AAA", events=((0, 0.10, 0), (60, 0.25, 0), (120, 0, 2.0))),
    # Suspensión con cierre movido por el proveedor y dos sesiones sin fila.
    Asset("BBB", zero_volume=(30, 31, 32), missing=(50, 51)),
    Asset("CCC", verified_from=100),
    Asset("DDD", start=80),
    Asset("EEE", end=-5),
    Asset("FFF", never_verified=True),
    # Split y dividendo en la misma sesión: el importe por acción es ambiguo.
    Asset("GGG", base=30.0, events=((90, 0.40, 1.5),)),
    Asset("HHH", off_grid_open=(70,)),
    # La primera sesión no negocia y el último cierre negociado verificado es de 2022.
    Asset("III", leading_zero_volume=True),
    # Igual, pero las filas de 2022 no están verificadas.
    Asset("JJJ", verified_from=0, leading_zero_volume=True),
]
CN = [
    # Apertura en el límite superior y en el inferior con el residuo de la reconstrucción.
    Asset(
        "600000.SS",
        base=10.0,
        zero_volume=(60, 61),
        overrides={
            39: dict(close=10.00),
            40: dict(open=11.00 * (1 - 5e-7) / (1 + 4e-7), close=11.00),
            44: dict(close=11.00),
            45: dict(open=9.90 * (1 + 5e-7) / (1 + 4e-7), close=9.95),
        },
    ),
    Asset("300750.SZ", base=50.0),
    Asset("000001.SZ", base=12.0, events=((100, 0.046154, 1.3),)),
    Asset("600010.SS", unverified=(20, 21, 22)),
]
EXPECTED_EXCLUSIONS = {
    "US/CCC": "unverified_rows_in_tape",
    "US/DDD": "no_verified_traded_close_at_start",
    "US/EEE": "missing_last_session",
    "US/FFF": "no_verified_rows",
    "US/JJJ": "no_verified_traded_close_at_start",
}


@pytest.fixture(scope="module")
def edition(tmp_path_factory):
    root = tmp_path_factory.mktemp("edition")
    write_edition(root, {"US": US, "CN": CN})
    return root


def build(edition, market="US", *, values=None, lag=0, **options):
    symbols = [asset.symbol for asset in (US if market == "US" else CN)]
    values = predictions(market, symbols) if values is None else values
    windows = options.pop("windows", None) or [evaluation_window(market, values)]
    return build_reconstructed_tape(
        edition,
        windows,
        options.pop("predictions", [values]),
        market=market,
        partition=options.pop("partition", "train"),
        dividend_payment_lag_sessions=lag,
        **options,
    )


def column(tape, asset):
    return tape.assets.index(asset)


def test_only_verified_assets_enter_and_every_exclusion_has_a_reason(edition):
    tape, report = build(edition)
    assert report["excluded"] == EXPECTED_EXCLUSIONS
    assert tape.assets == ["US/AAA", "US/BBB", "US/GGG", "US/HHH", "US/III"]
    assert sum(report["exclusions"].values()) == len(EXPECTED_EXCLUSIONS)
    assert report["dropped_predictions"] == len(EXPECTED_EXCLUSIONS) * len(tape)
    cn, cn_report = build(edition, "CN")
    assert cn_report["excluded"] == {"CN/600010.SS": "unverified_rows_in_tape"}
    assert cn.currency == "CNY" and len(cn.assets) == 3


def test_identity_declares_the_reconstructed_treatment_and_its_limits(edition):
    tape, _ = build(edition, lag=3)
    audit = tape.identity["audit"]
    assert audit["price_basis"] == "unadjusted_reconstructed"
    assert {key: audit[key] for key in RECONSTRUCTED_CONTRACT} == RECONSTRUCTED_CONTRACT
    assert audit["corporate_actions_complete"] is False
    assert audit["exit_returns"] == "unavailable"
    assert audit["population"] == "listed_through_2025_03"
    assert audit["assumptions"] == {"dividend_payment_lag_sessions": 3}
    assert audit["edition_id"] == read_edition(edition)["edition_id"]
    assert tape.identity["source"]["exclusions"]["missing_last_session"] == 1


def test_sessions_and_fit_ends_come_from_the_receipt_and_never_reach_2024(edition):
    tape, _ = build(edition)
    days = tape_days("US")
    assert len(tape) == len(days) == 250
    window = evaluation_window("US", predictions("US", [a.symbol for a in US]))
    start, end = window.segment("evaluation")
    assert start <= tape.close_times[0] and tape.close_times[-1] < end == FINAL_TEST_START_US
    assert (tape.open_times < tape.close_times).all()
    assert (tape.open_times[1:] > tape.close_times[:-1]).all()
    assert set(tape.identity["audit"]["prediction_fit_ends"]) == {window.labels_used_until}
    segment = tape.identity["audit"]["walk_forward"][0]
    assert segment["receipt_sha256"] == window.sha256 and segment["fold"] == "fold-018"


def test_zero_volume_and_missing_rows_have_no_execution_and_no_new_price(edition):
    tape, report = build(edition)
    i = column(tape, "US/BBB")
    traded_close = tape.prices[29, i, 3]
    for k in (30, 31, 32):
        assert np.isnan(tape.prices[k, i, 0]) and tape.prices[k, i, 4] == 0
        # El proveedor movió el cierre 0,37 durante la suspensión. La cinta lo ignora.
        assert tape.prices[k, i, 3] == traded_close
    for k in (50, 51):
        assert np.isnan(tape.prices[k, i, 0]) and np.isnan(tape.prices[k, i, 4])
        assert tape.prices[k, i, 3] == tape.prices[49, i, 3]
    counts = report["counts"]
    assert counts["zero_volume_sessions"] == 4 and counts["zero_volume_moved_closes"] == 4
    assert counts["missing_rows"] == 2


def test_carry_at_the_start_uses_only_verified_traded_closes(edition):
    tape, _ = build(edition)
    i = column(tape, "US/III")
    assert np.isnan(tape.prices[0, i, 0]) and tape.prices[0, i, 4] == 0
    assert np.isfinite(tape.prices[:, i, 3]).all()
    assert np.isfinite(tape.prices[:, :, 3]).all()


def test_prices_on_the_quote_grid_are_exact_and_off_grid_opens_do_not_execute(edition):
    tape, report = build(edition)
    traded = np.isfinite(tape.prices[:, :, 0])
    for value in tape.prices[:, :, [0, 3]][np.isfinite(tape.prices[:, :, [0, 3]])]:
        assert value == round(value, 2) and len(repr(float(value)).split(".")[1]) <= 2
    h = column(tape, "US/HHH")
    assert np.isnan(tape.prices[70, h, 0]) and not traded[70, h]
    assert report["counts"]["off_grid_opens"] == 1


def test_corporate_actions_come_from_verified_events_with_declared_payment(edition):
    tape, _ = build(edition, lag=2)
    actions = [a for a in tape.actions if a.asset == "US/AAA"]
    assert [(a.kind, a.value) for a in actions] == [("dividend", 0.25), ("split", 2.0)]
    dividend, split = actions
    assert dividend.effective_at == tape.open_times[60] and dividend.pay_at == tape.open_times[62]
    assert split.effective_at == tape.open_times[120] and split.verified
    assert not [a for a in tape.actions if a.effective_at == tape.open_times[0]]
    assert {a.asset for a in tape.actions} <= set(tape.assets)
    late, _ = build(edition, lag=252)
    assert next(a for a in late.actions if a.kind == "dividend").pay_at == late.close_times[-1] + 1


def test_same_day_split_and_dividend_pays_the_lower_amount_without_execution(edition):
    tape, report = build(edition)
    i = column(tape, "US/GGG")
    actions = [a for a in tape.actions if a.asset == "US/GGG"]
    assert [(a.kind, a.value) for a in actions] == [("dividend", 0.40), ("split", 1.5)]
    assert np.isnan(tape.prices[90, i, 0])
    assert report["counts"]["ambiguous_same_day_events"] == 1
    cn, _ = build(edition, "CN")
    bonus = [a for a in cn.actions if a.asset == "CN/000001.SZ"]
    # 0,046154 por acción posterior equivale a 0,06 por acción anterior: se abona el menor.
    assert [(a.kind, a.value) for a in bonus] == [("dividend", 0.046154), ("split", 1.3)]


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda m: m.update(edition_id="0" * 64), "identidad"),
        (lambda m: m.update(price_basis="unadjusted"), "reconstruidos"),
        (lambda m: m.update(corporate_actions_complete=True), "reconstruidos"),
        (lambda m: m["identity"]["policy"].update(cutoff="2024-12-31"), "reconstruidos"),
    ],
)
def test_editions_that_change_their_declaration_are_rejected(edition, tmp_path, change, message):
    copied = tmp_path / "copy"
    write_edition(copied, {"US": US[:2]})
    manifest = json.loads((copied / "manifest.json").read_text())
    change(manifest)
    (copied / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        build(copied, symbols=["AAA"])


def test_altered_or_post_cutoff_artifacts_are_rejected(tmp_path):
    root = tmp_path / "altered"
    write_edition(root, {"US": US[:2]})
    path = root / "assets" / "US" / "AAA" / "prices.parquet"
    data = bytearray(path.read_bytes())
    data[100] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="ha cambiado"):
        build(root, symbols=["AAA"])
    # Una fila de 2024 con huellas e identidad recalculadas sigue sin admitirse.
    later = tmp_path / "later"
    write_edition(later, {"US": US[:1]})
    folder = later / "assets" / "US" / "AAA"
    table = pq.read_table(folder / "prices.parquet")
    sessions = table.column("session").to_pylist()
    sessions[-1] = "2024-01-02"
    pq.write_table(table.set_column(0, "session", pa.array(sessions)), folder / "prices.parquet")
    manifest = json.loads((later / "manifest.json").read_text())
    manifest["receipts"]["US/AAA"]["prices.parquet"] = hashlib.sha256(
        (folder / "prices.parquet").read_bytes()
    ).hexdigest()
    digest = hashlib.sha256(json.dumps(manifest["identity"], sort_keys=True).encode())
    for artifacts in manifest["receipts"].values():
        digest.update(json.dumps(artifacts, sort_keys=True).encode())
    manifest["edition_id"] = digest.hexdigest()
    (later / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="posteriores a su corte"):
        build(later, symbols=["AAA"])


def test_predictions_must_match_the_receipt_and_their_decisions(edition):
    symbols = [a.symbol for a in US]
    values = predictions("US", symbols)
    window = evaluation_window("US", values)
    altered = dict(values, score=values["score"].copy())
    altered["score"][0] += 1e-9
    with pytest.raises(ValueError, match="huella"):
        build(edition, windows=[window], predictions=[altered])
    shifted = dict(values, prediction_at=values["prediction_at"] + 1)
    with pytest.raises(ValueError, match="decisión de su tramo"):
        build(edition, values=shifted)
    foreign = dict(values, asset_id=[a.replace("US/", "CN/") for a in values["asset_id"]])
    with pytest.raises(ValueError, match="otro mercado"):
        build(edition, values=foreign)


def test_in_sample_segments_cannot_feed_a_tape(edition):
    symbols = [a.symbol for a in US]
    # La calibración usa etiquetas que el predictor ya consultó: sus predicciones no valen.
    calibration = predictions("US", symbols, start="2022-10-01", end="2022-12-31")
    window = evaluation_window("US", calibration, partition="calibration")
    with pytest.raises(ValueError, match="ajuste anterior"):
        build(edition, windows=[window], predictions=[calibration], segment="calibration")
    with pytest.raises(ValueError, match="etiqueta usada"):
        evaluation_window("US", calibration, until=microseconds("2023-01-01"))


def test_tape_contract_rejects_a_softened_reconstructed_declaration(edition):
    tape, _ = build(edition)
    options = dict(
        domain="real",
        currency="USD",
        partition="train",
        prediction_times=tape.prediction_times,
        open_times=tape.open_times,
    )

    def rebuild(audit=None, prices=None, currency="USD"):
        return MarketTape(
            tape.prices if prices is None else prices,
            tape.close_times,
            tape.assets,
            tape.scores,
            audit=audit or copy.deepcopy(tape.identity["audit"]),
            **dict(options, currency=currency),
        )

    # La declaración original se admite. Cada variante suavizada se rechaza.
    assert rebuild().identity["audit"] == tape.identity["audit"]
    for key, value in (
        ("corporate_actions_complete", True),
        ("exit_returns", "provider"),
        ("population", "complete"),
        ("rows", "all"),
        ("valuation", "provider_close"),
        ("market", "CN"),
        ("assumptions", {}),
    ):
        audit = copy.deepcopy(tape.identity["audit"])
        audit[key] = value
        with pytest.raises(ValueError, match="tratamiento"):
            rebuild(audit)
    audit = copy.deepcopy(tape.identity["audit"])
    del audit["exit_returns"]
    with pytest.raises(ValueError, match="tratamiento"):
        rebuild(audit)
    with pytest.raises(ValueError, match="tratamiento"):
        rebuild(currency="CNY")
    audit = copy.deepcopy(tape.identity["audit"])
    audit["prediction_fit_ends"] = [audit["prediction_fit_ends"][0] - 1] * len(tape)
    with pytest.raises(ValueError, match="ajuste anterior"):
        rebuild(audit)
    audit = copy.deepcopy(tape.identity["audit"])
    audit["walk_forward"][0]["start"] = int(tape.close_times[1])
    with pytest.raises(ValueError, match="fuera de sus tramos"):
        rebuild(audit)
    audit = copy.deepcopy(tape.identity["audit"])
    audit["walk_forward"][0]["end"] = FINAL_TEST_START_US + 1
    with pytest.raises(ValueError, match="tramos walk-forward"):
        rebuild(audit)
    prices = tape.prices.copy()
    prices[100, 0, 3] = np.nan
    with pytest.raises(ValueError, match="cierre valorado"):
        rebuild(prices=prices)


def test_future_rows_and_predictions_do_not_change_the_past(edition, tmp_path):
    cut = 150
    symbols = [a.symbol for a in US]
    changed = [
        dataclasses.replace(
            spec, overrides={k: dict(close=55.55, open=44.44) for k in range(cut, 240)}
        )
        if spec.symbol in {"AAA", "BBB"}
        else spec
        for spec in US
    ]
    root = tmp_path / "future"
    write_edition(root, {"US": changed})
    values = predictions("US", symbols)
    future = predictions(
        "US", symbols, score=lambda k, i: 0.05 if k >= cut else 0.01 * ((i + k) % 5 - 1)
    )
    base, _ = build(edition, values=values)
    other, _ = build(root, values=future)
    assert np.array_equal(base.prices[:cut], other.prices[:cut], equal_nan=True)
    assert np.array_equal(base.scores[:cut], other.scores[:cut], equal_nan=True)
    first = trajectory(FinancialEnv(base), cut - 1)
    second = trajectory(FinancialEnv(other), cut - 1)
    assert first == second


def trajectory(env, steps, policy=lambda k: (5, 1, 3)[k % 3]):
    observation, _ = env.reset(seed=0)
    result = [observation.tobytes()]
    for k in range(steps):
        observation, reward, terminated, truncated, info = env.step(policy(k))
        result.append((observation.tobytes(), reward, terminated, truncated, repr(info)))
    return result


def executions(env, policy, *, checks):
    """Recorrer la cinta y comprobar cada operación con la información de su sesión."""
    tape = env.tape
    env.reset(seed=0)
    step, trades = 0, 0
    while not env.done:
        decision = env.cursor
        _, reward, _, _, info = env.step(policy(step))
        assert math.isfinite(reward) and info["reward_valid"]
        for trade in info["trades"]:
            i = column(tape, trade["asset"])
            checks(tape, decision, decision + 1, i, trade)
            trades += 1
        step += 1
    return trades


def ordinary(tape, decision, execution, i, trade):
    # Solo se ejecuta a la apertura siguiente, con apertura negociada y volumen de decisión.
    assert trade["price"] == tape.prices[execution, i, 0]
    assert np.isfinite(tape.prices[execution, i, 0])
    assert tape.prices[decision, i, 4] > 0


POLICIES = {
    "maximum_exposure": lambda step: 5,
    "maximum_turnover": lambda step: 5 if step % 2 == 0 else 1,
    "random": lambda step: int(np.random.default_rng(step).integers(0, 6)),
}


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_scripted_policies_never_trade_in_suspensions_gaps_or_ambiguous_sessions(edition, name):
    tape, _ = build(edition)
    trades = executions(FinancialEnv(tape), POLICIES[name], checks=ordinary)
    assert trades > 0


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_china_rules_apply_lots_and_daily_limits_on_reconstructed_prices(edition, name):
    tape, _ = build(edition, "CN")
    rules = {asset: china_a_share_instrument(asset) for asset in tape.assets}

    def checks(tape, decision, execution, i, trade):
        ordinary(tape, decision, execution, i, trade)
        reference = tape.prices[decision, i, 3]
        limits = rules[tape.assets[i]].limits(float(reference), int(tape.open_times[execution]))
        if trade["quantity"] > 0:
            assert trade["quantity"] % 100 == 0 and trade["price"] < limits[0]
        else:
            assert trade["price"] > limits[1]

    trades = executions(FinancialEnv(tape, instruments=rules), POLICIES[name], checks=checks)
    assert trades > 0


def test_orders_at_reconstructed_daily_limit_opens_do_not_fill(edition):
    symbols = [a.symbol for a in CN]
    # Solo 600000.SS tiene predicción positiva: es el único candidato de la cartera.
    values = predictions("CN", symbols, score=lambda k, i: 0.05 if i == 2 else -0.01)
    tape, _ = build(edition, "CN", values=values)
    i = column(tape, "CN/600000.SS")
    rules = {asset: china_a_share_instrument(asset) for asset in tape.assets}
    env = FinancialEnv(tape, instruments=rules)
    env.reset(seed=0)
    plan = {39: 5, 41: 5, 42: 0, 43: 0, 44: 1}
    outcomes = {}
    while env.cursor < 45:
        decision = env.cursor
        _, _, _, _, info = env.step(plan.get(decision, 1))
        outcomes[decision] = info
    assert outcomes[39]["trades"] == []
    assert {"asset": "CN/600000.SS", "reason": "limit_up"} in outcomes[39]["unfilled"]
    bought = outcomes[41]["trades"]
    assert len(bought) == 1 and bought[0]["quantity"] % 100 == 0
    assert outcomes[44]["trades"] == []
    assert {"asset": "CN/600000.SS", "reason": "limit_down"} in outcomes[44]["unfilled"]
    assert env.book.positions["CN/600000.SS"] == bought[0]["quantity"]
    # La edición guarda 10,999995 y 9,900005. Solo en la rejilla coinciden con los límites.
    assert tape.prices[39, i, 3] == 10.0 and tape.prices[40, i, 0] == 11.0
    assert tape.prices[44, i, 3] == 11.0 and tape.prices[45, i, 0] == 9.9


def test_reconstructed_chinese_tapes_require_the_a_share_rules(edition):
    tape, _ = build(edition, "CN")
    with pytest.raises(ValueError, match="reglas de acciones A"):
        FinancialEnv(tape)
    plain = {asset: china_a_share_instrument(asset) for asset in tape.assets}
    plain[tape.assets[0]] = dataclasses.replace(plain[tape.assets[0]], lot=1)
    with pytest.raises(ValueError, match="reglas de acciones A"):
        FinancialEnv(tape, instruments=plain)
    rules = {asset: china_a_share_instrument(asset) for asset in tape.assets}
    assert FinancialEnv(tape, instruments=rules).identity["instruments_sha256"]


def test_dividend_capture_earns_nothing_beyond_the_price_drop(tmp_path):
    # Precio plano de 20,00 que cae en la fecha ex exactamente el dividendo de 0,25.
    flat = {k: dict(open=20.0, close=20.0) for k in range(60)}
    flat.update({k: dict(open=19.75, close=19.75) for k in range(60, 250)})
    root = tmp_path / "capture"
    write_edition(
        root,
        {
            "US": [
                Asset("AAA", events=((60, 0.25, 0),), overrides=flat),
                Asset("ZZZ", overrides=flat),
            ]
        },
    )
    values = predictions("US", ["AAA", "ZZZ"], score=lambda k, i: 0.01 if i == 0 else -0.01)
    for lag in (0, 5):
        tape, _ = build(root, values=values, lag=lag)
        env = FinancialEnv(tape)
        env.reset(seed=0)
        rewards, costs = [], 0.0
        while not env.done:
            # Compra antes de la fecha ex, mantiene durante ella y vende después.
            action = {58: 5, 59: 0}.get(env.cursor, 1)
            _, reward, _, _, info = env.step(action)
            rewards.append(reward)
            costs = info["costs"]["USD"]
        assert costs > 0
        assert math.isclose(sum(rewards), math.log1p(-costs / env.capital), abs_tol=1e-12)


def only(symbol, symbols):
    """Puntuación positiva solo para ``symbol``: la política solo puede comprar ese activo."""
    index = sorted(symbols).index(symbol)
    return lambda k, i: 0.05 if i == index else -0.01


def replay(env, plan, until):
    env.reset(seed=0)
    outcomes = {}
    while env.cursor < until:
        decision = env.cursor
        outcomes[decision] = env.step(plan.get(decision, 1))[4]
    return outcomes


def traded(outcomes, decision, asset):
    return [trade for trade in outcomes[decision]["trades"] if trade["asset"] == asset]


def test_orders_into_suspensions_and_missing_rows_never_fill(edition):
    symbols = [a.symbol for a in US]
    tape, _ = build(edition, values=predictions("US", symbols, score=only("BBB", symbols)))
    env = FinancialEnv(tape)
    # Intentos de compra para ejecutar en la suspensión (30-32) y en las filas ausentes (50-51).
    plan = {29: 5, 30: 5, 31: 5, 32: 5, 49: 5, 50: 5}
    outcomes = replay(env, plan, 52)
    for decision in (29, 30, 31, 49, 50):
        assert traded(outcomes, decision, "US/BBB") == []
        assert {"asset": "US/BBB", "reason": "missing_open"} in outcomes[decision]["unfilled"]
    # La decisión de una sesión sin volumen no tiene capacidad aunque la siguiente negocie.
    assert traded(outcomes, 32, "US/BBB") == []
    assert {"asset": "US/BBB", "reason": "cash_liquidity_or_lot_limit"} in outcomes[32]["unfilled"]
    assert "US/BBB" not in env.book.positions


def test_off_grid_opens_and_ambiguous_event_sessions_never_fill(edition):
    symbols = [a.symbol for a in US]
    for symbol, decision in (("HHH", 69), ("GGG", 89)):
        values = predictions("US", symbols, score=only(symbol, symbols))
        tape, _ = build(edition, values=values)
        outcomes = replay(FinancialEnv(tape), {decision: 5}, decision + 1)
        assert traded(outcomes, decision, f"US/{symbol}") == []
        assert {"asset": f"US/{symbol}", "reason": "missing_open"} in outcomes[decision]["unfilled"]


def test_holding_through_an_ambiguous_event_receives_the_lower_cash_amount(edition):
    symbols = [a.symbol for a in US]
    tape, _ = build(edition, values=predictions("US", symbols, score=only("GGG", symbols)))
    env = FinancialEnv(tape)
    outcomes = replay(env, {80: 5, **{k: 0 for k in range(81, 89)}}, 89)
    held = sum(t["quantity"] for d in outcomes for t in traded(outcomes, d, "US/GGG"))
    cash = env.book.cash["USD"]
    before = env.book.snapshot()["state"]
    assert held > 0 and before["positions"]["US/GGG"] == held
    env.step(0)
    after = env.book.snapshot()["state"]
    # 0,40 por acción anterior, nunca 0,40 * 1,5, y las acciones se multiplican por 1,5.
    assert math.isclose(after["cash"]["USD"] - cash, held * 0.40, rel_tol=1e-12)
    assert math.isclose(after["positions"]["US/GGG"], held * 1.5, rel_tol=1e-12)


@requires_native_library
def test_native_and_python_engines_agree_on_a_reconstructed_us_tape(edition):
    # Suspensiones, filas ausentes, aperturas sin ejecución, dividendos y splits reales.
    tape, _ = build(edition, lag=2)
    reference, native = FinancialEnv(tape), FinancialEnv(tape, backend="native")
    np.testing.assert_array_equal(reference.reset(seed=5)[0], native.reset(seed=5)[0])
    for action in np.random.default_rng(6).integers(0, 6, len(tape) - 1):
        expected, actual = reference.step(int(action)), native.step(int(action))
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == pytest.approx(expected[1], abs=1e-12)
        assert actual[2:] == expected[2:]
