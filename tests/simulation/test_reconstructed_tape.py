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
from mars_titan.simulation.market import (
    RECONSTRUCTED_CONTRACT,
    WALK_FORWARD_SEGMENT,
    MarketTape,
    censors_fit,
)
from mars_titan.simulation.market_rules import china_a_share_instrument, tape_instruments
from mars_titan.simulation.reconstructed_tape import (
    NoAdmittedAssets,
    build_reconstructed_tape,
    read_edition,
)
from mars_titan.simulation.window_tapes import SEGMENT
from tests.environments.walk_forward_fixture import microseconds
from tests.simulation.native_library import requires_native_library
from tests.simulation.policy_tape_fixture import monthly_window
from tests.simulation.unadjusted_edition_fixture import (
    Asset,
    evaluation_window,
    listing_status,
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
    status = options.pop("listing_status", None) or listing_status(edition)
    return build_reconstructed_tape(
        edition,
        windows,
        options.pop("predictions", [values]),
        market=market,
        partition=options.pop("partition", "train"),
        dividend_payment_lag_sessions=lag,
        listing_status=status,
        **options,
    )


def column(tape, asset):
    return tape.assets.index(asset)


def test_only_verified_assets_enter_and_every_exclusion_has_a_reason(edition):
    tape, report = build(edition)
    assert report["excluded"] == EXPECTED_EXCLUSIONS
    # EEE termina su serie cinco sesiones antes del final: entra y sale con una baja.
    assert tape.assets == ["US/AAA", "US/BBB", "US/EEE", "US/GGG", "US/HHH", "US/III"]
    assert sum(report["exclusions"].values()) == len(EXPECTED_EXCLUSIONS)
    assert report["dropped_predictions"] == len(EXPECTED_EXCLUSIONS) * len(tape)
    assert report["predictions_after_series_end"] == 4
    cn, cn_report = build(edition, "CN")
    assert cn_report["excluded"] == {"CN/600010.SS": "unverified_rows_in_tape"}
    assert cn.currency == "CNY" and len(cn.assets) == 3


def test_a_tape_without_admitted_assets_reports_every_exclusion(edition):
    with pytest.raises(NoAdmittedAssets) as raised:
        build(edition, symbols=["FFF", "JJJ"])
    assert raised.value.excluded == {
        "US/FFF": "no_verified_rows",
        "US/JJJ": "no_verified_traded_close_at_start",
    }


def test_identity_declares_the_reconstructed_treatment_and_its_limits(edition):
    tape, _ = build(edition, lag=3)
    audit = tape.identity["audit"]
    assert audit["price_basis"] == "unadjusted_reconstructed"
    assert {key: audit[key] for key in RECONSTRUCTED_CONTRACT} == RECONSTRUCTED_CONTRACT
    assert audit["corporate_actions_complete"] is False
    assert audit["exit_returns"] == "source_exit_price_or_masked_position"
    assert audit["series_end"] == "delisting_at_next_open"
    assert audit["population"] == "listed_through_2025_03"
    assert audit["assumptions"] == {"dividend_payment_lag_sessions": 3}
    assert audit["edition_id"] == read_edition(edition)["edition_id"]
    assert audit["outside_universe"] == [] and audit["listing_status"]["assets"] == {}
    assert audit["delistings"] == {"US/EEE": {"last_session": tape_days("US")[-5], "exit": None}}
    assert "series_ends_in_tape" not in tape.identity["source"]["exclusions"]
    assert tape.identity["source"]["counts"]["series_ends_in_tape"] == 1
    assert tape.identity["source"]["final_session"] == (
        "missing_row_valued_at_last_traded_close_until_series_end_v2"
    )


def test_a_final_session_without_row_is_valued_at_the_last_traded_close(tmp_path):
    # Como DVN el 31 de diciembre de 2009: falta la fila de la última sesión, pero la serie
    # sigue después de la cinta. Se valora con el último cierre negociado, sin ejecución, y
    # ya no anula la ventana. Una serie que termina dentro de la cinta sale con una baja.
    year = tape_days("US")
    last = year.index("2023-11-30")
    write_edition(
        tmp_path,
        {
            "US": [
                Asset("GAP", base=30.0, missing=(last,)),
                Asset("END", base=40.0, end=last - 3),
                Asset("REF", base=50.0),
            ]
        },
    )
    symbols = ["END", "GAP", "REF"]
    # Solo GAP tiene puntuación positiva, así que la cartera invertida lo mantiene al final.
    window, values = monthly_window("US", -2, symbols, score=lambda k, i: 0.02 if i == 1 else -0.01)
    tape, report = build_reconstructed_tape(
        tmp_path,
        [window],
        [values],
        market="US",
        partition="validation",
        dividend_payment_lag_sessions=0,
        listing_status=listing_status(tmp_path),
    )
    assert report["excluded"] == {} and report["delistings"] == {
        "US/END": {"last_session": year[last - 3], "exit": None}
    }
    assert tape.assets == ["US/END", "US/GAP", "US/REF"]
    assert str(report["last_session"]) == "2023-11-30"
    assert report["counts"]["final_sessions_without_row"] == 1
    gap = tape.assets.index("US/GAP")
    assert np.isnan(tape.prices[-1, gap, [0, 1, 2, 4]]).all()
    assert tape.prices[-1, gap, 3] == tape.prices[-2, gap, 3]
    env = FinancialEnv(tape, capital=1_000_000)
    env.reset(seed=0)
    while not env.done:
        info = env.step(5 if env.cursor == 0 else 0)[4]
    assert info["reward_valid"] and info["reason"] == "episode_limit"
    held = env.book.positions["US/GAP"]
    cash = env.book.cash["USD"]
    assert env.book.nav["USD"] == pytest.approx(cash + held * tape.prices[-2, gap, 3], rel=1e-15)


def delisting_edition(root, *, end_offset=6):
    """Edición con END, que termina su serie en noviembre, y REF, que sigue cotizando."""
    year = tape_days("US")
    last = year.index("2023-11-30") - end_offset
    write_edition(root, {"US": [Asset("END", base=40.0, end=last), Asset("REF", base=50.0)]})
    window, values = monthly_window(
        "US", -2, ["END", "REF"], score=lambda k, i: 0.02 if i == 0 else -0.01
    )
    return year[last], window, values


def delisting_tape(root, window, values, status):
    return build_reconstructed_tape(
        root,
        [window],
        [values],
        market="US",
        partition="validation",
        dividend_payment_lag_sessions=0,
        listing_status=status,
    )


def hold_end(env):
    """Comprar END al empezar y mantenerlo. Devuelve la información de cada paso."""
    env.reset(seed=0)
    infos = []
    while not env.done:
        infos.append(env.step(5 if env.cursor == 0 else 0)[4])
    return infos


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("native", marks=requires_native_library)]
)
def test_a_held_position_in_an_unpriced_delisting_masks_the_episode(tmp_path, backend):
    last, window, values = delisting_edition(tmp_path)
    tape, report = delisting_tape(tmp_path, window, values, listing_status(tmp_path))
    end = tape.assets.index("US/END")
    at = tape.delisted_at["US/END"]
    assert tape.prices[at - 1, end, 3] > 0 and np.isnan(tape.prices[at:, end]).all()
    assert np.isnan(tape.scores[at:, end]).all() and report["predictions_after_series_end"] > 0
    (action,) = [a for a in tape.actions if a.asset == "US/END"]
    assert (action.kind, action.value, action.pay_at) == ("unpriced_delisting", 0.0, None)
    assert action.effective_at == tape.open_times[at]
    infos = hold_end(FinancialEnv(tape, capital=1_000_000, backend=backend))
    # El patrimonio pasa a ser desconocido al primer cierre sin cotización: no se inventa
    # ningún precio de salida y la recompensa queda enmascarada.
    assert len(infos) == at and not infos[-1]["reward_valid"]
    assert infos[-1]["reason"] == "unpriced_exit" and infos[-1]["nav"]["USD"] is None
    assert all(info["reward_valid"] for info in infos[:-1])


def test_a_delisted_column_keeps_no_price_or_score_after_its_delisting(tmp_path):
    last, window, values = delisting_edition(tmp_path)
    tape, _ = delisting_tape(tmp_path, window, values, listing_status(tmp_path))
    end, at = tape.assets.index("US/END"), tape.delisted_at["US/END"]
    options = dict(
        domain="real",
        currency="USD",
        partition="validation",
        prediction_times=tape.prediction_times,
        open_times=tape.open_times,
        actions=tape.actions,
        audit=tape.identity["audit"],
    )
    prices, scores = tape.prices.copy(), tape.scores.copy()
    prices[at + 1, end] = 40.0
    scores[at + 1, end] = 0.02
    for changed in (dict(prices=prices), dict(scores=scores)):
        arrays = {"prices": tape.prices, "scores": tape.scores, **changed}
        with pytest.raises(ValueError, match="conserva precios o predicciones posteriores"):
            MarketTape(arrays["prices"], tape.close_times, tape.assets, arrays["scores"], **options)


def test_only_an_unpriced_exit_censors_a_reconstructed_fit_source(tmp_path):
    """Las columnas sin precio de un activo fuera del universo no censuran un ajuste.

    Una baja sin precio de salida sí, porque deja sin valorar una posición abierta. En una
    cinta sintética cualquier cierre ausente sigue censurando, como antes de las bajas.
    """
    last, window, values = delisting_edition(tmp_path)
    unpriced, _ = delisting_tape(tmp_path, window, values, listing_status(tmp_path))
    days = tape_days("US")
    exit_ = dict(last_session=last, price=41.5, currency="USD", paid_on=days[days.index(last) + 3])
    paid = listing_status(tmp_path, exits={"US/END": exit_}, name="paid.json")
    priced, _ = delisting_tape(tmp_path, window, values, paid)
    outside, _ = build_reconstructed_tape(
        tmp_path,
        [window],
        [values],
        market="US",
        partition="validation",
        dividend_payment_lag_sessions=0,
        listing_status=listing_status(tmp_path),
        symbols=["END", "REF"],
        universe=["REF"],
    )
    assert np.isnan(outside.prices[:, outside.assets.index("US/END"), 3]).all()
    assert not outside.actions and not censors_fit(outside)
    assert np.isnan(priced.prices[:, :, 3]).any() and not censors_fit(priced)
    assert censors_fit(unpriced)
    synthetic = MarketTape(
        np.where(np.arange(4)[:, None, None] == 2, np.nan, np.full((4, 1, 5), 10.0)),
        microseconds("2023-01-03") + 86_400_000_000 * np.arange(4, dtype=np.int64),
        ["US/AAA"],
        np.zeros((4, 1)),
        domain="synthetic",
        currency="USD",
    )
    assert censors_fit(synthetic)


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("native", marks=requires_native_library)]
)
def test_without_a_position_an_unpriced_delisting_only_retires_the_asset(tmp_path, backend):
    _, window, values = delisting_edition(tmp_path)
    tape, _ = delisting_tape(tmp_path, window, values, listing_status(tmp_path))
    env = FinancialEnv(tape, capital=1_000_000, backend=backend)
    env.reset(seed=0)
    at = tape.delisted_at["US/END"]
    while not env.done:
        # Antes de la baja el activo sigue disponible. Desde ella queda retirado y ninguna
        # orden posterior puede volver a comprarlo.
        assert ("US/END" in env.book.retired) is (env.cursor >= at)
        info = env.step(1)[4]
    assert info["reward_valid"] and info["reason"] == "episode_limit"
    assert env.book.nav["USD"] == pytest.approx(1_000_000) and "US/END" in env.book.retired


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("native", marks=requires_native_library)]
)
def test_a_source_exit_price_pays_the_held_shares_on_its_payment_date(tmp_path, backend):
    last, window, values = delisting_edition(tmp_path)
    paid = tape_days("US")[tape_days("US").index(last) + 3]
    exits = {"US/END": dict(last_session=last, price=41.5, currency="USD", paid_on=paid)}
    status = listing_status(tmp_path, exits=exits)
    tape, report = delisting_tape(tmp_path, window, values, status)
    at = tape.delisted_at["US/END"]
    (action,) = [a for a in tape.actions if a.asset == "US/END"]
    assert (action.kind, action.value) == ("delisting", 41.5)
    assert action.pay_at == tape.open_times[at + 2]
    assert report["delistings"]["US/END"]["exit"] == dict(
        price=41.5, pay_at=action.pay_at, source_sha256=status[0]["sources"]["fixture"]["sha256"]
    )
    env = FinancialEnv(tape, capital=1_000_000, backend=backend)
    env.reset(seed=0)
    while env.cursor < at - 1:
        assert env.step(5 if env.cursor == 0 else 0)[4]["reward_valid"]
    held = env.book.snapshot()["state"]["positions"]["US/END"]
    info = env.step(0)[4]
    state = env.book.snapshot()["state"]
    # La baja cambia las acciones por el cobro pendiente, que cuenta en el patrimonio.
    assert held > 0 and info["reward_valid"]
    assert "US/END" not in state["positions"] and "US/END" in state["retired"]
    assert [entry["amount"] for entry in state["receivables"]] == [
        pytest.approx(held * 41.5, rel=1e-12)
    ]
    while not env.done:
        info = env.step(0)[4]
        assert info["reward_valid"]
    assert env.book.snapshot()["state"]["receivables"] == []


def test_an_exit_for_another_last_session_contradicts_the_edition(tmp_path):
    last, window, values = delisting_edition(tmp_path)
    other = tape_days("US")[tape_days("US").index(last) - 1]
    exits = {"US/END": dict(last_session=other, price=41.5, currency="USD", paid_on=last)}
    with pytest.raises(ValueError, match="última sesión de la serie"):
        delisting_tape(tmp_path, window, values, listing_status(tmp_path, exits=exits))


def test_an_event_after_the_last_row_of_a_series_excludes_the_asset(tmp_path):
    """Un dividendo fechado después de la última fila contradice la baja.

    La edición no puede saber si el evento o el final de la serie es el error, así que el
    activo se excluye con su motivo en lugar de detener la cinta o pagar un dividendo a una
    posición que ya no cotiza.
    """
    year = tape_days("US")
    last = year.index("2023-11-30") - 6
    ending = Asset("END", base=40.0, end=last, events=((last + 2, 0.5, 0.0),))
    write_edition(tmp_path, {"US": [ending, Asset("REF", base=50.0)]})
    window, values = monthly_window("US", -2, ["END", "REF"])
    tape, report = delisting_tape(tmp_path, window, values, listing_status(tmp_path))
    assert report["excluded"] == {"US/END": "invalid_event"} and report["delistings"] == {}
    assert tape.assets == ["US/REF"] and not tape.actions


def test_a_series_that_ended_before_the_tape_is_excluded_with_its_reason(tmp_path):
    year = tape_days("US")
    write_edition(
        tmp_path,
        {"US": [Asset("OLD", end=year.index("2023-10-31")), Asset("REF", base=50.0)]},
    )
    window, values = monthly_window("US", -2, ["OLD", "REF"])
    tape, report = delisting_tape(tmp_path, window, values, listing_status(tmp_path))
    assert report["excluded"] == {"US/OLD": "delisted_before_tape"}
    assert tape.assets == ["US/REF"] and report["dropped_predictions"] > 0


def test_assets_outside_the_universe_keep_their_column_without_prices(edition):
    tape, report = build(edition, symbols=["AAA", "BBB", "GGG"], universe=["AAA", "GGG"])
    assert tape.assets == ["US/AAA", "US/BBB", "US/GGG"]
    b = column(tape, "US/BBB")
    assert np.isnan(tape.prices[:, b]).all() and np.isnan(tape.scores[:, b]).all()
    assert report["outside_universe"] == 1 and report["outside_universe_predictions"] == len(tape)
    assert tape.identity["audit"]["outside_universe"] == ["US/BBB"]
    assert not [a for a in tape.actions if a.asset == "US/BBB"]
    assert tape.identity["source"]["universe_assets"] == 2
    # La cartera nunca opera ni valora el activo fuera del universo.
    env = FinancialEnv(tape)
    env.reset(seed=0)
    while not env.done:
        info = env.step(5)[4]
        assert all(trade["asset"] != "US/BBB" for trade in info["trades"])
    assert info["reward_valid"]
    with pytest.raises(ValueError, match="diseño"):
        build(edition, symbols=["AAA"], universe=["AAA", "BBB"])


def test_a_universe_column_outside_the_audit_or_with_prices_is_rejected(edition):
    tape, _ = build(edition, symbols=["AAA", "BBB"], universe=["AAA"])
    options = dict(
        domain="real",
        currency="USD",
        partition="train",
        prediction_times=tape.prediction_times,
        open_times=tape.open_times,
        actions=tape.actions,
    )

    def rebuild(audit=None, prices=None, scores=None):
        return MarketTape(
            tape.prices if prices is None else prices,
            tape.close_times,
            tape.assets,
            tape.scores if scores is None else scores,
            audit=audit or copy.deepcopy(tape.identity["audit"]),
            **options,
        )

    assert rebuild().identity["audit"] == tape.identity["audit"]
    prices = tape.prices.copy()
    prices[10, 1, 3] = 50.0
    with pytest.raises(ValueError, match="fuera del universo"):
        rebuild(prices=prices)
    scores = tape.scores.copy()
    scores[10, 1] = 0.5
    with pytest.raises(ValueError, match="fuera del universo"):
        rebuild(scores=scores)
    audit = copy.deepcopy(tape.identity["audit"])
    audit["outside_universe"] = []
    with pytest.raises(ValueError, match="cierre valorado"):
        rebuild(audit)
    audit["outside_universe"] = ["US/AAA", "US/BBB"]
    with pytest.raises(ValueError, match="dentro de su universo"):
        rebuild(audit)


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
    # Solo la baja de EEE deja sesiones sin cierre, todas posteriores a su última fila.
    listed = [k for k, asset in enumerate(tape.assets) if asset != "US/EEE"]
    assert np.isfinite(tape.prices[:, listed, 3]).all()
    e = column(tape, "US/EEE")
    assert np.isfinite(tape.prices[:-4, e, 3]).all() and np.isnan(tape.prices[-4:, e]).all()


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


def test_the_reader_segment_is_the_one_the_policy_stage_builds():
    assert WALK_FORWARD_SEGMENT == SEGMENT == "evaluation"


def test_in_sample_segments_cannot_feed_a_tape(edition):
    symbols = [a.symbol for a in US]
    # La calibración usa etiquetas que el predictor ya consultó: sus predicciones no valen.
    # El lector rechaza el tramo antes de comparar cada fin de ajuste con su decisión.
    calibration = predictions("US", symbols, start="2022-10-01", end="2022-12-31")
    window = evaluation_window("US", calibration, partition="calibration")
    with pytest.raises(ValueError, match="solo lleva predicciones de evaluación"):
        build(edition, windows=[window], predictions=[calibration], segment="calibration")
    with pytest.raises(ValueError, match="etiqueta usada"):
        evaluation_window("US", calibration, until=microseconds("2023-01-01"))


def test_tape_contract_rejects_a_softened_reconstructed_declaration(edition):
    tape, _ = build(edition)
    # Las bajas de la cinta viajan como acciones, sin ellas sus cierres ausentes no se admiten.
    options = dict(
        domain="real",
        currency="USD",
        partition="train",
        prediction_times=tape.prediction_times,
        open_times=tape.open_times,
        actions=tape.actions,
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
    # Una cinta de EE. UU. no lleva estado de cotización chino para sus activos.
    audit = copy.deepcopy(tape.identity["audit"])
    audit["listing_status"]["assets"] = {tape.assets[0]: {}}
    with pytest.raises(ValueError, match="estado de cotización de la cinta"):
        rebuild(audit)
    with pytest.raises(ValueError, match="tratamiento"):
        rebuild(currency="CNY")
    audit = copy.deepcopy(tape.identity["audit"])
    audit["prediction_fit_ends"] = [audit["prediction_fit_ends"][0] - 1] * len(tape)
    with pytest.raises(ValueError, match="ajuste anterior"):
        rebuild(audit)
    # Un límite igual al inicio del tramo cumple fin de ajuste <= decisión en cada sesión,
    # pero no el contrato del recibo, que exige dejar de ver etiquetas antes del tramo.
    audit = copy.deepcopy(tape.identity["audit"])
    start = audit["walk_forward"][0]["start"]
    assert start <= int(tape.prediction_times[0])
    audit["walk_forward"][0]["labels_used_until"] = start
    audit["prediction_fit_ends"] = [start] * len(tape)
    with pytest.raises(ValueError, match="solo lleva predicciones de evaluación"):
        rebuild(audit)
    # Validación y calibración son filas con las que el predictor eligió o calibró.
    for partition in ("train", "validation", "calibration"):
        audit = copy.deepcopy(tape.identity["audit"])
        audit["walk_forward"][0]["partition"] = partition
        with pytest.raises(ValueError, match="solo lleva predicciones de evaluación"):
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
    rules = tape_instruments(tape)

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
    env = FinancialEnv(tape, instruments=tape_instruments(tape))
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
    plain = dict(tape_instruments(tape))
    plain[tape.assets[0]] = dataclasses.replace(plain[tape.assets[0]], lot=1)
    with pytest.raises(ValueError, match="reglas de acciones A"):
        FinancialEnv(tape, instruments=plain)
    # Las reglas del tablero sin el estado de cotización tampoco bastan en una cinta real.
    board = {asset: china_a_share_instrument(asset) for asset in tape.assets}
    with pytest.raises(ValueError, match="reglas de acciones A"):
        FinancialEnv(tape, instruments=board)
    rules = tape_instruments(tape)
    assert {rule.rules[:13] for rule in rules.values()} == {"cn_a_share_v2"}
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


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("native", marks=requires_native_library)]
)
def test_a_session_without_any_row_creates_no_execution_no_price_and_no_reward(tmp_path, backend):
    # Una sesión del calendario sin fila de ningún activo, como la ausencia marcada por
    # máscara en la edición, no inventa precio, no ejecuta órdenes y no produce recompensa.
    hole = (100, 101)
    write_edition(
        tmp_path, {"US": [Asset("AAA", base=30.0, missing=hole), Asset("BBB", missing=hole)]}
    )
    symbols = ["AAA", "BBB"]
    values = predictions("US", symbols, score=lambda k, i: 0.02 + 0.01 * i)
    tape, report = build_reconstructed_tape(
        tmp_path,
        [evaluation_window("US", values)],
        [values],
        market="US",
        partition="train",
        dividend_payment_lag_sessions=0,
        listing_status=listing_status(tmp_path),
    )
    assert report["counts"]["missing_rows"] == 4 and report["excluded"] == {}
    for k in hole:
        assert np.isnan(tape.prices[k, :, [0, 1, 2, 4]]).all()
        np.testing.assert_array_equal(tape.prices[k, :, 3], tape.prices[99, :, 3])
    env = FinancialEnv(tape, capital=1_000_000, backend=backend)
    env.reset(seed=0)
    rewards, outcomes = {}, {}
    while env.cursor <= hole[-1]:
        decision = env.cursor
        # Compra al principio y vuelve a pedir exposición completa justo antes del hueco.
        action = 5 if decision in (0, hole[0] - 1, hole[0]) else 0
        _, rewards[decision], _, _, outcomes[decision] = env.step(action)
    assert env.book.positions
    for decision in (hole[0] - 1, hole[0]):
        assert outcomes[decision]["trades"] == [] and outcomes[decision]["unfilled"]
        assert all(miss["reason"] == "missing_open" for miss in outcomes[decision]["unfilled"])
        assert outcomes[decision]["costs"] == outcomes[hole[0] - 2]["costs"]
        assert rewards[decision] == 0.0
    # Las órdenes que no se ejecutaron en el hueco no se arrastran a la primera apertura real.
    assert outcomes[hole[-1]]["trades"] == []


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
