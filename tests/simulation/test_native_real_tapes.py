"""Cintas reales reconstruidas en el motor nativo: admisión, rechazos y paridad paso a paso.

Las ediciones son sintéticas con el formato real de la edición sin ajustar y se identifican
como fixture: suspensiones, filas ausentes, aperturas fuera de rejilla, dividendos con plazo,
splits y eventos ambiguos el mismo día. Las predicciones no proceden de ningún modelo. Las
acciones son las de las referencias fijas y no hay aprendizaje. Si se declaran
``MARS_TITAN_UNADJUSTED_EDITION`` y ``MARS_TITAN_LISTING_STATUS``, una prueba de humo repite
la paridad con pocos activos de la edición real de EE. UU. y China.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.evaluation import evaluate, fixed_policy
from mars_titan.simulation.listing_status import read_listing_status
from mars_titan.simulation.market_rules import china_a_share_instrument, tape_instruments
from mars_titan.simulation.native_portfolio import REASONS
from mars_titan.simulation.reconstructed_tape import build_reconstructed_tape
from mars_titan.simulation.storage import write_tape
from tests.simulation.native_library import simulator_path as simulator
from tests.simulation.policy_tape_fixture import monthly_window
from tests.simulation.unadjusted_edition_fixture import (
    Asset,
    evaluation_window,
    listing_status,
    predictions,
    tape_days,
    write_edition,
)

CAPITAL = 1_000_000
POLICIES = {
    "cash": lambda step: 1,
    "hold_initial": lambda step: 5 if step == 0 else 0,
    "rebalance_25": lambda step: 2,
    "rebalance_50": lambda step: 3,
    "rebalance_75": lambda step: 4,
    "rebalance_100": lambda step: 5,
}
EDITION = {
    "US": [
        # Suspensión, fila ausente, apertura fuera de rejilla, dividendo, split y evento ambiguo.
        Asset(
            "AAA",
            base=20.0,
            zero_volume=(30, 31),
            missing=(50,),
            off_grid_open=(70,),
            events=((90, 0.25, 0), (120, 0, 2.0), (160, 0.1, 1.5)),
        ),
        Asset("BBB", base=0.85, missing=(12, 13)),
        Asset("CCC", base=150.0, events=((100, 0.5, 0),)),
        Asset("DDD", base=45.0, zero_volume=(200,), off_grid_open=(5, 6), events=((40, 0.3, 0),)),
    ],
    "CN": [
        Asset("600000.SS", base=10.0, zero_volume=(60, 61), missing=(80,), off_grid_open=(90,)),
        Asset("300750.SZ", base=50.0, overrides={120: dict(close=50.0), 121: dict(open=60.0)}),
        Asset("000001.SZ", base=12.0, events=((100, 0.046154, 1.25), (150, 0.2, 0))),
        Asset("688981.SS", base=40.0, events=((170, 0.1, 1.5),)),
    ],
}


# Estado de cotización de la fixture: cada campo cambia la banda de 600000.SS dentro de 2023.
# STAR conserva su exención inicial y su banda del 20 % bajo advertencia, de modo que C++
# debe reconstruir las mismas reglas.
STATUS = {
    "CN/600000.SS": dict(
        special_treatment=[["2023-03-01", "2023-06-01"]],
        share_reform_pending=[["2023-07-03", "2023-08-01"]],
        limit_free_days=["2023-08-01", "2023-10-09"],
    ),
    "CN/688981.SS": dict(
        listed_on="2022-09-01",
        limit_free_until="2022-09-08",
        special_treatment=[["2023-04-03", "2023-09-01"]],
    ),
}


def rules(tape, market):
    return tape_instruments(tape) if market == "CN" else None


def build(root, market, symbols, *, lag=2, score=None, status=None):
    """Cinta de 2023 con la tabla de estado de la fixture o con la que se declare."""
    values = predictions(market, symbols, score=score)
    tape, _ = build_reconstructed_tape(
        root,
        [evaluation_window(market, values)],
        [values],
        market=market,
        partition="validation",
        dividend_payment_lag_sessions=lag,
        listing_status=status or listing_status(root, china=STATUS),
        symbols=symbols,
    )
    return tape


@pytest.fixture(scope="module")
def tapes(tmp_path_factory):
    root = tmp_path_factory.mktemp("edition")
    write_edition(root, EDITION)
    result = {}
    for market, assets in EDITION.items():
        tape = build(root, market, [asset.symbol for asset in assets])
        folder = tmp_path_factory.mktemp(f"tape-{market}")
        write_tape(tape, folder / "input", instruments=rules(tape, market))
        result[market] = tape, folder / "input"
    return result


def run(source, output, *arguments):
    result = subprocess.run(
        [str(simulator()), "--input", str(source), "--output", str(output), *arguments],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    for name in ("AddressSanitizer", "UndefinedBehaviorSanitizer", "runtime error:"):
        assert name not in result.stderr, result.stderr
    return result


def read(path):
    return json.loads(Path(path).read_text())


def assert_trace_parity(tape, market, trace, policy, cost):
    """Observación, recompensa, final, órdenes ejecutadas o no y cartera en cada paso."""
    env = FinancialEnv(tape, capital=CAPITAL, cost_bps=cost, instruments=rules(tape, market))
    observation, _ = env.reset(seed=0)
    np.testing.assert_array_equal(np.asarray(trace["initial_observation"], np.float32), observation)
    assets = tape.assets
    for step, event in enumerate(trace["steps"]):
        action = POLICIES[policy](step)
        assert event["action"] == action
        observation, reward, terminated, truncated, info = env.step(action)
        assert event["cursor"] == env.cursor
        np.testing.assert_array_equal(np.asarray(event["observation"], np.float32), observation)
        # Una recompensa enmascarada es nula en la traza C++ y cero en la interfaz de Gymnasium,
        # que exige un número. Las dos llevan `reward_valid` falso.
        expected = reward if info["reward_valid"] else None
        assert (event["reward"], event["reward_valid"]) == (expected, info["reward_valid"])
        assert info["reward_valid"] or reward == 0.0
        assert (event["terminated"], event["truncated"]) == (terminated, truncated)
        native = {}
        unfilled = []
        for trade in event["trades"]:
            if trade["quantity"] != 0:
                native[assets[trade["asset"]]] = (trade["quantity"], trade["price"], trade["cost"])
            if trade["reason"] != 0:
                unfilled.append(dict(asset=assets[trade["asset"]], reason=REASONS[trade["reason"]]))
        expected = {t["asset"]: (t["quantity"], t["price"], t["cost"]) for t in info["trades"]}
        assert native == expected
        key = lambda row: (row["reason"] == REASONS[3], row["asset"])  # noqa: E731
        assert sorted(unfilled, key=key) == info["unfilled"]
        assert [assets[i] for i in event["unvalued"]] == info["unvalued"]
        state = env.book.snapshot()["state"]
        currency = tape.currency
        account = event["account"]
        assert account["cash"] == state["cash"][currency]
        assert account["nav"] == state["nav"][currency]
        assert account["costs"] == state["costs"][currency]
        assert account["turnover"] == state["turnover"][currency]
        assert account["receivable"] == pytest.approx(
            sum(entry["amount"] for entry in state["receivables"]), rel=0, abs=1e-9
        )
        positions = {
            assets[i]: row["quantity"]
            for i, row in enumerate(event["positions"])
            if row["quantity"] != 0
        }
        assert positions == state["positions"]
        orders = {
            assets[i]: dict(
                target=row["target"], decision_at=row["decision_at"], capacity=row["capacity"]
            )
            for i, row in enumerate(event["positions"])
            if row["target"] is not None
        }
        assert orders == state["orders"]
    # Un episodio completo recorre la cinta entera. Uno truncado por una baja sin precio de
    # salida termina en su cursor, y la traza tampoco puede tener pasos posteriores.
    assert env.done and len(trace["steps"]) == env.cursor
    return env


@pytest.mark.parametrize("market", ["US", "CN"])
@pytest.mark.parametrize("policy", sorted(POLICIES))
@pytest.mark.parametrize("cost", [0, 25])
def test_cpp_session_matches_python_step_by_step_on_real_tapes(
    tapes, tmp_path, market, policy, cost
):
    tape, source = tapes[market]
    output = tmp_path / "output"
    arguments = ["--policy", policy, "--cost-bps", str(cost), "--capital", str(CAPITAL)]
    result = run(source, output, *arguments, "--trace")
    assert result.returncode == 0, result.stderr
    report = read(output / "run.json")
    assert report["domain"] == "real" and report["partition"] == "validation"
    assert report["final_test_opened"] is False and report["status"] == "completed"
    trace = read(output / "trace.json")
    assert report["trace"]["steps"] == len(trace["steps"])
    assert (
        report["trace"]["sha256"]
        == hashlib.sha256((output / "trace.json").read_bytes()).hexdigest()
    )
    assert_trace_parity(tape, market, trace, policy, cost)
    expected = evaluate(
        FinancialEnv(tape, capital=CAPITAL, cost_bps=cost, instruments=rules(tape, market)),
        fixed_policy(policy)
        if policy in ("cash", "hold_initial", "rebalance_50")
        else (lambda _o, step: POLICIES[policy](step)),
    )
    assert report["financial_validation"] == expected["financial_validation"]


@pytest.mark.parametrize("policy", ["hold_initial", "rebalance_100"])
def test_cpp_session_values_a_final_session_without_row_like_python(tmp_path, policy):
    # La serie sigue en diciembre, así que la última sesión de noviembre sin fila se valora
    # con el último cierre negociado en los dos motores.
    year = tape_days("US")
    write_edition(
        tmp_path / "edition",
        {"US": [Asset("GAP", base=30.0, missing=(year.index("2023-11-30"),)), Asset("REF")]},
    )
    window, values = monthly_window(
        "US", -2, ["GAP", "REF"], score=lambda k, i: 0.02 if i == 0 else 0.01
    )
    tape, report = build_reconstructed_tape(
        tmp_path / "edition",
        [window],
        [values],
        market="US",
        partition="validation",
        dividend_payment_lag_sessions=0,
        listing_status=listing_status(tmp_path / "edition"),
    )
    assert report["counts"]["final_sessions_without_row"] == 1
    write_tape(tape, tmp_path / "input")
    output = tmp_path / "output"
    arguments = ["--policy", policy, "--cost-bps", "10", "--capital", str(CAPITAL), "--trace"]
    result = run(tmp_path / "input", output, *arguments)
    assert result.returncode == 0, result.stderr
    env = assert_trace_parity(tape, "US", read(output / "trace.json"), policy, 10)
    assert env.book.positions["US/GAP"] > 0
    assert read(output / "run.json")["financial_validation"]["completed"] is True


@pytest.mark.parametrize("policy", ["hold_initial", "rebalance_100"])
@pytest.mark.parametrize("priced", [False, True], ids=["unpriced", "priced"])
def test_cpp_session_delists_and_ignores_an_outside_column_like_python(tmp_path, policy, priced):
    """END termina su serie en noviembre y OUT está en el diseño pero fuera del universo.

    Con precio de salida acreditado, las acciones de END se cambian por un cobro pendiente y el
    episodio sigue. Sin él, mantener END deja el patrimonio sin valorar y los dos motores
    truncan el episodio en la baja con el motivo `unpriced_exit`. OUT no tiene precios ni
    predicciones y ninguna política llega a comprarlo.
    """
    year = tape_days("US")
    last = year.index("2023-11-30") - 6
    assets = [Asset("END", base=40.0, end=last), Asset("OUT", base=60.0), Asset("REF", base=50.0)]
    write_edition(tmp_path / "edition", {"US": assets})
    window, values = monthly_window(
        "US", -2, ["END", "REF"], score=lambda k, i: 0.02 if i == 0 else 0.01
    )
    exits = dict(last_session=year[last], price=41.5, currency="USD", paid_on=year[last + 3])
    status = listing_status(tmp_path / "edition", exits={"US/END": exits} if priced else None)
    tape, report = build_reconstructed_tape(
        tmp_path / "edition",
        [window],
        [values],
        market="US",
        partition="validation",
        dividend_payment_lag_sessions=0,
        listing_status=status,
        symbols=["END", "OUT", "REF"],
        universe=["END", "REF"],
    )
    assert tape.identity["audit"]["outside_universe"] == ["US/OUT"]
    assert np.isnan(tape.prices[:, tape.assets.index("US/OUT")]).all()
    assert [action.kind for action in tape.actions] == [
        "delisting" if priced else ("unpriced_delisting")
    ]
    write_tape(tape, tmp_path / "input")
    output = tmp_path / "output"
    arguments = ["--policy", policy, "--cost-bps", "10", "--capital", str(CAPITAL), "--trace"]
    result = run(tmp_path / "input", output, *arguments)
    assert result.returncode == 0, result.stderr
    trace = read(output / "trace.json")
    env = assert_trace_parity(tape, "US", trace, policy, 10)
    at = tape.delisted_at["US/END"]
    assert "US/OUT" not in env.book.positions
    final = read(output / "run.json")["financial_validation"]
    if priced:
        # El cobro de la salida está pendiente tras la baja y cuenta en el patrimonio.
        assert len(trace["steps"]) == len(tape) - 1 and final["completed"] is True
        assert max(event["account"]["receivable"] for event in trace["steps"]) > 0
        assert "US/END" in env.book.retired and "US/END" not in env.book.positions
    else:
        # La posición se conserva sin cotización: nada la convierte en efectivo.
        assert env.book.positions["US/END"] > 0
        assert len(trace["steps"]) == at and trace["steps"][-1]["truncated"]
        assert trace["steps"][-1]["reward_valid"] is False and final["completed"] is False
    expected = evaluate(
        FinancialEnv(tape, capital=CAPITAL, cost_bps=10),
        fixed_policy(policy) if policy == "hold_initial" else (lambda _o, s: POLICIES[policy](s)),
    )
    assert final == expected["financial_validation"]


@pytest.mark.parametrize("market", ["US", "CN"])
def test_real_fixtures_exercise_events_suspensions_and_unfilled_orders(tapes, tmp_path, market):
    # La paridad no es vacía: hay dividendos cobrados, splits, órdenes sin ejecutar y ventas.
    tape, source = tapes[market]
    assert {action.kind for action in tape.actions} == {"dividend", "split"}
    assert np.isnan(tape.prices[:, :, 0]).any() and (tape.prices[:, :, 4] == 0).any()
    steps = []
    for policy in ("hold_initial", "rebalance_100"):
        output = tmp_path / policy
        run(source, output, "--policy", policy, "--capital", str(CAPITAL), "--trace")
        steps += read(output / "trace.json")["steps"]
    reasons = {t["reason"] for e in steps for t in e["trades"]}
    sells = sum(t["quantity"] < 0 for e in steps for t in e["trades"])
    receivable = max(e["account"]["receivable"] for e in steps)
    assert 1 in reasons and sells > 0 and receivable > 0


def test_identity_declares_the_real_contract_and_comparisons_accept_real_tapes(tapes, tmp_path):
    tape, source = tapes["CN"]
    result = run(source, tmp_path / "compare", "--compare", "--workers", "2")
    assert result.returncode == 0, result.stderr
    summary = read(tmp_path / "compare/comparison.json")
    assert summary["domain"] == "real" and summary["status"] == "completed"
    identity = summary["identity"]
    assert identity["historical_audit"] == "unadjusted_reconstructed_walk_forward_v1"
    edition = tape.identity["audit"]["edition_id"]
    assert identity["historical_basis"] == f"CN/{edition}/lag-2"
    assert identity["market_rules"] == "mt_simulation_step_v2/mt_rules_v1"
    assert (
        identity["manifest_sha256"]
        == hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
    )


def synthetic_manifest(tmp_path):
    """Escenario sintético mínimo para comprobar que su identidad no cambia."""
    from tests.simulation.test_native_runner import source

    return source(tmp_path)


def test_synthetic_runs_keep_their_identity_fields(tmp_path):
    result = run(synthetic_manifest(tmp_path), tmp_path / "out", "--diagnostic")
    assert result.returncode == 0, result.stderr
    identity = read(tmp_path / "out/identity.json")
    assert set(identity) == {
        "manifest_sha256",
        "parent_id",
        "native_version",
        "native_source_sha256",
        "native_build_sha256",
        "compiler_id",
        "compiler_version",
        "build_type",
        "config",
        "policy",
        "diagnostic",
        "partition",
        "rng",
    }
    assert read(tmp_path / "out/run.json")["domain"] == "technical"
    assert not (tmp_path / "out/trace.json").exists()


def copy_source(source, destination):
    destination.mkdir()
    for name in ("manifest.json", "market.parquet"):
        (destination / name).write_bytes((source / name).read_bytes())
    return destination


def rewrite(directory, manifest=None, table=None):
    if table is not None:
        path = directory / "market.parquet"
        pq.write_table(table, path, compression="zstd", row_group_size=table.num_rows)
        manifest = manifest or read(directory / "manifest.json")
        manifest["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["file_bytes"] = path.stat().st_size
    (directory / "manifest.json").write_text(json.dumps(manifest))


def segment(manifest):
    return manifest["identity"]["audit"]["walk_forward"][0]


def late_fit(manifest):
    # Cortes coherentes entre sí pero posteriores a las predicciones que acompañan.
    audit = manifest["identity"]["audit"]
    segment(manifest)["labels_used_until"] = segment(manifest)["end"]
    audit["prediction_fit_ends"] = [segment(manifest)["end"]] * len(audit["prediction_fit_ends"])


AUDIT_CHANGES = {
    "contract": lambda m: m["identity"]["audit"].update(rows="all_rows"),
    "complete": lambda m: m["identity"]["audit"].update(corporate_actions_complete=True),
    "extra_key": lambda m: m["identity"]["audit"].update(extra=1),
    "market": lambda m: m["identity"]["audit"].update(market="US"),
    "edition": lambda m: m["identity"]["audit"].update(edition_id="x"),
    "lag": lambda m: m["identity"]["audit"]["assumptions"].update(dividend_payment_lag_sessions=3),
    "lag_float": lambda m: m["identity"]["audit"]["assumptions"].update(
        dividend_payment_lag_sessions=2.0
    ),
    "segment_test": lambda m: segment(m).update(end=1_704_067_200_000_001),
    "segment_short": lambda m: segment(m).update(end=segment(m)["start"] + 1),
    "segment_partition": lambda m: segment(m).update(partition="test"),
    "segment_receipt": lambda m: segment(m).update(receipt_sha256="z" * 64),
    "late_fit": lambda m: late_fit(m),
    "fits": lambda m: m["identity"]["audit"]["prediction_fit_ends"].__setitem__(3, 0),
    "fits_length": lambda m: m["identity"]["audit"]["prediction_fit_ends"].pop(),
    "no_rules": lambda m: (m.pop("instruments"), m.update(schema_version=1)),
    "other_rules": lambda m: m["instruments"]["CN/600000.SS"]["price_limits"][0].update(band=0.15),
    "dividend_lag": lambda m: next(a for a in m["actions"] if a["kind"] == "dividend").update(
        pay_at=None
    ),
    "action_id": lambda m: m["actions"][0].update(id="renamed"),
    # El estado de la auditoría debe reproducir en C++ las reglas guardadas en la cinta.
    "status_span": lambda m: status(m)["special_treatment"][0].__setitem__(1, "2023-06-02"),
    "status_reform": lambda m: status(m).update(share_reform_pending=[]),
    "status_free_day": lambda m: status(m)["limit_free_days"].pop(),
    "status_unordered": lambda m: status(m)["limit_free_days"].reverse(),
    "status_coverage": lambda m: status(m)["special_treatment"][0].__setitem__(0, "2009-12-31"),
    "status_field": lambda m: status(m).update(note="x"),
    "status_coverage_with_rules": lambda m: before_coverage(m),
    "writeoff": lambda m: m["actions"][0].update(kind="writeoff", value=0, pay_at=None),
}


def status(manifest):
    return manifest["identity"]["audit"]["listing_status"]["assets"]["CN/600000.SS"]


def before_coverage(manifest):
    """Tramo anterior a la cobertura con las reglas guardadas recalculadas a juego.

    Sin recalcular las reglas, el lector C++ ya rechazaría la cinta por la diferencia. Así
    solo queda la comprobación de la cobertura del estado.
    """
    entry = status(manifest)
    entry["special_treatment"][0][0] = "2009-12-31"
    rules = china_a_share_instrument("CN/600000.SS", entry).identity()
    manifest["instruments"]["CN/600000.SS"] = rules


@pytest.mark.parametrize("change", sorted(AUDIT_CHANGES))
def test_altered_audit_or_rules_are_rejected_before_creating_output(tapes, tmp_path, change):
    _, source = tapes["CN"]
    directory = copy_source(source, tmp_path / "input")
    manifest = read(directory / "manifest.json")
    AUDIT_CHANGES[change](manifest)
    manifest["identity"]["actions"] = manifest["actions"]
    rewrite(directory, manifest)
    result = run(directory, tmp_path / "output", "--policy", "rebalance_50")
    assert result.returncode == 1 and result.stderr.startswith("Error: "), result.stderr
    assert not (tmp_path / "output").exists()


def column_change(table, name, mutate):
    values = table[name].to_numpy(zero_copy_only=False).copy()
    mutate(values, table)
    return table.set_column(table.schema.get_field_index(name), name, pa.array(values))


def first(table, condition):
    return int(np.flatnonzero(condition(table))[0])


PARQUET_CHANGES = {
    # Apertura ejecutable fuera de la rejilla de céntimos.
    "off_grid": lambda t: column_change(
        t,
        "open",
        lambda v, t: v.__setitem__(first(t, lambda t: np.isfinite(t["open"].to_numpy())), 10.003),
    ),
    # Sesión sin negociación con una apertura ejecutable.
    "quiet_open": lambda t: column_change(
        t,
        "open",
        lambda v, t: v.__setitem__(first(t, lambda t: t["volume"].to_numpy() == 0), 10.0),
    ),
    # Sesión negociada sin máximo.
    "traded_without_high": lambda t: column_change(
        t,
        "high",
        lambda v, t: v.__setitem__(first(t, lambda t: t["volume"].to_numpy() > 0), np.nan),
    ),
    "missing_close": lambda t: column_change(t, "close", lambda v, t: v.__setitem__(7, np.nan)),
}


@pytest.mark.parametrize("change", sorted(PARQUET_CHANGES))
def test_altered_prices_are_rejected_with_valid_hashes(tapes, tmp_path, change):
    _, source = tapes["CN"]
    directory = copy_source(source, tmp_path / "input")
    table = PARQUET_CHANGES[change](pq.read_table(directory / "market.parquet"))
    rewrite(directory, table=table)
    result = run(directory, tmp_path / "output")
    assert result.returncode == 1 and result.stderr.startswith("Error: "), result.stderr
    assert not (tmp_path / "output").exists()


def test_ambiguous_event_with_an_executable_open_is_rejected(tapes, tmp_path):
    tape, source = tapes["CN"]
    ambiguous = {
        (a.asset, a.effective_at)
        for a in tape.actions
        if a.kind == "dividend"
        and any(
            b.kind == "split" and (b.asset, b.effective_at) == (a.asset, a.effective_at)
            for b in tape.actions
        )
    }
    assert ambiguous
    asset, at = sorted(ambiguous)[0]
    session = int(np.flatnonzero(tape.open_times == at)[0])
    column = tape.assets.index(asset)
    row = session * len(tape.assets) + column
    directory = copy_source(source, tmp_path / "input")
    table = column_change(
        pq.read_table(directory / "market.parquet"),
        "open",
        lambda v, t: v.__setitem__(row, round(float(tape.prices[session, column, 3]), 2)),
    )
    rewrite(directory, table=table)
    result = run(directory, tmp_path / "output")
    assert result.returncode == 1 and "ambiguo" in result.stderr, result.stderr


@pytest.mark.parametrize("arguments", [("--resume",), ("--stop-after", "2"), ("--compare",)])
def test_trace_needs_a_complete_single_run(tapes, tmp_path, arguments):
    _, source = tapes["US"]
    result = run(source, tmp_path / "output", "--trace", *arguments)
    assert result.returncode == 1 and "--trace" in result.stderr
    assert not (tmp_path / "output").exists()


REAL_EDITION = os.environ.get("MARS_TITAN_UNADJUSTED_EDITION")
REAL_STATUS = os.environ.get("MARS_TITAN_LISTING_STATUS")
REAL_ASSETS = {
    "US": ["AAPL", "IBM", "MSFT", "JNJ", "XOM"],
    "CN": ["600519.SS", "600239.SS", "000001.SZ", "300750.SZ", "688981.SS"],
}


@pytest.mark.skipif(
    REAL_EDITION is None or REAL_STATUS is None,
    reason="Declara MARS_TITAN_UNADJUSTED_EDITION y MARS_TITAN_LISTING_STATUS",
)
@pytest.mark.parametrize("market", ["US", "CN"])
@pytest.mark.parametrize("policy", ["hold_initial", "rebalance_50", "rebalance_100"])
def test_real_edition_smoke_with_fixed_actions(tmp_path, market, policy):
    tape = build(
        Path(REAL_EDITION),
        market,
        REAL_ASSETS[market],
        lag=0,
        score=lambda k, i: 0.01 * ((i + k) % 5 - 1),
        status=read_listing_status(REAL_STATUS),
    )
    write_tape(tape, tmp_path / "input", instruments=rules(tape, market))
    result = run(
        tmp_path / "input",
        tmp_path / "out",
        "--policy",
        policy,
        "--capital",
        str(CAPITAL),
        "--trace",
    )
    assert result.returncode == 0, result.stderr
    assert_trace_parity(tape, market, read(tmp_path / "out/trace.json"), policy, 10)
