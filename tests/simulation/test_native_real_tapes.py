"""Cintas reales reconstruidas en el motor nativo: admisión, rechazos y paridad paso a paso.

Las ediciones son sintéticas con el formato real de la edición sin ajustar y se identifican
como fixture: suspensiones, filas ausentes, aperturas fuera de rejilla, dividendos con plazo,
splits y eventos ambiguos el mismo día. Las predicciones no proceden de ningún modelo. Las
acciones son las de las referencias fijas y no hay aprendizaje. Si se declara
``MARS_TITAN_UNADJUSTED_EDITION``, una prueba de humo repite la paridad con pocos activos
de la edición real de EE. UU. y China.
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
from mars_titan.simulation.market_rules import china_a_share_instrument
from mars_titan.simulation.native_portfolio import REASONS
from mars_titan.simulation.reconstructed_tape import build_reconstructed_tape
from mars_titan.simulation.storage import write_tape
from tests.simulation.unadjusted_edition_fixture import (
    Asset,
    evaluation_window,
    predictions,
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


def simulator():
    path = Path(
        os.environ.get("MARS_TITAN_SIM_EXECUTABLE", "build/native/native-release/mars-titan-sim")
    ).resolve()
    if not path.is_file():
        pytest.skip("Falta el ejecutable mars-titan-sim")
    return path


def rules(tape, market):
    return {a: china_a_share_instrument(a) for a in tape.assets} if market == "CN" else None


def build(root, market, symbols, *, lag=2, score=None):
    values = predictions(market, symbols, score=score)
    tape, _ = build_reconstructed_tape(
        root,
        [evaluation_window(market, values)],
        [values],
        market=market,
        partition="validation",
        dividend_payment_lag_sessions=lag,
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
        assert (event["reward"], event["reward_valid"]) == (reward, info["reward_valid"])
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
    assert env.done and len(trace["steps"]) == len(tape) - 1
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
    "writeoff": lambda m: m["actions"][0].update(kind="writeoff", value=0, pay_at=None),
}


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
REAL_ASSETS = {
    "US": ["AAPL", "IBM", "MSFT", "JNJ", "XOM"],
    "CN": ["600519.SS", "600239.SS", "000001.SZ", "300750.SZ", "688981.SS"],
}


@pytest.mark.skipif(REAL_EDITION is None, reason="Declara MARS_TITAN_UNADJUSTED_EDITION")
@pytest.mark.parametrize("market", ["US", "CN"])
@pytest.mark.parametrize("policy", ["hold_initial", "rebalance_50", "rebalance_100"])
def test_real_edition_smoke_with_fixed_actions(tmp_path, market, policy):
    tape = build(
        Path(REAL_EDITION),
        market,
        REAL_ASSETS[market],
        lag=0,
        score=lambda k, i: 0.01 * ((i + k) % 5 - 1),
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
