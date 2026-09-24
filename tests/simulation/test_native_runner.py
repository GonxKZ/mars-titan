"""Ejecutable C++ autónomo con Parquet, recuperación y comparación concurrente."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def binary():
    result = Path(
        os.environ.get("MARS_TITAN_SIM_EXECUTABLE", "build/native/native-debug/mars-titan-sim")
    ).resolve()
    assert result.is_file(), "Compila mars-titan-sim antes de ejecutar esta integración"
    return result


def source(tmp_path):
    path = tmp_path / "input"
    path.mkdir()
    assets, sessions = ["US/A", "US/B"], 5
    start = 1_672_531_200_000_000 + 16 * 3_600_000_000
    close = start + np.arange(sessions, dtype=np.int64) * 86_400_000_000
    prices = np.ones((sessions, 2, 5), dtype=np.float64) * 10
    prices[:, :, 4] = 1000
    columns = {
        key: prices[:, :, i].reshape(-1)
        for i, key in enumerate(("open", "high", "low", "close", "volume"))
    }
    columns.update(
        score=np.tile([0.2, 0.1], sessions),
        close_time=np.repeat(close, 2),
        open_time=np.repeat(close - 23_400_000_000, 2),
        prediction_time=np.repeat(close, 2),
        asset=np.tile(assets, sessions),
    )
    pq.write_table(pa.table(columns), path / "market.parquet", row_group_size=4)
    manifest = dict(
        schema_version=1,
        assets=2,
        sessions=sessions,
        final_test_opened=False,
        file_sha256=hashlib.sha256((path / "market.parquet").read_bytes()).hexdigest(),
        file_bytes=(path / "market.parquet").stat().st_size,
        tape_sha256="a" * 64,
        identity=dict(
            domain="synthetic",
            currency="USD",
            partition="validation",
            assets=assets,
            parent_id="fixture-v1",
            audit=None,
            actions=[],
            source={"fixture": "native-cli"},
        ),
        actions=[],
    )
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def execute(input_path, output, *args, success=True):
    result = subprocess.run(
        [str(binary()), "--input", str(input_path), "--output", str(output), "--diagnostic", *args],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if success:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
    return result


def read(path):
    return json.loads(path.read_text())


def test_cpp_reads_parquet_and_uses_actual_manifest_digest(tmp_path):
    original = source(tmp_path)
    output = tmp_path / "run"
    execute(original, output, "--policy", "cash")
    result = read(output / "run.json")
    assert result["status"] == "completed" and result["global_step"] == 4
    assert (
        result["identity"]["manifest_sha256"]
        == hashlib.sha256((original / "manifest.json").read_bytes()).hexdigest()
    )
    assert result["identity"]["manifest_sha256"] != "a" * 64
    assert result["domain"] == "technical" and result["partition"] == "validation"
    assert result["parent_frozen"] is True and result["final_test_opened"] is False
    assert result["financial_validation"] == dict(
        net_return=0,
        max_drawdown=0,
        costs=0,
        turnover=0,
        steps=4,
        completed=True,
        invalid_reason=None,
    )
    assert result["total_seconds"] >= 0 and result["process_peak_rss_bytes"] > 0
    assert result["process_lifetime_peak_rss_bytes"] == result["process_peak_rss_bytes"]
    assert result["currency"] == "USD" and result["cost_bps"] == 10
    assert result["checkpoint"]["step"] == 4
    assert result["checkpoint"]["resumable"] is True
    assert result["checkpoint"]["saved_at"].endswith("Z")
    linked = subprocess.run(["ldd", str(binary())], capture_output=True, text=True, check=True)
    assert "arrow_python" not in linked.stdout and "libpython" not in linked.stdout


def checkpoint(output):
    folder = output / "private/checkpoints"
    latest = read(folder / "latest.json")["states"]
    return folder / latest[0]["name"], latest


def test_native_resume_preserves_exact_state_and_does_not_restart_initial_policy(tmp_path):
    original = source(tmp_path)
    complete, partial = tmp_path / "complete", tmp_path / "partial"
    execute(original, complete, "--policy", "hold_initial", "--checkpoint-steps", "1")
    execute(
        original,
        partial,
        "--policy",
        "hold_initial",
        "--checkpoint-steps",
        "1",
        "--stop-after",
        "2",
    )
    paused = read(partial / "run.json")
    assert paused["status"] == "paused" and paused["global_step"] == 2
    assert paused["financial_validation"]["invalid_reason"] == "incomplete"
    execute(original, partial, "--policy", "hold_initial", "--checkpoint-steps", "1", "--resume")
    first, second = read(complete / "run.json"), read(partial / "run.json")
    assert first["financial_validation"] == second["financial_validation"]
    left, _ = checkpoint(complete)
    right, records = checkpoint(partial)
    assert read(left)["snapshot"] == read(right)["snapshot"]
    assert len(records) == 2


def test_corruption_and_different_policy_are_rejected_without_overwrite(tmp_path):
    original, output = source(tmp_path), tmp_path / "run"
    execute(original, output, "--stop-after", "0")
    path, _ = checkpoint(output)
    payload = path.read_bytes()
    execute(original, output, "--resume", "--policy", "rebalance_50", success=False)
    assert path.read_bytes() == payload
    path.write_bytes(payload + b" ")
    execute(original, output, "--resume", success=False)
    assert path.read_bytes() == payload + b" "


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_comparisons_have_the_same_results_for_every_worker_budget(tmp_path, workers):
    original = source(tmp_path)
    output = tmp_path / "comparison"
    execute(original, output, "--compare", "--workers", str(workers), "--capital", "1000")
    summary = read(output / "comparison.json")
    assert summary["status"] == "completed" and len(summary["runs"]) == 9
    assert summary["executable_peak_rss_bytes"] > 0
    assert summary["executable_peak_rss_method"] == "linux_proc_self_status_VmHWM"
    results = {}
    for record in summary["runs"]:
        report = read(output / record["path"])
        assert report["global_step"] == 4
        results[record["name"]] = report["financial_validation"]
    for cost in (0, 10, 25):
        assert results[f"cash-cost-{cost}"]["net_return"] == 0
    if workers != 1:
        reference = tmp_path / "single"
        execute(original, reference, "--compare", "--workers", "1", "--capital", "1000")
        for record in read(reference / "comparison.json")["runs"]:
            assert (
                read(reference / record["path"])["financial_validation"] == results[record["name"]]
            )


@pytest.mark.parametrize(
    "problem", ["hash", "bytes", "test", "real", "partition", "duplicate_asset", "row_count"]
)
def test_manifest_contract_rejects_changed_or_unadmitted_data(tmp_path, problem):
    original = source(tmp_path)
    path = original / "manifest.json"
    manifest = read(path)
    if problem == "hash":
        manifest["file_sha256"] = "b" * 64
    elif problem == "bytes":
        manifest["file_bytes"] += 1
    elif problem == "test":
        manifest["final_test_opened"] = True
    elif problem == "real":
        manifest["identity"]["domain"] = "real"
    elif problem == "partition":
        manifest["identity"]["partition"] = "test"
    elif problem == "duplicate_asset":
        manifest["identity"]["assets"][1] = "US/A"
    else:
        manifest["sessions"] = 4
    path.write_text(json.dumps(manifest))
    execute(original, tmp_path / "rejected", success=False)
    assert not (tmp_path / "rejected").exists()


def replace_table(directory, table):
    path = directory / "market.parquet"
    pq.write_table(table, path, row_group_size=4)
    meta = read(directory / "manifest.json")
    meta["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta["file_bytes"] = path.stat().st_size
    (directory / "manifest.json").write_text(json.dumps(meta))


@pytest.mark.parametrize(
    "problem",
    [
        "float32",
        "calendar_null",
        "calendar_mismatch",
        "asset_order",
        "extra_column",
        "infinite_price",
    ],
)
def test_parquet_contract_rejects_validly_hashed_but_incompatible_columns(tmp_path, problem):
    directory = source(tmp_path)
    table = pq.read_table(directory / "market.parquet")
    if problem == "float32":
        table = table.set_column(
            table.schema.get_field_index("score"), "score", table["score"].cast(pa.float32())
        )
    elif problem == "calendar_null":
        values = table["close_time"].to_pylist()
        values[1] = None
        table = table.set_column(
            table.schema.get_field_index("close_time"),
            "close_time",
            pa.array(values, type=pa.int64()),
        )
    elif problem == "calendar_mismatch":
        values = table["close_time"].to_numpy().copy()
        values[1] += 1
        table = table.set_column(
            table.schema.get_field_index("close_time"), "close_time", pa.array(values)
        )
    elif problem == "asset_order":
        values = table["asset"].to_pylist()
        values[0], values[1] = values[1], values[0]
        table = table.set_column(table.schema.get_field_index("asset"), "asset", pa.array(values))
    elif problem == "extra_column":
        table = table.append_column("unexpected", table["score"])
    else:
        values = table["open"].to_numpy().copy()
        values[1] = np.inf
        table = table.set_column(table.schema.get_field_index("open"), "open", pa.array(values))
    replace_table(directory, table)
    execute(directory, tmp_path / "output", success=False)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("forbidden", ["test", "real"])
def test_admission_is_rejected_before_reading_parquet(tmp_path, forbidden):
    directory = source(tmp_path)
    meta = read(directory / "manifest.json")
    meta["identity"]["partition" if forbidden == "test" else "domain"] = forbidden
    (directory / "manifest.json").write_text(json.dumps(meta))
    (directory / "market.parquet").rename(directory / "sealed.parquet")
    result = execute(directory, tmp_path / "output", success=False)
    assert any(text in result.stderr for text in ("partición", "test", "histórico"))


@pytest.mark.parametrize("policy", ["cash", "hold_initial", "rebalance_50"])
@pytest.mark.parametrize("cost", [0, 10, 25])
def test_native_financial_metrics_match_python_reference(tmp_path, policy, cost):
    from mars_titan.simulation.environment import FinancialEnv
    from mars_titan.simulation.evaluation import evaluate, fixed_policy
    from mars_titan.simulation.market import MarketTape

    directory = source(tmp_path)
    table = pq.read_table(directory / "market.parquet")
    prices = np.stack(
        [
            table[name].to_numpy().reshape(5, 2)
            for name in ("open", "high", "low", "close", "volume")
        ],
        axis=-1,
    )
    tape = MarketTape(
        prices,
        table["close_time"].to_numpy().reshape(5, 2)[:, 0],
        ["US/A", "US/B"],
        table["score"].to_numpy().reshape(5, 2),
        domain="synthetic",
        currency="USD",
        partition="validation",
        open_times=table["open_time"].to_numpy().reshape(5, 2)[:, 0],
    )
    expected = evaluate(FinancialEnv(tape, capital=1000, cost_bps=cost), fixed_policy(policy))
    output = tmp_path / "output"
    execute(directory, output, "--policy", policy, "--cost-bps", str(cost), "--capital", "1000")
    assert read(output / "run.json")["financial_validation"] == expected["financial_validation"]


def test_comparison_resumes_with_a_different_worker_count(tmp_path):
    directory = source(tmp_path)
    output = tmp_path / "output"
    execute(directory, output, "--compare", "--workers", "2", "--stop-after", "1")
    assert read(output / "comparison.json")["status"] == "paused"
    execute(directory, output, "--compare", "--workers", "8", "--resume")
    assert read(output / "comparison.json")["status"] == "completed"
    reference = tmp_path / "reference"
    execute(directory, reference, "--compare", "--workers", "1")
    for record in read(output / "comparison.json")["runs"]:
        assert (
            read(output / record["path"])["financial_validation"]
            == read(reference / record["path"])["financial_validation"]
        )


def test_numerical_failure_records_the_error_and_measured_resources(tmp_path):
    directory = source(tmp_path)
    meta = read(directory / "manifest.json")
    opening = pq.read_table(directory / "market.parquet")["open_time"].to_pylist()[4]
    action = dict(
        id="extreme-dividend",
        asset="US/A",
        kind="dividend",
        effective_at=opening,
        pay_at=opening,
        value=1e308,
        verified=True,
    )
    meta["actions"] = [action]
    meta["identity"]["actions"] = [action]
    (directory / "manifest.json").write_text(json.dumps(meta))
    output = tmp_path / "failure"
    execute(directory, output, "--policy", "hold_initial", success=False)
    report = read(output / "run.json")
    assert report["status"] == "failed"
    assert report["error_type"] and report["error"]
    assert report["process_lifetime_peak_rss_bytes"] > 0
    assert report["attempt_seconds"] >= 0
    assert report["checkpoint"]["resumable"] is True


@pytest.mark.parametrize("field", ["effective_at", "pay_at"])
def test_corporate_action_timestamps_cannot_be_truncated_from_fractional_json(tmp_path, field):
    directory = source(tmp_path)
    opening = pq.read_table(directory / "market.parquet")["open_time"].to_pylist()[4]
    action = dict(
        id="dividend",
        asset="US/A",
        kind="dividend",
        effective_at=opening,
        pay_at=opening,
        value=0.1,
        verified=True,
    )
    action[field] = float(opening) + 0.5
    meta = read(directory / "manifest.json")
    meta["actions"] = [action]
    meta["identity"]["actions"] = [action]
    (directory / "manifest.json").write_text(json.dumps(meta))
    execute(directory, tmp_path / "output", success=False)
    assert not (tmp_path / "output").exists()


def test_manifest_version_requires_an_integer(tmp_path):
    directory = source(tmp_path)
    meta = read(directory / "manifest.json")
    meta["schema_version"] = 1.0
    (directory / "manifest.json").write_text(json.dumps(meta))
    execute(directory, tmp_path / "output", success=False)
    assert not (tmp_path / "output").exists()


def test_checkpoint_index_version_cannot_be_silently_converted(tmp_path):
    directory, output = source(tmp_path), tmp_path / "output"
    execute(directory, output, "--stop-after", "0")
    path = output / "private/checkpoints/latest.json"
    index = read(path)
    index["schema_version"] = 1.0
    path.write_text(json.dumps(index))
    previous = path.read_bytes()
    execute(directory, output, "--resume", success=False)
    assert path.read_bytes() == previous


def test_executable_memory_uses_linux_vmhwm_with_explicit_scope(tmp_path):
    directory, output = source(tmp_path), tmp_path / "output"
    execute(directory, output)
    report = read(output / "run.json")
    assert report["executable_peak_rss_bytes"] > 0
    assert report["executable_peak_rss_method"] == "linux_proc_self_status_VmHWM"
    assert report["executable_peak_rss_reason"] is None


def test_build_identity_is_recorded_and_cannot_change_on_resume(tmp_path):
    directory, output = source(tmp_path), tmp_path / "output"
    execute(directory, output, "--stop-after", "0")
    report = read(output / "run.json")
    identity = report["identity"]
    assert len(identity["native_build_sha256"]) == 64
    assert identity["compiler_id"] and identity["compiler_version"] and identity["build_type"]
    marker = output / "identity.json"
    altered = read(marker)
    altered["native_build_sha256"] = "f" * 64
    marker.write_text(json.dumps(altered))
    previous = marker.read_bytes()
    execute(directory, output, "--resume", success=False)
    assert marker.read_bytes() == previous
