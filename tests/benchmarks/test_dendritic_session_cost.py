"""Protocolo de coste N2 y paridad de sus referencias, sin entrenamiento."""

import json
import subprocess
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def records():
    module = import_module("benchmarks.dendritic_session_cost")
    return [
        dict(
            assets=64,
            variant=name,
            repetition=i,
            warmup=i == 0,
            step_seconds=1000.0 if i == 0 else float(i),
            recovery_seconds=0.5,
            source_sha256="a" * 64,
            common_sha256="b" * 64,
            next_sha256="c" * 64,
        )
        for i in range(3)
        for name in module.VARIANTS
    ]


def test_protocol_is_bounded_and_keeps_the_zero_gain_control_separate():
    assert find_spec("benchmarks.dendritic_session_cost") is not None
    protocol = import_module("benchmarks.dendritic_session_cost").study_protocol([64, 256], 1, 3)
    assert protocol["total_cases"] == 40
    assert protocol["measured_cases"] == 30
    assert protocol["variants"] == [
        "reference",
        "zero_gain",
        "observable",
        "past_prefix",
        "constant",
    ]
    assert protocol["device"] == "cpu" and protocol["memory_reads"] == 1


@pytest.mark.parametrize(
    "arguments",
    [
        ([], 1, 3),
        ([[]], 1, 3),
        ([64, 64], 1, 3),
        ([True], 1, 3),
        ([32], 1, 3),
        ([64], -1, 3),
        ([64], 1, 0),
    ],
)
def test_protocol_rejects_unsupported_or_unbounded_requests(arguments):
    with pytest.raises(ValueError):
        import_module("benchmarks.dendritic_session_cost").study_protocol(*arguments)


def test_summary_excludes_warmup_and_preserves_each_variant():
    result = import_module("benchmarks.dendritic_session_cost").summarize(records())
    assert len(result) == 5
    assert all(row["samples"] == 2 and row["median_seconds"] == 1.5 for row in result)


@pytest.mark.parametrize(
    "corruption",
    [
        "zero_parity",
        "next_parity",
        "missing",
        "duplicate",
        "exposure",
        "source",
        "nan",
        "zero",
        "repeat_state",
    ],
)
def test_summary_rejects_misaligned_or_invalid_measurements(corruption):
    rows = [r for r in records() if not r["warmup"]]
    chosen = next(r for r in rows if r["variant"] == "zero_gain")
    if corruption == "zero_parity":
        for r in rows:
            if r["variant"] == "zero_gain":
                r["common_sha256"] = "d" * 64
    if corruption == "next_parity":
        for r in rows:
            if r["variant"] == "zero_gain":
                r["next_sha256"] = "d" * 64
    if corruption == "missing":
        rows = [r for r in rows if r["variant"] != "constant"]
    if corruption == "duplicate":
        rows.append(dict(chosen))
    if corruption == "exposure":
        rows.remove(chosen)
    if corruption == "source":
        chosen["source_sha256"] = "f" * 64
    if corruption == "nan":
        chosen["step_seconds"] = float("nan")
    if corruption == "zero":
        chosen["step_seconds"] = 0.0
    if corruption == "repeat_state":
        next(r for r in rows if r["variant"] == "observable")["common_sha256"] = "e" * 64
    with pytest.raises(ValueError):
        import_module("benchmarks.dendritic_session_cost").summarize(rows)


def test_protocol_cli_does_not_import_torch():
    code = (
        "import runpy,sys;sys.argv=['dendritic_session_cost'];"
        "runpy.run_module('benchmarks.dendritic_session_cost',run_name='__main__');"
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run(
        ["uv", "--no-cache", "run", "--no-sync", "python", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["total_cases"] == 40


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_native_cli_records_real_cost_and_zero_gain_parity_without_learning(tmp_path):
    import os

    output = tmp_path / "measurement"
    command = [
        "uv",
        "--no-cache",
        "run",
        "--no-sync",
        "python",
        "-m",
        "benchmarks.dendritic_session_cost",
        "--repository",
        str(ROOT),
        "--native",
        os.environ["MARS_TITAN_EPISODIC_NATIVE"],
        "--output",
        str(output),
        "--sizes",
        "4",
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--run",
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=180)
    report = json.loads((output / "measurement.json").read_text())
    assert report["status"] == "completed" and len(report["cases"]) == 5
    assert report["optimizer_attempts"] == []
    assert report["hold_sha256_before"] == report["hold_sha256_after"]
    assert all(r["step_seconds"] > 0 and r["recovery_seconds"] > 0 for r in report["cases"])
    assert len({r["source_sha256"] for r in report["cases"]}) == 1
    assert "benchmarks/dendritic_session_cost.py" in report["source_sha256"]
    assert all(
        r["consumer_id"] == report["fixtures"][0]["model_ids"][r["variant"]]
        for r in report["cases"]
    )
    assert all(
        (r["context_record_bytes"] == 0) == (r["variant"] == "reference") for r in report["cases"]
    )
    pair = [r for r in report["cases"] if r["variant"] in ("reference", "zero_gain")]
    assert len({r["common_sha256"] for r in pair}) == len({r["next_sha256"] for r in pair}) == 1
    assert not list(output.glob("case-*"))
