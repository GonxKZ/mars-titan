"""El benchmark durable usa solo fixtures y prueba sus transiciones recuperadas."""

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/lesion_session_cost.py"
# El benchmark arma su propia alarma con este presupuesto desde que empieza su cuerpo. El límite
# del proceso añade el arranque de uv, coverage, torch y el binario episódico y el cierre de
# coverage. Con la CPU compartida con otras baterías el cuerpo llegó a 159 s y el proceso a 201,
# y con 90 y 100 s la prueba falló dos veces en la regresión (#803).
BUDGET = 360
STARTUP = 120


def test_cost_summary_keeps_pairing_and_reports_added_time():
    from benchmarks.lesion_session_cost import cost_summary

    rows = [
        dict(
            assets=64,
            profile="bridge",
            diagnostics=active,
            repetition=rep,
            warmup=rep == 0,
            step_seconds=seconds,
            common_state_sha256="a" * 64,
        )
        for active, values in ((False, [99.0, 2.0, 4.0, 6.0]), (True, [0.1, 5.0, 7.0, 9.0]))
        for rep, seconds in enumerate(values)
    ]
    summary = cost_summary(rows)[0]
    assert summary["original"]["median_seconds"] == 4
    assert summary["diagnostic"]["median_seconds"] == 7
    assert summary["delta_median_seconds"] == 3
    assert summary["relative_extra_percent"] == 75
    assert summary["original"]["rows_per_second_at_median"] == 16
    assert summary["original"]["samples"] == 3
    assert "p99" not in summary["original"]


@pytest.mark.parametrize("seconds", [0, -1, math.nan, math.inf])
def test_invalid_timings_cannot_become_cost_comparisons(seconds):
    from benchmarks.lesion_session_cost import cost_summary

    rows = [
        dict(
            assets=64,
            profile="bridge",
            diagnostics=active,
            repetition=1,
            warmup=False,
            step_seconds=seconds,
            common_state_sha256="a" * 64,
        )
        for active in (False, True)
    ]
    with pytest.raises(ValueError):
        cost_summary(rows)


@pytest.mark.parametrize("fault", ["state", "repetition", "missing", "duplicate"])
def test_unpaired_measurements_cannot_be_reported_as_diagnostic_cost(fault):
    from benchmarks.lesion_session_cost import cost_summary

    rows = [
        dict(
            assets=64,
            profile="bridge",
            diagnostics=active,
            repetition=1,
            warmup=False,
            step_seconds=1.0,
            common_state_sha256="a" * 64,
        )
        for active in (False, True)
    ]
    if fault == "state":
        rows[1]["common_state_sha256"] = "b" * 64
    elif fault == "repetition":
        rows[1]["repetition"] += 1
    elif fault == "missing":
        rows.pop()
    else:
        rows.append(dict(rows[0]))
    with pytest.raises(ValueError):
        cost_summary(rows)


def arguments(output):
    return [
        "--repository",
        str(ROOT),
        "--output",
        str(output),
        "--native",
        os.environ["MARS_TITAN_EPISODIC_NATIVE"],
        "--sizes",
        "4",
        "--profiles",
        "bridge",
        "--warmup",
        "0",
        "--repeats",
        "2",
        "--profile",
        "--seconds",
        str(BUDGET),
    ]


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_benchmark_preserves_paired_originals_and_the_next_transition(tmp_path):
    assert SCRIPT.is_file(), "Falta el benchmark de sesión B4"
    output = tmp_path / "cost"
    coverage = tmp_path / "benchmark-coverage-db"
    command = [
        "uv",
        "run",
        "--no-sync",
        "coverage",
        "run",
        "--branch",
        "--source=benchmarks",
        f"--data-file={coverage}",
        str(SCRIPT),
        *arguments(output),
    ]
    child = subprocess.run(
        command,
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH="src:."),
        capture_output=True,
        text=True,
        timeout=BUDGET + STARTUP,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    report = json.loads((output / "measurement.json").read_text())
    assert report["status"] == "completed" and report["optimizer_attempts"] == []
    assert report["source_fixtures"][0]["label_origin"] == "manual_fixture_no_estimation"
    assert report["source_fixtures"][0]["input_logical_bytes"] > 0
    assert len(report["cases"]) == 4
    assert all(row["rows"] == row["applied"] == 4 for row in report["cases"])
    assert len({row["common_state_sha256"] for row in report["cases"]}) == 1
    for active in (False, True):
        selected = [row for row in report["cases"] if row["diagnostics"] == active]
        assert len({row["next_state_sha256"] for row in selected}) == 1
        assert len({row["repeated_state_sha256"] for row in selected}) == 1
        assert all(row["recovery_exact"] for row in selected)
        assert selected[1]["recovery_seconds"] > 0
    assert coverage.is_file()


@pytest.mark.parametrize("change", ["--sizes", "--seconds"])
@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_invalid_budget_or_shape_cannot_create_a_measurement(tmp_path, change):
    assert SCRIPT.is_file(), "Falta el benchmark de sesión B4"
    output = tmp_path / "invalid"
    args = arguments(output)
    args[args.index(change) + 1] = "0"
    child = subprocess.run(
        ["uv", "run", "--no-sync", "python", str(SCRIPT), *args],
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH="src:."),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode != 0
    assert "ValueError: La salida debe ser nueva y el presupuesto debe ser acotado" in child.stderr
    assert not output.exists()
