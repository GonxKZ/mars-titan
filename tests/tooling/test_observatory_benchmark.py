"""Reproducción de la medición local con datos sintéticos y una CLI real."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "benchmark_observatory.py"


def load_benchmark():
    specification = importlib.util.spec_from_file_location("observatory_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_fixture_is_deterministic_and_accepted_by_real_exporter(tmp_path):
    """Detecta reloj o contenido variables y datos ajenos al contrato del exportador."""
    benchmark = load_benchmark()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_summary = benchmark.build_fixture(first)
    second_summary = benchmark.build_fixture(second)
    assert first_summary == second_summary
    assert first_summary["runs"] == 32
    assert first_summary["history_points_per_run"] == 500
    digest = hashlib.sha256()
    paths = sorted(first.glob("*/status.json"))
    assert len(paths) == 32
    for path in paths:
        relative = path.relative_to(first)
        body = path.read_bytes()
        assert body == (second / relative).read_bytes()
        state = json.loads(body)
        assert state["variant_id"] == "synthetic-benchmark-v1"
        assert state["heartbeat_at"] == "2020-01-02T12:00:00Z"
        assert len(state["history"]) == 500
        assert [point["step"] for point in state["history"]] == list(range(500))
        digest.update(relative.as_posix().encode() + b"\0" + body)
    assert first_summary["sha256"] == digest.hexdigest()
    assert first_summary["input_bytes"] == sum(path.stat().st_size for path in paths)
    output = tmp_path / "public.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/export_observatory.py"),
            "--input-dir",
            str(first),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads(output.read_text())["runs"]) == 32
    with pytest.raises(FileExistsError):
        benchmark.build_fixture(first)


def test_benchmark_cli_reports_real_attempts_metadata_and_cleans_fixture(tmp_path):
    """Detecta RSS contaminado, medidas fabricadas y temporales retenidos."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        env=os.environ | {"TMPDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["measurement"] == "synthetic_cpu_export_not_training"
    assert report["command"] == "uv run --locked python scripts/benchmark_observatory.py"
    assert report["repetitions"] == 5
    assert len(report["elapsed_seconds"]) == 5
    assert all(value > 0 for value in report["elapsed_seconds"])
    assert report["median_seconds"] == sorted(report["elapsed_seconds"])[2]
    assert report["children_rss_before_kib"] == 0
    assert report["max_child_rss_kib"] > 0
    assert 1048576 < report["output_bytes"] <= report["limits"]["max_output_bytes"]
    assert report["failed_attempts"] == []
    assert report["environment"]["system"] == "Linux"
    assert report["environment"]["python"]
    assert report["environment"]["cpu_model"]
    assert len(report["commit"]) == 40
    assert report["sha256"]["generator"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    exporter = ROOT / "scripts/export_observatory.py"
    assert report["sha256"]["exporter"] == hashlib.sha256(exporter.read_bytes()).hexdigest()
    assert report["sha256"]["fixture"] == report["fixture"]["sha256"]
    assert list(tmp_path.iterdir()) == []


def test_failed_exports_are_recorded_without_successful_timings(tmp_path):
    """Detecta presentar una CLI fallida como repetición válida o reutilizar salida anterior."""
    benchmark = load_benchmark()
    source = tmp_path / "runs"
    benchmark.build_fixture(source)
    path = sorted(source.glob("*/status.json"))[0]
    state = json.loads(path.read_text())
    state["phase"] = "invalid-synthetic-phase"
    path.write_text(json.dumps(state))
    output = tmp_path / "previous.json"
    output.write_text('{"previous": true}')

    result = benchmark.measure_exports(source, output)

    assert result["elapsed_seconds"] == []
    assert result["median_seconds"] is None
    assert result["output_bytes"] is None
    assert [attempt["attempt"] for attempt in result["failed_attempts"]] == [1, 2, 3, 4, 5]
    assert all(attempt["returncode"] == 1 for attempt in result["failed_attempts"])
    assert all(attempt["elapsed_seconds"] > 0 for attempt in result["failed_attempts"])
    assert output.read_text() == '{"previous": true}'
