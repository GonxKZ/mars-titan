"""Contrato de coste técnico S2, con medidas ficticias de prueba y ejecución nativa acotada."""

import hashlib
import importlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.memory.test_native_episode_backend import native as native


def api():
    return importlib.import_module("benchmarks.failure_session_cost")


def cases():
    result = []
    for family, off, on in (("scalar", [1, 2, 3], [2, 4, 6]), ("mechanism", [4, 6, 8], [5, 7, 9])):
        for enabled, values in ((False, off), (True, on)):
            for index, value in enumerate(values):
                result.append(
                    dict(
                        assets=4,
                        family=family,
                        enabled=enabled,
                        warmup=False,
                        repetition=index,
                        step_seconds=float(value),
                        recovery_seconds=0.25,
                        persisted_bytes=100,
                        book_bytes=10 if enabled else 0,
                        common_state_sha256=("a" if family == "scalar" else "b") * 64,
                        next_state_sha256=("c" if family == "scalar" else "d") * 64,
                    )
                )
    return result


def test_summary_keeps_family_controls_and_excludes_warmup():
    rows = cases()
    rows.append(dict(rows[0], warmup=True, step_seconds=100.0))
    summary = api().summarize(rows)
    assert len(summary) == 4
    scalar = next(row for row in summary if row["family"] == "scalar" and row["enabled"])
    mechanism = next(row for row in summary if row["family"] == "mechanism" and row["enabled"])
    assert scalar["samples"] == 3
    assert scalar["median_seconds"] == 4.0
    assert scalar["relative_to_reference_percent"] == 100.0
    assert scalar["predictions_per_second"] == 1.0
    assert mechanism["median_seconds"] == 7.0
    assert mechanism["relative_to_reference_percent"] == pytest.approx(100 / 6)


def test_summary_rejects_missing_reference():
    rows = [row for row in cases() if row["enabled"]]
    with pytest.raises(ValueError):
        api().summarize(rows)


@pytest.mark.parametrize("field", ["common_state_sha256", "next_state_sha256"])
def test_summary_rejects_changed_predictions_or_recovered_state(field):
    rows = cases()
    rows[4][field] = "f" * 64
    with pytest.raises(ValueError):
        api().summarize(rows)


@pytest.mark.parametrize(
    "sizes,warmup,repeats",
    [([], 1, 3), ([4, 4], 1, 3), ([257], 1, 3), ([[4]], 1, 3), ([4], -1, 3), ([4], 1, 0)],
)
def test_protocol_rejects_unbounded_or_duplicate_cases(sizes, warmup, repeats):
    with pytest.raises(ValueError):
        api().study_protocol(sizes, warmup, repeats)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize(
    "warmup,cache_bytes,compare",
    [
        pytest.param(0, 0, False, id="0"),
        pytest.param(1, 0, False, id="1"),
        pytest.param(0, 8 * 1024**2, False, id="cached"),
        pytest.param(0, 8 * 1024**2, True, id="paired"),
    ],
)
def test_native_smoke_has_parity_no_optimizer_and_real_cost_records(
    native, tmp_path, warmup, cache_bytes, compare
):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "cost"
    command = [
        "uv",
        "--no-cache",
        "run",
        "--no-sync",
        "python",
        "-m",
        "benchmarks.failure_session_cost",
        "--repository",
        str(root),
        "--native",
        str(Path(native.__file__)),
        "--output",
        str(output),
        "--sizes",
        "4",
        "--warmup",
        str(warmup),
        "--repeats",
        "1",
        "--seconds",
        "120",
        "--decoded-cache-bytes",
        str(cache_bytes),
        "--run",
    ]
    if compare:
        command.append("--compare-decoded-cache")
    done = subprocess.run(
        command,
        cwd=root,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"),
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    report = json.loads((output / "measurement.json").read_text())
    assert report["status"] == "completed"
    assert report["optimizer_attempts"] == []
    assert report["hold_sha256_before"] == report["hold_sha256_after"]
    assert report["protocol"]["decoded_cache_bytes"] == cache_bytes
    driver = root / "benchmarks/failure_session_cost.py"
    assert (
        report["source_sha256"]["benchmarks/failure_session_cost.py"]
        == hashlib.sha256(driver.read_bytes()).hexdigest()
    )
    assert len(report["cases"]) == 4 * (warmup + 1) * (2 if compare else 1)
    assert len({row["data_source_sha256"] for row in report["cases"]}) == 1
    for row in report["cases"]:
        assert row["step_seconds"] > 0 and row["recovery_seconds"] > 0
        assert row["persisted_bytes"] > 0
        assert (row["book_bytes"] > 0) is row["enabled"]
        assert row["case_directory_retained"] is False


def test_summary_rejects_unpaired_repetitions():
    rows = cases()
    rows.pop()
    with pytest.raises(ValueError):
        api().summarize(rows)


def test_protocol_records_an_optional_bounded_decode_cache():
    record = api().study_protocol([4], 0, 1, decoded_cache_bytes=8 * 1024**2)
    assert record["decoded_cache_bytes"] == 8 * 1024**2
    with pytest.raises(ValueError):
        api().study_protocol([4], 0, 1, decoded_cache_bytes=-1)


def test_protocol_can_pair_cache_levels_on_the_same_fixture():
    record = api().study_protocol(
        [4], 0, 1, decoded_cache_bytes=8 * 1024**2, compare_decoded_cache=True
    )
    assert record["decoded_cache_budgets"] == [0, 8 * 1024**2]


def test_summary_keeps_the_two_cache_levels_separate():
    rows = cases()
    rows += [
        dict(
            row,
            decoded_cache_bytes=8 * 1024**2,
            step_seconds=row["step_seconds"] * (0.5 if row["enabled"] else 1),
        )
        for row in cases()
    ]
    summary = api().summarize(rows)
    assert len(summary) == 8
    cached = next(
        row
        for row in summary
        if row["family"] == "scalar" and row["enabled"] and row["decoded_cache_bytes"]
    )
    assert cached["median_seconds"] == 2.0
    assert cached["relative_to_reference_percent"] == 0.0
