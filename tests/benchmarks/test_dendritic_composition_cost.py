"""Composiciones N2 comparables, con fuentes comunes y sin aprendizaje."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from benchmarks import dendritic_session_cost as cost

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("n2", "e2_reservoir", "e2_utility", "mechanisms", "collection")


def rows():
    return [
        dict(
            assets=64,
            variant=name,
            repetition=i,
            warmup=i == 0,
            step_seconds=1000.0 if i == 0 else float(i),
            recovery_seconds=0.5,
            source_sha256="a" * 64,
            common_sha256=("d" if name == "e2_utility" else "b") * 64,
            next_sha256=("e" if name == "e2_utility" else "c") * 64,
        )
        for i in range(3)
        for name in VARIANTS
    ]


def test_composition_protocol_separates_head_and_integration_changes():
    protocol = cost.study_protocol([64, 256], 1, 3, compositions=True)
    assert protocol["variants"] == list(VARIANTS)
    assert protocol["head"] == "quantile_head_v1"
    assert protocol["total_cases"] == 40 and protocol["measured_cases"] == 30
    assert protocol["memory_reads"] == 1 and protocol["min_assets"] == 20


def test_composition_protocol_rejects_a_nonboolean_mode():
    with pytest.raises(ValueError):
        cost.study_protocol([64], 1, 3, compositions=1)


def test_unknown_composition_is_not_treated_as_the_reference():
    with pytest.raises(ValueError):
        cost._composition_options("unknown")


def test_composition_protocol_cli_does_not_import_torch():
    code = (
        "import runpy,sys;sys.argv=['dendritic_session_cost','--compositions'];"
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
    assert json.loads(result.stdout)["variants"] == list(VARIANTS)


def test_summary_keeps_utility_separate_and_excludes_warmup():
    result = cost.summarize(rows(), compositions=True)
    assert len(result) == 5
    assert all(row["samples"] == 2 and row["median_seconds"] == 1.5 for row in result)


@pytest.mark.parametrize("name", ["e2_reservoir", "mechanisms", "collection"])
@pytest.mark.parametrize("field", ["common_sha256", "next_sha256"])
def test_parity_is_required_for_semantically_equivalent_compositions(name, field):
    records = rows()
    for record in records:
        if record["variant"] == name:
            record[field] = "f" * 64
    with pytest.raises(ValueError):
        cost.summarize(records, compositions=True)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("assets", [4, 256])
def test_native_composition_cli_uses_actual_bindings_and_preserves_the_emitted_state(
    tmp_path, assets
):
    out = tmp_path / "cost"
    subprocess.run(
        [
            "uv",
            "--no-cache",
            "run",
            "--no-sync",
            "python",
            "-m",
            "benchmarks.dendritic_session_cost",
            "--compositions",
            "--repository",
            str(ROOT),
            "--native",
            os.environ["MARS_TITAN_EPISODIC_NATIVE"],
            "--output",
            str(out),
            "--sizes",
            str(assets),
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    report = json.loads((out / "measurement.json").read_text())
    assert report["status"] == "completed" and report["optimizer_attempts"] == []
    assert report["hold_sha256_before"] == report["hold_sha256_after"]
    actual = {row["variant"]: row for row in report["cases"]}
    assert set(actual) == set(VARIANTS)
    assert all(row["bank_count_before_step"] > 0 for row in actual.values())
    assert actual["e2_reservoir"]["spaced_policy"] == "reservoir"
    assert actual["e2_utility"]["spaced_policy"] == "utility"
    assert actual["collection"]["collection_enabled"] is True
    assert actual["mechanisms"]["collection_enabled"] is False
    assert actual["mechanisms"]["binding_kind"] == "titans_mechanisms_v1"
    assert len({row["contract_id"] for row in actual.values()}) == 5
    for row in actual.values():
        assert row["consumer_id"] == report["fixtures"][0]["model_ids"][row["variant"]]
    pair = [row for name, row in actual.items() if name != "e2_utility"]
    assert len({row["common_sha256"] for row in pair}) == 1
    assert len({row["next_sha256"] for row in pair}) == 1
    assert not list(out.glob("case-*"))
