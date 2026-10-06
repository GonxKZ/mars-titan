"""Cierre recuperable del análisis con recibos técnicos, sin entrenamiento."""

import fcntl
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mars_titan.data.storage import atomic_json, sha256


def coordinator(tmp_path, status="completed"):
    paths = {name: tmp_path / name for name in ("state", "references", "completion")}
    reference = dict(
        kind="temporal_reference_search",
        status="completed",
        planned_runs=2,
        completed_runs=2,
        final_test_opened=False,
    )
    atomic_json(paths["references"] / "summary.json", reference)
    completion = dict(
        kind="temporal_posttraining_completion",
        status="completed",
        planned_runs=3,
        completed_runs=3,
        final_test_opened=False,
        identity=dict(reference_sha256=sha256(paths["references"] / "summary.json")),
    )
    atomic_json(paths["completion"] / "summary.json", completion)
    reliability = paths["state"] / "reliability"
    reliability.mkdir(parents=True)
    (reliability / "cases.csv").write_text("model,confidence\nfixture,0.9\n")
    atomic_json(
        reliability / "reliability.json",
        dict(
            kind="frozen_campaign_reliability",
            status="completed",
            final_test_opened=False,
            target_kind="residual_return",
            coverage_guaranteed=False,
            counts=dict(models=2, prediction_files=4),
            provenance=dict(
                reference_sha256=sha256(paths["references"] / "summary.json"),
                completion_sha256=sha256(paths["completion"] / "summary.json"),
                domain="real",
                final_test_opened=False,
            ),
            artifacts={"cases.csv": sha256(reliability / "cases.csv")},
        ),
    )
    stages = [
        dict(
            name=name,
            unit=unit,
            planned=count,
            completed=count,
            status="completed",
            sha256=sha256(path),
        )
        for name, unit, count, path in (
            ("neural", "run", 2, paths["references"] / "summary.json"),
            ("completion", "run", 3, paths["completion"] / "summary.json"),
            ("reliability", "model", 2, reliability / "reliability.json"),
        )
    ]
    state = dict(
        schema_version=1,
        kind="real_retraining_campaign",
        domain="real",
        status=status,
        final_test_opened=False,
        planned_runs=5,
        completed_runs=5,
        planned_reliability_models=2,
        completed_reliability_models=2,
        stages=stages,
        identity=dict(
            final_test_opened=False,
            paths={
                "state_dir": str(paths["state"]),
                "references": str(paths["references"]),
                "completion": str(paths["completion"]),
            },
        ),
        updated_at_utc="2026-10-06T00:00:00+00:00",
    )
    state_path = paths["state"] / "summary.json"
    atomic_json(state_path, state)
    return state_path, state


def module():
    path = Path(__file__).parents[2] / "scripts/finish_real_campaign.py"
    assert path.is_file(), "Falta el comando de cierre"
    spec = importlib.util.spec_from_file_location("scripts.finish_real_campaign", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_open_campaign_waits_without_creating_output_or_loading_analysis(tmp_path, monkeypatch):
    finish = module()
    state, _ = coordinator(tmp_path, "running")

    def forbidden(*args, **kwargs):
        raise AssertionError("El análisis no debe comenzar mientras siga abierta")

    monkeypatch.setattr(finish, "_compare", forbidden)
    monkeypatch.setattr(finish, "_export", forbidden)
    output = tmp_path / "not-created" / "analysis"
    result = finish.finish_campaign(state, output)
    assert result["status"] == "waiting"
    assert not output.parent.exists()
    assert finish.main(["--state", str(state), "--output", str(output)]) == 3


def analyzers(finish, monkeypatch):
    calls = []
    monkeypatch.setattr(
        finish,
        "_code",
        lambda: dict(
            package={
                "evaluation/comparison_sources.py": "a" * 64,
                "evaluation/prediction_statistics.py": "b" * 64,
                "evaluation/campaign_comparison.py": "c" * 64,
                "training/real_campaign.py": "d" * 64,
            },
            scripts={"export_campaign_comparison.py": "b" * 64},
        ),
    )

    def compare(reference, completion, output):
        calls.append(("compare", output))
        output.mkdir()
        artifacts = {}
        for name in (
            "cases.csv",
            "methods.csv",
            "folds.csv",
            "intervals.csv",
            "session-errors.parquet",
        ):
            (output / name).write_bytes(b"fixture\n")
            artifacts[name] = sha256(output / name)
        atomic_json(
            output / "comparison.json",
            dict(
                status="completed",
                final_test_opened=False,
                counts=dict(models=2, prediction_files=4),
                provenance=dict(
                    reference_sha256=sha256(reference / "summary.json"),
                    completion_sha256=sha256(completion / "summary.json"),
                    domain="real",
                    final_test_opened=False,
                ),
                method=dict(repetitions=2000, seed=42),
                analysis_source_sha256={
                    name: digest
                    for name, digest in finish._code()["package"].items()
                    if name.startswith("evaluation/")
                },
                artifacts=artifacts,
            ),
        )

    def export(comparison, reliability, output):
        calls.append(("export", output))
        output.mkdir()
        for name in finish.EXPORT_FILES:
            (output / name).write_text("fixture\n")
        atomic_json(
            output / "evidence.json",
            dict(
                predictive=dict(
                    report_sha256=sha256(comparison / "comparison.json"),
                    final_test_opened=False,
                    source_artifacts={
                        name: sha256(comparison / name)
                        for name in ("cases.csv", "methods.csv", "folds.csv", "intervals.csv")
                    },
                ),
                reliability=dict(
                    report_sha256=sha256(reliability / "reliability.json"),
                    final_test_opened=False,
                    source_artifacts={"cases.csv": sha256(reliability / "cases.csv")},
                ),
                renderer=dict(
                    script_sha256=finish._code()["scripts"]["export_campaign_comparison.py"]
                ),
                artifacts={name: sha256(output / name) for name in finish.EXPORT_FILES},
            ),
        )

    monkeypatch.setattr(finish, "_compare", compare)
    monkeypatch.setattr(finish, "_export", export)
    return calls, compare, export


def test_completed_campaign_is_analyzed_once_and_reused_after_timestamp_change(
    tmp_path, monkeypatch
):
    finish = module()
    state_path, state = coordinator(tmp_path)
    calls, _, _ = analyzers(finish, monkeypatch)
    output = tmp_path / "analysis"
    first = finish.finish_campaign(state_path, output)
    assert first["status"] == "completed"
    assert [name for name, _ in calls] == ["compare", "export"]
    assert (output / "attempt-001/comparison/comparison.json").is_file()
    state["updated_at_utc"] = "2026-10-07T00:00:00+00:00"
    atomic_json(state_path, state)
    repeated = finish.finish_campaign(state_path, output)
    assert repeated["status"] == "completed"
    assert len(calls) == 2


@pytest.mark.parametrize(
    "problem", ["domain", "sealed", "stage", "counts", "state_dir", "stage_hash"]
)
def test_invalid_coordinator_never_starts_analysis(tmp_path, monkeypatch, problem):
    finish = module()
    path, state = coordinator(tmp_path)
    calls, _, _ = analyzers(finish, monkeypatch)
    if problem == "domain":
        state["domain"] = "synthetic"
    elif problem == "sealed":
        state["final_test_opened"] = True
    elif problem == "stage":
        state["stages"].pop()
    elif problem == "counts":
        state["completed_runs"] -= 1
    elif problem == "state_dir":
        state["identity"]["paths"]["state_dir"] = str(tmp_path)
    else:
        state["stages"][0]["sha256"] = "b" * 64
    atomic_json(path, state)
    output = tmp_path / "analysis"
    with pytest.raises(ValueError):
        finish.finish_campaign(path, output)
    assert not calls and not output.exists()


def test_failure_between_stages_is_preserved_and_next_attempt_is_separate(tmp_path, monkeypatch):
    finish = module()
    state, _ = coordinator(tmp_path)
    calls, _, export = analyzers(finish, monkeypatch)

    def fail(*args):
        raise RuntimeError("exportación interrumpida")

    monkeypatch.setattr(finish, "_export", fail)
    output = tmp_path / "analysis"
    with pytest.raises(RuntimeError, match="interrumpida"):
        finish.finish_campaign(state, output)
    partial = output / "attempt-001/comparison/comparison.json"
    digest = sha256(partial)
    receipt = json.loads((output / "summary.json").read_text())
    assert receipt["attempts"][0]["status"] == "failed"
    monkeypatch.setattr(finish, "_export", export)
    result = finish.finish_campaign(state, output)
    assert result["status"] == "completed" and len(result["attempts"]) == 2
    assert sha256(partial) == digest
    assert (output / "attempt-002/export/evidence.json").is_file()


def test_unexpected_termination_and_three_attempt_limit(tmp_path, monkeypatch):
    finish = module()
    state, _ = coordinator(tmp_path)
    calls, _, _ = analyzers(finish, monkeypatch)

    def fail(*args):
        raise RuntimeError("fallo técnico")

    monkeypatch.setattr(finish, "_compare", fail)
    output = tmp_path / "analysis"
    with pytest.raises(RuntimeError):
        finish.finish_campaign(state, output)
    path = output / "summary.json"
    receipt = json.loads(path.read_text())
    receipt["attempts"][0]["status"] = "running"
    receipt["status"] = "running"
    atomic_json(path, receipt)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            finish.finish_campaign(state, output)
    receipt = json.loads(path.read_text())
    assert receipt["attempts"][0]["status"] == "interrupted"
    assert len(receipt["attempts"]) == 3
    with pytest.raises(ValueError, match="tres"):
        finish.finish_campaign(state, output)
    assert not (output / "attempt-004").exists()


@pytest.mark.parametrize("problem", ["artifact", "code", "source", "identity"])
def test_completed_result_rejects_corruption_and_identity_changes(tmp_path, monkeypatch, problem):
    finish = module()
    state_path, state = coordinator(tmp_path)
    calls, _, _ = analyzers(finish, monkeypatch)
    output = tmp_path / "analysis"
    finish.finish_campaign(state_path, output)
    if problem == "artifact":
        (output / "attempt-001/export/predictive-periods.svg").write_text("alterado")
    elif problem == "code":
        monkeypatch.setattr(finish, "_code", lambda: {"fixture.py": "b" * 64})
    elif problem == "source":
        (state_path.parent / "reliability/cases.csv").write_text("alterado")
    else:
        state["identity"]["resource_change"] = True
        atomic_json(state_path, state)
    with pytest.raises(ValueError):
        finish.finish_campaign(state_path, output)
    assert len(calls) == 2


def test_lock_and_overlapping_paths_are_rejected(tmp_path, monkeypatch):
    finish = module()
    state, _ = coordinator(tmp_path)
    calls, _, _ = analyzers(finish, monkeypatch)
    with pytest.raises(ValueError):
        finish.finish_campaign(state, state.parent / "analysis")
    output = tmp_path / "analysis"
    output.mkdir()
    with (output / ".analysis.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            finish.finish_campaign(state, output)
    assert not calls


def test_reliability_provenance_cannot_open_the_reserve(tmp_path, monkeypatch):
    finish = module()
    path, state = coordinator(tmp_path)
    calls, _, _ = analyzers(finish, monkeypatch)
    reliable = path.parent / "reliability/reliability.json"
    report = json.loads(reliable.read_text())
    report["provenance"]["final_test_opened"] = True
    atomic_json(reliable, report)
    state["stages"][2]["sha256"] = sha256(reliable)
    atomic_json(path, state)
    with pytest.raises(ValueError):
        finish.finish_campaign(path, tmp_path / "out")
    assert not calls


def test_inconsistent_attempt_status_is_not_treated_as_a_new_attempt(tmp_path, monkeypatch):
    finish = module()
    state, _ = coordinator(tmp_path)
    calls, _, _ = analyzers(finish, monkeypatch)
    output = tmp_path / "analysis"
    finish.finish_campaign(state, output)
    path = output / "summary.json"
    report = json.loads(path.read_text())
    report["status"] = "failed"
    atomic_json(path, report)
    with pytest.raises(ValueError):
        finish.finish_campaign(state, output)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "problem",
    [
        "comparison_code",
        "unknown_null_code",
        "known_unrelated_code",
        "missing_required_code",
        "missing_code",
        "renderer_code",
        "predictive_test",
        "reliability_test",
        "predictive_sources",
        "reliability_sources",
    ],
)
def test_result_rejects_contradictory_producer_metadata(tmp_path, monkeypatch, problem):
    finish = module()
    state, _ = coordinator(tmp_path)
    _, compare, export = analyzers(finish, monkeypatch)

    def altered_compare(reference, completion, output):
        compare(reference, completion, output)
        if problem in {
            "comparison_code",
            "missing_code",
            "unknown_null_code",
            "known_unrelated_code",
            "missing_required_code",
        }:
            path = output / "comparison.json"
            report = json.loads(path.read_text())
            if problem == "comparison_code":
                report["analysis_source_sha256"]["evaluation/campaign_comparison.py"] = "f" * 64
            elif problem == "missing_code":
                report["analysis_source_sha256"] = {}
            elif problem == "unknown_null_code":
                report["analysis_source_sha256"] = {"unknown.py": None}
            elif problem == "known_unrelated_code":
                report["analysis_source_sha256"]["training/real_campaign.py"] = "d" * 64
            else:
                del report["analysis_source_sha256"]["evaluation/prediction_statistics.py"]
            atomic_json(path, report)

    def altered_export(comparison, reliability, output):
        export(comparison, reliability, output)
        path = output / "evidence.json"
        report = json.loads(path.read_text())
        if problem == "renderer_code":
            report["renderer"]["script_sha256"] = "f" * 64
        for kind in ("predictive", "reliability"):
            if problem == f"{kind}_test":
                report[kind]["final_test_opened"] = True
            if problem == f"{kind}_sources":
                report[kind]["source_artifacts"]["cases.csv"] = "f" * 64
        atomic_json(path, report)

    monkeypatch.setattr(finish, "_compare", altered_compare)
    monkeypatch.setattr(finish, "_export", altered_export)
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        finish.finish_campaign(state, output)
    assert json.loads((output / "summary.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("change", ["code", "source"])
def test_last_artifact_cannot_change_identity_before_confirmation(
    tmp_path, monkeypatch, reuse, change
):
    finish = module()
    state_path, state = coordinator(tmp_path)
    analyzers(finish, monkeypatch)
    output = tmp_path / "out"
    if reuse:
        finish.finish_campaign(state_path, output)
    original = finish._artifact
    seen = []

    def changed(path, digest):
        result = original(path, digest)
        if path.parent.name == "export":
            seen.append(path.name)
            if len(seen) == len(finish.EXPORT_FILES):
                if change == "code":
                    monkeypatch.setattr(finish, "_code", lambda: {"changed": "f" * 64})
                else:
                    source = Path(state["identity"]["paths"]["references"]) / "summary.json"
                    report = json.loads(source.read_text())
                    report["changed"] = True
                    atomic_json(source, report)
        return result

    monkeypatch.setattr(finish, "_artifact", changed)
    with pytest.raises(ValueError):
        finish.finish_campaign(state_path, output)
    assert len(seen) == len(finish.EXPORT_FILES)
    receipt = json.loads((output / "summary.json").read_text())
    assert receipt["status"] == receipt["attempts"][-1]["status"] == "failed"


def test_code_identity_does_not_import_the_scientific_stack():
    root = Path(__file__).parents[2]
    program = """import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('finish',Path('scripts/finish_real_campaign.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
identity=module._code()
names=('torch','numpy','pyarrow','matplotlib')
print(json.dumps(dict(identity=identity,loaded=[name for name in names if name in sys.modules])))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src"), "CUDA_VISIBLE_DEVICES": "-1"},
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    report = json.loads(result.stdout)
    assert report["loaded"] == []
    name = "evaluation/campaign_comparison.py"
    assert report["identity"]["package"][name] == sha256(root / "src/mars_titan" / name)
