"""La cola no avanza a tabulares con una campaña neuronal incompleta o alterada."""

import importlib
import json

import pytest

from mars_titan.data.storage import sha256


def module():
    return importlib.import_module("mars_titan.training.baseline_queue")


def study(tmp_path):
    root = tmp_path / "neural"
    (root / "views").mkdir(parents=True)
    view = root / "views/US.json"
    view.write_text(
        json.dumps(
            dict(
                scope="full_corpus",
                cohort_complete=True,
                counts={"train": 12, "validation": 6},
                source_manifest_sha256="a" * 64,
                final_test_opened=False,
            )
        )
    )
    run = root / "runs/case"
    run.mkdir(parents=True)
    artifacts = {}
    for name in ("model.pt", "train.parquet", "validation.parquet"):
        path = run / name
        path.write_bytes(name.encode())
        artifacts[name] = dict(path=name, sha256=sha256(path))
    report = dict(
        status="completed",
        scope="full_corpus",
        cohort_complete=True,
        samples={"train": 12, "validation": 6},
        final_test_opened=False,
        identity=dict(manifest_sha256=sha256(view)),
        checkpoint=artifacts["model.pt"],
        predictions={key: artifacts[key + ".parquet"] for key in ("train", "validation")},
    )
    (run / "run.json").write_text(json.dumps(report))
    summary = dict(
        kind="reference_search",
        status="completed",
        planned_runs=1,
        completed_runs=1,
        scope="full_corpus",
        cohort_complete=True,
        final_test_opened=False,
        identity=dict(manifest_sha256="a" * 64),
        runs=[
            dict(
                path="runs/case",
                status="completed",
                arm="US",
                weighting="natural",
                report_sha256=sha256(run / "run.json"),
            )
        ],
    )
    path = root / "summary.json"
    path.write_text(json.dumps(summary))
    return path, view, run


def test_gate_returns_the_exact_view_of_completed_neural_runs(tmp_path):
    summary, view, _ = study(tmp_path)
    result = module().reference_view(summary)
    assert result["manifest"] == str(view.resolve())
    assert result["manifest_sha256"] == sha256(view)
    assert result["reference_summary_sha256"] == sha256(summary)


@pytest.mark.parametrize(
    "problem",
    [
        "running",
        "paused",
        "count",
        "test",
        "cohort",
        "checkpoint",
        "predictions",
        "report",
        "view",
        "weighting",
        "path",
    ],
)
def test_gate_rejects_incomplete_or_inconsistent_parents(tmp_path, problem):
    path, view, run = study(tmp_path)
    meta = json.loads(path.read_text())
    if problem in {"running", "paused"}:
        meta["status"] = problem
    elif problem == "count":
        meta["completed_runs"] = 0
    elif problem == "test":
        meta["final_test_opened"] = True
    elif problem == "cohort":
        meta["cohort_complete"] = False
    elif problem == "weighting":
        meta["runs"][0]["weighting"] = "balanced_markets"
    elif problem == "path":
        meta["runs"][0]["path"] = "../outside"
    else:
        target = {
            "checkpoint": run / "model.pt",
            "predictions": run / "validation.parquet",
            "report": run / "run.json",
            "view": view,
        }[problem]
        target.write_bytes(target.read_bytes() + b" ")
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        module().reference_view(path)


@pytest.mark.parametrize("problem", ["missing_predictions", "report_weighting"])
def test_gate_checks_report_semantics_even_when_its_hash_is_updated(tmp_path, problem):
    path, _, run = study(tmp_path)
    report_path = run / "run.json"
    report = json.loads(report_path.read_text())
    if problem == "missing_predictions":
        report["predictions"].pop("validation")
    else:
        report["identity"]["weighting"] = "balanced_markets"
    report_path.write_text(json.dumps(report))
    meta = json.loads(path.read_text())
    meta["runs"][0]["report_sha256"] = sha256(report_path)
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        module().reference_view(path)
