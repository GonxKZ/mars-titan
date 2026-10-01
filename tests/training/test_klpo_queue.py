"""Admisión de los seis padres y recuperación de una cola completa de postentrenamiento."""

import importlib
import json
from pathlib import Path

import pytest

from mars_titan.data.storage import sha256
from tests.training.test_baseline_queue import study


def engine():
    return importlib.import_module("mars_titan.training.klpo_queue")


def parents(tmp_path):
    reference, view, run = study(tmp_path)
    meta = json.loads(reference.read_text())
    template = json.loads((run / "run.json").read_text())
    template["fitted_rows"] = 12
    for partition, count in (("train", 12), ("validation", 6)):
        template["predictions"][partition]["metrics"] = dict(samples=count, session_mae=0.1)
    runs = []
    for kind in ("rnn", "lstm", "gru", "dlinear"):
        folder = reference.parent / f"runs/{kind}"
        folder.mkdir()
        report = dict(template) | dict(identity=template["identity"] | {"case": {"kind": kind}})
        for name in ("model.pt", "train.parquet", "validation.parquet"):
            (folder / name).write_bytes((run / name).read_bytes())
        (folder / "run.json").write_text(json.dumps(report))
        runs.append(
            dict(
                id=kind,
                path=f"runs/{kind}",
                arm="US",
                weighting="natural",
                stage="search",
                case={"kind": kind},
                status="completed",
                report_sha256=sha256(folder / "run.json"),
                session_mae=0.1,
            )
        )
    meta.update(
        runs=runs,
        planned_runs=4,
        completed_runs=4,
        selected={f"US-natural/{kind}": kind for kind in ("rnn", "lstm", "gru", "dlinear")},
    )
    reference.write_text(json.dumps(meta))
    tabular = tmp_path / "tabular/summary.json"
    tabular.parent.mkdir()
    runs = []
    for kind in ("ridge", "xgboost"):
        relative = f"runs/{kind}/attempt-0001"
        folder = tabular.parent / relative
        folder.mkdir(parents=True)
        for name in ("model.pt", "train.parquet", "validation.parquet"):
            (folder / name).write_bytes((run / name).read_bytes())
        options = dict(alpha=1.0, batch_size=5) if kind == "ridge" else dict(rounds=2)
        report = dict(template) | dict(
            model="ridge" if kind == "ridge" else "xgboost_external_cuda",
            identity=template["identity"] | dict(options=options),
            **(options if kind == "ridge" else {"completed_rounds": 2}),
        )
        (folder / "run.json").write_text(json.dumps(report))
        runs.append(
            dict(
                id=kind,
                kind=kind,
                stage="search",
                status="completed",
                parameters=options,
                report_sha256=sha256(folder / "run.json"),
                session_mae=0.1,
                attempts=[dict(path=relative, status="completed")],
            )
        )
    tabular.write_text(
        json.dumps(
            dict(
                status="completed",
                planned_runs=2,
                completed_runs=2,
                runs=runs,
                selected={kind: kind for kind in ("ridge", "xgboost")},
                identity=dict(manifest_sha256=sha256(view)),
                scope="full_corpus",
                cohort_complete=True,
                final_test_opened=False,
                counts=template["samples"],
            )
        )
    )
    return reference, tabular, view


def test_selects_one_confirmed_search_winner_per_family_with_shared_population(tmp_path):
    reference, tabular, view = parents(tmp_path)
    proof = engine().selected_parents(reference, tabular)
    assert proof["manifest_sha256"] == sha256(view)
    assert set(proof["parents"]) == {"rnn", "lstm", "gru", "dlinear", "ridge", "xgboost"}
    assert all(Path(item["report"]).is_file() for item in proof["parents"].values())


@pytest.mark.parametrize(
    "problem",
    [
        "incomplete",
        "duplicate",
        "missing_family",
        "mismatch",
        "test",
        "wrong_winner",
        "path",
        "report",
        "nonfinite_score",
    ],
)
def test_queue_rejects_unconfirmed_or_unrelated_parents(tmp_path, problem):
    reference, tabular, _ = parents(tmp_path)
    path = reference if problem in {"missing_family", "wrong_winner"} else tabular
    summary = json.loads(path.read_text())
    if problem == "incomplete":
        summary["status"] = "running"
    elif problem == "duplicate":
        summary["runs"][1]["id"] = summary["runs"][0]["id"]
    elif problem == "missing_family":
        summary["selected"].pop("US-natural/gru")
    elif problem == "mismatch":
        summary["identity"]["manifest_sha256"] = "a" * 64
    elif problem == "test":
        summary["final_test_opened"] = True
    elif problem == "wrong_winner":
        summary["selected"]["US-natural/gru"] = "lstm"
    elif problem == "path":
        summary["runs"][0]["attempts"][0]["path"] = "../unrelated"
    elif problem == "report":
        summary["runs"][0]["report_sha256"] = "b" * 64
    else:
        summary["runs"][0]["session_mae"] = float("nan")
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        engine().selected_parents(reference, tabular)


def test_cuda_queue_resumes_after_one_parent_and_confirms_all_rows(tmp_path, monkeypatch):
    from mars_titan.training.checkpoints import StopRequest
    from tests.training.test_klpo_study import config
    from tests.training.test_predictive_parents import setup

    source_root = tmp_path / "sources"
    ordered, parent, _ = setup(source_root)
    ordered_meta = json.loads(ordered.read_text())
    original = source_root / "data/manifest.json"
    # La prueba de cola usa dos padres reales ya contrastados. La admisión de
    # las seis familias se comprueba aparte, sin ejecutar una búsqueda completa.
    proof = dict(
        manifest=str(original.resolve()),
        manifest_sha256=sha256(original),
        counts=ordered_meta["counts"],
        parents={
            key: dict(report=str(parent.resolve()), sha256=sha256(parent))
            for key in ("first", "second")
        },
    )
    module = engine()
    monkeypatch.setattr(module, "selected_parents", lambda *args, **kwargs: proof)
    plan = config() | dict(seeds=[42], betas=[0.1], epochs=1, batch_size=5)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(plan))
    stop, real = StopRequest(), module.run_predictive_study

    def pause(*args, **kwargs):
        result = real(*args, **kwargs)
        stop.request_stop()
        return result

    monkeypatch.setattr(module, "run_predictive_study", pause)
    output = tmp_path / "queue"
    first = module.run_queue(path, parent, parent, output, stop=stop)
    assert first["status"] == "paused" and first["completed_parents"] == 1
    first_sha = sha256(output / "parents/first/summary.json")
    monkeypatch.setattr(module, "run_predictive_study", real)
    completed = module.run_queue(path, parent, parent, output)
    assert completed["status"] == "completed" and completed["completed_parents"] == 2
    assert completed["completed_runs"] == 12 and completed["planned_runs"] == 12
    assert first_sha == sha256(output / "parents/first/summary.json")
    assert completed["counts"] == {"train": 12, "validation": 6}
    assert completed["final_test_opened"] is False
    snapshot = (output / "summary.json").read_text()
    corrupted = json.loads(snapshot)
    corrupted["counts"]["train"] = 13
    (output / "summary.json").write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="identidad|recuento"):
        module.run_queue(path, parent, parent, output)
    (output / "summary.json").write_text(snapshot)
    changed = config() | dict(seeds=[42], betas=[0.1], epochs=2, batch_size=5)
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identidad"):
        module.run_queue(path, parent, parent, output)
