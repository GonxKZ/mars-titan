"""Recuperación y selección de la campaña tabular sin repetir casos confirmados."""

import importlib
import json
from pathlib import Path

import pytest

from mars_titan.data.storage import sha256
from mars_titan.training.checkpoints import StopRequest


def module():
    return importlib.import_module("mars_titan.training.tabular_search")


def setup(tmp_path, monkeypatch):
    engine = module()
    monkeypatch.setattr(engine, "_environment", lambda: {"fixture": True})
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            dict(
                schema_version=1,
                ridge_alphas=[0.1, 1.0],
                depths=[3, 6],
                bins=[64],
                rates=[0.1],
                rounds=2,
                search_seed=42,
                finalist_seeds=[42, 43],
                batch_size=8,
                max_batch_bytes=64 * 1024**2,
                max_host_cache_bytes=2 * 1024**3,
                on_host=False,
                checkpoint_interval=1,
                final_test_opened=False,
            )
        )
    )
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                kind="corpus_supervision",
                scope="full_corpus",
                cohort_complete=True,
                counts={"train": 12, "validation": 6},
                roots={key: str(source / key) for key in ("prepared", "samples", "labels")},
                final_test_opened=False,
            )
        )
    )
    calls = []

    def backend(manifest, folder, *, kind=None, alpha=None, resume=False, stop=None, **options):
        calls.append(dict(kind=kind or "xgboost", alpha=alpha, resume=resume, **options))
        folder.mkdir(parents=True, exist_ok=resume)
        (folder / "model.bin").write_bytes(b"modelo de prueba")
        predictions = {}
        for partition in ("train", "validation"):
            path = folder / f"{partition}.parquet"
            path.write_bytes(partition.encode())
            predictions[partition] = dict(
                path=path.name,
                sha256=sha256(path),
                metrics=dict(
                    samples=12 if partition == "train" else 6,
                    session_mae=alpha if kind else 1 / options["max_depth"],
                ),
            )
        report = dict(
            status="completed",
            model=kind or "xgboost_external_cuda",
            manifest_sha256=sha256(manifest),
            samples={"train": 12, "validation": 6},
            fitted_rows=12,
            scope="full_corpus",
            cohort_complete=True,
            final_test_opened=False,
            checkpoint=dict(path="model.bin", sha256=sha256(folder / "model.bin")),
            predictions=predictions,
        )
        if kind:
            report.update(alpha=alpha, batch_size=options["batch_size"])
        else:
            report.update(
                identity=dict(manifest_sha256=sha256(manifest), options=options),
                completed_rounds=options["rounds"],
            )
        (folder / "run.json").write_text(json.dumps(report))
        return report

    monkeypatch.setattr(engine, "run_tabular_reference", backend)
    monkeypatch.setattr(engine, "run_external_reference", backend)
    return engine, config, manifest, calls, backend


def test_search_selects_seeds_after_all_cases_and_does_not_repeat_completed(tmp_path, monkeypatch):
    engine, config, manifest, calls, _ = setup(tmp_path, monkeypatch)
    output = tmp_path / "study"
    result = engine.run_tabular_search(config, manifest, output)
    assert (
        result["status"] == "completed" and result["completed_runs"] == result["planned_runs"] == 5
    )
    assert len(calls) == 5 and calls[-1]["seed"] == 43 and calls[-1]["max_depth"] == 6
    assert result["selected"]["ridge"] == "ridge-a0.1"
    assert engine.run_tabular_search(config, manifest, output, resume=True)["status"] == "completed"
    assert len(calls) == 5


def test_pause_between_cases_resumes_without_losing_completed_work(tmp_path, monkeypatch):
    engine, config, manifest, calls, backend = setup(tmp_path, monkeypatch)
    stop = StopRequest()

    def paused(*args, **kwargs):
        result = backend(*args, **kwargs)
        stop.request_stop()
        return result

    monkeypatch.setattr(engine, "run_tabular_reference", paused)
    result = engine.run_tabular_search(config, manifest, tmp_path / "study", stop=stop)
    assert result["status"] == "paused" and result["completed_runs"] == 1
    monkeypatch.setattr(engine, "run_tabular_reference", backend)
    result = engine.run_tabular_search(config, manifest, tmp_path / "study", resume=True)
    assert result["completed_runs"] == 5 and len(calls) == 5


def test_failed_ridge_keeps_attempt_and_restarts_in_a_new_directory(tmp_path, monkeypatch):
    engine, config, manifest, calls, backend = setup(tmp_path, monkeypatch)
    folders = []

    def broken(manifest, folder, **kwargs):
        folders.append(folder)
        folder.mkdir(parents=True)
        raise OSError("fallo de prueba antes de confirmar Ridge")

    monkeypatch.setattr(engine, "run_tabular_reference", broken)
    with pytest.raises(OSError):
        engine.run_tabular_search(config, manifest, tmp_path / "study")
    monkeypatch.setattr(engine, "run_tabular_reference", backend)
    result = engine.run_tabular_search(config, manifest, tmp_path / "study", resume=True)
    first = result["runs"][0]
    assert len(first["attempts"]) == 2 and folders[0].exists()
    assert first["attempts"][0]["path"] != first["attempts"][1]["path"]


@pytest.mark.parametrize("artifact", ["prediction", "report", "config", "manifest"])
def test_resume_rejects_changed_confirmed_artifacts(tmp_path, monkeypatch, artifact):
    engine, config, manifest, _, _ = setup(tmp_path, monkeypatch)
    output = tmp_path / "study"
    result = engine.run_tabular_search(config, manifest, output)
    folder = output / result["runs"][0]["attempts"][-1]["path"]
    target = {
        "prediction": folder / "validation.parquet",
        "report": folder / "run.json",
        "config": config,
        "manifest": manifest,
    }[artifact]
    target.write_text(target.read_text() + " ")
    with pytest.raises(ValueError):
        engine.run_tabular_search(config, manifest, output, resume=True)


def test_default_design_has_twelve_boosting_cases_and_three_ridge_candidates():
    _, cases, _ = module()._configuration(Path("configs/baselines/tabular-search-us.json"))
    assert sum(case["kind"] == "ridge" for case in cases) == 3
    assert sum(case["kind"] == "xgboost" for case in cases) == 12


@pytest.mark.parametrize(
    "field", ["counts", "scope", "cohort_complete", "final_test_opened", "completed_runs", "score"]
)
def test_resume_rejects_inconsistent_summary(tmp_path, monkeypatch, field):
    engine, config, manifest, _, _ = setup(tmp_path, monkeypatch)
    output = tmp_path / "study"
    engine.run_tabular_search(config, manifest, output)
    path = output / "summary.json"
    summary = json.loads(path.read_text())
    if field == "score":
        summary["runs"][0]["session_mae"] += 1
    else:
        summary[field] = {
            "counts": {"train": 99, "validation": 6},
            "scope": "development_snapshot",
            "cohort_complete": False,
            "final_test_opened": True,
            "completed_runs": 99,
        }[field]
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        engine.run_tabular_search(config, manifest, output, resume=True)


def test_last_case_cannot_confirm_after_orchestration_code_changes(tmp_path, monkeypatch):
    engine, config, manifest, _, backend = setup(tmp_path, monkeypatch)
    real_code = engine._code

    def changed(*args, **kwargs):
        report = backend(*args, **kwargs)
        if kwargs.get("seed") == 43:
            monkeypatch.setattr(engine, "_code", lambda: real_code() | {"changed": "yes"})
        return report

    monkeypatch.setattr(engine, "run_external_reference", changed)
    with pytest.raises(ValueError, match="código"):
        engine.run_tabular_search(config, manifest, tmp_path / "study")


def test_boosting_recovers_paused_rounds_in_the_same_attempt(tmp_path, monkeypatch):
    engine, config, manifest, calls, backend = setup(tmp_path, monkeypatch)
    paused_once = False

    def pause(manifest, folder, **kwargs):
        nonlocal paused_once
        result = backend(manifest, folder, **kwargs)
        if not paused_once:
            paused_once = True
            result.update(status="paused", completed_rounds=1)
            (folder / "run.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(engine, "run_external_reference", pause)
    output = tmp_path / "study"
    first = engine.run_tabular_search(config, manifest, output)
    assert first["status"] == "paused" and first["completed_runs"] == 2
    result = engine.run_tabular_search(config, manifest, output, resume=True)
    assert result["status"] == "completed" and len(calls) == 6
    assert calls[3]["resume"] is True and len(result["runs"][2]["attempts"]) == 1


def test_completed_backend_is_recovered_after_missing_campaign_confirmation(tmp_path, monkeypatch):
    engine, config, manifest, calls, _ = setup(tmp_path, monkeypatch)
    real = engine._completed

    def failure(*args):
        raise OSError("fallo al confirmar el resultado en el resumen")

    monkeypatch.setattr(engine, "_completed", failure)
    output = tmp_path / "study"
    with pytest.raises(OSError):
        engine.run_tabular_search(config, manifest, output)
    monkeypatch.setattr(engine, "_completed", real)
    result = engine.run_tabular_search(config, manifest, output, resume=True)
    assert result["completed_runs"] == 5 and len(calls) == 5
    assert len(result["runs"][0]["attempts"]) == 1


def test_two_campaign_writers_cannot_share_the_same_output(tmp_path, monkeypatch):
    engine, config, manifest, _, backend = setup(tmp_path, monkeypatch)
    output = tmp_path / "study"

    def collision(*args, **kwargs):
        with pytest.raises(BlockingIOError):
            engine.run_tabular_search(config, manifest, output, resume=True)
        return backend(*args, **kwargs)

    monkeypatch.setattr(engine, "run_tabular_reference", collision)
    assert engine.run_tabular_search(config, manifest, output)["status"] == "completed"


@pytest.mark.parametrize(
    "change",
    [
        {"rounds": 1001},
        {"search_seed": 2**31, "finalist_seeds": [2**31]},
        {"finalist_seeds": [42, 2**31]},
    ],
)
def test_unsupported_backend_limits_fail_before_fitting_any_model(tmp_path, monkeypatch, change):
    engine, config, manifest, calls, _ = setup(tmp_path, monkeypatch)
    values = json.loads(config.read_text()) | change
    config.write_text(json.dumps(values))
    with pytest.raises(ValueError):
        engine.run_tabular_search(config, manifest, tmp_path / "study")
    assert calls == []


def test_distinct_float_parameters_have_distinct_case_identifiers(tmp_path, monkeypatch):
    engine, config, _, _, _ = setup(tmp_path, monkeypatch)
    values = json.loads(config.read_text())
    values.update(ridge_alphas=[0.1, 0.10000000000001], rates=[0.1, 0.10000000000001])
    config.write_text(json.dumps(values))
    _, cases, _ = engine._configuration(config)
    assert len({case["id"] for case in cases}) == len(cases)


def test_failed_first_summary_can_resume_an_identified_empty_preparation(tmp_path, monkeypatch):
    engine, config, manifest, calls, _ = setup(tmp_path, monkeypatch)
    output = tmp_path / "study"
    real = engine.atomic_json

    def full(path, value):
        if path == output / "summary.json":
            raise OSError(28, "No hay espacio para el primer resumen")
        return real(path, value)

    monkeypatch.setattr(engine, "atomic_json", full)
    with pytest.raises(OSError):
        engine.run_tabular_search(config, manifest, output)
    assert calls == []
    monkeypatch.setattr(engine, "atomic_json", real)
    result = engine.run_tabular_search(config, manifest, output, resume=True)
    assert result["status"] == "completed" and len(calls) == 5


def test_missing_summary_cannot_adopt_foreign_or_previous_model_files(tmp_path, monkeypatch):
    engine, config, manifest, calls, _ = setup(tmp_path, monkeypatch)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "unrelated.txt").write_text("conservar")
    with pytest.raises(ValueError):
        engine.run_tabular_search(config, manifest, foreign, resume=True)
    output = tmp_path / "study"
    engine.run_tabular_search(config, manifest, output)
    (output / "summary.json").unlink()
    with pytest.raises(ValueError):
        engine.run_tabular_search(config, manifest, output, resume=True)
    assert len(calls) == 5
