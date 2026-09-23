"""Búsqueda, finalistas y controles que conservan población y resultados confirmados."""

import importlib
import json

import pytest

from mars_titan.data.storage import sha256
from tests.training.test_reference_run import training_corpus


def module():
    try:
        return importlib.import_module("mars_titan.training.reference_search")
    except ModuleNotFoundError:
        pytest.fail("Falta la búsqueda científica recuperable")


def setup(tmp_path):
    manifest = training_corpus(tmp_path / "data")
    config = tmp_path / "search.json"
    config.write_text(
        json.dumps(
            dict(
                schema_version=1,
                scope="development_snapshot",
                arms=["US"],
                models=["rnn"],
                search_seed=42,
                finalist_seeds=[42, 43, 44],
                max_epochs=2,
                patience=1,
                min_delta=0.0,
                batch_size=5,
                context_sessions=64,
                checkpoint_seconds=900,
                posttraining_epochs=1,
                posttraining_learning_rate=0.0001,
                posttraining_losses=["mae", "mse"],
                pooled_weightings=["natural", "balanced_markets"],
                final_test_opened=False,
            )
        )
    )
    return config, manifest


def test_search_resumes_without_retraining_confirmed_cases_and_reuses_seed42(tmp_path, monkeypatch):
    config, manifest = setup(tmp_path)
    engine = module()
    real = engine.run_reference_case
    calls = []

    def interrupt(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(result["identity"]["case"])
        kwargs["stop"].request_stop()
        return result

    monkeypatch.setattr(engine, "run_reference_case", interrupt)
    output = tmp_path / "study"
    paused = engine.run_search(config, manifest, output)
    assert paused["status"] == "paused" and paused["completed_runs"] == 1
    path = output / paused["runs"][0]["path"] / "run.json"
    digest, modified = sha256(path), path.stat().st_mtime_ns

    def observe_parent(*args, **kwargs):
        if kwargs.get("initialize_from") is not None:
            progress = json.loads((output / "summary.json").read_text())
            assert progress["selected"]
            assert any(f["seed"] == args[2]["seed"] for f in progress["finalists"])
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "run_reference_case", observe_parent)
    complete = engine.run_search(config, manifest, output, resume=True)
    assert (
        complete["status"] == "completed"
        and complete["completed_runs"] == complete["planned_runs"] == 20
    )
    assert len(calls) == 1
    assert sha256(path) == digest and path.stat().st_mtime_ns == modified
    assert len(complete["runs"]) == 20 and len(complete["finalists"]) == 3
    assert sum(item["stage"] == "search" for item in complete["runs"]) == 12
    assert sum(item["stage"] == "finalist" for item in complete["runs"]) == 2
    assert sum(item["stage"] == "posttraining" for item in complete["runs"]) == 6
    by_id = {item["id"]: item for item in complete["runs"]}
    finalist42 = next(f for f in complete["finalists"] if f["seed"] == 42)
    assert by_id[finalist42["run_id"]]["stage"] == "search"
    winner = complete["selected"]["US-natural/rnn"]
    scores = [item["session_mae"] for item in complete["runs"] if item["stage"] == "search"]
    assert by_id[winner]["session_mae"] == min(scores)
    for item in complete["runs"]:
        if item["stage"] == "posttraining":
            assert item["parent"] in {f["run_id"] for f in complete["finalists"]}


def test_full_scope_does_not_accept_a_development_snapshot(tmp_path):
    config, manifest = setup(tmp_path)
    plan = json.loads(config.read_text())
    plan["scope"] = "full_corpus"
    config.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="alcance|completo|edición"):
        module().run_search(config, manifest, tmp_path / "study")
    assert not (tmp_path / "study").exists()


def test_search_seed_must_have_a_declared_finalist(tmp_path):
    config, manifest = setup(tmp_path)
    plan = json.loads(config.read_text())
    plan["finalist_seeds"] = [43, 44]
    config.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="semilla|finalista"):
        module().run_search(config, manifest, tmp_path / "study")


def test_declared_context_must_match_the_frozen_population(tmp_path):
    config, manifest = setup(tmp_path)
    plan = json.loads(config.read_text())
    plan["context_sessions"] = 32
    config.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="contexto"):
        module().run_search(config, manifest, tmp_path / "study")
    assert not (tmp_path / "study").exists()


def pause_after_case(tmp_path, monkeypatch):
    config, manifest = setup(tmp_path)
    engine, output = module(), tmp_path / "study"
    real = engine.run_reference_case

    def interrupt(*args, **kwargs):
        result = real(*args, **kwargs)
        kwargs["stop"].request_stop()
        return result

    monkeypatch.setattr(engine, "run_reference_case", interrupt)
    engine.run_search(config, manifest, output)
    monkeypatch.setattr(engine, "run_reference_case", real)
    return config, manifest, output


def test_resume_publishes_running_status_before_applying_new_updates(tmp_path, monkeypatch):
    config, manifest, output = pause_after_case(tmp_path, monkeypatch)
    engine, statuses = module(), []
    real = engine.run_reference_case

    def inspect_status(*args, **kwargs):
        statuses.append(json.loads((output / "summary.json").read_text())["status"])
        result = real(*args, **kwargs)
        kwargs["stop"].request_stop()
        return result

    monkeypatch.setattr(engine, "run_reference_case", inspect_status)
    result = engine.run_search(config, manifest, output, resume=True)
    assert result["status"] == "paused" and result["completed_runs"] == 2
    assert statuses == ["running"]


@pytest.mark.parametrize(
    "changes", [dict(planned_runs=1), dict(scope="full_corpus"), dict(final_test_opened=True)]
)
def test_resume_rejects_altered_summary_contract(tmp_path, monkeypatch, changes):
    config, manifest, output = pause_after_case(tmp_path, monkeypatch)
    path = output / "summary.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))
    with pytest.raises(ValueError, match="contrato|identidad|alcance"):
        module().run_search(config, manifest, output, resume=True)


def test_changed_source_between_view_read_and_hash_is_not_adopted(tmp_path, monkeypatch):
    config, manifest = setup(tmp_path)
    engine, original = module(), module().campaign_views

    def change_source(path, arms):
        views = original(path, arms)
        source = json.loads(path.read_text())
        source["context_sessions"] = 32
        path.write_text(json.dumps(source))
        return views

    monkeypatch.setattr(engine, "campaign_views", change_source)
    with pytest.raises(ValueError, match="origen|cambiad|edición"):
        engine.run_search(config, manifest, tmp_path / "study")
    assert not (tmp_path / "study").exists()


def test_failed_output_verification_does_not_count_as_a_completed_trial(tmp_path, monkeypatch):
    config, manifest = setup(tmp_path)
    engine, real = module(), module().run_reference_case

    def corrupt(*args, **kwargs):
        report = real(*args, **kwargs)
        (args[1] / report["predictions"]["validation"]["path"]).write_bytes(b"Artefacto alterado")
        return report

    monkeypatch.setattr(engine, "run_reference_case", corrupt)
    output = tmp_path / "study"
    with pytest.raises(ValueError, match="cambiado"):
        engine.run_search(config, manifest, output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["completed_runs"] == 0
    assert summary["runs"][0]["status"] == "failed"
