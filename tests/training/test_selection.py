"""Selección explícita por error de sesión y recuperación de la época elegida."""

import importlib
import json

import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.storage import sha256
from mars_titan.training.checkpoints import load_training_state, save_training_state
from tests.training.test_reference_run import case, training_corpus


def advance(previous, score, epoch, **changes):
    try:
        module = importlib.import_module("mars_titan.training.selection")
    except ModuleNotFoundError:
        pytest.fail("Falta la selección por sesiones completas")
    return module.advance_selection(
        previous,
        score,
        epoch,
        {"metric": "session_mae", "patience": 2, "min_delta": 0.0, **changes},
    )


def test_patience_counts_complete_epochs_and_keeps_first_tie():
    state = None
    for epoch, value in enumerate((0.4, 0.2, 0.2, 0.3), 1):
        state = advance(state, value, epoch)
        assert state["should_stop"] is (epoch == 4)
    assert state["best_epoch"] == 2
    assert state["best_score"] == 0.2
    assert state["stale_epochs"] == 2


def test_invalid_or_repeated_epoch_cannot_change_selection():
    state = advance(None, 0.2, 1)
    for score, epoch in ((float("nan"), 2), (-1.0, 2), (0.1, 1), (0.1, 3)):
        with pytest.raises(ValueError):
            advance(state, score, epoch)
    assert state["best_epoch"] == 1


def test_minimum_improvement_is_measured_against_the_last_accepted_best():
    state = advance(None, 0.2, 1, min_delta=0.01)
    state = advance(state, 0.195, 2, min_delta=0.01)
    assert state["best_epoch"] == 1 and state["stale_epochs"] == 1
    state = advance(state, 0.18, 3, min_delta=0.01)
    assert state["best_epoch"] == 3 and state["stale_epochs"] == 0


def test_best_checkpoint_is_loaded_without_using_latest_as_a_substitute(tmp_path):
    identity = dict(experiment="selection_test")
    first = save_training_state(
        tmp_path, dict(global_step=1, weight=torch.tensor([1.0])), identity=identity, best=True
    )
    save_training_state(
        tmp_path, dict(global_step=2, weight=torch.tensor([2.0])), identity=identity
    )
    selected = load_training_state(tmp_path, expected_identity=identity, selection="best")
    assert selected["global_step"] == 1
    assert load_training_state(tmp_path, expected_identity=identity)["global_step"] == 2
    first.write_bytes(b"Estado elegido alterado")
    with pytest.raises(ValueError, match="seleccionado|íntegro|mejor"):
        load_training_state(tmp_path, expected_identity=identity, selection="best")


def test_explicit_checkpoint_identity_cannot_fall_back_after_metadata_corruption(tmp_path):
    identity = dict(experiment="strict_identity")
    save_training_state(tmp_path, dict(global_step=1), identity=identity)
    latest = save_training_state(tmp_path, dict(global_step=2), identity=identity)
    path = tmp_path / "latest.json"
    index = json.loads(path.read_text())
    index["latest"][0]["size"] += 1
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="íntegro|seleccionado|identidad"):
        load_training_state(tmp_path, expected_identity=identity, expected_sha256=sha256(latest))


def test_strict_load_never_substitutes_a_previous_state_after_a_read_race(tmp_path, monkeypatch):
    engine = importlib.import_module("mars_titan.training.checkpoints")
    identity = dict(experiment="integrity_race")
    save_training_state(tmp_path, dict(global_step=1), identity=identity)
    latest = save_training_state(tmp_path, dict(global_step=2), identity=identity)
    digest, original = sha256(latest), engine._intact
    changed = False

    def corrupt_after_check(directory, record):
        nonlocal changed
        result = original(directory, record)
        if record["global_step"] == 2 and not changed:
            latest.write_bytes(b"Cambio durante la lectura")
            changed = True
        return result

    monkeypatch.setattr(engine, "_intact", corrupt_after_check)
    with pytest.raises(ValueError, match="íntegro|seleccionado"):
        load_training_state(tmp_path, expected_identity=identity, expected_sha256=digest)


def selected_case():
    return {
        **case(),
        "epochs": 6,
        "architecture": dict(hidden_size=64, layers=2, dropout=0.1),
        "selection": dict(metric="session_mae", patience=1, min_delta=1.0),
    }


def test_early_stop_preserves_latest_state_but_predicts_from_selected_epoch(tmp_path):
    engine = importlib.import_module("mars_titan.training.reference_run")
    manifest = training_corpus(tmp_path / "data")
    options = selected_case()
    baseline = {k: v for k, v in options.items() if k != "selection"}
    baseline["epochs"] = 1
    engine.run_reference_case(manifest, tmp_path / "one_epoch", baseline, batch_size=5)
    report = engine.run_reference_case(manifest, tmp_path / "selected", options, batch_size=5)
    assert report["status"] == "completed" and report["stopped_early"] is True
    assert len(report["epochs"]) == 2 and report["global_step"] == 6
    assert report["selection"]["best_epoch"] == 1
    assert report["selection"]["best_score"] < 1.0
    latest = load_training_state(
        tmp_path / "selected/checkpoints", expected_identity=report["identity"]
    )
    best = load_training_state(
        tmp_path / "selected/checkpoints", expected_identity=report["identity"], selection="best"
    )
    assert latest["global_step"] == 6 and best["global_step"] == 3
    assert any(not torch.equal(v, best["model"][k]) for k, v in latest["model"].items())
    assert pq.read_table(tmp_path / "one_epoch/validation-predictions.parquet").equals(
        pq.read_table(tmp_path / "selected/validation-predictions.parquet")
    )
    child = engine.run_reference_case(
        manifest, tmp_path / "child", baseline, batch_size=5, initialize_from=tmp_path / "selected"
    )
    assert child["initialization"]["parent_checkpoint_sha256"] == report["checkpoint"]["sha256"]


def test_pause_during_selected_prediction_keeps_coherent_latest_optimizer_state(
    tmp_path, monkeypatch
):
    from mars_titan.training.checkpoints import StopRequest

    engine = importlib.import_module("mars_titan.training.reference_run")
    manifest = training_corpus(tmp_path / "data")
    options = selected_case()
    continuous = engine.run_reference_case(manifest, tmp_path / "continuous", options, batch_size=5)
    stop = StopRequest()
    real = engine._evaluate

    def interrupt(*args, **kwargs):
        if kwargs.get("destination") is not None:
            stop.request_stop()
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "_evaluate", interrupt)
    paused = engine.run_reference_case(
        manifest, tmp_path / "paused", options, batch_size=5, stop=stop
    )
    assert paused["status"] == "paused" and paused["global_step"] == 6
    a = load_training_state(
        tmp_path / "continuous/checkpoints", expected_identity=continuous["identity"]
    )
    b = load_training_state(tmp_path / "paused/checkpoints", expected_identity=paused["identity"])
    assert all(torch.equal(value, b["model"][key]) for key, value in a["model"].items())
    monkeypatch.setattr(engine, "_evaluate", real)
    completed = engine.run_reference_case(
        manifest, tmp_path / "paused", options, batch_size=5, resume=True
    )
    assert completed["global_step"] == 6 and len(completed["epochs"]) == 2
    assert pq.read_table(tmp_path / "continuous/validation-predictions.parquet").equals(
        pq.read_table(tmp_path / "paused/validation-predictions.parquet")
    )


def test_initial_checkpoint_failure_is_reported_as_failed(tmp_path, monkeypatch):
    engine = importlib.import_module("mars_titan.training.reference_run")
    manifest = training_corpus(tmp_path / "data")

    def fail(*args, **kwargs):
        raise OSError("Fallo de escritura simulado")

    monkeypatch.setattr(engine, "save_training_state", fail)
    with pytest.raises(OSError, match="simulado"):
        engine.run_reference_case(manifest, tmp_path / "run", selected_case(), batch_size=5)
    report = json.loads((tmp_path / "run/run.json").read_text())
    assert report["status"] == "failed" and report["error_type"] == "OSError"


def test_resume_rejects_best_pointer_redirected_to_another_valid_epoch(tmp_path, monkeypatch):
    from mars_titan.training.checkpoints import StopRequest

    engine = importlib.import_module("mars_titan.training.reference_run")
    manifest = training_corpus(tmp_path / "data")
    stop, real = StopRequest(), engine._evaluate

    def interrupt(*args, **kwargs):
        if kwargs.get("destination") is not None:
            stop.request_stop()
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "_evaluate", interrupt)
    output = tmp_path / "run"
    engine.run_reference_case(manifest, output, selected_case(), batch_size=5, stop=stop)
    path = output / "checkpoints/latest.json"
    index = json.loads(path.read_text())
    assert index["best"]["global_step"] != index["latest"][0]["global_step"]
    index["best"] = index["latest"][0]
    path.write_text(json.dumps(index))
    monkeypatch.setattr(engine, "_evaluate", real)
    with pytest.raises(ValueError, match="selección|época|seleccionad"):
        engine.run_reference_case(manifest, output, selected_case(), batch_size=5, resume=True)
