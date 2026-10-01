"""Recibos y recuperación de referencias sobre un corpus CUDA de 18 filas."""

import importlib

import pytest
import torch

from mars_titan.training.checkpoints import StopRequest, load_training_state
from tests.training.test_reference_run import case, training_corpus


def configured():
    return dict(
        case(),
        epochs=6,
        selection=dict(
            metric="session_mae",
            minimum_epochs=2,
            patience=2,
            min_delta=1e-5,
        ),
    )


def scores(monkeypatch, values, stop=None):
    engine = importlib.import_module("mars_titan.training.reference_run")
    original = engine._evaluate
    remaining = iter(values)

    def evaluate(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("destination") is None:
            result["session_mae"] = next(remaining)
        elif stop is not None:
            stop.request_stop()
        return result

    monkeypatch.setattr(engine, "_evaluate", evaluate)


@pytest.mark.parametrize(
    "curve,reason,improved,best",
    [
        ((0.5, 0.6, 0.6, 0.6), "validation_plateau", False, 1),
        ((0.6, 0.5, 0.4, 0.3, 0.2, 0.1), "budget_exhausted", True, 6),
    ],
)
def test_reference_receipt_identifies_actual_stop(
    tmp_path, monkeypatch, curve, reason, improved, best
):
    engine = importlib.import_module("mars_titan.training.reference_run")
    manifest = training_corpus(tmp_path / "data")
    scores(monkeypatch, curve)
    report = engine.run_reference_case(manifest, tmp_path / "run", configured(), batch_size=5)
    assert report["status"] == "completed"
    assert report["stop_reason"] == reason
    assert report["last_epoch_improved"] is improved
    assert report["selection"]["best_epoch"] == best
    assert len(report["epochs"]) == len(curve)


def test_reference_pause_at_selected_prediction_recovers_latest_weights(tmp_path, monkeypatch):
    engine = importlib.import_module("mars_titan.training.reference_run")
    manifest = training_corpus(tmp_path / "data")
    curve = (0.5, 0.6, 0.6, 0.6)
    with monkeypatch.context() as patch:
        scores(patch, curve)
        whole = engine.run_reference_case(manifest, tmp_path / "whole", configured(), batch_size=5)
    stop = StopRequest()
    with monkeypatch.context() as patch:
        scores(patch, curve, stop)
        paused = engine.run_reference_case(
            manifest, tmp_path / "split", configured(), batch_size=5, stop=stop
        )
    assert paused["status"] == "paused"
    assert "stop_reason" not in paused
    resumed = engine.run_reference_case(
        manifest, tmp_path / "split", configured(), batch_size=5, resume=True
    )
    assert resumed["stop_reason"] == "validation_plateau"
    for partition in ("train", "validation"):
        assert (
            whole["predictions"][partition]["sha256"] == resumed["predictions"][partition]["sha256"]
        )
        assert (
            whole["predictions"][partition]["metrics"]["session_mae"]
            == resumed["predictions"][partition]["metrics"]["session_mae"]
        )
    first = load_training_state(tmp_path / "whole/checkpoints", expected_identity=whole["identity"])
    second = load_training_state(
        tmp_path / "split/checkpoints", expected_identity=resumed["identity"]
    )
    assert first["selection"] == second["selection"]
    assert all(torch.equal(v, second["model"][k]) for k, v in first["model"].items())
