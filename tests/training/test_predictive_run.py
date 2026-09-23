"""Tres controles offline con población completa, evaluación y recuperación CUDA."""

import importlib

import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.training.checkpoints import StopRequest, load_training_state
from tests.training.test_predictive_parents import setup

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("duckdb", "gymnasium")),
    reason="Requiere los extras data y reinforcement",
)


def prepared(tmp_path):
    from mars_titan.training.predictive_parents import prepare_parent_cache

    ordered, parent, _ = setup(tmp_path)
    cache = tmp_path / "cache"
    prepare_parent_cache(ordered, parent, cache)
    return ordered, cache / "manifest.json"


def case(mode="reinforce"):
    return dict(mode=mode, epochs=2, seed=42, learning_rate=0.001, weight_decay=0.01, clip_norm=1.0)


def module():
    return importlib.import_module("mars_titan.training.predictive_run")


@pytest.mark.parametrize("mode", ["reinforce", "expected", "mae"])
def test_predictive_controls_visit_every_row_and_record_all_readouts(tmp_path, mode):
    ordered, cache = prepared(tmp_path)
    output = tmp_path / "run"
    result = module().run_predictive_case(ordered, cache, output, case(mode), batch_size=5)
    assert result["status"] == "completed" and result["device"] == "cuda:0"
    assert result["global_step"] == 6 and [r["train"]["samples"] for r in result["epochs"]] == [
        12,
        12,
    ]
    assert result["identity"]["fit_timing"] == "offline_after_training_cutoff"
    assert result["final_test_opened"] is False and result["parent_frozen"] is True
    assert result["trainable_parameters"] == 327
    for partition, count in (("train", 12), ("validation", 6)):
        record = result["predictions"][partition]
        table = pq.read_table(output / record["path"])
        assert len(table) == count and len(set(table["sample_id"].to_pylist())) == count
        assert {
            "center",
            "nearest",
            "prediction",
            "parent",
            "policy_entropy",
            "extreme_action_mass",
        } <= set(table.column_names)
        for readout in ("center", "nearest", "median", "parent", "zero"):
            assert record["metrics"][readout]["session_count"] == count // 2
        assert record["metrics"]["policy"]["samples"] == count
    assert (
        module().run_predictive_case(ordered, cache, output, case(mode), batch_size=5, resume=True)
        == result
    )


def test_resume_preserves_sampling_optimizer_and_next_update_exactly(tmp_path, monkeypatch):
    engine = module()
    ordered, cache = prepared(tmp_path)
    reference = engine.run_predictive_case(ordered, cache, tmp_path / "full", case(), batch_size=5)
    stop = StopRequest()
    real_save = engine.save_training_state

    def interrupted(directory, state, **kwargs):
        path = real_save(directory, state, **kwargs)
        if state["global_step"] == 2:
            stop.request_stop()
        return path

    monkeypatch.setattr(engine, "save_training_state", interrupted)
    first = engine.run_predictive_case(
        ordered, cache, tmp_path / "resumed", case(), batch_size=5, checkpoint_steps=1, stop=stop
    )
    assert first["status"] == "paused"
    monkeypatch.setattr(engine, "save_training_state", real_save)
    resumed = engine.run_predictive_case(
        ordered, cache, tmp_path / "resumed", case(), batch_size=5, checkpoint_steps=1, resume=True
    )
    assert resumed["status"] == "completed" and resumed["global_step"] == reference["global_step"]
    one = load_training_state(
        tmp_path / "full/checkpoints", expected_identity=reference["identity"]
    )
    two = load_training_state(
        tmp_path / "resumed/checkpoints", expected_identity=resumed["identity"]
    )
    assert all(torch.equal(value, two["model"][key]) for key, value in one["model"].items())
    assert torch.equal(one["sampling_rng"], two["sampling_rng"])
    for partition in ("train", "validation"):
        assert pq.read_table(
            tmp_path / "full" / reference["predictions"][partition]["path"]
        ).equals(pq.read_table(tmp_path / "resumed" / resumed["predictions"][partition]["path"]))


def test_checkpoint_is_confirmed_before_validation_can_be_interrupted(tmp_path, monkeypatch):
    engine = module()
    ordered, cache = prepared(tmp_path)
    stop = StopRequest()
    real = engine.evaluate_predictive

    def pause(*args, **kwargs):
        stop.request_stop()
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "evaluate_predictive", pause)
    report = engine.run_predictive_case(
        ordered, cache, tmp_path / "run", case(), batch_size=5, stop=stop
    )
    assert report["status"] == "paused"
    state = load_training_state(tmp_path / "run/checkpoints", expected_identity=report["identity"])
    assert state["global_step"] == 3 and state["statistics"]["samples"] == 12


def test_resume_rejects_changed_matmul_policy(tmp_path):
    engine = module()
    ordered, cache = prepared(tmp_path)
    output = tmp_path / "run"
    engine.run_predictive_case(ordered, cache, output, case(), batch_size=5)
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("medium" if previous != "medium" else "highest")
        with pytest.raises(ValueError, match="código|ejecución|identidad"):
            engine.run_predictive_case(ordered, cache, output, case(), batch_size=5, resume=True)
    finally:
        torch.set_float32_matmul_precision(previous)


def test_simultaneous_resume_cannot_write_the_same_training_run(tmp_path, monkeypatch):
    engine = module()
    ordered, cache = prepared(tmp_path)
    output = tmp_path / "run"
    real_save = engine.save_training_state
    checked = False

    def another_writer(directory, state, **kwargs):
        nonlocal checked
        path = real_save(directory, state, **kwargs)
        if state["global_step"] == 1 and not checked:
            checked = True
            with pytest.raises(BlockingIOError):
                engine.run_predictive_case(
                    ordered, cache, output, case(), batch_size=5, resume=True
                )
        return path

    monkeypatch.setattr(engine, "save_training_state", another_writer)
    assert (
        engine.run_predictive_case(
            ordered, cache, output, case(), batch_size=5, checkpoint_steps=1
        )["status"]
        == "completed"
    )
    assert checked


def test_failed_cuda_initialization_can_resume_same_preparation(tmp_path, monkeypatch):
    engine = module()
    ordered, cache = prepared(tmp_path)
    output = tmp_path / "run"
    real = engine.require_cuda

    def unavailable():
        raise RuntimeError("CUDA temporalmente no disponible")

    monkeypatch.setattr(engine, "require_cuda", unavailable)
    with pytest.raises(RuntimeError):
        engine.run_predictive_case(ordered, cache, output, case(), batch_size=5)
    assert (output / "initialization.json").is_file()
    monkeypatch.setattr(engine, "require_cuda", real)
    assert (
        engine.run_predictive_case(ordered, cache, output, case(), batch_size=5, resume=True)[
            "status"
        ]
        == "completed"
    )


def test_precomputed_training_normalization_is_reused_for_paired_controls(tmp_path, monkeypatch):
    import json

    from mars_titan.training.predictive_inputs import PredictiveDataset, fit_standardizer

    engine = module()
    ordered, cache = prepared(tmp_path)
    stats_path = tmp_path / "normalization.json"
    with PredictiveDataset(ordered, cache) as dataset:
        stats_path.write_text(json.dumps(fit_standardizer(dataset, batch_size=5)))

    def no_repetition(*args, **kwargs):
        pytest.fail("La estadística de entrenamiento confirmada no debe recalcularse")

    monkeypatch.setattr(engine, "fit_standardizer", no_repetition)
    for mode in ("reinforce", "expected", "mae"):
        report = engine.run_predictive_case(
            ordered, cache, tmp_path / mode, case(mode), batch_size=5, normalization=stats_path
        )
        assert report["status"] == "completed"


def test_failed_initial_checkpoint_can_restart_without_changing_predictions(tmp_path, monkeypatch):
    engine = module()
    ordered, cache = prepared(tmp_path)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    reference = engine.run_predictive_case(ordered, cache, full, case(), batch_size=5)
    real_save = engine.save_training_state

    def unavailable(*args, **kwargs):
        raise OSError(28, "No hay espacio para confirmar el primer estado")

    monkeypatch.setattr(engine, "save_training_state", unavailable)
    with pytest.raises(OSError, match="primer estado"):
        engine.run_predictive_case(ordered, cache, resumed, case(), batch_size=5)
    assert not (resumed / "checkpoints/latest.json").exists()
    monkeypatch.setattr(engine, "save_training_state", real_save)
    result = engine.run_predictive_case(ordered, cache, resumed, case(), batch_size=5, resume=True)
    assert result["status"] == "completed"
    for partition in ("train", "validation"):
        assert pq.read_table(full / reference["predictions"][partition]["path"]).equals(
            pq.read_table(resumed / result["predictions"][partition]["path"])
        )


def test_missing_confirmed_initial_checkpoint_cannot_silently_restart(tmp_path):
    engine = module()
    ordered, cache = prepared(tmp_path)
    output, stop = tmp_path / "run", StopRequest()
    stop.request_stop()
    report = engine.run_predictive_case(ordered, cache, output, case(), batch_size=5, stop=stop)
    assert report["status"] == "paused" and report["global_step"] == 0
    (output / "checkpoints/latest.json").unlink()
    with pytest.raises(ValueError, match="confirmado"):
        engine.run_predictive_case(ordered, cache, output, case(), batch_size=5, resume=True)
