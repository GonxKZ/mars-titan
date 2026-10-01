"""Configuraciones y recuperación de KLPO con los controles predictivos comunes."""

import json

import pytest
import torch

from mars_titan.training import predictive_run, predictive_study
from mars_titan.training.checkpoints import StopRequest, load_training_state
from tests.training.test_predictive_run import case, prepared


def klpo_case(mode="klpo_mc"):
    return case(mode) | dict(beta=0.1, behavior_epsilon=1e-6, auxiliary_samples=128)


def config():
    return dict(
        schema_version=2,
        modes=["reinforce", "expected", "mae", "klpo_full", "klpo_mc", "klpo_exact"],
        seeds=[42, 43, 44],
        betas=[0.03, 0.1, 0.3],
        behavior_epsilon=1e-6,
        auxiliary_samples=128,
        epochs=5,
        learning_rate=0.0001,
        weight_decay=0.01,
        clip_norm=1.0,
        batch_size=256,
        checkpoint_seconds=60,
        final_test_opened=False,
    )


def test_configuration_pairs_klpo_estimators_and_original_controls(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config()))
    _, cases, _ = predictive_study._configuration(path)
    assert len(cases) == 36 and len({item["path"] for item in cases}) == 36
    for beta in (0.03, 0.1, 0.3):
        matching = [item["case"] for item in cases if item["case"].get("beta") == beta]
        assert len(matching) == 9
        assert {item["mode"] for item in matching} == {"klpo_full", "klpo_mc", "klpo_exact"}
        assert {item["seed"] for item in matching} == {42, 43, 44}
    assert all(item["case"]["epochs"] == 5 for item in cases)


@pytest.mark.parametrize(
    "key,value",
    [
        ("beta", 0),
        ("beta", float("nan")),
        ("behavior_epsilon", 0),
        ("behavior_epsilon", 1),
        ("auxiliary_samples", 0),
        ("auxiliary_samples", True),
    ],
)
def test_klpo_options_reject_invalid_sampler_or_regularization(key, value, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(ValueError):
        predictive_run._options(klpo_case() | {key: value}, 256, 0, 60)


@pytest.mark.parametrize("mode", ["klpo_full", "klpo_mc", "klpo_exact"])
def test_klpo_cuda_uses_complete_population_and_records_sampler(tmp_path, mode):
    ordered, cache = prepared(tmp_path)
    result = predictive_run.run_predictive_case(
        ordered, cache, tmp_path / "run", klpo_case(mode), batch_size=5
    )
    assert result["status"] == "completed" and result["device"] == "cuda:0"
    assert result["global_step"] == 6
    assert [epoch["train"]["samples"] for epoch in result["epochs"]] == [12, 12]
    assert result["identity"]["sampler"]["epsilon"] == 1e-6
    assert result["identity"]["sampler"]["fixed_parent"] is True
    assert result["final_test_opened"] is False
    for partition in ("train", "validation"):
        metrics = result["predictions"][partition]["metrics"]["policy"]
        assert metrics["mean_behavior_kl_nats"] >= -1e-12
        assert metrics["mean_klpo_variance"] >= 0


def test_klpo_cuda_resume_restores_both_sampling_streams(tmp_path, monkeypatch):
    ordered, cache = prepared(tmp_path)
    run, stop = predictive_run.run_predictive_case, StopRequest()
    full = run(ordered, cache, tmp_path / "full", klpo_case(), batch_size=5)
    save = predictive_run.save_training_state

    def interrupted(directory, state, **kwargs):
        path = save(directory, state, **kwargs)
        if state["global_step"] == 2:
            stop.request_stop()
        return path

    monkeypatch.setattr(predictive_run, "save_training_state", interrupted)
    first = run(
        ordered,
        cache,
        tmp_path / "resumed",
        klpo_case(),
        batch_size=5,
        checkpoint_steps=1,
        stop=stop,
    )
    assert first["status"] == "paused"
    monkeypatch.setattr(predictive_run, "save_training_state", save)
    resumed = run(ordered, cache, tmp_path / "resumed", klpo_case(), batch_size=5, resume=True)
    assert resumed["status"] == "completed"
    states = [
        load_training_state(tmp_path / name / "checkpoints", expected_identity=report["identity"])
        for name, report in (("full", full), ("resumed", resumed))
    ]
    for key in ("sampling_rng", "auxiliary_rng"):
        assert torch.equal(states[0][key], states[1][key])
    for key, value in states[0]["model"].items():
        assert torch.equal(value, states[1]["model"][key])
    assert full["epochs"] == resumed["epochs"]
