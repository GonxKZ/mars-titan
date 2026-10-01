"""Recuperación exacta de objetivos estocásticos y continuaciones neuronales."""

import copy

import numpy as np
import pytest
import torch
from test_inputs import dataset

from mars_titan.data.cohort_files import read_manifest
from mars_titan.environments.actions import ActionGrid
from mars_titan.models.baselines.multimodal import MultimodalReference
from mars_titan.posttraining.inputs import fit_normalization
from mars_titan.posttraining.parents import FrozenParent
from mars_titan.posttraining.run import run_case
from mars_titan.training.checkpoints import StopRequest, load_training_state


def options(mode):
    return dict(
        mode=mode,
        condition="real_synthetic",
        seed=7,
        epochs=2,
        learning_rate=0.0001,
        weight_decay=0.01,
        clip_norm=1.0,
        beta=0.1,
        behavior_epsilon=0.001,
        auxiliary_samples=4,
    )


def grid(data):
    targets = np.concatenate([data.train(i)["target"] for i in range(len(data.train))])
    return ActionGrid.fit(targets, source_sha256=data.train.source_sha256, partition="train")


def state(folder):
    identity = read_manifest(folder / "run.json")[0]["identity"]
    return load_training_state(folder / "checkpoints", expected_identity=identity)


@pytest.mark.parametrize(
    "mode", ["reinforce", "expected", "mae", "klpo_full", "klpo_mc", "klpo_exact"]
)
def test_resume_recovers_exact_weights_optimizer_rng_and_budget(tmp_path, mode):
    data = dataset(tmp_path)
    scale = fit_normalization(data, batch_size=2)
    kwargs = dict(
        dataset=data,
        case=options(mode),
        grid=grid(data),
        normalization=scale,
        batch_size=2,
        device="cpu",
        diagnostic=True,
    )
    complete = run_case(output=tmp_path / "whole", **kwargs)
    paused = run_case(output=tmp_path / "split", max_updates=9, **kwargs)
    assert paused["status"] == "paused"
    assert state(tmp_path / "split")["cursor"]["consumed"] == 16
    resumed = run_case(output=tmp_path / "split", resume=True, **kwargs)
    assert complete["status"] == resumed["status"] == "completed"
    first, second = state(tmp_path / "whole"), state(tmp_path / "split")
    assert first["global_step"] == second["global_step"] == 144
    assert first["history"] == second["history"]
    for key, value in first["model"].items():
        assert torch.equal(value, second["model"][key]), key
    for key in ("actions", "auxiliaries"):
        assert torch.equal(first["generators"][key], second["generators"][key])
    for key, value in first["optimizer"]["state"].items():
        for name, tensor in value.items():
            assert torch.equal(tensor, second["optimizer"]["state"][key][name])
    assert (
        resumed["predictions"]["validation"]["metrics"]
        == complete["predictions"]["validation"]["metrics"]
    )
    assert all(epoch["train"]["real_rows"] == 98 for epoch in resumed["epochs"])
    assert all(epoch["train"]["extra_rows"] == 28 for epoch in resumed["epochs"])
    assert all(epoch["validation"]["samples"] == 32 for epoch in resumed["epochs"])
    data.parent.close()


@pytest.mark.parametrize("mode", ["neural_mae", "neural_mse"])
def test_neural_continuation_keeps_parent_frozen_and_resumes_dropout(tmp_path, mode):
    data = dataset(tmp_path)
    torch.manual_seed(42)
    model = MultimodalReference(
        "gru",
        {key: shape[-1] for key, shape in data.shapes.items()},
        context=4,
        hidden_size=32,
        layers=1,
        dropout=0.1,
    )
    parent = FrozenParent(model, dict(model="gru", checkpoint_sha256="a" * 64), data.shapes, "cpu")
    data.parent.predictor = parent.predict
    original = copy.deepcopy(parent.model.state_dict())
    kwargs = dict(
        dataset=data,
        case={**options(mode), "epochs": 1},
        grid=grid(data),
        normalization=fit_normalization(data, batch_size=4),
        parent=parent,
        batch_size=4,
        device="cpu",
        diagnostic=True,
    )
    run_case(output=tmp_path / "whole", **kwargs)
    run_case(output=tmp_path / "split", max_updates=7, **kwargs)
    run_case(output=tmp_path / "split", resume=True, **kwargs)
    first, second = state(tmp_path / "whole"), state(tmp_path / "split")
    assert all(torch.equal(v, second["model"][k]) for k, v in first["model"].items())
    assert all(torch.equal(v, parent.model.state_dict()[k]) for k, v in original.items())
    assert any(not torch.equal(v, original[k]) for k, v in first["model"].items())
    data.parent.close()


def test_stop_and_identity_changes_never_skip_unconfirmed_work(tmp_path):
    data = dataset(tmp_path)
    kwargs = dict(
        dataset=data,
        case=options("klpo_mc"),
        grid=grid(data),
        normalization=fit_normalization(data),
        batch_size=2,
        device="cpu",
        diagnostic=True,
    )
    stop = StopRequest()
    stop.request_stop()
    report = run_case(output=tmp_path / "case", stop=stop, **kwargs)
    assert report["global_step"] == 0
    assert report["status"] == "paused"
    changed = dict(kwargs, case={**kwargs["case"], "condition": "real_resampled"})
    with pytest.raises(ValueError, match="identidad"):
        run_case(output=tmp_path / "case", resume=True, **changed)
    changed = dict(kwargs, normalization={**kwargs["normalization"], "fit_partition": "validation"})
    with pytest.raises(ValueError, match="normalización"):
        run_case(output=tmp_path / "other", **changed)
    data.parent.close()


def test_missing_confirmed_checkpoint_cannot_restart_a_run(tmp_path):
    data = dataset(tmp_path)
    kwargs = dict(
        dataset=data,
        case=options("reinforce"),
        grid=grid(data),
        normalization=fit_normalization(data),
        batch_size=2,
        device="cpu",
        diagnostic=True,
    )
    output = tmp_path / "case"
    run_case(output=output, max_updates=3, **kwargs)
    index = output / "checkpoints/latest.json"
    index.rename(index.with_name("lost.json"))
    with pytest.raises(ValueError, match="checkpoint"):
        run_case(output=output, resume=True, **kwargs)
    data.parent.close()
