"""Selección desde el padre, parada recuperable y retención acotada en CPU."""

import importlib

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from test_inputs import dataset
from test_run import grid, options, state

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.models.baselines.multimodal import MultimodalReference
from mars_titan.posttraining.inputs import fit_normalization
from mars_titan.posttraining.parents import FrozenParent
from mars_titan.posttraining.run import run_case
from mars_titan.posttraining.selection import select_epoch, selection_policy
from mars_titan.training.checkpoints import StopRequest, load_training_state


def configuration(data, *, epochs=5, patience=None, min_delta=0.0, condition="real"):
    case = dict(options("klpo_mc"), condition=condition, epochs=epochs)
    case["selection"] = dict(
        version=2, metric="session_mae", patience=patience, min_delta=min_delta
    )
    return dict(
        dataset=data,
        case=case,
        grid=grid(data),
        normalization=fit_normalization(data),
        batch_size=32,
        device="cpu",
        diagnostic=True,
    )


def controlled_scores(monkeypatch, scores, *, stop_final=None):
    module = importlib.import_module("mars_titan.posttraining.run")
    evaluate = importlib.import_module("mars_titan.posttraining.evaluation").evaluate
    remaining = iter(scores)

    def assess(*args, **kwargs):
        if kwargs.get("destination") is not None and stop_final is not None:
            stop_final.request_stop()
        result = evaluate(*args, **kwargs)
        # La curva controla la selección, el modelo y los artefactos se ejecutan realmente.
        if kwargs.get("destination") is None:
            result["session_mae"] = next(remaining)
        return result

    monkeypatch.setattr(module, "evaluate", assess)


def selected(folder):
    identity = read_manifest(folder / "run.json")[0]["identity"]
    return load_training_state(folder / "checkpoints", expected_identity=identity, selection="best")


def assert_same_state(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_same_state(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            assert_same_state(a, b)
    else:
        assert first == second


@pytest.mark.parametrize("scores", [(0.1, 0.2, 0.3), (0.1, 0.1, 0.1)])
def test_initial_policy_wins_when_epochs_worsen_or_tie(tmp_path, monkeypatch, scores):
    data = dataset(tmp_path)
    kwargs = configuration(data, epochs=2)
    controlled_scores(monkeypatch, scores)
    report = run_case(output=tmp_path / "run", **kwargs)
    assert report["best_epoch"] == 0
    assert report["baseline"]["samples"] == data.counts["validation"]
    assert report["baseline"]["primary"] == "median"
    assert report["selection_policy"]["version"] == 2
    assert report["identity"]["selection_policy"] == report["selection_policy"]
    assert report["global_step"] == report["total_steps"]
    assert selected(tmp_path / "run")["global_step"] == 0
    output = pq.read_table(tmp_path / "run/validation-predictions.parquet")
    np.testing.assert_array_equal(output["center"].to_numpy(), output["parent"].to_numpy())
    data.parent.close()


def test_early_stop_preserves_selection_and_rng_across_resume(tmp_path, monkeypatch):
    data = dataset(tmp_path)
    kwargs = configuration(data, patience=2)
    scores = (0.4, 0.2, 0.2, 0.3)
    controlled_scores(monkeypatch, scores)
    whole = run_case(output=tmp_path / "whole", **kwargs)
    controlled_scores(monkeypatch, scores)
    paused = run_case(output=tmp_path / "split", max_updates=3, **kwargs)
    assert paused["status"] == "paused"
    split = run_case(output=tmp_path / "split", resume=True, **kwargs)
    assert split["status"] == "completed" and split["stopped_early"]
    assert split["selection"]["stale_epochs"] == 2
    assert split["best_epoch"] == 1
    assert len(split["epochs"]) == 3
    assert split["global_step"] == 3 * split["budget"]["updates"] < split["total_steps"]
    assert (
        whole["predictions"]["validation"]["metrics"]
        == split["predictions"]["validation"]["metrics"]
    )
    assert_same_state(state(tmp_path / "whole"), state(tmp_path / "split"))
    data.parent.close()


@pytest.mark.parametrize("mode", ["neural_mae", "neural_mse"])
def test_neural_early_stop_keeps_parent_and_resumes_dropout_exactly(tmp_path, monkeypatch, mode):
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
    original = {key: value.clone() for key, value in parent.model.state_dict().items()}
    data.parent.predictor = parent.predict
    kwargs = configuration(data, patience=2)
    kwargs["case"]["mode"] = mode
    kwargs["parent"] = parent
    controlled_scores(monkeypatch, (0.1, 0.2, 0.3))
    run_case(output=tmp_path / "whole", **kwargs)
    controlled_scores(monkeypatch, (0.1, 0.2, 0.3))
    run_case(output=tmp_path / "split", max_updates=3, **kwargs)
    result = run_case(output=tmp_path / "split", resume=True, **kwargs)
    assert result["best_epoch"] == 0 and result["stopped_early"]
    assert_same_state(state(tmp_path / "whole"), state(tmp_path / "split"))
    assert_same_state(original, selected(tmp_path / "split")["model"])
    assert_same_state(original, parent.model.state_dict())
    assert any(
        not torch.equal(v, original[k]) for k, v in state(tmp_path / "split")["model"].items()
    )
    data.parent.close()


def test_initial_validation_interruption_does_not_consume_patience(tmp_path, monkeypatch):
    data = dataset(tmp_path)
    kwargs = configuration(data, epochs=1, patience=2)
    run_case(output=tmp_path / "whole", **kwargs)
    batches, stop = data.batches, StopRequest()

    def interrupted(**options):
        for batch in batches(**options):
            yield batch
            if options["partition"] == "validation":
                stop.request_stop()

    with monkeypatch.context() as patch:
        patch.setattr(data, "batches", interrupted)
        result = run_case(output=tmp_path / "split", stop=stop, **kwargs)
    saved = state(tmp_path / "split")
    assert result["status"] == "paused" and saved["global_step"] == 0
    assert saved["baseline"] is None and saved["selection"] is None
    assert read_manifest(tmp_path / "split/checkpoints/latest.json")[0]["best"] is None
    run_case(output=tmp_path / "split", resume=True, **kwargs)
    assert_same_state(state(tmp_path / "whole"), state(tmp_path / "split"))
    data.parent.close()


def test_final_validation_interruption_keeps_latest_weights_with_their_optimizer(
    tmp_path, monkeypatch
):
    data = dataset(tmp_path)
    kwargs = configuration(data, patience=2)
    controlled_scores(monkeypatch, (0.1, 0.2, 0.3))
    run_case(output=tmp_path / "whole", **kwargs)
    stop = StopRequest()
    controlled_scores(monkeypatch, (0.1, 0.2, 0.3), stop_final=stop)
    result = run_case(output=tmp_path / "split", stop=stop, **kwargs)
    assert result["status"] == "paused" and result["global_step"] > 0
    assert_same_state(state(tmp_path / "whole"), state(tmp_path / "split"))
    controlled_scores(monkeypatch, ())
    resumed = run_case(output=tmp_path / "split", resume=True, **kwargs)
    assert resumed["best_epoch"] == 0 and resumed["stopped_early"]
    assert_same_state(state(tmp_path / "whole"), state(tmp_path / "split"))
    assert pq.read_table(tmp_path / "whole/validation-predictions.parquet").equals(
        pq.read_table(tmp_path / "split/validation-predictions.parquet")
    )
    data.parent.close()


def test_minimum_improvement_is_relative_to_the_last_accepted_best(tmp_path, monkeypatch):
    data = dataset(tmp_path)
    kwargs = configuration(data, epochs=3, patience=2, min_delta=0.1)
    controlled_scores(monkeypatch, (1.0, 0.94, 0.88, 0.80))
    report = run_case(output=tmp_path / "run", **kwargs)
    assert report["best_epoch"] == 2 and report["selection"]["best_score"] == 0.88
    assert report["selection"]["stale_epochs"] == 1
    data.parent.close()


def test_fixed_augmentation_budgets_survive_selection_of_epoch_zero(tmp_path, monkeypatch):
    data = dataset(tmp_path)
    results = []
    for condition in ("real_resampled", "real_synthetic"):
        controlled_scores(monkeypatch, (0.1,) * 6)
        report = run_case(output=tmp_path / condition, **configuration(data, condition=condition))
        assert report["best_epoch"] == 0 and not report["stopped_early"]
        assert len(report["epochs"]) == 5
        assert all(e["train"]["real_rows"] == data.counts["train"] for e in report["epochs"])
        results.append(report)
    assert results[0]["global_step"] == results[1]["global_step"] == results[0]["total_steps"]
    assert results[0]["budget"] == results[1]["budget"]
    with pytest.raises(ValueError, match="real|emparejad"):
        run_case(
            output=tmp_path / "invalid",
            **configuration(data, condition="real_synthetic", patience=2),
        )
    data.parent.close()


def test_many_saves_and_resumes_keep_only_two_recent_states_and_one_best(tmp_path, monkeypatch):
    module = importlib.import_module("mars_titan.posttraining.run")
    data = dataset(tmp_path)
    kwargs = configuration(data)
    controlled_scores(monkeypatch, (0.5, 0.4, 0.4, 0.3, 0.3, 0.35))
    save, records = module.save_training_state, []

    def checked_save(directory, state, **options):
        result = save(directory, state, **options)
        index = read_manifest(directory / "latest.json")[0]
        assert len(index["latest"]) <= 2 and index["pinned"] == []
        names = {r["name"] for r in index["latest"]}
        if index["best"] is not None:
            names.add(index["best"]["name"])
        assert names == {p.name for p in directory.glob("state-*.pt")}
        assert len(names) <= 3
        if records and not options.get("best", False):
            assert index["best"] == records[-1]["best"]
        records.append(index)
        return result

    monkeypatch.setattr(module, "save_training_state", checked_save)
    output = tmp_path / "run"
    report = run_case(output=output, max_updates=11, **kwargs)
    while report["status"] == "paused":
        report = run_case(output=output, resume=True, max_updates=11, **kwargs)
    assert len(records) > 20
    assert len({r["best"]["sha256"] for r in records if r["best"]}) == 3
    assert report["best_epoch"] == 3
    assert len(list((output / "checkpoints").glob("state-*.pt"))) <= 3
    data.parent.close()


@pytest.mark.parametrize(
    "changes",
    [
        None,
        {"version": 2.0},
        {"version": 1},
        {"patience": True},
        {"patience": 0},
        {"min_delta": float("nan")},
        {"metric": "mae"},
        {"extra": 1},
    ],
)
def test_selection_policy_rejects_ambiguous_or_invalid_options(changes):
    case = dict(options("mae"), condition="real")
    policy = dict(version=2, metric="session_mae", patience=2, min_delta=0.0)
    case["selection"] = None if changes is None else policy | changes
    with pytest.raises(ValueError, match="selección|paciencia|sesión"):
        selection_policy(case)


@pytest.mark.parametrize("epoch", [-1, False, 0.0])
def test_initial_selection_does_not_coerce_invalid_epoch_numbers(epoch):
    policy = dict(version=2, metric="session_mae", patience=2, min_delta=0.0)
    with pytest.raises(ValueError, match="época"):
        select_epoch(None, 0.1, epoch, policy, 5)


def test_initial_validation_cannot_be_confirmed_twice():
    policy = dict(version=2, metric="session_mae", patience=2, min_delta=0.0)
    initial = select_epoch(None, 0.1, 0, policy, 5)
    with pytest.raises(ValueError, match="inicial"):
        select_epoch(initial, 0.2, 0, policy, 5)


def test_interrupted_epoch_validation_does_not_advance_selection(tmp_path, monkeypatch):
    data = dataset(tmp_path)
    kwargs = configuration(data, epochs=1, patience=2)
    run_case(output=tmp_path / "whole", **kwargs)
    batches, stop = data.batches, StopRequest()
    evaluations = 0

    def interrupted(**options):
        nonlocal evaluations
        if options["partition"] == "validation":
            evaluations += 1
        for batch in batches(**options):
            yield batch
            if options["partition"] == "validation" and evaluations == 2:
                stop.request_stop()

    with monkeypatch.context() as patch:
        patch.setattr(data, "batches", interrupted)
        run_case(output=tmp_path / "split", stop=stop, **kwargs)
    saved = state(tmp_path / "split")
    assert saved["epoch"] == saved["selection"]["last_epoch"] == 0
    assert saved["selection"]["stale_epochs"] == 0
    assert saved["statistics"]["samples"] == data.counts["train"]
    run_case(output=tmp_path / "split", resume=True, **kwargs)
    assert_same_state(state(tmp_path / "whole"), state(tmp_path / "split"))
    data.parent.close()


def test_redirected_best_checkpoint_is_rejected_before_rotating_recovery(tmp_path, monkeypatch):
    data = dataset(tmp_path)
    kwargs = configuration(data, patience=2)
    stop = StopRequest()
    controlled_scores(monkeypatch, (0.1, 0.2, 0.3), stop_final=stop)
    output = tmp_path / "run"
    run_case(output=output, stop=stop, **kwargs)
    folder = output / "checkpoints"
    index = read_manifest(folder / "latest.json")[0]
    assert index["best"]["global_step"] == 0 < index["latest"][0]["global_step"]
    index["best"] = index["latest"][0]
    atomic_json(folder / "latest.json", index)
    before = {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="seleccionad"):
        run_case(output=output, resume=True, **kwargs)
    assert {p.name: sha256(p) for p in folder.iterdir() if p.is_file()} == before
    data.parent.close()
