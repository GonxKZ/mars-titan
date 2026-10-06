"""Evaluar estados congelados sin ajustar ni seleccionar con los bloques posteriores."""

from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.environments.actions import ActionGrid
from mars_titan.models.predictive_adaptation import LinearResidualPolicy
from mars_titan.posttraining.evaluation import evaluate
from mars_titan.posttraining.heldout import evaluate_partition
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.test_temporal_corpus import inputs as inputs
from tests.training.test_temporal_corpus import prepare


class Parent:
    def predict(self, inputs):
        return inputs["macro"][:, 0].astype(np.float64) / 100


def test_frozen_predictions_visit_only_the_requested_block(inputs, tmp_path):
    prepare(inputs, tmp_path / "views")
    data = CorpusDataset(tmp_path / "views/fold-000/manifest.json")
    seen = []
    original = data.batches

    def batches(**kwargs):
        seen.append(kwargs["partition"])
        return original(**kwargs)

    data.batches = batches
    parent = Parent()
    for partition in ("calibration", "evaluation"):
        path = tmp_path / f"{partition}.parquet"
        scores = evaluate_partition(data, parent, partition, path, stop=StopRequest(), device="cpu")
        rows = pq.read_table(path).to_pydict()
        assert len(rows["prediction"]) == data.manifest["counts"][partition]
        assert scores["prediction"]["samples"] == 1
        assert scores["prediction"]["mae"] == abs(rows["prediction"][0] - rows["target"][0])
        assert scores["prediction"] == scores["parent"]
        assert rows["zero"] == [0.0]
    assert seen == ["calibration", "evaluation"]
    for forbidden in ("train", "validation", "test", "test_reserved"):
        with pytest.raises(ValueError, match="calibración|evaluación"):
            evaluate_partition(
                data, parent, forbidden, tmp_path / "bad.parquet", stop=StopRequest(), device="cpu"
            )


def test_frozen_adapter_matches_the_existing_validation_formula(inputs, tmp_path):
    prepare(inputs, tmp_path / "views")
    data = CorpusDataset(tmp_path / "views/fold-000/manifest.json")
    parent = Parent()
    batch = next(data.batches(partition="calibration", batch_size=256, epoch=0, seed=0))
    inherited = parent.predict(batch["inputs"])
    from mars_titan.models.baselines.inputs import MODALITIES

    features = np.concatenate(
        [batch["inputs"][k].reshape(len(inherited), -1) for k in MODALITIES]
        + [inherited[:, None].astype(np.float32)],
        axis=1,
    )
    grid = ActionGrid.fit(np.array([-0.1, 0.1]), source_sha256="a" * 64, partition="train")
    model = LinearResidualPolicy(
        np.zeros(features.shape[1]), np.ones(features.shape[1]), target_scale=grid.scale
    )
    frozen = dict(batch, parent=inherited, features=features)
    existing = SimpleNamespace(counts={"validation": 1}, batches=lambda **kwargs: iter([frozen]))
    expected = evaluate(
        model, existing, grid, batch_size=256, neural=False, device="cpu", stop=StopRequest()
    )
    result = evaluate_partition(
        data,
        parent,
        "calibration",
        tmp_path / "adapter.parquet",
        model=model,
        grid=grid,
        neural=False,
        device="cpu",
        stop=StopRequest(),
    )
    for name, source in (
        ("prediction", expected),
        ("center", expected["center"]),
        ("parent", expected["parent"]),
    ):
        for metric in ("samples", "session_mae", "session_mse", "mae", "mse"):
            assert result[name][metric] == source[metric]
    assert all(parameter.grad is None for parameter in model.parameters())


def test_interrupted_evaluation_does_not_publish_partial_predictions(inputs, tmp_path):
    prepare(inputs, tmp_path / "views")
    data = CorpusDataset(tmp_path / "views/fold-000/manifest.json")
    stop = StopRequest()
    stop.requested = True
    destination = tmp_path / "paused.parquet"
    with pytest.raises(InterruptedError):
        evaluate_partition(data, Parent(), "evaluation", destination, stop=stop, device="cpu")
    assert not destination.exists()


@pytest.mark.parametrize("mode", ["expected", "klpo_mc", "neural_mse"])
def test_reconstructs_the_selected_checkpoint_and_rejects_a_changed_parent(tmp_path, mode):
    import torch
    from test_inputs import dataset
    from test_run import grid, options

    from mars_titan.models.baselines.multimodal import MultimodalReference
    from mars_titan.posttraining.heldout import _adjustment
    from mars_titan.posttraining.inputs import fit_normalization
    from mars_titan.posttraining.parents import FrozenParent
    from mars_titan.posttraining.run import _best_state, run_case

    data = dataset(tmp_path)
    parent = FrozenParent(
        MultimodalReference(
            "gru",
            {k: shape[-1] for k, shape in data.shapes.items()},
            context=4,
            hidden_size=32,
            layers=1,
            dropout=0.0,
        ),
        dict(model="gru", checkpoint_sha256="a" * 64),
        data.shapes,
        "cpu",
    )
    data.parent.predictor = parent.predict
    original = {k: v.clone() for k, v in parent.model.state_dict().items()}
    case = dict(
        options(mode),
        condition="real",
        epochs=1,
        selection=dict(version=2, metric="session_mae", patience=2, min_delta=0.0),
    )
    folder = tmp_path / "run"
    report = run_case(
        data,
        folder,
        case,
        grid(data),
        fit_normalization(data, batch_size=4),
        parent=parent,
        batch_size=4,
        device="cpu",
        diagnostic=True,
    )
    restored, action_grid, neural = _adjustment(folder / "run.json", report, parent, "cpu")
    selected = _best_state(
        folder, report["identity"], report["selection"], report["checkpoint"]["sha256"]
    )
    assert all(
        torch.equal(value, selected["model"][key]) for key, value in restored.state_dict().items()
    )
    assert all(
        torch.equal(value, parent.model.state_dict()[key]) for key, value in original.items()
    )
    if not neural:
        assert restored.target_scale == action_grid.scale
    assert not restored.training
    assert not any(parameter.requires_grad for parameter in restored.parameters())
    parent.identity = dict(parent.identity, checkpoint_sha256="b" * 64)
    with pytest.raises(ValueError, match="identidad"):
        _adjustment(folder / "run.json", report, parent, "cpu")
    data.parent.close()


def test_evaluation_resume_skips_confirmed_models_and_detects_corruption(
    inputs, tmp_path, monkeypatch
):
    import json

    from mars_titan.data.storage import atomic_json, sha256
    from mars_titan.posttraining import heldout

    prepare(inputs, tmp_path / "views")
    manifest = tmp_path / "views/fold-000/manifest.json"
    adjustments = tmp_path / "adjustments/summary.json"
    ordered = adjustments.parent / "ordered/manifest.json"
    atomic_json(ordered, dict(source_manifest=dict(sha256=sha256(manifest))))
    jobs = []
    for index in range(2):
        path = tmp_path / f"reference-{index}/run.json"
        atomic_json(path, dict(checkpoint={}, identity=dict(options=dict(seed=43))))
        jobs.append(
            dict(
                id=f"reference/model-{index}",
                stage="reference",
                phase="search",
                report=str(path),
                sha256=sha256(path),
                comparator=None,
            )
        )
    proof = dict(manifest=str(manifest), manifest_sha256=sha256(manifest))
    monkeypatch.setattr(heldout, "_jobs", lambda *args: (proof, {}, jobs))

    class Lease:
        record = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def check(self):
            pass

    class Frozen(Parent):
        kind = "gru"

    monkeypatch.setattr(heldout, "GpuLease", Lease)
    monkeypatch.setattr(heldout, "load_parent", lambda *args, **kwargs: Frozen())
    monkeypatch.setattr(heldout.torch.cuda, "max_memory_allocated", lambda _: 0)
    monkeypatch.setattr(heldout.torch.cuda, "empty_cache", lambda: None)
    evaluated = []
    original = heldout.evaluate_partition
    stop = StopRequest()

    def evaluation(*args, **kwargs):
        kwargs["device"] = "cpu"
        scores = original(*args, **kwargs)
        evaluated.append(str(args[3]))
        if len(evaluated) == 2:
            stop.requested = True
        return scores

    monkeypatch.setattr(heldout, "evaluate_partition", evaluation)
    folder = tmp_path / "evaluation"
    args = (
        tmp_path / "reference/summary.json",
        tmp_path / "tabular/summary.json",
        adjustments,
        folder,
    )
    paused = heldout.run_evaluation(*args, stop=stop)
    assert paused["status"] == "paused" and paused["completed_runs"] == 1
    resumed = heldout.run_evaluation(*args, stop=StopRequest())
    assert resumed["status"] == "completed" and resumed["completed_runs"] == 2
    assert len(evaluated) == 4
    assert resumed["frozen_at_utc"] == paused["frozen_at_utc"]
    summary_hash = sha256(folder / "summary.json")

    class BusyLease:
        def __enter__(self):
            raise RuntimeError("CUDA está ocupada")

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(heldout, "GpuLease", BusyLease)
    assert heldout.run_evaluation(*args)["completed_runs"] == 2
    assert sha256(folder / "summary.json") == summary_hash
    assert len(evaluated) == 4
    first = folder / resumed["runs"][jobs[0]["id"]]["path"]
    assert json.loads(first.read_text())["case"]["seed"] == 43
    predictions = json.loads(first.read_text())["predictions"]["evaluation"]
    (first.parent / predictions["path"]).write_bytes(b"corrupto")
    with pytest.raises(ValueError, match="artefacto|huella"):
        heldout.run_evaluation(*args)


def test_continuous_continuation_compares_against_the_actual_parent(inputs, tmp_path):
    prepare(inputs, tmp_path / "views")
    data = CorpusDataset(tmp_path / "views/fold-000/manifest.json")

    class Shifted(Parent):
        def predict(self, inputs):
            return super().predict(inputs) + 0.1

    path = tmp_path / "continuous.parquet"
    scores = evaluate_partition(
        data, Parent(), "evaluation", path, stop=StopRequest(), device="cpu", predictor=Shifted()
    )
    rows = pq.read_table(path).to_pydict()
    assert scores["prediction"] != scores["parent"]
    np.testing.assert_allclose(np.array(rows["prediction"]) - rows["parent"], 0.1)


def test_freeze_links_reference_continuations_and_rejects_unrelated_parent(tmp_path, monkeypatch):
    from mars_titan.data.storage import atomic_json, sha256
    from mars_titan.posttraining import heldout

    proof = {"fixture": "verified-selection"}
    monkeypatch.setattr(heldout, "selected_parents", lambda *args: proof)

    def run(stage, name, identity):
        path = tmp_path / stage / "runs" / name / "run.json"
        path.parent.mkdir(parents=True)
        checkpoint = path.parent / "model.bin"
        checkpoint.write_bytes(name.encode())
        report = dict(
            status="completed",
            final_test_opened=False,
            identity=identity,
            checkpoint=dict(path=checkpoint.name, sha256=sha256(checkpoint)),
        )
        atomic_json(path, report)
        return path, report

    parent, base = run("reference", "parent", {})
    child, child_report = run(
        "reference",
        "child",
        dict(initialization=dict(parent_checkpoint_sha256=base["checkpoint"]["sha256"])),
    )
    tabular, _ = run("tabular", "ridge", {})
    adjustment, _ = run("posttraining", "gru/expected", {})
    meta = dict(status="completed", final_test_opened=False)
    reference_summary = tmp_path / "reference/summary.json"
    atomic_json(
        reference_summary,
        dict(
            meta,
            planned_runs=2,
            completed_runs=2,
            runs=[
                dict(id="parent", path="runs/parent", report_sha256=sha256(parent), stage="search"),
                dict(
                    id="child",
                    path="runs/child",
                    report_sha256=sha256(child),
                    parent="parent",
                    stage="posttraining",
                ),
            ],
        ),
    )
    tabular_summary = tmp_path / "tabular/summary.json"
    atomic_json(
        tabular_summary,
        dict(
            meta,
            planned_runs=1,
            completed_runs=1,
            runs=[
                dict(
                    id="ridge",
                    stage="search",
                    attempts=[dict(path="runs/ridge")],
                    report_sha256=sha256(tabular),
                )
            ],
        ),
    )
    post_summary = tmp_path / "posttraining/summary.json"
    atomic_json(
        post_summary,
        dict(
            meta,
            planned_runs=1,
            completed_runs=1,
            identity=dict(proof=proof),
            runs={
                "gru/expected": dict(path="runs/gru/expected/run.json", sha256=sha256(adjustment))
            },
        ),
    )
    args = (reference_summary, tabular_summary, post_summary)
    actual, sources, jobs = heldout._jobs(*args)
    assert actual == proof and len(sources) == 3 and len(jobs) == 4
    assert jobs[0]["comparator"] is None
    assert jobs[1]["comparator"] == dict(path=str(parent), sha256=sha256(parent))
    assert jobs[1]["phase"] == "posttraining"
    child_report["identity"]["initialization"]["parent_checkpoint_sha256"] = "a" * 64
    atomic_json(child, child_report)
    import json

    reference = json.loads(reference_summary.read_text())
    reference["runs"][1]["report_sha256"] = sha256(child)
    atomic_json(reference_summary, reference)
    with pytest.raises(ValueError, match="pesos de su padre"):
        heldout._jobs(*args)


def test_freeze_preserves_matching_parents_and_rejects_a_rebound_seed(tmp_path, monkeypatch):
    from mars_titan.data.storage import atomic_json, sha256
    from mars_titan.posttraining import heldout
    from mars_titan.posttraining.parent_selection import matching_parents
    from tests.posttraining.test_parent_selection import matched_campaign

    reference, tabular, _ = matched_campaign(tmp_path)
    proof = matching_parents(reference, tabular, seeds=[42, 43, 44])
    root = tmp_path / "adjustments"
    run = root / "gru-seed43/run.json"
    run.parent.mkdir(parents=True)
    checkpoint = run.parent / "model.bin"
    checkpoint.write_bytes(b"adjusted-gru-43")
    atomic_json(
        run,
        dict(
            status="completed",
            final_test_opened=False,
            checkpoint=dict(path="model.bin", sha256=sha256(checkpoint)),
        ),
    )
    summary = dict(
        status="completed",
        final_test_opened=False,
        planned_runs=1,
        completed_runs=1,
        identity=dict(proof=proof),
        runs={"gru/seed-43/real/mae": dict(path="gru-seed43/run.json", sha256=sha256(run))},
    )
    path = root / "summary.json"
    atomic_json(path, summary)

    def forbidden(*_args):
        pytest.fail("La evaluación emparejada no debe resolver la prueba antigua")

    monkeypatch.setattr(heldout, "selected_parents", forbidden)
    actual, _, jobs = heldout._jobs(reference, tabular, path)
    assert actual == proof
    assert jobs[-1]["id"] == "posttraining/gru/seed-43/real/mae"
    summary["identity"]["proof"]["parents_by_seed"]["gru"]["43"]["sha256"] = "f" * 64
    atomic_json(path, summary)
    with pytest.raises(ValueError, match="referencias congeladas"):
        heldout._jobs(reference, tabular, path)
