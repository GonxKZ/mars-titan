"""Diseño fijado, codificadores ligados al corpus y preparación recuperable."""

import copy
from pathlib import Path

import numpy as np
import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.episodes.encoding import EncodedWorld
from mars_titan.episodes.worlds import WorldConfig, generate_world
from mars_titan.posttraining.preparation import (
    EpisodeFactory,
    encoder_contract,
    prepare_augmentation,
)
from mars_titan.posttraining.queue import read_design, run_queue
from mars_titan.training.checkpoints import StopRequest


class TestEncoders:
    __test__ = False
    spec = dict(device="cpu", fixture="posttraining")

    def text(self, value):
        return np.full(384, len(value) / 100, dtype=np.float32)

    def images(self, values):
        return np.ones((len(values), 512), dtype=np.float32)


def test_design_contains_all_conditions_and_continuations_only_for_neural_parents():
    design, cases, digest = read_design(Path("configs/baselines/paired-posttraining.json"))
    assert design["fraction"] == 0.25
    assert len(digest) == 64
    assert len(cases("gru")) == 72
    assert len(cases("ridge")) == 54
    assert len({case["id"] for case in cases("gru")}) == 72
    for mode in (
        "reinforce",
        "expected",
        "mae",
        "klpo_full",
        "klpo_mc",
        "klpo_exact",
        "neural_mae",
        "neural_mse",
    ):
        selected = [case for case in cases("gru") if case["case"]["mode"] == mode]
        assert {case["case"]["condition"] for case in selected} == {
            "real",
            "real_resampled",
            "real_synthetic",
        }


def test_encoder_contract_rejects_unrelated_or_open_test_manifests(tmp_path):
    encoded = dict(
        final_test_opened=False, context_sessions=4, configuration={"encoders": TestEncoders.spec}
    )
    atomic_json(tmp_path / "encoded.json", encoded)
    source = dict(
        final_test_opened=False,
        context_sessions=4,
        configuration={"source_manifest_sha256": sha256(tmp_path / "encoded.json")},
    )
    atomic_json(tmp_path / "source.json", source)
    assert (
        encoder_contract(tmp_path / "source.json", tmp_path / "encoded.json")["encoders"]
        == TestEncoders.spec
    )
    source["configuration"]["source_manifest_sha256"] = "a" * 64
    atomic_json(tmp_path / "source.json", source)
    with pytest.raises(ValueError, match="codificación"):
        encoder_contract(tmp_path / "source.json", tmp_path / "encoded.json")
    encoded["final_test_opened"] = True
    atomic_json(tmp_path / "encoded.json", encoded)
    source["configuration"]["source_manifest_sha256"] = sha256(tmp_path / "encoded.json")
    atomic_json(tmp_path / "source.json", source)
    with pytest.raises(ValueError, match="codificación"):
        encoder_contract(tmp_path / "source.json", tmp_path / "encoded.json")


def test_materialized_augmentation_resumes_and_rejects_a_different_encoder(tmp_path):
    encoders = TestEncoders()
    source = EncodedWorld(
        generate_world(WorldConfig(assets=2, sessions=12, context=4)),
        encoders,
        expected_spec=encoders.spec,
    )
    stop = StopRequest()
    stop.request_stop()
    kwargs = dict(seed=7, decisions=2, warmup=1, volatility=0.006, expected_spec=encoders.spec)
    with pytest.raises(InterruptedError):
        prepare_augmentation(source, tmp_path / "extra", encoders, stop=stop, **kwargs)
    result = prepare_augmentation(source, tmp_path / "extra", encoders, **kwargs)
    assert result["extra_rows"] == 4
    assert prepare_augmentation(source, tmp_path / "extra", encoders, **kwargs) == result
    with EpisodeFactory(tmp_path / "extra", result) as factory:
        view = factory(0)
        assert view(0)["inputs"]["news"].shape == (2, 384)
        assert view.source.partition == "train"
    other = copy.copy(encoders)
    other.spec = {**encoders.spec, "fixture": "different"}
    with pytest.raises(ValueError):
        prepare_augmentation(source, tmp_path / "extra", other, **kwargs)


def test_queue_never_loads_parents_without_an_available_gpu_lease(tmp_path, monkeypatch):
    class Busy:
        def __enter__(self):
            raise RuntimeError("Hay otra carga de cómputo activa en CUDA")

        def __exit__(self, *_):
            pass

    monkeypatch.setattr("mars_titan.posttraining.queue.GpuLease", Busy)
    with pytest.raises(RuntimeError, match="otra carga"):
        run_queue(
            Path("configs/baselines/paired-posttraining.json"),
            tmp_path / "reference.json",
            tmp_path / "tabular.json",
            tmp_path / "encoded.json",
            tmp_path / "run",
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "change", ["duplicate", "identity", "rows", "source_window", "synthetic_window"]
)
def test_partial_augmentation_receipt_cannot_change_the_matched_budget(tmp_path, change):
    from mars_titan.data.cohort_files import read_manifest

    encoders = TestEncoders()
    source = EncodedWorld(
        generate_world(WorldConfig(assets=2, sessions=12, context=4)),
        encoders,
        expected_spec=encoders.spec,
    )
    output = tmp_path / "extra"
    kwargs = dict(seed=7, decisions=2, warmup=1, volatility=0.006, expected_spec=encoders.spec)
    prepare_augmentation(source, output, encoders, **kwargs)
    path = output / "augmentation.json"
    receipt = read_manifest(path)[0]
    receipt["status"] = "paused"
    if change == "duplicate":
        receipt["episodes"].append(copy.deepcopy(receipt["episodes"][0]))
    elif change == "identity":
        receipt["identity"]["seed"] = 999
    elif change == "rows":
        receipt["extra_rows"] = 8
    elif change == "source_window":
        receipt["episodes"][0]["source_window"]["decision_start"] -= 1
    else:
        receipt["episodes"][0]["synthetic_window"]["decision_start"] += 1
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match="recibo|ventanas"):
        prepare_augmentation(source, output, encoders, **kwargs)


def test_queue_recovers_all_objectives_and_does_not_rewrite_completed_run(tmp_path, monkeypatch):
    import importlib

    import torch

    from mars_titan.data.cohort_files import read_manifest
    from mars_titan.environments.actions import ActionGrid
    from mars_titan.models.baselines.multimodal import MultimodalReference
    from mars_titan.posttraining.parents import FrozenParent

    module = importlib.import_module("mars_titan.posttraining.queue")
    encoders = TestEncoders()
    roots = tmp_path / "sources"
    roots.mkdir()
    encoded = roots / "encoded.json"
    atomic_json(
        encoded,
        dict(
            final_test_opened=False, context_sessions=4, configuration={"encoders": encoders.spec}
        ),
    )
    original = roots / "source.json"
    atomic_json(
        original,
        dict(
            final_test_opened=False,
            context_sessions=4,
            configuration={"source_manifest_sha256": sha256(encoded)},
        ),
    )
    parent_report = roots / "parent.json"
    atomic_json(parent_report, {"fixture": "cpu"})
    fixture_sources = {
        partition: EncodedWorld(
            generate_world(
                WorldConfig(
                    assets=2,
                    sessions=12 if partition == "train" else 8,
                    context=4,
                    partition=partition,
                )
            ),
            encoders,
            expected_spec=encoders.spec,
        )
        for partition in ("train", "validation")
    }
    counts = {p: sum(row[1] for row in source.index) for p, source in fixture_sources.items()}
    proof = dict(
        manifest=str(original),
        manifest_sha256=sha256(original),
        counts=counts,
        parents={"gru": dict(report=str(parent_report), sha256=sha256(parent_report))},
    )

    class CpuLease:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def check(self):
            pass

    class Source:
        def __init__(self, _manifest, *, partition):
            self.source = fixture_sources[partition]

        def __getattr__(self, name):
            return getattr(self.source, name)

        def __len__(self):
            return len(self.source)

        def __call__(self, i):
            return self.source(i)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    def ordered(_manifest, output, **_kwargs):
        output.mkdir(exist_ok=True)
        source = fixture_sources["train"]
        target = np.concatenate([source(i)["target"] for i in range(len(source))])
        grid = ActionGrid.fit(target, source_sha256=source.source_sha256, partition="train")
        meta = dict(status="completed", source_sha256=sha256(original), grid=grid.to_dict())
        atomic_json(output / "manifest.json", meta)
        return meta

    def parent(*_args, **_kwargs):
        torch.manual_seed(1)
        shapes = fixture_sources["train"].shapes
        model = MultimodalReference(
            "gru", {k: s[-1] for k, s in shapes.items()}, context=4, hidden_size=32
        )
        return FrozenParent(model, dict(model="gru", checkpoint_sha256="a" * 64), shapes, "cpu")

    monkeypatch.setattr(module, "GpuLease", CpuLease)
    monkeypatch.setattr(module, "selected_parents", lambda *_args: proof)
    monkeypatch.setattr(module, "prepare_causal_corpus", ordered)
    monkeypatch.setattr(module, "ParquetCohortSource", Source)
    monkeypatch.setattr(module, "FrozenEncoders", TestEncoders)
    monkeypatch.setattr(module, "load_parent", parent)
    real_run, first_stop = module.run_case, StopRequest()

    def cpu_run(*args, **kwargs):
        kwargs.update(device="cpu", diagnostic=True, lease=None)
        result = real_run(*args, **kwargs)
        first_stop.request_stop()
        return result

    monkeypatch.setattr(module, "run_case", cpu_run)
    plan = read_design(Path("configs/baselines/paired-posttraining.json"))[0]
    plan.update(seeds=[7], decisions=2, warmup=1, epochs=1, batch_size=8, auxiliary_samples=2)
    config = tmp_path / "config.json"
    atomic_json(config, plan)
    output = tmp_path / "queue"
    args = (config, roots / "reference.json", roots / "tabular.json", encoded, output)
    paused = run_queue(*args, stop=first_stop)
    assert paused["status"] == "paused" and paused["completed_runs"] == 1
    record = next(iter(paused["runs"].values()))
    previous = sha256(output / record["path"])
    completed = run_queue(*args)
    assert completed["status"] == "completed"
    assert completed["completed_runs"] == completed["planned_runs"] == 24
    assert sha256(output / record["path"]) == previous
    result = read_manifest(
        output / "parents/gru/runs/seed-7/real_synthetic/neural_mae/run.json", 8 * 1024**2
    )[0]
    assert result["budget"]["rows"] == 20
    assert result["epochs"][0]["validation"]["samples"] == 8
    assert result["final_test_opened"] is False
    path = output / "parents/gru/normalization.json"
    changed = read_manifest(path)[0]
    changed["mean"][0] += 1
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="artefacto"):
        run_queue(*args)
