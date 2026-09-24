"""Preparación de escenarios con padres verificados y recursos explícitos."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.episodes.worlds import WorldConfig, generate_world
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.storage import read_tape
from scripts.prepare_financial_scenarios import prepare_scenarios


def config(tmp_path):
    path = tmp_path / "config.json"
    atomic_json(
        path,
        dict(
            schema_version=1,
            generator=dict(assets=2, sessions=8, context=4),
            scenarios={"known_signal": 0.002},
            train_seeds=[7],
            validation_seeds=[8],
            final_test_opened=False,
        ),
    )
    return path


class FixtureEncoders:
    spec = dict(fixture="cpu", device="cpu")

    def text(self, text):
        return np.full(384, len(text) / 100, dtype=np.float32)

    def images(self, values):
        return np.ones((len(values), 512), dtype=np.float32)


def trained_inputs(tmp_path):
    import torch

    from mars_titan.models.baselines.multimodal import MultimodalReference
    from mars_titan.training.checkpoints import save_training_state

    root = tmp_path / "source"
    root.mkdir()
    paths = {name: root / f"{name}.json" for name in ("encoded", "supervision", "ordered")}
    counts = dict(train=8, validation=8)
    atomic_json(
        paths["encoded"],
        dict(
            final_test_opened=False,
            context_sessions=4,
            configuration=dict(encoders=FixtureEncoders.spec),
        ),
    )
    atomic_json(
        paths["supervision"],
        dict(
            final_test_opened=False,
            context_sessions=4,
            counts=counts,
            scope="full_corpus",
            cohort_complete=True,
            configuration=dict(source_manifest_sha256=sha256(paths["encoded"])),
        ),
    )
    dimensions = dict(prices=5, news=384, charts=512, fundamentals=45, macro=420)
    shapes = {key: [4, 5] if key == "prices" else [value] for key, value in dimensions.items()}
    atomic_json(
        paths["ordered"],
        dict(
            schema_version=1,
            kind="causal_prediction_corpus",
            status="completed",
            final_test_opened=False,
            scope="full_corpus",
            cohort_complete=True,
            source_sha256=sha256(paths["supervision"]),
            identity=dict(source_sha256=sha256(paths["supervision"])),
            counts=counts,
            shapes=shapes,
            partitions={"train": {"market_rows": {"US": 8}}},
        ),
    )
    folder = root / "parent"
    folder.mkdir()
    case = dict(kind="dlinear", architecture=dict(hidden_size=32, layers=1, dropout=0.0), epochs=1)
    identity = dict(
        manifest_sha256=sha256(paths["supervision"]),
        case=case,
        dimensions=dimensions,
        context=4,
        model_family="scientific_multimodal_reference",
        weighting="natural",
        market_weights={"US": 1.0},
        code={
            name: sha256(Path("src/mars_titan") / name)
            for name in (
                "training/corpus_inputs.py",
                "models/baselines/multimodal.py",
                "models/baselines/dlinear.py",
            )
        },
    )
    torch.manual_seed(3)
    model = MultimodalReference("dlinear", dimensions, context=4, **case["architecture"])
    checkpoint = save_training_state(
        folder / "checkpoints",
        dict(
            global_step=1,
            epoch=1,
            model=model.state_dict(),
            confirmed_cursor=None,
            statistics={"samples": 0},
        ),
        identity=identity,
    )
    paths["parent"] = folder / "run.json"
    atomic_json(
        paths["parent"],
        dict(
            status="completed",
            final_test_opened=False,
            identity=identity,
            scope="full_corpus",
            cohort_complete=True,
            samples=counts,
            predictions={"train": {}, "validation": {}},
            checkpoint=dict(path=str(checkpoint.relative_to(folder)), sha256=sha256(checkpoint)),
        ),
    )
    return paths, checkpoint


@pytest.fixture
def cpu_admission(monkeypatch):
    from mars_titan.data import embeddings
    from mars_titan.posttraining import parents
    from mars_titan.training import experiment_resources

    class Lease:
        active = False
        checks = 0

        def __enter__(self):
            self.active = True
            return self

        def __exit__(self, *_):
            self.active = False

        def check(self):
            assert self.active
            self.checks += 1

    lease, original = Lease(), parents.load_parent

    def load(*args, **kwargs):
        assert lease.active
        return original(*args, device="cpu", diagnostic=True)

    class Encoders(FixtureEncoders):
        def __init__(self):
            assert lease.active

    monkeypatch.setattr(experiment_resources, "GpuLease", lambda: lease)
    monkeypatch.setattr(parents, "load_parent", load)
    monkeypatch.setattr(embeddings, "FrozenEncoders", Encoders)
    return lease


def test_trained_parent_scores_every_observation_and_preserves_checkpoint(tmp_path, cpu_admission):
    paths, checkpoint = trained_inputs(tmp_path)
    before = sha256(checkpoint)
    output = tmp_path / "scenarios"
    result = prepare_scenarios(config(tmp_path), output, **paths)
    assert result["parent"] == before
    assert result["final_test_opened"] is False
    assert len(result["records"]) == 2
    for record in result["records"]:
        tape = read_tape(output / record["name"])
        assert tape.identity["parent_id"] == before
        assert tape.scores.shape == (5, 2)
        assert np.isfinite(tape.scores[-1]).all()
    assert cpu_admission.checks >= 10
    assert sha256(checkpoint) == before
    fingerprints = {
        record["name"]: sha256(output / record["name"] / "market.parquet")
        for record in result["records"]
    }
    resumed = prepare_scenarios(tmp_path / "config.json", output, resume=True, **paths)
    assert resumed == result
    assert all(
        sha256(output / name / "market.parquet") == value for name, value in fingerprints.items()
    )


@pytest.mark.parametrize("missing", ["parent", "ordered", "supervision", "encoded"])
def test_parent_arguments_are_an_indivisible_set(tmp_path, missing):
    paths = {
        name: tmp_path / f"{name}.json" for name in ("parent", "ordered", "supervision", "encoded")
    }
    paths.pop(missing)
    with pytest.raises(ValueError, match="conjunto"):
        prepare_scenarios(config(tmp_path), tmp_path / "output", **paths)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "change",
    ["supervision", "ordered_identity", "encoding", "dimensions", "context", "encoder_spec"],
)
def test_incompatible_provenance_is_rejected_before_scenario_output(
    tmp_path, cpu_admission, monkeypatch, change
):
    from mars_titan.data import embeddings

    paths, _ = trained_inputs(tmp_path)
    settings = config(tmp_path)
    if change == "encoder_spec":

        class Other(FixtureEncoders):
            spec = dict(fixture="other", device="cpu")

        monkeypatch.setattr(embeddings, "FrozenEncoders", Other)
    else:
        path = (
            settings
            if change == "context"
            else paths[
                "ordered"
                if change in {"dimensions", "supervision", "ordered_identity"}
                else "encoded"
            ]
        )
        value = json.loads(path.read_text())
        if change == "supervision":
            value["source_sha256"] = "b" * 64
        elif change == "ordered_identity":
            value["identity"]["source_sha256"] = "b" * 64
        elif change == "encoding":
            value["configuration"]["encoders"]["fixture"] = "other"
        elif change == "dimensions":
            value["shapes"]["news"] = [3]
        else:
            value["generator"]["context"] = 3
        atomic_json(path, value)
    with pytest.raises(ValueError):
        prepare_scenarios(settings, tmp_path / "output", **paths)
    assert not (tmp_path / "output").exists()


def test_analytic_cli_does_not_import_torch_and_rejects_unknown_partial_output(tmp_path):
    settings = config(tmp_path)
    output = tmp_path / "analytic"
    code = (
        "import runpy, sys\n"
        "sys.argv=['scripts/prepare_financial_scenarios.py','--config',sys.argv[1],'--output',sys.argv[2]]\n"
        "runpy.run_path('scripts/prepare_financial_scenarios.py', run_name='__main__')\n"
        "assert 'torch' not in sys.modules\n"
    )
    child = subprocess.run(
        [sys.executable, "-c", code, str(settings), str(output)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stderr
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    sentinel = unknown / "unrelated.txt"
    sentinel.write_text("conservar")
    with pytest.raises(ValueError):
        prepare_scenarios(settings, unknown, resume=True)
    assert sentinel.read_text() == "conservar"
    assert sorted(p.name for p in unknown.iterdir()) == ["unrelated.txt"]


def test_tape_resource_check_runs_before_inference_and_can_stop_a_cohort():
    world = generate_world(WorldConfig(assets=2, sessions=8, context=4))
    calls = []

    def check():
        calls.append("check")
        if len(calls) == 5:
            raise MemoryError("límite de prueba")

    def predict(inputs):
        calls.append("predict")
        return np.zeros(len(inputs["news"]))

    with pytest.raises(MemoryError, match="límite"):
        MarketTape.from_world(world, predict, check_resources=check)
    assert calls == ["check", "predict", "check", "predict", "check"]


def test_complete_world_survives_interruption_before_index_confirmation(
    tmp_path, cpu_admission, monkeypatch
):
    from scripts import prepare_financial_scenarios as preparation

    paths, _ = trained_inputs(tmp_path)
    settings, output = config(tmp_path), tmp_path / "interrupted"
    write = preparation.write_tape

    def interrupted(tape, destination):
        write(tape, destination)
        raise InterruptedError("parada después del mundo completo")

    monkeypatch.setattr(preparation, "write_tape", interrupted)
    with pytest.raises(InterruptedError):
        prepare_scenarios(settings, output, **paths)
    first = output / "known_signal-train-7/market.parquet"
    digest, modified = sha256(first), first.stat().st_mtime_ns
    monkeypatch.setattr(preparation, "write_tape", write)
    result = prepare_scenarios(settings, output, resume=True, **paths)
    assert len(result["records"]) == 2
    assert sha256(first) == digest and first.stat().st_mtime_ns == modified


def test_partial_world_and_changed_identity_are_not_overwritten(tmp_path):
    settings, output = config(tmp_path), tmp_path / "output"
    prepare_scenarios(settings, output)
    folder = output / "known_signal-train-7"
    path = folder / "market.parquet"
    digest = sha256(path)
    manifest = folder / "manifest.json"
    manifest.rename(folder / "unconfirmed.json")
    with pytest.raises(ValueError, match="manifiesto"):
        prepare_scenarios(settings, output, resume=True)
    assert not manifest.exists()
    assert sha256(path) == digest
    previous = (output / "index.json").read_bytes()
    changed = json.loads(settings.read_text())
    changed["train_seeds"] = [9]
    atomic_json(settings, changed)
    with pytest.raises(ValueError, match="identidad"):
        prepare_scenarios(settings, output, resume=True)
    assert (output / "index.json").read_bytes() == previous


def test_busy_gpu_is_rejected_before_loading_parent_or_encoders(
    tmp_path, cpu_admission, monkeypatch
):
    from mars_titan.training import experiment_resources

    class Busy:
        def __enter__(self):
            raise RuntimeError("otra carga CUDA")

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(experiment_resources, "GpuLease", Busy)
    paths, _ = trained_inputs(tmp_path)
    with pytest.raises(RuntimeError, match="otra carga"):
        prepare_scenarios(config(tmp_path), tmp_path / "blocked", **paths)
    assert not (tmp_path / "blocked").exists()


def test_cli_explains_the_missing_manifest_arguments(tmp_path):
    child = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_financial_scenarios.py",
            "--output",
            str(tmp_path / "output"),
            "--parent",
            str(tmp_path / "parent.json"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 2
    assert "conjunto obligatorio" in child.stderr
    assert (
        "--ordered" in child.stderr
        and "--supervision" in child.stderr
        and "--encoded" in child.stderr
    )


def test_changed_complete_world_cannot_be_reused_as_the_original(tmp_path):
    from mars_titan.simulation.storage import write_tape

    settings, output = config(tmp_path), tmp_path / "output"
    prepare_scenarios(settings, output)
    folder = output / "known_signal-train-7"
    original = read_tape(folder)
    changed = MarketTape(
        original.prices,
        original.close_times,
        original.assets,
        original.scores + 1,
        domain="synthetic",
        currency="USD",
        partition="train",
        prediction_times=original.prediction_times,
        open_times=original.open_times,
        parent_id=original.identity["parent_id"],
        source_identity=original.identity["source"],
    )
    replacement = tmp_path / "replacement"
    write_tape(changed, replacement)
    folder.rename(tmp_path / "original")
    replacement.rename(folder)
    with pytest.raises(ValueError, match="identidad"):
        prepare_scenarios(settings, output, resume=True)
    assert read_tape(folder).sha256 == changed.sha256


@pytest.mark.parametrize("field,value", [("domain", "real"), ("schema_version", 2)])
def test_changed_index_header_is_rejected(tmp_path, field, value):
    settings, output = config(tmp_path), tmp_path / "output"
    prepare_scenarios(settings, output)
    path = output / "index.json"
    index = json.loads(path.read_text())
    index[field] = value
    atomic_json(path, index)
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="identidad"):
        prepare_scenarios(settings, output, resume=True)
    assert path.read_bytes() == previous
