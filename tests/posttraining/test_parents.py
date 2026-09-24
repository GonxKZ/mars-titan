"""Admisión de padres completos y copias de pesos independientes."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.models.baselines.multimodal import MultimodalReference
from mars_titan.posttraining.parents import _inference_contract, load_parent
from mars_titan.training.checkpoints import save_training_state


def parent_files(tmp_path):
    folder = tmp_path / "parent"
    folder.mkdir()
    dimensions = dict(prices=5, news=2, charts=2, fundamentals=3, macro=2)
    case = dict(kind="gru", architecture=dict(hidden_size=32, layers=1, dropout=0.1), epochs=1)
    identity = dict(
        manifest_sha256="a" * 64,
        case=case,
        dimensions=dimensions,
        context=4,
        model_family="scientific_multimodal_reference",
        weighting="natural",
        market_weights={"US": 1.0},
        code={
            f"models/baselines/{name}": sha256(Path("src/mars_titan/models/baselines") / name)
            for name in ("multimodal.py", "dlinear.py")
        }
        | {"training/corpus_inputs.py": sha256(Path("src/mars_titan/training/corpus_inputs.py"))},
    )
    model = MultimodalReference("gru", dimensions, context=4, **case["architecture"])
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
    report = dict(
        status="completed",
        final_test_opened=False,
        identity=identity,
        scope="full_corpus",
        cohort_complete=True,
        samples=dict(train=10, validation=4),
        predictions={"train": {}, "validation": {}},
        checkpoint=dict(path=str(checkpoint.relative_to(folder)), sha256=sha256(checkpoint)),
    )
    ordered = dict(
        kind="causal_prediction_corpus",
        status="completed",
        final_test_opened=False,
        source_sha256="a" * 64,
        scope="full_corpus",
        cohort_complete=True,
        counts=report["samples"],
        partitions={"train": {"market_rows": {"US": 10}}},
        shapes={k: [4, 5] if k == "prices" else [v] for k, v in dimensions.items()},
    )
    atomic_json(folder / "run.json", report)
    atomic_json(tmp_path / "ordered.json", ordered)
    return tmp_path / "ordered.json", folder / "run.json", report


def test_loaded_parent_is_frozen_and_continuation_owns_its_weights(tmp_path):
    ordered, report, _ = parent_files(tmp_path)
    parent = load_parent(ordered, report, device="cpu", diagnostic=True)
    assert not parent.model.training
    assert all(not parameter.requires_grad for parameter in parent.model.parameters())
    child = parent.continuation()
    assert all(parameter.requires_grad for parameter in child.parameters())
    before = copy.deepcopy(parent.model.state_dict())
    with torch.no_grad():
        next(child.parameters()).add_(1)
    assert all(torch.equal(value, parent.model.state_dict()[key]) for key, value in before.items())
    inputs = {k: np.ones((3, *shape), dtype=np.float32) for k, shape in parent.shapes.items()}
    np.testing.assert_array_equal(parent.predict(inputs), parent.predict(inputs))
    inputs["news"] = np.ones((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="dimensiones"):
        parent.predict(inputs)


@pytest.mark.parametrize("change", ["status", "test", "samples", "shape", "checkpoint", "code"])
def test_incomplete_or_incompatible_parent_is_rejected(tmp_path, change):
    ordered, path, report = parent_files(tmp_path)
    if change == "status":
        report["status"] = "paused"
    elif change == "test":
        report["final_test_opened"] = True
    elif change == "samples":
        report["samples"]["train"] = 9
    elif change == "shape":
        report["identity"]["dimensions"]["news"] = 3
    elif change == "checkpoint":
        report["checkpoint"]["sha256"] = "b" * 64
    else:
        report["identity"]["code"]["models/baselines/dlinear.py"] = "c" * 64
    atomic_json(path, report)
    with pytest.raises(ValueError):
        load_parent(ordered, path, device="cpu", diagnostic=True)


def test_scientific_parent_requires_live_gpu_lease_before_loading(tmp_path):
    ordered, path, _ = parent_files(tmp_path)
    with pytest.raises(ValueError, match="GpuLease"):
        load_parent(ordered, path)


@pytest.mark.parametrize("kind", ["ridge", "xgboost_external_cuda"])
def test_tabular_inference_contract_rejects_changed_code(kind):
    paths = [
        "training/corpus_inputs.py",
        "models/baselines/inputs.py",
        f"models/baselines/{'ridge' if kind == 'ridge' else 'external_boosting'}.py",
    ]
    hashes = {name: sha256(Path("src/mars_titan") / name) for name in paths}
    report = dict(code=hashes) if kind == "ridge" else dict(identity=dict(code=hashes))
    _inference_contract(report, kind)
    hashes[paths[-1]] = "b" * 64
    with pytest.raises(ValueError, match="inferencia"):
        _inference_contract(report, kind)
