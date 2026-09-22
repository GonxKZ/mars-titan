import importlib
import json
import platform
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.storage import sha256
from mars_titan.models.baselines.ridge import RidgeModel
from tests.training.test_reference_run import training_corpus


def module():
    return importlib.import_module("mars_titan.training.tabular_corpus")


@pytest.mark.parametrize("kind,device", [("ridge", "cuda:0"), ("boosting", "cpu")])
def test_tabular_fit_and_predictions_use_the_whole_neural_population(tmp_path, kind, device):
    manifest = training_corpus(tmp_path / "data")
    output = tmp_path / "run"
    result = module().run_tabular_reference(manifest, output, kind=kind, batch_size=5)
    assert result["status"] == "completed" and result["device"] == device
    assert result["runtime"]["python"] == platform.python_version()
    assert result["runtime"]["cuda"] == torch.version.cuda
    assert result["runtime"]["gpu"] == (torch.cuda.get_device_name(0) if kind == "ridge" else None)
    assert result["samples"] == {"train": 12, "validation": 6}
    assert result["fitted_rows"] == 12 and result["final_test_opened"] is False
    for partition, expected in (("train", 12), ("validation", 6)):
        rows = pq.read_table(output / f"{partition}-predictions.parquet").to_pylist()
        assert len({row["sample_id"] for row in rows}) == len(rows) == expected
        metadata = json.loads(manifest.read_text())
        expected_values = {}
        for asset in metadata["assets"]:
            labels = (
                Path(metadata["roots"]["labels"])
                / asset["market"]
                / asset["symbol"]
                / "labels.parquet"
            )
            for label in pq.read_table(labels).to_pylist():
                if label["partition"] == partition:
                    timestamp = int(label["prediction_at"].timestamp() * 1_000_000)
                    key = f"{asset['market']}/{asset['symbol']}/{timestamp}"
                    expected_values[key] = label["target"]
        assert {r["sample_id"]: r["target"] for r in rows} == expected_values
        assert all(np.isfinite(row["prediction"]) for row in rows)
        assert all(row["zero"] == 0 for row in rows)
    assert result["restored_predictions_equal"] is True


def test_tabular_matrix_preserves_modalities_and_window_order():
    batch = {
        "target": np.array([0.1]),
        "inputs": {
            "prices": np.array([[[1.0, 2.0], [3.0, 4.0]]]),
            "news": np.array([[5.0, 6.0]]),
            "charts": np.array([[7.0]]),
            "fundamentals": np.array([[8.0]]),
            "macro": np.array([[9.0]]),
        },
    }
    np.testing.assert_array_equal(module()._matrix(batch), [[1, 2, 3, 4, 5, 6, 7, 8, 9]])


def test_boosting_budget_does_not_turn_into_row_sampling(tmp_path):
    manifest = training_corpus(tmp_path / "data")
    output = tmp_path / "run"
    report = module().run_tabular_reference(manifest, output, kind="boosting", max_matrix_bytes=1)
    assert report["status"] == "not_executed_budget"
    assert report["fitted_rows"] == 0 and report["samples"]["train"] == 12
    assert not (output / "model.joblib").exists()


def test_validation_targets_do_not_change_ridge_fit(tmp_path):
    manifest = training_corpus(tmp_path / "data")
    engine = module()
    engine.run_tabular_reference(manifest, tmp_path / "first", kind="ridge")
    meta = json.loads(manifest.read_text())
    for asset in meta["assets"]:
        path = tmp_path / "data/labels" / asset["market"] / asset["symbol"] / "labels.parquet"
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            if row["partition"] == "validation":
                row["target"] += 100.0
        pq.write_table(pa.Table.from_pylist(rows), path)
        asset["labels_sha256"] = sha256(path)
    manifest.write_text(json.dumps(meta))
    engine.run_tabular_reference(manifest, tmp_path / "second", kind="ridge")
    first, second = (RidgeModel.load(tmp_path / name / "model.npz") for name in ("first", "second"))
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.coefficient, second.coefficient)
    assert first.intercept == second.intercept


def test_ridge_without_cuda_fails_without_cpu_substitution(tmp_path, monkeypatch):
    manifest = training_corpus(tmp_path / "data")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA|cuda"):
        module().run_tabular_reference(manifest, tmp_path / "run", kind="ridge")
    assert not (tmp_path / "run").exists()


def test_existing_tabular_run_is_not_overwritten(tmp_path):
    manifest = training_corpus(tmp_path / "data")
    output = tmp_path / "run"
    module().run_tabular_reference(manifest, output, kind="ridge")
    before = sha256(output / "run.json")
    with pytest.raises(ValueError, match="nuev|existente"):
        module().run_tabular_reference(manifest, output, kind="ridge")
    assert sha256(output / "run.json") == before
