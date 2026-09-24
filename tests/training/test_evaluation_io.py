"""Contratos de evaluación con un modelo mínimo en CPU, sin carga científica CUDA."""

import importlib

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.training.checkpoints import StopRequest


class EvaluationRows:
    manifest = {"counts": {"validation": 3}}

    def batches(self, **_options):
        for start, end in ((0, 2), (2, 3)):
            yield dict(
                inputs={"prices": np.array([1, 3, 6], dtype=np.float32)[start:end]},
                sample_ids=["US/A/1", "US/B/1", "CN/C/2"][start:end],
                market=np.array(["US", "US", "CN"])[start:end],
                prediction_at=np.array([1, 1, 2], dtype=np.int64)[start:end],
                target=np.zeros(end - start, dtype=np.float32),
            )


class Echo(torch.nn.Module):
    def forward(self, values):
        return values["prices"]


@pytest.fixture
def engine(monkeypatch):
    module = importlib.import_module("mars_titan.training.reference_run")
    monkeypatch.setattr(
        module,
        "_inputs",
        lambda batch, _device: {"prices": torch.from_numpy(batch["inputs"]["prices"])},
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    return module


def numeric_metrics(result):
    return {
        key: value
        for key, value in result.items()
        if key not in {"elapsed_seconds", "samples_per_second"}
    }


def test_metrics_only_avoid_arrow_and_match_saved_predictions(engine, monkeypatch, tmp_path):
    destination = tmp_path / "predictions.parquet"
    saved = engine._evaluate(Echo(), EvaluationRows(), 2, destination=destination)
    table = pq.read_table(destination)
    assert table["prediction"].to_pylist() == [1.0, 3.0, 6.0]
    assert table["target"].to_pylist() == [0.0, 0.0, 0.0]
    assert table["asset_id"].to_pylist() == ["US/A", "US/B", "CN/C"]
    assert table["sample_id"].to_pylist() == ["US/A/1", "US/B/1", "CN/C/2"]
    assert table["zero"].to_pylist() == [0.0, 0.0, 0.0]

    def reject_table(*_args, **_kwargs):
        pytest.fail("Una evaluación sin destino no necesita tablas de predicciones")

    monkeypatch.setattr(engine.pa, "table", reject_table)
    model = Echo()
    result = engine._evaluate(model, EvaluationRows(), 2)
    assert numeric_metrics(result) == numeric_metrics(saved)
    assert result["samples"] == 3
    assert result["absolute_error"] == 10.0
    assert result["squared_error"] == 46.0
    assert result["session_mae"] == 4.0
    assert result["session_mse"] == 20.5
    assert not model.training


@pytest.mark.parametrize("persist", [False, True])
def test_stopped_evaluation_keeps_destination_unpublished(engine, tmp_path, persist):
    stop = StopRequest()

    class StopAfterFirst(Echo):
        def forward(self, values):
            stop.request_stop()
            return super().forward(values)

    destination = tmp_path / "predictions.parquet" if persist else None
    with pytest.raises(engine._Pause):
        engine._evaluate(StopAfterFirst(), EvaluationRows(), 2, stop=stop, destination=destination)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("invalid", ["shape", "nonfinite", "population"])
def test_invalid_evaluation_fails_with_or_without_prediction_storage(
    engine, tmp_path, persist, invalid
):
    class Invalid(Echo):
        def forward(self, values):
            output = super().forward(values)
            if invalid == "shape":
                return output[:, None]
            return output * float("nan") if invalid == "nonfinite" else output

    rows = EvaluationRows()
    if invalid == "population":
        rows.manifest = {"counts": {"validation": 4}}
    with pytest.raises(ValueError, match="forma|finitos|población"):
        engine._evaluate(
            Invalid(), rows, 2, destination=tmp_path / "predictions.parquet" if persist else None
        )
