"""Conservar los canales de precios de cada edición al materializar una medida."""

import importlib
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.mark.parametrize("channels", [5, 6])
def test_materialization_preserves_every_price_channel_without_reading_targets(
    tmp_path, monkeypatch, channels
):
    module = importlib.import_module("benchmarks.titans_paper_projections_cost")
    at = 1_600_000_000_000_000
    dates = np.array([at, at + 1], dtype="datetime64[us]")
    source = tmp_path / "view.json"
    source.write_text("fixture")
    sample = tmp_path / "samples.parquet"
    pq.write_table(
        pa.table(dict(prediction_at=pa.array(dates, type=pa.timestamp("us", tz="UTC")))), sample
    )
    vectors = {
        name: np.arange(6, dtype=np.float32).reshape(2, 3)
        for name in ("news", "charts", "fundamentals", "macro")
    }
    decoded = (dates, np.arange(2), vectors, None, None, None)

    class Dataset:
        def __init__(self, *args, **kwargs):
            self.context = 2
            self.price_channels = channels
            self.identity = "a" * 64
            self.assets = [dict(market="US", symbol=name) for name in ("A", "B")]
            self.manifest = dict(representation=dict(kind="fixture"))

        def _file(self, asset, kind):
            assert kind in {"samples", "prices"}
            return sample

        def _prices(self, asset):
            return None

        def _sample_group(self, *args):
            return None, *decoded

    def observation(dataset, asset, values, row, instant, prices):
        assert dataset.price_channels == channels
        return dict(
            rows=np.array([row]),
            vectors=vectors,
            prices=np.broadcast_to(np.arange(channels, dtype=np.float32), (1, 2, channels)),
            prediction_at=np.array([instant], dtype="datetime64[us]"),
            sample_at=np.array([instant], dtype="datetime64[us]"),
            price_available_at=np.array([instant - 1], dtype="datetime64[us]"),
            input_available_at=np.array([instant - 1], dtype="datetime64[us]"),
            availability_valid=np.ones(1, dtype=bool),
            presence=np.ones((1, 5), dtype=bool),
            key=f"US/{asset['symbol']}",
        )

    monkeypatch.setattr(module, "CorpusDataset", Dataset)
    monkeypatch.setattr(module, "_view_fold", lambda dataset: {})
    monkeypatch.setattr(
        module,
        "window_phases",
        lambda *args: dict(
            train=SimpleNamespace(
                partition="train", decision_start=at - 1000, decision_end=at + 1000
            )
        ),
    )
    monkeypatch.setattr(module, "_observation", observation)
    output = tmp_path / "inputs.npz"
    record = module.materialize(source, output, flows=2, instants=2, warmup_months=12)
    assert record["dimensions"]["prices"] == channels
    assert record["targets_read"] is False
    with np.load(output) as arrays:
        assert not any("target" in key for key in arrays.files)
        for instant in range(2):
            actual = arrays[f"{instant}/inputs/prices"]
            assert actual.shape == (2, 2, channels)
            np.testing.assert_array_equal(
                actual, np.broadcast_to(np.arange(channels), actual.shape)
            )
