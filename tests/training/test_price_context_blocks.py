"""Paridad de las ventanas agrupadas con la transformación original por fila."""

import numpy as np
import pytest

from mars_titan.data.streaming import price_features
from tests.training.test_corpus_inputs import corpus, module


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize("context", [2, 64, 512])
def test_block_transformation_matches_each_window_exactly(dtype, context):
    rng = np.random.default_rng(31)
    prices = rng.integers(1, 10000, size=(1024, 5)).astype(dtype)
    prices[:context, 4] = 0
    ends = np.r_[context - 1, rng.integers(context - 1, len(prices), size=255)]
    observed = module()._price_contexts(prices, ends, context)
    expected = np.stack([price_features(prices[e - context + 1 : e + 1]) for e in ends])
    assert observed.dtype == np.float32 and observed.flags.owndata
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize(
    "problem", ["nan", "negative_volume", "zero_price", "end", "float_end", "size"]
)
def test_invalid_selected_windows_fail_before_training(problem):
    prices = np.ones((64, 5))
    ends = np.array([63])
    if problem == "nan":
        prices[10, 0] = np.nan
    elif problem == "negative_volume":
        prices[10, 4] = -1
    elif problem == "zero_price":
        prices[10, 2] = 0
    elif problem == "end":
        ends[0] = 64
    elif problem == "float_end":
        ends = ends.astype(float)
    else:
        ends = np.repeat(ends, 257)
    with pytest.raises(ValueError):
        module()._price_contexts(prices, ends, 64)


def test_unselected_price_history_is_not_used_by_the_block():
    prices = np.ones((100, 5))
    prices[:25] = np.nan
    observed = module()._price_contexts(prices, np.array([99]), 64)
    np.testing.assert_array_equal(observed[0], price_features(prices[36:100]))


def test_unsigned_end_indices_are_checked_before_converting_to_native_indices():
    prices = np.ones((64, 5))
    np.testing.assert_array_equal(
        module()._price_contexts(prices, np.array([63], dtype=np.uint64), 64)[0],
        price_features(prices),
    )
    with pytest.raises(ValueError):
        module()._price_contexts(prices, np.array([2**64 - 1], dtype=np.uint64), 64)


def test_grouping_preserves_full_batches_cursor_and_modalities(tmp_path, monkeypatch):
    import json
    from datetime import timedelta

    import pyarrow as pa
    import pyarrow.parquet as pq

    from mars_titan.data.storage import sha256

    manifest = corpus(tmp_path, assets=3, rows=545, group_size=300)
    metadata = json.loads(manifest.read_text())
    for asset in metadata["assets"]:
        prices_path = tmp_path / "prepared/US" / asset["symbol"] / "prices.parquet"
        first_time = pq.read_table(prices_path)["available_at"][0].as_py()
        pq.write_table(
            pa.Table.from_pylist(
                [
                    dict(
                        open=float(10 + i),
                        high=float(13 + i),
                        low=float(9 + i),
                        close=float(11 + i),
                        volume=float(100 + i),
                        available_at=first_time + timedelta(minutes=i),
                    )
                    for i in range(548)
                ]
            ),
            prices_path,
        )
        samples_path = tmp_path / "samples/US" / asset["symbol"] / "samples.parquet"
        rows = pq.read_table(samples_path).to_pylist()
        for i, row in enumerate(rows):
            row["price_end_index"] = i + 2
        pq.write_table(pa.Table.from_pylist(rows), samples_path, row_group_size=300)
        asset.update(prices_sha256=sha256(prices_path), samples_sha256=sha256(samples_path))
    manifest.write_text(json.dumps(metadata))
    engine = module()
    options = dict(partition="train", batch_size=129, epoch=2, seed=43)
    optimized = list(engine.CorpusDataset(manifest).batches(**options))
    expected_prices = np.stack(
        [
            price_features(
                np.array(
                    [[10 + k, 13 + k, 9 + k, 11 + k, 100 + k] for k in (i + 1, i + 2)], dtype=float
                )
            )
            for i in range(545)
        ]
    )
    for batch in optimized:
        sample_rows = batch["inputs"]["news"][:, 0].astype(int)
        np.testing.assert_array_equal(batch["inputs"]["prices"], expected_prices[sample_rows])

    def reference(prices, ends, context):
        return np.stack([price_features(prices[e - context + 1 : e + 1]) for e in ends])

    monkeypatch.setattr(engine, "_price_contexts", reference)
    original = list(engine.CorpusDataset(manifest).batches(**options))
    for a, b in zip(optimized, original, strict=True):
        assert a["sample_ids"] == b["sample_ids"] and a["confirmed_cursor"] == b["confirmed_cursor"]
        for name in a["inputs"]:
            np.testing.assert_array_equal(a["inputs"][name], b["inputs"][name])
        for name in ("target", "prediction_at", "target_available_at", "input_available_at"):
            np.testing.assert_array_equal(a[name], b[name])
