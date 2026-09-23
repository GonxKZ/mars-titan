"""Lotes del adaptador con padre alineado, épocas completas y cursor confirmado."""

import importlib

import numpy as np
import pytest

from tests.training.test_predictive_parents import setup

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("duckdb", "gymnasium")),
    reason="Requiere los extras data y reinforcement",
)


def module():
    return importlib.import_module("mars_titan.training.predictive_inputs")


def dataset(tmp_path):
    from mars_titan.training.predictive_parents import prepare_parent_cache

    ordered, parent, _ = setup(tmp_path)
    cache = tmp_path / "cache"
    prepare_parent_cache(ordered, parent, cache)
    return module().PredictiveDataset(ordered, cache / "manifest.json")


def test_every_row_and_modality_survives_epoch_shuffle_and_resume(tmp_path):
    with dataset(tmp_path) as data:
        options = dict(partition="train", batch_size=5, epoch=0, seed=42)
        whole = list(data.batches(**options))
        assert [len(batch["target"]) for batch in whole] == [5, 5, 2]
        keys = [key for batch in whole for key in batch["sample_ids"]]
        assert len(keys) == len(set(keys)) == 12
        assert whole[0]["features"].shape == (5, 326)
        assert whole[0]["features"].dtype == np.float32
        np.testing.assert_allclose(whole[0]["features"][:, -1], whole[0]["parent"], rtol=1e-7)
        resumed = list(data.batches(**options, cursor=whole[0]["confirmed_cursor"]))
        for left, right in zip(whole[1:], resumed, strict=True):
            for name in ("features", "target", "parent", "prediction_at", "weight"):
                np.testing.assert_array_equal(left[name], right[name])
            assert left["sample_ids"] == right["sample_ids"]
        other = [
            key
            for batch in data.batches(**(options | dict(epoch=1)))
            for key in batch["sample_ids"]
        ]
        assert keys != other and set(keys) == set(other)
        assert list(data.batches(**options, cursor=whole[-1]["confirmed_cursor"])) == []
        with pytest.raises(ValueError, match="cursor"):
            list(data.batches(**(options | dict(seed=1)), cursor=whole[0]["confirmed_cursor"]))


def test_scaler_uses_training_only_and_preserves_constant_columns(tmp_path):
    with dataset(tmp_path) as data:
        expected = np.concatenate(
            [
                batch["features"]
                for batch in data.batches(
                    partition="train",
                    batch_size=5,
                    epoch=0,
                    seed=0,
                )
            ]
        ).astype(np.float64)
        stats = module().fit_standardizer(data, batch_size=5)
        assert stats["samples"] == 12 and stats["fit_partition"] == "train"
        np.testing.assert_allclose(stats["mean"], expected.mean(axis=0), rtol=1e-12, atol=1e-12)
        variable = expected.var(axis=0) > 1e-20
        np.testing.assert_allclose(
            np.array(stats["scale"])[variable],
            expected.std(axis=0)[variable],
            rtol=1e-10,
            atol=1e-12,
        )
        assert np.all(np.array(stats["scale"])[~variable] == 1)


def test_batches_own_their_arrays_and_reader_closure_is_explicit(tmp_path):
    data = dataset(tmp_path)
    iterator = data.batches(partition="train", batch_size=5, epoch=0, seed=42)
    first = next(iterator)
    before = first["features"].copy()
    next(iterator)
    np.testing.assert_array_equal(before, first["features"])
    data.close()
    with pytest.raises(ValueError, match="cerrad"):
        list(data.batches(partition="train", batch_size=5, epoch=0, seed=42))


def test_closing_a_reader_also_stops_an_existing_iterator(tmp_path):
    data = dataset(tmp_path)
    iterator = data.batches(partition="train", batch_size=1, epoch=0, seed=42)
    next(iterator)
    data.close()
    with pytest.raises(ValueError, match="cerrad"):
        next(iterator)
