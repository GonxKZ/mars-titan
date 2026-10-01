"""Misma información multimodal, etiquetas maduras y límites de los lotes."""

import importlib
from datetime import UTC, datetime

import numpy as np
import pytest


def module():
    try:
        return importlib.import_module("mars_titan.models.baselines.inputs")
    except ModuleNotFoundError:
        pytest.fail("Falta la preparación común de referencias")


def test_tabular_inputs_preserve_every_modality_in_a_stable_order():
    inputs = {
        "prices": np.array([[1, 2], [3, 4]]),
        "news": np.array([5]),
        "charts": np.array([6]),
        "fundamentals": np.array([7]),
        "macro": np.array([8]),
    }
    np.testing.assert_array_equal(module().feature_vector(inputs), np.arange(1, 9))
    del inputs["news"]
    with pytest.raises(ValueError):
        module().feature_vector(inputs)


def test_batches_exclude_other_partitions_and_leave_a_partial_last_batch():
    train, validation = datetime(2022, 1, 3, tzinfo=UTC), datetime(2023, 1, 3, tzinfo=UTC)
    records = [
        {
            "cursor": ("US/A", i),
            "prediction_at": day,
            "inputs": {
                name: np.array([i])
                for name in ("prices", "news", "charts", "fundamentals", "macro")
            },
        }
        for i, day in enumerate([train, validation])
    ]
    targets = {
        "US/A": {train.isoformat(): (0.5, "train"), validation.isoformat(): (0.8, "validation")}
    }
    batches = list(module().tabular_batches(records, targets, "train", batch_size=4))
    assert len(batches) == 1
    x, y = batches[0]
    assert x.shape == (1, 5)
    assert y.tolist() == [0.5]


def test_nonfinite_values_cannot_be_masked_by_concatenation():
    inputs = {name: np.ones(1) for name in ("prices", "news", "charts", "fundamentals", "macro")}
    inputs["macro"][0] = np.nan
    with pytest.raises(ValueError):
        module().feature_vector(inputs)
