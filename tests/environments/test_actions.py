"""Rejilla predictiva fijada con entrenamiento y pérdidas conocidas de todas las acciones."""

import importlib

import numpy as np
import pytest


def grid_module():
    return importlib.import_module("mars_titan.environments.actions")


def test_grid_uses_only_identified_training_and_has_twenty_one_ordered_actions():
    targets = np.linspace(-0.1, 0.1, 101)
    original = targets.copy()
    grid = grid_module().ActionGrid.fit(targets, source_sha256="a" * 64, partition="train")
    assert grid.values.shape == (21,) and np.all(np.diff(grid.values) > 0)
    assert grid.values[10] == 0
    assert grid.scale == pytest.approx(np.abs(targets).mean())
    assert grid.values[[0, -1]] == pytest.approx([-0.098, 0.098])
    assert grid.training_samples == 101
    np.testing.assert_array_equal(targets, original)
    with pytest.raises(ValueError, match="entrenamiento"):
        grid_module().ActionGrid.fit(targets, source_sha256="a" * 64, partition="validation")


def test_lower_median_and_exact_expected_loss_do_not_replace_mae_with_mean():
    grid = grid_module().ActionGrid.fit(
        np.linspace(-1, 1, 101), source_sha256="a" * 64, partition="train"
    )
    probabilities = np.zeros((2, 21))
    probabilities[0, [0, 20]] = [0.5, 0.5]
    probabilities[1, [0, 20]] = [0.49, 0.51]
    np.testing.assert_array_equal(grid.median(probabilities), [grid.values[0], grid.values[-1]])
    target = np.array([0.0, 0.2])
    expected = (probabilities * np.abs(grid.values - target[:, None])).sum(axis=1) / grid.scale
    np.testing.assert_allclose(grid.expected_loss(probabilities, target), expected)
    np.testing.assert_allclose(
        grid.reward(np.array([0, 20]), target), -np.abs(grid.values[[0, 20]] - target) / grid.scale
    )


def test_zero_training_targets_keep_a_finite_nonzero_scale_and_roundtrip():
    grid = grid_module().ActionGrid.fit(np.zeros(10), source_sha256="b" * 64, partition="train")
    assert grid.scale > 0 and len(np.unique(grid.values)) == 21
    restored = grid_module().ActionGrid.from_dict(grid.to_dict())
    assert restored.to_dict() == grid.to_dict()
    assert restored.saturation(np.array([-1.0, 0.0, 1.0])) == {"below": 1, "above": 1, "samples": 3}


@pytest.mark.parametrize(
    "probabilities", [np.ones((1, 21)), np.full((1, 21), np.nan), np.zeros((1, 20))]
)
def test_invalid_distributions_are_rejected(probabilities):
    grid = grid_module().ActionGrid.fit(
        np.arange(10) / 100, source_sha256="a" * 64, partition="train"
    )
    with pytest.raises(ValueError):
        grid.median(probabilities)


def test_grid_fails_before_allocating_an_oversized_copy():
    targets = np.broadcast_to(np.array([1.0]), (17_000_000,))
    with pytest.raises(ValueError, match="presupuesto"):
        grid_module().ActionGrid.fit(targets, source_sha256="a" * 64, partition="train")


def test_softmax_float32_rounding_is_normalized_without_accepting_missing_mass():
    grid = grid_module().ActionGrid.fit(
        np.arange(10) / 100, source_sha256="a" * 64, partition="train"
    )
    probabilities = np.full((1, 21), 1 / 21, dtype=np.float32)
    assert grid.median(probabilities).shape == (1,)
    with pytest.raises(ValueError):
        grid.median(probabilities * 0.9)


def test_median_resolves_distributed_tie_without_confusing_nearby_mass():
    grid = grid_module().ActionGrid.fit(
        np.linspace(-1, 1, 101), source_sha256="a" * 64, partition="train"
    )
    probabilities = np.zeros((2, 21))
    probabilities[:, :10] = 0.05
    probabilities[:, 20] = 0.5
    probabilities[1, 0] -= 1e-13
    probabilities[1, 20] += 1e-13
    np.testing.assert_array_equal(grid.median(probabilities), grid.values[[9, 20]])
