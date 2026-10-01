"""Ventajas por entorno, límites de episodio y rechazo de datos inválidos."""

import numpy as np
import pytest

from mars_titan.simulation.algorithms import generalized_advantage


def test_gae_keeps_terminal_truncation_and_continuation_separate_per_environment():
    advantages, returns = generalized_advantage(
        [[1, 2, 3], [10, 20, 30]],
        [[4, 5, 6], [7, 8, 9]],
        [[8, 10, 12], [14, 16, 18]],
        [[True, False, False], [False, False, False]],
        [[False, True, False], [True, True, True]],
        gamma=0.5,
        lam=0.5,
    )
    np.testing.assert_array_equal(advantages, [[-3, 2, 10.5], [10, 20, 30]])
    np.testing.assert_array_equal(returns, [[1, 7, 16.5], [17, 28, 39]])


def test_gae_does_not_carry_rewards_between_environment_columns():
    advantages, returns = generalized_advantage(
        [[1, 100], [2, 200]],
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.zeros((2, 2), dtype=bool),
        [[False, False], [True, True]],
        gamma=1,
        lam=1,
    )
    np.testing.assert_array_equal(advantages, [[3, 300], [2, 200]])
    np.testing.assert_array_equal(returns, [[3, 300], [2, 200]])


def test_single_time_step_keeps_only_the_admitted_bootstrap():
    advantages, returns = generalized_advantage(
        [[2, 2, 2]],
        [[1, 1, 1]],
        [[4, 4, 4]],
        [[False, True, False]],
        [[False, False, True]],
        gamma=0.5,
    )
    np.testing.assert_array_equal(advantages, [[3, 1, 3]])
    np.testing.assert_array_equal(returns, [[4, 2, 4]])


@pytest.mark.parametrize(
    "gamma,lam,expected", [(0, 1, [1, 10]), (0.5, 0, [1, 10]), (0.5, 1, [6, 10]), (1, 1, [11, 10])]
)
def test_gae_accepts_discount_boundaries_without_changing_the_time_axis(gamma, lam, expected):
    advantages, _ = generalized_advantage(
        [[1], [10]],
        [[0], [0]],
        [[0], [0]],
        [[False], [False]],
        [[False], [True]],
        gamma=gamma,
        lam=lam,
    )
    np.testing.assert_array_equal(advantages[:, 0], expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_gae_matches_each_separate_environment_with_read_only_strided_inputs(dtype):
    rng = np.random.default_rng(20260928)
    arrays = [rng.normal(size=(17, 6)).astype(dtype)[:, ::2] for _ in range(3)]
    arrays.extend([rng.random((17, 6))[:, ::2] < 0.2 for _ in range(2)])
    for array in arrays:
        array.setflags(write=False)
    before = [array.copy() for array in arrays]
    advantages, returns = generalized_advantage(*arrays, gamma=np.float64(0.9), lam=0.7)
    assert advantages.shape == returns.shape == (17, 3)
    assert advantages.dtype == returns.dtype == np.float64
    for lane in range(3):
        expected = generalized_advantage(*(array[:, lane] for array in arrays), gamma=0.9, lam=0.7)
        np.testing.assert_array_equal(advantages[:, lane], expected[0])
        np.testing.assert_array_equal(returns[:, lane], expected[1])
    for actual, original in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("shape", [(), (0,), (0, 2), (2, 0), (2, 2, 1)])
def test_gae_rejects_empty_or_unsupported_dimensions(shape):
    arrays = [np.zeros(shape) for _ in range(3)] + [np.zeros(shape, dtype=bool) for _ in range(2)]
    with pytest.raises(ValueError, match="dimensiones"):
        generalized_advantage(*arrays)


@pytest.mark.parametrize("field", range(1, 5))
def test_gae_rejects_broadcasting_of_mismatched_rollout_fields(field):
    arrays = [np.zeros((2, 2)) for _ in range(3)] + [np.zeros((2, 2), dtype=bool) for _ in range(2)]
    arrays[field] = arrays[field][:, :1]
    with pytest.raises(ValueError, match="dimensiones"):
        generalized_advantage(*arrays)


@pytest.mark.parametrize("field", [3, 4])
@pytest.mark.parametrize("mask", [[0, 1], [0.0, 1.0], [False, np.nan], ["", "True"]])
def test_gae_rejects_masks_that_are_not_boolean(field, mask):
    arrays = [[1, 2], [0, 0], [0, 0], [False, True], [False, False]]
    arrays[field] = mask
    with pytest.raises(ValueError, match="boolean"):
        generalized_advantage(*arrays)


@pytest.mark.parametrize("field", range(3))
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_gae_rejects_nonfinite_data_even_at_a_terminal_transition(field, invalid):
    arrays = [np.zeros(2) for _ in range(3)] + [np.ones(2, dtype=bool) for _ in range(2)]
    arrays[field][1] = invalid
    with pytest.raises(ValueError, match="finit"):
        generalized_advantage(*arrays)


@pytest.mark.parametrize("field", range(3))
@pytest.mark.parametrize("invalid", [["1", "2"], [1j, 2j], [True, False]])
def test_gae_rejects_nonreal_rollout_data(field, invalid):
    arrays = [[1, 2], [0, 0], [0, 0], [False, True], [False, False]]
    arrays[field] = invalid
    with pytest.raises(ValueError, match="real"):
        generalized_advantage(*arrays)


@pytest.mark.parametrize("parameter", ["gamma", "lam"])
@pytest.mark.parametrize("invalid", [-0.1, 1.1, np.nan, np.inf, True, "0.5", [0.5]])
def test_gae_rejects_invalid_discounts_before_accumulating(parameter, invalid):
    with pytest.raises(ValueError, match="gamma|lambda"):
        generalized_advantage([1], [0], [0], [False], [False], **{parameter: invalid})


def test_gae_rejects_overflow_from_finite_inputs():
    maximum = np.finfo(np.float64).max
    with pytest.raises(FloatingPointError):
        generalized_advantage([maximum], [-maximum], [0], [False], [True])
