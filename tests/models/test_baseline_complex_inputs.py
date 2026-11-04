"""Las referencias reales rechazan complejos antes de perder parte de su valor."""

import warnings

import numpy as np
import pytest

from mars_titan.models.baselines.inputs import MODALITIES, feature_vector, validated_blocks


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("imaginary", [0.0, 2.0])
@pytest.mark.parametrize("field", ["features", "target"])
@pytest.mark.parametrize("keep_float32", [False, True])
def test_complex_regression_blocks_fail_before_casting(field, imaginary, dtype, keep_float32):
    x, y = np.array([[3.0]], dtype=np.float32), np.array([4.0], dtype=np.float64)
    if field == "features":
        x = np.array([[3.0 + imaginary * 1j]], dtype=dtype)
    else:
        y = np.array([4.0 + imaginary * 1j], dtype=dtype)
    before_x, before_y = x.copy(), y.copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="reales"):
            next(validated_blocks(lambda: iter([(x, y)]), keep_float32=keep_float32))
    assert not caught
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(y, before_y)


@pytest.mark.parametrize("modality", MODALITIES)
@pytest.mark.parametrize("imaginary", [0.0, 2.0])
def test_each_complex_modality_is_rejected_without_a_partial_conversion(modality, imaginary):
    values = {name: np.array([float(index)]) for index, name in enumerate(MODALITIES)}
    values[modality] = np.array([3 + imaginary * 1j], dtype=np.complex128)
    before = {name: value.copy() for name, value in values.items()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="reales"):
            feature_vector(values)
    assert not caught
    for name in MODALITIES:
        np.testing.assert_array_equal(values[name], before[name])


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_real_strided_blocks_keep_their_values_and_requested_precision(dtype):
    x = np.arange(12, dtype=dtype).reshape(2, 6)[:, ::2]
    y = np.array([3, 4], dtype=dtype)
    actual_x, actual_y = next(validated_blocks(lambda: iter([(x, y)]), keep_float32=True))
    np.testing.assert_array_equal(actual_x, [[0, 2, 4], [6, 8, 10]])
    np.testing.assert_array_equal(actual_y, [3, 4])
    assert actual_x.dtype == (np.float32 if dtype == np.float32 else np.float64)
    assert actual_y.dtype == np.float64
