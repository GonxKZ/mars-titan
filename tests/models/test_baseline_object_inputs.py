"""Un array object tampoco puede ocultar valores complejos a las referencias reales."""

import warnings
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from mars_titan.models.baselines.inputs import MODALITIES, feature_vector, validated_blocks


@pytest.mark.parametrize("scalar_type", [complex, np.complex64, np.complex128, np.clongdouble])
@pytest.mark.parametrize("imaginary", [0.0, 2.0])
@pytest.mark.parametrize("field", ["features", "target", *MODALITIES])
def test_object_arrays_cannot_discard_complex_components(field, imaginary, scalar_type):
    value = scalar_type(3 + imaginary * 1j)
    x, y = np.array([[1.0]]), np.array([1.0])
    modalities = {name: np.array([1.0]) for name in MODALITIES}
    if field == "features":
        x = np.array([[value]], dtype=object)
    elif field == "target":
        y = np.array([value], dtype=object)
    else:
        modalities[field] = np.array([value], dtype=object)
    before_x, before_y = x.copy(), y.copy()
    before_modalities = {name: part.copy() for name, part in modalities.items()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="reales"):
            if field in {"features", "target"}:
                next(validated_blocks(lambda: [(x, y)]))
            else:
                feature_vector(modalities)
    assert not caught
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(y, before_y)
    for name, part in modalities.items():
        np.testing.assert_array_equal(part, before_modalities[name])


@pytest.mark.parametrize("value", [np.float32(1.25), Decimal("1.25"), Fraction(5, 4)])
def test_real_object_arrays_keep_their_previous_conversion(value):
    x = np.array([[value]], dtype=object)
    y = np.array([value], dtype=object)
    actual_x, actual_y = next(validated_blocks(lambda: [(x, y)]))
    np.testing.assert_array_equal(actual_x, [[1.25]])
    np.testing.assert_array_equal(actual_y, [1.25])
    modalities = {name: np.array([value], dtype=object) for name in MODALITIES}
    np.testing.assert_array_equal(feature_vector(modalities), np.full(5, 1.25))
