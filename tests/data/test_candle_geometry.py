"""Geometría lineal de velas, con referencias numéricas y contratos temporales."""

import importlib

import numpy as np
import pytest

from mars_titan.data.charts import chart_png


def module():
    try:
        return importlib.import_module("mars_titan.data.candle_geometry")
    except ModuleNotFoundError:
        pytest.fail("Falta la representación geométrica identificada")


def candles(count=64):
    return np.tile([10.0, 14.0, 8.0, 12.0], (count, 1))


def test_geometry_keeps_eight_channels_per_session_in_time_order():
    values = candles()
    values[63] = [12, 14, 8, 10]
    vector = module().geometry_vector(values, end_index=63)
    expected = np.tile([1 / 3, 1, 0, 2 / 3, 1 / 3, 1 / 3, 1 / 3, 1], (64, 1))
    expected[-1] = [2 / 3, 1, 0, 1 / 3, -1 / 3, 1 / 3, 1 / 3, 1]
    assert vector.shape == (512,) and vector.dtype == np.float32
    np.testing.assert_allclose(vector.reshape(64, 8), expected, rtol=0, atol=4e-8)


@pytest.mark.parametrize("scale", [1e-180, 0.125, 1e180])
def test_positive_price_rescaling_keeps_geometry(scale):
    original = module().geometry_vector(candles(), end_index=63)
    scaled = module().geometry_vector(candles() * scale, end_index=63)
    np.testing.assert_allclose(scaled, original, rtol=0, atol=6e-8)


@pytest.mark.parametrize("price", [1e-300, 1.0, 1e300])
def test_zero_range_keeps_observed_mask_and_finite_zero_geometry(price):
    vector = module().geometry_vector(np.full((64, 4), price), end_index=63).reshape(64, 8)
    assert np.isfinite(vector).all()
    assert np.array_equal(vector[:, :7], np.zeros((64, 7), dtype=np.float32))
    assert (vector[:, 7] == 1).all()


def test_flat_high_low_range_keeps_admitted_close_rounding():
    prices = np.full((64, 4), 10.0)
    prices[10, 3] += 5e-9
    original = prices.copy()
    vector = module().geometry_vector(prices, end_index=63, ordering_rtol=1e-9).reshape(64, 8)
    assert 0.49 < vector[10, 3] < 0.51
    assert vector[10, 5] == -vector[10, 3]
    assert np.isfinite(vector).all() and np.array_equal(prices, original)


def test_absences_keep_calendar_slots_without_creating_observations():
    present = np.ones(64, dtype=bool)
    present[[0, 5, 61, 62]] = False
    values = candles(60)
    values[-1] = [12, 14, 8, 10]
    vector = module().geometry_vector(values, end_index=59, present=present).reshape(64, 8)
    assert np.array_equal(vector[:, 7], present)
    assert (vector[~present] == 0).all() and not np.signbit(vector[~present]).any()
    assert vector[-1, 4] < 0 and (vector[present][:-1, 4] > 0).all()


def test_only_the_selected_past_window_affects_values_and_source_hash():
    values = candles(100)
    before = module().geometry_chart(values, end_index=70)
    values[:7] = np.nan
    values[71:] = np.inf
    after = module().geometry_chart(values, end_index=70)
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    values[70, 0], values[70, 3] = values[70, 3], values[70, 0]
    changed = module().geometry_chart(values, end_index=70)
    assert changed[0] != before[0] and not np.array_equal(changed[1], before[1])


def test_input_hash_distinguishes_values_hidden_by_pixel_rounding():
    before = candles()
    after = before.copy()
    after[12, 0] += 0.0001
    assert chart_png(before, end_index=63) == chart_png(after, end_index=63)
    old_hash, old = module().geometry_chart(before, end_index=63)
    new_hash, new = module().geometry_chart(after, end_index=63)
    assert old_hash != new_hash and not np.array_equal(old, new)
    scaled_hash, scaled = module().geometry_chart(before * 2, end_index=63)
    assert scaled_hash != old_hash and np.array_equal(scaled, old)


@pytest.mark.parametrize("column,value", [(3, 14 * (1 + 5e-10)), (2, 10 * (1 + 5e-10))])
def test_admitted_rounding_is_kept_without_repairing_prices(column, value):
    prices = candles()
    prices[0, column] = value
    original = prices.copy()
    with pytest.raises(ValueError, match="OHLC"):
        module().geometry_vector(prices, end_index=63)
    result = module().geometry_vector(prices, end_index=63, ordering_rtol=1e-9).reshape(64, 8)
    assert result[0, 5 if column == 3 else 6] < 0
    assert np.array_equal(prices, original)
    prices[0, column] *= 1.01
    with pytest.raises(ValueError, match="OHLC"):
        module().geometry_vector(prices, end_index=63, ordering_rtol=1e-9)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_observed_prices_are_rejected(value):
    prices = candles()
    prices[10, 0] = value
    with pytest.raises(ValueError, match="positivos|finitos"):
        module().geometry_vector(prices, end_index=63)


@pytest.mark.parametrize("end", [62, 64, -1, 63.0, True])
def test_window_end_must_identify_a_complete_past_window(end):
    with pytest.raises(ValueError, match="ventana|índice"):
        module().geometry_vector(candles(), end_index=end)


@pytest.mark.parametrize(
    "present",
    [
        np.ones(63, dtype=bool),
        np.ones(64),
        np.zeros(64, dtype=bool),
        np.array([True] * 63 + [False]),
    ],
)
def test_invalid_presence_is_rejected(present):
    with pytest.raises(ValueError, match="presencia|ventana"):
        module().geometry_vector(candles(), end_index=63, present=present)


@pytest.mark.parametrize(
    "prices", [[], np.ones((64, 3)), np.ones((1, 64, 4)), candles().astype(object)]
)
def test_input_shape_and_numeric_storage_are_explicit(prices):
    with pytest.raises(ValueError, match="OHLC|matriz"):
        module().geometry_vector(prices, end_index=63)


def test_wrapping_preserves_text_contract_and_identifies_only_the_new_image_path():
    class Base:
        spec = {
            "text_model": "fixture",
            "image_model": "fixture_png",
            "historical_simulation": False,
        }

        def text(self, text):
            return np.full(384, len(text), dtype=np.float32)

    base = Base()
    encoder = module().GeometryEncoders(base)
    assert np.array_equal(encoder.text("noticia"), base.text("noticia"))
    assert base.spec["image_model"] == "fixture_png"
    assert encoder.spec["image_model"] == "candle_geometry_linear64_v1"
    assert encoder.spec["historical_simulation"] is False
    assert encoder.spec["image_policy"]["learned_parameters"] == 0
    assert encoder.spec["image_policy"]["width"] == 512
    assert len(encoder.spec["image_policy"]["price_validation_sha256"]) == 64
    assert encoder.spec == module().GeometryEncoders(base).spec
    result = encoder.chart(candles(), end_index=63, context=64)
    assert len(result[0]) == 64 and result[1].shape == (512,)
    with pytest.raises(ValueError, match="OHLC"):
        encoder.images([b"png"])
    with pytest.raises(ValueError, match="64"):
        encoder.chart(candles(), end_index=63, context=2)
    base.spec = {**base.spec, "text_model": "changed"}
    with pytest.raises(ValueError, match="identidad"):
        encoder.text("noticia")


def test_geometry_identity_cannot_be_relabelled_after_construction():
    class Base:
        spec = {"text_model": "fixture"}

        def text(self, text):
            return np.zeros(384, dtype=np.float32)

    encoder = module().GeometryEncoders(Base())
    encoder.spec["image_policy"]["width"] = 256
    with pytest.raises(ValueError, match="identidad"):
        encoder.chart(candles(), end_index=63)


@pytest.mark.parametrize("base", [None, object(), type("Invalid", (), {"spec": []})()])
def test_composition_rejects_an_unidentified_text_encoder(base):
    with pytest.raises(ValueError, match="identificado"):
        module().GeometryEncoders(base)


def test_composition_cannot_publish_placeholder_text_from_a_collector():
    from mars_titan.data.vector_carry import CollectingEncoders, ReuseOnlyEncoders

    for base in (CollectingEncoders({"name": "fixture"}), ReuseOnlyEncoders({"name": "fixture"})):
        with pytest.raises(ValueError, match="pasada"):
            module().GeometryEncoders(base)
