"""Contextos artificiales identificados para comprobar S2, sin aprendizaje ni datos reales."""

import importlib
import io
from dataclasses import replace

import numpy as np
import pytest
import torch

AT = 1_609_459_200_000_000


def api():
    try:
        return importlib.import_module("mars_titan.memory.failure_context")
    except ModuleNotFoundError:
        pytest.fail("Falta el contexto inmutable del libro S2")


def layout(**changes):
    values = dict(
        input_id="a" * 64,
        codec_id="b" * 64,
        macro_catalog_id="c" * 64,
        macro_indicators=("fixture_macro_0", "fixture_macro_1"),
        macro_families=("family_0", "family_1"),
        family_names=tuple(f"family_{i}" for i in range(12)),
        area_names=("prices", "macro", "slot"),
        area_modalities=("prices", "macro", None),
        key_width=3,
    )
    return api().FailureContextLayout(**(values | changes))


def arguments(**changes):
    values = dict(
        flow_id="US/AAA",
        sample_id="US/AAA/fixture-1",
        prediction_at=AT,
        input_available_at=AT - 1,
        input_sha256="d" * 64,
        key_inputs=np.array([3.0, 4.0, 0.0], dtype=np.float64),
        macro=np.array([1.0, 2.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        presence=np.array([True, False, True, False, True]),
        action="fixed_consumer/no_tool",
    )
    return values | changes


def context(**changes):
    return api().FailureContext(layout(), **arguments(**changes))


def test_the_context_captures_original_bytes_and_the_actual_area_participation():
    values = arguments()
    record = api().FailureContext(layout(), **values)
    values["macro"][:] = 0
    values["key_inputs"][:] = 0
    values["presence"][:] = False
    np.testing.assert_array_equal(record.macro, [1, 2, 1, 1, 0, 0])
    np.testing.assert_array_equal(record.key_inputs, [3, 4, 0])
    np.testing.assert_array_equal(record.presence, [True, False, True, False, True])
    np.testing.assert_array_equal(record.area_activity, [True, True, True])
    np.testing.assert_allclose(record.normalized_key(), [0.6, 0.8, 0], rtol=0, atol=1e-15)
    assert np.all(record.macro_available_at == np.iinfo(np.int64).min)
    for array in (record.macro, record.key_inputs, record.presence, record.area_activity):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0


@pytest.mark.parametrize(
    "changes",
    [
        dict(input_available_at=AT + 1),
        dict(prediction_at=1_704_067_200_000_000),
        dict(prediction_at=True),
        dict(flow_id="US/../AAA"),
        dict(input_sha256="unknown"),
        dict(macro_available_at=np.array([AT + 1, AT - 1], dtype=np.int64)),
    ],
)
def test_future_or_unidentified_information_is_rejected(changes):
    assert context().prediction_at == AT
    with pytest.raises(ValueError):
        context(**changes)


@pytest.mark.parametrize(
    "macro,presence",
    [
        ([1, 2, 0.5, 1, 0, 0], [True, False, True, False, True]),
        ([1, 2, 0, 1, 0, 0], [True, False, True, False, True]),
        ([1, 2, 1, 1, -1, 0], [True, False, True, False, True]),
        ([0, 0, 0, 0, 0, 0], [True, False, True, False, True]),
        ([1, 2, 1, 1, 0, 0], [True, False, True, False, False]),
    ],
)
def test_the_existing_numeric_absence_contract_is_kept(macro, presence):
    with pytest.raises(ValueError):
        context(macro=np.array(macro, dtype=np.float32), presence=np.array(presence))


def test_known_publications_are_optional_but_never_invented_for_missing_concepts():
    dates = np.array([AT - 10, np.iinfo(np.int64).min], dtype=np.int64)
    record = context(macro_available_at=dates)
    np.testing.assert_array_equal(record.macro_available_at, dates)
    macro = np.array([0, 2, 0, 1, 0, 0], dtype=np.float32)
    with pytest.raises(ValueError, match="publicación"):
        context(macro=macro, macro_available_at=dates)


@pytest.mark.parametrize(
    "key",
    [
        np.zeros(3),
        np.array([np.nan, 1, 0]),
        np.array([np.inf, 1, 0]),
        np.ones(4),
        np.ones(3, dtype=np.int64),
    ],
)
def test_key_geometry_is_explicit_and_finite(key):
    with pytest.raises(ValueError):
        context(key_inputs=key)


def test_large_finite_keys_are_normalized_without_squaring_them_first():
    record = context(key_inputs=np.array([1e308, -1e308, 0]))
    value = record.normalized_key()
    assert np.isfinite(value).all()
    np.testing.assert_allclose(value @ value, 1, rtol=0, atol=5e-16)
    assert value[0] > 0 and value[1] < 0


def test_contexts_cannot_carry_a_financial_label():
    with pytest.raises(TypeError):
        api().FailureContext(layout(), **arguments(), target=0.01)


def test_roundtrip_and_version_mismatch_are_checked_before_use():
    original = context()
    payload = original.export()
    restored = api().FailureContext.restore(layout(), payload)
    assert restored.context_id == original.context_id
    assert restored.export() == payload
    with pytest.raises(ValueError, match="diseño"):
        api().FailureContext.restore(replace(layout(), codec_id="e" * 64), payload)
    damaged = dict(payload, action="another_tool")
    with pytest.raises(ValueError, match="huella"):
        api().FailureContext.restore(layout(), damaged)


@pytest.mark.parametrize(
    "changes",
    [
        dict(family_names=tuple(f"family_{i}" for i in range(11))),
        dict(macro_families=("unknown", "family_1")),
        dict(area_modalities=("prices", "unknown", None)),
        dict(key_width=0),
        dict(macro_indicators=("same", "same")),
    ],
)
def test_layouts_have_unique_and_bounded_meaning(changes):
    with pytest.raises(ValueError):
        layout(**changes)


def test_publication_dates_respect_the_input_cutoff_even_before_the_prediction():
    with pytest.raises(ValueError, match="disponibilidad"):
        context(macro_available_at=np.array([AT, AT - 10], dtype=np.int64))


def test_an_accidental_change_is_detected_at_serialization_and_query_boundaries():
    value = context()
    object.__setattr__(value, "action", "modified")
    for operation in (value.verify, value.export, value.normalized_key):
        with pytest.raises(ValueError, match="huella"):
            operation()


def test_the_context_round_trips_through_the_existing_tensor_archive_format():
    value = context()
    buffer = io.BytesIO()
    torch.save(value.export(), buffer)
    buffer.seek(0)
    restored = api().FailureContext.restore(
        layout(), torch.load(buffer, map_location="cpu", weights_only=True)
    )
    assert restored.export() == value.export()


@pytest.mark.parametrize(
    "changes",
    [
        dict(sample_id=""),
        dict(sample_id="x" * 1025),
        dict(action="bad\nmechanism"),
        dict(macro=np.zeros(5, dtype=np.float32)),
        dict(macro=np.zeros(6, dtype=np.float64)),
        dict(presence=np.ones(5, dtype=np.int64)),
        dict(macro_available_at=np.zeros(2, dtype=np.float64)),
    ],
)
def test_context_boundaries_reject_malformed_identifiers_and_buffers(changes):
    with pytest.raises(ValueError):
        context(**changes)


def test_a_context_needs_its_typed_layout_and_accepts_float32_codec_keys():
    with pytest.raises(ValueError, match="diseño"):
        api().FailureContext(None, **arguments())
    value = context(key_inputs=np.array([3, 4, 0], dtype=np.float32))
    np.testing.assert_array_equal(value.key_inputs, [3, 4, 0])
    no_areas = layout(area_names=(), area_modalities=())
    assert not api().FailureContext(no_areas, **arguments()).area_activity.size


@pytest.mark.parametrize(
    "field", ["key_inputs", "macro", "presence", "macro_available_at", "area_activity"]
)
def test_restored_buffers_must_keep_their_exact_length(field):
    payload = context().export()
    payload[field] += b"x"
    with pytest.raises(ValueError, match="buffers"):
        api().FailureContext.restore(layout(), payload)


def test_absent_macro_keeps_its_areas_inactive_and_the_shared_slot_active():
    record = context(
        macro=np.zeros(6, dtype=np.float32),
        presence=np.array([True, False, True, False, False]),
    )
    np.testing.assert_array_equal(record.area_activity, [True, False, True])
    assert not record.macro.any()


def test_macro_dtype_is_checked_even_when_its_numeric_contract_is_valid():
    assert context().macro.dtype == np.float32
    with pytest.raises(ValueError):
        context(macro=arguments()["macro"].astype(np.float64))
