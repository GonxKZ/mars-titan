"""Codec fijo por fila con inputs manuales, sin ajustar señales ni parámetros."""

import importlib
import math
import os

import numpy as np
import pytest
import torch
from test_historical_inputs import create, raw_batch, specification

from mars_titan.data.input_policy import MODALITIES
from mars_titan.models.candidate.input_adapter import CandidateInputAdapter
from mars_titan.models.titans.financial_inputs import DecisionBatch, validated_cpu_batch

pytestmark = pytest.mark.skipif(
    not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"), reason="Falta el enlace nativo compilado"
)


def codec_class():
    try:
        return importlib.import_module(
            "mars_titan.models.candidate.episode_codec"
        ).FrozenCandidateCodec
    except ModuleNotFoundError:
        pytest.fail("Falta el codec CPU fijo del candidato")


def reference(batch, dtype, config):
    tensors = DecisionBatch.from_validated(batch, device="cpu", dtype=dtype)
    pieces = [tensors.inputs["prices"].flatten(1)]
    pieces.extend(tensors.inputs[name] for name in MODALITIES[1:])
    if config.input_policy == "historical_masked_2000_v1":
        pieces.append(tensors.presence.to(dtype))
    inputs = torch.cat(pieces, dim=1)
    feature_generator = torch.Generator(device="cpu").manual_seed(config.feature_seed)
    key_generator = torch.Generator(device="cpu").manual_seed(config.key_seed)
    width = inputs.shape[1]
    feature = torch.empty((width, 256), dtype=dtype, device="cpu").normal_(
        0, 1 / math.sqrt(width), generator=feature_generator
    )
    key = torch.empty((256, 128), dtype=dtype, device="cpu").normal_(
        0, 1 / math.sqrt(256), generator=key_generator
    )
    values, keys = [], []
    for row in inputs:
        value = row[None] @ feature
        projected = value @ key
        scale = projected.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        scaled = projected / scale
        normalized = scaled / scaled.norm(2, dim=-1, keepdim=True).clamp_min(1e-12 / scale)
        values.append(value)
        keys.append(normalized)
    return torch.cat(keys).numpy(), torch.cat(values).numpy()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_codec_matches_fixed_projection_formula_without_learned_weights(dtype):
    adapter, batch = create(dtype=dtype)
    before = torch.random.get_rng_state().clone()
    codec = codec_class()(adapter)
    encoded = codec.encode(batch)
    expected_keys, expected_values = reference(batch, dtype, adapter.model.config)
    np.testing.assert_array_equal(encoded.keys, expected_keys)
    np.testing.assert_array_equal(encoded.values, expected_values)
    assert encoded.keys.shape == (2, 128)
    assert encoded.values.shape == (2, 256)
    other = CandidateInputAdapter(specification(), dtype=dtype, parameter_seed=202)
    alternate = codec_class()(other)
    assert other.model.parameter_fingerprint() != adapter.model.parameter_fingerprint()
    assert alternate.identity() == codec.identity()
    np.testing.assert_array_equal(alternate.encode(batch).keys, expected_keys)
    np.testing.assert_array_equal(alternate.encode(batch).values, expected_values)
    assert torch.equal(before, torch.random.get_rng_state())


def test_codec_bytes_are_invariant_to_physical_partition_and_permutation():
    adapter, batch = create()
    codec = codec_class()(adapter)
    expected = codec.encode(batch)
    raw = raw_batch()
    keys, values = [], []
    for index in (1, 0):
        selected = dict(
            inputs={name: value[[index]] for name, value in raw["inputs"].items()},
            presence=raw["presence"][[index]],
            sample_ids=[raw["sample_ids"][index]],
            prediction_at=raw["prediction_at"][[index]],
            input_available_at=raw["input_available_at"][[index]],
        )
        actual = codec.encode(validated_cpu_batch(selected, specification()))
        keys.append(actual.keys)
        values.append(actual.values)
    np.testing.assert_array_equal(np.concatenate(keys), expected.keys[[1, 0]])
    np.testing.assert_array_equal(np.concatenate(values), expected.values[[1, 0]])


def test_native_cpu_codec_is_a_separate_object_with_its_own_projection_storage():
    adapter, batch = create()
    assert hasattr(adapter.model, "cpu_episode_codec"), "Falta el codec nativo independiente"
    native_codec = adapter.model.cpu_episode_codec()
    tensors = DecisionBatch.from_validated(batch, device="cpu", dtype=torch.float64)
    inputs = adapter.native.CandidateInputs(
        *(tensors.inputs[name] for name in MODALITIES), tensors.presence
    )
    expected = native_codec.encode(inputs)
    del adapter
    with torch.no_grad():
        expected.keys.zero_()
        expected.values.zero_()
    actual = native_codec.encode(inputs)
    assert torch.count_nonzero(actual.values) > 0
    assert not actual.keys.requires_grad and not actual.values.requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_epsilon_rejects_subunit_keys_without_replacing_them_with_zero(dtype):
    adapter, _ = create(dtype=dtype, historical=False)
    native_codec = adapter.model.cpu_episode_codec()
    data = [
        torch.zeros((1, 64, 5), dtype=dtype, device="cpu"),
        torch.zeros((1, 3), dtype=dtype, device="cpu"),
        torch.zeros((1, 4), dtype=dtype, device="cpu"),
        torch.zeros((1, 3), dtype=dtype, device="cpu"),
        torch.zeros((1, 3), dtype=dtype, device="cpu"),
        torch.ones((1, 5), dtype=torch.bool, device="cpu"),
    ]
    exact_zero = native_codec.encode(adapter.native.CandidateInputs(*data))
    assert torch.count_nonzero(exact_zero.keys) == 0
    for tiny in (1e-30, 1e-40 if dtype == torch.float32 else 1e-200):
        data[0][0, 0, 0] = tiny
        with pytest.raises(ValueError, match="clave|norma"):
            native_codec.encode(adapter.native.CandidateInputs(*data))
    data[0][0, 0, 0] = 1e-6
    nonzero = native_codec.encode(adapter.native.CandidateInputs(*data))
    torch.testing.assert_close(
        nonzero.keys.norm(dim=1), torch.ones(1, dtype=dtype), rtol=1e-5, atol=1e-7
    )


def test_context_only_keeps_the_native_fusion_without_a_second_episode_projection():
    adapter, batch = create()
    tensors = DecisionBatch.from_validated(batch, dtype=torch.float64)
    inputs = adapter.native.CandidateInputs(
        *(tensors.inputs[name] for name in MODALITIES), tensors.presence
    )
    assert hasattr(adapter.model, "encode_context"), "Falta el contexto nativo separado"
    assert torch.equal(adapter.model.encode_context(inputs), adapter.model.encode(inputs).fused)


def test_codec_rejects_an_insufficient_budget_before_copying_projections():
    adapter, _ = create()
    with pytest.raises(ValueError, match="presupuesto"):
        codec_class()(adapter, max_working_bytes=1)


def test_encoded_arrays_are_read_only_and_owned():
    adapter, batch = create()
    encoded = codec_class()(adapter).encode(batch)
    encoded.verify()
    for values in (encoded.keys, encoded.values):
        with pytest.raises(ValueError):
            values.setflags(write=True)
    assert encoded.flow_ids == batch.flow_ids
    assert encoded.input_digest == batch.input_digest


def _read_only_change(values):
    changed = values.copy()
    changed[-1, -1] = np.nextafter(changed[-1, -1], np.inf)
    changed.setflags(write=False)
    return changed


@pytest.mark.parametrize("changed", ["shape", "metadata", "keys", "values"])
def test_encoding_verification_rejects_changed_contract(changed):
    adapter, batch = create()
    encoded = codec_class()(adapter).encode(batch)
    if changed == "shape":
        object.__setattr__(encoded, "keys", encoded.keys.reshape(1, 256))
    elif changed == "metadata":
        object.__setattr__(encoded, "input_digest", "0" * 64)
    else:
        object.__setattr__(encoded, changed, _read_only_change(getattr(encoded, changed)))
    with pytest.raises(ValueError):
        encoded.verify()


def test_codec_rejects_another_input_contract():
    adapter, _ = create()
    other = specification(source="d" * 64)
    batch = validated_cpu_batch(raw_batch(), other)
    with pytest.raises(ValueError, match="inputs"):
        codec_class()(adapter).encode(batch)


def test_codec_checks_execution_flags_without_changing_them():
    adapter, batch = create()
    codec = codec_class()(adapter)
    before = torch.get_num_threads()
    changed = 1 if before != 1 else 2
    try:
        torch.set_num_threads(changed)
        with pytest.raises(ValueError, match="ejecución"):
            codec.encode(batch)
        assert torch.get_num_threads() == changed
    finally:
        torch.set_num_threads(before)


def test_encode_budget_is_checked_before_scanning_nonfinite_inputs():
    adapter, batch = create()
    native = adapter.model.cpu_episode_codec()
    tight = adapter.model.cpu_episode_codec(native.estimated_bytes(2) - 1)
    tensors = DecisionBatch.from_validated(batch, dtype=torch.float64)
    tensors.inputs["news"][0, 0] = float("nan")
    inputs = adapter.native.CandidateInputs(
        *(tensors.inputs[name] for name in MODALITIES), tensors.presence
    )
    with pytest.raises(ValueError, match="presupuesto"):
        tight.encode(inputs)


def test_inference_mode_keeps_the_codec_arrays_detached_and_identified():
    adapter, batch = create()
    codec = codec_class()(adapter)
    with torch.inference_mode():
        encoded = codec.encode(batch)
    encoded.verify()
    assert encoded.codec_id == codec.fingerprint()


def test_codec_identity_separates_precisions_and_keeps_widths():
    identities = {}
    for dtype in (torch.float32, torch.float64):
        adapter, _ = create(dtype=dtype)
        codec = codec_class()(adapter)
        identity = codec.identity()
        assert (identity["key_width"], identity["value_width"]) == (128, 256)
        assert identity["dtype"] == str(dtype).removeprefix("torch.")
        identities[dtype] = codec.fingerprint()
    assert identities[torch.float32] != identities[torch.float64]


def test_codec_verifies_the_cpu_batch_before_reading_its_arrays():
    adapter, batch = create()
    object.__setattr__(batch, "input_digest", "0" * 64)
    with pytest.raises(ValueError, match="huella"):
        codec_class()(adapter).encode(batch)
