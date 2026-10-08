"""Representación fija de las entradas verificadas, sin ejecutar un predictor."""

import copy

import numpy as np
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.models.titans.financial_inputs import FinancialInputSpec, validated_cpu_batch

DIMENSIONS = dict(prices=5, news=4, charts=6, fundamentals=3, macro=3)


def specification(source="a" * 64):
    return FinancialInputSpec(
        source_sha256=source,
        view_sha256="b" * 64,
        input_policy=HISTORICAL_MASKED,
        dimensions=DIMENSIONS,
        representation=dict(
            **policy_identity(HISTORICAL_MASKED),
            fundamental_concepts=["assets"],
            macro_indicators=["rate"],
            encoders={"fixture": "frozen"},
            representation_code={"fixture.py": "c" * 64},
            text_aggregation="mean",
            context_sessions=64,
            news_lookback_sessions=5,
        ),
    )


def raw_batch(size=3, absent=True):
    random = np.random.default_rng(53)
    inputs = {
        name: random.normal(size=(size, 64, width) if name == "prices" else (size, width)).astype(
            np.float32
        )
        for name, width in DIMENSIONS.items()
    }
    inputs["fundamentals"][:] = [4.0, 1.0, 2.0]
    inputs["macro"][:] = [1.0, 1.0, 1.0]
    presence = np.ones((size, 5), dtype=bool)
    if absent:
        presence[-1, [1, 3, 4]] = False
        for name in ("news", "fundamentals", "macro"):
            inputs[name][-1] = 0
    at = 1_609_459_200_000_000
    return dict(
        inputs=inputs,
        presence=presence,
        sample_ids=[f"US/A{i:03d}/{at}" for i in range(size)],
        prediction_at=np.full(size, at, dtype="datetime64[us]"),
        input_available_at=np.full(size, at - 1, dtype="datetime64[us]"),
        target=object(),
        target_available_at=object(),
    )


def selected(raw, indices):
    return dict(
        inputs={name: values[indices] for name, values in raw["inputs"].items()},
        presence=raw["presence"][indices],
        sample_ids=[raw["sample_ids"][i] for i in indices],
        prediction_at=raw["prediction_at"][indices],
        input_available_at=raw["input_available_at"][indices],
    )


def test_encode_uses_real_immutable_cpu_view_and_preserves_inputs():
    spec = specification()
    raw = raw_batch()
    view = validated_cpu_batch(raw, spec)
    before = {name: values.tobytes() for name, values in view.inputs.items()}
    result = FrozenEpisodeCodec(spec).encode(view)
    assert result.key_inputs.shape == result.values.shape == (3, 64)
    assert result.key_inputs.dtype == result.values.dtype == np.float32
    np.testing.assert_array_equal(result.key_inputs[:, -1], 1.0)
    assert result.input_digest == view.input_digest
    assert result.sample_ids == view.sample_ids
    assert not hasattr(result, "target") and not hasattr(result, "target_available_at")
    assert all(values.tobytes() == before[name] for name, values in view.inputs.items())
    for values in (result.key_inputs, result.values, result.input_l2, result.key_contribution_l2):
        with pytest.raises(ValueError):
            values.setflags(write=True)
    result.verify()


def test_permutation_and_physical_batches_keep_exact_coordinates():
    spec = specification()
    raw = raw_batch(17)
    codec = FrozenEpisodeCodec(spec)
    result = codec.encode(validated_cpu_batch(raw, spec))
    order = list(range(16, -1, -1))
    permuted = codec.encode(validated_cpu_batch(selected(raw, order), spec))
    np.testing.assert_array_equal(permuted.key_inputs[::-1], result.key_inputs)
    np.testing.assert_array_equal(permuted.values[::-1], result.values)
    for indices in ([0], [1, 2], list(range(3, 17))):
        block = codec.encode(validated_cpu_batch(selected(raw, indices), spec))
        np.testing.assert_array_equal(block.key_inputs, result.key_inputs[indices])
        np.testing.assert_array_equal(block.values, result.values[indices])


def test_absent_and_observed_zero_remain_distinct_without_inventing_values():
    spec = specification()
    raw = raw_batch(2, absent=False)
    for values in raw["inputs"].values():
        values[:] = 0
    raw["inputs"]["fundamentals"][:] = [0, 1, 0]
    raw["inputs"]["macro"][:] = [0, 1, 0]
    raw["presence"][1, 1] = False
    result = FrozenEpisodeCodec(spec).encode(validated_cpu_batch(raw, spec))
    assert not np.array_equal(result.key_inputs[0], result.key_inputs[1])
    assert result.input_l2[:, 1].tolist() == [0, 0]
    assert result.normalized_l2[:, 1].tolist() == [0, 0]
    assert np.all(result.key_inputs[:, -1] == 1)


def test_weights_and_diagnostics_are_explicit_for_each_modality():
    spec = specification()
    weights = (2.0, 3.0, 4.0, 5.0, 6.0)
    codec = FrozenEpisodeCodec(spec, modality_weights=weights)
    result = codec.encode(validated_cpu_batch(raw_batch(), spec))
    np.testing.assert_allclose(result.normalized_l2[0], weights, rtol=1e-14, atol=0)
    assert result.normalized_l2[-1, 1] == result.normalized_l2[-1, 3] == 0
    assert result.key_contribution_l2.shape == result.value_contribution_l2.shape == (3, 6)
    assert result.block_names == ("prices", "news", "charts", "fundamentals", "macro", "presence")
    assert codec.identity()["modality_weights"] == list(weights)


def test_codec_rng_is_independent_and_recipe_reconstructs_exactly():
    spec = specification()
    before = np.random.get_state()
    first = FrozenEpisodeCodec(spec, seed=82)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[0] == after[0] and before[2:] == after[2:]
    second = FrozenEpisodeCodec.from_identity(first.identity(), spec)
    view = validated_cpu_batch(raw_batch(), spec)
    np.testing.assert_array_equal(first.encode(view).key_inputs, second.encode(view).key_inputs)
    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() != FrozenEpisodeCodec(spec, seed=83).fingerprint()


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", True),
        ("schema_version", True),
        ("key_width", 64.0),
        ("projection_sha256", "d" * 64),
    ],
)
def test_changed_recipe_or_json_types_are_rejected(field, value):
    spec = specification()
    codec = FrozenEpisodeCodec(spec)
    identity = codec.identity()
    identity[field] = value
    with pytest.raises(ValueError):
        FrozenEpisodeCodec.from_identity(identity, spec)


def test_other_input_contract_and_changed_cpu_digest_are_rejected():
    spec = specification()
    codec = FrozenEpisodeCodec(spec)
    with pytest.raises(ValueError, match="contrato"):
        codec.encode(validated_cpu_batch(raw_batch(), specification("f" * 64)))
    view = validated_cpu_batch(raw_batch(), spec)
    object.__setattr__(view, "input_digest", "f" * 64)
    with pytest.raises(ValueError, match="huella"):
        codec.encode(view)


def test_mutated_output_metadata_is_rejected():
    spec = specification()
    result = FrozenEpisodeCodec(spec).encode(validated_cpu_batch(raw_batch(), spec))
    object.__setattr__(result, "key_inputs", result.key_inputs.reshape(192))
    with pytest.raises(ValueError):
        result.verify()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"seed": -1},
        {"key_constant": 0},
        {"presence_weight": np.nan},
        {"modality_weights": (1, 1)},
        {"modality_weights": (1, 1, 0, 1, 1)},
        {"max_buffer_bytes": 1},
    ],
)
def test_invalid_recipe_or_memory_budget_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FrozenEpisodeCodec(specification(), **kwargs)


def test_finite_fp32_extremes_remain_finite_after_block_normalization():
    spec = specification()
    raw = raw_batch(2, absent=False)
    raw["inputs"]["prices"][0] = np.finfo(np.float32).max
    raw["inputs"]["prices"][1] = np.nextafter(np.float32(0), np.float32(1))
    result = FrozenEpisodeCodec(spec).encode(validated_cpu_batch(raw, spec))
    assert np.isfinite(result.key_inputs).all() and np.isfinite(result.values).all()
    np.testing.assert_allclose(result.normalized_l2[:, 0], 1.0, rtol=1e-14, atol=0)


def test_identity_copy_does_not_mutate_the_codec():
    codec = FrozenEpisodeCodec(specification())
    before = codec.fingerprint()
    identity = copy.deepcopy(codec.identity())
    identity["modality_weights"][0] = 16.0
    assert codec.fingerprint() == before


def test_mutated_projection_is_rejected_before_encoding():
    codec = FrozenEpisodeCodec(specification())
    changed = codec._projection.copy()
    changed[0, 0] += 1
    object.__setattr__(codec, "_projection", changed)
    with pytest.raises(ValueError, match="proyección"):
        codec.encode(validated_cpu_batch(raw_batch(), specification()))


def test_encode_budget_fails_before_projection_for_an_oversized_batch():
    spec = specification()
    codec = FrozenEpisodeCodec(spec, max_buffer_bytes=2 * 1024**2)
    with pytest.raises(ValueError, match="presupuesto"):
        codec.encode(validated_cpu_batch(raw_batch(256), spec))


def test_large_batch_traced_allocations_stay_below_estimate():
    import tracemalloc

    spec = specification()
    codec = FrozenEpisodeCodec(spec)
    view = validated_cpu_batch(raw_batch(256), spec)
    tracemalloc.start()
    try:
        result = codec.encode(view)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak + codec._projection.nbytes <= result.estimated_peak_bytes


@pytest.mark.parametrize("constant", [1e-100, 1e300, float("inf"), True])
def test_key_constant_must_remain_positive_finite_fp32(constant):
    with pytest.raises(ValueError):
        FrozenEpisodeCodec(specification(), key_constant=constant)


def test_projection_matches_independent_scalar_recipe_on_prepared_inputs():
    import math

    spec = specification()
    codec = FrozenEpisodeCodec(spec, modality_weights=(1, 2, 3, 4, 5), presence_weight=2)
    view = validated_cpu_batch(raw_batch(), spec)
    actual = codec.encode(view)
    expected = []
    for row in range(len(view.flow_ids)):
        start = 0
        contributions = []
        for index, name in enumerate(actual.block_names):
            if name == "presence":
                vector = [2 * float(value) / math.sqrt(5) for value in view.presence[row]]
            else:
                values = [float(value) for value in view.inputs[name][row].ravel()]
                norm = math.sqrt(math.fsum(value * value for value in values))
                vector = [(index + 1) * value / norm if norm else 0.0 for value in values]
            contributions.append(
                [
                    math.fsum(
                        value * float(weight)
                        for value, weight in zip(
                            vector, projection[start : start + len(vector)], strict=True
                        )
                    )
                    for projection in codec._projection
                ]
            )
            start += len(vector)
        expected.append([math.fsum(column) for column in zip(*contributions, strict=True)])
    expected = np.asarray(expected, dtype=np.float32)
    np.testing.assert_array_equal(actual.key_inputs[:, :63], expected[:, :63])
    np.testing.assert_array_equal(actual.values, expected[:, 63:])


def test_changed_output_provenance_fails_digest_verification():
    codec = FrozenEpisodeCodec(specification())
    result = codec.encode(validated_cpu_batch(raw_batch(), specification()))
    object.__setattr__(result, "input_digest", "f" * 64)
    with pytest.raises(ValueError, match="huella"):
        result.verify()


@pytest.mark.parametrize("identity", [None, {}, {"seed": 2}])
def test_incomplete_recipe_is_rejected(identity):
    with pytest.raises(ValueError):
        FrozenEpisodeCodec.from_identity(identity, specification())


def test_oversized_projected_values_fail_explicitly():
    codec = FrozenEpisodeCodec(specification(), modality_weights=(1e40,) * 5)
    with pytest.raises(ValueError, match="FP32"):
        codec.encode(validated_cpu_batch(raw_batch(), specification()))
