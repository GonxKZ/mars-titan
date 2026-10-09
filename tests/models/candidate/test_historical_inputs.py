"""Lotes manuales de la GRU histórica, sin residualizar ni ajustar parámetros."""

import copy
import importlib
import os

import numpy as np
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED, STRICT_INPUTS, policy_identity
from mars_titan.models.titans.financial_inputs import FinancialInputSpec, validated_cpu_batch

pytestmark = pytest.mark.skipif(
    not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"), reason="Falta el enlace nativo compilado"
)


def api():
    return importlib.import_module("mars_titan.models.candidate.input_adapter")


def specification(*, historical=True, catalog="assets", source="a" * 64):
    policy = HISTORICAL_MASKED if historical else STRICT_INPUTS
    return FinancialInputSpec(
        source_sha256=source,
        view_sha256="b" * 64,
        input_policy=policy,
        dimensions=dict(prices=5, news=3, charts=4, fundamentals=3, macro=3),
        representation=dict(
            **policy_identity(policy),
            fundamental_concepts=[catalog],
            macro_indicators=["rate"],
            encoders={"manual": "frozen-v1"},
            representation_code={"manual.py": "c" * 64},
            text_aggregation="mean",
            context_sessions=64,
            news_lookback_sessions=5,
        ),
    )


def raw_batch(*, absent=True):
    moment = 1_609_459_200_000_000
    inputs = dict(
        prices=np.linspace(-1, 1, 2 * 64 * 5, dtype=np.float32).reshape(2, 64, 5),
        news=np.array([[1, -1, 0], [0, 0, 0]], dtype=np.float32),
        charts=np.ones((2, 4), dtype=np.float32),
        fundamentals=np.tile(np.array([0, 1, 0], dtype=np.float32), (2, 1)),
        macro=np.tile(np.array([0, 1, 0], dtype=np.float32), (2, 1)),
    )
    presence = np.ones((2, 5), dtype=np.bool_)
    if absent:
        presence[1, [1, 3, 4]] = False
        inputs["fundamentals"][1] = 0
        inputs["macro"][1] = 0
    return dict(
        inputs=inputs,
        presence=presence,
        sample_ids=[f"US/AAA/{moment}", f"CN/000001/{moment}"],
        prediction_at=np.full(2, moment, dtype="datetime64[us]"),
        input_available_at=np.full(2, moment - 1, dtype="datetime64[us]"),
        target=object(),
        target_available_at=object(),
    )


def create(*, dtype=torch.float64, historical=True):
    spec = specification(historical=historical)
    raw = raw_batch(absent=historical)
    if not historical:
        raw.pop("presence")
    return api().CandidateInputAdapter(spec, dtype=dtype), validated_cpu_batch(raw, spec)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("refinements", [1, 2, 4])
def test_empty_memory_keeps_native_quantiles_and_gradients(dtype, refinements):
    before = torch.random.get_rng_state().clone()
    adapter, batch = create(dtype=dtype)
    prediction = adapter.forward(batch, refinements=refinements)
    assert prediction.quantiles.shape == (2, 5)
    assert prediction.quantiles.dtype == dtype
    assert torch.all(prediction.quantiles[:, 1:] >= prediction.quantiles[:, :-1])
    assert torch.equal(prediction.median, prediction.quantiles[:, 2])
    assert prediction.native.read.ids.shape == (2, 0)
    assert not prediction.native.encoded.episode_features.requires_grad
    prediction.median.sum().backward()
    parameters = adapter.model.named_parameters()
    assert parameters["price_weight_ih"].grad.abs().sum() > 0
    assert parameters["head_weight"].grad.abs().sum() > 0
    assert parameters["query_weight"].grad is None
    assert torch.equal(before, torch.random.get_rng_state())


def test_restore_checks_complete_identity_and_next_prediction():
    adapter, batch = create()
    expected = adapter.forward(batch, refinements=4).quantiles
    payload = adapter.export_state()
    restored = api().CandidateInputAdapter.restore(payload, specification(), device="cpu")
    assert restored.identity() == adapter.identity()
    assert torch.equal(restored.forward(batch, refinements=4).quantiles, expected)
    with pytest.raises(ValueError):
        api().CandidateInputAdapter.restore(payload, specification(catalog="cash"), device="cpu")
    with pytest.raises(ValueError):
        api().CandidateInputAdapter.restore(payload, specification(source="d" * 64), device="cpu")


def test_parameter_transfer_is_explicit_and_retains_destination_codec():
    original, original_batch = create(historical=False)
    historical, _ = create()
    batch = validated_cpu_batch(raw_batch(absent=False), specification())
    before_source = original.identity()
    before_codec = historical.model.representation_id()
    receipt = historical.transfer_strict_parameters(original)
    assert receipt["source_parameters"] == original.model.parameter_fingerprint()
    assert receipt["destination_codec"] == before_codec
    assert original.identity() == before_source
    torch.testing.assert_close(
        original.forward(original_batch).quantiles,
        historical.forward(batch).quantiles,
        rtol=1e-12,
        atol=1e-12,
    )
    incompatible = api().CandidateInputAdapter(specification(catalog="cash"), dtype=torch.float64)
    before = incompatible.identity()
    with pytest.raises(ValueError):
        incompatible.transfer_strict_parameters(original)
    assert incompatible.identity() == before


def test_changed_cpu_digest_and_other_contract_are_rejected():
    adapter, batch = create()
    with pytest.raises(ValueError):
        batch.inputs["news"].setflags(write=True)
    object.__setattr__(batch, "input_digest", "0" * 64)
    with pytest.raises(ValueError, match="huella"):
        adapter.forward(batch)
    other = specification(catalog="cash")
    with pytest.raises(ValueError):
        adapter.forward(validated_cpu_batch(raw_batch(), other))


@pytest.mark.parametrize("value", [1e-20, np.nan, np.inf])
def test_absent_payload_is_rejected_in_original_cpu_representation(value):
    adapter, _ = create()
    raw = raw_batch()
    raw["inputs"]["news"][1, 0] = value
    with pytest.raises(ValueError):
        adapter.from_corpus(raw)


def test_float64_masks_are_not_silently_rounded_to_float32():
    adapter, _ = create()
    raw = raw_batch()
    raw["inputs"]["fundamentals"] = raw["inputs"]["fundamentals"].astype(np.float64)
    raw["inputs"]["fundamentals"][0, 1] = 1.00000001
    with pytest.raises(ValueError):
        adapter.from_corpus(raw)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("refinements", [1, 2, 4])
def test_rows_are_invariant_to_permutation_and_physical_partition(dtype, refinements):
    adapter, batch = create(dtype=dtype)
    expected = adapter.forward(batch, refinements=refinements).quantiles
    outputs = []
    raw = raw_batch()
    for row in (1, 0):
        selected = dict(
            inputs={name: array[[row]] for name, array in raw["inputs"].items()},
            presence=raw["presence"][[row]],
            sample_ids=[raw["sample_ids"][row]],
            prediction_at=raw["prediction_at"][[row]],
            input_available_at=raw["input_available_at"][[row]],
        )
        outputs.append(adapter.from_corpus(selected, refinements=refinements).quantiles)
    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(torch.cat(outputs), expected[[1, 0]], rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("changed", ["archive", "identity", "schema", "extra", "kind"])
def test_restoration_rejects_forged_envelopes(changed):
    adapter, _ = create()
    payload = copy.deepcopy(adapter.export_state())
    if changed == "archive":
        payload["archive"] = payload["archive"][:-1]
    elif changed == "identity":
        payload["identity"]["parameters_sha256"] = "0" * 64
    elif changed == "schema":
        payload["schema_version"] = True
    elif changed == "extra":
        payload["unexpected"] = 1
    else:
        payload["archive"] = bytearray(payload["archive"])
    with pytest.raises(ValueError):
        api().CandidateInputAdapter.restore(payload, specification())


@pytest.mark.parametrize("kind", ["future", "unknown", "id", "presence", "mask", "age"])
def test_common_input_validation_keeps_causal_and_mask_rejections(kind):
    adapter, _ = create()
    raw = raw_batch()
    if kind == "future":
        raw["input_available_at"][:] = raw["prediction_at"] + np.timedelta64(1, "us")
    elif kind == "unknown":
        raw["prediction_at"][0] = np.datetime64("NaT", "us")
    elif kind == "id":
        raw["sample_ids"][0] = "US/AAA/1"
    elif kind == "presence":
        raw["presence"] = raw["presence"].astype(np.int64)
    elif kind == "mask":
        raw["inputs"]["macro"][0, 1] = 0.5
    else:
        raw["inputs"]["macro"][0, 2] = -1
    with pytest.raises(ValueError):
        adapter.from_corpus(raw)


def test_no_cross_call_graph_or_window_state_is_retained():
    adapter, batch = create()
    first = adapter.forward(batch).median
    first.sum().backward()
    adapter.model.zero_grad()
    second = adapter.forward(batch).median
    assert torch.equal(first, second)
    second.sum().backward()
    with torch.inference_mode():
        predicted = adapter.forward(batch).median
    assert not predicted.requires_grad
    assert torch.equal(second, predicted)


@pytest.mark.parametrize("count", [0, 3, True, 1.0])
def test_invalid_refinements_fail_before_running(count):
    adapter, batch = create()
    with pytest.raises(ValueError):
        adapter.forward(batch, refinements=count)
