"""Integración matemática con lotes técnicos, sin objetivos ni optimizadores."""

import importlib
import io
import warnings
from dataclasses import replace

import numpy as np
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED, STRICT_INPUTS, policy_identity
from mars_titan.models.baselines.multimodal import MultimodalReference

DIMENSIONS = dict(prices=5, news=2, charts=2, fundamentals=3, macro=3)
VARIANTS = ("transformer_direct", "mac_disabled", "mac_frozen", "mac_online")


def api():
    try:
        return importlib.import_module("mars_titan.models.titans.financial")
    except ModuleNotFoundError:
        pytest.fail("Falta el adaptador financiero explícito de Titans")


def specification(*, historical=True, source="a" * 64):
    policy = HISTORICAL_MASKED if historical else STRICT_INPUTS
    representation = dict(
        **policy_identity(policy),
        fundamental_concepts=["assets"],
        macro_indicators=["rate"],
        encoders={"fixture": "frozen-v1"},
        representation_code={"fixture.py": "c" * 64},
        text_aggregation="mean",
        context_sessions=64,
        news_lookback_sessions=5,
    )
    return api().FinancialInputSpec(
        source_sha256=source,
        view_sha256="b" * 64,
        representation=representation,
        dimensions=DIMENSIONS,
        input_policy=policy,
    )


def raw_batch(*, at=1_609_459_200_000_000, absent=True):
    generator = np.random.default_rng(67)
    values = {
        name: generator.normal(size=(2, 64, width) if name == "prices" else (2, width)).astype(
            np.float32
        )
        for name, width in DIMENSIONS.items()
    }
    values["fundamentals"][:] = [0, 1, 0]
    values["macro"][:] = [0, 1, 0]
    presence = np.ones((2, 5), dtype=np.bool_)
    if absent:
        presence[1, [1, 3, 4]] = False
        for name in ("news", "fundamentals", "macro"):
            values[name][1] = 0
    return dict(
        inputs=values,
        presence=presence,
        sample_ids=[f"US/AAA/{at}", f"US/BBB/{at}"],
        market=["US", "US"],
        prediction_at=np.full(2, at, dtype="datetime64[us]"),
        input_available_at=np.full(2, at - 1, dtype="datetime64[us]"),
        # La frontera no consulta ni transporta estos campos de supervisión.
        target=object(),
        target_available_at=object(),
    )


def setup(variant="mac_online", *, historical=True, seed=42):
    spec = specification(historical=historical)
    config = api().FinancialConfig(spec, variant=variant, hidden_size=32, layers=1, seed=seed)
    model = api().FinancialPredictor(config, dtype=torch.float64)
    raw = raw_batch(absent=historical)
    if not historical:
        raw.pop("presence")
    batch = api().DecisionBatch.from_corpus(raw, spec, dtype=torch.float64)
    return model, batch


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_control_advances_one_decision_without_mutating_received_state(variant):
    model, batch = setup(variant)
    state = model.initial_state(batch.flow_ids)
    previous = model.export_state(state)
    result = model.prepare(batch, state)
    assert result.point_predictions.shape == (2,)
    assert torch.isfinite(result.point_predictions).all()
    assert not result.point_predictions.requires_grad
    assert result.detached_tokens.shape == (2, 32)
    assert not result.detached_tokens.requires_grad
    assert result.next_state.observed_steps.tolist() == [1, 1]
    assert state.observed_steps.tolist() == [0, 0]
    if variant != "transformer_direct":
        expected = [1, 1] if variant == "mac_online" else [0, 0]
        assert result.next_state.mac.memory.steps.tolist() == expected
    assert model.export_state(state)["last_sample_ids"] == previous["last_sample_ids"]


def test_strict_direct_control_matches_the_existing_reference_modules():
    model, batch = setup("transformer_direct", historical=False)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(model.config.seed)
        reference = MultimodalReference(
            "transformer", DIMENSIONS, context=64, hidden_size=32, layers=1, dropout=0.0
        ).double()
    with torch.no_grad():
        expected = reference(batch.inputs)
    actual = model.prepare(batch, model.initial_state(batch.flow_ids))
    torch.testing.assert_close(actual.point_predictions, expected, rtol=0, atol=0)


def test_absent_modalities_are_zeroed_after_bias_and_presence_enters_common_fusion():
    model, batch = setup("transformer_direct")
    observed = {}

    def capture(module, arguments):
        observed["fusion_input"] = arguments[0].detach().clone()

    handle = model.fusion[0].register_forward_pre_hook(capture)
    try:
        model.prepare(batch, model.initial_state(batch.flow_ids))
    finally:
        handle.remove()
    fused = observed["fusion_input"]
    assert fused.shape == (2, 5 * 32 + 5)
    for index in (1, 3, 4):
        assert torch.count_nonzero(fused[1, index * 32 : (index + 1) * 32]) == 0
    torch.testing.assert_close(fused[:, -5:], batch.presence.double(), rtol=0, atol=0)
    assert fused[0, 3 * 32 : 4 * 32].abs().sum() > 0


def test_pairing_copies_real_parameters_and_does_not_relax_ordinary_contract_loading():
    source, batch = setup("mac_online", seed=42)
    source_parameters = dict(source.named_parameters())
    for variant in VARIANTS:
        target, _ = setup(variant, seed=97)
        assert not torch.equal(target.head.weight, source.head.weight)
        previous_state = target.initial_state(batch.flow_ids)
        receipt = api().copy_paired_parameters(source, target)
        for name, parameter in target.named_parameters():
            torch.testing.assert_close(parameter, source_parameters[name], rtol=0, atol=0)
            assert parameter.data_ptr() != source_parameters[name].data_ptr()
        assert receipt["runtime_state_transferred"] is False
        assert receipt["copied_parameters"]
        with pytest.raises(ValueError, match="identidad|parámetros"):
            target.prepare(batch, previous_state)
        if variant != "mac_online":
            with pytest.raises(ValueError, match="contrato|configur"):
                target.load_state_dict(source.state_dict())


def test_repeated_or_reversed_decision_is_rejected_before_another_write():
    model, batch = setup()
    state = model.prepare(batch, model.initial_state(batch.flow_ids)).next_state
    with pytest.raises(ValueError, match="repet|posterior|orden"):
        model.prepare(batch, state)
    older = api().DecisionBatch.from_corpus(
        raw_batch(at=1_609_372_800_000_000), model.config.inputs, dtype=torch.float64
    )
    with pytest.raises(ValueError, match="posterior|orden"):
        model.prepare(older, state)
    following = api().DecisionBatch.from_corpus(
        raw_batch(at=1_609_545_600_000_000), model.config.inputs, dtype=torch.float64
    )
    result = model.prepare(following, state)
    assert result.next_state.mac.memory.steps.tolist() == [2, 2]
    assert result.next_state.observed_steps.tolist() == [2, 2]


def test_joint_permutation_preserves_outputs_and_mismatched_flow_order_fails():
    model, batch = setup()
    state = model.initial_state(batch.flow_ids)
    expected = model.prepare(batch, state)
    permuted = batch.select([1, 0])
    with pytest.raises(ValueError, match="flujo|orden"):
        model.prepare(permuted, state)
    shuffled = model.select_state(state, permuted.flow_ids)
    actual = model.prepare(permuted, shuffled)
    torch.testing.assert_close(
        actual.point_predictions.flip(0), expected.point_predictions, rtol=1e-12, atol=1e-12
    )


def test_external_fixture_gradient_reaches_backbone_masked_fusion_and_fast_update():
    model, batch = setup()
    state = model.initial_state(batch.flow_ids, differentiable=True)
    result = model.prepare(batch, state, differentiable=True)
    result.point_predictions.square().sum().backward()
    for parameter in (
        model.price_encoder.projection.weight,
        model.fusion[0].weight,
        model.head.weight,
        model.mac.memory.theta_projection.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_recovery_preserves_next_prediction_and_rejects_another_control():
    model, batch = setup()
    state = model.prepare(batch, model.initial_state(batch.flow_ids)).next_state
    archive = io.BytesIO()
    torch.save(dict(model=model.state_dict(), runtime=model.export_state(state)), archive)
    archive.seek(0)
    stored = torch.load(archive, weights_only=True)
    replica, _ = setup()
    replica.load_state_dict(stored["model"])
    restored = replica.restore_state(stored["runtime"])
    following = api().DecisionBatch.from_corpus(
        raw_batch(at=1_609_545_600_000_000), model.config.inputs, dtype=torch.float64
    )
    expected = model.prepare(following, state)
    actual = replica.prepare(following, restored)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    incompatible, _ = setup("mac_frozen")
    with pytest.raises(ValueError, match="contrato|identidad"):
        incompatible.restore_state(stored["runtime"])


@pytest.mark.parametrize(
    "fault",
    [
        "missing_presence",
        "float_presence",
        "missing_fill",
        "concept_mask",
        "negative_age",
        "future",
        "reserved",
        "mixed_cutoff",
        "duplicate",
        "float64_vectors",
    ],
)
def test_historical_batch_rejects_incompatible_inputs_before_tensor_conversion(fault):
    spec = specification()
    raw = raw_batch()
    if fault == "missing_presence":
        raw.pop("presence")
    elif fault == "float_presence":
        raw["presence"] = raw["presence"].astype(float)
    elif fault == "missing_fill":
        raw["inputs"]["news"][1, 0] = 1e-8
    elif fault == "concept_mask":
        raw["inputs"]["macro"][0, 1] = 0.5
    elif fault == "negative_age":
        raw["inputs"]["macro"][0, 2] = -1
    elif fault == "future":
        raw["input_available_at"][0] += np.timedelta64(2, "us")
    elif fault == "reserved":
        raw = raw_batch(at=1_704_067_200_000_000)
    elif fault == "mixed_cutoff":
        raw["prediction_at"][1] += np.timedelta64(1, "D")
    elif fault == "duplicate":
        raw["sample_ids"][1] = raw["sample_ids"][0]
    elif fault == "float64_vectors":
        raw["inputs"]["macro"] = raw["inputs"]["macro"].astype(np.float64)
    with pytest.raises(ValueError):
        api().DecisionBatch.from_corpus(raw, spec, dtype=torch.float64)


def test_decision_boundary_does_not_copy_targets_and_copies_input_storage():
    spec = specification()
    raw = raw_batch()
    batch = api().DecisionBatch.from_corpus(raw, spec)
    assert not hasattr(batch, "target") and not hasattr(batch, "target_available_at")
    before = batch.inputs["prices"].clone()
    raw["inputs"]["prices"][:] = 9
    torch.testing.assert_close(batch.inputs["prices"], before, rtol=0, atol=0)


def test_changed_input_identity_and_external_parameter_mutation_reject_old_state():
    model, batch = setup()
    state = model.initial_state(batch.flow_ids)
    other = api().DecisionBatch.from_corpus(
        raw_batch(), specification(source="d" * 64), dtype=torch.float64
    )
    with pytest.raises(ValueError, match="identidad|contrato"):
        model.prepare(other, state)
    with torch.no_grad():
        model.head.weight.add_(1)
    with pytest.raises(ValueError, match="parámetros"):
        model.prepare(batch, state)


@pytest.mark.parametrize(
    "option", [dict(bank_policy="episodic"), dict(refinements=2), dict(variant="mars_titan_full")]
)
def test_unimplemented_amplifications_are_rejected(option):
    with pytest.raises(ValueError):
        api().FinancialConfig(specification(), **option)


def test_runtime_state_budget_counts_retained_tensor_storage():
    model, batch = setup()
    model = api().FinancialPredictor(
        replace(model.config, max_state_bytes=128 * 1024), dtype=torch.float64
    )
    state = model.initial_state(batch.flow_ids)
    retained = torch.zeros(1_000_000, dtype=torch.int64)
    invalid = replace(state, observed_steps=retained[:2])
    with pytest.raises(ValueError, match="presupuesto|bytes"):
        model.export_state(invalid)


def test_runtime_recovery_does_not_admit_a_cursor_inside_the_reserved_year():
    model, batch = setup("mac_frozen")
    payload = model.export_state(model.initial_state(batch.flow_ids))
    payload["observed_steps"] = torch.ones(2, dtype=torch.int64)
    payload["last_prediction_at"] = (1_704_067_200_000_000,) * 2
    payload["last_sample_ids"] = tuple(f"{flow}/1704067200000000" for flow in batch.flow_ids)
    with pytest.raises(ValueError, match="cursor|corte|2024"):
        model.restore_state(payload)


def test_runtime_usage_exposes_tensor_bytes_per_flow_and_total():
    model, batch = setup()
    state = model.initial_state(batch.flow_ids)
    usage = model.state_usage(state)
    assert usage["tensor_bytes_per_flow"] == 32784
    assert usage["tensor_bytes"] == 65568
    assert usage["flow_count"] == 2
    assert usage["total_bytes"] >= usage["tensor_bytes"]


def test_reader_and_cpu_adapter_share_the_historical_vector_validation():
    import pyarrow as pa

    from mars_titan.data.input_policy import validate_historical_vectors
    from mars_titan.training.corpus_inputs import _presence

    raw = raw_batch()
    representation = specification().representation
    vectors = {name: raw["inputs"][name] for name in DIMENSIONS if name != "prices"}
    table = pa.table({"presence": raw["presence"].tolist(), "news_count": [1, 0]})
    np.testing.assert_array_equal(_presence(table, vectors, representation), raw["presence"])
    vectors["macro"][0, 1] = 0.5
    for validate in (
        lambda: _presence(table, vectors, representation),
        lambda: validate_historical_vectors(vectors, raw["presence"], representation),
    ):
        with pytest.raises(ValueError, match="máscaras"):
            validate()


def test_mutating_a_validated_batch_is_rejected():
    model, batch = setup()
    batch.inputs["macro"][0, 1] = 0.5
    with pytest.raises(ValueError, match="cambiaron"):
        model.prepare(batch, model.initial_state(batch.flow_ids))


def test_recovery_restores_evaluation_mode_before_emitting_the_next_token():
    source, batch = setup("transformer_direct")
    source.eval()
    state = source.prepare(batch, source.initial_state(batch.flow_ids)).next_state
    replica, _ = setup("transformer_direct")
    replica.load_state_dict(source.state_dict())
    restored = replica.restore_state(source.export_state(state))
    following = api().DecisionBatch.from_corpus(
        raw_batch(at=1_609_545_600_000_000), source.config.inputs, dtype=torch.float64
    )
    actual = replica.prepare(following, restored)
    expected = source.prepare(following, state)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.detached_tokens, expected.detached_tokens, rtol=0, atol=0)


def test_unversioned_parameter_edit_is_detected_at_the_state_publication_boundary():
    model, batch = setup()
    state = model.initial_state(batch.flow_ids)
    # data puede eludir el contador. La huella de bytes se comprueba al publicar.
    model.head.weight.data.add_(1)
    with pytest.raises(ValueError, match="huella|parámetros"):
        model.export_state(state)


def test_default_float32_path_runs_without_precision_conversion_in_prepare():
    model = api().FinancialPredictor(
        api().FinancialConfig(specification(), variant="mac_online", hidden_size=32)
    )
    batch = api().DecisionBatch.from_corpus(raw_batch(), model.config.inputs)
    result = model.prepare(batch, model.initial_state(batch.flow_ids))
    assert result.point_predictions.dtype == torch.float32
    assert torch.isfinite(result.point_predictions).all()
    assert result.next_state.mac.memory.steps.tolist() == [1, 1]


@pytest.mark.parametrize(
    "fault", ["missing_modality", "wrong_shape", "nan", "unknown_time", "wrong_sample_id"]
)
def test_cpu_boundary_rejects_malformed_observations(fault):
    raw = raw_batch()
    if fault == "missing_modality":
        raw["inputs"].pop("charts")
    elif fault == "wrong_shape":
        raw["inputs"]["prices"] = raw["inputs"]["prices"][:, :63]
    elif fault == "nan":
        raw["inputs"]["news"][0, 0] = np.nan
    elif fault == "unknown_time":
        raw["input_available_at"][0] = np.datetime64("NaT", "us")
    else:
        raw["sample_ids"][0] = "CN/AAA/123"
    with pytest.raises(ValueError):
        api().DecisionBatch.from_corpus(raw, specification())


@pytest.mark.parametrize("variant", ["mac_frozen", "mac_disabled"])
def test_same_seed_and_shapes_do_not_allow_loading_another_mac_contract(variant):
    source, _ = setup("mac_online")
    target, _ = setup(variant)
    with pytest.raises(ValueError, match="contrato|configur"):
        target.load_state_dict(source.state_dict())


def test_verified_cpu_view_is_shared_with_the_tensor_adapter_without_labels():
    module = importlib.import_module("mars_titan.models.titans.financial_inputs")
    raw, spec = raw_batch(), specification()
    verified = module.validated_cpu_batch(raw, spec)
    before = verified.inputs["prices"].copy()
    raw["inputs"]["prices"][:] = 4
    np.testing.assert_array_equal(verified.inputs["prices"], before)
    with pytest.raises(ValueError):
        verified.inputs["prices"][0, 0, 0] = 1
    assert not hasattr(verified, "target") and not hasattr(verified, "target_available_at")
    adapted = api().DecisionBatch.from_validated(verified, dtype=torch.float64)
    torch.testing.assert_close(adapted.inputs["prices"], torch.tensor(before, dtype=torch.float64))
    assert adapted.input_digest == verified.input_digest


def test_cpu_observation_digest_distinguishes_missing_from_observed_zero():
    module = importlib.import_module("mars_titan.models.titans.financial_inputs")
    raw, spec = raw_batch(absent=False), specification()
    raw["inputs"]["news"][0] = 0
    observed = module.validated_cpu_batch(raw, spec)
    raw["presence"][0, 1] = False
    missing = module.validated_cpu_batch(raw, spec)
    assert observed.input_digest != missing.input_digest


def test_cpu_snapshot_cannot_reenable_writes_and_change_absent_inputs():
    module = importlib.import_module("mars_titan.models.titans.financial_inputs")
    verified = module.validated_cpu_batch(raw_batch(), specification())
    for values in (*verified.inputs.values(), verified.presence):
        with pytest.raises(ValueError):
            values.setflags(write=True)


def test_tensor_conversion_rejects_changed_interpretation_of_verified_cpu_bytes():
    module = importlib.import_module("mars_titan.models.titans.financial_inputs")
    verified = module.validated_cpu_batch(raw_batch(), specification())
    # NumPy 2.5 avisa de esta mutación de metadatos, que la frontera debe rechazar.
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        verified.inputs["news"].dtype = np.int32
    assert all(issubclass(notice.category, DeprecationWarning) for notice in notices)
    with pytest.raises(ValueError, match="contrato|cambi|huella"):
        api().DecisionBatch.from_validated(verified)


def test_list_and_tuple_select_the_same_rows_and_preserve_state_parity():
    model, batch = setup()
    by_list, by_tuple = batch.select([1, 0]), batch.select((1, 0))
    assert by_tuple.inputs["prices"].shape == (2, 64, 5)
    assert by_tuple.presence.shape == (2, 5)
    for name in DIMENSIONS:
        torch.testing.assert_close(by_tuple.inputs[name], by_list.inputs[name], rtol=0, atol=0)
    assert by_tuple.input_digest == by_list.input_digest
    state = model.select_state(model.initial_state(batch.flow_ids), by_list.flow_ids)
    left, right = model.prepare(by_list, state), model.prepare(by_tuple, state)
    torch.testing.assert_close(left.point_predictions, right.point_predictions, rtol=0, atol=0)
    for first, second in zip(
        left.next_state.mac.memory.weights, right.next_state.mac.memory.weights, strict=True
    ):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
