"""N2 con ventanas verificadas y estados reales de Titans, sin aprendizaje."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from mars_titan.models.titans.state import mac_tensors
from tests.models.test_price_presence_families import titans_specification
from tests.models.titans.test_financial_adapter import api, raw_batch

OPTIONS = dict(mode="observable", segments=4, min_assets=1, recent_returns=21, min_returns=42)


def raw(at=1_609_459_200_000_000):
    batch = raw_batch(at=at, absent=False)
    prices = np.zeros((2, 64, 6), dtype=np.float32)
    prices[..., 5] = 1
    prices[..., 3] = np.linspace(0, 0.3, 64)
    batch["inputs"]["prices"] = prices
    return batch


def test_zero_gain_keeps_the_actual_financial_prediction_and_shared_parameters():
    config = api().FinancialConfig(titans_specification(), variant="mac_online", hidden_size=32)
    reference = api().FinancialPredictor(config, dtype=torch.float64)
    model = api().FinancialPredictor(replace(config, active_dendrites=OPTIONS), dtype=torch.float64)
    batch = api().DecisionBatch.from_corpus(raw(), config.inputs, dtype=torch.float64)
    for name, value in reference.named_parameters():
        assert torch.equal(value, dict(model.named_parameters())[name])
    expected = reference.prepare(batch, reference.initial_state(batch.flow_ids))
    result = model.prepare(batch, model.initial_state(batch.flow_ids), session_complete=True)
    assert model.config.identity()["active_dendrites"]["mode"] == "observable"
    assert torch.equal(result.point_predictions, expected.point_predictions)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            mac_tensors(result.next_state.mac), mac_tensors(expected.next_state.mac), strict=True
        )
    )


def configured(mode="observable", variant="mac_online"):
    config = api().FinancialConfig(titans_specification(), variant=variant, hidden_size=32)
    model = api().FinancialPredictor(
        replace(config, active_dendrites={**OPTIONS, "mode": mode}), dtype=torch.float64
    )
    return model, api().DecisionBatch.from_corpus(raw(), config.inputs, dtype=torch.float64)


def activate(model):
    with torch.no_grad():
        model.active_dendrites.gain.fill_(1.0)
        model.active_dendrites.segment_weights.fill_(1.0)
    model._seal_parameters()


@pytest.mark.parametrize(
    "variant", ["transformer_direct", "mac_disabled", "mac_frozen", "mac_online"]
)
def test_the_configured_operator_reaches_the_real_prediction_and_gradients(variant):
    model, batch = configured(variant=variant)
    before = model.prepare(
        batch, model.initial_state(batch.flow_ids), session_complete=True
    ).point_predictions
    activate(model)
    result = model.prepare(
        batch,
        model.initial_state(batch.flow_ids, differentiable=True),
        session_complete=True,
        differentiable=True,
    )
    assert not torch.equal(result.point_predictions, before)
    result.point_predictions.sum().backward()
    assert model.active_dendrites.gain.grad.abs().sum() > 0
    assert model.active_dendrites.segment_weights.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("mode", ["observable", "shuffled_past_prefix", "constant"])
def test_restore_reproduces_the_next_prediction_and_fast_memory(mode):
    model, batch = configured(mode)
    activate(model)
    initial = model.initial_state(batch.flow_ids)
    result = model.prepare(batch, initial, session_complete=True)
    restored = model.restore_state(model.export_state_cpu(result.next_state))
    following = api().DecisionBatch.from_corpus(
        raw(batch.prediction_at[0] + 10), model.config.inputs, dtype=torch.float64
    )
    expected = model.prepare(following, result.next_state, session_complete=True)
    actual = model.prepare(following, restored, session_complete=True)
    assert torch.equal(actual.point_predictions, expected.point_predictions)
    for a, b in zip(
        mac_tensors(actual.next_state.mac), mac_tensors(expected.next_state.mac), strict=True
    ):
        assert torch.equal(a, b)


def test_session_completeness_is_required_and_row_order_does_not_change_aligned_outputs():
    model, batch = configured()
    activate(model)
    with pytest.raises(ValueError, match="sesión completa"):
        model.prepare(batch, model.initial_state(batch.flow_ids))
    reference = model.prepare(batch, model.initial_state(batch.flow_ids), session_complete=True)
    other = batch.select([1, 0])
    actual = model.prepare(other, model.initial_state(other.flow_ids), session_complete=True)
    assert torch.equal(actual.point_predictions.flip(0), reference.point_predictions)


def test_disabled_configuration_keeps_legacy_identity_and_checkpoint_loading():
    config = api().FinancialConfig(titans_specification(), variant="mac_online", hidden_size=32)
    first = api().FinancialPredictor(config, dtype=torch.float64)
    second = api().FinancialPredictor(replace(config, active_dendrites=None), dtype=torch.float64)
    assert "active_dendrites" not in config.identity()
    second.load_state_dict(first.state_dict())
    assert first._parameter_id == second._parameter_id
    assert set(first.state_dict()) == set(second.state_dict())


@pytest.mark.parametrize(
    "options", [{}, {"mode": "future"}, {**OPTIONS, "segments": 0}, {**OPTIONS, "min_assets": True}]
)
def test_invalid_or_incomplete_declarations_are_rejected(options):
    config = api().FinancialConfig(titans_specification(), variant="mac_online", hidden_size=32)
    with pytest.raises(ValueError):
        replace(config, active_dendrites=options)


def test_configuration_and_module_changes_are_rejected_after_initialization():
    model, batch = configured()
    state = model.initial_state(batch.flow_ids)
    model.config = replace(model.config, active_dendrites={**OPTIONS, "mode": "constant"})
    with pytest.raises(ValueError, match="N2"):
        model.prepare(batch, state, session_complete=True)


def test_training_registry_and_frozen_consumer_register_the_same_context():
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
    from mars_titan.training.search_cases import checked_predictor_changes

    change = {"active_dendrites": OPTIONS}
    assert checked_predictor_changes(change) == change
    model, _ = configured()
    reference = api().FinancialPredictor(
        replace(model.config, active_dendrites=None), dtype=torch.float64
    )
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        FrozenFinancialConsumer(reference.eval().requires_grad_(False))
        consumer = FrozenFinancialConsumer(model.eval().requires_grad_(False))
        assert "mars_titan.models.titans.active_dendrites" in consumer.identity()["implementation"]
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_distinct_context_modes_cannot_reuse_each_others_state():
    first, batch = configured()
    other, _ = configured("constant")
    payload = first.export_state(first.initial_state(batch.flow_ids))
    with pytest.raises(ValueError):
        other.restore_state(payload)


def test_the_observable_and_constant_contexts_change_the_composed_prediction():
    observed, _ = configured()
    constant, _ = configured("constant")
    values = raw()
    values["inputs"]["prices"][..., 3] *= -1
    batch = api().DecisionBatch.from_corpus(values, observed.config.inputs, dtype=torch.float64)
    predictions = []
    for model in (observed, constant):
        with torch.no_grad():
            model.active_dendrites.gain.fill_(1.0)
        model._seal_parameters()
        predictions.append(
            model.prepare(
                batch, model.initial_state(batch.flow_ids), session_complete=True
            ).point_predictions
        )
    assert not torch.equal(*predictions)
    assert observed.config.active_dendrites.identity()["selection"] == "full_received_window"
    assert constant.config.active_dendrites.identity()["selection"].startswith("fixed_first")


def test_context_uses_received_prices_before_the_session_rank_transform():
    from mars_titan.models.titans.dendritic_regime import regime_context

    first, batch = configured()
    config = replace(first.config, session_rank_inputs="session_rank_inputs_v1")
    model = api().FinancialPredictor(config, dtype=torch.float64)
    received = []
    hook = model.active_dendrites.register_forward_pre_hook(
        lambda _, args: received.append(args[1].detach().clone())
    )
    try:
        model.prepare(batch, model.initial_state(batch.flow_ids), session_complete=True)
    finally:
        hook.remove()
    expected = regime_context(config.active_dendrites, batch, seed=config.seed)
    assert len(received) == 1 and torch.equal(received[0], expected.values)


def test_local_stability_control_requires_its_own_matched_intervention():
    from mars_titan.models.titans.local_control import MACProjectionConfig
    from tests.models.titans.test_financial_adapter import specification

    model, _ = configured()
    with pytest.raises(ValueError, match="ventanas"):
        replace(model.config, inputs=specification())
    with pytest.raises(ValueError, match="contraste propio"):
        api().FinancialPredictor(model.config, local_control=MACProjectionConfig())


def test_declared_variants_build_distinct_identities_and_match_the_added_capacity():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    declaration = json.loads((root / "configs/experiments/active-dendrites-v1.json").read_text())
    assert not declaration["default_enabled"] and not declaration["final_test_access"]
    base = api().FinancialConfig(
        titans_specification(), variant=declaration["variant"], hidden_size=32
    )
    models = [
        api().FinancialPredictor(replace(base, **case["predictor_changes"]), dtype=torch.float64)
        for case in declaration["arms"].values()
    ]
    assert len({json.dumps(model.config.identity(), sort_keys=True) for model in models}) == 4
    counts = [sum(p.numel() for p in model.parameters()) for model in models]
    assert counts[1:] == [counts[0] + 32 * (4 * 4 + 1)] * 3
    for name, value in models[0].named_parameters():
        assert all(torch.equal(value, dict(other.named_parameters())[name]) for other in models[1:])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("variant", ["transformer_direct", "mac_online"])
def test_cpu_precisions_keep_finite_outputs_and_context_gradients(dtype, variant):
    model, _ = configured(variant=variant)
    model = model.to(dtype=dtype)
    activate(model)
    batch = api().DecisionBatch.from_corpus(raw(), model.config.inputs, dtype=dtype)
    result = model.prepare(
        batch,
        model.initial_state(batch.flow_ids, differentiable=True),
        differentiable=True,
        session_complete=True,
    )
    result.point_predictions.sum().backward()
    assert result.point_predictions.dtype == dtype
    assert torch.isfinite(result.point_predictions).all()
    assert model.active_dendrites.segment_weights.grad.abs().sum() > 0


@pytest.mark.cuda
def test_cuda_financial_context_and_recovery_match_fp32_cpu():
    from mars_titan.models.titans.dendritic_regime import regime_context

    cpu, _ = configured()
    cpu = cpu.float()
    activate(cpu)
    gpu = api().FinancialPredictor(cpu.config, device="cuda:0")
    gpu.load_state_dict(cpu.state_dict())
    before = api().DecisionBatch.from_corpus(raw(), cpu.config.inputs, dtype=torch.float32)
    batch = api().DecisionBatch.from_corpus(
        raw(), cpu.config.inputs, dtype=torch.float32, device="cuda:0"
    )
    expected = cpu.prepare(before, cpu.initial_state(before.flow_ids), session_complete=True)
    actual = gpu.prepare(batch, gpu.initial_state(batch.flow_ids), session_complete=True)
    torch.testing.assert_close(
        actual.point_predictions.cpu(), expected.point_predictions, rtol=2e-5, atol=2e-5
    )
    assert (
        regime_context(gpu.config.active_dendrites, batch, seed=gpu.config.seed).values.device.type
        == "cuda"
    )
    restored = gpu.restore_state(gpu.export_state_cpu(actual.next_state), device="cuda:0")
    following = api().DecisionBatch.from_corpus(
        raw(batch.prediction_at[0] + 10), gpu.config.inputs, dtype=torch.float32, device="cuda:0"
    )
    left = gpu.prepare(following, restored, session_complete=True)
    right = gpu.prepare(following, actual.next_state, session_complete=True)
    assert torch.equal(left.point_predictions, right.point_predictions)
