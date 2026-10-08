"""Control C en el predictor común, con fixtures sin objetivos financieros."""

import copy
from dataclasses import replace

import pytest
import torch
from test_financial_adapter import api, raw_batch, setup

from mars_titan.models.titans.local_control import MACProjectionConfig


def configured(mode="diagnostic", *, max_flows=1):
    original, batch = setup()
    control = MACProjectionConfig(
        mode=mode,
        rank=2,
        frequency=1,
        grid_size=13,
        threshold=0.0,
        weight=0.2 if mode == "penalty" else 0.0,
        max_flows=max_flows,
    )
    model = api().FinancialPredictor(original.config, local_control=control, dtype=torch.float64)
    return model, batch


def prepare(model, batch, state, *, differentiable=False, selection=None):
    if selection is None:
        selection = model.local_control.select_flows(
            batch.flow_ids, state.observed_steps, context_id="a" * 64
        )
    return model.prepare(
        batch,
        state,
        control_selection=selection,
        control_context_id="a" * 64,
        differentiable=differentiable,
    )


def test_none_keeps_previous_identity_and_working_state_matches_the_head():
    original, batch = setup()
    explicit = api().FinancialPredictor(original.config, local_control=None, dtype=torch.float64)
    assert original.get_extra_state() == explicit.get_extra_state()
    assert "local_control" not in explicit.get_extra_state()
    first = original.prepare(batch, original.initial_state(batch.flow_ids))
    second = explicit.prepare(batch, explicit.initial_state(batch.flow_ids))
    torch.testing.assert_close(first.point_predictions, second.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(
        explicit.head(second.working_state).squeeze(-1), second.point_predictions, rtol=0, atol=0
    )
    assert not second.working_state.requires_grad
    assert second.local_control is None


def test_explicit_disabled_and_diagnostic_share_math_and_one_real_update(monkeypatch):
    baseline, batch = configured("disabled")
    diagnostic, _ = configured("diagnostic")
    receipt = api().copy_paired_parameters(baseline, diagnostic)
    assert (
        receipt["local_control"]["basis_sha256"]
        == baseline.local_control.get_extra_state()["basis_sha256"]
    )
    state = diagnostic.initial_state(batch.flow_ids)
    calls = []
    original = diagnostic.mac.forward

    def record(segment, supplied, **options):
        calls.append(segment.shape[0])
        assert torch.backends.cuda.math_sdp_enabled()
        assert not torch.backends.cuda.flash_sdp_enabled()
        return original(segment, supplied, **options)

    monkeypatch.setattr(diagnostic.mac, "forward", record)
    first = baseline.prepare(batch, baseline.initial_state(batch.flow_ids))
    second = prepare(diagnostic, batch, state)
    torch.testing.assert_close(first.point_predictions, second.point_predictions, rtol=0, atol=0)
    assert calls == [2, 1, 1]
    assert second.local_control.reevaluations == 2
    assert second.next_state.mac.memory.steps.tolist() == [1, 1]
    assert state.mac.memory.steps.tolist() == [0, 0]
    for a, b in zip(
        first.next_state.mac.memory.weights, second.next_state.mac.memory.weights, strict=True
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_penalty_reaches_fusion_backbone_and_rates_without_accumulating_grad():
    model, batch = configured("penalty")
    result = prepare(model, batch, model.initial_state(batch.flow_ids), differentiable=True)
    assert result.working_state.requires_grad
    gradients = torch.autograd.grad(
        result.local_control.penalty,
        (
            model.fusion[0].weight,
            model.price_encoder.projection.weight,
            model.mac.memory.eta_projection.weight,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_group_penalty_and_selection_survive_physical_batches():
    model, batch = configured("penalty", max_flows=2)
    state = model.initial_state(batch.flow_ids)
    selection = model.local_control.select_flows(
        batch.flow_ids, state.observed_steps, context_id="a" * 64
    )
    whole = prepare(model, batch, state, differentiable=True, selection=selection)
    blocks = [
        prepare(
            model,
            batch.select([index]),
            model.select_state(state, [flow]),
            differentiable=True,
            selection=selection,
        )
        for index, flow in enumerate(batch.flow_ids)
    ]
    torch.testing.assert_close(
        sum(part.local_control.penalty for part in blocks),
        whole.local_control.penalty,
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        torch.cat([part.point_predictions for part in blocks]),
        whole.point_predictions,
        rtol=1e-10,
        atol=1e-12,
    )
    assert (
        sum(part.local_control.reevaluations for part in blocks)
        == whole.local_control.reevaluations
    )
    assert all(part.local_control.selection_id == selection.fingerprint() for part in blocks)


def test_control_state_roundtrip_preserves_next_prediction_and_rejects_other_mode():
    model, batch = configured("penalty")
    model.eval()
    first = prepare(model, batch, model.initial_state(batch.flow_ids))
    restored, _ = configured("penalty")
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    state = restored.restore_state(model.export_state(first.next_state))
    following = api().DecisionBatch.from_corpus(
        raw_batch(at=batch.prediction_at[0] + 10), model.config.inputs, dtype=torch.float64
    )
    expected = prepare(model, following, first.next_state)
    actual = prepare(restored, following, state)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.local_control.operators, expected.local_control.operators, rtol=0, atol=0
    )
    disabled, _ = configured("disabled")
    with pytest.raises(ValueError):
        disabled.load_state_dict(model.state_dict(), strict=False)


def test_corrupt_nested_basis_fails_before_other_parameters_are_copied():
    model, _ = configured()
    payload = copy.deepcopy(model.state_dict())
    payload["head.weight"] += 1
    payload["local_control.basis"][0, 0] += 0.01
    previous = model.head.weight.detach().clone()
    with pytest.raises(ValueError):
        model.load_state_dict(payload, strict=False)
    torch.testing.assert_close(model.head.weight, previous, rtol=0, atol=0)


def test_active_control_rejects_missing_selection_before_encoding(monkeypatch):
    model, batch = configured()
    monkeypatch.setattr(
        model.price_encoder, "forward", lambda *args: pytest.fail("No debe codificar")
    )
    with pytest.raises(ValueError, match="selección"):
        model.prepare(batch, model.initial_state(batch.flow_ids))


@pytest.mark.parametrize("variant", ["transformer_direct", "mac_frozen", "mac_disabled"])
def test_new_control_requires_online_mac(variant):
    model, _ = setup(variant)
    with pytest.raises(ValueError):
        api().FinancialPredictor(model.config, local_control=MACProjectionConfig())


def test_pairing_rejects_different_control_bases():
    source, batch = configured("disabled")
    target = api().FinancialPredictor(
        source.config,
        local_control=replace(source.local_control.config, seed=92),
        dtype=torch.float64,
    )
    with pytest.raises(ValueError):
        api().copy_paired_parameters(source, target)
    legacy, _ = setup()
    with pytest.raises(ValueError):
        api().copy_paired_parameters(legacy, source)
