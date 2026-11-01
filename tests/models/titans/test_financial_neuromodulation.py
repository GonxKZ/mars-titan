"""N1 sobre la actualización real de Titans, sin ajustes de parámetros."""

from dataclasses import replace

import pytest
import torch
from test_financial_adapter import raw_batch, setup

from mars_titan.models.titans.dual_neuromodulation import DualNeuromodulationConfig
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.reliability_gates import MatureReliability
from mars_titan.models.titans.state import mac_tensors


def pair():
    reference, batch = setup()
    config = replace(reference.config, dual_neuromodulation=DualNeuromodulationConfig())
    model = FinancialPredictor(config, dtype=torch.float64)
    return reference.eval(), model.eval(), batch


def test_zero_signals_preserve_shared_parameters_points_and_fast_state_exactly():
    reference, model, batch = pair()
    first = dict(reference.named_parameters())
    second = dict(model.named_parameters())
    assert first.keys() == second.keys()
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert "dual_neuromodulation" not in reference.config.identity()
    assert model.config.identity()["dual_neuromodulation"]["kind"] == "dual_neuromodulation_v1"
    expected = reference.prepare(batch, reference.initial_state(batch.flow_ids))
    previous = model.initial_state(batch.flow_ids)
    before = tuple(value.clone() for value in mac_tensors(previous.mac))
    result = model.prepare(
        batch, previous, neuromodulation=torch.zeros((2, 2), dtype=torch.float64)
    )
    assert torch.equal(result.point_predictions, expected.point_predictions)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            mac_tensors(result.next_state.mac), mac_tensors(expected.next_state.mac), strict=True
        )
    )
    assert all(torch.equal(a, b) for a, b in zip(mac_tensors(previous.mac), before, strict=True))


def test_actual_write_and_forget_gates_move_with_the_signals_but_momentum_does_not():
    reference, model, batch = pair()
    captured = []
    for predictor in (reference, model):
        values = []
        predictor.mac.memory.observer = lambda values=values, **row: values.append(
            tuple(row[name].detach().clone() for name in ("alpha", "eta", "theta"))
        )
        options = (
            {}
            if predictor is reference
            else dict(neuromodulation=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64))
        )
        predictor.prepare(batch, predictor.initial_state(batch.flow_ids), **options)
        captured.append(values[0])
    alpha0, eta0, theta0 = captured[0]
    alpha, eta, theta = captured[1]
    assert torch.equal(eta, eta0)
    assert torch.equal(alpha[0], alpha0[0]) and torch.all(alpha[1] > alpha0[1])
    assert torch.all(theta[0] < theta0[0]) and torch.all(theta[1] > theta0[1])


@pytest.mark.parametrize("kind", ["missing", "shape", "dtype"])
def test_configured_signal_is_mandatory_and_cannot_be_silently_replaced(kind):
    _, model, batch = pair()
    signal = None if kind == "missing" else torch.zeros((2, 3) if kind == "shape" else (2, 2))
    with pytest.raises(ValueError, match="N1"):
        model.prepare(batch, model.initial_state(batch.flow_ids), neuromodulation=signal)


def test_plain_predictor_refuses_an_unregistered_signal():
    model, batch = setup()
    with pytest.raises(ValueError, match="N1"):
        model.prepare(
            batch,
            model.initial_state(batch.flow_ids),
            neuromodulation=torch.zeros((2, 2), dtype=torch.float64),
        )


def test_recovery_and_next_prediction_use_the_same_explicit_signal():
    _, model, batch = pair()
    signal = torch.tensor([[0.2, 0.7], [0.6, 0.1]], dtype=torch.float64)
    result = model.prepare(batch, model.initial_state(batch.flow_ids), neuromodulation=signal)
    restored = model.restore_state(model.export_state(result.next_state))
    future = DecisionBatch.from_corpus(
        raw_batch(at=batch.prediction_at[0] + 1), model.config.inputs, dtype=torch.float64
    )
    first = model.prepare(future, result.next_state, neuromodulation=signal)
    second = model.prepare(future, restored, neuromodulation=signal)
    assert torch.equal(first.point_predictions, second.point_predictions)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            mac_tensors(first.next_state.mac), mac_tensors(second.next_state.mac), strict=True
        )
    )


def test_gradients_of_the_existing_write_projection_survive_the_new_control():
    _, model, batch = pair()
    result = model.prepare(
        batch,
        model.initial_state(batch.flow_ids),
        differentiable=True,
        neuromodulation=torch.ones((2, 2), dtype=torch.float64),
    )
    loss = sum(value.square().sum() for value in result.next_state.mac.memory.weights)
    (gradient,) = torch.autograd.grad(loss, model.mac.memory.theta_projection.weight)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("kind", ["variant", "reliability", "control"])
def test_unregistered_combinations_fail_before_executing(kind):
    model, _ = setup()
    fields = dict(dual_neuromodulation=DualNeuromodulationConfig())
    if kind == "variant":
        fields["variant"] = "mac_frozen"
    elif kind == "reliability":
        fields["mature_reliability"] = MatureReliability()
    with pytest.raises(ValueError, match="N1"):
        config = replace(model.config, **fields)
        FinancialPredictor(
            config,
            dtype=torch.float64,
            **(
                dict(local_control=MACProjectionConfig(mode="diagnostic"))
                if kind == "control"
                else {}
            ),
        )


def test_consumers_without_a_mature_source_cannot_enable_the_variant():
    _, model, _ = pair()
    model.requires_grad_(False)
    for consumer in (FrozenFinancialConsumer, FrozenMechanismConsumer):
        with pytest.raises(ValueError, match="N1"):
            consumer(model)


def test_config_mapping_and_runtime_configuration_are_verified():
    _, model, _ = pair()
    fields = dict(write_expected=1, write_unexpected=1, forget_unexpected=1, bound=1)
    mapped = replace(model.config, dual_neuromodulation=fields)
    assert mapped.identity() == model.config.identity()
    with pytest.raises(ValueError, match="N1"):
        replace(model.config, dual_neuromodulation={"bound": 1})
    model.dual_gates.config = DualNeuromodulationConfig(write_expected=0)
    with pytest.raises(ValueError, match="N1"):
        model.verify_parameter_identity()


def test_fp32_keeps_neutral_parity_and_applies_the_nonzero_control():
    parent, _ = setup()
    reference = FinancialPredictor(parent.config, dtype=torch.float32).eval()
    model = FinancialPredictor(
        replace(parent.config, dual_neuromodulation=DualNeuromodulationConfig()),
        dtype=torch.float32,
    ).eval()
    batch = DecisionBatch.from_corpus(raw_batch(), model.config.inputs, dtype=torch.float32)
    initial = model.initial_state(batch.flow_ids)
    plain = reference.prepare(batch, reference.initial_state(batch.flow_ids))
    neutral = model.prepare(batch, initial, neuromodulation=torch.zeros((2, 2)))
    assert torch.equal(plain.point_predictions, neutral.point_predictions)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            mac_tensors(plain.next_state.mac), mac_tensors(neutral.next_state.mac), strict=True
        )
    )
    changed = model.prepare(batch, initial, neuromodulation=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert any(
        not torch.equal(a, b)
        for a, b in zip(
            changed.next_state.mac.memory.weights,
            neutral.next_state.mac.memory.weights,
            strict=True,
        )
    )
