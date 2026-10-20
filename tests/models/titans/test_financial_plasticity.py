"""Variantes de plasticidad en la frontera financiera con entradas técnicas enmascaradas."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from test_financial_adapter import api, raw_batch, setup

from mars_titan.models.titans.config import MemoryMultiscale
from mars_titan.models.titans.plasticity_config import MemoryPlasticity
from mars_titan.models.titans.state import mac_tensors
from mars_titan.training.search_cases import checked_predictor_changes


def configured():
    base, batch = setup()
    config = replace(
        base.config,
        memory_plasticity=asdict(MemoryPlasticity(capture="eligibility")),
        memory_multiscale=MemoryMultiscale(),
    )
    return base, api().FinancialPredictor(config, dtype=torch.float64), batch


def test_financial_recipe_and_byte_accounting_include_the_complete_auxiliary_state():
    base, model, batch = configured()
    assert isinstance(model.config.memory_plasticity, MemoryPlasticity)
    assert "memory_plasticity" in model.config.identity()
    assert "memory_plasticity" not in base.config.identity()
    assert model.mac.config.memory.plasticity == model.config.memory_plasticity
    control = api().FinancialPredictor(
        replace(model.config, memory_plasticity=None), dtype=torch.float64
    )
    added = 2 * (32 * 32 + 2 * 32) * 8
    assert model._tensor_bytes_per_flow() - control._tensor_bytes_per_flow() == added
    state = model.initial_state(batch.flow_ids)
    assert (
        sum(value.numel() * value.element_size() for value in state.mac.memory.modulation)
        == 2 * added
    )


def test_cpu_export_restore_and_next_prediction_include_modulation():
    _, model, batch = configured()
    state = model.prepare(batch, model.initial_state(batch.flow_ids)).next_state
    payload = model.export_state_cpu(state)
    assert "modulation" in payload["mac"]["memory"]
    restored = model.restore_state(payload, device="cpu")
    for a, b in zip(mac_tensors(state.mac), mac_tensors(restored.mac), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.data_ptr() != b.data_ptr()
    next_batch = api().DecisionBatch.from_corpus(
        raw_batch(at=batch.prediction_at[0] + 10), model.config.inputs, dtype=torch.float64
    )
    expected = model.prepare(next_batch, state)
    actual = model.prepare(next_batch, restored)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)


def test_predictor_change_registry_admits_the_identified_neural_option():
    change = dict(memory_plasticity=asdict(MemoryPlasticity()))
    assert checked_predictor_changes(change) == change


@pytest.mark.parametrize("changes", [dict(extra=True), dict(beta=1), dict(write="posterior")])
def test_invalid_recipe_is_rejected(changes):
    base, _ = setup()
    values = asdict(MemoryPlasticity()) | changes
    with pytest.raises(ValueError):
        replace(base.config, memory_plasticity=values)


def test_catalogue_variants_share_parameters_and_keep_unity_prediction_parity():
    root = Path(__file__).resolve().parents[3]
    declaration = json.loads((root / "configs/experiments/neural-plasticity-v1.json").read_text())
    base, batch = setup()
    reference_parameters = dict(base.named_parameters())
    identities, outputs = set(), {}
    for name, arm in declaration["arms"].items():
        config = replace(base.config, **arm["predictor_changes"])
        identities.add(json.dumps(config.identity(), sort_keys=True))
        model = api().FinancialPredictor(config, dtype=torch.float64)
        for key, value in model.named_parameters():
            torch.testing.assert_close(value, reference_parameters[key], rtol=0, atol=0)
        outputs[name] = model.prepare(batch, model.initial_state(batch.flow_ids)).point_predictions
    assert len(identities) == len(declaration["arms"])
    torch.testing.assert_close(
        outputs["cascade_original"], outputs["cascade_unity"], rtol=0, atol=0
    )
    assert declaration["default_enabled"] is False
    assert declaration["final_test_access"] is False


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["write", "capture", "combined"])
def test_paper_projections_keep_cuda_gradients_and_next_prediction_after_recovery(dtype, kind):
    from mars_titan.models.titans.config import PAPER_PROJECTIONS

    base, _ = setup()
    config = replace(
        base.config,
        memory_projections=PAPER_PROJECTIONS,
        attention_layout="sequence_contiguous_v1",
        memory_multiscale=MemoryMultiscale() if kind != "write" else None,
        memory_plasticity=MemoryPlasticity(
            write="none" if kind == "capture" else "coherence",
            capture="none" if kind == "write" else "eligibility",
        ),
    )
    model = api().FinancialPredictor(config, device="cuda:0", dtype=dtype)
    reference = api().FinancialPredictor(config, dtype=dtype)
    initial_at = 1_609_459_200_000_000
    state = control = None
    for index in range(3):
        raw = raw_batch(at=initial_at + index * 10)
        batch = api().DecisionBatch.from_corpus(raw, config.inputs, device="cuda:0", dtype=dtype)
        cpu_batch = api().DecisionBatch.from_corpus(raw, config.inputs, dtype=dtype)
        if state is None:
            state = model.initial_state(batch.flow_ids, differentiable=True)
            control = reference.initial_state(batch.flow_ids, differentiable=True)
        actual = model.prepare(batch, state, differentiable=True)
        expected = reference.prepare(cpu_batch, control, differentiable=True)
        tolerance = 2e-5 if dtype == torch.float32 else 2e-12
        torch.testing.assert_close(
            actual.point_predictions.cpu(),
            expected.point_predictions,
            rtol=tolerance,
            atol=tolerance,
        )
        state, control = actual.next_state, expected.next_state
    actual.point_predictions.square().sum().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert model.mac.memory.key_projection.weight.grad.abs().sum() > 0
    restored = model.restore_state(model.export_state_cpu(state), device="cuda:0")
    batch = api().DecisionBatch.from_corpus(
        raw_batch(at=initial_at + 30), config.inputs, device="cuda:0", dtype=dtype
    )
    following = model.prepare(batch, state)
    recovered = model.prepare(batch, restored)
    torch.testing.assert_close(
        following.point_predictions, recovered.point_predictions, rtol=0, atol=0
    )
