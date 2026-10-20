"""Paridad, persistencia y puntos de inserción en la frontera financiera."""

from dataclasses import asdict, replace

import pytest
import torch
from test_financial_adapter import api, raw_batch, setup

from mars_titan.models.titans.config import PAPER_PROJECTIONS
from mars_titan.models.titans.key_competition_config import KeyCompetition
from mars_titan.models.titans.session_context import MARKET_MEMORY
from mars_titan.training.search_cases import checked_predictor_changes

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]


def test_financial_configuration_keeps_new_options_identified_and_validated():
    base, _ = setup()
    change = dict(memory_key_competition=asdict(KeyCompetition(center=True, retained=8)))
    assert checked_predictor_changes(change) == change
    config = replace(base.config, **change)
    assert isinstance(config.memory_key_competition, KeyCompetition)
    assert "memory_key_competition" not in base.config.identity()
    assert config.identity() != base.config.identity()
    with pytest.raises(ValueError):
        replace(config, memory_key_competition=dict(center=True, retained=8))
    with pytest.raises(ValueError):
        replace(config, memory_key_competition=KeyCompetition(retained=64))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_complete_financial_control_has_equal_gradients_and_no_extra_tensor_state(device, dtype):
    base, _ = setup()
    config = replace(
        base.config, memory_projections=PAPER_PROJECTIONS, attention_layout="sequence_contiguous_v1"
    )
    control = replace(config, memory_key_competition=KeyCompetition(retained=32))
    models = [api().FinancialPredictor(c, device=device, dtype=dtype) for c in (config, control)]
    assert models[0]._tensor_bytes_per_flow() == models[1]._tensor_bytes_per_flow()
    states = [None, None]
    for step in range(3):
        batch = api().DecisionBatch.from_corpus(
            raw_batch(at=1_609_459_200_000_000 + step * 10),
            config.inputs,
            device=device,
            dtype=dtype,
        )
        outputs = []
        for i, model in enumerate(models):
            if states[i] is None:
                states[i] = model.initial_state(batch.flow_ids, differentiable=True)
            result = model.prepare(batch, states[i], differentiable=True)
            states[i] = result.next_state
            outputs.append(result.point_predictions)
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
    gradients = [
        torch.autograd.grad(output.square().sum(), tuple(model.parameters()), allow_unused=True)
        for output, model in zip(outputs, models, strict=True)
    ]
    for left, right in zip(*gradients, strict=True):
        if left is None:
            assert right is None
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_active_features_recover_the_next_prediction_with_market_context(device):
    base, _ = setup()
    config = replace(
        base.config,
        memory_key_competition=KeyCompetition(center=True, retained=8),
        session_context=MARKET_MEMORY,
        attention_layout="sequence_contiguous_v1",
    )
    model = api().FinancialPredictor(config, device=device, dtype=torch.float64)
    assert model.mac.config.memory.key_competition == config.memory_key_competition
    assert model.session_context.memory.config.key_competition == config.memory_key_competition
    batch = api().DecisionBatch.from_corpus(
        raw_batch(), config.inputs, device=device, dtype=torch.float64
    )
    state = model.initial_state(batch.flow_ids)
    first = model.prepare(batch, state, context=model.initial_context())
    context = model.commit_context(first.next_context)
    restored_context = model.restore_context(model.export_context(context))
    restored = model.restore_state(model.export_state_cpu(first.next_state), device=device)
    following = api().DecisionBatch.from_corpus(
        raw_batch(at=1_609_459_200_000_010), config.inputs, device=device, dtype=torch.float64
    )
    actual = model.prepare(following, restored, context=restored_context)
    expected = model.prepare(following, first.next_state, context=context)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.unguarded_fit
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_complete_competition_control_matches_four_optimizer_updates(dtype):
    base, _ = setup()
    config = replace(
        base.config, memory_projections=PAPER_PROJECTIONS, attention_layout="sequence_contiguous_v1"
    )
    models = [
        api().FinancialPredictor(c, device="cuda:0", dtype=dtype)
        for c in (config, replace(config, memory_key_competition=KeyCompetition(retained=32)))
    ]
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3) for m in models]
    for step in range(4):
        for model, optimizer in zip(models, optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            state = None
            for decision in range(3):
                batch = api().DecisionBatch.from_corpus(
                    raw_batch(at=1_609_459_200_000_000 + (step * 3 + decision) * 10),
                    config.inputs,
                    device="cuda:0",
                    dtype=dtype,
                )
                if state is None:
                    state = model.initial_state(batch.flow_ids, differentiable=True)
                result = model.prepare(batch, state, differentiable=True)
                state = result.next_state
            result.point_predictions.square().sum().backward()
            optimizer.step()
            model._seal_parameters()
        for left, right in zip(models[0].parameters(), models[1].parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
