"""Activación explícita de PT4 antes de MAC, con estados y parámetros independientes."""

import numpy as np
import pytest
import torch
from test_market_memory import KINDS, SHARED, pt4
from test_memory_stability import decision

from mars_titan.models.titans.financial_inputs import DecisionBatch

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]


def context_model(kind, device):
    model = pt4(kind).to(device)
    weights = model.state_dict()
    weights["session_context.output"] = torch.eye(32, dtype=torch.float64, device=device) * 0.2
    model.load_state_dict(weights)
    return model


def batch_on(model, flows, event):
    batch = decision(model.config, flows, event)
    raw = dict(
        inputs={name: value.numpy().astype(np.float32) for name, value in batch.inputs.items()},
        presence=batch.presence.numpy(),
        sample_ids=list(batch.sample_ids),
        prediction_at=np.array(batch.prediction_at, dtype="datetime64[us]"),
        input_available_at=np.array(batch.input_available_at, dtype="datetime64[us]"),
    )
    return DecisionBatch.from_corpus(
        raw, model.config.inputs, device=model.head.weight.device, dtype=torch.float64
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
def test_context_off_matches_the_parent_without_context_and_performs_no_context_write(kind, device):
    model, control = context_model(kind, device), pt4().to(device)
    for name, parameter in control.named_parameters():
        if name.split(".", 1)[0] in SHARED:
            assert torch.equal(parameter, dict(model.named_parameters())[name])
    flows = ("US/AAA", "US/BBB")
    state, reference = model.initial_state(flows), control.initial_state(flows)
    context = model.initial_context()
    for event in range(3):
        batch = batch_on(model, flows, event)
        actual = model.prepare(batch, state, context=context, context_selection=(False, False))
        expected = control.prepare(batch, reference)
        assert torch.equal(actual.point_predictions, expected.point_predictions)
        assert actual.next_context is context
        context = model.commit_context(actual.next_context)
        assert context.committed_at == (-1, -1)
        state, reference = actual.next_state, expected.next_state
    assert state.observed_steps.tolist() == [3, 3]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
def test_all_selected_preserves_the_original_path_and_only_selected_tokens_enter_pending(
    kind, device
):
    model = context_model(kind, device)
    flows = ("US/AAA", "US/BBB")
    state, context = model.initial_state(flows), model.initial_context()
    first = batch_on(model, flows, 0)
    baseline = model.prepare(first, state, context=context)
    selected = model.prepare(first, state, context=context, context_selection=(True, True))
    assert torch.equal(baseline.point_predictions, selected.point_predictions)
    assert torch.equal(
        baseline.next_context.pending_tokens[0], selected.next_context.pending_tokens[0]
    )
    context = model.commit_context(selected.next_context)
    observed = []
    handle = model.mac.register_forward_pre_hook(
        lambda _module, args: observed.append(args[0].detach().clone())
    )
    try:
        mixed = model.prepare(
            batch_on(model, flows, 1),
            selected.next_state,
            context=context,
            context_selection=(True, False),
        )
    finally:
        handle.remove()
    assert mixed.next_context.pending_flows[0] == ("US/AAA",)
    assert torch.equal(mixed.next_context.pending_tokens[0][0], mixed.detached_tokens[0])
    assert torch.equal(observed[0][1, 0], mixed.detached_tokens[1])
    assert not torch.equal(observed[0][0, 0], mixed.detached_tokens[0])
    assert selected.next_state.observed_steps.tolist() == [1, 1]
    assert mixed.next_state.observed_steps.tolist() == [2, 2]


@pytest.mark.parametrize("selection", [(True,), (1, 0), [True, False], True])
def test_invalid_selection_is_a_contract_error_without_mutating_state(selection):
    model = context_model(KINDS[0], "cpu")
    flows = ("US/AAA", "US/BBB")
    state, context = model.initial_state(flows), model.initial_context()
    with pytest.raises(ValueError, match="selección"):
        model.prepare(
            batch_on(model, flows, 0), state, context=context, context_selection=selection
        )
    assert state.observed_steps.tolist() == [0, 0]
    assert context.pending_flows == ((), ())


def test_context_free_parent_rejects_an_unused_selection():
    model = pt4()
    flows = ("US/AAA",)
    with pytest.raises(ValueError, match="contexto"):
        model.prepare(
            batch_on(model, flows, 0), model.initial_state(flows), context_selection=(False,)
        )
