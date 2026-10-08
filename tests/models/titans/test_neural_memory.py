"""Referencias matemáticas CPU de memoria rápida, sin optimizador externo."""

import copy
import importlib
import io
from dataclasses import replace

import pytest
import torch


def api():
    return importlib.import_module("mars_titan.models.titans")


def memory(*, depth=1, **options):
    module = api()
    return module.NeuralMemory(
        module.MemoryConfig(dim=2, depth=depth, normalize_qk=False, **options), dtype=torch.float64
    )


def fixed_memory(**options):
    result = memory(**options)
    with torch.no_grad():
        result.key_projection.weight.copy_(torch.eye(2, dtype=torch.float64))
        result.value_projection.weight.copy_(
            torch.tensor([[0.6, 0.1], [-0.2, 0.5]], dtype=torch.float64)
        )
        result.alpha_projection.weight.copy_(
            torch.tensor([[0.4, -0.2], [-0.1, 0.3]], dtype=torch.float64)
        )
        result.eta_projection.weight.copy_(torch.tensor([[0.2, -0.1]], dtype=torch.float64))
        result.theta_projection.weight.copy_(torch.tensor([[-0.3, 0.4]], dtype=torch.float64))
    return result


def state_with_values(model, batch=1, *, requires_grad=False):
    state = model.initial_state(batch, differentiable=requires_grad)
    weight = (
        torch.tensor([[0.2, -0.1], [0.4, 0.3]], dtype=torch.float64)
        .expand(batch, -1, -1)
        .clone()
        .requires_grad_(requires_grad)
    )
    momentum = (
        torch.tensor([[0.01, -0.02], [0.05, 0.06]], dtype=torch.float64)
        .expand(batch, -1, -1)
        .clone()
        .requires_grad_(requires_grad)
    )
    return replace(
        state,
        weights=(weight,),
        momentum=(momentum,),
        steps=torch.full((batch,), 5, dtype=torch.int64),
    )


def manual_step(model, x, state):
    keys = model.key_projection(x)
    values = model.value_projection(x)
    alpha = model.alpha_projection(x).sigmoid().unsqueeze(-1)
    eta = model.eta_projection(x).sigmoid().unsqueeze(-1)
    theta = model.config.theta_max * model.theta_projection(x).sigmoid().unsqueeze(-1)
    predicted = torch.bmm(state.weights[0], keys.unsqueeze(-1)).squeeze(-1)
    gradient = 2 * (predicted - values).unsqueeze(-1) * keys.unsqueeze(-2)
    momentum = eta * state.momentum[0] - theta * gradient
    weights = (1 - alpha) * state.weights[0] + momentum
    return weights, momentum


def test_linear_update_matches_manual_equation_with_output_row_forgetting():
    model = fixed_memory(theta_max=0.5)
    state = state_with_values(model, batch=2)
    x = torch.tensor([[[0.3, -0.2]], [[0.1, 0.4]]], dtype=torch.float64)
    expected_weights, expected_momentum = manual_step(model, x[:, 0], state)
    actual = model.update(x, state, differentiable=False)
    torch.testing.assert_close(actual.weights[0], expected_weights, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(actual.momentum[0], expected_momentum, rtol=1e-13, atol=1e-13)
    assert actual.steps.tolist() == [6, 6]
    assert actual.weights[0].grad_fn is None and not actual.weights[0].requires_grad


def test_inner_gradient_holds_the_observed_input_fixed_without_cutting_outer_graph():
    model = fixed_memory()
    state = state_with_values(model, requires_grad=True)
    x = state.weights[0].sum(dim=-1).unsqueeze(1)
    expected_weights, _ = manual_step(model, x[:, 0], state)
    actual = model.update(x, state, differentiable=True)
    torch.testing.assert_close(actual.weights[0], expected_weights, rtol=1e-12, atol=1e-12)
    expected = torch.autograd.grad(
        expected_weights.square().sum(), state.weights[0], retain_graph=True
    )[0]
    observed = torch.autograd.grad(actual.weights[0].square().sum(), state.weights[0])[0]
    torch.testing.assert_close(observed, expected, rtol=1e-11, atol=1e-11)


def test_internal_derivative_does_not_accumulate_shared_grads_or_change_weights():
    model = fixed_memory()
    state = state_with_values(model, requires_grad=True)
    before = tuple(value.detach().clone() for value in model.parameters())
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    observed = state.weights[0].sum(-1).unsqueeze(1)
    original = state.weights[0].detach().clone()
    model.update(observed, state, differentiable=True)
    for parameter, old in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(parameter, old, rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.25), rtol=0, atol=0)
    assert state.weights[0].grad is None
    torch.testing.assert_close(state.weights[0], original, rtol=0, atol=0)


def test_momentum_survives_a_zero_current_gradient_and_forgetting_is_separate():
    model = fixed_memory()
    state = state_with_values(model)
    x = torch.zeros((1, 1, 2), dtype=torch.float64)
    actual = model.update(x, state)
    torch.testing.assert_close(actual.momentum[0], 0.5 * state.momentum[0])
    torch.testing.assert_close(actual.weights[0], 0.5 * state.weights[0] + 0.5 * state.momentum[0])


def test_reads_updates_and_counters_do_not_alias_or_mutate_input_state():
    model = fixed_memory()
    state = state_with_values(model)
    original = tuple(t.clone() for t in (*state.weights, *state.momentum, state.steps))
    x = torch.tensor([[[0.2, 0.1]]], dtype=torch.float64)
    read = model.read(x, state)
    read.zero_()
    actual = model.update(x, state)
    for tensor in (*actual.weights, *actual.momentum, actual.steps):
        tensor.zero_()
    for left, right in zip((*state.weights, *state.momentum, state.steps), original, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_each_flow_and_a_joint_permutation_match_independent_updates():
    model = fixed_memory()
    state = state_with_values(model, batch=2)
    x = torch.tensor([[[0.2, 0.1], [0.1, -0.3]], [[0.7, -0.2], [0.5, 0.4]]], dtype=torch.float64)
    result = model.update(x, state)
    for index in range(2):
        local = replace(
            state,
            weights=tuple(w[index : index + 1].clone() for w in state.weights),
            momentum=tuple(m[index : index + 1].clone() for m in state.momentum),
            steps=state.steps[index : index + 1].clone(),
        )
        separate = model.update(x[index : index + 1], local)
        torch.testing.assert_close(
            result.weights[0][index], separate.weights[0][0], rtol=1e-13, atol=1e-13
        )
    order = torch.tensor([1, 0])
    shuffled = replace(
        state,
        weights=tuple(w[order] for w in state.weights),
        momentum=tuple(m[order] for m in state.momentum),
        steps=state.steps[order],
    )
    reversed_result = model.update(x[order], shuffled)
    torch.testing.assert_close(reversed_result.weights[0][order], result.weights[0], rtol=0, atol=0)
    changed = x.clone()
    changed[1] += 2
    independent = model.update(changed, state)
    torch.testing.assert_close(independent.weights[0][0], result.weights[0][0], rtol=0, atol=0)
    assert not torch.equal(independent.weights[0][1], result.weights[0][1])


def test_two_layer_updates_pass_gradcheck_for_inputs_weights_and_momentum():
    model = memory(depth=2)
    state = model.initial_state(1, differentiable=True)
    values = torch.tensor([[[0.2, -0.1], [0.1, 0.3]]], dtype=torch.float64, requires_grad=True)
    parameters = (*state.weights, *(m.detach().requires_grad_() for m in state.momentum))

    def function(x, w1, w2, m1, m2):
        initial = replace(state, weights=(w1, w2), momentum=(m1, m2))
        result = model.update(x, initial, differentiable=True)
        return (*result.weights, *result.momentum)

    assert torch.autograd.gradcheck(function, (values, *parameters), eps=1e-6, atol=2e-5, rtol=2e-4)


def test_external_objective_reaches_initialization_keys_values_and_rates():
    model = memory(depth=2)
    state = model.initial_state(1, differentiable=True)
    inputs = torch.tensor([[[0.4, -0.2], [0.2, 0.3]]], dtype=torch.float64, requires_grad=True)
    final = model.update(inputs, state, differentiable=True)
    loss = model.read(inputs, final).square().sum()
    loss.backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    for projection in (
        model.key_projection,
        model.value_projection,
        model.alpha_projection,
        model.eta_projection,
        model.theta_projection,
    ):
        assert projection.weight.grad is not None
        assert (
            torch.isfinite(projection.weight.grad).all() and projection.weight.grad.abs().sum() > 0
        )
    assert all(
        weight.grad is not None and weight.grad.abs().sum() > 0 for weight in model.initial_weights
    )


def test_evaluation_has_no_persistent_graph_and_works_inside_no_grad():
    model = fixed_memory()
    state = model.initial_state(1, differentiable=True)
    inputs = torch.tensor([[[0.1, 0.2]]], dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        result = model.update(inputs, state, differentiable=False)
    assert all(
        not tensor.requires_grad and tensor.grad_fn is None
        for tensor in (*result.weights, *result.momentum, result.steps)
    )
    with torch.inference_mode(), pytest.raises(ValueError, match="inference|gradiente"):
        model.update(inputs, state)


def test_roundtrip_restores_the_next_read_and_update_without_aliases():
    model = fixed_memory()
    inputs = torch.tensor([[[0.1, 0.2], [0.2, -0.1]]], dtype=torch.float64)
    state = model.update(inputs, model.initial_state(1))
    payload = model.export_state(state)
    archive = io.BytesIO()
    torch.save(payload, archive)
    archive.seek(0)
    restored = model.restore_state(torch.load(archive, weights_only=True))
    expected = model.update(inputs, state)
    actual = model.update(inputs, restored)
    torch.testing.assert_close(
        model.read(inputs, restored), model.read(inputs, state), rtol=0, atol=0
    )
    torch.testing.assert_close(actual.weights[0], expected.weights[0], rtol=0, atol=0)
    payload["weights"][0].zero_()
    assert not torch.equal(payload["weights"][0], state.weights[0])
    assert not torch.equal(payload["weights"][0], restored.weights[0])


def test_same_dimensions_do_not_make_different_contracts_compatible():
    model = fixed_memory()
    state = model.initial_state(1)
    other = fixed_memory(theta_max=0.2)
    with pytest.raises(ValueError, match="configur|contrato"):
        other.read(torch.ones((1, 1, 2), dtype=torch.float64), state)
    with pytest.raises(ValueError, match="configur|contrato"):
        other.restore_state(model.export_state(state))
    with pytest.raises(ValueError, match="configur|contrato"):
        other.load_state_dict(model.state_dict())


@pytest.mark.parametrize(
    "bad",
    [
        dict(dim=True),
        dict(dim=0),
        dict(depth=3),
        dict(theta_max=float("nan")),
        dict(theta_max=0),
        dict(normalize_qk=1),
        dict(max_batch=True),
        dict(max_tokens=0),
        dict(max_state_bytes=0),
    ],
)
def test_invalid_configuration_is_rejected(bad):
    values = dict(dim=2)
    values.update(bad)
    with pytest.raises(ValueError):
        api().MemoryConfig(**values)


def test_batch_sequence_and_state_byte_limits_are_explicit():
    model = memory(max_batch=2, max_tokens=2, max_state_bytes=256)
    with pytest.raises(ValueError, match="lote|presupuesto"):
        model.initial_state(3)
    state = model.initial_state(1)
    with pytest.raises(ValueError, match="secuencia|segmento|presupuesto"):
        model.update(torch.zeros((1, 3, 2), dtype=torch.float64), state)
    small = memory(max_state_bytes=64)
    with pytest.raises(ValueError, match="bytes|presupuesto"):
        small.initial_state(1)


@pytest.mark.parametrize(
    "fault",
    [
        "nan_input",
        "infinite_weight",
        "bad_shape",
        "float_steps",
        "negative_steps",
        "overlapping_state",
        "expanded_state",
    ],
)
def test_invalid_inputs_or_state_are_rejected(fault):
    model = fixed_memory()
    state = model.initial_state(2)
    values = torch.ones((2, 1, 2), dtype=torch.float64)
    if fault == "nan_input":
        values[0, 0, 0] = float("nan")
    elif fault == "infinite_weight":
        state.weights[0][0, 0, 0] = float("inf")
    elif fault == "bad_shape":
        values = torch.ones((2, 2), dtype=torch.float64)
    elif fault == "float_steps":
        state = replace(state, steps=state.steps.double())
    elif fault == "negative_steps":
        state.steps[0] = -1
    elif fault == "overlapping_state":
        state = replace(state, momentum=state.weights)
    else:
        state = replace(state, weights=(state.weights[0][:1].expand(2, -1, -1),))
    with pytest.raises(ValueError):
        model.update(values, state)


def test_state_payload_rejects_boolean_version_and_unknown_fields():
    model = fixed_memory()
    payload = model.export_state(model.initial_state(1))
    for update in ({"schema_version": True}, {"unexpected": 0}):
        changed = copy.deepcopy(payload)
        changed.update(update)
        with pytest.raises(ValueError):
            model.restore_state(changed)


def test_loss_is_summed_per_flow_without_batch_or_feature_averaging():
    model = fixed_memory()
    state = state_with_values(model)
    token = torch.tensor([[[0.3, -0.2]]], dtype=torch.float64)
    single = model.update(token, state)
    duplicate = replace(
        state,
        weights=tuple(w.repeat(2, 1, 1) for w in state.weights),
        momentum=tuple(m.repeat(2, 1, 1) for m in state.momentum),
        steps=state.steps.repeat(2),
    )
    paired = model.update(token.repeat(2, 1, 1), duplicate)
    torch.testing.assert_close(paired.weights[0][0], single.weights[0][0], rtol=0, atol=0)
    torch.testing.assert_close(paired.weights[0][1], single.weights[0][0], rtol=0, atol=0)


def test_normalization_changes_keys_and_is_identified():
    raw = fixed_memory()
    normalized = api().NeuralMemory(replace(raw.config, normalize_qk=True), dtype=torch.float64)
    with torch.no_grad():
        for left, right in zip(normalized.parameters(), raw.parameters(), strict=True):
            left.copy_(right)
    state = state_with_values(raw)
    normalized_state = replace(state, config_id=normalized.config.fingerprint())
    token = torch.tensor([[[0.4, -0.2]]], dtype=torch.float64)
    actual = normalized.update(token, normalized_state)
    keys = torch.nn.functional.normalize(raw.key_projection(token[:, 0]), dim=-1)
    values = raw.value_projection(token[:, 0])
    gradient = (
        2
        * (torch.bmm(state.weights[0], keys.unsqueeze(-1)).squeeze(-1) - values).unsqueeze(-1)
        * keys.unsqueeze(-2)
    )
    theta = raw.config.theta_max * raw.theta_projection(token[:, 0]).sigmoid().unsqueeze(-1)
    eta = raw.eta_projection(token[:, 0]).sigmoid().unsqueeze(-1)
    expected = eta * state.momentum[0] - theta * gradient
    torch.testing.assert_close(actual.momentum[0], expected, rtol=1e-13, atol=1e-13)
    assert not torch.equal(actual.weights[0], raw.update(token, state).weights[0])


def test_state_budget_rejects_large_retained_storage_before_copying():
    model = memory(max_state_bytes=128)
    state = model.initial_state(1)
    retained = torch.zeros(100, dtype=torch.float64)
    state = replace(state, weights=(retained[:4].view(1, 2, 2),))
    with pytest.raises(ValueError, match="presupuesto"):
        model.export_state(state)


def test_counter_overflow_and_wrong_input_precision_are_rejected():
    model = memory()
    state = model.initial_state(1)
    with pytest.raises(ValueError, match="tipo"):
        model.update(torch.ones((1, 1, 2), dtype=torch.float32), state)
    state = replace(state, steps=torch.full((1,), torch.iinfo(torch.int64).max, dtype=torch.int64))
    with pytest.raises(ValueError, match="contador"):
        model.update(torch.ones((1, 1, 2), dtype=torch.float64), state)


def test_zero_keys_and_values_are_finite_under_normalization():
    model = api().NeuralMemory(api().MemoryConfig(dim=2), dtype=torch.float64)
    next_state = model.update(torch.zeros((1, 2, 2), dtype=torch.float64), model.initial_state(1))
    assert all(torch.isfinite(w).all() for w in next_state.weights)
    assert all(torch.equal(m, torch.zeros_like(m)) for m in next_state.momentum)


@pytest.mark.parametrize("projection", ["alpha_projection", "eta_projection", "theta_projection"])
def test_nonfinite_rate_logits_are_rejected_even_if_sigmoid_would_saturate(projection):
    model = fixed_memory()
    with torch.no_grad():
        getattr(model, projection).weight[0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN|infinito"):
        model.update(torch.ones((1, 1, 2), dtype=torch.float64), model.initial_state(1))
