"""Bloque MAC comprobado al cierre del segmento, sin predictor ni campañas."""

import importlib
from dataclasses import replace

import pytest
import torch


def api():
    return importlib.import_module("mars_titan.models.titans")


def model(mode="online", **kwargs):
    package = api()
    memory = package.MemoryConfig(dim=4, depth=2, max_batch=2, max_tokens=3)
    config = package.MACConfig(
        memory=memory, heads=2, persistent_tokens=2, max_segment=3, memory_mode=mode, **kwargs
    )
    return package.TitansMAC(config, dtype=torch.float64)


def test_mac_outputs_only_at_segment_close_and_writes_only_s_positions():
    core = model()
    state = core.initial_state(2)
    x = torch.arange(16, dtype=torch.float64).reshape(2, 2, 4) / 20
    prediction, next_state = core(x, state)
    assert prediction.shape == (2, 4)
    assert next_state.memory.steps.tolist() == [2, 2]
    assert state.memory.steps.tolist() == [0, 0]
    assert not prediction.requires_grad
    assert all(not w.requires_grad for w in next_state.memory.weights)


def test_future_segments_do_not_change_previous_outputs_or_input_state():
    core = model()
    initial = core.initial_state(1)
    first = torch.tensor([[[0.2, 0.1, -0.1, 0.3], [0.3, -0.2, 0.1, 0.4]]], dtype=torch.float64)
    previous, next_state = core(first, initial)
    saved = previous.clone()
    core(first * 7, next_state)
    core(first * -9, next_state)
    repeated, _ = core(first, initial)
    torch.testing.assert_close(previous, saved, rtol=0, atol=0)
    torch.testing.assert_close(repeated, saved, rtol=0, atol=0)
    assert initial.memory.steps.tolist() == [0]


def test_mac_is_flow_permutation_equivariant_without_hidden_state():
    core = model()
    state = core.initial_state(2)
    x = torch.arange(16, dtype=torch.float64).reshape(2, 2, 4) / 11
    actual, after = core(x, state)
    order = torch.tensor([1, 0])
    shuffled = replace(
        state,
        memory=replace(
            state.memory,
            weights=tuple(w[order] for w in state.memory.weights),
            momentum=tuple(v[order] for v in state.memory.momentum),
            steps=state.memory.steps[order],
        ),
    )
    permuted, permuted_state = core(x[order], shuffled)
    torch.testing.assert_close(permuted[order], actual, rtol=1e-12, atol=1e-12)
    for left, right in zip(permuted_state.memory.weights, after.memory.weights, strict=True):
        torch.testing.assert_close(left[order], right, rtol=1e-12, atol=1e-12)


def test_mac_differentiation_reaches_attention_persistent_tokens_and_memory_update():
    core = model()
    state = core.initial_state(1, differentiable=True)
    x = torch.tensor(
        [[[0.2, 0.1, -0.1, 0.3], [0.3, -0.2, 0.1, 0.4]]], dtype=torch.float64, requires_grad=True
    )
    output, _ = core(x, state, differentiable=True)
    output.square().sum().backward()
    for value in (
        x,
        core.persistent,
        core.attention.in_proj_weight,
        core.memory.theta_projection.weight,
    ):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["frozen", "disabled"])
def test_controls_keep_weights_momentum_and_steps_unchanged_without_alias(mode):
    core = model(mode)
    state = core.initial_state(1)
    x = torch.ones((1, 2, 4), dtype=torch.float64)
    _, result = core(x, state)
    for left, right in zip(
        (*state.memory.weights, *state.memory.momentum, state.memory.steps),
        (*result.memory.weights, *result.memory.momentum, result.memory.steps),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert left.data_ptr() != right.data_ptr()


def test_disabled_memory_uses_the_same_attention_without_prefix_or_memory_gate():
    core = model("disabled")
    state = core.initial_state(1)
    x = torch.arange(8, dtype=torch.float64).reshape(1, 2, 4) / 10
    mask = torch.ones((2, 2), dtype=torch.bool).triu(1)
    with torch.no_grad():
        expected, _ = core.attention(x, x, x, attn_mask=mask, need_weights=False)
    actual, _ = core(x, state)
    torch.testing.assert_close(actual, expected[:, -1], rtol=1e-12, atol=1e-12)


def test_mac_roundtrip_preserves_next_output_and_rejects_incompatible_gate_contract():
    core = model()
    x = torch.ones((1, 2, 4), dtype=torch.float64)
    _, state = core(x, core.initial_state(1))
    replica = model()
    replica.load_state_dict(core.state_dict())
    restored = replica.restore_state(core.export_state(state))
    expected, _ = core(x, state)
    actual, _ = replica(x, restored)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    incompatible = model("frozen")
    with pytest.raises(ValueError, match="configur|contrato"):
        incompatible.restore_state(core.export_state(state))


def test_mac_configuration_identifies_the_declared_adaptation():
    core = model()
    configuration = core.config.identity()
    assert configuration["schema_version"] == 1
    assert configuration["layout"] == "persistent_memory_segment"
    assert configuration["output_gate"] == "hadamard"
    assert configuration["output_granularity"] == "segment_close"
    assert configuration["write_positions"] == "segment_only"


def test_mac_rejects_oversize_segments_and_nonfinite_values():
    core = model()
    state = core.initial_state(1)
    with pytest.raises(ValueError):
        core(torch.ones((1, 4, 4), dtype=torch.float64), state)
    with pytest.raises(ValueError):
        core(torch.full((1, 1, 4), float("nan"), dtype=torch.float64), state)


def test_mac_equation_reads_previous_memory_then_writes_s_and_applies_hadamard():
    core = model()
    state = core.initial_state(1)
    x = torch.arange(8, dtype=torch.float64).reshape(1, 2, 4) / 10
    with torch.no_grad():
        queries = torch.nn.functional.normalize(core.query_projection(x), dim=-1, eps=1e-12)
        retrieved = core.memory.read(queries, state.memory)
        packed = torch.cat((core.persistent.unsqueeze(0), retrieved, x), dim=1)
        mask = torch.ones(packed.shape[1:2] * 2, dtype=torch.bool).triu(1)
        attended, _ = core.attention(packed, packed, packed, attn_mask=mask, need_weights=False)
        y = attended[:, -2:]
        updated = core.memory.update(y, state.memory)
        expected = y[:, -1] * core.memory.read(y[:, -1:], updated).squeeze(1)
    actual, actual_state = core(x, state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for left, right in zip(actual_state.memory.weights, updated.weights, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_external_objective_gradcheck_includes_attention_y_of_previous_memory():
    package = api()
    core = package.TitansMAC(
        package.MACConfig(
            memory=package.MemoryConfig(dim=2, depth=1, max_tokens=2),
            persistent_tokens=1,
            max_segment=2,
        ),
        dtype=torch.float64,
    )
    state = core.initial_state(1, differentiable=True)
    x = torch.tensor([[[0.2, -0.3], [0.4, 0.1]]], dtype=torch.float64, requires_grad=True)
    momentum = state.memory.momentum[0].detach().requires_grad_()

    def objective(segment, weight, surprise):
        previous = replace(
            state, memory=replace(state.memory, weights=(weight,), momentum=(surprise,))
        )
        output, updated = core(segment, previous, differentiable=True)
        return output.square().sum() + 0.2 * updated.memory.weights[0].square().sum()

    assert torch.autograd.gradcheck(
        objective, (x, state.memory.weights[0], momentum), eps=1e-6, atol=1e-5, rtol=1e-4
    )


def test_packed_triangular_mask_does_not_claim_per_token_causality():
    package = api()
    core = package.TitansMAC(
        package.MACConfig(
            memory=package.MemoryConfig(dim=2, depth=1, normalize_qk=False, max_tokens=2),
            persistent_tokens=0,
            max_segment=2,
        ),
        dtype=torch.float64,
    )
    state = core.initial_state(1)
    state = replace(
        state,
        memory=replace(state.memory, weights=(torch.eye(2, dtype=torch.float64).unsqueeze(0),)),
    )
    with torch.no_grad():
        core.query_projection.weight.copy_(torch.eye(2, dtype=torch.float64))
        core.attention.in_proj_weight.copy_(
            torch.cat((torch.zeros(4, 2, dtype=torch.float64), torch.eye(2, dtype=torch.float64)))
        )
        core.attention.out_proj.weight.copy_(torch.eye(2, dtype=torch.float64))

    def first_token_attention(segment):
        retrieved = core.memory.read(core.query_projection(segment), state.memory)
        packed = torch.cat((retrieved, segment), dim=1)
        mask = torch.ones((4, 4), dtype=torch.bool).triu(1)
        return core.attention(packed, packed, packed, attn_mask=mask, need_weights=False)[0][:, 2]

    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    changed = x.clone()
    changed[:, 1] *= 3
    assert not torch.equal(first_token_attention(x), first_token_attention(changed))
    assert core(x, state)[0].shape == (1, 2)


def test_initialization_preserves_rng_and_attention_weights_across_controls():
    original = torch.get_rng_state().clone()
    online = model()
    frozen = model("frozen")
    disabled = model("disabled")
    torch.testing.assert_close(torch.get_rng_state(), original, rtol=0, atol=0)
    for other in (frozen, disabled):
        torch.testing.assert_close(
            online.attention.in_proj_weight, other.attention.in_proj_weight, rtol=0, atol=0
        )
        torch.testing.assert_close(online.persistent, other.persistent, rtol=0, atol=0)


def test_attention_budget_rejects_before_attention_allocation(monkeypatch):
    core = model(max_attention_elements=10)

    def forbidden(*args, **kwargs):
        pytest.fail("La atención no debe ejecutarse sobre el presupuesto")

    monkeypatch.setattr(core.attention, "forward", forbidden)
    with pytest.raises(ValueError, match="presupuesto"):
        core(torch.ones((1, 2, 4), dtype=torch.float64), core.initial_state(1))


@pytest.mark.parametrize(
    "options",
    [
        dict(heads=3),
        dict(persistent_tokens=-1),
        dict(max_segment=4),
        dict(memory_mode="unknown"),
        dict(max_attention_elements=0),
    ],
)
def test_mac_configuration_rejects_incompatible_values(options):
    package = api()
    values = dict(memory=package.MemoryConfig(dim=4, max_tokens=3), max_segment=3)
    values.update(options)
    with pytest.raises(ValueError):
        package.MACConfig(**values)


def test_model_configuration_is_rejected_before_parameters_are_overwritten():
    online = model()
    frozen = model("frozen")
    with torch.no_grad():
        frozen.attention.in_proj_weight.add_(1)
    before = frozen.attention.in_proj_weight.detach().clone()
    with pytest.raises(ValueError, match="contrato"):
        frozen.load_state_dict(online.state_dict())
    torch.testing.assert_close(frozen.attention.in_proj_weight, before, rtol=0, atol=0)


def test_nested_model_loading_rejects_missing_contract_even_with_strict_false():
    core = model()
    nested = torch.nn.Sequential(core)
    payload = nested.state_dict()
    del payload["0._extra_state"]
    with pytest.raises(ValueError, match="contrato"):
        nested.load_state_dict(payload, strict=False)
