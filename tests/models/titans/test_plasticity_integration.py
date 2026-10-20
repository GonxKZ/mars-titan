"""Estado, causalidad y operador completo de las variantes de plasticidad."""

from dataclasses import replace

import pytest
import torch

from mars_titan.models.titans.config import MACConfig, MemoryConfig, MemoryMultiscale
from mars_titan.models.titans.mac import TitansMAC
from mars_titan.models.titans.neural_memory import NeuralMemory
from mars_titan.models.titans.plasticity_config import MemoryPlasticity
from mars_titan.models.titans.state import join_mac_rows, mac_tensors, map_mac_rows
from mars_titan.models.titans.transition_jacobian import (
    fast_state_dimension,
    fast_state_jacobian,
    fast_state_point,
    fast_state_transition,
)


def memory(plasticity, *, cascade=False, depth=1, dim=3, **options):
    config = MemoryConfig(
        dim=dim,
        depth=depth,
        plasticity=plasticity,
        multiscale=MemoryMultiscale() if cascade else None,
        **options,
    )
    return NeuralMemory(config, dtype=torch.float64)


def observations(batch=2, count=4, dim=3):
    return torch.arange(batch * count * dim, dtype=torch.float64).reshape(batch, count, dim) / 20


def same_state(left, right):
    assert left.config_id == right.config_id
    for a, b in zip(
        (*left.weights, *left.momentum, *left.levels, *left.modulation, left.steps),
        (*right.weights, *right.momentum, *right.levels, *right.modulation, right.steps),
        strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_internal_steps_reuse_the_checked_boundary_and_match_the_public_reference(monkeypatch):
    from mars_titan.models.titans import neural_memory, plasticity

    model = memory(MemoryPlasticity(capture="eligibility"), cascade=True)
    tokens = observations()
    state = model.initial_state(2)
    calls = []
    validate = plasticity._validate

    def counted(*args):
        calls.append(1)
        return validate(*args)

    monkeypatch.setattr(plasticity, "_validate", counted)
    optimized = model.update(tokens, state, differentiable=True)
    optimized_loss = sum(x.sum() for x in (*optimized.weights, *optimized.modulation))
    optimized_grads = torch.autograd.grad(
        optimized_loss, tuple(model.parameters()), allow_unused=True
    )
    assert calls == []
    monkeypatch.setattr(neural_memory, "advance_plasticity", plasticity.advance)
    reference = model.update(tokens, state, differentiable=True)
    reference_loss = sum(x.sum() for x in (*reference.weights, *reference.modulation))
    reference_grads = torch.autograd.grad(
        reference_loss, tuple(model.parameters()), allow_unused=True
    )
    same_state(optimized, reference)
    assert len(calls) == tokens.shape[1]
    for actual, expected in zip(optimized_grads, reference_grads, strict=True):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cascade", [False, True])
def test_unity_controls_preserve_the_original_weights_and_rng(cascade):
    base = memory(None, cascade=cascade)
    enabled = memory(
        MemoryPlasticity(
            write="constant",
            write_constant=1,
            capture="constant" if cascade else "none",
            capture_constant=1,
        ),
        cascade=cascade,
    )
    assert "plasticity" not in base.config.identity()
    first = base.initial_state(2)
    assert first.modulation == () and "modulation" not in base.export_state(first)
    rng = torch.random.get_rng_state().clone()
    left = base.update(observations(), first)
    right = enabled.update(observations(), enabled.initial_state(2))
    for a, b in zip(
        (*left.weights, *left.momentum, *left.levels),
        (*right.weights, *right.momentum, *right.levels),
        strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(right.modulation) == 3
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("kind", ["write", "capture", "combined"])
def test_sequence_partition_and_state_recovery_keep_the_next_update(kind):
    config = MemoryPlasticity(
        write="none" if kind == "capture" else "coherence",
        capture="none" if kind == "write" else "eligibility",
    )
    model = memory(config, cascade=kind != "write", depth=2)
    tokens = observations()
    whole = model.update(tokens, model.initial_state(2))
    prefix = model.update(tokens[:, :2].contiguous(), model.initial_state(2))
    restored = model.restore_state(model.export_state(prefix))
    assert all(not x.requires_grad for x in restored.modulation)
    actual = model.update(tokens[:, 2:].contiguous(), restored)
    same_state(whole, actual)
    assert all(torch.count_nonzero(x) == 0 for x in model.initial_state(2).modulation)


def test_future_suffix_cannot_change_the_prefix_or_its_gradients():
    model = memory(MemoryPlasticity(capture="eligibility"), cascade=True)
    tokens = observations(count=3).requires_grad_(True)
    state = model.update(tokens[:, :2].contiguous(), model.initial_state(2), differentiable=True)
    loss = sum(x.square().sum() for x in (*state.weights, *state.levels, *state.modulation))
    (gradient,) = torch.autograd.grad(loss, tokens)
    assert torch.count_nonzero(gradient[:, 2]) == 0
    assert torch.isfinite(gradient).all() and gradient[:, :2].abs().sum() > 0


def test_inference_does_not_retain_the_graph_of_received_plasticity_history():
    model = memory(MemoryPlasticity(capture="eligibility"), cascade=True)
    tokens = observations()
    with_graph = model.update(
        tokens[:, :2].contiguous(), model.initial_state(2), differentiable=True
    )
    assert any(value.requires_grad for value in with_graph.modulation)
    with torch.no_grad():
        following = model.update(tokens[:, 2:].contiguous(), with_graph)
    assert all(
        not value.requires_grad and value.grad_fn is None
        for value in (
            *following.weights,
            *following.momentum,
            *following.levels,
            *following.modulation,
        )
    )


@pytest.mark.parametrize("field", ["missing", "negative_energy", "bad_tag", "alias", "dtype"])
def test_incompatible_modulation_payload_is_rejected(field):
    model = memory(MemoryPlasticity())
    payload = model.export_state(model.initial_state(2))
    history = list(payload["modulation"])
    if field == "missing":
        payload.pop("modulation")
    else:
        if field == "negative_energy":
            history[1].fill_(-1)
        if field == "bad_tag":
            history[2].fill_(1.1)
        if field == "alias":
            history[2] = history[1]
        if field == "dtype":
            history[0] = history[0].float()
        payload["modulation"] = tuple(history)
    with pytest.raises(ValueError):
        model.restore_state(payload)


def test_auxiliary_storage_is_included_before_allocating_a_batch(monkeypatch):
    # Una capa de 3x3: dos matrices, pasos y m/v/z añaden 15 float64 por flujo.
    base = memory(None, max_state_bytes=160)
    base.initial_state(1)
    active = memory(MemoryPlasticity(), max_state_bytes=160)

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("La memoria intentó asignar un estado fuera de presupuesto")

    monkeypatch.setattr(torch.Tensor, "repeat", forbidden_allocation)
    with pytest.raises(ValueError, match="bytes"):
        active.initial_state(1)


def test_write_coherence_scales_the_actual_momentum_and_weight_update():
    model = memory(MemoryPlasticity(beta=0.5, write_floor=0.25), dim=2)
    with torch.no_grad():
        model.initial_weights[0].copy_(torch.eye(2, dtype=torch.float64))
        model.key_projection.weight.copy_(torch.eye(2, dtype=torch.float64))
        for layer in (
            model.value_projection,
            model.alpha_projection,
            model.eta_projection,
            model.theta_projection,
        ):
            layer.weight.zero_()
    state = model.initial_state(1)
    mean, energy, tag = state.modulation
    mean[0, 0, 0] = -2
    energy[0, 0, 0] = 8
    state = replace(state, steps=torch.tensor([1]))
    following = model.update(torch.tensor([[[1.0, 0.0]]], dtype=torch.float64), state)
    # g=(2,0) en la primera fila cancela la media previa: theta efectiva=0,05*0,25.
    torch.testing.assert_close(
        following.momentum[0],
        torch.tensor([[[-0.025, 0.0], [0.0, 0.0]]], dtype=torch.float64),
        rtol=0,
        atol=1e-15,
    )
    torch.testing.assert_close(
        following.weights[0],
        torch.tensor([[[0.475, 0.0], [0.0, 0.5]]], dtype=torch.float64),
        rtol=0,
        atol=1e-15,
    )


def test_capture_scales_both_sides_of_the_first_cascade_interface():
    model = memory(
        MemoryPlasticity(write="none", capture="constant", capture_constant=0.25),
        cascade=True,
        dim=2,
    )
    with torch.no_grad():
        model.key_projection.weight.zero_()
        model.value_projection.weight.zero_()
        model.alpha_projection.weight.zero_()
    state = model.initial_state(1)
    state.weights[0].fill_(1)
    for level in state.levels:
        level.zero_()
    following = model.update(torch.zeros(1, 1, 2, dtype=torch.float64), state)
    torch.testing.assert_close(
        following.weights[0], torch.full((1, 2, 2), 0.484375, dtype=torch.float64), rtol=0, atol=0
    )
    torch.testing.assert_close(
        following.levels[0], torch.full((1, 2, 2), 0.0078125, dtype=torch.float64), rtol=0, atol=0
    )
    assert torch.count_nonzero(following.levels[1]) == 0


def test_capture_requires_the_declared_cascade():
    with pytest.raises(ValueError):
        memory(MemoryPlasticity(capture="eligibility"))
    with pytest.raises(ValueError):
        MemoryConfig(
            3,
            plasticity=MemoryPlasticity(capture="constant"),
            multiscale=MemoryMultiscale(form="ema_bank"),
        )


def test_shared_memory_updates_one_history_independent_of_input_order():
    model = memory(MemoryPlasticity(capture="eligibility"), cascade=True)
    token = observations(4, 1).squeeze(1)
    left = model.update_shared(token, model.initial_state(1))
    right = model.update_shared(token.flip(0), model.initial_state(1))
    for a, b in zip(left.modulation, right.modulation, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)
    assert left.steps.tolist() == [1]


@pytest.mark.parametrize("kernel", [0, 2])
def test_mac_split_join_and_full_jacobian_include_every_auxiliary_value(kernel):
    config = MemoryConfig(
        2,
        depth=1,
        plasticity=MemoryPlasticity(capture="eligibility"),
        multiscale=MemoryMultiscale(),
        qkv_convolution=kernel,
    )
    mac = TitansMAC(MACConfig(config, heads=1, persistent_tokens=1), dtype=torch.float64)
    state = mac.initial_state(2)
    left = map_mac_rows(state, lambda value: value[:1].clone())
    right = map_mac_rows(state, lambda value: value[1:].clone())
    joined = join_mac_rows((left, right))
    for a, b in zip(mac_tensors(state), mac_tensors(joined), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # W, momentum, dos niveles y m/v/z: 16+4+2+2 = 24.
    size = 24 if kernel == 0 else 30
    assert fast_state_dimension(mac.config) == size
    assert fast_state_point(left).shape == (size,)
    token = torch.tensor([[[0.2, -0.1]]], dtype=torch.float64)
    jacobian = fast_state_jacobian(mac, token, left)
    assert jacobian.shape == (size, size) and torch.isfinite(jacobian).all()
    _, following = mac(token, left)
    torch.testing.assert_close(
        fast_state_transition(mac, token, left)(fast_state_point(left)),
        fast_state_point(following),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["write", "capture", "combined", "unity"])
def test_cuda_variants_preserve_recovery_gradients_and_cpu_reference(dtype, mode):
    options = MemoryPlasticity(
        write="constant" if mode == "unity" else "none" if mode == "capture" else "coherence",
        capture="none" if mode == "write" else "constant" if mode == "unity" else "eligibility",
        write_constant=1,
        capture_constant=1,
    )
    config = MemoryConfig(
        4, depth=2, plasticity=options, multiscale=None if mode == "write" else MemoryMultiscale()
    )
    cpu = NeuralMemory(config, dtype=dtype)
    gpu = NeuralMemory(config, dtype=dtype, device="cuda:0")
    tokens = observations(2, 3, 4).to(dtype)
    reference = cpu.update(tokens, cpu.initial_state(2))
    rng = torch.cuda.get_rng_state(0).clone()
    initial = gpu.initial_state(2)
    result = gpu.update(tokens.to("cuda:0"), initial)
    for a, b in zip(
        (*reference.weights, *reference.modulation),
        (*result.weights, *result.modulation),
        strict=True,
    ):
        torch.testing.assert_close(
            a,
            b.cpu(),
            rtol=2e-5 if dtype == torch.float32 else 1e-11,
            atol=2e-6 if dtype == torch.float32 else 1e-12,
        )
    restored = gpu.restore_state(gpu.export_state(result))
    following = tokens[:, :1].contiguous().to("cuda:0")
    same_state(gpu.update(following, result), gpu.update(following, restored))
    differentiable = gpu.update(tokens.to("cuda:0"), initial, differentiable=True)
    loss = sum(x.square().sum() for x in (*differentiable.weights, *differentiable.modulation))
    gradients = torch.autograd.grad(loss, (gpu.key_projection.weight, gpu.theta_projection.weight))
    assert all(torch.isfinite(x).all() and x.abs().sum() > 0 for x in gradients)
    assert torch.equal(rng, torch.cuda.get_rng_state(0))
    assert all(not x.requires_grad for x in result.modulation)
    if mode == "unity":
        baseline = NeuralMemory(replace(config, plasticity=None), dtype=dtype, device="cuda:0")
        original = baseline.update(tokens.to("cuda:0"), baseline.initial_state(2))
        for a, b in zip(original.weights, result.weights, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
