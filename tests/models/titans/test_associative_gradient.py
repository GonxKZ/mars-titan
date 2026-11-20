"""El gradiente asociativo cerrado reproduce la ruta de autograd en valores y metagradientes."""

from dataclasses import replace

import pytest
import torch

from mars_titan.models.titans.config import (
    ANALYTIC_INNER_GRADIENT,
    GateBias,
    MACConfig,
    MemoryConfig,
    MemoryStability,
)
from mars_titan.models.titans.mac import TitansMAC
from mars_titan.models.titans.neural_memory import NeuralMemory

CASES = {
    "depth2_residual_gate_bias": dict(depth=2, residual_layer_norm=True, gate_bias=GateBias()),
    "depth2_plain": dict(depth=2),
    "depth1_residual": dict(depth=1, residual_layer_norm=True),
    "paper_projections": dict(depth=2, residual_layer_norm=True, qkv_silu=True, qkv_convolution=4),
    "stability_clip": dict(
        depth=2,
        residual_layer_norm=True,
        gate_bias=GateBias(eta=0.2),
        stability=MemoryStability(gradient_clip=0.5, gate_box=True),
    ),
}


def _pair(options, *, dim=6, batch=5, tokens=3):
    config = MemoryConfig(dim=dim, max_batch=batch, max_tokens=tokens, **options)
    reference = NeuralMemory(config, dtype=torch.float64)
    analytic = NeuralMemory(
        replace(config, inner_gradient=ANALYTIC_INNER_GRADIENT), dtype=torch.float64
    )
    # La misma semilla sortea los mismos parámetros. Copiarlos no es posible: la carga
    # exige el mismo contrato de configuración.
    for (name, a), (_, b) in zip(
        reference.named_parameters(), analytic.named_parameters(), strict=True
    ):
        assert torch.equal(a, b), name
    return reference, analytic


def _trace(memory, observed, probe, *, differentiable):
    state = memory.initial_state(observed.shape[0], differentiable=differentiable)
    surprises = []
    for step in observed.split(1, dim=1):
        state = memory.update(step, state, differentiable=differentiable, surprise=surprises)
    read = memory.read(probe, state)
    return state, read, torch.stack(surprises)


@pytest.mark.parametrize("case", sorted(CASES))
def test_states_surprise_and_metagradients_match_autograd_in_float64(case):
    reference, analytic = _pair(CASES[case])
    generator = torch.Generator().manual_seed(7)
    observed = torch.randn(5, 3, 6, generator=generator, dtype=torch.float64)
    probe = torch.randn(5, 1, 6, generator=generator, dtype=torch.float64)
    results = []
    for memory in (reference, analytic):
        tokens = observed.clone().requires_grad_(True)
        state, read, surprise = _trace(memory, tokens, probe, differentiable=True)
        objective = (read * probe).sum() + sum((m * m).sum() for m in state.momentum)
        grads = torch.autograd.grad(objective, [*memory.parameters(), tokens])
        results.append((state, read, surprise, grads))
    (s0, r0, e0, g0), (s1, r1, e1, g1) = results
    for a, b in zip((*s0.weights, *s0.momentum), (*s1.weights, *s1.momentum), strict=True):
        torch.testing.assert_close(b, a, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(r1, r0, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(e1, e0, rtol=1e-12, atol=1e-13)
    for a, b in zip(g0, g1, strict=True):
        torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-12)


def test_detached_path_matches_and_leaves_no_graph():
    reference, analytic = _pair(CASES["depth2_residual_gate_bias"])
    observed = torch.randn(5, 3, 6, dtype=torch.float64)
    probe = torch.randn(5, 1, 6, dtype=torch.float64)
    s0, r0, e0 = _trace(reference, observed, probe, differentiable=False)
    s1, r1, e1 = _trace(analytic, observed, probe, differentiable=False)
    for a, b in zip((*s0.weights, *s0.momentum), (*s1.weights, *s1.momentum), strict=True):
        torch.testing.assert_close(b, a, rtol=1e-12, atol=1e-13)
        assert b.grad_fn is None and not b.requires_grad
    torch.testing.assert_close(e1, e0, rtol=1e-12, atol=1e-13)


def test_mac_segment_matches_autograd_with_memory_read_after_the_write():
    memory = MemoryConfig(
        dim=8, max_batch=4, max_tokens=1, residual_layer_norm=True, gate_bias=GateBias()
    )
    configs = (
        MACConfig(memory=memory, heads=2, persistent_tokens=2, max_segment=1),
        MACConfig(
            memory=replace(memory, inner_gradient=ANALYTIC_INNER_GRADIENT),
            heads=2,
            persistent_tokens=2,
            max_segment=1,
        ),
    )
    modules = [TitansMAC(config, dtype=torch.float64) for config in configs]
    for a, b in zip(modules[0].parameters(), modules[1].parameters(), strict=True):
        assert torch.equal(a, b)
    segments = torch.randn(3, 4, 1, 8, dtype=torch.float64)
    outputs, grads = [], []
    for module in modules:
        state = module.initial_state(4, differentiable=True)
        emitted = []
        for segment in segments:
            out, state = module(segment, state, differentiable=True)
            emitted.append(out)
        total = torch.stack(emitted)
        outputs.append(total)
        grads.append(torch.autograd.grad(total.square().sum(), list(module.parameters())))
    torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-12, atol=1e-13)
    for a, b in zip(*grads, strict=True):
        torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-12)


def test_default_identity_is_unchanged_and_the_option_is_declared_when_used():
    default = MemoryConfig(dim=4)
    assert "inner_gradient" not in default.identity()
    analytic = MemoryConfig(dim=4, inner_gradient=ANALYTIC_INNER_GRADIENT)
    assert analytic.identity()["inner_gradient"]["form"] == ANALYTIC_INNER_GRADIENT
    assert analytic.fingerprint() != default.fingerprint()


def test_fused_write_is_declared_in_the_identity_and_limited_to_its_kernel():
    from mars_titan.models.titans.config import FUSED_INNER_GRADIENT

    fused = MemoryConfig(dim=64, inner_gradient=FUSED_INNER_GRADIENT)
    analytic = MemoryConfig(dim=64, inner_gradient=ANALYTIC_INNER_GRADIENT)
    assert fused.identity()["inner_gradient"]["form"] == FUSED_INNER_GRADIENT
    assert fused.fingerprint() not in {analytic.fingerprint(), MemoryConfig(dim=64).fingerprint()}
    for options in (
        dict(dim=32),
        dict(dim=64, depth=1),
        dict(dim=64, stability=MemoryStability(gradient_clip=0.5)),
    ):
        with pytest.raises(ValueError, match="fusionada"):
            MemoryConfig(inner_gradient=FUSED_INNER_GRADIENT, **options)


def test_unsupported_memories_and_unknown_names_are_rejected():
    with pytest.raises(ValueError, match="inner_gradient"):
        MemoryConfig(dim=4, inner_gradient="fused")
    from mars_titan.models.titans.huber_memory import HuberMemory

    with pytest.raises(ValueError, match="Huber"):
        MemoryConfig(dim=4, inner_gradient=ANALYTIC_INNER_GRADIENT, huber=HuberMemory())
