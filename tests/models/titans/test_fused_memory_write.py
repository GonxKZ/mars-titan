"""La escritura fusionada de Triton reproduce la forma cerrada en valores y gradientes."""

from dataclasses import replace

import pytest
import torch

from mars_titan.models.titans.associative_gradient import associative_gradients
from mars_titan.models.titans.config import (
    ANALYTIC_INNER_GRADIENT,
    FUSED_INNER_GRADIENT,
    GateBias,
    MACConfig,
    MemoryConfig,
)
from mars_titan.models.titans.mac import TitansMAC
from mars_titan.models.titans.neural_memory import NeuralMemory

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="La escritura fusionada necesita CUDA"
)
CUDA = torch.device("cuda:0")
D = 64


def _reference(weights, momentum, keys, values, gates, layer_norm):
    alpha, eta, theta = gates
    gradients, elements = associative_gradients(
        weights, keys, values, residual_layer_norm=layer_norm, eps=1e-5
    )
    following = tuple(eta * m - theta * g for m, g in zip(momentum, gradients, strict=True))
    written = tuple((1 - alpha) * w + s for w, s in zip(weights, following, strict=True))
    return written, following, elements


def _inputs(rows, dtype, seed):
    generator = torch.Generator(device=CUDA).manual_seed(seed)

    def draw(*shape, scale):
        value = torch.randn(*shape, generator=generator, device=CUDA, dtype=dtype) * scale
        return value.requires_grad_()

    keys = torch.randn(rows, D, generator=generator, device=CUDA, dtype=dtype)
    keys = torch.nn.functional.normalize(keys, dim=-1).requires_grad_()
    gates = tuple(
        (torch.rand(*shape, generator=generator, device=CUDA, dtype=dtype) * top).requires_grad_()
        for shape, top in (((rows, D, 1), 0.1), ((rows, 1, 1), 0.9), ((rows, 1, 1), 0.05))
    )
    weights = (draw(rows, D, D, scale=0.15), draw(rows, D, D, scale=0.15))
    momentum = (draw(rows, D, D, scale=0.01), draw(rows, D, D, scale=0.01))
    return weights, momentum, keys, draw(rows, D, scale=0.5), gates


def _flat(result):
    written, following, elements = result
    return (*written, *following, elements)


def _close(actual, expected, rtol):
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=rtol, atol=rtol * b.abs().max().item())


@pytest.mark.parametrize("layer_norm", [True, False])
@pytest.mark.parametrize(("dtype", "rtol"), [(torch.float64, 1e-12), (torch.float32, 2e-5)])
@pytest.mark.parametrize("rows", [1, 37])
def test_write_and_its_vjp_match_the_closed_form(layer_norm, dtype, rtol, rows):
    from mars_titan.models.titans.fused_memory_write import fused_associative_write

    weights, momentum, keys, values, gates = _inputs(rows, dtype, seed=rows)
    leaves = (*weights, *momentum, keys, values, *gates)
    expected = _flat(_reference(weights, momentum, keys, values, gates, layer_norm))
    actual = _flat(
        fused_associative_write(
            weights, momentum, keys, values, gates, residual_layer_norm=layer_norm, eps=1e-5
        )
    )
    _close(actual, expected, rtol)
    upstream = [torch.randn_like(value) for value in expected]
    _close(
        torch.autograd.grad(actual, leaves, upstream),
        torch.autograd.grad(expected, leaves, upstream),
        rtol,
    )


def test_the_write_is_deterministic_and_leaves_no_graph_without_grad():
    from mars_titan.models.titans.fused_memory_write import fused_associative_write

    weights, momentum, keys, values, gates = _inputs(129, torch.float32, seed=3)
    with torch.no_grad():
        results = [
            _flat(
                fused_associative_write(
                    weights, momentum, keys, values, gates, residual_layer_norm=True, eps=1e-5
                )
            )
            for _ in range(2)
        ]
    for first, second in zip(*results, strict=True):
        assert torch.equal(first, second) and first.grad_fn is None


def test_inputs_outside_the_kernel_contract_are_rejected():
    from mars_titan.models.titans.fused_memory_write import fused_associative_write

    weights, momentum, keys, values, gates = _inputs(4, torch.float32, seed=5)
    with pytest.raises(ValueError, match="CUDA"):
        fused_associative_write(
            tuple(w.detach().cpu() for w in weights),
            tuple(m.detach().cpu() for m in momentum),
            keys.detach().cpu(),
            values.detach().cpu(),
            tuple(g.detach().cpu() for g in gates),
            residual_layer_norm=True,
            eps=1e-5,
        )
    with pytest.raises(ValueError, match="dimensión"):
        fused_associative_write(
            weights, momentum, keys[:, :32], values, gates, residual_layer_norm=True, eps=1e-5
        )


CASES = {
    "residual_gate_bias": dict(residual_layer_norm=True, gate_bias=GateBias()),
    "plain": dict(),
    "paper_projections": dict(residual_layer_norm=True, qkv_silu=True, qkv_convolution=4),
}


def _pair(options, *, batch=5, tokens=3):
    config = MemoryConfig(
        dim=D, max_batch=batch, max_tokens=tokens, inner_gradient=ANALYTIC_INNER_GRADIENT, **options
    )
    analytic = NeuralMemory(config, dtype=torch.float64).to(CUDA)
    fused = NeuralMemory(
        replace(config, inner_gradient=FUSED_INNER_GRADIENT), dtype=torch.float64
    ).to(CUDA)
    for (name, a), (_, b) in zip(
        analytic.named_parameters(), fused.named_parameters(), strict=True
    ):
        assert torch.equal(a, b), name
    return analytic, fused


def _trace(memory, observed, probe, *, differentiable):
    state = memory.initial_state(observed.shape[0], differentiable=differentiable)
    surprises = []
    for step in observed.split(1, dim=1):
        state = memory.update(step, state, differentiable=differentiable, surprise=surprises)
    return state, memory.read(probe, state), torch.stack(surprises)


@pytest.mark.parametrize("case", sorted(CASES))
def test_memory_states_surprise_and_metagradients_match_the_closed_form(case):
    analytic, fused = _pair(CASES[case])
    generator = torch.Generator(device=CUDA).manual_seed(11)
    observed = torch.randn(5, 3, D, generator=generator, device=CUDA, dtype=torch.float64)
    probe = torch.randn(5, 1, D, generator=generator, device=CUDA, dtype=torch.float64)
    results = []
    for memory in (analytic, fused):
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
    # Las sumas de 64 términos cambian de orden respecto de cuBLAS, y la convolución de las
    # proyecciones del artículo encadena esos redondeos: la tolerancia sigue la escala de cada
    # gradiente, como en la comparación del kernel.
    for a, b in zip(g0, g1, strict=True):
        torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-10 * a.abs().max().item())


def test_detached_memory_path_matches_and_leaves_no_graph():
    analytic, fused = _pair(CASES["residual_gate_bias"])
    observed = torch.randn(5, 3, D, device=CUDA, dtype=torch.float64)
    probe = torch.randn(5, 1, D, device=CUDA, dtype=torch.float64)
    s0, _, e0 = _trace(analytic, observed, probe, differentiable=False)
    s1, _, e1 = _trace(fused, observed, probe, differentiable=False)
    for a, b in zip((*s0.weights, *s0.momentum), (*s1.weights, *s1.momentum), strict=True):
        torch.testing.assert_close(b, a, rtol=1e-12, atol=1e-13)
        assert b.grad_fn is None and not b.requires_grad
    torch.testing.assert_close(e1, e0, rtol=1e-12, atol=1e-13)


def test_mac_segments_match_the_closed_form_with_the_read_after_the_write():
    memory = MemoryConfig(
        dim=D,
        max_batch=4,
        max_tokens=1,
        residual_layer_norm=True,
        gate_bias=GateBias(),
        inner_gradient=ANALYTIC_INNER_GRADIENT,
    )
    modules = [
        TitansMAC(
            MACConfig(memory=config, heads=2, persistent_tokens=2, max_segment=1),
            dtype=torch.float64,
        ).to(CUDA)
        for config in (memory, replace(memory, inner_gradient=FUSED_INNER_GRADIENT))
    ]
    for a, b in zip(modules[0].parameters(), modules[1].parameters(), strict=True):
        assert torch.equal(a, b)
    segments = torch.randn(3, 4, 1, D, device=CUDA, dtype=torch.float64)
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
