"""Filas de la tabla ecuación → módulo → prueba de Titans-MAC que no tenían prueba propia.

Las ecuaciones siguen la numeración de las actas de NeurIPS 2025. Solo se calculan
forward, pérdidas y gradientes. Ninguna prueba crea un optimizador ni modifica parámetros.
"""

import importlib

import pytest
import torch
from test_financial_adapter import api as financial
from test_financial_adapter import raw_batch, setup, specification
from torch.nn import functional as F

from mars_titan.training.financial_run import _split, _stack

DAY = 86_400_000_000


def titans():
    return importlib.import_module("mars_titan.models.titans")


def campaign_memory():
    """Memoria con las opciones de la receta historical-masked, en FP64 y anchura reducida."""
    package = titans()
    config = package.MemoryConfig(
        dim=4,
        depth=2,
        max_batch=2,
        max_tokens=8,
        gate_bias=package.GateBias(256.0, 0.5, 0.05),
        residual_layer_norm=True,
    )
    return package.NeuralMemory(config, dtype=torch.float64)


def chunked_update(model, observed, state, chunk):
    """Forma por chunks de la sección 2.2: u_t = ∇ℓ(M_t′; x_t) con t′ el inicio del chunk.

    Después aplica la recurrencia (5) del momentum y el olvido de (3) token a token. Con
    chunk = 1, t′ = t − 1 y la forma coincide con la actualización secuencial de (3).
    """
    weights, momentum = state.weights, state.momentum
    for start in range(0, observed.shape[1], chunk):
        anchor = weights
        for token in observed[:, start : start + chunk].unbind(1):
            keys = F.normalize(model.key_projection(token), dim=-1, eps=1e-12)
            values = model.value_projection(token)
            alpha = model.alpha_projection(token).sigmoid().unsqueeze(-1)
            eta = model.eta_projection(token).sigmoid().unsqueeze(-1)
            theta = model.config.theta_max * model.theta_projection(token).sigmoid().unsqueeze(-1)
            local = tuple(w.clone().requires_grad_(True) for w in anchor)
            with torch.enable_grad():
                residual = model._apply_memory(keys.unsqueeze(1), local).squeeze(1) - values
                gradients = torch.autograd.grad(residual.square().sum(), local)
            momentum = tuple(eta * m - theta * g for m, g in zip(momentum, gradients, strict=True))
            weights = tuple((1 - alpha) * w + s for w, s in zip(weights, momentum, strict=True))
    return weights, momentum


def test_sequential_update_is_the_chunk_size_one_case_of_the_paper_parallel_form():
    model = campaign_memory()
    generator = torch.Generator().manual_seed(11)
    observed = torch.randn((2, 6, 4), generator=generator, dtype=torch.float64)
    start = model.initial_state(2)
    # Momentum no nulo para que (5) arrastre el término η_t S_{t−1} desde el primer token.
    start = type(start)(
        start.weights,
        tuple(
            0.05 * torch.randn(m.shape, generator=generator, dtype=m.dtype) for m in start.momentum
        ),
        start.steps,
        start.config_id,
    )
    with torch.no_grad():
        sequential = model.update(observed, start)
        weights, momentum = chunked_update(model, observed, start, 1)
        for left, right in zip(
            (*sequential.weights, *sequential.momentum), (*weights, *momentum), strict=True
        ):
            torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
        # Con chunks mayores los gradientes se toman en el estado del inicio del chunk. Es
        # otra trayectoria, no una forma más rápida de la misma, como recoge la tabla.
        for chunk in (2, 3):
            other, _ = chunked_update(model, observed, start, chunk)
            difference = max(
                float((a - b).abs().max()) for a, b in zip(other, sequential.weights, strict=True)
            )
            assert difference > 1e-3


def mac(mode="online"):
    package = titans()
    memory = package.MemoryConfig(dim=4, depth=2, max_batch=2, max_tokens=1)
    config = package.MACConfig(
        memory=memory, heads=2, persistent_tokens=3, max_segment=1, memory_mode=mode
    )
    return package.TitansMAC(config, dtype=torch.float64)


def test_inference_writes_only_the_fast_state_and_keeps_persistent_memory_fixed():
    core = mac()
    before = {name: value.detach().clone() for name, value in core.named_parameters()}
    state = core.initial_state(2)
    x = torch.tensor([[[0.2, -0.1, 0.4, 0.3]], [[-0.3, 0.5, 0.1, -0.2]]], dtype=torch.float64)
    output, after = core(x, state)
    assert after.memory.steps.tolist() == [1, 1]
    assert any(
        not torch.equal(left, right)
        for left, right in zip(after.memory.weights, state.memory.weights, strict=True)
    )
    for name, value in core.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        assert value.grad is None
    # P entra en la atención de (7) y (8): otro prefijo cambia la salida de la misma entrada.
    with torch.no_grad():
        core.persistent.mul_(-2.0)
    changed, _ = core(x, state)
    assert not torch.allclose(changed, output)


@pytest.mark.parametrize("detach", [False, True])
def test_trainer_boundary_cuts_the_outer_gradient_through_the_fast_state(detach):
    model, first = setup("mac_online")
    later = raw_batch(at=1_609_459_200_000_000 + DAY)
    second = financial().DecisionBatch.from_corpus(later, specification(), dtype=torch.float64)
    state = model.initial_state(first.flow_ids, differentiable=True)
    prepared = model.prepare(first, state, differentiable=True)
    # Es la operación con la que el entrenador separa el tramo siguiente tras cada paso.
    rows = [row for _, row in _split(prepared.next_state, detach=detach)]
    continued = model.prepare(second, _stack(rows), differentiable=True)
    (gradient,) = torch.autograd.grad(
        continued.point_predictions.square().sum(),
        model.mac.memory.initial_weights[0],
        allow_unused=True,
    )
    if detach:
        # Sin camino por el estado rápido, la decisión siguiente no ve los pesos iniciales.
        assert gradient is None
    else:
        assert gradient is not None and float(gradient.abs().sum()) > 0
