"""Pérdida de Huber por canal de R1 (#786): fórmula, umbral, paridad y gradiente interno."""

import math

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from mars_titan.models.titans import neural_memory
from mars_titan.models.titans.config import GateBias, MACConfig, MemoryConfig
from mars_titan.models.titans.huber_memory import (
    HUBER_MEMORY,
    HuberMemory,
    huber_elements,
    huber_threshold,
)
from mars_titan.models.titans.mac import TitansMAC
from mars_titan.models.titans.neural_memory import NeuralMemory

DIM = 8


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def memory_config(huber=None):
    return MemoryConfig(
        dim=DIM,
        depth=2,
        max_batch=4,
        max_tokens=4,
        gate_bias=GateBias(),
        residual_layer_norm=True,
        huber=huber,
    )


def mac(huber=None):
    config = MACConfig(memory=memory_config(huber), heads=2, persistent_tokens=2, max_segment=1)
    return TitansMAC(config, dtype=torch.float64)


def walk(model, steps=4, rows=3):
    """Recorre `steps` tokens con grafo y devuelve salidas y gradientes de su suma de cuadrados."""
    generator = torch.Generator().manual_seed(3)
    tokens = [
        torch.randn(rows, 1, DIM, generator=generator, dtype=torch.float64) for _ in range(steps)
    ]
    state = model.initial_state(rows, differentiable=True)
    outputs = []
    with sdpa_kernel(SDPBackend.MATH):
        for token in tokens:
            output, state = model(token, state, differentiable=True)
            outputs.append(output)
    total = torch.stack(outputs)
    total.square().sum().backward()
    gradients = {name: value.grad for name, value in model.named_parameters()}
    return total.detach(), state, gradients


def test_each_channel_is_quadratic_inside_the_threshold_and_linear_outside():
    residual = torch.tensor([[0.5, -2.0, 3.0, 0.0, -1.0]], dtype=torch.float64, requires_grad=True)
    threshold = torch.tensor([[1.0, 1.0, 2.0, 0.5, 1.0]], dtype=torch.float64)
    elements = huber_elements(residual, threshold)
    # r² = 0,25 dentro, 2·1·2 − 1 = 3 y 2·2·3 − 4 = 8 fuera, y en |r| = δ ambas ramas valen δ².
    expected = torch.tensor([[0.25, 3.0, 8.0, 0.0, 1.0]], dtype=torch.float64)
    assert torch.equal(elements.detach(), expected)
    (gradient,) = torch.autograd.grad(elements.sum(), residual)
    assert torch.equal(gradient, torch.tensor([[1.0, -2.0, 4.0, 0.0, -2.0]], dtype=torch.float64))


def test_the_gradient_equals_l2_inside_the_threshold_and_clips_it_outside():
    generator = torch.Generator().manual_seed(11)
    residual = torch.randn(64, DIM, generator=generator, dtype=torch.float64) * 2
    residual.requires_grad_(True)
    threshold = torch.rand(64, DIM, generator=generator, dtype=torch.float64) + 0.1
    (huber,) = torch.autograd.grad(huber_elements(residual, threshold).sum(), residual)
    (l2,) = torch.autograd.grad(residual.square().sum(), residual)
    inside = residual.detach().abs() <= threshold
    assert inside.any() and (~inside).any()
    # Dentro del umbral el gradiente es el de L2 bit a bit. Fuera es el residuo recortado.
    assert torch.equal(huber[inside], l2[inside])
    clipped = 2 * residual.detach().clamp(-threshold, threshold)
    assert torch.equal(huber[~inside], clipped[~inside])


def test_the_threshold_is_the_softplus_times_the_value_rms_without_its_gradient():
    config = HuberMemory(0.5)
    assert math.isclose(math.log1p(math.exp(config.bias)), 0.5, rel_tol=1e-15)
    weight = torch.tensor([[0.2, -0.1, 0.0, 0.4]], dtype=torch.float64, requires_grad=True)
    bias = torch.full((4,), config.bias, dtype=torch.float64).requires_grad_(True)
    token = torch.tensor([[1.0, 2.0, -1.0, 0.5]], dtype=torch.float64)
    values = torch.tensor([[3.0, -4.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    threshold = huber_threshold(weight, bias, token, values)
    # w·x = 0,2 − 0,2 + 0 + 0,2 = 0,2 y la magnitud cuadrática media de v es √(25/4) = 2,5.
    kappa = math.log1p(math.exp(0.2 + config.bias))
    torch.testing.assert_close(threshold, torch.full((1, 4), 2.5 * kappa, dtype=torch.float64))
    threshold.sum().backward()
    assert values.grad is None
    assert weight.grad is not None and bias.grad is not None


def test_a_threshold_above_every_residual_reproduces_the_l2_core_bit_for_bit(monkeypatch):
    seen = []
    original = neural_memory.huber_elements

    def recorded(residual, threshold):
        seen.append(bool((residual.detach().abs() <= threshold.detach()).all()))
        return original(residual, threshold)

    monkeypatch.setattr(neural_memory, "huber_elements", recorded)
    reference = mac()
    model = mac(HuberMemory(100.0))
    expected, last, control = walk(reference)
    observed, final, gradients = walk(model)
    assert seen and all(seen)
    assert torch.equal(observed, expected)
    for left, right in zip(last.memory.weights, final.memory.weights, strict=True):
        assert torch.equal(left, right)
    added = {name: gradients.pop(name) for name in ("memory.huber_weight", "memory.huber_bias")}
    assert gradients.keys() == control.keys()
    for name, value in control.items():
        other = gradients[name]
        assert (value is None) == (other is None), name
        assert value is None or torch.equal(value, other), name
    # Dentro del umbral la pérdida no depende de δ, así que su gradiente es exactamente cero.
    assert all(torch.count_nonzero(value) == 0 for value in added.values())


def test_a_threshold_below_some_residuals_changes_the_writes_and_learns():
    expected, _, _ = walk(mac())
    observed, _, gradients = walk(mac(HuberMemory(0.3)))
    assert not torch.equal(observed, expected)
    assert torch.count_nonzero(gradients["memory.huber_weight"]) > 0
    assert torch.count_nonzero(gradients["memory.huber_bias"]) > 0


def test_the_inner_gradient_is_the_jacobian_times_the_clipped_residual(monkeypatch):
    seen = []
    monkeypatch.setattr(NeuralMemory, "observer", staticmethod(lambda **v: seen.append(v)))
    memory = NeuralMemory(memory_config(HuberMemory(0.3)), dtype=torch.float64)
    with torch.no_grad():
        memory.huber_weight.copy_(torch.linspace(-0.3, 0.4, DIM, dtype=torch.float64))
    generator = torch.Generator().manual_seed(7)
    token = torch.randn(3, DIM, generator=generator, dtype=torch.float64)
    state = memory.initial_state(3)
    following = memory.update(token.unsqueeze(1), state)
    # Desde momentum nulo la escritura deja m = −θ g, así que g = −m / θ.
    theta = seen[0]["theta"]
    written = tuple(-value / theta for value in following.momentum)
    keys = F.normalize(memory.key_projection(token), dim=-1, eps=1e-12)
    values = memory.value_projection(token)
    local = tuple(weight.detach().clone().requires_grad_(True) for weight in state.weights)
    prediction = memory._apply_memory(keys.unsqueeze(1), local).squeeze(1)
    residual = (prediction - values).detach()
    threshold = huber_threshold(memory.huber_weight, memory.huber_bias, token, values).detach()
    assert (residual.abs() > threshold).any() and (residual.abs() <= threshold).any()
    expected = torch.autograd.grad(
        prediction, local, grad_outputs=2 * residual.clamp(-threshold, threshold)
    )
    for left, right in zip(written, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-14)


def test_the_surprise_of_each_row_adds_up_to_the_huber_loss(monkeypatch):
    seen = []
    monkeypatch.setattr(NeuralMemory, "observer", staticmethod(lambda **v: seen.append(v)))
    memory = NeuralMemory(memory_config(HuberMemory(0.3)), dtype=torch.float64)
    token = torch.randn(3, 2, DIM, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    surprise = []
    memory.update(token, memory.initial_state(3), surprise=surprise)
    assert len(surprise) == len(seen) == 2
    for rows, observed in zip(surprise, seen, strict=True):
        assert rows.shape == (3,) and not rows.requires_grad
        torch.testing.assert_close(rows.sum(), observed["loss"].detach(), rtol=1e-14, atol=0)


def test_huber_adds_only_its_threshold_and_does_not_consume_the_generator():
    rng = torch.get_rng_state().clone()
    reference = dict(NeuralMemory(memory_config(), dtype=torch.float64).named_parameters())
    declared = HuberMemory(0.7)
    model = dict(NeuralMemory(memory_config(declared), dtype=torch.float64).named_parameters())
    assert torch.equal(torch.get_rng_state(), rng)
    assert set(model) - set(reference) == {"huber_weight", "huber_bias"}
    for name, value in reference.items():
        assert torch.equal(model[name], value), name
    assert model["huber_weight"].shape == (1, DIM) and not torch.count_nonzero(
        model["huber_weight"]
    )
    assert torch.equal(model["huber_bias"], torch.full((DIM,), declared.bias, dtype=torch.float64))


def test_the_identity_records_the_loss_only_when_declared():
    plain, declared = memory_config(), memory_config(HuberMemory(2))
    assert "associative_loss" not in plain.identity()
    assert declared.identity()["associative_loss"] == HuberMemory(2.0).identity()
    assert declared.identity()["associative_loss"]["name"] == HUBER_MEMORY
    assert plain.fingerprint() != declared.fingerprint()
    # κ₀ se guarda como real, así que 2 y 2,0 declaran la misma memoria.
    assert declared.fingerprint() == memory_config(HuberMemory(2.0)).fingerprint()
    model = NeuralMemory(declared, dtype=torch.float64)
    restored = NeuralMemory(declared, dtype=torch.float64)
    restored.load_state_dict(model.state_dict())
    with pytest.raises(ValueError, match="contrato del módulo"):
        NeuralMemory(plain, dtype=torch.float64).load_state_dict(model.state_dict())


def test_the_shared_memory_of_pt4_keeps_l2():
    memory = NeuralMemory(memory_config(HuberMemory()), dtype=torch.float64)
    with pytest.raises(ValueError, match="conserva la pérdida L2"):
        memory.update_shared(torch.zeros(2, DIM, dtype=torch.float64), memory.initial_state(1))


@pytest.mark.parametrize("ratio", [0, 0.009, 100.5, math.nan, math.inf, "1", True, None])
def test_invalid_ratios_are_rejected(ratio):
    with pytest.raises(ValueError, match="initial_ratio"):
        HuberMemory(ratio)


def test_the_memory_needs_a_declared_huber_object():
    with pytest.raises(ValueError, match="huber debe ser HuberMemory"):
        memory_config({"initial_ratio": 1.0})
