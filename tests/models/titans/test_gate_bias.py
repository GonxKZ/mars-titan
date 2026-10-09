"""Bias declarado de las puertas α, η y θ, sin optimizador externo ni datos."""

import functools
import hashlib
import importlib
import math
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.func import functional_call

# Capturadas en develop 91e49eaa antes de introducir gate_bias.
V1_MEMORY_FINGERPRINT = "fb306941d3c59afc43c818f12f4766595293e98c94fdef60dd15b8017be802fb"
V1_MAC_FINGERPRINT = "52e8ee7ca208bbb2f0a5f35af7166369ae5435ea78d303e6947a6589087eedb4"
V1_MEMORY_PARAMETERS = "2f4e8982509dc0d654e4f5226755e0890b8d0ba97db8a2d1d8e6418705a483f2"
V1_MAC_PARAMETERS = "87f9e8c1cc52fe28d81efc6f91af8eab46e0445697d0147c49e98c3070638e69"
GATES = ("alpha_projection", "eta_projection", "theta_projection")


def api():
    return importlib.import_module("mars_titan.models.titans")


def digest(module):
    result = hashlib.sha256()
    for name, value in module.state_dict().items():
        if isinstance(value, torch.Tensor):
            result.update(name.encode())
            result.update(value.detach().contiguous().numpy().tobytes())
    return result.hexdigest()


def configs(gate_bias=None):
    package = api()
    memory = package.MemoryConfig(dim=4, depth=2, max_batch=2, max_tokens=3, gate_bias=gate_bias)
    return memory, package.MACConfig(memory=memory, heads=2, persistent_tokens=2, max_segment=3)


def test_default_configuration_keeps_the_v1_identity_and_parameters_byte_for_byte():
    package = api()
    memory, mac = configs()
    assert memory.fingerprint() == V1_MEMORY_FINGERPRINT
    assert mac.fingerprint() == V1_MAC_FINGERPRINT
    assert "gate_bias" not in memory.identity()
    assert digest(package.NeuralMemory(memory, dtype=torch.float64)) == V1_MEMORY_PARAMETERS
    assert digest(package.TitansMAC(mac, dtype=torch.float64)) == V1_MAC_PARAMETERS
    core = package.NeuralMemory(memory, dtype=torch.float64)
    assert all(getattr(core, name).bias is None for name in GATES)


def test_gate_bias_is_a_new_identity_that_only_adds_constant_biases_to_v1_draws():
    package = api()
    rng = torch.get_rng_state().clone()
    v1_memory, v1_mac = configs()
    bias = package.GateBias(alpha_half_life=256, eta=0.5, theta=0.05)
    v2_memory, v2_mac = configs(bias)
    assert v2_memory.fingerprint() != V1_MEMORY_FINGERPRINT
    assert v2_mac.fingerprint() != V1_MAC_FINGERPRINT
    assert v2_memory.identity()["gate_bias"] == {
        "alpha_half_life": 256.0,
        "eta": 0.5,
        "theta": 0.05,
        "init": "constant_logit_bias_v1_weight_draws",
    }
    v1 = dict(package.TitansMAC(v1_mac, dtype=torch.float64).named_parameters())
    v2 = dict(package.TitansMAC(v2_mac, dtype=torch.float64).named_parameters())
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    added = {name: v2[name] for name in set(v2) - set(v1)}
    assert set(added) == {f"memory.{name}.bias" for name in GATES}
    for name, value in v1.items():
        torch.testing.assert_close(v2[name], value, rtol=0, atol=0)
    logits = bias.logits(v2_memory.theta_max)
    for name, logit in zip(GATES, logits, strict=True):
        value = added[f"memory.{name}.bias"]
        assert value.shape == ((4,) if name == "alpha_projection" else (1,))
        assert value.requires_grad
        torch.testing.assert_close(value, torch.full_like(value, logit), rtol=0, atol=0)


@pytest.mark.parametrize("half_life", [1.0, 2.5, 64.0, 256.0, 1e6])
@pytest.mark.parametrize(("eta", "theta"), [(0.5, 0.05), (0.9, 0.01), (1e-3, 0.0999)])
def test_logits_reproduce_the_declared_rates_and_half_life(half_life, eta, theta):
    bias = api().GateBias(alpha_half_life=half_life, eta=eta, theta=theta)
    alpha_logit, eta_logit, theta_logit = (
        torch.tensor(value, dtype=torch.float64) for value in bias.logits(0.1)
    )
    alpha = alpha_logit.sigmoid().item()
    assert math.isclose(alpha, -math.expm1(-math.log(2) / half_life), rel_tol=1e-12)
    assert math.isclose(math.log(0.5) / math.log1p(-alpha), half_life, rel_tol=1e-9)
    assert math.isclose(eta_logit.sigmoid().item(), eta, rel_tol=1e-12)
    assert math.isclose(0.1 * theta_logit.sigmoid().item(), theta, rel_tol=1e-12)


def test_zero_input_starts_at_declared_rates_and_longer_half_life_retains_more():
    package = api()
    zero = torch.zeros((3, 4), dtype=torch.float64)
    x = torch.randn((3, 4), generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    alphas = []
    for half_life in (16.0, 64.0, 256.0):
        bias = package.GateBias(alpha_half_life=half_life, eta=0.7, theta=0.02)
        memory = package.NeuralMemory(configs(bias)[0], dtype=torch.float64)
        expected = -math.expm1(-math.log(2) / half_life)
        alpha_zero = memory.alpha_projection(zero).sigmoid()
        torch.testing.assert_close(alpha_zero, torch.full_like(alpha_zero, expected))
        torch.testing.assert_close(
            memory.eta_projection(zero).sigmoid(), torch.full((3, 1), 0.7, dtype=torch.float64)
        )
        theta = memory.config.theta_max * memory.theta_projection(zero).sigmoid()
        torch.testing.assert_close(theta, torch.full((3, 1), 0.02, dtype=torch.float64))
        alphas.append(memory.alpha_projection(x).sigmoid())
    assert (alphas[0] > alphas[1]).all() and (alphas[1] > alphas[2]).all()


def test_rates_stay_inside_their_ranges_and_still_depend_on_the_input():
    package = api()
    memory = package.NeuralMemory(configs(package.GateBias())[0], dtype=torch.float64)
    generator = torch.Generator().manual_seed(9)
    inputs = 5 * torch.randn((64, 4), generator=generator, dtype=torch.float64)
    alpha = memory.alpha_projection(inputs).sigmoid()
    eta = memory.eta_projection(inputs).sigmoid()
    theta = memory.config.theta_max * memory.theta_projection(inputs).sigmoid()
    assert ((alpha > 0) & (alpha < 1)).all() and ((eta > 0) & (eta < 1)).all()
    assert ((theta > 0) & (theta < memory.config.theta_max)).all()
    for rate in (alpha, eta, theta):
        assert rate.std(dim=0).min() > 0
    first, second = inputs[:1], inputs[1:2]
    assert not torch.equal(
        memory.alpha_projection(first).sigmoid(), memory.alpha_projection(second).sigmoid()
    )


def test_linear_update_with_gate_bias_matches_the_manual_equation():
    package = api()
    config = package.MemoryConfig(
        dim=2, depth=1, normalize_qk=False, theta_max=0.5, gate_bias=package.GateBias(32, 0.6, 0.2)
    )
    memory = package.NeuralMemory(config, dtype=torch.float64)
    state = memory.initial_state(2)
    weights = torch.tensor([[[0.2, -0.1], [0.4, 0.3]]] * 2, dtype=torch.float64)
    momentum = torch.tensor([[[0.01, -0.02], [0.05, 0.06]]] * 2, dtype=torch.float64)
    state = replace(state, weights=(weights,), momentum=(momentum,))
    x = torch.tensor([[0.3, -0.2], [0.1, 0.4]], dtype=torch.float64)
    keys, values = memory.key_projection(x), memory.value_projection(x)
    alpha = torch.sigmoid(x @ memory.alpha_projection.weight.T + memory.alpha_projection.bias)
    eta = torch.sigmoid(x @ memory.eta_projection.weight.T + memory.eta_projection.bias)
    theta = 0.5 * torch.sigmoid(x @ memory.theta_projection.weight.T + memory.theta_projection.bias)
    residual = torch.bmm(weights, keys.unsqueeze(-1)).squeeze(-1) - values
    gradient = 2 * residual.unsqueeze(-1) * keys.unsqueeze(-2)
    expected_momentum = eta.unsqueeze(-1) * momentum - theta.unsqueeze(-1) * gradient
    expected = (1 - alpha.unsqueeze(-1)) * weights + expected_momentum
    actual = memory.update(x.unsqueeze(1), state)
    torch.testing.assert_close(actual.momentum[0], expected_momentum, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(actual.weights[0], expected, rtol=1e-13, atol=1e-13)


class _Update(nn.Module):
    def __init__(self, memory):
        super().__init__()
        self.memory = memory

    def forward(self, values, state):
        result = self.memory.update(values, state, differentiable=True)
        return (*result.weights, *result.momentum)


def test_two_layer_update_gradcheck_covers_gate_biases_inputs_and_state():
    package = api()
    config = package.MemoryConfig(dim=2, depth=2, normalize_qk=False, gate_bias=package.GateBias())
    wrapper = _Update(package.NeuralMemory(config, dtype=torch.float64))
    state = wrapper.memory.initial_state(1, differentiable=True)
    generator = torch.Generator().manual_seed(11)
    values = torch.randn((1, 2, 2), generator=generator, dtype=torch.float64).requires_grad_()
    biases = tuple(
        getattr(wrapper.memory, name).bias.detach().clone().requires_grad_() for name in GATES
    )
    leaves = tuple(w.detach().clone().requires_grad_() for w in state.weights)
    momentum = tuple(
        (0.1 * torch.randn(m.shape, generator=generator, dtype=torch.float64)).requires_grad_()
        for m in state.momentum
    )

    def function(alpha, eta, theta, x, w1, w2, m1, m2):
        replaced = dict(
            zip((f"memory.{name}.bias" for name in GATES), (alpha, eta, theta), strict=True)
        )
        initial = replace(state, weights=(w1, w2), momentum=(m1, m2))
        return functional_call(wrapper, replaced, (x, initial))

    assert torch.autograd.gradcheck(
        function, (*biases, values, *leaves, *momentum), eps=1e-6, atol=2e-5, rtol=2e-4
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(alpha_half_life=0.5),
        dict(alpha_half_life=float("inf")),
        dict(alpha_half_life=float("nan")),
        dict(alpha_half_life=True),
        dict(alpha_half_life=2e6),
        dict(eta=0),
        dict(eta=1),
        dict(theta=0),
        dict(theta=1),
        dict(theta="0.05"),
    ],
)
def test_invalid_gate_bias_values_are_rejected(options):
    with pytest.raises(ValueError):
        api().GateBias(**options)


def test_memory_configuration_rejects_theta_outside_theta_max_and_foreign_types():
    package = api()
    with pytest.raises(ValueError, match="theta_max"):
        package.MemoryConfig(dim=2, theta_max=0.05, gate_bias=package.GateBias(theta=0.05))
    with pytest.raises(ValueError, match="GateBias"):
        package.MemoryConfig(dim=2, gate_bias={"alpha_half_life": 256})


def test_v1_and_gate_bias_contracts_reject_each_other_parameters_and_states():
    package = api()
    v1 = package.TitansMAC(configs()[1], dtype=torch.float64)
    v2 = package.TitansMAC(configs(package.GateBias())[1], dtype=torch.float64)
    for source, target in ((v1, v2), (v2, v1)):
        with pytest.raises(ValueError, match="contrato"):
            target.load_state_dict(source.state_dict())
        with pytest.raises(ValueError, match="configur|contrato"):
            target.restore_state(source.export_state(source.initial_state(1)))
    copy = package.TitansMAC(configs(package.GateBias())[1], dtype=torch.float64)
    copy.load_state_dict(v2.state_dict())
    segment = torch.full((1, 2, 4), 0.1, dtype=torch.float64)
    state = v2.restore_state(v2.export_state(v2.initial_state(1)))
    torch.testing.assert_close(copy(segment, state)[0], v2(segment, state)[0], rtol=0, atol=0)


# Trayectorias largas: D=16, dos flujos, un token por observación y norma cercana a 1, como
# el token fusionado del adaptador financiero inicial.
DIM, FLOWS, WINDOW, LENGTHS = 16, 2, 8, (64, 256, 1024)
HALF_LIFE = 256.0


@functools.cache
def trajectory(variant):
    package = api()
    gate_bias = package.GateBias(alpha_half_life=HALF_LIFE) if variant == "gate_bias" else None
    memory = package.MemoryConfig(
        dim=DIM, depth=2, max_batch=FLOWS, max_tokens=1, gate_bias=gate_bias
    )
    mac = package.TitansMAC(
        package.MACConfig(memory=memory, heads=4, persistent_tokens=4, max_segment=1),
        dtype=torch.float64,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        head = nn.Linear(DIM, 1, dtype=torch.float64)
    generator = torch.Generator().manual_seed(2026)
    tokens = [
        torch.randn((FLOWS, 1, DIM), generator=generator, dtype=torch.float64) * DIM**-0.5
        for _ in range(max(LENGTHS))
    ]
    state = mac.initial_state(FLOWS)
    initial = [w.norm(dim=(1, 2)).mean().item() for w in state.memory.weights]
    norms, gradients = {}, {}
    names = [
        "head.weight",
        *(f"mac.memory.{name}.weight" for name in (*GATES, "key_projection", "value_projection")),
        "mac.query_projection.weight",
    ]
    parameters = dict(mac.named_parameters(prefix="mac"), **{"head.weight": head.weight})
    for step in range(1, max(LENGTHS) + 1):
        end = step + WINDOW - 1
        if end in LENGTHS:
            # Funcional lineal de las salidas: misma escala que la derivada del MAE.
            local, total = state, 0
            for token in tokens[step - 1 : end]:
                output, local = mac(token, local, differentiable=True)
                total = total + head(output).sum()
            values = torch.autograd.grad(total, [parameters[name] for name in names])
            gradients[end] = {name: v.norm().item() for name, v in zip(names, values, strict=True)}
        with torch.no_grad():
            _, state = mac(tokens[step - 1], state)
        if step in LENGTHS:
            norms[step] = [w.norm(dim=(1, 2)).mean().item() for w in state.memory.weights]
    return initial, norms, gradients


@pytest.mark.parametrize("length", LENGTHS)
def test_gate_bias_keeps_fast_memory_norms_inside_the_declared_retention(length):
    initial, norms, _ = trajectory("gate_bias")
    for layer, start in enumerate(initial):
        decayed = start * 0.5 ** (length / HALF_LIFE)
        assert 0.5 * decayed <= norms[length][layer] <= start


@pytest.mark.parametrize("length", LENGTHS)
def test_gate_bias_keeps_outer_gradients_finite_and_far_from_v1_collapse(length):
    _, _, gradients = trajectory("gate_bias")
    _, v1_norms, v1_gradients = trajectory("v1")
    for name, value in gradients[length].items():
        # Cota técnica de este fixture: como mucho cuatro órdenes de caída en 960
        # observaciones. v1 pierde más de treinta en las primeras 64.
        assert math.isfinite(value) and 1e-12 < value
        assert value >= 1e-4 * gradients[64][name]
        assert v1_gradients[length][name] < 1e-20 * value
    assert max(v1_norms[length]) < 1e-15
