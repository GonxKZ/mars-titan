"""Semividas geométricas del olvido de U2 (#786): fórmula, bias por fila, paridad y caja PT1."""

import math

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from mars_titan.models.titans.config import GateBias, MACConfig, MemoryConfig, MemoryStability
from mars_titan.models.titans.mac import TitansMAC
from mars_titan.models.titans.neural_memory import NeuralMemory
from mars_titan.models.titans.scale_invariant_time import (
    SCALE_INVARIANT_TIME,
    ScaleInvariantTime,
)

DIM = 8
REFERENCE = GateBias()
# La caja de PT1 exige η inicial por debajo de su techo de 3/10.
BOXED = GateBias(eta=0.2)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def memory_config(time_scales=None, *, stability=None, gate_bias=REFERENCE, dim=DIM):
    return MemoryConfig(
        dim=dim,
        depth=2,
        max_batch=4,
        max_tokens=4,
        gate_bias=gate_bias,
        residual_layer_norm=dim > 1,
        stability=stability,
        time_scales=time_scales,
    )


def walk(time_scales=None, steps=4, rows=3):
    config = MACConfig(
        memory=memory_config(time_scales), heads=2, persistent_tokens=2, max_segment=1
    )
    model = TitansMAC(config, dtype=torch.float64)
    generator = torch.Generator().manual_seed(3)
    state = model.initial_state(rows, differentiable=True)
    outputs = []
    with sdpa_kernel(SDPBackend.MATH):
        for _ in range(steps):
            token = torch.randn(rows, 1, DIM, generator=generator, dtype=torch.float64)
            output, state = model(token, state, differentiable=True)
            outputs.append(output)
    total = torch.stack(outputs)
    total.square().sum().backward()
    return total.detach(), state, {name: value.grad for name, value in model.named_parameters()}


def test_half_lives_are_geometric_around_the_declared_half_life():
    half_lives = ScaleInvariantTime(6).half_lives(256.0, 64)
    assert half_lives[0] == 4.0 and half_lives[-1] == 16384.0
    ratios = [right / left for left, right in zip(half_lives, half_lives[1:], strict=False)]
    assert all(math.isclose(ratio, 4096 ** (1 / 63), rel_tol=1e-12) for ratio in ratios)
    mean = math.exp(math.fsum(math.log(value) for value in half_lives) / len(half_lives))
    assert math.isclose(mean, 256.0, rel_tol=1e-12)
    assert ScaleInvariantTime(0).half_lives(256.0, 5) == (256.0,) * 5


def test_each_row_starts_with_the_forgetting_of_its_half_life():
    declared = ScaleInvariantTime(3)
    memory = NeuralMemory(memory_config(declared), dtype=torch.float64)
    bias = memory.alpha_projection.bias.detach()
    half_lives = declared.half_lives(256.0, DIM)
    assert bias.shape == (DIM,) and bias.requires_grad is False
    for value, half_life in zip(bias.tolist(), half_lives, strict=True):
        # Con entrada nula α = σ(b) y (1 − α)^h = 1/2 tras h observaciones.
        assert value == GateBias(alpha_half_life=half_life).logits(0.1)[0]
        assert math.isclose((1 - 1 / (1 + math.exp(-value))) ** half_life, 0.5, rel_tol=1e-9)
    assert all(left > right for left, right in zip(bias, bias[1:], strict=False))


def test_zero_octaves_reproduce_the_reference_bit_for_bit():
    expected, last, control = walk()
    observed, final, gradients = walk(ScaleInvariantTime(0))
    assert torch.equal(observed, expected)
    for left, right in zip(last.memory.weights, final.memory.weights, strict=True):
        assert torch.equal(left, right)
    assert gradients.keys() == control.keys()
    for name, value in control.items():
        assert (value is None) == (gradients[name] is None), name
        assert value is None or torch.equal(value, gradients[name]), name


def test_spread_half_lives_change_the_memory_and_their_biases_learn():
    expected, _, _ = walk()
    observed, _, gradients = walk(ScaleInvariantTime(6))
    assert not torch.equal(observed, expected)
    assert torch.count_nonzero(gradients["memory.alpha_projection.bias"]) > 0


def test_only_the_alpha_bias_changes_and_the_generator_is_not_consumed():
    rng = torch.get_rng_state().clone()
    reference = dict(NeuralMemory(memory_config(), dtype=torch.float64).named_parameters())
    spread = dict(
        NeuralMemory(memory_config(ScaleInvariantTime(6)), dtype=torch.float64).named_parameters()
    )
    assert torch.equal(torch.get_rng_state(), rng)
    assert spread.keys() == reference.keys()
    changed = {name for name in reference if not torch.equal(reference[name], spread[name])}
    assert changed == {"alpha_projection.bias"}


def test_the_box_of_pt1_rejects_half_lives_that_need_less_forgetting_than_its_floor():
    box = MemoryStability()
    memory_config(stability=box, gate_bias=BOXED)
    with pytest.raises(ValueError, match="alpha_floor"):
        memory_config(ScaleInvariantTime(6), stability=box, gate_bias=BOXED)
    # Con suelo 1/500 la semivida máxima es de unas 346 sesiones, por encima de 256·2^0,25.
    declared = ScaleInvariantTime(0.25)
    memory = NeuralMemory(
        memory_config(declared, stability=box, gate_bias=BOXED), dtype=torch.float64
    )
    floor = box.alpha_floor
    for value, half_life in zip(
        memory.alpha_projection.bias.tolist(), declared.half_lives(256.0, DIM), strict=True
    ):
        alpha = floor + (1 - floor) / (1 + math.exp(-value))
        assert math.isclose((1 - alpha) ** half_life, 0.5, rel_tol=1e-9)


def test_time_scales_need_gate_bias_and_more_than_one_row():
    with pytest.raises(ValueError, match="time_scales necesita ScaleInvariantTime y gate_bias"):
        memory_config(ScaleInvariantTime(), gate_bias=None)
    with pytest.raises(ValueError, match="time_scales necesita ScaleInvariantTime"):
        memory_config({"octaves": 6.0})
    with pytest.raises(ValueError, match="al menos dos filas"):
        memory_config(ScaleInvariantTime(), dim=1)
    # Las semividas deben quedar en el intervalo [1, 1e6] de GateBias.
    with pytest.raises(ValueError, match="alpha_half_life"):
        memory_config(ScaleInvariantTime(8), gate_bias=GateBias(alpha_half_life=64.0))


@pytest.mark.parametrize("octaves", [-0.5, 8.5, math.nan, math.inf, "6", True, None])
def test_invalid_octaves_are_rejected(octaves):
    with pytest.raises(ValueError, match="octaves"):
        ScaleInvariantTime(octaves)


def test_the_identity_records_the_spread_only_when_declared():
    plain, declared = memory_config(), memory_config(ScaleInvariantTime(6))
    assert "alpha_time_scales" not in plain.identity()
    identity = declared.identity()["alpha_time_scales"]
    assert identity["name"] == SCALE_INVARIANT_TIME and identity["octaves"] == 6.0
    assert plain.fingerprint() != declared.fingerprint()
