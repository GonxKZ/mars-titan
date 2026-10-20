"""Paridad de gradientes y ajustes de los controles escalares de plasticidad."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest
import torch
from test_financial_adapter import api, raw_batch, setup

from mars_titan.models.titans.config import PAPER_PROJECTIONS, MemoryConfig, MemoryMultiscale
from mars_titan.models.titans.neural_memory import NeuralMemory
from mars_titan.models.titans.plasticity_config import MemoryPlasticity

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]


def scalar_control(**changes):
    return MemoryPlasticity(gradient_contract="scalar_controls_v2", **changes)


def test_legacy_configuration_keeps_its_checkpoint_identities():
    plasticity = MemoryPlasticity(capture="eligibility")
    memory = MemoryConfig(3, plasticity=plasticity, multiscale=MemoryMultiscale())
    assert (
        memory.fingerprint() == "92aa3cd78f07132d551ed47a4e50ceee9a6772a675c846651757de100f4bde83"
    )
    base, _ = setup()
    config = replace(
        base.config, memory_plasticity=plasticity, memory_multiscale=MemoryMultiscale()
    )
    identity = json.dumps(
        config.identity(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert hashlib.sha256(identity.encode()).hexdigest() == (
        "8a8a7882dc820676b86ff8fc1b1c6fd918fccfdd1c1cd64429bfc1dbf3340f66"
    )
    historical = asdict(plasticity)
    historical.pop("gradient_contract", None)
    recovered = replace(config, memory_plasticity=historical)
    assert recovered.identity() == config.identity()
    changed = replace(config, memory_plasticity=scalar_control(capture="eligibility"))
    assert changed.identity() != config.identity()
    assert replace(changed, memory_plasticity=asdict(changed.memory_plasticity)) == changed
    with pytest.raises(ValueError):
        replace(config, memory_plasticity=historical | {"unknown": 1})
    with pytest.raises(ValueError):
        replace(config, memory_plasticity=historical | {"gradient_contract": "unknown"})


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "write,constant", [("none", 0.3), ("constant", 0.0), ("constant", 0.3), ("constant", 1.0)]
)
def test_scalar_write_preserves_the_gate_reduction_and_gradient(device, dtype, write, constant):
    generator = torch.Generator(device=device).manual_seed(731)
    options = scalar_control(write=write, write_constant=constant)
    model = NeuralMemory(MemoryConfig(64, depth=1, plasticity=options), device=device, dtype=dtype)
    state = model.initial_state(8)
    gradient = torch.randn(
        8, 64, 64, generator=generator, device=device, dtype=dtype, requires_grad=True
    )
    theta = torch.rand(8, 1, 1, generator=generator, device=device, dtype=dtype, requires_grad=True)
    eta = torch.full_like(theta, 0.5, requires_grad=True)
    alpha = torch.full((8, 64, 1), 0.1, device=device, dtype=dtype)
    upstream = torch.randn(8, 64, 64, generator=generator, device=device, dtype=dtype)
    result = model._write(
        state.weights,
        state.momentum,
        (),
        (gradient,),
        (alpha, eta, theta),
        True,
        modulation=state.modulation,
        steps=state.steps,
    )[1][0]
    coefficient = theta if write == "none" or constant == 1 else theta * constant
    reference = eta * state.momentum[0] - coefficient * gradient
    actual_grads = torch.autograd.grad(
        (result * upstream).sum(), (theta, eta, gradient), retain_graph=True
    )
    expected_grads = torch.autograd.grad((reference * upstream).sum(), (theta, eta, gradient))
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("capture", [0.0, 0.3, 1.0])
def test_scalar_capture_scales_the_first_interface_in_both_directions(capture):
    config = MemoryConfig(
        2,
        depth=1,
        multiscale=MemoryMultiscale(),
        plasticity=scalar_control(write="none", capture="constant", capture_constant=capture),
    )
    memory = NeuralMemory(config, dtype=torch.float64)
    state = memory.initial_state(1)
    state.weights[0].fill_(1)
    for level in state.levels:
        level.zero_()
    zero = torch.zeros(1, 1, 1, dtype=torch.float64)
    weights, _, levels, _ = memory._write(
        state.weights,
        state.momentum,
        state.levels,
        (torch.zeros_like(state.weights[0]),),
        (zero, zero, zero),
        True,
        modulation=state.modulation,
        steps=state.steps,
    )
    torch.testing.assert_close(weights[0], torch.full_like(weights[0], 1 - capture / 16))
    torch.testing.assert_close(levels[0], torch.full_like(levels[0], capture / 32))
    assert torch.count_nonzero(levels[1]) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("write,capture", [("none", "none"), ("constant", "constant")])
def test_unity_control_keeps_all_shared_parameter_gradients(device, dtype, write, capture):
    config = MemoryConfig(
        8, depth=2, multiscale=MemoryMultiscale(), qkv_silu=True, qkv_convolution=4
    )
    control = replace(
        config,
        plasticity=scalar_control(
            write=write, capture=capture, write_constant=1, capture_constant=1
        ),
    )
    models = [NeuralMemory(c, device=device, dtype=dtype) for c in (config, control)]
    generator = torch.Generator(device=device).manual_seed(79)
    tokens = torch.randn(4, 5, 8, generator=generator, device=device, dtype=dtype)
    states, gradients = [], []
    for model in models:
        state = model.update(tokens, model.initial_state(4), differentiable=True)
        states.append(state)
        loss = sum(x.square().sum() for x in (*state.weights, *state.momentum, *state.levels))
        gradients.append(torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True))
    for a, b in zip(
        (*states[0].weights, *states[0].momentum, *states[0].levels),
        (*states[1].weights, *states[1].momentum, *states[1].levels),
        strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(*gradients, strict=True):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    restored = models[1].restore_state(models[1].export_state(states[1]))
    next_tokens = tokens[:, :1].contiguous()
    expected = models[1].update(next_tokens, states[1])
    actual = models[1].update(next_tokens, restored)
    for a, b in zip(
        (*expected.weights, *expected.modulation),
        (*actual.weights, *actual.modulation),
        strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    legacy = NeuralMemory(
        replace(
            control, plasticity=replace(control.plasticity, gradient_contract="row_broadcast_v1")
        ),
        device=device,
        dtype=dtype,
    )
    with pytest.raises(ValueError):
        legacy.restore_state(models[1].export_state(states[1]))


@pytest.mark.cuda
@pytest.mark.unguarded_fit
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unity_financial_control_matches_four_optimizer_updates(dtype):
    base, _ = setup()
    config = replace(
        base.config,
        memory_projections=PAPER_PROJECTIONS,
        attention_layout="sequence_contiguous_v1",
        memory_multiscale=MemoryMultiscale(),
    )
    control = replace(
        config,
        memory_plasticity=scalar_control(
            write="constant",
            capture="constant",
            write_constant=1,
            capture_constant=1,
        ),
    )
    models = [api().FinancialPredictor(c, device="cuda:0", dtype=dtype) for c in (config, control)]
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-3) for model in models]
    for step in range(4):
        outputs = []
        for model, optimizer in zip(models, optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            state = None
            for decision in range(3):
                batch = api().DecisionBatch.from_corpus(
                    raw_batch(at=1_609_459_200_000_000 + (step * 3 + decision) * 10),
                    config.inputs,
                    device="cuda:0",
                    dtype=dtype,
                )
                if state is None:
                    state = model.initial_state(batch.flow_ids, differentiable=True)
                output = model.prepare(batch, state, differentiable=True)
                state = output.next_state
            output.point_predictions.square().sum().backward()
            outputs.append(output.point_predictions)
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        for (name, left), (other, right) in zip(
            models[0].named_parameters(), models[1].named_parameters(), strict=True
        ):
            assert name == other
            if left.grad is None:
                assert right.grad is None
            else:
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
        for model, optimizer in zip(models, optimizers, strict=True):
            optimizer.step()
            # FinancialRun confirma la nueva identidad al terminar cada ajuste.
            model._seal_parameters()
        for left, right in zip(models[0].parameters(), models[1].parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
