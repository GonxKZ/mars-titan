"""Captura por coherencia de la misma fila, con controles históricos y límites explícitos."""

from dataclasses import asdict, replace

import pytest
import torch

from mars_titan.models.titans.config import MemoryConfig, MemoryMultiscale
from mars_titan.models.titans.neural_memory import NeuralMemory
from mars_titan.models.titans.plasticity import advance, initial_history
from mars_titan.models.titans.plasticity_config import MemoryPlasticity

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]


def local(**changes):
    return MemoryPlasticity(
        **dict(
            capture="eligibility", capture_scope="row_local", gradient_contract="scalar_controls_v2"
        )
        | changes
    )


@pytest.mark.parametrize("device", DEVICES)
def test_local_capture_does_not_dilute_the_fixed_row_when_other_rows_are_inactive(device):
    config = local(beta=0, trace_decay=0, epsilon=1)
    gradient = torch.ones(2, 8, 8, dtype=torch.float64, device=device)
    gradient[1, 1:] = 0
    result = advance(
        config,
        gradient,
        initial_history(gradient),
        torch.zeros(2, dtype=torch.int64, device=device),
    )
    torch.testing.assert_close(
        result.capture_gain[:, 0, 0],
        torch.tensor([73 / 90, 73 / 90], dtype=torch.float64, device=device),
        rtol=1e-14,
        atol=1e-14,
    )
    assert torch.equal(result.capture_gain[0, 0], result.capture_gain[1, 0])


def test_default_scope_preserves_both_historical_configuration_fingerprints():
    expected = dict(
        row_broadcast_v1="92aa3cd78f07132d551ed47a4e50ceee9a6772a675c846651757de100f4bde83",
        scalar_controls_v2="a69697f5a889f436440c42690d75e6b2849ad3293b0c0291d5d3ab106dbba4d8",
    )
    for contract, fingerprint in expected.items():
        options = MemoryPlasticity(capture="eligibility", gradient_contract=contract)
        explicit = replace(options, capture_scope="population_mean")
        assert "capture_scope" not in options.configuration()
        assert explicit.configuration() == options.configuration()
        config = MemoryConfig(3, plasticity=options, multiscale=MemoryMultiscale())
        assert config.fingerprint() == fingerprint
    assert local().identity()["schema_version"] == 3
    assert local().identity()["capture_signal"] == "local_eligibility_times_row_coherence"
    assert local().configuration()["capture_scope"] == "row_local"


@pytest.mark.parametrize(
    "changes",
    [
        dict(capture_scope="unknown"),
        dict(capture_scope=None),
        dict(capture_scope=True),
        dict(capture="none"),
        dict(capture="constant"),
        dict(gradient_contract="row_broadcast_v1"),
    ],
)
def test_unused_or_unidentified_local_scope_is_rejected(changes):
    with pytest.raises(ValueError):
        local(**changes)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_row_history_is_independent_of_other_rows_through_a_trajectory(device, dtype):
    config = local(epsilon=0.5)
    generator = torch.Generator(device=device).manual_seed(841)
    gradient = torch.randn(2, 4, 4, generator=generator, dtype=dtype, device=device)
    left = right = initial_history(gradient)
    for step in range(12):
        changed = gradient.clone()
        changed[:, 1:] = 4 * torch.randn(2, 3, 4, generator=generator, dtype=dtype, device=device)
        count = torch.full((2,), step, dtype=torch.int64, device=device)
        first = advance(config, gradient, left, count)
        second = advance(config, changed, right, count)
        assert torch.equal(first.capture_gain[:, 0], second.capture_gain[:, 0])
        for a, b in zip(first.history, second.history, strict=True):
            assert torch.equal(a[:, 0], b[:, 0])
        assert ((first.capture_gain >= 0.1) & (first.capture_gain <= 1)).all()
        left, right = first.history, second.history


@pytest.mark.parametrize("device", DEVICES)
def test_local_capture_jacobian_has_no_cross_row_dependency_and_passes_gradcheck(device):
    gradient = torch.tensor(
        [[[0.2, -0.3], [0.7, 0.4]], [[-0.5, 0.6], [0.3, -0.8]]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    config = local(beta=0.7, trace_decay=0.6, epsilon=0.5)
    history, steps = initial_history(gradient), torch.zeros(2, dtype=torch.int64, device=device)
    gain = advance(config, gradient, history, steps).capture_gain
    jacobian = torch.autograd.grad(gain[0, 0, 0], gradient)[0]
    assert torch.equal(jacobian[0, 1], torch.zeros_like(jacobian[0, 1]))
    assert torch.equal(jacobian[1], torch.zeros_like(jacobian[1]))
    assert jacobian[0, 0].abs().sum() > 0
    global_gain = advance(
        replace(config, capture_scope="population_mean"), gradient, history, steps
    ).capture_gain
    global_jacobian = torch.autograd.grad(global_gain[0, 0, 0], gradient)[0]
    assert global_jacobian[:, 1].abs().sum() > 0
    assert torch.equal(global_jacobian[1], torch.zeros_like(global_jacobian[1]))
    assert torch.autograd.gradcheck(
        lambda g: advance(config, g, history, steps).capture_gain,
        (gradient,),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
    )


def test_financial_recipes_keep_legacy_fields_and_admit_the_complete_local_identity():
    from test_financial_adapter import setup

    base, _ = setup()
    for gradient_contract in ("row_broadcast_v1", "scalar_controls_v2"):
        options = MemoryPlasticity(capture="eligibility", gradient_contract=gradient_contract)
        config = replace(
            base.config,
            memory_multiscale=MemoryMultiscale(),
            memory_plasticity=options.configuration(),
        )
        assert config.memory_plasticity == options
    config = replace(
        base.config, memory_multiscale=MemoryMultiscale(), memory_plasticity=asdict(local())
    )
    assert config.memory_plasticity == local()
    with pytest.raises(ValueError):
        replace(config, memory_plasticity=asdict(local()) | dict(unknown=True))


@pytest.mark.parametrize("device", DEVICES)
def test_local_neural_memory_has_no_new_parameters_or_state_and_rejects_other_scope(device):
    options = dict(
        dim=4, depth=2, max_batch=2, max_tokens=2, multiscale=MemoryMultiscale(), parameter_seed=42
    )
    global_model = NeuralMemory(
        MemoryConfig(
            **options,
            plasticity=MemoryPlasticity(
                capture="eligibility", gradient_contract="scalar_controls_v2"
            ),
        ),
        device=device,
    )
    model = NeuralMemory(MemoryConfig(**options, plasticity=local()), device=device)
    for a, b in zip(global_model.parameters(), model.parameters(), strict=True):
        assert torch.equal(a, b)
    state = model.initial_state(2)
    global_state = global_model.initial_state(2)
    for name in ("weights", "momentum", "levels", "modulation"):
        assert sum(x.numel() for x in getattr(state, name)) == sum(
            x.numel() for x in getattr(global_state, name)
        )
    tokens = torch.arange(16, device=device, dtype=torch.float32).reshape(2, 2, 4) / 30
    updated = model.update(tokens, state)
    with pytest.raises(ValueError):
        global_model.update(tokens, updated)
    # Partir la misma secuencia conserva las escrituras y sus estadísticas.
    chunked = model.update(tokens[:, 1:], model.update(tokens[:, :1], state))
    for name in ("weights", "momentum", "levels", "modulation"):
        for a, b in zip(getattr(updated, name), getattr(chunked, name), strict=True):
            assert torch.equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_financial_local_state_restores_the_next_prediction_and_keeps_global_rng(device):
    from test_financial_adapter import api, raw_batch, setup

    base, _ = setup()
    rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device != "cpu" else None
    model = api().FinancialPredictor(
        replace(base.config, memory_multiscale=MemoryMultiscale(), memory_plasticity=local()),
        dtype=torch.float64,
        device=device,
    )
    batch = api().DecisionBatch.from_corpus(
        raw_batch(), model.config.inputs, dtype=torch.float64, device=device
    )
    assert torch.equal(rng, torch.random.get_rng_state())
    state = model.prepare(batch, model.initial_state(batch.flow_ids)).next_state
    payload = model.export_state_cpu(state)
    restored = model.restore_state(payload, device=device)
    next_batch = api().DecisionBatch.from_corpus(
        raw_batch(at=batch.prediction_at[0] + 10),
        model.config.inputs,
        dtype=torch.float64,
        device=device,
    )
    expected, actual = model.prepare(next_batch, state), model.prepare(next_batch, restored)
    assert torch.equal(expected.point_predictions, actual.point_predictions)
    assert torch.equal(rng, torch.random.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    global_model = api().FinancialPredictor(
        replace(model.config, memory_plasticity=replace(local(), capture_scope="population_mean")),
        dtype=torch.float64,
        device=device,
    )
    assert model._tensor_bytes_per_flow() == global_model._tensor_bytes_per_flow()
    with pytest.raises(ValueError):
        global_model.restore_state(payload, device=device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "overflow"])
def test_local_capture_does_not_hide_nonfinite_gradients_or_overflow(device, dtype, bad):
    value = torch.finfo(dtype).max / 2 if bad == "overflow" else bad
    gradient = torch.full((1, 2, 2), value, device=device, dtype=dtype)
    with pytest.raises(ValueError):
        advance(
            local(),
            gradient,
            initial_history(gradient),
            torch.zeros(1, dtype=torch.int64, device=device),
        )


@pytest.mark.parametrize("device", DEVICES)
def test_local_capture_changes_both_cascade_directions_with_the_actual_row_gain(device):
    model = NeuralMemory(
        MemoryConfig(
            8,
            depth=1,
            multiscale=MemoryMultiscale(),
            plasticity=local(write="none", beta=0, trace_decay=0, epsilon=1),
        ),
        device=device,
        dtype=torch.float64,
    )
    state = model.initial_state(1)
    state.weights[0].fill_(1)
    for level in state.levels:
        level.zero_()
    gradient = torch.zeros_like(state.weights[0])
    gradient[:, 0] = 1
    gate = torch.zeros(1, 1, 1, device=device, dtype=torch.float64)
    weights, momentum, levels, _ = model._write(
        state.weights,
        state.momentum,
        state.levels,
        (gradient,),
        (gate, gate, gate),
        True,
        modulation=state.modulation,
        steps=state.steps,
    )
    # La primera fila usa 73/90 y las inactivas conservan el suelo 1/10.
    exchange = gradient.new_full((1, 8, 1), 0.1 / 16)
    exchange[:, 0] = (73 / 90) / 16
    torch.testing.assert_close(weights[0], 1 - exchange.expand_as(gradient), rtol=0, atol=1e-15)
    torch.testing.assert_close(levels[0], exchange.expand_as(gradient) / 2, rtol=0, atol=1e-15)
    assert torch.count_nonzero(momentum[0]) == torch.count_nonzero(levels[1]) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_unit_capture_floor_preserves_weights_and_parameter_gradients(device):
    options = dict(dim=4, depth=2, multiscale=MemoryMultiscale(), parameter_seed=42)
    reference = NeuralMemory(MemoryConfig(**options), device=device, dtype=torch.float64)
    model = NeuralMemory(
        MemoryConfig(**options, plasticity=local(write="none", capture_floor=1)),
        device=device,
        dtype=torch.float64,
    )
    tokens = torch.arange(24, device=device, dtype=torch.float64).reshape(2, 3, 4) / 20
    outputs, gradients = [], []
    for current in (reference, model):
        state = current.update(tokens, current.initial_state(2), differentiable=True)
        values = (*state.weights, *state.momentum, *state.levels)
        outputs.append(values)
        gradients.append(
            torch.autograd.grad(
                sum(value.square().sum() for value in values),
                tuple(current.parameters()),
                allow_unused=True,
            )
        )
    for expected, actual in zip(outputs[0], outputs[1], strict=True):
        assert torch.equal(expected, actual)
    for expected, actual in zip(gradients[0], gradients[1], strict=True):
        if expected is None:
            assert actual is None
        else:
            assert torch.equal(expected, actual)


@pytest.mark.parametrize("device", DEVICES)
def test_local_shared_write_keeps_cohort_order_and_has_one_history_update(device):
    model = NeuralMemory(
        MemoryConfig(4, depth=2, multiscale=MemoryMultiscale(), plasticity=local()),
        device=device,
        dtype=torch.float64,
    )
    tokens = torch.arange(16, device=device, dtype=torch.float64).reshape(4, 4) / 20
    state = model.initial_state(1)
    first = model.update_shared(tokens, state, differentiable=True)
    second = model.update_shared(tokens.flip(0), state, differentiable=True)
    for name in ("weights", "momentum", "levels", "modulation"):
        for expected, actual in zip(getattr(first, name), getattr(second, name), strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    assert first.steps.tolist() == second.steps.tolist() == [1]
