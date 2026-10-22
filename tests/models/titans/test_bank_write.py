"""Parciales de la mezcla y momentum por banco, sin optimizador externo."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from mars_titan.models.titans.bank_readout import BankReadoutConfig
from mars_titan.models.titans.config import MemoryConfig, MemoryMultiscale
from mars_titan.models.titans.neural_memory import NeuralMemory
from mars_titan.models.titans.state import map_memory_rows

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def memory(rule="per_bank", *, device="cpu", dtype=torch.float64, **options):
    routing = options.pop("routing", "query")
    form = options.pop("form", "ema_bank")
    readout = BankReadoutConfig(routing=routing, write_gradient=rule)
    return NeuralMemory(
        MemoryConfig(3, multiscale=MemoryMultiscale(form=form), bank_readout=readout, **options),
        device=device,
        dtype=dtype,
    )


def banks(state, depth):
    return (state.weights,) + tuple(
        tuple(state.levels[2 * layer + region] for layer in range(depth)) for region in range(2)
    )


def distinct_state(model, batch=2):
    state = model.initial_state(batch)
    generator = torch.Generator(device="cpu").manual_seed(139)

    def noise_like(value):
        return torch.randn(value.shape, dtype=value.dtype, generator=generator).to(value.device)

    return replace(
        state,
        weights=tuple(noise_like(w) / 4 for w in state.weights),
        levels=tuple(noise_like(w) / 4 for w in state.levels),
        momentum=tuple(noise_like(w) / 100 for w in state.momentum),
    )


def reference(model, token, state, *, shared=False):
    """Derivar desde las tres MLP explícitas y las proyecciones congeladas del módulo."""
    keys = model.key_projection(token)
    values = model.value_projection(token)
    keys = F.normalize(keys, dim=-1, eps=1e-12)
    local = tuple(w.detach().clone().requires_grad_() for bank in banks(state, 2) for w in bank)
    direction = F.normalize(keys, dim=-1)
    if model.config.bank_readout.routing == "constant":
        direction = torch.ones_like(direction) / 3**0.5
    logits = direction @ F.normalize(model.bank_readout.addresses, dim=-1).T
    if model.config.bank_readout.routing == "uniform":
        logits = torch.zeros_like(logits)
    probabilities = logits.softmax(-1)
    output = sum(
        probabilities[..., r : r + 1]
        * (F.gelu(keys @ local[2 * r].transpose(-1, -2)) @ local[2 * r + 1].transpose(-1, -2))
        for r in range(3)
    )
    loss = (output - values).square().sum() / (token.shape[1] if shared else 1)
    gradients = torch.autograd.grad(loss, local)
    if model.config.bank_readout.write_gradient == "mean_partial":
        average = tuple(sum(gradients[2 * r + i] for r in range(3)) / 3 for i in range(2))
        gradients = average * 3
    gate_token = token.mean(1) if shared else token.squeeze(1)
    alpha = model.alpha_projection(gate_token).sigmoid().unsqueeze(-1)
    eta = model.eta_projection(gate_token).sigmoid().unsqueeze(-1)
    theta = model.config.theta_max * model.theta_projection(gate_token).sigmoid().unsqueeze(-1)
    decays = (alpha, *model.config.multiscale.bank_decays)
    following_momentum = tuple(
        eta * old - theta * gradient
        for old, gradient in zip(state.momentum, gradients, strict=True)
    )
    following_weights = tuple(
        (1 - decays[r]) * local[2 * r + i] + following_momentum[2 * r + i]
        for r in range(3)
        for i in range(2)
    )
    return following_weights, following_momentum


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("routing", ["uniform", "constant", "query"])
@pytest.mark.parametrize("rule", ["mean_partial", "per_bank"])
def test_update_uses_partials_of_the_actual_mixture_with_the_declared_projection(
    device, dtype, routing, rule
):
    model = memory(rule, device=device, dtype=dtype, routing=routing)
    state = distinct_state(model)
    token = torch.tensor([[[0.2, -0.5, 0.7]], [[0.3, 0.6, -0.2]]], dtype=dtype, device=device)
    expected_weights, expected_momentum = reference(model, token, state)
    result = model.update(token, state)
    actual = tuple(w for bank in banks(result, 2) for w in bank)
    tolerance = 1e-11 if dtype == torch.float64 else 1e-6
    for a, b in zip(
        (*actual, *result.momentum), (*expected_weights, *expected_momentum), strict=True
    ):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance * 10)
    assert len(result.momentum) == 6 and result.steps.tolist() == [1, 1]
    assert all(not t.requires_grad for t in (*actual, *result.momentum))
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rule", ["mean_partial", "per_bank"])
def test_shared_write_uses_the_mean_cohort_loss_for_each_bank(device, rule):
    model = memory(rule, device=device)
    state = distinct_state(model, 1)
    token = torch.tensor([[[0.2, -0.5, 0.7], [0.3, 0.6, -0.2]]], device=device)
    token = token.to(torch.float64)
    expected_weights, expected_momentum = reference(model, token, state, shared=True)
    result = model.update_shared(token.squeeze(0), state)
    actual = tuple(w for bank in banks(result, 2) for w in bank)
    for a, b in zip(
        (*actual, *result.momentum), (*expected_weights, *expected_momentum), strict=True
    ):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
    reordered = model.update_shared(token.squeeze(0).flip(0), state)
    for a, b in zip(
        (*actual, *result.momentum),
        (*sum(banks(reordered, 2), ()), *reordered.momentum),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("rule", ["mean_partial", "per_bank"])
@pytest.mark.parametrize("magnitude", [0.2, 10.0])
def test_complete_scalar_jacobian_includes_three_momentum_and_forcing(rule, magnitude):
    model = NeuralMemory(
        MemoryConfig(
            1,
            depth=1,
            normalize_qk=False,
            multiscale=MemoryMultiscale(form="ema_bank"),
            bank_readout=BankReadoutConfig(routing="constant", write_gradient=rule),
        ),
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.key_projection.weight.fill_(1)
        model.value_projection.weight.fill_(0.5)
        for gate in (model.alpha_projection, model.eta_projection, model.theta_projection):
            gate.weight.zero_()
        model.bank_readout.addresses.copy_(torch.tensor([[1.0], [-1.0], [1.0]]))
    state = model.initial_state(1)
    token = torch.tensor([[[magnitude]]], dtype=torch.float64)
    point = torch.tensor([0.1, 0.2, 0.3, 0.01, 0.02, 0.03], dtype=torch.float64)

    def transition(z):
        local = replace(
            state,
            weights=(z[0].reshape(1, 1, 1).clone(),),
            levels=tuple(z[i].reshape(1, 1, 1).clone() for i in (1, 2)),
            momentum=tuple(z[i].reshape(1, 1, 1).clone() for i in (3, 4, 5)),
        )
        result = model.update(token, local, differentiable=True)
        return torch.cat([w.flatten() for w in (*result.weights, *result.levels, *result.momentum)])

    a = torch.softmax(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64), 0)
    projection = torch.eye(3, dtype=torch.float64)
    if rule == "mean_partial":
        projection = torch.ones(3, 3, dtype=torch.float64) / 3
    alpha, eta, theta = 0.5, 0.5, 0.05
    decays = torch.tensor(
        [1 - alpha, *(1 - x for x in model.config.multiscale.bank_decays)], dtype=torch.float64
    )
    identity = torch.eye(3, dtype=torch.float64)
    hessian = 2 * theta * magnitude**2 * projection @ torch.outer(a, a)
    expected = torch.cat(
        (
            torch.cat((torch.diag(decays) - hessian, eta * identity), 1),
            torch.cat((-hessian, eta * identity), 1),
        ),
        0,
    )
    jacobian = torch.autograd.functional.jacobian(transition, point)
    torch.testing.assert_close(jacobian, expected, atol=1e-11, rtol=1e-11)
    forcing = 2 * theta * magnitude * (0.5 * magnitude) * (projection @ a)
    torch.testing.assert_close(
        transition(point), expected @ point + torch.cat((forcing, forcing)), atol=1e-11, rtol=1e-11
    )
    if magnitude == 10:
        assert torch.linalg.eigvals(jacobian).abs().max() > 1


@pytest.mark.parametrize("device", DEVICES)
def test_partition_resume_and_outer_gradients_cover_all_bank_momenta(device):
    model = memory(device=device)
    generator = torch.Generator(device="cpu").manual_seed(9)
    token = torch.randn(2, 3, 3, generator=generator, dtype=torch.float64).to(device)
    state = model.initial_state(2)
    continuous = model.update(token, state)
    reordered = model.update(token.flip(0), map_memory_rows(state, lambda value: value.flip(0)))
    for a, b in zip(
        (*sum(banks(continuous, 2), ()), *continuous.momentum),
        (*sum(banks(reordered, 2), ()), *reordered.momentum),
        strict=True,
    ):
        assert torch.equal(a, b.flip(0))
    first = model.update(token[:, :1], state)
    resumed = model.restore_state(model.export_state(first))
    resumed = model.update(token[:, 1:], resumed)
    for a, b in zip(
        (*sum(banks(continuous, 2), ()), *continuous.momentum),
        (*sum(banks(resumed, 2), ()), *resumed.momentum),
        strict=True,
    ):
        assert torch.equal(a, b)
    observed = token.detach().clone().requires_grad_()
    state = model.update(observed, model.initial_state(2, differentiable=True), differentiable=True)
    model.read(observed[:, :1], state).square().sum().backward()
    assert observed.grad is not None and torch.isfinite(observed.grad).all()
    for name in (
        "initial_weights.0",
        "key_projection.weight",
        "value_projection.weight",
        "bank_readout.addresses",
    ):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("rule", ["mean_partial", "per_bank"])
def test_extra_momentum_storage_is_budgeted_and_not_aliased(rule):
    old = memory("common_translation")
    model = memory(rule)
    state = model.initial_state(2)
    assert len(state.momentum) == 6 and model.config.matrix_elements == 108
    assert (
        sum(
            v.numel() * v.element_size()
            for v in (*state.weights, *state.levels, *state.momentum, state.steps)
        )
        == 1744
    )
    assert old.config.matrix_elements == 72
    with pytest.raises(ValueError, match="presupuesto|bytes"):
        NeuralMemory(
            replace(model.config, max_state_bytes=1168), dtype=torch.float64
        ).initial_state(2)
    for bad in (
        replace(state, momentum=state.momentum[:-1]),
        replace(state, momentum=(state.momentum[0],) * 6),
        replace(state, momentum=tuple(v.half() for v in state.momentum)),
        replace(state, momentum=tuple(torch.full_like(v, float("nan")) for v in state.momentum)),
    ):
        with pytest.raises(ValueError):
            model.validate_state(bad)
    with pytest.raises(ValueError):
        old.restore_state(model.export_state(state))


def test_unknown_bank_gradient_rule_is_rejected():
    with pytest.raises(ValueError):
        BankReadoutConfig(write_gradient="invented")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("residual", [False, True])
def test_each_bank_partial_matches_central_finite_differences(device, residual):
    model = memory(device=device, residual_layer_norm=residual)
    state = distinct_state(model)
    observed = torch.tensor(
        [[[0.2, -0.5, 0.7]], [[0.3, 0.6, -0.2]]], dtype=torch.float64, device=device
    )
    result = model.update(observed, state)
    eta = model.eta_projection(observed.squeeze(1)).sigmoid().unsqueeze(-1)
    theta = model.config.theta_max * model.theta_projection(
        observed.squeeze(1)
    ).sigmoid().unsqueeze(-1)
    gradients = tuple(
        -(new - eta * old) / theta for new, old in zip(result.momentum, state.momentum, strict=True)
    )
    keys = F.normalize(model.key_projection(observed), dim=-1, eps=1e-12)
    target = model.value_projection(observed)
    delta = 1e-5
    for region in range(3):
        for layer in range(2):
            losses = []
            for sign in (-1, 1):
                weights, levels = list(state.weights), list(state.levels)
                chosen, index = (
                    (weights, layer) if region == 0 else (levels, 2 * layer + region - 1)
                )
                chosen[index] = chosen[index].clone()
                chosen[index][0, 1, 2] += sign * delta
                local = replace(state, weights=tuple(weights), levels=tuple(levels))
                losses.append((model.read(keys, local) - target).square().sum())
            numeric = (losses[1] - losses[0]) / (2 * delta)
            torch.testing.assert_close(
                gradients[2 * region + layer][0, 1, 2], numeric, atol=1e-9, rtol=1e-7
            )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rule", ["mean_partial", "per_bank"])
def test_uniform_replica_control_keeps_equal_banks_and_momentum(device, rule):
    model = memory(rule, device=device, routing="uniform", form="replica_bank")
    observed = torch.linspace(-0.5, 0.7, 18, dtype=torch.float64, device=device).reshape(2, 3, 3)
    state = model.update(observed, model.initial_state(2))
    for region in (1, 2):
        for layer in range(2):
            assert torch.equal(banks(state, 2)[0][layer], banks(state, 2)[region][layer])
            assert torch.equal(state.momentum[layer], state.momentum[2 * region + layer])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rule", ["mean_partial", "per_bank"])
def test_unavailable_read_has_zero_current_gradient_but_keeps_momentum_evolution(device, rule):
    base = memory(rule, device=device)
    model = NeuralMemory(
        replace(
            base.config, bank_readout=replace(base.config.bank_readout, available=(False,) * 3)
        ),
        device=device,
        dtype=torch.float64,
    )
    state = distinct_state(model)
    observed = torch.ones(2, 1, 3, dtype=torch.float64, device=device) / 3
    result = model.update(observed, state)
    eta = model.eta_projection(observed.squeeze(1)).sigmoid().unsqueeze(-1)
    for actual, previous in zip(result.momentum, state.momentum, strict=True):
        torch.testing.assert_close(actual, eta * previous, atol=0, rtol=0)
    assert torch.count_nonzero(model.read(observed, result)) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_router_outer_gradient_through_the_inner_derivative_matches_finite_differences(device):
    model = memory(device=device)
    state = distinct_state(model)
    observed = torch.tensor(
        [[[0.2, -0.5, 0.7]], [[0.3, 0.6, -0.2]]], dtype=torch.float64, device=device
    )

    def state_loss(differentiable):
        following = model.update(observed, state, differentiable=differentiable)
        return sum(w.square().sum() for bank in banks(following, 2) for w in bank)

    gradient = torch.autograd.grad(state_loss(True), model.bank_readout.addresses)[0][1, 2]
    address = model.bank_readout.addresses.detach().clone()
    values = []
    try:
        for sign in (-1, 1):
            with torch.no_grad():
                model.bank_readout.addresses.copy_(address)
                model.bank_readout.addresses[1, 2] += sign * 1e-5
            values.append(state_loss(False))
    finally:
        with torch.no_grad():
            model.bank_readout.addresses.copy_(address)
    numeric = (values[1] - values[0]) / 2e-5
    assert gradient.abs() > 1e-8
    torch.testing.assert_close(gradient, numeric, atol=1e-8, rtol=1e-6)


def test_equal_size_states_and_checkpoints_cannot_cross_gradient_rules():
    import copy

    source, destination = memory("per_bank"), memory("mean_partial")
    with pytest.raises(ValueError):
        destination.restore_state(source.export_state(source.initial_state(1)))
    original = copy.deepcopy(destination.state_dict())
    payload = copy.deepcopy(source.state_dict())
    payload["initial_weights.0"] += 1
    with pytest.raises(ValueError):
        destination.load_state_dict(payload)
    for name, value in destination.named_parameters():
        assert torch.equal(value, original[name])


def test_per_bank_initialization_does_not_change_the_global_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(90231)
        before = torch.get_rng_state().clone()
        memory("per_bank")
        assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("device", DEVICES)
def test_opposite_bank_partials_can_cancel_in_the_tied_control_but_not_per_bank(device):
    results = {}
    for rule in ("common_translation", "mean_partial", "per_bank"):
        model = NeuralMemory(
            MemoryConfig(
                1,
                depth=2,
                multiscale=MemoryMultiscale(form="replica_bank"),
                bank_readout=BankReadoutConfig(routing="uniform", write_gradient=rule),
            ),
            device=device,
            dtype=torch.float64,
        ).requires_grad_(False)
        with torch.no_grad():
            model.key_projection.weight.fill_(1)
            model.value_projection.weight.fill_(1)
            for gate in (model.alpha_projection, model.eta_projection, model.theta_projection):
                gate.weight.zero_()
        state = model.initial_state(1)

        def scalar(value):
            return torch.full((1, 1, 1), value, dtype=torch.float64, device=device)

        state = replace(
            state,
            weights=(scalar(0), scalar(1)),
            levels=(scalar(0), scalar(0), scalar(-1), scalar(0)),
        )
        observed = scalar(1)
        assert model.read(observed, state).item() == 0
        following = model.update(observed, state)
        results[rule] = model.read(observed, following).item()
    assert results["common_translation"] == results["mean_partial"] == 0
    assert results["per_bank"] == pytest.approx(0.05 / 18, abs=1e-14, rel=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_finite_loss_with_overflowing_bank_gradient_is_rejected_without_repair(device):
    model = NeuralMemory(
        MemoryConfig(
            1,
            depth=1,
            normalize_qk=False,
            multiscale=MemoryMultiscale(form="replica_bank"),
            bank_readout=BankReadoutConfig(routing="uniform", write_gradient="per_bank"),
        ),
        device=device,
        dtype=torch.float64,
    ).requires_grad_(False)
    with torch.no_grad():
        model.initial_weights[0].zero_()
        model.key_projection.weight.fill_(1)
        model.value_projection.weight.fill_(1e-307)
        for gate in (model.alpha_projection, model.eta_projection, model.theta_projection):
            gate.weight.zero_()
    state = model.initial_state(1)
    observed = torch.full((1, 1, 1), 1e308, device=device, dtype=torch.float64)
    with pytest.raises(ValueError, match="actualización por banco"):
        model.update(observed, state)
    assert all(
        torch.count_nonzero(t) == 0 for t in (*state.weights, *state.levels, *state.momentum)
    )
