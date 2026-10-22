"""Bancos históricos, rutas y gradientes internos con vectores de prueba, sin ajuste."""

import importlib
import math
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from mars_titan.models.titans.config import MemoryConfig, MemoryMultiscale
from mars_titan.models.titans.neural_memory import NeuralMemory

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def config(**options):
    try:
        api = importlib.import_module("mars_titan.models.titans.bank_readout")
    except ModuleNotFoundError:
        pytest.fail("Falta la lectura contextual de bancos históricos")
    return api.BankReadoutConfig(**options)


def memory(device="cpu", dtype=torch.float64, *, depth=1, form="ema_bank", **options):
    return NeuralMemory(
        MemoryConfig(
            dim=2,
            depth=depth,
            normalize_qk=False,
            max_batch=4,
            max_tokens=8,
            multiscale=MemoryMultiscale(form=form),
            bank_readout=config(**options),
        ),
        device=device,
        dtype=dtype,
    )


def distinct(model):
    state = model.initial_state(1, differentiable=True)
    eye = torch.eye(2, dtype=state.weights[0].dtype, device=state.weights[0].device)[None]
    return replace(
        state,
        weights=tuple(eye.clone() for _ in state.weights),
        levels=tuple(scale * eye for _ in state.weights for scale in (2.0, 4.0)),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("routing", ["uniform", "query"])
def test_read_combines_complete_bank_functions_with_query_weights(device, routing):
    model = memory(device, routing=routing, depth=2)
    state = distinct(model)
    with torch.no_grad():
        model.bank_readout.addresses.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], device=device)
        )
    query = torch.tensor([[[1.0, -2.0]]], device=device, dtype=torch.float64)
    raw = [scale * F.gelu(scale * query) for scale in (1.0, 2.0, 4.0)]
    logits = torch.tensor(
        [1 / math.sqrt(5), -2 / math.sqrt(5), -1 / math.sqrt(5)], device=device, dtype=torch.float64
    )
    probabilities = (
        torch.softmax(logits, 0) if routing == "query" else torch.full_like(logits, 1 / 3)
    )
    expected = sum(p * value for p, value in zip(probabilities, raw, strict=True))
    result = model.read(query, state)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
    averaged_weights = sum((1.0, 2.0, 4.0)) / 3
    assert not torch.allclose(result, averaged_weights * F.gelu(averaged_weights * query))
    assert state.steps.tolist() == [0]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("routing", ["uniform", "query"])
def test_write_uses_gradient_of_common_translation_not_mean_weight_network(device, routing):
    model = memory(device, depth=2, routing=routing)
    state = distinct(model)
    with torch.no_grad():
        model.key_projection.weight.copy_(torch.eye(2, device=device))
        model.value_projection.weight.copy_(torch.eye(2, device=device))
        for projection in (model.alpha_projection, model.eta_projection, model.theta_projection):
            projection.weight.zero_()
    x = torch.tensor([[[0.2, -0.3]]], device=device, dtype=torch.float64)
    bank_weights = [state.weights] + [
        tuple(state.levels[2 * i + r] for i in range(2)) for r in range(2)
    ]
    delta = tuple(torch.zeros_like(w, requires_grad=True) for w in state.weights)
    logits = F.normalize(x, dim=-1) @ F.normalize(model.bank_readout.addresses, dim=-1).T
    probs = torch.softmax(logits, -1) if routing == "query" else torch.ones_like(logits) / 3
    outputs = []
    for weights in bank_weights:
        z = x
        for layer, (w, d) in enumerate(zip(weights, delta, strict=True)):
            z = z @ (w + d).transpose(1, 2)
            if layer == 0:
                z = F.gelu(z)
        outputs.append(z)
    predicted = (torch.stack(outputs, dim=2) * probs.unsqueeze(-1)).sum(2)
    gradients = torch.autograd.grad((predicted - x).square().sum(), delta)
    actual = model.update(x, state, differentiable=True)
    for i, g in enumerate(gradients):
        torch.testing.assert_close(actual.momentum[i], -0.05 * g, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(
            actual.weights[i], 0.5 * state.weights[i] - 0.05 * g, rtol=1e-11, atol=1e-12
        )
        h = 1e-6
        direction = torch.zeros_like(g)
        direction[0, 0, 0] = h
        losses = []
        for sign in (-1, 1):
            weights, levels = list(state.weights), list(state.levels)
            weights[i] = weights[i] + sign * direction
            for region in range(2):
                levels[2 * i + region] = levels[2 * i + region] + sign * direction
            shifted = replace(state, weights=tuple(weights), levels=tuple(levels))
            losses.append((model.read(x, shifted) - x).square().sum().item())
        assert (losses[1] - losses[0]) / (2 * h) == pytest.approx(
            g[0, 0, 0].item(), rel=1e-6, abs=1e-8
        )
    assert actual.steps.tolist() == [1]


@pytest.mark.parametrize("device", DEVICES)
def test_all_region_masks_are_finite_and_absent_banks_do_not_affect_read(device):
    import itertools

    query = torch.tensor([[[0.3, -0.2]]], device=device, dtype=torch.float64)
    for mask in itertools.product((False, True), repeat=3):
        model = memory(device, available=mask)
        state = distinct(model)
        result = model.read(query, state)
        changed = list((state.weights[0], *state.levels))
        for i, present in enumerate(mask):
            if not present:
                changed[i] = changed[i] + 1000
        erased = replace(state, weights=(changed[0],), levels=tuple(changed[1:]))
        torch.testing.assert_close(model.read(query, erased), result, rtol=0, atol=0)
        assert torch.isfinite(result).all()
        if not any(mask):
            assert torch.equal(result, torch.zeros_like(result))


@pytest.mark.parametrize("device", DEVICES)
def test_replica_control_keeps_identical_regions_with_the_same_state_bytes(device):
    model = memory(device, form="replica_bank")
    base = memory(device)
    state = model.initial_state(2)
    for _step in range(3):
        x = torch.tensor([[[0.1, -0.2]], [[0.3, -0.4]]], device=device, dtype=torch.float64)
        state = model.update(x, state)
        for layer, w in enumerate(state.weights):
            assert torch.equal(w, state.levels[2 * layer])
            assert torch.equal(w, state.levels[2 * layer + 1])
    assert sum(
        t.numel() * t.element_size()
        for t in (*state.weights, *state.momentum, *state.levels, state.steps)
    ) == sum(
        t.numel() * t.element_size()
        for t in (
            *base.initial_state(2).weights,
            *base.initial_state(2).momentum,
            *base.initial_state(2).levels,
            base.initial_state(2).steps,
        )
    )


@pytest.mark.parametrize(
    "options",
    [
        {"routing": "unknown"},
        {"temperature": 0},
        {"temperature": True},
        {"temperature": float("nan")},
        {"available": (True,)},
        {"available": (1, True, True)},
    ],
)
def test_invalid_bank_readout_contracts_fail(options):
    with pytest.raises(ValueError):
        config(**options)


def test_mean_weights_cannot_be_mistaken_for_the_query_dependent_operator():
    model = memory()
    with pytest.raises(ValueError, match="equivalente|consulta|media"):
        model.readable_weights(model.initial_state(1))


def test_invalid_child_checkpoint_does_not_partially_replace_memory_parameters():
    import copy

    model = memory()
    before = copy.deepcopy(model.state_dict())
    changed = copy.deepcopy(before)
    changed["key_projection.weight"] += 10
    changed["bank_readout.available"][0] = False
    with pytest.raises(ValueError):
        model.load_state_dict(changed)
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(model.state_dict()[name], value), name


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_readout_handles_zero_and_extreme_queries_without_hiding_invalid_inputs(device, dtype):
    model = memory(device, dtype=dtype, available=(False, False, False))
    limit = torch.finfo(dtype).max
    query = torch.tensor(
        [[[0.0, 0.0], [limit, -limit]]], device=device, dtype=dtype, requires_grad=True
    )
    outputs = torch.ones(1, 2, 3, 2, device=device, dtype=dtype, requires_grad=True)
    result = model.bank_readout(query, outputs)
    assert torch.equal(result, torch.zeros_like(result))
    result.sum().backward()
    assert torch.isfinite(query.grad).all() and torch.isfinite(outputs.grad).all()
    invalid = outputs.detach().clone()
    invalid[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN|infinito"):
        model.bank_readout(query, invalid)


@pytest.mark.parametrize("device", DEVICES)
def test_shared_read_and_update_keep_row_permutation_and_recover_the_next_state(device):
    model = memory(device, depth=2)
    x = torch.tensor([[0.2, -0.1], [0.4, 0.3], [-0.2, 0.1]], dtype=torch.float64, device=device)
    state = distinct(model)
    a = model.update_shared(x, state)
    b = model.update_shared(x.flip(0), state)
    for aa, bb in zip(
        (*a.weights, *a.levels, *a.momentum), (*b.weights, *b.levels, *b.momentum), strict=True
    ):
        torch.testing.assert_close(aa, bb, rtol=1e-12, atol=1e-12)
    restored = model.restore_state(model.export_state(a))
    torch.testing.assert_close(
        model.read_shared(x, restored), model.read_shared(x, a), rtol=0, atol=0
    )
    next_a, next_b = model.update_shared(x, a), model.update_shared(x, restored)
    for aa, bb in zip(
        (*next_a.weights, *next_a.levels), (*next_b.weights, *next_b.levels), strict=True
    ):
        torch.testing.assert_close(aa, bb, rtol=0, atol=0)


@pytest.mark.parametrize("magnitude", [0.2, 10.0])
def test_complete_linear_bank_jacobian_includes_shared_gradient_and_momentum(magnitude):
    model = NeuralMemory(
        MemoryConfig(
            dim=1,
            depth=1,
            normalize_qk=False,
            multiscale=MemoryMultiscale(form="ema_bank"),
            bank_readout=config(),
        ),
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.key_projection.weight.fill_(1)
        model.value_projection.weight.fill_(1)
        model.bank_readout.addresses.copy_(
            torch.tensor([[1.0], [0.0], [-1.0]], dtype=torch.float64)
        )
        for gate in (model.alpha_projection, model.eta_projection, model.theta_projection):
            gate.weight.zero_()
    original = model.initial_state(1)
    x = torch.tensor([[[magnitude]]], dtype=torch.float64)

    def update(vector):
        tensors = tuple(v.reshape(1, 1, 1).clone() for v in vector)
        state = replace(
            original, weights=(tensors[0],), levels=(tensors[1], tensors[2]), momentum=(tensors[3],)
        )
        state = model.update(x, state, differentiable=True)
        return torch.stack(
            [
                state.weights[0].reshape(()),
                state.levels[0].reshape(()),
                state.levels[1].reshape(()),
                state.momentum[0].reshape(()),
            ]
        )

    actual = torch.autograd.functional.jacobian(
        update, torch.tensor([0.1, 0.2, 0.3, 0.05], dtype=torch.float64)
    )
    a = torch.softmax(torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64), 0)
    expected = torch.zeros(4, 4, dtype=torch.float64)
    expected[:, :3] = -0.1 * magnitude**2 * a[None, :]
    expected[:, 3] = 0.5
    expected[:3, :3] += torch.diag(
        torch.tensor([0.5, 1 - 1 / 64, 1 - 1 / 256], dtype=torch.float64)
    )
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    if magnitude == 10.0:
        assert torch.linalg.eigvals(actual).abs().max() > 1


@pytest.mark.parametrize("device", DEVICES)
def test_learnable_constant_routing_is_a_query_independent_control(device):
    model = memory(device, routing="constant")
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64, device=device)
    probabilities = model.bank_readout.probabilities(query)
    torch.testing.assert_close(probabilities[:, 0], probabilities[:, 1], rtol=0, atol=0)
    (probabilities[..., 0].sum()).backward()
    assert model.bank_readout.addresses.grad is not None
    assert torch.count_nonzero(model.bank_readout.addresses.grad)


@pytest.mark.parametrize("device", DEVICES)
def test_bank_parameters_and_fast_state_reload_before_the_next_update(device):
    import copy

    model = memory(device, depth=2)
    state = distinct(model)
    with torch.no_grad():
        model.bank_readout.addresses.add_(0.25)
    saved = copy.deepcopy(model.state_dict())
    restored = memory(device, depth=2)
    restored.load_state_dict(saved)
    state2 = restored.restore_state(model.export_state(state))
    x = torch.tensor([[[0.2, -0.3]]], device=device, dtype=torch.float64)
    torch.testing.assert_close(restored.read(x, state2), model.read(x, state), rtol=0, atol=0)
    a, b = model.update(x, state), restored.update(x, state2)
    for aa, bb in zip(
        (*a.weights, *a.levels, *a.momentum), (*b.weights, *b.levels, *b.momentum), strict=True
    ):
        assert torch.equal(aa, bb)


@pytest.mark.parametrize("change", ["missing", "dtype", "shape", "extra"])
def test_bank_checkpoint_rejects_incompatible_contract_before_copying(change):
    import copy

    model = memory().bank_readout
    state = copy.deepcopy(model.state_dict())
    before = model.addresses.detach().clone()
    state["addresses"] += 1
    if change == "missing":
        del state["available"]
    elif change == "dtype":
        state["addresses"] = state["addresses"].float()
    elif change == "shape":
        state["addresses"] = state["addresses"][:2]
    else:
        state["_extra_state"]["configuration"]["temperature"] = 2.0
    with pytest.raises(ValueError):
        model.load_state_dict(state)
    assert torch.equal(model.addresses, before)


def test_query_budgets_types_and_autocast_are_explicit():
    model = memory().bank_readout
    for query in (
        torch.zeros(2, 2),
        torch.zeros(1, 16385, 2, dtype=torch.float64),
        torch.zeros(1, 1, 2, dtype=torch.float32),
    ):
        with pytest.raises(ValueError):
            model.probabilities(query)
    with (
        torch.autocast("cpu", dtype=torch.bfloat16),
        pytest.raises(ValueError, match="precisión|FP32"),
    ):
        model.probabilities(torch.ones(1, 1, 2, dtype=torch.float64))
    with pytest.raises(ValueError):
        model(
            torch.ones(1, 1, 2, dtype=torch.float64), torch.zeros(1, 1, 2, 2, dtype=torch.float64)
        )
    with pytest.raises(ValueError):
        model.operation_counts(1, mode="unknown", depth=2)


def test_replica_control_has_no_fixed_decay_horizon_and_reports_only_homogeneous_bound():
    cfg = MemoryMultiscale(form="replica_bank")
    with pytest.raises(ValueError, match="escala fija"):
        _ = cfg.timescales
    assert cfg.transition(0.25) == ((0.75, 0.0, 0.0), (0.0, 0.75, 0.0), (0.0, 0.0, 0.75))
    rho, weights = cfg.contraction(0.1)
    assert rho == pytest.approx(0.9) and weights == (1.0, 1.0, 1.0)
