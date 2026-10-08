"""Proyección del estado real con fixtures matemáticos, sin optimizadores."""

import copy
import importlib
from dataclasses import replace

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from mars_titan.models.titans import MACConfig, MemoryConfig, TitansMAC


def api():
    try:
        return importlib.import_module("mars_titan.models.titans.local_control")
    except ModuleNotFoundError:
        pytest.fail("Falta el control local proyectado del estado MAC")


def fixture(*, dimension=2, batch=1, dtype=torch.float64, **options):
    mac = TitansMAC(
        MACConfig(
            memory=MemoryConfig(dim=dimension, max_tokens=1, max_batch=4),
            persistent_tokens=2,
            max_segment=1,
        ),
        dtype=dtype,
    )
    settings = dict(mode="diagnostic", rank=2, frequency=1, grid_size=13)
    settings.update(options)
    control = api().MACProjectionControl(
        api().MACProjectionConfig(**settings), mac.config, dtype=dtype
    )
    token = torch.linspace(0.1, 0.3, batch * dimension, dtype=dtype).reshape(batch, 1, dimension)
    return mac, control, token, mac.initial_state(batch)


def evaluate(control, mac, token, state, *, flows=None, differentiable=False):
    flows = flows or tuple(f"flow-{index}" for index in range(token.shape[0]))
    counters = state.memory.steps.detach().cpu()
    selection = control.select_flows(flows, counters, context_id="a" * 64)
    return control(
        mac,
        token,
        state,
        flow_ids=flows,
        observed_steps=counters,
        selection=selection,
        context_id="a" * 64,
        differentiable=differentiable,
    )


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "unknown"},
        {"rank": 0},
        {"rank": True},
        {"rank": 5},
        {"frequency": 0},
        {"seed": -1},
        {"grid_size": 1},
        {"threshold": float("nan")},
        {"threshold": -1},
        {"weight": float("inf")},
        {"mode": "penalty", "weight": 0},
        {"mode": "diagnostic", "weight": 1},
        {"max_flows": 0},
        {"max_estimated_bytes": True},
    ],
)
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        api().MACProjectionConfig(**options)


def test_projected_operator_matches_dense_complete_state_jacobian():
    mac, control, token, state = fixture()
    result = evaluate(control, mac, token, state)
    flat = torch.cat([value.flatten() for value in (*state.memory.weights, *state.memory.momentum)])

    def transition(point):
        layers = tuple(value.reshape(1, 2, 2).clone() for value in point.split(4))
        supplied = replace(
            state, memory=replace(state.memory, weights=layers[:2], momentum=layers[2:])
        )
        _, following = mac(token, supplied, differentiable=True)
        return torch.cat(
            [value.flatten() for value in (*following.memory.weights, *following.memory.momentum)]
        )

    with sdpa_kernel(SDPBackend.MATH):
        complete = torch.autograd.functional.jacobian(transition, flat)
    expected = control.basis.T @ complete @ control.basis
    torch.testing.assert_close(result.operators[0], expected, rtol=1e-11, atol=1e-12)
    assert result.reevaluations == control.config.rank
    assert result.penalty is None
    assert not result.operators.requires_grad
    assert state.memory.steps.tolist() == [0]
    assert all(parameter.grad is None for parameter in mac.parameters())


def test_external_gradient_keeps_current_token_path_without_history_graph():
    mac, control, token, state = fixture(mode="penalty", threshold=0.0, weight=0.3)
    token.requires_grad_()
    original = state.memory.weights[0].requires_grad_()

    def penalty(value):
        return evaluate(control, mac, value, state, differentiable=True).penalty

    assert torch.autograd.gradcheck(penalty, (token,), eps=1e-5, atol=1e-6, rtol=1e-4)
    loss = penalty(token)
    gradients = torch.autograd.grad(
        loss, (token, original, mac.memory.eta_projection.weight), allow_unused=True
    )
    assert gradients[0].abs().sum() > 0
    assert gradients[1] is None
    assert gradients[2].abs().sum() > 0
    assert all(parameter.grad is None for parameter in mac.parameters())


def test_frequency_and_canonical_selection_are_invariant_to_row_order():
    mac, control, token, state = fixture(batch=3, max_flows=1, frequency=2)
    assert evaluate(control, mac, token, state, flows=("c", "a", "b")) is None
    state = replace(state, memory=replace(state.memory, steps=torch.ones(3, dtype=torch.int64)))
    result = evaluate(control, mac, token, state, flows=("c", "a", "b"))
    permutation = [2, 0, 1]
    memory = replace(
        state.memory,
        weights=tuple(value[permutation] for value in state.memory.weights),
        momentum=tuple(value[permutation] for value in state.memory.momentum),
        steps=state.memory.steps[permutation],
    )
    reordered = evaluate(
        control, mac, token[permutation], replace(state, memory=memory), flows=("b", "c", "a")
    )
    assert result.flow_ids == reordered.flow_ids == ("a",)
    assert result.indices == (1,)
    assert reordered.indices == (2,)
    assert result.eligible_flows == 3
    torch.testing.assert_close(result.operators, reordered.operators, rtol=0, atol=0)


def test_basis_roundtrip_dtype_conversion_and_rng_are_identified():
    rng = torch.random.get_rng_state().clone()
    mac, control, token, state = fixture()
    restored = api().MACProjectionControl(control.config, mac.config, dtype=torch.float64)
    restored.load_state_dict(copy.deepcopy(control.state_dict()))
    result = evaluate(control, mac, token, state)
    recovered = evaluate(restored, mac, token, state)
    torch.testing.assert_close(result.operators, recovered.operators, rtol=0, atol=0)
    basis = control.basis.clone()
    control.float().double()
    torch.testing.assert_close(control.basis, basis, rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("corruption", ["mode", "basis", "frequency", "mac"])
def test_loading_rejects_incompatible_contracts_before_copying(corruption):
    mac, control, _, _ = fixture()
    payload = copy.deepcopy(control.state_dict())
    if corruption == "basis":
        payload["basis"][0, 0] += 0.01
    else:
        extra = payload["_extra_state"]
        if corruption == "mac":
            extra["mac_contract"]["heads"] = 2
        else:
            extra["configuration"][corruption] = "disabled" if corruption == "mode" else 3
    before = control.basis.clone()
    with pytest.raises(ValueError):
        control.load_state_dict(payload, strict=False)
    torch.testing.assert_close(control.basis, before, rtol=0, atol=0)
    assert control.get_extra_state()["mac_contract"] == mac.config.identity()


def test_budget_is_checked_before_allocating_a_basis(monkeypatch):
    mac = TitansMAC(MACConfig(memory=MemoryConfig(dim=2)))
    configuration = api().MACProjectionConfig(max_estimated_bytes=1)
    monkeypatch.setattr(
        torch.linalg, "qr", lambda *args, **kwargs: pytest.fail("No debe construirse la base")
    )
    with pytest.raises(ValueError, match="presupuesto"):
        api().MACProjectionControl(configuration, mac.config)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_precision_and_nonfinite_inputs(dtype):
    mac, control, token, state = fixture(dtype=dtype)
    result = evaluate(control, mac, token, state)
    assert result.operators.dtype == dtype
    assert torch.isfinite(result.estimates.angular_corrected_estimate).all()
    invalid = token.clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        evaluate(control, mac, invalid, state)


def test_disabled_does_not_evaluate_transition(monkeypatch):
    mac, control, token, state = fixture(mode="disabled")
    monkeypatch.setattr(mac, "forward", lambda *args, **kwargs: pytest.fail("No debe evaluar F"))
    assert evaluate(control, mac, token, state) is None


@pytest.mark.parametrize("differentiable", [False, True])
def test_penalty_respects_explicit_gradient_mode(differentiable):
    mac, control, token, state = fixture(mode="penalty", threshold=0.0, weight=0.2)
    result = evaluate(control, mac, token, state, differentiable=differentiable)
    assert result.penalty > 0
    assert result.penalty.requires_grad is differentiable


def test_explicit_cpu_construction_ignores_ambient_meta_device():
    mac = TitansMAC(MACConfig(memory=MemoryConfig(dim=2)))
    with torch.device("meta"):
        control = api().MACProjectionControl(
            api().MACProjectionConfig(rank=2), mac.config, device="cpu"
        )
    assert control.basis.device.type == "cpu"


def test_logical_selection_and_penalty_are_invariant_to_physical_partition():
    mac, control, token, state = fixture(
        batch=3, mode="penalty", weight=0.2, threshold=0.0, max_flows=2
    )
    flows = ("c", "a", "b")
    selection = control.select_flows(flows, state.memory.steps, context_id="a" * 64)
    whole = control(
        mac,
        token,
        state,
        flow_ids=flows,
        observed_steps=state.memory.steps,
        selection=selection,
        context_id="a" * 64,
        differentiable=True,
    )
    penalties, matrices, calls = [], {}, 0
    for index, flow in enumerate(flows):
        memory = replace(
            state.memory,
            weights=tuple(value[index : index + 1].clone() for value in state.memory.weights),
            momentum=tuple(value[index : index + 1].clone() for value in state.memory.momentum),
            steps=state.memory.steps[index : index + 1].clone(),
        )
        result = control(
            mac,
            token[index : index + 1],
            replace(state, memory=memory),
            flow_ids=(flow,),
            observed_steps=memory.steps,
            selection=selection,
            context_id="a" * 64,
            differentiable=True,
        )
        if result is not None:
            penalties.append(result.penalty)
            matrices[flow] = result.operators[0]
            calls += result.reevaluations
            assert result.selection_id == whole.selection_id
    assert whole.flow_ids == ("a", "b")
    torch.testing.assert_close(torch.stack(penalties).sum(), whole.penalty, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(
        torch.stack([matrices[flow] for flow in whole.flow_ids]), whole.operators, rtol=0, atol=0
    )
    assert calls == whole.reevaluations == 4


def test_active_control_requires_explicit_logical_selection():
    mac, control, token, state = fixture()
    with pytest.raises(ValueError, match="selección"):
        control(mac, token, state, flow_ids=("a",), observed_steps=state.memory.steps)


def test_selection_rejects_different_counters_and_forged_ids_before_transition(monkeypatch):
    mac, control, token, state = fixture()
    selection = control.select_flows(("a",), state.memory.steps, context_id="a" * 64)
    stale = replace(state, memory=replace(state.memory, steps=torch.ones(1, dtype=torch.int64)))
    monkeypatch.setattr(mac, "forward", lambda *args, **kwargs: pytest.fail("No debe evaluar F"))
    for current, plan in (
        (stale, selection),
        (state, replace(selection, selected_flow_ids=("b",))),
    ):
        with pytest.raises(ValueError):
            control(
                mac,
                token,
                current,
                flow_ids=("a",),
                observed_steps=current.memory.steps,
                selection=plan,
                context_id="a" * 64,
            )
    with pytest.raises(ValueError, match="contexto"):
        control(
            mac,
            token,
            state,
            flow_ids=("a",),
            observed_steps=state.memory.steps,
            selection=selection,
            context_id="b" * 64,
        )


def test_logical_budget_is_not_reset_by_a_small_physical_batch():
    mac, generous, token, state = fixture(batch=2, max_flows=2)
    limit = generous.estimated_bytes(1)
    control = api().MACProjectionControl(
        replace(generous.config, max_estimated_bytes=limit), mac.config, dtype=torch.float64
    )
    with pytest.raises(ValueError, match="presupuesto"):
        control.select_flows(("a", "b"), state.memory.steps, context_id="a" * 64)


def test_selection_counters_keep_integer_types():
    mac, control, token, state = fixture()
    selection = control.select_flows(("a",), state.memory.steps, context_id="a" * 64)
    with pytest.raises(ValueError):
        control(
            mac,
            token,
            state,
            flow_ids=("a",),
            observed_steps=state.memory.steps,
            selection=replace(selection, eligible_flows=True),
            context_id="a" * 64,
        )


@pytest.mark.parametrize("dimension,rank", [(2, 1), (32, 4), (64, 4)])
def test_estimate_covers_observed_saved_storage_events(dimension, rank):
    mac, control, token, state = fixture(
        dimension=dimension, rank=rank, grid_size=64, mode="penalty", weight=0.2, threshold=0.0
    )
    saved = []

    def record(tensor):
        saved.append(tensor.untyped_storage().nbytes())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        result = evaluate(control, mac, token, state, differentiable=True)
        torch.autograd.grad(result.penalty, tuple(mac.parameters()), allow_unused=True)
    assert sum(saved) < control.estimated_bytes(1)


@pytest.mark.parametrize("case", ["contract", "frozen", "dimension", "dtype", "rank"])
def test_constructor_rejects_unsupported_contracts(case):
    settings = api().MACProjectionConfig(rank=2)
    mac = MACConfig(memory=MemoryConfig(dim=2))
    dtype = torch.float64
    if case == "contract":
        settings = None
    elif case == "frozen":
        mac = replace(mac, memory_mode="frozen")
    elif case == "dimension":
        mac = replace(mac, memory=MemoryConfig(dim=128))
    elif case == "dtype":
        dtype = torch.float16
    else:
        mac = replace(mac, memory=MemoryConfig(dim=1, depth=1))
        settings = replace(settings, rank=3)
    with pytest.raises(ValueError):
        api().MACProjectionControl(settings, mac, dtype=dtype)


@pytest.mark.parametrize("case", ["context", "list_ids", "float_counter", "negative", "overflow"])
def test_logical_plan_rejects_invalid_contexts_ids_and_counters(case):
    _, control, _, state = fixture()
    ids, counters, context = ("a",), state.memory.steps, "a" * 64
    if case == "context":
        context = "not-a-digest"
    elif case == "list_ids":
        ids = ["a"]
    elif case == "float_counter":
        counters = counters.float()
    elif case == "negative":
        counters = torch.tensor([-1], dtype=torch.int64)
    else:
        counters = torch.tensor([torch.iinfo(torch.int64).max], dtype=torch.int64)
    with pytest.raises(ValueError):
        control.select_flows(ids, counters, context_id=context)


def test_runtime_contract_precision_and_basis_changes_are_rejected():
    mac, control, token, state = fixture()
    other = TitansMAC(replace(mac.config, heads=2), dtype=torch.float64)
    with pytest.raises(ValueError):
        evaluate(control, other, token, state)
    with torch.inference_mode(), pytest.raises(ValueError):
        evaluate(control, mac, token, state)
    control.float()
    with pytest.raises(ValueError):
        evaluate(control, mac, token, state)
    control.double()
    with torch.no_grad():
        control.basis.add_(0.01)
    with pytest.raises(ValueError):
        evaluate(control, mac, token, state)


def test_strong_basis_boundary_detects_data_bypass_and_half_cast_leaves_basis_intact():
    _, control, _, _ = fixture()
    previous = control.basis.clone()
    with pytest.raises(ValueError):
        control.half()
    torch.testing.assert_close(control.basis, previous, rtol=0, atol=0)
    control.basis.data[0, 0] += 0.01
    with pytest.raises(ValueError):
        control.state_dict()
