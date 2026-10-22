"""Ecuación regional síncrona, contratos y derivadas sin ajustar parámetros."""

import hashlib
import importlib
import importlib.util
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_evidence_bridge import bridge, read, snapshot

from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig, SharedEvidenceBridge

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def api():
    name = "mars_titan.models.titans.bridge_recurrence"
    assert importlib.util.find_spec(name) is not None, "Falta la recurrencia regional opcional"
    return importlib.import_module(name)


def configuration(*, steps=2, budget=512, **options):
    recurrence = api().BridgeRecurrenceConfig(steps, budget)
    return EvidenceBridgeConfig(
        **dict(
            dict(
                codec_id="a" * 64,
                hidden_size=2,
                regions=("one", "two", "three"),
                cycles=0,
                recurrence=recurrence,
            ),
            **options,
        )
    )


def operands(device):
    local = torch.tensor(
        [[[0.2, -0.3], [0.7, 0.4]], [[-0.5, 0.1], [0.6, -0.2]]], dtype=torch.float64, device=device
    )
    graph = torch.tensor(
        [[[0.25, 0.75], [0.6, 0.4]], [[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float64, device=device
    )
    weights = torch.tensor(
        [[[1.2, 0.3], [-0.1, 0.7]], [[0.8, -0.4], [0.2, 1.1]]], dtype=torch.float64, device=device
    )
    available = torch.tensor([[True, True], [True, False]], device=device)
    return local, graph, weights, available


def independent(local, graph, weights, available, steps, gain):
    result = local * available[..., None]
    for _ in range(steps):
        rows = []
        for batch in range(len(local)):
            regions = []
            for target in range(local.shape[1]):
                message = sum(
                    (
                        graph[batch, target, sender] * result[batch, sender]
                        for sender in range(local.shape[1])
                        if available[batch, sender]
                    ),
                    start=torch.zeros_like(result[batch, target]),
                )
                value = (1 - gain) * result[batch, target] + gain * torch.tanh(
                    message @ weights[target]
                )
                regions.append(value if available[batch, target] else torch.zeros_like(value))
            rows.append(torch.stack(regions))
        result = torch.stack(rows)
    return result


def test_disabled_recurrence_preserves_pinned_legacy_identity_and_tensors():
    assert "recurrence" in EvidenceBridgeConfig.__dataclass_fields__, "Falta la opción desactivable"
    expected = json.loads(
        (Path(__file__).parent / "fixtures/shared_evidence_bridge_v1.json").read_text()
    )
    model = bridge()
    explicit = SharedEvidenceBridge(replace(model.config, recurrence=None), dtype=torch.float64)
    assert canonical(explicit.config.identity()) == expected["configuration_canonical"]
    assert sorted(explicit.state_dict()) == expected["state_keys"]
    for name, digest in expected["tensor_sha256"].items():
        value = explicit.state_dict()[name].detach().contiguous().numpy().tobytes()
        assert hashlib.sha256(value).hexdigest() == digest


@pytest.mark.parametrize("device", DEVICES)
def test_legacy_output_and_rng_are_unchanged(device):
    assert "recurrence" in EvidenceBridgeConfig.__dataclass_fields__
    expected = json.loads(
        (Path(__file__).parent / "fixtures/shared_evidence_bridge_v1.json").read_text()
    )
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone() if device != "cpu" else None
    model = bridge(device)
    query = torch.tensor([[0.25, -0.5], [-0.1, 0.4]], dtype=torch.float64, device=device)
    actual = read(model, query, snapshot(device))
    for name, values in expected["outputs"][device].items():
        value = getattr(actual, name).detach().cpu()
        wanted = torch.tensor(values, dtype=value.dtype)
        assert torch.equal(value, wanted)
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(0))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("steps", [1, 2, 4])
def test_nonlinear_steps_match_synchronous_receiver_weight_equation(device, steps):
    module = api()
    local, graph, weights, available = operands(device)
    before = local.clone()
    result = module.nonlinear_mix(local, graph, weights, available, steps=steps, gain=0.35)
    expected = independent(local, graph, weights, available, steps, 0.35)
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
    assert torch.equal(local, before)
    assert torch.count_nonzero(result[1, 1]) == 0
    altered = local.clone()
    altered[1, 1].fill_(1e100)
    torch.testing.assert_close(
        result,
        module.nonlinear_mix(altered, graph, weights, available, steps=steps, gain=0.35),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_active_path_keeps_one_pool_and_snapshot_and_uses_recurrence(device):
    module = api()
    before_rng = torch.random.get_rng_state().clone()
    model = SharedEvidenceBridge(configuration(), device=device, dtype=torch.float64)
    reference = bridge(device, cycles=0)
    assert dict(model.named_parameters()).keys() == dict(reference.named_parameters()).keys()
    for name, value in reference.named_parameters():
        assert torch.equal(value, dict(model.named_parameters())[name])
    query = torch.tensor([[0.2, -0.7], [0.4, 0.8]], dtype=torch.float64, device=device)
    source = snapshot(device)
    saved = source.export_cpu()
    available = torch.tensor([[True, True, True], [True, False, True]], device=device)
    initial = read(reference, query, source, available=available)
    calls = []
    handle = model.value.register_forward_pre_hook(
        lambda _model, arguments: calls.append(arguments[0].shape)
    )
    try:
        result = read(model, query, source, available=available)
    finally:
        handle.remove()
    expected = module.nonlinear_mix(
        initial.region_values,
        initial.bridge_weights,
        model.region_value,
        available,
        steps=2,
        gain=0.25,
    )
    torch.testing.assert_close(result.region_values, expected, atol=1e-12, rtol=1e-12)
    assert not torch.equal(result.region_values, initial.region_values)
    assert torch.equal(result.evidence_ids, initial.evidence_ids)
    assert torch.equal(result.evidence_weights, initial.evidence_weights)
    assert calls == [torch.Size((6, 65))]
    assert source.export_cpu()["sha256"] == saved["sha256"]
    assert torch.equal(before_rng, torch.random.get_rng_state())


@pytest.mark.parametrize("device", DEVICES)
def test_empty_and_missing_views_remain_zero(device):
    model = SharedEvidenceBridge(configuration(steps=4), device=device, dtype=torch.float64)
    query = torch.ones((2, 2), dtype=torch.float64, device=device)
    empty = read(model, query, snapshot(device, count=0))
    missing = read(
        model,
        query,
        snapshot(device),
        available=torch.zeros((2, 3), dtype=torch.bool, device=device),
    )
    assert not empty.values.count_nonzero() and not missing.values.count_nonzero()
    assert not empty.region_values.count_nonzero() and not missing.region_values.count_nonzero()


def test_active_identity_zero_gain_and_checkpoint_rejection_precede_copy():
    model = SharedEvidenceBridge(configuration(bridge_gain=0.0), dtype=torch.float64)
    old = bridge(cycles=0, bridge_gain=0.0)
    query = torch.tensor([[0.2, -0.5]], dtype=torch.float64)
    assert torch.equal(read(model, query, snapshot()).values, read(old, query, snapshot()).values)
    assert model.config.identity()["kind"] == "shared_evidence_bridge_nonlinear_v1"
    assert canonical(model.config.identity()) != canonical(old.config.identity())
    for target in (old, SharedEvidenceBridge(configuration(steps=4), dtype=torch.float64)):
        before = deepcopy(target.state_dict())
        with pytest.raises(ValueError):
            target.load_state_dict(deepcopy(model.state_dict()))
        for name, value in before.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, target.state_dict()[name])
    restored = SharedEvidenceBridge(model.config, dtype=torch.float64)
    restored.load_state_dict(deepcopy(model.state_dict()))
    assert torch.equal(
        read(restored, query, snapshot()).values, read(model, query, snapshot()).values
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(steps=0),
        dict(steps=3),
        dict(steps=True),
        dict(budget=True),
        dict(budget=0),
        dict(budget=1217),
    ],
)
def test_configuration_rejects_unbounded_or_ambiguous_work(options):
    with pytest.raises(ValueError):
        configuration(**options)


def test_active_recurrence_rejects_legacy_cycles_and_counts_j_without_k_reset():
    with pytest.raises(ValueError):
        configuration(cycles=1)
    recurrence = api().BridgeRecurrenceConfig(4, 128)
    assert recurrence.node_updates(8, 4) == 128
    assert recurrence.node_updates(3, 9) == 108
    assert recurrence.node_updates(3, 0) == 0
    for regions, reads in ((True, 1), (9, 1), (1, True), (1, 39), (0, 1)):
        with pytest.raises(ValueError):
            recurrence.node_updates(regions, reads)
    model = SharedEvidenceBridge(configuration(steps=4, budget=11), dtype=torch.float64)
    with pytest.raises(ValueError, match="presupuesto"):
        read(model, torch.ones((1, 2), dtype=torch.float64), snapshot())


def test_global_work_bound_includes_collection_and_emission():
    module = api()
    recurrence = module.BridgeRecurrenceConfig(4, 1216)
    assert recurrence.node_updates(8, 38) == 1216
    assert module.BridgeRecurrenceConfig(2, 228).node_updates(3, 38) == 228
    with pytest.raises(ValueError):
        recurrence.node_updates(8, 39)
    with pytest.raises(ValueError):
        module.BridgeRecurrenceConfig(4, 1217)


@pytest.mark.parametrize("device", DEVICES)
def test_recurrence_gradcheck_and_row_isolation(device):
    module = api()
    local, graph, weights, available = operands(device)
    local.requires_grad_()
    graph.requires_grad_()
    weights.requires_grad_()

    def function(state, edges, value):
        return module.nonlinear_mix(state, edges, value, available, steps=2, gain=0.35)

    gradients_match = torch.autograd.gradcheck(
        function, (local, graph, weights), eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=False
    )
    assert gradients_match, (
        "El Jacobiano de estados, grafo y pesos debe coincidir con diferencias finitas"
    )
    expected = function(local, graph, weights)
    permuted = module.nonlinear_mix(
        local.flip(0), graph.flip(0), weights, available.flip(0), steps=2, gain=0.35
    )
    torch.testing.assert_close(expected, permuted.flip(0), atol=0, rtol=0)
    single = module.nonlinear_mix(local[:1], graph[:1], weights, available[:1], steps=2, gain=0.35)
    torch.testing.assert_close(expected[:1], single, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bad", ["nan", "inf", "overflow", "mask", "steps", "gain"])
def test_invalid_inputs_and_overflow_are_not_hidden_by_masks(device, bad):
    module = api()
    local, graph, weights, available = operands(device)
    options = dict(steps=2, gain=0.25)
    if bad in ("nan", "inf"):
        local[1, 1, 0] = float(bad)
    if bad == "overflow":
        local.fill_(2)
        weights.fill_(torch.finfo(torch.float64).max)
    if bad == "mask":
        available = available.double()
    if bad == "steps":
        options["steps"] = True
    if bad == "gain":
        options["gain"] = float("nan")
    with pytest.raises(ValueError):
        module.nonlinear_mix(local, graph, weights, available, **options)


@pytest.mark.parametrize(
    "case",
    [
        "local_type",
        "local_rank",
        "local_layout",
        "local_dtype",
        "local_device",
        "empty",
        "regions",
        "width",
        "graph_type",
        "graph_shape",
        "graph_dtype",
        "graph_device",
        "weight_shape",
        "weight_layout",
        "mask_shape",
        "mask_layout",
        "mask_device",
        "gain_bool",
        "gain_range",
        "autocast",
        "recurrence_type",
    ],
)
def test_invalid_geometry_and_precision_fail_explicitly(case):
    module = api()
    local, graph, weights, available = operands("cpu")
    options = dict(steps=1, gain=0.25)
    if case == "local_type":
        local = None
    elif case == "local_rank":
        local = local[0]
    elif case == "local_layout":
        local = local.to_sparse()
    elif case == "local_dtype":
        local = local.half()
    elif case == "local_device":
        local = torch.empty((2, 2, 2), dtype=torch.float64, device="meta")
    elif case == "empty":
        local = local[:0]
    elif case == "regions":
        local = torch.zeros((2, 9, 2), dtype=torch.float64)
    elif case == "width":
        local = torch.zeros((2, 2, 129), dtype=torch.float64)
    elif case == "graph_type":
        graph = None
    elif case == "graph_shape":
        graph = graph[:1]
    elif case == "graph_dtype":
        graph = graph.float()
    elif case == "graph_device":
        graph = torch.empty(graph.shape, dtype=graph.dtype, device="meta")
    elif case == "weight_shape":
        weights = weights[:1]
    elif case == "weight_layout":
        weights = weights.to_sparse()
    elif case == "mask_shape":
        available = available[:1]
    elif case == "mask_layout":
        available = available.to_sparse()
    elif case == "mask_device":
        available = torch.empty(available.shape, dtype=torch.bool, device="meta")
    elif case == "gain_bool":
        options["gain"] = True
    elif case == "gain_range":
        options["gain"] = 1.1
    elif case == "recurrence_type":
        with pytest.raises(ValueError):
            EvidenceBridgeConfig(codec_id="a" * 64, cycles=0, recurrence={"steps": 1})
        return
    if case == "autocast":
        with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError):
            module.nonlinear_mix(local, graph, weights, available, **options)
    else:
        with pytest.raises(ValueError):
            module.nonlinear_mix(local, graph, weights, available, **options)


@pytest.mark.parametrize("device", DEVICES)
def test_active_query_gradcheck_and_parameter_gradients(device):
    model = SharedEvidenceBridge(configuration(), dtype=torch.float64, device=device)
    source = snapshot(device)
    state = torch.tensor([[0.37, -0.83]], dtype=torch.float64, device=device, requires_grad=True)
    positions = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)

    def smooth(query):
        return read(model, query, source, positions=positions).values

    assert torch.autograd.gradcheck(smooth, (state,), eps=1e-6, atol=1e-6, rtol=1e-4)
    smooth(state).square().sum().backward()
    assert torch.isfinite(state.grad).all()
    for value in model.parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_bounded_states_do_not_imply_a_contractive_four_step_derivative(device):
    module = api()
    initial = torch.zeros((1, 1, 1), dtype=torch.float64, device=device, requires_grad=True)
    result = module.nonlinear_mix(
        initial,
        torch.ones_like(initial),
        2 * torch.ones_like(initial),
        torch.ones((1, 1), dtype=torch.bool, device=device),
        steps=4,
        gain=1.0,
    )
    derivative = torch.autograd.grad(result.sum(), initial)[0]
    assert torch.equal(result, torch.zeros_like(result))
    assert (result.abs() <= 1).all()
    assert torch.equal(derivative, torch.full_like(initial, 16))


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype,tolerance",
    [(torch.float64, dict(atol=1e-12, rtol=1e-12)), (torch.float32, dict(atol=2e-6, rtol=1e-5))],
)
def test_cuda_matches_cpu_for_active_path(dtype, tolerance):
    cpu = SharedEvidenceBridge(configuration(steps=4), dtype=dtype)
    gpu = SharedEvidenceBridge(cpu.config, dtype=dtype, device="cuda:0")
    gpu.load_state_dict(cpu.state_dict())
    state = torch.tensor([[0.2, -0.7], [0.4, 0.8]], dtype=dtype)
    source = snapshot()
    from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot

    tensors = source.export_cpu()["tensors"]
    tensors["keys"] = tensors["keys"].float()
    tensors["values"] = tensors["values"].float()
    left_source = EpisodeSnapshot.create(
        **tensors, codec_id="a" * 64, context_id="b" * 64, cutoff=4, dtype=dtype, device="cpu"
    )
    right_source = EpisodeSnapshot.restore(
        left_source.export_cpu(),
        codec_id="a" * 64,
        context_id="b" * 64,
        cutoff=4,
        dtype=dtype,
        device="cuda:0",
    )
    left = read(cpu, state, left_source)
    right = read(gpu, state.to("cuda:0"), right_source)
    assert torch.equal(left.evidence_ids, right.evidence_ids.cpu())
    for name in ("values", "region_values", "evidence_weights", "bridge_weights"):
        torch.testing.assert_close(getattr(left, name), getattr(right, name).cpu(), **tolerance)
