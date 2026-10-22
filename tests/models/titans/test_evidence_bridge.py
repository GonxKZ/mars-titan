"""Una fuente episódica común, vistas regionales y puentes sin almacenar otra historia."""

import importlib
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from mars_titan.models.titans.episodic_readout import _normalize
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def api():
    try:
        return importlib.import_module("mars_titan.models.titans.evidence_bridge")
    except ModuleNotFoundError:
        pytest.fail("Falta el puente de vistas sobre una evidencia común")


def snapshot(device="cpu", count=3):
    return EpisodeSnapshot.create(
        keys=torch.eye(64, dtype=torch.float32)[:count],
        values=torch.arange(count * 64, dtype=torch.float32).reshape(count, 64) / 100,
        labels=torch.arange(count, dtype=torch.float64) / 10,
        ids=torch.arange(1, count + 1, dtype=torch.int64),
        decision_at=torch.full((count,), 2, dtype=torch.int64),
        available_at=torch.full((count,), 1, dtype=torch.int64),
        maturity_at=torch.full((count,), 3, dtype=torch.int64),
        cutoff=4,
        codec_id="a" * 64,
        context_id="b" * 64,
        dtype=torch.float64,
        device=device,
    )


def bridge(device="cpu", **options):
    module = api()
    config = module.EvidenceBridgeConfig(
        **dict(dict(codec_id="a" * 64, hidden_size=2, regions=("one", "two", "three")), **options)
    )
    return module.SharedEvidenceBridge(config, dtype=torch.float64, device=device)


def read(model, state, source, **options):
    return model(state, source, context_id="b" * 64, cutoff=4, **options)


@pytest.mark.parametrize("device", DEVICES)
def test_regions_reference_the_same_episode_without_mutating_its_snapshot(device):
    module = api()
    model = module.SharedEvidenceBridge(
        module.EvidenceBridgeConfig(codec_id="a" * 64, hidden_size=2, regions=("one", "two")),
        dtype=torch.float64,
        device=device,
    )
    source = snapshot(device)
    before = source.export_cpu()
    query = torch.tensor([[0.25, -0.5]], dtype=torch.float64, device=device)
    result = model(query, source, context_id="b" * 64, cutoff=4)
    assert result.values.shape == (1, 2)
    assert result.region_values.shape == (1, 2, 2)
    assert result.evidence_ids.shape == (1, 3)
    assert result.evidence_weights.shape == (1, 2, 3)
    torch.testing.assert_close(
        result.evidence_weights.sum(-1), torch.ones((1, 2), device=device, dtype=torch.float64)
    )
    assert not torch.equal(result.evidence_weights[:, 0], result.evidence_weights[:, 1])
    assert result.snapshot_sha256 == before["sha256"]
    after = source.export_cpu()
    assert after["sha256"] == before["sha256"]
    for name in before["tensors"]:
        assert torch.equal(before["tensors"][name], after["tensors"][name])
    assert all(not isinstance(value, EpisodeSnapshot) for value in vars(model).values())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("topology", ["self", "ring", "complete"])
def test_batched_views_and_bridges_match_an_explicit_equation_reference(device, topology):
    model = bridge(device, topology=topology, cycles=2)
    source = snapshot(device)
    state = torch.tensor([[0.25, -0.5], [-0.1, 0.4]], dtype=torch.float64, device=device)
    available = torch.tensor([[True, True, True], [True, False, True]], device=device)
    result = read(model, state, source, available=available)
    keys, values, labels, ids = source._values[:4]
    for row in range(2):
        raw = model.query(state[row])
        scores = (keys @ _normalize(raw)).detach()
        positions = sorted(range(source.count), key=lambda at: (-float(scores[at]), int(ids[at])))
        assert result.evidence_ids[row].tolist() == ids[positions].tolist()
        pool = model.value(torch.cat((values[positions], labels[positions, None]), -1))
        local = []
        for region in range(3):
            query = _normalize(raw + state[row] @ model.region_query[region])
            weights = torch.softmax(keys[positions] @ query / model.config.temperature, 0)
            if not available[row, region]:
                weights = weights * 0
            torch.testing.assert_close(
                result.evidence_weights[row, region], weights, rtol=1e-12, atol=1e-12
            )
            local.append(
                torch.tanh(((weights[:, None] * pool).sum(0)) @ model.region_value[region])
            )
        local = torch.stack(local)
        expected_graph = state.new_zeros((3, 3))
        for target in range(3):
            senders = [
                sender
                for sender in range(3)
                if available[row, target]
                and available[row, sender]
                and (
                    topology == "complete"
                    or sender == target
                    or (topology == "ring" and sender == (target + 1) % 3)
                )
            ]
            for sender in senders:
                expected_graph[target, sender] = 1 / len(senders)
        for _ in range(2):
            local = torch.stack(
                [
                    (1 - model.config.bridge_gain) * local[target]
                    + model.config.bridge_gain
                    * sum(
                        (expected_graph[target, sender] * local[sender] for sender in range(3)),
                        start=torch.zeros_like(local[target]),
                    )
                    for target in range(3)
                ]
            )
        torch.testing.assert_close(result.bridge_weights[row], expected_graph, rtol=0, atol=0)
        torch.testing.assert_close(result.region_values[row], local, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            result.values[row], model.output(local.flatten()), rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize("device", DEVICES)
def test_a_single_fact_has_distinct_views_and_bridges_change_the_output(device):
    connected, isolated = (
        bridge(device, bridge_gain=1.0, topology="complete"),
        bridge(device, cycles=0),
    )
    with torch.no_grad():
        isolated.region_query.copy_(connected.region_query)
        isolated.region_value.copy_(connected.region_value)
        for left, right in (
            (isolated.query, connected.query),
            (isolated.value, connected.value),
            (isolated.output, connected.output),
        ):
            left.load_state_dict(right.state_dict())
    source = snapshot(device, count=1)
    state = torch.tensor([[1.0, 0.5]], dtype=torch.float64, device=device)
    local, shared = read(isolated, state, source), read(connected, state, source)
    assert local.evidence_ids.tolist() == shared.evidence_ids.tolist() == [[1]]
    assert not torch.equal(local.region_values[:, 0], local.region_values[:, 1])
    torch.testing.assert_close(
        shared.region_values,
        local.region_values.mean(1, keepdim=True).expand_as(local.region_values),
    )
    assert not torch.allclose(shared.values, local.values, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_shared_value_projection_runs_once_without_a_regional_axis(device):
    model, source = bridge(device), snapshot(device)
    inputs = []
    handle = model.value.register_forward_pre_hook(
        lambda _module, args: inputs.append(args[0].shape)
    )
    try:
        read(model, torch.ones((2, 2), dtype=torch.float64, device=device), source)
    finally:
        handle.remove()
    assert inputs == [torch.Size((6, 65))]


@pytest.mark.parametrize("device", DEVICES)
def test_masks_empty_memory_gradients_and_row_permutation(device):
    model, source = bridge(device), snapshot(device)
    state = torch.tensor(
        [[0.3, -0.5], [0.9, 0.1]], device=device, dtype=torch.float64, requires_grad=True
    )
    result = read(model, state, source)
    result.values.square().sum().backward()
    assert torch.isfinite(state.grad).all() and state.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    permuted = read(model, state.flip(0), source)
    torch.testing.assert_close(result.values, permuted.values.flip(0), rtol=1e-12, atol=1e-12)
    empty = read(model, state, snapshot(device, count=0))
    assert empty.evidence_ids.shape == (2, 0) and empty.evidence_weights.shape == (2, 3, 0)
    assert torch.count_nonzero(empty.values) == 0
    missing = read(
        model, state, source, available=torch.zeros((2, 3), dtype=torch.bool, device=device)
    )
    assert (
        torch.count_nonzero(missing.values)
        == torch.count_nonzero(missing.evidence_weights)
        == torch.count_nonzero(missing.bridge_weights)
        == 0
    )


@pytest.mark.parametrize("device", DEVICES)
def test_recovery_preserves_the_next_read_and_leaves_rng_unchanged(device):
    before = torch.random.get_rng_state()
    gpu_before = torch.cuda.get_rng_state(0) if device != "cpu" else None
    model = bridge(device)
    assert torch.equal(torch.random.get_rng_state(), before)
    state, source = torch.ones((1, 2), dtype=torch.float64, device=device), snapshot(device)
    original = read(model, state, source)
    restored = bridge(device)
    restored.load_state_dict(deepcopy(model.state_dict()))
    actual = read(restored, state, source)
    for name in ("values", "region_values", "evidence_ids", "evidence_weights", "bridge_weights"):
        assert torch.equal(getattr(original, name), getattr(actual, name))
    assert torch.equal(torch.random.get_rng_state(), before)
    if gpu_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(0), gpu_before)


@pytest.mark.parametrize(
    "case",
    ["context", "cutoff", "dtype", "modified_snapshot", "budget", "nonfinite", "overflow", "mask"],
)
def test_inadmissible_inputs_are_rejected_without_hiding_overflow(case):
    model, source = bridge(), snapshot()
    state = torch.ones((1, 2), dtype=torch.float64)
    options = dict(context_id="b" * 64, cutoff=4)
    if case == "context":
        options["context_id"] = "c" * 64
    if case == "cutoff":
        options["cutoff"] = 5
    if case == "dtype":
        model, state = model.half(), state.half()
    if case == "modified_snapshot":
        source._values[1].add_(1)
    if case == "budget":
        model = bridge(max_working_bytes=1024)
    if case == "nonfinite":
        state[0, 0] = float("nan")
    if case == "overflow":
        with torch.no_grad():
            model.value.weight.zero_()
            model.value.bias.fill_(2)
            model.region_value.fill_(torch.finfo(torch.float64).max)
    if case == "mask":
        options["available"] = torch.ones((1, 3))
    with pytest.raises(ValueError):
        model(state, source, **options)


@pytest.mark.parametrize("case", ["field", "shape", "dtype", "nonfinite", "edges", "identity"])
def test_checkpoint_rejection_precedes_parameter_copy(case):
    model = bridge()
    before = deepcopy(model.state_dict())
    payload = deepcopy(before)
    if case == "field":
        payload.pop("region_query")
    if case == "shape":
        payload["region_query"] = payload["region_query"][0]
    if case == "dtype":
        payload["region_query"] = payload["region_query"].float()
    if case == "nonfinite":
        payload["region_query"].fill_(float("inf"))
    if case == "edges":
        payload["edges"][0, 0] = False
    if case == "identity":
        payload["_extra_state"]["configuration"]["regions"] = ("other",)
    with pytest.raises(ValueError):
        model.load_state_dict(payload)
    for name in before:
        if isinstance(before[name], torch.Tensor):
            assert torch.equal(before[name], model.state_dict()[name])


@pytest.mark.parametrize(
    "options",
    [
        dict(regions=()),
        dict(regions=["one"]),
        dict(regions=("one", "one")),
        dict(regions=("",)),
        dict(regions=("one",) * 9),
        dict(topology="unknown"),
        dict(cycles=5),
        dict(bridge_gain=float("nan")),
        dict(temperature=0),
        dict(hidden_size=True),
        dict(neighbors=9),
    ],
)
def test_configuration_rejects_ambiguous_or_unbounded_views(options):
    module = api()
    with pytest.raises(ValueError):
        replace(module.EvidenceBridgeConfig(codec_id="a" * 64), **options)


@pytest.mark.parametrize("device", DEVICES)
def test_smooth_read_has_a_correct_query_gradient(device):
    model, source = bridge(device), snapshot(device)
    state = torch.tensor([[0.37, -0.83]], dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda query: read(model, query, source).values, (state,), eps=1e-6, atol=1e-6, rtol=1e-4
    )


@pytest.mark.cuda
def test_cuda_matches_cpu_for_the_same_parameters_and_evidence():
    cpu, gpu = bridge(), bridge("cuda:0")
    gpu.load_state_dict(cpu.state_dict())
    state = torch.tensor([[0.2, -0.7], [0.4, 0.8]], dtype=torch.float64)
    left, right = read(cpu, state, snapshot()), read(gpu, state.cuda(0), snapshot("cuda:0"))
    for name in ("values", "region_values", "evidence_weights", "bridge_weights"):
        torch.testing.assert_close(
            getattr(left, name), getattr(right, name).cpu(), rtol=1e-12, atol=1e-12
        )
    assert torch.equal(left.evidence_ids, right.evidence_ids.cpu())


@pytest.mark.parametrize(
    "case",
    [
        "state_shape",
        "state_layout",
        "snapshot",
        "constructor",
        "precision",
        "estimate_batch",
        "estimate_episodes",
        "estimate_flag",
    ],
)
def test_shapes_and_estimate_contracts_fail_before_reading(case):
    module = api()
    model, source = bridge(), snapshot()
    with pytest.raises(ValueError):
        if case == "state_shape":
            read(model, torch.ones((1, 3), dtype=torch.float64), source)
        if case == "state_layout":
            read(model, torch.ones((1, 2), dtype=torch.float64).to_sparse(), source)
        if case == "snapshot":
            read(model, torch.ones((1, 2), dtype=torch.float64), None)
        if case == "constructor":
            module.SharedEvidenceBridge(None)
        if case == "precision":
            module.SharedEvidenceBridge(model.config, dtype=torch.float16)
        if case == "estimate_batch":
            model.estimated_bytes(0, 3)
        if case == "estimate_episodes":
            model.estimated_bytes(1, 1025)
        if case == "estimate_flag":
            model.estimated_bytes(1, 3, differentiable=1)
