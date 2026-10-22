"""Contrato y referencias independientes de los ocho estados anteriores a MAC."""

import copy
import importlib
import importlib.util
import itertools
import json
from dataclasses import FrozenInstanceError

import pytest
import torch

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))
DTYPES = (torch.float32, torch.float64)
GROUPS = ((0, 2), (1, 3, 4))
PERMUTED = ((0, 1), (2, 3, 4))
PROGRAM = "hierarchical_group_root_v1"
ARMS = (
    pytest.param({}, id="tree"),
    pytest.param({"topology": "star"}, id="star"),
    pytest.param({"groups": PERMUTED}, id="permuted_tree"),
    pytest.param({"feedback": False}, id="no_feedback"),
    pytest.param({"topology": "flat_control"}, id="flat_control"),
    pytest.param({"readout": "common_skip"}, id="tree_common_skip"),
    pytest.param({"topology": "flat_control", "readout": "common_skip"}, id="flat_common_skip"),
)


def hierarchy_api():
    name = "mars_titan.models.titans.regional_hierarchy"
    assert importlib.util.find_spec(name) is not None, (
        "Falta el componente regional de ocho estados"
    )
    module = importlib.import_module(name)
    for symbol in (
        "RegionalHierarchyConfig",
        "RegionalHierarchy",
        "RegionalHierarchyResult",
        "validate_hierarchy_operator_config",
        "verify_hierarchy_traces",
        "logical_workspace_bytes",
    ):
        assert hasattr(module, symbol), f"Falta la API regional {symbol}"
    return module


def make(device="cpu", dtype=torch.float64, *, hidden=3, max_batch=256, arm=None, **local):
    api = hierarchy_api()
    fusion = importlib.import_module("mars_titan.models.titans.regional_fusion")
    options = dict(topology="self", routing="uniform", cycles=2, local_steps=2, summary=True)
    options.update(local)
    operator = (
        fusion.RegionalFusion(
            hidden, fusion.RegionalFusionConfig(**options), seed=42, max_batch=max_batch
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    config = api.RegionalHierarchyConfig(**(arm or {}))
    run = api.RegionalHierarchy(hidden, config, max_batch=max_batch).to(device=device)
    return api, operator, run, config


def data(device="cpu", dtype=torch.float64, *, rows=3, hidden=3, all_masks=False):
    grid = torch.arange(rows * 5 * hidden, device=device, dtype=dtype).reshape(rows, 5, hidden)
    blocks = torch.sin(grid * 0.41) + torch.cos(grid * 0.17) * 0.2
    context = (
        torch.cos(torch.arange(rows * hidden, device=device, dtype=dtype)).reshape(rows, hidden)
        * 0.13
    )
    if all_masks:
        assert rows == 32
        masks = list(itertools.product((False, True), repeat=5))
    else:
        masks = [[True, True, False, True, True], [False] * 5, [True] * 5][:rows]
    presence = torch.tensor(masks, device=device, dtype=torch.bool)
    return blocks, presence, context


def close(actual, expected, *, gradient=False):
    if expected.dtype == torch.float64:
        atol, rtol = (2e-10, 2e-9) if gradient else (5e-12, 5e-12)
    else:
        atol, rtol = (1e-4, 1e-3) if gradient else (8e-6, 8e-6)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def neighbours(topology="tree", groups=GROUPS, feedback=True):
    if topology == "tree":
        sources = [
            {i, 5 + next(g for g, members in enumerate(groups) if i in members)} for i in range(5)
        ]
        sources += [{5 + g, 7, *members} for g, members in enumerate(groups)]
        sources += [{5, 6, 7}]
    elif topology == "star":
        sources = [{i, 7} for i in range(7)] + [set(range(8))]
    else:
        assert topology == "flat_control"
        sources = [{i, 7} for i in range(5)]
        sources += [{5 + g, *members} for g, members in enumerate(groups)]
        sources += [set(range(5))]
    if not feedback:
        sources[:5] = [{i} for i in range(5)]
    return [sorted(value) for value in sources]


def expected_availability(presence, groups=GROUPS):
    return torch.stack(
        [
            *(presence[:, i] for i in range(5)),
            *(presence[:, list(group)].any(1) for group in groups),
            presence.any(1),
        ],
        dim=1,
    )


def normalize(value, layer):
    centered = value - value.mean(-1, keepdim=True)
    return (
        centered / torch.sqrt(centered.square().mean(-1, keepdim=True) + layer.eps) * layer.weight
        + layer.bias
    )


def reference(
    operator,
    blocks,
    presence,
    context,
    *,
    topology="tree",
    groups=GROUPS,
    feedback=True,
    readout="local_only",
):
    """Desenrollar nodos y fases sin usar forwards ni geometría del componente nuevo."""
    if presence is None:
        presence = torch.ones(blocks.shape[:2], dtype=torch.bool, device=blocks.device)
    availability = expected_availability(presence, groups)
    senders = neighbours(topology, groups, feedback)
    initial = torch.where(presence[..., None], blocks, torch.zeros_like(blocks))
    state = torch.cat((initial, blocks.new_zeros(len(blocks), 3, blocks.shape[-1])), dim=1)
    local_inputs = []

    def message(old, receiver):
        terms = [
            torch.where(availability[:, j, None], old[:, j], torch.zeros_like(old[:, j]))
            for j in senders[receiver]
        ]
        denominator = availability[:, senders[receiver]].sum(1).clamp_min(1).to(blocks.dtype)
        value = torch.stack(terms, dim=1).sum(1) / denominator[:, None]
        return torch.where(availability[:, receiver, None], value, torch.zeros_like(value))

    for _ in range(2):
        for receivers in ((5, 6), (7,), (5, 6)):
            old, changes = state, {}
            for receiver in receivers:
                argument = normalize(old[:, receiver], operator.summary_norm) + message(
                    old, receiver
                )
                first = argument @ operator.summary_update[0].weight.T
                delta = (first * first.sigmoid()) @ operator.summary_update[2].weight.T
                updated = old[:, receiver] + operator.config.summary_gain * delta
                changes[receiver] = torch.where(
                    availability[:, receiver, None], updated, torch.zeros_like(updated)
                )
            state = torch.stack(
                [changes[i] if i in changes else old[:, i] for i in range(8)], dim=1
            )
        for _ in range(2):
            messages = torch.stack([message(state, i) for i in range(5)], dim=1)
            argument = (
                state[:, :5] + context[:, None] + messages @ operator.summary_to_local.weight.T
            )
            local_inputs.append(argument)
            normed = normalize(argument, operator.local_norm)
            normed = torch.where(presence[..., None], normed, torch.zeros_like(normed))
            value = (normed @ operator.qkv.weight.T)[..., 2 * blocks.shape[-1] :]
            updated = state[:, :5] + operator.config.local_gain * (
                value @ operator.local_out.weight.T
            )
            local = torch.where(presence[..., None], updated, torch.zeros_like(updated))
            state = torch.cat((local, state[:, 5:]), dim=1)
    difference = state[:, :5] - initial
    if readout == "common_skip":
        difference = (
            difference + (context + state[:, 7] @ operator.summary_to_local.weight.T)[:, None]
        )
    output = blocks + difference @ operator.readout.weight.T
    return torch.where(presence[..., None], output, blocks), state, local_inputs


def geometry_buffer(run):
    matches = [
        (name, value)
        for name, value in run.named_buffers()
        if value.dtype == torch.bool and value.shape == (8, 8)
    ]
    assert len(matches) == 1, "La familia necesita una geometría booleana registrada de ocho nodos"
    return matches[0]


def assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(actual[name], value), name
        else:
            assert actual[name] == value, name


def phase_records():
    result = []
    for round_index in (0, 1):
        for phase, receivers in (
            ("group", (5, 6)),
            ("root", (7,)),
            ("group", (5, 6)),
            ("local", (0, 1, 2, 3, 4)),
            ("local", (0, 1, 2, 3, 4)),
        ):
            version = len(result)
            result.append(
                dict(
                    round=round_index,
                    phase=phase,
                    receivers=receivers,
                    read_version=version,
                    write_version=version + 1,
                )
            )
    return result


def check_trace(result, presence, context, config):
    trace = result.trace
    assert set(trace) == {
        "configuration_id",
        "programs",
        "context_norms",
        "phases",
        "availability",
        "work",
    }
    assert len(trace["configuration_id"]) == 64
    assert tuple(trace["programs"]) == (PROGRAM,) * len(context)
    assert tuple(trace["phases"]) == (PROGRAM,)
    observed = trace["phases"][PROGRAM]
    assert len(observed) == 10
    for actual, expected in zip(observed, phase_records(), strict=True):
        assert set(actual) == set(expected)
        assert dict(actual, receivers=tuple(actual["receivers"])) == expected
        assert all(type(actual[key]) is int for key in ("round", "read_version", "write_version"))
    assert trace["availability"].dtype == torch.bool
    assert torch.equal(trace["availability"], expected_availability(presence, config.groups))
    norms = trace["context_norms"]
    assert norms.shape == (len(context),) and norms.dtype == context.dtype
    assert not norms.requires_grad and norms.grad_fn is None
    close(norms, torch.linalg.vector_norm(context.detach(), dim=-1))
    assert trace["work"] == dict(
        local_node_evaluations=20 * len(context), summary_evaluations=10 * len(context)
    )
    assert all(type(value) is int for value in trace["work"].values())


@pytest.mark.parametrize("arm", ARMS)
def test_config_roundtrips_json_and_keeps_the_old_family_distinct(arm):
    api = hierarchy_api()
    config = api.RegionalHierarchyConfig(**arm)
    identity = config.identity()
    assert identity["kind"] == "pre_mac_regional_hierarchy_v1"
    restored = api.RegionalHierarchyConfig.restore(json.loads(json.dumps(identity)))
    assert restored == config
    assert json.loads(json.dumps(restored.identity())) == json.loads(json.dumps(identity))
    raw_groups = [list(group) for group in config.groups]
    from_lists = api.RegionalHierarchyConfig(**dict(arm, groups=raw_groups))
    assert from_lists.groups == config.groups and type(from_lists.groups) is tuple
    assert all(type(group) is tuple for group in from_lists.groups)
    raw_groups[0][0] = 99
    assert from_lists.groups == config.groups
    with pytest.raises(FrozenInstanceError):
        config.feedback = False
    old = importlib.import_module("mars_titan.models.titans.regional_execution")
    assert set(old.RegionalExecutionConfig.__dataclass_fields__) == {
        "selection",
        "program",
        "context_threshold",
    }
    assert old.RegionalExecutionConfig().identity()["kind"] == "pre_mac_regional_program_v1"


@pytest.mark.parametrize(
    "options",
    [
        {"topology": "ring"},
        {"topology": True},
        {"readout": "unknown"},
        {"readout": False},
        {"feedback": 1},
        {"feedback": None},
        {"groups": ((0, 2), (1, 3))},
        {"groups": ((0, 2), (1, 3, 3))},
        {"groups": ((0, 2), (1, 3, 5))},
        {"groups": ((False, 2), (1, 3, 4))},
        {"groups": ((0.0, 2), (1, 3, 4))},
        {"groups": ((0, 1, 2), (3, 4))},
        {"groups": ((0, 3), (1, 2, 4))},
    ],
)
def test_unregistered_or_loosely_typed_configs_are_rejected(options):
    with pytest.raises(ValueError):
        hierarchy_api().RegionalHierarchyConfig(**options)


def test_restore_requires_the_complete_exact_family_identity():
    api = hierarchy_api()
    valid = api.RegionalHierarchyConfig().identity()
    old = importlib.import_module("mars_titan.models.titans.regional_execution")
    for changed in (
        {},
        old.RegionalExecutionConfig().identity(),
        dict(valid, unknown=True),
        dict(valid, kind="other"),
        dict(valid, feedback=1),
    ):
        with pytest.raises(ValueError):
            api.RegionalHierarchyConfig.restore(changed)


@pytest.mark.parametrize(
    "local",
    [
        dict(topology="ring"),
        dict(routing="attention"),
        dict(cycles=1),
        dict(local_steps=1),
        dict(summary=False),
    ],
)
def test_incompatible_local_operator_is_rejected_before_neural_work(local):
    api = hierarchy_api()
    fusion = importlib.import_module("mars_titan.models.titans.regional_fusion")
    values = dict(topology="self", routing="uniform", cycles=2, local_steps=2, summary=True)
    values.update(local)
    config = fusion.RegionalFusionConfig(**values)
    with pytest.raises(ValueError):
        api.validate_hierarchy_operator_config(config)
    _, operator, run, _ = make(**local)
    blocks, presence, context = data()
    calls = []
    handle = operator.qkv.register_forward_pre_hook(lambda module, args: calls.append(True))
    try:
        with pytest.raises(ValueError):
            run(operator, blocks, presence, context)
    finally:
        handle.remove()
    assert calls == []


@pytest.mark.parametrize(
    "hidden,max_batch", [(0, 8), (True, 8), (513, 8), (3, 0), (3, True), (3, 4097)]
)
def test_dimensions_and_physical_batch_are_explicit_and_bounded(hidden, max_batch):
    api = hierarchy_api()
    with pytest.raises(ValueError):
        api.RegionalHierarchy(hidden, api.RegionalHierarchyConfig(), max_batch=max_batch)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("arm", ARMS[:5])
def test_geometry_is_derived_for_eight_nodes_and_feedback_rows(device, arm):
    _, _, run, config = make(device, arm=arm)
    name, actual = geometry_buffer(run)
    sources = neighbours(config.topology, config.groups, config.feedback)
    expected = torch.tensor(
        [[j in sources[i] for j in range(8)] for i in range(8)], dtype=torch.bool, device=device
    )
    assert torch.equal(actual, expected)
    assert int(actual.sum()) == (22 if config.feedback else 17)
    assert name in run.state_dict()
    extra = run.get_extra_state()
    assert set(extra) == {"configuration", "hidden_size", "max_batch"}
    assert extra == dict(configuration=config.identity(), hidden_size=3, max_batch=256)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("arm", ARMS)
def test_all_32_masks_match_independent_unrolling_and_real_working_state(device, dtype, arm):
    _, operator, run, config = make(device, dtype, arm=arm)
    blocks, presence, context = data(device, dtype, rows=32, all_masks=True)
    actual = run(operator, blocks, presence, context)
    _, cpu_operator, _, _ = make("cpu", dtype, arm=arm)
    with torch.no_grad():
        expected, state, _ = reference(
            cpu_operator, blocks.cpu(), presence.cpu(), context.cpu(), **arm
        )
    close(actual.blocks.detach().cpu(), expected)
    close(actual.working_state.detach().cpu(), state)
    assert actual.working_state.shape == (32, 8, 3)
    assert actual.working_state.device == blocks.device and actual.working_state.dtype == dtype
    assert torch.equal(actual.blocks[~presence], blocks[~presence])
    assert torch.equal(
        actual.working_state[~actual.trace["availability"]],
        torch.zeros_like(actual.working_state[~actual.trace["availability"]]),
    )
    assert torch.equal(actual.working_state[0], torch.zeros_like(actual.working_state[0]))
    check_trace(actual, presence, context, config)


@pytest.mark.parametrize("device", DEVICES)
def test_recorded_work_matches_observed_mlp_and_local_evaluations(device):
    _, operator, run, config = make(device)
    blocks, presence, context = data(device)
    calls = []
    hidden, rows = blocks.shape[-1], len(blocks)

    def summary(module, args, output):
        calls.extend(["summary"] * (output.numel() // (rows * hidden)))

    def local(module, args, output):
        calls.extend(["local"] * (output.numel() // (rows * hidden)))

    handles = [
        operator.summary_update.register_forward_hook(summary),
        operator.local_norm.register_forward_hook(local),
    ]
    try:
        result = run(operator, blocks, presence, context)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == (["summary"] * 5 + ["local"] * 10) * 2
    check_trace(result, presence, context, config)


@pytest.mark.parametrize("device", DEVICES)
def test_supplied_context_is_reused_without_accumulating_it_across_local_phases(device):
    _, operator, run, _ = make(device)
    blocks, presence, context = data(device)
    before = context.clone()
    observed = []
    handle = operator.local_norm.register_forward_pre_hook(
        lambda module, args: observed.append(args[0].detach().clone())
    )
    try:
        actual = run(operator, blocks, presence, context)
    finally:
        handle.remove()
    _, _, expected_inputs = reference(operator, blocks, presence, context)
    assert len(observed) == len(expected_inputs) == 4
    for value, expected in zip(observed, expected_inputs, strict=True):
        close(value, expected)
    assert torch.equal(context, before)
    assert not torch.equal(
        run(operator, blocks, presence, torch.zeros_like(context)).blocks, actual.blocks
    )


@pytest.mark.parametrize("device", DEVICES)
def test_readout_policies_have_the_explicit_common_skip_with_zero_local_gain(device):
    _, operator, local_run, _ = make(device, hidden=2, local_gain=0.0)
    _, _, skip_run, _ = make(device, hidden=2, local_gain=0.0, arm=dict(readout="common_skip"))
    blocks, presence, context = data(device, rows=1, hidden=2)
    local = local_run(operator, blocks, presence, context)
    common = skip_run(operator, blocks, presence, context)
    assert torch.equal(local.blocks, blocks)
    close(local.working_state, common.working_state)
    delta = context + common.working_state[:, 7] @ operator.summary_to_local.weight.T
    expected = blocks + (delta @ operator.readout.weight.T)[:, None]
    close(common.blocks, torch.where(presence[..., None], expected, blocks))


def transmission_operator(operator):
    hidden = operator.hidden_size
    identity = torch.eye(hidden, device=operator.qkv.weight.device, dtype=operator.qkv.weight.dtype)
    with torch.no_grad():
        for parameter in operator.parameters():
            parameter.zero_()
        operator.local_norm.weight.fill_(1)
        operator.summary_norm.weight.fill_(1)
        operator.qkv.weight[2 * hidden :].copy_(identity)
        operator.local_out.weight.copy_(identity)
        operator.summary_to_local.weight[0, 0] = 1
        operator.summary_update[0].weight.copy_(torch.cat((identity, -identity)))
        operator.summary_update[2].weight.copy_(torch.cat((identity, -identity), dim=1))
        operator.readout.weight[1, 1] = 1


@pytest.mark.parametrize("device", DEVICES)
def test_cross_group_effect_disappears_without_feedback_with_context_clamped(device):
    _, operator, connected, _ = make(device, hidden=2)
    _, _, cut, _ = make(device, hidden=2, arm=dict(feedback=False))
    transmission_operator(operator)
    blocks = torch.zeros(1, 5, 2, device=device, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        blocks[0, 0, 0] = 1
    presence = torch.ones(1, 5, device=device, dtype=torch.bool)
    context = blocks.new_zeros(1, 2)
    first, disconnected = (
        connected(operator, blocks, presence, context),
        cut(operator, blocks, presence, context),
    )
    assert first.blocks[0, 1, 1] < -1e-3
    assert disconnected.blocks[0, 1, 1] == 0
    (derivative,) = torch.autograd.grad(disconnected.blocks[0, 1, 1], blocks, retain_graph=True)
    assert torch.equal(derivative[:, 0], torch.zeros_like(derivative[:, 0]))
    parameters = (*operator.summary_norm.parameters(), *operator.summary_update.parameters())
    gradients = torch.autograd.grad(
        disconnected.blocks.square().sum(), parameters, allow_unused=True
    )
    assert all(value is None or torch.equal(value, torch.zeros_like(value)) for value in gradients)
    changed = blocks.detach().clone()
    changed[0, 0, 0] += 2
    assert torch.equal(
        cut(operator, changed, presence, context).blocks[:, 1], disconnected.blocks[:, 1]
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "arm",
    [
        pytest.param({}, id="tree"),
        pytest.param(dict(topology="flat_control", readout="common_skip"), id="flat_skip"),
    ],
)
def test_shared_parameter_input_and_context_gradients_match_independent_unrolling(
    device, dtype, arm
):
    _, operator, run, _ = make(device, dtype, arm=arm)
    blocks, presence, context = data(device, dtype, rows=1)
    blocks, context = blocks.requires_grad_(), context.requires_grad_()
    variables = (*operator.parameters(), blocks, context)
    probe = torch.cos(torch.arange(15, device=device, dtype=dtype)).reshape(1, 5, 3)
    state_probe = torch.sin(torch.arange(24, device=device, dtype=dtype)).reshape(1, 8, 3)
    actual = run(operator, blocks, presence, context)
    assert actual.working_state.requires_grad
    gradients = torch.autograd.grad(
        (actual.blocks * probe).sum() + 0.1 * (actual.working_state * state_probe).sum(),
        variables,
        allow_unused=True,
    )
    expected, state, _ = reference(operator, blocks, presence, context, **arm)
    reference_gradients = torch.autograd.grad(
        (expected * probe).sum() + 0.1 * (state * state_probe).sum(), variables, allow_unused=True
    )
    for variable, left, right in zip(variables, gradients, reference_gradients, strict=True):
        close(
            torch.zeros_like(variable) if left is None else left,
            torch.zeros_like(variable) if right is None else right,
            gradient=True,
        )
    assert all(parameter.grad is None for parameter in operator.parameters())


@pytest.mark.parametrize("device", DEVICES)
def test_small_fp64_gradcheck_covers_blocks_context_and_auxiliary_working_state(device):
    _, operator, run, _ = make(device, hidden=2)
    operator.requires_grad_(False)
    blocks, presence, context = data(device, rows=1, hidden=2)

    def output(x, c):
        result = run(operator, x, presence, c)
        return result.blocks, result.working_state

    with torch.random.fork_rng(devices=[0] if device == "cuda:0" else []):
        assert torch.autograd.gradcheck(
            output,
            (blocks.requires_grad_(), context.requires_grad_()),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-3,
            fast_mode=True,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_directional_derivatives_match_finite_differences_in_each_precision(device, dtype):
    _, operator, run, _ = make(device, dtype)
    operator.requires_grad_(False)
    blocks, presence, context = data(device, dtype, rows=1)
    blocks, context = blocks.requires_grad_(), context.requires_grad_()
    probe = torch.cos(torch.arange(15, device=device, dtype=dtype)).reshape(1, 5, 3)

    def loss(x, c):
        return (run(operator, x, presence, c).blocks * probe).sum()

    analytic = torch.autograd.grad(loss(blocks, context), (blocks, context))
    epsilon = 1e-6 if dtype == torch.float64 else 2e-3
    tolerance = 2e-5 if dtype == torch.float64 else 2e-2
    for target, index in ((0, (0, 0, 0)), (1, (0, 1))):
        plus, minus = (
            [blocks.detach().clone(), context.detach().clone()],
            [blocks.detach().clone(), context.detach().clone()],
        )
        plus[target][index] += epsilon
        minus[target][index] -= epsilon
        numeric = (loss(*plus) - loss(*minus)) / (2 * epsilon)
        torch.testing.assert_close(analytic[target][index], numeric, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device", DEVICES)
def test_constructor_forward_and_repeated_calls_preserve_rng_inputs_weights_and_owner(device):
    api = hierarchy_api()
    _, operator, _, _ = make(device)
    blocks, presence, context = data(device)
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state(0).clone() if device == "cuda:0" else None
    run = api.RegionalHierarchy(3, api.RegionalHierarchyConfig(), max_batch=256).to(device)
    weights, own = copy.deepcopy(operator.state_dict()), copy.deepcopy(run.state_dict())
    saved = [value.clone() for value in (blocks, presence, context)]
    result = run(operator, blocks, presence, context)
    output, working = result.blocks.clone(), result.working_state.clone()
    run(operator, blocks * 1.3, ~presence, context + 0.4)
    assert torch.equal(result.blocks, output) and torch.equal(result.working_state, working)
    with torch.no_grad():
        result.working_state.zero_()
    repeated = run(operator, blocks, presence, context)
    assert torch.equal(repeated.blocks, output)
    assert torch.equal(repeated.working_state, working)
    assert list(run.parameters()) == []
    assert_state_equal(operator.state_dict(), weights)
    assert_state_equal(run.state_dict(), own)
    assert all(
        torch.equal(value, snapshot)
        for value, snapshot in zip((blocks, presence, context), saved, strict=True)
    )
    assert torch.equal(before_cpu, torch.get_rng_state())
    if before_cuda is not None:
        assert torch.equal(before_cuda, torch.cuda.get_rng_state(0))


@pytest.mark.parametrize("device", DEVICES)
def test_batch_permutation_partition_and_unrelated_rows_do_not_share_auxiliaries(device):
    _, operator, run, _ = make(device)
    blocks, presence, context = data(device)
    expected = run(operator, blocks, presence, context)
    order = torch.tensor([2, 0, 1], device=device)
    permuted = run(operator, blocks[order], presence[order], context[order])
    close(permuted.blocks, expected.blocks[order])
    close(permuted.working_state, expected.working_state[order])
    pieces = [run(operator, blocks[a:b], presence[a:b], context[a:b]) for a, b in ((0, 1), (1, 3))]
    close(torch.cat([piece.blocks for piece in pieces]), expected.blocks)
    close(torch.cat([piece.working_state for piece in pieces]), expected.working_state)
    assert sum(piece.trace["work"]["summary_evaluations"] for piece in pieces) == 30
    changed = blocks.clone()
    changed[2] += 3
    actual = run(operator, changed, presence, context)
    assert torch.equal(actual.blocks[:2], expected.blocks[:2])
    assert torch.equal(actual.working_state[:2], expected.working_state[:2])


@pytest.mark.parametrize("device", DEVICES)
def test_none_presence_means_all_five_sources_present(device):
    _, operator, run, config = make(device)
    blocks, _, context = data(device)
    complete = torch.ones(blocks.shape[:2], device=device, dtype=torch.bool)
    first, second = run(operator, blocks, None, context), run(operator, blocks, complete, context)
    assert torch.equal(first.blocks, second.blocks)
    assert torch.equal(first.working_state, second.working_state)
    check_trace(first, complete, context, config)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "fault",
    [
        "blocks_shape",
        "blocks_dtype",
        "blocks_sparse",
        "context_shape",
        "context_dtype",
        "context_sparse",
        "presence_shape",
        "presence_dtype",
        "presence_sparse",
        "empty",
        "oversized",
        "nonfinite_context",
        "nonfinite_absent_input",
    ],
)
def test_invalid_inputs_fail_before_any_neural_phase(device, fault):
    _, operator, run, _ = make(device, max_batch=8)
    blocks, presence, context = data(device)
    if fault == "blocks_shape":
        blocks = blocks[:, :4]
    elif fault == "blocks_dtype":
        blocks = blocks.float()
    elif fault == "blocks_sparse":
        blocks = blocks.to_sparse()
    elif fault == "context_shape":
        context = context[:, :2]
    elif fault == "context_dtype":
        context = context.float()
    elif fault == "context_sparse":
        context = context.to_sparse()
    elif fault == "presence_shape":
        presence = presence[:, :4]
    elif fault == "presence_dtype":
        presence = presence.float()
    elif fault == "presence_sparse":
        presence = presence.to_sparse()
    elif fault == "empty":
        blocks, presence, context = blocks[:0], presence[:0], context[:0]
    elif fault == "oversized":
        blocks, presence, context = (
            blocks[:1].expand(9, -1, -1),
            presence[:1].expand(9, -1),
            context[:1].expand(9, -1),
        )
    elif fault == "nonfinite_context":
        context[0, 0] = float("inf")
    else:
        blocks[1, 0, 0] = float("nan")
    calls = []
    handles = [
        operator.summary_update.register_forward_pre_hook(
            lambda module, args: calls.append("summary")
        ),
        operator.qkv.register_forward_pre_hook(lambda module, args: calls.append("local")),
    ]
    try:
        with pytest.raises(ValueError):
            run(operator, blocks, presence, context)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == []


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("part", ["summary_update", "local_out"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_intermediate_is_not_hidden_by_absent_auxiliaries_or_leaves(device, part, bad):
    _, operator, run, _ = make(device)
    blocks = torch.ones(1, 5, 3, device=device, dtype=torch.float64)
    presence = torch.zeros(1, 5, device=device, dtype=torch.bool)
    context = blocks.new_zeros(1, 3)
    observed = []

    def inject(module, args, output):
        observed.append(True)
        value = output.clone()
        value.reshape(-1)[0] = bad
        return value

    handle = getattr(operator, part).register_forward_hook(inject)
    try:
        with pytest.raises(ValueError, match="NaN|infinito|finito"):
            run(operator, blocks, presence, context)
    finally:
        handle.remove()
    assert observed


@pytest.mark.parametrize("device", DEVICES)
def test_fp32_absent_residual_overflow_is_rejected_before_output_mask(device):
    _, operator, run, _ = make(
        device,
        torch.float32,
        hidden=2,
        arm=dict(readout="common_skip"),
        local_gain=0.0,
        summary_gain=0.0,
    )
    blocks = torch.full((1, 5, 2), 3e38, device=device, dtype=torch.float32)
    presence = torch.zeros((1, 5), device=device, dtype=torch.bool)
    context = blocks.new_tensor([[1.0, 0.0]])
    with torch.no_grad():
        operator.summary_to_local.weight.zero_()
        operator.readout.weight.zero_()
        operator.readout.weight[0, 0] = 3e38
        projected = (context @ operator.readout.weight.T)[:, None].expand_as(blocks)
        before_mask = blocks + projected
        masked = torch.where(presence[..., None], before_mask, blocks)
    assert torch.isfinite(blocks).all() and torch.isfinite(projected).all()
    assert torch.isposinf(before_mask[..., 0]).all()
    assert torch.isfinite(masked).all() and torch.equal(masked, blocks)
    with pytest.raises(ValueError, match="NaN|infinito|finito"):
        run(operator, blocks, presence, context)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("part", ["query", "key"])
@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive_inf"),
        pytest.param(-float("inf"), id="negative_inf"),
    ],
)
def test_nonfinite_q_or_k_is_rejected_before_uniform_zeroing(device, dtype, part, bad):
    _, operator, run, _ = make(device, dtype)
    blocks, _, context = data(device, dtype, rows=1)
    presence = torch.ones((1, 5), device=device, dtype=torch.bool)
    hidden = blocks.shape[-1]
    start = 0 if part == "query" else hidden
    observed = []

    def inject(module, args, output):
        assert torch.isfinite(output).all()
        changed = output.clone()
        changed[..., start : start + hidden] = bad
        assert torch.isfinite(changed[..., 2 * hidden :]).all()
        assert not torch.isfinite(changed[..., start : start + hidden]).all()
        observed.append(True)
        return changed

    handle = operator.qkv.register_forward_hook(inject)
    try:
        with pytest.raises(ValueError, match="NaN|infinito|finito"):
            run(operator, blocks, presence, context)
    finally:
        handle.remove()
    assert observed


def test_autocast_cannot_relax_the_declared_precision():
    _, operator, run, _ = make(dtype=torch.float32)
    blocks, presence, context = data(dtype=torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            run(operator, blocks, presence, context)


@pytest.mark.parametrize(
    "rows,hidden,element_size", [(1, 1, 4), (256, 32, 4), (1024, 32, 8), (4096, 512, 8)]
)
def test_logical_workspace_formula_is_an_estimate_not_a_clamped_allocator_peak(
    rows, hidden, element_size
):
    api = hierarchy_api()
    assert api.logical_workspace_bytes(rows, hidden, element_size) == rows * (
        512 * hidden * element_size + 1024
    )


def test_logical_workspace_limit_rejects_before_mlp_or_local_work():
    api, operator, run, _ = make(hidden=32, max_batch=4096)
    assert api.logical_workspace_bytes(1024, 32, 8) > 128 * 1024**2
    blocks = torch.ones(1024, 5, 32, dtype=torch.float64)
    presence = torch.ones(1024, 5, dtype=torch.bool)
    context = torch.zeros(1024, 32, dtype=torch.float64)
    calls = []
    handles = [
        operator.summary_update.register_forward_pre_hook(
            lambda module, args: calls.append("summary")
        ),
        operator.qkv.register_forward_pre_hook(lambda module, args: calls.append("local")),
    ]
    try:
        with pytest.raises(ValueError):
            run(operator, blocks, presence, context)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == []


@pytest.mark.parametrize("device", DEVICES)
def test_valid_checkpoint_can_repair_installed_boolean_geometry(device):
    _, _, run, _ = make(device)
    payload = copy.deepcopy(run.state_dict())
    _, buffer = geometry_buffer(run)
    with torch.no_grad():
        buffer.logical_not_()
    run.load_state_dict(payload)
    assert_state_equal(run.state_dict(), payload)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fault", ["edges", "dtype", "shape", "sparse", "missing", "extra"])
def test_corrupted_geometry_payload_is_rejected_before_copy(device, fault):
    _, _, run, _ = make(device)
    original = copy.deepcopy(run.state_dict())
    payload = copy.deepcopy(original)
    name, value = geometry_buffer(run)
    if fault == "edges":
        payload[name] = ~payload[name]
    elif fault == "dtype":
        payload[name] = value.float()
    elif fault == "shape":
        payload[name] = value[:7].clone()
    elif fault == "sparse":
        payload[name] = value.to_sparse()
    elif fault == "missing":
        payload.pop(name)
    else:
        payload["unexpected"] = value.clone()
    with pytest.raises(ValueError):
        run.load_state_dict(payload)
    assert_state_equal(run.state_dict(), original)


@pytest.mark.parametrize("device", DEVICES)
def test_metadata_validation_precedes_even_a_repairing_buffer_copy(device):
    _, _, run, _ = make(device)
    payload = copy.deepcopy(run.state_dict())
    payload["_extra_state"]["hidden_size"] += 1
    _, installed = geometry_buffer(run)
    with torch.no_grad():
        installed[0, 0].logical_not_()
    before = copy.deepcopy(run.state_dict())
    with pytest.raises(ValueError):
        run.load_state_dict(payload)
    assert_state_equal(run.state_dict(), before)


def test_payload_validation_is_scoped_to_the_declared_parent_prefix():
    _, _, run, _ = make()
    prefix = "regional_execution."
    payload = {prefix + name: value for name, value in run.state_dict().items()}
    payload["other.weight"] = torch.ones(1)
    run.validate_payload(payload, prefix)
    payload[prefix + "unexpected"] = torch.ones(1)
    with pytest.raises(ValueError):
        run.validate_payload(payload, prefix)


def test_exported_rows_are_reusable_without_constructing_live_modules(monkeypatch):
    api, operator, run, _ = make()
    blocks, presence, context = data()
    result = run(operator, blocks, presence, context)
    rows = run.export_trace(result.trace, operator)
    descriptor = run.trace_descriptor(operator)
    assert set(descriptor) == {"schema_version", "execution", "operator", "dtype"}
    assert descriptor["schema_version"] == 2 and descriptor["dtype"] == "torch.float64"
    expected = dict(local_node_evaluations=60, summary_evaluations=30)
    assert run.verify_exported_traces(rows, operator) == expected
    for row in rows:
        assert set(row) == {
            "configuration_id",
            "operator_id",
            "program",
            "context_norm",
            "phases",
            "availability",
            "work",
        }
        assert row["program"] == PROGRAM
        assert len(row["availability"]) == 8 and all(
            type(value) is bool for value in row["availability"]
        )
        assert row["work"] == dict(local_node_evaluations=20, summary_evaluations=10)

    def forbidden(*args, **kwargs):
        pytest.fail("La verificación archivada no debe instanciar módulos")

    monkeypatch.setattr(api.RegionalHierarchy, "__init__", forbidden)
    assert (
        api.verify_hierarchy_traces(
            json.loads(json.dumps(rows)), json.loads(json.dumps(descriptor))
        )
        == expected
    )


@pytest.mark.parametrize(
    "fault", ["availability", "phase", "receiver", "version", "work", "program", "norm", "extra"]
)
def test_archive_rejects_structural_corruption(fault):
    api, operator, run, _ = make()
    blocks, presence, context = data()
    rows = json.loads(
        json.dumps(run.export_trace(run(operator, blocks, presence, context).trace, operator))
    )
    descriptor = json.loads(json.dumps(run.trace_descriptor(operator)))
    if fault == "availability":
        rows[0]["availability"][7] = False
    elif fault == "phase":
        rows[0]["phases"][0]["phase"] = "local"
    elif fault == "receiver":
        rows[0]["phases"][0]["receivers"] = [7]
    elif fault == "version":
        rows[0]["phases"][1]["read_version"] = 0
    elif fault == "work":
        rows[0]["work"]["summary_evaluations"] = 2
    elif fault == "program":
        rows[0]["program"] = "summary_interleaved_v1"
    elif fault == "norm":
        rows[0]["context_norm"] = -1.0
    else:
        rows[0]["unknown"] = True
    with pytest.raises(ValueError):
        api.verify_hierarchy_traces(rows, descriptor)
