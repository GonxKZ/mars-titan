"""Referencias independientes de los programas regionales anteriores a MAC."""

import copy
import importlib
import math

import pytest
import torch

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))
DTYPES = (torch.float32, torch.float64)
AT_END = "summary_at_end_v1"
INTERLEAVED = "summary_interleaved_v1"


def execution_api():
    name = "mars_titan.models.titans.regional_execution"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name == name:
            pytest.fail("Falta el ejecutor regional registrado")
        raise
    assert hasattr(module, "RegionalExecutionConfig"), "Falta la configuración regional"
    assert hasattr(module, "RegionalExecution"), "Falta el ejecutor regional"
    return module


def operator_on(
    device, dtype=torch.float64, *, hidden=3, topology="ring", routing="attention", summary=True
):
    api = importlib.import_module("mars_titan.models.titans.regional_fusion")
    config = api.RegionalFusionConfig(
        topology=topology, routing=routing, summary=summary, cycles=3, local_steps=1
    )
    return (
        api.RegionalFusion(hidden, config, seed=73, max_batch=8)
        .to(device=device, dtype=dtype)
        .eval()
    )


def executor(hidden=3, *, program=AT_END, selection="fixed", threshold=None, max_batch=8):
    api = execution_api()
    config = api.RegionalExecutionConfig(
        selection=selection, program=program, context_threshold=threshold
    )
    return api.RegionalExecution(hidden, config, max_batch=max_batch)


def inputs(device, dtype=torch.float64, *, rows=3, hidden=3):
    grid = torch.arange(rows * 5 * hidden, dtype=dtype, device=device).reshape(rows, 5, hidden)
    blocks = torch.sin(0.73 * grid) + 0.2 * torch.cos(0.31 * grid)
    context = 0.2 * torch.cos(
        0.43 * torch.arange(rows * hidden, dtype=dtype, device=device).reshape(rows, hidden)
    )
    patterns = (
        (True, True, False, True, True),
        (True, False, True, True, False),
        (False, False, False, False, False),
        (True, True, True, True, True),
    )
    presence = torch.tensor(patterns[:rows], dtype=torch.bool, device=device)
    return blocks, presence, context


def tolerance(dtype):
    return (1e-12, 1e-12) if dtype == torch.float64 else (1e-6, 1e-5)


def close(actual, expected):
    atol, rtol = tolerance(expected.dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def normalize(value, layer):
    centered = value - value.mean(-1, keepdim=True)
    scale = torch.sqrt(centered.square().mean(-1, keepdim=True) + layer.eps)
    return centered / scale * layer.weight + layer.bias


def reference(operator, blocks, presence, context, programs):
    """Evaluar filas y receptores sin llamar al forward ni a fases del operador."""
    if presence is None:
        presence = torch.ones(blocks.shape[:2], dtype=torch.bool, device=blocks.device)
    answers = []
    hidden = blocks.shape[-1]
    for row, program in enumerate(programs):
        mask = presence[row].detach().cpu().tolist()
        initial = torch.stack(
            [blocks[row, i] if mask[i] else torch.zeros_like(blocks[row, i]) for i in range(5)]
        )
        local = initial
        summary = context[row].new_zeros(hidden)
        current = context[row]
        cycles, steps = (1, 4) if program == AT_END else (2, 2)
        for _ in range(cycles):
            for _ in range(steps):
                normalized = torch.stack(
                    [
                        normalize(local[i] + current, operator.local_norm)
                        if mask[i]
                        else torch.zeros_like(local[i])
                        for i in range(5)
                    ]
                )
                projected = normalized @ operator.qkv.weight.T
                query, key, value = projected.split(hidden, dim=-1)
                updated = []
                for receiver in range(5):
                    allowed = (
                        [
                            sender
                            for sender in range(5)
                            if mask[sender] and bool(operator.edges[receiver, sender])
                        ]
                        if mask[receiver]
                        else [receiver]
                    )
                    logits = torch.stack(
                        [
                            torch.dot(query[receiver], key[sender]) / math.sqrt(hidden)
                            for sender in allowed
                        ]
                    )
                    if operator.config.routing == "uniform":
                        logits = torch.zeros_like(logits)
                    weights = logits.softmax(0)
                    message = sum(
                        (
                            weight * value[sender]
                            for weight, sender in zip(weights, allowed, strict=True)
                        ),
                        torch.zeros_like(local[receiver]),
                    )
                    update = local[receiver] + operator.config.local_gain * (
                        operator.local_out.weight @ message
                    )
                    updated.append(update if mask[receiver] else torch.zeros_like(update))
                local = torch.stack(updated)
            if operator.config.summary:
                pooled = local.sum(0) / max(1, sum(mask))
                argument = normalize(summary, operator.summary_norm) + pooled
                first = operator.summary_update[0].weight @ argument
                delta = operator.summary_update[2].weight @ (first * first.sigmoid())
                if any(mask):
                    summary = summary + operator.config.summary_gain * delta
                current = context[row] + operator.summary_to_local.weight @ summary
        projected = (local - initial + current) @ operator.readout.weight.T
        answers.append(
            torch.stack(
                [blocks[row, i] + projected[i] if mask[i] else blocks[row, i] for i in range(5)]
            )
        )
    return torch.stack(answers)


def check_trace(result, context, programs, *, summary):
    trace = result.trace
    assert isinstance(trace, dict)
    assert tuple(trace["programs"]) == tuple(programs)
    norms = trace["context_norms"]
    assert isinstance(norms, torch.Tensor) and norms.shape == context.shape[:1]
    assert norms.grad_fn is None and not norms.requires_grad
    close(norms.to(device=context.device, dtype=context.dtype), context.detach().norm(dim=-1))
    expected_phases = {
        AT_END: ("local",) * 4 + (("summary",) if summary else ()),
        INTERLEAVED: ("local", "local", "summary", "local", "local", "summary")
        if summary
        else ("local",) * 4,
    }
    assert set(trace["phases"]) == set(programs)
    for program in set(programs):
        assert tuple(trace["phases"][program]) == expected_phases[program]
    work = trace["work"]
    assert type(work["local_node_evaluations"]) is int
    assert type(work["summary_evaluations"]) is int
    assert work["local_node_evaluations"] == 20 * len(programs)
    assert work["summary_evaluations"] == (
        sum(1 if program == AT_END else 2 for program in programs) if summary else 0
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "program,topology,routing,summary",
    [
        (AT_END, "complete", "attention", True),
        (INTERLEAVED, "ring", "attention", True),
        (INTERLEAVED, "self", "uniform", True),
        (AT_END, "ring", "uniform", False),
        (INTERLEAVED, "complete", "attention", False),
    ],
)
def test_fixed_program_matches_independent_masked_reference(
    device, dtype, program, topology, routing, summary
):
    operator = operator_on(device, dtype, topology=topology, routing=routing, summary=summary)
    blocks, presence, context = inputs(device, dtype)
    run = executor(program=program)
    result = run(operator, blocks, presence, context)
    expected = reference(operator, blocks, presence, context, [program] * len(blocks))
    close(result.blocks, expected)
    assert result.blocks.shape == blocks.shape
    assert result.blocks.dtype == blocks.dtype and result.blocks.device == blocks.device
    assert torch.equal(result.blocks[2], blocks[2])
    check_trace(result, context, [program] * len(blocks), summary=summary)


@pytest.mark.parametrize("device", DEVICES)
def test_none_presence_is_exactly_the_all_present_control(device):
    operator = operator_on(device)
    blocks, _, context = inputs(device)
    run = executor(program=INTERLEAVED)
    first = run(operator, blocks, None, context)
    complete = torch.ones(blocks.shape[:2], dtype=torch.bool, device=device)
    second = run(operator, blocks, complete, context)
    close(first.blocks, second.blocks)
    close(first.blocks, reference(operator, blocks, None, context, [INTERLEAVED] * len(blocks)))
    check_trace(first, context, [INTERLEAVED] * len(blocks), summary=True)


@pytest.mark.parametrize("device", DEVICES)
def test_without_summary_programs_have_four_identical_local_steps(device):
    operator = operator_on(device, summary=False)
    blocks, presence, context = inputs(device)
    first = executor(program=AT_END)(operator, blocks, presence, context)
    second = executor(program=INTERLEAVED)(operator, blocks, presence, context)
    close(first.blocks, second.blocks)
    check_trace(first, context, [AT_END] * len(blocks), summary=False)
    check_trace(second, context, [INTERLEAVED] * len(blocks), summary=False)


@pytest.mark.parametrize("device", DEVICES)
def test_context_norm_selection_preserves_ties_permutation_and_partition(device):
    operator = operator_on(device)
    blocks, presence, _ = inputs(device, rows=4)
    context = blocks.new_tensor([[0, 0, 0], [3, 4, 0], [6, 8, 0], [0, 0, 6]])
    programs = [AT_END, AT_END, INTERLEAVED, INTERLEAVED]
    run = executor(selection="context_norm_v1", threshold=5.0)
    original = run(operator, blocks, presence, context)
    close(original.blocks, reference(operator, blocks, presence, context, programs))
    check_trace(original, context, programs, summary=True)
    order = torch.tensor([2, 0, 3, 1], device=device)
    permuted = run(operator, blocks[order], presence[order], context[order])
    close(permuted.blocks, original.blocks[order])
    check_trace(permuted, context[order], [programs[i] for i in (2, 0, 3, 1)], summary=True)
    pieces = [
        run(operator, blocks[start:stop], presence[start:stop], context[start:stop])
        for start, stop in ((0, 1), (1, 3), (3, 4))
    ]
    close(torch.cat([piece.blocks for piece in pieces]), original.blocks)
    assert sum(piece.trace["work"]["local_node_evaluations"] for piece in pieces) == 80
    assert sum(piece.trace["work"]["summary_evaluations"] for piece in pieces) == 6


@pytest.mark.parametrize("device", DEVICES)
def test_selection_uses_only_supplied_context_and_preserves_rng(device):
    operator = operator_on(device)
    blocks, presence, context = inputs(device)
    api = execution_api()
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state(0).clone() if device == "cuda:0" else None
    config = api.RegionalExecutionConfig(selection="context_norm_v1", context_threshold=0.25)
    run = api.RegionalExecution(3, config, max_batch=8)
    first = run(operator, blocks, presence, context)
    changed = run(operator, blocks + 0.7, ~presence, context)
    assert tuple(first.trace["programs"]) == tuple(changed.trace["programs"])
    close(first.trace["context_norms"], changed.trace["context_norms"])
    assert list(run.parameters()) == []
    assert torch.equal(before_cpu, torch.get_rng_state())
    if before_cuda is not None:
        assert torch.equal(before_cuda, torch.cuda.get_rng_state(0))


@pytest.mark.parametrize("device", DEVICES)
def test_execution_does_not_mutate_owner_or_keep_state_between_predictions(device):
    operator = operator_on(device)
    blocks, presence, context = inputs(device)
    before = copy.deepcopy(operator.state_dict())
    owner_config = operator.config
    old_output = operator(blocks, presence)
    run = executor(program=INTERLEAVED)
    first = run(operator, blocks, presence, context)
    first_norms = first.trace["context_norms"].clone()
    run(operator, blocks * 1.3, ~presence, context + 0.4)
    again = run(operator, blocks, presence, context)
    close(first.blocks, again.blocks)
    assert operator.config is owner_config
    assert (operator.config.cycles, operator.config.local_steps) == (3, 1)
    after = operator.state_dict()
    assert after.keys() == before.keys()
    for name, value in before.items():
        assert (
            torch.equal(after[name], value)
            if isinstance(value, torch.Tensor)
            else after[name] == value
        )
    assert torch.equal(old_output, operator(blocks, presence))
    assert torch.equal(first.trace["context_norms"], first_norms)


@pytest.mark.parametrize("device", DEVICES)
def test_interleaved_summary_changes_locals_when_the_final_summary_skip_is_zero(device):
    operator = operator_on(device, hidden=2, topology="self", routing="uniform")
    identity = torch.eye(2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for parameter in operator.parameters():
            parameter.zero_()
        operator.local_norm.weight.fill_(1)
        operator.qkv.weight[4:].copy_(identity)
        operator.local_out.weight.copy_(identity)
        operator.summary_to_local.weight[0, 0] = 1
        operator.summary_update[0].weight.copy_(torch.cat((identity, -identity)))
        operator.summary_update[2].weight.copy_(torch.cat((identity, -identity), dim=1))
        operator.readout.weight[1, 1] = 1
    assert torch.equal(
        operator.readout.weight @ operator.summary_to_local.weight, torch.zeros_like(identity)
    )
    blocks = torch.zeros((1, 5, 2), dtype=torch.float64, device=device)
    blocks[0, 0, 0] = 1
    presence = torch.ones((1, 5), dtype=torch.bool, device=device)
    context = blocks.new_zeros((1, 2))
    at_end = executor(2, program=AT_END)(operator, blocks, presence, context)
    interleaved = executor(2, program=INTERLEAVED)(operator, blocks, presence, context)
    assert torch.equal(at_end.blocks[0, 1], blocks[0, 1])
    assert interleaved.blocks[0, 1, 1] < -0.4
    close(interleaved.blocks, reference(operator, blocks, presence, context, [INTERLEAVED]))


@pytest.mark.parametrize("device", DEVICES)
def test_small_fp64_gradcheck_includes_blocks_and_context(device):
    operator = operator_on(device, hidden=2).requires_grad_(False)
    blocks, presence, context = inputs(device, rows=1, hidden=2)
    blocks = blocks.requires_grad_(True)
    context = context.requires_grad_(True)
    run = executor(2, program=INTERLEAVED)
    with torch.random.fork_rng(devices=[0] if device == "cuda:0" else []):
        assert torch.autograd.gradcheck(
            lambda x, c: run(operator, x, presence, c).blocks,
            (blocks, context),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-3,
            fast_mode=True,
        )
    result = run(operator, blocks, presence, context)
    assert result.blocks.requires_grad
    assert not result.trace["context_norms"].requires_grad


@pytest.mark.parametrize("device", DEVICES)
def test_shared_parameter_gradients_match_the_independent_reference(device):
    operator = operator_on(device)
    blocks, presence, context = inputs(device, rows=1)
    blocks = blocks.requires_grad_(True)
    context = context.requires_grad_(True)
    probe = torch.sin(torch.arange(15, dtype=torch.float64, device=device)).reshape(1, 5, 3)
    variables = (*operator.parameters(), blocks, context)
    actual = executor(program=INTERLEAVED)(operator, blocks, presence, context).blocks
    actual_gradients = torch.autograd.grad((actual * probe).sum(), variables)
    expected = reference(operator, blocks, presence, context, [INTERLEAVED])
    expected_gradients = torch.autograd.grad((expected * probe).sum(), variables)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize(
    "options",
    [
        {"selection": "unknown"},
        {"program": "unknown"},
        {"context_threshold": 0.0},
        {"selection": "context_norm_v1"},
        {"selection": "context_norm_v1", "context_threshold": -1.0},
        {"selection": "context_norm_v1", "context_threshold": float("nan")},
        {"selection": "context_norm_v1", "context_threshold": float("inf")},
        {"selection": "context_norm_v1", "context_threshold": True},
        {"selection": "context_norm_v1", "context_threshold": 1.0, "program": INTERLEAVED},
    ],
)
def test_invalid_selection_contract_is_rejected(options):
    with pytest.raises(ValueError):
        execution_api().RegionalExecutionConfig(**options)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "fault",
    [
        "block_shape",
        "block_dtype",
        "block_sparse",
        "context_shape",
        "context_dtype",
        "context_sparse",
        "presence_shape",
        "presence_dtype",
        "presence_sparse",
        "nan_context",
        "nan_absent_block",
        "empty",
    ],
)
def test_invalid_inputs_are_rejected_before_operator_phases(device, fault):
    operator = operator_on(device)
    blocks, presence, context = inputs(device)
    if fault == "block_shape":
        blocks = blocks[:, :4]
    elif fault == "block_dtype":
        blocks = blocks.float()
    elif fault == "block_sparse":
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
    elif fault == "nan_context":
        context[2, 0] = float("nan")
    elif fault == "nan_absent_block":
        blocks[2, 0, 0] = float("nan")
    else:
        blocks, presence, context = blocks[:0], presence[:0], context[:0]
    calls = []
    handles = [
        module.register_forward_pre_hook(lambda _module, _args, label=name: calls.append(label))
        for name, module in operator.named_modules()
        if name
    ]
    try:
        with pytest.raises(ValueError):
            executor()(operator, blocks, presence, context)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == []


@pytest.mark.cuda
def test_context_device_mismatch_is_rejected_before_operator_phases():
    operator = operator_on("cuda:0")
    blocks, presence, context = inputs("cuda:0")
    calls = []
    handle = operator.local_norm.register_forward_pre_hook(lambda *_args: calls.append("local"))
    try:
        with pytest.raises(ValueError):
            executor()(operator, blocks, presence, context.cpu())
    finally:
        handle.remove()
    assert calls == []
