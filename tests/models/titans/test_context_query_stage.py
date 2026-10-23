"""Lectura confirmada y tokens pendientes separados, con fixtures sin aprendizaje."""

from dataclasses import fields, is_dataclass

import pytest
import torch
from test_market_memory import (
    DAY,
    KINDS,
    MARKET_MEMORY,
    SECTOR_MEAN,
    START,
    context,
    tokens,
)

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))
FLOWS = ("US/AAA", "US/BBB", "CN/600000.SS")
NEXT = START + DAY


def module_on(kind, device):
    module = context(kind, gain=7).to(device)
    assert callable(getattr(module, "read_confirmed", None))
    assert callable(getattr(module, "stage_tokens", None))
    return module


def rows_on(device, *, seed=11):
    return tokens(len(FLOWS), seed=seed).to(device)


def confirmed(module, device):
    previous = rows_on(device, seed=5)
    _, pending = module.read(previous, FLOWS, (START,) * len(FLOWS), module.initial_state("p"))
    return module.commit(pending)


def capture(value):
    if isinstance(value, torch.Tensor):
        return (
            value.dtype,
            value.device,
            value.requires_grad,
            value._version,
            value.detach().clone(),
        )
    if is_dataclass(value):
        return {field.name: capture(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: capture(part) for key, part in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(capture(part) for part in value)
    return value


def assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.device == expected.device
        assert actual.shape == expected.shape and torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert_same(actual[key], value)
    elif isinstance(expected, tuple):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_same(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
def test_repeated_queries_preserve_both_confirmed_and_pending_state(kind, device):
    module = module_on(kind, device)
    state = confirmed(module, device)
    queries = rows_on(device)
    original = capture(state)
    first = module.read_confirmed(queries, FLOWS, (NEXT,) * len(FLOWS), state)
    assert_same(capture(state), original)
    assert first.shape == queries.shape and first.dtype == queries.dtype
    assert first.device == queries.device
    pending = module.stage_tokens(queries[:1] + 4, FLOWS[:1], (NEXT,), state)
    pending_before = capture(pending)
    for _ in range(3):
        actual = module.read_confirmed(queries, FLOWS, (NEXT,) * len(FLOWS), pending)
        assert torch.equal(actual, first)
        assert_same(capture(pending), pending_before)
        assert_same(capture(state), original)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
def test_staging_uses_the_write_tokens_once_and_does_not_read(kind, device, monkeypatch):
    module = module_on(kind, device)
    state = confirmed(module, device)
    queries, writes = rows_on(device), rows_on(device, seed=17)
    assert not torch.equal(queries, writes)
    before = capture(state)

    def forbidden_read(*args, **kwargs):
        pytest.fail("Preparar tokens pendientes no debe consultar la memoria")

    monkeypatch.setattr(module, "_read", forbidden_read)
    pending = module.stage_tokens(writes, FLOWS, (NEXT,) * len(FLOWS), state)
    assert pending.pending_flows == (FLOWS[:2], FLOWS[2:])
    assert torch.equal(pending.pending_tokens[0], writes[:2])
    assert torch.equal(pending.pending_tokens[1], writes[2:])
    assert pending.committed_at == state.committed_at
    pending_before = capture(pending)
    with pytest.raises(ValueError):
        module.stage_tokens(writes, FLOWS, (NEXT,) * len(FLOWS), pending)
    assert_same(capture(state), before)
    assert_same(capture(pending), pending_before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
def test_the_legacy_read_is_exactly_query_plus_stage_and_preserves_rng(kind, device):
    module = module_on(kind, device)
    state = confirmed(module, device)
    queries = rows_on(device)
    parameters, identity = capture(module.state_dict()), module.identity()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone() if device == "cuda:0" else None
    expected, expected_pending = module.read(queries, FLOWS, (NEXT,) * len(FLOWS), state)
    actual = module.read_confirmed(queries, FLOWS, (NEXT,) * len(FLOWS), state)
    actual_pending = module.stage_tokens(queries, FLOWS, (NEXT,) * len(FLOWS), state)
    assert torch.equal(actual, expected)
    assert_same(capture(actual_pending), capture(expected_pending))
    assert_same(capture(module.state_dict()), parameters)
    assert module.identity() == identity
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(0))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
def test_queries_reject_future_snapshots_and_staging_rejects_unclosed_sessions(kind, device):
    module = module_on(kind, device)
    state = confirmed(module, device)
    rows = rows_on(device)
    before = capture(state)
    for moment in (START - 1, START):
        with pytest.raises(ValueError):
            module.read_confirmed(rows, FLOWS, (moment,) * len(FLOWS), state)
    pending = module.stage_tokens(rows[:1], FLOWS[:1], (NEXT,), state)
    pending_before = capture(pending)
    with pytest.raises(ValueError):
        module.stage_tokens(rows[1:2], FLOWS[1:2], (NEXT + DAY,), pending)
    with pytest.raises(ValueError):
        module.stage_tokens(rows[:2], FLOWS[:2], (NEXT, NEXT + 1), state)
    with pytest.raises(ValueError):
        module.stage_tokens(rows[:2], (FLOWS[0], FLOWS[0]), (NEXT, NEXT), state)
    assert_same(capture(pending), pending_before)
    assert_same(capture(state), before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
def test_queries_ignore_pending_values_and_other_query_rows(kind, device):
    module = module_on(kind, device)
    state = confirmed(module, device)
    queries = rows_on(device)
    one = module.stage_tokens(queries[:1], FLOWS[:1], (NEXT,), state)
    two = module.stage_tokens(queries[:1] + 40, FLOWS[:1], (NEXT,), state)
    expected = module.read_confirmed(queries, FLOWS, (NEXT,) * len(FLOWS), one)
    actual = module.read_confirmed(queries, FLOWS, (NEXT,) * len(FLOWS), two)
    assert torch.equal(actual, expected)
    changed = queries.clone()
    changed[1, 0] += 3
    other = module.read_confirmed(changed, FLOWS, (NEXT,) * len(FLOWS), two)
    assert torch.equal(other[[0, 2]], expected[[0, 2]])
    if kind == SECTOR_MEAN:
        assert torch.equal(other, expected)
    else:
        assert not torch.equal(other[1], expected[1])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
def test_permutation_and_partition_keep_reads_and_canonical_commit(kind, device):
    module = module_on(kind, device)
    state = confirmed(module, device)
    queries, writes = rows_on(device), rows_on(device, seed=17)
    expected = module.read_confirmed(queries, FLOWS, (NEXT,) * len(FLOWS), state)
    whole = module.commit(module.stage_tokens(writes, FLOWS, (NEXT,) * len(FLOWS), state))
    pending = state
    for selected in ((1,), (2, 0)):
        positions = torch.tensor(selected, dtype=torch.int64, device=device)
        flows = tuple(FLOWS[index] for index in selected)
        moments = (NEXT,) * len(flows)
        actual = module.read_confirmed(queries[positions], flows, moments, pending)
        torch.testing.assert_close(actual, expected[positions], rtol=1e-13, atol=1e-15)
        pending = module.stage_tokens(writes[positions], flows, moments, pending)
    split = module.commit(pending)
    assert_same(capture(split), capture(whole))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
def test_a_row_selection_stages_only_selected_write_tokens(kind, device):
    module = module_on(kind, device)
    state = confirmed(module, device)
    queries, writes = rows_on(device), rows_on(device, seed=17)
    selection = torch.tensor([True, False, True], device=device)
    expected = module.read_confirmed(queries, FLOWS, (NEXT,) * len(FLOWS), state)
    selected_flows = (FLOWS[0], FLOWS[2])
    actual = module.read_confirmed(queries[selection], selected_flows, (NEXT, NEXT), state)
    scattered = torch.zeros_like(queries)
    scattered[selection] = actual
    torch.testing.assert_close(scattered[selection], expected[selection], rtol=1e-13, atol=1e-15)
    assert not scattered[~selection].any()
    assert state.pending_flows == ((), ()) and state.pending_tokens == (None, None)
    pending = module.stage_tokens(writes[selection], selected_flows, (NEXT, NEXT), state)
    assert pending.pending_flows == ((FLOWS[0],), (FLOWS[2],))
    assert torch.equal(pending.pending_tokens[0], writes[:1])
    assert torch.equal(pending.pending_tokens[1], writes[2:])


@pytest.mark.parametrize("device", DEVICES)
def test_market_queries_keep_gradients_to_the_query_and_output(device):
    module = module_on(MARKET_MEMORY, device)
    state = confirmed(module, device)
    queries = rows_on(device).requires_grad_(True)
    read = module.read_confirmed(queries, FLOWS, (NEXT,) * len(FLOWS), state)
    gradients = torch.autograd.grad(read.square().sum(), (queries, module.query, module.output))
    assert all(gradient.abs().sum() > 0 for gradient in gradients)
    assert state.pending_tokens == (None, None)


@pytest.mark.parametrize("device", DEVICES)
def test_sector_read_depends_on_confirmed_means_and_output_but_not_query_values(device):
    module = module_on(SECTOR_MEAN, device)
    state = confirmed(module, device)
    queries = rows_on(device).requires_grad_(True)
    read = module.read_confirmed(queries, FLOWS, (NEXT,) * len(FLOWS), state)
    query_grad, output_grad = torch.autograd.grad(
        read.square().sum(), (queries, module.output), allow_unused=True
    )
    assert query_grad is None or not query_grad.any()
    assert output_grad.abs().sum() > 0
    assert state.pending_tokens == (None, None)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
def test_staged_tokens_remain_differentiable_through_a_later_committed_read(kind, device):
    module = module_on(kind, device)
    initial = module.initial_state("p", differentiable=True)
    writes = rows_on(device, seed=17).requires_grad_(True)
    pending = module.stage_tokens(writes, FLOWS, (START,) * len(FLOWS), initial)
    assert all(value.requires_grad for value in pending.pending_tokens)
    state = module.commit(pending, differentiable=True)
    read = module.read_confirmed(rows_on(device), FLOWS, (NEXT,) * len(FLOWS), state)
    (gradient,) = torch.autograd.grad(read.square().sum(), writes)
    assert gradient.abs().sum() > 0
    assert initial.pending_tokens == (None, None)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
def test_a_nondifferentiable_commit_cuts_the_staged_token_graph(kind, device):
    module = module_on(kind, device)
    writes = rows_on(device, seed=17).requires_grad_(True)
    pending = module.stage_tokens(writes, FLOWS, (START,) * len(FLOWS), module.initial_state("p"))
    state = module.commit(pending, differentiable=False)
    read = module.read_confirmed(rows_on(device), FLOWS, (NEXT,) * len(FLOWS), state)
    (gradient,) = torch.autograd.grad(read.square().sum(), writes, allow_unused=True)
    assert gradient is None
    assert state.pending_tokens == (None, None)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
def test_split_methods_keep_legacy_identity_and_checkpoint_schema(kind, device):
    module = module_on(kind, device)
    state = confirmed(module, device)
    identity = module.identity()
    assert identity["read"] == "previous_sessions_only" and identity["output_initial"] == "zero"
    if kind == MARKET_MEMORY:
        assert set(identity) == {
            "kind",
            "read",
            "output_initial",
            "memory",
            "write",
            "gates",
            "query",
        }
        assert identity["write"] == "once_per_session_after_emission_mean_inner_gradient"
        assert set(module.state_dict()) == {
            "query",
            "output",
            *(f"memory.{name}" for name in module.memory.state_dict()),
        }
    else:
        assert set(identity) == {"kind", "read", "output_initial", "sectors", "write"}
        assert identity["write"] == "mean_token_of_the_sector_in_its_last_session"
        assert set(module.state_dict()) == {"output"}
    payload = module.export_state(state)
    assert set(payload) == {"schema_version", "configuration", "committed_at", "memory", "means"}
    assert payload["schema_version"] == 1 and payload["configuration"] == identity
    assert {field.name for field in fields(state)} == {
        "config_id",
        "parameter_id",
        "committed_at",
        "pending_at",
        "pending_flows",
        "pending_tokens",
        "memory",
        "means",
    }
    restored = module.restore_state(payload, "p", device=module.output.device)
    queries = rows_on(device)
    expected = module.read_confirmed(queries, FLOWS, (NEXT,) * len(FLOWS), state)
    actual = module.read_confirmed(queries, FLOWS, (NEXT,) * len(FLOWS), restored)
    assert torch.equal(actual, expected)
    assert_same(capture(module.export_state(state)), capture(module.export_state(restored)))
