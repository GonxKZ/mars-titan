"""Contratos de la lectura adaptativa sobre la evidencia común, sin ajustes."""

import hashlib
import importlib
import json
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_bridge_episodic_readout import CODEC, CONTEXT, bank
from test_bridge_episodic_readout import configuration as base_configuration
from test_episodic_readout import source as episode_source
from test_evidence_bridge import bridge as legacy_bridge
from test_evidence_bridge import read as legacy_read
from test_evidence_bridge import snapshot as legacy_snapshot

from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from tests.proposals.test_mature_proposals import no_optimizer_steps as no_optimizer_steps

BACKENDS = ("adaptive_reference_full", "adaptive_compact")
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))
DTYPES = (torch.float32, torch.float64)


def adaptive_configuration(*, backend=BACKENDS[0], budget=1216, k=1, steps=4, atol=1e308):
    module = importlib.import_module("mars_titan.models.titans.bridge_recurrence")
    stopping = module.BridgeStoppingConfig(atol, 0.0, backend=backend)
    recurrence = module.BridgeRecurrenceConfig(steps, budget, stopping=stopping)
    base = base_configuration(refinements=k, episode_selection="first_read")
    return replace(
        base,
        evidence_bridge=replace(base.evidence_bridge, cycles=0, recurrence=recurrence),
    )


def fixed_configuration(config, *, steps=1):
    geometry = config.evidence_bridge
    return replace(
        config,
        evidence_bridge=replace(
            geometry, recurrence=replace(geometry.recurrence, steps=steps, stopping=None)
        ),
    )


def snapshot(device, dtype, count=8):
    return EpisodeSnapshot.create(
        **episode_source(count),
        codec_id=CODEC,
        context_id=CONTEXT,
        cutoff=10,
        dtype=dtype,
        device=device,
    )


def query(device, dtype):
    if device != "cpu":
        assert torch.cuda.is_available(), "La comprobación CUDA no tiene alternativa CPU"
    return torch.linspace(-0.4, 0.6, 96, dtype=dtype, device=device).reshape(3, 32)


def assert_payload_equal(actual, expected):
    assert set(actual) == set(expected)
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(actual[name], value), name
        else:
            assert actual[name] == value, name


@contextmanager
def observed_python_calls(*functions):
    """Observar código sin alterar las firmas de los consumidores."""
    codes = {function.__code__ for function in functions}
    calls = []
    previous = sys.getprofile()

    def observe(frame, event, argument):
        if event == "call" and frame.f_code in codes:
            calls.append(frame.f_code.co_name)

    sys.setprofile(observe)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def test_adaptive_readout_is_reachable_in_call():
    reader = EpisodicReadout(adaptive_configuration(), dtype=torch.float64)
    state = torch.zeros((1, 32), dtype=torch.float64)
    result = reader(state, bank("cpu"), context_id=CONTEXT, cutoff=10)
    assert len(result.reads) == 1
    assert result.reads[0].bridge.selected_steps.tolist() == [1]
    assert result.reads[0].bridge.evaluated_steps.tolist() == [4]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_adaptive_j1_matches_fixed_j1_with_the_same_weights_and_evidence(backend, dtype, device):
    config = adaptive_configuration(backend=backend, k=2)
    active = EpisodicReadout(config, device=device, dtype=dtype)
    fixed = EpisodicReadout(fixed_configuration(config), device=device, dtype=dtype)
    assert set(dict(active.named_parameters())) == set(dict(fixed.named_parameters()))
    for name, value in active.named_parameters():
        assert torch.equal(value, dict(fixed.named_parameters())[name]), name
    source = snapshot(device, dtype)
    before_source = source.export_cpu()
    before_parameters = deepcopy(active.state_dict())
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone() if device != "cpu" else None
    actual = active(query(device, dtype), source, context_id=CONTEXT, cutoff=10)
    expected = fixed(query(device, dtype), source, context_id=CONTEXT, cutoff=10)
    assert torch.equal(actual.state, expected.state)
    assert len(actual.reads) == len(expected.reads) == 2
    for left, right in zip(actual.reads, expected.reads, strict=True):
        assert torch.equal(left.values, right.values)
        assert torch.equal(left.presence, right.presence)
        for name in (
            "values",
            "region_values",
            "evidence_ids",
            "evidence_weights",
            "bridge_weights",
        ):
            assert torch.equal(getattr(left.bridge, name), getattr(right.bridge, name)), name
        assert left.bridge.selected_steps.tolist() == [1, 1, 1]
        assert left.bridge.evaluated_steps.tolist() == (
            [4, 4, 4] if backend == BACKENDS[0] else [1, 1, 1]
        )
        assert left.bridge.criterion_met.tolist() == [True, True, True]
        for name, wanted in (
            ("selected_steps", torch.int64),
            ("evaluated_steps", torch.int64),
            ("criterion_met", torch.bool),
        ):
            value = getattr(left.bridge, name)
            assert value.dtype == wanted and value.shape == (3,)
            assert not value.requires_grad and value.grad_fn is None
    after_source = source.export_cpu()
    assert after_source["sha256"] == before_source["sha256"]
    assert_payload_equal(after_source["tensors"], before_source["tensors"])
    assert_payload_equal(active.state_dict(), before_parameters)
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(0))


@pytest.mark.cuda
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_readout_selected_branch_gradients_match_fixed_j1_away_from_the_threshold(backend, dtype):
    assert torch.cuda.is_available(), "La comprobación requiere CUDA"
    config = adaptive_configuration(backend=backend)
    active = EpisodicReadout(config, device="cuda:0", dtype=dtype)
    fixed = EpisodicReadout(fixed_configuration(config), device="cuda:0", dtype=dtype)
    left = query("cuda:0", dtype).requires_grad_()
    right = left.detach().clone().requires_grad_()
    source = snapshot("cuda:0", dtype)
    actual = active(left, source, context_id=CONTEXT, cutoff=10, differentiable=True)
    expected = fixed(right, source, context_id=CONTEXT, cutoff=10, differentiable=True)
    actual.state.square().sum().backward()
    expected.state.square().sum().backward()
    tolerance = (
        dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-12)
    )
    torch.testing.assert_close(left.grad, right.grad, **tolerance)
    for name, parameter in active.named_parameters():
        reference = dict(fixed.named_parameters())[name]
        assert parameter.grad is not None and reference.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, reference.grad, **tolerance)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ("empty", "absent"))
def test_empty_and_absent_snapshots_do_not_invent_adaptive_work(backend, device, kind):
    config = adaptive_configuration(backend=backend, k=2)
    if kind == "absent":
        config = replace(config, mode="no_bank")
    reader = EpisodicReadout(config, device=device, dtype=torch.float64)
    source = snapshot(device, torch.float64, count=0) if kind == "empty" else None
    result = reader(query(device, torch.float64), source, context_id=CONTEXT, cutoff=10)
    for read in result.reads:
        assert not read.presence.any() and not read.values.count_nonzero()
        if kind == "absent":
            assert read.bridge is None
        else:
            assert read.bridge.selected_steps.tolist() == [0, 0, 0]
            assert read.bridge.evaluated_steps.tolist() == [0, 0, 0]
            assert read.bridge.criterion_met.tolist() == [False, False, False]
            assert read.bridge.evidence_ids.shape == (3, 0)
    work = reader.recurrence_work(0)
    assert work["bridge_initial_views"] == 0
    assert work["bridge_selected_node_updates"] == 0
    assert work["bridge_node_updates"] == 0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", DEVICES)
def test_populated_pool_with_no_present_regions_selects_one_and_keeps_the_pool(backend, device):
    reader = EpisodicReadout(
        adaptive_configuration(backend=backend), device=device, dtype=torch.float64
    )
    source = snapshot(device, torch.float64)
    result = reader.evidence_bridge(
        query(device, torch.float64),
        source,
        context_id=CONTEXT,
        cutoff=10,
        available=torch.zeros((3, 3), device=device, dtype=torch.bool),
    )
    assert result.evidence_ids.shape == (3, 8)
    assert result.selected_steps.tolist() == [1, 1, 1]
    assert result.evaluated_steps.tolist() == ([4, 4, 4] if backend == BACKENDS[0] else [1, 1, 1])
    assert result.criterion_met.tolist() == [True, True, True]
    assert not result.region_values.count_nonzero()
    assert not result.evidence_weights.count_nonzero()
    assert not result.values.count_nonzero()


def test_none_preserves_the_pinned_plain_bridge_checkpoint_and_legacy_recurrence_identity():
    expected = json.loads(
        (Path(__file__).parent / "fixtures/shared_evidence_bridge_v1.json").read_text()
    )
    module = importlib.import_module("mars_titan.models.titans.bridge_recurrence")
    recurrence = module.BridgeRecurrenceConfig(4, 1216, stopping=None)
    assert recurrence.identity() == dict(
        schema_version=1,
        kind="regional_mix_tanh_v1",
        steps=4,
        max_node_updates_per_row=1216,
        weights="receiver_region_value_reused_at_every_step",
        update="synchronous_fixed_graph_mixture_tanh",
    )
    model = legacy_bridge()
    assert canonical(model.config.identity()) == expected["configuration_canonical"]
    assert sorted(model.state_dict()) == expected["state_keys"]
    for name, digest in expected["tensor_sha256"].items():
        value = model.state_dict()[name].detach().contiguous().numpy().tobytes()
        assert hashlib.sha256(value).hexdigest() == digest
    state = torch.tensor([[0.25, -0.5], [-0.1, 0.4]], dtype=torch.float64)
    actual = legacy_read(model, state, legacy_snapshot())
    for name, values in expected["outputs"]["cpu"].items():
        tensor = getattr(actual, name).detach().cpu()
        assert torch.equal(tensor, torch.tensor(values, dtype=tensor.dtype))


@pytest.mark.parametrize("change", ("backend", "tolerance"))
def test_adaptive_checkpoint_mismatch_is_rejected_before_copying_any_parameter(change):
    config = adaptive_configuration()
    source = EpisodicReadout(config, dtype=torch.float64)
    other = adaptive_configuration(
        backend=BACKENDS[1] if change == "backend" else BACKENDS[0],
        atol=1.0 if change == "tolerance" else 1e308,
    )
    destination = EpisodicReadout(other, dtype=torch.float64)
    with torch.no_grad():
        destination.refinement.bias.fill_(17)
    before = deepcopy(destination.state_dict())
    with pytest.raises(ValueError):
        destination.load_state_dict(source.state_dict())
    assert_payload_equal(destination.state_dict(), before)


@pytest.mark.parametrize("backend", BACKENDS)
def test_jmax_preflight_rejects_a_short_budget_before_the_first_bridge_projection(backend):
    reader = EpisodicReadout(
        adaptive_configuration(backend=backend, budget=47, k=4), dtype=torch.float64
    )
    with observed_python_calls(type(reader.evidence_bridge).forward) as calls:
        with pytest.raises(ValueError, match="presupuesto|actualizaciones|cuota"):
            reader(query("cpu", torch.float64), bank("cpu"), context_id=CONTEXT, cutoff=10)
    assert calls == []


def test_hierarchy_is_rejected_before_constructing_an_adaptive_reader():
    from mars_titan.models.titans.bridge_hierarchy import BridgeHierarchyConfig

    config = adaptive_configuration()
    with pytest.raises(ValueError, match="jerarqu"):
        replace(
            config.evidence_bridge,
            regions=("one", "two", "three", "four"),
            topology="self",
            hierarchy=BridgeHierarchyConfig(),
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_future_snapshot_is_rejected_before_any_learned_projection(backend):
    reader = EpisodicReadout(adaptive_configuration(backend=backend), dtype=torch.float64)
    with observed_python_calls(torch.nn.Linear.forward) as calls:
        with pytest.raises(ValueError):
            reader(query("cpu", torch.float64), bank("cpu"), context_id=CONTEXT, cutoff=1)
    assert calls == []


@pytest.mark.cuda
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_permutation_and_physical_blocks_preserve_adaptive_readout_and_metadata(backend, dtype):
    reader = EpisodicReadout(adaptive_configuration(backend=backend), device="cuda:0", dtype=dtype)
    state = query("cuda:0", dtype)
    source = snapshot("cuda:0", dtype)
    full = reader(state, source, context_id=CONTEXT, cutoff=10)
    order = torch.tensor([2, 0, 1], device="cuda:0")
    permuted = reader(state[order], source, context_id=CONTEXT, cutoff=10)
    tolerance = (
        dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-12)
    )
    torch.testing.assert_close(permuted.state, full.state[order], **tolerance)
    pieces = [
        reader(state[start:stop], source, context_id=CONTEXT, cutoff=10)
        for start, stop in ((0, 1), (1, 3))
    ]
    torch.testing.assert_close(
        torch.cat([piece.state for piece in pieces]), full.state, **tolerance
    )
    for name in ("selected_steps", "evaluated_steps", "criterion_met"):
        assert torch.equal(
            getattr(permuted.reads[0].bridge, name), getattr(full.reads[0].bridge, name)[order]
        )
        assert torch.equal(
            torch.cat([getattr(piece.reads[0].bridge, name) for piece in pieces]),
            getattr(full.reads[0].bridge, name),
        )
