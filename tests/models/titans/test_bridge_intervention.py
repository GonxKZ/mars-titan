"""Retiradas de lectores y cortes declarados, sin modificar el banco canónico."""

import importlib
import importlib.util
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_evidence_bridge import bridge, read, snapshot

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def api():
    name = "mars_titan.models.titans.bridge_intervention"
    assert importlib.util.find_spec(name) is not None, "Falta la intervención regional declarada"
    return importlib.import_module(name)


def test_intervention_is_an_explicit_optional_contract():
    from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig

    assert "intervention" in EvidenceBridgeConfig.__dataclass_fields__, (
        "Falta la opción independiente de retirada regional"
    )
    config = api().BridgeInterventionConfig((True, False, True), ((0, 1),))
    assert config.active_regions == (True, False, True)


def intervention(active=(True, False, True), cuts=()):
    return api().BridgeInterventionConfig(active, cuts)


def configured(device="cpu", **options):
    from mars_titan.models.titans.evidence_bridge import SharedEvidenceBridge

    config = replace(bridge(device).config, **dict(dict(intervention=intervention()), **options))
    return SharedEvidenceBridge(config, device=device, dtype=torch.float64)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cycles,gain", [(0, 0.0), (2, 0.5)])
def test_fixed_lesion_equals_a_runtime_mask_and_preserves_the_bank(device, cycles, gain):
    original = bridge(device, cycles=cycles, bridge_gain=gain)
    fixed = configured(device, cycles=cycles, bridge_gain=gain)
    source = snapshot(device)
    before = source.export_cpu()
    state = torch.ones((2, 2), dtype=torch.float64, device=device)
    mask = torch.tensor([[True, False, True]] * 2, device=device)
    expected = read(original, state, source, available=mask)
    actual = read(fixed, state, source)
    for name in ("values", "region_values", "evidence_ids", "evidence_weights", "bridge_weights"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    assert torch.count_nonzero(actual.region_values[:, 1]) == 0
    assert source.export_cpu()["sha256"] == before["sha256"]
    mask[:, 0] = False
    combined = read(fixed, state, source, available=mask)
    assert torch.count_nonzero(combined.region_values[:, :2]) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_directed_cut_keeps_the_reader_and_the_residual(device):
    model = configured(
        device,
        intervention=intervention((True, True, True), ((0, 0), (0, 1))),
        cycles=1,
        bridge_gain=0.25,
    )
    source, state = snapshot(device), torch.ones((1, 2), dtype=torch.float64, device=device)
    local = bridge(device, cycles=0)
    expected = read(local, state, source)
    actual = read(model, state, source)
    assert torch.count_nonzero(actual.bridge_weights[:, 0]) == 0
    torch.testing.assert_close(actual.region_values[:, 0], 0.75 * expected.region_values[:, 0])
    assert actual.evidence_weights[:, 0].sum() == 1
    assert torch.count_nonzero(actual.region_values[:, 0]) > 0


@pytest.mark.parametrize("device", DEVICES)
def test_inactive_all_is_not_an_empty_snapshot_and_recovery_is_exact(device):
    model = configured(device, intervention=intervention((False, False, False)))
    state, source = torch.ones((1, 2), dtype=torch.float64, device=device), snapshot(device)
    result = read(model, state, source)
    assert sorted(result.evidence_ids[0].tolist()) == [1, 2, 3]
    assert result.snapshot_sha256 == source._digest
    for name in ("values", "region_values", "evidence_weights", "bridge_weights"):
        assert torch.count_nonzero(getattr(result, name)) == 0
    restored = configured(device, intervention=intervention((False, False, False)))
    restored.load_state_dict(deepcopy(model.state_dict()))
    assert torch.equal(read(restored, state, source).values, result.values)


@pytest.mark.parametrize("device", DEVICES)
def test_fixed_mask_is_differentiable_in_the_nonlinear_path(device):
    from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig

    model = configured(device, cycles=0, recurrence=BridgeRecurrenceConfig(2, 228))
    source = snapshot(device)
    state = torch.tensor([[0.37, -0.83]], dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: read(model, value, source).values, (state,), eps=1e-6, atol=1e-6, rtol=1e-4
    )
    result = read(model, state, source)
    result.values.square().sum().backward()
    assert torch.isfinite(state.grad).all() and state.grad.count_nonzero()
    assert model.region_value.grad[1].count_nonzero() == 0
    assert model.region_query.grad[1].count_nonzero() == 0
    assert model.region_value.grad[[0, 2]].count_nonzero() > 0


@pytest.mark.parametrize("device", DEVICES)
def test_presence_is_false_for_silenced_readers_with_a_populated_snapshot(device):
    from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig

    reader = EpisodicReadout(
        EpisodicReadoutConfig(
            "a" * 64,
            hidden_size=2,
            evidence_bridge=replace(bridge(device).config, intervention=intervention((False,) * 3)),
        ),
        device=device,
        dtype=torch.float64,
    )
    state, source = torch.ones((1, 2), dtype=torch.float64, device=device), snapshot(device)
    result = reader(state, source, context_id="b" * 64, cutoff=4)
    assert result.reads[0].presence.tolist() == [[False]]
    assert result.reads[0].bridge.evidence_ids.shape == (1, 3)
    assert result.reads[0].bridge.snapshot_sha256 == source._digest


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["empty", "absent"])
def test_empty_and_absent_banks_keep_distinct_provenance_under_intervention(device, mode):
    from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig

    reader = EpisodicReadout(
        EpisodicReadoutConfig(
            "a" * 64,
            hidden_size=2,
            mode="no_bank" if mode == "absent" else "bank",
            evidence_bridge=replace(bridge(device).config, intervention=intervention((False,) * 3)),
        ),
        device=device,
        dtype=torch.float64,
    )
    source = None if mode == "absent" else snapshot(device, count=0)
    result = reader(
        torch.ones((1, 2), dtype=torch.float64, device=device),
        source,
        context_id="b" * 64,
        cutoff=4,
    )
    part = result.reads[0]
    assert part.presence.tolist() == [[False]]
    assert part.values.count_nonzero() == 0
    if mode == "absent":
        assert part.bridge is None
    else:
        assert part.bridge.snapshot_sha256 == source._digest
        assert part.bridge.evidence_ids.shape == (1, 0)
        assert torch.isfinite(part.bridge.bridge_weights).all()


@pytest.mark.parametrize("field", ["active_regions", "edges", "_extra_state"])
def test_fixed_buffer_and_identity_rejection_precedes_copy(field):
    model = configured()
    before = deepcopy(model.state_dict())
    payload = deepcopy(before)
    payload["query.weight"].fill_(99)
    if field == "_extra_state":
        payload[field]["configuration"]["intervention"]["active_regions"] = (True,) * 3
    else:
        payload[field].logical_not_()
    with pytest.raises(ValueError):
        model.load_state_dict(payload)
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, model.state_dict()[name])


def test_preexisting_edge_corruption_cannot_become_the_checkpoint_reference():
    model = configured()
    model.edges.logical_not_()
    before = model.query.weight.detach().clone()
    payload = deepcopy(model.state_dict())
    payload["query.weight"].fill_(99)
    with pytest.raises(ValueError, match="puentes"):
        model.load_state_dict(payload)
    assert torch.equal(before, model.query.weight)


@pytest.mark.parametrize("name", ["edges", "active_regions"])
def test_corrupted_buffer_dtype_cannot_redefine_checkpoint_precision(name):
    model = configured()
    setattr(model, name, getattr(model, name).double())
    before = model.query.weight.detach().clone()
    payload = deepcopy(model.state_dict())
    payload["query.weight"].fill_(99)
    with pytest.raises(ValueError, match="booleanos"):
        model.load_state_dict(payload)
    assert torch.equal(before, model.query.weight)


@pytest.mark.parametrize(
    "active,cuts",
    [
        ((), ()),
        ((1, True), ()),
        ([True, False], ()),
        ((True,) * 9, ()),
        ((True, False), [(0, 1)]),
        ((True, False), ((True, 1),)),
        ((True, False), ((0, 2),)),
        ((True, False), ((0, 1), (0, 1))),
        ((True, False), ((1, 0), (0, 1))),
    ],
)
def test_intervention_rejects_ambiguous_masks_or_edges(active, cuts):
    with pytest.raises(ValueError):
        intervention(active, cuts)


def test_mask_length_and_cut_must_match_the_declared_topology():
    with pytest.raises(ValueError):
        configured(intervention=intervention((True, False)))
    with pytest.raises(ValueError):
        configured(topology="self", intervention=intervention(cuts=((0, 1),)))


@pytest.mark.parametrize("device", DEVICES)
def test_noop_intervention_keeps_parameters_and_rng_but_has_its_own_identity(device):
    original = bridge(device)
    rng = torch.random.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(0).clone() if device != "cpu" else None
    active = configured(device, intervention=intervention((True,) * 3))
    assert torch.equal(rng, torch.random.get_rng_state())
    for name, value in original.named_parameters():
        assert torch.equal(value, dict(active.named_parameters())[name])
    assert active.config.identity()["kind"] == "shared_evidence_bridge_intervention_v1"
    assert active.config.identity() != original.config.identity()
    query = torch.ones((1, 2), dtype=torch.float64, device=device)
    assert torch.equal(
        read(original, query, snapshot(device)).values, read(active, query, snapshot(device)).values
    )
    if gpu_rng is not None:
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state(0))
