"""Lectura regional tipada y composición financiera sin modificar el lector anterior."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_episodic_readout import CODEC, CONTEXT, source
from test_financial_adapter import setup
from test_frozen_financial import frozen_backend as frozen_backend

from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.regional_episode_readout import (
    BridgedReadoutResult,
    RegionalEpisodeRead,
)

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def configuration(**options):
    common = dict(codec_id=CODEC, hidden_size=32)
    return EpisodicReadoutConfig(
        **common, evidence_bridge=EvidenceBridgeConfig(**common), **options
    )


def bank(device, count=8, cutoff=10):
    return EpisodeSnapshot.create(
        **source(count),
        codec_id=CODEC,
        context_id=CONTEXT,
        cutoff=cutoff,
        device=device,
        dtype=torch.float64,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("selection", ["per_step", "first_read"])
def test_bridge_reads_have_their_own_type_and_preserve_common_evidence(device, k, selection):
    model = EpisodicReadout(
        configuration(refinements=k, episode_selection=selection),
        device=device,
        dtype=torch.float64,
    )
    state = torch.linspace(-0.5, 0.5, 64, dtype=torch.float64, device=device).reshape(2, 32)
    snapshot = bank(device)
    before = snapshot.export_cpu()
    result = model(state, snapshot, context_id=CONTEXT, cutoff=10)
    assert type(result) is BridgedReadoutResult
    assert len(result.reads) == k and result.state.shape == state.shape
    for read in result.reads:
        assert type(read) is RegionalEpisodeRead
        assert not hasattr(read, "weights") and not hasattr(read, "ids")
        assert read.presence.tolist() == [[True], [True]]
        assert read.bridge.evidence_weights.shape == (2, 3, 8)
        assert read.bridge.snapshot_sha256 == before["sha256"]
    if selection == "first_read":
        assert all(
            torch.equal(read.bridge.evidence_ids, result.reads[0].bridge.evidence_ids)
            for read in result.reads
        )
    assert snapshot.export_cpu()["sha256"] == before["sha256"]


@pytest.mark.parametrize("device", DEVICES)
def test_frozen_composition_emits_with_one_mac_update_and_the_existing_head(device):
    model, batch = setup()
    model.to(device=device).eval().requires_grad_(False)
    if device != "cpu":
        from test_financial_adapter import raw_batch, specification

        from mars_titan.models.titans.financial_inputs import DecisionBatch, validated_cpu_batch

        batch = DecisionBatch.from_validated(
            validated_cpu_batch(raw_batch(), specification()), device=device, dtype=torch.float64
        )
    reader = (
        EpisodicReadout(configuration(refinements=2), device=device, dtype=torch.float64)
        .eval()
        .requires_grad_(False)
    )
    consumer = FrozenFinancialConsumer(model, readout=reader)
    previous = model.initial_state(batch.flow_ids)
    result = consumer.prepare(
        batch, previous, context_id=CONTEXT, snapshot=bank(device, cutoff=batch.prediction_at[0])
    )
    assert result.next_state.mac.memory.steps.tolist() == [1, 1]
    assert len(result.readout.reads) == 2
    with torch.no_grad():
        expected = model.head(result.readout.state).squeeze(-1)
    assert torch.equal(result.point_predictions, expected)
    assert "mars_titan.models.titans.evidence_bridge" in consumer.identity()["implementation"]


def test_disabled_configuration_retains_the_previous_contract_and_parameters():
    plain = EpisodicReadoutConfig(CODEC, hidden_size=32)
    explicit = replace(plain, evidence_bridge=None)
    assert "evidence_bridge" not in plain.identity()
    left, right = (
        EpisodicReadout(plain, dtype=torch.float64),
        EpisodicReadout(explicit, dtype=torch.float64),
    )
    assert set(left.state_dict()) == set(right.state_dict())
    for name, value in left.named_parameters():
        assert torch.equal(value, dict(right.named_parameters())[name])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode,count", [("bank", 0), ("no_bank", 0)])
def test_absent_read_is_typed_and_does_not_invent_a_snapshot(device, mode, count):
    model = EpisodicReadout(
        configuration(mode=mode, refinements=4), device=device, dtype=torch.float64
    )
    snapshot = None if mode == "no_bank" else bank(device, count)
    result = model(
        torch.zeros((2, 32), device=device, dtype=torch.float64),
        snapshot,
        context_id=CONTEXT,
        cutoff=10,
    )
    for read in result.reads:
        assert not read.presence.any() and not read.values.count_nonzero()
        assert (read.bridge is None) == (mode == "no_bank")


@pytest.mark.parametrize("device", DEVICES)
def test_bridge_gradients_and_checkpoint_resume_preserve_the_next_read(device):
    model = EpisodicReadout(
        configuration(refinements=2, episode_selection="first_read"),
        device=device,
        dtype=torch.float64,
    )
    state = torch.ones((2, 32), device=device, dtype=torch.float64, requires_grad=True)
    snapshot = bank(device)
    result = model(state, snapshot, context_id=CONTEXT, cutoff=10, differentiable=True)
    result.state.square().sum().backward()
    assert torch.isfinite(state.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    restored = EpisodicReadout(model.config, device=device, dtype=torch.float64)
    restored.load_state_dict(deepcopy(model.state_dict()))
    actual = restored(state.detach(), snapshot, context_id=CONTEXT, cutoff=10)
    assert torch.equal(result.state.detach(), actual.state)
    assert not actual.state.requires_grad


def test_child_checkpoint_is_validated_before_copying_the_parent():
    model = EpisodicReadout(configuration(), dtype=torch.float64)
    before = deepcopy(model.state_dict())
    payload = deepcopy(before)
    payload["step_logit"].add_(1)
    payload["evidence_bridge.edges"][0, 0] = False
    with pytest.raises(ValueError, match="puentes"):
        model.load_state_dict(payload)
    for name, expected in before.items():
        if isinstance(expected, torch.Tensor):
            assert torch.equal(expected, model.state_dict()[name])


@pytest.mark.parametrize(
    "name,value",
    [
        ("hidden_size", 16),
        ("codec_id", "c" * 64),
        ("neighbors", 4),
        ("max_batch", 16),
        ("temperature", 2.0),
        ("max_working_bytes", 1024),
    ],
)
def test_bridge_and_readout_cannot_declare_different_contracts(name, value):
    chosen = configuration()
    with pytest.raises(ValueError, match="compartir"):
        replace(chosen, evidence_bridge=replace(chosen.evidence_bridge, **{name: value}))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("invalid", ["negative", "large", "duplicate", "shape"])
def test_explicit_positions_are_rejected_before_any_gather(device, invalid):
    model = EpisodicReadout(configuration(), device=device, dtype=torch.float64)
    positions = torch.arange(8, dtype=torch.int64, device=device)[None]
    if invalid == "negative":
        positions[0, 0] = -1
    if invalid == "large":
        positions[0, 0] = 8
    if invalid == "duplicate":
        positions[0, 1] = 0
    if invalid == "shape":
        positions = positions[:, :2]
    with pytest.raises(ValueError, match="posiciones"):
        model.evidence_bridge(
            torch.zeros((1, 32), device=device, dtype=torch.float64),
            bank(device),
            context_id=CONTEXT,
            cutoff=10,
            positions=positions,
        )


def test_legacy_routes_reject_regional_weights_instead_of_averaging_them():
    from mars_titan.models.titans.refinement_routes import run_route

    model = EpisodicReadout(configuration(), dtype=torch.float64)
    with pytest.raises(ValueError, match="lector"):
        run_route(
            model,
            torch.zeros((2, 32), dtype=torch.float64),
            bank("cpu"),
            route="k1",
            context_id=CONTEXT,
            cutoff=10,
        )
