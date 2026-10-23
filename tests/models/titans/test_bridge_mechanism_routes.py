"""Rutas regionales y contrato explícito del controlador, sin ajustar parámetros."""

import importlib
from dataclasses import replace

import pytest
import torch
from test_bridge_episodic_readout import DEVICES, bank, configuration
from test_context_selection import batch_on
from test_frozen_financial import frozen_backend as frozen_backend
from test_market_memory import pt4

from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer

CONTRACT = "shared_evidence_bridge_v1"
ROUTES = ("k1", "k2c", "k2r", "k4cc", "k4cr", "k4rc")


def engine(device="cpu", route="k1"):
    model = pt4(None, head=QUANTILE_HEAD).to(device).eval().requires_grad_(False)
    reader = (
        EpisodicReadout(configuration(), device=device, dtype=torch.float64)
        .eval()
        .requires_grad_(False)
    )
    return FrozenMechanismConsumer(
        model,
        readout=reader,
        action=MechanismAction(route=route),
        catalog=MechanismCatalog(reader_contract=CONTRACT),
    )


def reference(reader, base, snapshot, route):
    refresh = {"k1": (), "k2c": (), "k2r": (1,), "k4cc": (), "k4cr": (2,), "k4rc": (1,)}
    positions, state, reads = None, base, []
    for step in range(int(route[1])):
        if step in refresh[route]:
            positions = None
        read, positions, _ = reader._read_parts(state, snapshot, "b" * 64, 10, positions=positions)
        state = reader.refine(state, base, read)
        reads.append(read)
    return state, reads


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("route", ROUTES)
def test_routes_preserve_common_ids_and_count_reprojections(device, route):
    api = importlib.import_module("mars_titan.models.titans.regional_routes")
    reader = EpisodicReadout(configuration(), device=device, dtype=torch.float64)
    base = torch.linspace(-1, 1, 64, dtype=torch.float64, device=device).reshape(2, 32)
    snapshot = bank(device)
    expected, reads = reference(reader, base, snapshot, route)
    first, result = api.run_route_pair(
        reader, base, snapshot, route=route, context_id="b" * 64, cutoff=10
    )
    assert torch.equal(result.state, expected)
    assert result.refinements == int(route[1])
    assert result.searches == 1 + int("r" in route)
    assert result.value_projections == int(route[1])
    assert torch.equal(first.state, reader.refine(base, base, reads[0]))
    assert not result.state.requires_grad
    for actual, wanted in zip(result.reads, reads, strict=True):
        assert not hasattr(actual, "weights")
        assert torch.equal(actual.bridge.evidence_ids, wanted.bridge.evidence_ids)
        assert torch.equal(actual.bridge.evidence_weights, wanted.bridge.evidence_weights)


@pytest.mark.parametrize("device", DEVICES)
def test_regional_routes_preserve_all_query_and_projection_gradients(device):
    api = importlib.import_module("mars_titan.models.titans.regional_routes")
    reader = EpisodicReadout(configuration(), device=device, dtype=torch.float64)
    base = torch.ones((2, 32), dtype=torch.float64, device=device, requires_grad=True)
    expected, _ = reference(reader, base, bank(device), "k4rc")
    parameters = (base, *reader.parameters())
    gradients = torch.autograd.grad(expected.square().sum(), parameters)
    _, result = api.run_route_pair(
        reader,
        base,
        bank(device),
        route="k4rc",
        context_id="b" * 64,
        cutoff=10,
        differentiable=True,
    )
    actual = torch.autograd.grad(result.state.square().sum(), parameters)
    for a, b in zip(gradients, actual, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_controller_uses_regional_contract_and_records_evidence_without_another_mac_step(device):
    consumer = engine(device, "k4cr")
    batch = batch_on(consumer.predictor, ("US/AAA", "US/BBB"), 0)
    at = batch.prediction_at[0]
    snapshot = bank(device, cutoff=at)
    event = consumer.begin(
        consumer.initial_state(),
        flow_ids=batch.flow_ids,
        cutoff=at,
        context_id="b" * 64,
        matures_at=(at + 10, at + 10),
        snapshot=snapshot,
    )
    block = consumer.prepare(batch, consumer.predictor.initial_state(batch.flow_ids), event)
    emitted = consumer.finish(block.event)
    assert block.next_state.mac.memory.steps.tolist() == [1, 1]
    assert emitted.work["reader_refinements"] == 8
    assert emitted.work["reader_searches"] == 4
    assert emitted.work["bridge_value_projections"] == 8
    for row in emitted.rows:
        trace = row["regional_readout"]
        assert trace["snapshot_sha256"] == snapshot._digest
        assert len(trace["reads"]) == 4
        assert len(trace["reads"][0]["regional_weights"]) == 3
        assert "labels" not in trace and "source_values" not in trace


def test_old_catalog_does_not_silently_enable_regional_execution():
    model = pt4(None).eval().requires_grad_(False)
    reader = EpisodicReadout(configuration(), dtype=torch.float64).eval().requires_grad_(False)
    with pytest.raises(ValueError):
        FrozenMechanismConsumer(model, readout=reader)
    with pytest.raises(ValueError):
        FrozenMechanismConsumer(model, catalog=MechanismCatalog(reader_contract=CONTRACT))
    regional = MechanismCatalog(reader_contract=CONTRACT)
    assert regional.fingerprint() != MechanismCatalog().fingerprint()
    assert "reader_contract" not in MechanismCatalog().identity()
    assert replace(regional, reader_contract="legacy").identity() == MechanismCatalog().identity()
