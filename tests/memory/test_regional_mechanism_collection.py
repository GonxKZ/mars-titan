"""Alternativas regionales con una sola evidencia y supervisión nativa posterior."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_bridge_episodic_readout import DEVICES
from test_episodic_readout import CODEC, CONTEXT, source
from test_financial_adapter import raw_batch
from test_frozen_financial import frozen_backend as frozen_backend
from test_market_memory import KINDS
from test_mechanism_consumer import calibration, consumer
from test_mechanism_observations import AT, DAY, observation_rows

from mars_titan.memory.mechanism_collection import CollectedMechanisms, MechanismCollector
from mars_titan.models.titans import regional_routes
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig
from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.refinement_routes import ROUTES

CONTRACT = "shared_evidence_bridge_v1"


def regional(device="cpu", *, populated=True, limit=None):
    original = consumer(device, kind=KINDS[0], reader=True, populated=populated)
    config = replace(
        original.readout.config,
        evidence_bridge=EvidenceBridgeConfig(CODEC, hidden_size=32),
    )
    if limit is not None:
        config = replace(
            config,
            max_working_bytes=limit,
            evidence_bridge=replace(config.evidence_bridge, max_working_bytes=limit),
        )
    reader = (
        EpisodicReadout(config, device=device, dtype=torch.float64).eval().requires_grad_(False)
    )
    return FrozenMechanismConsumer(
        original.predictor,
        readout=reader,
        calibration=calibration(),
        catalog=MechanismCatalog(reader_contract=CONTRACT),
    )


def collect(engine, snapshot=None, *, split=False):
    rows = observation_rows(engine)
    builder = MechanismCollector(
        engine,
        engine.initial_state(),
        rows=rows,
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + DAY,) * len(rows),
        snapshot=snapshot,
    )
    for block in (rows[1:], rows[:1]) if split else (rows,):
        builder.add(block, engine.predictor.initial_state(tuple(r.flow_id for r in block)))
    return builder.finish()


def snapshot_on(device, count=16):
    return EpisodeSnapshot.create(
        **source(count),
        cutoff=AT,
        codec_id=CODEC,
        context_id=CONTEXT,
        dtype=torch.float64,
        device=device,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("count", [0, 16])
def test_shared_regional_prefixes_match_each_route_and_preserve_gradients(device, count):
    reader = regional(device).readout
    reader.requires_grad_(True)
    snapshot = snapshot_on(device, count)
    base = torch.linspace(-1, 1, 64, dtype=torch.float64, device=device).reshape(2, 32)
    base.requires_grad_(True)
    expected = {
        route: regional_routes.run_route_pair(
            reader,
            base,
            snapshot,
            route=route,
            context_id=CONTEXT,
            cutoff=AT,
            differentiable=True,
        )[1]
        for route in ROUTES
    }
    actual = regional_routes.collect_routes(
        reader, base, snapshot, context_id=CONTEXT, cutoff=AT, differentiable=True
    )
    assert actual.refinements == 9
    assert actual.searches == (3 if count else 0)
    assert actual.value_projections == (9 if count else 0)
    for route in ROUTES:
        assert torch.equal(actual.routes[route].state, expected[route].state)
        assert actual.routes[route].searches == expected[route].searches
        assert actual.routes[route].value_projections == expected[route].value_projections
    parameters = (base, *reader.parameters())
    left = torch.autograd.grad(
        sum(r.state.square().sum() for r in actual.routes.values()), parameters, allow_unused=True
    )
    right = torch.autograd.grad(
        sum(r.state.square().sum() for r in expected.values()), parameters, allow_unused=True
    )
    for a, b in zip(left, right, strict=True):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("count", [0, 16])
def test_regional_collection_matches_all_fixed_emissions_without_mutating_prefix(device, count):
    engine, snapshot = regional(device), snapshot_on(device, count)
    before = snapshot.export_cpu()["sha256"]
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone() if device != "cpu" else None
    packet = collect(engine, snapshot)
    body = packet.payload()
    assert body["catalog"]["reader_contract"] == CONTRACT
    assert CollectedMechanisms.restore(packet.record()).record() == packet.record()
    assert body["work"]["shared_route_batches"] == 2
    assert body["work"]["reader_refinements"] == 36
    assert body["work"]["reader_searches"] == (12 if count else 0)
    assert body["work"]["reader_value_projections"] == (36 if count else 0)
    batch = DecisionBatch.from_corpus(
        raw_batch(), engine.predictor.config.inputs, device=device, dtype=torch.float64
    )
    for candidate in body["groups"][0]["candidates"]:
        assert candidate["status"] == "observed"
        event = engine.begin(
            engine.initial_state(),
            flow_ids=batch.flow_ids,
            cutoff=AT,
            context_id=CONTEXT,
            matures_at=(AT + DAY,) * 2,
            snapshot=snapshot,
            action=MechanismAction(*candidate["requested"].split("/")),
        )
        result = engine.finish(
            engine.prepare(batch, engine.predictor.initial_state(batch.flow_ids), event).event
        )
        np.testing.assert_allclose(candidate["quantiles"], result.quantiles, rtol=1e-12, atol=1e-12)
    assert snapshot.export_cpu()["sha256"] == before
    assert torch.equal(torch.get_rng_state(), rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(0), cuda_rng)


@pytest.mark.parametrize("device", DEVICES)
def test_regional_collection_preserves_outputs_when_shared_prefixes_do_not_fit(device):
    full = regional(device)
    cap = (
        full.readout.estimated_bytes(2, 16, refinements=4)
        + full.readout.estimated_bytes(2, 16, refinements=9)
    ) // 2
    limited = regional(device, limit=cap)
    snapshot = snapshot_on(device)
    first, second = collect(full, snapshot).payload(), collect(limited, snapshot).payload()
    assert second["work"]["shared_route_batches"] == 0
    assert second["work"]["individual_route_batches"] == 12
    assert second["work"]["reader_refinements"] == second["work"]["reader_value_projections"] == 68
    assert second["work"]["reader_searches"] == 36
    assert first["groups"] == second["groups"]
    split = collect(full, snapshot, split=True).payload()
    assert split["observation"] == first["observation"]
    for a, b in zip(
        first["groups"][0]["candidates"], split["groups"][0]["candidates"], strict=True
    ):
        np.testing.assert_allclose(a["quantiles"], b["quantiles"], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("contract", ["legacy", CONTRACT])
def test_catalogue_restore_rejects_changed_semantics_and_preserves_identity(contract):
    catalog = MechanismCatalog(reader_contract=contract)
    assert MechanismCatalog.restore(catalog.identity()).fingerprint() == catalog.fingerprint()
    for key, value in (
        ("schema_version", True),
        ("arbitrary_code_allowed", 0),
        ("fallback", "other"),
        ("reader_contract", "unknown"),
    ):
        changed = deepcopy(catalog.identity())
        changed[key] = value
        with pytest.raises(ValueError):
            MechanismCatalog.restore(changed)


def test_resealed_regional_packet_cannot_claim_cached_value_projection_costs():
    body = collect(regional(), snapshot_on("cpu")).payload()
    body["work"]["reader_value_projections"] = body["work"]["reader_searches"]
    with pytest.raises(ValueError, match="proyecciones|regional"):
        CollectedMechanisms(body)


@pytest.mark.parametrize(
    "identity", [None, {}, {"budget": None}, {"budget": {}}, {"budget": {"max_reader_steps": True}}]
)
def test_catalogue_restore_rejects_missing_or_untyped_contracts(identity):
    with pytest.raises(ValueError):
        MechanismCatalog.restore(identity)


def test_regional_collection_without_a_bank_does_not_invent_projection_work():
    packet = collect(regional(populated=False))
    assert packet.payload()["work"]["reader_value_projections"] == 0
    assert packet.payload()["work"]["reader_searches"] == 0
    assert packet.payload()["work"]["reader_refinements"] == 36


def test_shared_route_api_rejects_legacy_readers_and_insufficient_working_memory():
    base = torch.zeros((2, 32), dtype=torch.float64)
    legacy = consumer(reader=True).readout
    with pytest.raises(ValueError, match="regional"):
        regional_routes.collect_routes(legacy, base, context_id=CONTEXT, cutoff=AT)
    full = regional()
    cap = full.readout.estimated_bytes(2, 16, refinements=9) - 1
    with pytest.raises(ValueError, match="presupuesto|memoria"):
        regional_routes.collect_routes(
            regional(limit=cap).readout, base, snapshot_on("cpu"), context_id=CONTEXT, cutoff=AT
        )
