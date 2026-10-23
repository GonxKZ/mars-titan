"""Fábrica, trazas y recuperación del intercambio jerárquico sobre fixtures."""

import json
from dataclasses import replace

import pytest
import torch
from test_bridge_episodic_readout import CODEC, CONTEXT, bank
from test_bridge_hierarchy_contracts import (
    CONTRACT,
    DEVICES,
    hierarchy_config,
    hierarchy_reader,
)
from test_bridge_mechanism_session import engines
from test_financial_mechanisms import assert_same, session
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_market_memory import pt4
from test_mechanism_observations import AT, DAY, observation_rows
from test_native_episode_backend import native as native
from test_shared_bridge_variant import GEOMETRY, selected

from mars_titan.memory import mars_titan_variant as variants
from mars_titan.memory import regional_episode_trace as episode_trace
from mars_titan.memory.mechanism_collection import MechanismCollector
from mars_titan.models.titans import regional_mechanism_trace as mechanism_trace
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.regional_routes import run_route_pair


def test_factory_preserves_the_typed_hierarchy_and_omits_null():
    model = pt4(None).eval().requires_grad_(False)
    geometry = dict(
        GEOMETRY,
        regions=["view_1", "view_2", "view_3", "view_4"],
        topology="self",
        cycles=0,
        hierarchy=dict(
            topology="star", local_steps=2, feedback=False, max_node_updates_per_row=988
        ),
    )
    variant = selected(model, geometry=geometry, k=4)
    rebuilt = variants.select_variant(
        variants.load_declaration(), json.loads(json.dumps(variant.components)), base=variant.base
    )
    config = rebuilt.readout_config(CODEC, hidden_size=32)
    assert config.identity()["evidence_bridge"]["kind"] == CONTRACT
    assert config.evidence_bridge.hierarchy.feedback is False
    assert config.evidence_bridge.hierarchy.topology == "star"
    assert rebuilt.fingerprint() == variant.fingerprint()
    reader = EpisodicReadout(config, dtype=torch.float64).eval().requires_grad_(False)
    assert rebuilt.consumer(model, reader).readout is reader
    geometry["hierarchy"]["local_steps"] = 4
    assert rebuilt.readout_config(CODEC, hidden_size=32).evidence_bridge.hierarchy.local_steps == 2
    plain = selected(model)
    explicit = selected(model, geometry=dict(GEOMETRY, hierarchy=None))
    assert explicit.identity() == plain.identity()
    assert "hierarchy" not in explicit.components["evidence_bridge"]
    assert (
        "hierarchy" not in explicit.readout_config(CODEC, hidden_size=32).evidence_bridge.identity()
    )


@pytest.mark.parametrize("mode,count", [("bank", 8), ("bank", 0), ("no_bank", 0)])
@pytest.mark.parametrize("feedback", [True, False])
@pytest.mark.parametrize("topology", ["tree", "star"])
@pytest.mark.parametrize("device", DEVICES)
def test_both_traces_capture_actual_auxiliaries_and_the_effective_graph(
    mode, count, feedback, topology, device
):
    reader = hierarchy_reader(device, k=4, mode=mode, feedback=feedback, topology=topology)
    snapshot = None if mode == "no_bank" else bank(device, count)
    state = torch.linspace(-0.5, 0.5, 64, dtype=torch.float64, device=device).reshape(2, 32)
    result = reader(state, snapshot, context_id=CONTEXT, cutoff=10)
    flows = ("US/AAA", "US/BBB")
    trace = episode_trace.begin(
        reader, flows, snapshot, context_id=CONTEXT, cutoff=10, block_rows=2
    )
    episode_trace.append(trace, flows, result)
    assert trace["kind"] == "hierarchical_episode_trace_v1"
    episode_trace.verify(
        trace, reader, flows=flows, context_id=CONTEXT, cutoff=10, refinements=4, block_rows=2
    )
    expected_local = torch.eye(4, dtype=torch.float64) * (1.0 if feedback is False else 0.5)
    assert torch.equal(trace["bridge_weights"], expected_local)
    assert trace["hierarchy_weights"].shape == (7, 7)
    expected_root = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 1 / 3, 1 / 3, 1 / 3] if topology == "tree" else [1 / 7] * 7,
        dtype=torch.float64,
    )
    assert torch.equal(trace["hierarchy_weights"][6], expected_root)
    if feedback:
        assert trace["hierarchy_weights"][0, 4 if topology == "tree" else 6] == 0.5
    for raw, actual in zip(trace["blocks"][0]["reads"], result.reads, strict=True):
        if actual.bridge is not None:
            assert torch.equal(raw["group_values"], actual.bridge.group_values.cpu())
            assert torch.equal(raw["root_values"], actual.bridge.root_values.cpu())
            assert torch.equal(actual.bridge.bridge_weights[0].cpu(), expected_local)
            assert torch.equal(actual.bridge.hierarchy_weights[0, 6].cpu(), expected_root)
        else:
            assert (
                not raw["group_values"].count_nonzero() and not raw["root_values"].count_nonzero()
            )
    _, routed = run_route_pair(reader, state, snapshot, route="k4cc", context_id=CONTEXT, cutoff=10)
    rows = mechanism_trace.export(routed, snapshot, reader=reader)
    for row in rows:
        assert row["kind"] == "hierarchical_mechanism_trace_v1"
        assert row["bridge_weights"] == expected_local.tolist()
        assert row["hierarchy_weights"][6] == expected_root.tolist()
        mechanism_trace.verify(
            row,
            reader,
            snapshot_sha256=snapshot._digest if snapshot else None,
            steps=4,
            searches=int(count > 0),
            route="k4cc",
        )
    if count:
        for index, row in enumerate(rows):
            assert (
                row["reads"][0]["group_values"]
                == routed.reads[0].bridge.group_values[index].tolist()
            )
            assert (
                row["reads"][0]["root_values"] == routed.reads[0].bridge.root_values[index].tolist()
            )


@pytest.mark.parametrize(
    "fault", ["group_shape", "root_nan", "graph", "local_graph", "legacy_kind", "fields"]
)
def test_forged_hierarchical_archives_cannot_pass_as_valid_or_legacy(fault):
    reader = hierarchy_reader()
    snapshot = bank("cpu")
    state = torch.ones((1, 32), dtype=torch.float64)
    result = reader(state, snapshot, context_id=CONTEXT, cutoff=10)
    flows = ("US/AAA",)
    trace = episode_trace.begin(
        reader, flows, snapshot, context_id=CONTEXT, cutoff=10, block_rows=1
    )
    episode_trace.append(trace, flows, result)
    read = trace["blocks"][0]["reads"][0]
    if fault == "group_shape":
        read["group_values"] = read["group_values"][:, :1]
    elif fault == "root_nan":
        read["root_values"][0, 0, 0] = float("nan")
    elif fault == "graph":
        trace["hierarchy_weights"][0, 4] = 0.0
    elif fault == "local_graph":
        trace["bridge_weights"] = torch.eye(4, dtype=torch.float64)
    elif fault == "legacy_kind":
        trace["kind"] = "regional_episode_trace_v1"
    else:
        read["invented"] = torch.zeros(1)
    with pytest.raises(ValueError):
        episode_trace.verify(
            trace, reader, flows=flows, context_id=CONTEXT, cutoff=10, refinements=1, block_rows=1
        )
    _, routed = run_route_pair(reader, state, snapshot, route="k1", context_id=CONTEXT, cutoff=10)
    row = mechanism_trace.export(routed, snapshot, reader=reader)[0]
    if fault == "group_shape":
        row["reads"][0]["group_values"] = [[]]
    elif fault == "root_nan":
        row["reads"][0]["root_values"][0][0] = float("nan")
    elif fault == "graph":
        row["hierarchy_weights"][0][4] = 0.0
    elif fault == "local_graph":
        row["bridge_weights"] = torch.eye(4, dtype=torch.float64).tolist()
    elif fault == "legacy_kind":
        row["kind"] = "regional_mechanism_trace_v1"
    else:
        row["reads"][0]["invented"] = []
    with pytest.raises(ValueError):
        mechanism_trace.verify(
            row, reader, snapshot_sha256=snapshot._digest, steps=1, searches=1, route="k1"
        )


def test_hierarchical_export_needs_the_reader_even_when_no_bank_is_present():
    reader = hierarchy_reader(mode="no_bank")
    state = torch.ones((1, 32), dtype=torch.float64)
    _, routed = run_route_pair(reader, state, None, route="k1", context_id=CONTEXT, cutoff=10)
    row = mechanism_trace.export(routed, None, reader=reader)[0]
    assert row["kind"] == "hierarchical_mechanism_trace_v1"
    assert row["reads"][0]["group_values"] == []
    populated = hierarchy_reader()
    _, result = run_route_pair(
        populated, state, bank("cpu"), route="k1", context_id=CONTEXT, cutoff=10
    )
    with pytest.raises(ValueError):
        mechanism_trace.export(result, bank("cpu"))


def test_hierarchical_trace_is_detached_and_cannot_be_loaded_by_a_flat_reader():
    reader = hierarchy_reader()
    snapshot = bank("cpu")
    state = torch.ones((1, 32), dtype=torch.float64)
    result = reader(state, snapshot, context_id=CONTEXT, cutoff=10)
    flows = ("US/AAA",)
    trace = episode_trace.begin(
        reader, flows, snapshot, context_id=CONTEXT, cutoff=10, block_rows=1
    )
    episode_trace.append(trace, flows, result)
    expected = trace["blocks"][0]["reads"][0]["group_values"].clone()
    with torch.no_grad():
        result.reads[0].bridge.group_values.add_(1.0)
    assert torch.equal(trace["blocks"][0]["reads"][0]["group_values"], expected)
    flat = EpisodicReadout(
        replace(
            reader.config, evidence_bridge=replace(reader.config.evidence_bridge, hierarchy=None)
        ),
        dtype=torch.float64,
    )
    with pytest.raises(ValueError):
        episode_trace.verify(
            trace, flat, flows=flows, context_id=CONTEXT, cutoff=10, refinements=1, block_rows=1
        )
    legacy_result = flat(state, snapshot, context_id=CONTEXT, cutoff=10)
    legacy = episode_trace.begin(flat, flows, snapshot, context_id=CONTEXT, cutoff=10, block_rows=1)
    episode_trace.append(legacy, flows, legacy_result)
    with pytest.raises(ValueError):
        episode_trace.verify(
            legacy, reader, flows=flows, context_id=CONTEXT, cutoff=10, refinements=1, block_rows=1
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "steps,budget,allowed", [(2, 988, True), (2, 987, False), (4, 1596, True), (4, 1595, False)]
)
def test_collection_reserves_hierarchy_and_final_emission_before_mac(
    device, steps, budget, allowed, monkeypatch
):
    from test_mechanism_consumer import calibration
    from test_regional_mechanism_collection import regional, snapshot_on

    parent = regional(device)
    calls = []
    original = FinancialPredictor.prepare

    def observed(*args, **options):
        calls.append(True)
        return original(*args, **options)

    monkeypatch.setattr(FinancialPredictor, "prepare", observed)
    engine = FrozenMechanismConsumer(
        parent.predictor,
        readout=hierarchy_reader(device, budget=budget, steps=steps),
        calibration=calibration(),
        catalog=MechanismCatalog(reader_contract=CONTRACT),
        action=MechanismAction(route="k4cc"),
    )
    rows = observation_rows(engine)
    options = dict(
        rows=rows,
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + DAY,) * len(rows),
        snapshot=snapshot_on(device),
    )
    if allowed:
        collector = MechanismCollector(engine, engine.initial_state(), **options)
        assert collector.emission_event().snapshot is options["snapshot"]
        assert engine.readout.recurrence_work(22)["bridge_node_updates"] == (
            572 if steps == 2 else 924
        )
    else:
        with pytest.raises(ValueError, match="presupuesto.*MAC"):
            MechanismCollector(engine, engine.initial_state(), **options)
    assert calls == []


def test_archive_budgets_include_auxiliaries_before_admitting_any_mac(monkeypatch):
    from mars_titan.models.quantile_head import QUANTILE_HEAD

    active = hierarchy_reader(k=4)
    flat = EpisodicReadout(
        replace(
            active.config, evidence_bridge=replace(active.config.evidence_bridge, hierarchy=None)
        ),
        dtype=torch.float64,
    )
    flows = tuple(f"US/F{i:04d}" for i in range(64))
    plain_bytes = episode_trace.estimated_bytes(flat, flows, 8, 1)
    active_bytes = episode_trace.estimated_bytes(active, flows, 8, 1)
    assert active_bytes >= plain_bytes + 64 * 4 * 3 * 32 * 8
    limit = (plain_bytes + active_bytes) // 2
    config = replace(
        active.config,
        max_working_bytes=limit,
        evidence_bridge=replace(active.config.evidence_bridge, max_working_bytes=limit),
    )
    limited = EpisodicReadout(config, dtype=torch.float64)
    with pytest.raises(ValueError, match="antes de emitir"):
        episode_trace.begin(
            limited, flows, bank("cpu"), context_id=CONTEXT, cutoff=10, block_rows=1
        )
    json_plain = mechanism_trace.estimated_bytes(flat, 64, 8, 1)
    json_active = mechanism_trace.estimated_bytes(active, 64, 8, 1)
    assert json_active >= json_plain + 64 * 3 * 32 * 40
    limit = (json_plain + json_active) // 2
    config = hierarchy_config()
    config = replace(
        config,
        max_batch=1,
        max_working_bytes=limit,
        evidence_bridge=replace(config.evidence_bridge, max_batch=1, max_working_bytes=limit),
    )
    readout = EpisodicReadout(config, dtype=torch.float64).eval().requires_grad_(False)

    def forbidden(*args, **kwargs):
        raise AssertionError("No debe comenzar MAC con una traza que no cabe")

    monkeypatch.setattr(FinancialPredictor, "prepare", forbidden)
    engine = FrozenMechanismConsumer(
        pt4(None, head=QUANTILE_HEAD).eval().requires_grad_(False),
        readout=readout,
        catalog=MechanismCatalog(reader_contract=CONTRACT),
    )
    with pytest.raises(ValueError, match="traza regional.*MAC"):
        engine.begin(
            engine.initial_state(),
            flow_ids=flows,
            cutoff=10,
            context_id=CONTEXT,
            matures_at=(20,) * 64,
            snapshot=bank("cpu"),
        )


@pytest.mark.parametrize("shared", [True, False])
def test_actual_collection_and_emission_share_the_anchor_and_reconcile_dense_work(shared):
    from test_financial_adapter import raw_batch
    from test_mechanism_consumer import calibration
    from test_regional_mechanism_collection import regional, snapshot_on

    from mars_titan.memory.mechanism_collection import CollectedMechanisms
    from mars_titan.models.titans.financial_inputs import DecisionBatch

    parent = regional()
    reader = hierarchy_reader()
    if not shared:
        limit = (
            reader.estimated_bytes(2, 16, refinements=4)
            + reader.estimated_bytes(2, 16, refinements=9)
        ) // 2
        reader = (
            EpisodicReadout(
                replace(
                    reader.config,
                    max_working_bytes=limit,
                    evidence_bridge=replace(reader.config.evidence_bridge, max_working_bytes=limit),
                ),
                dtype=torch.float64,
            )
            .eval()
            .requires_grad_(False)
        )
    engine = FrozenMechanismConsumer(
        parent.predictor,
        readout=reader,
        calibration=calibration(),
        catalog=MechanismCatalog(reader_contract=CONTRACT),
        action=MechanismAction("parent", "k4cc"),
    )
    rows = observation_rows(engine)
    snapshot = snapshot_on("cpu")
    builder = MechanismCollector(
        engine,
        engine.initial_state(),
        rows=rows,
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + DAY,) * len(rows),
        snapshot=snapshot,
    )
    anchor = builder.emission_event()
    event = anchor
    batch = DecisionBatch.from_corpus(
        raw_batch(), engine.predictor.config.inputs, dtype=torch.float64
    )
    for block in (rows[1:], rows[:1]) if shared else (rows,):
        fast = engine.predictor.initial_state(tuple(row.flow_id for row in block))
        builder.add(block, fast)
        positions = [batch.flow_ids.index(row.flow_id) for row in block]
        event = engine.prepare(batch.select(positions), fast, event).event
        assert builder.emission_event() is anchor and anchor.rows == ()
        assert event is not anchor and anchor.snapshot is snapshot
    emitted = engine.finish(event)
    packet = builder.finish()
    body = packet.payload()
    assert body["work"]["reader_value_projections"] == (18 if shared else 34) * len(rows)
    assert body["work"]["individual_route_batches"] == (0 if shared else 12)
    assert "bridge_group_updates" not in body["work"]
    assert CollectedMechanisms.restore(packet.record()).record() == packet.record()
    projections = (
        body["work"]["reader_value_projections"] + emitted.work["bridge_value_projections"]
    )
    assert reader.recurrence_work(projections) == dict(
        bridge_initial_views=(88 if shared else 152) * len(rows),
        bridge_local_updates=(352 if shared else 608) * len(rows),
        bridge_group_updates=(176 if shared else 304) * len(rows),
        bridge_root_updates=(44 if shared else 76) * len(rows),
        bridge_node_updates=(572 if shared else 988) * len(rows),
    )


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("device", DEVICES)
def test_native_recovery_keeps_hierarchy_auxiliaries_and_the_next_emission(
    native, four_flow_source, tmp_path, device
):
    parent = engines(four_flow_source, device)
    config = hierarchy_config()
    codec = four_flow_source["codec"].fingerprint()
    config = replace(
        config, codec_id=codec, evidence_bridge=replace(config.evidence_bridge, codec_id=codec)
    )
    reader = (
        EpisodicReadout(config, dtype=torch.float64, device=device).eval().requires_grad_(False)
    )
    engine = FrozenMechanismConsumer(
        parent[3].predictor,
        readout=reader,
        catalog=MechanismCatalog(reader_contract=CONTRACT),
        action=MechanismAction(route="k4cc"),
    )
    data = (*parent[:3], engine, parent[4])
    with session(native, tmp_path / "reference", data, block_rows=2) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        first = run.step(four_flow_source["batches"][126], [])
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in first.predictions]
        run.step([], labels, kind="settlement", cutoff=moment(127))
        expected = run.step(four_flow_source["batches"][128], [])
        expected_fast, expected_event = run.fast_state(), run.mechanism_event(4)
        assert expected_event["payload"]["work"]["bridge_node_updates"] == 416
        assert expected_event["payload"]["work"]["bridge_group_updates"] == 128
        assert expected_event["payload"]["rows"][0]["regional_readout"]["reads"][0]["root_values"]
    with session(native, tmp_path / "cut", data, block_rows=2) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        run.step(four_flow_source["batches"][126], [])

        def cut(boundary):
            if boundary == native.Boundary.committed:
                raise RuntimeError("Corte tras confirmar feedback jerárquico")

        with pytest.raises(RuntimeError, match="Corte tras"):
            run.step([], labels, kind="settlement", cutoff=moment(127), fault=cut)
    with session(native, tmp_path / "cut", data, resume=True, block_rows=2) as run:
        actual = run.step(four_flow_source["batches"][128], [])
        assert [(p.asset, p.value) for p in actual.predictions] == [
            (p.asset, p.value) for p in expected.predictions
        ]
        assert_same(run.fast_state(), expected_fast)
        assert_same(run.mechanism_event(4), expected_event)
        assert run.snapshot()["applied"] == 4


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_direct_native_session_recovers_the_detailed_hierarchical_work(
    native, four_flow_source, tmp_path
):
    from test_bridge_readout_session import control, engine
    from test_bridge_readout_session import session as direct_session

    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

    parent = engine(four_flow_source, "cpu")
    config = hierarchy_config(k=2)
    codec = four_flow_source["codec"].fingerprint()
    config = replace(
        config, codec_id=codec, evidence_bridge=replace(config.evidence_bridge, codec_id=codec)
    )
    reader = EpisodicReadout(config, dtype=torch.float64).eval().requires_grad_(False)
    current = FrozenFinancialConsumer(parent.predictor, readout=reader)
    output = tmp_path / "direct"
    with direct_session(native, output, four_flow_source, current) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        first = run.step(four_flow_source["batches"][126], [])
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in first.predictions]
        run.step([], labels, kind="settlement", cutoff=moment(127))
        run.step(four_flow_source["batches"][128], [])
        expected_control, expected_fast = control(run), run.fast_state()
        assert expected_control["bridge_value_projections"] == 8
        assert expected_control["bridge_node_updates"] == 208
        assert expected_control["bridge_local_updates"] == 128
        assert expected_control["bridge_group_updates"] == 64
        assert expected_control["bridge_root_updates"] == 16
    with direct_session(native, output, four_flow_source, current, resume=True) as run:
        assert_same(control(run), expected_control)
        assert_same(run.fast_state(), expected_fast)
