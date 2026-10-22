"""Contratos y presupuestos del puente no lineal sobre fixtures, sin ajustes."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_bridge_controller_contracts import bound_policy
from test_bridge_episodic_readout import CODEC, CONTEXT, bank, configuration
from test_context_selection import batch_on
from test_financial_adapter import raw_batch
from test_financial_mechanisms import assert_same
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_market_memory import pt4
from test_mechanism_consumer import calibration
from test_mechanism_observations import AT, DAY, observation_rows
from test_native_episode_backend import native as native
from test_regional_mechanism_collection import regional, snapshot_on
from test_shared_bridge_variant import GEOMETRY, selected

from mars_titan.memory import mars_titan_variant as variants
from mars_titan.memory.mechanism_collection import CollectedMechanisms, MechanismCollector
from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer, _implementation
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer, _code

CONTRACT = "shared_evidence_bridge_nonlinear_v1"
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def nonlinear_config(*, k=1, regions=3, steps=2, budget=1216, limit=None):
    """Conservar las formas compartidas y fijar explícitamente J y su cuota."""
    config = configuration(refinements=k)
    bridge = replace(
        config.evidence_bridge,
        regions=tuple(f"view_{i}" for i in range(regions)),
        cycles=0,
        recurrence=BridgeRecurrenceConfig(steps, budget),
    )
    if limit is not None:
        bridge = replace(bridge, max_working_bytes=limit)
    return replace(
        config,
        evidence_bridge=bridge,
        max_working_bytes=config.max_working_bytes if limit is None else limit,
    )


def nonlinear_engine(device="cpu", *, regions=3, steps=2, budget=1216, limit=None, context="off"):
    """Reutilizar el predictor de contexto y calibración del control regional existente."""
    parent = regional(device)
    reader = (
        EpisodicReadout(
            nonlinear_config(regions=regions, steps=steps, budget=budget, limit=limit),
            device=device,
            dtype=torch.float64,
        )
        .eval()
        .requires_grad_(False)
    )
    return FrozenMechanismConsumer(
        parent.predictor,
        readout=reader,
        calibration=calibration(),
        action=MechanismAction(context, "k4cr"),
        catalog=MechanismCatalog(reader_contract=CONTRACT),
    )


def test_variant_roundtrip_preserves_optional_recurrence_and_legacy_null():
    parent = regional()
    geometry = dict(GEOMETRY, cycles=0, recurrence=dict(steps=2, max_node_updates_per_row=228))
    active = selected(parent.predictor, geometry=geometry, k=4)
    rebuilt = variants.select_variant(
        variants.load_declaration(), json.loads(json.dumps(active.components)), base=active.base
    )
    assert rebuilt.fingerprint() == active.fingerprint()
    config = rebuilt.readout_config(CODEC, hidden_size=32)
    assert type(config.evidence_bridge.recurrence) is BridgeRecurrenceConfig
    assert config.evidence_bridge.recurrence == BridgeRecurrenceConfig(2, 228)
    assert config.identity()["evidence_bridge"]["kind"] == CONTRACT
    geometry["recurrence"]["steps"] = 4
    assert active.components["evidence_bridge"]["recurrence"]["steps"] == 2
    plain = selected(parent.predictor)
    null = selected(parent.predictor, geometry=dict(GEOMETRY, recurrence=None))
    assert plain.identity() == null.identity()
    assert "recurrence" not in null.components["evidence_bridge"]
    assert (
        "recurrence" not in null.readout_config(CODEC, hidden_size=32).identity()["evidence_bridge"]
    )


@pytest.mark.parametrize(
    "value",
    [
        False,
        {},
        {"steps": 2},
        {"steps": True, "max_node_updates_per_row": 228},
        {"steps": 3, "max_node_updates_per_row": 228},
        {"steps": 2, "max_node_updates_per_row": 1217},
        {"steps": 2, "max_node_updates_per_row": 228, "other": 1},
    ],
)
def test_parser_rejects_untyped_or_incomplete_recurrence(value):
    parent = regional()
    with pytest.raises(ValueError):
        selected(parent.predictor, geometry=dict(GEOMETRY, cycles=0, recurrence=value))


def test_nonlinear_reader_rejects_legacy_catalog_and_policy():
    parent = regional()
    reader = EpisodicReadout(nonlinear_config(), dtype=torch.float64).eval().requires_grad_(False)
    catalog = MechanismCatalog(reader_contract=CONTRACT)
    assert MechanismCatalog.restore(catalog.identity()).fingerprint() == catalog.fingerprint()
    for old in (MechanismCatalog(), parent.catalog):
        with pytest.raises(ValueError, match="catálogo"):
            FrozenMechanismConsumer(parent.predictor, readout=reader, catalog=old)
        with pytest.raises(ValueError):
            FrozenMechanismConsumer(
                parent.predictor,
                readout=reader,
                catalog=catalog,
                policy=bound_policy(old.fingerprint()),
            )
    FrozenMechanismConsumer(
        parent.predictor,
        readout=reader,
        catalog=catalog,
        policy=bound_policy(catalog.fingerprint()),
    )


def test_new_module_is_sealed_in_both_runtime_implementations():
    name = "mars_titan.models.titans.bridge_recurrence"
    assert name in _implementation(), "Falta sellar la fuente de la recurrencia"
    assert _code()[name] == _implementation()[name]


def test_active_identity_describes_the_nonlinear_receiver_transition():
    config = nonlinear_config().evidence_bridge
    assert config.identity()["bridge"] == "synchronous_receiver_weights_reused_mixture_tanh"
    assert (
        replace(config, recurrence=None).identity()["bridge"]
        == "convex_messages_between_derived_views"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_j_and_k_share_one_node_budget_in_the_direct_reader(device):
    base = torch.zeros((2, 32), dtype=torch.float64, device=device)
    snapshot = bank(device)
    for budget in (128, 127):
        reader = EpisodicReadout(
            nonlinear_config(k=4, regions=8, steps=4, budget=budget),
            device=device,
            dtype=torch.float64,
        )
        if budget == 128:
            assert len(reader(base, snapshot, context_id=CONTEXT, cutoff=10).reads) == 4
        else:
            with pytest.raises(ValueError, match="presupuesto"):
                reader(base, snapshot, context_id=CONTEXT, cutoff=10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("budget", [128, 127])
def test_frozen_budget_rejection_precedes_the_first_mac(device, monkeypatch, budget):
    model = pt4(None).to(device).eval().requires_grad_(False)
    batch = batch_on(model, ("US/AAA", "US/BBB"), 0)
    snapshot = bank(device, cutoff=batch.prediction_at[0])
    reader = (
        EpisodicReadout(
            nonlinear_config(k=4, regions=8, steps=4, budget=budget),
            device=device,
            dtype=torch.float64,
        )
        .eval()
        .requires_grad_(False)
    )
    called = []
    original = FinancialPredictor.prepare

    def recorded(*args, **options):
        called.append(True)
        return original(*args, **options)

    monkeypatch.setattr(FinancialPredictor, "prepare", recorded)
    consumer = FrozenFinancialConsumer(model, readout=reader)
    previous = model.initial_state(batch.flow_ids)
    if budget == 128:
        result = consumer.prepare(batch, previous, context_id=CONTEXT, snapshot=snapshot)
        assert result.next_state.mac.memory.steps.tolist() == [1, 1]
        assert len(result.readout.reads) == 4
    else:
        with pytest.raises(ValueError, match="presupuesto"):
            consumer.prepare(batch, previous, context_id=CONTEXT, snapshot=snapshot)
    assert len(called) == int(budget == 128), "El rechazo debe preceder a cualquier MAC"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("budget,route", [(128, "k4cr"), (127, "k1")])
def test_mechanism_fallback_keeps_j_and_one_mac_update(device, budget, route):
    consumer = nonlinear_engine(device, regions=8, steps=4, budget=budget)
    batch = batch_on(consumer.predictor, ("US/AAA", "US/BBB"), 0)
    at = batch.prediction_at[0]
    snapshot = bank(device, cutoff=at)
    event = consumer.begin(
        consumer.initial_state(),
        flow_ids=batch.flow_ids,
        cutoff=at,
        context_id=CONTEXT,
        matures_at=(at + 10,) * len(batch.flow_ids),
        snapshot=snapshot,
    )
    plan = dict(event.plans)["US"]
    assert plan.applied.route == route
    assert plan.reasons == (() if budget == 128 else ("reader_budget",))
    block = consumer.prepare(batch, consumer.predictor.initial_state(batch.flow_ids), event)
    emitted = consumer.finish(block.event)
    assert block.next_state.mac.memory.steps.tolist() == [1, 1]
    assert consumer.readout.config.evidence_bridge.recurrence.steps == 4
    assert emitted.work["bridge_node_updates"] == len(batch.flow_ids) * 8 * 4 * int(route[1])
    assert emitted.work["bridge_initial_views"] == len(batch.flow_ids) * 8 * int(route[1])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("regions,steps,budget", [(3, 2, 228), (8, 4, 1216)])
def test_collection_reserves_both_contexts_and_final_emission_before_mac(
    device, split, monkeypatch, regions, steps, budget
):
    called = []
    original = FinancialPredictor.prepare

    def recorded(*args, **options):
        called.append(True)
        return original(*args, **options)

    monkeypatch.setattr(FinancialPredictor, "prepare", recorded)
    consumer = nonlinear_engine(device, regions=regions, steps=steps, budget=budget - 1)
    rows = observation_rows(consumer)
    with pytest.raises(ValueError, match="presupuesto.*MAC"):
        MechanismCollector(
            consumer,
            consumer.initial_state(),
            rows=rows,
            cutoff=AT,
            context_id=CONTEXT,
            matures_at=(AT + DAY,) * len(rows),
            snapshot=snapshot_on(device),
        )
    assert called == [], "La reserva combinada debe preceder a cualquier MAC"
    consumer = nonlinear_engine(
        device, regions=regions, steps=steps, budget=budget, context="parent"
    )
    builder = MechanismCollector(
        consumer,
        consumer.initial_state(),
        rows=rows,
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + DAY,) * len(rows),
        snapshot=snapshot_on(device),
    )
    event = builder.emission_event()
    assert dict(event.plans)["US"].applied.route == "k4cr"
    emission_event = event
    batch = DecisionBatch.from_corpus(
        raw_batch(), consumer.predictor.config.inputs, device=device, dtype=torch.float64
    )
    for block in (rows[1:], rows[:1]) if split else (rows,):
        fast = consumer.predictor.initial_state(tuple(row.flow_id for row in block))
        builder.add(block, fast)
        positions = [batch.flow_ids.index(row.flow_id) for row in block]
        emission_event = consumer.prepare(batch.select(positions), fast, emission_event).event
        assert builder.emission_event() is event and event.rows == ()
        assert emission_event is not event and len(emission_event.rows) > 0
    emission = consumer.finish(emission_event)
    assert event.context.committed_at != emission.state.context.committed_at
    packet = builder.finish()
    body = packet.payload()
    assert len(called) == (6 if split else 3)
    assert body["work"]["reader_value_projections"] == 18 * len(rows)
    assert CollectedMechanisms.restore(packet.record()).record() == packet.record()
    work = consumer.readout.recurrence_work(
        body["work"]["reader_value_projections"] + emission.work["bridge_value_projections"]
    )
    assert work == dict(
        bridge_initial_views=regions * 22 * len(rows),
        bridge_node_updates=regions * steps * 22 * len(rows),
    )
    body["work"]["reader_value_projections"] = body["work"]["reader_searches"]
    with pytest.raises(ValueError, match="proyecciones|regional"):
        CollectedMechanisms(body)


@pytest.mark.parametrize("device", DEVICES)
def test_collection_budget_handles_individual_routes_when_nine_steps_do_not_fit(device):
    full = nonlinear_engine(device, budget=228)
    cap = (
        full.readout.estimated_bytes(2, 16, refinements=4)
        + full.readout.estimated_bytes(2, 16, refinements=9)
    ) // 2
    limited = nonlinear_engine(device, budget=228, limit=cap)
    rows = observation_rows(limited)
    builder = MechanismCollector(
        limited,
        limited.initial_state(),
        rows=rows,
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + DAY,) * len(rows),
        snapshot=snapshot_on(device),
    )
    builder.add(rows, limited.predictor.initial_state(tuple(row.flow_id for row in rows)))
    work = builder.finish().payload()["work"]
    assert work["shared_route_batches"] == 0 and work["individual_route_batches"] == 12
    assert work["reader_value_projections"] == 34 * len(rows)
    assert limited.readout.recurrence_work(work["reader_value_projections"] + 4 * len(rows))[
        "bridge_node_updates"
    ] == 228 * len(rows)


@pytest.mark.parametrize(
    "change", ["steps", "budget", "mode", "topology", "dtype", "nan", "missing"]
)
def test_recurrence_checkpoint_rejection_precedes_all_reader_parameter_copies(change):
    reader = EpisodicReadout(nonlinear_config(), dtype=torch.float64)
    before = copy.deepcopy(reader.state_dict())
    payload = copy.deepcopy(before)
    if change in ("steps", "budget", "mode", "topology"):
        config = nonlinear_config(
            steps=4 if change == "steps" else 2, budget=227 if change == "budget" else 1216
        )
        if change == "mode":
            config = replace(
                config, evidence_bridge=replace(config.evidence_bridge, recurrence=None)
            )
        if change == "topology":
            config = replace(
                config, evidence_bridge=replace(config.evidence_bridge, topology="complete")
            )
        payload = EpisodicReadout(config, dtype=torch.float64).state_dict()
    elif change == "dtype":
        payload["refinement.weight"] = payload["refinement.weight"].float()
    elif change == "nan":
        payload["refinement.weight"][0, 0] = float("nan")
    else:
        del payload["refinement.weight"]
    for name in ("step_logit", "refinement.bias"):
        payload[name] = payload[name] + 0.125
    with pytest.raises(ValueError):
        reader.load_state_dict(payload)
    assert_same(reader.state_dict(), before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["bank", "no_bank"])
def test_empty_evidence_has_zero_r0_and_j_work_without_an_implicit_bank(device, mode):
    config = replace(nonlinear_config(k=4, regions=8, steps=4, budget=1), mode=mode)
    reader = EpisodicReadout(config, device=device, dtype=torch.float64)
    snapshot = bank(device, count=0) if mode == "bank" else None
    result = reader(
        torch.zeros((2, 32), device=device, dtype=torch.float64),
        snapshot,
        context_id=CONTEXT,
        cutoff=10,
    )
    assert len(result.reads) == 4
    assert reader.recurrence_work(0) == dict(bridge_initial_views=0, bridge_node_updates=0)
    for read in result.reads:
        assert not bool(read.values.any())


@pytest.mark.parametrize("name", ["bridge_initial_views", "bridge_node_updates"])
@pytest.mark.parametrize("value", [False, 0.0, 1])
def test_recurrent_work_cannot_be_forged_with_zero_like_or_changed_values(name, value):
    from mars_titan.models.titans.regional_mechanism_trace import verify_work

    reader = EpisodicReadout(nonlinear_config(), dtype=torch.float64)
    work = dict(bridge_value_projections=0, **reader.recurrence_work(0))
    verify_work(work, reader, value_projections=0)
    work[name] = value
    with pytest.raises(ValueError, match="actualizaciones"):
        verify_work(work, reader, value_projections=0)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("device", DEVICES)
def test_native_m3_collection_partition_order_and_failed_recovery_keep_one_evidence_pool(
    native, four_flow_source, tmp_path, device, request
):
    from test_bridge_mechanism_session import engines
    from test_mars_titan_session_parity import SCALERS

    from mars_titan.memory.financial_consumers import bank_retention
    from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
    from mars_titan.memory.native_mechanism_collection import MechanismCollectionConfig

    data = engines(four_flow_source, device, recurrence=BridgeRecurrenceConfig(2, 1216))

    def session(path, *, batch=4, collected=True, resume=False, selected_data=data):
        return FinancialSession(
            path,
            native=native,
            consumer=selected_data[3],
            codec=selected_data[2],
            prefixes=selected_data[4],
            retention=bank_retention("m3", capacity=8, seed=73, scalers=SCALERS),
            phase=FinancialPhase("validation", moment(125), moment(126), moment(130), moment(131)),
            admission="m3",
            world="nonlinear_bridge_fixture",
            fold="0",
            resume=resume,
            block_rows=batch,
            mechanism_collection=MechanismCollectionConfig() if collected else None,
        )

    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone() if device == "cuda:0" else None
    outputs = []
    for batch, collected in ((1, True), (4, True), (4, False)):
        path = tmp_path / f"block-{batch}-collection-{collected}"
        with session(path, batch=batch, collected=collected) as run:
            for index, kind in ((125, "warmup"), (126, "decision")):
                issued = run.step(list(reversed(four_flow_source["batches"][index])), [], kind=kind)
            labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in issued.predictions]
            run.step([], labels, kind="settlement", cutoff=moment(127))
            later = run.step(list(reversed(four_flow_source["batches"][128])), [])
            event = run.mechanism_event(4)
            assert event["payload"]["work"]["mac_online_observations"] == 4
            assert event["payload"]["work"]["bridge_initial_views"] == 48
            assert event["payload"]["work"]["bridge_node_updates"] == 96
            flows = tuple(sorted(run.fast_state()["references"]))
            state = data[3].predictor.export_state(run._fast_store.gather(run.fast_state(), flows))
            episodes = run.retained_episodes()
            assert len(episodes) == 4 and run.snapshot()["applied"] == 4
            assert all(row["error"] == row["label"] - row["issued_prediction"] for row in episodes)
            if collected:
                packet = run.mechanism_collection(2)
                assert all(row["value"] == 0.125 for row in packet["supervision"]["outcomes"])
            outputs.append(([(p.asset, p.value) for p in later.predictions], event, state))
            confirmed = run.snapshot()
        with session(path, batch=batch, collected=collected, resume=True) as run:
            assert run.snapshot() == confirmed
            assert_same(run.mechanism_event(4), event)
            assert run.snapshot()["applied"] == 4
    maximum_partition_error = 0.0

    def compare_state(actual, expected):
        nonlocal maximum_partition_error
        if isinstance(expected, torch.Tensor) and expected.is_floating_point():
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            if expected.numel():
                maximum_partition_error = max(
                    maximum_partition_error, float((actual - expected).abs().max().item())
                )
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                compare_state(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            assert type(actual) is type(expected) and len(actual) == len(expected)
            for left, right in zip(actual, expected, strict=True):
                compare_state(left, right)
        else:
            assert_same(actual, expected)

    for output in outputs[1:]:
        assert [item[0] for item in output[0]] == [item[0] for item in outputs[0][0]]
        np.testing.assert_allclose(
            [item[1] for item in output[0]],
            [item[1] for item in outputs[0][0]],
            atol=1e-12,
            rtol=1e-12,
        )
        assert [
            row["regional_readout"]["reads"][0]["evidence_ids"]
            for row in output[1]["payload"]["rows"]
        ] == [
            row["regional_readout"]["reads"][0]["evidence_ids"]
            for row in outputs[0][1]["payload"]["rows"]
        ]
        compare_state(output[2], outputs[0][2])
    request.node.user_properties.append(("partition_max_abs_fast_state", maximum_partition_error))
    assert outputs[1][0] == outputs[2][0]
    changed = engines(four_flow_source, device, recurrence=BridgeRecurrenceConfig(4, 1216))
    before = copy.deepcopy(changed[3].readout.state_dict())
    path = tmp_path / "block-4-collection-True"
    head = (path / "latest.json").read_bytes()
    with pytest.raises(ValueError):
        session(path, resume=True, selected_data=changed)
    assert (path / "latest.json").read_bytes() == head
    assert_same(changed[3].readout.state_dict(), before)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(0), cuda_rng)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("k", [1, 2, 4])
def test_factory_active_consumer_matches_the_explicit_typed_configuration(device, k):
    model = pt4(None).to(device).eval().requires_grad_(False)
    geometry = dict(GEOMETRY, cycles=0, recurrence=dict(steps=2, max_node_updates_per_row=1216))
    variant = selected(model, geometry=geometry, k=k)
    built = variant.readout_config(CODEC, hidden_size=32)
    manual_geometry = {name: value for name, value in geometry.items() if name != "recurrence"}
    manual_geometry["regions"] = tuple(manual_geometry["regions"])
    manual = EpisodicReadoutConfig(
        CODEC,
        hidden_size=32,
        refinements=k,
        evidence_bridge=EvidenceBridgeConfig(
            CODEC, hidden_size=32, recurrence=BridgeRecurrenceConfig(2, 1216), **manual_geometry
        ),
    )
    assert built.identity() == manual.identity()
    readers = [
        EpisodicReadout(config, device=device, dtype=torch.float64).eval().requires_grad_(False)
        for config in (built, manual)
    ]
    consumers = (
        variant.consumer(model, readers[0]),
        FrozenFinancialConsumer(model, readout=readers[1]),
    )
    assert consumers[0].identity() == consumers[1].identity()
    batch = batch_on(model, ("US/AAA", "US/BBB"), 0)
    snapshot = bank(device, cutoff=batch.prediction_at[0])
    states = [
        consumer.prepare(
            batch, model.initial_state(batch.flow_ids), context_id=CONTEXT, snapshot=snapshot
        )
        for consumer in consumers
    ]
    assert torch.equal(states[0].point_predictions, states[1].point_predictions)
    assert_same(model.export_state(states[0].next_state), model.export_state(states[1].next_state))
