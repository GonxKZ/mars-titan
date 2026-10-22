"""Intervenciones fijas sobre un episodio común, sin aprendizaje ni datos financieros reales."""

import copy
import json
from dataclasses import replace

import pytest
import torch
from test_bridge_controller_contracts import bound_policy
from test_bridge_episodic_readout import CODEC, CONTEXT, bank, configuration
from test_bridge_mechanism_session import engines
from test_context_selection import batch_on
from test_financial_mechanisms import assert_same, session
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_market_memory import pt4
from test_native_episode_backend import native as native
from test_shared_bridge_variant import GEOMETRY, selected

from mars_titan.memory import mars_titan_variant as variants
from mars_titan.memory import regional_episode_trace as episode_trace
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans import regional_mechanism_trace as mechanism_trace
from mars_titan.models.titans.bridge_intervention import BridgeInterventionConfig
from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer, _implementation
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer, _code
from mars_titan.models.titans.regional_routes import run_route_pair

CONTRACT = "shared_evidence_bridge_intervention_v1"
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))
PARTIAL = BridgeInterventionConfig((True, False, True), ((2, 0),))


def intervened_config(*, intervention=PARTIAL, k=1):
    config = configuration(refinements=k, episode_selection="first_read")
    return replace(
        config,
        evidence_bridge=replace(
            config.evidence_bridge,
            topology="complete",
            cycles=0,
            recurrence=BridgeRecurrenceConfig(2, 1216),
            intervention=intervention,
        ),
    )


def reader(config, device="cpu"):
    return EpisodicReadout(config, device=device, dtype=torch.float64).eval().requires_grad_(False)


def consumer(device="cpu"):
    model = pt4(None, head=QUANTILE_HEAD).to(device).eval().requires_grad_(False)
    return FrozenMechanismConsumer(
        model,
        readout=reader(intervened_config(), device),
        catalog=MechanismCatalog(reader_contract=CONTRACT),
        action=MechanismAction(route="k4cc"),
    )


@pytest.mark.parametrize("kind", ["financial", "mechanism"])
@pytest.mark.parametrize(
    "field",
    [
        "active_regions",
        "edges",
        "active_regions_dtype",
        "edges_dtype",
        "configuration",
        "absent_mask",
        "sparse_mask",
        "hidden_legacy_intervention",
        "hidden_absent_intervention",
    ],
)
def test_prefreeze_corruption_cannot_be_sealed_as_the_declared_intervention(kind, field):
    model = pt4(None, head=QUANTILE_HEAD).eval().requires_grad_(False)
    direct = reader(intervened_config())
    if field.startswith("hidden_"):
        direct.config = replace(
            direct.config,
            evidence_bridge=None
            if field == "hidden_absent_intervention"
            else replace(direct.config.evidence_bridge, intervention=None),
        )
    elif field == "configuration":
        direct.evidence_bridge.config = replace(
            direct.evidence_bridge.config, intervention=BridgeInterventionConfig((True,) * 3)
        )
    elif field == "absent_mask":
        direct.evidence_bridge.active_regions = None
    elif field == "sparse_mask":
        direct.evidence_bridge.active_regions = direct.evidence_bridge.active_regions.to_sparse()
    else:
        name = field.removesuffix("_dtype")
        value = getattr(direct.evidence_bridge, name)
        if field.endswith("_dtype"):
            setattr(direct.evidence_bridge, name, value.double())
        else:
            value.logical_not_()
    with pytest.raises(ValueError, match="regiones|puentes|booleanos"):
        if kind == "financial":
            FrozenFinancialConsumer(model, readout=direct)
        else:
            contract = (
                "legacy"
                if direct.config.evidence_bridge is None
                else direct.config.evidence_bridge.identity()["kind"]
            )
            FrozenMechanismConsumer(
                model, readout=direct, catalog=MechanismCatalog(reader_contract=contract)
            )


def test_variant_roundtrip_keeps_the_fixed_intervention_and_omits_null():
    model = pt4(None).eval().requires_grad_(False)
    geometry = dict(
        GEOMETRY,
        topology="complete",
        cycles=0,
        recurrence=dict(steps=2, max_node_updates_per_row=1216),
        intervention=dict(active_regions=[True, False, True], removed_edges=[[2, 0]]),
    )
    active = selected(model, geometry=geometry, k=4, episodes="first_read")
    rebuilt = variants.select_variant(
        variants.load_declaration(), json.loads(json.dumps(active.components)), base=active.base
    )
    assert rebuilt.fingerprint() == active.fingerprint()
    config = rebuilt.readout_config(CODEC, hidden_size=32)
    assert type(config.evidence_bridge.intervention) is BridgeInterventionConfig
    assert config.evidence_bridge.intervention == PARTIAL
    assert config.identity()["evidence_bridge"]["kind"] == CONTRACT
    built = reader(config)
    assert rebuilt.consumer(model, built).readout is built
    geometry["intervention"]["active_regions"][1] = True
    geometry["intervention"]["removed_edges"][0][0] = 0
    assert rebuilt.readout_config(CODEC, hidden_size=32).evidence_bridge.intervention == PARTIAL
    base = {key: value for key, value in geometry.items() if key != "intervention"}
    plain = selected(model, geometry=base)
    null = selected(model, geometry=dict(base, intervention=None))
    assert null.identity() == plain.identity()
    assert "intervention" not in null.components["evidence_bridge"]
    null_config = null.readout_config(CODEC, hidden_size=32)
    assert "intervention" not in null_config.identity()["evidence_bridge"]
    assert "active_regions" not in reader(null_config).evidence_bridge.state_dict()
    assert config.evidence_bridge.identity() != null_config.evidence_bridge.identity()
    name = "mars_titan.models.titans.bridge_intervention"
    assert name in _implementation() and _implementation()[name] == _code()[name]


@pytest.mark.parametrize(
    "value",
    [
        False,
        {},
        dict(active_regions=True, removed_edges=[]),
        dict(active_regions=[True, False, True], removed_edges=True),
        dict(active_regions=[True, False, True], removed_edges=[True]),
        dict(active_regions=[True, 0, True], removed_edges=[]),
        dict(active_regions=[True, False], removed_edges=[]),
        dict(active_regions=[True, False, True], removed_edges=[[3, 0]]),
        dict(active_regions=[True, False, True], removed_edges=[[False, 0]]),
        dict(active_regions=[True, False, True], removed_edges=[[2, 0], [2, 0]]),
        dict(active_regions=[True, False, True], removed_edges=[[2, 0], [0, 2]]),
        dict(active_regions=[True, False, True], removed_edges=[], other=True),
    ],
)
def test_parser_rejects_ambiguous_or_noncanonical_interventions(value):
    with pytest.raises(ValueError):
        selected(pt4(None), geometry=dict(GEOMETRY, topology="complete", intervention=value))


def test_typed_intervention_checks_masks_and_the_declared_topology():
    with pytest.raises(ValueError):
        BridgeInterventionConfig([True, False, True])
    with pytest.raises(ValueError):
        replace(intervened_config().evidence_bridge, intervention={"active_regions": []})
    with pytest.raises(ValueError):
        selected(
            pt4(None),
            geometry=dict(
                GEOMETRY,
                topology="self",
                intervention=dict(active_regions=[True] * 3, removed_edges=[[2, 0]]),
            ),
        )


def test_intervened_reader_rejects_previous_catalogs_and_their_policies():
    current = consumer()
    assert current.identity()["schema_version"] == 4
    restored = MechanismCatalog.restore(current.catalog.identity())
    assert restored.fingerprint() == current.catalog.fingerprint()
    for contract in ("legacy", "shared_evidence_bridge_v1", "shared_evidence_bridge_nonlinear_v1"):
        old = MechanismCatalog(reader_contract=contract)
        with pytest.raises(ValueError, match="catálogo"):
            FrozenMechanismConsumer(current.predictor, readout=current.readout, catalog=old)
        with pytest.raises(ValueError):
            FrozenMechanismConsumer(
                current.predictor,
                readout=current.readout,
                catalog=current.catalog,
                policy=bound_policy(old.fingerprint()),
            )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("k", [1, 4])
def test_silencing_all_views_keeps_common_ids_and_dense_work_without_useful_presence(device, k):
    config = intervened_config(intervention=BridgeInterventionConfig((False,) * 3), k=k)
    active = reader(config, device)
    control = reader(
        replace(config, evidence_bridge=replace(config.evidence_bridge, intervention=None)), device
    )
    assert_same(dict(active.named_parameters()), dict(control.named_parameters()))
    snapshot = bank(device)
    before = snapshot.export_cpu()
    state = torch.linspace(-0.5, 0.5, 64, dtype=torch.float64, device=device).reshape(2, 32)
    expected = control(state, snapshot, context_id=CONTEXT, cutoff=10)
    projections = []
    hook = active.evidence_bridge.value.register_forward_hook(lambda *_: projections.append(True))
    try:
        result = active(state, snapshot, context_id=CONTEXT, cutoff=10)
    finally:
        hook.remove()
    assert len(projections) == k
    assert active.recurrence_work(k) == dict(bridge_initial_views=3 * k, bridge_node_updates=6 * k)
    assert len(result.reads) == k
    for read in result.reads:
        assert read.bridge.evidence_ids.shape == (2, 8)
        assert torch.equal(read.bridge.evidence_ids, expected.reads[0].bridge.evidence_ids)
        assert not read.presence.any()
        for value in (
            read.values,
            read.bridge.evidence_weights,
            read.bridge.region_values,
            read.bridge.bridge_weights,
        ):
            assert not value.count_nonzero()
    flows = ("US/AAA", "US/BBB")
    trace = episode_trace.begin(
        active, flows, snapshot, context_id=CONTEXT, cutoff=10, block_rows=2
    )
    episode_trace.append(trace, flows, result)
    episode_trace.verify(
        trace, active, flows=flows, context_id=CONTEXT, cutoff=10, refinements=k, block_rows=2
    )
    _, routed = run_route_pair(
        active, state, snapshot, route="k1" if k == 1 else "k4cc", context_id=CONTEXT, cutoff=10
    )
    for row in mechanism_trace.export(routed, snapshot):
        mechanism_trace.verify(
            row,
            active,
            snapshot_sha256=snapshot._digest,
            steps=k,
            searches=1,
            route="k1" if k == 1 else "k4cc",
        )
    assert_same(snapshot.export_cpu(), before)


def test_both_archives_enforce_zero_weights_and_values_for_the_silenced_view():
    direct = reader(intervened_config(k=4))
    snapshot = bank("cpu")
    flows = ("US/AAA", "US/BBB")
    result = direct(
        torch.ones((2, 32), dtype=torch.float64), snapshot, context_id=CONTEXT, cutoff=10
    )
    trace = episode_trace.begin(
        direct, flows, snapshot, context_id=CONTEXT, cutoff=10, block_rows=2
    )
    episode_trace.append(trace, flows, result)
    expected_graph = torch.tensor([[0.5, 0, 0.5], [0, 0, 0], [0, 0, 1]], dtype=torch.float64)
    assert torch.equal(trace["bridge_weights"], expected_graph)
    episode_options = dict(flows=flows, context_id=CONTEXT, cutoff=10, refinements=4, block_rows=2)
    episode_trace.verify(trace, direct, **episode_options)
    for read in trace["blocks"][0]["reads"]:
        assert torch.allclose(
            read["regional_weights"].sum(-1),
            torch.tensor([[1.0, 0.0, 1.0]] * 2, dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )
        assert not read["regional_values"][:, 1].count_nonzero()
    for field in ("regional_weights", "regional_values"):
        invalid = copy.deepcopy(trace)
        invalid["blocks"][0]["reads"][0][field][0, 1, 0] = 1.0
        with pytest.raises(ValueError):
            episode_trace.verify(invalid, direct, **episode_options)
    engine = consumer()
    batch = batch_on(engine.predictor, flows, 0)
    cutoff = batch.prediction_at[0]
    current_snapshot = bank("cpu", cutoff=cutoff)
    event = engine.begin(
        engine.initial_state(),
        flow_ids=flows,
        cutoff=cutoff,
        context_id=CONTEXT,
        matures_at=(cutoff + 10,) * 2,
        snapshot=current_snapshot,
    )
    block = engine.prepare(batch, engine.predictor.initial_state(flows), event)
    emitted = engine.finish(block.event)
    assert emitted.work["bridge_value_projections"] == 8
    assert emitted.work["bridge_node_updates"] == 48
    for row in emitted.rows:
        record = row["regional_readout"]
        options = dict(snapshot_sha256=current_snapshot._digest, steps=4, searches=1, route="k4cc")
        mechanism_trace.verify(record, engine.readout, **options)
        for field in ("regional_weights", "regional_values"):
            invalid = copy.deepcopy(record)
            invalid["reads"][0][field][1][0] = 1.0
            with pytest.raises(ValueError):
                mechanism_trace.verify(invalid, engine.readout, **options)


def test_checkpoint_cannot_replace_the_intervention_before_copying_parent_weights():
    original = reader(intervened_config())
    other = reader(
        intervened_config(intervention=BridgeInterventionConfig((True, True, False), ((2, 0),)))
    )
    before = copy.deepcopy(original.state_dict())
    payload = copy.deepcopy(other.state_dict())
    payload["step_logit"].add_(0.125)
    payload["refinement.bias"].add_(0.125)
    with pytest.raises(ValueError):
        original.load_state_dict(payload)
    assert_same(original.state_dict(), before)


def test_checkpoint_validates_each_fixed_boolean_buffer_against_configuration():
    original = reader(intervened_config())
    valid = copy.deepcopy(original.state_dict())
    names = [
        name
        for name, value in valid.items()
        if isinstance(value, torch.Tensor) and value.dtype == torch.bool
    ]
    assert set(names) == {"evidence_bridge.edges", "evidence_bridge.active_regions"}
    for name in names:
        payload = copy.deepcopy(valid)
        index = (0,) * payload[name].ndim
        payload[name][index] = not payload[name][index].item()
        payload["step_logit"].add_(0.125)
        with pytest.raises(ValueError):
            original.load_state_dict(payload)
        assert_same(original.state_dict(), valid)
        tampered = reader(intervened_config())
        with torch.no_grad():
            dict(tampered.named_buffers())[name][index] = payload[name][index]
        before = copy.deepcopy(tampered.state_dict())
        payload = copy.deepcopy(before)
        payload["step_logit"].add_(0.125)
        with pytest.raises(ValueError):
            tampered.load_state_dict(payload)
        assert_same(tampered.state_dict(), before)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("device", DEVICES)
def test_native_recovery_preserves_the_next_intervened_emission_and_feedback_once(
    native, four_flow_source, tmp_path, device
):
    parent = engines(four_flow_source, device, recurrence=BridgeRecurrenceConfig(2, 1216))
    config = replace(
        parent[3].readout.config,
        evidence_bridge=replace(
            parent[3].readout.config.evidence_bridge, topology="complete", intervention=PARTIAL
        ),
    )
    engine = FrozenMechanismConsumer(
        parent[3].predictor,
        readout=reader(config, device),
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
        assert expected_event["payload"]["work"]["bridge_value_projections"] == 16
        assert expected_event["payload"]["work"]["bridge_node_updates"] == 96
        for row in expected_event["payload"]["rows"]:
            for read in row["regional_readout"]["reads"]:
                assert all(value == 0.0 for value in read["regional_weights"][1])
                assert all(value == 0.0 for value in read["regional_values"][1])
    with session(native, tmp_path / "cut", data, block_rows=2) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        run.step(four_flow_source["batches"][126], [])

        def cut(boundary):
            if boundary == native.Boundary.committed:
                raise RuntimeError("Corte tras confirmar feedback con intervención")

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
