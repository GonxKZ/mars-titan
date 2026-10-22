"""Contratos de la jerarquía temporal, sin optimizadores ni etiquetas futuras."""

import copy
import importlib
import importlib.util
from dataclasses import replace

import pytest
import torch
from test_bridge_controller_contracts import bound_policy
from test_bridge_episodic_readout import CONTEXT, bank, configuration
from test_context_selection import batch_on
from test_frozen_financial import frozen_backend as frozen_backend
from test_market_memory import pt4

from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer, _implementation
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer, _code

CONTRACT = "shared_evidence_bridge_hierarchy_v1"
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def assert_same_payload(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert actual[name].dtype == value.dtype and torch.equal(actual[name], value), name
        else:
            assert actual[name] == value, name


def hierarchy_config(*, k=1, mode="bank", budget=988, feedback=True, topology="tree", steps=2):
    name = "mars_titan.models.titans.bridge_hierarchy"
    assert importlib.util.find_spec(name) is not None, "Falta el contrato jerárquico tipado"
    api = importlib.import_module(name)
    assert hasattr(api, "BridgeHierarchyConfig"), "Falta la configuración jerárquica"
    base = configuration(refinements=k, mode=mode, episode_selection="first_read")
    assert hasattr(base.evidence_bridge, "hierarchy"), "Falta la opción de jerarquía"
    return replace(
        base,
        evidence_bridge=replace(
            base.evidence_bridge,
            regions=("view_1", "view_2", "view_3", "view_4"),
            topology="self",
            cycles=0,
            hierarchy=api.BridgeHierarchyConfig(topology, steps, feedback, budget),
        ),
    )


def hierarchy_reader(device="cpu", **options):
    return (
        EpisodicReadout(hierarchy_config(**options), dtype=torch.float64, device=device)
        .eval()
        .requires_grad_(False)
    )


def hierarchy_engine(device="cpu", **options):
    model = pt4(None, head=QUANTILE_HEAD).to(device).eval().requires_grad_(False)
    return FrozenMechanismConsumer(
        model,
        readout=hierarchy_reader(device, **options),
        catalog=MechanismCatalog(reader_contract=CONTRACT),
        action=MechanismAction(route="k4cc"),
    )


@pytest.mark.parametrize(
    "steps,expected", [(1, (8, 8, 2, 18)), (2, (16, 8, 2, 26)), (4, (32, 8, 2, 42))]
)
def test_reader_counts_local_group_and_root_updates_separately(steps, expected):
    reader = hierarchy_reader(steps=steps)
    local, groups, root, total = expected
    assert reader.recurrence_work(1) == dict(
        bridge_initial_views=4,
        bridge_local_updates=local,
        bridge_group_updates=groups,
        bridge_root_updates=root,
        bridge_node_updates=total,
    )
    assert all(value == 0 for value in reader.recurrence_work(0).values())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("budget", [104, 103])
def test_direct_k4_budget_is_checked_before_any_mac(device, monkeypatch, budget):
    model = pt4(None).to(device).eval().requires_grad_(False)
    reader = hierarchy_reader(device, k=4, budget=budget)
    batch = batch_on(model, ("US/AAA", "US/BBB"), 0)
    snapshot = bank(device, cutoff=batch.prediction_at[0])
    calls = []
    original = FinancialPredictor.prepare

    def observed(*args, **options):
        calls.append(True)
        return original(*args, **options)

    monkeypatch.setattr(FinancialPredictor, "prepare", observed)
    engine = FrozenFinancialConsumer(model, readout=reader)
    state = model.initial_state(batch.flow_ids)
    if budget == 104:
        result = engine.prepare(batch, state, context_id=CONTEXT, snapshot=snapshot)
        assert len(result.readout.reads) == 4
        assert result.next_state.mac.memory.steps.tolist() == [1, 1]
    else:
        with pytest.raises(ValueError, match="presupuesto"):
            engine.prepare(batch, state, context_id=CONTEXT, snapshot=snapshot)
    assert len(calls) == int(budget == 104)
    assert state.observed_steps.tolist() == [0, 0]


def test_hierarchy_requires_its_catalog_and_rejects_previous_policies():
    engine = hierarchy_engine()
    assert engine.identity()["schema_version"] == 5
    assert (
        MechanismCatalog.restore(engine.catalog.identity()).fingerprint()
        == engine.catalog.fingerprint()
    )
    for contract in (
        "legacy",
        "shared_evidence_bridge_v1",
        "shared_evidence_bridge_nonlinear_v1",
        "shared_evidence_bridge_intervention_v1",
    ):
        previous = MechanismCatalog(reader_contract=contract)
        with pytest.raises(ValueError, match="catálogo"):
            FrozenMechanismConsumer(engine.predictor, readout=engine.readout, catalog=previous)
        with pytest.raises(ValueError):
            FrozenMechanismConsumer(
                engine.predictor,
                readout=engine.readout,
                catalog=engine.catalog,
                policy=bound_policy(previous.fingerprint()),
            )
    name = "mars_titan.models.titans.bridge_hierarchy"
    assert name in _implementation() and _implementation()[name] == _code()[name]


@pytest.mark.parametrize("buffer", ["edges", "hierarchy_edges"])
def test_declared_geometry_is_checked_before_loading_or_freezing(buffer):
    reader = hierarchy_reader()
    assert hasattr(reader.evidence_bridge, buffer), "Falta la geometría persistente declarada"
    valid = copy.deepcopy(reader.state_dict())
    key = "evidence_bridge." + buffer
    payload = copy.deepcopy(valid)
    payload[key][0, 0] = False
    payload["step_logit"].add_(0.125)
    with pytest.raises(ValueError):
        reader.load_state_dict(payload)
    assert_same_payload(reader.state_dict(), valid)
    payload = copy.deepcopy(valid)
    payload[key] = payload[key].to(torch.float64)
    payload["step_logit"].add_(0.125)
    with pytest.raises(ValueError):
        reader.load_state_dict(payload)
    assert_same_payload(reader.state_dict(), valid)
    payload = copy.deepcopy(valid)
    payload["evidence_bridge._extra_state"]["configuration"]["hierarchy"]["feedback"] = False
    payload["step_logit"].add_(0.125)
    with pytest.raises(ValueError):
        reader.load_state_dict(payload)
    assert_same_payload(reader.state_dict(), valid)
    with torch.no_grad():
        getattr(reader.evidence_bridge, buffer)[0, 0] = False
    corrupt = copy.deepcopy(reader.state_dict())
    payload = copy.deepcopy(corrupt)
    payload["step_logit"].add_(0.125)
    with pytest.raises(ValueError):
        reader.load_state_dict(payload)
    assert_same_payload(reader.state_dict(), corrupt)
    model = pt4(None).eval().requires_grad_(False)
    with pytest.raises(ValueError):
        FrozenFinancialConsumer(model, readout=reader)
    quantile_model = pt4(None, head=QUANTILE_HEAD).eval().requires_grad_(False)
    with pytest.raises(ValueError):
        FrozenMechanismConsumer(
            quantile_model, readout=reader, catalog=MechanismCatalog(reader_contract=CONTRACT)
        )
