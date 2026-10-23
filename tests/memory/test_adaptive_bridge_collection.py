"""Trabajo adaptativo del DAG regional y su archivo nativo, sin ajustes."""

import importlib
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_financial_mechanisms import assert_same
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_mechanism_consumer import calibration
from test_native_episode_backend import native as native
from test_nonlinear_bridge_integration import nonlinear_engine
from test_regional_mechanism_collection import collect, snapshot_on
from test_regional_program_session import attempt, limits, session

from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.mechanism_catalog import MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from tests.memory.test_adaptive_bridge_session import engines as native_engines
from tests.memory.test_adaptive_bridge_session import issue_and_mature
from tests.models.titans.test_adaptive_bridge_contracts import BACKENDS, DEVICES
from tests.proposals.test_mature_proposals import no_optimizer_steps as no_optimizer_steps

CONTRACT = "shared_evidence_bridge_adaptive_v1"


def adaptive_engine(device, backend, *, budget=456, mode="bank"):
    parent = nonlinear_engine(device, steps=4, budget=budget, context="parent")
    module = importlib.import_module("mars_titan.models.titans.bridge_recurrence")
    config = replace(parent.readout.config, mode=mode)
    recurrence = replace(
        config.evidence_bridge.recurrence,
        stopping=module.BridgeStoppingConfig(1e308, 0.0, backend=backend),
    )
    reader = (
        EpisodicReadout(
            replace(config, evidence_bridge=replace(config.evidence_bridge, recurrence=recurrence)),
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
        action=parent.action,
        catalog=MechanismCatalog(reader_contract=CONTRACT),
    )


@contextmanager
def observed_kernel_work():
    """Contar filas reales de la transición sin sustituir resultados o métodos."""
    module = importlib.import_module("mars_titan.models.titans.bridge_recurrence")
    code = module._mix_step.__code__
    evaluated = []
    previous = sys.getprofile()

    def observe(frame, event, argument):
        if event == "call" and frame.f_code is code:
            current = frame.f_locals["current"]
            evaluated.append(current.shape[0] * current.shape[1])

    sys.setprofile(observe)
    try:
        yield evaluated
    finally:
        sys.setprofile(previous)


def test_adaptive_shared_collection_is_reachable_in_call():
    module = importlib.import_module("mars_titan.memory.mechanism_collection")
    assert callable(getattr(module, "MechanismCollector", None))
    engine = adaptive_engine("cpu", BACKENDS[0])
    packet = collect(engine, snapshot_on("cpu"))
    assert packet.payload()["work"]["bridge_selected_node_updates"] == 108
    assert packet.payload()["work"]["bridge_node_updates"] == 432


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", DEVICES)
def test_shared_prefixes_charge_nine_reads_per_context_once(backend, device):
    engine = adaptive_engine(device, backend)
    source = snapshot_on(device)
    snapshot_before = source.export_cpu()["sha256"]
    reader_before = deepcopy(engine.readout.state_dict())
    predictor_before = deepcopy(engine.predictor.state_dict())
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone() if device != "cpu" else None
    with observed_kernel_work() as evaluated:
        packet = collect(engine, source)
    body = packet.payload()
    work = body["work"]
    assert work["shared_route_batches"] == 2 and work["individual_route_batches"] == 0
    assert work["reader_refinements"] == work["reader_value_projections"] == 2 * 2 * 9
    assert work["reader_value_projections"] != 2 * 2 * sum((1, 2, 2, 4, 4, 4))
    assert work["bridge_initial_views"] == work["bridge_selected_node_updates"] == 108
    assert work["bridge_node_updates"] == (432 if backend == BACKENDS[0] else 108)
    assert work["bridge_node_updates"] == sum(evaluated)
    assert body["catalog"]["reader_contract"] == CONTRACT
    module = importlib.import_module("mars_titan.memory.mechanism_collection")
    assert module.CollectedMechanisms.restore(packet.record()).record() == packet.record()
    assert source.export_cpu()["sha256"] == snapshot_before
    assert_same(engine.readout.state_dict(), reader_before)
    assert_same(engine.predictor.state_dict(), predictor_before)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(0))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind", ("empty", "absent"))
def test_collection_without_evidence_has_no_adaptive_evaluations_or_projections(backend, kind):
    engine = adaptive_engine("cpu", backend, mode="no_bank" if kind == "absent" else "bank")
    source = snapshot_on("cpu", count=0) if kind == "empty" else None
    with observed_kernel_work() as evaluated:
        work = collect(engine, source).payload()["work"]
    assert work["reader_refinements"] == 36
    assert work["reader_value_projections"] == 0
    assert work["bridge_initial_views"] == 0
    assert work["bridge_selected_node_updates"] == work["bridge_node_updates"] == 0
    assert evaluated == []


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("field", ("bridge_selected_node_updates", "bridge_node_updates"))
def test_resealed_packet_rejects_a_changed_adaptive_aggregate(backend, field):
    module = importlib.import_module("mars_titan.memory.mechanism_collection")
    body = collect(adaptive_engine("cpu", backend), snapshot_on("cpu")).payload()
    forged = deepcopy(body)
    forged["work"][field] += 3
    with pytest.raises(ValueError):
        module.CollectedMechanisms(forged)


@pytest.mark.parametrize("backend", BACKENDS)
def test_partitioned_collection_preserves_candidates_and_physical_node_work(backend):
    engine = adaptive_engine("cpu", backend)
    source = snapshot_on("cpu")
    with observed_kernel_work() as full_calls:
        full = collect(engine, source).payload()
    with observed_kernel_work() as split_calls:
        split = collect(engine, source, split=True).payload()
    assert full["observation"] == split["observation"]
    for left, right in zip(
        full["groups"][0]["candidates"], split["groups"][0]["candidates"], strict=True
    ):
        assert left["requested"] == right["requested"] and left["status"] == right["status"]
        if left["quantiles"] is not None:
            torch.testing.assert_close(
                torch.tensor(left["quantiles"], dtype=torch.float64),
                torch.tensor(right["quantiles"], dtype=torch.float64),
                rtol=1e-12,
                atol=1e-12,
            )
    for field in (
        "reader_refinements",
        "reader_value_projections",
        "bridge_initial_views",
        "bridge_selected_node_updates",
        "bridge_node_updates",
    ):
        assert full["work"][field] == split["work"][field]
    assert sum(full_calls) == sum(split_calls) == full["work"]["bridge_node_updates"]


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("backend", BACKENDS)
def test_native_collection_plus_emission_charge_actual_dag_and_keep_the_worst_case_reservation(
    native, four_flow_source, tmp_path, backend
):
    data = native_engines(four_flow_source, "cpu", backend=backend, budget=456)
    output = tmp_path / "collection"
    options = dict(regional_work_limits=limits(), collection=True)
    with session(native, output, data, **options) as run:
        issue_and_mature(run, four_flow_source, native)
        with observed_kernel_work() as evaluated:
            run.step(four_flow_source["batches"][128], [])
        event = run.mechanism_event(4)
        packet = run.mechanism_collection(4)["collection"]
        module = importlib.import_module("mars_titan.memory.mechanism_collection")
        body = module.CollectedMechanisms.restore(packet).payload()
        work, emission = body["work"], event["payload"]["work"]
        assert work["reader_value_projections"] == work["reader_refinements"] == 4 * 2 * 9
        assert work["bridge_initial_views"] == work["bridge_selected_node_updates"] == 216
        assert emission["bridge_initial_views"] == emission["bridge_selected_node_updates"] == 48
        factor = 4 if backend == BACKENDS[0] else 1
        assert work["bridge_node_updates"] == 216 * factor
        assert emission["bridge_node_updates"] == 48 * factor
        assert work["bridge_node_updates"] + emission["bridge_node_updates"] == sum(evaluated)
        reserved = deepcopy(attempt(output))
        assert reserved["status"] == "confirmed"
        assert reserved["units"]["reader_refinements"] == 4 * 38
        assert reserved["units"]["reader_refinements"] > (
            work["reader_refinements"] + emission["reader_refinements"]
        )
        expected_head = (output / "latest.json").read_bytes()
        expected_snapshot, expected_fast = run.snapshot(), run.fast_state()
    with observed_kernel_work() as reopened:
        restored = session(native, output, data, resume=True, **options)
    with restored as run:
        assert reopened == []
        assert (output / "latest.json").read_bytes() == expected_head
        assert run.snapshot() == expected_snapshot
        assert_same(run.fast_state(), expected_fast)
        assert run.mechanism_collection(4)["collection"] == packet
        assert_same(run.mechanism_event(4), event)
        assert attempt(output) == reserved
