"""Coste y recuperación del puente adaptativo sobre la sesión nativa y datos manuales."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_financial_mechanisms import assert_same
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native
from test_regional_program_session import (
    DEVICES,
    attempt,
    limits,
    observed_calls,
    predictions,
    session,
)
from test_regional_program_session import engines as parent_engines

from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from tests.models.titans.test_adaptive_bridge_contracts import BACKENDS, adaptive_configuration
from tests.proposals.test_mature_proposals import no_optimizer_steps as no_optimizer_steps

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
CONTRACT = "shared_evidence_bridge_adaptive_v1"


def engines(source, device, *, backend, budget=1216):
    parent = parent_engines(source, device)
    model = parent[3].predictor
    base = adaptive_configuration(backend=backend, budget=budget, k=1)
    codec_id = source["codec"].fingerprint()
    config = replace(
        base,
        codec_id=codec_id,
        max_batch=2,
        evidence_bridge=replace(base.evidence_bridge, codec_id=codec_id, max_batch=2),
    )
    reader = EpisodicReadout(config, device=device, dtype=torch.float64)
    reader.eval().requires_grad_(False)
    consumer = FrozenMechanismConsumer(
        model,
        readout=reader,
        action=MechanismAction("parent", "k4cc", "none"),
        catalog=MechanismCatalog(
            reader_contract=CONTRACT, regional_execution=model.config.regional_execution
        ),
    )
    return parent[0], parent[1], parent[2], consumer, parent[4]


def issue_and_mature(run, source, native):
    assert source["label_origin"] == "manual_fixture_no_estimation"
    run.step(source["batches"][125], [], kind="warmup")
    first = run.step(source["batches"][126], [])
    assert len(first.predictions) == 4
    empty = run.mechanism_event(2)["payload"]
    assert empty["work"]["bridge_value_projections"] == 0
    assert empty["work"]["bridge_initial_views"] == 0
    assert empty["work"]["bridge_selected_node_updates"] == 0
    assert empty["work"]["bridge_node_updates"] == 0
    labels = [native.Feedback(row.id, 0, moment(127), 0.125) for row in first.predictions]
    run.step([], labels, kind="settlement", cutoff=moment(127))


def check_work(event, backend):
    payload = event["payload"]
    work = payload["work"]
    assert work["core_observations"] == work["mac_online_observations"] == 4
    assert work["reader_refinements"] == work["bridge_value_projections"] == 16
    assert work["bridge_initial_views"] == work["bridge_selected_node_updates"] == 48
    evaluated = 4 if backend == BACKENDS[0] else 1
    assert work["bridge_node_updates"] == 48 * evaluated
    selected_sum = evaluated_sum = projections = 0
    for row in payload["rows"]:
        trace = row["regional_readout"]
        assert trace["kind"] == "adaptive_regional_mechanism_trace_v1"
        assert trace["episodes"] == 4 and trace["value_projections"] == 4
        assert len(trace["reads"]) == 4
        common_ids = trace["reads"][0]["evidence_ids"]
        assert len(common_ids) == len(set(common_ids)) == 4
        for read in trace["reads"]:
            assert read["evidence_ids"] == common_ids
            assert read["stopping"] == dict(
                selected_steps=1, evaluated_steps=evaluated, criterion_met=True
            )
            assert type(read["stopping"]["selected_steps"]) is int
            assert type(read["stopping"]["evaluated_steps"]) is int
            assert type(read["stopping"]["criterion_met"]) is bool
            selected_sum += read["stopping"]["selected_steps"]
            evaluated_sum += read["stopping"]["evaluated_steps"]
            projections += 1
    assert work["bridge_initial_views"] == 3 * projections
    assert work["bridge_selected_node_updates"] == 3 * selected_sum
    assert work["bridge_node_updates"] == 3 * evaluated_sum


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", DEVICES)
def test_native_adaptive_cost_and_resume_reconstruct_the_next_emission(
    native, four_flow_source, tmp_path, backend, device
):
    data = engines(four_flow_source, device, backend=backend)
    consumer = data[3]
    assert consumer.identity()["schema_version"] == 7
    assert consumer.catalog.reader_contract == CONTRACT
    weights = deepcopy(consumer.predictor.state_dict()), deepcopy(consumer.readout.state_dict())
    options = dict(regional_work_limits=limits())
    with session(native, tmp_path / "reference", data, **options) as run:
        issue_and_mature(run, four_flow_source, native)
        expected = predictions(run.step(four_flow_source["batches"][128], []))
        expected_event, expected_fast = run.mechanism_event(4), run.fast_state()
        check_work(expected_event, backend)
        assert run.snapshot()["applied"] == 4
    output = tmp_path / "resumed"
    with session(native, output, data, **options) as run:
        issue_and_mature(run, four_flow_source, native)
        confirmed_head = (output / "latest.json").read_bytes()
        confirmed_snapshot = run.snapshot()
    with observed_calls(consumer) as calls:
        restored = session(native, output, data, resume=True, **options)
    with restored as run:
        assert all(count == 0 for count in calls.values())
        assert (output / "latest.json").read_bytes() == confirmed_head
        assert run.snapshot() == confirmed_snapshot
        actual = predictions(run.step(four_flow_source["batches"][128], []))
        assert actual == expected
        assert_same(run.fast_state(), expected_fast)
        assert_same(run.mechanism_event(4), expected_event)
        assert run.snapshot()["applied"] == 4
        check_work(run.mechanism_event(4), backend)
    assert_same(consumer.predictor.state_dict(), weights[0])
    assert_same(consumer.readout.state_dict(), weights[1])
    changed = engines(
        four_flow_source, device, backend=BACKENDS[1] if backend == BACKENDS[0] else BACKENDS[0]
    )
    with observed_calls(changed[3]) as calls:
        with pytest.raises(ValueError):
            session(native, output, changed, resume=True, **options)
    assert all(count == 0 for count in calls.values())


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_preflight_reserves_jmax_before_mac_even_when_the_rule_selects_j1(
    native, four_flow_source, tmp_path, backend
):
    data = engines(four_flow_source, "cpu", backend=backend, budget=455)
    consumer = data[3]
    output = tmp_path / "short_budget"
    options = dict(regional_work_limits=limits(), collection=True)
    with session(native, output, data, **options) as run:
        issue_and_mature(run, four_flow_source, native)
        before_head = (output / "latest.json").read_bytes()
        before_fast = run.fast_state()
        with observed_calls(consumer) as calls:
            with pytest.raises(ValueError, match="presupuesto|cuota|actualizaciones"):
                run.step(four_flow_source["batches"][128], [])
        assert calls["predictor_prepare"] == calls["consumer_prepare"] == calls["mac"] == 0
        assert (output / "latest.json").read_bytes() == before_head
        reserved = deepcopy(attempt(output))
        assert reserved["status"] == "reserved" and reserved["observed_units"] is None
        assert reserved["units"]["reader_refinements"] == 4 * 38
    with observed_calls(consumer) as calls:
        restored = session(native, output, data, resume=True, **options)
    with restored as run:
        assert all(count == 0 for count in calls.values())
        assert (output / "latest.json").read_bytes() == before_head
        assert_same(run.fast_state(), before_fast)
        current = attempt(output)
        assert current["reservation_id"] == reserved["reservation_id"]
        assert current["status"] == "abandoned" and current["observed_units"] is None
        assert current["units"] == reserved["units"]
