"""Profundidades observadas y recuperación del consumidor financiero sin selector."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_bridge_readout_session import control, engine, session
from test_financial_mechanisms import assert_same
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from mars_titan.memory import regional_episode_trace
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from tests.models.titans.test_adaptive_bridge_contracts import (
    BACKENDS,
    DEVICES,
    adaptive_configuration,
)

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


def consumer(source, device, backend):
    parent = engine(source, device)
    config = adaptive_configuration(backend=backend, k=2)
    codec = source["codec"].fingerprint()
    config = replace(
        config, codec_id=codec, evidence_bridge=replace(config.evidence_bridge, codec_id=codec)
    )
    reader = (
        EpisodicReadout(config, device=device, dtype=torch.float64).eval().requires_grad_(False)
    )
    return FrozenFinancialConsumer(parent.predictor, readout=reader)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", DEVICES)
def test_native_episode_depth_cost_and_recovery_preserve_the_confirmed_state(
    native, four_flow_source, tmp_path, backend, device
):
    source = four_flow_source
    model = consumer(source, device, backend)
    output = tmp_path / "adaptive"
    with session(native, output, source, model) as run:
        run.step(source["batches"][125], [], kind="warmup")
        first = run.step(source["batches"][126], [])
        empty = control(run)
        assert empty["bridge_node_updates"] == empty["bridge_selected_node_updates"] == 0
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in first.predictions]
        run.step([], labels, kind="settlement", cutoff=moment(127))
        run.step(source["batches"][128], [])
        expected_control, expected_fast = control(run), run.fast_state()
        trace = expected_control["regional_reads"]
        assert trace["kind"] == "adaptive_regional_episode_trace_v1"
        assert trace["schema_version"] == 2
        assert expected_control["bridge_value_projections"] == 8
        assert expected_control["bridge_initial_views"] == 24
        assert expected_control["bridge_selected_node_updates"] == 24
        assert expected_control["bridge_node_updates"] == (96 if backend == BACKENDS[0] else 24)
        assert run.snapshot()["applied"] == 4
        for block in trace["blocks"]:
            for read in block["reads"]:
                assert read["selected_steps"].tolist() == [1, 1]
                assert read["criterion_met"].tolist() == [True, True]
        tampered = deepcopy(trace)
        tampered["blocks"][0]["reads"][0]["evaluated_steps"].fill_(2)
        with pytest.raises(ValueError):
            regional_episode_trace.verify(
                tampered,
                model.readout,
                flows=tuple(flow for block in trace["blocks"] for flow in block["flow_ids"]),
                context_id=trace["context_id"],
                cutoff=moment(128),
                refinements=2,
                block_rows=2,
            )
    with session(native, output, source, model, resume=True) as run:
        assert_same(control(run), expected_control)
        assert_same(run.fast_state(), expected_fast)
        assert run.snapshot()["applied"] == 4


def test_native_decision_cannot_hide_its_trace_by_declaring_zero_refinements(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    model = consumer(source, "cpu", BACKENDS[0])
    with session(native, tmp_path / "forged", source, model) as run:
        run.step(source["batches"][125], [], kind="warmup")
        first = run.step(source["batches"][126], [])
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in first.predictions]
        run.step([], labels, kind="settlement", cutoff=moment(127))
        run.step(source["batches"][128], [])
        payload = deepcopy(control(run))
        assert payload["bridge_node_updates"] > 0
        payload.update(refinements=0, regional_reads=None)
        for name in (
            "bridge_value_projections",
            "bridge_initial_views",
            "bridge_selected_node_updates",
            "bridge_node_updates",
        ):
            payload[name] = 0
        bundle = deepcopy(run._bundle(run.snapshot()["state"]))
        bundle["control"] = run._stage(payload, "control")
        with pytest.raises(ValueError, match="refinamientos|traza"):
            run._verify_bundle(bundle, event_kind="decision")
