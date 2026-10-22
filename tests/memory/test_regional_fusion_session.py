"""Grafo regional y memoria de trabajo dentro de la publicación financiera."""

from dataclasses import replace

import pytest
import torch
from test_financial_mechanisms import assert_same, engines, session
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_mechanism_consumer import calibration
from test_native_episode_backend import native as native

from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.mechanism_catalog import MechanismAction
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def regional_data(source, device):
    base = engines(source, device, online=True)
    config = replace(
        base[3].predictor.config, regional_fusion=RegionalFusionConfig(topology="ring")
    )
    model = (
        FinancialPredictor(config, device=device, dtype=torch.float64).eval().requires_grad_(False)
    )
    values = model.state_dict()
    values["session_context.output"] = torch.eye(32, dtype=torch.float64, device=device) * 0.2
    model.load_state_dict(values)
    consumer = FrozenMechanismConsumer(
        model, calibration=calibration(), action=MechanismAction("parent", "k1", "online")
    )
    return (*base[:3], consumer, base[4])


@pytest.mark.parametrize("device", DEVICES)
def test_regional_work_is_archived_once_and_recovers_after_a_commit_cut(
    native, four_flow_source, tmp_path, device
):
    data = regional_data(four_flow_source, device)
    with session(native, tmp_path / "reference", data) as reference:
        reference.step(four_flow_source["batches"][125], [], kind="warmup")
        first = reference.step(four_flow_source["batches"][126], [])
        feedback = [native.Feedback(row.id, 0, moment(127), 0.125) for row in first.predictions]
        reference.step(four_flow_source["batches"][127], feedback)
        expected = reference.snapshot()
        expected_state = reference.mechanism_state()
        expected_trace = reference.mechanism_event(3)
        work = expected_trace["payload"]["work"]
        assert work["regional_local_node_evaluations"] == 80
        assert work["regional_summary_evaluations"] == 8
        assert work["mac_online_observations"] == 4
        assert expected_state["context"]["committed_at"].count(moment(127)) == 1
        reference.step(four_flow_source["batches"][128], [])
        following = reference.snapshot()
        following_state = reference.mechanism_state()
        following_trace = reference.mechanism_event(4)
    with session(native, tmp_path / "cut", data) as interrupted:
        interrupted.step(four_flow_source["batches"][125], [], kind="warmup")
        interrupted.step(four_flow_source["batches"][126], [])

        def cut(boundary):
            if boundary == native.Boundary.committed:
                raise RuntimeError("Corte después de publicar el grafo")

        with pytest.raises(RuntimeError, match="Corte después"):
            interrupted.step(four_flow_source["batches"][127], feedback, fault=cut)
    with session(native, tmp_path / "cut", data, resume=True) as recovered:
        assert recovered.snapshot() == expected
        assert_same(recovered.mechanism_state(), expected_state)
        assert recovered.mechanism_event(3) == expected_trace
        recovered.step(four_flow_source["batches"][128], [])
        assert recovered.snapshot() == following
        assert_same(recovered.mechanism_state(), following_state)
        assert recovered.mechanism_event(4) == following_trace
        assert (
            recovered.mechanism_event(2)["payload"]["work"]["regional_local_node_evaluations"] == 80
        )


def test_plain_titans_binding_also_records_regional_work(native, four_flow_source, tmp_path):
    base = engines(four_flow_source, "cpu")
    config = replace(base[3].predictor.config, regional_fusion=RegionalFusionConfig())
    model = FinancialPredictor(config, dtype=torch.float64).eval().requires_grad_(False)
    data = (*base[:3], FrozenFinancialConsumer(model), base[4])
    with session(native, tmp_path / "plain", data) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        run.step(four_flow_source["batches"][126], [])
        bundle = run._bundle(run.snapshot()["state"])
        work = run._read(bundle["control"], "control")
        assert work["regional_local_node_evaluations"] == 80
        assert work["regional_summary_evaluations"] == 8
        assert work["mac_updates"] == 4
