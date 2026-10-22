"""M2/M3 con contexto PT4 y feedback PT2, sobre etiquetas manuales maduras."""

import pytest
from test_financial_mechanisms import assert_same, engines
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_mars_titan_session_parity import varied
from test_mechanism_admission import DEVICES, PHASE, configured, feedback, inspect_bank, session
from test_native_episode_backend import native as native

from mars_titan.calibration.cohort_feedback import CohortFeedback
from mars_titan.models.titans.mechanism_catalog import MechanismAction
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("admission", ["m2", "m3"])
def test_mature_admission_and_online_context_calibration_share_the_native_generation(
    native, four_flow_source, tmp_path, device, admission
):
    source = varied(four_flow_source)
    parent = engines(source, device, online=True)
    regional = configured(source, device)[3]
    consumer = FrozenMechanismConsumer(
        parent[3].predictor,
        readout=regional.readout,
        calibration=CohortFeedback.restore(parent[3]._initial_calibration),
        catalog=regional.catalog,
        action=MechanismAction("parent", "k4cr", "online"),
    )
    data = (*parent[:3], consumer, parent[4])
    prior, points = [], {}
    with session(native, source, data, tmp_path / "online", admission, collect=True) as run:
        run.step(source["batches"][125], [], kind="warmup")
        for index in (126, 127, 128, 130):
            emitted = run.step(source["batches"][index], feedback(native, prior, index))
            prior = emitted.predictions
            points.update({(p.asset, p.decision_at): p.value for p in prior})
            inspect_bank(run, points)
            control = run.mechanism_event(run.snapshot()["cursor"])["payload"]
            assert control["work"]["context_updates_proposed"] == 1
            assert control["work"]["mac_online_observations"] == 4
            assert all(
                row["context"] == "parent" and row["calibration"] == "online"
                for row in control["rows"]
            )
            auxiliary = consumer.restore_state(run.mechanism_state())
            assert auxiliary.calibration.pending_rows == 4
        run.step([], [], kind="settlement", cutoff=PHASE.close_at, close_phase=True)
        assert run.snapshot()["applied"] == 12 and run.diagnostics()["pending"] == 0
        expected = run.mechanism_state()
    with session(
        native, source, data, tmp_path / "online", admission, collect=True, resume=True
    ) as restored:
        assert_same(restored.mechanism_state(), expected)
