"""La reutilización del control se activa solo en la sesión de mecanismos."""

import copy

import pytest
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_mars_titan_session_parity import SCALERS, varied
from test_mechanism_admission import PHASE, configured
from test_native_episode_backend import native as native
from test_session_artifact_cache import loads as loads

from mars_titan.memory.financial_session import FinancialSession
from mars_titan.memory.write_policy import CompositeScoreConfig
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("mechanism", [False, True])
def test_only_mechanism_sessions_reuse_control_deserialization(
    native, four_flow_source, tmp_path, loads, device, mechanism
):
    source = varied(four_flow_source)
    consumer = configured(source, device)[3]
    if not mechanism:
        consumer = FrozenFinancialConsumer(consumer.predictor, readout=consumer.readout)
    with FinancialSession(
        tmp_path / "run",
        native=native,
        consumer=consumer,
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=CompositeScoreConfig(SCALERS, capacity=4),
        phase=PHASE,
        world="control_cache_fixture",
        fold="0",
        admission="m3",
        block_rows=4,
    ) as session:
        session.step(source["batches"][125], [], kind="warmup")
        reference = session._bundle(session.snapshot()["state"])["control"]
        before = len(loads)
        first = session.artifacts.read(reference, identity=session.contract_id, kind="control")
        expected = copy.deepcopy(first)
        first["observations"] = -1
        second = session.artifacts.read(reference, identity=session.contract_id, kind="control")
        assert second == expected
        assert second["observations"] == 4
        assert len(loads) - before == (0 if mechanism else 2)
