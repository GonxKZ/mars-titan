"""Recuperación del controlador regional con publicación y feedback nativos."""

from copy import deepcopy

import pytest
import torch
from test_financial_mechanisms import assert_same, session
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_mechanism_policy import policy
from test_native_episode_backend import native as native

from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.mechanism_policy import MechanismPolicy

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def engines(source, device="cpu", *, automatic=False, recurrence=None):
    model = (
        FinancialPredictor(
            FinancialConfig(
                source["spec"], variant="mac_online", hidden_size=32, head=QUANTILE_HEAD
            ),
            device=device,
            dtype=torch.float64,
        )
        .eval()
        .requires_grad_(False)
    )
    common = dict(codec_id=source["codec"].fingerprint(), hidden_size=32)
    reader = (
        EpisodicReadout(
            EpisodicReadoutConfig(
                **common,
                evidence_bridge=EvidenceBridgeConfig(
                    **common, cycles=0 if recurrence is not None else 1, recurrence=recurrence
                ),
            ),
            device=device,
            dtype=torch.float64,
        )
        .eval()
        .requires_grad_(False)
    )
    catalog = MechanismCatalog(reader_contract=reader.config.evidence_bridge.identity()["kind"])
    options = dict(action=MechanismAction(route="k4cr"))
    if automatic:
        parent = policy(device=device)
        parent.support_max.fill_(100)
        options = dict(
            policy=MechanismPolicy(
                **{name: value for name, value in parent.named_buffers()},
                interval_weight=parent.interval_weight,
                min_gain=parent.min_gain,
                catalog_id=catalog.fingerprint(),
            )
        )
    consumer = FrozenMechanismConsumer(model, readout=reader, catalog=catalog, **options)
    return None, source["spec"], source["codec"], consumer, source["prefixes"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("nonlinear", [False, True])
def test_native_controller_recovers_the_next_route_and_applies_feedback_once(
    native, four_flow_source, tmp_path, device, automatic, nonlinear
):
    data = engines(
        four_flow_source,
        device,
        automatic=automatic,
        recurrence=BridgeRecurrenceConfig(2, 1216) if nonlinear else None,
    )
    with session(native, tmp_path / "reference", data, block_rows=2) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        first = run.step(four_flow_source["batches"][126], [])
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in first.predictions]
        run.step([], labels, kind="settlement", cutoff=moment(127))
        expected = run.step(four_flow_source["batches"][128], [])
        expected_fast, expected_event = run.fast_state(), run.mechanism_event(4)
        assert expected_event["payload"]["work"]["mac_online_observations"] == 4
        assert expected_event["payload"]["work"]["bridge_value_projections"] > 0
    with session(native, tmp_path / "cut", data, block_rows=2) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        run.step(four_flow_source["batches"][126], [])

        def cut(boundary):
            if boundary == native.Boundary.committed:
                raise RuntimeError("Corte tras confirmar feedback")

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


@pytest.mark.parametrize("counter", [False, 0.0])
def test_regional_work_requires_an_integer_even_when_the_bank_is_empty(
    native, four_flow_source, tmp_path, counter
):
    data = engines(four_flow_source)
    with session(native, tmp_path / "counter", data) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        run.step(four_flow_source["batches"][126], [])
        bundle = run._bundle(run.snapshot()["state"])
        control = deepcopy(run._read(bundle["control"], "control"))
        assert control["work"]["bridge_value_projections"] == 0
        control["work"]["bridge_value_projections"] = counter
        record, _ = run._binding._record(run, bundle["generation"])
        with pytest.raises(ValueError, match="proyecciones"):
            run._binding._check_record(control, record, cutoff=moment(126))
