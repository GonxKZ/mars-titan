"""Selección con máscara y recibo declarado en una sesión FP32, sin entrenamiento."""

import pytest
import torch
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_mechanism_consumer import calibration
from test_mechanism_policy import policy
from test_native_episode_backend import native as native
from test_native_mechanism_collection import session

from mars_titan.memory.mechanism_collection_record import CollectedMechanisms
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.mechanism_policy import MechanismPolicy

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


@pytest.mark.parametrize("device", DEVICES)
def test_fp32_session_masks_unknown_actions_collects_their_proposals_and_recovers(
    native, four_flow_source, tmp_path, device
):
    model = (
        FinancialPredictor(
            FinancialConfig(
                four_flow_source["spec"],
                variant="mac_online",
                hidden_size=32,
                head=QUANTILE_HEAD,
                session_context="market_memory_v1",
            ),
            device=device,
            dtype=torch.float32,
        )
        .eval()
        .requires_grad_(False)
    )
    reader = (
        EpisodicReadout(
            EpisodicReadoutConfig(four_flow_source["codec"].fingerprint(), hidden_size=32),
            device=device,
            dtype=torch.float32,
        )
        .eval()
        .requires_grad_(False)
    )
    supplied = policy(device=device, dtype=torch.float32)
    supplied.support_max.fill_(100)
    supplied.point_coefficients.zero_()
    supplied.interval_coefficients.zero_()
    supplied.point_coefficients[-1, 10] = 2
    supplied.interval_coefficients[-1, 32] = 1e6
    mask = torch.zeros(36, dtype=torch.bool, device=device)
    mask[[0, 30]] = True
    selected = MechanismPolicy(
        **dict(supplied.named_buffers()),
        interval_weight=0.5,
        min_gain=0.0,
        action_mask=mask,
        fit_receipt_sha256="a" * 64,
    )
    # Este hash y los coeficientes son fixtures declarados, no un ajuste ejecutado.
    engine = FrozenMechanismConsumer(
        model, readout=reader, calibration=calibration(), policy=selected
    )
    data = (
        None,
        four_flow_source["spec"],
        four_flow_source["codec"],
        engine,
        four_flow_source["prefixes"],
    )
    output = tmp_path / "fp32"
    with session(native, output, data) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        issued = run.step(four_flow_source["batches"][126], [])
        trace = run.mechanism_event(2)["payload"]["selection"]["groups"][0]
        assert trace["action_id"] == 30 and trace["action"] == "parent/k4cr/none"
        assert trace["available"][32] is False and trace["scores"][32] > trace["scores"][30]
        packet = CollectedMechanisms.restore(run.mechanism_collection(2)["collection"]).payload()
        assert packet["output_dtype"] == "torch.float32"
        assert packet["groups"][0]["candidates"][32]["status"] == "observed"
        assert [q[2] for q in packet["groups"][0]["candidates"][30]["quantiles"]] == pytest.approx(
            [p.value for p in issued.predictions], rel=1e-5, abs=1e-5
        )
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in issued.predictions]
        run.step([], labels, kind="settlement", cutoff=moment(127))
        snapshot = run.snapshot()
        packet = run.mechanism_collection(2)
        assert run.mechanism_collection_metrics(2, device=device)[0]["mature_rows"] == 4
        assert engine.identity()["new_learned_selector"] is None
        assert engine.identity()["selector"]["coefficient_origin"] == (
            "supplied_with_declared_fit_receipt_not_independently_verified"
        )
    with session(native, output, data, resume=True) as run:
        assert run.snapshot() == snapshot
        assert run.mechanism_collection(2) == packet
