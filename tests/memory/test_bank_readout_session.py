"""Bancos neuronales en la sesión nativa, separados del banco episódico y sus etiquetas."""

from dataclasses import replace

import pytest
import torch
from test_financial_mechanisms import engines
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native
from test_native_mechanism_collection import session

from mars_titan.models.titans.bank_readout import BankReadoutConfig
from mars_titan.models.titans.config import MemoryMultiscale
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


@pytest.mark.parametrize("device", DEVICES)
def test_native_cut_preserves_historical_banks_next_read_and_actual_work(
    native, four_flow_source, tmp_path, device
):
    prior = engines(four_flow_source, device)
    cfg = replace(
        prior[3].predictor.config,
        memory_multiscale=MemoryMultiscale(form="ema_bank"),
        memory_bank_readout=BankReadoutConfig(),
    )
    model = FinancialPredictor(cfg, device=device, dtype=torch.float64).eval().requires_grad_(False)
    consumer = FrozenMechanismConsumer(model, action=prior[3].action)
    data = (*prior[:3], consumer, prior[4])
    with session(native, tmp_path / "reference", data, enabled=False) as run:
        first = run.step(four_flow_source["batches"][126], [])
        event = run.mechanism_event(1)
        assert event["payload"]["work"]["bank_function_evaluations"] == 36
        assert event["payload"]["work"]["bank_query_routes"] == 12
        assert event["payload"]["work"]["bank_matrix_updates"] == 24
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in first.predictions]
        run.step([], labels, kind="settlement", cutoff=moment(127))
        expected = run.step(four_flow_source["batches"][128], [])
        expected_points = [(p.asset, p.value) for p in expected.predictions]
        expected_fast = run.fast_state()
    with session(native, tmp_path / "cut", data, enabled=False) as run:
        run.step(four_flow_source["batches"][126], [])

        def cut(boundary):
            if boundary == native.Boundary.committed:
                raise RuntimeError("Corte técnico tras confirmar bancos")

        with pytest.raises(RuntimeError, match="Corte técnico"):
            run.step([], labels, kind="settlement", cutoff=moment(127), fault=cut)
    with session(native, tmp_path / "cut", data, enabled=False, resume=True) as run:
        actual = run.step(four_flow_source["batches"][128], [])
        assert [(p.asset, p.value) for p in actual.predictions] == expected_points
        from test_financial_mechanisms import assert_same

        assert_same(run.fast_state(), expected_fast)
        assert run.snapshot()["applied"] == 4
