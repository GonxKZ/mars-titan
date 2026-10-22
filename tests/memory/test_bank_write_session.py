"""Escrituras por banco confirmadas por la sesión nativa, con etiquetas manuales maduras."""

from dataclasses import replace

import pytest
import torch
from test_financial_mechanisms import assert_same, engines
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.native_mechanism_collection import MechanismCollectionConfig
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.models.titans.bank_readout import BankReadoutConfig
from mars_titan.models.titans.config import MemoryMultiscale
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rule", ["mean_partial", "per_bank"])
@pytest.mark.parametrize("admission", ["m0", "m1"])
def test_native_cut_recovers_all_bank_momentum_and_never_duplicates_feedback(
    native, four_flow_source, tmp_path, device, rule, admission
):
    original = engines(four_flow_source, device)
    config = replace(
        original[3].predictor.config,
        memory_multiscale=MemoryMultiscale(form="ema_bank"),
        memory_bank_readout=BankReadoutConfig(write_gradient=rule),
        regional_fusion=RegionalFusionConfig(topology="ring"),
    )
    model = (
        FinancialPredictor(config, device=device, dtype=torch.float64).eval().requires_grad_(False)
    )
    consumer = FrozenMechanismConsumer(model, action=original[3].action)

    def session(name, resume=False):
        return FinancialSession(
            tmp_path / name,
            native=native,
            consumer=consumer,
            codec=four_flow_source["codec"],
            prefixes=four_flow_source["prefixes"],
            retention=RetentionConfig(capacity=4, frontier=2, new_candidates=4),
            phase=FinancialPhase("validation", moment(125), moment(126), moment(130), moment(131)),
            admission=admission,
            block_rows=2,
            world="bank_write_fixture",
            fold="0",
            resume=resume,
            mechanism_collection=MechanismCollectionConfig(),
        )

    with session("reference") as run:
        first = run.step(four_flow_source["batches"][126], [])
        event = run.mechanism_event(1)
        assert event["payload"]["work"]["bank_gradient_matrices"] == 24
        assert event["payload"]["work"]["bank_momentum_updates"] == 24
        assert event["payload"]["work"]["bank_matrix_updates"] == 24
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in first.predictions]
        run.step([], labels, kind="settlement", cutoff=moment(127))
        expected = run.step(four_flow_source["batches"][128], [])
        state = run.fast_state()
        receipt = run.mechanism_collection(1)
    with session("cut") as run:
        run.step(four_flow_source["batches"][126], [])

        def fault(boundary):
            if boundary == native.Boundary.committed:
                raise RuntimeError("Corte después de confirmar los momentum de bancos")

        with pytest.raises(RuntimeError, match="Corte después"):
            run.step([], labels, kind="settlement", cutoff=moment(127), fault=fault)
    with session("cut", resume=True) as run:
        actual = run.step(four_flow_source["batches"][128], [])
        assert [(p.asset, p.value) for p in actual.predictions] == [
            (p.asset, p.value) for p in expected.predictions
        ]
        assert_same(run.fast_state(), state)
        assert run.mechanism_collection(1) == receipt
        assert run.snapshot()["applied"] == 4
        assert run.diagnostics()["admitted"] == (4 if admission == "m1" else 0)
        view = run.mechanism_collections_at((1, 3), through_generation=3, before=moment(129))
        assert len(view["collections"]) == 2
        assert [t["value"] for t in view["collections"][0]["supervision"]["outcomes"]] == [
            0.125
        ] * 4
