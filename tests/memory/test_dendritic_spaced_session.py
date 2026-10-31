"""N2 y E2 sobre el ejecutor real, con supervisión manual y sin aprendizaje."""

from dataclasses import replace

import pytest
import torch
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from mars_titan.memory import dendritic_session
from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.spaced_retention import SpacedRetentionConfig
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from tests.memory.dendritic_fixtures import engine
from tests.memory.test_dendritic_session import source as source
from tests.memory.test_financial_session import moment
from tests.memory.test_financial_session_controls import (
    no_target_estimation as no_target_estimation,
)
from tests.memory.test_spaced_regional_readout import options
from tests.memory.test_spaced_session import feedback, open_session

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


def test_n2_with_original_reservoir_diagnostics_preserves_actual_predictions(
    native, source, tmp_path
):
    consumer = engine(source, "observable")
    with (
        open_session(native, source, consumer, tmp_path / "plain", capacity=16) as plain,
        open_session(
            native, source, consumer, tmp_path / "e2", policy="reservoir", capacity=16
        ) as spaced,
    ):
        previous = [], []
        for at in (125, 126, 127, 128):
            first = plain.step(source["batches"][at], feedback(native, previous[0], at))
            second = spaced.step(source["batches"][at], feedback(native, previous[1], at))
            assert [(p.asset, p.value) for p in first.predictions] == [
                (p.asset, p.value) for p in second.predictions
            ]
            assert plain.dendritic_event()["entries"] == spaced.dendritic_event()["entries"]
            previous = first.predictions, second.predictions
        assert spaced.spaced_state()["issued"] == 16
        assert spaced.spaced_state()["resolved"] == 12
        assert len(spaced.spaced_event()["emissions"]) == 4


@pytest.mark.parametrize(
    "policy", ["utility", "permuted_utility", "uniform_frontier", "recent", "reservoir"]
)
def test_each_policy_recovers_the_next_prediction_and_its_evidence(
    native, source, tmp_path, policy
):
    consumer = engine(source, "observable")
    results = []
    for name, reopen in (("continuous", False), ("resumed", True)):
        path = tmp_path / name
        session = open_session(
            native, source, consumer, path, policy=policy, block_rows=1, capacity=8
        )
        first = session.step(source["batches"][125], [])
        session.step(source["batches"][126], feedback(native, first.predictions, 126))
        before, ledger = session.snapshot(), session.spaced_state()
        if reopen:
            session.close()
            session = open_session(
                native, source, consumer, path, policy=policy, block_rows=1, capacity=8, resume=True
            )
            assert session.snapshot() == before and session.spaced_state() == ledger
        event = session.step(source["batches"][127], [])
        results.append(
            (
                [(p.asset, p.value) for p in event.predictions],
                session.spaced_state(),
                session.dendritic_event(),
            )
        )
        assert session.spaced_state()["issued"] == 12
        assert session.spaced_state()["resolved"] == 4
        session.close()
    assert results[0] == results[1]


def test_quantile_preparation_keeps_utility_in_its_named_field(native, source, tmp_path):
    scalar = engine(source, "observable")
    model = FinancialPredictor(
        replace(scalar.predictor.config, head=QUANTILE_HEAD), dtype=torch.float64
    )
    consumer = FrozenFinancialConsumer(model.eval().requires_grad_(False), readout=scalar.readout)
    session = open_session(
        native, source, consumer, tmp_path / "quantiles", policy="utility", capacity=8
    )
    rows = session._rows(source["batches"][125])
    cpu = session._decision_cpu_batch(rows)
    batch = DecisionBatch.from_validated(cpu, dtype=torch.float64)
    context_id = "a" * 64
    plan = dendritic_session.prepare(session, rows, context_id)
    bank, _ = session._bank(None)
    snapshot = session._binding.snapshot(bank, context_id, batch.prediction_at[0])
    result = session._binding.prepare_block(
        session,
        batch,
        model.initial_state(batch.flow_ids),
        context_id=context_id,
        snapshot=snapshot,
        warmup=False,
        selection=None,
        dendritic_context=plan.view(cpu),
    )
    assert result.quantiles.shape == (4, 5) and result.utility is not None
    assert len(result.utility.ledger_rows({flow: moment(126) for flow in batch.flow_ids})) == 4
    issued = session.step(source["batches"][125], [])
    session.step(source["batches"][126], feedback(native, issued.predictions, 126))
    assert session.spaced_state()["resolved"] == 4
    session.close()


def test_warmup_and_settlement_keep_dendritic_and_utility_states_separate(native, source, tmp_path):
    consumer = engine(source, "observable")
    session = FinancialSession(
        tmp_path / "warm",
        native=native,
        consumer=consumer,
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=SpacedRetentionConfig(capacity=8, frontier=2),
        phase=FinancialPhase("validation", moment(125), moment(126), moment(200), moment(201)),
        admission="m1",
        world="n2_e2_fixture",
        fold="0",
        block_rows=1,
    )
    result = session.step(source["batches"][125], [], kind="warmup")
    assert not result.predictions and session.dendritic_event() is not None
    assert session.spaced_state()["issued"] == 0
    issued = session.step(source["batches"][126], [])
    assert session.spaced_state()["issued"] == 4
    session.step(
        [], feedback(native, issued.predictions, 127), kind="settlement", cutoff=moment(127)
    )
    assert session.dendritic_event() is None and session.spaced_state()["resolved"] == 4
    session.close()


@pytest.mark.parametrize("kind", ["ring", "tree"])
def test_regional_e2_keeps_the_same_dendritic_context_after_resume(native, source, tmp_path, kind):
    scalar = engine(source, "observable")
    config = scalar.readout.config
    bridge = EvidenceBridgeConfig(
        codec_id=config.codec_id,
        hidden_size=config.hidden_size,
        neighbors=config.neighbors,
        temperature=config.temperature,
        max_batch=config.max_batch,
        max_working_bytes=config.max_working_bytes,
        **options(kind),
    )
    reader = (
        EpisodicReadout(replace(config, evidence_bridge=bridge), dtype=torch.float64)
        .eval()
        .requires_grad_(False)
    )
    consumer = FrozenFinancialConsumer(scalar.predictor, readout=reader)
    expected = []
    for name, resume in (("continuous", False), ("reopened", True)):
        path = tmp_path / name
        session = open_session(native, source, consumer, path, policy="utility", capacity=8)
        issued = session.step(source["batches"][125], [])
        session.step(source["batches"][126], feedback(native, issued.predictions, 126))
        assert session.spaced_state()["resolved"] == 4
        if resume:
            session.close()
            session = open_session(
                native, source, consumer, path, policy="utility", capacity=8, resume=True
            )
        result = session.step(source["batches"][127], [])
        assert session.spaced_event()["emissions"]
        expected.append(
            (
                [(p.asset, p.value) for p in result.predictions],
                session.spaced_state(),
                session.dendritic_event(),
            )
        )
        session.close()
    assert expected[0] == expected[1]
