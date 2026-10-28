"""Fases E2 regionales comparadas con la sesión durable y sin entrenamiento."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_mars_titan_session_parity import raw
from test_native_episode_backend import native as native
from test_spaced_inference import PHASE, emitted, inference, sequence
from test_spaced_regional_session import regional_consumer
from test_spaced_session import feedback, open_session

from mars_titan.memory.financial_observations import ObservationEvent
from mars_titan.memory.spaced_retention import SpacedRetentionConfig
from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.training.financial_run import Paused

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


def regional_inference(native, source, consumer, **changes):
    return inference(native, source, consumer, reader_contract="regional", **changes)


@pytest.mark.parametrize("kind", ["ring", "j4", "adaptive_compact", "tree", "masked"])
@pytest.mark.parametrize(
    "policy", ["utility", "permuted_utility", "uniform_frontier", "recent", "reservoir"]
)
def test_regional_phase_reproduces_durable_points_bank_and_evidence(
    native, four_flow_source, tmp_path, kind, policy
):
    source = four_flow_source
    consumer = regional_consumer(source, kind)
    engine = regional_inference(
        native,
        source,
        consumer,
        retention=SpacedRetentionConfig(policy=policy, capacity=4, frontier=1),
    )
    metrics = engine._pass(PHASE, sequence(source))
    previous, points = [], []
    with open_session(
        native, source, consumer, tmp_path / "durable", policy=policy, capacity=4
    ) as session:
        for index in (125, 126, 127, 128, 129, 130):
            inputs = [] if index == 129 else source["batches"][index]
            result = session.step(
                inputs,
                feedback(native, previous, index),
                kind="settlement" if index == 129 else "decision",
                cutoff=moment(index),
            )
            previous = result.predictions
            points.extend((p.asset, p.decision_at, p.value) for p in previous)
        session.step([], [], kind="settlement", cutoff=moment(201), close_phase=True)
        assert emitted(engine) == points and len(points) == 20
        assert engine.spaced_state() == session.spaced_state()
        bundle = session._bundle(session.snapshot()["state"])
        bank, _ = session._bank(bundle["bank"])
        assert engine.spaced_snapshot()["bank"] == bank.snapshot()
    assert metrics["labels"] == 16 and metrics["spaced"]["censored"] == 4
    assert metrics["refinements"] == 20
    assert metrics["spaced_regional_work"]["diagnostic_refinements"] == 48
    assert metrics["spaced_regional_work"]["decision_events"] == 5


def test_future_block_is_refused_before_any_parent_preparation(
    native, four_flow_source, monkeypatch
):
    original = FinancialPredictor.prepare
    calls = []

    def counted(self, *args, **kwargs):
        calls.append("prepare")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FinancialPredictor, "prepare", counted)
    source = four_flow_source
    consumer = regional_consumer(source)
    engine = regional_inference(native, source, consumer)
    event = ObservationEvent(
        moment(125),
        (raw(source["batches"][125][:2]), raw(source["batches"][126][2:])),
        (),
        False,
    )
    with pytest.raises(ValueError):
        engine._pass(PHASE, [event])
    assert calls == []


@pytest.mark.parametrize("contract", [None, False, 1, "other"])
def test_unknown_reader_contract_is_not_inferred(native, four_flow_source, contract):
    consumer = regional_consumer(four_flow_source)
    with pytest.raises(ValueError):
        inference(native, four_flow_source, consumer, reader_contract=contract)


def test_regional_contract_does_not_accept_the_standard_architecture(native, four_flow_source):
    from test_financial_session_controls import paired_consumers

    with pytest.raises(ValueError):
        regional_inference(native, four_flow_source, paired_consumers(four_flow_source)["disabled"])


def test_irregular_blocks_keep_points_and_count_actual_projection_calls(native, four_flow_source):
    source = four_flow_source
    consumer = regional_consumer(source, "j4")
    events = sequence(source)
    changed = []
    for event in events:
        if event.inputs:
            index = next(i for i in (125, 126, 127, 128, 130) if moment(i) == event.at)
            rows = source["batches"][index]
            event = replace(event, inputs=(raw(rows[:2]), raw(rows[2:3]), raw(rows[3:])))
        changed.append(event)
    engines = [
        regional_inference(
            native,
            source,
            consumer,
            retention=SpacedRetentionConfig(policy="reservoir", capacity=16, frontier=1),
        )
        for _ in range(2)
    ]
    metrics = [engines[0]._pass(PHASE, events), engines[1]._pass(PHASE, changed)]
    left, right = emitted(engines[0]), emitted(engines[1])
    assert [(v[0], v[1]) for v in left] == [(v[0], v[1]) for v in right]
    torch.testing.assert_close(
        torch.tensor([v[2] for v in left], dtype=torch.float64),
        torch.tensor([v[2] for v in right], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    assert metrics[0]["spaced_regional_work"]["extra_pool_projection_calls"] == 3
    assert metrics[1]["spaced_regional_work"]["extra_pool_projection_calls"] == 9
    assert (
        metrics[0]["spaced_regional_work"]["diagnostic_refinements"]
        == metrics[1]["spaced_regional_work"]["diagnostic_refinements"]
    )


@pytest.mark.parametrize("limit", ["nodes", "memory"])
def test_regional_budget_rejects_before_advancing_the_next_parent_state(
    native, four_flow_source, monkeypatch, limit
):
    original = FinancialPredictor.prepare
    calls = []

    def counted(self, *args, **kwargs):
        calls.append("prepare")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FinancialPredictor, "prepare", counted)
    changes = (
        dict(recurrence=BridgeRecurrenceConfig(4, 12))
        if limit == "nodes"
        else dict(working_bytes=400 * 1024)
    )
    consumer = regional_consumer(four_flow_source, "j4", **changes)
    engine = regional_inference(native, four_flow_source, consumer)
    with pytest.raises(ValueError, match="presupuesto"):
        engine._pass(PHASE, sequence(four_flow_source))
    assert len(calls) == 2
    assert engine._ledger is None and engine._block_state is None and engine._regional_work is None
    with pytest.raises(ValueError):
        engine.spaced_state()


def test_interrupted_regional_pass_discards_partial_totals_and_restarts(native, four_flow_source):
    source = four_flow_source
    consumer = regional_consumer(source, "tree")
    engine = regional_inference(native, source, consumer)
    stop = SimpleNamespace(requested=False)

    def events():
        original = sequence(source)
        yield original[0]
        stop.requested = True
        yield original[1]

    with pytest.raises(Paused):
        engine._pass(PHASE, events(), stop=stop)
    assert engine._regional_work is None and engine._regional_event_work is None
    expected = regional_inference(native, source, consumer)._pass(PHASE, sequence(source))
    assert engine._pass(PHASE, sequence(source)) == expected


def test_late_financial_outcomes_do_not_change_already_emitted_regional_points(
    native, four_flow_source
):
    source = four_flow_source
    consumer = regional_consumer(source, "adaptive_compact")
    initial = sequence(source)
    altered = copy.deepcopy(initial)
    altered[3] = replace(
        altered[3], labels=tuple((flow, at, target + 1) for flow, at, target in altered[3].labels)
    )
    points = []
    for events in (initial, altered):
        engine = regional_inference(native, source, consumer)
        engine._pass(PHASE, events)
        points.append([v for v in emitted(engine) if v[1] <= altered[3].at])
    assert points[0] == points[1]


def test_loaded_quantile_regional_reader_executes_the_same_frozen_phase(
    native, four_flow_source, tmp_path
):
    from test_spaced_regional_selected import components
    from test_spaced_selected import load

    from mars_titan.training.titans_walk_forward import PredictionRows
    from tests.memory.spaced_selected_fixture import selected_fixture

    source = four_flow_source
    selected = selected_fixture(
        tmp_path, dict(source, components=components("adaptive_compact")), quantiles=True
    )
    loaded = load(selected, reader_contract="regional")
    engine = regional_inference(native, source, loaded.consumer, loss="pinball")
    rows = PredictionRows(True)
    result = engine._pass(PHASE, sequence(source), rows=rows)
    tables = rows.finish()
    assert rows.count == result["labels"] == 16
    assert sum(table.num_rows for table in tables) == 16
    assert result["spaced_regional_work"]["diagnostic_refinements"] == 48
    assert result["spaced"]["issued"] == 20


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["ring", "tree"])
def test_cuda_regional_phase_is_replayable_without_retaining_partial_state(
    native, four_flow_source, kind
):
    source = four_flow_source
    consumer = regional_consumer(source, kind, device="cuda:0")
    engine = regional_inference(native, source, consumer)
    result = engine._pass(PHASE, sequence(source))
    expected = engine.spaced_state()
    assert engine._pass(PHASE, sequence(source)) == result
    assert engine.spaced_state() == expected and engine._regional_work is None
