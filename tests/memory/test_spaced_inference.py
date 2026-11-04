"""Paridad de E2 entre ventanas congeladas y la sesión durable, sin aprendizaje."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_financial_session import moment
from test_financial_session_controls import FLOWS, paired_consumers
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_mars_titan_session_parity import raw
from test_native_episode_backend import native as native
from test_spaced_session import feedback, open_session

from mars_titan.memory.financial_observations import ObservationEvent
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.memory.spaced_retention import POLICIES, SpacedRetentionConfig
from mars_titan.training.learning_hold import LearningHoldError
from mars_titan.training.mars_titan_run import ReadoutRecipe

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
PHASE = FinancialPhase("validation", moment(125), moment(125), moment(200), moment(201))


def inference(native, source, consumer, **changes):
    from mars_titan.training.spaced_inference import SpacedInference

    options = dict(
        prefixes=source["prefixes"],
        retention=SpacedRetentionConfig(capacity=16, frontier=1),
        native=native,
        codec=source["codec"],
        world="spaced_fixture",
        fold="0",
        audit=True,
    )
    loss = changes.pop("loss", "mae")
    options.update(changes)
    return SpacedInference(
        consumer,
        ReadoutRecipe(
            loss=loss,
            neighbors=4,
            block_rows=4,
            max_working_bytes=consumer.readout.config.max_working_bytes,
        ),
        **options,
    )


def sequence(source):
    previous = []
    result = []
    for index in (125, 126, 127, 128, 129, 130):
        inputs = () if index == 129 else (raw(source["batches"][index]),)
        labels = tuple((flow, at, 0.1 * (1 + int(flow[-1]))) for flow, at in previous)
        result.append(ObservationEvent(moment(index), inputs, labels, False))
        previous = [(flow, moment(index)) for flow in FLOWS] if inputs else []
    result.append(ObservationEvent(moment(201), (), (), True))
    return result


def emitted(engine):
    return [(e[2], e[3], e[4]) for e in engine.audit if e[0] == "prediction"]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("capacity", [4, 16])
def test_frozen_window_reproduces_durable_emissions_and_evidence(
    native, four_flow_source, tmp_path, policy, capacity
):
    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    engine = inference(
        native,
        source,
        consumer,
        retention=SpacedRetentionConfig(policy=policy, capacity=capacity, frontier=1),
    )
    metrics = engine._pass(PHASE, sequence(source))
    with open_session(
        native, source, consumer, tmp_path / "session", policy=policy, capacity=capacity
    ) as session:
        previous, points = [], []
        for index in (125, 126, 127, 128, 129, 130):
            inputs = [] if index == 129 else source["batches"][index]
            result = session.step(
                inputs,
                feedback(native, previous, index),
                kind="settlement" if index == 129 else "decision",
                cutoff=moment(index),
            )
            previous = result.predictions
            points.extend((p.asset, p.decision_at, p.value) for p in result.predictions)
        session.step([], [], kind="settlement", cutoff=moment(201), close_phase=True)
        assert emitted(engine) == points and len(points) == 20
        assert engine.spaced_state() == session.spaced_state()
        bundle = session._bundle(session.snapshot()["state"])
        bank, _ = session._bank(bundle["bank"])
        assert engine.spaced_snapshot()["bank"] == bank.snapshot()
        assert metrics["labels"] == metrics["admitted"] == 16
        assert metrics["unresolved"] == metrics["spaced"]["censored"] == 4
        assert metrics["spaced"]["resolved"] == 16


def test_new_pass_restarts_fast_memory_retention_and_evidence(native, four_flow_source):
    consumer = paired_consumers(four_flow_source)["disabled"]
    engine = inference(native, four_flow_source, consumer)
    first = engine._pass(PHASE, sequence(four_flow_source))
    points, state = emitted(engine), engine.spaced_state()
    engine.audit.clear()
    assert engine._pass(PHASE, sequence(four_flow_source)) == first
    assert emitted(engine) == points and engine.spaced_state() == state


def test_later_outcomes_cannot_change_earlier_emitted_points(native, four_flow_source):
    consumer = paired_consumers(four_flow_source)["disabled"]
    original = sequence(four_flow_source)
    altered = copy.deepcopy(original)
    altered[3] = replace(
        altered[3], labels=tuple((flow, at, target + 1) for flow, at, target in altered[3].labels)
    )
    values = []
    for events in (original, altered):
        engine = inference(native, four_flow_source, consumer)
        engine._pass(PHASE, events)
        values.append([p for p in emitted(engine) if p[1] <= altered[3].at])
    assert values[0] == values[1]


def test_window_requires_frozen_reader(native, four_flow_source):
    consumer = paired_consumers(four_flow_source)["disabled"]
    consumer.readout.step_logit.requires_grad_(True)
    with pytest.raises(ValueError):
        inference(native, four_flow_source, consumer)


def test_public_evaluation_checks_hold_before_reading_source(native, four_flow_source):
    consumer = paired_consumers(four_flow_source)["disabled"]
    engine = inference(native, four_flow_source, consumer)
    with pytest.raises(LearningHoldError):
        engine.evaluate(object())


def test_uncompleted_window_exposes_no_published_evidence(native, four_flow_source):
    consumer = paired_consumers(four_flow_source)["disabled"]
    engine = inference(native, four_flow_source, consumer)
    with pytest.raises(ValueError):
        engine.spaced_state()


def test_active_stability_is_rejected(native, four_flow_source):
    with pytest.raises(ValueError):
        inference(native, four_flow_source, paired_consumers(four_flow_source)["diagnostic"])


def test_diagnostics_must_match_the_issued_point(native, four_flow_source, monkeypatch):
    from mars_titan.training import spaced_inference

    consumer = paired_consumers(four_flow_source)["disabled"]
    engine = inference(native, four_flow_source, consumer)
    original = spaced_inference.apply_spaced_utility_readout

    def changed(*args, **kwargs):
        prediction, trace = original(*args, **kwargs)
        points = trace.points.detach().clone()
        points[:, 0] += 1
        return prediction, replace(trace, points=points)

    monkeypatch.setattr(spaced_inference, "apply_spaced_utility_readout", changed)
    with pytest.raises(ValueError, match="emitid"):
        engine._pass(PHASE, sequence(four_flow_source)[:1])
    with pytest.raises(ValueError):
        engine.spaced_state()


def test_interruption_discards_partial_evidence_and_the_phase_can_be_replayed(
    native, four_flow_source
):
    from mars_titan.training.financial_run import Paused

    consumer = paired_consumers(four_flow_source)["disabled"]
    engine = inference(native, four_flow_source, consumer)
    control = SimpleNamespace(requested=False)

    def interrupted():
        events = sequence(four_flow_source)
        yield events[0]
        control.requested = True
        yield events[1]

    with pytest.raises(Paused):
        engine._pass(PHASE, interrupted(), stop=control)
    with pytest.raises(ValueError):
        engine.spaced_state()
    assert engine._ledger is None and engine._block_state is None
    assert engine._pass(PHASE, sequence(four_flow_source))["spaced"]["issued"] == 20


@pytest.mark.parametrize("change", ["missing_close", "after_close", "no_labels"])
def test_unfinished_or_invalid_phase_does_not_publish_evidence(native, four_flow_source, change):
    consumer = paired_consumers(four_flow_source)["disabled"]
    engine = inference(native, four_flow_source, consumer)
    events = sequence(four_flow_source)
    if change == "missing_close":
        events = events[:-1]
    elif change == "after_close":
        events = [*events, ObservationEvent(moment(202), (), (), False)]
    else:
        events = [replace(event, labels=()) for event in events]
    with pytest.raises(ValueError):
        engine._pass(PHASE, events)
    with pytest.raises(ValueError):
        engine.spaced_state()


def test_prediction_rows_and_cpu_rng_are_unchanged_by_the_frozen_pass(native, four_flow_source):
    from mars_titan.training.titans_walk_forward import PredictionRows

    consumer = paired_consumers(four_flow_source)["disabled"]
    engine = inference(native, four_flow_source, consumer)
    rows = PredictionRows(False)
    rng = torch.get_rng_state().clone()
    metrics = engine._pass(PHASE, sequence(four_flow_source), rows=rows)
    tables = rows.finish()
    assert rows.count == metrics["labels"] == 16
    assert [row["prediction"] for table in tables for row in table.to_pylist()] == [
        point for _, _, point in emitted(engine)[:16]
    ]
    assert torch.equal(torch.get_rng_state(), rng)


def test_another_corpus_cannot_supply_expected_maturity(native, four_flow_source):
    consumer = paired_consumers(four_flow_source)["disabled"]
    prefixes = copy.copy(four_flow_source["prefixes"])
    prefixes.source_id = "0" * 64
    with pytest.raises(ValueError, match="prefijos"):
        inference(native, four_flow_source, consumer, prefixes=prefixes)


def test_real_observation_index_drives_the_frozen_adapter(
    native, four_flow_source, tmp_path, monkeypatch
):
    from mars_titan.data.input_policy import HISTORICAL_MASKED
    from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
    from mars_titan.memory.financial_observations import (
        FinancialObservationSource,
        prepare_observation_index,
    )
    from mars_titan.training import spaced_inference
    from mars_titan.training.corpus_inputs import CorpusDataset
    from mars_titan.training.partition_contract import LEGACY_BOUNDS

    # Edición sintética con etiquetas manuales. Sigue instalada la guarda de optimizadores.
    prefixes = four_flow_source["prefixes"]
    dataset = CorpusDataset(prefixes._manifest_paths[0], input_policy=HISTORICAL_MASKED)
    lower, upper, close = LEGACY_BOUNDS["validation"]
    phase = FinancialPhase("validation", lower, lower, upper, close)
    stream = FinancialObservationSource(
        dataset, prepare_observation_index(dataset, tmp_path / "index", phase=phase)
    )
    spec = stream.specification()
    source = dict(four_flow_source, spec=spec, codec=FrozenEpisodeCodec(spec))
    consumer = paired_consumers(source)["disabled"]
    engine = inference(native, source, consumer)
    monkeypatch.setattr(spaced_inference, "require_learning_allowed", lambda *_: None)
    metrics = engine.evaluate(stream)
    assert metrics["predictions"] == metrics["spaced"]["issued"] == 8
    assert metrics["labels"] == metrics["spaced"]["resolved"] == 4
    assert metrics["unresolved"] == metrics["spaced"]["censored"] == 4


def test_multiple_physical_blocks_share_the_snapshot_without_enabling_audit(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    engine = inference(native, source, consumer, audit=False)
    events = sequence(source)
    for position, index in enumerate((125, 126, 127, 128, 129, 130)):
        if index != 129:
            group = source["batches"][index]
            events[position] = replace(events[position], inputs=(raw(group[:2]), raw(group[2:])))
    engine._pass(PHASE, events)
    assert engine.audit is None
    with open_session(
        native,
        source,
        consumer,
        tmp_path / "session",
        policy="utility",
        capacity=16,
        block_rows=2,
    ) as session:
        previous = []
        for index in (125, 126, 127, 128, 129, 130):
            result = session.step(
                [] if index == 129 else source["batches"][index],
                feedback(native, previous, index),
                kind="settlement" if index == 129 else "decision",
                cutoff=moment(index),
            )
            previous = result.predictions
        session.step([], [], kind="settlement", cutoff=moment(201), close_phase=True)
        assert engine.spaced_state() == session.spaced_state()
        bundle = session._bundle(session.snapshot()["state"])
        bank, _ = session._bank(bundle["bank"])
        assert engine.spaced_snapshot()["bank"] == bank.snapshot()


def test_quantile_outputs_match_the_original_reservoir_without_extra_heads(
    native, four_flow_source
):
    from mars_titan.memory.retention_bank import RetentionConfig
    from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
    from mars_titan.training.mars_titan_run import MarsTitanInference
    from mars_titan.training.titans_walk_forward import PredictionRows

    source = four_flow_source
    model = (
        FinancialPredictor(
            FinancialConfig(
                source["spec"], variant="mac_online", hidden_size=32, head="quantile_head_v1"
            ),
            dtype=torch.float64,
            device="cpu",
        )
        .eval()
        .requires_grad_(False)
    )
    reader = paired_consumers(source)["disabled"].readout
    consumer = FrozenFinancialConsumer(model, readout=reader)
    engine = inference(
        native,
        source,
        consumer,
        loss="pinball",
        retention=SpacedRetentionConfig(policy="reservoir", capacity=16, frontier=1),
    )
    reference = MarsTitanInference(
        model,
        reader,
        engine.recipe,
        admission="m1",
        retention=RetentionConfig(capacity=16, frontier=1),
        native=native,
        codec=source["codec"],
        world="spaced_fixture",
        fold="0",
    )
    outputs = []
    for run in (reference, engine):
        rows = PredictionRows(True)
        run._pass(PHASE, sequence(source), rows=rows)
        outputs.append(rows.finish())
        assert rows.count == 16
    assert len(outputs[0]) == len(outputs[1]) == 1
    assert outputs[0][0].equals(outputs[1][0])
