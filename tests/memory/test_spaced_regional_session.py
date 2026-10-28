"""Sesiones E2 regionales sobre datos y etiquetas manuales, con persistencia nativa."""

import copy
import json
from dataclasses import replace

import pytest
import torch
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_financial_session_controls import paired_consumers, tensor_leaves
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native
from test_spaced_regional_readout import options
from test_spaced_session import feedback, open_session

from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.spaced_retention import SpacedRetentionConfig
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


def regional_consumer(
    source, kind="ring", *, working_bytes=None, quantile=False, device="cpu", **changes
):
    parent = paired_consumers(source)["disabled"]
    config = parent.readout.config
    if working_bytes is not None:
        config = replace(config, max_working_bytes=working_bytes)
    bridge = EvidenceBridgeConfig(
        codec_id=config.codec_id,
        hidden_size=config.hidden_size,
        neighbors=config.neighbors,
        temperature=config.temperature,
        max_batch=config.max_batch,
        max_working_bytes=config.max_working_bytes,
        **(options(kind) | changes),
    )
    reader = (
        EpisodicReadout(replace(config, evidence_bridge=bridge), dtype=torch.float64, device=device)
        .eval()
        .requires_grad_(False)
    )
    predictor = parent.predictor
    if quantile or device != "cpu":
        predictor = (
            FinancialPredictor(
                replace(predictor.config, head=QUANTILE_HEAD) if quantile else predictor.config,
                local_control=predictor.local_control.config if predictor.local_control else None,
                dtype=torch.float64,
                device=device,
            )
            .eval()
            .requires_grad_(False)
        )
    return FrozenFinancialConsumer(predictor, readout=reader)


def test_regional_reservoir_matches_its_original_session_and_keeps_durable_evidence(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    consumer = regional_consumer(source)
    with (
        open_session(native, source, consumer, tmp_path / "plain", capacity=4) as plain,
        open_session(
            native, source, consumer, tmp_path / "e2", policy="reservoir", capacity=4
        ) as e2,
    ):
        left_previous, right_previous = [], []
        for at in (125, 126, 127, 128):
            left = plain.step(source["batches"][at], feedback(native, left_previous, at))
            right = e2.step(source["batches"][at], feedback(native, right_previous, at))
            assert [(p.asset, p.value) for p in left.predictions] == [
                (p.asset, p.value) for p in right.predictions
            ]
            left_state, right_state = (
                tensor_leaves(plain.fast_state()),
                tensor_leaves(e2.fast_state()),
            )
            assert left_state.keys() == right_state.keys()
            assert all(left_state[key].equal(right_state[key]) for key in left_state)
            assert [(r["sample_id"], r["label"]) for r in plain.retained_episodes()] == [
                (r["sample_id"], r["label"]) for r in e2.retained_episodes()
            ]
            left_previous, right_previous = left.predictions, right.predictions
        assert e2.spaced_state()["issued"] == 16
        assert e2.spaced_state()["resolved"] == 12
        assert len(e2.spaced_event()["emissions"]) == 4


def test_the_standard_window_adapter_does_not_become_partly_regional_by_inheritance(
    native, four_flow_source
):
    from test_spaced_inference import inference

    consumer = regional_consumer(four_flow_source)
    with pytest.raises(ValueError, match="ventana.*estándar"):
        inference(native, four_flow_source, consumer)


@pytest.mark.parametrize("kind", ["j4", "adaptive_compact", "tree", "masked"])
@pytest.mark.parametrize(
    "policy", ["utility", "permuted_utility", "uniform_frontier", "recent", "reservoir"]
)
def test_regional_reader_and_each_policy_recover_the_same_next_generation(
    native, four_flow_source, tmp_path, kind, policy
):
    source = four_flow_source
    consumer = regional_consumer(source, kind)
    expected = []
    for name in ("reference", "reopened"):
        with open_session(
            native, source, consumer, tmp_path / name, policy=policy, capacity=16
        ) as run:
            first = run.step(source["batches"][125], [])
            second = run.step(source["batches"][126], feedback(native, first.predictions, 126))
            if name == "reference":
                result = run.step(source["batches"][127], feedback(native, second.predictions, 127))
                expected = (
                    [(p.asset, p.value) for p in result.predictions],
                    run.spaced_state(),
                    run.snapshot(),
                )
        if name == "reopened":
            with open_session(
                native,
                source,
                consumer,
                tmp_path / name,
                policy=policy,
                capacity=16,
                resume=True,
            ) as run:
                result = run.step(source["batches"][127], feedback(native, second.predictions, 127))
                assert [(p.asset, p.value) for p in result.predictions] == expected[0]
                assert run.spaced_state() == expected[1] and run.snapshot() == expected[2]


def test_warmup_settlement_and_closure_do_not_invent_regional_diagnostics(
    native, four_flow_source, tmp_path
):
    from test_financial_session import moment

    source = four_flow_source
    consumer = regional_consumer(source, "tree")
    with FinancialSession(
        tmp_path / "run",
        native=native,
        consumer=consumer,
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=SpacedRetentionConfig(capacity=4, frontier=1),
        phase=FinancialPhase("validation", moment(125), moment(126), moment(200), moment(201)),
        world="spaced_fixture",
        fold="0",
        admission="m1",
        block_rows=4,
    ) as run:
        run.step(source["batches"][125], [], kind="warmup")
        assert run.spaced_state()["issued"] == 0 and not run.spaced_event()["emissions"]
        bundle = run._bundle(run.snapshot()["state"])
        work = run._read(bundle["control"], "control")["spaced_regional"]
        assert work["diagnostic_refinements"] == work["extra_pool_projection_calls"] == 0
        first = run.step(source["batches"][126], [])
        run.step(
            [], feedback(native, first.predictions, 127), kind="settlement", cutoff=moment(127)
        )
        assert not run.spaced_event()["emissions"]
        assert run.spaced_state()["resolved"] == 4
        run.step(source["batches"][128], [])
        closed = run.step([], [], kind="settlement", cutoff=moment(201), close_phase=True)
        assert len(closed.finalized) == 4 and run.spaced_state()["censored"] == 4


@pytest.mark.parametrize(
    "boundary",
    [
        "before_predictions",
        "predictions_ready",
        "record_written",
        "feedback_applied",
        "checkpoint_written",
        "before_commit",
        "committed",
    ],
)
def test_regional_faults_preserve_the_complete_native_generation(
    native, four_flow_source, tmp_path, boundary
):
    source = four_flow_source
    consumer = regional_consumer(source, "adaptive_compact")
    with open_session(native, source, consumer, tmp_path / "reference", policy="utility") as run:
        first = run.step(source["batches"][125], [])
        second = run.step(source["batches"][126], feedback(native, first.predictions, 126))
        run.step(source["batches"][127], feedback(native, second.predictions, 127))
        expected, evidence = run.snapshot(), run.spaced_state()
    with open_session(native, source, consumer, tmp_path / "fault", policy="utility") as run:
        first = run.step(source["batches"][125], [])
        second = run.step(source["batches"][126], feedback(native, first.predictions, 126))

        def fail(point):
            if point == getattr(native.Boundary, boundary):
                raise RuntimeError("Corte técnico regional")

        with pytest.raises(RuntimeError, match="Corte técnico regional"):
            run.step(source["batches"][127], feedback(native, second.predictions, 127), fault=fail)
    with open_session(
        native, source, consumer, tmp_path / "fault", policy="utility", resume=True
    ) as run:
        if boundary != "committed":
            run.step(source["batches"][127], feedback(native, second.predictions, 127))
        assert run.snapshot() == expected and run.spaced_state() == evidence


@pytest.mark.parametrize(
    "fault", ["diagnostic_counter", "boolean_schema", "ordinary_counter", "weights", "flows"]
)
def test_recalculated_artifact_hash_does_not_bypass_regional_control_verification(
    native, four_flow_source, tmp_path, fault
):
    from mars_titan.memory import spaced_session

    source = four_flow_source
    consumer = regional_consumer(source)
    with open_session(
        native, source, consumer, tmp_path / "run", policy="utility", capacity=16
    ) as run:
        first = run.step(source["batches"][125], [])
        second = run.step(source["batches"][126], feedback(native, first.predictions, 126))
        run.step(source["batches"][127], feedback(native, second.predictions, 127))
        bundle = run._bundle(run.snapshot()["state"])
        control = copy.deepcopy(run._read(bundle["control"], "control"))
        if fault == "diagnostic_counter":
            control["spaced_regional"]["diagnostic_refinements"] += 1
        elif fault == "boolean_schema":
            control["spaced_regional"]["schema_version"] = True
        elif fault == "ordinary_counter":
            control["bridge_candidate_queries"] += 1
        elif fault == "weights":
            control["regional_reads"]["blocks"][0]["reads"][0]["regional_weights"][0, 0, 0] += 0.5
        else:
            control["regional_reads"]["blocks"][0]["flow_ids"] = ("US/WRONG",) * 4
        forged = dict(bundle, control=run._stage(control, "control"))
        pending = run._read(bundle["pending"], "pending")
        with pytest.raises(ValueError):
            spaced_session.verify(
                run, forged, [(row["flow_id"], row["prediction_at"]) for row in pending["rows"]]
            )


def test_confirmed_regional_trace_is_tied_to_the_native_event_kind(
    native, four_flow_source, tmp_path
):
    from mars_titan.memory import spaced_session

    source = four_flow_source
    consumer = regional_consumer(source)
    with open_session(native, source, consumer, tmp_path / "run", policy="utility") as run:
        run.step(source["batches"][125], [])
        bundle = run._bundle(run.snapshot()["state"])
        record = json.loads(run._executor.record_json(run.snapshot()["cursor"]))
        record["event_kind"] = "warmup"
        with pytest.raises(ValueError, match="emisiones.*observaciones"):
            spaced_session.verify_confirmed(run, bundle, record)


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param(None, id="none"),
        pytest.param(False, id="boolean"),
        pytest.param(0, id="zero"),
        pytest.param("", id="empty"),
    ],
)
def test_invalid_native_kind_cannot_fall_back_to_the_inferred_event(
    native, four_flow_source, tmp_path, kind
):
    from mars_titan.memory import spaced_session

    source = four_flow_source
    consumer = regional_consumer(source)
    with open_session(native, source, consumer, tmp_path / "run", policy="utility") as run:
        run.step(source["batches"][125], [])
        bundle = run._bundle(run.snapshot()["state"])
        record = json.loads(run._executor.record_json(run.snapshot()["cursor"]))
        record["event_kind"] = kind
        with pytest.raises(ValueError):
            spaced_session.verify_confirmed(run, bundle, record)


@pytest.mark.parametrize("limit", ["nodes", "memory"])
def test_whole_diagnostic_budget_fails_before_mac_and_keeps_the_published_state(
    native, four_flow_source, tmp_path, monkeypatch, limit
):
    original = FinancialPredictor.prepare
    calls = []

    def counted(self, *args, **kwargs):
        calls.append("mac")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FinancialPredictor, "prepare", counted)
    changes = (
        dict(recurrence=BridgeRecurrenceConfig(4, 12))
        if limit == "nodes"
        else dict(working_bytes=400 * 1024)
    )
    source = four_flow_source
    consumer = regional_consumer(source, "j4", **changes)
    with open_session(native, source, consumer, tmp_path / "run", policy="utility") as run:
        first = run.step(source["batches"][125], [])
        second = run.step(source["batches"][126], feedback(native, first.predictions, 126))
        before, evidence, called = run.snapshot(), run.spaced_state(), len(calls)
        with pytest.raises(ValueError, match="presupuesto"):
            run.step(source["batches"][127], feedback(native, second.predictions, 127))
        assert len(calls) == called
    with open_session(
        native, source, consumer, tmp_path / "run", policy="utility", resume=True
    ) as recovered:
        assert recovered.snapshot() == before and recovered.spaced_state() == evidence
        assert len(calls) == called


def test_quantile_parent_keeps_the_same_issued_point_with_regional_diagnostics(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    consumer = regional_consumer(source, "tree", quantile=True)
    results = []
    for name, policy in (("plain", None), ("e2", "reservoir")):
        with open_session(native, source, consumer, tmp_path / name, policy=policy) as run:
            previous, points = [], []
            for at in (125, 126, 127):
                event = run.step(source["batches"][at], feedback(native, previous, at))
                points.extend((p.asset, p.value) for p in event.predictions)
                previous = event.predictions
            results.append(points)
    assert results[0] == results[1]


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["ring", "tree"])
def test_cuda_regional_session_recovers_the_next_native_generation(
    native, four_flow_source, tmp_path, kind
):
    source = four_flow_source
    consumer = regional_consumer(source, kind, device="cuda:0")
    with open_session(native, source, consumer, tmp_path / "run", policy="reservoir") as run:
        first = run.step(source["batches"][125], [])
        run.step(source["batches"][126], feedback(native, first.predictions, 126))
        expected, ledger = run.snapshot(), run.spaced_state()
    with open_session(
        native, source, consumer, tmp_path / "run", policy="reservoir", resume=True
    ) as run:
        assert run.snapshot() == expected and run.spaced_state() == ledger
        next_event = run.step(source["batches"][127], [])
        assert len(next_event.predictions) == 4
        assert run.spaced_state()["issued"] == 12
