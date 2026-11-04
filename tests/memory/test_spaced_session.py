"""E2 en el coordinador financiero real, con calendario y etiquetas manuales de fixtures."""

import copy
import hashlib
import json

import pytest
from test_financial_session import moment
from test_financial_session_controls import (
    four_flow_source as four_flow_source,
)
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_financial_session_controls import paired_consumers, tensor_leaves
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.memory.spaced_retention import SpacedRetentionConfig
from mars_titan.memory.spaced_utility import _canonical

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


def open_session(
    native, source, consumer, output, *, policy=None, resume=False, block_rows=4, capacity=4
):
    retention = (
        RetentionConfig(capacity=capacity, frontier=1, new_candidates=1)
        if policy is None
        else SpacedRetentionConfig(policy=policy, capacity=capacity, frontier=1)
    )
    return FinancialSession(
        output,
        native=native,
        consumer=consumer,
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=retention,
        phase=FinancialPhase("validation", moment(125), moment(125), moment(200), moment(201)),
        world="spaced_fixture",
        fold="0",
        admission="m1",
        block_rows=block_rows,
        resume=resume,
    )


def feedback(native, predictions, at):
    return [native.Feedback(p.id, 0, moment(at), 0.1 * (1 + int(p.asset[-1]))) for p in predictions]


def test_original_reservoir_with_diagnostics_keeps_emissions_records_and_fast_state(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    with (
        open_session(native, source, consumer, tmp_path / "plain") as plain,
        open_session(native, source, consumer, tmp_path / "spaced", policy="reservoir") as spaced,
    ):
        previous_plain, previous_spaced = [], []
        for at in (125, 126, 127, 128):
            left = plain.step(source["batches"][at], feedback(native, previous_plain, at))
            right = spaced.step(source["batches"][at], feedback(native, previous_spaced, at))
            assert [(p.asset, p.value) for p in left.predictions] == [
                (p.asset, p.value) for p in right.predictions
            ]
            a, b = tensor_leaves(plain.fast_state()), tensor_leaves(spaced.fast_state())
            assert set(a) == set(b) and all(a[key].equal(b[key]) for key in a)
            assert plain.diagnostics() == spaced.diagnostics()
            old = plain.retained_episodes()
            new = spaced.retained_episodes()
            assert [(e["sample_id"], e["label"]) for e in old] == [
                (e["sample_id"], e["label"]) for e in new
            ]
            previous_plain, previous_spaced = left.predictions, right.predictions
        assert spaced.spaced_state()["issued"] == 16
        assert spaced.spaced_state()["resolved"] == 12
        assert len(spaced.spaced_event()["emissions"]) == 4


def test_reopen_reproduces_the_next_prediction_and_evidence(native, four_flow_source, tmp_path):
    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    with open_session(native, source, consumer, tmp_path / "reference", policy="utility") as run:
        first = run.step(source["batches"][125], [])
        run.step(source["batches"][126], feedback(native, first.predictions, 126))
        next_expected = run.step(source["batches"][127], [])
        expected_points = [(p.asset, p.value) for p in next_expected.predictions]
        expected_ledger = run.spaced_state()
    with open_session(native, source, consumer, tmp_path / "run", policy="utility") as run:
        first = run.step(source["batches"][125], [])
        run.step(source["batches"][126], feedback(native, first.predictions, 126))
        before, ledger = run.snapshot(), run.spaced_state()
    with open_session(
        native, source, consumer, tmp_path / "run", policy="utility", resume=True
    ) as restored:
        assert restored.snapshot() == before and restored.spaced_state() == ledger
        next_event = restored.step(source["batches"][127], [])
        assert len(next_event.predictions) == 4
        assert restored.spaced_state()["issued"] == 12
        assert [(p.asset, p.value) for p in next_event.predictions] == expected_points
        assert restored.spaced_state() == expected_ledger


def test_settlement_after_decision_at_same_cut_and_phase_closure(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    with open_session(native, source, consumer, tmp_path / "run", policy="utility") as run:
        first = run.step(source["batches"][125], [])
        second = run.step(source["batches"][126], [])
        state = copy.deepcopy(run.fast_state())
        result = run.step(
            [], feedback(native, first.predictions, 126), kind="settlement", cutoff=moment(126)
        )
        assert len(result.applied) == 4 and run.spaced_state()["resolved"] == 4
        assert tensor_leaves(state).keys() == tensor_leaves(run.fast_state()).keys()
        closed = run.step([], [], kind="settlement", cutoff=moment(201), close_phase=True)
        assert len(closed.finalized) == len(second.predictions)
        assert run.spaced_state()["censored"] == 4


def test_recovery_refuses_another_retention_policy(native, four_flow_source, tmp_path):
    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    with open_session(native, source, consumer, tmp_path / "run", policy="utility") as run:
        run.step(source["batches"][125], [])
    with pytest.raises(ValueError):
        open_session(native, source, consumer, tmp_path / "run", policy="recent", resume=True)


@pytest.mark.parametrize(
    "policy", ["utility", "permuted_utility", "uniform_frontier", "recent", "reservoir"]
)
def test_every_policy_recovers_and_keeps_its_native_evidence_pair(
    native, four_flow_source, tmp_path, policy
):
    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    previous = []
    with open_session(
        native, source, consumer, tmp_path / policy, policy=policy, capacity=16
    ) as run:
        for at in (125, 126, 127, 128):
            event = run.step(source["batches"][at], feedback(native, previous, at))
            previous = event.predictions
        state, ledger = run.snapshot(), run.spaced_state()
        assert ledger["completed_sessions"] == 3
        assert sum(score["sessions"] for _, score in ledger["episodes"]) > 0
    with open_session(
        native, source, consumer, tmp_path / policy, policy=policy, capacity=16, resume=True
    ) as run:
        assert run.snapshot() == state and run.spaced_state() == ledger
        run.step([], feedback(native, previous, 129), kind="settlement", cutoff=moment(129))
        assert run.spaced_state()["resolved"] == 16


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
def test_fault_recovery_preserves_exact_next_generation(
    native, four_flow_source, tmp_path, boundary
):
    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    with open_session(native, source, consumer, tmp_path / "reference", policy="utility") as run:
        first = run.step(source["batches"][125], [])
        run.step(source["batches"][126], feedback(native, first.predictions, 126))
        expected, ledger = run.snapshot(), run.spaced_state()
    with open_session(native, source, consumer, tmp_path / "fault", policy="utility") as run:
        issued = run.step(source["batches"][125], [])

        def fail(point):
            if point == getattr(native.Boundary, boundary):
                raise RuntimeError("Corte técnico E2")

        with pytest.raises(RuntimeError, match="Corte técnico E2"):
            run.step(source["batches"][126], feedback(native, issued.predictions, 126), fault=fail)
    with open_session(
        native, source, consumer, tmp_path / "fault", policy="utility", resume=True
    ) as run:
        if boundary != "committed":
            run.step(source["batches"][126], feedback(native, issued.predictions, 126))
        assert run.snapshot() == expected and run.spaced_state() == ledger


def test_asset_order_does_not_change_values_or_evidence(native, four_flow_source, tmp_path):
    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    results = []
    for reverse in (False, True):
        with open_session(
            native, source, consumer, tmp_path / str(reverse), policy="utility", block_rows=2
        ) as run:
            previous, points = [], []
            for at in (125, 126, 127, 128):
                inputs = source["batches"][at]
                labels = feedback(native, previous, at)
                event = run.step(
                    list(reversed(inputs)) if reverse else inputs,
                    list(reversed(labels)) if reverse else labels,
                )
                previous = event.predictions
                points.append([(p.asset, p.value) for p in previous])
            results.append((points, run.spaced_state(), run.retained_episodes()))
    assert results[0] == results[1]


@pytest.mark.parametrize("part", ["ledger", "event"])
def test_pending_emitted_point_is_checked_against_native_prediction(
    native, four_flow_source, tmp_path, part
):
    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    with open_session(native, source, consumer, tmp_path / "run", policy="utility") as run:
        run.step(source["batches"][125], [])
        bundle = run._bundle(run.snapshot()["state"])
        payload = copy.deepcopy(run._read(bundle["spaced"], "spaced"))
        if part == "ledger":
            payload["ledger"]["sessions"][0]["rows"][0]["point"] += 1
        else:
            payload["event"]["emissions"][0]["point"] += 1
        state = {k: v for k, v in payload["ledger"].items() if k != "sha256"}
        payload["ledger"]["sha256"] = hashlib.sha256(_canonical(state).encode()).hexdigest()
        forged = dict(bundle, spaced=run._stage(payload, "spaced"))
        from mars_titan.memory import spaced_session

        record = json.loads(run._executor.record_json(run.snapshot()["cursor"]))
        with pytest.raises(ValueError, match="predicci"):
            spaced_session.verify_confirmed(run, forged, record)


def test_legacy_configuration_still_exposes_no_spaced_state(native, four_flow_source, tmp_path):
    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    with open_session(native, source, consumer, tmp_path / "plain") as run:
        with pytest.raises(ValueError, match="activado"):
            run.spaced_state()


def test_active_stability_is_not_an_implicit_extra_axis(native, four_flow_source, tmp_path):
    source = four_flow_source
    consumer = paired_consumers(source)["diagnostic"]
    with pytest.raises(ValueError, match="C desactivado"):
        open_session(native, source, consumer, tmp_path / "invalid", policy="utility")


def test_evidence_pending_keys_must_match_the_financial_queue(native, four_flow_source, tmp_path):
    from mars_titan.memory import spaced_session

    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    with open_session(native, source, consumer, tmp_path / "run", policy="utility") as run:
        run.step(source["batches"][125], [])
        bundle = run._bundle(run.snapshot()["state"])
        with pytest.raises(ValueError, match="pendientes financieros"):
            spaced_session.verify(run, bundle, [])
