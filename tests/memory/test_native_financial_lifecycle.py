"""Eventos v2 con callbacks escalares técnicos, sin modelos ni optimizadores."""

import json
import os

import pytest
from test_native_episode_backend import definition

from mars_titan.memory.native_backend import load_native


@pytest.fixture
def native():
    path = os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    if not path:
        pytest.skip("Falta el enlace nativo CPU compilado")
    module = load_native(path)
    if not hasattr(module, "PhaseContract"):
        pytest.fail("Falta el ciclo financiero v2")
    return module


def contract(native, *, max_pending=32768):
    value = definition(native)
    value.tasks = [native.Task("residual", 1)]
    value.initial_state_json = json.dumps(dict(fast=0, labels=0, excluded=0, closed=0))
    value.prediction_mode = native.PredictionMode.financial
    value.phase = native.PhaseContract("validation", 1, 10, 30, 40, "e" * 64)
    limits = value.limits
    limits.max_pending = max_pending
    value.limits = limits
    return value


def event(native, cursor, at, kind="decision", *, close=False):
    value = native.Cohort()
    value.cursor, value.cutoff = cursor, at
    value.kind = getattr(native.EventKind, kind)
    value.close_phase = close
    value.observations = (
        []
        if kind == "settlement"
        else [native.Observation("US/A", at, [1.0]), native.Observation("US/B", at, [2.0])]
    )
    return value


def callbacks(native, trace):
    def prepare(kind, rows, tasks, at, before, batch_rows):
        state = json.loads(before)
        trace.append((kind, at, len(rows), batch_rows))
        values = (
            [] if kind == native.EventKind.warmup else [r.features[0] + state["fast"] for r in rows]
        )
        state["fast"] += len(rows)
        return values, json.dumps(state)

    def resolve(proposal, labels, excluded, finalized):
        state = json.loads(proposal)
        state["labels"] += len(labels)
        state["excluded"] += len(excluded)
        state["closed"] += len(finalized)
        return json.dumps(state)

    return prepare, resolve


def test_warmup_decisions_and_settlement_preserve_separate_counters(native, tmp_path):
    trace = []
    run = native.Executor(str(tmp_path / "run"), contract(native), *callbacks(native, trace), False)
    warmup = run.step(event(native, 0, 2, "warmup"), [], 1)
    assert not warmup.predictions and not run.pending()
    first = run.step(event(native, 1, 10), [], 1)
    assert [p.value for p in first.predictions] == [3.0, 4.0]
    second = run.step(event(native, 2, 20), [], 2)
    labels = [native.Feedback(p.id, 0, 15, 7.0) for p in first.predictions]
    settled = run.step(event(native, 3, 25, "settlement"), labels)
    assert not settled.predictions and len(settled.applied) == 2
    assert len(trace) == 3
    closed = run.step(event(native, 4, 40, "settlement", close=True), [])
    assert len(closed.finalized) == 2
    assert {r.prediction.id for r in closed.finalized} == {p.id for p in second.predictions}
    assert all(r.closed_at == 40 for r in closed.finalized)
    snapshot = json.loads(run.snapshot_json())
    assert snapshot["observed"] == 6
    assert snapshot["issued"] == 4 and snapshot["applied"] == 2 and snapshot["finalized"] == 2
    assert snapshot["phase_closed"] is True
    assert snapshot["state"] == dict(fast=6, labels=2, excluded=0, closed=2)
    assert not run.pending()
    run.close()
    recovered = native.Executor(
        str(tmp_path / "run"), contract(native), *callbacks(native, []), True
    )
    assert json.loads(recovered.snapshot_json()) == snapshot
    with pytest.raises((ValueError, RuntimeError)):
        recovered.step(event(native, 5, 41, "settlement"), [])
    recovered.close()


def exclusion(native, asset="US/A", at=10, *, pairs=125, variance=None):
    reason = (
        native.PrefixReason.insufficient_pairs
        if variance is None
        else native.PrefixReason.zero_market_variance
    )
    return native.PrefixExclusion(
        asset, native.Task("residual", 1), at, reason, pairs, variance, "d" * 64
    )


def test_prefix_exclusion_occurs_after_emission_and_survives_recovery(native, tmp_path):
    trace = []
    output = str(tmp_path / "run")
    run = native.Executor(output, contract(native), *callbacks(native, trace), False)
    cohort = event(native, 0, 10)
    cohort.prefix_exclusions = [exclusion(native)]
    result = run.step(cohort, [])
    assert len(result.predictions) == 2 and len(result.excluded) == 1
    assert result.excluded[0].prediction.value == result.predictions[0].value
    assert [p.asset for p in run.pending()] == ["US/B"]
    assert json.loads(run.snapshot_json())["state"] == dict(fast=2, labels=0, excluded=1, closed=0)
    assert len(trace) == 1
    run.close()
    run = native.Executor(output, contract(native), *callbacks(native, []), True)
    before = run.snapshot_json()
    duplicate = event(native, 1, 20, "settlement")
    duplicate.prefix_exclusions = [exclusion(native)]
    with pytest.raises(ValueError):
        run.step(duplicate, [])
    assert run.snapshot_json() == before
    run.close()


@pytest.mark.parametrize(
    "pairs,variance",
    [(126, None), (253, None), (125, 0.0), (126, 1.0), (126, float("nan")), (126, -1.0)],
)
def test_invalid_prefix_evidence_fails_before_prepare(native, tmp_path, pairs, variance):
    trace = []
    run = native.Executor(str(tmp_path / "run"), contract(native), *callbacks(native, trace), False)
    cohort = event(native, 0, 10)
    cohort.prefix_exclusions = [exclusion(native, pairs=pairs, variance=variance)]
    with pytest.raises(ValueError):
        run.step(cohort, [])
    assert not trace and run.cursor == 0
    run.close()


def test_prefix_resolution_releases_budget_without_hiding_predictions(native, tmp_path):
    trace = []
    run = native.Executor(
        str(tmp_path / "run"), contract(native, max_pending=1), *callbacks(native, trace), False
    )
    cohort = event(native, 0, 10)
    cohort.prefix_exclusions = [exclusion(native)]
    result = run.step(cohort, [])
    assert len(result.predictions) == 2 and len(run.pending()) == 1
    run.close()


def test_label_and_prefix_cannot_resolve_same_prediction(native, tmp_path):
    run = native.Executor(str(tmp_path / "run"), contract(native), *callbacks(native, []), False)
    first = run.step(event(native, 0, 10), [])
    cohort = event(native, 1, 20, "settlement")
    cohort.prefix_exclusions = [exclusion(native)]
    with pytest.raises(ValueError):
        run.step(cohort, [native.Feedback(first.predictions[0].id, 0, 15, 1.0)])
    assert len(run.pending()) == 2
    run.close()


@pytest.mark.parametrize(
    "phase",
    [
        ("selection", 1, 10, 30, 40, "e" * 64),
        ("validation", 0, 10, 30, 40, "e" * 64),
        ("validation", 11, 10, 30, 40, "e" * 64),
        ("validation", 1, 30, 30, 40, "e" * 64),
        ("validation", 1, 10, 30, 29, "e" * 64),
        ("validation", 1, 10, 30, 40, ""),
    ],
)
def test_invalid_phase_is_rejected_without_creating_output(native, tmp_path, phase):
    value = contract(native)
    value.phase = native.PhaseContract(*phase)
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        native.Executor(str(output), value, *callbacks(native, []), False)
    assert not output.exists()


def test_labels_maturing_at_phase_boundary_remain_pending(native, tmp_path):
    run = native.Executor(str(tmp_path / "run"), contract(native), *callbacks(native, []), False)
    issued = run.step(event(native, 0, 10), [])
    with pytest.raises(ValueError):
        run.step(
            event(native, 1, 30, "settlement"),
            [native.Feedback(issued.predictions[0].id, 0, 30, 1.0)],
        )
    assert len(run.pending()) == 2
    result = run.step(event(native, 1, 40, "settlement", close=True), [])
    assert len(result.finalized) == 2 and not result.applied
    run.close()


@pytest.mark.parametrize("tamper", ["observed", "excluded", "phase_closed", "last_event"])
def test_recovery_rejects_forged_financial_counters_even_after_rehash(native, tmp_path, tamper):
    import hashlib

    output = tmp_path / "run"
    run = native.Executor(str(output), contract(native), *callbacks(native, []), False)
    run.step(event(native, 0, 10), [])
    run.close()
    checkpoint = json.loads((output / "checkpoint-1.json").read_text())
    checkpoint[tamper] = {
        "observed": 3,
        "excluded": 1,
        "phase_closed": True,
        "last_event": "warmup",
    }[tamper]
    blob = json.dumps(checkpoint, sort_keys=True, separators=(",", ":")).encode()
    (output / "checkpoint-1.json").write_bytes(blob)
    head = json.loads((output / "latest.json").read_text())
    head["checkpoint_sha256"] = hashlib.sha256(blob).hexdigest()
    (output / "latest.json").write_text(json.dumps(head))
    with pytest.raises(ValueError):
        native.Executor(str(output), contract(native), *callbacks(native, []), True)


@pytest.mark.parametrize(
    "kind,at,close",
    [
        ("warmup", 10, False),
        ("decision", 30, False),
        ("settlement", 25, True),
        ("decision", 20, True),
        ("settlement", 41, True),
    ],
)
def test_phase_mismatches_fail_before_preparation(native, tmp_path, kind, at, close):
    trace = []
    run = native.Executor(str(tmp_path / "run"), contract(native), *callbacks(native, trace), False)
    with pytest.raises((ValueError, RuntimeError)):
        run.step(event(native, 0, at, kind, close=close), [])
    assert trace == []
    assert run.cursor == 0
    run.close()


def test_pending_without_evidence_is_kept_and_excess_fails_before_publication(native, tmp_path):
    trace = []
    run = native.Executor(
        str(tmp_path / "run"), contract(native, max_pending=2), *callbacks(native, trace), False
    )
    first = run.step(event(native, 0, 10), [])
    before = run.snapshot_json()
    with pytest.raises(ValueError):
        run.step(event(native, 1, 20), [])
    assert run.snapshot_json() == before
    assert len(trace) == 1
    labels = [native.Feedback(p.id, 0, 15, 1.0) for p in first.predictions]
    run.step(event(native, 1, 20), labels)
    assert len(run.pending()) == 2
    run.close()


def test_same_cutoff_settlement_runs_after_emission_without_another_prepare(native, tmp_path):
    trace = []
    run = native.Executor(str(tmp_path / "run"), contract(native), *callbacks(native, trace), False)
    issued = run.step(event(native, 0, 10), [])
    run.step(event(native, 1, 20), [])
    labels = [native.Feedback(p.id, 0, 15, 2.0) for p in issued.predictions]
    run.step(event(native, 2, 20, "settlement"), labels)
    assert len(trace) == 2 and len(run.pending()) == 2
    with pytest.raises(ValueError):
        run.step(event(native, 3, 20), [])
    run.close()


def test_terminal_settlement_requires_explicit_closure_before_publication(native, tmp_path):
    run = native.Executor(str(tmp_path / "run"), contract(native), *callbacks(native, []), False)
    try:
        run.step(event(native, 0, 10), [])
        before = run.snapshot_json()
        with pytest.raises(ValueError):
            run.step(event(native, 1, 40, "settlement"), [])
        assert run.snapshot_json() == before
        closed = run.step(event(native, 1, 40, "settlement", close=True), [])
        assert len(closed.finalized) == 2 and not run.pending()
    finally:
        run.close()


@pytest.mark.parametrize(
    "point",
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
def test_administrative_close_recovers_as_one_generation(native, tmp_path, point):
    output = str(tmp_path / "run")
    run = native.Executor(output, contract(native), *callbacks(native, []), False)
    run.step(event(native, 0, 10), [])

    def fail(boundary):
        if boundary == getattr(native.Boundary, point):
            raise RuntimeError("Corte de fixture")

    with pytest.raises(RuntimeError, match="Corte de fixture"):
        run.step(event(native, 1, 40, "settlement", close=True), [], 1, fail)
    run.close()
    recovered = native.Executor(output, contract(native), *callbacks(native, []), True)
    if point != "committed":
        assert len(recovered.pending()) == 2
        recovered.step(event(native, 1, 40, "settlement", close=True), [])
    assert not recovered.pending()
    assert json.loads(recovered.snapshot_json())["finalized"] == 2
    recovered.close()
