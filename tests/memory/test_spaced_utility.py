"""Evidencia E2 con pronósticos literales, sin entrenamiento ni utilidad financiera real."""

import copy
import hashlib
import json
import math

import pytest
import torch

from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot

DAY = 86_400_000_000
CODEC = "a" * 64


def snapshot(at, ids=(1,), codec=CODEC):
    keys = torch.zeros(len(ids), 64, dtype=torch.float32)
    keys[:, 0] = 1
    return EpisodeSnapshot.create(
        keys=keys,
        values=torch.zeros_like(keys),
        labels=torch.zeros(len(ids), dtype=torch.float64),
        ids=torch.tensor(ids, dtype=torch.int64),
        decision_at=torch.ones(len(ids), dtype=torch.int64),
        available_at=torch.zeros(len(ids), dtype=torch.int64),
        maturity_at=torch.full((len(ids),), 2, dtype=torch.int64),
        cutoff=at,
        codec_id=codec,
        context_id="b" * 64,
        dtype=torch.float64,
    )


def row(flow="US/A", *, point=1.0, deleted=((1, 2.0),), not_before=11):
    return dict(
        flow_id=flow, point=point, deleted_points=list(deleted), label_not_before=not_before
    )


def ledger(**options):
    from mars_titan.memory.spaced_utility import SpacedUtilityConfig, SpacedUtilityLedger

    return SpacedUtilityLedger(CODEC, SpacedUtilityConfig(**options))


def issue(run, at, rows, *, ids=(1,)):
    run.begin(at)
    run.issue(snapshot(at, ids), rows)
    return run.publish()


def test_two_assets_in_one_session_are_only_one_positive_confirmation():
    run = ledger()
    issue(run, 10, [row(), row("US/B", deleted=((1, 4.0),))])
    assert run.scores()[1]["sessions"] == 0
    run.begin(11)
    run.receive([("US/B", 10, 11, 0.0), ("US/A", 10, 11, 0.0)])
    assert run.scores()[1]["sessions"] == 0
    receipts = run.publish()
    assert receipts[0]["gains"][1] == dict(mean=2.0, rows=2)
    score = run.scores()[1]
    assert score["sessions"] == score["positive_sessions"] == 1
    assert score["mean_gain"] == 2 and not score["qualified"]


def test_positive_evidence_on_separated_days_is_required():
    run = ledger(min_gap_days=2)
    for at in (10, DAY + 10, 2 * DAY + 10):
        issue(run, at, [row(not_before=at + 1)])
        run.begin(at + 1)
        run.receive([("US/A", at, at + 1, 0.0)])
        run.publish()
        assert run.scores()[1]["qualified"] is (at == 2 * DAY + 10)


def test_partial_labels_wait_for_the_whole_session_and_censoring_is_not_a_target():
    run = ledger()
    issue(run, 10, [row(), row("US/B")])
    run.begin(11)
    run.receive([("US/A", 10, 11, 0.0)])
    assert run.publish() == []
    assert run.scores()[1]["sessions"] == 0
    run.begin(12)
    run.censor([("US/B", 10)], reason="excluded")
    receipt = run.publish()[0]
    assert receipt["gains"][1] == dict(mean=1.0, rows=1)
    assert receipt["censored"] == 1 and receipt["resolved"] == 1


def test_future_or_early_labels_cannot_change_the_state():
    run = ledger()
    issue(run, 10, [row(not_before=13)])
    run.begin(12)
    before = copy.deepcopy(run.scores())
    for available in (11, 14):
        with pytest.raises(ValueError):
            run.receive([("US/A", 10, available, 0.0)])
    assert run.scores() == before
    assert run.publish() == []


def test_recovery_preserves_partial_results_and_the_next_update():
    from mars_titan.memory.spaced_utility import SpacedUtilityLedger

    run = ledger()
    issue(run, 10, [row(), row("US/B")])
    run.begin(11)
    run.receive([("US/A", 10, 11, 0.0)])
    run.publish()
    restored = SpacedUtilityLedger.restore(CODEC, run.config, run.export())
    for item in (run, restored):
        item.begin(12)
        item.receive([("US/B", 10, 12, 0.0)])
        item.publish()
    assert restored.export() == run.export()


def test_scores_return_copies_and_missing_utility_is_not_fabricated():
    run = ledger()
    issue(run, 10, [row(deleted=())])
    values = run.scores()
    values[1]["mean_gain"] = 999
    assert run.scores()[1]["mean_gain"] is None
    run.begin(11)
    run.receive([("US/A", 10, 11, 0.0)])
    assert run.publish()[0]["gains"] == {}
    assert run.scores()[1]["sessions"] == 0


def test_two_markets_on_the_same_utc_day_do_not_create_spaced_evidence():
    run = ledger()
    run.begin(10)
    shared = snapshot(10)
    run.issue(shared, [row()])
    run.issue(shared, [row("CN/A")])
    run.publish()
    run.begin(11)
    run.receive([("CN/A", 10, 11, 0.0), ("US/A", 10, 11, 0.0)])
    run.publish()
    score = run.scores()[1]
    assert score["sessions"] == score["positive_sessions"] == 2
    assert not score["qualified"]


def test_late_old_session_uses_date_extrema_without_losing_spaced_confirmation():
    run = ledger(min_gap_days=2)
    issue(run, DAY + 10, [row(not_before=DAY + 11)])
    issue(run, 3 * DAY + 10, [row(not_before=3 * DAY + 11)])
    run.begin(3 * DAY + 11)
    run.receive([("US/A", 3 * DAY + 10, 3 * DAY + 11, 0.0)])
    run.publish()
    assert not run.scores()[1]["qualified"]
    run.begin(4 * DAY)
    run.receive([("US/A", DAY + 10, 4 * DAY, 0.0)])
    run.publish()
    assert run.scores()[1]["first_positive_day"] == 1
    assert run.scores()[1]["last_positive_day"] == 3
    assert run.scores()[1]["qualified"]


def test_negative_new_evidence_can_remove_qualification():
    run = ledger()
    for at in (10, DAY + 10):
        issue(run, at, [row(not_before=at + 1)])
        run.begin(at + 1)
        run.receive([("US/A", at, at + 1, 0.0)])
        run.publish()
    assert run.scores()[1]["qualified"]
    at = 2 * DAY + 10
    issue(run, at, [row(point=10.0, deleted=((1, 0.0),), not_before=at + 1)])
    run.begin(at + 1)
    run.receive([("US/A", at, at + 1, 0.0)])
    run.publish()
    assert run.scores()[1]["mean_gain"] < 0 and not run.scores()[1]["qualified"]


def test_eviction_does_not_resurrect_an_episode_when_its_old_outcome_arrives():
    run = ledger()
    issue(run, 10, [row()])
    issue(run, 12, [row("US/B", deleted=((2, 2.0),), not_before=13)], ids=(2,))
    run.begin(13)
    run.receive([("US/A", 10, 13, 0.0), ("US/B", 12, 13, 0.0)])
    receipts = run.publish()
    assert receipts[0]["gains"][1]["mean"] == 1
    assert set(run.scores()) == {2}
    assert run.scores()[2]["sessions"] == 1


def test_input_and_result_order_leave_the_same_published_state():
    def execute(reverse):
        run = ledger()
        rows = [row(), row("US/B", deleted=((1, 4.0),))]
        issue(run, 10, rows[::-1] if reverse else rows)
        run.begin(11)
        targets = [(r["flow_id"], 10, 11, 0.0) for r in rows]
        for item in targets[::-1] if reverse else targets:
            run.receive([item])
        run.publish()
        return run.export()

    assert execute(False) == execute(True)


def test_invalid_batch_is_atomic_and_emission_after_feedback_is_rejected():
    run = ledger()
    issue(run, 10, [row()])
    run.begin(11)
    with pytest.raises(ValueError):
        run.receive([("US/A", 10, 11, 0.0), ("US/MISSING", 10, 11, 0.0)])
    run.receive([("US/A", 10, 11, 0.0)])
    with pytest.raises(ValueError):
        run.issue(snapshot(11), [row("US/B", not_before=12)])
    receipt = run.publish()[0]
    assert receipt["resolved"] == 1 and run.resolved == 1


def test_unknown_future_maturity_can_only_be_censored():
    run = ledger()
    issue(run, 10, [row(not_before=None)])
    run.begin(11)
    with pytest.raises(ValueError):
        run.receive([("US/A", 10, 11, 0.0)])
    run.censor([("US/A", 10)], reason="finalized")
    receipt = run.publish()[0]
    assert receipt["resolved"] == 0 and receipt["censored"] == 1 and receipt["gains"] == {}


@pytest.mark.parametrize(
    "change",
    [
        dict(point=math.nan),
        dict(point=math.inf),
        dict(point=True),
        dict(point=10**1000),
        dict(deleted_points=[(2, 3.0)]),
        dict(deleted_points=[(1, 2.0), (1, 3.0)]),
        dict(label_not_before=10),
        dict(flow_id="UNKNOWN/A"),
        dict(target=1.0),
    ],
)
def test_invalid_emission_does_not_prune_registered_evidence(change):
    run = ledger()
    run.begin(10)
    invalid = row() | change
    with pytest.raises(ValueError):
        run.issue(snapshot(10), [invalid])
    assert run.scores() == {}
    run.issue(snapshot(10), [row()])
    run.publish()
    assert set(run.scores()) == {1}


@pytest.mark.parametrize(
    "options",
    [
        dict(min_gap_days=0),
        dict(min_gap_days=True),
        dict(capacity=1025),
        dict(positive_gain_margin=-1),
        dict(positive_gain_margin=math.nan),
        dict(max_pending_rows=0),
        dict(max_references=False),
    ],
)
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        ledger(**options)


@pytest.mark.parametrize("budget", ["max_pending_rows", "max_pending_sessions", "max_references"])
def test_pending_limits_reject_without_losing_the_original_session(budget):
    run = ledger(**{budget: 1})
    issue(run, 10, [row()])
    run.begin(11)
    with pytest.raises(ValueError):
        run.issue(snapshot(11), [row("US/B", not_before=12)])
    run.receive([("US/A", 10, 11, 0.0)])
    assert run.publish()[0]["gains"][1]["mean"] == 1


def test_snapshot_context_codec_and_bytes_are_checked():
    run = ledger()
    run.begin(10)
    for invalid in (snapshot(11), snapshot(10, codec="c" * 64)):
        with pytest.raises(ValueError):
            run.issue(invalid, [row()])
    invalid = snapshot(10)
    invalid._values[6].data[0] = 11
    with pytest.raises(ValueError):
        run.issue(invalid, [row()])
    run.issue(snapshot(10), [row()])
    other = snapshot(10, ids=(1, 2))
    with pytest.raises(ValueError):
        run.issue(other, [row("CN/A")])
    run.publish()


def test_overflow_is_rejected_before_committing_a_target():
    run = ledger()
    issue(run, 10, [row(point=-1e308)])
    run.begin(11)
    with pytest.raises(ValueError):
        run.receive([("US/A", 10, 11, 1e308)])
    assert run.resolved == 0
    run.censor([("US/A", 10)], reason="excluded")
    run.publish()


def resign(payload):
    result = copy.deepcopy(payload)
    state = {k: v for k, v in result.items() if k != "sha256"}
    result["sha256"] = hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return result


@pytest.mark.parametrize(
    "alter",
    [
        lambda s: s.update(issued=99),
        lambda s: s.update(schema_version=True),
        lambda s: s["episodes"][0][1].update(sessions=99),
        lambda s: s["episodes"][0][1].update(mean_gain=1),
        lambda s: s["last_issued"].update(US=11),
        lambda s: s["sessions"][0]["maturity"][0].__setitem__(1, 11),
        lambda s: s["sessions"][0]["rows"][0].update(status=["label", 10, 0.0]),
        lambda s: s["sessions"][0]["rows"][0].update(status=["excluded", 10, 3.0]),
    ],
)
def test_semantically_invalid_recovery_is_rejected_even_with_a_recomputed_hash(alter):
    from mars_titan.memory.spaced_utility import SpacedUtilityLedger

    run = ledger()
    issue(run, 10, [row()])
    state = run.export()
    alter(state)
    with pytest.raises(ValueError):
        SpacedUtilityLedger.restore(CODEC, run.config, resign(state))


def test_recovery_checks_hash_and_configuration_and_initial_state():
    from mars_titan.memory.spaced_utility import SpacedUtilityLedger

    run = ledger()
    assert SpacedUtilityLedger.restore(CODEC, run.config, run.export()).export() == run.export()
    issue(run, 10, [row()])
    state = run.export()
    state["issued"] += 1
    with pytest.raises(ValueError):
        SpacedUtilityLedger.restore(CODEC, run.config, state)
    with pytest.raises(ValueError):
        SpacedUtilityLedger.restore("c" * 64, run.config, run.export())
    run.begin(11)
    with pytest.raises(ValueError):
        run.export()
    run.publish()


def test_emission_and_resolution_are_not_replayed_and_caller_data_are_copied():
    run = ledger()
    rows = [row()]
    issue(run, 10, rows)
    rows[0]["point"] = 999
    run.begin(11)
    run.receive([("US/A", 10, 11, 0.0)])
    with pytest.raises(ValueError):
        run.receive([("US/A", 10, 11, 0.0)])
    assert run.publish()[0]["gains"][1]["mean"] == 1
    with pytest.raises(ValueError):
        run.begin(10)


def test_malformed_deep_state_is_rejected_before_recursive_serialization(monkeypatch):
    from mars_titan.memory import spaced_utility_state
    from mars_titan.memory.spaced_utility import SpacedUtilityLedger

    run = ledger()
    issue(run, 10, [row()])
    state = run.export()
    value = None
    for _ in range(1500):
        value = [value]
    state["sessions"][0]["rows"][0]["point"] = value

    def forbidden_serialization(_):
        pytest.fail("Un campo escalar inválido no puede recorrer la serialización recursiva")

    monkeypatch.setattr(spaced_utility_state, "_canonical", forbidden_serialization)
    with pytest.raises(ValueError):
        SpacedUtilityLedger.restore(CODEC, run.config, state)


def test_sessions_completed_together_publish_in_decision_then_market_order():
    run = ledger()
    issue(run, DAY + 10, [row("US/A", not_before=DAY + 11)])
    issue(run, 2 * DAY + 10, [row("CN/A", not_before=2 * DAY + 11)])
    run.begin(3 * DAY)
    run.receive([("CN/A", 2 * DAY + 10, 3 * DAY, 0.0), ("US/A", DAY + 10, 3 * DAY, 0.0)])
    assert [(r["market"], r["decision_at"]) for r in run.publish()] == [
        ("US", DAY + 10),
        ("CN", 2 * DAY + 10),
    ]


def test_qualified_evidence_and_recovered_state_do_not_alias_the_payload():
    from mars_titan.memory.spaced_utility import SpacedUtilityLedger

    run = ledger()
    for at in (10, DAY + 10):
        issue(run, at, [row(not_before=at + 1)])
        run.begin(at + 1)
        run.receive([("US/A", at, at + 1, 0.0)])
        run.publish()
    state = run.export()
    restored = SpacedUtilityLedger.restore(CODEC, run.config, state)
    assert restored.export() == state and restored.scores()[1]["qualified"]
    state["last_issued"]["US"] = 0
    state["episodes"][0][1]["mean_gain"] = -100
    assert restored.export() == run.export()


def test_exhausted_pending_budget_fails_before_copying_another_snapshot(monkeypatch):
    run = ledger(max_pending_rows=1)
    issue(run, 10, [row()])
    run.begin(11)

    def forbidden_copy(_):
        pytest.fail("Un presupuesto agotado no debe copiar otro snapshot")

    monkeypatch.setattr(EpisodeSnapshot, "export_cpu", forbidden_copy)
    with pytest.raises(ValueError):
        run.issue(snapshot(11), [row("US/B", not_before=12)])


def test_invalid_censor_identity_rejects_the_whole_batch():
    run = ledger()
    issue(run, 10, [row(), row("US/B")])
    run.begin(11)
    with pytest.raises(ValueError):
        run.censor([("US/A", 10), (["US/B"], 10)], reason="excluded")
    assert run.censored == 0
    run.receive([("US/A", 10, 11, 0.0), ("US/B", 10, 11, 0.0)])
    assert run.publish()[0]["resolved"] == 2


def test_valid_state_with_only_a_changed_checksum_is_rejected():
    from mars_titan.memory.spaced_utility import SpacedUtilityLedger

    run = ledger()
    issue(run, 10, [row()])
    state = run.export()
    state["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="huella"):
        SpacedUtilityLedger.restore(CODEC, run.config, state)


def test_evidence_and_recovery_do_not_consume_torch_randomness():
    from mars_titan.memory.spaced_utility import SpacedUtilityLedger

    before = torch.get_rng_state().clone()
    run = ledger()
    issue(run, 10, [row()])
    run = SpacedUtilityLedger.restore(CODEC, run.config, run.export())
    run.begin(11)
    run.receive([("US/A", 10, 11, 0.0)])
    run.publish()
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


def test_usage_counts_pending_rows_and_references_without_resolving_them():
    run = ledger()
    assert run.usage() == dict(episodes=0, pending_sessions=0, pending_rows=0, references=0)
    issue(run, 10, [row(), row("US/B", deleted=())])
    assert run.usage() == dict(episodes=1, pending_sessions=1, pending_rows=2, references=1)
    assert run.scores()[1]["mean_gain"] is None


def test_published_registry_can_follow_retention_without_reissuing_predictions():
    run = ledger()
    issue(run, 10, [row(), row("US/B")])
    run.begin(11)
    run.receive([("US/A", 10, 11, 0.0)])
    run.publish()
    run.synchronize(snapshot(11, ids=(2,)))
    assert set(run.scores()) == {2}
    assert run.usage()["pending_rows"] == 2
    run.begin(12)
    run.receive([("US/B", 10, 12, 0.0)])
    receipt = run.publish()[0]
    assert receipt["gains"][1]["mean"] == 1.0
    assert run.scores()[2]["mean_gain"] is None
    assert set(run.scores()) == {2}


def test_registry_synchronization_requires_a_published_matching_cut_and_codec():
    run = ledger()
    with pytest.raises(ValueError):
        run.synchronize(snapshot(10))
    issue(run, 10, [row()])
    state = run.export()
    for invalid in (snapshot(11), snapshot(10, codec="c" * 64)):
        with pytest.raises(ValueError):
            run.synchronize(invalid)
        assert run.export() == state
    run.begin(11)
    with pytest.raises(ValueError):
        run.synchronize(snapshot(11))


def test_registry_budget_fails_before_snapshot_copy(monkeypatch):
    run = ledger(capacity=1)
    run.begin(10)
    run.publish()
    data = snapshot(10, ids=(1, 2))
    monkeypatch.setattr(
        EpisodeSnapshot, "export_cpu", lambda *a: pytest.fail("Copia fuera de capacidad")
    )
    with pytest.raises(ValueError):
        run.synchronize(data)


def test_synchronized_registry_recovers_without_consuming_pending_results():
    from mars_titan.memory.spaced_utility import SpacedUtilityLedger

    run = ledger()
    issue(run, 10, [row()])
    run.synchronize(snapshot(10, ids=(1, 2)))
    restored = SpacedUtilityLedger.restore(CODEC, run.config, run.export())
    assert restored.usage() == run.usage()
    assert restored.export() == run.export()


def test_scoped_evidence_cannot_be_restored_as_another_stream_or_as_legacy():
    from mars_titan.memory.spaced_utility import SpacedUtilityLedger

    run = SpacedUtilityLedger(CODEC, scope_id="b" * 64)
    issue(run, 10, [row()])
    state = run.export()
    assert state["identity"]["scope_id"] == "b" * 64
    restored = SpacedUtilityLedger.restore(CODEC, run.config, state, scope_id="b" * 64)
    assert restored.export() == state
    for scope in (None, "c" * 64):
        with pytest.raises(ValueError):
            SpacedUtilityLedger.restore(CODEC, run.config, state, scope_id=scope)


def test_same_cut_feedback_cannot_issue_more_predictions():
    run = ledger()
    issue(run, 10, [row()])
    run.begin(11)
    run.publish()
    run.begin(11, feedback_only=True)
    with pytest.raises(ValueError):
        run.issue(snapshot(11), [row("US/B", not_before=12)])
    run.receive([("US/A", 10, 11, 0.0)])
    assert run.publish()[0]["resolved"] == 1
