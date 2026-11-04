"""Retención E2 con registros C++ reales y evidencia madura de fixtures, sin entrenamiento."""

import copy
import importlib

import numpy as np
import pytest
import torch
from test_retention_bank import native as native
from test_retention_bank import record
from test_spaced_utility import CODEC, DAY, row

from mars_titan.memory.financial_consumers import bank_snapshot
from mars_titan.memory.retention_bank import RetentionBank, RetentionConfig
from mars_titan.memory.spaced_utility import SpacedUtilityConfig, SpacedUtilityLedger

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


def api():
    try:
        return importlib.import_module("mars_titan.memory.spaced_retention")
    except ModuleNotFoundError:
        pytest.fail("Falta la retención E2 sobre el banco nativo")


def bank(native, policy="utility", **options):
    config = api().SpacedRetentionConfig(policy=policy, capacity=4, frontier=1, **options)
    return api().SpacedUtilityBank(
        native,
        config,
        codec_id=CODEC,
        model_id="d" * 64,
        world="fixture",
        partition="train",
        fold="0",
    )


def evidence(capacity=4, at=20, scope_id=None):
    value = SpacedUtilityLedger(CODEC, SpacedUtilityConfig(capacity=capacity), scope_id=scope_id)
    value.begin(at)
    value.publish()
    return value


def seed(native, policy="utility", **options):
    original = bank(native, policy, **options)
    result = original.propose(
        [record(native, i) for i in range(1, 5)],
        confirmed_at=20,
        evidence=evidence(scope_id=original.fingerprint()),
    )
    assert original.seen == 0
    assert result.bank.seen == 4
    assert set(result.evidence.scores()) == {1, 2, 3, 4}
    return result.bank, result.evidence


def qualify(memory, ledger, pairs=((1, 5.0), (2, 2.0), (3, 1.0))):
    for at in (30, DAY + 30):
        snap = bank_snapshot(
            memory,
            codec_id=CODEC,
            context_id="b" * 64,
            cutoff=at,
            dtype=torch.float64,
            device="cpu",
        )
        ledger.begin(at)
        ledger.issue(snap, [row(point=0.0, deleted=pairs, not_before=at + 1)])
        ledger.publish()
        ledger.begin(at + 1)
        ledger.receive([("US/A", at, at + 1, 0.0)])
        ledger.publish()
    return ledger


def test_utility_protects_spaced_evidence_and_reserves_a_new_episode(native):
    memory, ledger = seed(native)
    ledger = qualify(memory, ledger)
    before, previous_evidence = memory.snapshot(), ledger.export()
    incoming = [record(native, i) for i in (5, 6)]
    result = memory.propose(incoming, confirmed_at=ledger.clock, evidence=ledger)
    assert [r.id for r in result.bank.records()] == [1, 2, 3, 6]
    assert result.bank.receipt["protected_ids"] == [1, 2, 3]
    assert result.bank.receipt["frontier_ids"] == [6]
    assert memory.snapshot() == before and ledger.export() == previous_evidence
    assert set(result.evidence.scores()) == {1, 2, 3, 6}
    assert result.evidence.scores()[6]["mean_gain"] is None
    assert all(result.evidence.scores()[i]["qualified"] for i in (1, 2, 3))
    actual = result.bank.records()[-1]
    assert actual.key == native.normalize_key(incoming[-1].key)
    assert actual.value == incoming[-1].value and actual.label == incoming[-1].label


@pytest.mark.parametrize(
    "policy", ["utility", "permuted_utility", "uniform_frontier", "recent", "reservoir"]
)
def test_every_policy_keeps_capacity_real_membership_and_continuation(native, policy):
    memory, ledger = seed(native, policy)
    ledger = qualify(memory, ledger)
    first = memory.propose(
        [record(native, i) for i in range(5, 9)], confirmed_at=ledger.clock, evidence=ledger
    )
    restored = bank(native, policy).restore(first.bank.snapshot())
    restored.validate_evidence(first.evidence)
    for current in (first.evidence,):
        current.begin(current.clock + 1)
        current.publish()
    incoming = [record(native, i) for i in (9, 10)]
    expected = first.bank.propose(
        incoming, confirmed_at=first.evidence.clock, evidence=first.evidence
    )
    actual = restored.propose(incoming, confirmed_at=first.evidence.clock, evidence=first.evidence)
    assert len(actual.bank.records()) == 4 and actual.bank.seen == 10
    assert actual.bank.snapshot() == expected.bank.snapshot()
    assert actual.evidence.export() == expected.evidence.export()
    assert {r.id for r in actual.bank.records()} <= set(first.bank.receipt["retained_ids"]) | {
        9,
        10,
    }


def test_reservoir_control_matches_the_existing_causal_retention_records(native):
    memory, ledger = seed(native, "reservoir", seed=73)
    reference = RetentionBank(
        native,
        RetentionConfig(capacity=4, seed=73, frontier=1, new_candidates=1),
        codec_id=CODEC,
        world="fixture",
        partition="train",
        fold="0",
        memory_contract="causal_v2",
    )
    reference = reference.propose([record(native, i) for i in range(1, 5)], confirmed_at=20)
    for at, start in ((40, 5), (60, 13)):
        ledger.begin(at)
        ledger.publish()
        records = [record(native, i) for i in range(start, start + 8)]
        result = memory.propose(records, confirmed_at=at, evidence=ledger)
        memory, ledger = result.bank, result.evidence
        reference = reference.propose(records, confirmed_at=at)
        expected = sorted(reference.records(), key=lambda record: record.id)
        assert [r.id for r in memory.records()] == [r.id for r in expected]
        assert [r.key for r in memory.records()] == [r.key for r in expected]
        key = np.asarray([1.0] + [0.0] * 63, dtype=np.float32)
        assert [(n.record.id, n.similarity) for n in memory.query(key, at)] == [
            (n.record.id, n.similarity) for n in reference.query(key, at)
        ]


def test_bad_maturity_and_unpublished_evidence_leave_both_states_unchanged(native):
    memory, ledger = seed(native)
    before, previous = memory.snapshot(), ledger.export()
    incoming = [record(native, 5)]
    incoming[0].maturity_at = 21
    with pytest.raises(ValueError):
        memory.propose(incoming, confirmed_at=20, evidence=ledger)
    assert memory.snapshot() == before and ledger.export() == previous
    ledger.begin(21)
    with pytest.raises(ValueError):
        memory.propose(incoming, confirmed_at=21, evidence=ledger)


def test_new_retention_does_not_change_process_randomness(native):
    before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    memory, ledger = seed(native, "uniform_frontier")
    memory.propose([record(native, i) for i in (5, 6)], confirmed_at=ledger.clock, evidence=ledger)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert torch.equal(torch_before, torch.get_rng_state()) and not torch.cuda.is_initialized()


def test_permuted_utility_moves_the_control_ranking_but_not_the_real_statistics(native):
    memory, ledger = seed(native, "permuted_utility")
    qualify(memory, ledger)
    before = ledger.scores()
    rng_before = memory.snapshot()["rng"]
    result = memory.propose(
        [record(native, i) for i in (5, 6)], confirmed_at=ledger.clock, evidence=ledger
    )
    ranking = result.bank.receipt["ranking"]
    assert {source for _, source, _ in ranking} == {1, 2, 3}
    assert [gain for _, _, gain in ranking] == [5.0, 2.0, 1.0]
    assert all(gain == before[source]["mean_gain"] for _, source, gain in ranking)
    for identifier, score in result.evidence.scores().items():
        if identifier in before:
            assert score == before[identifier]
    assert result.bank.receipt["frontier_ids"] == [6]
    assert result.bank.snapshot()["rng"] != rng_before


def test_late_result_for_an_evicted_native_record_does_not_resurrect_it(native):
    memory, ledger = seed(native)
    qualify(memory, ledger)
    at = DAY + 40
    ledger.begin(at)
    snap = bank_snapshot(
        memory, codec_id=CODEC, context_id="b" * 64, cutoff=at, dtype=torch.float64, device="cpu"
    )
    ledger.issue(snap, [row("US/LATE", point=0.0, deleted=((4, 8.0),), not_before=at + 1)])
    ledger.publish()
    result = memory.propose([record(native, i) for i in (5, 6)], confirmed_at=at, evidence=ledger)
    assert 4 not in result.evidence.scores()
    result.evidence.begin(at + 2)
    result.evidence.receive([("US/LATE", at, at + 2, 0.0)])
    assert result.evidence.publish()[0]["gains"][4]["mean"] == 8.0
    result.bank.validate_evidence(result.evidence)
    assert 4 not in result.evidence.scores()


def test_exhausted_budget_rejects_before_exporting_evidence(native, monkeypatch):
    memory, ledger = seed(native, max_working_bytes=16 * 1024**2)
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "export", lambda: pytest.fail("Exportación anterior al presupuesto"))
        with pytest.raises(ValueError, match="presupuesto"):
            memory.propose([record(native, 5)] * 4096, confirmed_at=ledger.clock, evidence=ledger)


def test_evidence_from_another_predictor_is_rejected_even_with_identical_episode_ids(
    native, monkeypatch
):
    memory, ledger = seed(native)
    foreign = api().SpacedUtilityBank(
        native,
        memory.config,
        codec_id=CODEC,
        model_id="e" * 64,
        world="fixture",
        partition="train",
        fold="0",
    )
    foreign = foreign.propose(
        [record(native, i) for i in range(1, 5)],
        confirmed_at=20,
        evidence=evidence(scope_id=foreign.fingerprint()),
    ).bank
    assert [r.id for r in foreign.records()] == [r.id for r in memory.records()]
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "usage", lambda: pytest.fail("Uso de evidencia ajena"))
        with pytest.raises(ValueError, match="pertenece"):
            foreign.propose([record(native, 5)], confirmed_at=ledger.clock, evidence=ledger)
    with pytest.raises(ValueError):
        foreign.restore(memory.snapshot())


def test_future_evidence_is_rejected_before_inspecting_its_usage(native, monkeypatch):
    memory, ledger = seed(native)
    ledger.begin(21)
    ledger.publish()
    monkeypatch.setattr(ledger, "usage", lambda: pytest.fail("Uso de evidencia futura"))
    with pytest.raises(ValueError):
        memory.propose([record(native, 5)], confirmed_at=20, evidence=ledger)


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s["receipt"].update(after_seen=99),
        lambda s: s["receipt"].update(frontier_ids=[1]),
        lambda s: s["receipt"].update(protected_ids=[1, 2]),
        lambda s: s["receipt"]["ranking"][0].__setitem__(2, 0.0),
        lambda s: s["receipt"]["ranking"].reverse(),
        lambda s: s["receipt"]["ledger_usage"].update(references=999999),
        lambda s: s["rng"].update(has_uint32=False),
        lambda s: s.update(schema_version=True),
        lambda s: s["identity"].update(model_id="e" * 64),
    ],
)
def test_invalid_native_receipt_cannot_be_reinterpreted_with_a_new_checksum(native, change):
    from mars_titan.memory.spaced_retention_state import checksum

    memory, ledger = seed(native)
    qualify(memory, ledger)
    proposed = memory.propose(
        [record(native, i) for i in (5, 6)], confirmed_at=ledger.clock, evidence=ledger
    )
    payload = copy.deepcopy(proposed.bank.snapshot())
    change(payload)
    payload["sha256"] = checksum({k: v for k, v in payload.items() if k != "sha256"})
    with pytest.raises(ValueError):
        memory.restore(payload)


def test_deep_invalid_rank_is_rejected_before_recursive_checksum(native, monkeypatch):
    from mars_titan.memory import spaced_retention_state

    memory, ledger = seed(native)
    qualify(memory, ledger)
    result = memory.propose(
        [record(native, i) for i in (5, 6)], confirmed_at=ledger.clock, evidence=ledger
    )
    payload = result.bank.snapshot()
    value = None
    for _ in range(1500):
        value = [value]
    payload["receipt"]["ranking"][0][2] = value
    monkeypatch.setattr(
        spaced_retention_state,
        "_canonical",
        lambda *a: pytest.fail("Serialización recursiva prematura"),
    )
    with pytest.raises(ValueError):
        memory.restore(payload)


def test_one_slot_still_admits_new_evidence(native):
    config = api().SpacedRetentionConfig(capacity=1, frontier=1)
    memory = api().SpacedUtilityBank(
        native,
        config,
        codec_id=CODEC,
        model_id="d" * 64,
        world="fixture",
        partition="train",
        fold="0",
    )
    first = memory.propose(
        [record(native, 1)],
        confirmed_at=20,
        evidence=evidence(capacity=1, scope_id=memory.fingerprint()),
    )
    next_state = first.bank.propose([record(native, 2)], confirmed_at=20, evidence=first.evidence)
    assert [r.id for r in next_state.bank.records()] == [2]
    assert next_state.bank.receipt["protected_ids"] == []
    assert next_state.evidence.scores()[2]["mean_gain"] is None


def test_non_native_input_fails_without_mutating_the_proposal_parent(native):
    memory, ledger = seed(native)
    before = memory.snapshot()
    with pytest.raises(ValueError):
        memory.propose([object()], confirmed_at=ledger.clock, evidence=ledger)
    assert memory.snapshot() == before


def test_utility_ranking_changes_selection_when_qualified_episodes_exceed_slots(native):
    memory, ledger = seed(native)
    qualify(memory, ledger, ((1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)))
    result = memory.propose(
        [record(native, i) for i in (5, 6)], confirmed_at=ledger.clock, evidence=ledger
    )
    assert result.bank.receipt["protected_ids"] == [2, 3, 4]
    assert [r.id for r in result.bank.records()] == [2, 3, 4, 6]


def test_empty_snapshot_and_checksum_are_checked_separately(native):
    memory = bank(native)
    payload = memory.snapshot()
    assert memory.restore(payload).snapshot() == payload
    payload["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="huella"):
        memory.restore(payload)


@pytest.mark.parametrize(
    "options",
    [
        dict(policy="unknown"),
        dict(capacity=True),
        dict(capacity=0),
        dict(frontier=0),
        dict(seed=True),
        dict(max_working_bytes=1),
    ],
)
def test_invalid_retention_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        api().SpacedRetentionConfig(**options)
