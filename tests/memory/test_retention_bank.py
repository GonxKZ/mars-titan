"""Retención sobre registros nativos con objetivo calculado sobre E completo."""

import copy
import math
import os

import numpy as np
import pytest

from mars_titan.cm.medoids import MedoidBudgetExceeded
from mars_titan.memory.native_backend import load_native
from mars_titan.memory.retention_bank import RetentionBank, RetentionConfig


@pytest.fixture
def native():
    path = os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    if not path:
        pytest.skip("Falta el enlace nativo CPU compilado")
    return load_native(path)


def record(native, identity):
    value = native.MemoryRecord()
    value.id = identity
    value.decision_at = 2 * identity
    value.available_at = value.decision_at
    value.maturity_at = value.decision_at + 1
    value.key = [float(identity % 5), float(identity % 3), 1.0] + [0.0] * 61
    value.value = [float(identity)] * 64
    value.label, value.label_valid = identity * 0.25, True
    return value


def make_bank(native, policy, **changes):
    config = RetentionConfig(policy=policy, capacity=4, frontier=2, new_candidates=4, **changes)
    return RetentionBank(
        native, config, codec_id="c" * 64, world="fixture", partition="train", fold="0"
    )


@pytest.mark.parametrize("policy", ["reservoir", "uniform", "recent", "anchored"])
def test_proposals_keep_original_records_and_leave_previous_snapshot_unchanged(native, policy):
    bank = make_bank(native, policy)
    records = [record(native, i) for i in range(1, 9)]
    first = bank.propose(records[:4], confirmed_at=20)
    assert bank.seen == 0 and first.seen == 4
    result = first.propose(records[4:], confirmed_at=30)
    assert first.seen == 4 and result.seen == 8
    assert [r.id for r in first.records()] == [1, 2, 3, 4]
    for retained in result.records():
        original = records[retained.id - 1]
        assert retained.value == original.value
        assert retained.label == original.label
        assert retained.maturity_at == original.maturity_at
        assert retained.key == native.normalize_key(original.key)
    if policy == "recent":
        assert [r.id for r in result.records()] == [5, 6, 7, 8]


@pytest.mark.parametrize("policy", ["reservoir", "uniform", "recent", "anchored"])
def test_objective_includes_every_client_even_if_it_is_not_a_candidate(native, policy):
    first = make_bank(native, policy).propose(
        [record(native, i) for i in range(1, 5)], confirmed_at=10
    )
    incoming = [record(native, i) for i in range(5, 13)]
    points = [r.key for r in first.records()] + [native.normalize_key(r.key) for r in incoming]
    result = first.propose(incoming, confirmed_at=30)
    expected = math.fsum(min(math.dist(point, r.key) for r in result.records()) for point in points)
    assert result.receipt["client_ids"] == list(range(1, 13))
    assert result.receipt["objective"] == pytest.approx(expected, rel=1e-14, abs=0)
    assert result.receipt["background_backend"] == "numpy"
    assert result.receipt["coordinate_dtype"] == "float32"
    assert result.receipt["estimated_peak_bytes"] <= 64 * 1024**2
    if policy == "anchored":
        assert set(result.receipt["fixed_ids"]).issubset(result.receipt["retained_ids"])
        assert len(result.receipt["candidate_ids"]) <= 6


@pytest.mark.parametrize("policy", ["reservoir", "uniform", "recent", "anchored"])
def test_recovery_restores_retention_rng_and_continuation(native, policy):
    first = make_bank(native, policy).propose(
        [record(native, i) for i in range(1, 9)], confirmed_at=20
    )
    restored = make_bank(native, policy).restore(first.snapshot())
    incoming = [record(native, i) for i in range(9, 17)]
    expected = first.propose(incoming, confirmed_at=40)
    actual = restored.propose(incoming, confirmed_at=40)
    assert [r.id for r in actual.records()] == [r.id for r in expected.records()]
    assert actual.receipt == expected.receipt
    assert actual.snapshot()["retention_rng"] == expected.snapshot()["retention_rng"]
    assert actual.snapshot()["memory"] == expected.snapshot()["memory"]


def test_ineligible_labels_and_exhausted_budget_do_not_change_the_bank(native):
    bank = make_bank(native, "anchored", max_distance_pairs=1)
    first = bank.propose([record(native, i) for i in range(1, 5)], confirmed_at=10)
    incoming = [record(native, i) for i in range(5, 9)]
    before = first.snapshot()
    with pytest.raises(ValueError):
        first.propose(incoming, confirmed_at=11)
    with pytest.raises(MedoidBudgetExceeded):
        first.propose(incoming, confirmed_at=20)
    assert first.snapshot() == before


def test_identity_and_rng_json_types_are_checked_during_restore(native):
    bank = make_bank(native, "uniform")
    state = bank.snapshot()
    for name, changed in (("schema_version", True), ("native_sha256", "e" * 64)):
        invalid = copy.deepcopy(state)
        invalid["identity"][name] = changed
        with pytest.raises(ValueError):
            bank.restore(invalid)
    invalid = copy.deepcopy(state)
    invalid["retention_rng"]["has_uint32"] = False
    with pytest.raises(ValueError):
        bank.restore(invalid)


def test_uniform_rng_does_not_change_the_process_generator(native):
    before = np.random.get_state()
    bank = make_bank(native, "uniform")
    bank.propose([record(native, i) for i in range(1, 9)], confirmed_at=20)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[0] == after[0] and before[2:] == after[2:]


@pytest.mark.parametrize(
    "field,value",
    [
        ("after_seen", True),
        ("retained_ids", [True, 2, 3, 4]),
        ("objective", 0),
        ("confirmed_at", 11),
    ],
)
def test_receipt_types_and_counts_cannot_reinterpret_a_restored_bank(native, field, value):
    bank = make_bank(native, "recent").propose(
        [record(native, i) for i in range(1, 5)], confirmed_at=10
    )
    payload = bank.snapshot()
    payload["receipt"][field] = value
    with pytest.raises(ValueError):
        bank.restore(payload)
