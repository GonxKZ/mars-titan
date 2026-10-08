"""Semillas del mecanismo independientes de identidad y sufijos del corpus."""

import pytest
import test_native_episode_backend as backend_fixtures

from mars_titan.memory.retention_bank import RetentionBank, RetentionConfig

native = backend_fixtures.native


def bank(module, *, codec="a" * 64, world="first", partition="train", contract="causal_v2"):
    return RetentionBank(
        module,
        RetentionConfig(capacity=2, frontier=1, new_candidates=2),
        codec_id=codec,
        world=world,
        partition=partition,
        fold="0",
        memory_contract=contract,
    )


def test_seeded_retention_is_paired_across_scopes_and_codec_identities(native):
    first, second = bank(native), bank(native, codec="b" * 64, world="another_arm")
    assert first.fingerprint() != second.fingerprint()
    for identifier in range(1, 17):
        incoming = [backend_fixtures.record(native, identifier)]
        first = first.propose(incoming, confirmed_at=2 * identifier + 1)
        second = second.propose(incoming, confirmed_at=2 * identifier + 1)
        assert [r.id for r in first.records()] == [r.id for r in second.records()]
    with pytest.raises(ValueError):
        first.restore(second.snapshot())


@pytest.mark.parametrize("partition", ["train", "validation", "calibration", "evaluation"])
def test_v2_supports_each_phase_and_recovers_exactly(native, partition):
    initial = bank(native, partition=partition)
    current = initial.propose([backend_fixtures.record(native, 1)], confirmed_at=3)
    restored = initial.restore(current.snapshot())
    assert [r.id for r in restored.records()] == [1]
    assert restored.seen == 1


def test_legacy_contract_is_separate_and_keeps_its_partitions(native):
    legacy = bank(native, contract="legacy_v1")
    causal = bank(native)
    with pytest.raises(ValueError):
        causal.restore(legacy.snapshot())
    with pytest.raises(ValueError):
        bank(native, contract="legacy_v1", partition="calibration")
