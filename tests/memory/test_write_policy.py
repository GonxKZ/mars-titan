"""M2 con resultados manuales y tres índices, sin estimación de etiquetas."""

import copy
import importlib

import pytest
from test_native_episode_backend import native as native
from test_native_episode_backend import record


def api():
    try:
        return importlib.import_module("mars_titan.memory.write_policy")
    except ModuleNotFoundError:
        pytest.fail("Falta la política M2 de tres índices")


def bank(native, **options):
    module = api()
    return module.MatureErrorBank(
        native,
        module.MatureErrorConfig(**({"capacity": 4, "seed": 73} | options)),
        codec_id="a" * 64,
        world="fixture",
        partition="validation",
        fold="0",
    )


@pytest.mark.parametrize(
    "capacity,expected",
    [(4, (2, 1, 1)), (5, (3, 1, 1)), (6, (3, 2, 1)), (7, (3, 2, 2)), (1024, (512, 256, 256))],
)
def test_quotas_use_integer_largest_remainders(capacity, expected):
    config = api().MatureErrorConfig(capacity=capacity)
    assert tuple(config.quotas.values()) == expected
    config.quotas["selective"] = 0
    assert tuple(config.quotas.values()) == expected


def test_all_supported_capacities_have_positive_quotas_and_one_total_budget():
    for capacity in range(4, 1025):
        values = tuple(api().MatureErrorConfig(capacity=capacity).quotas.values())
        assert sum(values) == capacity and min(values) >= 1
        assert all(
            abs(4 * q - weight * capacity) < 4 for q, weight in zip(values, (2, 1, 1), strict=True)
        )


@pytest.mark.parametrize(
    "options",
    [
        {"capacity": 3},
        {"capacity": 1025},
        {"capacity": True},
        {"seed": -1},
        {"seed": 2**64},
        {"seed": True},
        {"max_working_bytes": 0},
        {"max_working_bytes": 64 * 1024**2 + 1},
    ],
)
def test_invalid_configurations_are_rejected(options):
    with pytest.raises(ValueError):
        api().MatureErrorConfig(**options)


def test_all_candidates_reach_three_indices_and_the_union_is_deduplicated(native):
    original = bank(native)
    before = original.snapshot()
    incoming = [record(native, i) for i in range(1, 5)]
    following = original.propose(incoming, errors={i: i / 8 for i in range(1, 5)}, confirmed_at=9)
    assert original.snapshot() == before
    assert following.seen == 4
    assert following.index_ids()["selective"] == (4,)
    assert following.index_ids()["recent"] == (4,)
    assert following.size == 3
    assert len({r.id for r in following.records()}) == following.size
    assert following.receipt["physical_slots"] == 4
    assert following.receipt["unique_episodes"] == 3
    assert following.receipt["index_offers"] == 12


def test_selective_index_keeps_exact_absolute_errors_and_canonical_ties(native):
    value = bank(native)
    value = value.propose(
        [record(native, i) for i in range(1, 5)],
        errors={1: -5.0, 2: 5.0, 3: 0.0, 4: 1e-50},
        confirmed_at=9,
    )
    assert value.index_ids()["selective"] == (1,)
    following = value.propose([record(native, 5)], errors={5: 4.0}, confirmed_at=11)
    assert following.index_ids()["selective"] == (1,)
    assert following.index_ids()["recent"] == (5,)
    assert following.snapshot()["scores"] == {1: 5.0}


def test_reservoir_matches_the_existing_native_algorithm(native):
    value = bank(native)
    scope = native.MemoryScope()
    scope.world, scope.partition, scope.fold, scope.representation = "oracle", "train", "0", "64"
    oracle = native.EpisodicMemory(scope, 73, 2, 2)
    for start in (1, 5, 9):
        incoming = [record(native, i) for i in range(start, start + 4)]
        cutoff = incoming[-1].maturity_at
        for row in incoming:
            oracle.write(row, cutoff)
        value = value.propose(
            incoming, errors={r.id: float(r.id) for r in incoming}, confirmed_at=cutoff
        )
        assert value.index_ids()["reservoir"] == tuple(
            sorted(r.id for r in oracle.retained_records())
        )
        assert value.seen == oracle.seen


@pytest.mark.parametrize("seed", [73, 2**64 - 1])
def test_snapshot_recovery_and_continuation_are_exact(native, seed):
    value = bank(native, seed=seed).propose(
        [record(native, i) for i in range(1, 9)],
        errors={i: float(i) for i in range(1, 9)},
        confirmed_at=17,
    )
    restored = bank(native, seed=seed).restore(value.snapshot())
    assert restored.snapshot() == value.snapshot()
    incoming = [record(native, 9), record(native, 10)]
    kwargs = dict(errors={9: -2.0, 10: 0.0}, confirmed_at=21)
    assert (
        restored.propose(incoming, **kwargs).snapshot()
        == value.propose(incoming, **kwargs).snapshot()
    )


def test_receipt_cannot_invent_episodes_before_the_first_offer(native):
    value = bank(native).propose([record(native, 1)], errors={1: 0.5}, confirmed_at=3)
    snapshot = value.snapshot()
    snapshot["receipt"]["before_ids"] = [999]
    snapshot["receipt"]["evicted_ids"] = [999]
    with pytest.raises(ValueError):
        bank(native).restore(snapshot)


def test_tiny_errors_are_ranked_in_float64(native):
    incoming = [record(native, 1), record(native, 2)]
    for row in incoming:
        row.label = 0.0
    value = bank(native).propose(incoming, errors={1: 0.0, 2: -1e-50}, confirmed_at=5)
    assert value.index_ids()["selective"] == (2,)
    assert value.snapshot()["scores"] == {2: 1e-50}


def test_duplicate_copies_must_match_float_bits(native):
    original = record(native, 1)
    original.value = [0.0] + [1.0] * 63
    value = bank(native).propose([original], errors={1: 0.5}, confirmed_at=3)
    replacement = bank(native)
    changed = record(native, 1)
    changed.value = [-0.0] + [1.0] * 63
    replacement._indices["selective"].retain_batch([changed], [1], 3)
    snapshot = value.snapshot()
    snapshot["indices"]["selective"] = replacement._indices["selective"].snapshot_bytes()
    with pytest.raises(ValueError, match="bits"):
        bank(native).restore(snapshot)


def test_proposal_budget_and_future_labels_preserve_the_original(native):
    value = bank(native, max_working_bytes=17 * 1024**2)
    before = value.snapshot()
    incoming = [record(native, i) for i in range(1, 513)]
    with pytest.raises(ValueError, match="presupuesto"):
        value.propose(incoming, errors={r.id: 0.1 for r in incoming}, confirmed_at=1025)
    with pytest.raises(ValueError):
        value.propose([record(native, 1)], errors={1: 0.1}, confirmed_at=2)
    assert value.snapshot() == before


def test_restore_checks_receipt_types_before_copying_unknown_objects(native):
    class CopyBomb:
        def __deepcopy__(self, memo):
            pytest.fail("No se debe copiar un objeto de recibo no admitido")

    value = bank(native).propose([record(native, 1)], errors={1: 0.5}, confirmed_at=3)
    snapshot = value.snapshot()
    snapshot["receipt"]["offered_ids"] = CopyBomb()
    with pytest.raises(ValueError):
        bank(native).restore(snapshot)


@pytest.mark.parametrize("errors", [{}, {1: True}, {1: float("nan")}, {1: float("inf")}, {2: 1.0}])
def test_invalid_error_evidence_preserves_the_original_bank(native, errors):
    value = bank(native)
    before = value.snapshot()
    with pytest.raises(ValueError):
        value.propose([record(native, 1)], errors=errors, confirmed_at=3)
    assert value.snapshot() == before


@pytest.mark.parametrize("change", ["extra", "identity", "roles", "score_type", "missing_score"])
def test_corrupt_snapshots_do_not_recover(native, change):
    value = bank(native).propose([record(native, 1)], errors={1: 0.5}, confirmed_at=3)
    snapshot = copy.deepcopy(value.snapshot())
    if change == "extra":
        snapshot["extra"] = 0
    elif change == "identity":
        snapshot["identity"]["config"]["seed"] = True
    elif change == "roles":
        snapshot["indices"]["selective"], snapshot["indices"]["recent"] = (
            snapshot["indices"]["recent"],
            snapshot["indices"]["selective"],
        )
    elif change == "score_type":
        snapshot["scores"][1] = True
    else:
        snapshot["scores"] = {}
    with pytest.raises((ValueError, RuntimeError)):
        bank(native).restore(snapshot)
