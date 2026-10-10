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


@pytest.mark.parametrize(
    "field",
    [
        "policy",
        "before_seen",
        "after_seen",
        "confirmed_at",
        "client_ids",
        "fixed_ids",
        "candidate_ids",
        "retained_ids",
        "objective",
        "status",
        "distance_pairs",
        "background_pairs",
        "variable_pairs",
        "estimated_peak_bytes",
        "coordinate_dtype",
        "distance_dtype",
        "background_backend",
        "client_geometry_sha256",
    ],
)
def test_receipt_requires_every_declared_field(native, field):
    bank = make_bank(native, "recent").propose(
        [record(native, i) for i in range(1, 9)], confirmed_at=20
    )
    payload = bank.snapshot()
    payload["receipt"].pop(field)
    with pytest.raises(ValueError):
        bank.restore(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"extra": "campo desconocido"},
        {"client_ids": []},
        {"client_ids": [1, 2, 3, 4, 5, 6, 7, 9]},
        {"client_ids": [1, 2, 3, 4, 5, 6, 8]},
        {"client_ids": [1, 2, 3, 4, 5, 6, 7, True]},
        {"fixed_ids": [1, 5, 6, 7]},
        {"candidate_ids": [9]},
        {"candidate_ids": [5]},
        {"candidate_ids": [1]},
        {"status": "swap_limit"},
        {"status": "empty_clients"},
        {"status": "enumerated_restricted"},
        {"status": True},
        {"before_seen": False},
        {"confirmed_at": 20.0},
        {"before_seen": 5},
        {"background_pairs": 0, "distance_pairs": 0},
        {"variable_pairs": 1, "distance_pairs": 17},
        {"distance_pairs": 17},
        {"background_pairs": True},
        {"estimated_peak_bytes": 1},
        {"estimated_peak_bytes": 64 * 1024**2 + 1},
        {"coordinate_dtype": "float64"},
        {"distance_dtype": "float32"},
        {"background_backend": "scipy_cdist_fp32_exploratory"},
        {"client_geometry_sha256": "g" * 64},
        {"client_geometry_sha256": "a" * 63},
        {"client_geometry_sha256": True},
    ],
)
def test_receipt_rejects_incoherent_schema_geometry_and_selection(native, changes):
    bank = make_bank(native, "recent").propose(
        [record(native, i) for i in range(1, 9)], confirmed_at=20
    )
    before = bank.snapshot()
    payload = copy.deepcopy(before)
    payload["receipt"].update(changes)
    with pytest.raises(ValueError):
        bank.restore(payload)
    assert bank.snapshot() == before


@pytest.mark.parametrize("status", ["fixed_only", "all_candidates", "swap_limit"])
def test_anchored_receipt_status_matches_its_variable_selection(native, status):
    bank = make_bank(native, "anchored").propose(
        [record(native, i) for i in range(1, 9)], confirmed_at=20
    )
    assert bank.receipt["status"] == "one_swap_local_restricted"
    payload = bank.snapshot()
    payload["receipt"]["status"] = status
    with pytest.raises(ValueError):
        bank.restore(payload)


def test_anchored_all_candidates_receipt_remains_recoverable(native):
    config = RetentionConfig(policy="anchored", capacity=4, frontier=2, new_candidates=2)
    bank = RetentionBank(
        native, config, codec_id="c" * 64, world="fixture", partition="train", fold="0"
    ).propose([record(native, i) for i in range(1, 7)], confirmed_at=20)
    assert bank.receipt["status"] == "all_candidates"
    restored = bank.restore(bank.snapshot())
    assert restored.snapshot() == bank.snapshot()


def test_anchored_bank_reuses_distances_and_bounds_the_background_with_the_reference_bits(
    native, monkeypatch
):
    from mars_titan.memory import retention_bank

    select = retention_bank.select_anchored_medoids
    applied = []

    def spied(*args, **kwargs):
        result = select(*args, **kwargs)
        applied.append((result.reused_distances, result.bounded_background))
        return result

    def reference(*args, **kwargs):
        return select(*args, **dict(kwargs, reuse_distances=False, bounded_background=False))

    def stream(selection):
        # Claves normales en 64 coordenadas: cada propuesta selecciona sobre 128 clientes.
        monkeypatch.setattr(retention_bank, "select_anchored_medoids", selection)
        rng = np.random.default_rng(3)
        config = RetentionConfig(policy="anchored", capacity=64, frontier=8, new_candidates=16)
        bank = RetentionBank(
            native, config, codec_id="c" * 64, world="fixture", partition="train", fold="0"
        )
        receipts = []
        for step in range(4):
            incoming = []
            for offset in range(64):
                value = record(native, 1 + 64 * step + offset)
                value.key = [float(x) for x in rng.standard_normal(64)]
                incoming.append(value)
            bank = bank.propose(incoming, confirmed_at=10_000 * (step + 1))
            receipts.append(dict(bank.receipt))
        return receipts, bank.snapshot()

    fast, fast_state = stream(spied)
    slow, slow_state = stream(reference)
    # La primera propuesta cabe entera y no tiene clientes fuera de los fijos, así que no
    # hay fondo que acotar. Las demás seleccionan entre 128 episodios con las dos opciones.
    assert applied == [(True, False)] + [(True, True)] * 3
    for step, (got, expected) in enumerate(zip(fast, slow, strict=True)):
        assert repr(got["objective"]) == repr(expected["objective"])
        # Solo cambia la memoria declarada, que suma la cota y la tabla reutilizada.
        more, less = got.pop("estimated_peak_bytes"), expected.pop("estimated_peak_bytes")
        assert more > less if step else more == less
        assert got == expected
    assert fast_state["memory"] == slow_state["memory"]
