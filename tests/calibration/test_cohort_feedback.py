"""Feedback parcial identificado con fixtures, sin ajuste de modelos ni corpus real."""

import copy
import hashlib
import json

import numpy as np
import pytest
from test_online_conformal import cohort, start

from mars_titan.calibration.cohort_feedback import CohortFeedback, CohortOutcome
from mars_titan.environments.cohorts import FINAL_TEST_START_US


def ledger(**options):
    return CohortFeedback.start(start(groups=("US", "CN")), **options)


def emitted(**options):
    return ledger(**options).emit("US", 10, ("a", "b"), cohort(2), (20, 21)).state


def label(sample, value, *, at=21, prediction_at=10, group="US"):
    return CohortOutcome(group, prediction_at, sample, at, value)


def reseal(payload):
    payload["state_sha256"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in payload.items() if k != "state_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return payload


def test_partial_feedback_and_a_single_batch_produce_the_same_complete_update():
    state = emitted()
    before = state.state_sha256()
    initial = state.corrections("US")
    first = state.feedback((label("a", 0.0, at=20),), at=20)
    assert first.settlements == ()
    assert first.state.pending_rows == 2
    assert first.state.corrections("US") == initial
    second = first.state.feedback((label("b", 40.0),), at=21)
    direct = state.feedback((label("b", 40.0), label("a", 0.0, at=20)), at=21)
    assert second.state.state_sha256() == direct.state.state_sha256()
    assert second.settlements == direct.settlements
    assert second.settlements[0]["status"] == "matured"
    assert second.state.pending_rows == 0
    assert state.state_sha256() == before


def test_joint_permutation_of_ids_quantiles_and_maturity_preserves_the_proposal():
    state = ledger()
    quantiles = cohort(3) + np.arange(3)[:, None]
    first = state.emit("US", 10, ("c", "a", "b"), quantiles, (22, 20, 21))
    order = np.array([1, 2, 0])
    second = state.emit("US", 10, ("a", "b", "c"), quantiles[order], (20, 21, 22))
    assert first.state.state_sha256() == second.state.state_sha256()
    np.testing.assert_array_equal(first.quantiles[order], second.quantiles)
    assert first.emission == second.emission
    assert state.pending_rows == 0


def test_later_complete_cohort_waits_for_the_oldest_and_keeps_the_issued_correction():
    state = emitted().emit("US", 11, ("c",), cohort(1), (22,)).state
    later = state.feedback((label("c", 40.0, at=22, prediction_at=11),), at=22)
    assert later.settlements == () and later.state.pending_rows == 3
    final = later.state.feedback((label("b", 0.0), label("a", 0.0)), at=23)
    assert [item["prediction_at"] for item in final.settlements] == [10, 11]
    assert final.state.pending_rows == 0
    assert final.settlements[1]["intervals"]["0.8"]["emitted"] == 7.0


def test_an_exclusion_cancels_the_whole_cohort_without_imputing_other_labels():
    state = emitted()
    before = state.corrections("US")
    excluded = CohortOutcome("US", 10, "a", 20, status="excluded", evidence_sha256="a" * 64)
    early = state.feedback((excluded,), at=20)
    assert early.settlements == ()
    result = early.state.feedback((), at=21)
    assert result.state.pending_rows == 0
    assert result.state.corrections("US") == before
    (record,) = result.settlements
    assert record["status"] == "cancelled"
    assert record["rows"] == 2 and record["labels"] == []
    assert record["unresolved_ids"] == ["b"]
    assert record["exclusions"][0]["evidence_sha256"] == "a" * 64
    assert result.state.export()["cancelled_cohorts"] == 1
    assert result.state.export()["calibrator"]["markets"]["US"]["matured"] == 0


def test_json_restore_keeps_partial_feedback_and_does_not_share_exported_storage():
    partial = emitted().feedback((label("b", 40.0),), at=21).state
    payload = json.loads(json.dumps(partial.export(), allow_nan=False))
    restored = CohortFeedback.restore(payload)
    assert restored.state_sha256() == partial.state_sha256()
    payload["cohorts"][0]["rows"][0]["sample_id"] = "changed"
    payload["calibrator"]["markets"]["US"]["pending"][0]["quantiles"][0][0] = 99
    assert restored.state_sha256() == partial.state_sha256()
    expected = partial.feedback((label("a", 0.0),), at=22)
    actual = restored.feedback((label("a", 0.0),), at=22)
    assert expected.state.state_sha256() == actual.state.state_sha256()
    assert expected.settlements == actual.settlements
    for state in (expected.state, actual.state):
        assert CohortFeedback.restore(state.export()).state_sha256() == state.state_sha256()


class UnreadableValue:
    def __float__(self):
        raise AssertionError("No se debe convertir una etiqueta futura")


@pytest.mark.parametrize(
    "outcome",
    [
        label("b", UnreadableValue(), at=30),
        label("b", 0.0, at=30),
        label("b", 0.0, at=20),
        label("missing", 0.0),
        label("b", np.nan),
        label("b", np.inf),
        label("b", True),
        label("b", 10**400),
        label("b", 0.0, prediction_at=9),
        CohortOutcome("US", 10, "b", 21, status="excluded", evidence_sha256="bad"),
        CohortOutcome("US", 10, "b", 21, 0.0, status="excluded", evidence_sha256="a" * 64),
        CohortOutcome("US", 10, "b", 21, status="unknown"),
    ],
)
def test_invalid_feedback_rejects_the_entire_batch_and_keeps_the_source(outcome):
    state = emitted()
    before = state.state_sha256()
    with pytest.raises(ValueError):
        state.feedback((label("a", 0.0, at=20), outcome), at=21)
    assert state.state_sha256() == before


def test_duplicate_feedback_is_rejected_both_within_and_between_proposals():
    state = emitted()
    item = label("a", 0.0)
    with pytest.raises(ValueError):
        state.feedback((item, item), at=21)
    partial = state.feedback((item,), at=21).state
    before = partial.state_sha256()
    with pytest.raises(ValueError):
        partial.feedback((item,), at=22)
    assert partial.state_sha256() == before


@pytest.mark.parametrize(
    "ids,quantiles,due",
    [
        (("a", "a"), cohort(2), (20, 21)),
        (("a", "b"), cohort(1), (20, 21)),
        (("a", "b"), cohort(2), (10, 21)),
        (("a", "b"), cohort(2), (20,)),
        (("a", ""), cohort(2), (20, 21)),
        (("a", "b"), cohort(2), (20, FINAL_TEST_START_US)),
    ],
)
def test_invalid_emission_does_not_change_the_snapshot(ids, quantiles, due):
    state = ledger()
    before = state.state_sha256()
    with pytest.raises(ValueError):
        state.emit("US", 10, ids, quantiles, due)
    assert state.state_sha256() == before


def test_clock_and_market_isolation_and_explicit_capacity_limits():
    state = emitted(max_rows=2)
    before = state.state_sha256()
    for action in (
        lambda: state.emit("CN", 11, ("c",), cohort(1), (22,)),
        lambda: state.feedback((), at=9),
        lambda: state.feedback((), at=FINAL_TEST_START_US),
    ):
        with pytest.raises(ValueError):
            action()
        assert state.state_sha256() == before
    completed = state.feedback((label("a", 40.0), label("b", 40.0)), at=21).state
    assert completed.corrections("US") != state.corrections("US")
    assert completed.corrections("CN") == state.corrections("CN")
    assert completed.emit("CN", 21, ("c",), cohort(1), (30,)).state.pending_rows == 1
    small = ledger(max_bytes=4096)
    with pytest.raises(ValueError):
        small.emit("US", 10, tuple(str(i).zfill(128) for i in range(32)), cohort(32), (20,) * 32)
    assert small.pending_rows == 0


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("clock", 0), ("policy", "shadow")])
def test_restore_rejects_incompatible_semantics_even_with_a_recomputed_hash(field, value):
    payload = copy.deepcopy(emitted().export())
    payload[field] = value
    with pytest.raises(ValueError):
        CohortFeedback.restore(reseal(payload))


def test_restore_rejects_mutation_and_duplicate_ids_with_a_valid_outer_hash():
    payload = emitted().export()
    damaged = copy.deepcopy(payload)
    damaged["cohorts"][0]["rows"][0]["sample_id"] = "aa"
    with pytest.raises(ValueError):
        CohortFeedback.restore(damaged)
    damaged = copy.deepcopy(payload)
    damaged["cohorts"][0]["rows"][1]["sample_id"] = "a"
    with pytest.raises(ValueError):
        CohortFeedback.restore(reseal(damaged))


def test_restore_checks_the_row_budget_before_serializing_rejected_quantiles(monkeypatch):
    from mars_titan.calibration import cohort_feedback_state

    payload = emitted(max_rows=2).export()
    payload["calibrator"]["markets"]["US"]["pending"][0]["quantiles"] *= 3

    def forbidden_serialization(_):
        raise AssertionError("La matriz rechazada no se debe serializar")

    monkeypatch.setattr(cohort_feedback_state, "_canonical", forbidden_serialization)
    with pytest.raises(ValueError, match="filas"):
        CohortFeedback.restore(payload)


@pytest.mark.parametrize("corruption", ["missing_intervals", "non_numeric_correction"])
def test_malformed_calibrator_state_has_a_domain_error(corruption):
    payload = emitted().export()
    market = payload["calibrator"]["markets"]["US"]
    if corruption == "missing_intervals":
        market["intervals"] = None
    else:
        market["intervals"]["0.8"]["correction"] = "invalid"
    with pytest.raises(ValueError):
        CohortFeedback.restore(reseal(payload))


@pytest.mark.parametrize(
    "max_rows,max_bytes",
    [(0, 8192), (True, 8192), (32769, 8192), (2, 1023), (2, True), (2, 16 * 1024**2 + 1)],
)
def test_invalid_budgets_are_rejected_before_a_snapshot_is_created(max_rows, max_bytes):
    with pytest.raises(ValueError):
        ledger(max_rows=max_rows, max_bytes=max_bytes)


def test_start_rejects_pending_calibration_without_its_row_identities():
    model = start()
    model.emit("US", 10, cohort(2), 20)
    before = model.state_sha256()
    with pytest.raises(ValueError):
        CohortFeedback.start(model)
    assert model.state_sha256() == before


def test_input_and_output_arrays_cannot_change_a_proposed_snapshot():
    state = ledger()
    values = cohort(2)
    proposal = state.emit("US", 10, ("a", "b"), values, (20, 21))
    digest = proposal.state.state_sha256()
    values[:] = 1000
    proposal.quantiles[:] = 2000
    proposal.emission["corrections"]["0.8"] = 3000
    assert proposal.state.state_sha256() == digest


def test_same_time_markets_and_feedback_permutations_produce_the_same_snapshot():
    state = ledger()
    left = state.emit("US", 10, ("a",), cohort(1), (20,)).state
    left = left.emit("CN", 10, ("b",), cohort(1), (20,)).state
    right = state.emit("CN", 10, ("b",), cohort(1), (20,)).state
    right = right.emit("US", 10, ("a",), cohort(1), (20,)).state
    assert left.state_sha256() == right.state_sha256()
    outcomes = (label("a", 0.0, at=20), label("b", 40.0, at=20, group="CN"))
    first, second = left.feedback(outcomes, at=20), right.feedback(outcomes[::-1], at=20)
    assert first.state.state_sha256() == second.state.state_sha256()
    assert first.settlements == second.settlements


def test_finalization_preserves_valid_received_labels_and_counts_no_imputed_loss():
    state = emitted().feedback((label("a", 0.0),), at=21).state
    final = CohortOutcome("US", 10, "b", 22, status="finalized", evidence_sha256="f" * 64)
    result = state.feedback((final,), at=22)
    (record,) = result.settlements
    assert record["status"] == "cancelled"
    assert [item["sample_id"] for item in record["labels"]] == ["a"]
    assert [item["status"] for item in record["exclusions"]] == ["finalized"]
    assert record["unresolved_ids"] == []
    assert result.state.corrections("US") == state.corrections("US")
