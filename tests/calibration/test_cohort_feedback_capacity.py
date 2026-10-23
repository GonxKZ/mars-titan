"""La admisión de PT2 usa el snapshot previo, sin adelantar feedback del mismo evento."""

import pytest
from test_cohort_feedback import emitted, label, ledger
from test_online_conformal import cohort, start

from mars_titan.calibration.cohort_feedback import CohortCapacityError, CohortFeedback


def test_capacity_query_does_not_apply_the_feedback_that_would_free_the_same_slot():
    state = CohortFeedback.start(start(max_pending=1), max_rows=2)
    state = state.emit("US", 10, ("a", "b"), cohort(2), (20, 20)).state
    before = state.state_sha256()
    assert state.can_emit("US", rows=1) is False
    with pytest.raises(CohortCapacityError):
        state.emit("US", 20, ("c",), cohort(1), (30,))
    assert state.state_sha256() == before
    following = state.feedback((label("a", 0.0, at=20), label("b", 1.0, at=20)), at=20).state
    assert following.can_emit("US", rows=1) is True
    assert following.emit("US", 20, ("c",), cohort(1), (30,)).state.pending_rows == 1


def test_serialized_byte_exhaustion_has_a_distinct_error_and_keeps_the_source():
    state = ledger(max_bytes=4096)
    before = state.state_sha256()
    with pytest.raises(CohortCapacityError):
        state.emit("US", 10, tuple(str(i).zfill(128) for i in range(32)), cohort(32), (20,) * 32)
    assert state.state_sha256() == before


def test_bad_inputs_and_unknown_markets_remain_contract_errors():
    state = emitted(max_rows=2)
    for group, rows in (("EU", 1), ("US", 0), ("US", True), ("US", 32769)):
        with pytest.raises(ValueError) as error:
            state.can_emit(group, rows=rows)
        assert not isinstance(error.value, CohortCapacityError)
