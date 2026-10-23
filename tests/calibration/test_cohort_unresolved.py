"""Claves de feedback que PT2 aún espera, sin confundirlas con toda la cola financiera."""

from test_cohort_feedback import emitted, label, ledger
from test_online_conformal import cohort

from mars_titan.calibration.cohort_feedback import CohortFeedback, CohortOutcome


def test_partial_feedback_keys_exclude_received_labels_without_dropping_the_cohort():
    state = emitted()
    original = state.state_sha256()
    keys = state.unresolved_keys()
    partial = state.feedback((label("a", 2.0, at=20),), at=20).state
    assert keys == frozenset({("US", 10, "a"), ("US", 10, "b")})
    assert partial.pending_rows == 2
    assert partial.unresolved_keys() == frozenset({("US", 10, "b")})
    assert state.state_sha256() == original
    assert keys == state.unresolved_keys()


def test_keys_distinguish_the_same_sample_name_in_other_markets_and_decisions():
    state = ledger()
    for group, at in (("CN", 10), ("US", 10), ("US", 11)):
        state = state.emit(group, at, ("a",), cohort(1), (20,)).state
    assert state.unresolved_keys() == frozenset({("CN", 10, "a"), ("US", 10, "a"), ("US", 11, "a")})
    restored = CohortFeedback.restore(state.export())
    assert restored.unresolved_keys() == state.unresolved_keys()


def test_exclusion_and_cancellation_remove_only_the_keys_the_calibrator_no_longer_waits_for():
    exclusion = CohortOutcome("US", 10, "a", 20, status="excluded", evidence_sha256="a" * 64)
    partial = emitted().feedback((exclusion,), at=20).state
    assert partial.unresolved_keys() == frozenset({("US", 10, "b")})
    cancelled = partial.feedback((), at=21).state
    assert cancelled.unresolved_keys() == frozenset()
    assert cancelled.pending_rows == 0
    assert ledger().unresolved_keys() == frozenset()


def test_key_query_does_not_export_or_materialize_numeric_calibration_state(monkeypatch):
    state = emitted()

    def forbidden():
        raise AssertionError("Consultar claves no debe exportar cuantiles ni etiquetas")

    monkeypatch.setattr(state, "export", forbidden)
    assert state.unresolved_keys() == frozenset({("US", 10, "a"), ("US", 10, "b")})
