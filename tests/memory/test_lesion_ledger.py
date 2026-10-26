"""Predicciones diagnósticas emitidas y resultados maduros, sin reconstrucción posterior."""

import copy

import pytest
import torch

from mars_titan.memory.area_memory import AreaLayout
from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.lesion_protocol import LesionProtocol

PLAN = LesionProtocol(seeds=(0,), fractions=((1, 2),)).bind(
    MatureCorrection(AssociativeMemoryConfig("proximal"), areas=AreaLayout())
)


def ledger(**options):
    from mars_titan.memory.lesion_ledger import LesionLedger

    return LesionLedger(PLAN, **options)


def issued():
    value = ledger()
    value.begin(10)
    predictions = torch.tensor([[1.0] * 5, [3.0] * 5, [10.0] * 5], dtype=torch.float64)
    value.issue([("US/A", 10), ("US/B", 10), ("CN/C", 10)], predictions)
    predictions.zero_()
    return value


def test_scoring_uses_emitted_values_and_session_means_before_aggregation():
    value = issued()
    value.begin(20)
    resolved = value.receive(
        [
            ("US/B", 10, 20, 0.0),
            ("CN/C", 10, 20, 0.0),
            ("US/A", 10, 20, 0.0),
        ]
    )
    metrics = value.metrics()
    assert metrics["cases"]["intact"]["session_mae"] == 6.0
    assert metrics["cases"]["intact"]["samples"] == 3
    assert metrics["pending"] == 0 and metrics["resolved"] == 3
    assert resolved["US/A", 10].tolist() == [1.0] * 5
    assert value.event()["resolved"]["predictions"].shape == (3, 5)


def test_future_feedback_and_bad_values_leave_the_ledger_unchanged():
    value = issued()
    value.begin(20)
    before = value.export()["state_sha256"]
    for records in [
        [("US/A", 10, 21, 0.0)],
        [("US/A", 10, 20, float("nan"))],
        [("US/A", 10, 20, 1e308)],
        [("US/A", 10, 10, 0.0)],
        [("US/A", 10, 20, 0.0)] * 2,
    ]:
        with pytest.raises(ValueError):
            value.receive(records)
        assert value.export()["state_sha256"] == before


def test_restore_reproduces_next_feedback_and_canonical_delivery_order():
    from mars_titan.memory.lesion_ledger import LesionLedger

    left = issued()
    right = LesionLedger.restore(PLAN, left.export())
    a = [("US/A", 10, 20, 0.1), ("US/B", 10, 20, -0.1), ("CN/C", 10, 20, 0.2)]
    left.begin(20)
    right.begin(20)
    left.receive(a)
    right.receive(a[::-1])
    assert left.export()["state_sha256"] == right.export()["state_sha256"]
    assert left.metrics() == right.metrics()


def test_censoring_removes_pending_without_inventing_targets_or_scores():
    value = issued()
    value.begin(20)
    value.discard([("US/A", 10)], reason="excluded")
    value.discard([("US/B", 10)], reason="finalized")
    metrics = value.metrics()
    assert metrics["censored"] == 2 and metrics["pending"] == 1
    assert metrics["cases"]["intact"]["session_mae"] is None
    assert len(value.event()["censored"]) == 2


def test_restore_after_mature_feedback_and_censoring_preserves_the_next_result():
    from mars_titan.memory.lesion_ledger import LesionLedger

    value = issued()
    value.begin(20)
    value.receive([("US/A", 10, 20, 0.3)])
    value.discard([("US/B", 10)], reason="excluded")
    restored = LesionLedger.restore(PLAN, value.export())
    assert restored.export()["state_sha256"] == value.export()["state_sha256"]
    for instance in (value, restored):
        instance.begin(30)
        instance.receive([("CN/C", 10, 30, 0.7)])
    assert restored.export()["state_sha256"] == value.export()["state_sha256"]
    assert restored.metrics() == value.metrics()


@pytest.mark.parametrize(
    "corruption",
    [
        "session_market",
        "session_future",
        "boolean_count",
        "wrong_dtype",
        "negative_error",
        "nonfinite_error",
        "count_mismatch",
        "duplicate_pending",
        "future_pending",
        "future_result",
        "invalid_target_type",
        "invalid_censor_reason",
    ],
)
def test_self_consistent_hash_does_not_bypass_semantic_state_validation(corruption):
    from mars_titan.memory.lesion_ledger import LesionLedger, _hash

    value = issued()
    value.begin(20)
    value.receive([("US/A", 10, 20, 0.3)])
    value.discard([("US/B", 10)], reason="excluded")
    payload = value.export()
    sessions, event, pending = payload["sessions"], payload["event"], payload["pending"]
    if corruption == "session_market":
        sessions["keys"][0][0] = "XX"
    elif corruption == "session_future":
        sessions["keys"][0][1] = 20
    elif corruption == "boolean_count":
        sessions["counts"][0] = True
    elif corruption == "wrong_dtype":
        sessions["errors"] = sessions["errors"].float()
    elif corruption == "negative_error":
        sessions["errors"][0, 0, 0] = -1.0
    elif corruption == "nonfinite_error":
        sessions["errors"][0, 0, 0] = float("nan")
    elif corruption == "count_mismatch":
        sessions["counts"][0] = 2
    elif corruption == "duplicate_pending":
        pending["keys"].append(pending["keys"][0])
    elif corruption == "future_pending":
        pending["keys"][0][1] = 21
    elif corruption == "future_result":
        event["resolved"]["available_at"][0] = 21
    elif corruption == "invalid_target_type":
        event["resolved"]["target"][0] = "nan"
    else:
        event["censored"][0][2] = "forgotten"
    # El checksum se vuelve a calcular para exigir la validación del significado del estado.
    payload["state_sha256"] = _hash(
        {key: item for key, item in payload.items() if key != "state_sha256"}
    )
    with pytest.raises(ValueError) as error:
        LesionLedger.restore(PLAN, payload)
    assert "huella" not in str(error.value)


def test_pending_and_session_budgets_fail_before_changing_state():
    value = ledger(max_pending=2, max_sessions=1)
    value.begin(10)
    before = value.export()["state_sha256"]
    with pytest.raises(ValueError):
        value.issue(
            [("US/A", 10), ("US/B", 10), ("CN/C", 10)], torch.zeros(3, 5, dtype=torch.float64)
        )
    assert value.export()["state_sha256"] == before
    value.issue([("US/A", 10)], torch.ones(1, 5, dtype=torch.float64))
    value.begin(20)
    value.receive([("US/A", 10, 20, 0.0)])
    value.issue([("US/A", 20)], torch.ones(1, 5, dtype=torch.float64))
    value.begin(30)
    before = value.export()["state_sha256"]
    with pytest.raises(ValueError):
        value.receive([("US/A", 20, 30, 0.0)])
    assert value.export()["state_sha256"] == before


@pytest.mark.parametrize("field", ["pending", "resolved", "clock", "configuration"])
def test_restore_rejects_modified_state(field):
    from mars_titan.memory.lesion_ledger import LesionLedger

    payload = copy.deepcopy(issued().export())
    payload[field] = None
    with pytest.raises(ValueError):
        LesionLedger.restore(PLAN, payload)


def test_mutating_exported_predictions_does_not_change_the_live_ledger():
    value = issued()
    before = value.export()["state_sha256"]
    payload = value.export()
    payload["pending"]["predictions"].zero_()
    assert value.export()["state_sha256"] == before


def test_restore_rejects_numerically_changed_predictions_with_the_old_hash():
    from mars_titan.memory.lesion_ledger import LesionLedger

    payload = issued().export()
    payload["pending"]["predictions"][0, 0] += 0.5
    with pytest.raises(ValueError, match="huella"):
        LesionLedger.restore(PLAN, payload)


def test_accumulation_overflow_in_one_case_does_not_commit_the_other_cases():
    value = ledger()
    value.begin(10)
    value.issue(
        [("US/A", 10), ("US/B", 10)],
        torch.tensor([[0.0, 1e154, 0.0, 0.0, 0.0]] * 2, dtype=torch.float64),
    )
    value.begin(20)
    value.receive([("US/A", 10, 20, 0.0)])
    before = value.export()["state_sha256"]
    with pytest.raises(ValueError, match="acumulación"):
        value.receive([("US/B", 10, 20, 0.0)])
    assert value.export()["state_sha256"] == before


@pytest.mark.parametrize(
    "changes", [dict(max_pending=0), dict(max_pending=True), dict(max_sessions=0)]
)
def test_invalid_capacity_is_rejected(changes):
    with pytest.raises(ValueError):
        ledger(**changes)


@pytest.mark.parametrize(
    "rows",
    [[("US/A", 9)], [("US/A", 11)], [("XX/A", 10)], [("US/", 10)], [("US/A", 10)] * 2],
)
def test_invalid_or_noncontemporaneous_emissions_do_not_change_state(rows):
    value = ledger()
    value.begin(10)
    before = value.export()["state_sha256"]
    with pytest.raises(ValueError):
        value.issue(rows, torch.zeros(len(rows), 5, dtype=torch.float64))
    assert value.export()["state_sha256"] == before


def test_empty_ledger_restores_and_backward_time_is_rejected_atomically():
    from mars_titan.memory.lesion_ledger import LesionLedger

    value = ledger()
    assert (
        LesionLedger.restore(PLAN, value.export()).export()["state_sha256"]
        == value.export()["state_sha256"]
    )
    value.begin(20)
    before = value.export()["state_sha256"]
    with pytest.raises(ValueError):
        value.begin(19)
    assert value.export()["state_sha256"] == before
