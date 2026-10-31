"""Libro S2 con emisiones y resultados artificiales, sin ajuste de parámetros."""

import importlib

import numpy as np
import pytest

from tests.memory.test_failure_context import AT, context, layout


def api():
    try:
        return importlib.import_module("mars_titan.memory.failure_ledger")
    except ModuleNotFoundError:
        pytest.fail("Falta el libro de fallos maduros S2")


def configuration(**changes):
    values = dict(
        threshold=0.5,
        threshold_manifest_id="e" * 64,
        threshold_fitted_until=AT - 100,
        source_manifest_id="f" * 64,
        source_fitted_until=AT - 100,
        capacity=4,
        max_pending=8,
        max_flows=8,
        max_bytes=32768,
        bandwidth=0.25,
        strength=0.5,
        block_size=2,
    )
    return api().FailureLedgerConfig(**(values | changes))


def ledger(**changes):
    return api().FailureLedger(layout(), configuration(**changes))


def row(asset="AAA", at=AT, key=(1.0, 0.0, 0.0), action="fixed_consumer/no_tool"):
    return context(
        flow_id=f"US/{asset}",
        sample_id=f"fixture/{asset}/{at}",
        prediction_at=at,
        input_available_at=at - 1,
        key_inputs=np.array(key, dtype=np.float64),
        action=action,
    )


def mature(book, records, targets, *, at=AT + 10):
    book.begin(at)
    book.resolve(
        [(r.flow_id, r.prediction_at, at, y) for r, y in zip(records, targets, strict=True)]
    )


def test_pending_labels_never_enter_the_current_snapshot_even_after_resolution():
    book, original = ledger(), row()
    book.begin(AT)
    assert book.query(original).confidence == 1
    book.issue([(original, 0.0)])
    book.begin(AT + 10)
    before = book.query(row(at=AT + 10))
    book.resolve([("US/AAA", AT, AT + 10, 1.0)])
    assert book.query(row(at=AT + 10)) == before
    book.begin(AT + 10)
    assert book.query(row(at=AT + 10)) == before
    book.begin(AT + 11)
    score = book.query(row(at=AT + 11))
    assert score.risk == 0.5 and score.confidence == 0.75
    assert score.context_id == original.context_id and score.eligible == 1
    assert book.counts() == dict(issued=1, resolved=1, censored=0, admitted=1, pending=0)


def test_error_uses_the_actual_issued_prediction_and_threshold_is_strict():
    book, a, b = ledger(), row(), row("BBB")
    book.begin(AT)
    book.issue([(a, 1.0), (b, 0.0)])
    mature(book, [a, b], [1.5, 0.75])
    book.begin(AT + 11)
    score = book.query(row(at=AT + 11))
    assert score.eligible == 1 and score.context_id == b.context_id
    assert score.risk == pytest.approx(1 - 0.5 / 0.75)


def test_a_late_backdated_result_cannot_rewrite_the_snapshot():
    book, original = ledger(), row()
    book.begin(AT)
    book.issue([(original, 0.0)])
    book.begin(AT + 100)
    book.resolve([("US/AAA", AT, AT + 1, 1.0)])
    assert book.query(row(at=AT + 100)).eligible == 0
    book.begin(AT + 101)
    assert book.query(row(at=AT + 101)).eligible == 1


@pytest.mark.parametrize("bad", [AT - 1, True, 1_704_067_200_000_000])
def test_clock_rejects_reverse_invalid_and_final_test_cuts(bad):
    book = ledger()
    book.begin(AT)
    with pytest.raises(ValueError):
        book.begin(bad)
    assert book.clock == AT


def test_failed_batches_leave_pending_counters_and_flow_cursors_unchanged():
    book, original = ledger(), row()
    book.begin(AT)
    with pytest.raises(ValueError):
        book.issue([(original, 0.0), (original, 0.0)])
    assert book.counts()["issued"] == 0
    book.issue([(original, 0.0)])
    book.begin(AT + 10)
    with pytest.raises(ValueError):
        book.resolve([("US/AAA", AT, AT + 1, 1.0), ("US/unknown", AT, AT + 1, 1.0)])
    assert book.counts()["pending"] == 1 and book.counts()["resolved"] == 0
    with pytest.raises(ValueError):
        book.resolve([("US/AAA", AT, AT + 11, 1.0)])
    assert book.counts()["pending"] == 1


def test_retention_and_query_are_invariant_to_asset_order():
    books = [ledger(capacity=1), ledger(capacity=1)]
    records = [row(), row("BBB")]
    for book, ordered in zip(books, (records, records[::-1]), strict=True):
        book.begin(AT)
        book.issue([(r, 0.0) for r in ordered])
        mature(book, ordered, [1.0, 1.0])
        book.begin(AT + 11)
    scores = [b.query(row(at=AT + 11)) for b in books]
    assert scores[0] == scores[1]
    assert scores[0].context_id == records[1].context_id


def test_queries_use_only_the_same_market_and_mechanism():
    book, original = ledger(), row()
    book.begin(AT)
    book.issue([(original, 0.0)])
    mature(book, [original], [1.0])
    book.begin(AT + 11)
    assert book.query(row(at=AT + 11, action="other_tool")).eligible == 0
    other = context(flow_id="CN/AAA", prediction_at=AT + 11)
    assert book.query(other).eligible == 0


def test_shuffling_preserves_records_and_severity_but_can_change_the_signal():
    records = [row(), row("BBB", key=(0.0, 1.0, 0.0))]
    values = []
    for geometry in ("associated", "shuffled"):
        book = ledger(geometry=geometry, seed=0)
        book.begin(AT)
        book.issue([(r, 0.0) for r in records])
        mature(book, records, [1.0, 4.0])
        book.begin(AT + 11)
        values.append(book.query(row(at=AT + 11)))
    assert values[0].eligible == values[1].eligible == 2
    assert values[0].risk == 0.5
    assert values[1].risk == 0.875
    assert values[0].failure_context_id == values[0].context_id == records[0].context_id
    assert values[1].context_id == records[0].context_id
    assert values[1].failure_context_id == records[1].context_id


def test_capacity_and_byte_limits_fail_before_committing_emissions():
    book = ledger(max_pending=1)
    book.begin(AT)
    with pytest.raises(ValueError):
        book.issue([(row(), 0.0), (row("BBB"), 0.0)])
    assert book.counts()["issued"] == 0
    tiny = ledger(max_bytes=128)
    tiny.begin(AT)
    with pytest.raises(ValueError, match="bytes"):
        tiny.issue([(row(), 0.0)])
    assert tiny.counts()["issued"] == 0


def test_threshold_and_source_must_precede_every_decision():
    for field in ("threshold_fitted_until", "source_fitted_until"):
        book = ledger(**{field: AT})
        with pytest.raises(ValueError, match="pasado"):
            book.begin(AT)


def test_numeric_aliases_cannot_replace_integer_decision_identities():
    book = ledger()
    book.begin(AT)
    book.issue([(row(), 0.0)])
    book.begin(AT + 10)
    with pytest.raises(ValueError):
        book.resolve([("US/AAA", float(AT), AT + 1, 1.0)])
    with pytest.raises(ValueError):
        book.censor([("US/AAA", float(AT))])
    assert book.counts()["pending"] == 1


def test_censoring_and_duplicate_results_do_not_reuse_an_emission():
    book = ledger()
    book.begin(AT)
    book.issue([(row(), 0.0)])
    book.censor([("US/AAA", AT)])
    assert book.counts() == dict(issued=1, resolved=0, censored=1, admitted=0, pending=0)
    with pytest.raises(ValueError):
        book.issue([(row(), 0.0)])
    book.begin(AT + 10)
    with pytest.raises(ValueError):
        book.resolve([("US/AAA", AT, AT + 1, 1.0)])


def test_nonfinite_emissions_and_overflowing_errors_are_rejected_atomically():
    book = ledger()
    book.begin(AT)
    with pytest.raises(ValueError):
        book.issue([(row(), float("nan"))])
    assert book.counts()["issued"] == 0
    book.issue([(row(), -1e308)])
    book.begin(AT + 10)
    with pytest.raises(ValueError):
        book.resolve([("US/AAA", AT, AT + 1, 1e308)])
    assert book.counts()["pending"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        dict(threshold=0),
        dict(threshold=float("nan")),
        dict(strength=-0.1),
        dict(strength=1.1),
        dict(bandwidth=1e-300),
        dict(bandwidth=float("inf")),
        dict(capacity=0),
        dict(max_pending=True),
        dict(max_flows=0),
        dict(max_bytes=127),
        dict(block_size=0),
        dict(seed=-1),
        dict(geometry="certified"),
        dict(threshold_manifest_id="unknown"),
    ],
)
def test_configuration_rejects_invalid_bounds_and_unidentified_provenance(changes):
    with pytest.raises(ValueError):
        configuration(**changes)


def test_disabled_strength_preserves_confidence_and_query_does_not_advance_global_rng():
    book, original = ledger(strength=0, geometry="shuffled"), row()
    second = row("BBB", key=(0.0, 1.0, 0.0))
    book.begin(AT)
    book.issue([(original, 0.0), (second, 0.0)])
    mature(book, [original, second], [1.0, 1.0])
    book.begin(AT + 11)
    state = np.random.get_state()
    score = book.query(row(at=AT + 11))
    restored = np.random.get_state()
    assert score.risk == 0.5 and score.confidence == 1
    assert state[0] == restored[0] and state[2:] == restored[2:]
    np.testing.assert_array_equal(state[1], restored[1])


def test_block_size_preserves_the_score_and_the_matched_failure():
    results = []
    records = [row(), row("BBB", key=(0.0, 1.0, 0.0)), row("CCC")]
    for block_size in (1, 2, 3):
        book = ledger(block_size=block_size, geometry="shuffled")
        book.begin(AT)
        book.issue([(r, 0.0) for r in records])
        mature(book, records, [1.0, 4.0, 2.0])
        book.begin(AT + 11)
        results.append(book.query(row(at=AT + 11)))
    assert results[0] == results[1] == results[2]


def test_recovery_preserves_the_snapshot_and_the_next_mature_update():
    book, original = ledger(), row()
    book.begin(AT)
    book.issue([(original, 0.0)])
    mature(book, [original], [1.0])
    # Un resultado maduro aún no debe aparecer en la instantánea del mismo corte.
    saved = book.export()
    restored = api().FailureLedger.restore(layout(), configuration(), saved)
    assert restored.export() == saved
    assert restored.query(row(at=AT + 10)).eligible == 0
    for value in (book, restored):
        value.begin(AT + 11)
        value.issue([(row(at=AT + 11), 0.25)])
        value.begin(AT + 20)
        value.resolve([("US/AAA", AT + 11, AT + 15, 0.9)])
    assert restored.export() == book.export()
    assert restored.query(row(at=AT + 20)) == book.query(row(at=AT + 20))


def test_recovery_keeps_evicted_records_referenced_by_the_current_snapshot():
    book, a, b = ledger(capacity=1), row(), row("BBB", at=AT + 11)
    book.begin(AT)
    book.issue([(a, 0.0)])
    mature(book, [a], [1.0])
    book.begin(AT + 11)
    book.issue([(b, 0.0)])
    book.begin(AT + 20)
    book.resolve([("US/BBB", AT + 11, AT + 15, 4.0)])
    restored = api().FailureLedger.restore(layout(), configuration(capacity=1), book.export())
    assert restored.query(row(at=AT + 20)).context_id == a.context_id
    restored.begin(AT + 21)
    assert restored.query(row(at=AT + 21)).context_id == b.context_id


def test_an_empty_ledger_roundtrips_and_stored_context_buffers_are_not_duplicated():
    book = ledger()
    assert (
        api().FailureLedger.restore(layout(), configuration(), book.export()).export()
        == book.export()
    )
    book.begin(AT)
    book.issue([(row(), 0.0)])
    mature(book, [row()], [1.0])
    book.begin(AT + 11)
    saved = book.export()
    assert len(saved["emissions"]) == len(saved["failures"]) == 1
    assert saved["retained"] == saved["snapshot"]


@pytest.mark.parametrize("field", ["counts", "clock", "layout_id", "configuration_id"])
def test_recovery_rejects_modified_state_or_configuration(field):
    book = ledger()
    book.begin(AT)
    saved = book.export()
    saved[field] = None
    with pytest.raises(ValueError):
        api().FailureLedger.restore(layout(), configuration(), saved)


def test_recovery_checks_counters_and_publication_semantics_even_with_a_recomputed_digest():
    book = ledger()
    book.begin(AT)
    book.issue([(row(), 0.0)])
    mature(book, [row()], [1.0])
    module = importlib.import_module("mars_titan.memory.failure_ledger_state")
    for edit in ("counts", "snapshot", "available_at"):
        saved = book.export()
        if edit == "counts":
            saved["counts"]["issued"] += 1
        elif edit == "snapshot":
            saved["snapshot"] = saved["retained"][:]
        else:
            saved["failures"][0]["available_at"] = AT
        saved["state_sha256"] = module._digest(saved)
        with pytest.raises(ValueError):
            api().FailureLedger.restore(layout(), configuration(), saved)


def test_recovery_rejects_two_contexts_of_the_same_decision_even_when_individually_valid():
    book = ledger()
    book.begin(AT)
    book.issue([(row(), 0.0)])
    mature(book, [row()], [1.0])
    book.begin(AT + 11)
    saved = book.export()
    alternative = context(key_inputs=np.array([0.0, 0.0, 1.0]))
    # Dos contextos válidos pueden describir una misma identidad de decisión.
    saved["emissions"].append(dict(context=alternative.export(), prediction=0.0))
    saved["emissions"].sort(key=lambda x: x["context"]["context_id"])
    duplicate = dict(saved["failures"][0], context_id=alternative.context_id)
    saved["failures"].append(duplicate)
    saved["failures"].sort(key=lambda x: x["context_id"])
    saved["retained"] = [item["context_id"] for item in saved["failures"]]
    saved["snapshot"] = saved["retained"][:]
    saved["counts"].update(issued=2, resolved=2, admitted=2)
    module = importlib.import_module("mars_titan.memory.failure_ledger_state")
    saved["state_sha256"] = module._digest(saved)
    with pytest.raises(ValueError):
        api().FailureLedger.restore(layout(), configuration(), saved)


def test_recovery_counts_all_distinct_failures_including_evicted_snapshot_entries():
    book = ledger(capacity=1)
    book.begin(AT)
    book.issue([(row(), 0.0)])
    mature(book, [row()], [1.0])
    book.begin(AT + 11)
    book.issue([(row("BBB", at=AT + 11), 0.0)])
    book.begin(AT + 20)
    book.resolve([("US/BBB", AT + 11, AT + 15, 2.0)])
    saved = book.export()
    assert len(saved["failures"]) == 2
    saved["counts"]["admitted"] = 1
    module = importlib.import_module("mars_titan.memory.failure_ledger_state")
    saved["state_sha256"] = module._digest(saved)
    with pytest.raises(ValueError):
        api().FailureLedger.restore(layout(), configuration(capacity=1), saved)


def test_an_unstarted_ledger_cannot_claim_completed_events():
    saved = ledger().export()
    saved["counts"].update(issued=1, censored=1)
    module = importlib.import_module("mars_titan.memory.failure_ledger_state")
    saved["state_sha256"] = module._digest(saved)
    with pytest.raises(ValueError):
        api().FailureLedger.restore(layout(), configuration(), saved)


def test_entry_points_require_a_cutoff_matching_context_and_bounded_batches():
    book = ledger(max_flows=1)
    for operation in (lambda: book.query(row()), lambda: book.issue([])):
        with pytest.raises(ValueError):
            operation()
    book.begin(AT)
    with pytest.raises(ValueError):
        book.query(row(at=AT + 1))
    with pytest.raises(ValueError):
        book.issue(None)
    with pytest.raises(ValueError):
        book.issue([(row(), 0.0), (row("BBB"), 0.0)])
    assert not book.counts()["issued"]


def test_persistence_roundtrips_through_the_existing_safe_tensor_archive():
    import io

    import torch

    book = ledger()
    book.begin(AT)
    book.issue([(row(), 0.0)])
    buffer = io.BytesIO()
    torch.save(book.export(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location="cpu", weights_only=True)
    assert api().FailureLedger.restore(layout(), configuration(), saved).export() == book.export()


def test_gaussian_distance_is_measured_between_normalized_keys():
    book, original = ledger(), row()
    book.begin(AT)
    book.issue([(original, 0.0)])
    mature(book, [original], [1.0])
    book.begin(AT + 11)
    orthogonal = book.query(row(at=AT + 11, key=(0.0, 100.0, 0.0)))
    assert orthogonal.risk == pytest.approx(0.5 * np.exp(-2 / (2 * 0.25**2)), rel=1e-15)
    collinear = book.query(row(at=AT + 11, key=(100.0, 0.0, 0.0)))
    assert collinear.risk == 0.5


def test_pending_capacity_also_bounds_several_emission_batches_of_the_same_cutoff():
    book = ledger(max_pending=1)
    book.begin(AT)
    book.issue([(row(), 0.0)])
    with pytest.raises(ValueError):
        book.issue([(row("BBB"), 0.0)])
    assert book.counts()["issued"] == book.counts()["pending"] == 1


def test_the_ledger_rejects_untyped_contracts_and_unknown_censoring():
    with pytest.raises(ValueError):
        api().FailureLedger(None, configuration())
    book = ledger()
    book.begin(AT)
    with pytest.raises(ValueError):
        book.censor([("US/unknown", AT)])


@pytest.mark.parametrize(
    "edit",
    [
        "missing_list",
        "malformed_cursor",
        "future_cursor",
        "duplicate_cursor",
        "emission_schema",
        "emission_order",
        "failure_schema",
        "duplicate_failure",
        "unknown_failure",
        "below_threshold",
        "failure_order",
        "duplicate_pending",
        "unknown_pending",
        "state_seal",
    ],
)
def test_malformed_archives_are_rejected_without_modifying_the_source_book(edit):
    book = ledger()
    records = [row(), row("BBB"), row("CCC")]
    book.begin(AT)
    book.issue([(r, 0.0) for r in records])
    mature(book, records[:2], [1.0, 2.0])
    book.begin(AT + 11)
    saved, intact = book.export(), book.export()
    if edit == "missing_list":
        saved["last"] = None
    elif edit == "malformed_cursor":
        saved["last"][0] = ["US/AAA"]
    elif edit == "future_cursor":
        saved["last"][0][1] = AT + 12
    elif edit == "duplicate_cursor":
        saved["last"].append(saved["last"][-1])
    elif edit == "emission_schema":
        saved["emissions"][0]["extra"] = True
    elif edit == "emission_order":
        saved["emissions"].reverse()
    elif edit == "failure_schema":
        saved["failures"][0]["extra"] = True
    elif edit == "duplicate_failure":
        saved["failures"].append(saved["failures"][0])
    elif edit == "unknown_failure":
        saved["failures"][0]["context_id"] = "1" * 64
    elif edit == "below_threshold":
        saved["failures"][0]["target"] = 0.25
    elif edit == "failure_order":
        saved["failures"].reverse()
    elif edit == "duplicate_pending":
        saved["pending"] *= 2
    elif edit == "unknown_pending":
        saved["pending"][0] = "2" * 64
    else:
        saved["state_sha256"] = "3" * 64
    with pytest.raises(ValueError):
        api().FailureLedger.restore(layout(), configuration(), saved)
    assert book.export() == intact
