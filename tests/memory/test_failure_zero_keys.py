"""Claves nulas admitidas por el codec GRU, sin inventar una dirección de semejanza."""

import numpy as np
import pytest

from tests.memory.test_failure_context import AT, api, arguments, context, layout
from tests.memory.test_failure_ledger import api as ledger_api
from tests.memory.test_failure_ledger import configuration


def test_the_original_nonzero_contract_preserves_its_existing_identities():
    assert (
        layout().fingerprint() == "b1bf0c3044b95d04c53c95a6cbedee05de38e452b0928629eaab5ed1ee4ce606"
    )
    assert (
        context().context_id == "65991b9fbe19aa1176af057a94e9a3c6b7d382634d3d188f7d119e48bd09432b"
    )


def capture(*, at=AT, flow="US/AAA", zero=True):
    return api().FailureContext(
        layout(allow_zero_key=True),
        **arguments(
            flow_id=flow,
            sample_id=f"fixture/{flow}/{at}",
            prediction_at=at,
            input_available_at=at - 1,
            key_inputs=np.zeros(3) if zero else np.array([1.0, 0.0, 0.0]),
        ),
    )


def test_a_zero_key_is_explicit_and_keeps_its_context_without_a_fabricated_unit_vector():
    record = capture()
    np.testing.assert_array_equal(record.normalized_key(), np.zeros(3))
    assert record.layout_id != layout().fingerprint()
    assert layout(allow_zero_key=True).identity()["zero_key_policy"] == (
        "keep_context_exclude_similarity_v1"
    )
    assert (
        record.export()
        == api().FailureContext.restore(layout(allow_zero_key=True), record.export()).export()
    )
    with pytest.raises(ValueError):
        api().FailureContext.restore(layout(), record.export())
    with pytest.raises(ValueError):
        context(key_inputs=np.zeros(3))


@pytest.mark.parametrize("cache_keys", [False, True])
def test_zero_keys_keep_emissions_and_mature_errors_but_do_not_enter_similarity(cache_keys):
    design, config = layout(allow_zero_key=True), configuration(cache_keys=cache_keys)
    book = ledger_api().FailureLedger(design, config)
    zero, directed = capture(), capture(flow="US/BBB", zero=False)
    book.begin(AT)
    book.issue([(zero, 0.0), (directed, 0.0)])
    book.begin(AT + 10)
    book.resolve([("US/AAA", AT, AT + 10, 4.0), ("US/BBB", AT, AT + 10, 1.0)])
    book.begin(AT + 11)
    for value in (book, ledger_api().FailureLedger.restore(design, config, book.export())):
        missing = value.query(capture(at=AT + 11))
        assert missing.risk == 0 and missing.confidence == 1 and missing.eligible == 0
        assert missing.context_id is None and missing.failure_context_id is None
        valid = value.query(capture(at=AT + 11, zero=False))
        assert valid.risk == 0.5 and valid.eligible == 1
        assert valid.failure_context_id == directed.context_id
        assert value.counts() == dict(issued=2, resolved=2, censored=0, admitted=2, pending=0)
        assert len(value.export()["failures"]) == 2
        for failure in value._snapshot:
            assert failure.emission.normalized_key().shape == (3,)


def test_zero_permission_does_not_accept_nonfinite_keys_or_implicit_truth_values():
    with pytest.raises(ValueError):
        layout(allow_zero_key="yes")
    with pytest.raises(ValueError):
        api().FailureContext(
            layout(allow_zero_key=True), **arguments(key_inputs=np.full(3, np.nan))
        )


def test_zero_keys_do_not_allocate_a_redundant_normalized_cache():
    books = []
    for enabled in (False, True):
        book = ledger_api().FailureLedger(
            layout(allow_zero_key=True), configuration(cache_keys=enabled)
        )
        book.begin(AT)
        book.issue([(capture(), 0.0), (capture(flow="US/BBB", zero=False), 0.0)])
        books.append(book)
    assert books[1].logical_bytes() - books[0].logical_bytes() == 3 * 8
