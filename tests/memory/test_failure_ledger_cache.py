"""Paridad y presupuesto de la caché S2, sin aprendizaje."""

import numpy as np
import pytest

from mars_titan.memory.failure_context import FailureContext
from tests.memory.test_failure_context import AT, layout
from tests.memory.test_failure_ledger import api, configuration, ledger, mature, row


@pytest.mark.parametrize("geometry", ["associated", "shuffled"])
def test_cached_and_reference_keys_preserve_scores_and_recovery_exactly(geometry):
    rng = np.random.default_rng(790)
    records = [row(f"A{i}", key=rng.normal(size=3)) for i in range(8)]
    books = [ledger(cache_keys=flag, geometry=geometry, max_pending=16) for flag in (False, True)]
    for book in books:
        book.begin(AT)
        book.issue([(record, 0.1) for record in records])
        mature(book, records, [1.0 + i / 10 for i in range(8)])
        book.begin(AT + 11)
    for _ in range(12):
        query = row(at=AT + 11, key=rng.normal(size=3))
        assert books[0].query(query) == books[1].query(query)
    for flag, book in zip((False, True), books, strict=True):
        restored = api().FailureLedger.restore(
            layout(),
            configuration(cache_keys=flag, geometry=geometry, max_pending=16),
            book.export(),
        )
        assert restored.query(query) == book.query(query)
        assert restored.logical_bytes() == book.logical_bytes()
    assert books[1].logical_bytes() - books[0].logical_bytes() == 4 * 3 * 8


def test_cached_queries_do_not_recompute_normalization_of_historical_contexts(monkeypatch):
    book = ledger(cache_keys=True)
    book.begin(AT)
    book.issue([(row(), 0.0)])
    mature(book, [row()], [1.0])
    book.begin(AT + 11)
    original = FailureContext.normalized_key

    def only_current(context):
        if context.prediction_at != AT + 11:
            pytest.fail("Se volvió a normalizar un contexto histórico ya capturado")
        return original(context)

    monkeypatch.setattr(FailureContext, "normalized_key", only_current)
    assert book.query(row(at=AT + 11)).risk == 0.5


def test_the_cache_has_an_explicit_boolean_switch():
    with pytest.raises(ValueError):
        configuration(cache_keys="auto")


def test_a_borrowed_cached_key_cannot_be_modified_in_place():
    book = ledger(cache_keys=True)
    book.begin(AT)
    book.issue([(row(key=(3.0, 4.0, 0.0)), 0.0)])
    emission = next(iter(book._pending.values()))
    key = emission.normalized_key()
    np.testing.assert_array_equal(key, [0.6, 0.8, 0.0])
    with pytest.raises(ValueError):
        key[0] = 0
    np.testing.assert_array_equal(emission.normalized_key(), [0.6, 0.8, 0.0])
