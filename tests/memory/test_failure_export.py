"""Exportaciones analíticas de fixtures S2, sin evaluación científica ni aprendizaje."""

import importlib
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.memory.test_failure_context import AT
from tests.memory.test_failure_ledger import ledger, mature, row


def api():
    return importlib.import_module("mars_titan.memory.failure_export")


def populated():
    book = ledger()
    book.begin(AT)
    records = [row(), row("BBB")]
    book.issue([(record, 0.25) for record in records])
    mature(book, records[:1], [1.25])
    return book, records


def test_arrow_exports_only_the_selected_mature_scope_and_keeps_the_actual_emission():
    book, records = populated()
    current = api().failure_reader(book, scope="retained", batch_rows=1).read_all()
    assert isinstance(current, pa.Table)
    assert current.num_rows == 1
    record = current.to_pylist()[0]
    assert record["context_id"] == records[0].context_id
    assert record["issued_prediction"] == 0.25
    assert record["target"] == 1.25 and record["error"] == 1.0
    assert record["macro_available_at"] == [None, None]
    assert record["area_activity"] == [True, True, True]
    metadata = json.loads(current.schema.metadata[b"failure_memory"])
    assert metadata["scope"] == "retained_after_feedback"
    assert metadata["pending_excluded"] == 1 and metadata["exported"] == 1
    assert metadata["history_complete"] is False
    assert api().failure_reader(book, scope="snapshot").read_all().num_rows == 0
    book.begin(AT + 11)
    assert api().failure_reader(book, scope="snapshot").read_all().num_rows == 1


def test_batches_are_bounded_and_a_later_mutation_does_not_change_the_prepared_export():
    book, _ = populated()
    book.begin(AT + 11)
    reader = api().failure_reader(book, batch_rows=1)
    book.resolve([("US/BBB", AT, AT + 10, 2.25)])
    book.begin(AT + 12)
    batches = list(reader)
    assert all(batch.num_rows <= 1 for batch in batches)
    assert sum(batch.num_rows for batch in batches) == 1


@pytest.mark.parametrize("as_of", [AT, AT + 12])
def test_a_current_checkpoint_cannot_reconstruct_another_time_by_filtering(as_of):
    book, _ = populated()
    with pytest.raises(ValueError):
        api().failure_reader(book, as_of=as_of)


def test_parquet_export_keeps_the_schema_and_does_not_overwrite_an_existing_result(tmp_path):
    book, _ = populated()
    path = api().write_failures(book, tmp_path, scope="retained", batch_rows=1)
    table = pq.read_table(path)
    assert table.num_rows == 1
    assert table.schema.metadata[b"failure_memory"]
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        api().write_failures(book, tmp_path, scope="retained", batch_rows=1)
    assert path.read_bytes() == before


def test_duckdb_counts_cover_only_the_eligible_exported_set():
    book, _ = populated()
    summary = api().failure_counts(book, scope="retained", batch_rows=1)
    assert summary == [
        dict(
            market="US",
            action="fixed_consumer/no_tool",
            eligible_failures=1,
            first_available_at=AT + 10,
            last_available_at=AT + 10,
        )
    ]


def test_empty_exports_have_the_same_explicit_schema_and_no_fake_rows(tmp_path):
    book = ledger()
    table = api().failure_reader(book).read_all()
    assert table.num_rows == 0
    assert "issued_prediction" in table.schema.names
    path = api().write_failures(book, tmp_path)
    restored = pq.read_table(path)
    assert restored.num_rows == 0
    assert restored.schema.equals(table.schema, check_metadata=True)
    assert api().failure_counts(book) == []


@pytest.mark.parametrize(
    "options",
    [dict(batch_rows=0), dict(batch_rows=True), dict(batch_rows=513), dict(scope="all_history")],
)
def test_export_limits_and_scope_are_explicit(options):
    with pytest.raises(ValueError):
        api().failure_reader(ledger(), **options)


def test_an_incoherent_book_is_rejected_before_any_batch_is_exposed():
    book, _ = populated()
    book._issued += 1
    with pytest.raises(ValueError):
        api().failure_reader(book)


def test_a_concurrent_publication_is_not_overwritten(monkeypatch, tmp_path):
    book, _ = populated()
    module = api()
    destination = tmp_path / f"failure-retained-{book.export()['state_sha256']}.parquet"
    original = module.atomic_parquet_batches

    def publish_between_check_and_write(path, tables):
        destination.write_bytes(b"fixture-concurrent-publication")
        return original(path, tables)

    monkeypatch.setattr(module, "atomic_parquet_batches", publish_between_check_and_write)
    with pytest.raises(FileExistsError):
        module.write_failures(book, tmp_path, scope="retained")
    assert destination.read_bytes() == b"fixture-concurrent-publication"


def test_several_failures_are_split_into_the_requested_batch_size():
    book = ledger()
    records = [row(), row("BBB"), row("CCC")]
    book.begin(AT)
    book.issue([(record, 0.0) for record in records])
    mature(book, records, [1.0, 2.0, 3.0])
    book.begin(AT + 11)
    batches = list(api().failure_reader(book, batch_rows=1))
    assert [batch.num_rows for batch in batches] == [1, 1, 1]


@pytest.mark.parametrize("scope", ["snapshot", "retained"])
def test_future_receipt_is_rejected_even_when_only_the_past_scope_is_requested(scope):
    book, _ = populated()
    book._retained = (replace(book._retained[0], received_at=AT + 11),)
    with pytest.raises(ValueError):
        api().failure_reader(book, scope=scope)


def test_duckdb_uses_the_declared_memory_threads_and_extension_limits(monkeypatch):
    module = api()
    connect = module.duckdb.connect
    observed = []

    def checked_connection(*args, **kwargs):
        observed.append(kwargs["config"])
        return connect(*args, **kwargs)

    monkeypatch.setattr(module.duckdb, "connect", checked_connection)
    module.failure_counts(ledger())
    assert observed == [
        dict(
            threads=1,
            memory_limit="64MiB",
            max_temp_directory_size="0B",
            autoinstall_known_extensions=False,
            autoload_known_extensions=False,
        )
    ]
