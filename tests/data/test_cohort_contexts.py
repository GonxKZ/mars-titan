"""Contextos temporales equivalentes a las referencias y con memoria acotada."""

import importlib
from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.fundamentals import snapshot
from mars_titan.data.samples import macro_vector


def module():
    try:
        return importlib.import_module("mars_titan.data.cohort_contexts")
    except ModuleNotFoundError:
        pytest.fail("Faltan los contextos temporales del corpus")


def test_news_windows_read_inclusive_bounds_across_groups(tmp_path):
    start = datetime(2023, 1, 1, tzinfo=UTC)
    rows = [
        dict(
            available_at=start + timedelta(days=i),
            content_hash=f"{i:064x}",
            event_id=f"{i + 10:064x}",
            content_kind="summary",
            cohort_id="original_audited",
            availability_rule="date_only_next_session",
            text=f"Texto {i}",
        )
        for i in range(9)
    ]
    path = tmp_path / "news.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    with module().NewsWindows(path) as window:
        selected = list(window.between(start + timedelta(days=2), start + timedelta(days=4)))
        assert selected == rows[2:5]
        assert (
            list(window.between(start + timedelta(days=7), start + timedelta(days=8))) == rows[7:]
        )
        assert window.cached_row_groups <= 2
        assert list(window.between(start - timedelta(days=2), start - timedelta(days=1))) == []


def test_news_group_over_budget_and_unordered_time_are_rejected(tmp_path):
    now = datetime(2023, 1, 3, tzinfo=UTC)
    rows = [
        dict(
            available_at=now - timedelta(days=i),
            content_hash=f"{i:064x}",
            event_id=f"{i:064x}",
            content_kind="article_candidate",
            cohort_id="original_audited",
            availability_rule="date_only_next_session",
            text="x" * 1000,
        )
        for i in range(2)
    ]
    path = tmp_path / "news.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)
    with pytest.raises(ValueError, match="orden"):
        module().NewsWindows(path)
    rows.reverse()
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="presupuesto"):
        module().NewsWindows(path, max_group_bytes=100)


def test_fact_cursor_preserves_revisions_ambiguities_and_matured_data():
    start = datetime(2023, 1, 1, tzinfo=UTC)
    base = dict(
        concept="assets", value=10.0, period_end="2022-12-31", period_start=None, accession="a"
    )
    rows = [
        {**base, "available_at": start},
        {**base, "available_at": start + timedelta(days=2), "value": 20.0},
        {**base, "available_at": start + timedelta(days=2), "value": 30.0, "accession": "b"},
        {**base, "available_at": start + timedelta(days=4), "value": 40.0},
    ]
    cursor = module().FactCursor(rows)
    for day in range(6):
        cutoff = start + timedelta(days=day)
        assert cursor.at(cutoff) == snapshot(rows, cutoff)
    with pytest.raises(ValueError, match="retroced"):
        cursor.at(start)


def macro_rows():
    start = datetime(2023, 1, 2, tzinfo=UTC)
    return [
        dict(
            prediction_at=start + timedelta(days=d),
            indicator_id=name,
            value=(None if name == "absent" else float(d + 1)),
            available_at=(None if name == "absent" else start),
            unit="ratio",
        )
        for d in range(3)
        for name in ("absent", "observed")
    ]


def test_macro_vectors_match_reference_and_keep_missing_values(tmp_path):
    rows = macro_rows()
    path = tmp_path / "macro.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
    cache = module().MacroVectors(path)
    assert cache.indicators == ["absent", "observed"]
    assert cache.nbytes < 1024
    for i in range(0, 6, 2):
        time = rows[i]["prediction_at"]
        vector, available = cache.at(time)
        expected, stamp = macro_vector(rows[i : i + 2], time)
        np.testing.assert_array_equal(vector, np.asarray(expected, dtype=np.float32))
        assert available == stamp
        assert vector.flags.writeable is False
    assert cache.at(datetime(2022, 1, 1, tzinfo=UTC)) is None


def test_macro_memory_limit_and_future_input_fail_closed(tmp_path):
    rows = macro_rows()
    path = tmp_path / "macro.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="presupuesto"):
        module().MacroVectors(path, max_bytes=1)
    rows[1]["available_at"] += timedelta(days=1)
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="futura"):
        module().MacroVectors(path)


def test_macro_duplicate_group_stops_before_collecting_the_whole_group(tmp_path, monkeypatch):
    rows = macro_rows()[:2] * 100
    path = tmp_path / "macro.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    original = pq.ParquetFile
    consumed = []

    class ObservedFile:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

        def iter_batches(self, **kwargs):
            if len(kwargs["columns"]) > 1:
                kwargs["batch_size"] = 1
            for batch in self.inner.iter_batches(**kwargs):
                if len(kwargs["columns"]) > 1:
                    consumed.append(batch.num_rows)
                yield batch

    monkeypatch.setattr(module().pq, "ParquetFile", ObservedFile)
    with pytest.raises(ValueError, match="indicadores"):
        module().MacroVectors(path)
    assert sum(consumed) <= 3


def test_dictionary_encoded_news_does_not_expand_whole_group_before_budget_check(
    tmp_path, monkeypatch
):
    moment = datetime(2023, 1, 1, tzinfo=UTC)
    row = dict(
        available_at=moment,
        content_hash="a" * 64,
        event_id="b" * 64,
        content_kind="summary",
        text="x" * 100_000,
        cohort_id="original_audited",
        availability_rule="date_only_next_session",
    )
    path = tmp_path / "news.parquet"
    pq.write_table(pa.Table.from_pylist([row] * 200), path)
    original, sizes = pq.ParquetFile, []

    class ObservedFile:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def read_row_group(self, *args, **kwargs):
            table = self.inner.read_row_group(*args, **kwargs)
            sizes.append(table.nbytes)
            return table

    monkeypatch.setattr(module().pq, "ParquetFile", ObservedFile)
    with module().NewsWindows(path, max_group_bytes=200_000) as windows:
        assert sum(1 for _ in windows.between(moment, moment)) == 200
    assert max(sizes) <= 200_000
