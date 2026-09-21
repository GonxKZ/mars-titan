"""Lecturas macro y escrituras Parquet acotadas con fallos recuperables."""

import importlib
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def module():
    try:
        return importlib.import_module("mars_titan.data.batches")
    except ModuleNotFoundError:
        pytest.fail("Falta la lectura y escritura acotada de datos preparados")


def macro_rows():
    start = datetime(2023, 1, 3, 21, 5, tzinfo=UTC)
    return [
        {
            "prediction_at": start + timedelta(days=day),
            "indicator_id": name,
            "value": day * 10 + offset,
            "available_at": start,
            "unit": "index",
        }
        for day in range(4)
        for offset, name in enumerate(["a", "b", "c"])
    ]


def test_macro_context_includes_both_sides_of_row_group_boundary(tmp_path):
    rows = macro_rows()
    path = tmp_path / "macro.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=4)
    with module().MacroContexts(path) as reader:
        assert reader.indicators == ["a", "b", "c"]
        assert reader.at(rows[3]["prediction_at"]) == rows[3:6]
        assert reader.at(rows[6]["prediction_at"]) == rows[6:9]
        assert reader.at(datetime(2000, 1, 1, tzinfo=UTC)) == []
        assert reader.cached_row_groups <= 2


def test_bounded_reader_refuses_oversized_group_before_decode(tmp_path):
    path = tmp_path / "macro.parquet"
    pq.write_table(pa.Table.from_pylist(macro_rows()), path)
    with pytest.raises(ValueError, match="presupuesto"):
        module().MacroContexts(path, max_group_bytes=1)


def test_atomic_batch_write_preserves_old_artifact_when_later_batch_fails(tmp_path):
    path = tmp_path / "data.parquet"
    original = pa.table({"value": [99]})
    pq.write_table(original, path)
    before = path.read_bytes()

    def broken():
        yield pa.table({"value": [1, 2]})
        raise ValueError("fallo de entrada")

    with pytest.raises(ValueError, match="entrada"):
        module().atomic_parquet_batches(path, broken())
    assert path.read_bytes() == before
    count = module().atomic_parquet_batches(
        path, [pa.table({"value": [1, 2]}), pa.table({"value": [3]})]
    )
    assert count == 3
    assert pq.read_table(path)["value"].to_pylist() == [1, 2, 3]
    assert pq.ParquetFile(path).metadata.num_row_groups == 2


def test_prepared_partition_limits_fail_before_decoding(tmp_path):
    path = tmp_path / "news.parquet"
    pq.write_table(pa.table({"text": ["abc"] * 5}), path)
    with pytest.raises(ValueError, match="presupuesto"):
        module().read_bounded_table(path, max_rows=4)
    assert module().read_bounded_table(path, max_rows=5).num_rows == 5


def test_dictionary_encoded_unit_is_rejected_before_full_group_expansion(tmp_path, monkeypatch):
    rows = [{**macro_rows()[0], "unit": "x" * 16384} for _ in range(512)]
    path = tmp_path / "macro.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    reader = module().MacroContexts(path, max_group_bytes=1024**2)
    monkeypatch.setattr(
        reader.file,
        "read_row_group",
        lambda *a, **k: pytest.fail("Se ha expandido el grupo completo"),
    )
    with reader, pytest.raises(ValueError, match="unidad"):
        reader.at(rows[0]["prediction_at"])


def test_dictionary_encoded_indicator_has_an_explicit_length_limit(tmp_path):
    path = tmp_path / "macro.parquet"
    pq.write_table(pa.Table.from_pylist([{**macro_rows()[0], "indicator_id": "x" * 200}]), path)
    with pytest.raises(ValueError, match="indicador"):
        module().MacroContexts(path)
