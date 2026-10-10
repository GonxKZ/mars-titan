"""Compactación y liberación de las tablas de predicciones con lectura bit a bit.

Las tablas son sintéticas, con el esquema y el orden de cada escritor de la campaña. Las
pruebas comprueban que una tabla compactada se lee igual que la original, con su orden y
sus bits, que una liberada conserva sus huellas y que cualquier cambio se rechaza.
"""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data import prediction_files as files
from mars_titan.data.storage import sha256
from mars_titan.training import storage_measurements as layouts

ASSETS = [
    (f"{market}/{symbol}", market, dict(validation=v, calibration=c, evaluation=e))
    for market, symbol, v, c, e in (
        ("US", "AAA", 40, 20, 80),
        ("US", "BBB", 35, 0, 70),
        ("CN", "000001.SZ", 38, 18, 75),
    )
]


def keys(partition="evaluation", seed=0):
    return layouts.synthetic_keys(ASSETS, partition, np.random.default_rng(seed))


def written(folder, table, name="evaluation-predictions.parquet"):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    pq.write_table(table, path, row_group_size=64)
    return path, sha256(path)


def writer(name, seed=1, partition="evaluation"):
    return layouts.writer_table(keys(partition), layouts.WRITERS[name], np.random.default_rng(seed))


@pytest.mark.parametrize("name", sorted(layouts.WRITERS))
def test_compacted_tables_read_back_with_the_original_schema_order_and_bits(tmp_path, name):
    table = writer(name)
    path, digest = written(tmp_path / "attempt", table)
    record = files.compact(path, digest, tmp_path / "rows")
    assert not path.exists()
    assert files.verify(path, digest) == files.COMPACTED
    restored = files.read(path, digest)
    assert files.same_table(restored, table)
    assert record["content_sha256"] == files.content_digest(table)
    assert record["rows"] == table.num_rows
    own = pq.read_table(path.parent / record["compact"]["path"])
    ordered = np.array_equal(files.canonical_order(table), np.arange(table.num_rows))
    assert (files.ROW_INDEX in own.column_names) is not ordered
    assert "zero" not in own.column_names
    if layouts.WRITERS[name].quantiles is not None:
        assert "prediction" not in own.column_names
    # Una lectura de columnas devuelve esas columnas de la tabla original.
    columns = ["asset_id", "prediction", "target"]
    assert files.same_table(files.read(path, digest, columns), table.select(columns))
    # Volver a compactar no cambia nada.
    assert files.compact(path, digest, tmp_path / "rows") == record


def test_fits_of_one_partition_share_a_single_rows_table(tmp_path):
    base = writer("neural", seed=2)
    other = writer("ridge", seed=3)
    # Las claves y el objetivo son los mismos, solo cambian los decimales.
    other = other.set_column(4, "target", base["target"])
    first, a = written(tmp_path / "a", base)
    second, b = written(tmp_path / "b", other)
    one = files.compact(first, a, tmp_path / "rows")
    two = files.compact(second, b, tmp_path / "rows")
    assert len(list((tmp_path / "rows").glob("rows-*.parquet"))) == 1
    assert (first.parent / one["compact"]["rows"]["path"]).resolve() == (
        second.parent / two["compact"]["rows"]["path"]
    ).resolve()
    assert files.rows_references(tmp_path / "a") == files.rows_references(tmp_path / "b")
    assert files.same_table(files.read(second, b), other)


def test_signed_zeros_nan_payloads_nulls_and_distinct_points_survive(tmp_path):
    table = writer("titans", seed=4)
    count = table.num_rows
    prediction = table["prediction"].to_numpy().copy()
    prediction[0] = -0.0
    nan = np.array([0x7FF8_0000_0000_0123], dtype=np.uint64).view(np.float64)[0]
    prediction[1] = nan
    median = table["quantile_0500"].to_numpy().copy()
    median[2] = 0.1  # La mediana deja de ser float32 exacto: se guarda en float64.
    zero = np.zeros(count)
    zero[3] = -0.0
    nulls = pa.array([None if i % 7 == 0 else float(i) for i in range(count)], pa.float32())
    table = (
        table.set_column(5, "prediction", pa.array(prediction))
        .set_column(6, "zero", pa.array(zero))
        .set_column(table.column_names.index("quantile_0500"), "quantile_0500", pa.array(median))
        .append_column("extra", nulls)
    )
    path, digest = written(tmp_path / "attempt", table)
    record = files.compact(path, digest, tmp_path / "rows")
    own = pq.read_table(path.parent / record["compact"]["path"])
    assert {"prediction", "zero", "extra"} <= set(own.column_names)
    assert own.schema.field("quantile_0500").type == pa.float64()
    assert own.schema.field("quantile_0100").type == pa.float32()
    restored = files.read(path, digest)
    assert files.same_table(restored, table)
    bits = restored["prediction"].to_numpy().view(np.uint64)
    assert bits[0] == 0x8000_0000_0000_0000 and bits[1] == 0x7FF8_0000_0000_0123


def test_narrowing_keeps_float64_unless_float32_round_trips_every_bit():
    exact = pa.array([0.5, -0.0, None, 3.0])
    assert files._narrow(exact).type == pa.float32()
    assert files._narrow(pa.array([0.5, 0.1])).type == pa.float64()
    assert files._narrow(pa.array([1.0], pa.float32())).type == pa.float32()
    payload = np.array([0x7FF8_0000_0000_0001], dtype=np.uint64).view(np.float64)
    assert files._narrow(pa.array(payload)).type == pa.float64()


def test_content_digest_sees_order_bits_nulls_and_schema():
    table = pa.table(
        dict(
            asset_id=pa.array(["US/A", "US/B", None]),
            prediction_at=pa.array([1, 2, 3], pa.timestamp("us", tz="UTC")),
            value=pa.array([0.0, 1.5, None]),
        )
    )
    digest = files.content_digest(table)
    assert files.content_digest(pa.Table.from_batches(table.to_batches(max_chunksize=1))) == (
        digest
    )
    assert files.content_digest(table.slice(1)) != digest
    assert files.content_digest(table.take([1, 0, 2])) != digest
    assert files.content_digest(table.set_column(2, "value", pa.array([-0.0, 1.5, None]))) != (
        digest
    )
    assert files.content_digest(table.set_column(2, "value", pa.array([0.0, 1.5, 0.0]))) != (digest)
    assert (
        files.content_digest(table.set_column(0, "asset_id", pa.array(["US/A", "US/B", ""])))
        != digest
    )
    assert (
        files.content_digest(table.set_column(0, "asset_id", pa.array(["US/AU", "S/B", None])))
        != digest
    )
    assert files.content_digest(table.rename_columns(["asset_id", "prediction_at", "x"])) != (
        digest
    )
    narrow = table.set_column(2, "value", table["value"].cast(pa.float32()))
    assert files.content_digest(narrow) != digest
    with pytest.raises(ValueError, match="sin huella"):
        files.content_digest(pa.table(dict(x=pa.array([[1]]))))


def test_released_files_keep_digests_and_only_accept_the_same_content(tmp_path):
    table = writer("episodic_gru", seed=5)
    path, digest = written(tmp_path / "attempt", table)
    record = files.release(path, digest, job="US/fold-000/gru/search-c00-s42", stage="base")
    assert not path.exists() and record["state"] == files.RELEASED
    assert record["job"] == "US/fold-000/gru/search-c00-s42"
    assert files.verify(path, digest) == files.RELEASED
    with pytest.raises(files.PredictionsReleased, match="Regenéralas"):
        files.read(path, digest)
    assert files.matches(path, digest, table)
    flipped = table["prediction"].to_numpy().copy()
    flipped[0] = np.nextafter(flipped[0], np.float32(np.inf))
    assert not files.matches(path, digest, table.set_column(5, "prediction", pa.array(flipped)))
    assert not files.matches(path, digest, table.take(files.canonical_order(table)))
    assert not files.matches(path, digest, table.slice(1))
    assert files.release(path, digest) == record


def test_releasing_a_compacted_file_removes_its_decimals_and_keeps_the_rows(tmp_path):
    table = writer("neural", seed=6)
    path, digest = written(tmp_path / "attempt", table)
    compacted = files.compact(path, digest, tmp_path / "rows")
    decimals = path.parent / compacted["compact"]["path"]
    record = files.release(path, digest)
    assert not decimals.exists()
    assert len(list((tmp_path / "rows").glob("rows-*.parquet"))) == 1
    assert files.rows_references(path.parent) == set()
    for key in ("sha256", "rows", "content_sha256", "canonical_sha256", "bytes"):
        assert record[key] == compacted[key]
    assert "compact" not in record
    assert files.matches(path, digest, table)


def test_present_files_are_read_and_matched_directly(tmp_path):
    table = writer("xgboost", seed=7)
    path, digest = written(tmp_path / "attempt", table)
    assert files.verify(path, digest) == files.PRESENT
    assert files.entry(path) is None
    assert files.same_table(files.read(path, digest), table)
    assert files.matches(path, digest, table)
    assert not files.matches(path, digest, table.slice(1))


def test_changed_missing_or_unregistered_files_are_rejected(tmp_path):
    table = writer("neural", seed=8)
    path, digest = written(tmp_path / "attempt", table)
    with pytest.raises(ValueError, match="no coincide"):
        files.verify(path, "0" * 64)
    record = files.compact(path, digest, tmp_path / "rows")
    with pytest.raises(ValueError, match="no es el archivo registrado"):
        files.verify(path, "0" * 64)
    with pytest.raises(ValueError, match="no coincide"):
        files.verify(path.with_name("other.parquet"), digest)
    # Un archivo nuevo con otra huella en la ruta liberada no se acepta.
    path.write_bytes(b"otro")
    with pytest.raises(ValueError, match="existe con otra huella"):
        files.verify(path, digest)
    path.unlink()
    decimals = path.parent / record["compact"]["path"]
    original = decimals.read_bytes()
    decimals.write_bytes(original + b"x")
    with pytest.raises(ValueError, match="decimales"):
        files.read(path, digest)
    decimals.write_bytes(original)
    rows = path.parent / record["compact"]["rows"]["path"]
    saved = rows.read_bytes()
    rows.write_bytes(saved + b"x")
    with pytest.raises(ValueError, match="tabla común"):
        files.verify(path, digest)
    rows.write_bytes(saved)
    assert files.same_table(files.read(path, digest), table)


def test_a_rows_table_with_other_targets_cannot_hold_the_decimals(tmp_path):
    table = writer("neural", seed=9)
    rows = files.shared_rows(table)
    target = rows["target"].to_numpy().copy()
    target[0] += 1.0
    with pytest.raises(ValueError, match="no son los de la tabla común"):
        files.split(table, rows.set_column(4, "target", pa.array(target)))
    with pytest.raises(ValueError, match="claves y el objetivo"):
        files.split(table.drop_columns(["target"]), rows)
    with pytest.raises(ValueError, match="reservada"):
        files.split(table.append_column("row", pa.array(np.zeros(table.num_rows))), rows)
    with pytest.raises(ValueError, match="decimales propios"):
        files.split(table.select(list(files.SHARED_COLUMNS)), rows)


def test_restore_rejects_decimals_without_schema_or_rows(tmp_path):
    table = writer("neural", seed=10)
    rows = files.shared_rows(table)
    own = files.split(table, rows)
    with pytest.raises(ValueError, match="esquema original"):
        files.restore(rows, own.replace_schema_metadata(None))
    with pytest.raises(ValueError, match="filas de la tabla común"):
        files.restore(rows.slice(1), own)
    with pytest.raises(ValueError, match="reconstruir"):
        files.restore(rows, own.drop_columns(["quantile_0500"]))


def test_a_cut_after_publishing_the_record_keeps_a_readable_file(tmp_path, monkeypatch):
    table = writer("titans", seed=11)
    path, digest = written(tmp_path / "attempt", table)
    unlink = type(path).unlink

    def interrupted(self, *args, **kwargs):
        if self.name == path.name:
            raise OSError("corte")
        return unlink(self, *args, **kwargs)

    monkeypatch.setattr(type(path), "unlink", interrupted)
    with pytest.raises(OSError, match="corte"):
        files.compact(path, digest, tmp_path / "rows")
    # El registro ya dice compactado, pero el original sigue presente y manda.
    assert files.entry(path)["state"] == files.COMPACTED
    assert files.verify(path, digest) == files.PRESENT
    with pytest.raises(OSError, match="corte"):
        files.release(path, digest)
    monkeypatch.setattr(type(path), "unlink", unlink)
    assert files.verify(path, digest) == files.PRESENT
    record = files.release(path, digest)
    assert record["state"] == files.RELEASED and not path.exists()
    assert files.matches(path, digest, table)


def test_a_cut_while_releasing_a_compacted_file_leaves_no_decimals_behind(tmp_path, monkeypatch):
    table = writer("neural", seed=12)
    path, digest = written(tmp_path / "attempt", table)
    record = files.compact(path, digest, tmp_path / "rows")
    decimals = path.parent / record["compact"]["path"]
    unlink = type(path).unlink

    def interrupted(self, *args, **kwargs):
        if self.name == decimals.name:
            raise OSError("corte")
        return unlink(self, *args, **kwargs)

    monkeypatch.setattr(type(path), "unlink", interrupted)
    with pytest.raises(OSError, match="corte"):
        files.release(path, digest)
    monkeypatch.setattr(type(path), "unlink", unlink)
    assert decimals.exists() and files.verify(path, digest) == files.RELEASED
    files.release(path, digest)
    assert not decimals.exists()


def test_the_retention_record_is_validated(tmp_path):
    folder = tmp_path / "attempt"
    folder.mkdir()
    (folder / files.RETENTION_FILE).write_text(json.dumps(dict(kind="other", files={})))
    with pytest.raises(ValueError, match="no es un registro"):
        files.entry(folder / "evaluation-predictions.parquet")
    (folder / files.RETENTION_FILE).write_text(json.dumps(dict(kind="prediction_retention")))
    with pytest.raises(ValueError, match="no es un registro"):
        files.entry(folder / "evaluation-predictions.parquet")
