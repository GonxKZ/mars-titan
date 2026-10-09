"""Disposiciones medidas de las tablas por fila: esquema real y relectura bit a bit.

Las tablas son sintéticas y pequeñas. Las pruebas no leen vistas ni modelos y comprueban
que cada disposición que se mide devuelve exactamente los bits escritos, y que la tabla
común con los decimales de un ajuste reconstruye su tabla original.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.training import storage_measurements as layouts

ASSETS = [
    (f"{market}/{symbol}", market, dict(validation=v, calibration=c, evaluation=e))
    for market, symbol, v, c, e in (
        ("US", "AAA", 40, 20, 80),
        ("US", "BBB", 35, 0, 70),
        ("CN", "000001.SZ", 38, 18, 75),
        ("CN", "600000.SH", 12, 9, 30),
    )
]


def keys(partition="evaluation", seed=0):
    return layouts.synthetic_keys(ASSETS, partition, np.random.default_rng(seed))


def test_synthetic_keys_keep_the_rows_of_each_asset_in_unique_ordered_sessions():
    table = keys()
    assert table.num_rows == 80 + 70 + 75 + 30
    asset = table["asset_id"].to_pylist()
    moment = table["prediction_at"].cast(pa.int64()).to_numpy()
    for key, _, counts in ASSETS:
        rows = [i for i, value in enumerate(asset) if value == key]
        assert len(rows) == counts["evaluation"]
        assert np.all(np.diff(moment[rows]) > 0)
    expected = [f"{a}/{t}" for a, t in zip(asset, moment.tolist(), strict=True)]
    assert table["sample_id"].to_pylist() == expected
    assert table.schema.field("prediction_at").type == pa.timestamp("us", tz="UTC")
    with pytest.raises(ValueError, match="no tiene filas"):
        layouts.synthetic_keys(
            [(key, m, dict(evaluation=0)) for key, m, _ in ASSETS],
            "evaluation",
            np.random.default_rng(0),
        )


@pytest.mark.parametrize("name", sorted(layouts.WRITERS))
def test_writer_tables_repeat_the_columns_types_and_order_of_each_writer(name):
    writer = layouts.WRITERS[name]
    table = layouts.writer_table(keys(), writer, np.random.default_rng(1))
    expected = ["sample_id", "asset_id", "market", "prediction_at", "target", "prediction", "zero"]
    if writer.quantiles is not None:
        expected += list(QUANTILE_COLUMNS)
        assert all(table.schema.field(c).type == writer.quantiles for c in QUANTILE_COLUMNS)
        levels = np.stack([table[c].to_numpy() for c in QUANTILE_COLUMNS], axis=1)
        assert np.all(np.diff(levels, axis=1) >= 0)
        assert layouts.same_table(
            table.select(["prediction"]),
            table.select(["quantile_0500"]).rename_columns(["prediction"]),
        )
    assert table.column_names == expected
    assert table.schema.field("prediction").type == writer.prediction
    assert table.schema.field("target").type == pa.float64()
    point = table["prediction"].to_numpy()
    assert np.array_equal(point.astype(np.float32).astype(point.dtype), point) == (
        writer.float32_values
    )
    moment = table["prediction_at"].cast(pa.int64()).to_numpy()
    assert bool(np.all(np.diff(moment) >= 0)) == writer.chronological


def test_current_layout_writes_one_row_group_per_writer_batch(tmp_path):
    writer = layouts.WRITERS["neural"]
    table = layouts.writer_table(keys(), writer, np.random.default_rng(2))
    layouts.write_current(table, tmp_path / "a.parquet", writer)
    layouts.write_large(table, tmp_path / "b.parquet")
    current = pq.ParquetFile(tmp_path / "a.parquet").metadata
    large = pq.ParquetFile(tmp_path / "b.parquet").metadata
    assert current.num_row_groups == -(-table.num_rows // writer.row_group)
    assert large.num_row_groups == 1
    encodings = large.row_group(0).column(table.column_names.index("target")).encodings
    assert "BYTE_STREAM_SPLIT" in encodings


def test_same_table_compares_float_bits_and_null_masks():
    base = pa.table({"x": pa.array([0.0, 1.5, 2.0])})
    assert layouts.same_table(base, pa.table({"x": pa.array([0.0, 1.5, 2.0])}))
    assert not layouts.same_table(base, pa.table({"x": pa.array([-0.0, 1.5, 2.0])}))
    bits = np.array([0.0, 1.5, 2.0]).view(np.uint64)
    bits[1] ^= 1
    assert not layouts.same_table(base, pa.table({"x": pa.array(bits.view(np.float64))}))
    assert not layouts.same_table(base, pa.table({"x": pa.array([0.0, None, 2.0])}))
    # Un nulo no equivale al cero con el que se rellena para comparar bits.
    assert not layouts.same_table(base, pa.table({"x": pa.array([None, 1.5, 2.0])}))
    assert not layouts.same_table(base, pa.table({"x": pa.array([0.0, 1.5, 2.0], pa.float32())}))
    assert not layouts.same_table(base, base.slice(0, 2))


def test_large_groups_keep_signed_zero_subnormals_and_extremes(tmp_path):
    values = np.array([-0.0, 0.0, 5e-324, -5e-324, 1.7976931348623157e308, np.nextafter(1.0, 2)])
    small = np.array([-0.0, 0.0, 1e-45, -1e-45, 3.4028235e38, np.nextafter(np.float32(1), 2)])
    table = pa.table({"x": values, "y": small.astype(np.float32)})
    layouts.write_large(table, tmp_path / "x.parquet")
    assert layouts.same_table(pq.read_table(tmp_path / "x.parquet"), table)


@pytest.mark.parametrize("name", sorted(layouts.WRITERS))
def test_shared_rows_rebuild_each_writer_table_bit_for_bit(tmp_path, name):
    writer = layouts.WRITERS[name]
    rng = np.random.default_rng(3)
    base = keys()
    table = layouts.writer_table(base, writer, rng)
    rows = layouts.canonical_rows(layouts.writer_table(base, layouts.WRITERS["neural"], rng))
    ordered = layouts.split_shared(table, rows, writer, keep_order=True)
    order = layouts._canonical_order(table)
    moved = not np.array_equal(order, np.arange(table.num_rows))
    assert ("row" in ordered.column_names) == moved
    assert moved or not writer.chronological
    assert "zero" not in ordered.column_names and "target" not in ordered.column_names
    assert ("prediction" in ordered.column_names) == (writer.quantiles is None)
    layouts.write_large(ordered, tmp_path / "o.parquet")
    restored = layouts.restore_shared(rows, pq.read_table(tmp_path / "o.parquet"), table.schema)
    assert layouts.same_table(restored, table)
    canonical = layouts.split_shared(table, rows, writer)
    assert "row" not in canonical.column_names
    restored = layouts.restore_shared(rows, canonical, table.schema)
    assert layouts.same_table(restored, table.take(order))


def test_shared_rows_reject_tables_they_cannot_rebuild():
    writer = layouts.WRITERS["neural"]
    rng = np.random.default_rng(4)
    table = layouts.writer_table(keys(), writer, rng)
    rows = layouts.canonical_rows(table)
    median = table["quantile_0500"].to_numpy().copy()
    median.view(np.uint32)[0] ^= 1
    changed = table.set_column(
        table.column_names.index("quantile_0500"), "quantile_0500", pa.array(median)
    )
    with pytest.raises(ValueError, match="mediana"):
        layouts.split_shared(changed, rows, writer)
    zero = np.zeros(table.num_rows)
    zero[3] = -0.0
    signed = table.set_column(table.column_names.index("zero"), "zero", pa.array(zero))
    with pytest.raises(ValueError, match="cero positivo"):
        layouts.split_shared(signed, rows, writer)
    target = table["target"].to_numpy().copy()
    target[5] = np.nextafter(target[5], 1)
    moved = table.set_column(table.column_names.index("target"), "target", pa.array(target))
    with pytest.raises(ValueError, match="tabla común"):
        layouts.split_shared(moved, rows, writer)


def test_measure_partition_checks_every_layout_and_orders_their_sizes(tmp_path):
    result = layouts.measure_partition(ASSETS * 6, "evaluation", tmp_path, seed=5)
    assert result["rows"] == 6 * (80 + 70 + 75 + 30)
    for name, sizes in result["writers"].items():
        writer = layouts.WRITERS[name]
        assert set(sizes) == set(layouts.LAYOUTS)
        assert sizes["shared_rows"] < sizes["large_groups"] < sizes["current"]
        assert sizes["shared_rows"] <= sizes["shared_rows_ordered"]
        if writer.chronological:
            assert sizes["shared_rows"] < sizes["shared_rows_ordered"]
    assert not list(tmp_path.iterdir())


def test_session_aggregates_add_up_to_the_rows_they_replace():
    rng = np.random.default_rng(6)
    table = layouts.writer_table(keys(), layouts.WRITERS["neural"], rng)
    aggregates = layouts.session_aggregates(table, rng, quantiles=True)
    stratum = aggregates["stratum"].to_numpy()
    total = aggregates["samples"].to_numpy()
    assert total[stratum == layouts.ALL_ROWS].sum() == table.num_rows
    assert total[stratum != layouts.ALL_ROWS].sum() == table.num_rows
    error = np.abs(table["prediction"].to_numpy().astype(float) - table["target"].to_numpy())
    absolute = aggregates["absolute_error"].to_numpy()
    assert np.isclose(absolute[stratum == layouts.ALL_ROWS].sum(), error.sum())
    directions = sum(
        aggregates[name].to_numpy() for name in ("up_rise", "up_fall", "down_rise", "down_fall")
    )
    assert np.array_equal(directions, total)
    assert aggregates["rank_ic"].null_count == int((stratum != layouts.ALL_ROWS).sum())
    point = layouts.session_aggregates(table, rng, quantiles=False)
    assert "inside_80" not in point.column_names and "pinball_0500" not in point.column_names


def test_index_events_mark_one_sample_and_one_mature_label_per_row():
    table = keys()
    events = layouts.index_events(table)
    assert events.num_rows == 2 * table.num_rows
    kind = events["kind"].to_numpy()
    sample, label = events.filter(pa.array(kind == 0)), events.filter(pa.array(kind == 1))
    assert np.array_equal(
        label["event_at"].to_numpy() - label["sample_at"].to_numpy(),
        np.full(table.num_rows, 86_400_000_000),
    )
    assert np.all(label["source_group"].to_numpy() == -1)
    assert np.array_equal(sample["event_at"].to_numpy(), sample["sample_at"].to_numpy())


def test_adapter_outputs_keep_what_the_shared_table_cannot_rebuild(tmp_path):
    rng = np.random.default_rng(7)
    base = keys()
    table = layouts.adapter_table(base, rng)
    own = layouts.adapter_outputs(table)
    assert own.column_names == [*QUANTILE_COLUMNS, "parent"]
    assert own.schema.field("parent").type == pa.float32()
    layouts.write_large(own, tmp_path / "a.parquet")
    read = pq.read_table(tmp_path / "a.parquet")
    median = read["quantile_0500"]
    rebuilt = {name: base[name] for name in (*layouts.KEY_COLUMNS, "target")}
    rebuilt.update(
        prediction=median,
        parent=read["parent"].cast(pa.float64()),
        zero=pa.array(np.zeros(base.num_rows)),
        center=median,
    )
    rebuilt.update({name: read[name] for name in QUANTILE_COLUMNS})
    assert layouts.same_table(pa.table(rebuilt), table)
    center = table["center"].to_numpy().copy()
    center[2] = np.nextafter(center[2], 1)
    moved = table.set_column(table.column_names.index("center"), "center", pa.array(center))
    with pytest.raises(ValueError, match="center"):
        layouts.adapter_outputs(moved)
    parent = table["parent"].to_numpy().copy()
    parent[0] = 0.1
    wide = table.set_column(table.column_names.index("parent"), "parent", pa.array(parent))
    with pytest.raises(ValueError, match="float32"):
        layouts.adapter_outputs(wide)
    zero = np.zeros(base.num_rows)
    zero[1] = -0.0
    signed = table.set_column(table.column_names.index("zero"), "zero", pa.array(zero))
    with pytest.raises(ValueError, match="cero positivo"):
        layouts.adapter_outputs(signed)
