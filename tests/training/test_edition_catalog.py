"""Catálogo de la edición técnica: filas, alineación, test reservado y verificación."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import MODALITIES
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training import edition_catalog
from mars_titan.training.edition_catalog import (
    SCHEMA,
    build_catalog,
    read_catalog_manifest,
    verify_catalog,
)
from tests.training.edition_catalog_fixture import built_catalog, edition_fixture


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return built_catalog(tmp_path_factory.mktemp("edition-catalog"))


def rows(path):
    manifest, _, catalog = read_catalog_manifest(path)
    return manifest, pq.read_table(catalog)


def test_one_row_per_sample_with_its_labels_and_without_vectors(built):
    manifest, table = rows(built.catalog)
    assert table.schema == SCHEMA
    assert not {"news", "charts", "fundamentals", "macro", "target"} & set(table.column_names)
    assert table.num_rows == built.metadata["samples"] == manifest["catalog"]["rows"]
    roots = built.metadata["roots"]
    offset = 0
    for asset in built.metadata["assets"]:
        part = table.slice(offset, asset["samples"])
        offset += asset["samples"]
        folder = Path(asset["market"]) / asset["symbol"]
        samples = pq.read_table(Path(roots["samples"]) / folder / "samples.parquet")
        labels = pq.read_table(Path(roots["labels"]) / folder / "labels.parquet")
        assert set(part["market"].to_pylist()) == {asset["market"]}
        assert set(part["asset_id"].to_pylist()) == {asset["symbol"]}
        assert part["prediction_at"].to_pylist() == samples["prediction_at"].to_pylist()
        presence = np.array(samples["presence"].to_pylist())
        for index, name in enumerate(MODALITIES):
            assert part[f"has_{name}"].to_pylist() == presence[:, index].tolist()
            available = part[f"{name}_available_at"].to_pylist()
            for value, moment in zip(available, part["prediction_at"].to_pylist(), strict=True):
                assert value is None or value <= moment
        by_row = {row["sample_row"]: row for row in labels.to_pylist()}
        for position, row in enumerate(part.to_pylist()):
            label = by_row[position]
            assert row["sample_row"] == position
            assert row["label_reason"] == label["reason"]
            assert row["target_partition"] == label["partition"]
            assert row["target_available_at"] == label["target_available_at"]
            assert row["inputs_available_at"] <= row["prediction_at"]
            assert row["session"] == row["prediction_at"].date()
            assert row["annual_boundary_label"] == (
                label["reason"] == "target_crosses_partition_boundary"
            )
    assert offset == table.num_rows
    assert manifest["counts"]["target_partitions"] == built.metadata["counts"]
    assert manifest["counts"]["markets"] == {
        market: sum(a["samples"] for a in built.metadata["assets"] if a["market"] == market)
        for market in ("CN", "US")
    }
    assert manifest["target_value_included"] is False
    assert manifest["edition"]["sha256"] == sha256(built.edition)
    assert manifest["targets"]["sha256"] == sha256(built.parent)


def test_sample_positions_point_to_the_source_row_group(built):
    _, table = rows(built.catalog)
    root = Path(built.metadata["roots"]["samples"])
    for row in table.slice(0, 20).to_pylist():
        with pq.ParquetFile(root / row["samples_file"]) as file:
            group = file.read_row_group(row["row_group"], columns=["prediction_at"])
        assert group["prediction_at"][row["row_in_group"]].as_py() == row["prediction_at"]


def test_the_catalog_is_verified_and_rebuilt_bit_for_bit(built, tmp_path):
    report = verify_catalog(built.catalog)
    assert report["status"] == "verified" and report["rows"] == built.metadata["samples"]
    again = build_catalog(built.parent, built.edition, tmp_path / "again", row_group_rows=4)
    manifest = json.loads(built.catalog.read_text())
    assert again["catalog"]["row_groups"] > manifest["catalog"]["row_groups"]
    assert pq.read_table(tmp_path / "again" / "catalog.parquet").equals(rows(built.catalog)[1])
    once = build_catalog(built.parent, built.edition, tmp_path / "once")
    assert once["catalog"]["sha256"] == manifest["catalog"]["sha256"]


@pytest.mark.parametrize(
    "column, value",
    [
        ("label_reason", "accepted"),
        ("has_news", None),
        ("target_partition", "validation"),
        ("prediction_at", None),
        ("label_row", None),
    ],
)
def test_verification_detects_a_changed_row(built, tmp_path, column, value):
    folder = tmp_path / "catalog"
    folder.mkdir()
    manifest, table = rows(built.catalog)
    values = table[column].to_pylist()
    position = next(
        index for index, item in enumerate(values) if item is not None and item != value
    )
    if value is None:
        if column == "has_news":
            value = not values[position]
        elif column == "prediction_at":
            value = values[position + 1]
        else:
            value = values[position] + 1
    values[position] = value
    changed = table.set_column(
        table.schema.get_field_index(column),
        column,
        pa.array(values, table.schema.field(column).type),
    )
    pq.write_table(changed, folder / "catalog.parquet")
    manifest["catalog"]["sha256"] = sha256(folder / "catalog.parquet")
    atomic_json(folder / "manifest.json", manifest)
    with pytest.raises(ValueError):
        verify_catalog(folder / "manifest.json")


def test_a_changed_catalog_file_is_rejected_before_use(built, tmp_path):
    folder = tmp_path / "catalog"
    folder.mkdir()
    manifest, table = rows(built.catalog)
    pq.write_table(table.slice(1), folder / "catalog.parquet")
    atomic_json(folder / "manifest.json", manifest)
    with pytest.raises(ValueError, match="no corresponde"):
        read_catalog_manifest(folder / "manifest.json")


def test_rows_of_2024_stop_the_catalog_before_their_samples_are_read(tmp_path, monkeypatch):
    days = {"US": ("2022-12-15", "2023-03-15", "2023-12-28", "2024-01-03")}
    data = edition_fixture(tmp_path / "data", ("US",), assets=1, days=days)
    reads = []
    original = pq.ParquetFile.read
    monkeypatch.setattr(
        pq.ParquetFile, "read", lambda self, *a, **k: reads.append(1) or original(self, *a, **k)
    )
    with pytest.raises(ValueError, match="test reservado"):
        build_catalog(data.parent, data.edition, tmp_path / "catalog")
    assert not reads and not (tmp_path / "catalog").exists()


def test_inputs_include_the_price_row_that_closes_the_window(tmp_path, monkeypatch):
    # Con los bloques un segundo antes, la fila de precios que cierra la ventana es la última
    # entrada disponible de cada muestra, como en la lectura de las vistas.
    data = edition_fixture(tmp_path / "data", ("US",), assets=1)
    original = edition_catalog._availability

    def earlier(table, **kwargs):
        available, valid = original(table, **kwargs)
        return available - 1_000_000, valid

    monkeypatch.setattr(edition_catalog, "_availability", earlier)
    build_catalog(data.parent, data.edition, tmp_path / "catalog")
    _, table = rows(tmp_path / "catalog" / "manifest.json")
    asset = data.metadata["assets"][0]
    roots = data.metadata["roots"]
    folder = Path(asset["market"]) / asset["symbol"]
    ends = pq.read_table(Path(roots["samples"]) / folder / "samples.parquet")["price_end_index"]
    closing = pq.read_table(Path(roots["prepared"]) / folder / "prices.parquet")["available_at"]
    closing = closing.cast(pa.int64()).to_numpy()[ends.to_numpy()]
    blocks = np.stack(
        [
            table[f"{name}_available_at"].cast(pa.int64()).fill_null(0).to_numpy()
            for name in MODALITIES
        ]
    ).max(axis=0)
    assert (closing > blocks - 1_000_000).all()
    actual = table["inputs_available_at"].cast(pa.int64()).to_numpy()
    assert np.array_equal(actual, np.maximum(blocks - 1_000_000, closing))


def test_the_targets_must_come_from_the_declared_edition(tmp_path):
    data = edition_fixture(tmp_path / "data", ("US",), assets=1)
    edition = json.loads(data.edition.read_text())
    atomic_json(data.edition, dict(edition, samples=edition["samples"] + 1))
    with pytest.raises(ValueError, match="no proceden de esta edición"):
        build_catalog(data.parent, data.edition, tmp_path / "catalog")
    assert not (tmp_path / "catalog").exists()


def test_an_input_later_than_the_prediction_stops_the_catalog(tmp_path, monkeypatch):
    data = edition_fixture(tmp_path / "data", ("US",), assets=1)
    original = edition_catalog._availability

    def late(table, **kwargs):
        available, valid = original(table, **kwargs)
        return available + 1, valid

    monkeypatch.setattr(edition_catalog, "_availability", late)
    with pytest.raises(ValueError, match="futura"):
        build_catalog(data.parent, data.edition, tmp_path / "catalog")


def test_an_existing_output_is_never_replaced(built):
    with pytest.raises(FileExistsError):
        build_catalog(built.parent, built.edition, built.catalog.parent)
