"""Proteger inmutabilidad, recuperación y procedencia del inventario."""

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def inventory_module():
    try:
        return importlib.import_module("mars_titan.data.inventory")
    except ModuleNotFoundError:
        pytest.fail("El inventario reanudable todavía no está implementado")


def test_inventory_hashes_every_file_and_resumes_without_duplicates(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "a.csv").write_text("a,b\n1,2\n")
    (source / ".hidden").write_bytes(b"same")
    (source / "duplicate").write_bytes(b"same")
    db = tmp_path / "inventory.sqlite"
    module = inventory_module()
    first = module.inventory(source, db)
    second = module.inventory(source, db)
    assert first["files"] == second["files"] == 3
    assert first["code_sha256"] == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    assert second["database_sha256"] == hashlib.sha256(db.read_bytes()).hexdigest()
    assert first["hashed_files"] == 3
    assert second["hashed_files"] == 0
    assert first["duplicate_content_files"] == 1
    entries = list(module.entries(db))
    assert len(entries) == 3
    assert {x["state"] for x in entries} == {"inspected"}
    for x in entries:
        assert x["sha256"] == hashlib.sha256((source / x["path"]).read_bytes()).hexdigest()


def test_changed_and_removed_sources_are_not_silently_reused(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    path = source / "a.csv"
    path.write_text("a\n1\n")
    db = tmp_path / "inventory.sqlite"
    module = inventory_module()
    module.inventory(source, db)
    path.write_text("a\n123\n")
    changed = module.inventory(source, db)
    assert changed["hashed_files"] == 1
    path.unlink()
    assert module.inventory(source, db)["files"] == 0


def test_inventory_refuses_to_write_inside_originals(tmp_path):
    module = inventory_module()
    with pytest.raises(ValueError, match="origen"):
        module.inventory(tmp_path, tmp_path / "manifest.sqlite")


def test_inventory_does_not_follow_links_outside_source(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    private = tmp_path / "private"
    private.write_text("not dataset")
    (source / "linked.csv").symlink_to(private)
    result = inventory_module().inventory(source, tmp_path / "manifest.sqlite")
    assert result["files"] == 0
    assert result["errors"] == 0
    assert result["hashed_files"] == 0


def test_inventory_cli_outputs_inspected_not_validated(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "record.csv").write_text("date,close\n2024-01-02,1\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mars_titan.data.cli",
            "inventory",
            "--source",
            str(source),
            "--database",
            str(tmp_path / "source.sqlite"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["files"] == 1


def test_inventory_supports_original_filenames_with_non_utf8_bytes(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    name = os.fsdecode(b"news_\xff.jsonl")
    (source / name).write_bytes(b"{}\n")
    db = tmp_path / "source.sqlite"
    module = inventory_module()
    assert module.inventory(source, db)["files"] == 1
    assert next(module.entries(db))["path"] == name
    assert module.inventory(source, db)["hashed_files"] == 0


def test_bounded_inspection_does_not_mistake_split_utf8_for_another_encoding(tmp_path):
    path = tmp_path / "news.json"
    path.write_bytes(b'{"t":"' + b"a" * 65529 + 'é"}'.encode())
    assert inventory_module().inspect_header(path)["encoding"] == "utf-8-sig"


def test_snapshot_digest_depends_on_content_not_inspection_time(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    path = source / "a.csv"
    path.write_text("a\n1\n")
    module = inventory_module()
    one = module.inventory(source, tmp_path / "inventory.sqlite")
    os.utime(path, None)
    two = module.inventory(source, tmp_path / "inventory.sqlite", verify=True)
    assert one["snapshot_sha256"] == two["snapshot_sha256"]
    path.write_text("a\n2\n")
    three = module.inventory(source, tmp_path / "inventory.sqlite")
    assert three["snapshot_sha256"] != one["snapshot_sha256"]


@pytest.mark.parametrize(
    "relative,expected",
    [
        ("image/image/S&P500_image_a/aa/aa_2000_H1_candlestick.png", ("US", "charts", "AA")),
        ("image/S&P500_image/BRK-B/brk-b_2020_H2_candlestick.PNG", ("US", "charts", "BRK-B")),
        (
            "image/HS300_image/000422.SZ/000001.SZ/000001.SZ_2006_H1_candlestick.png",
            ("CN", "charts", "000001.SZ"),
        ),
        (
            "image/image/hs300_image/600000.SH/600000.SS_2020_H1_candlestick.png",
            ("CN", "charts", "600000.SS"),
        ),
    ],
)
def test_chart_identity_uses_matching_parent_and_filename(relative, expected):
    assert inventory_module().classify(Path(relative)) == expected


def test_image_directory_documentation_remains_metadata():
    assert inventory_module().classify(Path("image/image.md")) == ("catalog", "metadata", None)


@pytest.mark.parametrize(
    "relative",
    [
        "image/AA.png",
        "image/S&P500_image/AA/BB_2020_H1_candlestick.png",
        "image/S&P500_image/AA/AA_2020_H3_candlestick.png",
        "image/S&P500_image/AA/AA.png",
        "image/unknown/AA/AA_2020_H1_candlestick.png",
        "image/HS300_image/S&P500_image/AA/AA_2020_H1_candlestick.png",
    ],
)
def test_chart_identity_rejects_conflicts_and_unknown_formats(relative):
    with pytest.raises(ValueError):
        inventory_module().classify(Path(relative))


def test_inventory_records_invalid_chart_without_inventing_an_identity(tmp_path):
    from PIL import Image

    source = tmp_path / "dataset"
    path = source / "image/S&P500_image/AA/BB_2020_H1_candlestick.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (1, 1)).save(path)
    database = tmp_path / "inventory.sqlite"
    report = inventory_module().inventory(source, database)
    assert report["files"] == 1 and report["errors"] == 1
    entry = next(inventory_module().entries(database))
    assert entry["state"] == "error"
    assert entry["symbol"] is None


def test_cached_image_metadata_is_reclassified_without_reading_unchanged_content(tmp_path):
    import sqlite3

    from PIL import Image

    source = tmp_path / "dataset"
    path = source / "image/image/S&P500_image_a/aa/aa_2000_H1_candlestick.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (1, 1)).save(path)
    database = tmp_path / "inventory.sqlite"
    module = inventory_module()
    initial = module.inventory(source, database)
    with sqlite3.connect(database) as db:
        (raw,) = db.execute("SELECT record FROM files").fetchone()
        record = json.loads(raw)
        record["symbol"] = "S&P500_IMAGE_A"
        record.pop("classification_version", None)
        db.execute("UPDATE files SET record=?", (json.dumps(record),))
    result = module.inventory(source, database)
    assert result["hashed_files"] == 0
    assert next(module.entries(database))["symbol"] == "AA"
    assert result["snapshot_sha256"] == initial["snapshot_sha256"]


def test_rejected_reclassification_preserves_cached_content_evidence(tmp_path):
    import sqlite3

    from PIL import Image

    source = tmp_path / "dataset"
    path = source / "image/S&P500_image/AA/BB_2020_H1_candlestick.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (1, 1)).save(path)
    database = tmp_path / "inventory.sqlite"
    module = inventory_module()
    module.inventory(source, database)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    schema = {"format": "png", "width": 1, "height": 1}
    with sqlite3.connect(database) as db:
        (raw,) = db.execute("SELECT record FROM files").fetchone()
        record = json.loads(raw)
        record.update(
            state="inspected",
            sha256=digest,
            schema=schema,
            inspected_at="2020-01-01T00:00:00+00:00",
            market="US",
            symbol="AA",
        )
        for key in ("error", "classification_error", "classification_version"):
            record.pop(key, None)
        db.execute("UPDATE files SET record=?", (json.dumps(record),))
    first = module.inventory(source, database)
    entry = next(module.entries(database))
    assert first["errors"] == 1 and first["hashed_files"] == 0
    assert entry["sha256"] == digest
    assert entry["schema"] == schema
    assert entry["inspected_at"] == "2020-01-01T00:00:00+00:00"
    assert entry["state"] == "error" and entry["symbol"] is None
    assert first["duplicate_content_files"] == 0
    second = module.inventory(source, database)
    assert second["hashed_files"] == 0
    assert second["snapshot_sha256"] == first["snapshot_sha256"]
