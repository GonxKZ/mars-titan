"""Proteger inmutabilidad, recuperación y procedencia del inventario."""

import hashlib
import importlib
import json
import os
import subprocess
import sys

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
    with pytest.raises(ValueError, match="source"):
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
