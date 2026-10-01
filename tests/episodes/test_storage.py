"""Almacenamiento recuperable y lectura selectiva de episodios."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.episodes.storage import EpisodeSource, write_world
from mars_titan.episodes.worlds import WorldConfig, generate_world


def test_parquet_roundtrip_is_selective_and_preserves_modalities(tmp_path):
    world = generate_world(WorldConfig(assets=4, sessions=40, context=8))
    report = write_world(world, tmp_path / "world")
    assert report["status"] == "completed"
    with EpisodeSource(tmp_path / "world/manifest.json") as source:
        raw = source(2)
        for key in world.shapes:
            np.testing.assert_array_equal(raw["inputs"][key], world(2)["inputs"][key])
        np.testing.assert_array_equal(raw["target"], world(2)["target"])
        assert source.read(2, modalities=("macro",))["inputs"].keys() == {"macro"}
        assert source.cache_bytes <= source.max_cache_bytes
        assert len(source) == len(world)


def test_interrupted_export_resumes_without_replacing_committed_blocks(tmp_path):
    world = generate_world(WorldConfig(assets=4, sessions=80, context=8))
    count = 0

    def stop():
        nonlocal count
        count += 1
        return count == 2

    report = write_world(world, tmp_path / "world", stop=stop)
    assert report["status"] == "paused"
    block = tmp_path / "world" / report["blocks"][0]["path"]
    signature = block.stat().st_mtime_ns
    resumed = write_world(world, tmp_path / "world", resume=True)
    assert resumed["status"] == "completed"
    assert block.stat().st_mtime_ns == signature
    with EpisodeSource(tmp_path / "world/manifest.json") as source:
        np.testing.assert_array_equal(
            source(len(source) - 1)["target"], world(len(world) - 1)["target"]
        )


def test_corrupt_blocks_or_changed_generator_cannot_resume(tmp_path):
    world = generate_world(WorldConfig(assets=4, sessions=40, context=8))
    report = write_world(world, tmp_path / "world")
    with pytest.raises(ValueError):
        write_world(generate_world(WorldConfig(seed=44)), tmp_path / "world", resume=True)
    block = tmp_path / "world" / report["blocks"][0]["path"]
    block.write_bytes(b"corrupto")
    with EpisodeSource(tmp_path / "world/manifest.json") as source:
        with pytest.raises(ValueError):
            source(0)


def test_resume_rejects_forged_cursor_before_writing_more_blocks(tmp_path):
    world = generate_world(WorldConfig(assets=4, sessions=40, context=8))
    write_world(world, tmp_path / "world")
    path = tmp_path / "world/progress.json"
    progress = json.loads(path.read_text())
    progress["blocks"][0]["cohorts"] += 1
    path.write_text(json.dumps(progress))
    with pytest.raises(ValueError):
        write_world(world, tmp_path / "world", resume=True)


def test_resource_budget_is_checked_during_export(tmp_path):
    world = generate_world(WorldConfig(assets=4, sessions=40, context=8))
    calls = []

    def check():
        calls.append(1)
        if len(calls) > 2:
            raise MemoryError("Presupuesto de prueba")

    with pytest.raises(MemoryError):
        write_world(world, tmp_path / "world", check_resources=check)
    assert len(calls) == 3


def test_changed_transform_version_rejects_recovery(tmp_path, monkeypatch):
    from mars_titan.episodes import storage

    world = generate_world(WorldConfig(assets=4, sessions=40, context=8))
    write_world(world, tmp_path / "world")
    monkeypatch.setattr(storage, "recipe_fingerprints", lambda: {"samples.py": "changed"})
    with pytest.raises(ValueError):
        write_world(world, tmp_path / "world", resume=True)


def test_variable_universe_roundtrips_and_recovers(tmp_path):
    world = generate_world(
        WorldConfig(
            assets=4, sessions=40, context=8, active_assets=tuple(3 + i % 2 for i in range(40))
        )
    )
    write_world(world, tmp_path / "world")
    write_world(world, tmp_path / "world", resume=True)
    with EpisodeSource(tmp_path / "world/manifest.json") as source:
        assert [len(source(i)["asset_ids"]) for i in range(len(source))] == [
            row[1] for row in world.index
        ]


def test_reader_reuses_one_open_block_and_closes_on_switch_and_exit(tmp_path, monkeypatch):
    world = generate_world(WorldConfig(assets=4, sessions=48, context=8))
    write_world(world, tmp_path / "world")
    opened, real_open = [], pq.ParquetFile

    def record_open(*args, **kwargs):
        reader = real_open(*args, **kwargs)
        opened.append(reader)
        return reader

    monkeypatch.setattr(pq, "ParquetFile", record_open)
    with EpisodeSource(tmp_path / "world/manifest.json") as source:
        retained = source(0)
        for position in (0, 1, 17, 18, 0):
            actual = source(position)
            for name in world.shapes:
                np.testing.assert_array_equal(
                    actual["inputs"][name], world(position)["inputs"][name]
                )
            assert sum(not reader.closed for reader in opened) == 1
            assert len(source.cache) <= 2
            assert source.cache_bytes <= source.max_cache_bytes
        assert len(opened) == 3
        assert all(reader.closed for reader in opened[:-1])
    assert all(reader.closed for reader in opened)
    for name in world.shapes:
        np.testing.assert_array_equal(retained["inputs"][name], world(0)["inputs"][name])
    assert source.cache_bytes == 0
    source.close()
    with pytest.raises(ValueError, match="abierta"):
        source(0)


@pytest.mark.parametrize("change", ["overwrite", "replace", "manifest"])
def test_open_reader_checks_integrity_before_returning_cached_cohort(tmp_path, change):
    world = generate_world(WorldConfig(assets=4, sessions=40, context=8))
    report = write_world(world, tmp_path / "world")
    manifest = tmp_path / "world/manifest.json"
    block = manifest.parent / report["blocks"][0]["path"]
    with EpisodeSource(manifest) as source:
        source(0)
        if change == "overwrite":
            with block.open("r+b") as handle:
                handle.write(b"FAIL")
        elif change == "replace":
            replacement = block.with_suffix(".new")
            replacement.write_bytes(block.read_bytes())
            replacement.replace(block)
        else:
            manifest.write_text(manifest.read_text() + " ")
        with pytest.raises(ValueError, match="cambió"):
            source(0)


def test_read_budget_failure_closes_open_reader_on_context_exit(tmp_path, monkeypatch):
    world = generate_world(WorldConfig(assets=4, sessions=40, context=8))
    write_world(world, tmp_path / "world")
    opened, real_open = [], pq.ParquetFile

    def record_open(*args, **kwargs):
        reader = real_open(*args, **kwargs)
        opened.append(reader)
        return reader

    monkeypatch.setattr(pq, "ParquetFile", record_open)
    with pytest.raises(ValueError, match="presupuesto"):
        with EpisodeSource(tmp_path / "world/manifest.json", max_cache_bytes=1) as source:
            source(0)
    assert opened and all(reader.closed for reader in opened)
    assert source.cache_bytes == 0
