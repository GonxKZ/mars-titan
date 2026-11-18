"""Correspondencia entre los eventos de origen y su copia de trabajo en Arrow."""

import json
import sys

import pytest

from benchmarks import comparison_input_archive as archive
from tests.training.test_financial_run import corpus


def test_archive_keeps_events_labels_and_blocks_across_replays(tmp_path):
    _, sources = corpus(tmp_path / "source")
    source = sources["train"]
    spec = source.specification()
    path = tmp_path / "events.arrow"
    count = len(source.metadata["groups"])
    stamp = archive.write(source, spec, path, start=0, count=count, block=4, max_bytes=8 << 20)
    for skip in (0, 1, count - 1):
        for width in (2, 4):
            measured = archive.fingerprint(
                archive.replay(source, spec, path, stamp, cursor=skip, block=width), spec
            )
            original = archive.fingerprint(
                source.batched_events(start_cursor=skip, block_rows=width), spec
            )
            assert measured == original
    with pytest.raises(ValueError, match="bloque"):
        list(archive.replay(source, spec, path, stamp, cursor=0, block=3))
    with pytest.raises(ValueError, match="otro origen"):
        list(archive.replay(sources["validation"], spec, path, stamp, cursor=0, block=4))
    with pytest.raises(FileExistsError):
        archive.write(source, spec, path, start=0, count=count, block=4, max_bytes=8 << 20)
    small = tmp_path / "too-small.arrow"
    with pytest.raises(ValueError, match="presupuesto de disco"):
        archive.write(source, spec, small, start=0, count=count, block=4, max_bytes=1)
    assert not small.exists() and path.exists()
    with path.open("r+b") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="huella"):
        list(archive.replay(source, spec, path, stamp, cursor=0, block=4))


def test_command_compares_four_passes_and_removes_its_working_copy(tmp_path, monkeypatch, capsys):
    data, sources = corpus(tmp_path / "source")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "comparison_input_archive",
            str(data.path),
            str(sources["train"].path),
            str(scratch),
            "--block-rows",
            "4",
        ],
    )
    archive.main()
    report = json.loads(capsys.readouterr().out)
    assert [p["mode"] for p in report["passes"]] == [
        "original",
        "preparada",
        "preparada",
        "original",
    ]
    assert len({p["data"]["digest"] for p in report["passes"]}) == 1
    assert report["passes"][0]["data"]["rows"] > 0
    assert list(scratch.iterdir()) == []
