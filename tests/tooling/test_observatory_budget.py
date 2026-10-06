"""Presupuestos del recolector y conservación de la última publicación válida."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def collect(tmp_path, *options):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.collect_observatory",
            "--root",
            str(tmp_path / "root"),
            "--config",
            str(tmp_path / "config.json"),
            "--state-dir",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "public"),
            *options,
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_explicit_limits_admit_sources_and_keep_publication_on_exhaustion(tmp_path):
    sources = []
    for name in ("first", "second"):
        folder = tmp_path / "root" / name
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps({"runs": [], "status": "completed"}))
        sources.append(dict(id=name, path=name, kind="archive", domain="technical"))
    (tmp_path / "config.json").write_text(json.dumps(dict(schema_version=1, sources=sources)))
    initial = collect(tmp_path, "--max-files", "2", "--max-bytes", "4096")
    assert initial.returncode == 0, initial.stderr
    published = (tmp_path / "public/observatory.json").read_bytes()
    assert len(json.loads(published)["campaigns"]) == 2
    for option, value in (("--max-files", "1"), ("--max-bytes", "1")):
        limited = collect(tmp_path, option, value)
        assert limited.returncode == 1
        assert (tmp_path / "public/observatory.json").read_bytes() == published


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--max-files", "0"),
        ("--max-files", "65537"),
        ("--max-bytes", "0"),
        ("--max-bytes", "1073741825"),
    ],
)
def test_invalid_limits_fail_before_opening_sources(tmp_path, option, value):
    result = collect(tmp_path, option, value)
    assert result.returncode == 2
    assert "Presupuesto del recolector fuera de los límites admitidos" in result.stderr
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("source_count, accepted", [(84, True), (129, False)])
def test_expanded_campaign_keeps_history_with_a_bounded_number_of_sources(
    tmp_path, source_count, accepted
):
    (tmp_path / "root").mkdir()
    sources = [
        dict(id=f"campaign-{i:03}", path=f"campaign-{i:03}", kind="archive", domain="real")
        for i in range(source_count)
    ]
    (tmp_path / "config.json").write_text(json.dumps(dict(schema_version=1, sources=sources)))
    result = collect(tmp_path)
    if accepted:
        assert result.returncode == 0, result.stderr
        output = json.loads((tmp_path / "public/observatory.json").read_text())
        assert len(output["campaigns"]) == source_count
        assert len({row["id"] for row in output["campaigns"]}) == source_count
    else:
        assert result.returncode == 2
        assert not (tmp_path / "cache").exists()
