"""Las rutas de salida se validan antes de CUDA, red o escritura."""

import sqlite3

import pytest


@pytest.mark.parametrize("target", ["report", "run"])
def test_training_rejects_writes_inside_originals(tmp_path, monkeypatch, target):
    pytest.importorskip("torch")
    from mars_titan import budget_training

    monkeypatch.chdir(tmp_path)
    original = tmp_path / "dataset"
    original.mkdir()
    sentinel = original / "README.md"
    sentinel.write_text("original")
    monkeypatch.setattr(
        budget_training, "require_cuda", lambda: pytest.fail("CUDA before path guard")
    )
    report = sentinel if target == "report" else tmp_path / "report.json"
    output = original / "new-run" if target == "run" else tmp_path / "run"
    with pytest.raises(ValueError, match="source"):
        budget_training.train_budget_grid(tmp_path / "prepared", report, output)
    assert sentinel.read_text() == "original"
    assert not output.exists()


@pytest.mark.parametrize("target", ["report", "run"])
def test_training_rejects_writes_inside_prepared_inputs(tmp_path, monkeypatch, target):
    pytest.importorskip("torch")
    from mars_titan import budget_training

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    sentinel = prepared / "manifest.json"
    sentinel.write_text("prepared input")
    monkeypatch.setattr(
        budget_training, "require_cuda", lambda: pytest.fail("CUDA before path guard")
    )
    report = sentinel if target == "report" else tmp_path / "report.json"
    output = prepared / "new-run" if target == "run" else tmp_path / "run"
    with pytest.raises(ValueError, match="source"):
        budget_training.train_budget_grid(prepared, report, output)
    assert sentinel.read_text() == "prepared input"
    assert not output.exists()


def test_macro_acquisition_rejects_destination_inside_originals(tmp_path, monkeypatch):
    from mars_titan.data import macro_acquisition

    monkeypatch.chdir(tmp_path)
    original = tmp_path / "dataset"
    original.mkdir()
    monkeypatch.setattr(
        macro_acquisition, "_initialize", lambda *a: pytest.fail("write before guard")
    )
    with pytest.raises(ValueError, match="source"):
        macro_acquisition.acquire_catalog(
            [],
            original / "macro",
            observation_start="2000-01-01",
            observation_end="2020-01-01",
            realtime_start="2000-01-01",
            realtime_end="2020-01-01",
        )
    assert not (original / "macro").exists()


def test_macro_acquisition_protects_original_database_through_symlink(tmp_path, monkeypatch):
    from mars_titan.data import macro_acquisition

    monkeypatch.chdir(tmp_path)
    original = tmp_path / "dataset/original.sqlite"
    original.parent.mkdir()
    with sqlite3.connect(original) as connection:
        connection.execute("CREATE TABLE original (value TEXT)")
        connection.execute("INSERT INTO original VALUES ('unchanged')")
    previous = original.read_bytes()
    output = tmp_path / "macro"
    output.mkdir()
    (output / "macro.sqlite3").symlink_to(original)
    with pytest.raises(ValueError, match="source"):
        macro_acquisition.acquire_catalog(
            [],
            output,
            observation_start="2000-01-01",
            observation_end="2020-01-01",
            realtime_start="2000-01-01",
            realtime_end="2020-01-01",
        )
    assert original.read_bytes() == previous


@pytest.mark.parametrize("link_path", ["raw", "raw/us_cpi"])
def test_macro_acquisition_rejects_archive_symlinks_before_writes_or_network(
    tmp_path, monkeypatch, link_path
):
    from mars_titan.data import macro_acquisition

    monkeypatch.chdir(tmp_path)
    original = tmp_path / "dataset"
    original.mkdir()
    output = tmp_path / "macro"
    link = output / link_path
    link.parent.mkdir(parents=True)
    link.symlink_to(original, target_is_directory=True)
    monkeypatch.setattr(
        macro_acquisition, "_request", lambda *args, **kwargs: pytest.fail("network before guard")
    )
    catalog = [
        {
            "id": "us_cpi",
            "kind": "raw",
            "provider": "BLS via FRED",
            "series_id": "CPIAUCSL",
            "source_url": "https://fred.stlouisfed.org/series/CPIAUCSL",
            "vintage_policy": "ALFRED_OR_RELEASE_ARCHIVE",
            "verification_status": "verified_metadata_not_ingested",
        }
    ]
    with pytest.raises(ValueError, match="source"):
        macro_acquisition.acquire_catalog(
            catalog,
            output,
            observation_start="2000-01-01",
            observation_end="2020-01-01",
            realtime_start="2000-01-01",
            realtime_end="2020-01-01",
        )
    assert not (output / "macro.sqlite3").exists()
