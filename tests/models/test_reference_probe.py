"""Precondiciones de las sondas antes de tocar GPU o escribir resultados."""

import importlib

import pytest


def module():
    try:
        return importlib.import_module("mars_titan.reference_probe")
    except ModuleNotFoundError:
        pytest.fail("Falta el ejecutor de referencias sobre datos estrictos")


def test_reference_probe_refuses_outputs_inside_prepared_inputs(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    sentinel = prepared / "manifest.json"
    sentinel.write_text("conservar")
    with pytest.raises(ValueError, match="origen"):
        module().run_reference_probe(
            prepared, tmp_path / "samples", prepared / "run", tmp_path / "report.json"
        )
    assert sentinel.read_text() == "conservar"


def test_reference_probe_does_not_replace_an_existing_run(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(ValueError, match="nuevo"):
        module().run_reference_probe(
            tmp_path / "prepared", tmp_path / "samples", output, tmp_path / "report.json"
        )


@pytest.mark.parametrize(
    "artifact", ["ridge.npz", "predictions.parquet", "targets/A-targets.parquet"]
)
def test_reference_report_cannot_overwrite_run_artifacts(tmp_path, artifact):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="origen"):
        module().run_reference_probe(
            tmp_path / "prepared", tmp_path / "samples", output, output / artifact
        )
    assert not output.exists()
