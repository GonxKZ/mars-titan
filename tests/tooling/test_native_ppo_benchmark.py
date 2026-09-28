"""Protección de artefactos y diagnóstico opcional del medidor PPO."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/benchmark_native_ppo.py"


@pytest.mark.parametrize("destination", ["existing", "inside_work"])
def test_output_protection_precedes_source_creation(tmp_path, destination):
    work = tmp_path / "work"
    output = (
        tmp_path / "measurement.json" if destination == "existing" else work / "train/manifest.json"
    )
    if destination == "existing":
        output.write_text("resultado anterior")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(ROOT),
            "--binary",
            str(tmp_path / "missing-binary"),
            "--private",
            str(work),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "La salida" in result.stderr
    assert not work.exists()
    if destination == "existing":
        assert output.read_text() == "resultado anterior"


def test_missing_perf_is_reported_without_blocking_the_measurement(tmp_path, monkeypatch):
    instrument = runpy.run_path(str(SCRIPT))
    monkeypatch.setattr(instrument["shutil"], "which", lambda _name: None)
    log = tmp_path / "perf.log"
    result = instrument["perf_probe"](log)
    assert result["status"] == "unavailable"
    assert result["returncode"] is None
    assert log.read_text().strip() == result["reason"]
