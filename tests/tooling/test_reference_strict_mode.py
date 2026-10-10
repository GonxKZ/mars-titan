"""Comprueba el modo estricto de las paridades con bibliotecas externas del grupo `reference`.

Igual que `test_native_strict_mode`, cada caso ejecuta pytest en un directorio temporal con
la configuración común como complemento. `MARS_TITAN_REQUIRE_REFERENCE=1` detiene la sesión
antes de recoger pruebas si falta el grupo o tiene otras versiones. Con el grupo instalado,
una prueba marcada con `external_reference` que se omite cuenta como fallo.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.suite_support import reference_problems

ROOT = Path(__file__).parents[2]
STRICT = "MARS_TITAN_REQUIRE_REFERENCE"
PROBE = """
import pytest


@pytest.mark.external_reference
def test_needs_the_group():
    pytest.skip("Falta scoringrules del grupo de dependencias reference")


def test_without_marker():
    pytest.skip("Otro motivo")


@pytest.mark.external_reference
def test_runs():
    pass
"""


def probe(tmp_path, **env):
    (tmp_path / "test_probe.py").write_text(PROBE, encoding="utf-8")
    environment = {key: value for key, value in os.environ.items() if key != STRICT}
    environment.update(env, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]))
    command = [sys.executable, "-m", "pytest", "-p", "tests.conftest", "-q", "-rs"]
    command += ["-c", os.devnull, "--rootdir", str(tmp_path), "-p", "no:cacheprovider"]
    command += ["-W", "ignore::pytest.PytestUnknownMarkWarning", "test_probe.py"]
    return subprocess.run(
        command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=600
    )


def test_general_suite_keeps_skipping_without_the_strict_mode(tmp_path):
    for value in ({}, {STRICT: "0"}):
        result = probe(tmp_path, **value)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 passed, 2 skipped" in result.stdout


def test_strict_mode_fails_a_skipped_reference_test_or_stops_without_the_group(tmp_path):
    """Con el grupo, la omisión marcada falla. Sin él, la sesión se detiene al empezar."""
    result = probe(tmp_path, **{STRICT: "1"})
    output = result.stdout + result.stderr
    problems = reference_problems()
    if problems:
        assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
        assert all(problem in output for problem in problems)
        assert "passed" not in result.stdout
    else:
        assert result.returncode == pytest.ExitCode.TESTS_FAILED, output
        assert "1 failed, 1 passed, 1 skipped" in result.stdout
        assert "prueba de referencia externa se omitió: Falta scoringrules" in result.stdout
