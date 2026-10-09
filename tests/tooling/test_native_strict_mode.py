"""Comprueba el modo estricto del enlace nativo en la comprobación local con una prueba aparte.

Cada caso ejecuta pytest en un directorio temporal con la configuración común como
complemento. Una prueba marcada con `native_binding` que se omite cuenta como fallo solo
con `MARS_TITAN_REQUIRE_NATIVE=1`, y ese modo exige cargar el enlace al empezar.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
NATIVE = "MARS_TITAN_EPISODIC_NATIVE"
STRICT = "MARS_TITAN_REQUIRE_NATIVE"
PROBE = """
import pytest


@pytest.mark.native_binding
def test_needs_the_binding():
    pytest.skip("Falta el enlace nativo CPU compilado")


def test_without_marker():
    pytest.skip("Otro motivo")


@pytest.mark.native_binding
def test_runs():
    pass
"""


def probe(tmp_path, **env):
    (tmp_path / "test_probe.py").write_text(PROBE, encoding="utf-8")
    environment = {key: value for key, value in os.environ.items() if key not in (NATIVE, STRICT)}
    environment.update(env, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]))
    command = [sys.executable, "-m", "pytest", "-p", "tests.conftest", "-q", "-rs"]
    command += ["-c", os.devnull, "--rootdir", str(tmp_path), "-p", "no:cacheprovider"]
    command += ["-W", "ignore::pytest.PytestUnknownMarkWarning", "test_probe.py"]
    return subprocess.run(
        command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=600
    )


def test_cpu_suite_keeps_skipping_without_the_strict_mode(tmp_path):
    for value in ({}, {STRICT: "0"}):
        result = probe(tmp_path, **value)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 passed, 2 skipped" in result.stdout


def test_strict_mode_needs_a_loadable_binding_before_collecting(tmp_path):
    result = probe(tmp_path, **{STRICT: "1"})
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "exige un enlace válido" in result.stderr
    result = probe(tmp_path, **{STRICT: "1", NATIVE: str(tmp_path / "absent.so")})
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    result = probe(tmp_path, **{STRICT: "yes"})
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "solo admite 0 o 1" in result.stderr


@pytest.mark.skipif(not os.environ.get(NATIVE), reason="Falta el enlace nativo compilado")
def test_strict_mode_turns_a_skipped_native_test_into_a_failure(tmp_path):
    result = probe(tmp_path, **{STRICT: "1", NATIVE: os.environ[NATIVE]})
    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout + result.stderr
    assert "1 failed, 1 passed, 1 skipped" in result.stdout
    assert "se omitió: Falta el enlace nativo CPU compilado" in result.stdout
