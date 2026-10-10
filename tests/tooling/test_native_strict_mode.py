"""Comprueba el modo estricto del enlace nativo en la comprobación local con una prueba aparte.

Cada caso ejecuta pytest en un directorio temporal con la configuración común como
complemento. `MARS_TITAN_REQUIRE_NATIVE=1` detiene la sesión antes de recoger pruebas si falta
alguno de los cinco binarios o si el enlace episódico no carga. Con el enlace cargado, una
prueba marcada con `native_binding` que se omite cuenta como fallo.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.suite_support import NATIVE_VARIABLES

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


def others(tmp_path):
    """Declara los otros cuatro binarios con archivos vacíos, porque solo se exige que existan."""
    declared = {}
    for variable in NATIVE_VARIABLES:
        if variable != NATIVE:
            path = tmp_path / variable.lower()
            path.write_text("")
            declared[variable] = str(path)
    return declared


def probe(tmp_path, **env):
    (tmp_path / "test_probe.py").write_text(PROBE, encoding="utf-8")
    removed = {*NATIVE_VARIABLES, STRICT}
    environment = {key: value for key, value in os.environ.items() if key not in removed}
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


def output(result):
    return result.stdout + result.stderr


def test_strict_mode_needs_a_loadable_binding_before_collecting(tmp_path):
    result = probe(tmp_path, **others(tmp_path), **{STRICT: "1"})
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output(result)
    assert f"falta declarar {NATIVE}" in output(result)
    absent = str(tmp_path / "absent.so")
    result = probe(tmp_path, **others(tmp_path), **{STRICT: "1", NATIVE: absent})
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output(result)
    assert f"{NATIVE} no apunta a un archivo" in output(result)
    # Un archivo que existe pero no es un enlace válido también detiene la sesión.
    empty = tmp_path / "empty.so"
    empty.write_bytes(b"")
    result = probe(tmp_path, **others(tmp_path), **{STRICT: "1", NATIVE: str(empty)})
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output(result)
    assert "no carga como enlace válido" in output(result)
    assert "passed" not in result.stdout
    result = probe(tmp_path, **{STRICT: "yes"})
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output(result)
    assert f"{STRICT} debe valer 0 o 1" in output(result)


@pytest.mark.skipif(not os.environ.get(NATIVE), reason="Falta el enlace nativo compilado")
def test_strict_mode_turns_a_skipped_native_test_into_a_failure(tmp_path):
    result = probe(tmp_path, **others(tmp_path), **{STRICT: "1", NATIVE: os.environ[NATIVE]})
    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout + result.stderr
    assert "1 failed, 1 passed, 1 skipped" in result.stdout
    assert "se omitió: Falta el enlace nativo CPU compilado" in result.stdout
