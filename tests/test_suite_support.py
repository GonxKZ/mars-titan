"""Apoyo de la suite: comprobación estricta y ejecutables falsos con rutas con espacios."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.suite_support import NATIVE_VARIABLES, python_shebang, strict_problems

ROOT = Path(__file__).resolve().parents[1]


def never():
    return False


def declared_binaries(tmp_path):
    environment = {}
    for variable in NATIVE_VARIABLES:
        path = tmp_path / variable.lower()
        path.write_text("")
        environment[variable] = str(path)
    return environment


def test_the_general_suite_requires_neither_binaries_nor_cuda():
    assert strict_problems({}, cuda=never) == []
    assert strict_problems({"MARS_TITAN_REQUIRE_NATIVE": "0"}, cuda=never) == []


def test_strict_native_check_names_each_undeclared_or_missing_binary(tmp_path):
    environment = declared_binaries(tmp_path) | {"MARS_TITAN_REQUIRE_NATIVE": "1"}
    assert strict_problems(environment, cuda=never) == []
    del environment["MARS_TITAN_PPO_EXECUTABLE"]
    environment["MARS_TITAN_SIM_EXECUTABLE"] = str(tmp_path / "missing")
    assert strict_problems(environment, cuda=never) == [
        f"MARS_TITAN_SIM_EXECUTABLE no apunta a un archivo: {tmp_path / 'missing'}",
        "falta declarar MARS_TITAN_PPO_EXECUTABLE",
    ]
    environment["MARS_TITAN_SIM_EXECUTABLE"] = str(tmp_path)
    assert (
        "MARS_TITAN_SIM_EXECUTABLE no apunta a un archivo"
        in strict_problems(environment, cuda=never)[0]
    )


def test_strict_cuda_check_requires_a_visible_device():
    assert strict_problems({"MARS_TITAN_REQUIRE_CUDA": "1"}, cuda=lambda: True) == []
    assert strict_problems({"MARS_TITAN_REQUIRE_CUDA": "1"}, cuda=never) == [
        "CUDA no está disponible"
    ]
    assert strict_problems({"MARS_TITAN_REQUIRE_CUDA": "0"}, cuda=never) == []


@pytest.mark.parametrize("value", ["", "true", "2"])
def test_switches_accept_only_zero_or_one(value):
    assert strict_problems({"MARS_TITAN_REQUIRE_CUDA": value}, cuda=lambda: True) == [
        "MARS_TITAN_REQUIRE_CUDA debe valer 0 o 1"
    ]


def test_a_strict_session_stops_before_running_any_test(tmp_path):
    environment = dict(
        os.environ,
        MARS_TITAN_REQUIRE_NATIVE="1",
        MARS_TITAN_EPISODIC_NATIVE=str(tmp_path / "missing.so"),
    )
    trivial = test_the_general_suite_requires_neither_binaries_nor_cuda.__name__
    target = f"{Path(__file__).relative_to(ROOT)}::{trivial}"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", target],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, result.stdout + result.stderr
    assert "Comprobación estricta incumplida" in result.stdout + result.stderr
    assert "MARS_TITAN_EPISODIC_NATIVE no apunta a un archivo" in result.stdout + result.stderr
    assert "passed" not in result.stdout


def spaced_interpreter(tmp_path):
    """El intérprete actual visto desde una ruta con espacios, como el repositorio principal."""
    executable = Path(sys.executable)
    environment = tmp_path / "TFM TITANS" / ".venv"
    environment.parent.mkdir()
    environment.symlink_to(executable.parent.parent, target_is_directory=True)
    return environment / executable.parent.name / executable.name


def tool_on_path(tmp_path, shebang):
    """Ejecutar `tool` con un falso en Python delante de otro real en PATH."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for path, body in ((first / "tool", shebang + "print('falso')\n"), (second / "tool", None)):
        path.write_text(body or "#!/bin/sh\necho real\n")
        path.chmod(0o700)
    result = subprocess.run(
        ["tool"], env={"PATH": f"{first}:{second}"}, capture_output=True, text=True, timeout=30
    )
    return result.stdout.strip()


def test_spaced_interpreter_in_shebang_falls_through_to_the_next_tool_on_path(tmp_path):
    # Es el fallo observado con el entorno del repositorio principal: el falso no arranca y
    # se ejecuta en silencio el siguiente programa del mismo nombre, como el nvidia-smi real.
    assert tool_on_path(tmp_path, f"#!{spaced_interpreter(tmp_path)}\n") == "real"


def test_linked_shebang_runs_the_fake_tool_with_the_same_environment(tmp_path):
    links = tmp_path / "links"
    links.mkdir()
    shebang = python_shebang(links, spaced_interpreter(tmp_path))
    assert " " not in shebang.strip()
    assert tool_on_path(tmp_path, shebang) == "falso"
    program = tmp_path / "prefix"
    # pytest solo está en el entorno virtual, así que importarlo prueba que el enlace lo conserva.
    program.write_text(shebang + "import sys, pytest\nprint(sys.prefix)\n")
    program.chmod(0o700)
    result = subprocess.run([program], capture_output=True, text=True, timeout=30, check=True)
    assert Path(result.stdout.strip()).resolve() == Path(sys.prefix).resolve()


def test_link_directory_with_spaces_is_rejected(tmp_path):
    directory = tmp_path / "con espacios"
    directory.mkdir()
    with pytest.raises(ValueError, match="espacios"):
        python_shebang(directory)
