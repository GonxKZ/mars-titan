"""Ejecutables falsos de la suite cuando el entorno vive en una ruta con espacios."""

import subprocess
import sys
from pathlib import Path

import pytest

from tests.suite_support import python_shebang


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
