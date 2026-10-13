"""Apoyo de la suite: modos estrictos, grupo `reference` y ejecutables falsos con espacios."""

import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

from tests.suite_support import (
    NATIVE_VARIABLES,
    episodic_load_problem,
    native_required,
    python_shebang,
    reference_module,
    reference_pins,
    reference_problems,
    strict_problems,
    strict_skip,
)

ROOT = Path(__file__).resolve().parents[1]


def never():
    return False


def loads(path):
    """Sustituto del cargador del enlace episódico que siempre lo acepta."""
    return None


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
    assert strict_problems(environment, cuda=never, episodic=loads) == []
    del environment["MARS_TITAN_PPO_EXECUTABLE"]
    environment["MARS_TITAN_SIM_EXECUTABLE"] = str(tmp_path / "missing")
    assert strict_problems(environment, cuda=never, episodic=loads) == [
        f"MARS_TITAN_SIM_EXECUTABLE no apunta a un archivo: {tmp_path / 'missing'}",
        "falta declarar MARS_TITAN_PPO_EXECUTABLE",
    ]
    environment["MARS_TITAN_SIM_EXECUTABLE"] = str(tmp_path)
    assert (
        "MARS_TITAN_SIM_EXECUTABLE no apunta a un archivo"
        in strict_problems(environment, cuda=never, episodic=loads)[0]
    )


def test_strict_native_check_loads_the_declared_episodic_binding(tmp_path):
    environment = declared_binaries(tmp_path) | {"MARS_TITAN_REQUIRE_NATIVE": "1"}
    calls = []

    def refuses(path):
        calls.append(path)
        return "MARS_TITAN_EPISODIC_NATIVE no carga como enlace válido: otro PyTorch"

    assert strict_problems(environment, cuda=never, episodic=refuses) == [
        "MARS_TITAN_EPISODIC_NATIVE no carga como enlace válido: otro PyTorch"
    ]
    assert calls == [environment["MARS_TITAN_EPISODIC_NATIVE"]]
    # Sin el modo estricto, o sin un archivo que cargar, el enlace no se intenta cargar.
    assert strict_problems(dict(environment, MARS_TITAN_REQUIRE_NATIVE="0"), episodic=refuses) == []
    environment["MARS_TITAN_EPISODIC_NATIVE"] = str(tmp_path / "missing.so")
    assert len(strict_problems(environment, cuda=never, episodic=refuses)) == 1
    assert len(calls) == 1


def test_the_real_loader_rejects_a_file_that_is_not_a_binding(tmp_path):
    empty = tmp_path / "empty.so"
    empty.write_bytes(b"")
    problem = episodic_load_problem(str(empty))
    assert problem.startswith("MARS_TITAN_EPISODIC_NATIVE no carga como enlace válido")


def test_native_required_reads_only_the_validated_switch():
    assert native_required({"MARS_TITAN_REQUIRE_NATIVE": "1"})
    assert not native_required({"MARS_TITAN_REQUIRE_NATIVE": "0"})
    assert not native_required({})


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


def test_strict_skip_names_the_switch_of_each_marker():
    native = {"MARS_TITAN_REQUIRE_NATIVE": "1"}
    reference = {"MARS_TITAN_REQUIRE_REFERENCE": "1"}
    assert strict_skip({"native_binding"}, native) == (
        "MARS_TITAN_REQUIRE_NATIVE=1 y la prueba del enlace nativo se omitió"
    )
    assert strict_skip({"external_reference"}, reference) == (
        "MARS_TITAN_REQUIRE_REFERENCE=1 y la prueba de referencia externa se omitió"
    )
    # Cada marca solo depende de su propia variable.
    assert strict_skip({"external_reference"}, native) is None
    assert strict_skip({"native_binding"}, reference) is None
    assert strict_skip({"parametrize"}, native | reference) is None
    assert strict_skip({"external_reference"}, {"MARS_TITAN_REQUIRE_REFERENCE": "0"}) is None
    # Una prueba con las dos marcas solo falla al omitirse si se exigen las dos cosas.
    both = {"native_binding", "external_reference"}
    assert strict_skip(both, reference) is None
    assert strict_skip(both, native) is None
    assert strict_skip(both, native | reference) == (
        "MARS_TITAN_REQUIRE_NATIVE=1, MARS_TITAN_REQUIRE_REFERENCE=1 y la prueba del enlace "
        "nativo y de referencia externa se omitió"
    )


def test_reference_pins_are_the_exact_versions_of_the_group():
    pins = reference_pins()
    assert pins == {
        "peft": "0.21.0",
        "accelerate": "1.15.0",
        "scoringrules": "0.11.0",
        "mapie": "1.5.0",
        "sb3-contrib": "2.9.0",
        "stable-baselines3": "2.9.0",
    }


@pytest.mark.parametrize(
    "requirement", ["peft>=0.21", "peft", "peft== 0.21.0", "peft==0.21.0 ; os_name == 'nt'"]
)
def test_reference_pins_reject_inexact_requirements(tmp_path, requirement):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f'[dependency-groups]\nreference = ["{requirement}"]\n')
    with pytest.raises(ValueError, match="versiones exactas"):
        reference_pins(pyproject)


def test_reference_problems_name_missing_and_different_versions():
    def installed(name):
        if name == "absent":
            raise metadata.PackageNotFoundError(name)
        return {"exact": "1.0.0", "other": "2.1.0"}[name]

    pins = {"exact": "1.0.0", "other": "2.0.0", "absent": "3.0.0"}
    assert reference_problems(pins, installed) == [
        "other 2.1.0 instalado, el grupo reference fija 2.0.0",
        "falta absent==3.0.0 del grupo reference",
    ]


def test_strict_reference_check_runs_only_with_its_switch():
    def problems():
        return ["falta peft==0.21.0 del grupo reference"]

    assert strict_problems({}, cuda=never, reference=problems) == []
    assert strict_problems({"MARS_TITAN_REQUIRE_REFERENCE": "0"}, reference=problems) == []
    assert strict_problems({"MARS_TITAN_REQUIRE_REFERENCE": "1"}, reference=problems) == problems()
    assert strict_problems({"MARS_TITAN_REQUIRE_REFERENCE": "yes"}, reference=problems) == [
        "MARS_TITAN_REQUIRE_REFERENCE debe valer 0 o 1"
    ]


def test_reference_module_skips_only_when_the_package_itself_is_missing(tmp_path, monkeypatch):
    with pytest.raises(pytest.skip.Exception, match="grupo de dependencias reference"):
        reference_module("mars_titan_reference_absent")
    # Un paquete presente cuya dependencia falta es una instalación rota y no se omite.
    package = tmp_path / "mars_titan_reference_broken"
    package.mkdir()
    (package / "__init__.py").write_text("import mars_titan_reference_dependency_absent\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="mars_titan_reference_dependency_absent"):
        reference_module("mars_titan_reference_broken")
