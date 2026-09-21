"""Comprobaciones reales del contrato CMake con programas C++ temporales."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

NATIVE = Path(__file__).resolve().parents[2] / "native"


def configure(tmp_path, source, *options):
    if not shutil.which("cmake") or not shutil.which("clang++"):
        pytest.skip("La comprobación nativa requiere CMake y Clang")
    project = tmp_path / "fixture"
    project.mkdir()
    (project / "probe.cpp").write_text(source)
    (project / "CMakeLists.txt").write_text(
        f"cmake_minimum_required(VERSION 3.24)\nproject(diagnostic_fixture LANGUAGES CXX)\n"
        f'add_subdirectory("{NATIVE}" native)\nadd_executable(probe probe.cpp)\n'
        "mars_titan_configure_target(probe)\n"
    )
    build = tmp_path / "build"
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(project),
            "-B",
            str(build),
            "-DCMAKE_CXX_COMPILER=clang++",
            "-DCMAKE_BUILD_TYPE=Debug",
            *options,
        ],
        capture_output=True,
        text=True,
    )
    return build, result


def compile_target(build):
    return subprocess.run(
        ["cmake", "--build", str(build), "--parallel", "2"], capture_output=True, text=True
    )


def test_correct_cpp_target_compiles_with_strict_options(tmp_path):
    build, result = configure(
        tmp_path,
        "#include <array>\n#ifndef __STRICT_ANSI__\n"
        "#error Se requieren extensiones desactivadas\n#endif\n"
        "static_assert(__cplusplus >= 202002L);\n"
        "int main() { const std::array<int,2> x{1,2}; return x[0]-1; }\n",
        "-DMARS_TITAN_WARNINGS_AS_ERRORS=ON",
    )
    assert result.returncode == 0, result.stderr
    built = compile_target(build)
    assert built.returncode == 0, built.stderr
    assert subprocess.run([str(build / "probe")], capture_output=True).returncode == 0
    assert (build / "compile_commands.json").is_file()


def test_compiler_warning_becomes_an_error(tmp_path):
    build, result = configure(
        tmp_path, "int main() { int unused; return 0; }\n", "-DMARS_TITAN_WARNINGS_AS_ERRORS=ON"
    )
    assert result.returncode == 0, result.stderr
    built = compile_target(build)
    assert built.returncode != 0
    assert "unused" in built.stderr


def test_address_sanitizer_detects_invalid_access_at_runtime(tmp_path):
    source = (
        "int main(int argc, char**) {\n"
        "    int* x = new int[2];\n"
        "    x[argc + 3] = 7;\n"
        "    const int result = x[argc + 3];\n"
        "    delete[] x;\n"
        "    return result;\n}\n"
    )
    build, result = configure(tmp_path, source, "-DMARS_TITAN_SANITIZER=address-undefined")
    assert result.returncode == 0, result.stderr
    built = compile_target(build)
    assert built.returncode == 0, built.stderr
    run = subprocess.run(
        [str(build / "probe")],
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "ASAN_OPTIONS": "symbolize=0", "DEBUGINFOD_URLS": ""},
    )
    assert run.returncode == 1
    assert "AddressSanitizer" in run.stderr


def test_sanitizers_and_profile_cannot_be_mixed(tmp_path):
    _, result = configure(
        tmp_path,
        "int main() { return 0; }\n",
        "-DMARS_TITAN_SANITIZER=address-undefined",
        "-DMARS_TITAN_PROFILE=ON",
    )
    assert result.returncode != 0
    assert "perfil" in result.stderr.lower()


def test_clang_tidy_rejects_a_null_dereference(tmp_path):
    tool = os.environ.get("MARS_TITAN_CLANG_TIDY") or shutil.which("clang-tidy")
    if not tool:
        pytest.skip("clang-tidy no está configurado")
    build, result = configure(
        tmp_path,
        "int main() { int* value = nullptr; return *value; }\n",
        "-DMARS_TITAN_ENABLE_CLANG_TIDY=ON",
        f"-DMARS_TITAN_CLANG_TIDY={tool}",
    )
    assert result.returncode == 0, result.stderr
    built = compile_target(build)
    assert built.returncode != 0
    assert "clang-analyzer-core.NullDereference" in built.stdout + built.stderr


def test_missing_requested_static_analyzer_fails_configuration(tmp_path):
    _, result = configure(
        tmp_path,
        "int main() { return 0; }\n",
        "-DMARS_TITAN_ENABLE_CLANG_TIDY=ON",
        f"-DMARS_TITAN_CLANG_TIDY={tmp_path / 'missing'}",
    )
    assert result.returncode != 0
    assert "clang-tidy" in result.stderr


def test_missing_analyzer_fails_even_without_scientific_targets(tmp_path):
    if not shutil.which("cmake"):
        pytest.skip("La comprobación requiere CMake")
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(NATIVE),
            "-B",
            str(tmp_path / "build"),
            "-DMARS_TITAN_ENABLE_CLANG_TIDY=ON",
            f"-DMARS_TITAN_CLANG_TIDY={tmp_path / 'missing'}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "clang-tidy" in result.stderr
