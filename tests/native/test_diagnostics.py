"""Comprobaciones reales del contrato CMake con programas C++ temporales."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

NATIVE = Path(__file__).resolve().parents[2] / "native"


def configure(tmp_path, source, *options):
    if not shutil.which("cmake") or not shutil.which("clang++"):
        pytest.skip(
            "Faltan cmake o clang++ en PATH. La comprobación nativa necesita CMake 3.24 o "
            "posterior, Ninja y Clang (native/README.md)"
        )
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


def analysis_tools():
    tools = {name: shutil.which(name) for name in ("cmake", "clang++")}
    if not all(tools.values()):
        pytest.skip("La comprobación necesita CMake y Clang")
    version = (
        subprocess.run(
            [tools["clang++"], "-dumpversion"], capture_output=True, text=True, check=True
        )
        .stdout.strip()
        .split(".")[0]
    )
    tools["clang-check"] = shutil.which(f"clang-check-{version}") or shutil.which("clang-check")
    if tools["clang-check"] is None:
        pytest.skip("La comprobación necesita clang-check de la versión instalada de Clang")
    return tools


def test_global_static_analysis_runs_the_learning_gate_analysis(tmp_path):
    tools = analysis_tools()
    project = tmp_path / "dispatcher"
    project.mkdir()
    marker = tmp_path / "gate-analyzed"
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(analysis_dispatch LANGUAGES C CXX)\n"
        'set(MARS_TITAN_BUILD_SIMULATION OFF CACHE BOOL "")\n'
        'set(MARS_TITAN_ENABLE_STATIC_ANALYZER ON CACHE BOOL "")\n'
        "add_custom_target(learning-gate-analysis\n"
        f'  COMMAND "${{CMAKE_COMMAND}}" -E touch "{marker}")\n'
        f'add_subdirectory("{NATIVE}" native)\n'
    )
    build = tmp_path / "build"
    configured = subprocess.run(
        [tools["cmake"], "-S", str(project), "-B", str(build), "-DCMAKE_CXX_COMPILER=clang++"],
        capture_output=True,
        text=True,
    )
    assert configured.returncode == 0, configured.stderr
    checked = subprocess.run(
        [tools["cmake"], "--build", str(build), "--target", "static-analysis"],
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stderr
    assert marker.is_file()


def test_static_analysis_includes_the_learning_gate_and_its_launcher(tmp_path):
    tools = analysis_tools()
    project = tmp_path / "gate"
    for folder, names in {
        "src": ("learning_hold.cpp", "learning_gate.cpp", "learning_gate_main.cpp"),
        "tests": ("learning_hold_tests.cpp", "learning_gate_tests.cpp"),
        "include/mars_titan": ("learning_hold.hpp", "learning_gate.hpp"),
    }.items():
        (project / folder).mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copyfile(NATIVE / folder / name, project / folder / name)
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(gate_analysis LANGUAGES CXX)\n"
        "set(CMAKE_CXX_STANDARD 20)\nset(CMAKE_EXPORT_COMPILE_COMMANDS ON)\n"
        "set(BUILD_TESTING ON)\nset(MARS_TITAN_ENABLE_STATIC_ANALYZER ON)\n"
        f'set(MARS_TITAN_CLANG_CHECK "{tools["clang-check"]}")\n'
        "add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)\n"
        "function(mars_titan_configure_target target)\nendfunction()\n"
        "function(mars_titan_sanitizer_test_environment target)\nendfunction()\n"
        f'include("{NATIVE / "cmake/LearningHold.cmake"}")\n'
    )
    build = tmp_path / "build"
    configured = subprocess.run(
        [tools["cmake"], "-S", str(project), "-B", str(build), "-DCMAKE_CXX_COMPILER=clang++"],
        capture_output=True,
        text=True,
    )
    assert configured.returncode == 0, configured.stderr
    checked = subprocess.run(
        [tools["cmake"], "--build", str(build), "--target", "learning-gate-analysis"],
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stderr
    assert (build / "learning_gate.plist").is_file()
    assert (build / "learning_gate_main.plist").is_file()


@pytest.mark.parametrize(("sanitizer", "expected"), [("none", (60, 120)), ("thread", (240, 480))])
def test_instrumented_reference_checks_keep_a_bounded_time_budget(tmp_path, sanitizer, expected):
    build, result = configure(
        tmp_path,
        "int main() { return 0; }\n",
        "-DBUILD_TESTING=ON",
        f"-DMARS_TITAN_SANITIZER={sanitizer}",
    )
    assert result.returncode == 0, result.stderr
    collected = subprocess.run(
        ["ctest", "--test-dir", str(build / "native"), "--show-only=json-v1"],
        capture_output=True,
        text=True,
        check=True,
    )
    cases = {
        test["name"]: {p["name"]: p["value"] for p in test["properties"]}
        for test in json.loads(collected.stdout)["tests"]
    }
    assert (
        tuple(cases[name]["TIMEOUT"] for name in ("simulation", "episodic_projection")) == expected
    )
    assert cases["concurrency"]["TIMEOUT"] == 60


@pytest.mark.parametrize("sanitizer", ["none", "address", "address-undefined"])
def test_correct_cpp_target_compiles_with_strict_options(tmp_path, sanitizer):
    build, result = configure(
        tmp_path,
        "#include <array>\n#ifndef __STRICT_ANSI__\n"
        "#error Se requieren extensiones desactivadas\n#endif\n"
        "static_assert(__cplusplus >= 202002L);\n"
        "int main() { const std::array<int,2> x{1,2}; return x[0]-1; }\n",
        "-DMARS_TITAN_WARNINGS_AS_ERRORS=ON",
        f"-DMARS_TITAN_SANITIZER={sanitizer}",
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


@pytest.mark.parametrize("sanitizer", ["address", "address-undefined"])
def test_address_sanitizer_detects_invalid_access_at_runtime(tmp_path, sanitizer):
    source = (
        "int main(int argc, char**) {\n"
        "    int* x = new int[2];\n"
        "    x[argc + 3] = 7;\n"
        "    const int result = x[argc + 3];\n"
        "    delete[] x;\n"
        "    return result;\n}\n"
    )
    build, result = configure(tmp_path, source, f"-DMARS_TITAN_SANITIZER={sanitizer}")
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
        pytest.skip(
            "Falta clang-tidy. Se instala con LLVM o se declara su ruta en MARS_TITAN_CLANG_TIDY"
        )
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
        pytest.skip("Falta cmake en PATH. La comprobación necesita CMake 3.24 o posterior")
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
