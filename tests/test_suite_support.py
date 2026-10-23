"""Apoyo de la suite: modos estrictos, marcas de selección, grupo `reference` y ejecutables
falsos con espacios."""

import ast
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

from tests.learning_hold_lowering import lowers_learning_hold
from tests.suite_support import (
    CUDA_REASON,
    NATIVE_ARTIFACTS,
    NATIVE_VARIABLES,
    REFERENCE_CPU_THREADS,
    SLOW_TESTS,
    cpu_threads,
    episodic_load_problem,
    function_id,
    inner_environment,
    missing_extra,
    missing_native,
    native_available,
    native_required,
    python_shebang,
    read_slow_tests,
    reference_module,
    reference_pins,
    reference_problems,
    selection_skip,
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


def test_selection_skip_names_the_missing_artifact_and_how_to_build_it(tmp_path):
    episodic = {"native": [("MARS_TITAN_EPISODIC_NATIVE",)]}
    reason = selection_skip(episodic, {}, cuda=never, root=tmp_path)
    assert reason == missing_native("MARS_TITAN_EPISODIC_NATIVE")
    assert "`make native-build`" in reason and "native-candidate-cuda" in reason
    # Una variable vacía equivale a no declararla.
    empty = {"MARS_TITAN_EPISODIC_NATIVE": ""}
    assert selection_skip(episodic, empty, cuda=never, root=tmp_path) == reason
    # Una ruta declarada cuenta como presente aunque no exista: la prueba falla al cargarla.
    declared = {"MARS_TITAN_EPISODIC_NATIVE": str(tmp_path / "missing.so")}
    assert selection_skip(episodic, declared, cuda=never, root=tmp_path) is None
    # `native` sin variables solo selecciona.
    assert selection_skip({"native": [()]}, {}, cuda=never, root=tmp_path) is None
    with pytest.raises(ValueError, match="Variable nativa desconocida"):
        selection_skip(
            {"native": [("MARS_TITAN_UNADJUSTED_EDITION",)]}, {}, cuda=never, root=tmp_path
        )


def test_artifacts_with_a_default_path_are_found_where_native_build_leaves_them(tmp_path):
    library = {"native": [("MARS_TITAN_NATIVE_LIBRARY",)]}
    reason = selection_skip(library, {}, cuda=never, root=tmp_path)
    assert "build/native/native-release/libmars_titan_simulation.so" in reason
    compiled = tmp_path / "build/native/native-release/libmars_titan_simulation.so"
    compiled.parent.mkdir(parents=True)
    compiled.write_text("")
    assert native_available("MARS_TITAN_NATIVE_LIBRARY", {}, tmp_path)
    assert selection_skip(library, {}, cuda=never, root=tmp_path) is None
    # El enlace episódico no tiene ruta por defecto y siempre hay que declararlo.
    (tmp_path / "build/native/native-candidate-cuda").mkdir(parents=True)
    (tmp_path / "build/native/native-candidate-cuda/_episodic_native.so").write_text("")
    assert not native_available("MARS_TITAN_EPISODIC_NATIVE", {}, tmp_path)


def test_every_marker_must_be_satisfied_and_cuda_never_falls_back(tmp_path):
    both = {"native": [("MARS_TITAN_PPO_EXECUTABLE",)], "cuda": [()]}
    present = {"MARS_TITAN_PPO_EXECUTABLE": "declared"}
    assert selection_skip(both, present, cuda=never) == CUDA_REASON
    assert selection_skip(both, present, cuda=lambda: True) is None
    assert selection_skip({"cuda": [()]}, {}, cuda=never) == CUDA_REASON
    policies = {"native": [("MARS_TITAN_PPO_EXECUTABLE", "MARS_TITAN_KLPO_EXECUTABLE")]}
    reason = selection_skip(policies, present, cuda=never, root=tmp_path)
    assert reason == missing_native("MARS_TITAN_KLPO_EXECUTABLE")
    assert "Nunca se sustituye por CPU" in CUDA_REASON


def test_missing_extra_names_module_extra_and_command():
    assert missing_extra("duckdb", "gymnasium") == (
        "Falta duckdb (extra data) o gymnasium (extra reinforcement). Se instala con "
        "`make setup`, que ejecuta `uv sync --locked --all-extras`"
    )


def test_inner_sessions_do_not_inherit_the_options_of_the_outer_one(monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "--junitxml=outer.xml")
    monkeypatch.setenv("PYTEST_PLUGINS", "outer_plugin")
    monkeypatch.setenv("MARS_TITAN_REQUIRE_NATIVE", "1")
    environment = inner_environment(["MARS_TITAN_REQUIRE_NATIVE"])
    assert not {"PYTEST_ADDOPTS", "PYTEST_PLUGINS", "MARS_TITAN_REQUIRE_NATIVE"} & set(environment)
    assert environment["PATH"] == os.environ["PATH"]


def test_function_id_drops_only_the_parameters():
    assert function_id("tests/a.py::test_x[cpu-1]") == "tests/a.py::test_x"
    assert function_id("tests/a.py::Case::test_x") == "tests/a.py::Case::test_x"


def defined(path, owners):
    """Indica si el módulo define la función, dentro de sus clases si las hay."""
    scope = ast.parse(path.read_text(encoding="utf-8")).body
    for owner in owners[:-1]:
        classes = [n for n in scope if isinstance(n, ast.ClassDef) and n.name == owner]
        if not classes:
            return False
        scope = classes[0].body
    return any(isinstance(n, ast.FunctionDef) and n.name == owners[-1] for n in scope)


def availability_skips(path, root=ROOT):
    """`skipif` de un módulo de prueba que deciden por CUDA o por un artefacto nativo."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and ast.unparse(node.func).endswith("mark.skipif")):
            continue
        condition = node.args[0] if node.args else None
        names = {
            constants.get(item.id) if isinstance(item, ast.Name) else item.value
            for item in ast.walk(condition)
            if isinstance(item, (ast.Name, ast.Constant))
        }
        if "cuda.is_available" in ast.unparse(condition) or names & set(NATIVE_ARTIFACTS):
            found.append(f"{path.relative_to(root)}:{node.lineno}")
    return found


def test_collected_modules_declare_cuda_and_native_needs_with_markers():
    # Con un skipif propio, `-m "not cuda"` o `-m "not native"` no dejaría fuera la prueba y
    # el motivo de la omisión no diría cómo conseguir lo que falta.
    found = [
        place
        for path in sorted((ROOT / "tests").rglob("test_*.py"))
        for place in availability_skips(path)
    ]
    assert found == [], "Usar pytest.mark.cuda o pytest.mark.native(...) (tests/README.md)"


def test_availability_skips_are_found_through_module_constants(tmp_path):
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        'NATIVE = "MARS_TITAN_PPO_EXECUTABLE"\n'
        "@pytest.mark.skipif(not os.environ.get(NATIVE), reason='x')\n"
        "def test_a(): pass\n"
        "@pytest.mark.skipif(not torch.cuda.is_available(), reason='x')\n"
        "def test_b(): pass\n"
        "@pytest.mark.skipif(sys.platform == 'win32', reason='x')\n"
        "def test_c(): pass\n",
        encoding="utf-8",
    )
    assert availability_skips(probe, tmp_path) == ["test_probe.py:2", "test_probe.py:4"]


def unmarked_native_reads(path, root=ROOT):
    """Pruebas que leen `os.environ[...]` de un artefacto nativo sin `mark.native`.

    Solo se mira la lectura directa con corchetes, que falla con KeyError si falta la ruta.
    Una lectura dentro de un fixture o de una función auxiliar no se detecta.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_marked = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
        )
        and "mark.native" in ast.unparse(node.value)
        for node in tree.body
    )
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
            continue
        marked = module_marked or any(
            "mark.native" in ast.unparse(decorator) for decorator in node.decorator_list
        )
        reads = any(
            isinstance(item, ast.Subscript)
            and ast.unparse(item.value) == "os.environ"
            and isinstance(item.slice, ast.Constant)
            and item.slice.value in NATIVE_ARTIFACTS
            for item in ast.walk(node)
        )
        if reads and not marked:
            found.append(f"{path.relative_to(root)}::{node.name}")
    return found


def test_native_environment_reads_are_marked():
    # Sin la marca, la prueba falla con KeyError cuando falta el binario y `-m "not native"`
    # tampoco la deja fuera de la batería rápida.
    found = [
        place
        for path in sorted((ROOT / "tests").rglob("test_*.py"))
        for place in unmarked_native_reads(path)
    ]
    assert found == [], 'Usar pytest.mark.native("VARIABLE") (tests/README.md)'


def test_unmarked_native_reads_respect_function_and_module_marks(tmp_path):
    source = (
        "import os\n"
        "def test_a(): os.environ['MARS_TITAN_EPISODIC_NATIVE']\n"
        "@pytest.mark.native('MARS_TITAN_EPISODIC_NATIVE')\n"
        "def test_b(): os.environ['MARS_TITAN_EPISODIC_NATIVE']\n"
        "def test_c(): os.environ['HOME']\n"
    )
    probe = tmp_path / "test_probe.py"
    probe.write_text(source, encoding="utf-8")
    assert unmarked_native_reads(probe, tmp_path) == ["test_probe.py::test_a"]
    probe.write_text(
        "pytestmark = pytest.mark.native('MARS_TITAN_EPISODIC_NATIVE')\n" + source,
        encoding="utf-8",
    )
    assert unmarked_native_reads(probe, tmp_path) == []


def test_test_module_names_are_unique():
    # Las carpetas de pruebas no son paquetes. Dos módulos con el mismo nombre no pueden
    # importarse en el mismo proceso y pytest los rechaza con «import file mismatch».
    paths = sorted(
        path
        for pattern in ("test_*.py", "*_test.py")
        for path in (ROOT / "tests").rglob(pattern)
        if ".venv" not in path.parts
    )
    names = {}
    for path in paths:
        names.setdefault(path.name, []).append(str(path.relative_to(ROOT)))
    assert {name: found for name, found in names.items() if len(found) > 1} == {}


def test_slow_registry_lists_existing_test_functions_once():
    lines = [line for line in SLOW_TESTS.read_text(encoding="utf-8").splitlines() if line]
    entries = [line for line in lines if not line.startswith("#")]
    assert entries == sorted(set(entries))
    assert read_slow_tests() == set(entries)
    stale = []
    for entry in entries:
        module, *owners = entry.split("::")
        if not (ROOT / module).is_file() or not owners or not defined(ROOT / module, owners):
            stale.append(entry)
    assert stale == [], "Funciones lentas que ya no existen: regenerar tests/slow_tests.txt"


def parametrized(path, owners):
    """Indica si la función del registro tiene casos parametrizados."""
    scope = ast.parse(path.read_text(encoding="utf-8")).body
    for owner in owners[:-1]:
        scope = next(n for n in scope if isinstance(n, ast.ClassDef) and n.name == owner).body
    function = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == owners[-1])
    return any("parametrize" in ast.unparse(mark) for mark in function.decorator_list)


def test_a_registered_slow_function_is_selected_by_its_marker():
    # Una función parametrizada comprueba que la marca se aplica a todos sus casos.
    entry = next(
        entry
        for entry in sorted(read_slow_tests())
        if parametrized(ROOT / entry.split("::")[0], entry.split("::")[1:])
    )
    selected = collect(entry, "-m", "slow")
    assert selected.returncode == 0 and f"{entry}[" in selected.stdout, selected.stdout
    excluded = collect(entry, "-m", "not slow")
    assert entry not in excluded.stdout


def collect(target, *options):
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--collect-only"]
        + list(options)
        + [target],
        cwd=ROOT,
        env=inner_environment(),
        capture_output=True,
        text=True,
        timeout=300,
    )


SELECTION_PROBE = """
import pytest


@pytest.mark.cuda
def test_cuda():
    pass


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_native():
    pass


@pytest.mark.native_binding
def test_binding():
    pass


@pytest.mark.unguarded_fit
def test_fit():
    pass


def test_plain():
    pass
"""


def selection_probe(tmp_path, allowed, *options):
    """Ejecutar la configuración común sobre un módulo con cada marca y una protección dada."""
    (tmp_path / "test_probe.py").write_text(SELECTION_PROBE, encoding="utf-8")
    hold = tmp_path / "hold.json"
    hold.write_text(f'{{"training_allowed": {str(allowed).lower()}}}', encoding="utf-8")
    environment = inner_environment([key for key in os.environ if key.startswith("MARS_TITAN_")])
    environment.update(
        CUDA_VISIBLE_DEVICES="-1",
        MARS_TITAN_TRAINING_HOLD=str(hold),
        PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
    )
    command = [sys.executable, "-m", "pytest", "-p", "tests.conftest", "-q", "-rs"]
    command += ["-c", os.devnull, "--rootdir", str(tmp_path), "-p", "no:cacheprovider"]
    command += ["-W", "ignore::pytest.PytestUnknownMarkWarning", *options, "test_probe.py"]
    return subprocess.run(
        command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=300
    )


def test_markers_skip_with_their_reason_and_select_with_m(tmp_path):
    result = selection_probe(tmp_path, False)
    output = result.stdout
    assert result.returncode == 0, output + result.stderr
    assert "2 passed, 3 skipped" in output
    assert "Requiere cuda:0 visible" in output
    assert "Falta el enlace episódico nativo" in output
    assert "rige el bloqueo de aprendizaje" in output
    # `native_binding` implica `native`, así que `-m "not native"` también la deja fuera.
    fast = selection_probe(tmp_path, False, "-m", "not native and not cuda")
    assert "1 passed, 1 skipped, 3 deselected" in fast.stdout, fast.stdout


@lowers_learning_hold(
    "tests/test_suite_support.py::test_unguarded_fits_run_when_the_hold_allows_learning"
)
def test_unguarded_fits_run_when_the_hold_allows_learning(tmp_path):
    result = selection_probe(tmp_path, True, "-m", "unguarded_fit")
    assert "1 passed, 4 deselected" in result.stdout, result.stdout


def test_cpu_threads_fix_the_count_and_restore_it_after_an_error():
    import torch

    previous = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="corte"):
        with cpu_threads():
            assert torch.get_num_threads() == REFERENCE_CPU_THREADS
            raise RuntimeError("corte")
    assert torch.get_num_threads() == previous
