"""Las pruebas de CTest que dan pasos de optimizador quedan fuera de todos los perfiles.

Mientras rija el bloqueo de aprendizaje, ninguna comprobación técnica puede aplicar un paso de
optimizador, tampoco con datos sintéticos. Estas pruebas leen CMake, C++ y los perfiles sin
configurar ni compilar nada. Localizan los ejecutables de cada prueba de CTest, buscan en sus
fuentes las llamadas que llegan a un paso y exigen que esas pruebas, y solo esas, lleven la
etiqueta `optimizer-steps` y que cada perfil de prueba la excluya.

La búsqueda es textual. Una función nueva de `native/src` que aplique o propague un paso tiene
que declararse en `STEPPING`, y la prueba de completitud falla mientras no se haga.
"""

import json
import re
import shutil
from pathlib import Path

import pytest

NATIVE = Path(__file__).resolve().parents[2] / "native"
LABEL = "optimizer-steps"

# Funciones que aplican un paso de optimizador o llegan a uno, por el archivo que las define.
STEPPING = {
    "ppo_policy.cpp": (
        "update",
        "update_from_cpu",
        "terminal_step",
        "consolidate",
        "update_double_dqn",
    ),
    "adapter_control.cpp": ("train",),
    "ppo_training.cpp": ("advance",),
    "klpo_learning.cpp": ("update_ready",),
    "ppo_experiment.cpp": ("run_ppo_experiment",),
    "klpo_experiment.cpp": ("run_klpo_experiment",),
}
# Llamadas homónimas de la biblioteca que no llegan a ningún optimizador.
NOT_STEPPING = {
    "cohort_execution.cpp": "update es la devolución de estado de las cohortes",
    "candidate_archive.cpp": "train cambia el modo del módulo de LibTorch",
}
# Pruebas que ejecutan un programa que entrena, pero que se detiene antes del primer paso.
# La primera solo pasa si aparece el mensaje de una protección bloqueada. La segunda debe
# fallar, porque el programa rechaza cuda:1 al leer los argumentos.
REFUSED = {"adapter_control_cli_hold": "hold", "adapter_control_cli_invalid": "arguments"}

NAMES = sorted({name for names in STEPPING.values() for name in names})
# Llamadas como método o con nombre cualificado. Las declaraciones de las cabeceras no cuentan.
CALL = re.compile(r"(?:\.|->|::)\s*(" + "|".join(NAMES) + r")\s*\(")
OPTIMIZER = re.compile(r"torch::optim::")
STEP = re.compile(r"(?:\.|->)\s*step\s*\(\s*\)")
TOKEN = re.compile(r'"((?:\\.|[^"\\])*)"|#[^\n]*|([()])|([^\s()#"]+)')


def commands(text):
    """Órdenes de CMake con sus argumentos, sin comentarios ni comillas."""
    found, current, depth, previous = [], None, 0, None
    for match in TOKEN.finditer(text):
        quoted, paren, word = match.groups()
        if quoted is None and paren is None and word is None:
            continue
        if paren == "(":
            if depth == 0:
                current = (previous, [])
            depth += 1
        elif paren == ")":
            depth -= 1
            if depth == 0:
                found.append(current)
                current = None
        elif depth:
            current[1].append(quoted if quoted is not None else word)
        previous = word
    return found


def _expand(body, variable, item):
    return [(name, [arg.replace(f"${{{variable}}}", item) for arg in args]) for name, args in body]


def _source(path):
    for prefix in ("${CMAKE_CURRENT_SOURCE_DIR}/", "${CMAKE_CURRENT_LIST_DIR}/../"):
        path = path.removeprefix(prefix)
    return path


class Registry:
    """Pruebas, fuentes de cada ejecutable y propiedades de CTest de toda la configuración.

    Se reúnen las ramas de todos los `if`, porque cualquier configuración puede registrar la
    prueba. Los bucles `foreach` se expanden con sus elementos o con una lista ya declarada.
    """

    def __init__(self):
        self.lists, self.sources, self.tests, self.properties = {}, {}, {}, {}

    def read(self, items):
        position = 0
        while position < len(items):
            name, args = items[position]
            if name == "foreach":
                depth, end = 1, position
                while depth:
                    end += 1
                    depth += {"foreach": 1, "endforeach": -1}.get(items[end][0], 0)
                self._loop(args, items[position + 1 : end])
                position = end + 1
                continue
            self._command(name, args)
            position += 1

    def _loop(self, args, body):
        variable, values = args[0], args[1:]
        if values[:2] == ["IN", "LISTS"]:
            values = [item for name in values[2:] for item in self.lists.get(name, [])]
        elif values[:2] == ["IN", "ITEMS"]:
            values = values[2:]
        for item in values:
            self.read(_expand(body, variable, item))

    def _command(self, name, args):
        if name == "set" and args:
            self.lists[args[0]] = args[1:]
        elif name == "add_executable":
            self.sources.setdefault(args[0], []).extend(_source(arg) for arg in args[1:])
        elif name == "target_sources" and args[0] in self.sources:
            self.sources[args[0]].extend(_source(arg) for arg in args[2:])
        elif name == "add_test" and args[:1] == ["NAME"]:
            assert args[2] == "COMMAND", args
            self.tests[args[1]] = args[3:]
        elif name == "set_tests_properties":
            split = args.index("PROPERTIES")
            pairs = args[split + 1 :]
            for test in args[:split]:
                for key, value in zip(pairs[::2], pairs[1::2], strict=True):
                    self._add(test, key, value.split(";"))
        elif name == "set_property" and args[:1] == ["TEST"]:
            split = args.index("PROPERTY")
            tests = [arg for arg in args[1:split] if arg not in ("APPEND", "APPEND_STRING")]
            for test in tests:
                self._add(test, args[split + 1], args[split + 2 :])

    def _add(self, test, key, values):
        self.properties.setdefault(test, {}).setdefault(key, []).extend(values)

    def labels(self, test):
        return set(self.properties.get(test, {}).get("LABELS", []))

    def executed(self, test):
        """Ejecutables propios que lanza una prueba, directamente o con `$<TARGET_FILE:...>`."""
        command = self.tests[test]
        targets = [command[0]] + [
            target for arg in command for target in re.findall(r"\$<TARGET_FILE:([^>]+)>", arg)
        ]
        return [target for target in targets if target in self.sources]


def registry(native=NATIVE):
    result = Registry()
    for path in [*sorted((native / "cmake").glob("*.cmake")), native / "CMakeLists.txt"]:
        result.read(commands(path.read_text()))
    return result


def steps(path):
    """Indica si un archivo de pruebas o de un programa llega a un paso de optimizador."""
    text = path.read_text()
    return bool(CALL.search(text)) or bool(OPTIMIZER.search(text) and STEP.search(text))


def stepping_tests(found, native=NATIVE):
    result = set()
    for test in found.tests:
        sources = [
            native / source for target in found.executed(test) for source in found.sources[target]
        ]
        missing = [str(source) for source in sources if not source.is_file()]
        assert not missing, f"{test} usa fuentes que no existen: {missing}"
        if any(steps(source) for source in sources) and test not in REFUSED:
            result.add(test)
    return result


def library_findings(native=NATIVE):
    """Archivos de la biblioteca que aplican o propagan un paso sin estar declarados."""
    undeclared, stale = [], []
    for path in sorted([*(native / "src").glob("*.cpp"), *(native / "include").rglob("*.hpp")]):
        if path.name.endswith("_main.cpp"):
            continue
        text = path.read_text()
        names = set(CALL.findall(text)) - set(STEPPING.get(path.name, ()))
        if OPTIMIZER.search(text) and path.name not in STEPPING:
            undeclared.append(f"{path.name} usa torch::optim")
        if names and path.name not in STEPPING and path.name not in NOT_STEPPING:
            undeclared.append(f"{path.name} llama a {sorted(names)}")
        if path.name in NOT_STEPPING and not names:
            stale.append(path.name)
    for name, functions in STEPPING.items():
        text = (native / "src" / name).read_text()
        stale += [
            f"{name}:{function}"
            for function in functions
            if not re.search(rf"\b{function}\s*\(", text)
        ]
    return undeclared, stale


def exclusions(native=NATIVE):
    """Expresión de etiquetas excluidas que hereda cada perfil de prueba visible."""
    presets = {
        item["name"]: item
        for item in json.loads((native / "CMakePresets.json").read_text())["testPresets"]
    }

    def excluded(name):
        preset = presets[name]
        label = preset.get("filter", {}).get("exclude", {}).get("label")
        if label is not None:
            return label
        parents = preset.get("inherits", [])
        for parent in [parents] if isinstance(parents, str) else parents:
            if (found := excluded(parent)) is not None:
                return found
        return None

    return {name: excluded(name) for name, preset in presets.items() if not preset.get("hidden")}


def test_every_test_that_reaches_an_optimizer_step_carries_the_label():
    found = registry()
    labelled = {test for test in found.tests if LABEL in found.labels(test)}
    expected = stepping_tests(found)
    assert expected - labelled == set(), "Pruebas con pasos de optimizador sin la etiqueta"
    assert labelled - expected == set(), "Etiqueta en pruebas sin pasos de optimizador"
    # Las seis pruebas de PPO y del adaptador, más los programas del control de adaptadores.
    assert {
        "ppo_policy",
        "ppo_training",
        "ppo_collection",
        "ppo_gru_packing",
        "ppo_variant_state",
        "adapter_control",
        "adapter_control_cli",
        "adapter_control_cuda",
        "adapter_control_cli_cuda",
    } == expected


def test_every_test_preset_excludes_the_label_and_nothing_else():
    found = registry()
    other = {label for test in found.tests for label in found.labels(test)} - {LABEL}
    for name, expression in exclusions().items():
        assert expression is not None, f"{name} no excluye {LABEL}"
        assert re.search(expression, LABEL), name
        assert not any(re.search(expression, label) for label in other), name


def test_refused_runs_stop_before_the_first_step():
    found = registry()
    for test, reason in REFUSED.items():
        properties = found.properties[test]
        if reason == "hold":
            environment = properties["ENVIRONMENT"]
            expected = properties["PASS_REGULAR_EXPRESSION"]
            assert "MARS_TITAN_TRAINING_HOLD=${MARS_TITAN_BLOCKED_HOLD}" in environment
            assert any("Bloqueo de aprendizaje vigente" in value for value in expected)
        else:
            assert properties["WILL_FAIL"] == ["TRUE"] and found.tests[test][1] == "cuda:1"
        assert LABEL not in found.labels(test)


def test_the_library_declares_every_file_that_applies_or_propagates_a_step():
    undeclared, stale = library_findings()
    assert undeclared == [], "Declarar en STEPPING o en NOT_STEPPING"
    assert stale == [], "Declaraciones que ya no corresponden al código"


@pytest.fixture
def native_copy(tmp_path):
    copy = tmp_path / "native"
    copy.mkdir()
    for part in ("CMakeLists.txt", "CMakePresets.json"):
        shutil.copy(NATIVE / part, copy / part)
    for folder in ("cmake", "src", "include", "tests"):
        shutil.copytree(NATIVE / folder, copy / folder)
    return copy


def _edit(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new))


def test_a_missing_label_a_new_stepping_test_or_an_open_preset_is_reported(native_copy):
    lists = native_copy / "CMakeLists.txt"
    _edit(lists, "ppo_variant_state adapter_control adapter", "adapter_control adapter")
    found = registry(native_copy)
    assert "ppo_variant_state" in stepping_tests(found, native_copy) - {
        test for test in found.tests if LABEL in found.labels(test)
    }
    # Un ejecutable nuevo que llama a PpoTrainer::advance también exige la etiqueta.
    (native_copy / "tests" / "ppo_new_tests.cpp").write_text(
        "void run(T& trainer) { trainer.advance(); }\n"
    )
    lists.write_text(
        lists.read_text()
        + "add_executable(ppo_new_tests tests/ppo_new_tests.cpp)\n"
        + "add_test(NAME ppo_new COMMAND ppo_new_tests)\n"
    )
    assert "ppo_new" in stepping_tests(registry(native_copy), native_copy)
    presets = native_copy / "CMakePresets.json"
    _edit(presets, ',"filter":{"exclude":{"label":"^optimizer-steps$"}}', "")
    assert all(expression is None for expression in exclusions(native_copy).values())


def test_a_new_library_path_to_an_optimizer_must_be_declared(native_copy):
    (native_copy / "src" / "campaign_fit.cpp").write_text(
        "void fit(P& policy) { policy.update_ready(); }\n"
    )
    (native_copy / "src" / "custom_optimizer.cpp").write_text("torch::optim::SGD sgd(p, 0.1);\n")
    undeclared, _ = library_findings(native_copy)
    assert undeclared == [
        "campaign_fit.cpp llama a ['update_ready']",
        "custom_optimizer.cpp usa torch::optim",
    ]
