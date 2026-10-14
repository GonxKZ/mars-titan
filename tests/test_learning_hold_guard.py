"""Ninguna prueba, medida o script rebaja la protección fuera de lo declarado (#546).

La guarda es estática: lee los archivos sin importarlos ni ejecutarlos. En Python recorre con
`ast` los archivos de `tests`, `benchmarks`, `scripts` y `reports` y reconoce cuatro formas de
rebajar o redirigir la protección:

- `learning_hold(True)`, o `learning_hold(None)`, que la deja ausente. Solo se admiten dentro
  de una prueba de `INVENTORY`, que `lowers_learning_hold` omite mientras la protección real
  bloquee. Con un valor que no se conoce sin ejecutar, solo en los archivos de `READERS`.
- Una protección escrita a mano: un diccionario o un `dict(...)` con `training_allowed`, o un
  texto JSON con `"training_allowed": true`. Con el valor verdadero solo la escribe
  `allowed_hold`. Con un valor que no se conoce sin ejecutar, solo los archivos de `READERS`,
  que prueban el lector de la protección o el fixture `learning_hold`. Con el valor falso
  bloquea y se admite en cualquier prueba.
- Los dobles `learning_doubles`, `learning_doubles_module` y `allowed_hold`, con el número de
  usos de cada archivo declarado en `DOUBLES`.
- Cualquier referencia a la variable o a la ruta de la protección (`HOLD_ENV`,
  `MARS_TITAN_TRAINING_HOLD`, `DEFAULT_HOLD`), con el número de cada archivo declarado en
  `REDIRECTIONS`. Así una prueba nueva tampoco puede apuntar a una protección ausente o
  reutilizar la permitida de otro fixture sin declararlo.

En C++ y CMake busca el texto de una protección permitida fuera de `READERS`.
"""

import ast
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest

from mars_titan.training.learning_hold import HOLD_ENV
from tests.learning_hold_lowering import (
    DOUBLES,
    INVENTORY,
    READERS,
    REDIRECTIONS,
    lowers_learning_hold,
)

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("tests", "benchmarks", "scripts", "reports")
# El inventario declara el ayudante y esta guarda contiene los patrones que busca.
SELF = {"tests/learning_hold_lowering.py", "tests/test_learning_hold_guard.py"}
NATIVE = ("native/tests", "native/cmake", "native/benchmarks", "native/CMakeLists.txt")
ALLOWED_JSON = re.compile(r'\\?"training_allowed\\?"\s*:\s*true')
DOUBLE_NAMES = {"learning_doubles", "learning_doubles_module", "allowed_hold"}
HOLD_NAMES = {"HOLD_ENV", "DEFAULT_HOLD"}
HOLD_TEXT = ("MARS_TITAN_TRAINING_HOLD", "training-hold-2000")


def _true(node):
    return isinstance(node, ast.Constant) and node.value is True


def _opens(call):
    """Cómo abre la protección una llamada a `learning_hold`, con el valor posicional o nombrado.

    Devuelve `True` con `True` o `None`, que la deja ausente, `None` si el valor solo se conoce
    al ejecutar y `False` si la bloquea o la llamada es otra.
    """
    function = call.func
    name = function.id if isinstance(function, ast.Name) else getattr(function, "attr", None)
    if name != "learning_hold":
        return False
    values = [*call.args[:1], *(k.value for k in call.keywords if k.arg == "allowed")]
    if any(isinstance(v, ast.Constant) and v.value in (True, None) for v in values):
        return True
    return None if any(not isinstance(v, ast.Constant) for v in values) else False


def _document(node):
    """Valor de `training_allowed` en un diccionario, `dict(...)` o JSON escritos a mano.

    Devuelve `True` para una protección permitida, `None` si el valor solo se conoce al
    ejecutar y `False` si el nodo no escribe una protección que pueda permitir.
    """
    if isinstance(node, ast.Dict):
        values = [
            value
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant) and key.value == "training_allowed"
        ]
    elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dict":
        values = [k.value for k in node.keywords if k.arg == "training_allowed"]
    else:
        text = isinstance(node, ast.Constant) and isinstance(node.value, str)
        return bool(text and ALLOWED_JSON.search(node.value))
    if any(_true(value) for value in values):
        return True
    return None if any(not isinstance(value, ast.Constant) for value in values) else False


def _named(node, names):
    """Referencias a un nombre: uso, parámetro, definición o texto de un fixture."""
    if isinstance(node, ast.Name | ast.arg | ast.FunctionDef | ast.Attribute):
        value = {
            ast.Name: "id",
            ast.arg: "arg",
            ast.FunctionDef: "name",
            ast.Attribute: "attr",
        }[type(node)]
        return getattr(node, value) in names
    return isinstance(node, ast.Constant) and node.value in names


def _redirects(node):
    if _named(node, HOLD_NAMES):
        return True
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and any(text in node.value for text in HOLD_TEXT)
    )


def _owners(tree):
    """Función de nivel superior que contiene cada nodo, para las claves del inventario."""
    owner = {}
    for top in tree.body:
        for node in ast.walk(top):
            owner[node] = top.name if isinstance(top, ast.FunctionDef) else None
    return owner


def _docstrings(tree):
    """Textos de documentación, que pueden nombrar la protección sin redirigirla."""
    found = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                found.add(value)
    return found


def scan(root=ROOT):
    """Hallazgos de rebajas y redirecciones de la protección bajo `root`.

    Devuelve las pruebas que abren la protección con el fixture, los sitios que escriben una
    protección permitida o con un valor que solo se conoce al ejecutar y, por archivo, los
    usos de dobles y las redirecciones.
    """
    opened, written, dynamic, doubles, redirections = [], [], [], Counter(), Counter()
    paths = [path for folder in FOLDERS for path in sorted((root / folder).rglob("*.py"))]
    for path in paths:
        name = path.relative_to(root).as_posix()
        if name in SELF or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
        owner, documentation = _owners(tree), _docstrings(tree)
        for node in ast.walk(tree):
            if node in documentation:
                continue
            opens = _opens(node) if isinstance(node, ast.Call) else False
            if opens is True:
                opened.append(f"{name}::{owner.get(node)}")
            elif opens is None:
                dynamic.append(f"{name}:{node.lineno}")
            value = _document(node)
            if value is True:
                written.append(f"{name}:{node.lineno}")
            elif value is None:
                dynamic.append(f"{name}:{node.lineno}")
            doubles[name] += _named(node, DOUBLE_NAMES)
            redirections[name] += _redirects(node)
    for part in NATIVE:
        folder = root / part
        for path in [folder] if folder.is_file() else sorted(folder.rglob("*")):
            # Los módulos de Python de `native/tests` que importa la suite dejan bytecode.
            if "__pycache__" in path.parts:
                continue
            if path.is_file() and ALLOWED_JSON.search(path.read_text(encoding="utf-8")):
                written.append(path.relative_to(root).as_posix())
    return opened, written, dynamic, +doubles, +redirections


def problems(root=ROOT):
    """Rebajas fuera de lo declarado y declaraciones que ya no corresponden al código."""
    opened, written, dynamic, doubles, redirections = scan(root)
    found = [
        f"{key} rebaja la protección fuera del inventario" for key in opened if key not in INVENTORY
    ]
    # En Python solo `allowed_hold` escribe una protección permitida: cada sitio lleva su línea
    # y nunca coincide con un archivo declarado. En C++ se admite el lector nativo de READERS.
    found += [f"{site} escribe una protección permitida" for site in written if site not in READERS]
    found += [
        f"{site} escribe una protección con un valor desconocido"
        for site in dynamic
        if site.partition(":")[0] not in READERS
    ]
    for label, counted, declared in (
        ("usa dobles", doubles, DOUBLES),
        ("redirige la protección", redirections, REDIRECTIONS),
    ):
        for name in sorted(set(counted) | set(declared)):
            expected = declared.get(name, (0, None))[0]
            if counted[name] != expected:
                found.append(f"{name} {label} {counted[name]} veces y declara {expected}")
    return found


def test_no_test_benchmark_or_script_lowers_the_hold_outside_the_inventory():
    assert problems() == []


def test_every_inventoried_test_exists_and_skips_while_the_hold_blocks():
    for key, reason in INVENTORY.items():
        path, _, name = key.partition("::")
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        if not name:
            # Un módulo entero se marca con `pytestmark`.
            marks = [
                ast.unparse(node.value)
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(getattr(target, "id", None) == "pytestmark" for target in node.targets)
            ]
        else:
            functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
            assert name in functions, key
            marks = [ast.unparse(item) for item in functions[name].decorator_list]
        assert any("lowers_learning_hold" in mark for mark in marks), key
        assert reason


def test_the_mark_skips_with_a_blocking_hold_and_refuses_an_unknown_key(monkeypatch, tmp_path):
    blocking = tmp_path / "hold.json"
    blocking.write_text('{"training_allowed": false}', encoding="utf-8")
    monkeypatch.setenv(HOLD_ENV, str(blocking))
    mark = lowers_learning_hold(next(iter(INVENTORY))).mark
    assert mark.args == (True,) and "inventario de #546" in mark.kwargs["reason"]
    with pytest.raises(KeyError, match="no figura en el inventario"):
        lowers_learning_hold("tests/otra.py::test_x")


def test_every_declaration_has_its_reason():
    for table in (DOUBLES, REDIRECTIONS):
        for name, (count, reason) in table.items():
            assert count > 0 and reason and (ROOT / name).is_file(), name
    for name, reason in READERS.items():
        assert reason and (ROOT / name).is_file(), name


@pytest.fixture(scope="module")
def tree_copy(tmp_path_factory):
    """Archivos recorridos por la guarda, copiados para añadir rebajas sin tocar el repositorio.

    De las carpetas de Python solo se copian los `.py`, que son lo único que lee la guarda.
    """
    root = tmp_path_factory.mktemp("guard")
    for folder in FOLDERS:
        for path in (ROOT / folder).rglob("*.py"):
            if "__pycache__" not in path.parts:
                target = root / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, target)
    for part in NATIVE:
        source = ROOT / part
        if source.is_file():
            (root / part).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, root / part)
        elif source.is_dir():
            shutil.copytree(source, root / part)
    return root


@pytest.fixture
def changed(tree_copy):
    """Restaurar después de cada prueba los archivos que añade o cambia."""
    saved = {}

    def write(name, text):
        path = tree_copy / name
        saved.setdefault(path, path.read_text(encoding="utf-8") if path.exists() else None)
        path.write_text(text, encoding="utf-8")
        return path

    yield write
    for path, text in saved.items():
        if text is None:
            path.unlink()
        else:
            path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Un archivo de protección permitida falso, de cuatro maneras.
        (
            "def test_x(tmp_path):\n    p = tmp_path / 'h.json'\n"
            "    p.write_text(json.dumps({'training_allowed': True}))\n",
            "escribe una protección permitida",
        ),
        ("def test_x():\n    return dict(training_allowed=True)\n", "escribe una protección"),
        (
            "def test_x(p):\n    p.write_text('{\"training_allowed\": true}')\n",
            "escribe una protección permitida",
        ),
        (
            'def test_x(p):\n    p.write_text("""{\n  \\"training_allowed\\" : true}""")\n',
            "escribe una protección permitida",
        ),
        # Un valor que solo se conoce al ejecutar, fuera de las pruebas del lector.
        (
            "def test_x(p, flag):\n    p.write_text(json.dumps({'training_allowed': flag}))\n",
            "valor desconocido",
        ),
        # La protección abierta o ausente con el fixture.
        ("def test_x(learning_hold):\n    learning_hold(True)\n", "fuera del inventario"),
        ("def test_x(learning_hold):\n    learning_hold(allowed=None)\n", "fuera del inventario"),
        ("def test_x(learning_hold, flag):\n    learning_hold(flag)\n", "valor desconocido"),
        # Dobles y redirecciones nuevas.
        ("def test_x(learning_doubles):\n    pass\n", "usa dobles 1 veces y declara 0"),
        (
            "def test_x(monkeypatch, tmp_path):\n"
            "    monkeypatch.setenv(HOLD_ENV, str(tmp_path / 'absent.json'))\n",
            "redirige la protección 1 veces y declara 0",
        ),
        (
            "import os\nos.environ['MARS_TITAN_TRAINING_HOLD'] = '/dev/null'\n",
            "redirige la protección 1 veces y declara 0",
        ),
    ],
)
def test_the_guard_reports_each_new_way_of_lowering_the_hold(tree_copy, changed, text, expected):
    changed("tests/test_new_lowering.py", text)
    found = problems(tree_copy)
    assert any(item.startswith("tests/test_new_lowering.py") and expected in item for item in found)


def test_the_guard_reports_a_new_use_in_a_declared_file_and_a_native_fake_hold(tree_copy, changed):
    assert problems(tree_copy) == []
    name = next(iter(DOUBLES))
    text = (tree_copy / name).read_text(encoding="utf-8")
    changed(name, text + "\n\ndef test_more(learning_doubles):\n    pass\n")
    count = DOUBLES[name][0]
    assert f"{name} usa dobles {count + 1} veces y declara {count}" in problems(tree_copy)
    changed("native/tests/fake_hold.cpp", 'auto text = R"({"training_allowed": true})";\n')
    assert "native/tests/fake_hold.cpp escribe una protección permitida" in problems(tree_copy)


def test_an_inventoried_test_may_open_the_hold_only_inside_itself(tree_copy, changed):
    key = next(iter(INVENTORY))
    path, _, function = key.partition("::")
    text = (tree_copy / path).read_text(encoding="utf-8")
    changed(path, text + f"\n\ndef {function}_again(learning_hold):\n    learning_hold(True)\n")
    assert f"{key}_again rebaja la protección fuera del inventario" in problems(tree_copy)
