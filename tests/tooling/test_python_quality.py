"""Cálculo común de cobertura, complejidad y CRAP de las fuentes Python."""

import ast
import importlib
import json
import sys
import textwrap
from pathlib import Path

import coverage
import pytest

quality = importlib.import_module("scripts.summarize_python_quality")
ROOT = Path(__file__).resolve().parents[2]


def function_complexity(code, name="f"):
    tree = ast.parse(textwrap.dedent(code))
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and item.name == name
    )
    return quality.complexity(node)


# Valores de ComplexityVisitor en radon/visitors.py de Radon 6.0.1. Fijan la convención de los
# recibos, así que otra versión de Radon que cambie una regla rompe esta prueba.
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("def f():\n    return 1", 1),
        (
            "def f(x):\n    if x:\n        return 1\n    elif x > 1:\n        return 2\n"
            "    else:\n        return 3",
            3,
        ),
        ("def f(x):\n    if x:\n        pass\n    else:\n        if x > 1:\n            pass", 3),
        # El else de un bucle suma uno además de lo que contiene.
        ("def f(x):\n    for i in x:\n        pass\n    else:\n        if i:\n            pass", 4),
        ("def f(x):\n    while x:\n        x -= 1", 2),
        (
            "def f():\n    try:\n        pass\n    except ValueError:\n        pass\n"
            "    except KeyError:\n        pass\n    else:\n        pass\n"
            "    finally:\n        pass",
            4,
        ),
        # Radon 6.0.1 no reconoce los manejadores de except*.
        ("def f():\n    try:\n        pass\n    except* ValueError:\n        pass", 1),
        ("def f(x):\n    with x:\n        if x:\n            pass", 2),
        ("def f(x):\n    return [i for i in x if i and x] if x or not x else (lambda y: y)", 6),
        ("def f(x):\n    return [j for i in x for j in i if j if i]", 5),
        ("def f(x):\n    return lambda y: y and x", 2),
        ("def f(x):\n    assert x and not x\n    return x", 2),
        ("def f(x=1 or 2):\n    return x", 1),
        (
            "def f(x):\n    match x:\n        case 1:\n            pass\n        case 2:\n"
            "            pass\n        case _:\n            pass",
            3,
        ),
        (
            "def f(x):\n    match x:\n        case 1:\n            pass\n        case y:\n"
            "            pass",
            2,
        ),
        # La guarda no cuenta: un `case _` también resta con condición.
        (
            "def f(x):\n    match x:\n        case 1:\n            pass\n        case _ if x:\n"
            "            pass",
            2,
        ),
        # Una alternativa o una captura con subpatrón no restan aunque sean irrefutables.
        (
            "def f(x):\n    match x:\n        case 1:\n            pass\n        case 2 | _:\n"
            "            pass",
            3,
        ),
        (
            "def f(x):\n    match x:\n        case 1:\n            pass\n"
            "        case (2 | _) as y:\n            pass",
            3,
        ),
        (
            "def f(x):\n    match x:\n        case 1:\n            pass\n        case [_]:\n"
            "            pass",
            3,
        ),
        ("def f(x):\n    match x:\n        case _:\n            pass", 1),
        (
            "def f(x):\n    def g():\n        if x:\n            pass\n        for _ in x:\n"
            "            pass\n    return g",
            1,
        ),
        (
            "def f(x):\n    class Local:\n        if x:\n            y = 1\n\n"
            "        def m(self):\n            while x:\n                pass\n    return Local",
            1,
        ),
        (
            "async def f(x):\n    async for i in x:\n        async with i:\n            if i:\n"
            "                pass",
            3,
        ),
    ],
)
def test_complexity_follows_the_rules_of_radon(code, expected):
    assert function_complexity(code) == expected


def test_complexity_only_accepts_a_function():
    # Radon devolvería las funciones de un módulo o ninguna de una clase sin avisar.
    for code in ("x = 1", "class C:\n    def f(self):\n        return 1"):
        with pytest.raises(TypeError, match="definición de función"):
            quality.complexity(ast.parse(code).body[0])


def test_regions_are_the_ones_coverage_measures():
    found = {
        item["function"]: item
        for item in quality.regions(
            textwrap.dedent("""
        def outer(x):
            class Local:
                def method(self):
                    if x:
                        return 1
                    return 2

            def inner():
                for _ in x:
                    pass

            return inner, Local


        class Top:
            class Nested:
                def deep(self):
                    while True:
                        break

            @property
            def value(self):
                return 1


        def branchy(flag):
            if flag:
                def chosen():
                    return 1
            else:
                def chosen():
                    return 2
            return chosen


        try:
            import json
        except ImportError:
            def fallback():
                return None
        """)
        )
    }
    assert {name: item["complexity"] for name, item in found.items()} == {
        "outer": 1,
        "outer.Local.method": 2,
        "outer.inner": 2,
        "Top.Nested.deep": 2,
        "Top.value": 1,
        "branchy": 2,
        "branchy.chosen": 1,
    }
    # La región empieza en la línea `def`, no en el decorador, como en coverage.py.
    assert found["Top.value"]["start_line"] == 23
    # coverage.py solo recorre los cuerpos: una función en un else o un except no es región.
    assert found["branchy.chosen"]["start_line"] == 29


@pytest.mark.parametrize(
    ("complexity", "fraction", "expected"),
    [(1, 1.0, 1.0), (5, 0.0, 30.0), (4, 0.5, 6.0), (10, 0.9, 10.1)],
)
def test_crap_formula(complexity, fraction, expected):
    assert quality.crap(complexity, fraction) == pytest.approx(expected)


MODULE = """\
def covered(x):
    if x:
        return 1
    return 2


def partial(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    for _ in range(3):
        pass
    return 0


def empty():
    \"\"\"Solo documentación.\"\"\"


class Holder:
    @property
    def value(self):
        return 1

    @value.setter
    def value(self, item):
        if item:
            self.item = item


def outer():
    def inner(flag):
        while flag:
            flag = False
        return flag

    return inner(True)


def branchy(flag):
    if flag:
        def chosen():
            return 1
    else:
        def chosen():
            return 2
    return chosen() and flag
"""


def measure(tmp_path, *, branch=True, source=MODULE):
    module = tmp_path / "measured_module.py"
    module.write_text(source, encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        # Sin leer configuración: si la suite se mide con coverage.py, COVERAGE_RCFILE
        # declara otras fuentes y esta medida quedaría vacía.
        cov = coverage.Coverage(
            branch=branch, data_file=None, include=[str(module)], config_file=False
        )
        cov.start()
        try:
            imported = importlib.import_module("measured_module")
            imported.covered(1)
            imported.partial(1)
            imported.outer()
            holder = imported.Holder()
            holder.value = 3
            imported.branchy(False)
        finally:
            cov.stop()
        report = tmp_path / "coverage.json"
        cov.json_report(outfile=str(report), include=[str(module)])
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("measured_module", None)
    data = json.loads(report.read_text())
    # Rutas relativas a la raíz de la medida, como en una medida desde el repositorio.
    data["files"] = {Path(name).name: value for name, value in data["files"].items()}
    return data


def test_summary_joins_coverage_regions_and_complexity(tmp_path):
    report = quality.summarize(measure(tmp_path), tmp_path, require_branches=True)
    rows = {row["function"]: row for row in report["functions"]}
    assert set(rows) == {
        "covered",
        "partial",
        "empty",
        "Holder.value",
        "outer",
        "outer.inner",
        "branchy",
        "branchy.chosen",
    }
    assert rows["covered"]["cyclomatic_complexity"] == 2
    assert rows["covered"]["coverage_fraction"] == pytest.approx(2 / 3)
    assert rows["partial"]["cyclomatic_complexity"] == 4
    fraction = rows["partial"]["covered_statements"] / rows["partial"]["statements"]
    assert rows["partial"]["crap"] == pytest.approx(16 * (1 - fraction) ** 3 + 4)
    assert rows["empty"]["statements"] == 0 and rows["empty"]["coverage_fraction"] is None
    assert rows["empty"]["crap"] == 1
    assert rows["outer"]["cyclomatic_complexity"] == 1
    assert rows["outer.inner"]["cyclomatic_complexity"] == 2
    # La función del else no es región: su def y su return cuentan en branchy.
    assert rows["branchy"]["cyclomatic_complexity"] == 3
    assert rows["branchy"]["statements"] == 5 and rows["branchy"]["covered_statements"] == 4
    assert rows["branchy.chosen"]["covered_statements"] == 0
    # El setter es la región que mide coverage.py y el getter queda declarado aparte.
    assert rows["Holder.value"]["start_line"] == 27
    assert report["shadowed_regions"] == [
        {"path": "measured_module.py", "function": "Holder.value", "start_line": 23}
    ]
    assert report["tools"]["coverage"] == coverage.__version__
    assert report["tools"]["radon"] == "6.0.1"
    assert [row["crap"] for row in report["functions"]] == sorted(
        (row["crap"] for row in report["functions"]), reverse=True
    )
    assert report["totals"]["branches"] > 0


def test_changed_source_after_the_measurement_is_rejected(tmp_path):
    data = measure(tmp_path)
    (tmp_path / "measured_module.py").write_text("\n" + MODULE, encoding="utf-8")
    with pytest.raises(ValueError, match="cambió desde la medida"):
        quality.summarize(data, tmp_path)


def test_statement_added_without_moving_a_definition_is_rejected(tmp_path):
    data = measure(tmp_path)
    # Ninguna línea def cambia, pero branchy tiene una sentencia que no se midió.
    changed = MODULE.replace(
        "    return chosen() and flag\n", "    flag = not flag\n    return chosen() and flag\n"
    )
    (tmp_path / "measured_module.py").write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="sentencias medidas"):
        quality.summarize(data, tmp_path)


def test_functions_that_swap_names_are_rejected(tmp_path):
    data = measure(tmp_path)
    # Las líneas de sentencia no cambian, pero cada nombre apunta ya a otra región.
    swapped = (
        MODULE.replace("def covered(", "def swapped(")
        .replace("def partial(", "def covered(")
        .replace("def swapped(", "def partial(")
    )
    (tmp_path / "measured_module.py").write_text(swapped, encoding="utf-8")
    with pytest.raises(ValueError, match="no coincide con la fuente actual"):
        quality.summarize(data, tmp_path)


def test_lines_excluded_in_the_measurement_are_not_a_change(tmp_path):
    source = MODULE.replace("def empty():", "def empty():  # pragma: no cover")
    report = quality.summarize(measure(tmp_path, source=source), tmp_path)
    assert "empty" in {row["function"] for row in report["functions"]}


def test_function_added_after_the_measurement_is_rejected(tmp_path):
    data = measure(tmp_path)
    # Las regiones medidas conservan su línea, pero la nueva función no tiene medida.
    added = MODULE + "\n\ndef added():\n    return 1\n"
    (tmp_path / "measured_module.py").write_text(added, encoding="utf-8")
    with pytest.raises(ValueError, match="no registra la región de added"):
        quality.summarize(data, tmp_path)


def test_measurement_without_branches_is_rejected_when_branches_are_required(tmp_path):
    data = measure(tmp_path, branch=False)
    with pytest.raises(ValueError, match="ramas"):
        quality.summarize(data, tmp_path, require_branches=True)
    assert quality.summarize(data, tmp_path)["totals"]["branches"] == 0


def test_selection_and_format_are_checked(tmp_path):
    data = measure(tmp_path)
    with pytest.raises(ValueError, match="selección"):
        quality.summarize(data, tmp_path, include=["otro.py"])
    with pytest.raises(ValueError, match="formato 3"):
        quality.summarize({**data, "meta": {**data["meta"], "format": 2}}, tmp_path)
    without_regions = json.loads(json.dumps(data))
    for item in without_regions["files"].values():
        item.pop("functions")
    with pytest.raises(ValueError, match="regiones de función"):
        quality.summarize(without_regions, tmp_path)


def test_cli_writes_a_deterministic_report(tmp_path):
    data = measure(tmp_path)
    source = tmp_path / "coverage.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    outputs = []
    for name in ("first.json", "second.json"):
        output = tmp_path / name
        quality.main(
            [
                "--coverage",
                str(source),
                "--output",
                str(output),
                "--root",
                str(tmp_path),
                "--require-branches",
                "--top",
                "2",
            ]
        )
        outputs.append(output.read_text(encoding="utf-8"))
    assert outputs[0] == outputs[1]
    report = json.loads(outputs[0])
    assert report["functions_measured"] == 8 and len(report["functions"]) == 2
    assert report["measured_at"] == data["meta"]["timestamp"]
    assert len(report["coverage_sha256"]) == 64
