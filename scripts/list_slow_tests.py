"""Lista las funciones de prueba lentas a partir de los informes JUnit de pytest.

Una función es lenta si la suma de los tiempos de todos sus casos parametrizados alcanza el
umbral. Cada informe debe contener todos los casos de las funciones que mide, como ocurre al
ejecutar la suite por carpetas. Si una función aparece en varios informes, cuenta el mayor de
sus tiempos, de modo que se pueden juntar ejecuciones distintas sin sumarlas.

JUnit atribuye la preparación de un fixture de módulo a la primera prueba que lo usa. Si esa
prueba pasa a ser lenta, la siguiente que use el fixture pagaría su coste en el subconjunto
rápido. Por eso una función que alcanza `--shared` segundos marca también como lentas las
funciones de su módulo que comparten con ella un fixture de módulo o de sesión.

La salida sustituye a `tests/slow_tests.txt`, que `tests/conftest.py` lee para añadir la marca
`slow`. Los tiempos dependen del equipo y de la carga, así que el archivo se regenera con una
ejecución completa y se revisa el diff antes de publicarlo.

Un informe de `make test-fast` no mide las funciones que ya están en la lista, porque esa
selección las excluye. Con `--previous` se conservan las de una lista anterior y se añaden las
nuevas que alcanzan el umbral, sin recorrer la suite completa.

Uso:
    uv run pytest --junitxml=suite.xml ...
    uv run python scripts/list_slow_tests.py --threshold 1 --shared 10 suite.xml \
        > tests/slow_tests.txt
    uv run python scripts/list_slow_tests.py --previous tests/slow_tests.txt fast.xml \
        > nueva.txt
"""

from __future__ import annotations

import argparse
import ast
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def function_id(classname: str, name: str) -> str:
    """Identificador de pytest de la función, sin la parte de los parámetros."""
    parts = classname.split(".")
    modules = [index for index, part in enumerate(parts) if part.startswith("test_")]
    if not modules:
        raise ValueError(f"Clase de JUnit sin módulo de prueba: {classname}")
    module = modules[-1]
    node = "/".join(parts[: module + 1]) + ".py"
    for owner in parts[module + 1 :]:
        node += f"::{owner}"
    return f"{node}::{name.split('[', 1)[0]}"


def totals(reports: list[Path]) -> dict[str, float]:
    """Segundos de cada función: suma de sus casos en un informe y máximo entre informes."""
    seconds: dict[str, float] = {}
    for report in reports:
        current: dict[str, float] = defaultdict(float)
        for case in ET.parse(report).iter("testcase"):
            current[function_id(case.get("classname", ""), case.get("name", ""))] += float(
                case.get("time", 0)
            )
        for node, value in current.items():
            seconds[node] = max(seconds.get(node, 0.0), value)
    return seconds


def module_scoped(decorator: ast.expr) -> bool:
    """Indica si el decorador declara un fixture de módulo o de sesión."""
    if not isinstance(decorator, ast.Call) or not ast.unparse(decorator.func).endswith("fixture"):
        return False
    scopes = [k.value for k in decorator.keywords if k.arg == "scope"]
    return any(isinstance(v, ast.Constant) and v.value in ("module", "session") for v in scopes)


def fixture_partners(root: Path, node: str) -> set[str]:
    """Funciones de prueba del módulo que comparten con `node` un fixture de módulo o sesión.

    Solo considera funciones de primer nivel y los fixtures definidos en el mismo módulo,
    siguiendo los argumentos de forma transitiva.
    """
    module, *owners = node.split("::")
    path = root / module
    if len(owners) != 1 or not path.is_file():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    arguments = {n.name: {a.arg for a in n.args.args} for n in functions}
    scoped = {n.name for n in functions if any(module_scoped(d) for d in n.decorator_list)}

    def closure(name: str) -> set[str]:
        found, pending = set(), [name]
        while pending:
            for used in arguments.get(pending.pop(), ()):
                if used not in found:
                    found.add(used)
                    pending.append(used)
        return found

    shared = closure(owners[0]) & scoped
    if not shared:
        return set()
    return {
        f"{module}::{name}"
        for name in arguments
        if name.startswith("test_") and closure(name) & shared
    }


def slow_functions(seconds: dict[str, float], threshold: float, shared: float, root: Path):
    slow = {node for node, value in seconds.items() if value >= threshold}
    for node, value in seconds.items():
        if value >= shared:
            slow |= fixture_partners(root, node)
    return slow


def previous_functions(path: Path) -> set[str]:
    """Funciones de una lista anterior, sin la cabecera ni las líneas vacías."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def render(slow: set[str], threshold: float, shared: float) -> str:
    header = [
        "# Funciones de prueba lentas: la suma de sus casos alcanza "
        f"{threshold:g} s en alguna de las medidas de referencia,",
        f"# o comparten un fixture de módulo con una función que alcanza {shared:g} s.",
        "# Se regenera con scripts/list_slow_tests.py. Ver tests/README.md.",
    ]
    return "\n".join(header + sorted(slow)) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", type=Path, help="informes JUnit de pytest")
    parser.add_argument("--threshold", type=float, default=1.0, help="segundos por función")
    parser.add_argument(
        "--shared",
        type=float,
        default=10.0,
        help="segundos desde los que una función marca a las que comparten su fixture de módulo",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="raíz del repositorio")
    parser.add_argument(
        "--previous",
        type=Path,
        help="lista anterior cuyas funciones se conservan, para informes que no las miden",
    )
    args = parser.parse_args(argv)
    if args.threshold <= 0 or args.shared < args.threshold:
        parser.error("El umbral debe ser positivo y --shared no puede ser menor que él")
    slow = slow_functions(totals(args.reports), args.threshold, args.shared, args.root)
    if args.previous is not None:
        slow |= previous_functions(args.previous)
    sys.stdout.write(render(slow, args.threshold, args.shared))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
