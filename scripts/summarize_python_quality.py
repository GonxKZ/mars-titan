# /// script
# requires-python = ">=3.12"
# dependencies = ["coverage==7.16.1", "radon==6.0.1"]
# ///
"""Cobertura, complejidad y CRAP por función de las fuentes Python propias.

Es el cálculo común de los informes de calidad en Python, equivalente a
`summarize_native_quality.py` en la parte nativa. Lee el JSON de coverage.py (formato 3, con
las regiones de función que coverage.py publica desde la versión 7.6) y, para cada función,
calcula su complejidad ciclomática sobre la misma región que coverage.py usa para contar sus
sentencias.

Convenciones:

- Regiones: las de `coverage.regions.code_regions`, la misma función con la que coverage.py
  reparte las sentencias. Una función anidada es su propia región. Una función definida en
  un `else`, un `except` o un `finally` no es región para coverage.py y sus sentencias
  cuentan en la región que la contiene.
- Complejidad: `ComplexityVisitor` de Radon 6.0.1, la herramienta con la que se calcularon
  los recibos anteriores, aplicado a cada función. Suman uno cada `if`, `elif` y expresión
  condicional, cada `for` y `while` y su `else`, cada `except` y el `else` de un `try`, cada
  `assert`, cada operando booleano después del primero y cada cláusula `for` o `if` de una
  comprensión. Un `match` suma sus casos menos uno si alguno captura sin patrón (`case _` o
  `case x`). No suman `with`, `finally`, los manejadores de `except*`, las funciones y clases
  anidadas ni la condición de un `assert`.
- Cobertura: sentencias ejecutadas entre sentencias de la región según coverage.py. Las
  ramas se publican aparte y no entran en CRAP.
- CRAP = C² × (1 − cobertura)³ + C. Una función sin sentencias medibles conserva C.

La salida no depende del momento en que se ejecuta: registra la fecha de la medida de
coverage.py y las huellas de las fuentes y del JSON leído.
"""

import argparse
import ast
import fnmatch
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

from coverage.python import PythonParser
from coverage.regions import code_regions
from radon.visitors import ComplexityVisitor

SCHEMA_VERSION = 1
COMPLEXITY_CONVENTION = (
    "ComplexityVisitor de Radon sobre la región de coverage.py: if, elif, expresión condicional, "
    "for y while con su else, except, else de try, assert, cada operando booleano después del "
    "primero y cada cláusula de comprensión suman uno. Las funciones anidadas son regiones "
    "propias."
)
COVERAGE_CONVENTION = "Sentencias ejecutadas entre sentencias de la región según coverage.py."
CRAP_CONVENTION = "C² × (1 − cobertura de sentencias)³ + C."
LIMITS = [
    "CRAP combina complejidad y recorridos observados. No demuestra ausencia de errores.",
    "La cobertura de sentencias no distingue qué operandos de una condición se evaluaron, "
    "aunque la complejidad sí los cuenta.",
    "Como en Radon 6.0.1, los manejadores de except* y la condición de un assert no suman, y "
    "el cuerpo de una clase definida dentro de una función no suma en ella aunque "
    "coverage.py cuente allí sus sentencias.",
    "Una función sin sentencias medibles (solo documentación o puntos suspensivos) conserva C.",
    "Si dos funciones comparten nombre cualificado en un archivo, coverage.py solo mide la "
    "última. Las anteriores aparecen en shadowed_regions sin CRAP.",
    "La fuente se contrasta con la medida por sus regiones y sus líneas de sentencia. Un "
    "cambio que las conserva, como otra condición en la misma línea, no se detecta, así que "
    "el resumen se ejecuta sobre el mismo árbol que se midió.",
]
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def complexity(function):
    """Complejidad de Radon de la región propia de una función, sin sus funciones ni clases
    anidadas, que Radon cuenta aparte."""
    if not isinstance(function, _FUNCTIONS):
        raise TypeError("Se esperaba una definición de función")
    return ComplexityVisitor.from_ast(function).functions[0].complexity


def regions(source):
    """Funciones con el nombre cualificado y la línea `def` de las regiones de coverage.py."""
    tree = ast.parse(source)
    nodes = {node.lineno: node for node in ast.walk(tree) if isinstance(node, _FUNCTIONS)}
    found = []
    for region in code_regions(source):
        if region.kind != "function":
            continue
        node = nodes[region.start]
        found.append(
            {
                "function": region.name,
                "start_line": region.start,
                "end_line": node.end_lineno,
                "complexity": complexity(node),
            }
        )
    return found


def statement_lines(source, excluded):
    """Líneas de sentencia de la fuente actual según coverage.py, sin las excluidas al medir."""
    parser = PythonParser(text=source)
    parser.parse_source()
    return set(parser.statements) - set(excluded)


def crap(complexity_, fraction):
    return complexity_**2 * (1 - fraction) ** 3 + complexity_


def _selected(path, include):
    return not include or any(
        fnmatch.fnmatch(path, pattern)
        or path == pattern.rstrip("/")
        or path.startswith(pattern.rstrip("/") + "/")
        for pattern in include
    )


def summarize(coverage, root, *, include=(), require_branches=False):
    """Unir el JSON de coverage.py con las regiones de las fuentes actuales."""
    meta = coverage.get("meta", {})
    if meta.get("format") != 3:
        raise ValueError("Se necesita el formato 3 del JSON de coverage.py")
    if require_branches and not meta.get("branch_coverage"):
        raise ValueError("La medida no registró ramas y se pidieron")
    files, functions, shadowed = [], [], []
    for path, data in sorted(coverage.get("files", {}).items()):
        if not _selected(path, include):
            continue
        if "functions" not in data:
            raise ValueError("El JSON de coverage.py no trae regiones de función (versión 7.6+)")
        source_path = Path(path) if Path(path).is_absolute() else root / path
        source = source_path.read_bytes()
        measured = {name: item for name, item in data["functions"].items() if name}
        text = source.decode("utf-8")
        found = regions(text)
        # coverage.py guarda una región por nombre y se queda con la última definición, como
        # un setter de propiedad o una función redefinida. Las anteriores se declaran aparte.
        current = {item["function"]: item for item in found}
        shadowed.extend(
            {"path": path, "function": item["function"], "start_line": item["start_line"]}
            for item in found
            if current[item["function"]] is not item
        )
        for name, item in measured.items():
            region = current.get(name)
            if region is None or region["start_line"] != item.get("start_line"):
                raise ValueError(
                    f"{path}: {name} no coincide con la fuente actual. "
                    "La fuente cambió desde la medida"
                )
        summary = data["summary"]
        files.append(
            {
                "path": path,
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "statements": summary["num_statements"],
                "covered_statements": summary["covered_lines"],
                "branches": summary.get("num_branches", 0),
                "covered_branches": summary.get("covered_branches", 0),
            }
        )
        for name, region in current.items():
            item = measured.get(name)
            if item is None:
                raise ValueError(f"{path}: coverage.py no registra la región de {name}")
            counts = item["summary"]
            statements = counts["num_statements"]
            fraction = counts["covered_lines"] / statements if statements else None
            functions.append(
                {
                    "path": path,
                    "function": name,
                    "start_line": region["start_line"],
                    "end_line": region["end_line"],
                    "cyclomatic_complexity": region["complexity"],
                    "statements": statements,
                    "covered_statements": counts["covered_lines"],
                    "coverage_fraction": fraction,
                    "branches": counts.get("num_branches", 0),
                    "covered_branches": counts.get("covered_branches", 0),
                    "crap": crap(region["complexity"], 1.0 if fraction is None else fraction),
                }
            )
        # Las regiones pueden conservar su línea aunque cambie el cuerpo de una función.
        lines = set(data["executed_lines"]) | set(data["missing_lines"])
        if statement_lines(text, data["excluded_lines"]) != lines:
            raise ValueError(
                f"{path}: las sentencias medidas no son las de la fuente actual. "
                "La fuente cambió desde la medida"
            )
    if not files:
        raise ValueError("Ninguna fuente medida coincide con la selección")
    functions.sort(key=lambda item: (-item["crap"], item["path"], item["start_line"]))
    totals = {
        key: sum(item[key] for item in files)
        for key in ("statements", "covered_statements", "branches", "covered_branches")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "measured_at": meta.get("timestamp"),
        "tools": {
            "coverage": meta.get("version"),
            "radon": metadata.version("radon"),
            "branch_coverage": bool(meta.get("branch_coverage")),
            "python": platform.python_version(),
            "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "complexity_convention": COMPLEXITY_CONVENTION,
        "coverage_convention": COVERAGE_CONVENTION,
        "crap_convention": CRAP_CONVENTION,
        "limits": LIMITS,
        "include": list(include),
        "totals": totals,
        "files": files,
        "functions": functions,
        "shadowed_regions": shadowed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True, help="JSON de coverage.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Raíz desde la que se resuelven las rutas relativas del JSON",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Archivo, carpeta o patrón de las fuentes que se resumen",
    )
    parser.add_argument("--require-branches", action="store_true")
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Conservar solo las N funciones con mayor CRAP (0 conserva todas)",
    )
    args = parser.parse_args(argv)
    raw = args.coverage.read_bytes()
    report = summarize(
        json.loads(raw),
        args.root.resolve(),
        include=args.include,
        require_branches=args.require_branches,
    )
    report["coverage_sha256"] = hashlib.sha256(raw).hexdigest()
    report["functions_measured"] = len(report["functions"])
    if args.top:
        report["functions"] = report["functions"][: args.top]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps({"functions": report["functions_measured"], **report["totals"]}))


if __name__ == "__main__":
    sys.exit(main())
