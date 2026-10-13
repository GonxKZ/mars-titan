"""Recalcular con el resumen común la complejidad que registraron los recibos anteriores.

Recorre los JSON de `reports/` y reúne cada función Python con su complejidad registrada
(`ccn`, `complexity`, `cyclomatic_complexity` o `cc`). Si la función no nombra su archivo,
se toma el `scope` del recibo cuando es un único archivo Python. Para cada una toma la fuente
del commit más antiguo con el contenido actual del recibo y la compara con `complexity` de
`scripts/summarize_python_quality.py`. Una discrepancia se acompaña de la huella de la
fuente en esa revisión y de la que registró el recibo, si la registró. Cada fila conserva la
herramienta que declaró el recibo (Radon o Lizard).

Uso, desde la raíz del repositorio:

    uv run python reports/engineering/python-quality-20261010/radon_receipts.py \
        --output reports/engineering/python-quality-20261010/radon-receipts.json
"""

import argparse
import ast
import functools
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_python_quality as quality  # noqa: E402

FIELDS = ("ccn", "complexity", "cyclomatic_complexity", "cc")


def git(*arguments):
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments], capture_output=True, text=True, check=False
    ).stdout


@functools.cache
def authored(receipt):
    """Commit más antiguo que introduce el contenido actual del recibo.

    El último commit que toca el archivo no sirve: las publicaciones en main vuelven a
    escribir todos los recibos con fuentes posteriores a su medida.
    """
    blob = git("rev-parse", f"HEAD:{receipt}").strip()
    found = git("log", "--reverse", "--format=%H", f"--find-object={blob}", "--", receipt)
    return found.split()[0]


def recorded(node, receipt, rows, hashes, scope=None):
    if isinstance(node, dict):
        path = node.get("file") or node.get("path") or scope
        name = node.get("function") or node.get("name")
        value = next((node[key] for key in FIELDS if type(node.get(key)) is int), None)
        if isinstance(path, str) and path.endswith(".py") and isinstance(name, str):
            if value is not None:
                line = node.get("line") or node.get("lineno") or node.get("start_line")
                rows.append(dict(receipt=receipt, file=path, function=name, ccn=value, line=line))
        for key, value in node.items():
            if key.endswith(".py") and isinstance(value, str | dict):
                digest = value if isinstance(value, str) else value.get("sha256")
                if isinstance(digest, str) and len(digest) == 64:
                    hashes[key] = digest
            recorded(value, receipt, rows, hashes, scope)
    elif isinstance(node, list):
        for value in node:
            recorded(value, receipt, rows, hashes, scope)


def declared_tool(document):
    """Herramienta de complejidad que declara el recibo, por sus claves de herramientas."""
    tools = json.dumps(document.get("tools", {}) if isinstance(document, dict) else {}).lower()
    return "lizard" if "lizard" in tools and "radon" not in tools else "radon"


def definitions(source):
    """Funciones por nombre simple y cualificado, cada nodo una sola vez por nombre."""
    found = {}

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                for key in {f"{prefix}{child.name}", child.name}:
                    found.setdefault(key, []).append(child)
                visit(child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)

    visit(ast.parse(source), "")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, hashes = [], {}
    for path in sorted((ROOT / "reports").rglob("*.json")):
        if path.resolve() == args.output.resolve():
            # La salida de una ejecución anterior no es un recibo.
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        receipt = str(path.relative_to(ROOT))
        scope = document.get("scope") if isinstance(document, dict) else None
        scope = scope if isinstance(scope, str) and scope.endswith(".py") else None
        found, start = {}, len(rows)
        recorded(document, receipt, rows, found, scope)
        for row in rows[start:]:
            row["tool"] = declared_tool(document)
        hashes[receipt] = found
    sources = {}
    for row in rows:
        commit = authored(row["receipt"])
        if "/" not in row["file"]:
            # Algunos recibos solo nombran el archivo. Se resuelve si es único en esa revisión.
            matches = [
                name
                for name in git("ls-tree", "-r", "--name-only", commit).splitlines()
                if name.rsplit("/", 1)[-1] == row["file"]
            ]
            if len(matches) == 1:
                row["resolved_file"] = matches[0]
        key = commit, row.get("resolved_file", row["file"])
        if key not in sources:
            text = git("show", f"{commit}:{key[1]}")
            sources[key] = (
                (definitions(text), hashlib.sha256(text.encode()).hexdigest()) if text else None
            )
        row["commit"] = commit
        if sources[key] is None:
            row["status"] = "source_missing"
            continue
        found, digest = sources[key]
        candidates = found.get(row["function"]) or found.get(row["function"].split(".")[-1], [])
        if row["line"]:
            candidates = [node for node in candidates if node.lineno == row["line"]] or candidates
        if len(candidates) != 1:
            row["status"] = "ambiguous" if candidates else "not_found"
            # Sin línea no se sabe cuál registró el recibo. Se guardan todas las candidatas.
            row["candidates"] = sorted(quality.complexity(node) for node in candidates)
            continue
        row["summary"] = quality.complexity(candidates[0])
        row["status"] = "equal" if row["summary"] == row["ccn"] else "different"
        if row["status"] == "different":
            row["source_sha256_at_commit"] = digest
            row["source_sha256_recorded"] = hashes[row["receipt"]].get(row["file"])
    summary = dict(
        receipts=len({row["receipt"] for row in rows}),
        functions=len(rows),
        status=dict(sorted(Counter(row["status"] for row in rows).items())),
        status_by_tool={
            tool: dict(
                sorted(Counter(row["status"] for row in rows if row["tool"] == tool).items())
            )
            for tool in sorted({row["tool"] for row in rows})
        },
        summarizer_sha256=hashlib.sha256(
            (ROOT / "scripts/summarize_python_quality.py").read_bytes()
        ).hexdigest(),
    )
    args.output.write_text(
        json.dumps(dict(summary=summary, functions=rows), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
