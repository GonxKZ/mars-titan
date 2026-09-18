"""Comprobar documentación, bibliografía y planificación sin acceder a datos privados."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import bibtexparser
import yaml
from bibtexparser.bparser import BibTexParser
from reference_library import read_catalogs

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    paths = subprocess.check_output(
        ["rg", "--files", "--hidden", "-g", "!.git"], cwd=ROOT, text=True
    ).splitlines()
    errors = []
    for name in paths:
        path = ROOT / name
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"Archivo versionable mayor de 5 MiB: {name}")
        if path.suffix == ".pdf":
            errors.append(f"PDF fuera de las carpetas locales excluidas: {name}")
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix in {".yml", ".yaml", ".cff"}:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        elif path.suffix == ".toml" or path.name == "uv.lock":
            tomllib.loads(path.read_text(encoding="utf-8"))

    source_manifest = json.loads(
        (ROOT / "docs/academic/source-manifest.json").read_text(encoding="utf-8")
    )
    local_sources = {ROOT / entry["path"] for entry in source_manifest["files"]}
    for name in paths:
        if not name.endswith(".md"):
            continue
        path = ROOT / name
        content = path.read_text(encoding="utf-8")
        prose = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", prose):
            target = urlsplit(link.strip("<>"))
            if target.scheme or target.netloc or not target.path:
                continue
            resolved = (path.parent / unquote(target.path)).resolve()
            if resolved not in local_sources and not resolved.exists():
                errors.append(f"Enlace local inexistente en {name}: {link}")

    reference_dir = ROOT / "docs/references"
    registry = json.loads((reference_dir / "catalogs.json").read_text(encoding="utf-8"))
    entries = read_catalogs([reference_dir / entry["sources"] for entry in registry])
    bibliography = []
    for catalog in registry:
        parser = BibTexParser()
        parser.ignore_nonstandard_types = False
        bibliography.extend(
            bibtexparser.loads(
                (reference_dir / catalog["bibliography"]).read_text(encoding="utf-8"), parser
            ).entries
        )
    bib_ids = [entry["ID"] for entry in bibliography]
    source_ids = {entry["id"] for entry in entries}
    if len(bib_ids) != len(set(bib_ids)) or set(bib_ids) != source_ids:
        errors.append("Los identificadores BibTeX y los catálogos no coinciden sin duplicados")
    for entry in entries:
        required = {"id", "title", "authors", "url", "access", "redistribution", "phase"}
        if not required <= entry.keys() or not set(entry["phase"]) <= set(range(1, 7)):
            errors.append(f"Referencia incompleta o fase inválida: {entry['id']}")
        if not entry["url"].startswith("https://"):
            errors.append(f"Referencia sin HTTPS: {entry['id']}")

    plan = json.loads((ROOT / ".github/planning/issues.json").read_text(encoding="utf-8"))
    issues = {issue["id"]: issue for issue in plan["issues"]}
    if len(issues) != len(plan["issues"]):
        errors.append("Identificadores de tareas duplicados")
    labels = {label["name"] for label in plan["labels"]}
    milestones = {milestone["key"] for milestone in plan["milestones"]}
    visited, active = set(), set()

    def visit(identifier: str) -> None:
        if identifier in active:
            raise ValueError(f"Ciclo de dependencias en {identifier}")
        if identifier in visited:
            return
        if identifier not in issues:
            raise ValueError(f"Dependencia inexistente: {identifier}")
        active.add(identifier)
        for dependency in issues[identifier]["dependencies"]:
            visit(dependency)
        active.remove(identifier)
        visited.add(identifier)

    for identifier, issue in issues.items():
        visit(identifier)
        for heading in ("Contexto", "Trabajo delimitado", "Criterios de aceptación"):
            if f"## {heading}\n\n" not in issue["body"]:
                errors.append(f"Sección Markdown mal delimitada en {identifier}: {heading}")
        if not set(issue["labels"]) <= labels:
            errors.append(f"Etiquetas no definidas en {identifier}")
        if issue["milestone"] is not None and issue["milestone"] not in milestones:
            errors.append(f"Hito no definido en {identifier}")

    with (ROOT / "data/catalogs/macro-indicators.csv").open(encoding="utf-8", newline="") as stream:
        macro_rows = list(csv.DictReader(stream))
    macro = {row["id"]: row for row in macro_rows}
    if len(macro) != len(macro_rows):
        errors.append("Identificadores macroeconómicos duplicados")
    macro_visited, macro_active = set(), set()

    def visit_macro(identifier: str) -> None:
        if identifier in macro_active:
            raise ValueError(f"Ciclo de fórmulas macroeconómicas en {identifier}")
        if identifier in macro_visited:
            return
        if identifier not in macro:
            raise ValueError(f"Dependencia macroeconómica inexistente: {identifier}")
        macro_active.add(identifier)
        row = macro[identifier]
        for dependency in filter(None, row["input_ids"].split("|")):
            visit_macro(dependency)
            if macro[dependency]["verification_status"] == "provider_verified_identifier_pending":
                errors.append(f"Derivado {identifier} depende de una serie no identificada")
        if row["kind"] == "derived" and (not row["formula"] or not row["input_ids"]):
            errors.append(f"Derivado sin fórmula o entradas: {identifier}")
        macro_active.remove(identifier)
        macro_visited.add(identifier)

    for identifier in macro:
        visit_macro(identifier)

    for name in ("README.md", "LICENSE", "CITATION.cff", "AGENTS.md", "uv.lock"):
        if not (ROOT / name).is_file():
            errors.append(f"Falta {name}")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Verificados {len(paths)} archivos, {len(entries)} referencias y {len(issues)} tareas.")
    print(f"Catálogo macroeconómico: {len(macro_rows)} entradas con dependencias válidas.")
    print("Enlaces locales, formatos, BibTeX y dependencias de tareas correctos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
