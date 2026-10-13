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

# Documentos de entrada cuyas órdenes deben poder ejecutarse en un clon limpio.
COMMAND_DOCUMENTS = (
    "README.md",
    "CONTRIBUTING.md",
    "tests/README.md",
    "scripts/README.md",
    "native/README.md",
    "site/README.md",
    "docs/engineering/environment-variables.md",
)
ENVIRONMENT_TABLE = "docs/engineering/environment-variables.md"
CODE_DIRECTORIES = ("src", "scripts", "tests", "benchmarks", "native", "configs")
CODE_SUFFIXES = {".py", ".c", ".h", ".cpp", ".hpp", ".cu", ".cmake", ".json", ".service", ".sh"}
# Prefijo del proyecto sin el final que añade un formato, como en f"MARS_TITAN_{name}".
VARIABLE = re.compile(r"MARS_TITAN_[A-Z0-9_]*[A-Z0-9](?![A-Za-z0-9_{])")
MAKE_TARGET = re.compile(r"(?<![\w-])make ([a-z][a-z-]*)")
SCRIPT_PATH = re.compile(r"(?<![\w/])((?:scripts|tests)/[\w./-]+\.py)\b")
PRESET = re.compile(r"--preset ([\w-]+)")
MODULE = re.compile(r"python -m (mars_titan(?:\.\w+)+)")


def code_spans(content: str) -> str:
    """Texto de los bloques y fragmentos de código de un documento Markdown."""
    blocks = re.findall(r"```.*?```", content, flags=re.DOTALL)
    inline = re.findall(r"`([^`\n]+)`", re.sub(r"```.*?```", "", content, flags=re.DOTALL))
    return "\n".join(blocks + inline)


def command_problems(root: Path) -> list[str]:
    """Órdenes documentadas que apuntan a objetivos, guiones, módulos o presets inexistentes."""
    makefile = root / "Makefile"
    targets = set()
    if makefile.is_file():
        targets = set(re.findall(r"^([a-z][a-z-]*):", makefile.read_text(encoding="utf-8"), re.M))
    presets_file = root / "native/CMakePresets.json"
    presets = set()
    if presets_file.is_file():
        document = json.loads(presets_file.read_text(encoding="utf-8"))
        for kind in ("configurePresets", "buildPresets", "testPresets"):
            presets |= {preset["name"] for preset in document.get(kind, [])}
    problems = []
    for name in COMMAND_DOCUMENTS:
        path = root / name
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        code = code_spans(content)
        for target in MAKE_TARGET.findall(code):
            if target not in targets:
                problems.append(f"Objetivo de make inexistente en {name}: {target}")
        for script in SCRIPT_PATH.findall(content):
            if not (root / script).is_file():
                problems.append(f"Guion inexistente en {name}: {script}")
        for preset in PRESET.findall(code):
            if preset not in presets:
                problems.append(f"Preset de CMake inexistente en {name}: {preset}")
        for module in MODULE.findall(code):
            stem = root / "src" / Path(*module.split("."))
            if not stem.with_suffix(".py").is_file() and not (stem / "__main__.py").is_file():
                problems.append(f"Módulo inexistente en {name}: {module}")
    return problems


def versioned_files(root: Path) -> list[str]:
    """Archivos que se publicarían, sin los ignorados por Git ni el propio `.git`."""
    return subprocess.check_output(
        ["rg", "--files", "--hidden", "-g", "!.git"], cwd=root, text=True
    ).splitlines()


def code_files(paths: list[str]) -> list[str]:
    return sorted(name for name in paths if name.split("/", 1)[0] in CODE_DIRECTORIES)


def code_variables(root: Path, paths: list[str]) -> set[str]:
    """Variables de entorno del proyecto que lee el código.

    Son los nombres con el prefijo del proyecto que aparecen en el código, salvo las variables
    de la caché de CMake y las macros de C y C++, que comparten el prefijo.
    """
    names, build = set(), set()
    for name in code_files(paths):
        path = root / name
        cmake = path.suffix == ".cmake" or path.name in {"CMakeLists.txt", "CMakePresets.json"}
        if not cmake and path.suffix not in CODE_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if cmake:
            build |= set(VARIABLE.findall(text))
        elif path.suffix in {".c", ".h", ".cpp", ".hpp", ".cu"}:
            for line in text.splitlines():
                (build if line.lstrip().startswith("#") else names).update(VARIABLE.findall(line))
        else:
            names |= set(VARIABLE.findall(text))
    return names - build


def environment_problems(root: Path, paths: list[str]) -> list[str]:
    """Variables leídas que faltan en la tabla única y variables de la tabla sin uso."""
    used = code_variables(root, paths)
    table = root / ENVIRONMENT_TABLE
    if not table.is_file():
        return [f"Falta {ENVIRONMENT_TABLE}"] if used else []
    documented = set(VARIABLE.findall(table.read_text(encoding="utf-8")))
    every = set()
    for name in code_files(paths):
        every |= set(VARIABLE.findall((root / name).read_text(encoding="utf-8", errors="replace")))
    problems = [
        f"Variable de entorno sin documentar en {ENVIRONMENT_TABLE}: {name}"
        for name in sorted(used - documented)
    ]
    problems += [
        f"Variable documentada que ningún código usa: {name}" for name in sorted(documented - every)
    ]
    return problems


# Calcos del inglés que no se usan en la prosa española del proyecto, con su sustituto.
VOCABULARY = (
    (re.compile(r"\bbrazos?\b", re.IGNORECASE), "modelo, variante o configuración"),
    (
        re.compile(r"\bor[aá]culos?\b", re.IGNORECASE),
        "referencia, óptimo exacto o cota con información futura",
    ),
    (re.compile(r"\bguionizad[oa]s?\b", re.IGNORECASE), "de reglas fijas"),
    (re.compile(r"\bcanarios?\b", re.IGNORECASE), "información futura inyectada a propósito"),
)
# Evidencia generada, copias entregadas, recetas cuya huella fijan las pruebas y este control.
VOCABULARY_EXCEPTIONS = (
    re.compile(r"^reports/.*\.json$"),
    re.compile(r"^thesis/revisions/"),
    re.compile(r"^configs/titans/chronological-training(-quantile)?\.json$"),
    re.compile(r"^scripts/check_repository\.py$"),
    re.compile(r"^tests/tooling/test_repository_check\.py$"),
)
TEXT_SUFFIXES = set(
    ".md .tex .py .mjs .js .html .css .json .toml .yml .yaml .cff .txt .sh".split()
    + ".cpp .hpp .h .cu .cuh .cmake".split()
)


def vocabulary_errors(name: str, text: str) -> list[str]:
    """Señala cada calco con su línea y el término que lo sustituye."""
    errors = []
    for number, line in enumerate(text.splitlines(), start=1):
        for pattern, replacement in VOCABULARY:
            for match in pattern.finditer(line):
                errors.append(f"Término «{match.group(0)}» en {name}:{number}. Usar {replacement}")
    return errors


def main() -> int:
    paths = versioned_files(ROOT)
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
        if path.suffix in TEXT_SUFFIXES and not any(
            pattern.search(name) for pattern in VOCABULARY_EXCEPTIONS
        ):
            errors.extend(vocabulary_errors(name, path.read_text(encoding="utf-8")))

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
            if not resolved.exists():
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

    errors.extend(command_problems(ROOT))
    errors.extend(environment_problems(ROOT, paths))

    for name in ("README.md", "LICENSE", "CITATION.cff", "uv.lock"):
        if not (ROOT / name).is_file():
            errors.append(f"Falta {name}")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Verificados {len(paths)} archivos, {len(entries)} referencias y {len(issues)} tareas.")
    print(f"Catálogo macroeconómico: {len(macro_rows)} entradas con dependencias válidas.")
    print("Enlaces locales, formatos, BibTeX y dependencias de tareas correctos.")
    print("Órdenes documentadas y tabla de variables de entorno coherentes con el código.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
