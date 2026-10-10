"""Registro previo de comparaciones y de sus desviaciones en una cadena de huellas.

Una comparación solo es creíble si su configuración, sus métricas y sus reglas de decisión
se fijaron antes de ver resultados. El registro es un archivo JSON Lines que solo crece:
- `declaration`: huella SHA-256 de un documento declarado, como la configuración de una
  comparación walk-forward, con su ruta y una nota;
- `deviation`: cambio posterior sobre una declaración, con su motivo y, si existe, la
  huella del documento que la sustituye.

Cada entrada guarda la huella canónica de la anterior, de modo que editar o borrar una
línea rompe la cadena. Las fechas locales las fija el propio autor. La prueba externa es el
commit que añade la entrada, una vez publicado en el remoto antes de crear el informe.
`check_report` comprueba ambas cosas y las informa por separado.
"""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.storage import sha256

KIND = "preregistration_check"
GENESIS = "0" * 64
KINDS = ("declaration", "deviation")
_FIELDS = {
    "declaration": {"subject_path", "subject_sha256", "note"},
    "deviation": {"declaration_sha256", "reason", "replacement_sha256"},
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _entry_sha256(entry):
    body = {key: value for key, value in entry.items() if key != "entry_sha256"}
    return hashlib.sha256(_canonical(body).encode()).hexdigest()


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def validate_chain(entries):
    """Exigir orden, enlaces, huellas, fechas y referencias de una lista de entradas."""
    previous, moment = GENESIS, ""
    for number, entry in enumerate(entries):
        where = f"Entrada {number} del registro"
        kind = entry.get("kind")
        _require(kind in KINDS, f"{where}: tipo desconocido")
        common = {"schema_version", "sequence", "kind", "created_at_utc", "previous_sha256"}
        _require(set(entry) == common | _FIELDS[kind] | {"entry_sha256"}, f"{where}: campos")
        _require(entry["schema_version"] == 1 and entry["sequence"] == number, f"{where}: orden")
        _require(entry["previous_sha256"] == previous, f"{where}: rompe la cadena")
        _require(entry["entry_sha256"] == _entry_sha256(entry), f"{where}: huella alterada")
        _require(entry["created_at_utc"] >= moment, f"{where}: fecha anterior a la previa")
        digests = (
            [entry["subject_sha256"]]
            if kind == "declaration"
            else [entry["declaration_sha256"]]
            + ([entry["replacement_sha256"]] if entry["replacement_sha256"] is not None else [])
        )
        _require(all(_hex(value) for value in digests), f"{where}: huella mal formada")
        previous, moment = entry["entry_sha256"], entry["created_at_utc"]
    declared = [e["subject_sha256"] for e in entries if e["kind"] == "declaration"]
    _require(len(declared) == len(set(declared)), "Un documento declarado dos veces")
    known = {e["entry_sha256"] for e in entries if e["kind"] == "declaration"}
    _require(
        all(e["declaration_sha256"] in known for e in entries if e["kind"] == "deviation"),
        "Una desviación no apunta a una declaración del registro",
    )
    return entries


def read_chain(registry):
    """Leer y validar la cadena completa. Un archivo inexistente es una cadena vacía."""
    registry = Path(registry)
    if not registry.exists():
        return []
    lines = registry.read_text(encoding="utf-8").splitlines()
    return validate_chain([json.loads(line) for line in lines])


def _append(registry, kind, fields, now):
    registry = Path(registry)
    entries = read_chain(registry)
    entry = dict(
        schema_version=1,
        sequence=len(entries),
        kind=kind,
        created_at_utc=(now or datetime.now(UTC)).astimezone(UTC).isoformat(),
        previous_sha256=entries[-1]["entry_sha256"] if entries else GENESIS,
        **fields,
    )
    entry["entry_sha256"] = _entry_sha256(entry)
    # Validar antes de escribir: una entrada inválida nunca llega al archivo.
    validate_chain([*entries, entry])
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as stream:
        stream.write(_canonical(entry) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return entry


def declare(registry, document, *, note, root=None, now=None):
    """Registrar la huella de un documento antes de producir ningún resultado con él."""
    document = Path(document)
    _require(document.is_file() and not document.is_symlink(), "El documento no es un archivo")
    _require(isinstance(note, str) and 0 < len(note) <= 2000, "La nota es obligatoria")
    path = document.resolve()
    if root is not None:
        path = path.relative_to(Path(root).resolve())
    return _append(
        registry,
        "declaration",
        dict(subject_path=str(path), subject_sha256=sha256(document), note=note),
        now,
    )


def deviate(registry, declaration_sha256, *, reason, replacement=None, now=None):
    """Registrar un cambio sobre una declaración, sin borrar la original."""
    _require(isinstance(reason, str) and 0 < len(reason) <= 4000, "El motivo es obligatorio")
    replacement_sha256 = None if replacement is None else sha256(Path(replacement))
    return _append(
        registry,
        "deviation",
        dict(
            declaration_sha256=declaration_sha256,
            reason=reason,
            replacement_sha256=replacement_sha256,
        ),
        now,
    )


def _git(repository, *arguments):
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def commit_evidence(repository, registry, entry_sha256):
    """Primer commit que añade la entrada, su fecha de commit y si está en algún remoto."""
    relative = Path(registry).resolve().relative_to(Path(repository).resolve())
    commits = _git(
        repository, "log", "--format=%H %cI", "--reverse", "-S", entry_sha256, "--", str(relative)
    )
    if not commits:
        return dict(commit=None, committed_at=None, on_remote=False)
    commit, committed_at = commits.splitlines()[0].split(" ", 1)
    remote = _git(repository, "branch", "-r", "--contains", commit)
    return dict(commit=commit, committed_at=committed_at, on_remote=bool(remote))


def _moment(text):
    return datetime.fromisoformat(text).astimezone(UTC)


def check_report(registry, report_path, *, repository=None):
    """Comprobar que la configuración de un informe se declaró antes de crearlo."""
    entries = read_chain(registry)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    configuration = report["configuration"]["sha256"]
    created = _moment(report["created_at_utc"])
    matches = [
        e for e in entries if e["kind"] == "declaration" and e["subject_sha256"] == configuration
    ]
    result = dict(
        schema_version=1,
        kind=KIND,
        report=str(report_path),
        configuration_sha256=configuration,
        report_created_at_utc=created.isoformat(),
        declared=bool(matches),
    )
    if not matches:
        return dict(result, passed=False, failure="La configuración no está declarada")
    declaration = matches[0]
    deviations = [
        dict(reason=e["reason"], replacement_sha256=e["replacement_sha256"], at=e["created_at_utc"])
        for e in entries
        if e["kind"] == "deviation" and e["declaration_sha256"] == declaration["entry_sha256"]
    ]
    local_before = _moment(declaration["created_at_utc"]) < created
    result.update(
        declaration_sha256=declaration["entry_sha256"],
        declared_at_utc=declaration["created_at_utc"],
        declared_before_report=local_before,
        deviations=deviations,
    )
    passed = local_before
    if repository is not None:
        evidence = commit_evidence(repository, registry, declaration["entry_sha256"])
        committed_before = (
            evidence["committed_at"] is not None and _moment(evidence["committed_at"]) < created
        )
        result.update(
            commit=evidence,
            committed_before_report=committed_before,
            published_before_report=committed_before and evidence["on_remote"],
        )
        passed = passed and committed_before
    return dict(result, passed=passed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("declare")
    add.add_argument("registry")
    add.add_argument("document")
    add.add_argument("--note", required=True)
    change = commands.add_parser("deviate")
    change.add_argument("registry")
    change.add_argument("declaration")
    change.add_argument("--reason", required=True)
    change.add_argument("--replacement")
    check = commands.add_parser("check")
    check.add_argument("registry")
    check.add_argument("report")
    check.add_argument("--repository")
    check.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "declare":
        print(
            declare(args.registry, args.document, note=args.note, root=Path.cwd())["entry_sha256"]
        )
        return 0
    if args.command == "deviate":
        entry = deviate(
            args.registry, args.declaration, reason=args.reason, replacement=args.replacement
        )
        print(entry["entry_sha256"])
        return 0
    result = check_report(args.registry, args.report, repository=args.repository)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
