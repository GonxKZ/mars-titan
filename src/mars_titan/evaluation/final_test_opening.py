"""Procedimiento de apertura única del test final de 2024, declarado y comprobable.

El orden declarado (``steps``) es congelar, comprobar, confirmar, registrar la apertura,
preparar los objetivos de 2024, predecir con los estados congelados, evaluar la
comparación declarada y publicar el registro. Este módulo implementa congelar,
comprobar, confirmar, registrar y publicar. Los tres pasos que leen 2024 forman el
*abridor*, que no está conectado: nada en este repositorio prepara objetivos de 2024 ni
lee sus filas, y la orden ``open`` se niega antes de crear ningún registro.

Congelar escribe en un directorio de estado privado un manifiesto con la huella de la
comparación declarada y de sus protocolos, de cada artefacto congelado (registro de
ensayos, exclusiones, política de memoria, recibos de selección y edición de datos), el
número de ensayos, las familias confirmatorias (las demás quedan como exploratorias) y
el commit de un repositorio limpio. Se puede volver a congelar mientras no haya
apertura. La huella del manifiesto es la confirmación que exige la apertura.

Las condiciones se comprueban en el orden declarado y deben cumplirse todas. La apertura
crea el registro privado con creación exclusiva antes de llamar al abridor, de modo que
un fallo posterior también consume la única apertura. El registro del repositorio
(``repository_record``) se escribe al terminar, con éxito o con fallo, y su presencia
también impide abrir. Borrar uno de los dos no basta para repetir.
"""

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.learning_hold import hold_path, learning_blocked

from . import walk_forward_comparison as walk

KIND = "final_test_opening_procedure"
FROZEN_KIND = "final_test_frozen_manifest"
RECORD_KIND = "final_test_opening_record"
FINAL_TEST = {"start": walk.FINAL_TEST_START, "end": "2025-01-01"}
ARTEFACTS = ("trial_registry", "exclusions", "memory_policy", "selection_receipts", "data_edition")
STEPS = (
    "freeze",
    "check",
    "confirm",
    "record_opening",
    "prepare_final_targets",
    "predict_with_frozen_states",
    "evaluate_declared_comparison",
    "publish_record",
)
CONDITIONS = (
    "frozen_manifest_unchanged",
    "repository_clean_at_frozen_commit",
    "learning_hold_lifted",
    "no_previous_opening",
)
REPEAT = "never_even_after_a_failed_opening"
FROZEN = "frozen.json"
_FIELDS = {
    "schema_version",
    "kind",
    "status",
    "declared_at",
    "final_test",
    "comparison",
    "frozen_artefacts",
    "steps",
    "conditions",
    "ledger",
    "repository_record",
    "repeat",
    "opener",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _now():
    return datetime.now(UTC).isoformat()


def load_procedure(path):
    """Validar la declaración y su comparación sin leer predicciones ni datos de 2024."""
    path = Path(path)
    procedure, digest = read_manifest(path, 1024**2)
    record = procedure.get("repository_record") if isinstance(procedure, dict) else None
    _require(
        isinstance(procedure, dict)
        and set(procedure) == _FIELDS
        and procedure["schema_version"] == 1
        and procedure["kind"] == KIND
        and procedure["status"] == "declared_not_executed"
        and procedure["final_test"] == FINAL_TEST
        and procedure["frozen_artefacts"] == list(ARTEFACTS)
        and procedure["steps"] == list(STEPS)
        and procedure["conditions"] == list(CONDITIONS)
        and procedure["repeat"] == REPEAT
        and procedure["opener"] == "not_connected"
        and isinstance(procedure["ledger"], str)
        and procedure["ledger"].endswith(".json")
        and PurePosixPath(procedure["ledger"]).name == procedure["ledger"]
        and isinstance(record, str)
        and not PurePosixPath(record).is_absolute()
        and ".." not in PurePosixPath(record).parts,
        "El procedimiento de apertura del test final no cumple su contrato",
    )
    comparison = walk.load_config(path.parent / procedure["comparison"])
    return dict(procedure, sha256=digest, comparison=comparison)


def _git(repository, *args):
    return subprocess.run(
        ["git", "-C", str(repository), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _repository(repository):
    """Commit actual y si el árbol de trabajo está limpio, con archivos no versionados."""
    return _git(repository, "rev-parse", "HEAD"), _git(repository, "status", "--porcelain") == ""


def _records(procedure, state, repository):
    return Path(state) / procedure["ledger"], Path(repository) / procedure["repository_record"]


def _opened(procedure, state, repository):
    return any(path.exists() for path in _records(procedure, state, repository))


def _comparison(procedure):
    comparison = procedure["comparison"]
    return dict(
        name=comparison["name"],
        sha256=comparison["sha256"],
        protocol_sha256={
            scope: resolved["protocol_sha256"]
            for scope, resolved in comparison["resolved_scopes"].items()
        },
    )


def freeze(procedure_path, state, *, artefacts, trials, confirmatory, repository):
    """Congelar comparación, artefactos, ensayos, familias y commit antes de abrir."""
    procedure = load_procedure(procedure_path)
    _require(
        not _opened(procedure, state, repository),
        "El test final ya se abrió: no se puede volver a congelar",
    )
    _require(
        isinstance(artefacts, dict) and set(artefacts) == set(ARTEFACTS),
        f"Hay que congelar exactamente {', '.join(ARTEFACTS)}",
    )
    paths = {name: Path(value).resolve() for name, value in artefacts.items()}
    _require(all(path.is_file() for path in paths.values()), "Cada artefacto debe ser un archivo")
    _require(type(trials) is int and trials >= 1, "El número de ensayos debe ser positivo")
    families = list(procedure["comparison"]["resolved_families"])
    _require(
        isinstance(confirmatory, list)
        and confirmatory
        and len(set(confirmatory)) == len(confirmatory)
        and all(name in families for name in confirmatory),
        "Las familias confirmatorias deben ser familias distintas de la comparación",
    )
    commit, clean = _repository(repository)
    _require(clean, "El repositorio debe estar limpio al congelar")
    manifest = dict(
        schema_version=1,
        kind=FROZEN_KIND,
        procedure_sha256=procedure["sha256"],
        comparison=_comparison(procedure),
        artefacts={
            name: dict(path=str(paths[name]), sha256=sha256(paths[name])) for name in ARTEFACTS
        },
        trials=trials,
        confirmatory_families=confirmatory,
        exploratory_families=[name for name in families if name not in confirmatory],
        commit=commit,
        final_test_opened=False,
        frozen_at_utc=_now(),
    )
    destination = Path(state) / FROZEN
    atomic_json(destination, manifest)
    return sha256(destination)


def _frozen_problem(procedure, frozen):
    """Motivo por el que el manifiesto congelado no vale, o None si sigue intacto."""
    if frozen is None:
        return "No hay manifiesto congelado"
    if frozen.get("kind") != FROZEN_KIND or frozen.get("procedure_sha256") != procedure["sha256"]:
        return "El manifiesto congelado pertenece a otro procedimiento"
    if frozen.get("comparison") != _comparison(procedure):
        return "La comparación o sus protocolos cambiaron después de congelar"
    changed = [
        name
        for name, record in frozen["artefacts"].items()
        if not Path(record["path"]).is_file() or sha256(Path(record["path"])) != record["sha256"]
    ]
    if changed:
        return f"Cambiaron artefactos congelados: {', '.join(changed)}"
    return None


def _state(procedure, state, repository):
    """Manifiesto congelado, su huella y cada condición en el orden declarado."""
    path = Path(state) / FROZEN
    frozen, token = read_manifest(path, 8 * 1024**2) if path.is_file() else (None, None)
    commit, clean = _repository(repository)
    problem = _frozen_problem(procedure, frozen)
    at_commit = frozen is not None and frozen.get("commit") == commit
    blocked = learning_blocked()
    opened = _opened(procedure, state, repository)
    details = {
        "frozen_manifest_unchanged": problem,
        "repository_clean_at_frozen_commit": None
        if clean and at_commit
        else f"El repositorio está en {commit} {'limpio' if clean else 'con cambios'} y se "
        f"congeló en {frozen.get('commit') if frozen else 'ningún commit'}",
        "learning_hold_lifted": f"La protección {hold_path()} sigue vigente" if blocked else None,
        "no_previous_opening": "Ya existe un registro de apertura" if opened else None,
    }
    conditions = [
        dict(condition=name, satisfied=details[name] is None, detail=details[name])
        for name in procedure["conditions"]
    ]
    return frozen, token, commit, conditions


def check(procedure_path, state, *, repository):
    """Comprobar las condiciones sin abrir nada ni escribir ningún registro."""
    procedure = load_procedure(procedure_path)
    _, token, _, conditions = _state(procedure, state, repository)
    return dict(
        ready=all(item["satisfied"] for item in conditions),
        frozen_sha256=token,
        conditions=conditions,
        steps=procedure["steps"],
        opener=procedure["opener"],
        final_test_opened=_opened(procedure, state, repository),
    )


def open_final_test(procedure_path, state, *, confirm, repository, opener=None):
    """Abrir una sola vez: condiciones, confirmación, registro exclusivo y abridor.

    ``opener`` recibe el manifiesto congelado y devuelve los artefactos que produce
    (nombre y ruta). Sin abridor conectado la apertura se niega antes de registrar nada.
    """
    procedure = load_procedure(procedure_path)
    frozen, token, commit, conditions = _state(procedure, state, repository)
    failed = [item["condition"] for item in conditions if not item["satisfied"]]
    _require(not failed, f"No se cumplen las condiciones: {', '.join(failed)}")
    _require(confirm == token, "La confirmación no coincide con la huella del manifiesto congelado")
    _require(
        callable(opener),
        "El abridor del test de 2024 no está conectado: ningún código prepara sus objetivos",
    )
    ledger, published = _records(procedure, state, repository)
    record = dict(
        schema_version=1,
        kind=RECORD_KIND,
        status="opening",
        procedure_sha256=procedure["sha256"],
        frozen_sha256=token,
        commit=commit,
        final_test_opened=True,
        started_at_utc=_now(),
    )
    # La creación exclusiva es el punto sin retorno: a partir de aquí no hay otra apertura.
    descriptor = os.open(ledger, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        outputs = opener(frozen)
        record.update(
            status="opened",
            outputs={
                name: dict(path=str(path), sha256=sha256(Path(path)))
                for name, path in outputs.items()
            },
        )
    except BaseException as error:
        record.update(status="failed", error=repr(error)[:2000])
        raise
    finally:
        record["finished_at_utc"] = _now()
        atomic_json(ledger, record)
        atomic_json(published, record)
    return record


def _artefacts(values):
    artefacts = {}
    for value in values or []:
        name, _, path = value.partition("=")
        _require(name in ARTEFACTS and path, "Usa --artefact NOMBRE=RUTA")
        _require(name not in artefacts, f"El artefacto {name} aparece dos veces")
        artefacts[name] = Path(path)
    return artefacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    parsers = {
        name: commands.add_parser(name, help=text)
        for name, text in (
            ("check", "Comprobar las condiciones sin abrir"),
            ("freeze", "Congelar comparación, artefactos y commit"),
            ("open", "Abrir una sola vez con la confirmación"),
        )
    }
    for command in parsers.values():
        command.add_argument("--procedure", type=Path, required=True)
        command.add_argument("--state", type=Path, required=True)
        command.add_argument("--repository", type=Path, default=Path.cwd())
    parsers["freeze"].add_argument("--artefact", action="append")
    parsers["freeze"].add_argument("--trials", type=int, required=True)
    parsers["freeze"].add_argument("--confirmatory", action="append", required=True)
    parsers["open"].add_argument("--confirm", required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        result = check(args.procedure, args.state, repository=args.repository)
    elif args.command == "freeze":
        token = freeze(
            args.procedure,
            args.state,
            artefacts=_artefacts(args.artefact),
            trials=args.trials,
            confirmatory=args.confirmatory,
            repository=args.repository,
        )
        result = dict(frozen_sha256=token, final_test_opened=False)
    else:
        # Sin abridor conectado: la orden comprueba todo y se niega antes de registrar.
        result = open_final_test(
            args.procedure, args.state, confirm=args.confirm, repository=args.repository
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
