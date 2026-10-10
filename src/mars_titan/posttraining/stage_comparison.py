"""Comparar los brazos postentrenados de cada padre con su padre congelado y su continuación.

La declaración (``posttraining_stage_comparison``) se fija antes de evaluar y enlaza la
etapa de postentrenamiento. La comparación se deriva del plan de la etapa, sin listar
brazos a mano. Por cada brazo base y ámbito hay una comparación walk-forward con el padre
congelado, la continuación completa (control ``full_continuation`` de la matriz) y los
brazos adaptados de la matriz para la familia del padre.

En el walk-forward por etapas de la variante A, el padre congelado es el trabajo
``frozen`` de la etapa, que aplica a la ventana k el estado elegido por la base en k-1. El
brazo base, reentrenado por la campaña en k, sigue en la comparación como contraste de
nivel y no entra en las familias declaradas. Como la primera ventana de cada ámbito no
tiene postentrenamiento, la comparación solo cubre las ventanas con trabajos de la etapa.
En un plan sin padre congelado, como el anclado de B, el padre congelado es el propio
brazo base con las predicciones de la campaña.

Así una familia nueva de la matriz entra sin reescribir la declaración. Cada comparación
hereda de la comparación de la campaña la política, los protocolos, las métricas, la
calibración común, el remuestreo y las secciones secundarias. Solo cambian los brazos y
las familias de contrastes, que salen de los papeles declarados. La corrección por
máximo estudentizado se aplica dentro de cada familia, como en la comparación principal.
No hay corrección entre padres: cada padre es un análisis secundario propio.

El manifiesto de fuentes de un padre y un ámbito une las predicciones del padre elegido,
leídas del manifiesto de fuentes ya validado de la campaña, con los recibos confirmados de
la etapa. Antes de escribirlo se comprueba que la etapa corresponde a esta declaración y
usó las mismas vistas. La comparación exige después las mismas filas y objetivos en todos
los brazos. Nada de este módulo ajusta, carga modelos ni abre la reserva de 2024.
"""

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation import long_short_comparison
from mars_titan.evaluation import walk_forward_comparison as walk

from .campaign_stage import FROZEN as FROZEN_JOB
from .campaign_stage import (
    RECEIPT_KIND,
    RUN_KIND,
    TABULAR,
    _digest,
    load_stage,
    plan_stage,
    stage_arms,
)

KIND = "posttraining_stage_comparison"
DECLARED = "declared_before_evaluation"
USE = "secondary_contrasts_not_for_selecting_the_base_architecture"
FROZEN, CONTINUATION, ADAPTED = "frozen_parent", "full_continuation", "adapted"
ROLES = (FROZEN, CONTINUATION, ADAPTED)
_FIELDS = {"schema_version", "kind", "status", "name", "stage", "use", "families"}
# Familia de cada brazo derivado en la configuración de la comparación.
ARM_FAMILIES = {
    FROZEN: "posttraining_frozen_parent",
    CONTINUATION: "posttraining_control",
    ADAPTED: "posttraining_adapter",
}
# Solo calibración y evaluación entran en la comparación. La validación de los recibos
# por etapas sirve para elegir el predictor de la cadena, no para compararlo.
COMPARED = ("calibration", "evaluation")
# Secciones derivadas de la etapa. El resto se hereda de la comparación de la campaña.
DERIVED = {"name", "arms", "scopes"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _families(families):
    """Validar las familias declaradas por papel: base y variantes distintas."""
    _require(isinstance(families, dict) and families, "Faltan familias de contrastes")
    for name, family in families.items():
        _require(
            walk._name(name)
            and name != "levels"
            and isinstance(family, dict)
            and set(family) == {"base", "variants"}
            and family["base"] in (FROZEN, CONTINUATION)
            and isinstance(family["variants"], list)
            and family["variants"]
            and len(set(family["variants"])) == len(family["variants"])
            and all(role in ROLES for role in family["variants"]),
            f"La familia {name} debe contrastar papeles declarados con una base congelada "
            "o de continuación",
        )
    return families


def _groups(stage):
    """Brazos de cada padre por papel, en el orden del plan, y sus trabajos por ámbito.

    Un trabajo ``frozen`` del plan por etapas fija el brazo del padre congelado. Sin él,
    el padre congelado es el propio brazo base. Ridge y XGBoost quedan fuera: su cadena solo
    tiene el padre congelado y no hay continuación ni adaptadores que contrastar.
    """
    trivial = {arm for arm, spec in stage_arms(stage)[0].items() if spec["design"] == TABULAR}
    groups = {}
    for job in plan_stage(stage):
        if job["base_arm"] in trivial:
            continue
        group = groups.setdefault(
            job["base_arm"], {FROZEN: job["base_arm"], CONTINUATION: None, ADAPTED: [], "jobs": {}}
        )
        control = job["control"]
        _require(
            control in (None, CONTINUATION) or job.get("kind") == FROZEN_JOB,
            f"El control {control} de {job['arm']} no tiene papel en la comparación",
        )
        if job.get("kind") == FROZEN_JOB:
            _require(
                group[FROZEN] in (job["base_arm"], job["arm"]),
                f"{job['base_arm']} declara dos padres congelados",
            )
            group[FROZEN] = job["arm"]
        elif control == CONTINUATION:
            _require(
                group[CONTINUATION] in (None, job["arm"]),
                f"{job['base_arm']} declara dos continuaciones completas",
            )
            group[CONTINUATION] = job["arm"]
        elif job["arm"] not in group[ADAPTED]:
            group[ADAPTED].append(job["arm"])
        group["jobs"].setdefault(job["scope"], []).append(job)
    for base_arm, group in groups.items():
        _require(group[CONTINUATION] is not None, f"{base_arm} no tiene continuación completa")
        _require(group[ADAPTED], f"{base_arm} no tiene brazos adaptados")
    return groups


def load_declaration(path):
    """Validar la declaración, cargar su etapa y derivar las comparaciones sin leer datos."""
    path = Path(path)
    declaration, digest = read_manifest(path, 1024**2)
    _require(
        isinstance(declaration, dict)
        and set(declaration) == _FIELDS
        and declaration["schema_version"] == 1
        and declaration["kind"] == KIND
        and declaration["status"] == DECLARED
        and declaration["use"] == USE
        and isinstance(declaration["name"], str)
        and walk._name(declaration["name"].replace("-", "_")),
        "La declaración de la comparación postentrenada no cumple su contrato",
    )
    _families(declaration["families"])
    stage = load_stage((path.parent / declaration["stage"]).resolve())
    groups = _groups(stage)
    configs = {base_arm: derive_config(declaration, stage, base_arm, groups) for base_arm in groups}
    return dict(declaration, sha256=digest, stage=stage, groups=groups, configs=configs)


def derive_config(declaration, stage, base_arm, groups):
    """Configuración walk-forward validada de un padre, derivada de la de la campaña."""
    campaign = stage["campaign"]
    inherited = campaign["comparison_config"]
    group = groups[base_arm]
    parent = inherited["arms"][base_arm]
    roles = {FROZEN: [group[FROZEN]], CONTINUATION: [group[CONTINUATION]], ADAPTED: group[ADAPTED]}
    arms = {base_arm: dict(parent)}
    for role in ROLES:
        for arm in roles[role]:
            if arm == base_arm:
                continue
            arms[arm] = dict(
                family=ARM_FAMILIES[role], output=parent["output"], seeds=parent["seeds"]
            )
    families = {
        name: dict(
            kind="delta",
            base=roles[family["base"]][0],
            variants=[arm for role in family["variants"] for arm in roles[role]],
        )
        for name, family in declaration["families"].items()
    }
    families["levels"] = dict(kind="level", arms=list(arms))
    raw = {
        key: value
        for key, value in inherited.items()
        if key not in {"sha256", "resolved_scopes", "resolved_families", *DERIVED}
    }
    config = dict(
        raw,
        name=f"{declaration['name']}-{base_arm}",
        scopes={scope: _stage_windows(inherited, scope, group) for scope in stage["scopes"]},
        arms=arms,
        comparison=dict(inherited["comparison"], families=families),
    )
    folder = Path(campaign["comparison_path"]).parent
    return walk.validate_config(config, _digest(config), folder)


def _stage_windows(inherited, scope, group):
    """Ámbito heredado con solo las ventanas que tienen trabajos de la etapa.

    En el plan por etapas la primera ventana no tiene postentrenamiento y queda fuera. Si la
    etapa cubre todas las ventanas, la declaración del ámbito no cambia.
    """
    present = {job["window"] for job in group["jobs"].get(scope, [])}
    windows = [
        window for window in inherited["resolved_scopes"][scope]["windows"] if window in present
    ]
    _require(windows, f"La etapa no tiene trabajos de {scope}")
    declared = inherited["scopes"][scope]
    if len(windows) == len(inherited["resolved_scopes"][scope]["windows"]):
        return declared
    return dict(declared, windows=windows)


def check_declaration(path):
    """Resumen de los padres, brazos y familias derivados, sin leer predicciones."""
    loaded = load_declaration(path)
    return dict(
        status="checked",
        name=loaded["name"],
        declaration_sha256=loaded["sha256"],
        stage_sha256=loaded["stage"]["sha256"],
        scopes=loaded["stage"]["scopes"],
        parents={
            base_arm: dict(
                configuration_sha256=config["sha256"],
                arms=len(config["arms"]),
                continuation=loaded["groups"][base_arm][CONTINUATION],
                adapted=loaded["groups"][base_arm][ADAPTED],
                families={
                    name: list(contrasts) for name, contrasts in config["resolved_families"].items()
                },
            )
            for base_arm, config in loaded["configs"].items()
        },
        final_test_opened=False,
    )


def _base_sources(path, inherited, scope, base_arm):
    """Fuentes ya publicadas de la campaña, validadas con los brazos que contienen."""
    manifest, _ = read_manifest(path, 16 * 1024**2)
    present = set(manifest.get("arms", {})) if isinstance(manifest, dict) else set()
    _require(base_arm in present, f"Las fuentes de la campaña no contienen {base_arm}")
    arms = {
        name: arm
        for name, arm in inherited["arms"].items()
        if name in present or arm["output"] == walk.ZERO_CONTROL
    }
    return walk.load_sources(path, dict(inherited, arms=arms), scope)


def _stage_marker(stage_output, stage, views, scope):
    """Identidad de la ejecución de la etapa: esta etapa, la reserva cerrada y las vistas."""
    marker, _ = read_manifest(stage_output / "stage.json", 8 * 1024**2)
    _require(
        isinstance(marker, dict)
        and marker.get("kind") == RUN_KIND
        and marker.get("stage_sha256") == stage["sha256"]
        and marker.get("final_test_opened") is False,
        "La salida no pertenece a la etapa declarada",
    )
    _require(
        marker.get("views", {}).get(scope) == views,
        "La etapa usó otras vistas que las fuentes de la campaña",
    )
    return _digest(marker)


def _stage_receipt(stage_output, job, identity, view_sha256):
    """Recibo confirmado de un trabajo de la etapa, con su identidad y su vista."""
    receipt, _ = read_manifest(stage_output / "jobs" / job["id"] / "receipt.json", 8 * 1024**2)
    expected = receipt.get("identity") if isinstance(receipt, dict) else None
    _require(
        isinstance(expected, dict)
        and receipt.get("kind") == RECEIPT_KIND
        and receipt.get("status") == "completed"
        and receipt.get("final_test_opened") is False
        and expected.get("id") == job["id"]
        and expected.get("stage_identity_sha256") == identity
        and expected.get("view_sha256") == view_sha256,
        f"El recibo de {job['id']} no corresponde a la etapa ni a su vista",
    )
    return receipt


def write_sources(declaration_path, scope, base_arm, *, base_sources, stage_output):
    """Publicar el manifiesto de fuentes de un padre y un ámbito y validarlo."""
    loaded = load_declaration(declaration_path)
    stage = loaded["stage"]
    _require(scope in stage["scopes"], "El ámbito no pertenece a la etapa")
    _require(base_arm in loaded["groups"], "El brazo no es un padre de la etapa")
    inherited = stage["campaign"]["comparison_config"]
    config = loaded["configs"][base_arm]
    base = _base_sources(Path(base_sources), inherited, scope, base_arm)
    stage_output = Path(stage_output)
    identity = _stage_marker(stage_output, stage, base["views"], scope)
    folder = stage_output / "sources" / scope
    destination = folder / f"{base_arm}.json"

    def relative(path):
        return os.path.relpath(Path(path).resolve(), folder.resolve())

    arms = {name: {} for name in config["arms"]}
    policy = config["input_policy"]
    # La comparación cubre las ventanas de la configuración derivada, que pueden ser menos
    # que las de la campaña. La etapa, en cambio, debe haber usado todas sus vistas.
    views = {
        window: base["views"][window] for window in config["resolved_scopes"][scope]["windows"]
    }
    for seed in config["arms"][base_arm]["seeds"]:
        entries = arms[base_arm][str(seed)] = {}
        for window, view_sha256 in views.items():
            files = base["files"][base_arm, seed, window]
            entries[window] = dict(input_policy=policy, view_sha256=view_sha256)
            for part, record in files.items():
                entries[window][part] = dict(path=relative(record["path"]), sha256=record["sha256"])
    for job in loaded["groups"][base_arm]["jobs"][scope]:
        view_sha256 = base["views"][job["window"]]
        receipt = _stage_receipt(stage_output, job, identity, view_sha256)
        entry = dict(input_policy=policy, view_sha256=view_sha256)
        for part in COMPARED:
            record = receipt["predictions"][part]
            entry[part] = dict(
                path=relative(stage_output / record["path"]), sha256=record["sha256"]
            )
        arms[job["arm"]].setdefault(str(job["seed"]), {})[job["window"]] = entry
    manifest = dict(
        schema_version=1,
        kind=walk.SOURCES_KIND,
        scope=scope,
        input_policy=policy,
        windows={
            window: dict(view=dict(path=relative(base["view_paths"][window]), sha256=digest))
            for window, digest in views.items()
        },
        arms=arms,
    )
    safe_destination(destination)
    candidate = folder / f".{base_arm}.candidate.json"
    atomic_json(candidate, manifest)
    try:
        walk.load_sources(candidate, config, scope)
    except BaseException:
        candidate.unlink()
        raise
    os.replace(candidate, destination)
    return destination


def evaluate(declaration_path, sources_path, scope, base_arm, *, edition=None):
    """Informe walk-forward del padre y, con ``edition``, su cartera larga y corta."""
    loaded = load_declaration(declaration_path)
    _require(base_arm in loaded["configs"], "El brazo no es un padre de la etapa")
    config = loaded["configs"][base_arm]
    report, sessions = walk.evaluate_walk_forward(config, sources_path, scope)
    report["posttraining"] = dict(
        declaration=dict(name=loaded["name"], sha256=loaded["sha256"]),
        stage_sha256=loaded["stage"]["sha256"],
        base_arm=base_arm,
        windows=list(config["resolved_scopes"][scope]["windows"]),
        roles={
            FROZEN: loaded["groups"][base_arm][FROZEN],
            CONTINUATION: loaded["groups"][base_arm][CONTINUATION],
            ADAPTED: loaded["groups"][base_arm][ADAPTED],
        },
        families=loaded["families"],
        use=USE,
    )
    portfolio = None
    if edition is not None:
        portfolio = long_short_comparison.evaluate_long_short(config, sources_path, scope, edition)
    return config, report, sessions, portfolio


def write(declaration_path, sources_path, scope, base_arm, output, *, edition=None):
    """Publicar configuración derivada, informe y sesiones en un directorio nuevo."""
    output = Path(output)
    safe_destination(output)
    _require(not output.exists(), "La salida debe ser nueva")
    sources = [Path(declaration_path).parent, Path(sources_path).parent]
    if edition is not None:
        sources.append(Path(edition))
    for source in sources:
        outside_source(source, output)
        outside_source(output, source)
    config, report, sessions, portfolio = evaluate(
        declaration_path, sources_path, scope, base_arm, edition=edition
    )
    json.dumps(report, allow_nan=False)
    if portfolio is not None:
        json.dumps(portfolio[0], allow_nan=False)
    output.mkdir(parents=True)
    derived = {key: value for key, value in config.items() if not key.startswith("resolved_")}
    atomic_json(output / "configuration.json", derived)
    pq.write_table(sessions, output / "sessions.parquet", compression="zstd")
    report["artifacts"] = {
        name: sha256(output / name) for name in ("configuration.json", "sessions.parquet")
    }
    if portfolio is not None:
        book, rows = portfolio
        pq.write_table(rows, output / "long_short_sessions.parquet", compression="zstd")
        book["artifacts"] = {
            "long_short_sessions.parquet": sha256(output / "long_short_sessions.parquet")
        }
        atomic_json(output / "long_short.json", book)
    atomic_json(output / "comparison.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Derivar padres, brazos y familias sin datos")
    sources = commands.add_parser("sources", help="Publicar las fuentes de un padre")
    run = commands.add_parser("evaluate", help="Comparar un padre con sus brazos postentrenados")
    for command in (check, sources, run):
        command.add_argument("--declaration", type=Path, required=True)
    for command in (sources, run):
        command.add_argument("--scope", choices=tuple(walk.SCOPES), required=True)
        command.add_argument("--base-arm", required=True)
    sources.add_argument("--base-sources", type=Path, required=True)
    sources.add_argument("--stage-output", type=Path, required=True)
    run.add_argument("--sources", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--edition", type=Path)
    args = parser.parse_args(argv)
    if args.command == "check":
        result = check_declaration(args.declaration)
    elif args.command == "sources":
        destination = write_sources(
            args.declaration,
            args.scope,
            args.base_arm,
            base_sources=args.base_sources,
            stage_output=args.stage_output,
        )
        result = dict(sources=str(destination), sha256=sha256(destination))
    else:
        report = write(
            args.declaration,
            args.sources,
            args.scope,
            args.base_arm,
            args.output,
            edition=args.edition,
        )
        result = dict(arms=len(report["arms"]), windows=len(report["windows"]))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
