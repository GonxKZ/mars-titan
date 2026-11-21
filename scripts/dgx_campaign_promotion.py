"""Promoción de la criba de una campaña DGX: informe del origen de criba y edición del plan.

Es una fase «orden» del relevo de campañas. Con el origen de criba terminado:

1. escribe el informe de la criba (`dgx_screening_report`) en la carpeta de registros;
2. aplica la regla fijada antes de ver los resultados, con el contraste pareado por sesión de
   cada variante con su control en validación y calibración:
   - siguen en los demás orígenes de la criba las `--keep` variantes con mayor t, las que se
     conservan siempre (los contrastes de modalidades) y los controles que necesitan;
   - pasan a los orígenes finales las que superan a su control en al menos `--gain` con t de al
     menos `--t`, con sus controles;
3. edita solo las fases del plan posteriores a la suya: quita de los orígenes de criba los
   niveles de las variantes que no siguen y añade a los orígenes finales los de las promovidas.

Deja en la carpeta de registros el informe, la decisión (`promotion.json` y `promotion.md`) y
una copia del plan anterior. Si algo falla, no toca el plan: quedan las fases que tenía, que son
válidas sin promoción, y el error queda en `promotion.json`.
"""

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

CLEAN = re.compile(r"limpia\|/.+/packed-inputs/[0-9]{4}")
PREPARE = re.compile(r"prepara\|/[^|]+\|/[^|]+\|[^|]+\|20[0-9]{2}")
ORDER = re.compile(r"orden\|/[^|]+\|.*")
RUN = re.compile(r"/[^|]+\|/[^|]+\|/[^|]+\|[^|]+\|[1-8] .+")


def report_module():
    path = Path(__file__).resolve().with_name("dgx_screening_report.py")
    spec = importlib.util.spec_from_file_location("dgx_screening_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_phase(line):
    return bool(line.strip()) and not line.lstrip().startswith("#")


def valid(line):
    """Las mismas formas de fase que acepta el relevo."""
    return any(pattern.fullmatch(line) for pattern in (CLEAN, PREPARE, ORDER, RUN))


def parse_run(line):
    """Estado, orígenes, semillas y niveles de una fase de ejecución, o None si es de otro tipo."""
    if not RUN.fullmatch(line) or line.startswith(("limpia|", "prepara|", "orden|")):
        return None
    copy, state, views, declaration, phase = line.split("|")
    tokens = phase.split()
    options = {"--origin": [], "--seed": [], "--tier": []}
    index = 1
    while index < len(tokens):
        if tokens[index] in options and index + 1 < len(tokens):
            options[tokens[index]].append(tokens[index + 1])
            index += 2
        else:
            index += 1
    return dict(
        prefix="|".join((copy, state, views, declaration)),
        state=state,
        tokens=tokens,
        origins=[int(year) for year in options["--origin"]],
        seeds=[int(seed) for seed in options["--seed"]],
        tiers=options["--tier"],
    )


def with_tiers(run, tiers):
    """La misma fase con otra lista de niveles, en el orden dado."""
    tokens, kept, index = run["tokens"], [], 0
    while index < len(tokens):
        if tokens[index] == "--tier" and index + 1 < len(tokens):
            index += 2
        else:
            kept.append(tokens[index])
            index += 1
    for tier in tiers:
        kept += ["--tier", tier]
    return run["prefix"] + "|" + " ".join(kept)


def decide(origin, controls, always, *, keep, gain, t_value, prefixes=("v_",)):
    """Aplicar la regla al informe del origen de criba. Devuelve niveles y su justificación.

    `prefixes` son los prefijos de los niveles que se valoran: las variantes de la criba (`v_`)
    o, en otra orden, otro grupo de modelos con sus controles declarados.
    """
    models = origin["models"]
    tier = {name: model["tier"] for name, model in models.items()}
    rows = {row["model"]: row for row in origin["contrasts"]}
    prefixes = tuple(prefixes)
    variants = sorted(name for name in rows if (tier.get(name) or "").startswith(prefixes))

    def needed(name):
        chain, current = [], controls.get(name)
        while current is not None and (tier.get(current) or "").startswith(prefixes):
            chain.append(current)
            current = controls.get(current)
        return chain

    def score(name):
        value = rows[name]["decision"]["t"]
        return -1e9 if value is None else value

    ranked = sorted(variants, key=lambda name: (-score(name), name))
    top = ranked[:keep]
    promoted = [
        name
        for name in ranked
        if (rows[name]["decision"]["mean"] or 0) >= gain and score(name) >= t_value
    ]
    kept = set(top) | {name for name in always if name in models}
    for name in list(kept):
        kept.update(needed(name))
    final = set(promoted)
    for name in promoted:
        final.update(needed(name))
    return dict(
        ranked=[
            dict(
                model=name,
                tier=tier[name],
                control=rows[name]["control"],
                decision_mean=rows[name]["decision"]["mean"],
                decision_t=rows[name]["decision"]["t"],
                evaluation_mean=rows[name]["evaluation"]["mean"],
            )
            for name in ranked
        ],
        kept_tiers=sorted({tier[name] for name in kept}),
        promoted_models=promoted,
        promoted_tiers=sorted({tier[name] for name in final}),
        assessed_tiers=sorted({tier[name] for name in variants}),
    )


def edit_plan(lines, phase, decision, prune, promote, prefixes=("v_",)):
    """Quitar y añadir niveles en las fases posteriores a `phase`. Devuelve líneas y cambios."""
    prefixes = tuple(prefixes)
    result, changes, position = list(lines), [], 0
    for index, line in enumerate(lines):
        if not is_phase(line):
            continue
        position += 1
        run = parse_run(line) if position > phase else None
        if run is None or run["seeds"] != [42] or len(run["origins"]) != 1:
            continue
        year, tiers = run["origins"][0], run["tiers"]
        updated = None
        if year in prune.get(run["state"], ()) and any(t.startswith(prefixes) for t in tiers):
            # Un nivel sin valorar (su variante no terminó en el origen de criba) no sigue.
            updated = [
                t for t in tiers if not t.startswith(prefixes) or t in decision["kept_tiers"]
            ]
        elif year in promote.get(run["state"], ()) and tiers and "core" in tiers:
            updated = tiers + [t for t in decision["promoted_tiers"] if t not in tiers]
        if updated is not None and updated != tiers:
            result[index] = with_tiers(run, updated)
            changes.append(
                dict(
                    phase=position,
                    state=run["state"],
                    origin=year,
                    removed=[t for t in tiers if t not in updated],
                    added=[t for t in updated if t not in tiers],
                )
            )
    return result, changes


def _targets(values):
    """`estado:2014,2015` repetido -> {estado: {2014, 2015}}."""
    targets = {}
    for value in values:
        state, _, years = value.rpartition(":")
        targets.setdefault(state, set()).update(int(year) for year in years.split(",") if year)
    return targets


def _markdown(document):
    rule = document["rule"]
    lines = [
        f"# Promoción de la criba del origen {document['year']}",
        "",
        f"Regla fijada de antemano: siguen las {rule['keep']} variantes con mayor t y los "
        "contrastes de modalidades; se promueven las que superan a su control en al menos "
        f"{rule['gain']} con t de al menos {rule['t']}. Solo deciden validación y calibración.",
        "",
    ]
    if document.get("error"):
        lines += [f"**No se ha tocado el plan**: {document['error']}", ""]
        return "\n".join(lines)
    decision = document["decision"]
    promoted = ", ".join(f"`{name}`" for name in decision["promoted_models"]) or "ninguna"
    lines += [
        f"Promovidas a los orígenes finales: {promoted}.",
        "",
        "| Variante | Control | Diferencia de decisión | t | Diferencia en evaluación | Sigue |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    def cell(value, digits=4):
        return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")

    for row in decision["ranked"]:
        follows = "sí" if row["tier"] in decision["kept_tiers"] else "no"
        lines.append(
            f"| `{row['model']}` | `{row['control']}` | {cell(row['decision_mean'])} | "
            f"{cell(row['decision_t'], 2)} | {cell(row['evaluation_mean'])} | {follows} |"
        )
    lines += ["", "## Fases cambiadas", ""]
    for change in document["changes"]:
        removed = ", ".join(change["removed"]) or "nada"
        added = ", ".join(change["added"]) or "nada"
        lines.append(
            f"- Fase {change['phase']} (origen {change['origin']}): quita {removed}; añade {added}."
        )
    if not document["changes"]:
        lines.append("Ninguna.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contrasts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=os.environ.get("MARS_TITAN_RELAY_PLAN"))
    parser.add_argument("--phase", type=int, default=os.environ.get("MARS_TITAN_RELAY_PHASE"))
    parser.add_argument("--prune", action="append", default=[], help="estado:años de la criba")
    parser.add_argument("--promote", action="append", default=[], help="estado:años finales")
    parser.add_argument(
        "--candidates",
        action="append",
        metavar="PREFIJO",
        help="Prefijo de los niveles que se valoran (v_ por defecto); se puede repetir",
    )
    parser.add_argument("--keep", type=int, default=12)
    parser.add_argument("--gain", type=float, default=0.003)
    parser.add_argument("--t", type=float, default=2.0)
    args = parser.parse_args(argv)
    prefixes = tuple(args.candidates or ["v_"])
    args.output.mkdir(parents=True, exist_ok=True)
    document = dict(
        schema_version=1,
        kind="dgx_campaign_promotion",
        year=args.year,
        seed=args.seed,
        state=str(args.state),
        rule=dict(
            keep=args.keep,
            gain=args.gain,
            t=args.t,
            decision_splits=["validation", "calibration"],
            candidates=list(prefixes),
        ),
        created_at_utc=datetime.now(UTC).isoformat(),
        changes=[],
    )
    code = 0
    try:
        declared = json.loads(args.contrasts.read_text())
        screening = report_module()
        report = screening.build(
            args.state, [args.year], args.seed, declared["controls"], {}, references=args.reference
        )
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1) + "\n"
        )
        (args.output / "report.md").write_text(screening.markdown(report))
        decision = decide(
            report["origins"][0],
            declared["controls"],
            declared.get("always_kept", []),
            keep=args.keep,
            gain=args.gain,
            t_value=args.t,
            prefixes=prefixes,
        )
        document["decision"] = decision
        if not decision["assessed_tiers"]:
            raise ValueError("Ninguna variante ha terminado en el origen de criba")
        if args.plan is None or args.phase is None:
            raise ValueError("Falta el plan o el número de esta fase: no se edita ningún plan")
        plan, phase = Path(args.plan), int(args.phase)
        lines = plan.read_text().split("\n")
        edited, changes = edit_plan(
            lines, phase, decision, _targets(args.prune), _targets(args.promote), prefixes
        )
        phases = [line for line in edited if is_phase(line)]
        before = [line for line in lines if is_phase(line)]
        if len(phases) != len(before) or phases[:phase] != before[:phase]:
            raise ValueError("La edición cambiaría fases ya recorridas o su número")
        wrong = [line for line in phases if not valid(line)]
        if wrong:
            raise ValueError(f"La edición dejaría una fase mal escrita: {wrong[0][:120]}")
        if changes:
            shutil.copyfile(plan, args.output / "plan-anterior.txt")
            staged = plan.with_name(plan.name + ".promocion")
            staged.write_text("\n".join(edited))
            os.replace(staged, plan)
        document["changes"] = changes
    # Cualquier fallo deja el plan como estaba y queda registrado: la campaña sigue sin promoción.
    except Exception as error:  # noqa: BLE001
        document["error"] = f"{type(error).__name__}: {error}"
        document["traceback"] = traceback.format_exc(limit=6)
        code = 1
    (args.output / "promotion.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=1) + "\n"
    )
    (args.output / "promotion.md").write_text(_markdown(document))
    print(
        f"promoción del origen {args.year}: {len(document['changes'])} fases cambiadas"
        + (f"; error: {document['error']}" if code else "")
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
