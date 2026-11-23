"""WP5: cada órgano episódico frente a su padre y a su placebo, por semilla y en la media.

Programa §5.14, fijado antes de medir. Lee los estudios que deja la emisión de órganos
(`<raíz>/studies/<origen>--seed-<s>.json` y `.parquet`, con el IC por sesión de cada variante)
de una raíz cuya declaración es la registrada, `posttraining-memory-organs-wp5-v10.json`. No elige
nada: la emisión eligió cada hiperparámetro en calibración. Para cada padre declarado y cada
órgano, en evaluación y con la t de Newey-West de la criba:

- órgano − padre, placebo − padre y órgano − placebo, diferencias pareadas por sesión, por
  semilla y en la media de semillas (el IC de cada sesión es la media de los IC de las semillas);
- el placebo falla como debe si no gana al padre con t ≥ 2. Si le gana, es una alarma y el órgano
  no se lee hasta saber por qué;
- el órgano trae señal propia si gana al padre y a su placebo, las dos veces con t ≥ 2.

Es descriptivo. Lo que decide es la regla de F6 con cada órgano (las variantes de WP5 de
`holographic`).
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mars_titan.evaluation import organ_aggregation as og
from mars_titan.evaluation import posttraining_pilots as pilots
from mars_titan.memory import episodic_organs as eo

KIND = "posttraining_episodic_contrasts_v1"
# Huella canónica de configs/experiments/posttraining-memory-organs-wp5-v10.json.
REGISTERED_SHA256 = "6c862b76066e25f7e1579a788d006177b0f233161c9dff2c2ebb5d87f4e96ca2"
MIN_T = 2.0
CONTRASTS = ("vs_parent", "placebo_vs_parent", "vs_placebo")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def series(rows, model, split, variant):
    """El IC por sesión de una variante: {(mercado, instante): IC}."""
    return {
        (row["market"], row["instant"]): row["value"]
        for row in rows
        if row["model"] == model and row["split"] == split and row["variant"] == variant
    }


def seed_mean(per_seed):
    """En cada sesión común a todas las semillas, la media de sus IC."""
    common = set.intersection(*(set(values) for values in per_seed))
    return {key: float(np.mean([values[key] for values in per_seed])) for key in sorted(common)}


def block(tool, organ, parent, placebo):
    """Niveles medios y los tres contrastes pareados por sesión."""
    return dict(
        levels={
            name: float(np.mean(list(values.values()))) if values else None
            for name, values in (("parent", parent), ("organ", organ), ("placebo", placebo))
        },
        vs_parent=tool.paired(organ, parent),
        placebo_vs_parent=tool.paired(placebo, parent),
        vs_placebo=tool.paired(organ, placebo),
    )


def _wins(contrast, min_t):
    mean, t = contrast.get("mean"), contrast.get("t")
    return mean is not None and t is not None and math.isfinite(t) and mean > 0 and t >= min_t


def reading(entry, min_t=MIN_T):
    """La lectura declarada: el placebo falla como debe y el órgano trae señal propia."""
    placebo_fails = not _wins(entry["placebo_vs_parent"], min_t)
    own = _wins(entry["vs_parent"], min_t) and _wins(entry["vs_placebo"], min_t)
    if not placebo_fails:
        text = "alarma: el placebo gana al padre; el órgano no se lee hasta saber por qué"
    elif own:
        text = "trae señal propia: gana al padre y a su placebo"
    else:
        text = "sin señal propia demostrada"
    return dict(placebo_fails=placebo_fails, own_signal=own, text=text)


def _digest(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return document, hashlib.sha256(pilots._canonical(document).encode()).hexdigest()


def inputs(root, origin, seeds):
    """Los archivos que se leen; `og.MissingInputs` si falta alguno."""
    files = {"declaration": Path(root) / "declaration.json"}
    for seed in seeds:
        for suffix in ("json", "parquet"):
            files[f"{seed}/{suffix}"] = Path(root) / "studies" / f"{origin}--seed-{seed}.{suffix}"
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise og.MissingInputs(missing)
    return {name: dict(path=str(path), sha256=pilots._sha256(path)) for name, path in files.items()}


def compute(root, origin, seeds, *, registered=REGISTERED_SHA256):
    """El resumen de un origen; `registered=None` solo en las pruebas sintéticas."""
    seeds = list(seeds)
    _require(seeds and len(set(seeds)) == len(seeds), "Las semillas son distintas y hay alguna")
    prints = inputs(root, origin, seeds)
    declaration, digest = _digest(prints["declaration"]["path"])
    _require(
        registered is None or digest == registered,
        f"La declaración de {root} no es la registrada para WP5 (huella {digest[:12]}, registrada "
        f"{str(registered)[:12]})",
    )
    arms = [arm for arm in eo.EPISODIC_ARMS if arm in declaration.get("episodic_organs", {})]
    _require(arms, f"{root} no declara órganos episódicos")
    tool = pilots.screening()
    reports, rows = {}, {}
    for seed in seeds:
        report = json.loads(Path(prints[f"{seed}/json"]["path"]).read_text(encoding="utf-8"))
        _require(
            report.get("declaration_sha256") == digest and report.get("seed") == seed,
            f"El estudio de la semilla {seed} no es de esta declaración",
        )
        reports[seed] = report
        rows[seed] = pq.read_table(prints[f"{seed}/parquet"]["path"]).to_pylist()
    results = {}
    for model in declaration["models"]:
        results[model] = {}
        for arm in arms:
            per_seed, kept = {}, {"organ": [], "parent": [], "placebo": []}
            for seed in seeds:
                found = {
                    name: series(rows[seed], model, "evaluation", variant)
                    for name, variant in (
                        ("organ", arm),
                        ("parent", "parent"),
                        ("placebo", f"{arm}_placebo"),
                    )
                }
                entry = block(tool, found["organ"], found["parent"], found["placebo"])
                study = reports[seed]["episodic_organs"][model][arm]
                for name in CONTRASTS:
                    left, right = entry[name]["mean"], study["contrasts"][name]["mean"]
                    _require(
                        left is not None and right is not None and abs(left - right) <= 1e-12,
                        f"{model}/{arm}, semilla {seed}: {name} no es el del estudio",
                    )
                per_seed[seed] = dict(
                    selection=study["selection"],
                    calibration=study["calibration"],
                    placebo_calibration=study["placebo_calibration"],
                    uncertainty=study.get("uncertainty"),
                    **entry,
                    reading=reading(entry),
                )
                for name in kept:
                    kept[name].append(found[name])
            mean = block(tool, *(seed_mean(kept[name]) for name in ("organ", "parent", "placebo")))
            results[model][arm] = dict(seeds=per_seed, mean=dict(mean, reading=reading(mean)))
    return dict(
        schema_version=1,
        kind=KIND,
        status="completed",
        origin=origin,
        seeds=seeds,
        declaration_sha256=digest,
        organs_root=str(root),
        min_t=MIN_T,
        inputs=prints,
        results=results,
    )


def _f(value, digits=4):
    return "—" if value is None else f"{value:+.{digits}f}".replace("-", "−").replace(".", ",")


def _c(entry):
    t = entry.get("t")
    return f"{_f(entry.get('mean'))} (t {_f(t, 2)})" if t is not None else _f(entry.get("mean"))


def markdown(summary):
    lines = [
        f"# WP5: cada órgano frente a su padre y a su placebo, origen {summary['origin']}",
        "",
        f"Declaración {summary['declaration_sha256'][:12]}; semillas {summary['seeds']}; "
        "evaluación, diferencias pareadas por sesión con t de Newey-West. Es descriptivo: decide "
        "la regla de F6 con cada órgano.",
    ]
    for model, arms in summary["results"].items():
        lines += [
            "",
            f"## Padre {model} (media de semillas)",
            "",
            "| Órgano | IC padre | IC órgano | órgano − padre | placebo − padre | "
            "órgano − placebo | Lectura |",
            "|---|---|---|---|---|---|---|",
        ]
        for arm, result in arms.items():
            mean = result["mean"]
            lines.append(
                f"| {arm} | {_f(mean['levels']['parent'])} | {_f(mean['levels']['organ'])} | "
                f"{_c(mean['vs_parent'])} | {_c(mean['placebo_vs_parent'])} | "
                f"{_c(mean['vs_placebo'])} | {mean['reading']['text']} |"
            )
        lines += [
            "",
            "| Órgano | Semilla | Elección en calibración | órgano − padre |",
            "|---|---|---|---|",
        ]
        for arm, result in arms.items():
            for seed, entry in result["seeds"].items():
                chosen = {
                    k: v
                    for k, v in entry["selection"].items()
                    if k in ("tau", "kappa", "half_life", "dim", "decay", "rate")
                }
                lines.append(f"| {arm} | {seed} | {chosen} | {_c(entry['vs_parent'])} |")
    return "\n".join(lines) + "\n"


def measure(root, origin, seeds, output, *, log=print, registered=REGISTERED_SHA256):
    """Publica el resumen en `<output>/<origen>`; si ya está con estas entradas, lo devuelve."""
    target = Path(output) / str(origin)
    if target.exists():
        prints = inputs(root, origin, seeds)
        _, digest = _digest(prints["declaration"]["path"])
        return og._already(target, digest, prints, log)
    summary = compute(root, origin, seeds, registered=registered)
    Path(output).mkdir(parents=True, exist_ok=True)
    work = Path(output) / f".{origin}.{os.getpid()}.tmp"
    try:
        work.mkdir(parents=True)
        text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        (work / "resumen.json").write_text(text, encoding="utf-8")
        (work / "resumen.md").write_text(markdown(summary), encoding="utf-8")
        os.rename(work, target)
    finally:
        if work.exists():
            shutil.rmtree(work)
    log(f"publicado en {target}")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("medir", help="Contrastes de los órganos de WP5 de un origen")
    run.add_argument("--organs", type=Path, required=True, help="Raíz de los órganos de WP5")
    run.add_argument("--origin", type=int, required=True)
    run.add_argument("--seed", type=int, action="append", dest="seeds", required=True)
    run.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)

    def log(line):
        print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {line}", flush=True)

    try:
        measure(args.organs, args.origin, args.seeds, args.output, log=log)
    except og.MissingInputs as missing:
        print(f"Faltan {len(missing.items)} entradas de WP5; no se escribe nada:", file=sys.stderr)
        for item in missing.items:
            print(f"- {item}", file=sys.stderr)
        return og.MISSING_EXIT
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
