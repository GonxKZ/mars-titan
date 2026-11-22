#!/usr/bin/env python3
"""Contrastes con varias semillas: cada modelo frente a su control en un origen.

Las semillas de un mismo origen recorren las mismas sesiones, así que no son réplicas
independientes de una sesión. El IC de rangos de cada sesión se promedia entre semillas y el
contraste pareado con su control se hace sobre esa serie, con error con autocorrelación. La
diferencia de cada semilla se informa aparte, para ver si el signo se repite.

Además se mide el conjunto de cada modelo: el IC de la media de los rangos de sus semillas, que
es lo que se usaría al predecir con las tres. Su ganancia sobre la semilla media y el contraste
entre el conjunto de un modelo y el de su control van pareados por sesión.

Un modelo al que le falta alguna semilla no se contrasta con las que tenga: se anota como
incompleto.

Con varios orígenes (`--year` repetido) se añade el resumen entre orígenes. Cada origen cuenta
como una réplica: la media de las diferencias de los orígenes, su error entre orígenes y una t
con K - 1 grados de libertad, que no supone nada sobre la dependencia dentro de un año. Al lado
va el contraste con todas las sesiones juntas, cuya autocorrelación no cruza de un origen a
otro. Un origen en el que al modelo o a su control le falta una semilla queda fuera de ese
contraste y se nombra.

Uso:
    dgx_seed_contrasts.py --state <estado> --year 2013 --seed 42 --seed 43 --seed 44 \
        --contrasts <controles.json> --output <carpeta nueva>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

SPLITS = ("validation", "calibration", "evaluation")
DECISION = ("validation", "calibration")


def _report_module():
    """El informe de la criba, de donde salen la lectura de emisiones y el contraste pareado."""
    name = "dgx_screening_report"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("dgx_screening_report.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def seed_splits(state, model, year, seed):
    """Emisiones de cada tramo de una ejecución, o None si le falta alguno."""
    report = _report_module()
    run = Path(state) / "runs" / model / str(year) / f"seed-{seed}"
    result = {}
    for split in SPLITS:
        data = report.load_split(run, split)
        if data is None:
            return None
        result[split] = data
    return result


def seed_sessions(splits):
    """IC de rangos por sesión de cada tramo de una ejecución."""
    report = _report_module()
    return {split: report.session_ic(splits[split]) for split in SPLITS}


def averaged(per_seed, model):
    """Media entre semillas del IC de cada sesión. Las semillas deben compartir sesiones."""
    result = {}
    for split in SPLITS:
        series = [sessions[split] for sessions in per_seed]
        keys = set(series[0])
        if any(set(other) != keys for other in series[1:]):
            raise ValueError(f"Las semillas de {model} no comparten sesiones en {split}")
        result[split] = {key: float(np.mean([one[key] for one in series])) for key in keys}
    return result


def _ranks(data):
    """Rangos medios de la predicción dentro de cada sesión, sin normalizar.

    Son múltiplos de 0,5 y su suma entre semillas es exacta en coma flotante. Normalizados
    antes de sumar, dos activos con la misma suma de rangos podrían diferir en el último bit y
    dejar de contar como empate.
    """
    report = _report_module()
    ranks = np.empty(len(data["target"]), dtype=np.float64)
    for _, _, start, end in data["sessions"]:
        ranks[start:end] = report.average_ranks(data["prediction"][start:end])
    return ranks


def ensemble(members, model):
    """IC por sesión de la media de rangos de las semillas: el conjunto de un modelo."""
    report = _report_module()
    result = {}
    for split in SPLITS:
        base = members[0][split]
        if any(other[split]["sample"] != base["sample"] for other in members[1:]):
            raise ValueError(f"Las semillas de {model} no comparten muestras en {split}")
        # Las semillas comparten filas, así que la suma de rangos ordena igual que su media.
        result[split] = report.session_ic(base, sum(_ranks(member[split]) for member in members))
    return result


def _pooled(sessions, splits):
    return {(split, *key): value for split in splits for key, value in sessions[split].items()}


def _paired(candidate, control):
    """Contraste pareado en cada tramo y en los dos de decisión juntos."""
    report = _report_module()
    entry = {split: report.paired(candidate[split], control[split]) for split in SPLITS}
    entry["decision"] = report.paired(_pooled(candidate, DECISION), _pooled(control, DECISION))
    return entry


def _level(sessions, per_seed, combined):
    report = _report_module()
    entry = {split: report.summary(sessions[split]) for split in SPLITS}
    entry["decision"] = report.summary(_pooled(sessions, DECISION))
    entry["per_seed"] = {
        split: [float(np.mean(list(one[split].values()))) for one in per_seed] for split in SPLITS
    }
    entry["ensemble"] = {split: report.summary(combined[split]) for split in SPLITS}
    entry["ensemble"]["decision"] = report.summary(_pooled(combined, DECISION))
    # Lo que gana el conjunto sobre la semilla media, sesión a sesión.
    entry["ensemble_gain"] = _paired(combined, sessions)
    return entry


def build(state, year, seeds, controls):
    """Niveles de cada modelo y contraste de cada uno con su control."""
    return _origin(state, year, seeds, controls)[0]


def _origin(state, year, seeds, controls):
    """El documento de un origen y el IC por sesión de la semilla media y del conjunto."""
    report = _report_module()
    seeds = list(dict.fromkeys(seeds))
    names = sorted(set(controls) | set(controls.values()))
    per_seed, mean, combined, missing = {}, {}, {}, {}
    for name in names:
        # Las emisiones de un modelo se sueltan al pasar al siguiente: solo queda el IC.
        runs = [seed_splits(state, name, year, seed) for seed in seeds]
        absent = [seed for seed, run in zip(seeds, runs, strict=True) if run is None]
        if absent:
            missing[name] = absent
            continue
        per_seed[name] = [seed_sessions(run) for run in runs]
        mean[name] = averaged(per_seed[name], name)
        combined[name] = ensemble(runs, name)
    document = dict(
        schema_version=1,
        kind="dgx_seed_contrasts_v1",
        final_test_opened=False,
        year=year,
        seeds=seeds,
        models={name: _level(mean[name], per_seed[name], combined[name]) for name in mean},
        contrasts={},
        incomplete={},
    )
    for name, control in sorted(controls.items()):
        if name in missing or control in missing:
            document["incomplete"][name] = sorted(
                set(missing.get(name, [])) | set(missing.get(control, []))
            )
            continue
        entry = dict(control=control, seeds=seeds, per_seed={}, signs={})
        for split in SPLITS:
            entry[split] = report.paired(mean[name][split], mean[control][split])
            differences = [
                report.paired(one[split], other[split])["mean"]
                for one, other in zip(per_seed[name], per_seed[control], strict=True)
            ]
            entry["per_seed"][split] = differences
            entry["signs"][split] = "".join("+" if value > 0 else "-" for value in differences)
        entry["decision"] = report.paired(
            _pooled(mean[name], DECISION), _pooled(mean[control], DECISION)
        )
        entry["ensemble"] = _paired(combined[name], combined[control])
        document["contrasts"][name] = entry
    return document, mean, combined


def _between(values):
    """Media de las diferencias de los orígenes con su error entre orígenes."""
    count = len(values)
    result = dict(
        origins=count,
        mean=float(np.mean(values)),
        se=None,
        t=None,
        df=count - 1,
        p=None,
        positive=int(sum(value > 0 for value in values)),
    )
    if count < 2:
        return result
    error = float(np.std(values, ddof=1) / math.sqrt(count))
    result["se"] = error
    if error > 0:
        from scipy import stats

        result["t"] = result["mean"] / error
        result["p"] = float(2 * stats.t.sf(abs(result["t"]), count - 1))
    return result


def _split(sessions, split):
    """Las sesiones de un tramo, o las de los dos de decisión juntas."""
    return _pooled(sessions, DECISION) if split == "decision" else sessions[split]


def build_origins(state, years, seeds, controls):
    """Los contrastes de cada origen y su resumen entre orígenes."""
    report = _report_module()
    years = list(dict.fromkeys(years))
    parts = [_origin(state, year, seeds, controls) for year in years]
    document = dict(
        schema_version=1,
        kind="dgx_seed_contrasts_v1",
        final_test_opened=False,
        years=years,
        seeds=parts[0][0]["seeds"],
        origins=[part[0] for part in parts],
        across={},
    )
    for name, control in sorted(controls.items()):
        present = [part for part in parts if name in part[0]["contrasts"]]
        if not present:
            continue
        inside = [part[0]["year"] for part in present]
        entry = dict(
            control=control, years=inside, missing=[year for year in years if year not in inside]
        )
        for kind in ("seeds", "ensemble"):
            entry[kind] = {}
            for split in ("decision", "evaluation"):
                means, left, right = [], {}, {}
                for origin, mean, combined in present:
                    contrast = origin["contrasts"][name]
                    means.append(
                        (contrast if kind == "seeds" else contrast["ensemble"])[split]["mean"]
                    )
                    sessions = mean if kind == "seeds" else combined
                    for target, model in ((left, name), (right, control)):
                        target.update(
                            {
                                (origin["year"], *key): value
                                for key, value in _split(sessions[model], split).items()
                            }
                        )
                entry[kind][split] = dict(
                    origin_means=means, between=_between(means), pooled=report.paired(left, right)
                )
        document["across"][name] = entry
    return document


def _number(value, digits=4, sign=False):
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def markdown(document):
    lines = [
        f"# Contrastes con {len(document['seeds'])} semillas, origen {document['year']}",
        "",
        "El IC de cada sesión es la media de las semillas. La t usa error con autocorrelación.",
        "El conjunto es el IC de la media de rangos de las semillas, y su Δ compara el conjunto",
        "del modelo con el de su control, en evaluación.",
        "",
        "| Modelo | Control | Semillas | IC de evaluación | Conjunto | Δ decisión (t) "
        "| Δ evaluación (t) | Δ conjunto (t) | Signo por semilla |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, entry in document["contrasts"].items():
        model = document["models"][name]
        level, combined = model["evaluation"]["mean"], model["ensemble"]["evaluation"]["mean"]
        decision, evaluation = entry["decision"], entry["evaluation"]
        between = entry["ensemble"]["evaluation"]
        lines.append(
            f"| {name} | {entry['control']} | {len(entry['seeds'])} | {_number(level)} "
            f"| {_number(combined)} "
            f"| {_number(decision['mean'], sign=True)} ({_number(decision['t'], 1, True)}) "
            f"| {_number(evaluation['mean'], sign=True)} ({_number(evaluation['t'], 1, True)}) "
            f"| {_number(between['mean'], sign=True)} ({_number(between['t'], 1, True)}) "
            f"| {entry['signs']['evaluation']} |"
        )
    lines += [
        "",
        "## Ganancia del conjunto sobre la semilla media",
        "",
        "| Modelo | IC de la semilla media | IC del conjunto | Ganancia en decisión (t) "
        "| Ganancia en evaluación (t) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, model in document["models"].items():
        gain = model["ensemble_gain"]
        lines.append(
            f"| {name} | {_number(model['evaluation']['mean'])} "
            f"| {_number(model['ensemble']['evaluation']['mean'])} "
            f"| {_number(gain['decision']['mean'], sign=True)} "
            f"({_number(gain['decision']['t'], 1, True)}) "
            f"| {_number(gain['evaluation']['mean'], sign=True)} "
            f"({_number(gain['evaluation']['t'], 1, True)}) |"
        )
    if document["incomplete"]:
        lines += ["", "Sin contrastar, por semillas que faltan:", ""]
        lines += [
            f"- {name}: semillas {', '.join(map(str, seeds))}"
            for name, seeds in document["incomplete"].items()
        ]
    return "\n".join(lines) + "\n"


def markdown_origins(document):
    years = document["years"]
    lines = [
        f"# Contrastes con {len(document['seeds'])} semillas en {len(years)} orígenes",
        "",
        f"Orígenes: {', '.join(map(str, years))}.",
        "El IC de cada sesión es la media de las semillas.",
        "",
        "## Entre orígenes",
        "",
        "Evaluación. Cada origen cuenta como una réplica: la t entre orígenes tiene K - 1 grados",
        "de libertad. La Δ por sesión junta todas las sesiones, con error con autocorrelación",
        "dentro de cada origen.",
        "",
        "| Modelo | Control | Orígenes | Δ media | Error entre orígenes | t entre orígenes (gl) "
        "| p | A favor | Δ por sesión (t) | Δ del conjunto (t entre orígenes) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, entry in document["across"].items():
        block, combined = entry["seeds"]["evaluation"], entry["ensemble"]["evaluation"]
        between, pooled, other = block["between"], block["pooled"], combined["between"]
        left_out = "".join(f" (sin {year})" for year in entry["missing"])
        lines.append(
            f"| {name} | {entry['control']} | {between['origins']}{left_out} "
            f"| {_number(between['mean'], sign=True)} | {_number(between['se'])} "
            f"| {_number(between['t'], 1, True)} ({between['df']}) | {_number(between['p'], 3)} "
            f"| {between['positive']} de {between['origins']} "
            f"| {_number(pooled['mean'], sign=True)} ({_number(pooled['t'], 1, True)}) "
            f"| {_number(other['mean'], sign=True)} ({_number(other['t'], 1, True)}) |"
        )
    lines += [
        "",
        "## Δ de evaluación de cada origen",
        "",
        f"| Modelo | {' | '.join(map(str, years))} |",
        f"| --- | {' | '.join('---' for _ in years)} |",
    ]
    for name, entry in document["across"].items():
        by_year = dict(
            zip(entry["years"], entry["seeds"]["evaluation"]["origin_means"], strict=True)
        )
        cells = " | ".join(_number(by_year.get(year), sign=True) for year in years)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--year", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--contrasts", type=Path, required=True, help="JSON con 'controls'")
    parser.add_argument("--output", type=Path, required=True, help="Carpeta nueva")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("La carpeta de salida ya existe")
    controls = json.loads(args.contrasts.read_text(encoding="utf-8"))["controls"]
    if len(args.year) == 1:
        document = build(args.state, args.year[0], args.seed, controls)
        table = markdown(document)
    else:
        document = build_origins(args.state, args.year, args.seed, controls)
        table = markdown_origins(document)
    args.output.mkdir(parents=True)
    text = json.dumps(document, ensure_ascii=False, indent=2, default=str) + "\n"
    (args.output / "seed-contrasts.json").write_text(text, encoding="utf-8")
    (args.output / "seed-contrasts.md").write_text(table, encoding="utf-8")
    print(table, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
