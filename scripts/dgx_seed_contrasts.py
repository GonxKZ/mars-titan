#!/usr/bin/env python3
"""Contrastes con varias semillas: cada modelo frente a su control en un origen.

Las semillas de un mismo origen recorren las mismas sesiones, así que no son réplicas
independientes de una sesión. El IC de rangos de cada sesión se promedia entre semillas y el
contraste pareado con su control se hace sobre esa serie, con error con autocorrelación. La
diferencia de cada semilla se informa aparte, para ver si el signo se repite.

Un modelo al que le falta alguna semilla no se contrasta con las que tenga: se anota como
incompleto.

Uso:
    dgx_seed_contrasts.py --state <estado> --year 2013 --seed 42 --seed 43 --seed 44 \
        --contrasts <controles.json> --output <carpeta nueva>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


def seed_sessions(state, model, year, seed):
    """IC de rangos por sesión de cada tramo de una ejecución, o None si le falta alguno."""
    report = _report_module()
    run = Path(state) / "runs" / model / str(year) / f"seed-{seed}"
    result = {}
    for split in SPLITS:
        data = report.load_split(run, split)
        if data is None:
            return None
        result[split] = report.session_ic(data)
    return result


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


def _pooled(sessions, splits):
    return {(split, *key): value for split in splits for key, value in sessions[split].items()}


def _level(sessions, per_seed):
    report = _report_module()
    entry = {split: report.summary(sessions[split]) for split in SPLITS}
    entry["decision"] = report.summary(_pooled(sessions, DECISION))
    entry["per_seed"] = {
        split: [float(np.mean(list(one[split].values()))) for one in per_seed] for split in SPLITS
    }
    return entry


def build(state, year, seeds, controls):
    """Niveles de cada modelo y contraste de cada uno con su control."""
    report = _report_module()
    seeds = list(dict.fromkeys(seeds))
    names = sorted(set(controls) | set(controls.values()))
    per_seed, missing = {}, {}
    for name in names:
        runs = [seed_sessions(state, name, year, seed) for seed in seeds]
        absent = [seed for seed, run in zip(seeds, runs, strict=True) if run is None]
        if absent:
            missing[name] = absent
        else:
            per_seed[name] = runs
    mean = {name: averaged(runs, name) for name, runs in per_seed.items()}
    document = dict(
        schema_version=1,
        kind="dgx_seed_contrasts_v1",
        final_test_opened=False,
        year=year,
        seeds=seeds,
        models={name: _level(mean[name], per_seed[name]) for name in mean},
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
        document["contrasts"][name] = entry
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
        "",
        "| Modelo | Control | Semillas | IC de evaluación | Δ decisión (t) | Δ evaluación (t) "
        "| Signo por semilla |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, entry in document["contrasts"].items():
        level = document["models"][name]["evaluation"]["mean"]
        decision, evaluation = entry["decision"], entry["evaluation"]
        lines.append(
            f"| {name} | {entry['control']} | {len(entry['seeds'])} | {_number(level)} "
            f"| {_number(decision['mean'], sign=True)} ({_number(decision['t'], 1, True)}) "
            f"| {_number(evaluation['mean'], sign=True)} ({_number(evaluation['t'], 1, True)}) "
            f"| {entry['signs']['evaluation']} |"
        )
    if document["incomplete"]:
        lines += ["", "Sin contrastar, por semillas que faltan:", ""]
        lines += [
            f"- {name}: semillas {', '.join(map(str, seeds))}"
            for name, seeds in document["incomplete"].items()
        ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--contrasts", type=Path, required=True, help="JSON con 'controls'")
    parser.add_argument("--output", type=Path, required=True, help="Carpeta nueva")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("La carpeta de salida ya existe")
    controls = json.loads(args.contrasts.read_text(encoding="utf-8"))["controls"]
    document = build(args.state, args.year, args.seed, controls)
    args.output.mkdir(parents=True)
    text = json.dumps(document, ensure_ascii=False, indent=2, default=str) + "\n"
    (args.output / "seed-contrasts.json").write_text(text, encoding="utf-8")
    table = markdown(document)
    (args.output / "seed-contrasts.md").write_text(table, encoding="utf-8")
    print(table, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
