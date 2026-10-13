"""Razones emparejadas de tiempo entre variantes de native-ppo-release y la compilación base.

Cada ronda de `benchmarks/native_build_variants.py run` ejecuta todas las variantes una vez,
en un orden que rota entre rondas, así que la razón variante/base de una misma ronda
comparte la carga de la máquina en ese momento. mold no cambia el código generado, solo la
disposición del ejecutable, y sirve de control del ruido entre procesos.

La regla de adopción se fijó tras la primera tanda y antes de ver la segunda: una variante
gana si en las dos tandas y en los cuatro escenarios (US y CN, CPU y cuda:0) la mediana de
su razón queda por debajo de 1 y por debajo de la mediana del control mold, con las mismas
huellas de contenido que base.

Uso: runtime_ratios.py SALIDA TANDA=PREFIJO [TANDA=PREFIJO ...]

Cada escenario se lee de PREFIJO seguido de su nombre y `.json`, por ejemplo
`evidence/runtime-t1-US-cpu.json` con el prefijo `evidence/runtime-t1-`.
"""

import argparse
import json
import statistics
from pathlib import Path

SCENARIOS = ("US-cpu", "US-cuda", "CN-cpu", "CN-cuda")
CONTROL = "mold"


def ratios(report, label):
    rounds = {}
    for run in report["runs"]:
        rounds.setdefault(run["round"], {})[run["label"]] = run
    return [
        rounds[index][label]["phases"]["total_seconds"]
        / rounds[index]["base"]["phases"]["total_seconds"]
        for index in sorted(rounds)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("batches", nargs="+")
    args = parser.parse_args()

    batches, verdict = {}, {}
    for item in args.batches:
        name, _, prefix = item.partition("=")
        scenarios = {}
        for scenario in SCENARIOS:
            report = json.loads(Path(f"{prefix}{scenario}.json").read_text(encoding="utf-8"))
            labels = [label for label in report["variants"] if label != "base"]
            entry = {"parity": report["digest_parity"], "rounds": report["rounds"]}
            for label in labels:
                values = ratios(report, label)
                entry[label] = {
                    "ratios": values,
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                }
            scenarios[scenario] = entry
            control = entry[CONTROL]["median"]
            for label in labels:
                if label == CONTROL:
                    continue
                wins = (
                    all(entry["parity"].values())
                    and entry[label]["median"] < 1
                    and entry[label]["median"] < control
                )
                verdict.setdefault(label, []).append(wins)
        batches[name] = scenarios
    payload = {
        "rule": "mediana < 1 y < mediana de mold en todos los escenarios y tandas, con paridad",
        "batches": batches,
        "adopted": {label: all(values) for label, values in verdict.items()},
        "scenarios_won": {label: sum(values) for label, values in verdict.items()},
        "scenarios_total": {label: len(values) for label, values in verdict.items()},
    }
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    for name, scenarios in batches.items():
        for scenario, entry in scenarios.items():
            medians = {
                label: round(value["median"], 3)
                for label, value in entry.items()
                if isinstance(value, dict) and "median" in value
            }
            print(name, scenario, medians)
    print(json.dumps({key: payload[key] for key in ("adopted", "scenarios_won")}))


if __name__ == "__main__":
    main()
