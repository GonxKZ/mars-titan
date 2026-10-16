"""Recibo de una medida de suite_parallelism.py (#491): ejecuciones y resumen por configuración.

Uso: receipt.py DIRECTORIO SALIDA.json [MUESTRAS.jsonl]

DIRECTORIO es la salida de suite_parallelism.py, con un JSON y un JUnit por ejecución. Las
ejecuciones de la primera medida no guardaban la memoria anónima del scope, que entonces se
muestreó desde fuera con un JSON por ejecución en MUESTRAS.jsonl (`run` y `max_anon`). El
recibo deja fuera el resultado de cada prueba y compara cada ejecución terminada con la primera
en serie: mismas pruebas, cada una con el mismo resultado (superada, omitida o fallida).

Chrome, Kaizten, Tomcat y uv también crean entradas en /tmp mientras dura una ejecución. Las
que podrían venir de las pruebas son las que empiezan como las de tempfile, pytest o torch.
"""

import json
import sys
from pathlib import Path
from statistics import median

OWN_TMP_PREFIXES = ("tmp", "pytest", "mars", "torch")

KEEP = (
    "config",
    "round",
    "started",
    "load_before",
    "load_after",
    "command",
    "returncode",
    "seconds",
    "anon_peak_bytes",
    "peak_bytes",
    "oom_kill",
    "max_events",
    "stopped_before_measure",
    "stop_reason",
    "new_tmp_entries",
)


def counts(outcomes):
    result = {}
    for kind in outcomes.values():
        result[kind] = result.get(kind, 0) + 1
    return result


def main():
    directory, output = Path(sys.argv[1]), Path(sys.argv[2])
    sampled = {}
    if len(sys.argv) > 3:
        for line in Path(sys.argv[3]).read_text().splitlines():
            record = json.loads(line)
            sampled[record["run"]] = record["max_anon"]
    records = [
        json.loads(path.read_text())
        for path in sorted(directory.glob("*-*.json"))
        if not path.name.endswith(".measure.json")
    ]
    records.sort(key=lambda r: r["started"])
    reference = next(r["outcomes"] for r in records if r["config"] == "serie" and "outcomes" in r)
    runs = []
    for record in records:
        run = {key: record[key] for key in KEEP if key in record}
        name = f"{record['round']}-{record['config']}"
        if "anon_peak_bytes" not in run and name in sampled:
            run["anon_peak_bytes"] = sampled[name]
            run["anon_peak_source"] = "muestreo externo cada segundo"
        run["finished"] = record["returncode"] == 0 and "outcomes" in record
        own = [n for n in record["new_tmp_entries"] if n.lower().startswith(OWN_TMP_PREFIXES)]
        run["possible_own_tmp_entries"] = own
        if "outcomes" in record:
            run["counts"] = counts(record["outcomes"])
            run["same_outcomes_as_serial"] = record["outcomes"] == reference
        runs.append(run)
    configs = {}
    for run in runs:
        configs.setdefault(run["config"], []).append(run)
    summary = {}
    for config, items in configs.items():
        done = [r for r in items if r["finished"]]
        seconds = [r["seconds"] for r in done]
        anon = [r["anon_peak_bytes"] for r in items if "anon_peak_bytes" in r]
        summary[config] = dict(
            runs=len(items),
            finished=len(done),
            stopped=len(items) - len(done),
            seconds=sorted(seconds),
            median_seconds=median(seconds) if seconds else None,
            max_anon_peak_bytes=max(anon) if anon else None,
            all_same_outcomes=all(r.get("same_outcomes_as_serial") for r in done) if done else None,
            possible_own_tmp_entries=sum(len(r["possible_own_tmp_entries"]) for r in items),
        )
    receipt = dict(
        kind="test_suite_parallelism",
        issue=491,
        reference_counts=counts(reference),
        configs=summary,
        runs=runs,
        optimizer_steps=0,
        training_executed=False,
    )
    output.write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + "\n")
    for config, value in summary.items():
        print(config, json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
