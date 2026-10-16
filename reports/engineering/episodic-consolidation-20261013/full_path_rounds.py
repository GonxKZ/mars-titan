"""Recibo de las rondas A B B A del recorrido completo de los lectores (#547 y #628).

Uso: full_path_rounds.py ISSUE DIRECTORIO BITS.json SALIDA.json

DIRECTORIO es el de las rondas: `results.jsonl` de `benchmarks/chronological_replay.py`, si
existe `admission.jsonl` de `admission_timer.py` y, por ronda, `ETIQUETA.pmon` con el uso de
la SM por proceso cada segundo (`nvidia-smi pmon -s u -d 1`). BITS.json es una lista con las
salidas de `compare_capture.py`.

Una ronda está limpia si el llama-server de Kaizten no usó la SM entre la primera y la última
muestra en que la usó la ronda. pmon lista los procesos de cada segundo por PID creciente,
así que una muestra nueva empieza cuando el PID no crece. Las etiquetas siguen la forma
`MODELO[-qN]-rK-LADO[-ÁRBOL]`, donde qN es la tanda (sin sufijo, la primera).
"""

import json
import re
import sys
from pathlib import Path
from statistics import mean

LABEL = re.compile(r"^(?P<arm>.+?)(?:-q(?P<q>\d+))?-r(?P<r>\d)-(?P<side>[AB])(?:-(?P<tree>.+))?$")
FUNCTIONS = ("propose", "select_anchored_medoids", "bank_snapshot")


def gpu_samples(path):
    """Muestras con SM de la ronda y cuántas de ellas tuvo también el llama-server."""
    sample, previous, own, foreign = 0, -1, set(), set()
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#") or len(fields) < 4:
            continue
        pid = int(fields[1]) if fields[1].isdigit() else -1
        if pid <= previous:
            sample += 1
        previous = pid
        if not (fields[3].isdigit() and int(fields[3]) > 0):
            continue
        if "python" in fields[-1]:
            own.add(sample)
        elif "llama-server" in fields[-1]:
            foreign.add(sample)
    if not own:
        return dict(own_samples=0, foreign_samples=0, clean=False)
    span = range(min(own), max(own) + 1)
    busy = sum(sample in foreign for sample in span)
    return dict(own_samples=len(span), foreign_samples=busy, clean=busy == 0)


def admission(record):
    if record is None:
        return None
    result = dict(window_seconds=record["window_seconds"])
    for name in FUNCTIONS:
        part = record[name]
        result[name] = dict(
            calls=part["calls"],
            seconds=part["seconds"],
            ms_per_call=1000 * part["seconds"] / part["calls"],
            fraction=part["fraction"],
        )
    return result


def ratios(items):
    """Filas/s de cada lado, cociente de medias y peor par (B más lenta entre A más rápida)."""
    a = [i["rows_per_s"] for i in items if i["side"] == "A"]
    b = [i["rows_per_s"] for i in items if i["side"] == "B"]
    if len(a) < 2 or len(b) < 2:
        return None
    return dict(
        rounds=[i["label"] for i in items],
        rows_per_s_A=a,
        rows_per_s_B=b,
        mean_ratio=mean(b) / mean(a),
        worst_pair=min(b) / max(a),
    )


def summary(rounds):
    """Cocientes de B entre A en cada tanda limpia entera y con todas las rondas limpias juntas.

    Una tanda está limpia entera si sus cuatro rondas lo están. Juntar las rondas limpias exige
    al menos dos de cada lado.
    """
    tandas = {}
    for item in rounds:
        tandas.setdefault(item["tanda"], []).append(item)
    clean = [
        q
        for q, items in sorted(tandas.items())
        if len(items) == 4 and all(i["clean"] for i in items)
    ]
    result = dict(
        clean_tandas=clean,
        per_tanda={q: ratios(tandas[q]) for q in clean},
        pooled_clean=ratios([i for i in rounds if i["clean"]]),
    )
    for name in FUNCTIONS:
        for side in "AB":
            times = [
                i["admission"][name]["ms_per_call"]
                for i in rounds
                if i["side"] == side and i["admission"] and i["clean"]
            ]
            if times:
                result[f"{name}_ms_per_call_{side}"] = [min(times), max(times)]
    return result


def main():
    issue = int(sys.argv[1])
    directory, bits_path, output = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    timers, timer_path = {}, directory / "admission.jsonl"
    for line in timer_path.read_text().splitlines() if timer_path.exists() else []:
        record = json.loads(line)
        timers[record["label"]] = record
    arms = {}
    for line in (directory / "results.jsonl").read_text().splitlines():
        record = json.loads(line)
        match = LABEL.match(record["label"])
        if match is None:
            raise SystemExit(f"etiqueta inesperada: {record['label']}")
        keep = (
            "rows_per_s",
            "seconds",
            "rows",
            "labels",
            "optimizer_calls",
            "peak_allocated_mib",
            "peak_reserved_mib",
            "peak_rss_mib",
            "setup_seconds",
            "pass_seconds",
            "loadavg",
            "tf32_matmul",
            "tf32_cudnn",
            "matmul_precision",
        )
        round_ = dict(
            label=record["label"],
            tanda=int(match["q"] or 1),
            round=int(match["r"]),
            side=match["side"],
            tree=match["tree"],
            **{key: record[key] for key in keep},
            **gpu_samples(directory / f"{record['label']}.pmon"),
            admission=admission(timers.get(record["label"])),
        )
        arms.setdefault(match["arm"], []).append(round_)
    bits = json.loads(bits_path.read_text())
    receipt = dict(
        kind="chronological_replay_abba",
        issue=issue,
        arms={arm: dict(rounds=rounds, summary=summary(rounds)) for arm, rounds in arms.items()},
        bits=bits,
        bits_all_equal=all(
            b["gradients_equal"] and b["names_equal"] and b["pending_equal"] for b in bits
        ),
        optimizer_steps=0,
        training_executed=False,
    )
    output.write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + "\n")
    for arm, value in receipt["arms"].items():
        print(arm, json.dumps(value["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
