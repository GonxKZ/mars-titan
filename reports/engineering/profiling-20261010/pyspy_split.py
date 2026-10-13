"""Reparto del tiempo de Python de un recorrido por función a partir de dos perfiles de py-spy.

Lee dos capturas `--format raw --threads` del mismo recorrido: una con todas las muestras
activas y otra con `--gil`, que solo guarda las muestras en las que el hilo tiene el GIL.
Se queda con el hilo principal (el que ejecuta el guion medido) y con las muestras dentro de
`_train_pass`. Para cada función devuelve:

- `self_share`: fracción de las muestras del recorrido cuyo marco más interno es la función.
- `inclusive_share`: fracción de las muestras en las que la función está en la pila.
- `gil_fraction`: fracción de ese tiempo con el GIL, como cociente de las tasas de muestras
  de las dos capturas, cada una dividida por la duración de su recorrido.

PyTorch suelta el GIL mientras ejecuta un operador desde Python, así que el tiempo con el GIL
es el del intérprete y el de la preparación de cada llamada, y el resto es el de dentro de los
operadores, las copias y las esperas a la GPU llamadas desde esa línea.

Varias capturas de cada modo se suman, cada una con la duración de su recorrido.

Uso: pyspy_split.py --all A1 A2 --gil G1 G2 --all-seconds S1 S2 --gil-seconds S1 S2
     --output SALIDA.json [--script chronological_replay.py] [--top 40]
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

THREAD = re.compile(r"^thread \((?P<id>\w+)\)(?:: (?P<name>.*))?$")
FRAME = re.compile(r"^(?P<function>.+?) \((?P<file>[^()]*?)(?::(?P<line>\d+))?\)$")
PROJECT = ("/src/mars_titan/", "/benchmarks/")


def _short(path):
    """Ruta sin la parte local del equipo."""
    for marker in PROJECT:
        if marker in path:
            return marker.strip("/").split("/")[-1] + "/" + path.split(marker, 1)[1]
    if "site-packages/" in path:
        return path.split("site-packages/", 1)[1]
    if "/lib/python3" in path:
        return "python/" + Path(path).name
    return Path(path).name if path.startswith("/") else path


def parse(path):
    """Muestras por hilo: lista de (pila de funciones del más externo al más interno, cuenta)."""
    threads = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        stack, _, count = line.rpartition(" ")
        frames = stack.split(";")
        thread = "sin hilo"
        match = THREAD.match(frames[0]) if frames else None
        if match:
            thread, frames = match["name"] or match["id"], frames[1:]
        parsed = []
        for frame in frames:
            found = FRAME.match(frame)
            if found:
                parsed.append(f"{_short(found['file'])} {found['function']}")
            elif not frame.startswith("process "):
                parsed.append(frame)
        threads[thread].append((parsed, int(count)))
    return threads


def main_thread(threads, script):
    """El hilo con más muestras que pasan por el guion medido (con `--threads`, MainThread)."""
    scored = {
        thread: sum(count for stack, count in samples if any(script in f for f in stack))
        for thread, samples in threads.items()
    }
    return max(scored, key=scored.get)


def in_pass(stack):
    return any(frame.endswith(" _train_pass") for frame in stack)


def tally(samples):
    own, inclusive, total = Counter(), Counter(), 0
    for stack, count in samples:
        if not in_pass(stack):
            continue
        total += count
        if stack:
            own[stack[-1]] += count
        for frame in set(stack):
            inclusive[frame] += count
    return own, inclusive, total


def capture(paths, script):
    """Suma de varias capturas del mismo modo, cada una con su hilo principal."""
    own, inclusive, total, samples, main_samples, per_capture = Counter(), Counter(), 0, 0, 0, []
    for path in paths:
        threads = parse(path)
        main = main_thread(threads, script)
        part_own, part_inclusive, part_total = tally(threads[main])
        own.update(part_own)
        inclusive.update(part_inclusive)
        total += part_total
        samples += sum(count for values in threads.values() for _, count in values)
        main_samples += sum(count for _, count in threads[main])
        per_capture.append(part_total)
    return dict(
        own=own,
        inclusive=inclusive,
        total=total,
        samples=samples,
        main_samples=main_samples,
        pass_samples=per_capture,
    )


def summarize(args):
    if len(args.all) != len(args.all_seconds) or len(args.gil) != len(args.gil_seconds):
        raise SystemExit("Cada captura necesita la duración de su recorrido")
    captures = {"all": capture(args.all, args.script), "gil": capture(args.gil, args.script)}
    seconds = {"all": sum(args.all_seconds), "gil": sum(args.gil_seconds)}

    def rate(name, value):
        return value / seconds[name]

    everything = captures["all"]

    def rows(kind, top):
        counter = everything[kind]
        result = []
        for frame, count in counter.most_common(top):
            gil = captures["gil"][kind].get(frame, 0)
            result.append(
                dict(
                    frame=frame,
                    samples=count,
                    share=count / everything["total"],
                    gil_samples=gil,
                    gil_fraction=rate("gil", gil) / rate("all", count) if count else None,
                )
            )
        return result

    pass_rate = rate("all", everything["total"]) / args.rate
    gil_rate = rate("gil", captures["gil"]["total"]) / args.rate
    return dict(
        sampling_rate_hz=args.rate,
        pass_seconds={"all": args.all_seconds, "gil": args.gil_seconds},
        samples={
            name: {key: value[key] for key in ("samples", "main_samples", "total", "pass_samples")}
            for name, value in captures.items()
        },
        active_fraction_by_capture=[
            count / (seconds * args.rate)
            for count, seconds in zip(
                captures["all"]["pass_samples"], args.all_seconds, strict=True
            )
        ],
        gil_fraction_by_capture=[
            count / (seconds * args.rate)
            for count, seconds in zip(
                captures["gil"]["pass_samples"], args.gil_seconds, strict=True
            )
        ],
        main_thread_active_fraction=pass_rate,
        main_thread_gil_fraction=gil_rate,
        self_by_function=rows("own", args.top),
        inclusive_by_function=rows("inclusive", args.top),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", type=Path, nargs="+", required=True)
    parser.add_argument("--gil", type=Path, nargs="+", required=True)
    parser.add_argument("--all-seconds", type=float, nargs="+", required=True)
    parser.add_argument("--gil-seconds", type=float, nargs="+", required=True)
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--script", default="chronological_replay.py")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args)
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("main_thread_active_fraction", "main_thread_gil_fraction")
            }
        )
    )


if __name__ == "__main__":
    main()
