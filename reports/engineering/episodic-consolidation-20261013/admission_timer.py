"""Cronometra la admisión del banco episódico en la ventana que mide chronological_replay.py.

Uso: admission_timer.py SALIDA.jsonl ETIQUETA -- benchmarks/chronological_replay.py ARGUMENTOS...

Envuelve `RetentionBank.propose`, la selección de medoids que llama la propuesta y
`bank_snapshot`, y ejecuta el benchmark sin cambiar nada más. Al terminar recorta cada
llamada a la ventana que mide el benchmark (de la marca posterior al calentamiento a la
última) y añade una línea a SALIDA.jsonl. Las llamadas anidadas de una misma función solo
cuentan una vez. Ninguna de las tres funciones depende del cronómetro, así que los bits de
la captura son los mismos con él y sin él.
"""

import json
import runpy
import sys
import time
from pathlib import Path

output, label = Path(sys.argv[1]), sys.argv[2]
if sys.argv[3] != "--":
    raise SystemExit("falta -- antes del benchmark")
script = sys.argv[4]
sys.argv = [script, *sys.argv[5:]]

# Ninguno de estos dos módulos importa los que parchea chronological_replay_patches.py.
import mars_titan.memory.financial_consumers as consumers  # noqa: E402
import mars_titan.memory.retention_bank as retention  # noqa: E402

SPANS = {}


def timed(name, function):
    spans, depth = SPANS.setdefault(name, []), [0]

    def wrapper(*args, **kwargs):
        depth[0] += 1
        began = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            depth[0] -= 1
            if depth[0] == 0:
                spans.append((began, time.perf_counter()))

    wrapper.__wrapped__ = function
    return wrapper


retention.RetentionBank.propose = timed("propose", retention.RetentionBank.propose)
retention.select_anchored_medoids = timed(
    "select_anchored_medoids", retention.select_anchored_medoids
)
# mars_titan_run importa bank_snapshot por nombre más tarde, así que recibe la envuelta.
consumers.bank_snapshot = timed("bank_snapshot", consumers.bank_snapshot)

namespace = runpy.run_path(script, run_name="__main__")
budget, warmup = namespace["budget"], namespace["args"].warmup
first, last = budget.marks[warmup][0], budget.marks[-1][0]


def clipped(spans):
    inside = [(max(a, first), min(b, last)) for a, b in spans if b > first and a < last]
    return len(inside), sum(b - a for a, b in inside)


record = dict(label=label, window_seconds=last - first)
for name, spans in SPANS.items():
    calls, seconds = clipped(spans)
    record[name] = dict(
        calls=calls,
        seconds=seconds,
        fraction=seconds / (last - first),
        calls_total=len(spans),
        seconds_total=sum(b - a for a, b in spans),
    )
with output.open("a") as handle:
    handle.write(json.dumps(record) + "\n")
print(json.dumps(record), flush=True)
