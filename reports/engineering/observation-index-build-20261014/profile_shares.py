"""Parte inclusiva de cada función en el hilo principal de un perfil `py-spy --format raw`.

Uso: profile_shares.py PERFIL.txt SALIDA.json FUNCIÓN [...]

Una función cuenta en una muestra si aparece en su pila, una sola vez aunque se repita. La parte
propia de `_labels` es la de las muestras cuya pila termina en esa función.
"""

import json
import sys
from collections import Counter
from pathlib import Path

source, out, *names = sys.argv[1:]
inclusive, own, total = Counter(), Counter(), 0
for line in Path(source).read_text().splitlines():
    stack, _, count = line.rpartition(" ")
    thread, *frames = stack.split(";")
    if not thread.endswith("MainThread"):
        continue
    frames = [frame.split(" (")[0] for frame in frames]
    total += int(count)
    for name in set(frames) & set(names):
        inclusive[name] += int(count)
    own[frames[-1] if frames else thread] += int(count)
Path(out).write_text(
    json.dumps(
        dict(
            profile=Path(source).name,
            main_thread_samples=total,
            inclusive_percent={n: round(100 * inclusive[n] / total, 1) for n in names},
            own_percent={n: round(100 * own[n] / total, 1) for n in names},
        ),
        indent=1,
    )
    + "\n"
)
print(Path(out).read_text())
