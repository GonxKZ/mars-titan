"""Resumen de un perfil de py-spy en formato `raw` (pilas plegadas con su número de muestras).

Cuenta las muestras de las pilas que pasan por la función RAÍZ y, dentro de ellas, la parte
inclusiva de cada función pedida y las funciones hoja con más muestras propias. Una función
se identifica por su nombre y su archivo, sin el número de línea, así que todas sus líneas
suman juntas y una recursión cuenta una vez por muestra. `nombre@archivo` limita una función
a los marcos cuyo archivo contiene ese texto, para separar funciones con el mismo nombre.

Uso: `python benchmarks/collapsed_profile_summary.py PERFIL SALIDA.json RAÍZ [FUNCIÓN ...]`
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

FRAME = re.compile(r"^(?P<name>.+) \((?P<file>.+?)(?::\d+)?\)$")
TOP = 15


def frame_key(frame):
    match = FRAME.match(frame)
    return f"{match['name']} ({match['file']})" if match else frame


def matches(function, frames):
    """Si alguna pareja (nombre, archivo) de la pila corresponde a `nombre[@archivo]`."""
    name, _, place = function.partition("@")
    return any(frame == name and place in file for frame, file in frames)


def summary(path, root, functions):
    total = inside = 0
    inclusive, leaves = Counter(), Counter()
    for line in Path(path).read_text().splitlines():
        stack, _, count = line.rpartition(" ")
        samples = int(count)
        total += samples
        keys = [frame_key(frame) for frame in stack.split(";")]
        frames = {(key.split(" (")[0], key.partition(" (")[2]) for key in keys}
        if not matches(root, frames):
            continue
        inside += samples
        for function in functions:
            if matches(function, frames):
                inclusive[function] += samples
        leaves[keys[-1]] += samples

    def share(value):
        return round(value / inside, 4) if inside else None

    return dict(
        profile=Path(path).name,
        samples=total,
        root=root,
        root_samples=inside,
        inclusive={
            name: dict(samples=inclusive[name], share=share(inclusive[name])) for name in functions
        },
        top_self=[
            dict(frame=key, samples=value, share=share(value))
            for key, value in leaves.most_common(TOP)
        ],
    )


if __name__ == "__main__":
    profile, output, root, *names = sys.argv[1:]
    Path(output).write_text(
        json.dumps(summary(profile, root, names), ensure_ascii=False, indent=2) + "\n"
    )
