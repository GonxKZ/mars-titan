"""Memoria viva en el pico de una captura de memray, por lugar del proyecto y por biblioteca.

Funciona con capturas completas y agregadas (`memray run --aggregate`). Cada reserva viva en
el pico se atribuye a su marco más interno de `mars_titan` o de `benchmarks` (si no hay
ninguno, al marco más interno) y a la biblioteca de su marco de Python más interno.

Uso: `python benchmarks/memray_peak_summary.py CAPTURA.bin SALIDA.json`
"""

import json
import sys
from collections import Counter
from pathlib import Path

from memray import FileReader

TOP = 15
LIBRARIES = ("torch", "numpy", "mars_titan", "benchmarks", "memray")


def _short(path):
    for marker in ("site-packages/", "src/", "worktrees/"):
        if marker in path:
            path = path.split(marker, 1)[1]
    return path


def _library(path):
    return next((name for name in LIBRARIES if f"/{name}/" in path), "otros")


def summary(path):
    reader = FileReader(path)
    places, libraries = Counter(), Counter()
    total = 0
    for record in reader.get_high_watermark_allocation_records(merge_threads=True):
        stack = record.stack_trace()
        total += record.size
        own = next(
            (frame for frame in stack if "/mars_titan/" in frame[1] or "/benchmarks/" in frame[1]),
            stack[0] if stack else None,
        )
        place = f"{own[0]} ({_short(own[1])}:{own[2]})" if own else "sin pila de Python"
        places[place] += record.size
        libraries[_library(stack[0][1]) if stack else "sin pila de Python"] += record.size
    metadata = reader.metadata
    return dict(
        capture=Path(path).name,
        file_format=metadata.file_format.name,
        peak_bytes=metadata.peak_memory,
        live_at_peak_bytes=total,
        total_allocations=metadata.total_allocations,
        by_library=dict(libraries.most_common()),
        top_places=[dict(place=key, bytes=value) for key, value in places.most_common(TOP)],
    )


if __name__ == "__main__":
    capture, output = sys.argv[1:]
    Path(output).write_text(json.dumps(summary(capture), ensure_ascii=False, indent=2) + "\n")
