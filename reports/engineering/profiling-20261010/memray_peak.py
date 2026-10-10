"""Reparto del pico de memoria de una captura de memray por función del proyecto.

Lee el máximo de memoria viva de la captura (`get_high_watermark_allocation_records`) y
agrupa cada asignación viva en ese instante por el marco más interno que pertenece a
MARS-TITAN (`src/mars_titan`, `benchmarks` o este informe), por el marco Python más interno
y por el asignador. Una asignación sin marco del proyecto (por ejemplo, la importación de
una biblioteca) queda en «fuera del proyecto».

Uso: memray_peak.py CAPTURA.bin SALIDA.json [--top N]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from memray import AllocatorType, FileReader

PROJECT = ("/src/mars_titan/", "/benchmarks/", "/reports/engineering/profiling-")
OUTSIDE = "fuera del proyecto"


def _relative(filename):
    """Ruta sin la parte local: relativa al proyecto o a site-packages."""
    for marker in PROJECT:
        if marker in filename:
            return marker.strip("/").split("/")[-1] + "/" + filename.split(marker, 1)[1]
    if "site-packages/" in filename:
        return filename.split("site-packages/", 1)[1]
    if "/lib/python3" in filename:
        return "python/" + Path(filename).name
    return filename


def _project_frame(stack):
    """Marco más interno del proyecto. memray da la pila del más interno al más externo."""
    for function, filename, line in stack:
        if any(marker in filename for marker in PROJECT):
            return f"{_relative(filename)}:{line} {function}"
    return OUTSIDE


def _python_frame(stack):
    if not stack:
        return "sin pila de Python"
    function, filename, line = stack[0]
    return f"{_relative(filename)}:{line} {function}"


def _top(groups, total, top):
    ordered = sorted(groups.items(), key=lambda item: -item[1][0])[:top]
    return [
        dict(frame=key, bytes=size, share=size / total if total else None, allocations=count)
        for key, (size, count) in ordered
    ]


def summarize(path, top):
    reader = FileReader(path)
    metadata = reader.metadata
    by_project, by_python, by_allocator = (defaultdict(lambda: [0, 0]) for _ in range(3))
    total = 0
    for record in reader.get_high_watermark_allocation_records(merge_threads=True):
        stack = record.stack_trace()
        size, count = record.size, record.n_allocations
        total += size
        for groups, key in (
            (by_project, _project_frame(stack)),
            (by_python, _python_frame(stack)),
            (by_allocator, AllocatorType(record.allocator).name.lower()),
        ):
            groups[key][0] += size
            groups[key][1] += count
    return dict(
        capture=Path(path).name,
        peak_memory_bytes=metadata.peak_memory,
        live_at_peak_bytes=total,
        total_allocations=metadata.total_allocations,
        native_traces=metadata.has_native_traces,
        python_allocator=metadata.python_allocator,
        file_format=metadata.file_format.name.lower(),
        by_project_frame=_top(by_project, total, top),
        by_python_frame=_top(by_python, total, top),
        by_allocator=_top(by_allocator, total, top),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    report = summarize(args.capture, args.top)
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("peak_memory_bytes", "live_at_peak_bytes")}))


if __name__ == "__main__":
    main()
