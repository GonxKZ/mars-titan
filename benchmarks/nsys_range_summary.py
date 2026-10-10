"""Llamadas a CUDA, núcleos y copias dentro de cada rango NVTX de un perfil de Nsight Systems.

Lee la exportación SQLite de `nsys export --type sqlite`. Una llamada a la API pertenece a un
rango si empieza dentro de él, y un núcleo o una copia pertenece al rango de la llamada que
lo lanzó (por su `correlationId`). El tiempo de GPU es la suma de duraciones de núcleos, que
en un solo stream no se solapan. Con `--repeats N` divide además los totales entre las N
repeticiones del rango.

Uso: `python benchmarks/nsys_range_summary.py EXPORT.sqlite SALIDA.json [--repeats N] RANGO...`
"""

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path


def _ranges(db, names):
    rows = db.execute(
        "SELECT COALESCE(e.text, s.value), e.start, e.end FROM NVTX_EVENTS e "
        "LEFT JOIN StringIds s ON s.id = e.textId WHERE e.end IS NOT NULL"
    ).fetchall()
    found = {name: [(start, end) for text, start, end in rows if text == name] for name in names}
    missing = [name for name, spans in found.items() if len(spans) != 1]
    if missing:
        raise SystemExit(f"Cada rango debe aparecer una vez: {missing}")
    return {name: spans[0] for name, spans in found.items()}


def _copy_kinds(db):
    try:
        return dict(db.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER").fetchall())
    except sqlite3.OperationalError:
        return {}


def summary(path, names, repeats=1):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    strings = dict(db.execute("SELECT id, value FROM StringIds").fetchall())
    kinds = _copy_kinds(db)
    calls = db.execute(
        "SELECT start, end, nameId, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ).fetchall()
    kernels = dict(
        db.execute("SELECT correlationId, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    )
    copies = {
        row[0]: row[1:]
        for row in db.execute(
            "SELECT correlationId, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        ).fetchall()
    }
    result = {}
    for name, (begin, finish) in _ranges(db, names).items():
        api, api_ns = Counter(), Counter()
        kernel_count = kernel_ns = 0
        copy_count, copy_bytes = Counter(), Counter()
        for start, end, name_id, correlation in calls:
            if not begin <= start <= finish:
                continue
            call = strings.get(name_id, str(name_id)).split("_v")[0]
            api[call] += 1
            api_ns[call] += end - start
            if correlation in kernels:
                kernel_count += 1
                kernel_ns += kernels[correlation]
            if correlation in copies:
                kind, size = copies[correlation]
                label = kinds.get(kind, str(kind))
                copy_count[label] += 1
                copy_bytes[label] += size
        wall = finish - begin
        result[name] = dict(
            repeats=repeats,
            wall_ms_per_repeat=round(wall / repeats / 1e6, 4),
            kernels_per_repeat=round(kernel_count / repeats, 2),
            kernel_ms_per_repeat=round(kernel_ns / repeats / 1e6, 4),
            gpu_kernel_fraction=round(kernel_ns / wall, 4) if wall else None,
            api_calls_per_repeat={
                call: dict(
                    count=round(count / repeats, 2),
                    ms=round(api_ns[call] / repeats / 1e6, 4),
                )
                for call, count in api.most_common()
            },
            copies_per_repeat={
                label: dict(
                    count=round(count / repeats, 2),
                    bytes=round(copy_bytes[label] / repeats),
                )
                for label, count in copy_count.items()
            },
        )
    return result


if __name__ == "__main__":
    arguments = sys.argv[1:]
    repeats = 1
    if "--repeats" in arguments:
        index = arguments.index("--repeats")
        repeats = int(arguments[index + 1])
        del arguments[index : index + 2]
    export, output, *names = arguments
    Path(output).write_text(
        json.dumps(summary(export, names, repeats), ensure_ascii=False, indent=2) + "\n"
    )
