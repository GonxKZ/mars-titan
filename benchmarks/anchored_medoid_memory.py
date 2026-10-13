"""Memoria real de una selección anclada frente a la que declara, con memray.

Usa el problema de `anchored_medoid_options.py` (la forma de las llamadas de los lectores)
y registra con `memray.Tracker` las reservas hechas durante la llamada, de Python, de NumPy
y de las bibliotecas nativas. En el pico separa por la pila nativa (las funciones `cblas_`)
lo que reserva OpenBLAS de lo que reserva la propia selección, que es lo que cuenta
`estimated_peak_bytes`.

El área de trabajo que OpenBLAS guarda por proceso se mide antes y aparte, con un producto
de matrices de la forma de un bloque del fondo: bytes que registra memray y cambio de la
memoria virtual y de la residente anónima de un intérprete nuevo sin memray.

Uso: `OMP_NUM_THREADS=2 PYTHONPATH=src python benchmarks/anchored_medoid_memory.py CARPETA`
"""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import memray
import numpy as np
from anchored_medoid_options import OPTIONS, RETAINED, problem
from threadpoolctl import threadpool_info

from mars_titan.cm.anchored_medoids import select_anchored_medoids

VARIANTS = {"reference": {}, "bounded_and_reused": OPTIONS}


def status(field):
    """Valor en bytes de un campo de /proc/self/status."""
    with open("/proc/self/status", encoding="ascii") as handle:
        for line in handle:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    raise KeyError(field)


def tracked(capture, function):
    """Ejecuta `function` bajo memray y devuelve su resultado y las reservas en el pico."""
    capture.unlink(missing_ok=True)
    with memray.Tracker(capture, native_traces=True, trace_python_allocators=True):
        result = function()
    reader = memray.FileReader(capture)
    blas, own, places = 0, 0, Counter()
    for record in reader.get_high_watermark_allocation_records(merge_threads=True):
        # NumPy entra en OpenBLAS por sus funciones `cblas_`. Los marcos nativos no traen
        # el archivo de la biblioteca, así que se reconocen por el símbolo.
        if any("cblas_" in frame[0] for frame in record.native_stack_trace()):
            blas += record.size
            continue
        own += record.size
        stack = record.stack_trace()
        places[f"{stack[0][0]}:{stack[0][2]}" if stack else "sin pila de Python"] += record.size
    peak = reader.metadata.peak_memory
    capture.unlink()
    return result, peak, blas, own, places


def first_product(rows, columns, dimensions):
    """Memoria virtual y residente anónima que añade el primer producto de un proceso."""
    rng = np.random.default_rng(0)
    left = rng.standard_normal((rows, dimensions))
    right = rng.standard_normal((columns, dimensions))
    out = np.empty((rows, columns))
    before = {name: status(name) for name in ("VmSize", "RssAnon")}
    np.matmul(left, right.T, out=out)
    return dict(
        vm_size_delta_bytes=status("VmSize") - before["VmSize"],
        rss_anon_delta_bytes=status("RssAnon") - before["RssAnon"],
    )


def blas_workspace(folder, shape, dimensions):
    """Primer producto de matrices con la forma de un bloque del fondo.

    memray registra sus reservas en este proceso. La memoria del proceso se lee en otro
    intérprete sin memray, porque el propio registro también ocupa memoria.
    """
    rng = np.random.default_rng(0)
    left = rng.standard_normal((shape[0], dimensions))
    right = rng.standard_normal((shape[1], dimensions))
    out = np.empty(shape)
    _, peak, blas, _, _ = tracked(folder / "blas.bin", lambda: np.matmul(left, right.T, out=out))
    fresh = subprocess.run(
        [sys.executable, __file__, "--first-product", *map(str, (*shape, dimensions))],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(
        tile_shape=list(shape),
        dimensions=dimensions,
        memray_peak_bytes=peak,
        memray_blas_bytes=blas,
        **json.loads(fresh.stdout),
    )


def main(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    points, ids, fixed, candidates = problem("clustered")
    rows, workspace = {}, None
    for name, options in VARIANTS.items():
        if options.get("bounded_background") and workspace is None:
            # La referencia no usa la BLAS, así que este es el primer producto del proceso.
            shape = rows["reference"]["background_tile_shape"]
            workspace = blas_workspace(folder, shape, points.shape[1])
        result, peak, blas, own, places = tracked(
            folder / f"{name}.bin",
            lambda options=options: select_anchored_medoids(
                points, ids, fixed, candidates, RETAINED, **options
            ),
        )
        rows[name] = dict(
            estimated_peak_bytes=result.estimated_peak_bytes,
            memray_peak_bytes=peak,
            blas_bytes_at_peak=blas,
            own_bytes_at_peak=own,
            within_estimate=own <= result.estimated_peak_bytes,
            largest_own_at_peak=[
                dict(location=place, bytes=size) for place, size in places.most_common(6)
            ],
            background_tile_shape=list(result.background_tile_shape),
            applied=dict(
                reused_distances=result.reused_distances,
                bounded_background=result.bounded_background,
            ),
        )
    report = dict(
        problem="clustered",
        tracker="memray.Tracker, native_traces=True, trace_python_allocators=True",
        memray=memray.__version__,
        numpy=np.__version__,
        blas=[
            {
                key: info.get(key)
                for key in ("internal_api", "version", "num_threads", "architecture")
            }
            for info in threadpool_info()
            if info.get("user_api") == "blas"
        ],
        threads={
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        blas_workspace=workspace,
        variants=rows,
        limits=[
            "El área de trabajo de OpenBLAS se reserva una vez por proceso y no se libera.",
            "La memoria residente depende de las páginas que toque cada producto.",
        ],
    )
    (folder / "anchored-medoid-memory.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    if not all(row["within_estimate"] for row in rows.values()):
        raise SystemExit("La memoria propia de una variante supera su estimación")


if __name__ == "__main__":
    if sys.argv[1] == "--first-product":
        print(json.dumps(first_product(*map(int, sys.argv[2:5]))))
    else:
        main(sys.argv[1])
