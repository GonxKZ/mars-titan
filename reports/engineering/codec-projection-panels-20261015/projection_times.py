"""Tiempo de `project_codec_block` con varios enlaces, alternados muestra a muestra.

Uso: projection_times.py SALIDA.json NOMBRE=ENLACE[:HILOS] ...

Cada variante vive en su propio proceso, porque dos enlaces con el mismo nombre de módulo no
pueden cargarse en uno. Todas preparan los mismos bloques con la misma semilla: los seis del
codec v3.1 (127 salidas y anchuras de precios, noticias, gráficos, fundamentales, macro y
presencia) con 1, 8, 31, 64, 278, 512 y 1.000 filas. HILOS fija `torch.set_num_threads` en
ese proceso y, si falta, se usan los de `OMP_NUM_THREADS`.

El proceso principal pide una muestra a cada variante por turnos, en un orden que rota en cada
ronda, así que los cambios de carga del equipo afectan a todas por igual. Una muestra es el
tiempo medio de proyectar los seis bloques en el orden de `encode`, repetido las veces
necesarias para que dure unos 5 ms con la primera variante. Para cada número de filas se
guardan las medianas, la razón de cada variante frente a la primera en cada ronda con un
intervalo bootstrap del 95 % de su mediana y la carga media. Antes de medir se exige que todas
las variantes den los mismos bytes. No hay modelo, GPU ni pasos de optimizador.
"""

import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime

import numpy as np

WIDTHS = {"prices": 384, "news": 384, "charts": 512, "fundamentals": 78, "macro": 420}
PRESENCE = 5
OUTPUTS = 127
ROWS = (1, 8, 31, 64, 278, 512, 1000)
ROUNDS = 42
WARMUP = 3
TARGET_SECONDS = 0.005
BOOTSTRAP = 4000


def worker(path, threads):
    """Prepara los bloques, anuncia la huella de sus salidas y mide cuando se le pide."""
    import torch

    from mars_titan.memory.native_backend import load_native

    if threads:
        torch.set_num_threads(int(threads))
    native = load_native(path)
    widths = [*WIDTHS.values(), PRESENCE]
    random = np.random.default_rng(628)
    transposed = random.standard_normal((sum(widths), OUTPUTS))
    offsets = np.cumsum([0, *widths])
    weights = [
        np.ascontiguousarray(transposed[a:b]) for a, b in zip(offsets, offsets[1:], strict=False)
    ]
    blocks = {rows: [random.standard_normal((rows, w)) for w in widths] for rows in ROWS}
    digest = hashlib.sha256()
    for rows in ROWS:
        for block, weight in zip(blocks[rows], weights, strict=True):
            digest.update(native.project_codec_block(block, weight).tobytes())
    counts = getattr(native, "codec_projection_threads", None)
    hello = dict(
        isa=native.codec_projection_isa,
        aten_threads=torch.get_num_threads(),
        outputs_sha256=digest.hexdigest(),
        projection_threads=None
        if counts is None
        else {rows: [counts(rows, w, OUTPUTS) for w in widths] for rows in ROWS},
    )
    print(json.dumps(hello), flush=True)
    for line in sys.stdin:
        rows, calls = (int(x) for x in line.split())
        pairs = list(zip(blocks[rows], weights, strict=True))
        began = time.perf_counter()
        for _ in range(calls):
            for block, weight in pairs:
                native.project_codec_block(block, weight)
        print((time.perf_counter() - began) / calls, flush=True)


class Variant:
    def __init__(self, spec):
        self.name, rest = spec.split("=", 1)
        self.path, _, self.threads = rest.partition(":")
        self.process = subprocess.Popen(
            [sys.executable, __file__, "--worker", self.path, self.threads],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.hello = json.loads(self.process.stdout.readline())

    def sample(self, rows, calls):
        self.process.stdin.write(f"{rows} {calls}\n")
        return float(self.process.stdout.readline())

    def close(self):
        self.process.stdin.close()
        self.process.wait()


def bootstrap_median(values, generator):
    values = np.asarray(values)
    picks = generator.integers(0, len(values), size=(BOOTSTRAP, len(values)))
    medians = np.median(values[picks], axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def main(output, specs):
    variants = [Variant(spec) for spec in specs]
    hashes = {v.name: v.hello["outputs_sha256"] for v in variants}
    if len(set(hashes.values())) != 1:
        raise SystemExit(f"Las variantes no dan los mismos bytes: {hashes}")
    generator = np.random.default_rng(628)
    count = len(variants)
    # Rotaciones en los dos sentidos, para que cada variante ocupe cada posición por igual.
    orders = [
        [(start + step * k) % count for k in range(count)]
        for step in (1, -1)
        for start in range(count)
    ]
    result = dict(
        started=datetime.now(UTC).isoformat(timespec="seconds"),
        cpus=os.cpu_count(),
        rounds=ROUNDS,
        target_seconds=TARGET_SECONDS,
        reference=variants[0].name,
        variants={v.name: dict(threads=v.threads or None, **v.hello) for v in variants},
        rows={},
    )
    for rows in ROWS:
        load_before = os.getloadavg()
        first = min(variants[0].sample(rows, 1) for _ in range(3))
        calls = max(1, math.ceil(TARGET_SECONDS / first))
        for variant in variants:
            for _ in range(WARMUP):
                variant.sample(rows, calls)
        times = {v.name: [] for v in variants}
        for round_index in range(ROUNDS):
            for index in orders[round_index % len(orders)]:
                variant = variants[index]
                times[variant.name].append(variant.sample(rows, calls))
        reference = times[variants[0].name]
        item = dict(calls=calls, load_before=load_before, load_after=os.getloadavg(), variants={})
        for variant in variants:
            ratios = [r / t for r, t in zip(reference, times[variant.name], strict=True)]
            item["variants"][variant.name] = dict(
                median_us=1e6 * statistics.median(times[variant.name]),
                min_us=1e6 * min(times[variant.name]),
                max_us=1e6 * max(times[variant.name]),
                speedup_median=statistics.median(ratios),
                speedup_ci95=bootstrap_median(ratios, generator),
            )
        result["rows"][rows] = item
        summary = {name: round(v["speedup_median"], 2) for name, v in item["variants"].items()}
        print(rows, summary, flush=True)
    for variant in variants:
        variant.close()
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1)
        stream.write("\n")


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2:])
