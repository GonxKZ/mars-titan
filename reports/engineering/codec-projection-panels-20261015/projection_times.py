"""Tiempo de `project_codec_block` con el enlace de un árbol, como lo llama `encode`.

Uso: projection_times.py ENLACE SALIDA.json

Con el enlace de ENLACE, proyecta los seis bloques del codec v3.1 (127 salidas y anchuras de
precios, noticias, gráficos, fundamentales, macro y presencia) con 1, 8, 31, 64, 278, 512 y
1.000 filas, en el orden de `encode`. Cada medida es la mediana de 31 llamadas de los seis
bloques, con los hilos de ATen del proceso (`OMP_NUM_THREADS`). Guarda las medianas, la carga
media, los hilos que usaría el enlace si lo dice y una huella SHA-256 de todas las salidas para
comparar los enlaces bit a bit. No hay modelo, GPU ni pasos de optimizador.
"""

import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

from mars_titan.memory.native_backend import load_native

WIDTHS = {"prices": 384, "news": 384, "charts": 512, "fundamentals": 78, "macro": 420}
PRESENCE = 5
OUTPUTS = 127
ROWS = (1, 8, 31, 64, 278, 512, 1000)
REPETITIONS = 31

native = load_native(sys.argv[1])
widths = [*WIDTHS.values(), PRESENCE]
random = np.random.default_rng(628)
transposed = random.standard_normal((sum(widths), OUTPUTS))
offsets = np.cumsum([0, *widths])
weights = [
    np.ascontiguousarray(transposed[a:b]) for a, b in zip(offsets, offsets[1:], strict=False)
]
digest = hashlib.sha256()
result = dict(
    isa=native.codec_projection_isa,
    aten_threads=torch.get_num_threads(),
    loadavg_before=os.getloadavg(),
    rows={},
)
for rows in ROWS:
    blocks = [random.standard_normal((rows, width)) for width in widths]
    for block, weight in zip(blocks, weights, strict=True):
        digest.update(native.project_codec_block(block, weight).tobytes())
    seconds = []
    for _ in range(REPETITIONS):
        began = time.perf_counter()
        for block, weight in zip(blocks, weights, strict=True):
            native.project_codec_block(block, weight)
        seconds.append(time.perf_counter() - began)
    threads = getattr(native, "codec_projection_threads", None)
    result["rows"][rows] = dict(
        median_us=1e6 * statistics.median(seconds),
        threads=None if threads is None else [threads(rows, w, OUTPUTS) for w in widths],
    )
result["loadavg_after"] = os.getloadavg()
result["outputs_sha256"] = digest.hexdigest()
with open(sys.argv[2], "w", encoding="utf-8") as output:
    json.dump(result, output, indent=1)
print(json.dumps({rows: round(item["median_us"]) for rows, item in result["rows"].items()}))
