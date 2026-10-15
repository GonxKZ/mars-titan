"""Huella bit a bit y caudal de los lotes de `batched_events` en una vista real.

Uso: reader_parity.py VISTA ÍNDICES SALIDA.json FASE CURSOR EVENTOS [BLOQUE]
Lee EVENTOS instantes de FASE desde CURSOR (un número o `mitad`) con la tubería que declaran
las variables de entorno y guarda una huella SHA-256 por instante de todo lo que entrega:
cada array con su tipo y su forma, identificadores, mercados, etiquetas y cierre. No ajusta.
"""

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

from mars_titan.training import titans_walk_forward as titans
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.financial_run import load_recipe

view, root, output, phase = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
cursor, count = sys.argv[5], int(sys.argv[6])
block = int(sys.argv[7]) if len(sys.argv) > 7 else 1024


def feed(digest, value):
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(f"{array.dtype.str}{array.shape}".encode())
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        for key in sorted(value):
            digest.update(str(key).encode())
            feed(digest, value[key])
    elif isinstance(value, list | tuple):
        digest.update(f"[{len(value)}".encode())
        for item in value:
            feed(digest, item)
    else:
        digest.update(repr((type(value).__name__, value)).encode())


began = time.perf_counter()
dataset = CorpusDataset(view, input_policy="historical_masked_2000_v1")
_, document = load_recipe("configs/titans/chronological-training-historical-masked.json")
_, _, _, fold = titans._protocol(titans.view_protocol(view), view.parent.name, 42)
phases = titans.window_phases(fold, document["walk_forward"]["warmup_months"])
source = titans._sources(dataset, {phase: phases[phase]}, root)[phase]
total = len(source.metadata["groups"])
if cursor == "mitad":
    cursor = total // 2
elif cursor.startswith("fecha:"):
    # Primer instante desde esa fecha (UTC), para cubrir un tramo concreto del calendario.
    moment = int(np.datetime64(cursor.removeprefix("fecha:"), "us").astype(np.int64))
    cursor = next(i for i, (at, _) in enumerate(source.metadata["groups"]) if at >= moment)
else:
    cursor = int(cursor)
opened = time.perf_counter() - began
digests, rows, marks, gapped = [], 0, [], 0
start = time.perf_counter()
# Con READER_TIMING_ONLY=1 solo se cuenta: el consumidor no compite con el lector por el GIL.
timing_only = os.environ.get("READER_TIMING_ONLY") == "1"
for event in source.batched_events(start_cursor=cursor, block_rows=block):
    rows += sum(len(batch["market"]) for batch in event.inputs)
    if timing_only:
        digests.append(None)
    else:
        digest = hashlib.sha256()
        feed(digest, [event.at, event.close_phase, list(event.inputs), list(event.labels)])
        digests.append(digest.hexdigest())
        # Filas cuya ventana tiene alguna sesión ausente de todo el mercado (presencia cero).
        gapped += sum(
            int((batch["inputs"]["prices"][..., -1] == 0).any(axis=1).sum())
            for batch in event.inputs
        )
    marks.append((time.perf_counter() - start, rows))
    if len(digests) >= count:
        break
seconds = time.perf_counter() - start
warm = next((i for i, (_, r) in enumerate(marks) if r >= rows // 5), 0)
steady = (rows - marks[warm][1]) / (seconds - marks[warm][0]) if warm < len(marks) - 1 else None
result = dict(
    view=str(view),
    phase=phase,
    cursor=cursor,
    events_total=total,
    events=len(digests),
    rows=rows,
    gapped_rows=gapped,
    block_rows=block,
    open_and_index_seconds=opened,
    read_seconds=seconds,
    rows_per_s=rows / seconds,
    steady_rows_per_s=steady,
    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    pipeline=dict(
        decode_workers=dataset.pipeline.decode_workers,
        prefetch_batches=dataset.pipeline.prefetch_batches,
        group_cache_mib=dataset.pipeline.group_cache_mib,
    ),
    combined=None if timing_only else hashlib.sha256("".join(digests).encode()).hexdigest(),
    digests=None if timing_only else digests,
    timing_only=timing_only,
    loadavg=os.getloadavg(),
)
output.write_text(json.dumps(result, indent=1))
print(json.dumps({k: v for k, v in result.items() if k != "digests"}))
