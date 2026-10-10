"""Materializar una vez los eventos de ajuste de la ventana medida, sin GPU.

Guarda eventos completos, un bloque de hasta 4.096 flujos por evento, desde la mitad del
tramo de ajuste. `chronological_replay.py --events` los repite para medir el cálculo de
cada familia sin el lector cronológico, que domina el tiempo en US+CN. La caché de bloques
se amplía para que el lector no colapse con el acceso cíclico de miles de activos.

Uso: chronological_events.py VIEW WORK SALIDA EVENTOS CACHE_GIB
"""

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import chronological_replay_patches  # noqa: F401,E402

from mars_titan.training import campaign_throughput as ct  # noqa: E402
from mars_titan.training.financial_run import load_recipe  # noqa: E402

view, work, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
count, cache = int(sys.argv[4]), int(sys.argv[5]) * 1024**3
_, document = load_recipe("configs/titans/chronological-training-historical-masked.json")
_, sources, inputs = ct._chronological_sources(view, document, work)
start = len(inputs) // 2
began, events, rows = time.perf_counter(), [], 0
iterator = sources["train"].batched_events(
    start_cursor=start, block_rows=4096, max_cached_bytes=cache
)
for event in iterator:
    events.append(event)
    rows += sum(len(raw["market"]) for raw in event.inputs)
    print("event", len(events), rows, round(time.perf_counter() - began, 1), flush=True)
    if len(events) >= count:
        break
seconds = time.perf_counter() - began
with output.open("wb") as handle:
    pickle.dump(dict(start=start, events_total=len(inputs), events=events), handle, protocol=5)
print(
    "events",
    len(events),
    "rows",
    rows,
    "seconds",
    round(seconds, 1),
    "rows/s",
    round(rows / seconds),
)
