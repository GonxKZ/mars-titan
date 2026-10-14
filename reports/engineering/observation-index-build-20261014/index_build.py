"""Tiempo, memoria y huellas de los cuatro índices de observaciones de una vista.

Uso: index_build.py VISTA ÍNDICES SALIDA.json
Abre la vista y construye en ÍNDICES (carpeta nueva) los índices de las cuatro fases de la
receta de Titans-MAC, uno tras otro, como el primer trabajo de la ventana. Guarda el tiempo
de cada fase, el pico de memoria del proceso y la huella SHA-256 de cada archivo de eventos.
No lee modalidades ni ajusta.
"""

import json
import os
import resource
import sys
import time
from pathlib import Path

from mars_titan.training import titans_walk_forward as titans
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.financial_run import load_recipe

view, root, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
if root.exists():
    raise SystemExit(f"{root} ya existe: la medida necesita construir desde cero")
result = dict(view=str(view), loadavg_before=os.getloadavg())
began = time.perf_counter()
dataset = CorpusDataset(view, input_policy="historical_masked_2000_v1")
result["open_seconds"] = time.perf_counter() - began
result["assets"] = len(dataset.assets)
_, document = load_recipe("configs/titans/chronological-training-historical-masked.json")
_, _, _, fold = titans._protocol(titans.view_protocol(view), view.parent.name, 42)
phases = titans.window_phases(fold, document["walk_forward"]["warmup_months"])
result["phases"] = {}
for name, phase in phases.items():
    start = time.perf_counter()
    source = titans._sources(dataset, {name: phase}, root)[name]
    result["phases"][name] = dict(
        seconds=time.perf_counter() - start,
        rows=source.metadata["rows"],
        bytes=source.metadata["bytes"],
        events_sha256=source.metadata["events_sha256"],
    )
result["build_seconds"] = sum(item["seconds"] for item in result["phases"].values())
result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
result["loadavg_after"] = os.getloadavg()
output.write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps(dict(build_seconds=result["build_seconds"], peak=result["peak_rss_mib"])))
