"""Coste de construir los índices de las cuatro fases de una vista, sin leer ni ajustar.

Uso: index_costs.py VISTA ÍNDICES SALIDA.json
Abre la vista y construye en ÍNDICES (carpeta nueva) los índices de las cuatro fases de la
receta de Titans-MAC, uno tras otro, como el primer trabajo de la ventana.
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
result = dict(view=str(view), root=str(root), loadavg_before=os.getloadavg())
began = time.perf_counter()
dataset = CorpusDataset(view, input_policy="historical_masked_2000_v1")
result["open_seconds"] = time.perf_counter() - began
result["assets"] = len(dataset.assets)
_, document = load_recipe("configs/titans/chronological-training-historical-masked.json")
_, _, _, fold = titans._protocol(titans.view_protocol(view), view.parent.name, 42)
phases = titans.window_phases(fold, document["walk_forward"]["warmup_months"])
result["index_seconds"], result["index_rows"], result["index_bytes"] = {}, {}, {}
for name, phase in phases.items():
    start = time.perf_counter()
    source = titans._sources(dataset, {name: phase}, root)[name]
    result["index_seconds"][name] = time.perf_counter() - start
    result["index_rows"][name] = source.metadata["rows"]
    result["index_bytes"][name] = source.metadata["bytes"]
result["build_seconds"] = sum(result["index_seconds"].values())
result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
result["loadavg_after"] = os.getloadavg()
output.write_text(json.dumps(result, indent=1))
print(json.dumps(result))
