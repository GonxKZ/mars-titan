"""Tiempo y memoria de validar las etiquetas de todos los activos de una vista, tramo a tramo.

Uso: label_passes.py VISTA SALIDA.json
Abre la vista y pide `_labels` de cada activo en los cuatro tramos, en el orden en que los
recorre la construcción de los índices. Guarda el tiempo de cada pasada (solo las llamadas a
`_labels`), los bytes de etiquetas que quedan en la caché, el pico de memoria del proceso y una
huella SHA-256 de los tipos y bytes de lo devuelto. No lee modalidades ni ajusta.
"""

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

from mars_titan.training.corpus_inputs import CorpusDataset

view, output = Path(sys.argv[1]), Path(sys.argv[2])
result = dict(view=str(view), loadavg_before=os.getloadavg())
dataset = CorpusDataset(view, input_policy="historical_masked_2000_v1")
rows = []
for asset in dataset.assets:
    with pq.ParquetFile(dataset._file(asset, "samples")) as file:
        rows.append(file.metadata.num_rows)
result |= dict(assets=len(dataset.assets), cache_limit=dataset.cache_limit, passes={})
for partition in dataset.partitions:
    digest, seconds, labels = hashlib.sha256(), 0.0, 0
    for asset, count in zip(dataset.assets, rows, strict=True):
        start = time.perf_counter()
        arrays = dataset._labels(asset, partition, count)
        seconds += time.perf_counter() - start
        labels += len(arrays[0])
        for array in arrays:
            digest.update(array.dtype.str.encode())
            digest.update(array.tobytes())
    cached = [size for key, (_, _, size) in dataset._cache.items() if key[0] == "labels"]
    result["passes"][partition] = dict(
        seconds=seconds,
        labels=labels,
        sha256=digest.hexdigest(),
        cached_label_files=len(cached),
        cached_label_mib=sum(cached) / 2**20,
    )
result["label_seconds"] = sum(item["seconds"] for item in result["passes"].values())
result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
result["loadavg_after"] = os.getloadavg()
output.write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps(dict(label_seconds=result["label_seconds"], peak=result["peak_rss_mib"])))
