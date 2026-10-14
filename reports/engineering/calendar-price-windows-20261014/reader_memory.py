"""Desglose de memoria del lector por bloques tras leer EVENTOS instantes, con huellas.

Uso: reader_memory.py VISTA ÍNDICES SALIDA.json FASE CURSOR EVENTOS
Recorre los instantes como reader_parity.py (con huella de cada uno, que frena al consumidor)
y al final mide las cachés del lector, el estado de malloc (mallinfo2) y la memoria residente.
"""

import ctypes
import hashlib
import json
import resource
import sys
from pathlib import Path

import numpy as np

from mars_titan.training import titans_walk_forward as titans
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.financial_run import load_recipe

view, root, output, phase = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
cursor, count = int(sys.argv[5]), int(sys.argv[6])


class Mallinfo2(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_size_t)
        for name in (
            "arena",
            "ordblks",
            "smblks",
            "hblks",
            "hblkhd",
            "usmblks",
            "fsmblks",
            "uordblks",
            "fordblks",
            "keepcost",
        )
    ]


libc = ctypes.CDLL("libc.so.6")
libc.mallinfo2.restype = Mallinfo2


def malloc_state():
    info = libc.mallinfo2()
    return {name: getattr(info, name) / 2**20 for name, _ in info._fields_}


def rss():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024


def nbytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(nbytes(v) for v in value.values())
    if isinstance(value, list | tuple):
        return sum(nbytes(v) for v in value)
    return 0


dataset = CorpusDataset(view, input_policy="historical_masked_2000_v1")
_, document = load_recipe("configs/titans/chronological-training-historical-masked.json")
_, _, _, fold = titans._protocol(titans.view_protocol(view), view.parent.name, 42)
phases = titans.window_phases(fold, document["walk_forward"]["warmup_months"])
source = titans._sources(dataset, {phase: phases[phase]}, root)[phase]
seen = 0
for event in source.batched_events(start_cursor=cursor, block_rows=1024):
    digest = hashlib.sha256()
    for batch in event.inputs:
        for value in batch["inputs"].values():
            digest.update(np.ascontiguousarray(value).tobytes())
    seen += 1
    if seen >= count:
        break
reader = source.last_reader
result = dict(
    events=seen,
    rss_mib=rss(),
    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    group_cache_mib=reader.cached_bytes / 2**20,
    peak_group_cache_mib=reader.peak_cached_bytes / 2**20,
    groups_cached=len(reader.groups),
    prices_mib=nbytes(list(reader._prices.values())) / 2**20,
    prices_assets=len(reader._prices),
    labels_mib=nbytes(list(reader._labels.values())) / 2**20,
    positions_mib=sum(p[0].nbytes for p in getattr(reader, "_positions", {}).values()) / 2**20,
    dataset_cache_mib=dataset.cached_bytes / 2**20,
    malloc=malloc_state(),
)
output.write_text(json.dumps(result, indent=1))
print(json.dumps(result))
