"""Pico de asignaciones de Python y NumPy de cada propuesta del banco episódico, sin tiempos.

Uso: consolidation_memory.py ÁRBOL SALIDA.json

Repite el flujo de `consolidation_counts.py` (capacidad 1.024, 12 propuestas de 128
episodios, `reservoir` y `anchored`, contratos `legacy_v1` y `causal_v2`) y mide con
tracemalloc el pico de cada una de las cuatro últimas propuestas, con el banco lleno. Lo
compara con `estimated_peak_bytes` del recibo. tracemalloc no ve las asignaciones internas
del enlace C++, así que el pico medido es una cota inferior del total.
"""

import json
import os
import sys
import tracemalloc

TREE, OUT = sys.argv[1:3]
sys.path[:0] = [f"{TREE}/src"]

import numpy as np  # noqa: E402

from mars_titan.memory import retention_bank as module  # noqa: E402
from mars_titan.memory.native_backend import load_native  # noqa: E402

native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
module._CODE = {name: "0" * 64 for name in module._CODE}


def record(identity, rng):
    value = native.MemoryRecord()
    value.id = identity
    value.decision_at = 2 * identity
    value.available_at = value.decision_at
    value.maturity_at = value.decision_at + 1
    value.key = [float(x) for x in rng.standard_normal(64)]
    value.value = [float(identity)] * 64
    value.label, value.label_valid = identity * 0.25, True
    return value


results = {}
for contract in ("legacy_v1", "causal_v2"):
    for policy in ("reservoir", "anchored"):
        rng = np.random.default_rng(547)
        config = module.RetentionConfig(policy=policy, capacity=1024, frontier=8, new_candidates=8)
        bank = module.RetentionBank(
            native,
            config,
            codec_id="c" * 64,
            world="fixture",
            partition="train",
            fold="0",
            memory_contract=contract,
        )
        peaks = []
        for step in range(12):
            incoming = [record(1 + 128 * step + offset, rng) for offset in range(128)]
            if step >= 8:
                tracemalloc.start()
                tracemalloc.reset_peak()
                bank = bank.propose(incoming, confirmed_at=1_000 * (step + 1))
                peaks.append(
                    dict(
                        traced_peak_bytes=tracemalloc.get_traced_memory()[1],
                        estimated_peak_bytes=bank.receipt["estimated_peak_bytes"],
                    )
                )
                tracemalloc.stop()
            else:
                bank = bank.propose(incoming, confirmed_at=1_000 * (step + 1))
        results[f"{contract}/{policy}"] = peaks
        print(contract, policy, peaks, flush=True)
with open(OUT, "w") as handle:
    json.dump(dict(tree=TREE, results=results), handle, indent=1)
