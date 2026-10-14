"""Trabajo contado y estados de las propuestas del banco episódico, sin cronometrar.

Uso: consolidation_counts.py ÁRBOL SALIDA.pkl

Con el código del árbol y el enlace de MARS_TITAN_EPISODIC_NATIVE, llena un banco de
capacidad 1.024 con 12 propuestas de 128 episodios de claves normales fijas, con las
políticas `reservoir` y `anchored` (frontera 8 y 8 candidatos nuevos, como CM-v1) y los
contratos `legacy_v1` y `causal_v2`. En las
cuatro últimas propuestas, con el banco lleno, cuenta llamadas por función (cProfile, solo
ncalls, también las del enlace), caracteres de `json.dumps`, llamadas a `json.loads`, bytes
resumidos con SHA-256 y lecturas de claves de los registros que ya estaban retenidos. Guarda
recibos, instantáneas y registros de cada propuesta para compararlos bit a bit.
"""

import cProfile
import hashlib
import json
import os
import pickle
import pstats
import sys
from collections import Counter

TREE, OUT = sys.argv[1:3]
sys.path[:0] = [f"{TREE}/src"]

import numpy as np  # noqa: E402

from mars_titan.memory import retention_bank as module  # noqa: E402
from mars_titan.memory.native_backend import load_native  # noqa: E402

native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
# La identidad del banco incluye el SHA-256 de su código, y con el contrato legacy_v1 esa
# identidad también siembra el reservorio nativo. Para comparar los dos árboles se fija el
# mismo valor en ambos, de modo que solo cuenta el cálculo.
module._CODE = {name: "0" * 64 for name in module._CODE}
real_sha256, real_dumps, real_loads = hashlib.sha256, json.dumps, json.loads
counts = Counter()


class Counted:
    def __init__(self, data=b"", **kwargs):
        self._hash = real_sha256(data, **kwargs)
        counts["sha256_calls"] += 1
        counts["sha256_bytes"] += memoryview(data).nbytes

    def update(self, data):
        counts["sha256_bytes"] += memoryview(data).nbytes
        self._hash.update(data)

    def __getattr__(self, name):
        return getattr(self._hash, name)


def dumps(*args, **kwargs):
    text = real_dumps(*args, **kwargs)
    counts["json_dumps_calls"] += 1
    counts["json_dumps_chars"] += len(text)
    return text


def loads(*args, **kwargs):
    counts["json_loads_calls"] += 1
    return real_loads(*args, **kwargs)


class CountedKey:
    def __init__(self, record):
        self._record = record

    def __getattr__(self, name):
        if name == "key":
            counts["retained_key_reads"] += 1
        return getattr(self._record, name)


records = module.RetentionBank.records


def counted_records(self):
    return [CountedKey(r) for r in records(self)]


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


results, states = {}, {}
for contract, policy in [
    (contract, policy)
    for contract in ("legacy_v1", "causal_v2")
    for policy in ("reservoir", "anchored")
]:
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
    counts.clear()
    profiler = cProfile.Profile()
    history = []
    for step in range(12):
        incoming = [record(1 + 128 * step + offset, rng) for offset in range(128)]
        measured = step >= 8
        if measured:
            hashlib.sha256, json.dumps, json.loads = Counted, dumps, loads
            module.RetentionBank.records = counted_records
            profiler.enable()
        try:
            bank = bank.propose(incoming, confirmed_at=1_000 * (step + 1))
        finally:
            if measured:
                profiler.disable()
                hashlib.sha256, json.dumps, json.loads = real_sha256, real_dumps, real_loads
                module.RetentionBank.records = records
        retained = bank.records()
        history.append(
            dict(
                receipt=bank.receipt,
                snapshot=bank.snapshot(),
                records=[
                    (r.id, list(r.key), list(r.value), r.label, r.decision_at, r.maturity_at)
                    for r in retained
                ],
            )
        )
    calls = {}
    for (filename, line, name), (_, ncalls, *_rest) in pstats.Stats(profiler).stats.items():
        short = filename.split("src/mars_titan/")[-1] if "mars_titan" in filename else filename
        calls[f"{short}:{line}:{name}"] = ncalls
    results[f"{contract}/{policy}"] = dict(proposals=4, **counts, function_calls=calls)
    states[f"{contract}/{policy}"] = history
    print(contract, policy, dict(counts), flush=True)
with open(OUT, "wb") as handle:
    pickle.dump(dict(tree=TREE, results=results, states=states), handle)
