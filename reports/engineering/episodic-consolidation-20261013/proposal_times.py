"""Tiempo de las propuestas del banco episódico y de su instantánea, con huellas de los estados.

Uso: proposal_times.py ÁRBOL POLÍTICA SALIDA.json

Con el código del árbol y el enlace de MARS_TITAN_EPISODIC_NATIVE, llena un banco de
capacidad 1.024 con 7 propuestas de 4.040 episodios de claves normales fijas, la forma de las
llamadas de los lectores (1.024 retenidos y 4.040 nuevos). POLÍTICA es `reservoir` (M1) o
`anchored` (frontera 8 y 8 candidatos nuevos, como CM-v1), con el contrato `causal_v2`.
Cronometra cada `propose` y cada `bank_snapshot` en CPU, y guarda la huella SHA-256 de los
recibos, de la instantánea del banco y de los tensores de la instantánea de lectura, para
comparar los árboles bit a bit. No hay modelo, GPU ni pasos de optimizador.
"""

import hashlib
import json
import os
import platform
import resource
import sys
import time

TREE, POLICY, OUT = sys.argv[1:4]
sys.path[:0] = [f"{TREE}/src"]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mars_titan.memory import financial_consumers  # noqa: E402
from mars_titan.memory import retention_bank as module  # noqa: E402
from mars_titan.memory.native_backend import load_native  # noqa: E402

native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
binary = native.binary_sha256
# La identidad del banco incluye el SHA-256 de su código y del enlace. Para comparar los dos
# árboles se fija el mismo valor en ambos, de modo que solo cuenta el cálculo.
module._CODE = {name: "0" * 64 for name in module._CODE}
native.binary_sha256 = "0" * 64

PROPOSALS, INCOMING = 7, 4040
config = module.RetentionConfig(policy=POLICY, capacity=1024, frontier=8, new_candidates=8)
bank = module.RetentionBank(
    native,
    config,
    codec_id="c" * 64,
    world="medida",
    partition="train",
    fold="0",
    memory_contract="causal_v2",
)
rng = np.random.default_rng(547)
digest = hashlib.sha256()
propose_seconds, snapshot_seconds = [], []
for step in range(PROPOSALS):
    incoming = []
    for offset in range(INCOMING):
        value = native.MemoryRecord()
        value.id = 1 + step * INCOMING + offset
        value.decision_at = 2 * value.id
        value.available_at = value.decision_at
        value.maturity_at = value.decision_at + 1
        value.key = rng.standard_normal(64).astype(np.float32).tolist()
        value.value = rng.standard_normal(64).astype(np.float32).tolist()
        value.label, value.label_valid = float(rng.standard_normal()), True
        incoming.append(value)
    began = time.perf_counter()
    bank = bank.propose(incoming, confirmed_at=10 * (step + 1) * INCOMING)
    propose_seconds.append(time.perf_counter() - began)
    began = time.perf_counter()
    snapshot = financial_consumers.bank_snapshot(
        bank,
        codec_id="c" * 64,
        context_id="d" * 64,
        cutoff=10 * (step + 1) * INCOMING + 1,
        dtype=torch.float32,
        device="cpu",
    )
    snapshot_seconds.append(time.perf_counter() - began)
    digest.update(json.dumps(bank.receipt, sort_keys=True).encode())
    digest.update(bank.snapshot()["memory"])
    for tensor in snapshot._values:
        digest.update(str(tensor.dtype).encode() + tensor.cpu().numpy().tobytes())
result = dict(
    tree=TREE,
    policy=POLICY,
    native=os.environ["MARS_TITAN_EPISODIC_NATIVE"],
    native_sha256=binary,
    proposals=PROPOSALS,
    incoming=INCOMING,
    propose_seconds=propose_seconds,
    snapshot_seconds=snapshot_seconds,
    # La primera propuesta llena el banco. Las demás tienen 1.024 retenidos y 4.040 nuevos.
    full_bank_propose_seconds=sum(propose_seconds[1:]),
    full_bank_snapshot_seconds=sum(snapshot_seconds[1:]),
    states_sha256=digest.hexdigest(),
    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    threads={name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    load_average=list(os.getloadavg()),
    versions=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__),
)
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=1)
    handle.write("\n")
print(json.dumps({k: result[k] for k in ("full_bank_propose_seconds", "states_sha256")}))
