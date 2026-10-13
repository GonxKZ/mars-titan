"""Mutaciones dirigidas de #547. Cada una debe hacer fallar alguna prueba.

Uso: mutations.py ÁRBOL SALIDA.json. Cada mutación cambia un único fragmento de
`retention_bank.py`, ejecuta `tests/memory/test_retention_bank.py` en CPU con pytest del mismo
intérprete, `-x` y sin orden aleatorio, y restaura el archivo. Necesita
MARS_TITAN_EPISODIC_NATIVE.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

W, OUT = Path(sys.argv[1]), Path(sys.argv[2])
BANK = "src/mars_titan/memory/retention_bank.py"
TESTS = ["tests/memory/test_retention_bank.py"]
mutants = [
    (
        "fork_without_generator",
        "        candidate._rng.bit_generator.state = self._rng.bit_generator.state\n",
        "",
    ),
    (
        "fork_without_memory",
        "        candidate._memory.restore_bytes(self._memory.snapshot_bytes())\n",
        "",
    ),
    (
        "proposal_without_key_cache",
        "        candidate._keys = {identifier: rows[identifier] for identifier in retained}\n",
        "",
    ),
    (
        "cache_keeps_every_client",
        "        candidate._keys = {identifier: rows[identifier] for identifier in retained}\n",
        "        candidate._keys = dict(rows)\n",
    ),
    ("writable_cached_rows", "                row.flags.writeable = False\n", ""),
]
environment = dict(
    os.environ,
    PYTHONPATH=str(W / "src"),
    CUDA_VISIBLE_DEVICES="-1",
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
)
results = []
path = W / BANK
for name, old, new in mutants:
    original = path.read_text()
    assert original.count(old) == 1, name
    path.write_text(original.replace(old, new))
    try:
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "-x",
                "-p",
                "no:randomly",
                *TESTS,
            ],
            capture_output=True,
            text=True,
            env=environment,
            timeout=1800,
            cwd=W,
        )
    finally:
        path.write_text(original)
    failed = [line for line in run.stdout.splitlines() if line.startswith("FAILED")]
    status = "detectada" if run.returncode else "SOBREVIVE"
    results.append(
        dict(
            name=name,
            file=BANK,
            tests=TESTS,
            status=status,
            failed=failed[:2],
            summary=run.stdout.strip().splitlines()[-1:],
        )
    )
    print(name, status, failed[:1] or run.stdout.strip().splitlines()[-1:], flush=True)
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
