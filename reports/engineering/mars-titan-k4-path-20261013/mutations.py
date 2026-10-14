"""Mutaciones dirigidas de #548. Cada una debe hacer fallar alguna prueba.

Uso: mutations.py ÁRBOL SALIDA.json. Cada mutación cambia un único fragmento de
`episodic_readout.py`, ejecuta sus pruebas en CPU con pytest del mismo intérprete, `-x` y sin
orden aleatorio, y restaura el archivo.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

W, OUT = Path(sys.argv[1]), Path(sys.argv[2])
READOUT = "src/mars_titan/models/titans/episodic_readout.py"
REFINEMENTS = ["tests/models/titans/test_episodic_readout_refinements.py"]
BASIC = "tests/models/titans/test_episodic_readout.py"
mutants = [
    (
        "first_step_without_validation",
        "repeated=step > 0\n",
        "repeated=True\n",
        [*REFINEMENTS, f"{BASIC}::test_reader_rejects_context_and_input_mismatches"],
    ),
    ("every_step_validates", "repeated=step > 0\n", "repeated=False\n", REFINEMENTS),
    (
        "public_read_without_validation",
        "        if not repeated:\n            self._validate_read(",
        "        if False:\n            self._validate_read(",
        [f"{BASIC}::test_public_read_cannot_bypass_the_working_budget", *REFINEMENTS],
    ),
    (
        "forward_without_gate_check",
        "                if step == 0:\n                    # Los parámetros no cambian dentro"
        " de la llamada.",
        "                if False:\n                    # Los parámetros no cambian dentro"
        " de la llamada.",
        [f"{BASIC}::test_nonfinite_gate_is_rejected_even_if_sigmoid_saturates", *REFINEMENTS],
    ),
    (
        "gate_checked_every_step",
        "                if step == 0:\n                    # Los parámetros no cambian dentro"
        " de la llamada.",
        "                if True:\n                    # Los parámetros no cambian dentro"
        " de la llamada.",
        REFINEMENTS,
    ),
    (
        "forward_through_public_refine",
        "                state = self._step(state, base, read)\n",
        "                state = self.refine(state, base, read)\n",
        REFINEMENTS,
    ),
    (
        "public_refine_without_zero_fill",
        '            "Una lectura ausente necesita relleno cero",\n        )\n'
        "        return self._step(state, base, read)\n",
        '            "Una lectura ausente necesita relleno cero",\n        ) if False else None\n'
        "        return self._step(state, base, read)\n",
        [f"{BASIC}::test_refinement_rejects_nonzero_absent_read_values", *REFINEMENTS],
    ),
]
environment = dict(
    os.environ,
    PYTHONPATH=str(W / "src"),
    CUDA_VISIBLE_DEVICES="-1",
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
)
results = []
path = W / READOUT
for name, old, new, tests in mutants:
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
                *tests,
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
            file=READOUT,
            tests=tests,
            status=status,
            failed=failed[:2],
            summary=run.stdout.strip().splitlines()[-1:],
        )
    )
    print(name, status, failed[:1] or run.stdout.strip().splitlines()[-1:], flush=True)
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
