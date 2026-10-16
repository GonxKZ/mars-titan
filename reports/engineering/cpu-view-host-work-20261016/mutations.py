"""Mutaciones dirigidas de #638. Cada una debe hacer fallar alguna prueba.

Uso: mutations.py ÁRBOL SALIDA.json. Cada mutación cambia un único fragmento de un archivo,
ejecuta sus pruebas en CPU con pytest del mismo intérprete, `-x` y sin orden aleatorio, y
restaura el archivo. Las del lector necesitan MARS_TITAN_EPISODIC_NATIVE.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

W, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PIPELINE = "src/mars_titan/training/input_pipeline.py"
TITANS = "src/mars_titan/training/financial_run.py"
READER = "src/mars_titan/training/mars_titan_run.py"
PIPELINE_TESTS = ["tests/training/test_input_pipeline.py"]
TITANS_TESTS = ["tests/training/test_financial_run.py", "-k", "prefetched or validated_ahead"]
READER_TESTS = ["tests/training/test_mars_titan_run.py", "-k", "prefetched or validated_ahead"]
OBSERVE = """            if blocks is None:
                validated = validated_cpu_batches(event.inputs, specification)
            else:
                validated = blocks.result()
"""
mutants = [
    (
        "preparation_error_raised_on_read",
        PIPELINE,
        """            try:
                outcome.set_result(prepare(item))
            except Exception as error:  # Se lanza cuando el consumidor pide el resultado.
                outcome.set_exception(error)
""",
        "            outcome.set_result(prepare(item))\n",
        PIPELINE_TESTS,
    ),
    (
        "preparation_error_raised_on_read_in_the_trainer",
        PIPELINE,
        """            try:
                outcome.set_result(prepare(item))
            except Exception as error:  # Se lanza cuando el consumidor pide el resultado.
                outcome.set_exception(error)
""",
        "            outcome.set_result(prepare(item))\n",
        ["tests/training/test_financial_run.py", "-k", "prefetched_validation_error"],
    ),
    (
        "prefetch_without_producer_thread",
        PIPELINE,
        "    return background(_with_outcomes(items, prepare), depth)\n",
        "    return _with_outcomes(items, prepare)\n",
        PIPELINE_TESTS,
    ),
    (
        "prefetch_without_producer_thread_in_the_trainer",
        PIPELINE,
        "    return background(_with_outcomes(items, prepare), depth)\n",
        "    return _with_outcomes(items, prepare)\n",
        TITANS_TESTS,
    ),
    (
        "sequential_preparation_ahead_of_the_consumer",
        PIPELINE,
        "        return ((item, _Later(prepare, item)) for item in items)\n",
        "        return _with_outcomes(items, prepare)\n",
        PIPELINE_TESTS,
    ),
    (
        "titans_observe_validates_again",
        TITANS,
        OBSERVE,
        "            validated = validated_cpu_batches(event.inputs, specification)\n",
        TITANS_TESTS,
    ),
    (
        "titans_pass_drops_the_blocks",
        TITANS,
        "self._observe(run, source, event, differentiable=True, blocks=blocks)",
        "self._observe(run, source, event, differentiable=True)",
        TITANS_TESTS,
    ),
    (
        "reader_observe_validates_again",
        READER,
        OBSERVE,
        "            validated = validated_cpu_batches(event.inputs, specification)\n",
        READER_TESTS,
    ),
    (
        "reader_pass_drops_the_blocks",
        READER,
        "self._observe(run, phase, event, train=True, blocks=blocks)",
        "self._observe(run, phase, event, train=True)",
        READER_TESTS,
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
for name, file, old, new, tests in mutants:
    path = W / file
    original = path.read_text()
    assert original.count(old) == 1, name
    path.write_text(original.replace(old, new))
    try:
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x"]
        run = subprocess.run(
            [*command, "-p", "no:randomly", *tests],
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
            file=file,
            tests=tests,
            status=status,
            failed=failed[:2],
            summary=run.stdout.strip().splitlines()[-1:],
        )
    )
    print(name, status, failed[:1] or run.stdout.strip().splitlines()[-1:], flush=True)
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
