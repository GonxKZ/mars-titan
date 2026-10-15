"""Mutaciones dirigidas de #625: cambia un fragmento, ejecuta las pruebas y restaura.

Uso: mutations.py ÁRBOL SALIDA.json, sobre una copia del árbol.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.parent.mkdir(parents=True, exist_ok=True)
INPUTS = "src/mars_titan/training/corpus_inputs.py"
COHORTS = "src/mars_titan/training/cohort_contract.py"
TEMPORAL = "src/mars_titan/training/temporal_corpus.py"
TESTS = ["tests/training/test_label_validation.py"]
MUTATIONS = {
    "repeated_ties_allowed": (
        INPUTS,
        "bool(np.all(values[1:] > values[:-1]))",
        "bool(np.all(values[1:] >= values[:-1]))",
    ),
    "repeated_never": (
        INPUTS,
        "    return len(np.unique(values)) != len(values)\n",
        "    return False\n",
    ),
    "single_chunk_dictionary": (
        INPUTS,
        "    column = column.combine_chunks()\n",
        "    column = column.chunk(0)\n",
    ),
    "reasons_without_nulls": (
        INPUTS,
        'if not set(table["reason"].unique().to_pylist()) <= allowed:',
        'if not set(table["reason"].unique().to_pylist()) - {None} <= allowed:',
    ),
    "empty_reason_compared": (
        INPUTS,
        "            if len(table):\n                accepted = pc.equal(",
        "            if True:\n                accepted = pc.equal(",
    ),
    "excluded_partition_allowed": (
        INPUTS,
        '        if np.any(table["partition"].is_valid().to_numpy() & ~accepted):',
        "        if False:",
    ),
    "excluded_check_inverted": (
        INPUTS,
        '        if np.any(table["partition"].is_valid().to_numpy() & ~accepted):',
        '        if np.any(table["partition"].is_valid().to_numpy() & accepted):',
    ),
    "missing_partition_encoded": (
        INPUTS,
        '        if table["partition"].null_count:\n            # Una etiqueta',
        "        if False:\n            # Una etiqueta",
    ),
    "partition_code_inverted": (
        INPUTS,
        "selected = np.flatnonzero(codes == found[0]) if len(found)",
        "selected = np.flatnonzero(codes != found[0]) if len(found)",
    ),
    "unknown_partition_selects_all": (
        INPUTS,
        "if len(found) else np.empty(0, np.int64)",
        "if len(found) else np.arange(len(codes))",
    ),
    "codes_not_narrowed": (
        INPUTS,
        "        codes = codes.astype(np.min_scalar_type(len(names)))\n",
        "        codes = codes.astype(np.int64)\n",
    ),
    "dictionary_not_decoded": (
        INPUTS,
        "        column = column.dictionary_decode()\n",
        "        pass\n",
    ),
    "missing_partition_counted_once": (
        INPUTS,
        "    return [(p, found.get(p, 0)) for p in partitions]",
        "    return [(p, found.get(p, 1)) for p in partitions]",
    ),
    "counts_not_checked": (
        INPUTS,
        'count != asset["counts"][p] for p, count in _counts(',
        "False for p, count in _counts(",
    ),
    "temporal_reasons_ignored": (
        INPUTS,
        '            valid = (expected == partitions) & (reasons == "accepted")',
        "            valid = expected == partitions",
    ),
    "reserved_reason_lost": (
        TEMPORAL,
        '        reason[reserved] = "final_test_reserved"\n',
        "",
    ),
    "assigned_reason_lost": (
        TEMPORAL,
        ', reason[selected] = assigned["partition"], assigned["reason"]',
        ', _ = assigned["partition"], assigned["reason"]',
    ),
    "cohort_superset": (
        COHORTS,
        'or not set(table["cohort_id"].unique().to_pylist()) <= {cohort}',
        'or not set(table["cohort_id"].unique().to_pylist()) >= {cohort}',
    ),
    "cohort_without_nulls": (
        COHORTS,
        'or not set(table["cohort_id"].unique().to_pylist()) <= {cohort}',
        'or not set(table["cohort_id"].unique().to_pylist()) - {None} <= {cohort}',
    ),
}
results = {}
for name, (relative, old, new) in MUTATIONS.items():
    path = ROOT / relative
    source = path.read_text()
    assert source.count(old) == 1, name
    path.write_text(source.replace(old, new))
    try:
        run = subprocess.run(
            [sys.executable, "-m", "pytest", *TESTS, "-x", "-q", "-p", "no:cacheprovider"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    finally:
        path.write_text(source)
    lines = run.stdout.splitlines()
    failed = [line.split(" ")[1] for line in lines if line.startswith("FAILED")]
    results[name] = dict(detected=run.returncode != 0, failed=failed, tail=lines[-1:])
    print(name, results[name], flush=True)
OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
