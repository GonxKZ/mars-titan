"""Mutaciones dirigidas de #622: cambia un fragmento, ejecuta las pruebas y restaura.

Uso: mutations.py ÁRBOL SALIDA.json, sobre una copia del árbol.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.parent.mkdir(parents=True, exist_ok=True)
INDEX = ROOT / "src/mars_titan/memory/financial_observations.py"
TESTS = [
    "tests/memory/test_observation_index_records.py",
    "tests/memory/test_financial_observation_empty_groups.py",
]
MUTATIONS = {
    "no_order_check": (
        "        if np.any(np.diff(stamps) <= 0):\n"
        '            raise ValueError("Las muestras necesitan orden temporal único")\n',
        "",
    ),
    "repeated_dates_allowed": (
        "        if np.any(np.diff(stamps) <= 0):",
        "        if np.any(np.diff(stamps) < 0):",
    ),
    "group_left_side": (
        '    groups = np.searchsorted(ends, rows, side="right")',
        '    groups = np.searchsorted(ends, rows, side="left")',
    ),
    "row_without_group_start": (
        "        source_row=rows - (ends - sizes)[groups],",
        "        source_row=rows,",
    ),
    "empty_file_read_as_rows": (
        "    if sizes.sum():",
        "    if False:",
    ),
    "empty_tables_written": (
        '            if len(columns["event_at"]):',
        "            if True:",
    ),
    "no_limit_after_samples": (
        '            total += len(samples["event_at"])\n'
        "            if total >= _MAX_ROWS:\n"
        '                raise ValueError("El índice supera su número máximo de registros")\n',
        '            total += len(samples["event_at"])\n',
    ),
    "no_limit_after_labels": (
        "            total += len(selected)\n"
        "            if total >= _MAX_ROWS:\n"
        '                raise ValueError("El índice supera su número máximo de registros")\n',
        "            total += len(selected)\n",
    ),
    "buffer_not_cleared": (
        "            pending, buffered = [], 0",
        "            buffered = 0",
    ),
    "close_row_lost": (
        "    yield pa.concat_tables(pending)\n\n\ndef prepare_observation_index(",
        "    yield pa.concat_tables(pending[:-1])\n\n\ndef prepare_observation_index(",
    ),
}
results = {}
for name, (old, new) in MUTATIONS.items():
    source = INDEX.read_text()
    assert source.count(old) == 1, name
    INDEX.write_text(source.replace(old, new))
    try:
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *TESTS,
                "-x",
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(OUT.parent / f"base-{name}"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    finally:
        INDEX.write_text(source)
    lines = run.stdout.splitlines()
    failed = [line.split(" ")[1] for line in lines if line.startswith("FAILED")]
    results[name] = dict(detected=run.returncode != 0, failed=failed, tail=lines[-1:])
    print(name, results[name], flush=True)
OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
