"""Mutaciones dirigidas de #619: cambia un fragmento, ejecuta las pruebas y restaura."""

import json
import subprocess
import sys
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.parent.mkdir(parents=True, exist_ok=True)
READER = ROOT / "src/mars_titan/memory/financial_observations.py"
# Pruebas de las ventanas de calendario por bloque, con el filtro que las selecciona.
WINDOWS = (
    [
        "tests/memory/test_financial_observation_blocks.py",
        "tests/training/test_calendar_price_windows.py",
    ],
    "calendar or once_per_pass or gapped",
)
# Pruebas de la agrupación por instante del índice.
INSTANTS = (["tests/memory/test_observation_instants.py"], None)
MUTATIONS = {
    "full_without_span": (
        READER,
        (
            "        if context - 1 <= end < len(positions) and (\n"
            "            positions[end] - positions[end - context + 1] == context - 1\n"
            "        ):"
        ),
        "        if context - 1 <= end < len(positions):",
        WINDOWS,
    ),
    "presence_zero": (
        READER,
        "        result[full, :, len(PRICE_FEATURES)] = 1",
        "        result[full, :, len(PRICE_FEATURES)] = 0",
        WINDOWS,
    ),
    "positions_not_cached": (
        READER,
        "        if identity not in self._positions:",
        "        if True:",
        WINDOWS,
    ),
    "full_rows_shifted": (
        READER,
        "        chosen = np.asarray(ends)[full]",
        "        chosen = np.asarray(ends)[full] - 1",
        WINDOWS,
    ),
    "gap_rows_without_window_rows": (
        READER,
        "        rows = window_rows(positions, np.array([end]), context, absent)\n",
        "        rows = (end - np.arange(context - 1, -1, -1))[None]\n",
        WINDOWS,
    ),
    "budget_off_by_one": (
        READER,
        "            if len(rows) + right - left > _EVENT_ROWS:",
        "            if len(rows) + right - left >= _EVENT_ROWS:",
        INSTANTS,
    ),
    "no_step_back_check": (
        READER,
        (
            "            if previous is not None and at < previous:\n"
            '                raise ValueError("El índice ha perdido su orden temporal")\n'
            "            if previous is not None and at != previous:\n"
            "                groups.append"
        ),
        ("            if previous is not None and at != previous:\n                groups.append"),
        INSTANTS,
    ),
    "cursor_off_by_one": (
        READER,
        (
            "                if len(groups) > start_cursor:\n"
            "                    yield previous, rows\n"
            "                rows = []"
        ),
        (
            "                if len(groups) >= start_cursor:\n"
            "                    yield previous, rows\n"
            "                rows = []"
        ),
        INSTANTS,
    ),
    "one_instant_per_batch": (
        READER,
        "        cuts = (np.flatnonzero(moments[1:] != moments[:-1]) + 1).tolist()",
        "        cuts = []",
        INSTANTS,
    ),
    "empty_batch_unguarded": (
        READER,
        ("        if not items:\n            continue\n"),
        "",
        INSTANTS,
    ),
}
results = {}
for name, (path, old, new, (tests, keyword)) in MUTATIONS.items():
    source = path.read_text()
    assert source.count(old) == 1, name
    path.write_text(source.replace(old, new))
    try:
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *tests,
                "-x",
                "-q",
                "-p",
                "no:cacheprovider",
                *(["-k", keyword] if keyword else []),
                "--basetemp",
                str(OUT.parent / f"base-{name}"),
            ],
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
OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
