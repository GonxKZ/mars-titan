"""Mutaciones dirigidas de #618: cambia un fragmento, ejecuta las pruebas y restaura."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.parent.mkdir(parents=True, exist_ok=True)
INDICES = ROOT / "src/mars_titan/training/window_indices.py"
READOUTS = ROOT / "src/mars_titan/training/mars_titan_walk_forward.py"
ROLLING = ROOT / "src/mars_titan/training/rolling_retention.py"
INDICES_TESTS = ["tests/training/test_window_indices.py"]
RELEASE_TESTS = [
    "tests/posttraining/test_retention_v3.py::"
    "test_v3_releases_the_adapter_rows_by_blocks_and_publishes_the_same_comparison"
]
MUTATIONS = {
    "key_without_code": (
        INDICES,
        "key = canonical(index_identity(dataset, phase))",
        "key = canonical(dict(view=dataset.identity, phase=str(phase)))",
        INDICES_TESTS,
    ),
    "build_without_lock": (
        INDICES,
        "        fcntl.flock(lock, fcntl.LOCK_EX)\n",
        "",
        INDICES_TESTS,
    ),
    "rename_before_cleaning": (
        INDICES,
        (
            "    freed = sum(\n"
            "        _remove(retired)\n"
            '        for retired in sorted(root.glob(f"{_RELEASED}*"))\n'
            "        if retired.is_dir() and not retired.is_symlink()\n"
            "    )\n"
            "    folder = _view_folder(root, identity)\n"
            "    if folder.is_dir() and not folder.is_symlink():\n"
            '        retired = root / f"{_RELEASED}{folder.name}"\n'
            "        os.replace(folder, retired)\n"
            "        freed += _remove(retired)\n"
            "    return freed"
        ),
        (
            "    folder = _view_folder(root, identity)\n"
            "    if folder.is_dir() and not folder.is_symlink():\n"
            '        os.replace(folder, root / f"{_RELEASED}{folder.name}")\n'
            "    return sum(\n"
            "        _remove(retired)\n"
            '        for retired in sorted(root.glob(f"{_RELEASED}*"))\n'
            "        if retired.is_dir() and not retired.is_symlink()\n"
            "    )"
        ),
        INDICES_TESTS,
    ),
    "release_without_rename": (
        INDICES,
        ("        os.replace(folder, retired)\n        freed += _remove(retired)"),
        "        freed += _remove(folder)",
        INDICES_TESTS,
    ),
    "scalers_without_code": (
        READOUTS,
        "            code=_scaler_code(),\n",
        "",
        INDICES_TESTS,
    ),
    "scalers_without_check": (
        READOUTS,
        "                scalers.source_sha256 == train.identity\n",
        "                True\n",
        INDICES_TESTS,
    ),
    "release_other_views": (
        INDICES,
        "    folder = _view_folder(root, identity)\n",
        (
            "    folder = _view_folder(root, identity)\n"
            "    for other in root.iterdir():\n"
            "        if not other.name.startswith('.') and other.is_dir():\n"
            "            shutil.rmtree(other)\n"
        ),
        INDICES_TESTS,
    ),
    "release_before_regenerating": (
        ROLLING,
        (
            "        freed = 0\n"
            "        for job in jobs:\n"
            '            if job.get("phase") == PLATEAU:'
        ),
        (
            "        freed = self._release_indices(base, index) * 0\n"
            "        for job in jobs:\n"
            '            if job.get("phase") == PLATEAU:'
        ),
        RELEASE_TESTS,
    ),
}
results = {}
for name, (path, old, new, tests) in MUTATIONS.items():
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
    results[name] = dict(
        detected=run.returncode != 0, failed=failed, tail=run.stdout.splitlines()[-1:]
    )
    print(name, results[name])
OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
