"""Mutaciones dirigidas de los núcleos de los medoids y de las exportaciones del enlace (#547).

Uso: kernel_mutations.py ÁRBOL SALIDA.json, sobre una copia del árbol.

Cada mutación cambia un único fragmento y lo restaura después. Las de C++ compilan con g++ el
núcleo y su prueba (`native/tests/medoid_kernels_tests.cpp`) con las opciones de coma flotante
del proyecto y la ejecutan. Las de Python ejecutan con pytest, `-x` y sin orden aleatorio, las
pruebas nativas de los medoids y las del banco de retención, con el enlace compilado desde el
mismo árbol en MARS_TITAN_EPISODIC_NATIVE. Una mutación cuenta como detectada si la prueba falla.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
KERNEL = "native/src/medoid_kernels.cpp"
ANCHORED = "src/mars_titan/cm/anchored_medoids.py"
COSTS = "src/mars_titan/cm/medoids.py"
BANK = "src/mars_titan/memory/retention_bank.py"
CONSUMERS = "src/mars_titan/memory/financial_consumers.py"
PYTHON_TESTS = [
    "tests/cm/test_anchored_medoids.py",
    "tests/memory/test_retention_bank.py",
]
CPP = {
    "bound_without_relative_margin": (
        "        double value = square - limit * limit * relative;\n",
        "        double value = square;\n",
    ),
    "bound_without_absolute_margin": ("        value -= limit * absolute;\n", ""),
    "bound_without_shrink": ("        return value * shrink;\n", "        return value;\n"),
    "only_first_candidate": (
        "if (column != first && bound(square[column], root, roots[column]) <= best) {",
        "if (column != first && bound(square[column], root, roots[column]) < 0) {",
    ),
    "chain_reversed": (
        "distance = std::hypot(distance, left[dimension] - right[dimension]);",
        "distance = std::hypot(distance, left[dimensions - 1 - dimension] -"
        " right[dimensions - 1 - dimension]);",
    ),
    "fp32_check_removed": (
        "if (!std::ranges::all_of(clients, from_float) ||"
        " !std::ranges::all_of(fixed, from_float)) {",
        "if (false) {",
    ),
    "finite_check_removed": (
        "    if (!std::ranges::all_of(client_norms, finite) || "
        "!std::ranges::all_of(center_norms, finite) ||\n"
        "        !std::ranges::all_of(products, finite)) {",
        "    if (false) {",
    ),
    "shape_check_removed": (
        "    if (!shape_matches(rows, centers, products.size()) || client_norms.size() != rows ||",
        "    if (false ||",
    ),
    "sums_without_base": (
        "!sums[column].add(value < limit ? value : limit)",
        "!sums[column].add(value)",
    ),
    "sums_accept_nan": (
        "if (std::isnan(value) || !sums[column].add(",
        "if (!sums[column].add(",
    ),
    "sums_written_on_failure": (
        "limit ? value : limit)) {\n                    return false;\n",
        "limit ? value : limit)) {\n                    std::ranges::fill(totals, 0.0);\n"
        "                    return false;\n",
    ),
    "sums_shape_unchecked": (
        "    if (base.size() != distance.rows || totals.size() != distance.columns) {",
        "    if (false) {",
    ),
}
PYTHON = {
    "native_without_whole_row": (
        ANCHORED,
        "    if not bounded_background or rows * fixed_columns < len(fixed):\n",
        "    if not bounded_background:\n",
    ),
    "native_flag_lost": (ANCHORED, "        kernel is not None,\n", "        False,\n"),
    "sums_flag_lost": (ANCHORED, "        costs.sums is not None,\n", "        False,\n"),
    "sums_ignore_outgoing": (
        COSTS,
        "totals = None if self.sums is None else self.sums(distance, base)",
        "totals = None if self.sums is None else self.sums(distance, self.nearest[start:end])",
    ),
    "sums_for_integers": (
        COSTS,
        "        self.sums = None if integer else sums\n",
        "        self.sums = sums\n",
    ),
    "fork_without_copy": (BANK, "            copy(self._memory)\n", "            pass\n"),
    "exported_keys_reversed": (
        BANK,
        "                export(records)\n",
        "                export(records[::-1])\n",
    ),
    "retained_rows_shared": (
        BANK,
        "                row = points[bisect_left(ids, identifier)].copy()\n",
        "                row = points[bisect_left(ids, identifier)]\n",
    ),
    "retained_rows_shifted": (
        BANK,
        "                row = points[bisect_left(ids, identifier)].copy()\n",
        "                row = points[bisect_left(ids, identifier) - 1].copy()\n",
    ),
    "columns_reversed": (
        CONSUMERS,
        "        return {name: torch.from_numpy(array) for name, array in columns.items()}\n",
        "        return {name: torch.from_numpy(array[::-1].copy()) for name, array in"
        " columns.items()}\n",
    ),
    "columns_ignored": (
        BANK,
        '        export = getattr(self._memory, "retained_arrays", None)\n'
        "        return None if export is None else export()\n",
        '        export = getattr(self._memory, "retained_arrays", None)\n'
        "        return None if export is None else dict(export(), labels=export()['ids'] * 0.0)\n",
    ),
}


def mutate(path, old, new):
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise SystemExit(f"{path}: el fragmento no aparece una sola vez: {old!r}")
    path.write_text(source.replace(old, new), encoding="utf-8")
    return source


def native_test(build):
    binary = build / "medoid_kernels_tests"
    compiled = subprocess.run(
        [
            "g++",
            "-std=c++20",
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            f"-I{ROOT / 'native/include'}",
            str(ROOT / KERNEL),
            str(ROOT / "native/tests/medoid_kernels_tests.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    if compiled.returncode:
        return dict(compiled=False, detected=None, tail=compiled.stderr.splitlines()[-3:])
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=600)
    return dict(compiled=True, detected=run.returncode != 0, tail=(run.stdout + run.stderr)[-300:])


def python_test():
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:randomly", "-p", "no:cacheprovider"]
        + PYTHON_TESTS,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    failed = [line.split()[1] for line in run.stdout.splitlines() if line.startswith("FAILED")]
    return dict(detected=run.returncode != 0, failed=failed, tail=run.stdout.splitlines()[-1:])


results = {}
with tempfile.TemporaryDirectory(dir=ROOT) as build:
    baseline = native_test(Path(build))
    if not baseline["compiled"] or baseline["detected"]:
        raise SystemExit(f"La prueba nativa no pasa sin mutar: {baseline}")
    for name, (old, new) in CPP.items():
        source = mutate(ROOT / KERNEL, old, new)
        try:
            results[name] = dict(kind="c++", **native_test(Path(build)))
        finally:
            (ROOT / KERNEL).write_text(source, encoding="utf-8")
        print(name, results[name], flush=True)
for name, (path, old, new) in PYTHON.items():
    source = mutate(ROOT / path, old, new)
    try:
        results[name] = dict(kind="python", **python_test())
    finally:
        (ROOT / path).write_text(source, encoding="utf-8")
    print(name, results[name], flush=True)
OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("detectadas", sum(bool(r["detected"]) for r in results.values()), "de", len(results))
