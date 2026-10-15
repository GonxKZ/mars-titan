"""Mutaciones dirigidas del núcleo de la proyección del codec (#628).

Uso: mutations.py ÁRBOL SALIDA.json, sobre una copia del árbol.

Cada mutación cambia un único fragmento de `native/src/episodic_projection.cpp` (el orden de
sumas compartido, el recorrido por paneles, el directo, el reparto entre hilos o la regla de
hilos del enlace), compila el núcleo con su prueba (`native/tests/episodic_projection_tests.cpp`)
con las opciones de coma flotante del proyecto, la ejecuta y restaura el archivo.
`fp_contract_fast` no cambia el código: compila con `-ffp-contract=fast`, que permite fusionar
multiplicación y suma. Una mutación cuenta como detectada si la compilación o la prueba fallan.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
KERNEL = ROOT / "native/src/episodic_projection.cpp"
TEST = ROOT / "native/tests/episodic_projection_tests.cpp"
MUTATIONS = {
    # Orden de las sumas, compartido por los dos recorridos.
    "pairs_ascending": (
        "        for (std::size_t offset = chunk; offset > 0; offset -= pair) {\n"
        "            tile.step(index + offset - pair, true);\n",
        "        for (std::size_t offset = 0; offset < chunk; offset += pair) {\n"
        "            tile.step(index + offset, true);\n",
    ),
    "remainder_without_odd": (
        "        tile.step(index, index + 1 < width);\n",
        "        tile.step(index, false);\n",
    ),
    # Recorrido por paneles.
    "single_accumulator": (
        "                lane1[row] = block[row * width + index + 1] * odd + lane1[row];\n",
        "                lane0[row] = block[row * width + index + 1] * odd + lane0[row];\n",
    ),
    "final_without_odd": (
        "            const Lanes total = lane0[row] + lane1[row];\n",
        "            const Lanes total = lane0[row];\n",
    ),
    "writes_every_lane": (
        "            std::memcpy(out + row * outputs, values.data(), kept * sizeof(double));\n",
        "            std::memcpy(out + row * outputs, values.data(), sizeof(Lanes));\n",
    ),
    "panel_from_next_column": (
        "                    problem.transposed + index * problem.outputs + first,"
        " kept * sizeof(double));\n",
        "                    problem.transposed + index * problem.outputs + first + 1,\n"
        "                    (kept - 1) * sizeof(double));\n",
    ),
    # Recorrido directo.
    "direct_single_accumulator": (
        "                    lane1[row][vector] = second * odd[vector] + lane1[row][vector];\n",
        "                    lane0[row][vector] = second * odd[vector] + lane0[row][vector];\n",
    ),
    "direct_final_without_odd": (
        "                const Lanes total = lane0[row][vector] + lane1[row][vector];\n",
        "                const Lanes total = lane0[row][vector];\n",
    ),
    "direct_odd_from_even": (
        "                load(odd[vector], first + outputs, count);\n",
        "                load(odd[vector], first, count);\n",
    ),
    "direct_tail_skipped": (
        "    for (; column < end; column += projection_lanes) {\n",
        "    for (; column + projection_lanes <= end; column += projection_lanes) {\n",
    ),
    "direct_last_row_skipped": (
        "        project_direct_rows<1, single_vectors>(problem, row, begin, end);\n",
        "",
    ),
    # Tramos de grupos y regla de hilos del enlace.
    "direct_range_from_zero": (
        "    const std::size_t begin = first * projection_lanes;\n",
        "    const std::size_t begin = 0;\n",
    ),
    "direct_range_to_the_end": (
        "    const std::size_t end = std::min(last * projection_lanes, problem.outputs);\n",
        "    const std::size_t end = problem.outputs;\n",
    ),
    "inverted_range_accepted": (
        "    if (first > last || last > projection_groups(outputs)) {\n",
        "    if (last > projection_groups(outputs)) {\n",
    ),
    "missing_groups_accepted": (
        "    if (first > last || last > projection_groups(outputs)) {\n",
        "    if (first > last) {\n",
    ),
    "threads_below_the_work": (
        "width * outputs >= (projection_parallel_work + rows - 1) / rows;",
        "width * outputs > (projection_parallel_work + rows - 1) / rows;",
    ),
    "threads_without_rows": (
        "        rows > 0 && width * outputs >= (projection_parallel_work + rows - 1) / rows;\n",
        "        width * outputs >= (projection_parallel_work + rows) / (rows + 1);\n",
    ),
    "threads_may_be_zero": (
        "    return parallel ? std::max<std::size_t>(available, 1) : 1;\n",
        "    return parallel ? available : 1;\n",
    ),
    "default_is_narrowest": (
        "    static const ProjectionIsa widest = available_projection_isas().back();\n",
        "    static const ProjectionIsa widest = available_projection_isas().front();\n",
    ),
}


def run(build, contract="off"):
    binary = build / "episodic_projection_tests"
    compiled = subprocess.run(
        [
            "g++",
            "-std=c++20",
            "-O2",
            "-pthread",
            "-fno-fast-math",
            f"-ffp-contract={contract}",
            f"-I{ROOT / 'native/include'}",
            str(KERNEL),
            str(TEST),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    if compiled.returncode:
        return dict(compiled=False, detected=True, tail=compiled.stderr.splitlines()[-2:])
    try:
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return dict(compiled=True, detected=True, tail="tiempo agotado")
    output = (result.stdout + result.stderr).strip()
    return dict(compiled=True, detected=result.returncode != 0, tail=output[-200:])


source = KERNEL.read_text(encoding="utf-8")
results = {}
with tempfile.TemporaryDirectory(dir=ROOT) as build:
    baseline = run(Path(build))
    if baseline["detected"]:
        raise SystemExit(f"La prueba no pasa sin mutar: {baseline}")
    for name, (old, new) in MUTATIONS.items():
        if source.count(old) != 1:
            raise SystemExit(f"{name}: el fragmento no aparece una sola vez")
        KERNEL.write_text(source.replace(old, new), encoding="utf-8")
        try:
            results[name] = run(Path(build))
        finally:
            KERNEL.write_text(source, encoding="utf-8")
        print(name, results[name], flush=True)
    results["fp_contract_fast"] = run(Path(build), contract="fast")
    print("fp_contract_fast", results["fp_contract_fast"], flush=True)
OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("detectadas", sum(bool(r["detected"]) for r in results.values()), "de", len(results))
